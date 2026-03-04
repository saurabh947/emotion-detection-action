#!/usr/bin/env python3
"""Phase 1 training — freeze backbones, train heads + fusion only.

What Phase 1 trains
-------------------
The VideoMAE and AST backbone weights are **frozen** (no gradient updates).
Only the following layers learn new weights:

* Video projection  (backbone hidden → 512)
* Audio projection  (backbone hidden → 512)
* Absent-modality tokens  (learned fallback embeddings)
* Cross-attention blocks  (video ↔ audio interaction)
* Temporal GRU buffer
* Emotion head  (8-class softmax, including "unclear")
* Metrics head  (stress / engagement / arousal Sigmoid)

Why this order
--------------
The pretrained backbones already encode rich visual and audio features.
Training only the "glue" layers first is faster (fewer parameters), avoids
catastrophic forgetting, and gives the fusion layers a clean starting point
before the backbone is adapted in Phase 2.

Runtime estimate (single H100)
-------------------------------
* Synthetic data         : ~1-2 min / epoch
* Real data (16 frames)  : ~3-6 min / epoch

Multi-GPU (8× H100 with torchrun)
----------------------------------
* Real data  : ~1 min / epoch  (8× speedup, effective batch = batch_size × 8)

Usage
-----
::

    # Single GPU (or CPU):
    python training/train_phase1.py --data-dir data/combined --pretrained --device cuda

    # Multi-GPU with torchrun (e.g. 8 H100s on SF Compute):
    torchrun --nproc_per_node=8 training/train_phase1.py \\
        --data-dir data/combined --pretrained --batch-size 16 --num-workers 4

    # Quick smoke-test (no data files needed):
    python training/train_phase1.py --synthetic --epochs 3

    # Resume an interrupted run:
    python training/train_phase1.py --data-dir data/combined \\
        --pretrained --resume outputs/phase1_last.pt

Output
------
* ``outputs/phase1_best.pt``  — checkpoint with the best validation accuracy
* ``outputs/phase1_last.pt``  — checkpoint from the final epoch (for resuming)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import warnings
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.distributed as dist
import torch.nn as nn

from emotion_detection_action.models.backbones import BackboneConfig
from emotion_detection_action.models.fusion import NeuralFusionModel

from common import (
    AverageMeter,
    EmotionVideoDataset,
    MultiTaskLoss,
    SyntheticEmotionDataset,
    accuracy,
    build_dataloaders,
    compute_epoch_metrics,
    load_checkpoint,
    print_epoch,
    samples_from_csv,
    samples_from_directory,
    save_checkpoint,
)

# ---------------------------------------------------------------------------
# Warning suppression
# ---------------------------------------------------------------------------


def suppress_warnings(is_main: bool = True) -> None:
    """Filter noisy warnings, keeping useful HuggingFace output on rank 0 only.

    Rank 0 (is_main=True):
      - HuggingFace weight-loading messages (WARNING level) → visible
      - Model download progress bars                        → visible
      - Python deprecation / UserWarnings                   → suppressed
      - datasets / filelock chatter                         → suppressed

    Ranks 1-7 (is_main=False):
      - Everything suppressed to avoid 8× duplicate output
    """
    # Always suppress noisy Python-level warnings on every rank
    warnings.filterwarnings("ignore")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Suppress chatter that is never useful regardless of rank
    for name in ("datasets", "filelock"):
        logging.getLogger(name).setLevel(logging.ERROR)

    if not is_main:
        # Non-primary ranks: silence everything to avoid duplicate output
        for name in ("transformers", "huggingface_hub"):
            logging.getLogger(name).setLevel(logging.ERROR)
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    # Rank 0 keeps the transformers/huggingface_hub defaults so that
    # weight-loading info and download progress bars remain visible.


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------


def setup_ddp() -> tuple[bool, int, int]:
    """Detect and initialise DDP when launched with torchrun.

    torchrun sets LOCAL_RANK, RANK, and WORLD_SIZE automatically.
    When running with plain ``python``, LOCAL_RANK is not set and this
    function returns (False, 0, 1) so the script runs on a single device.

    Returns:
        (is_ddp, local_rank, world_size)
    """
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank == -1:
        return False, 0, 1
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return True, local_rank, dist.get_world_size()


def cleanup_ddp(is_ddp: bool) -> None:
    if is_ddp and dist.is_initialized():
        dist.destroy_process_group()


def reduce_scalar(value: float, device: str) -> float:
    """Average a scalar across all DDP ranks."""
    t = torch.tensor(value, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return t.item()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 1: train fusion heads with frozen backbones",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data source (exactly one required) ──────────────────────────────────
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--synthetic", action="store_true",
        help="Use randomly generated data (no files needed — for pipeline testing)",
    )
    group.add_argument(
        "--data-dir", metavar="DIR",
        help="Root directory with {emotion}/ sub-folders containing video files",
    )
    group.add_argument(
        "--csv", metavar="FILE",
        help="CSV with columns: video_path, emotion[, stress, engagement, arousal]",
    )

    # ── Model ────────────────────────────────────────────────────────────────
    p.add_argument("--pretrained", action="store_true",
                   help="Load HuggingFace pretrained backbone weights (~1.8 GB download)")
    p.add_argument("--video-model", choices=["videomae", "vivit"], default="videomae",
                   help="Video backbone (videomae=16 frames recommended)")
    p.add_argument("--d-model", type=int, default=512,
                   help="Shared embedding dimension (must match any loaded checkpoint)")
    p.add_argument("--cross-attn-layers", type=int, default=2)
    p.add_argument("--gru-layers", type=int, default=2)

    # ── Training hyper-params ────────────────────────────────────────────────
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16,
                   help="Batch size **per GPU**. Effective batch = batch_size × num_gpus")
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Base learning rate. Auto-scaled by world_size when --scale-lr is set")
    p.add_argument("--scale-lr", action="store_true", default=True,
                   help="Scale LR by world_size (linear scaling rule for multi-GPU)")
    p.add_argument("--no-scale-lr", dest="scale_lr", action="store_false",
                   help="Disable automatic LR scaling")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader workers per GPU (4 recommended on Linux NVMe)")
    p.add_argument("--augment", action="store_true",
                   help="Enable simple video augmentation (flip + brightness jitter)")
    p.add_argument("--emotion-loss-weight", type=float, default=1.0)
    p.add_argument("--metrics-loss-weight", type=float, default=0.5)

    # ── Scheduler ────────────────────────────────────────────────────────────
    p.add_argument("--scheduler", choices=["cosine", "step", "none"], default="cosine",
                   help="LR scheduler: cosine annealing, step decay, or constant")
    p.add_argument("--warmup-epochs", type=int, default=2,
                   help="Linear warm-up epochs before the main scheduler")

    # ── Checkpoints ─────────────────────────────────────────────────────────
    p.add_argument("--output-dir", default="outputs",
                   help="Directory to save checkpoints")
    p.add_argument("--resume", metavar="PT",
                   help="Resume training from a previous Phase 1 checkpoint")
    p.add_argument("--save-every", type=int, default=5,
                   help="Save a 'last' checkpoint every N epochs")

    # ── Device (single-GPU mode only; ignored when torchrun sets LOCAL_RANK) ─
    p.add_argument("--device", default="cuda",
                   help="Torch device for single-GPU mode: cuda | cpu | mps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quiet", action="store_true", default=True,
                   help="Suppress HuggingFace and Python deprecation warnings (default: on)")
    p.add_argument("--no-quiet", dest="quiet", action="store_false",
                   help="Show all warnings (useful for debugging)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Scheduler helper
# ---------------------------------------------------------------------------


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    epochs: int,
    warmup_epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if scheduler_type == "none":
        return None

    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_epochs
        )
        if scheduler_type == "cosine":
            main = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, epochs - warmup_epochs)
            )
        else:  # step
            main = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, main], milestones=[warmup_epochs],
        )

    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # ── DDP setup (must happen before suppress_warnings so we know the rank) ─
    is_ddp, local_rank, world_size = setup_ddp()
    is_main = (local_rank == 0)

    if args.quiet:
        suppress_warnings(is_main=is_main)

    torch.manual_seed(args.seed + local_rank)   # different seed per rank

    # ── Device ────────────────────────────────────────────────────────────
    if is_ddp:
        device = f"cuda:{local_rank}"
    else:
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            print("WARNING: CUDA not available — falling back to CPU.")
            device = "cpu"
        if device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            print("WARNING: MPS not available — falling back to CPU.")
            device = "cpu"

    if is_main:
        if is_ddp:
            print(f"\n  DDP mode  : {world_size} GPUs")
            print(f"  Devices   : cuda:0 … cuda:{world_size - 1}")
        else:
            print(f"\n  Device    : {device}")

    # ── LR scaling ────────────────────────────────────────────────────────
    effective_lr = args.lr * world_size if (args.scale_lr and world_size > 1) else args.lr
    effective_batch = args.batch_size * world_size

    if is_main and world_size > 1:
        print(f"  Batch/GPU : {args.batch_size}  ×  {world_size} GPUs  =  {effective_batch} effective")
        print(f"  LR        : {args.lr:.1e} × {world_size} = {effective_lr:.1e}  (linear scaling)")

    # ── Model ─────────────────────────────────────────────────────────────
    num_frames = 32 if args.video_model == "vivit" else 16
    config = BackboneConfig(
        video_model=args.video_model,   # type: ignore[arg-type]
        pretrained=args.pretrained,
        d_model=args.d_model,
        video_num_frames=num_frames,
    )

    if is_main:
        print(f"\n  Building NeuralFusionModel (d_model={args.d_model}) …")

    model = NeuralFusionModel(
        config,
        num_cross_attn_layers=args.cross_attn_layers,
        gru_layers=args.gru_layers,
    ).to(device)

    # Phase 1: freeze both backbones before wrapping in DDP
    model.freeze_backbones()

    if is_main:
        trainable = model.count_parameters(trainable_only=True)
        total     = model.count_parameters(trainable_only=False)
        print(f"  Trainable : {trainable:,} / {total:,} ({trainable/total:.1%})  — backbones FROZEN")

    # ── Wrap in DDP ───────────────────────────────────────────────────────
    if is_ddp:
        # find_unused_parameters=True is required because:
        # - absent_video_token / absent_audio_token are only used when a
        #   modality is missing; batches with both modalities never trigger them.
        # - GRU parameters are skipped (use_temporal=False during training).
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True,
        )

    # Convenience reference to the underlying NeuralFusionModel (for checkpointing)
    raw_model: NeuralFusionModel = model.module if is_ddp else model   # type: ignore[assignment]

    # ── Dataset / DataLoaders ─────────────────────────────────────────────
    if is_main:
        print("\n  Loading dataset …")

    if args.synthetic:
        dataset = SyntheticEmotionDataset(size=200, num_frames=num_frames)
        if is_main:
            print("  Synthetic dataset : 200 random samples")
    elif args.data_dir:
        samples = samples_from_directory(args.data_dir)
        dataset = EmotionVideoDataset(samples, num_frames=num_frames, augment=args.augment)
        if is_main:
            print(f"  Directory dataset : {len(samples)} clips from {args.data_dir!r}")
    else:
        samples = samples_from_csv(args.csv)
        dataset = EmotionVideoDataset(samples, num_frames=num_frames, augment=args.augment)
        if is_main:
            print(f"  CSV dataset : {len(samples)} clips from {args.csv!r}")

    train_loader, val_loader, train_sampler = build_dataloaders(
        dataset,
        val_split=args.val_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        distributed=is_ddp,
        rank=local_rank,
        world_size=world_size,
    )

    if is_main:
        print(f"  Train batches/GPU : {len(train_loader)}   Val batches : {len(val_loader)}")

    # ── Optimiser & loss ──────────────────────────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=effective_lr, weight_decay=args.weight_decay)
    criterion = MultiTaskLoss(
        emotion_weight=args.emotion_loss_weight,
        metrics_weight=args.metrics_loss_weight,
    )
    scheduler = build_scheduler(optimizer, args.scheduler, args.epochs, args.warmup_epochs)

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 1
    best_val_acc = 0.0
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    best_path = str(Path(args.output_dir) / "phase1_best.pt")
    last_path = str(Path(args.output_dir) / "phase1_last.pt")

    if args.resume:
        if is_main:
            print(f"\n  Resuming from {args.resume!r} …")
        ckpt = load_checkpoint(args.resume, raw_model, optimizer, device)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        if is_main:
            print(f"  Restored epoch {start_epoch - 1}, best val acc = {best_val_acc:.1%}")

    # ── Training loop ─────────────────────────────────────────────────────
    if is_main:
        print(f"\n{'='*65}")
        print(f"  Phase 1 Training — {args.epochs} epochs")
        print(f"  LR={effective_lr:.1e}  batch/GPU={args.batch_size}"
              f"  effective_batch={effective_batch}  scheduler={args.scheduler}")
        print(f"{'='*65}")

    for epoch in range(start_epoch, args.epochs + 1):
        # Important: set epoch on sampler so each epoch gets a different shuffle
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_t0 = time.perf_counter()
        loss_m = AverageMeter()
        acc_m  = AverageMeter()

        for batch in train_loader:
            video = batch["video"].to(device, non_blocking=True)
            audio = batch["audio"]
            if audio is not None:
                audio = audio.to(device, non_blocking=True)
            emotion_t = batch["emotion"].to(device, non_blocking=True)
            metrics_t = batch["metrics"].to(device, non_blocking=True)

            optimizer.zero_grad()
            out = model(video, audio, use_temporal=False)
            loss, _, _ = criterion(out.emotion_logits, out.metrics, emotion_t, metrics_t)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            n = video.shape[0]
            loss_m.update(loss.item(), n)
            acc_m.update(accuracy(out.emotion_logits, emotion_t), n)

        if scheduler is not None:
            scheduler.step()

        # All-reduce train metrics so rank-0 logs the true average across all GPUs
        if is_ddp:
            train_loss = reduce_scalar(loss_m.avg, device)
            train_acc  = reduce_scalar(acc_m.avg,  device)
        else:
            train_loss = loss_m.avg
            train_acc  = acc_m.avg

        # ── Validation and logging — rank 0 only ──────────────────────────
        if is_main:
            val_loss, val_acc = compute_epoch_metrics(raw_model, val_loader, criterion, device)

            current_lr = optimizer.param_groups[0]["lr"]
            elapsed    = time.perf_counter() - epoch_t0
            print_epoch(epoch, args.epochs, train_loss, train_acc, val_loss, val_acc, current_lr)
            print(f"          ({elapsed:.0f}s/epoch)")

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(
                    best_path, model, optimizer, epoch, best_val_acc,
                    {"d_model": args.d_model, "video_model": args.video_model,
                     "cross_attn_layers": args.cross_attn_layers, "gru_layers": args.gru_layers},
                    {"phase": 1},
                )
                print(f"          ✓ Best checkpoint → {best_path}  (acc={val_acc:.1%})")

            if epoch % args.save_every == 0 or epoch == args.epochs:
                save_checkpoint(
                    last_path, model, optimizer, epoch, best_val_acc,
                    {"d_model": args.d_model, "video_model": args.video_model,
                     "cross_attn_layers": args.cross_attn_layers, "gru_layers": args.gru_layers},
                    {"phase": 1},
                )

        # Synchronise all ranks before the next epoch
        if is_ddp:
            dist.barrier()

    # ── Done ──────────────────────────────────────────────────────────────
    if is_main:
        print(f"\n{'='*65}")
        print(f"  Phase 1 complete.")
        print(f"  Best val accuracy : {best_val_acc:.1%}")
        print(f"  Checkpoint        : {best_path}")
        print(f"\n  Next step → Phase 2:")
        if is_ddp:
            print(f"  torchrun --nproc_per_node={world_size} training/train_phase2.py \\")
        else:
            print(f"  python training/train_phase2.py \\")
        print(f"      --checkpoint {best_path} --data-dir <DIR> --pretrained")
        print(f"{'='*65}")

    cleanup_ddp(is_ddp)


if __name__ == "__main__":
    main()
