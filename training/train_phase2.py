#!/usr/bin/env python3
"""Phase 2 training — unfreeze top backbone layers and fine-tune end-to-end.

What Phase 2 trains
-------------------
Starting from a Phase 1 checkpoint (where only the fusion heads learned),
Phase 2 **unfreezes the top N encoder layers** of both the VideoMAE and AST
backbones and fine-tunes the whole network jointly.

Two learning-rate groups are used:

* **Backbone layers** : very small LR (``--backbone-lr``, default 1e-5) to
  prevent catastrophic forgetting of the ImageNet/AudioSet pretraining.
* **Heads + fusion**  : larger LR (``--head-lr``, default 1e-4) — same as
  Phase 1, keeps adapting to the emotion task.

Why run Phase 2 after Phase 1?
-------------------------------
If you fine-tune the backbones from the start (random heads + frozen→unfrozen
in one go), the noisy gradients from randomly-initialised heads damage the
carefully-pretrained backbone representations in the first few epochs.
Running Phase 1 first ensures the heads produce sensible gradients *before*
any backbone weights are touched, resulting in better final accuracy.

Runtime estimate (single H100)
-------------------------------
* Real data (16 frames)  : ~8-15 min / epoch

Multi-GPU (8× H100 with torchrun)
----------------------------------
* Real data  : ~1-2 min / epoch  (8× speedup)

Usage
-----
::

    # Single GPU:
    python training/train_phase2.py \\
        --checkpoint outputs/phase1_best.pt \\
        --data-dir data/combined --pretrained --device cuda

    # Multi-GPU with torchrun (e.g. 8 H100s on SF Compute):
    torchrun --nproc_per_node=8 training/train_phase2.py \\
        --checkpoint outputs/phase1_best.pt \\
        --data-dir data/combined --pretrained \\
        --batch-size 8 --num-workers 4

    # Smoke test with synthetic data:
    python training/train_phase2.py \\
        --checkpoint outputs/phase1_best.pt \\
        --synthetic --epochs 3

    # Resume an interrupted Phase 2 run:
    python training/train_phase2.py \\
        --checkpoint outputs/phase2_last.pt \\
        --data-dir data/combined --pretrained --resume

    # Unfreeze more backbone layers for a larger dataset:
    python training/train_phase2.py \\
        --checkpoint outputs/phase1_best.pt \\
        --data-dir data/combined --pretrained --unfreeze-layers 6

Output
------
* ``outputs/phase2_best.pt``  — best validation accuracy checkpoint
* ``outputs/phase2_last.pt``  — last epoch checkpoint (for resuming)

These checkpoints are directly usable by the SDK::

    from emotion_detection_action import Config, EmotionDetector
    cfg = Config(two_tower_pretrained=True,
                 two_tower_model_path="outputs/phase2_best.pt")
    detector = EmotionDetector(cfg)
    detector.initialize()
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
    warnings.filterwarnings("ignore")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    for name in ("datasets", "filelock"):
        logging.getLogger(name).setLevel(logging.ERROR)

    if not is_main:
        for name in ("transformers", "huggingface_hub"):
            logging.getLogger(name).setLevel(logging.ERROR)
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"


# ---------------------------------------------------------------------------
# DDP helpers  (identical pattern to train_phase1.py)
# ---------------------------------------------------------------------------


def setup_ddp() -> tuple[bool, int, int]:
    """Detect and initialise DDP when launched with torchrun.

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
        description="Phase 2: fine-tune backbones + heads end-to-end",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Checkpoint (required) ────────────────────────────────────────────
    p.add_argument(
        "--checkpoint", required=True, metavar="PT",
        help="Path to a Phase 1 (or Phase 2) checkpoint to start from",
    )

    # ── Data source (exactly one required) ──────────────────────────────────
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--synthetic", action="store_true",
                       help="Use randomly generated data (no files needed — for pipeline testing)")
    group.add_argument("--data-dir", metavar="DIR",
                       help="Root directory with {emotion}/ sub-folders containing video files")
    group.add_argument("--csv", metavar="FILE",
                       help="CSV with columns: video_path, emotion[, stress, engagement, arousal]")

    # ── Model ────────────────────────────────────────────────────────────────
    p.add_argument("--pretrained", action="store_true",
                   help="Load HuggingFace pretrained backbone weights (required for real data)")
    p.add_argument("--video-model", choices=["videomae", "vivit"], default="videomae")
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--cross-attn-layers", type=int, default=2)
    p.add_argument("--gru-layers", type=int, default=2)
    p.add_argument(
        "--unfreeze-layers", type=int, default=4, metavar="N",
        help="Number of top backbone encoder layers to unfreeze (2-6 recommended)",
    )

    # ── Training hyper-params ────────────────────────────────────────────────
    p.add_argument("--epochs", type=int, default=10,
                   help="Phase 2 needs fewer epochs — backbones only need nudging")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Batch size **per GPU**. Effective batch = batch_size × num_gpus")
    p.add_argument("--backbone-lr", type=float, default=1e-5,
                   help="LR for unfrozen backbone layers (small to avoid forgetting)")
    p.add_argument("--head-lr", type=float, default=1e-4,
                   help="LR for fusion, cross-attn, GRU, and output heads")
    p.add_argument("--scale-lr", action="store_true", default=True,
                   help="Scale LRs by world_size (linear scaling rule for multi-GPU)")
    p.add_argument("--no-scale-lr", dest="scale_lr", action="store_false",
                   help="Disable automatic LR scaling")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--emotion-loss-weight", type=float, default=1.0)
    p.add_argument("--metrics-loss-weight", type=float, default=0.5)

    # ── Scheduler ────────────────────────────────────────────────────────────
    p.add_argument("--scheduler", choices=["cosine", "step", "none"], default="cosine")
    p.add_argument("--warmup-epochs", type=int, default=1)

    # ── Checkpoints ─────────────────────────────────────────────────────────
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--resume", action="store_true",
                   help="Treat --checkpoint as a Phase 2 checkpoint and resume training")
    p.add_argument("--save-every", type=int, default=2)

    # ── Device (single-GPU mode only; ignored when torchrun sets LOCAL_RANK) ─
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quiet", action="store_true", default=True,
                   help="Suppress HuggingFace and Python deprecation warnings (default: on)")
    p.add_argument("--no-quiet", dest="quiet", action="store_false",
                   help="Show all warnings (useful for debugging)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Partial backbone unfreeze
# ---------------------------------------------------------------------------


def unfreeze_top_layers(model: NeuralFusionModel, n_layers: int) -> int:
    """Unfreeze the top ``n_layers`` transformer encoder blocks in each backbone.

    The Video and Audio backbones both have a stack of transformer encoder
    layers.  We unfreeze only the top N (closest to the output) to expose
    the most task-specific representations to the emotion loss, while keeping
    the early, more generic feature layers frozen.

    Returns:
        Number of newly unfrozen parameters.
    """
    unfrozen = 0

    for backbone_name, backbone in [
        ("video_backbone", model.video_backbone),
        ("audio_backbone", model.audio_backbone),
    ]:
        try:
            encoder_layers = backbone._backbone.encoder.layer
        except AttributeError:
            print(f"    {backbone_name}: no encoder layers found (stub?), skipping partial unfreeze")
            backbone.unfreeze()
            continue

        n_total = len(encoder_layers)
        n_freeze = max(0, n_total - n_layers)
        print(f"    {backbone_name}: {n_total} layers → freezing first {n_freeze}, "
              f"unfreezing top {min(n_layers, n_total)}")

        for i, layer in enumerate(encoder_layers):
            requires_grad = i >= n_freeze
            for param in layer.parameters():
                if requires_grad and not param.requires_grad:
                    param.requires_grad_(True)
                    unfrozen += param.numel()
                elif not requires_grad:
                    param.requires_grad_(False)

        for name in ("layernorm", "norm", "fc_norm"):
            submod = getattr(backbone._backbone, name, None)
            if submod is not None:
                for param in submod.parameters():
                    if not param.requires_grad:
                        param.requires_grad_(True)
                        unfrozen += param.numel()

    return unfrozen


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
            main_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, epochs - warmup_epochs)
            )
        else:
            main_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, main_sched], milestones=[warmup_epochs],
        )

    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)


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

    torch.manual_seed(args.seed + local_rank)

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
    scale = world_size if (args.scale_lr and world_size > 1) else 1
    effective_backbone_lr = args.backbone_lr * scale
    effective_head_lr     = args.head_lr     * scale
    effective_batch       = args.batch_size  * world_size

    if is_main and world_size > 1:
        print(f"  Batch/GPU : {args.batch_size}  ×  {world_size} GPUs  =  {effective_batch} effective")
        print(f"  Backbone LR : {args.backbone_lr:.1e} × {scale} = {effective_backbone_lr:.1e}")
        print(f"  Head LR     : {args.head_lr:.1e} × {scale} = {effective_head_lr:.1e}")

    # ── Model ─────────────────────────────────────────────────────────────
    num_frames = 32 if args.video_model == "vivit" else 16
    config = BackboneConfig(
        video_model=args.video_model,   # type: ignore[arg-type]
        pretrained=args.pretrained,
        d_model=args.d_model,
        video_num_frames=num_frames,
    )

    if is_main:
        print(f"\n  Building NeuralFusionModel …")

    model = NeuralFusionModel(
        config,
        num_cross_attn_layers=args.cross_attn_layers,
        gru_layers=args.gru_layers,
    ).to(device)

    # ── Load Phase 1 (or Phase 2 resume) checkpoint ───────────────────────
    if is_main:
        print(f"  Loading checkpoint : {args.checkpoint!r}")

    if not Path(args.checkpoint).exists():
        if is_main:
            print(f"ERROR: checkpoint not found: {args.checkpoint!r}")
            print("  Run Phase 1 first: python training/train_phase1.py --help")
        cleanup_ddp(is_ddp)
        sys.exit(1)

    ckpt = torch.load(args.checkpoint, map_location=device)
    missing, _ = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing and is_main:
        print(f"  WARNING: missing checkpoint keys: {missing[:5]} …")

    # ── Unfreeze top backbone layers (on raw model, before DDP wrap) ──────
    model.freeze_backbones()
    if is_main:
        print(f"\n  Unfreezing top {args.unfreeze_layers} backbone layers …")
    newly_unfrozen = unfreeze_top_layers(model, args.unfreeze_layers)

    if is_main:
        trainable = model.count_parameters(trainable_only=True)
        total     = model.count_parameters(trainable_only=False)
        print(f"  Newly unfrozen backbone params : {newly_unfrozen:,}")
        print(f"  Total trainable                : {trainable:,} / {total:,} ({trainable/total:.1%})")

    # ── Wrap in DDP ───────────────────────────────────────────────────────
    if is_ddp:
        # find_unused_parameters=True is required because absent_video_token /
        # absent_audio_token are skipped when both modalities are present, and
        # GRU parameters are skipped (use_temporal=False during training).
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True,
        )

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

    # ── Optimiser with separate LR groups ─────────────────────────────────
    param_groups = raw_model.get_trainable_parameter_groups(
        backbone_lr=effective_backbone_lr,
        head_lr=effective_head_lr,
    )
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    criterion = MultiTaskLoss(
        emotion_weight=args.emotion_loss_weight,
        metrics_weight=args.metrics_loss_weight,
    )
    scheduler = build_scheduler(optimizer, args.scheduler, args.epochs, args.warmup_epochs)

    # ── Resume bookkeeping ────────────────────────────────────────────────
    start_epoch = 1
    best_val_acc = ckpt.get("best_val_acc", 0.0)
    if args.resume and "optimizer_state" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        except ValueError:
            if is_main:
                print("  WARNING: optimizer state incompatible — starting fresh")
        start_epoch = ckpt.get("epoch", 0) + 1
        if is_main:
            print(f"  Resuming Phase 2 from epoch {start_epoch - 1}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    best_path = str(Path(args.output_dir) / "phase2_best.pt")
    last_path = str(Path(args.output_dir) / "phase2_last.pt")

    # ── Training loop ─────────────────────────────────────────────────────
    if is_main:
        print(f"\n{'='*65}")
        print(f"  Phase 2 Training — {args.epochs} epochs")
        print(f"  backbone_lr={effective_backbone_lr:.1e}  head_lr={effective_head_lr:.1e}"
              f"  batch/GPU={args.batch_size}  effective_batch={effective_batch}")
        print(f"{'='*65}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    for epoch in range(start_epoch, args.epochs + 1):
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
                    {"phase": 2, "unfreeze_layers": args.unfreeze_layers},
                )
                print(f"          ✓ Best checkpoint → {best_path}  (acc={val_acc:.1%})")

            if epoch % args.save_every == 0 or epoch == args.epochs:
                save_checkpoint(
                    last_path, model, optimizer, epoch, best_val_acc,
                    {"d_model": args.d_model, "video_model": args.video_model,
                     "cross_attn_layers": args.cross_attn_layers, "gru_layers": args.gru_layers},
                    {"phase": 2, "unfreeze_layers": args.unfreeze_layers},
                )

        if is_ddp:
            dist.barrier()

    # ── Done ──────────────────────────────────────────────────────────────
    if is_main:
        print(f"\n{'='*65}")
        print(f"  Phase 2 complete.")
        print(f"  Best val accuracy : {best_val_acc:.1%}")
        print(f"  Checkpoint        : {best_path}")
        print(f"\n  Use with the SDK:")
        print(f"    from emotion_detection_action import Config, EmotionDetector")
        print(f"    cfg = Config(two_tower_pretrained=True,")
        print(f"                 two_tower_model_path={best_path!r})")
        print(f"    detector = EmotionDetector(cfg)")
        print(f"    detector.initialize()")
        print(f"{'='*65}")

    cleanup_ddp(is_ddp)


if __name__ == "__main__":
    main()
