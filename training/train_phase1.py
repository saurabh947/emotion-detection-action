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
* Emotion head  (7-class softmax)
* Metrics head  (stress / engagement / arousal Sigmoid)

Why this order
--------------
The pretrained backbones already encode rich visual and audio features.
Training only the "glue" layers first is faster (fewer parameters), avoids
catastrophic forgetting, and gives the fusion layers a clean starting point
before the backbone is adapted in Phase 2.

Runtime estimate
----------------
* CPU-only, synthetic data  : ~3-5 min / epoch  (fast enough to check the loop)
* CPU, real data (16 frames): ~30-90 min / epoch  (not recommended for production)
* GPU (A100 / 4090)         : ~5-15 min / epoch on RAVDESS (~1 400 clips)
* Apple MPS                 : ~20-40 min / epoch

Usage
-----
::

    # Quick smoke-test with generated data (no files, no download):
    python training/train_phase1.py --synthetic --epochs 3

    # Train on a real directory of videos (pretrained backbones downloaded once):
    python training/train_phase1.py --data-dir /data/RAVDESS --pretrained --epochs 20

    # Train from a CSV manifest:
    python training/train_phase1.py --csv /data/manifest.csv --pretrained --epochs 20

    # Resume an interrupted run:
    python training/train_phase1.py --data-dir /data/RAVDESS --resume outputs/phase1_last.pt

    # GPU training:
    python training/train_phase1.py --data-dir /data/RAVDESS --pretrained --device cuda --batch-size 8

Output
------
* ``outputs/phase1_best.pt``  — checkpoint with the best validation accuracy
* ``outputs/phase1_last.pt``  — checkpoint from the final epoch (for resuming)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
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
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Learning rate for heads and fusion layers")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader worker processes (0 = main thread, safer on macOS)")
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

    # ── Device ──────────────────────────────────────────────────────────────
    p.add_argument("--device", default="cpu",
                   help="Torch device: cpu | cuda | mps")
    p.add_argument("--seed", type=int, default=42)

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
            optimizer,
            schedulers=[warmup, main],
            milestones=[warmup_epochs],
        )

    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)

    # ── Device ────────────────────────────────────────────────────────────
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available — falling back to CPU.")
        device = "cpu"
    if device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        print("WARNING: MPS not available — falling back to CPU.")
        device = "cpu"
    print(f"  Device : {device}")

    # ── Model ─────────────────────────────────────────────────────────────
    num_frames = 32 if args.video_model == "vivit" else 16
    config = BackboneConfig(
        video_model=args.video_model,  # type: ignore[arg-type]
        pretrained=args.pretrained,
        d_model=args.d_model,
        video_num_frames=num_frames,
    )

    print(f"\n  Building NeuralFusionModel (d_model={args.d_model}) …")
    model = NeuralFusionModel(
        config,
        num_cross_attn_layers=args.cross_attn_layers,
        gru_layers=args.gru_layers,
    ).to(device)

    # ── Phase 1: freeze both backbones ────────────────────────────────────
    model.freeze_backbones()
    trainable = model.count_parameters(trainable_only=True)
    total = model.count_parameters(trainable_only=False)
    print(f"  Trainable params : {trainable:,} / {total:,} ({trainable/total:.1%})")
    print(f"  Backbones        : FROZEN")

    # ── Dataset / DataLoaders ─────────────────────────────────────────────
    print("\n  Loading dataset …")
    if args.synthetic:
        dataset = SyntheticEmotionDataset(size=200, num_frames=num_frames)
        print(f"  Synthetic dataset : 200 random samples")
    elif args.data_dir:
        samples = samples_from_directory(args.data_dir)
        dataset = EmotionVideoDataset(samples, num_frames=num_frames, augment=args.augment)
        print(f"  Directory dataset : {len(samples)} clips from {args.data_dir!r}")
    else:
        samples = samples_from_csv(args.csv)
        dataset = EmotionVideoDataset(samples, num_frames=num_frames, augment=args.augment)
        print(f"  CSV dataset : {len(samples)} clips from {args.csv!r}")

    train_loader, val_loader = build_dataloaders(
        dataset,
        val_split=args.val_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    print(f"  Train batches : {len(train_loader)}   Val batches : {len(val_loader)}")

    # ── Optimiser & loss ──────────────────────────────────────────────────
    # Phase 1 only trains heads + fusion, so use a flat learning rate
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = MultiTaskLoss(
        emotion_weight=args.emotion_loss_weight,
        metrics_weight=args.metrics_loss_weight,
    )
    scheduler = build_scheduler(optimizer, args.scheduler, args.epochs, args.warmup_epochs)

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 1
    best_val_acc = 0.0
    best_path = str(Path(args.output_dir) / "phase1_best.pt")
    last_path = str(Path(args.output_dir) / "phase1_last.pt")

    if args.resume:
        print(f"\n  Resuming from {args.resume!r} …")
        ckpt = load_checkpoint(args.resume, model, optimizer, device)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"  Restored epoch {start_epoch - 1}, best val acc = {best_val_acc:.1%}")

    # ── Training loop ─────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Phase 1 Training — {args.epochs} epochs")
    print(f"  LR={args.lr}  batch={args.batch_size}  scheduler={args.scheduler}")
    print(f"{'='*65}")

    model.train()
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_t0 = time.perf_counter()
        loss_m = AverageMeter()
        acc_m = AverageMeter()

        for batch in train_loader:
            video = batch["video"].to(device)
            audio = batch["audio"]
            if audio is not None:
                audio = audio.to(device)
            emotion_t = batch["emotion"].to(device)
            metrics_t = batch["metrics"].to(device)

            optimizer.zero_grad()
            out = model(video, audio, use_temporal=False)
            loss, _, _ = criterion(
                out.emotion_logits, out.metrics, emotion_t, metrics_t
            )
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            n = video.shape[0]
            loss_m.update(loss.item(), n)
            acc_m.update(accuracy(out.emotion_logits, emotion_t), n)

        if scheduler is not None:
            scheduler.step()

        # ── Validation ────────────────────────────────────────────────
        val_loss, val_acc = compute_epoch_metrics(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.perf_counter() - epoch_t0
        print_epoch(epoch, args.epochs, loss_m.avg, acc_m.avg, val_loss, val_acc, current_lr)
        print(f"          ({elapsed:.0f}s/epoch)")

        # ── Save best ─────────────────────────────────────────────────
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                best_path, model, optimizer, epoch, best_val_acc,
                {"d_model": args.d_model, "video_model": args.video_model,
                 "cross_attn_layers": args.cross_attn_layers, "gru_layers": args.gru_layers},
                {"phase": 1},
            )
            print(f"          ✓ Best checkpoint saved → {best_path}  (acc={val_acc:.1%})")

        # ── Save last every N epochs ──────────────────────────────────
        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(
                last_path, model, optimizer, epoch, best_val_acc,
                {"d_model": args.d_model, "video_model": args.video_model,
                 "cross_attn_layers": args.cross_attn_layers, "gru_layers": args.gru_layers},
                {"phase": 1},
            )

    print(f"\n{'='*65}")
    print(f"  Phase 1 complete.")
    print(f"  Best val accuracy : {best_val_acc:.1%}")
    print(f"  Checkpoint        : {best_path}")
    print(f"\n  Next step → run Phase 2 to fine-tune the backbones:")
    print(f"  python training/train_phase2.py --checkpoint {best_path} "
          f"--pretrained [--data-dir DIR | --csv FILE]")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
