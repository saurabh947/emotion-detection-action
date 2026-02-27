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

Runtime estimate
----------------
* GPU (A100 / 4090)  : ~15-30 min / epoch on RAVDESS
* Apple MPS          : ~60-90 min / epoch
* CPU                : not recommended (several hours / epoch)

Usage
-----
::

    # Typical Phase 2 run (requires a Phase 1 checkpoint):
    python training/train_phase2.py \\
        --checkpoint outputs/phase1_best.pt \\
        --data-dir /data/RAVDESS \\
        --pretrained --epochs 10

    # Fine-tune with synthetic data (smoke test):
    python training/train_phase2.py \\
        --checkpoint outputs/phase1_best.pt \\
        --synthetic --epochs 3

    # Resume an interrupted Phase 2 run:
    python training/train_phase2.py \\
        --checkpoint outputs/phase2_last.pt \\
        --data-dir /data/RAVDESS --pretrained --resume

    # Unfreeze more backbone layers for a larger dataset:
    python training/train_phase2.py \\
        --checkpoint outputs/phase1_best.pt \\
        --data-dir /data/RAVDESS --pretrained --unfreeze-layers 6

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
                   help="Phase 2 needs fewer epochs than Phase 1 — backbones only need nudging")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--backbone-lr", type=float, default=1e-5,
                   help="LR for unfrozen backbone layers (keep small to avoid forgetting)")
    p.add_argument("--head-lr", type=float, default=1e-4,
                   help="LR for fusion, cross-attn, GRU, and output heads")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--emotion-loss-weight", type=float, default=1.0)
    p.add_argument("--metrics-loss-weight", type=float, default=0.5)

    # ── Scheduler ────────────────────────────────────────────────────────────
    p.add_argument("--scheduler", choices=["cosine", "step", "none"], default="cosine")
    p.add_argument("--warmup-epochs", type=int, default=1)

    # ── Checkpoints ─────────────────────────────────────────────────────────
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--resume", action="store_true",
                   help="If set, treat --checkpoint as a Phase 2 checkpoint and resume training")
    p.add_argument("--save-every", type=int, default=2)

    # ── Device ──────────────────────────────────────────────────────────────
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)

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
        # Try to locate the encoder layers list.
        # VideoMAE: backbone._backbone.encoder.layer  (list of VideoMAELayer)
        # AST:      backbone._backbone.encoder.layer  (list of ASTLayer)
        # Stub:     no encoder attribute — skip gracefully
        try:
            encoder_layers = backbone._backbone.encoder.layer
        except AttributeError:
            # Stub backbones used in synthetic/no-pretrained mode
            print(f"    {backbone_name}: no encoder layers found (stub?), skipping partial unfreeze")
            backbone.unfreeze()
            continue

        n_total = len(encoder_layers)
        n_freeze = max(0, n_total - n_layers)
        print(f"    {backbone_name}: {n_total} layers → freezing first {n_freeze}, "
              f"unfreezing top {min(n_layers, n_total)}")

        for i, layer in enumerate(encoder_layers):
            requires_grad = i >= n_freeze
            for p in layer.parameters():
                if requires_grad and not p.requires_grad:
                    p.requires_grad_(True)
                    unfrozen += p.numel()
                elif not requires_grad:
                    p.requires_grad_(False)

        # Always unfreeze the backbone's final layer-norm / head (if present)
        for name in ("layernorm", "norm", "fc_norm"):
            submod = getattr(backbone._backbone, name, None)
            if submod is not None:
                for p in submod.parameters():
                    if not p.requires_grad:
                        p.requires_grad_(True)
                        unfrozen += p.numel()

    return unfrozen


# ---------------------------------------------------------------------------
# Scheduler helper (shared with Phase 1)
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
            optimizer,
            schedulers=[warmup, main_sched],
            milestones=[warmup_epochs],
        )

    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)


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

    print(f"\n  Building NeuralFusionModel …")
    model = NeuralFusionModel(
        config,
        num_cross_attn_layers=args.cross_attn_layers,
        gru_layers=args.gru_layers,
    ).to(device)

    # ── Load Phase 1 (or Phase 2 resume) checkpoint ───────────────────────
    print(f"  Loading checkpoint : {args.checkpoint!r}")
    if not Path(args.checkpoint).exists():
        print(f"ERROR: checkpoint not found: {args.checkpoint!r}")
        print("  Run Phase 1 first: python training/train_phase1.py --help")
        sys.exit(1)

    ckpt = torch.load(args.checkpoint, map_location=device)
    # Load weights (strict=False so Phase 1 → Phase 2 arch changes are tolerated)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing:
        print(f"  WARNING: missing keys in checkpoint: {missing[:5]} …")

    # ── Unfreeze top backbone layers ──────────────────────────────────────
    # Start with everything frozen, then selectively unfreeze
    model.freeze_backbones()
    print(f"\n  Unfreezing top {args.unfreeze_layers} backbone layers …")
    newly_unfrozen = unfreeze_top_layers(model, args.unfreeze_layers)
    trainable = model.count_parameters(trainable_only=True)
    total = model.count_parameters(trainable_only=False)
    print(f"  Newly unfrozen backbone params : {newly_unfrozen:,}")
    print(f"  Total trainable params         : {trainable:,} / {total:,} ({trainable/total:.1%})")

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

    # ── Optimiser with separate LR groups ─────────────────────────────────
    param_groups = model.get_trainable_parameter_groups(
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
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
            print("  WARNING: optimizer state incompatible (LR groups changed?) — starting fresh optimiser")
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"  Resuming Phase 2 from epoch {start_epoch - 1}")

    best_path = str(Path(args.output_dir) / "phase2_best.pt")
    last_path = str(Path(args.output_dir) / "phase2_last.pt")

    # ── Training loop ─────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Phase 2 Training — {args.epochs} epochs")
    print(f"  backbone_lr={args.backbone_lr}  head_lr={args.head_lr}  "
          f"batch={args.batch_size}  scheduler={args.scheduler}")
    print(f"{'='*65}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]

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

        val_loss, val_acc = compute_epoch_metrics(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.perf_counter() - epoch_t0
        print_epoch(epoch, args.epochs, loss_m.avg, acc_m.avg, val_loss, val_acc, current_lr)
        print(f"          ({elapsed:.0f}s/epoch)")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                best_path, model, optimizer, epoch, best_val_acc,
                {"d_model": args.d_model, "video_model": args.video_model,
                 "cross_attn_layers": args.cross_attn_layers, "gru_layers": args.gru_layers},
                {"phase": 2, "unfreeze_layers": args.unfreeze_layers},
            )
            print(f"          ✓ Best checkpoint saved → {best_path}  (acc={val_acc:.1%})")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(
                last_path, model, optimizer, epoch, best_val_acc,
                {"d_model": args.d_model, "video_model": args.video_model,
                 "cross_attn_layers": args.cross_attn_layers, "gru_layers": args.gru_layers},
                {"phase": 2, "unfreeze_layers": args.unfreeze_layers},
            )

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


if __name__ == "__main__":
    main()
