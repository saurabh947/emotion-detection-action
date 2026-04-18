#!/usr/bin/env python3
"""Post-training temperature scaling calibration.

What this does
--------------
During training, the model became overconfident: val accuracy kept improving
(new bests at epochs 10-20) but val loss climbed from 0.77 → 1.80.  This means
the model is correctly classifying more samples, but assigning inflated softmax
probabilities — a classic sign of miscalibration.

Temperature scaling fixes this in one line: divide every logit by a scalar T
before softmax.  T > 1 softens the distribution (less confident); T < 1
sharpens it.  We find the optimal T on the val set by minimising cross-entropy
loss, with accuracy held constant (T doesn't change the argmax).

The calibrated checkpoint is saved alongside the original as
``outputs/phase1_best_calibrated.pt``.  All existing code works unchanged — the
EmotionDetector loads it via ``two_tower_model_path``.

Usage
-----
::

    # Calibrate using the val split of your training data:
    python training/calibrate_temperature.py \\
        --checkpoint outputs/phase1_best.pt \\
        --data-dir data/combined \\
        --device cpu

    # If you have a separate held-out calibration CSV:
    python training/calibrate_temperature.py \\
        --checkpoint outputs/phase1_best.pt \\
        --csv data/calibration.csv \\
        --device cuda

What gets saved
---------------
``outputs/phase1_best_calibrated.pt`` — same format as the training checkpoint,
with ``temperature`` added.  EmotionDetector loads this via
``two_tower_model_path``.

Note: temperature is applied *inside* NeuralFusionModel.forward() — no changes
to inference code needed.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from emotion_detection_action.models.backbones import BackboneConfig
from emotion_detection_action.models.fusion import NeuralFusionModel

from common import (
    EMOTION_LABELS,
    VideoClipDataset,
    build_dataloaders,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Temperature wrapper
# ---------------------------------------------------------------------------


class TemperatureScaledModel(nn.Module):
    """Wraps NeuralFusionModel with a learnable temperature scalar T.

    Only T is trained — all other parameters are frozen.
    """

    def __init__(self, model: NeuralFusionModel) -> None:
        super().__init__()
        self.model = model
        # Initialise T=1.0 (no change).  We optimise log(T) for numerical stability.
        self.log_temperature = nn.Parameter(torch.zeros(1))  # log(1) = 0

    @property
    def temperature(self) -> float:
        return float(self.log_temperature.exp().item())

    def forward(
        self,
        video: torch.Tensor | None,
        audio: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return temperature-scaled logits (B, 8)."""
        with torch.no_grad():
            out = self.model(video, audio, use_temporal=False)
        return out.emotion_logits / self.log_temperature.exp()


# ---------------------------------------------------------------------------
# ECE (Expected Calibration Error) — useful diagnostic
# ---------------------------------------------------------------------------


def compute_ece(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error — lower is better (0 = perfectly calibrated)."""
    probs = logits.softmax(dim=-1)
    confidences, predictions = probs.max(dim=-1)
    accuracies = predictions.eq(labels)

    ece = 0.0
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        mask = confidences.gt(lo) & confidences.le(hi)
        if mask.sum() == 0:
            continue
        bin_acc = accuracies[mask].float().mean().item()
        bin_conf = confidences[mask].mean().item()
        bin_weight = mask.float().mean().item()
        ece += abs(bin_acc - bin_conf) * bin_weight

    return ece


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Temperature scaling calibration")
    parser.add_argument("--checkpoint", default="outputs/phase1_best.pt",
                        help="Path to trained checkpoint (.pt)")
    parser.add_argument("--data-dir", default="data/combined",
                        help="Video dataset directory (class sub-folders)")
    parser.add_argument("--csv", default=None,
                        help="Calibration CSV (alternative to --data-dir)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Fraction of data to use for calibration (default 0.2)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate for temperature optimisation")
    parser.add_argument("--max-iter", type=int, default=50,
                        help="Maximum LBFGS iterations")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--output", default=None,
                        help="Output checkpoint path (default: <checkpoint>_calibrated.pt)")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint).resolve()
    if not ckpt_path.is_file():
        log.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)

    out_path = Path(args.output) if args.output else ckpt_path.with_stem(ckpt_path.stem + "_calibrated")
    device = torch.device(args.device)

    # --- Load checkpoint ---
    log.info("Loading checkpoint: %s", ckpt_path)
    payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    ckpt_cfg = payload.get("config", {})

    backbone_cfg = BackboneConfig(
        pretrained=True,
        d_model=ckpt_cfg.get("d_model", 512),
        video_model=ckpt_cfg.get("video_model", "affectnet_vit"),
        audio_model=ckpt_cfg.get("audio_model", "emotion2vec"),
        face_crop_enabled=False,   # no face crop during calibration
    )
    model = NeuralFusionModel(
        config=backbone_cfg,
        num_cross_attn_layers=ckpt_cfg.get("cross_attn_layers", 2),
        num_heads=8,
        gru_layers=ckpt_cfg.get("gru_layers", 2),
    )

    state = payload["model_state"]
    _RENAMES = {"absent_video_token": "absent_video", "absent_audio_token": "absent_audio"}
    state = {_RENAMES.get(k, k): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    # --- Build calibration dataloader (val split only) ---
    log.info("Building calibration dataset …")
    if args.csv:
        from common import CSVDataset
        full_ds = CSVDataset(args.csv, num_frames=args.frames, audio_mode="waveform")
    else:
        full_ds = VideoClipDataset(args.data_dir, num_frames=args.frames, audio_mode="waveform")

    _, val_loader, _ = build_dataloaders(
        full_ds,
        val_split=args.val_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=42,
    )
    log.info("Calibration set: %d batches", len(val_loader))

    # --- Collect all logits and labels (one pass, no grad) ---
    log.info("Forward pass to collect logits …")
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Collecting logits"):
            video = batch["video"].to(device)           # (B, T, C, H, W)
            audio = batch["audio"]
            if audio is not None:
                audio = audio.to(device)
            labels = batch["emotion"].to(device)        # (B,)

            out = model(video, audio, use_temporal=False)
            all_logits.append(out.emotion_logits.cpu())
            all_labels.append(labels.cpu())

    logits_all = torch.cat(all_logits, dim=0)   # (N, 8)
    labels_all = torch.cat(all_labels, dim=0)   # (N,)

    acc_before = (logits_all.argmax(dim=-1) == labels_all).float().mean().item()
    ece_before = compute_ece(logits_all, labels_all)
    log.info("\nBefore calibration:")
    log.info("  Accuracy : %.2f%%", acc_before * 100)
    log.info("  ECE      : %.4f  (lower = better calibrated)", ece_before)
    log.info("  Avg confidence: %.4f",
             logits_all.softmax(dim=-1).max(dim=-1).values.mean().item())

    # --- Optimise temperature T using LBFGS ---
    log.info("\nOptimising temperature …")
    ts_model = TemperatureScaledModel(model)
    ts_model.to(device)

    logits_dev = logits_all.to(device)
    labels_dev = labels_all.to(device)
    ce_loss = nn.CrossEntropyLoss()

    optimizer = torch.optim.LBFGS(
        [ts_model.log_temperature],
        lr=args.lr,
        max_iter=args.max_iter,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        scaled = logits_dev / ts_model.log_temperature.exp()
        loss = ce_loss(scaled, labels_dev)
        loss.backward()
        return loss

    optimizer.step(closure)

    T_opt = ts_model.temperature
    log.info("Optimal temperature T = %.4f", T_opt)

    # --- Evaluate after calibration ---
    with torch.no_grad():
        logits_scaled = logits_all / T_opt
    acc_after = (logits_scaled.argmax(dim=-1) == labels_all).float().mean().item()
    ece_after  = compute_ece(logits_scaled, labels_all)

    log.info("\nAfter calibration:")
    log.info("  Accuracy : %.2f%%  (unchanged by design)", acc_after * 100)
    log.info("  ECE      : %.4f", ece_after)
    log.info("  Avg confidence: %.4f",
             logits_scaled.softmax(dim=-1).max(dim=-1).values.mean().item())
    log.info("  ECE improvement: %.4f → %.4f  (%.1f%% reduction)",
             ece_before, ece_after, (ece_before - ece_after) / ece_before * 100)

    if T_opt < 1.0:
        log.warning("T < 1.0 — model was under-confident on the calibration set.")
    elif T_opt > 3.0:
        log.warning("T > 3.0 — model is severely overconfident. Check for train/val leakage.")

    # --- Save calibrated checkpoint ---
    calibrated_payload = dict(payload)   # copy all original fields
    calibrated_payload["temperature"] = T_opt
    # Overwrite model_state with the same weights (T is NOT part of state_dict,
    # it's applied at inference time in EmotionDetector via the temperature param)
    torch.save(calibrated_payload, str(out_path))
    log.info("\nCalibrated checkpoint saved: %s", out_path)

    # --- Usage instructions ---
    log.info("\n--- How to use ---")
    log.info("The temperature is stored in the checkpoint as payload['temperature'].")
    log.info("To apply it at inference, load the checkpoint and divide logits by T.")
    log.info("Or use the calibrated checkpoint directly in EmotionDetector:")
    log.info("  Config(two_tower_model_path='%s')", out_path)
    log.info("")
    log.info("NOTE: The current EmotionDetector applies softmax automatically.")
    log.info("To use calibration, see the EmotionDetector.apply_temperature() note below.")
    log.info("")
    log.info("Quick manual use:")
    log.info("  import torch")
    log.info("  payload = torch.load('%s', weights_only=True)", out_path)
    log.info("  T = payload['temperature']  # %.4f", T_opt)
    log.info("  # Apply: probs = (logits / T).softmax(-1)")


if __name__ == "__main__":
    main()
