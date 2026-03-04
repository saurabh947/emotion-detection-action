"""Shared utilities for all training scripts.

Dataset formats supported
--------------------------
1. **Synthetic** (``--synthetic``) — always works, no files needed.
   Generates random tensors with plausible labels.  Use this to verify the
   training loop without downloading a real dataset.

2. **Directory** (``--data-dir PATH``) — expects::

       PATH/
       ├── angry/        ← one sub-folder per emotion label
       │   ├── clip1.mp4
       │   └── clip2.avi
       ├── happy/
       ├── sad/
       ├── neutral/
       ├── fearful/
       ├── disgusted/
       └── surprised/

3. **CSV** (``--csv PATH``) — expects a comma-separated file::

       video_path,emotion[,stress,engagement,arousal]

       /data/clip1.mp4,happy,0.1,0.9,0.7
       /data/clip2.mp4,sad
       ...

   The last three columns (stress, engagement, arousal) are optional.
   When absent, pseudo-labels derived from the emotion class are used.

Pseudo-labels for attention metrics
-------------------------------------
Most public datasets (RAVDESS, CREMA-D, AffectNet) do not supply explicit
stress / engagement / arousal annotations.  ``EMOTION_METRIC_DEFAULTS`` maps
each emotion to a plausible ``[stress, engagement, arousal]`` triplet so the
metrics head can still be supervised during Phase 1.  Replace these values
with real annotations if your dataset provides them.
"""

from __future__ import annotations

import csv
import os
import pathlib
import random
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.distributed import DistributedSampler

# ---------------------------------------------------------------------------
# Emotion / label constants
# ---------------------------------------------------------------------------

EMOTION_LABELS: list[str] = [
    "angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised",
    "unclear",  # 8th class — no person present, noisy signal, or low confidence
]
EMOTION_TO_IDX: dict[str, int] = {e: i for i, e in enumerate(EMOTION_LABELS)}

# Heuristic pseudo-labels: [stress, engagement, arousal]
# For "unclear" all metrics are 0 — there is no identifiable subject to measure.
EMOTION_METRIC_DEFAULTS: dict[str, list[float]] = {
    "angry":     [0.85, 0.60, 0.80],
    "disgusted": [0.50, 0.40, 0.45],
    "fearful":   [0.90, 0.65, 0.85],
    "happy":     [0.15, 0.90, 0.75],
    "neutral":   [0.25, 0.50, 0.30],
    "sad":       [0.50, 0.30, 0.20],
    "surprised": [0.40, 0.85, 0.75],
    "unclear":   [0.00, 0.00, 0.00],
}

VIDEO_EXTENSIONS: set[str] = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def _load_video_frames(
    path: str,
    num_frames: int = 16,
    image_size: int = 224,
) -> torch.Tensor:
    """Load a video file and return ``(num_frames, 3, H, W)`` float tensor.

    Falls back to a black tensor if OpenCV cannot read the file.
    """
    try:
        import cv2  # type: ignore[import]
    except ImportError:
        raise RuntimeError("opencv-python is required: pip3 install opencv-python")

    cap = cv2.VideoCapture(path)
    raw_frames: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        raw_frames.append(frame)
    cap.release()

    if not raw_frames:
        # Return black frames if the file is unreadable
        return torch.zeros(num_frames, 3, image_size, image_size)

    # Uniform temporal sampling
    indices = np.linspace(0, len(raw_frames) - 1, num_frames, dtype=int)
    sampled = [raw_frames[i] for i in indices]

    tensors: list[torch.Tensor] = []
    for bgr in sampled:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (image_size, image_size))
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0  # (3, H, W)
        tensors.append(t)

    return torch.stack(tensors)  # (T, 3, H, W)


def _load_audio_mel(
    path: str,
    sample_rate: int = 16000,
    n_mels: int = 128,
    n_fft: int = 400,
    hop_length: int = 160,
    target_length: int = 1024,
) -> torch.Tensor | None:
    """Load audio from a file and return a ``(time, mel)`` log mel-spectrogram.

    Returns ``None`` when the file has no audio track or torchaudio cannot
    read it — the model will substitute its learned absent-audio token.
    """
    try:
        import torchaudio  # type: ignore[import]
        import torchaudio.transforms as AT  # type: ignore[import]
    except ImportError:
        return None

    try:
        waveform, sr = torchaudio.load(path)  # (C, samples)
    except Exception:
        return None

    waveform = waveform.mean(0, keepdim=True)  # mono
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

    mel_transform = AT.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    mel = mel_transform(waveform)      # (1, mel, time)
    mel = torch.log1p(mel).squeeze(0)  # (mel, time)
    mel = mel.transpose(0, 1)          # (time, mel)

    # Pad or truncate to target_length
    t = mel.shape[0]
    if t < target_length:
        mel = torch.cat([mel, torch.zeros(target_length - t, n_mels)], dim=0)
    else:
        mel = mel[:target_length]

    return mel  # (target_length, mel)


# ---------------------------------------------------------------------------
# Dataset — synthetic
# ---------------------------------------------------------------------------


class SyntheticEmotionDataset(Dataset):
    """Randomly generated (video, audio, label) triples for fast loop testing.

    No files, no downloads — useful for verifying the training pipeline
    end-to-end on any machine in < 1 minute.

    Args:
        size: Number of synthetic samples.
        num_frames: Frames per video clip (must match backbone, default 16).
        image_size: Spatial resolution of each frame.
        mel_time: Time steps in the mel-spectrogram.
        n_mels: Mel filter-bank bins.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        size: int = 200,
        num_frames: int = 16,
        image_size: int = 224,
        mel_time: int = 1024,
        n_mels: int = 128,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        self._video = torch.from_numpy(
            rng.random((size, num_frames, 3, image_size, image_size), dtype=np.float32)
        )
        self._audio = torch.from_numpy(
            rng.random((size, mel_time, n_mels), dtype=np.float32)
        )
        labels = rng.integers(0, len(EMOTION_LABELS), size=size)
        self._emotion = torch.from_numpy(labels.astype(np.int64))
        metrics = np.array(
            [EMOTION_METRIC_DEFAULTS[EMOTION_LABELS[l]] for l in labels],
            dtype=np.float32,
        )
        self._metrics = torch.from_numpy(metrics)

    def __len__(self) -> int:
        return len(self._emotion)

    def __getitem__(self, idx: int) -> dict:
        return {
            "video": self._video[idx],
            "audio": self._audio[idx],
            "emotion": self._emotion[idx],
            "metrics": self._metrics[idx],
        }


# ---------------------------------------------------------------------------
# Dataset — real files
# ---------------------------------------------------------------------------


class EmotionVideoDataset(Dataset):
    """Load emotion data from a directory tree or CSV file.

    Each item is a dict with keys: ``video`` (T,C,H,W), ``audio`` (time,mel)
    or ``None``, ``emotion`` (long scalar), ``metrics`` (3-dim float tensor).

    Args:
        samples: List of ``(video_path, emotion_idx, [stress, engagement, arousal])``.
        num_frames: Frames to sample from each clip.
        image_size: Spatial resolution to resize frames to.
        mel_time: Mel-spectrogram time length.
        n_mels: Mel filter-bank bins.
        augment: When ``True``, apply simple temporal jitter and horizontal flip.
    """

    def __init__(
        self,
        samples: list[tuple[str, int, list[float]]],
        num_frames: int = 16,
        image_size: int = 224,
        mel_time: int = 1024,
        n_mels: int = 128,
        augment: bool = False,
    ) -> None:
        self._samples = samples
        self._num_frames = num_frames
        self._image_size = image_size
        self._mel_time = mel_time
        self._n_mels = n_mels
        self._augment = augment

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        path, emotion_idx, metrics_vals = self._samples[idx]

        video = _load_video_frames(path, self._num_frames, self._image_size)
        audio = _load_audio_mel(path, target_length=self._mel_time, n_mels=self._n_mels)

        if self._augment:
            # Horizontal flip with 50% probability
            if random.random() < 0.5:
                video = video.flip(-1)
            # Small brightness jitter
            video = (video + torch.randn_like(video) * 0.02).clamp(0, 1)

        return {
            "video": video,
            "audio": audio,  # may be None — handled in collate_fn
            "emotion": torch.tensor(emotion_idx, dtype=torch.long),
            "metrics": torch.tensor(metrics_vals, dtype=torch.float32),
        }


# ---------------------------------------------------------------------------
# Dataset factories
# ---------------------------------------------------------------------------


def samples_from_directory(data_dir: str) -> list[tuple[str, int, list[float]]]:
    """Scan ``data_dir/{emotion}/`` for video files and return sample list.

    Each entry is ``(video_path, emotion_idx, [stress, engagement, arousal])``.
    """
    root = pathlib.Path(data_dir)
    samples: list[tuple[str, int, list[float]]] = []

    for label in EMOTION_LABELS:
        subdir = root / label
        if not subdir.is_dir():
            continue
        idx = EMOTION_TO_IDX[label]
        default_metrics = EMOTION_METRIC_DEFAULTS[label]
        for fpath in subdir.iterdir():
            if fpath.suffix.lower() in VIDEO_EXTENSIONS:
                samples.append((str(fpath), idx, default_metrics))

    if not samples:
        raise ValueError(
            f"No video files found under {data_dir!r}.\n"
            "Expected sub-folders named: " + ", ".join(EMOTION_LABELS)
        )
    return samples


def samples_from_csv(csv_path: str) -> list[tuple[str, int, list[float]]]:
    """Read a CSV file and return sample list.

    Required columns: ``video_path``, ``emotion``
    Optional columns: ``stress``, ``engagement``, ``arousal``
    """
    samples: list[tuple[str, int, list[float]]] = []
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            emotion = row["emotion"].strip().lower()
            if emotion not in EMOTION_TO_IDX:
                continue
            idx = EMOTION_TO_IDX[emotion]
            defaults = EMOTION_METRIC_DEFAULTS[emotion]
            metrics = [
                float(row.get("stress", defaults[0]) or defaults[0]),
                float(row.get("engagement", defaults[1]) or defaults[1]),
                float(row.get("arousal", defaults[2]) or defaults[2]),
            ]
            samples.append((row["video_path"].strip(), idx, metrics))

    if not samples:
        raise ValueError(f"No valid rows found in {csv_path!r}")
    return samples


# ---------------------------------------------------------------------------
# DataLoader construction
# ---------------------------------------------------------------------------


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate that handles optional ``None`` audio tensors.

    When a sample has no audio (file had no audio track), the batch entry for
    that position is a zero-tensor of the same shape as the others so we can
    still form a padded batch.  A ``audio_mask`` boolean tensor marks which
    batch items have real audio.
    """
    videos = torch.stack([b["video"] for b in batch])
    emotions = torch.stack([b["emotion"] for b in batch])
    metrics = torch.stack([b["metrics"] for b in batch])

    # Determine audio shape from the first non-None sample
    audio_list = [b["audio"] for b in batch]
    ref_audio = next((a for a in audio_list if a is not None), None)
    if ref_audio is None:
        # No audio at all — pass None to the model
        return {"video": videos, "audio": None, "emotion": emotions, "metrics": metrics,
                "audio_mask": torch.zeros(len(batch), dtype=torch.bool)}

    shape = ref_audio.shape
    audio_mask = torch.tensor([a is not None for a in audio_list])
    filled = [a if a is not None else torch.zeros(shape) for a in audio_list]
    audios = torch.stack(filled)
    return {
        "video": videos,
        "audio": audios,
        "emotion": emotions,
        "metrics": metrics,
        "audio_mask": audio_mask,
    }


def build_dataloaders(
    dataset: Dataset,
    val_split: float = 0.2,
    batch_size: int = 4,
    num_workers: int = 0,
    seed: int = 42,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[DataLoader, DataLoader, DistributedSampler | None]:
    """Split ``dataset`` into train/val and return two DataLoaders.

    Args:
        dataset: Full dataset (train + val combined).
        val_split: Fraction held out for validation.
        batch_size: Samples **per GPU** (effective batch = batch_size × world_size).
        num_workers: Parallel data loading workers per process.
        seed: Random seed for reproducible splits.
        distributed: If True, use DistributedSampler (for torchrun / DDP).
        rank: This process's rank (only used when distributed=True).
        world_size: Total number of processes (only used when distributed=True).

    Returns:
        ``(train_loader, val_loader, train_sampler)``
        ``train_sampler`` is ``None`` in non-distributed mode.
        Call ``train_sampler.set_epoch(epoch)`` each epoch to re-shuffle.
    """
    n_val = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)

    train_sampler: DistributedSampler | None = None
    if distributed:
        # Each rank sees a non-overlapping subset — DistributedSampler handles shuffling
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=seed
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),   # shuffle=False when sampler is set
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=distributed,            # pin_memory speeds up GPU transfers in DDP
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=distributed,
    )
    return train_loader, val_loader, train_sampler


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------


class MultiTaskLoss(nn.Module):
    """Weighted sum of emotion classification + attention metrics losses.

    Args:
        emotion_weight: Weight for the cross-entropy emotion loss.
        metrics_weight: Weight for the MSE attention metrics loss.
    """

    def __init__(self, emotion_weight: float = 1.0, metrics_weight: float = 0.5) -> None:
        super().__init__()
        self._ce = nn.CrossEntropyLoss()
        self._mse = nn.MSELoss()
        self._ew = emotion_weight
        self._mw = metrics_weight

    def forward(
        self,
        emotion_logits: torch.Tensor,
        metrics_pred: torch.Tensor,
        emotion_targets: torch.Tensor,
        metrics_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute combined loss.

        Returns:
            ``(total_loss, emotion_loss, metrics_loss)``
        """
        emotion_loss = self._ce(emotion_logits, emotion_targets)
        metrics_loss = self._mse(metrics_pred, metrics_targets)
        total = self._ew * emotion_loss + self._mw * metrics_loss
        return total, emotion_loss, metrics_loss


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_acc: float,
    config_dict: dict,
    extra: dict | None = None,
) -> None:
    """Save model + optimiser state to a ``.pt`` checkpoint file.

    Handles DDP-wrapped models automatically — always saves the underlying
    module's state dict so checkpoints are device-agnostic.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Unwrap DDP if needed so checkpoints are always plain NeuralFusionModel states
    raw_model = model.module if hasattr(model, "module") else model
    payload: dict = {
        "epoch": epoch,
        "model_state": raw_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_acc": best_val_acc,
        "config": config_dict,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: str = "cpu",
) -> dict:
    """Load a checkpoint and restore model (+ optionally optimizer) state.

    Returns the checkpoint dict so callers can read ``epoch``, ``best_val_acc``,
    ``config``, etc.
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt


# ---------------------------------------------------------------------------
# Training-loop utilities
# ---------------------------------------------------------------------------


class AverageMeter:
    """Computes and stores a running mean."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-1 accuracy as a Python float in [0, 1]."""
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def print_epoch(
    epoch: int,
    total_epochs: int,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
    lr: float,
) -> None:
    """Print a single-line training summary."""
    bar_w = 20
    progress = int(bar_w * epoch / total_epochs)
    bar = "█" * progress + "░" * (bar_w - progress)
    print(
        f"  [{bar}] epoch {epoch:3d}/{total_epochs}"
        f"  train loss={train_loss:.4f} acc={train_acc:.1%}"
        f"  val loss={val_loss:.4f} acc={val_acc:.1%}"
        f"  lr={lr:.1e}"
    )


def compute_epoch_metrics(
    model: nn.Module,
    loader: DataLoader,
    criterion: MultiTaskLoss,
    device: str,
) -> tuple[float, float]:
    """Evaluate model on ``loader`` without gradient updates.

    Returns:
        ``(avg_loss, accuracy)``
    """
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    with torch.no_grad():
        for batch in loader:
            video = batch["video"].to(device)
            audio = batch["audio"]
            if audio is not None:
                audio = audio.to(device)
            emotion_t = batch["emotion"].to(device)
            metrics_t = batch["metrics"].to(device)

            out = model(video, audio, use_temporal=False)

            loss, _, _ = criterion(out.emotion_logits, out.metrics, emotion_t, metrics_t)
            acc = accuracy(out.emotion_logits, emotion_t)

            n = video.shape[0]
            loss_meter.update(loss.item(), n)
            acc_meter.update(acc, n)

    model.train()
    return loss_meter.avg, acc_meter.avg
