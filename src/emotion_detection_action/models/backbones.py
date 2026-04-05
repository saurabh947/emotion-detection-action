"""Video and audio backbone wrappers for the Neural Emotion Transformer.

Model selection guide
---------------------

Video backbone — AffectNet ViT (recommended for highest accuracy)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``trpakov/vit-face-expression`` is a ViT-B/16 fine-tuned directly on
AffectNet (450 K facial images, 8-class emotion).  Unlike VideoMAE or ViViT —
which were pretrained for scene/action understanding and must *learn* emotion
from scratch during fine-tuning — AffectNet ViT already speaks the emotion
feature language.  The fusion, projection, and head layers only need to learn
cross-modal alignment, not emotion itself.

+---------------------------------+------------------------------------+--------------------------------------+
| Property                        | AffectNet ViT (recommended ✓)      | VideoMAE (legacy)                    |
+=================================+====================================+======================================+
| HuggingFace ID                  | trpakov/vit-face-expression        | MCG-NJU/videomae-base                |
+---------------------------------+------------------------------------+--------------------------------------+
| Pre-training task               | Emotion classification on AffectNet| Masked autoencoding (MAE)            |
+---------------------------------+------------------------------------+--------------------------------------+
| Input                           | Single face-cropped frame 224×224  | 16-frame clip (T, C, H, W)           |
+---------------------------------+------------------------------------+--------------------------------------+
| Output per call                 | (B, T, 768) — T CLS tokens         | (B, ~1568, 768) patch tokens         |
+---------------------------------+------------------------------------+--------------------------------------+
| Domain fit for emotion          | **Direct** — trained on AffectNet  | Indirect — action recognition        |
+---------------------------------+------------------------------------+--------------------------------------+
| Requires face crop              | Yes (MediaPipe — auto-applied)     | No                                   |
+---------------------------------+------------------------------------+--------------------------------------+

Audio backbone — emotion2vec
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``iic/emotion2vec_base`` is a data2vec-style transformer pre-trained on
multiple emotion speech corpora (IEMOCAP, MSP-Podcast, RAVDESS, CREMA-D,
and others).  Like AffectNet ViT on the visual side, emotion2vec already
encodes speech *emotion* — not just acoustic features.

Loaded via **FunASR** ``AutoModel`` (``funasr`` is a required dependency —
``pip install funasr modelscope``).

The emotion2vec backbone is **always frozen**: FunASR does not expose
gradient flow through the PyTorch computation graph.  During Phase 2
fine-tuning, only the projection layer, cross-attention, GRU, and output
heads receive gradient updates on the audio side.

Input:  ``(B, samples)`` raw float32 waveform at 16 kHz.
Output: ``(B, T_a, 768)`` sequence of contextualised frame features.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field  # noqa: F401 (field kept for callers)
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Optional-dependency guards
# ---------------------------------------------------------------------------

try:
    from transformers import ViTModel, VideoMAEModel, VivitModel, ASTModel

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

from funasr import AutoModel as _FunASRAutoModel  # type: ignore[import]

try:
    import mediapipe as mp  # type: ignore[import]

    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    _MEDIAPIPE_AVAILABLE = False

# ImageNet-style normalization used by ViT models (including AffectNet ViT).
_VIT_MEAN = torch.tensor([0.5, 0.5, 0.5]).view(1, 1, 3, 1, 1)
_VIT_STD = torch.tensor([0.5, 0.5, 0.5]).view(1, 1, 3, 1, 1)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class BackboneConfig:
    """Configuration for video and audio backbone models.

    Args:
        video_model: Which video backbone to use.  ``"affectnet_vit"``
            (default) gives the best emotion accuracy.  ``"videomae"`` and
            ``"vivit"`` are retained for backward compatibility.
        video_model_name: HuggingFace model ID for the video backbone.
        video_num_frames: Frames per clip.  16 for AffectNet ViT / VideoMAE,
            32 for ViViT.
        video_image_size: Spatial resolution expected by the backbone.
        video_freeze_layers: Early encoder layers to freeze during fine-tuning.
        face_crop_enabled: When ``True`` (default) each frame is cropped around
            the detected face before being fed to the AffectNet ViT backbone.
            Ignored for VideoMAE / ViViT.
        face_crop_margin: Fractional padding added on each side of the face
            bounding box before cropping (e.g. 0.2 = 20 % margin).
        face_min_confidence: Minimum MediaPipe face-detection confidence.
            Frames with no face above this threshold use a centre crop.
        audio_model: Which audio backbone to use.  ``"emotion2vec"`` (default)
            gives the best accuracy.  ``"ast"`` uses the legacy mel-spectrogram
            AST for backward compatibility.
        audio_model_name: FunASR model ID (for emotion2vec) or HuggingFace ID
            (for AST).
        audio_freeze_layers: Early audio encoder layers to freeze.
        pretrained: When ``False``, lightweight stub backbones are used — useful
            for architecture testing without downloading weights.
        d_model: Shared projection dimension.
    """

    video_model: Literal["affectnet_vit", "videomae", "vivit"] = "affectnet_vit"
    video_model_name: str = "trpakov/vit-face-expression"
    video_num_frames: int = 16
    video_image_size: int = 224
    video_freeze_layers: int = 6

    face_crop_enabled: bool = True
    face_crop_margin: float = 0.2
    face_min_confidence: float = 0.5

    audio_model: Literal["emotion2vec", "ast"] = "emotion2vec"
    audio_model_name: str = "iic/emotion2vec_base"
    audio_freeze_layers: int = 6

    pretrained: bool = True
    d_model: int = 512


# ---------------------------------------------------------------------------
# Face crop pipeline (MediaPipe)
# ---------------------------------------------------------------------------


class FaceCropPipeline:
    """Crop the dominant face from an RGB frame using MediaPipe.

    Falls back to a centre crop when MediaPipe is unavailable or no face is
    detected above ``min_confidence``.

    Args:
        margin: Fractional padding around the detected bounding box.
        min_confidence: Minimum detection confidence score (0–1).
        image_size: Target output resolution (square) in pixels.
    """

    def __init__(
        self,
        margin: float = 0.2,
        min_confidence: float = 0.5,
        image_size: int = 224,
    ) -> None:
        self._margin = margin
        self._min_conf = min_confidence
        self._size = image_size
        self._detector: object | None = None

        if _MEDIAPIPE_AVAILABLE:
            try:
                self._detector = (
                    mp.solutions.face_detection.FaceDetection(  # type: ignore[attr-defined]
                        min_detection_confidence=min_confidence,
                        model_selection=0,  # short-range model (< 2 m)
                    )
                )
            except Exception:
                self._detector = None

    def crop(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Return a face-cropped and resized ``(image_size, image_size, 3)`` frame.

        Args:
            frame_rgb: ``(H, W, 3)`` uint8 RGB frame.

        Returns:
            ``(image_size, image_size, 3)`` uint8 RGB array.
        """
        import cv2  # type: ignore[import]

        H, W = frame_rgb.shape[:2]
        bbox = self._detect_face(frame_rgb, H, W)
        x1, y1, x2, y2 = bbox
        crop = frame_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            crop = frame_rgb  # safety fallback
        return cv2.resize(crop, (self._size, self._size))

    def _detect_face(
        self, frame_rgb: np.ndarray, H: int, W: int
    ) -> tuple[int, int, int, int]:
        """Return pixel bounding box ``(x1, y1, x2, y2)`` or centre crop."""
        if self._detector is None:
            return self._centre_crop(H, W)

        try:
            results = self._detector.process(frame_rgb)  # type: ignore[union-attr]
        except Exception:
            return self._centre_crop(H, W)

        if not results or not results.detections:
            return self._centre_crop(H, W)

        det = results.detections[0]
        bb = det.location_data.relative_bounding_box
        m = self._margin
        x1 = max(0, int((bb.xmin - m * bb.width) * W))
        y1 = max(0, int((bb.ymin - m * bb.height) * H))
        x2 = min(W, int((bb.xmin + (1 + m) * bb.width) * W))
        y2 = min(H, int((bb.ymin + (1 + m) * bb.height) * H))
        if x2 <= x1 or y2 <= y1:
            return self._centre_crop(H, W)
        return x1, y1, x2, y2

    def _centre_crop(self, H: int, W: int) -> tuple[int, int, int, int]:
        side = min(H, W)
        x1 = (W - side) // 2
        y1 = (H - side) // 2
        return x1, y1, x1 + side, y1 + side

    def __del__(self) -> None:
        if self._detector is not None:
            try:
                self._detector.close()  # type: ignore[union-attr]
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Stub backbones (offline / unit-test use only)
# ---------------------------------------------------------------------------


class _StubVideoBackbone(nn.Module):
    """Minimal video backbone for offline testing.

    Mimics the real ViT (HuggingFace ViTModel) interface:
    - Accepts ``(N, C, H, W)`` float tensors (N = B*T after the caller flattens).
    - Returns an object with ``last_hidden_state`` of shape ``(N, 1, hidden_size)``.
      The caller (``VideoBackbone._forward_affectnet_vit``) extracts ``[:, 0, :]``
      as the CLS token then reshapes to ``(B, T, hidden_size)``.
    """

    def __init__(self, hidden_size: int = 768) -> None:
        super().__init__()
        self.config = type("_Cfg", (), {"hidden_size": hidden_size})()
        self._hidden = hidden_size
        self._pool = nn.AdaptiveAvgPool2d((1, 1))
        self._proj = nn.Linear(3, hidden_size)

    def forward(self, pixel_values: torch.Tensor, **_: object) -> object:  # noqa: ANN001
        # pixel_values is (N, C, H, W) — N = B*T after flattening in the caller.
        N, C, H, W = pixel_values.shape
        x = self._pool(pixel_values.float()).reshape(N, C)  # (N, 3)
        cls = self._proj(x).unsqueeze(1)                    # (N, 1, hidden)

        class _Out:
            last_hidden_state = cls  # type: ignore[assignment]

        return _Out()


class _StubAudioBackbone(nn.Module):
    """Minimal audio backbone for offline testing.

    For emotion2vec compatibility: accepts ``(B, samples)`` raw waveform and
    returns ``(B, T_audio, hidden_size)`` where T_audio = samples // 320.
    """

    def __init__(self, hidden_size: int = 768) -> None:
        super().__init__()
        self.config = type("_Cfg", (), {"hidden_size": hidden_size})()
        self._hidden = hidden_size
        self._proj = nn.Linear(1, hidden_size)

    def forward(self, input_values: torch.Tensor, **_: object) -> object:  # noqa: ANN001
        x = input_values.float()  # (B, samples)
        # Down-sample to audio frame rate: 1 frame per 320 samples (like Wav2Vec2)
        stride = 320
        B, S = x.shape
        T = max(1, S // stride)
        # Average-pool to T frames
        x = x[:, : T * stride].reshape(B, T, stride).mean(dim=-1, keepdim=True)  # (B, T, 1)
        x = self._proj(x)  # (B, T, hidden)

        class _Out:
            last_hidden_state = x  # type: ignore[assignment]

        return _Out()


class _StubASTBackbone(nn.Module):
    """Minimal AST stub for legacy mel-spectrogram testing."""

    def __init__(self, hidden_size: int = 768, mel_bins: int = 128) -> None:
        super().__init__()
        self.config = type("_Cfg", (), {"hidden_size": hidden_size})()
        self._hidden = hidden_size
        self._mel = mel_bins
        self._proj = nn.Linear(mel_bins, hidden_size)

    def forward(self, input_values: torch.Tensor, **_: object) -> object:  # noqa: ANN001
        x = input_values.float()
        if x.shape[-1] != self._mel:
            x = F.interpolate(
                x.unsqueeze(1),
                size=(x.shape[1], self._mel),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
        x = self._proj(x)  # (B, time, hidden)

        class _Out:
            last_hidden_state = x  # type: ignore[assignment]

        return _Out()


# ---------------------------------------------------------------------------
# FunASR emotion2vec wrapper
# ---------------------------------------------------------------------------


class _FunASREmotionBackbone(nn.Module):
    """Wraps FunASR emotion2vec as a frozen PyTorch-compatible feature extractor.

    emotion2vec is always frozen: FunASR does not expose gradient flow through
    the PyTorch computation graph.  The projection layer, cross-attention, GRU,
    and output heads are the only learnable components on the audio side.

    hidden_size is 768 (data2vec-base architecture).

    Args:
        model_name: FunASR / ModelScope model identifier
            (e.g. ``"iic/emotion2vec_base"``).
    """

    def __init__(self, model_name: str = "iic/emotion2vec_base") -> None:
        super().__init__()
        self._funasr = _FunASRAutoModel(model=model_name, disable_update=True)
        self.config = type("_Cfg", (), {"hidden_size": 768})()
        # FunASR model is not a standard nn.Module — mark as frozen
        for p in self.parameters():
            p.requires_grad = False
        # Silence FunASR's chatty INFO logging (timing dicts, download progress)
        import logging
        for _noisy in ("funasr", "modelscope", "urllib3", "filelock"):
            logging.getLogger(_noisy).setLevel(logging.WARNING)

    @torch.no_grad()
    def forward(self, input_values: torch.Tensor, **_: object) -> object:  # noqa: ANN001
        """Extract frame-level features from a batch of raw waveforms.

        Args:
            input_values: ``(B, samples)`` float32 waveform at 16 kHz.

        Returns:
            Object with ``.last_hidden_state`` of shape ``(B, T_a, 768)``.
        """
        import contextlib
        import io
        import logging

        device = input_values.device
        results: list[torch.Tensor] = []
        _silence = io.StringIO()

        for i in range(input_values.shape[0]):
            wav_np: np.ndarray = input_values[i].cpu().float().numpy()
            try:
                with contextlib.redirect_stdout(_silence), contextlib.redirect_stderr(_silence):
                    res = self._funasr.generate(
                        wav_np,
                        sample_rate=16000,
                        granularity="frame",
                        extract_embedding=True,
                    )
                feats = res[0].get("feats", res[0].get("hidden_states", None))
            except Exception as exc:
                logging.getLogger(__name__).warning(
                    "AudioBackbone: FunASR generate() failed for batch sample %d, "
                    "substituting zero features. Reason: %s", i, exc,
                )
                feats = None
            if feats is None:
                feats = np.zeros((1, 768), dtype=np.float32)
            if not isinstance(feats, np.ndarray):
                feats = np.array(feats, dtype=np.float32)
            results.append(torch.from_numpy(feats))

        # Pad to equal length
        max_len = max(f.shape[0] for f in results)
        padded = torch.zeros(len(results), max_len, 768)
        for i, f in enumerate(results):
            padded[i, : f.shape[0]] = f

        padded = padded.to(device)

        class _Out:
            last_hidden_state = padded  # type: ignore[assignment]

        return _Out()

    def freeze(self) -> None:
        pass  # already frozen

    def unfreeze(self) -> None:
        pass  # cannot unfreeze FunASR model


# ---------------------------------------------------------------------------
# Public backbone wrappers
# ---------------------------------------------------------------------------


class VideoBackbone(nn.Module):
    """Video backbone supporting AffectNet ViT, VideoMAE, and ViViT.

    **AffectNet ViT** (default, recommended):
        Processes each of the T input frames independently through
        ``trpakov/vit-face-expression``.  Returns one CLS token per frame →
        ``(B, T, 768)``.  Expects face-cropped 224×224 RGB frames, ImageNet
        normalised (mean=0.5, std=0.5).  Use :class:`FaceCropPipeline` in the
        data pre-processing pipeline to produce the correct crops.

    **VideoMAE / ViViT** (legacy):
        Full-clip models that accept ``(B, T, C, H, W)`` and return all
        spatio-temporal patch tokens ``(B, num_tokens, hidden_size)``.

    All variants are accessed through the same ``forward(pixel_values)`` API.

    Args:
        config: Backbone configuration.
    """

    def __init__(self, config: BackboneConfig) -> None:
        super().__init__()
        self._model_type: str = config.video_model

        if not config.pretrained:
            self._backbone: nn.Module = _StubVideoBackbone(hidden_size=512)
            self.hidden_size: int = 512
        elif config.video_model == "affectnet_vit":
            if not _TRANSFORMERS_AVAILABLE:
                raise ImportError(
                    "transformers not installed — cannot load AffectNet ViT backbone."
                )
            self._backbone = ViTModel.from_pretrained(config.video_model_name, token=os.environ.get("HF_TOKEN") or None)
            self.hidden_size = self._backbone.config.hidden_size  # type: ignore[attr-defined]
            self._freeze_layers(
                self._backbone.encoder.layer,  # type: ignore[attr-defined]
                config.video_freeze_layers,
            )
        elif config.video_model == "vivit":
            if not _TRANSFORMERS_AVAILABLE:
                raise ImportError(
                    "transformers not installed — cannot load ViViT backbone."
                )
            self._backbone = VivitModel.from_pretrained(config.video_model_name, token=os.environ.get("HF_TOKEN") or None)
            self.hidden_size = self._backbone.config.hidden_size  # type: ignore[attr-defined]
            self._freeze_layers(
                self._backbone.encoder.layer,  # type: ignore[attr-defined]
                config.video_freeze_layers,
            )
        else:
            # VideoMAE (legacy default)
            if not _TRANSFORMERS_AVAILABLE:
                raise ImportError(
                    "transformers not installed — cannot load VideoMAE backbone."
                )
            self._backbone = VideoMAEModel.from_pretrained(config.video_model_name, token=os.environ.get("HF_TOKEN") or None)
            self.hidden_size = self._backbone.config.hidden_size  # type: ignore[attr-defined]
            self._freeze_layers(
                self._backbone.encoder.layer,  # type: ignore[attr-defined]
                config.video_freeze_layers,
            )

        # Register ImageNet normalisation constants as buffers so they move
        # with the model's device automatically.
        self.register_buffer(
            "_vit_mean", torch.tensor([0.5, 0.5, 0.5]).view(1, 1, 3, 1, 1)
        )
        self.register_buffer(
            "_vit_std", torch.tensor([0.5, 0.5, 0.5]).view(1, 1, 3, 1, 1)
        )

    @staticmethod
    def _freeze_layers(layers: object, n: int) -> None:
        """Freeze the first *n* transformer encoder layers."""
        try:
            for i, layer in enumerate(layers):  # type: ignore[call-overload]
                if i < n:
                    for p in layer.parameters():
                        p.requires_grad = False
        except TypeError:
            pass

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run forward pass.

        Args:
            pixel_values: ``(B, T, C, H, W)`` float tensor, values in ``[0, 1]``.
                For AffectNet ViT, each frame should be a face crop — use
                :class:`FaceCropPipeline` during preprocessing.

        Returns:
            - AffectNet ViT: ``(B, T, hidden_size)`` — one CLS token per frame.
            - VideoMAE / ViViT: ``(B, num_tokens, hidden_size)`` — all patch
              tokens from the full spatio-temporal clip.
        """
        if self._model_type == "affectnet_vit":
            return self._forward_affectnet_vit(pixel_values)
        # VideoMAE and ViViT: pass the full clip as-is
        out = self._backbone(pixel_values=pixel_values)
        return out.last_hidden_state  # type: ignore[return-value]

    def _forward_affectnet_vit(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Process each frame independently and stack CLS tokens.

        Args:
            pixel_values: ``(B, T, C, H, W)`` in ``[0, 1]``.

        Returns:
            ``(B, T, hidden_size)`` — one CLS token per frame.
        """
        B, T, C, H, W = pixel_values.shape
        # Normalise: [0, 1] → [-1, 1] (ViT standard)
        x = (pixel_values - self._vit_mean) / self._vit_std  # type: ignore[operator]

        # Flatten batch and time → run all BT frames through ViT in one shot
        x_flat = x.reshape(B * T, C, H, W)  # (B*T, C, H, W)
        out = self._backbone(pixel_values=x_flat)
        # CLS token is the first token in last_hidden_state
        cls_tokens = out.last_hidden_state[:, 0, :]  # type: ignore[index]  (B*T, hidden)
        return cls_tokens.reshape(B, T, -1)          # (B, T, hidden)

    def freeze(self) -> None:
        """Freeze all backbone parameters."""
        for p in self._backbone.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all backbone parameters."""
        for p in self._backbone.parameters():
            p.requires_grad = True


class AudioBackbone(nn.Module):
    """Audio backbone supporting emotion2vec and the legacy AST.

    **emotion2vec** (default, recommended):
        Loads ``iic/emotion2vec_base`` via FunASR (required dependency).
        Always frozen — FunASR does not expose gradient flow, so only the
        downstream projection layer and fusion layers are trainable.

        Input:  ``(B, samples)`` raw float32 waveform at 16 kHz.
        Output: ``(B, T_a, 768)`` sequence of contextualised emotion features.

    **AST** (legacy):
        ``MIT/ast-finetuned-audioset-10-10-0.4593`` — processes mel-spectrograms.
        Input:  ``(B, time_steps, mel_bins)`` log mel-spectrogram.
        Output: ``(B, num_tokens, 768)`` CLS + patch tokens.

    Args:
        config: Backbone configuration.
    """

    def __init__(self, config: BackboneConfig) -> None:
        super().__init__()
        self._audio_model: str = config.audio_model

        if not config.pretrained:
            if config.audio_model == "ast":
                self._backbone: nn.Module = _StubASTBackbone(hidden_size=512)
            else:
                self._backbone = _StubAudioBackbone(hidden_size=512)
            self.hidden_size: int = 512
        elif config.audio_model == "ast":
            if not _TRANSFORMERS_AVAILABLE:
                raise ImportError(
                    "transformers not installed — cannot load AST backbone."
                )
            self._backbone = ASTModel.from_pretrained(config.audio_model_name, token=os.environ.get("HF_TOKEN") or None)
            self.hidden_size = self._backbone.config.hidden_size  # type: ignore[attr-defined]
            self._freeze_layers(
                self._backbone.encoder.layer,  # type: ignore[attr-defined]
                config.audio_freeze_layers,
            )
        else:
            # emotion2vec via FunASR (required dependency)
            self._backbone = _FunASREmotionBackbone(config.audio_model_name)
            self.hidden_size = self._backbone.config.hidden_size  # type: ignore[attr-defined]

    @staticmethod
    def _freeze_layers(layers: object, n: int) -> None:
        try:
            for i, layer in enumerate(layers):  # type: ignore[call-overload]
                if i < n:
                    for p in layer.parameters():
                        p.requires_grad = False
        except TypeError:
            pass

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """Run forward pass.

        Args:
            input_values: For emotion2vec / HuBERT: ``(B, samples)`` raw
                float32 waveform at 16 kHz.
                For AST: ``(B, time_steps, mel_bins)`` log mel-spectrogram.

        Returns:
            ``(B, T_a, hidden_size)`` last hidden state.
        """
        out = self._backbone(input_values=input_values)
        return out.last_hidden_state  # type: ignore[return-value]

    def freeze(self) -> None:
        for p in self._backbone.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        for p in self._backbone.parameters():
            p.requires_grad = True
