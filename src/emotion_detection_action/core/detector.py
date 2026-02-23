"""Pure-neural, platform-agnostic EmotionDetector.

The detector accepts raw numpy frames and numpy/tensor audio — no MediaPipe,
no Silero VAD, no hardware-specific code.  The entire pipeline runs inside a
single :class:`~models.fusion.NeuralFusionModel` that combines VideoMAE
(video) and AST (audio) through bidirectional cross-attention and a GRU
temporal context buffer.

Platform agnosticism
--------------------
* Inputs are plain ``numpy.ndarray`` (frames) and ``numpy.ndarray`` / ``torch.Tensor`` (audio).
* The :class:`BaseActionHandler` interface is the only robot-facing dependency;
  users subclass it for their specific platform.
* No GPIO, no ROS, no robot-SDK imports anywhere in this module.

Output contract
---------------
Every call returns a :class:`~core.types.NeuralEmotionResult` Pydantic model
containing:

* ``dominant_emotion`` — string label
* ``latent_embedding`` — 512-dim float list (VLA input)
* ``metrics`` — ``{stress, engagement, arousal}`` in ``[0, 1]``
* ``confidence`` — max softmax probability
"""

from __future__ import annotations

import time
from collections import deque
from typing import AsyncIterator, Iterator, Literal

import numpy as np
import torch
import torch.nn.functional as F

from emotion_detection_action.actions.base import BaseActionHandler
from emotion_detection_action.actions.logging_handler import LoggingActionHandler
from emotion_detection_action.core.config import Config
from emotion_detection_action.core.types import NeuralEmotionResult
from emotion_detection_action.models.backbones import BackboneConfig
from emotion_detection_action.models.fusion import NeuralFusionModel

try:
    import torchaudio.transforms as _T

    _TORCHAUDIO_AVAILABLE = True
except ImportError:
    _TORCHAUDIO_AVAILABLE = False

# Target spatial resolution expected by VideoMAE and ViViT.
_VIDEO_FRAME_SIZE: int = 224


class EmotionDetector:
    """Pure-neural, platform-agnostic multimodal emotion detector.

    Wraps a :class:`~models.fusion.NeuralFusionModel` and handles:

    * Rolling frame buffer (clips sent to the video backbone)
    * Raw PCM → log mel-spectrogram conversion (torchaudio)
    * Temporal GRU state management
    * Quantization for deployment
    * A :class:`~actions.base.BaseActionHandler` integration point

    Typical usage::

        detector = EmotionDetector()
        detector.initialize()

        # Single-frame API (builds clip buffer internally)
        for bgr_frame, audio_chunk in my_sensor_loop():
            result = detector.process_frame(bgr_frame, audio_chunk)
            if result:
                print(result.dominant_emotion, result.metrics)

        # Clip API (caller manages frame batching)
        clip = np.stack([frame1, frame2, ..., frame16])  # (16, H, W, 3)
        result = detector.process(clip, audio_samples)

        # Quantize for low-latency deployment
        detector.quantize("dynamic")

    Args:
        config: SDK configuration.  Defaults are used if ``None``.
        action_handler: Optional handler for robot actions.
    """

    def __init__(
        self,
        config: Config | None = None,
        action_handler: BaseActionHandler | None = None,
    ) -> None:
        self.config = config or Config()
        self.action_handler = action_handler or LoggingActionHandler(
            verbose=self.config.verbose
        )

        self._model: NeuralFusionModel | None = None
        self._mel_transform: object | None = None

        # Frame buffer: (T, H, W, 3) RGB numpy arrays
        self._frame_buffer: deque[np.ndarray] = deque(
            maxlen=self.config.two_tower_video_frames
        )
        # Raw PCM buffer: list of float32 numpy arrays
        self._audio_buffer: list[np.ndarray] = []

        self._initialized = False
        self._is_quantized = False

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def is_quantized(self) -> bool:
        return self._is_quantized

    def initialize(self) -> None:
        """Load backbone weights and prepare the mel-spectrogram transform.

        Calling this explicitly is optional — ``process`` and ``process_frame``
        call it lazily on the first invocation.
        """
        if self._initialized:
            return

        cfg = self.config

        backbone_cfg = BackboneConfig(
            video_model=cfg.two_tower_video_model,
            video_model_name=cfg.two_tower_video_backbone,
            video_num_frames=cfg.two_tower_video_frames,
            audio_model_name=cfg.two_tower_audio_backbone,
            audio_freeze_layers=cfg.two_tower_audio_freeze_layers,
            video_freeze_layers=cfg.two_tower_video_freeze_layers,
            pretrained=cfg.two_tower_pretrained,
            d_model=cfg.two_tower_d_model,
        )

        self._model = NeuralFusionModel(
            config=backbone_cfg,
            num_cross_attn_layers=cfg.two_tower_cross_attn_layers,
            num_heads=8,
            gru_layers=cfg.two_tower_gru_layers,
        )

        if cfg.two_tower_model_path:
            state = torch.load(
                cfg.two_tower_model_path, map_location=cfg.two_tower_device
            )
            self._model.load_state_dict(state)
            if cfg.verbose:
                print(f"Loaded checkpoint: {cfg.two_tower_model_path}")

        self._model.to(cfg.two_tower_device)
        self._model.eval()

        if _TORCHAUDIO_AVAILABLE:
            self._mel_transform = _T.MelSpectrogram(
                sample_rate=cfg.sample_rate,
                n_mels=cfg.two_tower_n_mels,
                n_fft=cfg.two_tower_n_fft,
                hop_length=cfg.two_tower_hop_length,
                power=2.0,
            )
        elif cfg.verbose:
            print("torchaudio not available — audio tower will use zero tensors.")

        self.action_handler.connect()
        self._initialized = True

        if cfg.verbose:
            n = self._model.count_parameters()
            print(f"NeuralFusionModel ready ({n:,} trainable params) on {cfg.two_tower_device}")

    def shutdown(self) -> None:
        """Release resources and reset buffers."""
        self._frame_buffer.clear()
        self._audio_buffer.clear()
        if self._model is not None:
            self._model.reset_temporal_state()
        self.action_handler.disconnect()
        self._initialized = False

    # ------------------------------------------------------------------ #
    # Quantization                                                         #
    # ------------------------------------------------------------------ #

    def quantize(self, mode: Literal["dynamic", "static"] = "dynamic") -> None:
        """Quantize the underlying model to INT8 for faster CPU inference.

        Should be called **after** ``initialize()`` and **before** streaming.
        Dynamic quantization is recommended — no calibration data required.

        Args:
            mode: ``"dynamic"`` (default) — weights quantized statically,
                activations computed at runtime.  ``~2-3×`` speedup on CPU
                with ``~4×`` memory reduction.

        Raises:
            RuntimeError: If ``initialize()`` has not been called yet.
        """
        if self._model is None:
            raise RuntimeError("Call initialize() before quantize().")
        self._model = self._model.quantize(mode)
        self._is_quantized = True
        if self.config.verbose:
            print(f"Model quantized ({mode} INT8).")

    # ------------------------------------------------------------------ #
    # Single-clip processing (primary API)                                 #
    # ------------------------------------------------------------------ #

    def process(
        self,
        frames: np.ndarray,
        audio: np.ndarray | torch.Tensor | None = None,
        timestamp: float | None = None,
    ) -> NeuralEmotionResult:
        """Process a complete clip of frames with optional audio.

        Args:
            frames: ``(T, H, W, 3)`` uint8 or float32 RGB numpy array.
                Any spatial resolution is accepted — frames are resized to
                224×224 internally.
            audio: Raw PCM samples (float32, mono) as a numpy array or
                torch.Tensor.  Pass ``None`` for video-only mode.
            timestamp: Clip timestamp in seconds.  Defaults to ``time.time()``.

        Returns:
            :class:`~core.types.NeuralEmotionResult` with all output fields.
        """
        if not self._initialized:
            self.initialize()

        ts = timestamp if timestamp is not None else time.time()
        assert self._model is not None

        video_tensor = self._frames_to_tensor(frames)        # (1, T, 3, 224, 224)
        audio_tensor = self._audio_to_tensor(audio)          # (1, time, mel) or None
        device = self.config.two_tower_device
        video_tensor = video_tensor.to(device)
        if audio_tensor is not None:
            audio_tensor = audio_tensor.to(device)

        with torch.no_grad():
            out = self._model(video_tensor, audio_tensor, use_temporal=True)

        return self._build_result(out, ts)

    def process_frame(
        self,
        frame: np.ndarray,
        audio: np.ndarray | None = None,
        timestamp: float | None = None,
    ) -> NeuralEmotionResult | None:
        """Accumulate one BGR frame into the rolling buffer and run inference.

        Accepts a **single** BGR frame (as OpenCV produces) and accumulates it
        into the internal rolling buffer.  Inference runs every time the buffer
        is primed (from the first call onward, with repeat-padding for early
        frames).

        Args:
            frame: ``(H, W, 3)`` BGR or RGB uint8 frame from a camera.
                BGR frames are converted to RGB automatically.
            audio: Optional raw PCM audio chunk (float32, mono).
            timestamp: Frame timestamp.  Defaults to ``time.time()``.

        Returns:
            :class:`NeuralEmotionResult`, or ``None`` if the frame buffer is
            completely empty (only possible on the very first call).
        """
        if not self._initialized:
            self.initialize()

        ts = timestamp if timestamp is not None else time.time()

        # Convert BGR → RGB if the frame looks like a typical OpenCV frame.
        rgb = frame[..., ::-1].copy() if frame.ndim == 3 and frame.shape[2] == 3 else frame
        self._frame_buffer.append(rgb)

        if audio is not None:
            self._audio_buffer.append(audio.astype(np.float32))

        if not self._frame_buffer:
            return None

        frames_arr = np.stack(list(self._frame_buffer), axis=0)  # (T, H, W, 3)
        audio_arr = (
            np.concatenate(self._audio_buffer[-self.config.sample_rate * 3 :], axis=0)
            if self._audio_buffer
            else None
        )
        return self.process(frames_arr, audio_arr, timestamp=ts)

    # ------------------------------------------------------------------ #
    # Streaming helpers                                                    #
    # ------------------------------------------------------------------ #

    def stream_frames(
        self,
        frame_iterator: Iterator[tuple[np.ndarray, np.ndarray | None]],
    ) -> Iterator[NeuralEmotionResult]:
        """Wrap a (frame, audio) iterator to yield :class:`NeuralEmotionResult`.

        Args:
            frame_iterator: An iterator yielding ``(frame, audio_chunk)`` tuples.
                ``audio_chunk`` may be ``None``.

        Yields:
            :class:`NeuralEmotionResult` for each frame.

        Example::

            def my_camera() -> Iterator[tuple[np.ndarray, None]]:
                cap = cv2.VideoCapture(0)
                while True:
                    ret, frame = cap.read()
                    if ret:
                        yield frame, None

            for result in detector.stream_frames(my_camera()):
                print(result.dominant_emotion)
        """
        if not self._initialized:
            self.initialize()
        for frame, audio in frame_iterator:
            result = self.process_frame(frame, audio)
            if result is not None:
                yield result

    # ------------------------------------------------------------------ #
    # State management                                                     #
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Reset frame buffer, audio buffer, and GRU temporal state.

        Call this when the subject changes (e.g., a new person sits in front
        of the robot) so past context does not bleed into the new session.
        """
        self._frame_buffer.clear()
        self._audio_buffer.clear()
        if self._model is not None:
            self._model.reset_temporal_state()

    # ------------------------------------------------------------------ #
    # Tensor construction helpers                                          #
    # ------------------------------------------------------------------ #

    def _frames_to_tensor(self, frames: np.ndarray) -> torch.Tensor:
        """Convert ``(T, H, W, 3)`` numpy frames → ``(1, T, 3, 224, 224)`` tensor.

        * Handles both uint8 [0, 255] and float32 [0, 1] inputs.
        * Repeat-pads at the start if fewer than ``two_tower_video_frames`` frames
          are provided (preserves the most recent frames on the right).
        * Resizes to 224×224 via bilinear interpolation.
        """
        target_t = self.config.two_tower_video_frames

        # Normalise to float32 [0, 1]
        if frames.dtype == np.uint8:
            frames = frames.astype(np.float32) / 255.0

        # Repeat-pad along the time axis if needed
        T = frames.shape[0]
        if T < target_t:
            pad = np.repeat(frames[:1], target_t - T, axis=0)
            frames = np.concatenate([pad, frames], axis=0)
        elif T > target_t:
            frames = frames[-target_t:]

        # (T, H, W, 3) → (T, 3, H, W)
        t = torch.from_numpy(frames).permute(0, 3, 1, 2).float()

        # Resize spatial dims if needed
        H, W = t.shape[-2:]
        if (H, W) != (_VIDEO_FRAME_SIZE, _VIDEO_FRAME_SIZE):
            t = F.interpolate(
                t,
                size=(_VIDEO_FRAME_SIZE, _VIDEO_FRAME_SIZE),
                mode="bilinear",
                align_corners=False,
            )

        return t.unsqueeze(0)  # (1, T, 3, 224, 224)

    def _audio_to_tensor(
        self,
        audio: np.ndarray | torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Convert raw PCM audio → ``(1, time_steps, mel_bins)`` tensor.

        Returns ``None`` if ``audio`` is ``None`` or torchaudio is unavailable.
        """
        if audio is None or self._mel_transform is None:
            return None

        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio.astype(np.float32))

        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # (1, samples)

        mel = self._mel_transform(audio)             # (1, mel, time)
        mel = torch.log1p(mel)                       # log-compression
        mel = mel.squeeze(0).transpose(0, 1)         # (time, mel)
        return mel.unsqueeze(0)                      # (1, time, mel)

    # ------------------------------------------------------------------ #
    # Result construction                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_result(
        out: "NeuralFusionModel.__class__.__mro__[0]",
        timestamp: float,
    ) -> NeuralEmotionResult:
        from emotion_detection_action.models.fusion import NeuralFusionModel

        probs = out.emotion_probs[0].cpu().tolist()
        emotion_scores = dict(zip(NeuralFusionModel.EMOTION_ORDER, probs))
        dominant = max(emotion_scores, key=emotion_scores.__getitem__)

        metrics_vals = out.metrics[0].cpu().tolist()
        metrics = dict(zip(NeuralFusionModel.METRIC_ORDER, metrics_vals))

        embedding = out.latent_embedding[0].cpu().tolist()

        return NeuralEmotionResult(
            dominant_emotion=dominant,
            emotion_scores=emotion_scores,
            latent_embedding=embedding,
            metrics=metrics,
            confidence=float(max(probs)),
            timestamp=timestamp,
            video_missing=out.video_missing,
            audio_missing=out.audio_missing,
        )

    # ------------------------------------------------------------------ #
    # Context manager                                                      #
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "EmotionDetector":
        self.initialize()
        return self

    def __exit__(self, *_: object) -> None:
        self.shutdown()

    def __repr__(self) -> str:
        status = "initialized" if self._initialized else "not initialized"
        q = " [quantized]" if self._is_quantized else ""
        return f"EmotionDetector({status}{q}, video={self.config.two_tower_video_model!r})"
