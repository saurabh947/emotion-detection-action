"""Pure-neural, platform-agnostic EmotionDetector.

The detector accepts raw numpy frames and numpy/tensor audio — no Silero VAD,
no hardware-specific code.  The entire pipeline runs inside a single
:class:`~models.fusion.NeuralFusionModel` that combines AffectNet ViT (video)
and emotion2vec (audio) through an intra-clip temporal self-attention block,
bidirectional cross-attention, and a GRU temporal context buffer.

Platform agnosticism
--------------------
* Inputs are plain ``numpy.ndarray`` (frames) and ``numpy.ndarray`` /
  ``torch.Tensor`` (audio).
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
from emotion_detection_action.models.backbones import BackboneConfig, FaceCropPipeline
from emotion_detection_action.models.fusion import NeuralFusionModel

# Target spatial resolution expected by ViT-based backbones (AffectNet ViT,
# VideoMAE, ViViT).
_VIDEO_FRAME_SIZE: int = 224

# Maximum raw waveform duration fed to the audio backbone (seconds).
# emotion2vec operates efficiently on 2–3 s chunks at 16 kHz.
_MAX_AUDIO_SECONDS: int = 3


class EmotionDetector:
    """Pure-neural, platform-agnostic multimodal emotion detector.

    Wraps a :class:`~models.fusion.NeuralFusionModel` and handles:

    * Rolling frame buffer (clips sent to the video backbone)
    * Face cropping per-frame via MediaPipe (**AffectNet ViT** mode)
    * Raw PCM → backbone-ready tensor conversion
      - emotion2vec (default): ``(1, samples)`` raw waveform
      - AST (legacy): ``(1, time, mel)`` log mel-spectrogram
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
        self._face_cropper: FaceCropPipeline | None = None

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
        """Load backbone weights and prepare audio / face-crop pipelines.

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
            video_freeze_layers=cfg.two_tower_video_freeze_layers,
            face_crop_enabled=cfg.two_tower_face_crop_enabled,
            face_crop_margin=cfg.two_tower_face_crop_margin,
            face_min_confidence=cfg.two_tower_face_min_confidence,
            audio_model=cfg.two_tower_audio_model,
            audio_model_name=cfg.two_tower_audio_backbone,
            audio_freeze_layers=cfg.two_tower_audio_freeze_layers,
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
            import pathlib

            ckpt_path = pathlib.Path(cfg.two_tower_model_path).resolve()
            if not ckpt_path.is_file():
                raise FileNotFoundError(
                    f"two_tower_model_path does not exist or is not a file: {ckpt_path}"
                )
            if ckpt_path.suffix not in (".pt", ".pth"):
                raise ValueError(
                    f"two_tower_model_path must be a .pt or .pth file, got: {ckpt_path.name}"
                )
            payload = torch.load(
                str(ckpt_path),
                map_location=cfg.two_tower_device,
                weights_only=True,
            )
            state = (
                payload["model_state"]
                if isinstance(payload, dict) and "model_state" in payload
                else payload
            )
            _RENAMES = {
                "absent_video_token": "absent_video",
                "absent_audio_token": "absent_audio",
            }
            state = {_RENAMES.get(k, k): v for k, v in state.items()}
            missing, unexpected = self._model.load_state_dict(state, strict=False)
            real_missing = [
                k
                for k in missing
                if not k.startswith(("absent_video", "absent_audio", "video_temporal"))
            ]
            if real_missing:
                import warnings

                warnings.warn(
                    f"Checkpoint {ckpt_path.name} is missing {len(real_missing)} key(s) "
                    f"that will use random initialisation: {real_missing[:5]}"
                    + (" (…)" if len(real_missing) > 5 else ""),
                    stacklevel=2,
                )
            if cfg.verbose:
                print(f"Loaded checkpoint: {cfg.two_tower_model_path}")
                if missing:
                    print(f"  Keys missing in checkpoint (re-initialised): {missing}")
                if unexpected:
                    print(f"  Unexpected keys in checkpoint (ignored): {unexpected}")

        self._model.to(cfg.two_tower_device)
        self._model.eval()

        # Face crop pipeline — only used for AffectNet ViT.
        if (
            cfg.two_tower_video_model == "affectnet_vit"
            and cfg.two_tower_face_crop_enabled
        ):
            self._face_cropper = FaceCropPipeline(
                margin=cfg.two_tower_face_crop_margin,
                min_confidence=cfg.two_tower_face_min_confidence,
                image_size=_VIDEO_FRAME_SIZE,
            )

        # Legacy AST mel-spectrogram transform.
        if cfg.two_tower_audio_model == "ast":
            try:
                import torchaudio.transforms as _T  # type: ignore[import]

                self._mel_transform = _T.MelSpectrogram(
                    sample_rate=cfg.sample_rate,
                    n_mels=cfg.two_tower_n_mels,
                    n_fft=cfg.two_tower_n_fft,
                    hop_length=cfg.two_tower_hop_length,
                    power=2.0,
                )
            except ImportError:
                if cfg.verbose:
                    print("torchaudio not available — AST audio tower disabled.")

        self.action_handler.connect()
        self._initialized = True

        if cfg.verbose:
            n = self._model.count_parameters()
            device_info = cfg.two_tower_device
            audio_info = f"{cfg.two_tower_audio_model} ({cfg.two_tower_audio_backbone})"
            video_info = f"{cfg.two_tower_video_model} ({cfg.two_tower_video_backbone})"
            print(
                f"NeuralFusionModel ready ({n:,} trainable params) on {device_info}\n"
                f"  video : {video_info}  face_crop={cfg.two_tower_face_crop_enabled}\n"
                f"  audio : {audio_info}"
            )

    def shutdown(self) -> None:
        """Release resources and reset buffers."""
        self._frame_buffer.clear()
        self._audio_buffer.clear()
        if self._model is not None:
            self._model.reset_temporal_state()
        self._face_cropper = None
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
                Any spatial resolution is accepted — frames are resized
                (and face-cropped if enabled) internally.
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

        video_tensor = self._frames_to_tensor(frames)   # (1, T, 3, 224, 224)
        audio_tensor = self._audio_to_tensor(audio)     # (1, samples) or (1, time, mel) or None
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
        into the internal rolling buffer.  Inference runs on every call.

        **Warm-up period:** Until the buffer holds ``config.two_tower_video_frames``
        frames (default 16), the model sees repeat-padded copies of the earliest
        frame prepended to the clip.  Predictions during this warm-up period are
        less reliable — treat any result produced before at least 16 frames have
        been submitted as indicative only.  Call :meth:`reset` when switching
        subjects to restart the warm-up.

        Args:
            frame: ``(H, W, 3)`` BGR uint8 frame from a camera.
                BGR is converted to RGB automatically.  Grayscale or already-RGB
                frames must be explicitly converted before passing.
            audio: Optional raw PCM audio chunk (float32, mono, 16 kHz).
            timestamp: Frame timestamp.  Defaults to ``time.time()``.

        Returns:
            :class:`NeuralEmotionResult`, or ``None`` if the frame buffer is
            completely empty (only possible on the very first call).
        """
        if not self._initialized:
            self.initialize()

        ts = timestamp if timestamp is not None else time.time()

        # Convert BGR → RGB.
        rgb = frame[..., ::-1].copy() if frame.ndim == 3 and frame.shape[2] == 3 else frame
        self._frame_buffer.append(rgb)

        if audio is not None:
            self._audio_buffer.append(audio.astype(np.float32))

        frames_arr = np.stack(list(self._frame_buffer), axis=0)  # (T, H, W, 3)

        if self._audio_buffer:
            full_audio = np.concatenate(list(self._audio_buffer), axis=0)
            max_samples = self.config.sample_rate * _MAX_AUDIO_SECONDS
            audio_arr: np.ndarray | None = full_audio[-max_samples:]
            if len(full_audio) > max_samples:
                self._audio_buffer.clear()
                self._audio_buffer.append(audio_arr)
        else:
            audio_arr = None

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
          are provided.
        * When ``two_tower_video_model == "affectnet_vit"`` and face crop is
          enabled, each frame is first cropped around the detected face using
          :class:`~models.backbones.FaceCropPipeline`, then resized to 224×224.
          VideoMAE / ViViT receive a standard bilinear resize instead.
        """
        import cv2  # type: ignore[import]

        target_t = self.config.two_tower_video_frames

        # Normalise to uint8 for face cropping
        if frames.dtype != np.uint8:
            frames = (frames * 255).clip(0, 255).astype(np.uint8)

        # Repeat-pad along the time axis if needed
        T = frames.shape[0]
        if T < target_t:
            pad = np.repeat(frames[:1], target_t - T, axis=0)
            frames = np.concatenate([pad, frames], axis=0)
        elif T > target_t:
            frames = frames[-target_t:]

        use_face_crop = (
            self.config.two_tower_video_model == "affectnet_vit"
            and self._face_cropper is not None
        )

        processed: list[np.ndarray] = []
        for frame in frames:
            if use_face_crop:
                # Crop face, resize to 224×224
                cropped = self._face_cropper.crop(frame)  # type: ignore[union-attr]
            else:
                cropped = cv2.resize(frame, (_VIDEO_FRAME_SIZE, _VIDEO_FRAME_SIZE))
            processed.append(cropped)

        # Stack → (T, H, W, 3) uint8
        stacked = np.stack(processed, axis=0)
        # (T, H, W, 3) → (T, 3, H, W) float32 [0, 1]
        t = torch.from_numpy(stacked).permute(0, 3, 1, 2).float() / 255.0
        return t.unsqueeze(0)  # (1, T, 3, 224, 224)

    def _audio_to_tensor(
        self,
        audio: np.ndarray | torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Convert raw PCM audio to backbone-ready tensor.

        * **emotion2vec** (default): returns ``(1, samples)`` raw float32
          waveform.  No transformation is needed — the backbone processes raw
          waveform directly.
        * **AST** (legacy): returns ``(1, time_steps, mel_bins)`` log
          mel-spectrogram via torchaudio.

        Returns ``None`` if ``audio`` is ``None``.
        """
        if audio is None:
            return None

        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio.astype(np.float32))

        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # (1, samples)

        if self.config.two_tower_audio_model == "ast":
            # Legacy AST path — need mel-spectrogram
            if self._mel_transform is None:
                return None
            mel = self._mel_transform(audio)          # (1, mel, time)
            mel = torch.log1p(mel)
            mel = mel.squeeze(0).transpose(0, 1)      # (time, mel)
            return mel.unsqueeze(0)                   # (1, time, mel)

        # emotion2vec — return raw waveform (1, samples)
        return audio  # already (1, samples)

    # ------------------------------------------------------------------ #
    # Result construction                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_result(
        out: "NeuralModelOutput",  # noqa: F821
        timestamp: float,
    ) -> NeuralEmotionResult:
        from emotion_detection_action.models.fusion import NeuralFusionModel, NeuralModelOutput  # noqa: F401

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
        crop = (
            f" face_crop={self.config.two_tower_face_crop_enabled}"
            if self.config.two_tower_video_model == "affectnet_vit"
            else ""
        )
        return (
            f"EmotionDetector({status}{q}, "
            f"video={self.config.two_tower_video_model!r}{crop}, "
            f"audio={self.config.two_tower_audio_model!r})"
        )
