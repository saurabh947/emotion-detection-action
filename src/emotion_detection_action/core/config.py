"""Configuration management for the emotion detector SDK."""

from dataclasses import dataclass, field
from typing import Any, Literal

# Default backbone choices — see models/backbones.py for full comparison table.
_DEFAULT_VIDEO_MODEL: Literal["videomae", "vivit"] = "videomae"
_DEFAULT_VIDEO_MODEL_NAME = "MCG-NJU/videomae-base"
_DEFAULT_AUDIO_MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"


@dataclass
class ModelConfig:
    """Configuration for a specific model."""

    model_id: str
    device: str = "cpu"
    dtype: str = "float32"
    cache_dir: str | None = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    """Main configuration for the EmotionDetector SDK.

    The SDK uses a Two-Tower Multimodal Emotion Recognition Transformer as its
    core model.  The video tower (VideoMAE) processes a rolling window of
    frames; the audio tower (AST) processes the corresponding mel-spectrogram.
    Bidirectional cross-attention fuses both modalities into a shared embedding
    from which two task heads decode:
      - 8-class emotion probabilities (angry, disgusted, fearful, happy, neutral,
        sad, surprised, unclear)
      - 3 continuous attention metrics (stress, engagement, arousal)

    The model runs out-of-the-box with stub (randomly-initialised) backbones
    for architecture verification.  For production, set ``two_tower_pretrained``
    to ``True`` so the VideoMAE and AST backbones are loaded from HuggingFace.
    For fine-tuned weights, point ``two_tower_model_path`` to a saved checkpoint.
    """

    # ------------------------------------------------------------------ #
    # VLA model settings                                                   #
    # ------------------------------------------------------------------ #
    vla_model: str = "openvla/openvla-7b"
    vla_enabled: bool = True

    # ------------------------------------------------------------------ #
    # Device settings                                                      #
    # ------------------------------------------------------------------ #
    # NOTE: `device` is used only by legacy/VLA sub-models (get_face_detection_config,
    # get_vla_config). Neural inference always uses `two_tower_device` (default "cpu").
    # Do not rely on `device` for the emotion recognition model.
    device: str = "cpu"    # used by face detection and VLA sub-models
    dtype: str = "float32"

    # ------------------------------------------------------------------ #
    # Face detection (MediaPipe)                                           #
    # ------------------------------------------------------------------ #
    face_detection_model: str = "mediapipe"
    face_detection_threshold: float = 0.5
    face_min_size: int = 20

    # ------------------------------------------------------------------ #
    # Voice activity detection                                             #
    # ------------------------------------------------------------------ #
    sample_rate: int = 16000

    # ------------------------------------------------------------------ #
    # Neural Fusion Model (primary emotion / attention model)              #
    # ------------------------------------------------------------------ #

    # Video backbone selection.
    #   "videomae"  → MCG-NJU/videomae-base  (RECOMMENDED — 16 frames, faster)
    #   "vivit"     → google/vivit-b-16x2-kinetics400 (32 frames, higher latency)
    # See models/backbones.py for the full comparison table.
    two_tower_video_model: Literal["videomae", "vivit"] = _DEFAULT_VIDEO_MODEL

    # HuggingFace model IDs (override to use fine-tuned checkpoints).
    two_tower_video_backbone: str = _DEFAULT_VIDEO_MODEL_NAME
    two_tower_audio_backbone: str = _DEFAULT_AUDIO_MODEL_NAME

    # Load pretrained HuggingFace weights.
    # Set False for offline testing / architecture verification only.
    two_tower_pretrained: bool = True

    # Path to a fine-tuned NeuralFusionModel checkpoint (optional).
    # When None the model runs with pretrained backbone weights and
    # randomly-initialised cross-attention + heads (works before fine-tuning).
    two_tower_model_path: str | None = None

    # Torch device for inference.  Recommend "cuda" or "mps" in production.
    two_tower_device: str = "cpu"

    # Shared projection / attention dimension.
    two_tower_d_model: int = 512

    # Number of stacked bidirectional cross-attention layers.
    two_tower_cross_attn_layers: int = 2

    # GRU layers in the temporal context buffer (2 recommended).
    two_tower_gru_layers: int = 2

    # Frames per clip: 16 for VideoMAE, 32 for ViViT.
    two_tower_video_frames: int = 16

    # Number of early backbone encoder layers to freeze during fine-tuning.
    two_tower_video_freeze_layers: int = 8
    two_tower_audio_freeze_layers: int = 6

    # Mel-spectrogram parameters for the audio tower.
    two_tower_n_mels: int = 128
    two_tower_n_fft: int = 400
    two_tower_hop_length: int = 160

    # ------------------------------------------------------------------ #
    # Attention analysis settings                                          #
    # ------------------------------------------------------------------ #
    # MediaPipe-based gaze tracking (used for gaze visualisation overlay).
    # Attention *scores* (stress, engagement, arousal) come from the
    # Two-Tower attention head — not from a deterministic algorithm.
    attention_analysis_enabled: bool = True
    mediapipe_delegate: Literal["cpu", "gpu"] = "cpu"

    # ------------------------------------------------------------------ #
    # Temporal smoothing settings                                          #
    # ------------------------------------------------------------------ #
    smoothing_strategy: Literal["none", "rolling", "ema", "hysteresis"] = "none"
    smoothing_window: int = 5
    smoothing_ema_alpha: float = 0.3
    smoothing_hysteresis_threshold: float = 0.15
    smoothing_hysteresis_frames: int = 3

    # ------------------------------------------------------------------ #
    # Performance settings                                                 #
    # ------------------------------------------------------------------ #
    max_faces: int = 5
    frame_skip: int = 1
    cache_dir: str | None = None
    verbose: bool = False

    def __post_init__(self) -> None:
        """Validate configuration values and apply model-specific defaults."""
        if not 0 <= self.face_detection_threshold <= 1:
            raise ValueError("face_detection_threshold must be between 0 and 1")
        if self.smoothing_window < 1:
            raise ValueError("smoothing_window must be >= 1")
        if not 0 < self.smoothing_ema_alpha <= 1:
            raise ValueError("smoothing_ema_alpha must be in (0, 1]")
        if not 0 <= self.smoothing_hysteresis_threshold <= 1:
            raise ValueError("smoothing_hysteresis_threshold must be in [0, 1]")
        if self.smoothing_hysteresis_frames < 1:
            raise ValueError("smoothing_hysteresis_frames must be >= 1")
        if self.two_tower_video_frames < 1:
            raise ValueError("two_tower_video_frames must be >= 1")
        if self.max_faces < 1:
            raise ValueError("max_faces must be >= 1")
        if self.sample_rate < 1:
            raise ValueError("sample_rate must be >= 1")
        if self.two_tower_d_model < 1:
            raise ValueError("two_tower_d_model must be >= 1")
        if self.two_tower_d_model % 8 != 0:
            raise ValueError("two_tower_d_model must be divisible by 8 (num_heads=8)")
        if self.two_tower_cross_attn_layers < 1:
            raise ValueError("two_tower_cross_attn_layers must be >= 1")
        if self.two_tower_gru_layers < 1:
            raise ValueError("two_tower_gru_layers must be >= 1")
        if self.two_tower_video_freeze_layers < 0:
            raise ValueError("two_tower_video_freeze_layers must be >= 0")
        if self.two_tower_audio_freeze_layers < 0:
            raise ValueError("two_tower_audio_freeze_layers must be >= 0")
        # Auto-set correct frame count for ViViT (needs 32 frames).
        if self.two_tower_video_model == "vivit" and self.two_tower_video_frames == 16:
            self.two_tower_video_frames = 32

    def get_face_detection_config(self) -> ModelConfig:
        """Get configuration for the face detection model."""
        return ModelConfig(
            model_id=self.face_detection_model,
            device=self.device,
            dtype=self.dtype,
            cache_dir=self.cache_dir,
            extra_kwargs={"threshold": self.face_detection_threshold},
        )

    def get_vla_config(self) -> ModelConfig:
        """Get configuration for the VLA model."""
        return ModelConfig(
            model_id=self.vla_model,
            device=self.device,
            dtype=self.dtype,
            cache_dir=self.cache_dir,
            load_in_8bit=True,
        )

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "Config":
        """Create Config from a dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        """Convert Config to a dictionary."""
        return {key: getattr(self, key) for key in self.__dataclass_fields__}
