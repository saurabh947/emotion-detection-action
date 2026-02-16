"""Configuration management for the emotion detector SDK."""

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class ModelConfig:
    """Configuration for a specific model."""

    model_id: str
    device: str = "cpu"
    dtype: str = "float32"
    cache_dir: str | None = None
    load_in_8bit: bool = False
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    """Main configuration for the EmotionDetector SDK.

    The SDK uses AI models for facial emotion, speech emotion, and attention
    analysis. Results are automatically fused using a neural network (MLP)
    that works out-of-the-box with sensible defaults (60% facial, 40% speech).

    For better accuracy, you can train a custom fusion model and provide its
    path via `fusion_model_path`.
    """

    # VLA Model settings
    vla_model: str = "openvla/openvla-7b"
    vla_enabled: bool = True  # Can disable VLA for emotion-only mode

    # Device settings
    device: str = "cuda"  # "cuda", "cpu", "mps"
    dtype: str = "float16"  # "float16", "float32", "bfloat16"

    # Face detection settings (MediaPipe)
    face_detection_model: str = "mediapipe"  # "mediapipe" (short-range) or "mediapipe-full" (long-range)
    face_detection_threshold: float = 0.5  # MediaPipe confidence threshold
    face_min_size: int = 20

    # Voice activity detection settings
    vad_aggressiveness: int = 2  # 0-3, higher = more aggressive filtering
    sample_rate: int = 16000

    # Emotion model settings
    facial_emotion_model: str = "trpakov/vit-face-expression"
    speech_emotion_model: str = "superb/wav2vec2-base-superb-er"

    # Fusion settings (ML-based neural network fusion)
    # By default, uses sensible weights that work out-of-the-box.
    # Provide a trained model path for improved accuracy.
    fusion_model_path: str | None = None  # Path to custom trained fusion model
    fusion_device: str = "cpu"  # Device for fusion model ("cpu", "cuda", "mps")

    # Temporal smoothing settings
    smoothing_strategy: Literal["none", "rolling", "ema", "hysteresis"] = "none"
    smoothing_window: int = 5  # Window size for rolling average
    smoothing_ema_alpha: float = 0.3  # EMA smoothing factor (0-1, lower = smoother)
    smoothing_hysteresis_threshold: float = 0.15  # Min confidence difference to change
    smoothing_hysteresis_frames: int = 3  # Frames emotion must persist

    # Attention analysis settings
    attention_analysis_enabled: bool = True  # Enable attention/gaze analysis

    # Performance settings
    max_faces: int = 5  # Maximum faces to process per frame
    frame_skip: int = 1  # Process every nth frame
    cache_dir: str | None = None

    # Logging
    verbose: bool = False

    def __post_init__(self) -> None:
        """Validate and normalize configuration."""
        # Validate thresholds
        if not (0 <= self.face_detection_threshold <= 1):
            raise ValueError("face_detection_threshold must be between 0 and 1")

        # Validate VAD aggressiveness
        if self.vad_aggressiveness not in (0, 1, 2, 3):
            raise ValueError("vad_aggressiveness must be 0, 1, 2, or 3")

        # Validate smoothing settings
        if self.smoothing_window < 1:
            raise ValueError("smoothing_window must be >= 1")
        if not 0 < self.smoothing_ema_alpha <= 1:
            raise ValueError("smoothing_ema_alpha must be in (0, 1]")
        if not 0 <= self.smoothing_hysteresis_threshold <= 1:
            raise ValueError("smoothing_hysteresis_threshold must be in [0, 1]")
        if self.smoothing_hysteresis_frames < 1:
            raise ValueError("smoothing_hysteresis_frames must be >= 1")

    def get_face_detection_config(self) -> ModelConfig:
        """Get configuration for face detection model."""
        return ModelConfig(
            model_id=self.face_detection_model,
            device=self.device,
            dtype=self.dtype,
            cache_dir=self.cache_dir,
            extra_kwargs={"threshold": self.face_detection_threshold},
        )

    def get_facial_emotion_config(self) -> ModelConfig:
        """Get configuration for facial emotion model."""
        return ModelConfig(
            model_id=self.facial_emotion_model,
            device=self.device,
            dtype=self.dtype,
            cache_dir=self.cache_dir,
        )

    def get_speech_emotion_config(self) -> ModelConfig:
        """Get configuration for speech emotion model."""
        return ModelConfig(
            model_id=self.speech_emotion_model,
            device=self.device,
            dtype=self.dtype,
            cache_dir=self.cache_dir,
        )

    def get_vla_config(self) -> ModelConfig:
        """Get configuration for VLA model."""
        return ModelConfig(
            model_id=self.vla_model,
            device=self.device,
            dtype=self.dtype,
            cache_dir=self.cache_dir,
            load_in_8bit=True,  # VLA models are large, use quantization by default
        )

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "Config":
        """Create Config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {}
        for key in self.__dataclass_fields__:
            value = getattr(self, key)
            result[key] = value
        return result
