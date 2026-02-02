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
    load_in_4bit: bool = False
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    """Main configuration for the EmotionDetector SDK."""

    # VLA Model settings
    vla_model: str = "openvla/openvla-7b"
    vla_enabled: bool = True  # Can disable VLA for emotion-only mode

    # Device settings
    device: str = "cuda"  # "cuda", "cpu", "mps"
    dtype: str = "float16"  # "float16", "float32", "bfloat16"

    # Face detection settings
    face_detection_model: str = "retinaface"  # "retinaface" or "mtcnn"
    face_detection_threshold: float = 0.9
    face_min_size: int = 20

    # Voice activity detection settings
    vad_aggressiveness: int = 2  # 0-3, higher = more aggressive filtering
    voice_activity_threshold: float = 0.5
    sample_rate: int = 16000

    # Emotion model settings
    facial_emotion_model: str = "trpakov/vit-face-expression"
    speech_emotion_model: str = "superb/wav2vec2-base-superb-er"

    # Fusion settings
    fusion_strategy: Literal["average", "weighted", "max", "confidence"] = "confidence"
    facial_weight: float = 0.6
    speech_weight: float = 0.4
    fusion_confidence_threshold: float = 0.3  # Ignore predictions below this confidence

    # Temporal smoothing settings
    smoothing_strategy: Literal["none", "rolling", "ema", "hysteresis"] = "none"
    smoothing_window: int = 5  # Window size for rolling average
    smoothing_ema_alpha: float = 0.3  # EMA smoothing factor (0-1, lower = smoother)
    smoothing_hysteresis_threshold: float = 0.15  # Min confidence difference to change
    smoothing_hysteresis_frames: int = 3  # Frames emotion must persist

    # Attention analysis settings
    attention_analysis_enabled: bool = True  # Enable attention/gaze analysis
    attention_weight: float = 0.2  # How much attention affects emotion fusion (0-1)
    attention_stress_amplification: float = 1.5  # How much stress amplifies negative emotions
    attention_engagement_threshold: float = 0.3  # Below this, reduce confidence

    # Performance settings
    batch_size: int = 1
    max_faces: int = 5  # Maximum faces to process per frame
    frame_skip: int = 1  # Process every nth frame
    cache_dir: str | None = None

    # Logging
    verbose: bool = False

    def __post_init__(self) -> None:
        """Validate and normalize configuration."""
        # Validate weights
        if not (0 <= self.facial_weight <= 1):
            raise ValueError("facial_weight must be between 0 and 1")
        if not (0 <= self.speech_weight <= 1):
            raise ValueError("speech_weight must be between 0 and 1")

        # Validate thresholds
        if not (0 <= self.face_detection_threshold <= 1):
            raise ValueError("face_detection_threshold must be between 0 and 1")
        if not (0 <= self.voice_activity_threshold <= 1):
            raise ValueError("voice_activity_threshold must be between 0 and 1")

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

    def get_smoothing_config(self) -> dict[str, Any]:
        """Get configuration for temporal smoothing.

        Returns:
            Dictionary with smoothing configuration parameters.
        """
        return {
            "strategy": self.smoothing_strategy,
            "window_size": self.smoothing_window,
            "ema_alpha": self.smoothing_ema_alpha,
            "hysteresis_threshold": self.smoothing_hysteresis_threshold,
            "hysteresis_frames": self.smoothing_hysteresis_frames,
        }

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

