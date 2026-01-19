"""Configuration management for the emotion detector SDK."""

from dataclasses import dataclass, field
from typing import Any, Literal

from emotion_detector.core.types import ProcessingMode


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

    # Processing mode
    mode: ProcessingMode | str = ProcessingMode.BATCH

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
    fusion_strategy: Literal["average", "weighted", "max", "learned"] = "weighted"
    facial_weight: float = 0.6
    speech_weight: float = 0.4

    # Performance settings
    batch_size: int = 1
    max_faces: int = 5  # Maximum faces to process per frame
    frame_skip: int = 1  # Process every nth frame
    cache_dir: str | None = None

    # Logging
    verbose: bool = False

    def __post_init__(self) -> None:
        """Validate and normalize configuration."""
        # Normalize mode to ProcessingMode enum
        if isinstance(self.mode, str):
            self.mode = ProcessingMode(self.mode.lower())

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
            if isinstance(value, ProcessingMode):
                value = value.value
            result[key] = value
        return result

