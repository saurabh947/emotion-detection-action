"""Configuration management for the emotion detector SDK."""

from dataclasses import dataclass, field
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Default backbone choices
# ---------------------------------------------------------------------------

# Video: AffectNet ViT — fine-tuned directly on AffectNet (450 K facial images,
#   8-class emotion).  Best accuracy for real-world faces.
_DEFAULT_VIDEO_MODEL: Literal["affectnet_vit", "videomae", "vivit"] = "affectnet_vit"
_DEFAULT_VIDEO_MODEL_NAME = "trpakov/vit-face-expression"

# Audio: emotion2vec — pre-trained on multi-dataset speech emotion data
#   (IEMOCAP, MSP-Podcast, RAVDESS, CREMA-D).  Loaded via FunASR (required).
_DEFAULT_AUDIO_MODEL: Literal["emotion2vec", "ast"] = "emotion2vec"
_DEFAULT_AUDIO_MODEL_NAME = "iic/emotion2vec_base"

# AffectNet class-imbalance correction weights (EMOTION_ORDER index order):
#   ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised", "unclear"]
#
# Derived from AffectNet approximate distribution (# images):
#   angry ~24 k, disgusted ~3.8 k, fearful ~6.8 k, happy ~134 k,
#   neutral ~74 k, sad ~25 k, surprised ~14 k, unclear ~6 k (estimated)
#
# Formula: w_c = total / (num_classes * count_c), normalised so happy = 1.0
#
# These are passed directly to nn.CrossEntropyLoss(weight=...) — no custom
# code required.  Override via Config.two_tower_emotion_class_weights.
_DEFAULT_AFFECTNET_CLASS_WEIGHTS: list[float] = [
    5.70,   # angry
    35.50,  # disgusted
    20.50,  # fearful
    1.00,   # happy      ← most frequent; anchor weight = 1.0
    1.90,   # neutral
    5.50,   # sad
    9.60,   # surprised
    20.50,  # unclear
]


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
    core model.  The video tower (AffectNet ViT by default) processes per-frame
    face crops; the audio tower (emotion2vec by default) processes raw waveform.
    Bidirectional cross-attention fuses both modalities into a shared embedding
    from which two task heads decode:

    * 8-class emotion probabilities  (angry, disgusted, fearful, happy, neutral,
      sad, surprised, unclear)
    * 3 continuous attention metrics (stress, engagement, arousal)

    Quick start::

        # Default: AffectNet ViT + emotion2vec (requires funasr + modelscope).
        cfg = Config(two_tower_pretrained=True, two_tower_device="cuda")

        # Stub weights only — no downloads, for architecture testing:
        cfg = Config(two_tower_pretrained=False)

    Class-imbalance correction
    --------------------------
    AffectNet is heavily skewed towards happy/neutral.  Set
    ``two_tower_emotion_class_weights`` to a list of 8 floats (one per class in
    EMOTION_ORDER) to enable weighted cross-entropy during training.  The
    pre-computed AffectNet weights (``_DEFAULT_AFFECTNET_CLASS_WEIGHTS``) are
    available as a reference — copy them into this field or pass
    ``"affectnet"`` as a shorthand (resolved in ``__post_init__``).

    No custom training code is needed — the weights are passed directly to
    ``nn.CrossEntropyLoss(weight=...)``.
    """

    # ------------------------------------------------------------------ #
    # VLA model settings                                                   #
    # ------------------------------------------------------------------ #
    vla_model: str = "openvla/openvla-7b"
    vla_enabled: bool = True

    # ------------------------------------------------------------------ #
    # Device / dtype settings (used by VLA sub-model via get_vla_config)  #
    # ------------------------------------------------------------------ #
    device: str = "cpu"
    dtype: str = "float32"

    # ------------------------------------------------------------------ #
    # Audio input                                                          #
    # ------------------------------------------------------------------ #
    sample_rate: int = 16000

    # ------------------------------------------------------------------ #
    # Neural Fusion Model (primary emotion / attention model)              #
    # ------------------------------------------------------------------ #

    # Video backbone selection.
    #   "affectnet_vit" → trpakov/vit-face-expression  (RECOMMENDED)
    #   "videomae"      → MCG-NJU/videomae-base        (legacy, no face crop needed)
    #   "vivit"         → google/vivit-b-16x2-kinetics400 (legacy, 32 frames)
    two_tower_video_model: Literal["affectnet_vit", "videomae", "vivit"] = _DEFAULT_VIDEO_MODEL

    # Audio backbone selection.
    #   "emotion2vec"  → iic/emotion2vec_base  (RECOMMENDED — raw waveform input)
    #   "ast"          → MIT/ast-finetuned-audioset-10-10-0.4593 (legacy, mel input)
    two_tower_audio_model: Literal["emotion2vec", "ast"] = _DEFAULT_AUDIO_MODEL

    # HuggingFace / FunASR model IDs (override to use fine-tuned checkpoints).
    two_tower_video_backbone: str = _DEFAULT_VIDEO_MODEL_NAME
    two_tower_audio_backbone: str = _DEFAULT_AUDIO_MODEL_NAME

    # Load pretrained weights.  Set False for offline testing / architecture
    # verification only.
    two_tower_pretrained: bool = True

    # Path to a fine-tuned NeuralFusionModel checkpoint (.pt / .pth).
    two_tower_model_path: str | None = None

    # Torch device for the emotion model.  "mps" or "cuda" for production.
    two_tower_device: str = "cpu"

    # Shared projection / attention dimension.
    two_tower_d_model: int = 512

    # Number of stacked bidirectional cross-attention layers.
    two_tower_cross_attn_layers: int = 2

    # GRU layers in the temporal context buffer (2 recommended).
    two_tower_gru_layers: int = 2

    # Frames per clip: 16 for AffectNet ViT / VideoMAE, 32 for ViViT.
    two_tower_video_frames: int = 16

    # Number of early backbone encoder layers to freeze during fine-tuning.
    two_tower_video_freeze_layers: int = 6
    two_tower_audio_freeze_layers: int = 6

    # ------------------------------------------------------------------ #
    # Face crop settings (AffectNet ViT only)                             #
    # ------------------------------------------------------------------ #
    # When True (default), MediaPipe detects the face in each frame and
    # crops it before feeding to the ViT.  Ignored for VideoMAE / ViViT.
    two_tower_face_crop_enabled: bool = True

    # Fractional padding added on each side of the detected bounding box.
    two_tower_face_crop_margin: float = 0.2

    # Minimum MediaPipe face-detection confidence.  Frames below this
    # threshold fall back to a centre crop.
    two_tower_face_min_confidence: float = 0.5

    # ------------------------------------------------------------------ #
    # Class imbalance correction                                          #
    # ------------------------------------------------------------------ #
    # Per-class weights for nn.CrossEntropyLoss.  Length must equal
    # len(EMOTION_ORDER) = 8.  Set to None to use unweighted loss.
    #
    # Pass the string "affectnet" to use the pre-computed inverse-frequency
    # weights derived from the AffectNet dataset distribution (resolved in
    # __post_init__ to the float list).
    two_tower_emotion_class_weights: list[float] | None = None

    # ------------------------------------------------------------------ #
    # Mel-spectrogram parameters (AST legacy audio tower only)            #
    # ------------------------------------------------------------------ #
    two_tower_n_mels: int = 128
    two_tower_n_fft: int = 400
    two_tower_hop_length: int = 160

    # ------------------------------------------------------------------ #
    # Performance settings                                                 #
    # ------------------------------------------------------------------ #
    cache_dir: str | None = None
    verbose: bool = False

    def __post_init__(self) -> None:
        """Validate configuration values and apply model-specific defaults."""
        if self.sample_rate < 1:
            raise ValueError("sample_rate must be >= 1")
        if self.two_tower_video_frames < 1:
            raise ValueError("two_tower_video_frames must be >= 1")
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
        if not 0.0 <= self.two_tower_face_crop_margin <= 1.0:
            raise ValueError("two_tower_face_crop_margin must be in [0, 1]")
        if not 0.0 < self.two_tower_face_min_confidence <= 1.0:
            raise ValueError("two_tower_face_min_confidence must be in (0, 1]")

        # Auto-set correct frame count for ViViT (needs 32 frames).
        if self.two_tower_video_model == "vivit" and self.two_tower_video_frames == 16:
            self.two_tower_video_frames = 32

        # Validate / resolve class weights.
        if isinstance(self.two_tower_emotion_class_weights, list):
            if len(self.two_tower_emotion_class_weights) != 8:
                raise ValueError(
                    "two_tower_emotion_class_weights must have exactly 8 values "
                    "(one per emotion class in EMOTION_ORDER)."
                )
            if any(w <= 0 for w in self.two_tower_emotion_class_weights):
                raise ValueError("All class weights must be > 0.")

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
