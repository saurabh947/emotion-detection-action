"""Neural model components for the Emotion Detection SDK.

Primary exports
---------------
* :class:`NeuralFusionModel` — the complete Two-Tower Transformer (video + audio
  → cross-attention → GRU → emotion + metrics heads).
* :class:`BackboneConfig` — unified configuration for both backbones.
* :class:`VideoBackbone` / :class:`AudioBackbone` — individual backbone wrappers.
* :class:`CrossAttentionBlock` / :class:`TemporalContextBuffer` — fusion building blocks.
* :class:`NeuralModelOutput` — raw tensor output dataclass.

VLA integration
---------------
The :attr:`NeuralModelOutput.latent_embedding` field (shape ``(B, d_model)``)
is the 512-dim fused emotional state vector ready for consumption by downstream
VLA models (e.g., OpenVLA).
"""

from emotion_detection_action.models.backbones import (
    AudioBackbone,
    BackboneConfig,
    VideoBackbone,
)
from emotion_detection_action.models.base import BaseModel
from emotion_detection_action.models.fusion import (
    CrossAttentionBlock,
    EMOTION_ORDER,
    METRIC_ORDER,
    NeuralFusionModel,
    NeuralModelOutput,
    TemporalContextBuffer,
)
from emotion_detection_action.models.registry import ModelRegistry

__all__ = [
    # Backbone
    "BackboneConfig",
    "VideoBackbone",
    "AudioBackbone",
    # Fusion model
    "NeuralFusionModel",
    "NeuralModelOutput",
    "CrossAttentionBlock",
    "TemporalContextBuffer",
    # Label order constants
    "EMOTION_ORDER",
    "METRIC_ORDER",
    # Legacy registry
    "BaseModel",
    "ModelRegistry",
]
