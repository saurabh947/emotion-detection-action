"""Emotion recognition modules.

The primary model is :class:`~models.fusion.NeuralFusionModel` in
:mod:`emotion_detection_action.models.fusion`.

This package exposes:

* :class:`TwoTowerEmotionTransformer` — backward-compatible alias for
  ``NeuralFusionModel`` (imports from :mod:`two_tower_transformer`).
* :class:`FacialEmotionRecognizer` / :class:`SpeechEmotionRecognizer` —
  individual modality models for custom pipelines.
* :class:`EmotionSmoother` — temporal smoothing for legacy pipelines.
"""

from emotion_detection_action.emotion.facial import FacialEmotionRecognizer
from emotion_detection_action.emotion.smoothing import EmotionSmoother, SmoothingConfig
from emotion_detection_action.emotion.speech import SpeechEmotionRecognizer
from emotion_detection_action.emotion.two_tower_transformer import (
    ATTENTION_ORDER,
    EMOTION_ORDER,
    TwoTowerConfig,
    TwoTowerEmotionTransformer,
    TwoTowerOutput,
)
from emotion_detection_action.models.fusion import NeuralFusionModel, NeuralModelOutput
from emotion_detection_action.models.backbones import BackboneConfig

__all__ = [
    # Preferred
    "NeuralFusionModel",
    "NeuralModelOutput",
    "BackboneConfig",
    # Legacy shims
    "TwoTowerEmotionTransformer",
    "TwoTowerConfig",
    "TwoTowerOutput",
    "EMOTION_ORDER",
    "ATTENTION_ORDER",
    # Individual modality models
    "FacialEmotionRecognizer",
    "SpeechEmotionRecognizer",
    # Smoothing
    "EmotionSmoother",
    "SmoothingConfig",
]
