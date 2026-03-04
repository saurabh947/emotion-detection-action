"""Emotion Detection Action SDK.

A pure-neural, platform-agnostic multimodal emotion detection SDK built on a
Two-Tower Transformer (VideoMAE + AST) with bidirectional cross-attention and a
GRU temporal context buffer.

Quick start::

    from emotion_detection_action import EmotionDetector, Config

    detector = EmotionDetector(Config(two_tower_pretrained=False))  # stub mode
    detector.initialize()

    import numpy as np
    frames = np.random.randint(0, 255, (16, 480, 640, 3), dtype=np.uint8)
    result = detector.process(frames)
    print(result.dominant_emotion)
    print(result.latent_embedding[:4])   # 512-dim VLA context vector
    print(result.metrics)                # stress, engagement, arousal

Primary public API
------------------
* :class:`EmotionDetector` — the main SDK entry point.
* :class:`Config` — all configuration knobs.
* :class:`NeuralEmotionResult` — Pydantic output contract.
* :class:`NeuralFusionModel` — the underlying PyTorch model.
* :class:`BackboneConfig` — configure video/audio backbones directly.
"""

from emotion_detection_action.core.config import Config
from emotion_detection_action.core.detector import EmotionDetector
from emotion_detection_action.core.inference_worker import InferenceWorker, WorkerStats
from emotion_detection_action.core.types import (
    ActionCommand,
    AttentionMetrics,
    AttentionResult,
    DetectionResult,
    EmotionResult,
    FaceDetection,
    GazeDetection,
    NeuralEmotionResult,
    VoiceDetection,
)
from emotion_detection_action.models import (
    BackboneConfig,
    NeuralFusionModel,
    NeuralModelOutput,
)

__version__ = "0.2.0"

__all__ = [
    # Core
    "EmotionDetector",
    "InferenceWorker",
    "WorkerStats",
    "Config",
    # Primary output contract
    "NeuralEmotionResult",
    # Model (for direct use / fine-tuning)
    "NeuralFusionModel",
    "NeuralModelOutput",
    "BackboneConfig",
    # Legacy types (kept for backward compatibility)
    "EmotionResult",
    "DetectionResult",
    "ActionCommand",
    "FaceDetection",
    "VoiceDetection",
    "GazeDetection",
    "AttentionResult",
    "AttentionMetrics",
]
