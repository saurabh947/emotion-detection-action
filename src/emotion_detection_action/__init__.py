"""Emotion Detection Action SDK.

A pure-neural, platform-agnostic multimodal emotion detection SDK built on a
Two-Tower Transformer with bidirectional cross-attention and a GRU temporal
context buffer.

Default backbones
-----------------
* **Video** — ``trpakov/vit-face-expression`` (AffectNet ViT-B/16, fine-tuned
  on 450 K facial images, 8-class emotion).  MediaPipe face cropping is applied
  automatically to each frame before the backbone.
* **Audio** — ``iic/emotion2vec_base`` via FunASR, pre-trained on multi-corpus
  speech emotion data (IEMOCAP, MSP-Podcast, RAVDESS, CREMA-D).

Quick start::

    from emotion_detection_action import EmotionDetector, Config

    detector = EmotionDetector(Config(two_tower_pretrained=False))  # stub mode
    detector.initialize()

    import numpy as np
    frames = np.random.randint(0, 255, (16, 480, 640, 3), dtype=np.uint8)
    result = detector.process_frames(frames)
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
    EmotionLabel,
    NeuralEmotionResult,
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
    "EmotionLabel",
    "ActionCommand",
    # Model (for direct use / fine-tuning)
    "NeuralFusionModel",
    "NeuralModelOutput",
    "BackboneConfig",
]
