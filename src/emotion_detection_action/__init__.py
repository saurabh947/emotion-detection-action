"""
Emotion Detector SDK - Real-time human emotion detection for robotics.

This SDK combines facial recognition, speech analysis, and attention tracking
using AI models, then fuses them via a neural network into unified emotion
predictions. The ML-based fusion works out-of-the-box with sensible defaults.

Key components:
- Facial emotion recognition (ViT model - 7 emotions)
- Speech emotion recognition (Wav2Vec2 model - 4 emotions)
- Attention analysis (stress, engagement, nervousness metrics)
- ML-based multimodal fusion (neural network)
- VLA-based action generation (optional, for robotics)
"""

from emotion_detection_action.core.config import Config
from emotion_detection_action.core.detector import EmotionDetector
from emotion_detection_action.core.types import (
    ActionCommand,
    AttentionMetrics,
    AttentionResult,
    DetectionResult,
    EmotionResult,
    FaceDetection,
    GazeDetection,
    VoiceDetection,
)

__version__ = "0.1.0"

__all__ = [
    "EmotionDetector",
    "Config",
    "EmotionResult",
    "DetectionResult",
    "ActionCommand",
    "FaceDetection",
    "VoiceDetection",
    "GazeDetection",
    "AttentionResult",
    "AttentionMetrics",
]

