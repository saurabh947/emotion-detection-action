"""Core module containing the main detector, configuration, and type definitions."""

from emotion_detection_action.core.config import Config
from emotion_detection_action.core.detector import EmotionDetector
from emotion_detection_action.core.inference_worker import InferenceWorker, WorkerStats
from emotion_detection_action.core.types import (
    ActionCommand,
    EmotionLabel,
    NeuralEmotionResult,
)

__all__ = [
    "EmotionDetector",
    "InferenceWorker",
    "WorkerStats",
    "Config",
    "NeuralEmotionResult",
    "ActionCommand",
    "EmotionLabel",
]
