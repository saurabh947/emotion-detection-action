"""Core module containing the main detector, configuration, and type definitions."""

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

__all__ = [
    # Primary API
    "EmotionDetector",
    "InferenceWorker",
    "WorkerStats",
    "Config",
    "NeuralEmotionResult",
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

