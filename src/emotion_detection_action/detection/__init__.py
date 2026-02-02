"""Detection modules for face, voice, and attention detection."""

from emotion_detection_action.detection.attention import AttentionDetector
from emotion_detection_action.detection.face import FaceDetector
from emotion_detection_action.detection.voice import VoiceActivityDetector

__all__ = [
    "FaceDetector",
    "VoiceActivityDetector",
    "AttentionDetector",
]

