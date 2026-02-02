"""Emotion recognition modules for facial, speech, and attention analysis."""

from emotion_detection_action.emotion.attention import AttentionAnalyzer
from emotion_detection_action.emotion.facial import FacialEmotionRecognizer
from emotion_detection_action.emotion.fusion import EmotionFusion
from emotion_detection_action.emotion.speech import SpeechEmotionRecognizer

__all__ = [
    "FacialEmotionRecognizer",
    "SpeechEmotionRecognizer",
    "EmotionFusion",
    "AttentionAnalyzer",
]

