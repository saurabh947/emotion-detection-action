"""Type definitions and dataclasses for the emotion detector SDK."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class EmotionLabel(str, Enum):
    """Standard emotion labels."""

    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    SURPRISED = "surprised"
    DISGUSTED = "disgusted"
    NEUTRAL = "neutral"


@dataclass
class BoundingBox:
    """Bounding box for detected regions."""

    x: int
    y: int
    width: int
    height: int

    def to_tuple(self) -> tuple[int, int, int, int]:
        """Convert to (x, y, w, h) tuple."""
        return (self.x, self.y, self.width, self.height)

    def to_xyxy(self) -> tuple[int, int, int, int]:
        """Convert to (x1, y1, x2, y2) format."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)


@dataclass
class FaceDetection:
    """Result of face detection."""

    bbox: BoundingBox
    confidence: float
    landmarks: np.ndarray | None = None  # Facial landmarks if available
    face_image: np.ndarray | None = None  # Cropped face image

    def __post_init__(self) -> None:
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError("Confidence must be between 0 and 1")


@dataclass
class VoiceDetection:
    """Result of voice activity detection."""

    is_speech: bool
    confidence: float
    start_time: float  # Start time in seconds
    end_time: float  # End time in seconds
    audio_segment: np.ndarray | None = None  # Audio data for the segment

    @property
    def duration(self) -> float:
        """Duration of the voice segment in seconds."""
        return self.end_time - self.start_time


@dataclass
class DetectionResult:
    """Combined detection results for a frame/segment."""

    timestamp: float
    faces: list[FaceDetection] = field(default_factory=list)
    voice: VoiceDetection | None = None
    gaze: GazeDetection | None = None
    frame: np.ndarray | None = None  # Original frame if available


@dataclass
class EmotionScores:
    """Emotion probability scores."""

    happy: float = 0.0
    sad: float = 0.0
    angry: float = 0.0
    fearful: float = 0.0
    surprised: float = 0.0
    disgusted: float = 0.0
    neutral: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "happy": self.happy,
            "sad": self.sad,
            "angry": self.angry,
            "fearful": self.fearful,
            "surprised": self.surprised,
            "disgusted": self.disgusted,
            "neutral": self.neutral,
        }

    @property
    def dominant_emotion(self) -> EmotionLabel:
        """Get the emotion with highest probability."""
        scores = self.to_dict()
        dominant = max(scores, key=lambda k: scores[k])
        return EmotionLabel(dominant)

    @classmethod
    def from_dict(cls, scores: dict[str, float]) -> "EmotionScores":
        """Create from dictionary."""
        return cls(
            happy=scores.get("happy", 0.0),
            sad=scores.get("sad", 0.0),
            angry=scores.get("angry", 0.0),
            fearful=scores.get("fearful", 0.0),
            surprised=scores.get("surprised", 0.0),
            disgusted=scores.get("disgusted", 0.0),
            neutral=scores.get("neutral", 0.0),
        )


@dataclass
class EyeDetection:
    """Result of eye/gaze detection for a single eye."""

    center: tuple[float, float]  # (x, y) center of eye
    landmarks: np.ndarray | None = None  # Eye landmarks if available
    pupil_size: float = 0.0  # Estimated pupil size (normalized)
    openness: float = 1.0  # Eye openness ratio (0=closed, 1=fully open)


@dataclass
class GazeDetection:
    """Result of gaze detection."""

    left_eye: EyeDetection | None = None
    right_eye: EyeDetection | None = None
    gaze_direction: tuple[float, float] = (0.0, 0.0)  # (x, y) normalized gaze vector
    gaze_point: tuple[float, float] | None = None  # Where user is looking on screen
    confidence: float = 0.0

    @property
    def avg_pupil_size(self) -> float:
        """Average pupil size from both eyes."""
        sizes = []
        if self.left_eye:
            sizes.append(self.left_eye.pupil_size)
        if self.right_eye:
            sizes.append(self.right_eye.pupil_size)
        return sum(sizes) / len(sizes) if sizes else 0.0

    @property
    def avg_eye_openness(self) -> float:
        """Average eye openness from both eyes."""
        openness = []
        if self.left_eye:
            openness.append(self.left_eye.openness)
        if self.right_eye:
            openness.append(self.right_eye.openness)
        return sum(openness) / len(openness) if openness else 1.0


@dataclass
class AttentionMetrics:
    """Metrics derived from attention analysis."""

    # Raw measurements
    pupil_dilation: float = 0.0  # Change from baseline (positive = dilated)
    gaze_stability: float = 1.0  # How stable the gaze is (0=unstable, 1=stable)
    blink_rate: float = 0.0  # Blinks per minute
    eye_contact_ratio: float = 0.0  # Ratio of time looking at camera

    # Derived scores (0-1, higher = more intense)
    stress_score: float = 0.0  # Based on pupil dilation + blink rate
    engagement_score: float = 0.0  # Based on eye contact + fixation
    nervousness_score: float = 0.0  # Based on gaze aversion + instability

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "pupil_dilation": self.pupil_dilation,
            "gaze_stability": self.gaze_stability,
            "blink_rate": self.blink_rate,
            "eye_contact_ratio": self.eye_contact_ratio,
            "stress_score": self.stress_score,
            "engagement_score": self.engagement_score,
            "nervousness_score": self.nervousness_score,
        }


@dataclass
class AttentionResult:
    """Result of attention analysis."""

    timestamp: float
    gaze: GazeDetection | None = None
    metrics: AttentionMetrics = field(default_factory=AttentionMetrics)
    confidence: float = 0.0

    @property
    def stress_score(self) -> float:
        """Shortcut to stress score."""
        return self.metrics.stress_score

    @property
    def engagement_score(self) -> float:
        """Shortcut to engagement score."""
        return self.metrics.engagement_score

    @property
    def nervousness_score(self) -> float:
        """Shortcut to nervousness score."""
        return self.metrics.nervousness_score


@dataclass
class FacialEmotionResult:
    """Result of facial emotion recognition."""

    face_detection: FaceDetection
    emotions: EmotionScores
    confidence: float


@dataclass
class SpeechEmotionResult:
    """Result of speech emotion recognition."""

    voice_detection: VoiceDetection
    emotions: EmotionScores
    confidence: float


@dataclass
class EmotionResult:
    """Combined multimodal emotion result."""

    timestamp: float
    emotions: EmotionScores
    facial_result: FacialEmotionResult | None = None
    speech_result: SpeechEmotionResult | None = None
    attention_result: AttentionResult | None = None
    fusion_confidence: float = 0.0

    @property
    def dominant_emotion(self) -> EmotionLabel:
        """Get the dominant emotion."""
        return self.emotions.dominant_emotion

    @property
    def attention(self) -> AttentionResult | None:
        """Shortcut to attention result."""
        return self.attention_result


@dataclass
class ActionCommand:
    """Robot action command generated by VLA model."""

    action_type: str
    parameters: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    raw_output: Any = None  # Raw VLA model output

    @classmethod
    def stub(cls, emotion_context: EmotionResult) -> "ActionCommand":
        """Create a stub action based on emotion context."""
        return cls(
            action_type="stub",
            parameters={
                "emotion": emotion_context.dominant_emotion.value,
                "emotion_scores": emotion_context.emotions.to_dict(),
            },
            confidence=0.0,
            raw_output=None,
        )


@dataclass
class PipelineResult:
    """Complete result from the emotion detection pipeline."""

    timestamp: float
    detection: DetectionResult
    emotion: EmotionResult
    action: ActionCommand

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "emotions": self.emotion.emotions.to_dict(),
            "dominant_emotion": self.emotion.dominant_emotion.value,
            "action": {
                "type": self.action.action_type,
                "parameters": self.action.parameters,
                "confidence": self.action.confidence,
            },
        }

