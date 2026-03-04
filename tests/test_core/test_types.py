"""Tests for core type definitions."""

import numpy as np
import pytest

from emotion_detection_action.core.types import (
    ActionCommand,
    BoundingBox,
    DetectionResult,
    EmotionLabel,
    EmotionResult,
    EmotionScores,
    FaceDetection,
    FacialEmotionResult,
    NeuralEmotionResult,
    PipelineResult,
    SpeechEmotionResult,
    VoiceDetection,
)


class TestBoundingBox:
    """Tests for BoundingBox dataclass."""

    def test_creation(self):
        """Test creating a bounding box."""
        bbox = BoundingBox(x=10, y=20, width=100, height=150)
        assert bbox.x == 10
        assert bbox.y == 20
        assert bbox.width == 100
        assert bbox.height == 150

    def test_to_tuple(self):
        """Test converting to (x, y, w, h) tuple."""
        bbox = BoundingBox(x=10, y=20, width=100, height=150)
        assert bbox.to_tuple() == (10, 20, 100, 150)

    def test_to_xyxy(self):
        """Test converting to (x1, y1, x2, y2) format."""
        bbox = BoundingBox(x=10, y=20, width=100, height=150)
        assert bbox.to_xyxy() == (10, 20, 110, 170)


class TestFaceDetection:
    """Tests for FaceDetection dataclass."""

    def test_creation(self):
        """Test creating a face detection."""
        bbox = BoundingBox(x=0, y=0, width=100, height=100)
        detection = FaceDetection(bbox=bbox, confidence=0.95)
        assert detection.confidence == 0.95
        assert detection.landmarks is None
        assert detection.face_image is None

    def test_invalid_confidence_high(self):
        """Test that confidence > 1 raises error."""
        bbox = BoundingBox(x=0, y=0, width=100, height=100)
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            FaceDetection(bbox=bbox, confidence=1.5)

    def test_invalid_confidence_low(self):
        """Test that confidence < 0 raises error."""
        bbox = BoundingBox(x=0, y=0, width=100, height=100)
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            FaceDetection(bbox=bbox, confidence=-0.1)

    def test_with_face_image(self):
        """Test creating detection with face image."""
        bbox = BoundingBox(x=0, y=0, width=100, height=100)
        face_img = np.zeros((100, 100, 3), dtype=np.uint8)
        detection = FaceDetection(bbox=bbox, confidence=0.9, face_image=face_img)
        assert detection.face_image is not None
        assert detection.face_image.shape == (100, 100, 3)


class TestVoiceDetection:
    """Tests for VoiceDetection dataclass."""

    def test_creation(self):
        """Test creating a voice detection."""
        detection = VoiceDetection(
            is_speech=True,
            confidence=0.8,
            start_time=1.0,
            end_time=2.5,
        )
        assert detection.is_speech is True
        assert detection.confidence == 0.8
        assert detection.start_time == 1.0
        assert detection.end_time == 2.5

    def test_duration_property(self):
        """Test duration calculation."""
        detection = VoiceDetection(
            is_speech=True,
            confidence=0.8,
            start_time=1.0,
            end_time=3.5,
        )
        assert detection.duration == 2.5


class TestEmotionScores:
    """Tests for EmotionScores dataclass."""

    _ALL_KEYS = [
        "happy", "sad", "angry", "fearful",
        "surprised", "disgusted", "neutral", "unclear",
    ]

    def test_default_values(self):
        """All 8 fields default to 0.0."""
        scores = EmotionScores()
        assert scores.happy == 0.0
        assert scores.sad == 0.0
        assert scores.angry == 0.0
        assert scores.neutral == 0.0
        assert scores.unclear == 0.0

    def test_to_dict_has_all_8_classes(self):
        """to_dict() must include all 8 emotion keys."""
        d = EmotionScores().to_dict()
        assert set(d.keys()) == set(self._ALL_KEYS)

    def test_to_dict(self):
        """Test converting scores to dictionary."""
        scores = EmotionScores(happy=0.8, neutral=0.2)
        d = scores.to_dict()
        assert d["happy"] == 0.8
        assert d["neutral"] == 0.2
        assert d["sad"] == 0.0
        assert d["unclear"] == 0.0

    def test_from_dict(self):
        """Test creating scores from dictionary."""
        d = {"happy": 0.7, "sad": 0.1, "neutral": 0.2}
        scores = EmotionScores.from_dict(d)
        assert scores.happy == 0.7
        assert scores.sad == 0.1
        assert scores.neutral == 0.2
        assert scores.angry == 0.0   # default for missing keys
        assert scores.unclear == 0.0  # default for missing keys

    def test_from_dict_with_unclear(self):
        """from_dict() correctly round-trips the unclear field."""
        d = {k: 0.0 for k in self._ALL_KEYS}
        d["unclear"] = 0.9
        scores = EmotionScores.from_dict(d)
        assert scores.unclear == 0.9

    def test_dominant_emotion_unclear(self):
        """Dominant emotion can be UNCLEAR when it has the highest score."""
        scores = EmotionScores(unclear=0.9)
        assert scores.dominant_emotion == EmotionLabel.UNCLEAR

    def test_dominant_emotion(self):
        """Test getting the dominant emotion."""
        scores = EmotionScores(happy=0.1, sad=0.8, neutral=0.1)
        assert scores.dominant_emotion == EmotionLabel.SAD

    def test_dominant_emotion_tie(self):
        """Test dominant emotion when tied (returns consistently)."""
        scores = EmotionScores(happy=0.5, sad=0.5)
        dominant = scores.dominant_emotion
        assert dominant in (EmotionLabel.HAPPY, EmotionLabel.SAD)


class TestEmotionLabel:
    """Tests for EmotionLabel enum."""

    def test_all_labels_exist(self):
        """Test all expected emotion labels exist."""
        expected = [
            "happy", "sad", "angry", "fearful",
            "surprised", "disgusted", "neutral", "unclear",
        ]
        for label in expected:
            assert EmotionLabel(label) is not None

    def test_unclear_label(self):
        """Test that the unclear label is defined and accessible."""
        assert EmotionLabel.UNCLEAR.value == "unclear"
        assert EmotionLabel("unclear") == EmotionLabel.UNCLEAR

    def test_string_value(self):
        """Test enum string values."""
        assert EmotionLabel.HAPPY.value == "happy"
        assert EmotionLabel.SAD.value == "sad"


class TestActionCommand:
    """Tests for ActionCommand dataclass."""

    def test_creation(self):
        """Test creating an action command."""
        action = ActionCommand(
            action_type="greeting",
            parameters={"gesture": "wave"},
            confidence=0.9,
        )
        assert action.action_type == "greeting"
        assert action.parameters == {"gesture": "wave"}
        assert action.confidence == 0.9

    def test_default_values(self):
        """Test default values."""
        action = ActionCommand(action_type="idle")
        assert action.parameters == {}
        assert action.confidence == 0.0
        assert action.raw_output is None

    def test_stub_creation(self):
        """Test creating stub action from emotion result."""
        scores = EmotionScores(happy=0.9)
        emotion_result = EmotionResult(
            timestamp=0.0,
            emotions=scores,
            fusion_confidence=0.9,
        )
        action = ActionCommand.stub(emotion_result)
        assert action.action_type == "stub"
        assert action.parameters["emotion"] == "happy"
        assert "emotion_scores" in action.parameters


class TestDetectionResult:
    """Tests for DetectionResult dataclass."""

    def test_creation(self):
        """Test creating detection result."""
        result = DetectionResult(timestamp=1.5)
        assert result.timestamp == 1.5
        assert result.faces == []
        assert result.voice is None
        assert result.frame is None

    def test_with_faces(self):
        """Test detection result with faces."""
        bbox = BoundingBox(x=0, y=0, width=100, height=100)
        face = FaceDetection(bbox=bbox, confidence=0.9)
        result = DetectionResult(timestamp=1.5, faces=[face])
        assert len(result.faces) == 1


class TestPipelineResult:
    """Tests for PipelineResult dataclass."""

    def test_to_dict(self):
        """Test converting pipeline result to dictionary."""
        scores = EmotionScores(happy=0.8, neutral=0.2)
        emotion = EmotionResult(timestamp=1.0, emotions=scores, fusion_confidence=0.85)
        detection = DetectionResult(timestamp=1.0)
        action = ActionCommand(action_type="acknowledge", confidence=0.9)

        result = PipelineResult(
            timestamp=1.0,
            detection=detection,
            emotion=emotion,
            action=action,
        )

        d = result.to_dict()
        assert d["timestamp"] == 1.0
        assert d["emotions"]["happy"] == 0.8
        assert d["dominant_emotion"] == "happy"
        assert d["action"]["type"] == "acknowledge"


# ---------------------------------------------------------------------------
# NeuralEmotionResult tests (primary output contract)
# ---------------------------------------------------------------------------

_EMOTION_KEYS = [
    "angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised", "unclear",
]


def _make_neural_result(**overrides) -> NeuralEmotionResult:
    """Build a minimal valid NeuralEmotionResult."""
    base = dict(
        dominant_emotion="happy",
        emotion_scores={k: (0.8 if k == "happy" else 0.02) for k in _EMOTION_KEYS},
        latent_embedding=[0.1] * 512,
        metrics={"stress": 0.2, "engagement": 0.7, "arousal": 0.4},
        confidence=0.8,
        timestamp=1234567890.0,
    )
    base.update(overrides)
    return NeuralEmotionResult(**base)


class TestNeuralEmotionResult:
    """Tests for the primary Pydantic output contract."""

    def test_creation(self):
        result = _make_neural_result()
        assert result.dominant_emotion == "happy"
        assert result.confidence == 0.8

    def test_emotion_scores_has_all_8_classes(self):
        result = _make_neural_result()
        assert set(result.emotion_scores.keys()) == set(_EMOTION_KEYS)

    def test_latent_embedding_length(self):
        result = _make_neural_result()
        assert len(result.latent_embedding) == 512

    def test_metrics_keys(self):
        result = _make_neural_result()
        assert set(result.metrics.keys()) == {"stress", "engagement", "arousal"}

    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            _make_neural_result(confidence=1.5)
        with pytest.raises(Exception):
            _make_neural_result(confidence=-0.1)

    def test_default_missing_flags(self):
        result = _make_neural_result()
        assert result.video_missing is False
        assert result.audio_missing is False

    def test_missing_flags_set(self):
        result = _make_neural_result(video_missing=True, audio_missing=True)
        assert result.video_missing is True
        assert result.audio_missing is True

    def test_unclear_dominant_emotion(self):
        result = _make_neural_result(
            dominant_emotion="unclear",
            emotion_scores={k: (0.9 if k == "unclear" else 0.01) for k in _EMOTION_KEYS},
            confidence=0.9,
        )
        assert result.dominant_emotion == "unclear"

    def test_frozen_model(self):
        """NeuralEmotionResult is immutable (Pydantic frozen=True)."""
        result = _make_neural_result()
        with pytest.raises(Exception):
            result.dominant_emotion = "sad"  # type: ignore[misc]

    def test_json_serialisable(self):
        """All fields should be JSON-serialisable via model_dump."""
        result = _make_neural_result()
        d = result.model_dump()
        assert isinstance(d, dict)
        assert d["dominant_emotion"] == "happy"
        assert isinstance(d["latent_embedding"], list)

