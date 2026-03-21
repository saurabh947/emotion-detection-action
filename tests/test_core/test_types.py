"""Tests for core type definitions."""

import pytest

from emotion_detection_action.core.types import (
    ActionCommand,
    EmotionLabel,
    NeuralEmotionResult,
)

_EMOTION_KEYS = [
    "angry", "disgusted", "fearful", "happy",
    "neutral", "sad", "surprised", "unclear",
]


def _make_result(**overrides) -> NeuralEmotionResult:
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


class TestEmotionLabel:
    """Tests for EmotionLabel enum."""

    def test_all_8_labels_exist(self):
        for label in _EMOTION_KEYS:
            assert EmotionLabel(label) is not None

    def test_unclear_value(self):
        assert EmotionLabel.UNCLEAR.value == "unclear"
        assert EmotionLabel("unclear") == EmotionLabel.UNCLEAR

    def test_str_enum_equality(self):
        """EmotionLabel is a str subclass — compares equal to its value string."""
        assert EmotionLabel.HAPPY == "happy"
        assert EmotionLabel.SAD == "sad"

    def test_value_attribute(self):
        assert EmotionLabel.HAPPY.value == "happy"
        assert EmotionLabel.ANGRY.value == "angry"


class TestActionCommand:
    """Tests for ActionCommand dataclass."""

    def test_creation(self):
        action = ActionCommand(
            action_type="greeting",
            parameters={"gesture": "wave"},
            confidence=0.9,
        )
        assert action.action_type == "greeting"
        assert action.parameters == {"gesture": "wave"}
        assert action.confidence == 0.9

    def test_default_values(self):
        action = ActionCommand(action_type="idle")
        assert action.parameters == {}
        assert action.confidence == 0.0
        assert action.raw_output is None


class TestNeuralEmotionResult:
    """Tests for the primary Pydantic output contract."""

    def test_creation(self):
        result = _make_result()
        assert result.dominant_emotion == "happy"
        assert result.confidence == 0.8

    def test_emotion_scores_has_all_8_classes(self):
        result = _make_result()
        assert set(result.emotion_scores.keys()) == set(_EMOTION_KEYS)

    def test_latent_embedding_length(self):
        result = _make_result()
        assert len(result.latent_embedding) == 512

    def test_metrics_keys(self):
        result = _make_result()
        assert set(result.metrics.keys()) == {"stress", "engagement", "arousal"}

    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            _make_result(confidence=1.5)
        with pytest.raises(Exception):
            _make_result(confidence=-0.1)

    def test_default_missing_flags(self):
        result = _make_result()
        assert result.video_missing is False
        assert result.audio_missing is False

    def test_missing_flags_set(self):
        result = _make_result(video_missing=True, audio_missing=True)
        assert result.video_missing is True
        assert result.audio_missing is True

    def test_unclear_dominant_emotion(self):
        result = _make_result(
            dominant_emotion="unclear",
            emotion_scores={k: (0.9 if k == "unclear" else 0.01) for k in _EMOTION_KEYS},
            confidence=0.9,
        )
        assert result.dominant_emotion == "unclear"

    def test_frozen_immutability(self):
        """NeuralEmotionResult is immutable (Pydantic frozen=True)."""
        result = _make_result()
        with pytest.raises(Exception):
            result.dominant_emotion = "sad"  # type: ignore[misc]

    def test_json_serialisable(self):
        result = _make_result()
        d = result.model_dump()
        assert isinstance(d, dict)
        assert d["dominant_emotion"] == "happy"
        assert isinstance(d["latent_embedding"], list)
        assert len(d["latent_embedding"]) == 512

    def test_timestamp_defaults_to_zero(self):
        result = NeuralEmotionResult(
            dominant_emotion="neutral",
            emotion_scores={k: 0.125 for k in _EMOTION_KEYS},
            latent_embedding=[0.0] * 512,
            metrics={"stress": 0.0, "engagement": 0.0, "arousal": 0.0},
            confidence=0.125,
        )
        assert result.timestamp == 0.0
