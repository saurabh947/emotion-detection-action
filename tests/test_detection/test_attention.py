"""Tests for attention detection module."""

import pytest

from emotion_detection_action.core.config import ModelConfig
from emotion_detection_action.core.types import (
    AttentionMetrics,
    AttentionResult,
    EyeDetection,
    GazeDetection,
)


class TestEyeDetection:
    """Tests for EyeDetection dataclass."""

    def test_creation(self) -> None:
        """Test basic creation of EyeDetection."""
        eye = EyeDetection(
            center=(100.0, 150.0),
            pupil_size=0.4,
            openness=0.8,
        )
        assert eye.center == (100.0, 150.0)
        assert eye.pupil_size == 0.4
        assert eye.openness == 0.8
        assert eye.landmarks is None

    def test_default_values(self) -> None:
        """Test default values."""
        eye = EyeDetection(center=(0.0, 0.0))
        assert eye.pupil_size == 0.0
        assert eye.openness == 1.0


class TestGazeDetection:
    """Tests for GazeDetection dataclass."""

    def test_creation(self) -> None:
        """Test basic creation of GazeDetection."""
        left_eye = EyeDetection(center=(100.0, 150.0), pupil_size=0.4)
        right_eye = EyeDetection(center=(200.0, 150.0), pupil_size=0.5)

        gaze = GazeDetection(
            left_eye=left_eye,
            right_eye=right_eye,
            gaze_direction=(0.1, -0.2),
            confidence=0.9,
        )
        assert gaze.gaze_direction == (0.1, -0.2)
        assert gaze.confidence == 0.9

    def test_avg_pupil_size(self) -> None:
        """Test average pupil size calculation."""
        left_eye = EyeDetection(center=(100.0, 150.0), pupil_size=0.4)
        right_eye = EyeDetection(center=(200.0, 150.0), pupil_size=0.6)

        gaze = GazeDetection(left_eye=left_eye, right_eye=right_eye)
        assert gaze.avg_pupil_size == pytest.approx(0.5)

    def test_avg_pupil_size_one_eye(self) -> None:
        """Test average pupil size with one eye."""
        left_eye = EyeDetection(center=(100.0, 150.0), pupil_size=0.4)

        gaze = GazeDetection(left_eye=left_eye, right_eye=None)
        assert gaze.avg_pupil_size == pytest.approx(0.4)

    def test_avg_pupil_size_no_eyes(self) -> None:
        """Test average pupil size with no eyes."""
        gaze = GazeDetection()
        assert gaze.avg_pupil_size == 0.0

    def test_avg_eye_openness(self) -> None:
        """Test average eye openness calculation."""
        left_eye = EyeDetection(center=(100.0, 150.0), openness=0.8)
        right_eye = EyeDetection(center=(200.0, 150.0), openness=0.6)

        gaze = GazeDetection(left_eye=left_eye, right_eye=right_eye)
        assert gaze.avg_eye_openness == pytest.approx(0.7)


class TestAttentionMetrics:
    """Tests for AttentionMetrics dataclass."""

    def test_creation(self) -> None:
        """Test basic creation."""
        metrics = AttentionMetrics(
            pupil_dilation=0.1,
            gaze_stability=0.9,
            blink_rate=15.0,
            eye_contact_ratio=0.8,
            stress_score=0.3,
            engagement_score=0.7,
            nervousness_score=0.2,
        )
        assert metrics.stress_score == 0.3
        assert metrics.engagement_score == 0.7

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        metrics = AttentionMetrics(
            pupil_dilation=0.1,
            stress_score=0.5,
            engagement_score=0.8,
        )
        d = metrics.to_dict()
        assert d["pupil_dilation"] == 0.1
        assert d["stress_score"] == 0.5
        assert d["engagement_score"] == 0.8
        assert "nervousness_score" in d

    def test_default_values(self) -> None:
        """Test default values."""
        metrics = AttentionMetrics()
        assert metrics.pupil_dilation == 0.0
        assert metrics.gaze_stability == 1.0
        assert metrics.blink_rate == 0.0
        assert metrics.stress_score == 0.0


class TestAttentionResult:
    """Tests for AttentionResult dataclass."""

    def test_creation(self) -> None:
        """Test basic creation."""
        metrics = AttentionMetrics(stress_score=0.5, engagement_score=0.7)
        result = AttentionResult(
            timestamp=1.0,
            metrics=metrics,
            confidence=0.9,
        )
        assert result.timestamp == 1.0
        assert result.confidence == 0.9
        assert result.stress_score == 0.5
        assert result.engagement_score == 0.7

    def test_shortcuts(self) -> None:
        """Test property shortcuts."""
        metrics = AttentionMetrics(
            stress_score=0.3,
            engagement_score=0.8,
            nervousness_score=0.2,
        )
        result = AttentionResult(timestamp=0.0, metrics=metrics)
        
        assert result.stress_score == 0.3
        assert result.engagement_score == 0.8
        assert result.nervousness_score == 0.2

    def test_default_metrics(self) -> None:
        """Test that default metrics are created."""
        result = AttentionResult(timestamp=0.0)
        assert result.metrics is not None
        assert result.stress_score == 0.0


class TestAttentionDetector:
    """Tests for AttentionDetector class."""

    def test_initialization(self) -> None:
        """Test detector initialization."""
        from emotion_detection_action.detection.attention import AttentionDetector

        config = ModelConfig(model_id="mediapipe", device="cpu")
        detector = AttentionDetector(config, history_size=30)
        
        assert not detector.is_loaded
        assert detector._history_size == 30

    def test_requires_load(self) -> None:
        """Test that predict requires load()."""
        from emotion_detection_action.detection.attention import AttentionDetector
        import numpy as np

        config = ModelConfig(model_id="mediapipe", device="cpu")
        detector = AttentionDetector(config)
        
        with pytest.raises(RuntimeError, match="not loaded"):
            detector.predict(np.zeros((100, 100, 3), dtype=np.uint8))

    def test_get_blink_rate_empty(self) -> None:
        """Test blink rate with no blinks."""
        from emotion_detection_action.detection.attention import AttentionDetector

        config = ModelConfig(model_id="mediapipe", device="cpu")
        detector = AttentionDetector(config)
        
        assert detector.get_blink_rate() == 0.0

    def test_get_pupil_dilation_no_baseline(self) -> None:
        """Test pupil dilation with no baseline."""
        from emotion_detection_action.detection.attention import AttentionDetector

        config = ModelConfig(model_id="mediapipe", device="cpu")
        detector = AttentionDetector(config)
        
        assert detector.get_pupil_dilation() == 0.0

    def test_get_gaze_stability_empty(self) -> None:
        """Test gaze stability with no history."""
        from emotion_detection_action.detection.attention import AttentionDetector

        config = ModelConfig(model_id="mediapipe", device="cpu")
        detector = AttentionDetector(config)
        
        assert detector.get_gaze_stability() == 1.0

    def test_repr(self) -> None:
        """Test string representation."""
        from emotion_detection_action.detection.attention import AttentionDetector

        config = ModelConfig(model_id="mediapipe", device="cpu")
        detector = AttentionDetector(config)
        
        repr_str = repr(detector)
        assert "AttentionDetector" in repr_str
        assert "loaded=False" in repr_str
