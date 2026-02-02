"""Tests for attention analysis module."""

import pytest

from emotion_detection_action.core.types import (
    EyeDetection,
    GazeDetection,
)
from emotion_detection_action.emotion.attention import (
    AttentionAnalyzer,
    AttentionAnalyzerConfig,
)


class TestAttentionAnalyzerConfig:
    """Tests for AttentionAnalyzerConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = AttentionAnalyzerConfig()

        assert config.pupil_dilation_weight == 0.6
        assert config.blink_rate_weight == 0.4
        assert config.normal_blink_rate == 15.0
        assert config.high_blink_rate == 25.0
        assert config.eye_contact_threshold == 0.3

    def test_custom_values(self) -> None:
        """Test custom configuration."""
        config = AttentionAnalyzerConfig(
            pupil_dilation_weight=0.7,
            blink_rate_weight=0.3,
            normal_blink_rate=20.0,
        )
        assert config.pupil_dilation_weight == 0.7
        assert config.blink_rate_weight == 0.3
        assert config.normal_blink_rate == 20.0


class TestAttentionAnalyzer:
    """Tests for AttentionAnalyzer class."""

    def test_initialization(self) -> None:
        """Test analyzer initialization."""
        analyzer = AttentionAnalyzer()
        assert analyzer.config is not None

    def test_initialization_custom_config(self) -> None:
        """Test analyzer with custom config."""
        config = AttentionAnalyzerConfig(history_size=50)
        analyzer = AttentionAnalyzer(config=config)
        assert analyzer.config.history_size == 50

    def test_analyze_no_gaze(self) -> None:
        """Test analysis with no gaze detection."""
        analyzer = AttentionAnalyzer()
        result = analyzer.analyze(
            gaze=None,
            blink_rate=15.0,
            pupil_dilation=0.0,
            gaze_stability=1.0,
            timestamp=0.0,
        )

        assert result.timestamp == 0.0
        assert result.confidence == 0.0
        assert result.gaze is None

    def test_analyze_with_gaze(self) -> None:
        """Test analysis with gaze detection."""
        analyzer = AttentionAnalyzer()

        left_eye = EyeDetection(center=(100.0, 150.0), pupil_size=0.4)
        right_eye = EyeDetection(center=(200.0, 150.0), pupil_size=0.4)
        gaze = GazeDetection(
            left_eye=left_eye,
            right_eye=right_eye,
            gaze_direction=(0.0, 0.0),  # Looking at camera
            confidence=0.9,
        )

        result = analyzer.analyze(
            gaze=gaze,
            blink_rate=15.0,
            pupil_dilation=0.0,
            gaze_stability=0.9,
            timestamp=1.0,
        )

        assert result.timestamp == 1.0
        assert result.confidence == 0.9
        assert result.gaze is not None

    def test_stress_score_low(self) -> None:
        """Test stress score with normal conditions."""
        analyzer = AttentionAnalyzer()

        gaze = GazeDetection(gaze_direction=(0.0, 0.0), confidence=0.9)

        result = analyzer.analyze(
            gaze=gaze,
            blink_rate=15.0,  # Normal blink rate
            pupil_dilation=0.0,  # No dilation
            gaze_stability=0.9,
            timestamp=0.0,
        )

        # Should have low stress
        assert result.metrics.stress_score < 0.3

    def test_stress_score_high(self) -> None:
        """Test stress score with stressed conditions."""
        analyzer = AttentionAnalyzer()

        gaze = GazeDetection(gaze_direction=(0.0, 0.0), confidence=0.9)

        result = analyzer.analyze(
            gaze=gaze,
            blink_rate=30.0,  # High blink rate
            pupil_dilation=0.3,  # Dilated pupils
            gaze_stability=0.5,
            timestamp=0.0,
        )

        # Should have higher stress
        assert result.metrics.stress_score > 0.5

    def test_engagement_score_high(self) -> None:
        """Test engagement score with engaged user."""
        analyzer = AttentionAnalyzer()

        # Looking directly at camera
        gaze = GazeDetection(gaze_direction=(0.0, 0.0), confidence=0.9)

        result = analyzer.analyze(
            gaze=gaze,
            blink_rate=15.0,
            pupil_dilation=0.0,
            gaze_stability=0.95,  # Very stable gaze
            timestamp=0.0,
        )

        # Should have high engagement
        assert result.metrics.engagement_score > 0.7

    def test_engagement_score_low(self) -> None:
        """Test engagement score with distracted user."""
        analyzer = AttentionAnalyzer()

        # Looking away from camera
        gaze = GazeDetection(gaze_direction=(0.8, 0.5), confidence=0.9)

        result = analyzer.analyze(
            gaze=gaze,
            blink_rate=15.0,
            pupil_dilation=0.0,
            gaze_stability=0.3,  # Unstable gaze
            timestamp=0.0,
        )

        # Should have lower engagement
        assert result.metrics.engagement_score < 0.5

    def test_nervousness_score(self) -> None:
        """Test nervousness score calculation."""
        analyzer = AttentionAnalyzer()

        # Averted gaze, unstable
        gaze = GazeDetection(gaze_direction=(0.6, 0.3), confidence=0.9)

        result = analyzer.analyze(
            gaze=gaze,
            blink_rate=25.0,  # Elevated blink rate
            pupil_dilation=0.0,
            gaze_stability=0.3,  # Unstable
            timestamp=0.0,
        )

        # Should have nervousness signal
        assert result.metrics.nervousness_score > 0.3

    def test_smoothing_over_time(self) -> None:
        """Test that scores are smoothed over time."""
        analyzer = AttentionAnalyzer()

        gaze = GazeDetection(gaze_direction=(0.0, 0.0), confidence=0.9)

        # Initial reading
        result1 = analyzer.analyze(
            gaze=gaze,
            blink_rate=15.0,
            pupil_dilation=0.0,
            gaze_stability=0.9,
            timestamp=0.0,
        )

        # Sudden spike
        result2 = analyzer.analyze(
            gaze=gaze,
            blink_rate=30.0,  # Sudden high blink
            pupil_dilation=0.5,  # Sudden dilation
            gaze_stability=0.3,
            timestamp=1.0,
        )

        # The smoothed result should be between the two extremes
        # (not as high as the raw spike)
        assert result2.metrics.stress_score > result1.metrics.stress_score
        # Due to EMA smoothing, it shouldn't jump to maximum immediately
        assert result2.metrics.stress_score < 1.0

    def test_reset(self) -> None:
        """Test reset functionality."""
        analyzer = AttentionAnalyzer()

        gaze = GazeDetection(gaze_direction=(0.0, 0.0), confidence=0.9)

        # Add some history
        for _ in range(5):
            analyzer.analyze(
                gaze=gaze,
                blink_rate=15.0,
                pupil_dilation=0.0,
                gaze_stability=0.9,
                timestamp=0.0,
            )

        # Reset
        analyzer.reset()

        # History should be empty
        assert len(analyzer._stress_history) == 0
        assert len(analyzer._engagement_history) == 0

    def test_repr(self) -> None:
        """Test string representation."""
        analyzer = AttentionAnalyzer()
        repr_str = repr(analyzer)
        assert "AttentionAnalyzer" in repr_str
        assert "history_size" in repr_str
