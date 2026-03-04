"""Tests for temporal emotion smoothing."""

import pytest

from emotion_detection_action.core.types import (
    EmotionLabel,
    EmotionResult,
    EmotionScores,
)
from emotion_detection_action.emotion.smoothing import (
    EmotionSmoother,
    MultiPersonSmoother,
    SmoothingConfig,
)


def create_emotion_result(
    happy: float = 0.0,
    sad: float = 0.0,
    angry: float = 0.0,
    neutral: float = 0.0,
    timestamp: float = 0.0,
) -> EmotionResult:
    """Helper to create emotion result."""
    scores = EmotionScores(happy=happy, sad=sad, angry=angry, neutral=neutral)
    return EmotionResult(
        timestamp=timestamp,
        emotions=scores,
        fusion_confidence=0.9,
    )


class TestSmoothingConfig:
    """Tests for SmoothingConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = SmoothingConfig()
        assert config.strategy == "ema"
        assert config.window_size == 5
        assert config.ema_alpha == 0.3
        assert config.hysteresis_threshold == 0.15
        assert config.hysteresis_frames == 3

    def test_custom_config(self):
        """Test custom configuration."""
        config = SmoothingConfig(
            strategy="rolling",
            window_size=10,
            ema_alpha=0.5,
        )
        assert config.strategy == "rolling"
        assert config.window_size == 10
        assert config.ema_alpha == 0.5

    def test_invalid_window_size(self):
        """Test validation of window_size."""
        with pytest.raises(ValueError, match="window_size must be >= 1"):
            SmoothingConfig(window_size=0)

    def test_invalid_ema_alpha_low(self):
        """Test validation of ema_alpha (too low)."""
        with pytest.raises(ValueError, match="ema_alpha must be in"):
            SmoothingConfig(ema_alpha=0.0)

    def test_invalid_ema_alpha_high(self):
        """Test validation of ema_alpha (too high)."""
        with pytest.raises(ValueError, match="ema_alpha must be in"):
            SmoothingConfig(ema_alpha=1.5)

    def test_invalid_hysteresis_threshold(self):
        """Test validation of hysteresis_threshold."""
        with pytest.raises(ValueError, match="hysteresis_threshold must be in"):
            SmoothingConfig(hysteresis_threshold=-0.1)

    def test_invalid_hysteresis_frames(self):
        """Test validation of hysteresis_frames."""
        with pytest.raises(ValueError, match="hysteresis_frames must be >= 1"):
            SmoothingConfig(hysteresis_frames=0)


class TestEmotionSmootherNone:
    """Tests for EmotionSmoother with 'none' strategy."""

    def test_no_smoothing(self):
        """Test that 'none' strategy passes through unchanged."""
        config = SmoothingConfig(strategy="none")
        smoother = EmotionSmoother(config)

        result = create_emotion_result(happy=0.8)
        smoothed = smoother.smooth(result)

        assert smoothed.emotions.happy == 0.8


class TestEmotionSmootherRolling:
    """Tests for EmotionSmoother with 'rolling' strategy."""

    def test_rolling_single_frame(self):
        """Test rolling average with single frame."""
        config = SmoothingConfig(strategy="rolling", window_size=3)
        smoother = EmotionSmoother(config)

        result = create_emotion_result(happy=0.9)
        smoothed = smoother.smooth(result)

        # Single frame should return same value
        assert smoothed.emotions.happy == 0.9

    def test_rolling_multiple_frames(self):
        """Test rolling average with multiple frames."""
        config = SmoothingConfig(strategy="rolling", window_size=3)
        smoother = EmotionSmoother(config)

        # Add three frames
        smoother.smooth(create_emotion_result(happy=0.3))
        smoother.smooth(create_emotion_result(happy=0.6))
        smoothed = smoother.smooth(create_emotion_result(happy=0.9))

        # Average of 0.3, 0.6, 0.9 = 0.6
        assert smoothed.emotions.happy == pytest.approx(0.6, abs=0.01)

    def test_rolling_window_limit(self):
        """Test that rolling window respects size limit."""
        config = SmoothingConfig(strategy="rolling", window_size=2)
        smoother = EmotionSmoother(config)

        smoother.smooth(create_emotion_result(happy=0.1))  # Will be dropped
        smoother.smooth(create_emotion_result(happy=0.5))
        smoothed = smoother.smooth(create_emotion_result(happy=0.9))

        # Average of 0.5 and 0.9 = 0.7 (0.1 should be dropped)
        assert smoothed.emotions.happy == pytest.approx(0.7, abs=0.01)


class TestEmotionSmootherEMA:
    """Tests for EmotionSmoother with 'ema' strategy."""

    def test_ema_first_frame(self):
        """Test EMA on first frame returns original."""
        config = SmoothingConfig(strategy="ema", ema_alpha=0.3)
        smoother = EmotionSmoother(config)

        result = create_emotion_result(happy=0.8)
        smoothed = smoother.smooth(result)

        assert smoothed.emotions.happy == 0.8

    def test_ema_smoothing(self):
        """Test EMA smoothing calculation."""
        config = SmoothingConfig(strategy="ema", ema_alpha=0.5)
        smoother = EmotionSmoother(config)

        # First frame
        smoother.smooth(create_emotion_result(happy=1.0))

        # Second frame: 0.5 * 0.0 + 0.5 * 1.0 = 0.5
        smoothed = smoother.smooth(create_emotion_result(happy=0.0))

        assert smoothed.emotions.happy == pytest.approx(0.5, abs=0.01)

    def test_ema_low_alpha_more_smoothing(self):
        """Test that lower alpha gives more smoothing."""
        config_low = SmoothingConfig(strategy="ema", ema_alpha=0.1)
        config_high = SmoothingConfig(strategy="ema", ema_alpha=0.9)

        smoother_low = EmotionSmoother(config_low)
        smoother_high = EmotionSmoother(config_high)

        # Initialize both with happy=0
        smoother_low.smooth(create_emotion_result(happy=0.0))
        smoother_high.smooth(create_emotion_result(happy=0.0))

        # Add happy=1.0
        smoothed_low = smoother_low.smooth(create_emotion_result(happy=1.0))
        smoothed_high = smoother_high.smooth(create_emotion_result(happy=1.0))

        # Low alpha should have lower response (more smoothing)
        assert smoothed_low.emotions.happy < smoothed_high.emotions.happy


class TestEmotionSmootherHysteresis:
    """Tests for EmotionSmoother with 'hysteresis' strategy."""

    def test_hysteresis_same_emotion(self):
        """Test that same emotion updates normally."""
        config = SmoothingConfig(
            strategy="hysteresis",
            hysteresis_threshold=0.15,
            hysteresis_frames=3,
        )
        smoother = EmotionSmoother(config)

        result1 = create_emotion_result(happy=0.8)
        result2 = create_emotion_result(happy=0.9)

        smoother.smooth(result1)
        smoothed = smoother.smooth(result2)

        # Same dominant emotion, should update
        assert smoothed.emotions.happy == 0.9

    def test_hysteresis_prevents_quick_change(self):
        """Test that hysteresis prevents rapid emotion changes."""
        config = SmoothingConfig(
            strategy="hysteresis",
            hysteresis_threshold=0.15,
            hysteresis_frames=3,
        )
        smoother = EmotionSmoother(config)

        # Start with happy
        smoother.smooth(create_emotion_result(happy=0.8, sad=0.2))

        # Try to switch to sad (only 1 frame)
        smoothed = smoother.smooth(create_emotion_result(sad=0.8, happy=0.2))

        # Should still show happy (hasn't persisted)
        assert smoothed.emotions.happy == 0.8

    def test_hysteresis_allows_sustained_change(self):
        """Test that hysteresis allows a sustained emotion change.

        After seeing the new emotion for at least hysteresis_frames consecutive
        frames the smoother should commit to the new label.
        """
        config = SmoothingConfig(
            strategy="hysteresis",
            hysteresis_threshold=0.15,
            hysteresis_frames=3,
        )
        smoother = EmotionSmoother(config)

        # Establish happy as the current emotion
        smoother.smooth(create_emotion_result(happy=0.8, sad=0.2))

        # Drive sad for hysteresis_frames + 1 frames to guarantee a committed switch
        smoothed = None
        for _ in range(config.hysteresis_frames + 1):
            smoothed = smoother.smooth(create_emotion_result(sad=0.8, happy=0.2))

        assert smoothed is not None
        assert smoothed.dominant_emotion == EmotionLabel.SAD

    def test_hysteresis_threshold(self):
        """Test that small differences don't trigger change."""
        config = SmoothingConfig(
            strategy="hysteresis",
            hysteresis_threshold=0.3,  # High threshold
            hysteresis_frames=1,
        )
        smoother = EmotionSmoother(config)

        # Start with happy
        smoother.smooth(create_emotion_result(happy=0.5, sad=0.3))

        # Try to switch to sad (difference < threshold)
        smoothed = smoother.smooth(create_emotion_result(sad=0.5, happy=0.3))

        # Should NOT switch due to threshold
        assert smoothed.emotions.happy == 0.5


class TestEmotionSmootherReset:
    """Tests for EmotionSmoother reset functionality."""

    def test_reset_clears_history(self):
        """Test that reset clears rolling history."""
        config = SmoothingConfig(strategy="rolling", window_size=3)
        smoother = EmotionSmoother(config)

        smoother.smooth(create_emotion_result(happy=0.3))
        smoother.smooth(create_emotion_result(happy=0.6))

        assert smoother.get_history_length() == 2

        smoother.reset()

        assert smoother.get_history_length() == 0

    def test_reset_clears_ema(self):
        """Test that reset clears EMA state."""
        config = SmoothingConfig(strategy="ema", ema_alpha=0.5)
        smoother = EmotionSmoother(config)

        smoother.smooth(create_emotion_result(happy=0.0))
        smoother.smooth(create_emotion_result(happy=1.0))

        smoother.reset()

        # After reset, first frame should be unchanged
        smoothed = smoother.smooth(create_emotion_result(happy=0.5))
        assert smoothed.emotions.happy == 0.5


class TestMultiPersonSmoother:
    """Tests for MultiPersonSmoother."""

    def test_independent_smoothing(self):
        """Test that each person is smoothed independently."""
        config = SmoothingConfig(strategy="ema", ema_alpha=0.5)
        multi = MultiPersonSmoother(config)

        # Person A starts with happy
        multi.smooth("person_a", create_emotion_result(happy=1.0))

        # Person B starts with sad
        multi.smooth("person_b", create_emotion_result(sad=1.0))

        # Person A now sad - should be smoothed
        smoothed_a = multi.smooth("person_a", create_emotion_result(sad=1.0))

        # Person B now happy - should be smoothed
        smoothed_b = multi.smooth("person_b", create_emotion_result(happy=1.0))

        # Both should have smoothed values, not raw
        assert smoothed_a.emotions.happy > 0  # Some happy left from EMA
        assert smoothed_b.emotions.sad > 0    # Some sad left from EMA

    def test_max_persons_limit(self):
        """Test that old persons are removed when at capacity."""
        config = SmoothingConfig(strategy="none")
        multi = MultiPersonSmoother(config, max_persons=2)

        multi.smooth("a", create_emotion_result(happy=0.9))
        multi.smooth("b", create_emotion_result(sad=0.9))
        multi.smooth("c", create_emotion_result(angry=0.9))  # Should evict "a"

        assert "c" in multi.tracked_persons
        assert "b" in multi.tracked_persons
        assert "a" not in multi.tracked_persons

    def test_lru_eviction(self):
        """Test that least recently used person is evicted."""
        config = SmoothingConfig(strategy="none")
        multi = MultiPersonSmoother(config, max_persons=2)

        multi.smooth("a", create_emotion_result(happy=0.9))
        multi.smooth("b", create_emotion_result(sad=0.9))

        # Access "a" again to make it recent
        multi.smooth("a", create_emotion_result(happy=0.8))

        # Add "c" - should evict "b" (least recently used)
        multi.smooth("c", create_emotion_result(angry=0.9))

        assert "a" in multi.tracked_persons
        assert "c" in multi.tracked_persons
        assert "b" not in multi.tracked_persons

    def test_reset_single_person(self):
        """Test resetting a single person."""
        config = SmoothingConfig(strategy="rolling", window_size=5)
        multi = MultiPersonSmoother(config)

        multi.smooth("a", create_emotion_result(happy=0.9))
        multi.smooth("b", create_emotion_result(sad=0.9))

        multi.reset("a")

        assert "a" in multi.tracked_persons  # Still tracked
        assert "b" in multi.tracked_persons

    def test_reset_all(self):
        """Test resetting all persons."""
        config = SmoothingConfig(strategy="none")
        multi = MultiPersonSmoother(config)

        multi.smooth("a", create_emotion_result(happy=0.9))
        multi.smooth("b", create_emotion_result(sad=0.9))

        multi.reset()

        assert len(multi.tracked_persons) == 0

    def test_remove_person(self):
        """Test removing a specific person."""
        config = SmoothingConfig(strategy="none")
        multi = MultiPersonSmoother(config)

        multi.smooth("a", create_emotion_result(happy=0.9))
        multi.smooth("b", create_emotion_result(sad=0.9))

        multi.remove_person("a")

        assert "a" not in multi.tracked_persons
        assert "b" in multi.tracked_persons


class TestSmootherProperties:
    """Tests for smoother properties."""

    def test_strategy_property(self):
        """Test strategy property."""
        config = SmoothingConfig(strategy="rolling")
        smoother = EmotionSmoother(config)

        assert smoother.strategy == "rolling"

    def test_history_length_rolling(self):
        """Test history length for rolling strategy."""
        config = SmoothingConfig(strategy="rolling", window_size=10)
        smoother = EmotionSmoother(config)

        for i in range(5):
            smoother.smooth(create_emotion_result(happy=0.1 * i))

        assert smoother.get_history_length() == 5

    def test_history_length_non_rolling(self):
        """Test history length for non-rolling strategy."""
        config = SmoothingConfig(strategy="ema")
        smoother = EmotionSmoother(config)

        smoother.smooth(create_emotion_result(happy=0.5))

        # EMA doesn't use history deque
        assert smoother.get_history_length() == 0
