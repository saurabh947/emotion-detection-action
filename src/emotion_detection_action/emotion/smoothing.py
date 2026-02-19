"""Temporal emotion smoothing module.

Provides smoothing algorithms to reduce flickering in emotion predictions
by considering temporal context across multiple frames.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from emotion_detection_action.core.types import EmotionLabel, EmotionResult, EmotionScores


@dataclass
class SmoothingConfig:
    """Configuration for emotion smoothing."""

    # Smoothing strategy
    strategy: Literal["none", "rolling", "ema", "hysteresis"] = "ema"

    # Rolling average settings
    window_size: int = 5  # Number of frames for rolling average

    # Exponential Moving Average settings
    ema_alpha: float = 0.3  # Smoothing factor (0-1, lower = smoother)

    # Hysteresis settings
    hysteresis_threshold: float = 0.15  # Min difference to trigger change
    hysteresis_frames: int = 3  # Frames emotion must persist before change

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.window_size < 1:
            raise ValueError("window_size must be >= 1")
        if not 0 < self.ema_alpha <= 1:
            raise ValueError("ema_alpha must be in (0, 1]")
        if not 0 <= self.hysteresis_threshold <= 1:
            raise ValueError("hysteresis_threshold must be in [0, 1]")
        if self.hysteresis_frames < 1:
            raise ValueError("hysteresis_frames must be >= 1")


class EmotionSmoother:
    """Smooths emotion predictions over time to reduce flickering.

    Supports multiple smoothing strategies:
    - none: No smoothing (pass-through)
    - rolling: Rolling average over a window of frames
    - ema: Exponential Moving Average (default, good balance)
    - hysteresis: Requires sustained emotion change before switching

    Example:
        >>> smoother = EmotionSmoother(SmoothingConfig(strategy="ema", ema_alpha=0.3))
        >>> for result in emotion_results:
        ...     smoothed = smoother.smooth(result)
        ...     print(smoothed.dominant_emotion)
    """

    def __init__(self, config: SmoothingConfig | None = None) -> None:
        """Initialize the emotion smoother.

        Args:
            config: Smoothing configuration. Uses defaults if None.
        """
        self.config = config or SmoothingConfig()

        # Rolling average history
        self._history: deque[EmotionScores] = deque(maxlen=self.config.window_size)

        # EMA state
        self._ema_scores: EmotionScores | None = None

        # Hysteresis state
        self._current_emotion: EmotionLabel | None = None
        self._candidate_emotion: EmotionLabel | None = None
        self._candidate_count: int = 0
        self._last_scores: EmotionScores | None = None

    def smooth(self, result: EmotionResult) -> EmotionResult:
        """Apply smoothing to an emotion result.

        Args:
            result: Raw emotion result from the pipeline.

        Returns:
            Smoothed emotion result.
        """
        if self.config.strategy == "none":
            return result

        if self.config.strategy == "rolling":
            smoothed_scores = self._smooth_rolling(result.emotions)
        elif self.config.strategy == "ema":
            smoothed_scores = self._smooth_ema(result.emotions)
        elif self.config.strategy == "hysteresis":
            smoothed_scores = self._smooth_hysteresis(result.emotions)
        else:
            smoothed_scores = result.emotions

        return EmotionResult(
            timestamp=result.timestamp,
            emotions=smoothed_scores,
            facial_result=result.facial_result,
            speech_result=result.speech_result,
            attention_result=result.attention_result,
            fusion_confidence=result.fusion_confidence,
        )

    def _smooth_rolling(self, scores: EmotionScores) -> EmotionScores:
        """Apply rolling average smoothing.

        Args:
            scores: Current emotion scores.

        Returns:
            Smoothed emotion scores.
        """
        self._history.append(scores)

        if len(self._history) == 1:
            return scores

        # Average across all scores in history
        avg_dict: dict[str, float] = {
            "happy": 0.0,
            "sad": 0.0,
            "angry": 0.0,
            "fearful": 0.0,
            "surprised": 0.0,
            "disgusted": 0.0,
            "neutral": 0.0,
        }

        for hist_scores in self._history:
            hist_dict = hist_scores.to_dict()
            for key in avg_dict:
                avg_dict[key] += hist_dict[key]

        n = len(self._history)
        for key in avg_dict:
            avg_dict[key] /= n

        return EmotionScores.from_dict(avg_dict)

    def _smooth_ema(self, scores: EmotionScores) -> EmotionScores:
        """Apply Exponential Moving Average smoothing.

        EMA formula: smoothed = alpha * current + (1 - alpha) * previous

        Args:
            scores: Current emotion scores.

        Returns:
            Smoothed emotion scores.
        """
        if self._ema_scores is None:
            self._ema_scores = scores
            return scores

        alpha = self.config.ema_alpha
        current_dict = scores.to_dict()
        prev_dict = self._ema_scores.to_dict()

        smoothed_dict: dict[str, float] = {}
        for key in current_dict:
            smoothed_dict[key] = alpha * current_dict[key] + (1 - alpha) * prev_dict[key]

        self._ema_scores = EmotionScores.from_dict(smoothed_dict)
        return self._ema_scores

    def _smooth_hysteresis(self, scores: EmotionScores) -> EmotionScores:
        """Apply hysteresis smoothing.

        Only changes the dominant emotion if a new emotion persists for
        multiple frames with sufficient confidence difference.

        Args:
            scores: Current emotion scores.

        Returns:
            Smoothed emotion scores (may return previous scores).
        """
        current_dominant = scores.dominant_emotion

        # Initialize on first call
        if self._current_emotion is None:
            self._current_emotion = current_dominant
            self._last_scores = scores
            return scores

        # Same emotion as current - reset candidate
        if current_dominant == self._current_emotion:
            self._candidate_emotion = None
            self._candidate_count = 0
            self._last_scores = scores
            return scores

        # Different emotion - check threshold
        current_dict = scores.to_dict()
        assert self._last_scores is not None
        last_dict = self._last_scores.to_dict()

        current_conf = current_dict[current_dominant.value]
        last_conf = last_dict[self._current_emotion.value]

        # Check if difference exceeds threshold
        if current_conf - last_conf < self.config.hysteresis_threshold:
            # Not enough difference, keep current emotion
            return self._last_scores

        # Track candidate emotion
        if current_dominant == self._candidate_emotion:
            self._candidate_count += 1
        else:
            self._candidate_emotion = current_dominant
            self._candidate_count = 1

        # Check if candidate has persisted long enough
        if self._candidate_count >= self.config.hysteresis_frames:
            # Accept the new emotion
            self._current_emotion = current_dominant
            self._candidate_emotion = None
            self._candidate_count = 0
            self._last_scores = scores
            return scores

        # Not enough persistence, return previous scores
        return self._last_scores

    def reset(self) -> None:
        """Reset the smoother state.

        Call this when switching to a new video/stream or when you
        want to clear the history.
        """
        self._history.clear()
        self._ema_scores = None
        self._current_emotion = None
        self._candidate_emotion = None
        self._candidate_count = 0
        self._last_scores = None

    @property
    def strategy(self) -> str:
        """Get the current smoothing strategy."""
        return self.config.strategy

    def get_history_length(self) -> int:
        """Get the current history length (for rolling average)."""
        return len(self._history)


class MultiPersonSmoother:
    """Manages smoothing for multiple tracked persons.

    When tracking multiple faces, each person needs their own smoother
    to maintain temporal consistency.

    Example:
        >>> multi_smoother = MultiPersonSmoother(config)
        >>> for person_id, result in detected_emotions.items():
        ...     smoothed = multi_smoother.smooth(person_id, result)
    """

    def __init__(
        self,
        config: SmoothingConfig | None = None,
        max_persons: int = 10,
    ) -> None:
        """Initialize multi-person smoother.

        Args:
            config: Smoothing configuration for each person.
            max_persons: Maximum number of persons to track.
        """
        self.config = config or SmoothingConfig()
        self.max_persons = max_persons
        self._smoothers: dict[str | int, EmotionSmoother] = {}
        self._access_order: deque[str | int] = deque()

    def smooth(self, person_id: str | int, result: EmotionResult) -> EmotionResult:
        """Smooth emotion for a specific person.

        Args:
            person_id: Unique identifier for the person.
            result: Emotion result for that person.

        Returns:
            Smoothed emotion result.
        """
        if person_id not in self._smoothers:
            self._add_person(person_id)

        # Update access order (LRU)
        if person_id in self._access_order:
            self._access_order.remove(person_id)
        self._access_order.append(person_id)

        return self._smoothers[person_id].smooth(result)

    def _add_person(self, person_id: str | int) -> None:
        """Add a new person's smoother.

        Args:
            person_id: Unique identifier for the person.
        """
        # Remove oldest if at capacity
        while len(self._smoothers) >= self.max_persons:
            oldest = self._access_order.popleft()
            del self._smoothers[oldest]

        self._smoothers[person_id] = EmotionSmoother(self.config)

    def reset(self, person_id: str | int | None = None) -> None:
        """Reset smoother state.

        Args:
            person_id: Specific person to reset. If None, resets all.
        """
        if person_id is not None:
            if person_id in self._smoothers:
                self._smoothers[person_id].reset()
        else:
            self._smoothers.clear()
            self._access_order.clear()

    def remove_person(self, person_id: str | int) -> None:
        """Remove a person's smoother.

        Args:
            person_id: Person to remove.
        """
        if person_id in self._smoothers:
            del self._smoothers[person_id]
            self._access_order.remove(person_id)

    @property
    def tracked_persons(self) -> list[str | int]:
        """Get list of tracked person IDs."""
        return list(self._smoothers.keys())
