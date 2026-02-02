"""Attention analysis module for computing stress, engagement, and nervousness scores."""

from collections import deque
from dataclasses import dataclass

import numpy as np

from emotion_detection_action.core.types import (
    AttentionMetrics,
    AttentionResult,
    GazeDetection,
)
from emotion_detection_action.detection.attention import AttentionDetector


@dataclass
class AttentionAnalyzerConfig:
    """Configuration for attention analysis."""

    # Stress score weights
    pupil_dilation_weight: float = 0.6
    blink_rate_weight: float = 0.4

    # Engagement score weights
    eye_contact_weight: float = 0.5
    fixation_weight: float = 0.5

    # Nervousness score weights
    gaze_aversion_weight: float = 0.4
    instability_weight: float = 0.4
    blink_weight: float = 0.2

    # Thresholds
    normal_blink_rate: float = 15.0  # Normal is 15-20 blinks/min
    high_blink_rate: float = 25.0  # Above this indicates stress
    eye_contact_threshold: float = 0.3  # Gaze within this range = eye contact

    # Smoothing
    history_size: int = 30  # Frames to keep for smoothing


class AttentionAnalyzer:
    """Analyzes attention metrics to compute stress, engagement, and nervousness.

    Takes raw gaze detection data from AttentionDetector and computes
    higher-level psychological indicators.

    Example:
        >>> analyzer = AttentionAnalyzer()
        >>> result = analyzer.analyze(gaze_detection, blink_rate, pupil_dilation, timestamp)
        >>> print(f"Stress: {result.stress_score:.2f}")
    """

    def __init__(
        self,
        config: AttentionAnalyzerConfig | None = None,
        detector: AttentionDetector | None = None,
    ) -> None:
        """Initialize attention analyzer.

        Args:
            config: Analysis configuration.
            detector: Optional reference to attention detector for metrics.
        """
        self.config = config or AttentionAnalyzerConfig()
        self._detector = detector

        # History for smoothing
        self._stress_history: deque = deque(maxlen=self.config.history_size)
        self._engagement_history: deque = deque(maxlen=self.config.history_size)
        self._nervousness_history: deque = deque(maxlen=self.config.history_size)
        self._eye_contact_history: deque = deque(maxlen=self.config.history_size)

    def analyze(
        self,
        gaze: GazeDetection | None,
        blink_rate: float,
        pupil_dilation: float,
        gaze_stability: float,
        timestamp: float,
    ) -> AttentionResult:
        """Analyze attention metrics and compute scores.

        Args:
            gaze: Gaze detection result.
            blink_rate: Current blinks per minute.
            pupil_dilation: Pupil dilation relative to baseline.
            gaze_stability: Gaze stability score (0-1).
            timestamp: Current timestamp.

        Returns:
            AttentionResult with computed metrics.
        """
        # Calculate eye contact ratio
        eye_contact = self._calculate_eye_contact(gaze)
        self._eye_contact_history.append(eye_contact)

        # Calculate raw scores
        stress = self._calculate_stress_score(pupil_dilation, blink_rate)
        engagement = self._calculate_engagement_score(eye_contact, gaze_stability)
        nervousness = self._calculate_nervousness_score(
            eye_contact, gaze_stability, blink_rate
        )

        # Update history
        self._stress_history.append(stress)
        self._engagement_history.append(engagement)
        self._nervousness_history.append(nervousness)

        # Apply smoothing (EMA)
        smoothed_stress = self._smooth_score(self._stress_history)
        smoothed_engagement = self._smooth_score(self._engagement_history)
        smoothed_nervousness = self._smooth_score(self._nervousness_history)

        # Calculate average eye contact ratio over history
        avg_eye_contact = (
            sum(self._eye_contact_history) / len(self._eye_contact_history)
            if self._eye_contact_history
            else 0.0
        )

        metrics = AttentionMetrics(
            pupil_dilation=pupil_dilation,
            gaze_stability=gaze_stability,
            blink_rate=blink_rate,
            eye_contact_ratio=avg_eye_contact,
            stress_score=smoothed_stress,
            engagement_score=smoothed_engagement,
            nervousness_score=smoothed_nervousness,
        )

        confidence = gaze.confidence if gaze else 0.0

        return AttentionResult(
            timestamp=timestamp,
            gaze=gaze,
            metrics=metrics,
            confidence=confidence,
        )

    def _calculate_eye_contact(self, gaze: GazeDetection | None) -> float:
        """Calculate if user is making eye contact (looking at camera).

        Args:
            gaze: Gaze detection result.

        Returns:
            Eye contact score (0-1).
        """
        if gaze is None:
            return 0.0

        gx, gy = gaze.gaze_direction
        threshold = self.config.eye_contact_threshold

        # Eye contact = gaze near center (looking at camera)
        distance_from_center = np.sqrt(gx ** 2 + gy ** 2)

        if distance_from_center < threshold:
            # Full eye contact when very centered
            return 1.0 - (distance_from_center / threshold)
        else:
            # Partial credit for being close
            return max(0.0, 1.0 - distance_from_center)

    def _calculate_stress_score(
        self,
        pupil_dilation: float,
        blink_rate: float,
    ) -> float:
        """Calculate stress score from physiological indicators.

        Higher pupil dilation and higher blink rate indicate stress.

        Args:
            pupil_dilation: Pupil dilation relative to baseline.
            blink_rate: Current blinks per minute.

        Returns:
            Stress score (0-1).
        """
        # Pupil dilation component (positive dilation = stress)
        # Typical dilation under stress is 10-30%
        pupil_score = max(0.0, min(1.0, pupil_dilation / 0.3))

        # Blink rate component
        # Normal: 15-20/min, Stressed: 25+/min
        if blink_rate <= self.config.normal_blink_rate:
            blink_score = 0.0
        elif blink_rate >= self.config.high_blink_rate:
            blink_score = 1.0
        else:
            blink_score = (blink_rate - self.config.normal_blink_rate) / (
                self.config.high_blink_rate - self.config.normal_blink_rate
            )

        # Weighted combination
        stress = (
            pupil_score * self.config.pupil_dilation_weight +
            blink_score * self.config.blink_rate_weight
        )

        return max(0.0, min(1.0, stress))

    def _calculate_engagement_score(
        self,
        eye_contact: float,
        gaze_stability: float,
    ) -> float:
        """Calculate engagement score from attention indicators.

        Higher eye contact and stable gaze indicate engagement.

        Args:
            eye_contact: Current eye contact score.
            gaze_stability: Gaze stability score.

        Returns:
            Engagement score (0-1).
        """
        engagement = (
            eye_contact * self.config.eye_contact_weight +
            gaze_stability * self.config.fixation_weight
        )

        return max(0.0, min(1.0, engagement))

    def _calculate_nervousness_score(
        self,
        eye_contact: float,
        gaze_stability: float,
        blink_rate: float,
    ) -> float:
        """Calculate nervousness score from behavioral indicators.

        Gaze aversion, instability, and high blink rate indicate nervousness.

        Args:
            eye_contact: Current eye contact score.
            gaze_stability: Gaze stability score.
            blink_rate: Current blinks per minute.

        Returns:
            Nervousness score (0-1).
        """
        # Gaze aversion (low eye contact)
        gaze_aversion = 1.0 - eye_contact

        # Instability (low stability)
        instability = 1.0 - gaze_stability

        # Elevated blink rate
        blink_score = max(0.0, min(1.0, blink_rate / self.config.high_blink_rate))

        nervousness = (
            gaze_aversion * self.config.gaze_aversion_weight +
            instability * self.config.instability_weight +
            blink_score * self.config.blink_weight
        )

        return max(0.0, min(1.0, nervousness))

    def _smooth_score(self, history: deque, alpha: float = 0.3) -> float:
        """Apply exponential moving average smoothing.

        Args:
            history: Deque of historical values.
            alpha: Smoothing factor (higher = less smoothing).

        Returns:
            Smoothed value.
        """
        if not history:
            return 0.0

        if len(history) == 1:
            return history[0]

        # EMA smoothing
        smoothed = history[0]
        for value in list(history)[1:]:
            smoothed = alpha * value + (1 - alpha) * smoothed

        return smoothed

    def reset(self) -> None:
        """Reset all history."""
        self._stress_history.clear()
        self._engagement_history.clear()
        self._nervousness_history.clear()
        self._eye_contact_history.clear()

    def __repr__(self) -> str:
        return f"AttentionAnalyzer(history_size={self.config.history_size})"
