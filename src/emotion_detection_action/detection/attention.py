"""Attention detection using MediaPipe Face Mesh for eye/gaze tracking."""

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from emotion_detection_action.core.config import ModelConfig
from emotion_detection_action.core.types import EyeDetection, GazeDetection
from emotion_detection_action.models.base import BaseModel

# Try to import mediapipe
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


# MediaPipe Face Mesh landmark indices for eyes
# Left eye landmarks (from user's perspective, so right side of image)
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]  # Upper/lower eyelid
LEFT_IRIS_INDICES = [474, 475, 476, 477]  # Iris landmarks
LEFT_EYE_INNER = 463
LEFT_EYE_OUTER = 359

# Right eye landmarks
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]  # Upper/lower eyelid
RIGHT_IRIS_INDICES = [469, 470, 471, 472]  # Iris landmarks
RIGHT_EYE_INNER = 133
RIGHT_EYE_OUTER = 33


@dataclass
class EyeTrackingState:
    """State for tracking eye metrics over time."""

    pupil_history: deque  # History of pupil sizes
    gaze_history: deque  # History of gaze positions
    blink_timestamps: deque  # Timestamps of detected blinks
    baseline_pupil: float | None = None  # Baseline pupil size
    last_eye_openness: float = 1.0  # For blink detection


class AttentionDetector(BaseModel):
    """Detects eye gaze, pupil size, and attention metrics using MediaPipe.

    This detector uses MediaPipe Face Mesh to track:
    - Eye landmarks and iris position
    - Estimated pupil size (relative)
    - Gaze direction
    - Blink detection

    Example:
        >>> detector = AttentionDetector(config)
        >>> detector.load()
        >>> gaze = detector.predict(rgb_frame)
        >>> print(f"Gaze direction: {gaze.gaze_direction}")
    """

    def __init__(
        self,
        config: ModelConfig,
        history_size: int = 30,
        blink_threshold: float = 0.2,
    ) -> None:
        """Initialize attention detector.

        Args:
            config: Model configuration.
            history_size: Number of frames to keep in history for smoothing.
            blink_threshold: Eye openness threshold for blink detection.
        """
        super().__init__(config)
        self._face_mesh: Any = None
        self._history_size = history_size
        self._blink_threshold = blink_threshold

        # Tracking state
        self._state = EyeTrackingState(
            pupil_history=deque(maxlen=history_size),
            gaze_history=deque(maxlen=history_size),
            blink_timestamps=deque(maxlen=100),
        )

    def load(self) -> None:
        """Load the MediaPipe Face Mesh model."""
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError(
                "MediaPipe is not available. Install with: pip install mediapipe"
            )

        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # Enables iris landmarks
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._is_loaded = True

    def unload(self) -> None:
        """Unload the model."""
        if self._face_mesh:
            self._face_mesh.close()
            self._face_mesh = None
        self._is_loaded = False
        self._reset_state()

    def _reset_state(self) -> None:
        """Reset tracking state."""
        self._state = EyeTrackingState(
            pupil_history=deque(maxlen=self._history_size),
            gaze_history=deque(maxlen=self._history_size),
            blink_timestamps=deque(maxlen=100),
        )

    def predict(self, image: np.ndarray) -> GazeDetection | None:
        """Detect gaze and eye metrics from image.

        Args:
            image: RGB image as numpy array.

        Returns:
            GazeDetection result or None if no face detected.
        """
        if not self._is_loaded or self._face_mesh is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Process with MediaPipe
        results = self._face_mesh.process(image)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0]
        h, w = image.shape[:2]

        # Extract eye detections
        left_eye = self._extract_eye(landmarks, w, h, is_left=True)
        right_eye = self._extract_eye(landmarks, w, h, is_left=False)

        # Calculate gaze direction
        gaze_direction = self._calculate_gaze_direction(landmarks, w, h)

        # Detect blinks
        avg_openness = (left_eye.openness + right_eye.openness) / 2
        self._detect_blink(avg_openness)

        # Update pupil history
        avg_pupil = (left_eye.pupil_size + right_eye.pupil_size) / 2
        self._state.pupil_history.append(avg_pupil)
        self._state.gaze_history.append(gaze_direction)

        # Update baseline if needed
        if self._state.baseline_pupil is None and len(self._state.pupil_history) >= 10:
            self._state.baseline_pupil = np.median(list(self._state.pupil_history))

        # Calculate confidence based on landmark visibility
        confidence = min(left_eye.openness, right_eye.openness)
        confidence = max(0.3, confidence)  # Minimum confidence if eyes detected

        return GazeDetection(
            left_eye=left_eye,
            right_eye=right_eye,
            gaze_direction=gaze_direction,
            gaze_point=None,  # Would need calibration for screen mapping
            confidence=confidence,
        )

    def _extract_eye(
        self,
        landmarks: Any,
        width: int,
        height: int,
        is_left: bool,
    ) -> EyeDetection:
        """Extract eye detection from landmarks.

        Args:
            landmarks: MediaPipe face landmarks.
            width: Image width.
            height: Image height.
            is_left: Whether to extract left eye (True) or right eye (False).

        Returns:
            EyeDetection for the specified eye.
        """
        if is_left:
            eye_indices = LEFT_EYE_INDICES
            iris_indices = LEFT_IRIS_INDICES
            inner_idx = LEFT_EYE_INNER
            outer_idx = LEFT_EYE_OUTER
        else:
            eye_indices = RIGHT_EYE_INDICES
            iris_indices = RIGHT_IRIS_INDICES
            inner_idx = RIGHT_EYE_INNER
            outer_idx = RIGHT_EYE_OUTER

        # Get eye landmarks
        eye_points = []
        for idx in eye_indices:
            lm = landmarks.landmark[idx]
            eye_points.append([lm.x * width, lm.y * height])
        eye_points = np.array(eye_points)

        # Get iris landmarks for pupil estimation
        iris_points = []
        for idx in iris_indices:
            lm = landmarks.landmark[idx]
            iris_points.append([lm.x * width, lm.y * height])
        iris_points = np.array(iris_points)

        # Calculate eye center from iris
        center = (float(np.mean(iris_points[:, 0])), float(np.mean(iris_points[:, 1])))

        # Estimate pupil size from iris diameter (normalized by eye width)
        inner_corner = landmarks.landmark[inner_idx]
        outer_corner = landmarks.landmark[outer_idx]
        eye_width = np.sqrt(
            (outer_corner.x - inner_corner.x) ** 2 +
            (outer_corner.y - inner_corner.y) ** 2
        ) * width

        iris_diameter = np.max(iris_points[:, 0]) - np.min(iris_points[:, 0])
        pupil_size = iris_diameter / eye_width if eye_width > 0 else 0.0

        # Calculate eye openness (eye aspect ratio)
        openness = self._calculate_eye_aspect_ratio(eye_points)

        return EyeDetection(
            center=center,
            landmarks=eye_points,
            pupil_size=float(pupil_size),
            openness=float(openness),
        )

    def _calculate_eye_aspect_ratio(self, eye_points: np.ndarray) -> float:
        """Calculate Eye Aspect Ratio (EAR) for blink detection.

        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)

        Args:
            eye_points: 6 eye landmark points.

        Returns:
            Eye aspect ratio (0 = closed, ~0.3 = open).
        """
        if len(eye_points) < 6:
            return 1.0

        # Vertical distances
        v1 = np.linalg.norm(eye_points[1] - eye_points[5])
        v2 = np.linalg.norm(eye_points[2] - eye_points[4])

        # Horizontal distance
        h = np.linalg.norm(eye_points[0] - eye_points[3])

        if h == 0:
            return 1.0

        ear = (v1 + v2) / (2.0 * h)

        # Normalize to 0-1 range (typical EAR is 0.2-0.3 when open)
        return min(1.0, ear / 0.3)

    def _calculate_gaze_direction(
        self,
        landmarks: Any,
        width: int,
        height: int,
    ) -> tuple[float, float]:
        """Calculate normalized gaze direction.

        Uses iris position relative to eye corners to estimate gaze.

        Args:
            landmarks: MediaPipe face landmarks.
            width: Image width.
            height: Image height.

        Returns:
            Normalized gaze direction (x, y) where (0, 0) is center.
        """
        # Get iris centers
        left_iris = []
        for idx in LEFT_IRIS_INDICES:
            lm = landmarks.landmark[idx]
            left_iris.append([lm.x, lm.y])
        left_iris = np.mean(left_iris, axis=0)

        right_iris = []
        for idx in RIGHT_IRIS_INDICES:
            lm = landmarks.landmark[idx]
            right_iris.append([lm.x, lm.y])
        right_iris = np.mean(right_iris, axis=0)

        # Get eye corners for reference
        left_inner = landmarks.landmark[LEFT_EYE_INNER]
        left_outer = landmarks.landmark[LEFT_EYE_OUTER]
        right_inner = landmarks.landmark[RIGHT_EYE_INNER]
        right_outer = landmarks.landmark[RIGHT_EYE_OUTER]

        # Calculate iris position relative to eye (0 = inner corner, 1 = outer corner)
        left_eye_width = left_outer.x - left_inner.x
        right_eye_width = right_inner.x - right_outer.x

        if left_eye_width != 0 and right_eye_width != 0:
            left_ratio = (left_iris[0] - left_inner.x) / left_eye_width
            right_ratio = (right_iris[0] - right_outer.x) / right_eye_width

            # Average and normalize to -1 to 1 (center = 0)
            gaze_x = (left_ratio + right_ratio) / 2 * 2 - 1
        else:
            gaze_x = 0.0

        # Vertical gaze (simplified - based on iris vertical position)
        nose_tip = landmarks.landmark[1]
        avg_iris_y = (left_iris[1] + right_iris[1]) / 2
        gaze_y = (avg_iris_y - nose_tip.y) * 5  # Scale factor

        # Clamp to -1 to 1
        gaze_x = max(-1.0, min(1.0, gaze_x))
        gaze_y = max(-1.0, min(1.0, gaze_y))

        return (float(gaze_x), float(gaze_y))

    def _detect_blink(self, openness: float) -> bool:
        """Detect if a blink occurred.

        Args:
            openness: Current average eye openness.

        Returns:
            True if blink detected.
        """
        import time

        was_closed = self._state.last_eye_openness < self._blink_threshold
        is_closed = openness < self._blink_threshold

        self._state.last_eye_openness = openness

        # Blink = transition from closed to open
        if was_closed and not is_closed:
            self._state.blink_timestamps.append(time.time())
            return True

        return False

    def get_blink_rate(self, window_seconds: float = 60.0) -> float:
        """Get blinks per minute over the specified window.

        Args:
            window_seconds: Time window to calculate rate over.

        Returns:
            Blinks per minute.
        """
        import time

        current_time = time.time()
        cutoff = current_time - window_seconds

        recent_blinks = [t for t in self._state.blink_timestamps if t > cutoff]
        if not recent_blinks:
            return 0.0

        # Scale to per-minute rate
        rate = len(recent_blinks) * (60.0 / window_seconds)
        return rate

    def get_pupil_baseline(self) -> float | None:
        """Get the established pupil baseline."""
        return self._state.baseline_pupil

    def get_pupil_dilation(self) -> float:
        """Get current pupil dilation relative to baseline.

        Returns:
            Dilation value (positive = dilated, negative = constricted).
        """
        if not self._state.pupil_history or self._state.baseline_pupil is None:
            return 0.0

        current = self._state.pupil_history[-1]
        return (current - self._state.baseline_pupil) / self._state.baseline_pupil

    def get_gaze_stability(self, window: int = 10) -> float:
        """Calculate gaze stability over recent frames.

        Args:
            window: Number of frames to consider.

        Returns:
            Stability score (0 = very unstable, 1 = very stable).
        """
        if len(self._state.gaze_history) < 2:
            return 1.0

        recent = list(self._state.gaze_history)[-window:]
        if len(recent) < 2:
            return 1.0

        # Calculate variance in gaze positions
        x_vals = [g[0] for g in recent]
        y_vals = [g[1] for g in recent]

        variance = np.var(x_vals) + np.var(y_vals)

        # Convert variance to stability (lower variance = higher stability)
        # Typical variance range is 0-0.1 for stable gaze
        stability = 1.0 / (1.0 + variance * 10)
        return float(stability)

    def reset_baseline(self) -> None:
        """Reset the pupil baseline for recalibration."""
        self._state.baseline_pupil = None

    def __repr__(self) -> str:
        return f"AttentionDetector(loaded={self._is_loaded})"
