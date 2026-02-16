"""Video input handler for real-time camera streams."""

import cv2
import numpy as np

from emotion_detection_action.inputs.base import BaseInput, VideoFrame


class VideoInput(BaseInput[VideoFrame]):
    """Video input handler for real-time camera streams.

    Handles reading from live camera feeds using OpenCV.

    Example:
        >>> with VideoInput() as video:
        ...     video.open(0)  # Camera index (0 = default)
        ...     for frame in video:
        ...         process(frame)
    """

    def __init__(self, frame_skip: int = 1) -> None:
        """Initialize video input handler.

        Args:
            frame_skip: Process every nth frame (1 = all frames).
        """
        super().__init__()
        self._cap: cv2.VideoCapture | None = None
        self._frame_number = 0
        self._frame_skip = max(1, frame_skip)
        self._fps: float = 30.0
        self._camera_index: int = 0

    @property
    def fps(self) -> float:
        """Frames per second of the camera."""
        return self._fps

    @property
    def current_frame(self) -> int:
        """Current frame number."""
        return self._frame_number

    @property
    def width(self) -> int:
        """Frame width."""
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        """Frame height."""
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def open(self, camera_index: int = 0) -> None:
        """Open a camera for real-time streaming.

        Args:
            camera_index: Camera device index (0 = default camera).

        Raises:
            ValueError: If camera cannot be opened.
        """
        if self._is_open:
            self.close()

        self._camera_index = camera_index
        self._cap = cv2.VideoCapture(camera_index)

        if not self._cap.isOpened():
            raise ValueError(f"Could not open camera at index: {camera_index}")

        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._frame_number = 0
        self._is_open = True

    def close(self) -> None:
        """Close the camera stream."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._is_open = False
        self._frame_number = 0

    def read(self) -> VideoFrame | None:
        """Read the next frame from camera.

        Returns:
            VideoFrame or None if no frame available.
        """
        if not self._is_open or self._cap is None:
            return None

        # Skip frames if needed
        for _ in range(self._frame_skip - 1):
            self._cap.read()
            self._frame_number += 1

        ret, frame = self._cap.read()
        if not ret:
            return None

        timestamp = self._frame_number / self._fps
        result = VideoFrame(
            data=frame,
            timestamp=timestamp,
            frame_number=self._frame_number,
        )
        self._frame_number += 1
        return result

    @staticmethod
    def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
        """Convert BGR frame to RGB.

        Args:
            frame: BGR image array.

        Returns:
            RGB image array.
        """
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def __repr__(self) -> str:
        return (
            f"VideoInput(camera={self._camera_index}, open={self._is_open}, "
            f"frame={self._frame_number})"
        )
