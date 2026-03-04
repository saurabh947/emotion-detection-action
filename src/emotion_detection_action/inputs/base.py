"""Base input handler interface."""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Generic, Iterator, TypeVar

import numpy as np

FrameT = TypeVar("FrameT")


class BaseInput(ABC, Generic[FrameT]):
    """Abstract base class for input handlers.

    Input handlers manage reading from various sources (files, cameras,
    microphones) and provide both sync and async interfaces.
    """

    def __init__(self) -> None:
        """Initialize the input handler."""
        self._is_open = False

    @property
    def is_open(self) -> bool:
        """Check if the input source is open."""
        return self._is_open

    @abstractmethod
    def open(self, source: Any) -> None:
        """Open the input source.

        Args:
            source: Source identifier (file path, device index, etc.)
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the input source and release resources."""
        pass

    @abstractmethod
    def read(self) -> FrameT | None:
        """Read a single frame/sample from the source.

        Returns:
            Frame data or None if no more data.
        """
        pass

    def read_batch(self, batch_size: int) -> list[FrameT]:
        """Read multiple frames/samples.

        Args:
            batch_size: Number of frames to read.

        Returns:
            List of frames (may be shorter if source ends).
        """
        frames = []
        for _ in range(batch_size):
            frame = self.read()
            if frame is None:
                break
            frames.append(frame)
        return frames

    def __iter__(self) -> Iterator[FrameT]:
        """Iterate over all frames in the source."""
        while True:
            frame = self.read()
            if frame is None:
                break
            yield frame

    async def __aiter__(self) -> AsyncIterator[FrameT]:  # type: ignore[override]
        """Async iterate over frames.

        Note: calls the synchronous ``read()`` on each iteration.  For true
        async I/O (e.g. network streams), override this method in a subclass
        and use ``await`` appropriately.
        """
        import asyncio

        while True:
            frame = await asyncio.get_event_loop().run_in_executor(None, self.read)
            if frame is None:
                break
            yield frame

    def __enter__(self) -> "BaseInput[FrameT]":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()

    async def __aenter__(self) -> "BaseInput[FrameT]":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        self.close()


class VideoFrame:
    """Container for video frame data."""

    def __init__(
        self,
        data: np.ndarray,
        timestamp: float,
        frame_number: int,
    ) -> None:
        """Initialize a video frame.

        Args:
            data: Frame image data (H, W, C) in BGR or RGB format.
            timestamp: Timestamp in seconds.
            frame_number: Frame index in the video.
        """
        self.data = data
        self.timestamp = timestamp
        self.frame_number = frame_number

    @property
    def height(self) -> int:
        """Frame height."""
        return self.data.shape[0]

    @property
    def width(self) -> int:
        """Frame width."""
        return self.data.shape[1]

    @property
    def channels(self) -> int:
        """Number of color channels."""
        return self.data.shape[2] if len(self.data.shape) > 2 else 1


class AudioChunk:
    """Container for audio data."""

    def __init__(
        self,
        data: np.ndarray,
        sample_rate: int,
        start_time: float,
        channels: int = 1,
    ) -> None:
        """Initialize an audio chunk.

        Args:
            data: Audio samples.
            sample_rate: Sample rate in Hz.
            start_time: Start timestamp in seconds.
            channels: Number of audio channels.
        """
        self.data = data
        self.sample_rate = sample_rate
        self.start_time = start_time
        self.channels = channels

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return len(self.data) / self.sample_rate

    @property
    def end_time(self) -> float:
        """End timestamp in seconds."""
        return self.start_time + self.duration

    @property
    def num_samples(self) -> int:
        """Number of samples."""
        return len(self.data)

