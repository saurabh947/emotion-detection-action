"""Background inference worker for non-blocking real-time emotion detection.

The :class:`InferenceWorker` decouples the camera capture loop (main thread)
from model inference (background thread), so the camera never stalls waiting
for a forward pass to complete.

Design
------
::

    ┌─────────────────────────────────────────────────────────────────┐
    │  Main thread                                                    │
    │                                                                 │
    │  cap.read() → worker.push_frame(frame, audio)                  │
    │                          │                                      │
    │                    [FrameQueue]  ← drop-oldest when full       │
    │                          │                                      │
    │              ┌───────────▼──────────┐                          │
    │              │  Inference thread(s) │  (1 … num_workers)       │
    │              │  detector.process_frame()                       │
    │              └───────────┬──────────┘                          │
    │                          │                                      │
    │                  [latest_result]  ← atomic swap                │
    │                          │                                      │
    │  result = worker.latest_result   # always non-blocking         │
    │  display(result)                                                │
    └─────────────────────────────────────────────────────────────────┘

Queue semantics
---------------
* The frame queue has a configurable maximum size (``max_queue_size``).
* When the queue is full (inference is slower than capture), the **oldest**
  pending packet is silently dropped so the worker always stays close to the
  present moment.  The drop count is tracked in :attr:`stats`.
* A ``max_queue_size`` of 1 gives minimum latency (always process the newest
  frame); larger values smooth over transient inference spikes.

GPU scaling
-----------
Pass ``num_workers > 1`` to spin up multiple inference threads that share the
same model.  Because PyTorch's CUDA kernels are thread-safe under
``torch.no_grad()``, this effectively pipelines back-to-back clips on a GPU
and scales throughput nearly linearly with worker count — up to the point where
the GPU is saturated.  On CPU, threading is limited by the GIL and gives
diminishing returns beyond ``num_workers=2``.

Recommended settings
--------------------
+-----------------------+--------------------+--------------+--------------+
| Hardware              | device             | num_workers  | max_queue_sz |
+=======================+====================+==============+==============+
| CPU only              | ``"cpu"``          | 1            | 2            |
+-----------------------+--------------------+--------------+--------------+
| Apple Silicon (MPS)   | ``"mps"``          | 1–2          | 4            |
+-----------------------+--------------------+--------------+--------------+
| NVIDIA GPU (mid-range)| ``"cuda"``         | 2–4          | 8            |
+-----------------------+--------------------+--------------+--------------+
| NVIDIA GPU (high-end) | ``"cuda"``         | 4–8          | 16           |
+-----------------------+--------------------+--------------+--------------+
"""

from __future__ import annotations

import queue
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from emotion_detection_action.core.detector import EmotionDetector
from emotion_detection_action.core.types import NeuralEmotionResult


# ---------------------------------------------------------------------------
# Internal packet type
# ---------------------------------------------------------------------------


@dataclass
class _FramePacket:
    """One unit of work queued for inference."""

    frame: np.ndarray           # shape (H, W, 3) BGR frame
    audio: np.ndarray | None    # raw PCM samples, or None when no mic
    timestamp: float


# ---------------------------------------------------------------------------
# Worker stats (read from main thread, written from worker threads)
# ---------------------------------------------------------------------------


@dataclass
class WorkerStats:
    """Snapshot of :class:`InferenceWorker` performance counters.

    All fields are read-only snapshots — they do not update live.
    """

    frames_submitted: int = 0
    """Total frames pushed via :meth:`~InferenceWorker.push_frame`."""

    frames_dropped: int = 0
    """Frames discarded because the queue was full."""

    frames_processed: int = 0
    """Frames that were actually passed through the model."""

    results_produced: int = 0
    """Non-None results returned by the detector."""

    queue_depth: int = 0
    """Current number of packets waiting in the queue."""

    inference_latency_ms_p50: float = 0.0
    """Median inference latency over the last 30 inferences (ms)."""

    inference_latency_ms_p95: float = 0.0
    """95th-percentile inference latency over the last 30 inferences (ms)."""

    effective_fps: float = 0.0
    """Approximate results per second delivered to the main thread."""

    drop_rate: float = 0.0
    """Fraction of submitted frames that were dropped (0–1)."""

    @property
    def is_keeping_up(self) -> bool:
        """``True`` when the drop rate is below 10%."""
        return self.drop_rate < 0.1


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class InferenceWorker:
    """Non-blocking inference worker for real-time emotion detection.

    Wraps an :class:`~core.detector.EmotionDetector` and runs its
    ``process_frame`` method on one or more background threads, exposing a
    simple push/pull API to the main (camera) thread.

    Example — webcam loop::

        config = Config(two_tower_pretrained=True, two_tower_device="cuda")
        detector = EmotionDetector(config)
        detector.initialize()

        worker = InferenceWorker(detector, num_workers=2, max_queue_size=8)
        worker.start()

        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            audio_chunk = mic.read()          # your audio capture
            worker.push_frame(frame, audio_chunk)

            result = worker.latest_result     # always instant, never blocks
            if result:
                display(result)

        worker.stop()

    Args:
        detector: An initialised :class:`~core.detector.EmotionDetector`.
        num_workers: Number of background inference threads.  Use ``1`` for CPU,
            ``2–4`` for a mid-range GPU.  See module docstring for guidelines.
        max_queue_size: Maximum number of frame packets buffered before the
            oldest is dropped.  Smaller values → lower latency but more drops.
        on_result: Optional callback called on the *worker thread* every time
            a new result is produced.  Useful for logging or action dispatch
            without polling ``latest_result`` on the main thread.
            Signature: ``(result: NeuralEmotionResult) -> None``.
    """

    def __init__(
        self,
        detector: EmotionDetector,
        num_workers: int = 1,
        max_queue_size: int = 4,
        on_result: Callable[[NeuralEmotionResult], None] | None = None,
    ) -> None:
        import warnings

        if num_workers < 1:
            raise ValueError("num_workers must be >= 1")
        if max_queue_size < 1:
            raise ValueError("max_queue_size must be >= 1")
        if num_workers > 1:
            warnings.warn(
                f"InferenceWorker: num_workers={num_workers} shares one EmotionDetector "
                "instance across multiple threads. The GRU temporal buffer is protected "
                "by a lock, so correctness is guaranteed, but temporal continuity may be "
                "interrupted when different threads process consecutive frames out of "
                "order. Consider num_workers=1 if smooth temporal predictions matter.",
                stacklevel=2,
            )

        self._detector = detector
        self._num_workers = num_workers
        self._max_queue_size = max_queue_size
        self._on_result = on_result

        # Shared frame queue (producers: main thread; consumers: worker threads)
        self._frame_queue: queue.Queue[_FramePacket] = queue.Queue(maxsize=max_queue_size)

        # Latest result — written by worker threads, read by main thread
        self._latest_result: NeuralEmotionResult | None = None
        self._result_lock = threading.Lock()

        # Performance counters — all protected by _stats_lock
        self._stats_lock = threading.Lock()
        self._frames_submitted = 0
        self._frames_dropped = 0
        self._frames_processed = 0
        self._results_produced = 0
        self._result_timestamps: deque[float] = deque(maxlen=60)
        self._latency_samples: deque[float] = deque(maxlen=30)

        self._running = False
        self._threads: list[threading.Thread] = []
        self._fatal_error: BaseException | None = None

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def start(self) -> "InferenceWorker":
        """Start background inference threads.

        Returns ``self`` for chaining::

            worker = InferenceWorker(detector).start()
        """
        if self._running:
            return self
        self._running = True
        for i in range(self._num_workers):
            t = threading.Thread(
                target=self._inference_loop,
                name=f"InferenceWorker-{i}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)
        return self

    def stop(self, timeout: float = 5.0) -> None:
        """Signal threads to stop and wait for them to finish.

        Args:
            timeout: Maximum seconds to wait for each thread.
        """
        self._running = False
        # Unblock any threads blocked on queue.get()
        for _ in self._threads:
            try:
                self._frame_queue.put_nowait(_FramePacket(None, None, 0.0))  # type: ignore[arg-type]
            except queue.Full:
                pass
        for t in self._threads:
            t.join(timeout=timeout)
        self._threads.clear()

    # ------------------------------------------------------------------ #
    # Main-thread API                                                      #
    # ------------------------------------------------------------------ #

    def push_frame(
        self,
        frame: np.ndarray,
        audio: np.ndarray | None = None,
        timestamp: float | None = None,
    ) -> bool:
        """Submit a frame for asynchronous inference.  Never blocks.

        If the queue is full, the **oldest** pending packet is dropped to make
        room so the worker always sees approximately the current moment.

        Args:
            frame: ``(H, W, 3)`` BGR or RGB numpy array from the camera.
            audio: Optional raw PCM audio chunk (float32 numpy array).
            timestamp: Frame timestamp in seconds.  Defaults to ``time.time()``.

        Returns:
            ``True`` if the frame was queued, ``False`` if it was dropped.
        """
        ts = timestamp if timestamp is not None else time.time()
        packet = _FramePacket(frame=frame, audio=audio, timestamp=ts)

        with self._stats_lock:
            self._frames_submitted += 1

        # Try non-blocking insert first (common path when keeping up)
        try:
            self._frame_queue.put_nowait(packet)
            return True
        except queue.Full:
            pass

        # Queue is full — evict oldest, insert newest
        try:
            self._frame_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            self._frame_queue.put_nowait(packet)
            with self._stats_lock:
                self._frames_dropped += 1
            return False
        except queue.Full:
            with self._stats_lock:
                self._frames_dropped += 1
            return False

    @property
    def latest_result(self) -> NeuralEmotionResult | None:
        """Most recent inference result.  Always instant — never blocks.

        Returns ``None`` until the first result has been produced.
        """
        with self._result_lock:
            return self._latest_result

    @property
    def fatal_error(self) -> BaseException | None:
        """Set when a worker thread has encountered an unrecoverable error
        (e.g. CUDA OOM) and shut itself down.  Poll this from the main thread
        if you need to detect and restart the worker."""
        return self._fatal_error

    @property
    def stats(self) -> WorkerStats:
        """Snapshot of performance counters (safe to call from any thread)."""
        with self._stats_lock:
            submitted = self._frames_submitted
            dropped = self._frames_dropped
            processed = self._frames_processed
            produced = self._results_produced
            latencies = list(self._latency_samples)
            result_ts = list(self._result_timestamps)

        p50 = statistics.median(latencies) if latencies else 0.0
        p95 = (
            sorted(latencies)[int(len(latencies) * 0.95)]
            if len(latencies) >= 2
            else (latencies[0] if latencies else 0.0)
        )

        # Effective FPS: count results in the last 5 seconds
        now = time.monotonic()
        recent = sum(1 for t in result_ts if now - t < 5.0)
        eff_fps = recent / 5.0

        return WorkerStats(
            frames_submitted=submitted,
            frames_dropped=dropped,
            frames_processed=processed,
            results_produced=produced,
            queue_depth=self._frame_queue.qsize(),
            inference_latency_ms_p50=round(p50, 1),
            inference_latency_ms_p95=round(p95, 1),
            effective_fps=round(eff_fps, 1),
            drop_rate=round(dropped / max(submitted, 1), 3),
        )

    def reset(self) -> None:
        """Flush the frame queue and reset the detector's GRU temporal state.

        Call when the subject changes (new person in front of the camera).
        """
        # Drain the queue
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break
        self._detector.reset()
        with self._result_lock:
            self._latest_result = None

    # ------------------------------------------------------------------ #
    # Worker thread loop                                                   #
    # ------------------------------------------------------------------ #

    def _inference_loop(self) -> None:
        """Background inference loop — runs on each worker thread."""
        while self._running:
            try:
                packet = self._frame_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            # Sentinel sent by stop()
            if packet.frame is None:
                break

            t0 = time.perf_counter()
            try:
                result = self._detector.process_frame(
                    packet.frame,
                    packet.audio,
                    timestamp=packet.timestamp,
                )
            except Exception as exc:
                import logging
                _log = logging.getLogger(__name__)
                # Treat CUDA/OOM errors as unrecoverable — stop this thread and
                # surface the error so the main thread can detect and restart.
                _msg = str(exc).lower()
                _is_fatal = isinstance(exc, MemoryError) or (
                    isinstance(exc, RuntimeError)
                    and ("cuda" in _msg or "out of memory" in _msg)
                )
                if _is_fatal:
                    _log.error(
                        "InferenceWorker [%s]: unrecoverable error, shutting down thread. %s",
                        threading.current_thread().name, exc,
                        exc_info=True,
                    )
                    self._fatal_error = exc
                    self._running = False
                    break
                _log.warning(
                    "InferenceWorker: frame dropped due to inference error: %s", exc,
                    exc_info=True,
                )
                with self._stats_lock:
                    self._frames_processed += 1  # count as processed (not lost silently)
                continue

            elapsed_ms = (time.perf_counter() - t0) * 1000

            with self._stats_lock:
                self._frames_processed += 1
                self._latency_samples.append(elapsed_ms)

            if result is not None:
                with self._result_lock:
                    self._latest_result = result
                with self._stats_lock:
                    self._results_produced += 1
                    self._result_timestamps.append(time.monotonic())

                if self._on_result is not None:
                    try:
                        self._on_result(result)
                    except Exception as exc:
                        import logging
                        logging.getLogger(__name__).warning(
                            "InferenceWorker: on_result callback raised: %s", exc,
                            exc_info=True,
                        )

    # ------------------------------------------------------------------ #
    # Context manager                                                      #
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "InferenceWorker":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    def __repr__(self) -> str:
        s = self.stats
        return (
            f"InferenceWorker(workers={self._num_workers}, "
            f"queue={s.queue_depth}/{self._max_queue_size}, "
            f"fps={s.effective_fps}, "
            f"p50={s.inference_latency_ms_p50}ms, "
            f"drop={s.drop_rate:.1%})"
        )
