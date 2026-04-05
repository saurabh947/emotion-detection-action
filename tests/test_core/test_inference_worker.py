"""Tests for InferenceWorker — threading, queue semantics, stats, lifecycle."""

import time

import numpy as np
import pytest

try:
    from emotion_detection_action import Config, EmotionDetector
    from emotion_detection_action.core.inference_worker import InferenceWorker, WorkerStats
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")

_STUB_CONFIG = dict(
    two_tower_pretrained=False,
    two_tower_device="cpu",
    two_tower_face_crop_enabled=False,
    vla_enabled=False,
)


@pytest.fixture
def detector():
    d = EmotionDetector(Config(**_STUB_CONFIG))
    d.initialize()
    return d


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------

class TestWorkerConstruction:
    def test_invalid_num_workers_raises(self, detector):
        with pytest.raises(ValueError, match="num_workers"):
            InferenceWorker(detector, num_workers=0)

    def test_invalid_queue_size_raises(self, detector):
        with pytest.raises(ValueError, match="max_queue_size"):
            InferenceWorker(detector, max_queue_size=0)

    def test_multi_worker_no_error(self, detector):
        """num_workers > 1 is allowed and should not raise (may warn)."""
        # pytest.ini suppresses the multi-worker advisory warning in tests
        InferenceWorker(detector, num_workers=2)


# ---------------------------------------------------------------------------
# Start / stop lifecycle
# ---------------------------------------------------------------------------

class TestWorkerLifecycle:
    def test_start_stop(self, detector):
        worker = InferenceWorker(detector, max_queue_size=2)
        worker.start()
        assert worker._running is True
        worker.stop(timeout=2.0)
        assert len(worker._threads) == 0

    def test_double_start_is_idempotent(self, detector):
        worker = InferenceWorker(detector, max_queue_size=2)
        worker.start()
        worker.start()  # second call should be a no-op
        assert len(worker._threads) == 1
        worker.stop(timeout=2.0)

    def test_context_manager(self, detector):
        with InferenceWorker(detector, max_queue_size=2) as worker:
            assert worker._running is True
        assert len(worker._threads) == 0


# ---------------------------------------------------------------------------
# Push / result
# ---------------------------------------------------------------------------

class TestWorkerPushResult:
    def test_latest_result_none_before_first_result(self, detector):
        worker = InferenceWorker(detector, max_queue_size=2)
        worker.start()
        assert worker.latest_result is None
        worker.stop(timeout=2.0)

    def test_push_frame_returns_true_when_space_available(self, detector):
        worker = InferenceWorker(detector, max_queue_size=4)
        worker.start()
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = worker.push_frame(frame)
        assert result is True
        worker.stop(timeout=3.0)

    def test_produces_result_after_inference(self, detector):
        worker = InferenceWorker(detector, max_queue_size=4)
        worker.start()
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

        for _ in range(20):
            worker.push_frame(frame)

        # Give worker time to process
        deadline = time.monotonic() + 10.0
        while worker.latest_result is None and time.monotonic() < deadline:
            time.sleep(0.05)

        worker.stop(timeout=3.0)
        assert worker.latest_result is not None

    def test_on_result_callback_called(self, detector):
        results = []

        def callback(r):
            results.append(r)

        worker = InferenceWorker(detector, max_queue_size=4, on_result=callback)
        worker.start()
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

        for _ in range(20):
            worker.push_frame(frame)

        deadline = time.monotonic() + 10.0
        while len(results) == 0 and time.monotonic() < deadline:
            time.sleep(0.05)

        worker.stop(timeout=3.0)
        assert len(results) > 0


# ---------------------------------------------------------------------------
# Queue drop semantics
# ---------------------------------------------------------------------------

class TestWorkerQueueDrop:
    def test_queue_full_increments_dropped(self, detector):
        """Filling the queue past capacity should increment frames_dropped."""
        worker = InferenceWorker(detector, max_queue_size=1)
        # Don't start — so nothing drains the queue
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

        # Push enough frames to fill and overflow the queue
        for _ in range(5):
            worker.push_frame(frame)

        stats = worker.stats
        assert stats.frames_submitted == 5
        # At least some frames should have been dropped
        assert stats.frames_dropped >= 1


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestWorkerStats:
    def test_stats_fields_present(self, detector):
        worker = InferenceWorker(detector)
        s = worker.stats
        assert isinstance(s, WorkerStats)
        assert hasattr(s, "frames_submitted")
        assert hasattr(s, "frames_dropped")
        assert hasattr(s, "frames_processed")
        assert hasattr(s, "drop_rate")
        assert hasattr(s, "is_keeping_up")

    def test_is_keeping_up_true_when_no_drops(self, detector):
        worker = InferenceWorker(detector)
        assert worker.stats.is_keeping_up is True  # drop_rate = 0 < 0.1

    def test_drop_rate_calculated_correctly(self, detector):
        worker = InferenceWorker(detector, max_queue_size=1)
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        for _ in range(10):
            worker.push_frame(frame)
        s = worker.stats
        assert s.drop_rate == pytest.approx(s.frames_dropped / max(s.frames_submitted, 1))


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestWorkerReset:
    def test_reset_clears_latest_result(self, detector):
        worker = InferenceWorker(detector, max_queue_size=4)
        worker.start()
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

        for _ in range(20):
            worker.push_frame(frame)

        deadline = time.monotonic() + 10.0
        while worker.latest_result is None and time.monotonic() < deadline:
            time.sleep(0.05)

        worker.stop(timeout=3.0)
        worker.reset()
        assert worker.latest_result is None
