"""Tests for EmotionDetector — quantization, checkpoint loading,
temporal-state management, absent-modality paths, and frame-buffer logic.

All tests use stub weights (two_tower_pretrained=False) and face_crop_enabled=False
so no downloads or MediaPipe are needed.
"""

import os
import tempfile

import numpy as np
import pytest
import torch

try:
    from emotion_detection_action import Config, EmotionDetector
    from emotion_detection_action.core.types import NeuralEmotionResult
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

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
# Initialization
# ---------------------------------------------------------------------------

class TestDetectorInit:
    def test_initialize_sets_flag(self):
        d = EmotionDetector(Config(**_STUB_CONFIG))
        assert not d.is_initialized
        d.initialize()
        assert d.is_initialized

    def test_double_initialize_is_idempotent(self):
        d = EmotionDetector(Config(**_STUB_CONFIG))
        d.initialize()
        d.initialize()  # should not raise or re-create the model
        assert d.is_initialized

    def test_lazy_initialize_on_process(self):
        d = EmotionDetector(Config(**_STUB_CONFIG))
        assert not d.is_initialized
        clip = np.random.randint(0, 255, (16, 64, 64, 3), dtype=np.uint8)
        d.process(clip)   # should trigger lazy init
        assert d.is_initialized

    def test_context_manager_initializes_and_shuts_down(self):
        with EmotionDetector(Config(**_STUB_CONFIG)) as d:
            assert d.is_initialized
        assert not d.is_initialized


# ---------------------------------------------------------------------------
# Basic inference
# ---------------------------------------------------------------------------

class TestDetectorProcess:
    def test_returns_neural_emotion_result(self, detector):
        clip = np.random.randint(0, 255, (16, 64, 64, 3), dtype=np.uint8)
        result = detector.process(clip)
        assert isinstance(result, NeuralEmotionResult)

    def test_dominant_emotion_is_valid_label(self, detector):
        clip = np.random.randint(0, 255, (16, 64, 64, 3), dtype=np.uint8)
        result = detector.process(clip)
        assert result.dominant_emotion in [
            "angry", "disgusted", "fearful", "happy",
            "neutral", "sad", "surprised", "unclear",
        ]

    def test_confidence_in_range(self, detector):
        clip = np.random.randint(0, 255, (16, 64, 64, 3), dtype=np.uint8)
        result = detector.process(clip)
        assert 0.0 <= result.confidence <= 1.0

    def test_embedding_length(self, detector):
        clip = np.random.randint(0, 255, (16, 64, 64, 3), dtype=np.uint8)
        result = detector.process(clip)
        assert len(result.latent_embedding) == detector.config.two_tower_d_model

    def test_metrics_keys(self, detector):
        clip = np.random.randint(0, 255, (16, 64, 64, 3), dtype=np.uint8)
        result = detector.process(clip)
        assert set(result.metrics.keys()) == {"stress", "engagement", "arousal"}

    def test_video_only_sets_audio_missing(self, detector):
        clip = np.random.randint(0, 255, (16, 64, 64, 3), dtype=np.uint8)
        result = detector.process(clip, audio=None)
        assert result.audio_missing is True

    def test_with_audio(self, detector):
        clip = np.random.randint(0, 255, (16, 64, 64, 3), dtype=np.uint8)
        audio = np.random.randn(16000).astype(np.float32)
        result = detector.process(clip, audio)
        assert isinstance(result, NeuralEmotionResult)
        assert not result.audio_missing

    def test_short_clip_padded(self, detector):
        """Fewer frames than two_tower_video_frames should be repeat-padded."""
        clip = np.random.randint(0, 255, (4, 64, 64, 3), dtype=np.uint8)
        result = detector.process(clip)
        assert isinstance(result, NeuralEmotionResult)

    def test_float32_frames_accepted(self, detector):
        """Float32 [0,1] frames should be handled without error."""
        clip = np.random.rand(16, 64, 64, 3).astype(np.float32)
        result = detector.process(clip)
        assert isinstance(result, NeuralEmotionResult)


# ---------------------------------------------------------------------------
# process_frame (rolling buffer API)
# ---------------------------------------------------------------------------

class TestProcessFrame:
    def test_accumulates_and_returns_result(self, detector):
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = None
        for _ in range(20):
            result = detector.process_frame(frame)
        assert result is not None
        assert isinstance(result, NeuralEmotionResult)

    def test_bgr_to_rgb_conversion(self, detector):
        """BGR frame (as from OpenCV) should not crash."""
        bgr_frame = np.zeros((64, 64, 3), dtype=np.uint8)
        bgr_frame[:, :, 2] = 255  # red channel in BGR
        result = detector.process_frame(bgr_frame)
        assert result is not None


# ---------------------------------------------------------------------------
# Temporal state (GRU reset)
# ---------------------------------------------------------------------------

class TestTemporalState:
    def test_reset_clears_frame_buffer(self, detector):
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        for _ in range(8):
            detector.process_frame(frame)
        assert len(detector._frame_buffer) > 0
        detector.reset()
        assert len(detector._frame_buffer) == 0

    def test_reset_clears_audio_buffer(self, detector):
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        audio = np.random.randn(1600).astype(np.float32)
        detector.process_frame(frame, audio)
        detector.reset()
        assert len(detector._audio_buffer) == 0

    def test_reset_clears_gru_state(self, detector):
        """After reset, two identical clips should give identical results."""
        clip = np.random.randint(0, 255, (16, 64, 64, 3), dtype=np.uint8)

        result1 = detector.process(clip)
        detector.reset()
        result2 = detector.process(clip)

        # Same clip from a clean GRU state should produce the same dominant emotion.
        assert result1.dominant_emotion == result2.dominant_emotion


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

class TestQuantization:
    def test_quantize_sets_flag(self, detector):
        assert not detector.is_quantized
        detector.quantize("dynamic")
        assert detector.is_quantized

    def test_quantize_before_init_raises(self):
        d = EmotionDetector(Config(**_STUB_CONFIG))
        with pytest.raises(RuntimeError, match="initialize"):
            d.quantize("dynamic")

    def test_quantized_model_still_runs(self, detector):
        detector.quantize("dynamic")
        clip = np.random.randint(0, 255, (16, 64, 64, 3), dtype=np.uint8)
        result = detector.process(clip)
        assert isinstance(result, NeuralEmotionResult)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

class TestCheckpointLoading:
    def test_load_valid_checkpoint(self):
        """Save the stub model's state dict and reload it."""
        d = EmotionDetector(Config(**_STUB_CONFIG))
        d.initialize()
        assert d._model is not None

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "test.pt")
            payload = {
                "model_state": d._model.state_dict(),
                "epoch": 1,
                "phase": 1,
            }
            torch.save(payload, ckpt_path)

            d2 = EmotionDetector(Config(
                **_STUB_CONFIG,
                two_tower_model_path=ckpt_path,
            ))
            d2.initialize()
            assert d2.is_initialized

    def test_nonexistent_checkpoint_raises(self):
        d = EmotionDetector(Config(
            **_STUB_CONFIG,
            two_tower_model_path="/nonexistent/path/model.pt",
        ))
        with pytest.raises(FileNotFoundError):
            d.initialize()

    def test_wrong_extension_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_path = os.path.join(tmpdir, "model.bin")
            with open(bad_path, "w") as f:
                f.write("not a model")
            d = EmotionDetector(Config(
                **_STUB_CONFIG,
                two_tower_model_path=bad_path,
            ))
            with pytest.raises(ValueError, match=".pt or .pth"):
                d.initialize()


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------

class TestShutdown:
    def test_shutdown_resets_initialized_flag(self, detector):
        detector.shutdown()
        assert not detector.is_initialized

    def test_shutdown_clears_buffers(self, detector):
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        for _ in range(5):
            detector.process_frame(frame)
        detector.shutdown()
        assert len(detector._frame_buffer) == 0
        assert len(detector._audio_buffer) == 0
