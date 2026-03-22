"""Integration tests for the neural emotion detection pipeline."""

import numpy as np
import pytest

from emotion_detection_action.actions.logging_handler import MockActionHandler
from emotion_detection_action.core.config import Config
from emotion_detection_action.core.types import ActionCommand, NeuralEmotionResult

# Check if PyTorch is available for neural model tests
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _make_result(dominant: str = "happy", confidence: float = 0.9) -> NeuralEmotionResult:
    """Build a minimal NeuralEmotionResult for testing."""
    scores = {e: 0.0 for e in ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised", "unclear"]}
    scores[dominant] = confidence
    return NeuralEmotionResult(
        dominant_emotion=dominant,
        emotion_scores=scores,
        latent_embedding=[0.0] * 512,
        metrics={"stress": 0.1, "engagement": 0.8, "arousal": 0.5},
        confidence=confidence,
        timestamp=1.0,
    )


class TestActionHandlerIntegration:
    """Integration tests for action handler + NeuralEmotionResult."""

    def test_emotion_to_action_flow(self):
        """Happy emotion triggers acknowledge action."""
        result = _make_result("happy", 0.9)
        handler = MockActionHandler()
        handler.connect()
        handler.expect_action("acknowledge")

        handler.execute_for_emotion(result)

        success, _ = handler.verify_expectations()
        assert success is True

    def test_default_action_mapping(self):
        """Each emotion class maps to the correct default action."""
        cases = [
            ("happy",     "acknowledge"),
            ("sad",       "comfort"),
            ("angry",     "de_escalate"),
            ("fearful",   "reassure"),
            ("surprised", "wait"),
            ("disgusted", "retreat"),
            ("neutral",   "idle"),
            ("unclear",   "idle"),
        ]
        handler = MockActionHandler()
        handler.connect()

        for emotion, expected_action in cases:
            result = _make_result(emotion, 0.9)
            handler.reset_expectations()
            handler.expect_action(expected_action)
            handler.execute_for_emotion(result)
            success, msg = handler.verify_expectations()
            assert success, f"Failed for {emotion}: {msg}"

    def test_rapid_emotion_changes(self):
        """Handler survives rapid consecutive emotion changes."""
        handler = MockActionHandler()
        handler.connect()

        for emotion in ["happy", "sad", "angry", "happy"]:
            handler.execute_for_emotion(_make_result(emotion))

        stats = handler.get_statistics()
        assert stats["total_actions"] == 4


class TestConfigIntegration:
    """Configuration integration with the neural pipeline."""

    def test_neural_pipeline_config_fields(self):
        """Two-Tower config fields are accessible."""
        config = Config(
            two_tower_pretrained=False,
            two_tower_device="cpu",
            two_tower_model_path="outputs/phase2_best.pt",
        )
        assert config.two_tower_pretrained is False
        assert config.two_tower_device == "cpu"
        assert config.two_tower_model_path == "outputs/phase2_best.pt"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestNeuralPipeline:
    """End-to-end tests for the pure-neural detector."""

    def test_process_clip_returns_valid_result(self):
        """detector.process() returns a well-formed NeuralEmotionResult."""
        from emotion_detection_action import Config, EmotionDetector

        detector = EmotionDetector(Config(
            two_tower_pretrained=False,
            two_tower_device="cpu",
            two_tower_face_crop_enabled=False,  # skip MediaPipe in CI
            vla_enabled=False,
        ))
        detector.initialize()

        clip = np.random.randint(0, 255, (16, 480, 640, 3), dtype=np.uint8)
        # emotion2vec stub receives raw waveform (samples,)
        audio = np.random.randn(16000 * 3).astype("float32")
        result = detector.process(clip, audio)

        assert isinstance(result, NeuralEmotionResult)
        assert result.dominant_emotion in [
            "angry", "disgusted", "fearful", "happy",
            "neutral", "sad", "surprised", "unclear",
        ]
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.latent_embedding) == 512
        assert set(result.metrics.keys()) == {"stress", "engagement", "arousal"}

    def test_process_frame_accumulates_and_emits(self):
        """process_frame() returns a result once the frame buffer is full."""
        from emotion_detection_action import Config, EmotionDetector

        detector = EmotionDetector(Config(
            two_tower_pretrained=False,
            two_tower_device="cpu",
            two_tower_face_crop_enabled=False,
            vla_enabled=False,
        ))
        detector.initialize()

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = None
        for _ in range(20):
            result = detector.process_frame(frame)

        assert result is not None
        assert isinstance(result, NeuralEmotionResult)

    def test_result_action_integration(self):
        """NeuralEmotionResult flows correctly into an action handler."""
        from emotion_detection_action import Config, EmotionDetector

        detector = EmotionDetector(Config(
            two_tower_pretrained=False,
            two_tower_device="cpu",
            two_tower_face_crop_enabled=False,
            vla_enabled=False,
        ))
        detector.initialize()
        handler = MockActionHandler()
        handler.connect()

        clip = np.random.randint(0, 255, (16, 480, 640, 3), dtype=np.uint8)
        result = detector.process(clip)

        action = handler._generate_default_action(result)
        assert isinstance(action, ActionCommand)
        assert action.action_type in handler.get_supported_actions()
