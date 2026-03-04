"""Integration tests for the emotion detection pipeline."""

import numpy as np
import pytest

from emotion_detection_action.actions.logging_handler import MockActionHandler
from emotion_detection_action.core.config import Config
from emotion_detection_action.core.types import (
    ActionCommand,
    BoundingBox,
    DetectionResult,
    EmotionResult,
    EmotionScores,
    FaceDetection,
    FacialEmotionResult,
    PipelineResult,
)

# Check if PyTorch is available for fusion tests
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestPipelineIntegration:
    """Integration tests for the full pipeline flow."""

    def test_detection_to_emotion_flow(self):
        """Test flow from detection to emotion result."""
        # Simulate face detection
        bbox = BoundingBox(x=100, y=100, width=200, height=200)
        face = FaceDetection(
            bbox=bbox,
            confidence=0.95,
            face_image=np.zeros((200, 200, 3), dtype=np.uint8),
        )

        # Create detection result
        detection = DetectionResult(
            timestamp=1.0,
            faces=[face],
            voice=None,
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
        )

        assert len(detection.faces) == 1
        assert detection.faces[0].confidence == 0.95

    def test_emotion_to_action_flow(self):
        """Test flow from emotion result to action."""
        # Create emotion result
        scores = EmotionScores(happy=0.8, neutral=0.2)
        emotion = EmotionResult(
            timestamp=1.0,
            emotions=scores,
            fusion_confidence=0.85,
        )

        # Use mock handler
        handler = MockActionHandler()
        handler.connect()
        handler.expect_action("acknowledge")  # Happy -> acknowledge

        # Execute for emotion
        handler.execute_for_emotion(emotion)

        # Verify
        success, _ = handler.verify_expectations()
        assert success is True

    def test_full_pipeline_result(self):
        """Test creating a full pipeline result."""
        # Detection
        detection = DetectionResult(timestamp=1.0)

        # Emotion
        scores = EmotionScores(surprised=0.7, neutral=0.3)
        emotion = EmotionResult(
            timestamp=1.0,
            emotions=scores,
            fusion_confidence=0.8,
        )

        # Action
        action = ActionCommand(
            action_type="wait",
            parameters={"duration": 2.0},
            confidence=0.9,
        )

        # Full result
        result = PipelineResult(
            timestamp=1.0,
            detection=detection,
            emotion=emotion,
            action=action,
        )

        # Verify serialization
        d = result.to_dict()
        assert d["timestamp"] == 1.0
        assert d["dominant_emotion"] == "surprised"
        assert d["action"]["type"] == "wait"

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_neural_pipeline_flow(self):
        """Test neural pipeline (NeuralFusionModel) with stub backbones."""
        import numpy as np
        import torch
        from emotion_detection_action import Config, EmotionDetector

        config = Config(
            two_tower_pretrained=False,
            two_tower_device="cpu",
            vla_enabled=False,
        )
        detector = EmotionDetector(config)
        detector.initialize()

        # Single-frame API
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = detector.process_frame(frame)
        # process_frame accumulates frames; result may be None until buffer is full
        # Drive it to full by submitting enough frames
        for _ in range(20):
            result = detector.process_frame(frame)

        assert result is not None
        assert result.dominant_emotion in [
            "angry", "disgusted", "fearful", "happy",
            "neutral", "sad", "surprised", "unclear",
        ]
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.latent_embedding) == 512
        assert set(result.metrics.keys()) == {"stress", "engagement", "arousal"}

    def test_action_handler_integration(self):
        """Test action handler receives correct actions."""
        handler = MockActionHandler()
        handler.connect()

        # Simulate different emotion scenarios
        emotions_and_expected = [
            (EmotionScores(happy=0.9), "acknowledge"),
            (EmotionScores(sad=0.9), "comfort"),
            (EmotionScores(angry=0.9), "de_escalate"),
            (EmotionScores(neutral=0.9), "idle"),
        ]

        for scores, expected_action in emotions_and_expected:
            emotion = EmotionResult(
                timestamp=0.0,
                emotions=scores,
                fusion_confidence=0.9,
            )

            handler.reset_expectations()
            handler.expect_action(expected_action)
            handler.execute_for_emotion(emotion)

            success, msg = handler.verify_expectations()
            assert success, f"Failed for {scores}: {msg}"


class TestPipelineEdgeCases:
    """Test edge cases in the pipeline."""

    def test_no_face_detected(self):
        """Test handling when no face is detected."""
        detection = DetectionResult(
            timestamp=1.0,
            faces=[],  # No faces
            voice=None,
        )

        assert len(detection.faces) == 0

    def test_multiple_faces_detected(self):
        """Test handling multiple faces."""
        faces = []
        for i in range(3):
            bbox = BoundingBox(x=i * 100, y=0, width=100, height=100)
            faces.append(FaceDetection(bbox=bbox, confidence=0.9 - i * 0.1))

        detection = DetectionResult(timestamp=1.0, faces=faces)

        assert len(detection.faces) == 3
        # First face has highest confidence
        assert detection.faces[0].confidence == 0.9

    def test_low_confidence_results(self):
        """Test handling low confidence results."""
        # Low confidence emotion
        scores = EmotionScores(neutral=0.3, happy=0.3, sad=0.2, angry=0.2)
        emotion = EmotionResult(
            timestamp=1.0,
            emotions=scores,
            fusion_confidence=0.3,  # Low confidence
        )

        handler = MockActionHandler()
        handler.connect()

        # Should still execute action
        result = handler.execute_for_emotion(emotion)
        assert result is True

    def test_rapid_emotion_changes(self):
        """Test handling rapid emotion changes."""
        handler = MockActionHandler()
        handler.connect()

        emotions = [
            EmotionScores(happy=0.9),
            EmotionScores(sad=0.9),
            EmotionScores(angry=0.9),
            EmotionScores(happy=0.9),
        ]

        for scores in emotions:
            emotion = EmotionResult(
                timestamp=0.0,
                emotions=scores,
                fusion_confidence=0.9,
            )
            handler.execute_for_emotion(emotion)

        stats = handler.get_statistics()
        assert stats["total_actions"] == 4


class TestConfigIntegration:
    """Test configuration integration with the neural pipeline."""

    def test_neural_pipeline_config_settings(self):
        """Neural pipeline (Two-Tower Transformer) config fields are accessible."""
        config = Config(
            two_tower_pretrained=False,
            two_tower_device="cpu",
            two_tower_model_path="outputs/phase2_best.pt",
        )
        assert config.two_tower_pretrained is False
        assert config.two_tower_device == "cpu"
        assert config.two_tower_model_path == "outputs/phase2_best.pt"

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_neural_detector_produces_valid_output(self):
        """NeuralFusionModel stub produces correctly-shaped NeuralEmotionResult."""
        import numpy as np
        from emotion_detection_action import Config, EmotionDetector
        from emotion_detection_action.core.types import NeuralEmotionResult

        config = Config(
            two_tower_pretrained=False,
            two_tower_device="cpu",
            vla_enabled=False,
        )
        detector = EmotionDetector(config)
        detector.initialize()

        clip = np.random.randint(0, 255, (16, 480, 640, 3), dtype=np.uint8)
        audio = np.random.randn(8000).astype("float32")
        result = detector.process(clip, audio)

        assert isinstance(result, NeuralEmotionResult)
        assert result.dominant_emotion in [
            "angry", "disgusted", "fearful", "happy",
            "neutral", "sad", "surprised", "unclear",
        ]
        assert len(result.latent_embedding) == 512
        assert set(result.metrics.keys()) == {"stress", "engagement", "arousal"}
