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
    def test_multimodal_pipeline_flow(self):
        """Test multimodal pipeline with both visual and audio."""
        from emotion_detection_action.emotion.fusion import EmotionFusion

        # Create facial result
        bbox = BoundingBox(x=0, y=0, width=100, height=100)
        face = FaceDetection(bbox=bbox, confidence=0.95)
        facial_scores = EmotionScores(happy=0.7, neutral=0.3)
        facial_result = FacialEmotionResult(
            face_detection=face,
            emotions=facial_scores,
            confidence=0.85,
        )

        # Create speech result (simulated)
        from emotion_detection_action.core.types import SpeechEmotionResult, VoiceDetection

        voice = VoiceDetection(
            is_speech=True,
            confidence=0.9,
            start_time=0.0,
            end_time=1.0,
        )
        speech_scores = EmotionScores(happy=0.9, neutral=0.1)
        speech_result = SpeechEmotionResult(
            voice_detection=voice,
            emotions=speech_scores,
            confidence=0.8,
        )

        # Fuse using ML-based fusion
        fusion = EmotionFusion()
        fused = fusion.fuse(facial_result, speech_result, timestamp=0.5)

        # Both modalities indicate happy - ML model should produce a valid result
        assert fused is not None
        assert fused.dominant_emotion is not None
        assert fused.fusion_confidence > 0

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
    """Test configuration integration with pipeline."""

    def test_config_fusion_settings(self):
        """Test that config fusion settings are available."""
        config = Config(
            fusion_model_path="models/test.pt",
            fusion_device="cpu",
        )
        assert config.fusion_model_path == "models/test.pt"
        assert config.fusion_device == "cpu"

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_fusion_produces_valid_output(self):
        """Test that fusion produces valid emotion output."""
        from emotion_detection_action.emotion.fusion import EmotionFusion

        fusion = EmotionFusion()

        # Create conflicting results
        bbox = BoundingBox(x=0, y=0, width=100, height=100)
        face = FaceDetection(bbox=bbox, confidence=0.95)
        facial = FacialEmotionResult(
            face_detection=face,
            emotions=EmotionScores(happy=1.0),
            confidence=0.9,
        )

        from emotion_detection_action.core.types import SpeechEmotionResult, VoiceDetection

        voice = VoiceDetection(is_speech=True, confidence=0.9, start_time=0.0, end_time=1.0)
        speech = SpeechEmotionResult(
            voice_detection=voice,
            emotions=EmotionScores(sad=1.0),
            confidence=0.9,
        )

        result = fusion.fuse(facial, speech)

        # ML-based fusion should produce a valid result
        assert result is not None
        assert result.dominant_emotion is not None
        # Emotions should sum to approximately 1
        scores = result.emotions.to_dict()
        total = sum(scores.values())
        assert 0.99 < total < 1.01
