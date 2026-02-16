"""Tests for multimodal emotion fusion using machine learning."""

import tempfile
from pathlib import Path

import pytest

from emotion_detection_action.core.types import (
    AttentionMetrics,
    AttentionResult,
    BoundingBox,
    EmotionScores,
    FaceDetection,
    FacialEmotionResult,
    SpeechEmotionResult,
    VoiceDetection,
)

# Check if PyTorch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")


def create_facial_result(
    happy: float = 0.0,
    sad: float = 0.0,
    angry: float = 0.0,
    neutral: float = 0.0,
    fearful: float = 0.0,
    confidence: float = 0.9,
) -> FacialEmotionResult:
    """Helper to create facial emotion result."""
    bbox = BoundingBox(x=0, y=0, width=100, height=100)
    face = FaceDetection(bbox=bbox, confidence=0.95)
    scores = EmotionScores(
        happy=happy, sad=sad, angry=angry, neutral=neutral, fearful=fearful
    )
    return FacialEmotionResult(face_detection=face, emotions=scores, confidence=confidence)


def create_speech_result(
    happy: float = 0.0,
    sad: float = 0.0,
    angry: float = 0.0,
    neutral: float = 0.0,
    fearful: float = 0.0,
    confidence: float = 0.85,
) -> SpeechEmotionResult:
    """Helper to create speech emotion result."""
    voice = VoiceDetection(is_speech=True, confidence=0.9, start_time=0.0, end_time=1.0)
    scores = EmotionScores(
        happy=happy, sad=sad, angry=angry, neutral=neutral, fearful=fearful
    )
    return SpeechEmotionResult(voice_detection=voice, emotions=scores, confidence=confidence)


def create_attention_result(
    stress: float = 0.0,
    engagement: float = 0.5,
    nervousness: float = 0.0,
    confidence: float = 0.9,
) -> AttentionResult:
    """Helper to create attention result."""
    metrics = AttentionMetrics(
        stress_score=stress,
        engagement_score=engagement,
        nervousness_score=nervousness,
    )
    return AttentionResult(timestamp=0.0, metrics=metrics, confidence=confidence)


class TestEmotionFusion:
    """Tests for EmotionFusion class (ML-based)."""

    def test_initialization(self):
        """Test fusion initialization."""
        from emotion_detection_action.emotion.fusion import EmotionFusion

        fusion = EmotionFusion()
        assert fusion.is_loaded

    def test_initialization_with_device(self):
        """Test fusion initialization with device option."""
        from emotion_detection_action.emotion.fusion import EmotionFusion

        fusion = EmotionFusion(device="cpu")
        assert fusion.device == "cpu"
        assert fusion.is_loaded

    def test_fuse_facial_only(self):
        """Test fusion with only facial result."""
        from emotion_detection_action.emotion.fusion import EmotionFusion

        fusion = EmotionFusion()
        facial = create_facial_result(happy=0.8, neutral=0.2)

        result = fusion.fuse(facial_result=facial, speech_result=None, timestamp=1.0)

        assert result.timestamp == 1.0
        assert result.facial_result is not None
        assert result.speech_result is None
        # All emotions should sum to ~1
        scores = result.emotions.to_dict()
        total = sum(scores.values())
        assert 0.99 < total < 1.01

    def test_fuse_speech_only(self):
        """Test fusion with only speech result."""
        from emotion_detection_action.emotion.fusion import EmotionFusion

        fusion = EmotionFusion()
        speech = create_speech_result(sad=0.7, neutral=0.3)

        result = fusion.fuse(facial_result=None, speech_result=speech, timestamp=2.0)

        assert result.timestamp == 2.0
        assert result.facial_result is None
        assert result.speech_result is not None

    def test_fuse_both_modalities(self):
        """Test fusion with both facial and speech."""
        from emotion_detection_action.emotion.fusion import EmotionFusion

        fusion = EmotionFusion()
        facial = create_facial_result(happy=0.8)
        speech = create_speech_result(happy=0.6)

        result = fusion.fuse(facial_result=facial, speech_result=speech)

        assert result.facial_result is not None
        assert result.speech_result is not None
        assert result.fusion_confidence > 0

    def test_fuse_with_attention(self):
        """Test fusion with attention metrics."""
        from emotion_detection_action.emotion.fusion import EmotionFusion

        fusion = EmotionFusion()
        facial = create_facial_result(fearful=0.5, neutral=0.5)
        attention = create_attention_result(stress=0.8, nervousness=0.7)

        result = fusion.fuse(
            facial_result=facial,
            speech_result=None,
            attention_result=attention,
        )

        assert result.attention_result is not None
        assert result.attention_result.stress_score == 0.8

    def test_fuse_no_inputs_raises(self):
        """Test that fusion with no inputs raises error."""
        from emotion_detection_action.emotion.fusion import EmotionFusion

        fusion = EmotionFusion()

        with pytest.raises(ValueError, match="At least one of facial or speech"):
            fusion.fuse(facial_result=None, speech_result=None)

    def test_dominant_emotion(self):
        """Test that dominant emotion is identified."""
        from emotion_detection_action.emotion.fusion import EmotionFusion

        fusion = EmotionFusion()
        facial = create_facial_result(happy=0.9)

        result = fusion.fuse(facial_result=facial, speech_result=None)

        assert result.dominant_emotion is not None


class TestEmotionFusionWithCustomModel:
    """Tests for EmotionFusion with custom trained model."""

    def test_with_custom_model_path(self):
        """Test fusion with a custom model path."""
        from emotion_detection_action.emotion.fusion import EmotionFusion
        from emotion_detection_action.emotion.learned_fusion import FusionMLP, save_model

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save a model
            model = FusionMLP()
            model_path = Path(tmpdir) / "custom.pt"
            save_model(model, model_path)

            # Use it in EmotionFusion
            fusion = EmotionFusion(model_path=str(model_path))
            assert fusion.is_loaded

            facial = create_facial_result(angry=0.8)
            result = fusion.fuse(facial, None)

            assert result is not None

    def test_reload_model(self):
        """Test reloading model."""
        from emotion_detection_action.emotion.fusion import EmotionFusion
        from emotion_detection_action.emotion.learned_fusion import FusionMLP, save_model

        fusion = EmotionFusion()
        assert fusion.is_loaded

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save a different model
            model = FusionMLP(hidden_dims=[128, 64])
            model_path = Path(tmpdir) / "new_model.pt"
            save_model(model, model_path)

            # Reload with new model
            fusion.reload(model_path=str(model_path))
            assert fusion.is_loaded

            facial = create_facial_result(happy=0.9)
            result = fusion.fuse(facial, None)
            assert result is not None


class TestFusionEdgeCases:
    """Test edge cases for emotion fusion."""

    def test_all_emotions_zero(self):
        """Test fusion when all emotion scores are zero."""
        from emotion_detection_action.emotion.fusion import EmotionFusion

        fusion = EmotionFusion()
        facial = create_facial_result()  # All zeros
        speech = create_speech_result()  # All zeros

        result = fusion.fuse(facial, speech)

        # Should still return a result
        assert result is not None
        assert result.dominant_emotion is not None

    def test_attention_preserves_in_result(self):
        """Test that attention result is preserved in output."""
        from emotion_detection_action.emotion.fusion import EmotionFusion

        fusion = EmotionFusion()
        facial = create_facial_result(neutral=0.9)
        attention = create_attention_result(stress=0.5, engagement=0.7, nervousness=0.3)

        result = fusion.fuse(facial, None, attention)

        # Original attention should be accessible
        assert result.attention is not None
        assert result.attention.stress_score == 0.5
        assert result.attention.engagement_score == 0.7
        assert result.attention.nervousness_score == 0.3
