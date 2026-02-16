"""Tests for learned emotion fusion module."""

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


class TestFusionMLP:
    """Tests for FusionMLP model."""

    def test_initialization(self):
        """Test model initialization."""
        from emotion_detection_action.emotion.learned_fusion import FusionMLP

        model = FusionMLP()
        assert model.facial_dim == 7
        assert model.speech_dim == 7
        assert model.attention_dim == 3
        assert model.num_emotions == 7

    def test_custom_hidden_dims(self):
        """Test model with custom hidden dimensions."""
        from emotion_detection_action.emotion.learned_fusion import FusionMLP

        model = FusionMLP(hidden_dims=[128, 64, 32])
        assert model.get_num_parameters() > 0

    def test_forward_pass(self):
        """Test forward pass with sample input."""
        from emotion_detection_action.emotion.learned_fusion import FusionMLP

        model = FusionMLP()
        model.eval()

        facial = torch.rand(1, 7)
        speech = torch.rand(1, 7)
        attention = torch.rand(1, 3)

        emotion_probs, confidence = model(facial, speech, attention)

        assert emotion_probs.shape == (1, 7)
        assert confidence.shape == (1, 1)
        # Probabilities should sum to 1
        assert torch.allclose(emotion_probs.sum(dim=-1), torch.ones(1), atol=1e-5)
        # Confidence should be between 0 and 1
        assert (confidence >= 0).all() and (confidence <= 1).all()

    def test_batch_forward(self):
        """Test forward pass with batch input."""
        from emotion_detection_action.emotion.learned_fusion import FusionMLP

        model = FusionMLP()
        model.eval()

        batch_size = 8
        facial = torch.rand(batch_size, 7)
        speech = torch.rand(batch_size, 7)
        attention = torch.rand(batch_size, 3)

        emotion_probs, confidence = model(facial, speech, attention)

        assert emotion_probs.shape == (batch_size, 7)
        assert confidence.shape == (batch_size, 1)

    def test_no_confidence_output(self):
        """Test model without confidence output."""
        from emotion_detection_action.emotion.learned_fusion import FusionMLP

        model = FusionMLP(output_confidence=False)
        model.eval()

        facial = torch.rand(1, 7)
        speech = torch.rand(1, 7)
        attention = torch.rand(1, 3)

        emotion_probs, confidence = model(facial, speech, attention)

        assert emotion_probs.shape == (1, 7)
        assert confidence is None

    def test_parameter_count(self):
        """Test that model has expected number of parameters."""
        from emotion_detection_action.emotion.learned_fusion import FusionMLP

        model = FusionMLP(hidden_dims=[64, 32])
        params = model.get_num_parameters()

        # Expected: 17*64 + 64 + 64*32 + 32 + 32*8 + 8 = 1088 + 64 + 2048 + 32 + 256 + 8 = 3496
        # With confidence output: 32*8 = 256
        assert params > 3000 and params < 5000


class TestLearnedEmotionFusion:
    """Tests for LearnedEmotionFusion wrapper."""

    def test_initialization(self):
        """Test fusion wrapper initialization."""
        from emotion_detection_action.emotion.learned_fusion import LearnedEmotionFusion

        fusion = LearnedEmotionFusion()
        assert not fusion.is_loaded

    def test_load_unload(self):
        """Test loading and unloading model."""
        from emotion_detection_action.emotion.learned_fusion import LearnedEmotionFusion

        fusion = LearnedEmotionFusion()
        assert not fusion.is_loaded

        fusion.load()
        assert fusion.is_loaded

        fusion.unload()
        assert not fusion.is_loaded

    def test_fuse_facial_only(self):
        """Test fusion with only facial input."""
        from emotion_detection_action.emotion.learned_fusion import LearnedEmotionFusion

        fusion = LearnedEmotionFusion()
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
        """Test fusion with only speech input."""
        from emotion_detection_action.emotion.learned_fusion import LearnedEmotionFusion

        fusion = LearnedEmotionFusion()
        speech = create_speech_result(sad=0.7, neutral=0.3)

        result = fusion.fuse(facial_result=None, speech_result=speech, timestamp=2.0)

        assert result.timestamp == 2.0
        assert result.facial_result is None
        assert result.speech_result is not None

    def test_fuse_both_modalities(self):
        """Test fusion with both facial and speech."""
        from emotion_detection_action.emotion.learned_fusion import LearnedEmotionFusion

        fusion = LearnedEmotionFusion()
        facial = create_facial_result(happy=0.8)
        speech = create_speech_result(happy=0.6)

        result = fusion.fuse(facial_result=facial, speech_result=speech)

        assert result.facial_result is not None
        assert result.speech_result is not None
        assert result.fusion_confidence > 0

    def test_fuse_with_attention(self):
        """Test fusion with attention metrics."""
        from emotion_detection_action.emotion.learned_fusion import LearnedEmotionFusion

        fusion = LearnedEmotionFusion()
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
        from emotion_detection_action.emotion.learned_fusion import LearnedEmotionFusion

        fusion = LearnedEmotionFusion()

        with pytest.raises(ValueError, match="At least one of facial or speech"):
            fusion.fuse(facial_result=None, speech_result=None)

    def test_emotion_order_consistency(self):
        """Test that emotion order is consistent."""
        from emotion_detection_action.emotion.learned_fusion import EMOTION_ORDER

        expected = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
        assert EMOTION_ORDER == expected


class TestLearnedFusionConfig:
    """Tests for LearnedFusionConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from emotion_detection_action.emotion.learned_fusion import LearnedFusionConfig

        config = LearnedFusionConfig()
        assert config.model_path is None
        assert config.device == "cpu"
        assert config.hidden_dims == [64, 32]
        assert config.dropout == 0.3

    def test_custom_config(self):
        """Test custom configuration."""
        from emotion_detection_action.emotion.learned_fusion import LearnedFusionConfig

        config = LearnedFusionConfig(
            model_path="path/to/model.pt",
            device="cuda",
            hidden_dims=[128, 64],
            dropout=0.5,
        )
        assert config.model_path == "path/to/model.pt"
        assert config.device == "cuda"
        assert config.hidden_dims == [128, 64]


class TestModelSaveLoad:
    """Tests for saving and loading models."""

    def test_save_load_model(self):
        """Test saving and loading model weights."""
        from emotion_detection_action.emotion.learned_fusion import (
            FusionMLP,
            load_model,
            save_model,
        )

        # Create and save model
        model = FusionMLP(hidden_dims=[32, 16])
        model.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.pt"
            save_model(model, model_path)

            assert model_path.exists()

            # Load model
            loaded_model = load_model(model_path)
            loaded_model.eval()

            # Compare outputs
            facial = torch.rand(1, 7)
            speech = torch.rand(1, 7)
            attention = torch.rand(1, 3)

            with torch.no_grad():
                original_out, _ = model(facial, speech, attention)
                loaded_out, _ = loaded_model(facial, speech, attention)

            assert torch.allclose(original_out, loaded_out)

    def test_learned_fusion_with_saved_model(self):
        """Test LearnedEmotionFusion with a saved model."""
        from emotion_detection_action.emotion.learned_fusion import (
            FusionMLP,
            LearnedEmotionFusion,
            LearnedFusionConfig,
            save_model,
        )

        # Create and save a model
        model = FusionMLP()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "fusion.pt"
            save_model(model, model_path)

            # Load via LearnedEmotionFusion
            config = LearnedFusionConfig(model_path=str(model_path))
            fusion = LearnedEmotionFusion(config=config)
            fusion.load()

            assert fusion.is_loaded

            # Test inference
            facial = create_facial_result(happy=0.9)
            result = fusion.fuse(facial_result=facial, speech_result=None)
            assert result is not None


class TestCreateUntrainedModel:
    """Tests for create_untrained_model function."""

    def test_create_untrained_model(self):
        """Test creating an untrained model."""
        from emotion_detection_action.emotion.learned_fusion import create_untrained_model

        model = create_untrained_model()
        assert model is not None
        assert model.get_num_parameters() > 0

    def test_create_untrained_model_custom(self):
        """Test creating model with custom params."""
        from emotion_detection_action.emotion.learned_fusion import create_untrained_model

        model = create_untrained_model(
            hidden_dims=[128, 64, 32],
            output_confidence=False,
        )
        assert model.output_confidence is False
