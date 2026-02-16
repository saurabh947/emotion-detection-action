"""Tests for configuration management."""

import pytest

from emotion_detection_action.core.config import Config, ModelConfig


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_creation(self):
        """Test creating model config."""
        config = ModelConfig(model_id="test/model")
        assert config.model_id == "test/model"
        assert config.device == "cpu"
        assert config.dtype == "float32"

    def test_with_options(self):
        """Test model config with options."""
        config = ModelConfig(
            model_id="test/model",
            device="cuda",
            dtype="float16",
            load_in_8bit=True,
        )
        assert config.device == "cuda"
        assert config.dtype == "float16"
        assert config.load_in_8bit is True


class TestConfig:
    """Tests for main Config class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = Config()
        assert config.vla_model == "openvla/openvla-7b"
        assert config.device == "cuda"
        assert config.face_detection_threshold == 0.5
        assert config.face_detection_model == "mediapipe"

    def test_invalid_facial_weight(self):
        """Test validation of facial_weight."""
        with pytest.raises(ValueError, match="facial_weight must be between 0 and 1"):
            Config(facial_weight=1.5)

        with pytest.raises(ValueError, match="facial_weight must be between 0 and 1"):
            Config(facial_weight=-0.1)

    def test_invalid_speech_weight(self):
        """Test validation of speech_weight."""
        with pytest.raises(ValueError, match="speech_weight must be between 0 and 1"):
            Config(speech_weight=2.0)

    def test_invalid_face_detection_threshold(self):
        """Test validation of face_detection_threshold."""
        with pytest.raises(ValueError, match="face_detection_threshold must be between 0 and 1"):
            Config(face_detection_threshold=1.1)

    def test_invalid_vad_aggressiveness(self):
        """Test validation of vad_aggressiveness."""
        with pytest.raises(ValueError, match="vad_aggressiveness must be 0, 1, 2, or 3"):
            Config(vad_aggressiveness=5)

    def test_get_face_detection_config(self):
        """Test generating face detection model config."""
        config = Config(
            device="cuda",
            dtype="float16",
            face_detection_model="mediapipe-full",
            face_detection_threshold=0.6,
        )
        model_config = config.get_face_detection_config()
        assert model_config.model_id == "mediapipe-full"
        assert model_config.device == "cuda"
        assert model_config.extra_kwargs["threshold"] == 0.6

    def test_get_vla_config(self):
        """Test generating VLA model config."""
        config = Config(vla_model="custom/vla-model")
        vla_config = config.get_vla_config()
        assert vla_config.model_id == "custom/vla-model"
        assert vla_config.load_in_8bit is True  # Default for VLA

    def test_from_dict(self):
        """Test creating config from dictionary."""
        d = {
            "device": "cpu",
            "face_detection_threshold": 0.8,
            "unknown_key": "ignored",  # Should be ignored
        }
        config = Config.from_dict(d)
        assert config.device == "cpu"
        assert config.face_detection_threshold == 0.8

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = Config(device="cpu")
        d = config.to_dict()
        assert d["device"] == "cpu"
        assert "vla_model" in d

    def test_fusion_settings(self):
        """Test fusion-related settings."""
        config = Config(
            fusion_strategy="weighted",
            facial_weight=0.7,
            speech_weight=0.3,
        )
        assert config.fusion_strategy == "weighted"
        assert config.facial_weight == 0.7
        assert config.speech_weight == 0.3
