"""Tests for configuration management."""

import pytest

from emotion_detection_action.core.config import Config, ModelConfig


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_creation(self):
        config = ModelConfig(model_id="test/model")
        assert config.model_id == "test/model"
        assert config.device == "cpu"
        assert config.dtype == "float32"

    def test_with_quantisation_options(self):
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
        config = Config()
        assert config.vla_model == "openvla/openvla-7b"
        assert config.device == "cpu"
        assert config.sample_rate == 16000
        assert config.two_tower_pretrained is True
        assert config.two_tower_device == "cpu"
        assert config.two_tower_d_model == 512
        # New defaults: AffectNet ViT + emotion2vec
        assert config.two_tower_video_model == "affectnet_vit"
        assert config.two_tower_audio_model == "emotion2vec"
        assert config.two_tower_face_crop_enabled is True

    def test_get_vla_config(self):
        config = Config(vla_model="custom/vla-model")
        vla_config = config.get_vla_config()
        assert vla_config.model_id == "custom/vla-model"
        assert vla_config.load_in_8bit is True

    def test_from_dict(self):
        d = {
            "device": "cpu",
            "sample_rate": 22050,
            "unknown_key": "ignored",
        }
        config = Config.from_dict(d)
        assert config.device == "cpu"
        assert config.sample_rate == 22050

    def test_to_dict(self):
        config = Config(device="cpu")
        d = config.to_dict()
        assert d["device"] == "cpu"
        assert "vla_model" in d
        assert "two_tower_pretrained" in d

    def test_invalid_sample_rate(self):
        with pytest.raises(ValueError, match="sample_rate must be >= 1"):
            Config(sample_rate=0)

    def test_invalid_d_model_not_divisible(self):
        with pytest.raises(ValueError, match="divisible by 8"):
            Config(two_tower_d_model=513)

    def test_neural_pipeline_settings(self):
        config = Config(
            two_tower_pretrained=False,
            two_tower_device="cpu",
            two_tower_video_model="videomae",
            two_tower_audio_model="ast",
            two_tower_model_path="outputs/phase2_best.pt",
        )
        assert config.two_tower_pretrained is False
        assert config.two_tower_device == "cpu"
        assert config.two_tower_video_model == "videomae"
        assert config.two_tower_audio_model == "ast"
        assert config.two_tower_model_path == "outputs/phase2_best.pt"

    def test_vivit_auto_sets_frame_count(self):
        """ViViT backbone auto-sets two_tower_video_frames to 32."""
        config = Config(two_tower_video_model="vivit")
        assert config.two_tower_video_frames == 32

    def test_videomae_default_frames(self):
        """VideoMAE uses 16 frames by default."""
        config = Config(two_tower_video_model="videomae")
        assert config.two_tower_video_frames == 16

    def test_affectnet_vit_default_frames(self):
        """AffectNet ViT uses 16 frames (single images, stacked)."""
        config = Config(two_tower_video_model="affectnet_vit")
        assert config.two_tower_video_frames == 16

    def test_face_crop_settings(self):
        config = Config(
            two_tower_face_crop_enabled=True,
            two_tower_face_crop_margin=0.3,
            two_tower_face_min_confidence=0.6,
        )
        assert config.two_tower_face_crop_enabled is True
        assert config.two_tower_face_crop_margin == 0.3
        assert config.two_tower_face_min_confidence == 0.6

    def test_invalid_face_crop_margin(self):
        with pytest.raises(ValueError, match="face_crop_margin"):
            Config(two_tower_face_crop_margin=1.5)

    def test_class_weights_valid(self):
        weights = [5.70, 35.50, 20.50, 1.00, 1.90, 5.50, 9.60, 20.50]
        config = Config(two_tower_emotion_class_weights=weights)
        assert config.two_tower_emotion_class_weights == weights

    def test_class_weights_wrong_length(self):
        with pytest.raises(ValueError, match="exactly 8"):
            Config(two_tower_emotion_class_weights=[1.0, 2.0, 3.0])

    def test_class_weights_negative(self):
        with pytest.raises(ValueError, match="> 0"):
            Config(two_tower_emotion_class_weights=[1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    def test_vla_disabled(self):
        config = Config(vla_enabled=False)
        assert config.vla_enabled is False
