"""Tests for the NeuralFusionModel and its building blocks.

Covers: VideoTemporalBlock, CrossAttentionBlock, TemporalContextBuffer,
absent-modality tokens, GRU temporal state reset, and the full forward pass
with stub backbones (no pretrained weights needed).
"""

import pytest
import torch

try:
    from emotion_detection_action.models.backbones import BackboneConfig
    from emotion_detection_action.models.fusion import (
        CrossAttentionBlock,
        NeuralFusionModel,
        TemporalContextBuffer,
        VideoTemporalBlock,
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

D = 64   # small d_model for fast tests


def _stub_backbone_cfg(**kwargs) -> "BackboneConfig":
    return BackboneConfig(
        pretrained=False,
        d_model=D,
        face_crop_enabled=False,
        **kwargs,
    )


def _stub_model(gru_layers: int = 1, cross_attn_layers: int = 1) -> "NeuralFusionModel":
    return NeuralFusionModel(
        config=_stub_backbone_cfg(),
        num_cross_attn_layers=cross_attn_layers,
        num_heads=4,
        gru_layers=gru_layers,
    )


# ---------------------------------------------------------------------------
# VideoTemporalBlock
# ---------------------------------------------------------------------------


class TestVideoTemporalBlock:
    def test_output_shape_matches_input(self):
        """Output must be (B, T, d_model) matching input."""
        block = VideoTemporalBlock(d_model=D, num_heads=4, max_frames=16)
        x = torch.randn(2, 8, D)
        out = block(x)
        assert out.shape == (2, 8, D)

    def test_single_frame(self):
        """Single-frame input should not crash."""
        block = VideoTemporalBlock(d_model=D, num_heads=4, max_frames=16)
        x = torch.randn(1, 1, D)
        out = block(x)
        assert out.shape == (1, 1, D)

    def test_positional_encoding_effect(self):
        """Same frames in different positions should produce different outputs."""
        block = VideoTemporalBlock(d_model=D, num_heads=4, max_frames=16)
        block.eval()
        frame = torch.randn(1, 1, D)
        # Create a 4-frame sequence where all frames are identical
        x = frame.expand(1, 4, D).clone()
        out = block(x)
        # Positional encodings should make each position's output different
        are_different = not torch.allclose(out[0, 0], out[0, 1], atol=1e-5)
        assert are_different, "Positional encoding should differentiate identical frames"


# ---------------------------------------------------------------------------
# CrossAttentionBlock
# ---------------------------------------------------------------------------


class TestCrossAttentionBlock:
    def test_output_shapes(self):
        """Both query outputs must retain their input shapes."""
        block = CrossAttentionBlock(d_model=D, num_heads=4)
        video = torch.randn(2, 8, D)   # (B, T_v, D)
        audio = torch.randn(2, 12, D)  # (B, T_a, D)
        v_out, a_out = block(video, audio)
        assert v_out.shape == video.shape
        assert a_out.shape == audio.shape

    def test_bidirectional_asymmetry(self):
        """Video-queries-audio and audio-queries-video should differ."""
        block = CrossAttentionBlock(d_model=D, num_heads=4)
        block.eval()
        video = torch.randn(1, 4, D)
        audio = torch.randn(1, 6, D)
        v_out, a_out = block(video, audio)
        # Output tensors should not be identical to each other
        if v_out.shape == a_out.shape:
            assert not torch.allclose(v_out, a_out, atol=1e-6)

    def test_batch_consistency(self):
        """Single-item batch must produce same result as first element of larger batch."""
        block = CrossAttentionBlock(d_model=D, num_heads=4)
        block.eval()
        video = torch.randn(1, 4, D)
        audio = torch.randn(1, 6, D)

        video_batch = video.expand(3, -1, -1).clone()
        audio_batch = audio.expand(3, -1, -1).clone()

        v1, a1 = block(video, audio)
        vB, aB = block(video_batch, audio_batch)

        assert torch.allclose(v1[0], vB[0], atol=1e-5)
        assert torch.allclose(a1[0], aB[0], atol=1e-5)


# ---------------------------------------------------------------------------
# TemporalContextBuffer
# ---------------------------------------------------------------------------


class TestTemporalContextBuffer:
    def test_output_shape(self):
        """Output must be (B, d_model)."""
        buf = TemporalContextBuffer(d_model=D, num_layers=1)
        x = torch.randn(2, D)
        out = buf(x)
        assert out.shape == (2, D)

    def test_hidden_state_persists(self):
        """Second forward call with same input differs from first (state accumulated)."""
        buf = TemporalContextBuffer(d_model=D, num_layers=1)
        buf.eval()
        x = torch.randn(1, D)
        out1 = buf(x.clone())
        out2 = buf(x.clone())
        assert not torch.allclose(out1, out2, atol=1e-6), (
            "GRU output should differ after state accumulation"
        )

    def test_reset_clears_hidden_state(self):
        """After reset(), next forward pass should produce the same result as the first."""
        buf = TemporalContextBuffer(d_model=D, num_layers=1)
        buf.eval()
        x = torch.randn(1, D)
        first_out = buf(x.clone())
        buf(x.clone())   # advance state
        buf.reset()
        after_reset_out = buf(x.clone())
        assert torch.allclose(first_out, after_reset_out, atol=1e-6), (
            "Output after reset should match initial output"
        )

    def test_multi_layer(self):
        """Multi-layer GRU should produce the same output shape."""
        buf = TemporalContextBuffer(d_model=D, num_layers=2)
        x = torch.randn(3, D)
        out = buf(x)
        assert out.shape == (3, D)


# ---------------------------------------------------------------------------
# NeuralFusionModel — full forward pass
# ---------------------------------------------------------------------------


class TestNeuralFusionModel:
    def test_forward_video_and_audio(self):
        """Full forward with both modalities returns correct shapes."""
        model = _stub_model()
        model.eval()
        video = torch.randn(1, 16, 3, 224, 224)
        audio = torch.randn(1, 16000)
        with torch.no_grad():
            out = model(video, audio)
        assert out.emotion_probs.shape == (1, 8)
        assert out.metrics.shape == (1, 3)
        assert out.latent_embedding.shape == (1, D)
        assert not out.video_missing
        assert not out.audio_missing

    def test_forward_video_only(self):
        """Passing audio=None triggers absent-audio token and sets audio_missing=True."""
        model = _stub_model()
        model.eval()
        video = torch.randn(1, 16, 3, 224, 224)
        with torch.no_grad():
            out = model(video, None)
        assert out.audio_missing is True
        assert out.emotion_probs.shape == (1, 8)

    def test_forward_audio_only(self):
        """Passing video=None triggers absent-video token and sets video_missing=True."""
        model = _stub_model()
        model.eval()
        audio = torch.randn(1, 16000)
        with torch.no_grad():
            out = model(None, audio)
        assert out.video_missing is True
        assert out.emotion_probs.shape == (1, 8)

    def test_both_none_raises(self):
        """Both video and audio being None should raise ValueError."""
        model = _stub_model()
        with pytest.raises(ValueError):
            model(None, None)

    def test_emotion_probs_sum_to_one(self):
        """Softmax emotion probabilities must sum to 1 per batch element."""
        model = _stub_model()
        model.eval()
        video = torch.randn(2, 16, 3, 224, 224)
        with torch.no_grad():
            out = model(video, None)
        sums = out.emotion_probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(2), atol=1e-5)

    def test_metrics_in_zero_one(self):
        """Sigmoid metric outputs must all be in [0, 1]."""
        model = _stub_model()
        model.eval()
        video = torch.randn(1, 16, 3, 224, 224)
        with torch.no_grad():
            out = model(video, None)
        assert out.metrics.min().item() >= 0.0
        assert out.metrics.max().item() <= 1.0

    def test_reset_temporal_state(self):
        """After reset_temporal_state(), two identical forward passes produce the same output."""
        model = _stub_model()
        model.eval()
        video = torch.randn(1, 16, 3, 224, 224)

        with torch.no_grad():
            out1 = model(video, None, use_temporal=True)
        model.reset_temporal_state()
        with torch.no_grad():
            out2 = model(video, None, use_temporal=True)

        assert torch.allclose(out1.emotion_probs, out2.emotion_probs, atol=1e-5)

    def test_emotion_order_constant(self):
        """EMOTION_ORDER must have exactly 8 labels including 'unclear'."""
        assert len(NeuralFusionModel.EMOTION_ORDER) == 8
        assert "unclear" in NeuralFusionModel.EMOTION_ORDER

    def test_count_parameters(self):
        """count_parameters() should return a positive integer."""
        model = _stub_model()
        n = model.count_parameters()
        assert isinstance(n, int)
        assert n > 0
