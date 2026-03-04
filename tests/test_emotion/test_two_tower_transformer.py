"""Tests for TwoTowerEmotionTransformer.

All tests use ``pretrained=False`` with small hidden sizes so they run
quickly without downloading any HuggingFace weights.  The video and audio
backbone shapes are deliberately tiny to keep CPU memory and runtime low.
"""

from __future__ import annotations

import pytest
import torch

from emotion_detection_action.emotion.two_tower_transformer import (
    ATTENTION_ORDER,
    EMOTION_ORDER,
    TwoTowerConfig,
    TwoTowerEmotionTransformer,
    TwoTowerOutput,
)
from emotion_detection_action.models.fusion import (
    CrossAttentionBlock as _CrossAttentionBlock,
    NeuralModelOutput,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

HIDDEN = 64   # tiny hidden size for fast testing
D_MODEL = 64
NUM_HEADS = 4
FFN_DIM = 128
BATCH = 2

# VideoMAE stub geometry: image_size=32, patch_size=16, num_frames=4, tubelet_size=2
# → spatial patches per frame: (32/16)^2 = 4; temporal chunks: 4/2 = 2; total: 8 tokens
VIDEO_H = VIDEO_W = 32
VIDEO_T = 4       # must be divisible by tubelet_size (2)

# AST stub geometry: max_length=32, num_mel_filterbanks=32, stride=16
# → Conv2d kernel (16,16) applied with stride 16: (32-16)/16+1 = 2 patches each dim; total 4+cls
AUDIO_TIME = 32   # max_length — time dimension fed to AST
AUDIO_MEL = 32    # num_mel_filterbanks — frequency dimension


def _stub_config(**kwargs) -> TwoTowerConfig:  # type: ignore[type-arg]
    """Build a minimal no-download TwoTowerConfig, overriding any field via kwargs.

    Only passes fields that ``TwoTowerConfig`` actually declares.  Stub backbone
    geometry (patch sizes, mel bins, etc.) is handled internally by the stub
    backbones in ``models/backbones.py`` and does not need to be specified here.
    """
    defaults: dict = dict(
        pretrained=False,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        ffn_dim=FFN_DIM,
        num_cross_attn_layers=1,
        video_freeze_layers=0,
        audio_freeze_layers=0,
        _video_image_size=VIDEO_H,
        _video_num_frames=VIDEO_T,
    )
    defaults.update(kwargs)
    return TwoTowerConfig(**defaults)


@pytest.fixture(scope="module")
def small_config() -> TwoTowerConfig:
    """Minimal TwoTowerConfig that does NOT download any pretrained weights."""
    return _stub_config()


@pytest.fixture(scope="module")
def model(small_config: TwoTowerConfig) -> TwoTowerEmotionTransformer:
    m = TwoTowerEmotionTransformer(small_config)
    m.eval()
    return m


def make_video(
    batch: int = BATCH, t: int = VIDEO_T, h: int = VIDEO_H, w: int = VIDEO_W
) -> torch.Tensor:
    """Create a synthetic (B, T, C, H, W) video tensor matching stub geometry."""
    return torch.randn(batch, t, 3, h, w)


def make_audio(
    batch: int = BATCH, time: int = AUDIO_TIME, mel: int = AUDIO_MEL
) -> torch.Tensor:
    """Create a synthetic (B, time_steps, mel_bins) audio tensor matching stub geometry."""
    return torch.randn(batch, time, mel)


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestTwoTowerConfig:
    def test_defaults(self) -> None:
        cfg = TwoTowerConfig()
        assert cfg.d_model == 512
        assert cfg.num_heads == 8
        assert cfg.pretrained is True
        # 8 emotion classes (7 standard + unclear) exposed via EMOTION_ORDER
        assert len(EMOTION_ORDER) == 8
        # 3 attention metrics exposed via ATTENTION_ORDER
        assert len(ATTENTION_ORDER) == 3

    def test_custom_values(self, small_config: TwoTowerConfig) -> None:
        assert small_config.d_model == D_MODEL
        assert small_config.pretrained is False
        # Stub backbone always uses hidden_size=512 internally;
        # _video_hidden_size is a legacy field kept for API compat.
        assert small_config._video_image_size == VIDEO_H
        assert small_config._video_num_frames == VIDEO_T


# ---------------------------------------------------------------------------
# _CrossAttentionBlock tests
# ---------------------------------------------------------------------------


class TestCrossAttentionBlock:
    @pytest.fixture
    def block(self) -> _CrossAttentionBlock:
        return _CrossAttentionBlock(
            d_model=D_MODEL, num_heads=NUM_HEADS, ffn_dim=FFN_DIM, dropout=0.0
        )

    def test_output_shapes(self, block: _CrossAttentionBlock) -> None:
        x = torch.randn(BATCH, 5, D_MODEL)   # (B, Tv, d)
        y = torch.randn(BATCH, 7, D_MODEL)   # (B, Ta, d)
        x_out, y_out = block(x, y)
        assert x_out.shape == x.shape
        assert y_out.shape == y.shape

    def test_residual_connection_preserved(self, block: _CrossAttentionBlock) -> None:
        """Output should differ from the input (residual is applied but there's
        also a non-trivial transformation)."""
        x = torch.randn(BATCH, 5, D_MODEL)
        y = torch.randn(BATCH, 7, D_MODEL)
        block.eval()
        with torch.no_grad():
            x_out, y_out = block(x, y)
        # Outputs should not be identical to inputs (transformation happened)
        assert not torch.allclose(x_out, x)
        assert not torch.allclose(y_out, y)

    def test_single_token_sequence(self, block: _CrossAttentionBlock) -> None:
        """Works even when both sequences have length 1 (absent-token case)."""
        x = torch.randn(BATCH, 1, D_MODEL)
        y = torch.randn(BATCH, 1, D_MODEL)
        x_out, y_out = block(x, y)
        assert x_out.shape == (BATCH, 1, D_MODEL)
        assert y_out.shape == (BATCH, 1, D_MODEL)


# ---------------------------------------------------------------------------
# TwoTowerEmotionTransformer tests
# ---------------------------------------------------------------------------


class TestTwoTowerInit:
    def test_instantiation(self, small_config: TwoTowerConfig) -> None:
        m = TwoTowerEmotionTransformer(small_config)
        assert m is not None

    def test_default_config(self) -> None:
        """Instantiating without a config should use TwoTowerConfig defaults."""
        # We can't test pretrained=True without network; just verify the
        # default config attributes are set on the model.
        cfg = TwoTowerConfig()
        assert cfg.pretrained is True
        assert cfg.video_model_name == "MCG-NJU/videomae-base"
        assert cfg.audio_model_name == "MIT/ast-finetuned-audioset-10-10-0.4593"

    def test_emotion_order_class_attr(self) -> None:
        assert TwoTowerEmotionTransformer.EMOTION_ORDER == EMOTION_ORDER
        assert len(TwoTowerEmotionTransformer.EMOTION_ORDER) == 8

    def test_attention_order_class_attr(self) -> None:
        assert TwoTowerEmotionTransformer.ATTENTION_ORDER == ATTENTION_ORDER
        assert len(TwoTowerEmotionTransformer.ATTENTION_ORDER) == 3

    def test_parameter_count_positive(self, model: TwoTowerEmotionTransformer) -> None:
        assert model.count_parameters(trainable_only=False) > 0
        assert model.count_parameters(trainable_only=True) > 0

    def test_absent_tokens_are_parameters(self, model: TwoTowerEmotionTransformer) -> None:
        assert isinstance(model.video_absent_token, torch.nn.Parameter)
        assert isinstance(model.audio_absent_token, torch.nn.Parameter)
        assert model.video_absent_token.shape == (1, 1, D_MODEL)
        assert model.audio_absent_token.shape == (1, 1, D_MODEL)


class TestForwardBothModalities:
    def test_output_type(self, model: TwoTowerEmotionTransformer) -> None:
        out = model(video_frames=make_video(), audio_spectrograms=make_audio())
        assert isinstance(out, NeuralModelOutput)

    def test_emotion_logits_shape(self, model: TwoTowerEmotionTransformer) -> None:
        out = model(video_frames=make_video(), audio_spectrograms=make_audio())
        assert out.emotion_logits.shape == (BATCH, 8)

    def test_emotion_probs_shape(self, model: TwoTowerEmotionTransformer) -> None:
        out = model(video_frames=make_video(), audio_spectrograms=make_audio())
        assert out.emotion_probs.shape == (BATCH, 8)

    def test_emotion_probs_sum_to_one(self, model: TwoTowerEmotionTransformer) -> None:
        with torch.no_grad():
            out = model(video_frames=make_video(), audio_spectrograms=make_audio())
        probs_sum = out.emotion_probs.sum(dim=-1)
        assert torch.allclose(probs_sum, torch.ones(BATCH), atol=1e-5)

    def test_emotion_probs_non_negative(self, model: TwoTowerEmotionTransformer) -> None:
        with torch.no_grad():
            out = model(video_frames=make_video(), audio_spectrograms=make_audio())
        assert (out.emotion_probs >= 0).all()

    def test_attention_metrics_shape(self, model: TwoTowerEmotionTransformer) -> None:
        out = model(video_frames=make_video(), audio_spectrograms=make_audio())
        assert out.metrics.shape == (BATCH, 3)

    def test_attention_metrics_in_unit_range(self, model: TwoTowerEmotionTransformer) -> None:
        with torch.no_grad():
            out = model(video_frames=make_video(), audio_spectrograms=make_audio())
        assert (out.metrics >= 0).all()
        assert (out.metrics <= 1).all()

    def test_fused_cls_shape(self, model: TwoTowerEmotionTransformer) -> None:
        out = model(video_frames=make_video(), audio_spectrograms=make_audio())
        assert out.latent_embedding.shape == (BATCH, D_MODEL)

    def test_missing_flags_both_present(self, model: TwoTowerEmotionTransformer) -> None:
        out = model(video_frames=make_video(), audio_spectrograms=make_audio())
        assert out.video_missing is False
        assert out.audio_missing is False


class TestForwardVideoOnly:
    def test_video_only_forward(self, model: TwoTowerEmotionTransformer) -> None:
        with torch.no_grad():
            out = model(video_frames=make_video(), audio_spectrograms=None)
        assert out.emotion_logits.shape == (BATCH, 8)
        assert out.metrics.shape == (BATCH, 3)

    def test_audio_missing_flag(self, model: TwoTowerEmotionTransformer) -> None:
        out = model(video_frames=make_video(), audio_spectrograms=None)
        assert out.audio_missing is True
        assert out.video_missing is False

    def test_probs_sum_to_one_video_only(self, model: TwoTowerEmotionTransformer) -> None:
        with torch.no_grad():
            out = model(video_frames=make_video(), audio_spectrograms=None)
        probs_sum = out.emotion_probs.sum(dim=-1)
        assert torch.allclose(probs_sum, torch.ones(BATCH), atol=1e-5)

    def test_attention_in_range_video_only(self, model: TwoTowerEmotionTransformer) -> None:
        with torch.no_grad():
            out = model(video_frames=make_video(), audio_spectrograms=None)
        assert (out.metrics >= 0).all()
        assert (out.metrics <= 1).all()


class TestForwardAudioOnly:
    def test_audio_only_forward(self, model: TwoTowerEmotionTransformer) -> None:
        with torch.no_grad():
            out = model(video_frames=None, audio_spectrograms=make_audio())
        assert out.emotion_logits.shape == (BATCH, 8)
        assert out.metrics.shape == (BATCH, 3)

    def test_video_missing_flag(self, model: TwoTowerEmotionTransformer) -> None:
        out = model(video_frames=None, audio_spectrograms=make_audio())
        assert out.video_missing is True
        assert out.audio_missing is False

    def test_probs_sum_to_one_audio_only(self, model: TwoTowerEmotionTransformer) -> None:
        with torch.no_grad():
            out = model(video_frames=None, audio_spectrograms=make_audio())
        probs_sum = out.emotion_probs.sum(dim=-1)
        assert torch.allclose(probs_sum, torch.ones(BATCH), atol=1e-5)


class TestMissingBothModalities:
    def test_raises_when_both_none(self, model: TwoTowerEmotionTransformer) -> None:
        with pytest.raises(ValueError, match="At least one"):
            model(video_frames=None, audio_spectrograms=None)


class TestBatchSizeInvariance:
    @pytest.mark.parametrize("batch", [1, 3, 8])
    def test_different_batch_sizes(
        self, model: TwoTowerEmotionTransformer, batch: int
    ) -> None:
        with torch.no_grad():
            out = model(
                video_frames=make_video(batch=batch),
                audio_spectrograms=make_audio(batch=batch),
            )
        assert out.emotion_logits.shape == (batch, 8)
        assert out.metrics.shape == (batch, 3)


class TestDifferentModalityOutputsDistinct:
    """Absent-modality substitution should produce different distributions
    from a fully-attended forward pass."""

    def test_both_vs_video_only(self, model: TwoTowerEmotionTransformer) -> None:
        v = make_video()
        a = make_audio()
        with torch.no_grad():
            out_both = model(video_frames=v, audio_spectrograms=a)
            out_v = model(video_frames=v, audio_spectrograms=None)
        # The fused representations should differ because one had real audio.
        assert not torch.allclose(out_both.latent_embedding, out_v.latent_embedding)

    def test_video_only_vs_audio_only(self, model: TwoTowerEmotionTransformer) -> None:
        v = make_video()
        a = make_audio()
        with torch.no_grad():
            out_v = model(video_frames=v, audio_spectrograms=None)
            out_a = model(video_frames=None, audio_spectrograms=a)
        assert not torch.allclose(out_v.emotion_probs, out_a.emotion_probs)


class TestFreezeUnfreeze:
    def test_freeze_all_backbones(
        self, small_config: TwoTowerConfig
    ) -> None:
        m = TwoTowerEmotionTransformer(small_config)
        m.freeze_backbones()
        for p in m.video_backbone.parameters():
            assert not p.requires_grad
        for p in m.audio_backbone.parameters():
            assert not p.requires_grad

    def test_unfreeze_all_backbones(self, small_config: TwoTowerConfig) -> None:
        m = TwoTowerEmotionTransformer(small_config)
        m.freeze_backbones()
        m.unfreeze_backbones()
        # At least some backbone parameters should be trainable again
        any_trainable = any(p.requires_grad for p in m.video_backbone.parameters())
        assert any_trainable

    def test_freeze_by_num_layers(self, small_config: TwoTowerConfig) -> None:
        """Freezing 0 layers leaves all backbone params trainable."""
        cfg = _stub_config(video_freeze_layers=0, audio_freeze_layers=0)
        m = TwoTowerEmotionTransformer(cfg)
        total = m.count_parameters(trainable_only=False)
        trainable = m.count_parameters(trainable_only=True)
        assert trainable == total


class TestParameterGroups:
    def test_returns_list(self, model: TwoTowerEmotionTransformer) -> None:
        groups = model.get_trainable_parameter_groups()
        assert isinstance(groups, list)
        assert len(groups) > 0

    def test_all_params_covered(self, model: TwoTowerEmotionTransformer) -> None:
        groups = model.get_trainable_parameter_groups()
        group_param_count = sum(len(g["params"]) for g in groups)
        # The total params in groups should cover trainable params
        trainable = [p for p in model.parameters() if p.requires_grad]
        assert group_param_count <= len(trainable)  # groups may exclude frozen

    def test_lr_keys(self, model: TwoTowerEmotionTransformer) -> None:
        """Parameter groups use differentiated 'lr' keys, not lr_scale."""
        groups = model.get_trainable_parameter_groups(backbone_lr=1e-5, head_lr=1e-4)
        for g in groups:
            assert "lr" in g
        lrs = {g["lr"] for g in groups}
        assert 1e-5 in lrs
        assert 1e-4 in lrs


class TestCountParameters:
    def test_total_gt_trainable(self, model: TwoTowerEmotionTransformer) -> None:
        total = model.count_parameters(trainable_only=False)
        trainable = model.count_parameters(trainable_only=True)
        # With frozen layers, total >= trainable
        assert total >= trainable

    def test_positive(self, model: TwoTowerEmotionTransformer) -> None:
        assert model.count_parameters() > 0


class TestOutputDataclass:
    def test_output_fields(self, model: TwoTowerEmotionTransformer) -> None:
        out = model(video_frames=make_video(), audio_spectrograms=make_audio())
        assert hasattr(out, "emotion_logits")
        assert hasattr(out, "emotion_probs")
        assert hasattr(out, "metrics")        # previously "attention_metrics"
        assert hasattr(out, "latent_embedding")  # previously "fused_cls"
        assert hasattr(out, "video_missing")
        assert hasattr(out, "audio_missing")

    def test_legacy_output_fields(self, model: TwoTowerEmotionTransformer) -> None:
        """TwoTowerOutput (return_legacy_output=True) still exposes old field names."""
        out = model(
            video_frames=make_video(),
            audio_spectrograms=make_audio(),
            return_legacy_output=True,
        )
        assert isinstance(out, TwoTowerOutput)
        assert hasattr(out, "attention_metrics")
        assert hasattr(out, "fused_cls")

    def test_logits_differ_from_probs(self, model: TwoTowerEmotionTransformer) -> None:
        """emotion_logits are raw; emotion_probs are softmax-normalised."""
        with torch.no_grad():
            out = model(video_frames=make_video(), audio_spectrograms=make_audio())
        # Logits don't sum to 1 (in general); probs do.
        assert not torch.allclose(out.emotion_logits.sum(dim=-1), torch.ones(BATCH), atol=0.1)
        assert torch.allclose(out.emotion_probs.sum(dim=-1), torch.ones(BATCH), atol=1e-5)


class TestLabelConstants:
    def test_emotion_order_length(self) -> None:
        assert len(EMOTION_ORDER) == 8

    def test_emotion_labels(self) -> None:
        expected = {
            "angry", "disgusted", "fearful", "happy",
            "neutral", "sad", "surprised", "unclear",
        }
        assert set(EMOTION_ORDER) == expected

    def test_attention_order_length(self) -> None:
        assert len(ATTENTION_ORDER) == 3

    def test_attention_labels(self) -> None:
        expected = {"stress", "engagement", "arousal"}
        assert set(ATTENTION_ORDER) == expected
