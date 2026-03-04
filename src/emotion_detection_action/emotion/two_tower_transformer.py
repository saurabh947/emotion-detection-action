"""Backward-compatible shim for the Two-Tower Multimodal Emotion Transformer.

The canonical implementation now lives in:
  - :mod:`emotion_detection_action.models.backbones`  — VideoMAE / ViViT / AST backbones
  - :mod:`emotion_detection_action.models.fusion`     — ``NeuralFusionModel`` (full model)

This module re-exports those symbols under the legacy names
``TwoTowerEmotionTransformer``, ``TwoTowerConfig``, and ``TwoTowerOutput`` so
any existing code that imported from here continues to work unchanged.

Usage (new, preferred)::

    from emotion_detection_action.models.fusion import NeuralFusionModel, NeuralModelOutput
    from emotion_detection_action.models.backbones import BackboneConfig

Usage (legacy, still works)::

    from emotion_detection_action.emotion.two_tower_transformer import (
        TwoTowerEmotionTransformer, TwoTowerConfig, TwoTowerOutput,
        EMOTION_ORDER, ATTENTION_ORDER,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# Re-export canonical names from models/
# ---------------------------------------------------------------------------

from emotion_detection_action.models.backbones import BackboneConfig
from emotion_detection_action.models.fusion import (
    NeuralFusionModel,
    NeuralModelOutput,
    _EMOTION_ORDER,
    _METRIC_ORDER,
)

# Legacy label lists (kept for backward compatibility)
EMOTION_ORDER: list[str] = _EMOTION_ORDER
# Migration note: the third metric was renamed from "nervousness" to "arousal"
# when the model was upgraded to the pure-neural pipeline.
# Old values: ["stress", "engagement", "nervousness"]
# New values: ["stress", "engagement", "arousal"]
# ATTENTION_ORDER reflects the new values; update any code that hardcoded "nervousness".
ATTENTION_ORDER: list[str] = _METRIC_ORDER


# ---------------------------------------------------------------------------
# Legacy config shim
# ---------------------------------------------------------------------------


@dataclass
class TwoTowerConfig:
    """Backward-compatible wrapper around :class:`~models.backbones.BackboneConfig`.

    All constructor arguments are forwarded to ``BackboneConfig``.  New code
    should use ``BackboneConfig`` directly.

    .. deprecated::
        Use :class:`~models.backbones.BackboneConfig` instead.
    """

    video_model_name: str = "MCG-NJU/videomae-base"
    audio_model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    pretrained: bool = True
    d_model: int = 512
    num_heads: int = 8
    ffn_dim: int = 2048
    num_cross_attn_layers: int = 2
    video_freeze_layers: int = 8
    audio_freeze_layers: int = 6
    dropout: float = 0.1
    # Internal stub geometry (ignored when pretrained=True)
    _video_hidden_size: int = 512
    _audio_hidden_size: int = 512
    _video_image_size: int = 224
    _video_num_frames: int = 16

    def to_backbone_config(self) -> BackboneConfig:
        return BackboneConfig(
            video_model_name=self.video_model_name,
            audio_model_name=self.audio_model_name,
            pretrained=self.pretrained,
            d_model=self.d_model,
            video_freeze_layers=self.video_freeze_layers,
            audio_freeze_layers=self.audio_freeze_layers,
            video_num_frames=self._video_num_frames,
            video_image_size=self._video_image_size,
        )


# ---------------------------------------------------------------------------
# Legacy output shim
# ---------------------------------------------------------------------------


@dataclass
class TwoTowerOutput:
    """Backward-compatible output wrapper.

    .. deprecated::
        Use :class:`~models.fusion.NeuralModelOutput` instead.
    """

    emotion_logits: object
    emotion_probs: object
    attention_metrics: object  # was "attention_metrics", now called "metrics"
    fused_cls: object          # was "fused_cls", now called "latent_embedding"
    video_missing: bool = False
    audio_missing: bool = False


# ---------------------------------------------------------------------------
# Legacy model shim
# ---------------------------------------------------------------------------


class TwoTowerEmotionTransformer(NeuralFusionModel):
    """Backward-compatible alias for :class:`~models.fusion.NeuralFusionModel`.

    Accepts a :class:`TwoTowerConfig` and delegates to the canonical model.
    The ``forward()`` signature is extended to also return a :class:`TwoTowerOutput`
    when ``return_legacy_output=True`` is passed.

    .. deprecated::
        Use :class:`~models.fusion.NeuralFusionModel` with
        :class:`~models.backbones.BackboneConfig` directly.
    """

    # Legacy name alias for METRIC_ORDER
    ATTENTION_ORDER: list[str] = NeuralFusionModel.METRIC_ORDER

    def __init__(self, config: TwoTowerConfig | None = None) -> None:
        cfg = config or TwoTowerConfig()
        backbone_cfg = cfg.to_backbone_config()
        super().__init__(
            config=backbone_cfg,
            num_cross_attn_layers=cfg.num_cross_attn_layers,
            num_heads=cfg.num_heads,
            ffn_dim=cfg.ffn_dim,
            dropout=cfg.dropout,
        )

    # ------------------------------------------------------------------ #
    # Backward-compatible property aliases for the absent-modality tokens #
    # ------------------------------------------------------------------ #

    @property
    def video_absent_token(self) -> "torch.nn.Parameter":  # type: ignore[name-defined]
        """Legacy alias for :attr:`absent_video`."""
        return self.absent_video  # type: ignore[return-value]

    @property
    def audio_absent_token(self) -> "torch.nn.Parameter":  # type: ignore[name-defined]
        """Legacy alias for :attr:`absent_audio`."""
        return self.absent_audio  # type: ignore[return-value]

    def forward(  # type: ignore[override]
        self,
        video_frames: object = None,
        audio_spectrograms: object = None,
        use_temporal: bool = True,
        return_legacy_output: bool = False,
    ) -> "NeuralModelOutput | TwoTowerOutput":
        out = super().forward(video_frames, audio_spectrograms, use_temporal=use_temporal)  # type: ignore[arg-type]
        if return_legacy_output:
            return TwoTowerOutput(
                emotion_logits=out.emotion_logits,
                emotion_probs=out.emotion_probs,
                attention_metrics=out.metrics,
                fused_cls=out.latent_embedding,
                video_missing=out.video_missing,
                audio_missing=out.audio_missing,
            )
        return out


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch

    print("TwoTowerEmotionTransformer (legacy shim) sanity check")

    cfg = TwoTowerConfig(pretrained=False, d_model=256, _video_num_frames=16)
    model = TwoTowerEmotionTransformer(cfg)
    model.eval()

    video = torch.randn(1, 16, 3, 64, 64)
    audio = torch.randn(1, 50, 128)

    with torch.no_grad():
        out = model(video_frames=video, audio_spectrograms=audio)

    print(f"emotion_probs : {tuple(out.emotion_probs.shape)}")
    print(f"metrics       : {tuple(out.metrics.shape)}")
    print(f"latent_emb    : {tuple(out.latent_embedding.shape)}")
    print(f"dominant      : {EMOTION_ORDER[out.emotion_probs[0].argmax()]}")

    # Legacy output format
    out_legacy = model(video, audio, return_legacy_output=True)
    assert isinstance(out_legacy, TwoTowerOutput)
    print(f"Legacy output : attention_metrics={tuple(out_legacy.attention_metrics.shape)}")
    print("All checks passed ✓")
