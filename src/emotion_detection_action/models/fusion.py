"""Cross-attention fusion and temporal context buffer.

This module provides the full ``NeuralFusionModel`` — a Two-Tower Multimodal
Emotion Transformer that combines:

1. **VideoBackbone** + **AudioBackbone** (from :mod:`models.backbones`)
2. **Projection layers** — map backbone hidden states to a shared ``d_model``
3. **Absent-modality tokens** — learned substitutes when a sensor is offline
4. **VideoTemporalBlock** — intra-clip self-attention over per-frame CLS tokens
   (especially effective for AffectNet ViT where each token = one time step)
5. **Bidirectional CrossAttentionBlock** — video ↔ audio token interaction
6. **TemporalContextBuffer** — GRU rolling window for 2-second temporal memory
7. **Multi-task output heads** — emotion (8-class softmax) + metrics (3 Sigmoid)

Temporal modelling design
-------------------------
With AffectNet ViT, the video backbone returns ``(B, T, d_model)`` — one CLS
token per frame.  These tokens have a natural temporal order (frame 0 → frame
T-1) but no positional encoding beyond what the ViT applies per-frame.

``VideoTemporalBlock`` adds:

* A learned sinusoidal-style positional embedding over the T frame tokens so
  the cross-attention downstream can reason about *when* each frame occurred.
* A self-attention pass so early frames can attend to later ones (e.g., the
  beginning of a smile can attend to its peak) before the video tower queries
  the audio tower.

The existing ``TemporalContextBuffer`` (GRU) handles *inter-clip* temporal
smoothing — it persists the hidden state across successive clips during a live
session, providing a ~2-second rolling emotional context.

Outputs
-------
The model returns a :class:`NeuralModelOutput` dataclass whose ``latent_embedding``
(shape ``(B, d_model)``) is the temporally-smoothed fused representation — the
canonical input for downstream VLA models (e.g., OpenVLA).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn

from emotion_detection_action.models.backbones import AudioBackbone, BackboneConfig, VideoBackbone


# ---------------------------------------------------------------------------
# Output contract (internal — see core/types.py for the public Pydantic model)
# ---------------------------------------------------------------------------


@dataclass
class NeuralModelOutput:
    """Raw tensor output from :class:`NeuralFusionModel`.

    Attributes:
        emotion_logits: ``(B, 8)`` unnormalised logits (7 standard + unclear).
        emotion_probs: ``(B, 8)`` softmax probabilities.
        metrics: ``(B, 3)`` Sigmoid outputs for [stress, engagement, arousal].
        latent_embedding: ``(B, d_model)`` temporally-smoothed fused CLS token.
            This is the high-dimensional "raw emotional state" vector that VLAs
            like OpenVLA consume directly.
        video_missing: Whether the video modality was absent in this forward pass.
        audio_missing: Whether the audio modality was absent in this forward pass.
    """

    emotion_logits: torch.Tensor
    emotion_probs: torch.Tensor
    metrics: torch.Tensor
    latent_embedding: torch.Tensor
    video_missing: bool = False
    audio_missing: bool = False


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class _PositionWiseFFN(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VideoTemporalBlock(nn.Module):
    """Intra-clip temporal self-attention for per-frame video tokens.

    When the video backbone is AffectNet ViT, each of the T input frames
    produces exactly one CLS token.  These T tokens carry no relative
    temporal position — ``VideoTemporalBlock`` injects a learned positional
    encoding and runs a single self-attention pass so frames can attend to
    each other before the cross-modal fusion step.

    For VideoMAE / ViViT the block still applies (the ~1568 spatio-temporal
    patch tokens also benefit from explicit self-attention), though the
    benefit is less pronounced since those backbones already model temporal
    relationships internally.

    Args:
        d_model: Shared embedding dimension.
        num_heads: Attention heads (must divide ``d_model``).
        max_frames: Maximum clip length in frames (controls positional table).
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        max_frames: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pos_emb = nn.Embedding(max_frames, d_model)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, video_tokens: torch.Tensor) -> torch.Tensor:
        """Apply positional encoding + self-attention over video frame tokens.

        Args:
            video_tokens: ``(B, T, d_model)`` sequence of per-frame embeddings.

        Returns:
            ``(B, T, d_model)`` updated sequence with temporal context.
        """
        B, T, D = video_tokens.shape
        positions = torch.arange(T, device=video_tokens.device).unsqueeze(0)  # (1, T)
        x = video_tokens + self.pos_emb(positions)  # broadcast over B
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + self.dropout(attn_out))


class CrossAttentionBlock(nn.Module):
    """One layer of bidirectional cross-attention between video and audio tokens.

    Video attends over audio (V→A) **and** audio attends over video (A→V) in
    parallel, then both representations are updated with position-wise FFNs and
    layer normalisations.

    Args:
        d_model: Shared embedding dimension.
        num_heads: Number of attention heads (must divide ``d_model``).
        ffn_dim: Hidden dimension of the position-wise feed-forward networks.
        dropout: Dropout probability applied inside attention and FFN.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        ffn_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        ffn_dim = ffn_dim or d_model * 4

        self.v_attn_a = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.a_attn_v = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        self.v_ffn = _PositionWiseFFN(d_model, ffn_dim, dropout)
        self.a_ffn = _PositionWiseFFN(d_model, ffn_dim, dropout)

        self.v_norm1 = nn.LayerNorm(d_model)
        self.v_norm2 = nn.LayerNorm(d_model)
        self.a_norm1 = nn.LayerNorm(d_model)
        self.a_norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Bidirectional cross-attention update.

        Args:
            video: ``(B, Tv, d_model)`` video token sequence.
            audio: ``(B, Ta, d_model)`` audio token sequence.

        Returns:
            Updated ``(video, audio)`` tuple with the same shapes.
        """
        # Video attends to audio
        v_ctx, _ = self.v_attn_a(query=video, key=audio, value=audio)
        video = self.v_norm1(video + v_ctx)
        video = self.v_norm2(video + self.v_ffn(video))

        # Audio attends to video
        a_ctx, _ = self.a_attn_v(query=audio, key=video, value=video)
        audio = self.a_norm1(audio + a_ctx)
        audio = self.a_norm2(audio + self.a_ffn(audio))

        return video, audio


class TemporalContextBuffer(nn.Module):
    """GRU-based rolling temporal memory buffer for ~2-second context.

    At 30fps with 16-frame clips, one clip ≈ 0.53 s.  With 4 GRU steps the
    buffer covers ≈ 2.1 s of history, capturing slow emotional transitions.

    The GRU hidden state persists between calls to :meth:`forward` within a
    session.  Call :meth:`reset` at the start of each new interaction.

    Args:
        d_model: Dimensionality of the fused clip embedding (input + output).
        num_layers: Number of stacked GRU layers (2 recommended for depth).
        dropout: Recurrent dropout between layers (applied when ``num_layers > 1``).
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self._hidden: torch.Tensor | None = None
        self._lock = threading.Lock()  # guards _hidden for multi-threaded inference

    def __getstate__(self) -> dict:
        """Custom pickling: replace the unpicklable Lock with None so deepcopy works."""
        state = self.__dict__.copy()
        state["_lock"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore state after unpickling, recreating the Lock."""
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Process one clip embedding, returning the temporally-smoothed version.

        Args:
            embedding: ``(B, d_model)`` fused CLS embedding from the current clip.

        Returns:
            ``(B, d_model)`` temporally-smoothed embedding.
        """
        x = embedding.unsqueeze(1)                           # (B, 1, d_model)
        with self._lock:
            out, self._hidden = self.gru(x, self._hidden)   # hidden persists!
        return out.squeeze(1)                                # (B, d_model)

    def reset(self) -> None:
        """Reset the GRU hidden state (call at the start of each new session)."""
        with self._lock:
            self._hidden = None

    def detach_hidden(self) -> None:
        """Detach hidden state from the computation graph (for BPTT truncation)."""
        with self._lock:
            if self._hidden is not None:
                self._hidden = self._hidden.detach()


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

# Public constants — importable without the leading underscore.
EMOTION_ORDER: list[str] = [
    "angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised",
    "unclear",  # 8th class: no person present, noisy signal, or confidence too low
]
METRIC_ORDER: list[str] = ["stress", "engagement", "arousal"]

# Private aliases kept for internal use and backward-compatible imports.
_EMOTION_ORDER = EMOTION_ORDER
_METRIC_ORDER = METRIC_ORDER


class NeuralFusionModel(nn.Module):
    """Two-Tower Multimodal Emotion Transformer with temporal GRU memory.

    Complete end-to-end model combining:

    * Video backbone (AffectNet ViT **recommended**, VideoMAE or ViViT legacy)
      — see :mod:`models.backbones` for the comparison table.
    * Audio backbone (emotion2vec **recommended**, AST legacy)
    * Learnable absent-modality tokens for graceful sensor-failure handling
    * :class:`VideoTemporalBlock` — intra-clip self-attention over frame tokens
    * Stacked bidirectional :class:`CrossAttentionBlock` layers
    * :class:`TemporalContextBuffer` GRU for 2-second inter-clip rolling context
    * Multi-task heads:

      - Emotion head → 8-class softmax  (Angry · Disgusted · Fearful · Happy ·
        Neutral · Sad · Surprised · Unclear)
      - Metrics head → 3 Sigmoid outputs  (Stress · Engagement · Arousal)

    The ``latent_embedding`` output (shape ``(B, d_model)``) is the
    temporally-smoothed fused CLS token — a 512-dim vector that VLA models
    (e.g., OpenVLA) consume directly as an emotion context signal.

    Args:
        config: :class:`BackboneConfig` with all model hyper-parameters.
        num_cross_attn_layers: Number of :class:`CrossAttentionBlock` stacks.
        num_heads: Attention heads per cross-attention block.
        ffn_dim: FFN hidden size in cross-attention blocks (default 4×d_model).
        dropout: Dropout used throughout.
        gru_layers: Depth of the temporal GRU (2 recommended).
    """

    EMOTION_ORDER: list[str] = EMOTION_ORDER   # type: ignore[assignment]
    METRIC_ORDER: list[str] = METRIC_ORDER     # type: ignore[assignment]

    def __init__(
        self,
        config: BackboneConfig | None = None,
        num_cross_attn_layers: int = 2,
        num_heads: int = 8,
        ffn_dim: int | None = None,
        dropout: float = 0.1,
        gru_layers: int = 2,
    ) -> None:
        super().__init__()
        if config is None:
            config = BackboneConfig()

        self.config = config
        d_model = config.d_model

        # Temperature scalar for post-training calibration (default 1.0 = no-op).
        # Set via load_temperature() after loading a calibrated checkpoint.
        self.temperature: float = 1.0
        ffn_dim = ffn_dim or d_model * 4

        # --- Backbone towers ---
        self.video_backbone = VideoBackbone(config)
        self.audio_backbone = AudioBackbone(config)

        # --- Projection layers (backbone hidden → d_model) ---
        self.video_proj = nn.Sequential(
            nn.Linear(self.video_backbone.hidden_size, d_model),
            nn.LayerNorm(d_model),
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(self.audio_backbone.hidden_size, d_model),
            nn.LayerNorm(d_model),
        )

        # --- Absent-modality learned tokens ---
        self.absent_video = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.absent_audio = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # --- Intra-clip temporal self-attention for video frame tokens ---
        # Gives AffectNet ViT's per-frame CLS tokens positional encoding and
        # lets frames attend to each other before cross-modal fusion.
        self.video_temporal = VideoTemporalBlock(
            d_model,
            num_heads=num_heads,
            max_frames=max(64, config.video_num_frames * 2),
            dropout=dropout,
        )

        # --- Bidirectional cross-attention layers ---
        self.cross_attn = nn.ModuleList([
            CrossAttentionBlock(d_model, num_heads, ffn_dim, dropout)
            for _ in range(num_cross_attn_layers)
        ])

        # --- Temporal GRU context buffer ---
        self.temporal_buffer = TemporalContextBuffer(d_model, num_layers=gru_layers, dropout=dropout)

        # --- Multi-task output heads ---
        self.emotion_head = nn.Linear(d_model, len(EMOTION_ORDER))    # 8 classes
        self.metrics_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, len(METRIC_ORDER)),                # 3 metrics
            nn.Sigmoid(),
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        video_frames: torch.Tensor | None,
        audio_spectrograms: torch.Tensor | None,
        use_temporal: bool = True,
    ) -> NeuralModelOutput:
        """Run the full forward pass.

        Args:
            video_frames: ``(B, T, C, H, W)`` float tensor, values in ``[0, 1]``.
                For AffectNet ViT, each frame should be a pre-processed face
                crop (handled automatically by :class:`~core.detector.EmotionDetector`).
                Pass ``None`` when video is unavailable.
            audio_spectrograms: For emotion2vec / HuBERT: ``(B, samples)`` raw
                float32 waveform at 16 kHz.
                For legacy AST: ``(B, time, mel)`` log mel-spectrogram.
                Pass ``None`` when audio is unavailable.
            use_temporal: When ``True`` (default) the GRU temporal buffer is
                updated with the current clip embedding.  Set ``False`` for
                batch training (GRU is applied per-sample, not across time).

        Returns:
            :class:`NeuralModelOutput` with all output tensors.

        Raises:
            ValueError: If both modalities are ``None``.
        """
        if video_frames is None and audio_spectrograms is None:
            raise ValueError("At least one of video_frames or audio_spectrograms must be provided.")

        video_missing = video_frames is None
        audio_missing = audio_spectrograms is None

        # --- Encode video ---
        if video_frames is not None:
            v_tokens = self.video_proj(self.video_backbone(video_frames))   # (B, Tv, d)
            # Intra-clip temporal self-attention: lets frame tokens attend to
            # each other and injects positional encoding.  Most impactful for
            # AffectNet ViT (each token = one time step); still beneficial for
            # VideoMAE/ViViT patch tokens.
            v_tokens = self.video_temporal(v_tokens)                        # (B, Tv, d)
        else:
            B = audio_spectrograms.shape[0]  # type: ignore[union-attr]
            v_tokens = self.absent_video.expand(B, -1, -1)                  # (B, 1, d)

        # --- Encode audio ---
        if audio_spectrograms is not None:
            a_tokens = self.audio_proj(self.audio_backbone(audio_spectrograms))  # (B, Ta, d)
        else:
            B = video_frames.shape[0]  # type: ignore[union-attr]
            a_tokens = self.absent_audio.expand(B, -1, -1)                        # (B, 1, d)

        # --- Bidirectional cross-attention ---
        for layer in self.cross_attn:
            v_tokens, a_tokens = layer(v_tokens, a_tokens)

        # --- Mean-pool → fused CLS embedding ---
        fused = torch.cat([v_tokens, a_tokens], dim=1).mean(dim=1)  # (B, d_model)

        # --- Temporal GRU smoothing (single-sample inference mode only) ---
        if use_temporal:
            if fused.shape[0] == 1:
                fused = self.temporal_buffer(fused)
            else:
                import warnings
                warnings.warn(
                    "NeuralFusionModel: use_temporal=True is ignored for batch_size > 1. "
                    "Temporal GRU requires single-sample inference (batch_size=1). "
                    "Pass use_temporal=False during batch training to suppress this warning.",
                    stacklevel=2,
                )

        # --- Task heads ---
        emotion_logits = self.emotion_head(fused)           # (B, 8)
        # Apply temperature scaling if a calibrated checkpoint was loaded.
        # T > 1 softens over-confident distributions; T=1.0 is a no-op.
        emotion_probs = torch.softmax(emotion_logits / self.temperature, dim=-1)
        metrics = self.metrics_head(fused)                   # (B, 3)

        return NeuralModelOutput(
            emotion_logits=emotion_logits,
            emotion_probs=emotion_probs,
            metrics=metrics,
            latent_embedding=fused,
            video_missing=video_missing,
            audio_missing=audio_missing,
        )

    # ------------------------------------------------------------------
    # Temporal state
    # ------------------------------------------------------------------

    def reset_temporal_state(self) -> None:
        """Reset the GRU hidden state.

        Call this at the start of each new interaction or when the subject
        changes, so past context does not bleed into the new session.
        """
        self.temporal_buffer.reset()

    def detach_temporal_state(self) -> None:
        """Detach the GRU hidden state (use during long training sequences)."""
        self.temporal_buffer.detach_hidden()

    # ------------------------------------------------------------------
    # Quantization
    # ------------------------------------------------------------------

    def quantize(
        self,
        mode: Literal["dynamic", "static"] = "dynamic",
    ) -> "NeuralFusionModel":
        """Return an INT8-quantized copy of this model for faster CPU inference.

        ``"dynamic"`` quantization (recommended) statically quantizes weights to
        INT8 while computing activations at runtime.  No calibration data needed.

        The method automatically selects the correct quantization engine:

        * ``qnnpack`` on ARM64 / Apple Silicon (macOS arm64)
        * ``fbgemm``  on x86 / x86-64

        Args:
            mode: ``"dynamic"`` (default) or ``"static"``.

        Returns:
            A new quantized ``NeuralFusionModel`` instance (original unchanged).

        Example::

            model = NeuralFusionModel()
            model_q = model.quantize("dynamic")
            out = model_q(video, audio, use_temporal=False)
        """
        if mode != "dynamic":
            raise NotImplementedError(
                "Static quantization requires calibration data. Use mode='dynamic'."
            )

        import platform

        # Select the appropriate backend for the current platform.
        machine = platform.machine().lower()
        if machine in ("arm64", "aarch64"):
            torch.backends.quantized.engine = "qnnpack"
        else:
            try:
                torch.backends.quantized.engine = "fbgemm"
            except RuntimeError:
                torch.backends.quantized.engine = "qnnpack"

        # nn.GRU quantization is unsupported on some builds; fall back to
        # Linear-only quantization with a warning so the caller is aware.
        try:
            return torch.ao.quantization.quantize_dynamic(  # type: ignore[return-value]
                self,
                qconfig_spec={nn.Linear, nn.GRU},
                dtype=torch.qint8,
            )
        except Exception as exc:
            import warnings
            warnings.warn(
                f"NeuralFusionModel.quantize(): GRU quantization failed ({exc}). "
                "Falling back to Linear-only INT8 quantization. "
                "The temporal context buffer will remain in FP32.",
                stacklevel=2,
            )
            return torch.ao.quantization.quantize_dynamic(  # type: ignore[return-value]
                self,
                qconfig_spec={nn.Linear},
                dtype=torch.qint8,
            )

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters.

        Args:
            trainable_only: When ``True`` (default), count only parameters with
                ``requires_grad = True``.

        Returns:
            Integer parameter count.
        """
        params = (
            p for p in self.parameters()
            if not trainable_only or p.requires_grad
        )
        return sum(p.numel() for p in params)

    def freeze_backbones(self) -> None:
        """Freeze both backbone towers (keep heads + cross-attention trainable)."""
        self.video_backbone.freeze()
        self.audio_backbone.freeze()

    def unfreeze_backbones(self) -> None:
        """Unfreeze both backbone towers."""
        self.video_backbone.unfreeze()
        self.audio_backbone.unfreeze()

    def get_trainable_parameter_groups(
        self,
        backbone_lr: float = 1e-5,
        head_lr: float = 1e-4,
    ) -> list[dict]:
        """Return parameter groups with different learning rates for fine-tuning.

        Backbone parameters use ``backbone_lr``; projection, cross-attention,
        temporal buffer, and task-head parameters use ``head_lr``.

        Args:
            backbone_lr: Learning rate for backbone layers (should be small).
            head_lr: Learning rate for newly-initialised layers.

        Returns:
            List of parameter-group dicts suitable for ``torch.optim.AdamW``.

        Example::

            groups = model.get_trainable_parameter_groups(backbone_lr=1e-5, head_lr=1e-4)
            optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
        """
        backbone_params = list(self.video_backbone.parameters()) + list(
            self.audio_backbone.parameters()
        )
        backbone_ids = {id(p) for p in backbone_params}
        head_params = [p for p in self.parameters() if id(p) not in backbone_ids]
        return [
            {"params": [p for p in backbone_params if p.requires_grad], "lr": backbone_lr},
            {"params": [p for p in head_params if p.requires_grad], "lr": head_lr},
        ]


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("NeuralFusionModel — architecture sanity check")
    print("=" * 60)

    # Use affectnet_vit + emotion2vec stubs (no pretrained weights needed)
    cfg = BackboneConfig(
        pretrained=False,
        d_model=256,
        video_num_frames=16,
        video_model="affectnet_vit",
        audio_model="emotion2vec",
    )
    model = NeuralFusionModel(cfg, num_cross_attn_layers=2, num_heads=4, gru_layers=2)
    model.eval()

    B, T, C, H, W = 1, 16, 3, 64, 64
    # emotion2vec stub: (B, samples) raw waveform
    SAMPLES = 48000  # 3 s at 16 kHz

    video = torch.randn(B, T, C, H, W)
    audio = torch.randn(B, SAMPLES)

    with torch.no_grad():
        # Warm-up (initialises any lazy layers)
        _ = model(video, audio)

    print(f"\nParameters (trainable): {model.count_parameters():,}")

    with torch.no_grad():
        print("\n[Case 1] Both modalities (AffectNet ViT + emotion2vec stubs):")
        out = model(video, audio)
        assert out.emotion_probs.shape == (B, 8), out.emotion_probs.shape
        assert out.metrics.shape == (B, 3), out.metrics.shape
        assert out.latent_embedding.shape == (B, 256), out.latent_embedding.shape
        print(f"  emotion_probs   : {tuple(out.emotion_probs.shape)}  sum={out.emotion_probs.sum(-1).tolist()}")
        print(f"  metrics         : {tuple(out.metrics.shape)}  {dict(zip(NeuralFusionModel.METRIC_ORDER, out.metrics[0].tolist()))}")
        print(f"  latent_embedding: {tuple(out.latent_embedding.shape)}")

        print("\n[Case 2] Video only:")
        model.reset_temporal_state()
        out_v = model(video, None)
        print(f"  audio_missing={out_v.audio_missing}  dominant={NeuralFusionModel.EMOTION_ORDER[out_v.emotion_probs[0].argmax()]}")

        print("\n[Case 3] Audio only:")
        model.reset_temporal_state()
        out_a = model(None, audio)
        print(f"  video_missing={out_a.video_missing}  dominant={NeuralFusionModel.EMOTION_ORDER[out_a.emotion_probs[0].argmax()]}")

        print("\n[Temporal consistency — 5 clips]")
        model.reset_temporal_state()
        for i in range(5):
            v = torch.randn(1, T, C, H, W)
            a = torch.randn(1, SAMPLES)
            out_t = model(v, a, use_temporal=True)
            print(f"  clip {i+1}: dominant={NeuralFusionModel.EMOTION_ORDER[out_t.emotion_probs[0].argmax()]}")

    print("\n[Quantization]")
    model_q = model.quantize("dynamic")
    with torch.no_grad():
        out_q = model_q(video, audio, use_temporal=False)
    print(f"  Quantised model emotion_probs shape: {tuple(out_q.emotion_probs.shape)}")

    print("\nAll checks passed ✓")
