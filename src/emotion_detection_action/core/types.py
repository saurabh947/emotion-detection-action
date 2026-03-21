"""Type definitions for the emotion detector SDK.

The :class:`NeuralEmotionResult` Pydantic model is the primary output contract
of the pure-neural :class:`~core.detector.EmotionDetector`.  It exposes the
``latent_embedding`` — a 512-dim vector encoding the raw emotional state — that
VLA models (e.g., OpenVLA) can consume directly.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EmotionLabel(str, Enum):
    """Standard emotion labels.

    ``UNCLEAR`` is the 8th class.  It is predicted when no person is present,
    the signal is too noisy to classify reliably, or the model's confidence is
    below a meaningful threshold.  It is a *trained* label — label your
    "empty scene", occluded, or ambiguous samples as ``unclear`` and include
    them in Phase 1 / Phase 2 training.
    """

    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    SURPRISED = "surprised"
    DISGUSTED = "disgusted"
    NEUTRAL = "neutral"
    UNCLEAR = "unclear"


@dataclass
class ActionCommand:
    """Robot action command generated from detected emotion."""

    action_type: str
    parameters: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    raw_output: Any = None


class NeuralEmotionResult(BaseModel):
    """Output contract for the pure-neural :class:`~core.detector.EmotionDetector`.

    All fields are JSON-serialisable so results can be sent over WebSockets or
    stored in structured logs without extra conversion.

    Attributes:
        dominant_emotion: Label of the highest-probability emotion class
            (e.g. ``"happy"`` or ``"unclear"``).
        emotion_scores: Per-class softmax probabilities keyed by emotion label.
            Keys: ``angry · disgusted · fearful · happy · neutral · sad ·
            surprised · unclear``.
        latent_embedding: 512-dim float list encoding the "raw" emotional state
            *after* temporal GRU smoothing.  Feed this directly to VLA models
            (e.g., OpenVLA) as the emotion context vector.
        metrics: Continuous [0, 1] Sigmoid outputs from the neural attention head:
            ``stress``, ``engagement``, ``arousal``.  These are near-zero when
            ``dominant_emotion == "unclear"``.
        confidence: Max softmax probability — proxy for prediction certainty.
        timestamp: Frame timestamp in seconds.
        video_missing: ``True`` when no video was available for this clip.
        audio_missing: ``True`` when no audio was available for this clip.

    Usage tip — gate on the ``"unclear"`` label before acting::

        result = detector.process(clip)
        if result.dominant_emotion != "unclear":
            robot.react_to(result.dominant_emotion)
    """

    dominant_emotion: str = Field(description="Highest-probability emotion label.")
    emotion_scores: dict[str, float] = Field(
        description="Softmax probabilities for all 8 emotion classes "
                    "(7 standard emotions + unclear)."
    )
    latent_embedding: list[float] = Field(
        description="512-dim GRU-smoothed fused embedding for VLA integration."
    )
    metrics: dict[str, float] = Field(
        description="Neural attention metrics: stress, engagement, arousal in [0, 1]."
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Max emotion softmax score.")
    timestamp: float = Field(default=0.0, description="Frame timestamp (seconds).")
    video_missing: bool = Field(default=False)
    audio_missing: bool = Field(default=False)

    model_config = {"frozen": True}
