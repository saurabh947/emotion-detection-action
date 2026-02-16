"""Multimodal emotion fusion module."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from emotion_detection_action.core.types import (
    AttentionResult,
    EmotionResult,
    EmotionScores,
    FacialEmotionResult,
    SpeechEmotionResult,
)

if TYPE_CHECKING:
    from emotion_detection_action.emotion.learned_fusion import LearnedEmotionFusion


class EmotionFusion:
    """Fuses facial, speech, and attention analysis results.

    Combines emotion predictions from multiple modalities (visual, audio, and
    attention) into a unified emotion result using various fusion strategies.

    Strategies:
    - "weighted": Weighted average based on visual/audio weights
    - "confidence": Weight by model confidence scores (default)
    - "learned": Use a trained neural network for fusion

    Attention analysis affects the fusion by:
    - High stress amplifies negative emotions (sad, angry, fearful)
    - Low engagement reduces overall confidence
    - High nervousness increases fearful/anxious signals

    Example:
        >>> fusion = EmotionFusion(strategy="confidence", confidence_threshold=0.3)
        >>> result = fusion.fuse(facial_result, speech_result, attention_result, timestamp=1.5)
        >>> print(result.emotions.dominant_emotion)

        # Using learned fusion with a trained model
        >>> fusion = EmotionFusion(strategy="learned", learned_model_path="models/fusion.pt")
        >>> result = fusion.fuse(facial_result, speech_result, attention_result)
    """

    def __init__(
        self,
        strategy: Literal["weighted", "confidence", "learned"] = "confidence",
        visual_weight: float = 0.6,
        audio_weight: float = 0.4,
        confidence_threshold: float = 0.3,
        attention_weight: float = 0.2,
        attention_stress_amplification: float = 1.5,
        attention_engagement_threshold: float = 0.3,
        learned_model_path: str | Path | None = None,
        learned_model_device: str = "cpu",
    ) -> None:
        """Initialize the fusion module.

        Args:
            strategy: Fusion strategy to use.
                - "weighted": Weighted average based on visual/audio weights
                - "confidence": Weight by model confidence scores (default)
                - "learned": Use a trained neural network for fusion
            visual_weight: Weight for visual (facial) emotions (0-1).
            audio_weight: Weight for audio (speech) emotions (0-1).
            confidence_threshold: Minimum confidence to include a modality in fusion.
                Results below this threshold are ignored. Set to 0 to disable.
            attention_weight: How much attention affects the final result (0-1).
            attention_stress_amplification: Factor to amplify negative emotions under stress.
            attention_engagement_threshold: Below this engagement, reduce confidence.
            learned_model_path: Path to trained fusion model weights (for "learned" strategy).
            learned_model_device: Device to run learned model on ("cpu", "cuda", "mps").
        """
        self.strategy = strategy
        self.visual_weight = visual_weight
        self.audio_weight = audio_weight
        self.confidence_threshold = confidence_threshold
        self.attention_weight = attention_weight
        self.attention_stress_amplification = attention_stress_amplification
        self.attention_engagement_threshold = attention_engagement_threshold
        self.learned_model_path = learned_model_path
        self.learned_model_device = learned_model_device

        # Normalize weights
        total = visual_weight + audio_weight
        if total > 0:
            self.visual_weight = visual_weight / total
            self.audio_weight = audio_weight / total

        # Initialize learned fusion model if needed
        self._learned_fusion: LearnedEmotionFusion | None = None
        if strategy == "learned":
            self._init_learned_fusion()

    def _init_learned_fusion(self) -> None:
        """Initialize the learned fusion model."""
        from emotion_detection_action.emotion.learned_fusion import (
            LearnedEmotionFusion,
            LearnedFusionConfig,
        )

        config = LearnedFusionConfig(
            model_path=str(self.learned_model_path) if self.learned_model_path else None,
            device=self.learned_model_device,
        )
        self._learned_fusion = LearnedEmotionFusion(config=config)
        self._learned_fusion.load()

    def fuse(
        self,
        facial_result: FacialEmotionResult | None = None,
        speech_result: SpeechEmotionResult | None = None,
        attention_result: AttentionResult | None = None,
        timestamp: float = 0.0,
    ) -> EmotionResult:
        """Fuse emotion results from multiple modalities.

        Applies confidence thresholding: results with confidence below
        the threshold are treated as if they don't exist.

        Attention analysis modifies the final result by:
        - Amplifying negative emotions when stress is detected
        - Reducing confidence when engagement is low
        - Adding nervousness signals to fearful emotions

        Args:
            facial_result: Facial emotion recognition result.
            speech_result: Speech emotion recognition result.
            attention_result: Attention analysis result (optional).
            timestamp: Timestamp for the fused result.

        Returns:
            Fused emotion result.

        Raises:
            ValueError: If no results pass the confidence threshold.
        """
        # Use learned fusion if strategy is "learned"
        if self.strategy == "learned" and self._learned_fusion is not None:
            return self._learned_fusion.fuse(
                facial_result=facial_result,
                speech_result=speech_result,
                attention_result=attention_result,
                timestamp=timestamp,
            )

        # Apply confidence thresholding
        facial_valid = (
            facial_result is not None and 
            facial_result.confidence >= self.confidence_threshold
        )
        speech_valid = (
            speech_result is not None and 
            speech_result.confidence >= self.confidence_threshold
        )

        # Use original results only if they pass threshold
        facial_to_use = facial_result if facial_valid else None
        speech_to_use = speech_result if speech_valid else None

        if facial_to_use is None and speech_to_use is None:
            raise ValueError(
                f"No emotion results pass confidence threshold ({self.confidence_threshold}). "
                f"Facial: {facial_result.confidence if facial_result else 'N/A'}, "
                f"Speech: {speech_result.confidence if speech_result else 'N/A'}"
            )

        # Single modality cases (after thresholding)
        if facial_to_use is None and speech_to_use is not None:
            emotions = speech_to_use.emotions
            base_confidence = speech_to_use.confidence
        elif speech_to_use is None and facial_to_use is not None:
            emotions = facial_to_use.emotions
            base_confidence = facial_to_use.confidence
        else:
            # Both modalities available and pass threshold - fuse them
            assert facial_to_use is not None and speech_to_use is not None

            if self.strategy == "weighted":
                emotions = self._fuse_weighted(facial_to_use, speech_to_use)
            else:  # "confidence" (default)
                emotions = self._fuse_confidence(facial_to_use, speech_to_use)

            # Calculate fusion confidence
            base_confidence = self._calculate_fusion_confidence(
                facial_to_use, speech_to_use
            )

        # Apply attention-based modifications
        if attention_result is not None and attention_result.confidence > 0:
            emotions = self._apply_attention_modulation(emotions, attention_result)
            base_confidence = self._apply_engagement_adjustment(
                base_confidence, attention_result
            )

        return EmotionResult(
            timestamp=timestamp,
            emotions=emotions,
            facial_result=facial_to_use if facial_valid else facial_result,
            speech_result=speech_to_use if speech_valid else speech_result,
            attention_result=attention_result,
            fusion_confidence=base_confidence,
        )

    def _apply_attention_modulation(
        self,
        emotions: EmotionScores,
        attention: AttentionResult,
    ) -> EmotionScores:
        """Apply attention-based modulation to emotion scores.

        High stress amplifies negative emotions.
        High nervousness increases fearful signals.

        Args:
            emotions: Base emotion scores.
            attention: Attention analysis result.

        Returns:
            Modulated emotion scores.
        """
        scores = emotions.to_dict()
        stress = attention.metrics.stress_score
        nervousness = attention.metrics.nervousness_score

        # Negative emotions to amplify under stress
        negative_emotions = ["sad", "angry", "fearful", "disgusted"]

        # Apply stress amplification to negative emotions
        if stress > 0.3:  # Only apply if stress is notable
            amplification = 1.0 + (stress * (self.attention_stress_amplification - 1.0))
            amplification *= self.attention_weight  # Scale by attention weight

            for emotion in negative_emotions:
                if emotion in scores:
                    scores[emotion] = min(1.0, scores[emotion] * (1.0 + (amplification - 1.0) * 0.5))

            # Slightly reduce positive emotions under stress
            scores["happy"] = scores["happy"] * (1.0 - stress * self.attention_weight * 0.3)

        # Nervousness specifically boosts fearful
        if nervousness > 0.4:
            boost = nervousness * self.attention_weight * 0.3
            scores["fearful"] = min(1.0, scores["fearful"] + boost)

        # Normalize scores to sum to 1
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}

        return EmotionScores.from_dict(scores)

    def _apply_engagement_adjustment(
        self,
        confidence: float,
        attention: AttentionResult,
    ) -> float:
        """Adjust confidence based on engagement level.

        Low engagement reduces confidence in the emotion reading.

        Args:
            confidence: Base confidence score.
            attention: Attention analysis result.

        Returns:
            Adjusted confidence score.
        """
        engagement = attention.metrics.engagement_score

        if engagement < self.attention_engagement_threshold:
            # Reduce confidence when user isn't engaged
            penalty = (self.attention_engagement_threshold - engagement) * self.attention_weight
            confidence = confidence * (1.0 - penalty)

        return max(0.0, min(1.0, confidence))

    def _fuse_weighted(
        self,
        facial: FacialEmotionResult,
        speech: SpeechEmotionResult,
    ) -> EmotionScores:
        """Weighted average fusion using predefined weights."""
        facial_dict = facial.emotions.to_dict()
        speech_dict = speech.emotions.to_dict()

        fused = {
            key: (
                facial_dict[key] * self.visual_weight +
                speech_dict[key] * self.audio_weight
            )
            for key in facial_dict
        }

        return EmotionScores.from_dict(fused)

    def _fuse_confidence(
        self,
        facial: FacialEmotionResult,
        speech: SpeechEmotionResult,
    ) -> EmotionScores:
        """Confidence-weighted fusion - weight by model confidence."""
        facial_dict = facial.emotions.to_dict()
        speech_dict = speech.emotions.to_dict()

        total_conf = facial.confidence + speech.confidence
        if total_conf == 0:
            # Fallback to weighted fusion if both confidences are zero
            return self._fuse_weighted(facial, speech)

        facial_weight = facial.confidence / total_conf
        speech_weight = speech.confidence / total_conf

        fused = {
            key: (
                facial_dict[key] * facial_weight +
                speech_dict[key] * speech_weight
            )
            for key in facial_dict
        }

        return EmotionScores.from_dict(fused)

    def _calculate_fusion_confidence(
        self,
        facial: FacialEmotionResult,
        speech: SpeechEmotionResult,
    ) -> float:
        """Calculate confidence score for the fused result.

        Considers:
        - Individual model confidences
        - Agreement between modalities

        Args:
            facial: Facial emotion result.
            speech: Speech emotion result.

        Returns:
            Fusion confidence score (0-1).
        """
        # Base confidence from individual models
        base_confidence = (
            facial.confidence * self.visual_weight +
            speech.confidence * self.audio_weight
        )

        # Agreement bonus/penalty
        facial_dominant = facial.emotions.dominant_emotion
        speech_dominant = speech.emotions.dominant_emotion

        if facial_dominant == speech_dominant:
            # Modalities agree - boost confidence
            agreement_factor = 1.2
        else:
            # Check if they at least have similar top emotions
            facial_dict = facial.emotions.to_dict()
            speech_dict = speech.emotions.to_dict()

            # Get top 2 emotions from each
            facial_top2 = sorted(facial_dict, key=facial_dict.get, reverse=True)[:2]  # type: ignore
            speech_top2 = sorted(speech_dict, key=speech_dict.get, reverse=True)[:2]  # type: ignore

            if set(facial_top2) & set(speech_top2):
                # Some overlap in top emotions
                agreement_factor = 1.0
            else:
                # Complete disagreement - reduce confidence
                agreement_factor = 0.8

        final_confidence = min(1.0, base_confidence * agreement_factor)
        return final_confidence

    def fuse_multiple(
        self,
        facial_results: list[FacialEmotionResult],
        speech_results: list[SpeechEmotionResult],
        timestamps: list[float] | None = None,
    ) -> list[EmotionResult]:
        """Fuse multiple pairs of emotion results.

        Handles cases where there are different numbers of facial and
        speech results by aligning based on timestamps.

        Args:
            facial_results: List of facial emotion results.
            speech_results: List of speech emotion results.
            timestamps: List of timestamps. If None, uses indices.

        Returns:
            List of fused emotion results.
        """
        if not facial_results and not speech_results:
            return []

        if timestamps is None:
            max_len = max(len(facial_results), len(speech_results))
            timestamps = [float(i) for i in range(max_len)]

        results = []

        # Simple alignment by index for now
        # A more sophisticated approach would align by timestamp
        max_len = max(len(facial_results), len(speech_results))

        for i in range(max_len):
            facial = facial_results[i] if i < len(facial_results) else None
            speech = speech_results[i] if i < len(speech_results) else None
            ts = timestamps[i] if i < len(timestamps) else float(i)

            if facial is not None or speech is not None:
                results.append(self.fuse(facial, speech, ts))

        return results

