"""Multimodal emotion fusion module using machine learning."""

from __future__ import annotations

from pathlib import Path

from emotion_detection_action.core.types import (
    AttentionResult,
    EmotionResult,
    FacialEmotionResult,
    SpeechEmotionResult,
)
from emotion_detection_action.emotion.learned_fusion import (
    LearnedEmotionFusion,
    LearnedFusionConfig,
)


class EmotionFusion:
    """Fuses facial, speech, and attention analysis results using machine learning.

    Uses a neural network to combine emotion predictions from multiple modalities
    (visual, audio, and attention) into a unified emotion result. This is the
    default fusion method that runs automatically after the AI models process
    facial, speech, and attention data.

    By default (without a trained model), the network uses sensible weights that
    approximate weighted average fusion (60% facial, 40% speech). For better
    accuracy, you can train the model on your own labeled data.

    Example:
        >>> # Works out-of-the-box with default weights
        >>> fusion = EmotionFusion()
        >>> result = fusion.fuse(facial_result, speech_result, attention_result)
        >>> print(result.emotions.dominant_emotion)

        # Using a custom trained model for better accuracy
        >>> fusion = EmotionFusion(model_path="models/custom_fusion.pt")
        >>> result = fusion.fuse(facial_result, speech_result, attention_result)
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        """Initialize the fusion module.

        Args:
            model_path: Path to trained fusion model weights. If None, uses
                default weights that approximate weighted average fusion
                (60% facial, 40% speech). Works out-of-the-box.
            device: Device to run the model on ("cpu", "cuda", "mps").
        """
        self.model_path = model_path
        self.device = device

        config = LearnedFusionConfig(
            model_path=str(model_path) if model_path else None,
            device=device,
        )
        self._fusion = LearnedEmotionFusion(config=config)
        self._fusion.load()

    def fuse(
        self,
        facial_result: FacialEmotionResult | None = None,
        speech_result: SpeechEmotionResult | None = None,
        attention_result: AttentionResult | None = None,
        timestamp: float = 0.0,
    ) -> EmotionResult:
        """Fuse emotion results from multiple modalities using ML.

        The neural network takes emotion scores from facial recognition,
        speech recognition, and attention metrics, and outputs fused
        emotion probabilities along with a confidence score.

        Args:
            facial_result: Facial emotion recognition result.
            speech_result: Speech emotion recognition result.
            attention_result: Attention analysis result (optional).
            timestamp: Timestamp for the fused result.

        Returns:
            Fused emotion result.

        Raises:
            ValueError: If neither facial nor speech result is provided.
        """
        return self._fusion.fuse(
            facial_result=facial_result,
            speech_result=speech_result,
            attention_result=attention_result,
            timestamp=timestamp,
        )

    @property
    def is_loaded(self) -> bool:
        """Check if the fusion model is loaded."""
        return self._fusion.is_loaded

    def reload(self, model_path: str | Path | None = None) -> None:
        """Reload the model, optionally with a new path.

        Args:
            model_path: New model path. If None, reloads current model.
        """
        if model_path is not None:
            self.model_path = model_path
            self._fusion.config.model_path = str(model_path)
        self._fusion.unload()
        self._fusion.load()
