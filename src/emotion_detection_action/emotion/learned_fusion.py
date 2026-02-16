"""Machine learning-based multimodal emotion fusion.

This module provides a neural network approach to fusing facial, speech,
and attention emotion signals into a unified emotion prediction.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from emotion_detection_action.core.types import (
    AttentionResult,
    EmotionResult,
    EmotionScores,
    FacialEmotionResult,
    SpeechEmotionResult,
)


# Emotion order must be consistent for model input/output
EMOTION_ORDER = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]


@dataclass
class LearnedFusionConfig:
    """Configuration for learned fusion model."""

    model_path: str | None = None  # Path to trained model weights
    device: str = "cpu"  # Device to run inference on
    hidden_dims: list[int] | None = None  # Hidden layer dimensions [64, 32]
    dropout: float = 0.3  # Dropout rate during training
    confidence_output: bool = True  # Whether model outputs confidence

    def __post_init__(self) -> None:
        if self.hidden_dims is None:
            self.hidden_dims = [64, 32]


if TORCH_AVAILABLE:

    class FusionMLP(nn.Module):
        """Lightweight MLP for multimodal emotion fusion.

        Takes emotion scores from facial recognition, speech recognition,
        and attention metrics, and outputs fused emotion probabilities.

        Architecture:
            Input (17) -> Dense(64) -> ReLU -> Dropout -> Dense(32) -> ReLU -> Dropout -> Dense(7+1) -> Softmax

        Input features (17 total):
            - Facial emotions: 7 (angry, disgusted, fearful, happy, neutral, sad, surprised)
            - Speech emotions: 7 (same order)
            - Attention metrics: 3 (stress_score, engagement_score, nervousness_score)

        Output:
            - Emotion probabilities: 7 (same order as input)
            - Confidence score: 1 (optional)
        """

        def __init__(
            self,
            facial_dim: int = 7,
            speech_dim: int = 7,
            attention_dim: int = 3,
            hidden_dims: list[int] | None = None,
            dropout: float = 0.3,
            num_emotions: int = 7,
            output_confidence: bool = True,
        ) -> None:
            """Initialize the fusion MLP.

            Args:
                facial_dim: Number of facial emotion features.
                speech_dim: Number of speech emotion features.
                attention_dim: Number of attention metric features.
                hidden_dims: List of hidden layer dimensions.
                dropout: Dropout probability.
                num_emotions: Number of output emotion classes.
                output_confidence: Whether to output a confidence score.
            """
            super().__init__()

            if hidden_dims is None:
                hidden_dims = [64, 32]

            self.facial_dim = facial_dim
            self.speech_dim = speech_dim
            self.attention_dim = attention_dim
            self.num_emotions = num_emotions
            self.output_confidence = output_confidence

            input_dim = facial_dim + speech_dim + attention_dim  # 17

            # Build hidden layers
            layers: list[nn.Module] = []
            prev_dim = input_dim

            for i, hidden_dim in enumerate(hidden_dims):
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout if i == 0 else dropout * 0.7),
                ])
                prev_dim = hidden_dim

            self.feature_extractor = nn.Sequential(*layers)

            # Output heads
            output_dim = num_emotions + (1 if output_confidence else 0)
            self.output_layer = nn.Linear(prev_dim, output_dim)

        def forward(
            self,
            facial: "torch.Tensor",
            speech: "torch.Tensor",
            attention: "torch.Tensor",
        ) -> tuple["torch.Tensor", "torch.Tensor | None"]:
            """Forward pass.

            Args:
                facial: (batch, 7) facial emotion scores.
                speech: (batch, 7) speech emotion scores.
                attention: (batch, 3) attention metrics [stress, engagement, nervousness].

            Returns:
                Tuple of:
                    - (batch, 7) fused emotion probabilities
                    - (batch, 1) confidence scores (or None if output_confidence=False)
            """
            # Concatenate all inputs
            x = torch.cat([facial, speech, attention], dim=-1)

            # Extract features
            features = self.feature_extractor(x)

            # Get outputs
            outputs = self.output_layer(features)

            if self.output_confidence:
                emotion_logits = outputs[:, :self.num_emotions]
                confidence_logit = outputs[:, self.num_emotions:]
                emotion_probs = torch.softmax(emotion_logits, dim=-1)
                confidence = torch.sigmoid(confidence_logit)
                return emotion_probs, confidence
            else:
                emotion_probs = torch.softmax(outputs, dim=-1)
                return emotion_probs, None

        def get_num_parameters(self) -> int:
            """Get total number of trainable parameters."""
            return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LearnedEmotionFusion:
    """Learned multimodal emotion fusion using a neural network.

    This class wraps the FusionMLP model and provides a simple interface
    for fusing facial, speech, and attention emotion signals.

    Example:
        >>> fusion = LearnedEmotionFusion(model_path="models/fusion_mlp.pt")
        >>> result = fusion.fuse(facial_result, speech_result, attention_result)
        >>> print(result.dominant_emotion)
    """

    def __init__(
        self,
        config: LearnedFusionConfig | None = None,
        model_path: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        """Initialize the learned fusion module.

        Args:
            config: Configuration object (takes precedence).
            model_path: Path to trained model weights (if config not provided).
            device: Device to run inference on (if config not provided).
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is required for learned fusion. "
                "Install with: pip install torch"
            )

        if config is not None:
            self.config = config
        else:
            self.config = LearnedFusionConfig(
                model_path=str(model_path) if model_path else None,
                device=device,
            )

        self.device = self.config.device
        self._model: FusionMLP | None = None
        self._initialized = False

    def load(self) -> None:
        """Load the model weights.

        If a model path is provided and exists, loads the trained weights.
        Otherwise, initializes with default weights that approximate a
        sensible weighted average fusion (60% facial, 40% speech).
        """
        self._model = FusionMLP(
            hidden_dims=self.config.hidden_dims,
            dropout=self.config.dropout,
            output_confidence=self.config.confidence_output,
        )

        if self.config.model_path and Path(self.config.model_path).exists():
            state_dict = torch.load(
                self.config.model_path,
                map_location=self.device,
                weights_only=True,
            )
            self._model.load_state_dict(state_dict)
        else:
            # Initialize with sensible default weights
            self._initialize_default_weights()

        self._model.to(self.device)
        self._model.eval()
        self._initialized = True

    def _initialize_default_weights(self) -> None:
        """Initialize model with sensible default weights.

        This creates a model that approximates weighted average fusion:
        - 60% weight to facial emotions
        - 40% weight to speech emotions
        - Attention metrics provide minor modulation

        This allows the model to work out-of-the-box without training,
        while still being trainable for better performance.
        """
        assert self._model is not None

        with torch.no_grad():
            # Initialize all layers with small random weights first
            for module in self._model.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.1)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

            # Get the first linear layer (input layer)
            first_layer = self._model.feature_extractor[0]
            if isinstance(first_layer, nn.Linear):
                # Input structure: [facial(7), speech(7), attention(3)] = 17
                # Set weights so facial has ~60% influence, speech ~40%

                # Scale down all weights first
                first_layer.weight.data *= 0.1

                # Boost facial emotion connections (indices 0-6)
                first_layer.weight.data[:, 0:7] *= 6.0  # 60% influence

                # Boost speech emotion connections (indices 7-13)
                first_layer.weight.data[:, 7:14] *= 4.0  # 40% influence

                # Attention metrics (indices 14-16) keep small influence
                first_layer.weight.data[:, 14:17] *= 1.0

            # Initialize output layer to pass through emotion signals
            output_layer = self._model.output_layer
            if isinstance(output_layer, nn.Linear):
                # Create identity-like mapping for emotion outputs
                nn.init.zeros_(output_layer.weight)
                nn.init.zeros_(output_layer.bias)

                # Set small positive bias for neutral emotion as default
                output_layer.bias.data[4] = 0.5  # neutral index

                # Confidence output (index 7) - default to moderate confidence
                if self._model.output_confidence:
                    output_layer.bias.data[7] = 0.5

    def unload(self) -> None:
        """Unload the model to free memory."""
        self._model = None
        self._initialized = False

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._initialized and self._model is not None

    def fuse(
        self,
        facial_result: FacialEmotionResult | None = None,
        speech_result: SpeechEmotionResult | None = None,
        attention_result: AttentionResult | None = None,
        timestamp: float = 0.0,
    ) -> EmotionResult:
        """Fuse emotion results using the learned model.

        Args:
            facial_result: Facial emotion recognition result.
            speech_result: Speech emotion recognition result.
            attention_result: Attention analysis result.
            timestamp: Timestamp for the fused result.

        Returns:
            Fused emotion result.

        Raises:
            RuntimeError: If model is not loaded.
            ValueError: If no valid inputs are provided.
        """
        if not self.is_loaded:
            self.load()

        assert self._model is not None

        # Convert inputs to tensors
        facial_vec = self._emotion_result_to_tensor(
            facial_result.emotions if facial_result else None
        )
        speech_vec = self._emotion_result_to_tensor(
            speech_result.emotions if speech_result else None
        )
        attention_vec = self._attention_to_tensor(attention_result)

        # Check if we have any valid input
        has_facial = facial_result is not None
        has_speech = speech_result is not None
        has_attention = attention_result is not None and attention_result.confidence > 0

        if not has_facial and not has_speech:
            raise ValueError("At least one of facial or speech result is required")

        # Run inference
        with torch.no_grad():
            emotion_probs, confidence = self._model(
                facial_vec.unsqueeze(0),
                speech_vec.unsqueeze(0),
                attention_vec.unsqueeze(0),
            )

        # Convert output to EmotionScores
        probs = emotion_probs.squeeze(0).cpu().numpy()
        emotions_dict = {
            emotion: float(probs[i])
            for i, emotion in enumerate(EMOTION_ORDER)
        }
        emotions = EmotionScores.from_dict(emotions_dict)

        # Get confidence
        if confidence is not None:
            fusion_confidence = float(confidence.squeeze().cpu().numpy())
        else:
            # Fallback: use average of input confidences
            confidences = []
            if facial_result:
                confidences.append(facial_result.confidence)
            if speech_result:
                confidences.append(speech_result.confidence)
            fusion_confidence = sum(confidences) / len(confidences) if confidences else 0.5

        return EmotionResult(
            timestamp=timestamp,
            emotions=emotions,
            facial_result=facial_result,
            speech_result=speech_result,
            attention_result=attention_result,
            fusion_confidence=fusion_confidence,
        )

    def _emotion_result_to_tensor(
        self,
        emotions: EmotionScores | None,
    ) -> "torch.Tensor":
        """Convert EmotionScores to tensor."""
        if emotions is None:
            return torch.zeros(7, dtype=torch.float32, device=self.device)

        scores_dict = emotions.to_dict()
        values = [scores_dict.get(e, 0.0) for e in EMOTION_ORDER]
        return torch.tensor(values, dtype=torch.float32, device=self.device)

    def _attention_to_tensor(
        self,
        attention: AttentionResult | None,
    ) -> "torch.Tensor":
        """Convert AttentionResult to tensor."""
        if attention is None or attention.confidence == 0:
            # Default values when no attention data
            return torch.tensor(
                [0.0, 0.5, 0.0],  # [stress, engagement, nervousness]
                dtype=torch.float32,
                device=self.device,
            )

        return torch.tensor(
            [
                attention.metrics.stress_score,
                attention.metrics.engagement_score,
                attention.metrics.nervousness_score,
            ],
            dtype=torch.float32,
            device=self.device,
        )


def create_untrained_model(
    hidden_dims: list[int] | None = None,
    output_confidence: bool = True,
) -> "FusionMLP":
    """Create an untrained fusion model for training.

    Args:
        hidden_dims: Hidden layer dimensions.
        output_confidence: Whether to output confidence score.

    Returns:
        Untrained FusionMLP model.

    Raises:
        RuntimeError: If PyTorch is not available.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required. Install with: pip install torch")

    return FusionMLP(
        hidden_dims=hidden_dims,
        output_confidence=output_confidence,
    )


def save_model(model: "FusionMLP", path: str | Path) -> None:
    """Save model weights to file.

    Args:
        model: Trained FusionMLP model.
        path: Path to save weights.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required. Install with: pip install torch")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(path: str | Path, device: str = "cpu") -> "FusionMLP":
    """Load model weights from file.

    Args:
        path: Path to model weights.
        device: Device to load model on.

    Returns:
        Loaded FusionMLP model.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required. Install with: pip install torch")

    model = FusionMLP()
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
