"""Base interface for Vision-Language-Action models."""

from abc import abstractmethod
from typing import Any

import numpy as np

from emotion_detection_action.core.config import ModelConfig
from emotion_detection_action.core.types import ActionCommand, NeuralEmotionResult
from emotion_detection_action.models.base import BaseModel


class VLAInput:
    """Input container for VLA models."""

    def __init__(
        self,
        image: np.ndarray | None = None,
        text_prompt: str = "",
        emotion_context: NeuralEmotionResult | None = None,
        additional_context: dict[str, Any] | None = None,
    ) -> None:
        """Initialize VLA input.

        Args:
            image: Visual input (frame from video).
            text_prompt: Text instruction or context.
            emotion_context: Neural emotion result to inform the action prompt.
            additional_context: Any additional context for the model.
        """
        self.image = image
        self.text_prompt = text_prompt
        self.emotion_context = emotion_context
        self.additional_context = additional_context or {}

    def build_prompt(self) -> str:
        """Build a text prompt incorporating emotion context.

        Returns:
            Formatted prompt string.
        """
        parts = []

        if self.emotion_context:
            emotion = self.emotion_context.dominant_emotion       # str
            confidence = self.emotion_context.confidence
            parts.append(
                f"The human appears to be feeling {emotion} "
                f"(confidence: {confidence:.2f})."
            )

        if self.text_prompt:
            parts.append(self.text_prompt)
        else:
            parts.append("What action should the robot take?")

        return " ".join(parts)


class BaseVLAModel(BaseModel[VLAInput, ActionCommand]):
    """Abstract base class for Vision-Language-Action models.

    VLA models take visual input and language instructions to generate
    robot actions. This base class defines the interface for emotion-aware
    action generation.

    Subclasses should implement:
    - load(): Load model weights
    - unload(): Free model resources
    - predict(): Generate action from VLA input
    """

    def __init__(self, config: ModelConfig) -> None:
        """Initialize VLA model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)

    @abstractmethod
    def predict(self, input_data: VLAInput) -> ActionCommand:
        """Generate an action command based on visual and emotional context.

        Args:
            input_data: VLA input containing image, prompt, and emotion context.

        Returns:
            ActionCommand with robot action parameters.
        """
        pass

    def generate_emotion_response(
        self,
        emotion_result: NeuralEmotionResult,
        image: np.ndarray | None = None,
        custom_prompt: str | None = None,
    ) -> ActionCommand:
        """Generate an appropriate action response to detected emotion.

        Convenience method that wraps predict() with emotion-focused input.

        Args:
            emotion_result: NeuralEmotionResult from the detector.
            image: Optional visual context.
            custom_prompt: Optional custom instruction.

        Returns:
            ActionCommand appropriate for the detected emotion.
        """
        prompt = custom_prompt or self._get_default_emotion_prompt(emotion_result)

        vla_input = VLAInput(
            image=image,
            text_prompt=prompt,
            emotion_context=emotion_result,
        )

        return self.predict(vla_input)

    def _get_default_emotion_prompt(self, emotion_result: NeuralEmotionResult) -> str:
        """Get a default prompt based on detected emotion.

        Args:
            emotion_result: NeuralEmotionResult from the detector.

        Returns:
            Appropriate prompt for the emotion.
        """
        emotion = emotion_result.dominant_emotion  # already a str

        prompts = {
            "happy":     "The person is happy. Generate a friendly gesture or acknowledgment.",
            "sad":       "The person appears sad. Generate a comforting or supportive action.",
            "angry":     "The person seems upset. Generate a calming gesture or give space.",
            "fearful":   "The person appears scared. Generate a reassuring, non-threatening action.",
            "surprised": "The person is surprised. Wait and observe before acting.",
            "disgusted": "The person shows disgust. Step back and assess the situation.",
            "neutral":   "The person is neutral. Await further instruction or continue current task.",
            "unclear":   "No clear emotional signal. Maintain current state and continue observation.",
        }

        return prompts.get(emotion, prompts["neutral"])

