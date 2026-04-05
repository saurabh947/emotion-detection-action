"""OpenVLA model implementation."""

from typing import Any

import numpy as np

from emotion_detection_action.core.config import ModelConfig
from emotion_detection_action.core.types import ActionCommand
from emotion_detection_action.models.vla.base import BaseVLAModel, VLAInput

# Try to import transformers for OpenVLA
try:
    from transformers import AutoModelForVision2Seq, AutoProcessor
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class OpenVLAModel(BaseVLAModel):
    """OpenVLA 7B model implementation.

    OpenVLA is a Vision-Language-Action model that generates robot actions
    from visual observations and language instructions. This implementation
    wraps the HuggingFace model and adapts it for emotion-aware action generation.

    Note: OpenVLA is a large model (7B parameters) and requires significant
    GPU memory. Consider using quantization (8-bit or 4-bit) for reduced
    memory usage.

    Example:
        >>> config = ModelConfig(
        ...     model_id="openvla/openvla-7b",
        ...     device="cuda",
        ...     load_in_8bit=True
        ... )
        >>> model = OpenVLAModel(config)
        >>> model.load()
        >>> action = model.predict(vla_input)
    """

    # Default model ID
    DEFAULT_MODEL_ID = "openvla/openvla-7b"

    def __init__(self, config: ModelConfig) -> None:
        """Initialize OpenVLA model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)

        # Override model_id if not specified or using alias
        if not config.model_id or config.model_id == "openvla":
            self.config.model_id = self.DEFAULT_MODEL_ID

        self._processor: Any = None
        self._model: Any = None
        self._device: Any = None

    def load(self) -> None:
        """Load the OpenVLA model.

        Loads the model with optional quantization based on config.

        Raises:
            RuntimeError: If transformers is not available.
        """
        if self._is_loaded:
            return

        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "transformers not available. Install with: "
                "pip install transformers torch accelerate"
            )

        model_id = self.config.model_id

        # Load processor
        self._processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            cache_dir=self.config.cache_dir,
        )

        # Prepare loading kwargs
        load_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "cache_dir": self.config.cache_dir,
        }

        # Handle quantization
        if self.config.load_in_8bit:
            try:
                import bitsandbytes  # noqa: F401
                load_kwargs["load_in_8bit"] = True
                load_kwargs["device_map"] = "auto"
            except ImportError:
                print(
                    "Warning: bitsandbytes not available, loading without 8-bit quantization"
                )
        elif self.config.load_in_4bit:
            try:
                import bitsandbytes  # noqa: F401
                load_kwargs["load_in_4bit"] = True
                load_kwargs["device_map"] = "auto"
            except ImportError:
                print(
                    "Warning: bitsandbytes not available, loading without 4-bit quantization"
                )

        # Set dtype
        if self.config.dtype == "float16":
            load_kwargs["torch_dtype"] = torch.float16
        elif self.config.dtype == "bfloat16":
            load_kwargs["torch_dtype"] = torch.bfloat16

        # Load model
        self._model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            **load_kwargs,
        )

        # Set device if not using device_map
        if "device_map" not in load_kwargs:
            self._device = torch.device(self.config.device)
            self._model = self._model.to(self._device)

        self._model.eval()
        self._is_loaded = True

    def unload(self) -> None:
        """Unload the model and free memory."""
        if self._model is not None:
            del self._model
            self._model = None

        if self._processor is not None:
            del self._processor
            self._processor = None

        self._device = None
        self._is_loaded = False

        # Try to free GPU memory
        if TRANSFORMERS_AVAILABLE:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def predict(self, input_data: VLAInput) -> ActionCommand:
        """Generate an action from VLA input.

        Args:
            input_data: VLA input with image, prompt, and context.

        Returns:
            ActionCommand with generated action.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Build the prompt
        prompt = input_data.build_prompt()

        # Prepare image
        image = input_data.image
        if image is None:
            # Create a dummy image if none provided
            image = np.zeros((224, 224, 3), dtype=np.uint8)

        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            import cv2
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process inputs
        from PIL import Image as PILImage
        pil_image = PILImage.fromarray(image)

        inputs = self._processor(
            prompt,
            pil_image,
            return_tensors="pt",
        )

        # Move to device
        if self._device is not None:
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
        else:
            # When using device_map, move to first device
            inputs = {
                k: v.to(self._model.device) if hasattr(v, "to") else v
                for k, v in inputs.items()
            }

        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )

        # Decode output
        generated = self._processor.batch_decode(
            outputs,
            skip_special_tokens=True,
        )[0]

        # Parse the action from model output
        action = self._parse_action_output(generated, input_data)

        return action

    def _parse_action_output(
        self,
        model_output: str,
        input_data: VLAInput,
    ) -> ActionCommand:
        """Parse model output into an ActionCommand.

        OpenVLA typically outputs robot actions in a specific format.
        This method parses that output.

        Args:
            model_output: Raw model output string.
            input_data: Original VLA input for context.

        Returns:
            Parsed ActionCommand.
        """
        # OpenVLA outputs can vary based on the task
        # For emotion-based responses, we'll interpret the output
        # as high-level action descriptions

        action_type = "generated"
        parameters: dict[str, Any] = {
            "raw_response": model_output,
        }

        # Try to extract action type from output
        output_lower = model_output.lower()

        if any(word in output_lower for word in ["wave", "greet", "hello"]):
            action_type = "greeting"
        elif any(word in output_lower for word in ["approach", "move", "forward"]):
            action_type = "approach"
        elif any(word in output_lower for word in ["back", "retreat", "away"]):
            action_type = "retreat"
        elif any(word in output_lower for word in ["wait", "pause", "stop"]):
            action_type = "wait"
        elif any(word in output_lower for word in ["nod", "acknowledge"]):
            action_type = "acknowledge"

        # Add emotion context to parameters
        if input_data.emotion_context:
            parameters["emotion_context"] = {
                "emotion": input_data.emotion_context.dominant_emotion,
                "confidence": input_data.emotion_context.confidence,
            }

        return ActionCommand(
            action_type=action_type,
            parameters=parameters,
            confidence=0.8,  # Model doesn't provide confidence, use default
            raw_output=model_output,
        )

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model.

        Returns:
            Dictionary with model information.
        """
        info = {
            "model_id": self.config.model_id,
            "loaded": self._is_loaded,
            "device": str(self._device) if self._device else "auto",
            "quantization": None,
        }

        if self.config.load_in_8bit:
            info["quantization"] = "8-bit"
        elif self.config.load_in_4bit:
            info["quantization"] = "4-bit"

        return info

