"""Base model interface for all models in the SDK."""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from emotion_detection_action.core.config import ModelConfig

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class BaseModel(ABC, Generic[InputT, OutputT]):
    """Abstract base class for all models."""

    def __init__(self, config: ModelConfig) -> None:
        """Initialize the model with configuration.

        Args:
            config: Model configuration including model ID, device, etc.
        """
        self.config = config
        self._model: Any = None
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._is_loaded

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory.

        Should be called before inference. Implementations should set
        self._is_loaded = True after successful loading.
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload the model from memory.

        Frees resources. Implementations should set self._is_loaded = False.
        """
        pass

    @abstractmethod
    def predict(self, input_data: InputT) -> OutputT:
        """Run inference on input data.

        Args:
            input_data: Input data for the model.

        Returns:
            Model output.

        Raises:
            RuntimeError: If model is not loaded.
        """
        pass

    def ensure_loaded(self) -> None:
        """Ensure the model is loaded, loading it if necessary."""
        if not self._is_loaded:
            self.load()

    def __enter__(self) -> "BaseModel[InputT, OutputT]":
        """Context manager entry - load the model."""
        self.load()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - unload the model."""
        self.unload()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_id={self.config.model_id!r}, loaded={self._is_loaded})"

