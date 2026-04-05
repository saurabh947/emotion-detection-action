"""Model registry for managing and instantiating models."""

from typing import Any, Callable, TypeVar

from emotion_detection_action.core.config import ModelConfig
from emotion_detection_action.models.base import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel[Any, Any])


class ModelRegistry:
    """Registry for model classes enabling runtime model selection.

    The registry allows registering model implementations by name and
    instantiating them based on configuration. This enables swapping
    models without code changes.

    Example:
        >>> registry = ModelRegistry()
        >>> registry.register("openvla", OpenVLAModel)
        >>> registry.register("custom_vla", MyCustomVLAModel)
        >>>
        >>> # Later, instantiate based on config
        >>> model = registry.create("openvla", config)
    """

    _instance: "ModelRegistry | None" = None

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._registry: dict[str, dict[str, type[BaseModel[Any, Any]]]] = {
            "vla": {},
        }

    @classmethod
    def get_instance(cls) -> "ModelRegistry":
        """Get the singleton instance of the registry."""
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._register_defaults()
        return cls._instance

    def _register_defaults(self) -> None:
        """Register default model implementations."""
        # Import here to avoid circular imports
        from emotion_detection_action.models.vla.openvla import OpenVLAModel

        self.register("vla", "openvla/openvla-7b", OpenVLAModel)
        self.register("vla", "openvla", OpenVLAModel)  # Alias

    def register(
        self,
        category: str,
        name: str,
        model_class: type[BaseModel[Any, Any]],
    ) -> None:
        """Register a model class.

        Args:
            category: Model category (e.g., "vla").
            name: Model name/identifier.
            model_class: The model class to register.

        Raises:
            ValueError: If category is unknown.
        """
        if category not in self._registry:
            raise ValueError(
                f"Unknown category: {category}. "
                f"Valid categories: {list(self._registry.keys())}"
            )
        self._registry[category][name] = model_class

    def get(
        self,
        category: str,
        name: str,
    ) -> type[BaseModel[Any, Any]] | None:
        """Get a registered model class.

        Args:
            category: Model category.
            name: Model name/identifier.

        Returns:
            The registered model class, or None if not found.
        """
        return self._registry.get(category, {}).get(name)

    def create(
        self,
        category: str,
        name: str,
        config: ModelConfig,
    ) -> BaseModel[Any, Any]:
        """Create a model instance.

        Args:
            category: Model category.
            name: Model name/identifier.
            config: Model configuration.

        Returns:
            Instantiated model.

        Raises:
            ValueError: If model is not registered.
        """
        model_class = self.get(category, name)
        if model_class is None:
            available = list(self._registry.get(category, {}).keys())
            raise ValueError(
                f"Model '{name}' not found in category '{category}'. "
                f"Available: {available}"
            )
        return model_class(config)

    def list_models(self, category: str | None = None) -> dict[str, list[str]]:
        """List registered models.

        Args:
            category: Optional category to filter by.

        Returns:
            Dictionary mapping categories to lists of model names.
        """
        if category:
            return {category: list(self._registry.get(category, {}).keys())}
        return {cat: list(models.keys()) for cat, models in self._registry.items()}

    def register_decorator(
        self,
        category: str,
        name: str,
    ) -> Callable[[type[ModelT]], type[ModelT]]:
        """Decorator for registering a model class.

        Args:
            category: Model category.
            name: Model name/identifier.

        Returns:
            Decorator function.

        Example:
            >>> @registry.register_decorator("vla", "my_model")
            ... class MyVLAModel(BaseVLAModel):
            ...     pass
        """

        def decorator(cls: type[ModelT]) -> type[ModelT]:
            self.register(category, name, cls)
            return cls

        return decorator


# Global registry instance
def get_registry() -> ModelRegistry:
    """Get the global model registry instance."""
    return ModelRegistry.get_instance()

