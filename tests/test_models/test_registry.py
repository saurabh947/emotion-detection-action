"""Tests for model registry."""

import pytest

from emotion_detection_action.core.config import ModelConfig
from emotion_detection_action.models.base import BaseModel
from emotion_detection_action.models.registry import ModelRegistry, get_registry


class DummyModel(BaseModel[str, str]):
    """Dummy model for testing."""

    def load(self) -> None:
        self._is_loaded = True

    def unload(self) -> None:
        self._is_loaded = False

    def predict(self, input_data: str) -> str:
        if not self._is_loaded:
            raise RuntimeError("Model not loaded")
        return f"processed: {input_data}"


class TestModelRegistry:
    """Tests for ModelRegistry class."""

    def test_register_and_get(self):
        """Test registering and retrieving a model."""
        registry = ModelRegistry()
        registry.register("vla", "test_model", DummyModel)

        model_class = registry.get("vla", "test_model")
        assert model_class == DummyModel

    def test_get_nonexistent(self):
        """Test getting a non-existent model returns None."""
        registry = ModelRegistry()
        model_class = registry.get("vla", "nonexistent")
        assert model_class is None

    def test_register_invalid_category(self):
        """Registering to an unknown category raises ValueError."""
        registry = ModelRegistry()
        with pytest.raises(ValueError, match="Unknown category"):
            registry.register("face_detection", "model", DummyModel)

    def test_create_model(self):
        """Test creating a model instance."""
        registry = ModelRegistry()
        registry.register("vla", "test_model", DummyModel)

        config = ModelConfig(model_id="test_model")
        model = registry.create("vla", "test_model", config)

        assert isinstance(model, DummyModel)
        assert model.config == config

    def test_create_nonexistent_raises(self):
        """Test creating non-existent model raises error."""
        registry = ModelRegistry()
        config = ModelConfig(model_id="nonexistent")

        with pytest.raises(ValueError, match="Model 'nonexistent' not found"):
            registry.create("vla", "nonexistent", config)

    def test_list_models(self):
        """Test listing registered models."""
        registry = ModelRegistry()
        registry.register("vla", "model_a", DummyModel)
        registry.register("vla", "model_b", DummyModel)

        # List all
        all_models = registry.list_models()
        assert "model_a" in all_models["vla"]
        assert "model_b" in all_models["vla"]

        # List specific category
        vla_models = registry.list_models("vla")
        assert "model_a" in vla_models["vla"]

    def test_register_decorator(self):
        """Test using register as decorator."""
        registry = ModelRegistry()

        @registry.register_decorator("vla", "decorated_model")
        class DecoratedModel(BaseModel[str, str]):
            def load(self) -> None:
                pass

            def unload(self) -> None:
                pass

            def predict(self, input_data: str) -> str:
                return input_data

        model_class = registry.get("vla", "decorated_model")
        assert model_class == DecoratedModel

    def test_singleton_instance(self):
        """Test that get_instance returns singleton."""
        registry1 = ModelRegistry.get_instance()
        registry2 = ModelRegistry.get_instance()
        assert registry1 is registry2

    def test_global_registry(self):
        """Test global registry helper function."""
        registry = get_registry()
        assert isinstance(registry, ModelRegistry)

        # Should have default models registered
        models = registry.list_models("vla")
        assert len(models["vla"]) > 0


class TestBaseModel:
    """Tests for BaseModel interface."""

    def test_context_manager(self):
        """Test using model as context manager."""
        config = ModelConfig(model_id="test")
        model = DummyModel(config)

        assert not model.is_loaded

        with model as m:
            assert m.is_loaded
            result = m.predict("hello")
            assert result == "processed: hello"

        assert not model.is_loaded

    def test_ensure_loaded(self):
        """Test ensure_loaded auto-loads model."""
        config = ModelConfig(model_id="test")
        model = DummyModel(config)

        assert not model.is_loaded
        model.ensure_loaded()
        assert model.is_loaded

    def test_predict_without_load_raises(self):
        """Test predicting without loading raises error."""
        config = ModelConfig(model_id="test")
        model = DummyModel(config)

        with pytest.raises(RuntimeError, match="Model not loaded"):
            model.predict("hello")

    def test_repr(self):
        """Test string representation."""
        config = ModelConfig(model_id="test/model")
        model = DummyModel(config)

        repr_str = repr(model)
        assert "DummyModel" in repr_str
        assert "test/model" in repr_str
        assert "loaded=False" in repr_str

