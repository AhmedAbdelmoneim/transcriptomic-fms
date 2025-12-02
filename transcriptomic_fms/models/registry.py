"""Model registry for discovering and loading embedding models."""

from typing import Any, Optional

from transcriptomic_fms.models.base import BaseEmbeddingModel

# Global registry of available models
_MODEL_REGISTRY: dict[str, type[BaseEmbeddingModel]] = {}


def register_model(
    model_class: Optional[type[BaseEmbeddingModel]] = None,
    model_name: Optional[str] = None,
) -> type[BaseEmbeddingModel] | Any:
    """
    Register a model class in the global registry.
    
    Models should be registered by decorating the class:
    
    @register_model
    class MyModel(BaseEmbeddingModel):
        ...
    
    Or with explicit name:
    
    @register_model("custom_name")
    class MyModel(BaseEmbeddingModel):
        ...
    
    Args:
        model_class: The model class to register (when used as decorator)
        model_name: Optional explicit model name (if not provided, uses class name)
        
    Returns:
        The model class (for use as decorator)
    """
    # Handle both @register_model and @register_model("name") usage
    if model_class is None:
        # Called with name: @register_model("name")
        def decorator(cls: type[BaseEmbeddingModel]) -> type[BaseEmbeddingModel]:
            name = model_name or cls.__name__.lower().replace("model", "").replace(
                "embedding", ""
            ).strip("_")
            _MODEL_REGISTRY[name] = cls
            return cls
        return decorator
    else:
        # Called directly: @register_model
        name = model_name or model_class.__name__.lower().replace("model", "").replace(
            "embedding", ""
        ).strip("_")
        _MODEL_REGISTRY[name] = model_class
        return model_class




def _auto_register_models() -> None:
    """Auto-register models by importing model modules."""
    # Import model modules to trigger registration
    try:
        from transcriptomic_fms.models import pca  # noqa: F401
    except ImportError:
        pass

    # Add more model imports here as they are created
    # try:
    #     from transcriptomic_fms.models import scgpt  # noqa: F401
    # except ImportError:
    #     pass


# Auto-register on import (lazy - only when needed)
def _ensure_models_loaded() -> None:
    """Ensure models are loaded into registry."""
    if not _MODEL_REGISTRY:
        _auto_register_models()


# Override get_model and list_models to auto-load
def get_model(model_name: str, **kwargs: Any) -> BaseEmbeddingModel:
    """
    Get an instance of a registered model.
    
    Args:
        model_name: Name of the model to load
        **kwargs: Arguments to pass to model constructor
        
    Returns:
        Instance of the requested model
        
    Raises:
        ValueError: If model is not registered
    """
    _ensure_models_loaded()
    if model_name not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY.keys()))
        raise ValueError(
            f"Model '{model_name}' not found. Available models: {available}"
        )

    model_class = _MODEL_REGISTRY[model_name]
    return model_class(model_name=model_name, **kwargs)


def list_models() -> list[str]:
    """
    List all registered model names.
    
    Returns:
        List of registered model names
    """
    _ensure_models_loaded()
    return sorted(_MODEL_REGISTRY.keys())

