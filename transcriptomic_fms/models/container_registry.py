"""Registry for model-specific container definitions."""

from pathlib import Path
from typing import Optional

_MODEL_CONTAINERS: dict[str, Path] = {}


def register_model_container(model_name: str, container_def_path: Path) -> None:
    """
    Register a container definition path for a model.

    Args:
        model_name: Name of the model
        container_def_path: Path to Singularity.def file for this model
    """
    _MODEL_CONTAINERS[model_name] = container_def_path


def get_model_container(model_name: str) -> Optional[Path]:
    """
    Get container definition path for a model.

    Args:
        model_name: Name of the model

    Returns:
        Path to Singularity.def file, or None if not found
    """
    return _MODEL_CONTAINERS.get(model_name)


def list_model_containers() -> dict[str, Path]:
    """List all registered model containers."""
    return _MODEL_CONTAINERS.copy()
