"""Model interface and registry for embedding generation."""

from transcriptomic_fms.models.base import BaseEmbeddingModel
from transcriptomic_fms.models.registry import get_model, list_models, register_model

__all__ = ["BaseEmbeddingModel", "get_model", "list_models", "register_model"]

