"""Base interface for embedding models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np
import scanpy as sc


class BaseEmbeddingModel(ABC):
    """
    Abstract base class for embedding models.
    
    Each model implementation should inherit from this class and implement
    the required methods. Models can be run locally or in containers.
    """

    def __init__(
        self,
        model_name: str,
        container_image: Optional[Path] = None,
        requires_gpu: bool = False,
        **kwargs: Any,
    ):
        """
        Initialize the model.
        
        Args:
            model_name: Unique name identifier for this model
            container_image: Path to container image (if model runs in container)
            requires_gpu: Whether this model requires GPU access
            **kwargs: Model-specific configuration parameters
        """
        self.model_name = model_name
        self.container_image = container_image
        self.requires_gpu = requires_gpu
        self.config = kwargs

    @abstractmethod
    def preprocess(
        self, adata: sc.AnnData, output_path: Optional[Path] = None
    ) -> sc.AnnData:
        """
        Preprocess AnnData for this model.
        
        This method should handle any model-specific preprocessing steps
        such as normalization, filtering, gene selection, etc.
        
        Args:
            adata: Input AnnData object
            output_path: Optional path to save preprocessed data
            
        Returns:
            Preprocessed AnnData object
        """
        pass

    @abstractmethod
    def embed(
        self,
        adata: sc.AnnData,
        output_path: Path,
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Generate embeddings from preprocessed AnnData.
        
        This is the main method that should be implemented by each model.
        It takes preprocessed AnnData and returns embeddings as a numpy array.
        
        Args:
            adata: Preprocessed AnnData object
            output_path: Path where embeddings will be saved (.npy file)
            batch_size: Optional batch size for processing
            **kwargs: Model-specific parameters
            
        Returns:
            Embeddings array of shape (n_cells, n_dimensions)
        """
        pass

    def decode(
        self, embeddings: np.ndarray, output_path: Optional[Path] = None
    ) -> sc.AnnData:
        """
        Decode embeddings back to gene expression space (optional).
        
        Not all models support decoding. Default implementation raises NotImplementedError.
        
        Args:
            embeddings: Embeddings array
            output_path: Optional path to save decoded expression
            
        Returns:
            AnnData object with decoded expression in .X
        """
        raise NotImplementedError(
            f"Model {self.model_name} does not support decoding"
        )

    def validate_embeddings(
        self, embeddings: np.ndarray, n_cells: int
    ) -> None:
        """
        Validate that embeddings have correct shape and properties.
        
        Args:
            embeddings: Embeddings array to validate
            n_cells: Expected number of cells
            
        Raises:
            ValueError: If embeddings are invalid
        """
        if not isinstance(embeddings, np.ndarray):
            raise ValueError("Embeddings must be a numpy array")
        if embeddings.shape[0] != n_cells:
            raise ValueError(
                f"Embeddings shape mismatch: expected {n_cells} cells, "
                f"got {embeddings.shape[0]}"
            )
        if embeddings.ndim != 2:
            raise ValueError(
                f"Embeddings must be 2D array, got shape {embeddings.shape}"
            )
        if np.any(np.isnan(embeddings)):
            raise ValueError("Embeddings contain NaN values")
        if np.any(np.isinf(embeddings)):
            raise ValueError("Embeddings contain Inf values")

    def get_container_command(
        self,
        adata_path: Path,
        output_path: Path,
        **kwargs: Any,
    ) -> list[str]:
        """
        Get command to run this model in a container.
        
        This method should return the command that will be executed inside
        the container to generate embeddings. The command should be a list
        of strings suitable for subprocess execution.
        
        Args:
            adata_path: Path to input AnnData file (inside container)
            output_path: Path to output embeddings file (inside container)
            **kwargs: Additional model-specific parameters
            
        Returns:
            List of command strings
        """
        # Default implementation: use Python module interface
        return [
            "python",
            "-m",
            "transcriptomic_fms.cli",
            "embed",
            "--model",
            self.model_name,
            "--input",
            str(adata_path),
            "--output",
            str(output_path),
        ] + [f"--{k}={v}" for k, v in kwargs.items()]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name={self.model_name!r})"

