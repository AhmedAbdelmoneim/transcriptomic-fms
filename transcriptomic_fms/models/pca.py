"""PCA baseline embedding model."""

from pathlib import Path
from typing import Any, Optional

import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA

from transcriptomic_fms.models.base import BaseEmbeddingModel
from transcriptomic_fms.models.registry import register_model


class PCAModel(BaseEmbeddingModel):
    """PCA embedding model with optional HVG selection."""

    def __init__(
        self,
        model_name: str,
        n_components: int = 50,
        use_hvg: bool = False,
        n_hvg: int = 2000,
        **kwargs: Any,
    ):
        """
        Initialize PCA model.
        
        Args:
            model_name: Model identifier
            n_components: Number of PCA components
            use_hvg: Whether to use highly variable genes
            n_hvg: Number of HVGs to select (if use_hvg=True)
            **kwargs: Additional arguments
        """
        super().__init__(model_name=model_name, **kwargs)
        self.n_components = n_components
        self.use_hvg = use_hvg
        self.n_hvg = n_hvg
        self.pca_model: Optional[PCA] = None
        self.hvg_genes: Optional[list[str]] = None

    def preprocess(
        self, adata: sc.AnnData, output_path: Optional[Path] = None
    ) -> sc.AnnData:
        """Preprocess data: normalize, log transform, optionally select HVGs."""
        adata = adata.copy()

        # Normalize and log transform
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        # Select HVGs if requested
        if self.use_hvg:
            sc.pp.highly_variable_genes(adata, n_top_genes=self.n_hvg)
            self.hvg_genes = adata.var_names[
                adata.var["highly_variable"]
            ].tolist()
            adata = adata[:, self.hvg_genes].copy()

        # Scale
        sc.pp.scale(adata, max_value=10)

        if output_path:
            adata.write(output_path)

        return adata

    def embed(
        self,
        adata: sc.AnnData,
        output_path: Path,
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Generate PCA embeddings."""
        # Get data as dense array
        X = adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()

        # Fit PCA if not already fitted
        if self.pca_model is None:
            self.pca_model = PCA(
                n_components=self.n_components,
                svd_solver="arpack",
                random_state=42,
            )
            self.pca_model.fit(X)

        # Transform
        embeddings = self.pca_model.transform(X)

        return embeddings


# Register the model class
# Note: When instantiating, set use_hvg=True for HVG version
register_model("pca")(PCAModel)
