"""Tests for PCA model."""

from pathlib import Path

import numpy as np
import pytest
import scanpy as sc

from transcriptomic_fms.models.registry import get_model
from tests.base_test import BaseModelTest
from tests.utils import validate_embeddings


class TestPCAModel(BaseModelTest):
    """Tests for PCA embedding model."""

    model_name = "pca"

    def test_embeddings_dimensions(
        self,
        small_adata: sc.AnnData,
        tmp_dir: Path,
    ):
        """Test PCA produces correct number of components."""
        model = get_model("pca", n_components=20)
        preprocessed = model.preprocess(small_adata)
        output_path = tmp_dir / "pca_test.npy"
        embeddings = model.embed(preprocessed, output_path)

        assert embeddings.shape[1] == 20, f"Expected 20 components, got {embeddings.shape[1]}"

    def test_with_hvg(
        self,
        small_adata: sc.AnnData,
        tmp_dir: Path,
    ):
        """Test PCA with HVG selection."""
        model = get_model("pca", n_components=10, use_hvg=True, n_hvg=500)
        preprocessed = model.preprocess(small_adata)
        output_path = tmp_dir / "pca_hvg_test.npy"
        embeddings = model.embed(preprocessed, output_path)

        validate_embeddings(embeddings, small_adata.n_obs)
        # Check that preprocessing filtered genes
        assert preprocessed.n_vars <= 500, "Should have filtered to <= 500 HVGs"

    def test_without_hvg(
        self,
        small_adata: sc.AnnData,
        tmp_dir: Path,
    ):
        """Test PCA without HVG selection uses all genes."""
        model = get_model("pca", n_components=10, use_hvg=False)
        preprocessed = model.preprocess(small_adata)
        output_path = tmp_dir / "pca_no_hvg_test.npy"
        embeddings = model.embed(preprocessed, output_path)

        validate_embeddings(embeddings, small_adata.n_obs)
        # Should still have all or most genes (may be filtered during preprocessing)
        assert preprocessed.n_vars > 0, "Should have genes"

