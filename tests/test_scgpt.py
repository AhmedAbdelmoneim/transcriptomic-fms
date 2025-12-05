"""Tests for scGPT model."""

from pathlib import Path

import pytest
import scanpy as sc

from transcriptomic_fms.models.registry import get_model
from tests.base_test import BaseModelTest
from tests.utils import validate_embeddings


class TestSCGPTModel(BaseModelTest):
    """Tests for scGPT embedding model."""

    model_name = "scgpt"

    @pytest.mark.skipif(
        not (hasattr(__import__("torch"), "cuda") and __import__("torch").cuda.is_available()),
        reason="GPU not available",
    )
    def test_gpu_fallback(
        self,
        small_adata: sc.AnnData,
        tmp_dir: Path,
    ):
        """Test scGPT can fallback to CPU if needed."""
        # This test verifies scGPT can run on CPU if GPU unavailable
        # (though it's slower, it should still work)
        model = get_model("scgpt", n_hvg=100, batch_size=32, device="cpu")
        preprocessed = model.preprocess(small_adata[:20].copy())  # Small subset
        output_path = tmp_dir / "scgpt_cpu_test.npy"
        embeddings = model.embed(preprocessed, output_path)

        validate_embeddings(embeddings, 20)

    def test_with_custom_hvg(
        self,
        small_adata: sc.AnnData,
        tmp_dir: Path,
    ):
        """Test scGPT with custom HVG count."""
        model = get_model("scgpt", n_hvg=200, batch_size=32)
        preprocessed = model.preprocess(small_adata[:20].copy())  # Small subset for speed
        output_path = tmp_dir / "scgpt_custom_hvg.npy"

        # Skip if GPU not available (scGPT is slow on CPU)
        try:
            import torch

            if not torch.cuda.is_available():
                pytest.skip("scGPT requires GPU for reasonable performance")
        except ImportError:
            pytest.skip("PyTorch not available")

        embeddings = model.embed(preprocessed, output_path)
        validate_embeddings(embeddings, 20)

