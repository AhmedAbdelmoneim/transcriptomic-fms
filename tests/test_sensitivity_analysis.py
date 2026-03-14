"""Tests for sensitivity-analysis CLI and model contract."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import scanpy as sc


class TestSensitivityAnalysisContract(unittest.TestCase):
    """Test that models respect the sensitivity-analysis contract."""

    def test_base_compute_sensitivity_raises(self) -> None:
        """BaseEmbeddingModel.compute_sensitivity raises NotImplementedError by default."""
        from transcriptomic_fms.models.base import BaseEmbeddingModel

        class StubModel(BaseEmbeddingModel):
            def preprocess(self, adata, output_path=None):
                return adata

            def embed(self, adata, output_path, batch_size=None, **kwargs):
                return sc.AnnData(X=np.zeros((adata.n_obs, 10)), obs=adata.obs.copy())

        model = StubModel(model_name="stub")
        adata = sc.AnnData(X=np.random.randn(5, 20))
        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as f:
            out = Path(f.name)
        try:
            with self.assertRaises(NotImplementedError) as ctx:
                model.compute_sensitivity(adata, out)
            self.assertIn("stub", str(ctx.exception))
            self.assertIn("sensitivity", str(ctx.exception).lower())
        finally:
            out.unlink(missing_ok=True)

    def test_pca_compute_sensitivity_raises(self) -> None:
        """PCA model does not support sensitivity analysis and raises."""
        from transcriptomic_fms.models.pca import PCAModel

        model = PCAModel(model_name="pca")
        adata = sc.AnnData(X=np.random.randn(3, 50))
        adata = model.preprocess(adata)
        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as f:
            out = Path(f.name)
        try:
            with self.assertRaises(NotImplementedError) as ctx:
                model.compute_sensitivity(adata, out)
            self.assertIn("pca", str(ctx.exception).lower())
        finally:
            out.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
