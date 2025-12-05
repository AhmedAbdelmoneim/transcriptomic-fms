"""Template for adding tests for a new model.

Copy this file to test_<model_name>.py and customize as needed.
"""

from pathlib import Path

import pytest
import scanpy as sc

from transcriptomic_fms.models.registry import get_model
from tests.base_test import BaseModelTest
from tests.utils import validate_embeddings


class TestMyModel(BaseModelTest):
    """Tests for MyModel embedding model."""

    model_name = "my_model"  # Change this to your model name

    # Optional: Add model-specific tests here
    # The base class already provides:
    # - test_model_smoke_local
    # - test_model_smoke_hpc

    def test_something_specific(
        self,
        small_adata: sc.AnnData,
        tmp_dir: Path,
    ):
        """Test something unique to this model."""
        model = get_model(self.model_name)
        # Your test code here
        pass

