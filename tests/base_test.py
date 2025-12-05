"""Base test classes for model testing."""

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import scanpy as sc

from transcriptomic_fms.models.registry import get_model
from tests.utils import (
    detect_environment,
    get_test_config,
    run_apptainer_command,
    validate_embeddings,
)


class BaseModelTest:
    """Base test class for model smoke tests."""

    model_name: str = ""

    def test_model_smoke_local(
        self,
        small_adata: sc.AnnData,
        tmp_dir: Path,
    ):
        """Test model embedding locally (smoke test)."""
        env = detect_environment()
        config = get_test_config(self.model_name)

        # Skip if model doesn't support local testing
        if not config["supports_local"]:
            pytest.skip(f"{self.model_name} does not support local testing")

        # Skip if GPU required but not available
        if config["requires_gpu"] and not env["has_gpu"]:
            pytest.skip(f"{self.model_name} requires GPU but none available")

        # Create model with test config
        model = get_model(self.model_name, **config["test_kwargs"])

        # Use subset of data for faster testing
        n_test_cells = min(config["min_cells"], small_adata.n_obs)
        test_adata = small_adata[:n_test_cells].copy()

        # Preprocess
        preprocessed = model.preprocess(test_adata)

        # Generate embeddings
        output_path = tmp_dir / f"{self.model_name}_embeddings.npy"
        embeddings = model.embed(preprocessed, output_path)

        # Validate embeddings
        validate_embeddings(embeddings, n_test_cells)

        # Check output file exists
        assert output_path.exists(), "Embedding file should be created"

        # Try decode if supported
        if config["supports_decode"]:
            try:
                decoded = model.decode(embeddings)
                assert isinstance(decoded, sc.AnnData), "Decode should return AnnData"
                assert decoded.n_obs == n_test_cells, "Decoded should have same number of cells"
                assert decoded.n_vars > 0, "Decoded should have genes"
            except NotImplementedError:
                pytest.skip(f"{self.model_name} decode not implemented")

    def test_model_smoke_hpc(
        self,
        small_adata: sc.AnnData,
        tmp_dir: Path,
        project_root: Path,
    ):
        """Test model embedding in HPC/Apptainer environment."""
        env = detect_environment()
        config = get_test_config(self.model_name)

        # Skip if model doesn't support HPC testing
        if not config["supports_hpc"]:
            pytest.skip(f"{self.model_name} does not support HPC testing")

        # Skip if not in HPC environment or container not available
        if not env["has_apptainer"]:
            pytest.skip("Apptainer container not available")

        # Find container image - try model-specific first, then fallback to base
        model_class = get_model(self.model_name, **config["test_kwargs"]).__class__
        model_instance = model_class(model_name=self.model_name)
        
        # Try model-specific container first (e.g., transcriptomic-fms-scgpt.sif)
        container_image = model_instance.get_container_image_path(project_root)
        
        # Fallback to base container if model doesn't have its own
        if container_image is None:
            container_image = project_root / "transcriptomic-fms.sif"
        
        # Check if container exists
        if not container_image.exists():
            # Provide helpful error message
            model_container_name = model_instance.get_container_name()
            if model_container_name:
                pytest.skip(
                    f"Model-specific container not found: {container_image}\n"
                    f"Build it with: make build-model-container MODEL={self.model_name}\n"
                    f"Or use base container: make setup-hpc"
                )
            else:
                pytest.skip(
                    f"Base container not found: {container_image}\n"
                    f"Build it with: make setup-hpc"
                )

        # Save test data (inside project directory so container can access it)
        project_test_dir = project_root / "tests" / "tmp"
        project_test_dir.mkdir(parents=True, exist_ok=True)
        test_h5ad = project_test_dir / "test_input.h5ad"
        small_adata.write(test_h5ad)

        # Prepare command
        output_path = project_test_dir / f"{self.model_name}_embeddings.npy"
        # Paths inside container (relative to project root)
        container_input = Path("/transcriptomic-fms") / "tests" / "tmp" / "test_input.h5ad"
        container_output = (
            Path("/transcriptomic-fms") / "tests" / "tmp" / f"{self.model_name}_embeddings.npy"
        )

        # Build model command
        model = get_model(self.model_name, **config["test_kwargs"])
        command = model.get_container_command(
            adata_path=container_input,
            output_path=container_output,
        )

        # Run in container
        result = run_apptainer_command(
            command=command,
            container_image=container_image,
            gpu=config["requires_gpu"] and env["has_gpu"],
            project_root=project_root,
        )

        # Check command succeeded
        assert result.returncode == 0, (
            f"Container command failed:\n"
            f"Command: {' '.join(command)}\n"
            f"STDOUT: {result.stdout}\n"
            f"STDERR: {result.stderr}"
        )

        # Check output file exists
        assert output_path.exists(), f"Embedding file should be created: {output_path}"

        # Load and validate embeddings
        embeddings = np.load(output_path)
        # Note: container test uses full dataset, not subset
        validate_embeddings(embeddings, small_adata.n_obs)

