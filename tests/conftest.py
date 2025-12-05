"""Shared pytest fixtures for transcriptomic-fms tests."""

import os
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest
import scanpy as sc

# Import model registry - models are loaded lazily


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Get project root directory."""
    # This file is in tests/, so project root is parent
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_data_dir(project_root: Path) -> Path:
    """Get test data directory."""
    data_dir = project_root / "tests" / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


@pytest.fixture(scope="session")
def small_adata(test_data_dir: Path) -> sc.AnnData:
    """
    Create a small synthetic AnnData for testing.

    Returns:
        AnnData with 100 cells and 1000 genes
    """
    # Create synthetic data: 100 cells, 1000 genes
    n_cells = 100
    n_genes = 1000

    # Generate random count-like data
    np.random.seed(42)
    X = np.random.negative_binomial(5, 0.3, size=(n_cells, n_genes)).astype(float)

    # Create gene names
    gene_names = [f"Gene_{i:04d}" for i in range(n_genes)]

    # Create AnnData
    adata = sc.AnnData(X)
    adata.var_names = gene_names
    adata.obs_names = [f"Cell_{i:04d}" for i in range(n_cells)]

    # Add some basic metadata
    adata.var["gene_symbols"] = gene_names

    return adata


@pytest.fixture
def tmp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def tmp_h5ad(small_adata: sc.AnnData, tmp_dir: Path) -> Path:
    """Save small_adata to a temporary h5ad file."""
    output_path = tmp_dir / "test_data.h5ad"
    small_adata.write(output_path)
    return output_path


@pytest.fixture(scope="session")
def is_hpc_environment() -> bool:
    """Check if running in HPC/container environment."""
    # Check for common HPC indicators
    return (
        os.environ.get("APPTAINER_CONTAINER") is not None
        or os.environ.get("SINGULARITY_CONTAINER") is not None
        or os.environ.get("SLURM_JOB_ID") is not None
        or Path("/.singularity.d").exists()
    )


@pytest.fixture(scope="session")
def has_gpu() -> bool:
    """Check if GPU is available."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False



