"""Utility functions for testing."""

import os
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np


def detect_environment() -> dict[str, bool]:
    """
    Detect the test environment.

    Returns:
        Dictionary with environment flags:
        - is_hpc: True if running in HPC/container environment
        - has_gpu: True if GPU is available
        - has_apptainer: True if Apptainer container available
    """
    is_hpc = (
        os.environ.get("APPTAINER_CONTAINER") is not None
        or os.environ.get("SINGULARITY_CONTAINER") is not None
        or os.environ.get("SLURM_JOB_ID") is not None
        or Path("/.singularity.d").exists()
    )

    has_gpu = False
    try:
        import torch

        has_gpu = torch.cuda.is_available()
    except ImportError:
        pass

    has_apptainer = False
    apptainer_image = os.environ.get("APPTAINER_IMAGE")
    if apptainer_image and Path(apptainer_image).exists():
        has_apptainer = True
    else:
        # Check for default container
        project_root = Path(__file__).parent.parent
        default_image = project_root / "transcriptomic-fms.sif"
        if default_image.exists():
            has_apptainer = True

    return {
        "is_hpc": is_hpc,
        "has_gpu": has_gpu,
        "has_apptainer": has_apptainer,
    }


def validate_embeddings(embeddings: np.ndarray, expected_n_cells: int) -> None:
    """
    Validate embeddings array.

    Args:
        embeddings: Embeddings array to validate
        expected_n_cells: Expected number of cells

    Raises:
        AssertionError: If embeddings are invalid
    """
    assert isinstance(embeddings, np.ndarray), "Embeddings must be numpy array"
    assert embeddings.ndim == 2, f"Embeddings must be 2D, got {embeddings.ndim}D"
    assert embeddings.shape[0] == expected_n_cells, (
        f"Expected {expected_n_cells} cells, got {embeddings.shape[0]}"
    )
    assert not np.any(np.isnan(embeddings)), "Embeddings contain NaN values"
    assert not np.any(np.isinf(embeddings)), "Embeddings contain Inf values"
    assert embeddings.dtype in [np.float32, np.float64], (
        f"Embeddings should be float, got {embeddings.dtype}"
    )


def run_apptainer_command(
    command: list[str],
    container_image: Path,
    gpu: bool = False,
    project_root: Optional[Path] = None,
) -> subprocess.CompletedProcess:
    """
    Run a command inside an Apptainer container.

    Args:
        command: Command to run (list of strings)
        container_image: Path to container image
        gpu: Whether to enable GPU access
        project_root: Project root directory (for bind mounts)

    Returns:
        CompletedProcess from subprocess.run
    """
    if project_root is None:
        project_root = Path(__file__).parent.parent

    apptainer_cmd = ["apptainer", "exec"]

    if gpu:
        apptainer_cmd.append("--nv")

    # Bind mount project directory
    apptainer_cmd.extend(["--bind", f"{project_root}:/transcriptomic-fms"])

    apptainer_cmd.append(str(container_image))
    apptainer_cmd.extend(command)

    return subprocess.run(
        apptainer_cmd,
        capture_output=True,
        text=True,
        check=False,
    )


def get_test_config(model_name: str) -> dict:
    """
    Get test configuration for a model.

    Args:
        model_name: Name of the model

    Returns:
        Test configuration dictionary

    Raises:
        ValueError: If model not found in test configs
    """
    configs = {
        "pca": {
            "supports_local": True,
            "supports_hpc": True,
            "requires_gpu": False,
            "test_kwargs": {"n_components": 10, "use_hvg": False},
            "supports_decode": False,
            "min_cells": 10,
        },
        "scgpt": {
            "supports_local": True,
            "supports_hpc": True,
            "requires_gpu": True,
            "test_kwargs": {"n_hvg": 100, "batch_size": 32},
            "supports_decode": False,
            "min_cells": 10,
        },
    }

    if model_name not in configs:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(configs.keys())}")

    return configs[model_name]

