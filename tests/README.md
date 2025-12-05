# Test Suite

Functional/end-to-end tests for transcriptomic-fms models.

## Overview

This test suite provides smoke tests for each model:
- **Local testing**: Run models directly (no container)
- **HPC testing**: Run models in Apptainer containers with GPU support

Each test:
1. Creates a small synthetic dataset
2. Runs preprocessing
3. Generates embeddings
4. Validates output (shape, no NaN/Inf, etc.)
5. Optionally tests decoding (if supported)

## Running Tests

### Run all tests locally

```bash
# Install test dependencies
uv sync --extra dev

# Run all tests
pytest tests/

# Run specific model tests
pytest tests/test_models.py::TestModels::test_model_smoke_local -k pca
```

### Run tests in HPC environment

```bash
# Ensure container is built
make setup-hpc

# Set environment variable for container
export APPTAINER_IMAGE=./transcriptomic-fms.sif

# Run HPC tests
pytest tests/test_models.py::TestModels::test_model_smoke_hpc -k pca
```

### Run specific models

```bash
# Test only PCA
TEST_MODELS=pca pytest tests/

# Test only scGPT
TEST_MODELS=scgpt pytest tests/
```

### Using Makefile

```bash
# Run all local tests
make test

# Run HPC tests
make test-hpc

# Run tests for specific model
make test MODEL=pca
```

## Test Configuration

Test configurations are defined in `tests/utils.py` in the `get_test_config()` function. Each model has:

- `supports_local`: Can test locally
- `supports_hpc`: Can test in containers
- `requires_gpu`: Needs GPU
- `test_kwargs`: Default model parameters for testing
- `supports_decode`: Model has decoder
- `min_cells`: Minimum cells needed for test

## Container Images

The test suite automatically handles model-specific containers:

- **Model-specific containers** (e.g., `transcriptomic-fms-scgpt.sif`): Used if available
- **Base container** (`transcriptomic-fms.sif`): Used as fallback for models without their own container

Models indicate their container via `get_container_name()` method:
- `scgpt` model → looks for `transcriptomic-fms-scgpt.sif`
- `pca` model (no container name) → uses base `transcriptomic-fms.sif`

If a model-specific container is expected but not found, the test will skip with instructions to build it.

## Adding New Models

To add tests for a new model:

1. **Add test config** in `tests/utils.py::get_test_config()`:

```python
"my_model": {
    "supports_local": True,
    "supports_hpc": True,
    "requires_gpu": False,
    "test_kwargs": {"param1": "value1"},
    "supports_decode": False,
    "min_cells": 10,
}
```

2. **Create test file** `tests/test_my_model.py`:

   - Copy `tests/test_template.py` to `tests/test_my_model.py`
   - Update the class name and `model_name` attribute
   - (Optional) Add model-specific tests

   Example:
```python
"""Tests for MyModel."""

from tests.base_test import BaseModelTest

class TestMyModel(BaseModelTest):
    """Tests for MyModel embedding model."""
    
    model_name = "my_model"
    
    # (Optional) Add model-specific tests here
    def test_something_specific(self, small_adata, tmp_dir):
        """Test something unique to this model."""
        # Your test code here
        pass
```

3. **Done!** The base class provides smoke tests automatically. Add model-specific tests as needed.

## Test Structure

```
tests/
├── __init__.py          # Package init
├── conftest.py          # Shared fixtures
├── base_test.py         # Base test class for models
├── utils.py             # Test utilities and configs
├── test_pca.py          # PCA-specific tests
├── test_scgpt.py        # scGPT-specific tests
├── test_template.py     # Template for adding new model tests
└── README.md           # This file
```

**Key Design:**
- **Base test class** (`base_test.py`): Contains shared smoke tests (local + HPC)
- **Per-model files** (`test_<model>.py`): Model-specific tests inherit from base class
- **Easy to scale**: Just create `test_<model>.py` for each new model

## Environment Detection

Tests automatically detect:
- **HPC environment**: Checks for `APPTAINER_CONTAINER`, `SLURM_JOB_ID`, etc.
- **GPU availability**: Checks `torch.cuda.is_available()`
- **Container availability**: Checks for container images

Tests will skip appropriately if requirements aren't met.

