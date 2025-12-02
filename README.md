# Transcriptomic Foundation Models

Common interface for generating embeddings from transcriptomic foundation models. Provides a unified API for running various single-cell foundation models, with support for both local execution and HPC cluster deployment via Apptainer containers.

## Quick Start

### Installation

```bash
uv sync
```

### List Available Models

```bash
make list-models
```

### Generate Embeddings Locally

```bash
# Basic usage
make embed MODEL=pca INPUT=data/test.h5ad OUTPUT=output/embeddings.npy

# With model-specific arguments
make embed MODEL=pca INPUT=data/test.h5ad OUTPUT=output/embeddings.npy \
    MODEL_ARGS="--n-components 100 --use-hvg --n-hvg 2000"
```

### Generate Embeddings on HPC

1. **Set up HPC environment** (one-time):
```bash
make setup-hpc
```

2. **Submit a job**:
```bash
# Using Makefile
make hpc-embed MODEL=pca INPUT=data/test.h5ad OUTPUT=output/embeddings.npy \
    MODEL_ARGS="--n-components 100"

# Or directly with sbatch
sbatch transcriptomic_fms/hpc/run_job.sh embed \
    --model pca \
    --input data/test.h5ad \
    --output output/embeddings.npy \
    --n-components 100
```

## Architecture

### Model Interface

All models inherit from `BaseEmbeddingModel` and implement:
- `preprocess()`: Preprocess AnnData for the model
- `embed()`: Generate embeddings from preprocessed data
- `decode()`: (Optional) Decode embeddings back to expression space

### Adding a New Model

```python
from transcriptomic_fms.models.base import BaseEmbeddingModel
from transcriptomic_fms.models.registry import register_model

@register_model("my_model")
class MyModel(BaseEmbeddingModel):
    def preprocess(self, adata, output_path=None):
        return adata
    
    def embed(self, adata, output_path, **kwargs):
        return embeddings
```

The model will be automatically registered and available via CLI.

## CLI Reference

### Embed Command

```bash
python -m transcriptomic_fms.cli.main embed \
    --model <model_name> \
    --input <path/to/input.h5ad> \
    --output <path/to/output.npy> \
    [--model-arg-name <value>] \
    ...
```

**Model-Specific Arguments**: Any arguments after the required ones are passed to the model. Use `--key=value`, `--key value`, or `--flag` format. Dashes are converted to underscores.

### List Command

```bash
python -m transcriptomic_fms.cli.main list
```

## Available Models

### PCA

Baseline PCA embedding with optional HVG selection.

**Arguments:**
- `--n-components <int>`: Number of PCA components (default: 50)
- `--use-hvg`: Enable highly variable gene selection
- `--n-hvg <int>`: Number of HVGs to select (default: 2000)

**Examples:**
```bash
make embed MODEL=pca INPUT=data/test.h5ad OUTPUT=output/pca.npy
make embed MODEL=pca INPUT=data/test.h5ad OUTPUT=output/pca.npy \
    MODEL_ARGS="--n-components 100 --use-hvg"
```

## HPC Details

### Default SLURM Options

The `run_job.sh` script includes these defaults:
- `--time=4:00:00`
- `--mem=64G`
- `--cpus-per-task=8`
- `--gres=gpu:1`

Override by passing options to `sbatch`:
```bash
sbatch --time=8:00:00 --mem=128G transcriptomic_fms/hpc/run_job.sh embed ...
```

### GPU Support

GPU access is automatically detected from SLURM environment variables. The `--gres=gpu:1` flag is included by default in `run_job.sh`.

## Development

```bash
# Lint
make lint

# Format
make format
```

## License

[Your License Here]

## Authors

- Ahmed Sallam
