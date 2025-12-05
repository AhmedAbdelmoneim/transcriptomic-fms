# Transcriptomic Foundation Models

Common interface for generating embeddings from transcriptomic foundation models. Provides a unified API for running various single-cell foundation models, with support for both local execution and HPC cluster deployment via Apptainer containers.

## Quick Start

### Installation

```bash
# Install base dependencies
make create_environment
make requirements

# Install dependencies for specific models
make install-model MODEL=scgpt
# or
uv sync --extra scgpt

# On HPC clusters (Compute Canada), load CUDA module before installing scgpt:
# module load cuda/11.7  # or appropriate CUDA version
# make install-model MODEL=scgpt
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

2. **Build model-specific container** (for scgpt, etc.):
```bash
module load apptainer
make build-model-container MODEL=scgpt
```

3. **Run interactively** (for testing/debugging):
```bash
# First, get an interactive GPU node
salloc --time=4:00:00 --nodes=1 --cpus-per-task=8 --mem=64G \
    --account=def-jagillis --gres=gpu:1

# Once you have the node, run:
make hpc-embed-interactive MODEL=scgpt \
    INPUT=data/test.h5ad \
    OUTPUT=output/embeddings.npy \
    MODEL_ARGS="--device cuda"

# Or manually:
export APPTAINER_USE_GPU=1
./run_apptainer.sh python -m transcriptomic_fms.cli.main embed \
    --model scgpt \
    --input data/test.h5ad \
    --output output/embeddings.npy \
    --device cuda
```

4. **Submit a batch job** (for production):
```bash
# Using Makefile
make hpc-embed MODEL=scgpt INPUT=data/test.h5ad OUTPUT=output/embeddings.npy \
    MODEL_ARGS="--device cuda"

# Or directly with sbatch
sbatch transcriptomic_fms/hpc/run_job.sh embed \
    --model scgpt \
    --input data/test.h5ad \
    --output output/embeddings.npy \
    --device cuda
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

## Testing

Run functional tests to verify models work correctly:

```bash
# Install test dependencies
make install-test

# Run all tests locally
make test

# Test specific model
make test MODEL=pca

# Run HPC/container tests (requires container built)
make test-hpc MODEL=scgpt
```

Tests automatically:
- Create synthetic test data
- Run preprocessing and embedding
- Validate output (shape, no NaN/Inf, etc.)
- Test decoding if supported

See [tests/README.md](tests/README.md) for details.

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

### scGPT

scGPT foundation model embeddings. Requires scGPT installation and model checkpoint.

**Installation:**
```bash
# Recommended: use make command
make install-model MODEL=scgpt

# Or manually
uv sync --extra scgpt
# or
pip install scgpt "torch<=2.2.2" "numpy<2" gdown
```

**Arguments:**
- `--model-dir <path>`: Path to scGPT model directory (must contain `best_model.pt`, `args.json`, `vocab.json`). 
  If not provided and `--auto-download` is True, model will be downloaded automatically.
- `--n-hvg <int>`: Number of highly variable genes to select (default: 1200). Ignored if `--hvg-list` is provided.
  Set to 0 or omit to use all genes.
- `--hvg-list <path>`: Path to file containing pre-computed HVG list (one gene per line). 
  If provided, `--n-hvg` is ignored. The file should contain one gene symbol per line.
- `--n-bins <int>`: Number of bins for value binning (default: 51)
- `--batch-size <int>`: Batch size for embedding (default: 64)
- `--device <str>`: Device to use ('cuda' or 'cpu', auto-detects if not specified)
- `--auto-download`: Automatically download model if not found (default: True)
- `--download-dir <path>`: Directory to download model to (default: `models/scGPT_human` in project root)

**Gene Symbol Requirements:**
The input AnnData must have gene symbols available in one of these formats:
- `var.index` contains gene symbols (most common)
- `var['gene_symbols']` column exists
- `var['gene_name']` column exists (fallback)

**Examples:**
```bash
# Auto-download model (default behavior)
make embed MODEL=scgpt INPUT=data/test.h5ad OUTPUT=output/scgpt.npy

# Use existing model directory
make embed MODEL=scgpt INPUT=data/test.h5ad OUTPUT=output/scgpt.npy \
    MODEL_ARGS="--model-dir /path/to/scGPT_human --batch-size 32"

# Use pre-computed HVG list from file
make embed MODEL=scgpt INPUT=data/test.h5ad OUTPUT=output/scgpt.npy \
    MODEL_ARGS="--hvg-list /path/to/hvg_genes.txt"

# Use all genes (no HVG filtering)
make embed MODEL=scgpt INPUT=data/test.h5ad OUTPUT=output/scgpt.npy \
    MODEL_ARGS="--n-hvg 0"

# Disable auto-download
make embed MODEL=scgpt INPUT=data/test.h5ad OUTPUT=output/scgpt.npy \
    MODEL_ARGS="--model-dir /path/to/scGPT_human --no-auto-download"
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

## Authors

- Ahmed Sallam
