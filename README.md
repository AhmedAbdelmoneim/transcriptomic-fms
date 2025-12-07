# Transcriptomic Foundation Models

Unified interface for generating embeddings from single-cell foundation models. Supports local execution and HPC deployment via Apptainer containers.

## Quick Start

### Installation

```bash
# Create environment and install base dependencies
make create_environment
make requirements

# Install model-specific dependencies (optional, for local use)
make install-model MODEL=scgpt
# or
uv sync --extra scgpt
```

### List Available Models

```bash
make list-models
```

### Generate Embeddings

**Locally:**
```bash
make embed MODEL=pca INPUT=data/test.h5ad OUTPUT=output/embeddings.npy
```

**On HPC (interactive):**
```bash
# Get interactive GPU node first
salloc --time=4:00:00 --nodes=1 --cpus-per-task=8 --mem=64G \
    --account=def-jagillis --gres=gpu:1

# Run embedding
make hpc-embed-interactive MODEL=scgpt \
    INPUT=data/test.h5ad \
    OUTPUT=output/embeddings.npy \
    MODEL_ARGS="--device cuda"
```

**On HPC (batch job):**
```bash
make hpc-embed MODEL=scgpt \
    INPUT=data/test.h5ad \
    OUTPUT=output/embeddings.npy \
    MODEL_ARGS="--device cuda"
```

**Note:** Output files are automatically prefixed with the model name (e.g., `scgpt_embeddings.npy`).

## Available Models

### PCA

Baseline PCA embedding with optional HVG selection.

```bash
make embed MODEL=pca INPUT=data/test.h5ad OUTPUT=output/embeddings.npy \
    MODEL_ARGS="--n-components 100 --use-hvg --n-hvg 2000"
```

**Arguments:**
- `--n-components <int>`: Number of PCA components (default: 50)
- `--use-hvg`: Enable highly variable gene selection
- `--n-hvg <int>`: Number of HVGs to select (default: 2000)

### scGPT

scGPT foundation model for single-cell embeddings.

**Installation:**
```bash
make install-model MODEL=scgpt
```

**Arguments:**
- `--model-dir <path>`: Path to scGPT model directory (auto-downloads if not provided)
- `--n-hvg <int>`: Number of HVGs (default: 1200, use 0 for all genes)
- `--hvg-list <path>`: File with pre-computed HVG list (one gene per line)
- `--n-bins <int>`: Binning resolution (default: 51)
- `--batch-size <int>`: Batch size (default: 64)
- `--device <str>`: 'cuda' or 'cpu' (auto-detects if not specified)

**Examples:**
```bash
# Auto-download model
make embed MODEL=scgpt INPUT=data/test.h5ad OUTPUT=output/embeddings.npy

# Use existing model
make embed MODEL=scgpt INPUT=data/test.h5ad OUTPUT=output/embeddings.npy \
    MODEL_ARGS="--model-dir /path/to/scGPT_human --batch-size 32"

# Use all genes (no HVG filtering)
make embed MODEL=scgpt INPUT=data/test.h5ad OUTPUT=output/embeddings.npy \
    MODEL_ARGS="--n-hvg 0"
```

### SCimilarity

SCimilarity model for cell state similarity embeddings.

**Installation:**
```bash
make install-model MODEL=scimilarity
```

**Arguments:**
- `--model-path <path>`: Path to SCimilarity model directory (auto-downloads if not provided)
- `--device <str>`: 'cuda' or 'cpu' (auto-detects if not specified)
- `--use-gpu <bool>`: Explicitly enable/disable GPU

**Examples:**
```bash
# Auto-download model (~30GB, may take time)
make embed MODEL=scimilarity INPUT=data/test.h5ad OUTPUT=output/embeddings.npy

# Use existing model
make embed MODEL=scimilarity INPUT=data/test.h5ad OUTPUT=output/embeddings.npy \
    MODEL_ARGS="--model-path /path/to/scimilarity/model_v1.1"
```

**Note:** SCimilarity expects raw counts (not log-normalized). The model will automatically log-normalize during preprocessing.

### Geneformer

Geneformer foundation transformer model for single-cell embeddings (V1-10M, cell embeddings).

**Installation:**
```bash
make install-model MODEL=geneformer
```

**Arguments:**
- `--model-path <path>`: Path to Geneformer model directory (auto-downloads if not provided)
- `--batch-size <int>`: Batch size for forward pass (default: 100)
- `--device <str>`: 'cuda' or 'cpu' (auto-detects if not specified)

**Examples:**
```bash
# Auto-download model (requires git-lfs)
make embed MODEL=geneformer INPUT=data/test.h5ad OUTPUT=output/embeddings.npy

# Use existing model with custom batch size
make embed MODEL=geneformer INPUT=data/test.h5ad OUTPUT=output/embeddings.npy \
    MODEL_ARGS="--model-path /path/to/Geneformer --batch-size 50"

# On HPC with custom batch size
make hpc-embed-interactive MODEL=geneformer \
    INPUT=data/test.h5ad \
    OUTPUT=output/embeddings.npy \
    MODEL_ARGS="--batch-size 50 --device cuda"
```

**Note:** 
- Geneformer requires Ensembl IDs (not gene symbols) in `var.index` or `var['ensembl_id']` column
- Data should be raw counts with `n_counts` attribute per cell (computed automatically if missing)
- Tokenization creates intermediate .dataset files automatically
- Model auto-download requires git-lfs to be installed

## HPC Deployment

### Building Containers

```bash
module load apptainer
make build-container MODEL=scgpt
make build-container MODEL=scimilarity
make build-container MODEL=geneformer
```

### Container Updates

Containers use editable installs, so code changes are picked up automatically without rebuilding. However, if you add new dependencies or change container configuration, rebuild the container.

### SLURM Job Options

Default options in `run_job.sh`:
- `--time=4:00:00`
- `--mem=64G`
- `--cpus-per-task=8`
- `--gres=gpu:1`

Override by passing options to `sbatch`:
```bash
sbatch --time=8:00:00 --mem=128G transcriptomic_fms/hpc/run_job.sh embed ...
```

## Data Requirements

All models require AnnData objects (`.h5ad` files) with gene identifiers:
- **Most models**: Gene symbols in `var.index`, `var['gene_symbols']`, `var['feature_name']`, or `var['gene_name']`
- **Geneformer**: Requires Ensembl IDs in `var.index` or `var['ensembl_id']` column

## Architecture

Models inherit from `BaseEmbeddingModel` and implement:
- `preprocess()`: Model-specific preprocessing
- `embed()`: Generate embeddings
- `decode()`: (Optional) Decode embeddings back to expression

Models are auto-registered via the `@register_model` decorator.

## CLI Reference

```bash
# Embed command
python -m transcriptomic_fms.cli.main embed \
    --model <model_name> \
    --input <path/to/input.h5ad> \
    --output <path/to/output.npy> \
    [--model-arg-name <value>]

# List models
python -m transcriptomic_fms.cli.main list
```

Model-specific arguments are passed as `--key=value`, `--key value`, or `--flag`.

## Authors

- Ahmed Sallam
