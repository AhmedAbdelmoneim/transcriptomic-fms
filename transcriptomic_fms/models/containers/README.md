# Model-Specific Containers

This directory contains model-specific container definitions for HPC deployment.

## Structure

Each model can have its own container definition:

```
models/containers/
├── scgpt/
│   └── Singularity.def    # scGPT-specific container
├── pca/
│   └── Singularity.def   # PCA container (optional, can use base)
└── ...
```

## Base Container

The base container in `transcriptomic_fms/hpc/Singularity.def` includes:
- Core dependencies (scanpy, numpy, pandas, etc.)
- PyTorch with CUDA support
- flash-attn (compiled, takes 30-60 minutes but reusable)
- Base package installation

**Note:** The base container includes GPU dependencies so they only need to be compiled once. Model-specific containers extend this base to avoid recompiling flash-attn.

## Model-Specific Containers

Model-specific containers:
- **Extend the base container** using `Bootstrap: localimage` (avoids recompiling flash-attn)
- Add model-specific dependencies (e.g., scGPT)
- Include model checkpoints
- Customize environment variables

**Building order:**
1. First build the base container: `make setup-hpc` (includes flash-attn compilation)
2. Then build model containers: `make build-model-container MODEL=scgpt` (fast, extends base)

## Building Model Containers

```bash
# Build scGPT container
make build-model-container MODEL=scgpt

# Build all model containers
make build-all-containers
```

## Usage

When submitting HPC jobs, the system will automatically use the model-specific container if available, otherwise falls back to the base container.

