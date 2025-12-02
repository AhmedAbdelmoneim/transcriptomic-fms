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
- Base package installation

## Model-Specific Containers

Model-specific containers can:
- Extend the base container
- Add model-specific dependencies (e.g., scGPT, torch with specific CUDA version)
- Include model checkpoints
- Customize environment variables

## Building Model Containers

```bash
# Build scGPT container
make build-model-container MODEL=scgpt

# Build all model containers
make build-all-containers
```

## Usage

When submitting HPC jobs, the system will automatically use the model-specific container if available, otherwise falls back to the base container.

