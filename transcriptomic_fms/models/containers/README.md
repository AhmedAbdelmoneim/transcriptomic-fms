# Model-Specific Containers

This directory contains model-specific container definitions for HPC deployment.

## Structure

Each model has its own self-contained container definition:

```
models/containers/
├── scgpt/
│   └── Singularity.def    # scGPT container (self-contained)
├── pca/
│   └── Singularity.def   # PCA container (self-contained)
└── ...
```

## Container Structure

Each model container is self-contained and includes:
- System dependencies (HDF5, CUDA, etc.)
- Core Python dependencies (scanpy, numpy, pandas, etc.)
- PyTorch with CUDA support
- flash-attn (installed using pre-built wheels - fast)
- Model-specific dependencies
- Base package installation

**Note:** Each container is independent - no inheritance or base containers. This simplifies maintenance and avoids dependency conflicts.

## Building Model Containers

```bash
# Build scGPT container
make build-container MODEL=scgpt

# Containers are built from the project root
# File paths in Singularity.def are relative to the build context
```

## Usage

When submitting HPC jobs, specify the model name and the system will use the corresponding container:

```bash
make hpc-embed-interactive MODEL=scgpt INPUT=data/test.h5ad OUTPUT=output/embeddings.npy
```

The container `transcriptomic-fms-scgpt.sif` will be used automatically.
