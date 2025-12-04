# flash-attn Installation

flash-attn is automatically installed during container build. The CUDA toolkit is included in the container, so the build can be done on any node (no GPU node required for build).

## Build Time

The container build takes **30-60 minutes longer** due to flash-attn compilation. This is a one-time cost - flash-attn is baked into the container image.

## What Gets Installed

- CUDA toolkit (for nvcc compiler) - installed in container during build
- flash-attn - compiled and installed in container during build

## Verification

After building, verify flash-attn is installed:

```bash
apptainer exec transcriptomic-fms.sif python -c "import flash_attn; print(flash_attn.__version__)"
```
