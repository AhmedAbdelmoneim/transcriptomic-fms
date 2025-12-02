#!/bin/bash
# Setup script for Compute Canada HPC
# This script builds the Apptainer container and sets up the environment

set -e

echo "========================================="
echo "transcriptomic-fms HPC Setup"
echo "========================================="

# Check if Apptainer is available
if ! command -v apptainer &> /dev/null; then
    echo "Error: Apptainer is not available. Please load the Apptainer module:"
    echo "  module load apptainer"
    exit 1
fi

echo "Apptainer version: $(apptainer --version)"

# Set project root (parent of hpc directory)
HPC_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJ_ROOT=$(cd "$HPC_DIR/.." && pwd)
cd "$PROJ_ROOT"

# Build Apptainer image
echo ""
echo "Building Apptainer image..."
APPTAINER_IMAGE="$PROJ_ROOT/transcriptomic-fms.sif"
if [ -f "$APPTAINER_IMAGE" ]; then
    echo "Warning: $APPTAINER_IMAGE already exists. Removing it..."
    rm -f "$APPTAINER_IMAGE"
fi

apptainer build "$APPTAINER_IMAGE" "$HPC_DIR/Singularity.def"

echo ""
echo "Apptainer image built successfully: $APPTAINER_IMAGE"

# Create data directories if they don't exist
echo ""
echo "Creating data directories..."
mkdir -p data
mkdir -p output
mkdir -p models

# Set up environment variables
echo ""
echo "Setting up environment..."
export APPTAINER_IMAGE="$APPTAINER_IMAGE"
export DATA_DIR="${DATA_DIR:-$PROJ_ROOT/data}"
export OUTPUT_DIR="${OUTPUT_DIR:-$PROJ_ROOT/output}"

# Create a wrapper script for running commands
cat > run_apptainer.sh << 'EOF'
#!/bin/bash
# Wrapper script to run commands in Apptainer container
# Supports GPU access via --nv flag when APPTAINER_USE_GPU is set

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Default to image in project root if APPTAINER_IMAGE not set
APPTAINER_IMAGE="${APPTAINER_IMAGE:-$PROJ_ROOT/transcriptomic-fms.sif}"
# If APPTAINER_IMAGE is a relative path, make it absolute relative to PROJ_ROOT
if [[ "$APPTAINER_IMAGE" != /* ]]; then
    APPTAINER_IMAGE="$PROJ_ROOT/$APPTAINER_IMAGE"
fi

# Build apptainer command
APPTAINER_CMD="apptainer exec"

# Add GPU support if requested
# Check multiple conditions for GPU availability
GPU_REQUESTED=0
if [ "${APPTAINER_USE_GPU:-0}" = "1" ]; then
    GPU_REQUESTED=1
elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "${CUDA_VISIBLE_DEVICES:-}" != "NoDevFiles" ]; then
    GPU_REQUESTED=1
elif [ -n "${SLURM_GPUS_ON_NODE:-}" ] && [ "${SLURM_GPUS_ON_NODE:-0}" -gt 0 ]; then
    GPU_REQUESTED=1
fi

if [ "$GPU_REQUESTED" = "1" ]; then
    # Add --nv flag for GPU support in Apptainer
    APPTAINER_CMD="$APPTAINER_CMD --nv"
    echo "GPU access enabled (--nv flag)"
    
    # Display GPU information if available
    if command -v nvidia-smi &> /dev/null; then
        echo "Available GPU(s):"
        nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    else
        echo "Note: nvidia-smi not found on host, but GPU access enabled in container"
    fi
fi

# Bind mount data, output, and models directories
$APPTAINER_CMD \
    --bind "$PROJ_ROOT/data:/transcriptomic-fms/data" \
    --bind "$PROJ_ROOT/output:/transcriptomic-fms/output" \
    --bind "$PROJ_ROOT/models:/transcriptomic-fms/models" \
    "$APPTAINER_IMAGE" \
    "$@"
EOF

chmod +x run_apptainer.sh

echo ""
echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "To run commands in the Apptainer container:"
echo "  ./run_apptainer.sh python -m transcriptomic_fms.cli.main embed --model pca --input data/test.h5ad --output output/embeddings.npy"
echo ""
echo "Or use the Makefile commands which will use the Apptainer container automatically."
echo ""

