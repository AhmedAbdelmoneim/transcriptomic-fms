## Variables
PYTHON := python
UV := uv

## Default target
.DEFAULT_GOAL := help

## Help target - shows all available commands
.PHONY: help
help:
	@echo "Available targets:"
	@echo ""
	@grep -E '^##' $(MAKEFILE_LIST) | sed 's/^## //' | sed 's/^##//'

## Create Python environment using uv
.PHONY: create_environment
create_environment:
	$(UV) venv

## Install project dependencies
.PHONY: requirements
requirements:
	$(UV) pip install -e .

## List available models
.PHONY: list-models
list-models:
	$(UV) run python -m transcriptomic_fms.cli.main list

## Generate embeddings locally
## Usage: make embed MODEL=<model_name> INPUT=<path/to/input.h5ad> OUTPUT=<path/to/output.npy> [MODEL_ARGS="--arg1 value1 --arg2"]
## Example: make embed MODEL=pca INPUT=data/test.h5ad OUTPUT=output/embeddings.npy MODEL_ARGS="--n-components 100 --use-hvg"
.PHONY: embed
embed:
	@if [ -z "$(MODEL)" ] || [ -z "$(INPUT)" ] || [ -z "$(OUTPUT)" ]; then \
		echo "Usage: make embed MODEL=<model_name> INPUT=<path/to/input.h5ad> OUTPUT=<path/to/output.npy> [MODEL_ARGS=\"--arg1 value1 --arg2\"]"; \
		echo ""; \
		echo "Available models:"; \
		uv run python -m transcriptomic_fms.cli.main list; \
		echo ""; \
		echo "Examples:"; \
		echo "  make embed MODEL=pca INPUT=data/test.h5ad OUTPUT=output/embeddings.npy"; \
		echo "  make embed MODEL=pca INPUT=data/test.h5ad OUTPUT=output/embeddings.npy MODEL_ARGS=\"--n-components 100 --use-hvg\""; \
		exit 1; \
	fi
	uv run python -m transcriptomic_fms.cli.main embed \
		--model $(MODEL) \
		--input $(INPUT) \
		--output $(OUTPUT) \
		$(MODEL_ARGS)

## Build model container for HPC
## Usage: make build-container MODEL=<model_name>
## Example: make build-container MODEL=scgpt
.PHONY: build-container
build-container:
	@if [ -z "$(MODEL)" ]; then \
		echo "Usage: make build-container MODEL=<model_name>"; \
		echo ""; \
		echo "Available models with containers:"; \
		ls -d transcriptomic_fms/models/containers/*/ 2>/dev/null | sed 's|transcriptomic_fms/models/containers/||; s|/||' || echo "  (none found)"; \
		exit 1; \
	fi
	@if [ ! -f "transcriptomic_fms/models/containers/$(MODEL)/Singularity.def" ]; then \
		echo "Error: Container definition not found for model $(MODEL)"; \
		echo "Expected: transcriptomic_fms/models/containers/$(MODEL)/Singularity.def"; \
		exit 1; \
	fi
	@if ! command -v apptainer &> /dev/null; then \
		echo "Error: Apptainer is not available. Please load the Apptainer module:"; \
		echo "  module load apptainer"; \
		exit 1; \
	fi
	@echo "Building container for model $(MODEL)..."; \
	CONTAINER_NAME="transcriptomic-fms-$(MODEL).sif"; \
	CONTAINER_DEF="transcriptomic_fms/models/containers/$(MODEL)/Singularity.def"; \
	if [ -f "$$CONTAINER_NAME" ]; then \
		echo "Warning: $$CONTAINER_NAME already exists. Removing it..."; \
		rm -f "$$CONTAINER_NAME"; \
	fi; \
	echo "Building from project root (file paths in Singularity.def are relative to build context)"; \
	echo "Container definition: $$CONTAINER_DEF"; \
	echo "Output container: $$CONTAINER_NAME"; \
	apptainer build "$$CONTAINER_NAME" "$$CONTAINER_DEF" && \
	echo "Container built successfully: $$CONTAINER_NAME"

## Run embedding interactively on HPC (requires interactive node via salloc)
## Usage: make hpc-embed-interactive MODEL=<model_name> INPUT=<path/to/input.h5ad> OUTPUT=<path/to/output.npy> [MODEL_ARGS="--arg1 value1 --arg2"]
## Note: Run this after getting an interactive node with: salloc --gres=gpu:1 ...
.PHONY: hpc-embed-interactive
hpc-embed-interactive:
	@if [ -z "$(MODEL)" ] || [ -z "$(INPUT)" ] || [ -z "$(OUTPUT)" ]; then \
		echo "Usage: make hpc-embed-interactive MODEL=<model_name> INPUT=<path/to/input.h5ad> OUTPUT=<path/to/output.npy> [MODEL_ARGS=\"--arg1 value1 --arg2\"]"; \
		echo ""; \
		echo "Example:"; \
		echo "  # First, get an interactive GPU node:"; \
		echo "  salloc --time=4:00:00 --nodes=1 --cpus-per-task=8 --mem=64G --account=def-jagillis --gres=gpu:1"; \
		echo ""; \
		echo "  # Then run:"; \
		echo "  make hpc-embed-interactive MODEL=scgpt INPUT=data/test.h5ad OUTPUT=output/embeddings.npy MODEL_ARGS=\"--device cuda\""; \
		exit 1; \
	fi
	@MODEL_NAME="$(MODEL)"; \
	CONTAINER="transcriptomic-fms-$$MODEL_NAME.sif"; \
	if [ ! -f "$$CONTAINER" ]; then \
		echo "Error: Container not found: $$CONTAINER"; \
		echo "Build it with: make build-container MODEL=$$MODEL_NAME"; \
		exit 1; \
	fi
	@export PYTHONNOUSERSITE=1; \
	export APPTAINER_USE_GPU=1; \
	apptainer exec --nv \
		--bind "$(shell pwd)/data:/transcriptomic-fms/data" \
		--bind "$(shell pwd)/output:/transcriptomic-fms/output" \
		--bind "$(shell pwd)/models:/transcriptomic-fms/models" \
		"transcriptomic-fms-$(MODEL).sif" \
		python -m transcriptomic_fms.cli.main embed \
		--model $(MODEL) \
		--input $(INPUT) \
		--output $(OUTPUT) \
		$(MODEL_ARGS)

## Check GPU compatibility and availability
## Usage: make check-gpu MODEL=<model_name>
## This checks GPU access both on host and inside container
.PHONY: check-gpu
check-gpu:
	@if [ -z "$(MODEL)" ]; then \
		echo "Usage: make check-gpu MODEL=<model_name>"; \
		echo ""; \
		echo "Example: make check-gpu MODEL=scgpt"; \
		exit 1; \
	fi
	@MODEL_NAME="$(MODEL)"; \
	CONTAINER="transcriptomic-fms-$$MODEL_NAME.sif"; \
	if [ ! -f "$$CONTAINER" ]; then \
		echo "Error: Container not found: $$CONTAINER"; \
		echo "Build it with: make build-container MODEL=$$MODEL_NAME"; \
		exit 1; \
	fi
	@echo "========================================="; \
	echo "GPU Compatibility Diagnostics"; \
	echo "========================================="; \
	echo ""; \
	echo "1. HOST SYSTEM GPU CHECK:"; \
	echo "------------------------"; \
	if command -v nvidia-smi &> /dev/null; then \
		echo "✓ nvidia-smi available"; \
		echo ""; \
		echo "GPU Information:"; \
		nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader 2>/dev/null || nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv || echo "  (nvidia-smi query failed)"; \
		echo ""; \
		echo "Full GPU status:"; \
		nvidia-smi 2>/dev/null | head -15 || echo "  (nvidia-smi failed)"; \
		echo ""; \
		echo "CUDA_VISIBLE_DEVICES: $${CUDA_VISIBLE_DEVICES:-not set}"; \
		echo "SLURM_GPUS_ON_NODE: $${SLURM_GPUS_ON_NODE:-not set}"; \
		echo "SLURM_GPUS: $${SLURM_GPUS:-not set}"; \
	else \
		echo "✗ nvidia-smi not found (GPU drivers may not be available)"; \
	fi; \
	echo ""; \
	echo "2. CONTAINER ENVIRONMENT CHECK:"; \
	echo "------------------------------"; \
	echo "Checking what's available in container..."; \
	apptainer exec --nv "$$CONTAINER" sh -c 'echo "PATH: $$PATH"; echo ""; echo "Python locations:"; ls -la /usr/bin/python* 2>/dev/null || echo "  (not found)"; echo ""; echo "Shell locations:"; ls -la /bin/sh /bin/bash /usr/bin/bash 2>/dev/null | head -3 || echo "  (not found)"'; \
	echo ""; \
	echo "3. CONTAINER GPU ACCESS CHECK:"; \
	echo "-----------------------------"; \
	export PYTHONNOUSERSITE=1; \
	export APPTAINER_USE_GPU=1; \
	echo "Running diagnostics inside container..."; \
	if apptainer exec --nv "$$CONTAINER" sh -c 'test -x /usr/bin/python3' 2>/dev/null; then \
		PYTHON_CMD="/usr/bin/python3"; \
	elif apptainer exec --nv "$$CONTAINER" sh -c 'test -x /usr/bin/python' 2>/dev/null; then \
		PYTHON_CMD="/usr/bin/python"; \
	else \
		echo "⚠ ERROR: Python not found in container. Container may be corrupted."; \
		echo "Try rebuilding: make build-container MODEL=$$MODEL_NAME"; \
		PYTHON_CMD="/usr/bin/python3"; \
	fi; \
	echo "Using Python: $$PYTHON_CMD"; \
	apptainer exec --env PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin --nv "$$CONTAINER" $$PYTHON_CMD -c "\
import sys; \
import torch; \
print('Python:', sys.version.split()[0]); \
print('PyTorch version:', torch.__version__); \
print(''); \
print('CUDA available:', torch.cuda.is_available()); \
print('CUDA device count:', torch.cuda.device_count()); \
if torch.cuda.is_available(): \
    print('CUDA version (PyTorch):', torch.version.cuda); \
    for i in range(torch.cuda.device_count()): \
        print(f'  Device {i}: {torch.cuda.get_device_name(i)}'); \
        print(f'    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB'); \
else: \
    print('⚠ WARNING: CUDA not available in container'); \
    print(''); \
    print('Troubleshooting:'); \
    print('  1. Verify you have GPU allocation: salloc --gres=gpu:1 ...'); \
    print('  2. Check CUDA_VISIBLE_DEVICES is set (should be visible above)'); \
    print('  3. Verify --nv flag is working (this script uses it)'); \
    print('  4. Check container CUDA version matches cluster drivers'); \
"; \
	echo ""; \
	echo "4. CONTAINER CUDA LIBRARIES:"; \
	echo "---------------------------"; \
	apptainer exec --env PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin --nv "$$CONTAINER" /bin/sh -c "\
if [ -d /usr/local/cuda ]; then \
    echo 'CUDA installation found: /usr/local/cuda'; \
    if [ -f /usr/local/cuda/version.txt ]; then \
        echo 'CUDA version (from container):'; \
        cat /usr/local/cuda/version.txt; \
    fi; \
    echo ''; \
    echo 'CUDA libraries:'; \
    ls -lh /usr/local/cuda/lib64/libcuda*.so* 2>/dev/null | head -3 || echo '  (library check skipped)'; \
else \
    echo '⚠ CUDA directory not found in container'; \
fi"; \
	echo ""; \
	echo ""; \
	echo "5. COMPATIBILITY CHECK:"; \
	echo "----------------------"; \
	if apptainer exec --nv "$$CONTAINER" sh -c 'test -x /usr/bin/python3' 2>/dev/null; then \
		PYTHON_CMD="/usr/bin/python3"; \
	else \
		PYTHON_CMD="/usr/bin/python"; \
	fi; \
	apptainer exec --env PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin --nv "$$CONTAINER" $$PYTHON_CMD -c "\
import torch; \
if torch.cuda.is_available() and torch.cuda.device_count() > 0: \
    print('✓ GPU is accessible inside container'); \
    print('✓ PyTorch can detect CUDA devices'); \
    print('✓ Container setup appears correct'); \
else: \
    print('✗ GPU is NOT accessible inside container'); \
    print(''); \
    print('Possible issues:'); \
    print('  - Container built with CUDA 11.7, but host driver may not support it'); \
    print('  - --nv flag not properly exposing GPU to container'); \
    print('  - CUDA_VISIBLE_DEVICES restriction'); \
"; \
	echo ""; \
	echo "========================================="

## Run embedding job on HPC (requires SLURM)
## Usage: make hpc-embed MODEL=<model_name> INPUT=<path/to/input.h5ad> OUTPUT=<path/to/output.npy> [MODEL_ARGS="--arg1 value1 --arg2"]
.PHONY: hpc-embed
hpc-embed:
	@if [ -z "$(MODEL)" ] || [ -z "$(INPUT)" ] || [ -z "$(OUTPUT)" ]; then \
		echo "Usage: make hpc-embed MODEL=<model_name> INPUT=<path/to/input.h5ad> OUTPUT=<path/to/output.npy> [MODEL_ARGS=\"--arg1 value1 --arg2\"]"; \
		echo ""; \
		echo "Example:"; \
		echo "  make hpc-embed MODEL=pca INPUT=data/test.h5ad OUTPUT=output/embeddings.npy MODEL_ARGS=\"--n-components 100\""; \
		exit 1; \
	fi
	@if [ ! -f "transcriptomic_fms/hpc/run_job.sh" ]; then \
		echo "Error: run_job.sh not found."; \
		exit 1; \
	fi
	@MODEL_NAME="$(MODEL)"; \
	CONTAINER="transcriptomic-fms-$$MODEL_NAME.sif"; \
	if [ ! -f "$$CONTAINER" ]; then \
		echo "Error: Container not found: $$CONTAINER"; \
		echo "Build it with: make build-container MODEL=$$MODEL_NAME"; \
		exit 1; \
	fi
	mkdir -p run_logs
	sbatch transcriptomic_fms/hpc/run_job.sh embed \
		--model $(MODEL) \
		--input $(INPUT) \
		--output $(OUTPUT) \
		$(MODEL_ARGS)

## Install model-specific dependencies locally
## Usage: make install-model MODEL=<model_name>
## Example: make install-model MODEL=scgpt
.PHONY: install-model
install-model:
	@if [ -z "$(MODEL)" ]; then \
		echo "Usage: make install-model MODEL=<model_name>"; \
		echo ""; \
		echo "Available models:"; \
		uv run python -m transcriptomic_fms.cli.main list; \
		exit 1; \
	fi
	@echo "Checking dependencies for $(MODEL)..."
	@DEP_GROUP=$$(uv run python -m transcriptomic_fms.models.get_dep_group $(MODEL) 2>/dev/null); \
	if [ -z "$$DEP_GROUP" ]; then \
		echo "Model $(MODEL) has no special dependencies."; \
	else \
		if [ "$$DEP_GROUP" = "scgpt" ]; then \
			echo ""; \
			echo "Note: Installing scgpt dependencies (includes flash-attn which requires CUDA/nvcc)."; \
			echo "On HPC clusters, ensure CUDA module is loaded:"; \
			echo "  module load cuda/11.7  # or appropriate CUDA version"; \
			echo "  module avail cuda     # to see available versions"; \
			echo ""; \
		fi; \
		echo "Installing dependencies for $(MODEL) (group: $$DEP_GROUP)..."; \
		uv sync --extra $$DEP_GROUP; \
	fi

## Format code
.PHONY: format
format:
	$(UV) run ruff format transcriptomic_fms
	$(UV) run ruff format pyproject.toml

## Lint code
.PHONY: lint
lint:
	$(UV) run ruff check transcriptomic_fms

## Run both format and lint
.PHONY: check
check: format lint
