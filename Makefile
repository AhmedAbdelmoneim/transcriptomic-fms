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
