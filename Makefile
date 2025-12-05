#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = transcriptomic-fms
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync

## Generate embeddings using a model
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

## List available models
.PHONY: list-models
list-models:
	@python -m transcriptomic_fms.cli.main list

## Install dependencies for a specific model
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

## Set up HPC environment (build base Apptainer container)
.PHONY: setup-hpc
setup-hpc:
	@if [ ! -f "transcriptomic_fms/hpc/setup_hpc.sh" ]; then \
		echo "Error: setup_hpc.sh not found"; \
		exit 1; \
	fi
	bash transcriptomic_fms/hpc/setup_hpc.sh

## Build model-specific container
## Usage: make build-model-container MODEL=<model_name>
## Example: make build-model-container MODEL=scgpt
.PHONY: build-model-container
build-model-container:
	@if [ -z "$(MODEL)" ]; then \
		echo "Usage: make build-model-container MODEL=<model_name>"; \
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
	BASE_CONTAINER="transcriptomic-fms.sif"; \
	if [ ! -f "$$CONTAINER_DEF" ]; then \
		echo "Error: Container definition not found: $$CONTAINER_DEF"; \
		exit 1; \
	fi; \
	# Check if model container extends base container and verify base exists \
	if grep -q "Bootstrap: localimage" "$$CONTAINER_DEF" 2>/dev/null; then \
		if [ ! -f "$$BASE_CONTAINER" ]; then \
			echo "Error: Base container ($$BASE_CONTAINER) not found."; \
			echo "Model container extends base container. Please build base container first:"; \
			echo "  make setup-hpc"; \
			exit 1; \
		fi; \
		echo "Model container extends base container: $$BASE_CONTAINER"; \
	fi; \
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
	@if [ ! -f "run_apptainer.sh" ]; then \
		echo "Error: run_apptainer.sh not found. Run 'make setup-hpc' first."; \
		exit 1; \
	fi
	@export APPTAINER_USE_GPU=1; \
	./run_apptainer.sh python -m transcriptomic_fms.cli.main embed \
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
		echo "Error: run_job.sh not found. Run 'make setup-hpc' first."; \
		exit 1; \
	fi
	mkdir -p run_logs
	sbatch transcriptomic_fms/hpc/run_job.sh embed \
		--model $(MODEL) \
		--input $(INPUT) \
		--output $(OUTPUT) \
		$(MODEL_ARGS)



## Run tests locally
## Usage: make test [MODEL=<model_name>]
## Example: make test MODEL=pca
.PHONY: test
test:
	@if [ -z "$(MODEL)" ]; then \
		echo "Running all model tests..."; \
		uv run pytest tests/ -v; \
	else \
		echo "Running tests for model: $(MODEL)"; \
		uv run pytest tests/test_$(MODEL).py -v; \
	fi

## Run tests in HPC/container environment
## Usage: make test-hpc [MODEL=<model_name>]
## Example: make test-hpc MODEL=pca
## Note: Requires container to be built (make setup-hpc)
.PHONY: test-hpc
test-hpc:
	@if [ ! -f "transcriptomic-fms.sif" ] && [ -z "$$APPTAINER_IMAGE" ]; then \
		echo "Error: Container image not found. Build it first:"; \
		echo "  make setup-hpc"; \
		exit 1; \
	fi
	@if [ -z "$$APPTAINER_IMAGE" ]; then \
		export APPTAINER_IMAGE=./transcriptomic-fms.sif; \
	fi
	@if [ -z "$(MODEL)" ]; then \
		echo "Running all HPC model tests..."; \
		uv run pytest tests/ -v -k "test_model_smoke_hpc"; \
	else \
		echo "Running HPC tests for model: $(MODEL)"; \
		uv run pytest tests/test_$(MODEL).py -v -k "test_model_smoke_hpc"; \
	fi

## Install test dependencies
.PHONY: install-test
install-test:
	uv sync --extra dev

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format





## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
