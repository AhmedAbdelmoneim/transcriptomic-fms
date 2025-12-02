"""Main CLI entry point for embedding generation."""

import argparse
from pathlib import Path
import sys
from typing import Any

import scanpy as sc

from transcriptomic_fms.models import get_model, list_models
from transcriptomic_fms.utils.logging import get_logger, setup_logging

# Set up logging
logger = get_logger(__name__)


def _load_hvg_list(hvg_list_path: str) -> list[str]:
    """Load HVG list from file (one gene per line)."""
    path = Path(hvg_list_path)
    if not path.exists():
        raise FileNotFoundError(f"HVG list file not found: {hvg_list_path}")

    with open(path, "r") as f:
        genes = [line.strip() for line in f if line.strip()]

    return genes


def _parse_model_args(known_args: list[str], unknown_args: list[str]) -> dict[str, Any]:
    """
    Parse unknown arguments into model parameters.

    Handles:
    - --key=value -> {"key": value}
    - --key value -> {"key": value}
    - --flag -> {"flag": True}
    - --no-flag -> {"flag": False}

    Args:
        known_args: List of known argument names (to exclude)
        unknown_args: List of unknown arguments

    Returns:
        Dictionary of parsed model parameters
    """
    model_params: dict[str, Any] = {}
    i = 0

    while i < len(unknown_args):
        arg = unknown_args[i]

        # Skip known arguments
        if arg in known_args:
            i += 1
            continue

        # Handle --key=value format
        if arg.startswith("--") and "=" in arg:
            key, value = arg[2:].split("=", 1)
            key_normalized = key.replace("-", "_")

            # Special handling for hvg_list (load from file)
            if key_normalized == "hvg_list" or key_normalized == "hvg_list_path":
                model_params["hvg_list"] = _load_hvg_list(value)
            else:
                # Convert value to appropriate type
                model_params[key_normalized] = _convert_value(value)
            i += 1
        # Handle --key value format
        elif arg.startswith("--"):
            key = arg[2:].replace("-", "_")
            # Check if next arg is a value (not another flag)
            if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith("--"):
                value = unknown_args[i + 1]
                # Special handling for hvg_list (load from file)
                if key == "hvg_list" or key == "hvg_list_path":
                    model_params["hvg_list"] = _load_hvg_list(value)
                else:
                    model_params[key] = _convert_value(value)
                i += 2
            else:
                # Boolean flag (--flag means True)
                if key.startswith("no_"):
                    # --no-flag means False
                    model_params[key[3:]] = False
                else:
                    model_params[key] = True
                i += 1
        else:
            i += 1

    return model_params


def _convert_value(value: str) -> Any:
    """Convert string value to appropriate Python type."""
    # Try boolean
    if value.lower() in ("true", "yes", "1"):
        return True
    if value.lower() in ("false", "no", "0"):
        return False

    # Try int
    try:
        return int(value)
    except ValueError:
        pass

    # Try float
    try:
        return float(value)
    except ValueError:
        pass

    # Return as string
    return value


def embed_command(args: argparse.Namespace, model_args: dict[str, Any]) -> None:
    """Generate embeddings using a specified model."""
    # Validate inputs
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    output_path = Path(args.output)

    # Ensure output is .npy file
    if output_path.suffix != ".npy":
        logger.warning(
            f"Output path should be .npy file. Changing {output_path} to {output_path.with_suffix('.npy')}"
        )
        output_path = output_path.with_suffix(".npy")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model with model-specific arguments
    try:
        logger.info(f"Loading model: {args.model}")
        model = get_model(args.model, **model_args)
    except ImportError as e:
        # Check if this is a missing dependency error
        dep_group = None
        try:
            # Try to get model class to check dependencies
            from transcriptomic_fms.models.registry import _MODEL_REGISTRY, _ensure_models_loaded

            _ensure_models_loaded()
            if args.model in _MODEL_REGISTRY:
                temp_model = _MODEL_REGISTRY[args.model](model_name=args.model)
                dep_group = temp_model.get_optional_dependency_group()
        except Exception:
            pass

        logger.error(f"Failed to import model: {e}")
        if dep_group:
            logger.error("")
            logger.error(f"To install dependencies for {args.model}, run:")
            logger.error(f"  make install-model MODEL={args.model}")
            logger.error(f"  or: uv sync --extra {dep_group}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Model error: {e}")
        sys.exit(1)

    # Load AnnData
    logger.info(f"Loading AnnData from {input_path}...")
    adata = sc.read_h5ad(input_path)
    logger.info(f"Loaded: {adata.shape}")

    # Preprocess
    logger.info("Preprocessing data...")
    adata = model.preprocess(adata)

    # Generate embeddings (pass batch_size if provided)
    embed_kwargs = {}
    if hasattr(args, "batch_size") and args.batch_size is not None:
        embed_kwargs["batch_size"] = args.batch_size

    # Also pass any model args that might be for embed() method
    embed_kwargs.update(model_args)

    logger.info(f"Generating embeddings with {args.model}...")
    embeddings = model.embed(adata, output_path, **embed_kwargs)

    # Validate
    logger.info("Validating embeddings...")
    model.validate_embeddings(embeddings, adata.n_obs)

    # Save
    logger.info(f"Saving embeddings to {output_path}...")
    import numpy as np

    np.save(output_path, embeddings)
    logger.info(f"Done! Embeddings shape: {embeddings.shape}")


def list_command(args: argparse.Namespace) -> None:
    """List all available models."""
    models = list_models()
    if not models:
        logger.info("No models registered.")
        return

    # Check pyproject.toml for optional dependencies (without requiring them to be installed)
    optional_deps = {}
    try:
        from pathlib import Path
        import tomllib

        project_root = Path(__file__).parent.parent.parent
        pyproject_path = project_root / "pyproject.toml"
        if pyproject_path.exists():
            with open(pyproject_path, "rb") as f:
                config = tomllib.load(f)
            optional_deps = config.get("project", {}).get("optional-dependencies", {})
    except Exception:
        pass

    logger.info("Available models:")
    for model_name in models:
        # Check if model has optional dependencies from pyproject.toml
        # This doesn't require the dependencies to be installed
        if model_name in optional_deps:
            logger.info(f"  - {model_name} (requires optional dependencies: {model_name})")
        else:
            # Try to get from model instance if available (but don't fail if not)
            try:
                model = get_model(model_name)
                deps = model.get_optional_dependency_group()
                if deps:
                    logger.info(f"  - {model_name} (requires: {deps})")
                else:
                    logger.info(f"  - {model_name}")
            except Exception:
                # Model can't be instantiated (missing deps), but we know it exists
                if model_name in optional_deps:
                    logger.info(f"  - {model_name} (requires optional dependencies: {model_name})")
                else:
                    logger.info(f"  - {model_name}")


def install_model_command(args: argparse.Namespace) -> None:
    """Install dependencies for a specific model."""
    if not args.model:
        print("Error: --model is required", file=sys.stderr)
        sys.exit(1)

    try:
        model = get_model(args.model)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    dep_group = model.get_optional_dependency_group()
    if not dep_group:
        print(f"Model {args.model} has no special dependencies to install.")
        return

    print(f"Installing dependencies for {args.model}...")
    print(f"Optional dependency group: {dep_group}")
    print("")
    print("To install, run:")
    print(f"  uv sync --extra {dep_group}")
    print(f"  or: pip install {' '.join(model.get_required_dependencies())}")
    print("")
    print("Note: This command only shows installation instructions.")
    print("Run the command above to actually install dependencies.")


def main() -> None:
    """Main CLI entry point."""
    # Set up logging
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Transcriptomic Foundation Models - Embedding Generation",
        allow_abbrev=False,  # Disable abbreviation to avoid conflicts
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Embed command
    embed_parser = subparsers.add_parser(
        "embed",
        help="Generate embeddings",
        allow_abbrev=False,
    )
    embed_parser.add_argument("--model", required=True, help="Model name to use")
    embed_parser.add_argument(
        "--input", required=True, type=Path, help="Input AnnData file (.h5ad)"
    )
    embed_parser.add_argument(
        "--output", required=True, type=Path, help="Output embeddings file (.npy)"
    )
    embed_parser.add_argument("--batch-size", type=int, help="Batch size for processing")
    embed_parser.set_defaults(func=embed_command)

    # List command
    list_parser = subparsers.add_parser("list", help="List available models")
    list_parser.set_defaults(func=list_command)

    # Install model dependencies command
    install_parser = subparsers.add_parser(
        "install-model", help="Install dependencies for a model"
    )
    install_parser.add_argument(
        "--model", required=True, help="Model name to install dependencies for"
    )
    install_parser.set_defaults(func=install_model_command)

    # Parse known arguments and allow unknown ones for model-specific params
    args, unknown = parser.parse_known_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # For embed command, parse model-specific arguments
    if args.command == "embed":
        known_arg_names = ["--model", "--input", "--output", "--batch-size"]
        model_args = _parse_model_args(known_arg_names, unknown)
        args.func(args, model_args)
    else:
        args.func(args)


if __name__ == "__main__":
    main()
