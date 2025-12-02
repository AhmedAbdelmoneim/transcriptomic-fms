"""Main CLI entry point for embedding generation."""

import argparse
import sys
from pathlib import Path
from typing import Any

import scanpy as sc

from transcriptomic_fms.models import get_model, list_models


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
            # Convert value to appropriate type
            model_params[key.replace("-", "_")] = _convert_value(value)
            i += 1
        # Handle --key value format
        elif arg.startswith("--"):
            key = arg[2:].replace("-", "_")
            # Check if next arg is a value (not another flag)
            if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith("--"):
                value = unknown_args[i + 1]
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
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model with model-specific arguments
    try:
        model = get_model(args.model, **model_args)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Load AnnData
    print(f"Loading AnnData from {input_path}...")
    adata = sc.read_h5ad(input_path)
    print(f"Loaded: {adata.shape}")

    # Preprocess
    print("Preprocessing data...")
    adata = model.preprocess(adata)

    # Generate embeddings (pass batch_size if provided)
    embed_kwargs = {}
    if hasattr(args, "batch_size") and args.batch_size is not None:
        embed_kwargs["batch_size"] = args.batch_size
    
    # Also pass any model args that might be for embed() method
    embed_kwargs.update(model_args)

    print(f"Generating embeddings with {args.model}...")
    embeddings = model.embed(adata, output_path, **embed_kwargs)

    # Validate
    model.validate_embeddings(embeddings, adata.n_obs)

    # Save
    print(f"Saving embeddings to {output_path}...")
    import numpy as np

    np.save(output_path, embeddings)
    print(f"Done! Embeddings shape: {embeddings.shape}")


def list_command(args: argparse.Namespace) -> None:
    """List all available models."""
    models = list_models()
    if not models:
        print("No models registered.")
        return

    print("Available models:")
    for model_name in models:
        print(f"  - {model_name}")


def main() -> None:
    """Main CLI entry point."""
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
    embed_parser.add_argument(
        "--model", required=True, help="Model name to use"
    )
    embed_parser.add_argument(
        "--input", required=True, type=Path, help="Input AnnData file (.h5ad)"
    )
    embed_parser.add_argument(
        "--output", required=True, type=Path, help="Output embeddings file (.npy)"
    )
    embed_parser.add_argument(
        "--batch-size", type=int, help="Batch size for processing"
    )
    embed_parser.set_defaults(func=embed_command)

    # List command
    list_parser = subparsers.add_parser("list", help="List available models")
    list_parser.set_defaults(func=list_command)

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

