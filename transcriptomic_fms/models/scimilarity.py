"""SCimilarity embedding model."""

from pathlib import Path
from typing import Any, Optional

import numpy as np
import scanpy as sc

from transcriptomic_fms.models.base import BaseEmbeddingModel
from transcriptomic_fms.models.registry import register_model

try:
    from scimilarity import CellEmbedding
    from scimilarity.utils import (
        align_dataset,
        consolidate_duplicate_symbols,
        lognorm_counts,
    )
except ImportError:
    CellEmbedding = None
    align_dataset = None
    consolidate_duplicate_symbols = None
    lognorm_counts = None

# Default SCimilarity model Zenodo URL
SCIMILARITY_MODEL_URL = "https://zenodo.org/records/10685499/files/model_v1.1.tar.gz"
SCIMILARITY_MODEL_MD5 = "546251b7c435f3b1dbe38e2e420ad57f"


@register_model("scimilarity")
class SCimilarityModel(BaseEmbeddingModel):
    """SCimilarity embedding model."""

    def __init__(
        self,
        model_name: str,
        model_path: Optional[str] = None,
        model_dir: Optional[str] = None,
        batch_size: int = 64,
        device: Optional[str] = None,
        use_gpu: Optional[bool] = None,
        auto_download: bool = True,
        download_dir: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initialize SCimilarity model.

        Args:
            model_name: Model identifier
            model_path: Path to SCimilarity model directory (must contain model files).
                       If None and auto_download=True, will download to download_dir.
                       This can also be a path to the extracted model directory.
            model_dir: Alias for model_path (for consistency with other models)
            batch_size: Batch size for embedding generation (not used by SCimilarity but kept for API consistency)
            device: Device to use ('cuda' or 'cpu'). Auto-detects if None.
            use_gpu: Whether to use GPU. If None, auto-detects based on device and CUDA availability.
            auto_download: If True, automatically download model if not found
            download_dir: Directory to download model to if auto_download=True and model_path is None.
                          Defaults to {project_root}/models/scimilarity
            **kwargs: Additional arguments
        """
        super().__init__(model_name=model_name, requires_gpu=False, **kwargs)

        if CellEmbedding is None:
            raise ImportError(
                "SCimilarity is not installed. Install with:\n"
                "  make install-model MODEL=scimilarity\n"
                "  or: uv sync --extra scimilarity\n"
                "  or: pip install scimilarity"
            )

        self.batch_size = batch_size
        self.auto_download = auto_download

        # Handle model_path vs model_dir (support both for consistency)
        if model_path:
            self.model_path = Path(model_path)
        elif model_dir:
            self.model_path = Path(model_dir)
        elif auto_download:
            # Use default download location
            if download_dir:
                self.model_path = Path(download_dir)
            else:
                # Default to root/models/scimilarity
                import transcriptomic_fms

                package_path = Path(transcriptomic_fms.__file__).parent
                project_root = package_path.parent
                default_model_dir = project_root / "models" / "scimilarity"
                self.model_path = default_model_dir
        else:
            self.model_path = None

        # Auto-detect device and GPU usage
        if device is None:
            try:
                import torch

                cuda_available = torch.cuda.is_available() and torch.cuda.device_count() > 0
                self.device = "cuda" if cuda_available else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        # Determine GPU usage
        if use_gpu is None:
            # Auto-detect: use GPU if device is cuda
            self.use_gpu = self.device == "cuda"
        else:
            self.use_gpu = use_gpu

        # Check if model exists, download if needed
        if self.model_path:
            # Create directory if it doesn't exist (for downloads)
            if not self.model_path.exists():
                if auto_download:
                    self.model_path.mkdir(parents=True, exist_ok=True)
                else:
                    raise ValueError(
                        f"Model directory does not exist: {self.model_path}. "
                        f"Set auto_download=True to download automatically."
                    )

            # Check if model files exist
            if not self._model_exists():
                if auto_download:
                    print(f"Model not found at {self.model_path}. Downloading...")
                    self._download_model()
                else:
                    raise ValueError(
                        f"Model files not found in {self.model_path}. "
                        f"Set auto_download=True to download automatically."
                    )
            else:
                import sys

                print(f"Using SCimilarity model from: {self.model_path}", file=sys.stderr)

        # Initialize model (will be done lazily in embed() if needed)
        self._model: Optional[CellEmbedding] = None
        # Store the actual model directory (may be in a subdirectory like model_v1.1)
        self._actual_model_path: Optional[Path] = None

    def _find_actual_model_path(self) -> Path:
        """
        Find the actual model directory.

        The model may be in a subdirectory (e.g., model_v1.1) after extraction.
        This method searches for the actual model directory containing gene_order.tsv.

        Returns:
            Path to the actual model directory

        Raises:
            ValueError: If model directory cannot be found
        """
        if self.model_path is None:
            raise ValueError(
                "model_path must be provided. Set it in model initialization, "
                "via --model-path argument, or enable auto_download."
            )

        # Check if model_path itself contains gene_order.tsv
        if (self.model_path / "gene_order.tsv").exists():
            return self.model_path

        # Look for subdirectories that might contain the model
        # Common patterns: model_v1.1, model, v1.1, etc.
        for item in self.model_path.iterdir():
            if item.is_dir():
                # Check if this subdirectory contains gene_order.tsv
                if (item / "gene_order.tsv").exists():
                    return item

        # If not found, raise an error
        raise ValueError(
            f"Model files not found in {self.model_path} or its subdirectories. "
            f"Expected to find gene_order.tsv file. "
            f"Please ensure the model is extracted correctly."
        )

    def _get_model(self) -> CellEmbedding:
        """Get or initialize the CellEmbedding model."""
        if self._model is None:
            if self._actual_model_path is None:
                self._actual_model_path = self._find_actual_model_path()
            self._model = CellEmbedding(
                model_path=str(self._actual_model_path), use_gpu=self.use_gpu
            )
        return self._model

    def preprocess(self, adata: sc.AnnData, output_path: Optional[Path] = None) -> sc.AnnData:
        """
        Preprocess data for SCimilarity: align genes and log-normalize.

        Requirements:
        - Gene symbols must be available either as:
          * var.index (gene symbols as index)
          * var['gene_symbols'] column
          * var['feature_name'] column (common in some formats)
          * var['gene_name'] column (fallback)
        - Data should be raw counts (not log-normalized)

        Args:
            adata: Input AnnData object
            output_path: Optional path to save preprocessed data

        Returns:
            Preprocessed AnnData object
        """
        adata = adata.copy()

        # Get the actual model path first (to access gene_order)
        # We need to find the actual model directory before initializing the model
        if self._actual_model_path is None:
            self._actual_model_path = self._find_actual_model_path()

        # Get the model to access gene_order
        model = self._get_model()

        # Determine gene symbols
        gene_symbols = self._get_gene_symbols(adata)

        # Store original gene names if needed
        if "feature_name" not in adata.var.columns:
            adata.var["feature_name"] = gene_symbols

        # Log-normalize counts (SCimilarity expects log-normalized data)
        # Store raw counts in layers if not already there
        if "counts" not in adata.layers:
            adata.layers["counts"] = adata.X.copy()

        # Set var_names to uppercase gene symbols (SCimilarity expects uppercase)
        adata.var_names = [g.upper() for g in gene_symbols]

        # Consolidate duplicate symbols
        adata = consolidate_duplicate_symbols(adata)

        # Align dataset to model's gene order
        adata = align_dataset(adata, model.gene_order)

        # Log-normalize
        adata = lognorm_counts(adata)

        if output_path:
            adata.write(output_path)

        return adata

    def _get_gene_symbols(self, adata: sc.AnnData) -> list[str]:
        """
        Extract gene symbols from AnnData.

        Checks in order:
        1. var['gene_symbols'] column
        2. var['feature_name'] column
        3. var['gene_name'] column
        4. var.index (assumes index contains gene symbols)

        Returns:
            List of gene symbols

        Raises:
            ValueError: If gene symbols cannot be determined
        """
        # Check for gene_symbols column
        if "gene_symbols" in adata.var.columns:
            return adata.var["gene_symbols"].tolist()

        # Check for feature_name column (common in some formats)
        if "feature_name" in adata.var.columns:
            return adata.var["feature_name"].tolist()

        # Check for gene_name column (common alternative)
        if "gene_name" in adata.var.columns:
            return adata.var["gene_name"].tolist()

        # Use index as gene symbols
        if adata.var_names is not None and len(adata.var_names) > 0:
            return adata.var_names.tolist()

        raise ValueError(
            "Cannot determine gene symbols. "
            "Please ensure one of the following:\n"
            "  - var.index contains gene symbols\n"
            "  - var['gene_symbols'] column exists\n"
            "  - var['feature_name'] column exists\n"
            "  - var['gene_name'] column exists"
        )

    def _model_exists(self) -> bool:
        """Check if model files exist in model directory."""
        if not self.model_path or not self.model_path.exists():
            return False

        # Check if model_path itself contains gene_order.tsv (the key file)
        if (self.model_path / "gene_order.tsv").exists():
            return True

        # Look for subdirectories that contain the model
        # The model is typically in a subdirectory like model_v1.1 after extraction
        for item in self.model_path.iterdir():
            if item.is_dir() and (item / "gene_order.tsv").exists():
                return True

        return False

    def _download_model(self) -> None:
        """Download SCimilarity model from Zenodo."""
        try:
            import requests
            import tarfile
            import hashlib
        except ImportError:
            raise ImportError(
                "requests is required for automatic model download. Install with: pip install requests"
            )

        # Create model directory if it doesn't exist
        self.model_path.mkdir(parents=True, exist_ok=True)

        print(f"Downloading SCimilarity model to {self.model_path}...")
        print(f"Source: {SCIMILARITY_MODEL_URL}")
        print("Note: This is a large file (~30GB). Download may take a while.")

        try:
            # Download the model
            response = requests.get(SCIMILARITY_MODEL_URL, stream=True)
            response.raise_for_status()

            # Save to temporary file first
            tar_path = self.model_path / "model_v1.1.tar.gz"
            total_size = int(response.headers.get("content-length", 0))

            with open(tar_path, "wb") as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\rDownloaded: {percent:.1f}%", end="", flush=True)

            print()  # New line after progress

            # Verify MD5 checksum
            print("Verifying checksum...")
            with open(tar_path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()

            if file_hash != SCIMILARITY_MODEL_MD5:
                tar_path.unlink()  # Remove corrupted file
                raise RuntimeError(
                    f"MD5 checksum mismatch. Expected {SCIMILARITY_MODEL_MD5}, got {file_hash}. "
                    f"Download may be corrupted. Please try again."
                )

            print("Checksum verified.")

            # Extract the tarball
            print("Extracting model files...")
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=self.model_path)

            # Remove the tarball to save space
            tar_path.unlink()

            # Verify extraction
            if not self._model_exists():
                raise RuntimeError(
                    "Model extraction completed but model files are missing. "
                    f"Expected to find gene_order.tsv in {self.model_path} or its subdirectories."
                )

            # Find and report the actual model path
            try:
                actual_path = self._find_actual_model_path()
                print(f"Model downloaded and extracted successfully.")
                print(f"Model directory: {self.model_path}")
                print(f"Actual model path: {actual_path}")
            except ValueError:
                # This shouldn't happen if _model_exists() returned True, but handle gracefully
                print(f"Model downloaded and extracted to {self.model_path}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to download SCimilarity model: {e}\n"
                f"You can manually download from: {SCIMILARITY_MODEL_URL}\n"
                f"Extract to: {self.model_path}\n"
                f"Or set model_path to point to an existing model directory."
            ) from e

    def embed(
        self,
        adata: sc.AnnData,
        output_path: Path,
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Generate SCimilarity embeddings.

        Args:
            adata: Preprocessed AnnData object
            output_path: Path where embeddings will be saved
            batch_size: Not used by SCimilarity but kept for API consistency
            **kwargs: Additional arguments (ignored)

        Returns:
            Embeddings array of shape (n_cells, n_dimensions)
        """
        model = self._get_model()

        # Generate embeddings using SCimilarity's get_embeddings method
        # This expects log-normalized data in adata.X
        embeddings = model.get_embeddings(adata.X)

        # Ensure embeddings are numpy array
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)

        # Validate embeddings before returning
        self.validate_embeddings(embeddings, adata.n_obs)

        return embeddings

    def validate_embeddings(self, embeddings: np.ndarray, n_cells: int) -> None:
        """
        Validate that embeddings have correct shape and properties.

        Overrides base class to add SCimilarity-specific validation.
        """
        # Call base class validation
        super().validate_embeddings(embeddings, n_cells)

        # Additional SCimilarity-specific validation could go here if needed
        # For now, base validation is sufficient

    def get_container_command(
        self,
        adata_path: Path,
        output_path: Path,
        **kwargs: Any,
    ) -> list[str]:
        """
        Get command to run this model in a container.

        Overrides base class to include SCimilarity-specific arguments.
        """
        cmd = [
            "python",
            "-m",
            "transcriptomic_fms.cli.main",
            "embed",
            "--model",
            self.model_name,
            "--input",
            str(adata_path),
            "--output",
            str(output_path),
        ]

        # Add SCimilarity-specific arguments
        if self.model_path:
            cmd.extend(["--model-path", str(self.model_path)])
        if self.device:
            cmd.extend(["--device", self.device])
        if self.use_gpu is not None:
            cmd.extend(["--use-gpu", str(self.use_gpu).lower()])

        # Add any additional kwargs
        for k, v in kwargs.items():
            if v is not None:
                cmd.extend([f"--{k.replace('_', '-')}", str(v)])

        return cmd

    def get_required_dependencies(self) -> list[str]:
        """Get required dependencies for SCimilarity."""
        return ["scimilarity", "requests"]

    def get_optional_dependency_group(self) -> Optional[str]:
        """Get optional dependency group name for SCimilarity."""
        return "scimilarity"

    def get_container_name(self) -> Optional[str]:
        """Get container name for SCimilarity."""
        return "scimilarity"
