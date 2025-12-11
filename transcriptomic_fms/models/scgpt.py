"""scGPT embedding model."""

from pathlib import Path
from typing import Any, Optional

import numpy as np
import scanpy as sc
import torch

from transcriptomic_fms.models.base import BaseEmbeddingModel
from transcriptomic_fms.models.registry import register_model

try:
    import scgpt as scg
    from scgpt.preprocess import Preprocessor
    from scgpt.tasks import embed_data
except ImportError:
    scg = None
    Preprocessor = None
    embed_data = None

try:
    import gdown
except ImportError:
    gdown = None

# Default scGPT model Google Drive folder
SCGPT_MODEL_DRIVE_ID = "1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y"
SCGPT_MODEL_URL = f"https://drive.google.com/drive/folders/{SCGPT_MODEL_DRIVE_ID}"


@register_model("scgpt")
class SCGPTModel(BaseEmbeddingModel):
    """scGPT embedding model."""

    def __init__(
        self,
        model_name: str,
        model_dir: Optional[str] = None,
        n_hvg: Optional[int] = 1200,
        hvg_list: Optional[list[str]] = None,
        n_bins: int = 51,
        batch_size: int = 64,
        device: Optional[str] = None,
        auto_download: bool = True,
        download_dir: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initialize scGPT model.

        Args:
            model_name: Model identifier
            model_dir: Path to scGPT model directory (must contain best_model.pt, args.json, vocab.json).
                       If None and auto_download=True, will download to download_dir.
            n_hvg: Number of highly variable genes to select. Ignored if hvg_list is provided.
                   If None and hvg_list is None, uses all genes.
            hvg_list: Pre-computed list of highly variable gene names to use. If provided, n_hvg is ignored.
            n_bins: Number of bins for value binning
            batch_size: Batch size for embedding generation
            device: Device to use ('cuda' or 'cpu'). Auto-detects if None.
            auto_download: If True, automatically download model if not found
            download_dir: Directory to download model to if auto_download=True and model_dir is None.
                          Defaults to {project_root}/models/scGPT_human
            **kwargs: Additional arguments
        """
        super().__init__(model_name=model_name, requires_gpu=True, **kwargs)

        if scg is None:
            raise ImportError(
                "scGPT is not installed. Install with:\n"
                "  make install-model MODEL=scgpt\n"
                "  or: uv sync --extra scgpt\n"
                "  or: pip install scgpt 'torch<=2.2.2' 'numpy<2' gdown"
            )

        self.n_hvg = n_hvg
        self.hvg_list = hvg_list
        self.n_bins = n_bins
        self.batch_size = batch_size
        self.auto_download = auto_download

        # Auto-detect device
        # Check both that CUDA is available AND that devices are accessible
        # (device_count() can be 0 even if CUDA libraries are installed)
        cuda_available = torch.cuda.is_available() and torch.cuda.device_count() > 0

        if device is None:
            self.device = "cuda" if cuda_available else "cpu"
        else:
            # User specified a device - validate it
            if device == "cuda" and not cuda_available:
                import warnings

                warnings.warn(
                    "CUDA requested but no CUDA devices available. "
                    f"torch.cuda.is_available()={torch.cuda.is_available()}, "
                    f"torch.cuda.device_count()={torch.cuda.device_count()}. "
                    "Falling back to CPU.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.device = "cpu"
            else:
                self.device = device

        # Determine model directory
        if model_dir:
            self.model_dir = Path(model_dir)
        elif auto_download:
            # Use default download location
            if download_dir:
                self.model_dir = Path(download_dir)
            else:
                # Default to root/models/{model_name}
                # Try to find project root (parent of transcriptomic_fms package)
                import transcriptomic_fms

                package_path = Path(transcriptomic_fms.__file__).parent
                project_root = package_path.parent
                default_model_dir = project_root / "models" / "scGPT_human"
                self.model_dir = default_model_dir
        else:
            self.model_dir = None

        # Check if model exists, download if needed
        if self.model_dir:
            # Create directory if it doesn't exist (for downloads)
            if not self.model_dir.exists():
                if auto_download:
                    self.model_dir.mkdir(parents=True, exist_ok=True)
                else:
                    raise ValueError(
                        f"Model directory does not exist: {self.model_dir}. "
                        f"Set auto_download=True to download automatically."
                    )

            # Check if model files exist
            if not self._model_exists():
                if auto_download:
                    print(f"Model not found at {self.model_dir}. Downloading...")
                    self._download_model()
                else:
                    raise ValueError(
                        f"Model files not found in {self.model_dir}. "
                        f"Required files: best_model.pt, args.json, vocab.json. "
                        f"Set auto_download=True to download automatically."
                    )
            else:
                import sys

                print(f"Using scGPT model from: {self.model_dir}", file=sys.stderr)

    def preprocess(self, adata: sc.AnnData, output_path: Optional[Path] = None) -> sc.AnnData:
        """
        Preprocess data for scGPT: select HVGs and bin values.

        Requirements:
        - Gene symbols must be available either as:
          * var.index (gene symbols as index)
          * var['gene_symbols'] column
          * var['gene_name'] column (fallback)

        Args:
            adata: Input AnnData object
            output_path: Optional path to save preprocessed data

        Returns:
            Preprocessed AnnData object

        Raises:
            ValueError: If gene symbols cannot be determined
        """
        adata = adata.copy()

        # Determine gene symbols
        gene_symbols = self._get_gene_symbols(adata)
        adata.var["gene_symbols"] = gene_symbols

        # Handle HVG selection
        if self.hvg_list is not None:
            # Use pre-provided HVG list
            # Check which genes are available
            available_hvgs = [g for g in self.hvg_list if g in adata.var_names]
            if len(available_hvgs) == 0:
                raise ValueError(
                    f"None of the provided HVG genes are found in adata.var_names. "
                    f"First few provided: {self.hvg_list[:5]}"
                )
            if len(available_hvgs) < len(self.hvg_list):
                print(
                    f"Warning: Only {len(available_hvgs)}/{len(self.hvg_list)} "
                    f"provided HVGs found in data. Using available subset."
                )
            adata = adata[:, available_hvgs].copy()
        elif self.n_hvg is not None and self.n_hvg > 0:
            # Select HVGs using scanpy
            sc.pp.highly_variable_genes(
                adata, n_top_genes=self.n_hvg, flavor="seurat_v3", inplace=True
            )
            adata = adata[:, adata.var.highly_variable].copy()
        # else: use all genes (no filtering)

        # Ensure gene_symbols column exists after filtering
        if "gene_symbols" not in adata.var.columns:
            adata.var["gene_symbols"] = adata.var.index

        # Filter out cells with all-zero expression (empty rows)
        # This prevents errors in binning when encountering zero-size arrays
        import numpy as np

        cell_sums = np.array(adata.X.sum(axis=1)).flatten()
        non_zero_cells = cell_sums > 0
        if not np.all(non_zero_cells):
            n_filtered = np.sum(~non_zero_cells)
            print(
                f"Filtering out {n_filtered} cells with all-zero expression "
                f"(out of {adata.n_obs} total cells)"
            )
            adata = adata[non_zero_cells, :].copy()

        # Create preprocessor and bin data
        preprocessor = Preprocessor(
            use_key="X",  # Use raw data from .X
            filter_gene_by_counts=False,
            filter_cell_by_counts=False,
            normalize_total=False,  # Already normalized if needed
            log1p=False,  # Already log-transformed if needed
            subset_hvg=False,  # Already subset to HVGs if needed
            binning=self.n_bins,
            result_binned_key="X_binned",
        )

        # Apply preprocessing (bins the data)
        preprocessor(adata)

        if output_path:
            adata.write(output_path)

        return adata

    def _model_exists(self) -> bool:
        """Check if model files exist in model directory."""
        if not self.model_dir or not self.model_dir.exists():
            return False

        required_files = ["best_model.pt", "args.json", "vocab.json"]
        return all((self.model_dir / f).exists() for f in required_files)

    def _download_model(self) -> None:
        """Download scGPT model from Google Drive."""
        if gdown is None:
            raise ImportError(
                "gdown is required for automatic model download. Install with: pip install gdown"
            )

        # Create model directory if it doesn't exist
        self.model_dir.mkdir(parents=True, exist_ok=True)

        print(f"Downloading scGPT model to {self.model_dir}...")
        print(f"Source: {SCGPT_MODEL_URL}")

        try:
            # Set SSL certificate path for gdown (Ubuntu/Debian locations)
            import os
            import ssl

            cert_paths = [
                "/etc/ssl/certs/ca-certificates.crt",
                "/etc/ssl/certs/ca-bundle.crt",
                "/usr/lib/ssl/certs/ca-certificates.crt",
            ]
            for cert_path in cert_paths:
                if os.path.exists(cert_path):
                    os.environ.setdefault("REQUESTS_CA_BUNDLE", cert_path)
                    os.environ.setdefault("SSL_CERT_FILE", cert_path)
                    break

            gdown.download_folder(
                SCGPT_MODEL_URL,
                output=str(self.model_dir),
                quiet=False,
            )

            # Verify download
            if not self._model_exists():
                raise RuntimeError(
                    "Model download completed but required files are missing. "
                    f"Expected files in {self.model_dir}: best_model.pt, args.json, vocab.json"
                )

            print(f"Model downloaded successfully to {self.model_dir}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to download scGPT model: {e}\n"
                f"You can manually download from: {SCGPT_MODEL_URL}\n"
                f"Or set model_dir to point to an existing model directory."
            ) from e

    def embed(
        self,
        adata: sc.AnnData,
        output_path: Path,
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> sc.AnnData:
        """
        Generate scGPT embeddings.

        Uses the embed_data function from scgpt.tasks which handles:
        - Loading the model
        - Tokenizing genes
        - Generating embeddings
        """
        if self.model_dir is None:
            raise ValueError(
                "model_dir must be provided. Set it in model initialization, "
                "via --model-dir argument, or enable auto_download."
            )

        # Ensure gene_symbols column exists (should already be set by preprocess)
        if "gene_symbols" not in adata.var.columns:
            gene_symbols = self._get_gene_symbols(adata)
            adata.var["gene_symbols"] = gene_symbols

        # Use provided batch_size or default
        effective_batch_size = batch_size or self.batch_size

        # Generate embeddings using scGPT's embed_data function
        # This function handles model loading, tokenization, and embedding generation
        embedded_adata = embed_data(
            adata,
            str(self.model_dir),
            gene_col="gene_symbols",
            batch_size=effective_batch_size,
            device=self.device,
        )

        # Extract embeddings from obsm
        if "X_scGPT" in embedded_adata.obsm:
            embeddings = embedded_adata.obsm["X_scGPT"]
        else:
            raise ValueError(
                "Embeddings not found in adata.obsm['X_scGPT']. "
                "Check that embed_data completed successfully."
            )

        # Ensure embeddings are numpy array
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        # Create barebones AnnData with embeddings in X and obs preserved
        # Use obs from embedded_adata if available (may have filtered cells), otherwise use original
        result_obs = embedded_adata.obs.copy() if embedded_adata.n_obs == embeddings.shape[0] else adata.obs.copy()
        result_adata = sc.AnnData(
            X=embeddings,
            obs=result_obs,
        )
        # No var needed for embeddings

        # Validate embeddings
        self.validate_embeddings(result_adata)

        return result_adata

    def validate_embeddings(self, adata: sc.AnnData) -> None:
        """
        Validate that embeddings have correct shape and properties.

        Overrides base class to add scGPT-specific validation.
        """
        # Call base class validation
        super().validate_embeddings(adata)

        # Additional scGPT-specific validation could go here if needed
        # For now, base validation is sufficient

    def get_container_command(
        self,
        adata_path: Path,
        output_path: Path,
        **kwargs: Any,
    ) -> list[str]:
        """
        Get command to run this model in a container.

        Overrides base class to include scGPT-specific arguments.
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

        # Add scGPT-specific arguments
        if self.model_dir:
            cmd.extend(["--model-dir", str(self.model_dir)])
        if self.n_hvg is not None:
            cmd.extend(["--n-hvg", str(self.n_hvg)])
        if self.hvg_list is not None:
            # For container, we'd need to pass the list differently
            # For now, skip it (user can provide via file path)
            pass
        if self.batch_size:
            cmd.extend(["--batch-size", str(self.batch_size)])
        if self.device:
            cmd.extend(["--device", self.device])

        # Add any additional kwargs
        for k, v in kwargs.items():
            if v is not None:
                cmd.extend([f"--{k.replace('_', '-')}", str(v)])

        return cmd

    def get_required_dependencies(self) -> list[str]:
        """Get required dependencies for scGPT."""
        return ["scgpt", "torch<=2.2.2", "numpy<2", "gdown"]

    def get_optional_dependency_group(self) -> Optional[str]:
        """Get optional dependency group name for scGPT."""
        return "scgpt"

    def get_container_name(self) -> Optional[str]:
        """Get container name for scGPT."""
        return "scgpt"
