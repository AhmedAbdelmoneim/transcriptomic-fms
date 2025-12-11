"""Geneformer embedding model."""

import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse

from transcriptomic_fms.models.base import BaseEmbeddingModel
from transcriptomic_fms.models.registry import register_model
from transcriptomic_fms.utils.logging import get_logger

logger = get_logger(__name__)

try:
    from geneformer import EmbExtractor, TranscriptomeTokenizer

    try:
        from geneformer.tokenizer import TOKEN_DICTIONARY_FILE
    except ImportError:
        TOKEN_DICTIONARY_FILE = None
except ImportError:
    EmbExtractor = None
    TranscriptomeTokenizer = None
    TOKEN_DICTIONARY_FILE = None

try:
    import loompy
except ImportError:
    loompy = None

# Default Geneformer model HuggingFace repository
GENEFORMER_MODEL_REPO = "https://huggingface.co/ctheodoris/Geneformer"
GENEFORMER_MODEL_VERSION = "V1"  # Model version for EmbExtractor (V1 or V2)
GENEFORMER_MODEL_DIR = "Geneformer-V1-10M"  # Directory name in repository


@register_model("geneformer")
class GeneformerModel(BaseEmbeddingModel):
    """Geneformer embedding model (V1-10M, cell embeddings only)."""

    def __init__(
        self,
        model_name: str,
        model_path: Optional[str] = None,
        model_dir: Optional[str] = None,
        batch_size: int = 100,
        device: Optional[str] = None,
        auto_download: bool = True,
        download_dir: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initialize Geneformer model.

        Args:
            model_name: Model identifier
            model_path: Path to Geneformer model directory (must contain Geneformer-V1-10M subdirectory).
                       If None and auto_download=True, will download to download_dir.
            model_dir: Alias for model_path (for consistency with other models)
            batch_size: Batch size for forward pass (default: 100)
            device: Device to use ('cuda' or 'cpu'). Auto-detects if None.
            auto_download: If True, automatically download model if not found
            download_dir: Directory to download model to if auto_download=True and model_path is None.
                          Defaults to {project_root}/models/Geneformer
            **kwargs: Additional arguments
        """
        super().__init__(model_name=model_name, requires_gpu=True, **kwargs)

        if EmbExtractor is None:
            raise ImportError(
                "Geneformer is not installed. Install with:\n"
                "  make install-model MODEL=geneformer\n"
                "  or: uv sync --extra geneformer\n"
                "  or: pip install geneformer"
            )

        self.batch_size = batch_size
        self.auto_download = auto_download

        # Auto-detect device
        if device is None:
            try:
                import torch

                cuda_available = torch.cuda.is_available() and torch.cuda.device_count() > 0
                self.device = "cuda" if cuda_available else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        # Handle model_path vs model_dir
        if model_path:
            self.model_path = Path(model_path)
        elif model_dir:
            self.model_path = Path(model_dir)
        elif auto_download:
            # Use default download location
            if download_dir:
                self.model_path = Path(download_dir)
            else:
                # Default to root/models/Geneformer
                import transcriptomic_fms

                package_path = Path(transcriptomic_fms.__file__).parent
                project_root = package_path.parent
                default_model_dir = project_root / "models" / "Geneformer"
                self.model_path = default_model_dir
        else:
            self.model_path = None

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

                print(f"Using Geneformer model from: {self.model_path}", file=sys.stderr)

        # Find token dictionary files in model directory
        self.token_dict_file = self._find_token_dict()
        self.median_dict_file = self._find_median_dict()

        # Initialize extractor (will be done lazily)
        self._extractor: Optional[EmbExtractor] = None

    def _find_token_dict(self) -> Optional[Path]:
        """Find token dictionary file in model directory."""
        if not self.model_path or not self.model_path.exists():
            return None

        # Check common locations
        candidates = [
            self.model_path
            / GENEFORMER_MODEL_DIR
            / "geneformer"
            / "gene_dictionaries_30m"
            / "token_dictionary_gc30M.pkl",
            self.model_path / GENEFORMER_MODEL_DIR / "token_dictionary_gc30M.pkl",
            self.model_path
            / "geneformer"
            / "gene_dictionaries_30m"
            / "token_dictionary_gc30M.pkl",
            self.model_path / "token_dictionary_gc30M.pkl",
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        # Try using default from package
        if TOKEN_DICTIONARY_FILE and Path(TOKEN_DICTIONARY_FILE).exists():
            return Path(TOKEN_DICTIONARY_FILE)

        return None

    def _find_median_dict(self) -> Optional[Path]:
        """Find gene median dictionary file in model directory."""
        if not self.model_path or not self.model_path.exists():
            return None

        # Check common locations
        candidates = [
            self.model_path
            / GENEFORMER_MODEL_DIR
            / "geneformer"
            / "gene_dictionaries_30m"
            / "gene_median_dictionary_gc30M.pkl",
            self.model_path / GENEFORMER_MODEL_DIR / "gene_median_dictionary_gc30M.pkl",
            self.model_path
            / "geneformer"
            / "gene_dictionaries_30m"
            / "gene_median_dictionary_gc30M.pkl",
            self.model_path / "gene_median_dictionary_gc30M.pkl",
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return None

    def _get_extractor(self) -> EmbExtractor:
        """Get or initialize the EmbExtractor."""
        if self._extractor is None:
            if self.model_path is None:
                raise ValueError(
                    "model_path must be provided. Set it in model initialization, "
                    "via --model-path argument, or enable auto_download."
                )

            # Find actual model directory (Geneformer-V1-10M subdirectory)
            actual_model_dir = self._find_actual_model_dir()

            # Initialize extractor
            # Note: emb_label requires the attribute to be in dataset features
            # We'll handle cell ordering separately by checking the tokenized dataset
            self._extractor = EmbExtractor(
                model_type="Pretrained",
                emb_mode="cell",  # Fixed to cell embeddings
                max_ncells=None,  # Process all cells
                forward_batch_size=self.batch_size,
                nproc=4,  # Default number of processes
                emb_layer=-1,  # 2nd to last layer (fixed)
                model_version=GENEFORMER_MODEL_VERSION,  # "V1" or "V2"
                token_dictionary_file=str(self.token_dict_file) if self.token_dict_file else None,
            )
        return self._extractor

    def _find_actual_model_dir(self) -> Path:
        """Find the actual model directory (Geneformer-V1-10M subdirectory)."""
        if not self.model_path or not self.model_path.exists():
            raise ValueError(f"Model directory does not exist: {self.model_path}")

        # Check if model_path itself is the model directory
        model_dir = self.model_path / GENEFORMER_MODEL_DIR
        if model_dir.exists() and (model_dir / "config.json").exists():
            return model_dir

        # Check if model_path contains the model files directly
        if (self.model_path / "config.json").exists():
            return self.model_path

        raise ValueError(
            f"Model directory not found in {self.model_path}. "
            f"Expected to find {GENEFORMER_MODEL_DIR} subdirectory or model files directly."
        )

    def preprocess(self, adata: sc.AnnData, output_path: Optional[Path] = None) -> sc.AnnData:
        """
        Preprocess data for Geneformer: prepare for tokenization.

        Requirements:
        - Gene symbols must be available as Ensembl IDs in:
          * var.index (Ensembl IDs as index)
          * var['ensembl_id'] column
          * var['gene_id'] column (fallback)
        - Data should be raw counts
        - Each cell should have 'n_counts' attribute (will be computed if missing)

        Args:
            adata: Input AnnData object
            output_path: Optional path to save preprocessed data

        Returns:
            AnnData object (unchanged, tokenization is stored separately)
        """
        adata = adata.copy()

        # Determine Ensembl IDs
        ensembl_ids = self._get_ensembl_ids(adata)
        adata.var["ensembl_id"] = ensembl_ids

        # Ensure n_counts exists for each cell
        if "n_counts" not in adata.obs.columns:
            if scipy.sparse.issparse(adata.X):
                adata.obs["n_counts"] = np.array(adata.X.sum(axis=1)).flatten()
            else:
                adata.obs["n_counts"] = adata.X.sum(axis=1)

        # Tokenization will be done in embed() method
        # We just prepare the data here
        if output_path:
            adata.write(output_path)

        return adata

    def _get_ensembl_ids(self, adata: sc.AnnData) -> list[str]:
        """
        Extract Ensembl IDs from AnnData.

        Checks in order:
        1. var['ensembl_id'] column
        2. var['gene_id'] column
        3. var.index (assumes index contains Ensembl IDs)

        Returns:
            List of Ensembl IDs

        Raises:
            ValueError: If Ensembl IDs cannot be determined
        """
        # Check for ensembl_id column
        if "ensembl_id" in adata.var.columns:
            return adata.var["ensembl_id"].tolist()

        # Check for gene_id column (common alternative)
        if "gene_id" in adata.var.columns:
            return adata.var["gene_id"].tolist()

        # Use index as Ensembl IDs
        if adata.var_names is not None and len(adata.var_names) > 0:
            return adata.var_names.tolist()

        raise ValueError(
            "Cannot determine Ensembl IDs. "
            "Please ensure one of the following:\n"
            "  - var.index contains Ensembl IDs\n"
            "  - var['ensembl_id'] column exists\n"
            "  - var['gene_id'] column exists"
        )

    def _model_exists(self) -> bool:
        """Check if model files exist in model directory."""
        if not self.model_path or not self.model_path.exists():
            return False

        # Check for model directory structure
        # Model should have config.json and model files
        model_dir = self.model_path / GENEFORMER_MODEL_DIR
        if model_dir.exists():
            # Check for key model files
            required_files = ["config.json"]
            return all((model_dir / f).exists() for f in required_files)

        # Check if model files are directly in model_path
        if (self.model_path / "config.json").exists():
            return True

        return False

    def _download_model(self) -> None:
        """Download Geneformer model from HuggingFace."""
        # Check if git-lfs is available
        try:
            subprocess.run(["git", "lfs", "version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "git-lfs is required for downloading Geneformer models. "
                "Install with: apt-get install git-lfs (or brew install git-lfs)\n"
                "Then run: git lfs install"
            )

        print(f"Downloading Geneformer model from HuggingFace...")
        print(f"Repository: {GENEFORMER_MODEL_REPO}")
        print(f"Model version: {GENEFORMER_MODEL_VERSION}")
        print("Note: This may take a while as models are large. Using git-lfs...")

        try:
            # Clone the repository to a temporary location first
            with tempfile.TemporaryDirectory() as tmpdir:
                clone_dir = Path(tmpdir) / "Geneformer"

                # Clone with git-lfs
                print("Cloning repository...")
                subprocess.run(
                    ["git", "clone", GENEFORMER_MODEL_REPO, str(clone_dir)],
                    check=True,
                    capture_output=False,
                )

                # Pull LFS files
                print("Downloading model files (git-lfs)...")
                subprocess.run(
                    ["git", "lfs", "pull"],
                    cwd=str(clone_dir),
                    check=True,
                    capture_output=False,
                )

                # Move the specific model directory to target location
                source_model_dir = clone_dir / GENEFORMER_MODEL_DIR
                if not source_model_dir.exists():
                    raise RuntimeError(
                        f"Model directory {GENEFORMER_MODEL_DIR} not found in cloned repository. "
                        f"Available directories: {[d.name for d in clone_dir.iterdir() if d.is_dir()]}"
                    )

                # Move model directory to target
                target_model_dir = self.model_path / GENEFORMER_MODEL_DIR
                if target_model_dir.exists():
                    import shutil

                    shutil.rmtree(target_model_dir)

                import shutil

                shutil.move(str(source_model_dir), str(target_model_dir))

                # Also copy token dictionaries if they exist
                token_dict_source = clone_dir / "geneformer" / "gene_dictionaries_30m"
                if token_dict_source.exists():
                    token_dict_target = self.model_path / "geneformer" / "gene_dictionaries_30m"
                    token_dict_target.parent.mkdir(parents=True, exist_ok=True)
                    if token_dict_target.exists():
                        shutil.rmtree(token_dict_target)
                    shutil.copytree(str(token_dict_source), str(token_dict_target))

            # Verify download
            if not self._model_exists():
                raise RuntimeError(
                    "Model download completed but required files are missing. "
                    f"Expected to find {GENEFORMER_MODEL_DIR} in {self.model_path}"
                )

            print(f"Model downloaded successfully to {self.model_path}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to download Geneformer model: {e}\n"
                f"You can manually download from: {GENEFORMER_MODEL_REPO}\n"
                f"Clone to: {self.model_path}\n"
                f"Or set model_path to point to an existing model directory."
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to download Geneformer model: {e}\n"
                f"You can manually download from: {GENEFORMER_MODEL_REPO}\n"
                f"Clone to: {self.model_path}\n"
                f"Or set model_path to point to an existing model directory."
            ) from e

    def _tokenize_data(self, adata: sc.AnnData, output_dir: Path) -> Path:
        """
        Tokenize AnnData and save as .dataset format.

        Args:
            adata: Input AnnData object
            output_dir: Directory to save tokenized .dataset

        Returns:
            Path to tokenized .dataset directory
        """
        if loompy is None:
            raise ImportError(
                "loompy is required for tokenization. Install with: pip install loompy"
            )

        # Create temporary loom file for tokenization
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_loom = Path(tmpdir) / "data.loom"

            # Convert AnnData to loom format
            gene_names = adata.var["ensembl_id"].values
            cell_names = adata.obs_names.values

            # Convert to dense if needed (loompy requires dense arrays)
            if scipy.sparse.issparse(adata.X):
                X_dense = adata.X.toarray()
            else:
                X_dense = adata.X

            # Create loom file
            # Ensure all attributes are numpy arrays
            row_attrs = {"ensembl_id": np.array(gene_names)}
            col_attrs = {
                "cell_id": np.array(cell_names),
                "n_counts": np.array(adata.obs["n_counts"].values),
            }

            # Add any additional cell attributes (convert to numpy arrays)
            for col in adata.obs.columns:
                if col not in col_attrs:
                    values = adata.obs[col].values
                    # Convert to numpy array, handling different types
                    if not isinstance(values, np.ndarray):
                        values = np.array(values)
                    # Convert object arrays to strings if needed
                    if values.dtype == object:
                        values = values.astype(str)
                    col_attrs[col] = values

            loompy.create(str(tmp_loom), X_dense.T, row_attrs, col_attrs)

            # Tokenize using geneformer tokenizer
            if TranscriptomeTokenizer is None:
                raise ImportError(
                    "TranscriptomeTokenizer not available. "
                    "Ensure geneformer is properly installed."
                )

            # Initialize tokenizer
            # V1-10M model settings: special_token=False, model_input_size=2048
            # Use custom_attr_name_dict to preserve cell_id in tokenized dataset
            tokenizer_kwargs = {
                "nproc": 4,  # Default number of processes
                "special_token": False,  # V1 models don't use special tokens
                "model_input_size": 2048,  # V1 model input size
                "custom_attr_name_dict": {"cell_id": "cell_id"},  # Preserve cell_id attribute
            }

            if self.token_dict_file:
                tokenizer_kwargs["token_dictionary_file"] = str(self.token_dict_file)
            if self.median_dict_file:
                tokenizer_kwargs["gene_median_file"] = str(self.median_dict_file)

            tokenizer = TranscriptomeTokenizer(**tokenizer_kwargs)

            # Tokenize the data
            # tokenize_data expects a directory containing input files
            # Create a temporary input directory with the loom file
            import shutil

            input_dir = Path(tmpdir) / "input_data"
            input_dir.mkdir(exist_ok=True)
            shutil.copy(str(tmp_loom), str(input_dir / "data.loom"))

            tokenizer.tokenize_data(
                data_directory=str(input_dir),
                output_directory=str(output_dir),
                output_prefix="tokenized",
                file_format="loom",
            )

        return output_dir

    def embed(
        self,
        adata: sc.AnnData,
        output_path: Path,
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> sc.AnnData:
        """
        Generate Geneformer embeddings.

        This method tokenizes the data (if not already tokenized) and then
        extracts embeddings using EmbExtractor.

        Args:
            adata: Preprocessed AnnData object
            output_path: Path where embeddings will be saved
            batch_size: Batch size for forward pass (overrides default if provided)
            **kwargs: Additional arguments (ignored)

        Returns:
            AnnData object with embeddings in X (shape: n_cells, n_dimensions)
            and obs preserved for cell mapping. No var needed.
        """
        # Store original cell order to preserve it in output
        original_cell_order = adata.obs_names.values.copy()

        # Create temporary directory for tokenized data
        with tempfile.TemporaryDirectory() as tmpdir:
            tokenized_dir = Path(tmpdir) / "tokenized_data"
            tokenized_dir.mkdir()

            # Tokenize data
            logger.info("Tokenizing data for Geneformer...")
            tokenized_data_dir = self._tokenize_data(adata, tokenized_dir)

            # The tokenizer creates a .dataset directory (HuggingFace Dataset format)
            # The output is: {output_dir}/{output_prefix}.dataset/
            # Check what was actually created
            tokenized_contents = list(tokenized_data_dir.iterdir())

            # Look for .dataset directory - this is what extract_embs expects
            dataset_dirs = [
                d for d in tokenized_contents if d.is_dir() and d.name.endswith(".dataset")
            ]
            if dataset_dirs:
                # Found .dataset directory - extract_embs expects the path to this directory
                dataset_dir = dataset_dirs[0]
                input_data_path = str(dataset_dir)  # Path to the .dataset directory itself
            else:
                # Check if tokenized_data_dir itself is structured as a dataset
                # Or look for any subdirectory
                if any(d.is_dir() for d in tokenized_contents):
                    # Try using the first subdirectory
                    dataset_dir = [d for d in tokenized_contents if d.is_dir()][0]
                    input_data_path = str(dataset_dir)
                else:
                    raise RuntimeError(
                        f"Tokenized dataset not found in expected format. "
                        f"Contents of {tokenized_data_dir}: {[str(p.name) for p in tokenized_contents]}. "
                        "Expected a .dataset directory or dataset structure."
                    )

            # Get extractor
            extractor = self._get_extractor()

            # Override batch size if provided
            if batch_size is not None:
                extractor.forward_batch_size = batch_size

            # Find actual model directory
            actual_model_dir = self._find_actual_model_dir()

            # Extract embeddings
            # extract_embs expects input_data_file to be a directory containing the dataset
            logger.info("Extracting embeddings with Geneformer...")
            with tempfile.TemporaryDirectory() as emb_output_dir:
                extractor.extract_embs(
                    model_directory=str(actual_model_dir),
                    input_data_file=input_data_path,  # Directory containing the dataset
                    output_directory=emb_output_dir,
                    output_prefix="embeddings",
                    output_torch_embs=False,  # We'll load from CSV
                )

                # Load embeddings from CSV
                emb_csv = Path(emb_output_dir) / "embeddings.csv"
                if not emb_csv.exists():
                    raise RuntimeError(
                        f"Embeddings file not found: {emb_csv}. "
                        "Check that extract_embs completed successfully."
                    )

                # Read embeddings CSV
                # Geneformer outputs CSV with numeric indices, but embeddings are in same order as input
                emb_df = pd.read_csv(emb_csv, index_col=0)

                # Extract embedding columns (all columns except metadata)
                # Embedding columns are typically named 'emb_0', 'emb_1', etc.
                emb_cols = [col for col in emb_df.columns if col.startswith("emb_")]
                if not emb_cols:
                    # Try to find numeric columns (embeddings)
                    emb_cols = [
                        col
                        for col in emb_df.columns
                        if emb_df[col].dtype in [np.float64, np.float32]
                        and col not in ["n_counts", "filter_pass"]
                    ]

                if not emb_cols:
                    raise ValueError(
                        "Could not identify embedding columns in output CSV. "
                        f"Available columns: {emb_df.columns.tolist()}"
                    )

                # Ensure embeddings match number of cells
                if emb_df.shape[0] != adata.n_obs:
                    raise ValueError(
                        f"Embeddings shape mismatch: expected {adata.n_obs} cells, "
                        f"got {emb_df.shape[0]}"
                    )

                # Verify embeddings are in correct order by checking tokenized dataset
                # The tokenized dataset preserves cell_id, and embeddings are extracted in that order
                try:
                    from datasets import load_from_disk

                    tokenized_dataset = load_from_disk(str(input_data_path))

                    if "cell_id" in tokenized_dataset.features:
                        tokenized_cell_ids = tokenized_dataset["cell_id"]
                        # Check if order matches input order
                        if list(tokenized_cell_ids) != list(original_cell_order):
                            # Reorder embeddings to match original cell order
                            tokenized_to_original_idx = {
                                cell_id: orig_idx
                                for orig_idx, cell_id in enumerate(original_cell_order)
                            }
                            reorder_idx = [
                                tokenized_to_original_idx.get(cell_id, None)
                                for cell_id in tokenized_cell_ids
                            ]
                            if None in reorder_idx:
                                raise ValueError(
                                    "Some cells in tokenized dataset don't match original cells"
                                )
                            embeddings = emb_df.iloc[reorder_idx][emb_cols].values
                        else:
                            # Order matches - embeddings are already in correct order
                            embeddings = emb_df[emb_cols].values
                    else:
                        # No cell_id in dataset - assume order is preserved
                        embeddings = emb_df[emb_cols].values
                except Exception:
                    # Fallback: assume order is preserved (tokenized dataset should match input order)
                    embeddings = emb_df[emb_cols].values

                # Verify we have the right number of embeddings
                if embeddings.shape[0] != len(original_cell_order):
                    raise ValueError(
                        f"Embedding count mismatch: got {embeddings.shape[0]} embeddings, "
                        f"expected {len(original_cell_order)} cells. "
                        f"This suggests cells were filtered during processing."
                    )

                # Create barebones AnnData with embeddings in X and obs preserved
                # Ensure obs matches the order of embeddings (which should match original_cell_order)
                result_obs = adata.obs.loc[original_cell_order].copy()
                result_adata = sc.AnnData(
                    X=embeddings,
                    obs=result_obs,
                )
                # No var needed for embeddings

                # Validate embeddings
                self.validate_embeddings(result_adata)

                return result_adata

    def get_container_command(
        self,
        adata_path: Path,
        output_path: Path,
        **kwargs: Any,
    ) -> list[str]:
        """Get command to run this model in a container."""
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

        # Add Geneformer-specific arguments
        if self.model_path:
            cmd.extend(["--model-path", str(self.model_path)])
        if self.batch_size:
            cmd.extend(["--batch-size", str(self.batch_size)])

        return cmd

    def get_required_dependencies(self) -> list[str]:
        """Get required dependencies for Geneformer."""
        return ["geneformer", "loompy"]

    def get_optional_dependency_group(self) -> Optional[str]:
        """Get optional dependency group name for Geneformer."""
        return "geneformer"

    def get_container_name(self) -> Optional[str]:
        """Get container name for Geneformer."""
        return "geneformer"
