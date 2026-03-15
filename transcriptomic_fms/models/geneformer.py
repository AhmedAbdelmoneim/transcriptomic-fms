"""Geneformer embedding model."""

from pathlib import Path
import shutil
import subprocess
import tempfile
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

try:
    import torch
    from transformers import AutoConfig, AutoModel
except ImportError:
    torch = None
    AutoModel = None
    AutoConfig = None

try:
    from datasets import load_from_disk
except ImportError:
    load_from_disk = None

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
                    logger.info(f"Model not found at {self.model_path}. Downloading...")
                    self._download_model()
                else:
                    raise ValueError(
                        f"Model files not found in {self.model_path}. "
                        f"Set auto_download=True to download automatically."
                    )
            else:
                logger.info(f"Using Geneformer model from: {self.model_path}")

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
            self._find_actual_model_dir()

            # Initialize extractor
            # Note: emb_label requires the attribute to be in dataset features
            # We'll handle cell ordering separately by checking the tokenized dataset
            self._extractor = EmbExtractor(
                model_type="Pretrained",
                emb_mode="cell",
                max_ncells=None,
                forward_batch_size=self.batch_size,
                nproc=4,
                emb_layer=-1,
                model_version=GENEFORMER_MODEL_VERSION,
                token_dictionary_file=str(self.token_dict_file) if self.token_dict_file else None,
                emb_label=["cell_id"],  # attach cell_id to each row so we can align after sort
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

        logger.info("Downloading Geneformer model from HuggingFace...")
        logger.info(f"Repository: {GENEFORMER_MODEL_REPO}")
        logger.info(f"Model version: {GENEFORMER_MODEL_VERSION}")
        logger.info("Note: This may take a while as models are large. Using git-lfs...")

        try:
            # Clone the repository to a temporary location first
            with tempfile.TemporaryDirectory() as tmpdir:
                clone_dir = Path(tmpdir) / "Geneformer"

                # Clone with git-lfs
                logger.info("Cloning repository...")
                subprocess.run(
                    ["git", "clone", GENEFORMER_MODEL_REPO, str(clone_dir)],
                    check=True,
                    capture_output=False,
                )

                # Pull LFS files
                logger.info("Downloading model files (git-lfs)...")
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
                    shutil.rmtree(target_model_dir)

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

            logger.info(f"Model downloaded successfully to {self.model_path}")
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
                "cell_id": np.array(
                    cell_names, dtype=str
                ),  # force string regardless of index type
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

    def _load_hf_model(self, model_dir: Path):
        """Load raw HuggingFace transformer from Geneformer model directory."""
        if AutoModel is None or AutoConfig is None:
            raise ImportError(
                "transformers is required for sensitivity analysis. "
                "Install with: pip install transformers"
            )
        config = AutoConfig.from_pretrained(str(model_dir))
        model = AutoModel.from_pretrained(str(model_dir), config=config)
        model = model.to(self.device)
        model.eval()
        return model

    K_JACOBIAN_SVD = 50

    def _jacobian_svd(
        self, J: np.ndarray, d_emb: int, d_input: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Truncated SVD of Jacobian J (d_emb, d_input). Returns U (d_emb, 50) float16, S (50) float32."""
        from scipy.sparse.linalg import svds

        k = self.K_JACOBIAN_SVD
        k_svd = min(k, min(J.shape) - 1)
        if k_svd < 1:
            U = np.zeros((d_emb, k), dtype=np.float16)
            S = np.zeros(k, dtype=np.float32)
            return U, S
        U, S, _ = svds(J.astype(np.float64), k=k_svd)
        U = np.flip(U, axis=1).copy()
        S = np.flip(S).copy()
        U = U.astype(np.float16)
        S = S.astype(np.float32)
        if U.shape[1] < k:
            U_pad = np.zeros((d_emb, k), dtype=np.float16)
            U_pad[:, : U.shape[1]] = U
            S_pad = np.zeros(k, dtype=np.float32)
            S_pad[: S.shape[0]] = S
            U, S = U_pad, S_pad
        return U, S

    def _sensitivity_autograd(
        self,
        hf_model: Any,
        input_ids_1d: np.ndarray,
        max_seq_len: int = 2048,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute ∂(mean_pooled_embedding)/∂(input_token_embeddings), then truncated SVD.

        Geneformer uses mean pooling over token positions (no CLS). J has shape
        (d_emb, seq_len * d_token). Immediately after computing J we reshape to
        (d_emb, seq_len*d_token), run SVD, keep top k=50 U and S, discard J.

        Returns:
            baseline_emb: (d_out,) cell embedding
            jacobian_U: (d_out, 50) float16
            jacobian_S: (50,) float32
        """
        if torch is None:
            raise ImportError("torch is required for sensitivity analysis.")
        seq_len = len(input_ids_1d)
        model = hf_model
        device = self.device
        d_in = model.config.hidden_size
        d_out = model.config.hidden_size

        padded = np.zeros(max_seq_len, dtype=np.int64)
        padded[:seq_len] = input_ids_1d
        att_mask = np.zeros(max_seq_len, dtype=np.int64)
        att_mask[:seq_len] = 1

        ids_t = torch.tensor(padded, dtype=torch.long).unsqueeze(0).to(device)
        mask_t = torch.tensor(att_mask, dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            word_emb = model.embeddings.word_embeddings(ids_t)
        word_emb = word_emb.detach().requires_grad_(True)

        position_ids = torch.arange(max_seq_len, dtype=torch.long, device=device).unsqueeze(0)
        position_emb = model.embeddings.position_embeddings(position_ids)
        if hasattr(model.embeddings, "token_type_embeddings"):
            token_type_ids = torch.zeros_like(ids_t)
            token_type_emb = model.embeddings.token_type_embeddings(token_type_ids)
        else:
            token_type_emb = 0

        full_emb = word_emb + position_emb + token_type_emb
        if hasattr(model.embeddings, "LayerNorm"):
            full_emb = model.embeddings.LayerNorm(full_emb)
        if hasattr(model.embeddings, "dropout"):
            full_emb = model.embeddings.dropout(full_emb)

        extended_mask = model.get_extended_attention_mask(mask_t, ids_t.shape, device=device)
        encoder_out = model.encoder(full_emb, attention_mask=extended_mask)
        hidden = encoder_out.last_hidden_state

        mask_f = mask_t.unsqueeze(-1).float()
        cell_emb = (hidden * mask_f).sum(1) / mask_f.sum(1)

        d_input = seq_len * d_in
        J = np.zeros((d_out, d_input), dtype=np.float64)
        for i in range(d_out):
            if word_emb.grad is not None:
                word_emb.grad.zero_()
            cell_emb[0, i].backward(retain_graph=(i < d_out - 1))
            J[i, :] = word_emb.grad[0, :seq_len, :].detach().cpu().numpy().flatten()

        baseline = cell_emb[0].detach().cpu().numpy()
        U, S = self._jacobian_svd(J, d_out, d_input)
        return baseline, U, S

    def compute_sensitivity(
        self,
        adata: sc.AnnData,
        output_path: Path,
        batch_size: Optional[int] = None,
        n_cells: Optional[int] = None,
        max_seq_len: int = 2048,
        **kwargs: Any,
    ) -> None:
        """
        Compute ∂(mean_pooled_embedding)/∂(token_embeddings), SVD per cell; write to output_path.

        Cells are processed in the order of adata (use CLI --chunk-size to run in chunks).
        Passes through input adata.obs for output cells and adds seq_length. Writes obsm['X_baseline'] (n_cells, d_emb),
        obsm['jacobian_U'] (n_cells, d_emb, 50) float16, obsm['jacobian_S'] (n_cells, 50) float32.
        Full Jacobian is not stored.
        """
        if n_cells is not None:
            adata = adata[:n_cells].copy()
        model_dir = self._find_actual_model_dir()

        logger.info("Loading HuggingFace model for sensitivity analysis...")
        hf_model = self._load_hf_model(model_dir)

        logger.info("Tokenising data...")
        if load_from_disk is None:
            raise ImportError(
                "datasets is required for sensitivity analysis. pip install datasets"
            )

        cell_type_col = "cell_type" if "cell_type" in adata.obs.columns else None
        cell_ids_out = []
        cell_types_out = []
        baselines_out = []
        U_out = []
        S_out = []
        seqlen_out = []

        with tempfile.TemporaryDirectory() as tmpdir:
            tok_dir = Path(tmpdir) / "tokenized"
            tok_dir.mkdir()
            self._tokenize_data(adata, tok_dir)
            dataset_dirs = [
                d for d in tok_dir.iterdir() if d.is_dir() and d.name.endswith(".dataset")
            ]
            if not dataset_dirs:
                dataset_dirs = [d for d in tok_dir.iterdir() if d.is_dir()]
            if not dataset_dirs:
                raise RuntimeError(f"No tokenised dataset found in {tok_dir}")
            dataset = load_from_disk(str(dataset_dirs[0]))

            id_to_row = (
                {str(cid): i for i, cid in enumerate(dataset["cell_id"])}
                if "cell_id" in dataset.column_names
                else {str(i): i for i in range(len(dataset))}
            )

            from tqdm import tqdm

            n_adata = len(adata)
            logger.info("Sensitivity analysis: %d cells (tokenised)", n_adata)
            for obs_idx in tqdm(
                range(n_adata),
                desc="Geneformer sensitivity",
                unit="cell",
                leave=True,
            ):
                cell_id = str(adata.obs_names[obs_idx])
                ct = adata.obs[cell_type_col].iloc[obs_idx] if cell_type_col else "unknown"
                row_idx = id_to_row.get(cell_id)
                if row_idx is None:
                    logger.warning("Skipping %s — not in tokenised dataset", cell_id)
                    continue
                input_ids = np.array(dataset[row_idx]["input_ids"], dtype=np.int64)
                seq_len = len(input_ids)

                baseline, U, S = self._sensitivity_autograd(
                    hf_model, input_ids, max_seq_len=max_seq_len
                )
                cell_ids_out.append(cell_id)
                cell_types_out.append(ct)
                baselines_out.append(baseline)
                U_out.append(U)
                S_out.append(S)
                seqlen_out.append(seq_len)

        baselines_arr = np.array(baselines_out, dtype=np.float32)
        jacobian_U = np.array(U_out, dtype=np.float16)
        jacobian_S = np.array(S_out, dtype=np.float32)
        seq_lengths_arr = np.array(seqlen_out)

        obs_out = adata.obs.loc[cell_ids_out].copy()
        obs_out["seq_length"] = seq_lengths_arr

        result_adata = sc.AnnData(
            X=baselines_arr,
            obs=obs_out,
            obsm={
                "X_baseline": baselines_arr,
                "jacobian_U": jacobian_U,
                "jacobian_S": jacobian_S,
            },
        )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_adata.write(output_path)
        logger.info("Wrote sensitivity results to %s (%d cells)", output_path, len(cell_ids_out))

    def embed(
        self,
        adata: sc.AnnData,
        output_path: Path,
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> sc.AnnData:
        """
        Generate Geneformer embeddings (Robust implementation).
        """
        # 1. Tokenize (preserves cell_id in the .dataset)
        with tempfile.TemporaryDirectory() as tmpdir:
            tokenized_dir = Path(tmpdir) / "tokenized_data"
            tokenized_dir.mkdir()

            logger.info("Tokenizing data for Geneformer...")
            tokenized_data_dir = self._tokenize_data(adata, tokenized_dir)

            # Locate the .dataset directory (Geneformer/HuggingFace format)
            tokenized_contents = list(tokenized_data_dir.iterdir())
            dataset_dirs = [
                d for d in tokenized_contents if d.is_dir() and d.name.endswith(".dataset")
            ]

            if dataset_dirs:
                input_data_path = str(dataset_dirs[0])
            elif any(d.is_dir() for d in tokenized_contents):
                # Fallback: assume the first directory is the dataset
                input_data_path = str([d for d in tokenized_contents if d.is_dir()][0])
            else:
                raise RuntimeError("Could not find valid tokenized dataset directory.")

            # 2. Extract Embeddings
            extractor = self._get_extractor()
            if batch_size is not None:
                extractor.forward_batch_size = batch_size

            actual_model_dir = self._find_actual_model_dir()

            logger.info("Extracting embeddings...")
            with tempfile.TemporaryDirectory() as emb_output_dir:
                extractor.extract_embs(
                    model_directory=str(actual_model_dir),
                    input_data_file=input_data_path,
                    output_directory=emb_output_dir,
                    output_prefix="embeddings",
                    output_torch_embs=False,
                )

                emb_csv = Path(emb_output_dir) / "embeddings.csv"
                if not emb_csv.exists():
                    actual_contents = [p.name for p in Path(emb_output_dir).iterdir()]
                    raise RuntimeError(
                        f"Embeddings CSV not found at {emb_csv}. "
                        f"Actual contents of output dir: {actual_contents}"
                    )

                emb_df = pd.read_csv(emb_csv, index_col=0)

                # Validate precondition before using cell_id
                if "cell_id" not in emb_df.columns:
                    raise RuntimeError(
                        "cell_id column missing from embeddings CSV. "
                        "Ensure cell_id is preserved during tokenization and emb_label=['cell_id'] is set."
                    )

                # Identify embedding columns by excluding known metadata columns.
                # Note: Geneformer's default DataFrame uses integer column names (0, 1, 2...)
                # which become strings after CSV round-trip, so meta_cols uses string keys.
                meta_cols = {"cell_id", "n_counts", "filter_pass"}
                emb_cols = [c for c in emb_df.columns if c not in meta_cols]

                # emb_df rows are in Geneformer's internal sort order (sorted by token length
                # via downsample_and_sort). cell_id column carries the original identity for
                # each row — use it to realign with the input adata.
                surviving_cell_ids = emb_df["cell_id"].astype(str).tolist()
                embeddings = emb_df[emb_cols].values

        # Both temp dirs are now cleaned up; only plain objects remain.
        n_dropped = adata.n_obs - len(surviving_cell_ids)
        if n_dropped > 0:
            logger.warning(
                f"Geneformer filtered out {n_dropped} cells "
                f"({(n_dropped / adata.n_obs) * 100:.1f}%) due to low token counts."
            )

        # Slice original obs to get surviving cells in embedding order,
        # preserving all original obs columns.
        result_obs = adata.obs.loc[surviving_cell_ids].copy()
        result_adata = sc.AnnData(X=embeddings, obs=result_obs)
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
