"""scGPT embedding model."""

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import scanpy as sc
import torch

from transcriptomic_fms.models.base import BaseEmbeddingModel
from transcriptomic_fms.models.registry import register_model
from transcriptomic_fms.utils.gene_ids import normalize_gene_symbol
from transcriptomic_fms.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import scgpt as scg
    from scgpt.preprocess import Preprocessor
    from scgpt.tasks import embed_data
except ImportError:
    scg = None
    Preprocessor = None
    embed_data = None

# For sensitivity we need the transformer and vocab directly (same as tasks/cell_emb)
try:
    from scgpt.model import TransformerModel as _TransformerModel
    from scgpt.tokenizer import GeneVocab as _GeneVocab
    from scgpt.utils import load_pretrained as _load_pretrained
except ImportError:
    try:
        from scgpt.model.model import TransformerModel as _TransformerModel
        from scgpt.tokenizer.gene_tokenizer import GeneVocab as _GeneVocab
        from scgpt.utils.util import load_pretrained as _load_pretrained
    except ImportError:
        _TransformerModel = None
        _GeneVocab = None
        _load_pretrained = None

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
                    logger.info(f"Model not found at {self.model_dir}. Downloading...")
                    self._download_model()
                else:
                    raise ValueError(
                        f"Model files not found in {self.model_dir}. "
                        f"Required files: best_model.pt, args.json, vocab.json. "
                        f"Set auto_download=True to download automatically."
                    )
            else:
                logger.info(f"Using scGPT model from: {self.model_dir}")

    def preprocess(self, adata: sc.AnnData, output_path: Optional[Path] = None) -> sc.AnnData:
        """
        Preprocess data for scGPT: select HVGs and bin values.

        Requirements:
        - Gene symbols must be available either as:
          * var['gene_symbol'] column (preferred, singular)
          * var['gene_symbols'] column (plural, backward compatibility)
          * var.index (gene symbols as index)
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

        # Uppercase symbols for scGPT human vocab matching (e.g. mouse Pf4 -> PF4).
        gene_symbols = [
            normalize_gene_symbol(symbol) for symbol in self._get_gene_symbols(adata)
        ]
        adata.var["gene_symbols"] = gene_symbols

        # Handle HVG selection
        if self.hvg_list is not None:
            available_hvgs = self._resolve_hvg_columns(adata, self.hvg_list)
            if len(available_hvgs) == 0:
                raise ValueError(
                    f"None of the provided HVG genes are found in adata.var_names. "
                    f"First few provided: {self.hvg_list[:5]}"
                )
            if len(available_hvgs) < len(self.hvg_list):
                logger.warning(
                    f"Only {len(available_hvgs)}/{len(self.hvg_list)} "
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
        # Note: gene_symbols column is already set above, filtering preserves it

        # Filter out cells with all-zero expression (empty rows)
        # This prevents errors in binning when encountering zero-size arrays
        cell_sums = np.array(adata.X.sum(axis=1)).flatten()
        non_zero_cells = cell_sums > 0
        if not np.all(non_zero_cells):
            n_filtered = np.sum(~non_zero_cells)
            logger.info(
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

    def _resolve_hvg_columns(self, adata: sc.AnnData, hvg_list: list[str]) -> list[str]:
        """Map an HVG list onto ``adata.var_names`` with case-insensitive fallback."""
        available: list[str] = []
        seen: set[str] = set()

        exact = set(adata.var_names.astype(str))
        by_upper = {
            normalize_gene_symbol(name): str(name) for name in adata.var_names.astype(str)
        }

        for gene in hvg_list:
            gene_str = str(gene)
            if gene_str in exact and gene_str not in seen:
                available.append(gene_str)
                seen.add(gene_str)
                continue
            mapped = by_upper.get(normalize_gene_symbol(gene_str))
            if mapped is not None and mapped not in seen:
                available.append(mapped)
                seen.add(mapped)

        return available

    def _normalize_gene_symbols_column(self, adata: sc.AnnData) -> sc.AnnData:
        """Ensure ``var['gene_symbols']`` exists and is uppercased for vocab matching."""
        if "gene_symbols" not in adata.var.columns:
            adata = adata.copy()
            adata.var["gene_symbols"] = [
                normalize_gene_symbol(symbol) for symbol in self._get_gene_symbols(adata)
            ]
            return adata

        symbols = [normalize_gene_symbol(symbol) for symbol in adata.var["gene_symbols"]]
        if list(adata.var["gene_symbols"]) != symbols:
            adata = adata.copy()
            adata.var["gene_symbols"] = symbols
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

        logger.info(f"Downloading scGPT model to {self.model_dir}...")
        logger.info(f"Source: {SCGPT_MODEL_URL}")

        try:
            # Set SSL certificate path for gdown (Ubuntu/Debian locations)
            import os

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

            logger.info(f"Model downloaded successfully to {self.model_dir}")
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

        # Ensure gene_symbols are uppercased for human-vocab matching
        adata = self._normalize_gene_symbols_column(adata)

        vocab_file = self.model_dir / "vocab.json"

        if vocab_file.exists():
            with open(vocab_file, "r") as f:
                vocab = json.load(f)

            # scGPT vocab keys are uppercase gene symbols
            vocab_genes = {normalize_gene_symbol(gene) for gene in vocab.keys()}

            # Identify genes that are in the model vocabulary
            common_genes = adata.var[adata.var["gene_symbols"].isin(vocab_genes)].index

            # 3. Subset to these genes locally first
            adata_safe = adata[:, common_genes].copy()

            # 4. CRITICAL: Remove cells that are now empty due to this subset
            sc.pp.filter_cells(adata_safe, min_counts=1)

            logger.info(
                f"Safety filter: Reduced from {adata.n_obs} to {adata_safe.n_obs} cells "
                f"after intersecting with model vocabulary "
                f"({len(common_genes)}/{adata.n_vars} genes in vocab)."
            )
            adata = adata_safe

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
            use_fast_transformer=False,
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
        result_obs = (
            embedded_adata.obs.copy()
            if embedded_adata.n_obs == embeddings.shape[0]
            else adata.obs.copy()
        )
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

    def compute_sensitivity(
        self,
        adata: sc.AnnData,
        output_path: Path,
        batch_size: Optional[int] = None,
        n_cells: Optional[int] = None,
        max_length: int = 1200,
        **kwargs: Any,
    ) -> None:
        """
        Compute ∂(CLS cell embedding)/∂(input expression values), SVD per cell; write to output_path.

        Uses the same pipeline as embed_data: vocab, model._encode, CLS token at position 0.
        Passes through input adata.obs and adds seq_length. Writes obsm['X_baseline'],
        obsm['jacobian_U'] (n_cells, d_emb, 50) float16, obsm['jacobian_S'] (n_cells, 50) float32.
        """
        if _TransformerModel is None or _GeneVocab is None or _load_pretrained is None:
            raise ImportError(
                "scGPT sensitivity requires internal modules (model, tokenizer, utils). "
                "Ensure scgpt is installed and the package structure provides TransformerModel, "
                "GeneVocab, and load_pretrained."
            )
        if self.model_dir is None:
            raise ValueError("model_dir must be set for sensitivity analysis.")
        model_dir = Path(self.model_dir)
        vocab_file = model_dir / "vocab.json"
        model_config_file = model_dir / "args.json"
        model_file = model_dir / "best_model.pt"
        if not model_config_file.exists() or not model_file.exists() or not vocab_file.exists():
            raise FileNotFoundError(
                f"Model files not found in {model_dir}. Need args.json, best_model.pt, vocab.json."
            )

        device = torch.device(self.device if isinstance(self.device, str) else self.device)
        gene_col = "gene_symbols"
        adata = self._normalize_gene_symbols_column(adata)

        vocab = _GeneVocab.from_file(str(vocab_file))
        pad_token = "<pad>"
        special_tokens = [pad_token, "<cls>", "<eoc>"]
        for s in special_tokens:
            if s not in vocab:
                vocab.append_token(s)
        adata.var["id_in_vocab"] = [
            vocab[gene] if gene in vocab else -1 for gene in adata.var[gene_col]
        ]
        in_vocab = np.array(adata.var["id_in_vocab"]) >= 0
        adata = adata[:, in_vocab].copy()
        gene_ids_var = np.array(adata.var["id_in_vocab"])

        with open(model_config_file, "r") as f:
            model_configs = json.load(f)
        vocab.set_default_index(vocab[pad_token])
        pad_value = float(model_configs.get("pad_value", 0.0))
        pad_token_id = vocab[model_configs["pad_token"]]
        cls_id = vocab["<cls>"]

        model = _TransformerModel(
            ntoken=len(vocab),
            d_model=model_configs["embsize"],
            nhead=model_configs["nheads"],
            d_hid=model_configs["d_hid"],
            nlayers=model_configs["nlayers"],
            nlayers_cls=model_configs.get("n_layers_cls", 3),
            n_cls=1,
            vocab=vocab,
            dropout=model_configs.get("dropout", 0.1),
            pad_token=model_configs.get("pad_token", "<pad>"),
            pad_value=model_configs["pad_value"],
            do_mvc=True,
            do_dab=False,
            use_batch_labels=False,
            domain_spec_batchnorm=False,
            explicit_zero_prob=False,
            use_fast_transformer=False,
            fast_transformer_backend="flash",
            pre_norm=False,
            input_emb_style=model_configs.get("input_emb_style", "continuous"),
            n_input_bins=model_configs.get("n_input_bins"),
        )
        _load_pretrained(model, torch.load(model_file, map_location=device), verbose=False)
        model.to(device)
        model.eval()

        if hasattr(adata.X, "toarray"):
            count_matrix = adata.X.toarray()
        else:
            count_matrix = np.asarray(adata.X, dtype=np.float32)
        if n_cells is not None:
            adata = adata[:n_cells].copy()
            count_matrix = count_matrix[:n_cells]
        n_cells_data = count_matrix.shape[0]
        n_genes = count_matrix.shape[1]

        logger.info(
            "Sensitivity analysis: %d cells, %d genes, device=%s, max_length=%d",
            n_cells_data,
            n_genes,
            device,
            max_length,
        )

        baselines_out = []
        U_out = []
        S_out = []
        seqlen_out = []

        from tqdm import tqdm

        for idx in tqdm(
            range(n_cells_data),
            desc="scGPT sensitivity",
            unit="cell",
            leave=True,
        ):
            row = count_matrix[idx]
            nonzero_idx = np.nonzero(row)[0]
            values = row[nonzero_idx].astype(np.float32)
            genes = gene_ids_var[nonzero_idx]
            genes = np.insert(genes, 0, cls_id)
            values = np.insert(values, 0, pad_value)
            seq_len = len(genes)
            if seq_len > max_length:
                genes = genes[:max_length]
                values = values[:max_length]
                seq_len = max_length
            input_gene_ids = np.full(max_length, pad_token_id, dtype=np.int64)
            input_gene_ids[:seq_len] = genes
            expr_np = np.zeros(max_length, dtype=np.float32)
            expr_np[:seq_len] = values
            expr_np[seq_len:] = pad_value

            input_gene_ids_t = torch.from_numpy(input_gene_ids).long().unsqueeze(0).to(device)
            expr_t = torch.from_numpy(expr_np).float().unsqueeze(0).to(device)
            expr_t.requires_grad_(True)
            src_key_padding_mask = input_gene_ids_t.eq(pad_token_id)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                embeddings = model._encode(
                    input_gene_ids_t,
                    expr_t,
                    src_key_padding_mask,
                    batch_labels=None,
                )
            cell_emb = embeddings[:, 0, :].squeeze(0)
            baseline = cell_emb.detach().cpu().numpy()
            d_emb = baseline.shape[0]

            def _cell_emb_from_expr(expr_flat: torch.Tensor) -> torch.Tensor:
                e = expr_flat.unsqueeze(0)
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    out = model._encode(
                        input_gene_ids_t,
                        e,
                        src_key_padding_mask,
                        batch_labels=None,
                    )
                return out[:, 0, :].squeeze(0)

            J_tensor = torch.autograd.functional.jacobian(
                _cell_emb_from_expr,
                expr_t.squeeze(0),
                vectorize=True,
            )
            J_full = J_tensor.detach().cpu().numpy().astype(np.float64)
            J = J_full[:, :seq_len]
            d_input = seq_len
            U, S = self._jacobian_svd(J, d_emb, d_input)

            baselines_out.append(baseline)
            U_out.append(U)
            S_out.append(S)
            seqlen_out.append(seq_len)

        baselines_arr = np.array(baselines_out, dtype=np.float32)
        jacobian_U = np.array(U_out, dtype=np.float16)
        jacobian_S = np.array(S_out, dtype=np.float32)
        seq_lengths_arr = np.array(seqlen_out)

        obs_out = adata.obs.copy()
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
        logger.info(
            "Wrote sensitivity results to %s (%d cells)",
            output_path,
            n_cells_data,
        )

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
