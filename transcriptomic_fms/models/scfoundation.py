"""scFoundation embedding model."""

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import scanpy as sc

from transcriptomic_fms.models.base import BaseEmbeddingModel
from transcriptomic_fms.models.registry import register_model
from transcriptomic_fms.utils.logging import get_logger

logger = get_logger(__name__)

# URLs for downloading scFoundation resources
SCFOUNDATION_GENE_INDEX_URL = (
    "https://raw.githubusercontent.com/biomap-research/scFoundation/main/model/"
    "OS_scRNA_gene_index.19264.tsv"
)
SCFOUNDATION_REPO_URL = "https://github.com/biomap-research/scFoundation.git"
# Note: Model checkpoint must be downloaded manually from SharePoint:
# https://hopebio2020.sharepoint.com/:f:/s/PublicSharedfiles/IgBlEJ72TBE5Q76AmgXbgjXiAR69fzcrgzqgUYdSThPLrqk

try:
    # Try importing scFoundation model loading functions
    # scFoundation expects the model code to be available
    # We'll need to handle the scFoundation imports dynamically
    # The actual scFoundation package structure may vary
    _scFoundation_available = False
    try:
        # Try importing from a potential scFoundation package
        from load import (
            gatherData,
            getEncoerDecoderData,
            load_model_frommmf,
        )
        from pretrainmodels import select_model

        _scFoundation_available = True
    except ImportError:
        # If direct import fails, we'll handle it in __init__
        pass
except ImportError:
    _scFoundation_available = False
    load_model_frommmf = None
    gatherData = None
    getEncoerDecoderData = None
    select_model = None


@register_model("scfoundation")
class SCFoundationModel(BaseEmbeddingModel):
    """scFoundation pretraining model for single-cell or bulk RNA-seq embeddings."""

    def __init__(
        self,
        model_name: str,
        model_path: Optional[str] = None,
        ckpt_name: str = "01B-resolution",
        gene_index_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        requires_gpu: bool = True,
        input_type: str = "singlecell",
        output_type: str = "cell",
        pool_type: str = "all",
        tgthighres: str = "t4",
        pre_normalized: str = "F",
        version: str = "ce",
        auto_download: bool = True,
        download_dir: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initialize scFoundation model.

        Args:
            model_name: Model identifier (fixed to ``"scfoundation"`` in registry).
            model_path: Path to the model checkpoint file. If None and auto_download=True,
                will use default path in download_dir or {project_root}/models/scfoundation/models.ckpt.
            ckpt_name: Checkpoint name (default: ``"01B-resolution"``).
            gene_index_path: Path to the gene index file
                ``OS_scRNA_gene_index.19264.tsv``. If None and auto_download=True,
                will download to download_dir or {project_root}/models/scfoundation/.
            cache_dir: Directory to cache downloaded models (unused for now).
            device: Device hint for scFoundation / PyTorch (``"cuda"`` or ``"cpu"``).
                    If ``None``, will auto-detect.
            requires_gpu: Whether this model requires a GPU (default: True).
            input_type: ``"singlecell"`` or ``"bulk"`` (matches scFoundation ``get_embedding.py``).
                Bulk uses library-sum style tail tokens; ``tgthighres`` applies only to single-cell.
            output_type: Type of output embeddings. Choices: ``"cell"``, ``"gene"``,
                ``"gene_batch"``. Default: ``"cell"``.
            pool_type: Pooling type for cell embeddings. Choices: ``"all"``, ``"max"``.
                Only valid when ``output_type="cell"``. Default: ``"all"``.
            tgthighres: Target high resolution token value. Single-cell only. Can be:
                - ``"t<number>"``: targeted high resolution (T=number)
                - ``"f<number>"``: fold change (T/S=number)
                - ``"a<number>"``: addition (T=S+number)
                Default: ``"t4"``.
            pre_normalized: Normalization mode (see scFoundation README). For **single-cell**:
                ``F``/``T`` = raw vs log1p(CPM); ``A`` = GEARS-style (last column = total count).
                For **bulk**: only ``F`` or ``T`` — ``F`` runs ``normalize_total`` + ``log1p`` in
                :meth:`preprocess`, and tail tokens use ``log10(row sum)``; ``T`` uses raw values
                and ``sum`` for tail tokens. Default: ``"F"``.
            version: Model version. Choices: ``"ce"`` (cell embedding), ``"rde"`` (read depth
                enhancement). Only valid when ``output_type="cell"``. Default: ``"ce"``.
            auto_download: If True, automatically download gene index file if not found.
                Note: Model checkpoint must be downloaded manually from SharePoint.
            download_dir: Directory to download files to if auto_download=True.
                          Defaults to {project_root}/models/scfoundation
            **kwargs: Additional, model-specific configuration (stored in ``self.config``).
        """
        super().__init__(model_name=model_name, requires_gpu=requires_gpu, **kwargs)

        if input_type not in ("singlecell", "bulk"):
            raise ValueError(f"input_type must be 'singlecell' or 'bulk', got {input_type!r}")
        if input_type == "bulk" and pre_normalized not in ("F", "T"):
            raise ValueError(
                "For input_type='bulk', pre_normalized must be 'F' or 'T' "
                "(append-total mode 'A' is single-cell only in scFoundation)."
            )

        # Check if scFoundation dependencies are available
        if not _scFoundation_available:
            # Try to import by adding common paths
            try:
                import sys

                # Try to find scFoundation in common locations
                # The model files (load.py, pretrainmodels.py) are in the model/ subdirectory
                # Path is already imported at module level
                potential_paths = [
                    Path(__file__).parent.parent.parent / "scFoundation" / "model",
                    Path.cwd() / "scFoundation" / "model",
                    Path.home() / "scFoundation" / "model",
                ]

                for path in potential_paths:
                    if path.exists():
                        # Add the model directory itself to sys.path so we can import load.py and pretrainmodels.py
                        sys.path.insert(0, str(path))
                        break

                # Try importing again
                from load import (
                    gatherData,
                    getEncoerDecoderData,
                    load_model_frommmf,
                )
                from pretrainmodels import select_model

                # Make available globally
                globals()["load_model_frommmf"] = load_model_frommmf
                globals()["gatherData"] = gatherData
                globals()["getEncoerDecoderData"] = getEncoerDecoderData
                globals()["select_model"] = select_model
                globals()["_scFoundation_available"] = True
            except ImportError:
                raise ImportError(
                    "scFoundation is not installed or not found. "
                    "Please ensure scFoundation model code is available.\n"
                    "Install with:\n"
                    "  - Clone the scFoundation repository: "
                    "git clone https://github.com/biomap-research/scFoundation.git\n"
                    "  - Ensure the model/ directory is accessible\n"
                    "  - Or install scFoundation package if available"
                )

        self.ckpt_name = ckpt_name
        self.input_type = input_type
        self.output_type = output_type
        self.pool_type = pool_type
        self.tgthighres = tgthighres
        self.pre_normalized = pre_normalized
        self.version = version
        self.cache_dir = cache_dir
        self.auto_download = auto_download

        # Auto-detect device if not provided
        if device is None:
            try:
                import torch

                cuda_available = torch.cuda.is_available() and torch.cuda.device_count() > 0
                self.device = "cuda" if cuda_available else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        # Determine download directory
        if download_dir:
            self.download_dir = Path(download_dir)
        elif auto_download:
            # Default to root/models/scfoundation
            import transcriptomic_fms

            package_path = Path(transcriptomic_fms.__file__).parent
            project_root = package_path.parent
            self.download_dir = project_root / "models" / "scfoundation"
        else:
            self.download_dir = None

        # Determine model path
        if model_path:
            self.model_path = Path(model_path)
        elif auto_download and self.download_dir:
            # Use default location in download_dir
            self.model_path = self.download_dir / "models" / "models.ckpt"
        else:
            # Fallback to relative path
            self.model_path = Path("./models/models.ckpt")

        # Check if model exists
        if self.model_path and not self.model_path.exists():
            if auto_download:
                logger.warning(
                    f"Model checkpoint not found at {self.model_path}.\n"
                    "The scFoundation model checkpoint must be downloaded manually from:\n"
                    "  https://hopebio2020.sharepoint.com/:f:/s/PublicSharedfiles/"
                    "IgBlEJ72TBE5Q76AmgXbgjXiAR69fzcrgzqgUYdSThPLrqk\n"
                    f"Please download it and place it at: {self.model_path}"
                )
            else:
                raise ValueError(
                    f"Model checkpoint not found at {self.model_path}. "
                    "Set auto_download=True or provide model_path."
                )

        # Determine gene index path
        if gene_index_path:
            self.gene_index_path = Path(gene_index_path)
        elif auto_download and self.download_dir:
            # Use default location in download_dir
            self.gene_index_path = self.download_dir / "OS_scRNA_gene_index.19264.tsv"
        else:
            # Look for gene index in common locations
            potential_paths = [
                self.model_path.parent / "OS_scRNA_gene_index.19264.tsv",
                Path(__file__).parent.parent.parent
                / "scFoundation"
                / "model"
                / "OS_scRNA_gene_index.19264.tsv",
                Path.cwd() / "OS_scRNA_gene_index.19264.tsv",
                Path.cwd() / "scFoundation" / "model" / "OS_scRNA_gene_index.19264.tsv",
            ]

            for path in potential_paths:
                if path.exists():
                    self.gene_index_path = path
                    break
            else:
                self.gene_index_path = None

        # Download gene index if needed
        if self.gene_index_path and not self.gene_index_path.exists():
            if auto_download:
                logger.info(f"Gene index file not found. Downloading to {self.gene_index_path}...")
                self._download_gene_index()
            else:
                raise ValueError(
                    f"Gene index file not found at {self.gene_index_path}. "
                    "Set auto_download=True to download automatically."
                )
        elif self.gene_index_path is None:
            if auto_download and self.download_dir:
                self.gene_index_path = self.download_dir / "OS_scRNA_gene_index.19264.tsv"
                if not self.gene_index_path.exists():
                    logger.info(f"Downloading gene index to {self.gene_index_path}...")
                    self._download_gene_index()
            else:
                raise ValueError(
                    "Gene index file not found. Please provide gene_index_path or "
                    "set auto_download=True."
                )

        # Load gene list
        self.gene_list = self._load_gene_list()

        # Model will be loaded lazily in embed()
        self._model = None
        self._config = None

    def _model_exists(self) -> bool:
        """Check if model checkpoint file exists."""
        return self.model_path.exists() and self.model_path.is_file()

    def _download_gene_index(self) -> None:
        """Download gene index file from GitHub."""
        try:
            import requests
        except ImportError:
            raise ImportError(
                "requests is required for automatic gene index download. "
                "Install with: pip install requests"
            )

        # Create directory if it doesn't exist
        self.gene_index_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading gene index from {SCFOUNDATION_GENE_INDEX_URL}...")

        try:
            response = requests.get(SCFOUNDATION_GENE_INDEX_URL, stream=True)
            response.raise_for_status()

            # Save the file
            with open(self.gene_index_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            logger.info(f"Gene index downloaded successfully to {self.gene_index_path}")

            # Verify the file
            gene_list_df = pd.read_csv(str(self.gene_index_path), header=0, delimiter="\t")
            if len(gene_list_df) != 19264:
                logger.warning(
                    f"Gene index file has {len(gene_list_df)} genes, expected 19264. "
                    "File may be incomplete."
                )
        except Exception as e:
            # Remove partial file if download failed
            if self.gene_index_path.exists():
                self.gene_index_path.unlink()
            raise RuntimeError(
                f"Failed to download gene index file: {e}\n"
                f"You can manually download from: {SCFOUNDATION_GENE_INDEX_URL}\n"
                f"Or set gene_index_path to point to an existing file."
            ) from e

    def _load_gene_list(self) -> list[str]:
        """Load the gene list from the index file."""
        if self.gene_index_path is None or not self.gene_index_path.exists():
            raise ValueError(
                f"Gene index file not found at {self.gene_index_path}. "
                "Please provide a valid path to OS_scRNA_gene_index.19264.tsv"
            )

        # Convert Path to string for pandas
        gene_list_df = pd.read_csv(str(self.gene_index_path), header=0, delimiter="\t")
        return list(gene_list_df["gene_name"])

    def _main_gene_selection(
        self, X_df: pd.DataFrame, gene_list: list[str]
    ) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
        """
        Rebuild input data to select target genes.

        Args:
            X_df: Input data with cells in rows and genes in columns
            gene_list: Target gene list (19264 genes)

        Returns:
            Tuple of (selected_data, to_fill_columns, var_dataframe)
        """
        to_fill_columns = list(set(gene_list) - set(X_df.columns))
        padding_df = pd.DataFrame(
            np.zeros((X_df.shape[0], len(to_fill_columns))),
            columns=to_fill_columns,
            index=X_df.index,
        )
        X_df = pd.DataFrame(
            np.concatenate([df.values for df in [X_df, padding_df]], axis=1),
            index=X_df.index,
            columns=list(X_df.columns) + list(padding_df.columns),
        )
        X_df = X_df[gene_list]

        var = pd.DataFrame(index=X_df.columns)
        var["mask"] = [1 if i in to_fill_columns else 0 for i in list(var.index)]
        return X_df, to_fill_columns, var

    def _get_gene_symbols(self, adata: sc.AnnData) -> list[str]:
        """
        Extract gene symbols from AnnData.

        Checks in order:
        1. var['gene_symbol'] column (preferred, singular)
        2. var['gene_symbols'] column (plural, backward compatibility)
        3. var['feature_name'] column
        4. var['gene_name'] column
        5. var.index (assumes index contains gene symbols)
        """
        if "gene_symbol" in adata.var.columns:
            return adata.var["gene_symbol"].tolist()

        if "gene_symbols" in adata.var.columns:
            return adata.var["gene_symbols"].tolist()

        if "feature_name" in adata.var.columns:
            return adata.var["feature_name"].tolist()

        if "gene_name" in adata.var.columns:
            return adata.var["gene_name"].tolist()

        if adata.var_names is not None and len(adata.var_names) > 0:
            return adata.var_names.tolist()

        raise ValueError(
            "Cannot determine gene symbols for scFoundation. "
            "Please ensure one of the following:\n"
            "  - var['gene_symbol'] column exists (preferred)\n"
            "  - var['gene_symbols'] column exists\n"
            "  - var.index contains gene symbols\n"
            "  - var['feature_name'] column exists\n"
            "  - var['gene_name'] column exists"
        )

    def preprocess(self, adata: sc.AnnData, output_path: Optional[Path] = None) -> sc.AnnData:
        """
        Preprocess data for scFoundation.

        Requirements:
        - Gene identifiers should be gene symbols accessible via:
          * ``var['gene_symbol']`` (preferred, singular)
          * ``var['gene_symbols']`` (plural, backward compatibility)
          * ``var['feature_name']``
          * ``var['gene_name']``
          * ``var.index`` (fallback)
        - Data is expected to contain (raw or normalized) counts in ``adata.X``.

        This method ensures that the data matches the 19264 gene list required by scFoundation.

        For ``input_type="bulk"`` with ``pre_normalized="F"``, applies Scanpy
        ``normalize_total`` then ``log1p`` on the aligned matrix, matching
        `get_embedding.py` in the upstream repo.
        """
        adata = adata.copy()

        # Convert to DataFrame for gene selection
        gene_symbols = self._get_gene_symbols(adata)

        # Create DataFrame with cells as rows and genes as columns
        if hasattr(adata.X, "toarray"):
            X_df = pd.DataFrame(adata.X.toarray(), index=adata.obs_names, columns=gene_symbols)
        else:
            X_df = pd.DataFrame(adata.X, index=adata.obs_names, columns=gene_symbols)

        # Match genes to scFoundation's 19264 gene list (reorder, select, pad as needed)
        logger.info("Converting gene features to match scFoundation's 19264 gene list")
        X_df, to_fill_columns, var = self._main_gene_selection(X_df, self.gene_list)
        assert X_df.shape[1] == len(self.gene_list) == 19264

        # Update AnnData with selected genes
        adata = sc.AnnData(X_df.values, obs=adata.obs.copy(), var=var)
        adata.var_names = X_df.columns.tolist()

        if self.input_type == "bulk" and self.pre_normalized == "F":
            logger.info(
                "Bulk + pre_normalized=F: applying scanpy normalize_total and log1p "
                "(scFoundation get_embedding.py convention)"
            )
            sc.pp.normalize_total(adata, inplace=True)
            sc.pp.log1p(adata, inplace=True)

        if output_path:
            adata.write(output_path)

        return adata

    def _load_model(self) -> tuple[Any, dict[str, Any]]:
        """Load the scFoundation model and config."""
        if self._model is not None:
            return self._model, self._config

        # Determine the key based on output_type and version
        if self.output_type == "cell":
            if self.version == "ce":
                key = "cell"
            elif self.version == "rde":
                key = "rde"
            else:
                raise ValueError(f"Invalid version '{self.version}' for output_type='cell'")
        elif self.output_type == "gene":
            key = "gene"
        elif self.output_type == "gene_batch":
            key = "gene"
        else:
            raise ValueError(
                f"output_type must be one of 'cell', 'gene', 'gene_batch', "
                f"got '{self.output_type}'"
            )

        logger.info(f"Loading scFoundation model from {self.model_path} (key={key})...")

        # Convert Path to string for load_model_frommmf
        model_path_str = str(self.model_path)
        pretrainmodel, pretrainconfig = load_model_frommmf(model_path_str, key=key)
        pretrainmodel.eval()
        pretrainmodel.to(self.device)

        self._model = pretrainmodel
        self._config = pretrainconfig

        return self._model, self._config

    def embed(
        self,
        adata: sc.AnnData,
        output_path: Path,
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> sc.AnnData:
        """
        Generate scFoundation embeddings.

        Args:
            adata: Preprocessed AnnData object (must match 19264 gene list).
            output_path: Path where embeddings will be saved.
            batch_size: Number of cells per GPU batch (default: 64).
            **kwargs: Additional keyword arguments (unused for now).

        Returns:
            AnnData object with embeddings in X and original obs preserved.
        """
        import torch
        from tqdm import tqdm

        pretrainmodel, pretrainconfig = self._load_model()
        device = self.device

        # Use NumPy directly (avoid pandas)
        if hasattr(adata.X, "toarray"):
            X = adata.X.toarray()
        else:
            X = np.asarray(adata.X)

        num_cells, num_genes = X.shape
        if num_genes != 19264:
            raise ValueError(f"Data must have 19264 genes after preprocessing, got {num_genes}.")

        if batch_size is None:
            batch_size = 64 if self.output_type == "cell" else 1
        batch_size = int(batch_size)

        logger.info(
            f"Extracting scFoundation embeddings "
            f"(input_type={self.input_type}, output_type={self.output_type}, "
            f"version={self.version}, device={device}, batch_size={batch_size})..."
        )

        # Pre-create gene id tensor once
        gene_ids = torch.arange(19266, dtype=torch.long, device=device).unsqueeze(0)

        embeddings_out = []

        with torch.no_grad():
            for start in tqdm(range(0, num_cells, batch_size)):
                end = min(start + batch_size, num_cells)
                batch = X[start:end]  # (B, 19264)

                # -----------------------
                # Build input tensor (match scFoundation get_embedding.py)
                # -----------------------
                if self.input_type == "bulk":
                    batch_proc = np.asarray(batch, dtype=np.float32)
                    row_sums = batch_proc.sum(axis=1, keepdims=True)
                    if self.pre_normalized == "T":
                        total_token = row_sums
                    elif self.pre_normalized == "F":
                        total_token = np.log10(row_sums)
                    else:
                        raise ValueError(
                            f"bulk pre_normalized must be T or F, got '{self.pre_normalized}'"
                        )
                    batch_full = np.concatenate([batch_proc, total_token, total_token], axis=1)
                elif self.pre_normalized == "F":
                    totalcounts = batch.sum(axis=1, keepdims=True)
                    batch_proc = np.log1p(batch / totalcounts * 1e4)
                    if self.tgthighres[0] == "f":
                        hr = np.log10(totalcounts * float(self.tgthighres[1:]))
                        batch_full = np.concatenate(
                            [batch_proc, hr, np.log10(totalcounts)], axis=1
                        )
                    elif self.tgthighres[0] == "a":
                        hr = np.log10(totalcounts) + float(self.tgthighres[1:])
                        batch_full = np.concatenate(
                            [batch_proc, hr, np.log10(totalcounts)], axis=1
                        )
                    elif self.tgthighres[0] == "t":
                        hr = np.full_like(totalcounts, float(self.tgthighres[1:]))
                        batch_full = np.concatenate(
                            [batch_proc, hr, np.log10(totalcounts)], axis=1
                        )
                    else:
                        raise ValueError(
                            f"tgthighres must start with f, a or t, got '{self.tgthighres}'"
                        )
                elif self.pre_normalized == "T":
                    batch_proc = batch
                    totalcounts = batch.sum(axis=1, keepdims=True)
                    if self.tgthighres[0] == "f":
                        hr = np.log10(totalcounts * float(self.tgthighres[1:]))
                        batch_full = np.concatenate(
                            [batch_proc, hr, np.log10(totalcounts)], axis=1
                        )
                    elif self.tgthighres[0] == "a":
                        hr = np.log10(totalcounts) + float(self.tgthighres[1:])
                        batch_full = np.concatenate(
                            [batch_proc, hr, np.log10(totalcounts)], axis=1
                        )
                    elif self.tgthighres[0] == "t":
                        hr = np.full_like(totalcounts, float(self.tgthighres[1:]))
                        batch_full = np.concatenate(
                            [batch_proc, hr, np.log10(totalcounts)], axis=1
                        )
                    else:
                        raise ValueError(
                            f"tgthighres must start with f, a or t, got '{self.tgthighres}'"
                        )
                elif self.pre_normalized == "A":
                    batch_proc = batch[:, :-1]
                    totalcounts = batch[:, -1:]
                    if self.tgthighres[0] == "f":
                        hr = np.log10(totalcounts * float(self.tgthighres[1:]))
                        batch_full = np.concatenate(
                            [batch_proc, hr, np.log10(totalcounts)], axis=1
                        )
                    elif self.tgthighres[0] == "a":
                        hr = np.log10(totalcounts) + float(self.tgthighres[1:])
                        batch_full = np.concatenate(
                            [batch_proc, hr, np.log10(totalcounts)], axis=1
                        )
                    elif self.tgthighres[0] == "t":
                        hr = np.full_like(totalcounts, float(self.tgthighres[1:]))
                        batch_full = np.concatenate(
                            [batch_proc, hr, np.log10(totalcounts)], axis=1
                        )
                    else:
                        raise ValueError(
                            f"tgthighres must start with f, a or t, got '{self.tgthighres}'"
                        )
                else:
                    raise ValueError(
                        f"pre_normalized must be T, F or A, got '{self.pre_normalized}'"
                    )

                batch_full = torch.from_numpy(batch_full).float().to(device)  # (B, 19266)
                batch_gene_ids = gene_ids.expand(batch_full.shape[0], -1)

                # -----------------------
                # scFoundation forward
                # -----------------------
                value_labels = batch_full > 0
                x, x_padding = gatherData(batch_full, value_labels, pretrainconfig["pad_token_id"])
                pos_ids, _ = gatherData(
                    batch_gene_ids, value_labels, pretrainconfig["pad_token_id"]
                )

                x = pretrainmodel.token_emb(torch.unsqueeze(x, 2), output_weight=0)
                x = x + pretrainmodel.pos_emb(pos_ids)
                geneemb = pretrainmodel.encoder(x, x_padding)  # (B, L, D)

                if self.output_type == "cell":
                    geneemb1 = geneemb[:, -1, :]
                    geneemb2 = geneemb[:, -2, :]
                    geneemb3, _ = torch.max(geneemb[:, :-2, :], dim=1)
                    geneemb4 = torch.mean(geneemb[:, :-2, :], dim=1)

                    if self.pool_type == "all":
                        cell_emb = torch.cat([geneemb1, geneemb2, geneemb3, geneemb4], dim=1)
                    elif self.pool_type == "max":
                        cell_emb, _ = torch.max(geneemb, dim=1)
                    else:
                        raise ValueError(f"pool_type must be all or max, got '{self.pool_type}'")

                    embeddings_out.append(cell_emb.cpu().numpy())

                elif self.output_type == "gene":
                    embeddings_out.append(geneemb[:, :19264, :].cpu().numpy())

                else:
                    raise ValueError(
                        f"output_type must be one of 'cell', 'gene', 'gene_batch', "
                        f"got '{self.output_type}'"
                    )

                # Explicit cleanup helps fragmentation in long runs
                del batch_full, batch_gene_ids, value_labels, x, x_padding, pos_ids, geneemb
                if device == "cuda":
                    torch.cuda.synchronize()

        embeddings = np.concatenate(embeddings_out, axis=0)
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")

        result_adata = sc.AnnData(X=embeddings, obs=adata.obs.copy())
        self.validate_embeddings(result_adata)
        return result_adata

    def compute_sensitivity(
        self,
        adata: sc.AnnData,
        output_path: Path,
        batch_size: Optional[int] = None,
        n_cells: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """
        Compute ∂(cell_embedding)/∂(input_token_embeddings), SVD per cell; write to output_path.

        scFoundation uses custom pooling: last position, second-to-last, max over rest, mean over rest
        (pool_type "all"), or max over all positions. J shape (d_emb, seq_len*d_token). Top k=50
        SVD kept; full Jacobian discarded. Use CLI --chunk-size to process in chunks.
        Passes through input adata.obs and adds seq_length. Writes obsm['X_baseline'], obsm['jacobian_U']
        (n_cells, d_emb, 50) float16, obsm['jacobian_S'] (n_cells, 50) float32.
        """
        from scipy.sparse.linalg import svds
        import torch

        if self.input_type == "bulk":
            raise NotImplementedError(
                "Sensitivity analysis is only implemented for input_type='singlecell'."
            )

        if self.output_type != "cell":
            raise NotImplementedError(
                f"sensitivity analysis only supported for output_type='cell', got '{self.output_type}'"
            )

        if n_cells is not None:
            adata = adata[:n_cells].copy()

        pretrainmodel, pretrainconfig = self._load_model()
        device = self.device

        if hasattr(adata.X, "toarray"):
            X = adata.X.toarray()
        else:
            X = np.asarray(adata.X)

        num_cells, num_genes = X.shape
        if num_genes != 19264:
            raise ValueError(f"Data must have 19264 genes after preprocessing, got {num_genes}.")

        gene_ids = torch.arange(19266, dtype=torch.long, device=device).unsqueeze(0)

        logger.info(
            "Sensitivity analysis: %d cells, %d genes, device=%s",
            num_cells,
            num_genes,
            device,
        )

        baselines_out = []
        U_out = []
        S_out = []
        seqlen_out = []

        from tqdm import tqdm

        for idx in tqdm(
            range(num_cells),
            desc="scFoundation sensitivity",
            unit="cell",
            leave=True,
        ):
            batch = X[idx : idx + 1]
            batch_full = self._batch_to_tensor(batch, device)
            batch_gene_ids = gene_ids.expand(1, -1)
            value_labels = batch_full > 0
            x, x_padding = gatherData(batch_full, value_labels, pretrainconfig["pad_token_id"])
            pos_ids, _ = gatherData(batch_gene_ids, value_labels, pretrainconfig["pad_token_id"])

            x_emb = pretrainmodel.token_emb(torch.unsqueeze(x, 2), output_weight=0)
            x_emb = x_emb.detach().requires_grad_(True)
            x_in = x_emb + pretrainmodel.pos_emb(pos_ids)
            geneemb = pretrainmodel.encoder(x_in, x_padding)

            geneemb1 = geneemb[:, -1, :]
            geneemb2 = geneemb[:, -2, :]
            geneemb3, _ = torch.max(geneemb[:, :-2, :], dim=1)
            geneemb4 = torch.mean(geneemb[:, :-2, :], dim=1)
            if self.pool_type == "all":
                cell_emb = torch.cat([geneemb1, geneemb2, geneemb3, geneemb4], dim=1)
            else:
                cell_emb, _ = torch.max(geneemb, dim=1)

            d_emb = cell_emb.shape[1]
            k = 50
            seq_len = x_emb.shape[1]
            d_in = x_emb.shape[2]
            d_input = seq_len * d_in
            J = np.zeros((d_emb, d_input), dtype=np.float64)
            for i in range(d_emb):
                if x_emb.grad is not None:
                    x_emb.grad.zero_()
                cell_emb[0, i].backward(retain_graph=(i < d_emb - 1))
                J[i, :] = x_emb.grad[0].detach().cpu().numpy().flatten()

            baseline = cell_emb[0].detach().cpu().numpy()
            k_svd = min(k, min(J.shape) - 1)
            if k_svd < 1:
                U = np.zeros((d_emb, k), dtype=np.float16)
                S = np.zeros(k, dtype=np.float32)
            else:
                U, S, _ = svds(J.astype(np.float64), k=k_svd)
                U = np.flip(U, axis=1).copy().astype(np.float16)
                S = np.flip(S).copy().astype(np.float32)
                if U.shape[1] < k:
                    U_pad = np.zeros((d_emb, k), dtype=np.float16)
                    U_pad[:, : U.shape[1]] = U
                    S_pad = np.zeros(k, dtype=np.float32)
                    S_pad[: S.shape[0]] = S
                    U, S = U_pad, S_pad

            baselines_out.append(baseline)
            U_out.append(U)
            S_out.append(S)
            seqlen_out.append(seq_len)

        baselines_arr = np.array(baselines_out, dtype=np.float32)
        jacobian_U = np.array(U_out, dtype=np.float16)
        jacobian_S = np.array(S_out, dtype=np.float32)
        seq_lengths_arr = np.array(seqlen_out)

        obs_out = adata.obs.iloc[:num_cells].copy()
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
        logger.info("Wrote sensitivity results to %s (%d cells)", output_path, num_cells)

    def _batch_to_tensor(self, batch: np.ndarray, device: str) -> Any:
        """Convert a batch (B, 19264) to batch_full (B, 19266) tensor."""
        import torch

        if self.pre_normalized == "F":
            totalcounts = batch.sum(axis=1, keepdims=True)
            batch_proc = np.log1p(batch / totalcounts * 1e4)
        elif self.pre_normalized == "T":
            batch_proc = batch
            totalcounts = batch.sum(axis=1, keepdims=True)
        elif self.pre_normalized == "A":
            batch_proc = batch[:, :-1]
            totalcounts = batch[:, -1:]
        else:
            raise ValueError(f"pre_normalized must be T, F or A, got '{self.pre_normalized}'")

        if self.tgthighres[0] == "f":
            hr = np.log10(totalcounts * float(self.tgthighres[1:]))
            batch_full = np.concatenate([batch_proc, hr, np.log10(totalcounts)], axis=1)
        elif self.tgthighres[0] == "a":
            hr = np.log10(totalcounts) + float(self.tgthighres[1:])
            batch_full = np.concatenate([batch_proc, hr, np.log10(totalcounts)], axis=1)
        elif self.tgthighres[0] == "t":
            hr = np.full_like(totalcounts, float(self.tgthighres[1:]))
            batch_full = np.concatenate([batch_proc, hr, np.log10(totalcounts)], axis=1)
        else:
            raise ValueError(f"tgthighres must start with f, a or t, got '{self.tgthighres}'")
        return torch.from_numpy(batch_full).float().to(device)

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

        # Propagate core scFoundation-specific arguments
        if self.model_path:
            cmd.extend(["--model-path", str(self.model_path)])
        if self.ckpt_name:
            cmd.extend(["--ckpt-name", self.ckpt_name])
        if self.input_type:
            cmd.extend(["--input-type", self.input_type])
        if self.gene_index_path:
            cmd.extend(["--gene-index-path", str(self.gene_index_path)])
        if self.device:
            cmd.extend(["--device", self.device])
        if self.output_type:
            cmd.extend(["--output-type", self.output_type])
        if self.pool_type:
            cmd.extend(["--pool-type", self.pool_type])
        if self.tgthighres:
            cmd.extend(["--tgthighres", self.tgthighres])
        if self.pre_normalized:
            cmd.extend(["--pre-normalized", self.pre_normalized])
        if self.version:
            cmd.extend(["--version", self.version])
        if self.auto_download:
            cmd.extend(["--auto-download", "true"])
        if self.download_dir:
            cmd.extend(["--download-dir", str(self.download_dir)])

        # Additional kwargs as CLI flags
        for k, v in kwargs.items():
            if v is not None:
                cmd.extend([f"--{k.replace('_', '-')}", str(v)])

        return cmd

    def get_required_dependencies(self) -> list[str]:
        """Get required dependencies for scFoundation."""
        return [
            "torch",
            "numpy",
            "pandas",
            "scipy",
            "scanpy",
            "einops",
            "tqdm",
            "requests",
        ]

    def get_optional_dependency_group(self) -> Optional[str]:
        """Get optional dependency group name for scFoundation."""
        return "scfoundation"

    def get_container_name(self) -> Optional[str]:
        """Get container name for scFoundation."""
        return "scfoundation"
