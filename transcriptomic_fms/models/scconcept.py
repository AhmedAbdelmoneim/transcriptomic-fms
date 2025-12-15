"""scConcept embedding model."""

from pathlib import Path
from typing import Any, Optional

import numpy as np
import scanpy as sc

from transcriptomic_fms.models.base import BaseEmbeddingModel
from transcriptomic_fms.models.registry import register_model
from transcriptomic_fms.utils.logging import get_logger

logger = get_logger(__name__)

try:
    # scConcept package exposes the main class as `scConcept`
    from concept import scConcept as _scConcept
except ImportError as e:  # pragma: no cover - import-time guard
    # Only treat a missing top-level `concept` module as "not installed".
    # For other ImportErrors (e.g. missing transitive deps), re-raise so users
    # see the real underlying problem instead of a generic message.
    if getattr(e, "name", None) == "concept":
        _scConcept = None
    else:
        raise


@register_model("scconcept")
class SCConceptModel(BaseEmbeddingModel):
    """scConcept contrastive pretraining model for single-cell embeddings."""

    def __init__(
        self,
        model_name: str,
        pretrained_model_name: str = "Corpus-30M",
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        requires_gpu: bool = True,
        **kwargs: Any,
    ):
        """
        Initialize scConcept model.

        Args:
            model_name: Model identifier (fixed to ``"scconcept"`` in registry).
            pretrained_model_name: Name of the pretrained model to load from HuggingFace
                (e.g. ``"Corpus-30M"``).
            cache_dir: Directory to cache downloaded models and mappings.
            device: Device hint for scConcept / PyTorch (``"cuda"`` or ``"cpu"``).
                    If ``None``, will auto-detect.
            requires_gpu: Whether this model requires a GPU (default: True).
            **kwargs: Additional, model-specific configuration (stored in ``self.config``).
        """
        super().__init__(model_name=model_name, requires_gpu=requires_gpu, **kwargs)

        if _scConcept is None:
            raise ImportError(
                "scConcept is not installed. Install with:\n"
                "  make install-model MODEL=scconcept\n"
                "  or: uv sync --extra scconcept\n"
                "  or: pip install "
                "concept git+https://github.com/theislab/lamin_dataloader.git "
                "'flash-attn==2.7.*'\n"
                "Note: flash-attn requires a working CUDA toolchain and matching PyTorch build."
            )

        self.pretrained_model_name = pretrained_model_name
        self.cache_dir = cache_dir

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

    def _get_gene_ids(self, adata: sc.AnnData) -> list[str]:
        """
        Extract gene IDs suitable for scConcept.

        scConcept expects an Ensembl-style gene identifier column, typically
        provided via ``adata.var['gene_id']``.

        Checks in order:
        1. ``var['gene_id']`` column
        2. ``var['ensembl_id']`` column
        3. ``var.index`` (assumes index contains gene IDs)
        """
        if "gene_id" in adata.var.columns:
            return adata.var["gene_id"].tolist()

        if "ensembl_id" in adata.var.columns:
            return adata.var["ensembl_id"].tolist()

        if adata.var_names is not None and len(adata.var_names) > 0:
            return adata.var_names.tolist()

        raise ValueError(
            "Cannot determine gene IDs for scConcept. "
            "Please ensure one of the following:\n"
            "  - var.index contains Ensembl-style gene IDs\n"
            "  - var['gene_id'] column exists\n"
            "  - var['ensembl_id'] column exists"
        )

    def preprocess(self, adata: sc.AnnData, output_path: Optional[Path] = None) -> sc.AnnData:
        """
        Preprocess data for scConcept.

        Requirements:
        - Gene identifiers should be Ensembl IDs accessible via:
          * ``var['gene_id']`` (preferred)
          * ``var['ensembl_id']``
          * ``var.index`` (fallback)
        - Data is expected to contain (raw or normalized) counts in ``adata.X``.

        This method ensures that ``adata.var['gene_id']`` is populated, which is
        what scConcept uses via the ``gene_id_column`` argument.
        """
        adata = adata.copy()

        gene_ids = self._get_gene_ids(adata)
        adata.var["gene_id"] = gene_ids

        if output_path:
            adata.write(output_path)

        return adata

    def embed(
        self,
        adata: sc.AnnData,
        output_path: Path,
        batch_size: Optional[int] = None,
        gene_id_column: str = "gene_id",
        **kwargs: Any,
    ) -> sc.AnnData:
        """
        Generate scConcept embeddings.

        This wraps ``concept.scConcept.extract_embeddings`` and returns an
        AnnData with embeddings in ``X`` and the original ``obs`` preserved.

        Args:
            adata: Preprocessed AnnData object (must contain ``gene_id_column`` in ``var``).
            output_path: Path where embeddings will be saved (not used directly here,
                         but kept for API consistency).
            batch_size: Unused for now (scConcept controls batching internally),
                        kept for API compatibility.
            gene_id_column: Column in ``adata.var`` that contains gene identifiers
                            for scConcept (default: ``"gene_id"``).
            **kwargs: Additional keyword arguments forwarded to
                      ``scConcept.extract_embeddings``.
        """
        # Ensure required gene ID column exists
        if gene_id_column not in adata.var.columns:
            raise ValueError(
                f"Required gene ID column '{gene_id_column}' not found in adata.var.\n"
                "Run preprocess() first, or provide the correct column name via "
                "`--gene-id-column`."
            )

        # Instantiate scConcept object
        concept_kwargs: dict[str, Any] = {}
        if self.cache_dir is not None:
            concept_kwargs["cache_dir"] = self.cache_dir

        # Some versions of scConcept accept a `device` argument in the constructor;
        # if that fails we silently fall back to the default behaviour.
        try:
            concept_kwargs["device"] = self.device
            concept_model = _scConcept(**concept_kwargs)
        except TypeError:
            concept_kwargs.pop("device", None)
            concept_model = _scConcept(**concept_kwargs)

        # Load pretrained model (from HuggingFace or local config)
        # By default we follow the README's recommended interface:
        #   concept.load_config_and_model(model_name='Corpus-30M')
        concept_model.load_config_and_model(model_name=self.pretrained_model_name)

        logger.info(
            "Extracting scConcept embeddings "
            f"(model={self.pretrained_model_name}, device={self.device})..."
        )

        result = concept_model.extract_embeddings(
            adata=adata,
            gene_id_column=gene_id_column,
            **kwargs,
        )

        # scConcept returns a dict-like object with keys such as 'cls_cell_emb'
        if "cls_cell_emb" not in result:
            raise ValueError(
                "scConcept did not return 'cls_cell_emb' in the result. "
                f"Available keys: {list(result.keys())}"
            )

        embeddings = result["cls_cell_emb"]

        # Convert to numpy if needed (torch.Tensor or other array-like)
        try:
            import torch

            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.detach().cpu().numpy()
        except ImportError:
            # If torch is not available here, just rely on numpy conversion
            pass

        if not isinstance(embeddings, np.ndarray):
            embeddings = np.asarray(embeddings)

        # Build output AnnData with embeddings and preserved obs
        result_adata = sc.AnnData(
            X=embeddings,
            obs=adata.obs.copy(),
        )

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

        # Propagate core scConcept-specific arguments
        if self.pretrained_model_name:
            cmd.extend(["--pretrained-model-name", self.pretrained_model_name])
        if self.cache_dir:
            cmd.extend(["--cache-dir", self.cache_dir])
        if self.device:
            cmd.extend(["--device", self.device])

        # Additional kwargs (e.g. gene_id_column) as CLI flags
        for k, v in kwargs.items():
            if v is not None:
                cmd.extend([f"--{k.replace('_', '-')}", str(v)])

        return cmd

    def get_required_dependencies(self) -> list[str]:
        """Get required dependencies for scConcept."""
        return [
            "scconcept",
            "lamin_dataloader",
            "flash-attn==2.7.*",
        ]

    def get_optional_dependency_group(self) -> Optional[str]:
        """Get optional dependency group name for scConcept."""
        return "scconcept"

    def get_container_name(self) -> Optional[str]:
        """Get container name for scConcept."""
        return "scconcept"
