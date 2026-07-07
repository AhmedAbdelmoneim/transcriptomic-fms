"""Tests for Ensembl ID normalization helpers."""

from __future__ import annotations

import unittest

import anndata as ad
import numpy as np
import scipy.sparse as sp

from transcriptomic_fms.models.geneformer import GeneformerModel
from transcriptomic_fms.models.scconcept import SCConceptModel
from transcriptomic_fms.utils.gene_ids import (
    normalize_ensembl_id,
    normalize_ensembl_ids,
    normalize_ensembl_set,
)


class TestNormalizeEnsemblIds(unittest.TestCase):
    def test_strips_version_suffix(self) -> None:
        self.assertEqual(normalize_ensembl_id("ENSG00000000003.15"), "ENSG00000000003")

    def test_leaves_unversioned_ids_unchanged(self) -> None:
        self.assertEqual(normalize_ensembl_id("ENSG00000000003"), "ENSG00000000003")

    def test_leaves_non_ensembl_ids_unchanged(self) -> None:
        self.assertEqual(normalize_ensembl_id("TSPAN6"), "TSPAN6")
        self.assertEqual(normalize_ensembl_id("GENE.NAME"), "GENE.NAME")

    def test_normalize_list_and_set(self) -> None:
        values = ["ENSG00000000003.15", "ENSG00000000005.6", "TSPAN6"]
        self.assertEqual(
            normalize_ensembl_ids(values),
            ["ENSG00000000003", "ENSG00000000005", "TSPAN6"],
        )
        self.assertEqual(
            normalize_ensembl_set(values),
            {"ENSG00000000003", "ENSG00000000005", "TSPAN6"},
        )


class TestModelEnsemblExtraction(unittest.TestCase):
    def _make_adata(self) -> ad.AnnData:
        x = sp.csr_matrix(np.ones((2, 3), dtype=np.float32))
        return ad.AnnData(
            X=x,
            var={
                "gene_name": ["A", "B", "C"],
                "ensembl_id": [
                    "ENSG00000000003.15",
                    "ENSG00000000005.6",
                    "ENSG00000000419.14",
                ],
                "gene_id": [
                    "ENSG00000000003.15",
                    "ENSG00000000005.6",
                    "ENSG00000000419.14",
                ],
            },
        )

    def test_geneformer_get_ensembl_ids_strips_versions(self) -> None:
        model = object.__new__(GeneformerModel)
        adata = self._make_adata()
        self.assertEqual(
            model._get_ensembl_ids(adata),
            ["ENSG00000000003", "ENSG00000000005", "ENSG00000000419"],
        )

    def test_scconcept_get_gene_ids_strips_versions(self) -> None:
        model = object.__new__(SCConceptModel)
        adata = self._make_adata()
        self.assertEqual(
            model._get_gene_ids(adata),
            ["ENSG00000000003", "ENSG00000000005", "ENSG00000000419"],
        )


if __name__ == "__main__":
    unittest.main()
