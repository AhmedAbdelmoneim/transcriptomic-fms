"""Tests for scGPT gene-symbol normalization."""

from __future__ import annotations

import unittest

import anndata as ad
import scipy.sparse as sp

from transcriptomic_fms.models.scgpt import SCGPTModel


class TestSCGPTGeneMatching(unittest.TestCase):
    def test_normalize_gene_symbols_column_uppercases(self) -> None:
        model = object.__new__(SCGPTModel)
        adata = ad.AnnData(
            X=sp.csr_matrix([[1.0, 2.0, 3.0]]),
            var={"gene_name": ["Pf4", "S100a8", "Gapdh"]},
        )

        out = model._normalize_gene_symbols_column(adata)

        self.assertEqual(list(out.var["gene_symbols"]), ["PF4", "S100A8", "GAPDH"])

    def test_normalize_gene_symbols_column_reuses_existing_column(self) -> None:
        model = object.__new__(SCGPTModel)
        adata = ad.AnnData(
            X=sp.csr_matrix([[1.0, 2.0]]),
            var={"gene_symbols": ["Pf4", "S100a8"]},
        )

        out = model._normalize_gene_symbols_column(adata)

        self.assertEqual(list(out.var["gene_symbols"]), ["PF4", "S100A8"])

    def test_resolve_hvg_columns_case_insensitive(self) -> None:
        model = object.__new__(SCGPTModel)
        adata = ad.AnnData(
            X=sp.csr_matrix([[1.0, 2.0, 3.0]]),
            var={"gene_name": ["Pf4", "S100a8", "Gapdh"]},
        )
        adata.var_names = ["Pf4", "S100a8", "Gapdh"]

        resolved = model._resolve_hvg_columns(adata, ["PF4", "s100a8", "Missing"])

        self.assertEqual(resolved, ["Pf4", "S100a8"])


if __name__ == "__main__":
    unittest.main()
