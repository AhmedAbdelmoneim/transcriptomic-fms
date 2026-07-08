"""Tests for scFoundation gene-symbol normalization."""

from __future__ import annotations

import unittest

import anndata as ad
import numpy as np
import scipy.sparse as sp

from transcriptomic_fms.models.scfoundation import SCFoundationModel


class TestSCFoundationGeneMatching(unittest.TestCase):
    def test_build_expression_dataframe_uppercases_symbols(self) -> None:
        model = object.__new__(SCFoundationModel)
        adata = ad.AnnData(
            X=sp.csr_matrix([[1.0, 2.0, 3.0]]),
            var={"gene_name": ["Xkr4", "Sox17", "Gapdh"]},
        )
        adata.var_names = ["Xkr4", "Sox17", "Gapdh"]

        frame = model._build_expression_dataframe(adata)

        self.assertEqual(list(frame.columns), ["XKR4", "SOX17", "GAPDH"])
        self.assertEqual(frame.loc[adata.obs_names[0], "XKR4"], 1.0)

    def test_main_gene_selection_matches_uppercased_symbols(self) -> None:
        model = object.__new__(SCFoundationModel)
        adata = ad.AnnData(
            X=sp.csr_matrix([[4.0, 0.0], [0.0, 8.0]]),
            var={"gene_name": ["Xkr4", "Sox17"]},
        )

        frame = model._build_expression_dataframe(adata)
        selected, to_fill, _var = model._main_gene_selection(
            frame,
            ["XKR4", "SOX17", "GAPDH"],
        )

        self.assertEqual(list(selected.columns), ["XKR4", "SOX17", "GAPDH"])
        self.assertEqual(to_fill, ["GAPDH"])
        self.assertEqual(float(selected.loc[adata.obs_names[0], "XKR4"]), 4.0)
        self.assertEqual(float(selected.loc[adata.obs_names[1], "SOX17"]), 8.0)

    def test_build_expression_dataframe_aggregates_duplicate_symbols(self) -> None:
        model = object.__new__(SCFoundationModel)
        adata = ad.AnnData(
            X=sp.csr_matrix([[1.0, 2.0], [3.0, 4.0]]),
            var={"gene_name": ["Xkr4", "xkr4"]},
        )

        frame = model._build_expression_dataframe(adata)

        self.assertEqual(list(frame.columns), ["XKR4"])
        np.testing.assert_allclose(frame["XKR4"].to_numpy(), [3.0, 7.0])


if __name__ == "__main__":
    unittest.main()
