"""Tests for Geneformer tokenized-dataset validation."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from transcriptomic_fms.models.geneformer import GeneformerModel


class DummyDataset:
    def __init__(self, rows):
        self.rows = rows
        self.column_names = ["input_ids"] if rows is not None else []

    def __len__(self):
        return len(self.rows or [])

    def __getitem__(self, idx):
        return self.rows[idx]


class TestGeneformerTokenizationValidation(unittest.TestCase):
    def test_rejects_empty_dataset(self) -> None:
        model = object.__new__(GeneformerModel)
        with patch(
            "transcriptomic_fms.models.geneformer.load_from_disk",
            return_value=DummyDataset([]),
        ):
            with self.assertRaisesRegex(RuntimeError, "empty dataset"):
                model._validate_tokenized_dataset("/tmp/tokenized.dataset")

    def test_rejects_non_integer_input_ids(self) -> None:
        model = object.__new__(GeneformerModel)
        dataset = DummyDataset([{"input_ids": [1.0, 2.0]}])
        with patch(
            "transcriptomic_fms.models.geneformer.load_from_disk",
            return_value=dataset,
        ):
            with self.assertRaisesRegex(RuntimeError, "non-integer input_ids"):
                model._validate_tokenized_dataset("/tmp/tokenized.dataset")

    def test_accepts_integer_input_ids(self) -> None:
        model = object.__new__(GeneformerModel)
        dataset = DummyDataset([{"input_ids": [1, 2, 3]}])
        with patch(
            "transcriptomic_fms.models.geneformer.load_from_disk",
            return_value=dataset,
        ):
            model._validate_tokenized_dataset("/tmp/tokenized.dataset")


if __name__ == "__main__":
    unittest.main()
