"""Ensembl gene ID helpers shared by model adapters and validation."""

from __future__ import annotations

import re
from typing import Iterable

_ENSEMBL_RE = re.compile(r"^ENS[A-Z]*G\d+(?:\.\d+)?$", re.IGNORECASE)
_SPECIAL_TOKEN_RE = re.compile(r"^<.*>$")


def looks_like_ensembl_id(value: object) -> bool:
    """Return True when ``value`` matches a core Ensembl gene ID pattern."""
    return bool(_ENSEMBL_RE.match(str(value).strip()))


def normalize_ensembl_id(value: object) -> str:
    """Strip Ensembl version suffix (e.g. ``ENSG....15`` -> ``ENSG...``)."""
    text = str(value).strip()
    if not text or text.lower() == "nan" or _SPECIAL_TOKEN_RE.match(text):
        return text
    if looks_like_ensembl_id(text) and "." in text:
        return text.split(".", maxsplit=1)[0]
    return text


def normalize_ensembl_ids(values: Iterable[object]) -> list[str]:
    """Normalize a sequence of Ensembl gene identifiers."""
    return [normalize_ensembl_id(value) for value in values]


def normalize_ensembl_set(values: Iterable[object]) -> set[str]:
    """Normalize Ensembl identifiers for set overlap checks."""
    return {
        normalize_ensembl_id(gene)
        for gene in values
        if str(gene).strip()
        and str(gene).strip().lower() != "nan"
        and not _SPECIAL_TOKEN_RE.match(str(gene).strip())
    }
