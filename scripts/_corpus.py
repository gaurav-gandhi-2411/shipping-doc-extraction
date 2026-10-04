"""Shared train+dev corpus loader for the provenance and ceiling scripts (labels + groups)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shipdoc import meta, ocr
from shipdoc.eval import load_scorer

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class DocRec:
    """One train/dev doc: gold label, supplier group, scanned flag."""

    doc_id: str
    split: str
    gold: dict[str, Any]
    group: str
    scanned: bool

    @property
    def doc_type(self) -> str:
        """invoice or waybill."""
        return str(self.gold["doc_type"])

    def pages(self) -> list[ocr.PageOcr]:
        """Cached OCR pages in order."""
        return ocr.doc_pages(self.doc_id)


def is_empty(v: Any) -> bool:
    """None or blank string."""
    return v is None or (isinstance(v, str) and not v.strip())


def load_corpus() -> list[DocRec]:
    """All train+dev docs, sorted by doc_id within split (train first)."""
    groups = json.loads((ROOT / "meta" / "supplier_groups.json").read_text(encoding="utf-8"))
    out: list[DocRec] = []
    for split in ("train", "dev"):
        tags = {m["doc_id"]: m for m in json.loads((ROOT / "meta" / f"{split}.json").read_text())}
        for g in meta.load_labels(split):
            d = g["doc_id"]
            out.append(DocRec(d, split, g, groups[d], bool(tags[d]["scanned"])))
    return out


def header_fields(doc_type: str) -> list[str]:
    """The scorer's header fields for a doc type."""
    return list(load_scorer().HEADER[doc_type])


def row_fields() -> list[str]:
    """The scorer's row fields."""
    return list(load_scorer().ROW)
