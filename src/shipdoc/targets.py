"""Gold document -> per-page target dicts / text in the CURRENT output format (spec Phase 4.1).

Used by the token-budget measurement (scripts/token_budget.py) and the compact-format round-trip
tests. Gold does not record which page a row is on: callers pass ``row_pages`` (from
``<SHIPDOC_RUNS_DIR>/provenance/locations.jsonl``) or accept an even split.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from shipdoc.extract import HEADER_KEYS, ROW_KEYS

#: Header keys read from the LAST page (totals); every other header key sits on page 1. Mirrors
#: `MockBackend.TOTALS` and `shipdoc.merge.TOTAL_FIELDS` (field provenance, reports/provenance.md).
LAST_PAGE_FIELDS: dict[str, tuple[str, ...]] = {
    "invoice": ("total_amount",),
    "waybill": ("pieces", "gross_weight_kg"),
}


def page_kind_of(page_index: int, n_pages: int) -> str:
    """``single`` / ``first`` / ``continuation`` for a 0-based page of an `n_pages`-page doc."""
    if n_pages == 1:
        return "single"
    return "first" if page_index == 0 else "continuation"


def gold_page_payloads(
    gold: dict[str, Any], n_pages: int, row_pages: Sequence[int] | None = None
) -> list[dict[str, Any]]:
    """Gold document -> the per-page target dicts in the CURRENT output format (`page_schema`).

    Identity header fields go on page 1, totals on the last page, the other header keys are null.
    ``row_pages[i]`` is the 0-based page of gold row i (clamped into range); when None the rows
    are split into contiguous chunks across pages like `MockBackend`. Row order inside a page
    follows the gold order.
    """
    n_pages = max(1, n_pages)
    doc_type = gold["doc_type"]
    totals = LAST_PAGE_FIELDS[doc_type]
    rows = gold.get("line_items") or []
    if row_pages is None:
        base, extra = divmod(len(rows), n_pages)
        row_pages = [p for p in range(n_pages) for _ in range(base + (1 if p < extra else 0))]
    out = []
    for p in range(n_pages):
        header: dict[str, str | None] = dict.fromkeys(HEADER_KEYS)
        for k, v in gold["header"].items():
            if (p == n_pages - 1) if k in totals else (p == 0):
                header[k] = v
        items = [
            {k: r.get(k) for k in ROW_KEYS}
            for r, rp in zip(rows, row_pages, strict=True)
            if min(max(rp, 0), n_pages - 1) == p
        ]
        out.append(
            {
                "doc_type": doc_type,
                "header": header,
                "line_items": items,
                "page_kind": page_kind_of(p, n_pages),
            }
        )
    return out


def render_target(payload: dict[str, Any]) -> str:
    """Page target text as the constrained decoder emits it (single line, no indent).

    xgrammar with ``any_whitespace=False`` and no indent uses the separators ``", "`` and
    ``": "`` (the same ones `MockBackend` writes); keys follow the schema's property order.
    """
    return json.dumps(payload, separators=(", ", ": "), ensure_ascii=False)
