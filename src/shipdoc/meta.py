"""Slice tagging and CV-fold helpers (no label values ever leave this module).

``score.py --meta`` reads only bool tags, so every tag here is a real bool; the string
``supplier_group`` is carried alongside for the extra per-group slices in ``shipdoc.eval``.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any

from shipdoc import paths

ROOT = Path(__file__).resolve().parents[2]

# Same threshold and meaning as scripts/recon.py: a header field whose pooled null rate within
# its doc type is >= this is an "optional line" (null = the line is not printed on the page).
OPTIONAL_NULL_RATE = 0.10

BOOL_TAGS = (
    "scanned",
    "multipage",
    "repeated_parts",
    "illegible",
    "waybill",
    "awb_absent",
    "hawb_absent",
)


def _empty(v: Any) -> bool:
    return v is None or (isinstance(v, str) and not v.strip())


def load_labels(split: str, root: Path | None = None) -> list[dict[str, Any]]:
    """All gold label dicts of one split, sorted by file name.

    Reads ``<data dir>/<split>/labels``; `root` (a folder holding ``data/``) overrides.
    """
    data = paths.data_dir() if root is None else root / "data"
    files = sorted(glob.glob(str(data / split / "labels" / "*.json")))
    return [json.loads(Path(f).read_text(encoding="utf-8")) for f in files]


def not_printed_fields(
    docs: list[dict[str, Any]], threshold: float = OPTIONAL_NULL_RATE
) -> dict[str, set[str]]:
    """Per doc_type, header fields whose nulls mean "line not printed" (recon item 3 rule).

    A field is not-printed (optional line) iff its pooled null rate within the doc type is
    >= ``threshold``. Every other field is required, so a null there is a redaction or
    scribble, i.e. illegible. Derived from the labels passed in (pass train+dev pooled so
    both splits are classified identically), not hard-coded field names.
    """
    by_type: dict[str, list[dict[str, Any]]] = {}
    for d in docs:
        by_type.setdefault(d["doc_type"], []).append(d)
    out: dict[str, set[str]] = {}
    for t, ds in by_type.items():
        fields = {f for d in ds for f in d["header"]}
        out[t] = {
            f for f in fields if sum(_empty(d["header"].get(f)) for d in ds) / len(ds) >= threshold
        }
    return out


def is_illegible(doc: dict[str, Any], not_printed: dict[str, set[str]]) -> bool:
    """True iff the doc has >=1 gold header null that is NOT a not-printed field."""
    skip = not_printed.get(doc["doc_type"], set())
    return any(_empty(v) and f not in skip for f, v in doc["header"].items())


def has_repeated_parts(doc: dict[str, Any]) -> bool:
    """True iff any supplier_part_number appears on >=2 rows."""
    seen: set[str] = set()
    for r in doc.get("line_items", []):
        p = r.get("supplier_part_number")
        if _empty(p):
            continue
        if p in seen:
            return True
        seen.add(p)
    return False


def is_awb_absent(doc: dict[str, Any]) -> bool:
    """True iff the doc is an invoice whose gold awb_number is null/blank (line not printed)."""
    return doc["doc_type"] == "invoice" and _empty(doc["header"].get("awb_number"))


def is_hawb_absent(doc: dict[str, Any]) -> bool:
    """True iff the doc is a waybill whose gold hawb is null/blank (line not printed)."""
    return doc["doc_type"] == "waybill" and _empty(doc["header"].get("hawb"))


def is_mixed_scan(doc: dict[str, Any]) -> bool:
    """True iff some pages are .jpg and some are not (png)."""
    flags = [p.lower().endswith(".jpg") for p in doc["pages"]]
    return any(flags) and not all(flags)


def tag_doc(doc: dict[str, Any], group: str, not_printed: dict[str, set[str]]) -> dict[str, Any]:
    """Meta record for one doc: bool tags plus the string supplier_group."""
    return {
        "doc_id": doc["doc_id"],
        "scanned": any(p.lower().endswith(".jpg") for p in doc["pages"]),
        "multipage": len(doc["pages"]) > 1,
        "repeated_parts": has_repeated_parts(doc),
        "illegible": is_illegible(doc, not_printed),
        "waybill": doc["doc_type"] == "waybill",
        "awb_absent": is_awb_absent(doc),
        "hawb_absent": is_hawb_absent(doc),
        "supplier_group": group,
    }


def fold_meta(meta: list[dict[str, Any]], folds: dict[str, Any], fold: int) -> list[dict[str, Any]]:
    """Per-doc meta with ``unseen`` = True for docs in validation fold ``fold``.

    In out-of-fold scoring every scored doc is in its own validation fold, so each is unseen
    by the model that scored it (its supplier group was held out too).
    """
    doc_fold = folds["doc_fold"]
    return [{**m, "unseen": doc_fold[m["doc_id"]] == fold} for m in meta]


def make_folds(rows: list[dict[str, Any]], k: int = 3, seed: int = 42) -> dict[str, Any]:
    """K group folds over meta rows, stratified on doc type; asserts the constraints.

    Raises AssertionError (never retries another seed) if a supplier_group spans two folds
    or a fold lacks invoices or waybills.
    """
    import numpy as np
    from sklearn.model_selection import StratifiedGroupKFold

    ids = [r["doc_id"] for r in rows]
    y = np.array(["waybill" if r["waybill"] else "invoice" for r in rows])
    g = np.array([r["supplier_group"] for r in rows])
    sgk = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = []
    for i, (_, val) in enumerate(sgk.split(np.zeros(len(ids)), y, g)):
        folds.append(
            {
                "fold": i,
                "val_doc_ids": sorted(ids[j] for j in val),
                "val_groups": sorted(set(g[val].tolist())),
                "_types": set(y[val].tolist()),
            }
        )
    for f in folds:
        assert f.pop("_types") == {"invoice", "waybill"}, f"fold {f['fold']} lacks a doc type"
    all_groups = [x for f in folds for x in f["val_groups"]]
    assert len(all_groups) == len(set(all_groups)), "a supplier_group spans two folds"
    doc_fold = {d: f["fold"] for f in folds for d in f["val_doc_ids"]}
    assert len(doc_fold) == len(ids), "docs not covered exactly once"
    return {
        "k": k,
        "seed": seed,
        "group_key": "supplier_group",
        "folds": folds,
        "doc_fold": doc_fold,
    }
