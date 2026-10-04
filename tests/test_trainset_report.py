"""scripts/trainset_report.py: the report builder on a synthetic corpus (counts only)."""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "trainset_report", ROOT / "scripts" / "trainset_report.py"
)
tsr = importlib.util.module_from_spec(spec)
sys.modules["trainset_report"] = tsr
spec.loader.exec_module(tsr)

from shipdoc import trainset as ts  # noqa: E402


def gold(doc_id: str, n_pages: int, n_rows: int) -> dict:
    return {
        "doc_id": doc_id,
        "doc_type": "invoice",
        "pages": [f"{doc_id}_p{k + 1}.png" for k in range(n_pages)],
        "header": {
            "invoice_number": "INV-1", "invoice_date": "2026-01-02", "supplier_name": "Acme Zed",
            "buyer_name": "B", "ship_to_name": "T", "currency": "USD", "total_amount": "5",
            "awb_number": None,
        },
        "line_items": [
            {"supplier_part_number": f"P{i}", "customer_part_number": None,
             "purchase_order": None, "quantity": "1"}
            for i in range(n_rows)
        ],
    }  # fmt: skip


FOLDS = {
    "folds": [
        {"fold": 0, "val_doc_ids": ["dev_0000", "train_0000"]},
        {"fold": 1, "val_doc_ids": ["dev_0001", "train_0001"]},
        {"fold": 2, "val_doc_ids": ["dev_0002", "train_0002"]},
    ]
}


def test_report_has_every_section_and_no_values() -> None:
    ids = [d for f in FOLDS["folds"] for d in f["val_doc_ids"]]
    golds = {d: gold(d, 2, 3) for d in ids}
    asg = {d: (["line", "line", "line"], [0, 0, 1]) for d in ids}
    asg["train_0000"] = (["line", "unassigned", "line"], [0, None, 1])  # one ambiguous row
    cands = {
        "train_0001": [ts.OcclusionCandidate("invoice_number", 0, ((0, (1.0, 2.0, 30.0, 12.0)),))]
    }
    text = tsr.build_report(golds, asg, cands, Counter(eligible=1, null_gold=2), (5, 0), None,
                            FOLDS, "abc1234")  # fmt: skip
    for heading in ("## Row -> page plan", "## Round trip", "## Stage splits", "## Augmentation"):
        assert heading in text
    assert "header_only" in text and "abc1234" in text and "differ" in text
    assert "1 rows / 1 docs" in text  # the ambiguous row, with the boundary rule on and off
    assert "Acme Zed" not in text and "INV-1" not in text  # counts only
    assert "UNVERIFIED" in text


def test_realised_rates_count_selected_pages() -> None:
    golds = {"train_0000": gold("train_0000", 2, 2)}
    prepared = ts.prepare(golds, {"train_0000": (["line", "line"], [0, 1])})
    prepared.candidates = {"train_0000": [ts.OcclusionCandidate("invoice_number", 0, ())]}
    r = tsr.realised_rates(prepared, ["train_0000"], 1.0, 42, 3)
    assert r == {
        "pages": 6,
        "hits": 3,
        "fields": Counter(invoice_number=3),
    }  # only page 0 carries it
    assert tsr.realised_rates(prepared, ["train_0000"], 0.0, 42, 3)["hits"] == 0
