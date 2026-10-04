"""scripts/false_fill_diag.py pure pieces on synthetic documents (no run folder, no OCR cache)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from _requires import SKIP_REASON, missing_inputs

from shipdoc import eval as ev
from shipdoc.oof import over_null_counts

if missing_inputs():  # the scorer is loaded at import time below: skip the module, not error
    pytest.skip(SKIP_REASON, allow_module_level=True)

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "false_fill_diag", ROOT / "scripts" / "false_fill_diag.py"
)
ffd = importlib.util.module_from_spec(_spec)
sys.modules["false_fill_diag"] = ffd
_spec.loader.exec_module(ffd)

SC = ev.load_scorer()


def row(spn: str, cpn: Any = None, po: Any = None, qty: Any = "1") -> dict[str, Any]:
    return {
        "supplier_part_number": spn,
        "customer_part_number": cpn,
        "purchase_order": po,
        "quantity": qty,
    }


def doc(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """A prediction / gold document; the header is irrelevant to the row-cell functions."""
    return {"doc_type": "invoice", "header": {}, "line_items": rows}


# gold: row 0 stores the PO text under customer_part_number and leaves purchase_order null
GOLD = {"d1": doc([row("S1", cpn="PO-77"), row("S2", po="PO-9")])}


def test_false_fill_cell_found_and_matches_over_null_counts() -> None:
    pred = {"d1": doc([row("S1", cpn="PO-77", po="PO-77"), row("S2", po="PO-9")])}
    assert ffd.false_fill_cells(SC, pred, GOLD) == [("d1", 0, "purchase_order")]
    assert (
        len(ffd.false_fill_cells(SC, pred, GOLD))
        == over_null_counts(pred, _hdr(GOLD))["row_false_fill"]
    )


def _hdr(gold: dict[str, Any]) -> dict[str, Any]:
    """Gold with a complete (all-null) header so over_null_counts can read it."""
    return {d: g | {"header": {f: None for f in SC.HEADER[g["doc_type"]]}} for d, g in gold.items()}


def test_no_false_fill_when_prediction_is_right() -> None:
    pred = {"d1": doc([row("S1", cpn="PO-77"), row("S2", po="PO-9")])}
    assert ffd.false_fill_cells(SC, pred, GOLD) == []


def test_cell_state_null_value_unpaired() -> None:
    cell = ("d1", 0, "purchase_order")
    value = {"d1": doc([row("S1", cpn="PO-77", po="PO-77")])}
    null = {"d1": doc([row("S1", cpn="PO-77")])}
    unpaired = {"d1": doc([row("OTHER", po="PO-77")])}
    missing: dict[str, Any] = {}
    assert ffd.cell_state(SC, value, GOLD, cell) == "value"
    assert ffd.cell_state(SC, null, GOLD, cell) == "null"
    assert ffd.cell_state(SC, unpaired, GOLD, cell) == "unpaired"
    assert ffd.cell_state(SC, missing, GOLD, cell) == "unpaired"


def test_categorise_a_when_value_is_gold_of_other_slot() -> None:
    p = row("S1", cpn="PO-77", po="po 77")
    c = ffd.categorise(SC, "purchase_order", p, GOLD["d1"]["line_items"][0], None)
    assert c["category"] == "a"
    assert c["equals_gold_other_slot"] == ["customer_part_number"]
    assert c["duplicated_in_pred_slot"] == ["customer_part_number"]


def test_categorise_b_and_c_depend_on_ocr_support(monkeypatch: Any) -> None:
    p = row("S1", cpn="PO-77", po="ZZ-1")
    g = GOLD["d1"]["line_items"][0]
    monkeypatch.setattr(ffd.locate, "locate", lambda *a, **k: SimpleNamespace(level="fuzzy"))
    b = ffd.categorise(SC, "purchase_order", p, g, object())
    assert (b["category"], b["in_ocr"], b["ocr_level"]) == ("b", True, "fuzzy")
    monkeypatch.setattr(ffd.locate, "locate", lambda *a, **k: None)
    c = ffd.categorise(SC, "purchase_order", p, g, object())
    assert (c["category"], c["in_ocr"], c["ocr_level"]) == ("c", False, None)


def test_cell_set_change_and_count_table() -> None:
    a, b, c = ("d", 0, "x"), ("d", 1, "x"), ("d", 2, "x")
    assert ffd.cell_set_change([a, b], [b, c]) == {"before": 2, "kept": 1, "cleared": 1, "new": 1}
    pred = {"d1": doc([row("S1", cpn="PO-77", po="PO-77"), row("S2", po="PO-9")])}
    t = ffd.count_table({"raw": pred, "empty": {}}, _hdr(GOLD))
    assert t["raw"]["false_fill_total"] == 1 and t["raw"]["over_null_total"] == 0
    assert t["empty"]["false_fill_total"] == 0 and t["empty"]["rows_unmatched_gold"] == 2


def test_render_prints_no_value() -> None:
    cell = {
        "doc": "d1", "field": "purchase_order", "doc_type": "invoice", "scanned": False,
        "group": "inv_g01", "pairing": "partial", "zs_raw_state": "null",
        "zs_rules_state": "null", "ft_rules_state": "null", "category": "a",
        "equals_gold_other_slot": ["customer_part_number"], "duplicated_in_pred_slot": [],
        "in_ocr": True, "ocr_level": "exact",
    }  # fmt: skip
    z = {k: 0 for k in (
        "header_false_fill", "row_false_fill", "false_fill_total", "header_over_null",
        "row_over_null", "over_null_total", "rows_unmatched_gold")}  # fmt: skip
    res = {
        "n_docs": 1, "cells": [cell],
        "counts": {a: z for a in ("ZS-raw", "FT-raw", "ZS+rules", "FT+rules")},
        "rules_cell_change": {"ZS": ffd.cell_set_change([], []), "FT": ffd.cell_set_change([], [])},
    }  # fmt: skip
    text = ffd.render(res, 0, {"FT run": "r"})
    assert "PO-77" not in text and "1 cells; 1 distinct docs" in text


def _mini_run(tmp: Path, name: str, h: str, oof: bool) -> Path:
    import json

    d = tmp / name
    d.mkdir()
    man: dict[str, Any] = {"config": {"name": "cfg", "hash": h}}
    if oof:
        man["oof"] = {
            "fold": 0, "adapter_sha256": "a" * 64, "train_code_sha": "b" * 40,
            "verification": {"ok": True},
        }  # fmt: skip
    (d / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    (d / "progress.json").write_text('{"status": "complete"}', encoding="utf-8")
    return d


def test_refuses_a_native_ft_run_against_a_1260_zs_run(tmp_path: Path) -> None:
    import pytest

    ft, zs = _mini_run(tmp_path, "oof_fold0_x", "hn", True), _mini_run(tmp_path, "zs", "h1", False)
    with pytest.raises(SystemExit, match=r"mixed-resolution.*FT run `oof_fold0_x`.*ZS run `zs`"):
        ffd.main(["--oof-run-dir", str(ft), "--zs-run-dir", str(zs), "--out", str(tmp_path / "o")])
    assert not (tmp_path / "o").exists()
