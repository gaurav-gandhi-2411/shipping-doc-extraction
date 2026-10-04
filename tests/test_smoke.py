from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from shipdoc import smoke

ROOT = Path(__file__).resolve().parents[1]
MAX_NEW = 1152


def _row(q: Any = "10", part: Any = "P1") -> dict[str, Any]:
    return {
        "supplier_part_number": part,
        "customer_part_number": None,
        "purchase_order": None,
        "quantity": q,
    }


def _page(n: int = 1, valid: bool = True, tokens: int = 300, raw: str | None = None) -> dict:
    return {
        "page": n,
        "json_valid": valid,
        "raw_text": raw if raw is not None else '{"a": 1}',
        "meta": {"n_output_tokens": tokens},
    }


def _setup(
    tmp_path: Path,
    *,
    rows_per_invoice: int = 4,
    gold_rows: int = 4,
    quantity: Any = "10",
    total: Any = "99.5",
    page: dict | None = None,
    null_row: bool = False,
    empty_doc: str | None = None,
    n_invoices: int = 3,
    total_missing_on: int = 0,
) -> tuple[Path, Path]:
    """Synthetic run dir (trace.jsonl + predictions.json) and labels dir; defaults pass."""
    run, labels = tmp_path / "run", tmp_path / "labels"
    run.mkdir(parents=True)
    labels.mkdir()
    traces, preds = [], {}
    docs = [(f"inv{i}", "invoice") for i in range(n_invoices)] + [("wb0", "waybill")]
    for k, (doc_id, kind) in enumerate(docs):
        (labels / f"{doc_id}.json").write_text(
            json.dumps(
                {
                    "doc_id": doc_id,
                    "doc_type": kind,
                    "header": {},
                    "line_items": [_row()] * (gold_rows if kind == "invoice" else 0),
                }
            )
        )
        traces.append({"doc_id": doc_id, "pages": [page or _page()]})
        if kind == "invoice":
            n = 0 if doc_id == empty_doc else rows_per_invoice
            rows = [_row(quantity, f"P{j}") for j in range(n)]
            if null_row:
                rows.append(_row(None, None))
            t = None if k < total_missing_on else total
            preds[doc_id] = {"doc_type": kind, "header": {"total_amount": t}, "line_items": rows}
        else:
            preds[doc_id] = {"doc_type": kind, "header": {"pieces": "3"}, "line_items": []}
    (run / "trace.jsonl").write_text("".join(json.dumps(t) + "\n" for t in traces))
    (run / "predictions.json").write_text(json.dumps(preds))
    return run, labels


def _failed(tmp_path: Path, **kw: Any) -> list[str]:
    run, labels = _setup(tmp_path, **kw)
    return [c.name for c in smoke.load_and_check(run, labels, MAX_NEW) if not c.passed]


def test_all_assertions_pass_on_a_healthy_run(tmp_path: Path) -> None:
    run, labels = _setup(tmp_path)
    checks = smoke.load_and_check(run, labels, MAX_NEW)
    assert [c.name[0] for c in checks] == list("abcdef")
    assert all(c.passed for c in checks), smoke.format_table("m", checks)


def test_a_invalid_json_page_fails(tmp_path: Path) -> None:
    failed = _failed(tmp_path, page=_page(valid=False, raw='{"a": 1}'))
    assert failed == ["a_json_valid_rate"]


def test_b_page_at_token_cap_fails(tmp_path: Path) -> None:
    assert _failed(tmp_path, page=_page(tokens=MAX_NEW)) == ["b_no_truncation"]


def test_b_unterminated_raw_output_fails(tmp_path: Path) -> None:
    # json_valid is True here on purpose: the raw-text check must stand on its own.
    assert _failed(tmp_path, page=_page(raw='{"a": 1')) == ["b_no_truncation"]


def test_c_invoice_without_rows_fails(tmp_path: Path) -> None:
    assert _failed(tmp_path, empty_doc="inv1") == ["c_row_counts"]


def test_c_row_count_outside_half_tolerance_fails(tmp_path: Path) -> None:
    # gold 3 invoices x 4 = 12: 6..18 allowed; 1 row each = 3 emitted is too few, 6 each too many
    assert _failed(tmp_path, rows_per_invoice=1) == ["c_row_counts"]
    assert _failed(tmp_path / "hi", rows_per_invoice=7) == ["c_row_counts"]
    assert _failed(tmp_path / "edge", rows_per_invoice=6, gold_rows=4) == []  # 18 == 1.5 x 12


def test_d_missing_quantities_fail(tmp_path: Path) -> None:
    assert _failed(tmp_path, quantity=None) == ["d_quantity_share"]
    assert _failed(tmp_path / "blank", quantity="  ") == ["d_quantity_share"]


def test_e_total_amount_share_below_two_thirds_fails(tmp_path: Path) -> None:
    assert _failed(tmp_path, total_missing_on=2) == ["e_total_amount_share"]  # 1/3
    assert _failed(tmp_path / "ok", total_missing_on=1) == []  # 2/3 passes


def test_f_all_null_row_after_merge_fails(tmp_path: Path) -> None:
    # one extra null row per invoice stays inside the +-50% window, so only (f) trips
    assert _failed(tmp_path, null_row=True) == ["f_no_all_null_rows"]


def test_json_numbers_count_as_values(tmp_path: Path) -> None:
    assert _failed(tmp_path, quantity=10, total=1234.5) == []
    assert _failed(tmp_path / "zero", quantity=0) == []  # 0 is a value, not a blank


def test_no_invoices_or_pages_fails_closed() -> None:
    checks = smoke.check_smoke([], {}, {}, MAX_NEW)
    assert [c.name for c in checks if not c.passed] == [
        "a_json_valid_rate",
        "c_row_counts",
        "d_quantity_share",
        "e_total_amount_share",
    ]


def test_format_table_marks_pass_and_fail(tmp_path: Path) -> None:
    run, labels = _setup(tmp_path, quantity=None)
    text = smoke.format_table("qwen35_4b", smoke.load_and_check(run, labels, MAX_NEW))
    assert "qwen35_4b" in text and "FAIL  0.000 of 12 rows" in text and "PASS" in text


def test_committed_smoke5_matches_the_picker_and_composition() -> None:
    spike = json.loads((ROOT / "splits" / "spike40.json").read_text(encoding="utf-8"))
    meta = json.loads((ROOT / "meta" / "dev.json").read_text(encoding="utf-8"))
    ids = json.loads((ROOT / "splits" / "smoke5.json").read_text(encoding="utf-8"))
    assert ids == smoke.pick_smoke_docs(spike, meta)
    by = {m["doc_id"]: m for m in meta}
    assert set(ids) <= set(spike) and len(ids) == 5
    assert sum(not by[d]["waybill"] for d in ids) >= 3
    assert any(by[d]["multipage"] for d in ids) and any(by[d]["waybill"] for d in ids)


def test_picker_fails_when_composition_is_impossible() -> None:
    meta = [
        {"doc_id": f"d{i}", "waybill": False, "multipage": False, "illegible": False,
         "scanned": False}
        for i in range(6)
    ]  # fmt: skip
    with pytest.raises(ValueError, match="cannot satisfy"):
        smoke.pick_smoke_docs([m["doc_id"] for m in meta], meta)


def _cpn_row(cpn: Any, po: Any) -> dict[str, Any]:
    return {
        "supplier_part_number": "P1",
        "customer_part_number": cpn,
        "purchase_order": po,
        "quantity": "1",
    }


def test_cpn_equals_po_warning_uses_identifier_rule() -> None:
    preds = {
        "d1": {
            "line_items": [_cpn_row("po 77", "PO77"), _cpn_row("PO-8", "PO8"), _cpn_row(None, None)]
        },
        "d2": {"line_items": [_cpn_row("X1", "X1"), _cpn_row("X1", None), _cpn_row("", "")]},
    }
    w = smoke.warn_cpn_equals_po(preds)
    assert w.name == "cpn_equals_po" and w.count == 2  # whitespace/case ignored, '-' kept
    assert "2 of 6 rows" in w.detail


def test_cpn_warning_never_blocks_and_never_raises(tmp_path: Path) -> None:
    run, labels = _setup(tmp_path)
    preds = json.loads((run / "predictions.json").read_text())
    for p in preds.values():
        for r in p.get("line_items", []):
            r["customer_part_number"] = r["purchase_order"] = "SAME"
    (run / "predictions.json").write_text(json.dumps(preds))
    checks = smoke.load_and_check(run, labels, MAX_NEW)
    assert all(c.passed for c in checks)  # the warning is not a Check
    warns = smoke.load_warnings(run)
    assert warns[0].count and warns[0].count > 0
    assert "WARN" in smoke.format_table(
        "m", checks, warns
    ) and "non-blocking" in smoke.format_table("m", checks, warns)
    (run / "predictions.json").write_text("not json")
    bad = smoke.load_warnings(run)
    assert bad[0].count is None and "not computed" in bad[0].detail
    assert smoke.load_warnings(tmp_path / "absent")[0].count is None
