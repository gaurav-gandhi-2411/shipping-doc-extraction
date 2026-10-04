"""Gold -> per-page target rendering and the token-budget arithmetic (scripts/token_budget.py)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from shipdoc import extract as ex
from shipdoc import targets as T

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))  # token_budget imports its sibling _corpus
_spec = importlib.util.spec_from_file_location("token_budget", ROOT / "scripts" / "token_budget.py")
tb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tb)


def _gold(n_rows: int = 5) -> dict:
    rows = [
        {
            "supplier_part_number": f"P{i}",
            "customer_part_number": None,
            "purchase_order": None if i % 2 else f"PO{i}",
            "quantity": str(10 * i),
        }
        for i in range(n_rows)
    ]
    header = {
        "invoice_number": "INV-1",
        "invoice_date": "2026-01-02",
        "supplier_name": "Acme",
        "buyer_name": "Buyer",
        "ship_to_name": "Buyer",
        "currency": "USD",
        "total_amount": "123.45",
        "awb_number": None,
    }
    return {"doc_type": "invoice", "header": header, "line_items": rows}


def test_header_identity_on_first_page_total_on_last() -> None:
    pages = T.gold_page_payloads(_gold(), 3, [0, 0, 1, 2, 2])
    assert [p["page_kind"] for p in pages] == ["first", "continuation", "continuation"]
    assert (
        pages[0]["header"]["invoice_number"] == "INV-1"
        and pages[0]["header"]["total_amount"] is None
    )
    assert (
        pages[2]["header"]["total_amount"] == "123.45"
        and pages[2]["header"]["invoice_number"] is None
    )
    assert all(set(p["header"]) == set(ex.HEADER_KEYS) for p in pages)  # union header, nulls
    assert [len(p["line_items"]) for p in pages] == [2, 1, 2]
    assert ex.schema_errors(pages[1]) == []


def test_rows_keep_gold_order_and_are_never_dropped_or_clamped_wrongly() -> None:
    gold = _gold(4)
    pages = T.gold_page_payloads(gold, 2, [5, -1, 1, 0])  # out of range -> clamped into range
    flat = [r["supplier_part_number"] for p in pages for r in p["line_items"]]
    assert sorted(flat) == ["P0", "P1", "P2", "P3"]
    assert [r["supplier_part_number"] for r in pages[1]["line_items"]] == ["P0", "P2"]


def test_even_split_default_and_single_page() -> None:
    pages = T.gold_page_payloads(_gold(5), 2)
    assert [len(p["line_items"]) for p in pages] == [3, 2]
    one = T.gold_page_payloads(_gold(), 1)
    assert len(one) == 1 and one[0]["page_kind"] == "single"
    assert (
        one[0]["header"]["total_amount"] == "123.45"
        and one[0]["header"]["invoice_number"] == "INV-1"
    )


def test_row_pages_length_mismatch_is_an_error() -> None:
    with pytest.raises(ValueError):
        T.gold_page_payloads(_gold(3), 2, [0, 1])


def test_render_target_matches_xgrammar_default_separators() -> None:
    text = T.render_target({"a": None, "b": [1, 2]})
    assert text == '{"a": null, "b": [1, 2]}'
    assert json.loads(text) == {"a": None, "b": [1, 2]}


def test_budget_formula_rounds_up_to_multiple_of_64() -> None:
    assert tb.round_up(1488) == 1536 and tb.round_up(1536) == 1536 and tb.round_up(1537) == 1600
    assert tb.budget_from_p99(1190) == 1536  # ceil(1190 * 1.25) = 1488 -> 1536
    assert tb.budget_from_p99(1229) == 1600  # ceil(1536.25) = 1537 -> 1600
    assert tb.budget_from_p99(100) == 128  # 125 -> 128


def test_dist_percentiles() -> None:
    d = tb.dist(list(range(1, 101)))
    assert d["mean"] == 50.5 and d["max"] == 100 and d["p50"] == 50.5 and d["n"] == 100
    assert d["p99"] == pytest.approx(99.01)
