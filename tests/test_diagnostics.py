from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from shipdoc import diagnostics as dg

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    not (ROOT / "assignment" / "score.py").is_file(), reason="assignment/score.py absent"
)


def row(spn: Any, cpn: Any, po: Any, qty: Any = "3") -> dict[str, Any]:
    return {
        "supplier_part_number": spn,
        "customer_part_number": cpn,
        "purchase_order": po,
        "quantity": qty,
    }


def doc(rows: list[dict[str, Any]], doc_type: str = "invoice") -> dict[str, Any]:
    return {"doc_type": doc_type, "header": {}, "line_items": rows}


def test_rotation_among_spn_cpn_po_is_not_a_clean_swap() -> None:
    gold = doc([row("AX100", "BX200", "CX300")])
    pred = doc([row("BX200", "CX300", "AX100")])  # every value one column to the left
    c = dg.row_convention_errors(pred, gold)
    assert c["unpaired_gold_rows"] == 1
    assert c["rotations"] == 1 and c["clean_swaps"] == 0


def test_clean_spn_cpn_swap_counts_as_rotation_and_swap() -> None:
    gold = doc([row("AX100", "BX200", "CX300", "3")])
    pred = doc([row("BX200", "AX100", "CX300", "9")])
    c = dg.row_convention_errors(pred, gold)
    assert c["rotations"] == 1 and c["clean_swaps"] == 1


def test_misread_part_number_is_unpaired_but_not_a_shift() -> None:
    gold = doc([row("AX100", "BX200", "CX300")])
    pred = doc([row("AX101", "BX200", "CX300")])  # one char off, nothing in another column
    c = dg.row_convention_errors(pred, gold)
    assert c["unpaired_gold_rows"] == 1 and c["rotations"] == 0


def test_correct_rows_count_nothing() -> None:
    d = doc([row("AX100", "BX200", "CX300"), row("AX101", None, "CX301")])
    assert dg.row_convention_errors(d, d) == dict.fromkeys(dg.COUNT_KEYS, 0)


def test_truncated_doc_rows_are_not_rotations() -> None:
    gold = doc([row("AX100", "BX200", "CX300")])
    pred = doc([row("BX200", "CX300", "AX100")])
    c = dg.row_convention_errors(pred, gold, truncated=True)
    assert c["unpaired_gold_rows"] == 1 and c["rotations"] == 0


def test_cpn_equals_po_uses_identifier_rule() -> None:
    gold = doc([row("AX100", "BX200", "CX300"), row("AX101", "BX201", "CX301")])
    pred = doc(
        [row("AX100", "cx 300", "CX300"), row("AX101", "BX201", "CX-301")]  # case/space ok, - kept
    )
    assert dg.row_convention_errors(pred, gold)["cpn_equals_po"] == 1


def test_cpn_filled_where_gold_has_none() -> None:
    gold = doc([row("AX100", None, "CX300"), row("AX101", "", "CX301")])
    pred = doc([row("AX100", "CX300", "CX300"), row("AX101", None, "CX301")])
    c = dg.row_convention_errors(pred, gold)
    assert c["cpn_filled_gold_empty"] == 1
    assert (
        dg.row_convention_errors(pred, doc([row("AX100", "BX9", "CX300")]))["cpn_filled_gold_empty"]
        == 0
    )


def test_waybills_and_type_mismatch_are_zero() -> None:
    way = doc([], "waybill")
    inv = doc([row("AX100", "BX200", "CX300")])
    zero = dict.fromkeys(dg.COUNT_KEYS, 0)
    assert dg.row_convention_errors(way, way) == zero
    assert dg.row_convention_errors(way, inv) == zero


def test_sum_over_docs_and_missing_prediction() -> None:
    gold = {
        "d1": doc([row("AX100", "BX200", "CX300")]),
        "d2": doc([row("AX101", "BX201", "CX301")]),
        "d3": doc([row("AX102", "BX202", "CX302")]),
    }
    preds = {"d1": doc([row("BX200", "CX300", "AX100")]), "d2": gold["d2"]}  # d3 missing
    c = dg.sum_row_convention_errors(preds, gold)
    assert c["rotations"] == 1 and c["unpaired_gold_rows"] == 1  # d3: no invoice pred, skipped
    t = dg.sum_row_convention_errors(preds, gold, truncated_docs=["d1"])
    assert t["rotations"] == 0
