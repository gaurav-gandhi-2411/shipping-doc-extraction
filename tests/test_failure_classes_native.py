"""scripts/failure_classes_native.py: cell counting, class tally, ranking, rendering."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_report as br  # noqa: E402
import failure_classes_native as fc  # noqa: E402


class _Scorer:
    """Stand-in for score.py: equal after treating None / blank as empty."""

    @staticmethod
    def same(field: str, pred: Any, gold: Any) -> bool:
        def norm(v: Any) -> Any:
            return None if v is None or (isinstance(v, str) and not v.strip()) else v

        return norm(pred) == norm(gold)


def test_wrong_fields_counts_only_differing_row_fields_null_aware() -> None:
    gold = {"supplier_part_number": "A1", "customer_part_number": None, "quantity": 3}
    pred = {"supplier_part_number": "A1", "customer_part_number": "X", "quantity": 3}
    assert fc.wrong_fields(_Scorer, pred, gold) == ["customer_part_number"]
    assert fc.wrong_fields(_Scorer, gold, gold) == []
    # a prediction that is null where gold has a value is wrong too (an over-null)
    assert fc.wrong_fields(_Scorer, {"quantity": None}, {"quantity": 3}) == ["quantity"]


def _recs() -> list[dict[str, Any]]:
    return [
        {"doc": "dev_a", "cls": "column_shift", "fields": ["supplier_part_number", "quantity"],
         "scanned": True},
        {"doc": "dev_a", "cls": "column_shift", "fields": ["customer_part_number"],
         "scanned": True},
        {"doc": "train_b", "cls": "column_shift", "fields": ["quantity"], "scanned": False},
        {"doc": "train_b", "cls": "spn_misread", "fields": ["supplier_part_number"],
         "scanned": False},
        {"doc": "train_c", "cls": "cpn_false_fill", "fields": ["customer_part_number"],
         "scanned": False},
        {"doc": "train_c", "cls": "cpn_false_fill", "fields": ["customer_part_number"],
         "scanned": False},
    ]  # fmt: skip


def test_tally_rows_cells_docs_and_scan_split() -> None:
    t = fc.tally(_recs())
    cs = t["column_shift"]
    assert (cs["rows"], cs["cells"], cs["docs"], cs["scanned_rows"]) == (3, 4, 2, 2)
    assert cs["by_field"] == {"supplier_part_number": 1, "quantity": 2, "customer_part_number": 1}
    assert t["cpn_false_fill"]["rows"] == 2 and t["cpn_false_fill"]["docs"] == 1


def test_tally_filter_keeps_only_the_selected_docs() -> None:
    t = fc.tally(_recs(), lambda d: d.startswith("dev_"))
    assert set(t) == {"column_shift"} and t["column_shift"]["rows"] == 2


def test_top_by_cells_orders_by_cells_then_rows_then_name() -> None:
    t = fc.tally(_recs())
    assert fc.top_by_cells(t) == ["column_shift", "cpn_false_fill", "spn_misread"]
    assert fc.top_by_cells(t, 1) == ["column_shift"]
    tie = {"b": {"cells": 2, "rows": 1}, "a": {"cells": 2, "rows": 1}, "c": {"cells": 2, "rows": 2}}
    assert fc.top_by_cells(tie) == ["c", "a", "b"]


def test_every_fix_cites_an_artifact_and_claims_no_unmeasured_gain() -> None:
    for name, fx in fc.FIXES.items():
        assert "reports/" in fx["evidence"], name
    assert "no fine-tune gain is claimed" in fc.FIXES["spn_misread"]["evidence"]
    assert "ORACLE" in fc.FIXES["column_shift"]["evidence"]


def test_render_is_aggregates_only_and_names_the_top_three() -> None:
    all_t, dev_t = fc.tally(_recs()), fc.tally(_recs(), lambda d: d.startswith("dev_"))
    res = {
        "n_docs": 500,
        "gold_rows": 4930,
        "gold_rows_dev": 924,
        "all500": all_t,
        "dev100": dev_t,
        "top3_all500": fc.top_by_cells(all_t),
        "top3_dev100": fc.top_by_cells(dev_t),
        "wrong_rows_all500": 6,
        "wrong_cells_all500": 7,
        "wrong_rows_dev100": 2,
        "wrong_cells_dev100": 3,
        "header_wrong_all500": 7,
        "header_wrong_dev100": 2,
        "scorer_all500": {"gold_rows": 4930, "wrong_rows": 6, "cells": 20},
        "scorer_dev100": {"gold_rows": 924, "wrong_rows": 2, "cells": 8},
    }
    md = fc.render(res, "cmd", "abc")
    assert "### 1. column_shift" in md and "### 2. cpn_false_fill" in md
    assert "### 3. spn_misread" in md
    assert "DIFFER" in md  # dev100 has only one class: the two top-3 lists differ and it says so
    assert br.find_doc_ids(md) == []


def test_scorer_cells_counts_unpaired_rows_as_all_cells_wrong() -> None:
    class Sc:
        @staticmethod
        def score_doc(pred: Any, gold: Any) -> dict[str, Any]:
            # 5 gold rows, 3 fully right; one field of one matched row wrong, one row unmatched
            return {"rows_gold": 5, "rows_full": 3, "row_fields": {"a": 4, "b": 5, "c": 4, "d": 4}}

    out = fc.scorer_cells(Sc, {"x": {}, "y": {}}, {"x": {}, "y": {}}, lambda d: d == "x")
    assert out == {"gold_rows": 5, "wrong_rows": 2, "cells": 3}
