"""Helpers of scripts/postproc_ablation.py (no OCR cache, no run directory needed)."""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCORER = ROOT / "assignment" / "score.py"
pytestmark = pytest.mark.skipif(not SCORER.is_file(), reason="assignment/score.py absent")

sys.path.insert(0, str(ROOT / "scripts"))  # the script imports its sibling _corpus
spec = importlib.util.spec_from_file_location(
    "postproc_ablation", ROOT / "scripts" / "postproc_ablation.py"
)
pa = importlib.util.module_from_spec(spec)
sys.modules["postproc_ablation"] = pa  # dataclasses resolve annotations through sys.modules
spec.loader.exec_module(pa)


def _gold(i: int, airport: str = "ICN") -> dict[str, Any]:
    return {
        "doc_id": f"dev_{i:04d}",
        "doc_type": "waybill",
        "header": {
            "carrier": "Acme",
            "mawb": "123-12345678",
            "hawb": None,
            "origin_airport": airport,
            "destination_airport": "HKG",
            "shipper_name": "S",
            "consignee_name": "C",
            "pieces": "3",
            "gross_weight_kg": "10.5",
        },
        "line_items": [],
    }


def _pred(g: dict[str, Any], airport: str) -> dict[str, Any]:
    return {**g, "header": {**g["header"], "origin_airport": airport}}


def test_paired_identical_predictions_are_exactly_zero_and_skip_the_bootstrap() -> None:
    gold = {f"dev_{i:04d}": _gold(i) for i in range(10)}
    res = pa.paired(gold, gold, gold)
    assert res["identical"] is True and res["OVERALL"] == {"delta": 0.0, "lo": 0.0, "hi": 0.0}


def test_paired_detects_a_consistent_gain_and_the_gate_verdicts() -> None:
    gold = {f"dev_{i:04d}": _gold(i) for i in range(40)}
    bad = {d: _pred(g, "ICN SEOUL") for d, g in gold.items()}
    good = {d: _pred(g, "ICN") for d, g in gold.items()}
    res = pa.paired(bad, good, gold)
    assert (
        res["OVERALL"]["delta"] > 0 and res["OVERALL"]["lo"] > 0 and pa.excludes_0(res["OVERALL"])
    )
    assert pa.kept_decision("R2b", {"vs_prev": res}).startswith("KEEP")
    flat = pa.paired(good, good, gold)
    assert pa.kept_decision("R1a", {"vs_prev": flat}).startswith("FAIL gate (no measured effect)")
    worse = pa.paired(good, bad, gold)
    assert pa.kept_decision("R2b", {"vs_prev": worse}).startswith("DROP")
    # false fill lower with OVERALL not lower passes the second branch of the gate
    lower_ff = {
        "OVERALL": {"delta": 0.0, "lo": -0.01, "hi": 0.01},
        "false_fill_rate": {"delta": -0.05, "lo": -0.1, "hi": 0.0},
    }
    assert pa.kept_decision("R9", {"vs_prev": lower_ff}).startswith("KEEP (false-fill")


def test_cell_changes_header_flips_and_no_invention() -> None:
    gold = {f"dev_{i:04d}": _gold(i) for i in range(4)}
    a = {d: _pred(g, "ICN SEOUL") for d, g in gold.items()}
    b = {d: _pred(g, "ICN") for d, g in gold.items()}
    assert pa.cell_changes(a, b) == {"header.origin_airport": 4}
    assert pa.header_flips(a, b, gold) == {"origin_airport": {"gain": 4, "loss": 0}}
    assert pa.no_invention(a, b) == {"changed_cells": 4, "violations": 0}
    c = {d: _pred(g, "LAX") for d, g in gold.items()}  # not contained in the raw value
    assert pa.no_invention(a, c)["violations"] == 4


def test_wilson_interval() -> None:
    lo, hi = pa.wilson(0, 77)
    assert lo == pytest.approx(0.0, abs=1e-9) and 0.04 < hi < 0.06
    assert all(x != x for x in pa.wilson(0, 0))  # nan for an empty sample


def test_render_date_round_trips_through_the_normaliser() -> None:
    from shipdoc.normalize import normalize_date

    for iso in ("2026-05-25", "2026-09-30"):  # day > 12: unambiguous in every numeric format
        for fmt in ("iso", "dmy/", "mdy/", "D-MON-Y", "Month D, Y"):
            assert normalize_date(pa.render_date(iso, fmt))[0] == iso, (iso, fmt)
    assert pa.render_date("2026-03-04", "dmy/") == "04/03/2026"
    assert pa.render_date("2026-03-04", "mdy/") == "03/04/2026"


def test_mutations_change_the_value_and_are_deterministic() -> None:
    rng = random.Random(42)
    for kind in ("look_alike", "same_class_substitution", "drop_char", "insert_char"):
        out = pa._mutate("123-12345678", kind, rng)
        assert out is not None and out != "123-12345678", kind
    r1, r2 = random.Random(7), random.Random(7)
    assert pa._mutate("HKG", "same_class_substitution", r1) == pa._mutate(
        "HKG", "same_class_substitution", r2
    )


def test_validator_report_leaves_predictions_untouched_and_counts() -> None:
    gold = {f"dev_{i:04d}": _gold(i) for i in range(6)}
    pred = {d: _pred(g, "ICN") for d, g in gold.items()}
    pred["dev_0000"] = _pred(gold["dev_0000"], "ZZZ9")  # wrong and malformed: flagged
    pred["dev_0001"] = _pred(gold["dev_0001"], "LAX")  # wrong but a valid code: not flagged
    rep = pa.validator_report(pred, gold, [g for g in gold.values()])
    air = rep["R3 airports (IATA)"]
    assert air["wrong"] == 2 and air["flagged"] == 1 and air["flagged_wrong"] == 1
    assert air["recall_of_errors"] == 0.5 and air["flag_precision"] == 1.0
    assert air["false_alarm_rate"] == 0.0


# ---------------------------------------------------------------- 500-doc re-gate mode


def _rec(doc_id: str, doc_type: str = "waybill", scanned: bool = False, split: str = "dev") -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(
        doc_id=doc_id, doc_type=doc_type, scanned=scanned, split=split, gold=_gold(0)
    )


def test_select_docs_keeps_trace_order_and_fails_closed() -> None:
    corp = [_rec("a"), _rec("b"), _rec("c")]
    assert list(pa.select_docs(corp, ["c", "a"])) == ["c", "a"]
    with pytest.raises(ValueError, match="not in the train.dev corpus"):
        pa.select_docs(corp, ["a", "zzz"])
    with pytest.raises(ValueError, match="duplicate"):
        pa.select_docs(corp, ["a", "a"])


def test_merge_block_never_touches_the_other_block() -> None:
    assert pa.BEGIN not in pa.BEGIN_500 and pa.END != pa.END_500
    old = f"# T\n\n{pa.BEGIN}\nOLD\n{pa.END}\n\n## Sources\ntext\n"
    added = pa.merge_block(old, "NEW", pa.BEGIN_500, pa.END_500)
    assert added.startswith(old.rstrip("\n"))  # appended after, old text byte-identical
    assert f"{pa.BEGIN_500}\nNEW\n{pa.END_500}\n" in added
    again = pa.merge_block(added, "NEW2", pa.BEGIN_500, pa.END_500)  # idempotent replace
    assert again.count(pa.BEGIN_500) == 1 and "NEW2" in again and "NEW\n" not in again
    assert again.startswith(old.rstrip("\n"))
    redo_old = pa.merge_block(again, "OLD2")  # dev100 write leaves the 500 block alone
    assert "OLD2" in redo_old and f"{pa.BEGIN_500}\nNEW2\n{pa.END_500}\n" in redo_old


def test_tagged_500_block_is_separate_from_the_default_500_block() -> None:
    assert pa.markers_500() == (pa.BEGIN_500, pa.END_500)
    begin, end = pa.markers_500("native resolution (max_pixels 2196480)")
    assert begin not in (pa.BEGIN_500, pa.BEGIN) and end not in (pa.END_500, pa.END)
    assert pa.BEGIN_500 not in begin and pa.END_500 not in end  # substring-safe for merge_block
    one = f"{pa.BEGIN_500}\nONE\n{pa.END_500}\n"
    old = "# T\n\n" + one
    added = pa.merge_block(old, "TWO", begin, end)
    assert one in added and f"{begin}\nTWO\n{end}\n" in added
    again = pa.merge_block(added, "TWO2", begin, end)
    assert again.count(begin) == 1 and "ONE" in again and "TWO2" in again and "TWO\n" not in again


def test_slice_rows_partitions_by_type_scan_and_split() -> None:
    recs = {
        f"dev_{i:04d}": _rec(f"dev_{i:04d}", scanned=i % 2 == 0, split="train" if i < 4 else "dev")
        for i in range(8)
    }
    gold = {d: _gold(i) for i, d in enumerate(recs)}
    bad = {d: _pred(g, "ICN SEOUL") for d, g in gold.items()}
    good = {d: _pred(g, "ICN") for d, g in gold.items()}
    rows = {r["slice"]: r for r in pa.slice_rows(bad, good, gold, recs)}
    assert [rows[k]["n"] for k in ("all", "scanned", "digital", "train docs", "dev docs")] == [
        8, 4, 4, 4, 4,
    ]  # fmt: skip
    assert "invoices" not in rows  # an empty slice is omitted, not reported as 0 docs
    assert rows["all"]["paired"]["OVERALL"]["delta"] > 0


def test_differing_docs_and_explain_differences_name_the_rule() -> None:
    from shipdoc.merge import FieldProvenance

    page = {"doc_type": "waybill", "header": {**_gold(0)["header"], "origin_airport": "ICN SEOUL"}}
    page["line_items"] = []
    ctx = pa.Ctx.__new__(pa.Ctx)
    ctx.prov, ctx.finalize = FieldProvenance.load(), False
    ctx.pages, ctx.total_page, ctx.date_order = {"d1": [page]}, {"d1": None}, {}
    base = pa.replace(pa.FULL, total_ocr=False, cluster_dates=False)
    saved = {"d1": pa.run_doc("d1", pa.replace(base, codes=False), ctx)[0]}  # as if R2b were off
    now = pa.run_all(base, ctx)
    assert pa.differing_docs(now, saved) == ["d1"] and pa.differing_docs(saved, saved) == []
    assert pa.explain_differences(ctx, saved, base, ["d1"]) == {"R2b": 1}
    assert pa.explain_differences(ctx, {"d1": {"x": 1}}, base, ["d1"]) == {"unexplained": 1}


def test_gate_word_and_validator_criterion() -> None:
    assert pa.gate_word("KEEP (CI excludes 0)") == "PASS gate"
    assert pa.gate_word("FAIL gate (CI includes 0)") == "FAIL gate"
    assert pa.gate_word("DROP (CI excludes 0, negative)") == "FAIL gate"
    few = {"wrong": 3, "recall_of_errors": 1.0, "flag_precision": 1.0, "base_error_rate": 0.01}
    assert "NOT ASSESSABLE" in pa.validator_criterion(few)
    good = {"wrong": 6, "recall_of_errors": 0.6, "flag_precision": 0.5, "base_error_rate": 0.1}
    assert "criterion MET" in pa.validator_criterion(good)
    poor = {"wrong": 6, "recall_of_errors": 0.4, "flag_precision": 0.5, "base_error_rate": 0.1}
    assert "NOT met" in pa.validator_criterion(poor)
