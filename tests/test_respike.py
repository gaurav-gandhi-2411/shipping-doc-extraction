"""Unit tests for scripts/respike.py on small synthetic fixtures (no gold labels needed)."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    not (ROOT / "assignment" / "score.py").is_file(), reason="assignment/score.py absent"
)

_spec = importlib.util.spec_from_file_location("respike", ROOT / "scripts" / "respike.py")
rs = importlib.util.module_from_spec(_spec)
sys.modules["respike"] = rs  # dataclasses resolve annotations via sys.modules
_spec.loader.exec_module(rs)

CPN, PO = "customer_part_number", "purchase_order"


def _row(spn: str, cpn: Any, po: Any, qty: str = "3") -> dict[str, Any]:
    return {"supplier_part_number": spn, CPN: cpn, PO: po, "quantity": qty}


def _inv_gold(i: int, awb: Any = None) -> dict[str, Any]:
    return {
        "doc_id": f"dev_{i:04d}",
        "doc_type": "invoice",
        "header": {
            "invoice_number": f"INV-{1000 + i}",
            "invoice_date": "2026-05-25",
            "supplier_name": f"Acme {i}",
            "buyer_name": "Buyer Ltd",
            "ship_to_name": "Ship Co",
            "currency": "USD",
            "total_amount": f"{100 + i}.50",
            "awb_number": awb,
        },
        "line_items": [_row(f"AX{i}", None, f"PO{i}"), _row(f"BX{i}", f"CP{i}", f"PO{i}")],
    }


def _wb_gold(i: int) -> dict[str, Any]:
    return {
        "doc_id": f"dev_{i:04d}",
        "doc_type": "waybill",
        "header": {
            "carrier": "Synthetic Carrier Alpha",
            "mawb": f"123-{10000000 + i}",
            "hawb": None,
            "origin_airport": "BOM",
            "destination_airport": "FRA",
            "shipper_name": "Ship Co",
            "consignee_name": "Cons Co",
            "pieces": "4",
            "gross_weight_kg": "12.5",
        },
        "line_items": [],
    }


def _gold_set() -> dict[str, dict[str, Any]]:
    docs = [_inv_gold(0, awb="123-45678901"), _inv_gold(1, awb=None), _wb_gold(2)]
    return {d["doc_id"]: d for d in docs}


def _pred_from_gold(gold: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        d: {
            "doc_type": g["doc_type"],
            "header": dict(g["header"]),
            "line_items": [dict(r) for r in g["line_items"]],
        }
        for d, g in gold.items()
    }


def _meta() -> list[dict[str, Any]]:
    return [
        {"doc_id": "dev_0000", "scanned": True, "multipage": False, "repeated_parts": False,
         "illegible": False, "waybill": False},
        {"doc_id": "dev_0001", "scanned": False, "multipage": True, "repeated_parts": True,
         "illegible": True, "waybill": False},
        {"doc_id": "dev_0002", "scanned": False, "multipage": False, "repeated_parts": False,
         "illegible": False, "waybill": True},
    ]  # fmt: skip


SCANNED = {"dev_0000": True, "dev_0001": False, "dev_0002": False}


def test_helpers_format_numbers() -> None:
    assert rs.pct(0.12345) == "12.35"
    assert rs.pct(None) == "n/a"
    assert rs.pct(float("nan")) == "n/a"
    assert rs.pci({"point": 0.5, "lo": 0.25, "hi": 0.75}) == "50.00 [25.00, 75.00]"
    assert rs.rate_str(1, 4) == "1/4 (25.0%)"
    assert rs.rate_str(0, 0) == "0/0 (n/a)"
    assert rs.ratio(1, 0) is None


def test_confusion_counts_precision_recall() -> None:
    cells = [
        {"gold_null": True, "pred_null": True},  # true null
        {"gold_null": True, "pred_null": False},  # false fill
        {"gold_null": False, "pred_null": True},  # over-null
        {"gold_null": False, "pred_null": True},  # over-null
        {"gold_null": False, "pred_null": False},
    ]
    c = rs.confusion(cells)
    assert (c["gold_null"], c["pred_null"], c["true_null"]) == (2, 3, 1)
    assert (c["over_null"], c["false_fill"], c["gold_valued"]) == (2, 1, 3)
    assert c["null_precision"] == pytest.approx(1 / 3)
    assert c["null_recall"] == pytest.approx(1 / 2)
    assert c["over_null_rate"] == pytest.approx(2 / 3)


def test_confusion_empty_is_none_not_zero() -> None:
    c = rs.confusion([{"gold_null": False, "pred_null": False}])
    assert c["null_precision"] is None and c["null_recall"] is None


def test_header_cells_classes_and_null_report() -> None:
    gold = _gold_set()
    pred = _pred_from_gold(gold)
    pred["dev_0000"]["header"]["invoice_number"] = None  # over-null on a required field
    pred["dev_0001"]["header"]["awb_number"] = "999-00000000"  # false fill on an optional line
    not_printed = {"invoice": {"awb_number"}, "waybill": {"hawb"}}
    cells = rs.header_cells(pred, gold, not_printed, SCANNED)
    assert len(cells) == 8 + 8 + 9
    rep = rs.header_null_report(cells)
    assert rep["required"]["over_null"] == 1
    assert rep["absent_line"]["false_fill"] == 1
    assert rep["pooled"]["gold_null"] == 2  # awb of doc 1 and the waybill hawb; no redactions
    assert rep["fields"]["invoice.invoice_number"]["scanned"]["over_null"] == 1
    assert rep["fields"]["invoice.invoice_number"]["digital"]["over_null"] == 0


def test_header_cells_follow_gold_type_when_pred_type_is_wrong() -> None:
    gold = _gold_set()
    pred = _pred_from_gold(gold)
    pred["dev_0002"] = {"doc_type": "invoice", "header": {}, "line_items": []}
    cells = [c for c in rs.header_cells(pred, gold, {}, SCANNED) if c["doc"] == "dev_0002"]
    assert {c["field"] for c in cells} == set(gold["dev_0002"]["header"])
    assert sum(c["pred_null"] for c in cells) == 9


def test_oracle_fill_header_fills_only_over_nulls_and_keeps_input() -> None:
    gold = _gold_set()
    pred = _pred_from_gold(gold)
    pred["dev_0000"]["header"]["buyer_name"] = None
    pred["dev_0001"]["header"]["awb_number"] = "999-00000000"  # false fill: untouched
    before = copy.deepcopy(pred)
    out, n = rs.oracle_fill(pred, gold, header=True)
    assert pred == before  # input not mutated
    assert n == {"header": 1, "row": 0, "blanked": 0}
    assert out["dev_0000"]["header"]["buyer_name"] == gold["dev_0000"]["header"]["buyer_name"]
    assert out["dev_0001"]["header"]["awb_number"] == "999-00000000"
    base = rs.ev.score(pred, gold)["all"]["OVERALL"]
    assert rs.ev.score(out, gold)["all"]["OVERALL"] > base


def test_oracle_row_fill_vs_null_status_when_po_sits_in_cpn_slot() -> None:
    gold = {"dev_0000": _inv_gold(0)}
    pred = _pred_from_gold(gold)
    # row 0: PO value written in the cpn slot, PO null (gold cpn null, gold PO valued)
    pred["dev_0000"]["line_items"][0][CPN] = "PO0"
    pred["dev_0000"]["line_items"][0][PO] = None
    base = rs.ev.score(pred, gold)["all"]
    assert base["row_f1"] < 1
    filled, nf = rs.oracle_fill(pred, gold, row_fields=(CPN, PO))
    assert nf["row"] == 1 and nf["blanked"] == 0
    assert rs.ev.score(filled, gold)["all"]["row_f1"] == base["row_f1"]  # cpn still a false fill
    both, nb = rs.oracle_fill(pred, gold, row_fields=(CPN, PO), blank_false_fill=(CPN, PO))
    assert nb == {"header": 0, "row": 1, "blanked": 1}
    assert rs.ev.score(both, gold)["all"]["row_f1"] == 1.0


def test_slot_misplacements_and_row_cells() -> None:
    gold = {"dev_0000": _inv_gold(0)}
    pred = _pred_from_gold(gold)
    pred["dev_0000"]["line_items"][0][CPN] = "PO0"
    pred["dev_0000"]["line_items"][0][PO] = None
    slots = rs.slot_misplacements(pred, gold, {"dev_0000": False})
    assert slots["digital"] == {
        "paired_rows": 2,
        "null_status_rows": 1,
        "po_in_cpn_slot": 1,
        "cpn_in_po_slot": 0,
    }
    cells, marg = rs.row_cells(pred, gold, {"dev_0000": False})
    rep = rs.row_null_report(cells, marg)
    assert rep["fields"][CPN]["false_fill"] == 1
    assert rep["fields"][PO]["over_null"] == 1
    assert marg["digital"]["gold_rows"] == 2 and marg["scanned"]["gold_rows"] == 0
    ff = rs.row_false_fill_counts(cells)
    assert ff[CPN]["false_fill"] == 1 and ff[PO]["over_null"] == 1


def test_over_null_sources_model_vs_postprocessing() -> None:
    cells = [
        {"doc": "a", "type": "waybill", "field": "mawb", "gold_null": False, "pred_null": True},
        {"doc": "b", "type": "waybill", "field": "mawb", "gold_null": False, "pred_null": True},
        {"doc": "c", "type": "waybill", "field": "mawb", "gold_null": False, "pred_null": True},
        {"doc": "a", "type": "waybill", "field": "hawb", "gold_null": True, "pred_null": True},
    ]
    traces = [
        {"doc_id": "a", "pages": [{"parsed": {"header": {"mawb": None}}}]},
        {"doc_id": "b", "pages": [{"parsed": {"header": {"mawb": "123-45678901"}}}]},
        {"doc_id": "c", "pages": [{"parsed": None}]},
    ]
    out = rs.over_null_sources(cells, traces)
    assert out == {
        "waybill.mawb": {"dropped_after_parse": 1, "model_null": 1, "page_parse_failed": 1}
    }


def test_parsed_vs_final_row_nulls_counts_values() -> None:
    pred = {"a": {"doc_type": "invoice", "line_items": [_row("X", None, "P1")]}}
    traces = [
        {
            "doc_id": "a",
            "pages": [{"parsed": {"line_items": [_row("X", "C1", "P1"), _row("Y", None, None)]}}],
        }
    ]
    out = rs.parsed_vs_final_row_nulls(pred, traces)
    assert out[CPN] == {"parsed_non_null": 1, "final_non_null": 0}
    assert out[PO] == {"parsed_non_null": 1, "final_non_null": 1}


def test_slice_ids_derive_awb_and_hawb_absent_from_gold() -> None:
    gold = _gold_set()
    sl = rs.slice_ids(gold, _meta())
    assert sl["awb_absent (invoice)"] == ["dev_0001"]
    assert sl["awb_present (invoice)"] == ["dev_0000"]
    assert sl["hawb_absent (waybill)"] == ["dev_0002"]
    assert sl["scanned"] == ["dev_0000"] and sl["digital"] == ["dev_0001", "dev_0002"]
    assert sl["waybill"] == ["dev_0002"] and sl["illegible"] == ["dev_0001"]


def test_slice_ids_missing_meta_fails_closed() -> None:
    with pytest.raises(ValueError, match="no meta"):
        rs.slice_ids(_gold_set(), _meta()[:2])


def test_compare_metrics_detects_difference() -> None:
    agg = {k: 0.5 for k in rs.CHECK_KEYS}
    saved = {"slices": {"all": dict(agg)}}
    assert rs.compare_metrics(agg, saved)["all_equal"] is True
    saved["slices"]["all"]["OVERALL"] = 0.5000001
    res = rs.compare_metrics(agg, saved)
    assert res["all_equal"] is False and res["keys"]["OVERALL"]["equal"] is False
    assert rs.compare_metrics(agg, {})["all_equal"] is False  # nothing to compare = not equal


def _ab(
    row_delta: float, row_lo: float, ov_delta: float, ov_lo: float, ov_hi: float, rk: int, rc: int
) -> dict[str, Any]:
    return {
        "deltas": {
            "row_f1": {"delta": row_delta, "ci95": [row_lo, row_lo + 0.2]},
            "OVERALL": {"delta": ov_delta, "ci95": [ov_lo, ov_hi]},
        },
        "keyed": {"rotations": rk},
        "compact": {"rotations": rc},
    }  # fmt: skip


@pytest.mark.parametrize(
    ("ab", "decision", "clause"),
    [
        (_ab(0.05, 0.01, -0.02, -0.05, 0.01, 9, 9), "keyed", "clause 1"),  # row F1 gain
        (_ab(-0.04, -0.2, -0.03, -0.09, 0.03, 3, 5), "keyed", "clause 2"),  # fewer rotations
        (_ab(0.05, -0.01, 0.0, -0.05, 0.05, 5, 5), "compact", "none"),  # CI includes 0, same rot
        (_ab(0.0, -0.1, -0.06, -0.09, -0.02, 3, 5), "compact", "none"),  # OVERALL worse for sure
    ],
)
def test_rederive_decision_matches_library_decide(
    ab: dict[str, Any], decision: str, clause: str
) -> None:
    out = rs.rederive_decision(ab)
    assert out["decision"] == decision and out["fired"].startswith(clause)
    lib, _ = rs.fab.decide(ab["deltas"], ab["keyed"]["rotations"], ab["compact"]["rotations"])
    assert lib == decision


def test_status_seconds_reads_status_files_and_skips_smoke(tmp_path: Path) -> None:
    (tmp_path / "ab_status.json").write_text(
        json.dumps({"r1": {"seconds": 12.5}}), encoding="utf-8"
    )
    (tmp_path / "smoke_status.json").write_text(json.dumps({"s": {"seconds": 1}}), encoding="utf-8")
    (tmp_path / "x_status.json").write_text(json.dumps({"r2": {"state": "x"}}), encoding="utf-8")
    assert rs.status_seconds(tmp_path) == {"r1": 12.5}


def test_run_identity_reads_format_and_raw_text_shape() -> None:
    page = {"raw_text": ' {"doc_type": "invoice"}', "json_valid": True, "meta": {}}
    trace = {
        "doc_id": "a",
        "output_format": "json",
        "pages": [page, {**page, "raw_text": '{"dt": "in"}'}],
        "config": {"name": "c", "hash": "h1"},
        "prompt": {"version": "v2", "hash": "p"},
        "git_commit": "2abf481deadbeef",
        "model": {"id": "m", "revision": "r"},
    }
    run = rs.RunData("run", Path("."), {}, [trace], {"output_format": "json"})
    ident = rs.run_identity(run)
    assert ident["raw_text_shape_pages"] == {"keyed": 1, "compact": 1, "other": 0}
    assert ident["git_commit"] == "2abf481" and ident["output_format_trace"] == ["json"]


def test_field_cis_has_point_inside_interval_and_handles_missing_fields() -> None:
    gold = _gold_set()
    pred = _pred_from_gold(gold)
    pred["dev_0000"]["header"]["buyer_name"] = "wrong"
    per = list(rs.ev.per_doc_results(pred, gold).values())
    fc = rs.field_cis(per, n=100, seed=1)
    h = fc["header"]["buyer_name"]
    assert h["lo"] <= h["point"] <= h["hi"] and h["n_docs"] == 2
    assert fc["header"]["mawb"]["n_docs"] == 1  # waybill-only field
    assert set(fc["row"]) == {"supplier_part_number", CPN, PO, "quantity"}


def test_tracked_report_contains_no_value_level_content(tmp_path: Path) -> None:
    """The aggregate renderer must not print a gold or predicted value from the cells."""
    gold = _gold_set()
    pred = _pred_from_gold(gold)
    pred["dev_0000"]["header"]["buyer_name"] = None
    cells = rs.header_cells(pred, gold, {"invoice": {"awb_number"}}, SCANNED)
    text = rs.header_null_tables(rs.header_null_report(cells))
    for g in gold.values():
        for v in g["header"].values():
            if isinstance(v, str) and len(v) >= 8:
                assert v not in text
