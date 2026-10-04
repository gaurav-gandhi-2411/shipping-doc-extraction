"""Unit tests for scripts/row_error_diagnosis.py on small synthetic fixtures."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
HAS_SCORER = (ROOT / "assignment" / "score.py").is_file()
needs_scorer = pytest.mark.skipif(not HAS_SCORER, reason="assignment/score.py absent")

_spec = importlib.util.spec_from_file_location(
    "row_error_diagnosis", ROOT / "scripts" / "row_error_diagnosis.py"
)
red = importlib.util.module_from_spec(_spec)
sys.modules["row_error_diagnosis"] = red  # dataclasses resolve annotations via sys.modules
_spec.loader.exec_module(red)

SPN, CPN, PO, QTY = red.SPN, red.CPN, red.PO, red.QTY


def same(field: str, a: Any, b: Any) -> bool:
    """Stand-in for the scorer's `same`: alphanumeric, case-insensitive equality."""
    na = re.sub(r"[^0-9a-z]", "", str(a).lower()) if a is not None else ""
    nb = re.sub(r"[^0-9a-z]", "", str(b).lower()) if b is not None else ""
    return bool(na) and na == nb


def row(spn: Any, cpn: Any = None, po: Any = None, qty: Any = "5") -> dict[str, Any]:
    return {SPN: spn, CPN: cpn, PO: po, QTY: qty}


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------


def test_ranks_and_near_boundary() -> None:
    pages = [0] * 8 + [1] * 2 + [None]
    rk, cnt = red.ranks(pages)
    assert cnt == {0: 8, 1: 2}
    assert rk[:8] == list(range(8)) and rk[8:10] == [0, 1] and rk[10] is None
    near = [red.near_boundary(rk[i], cnt.get(pages[i], 0)) for i in range(11)]
    # first / last 3 of an 8-row page: ranks 0,1,2 and 5,6,7 are near, 3 and 4 are interior
    assert near[:8] == [True, True, True, False, False, True, True, True]
    assert near[8:10] == [True, True]  # a 2-row page is all boundary
    assert near[10] is None  # unknown page


def test_length_bucket_and_label_only() -> None:
    assert [red.length_bucket(n) for n in (1, 5, 6, 15, 16, 25, 26, 90)] == [
        "1-5", "1-5", "6-15", "6-15", "16-25", "16-25", "26+", "26+",
    ]  # fmt: skip
    assert red.label_only("Part No.") and red.label_only("Customer PO")
    assert not red.label_only("PO5551230001")  # a digit means data
    assert not red.label_only(None) and not red.label_only("Widget")


def test_md_table_rate_pts() -> None:
    assert red.md_table(["a", "b"], [[1, 2]]) == "| a | b |\n|---|---|\n| 1 | 2 |\n"
    assert red.rate(1, 4) == "1/4 (25.0%)" and red.rate(0, 0) == "0/0 (n/a)"
    assert red.pts(0.0123) == "+1.23" and red.pts(-0.005) == "-0.50"


def test_edit_profile() -> None:
    assert red.edit_profile("QW/777/1G", "QW/777/IG") == "glyph_confusion"
    assert red.edit_profile("ZKT5521K10", "ZKT5521K1O") == "glyph_confusion"
    assert red.edit_profile("MTRV30471W", "MTRV3047I1W") == "char_dropped_or_added"
    assert red.edit_profile("TT/55120-9", "TT/55120/9") == "punctuation_only"
    assert red.edit_profile("ABC7", "ABX9") == "other_substitution"
    assert red.edit_profile("ABCD7", "ABC9") == "mixed"


def test_shapes() -> None:
    assert red.shape_of("PO9100000001") == "AA9999999999"
    assert red.shape_of("B91000002-009") == "A99999999-999"
    labels = [
        {
            "doc_type": "invoice",
            "line_items": [row("S1", "CP-123", "PO55"), row("S2", "99-12", "PO77")],
        },
        {"doc_type": "waybill", "line_items": []},
    ]
    sh = red.learn_slot_shapes(labels)
    assert sh["cpn_only"] == ["99-99", "AA-999"] and sh["po_only"] == ["AA99"]
    assert sh["n_overlap"] == 0


def test_group_columns_and_state() -> None:
    labels = [
        {
            "doc_id": "a",
            "doc_type": "invoice",
            "line_items": [row("S", None, "P"), row("T", None, "Q")],
        },
        {
            "doc_id": "b",
            "doc_type": "invoice",
            "line_items": [row("S", "C", "P"), row("T", None, "Q")],
        },
        {"doc_id": "w", "doc_type": "waybill", "line_items": []},
    ]
    gc = red.group_columns(labels, {"a": "inv_g01", "b": "inv_g02", "w": "wb_g01"})
    assert gc["inv_g01"] == {"docs": 1, "rows": 2, "cpn": 0, "po": 2}
    assert red.column_state(0, 2) == "not printed"
    assert red.column_state(2, 2) == "printed"
    assert red.column_state(1, 2) == "mixed"
    assert red.column_state(0, 0) == "n/a"
    assert "wb_g01" not in gc


# --------------------------------------------------------------------------------------------
# linking and classification
# --------------------------------------------------------------------------------------------


def test_link_by_identifier_then_quantity_then_position() -> None:
    gr = [
        row("AAA-1", "C1", "P1", "10"),
        row("BBB-2", None, None, "20"),
        row("CCC-3", None, None, "30"),
    ]
    pr = [
        row("AAA-I", "C1", "P1", "10"),
        row("BBB-Z", None, None, "20"),
        row("CCC-9", None, None, "31"),
    ]
    pages = [0, 0, 0]
    links, lg, lp = red.link_unpaired(gr, pr, [0, 1, 2], [0, 1, 2], pages, pages, same)
    assert [(lk.gi, lk.pi, lk.kind) for lk in links] == [
        (0, 0, "ident"), (1, 1, "qty"), (2, 2, "position"),
    ]  # fmt: skip
    assert lg == [] and lp == []


def test_link_leaves_unequal_leftovers_unlinked() -> None:
    gr = [row("A1", None, None, "1"), row("B2", None, None, "2")]
    pr = [row("Z9", None, None, "9")]
    pages = [0, 0]
    links, lg, lp = red.link_unpaired(gr, pr, [0, 1], [0], pages, [0], same)
    assert links == [] and lg == [0, 1] and lp == [0]  # 2 vs 1 on the page: no positional guess


def test_link_prefers_same_page() -> None:
    gr = [row("A1", "C1", None), row("A2", "C1", None)]
    pr = [row("Q1", "C1", None)]
    links, lg, _ = red.link_unpaired(gr, pr, [0, 1], [0], [0, 1], [1], same)
    assert [(lk.gi, lk.pi) for lk in links] == [(1, 0)] and lg == [0]


def test_classify_pair_causes() -> None:
    g = row("ZZQ/501/XYZ", "11-2233-44", "PO9100000002", "3000")
    shifted = row("PO9100000002", "ZZQ/501/XYZ", None, "3000")
    assert red.classify_pair(g, shifted, "ident", same) == "column_shift"
    copy_po = row("PO9100000002", "11-2233-44", "PO9100000002", "3000")
    assert red.classify_pair(g, copy_po, "ident", same) == "spn_copies_other_slot"
    misread = row("ZZQ/501/XY2", "11-2233-44", "PO9100000002", "3000")
    assert red.classify_pair(g, misread, "ident", same) == "spn_misread"
    assert (
        red.classify_pair(g, row(None, "11-2233-44", "PO9100000002"), "ident", same) == "spn_null"
    )
    far = row("ZZZZZZZZZZ", "11-2233-44", "PO9100000002", "3000")
    assert red.classify_pair(g, far, "ident", same) == "spn_other"
    # linked by quantity only and not a near misread -> weak link, by position likewise
    g2, p2 = row("AAAAAAAA", qty="7"), row("QQQQQQQQ", qty="7")
    assert red.classify_pair(g2, p2, "qty", same) == "weak_link_qty"
    assert red.classify_pair(g2, p2, "position", same) == "weak_link_position"
    # a near misread linked only by quantity is still a misread (the part number decides)
    assert (
        red.classify_pair(row("ZKT5521K1O", qty="9"), row("ZKT5521K10", qty="9"), "qty", same)
        == "spn_misread"
    )


def test_classify_gold_only() -> None:
    gr = [row("A1", qty="2"), row("A1", qty="3"), row("B2", qty="5")]
    pr = [row("A1", qty="5")]
    common = dict(gr=gr, pr=pr, same=same)
    # page 1 has no predicted rows at all
    assert (
        red.classify_gold_only(
            2,
            gpage=[0, 0, 1],
            pred_rows_on_page={0: 1},
            gold_rows_on_page={0: 2, 1: 1},
            linked_pred={},
            truncated=False,
            **common,
        )
        == "page_dropped"
    )
    assert (
        red.classify_gold_only(
            2,
            gpage=[0, 0, 1],
            pred_rows_on_page={0: 1},
            gold_rows_on_page={0: 2, 1: 1},
            linked_pred={},
            truncated=True,
            **common,
        )
        == "truncated_page"
    )
    # predicted row 0 is linked to gold 0 and carries qty 2 + 3 = 5 -> gold row 1 was merged into it
    assert (
        red.classify_gold_only(
            1,
            gpage=[0, 0, 0],
            pred_rows_on_page={0: 1},
            gold_rows_on_page={0: 3},
            linked_pred={0: 0},
            truncated=False,
            **common,
        )
        == "merged"
    )
    # no merge pattern, page has fewer predicted than gold rows
    assert (
        red.classify_gold_only(
            2,
            gpage=[0, 0, 0],
            pred_rows_on_page={0: 1},
            gold_rows_on_page={0: 3},
            linked_pred={},
            truncated=False,
            **common,
        )
        == "page_short"
    )
    assert (
        red.classify_gold_only(
            2,
            gpage=[0, 0, 0],
            pred_rows_on_page={0: 3},
            gold_rows_on_page={0: 3},
            linked_pred={},
            truncated=False,
            **common,
        )
        == "missing_other"
    )


def test_classify_pred_only() -> None:
    # header row: values are table-label words
    pr = [row("Part No.", None, None, "5")]
    assert red.classify_pred_only(0, pr, [0], [], {}, same) == "header_row"
    # duplicate boundary: same part number at the end of page 0 and the start of page 1
    pr = [row("X0")] * 1 + [row("X1")] * 6 + [row("DUP")] + [row("DUP")] + [row("X2")] * 6
    pages = [0] * 8 + [1] * 7
    gr = [row("DUP")]
    assert red.classify_pred_only(8, pr, pages, gr, {}, same) == "duplicate_boundary"
    # same twin far from a boundary -> duplicate_other
    pr2 = [row("DUP")] + [row(f"Y{i}") for i in range(10)] + [row("DUP")]
    assert red.classify_pred_only(11, pr2, [0] * 12, gr, {}, same) == "duplicate_other"
    # split: two predicted rows, quantities add up to the single gold row
    gr = [row("SPLIT", qty="10")]
    pr3 = [row("SPLIT", qty="4"), row("SPLIT", qty="6")]
    assert red.classify_pred_only(1, pr3, [0, 0], gr, {0: 0}, same) == "split"
    assert red.classify_pred_only(0, [row("NOPE", qty="1")], [0], gr, {}, same) == "extra_other"


# --------------------------------------------------------------------------------------------
# predicted row pages
# --------------------------------------------------------------------------------------------


def _trace(*pages: list[dict[str, Any]]) -> dict[str, Any]:
    return {"pages": [{"parsed": {"line_items": items}} for items in pages]}


def test_predicted_row_pages_drops_what_merge_drops() -> None:
    tr = _trace(
        [row("A1", "C", "P"), row("Part No.", None, None, None)],  # second is a header row
        [row(None, None, None, None), row("B1", "C", "P")],  # first is all-null
    )
    assert red.predicted_row_pages(tr, 2) == [0, 1]
    assert red.predicted_row_pages(tr, 3) == [None, None, None]  # replay mismatch: never guess


# --------------------------------------------------------------------------------------------
# ORACLE patching
# --------------------------------------------------------------------------------------------


def _unit(**kw: Any) -> Any:
    base = dict(doc="d1", group="inv_g01", scanned=True, repeated=False, n_pages=1, n_gold_rows=2)
    return red.Unit(**{**base, **kw})


def test_apply_oracle_pair_gold_only_pred_only_slot_header() -> None:
    gold = {
        "d1": {"doc_type": "invoice", "line_items": [row("G0", "C0", "P0"), row("G1", None, "P1")]},
        "w1": {"doc_type": "waybill", "header": {"mawb": "123-45678901", "carrier": "Acme"}},
    }
    pred = {
        "d1": {
            "doc_type": "invoice",
            "line_items": [row("X0", "C0", "P0"), row("G1", "P1", None), row("EXTRA")],
        },
        "w1": {"doc_type": "waybill", "header": {"mawb": None, "carrier": "Acme"}},
    }
    units = [
        _unit(side="pair", cause="spn_misread", gi=0, pi=0),
        _unit(side="pred", cause="extra_other", pi=2),
        _unit(side="gold", cause="page_short", gi=1),
    ]
    slots = [{"doc": "d1", "pi": 1, "gi": 1, "kind": "po_in_cpn_slot"}]
    spec = red.OracleSpec(
        causes=frozenset({"spn_misread", "extra_other", "page_short"}),
        slot_kinds=frozenset({"po_in_cpn_slot"}),
        header_fields=frozenset({"mawb"}),
    )
    out = red.apply_oracle(pred, gold, units, slots, spec)
    items = out["d1"]["line_items"]
    assert items[0][SPN] == "G0"  # pair replaced by the gold row
    assert items[1][CPN] is None and items[1][PO] == "P1"  # slot fixed
    assert [r[SPN] for r in items] == ["G0", "G1", "G1"]  # EXTRA dropped, gold-only row appended
    assert out["w1"]["header"]["mawb"] == "123-45678901"
    assert pred["d1"]["line_items"][0][SPN] == "X0"  # the input is never mutated
    # only the named causes are touched
    only = red.apply_oracle(
        pred, gold, units, [], red.OracleSpec(causes=frozenset({"extra_other"}))
    )
    assert [r[SPN] for r in only["d1"]["line_items"]] == ["X0", "G1"]


# --------------------------------------------------------------------------------------------
# candidate rules (no scorer needed for R1 / R2)
# --------------------------------------------------------------------------------------------


def test_rule_carrier_from_supplier() -> None:
    gold = {"w1": {"doc_type": "waybill", "header": {"carrier": "Acme Air"}}}
    pred = {"w1": {"doc_type": "waybill", "header": {"carrier": None}}}
    traces = {"w1": {"pages": [{"parsed": {"header": {"supplier_name": "ACME AIR"}}}]}}
    out, rec = red.rule_carrier_from_supplier(pred, gold, traces, same)
    assert out["w1"]["header"]["carrier"] == "ACME AIR"
    assert rec == [{"before_ok": False, "after_ok": True}]
    assert pred["w1"]["header"]["carrier"] is None


def test_rule_pattern_backfill_unique_only(monkeypatch: pytest.MonkeyPatch) -> None:
    gold = {
        "w1": {"doc_type": "waybill", "header": {"mawb": "999-00000001", "hawb": "QQ00000001"}},
        "w2": {"doc_type": "waybill", "header": {"mawb": "999-00000002", "hawb": None}},
    }
    pred = {
        "w1": {"doc_type": "waybill", "header": {"mawb": None, "hawb": None}},
        "w2": {"doc_type": "waybill", "header": {"mawb": None, "hawb": None}},
    }
    texts = {"w1": "MAWB 999-00000001 HAWB QQ00000001", "w2": "AWB 999-00000002 and 333-44444444"}
    monkeypatch.setattr(red, "_doc_text", lambda idx: texts[idx])
    out, rec = red.rule_pattern_backfill(pred, gold, {"w1": "w1", "w2": "w2"}, same)
    assert out["w1"]["header"] == {"mawb": "999-00000001", "hawb": "QQ00000001"}
    assert out["w2"]["header"]["mawb"] is None  # two candidates: no guess
    assert out["w2"]["header"]["hawb"] is None  # no match and gold null: stays null
    assert len(rec) == 2 and all(r["after_ok"] for r in rec)


def test_pattern_backfill_check(monkeypatch: pytest.MonkeyPatch) -> None:
    class Page:
        def __init__(self, text: str) -> None:
            self.text = text

    monkeypatch.setattr(red, "page_text", lambda p: p.text)
    labels = [
        {"doc_id": "a", "doc_type": "waybill", "header": {"mawb": "999-00000001", "hawb": None}},
        {"doc_id": "b", "doc_type": "waybill", "header": {"mawb": "999-00000002", "hawb": None}},
    ]
    ocr = {"a": [Page("999-00000001 AB12345678")], "b": [Page("no numbers here")]}
    out = red.pattern_backfill_check(labels, ocr, same)
    assert out["mawb"] == {"gold_docs": 2, "unique_right": 1, "no_match": 1}
    assert out["hawb"]["gold_null_docs"] == 2 and out["hawb"]["gold_null_unique_match"] == 1


# --------------------------------------------------------------------------------------------
# document-level diagnosis against the real scorer
# --------------------------------------------------------------------------------------------


@needs_scorer
def test_diagnose_doc_partitions_exactly_the_unpaired_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    from shipdoc import eval as ev

    sc = ev.load_scorer()
    gold_rows = [
        row("AAA-100", "CP-1", "PO1", "10"),
        row("BBB-200", "CP-2", "PO2", "20"),
        row("CCC-300", "CP-3", "PO3", "30"),  # page 2: shifted by the model
        row("DDD-400", "CP-4", "PO4", "40"),  # page 2: shifted by the model
        row("EEE-500", None, None, "50"),  # page 2: spn misread
    ]
    pred_rows = [
        row("AAA-100", "CP-1", "PO1", "10"),  # right
        row("BBB-20O", "CP-2", "PO2", "20"),  # misread (0 -> O)
        row("PO3", "CCC-300", None, "30"),  # shifted
        row("PO4", "DDD-400", None, "40"),  # shifted
        row("EEE-500", None, None, "50"),  # right
    ]
    gold = {"doc_type": "invoice", "pages": ["a_p1.png", "a_p2.png"], "line_items": gold_rows}
    pred = {"doc_type": "invoice", "line_items": pred_rows}
    trace = _trace(pred_rows[:2], pred_rows[2:])
    monkeypatch.setattr(
        red, "gold_row_pages", lambda g, idx: ([0, 0, 1, 1, 1], ["line"] * 5, ["exact"] * 5)
    )
    monkeypatch.setattr(
        red,
        "spn_evidence",
        lambda *a, **k: {
            "edit": "x",
            "snap": "no_candidate",
            "gold_in_ocr": "exact",
            "pred_in_ocr": False,
        },
    )
    ctx = {"group": "inv_g99", "scanned": False, "repeated": False}
    units = red.diagnose_doc("d", pred, gold, trace, object(), ctx, sc)
    _, _, free_p, free_g = ev._pair_rows(sc, pred_rows, gold_rows)
    # invariant: units cover every unpaired gold row once and every unpaired predicted row once
    assert sorted(u.gi for u in units if u.side in ("pair", "gold")) == sorted(free_g)
    assert sorted(u.pi for u in units if u.side in ("pair", "pred")) == sorted(free_p)
    causes = {u.gi: u.cause for u in units}
    assert causes[1] == "spn_misread" and causes[2] == causes[3] == "column_shift"
    assert all(u.g_pos in ("first", "last") for u in units)
    assert {u.gi: u.g_pos for u in units}[2] == "last"


@needs_scorer
def test_rule_slot_shape_moves_only_other_column_shapes() -> None:
    from shipdoc import eval as ev

    sc = ev.load_scorer()
    shapes = {"cpn_only": ["AA-999999"], "po_only": ["AA9999999999"]}
    gold = {
        "d": {
            "doc_type": "invoice",
            "line_items": [row("S1", None, "PO9100000001"), row("S2", "CP-123456", None)],
        }
    }
    pred = {
        "d": {
            "doc_type": "invoice",
            "line_items": [row("S1", "PO9100000001", None), row("S2", "CP-123456", None)],
        }
    }
    out, rec = red.rule_slot_shape(pred, gold, shapes, sc)
    first, second = out["d"]["line_items"]
    assert first[CPN] is None and first[PO] == "PO9100000001"
    assert second[CPN] == "CP-123456" and second[PO] is None  # a real cpn is left alone
    assert rec == [{"before_ok": False, "after_ok": True}]


# --------------------------------------------------------------------------------------------
# the committed report must not carry gold values
# --------------------------------------------------------------------------------------------


def test_committed_report_has_no_gold_values() -> None:
    report = ROOT / "reports" / "row_errors.md"
    labels = ROOT / "data" / "dev" / "labels"
    if not (report.is_file() and labels.is_dir()):
        pytest.skip("report or data/ absent")
    text = report.read_text(encoding="utf-8")
    leaked = []
    for f in sorted(labels.glob("*.json")):
        g = json.loads(f.read_text(encoding="utf-8"))
        vals = [v for v in g["header"].values() if isinstance(v, str)]
        vals += [
            v
            for r in g.get("line_items") or []
            for k, v in r.items()
            if k in (SPN, CPN, PO) and isinstance(v, str)
        ]
        leaked += [v for v in vals if len(v) >= 6 and v in text]
    assert not leaked, f"{len(leaked)} gold values appear in reports/row_errors.md"
