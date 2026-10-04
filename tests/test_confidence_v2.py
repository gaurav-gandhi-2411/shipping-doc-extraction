"""Tests for shipdoc.confidence_v2 and scripts/calibrate_v2.py (synthetic data, no real I/O)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from shipdoc import confidence as cf
from shipdoc import confidence_v2 as c2
from shipdoc.ocr import OcrItem, PageOcr

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("calibrate_v2", ROOT / "scripts" / "calibrate_v2.py")
cal2 = importlib.util.module_from_spec(_spec)
sys.modules["calibrate_v2"] = cal2
_spec.loader.exec_module(cal2)
needs_scorer = pytest.mark.skipif(
    not (ROOT / "assignment" / "score.py").is_file(), reason="assignment/score.py absent"
)
X = c2.EXTRA_FEATURES


def _item(text: str, x0: float, y0: float, x1: float, y1: float) -> OcrItem:
    return OcrItem(text, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], 0.9)


def _page(lines: list[list[str]]) -> PageOcr:
    """One OCR line per entry, cells laid out left to right."""
    items = []
    for r, cells in enumerate(lines):
        for c, t in enumerate(cells):
            items.append(
                _item(t, 50 + 220 * c, 100 + 50 * r, 50 + 220 * c + 15 * len(t), 120 + 50 * r)
            )
    return PageOcr("d_0000_p1.png", 1240, 1754, "paddleocr", "line", 0, items=items)


HDR_LINE = ["Item", "Customer Part", "PO No", "Qty", "Description"]


def _row(spn: str | None, cpn: str | None, po: str | None, qty: str | None) -> dict[str, Any]:
    return {
        "supplier_part_number": spn,
        "customer_part_number": cpn,
        "purchase_order": po,
        "quantity": qty,
    }


def _trace(n_rows_per_page: list[int]) -> dict[str, Any]:
    pages = []
    for n in n_rows_per_page:
        rows = [_row(f"AB-{i}", None, None, "1") for i in range(n)]
        pages.append({"image": "p.png", "parsed": {"header": {}, "line_items": rows}})
    return {"pages": pages}


def _doc(rows: list[dict[str, Any]]) -> dict[str, Any]:
    head = dict.fromkeys(cf.HEADER_FIELDS["invoice"])
    return {"doc_type": "invoice", "header": head, "line_items": rows}


def _extras(
    rows: list[dict[str, Any]],
    pages: list[PageOcr] | None,
    n_per_page: list[int],
    changes: list[dict[str, Any]] | None = None,
) -> tuple[list[cf.FieldRow], np.ndarray]:
    doc, tr = _doc(rows), _trace(n_per_page)
    fr = cf.build_doc_features("d_0000", doc, tr, pages)
    return fr, c2.build_extras(fr, doc, tr, pages, changes or [])


def _get(
    fr: list[cf.FieldRow], ex: np.ndarray, scope: str, name: str, ridx: int, feat: str
) -> float:
    i = next(i for i, r in enumerate(fr) if r.key[1:] == (scope, name, ridx))
    return float(ex[i, X.index(feat)])


def test_column_presence_regexes() -> None:
    got = c2.column_presence(["Item", "Customer Part", "PO No", "Qty"])
    assert got == {"cpn": 1.0, "po": 1.0, "qty": 1.0, "spn": 1.0}
    only_qty = c2.column_presence(["Description", "Quantity"])
    assert only_qty == {"cpn": 0.0, "po": 0.0, "qty": 1.0, "spn": 0.0}
    cust_only = c2.column_presence(["Your Part No"])
    assert cust_only["cpn"] == 1.0 and cust_only["spn"] == 0.0  # part-like but customer-like


def test_header_columns_from_the_row_page_with_page1_fallback_and_none() -> None:
    rows = [_row("A1", None, None, "1"), _row("A2", None, None, "2"), _row("A3", None, None, "3")]
    p1 = _page([["Invoice"], HDR_LINE, ["a", "b", "c", "4", "x"], ["d", "e", "f", "5", "y"]])
    p2 = _page([["Continued"], ["z", "z", "z", "3", "x"], ["z", "z", "z", "4", "x"]])
    fr, ex = _extras(rows, [p1, p2], [2, 1])
    assert _get(fr, ex, "row", "quantity", 0, "hc_qty") == 1.0
    assert _get(fr, ex, "row", "quantity", 0, "hc_fallback_p1") == 0.0
    # row 2 sits on page 2 (no table header line): page 1's header is used, flagged as fallback
    assert _get(fr, ex, "row", "customer_part_number", 2, "hc_own") == 1.0
    assert _get(fr, ex, "row", "customer_part_number", 2, "hc_fallback_p1") == 1.0
    fr, ex = _extras(rows, [p2, p2], [2, 1])
    assert _get(fr, ex, "row", "quantity", 0, "hc_none") == 1.0
    assert _get(fr, ex, "row", "quantity", 0, "hc_own") == 0.0
    fr, ex = _extras(rows, None, [2, 1])  # no OCR at all
    assert _get(fr, ex, "row", "quantity", 0, "hc_none") == 1.0


def test_rule_touch_map_and_flags() -> None:
    ch = [
        {"rule": "R3", "field": "cpn_po", "row": 1, "kind": "swap"},
        {"rule": "R1", "field": "carrier", "row": None, "kind": "fill"},
        {"rule": "R2", "field": "mawb", "row": None, "kind": "fill"},
    ]
    m = c2.rule_touch_map(ch)
    assert m[("row", "customer_part_number", 1)] == "R3"
    assert m[("row", "purchase_order", 1)] == "R3"
    assert m[("header", "carrier", -1)] == "R1" and m[("header", "mawb", -1)] == "R2"
    rows = [_row("A1", "C1", None, "1"), _row("A2", None, "P9", "2")]
    fr, ex = _extras(rows, None, [2], ch)
    assert _get(fr, ex, "row", "customer_part_number", 1, "rt_R3") == 1.0
    assert _get(fr, ex, "row", "purchase_order", 1, "rt_touched") == 1.0
    assert _get(fr, ex, "row", "quantity", 1, "rt_touched") == 0.0
    assert _get(fr, ex, "row", "customer_part_number", 0, "rt_touched") == 0.0
    assert _get(fr, ex, "header", "invoice_number", -1, "rt_R1") == 0.0


def test_shape_agreement_and_row_position() -> None:
    rows = [_row(f"AB-{i}", None, None, "12") for i in range(3)] + [_row("ZZ", None, None, "7")]
    rows[3]["supplier_part_number"] = "12345"  # a different shape in the column
    fr, ex = _extras(rows, None, [3, 1])
    # row 0: others on page 1 are AB-1 (same shape AA-9), AB-2 (same) -> 2 others, all same
    assert _get(fr, ex, "row", "supplier_part_number", 0, "sh_frac_same") == 1.0
    assert _get(fr, ex, "row", "supplier_part_number", 0, "sh_dom_share") == 1.0
    assert _get(fr, ex, "row", "supplier_part_number", 0, "sh_lt2") == 0.0
    # row 3 is alone on page 2: fewer than 2 others -> NaN and indicator
    assert np.isnan(_get(fr, ex, "row", "supplier_part_number", 3, "sh_frac_same"))
    assert _get(fr, ex, "row", "supplier_part_number", 3, "sh_lt2") == 1.0
    assert _get(fr, ex, "row", "supplier_part_number", 3, "pos_cont_page") == 1.0
    assert _get(fr, ex, "row", "supplier_part_number", 2, "pos_idx_in_page") == 2.0
    assert _get(fr, ex, "row", "supplier_part_number", 0, "pos_first_page") == 1.0
    assert _get(fr, ex, "row", "supplier_part_number", 0, "pos_n_rows") == 4.0
    assert np.isnan(_get(fr, ex, "header", "invoice_number", -1, "pos_first_page"))
    # a mixed column: 2 same + 1 different -> frac 0.5 for an AB row among {AB, AB, 12345}
    rows2 = [
        _row("AB-1", None, None, "1"),
        _row("AB-2", None, None, "1"),
        _row("12345", None, None, "1"),
    ]
    fr, ex = _extras(rows2, None, [3])
    assert _get(fr, ex, "row", "supplier_part_number", 0, "sh_frac_same") == 0.5
    assert _get(fr, ex, "row", "supplier_part_number", 0, "sh_dom_share") == pytest.approx(2 / 3)


def test_unmapped_rows_fall_back_without_inventing_a_page() -> None:
    rows = [_row("A1", None, None, "1"), _row("A2", None, None, "1")]
    fr, ex = _extras(rows, None, [3])  # 3 merged rows vs 2 predicted: not mapped
    assert _get(fr, ex, "row", "quantity", 0, "pos_unmapped") == 1.0
    assert np.isnan(_get(fr, ex, "row", "quantity", 0, "pos_first_page"))


def test_nested_tau_never_uses_the_evaluated_fold() -> None:
    rng = np.random.Generator(np.random.PCG64(42))
    n = 600
    fold_of = np.repeat([0, 1, 2], n // 3)
    conf = rng.uniform(0, 1, n)
    correct = rng.uniform(0, 1, n) < conf
    mask = np.ones(n, dtype=bool)
    acc, taus, used = c2.nested_accept(conf, correct, fold_of, mask, 0.9)
    for k, sel in used.items():
        assert not (fold_of[sel] == k).any()
        assert set(fold_of[sel].tolist()) == {0, 1, 2} - {k}
    # corrupting the evaluated fold's labels must not move its tau
    bad = correct.copy()
    bad[fold_of == 1] = ~bad[fold_of == 1]
    _, taus2, _ = c2.nested_accept(conf, bad, fold_of, mask, 0.9)
    assert taus2[1] == taus[1]
    assert taus2[0] != taus[0] or taus2[2] != taus[2]
    # an unreachable target accepts nothing
    acc0, taus0, _ = c2.nested_accept(conf, np.zeros(n, dtype=bool), fold_of, mask, 0.9)
    assert not acc0.any() and all(t is None for t in taus0.values())


def test_auroc_matches_rank_formula_and_is_deterministic() -> None:
    s = np.array([0.1, 0.4, 0.35, 0.8, 0.8])
    y = np.array([0, 0, 1, 1, 0])
    # pairs (pos, neg): 0.35 vs {0.1 win, 0.4 loss, 0.8 loss}, 0.8 vs {0.1, 0.4 win; 0.8 tie}
    assert c2.auroc(s, y) == pytest.approx((1 + 0 + 0 + 1 + 1 + 0.5) / 6)
    assert np.isnan(c2.auroc(s, np.ones(5, dtype=int)))
    rng = np.random.Generator(np.random.PCG64(1))
    sc = rng.uniform(size=200)
    yy = (rng.uniform(size=200) < sc).astype(int)
    docs = np.array([f"d{i // 4}" for i in range(200)], dtype=object)
    a = c2.auroc_boot(sc, yy, docs, n_boot=200)
    b = c2.auroc_boot(sc, yy, docs, n_boot=200)
    assert a == b and a["lo"] < a["point"] < a["hi"]
    assert a["point"] == pytest.approx(c2.auroc(sc, yy))


def test_fail_closed_on_test_ids() -> None:
    with pytest.raises(c2.NoTestDataError):
        c2.assert_no_test_ids(["train_0001", "test_0003"])
    c2.assert_no_test_ids(["train_0001", "dev_0002"])
    tv = c2.TableV2(
        cf.assemble([cf.FieldRow(cf.FieldKey("test_0001", "header", "currency", -1), True, {})]),
        np.zeros((1, len(X))),
    )
    with pytest.raises(c2.NoTestDataError):
        c2.doc_table(tv, np.array([0.5]), ["test_0001"])


@needs_scorer
def test_doc_label_is_the_scorers_fully_correct() -> None:
    from shipdoc import eval as ev

    sc = ev.load_scorer()
    inv = list(sc.HEADER["invoice"])
    gold_rows = [_row("A1", "C1", "P1", "5"), _row("A2", None, None, "6")]
    gold = {
        "doc_id": "d",
        "doc_type": "invoice",
        "header": {
            k: "2024-01-02" if k == "invoice_date" else "12" if sc.KIND.get(k) == "num" else "x1"
            for k in inv
        },
        "line_items": gold_rows,
    }
    ok = {"doc_type": "invoice", "header": dict(gold["header"]), "line_items": list(gold_rows)}
    assert c2.doc_exact(sc, ok, gold) is True
    missing = {**ok, "line_items": gold_rows[:1]}
    extra = {**ok, "line_items": [*gold_rows, _row("A3", None, None, "1")]}
    wrong_hdr = {**ok, "header": {**gold["header"], "currency": "zz"}}
    wrong_type = {**ok, "doc_type": "waybill"}
    for bad in (missing, extra, wrong_hdr, wrong_type):
        assert c2.doc_exact(sc, bad, gold) is False
    assert c2.doc_exact(sc, ok, gold) == sc.score_doc(ok, gold)["exact"]


def _synthetic_tv(
    n_docs: int = 60,
) -> tuple[c2.TableV2, np.ndarray, dict[str, int], dict[str, str]]:
    rng = np.random.Generator(np.random.PCG64(42))
    keys: list[cf.FieldKey] = []
    doc_fold: dict[str, int] = {}
    groups: dict[str, str] = {}
    for d in range(n_docs):
        did = f"train_{d:04d}"
        doc_fold[did], groups[did] = d % 3, f"g{d % 3}_{d % 4}"
        keys += [cf.FieldKey(did, "header", "currency", -1)]
        keys += [cf.FieldKey(did, "row", f, 0) for f in ("supplier_part_number", "quantity")]
        keys += [cf.FieldKey(did, "row", f, 1) for f in ("supplier_part_number", "quantity")]
    n = len(keys)
    base = np.zeros((n, len(cf.BASE_FEATURES)))
    base[:, cf.BASE_FEATURES.index("ocr_fuzzy")] = rng.uniform(size=n)
    base[:, cf.BASE_FEATURES.index("has_ocr")] = 1.0
    table = cf.FieldTable(keys, base, np.ones(n, dtype=bool))
    y = (rng.uniform(size=n) < 0.3 + 0.7 * base[:, cf.BASE_FEATURES.index("ocr_fuzzy")]).astype(
        bool
    )
    return c2.TableV2(table, np.zeros((n, len(X)))), y, doc_fold, groups


def test_models_cross_fit_by_fold_and_are_deterministic() -> None:
    tv, y, doc_fold, groups = _synthetic_tv()
    a = c2.cross_fit_v2(tv, y, tv.emitted, doc_fold, "lr", None, groups, "per_type")
    b = c2.cross_fit_v2(tv, y, tv.emitted, doc_fold, "lr", None, groups, "per_type")
    assert np.array_equal(a, b) and not np.isnan(a).any()
    ch = c2.select_structure(tv, y, doc_fold, None, groups)
    assert ch.structure in ("pooled", "per_type") and ch.kind in ("lr", "gbm")
    assert set(ch.logloss) == {"pooled/lr", "pooled/gbm", "per_type/lr", "per_type/gbm"}
    # doc level: one row per doc, OOF probabilities in (0, 1)
    order = sorted(doc_fold)
    Xd, o = c2.doc_table(tv, ch.oof, order)
    assert Xd.shape == (len(order), len(c2.DOC_FEATURES)) and o == order
    yd = np.arange(len(order)) % 2 == 0
    pd_ = c2.cross_fit_docs(Xd, yd, order, doc_fold, None, groups)
    assert ((pd_ >= 0) & (pd_ <= 1)).all()


def test_best_precision_and_coverage_at_target() -> None:
    conf = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    ok = np.array([1, 1, 1, 1, 0, 1, 0, 0, 0, 0], dtype=bool)
    assert c2.coverage_at_target(conf, ok, 0.99) == pytest.approx(0.4)
    assert c2.coverage_at_target(conf, np.zeros(10, dtype=bool), 0.5) == 0.0
    best = c2.best_precision_at_coverage(conf, ok, 0.2)
    assert best["precision"] == 1.0 and best["coverage"] == pytest.approx(0.4)


def test_csv_outputs_hold_no_values(tmp_path: Path) -> None:
    secret = "SECRET-VALUE-123"
    tv, y, doc_fold, groups = _synthetic_tv(6)
    meta = {d: {"waybill": False} for d in doc_fold}
    arrays = {
        "p": np.full(len(y), 0.5),
        "p_doc": np.full(len(doc_fold), 0.5),
        "doc_order": sorted(doc_fold),
        "y_doc": np.ones(len(doc_fold), dtype=bool),
    }
    res = {"_arrays": arrays, "_reliability": [], "_curves": [], "auroc": [], "tau": [],
           "doc_level": [], "calibration": [], "note": "only aggregates"}  # fmt: skip
    # a predicted value that must never reach any file: it lives in no structure passed here
    _ = {"header": {"currency": secret}}
    cal2.write_outputs(res, tv, y, ~y, doc_fold, meta, tmp_path)
    for f in tmp_path.iterdir():
        assert secret not in f.read_text(encoding="utf-8")
    head = (tmp_path / "oof_fields_v2.csv").read_text(encoding="utf-8").splitlines()[0]
    assert head == ("doc_id,scope,field,row_idx,emitted,p_correct_v2,y_correct,y_null,rule_touched")


def test_fold_shape_fn_is_supplier_disjoint() -> None:
    def g(did: str, cpn: str | None, po: str | None) -> dict[str, Any]:
        row = _row("S1", cpn, po, "1")
        return {"doc_id": did, "doc_type": "invoice", "header": {}, "line_items": [row]}

    labels = [
        g("train_0", "AB-12", "12345"),
        g("train_1", "CD-34", "67890"),
        g("train_2", "EF-56", "11111"),
    ]
    doc_fold = {"train_0": 0, "train_1": 1, "train_2": 2}
    groups = {"train_0": "a", "train_1": "b", "train_2": "c"}
    fn, equal = cal2.fold_shape_fn(labels, doc_fold, groups)
    assert fn("train_0").cpn_only == frozenset({"AA-99"})
    assert equal == {0: True, 1: True, 2: True}
    with pytest.raises(SystemExit):
        cal2.fold_shape_fn(labels, doc_fold, {"train_0": "a", "train_1": "a", "train_2": "c"})
