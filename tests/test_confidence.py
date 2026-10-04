"""Tests for shipdoc.confidence (synthetic data only, no I/O besides the optional scorer)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from shipdoc import confidence as cf
from shipdoc.ocr import OcrItem, PageOcr

ROOT = Path(__file__).resolve().parents[1]
needs_scorer = pytest.mark.skipif(
    not (ROOT / "assignment" / "score.py").is_file(), reason="assignment/score.py absent"
)
B = cf.BASE_FEATURES


def _feat(rows: list[cf.FieldRow], scope: str, name: str, ridx: int = -1) -> dict[str, float]:
    return next(r.feats for r in rows if r.key[1:] == (scope, name, ridx))


def _item(text: str, x0: float, y0: float, x1: float, y1: float) -> OcrItem:
    return OcrItem(text, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], 0.9)


def _ocr_page(lines: list[str]) -> PageOcr:
    items = [
        _item(t, 50, 100 + 50 * r, 50 + 25 * len(t), 120 + 50 * r) for r, t in enumerate(lines)
    ]
    return PageOcr("d_0000_p1.png", 1240, 1754, "paddleocr", "line", 0, items=items)


def _lp(scope: str, row_idx: int | None, field: str, mn: float, me: float) -> dict[str, Any]:
    return {
        "scope": scope,
        "row_idx": row_idx,
        "field": field,
        "min": mn,
        "mean": me,
        "n_tokens": 3,
    }


HEADER_NULLS = {k: None for k in cf.HEADER_FIELDS["invoice"] + cf.HEADER_FIELDS["waybill"]}
ROW = {
    "supplier_part_number": "AB-100",
    "customer_part_number": None,
    "purchase_order": None,
    "quantity": "12",
}


def _pred(**header: Any) -> dict[str, Any]:
    base = {k: None for k in cf.HEADER_FIELDS["invoice"]}
    return {"doc_type": "invoice", "header": {**base, **header}, "line_items": [dict(ROW)]}


def _trace(pages: list[dict[str, Any]], field_pages: dict[str, int | None]) -> dict[str, Any]:
    return {"pages": pages, "merge": {"field_pages": field_pages}}


def _page(
    header: dict[str, Any],
    rows: list[dict[str, Any]] | None = None,
    lps: Any = None,
    img: str = "x_p1.png",
) -> dict[str, Any]:
    p: dict[str, Any] = {
        "image": img,
        "parsed": {
            "doc_type": "invoice",
            "header": {**HEADER_NULLS, **header},
            "line_items": rows or [],
        },
    }
    if lps is not None:
        p["field_logprobs"] = lps
    return p


# --------------------------------------------------------------------------------------------
# Feature builder
# --------------------------------------------------------------------------------------------


@needs_scorer
def test_feature_builder_on_a_hand_made_page() -> None:
    ocr = [_ocr_page(["Invoice No.: INV-2026-0042", "Date: 17/02/2026", "AB-100 12 widget"])]
    pred = _pred(
        invoice_number="INV-2026-0042",
        invoice_date="2026-02-17",
        ship_to_name="Hallucinated Corp",
        awb_number="999 1234",
    )
    lps = [
        _lp("header", None, "invoice_number", -0.1, -0.05),
        _lp("header", None, "buyer_name", -0.3, -0.2),
        _lp("row", 0, "quantity", -0.7, -0.4),
    ]
    page = _page(pred["header"], [dict(ROW)], lps, img="x_p1.jpg")
    fp = {"invoice_number": 1, "invoice_date": 1, "ship_to_name": 1, "awb_number": 1}
    rows = cf.build_doc_features("d_0000", pred, _trace([page], fp), ocr)
    inv = _feat(rows, "header", "invoice_number")
    assert inv["ocr_exact_norm"] == 1.0 and inv["ocr_fuzzy"] == 1.0 and inv["has_ocr"] == 1.0
    assert (inv["lp_min"], inv["lp_mean"], inv["has_logprobs"]) == (-0.1, -0.05, 1.0)
    assert inv["scanned"] == 1.0 and inv["xp_single"] == 1.0 and inv["was_null"] == 0.0
    assert _feat(rows, "header", "invoice_date")["ocr_exact_norm"] == 1.0  # 17/02/2026 -> ISO
    assert _feat(rows, "header", "ship_to_name")["ocr_exact_norm"] == 0.0  # not on the page
    assert _feat(rows, "header", "awb_number")["v_awb_bad"] == 1.0
    buyer = _feat(rows, "header", "buyer_name")  # null value: its null-token logprob is used
    assert buyer["was_null"] == 1.0 and buyer["lp_min"] == -0.3
    assert next(r for r in rows if r.key.field == "buyer_name").emitted is False
    qty = _feat(rows, "row", "quantity", 0)
    assert qty["lp_min"] == -0.7 and qty["ocr_exact_norm"] == 1.0
    # one row per header field of the invoice type and per field of the one row
    assert len(rows) == len(cf.HEADER_FIELDS["invoice"]) + 4
    assert not any(r.key.field == "carrier" for r in rows)  # waybill field of an invoice


def test_missing_logprobs_are_nan_with_indicator_zero() -> None:
    page = _page(_pred(invoice_number="A1")["header"], [dict(ROW)])  # no field_logprobs key
    rows = cf.build_doc_features("d_0000", _pred(invoice_number="A1"), _trace([page], {}), None)
    for r in rows:
        assert np.isnan(r.feats["lp_min"]) and np.isnan(r.feats["lp_mean"])
        assert r.feats["has_logprobs"] == 0.0 and r.feats["has_ocr"] == 0.0
    table = cf.assemble(rows)
    assert np.isnan(table.X[:, B.index("lp_min")]).all()


def test_lr_ignores_a_missing_feature_without_leaking() -> None:
    syn = cf.synthetic_data(n_docs=60, seed=3)
    X = cf.design_matrix(syn.table)
    X_nan, X_zero = X.copy(), X.copy()
    cols = [B.index("lp_min"), B.index("lp_mean")]
    X_nan[:, cols] = np.nan
    X_zero[:, cols] = 0.0
    y = syn.y_null.astype(int)
    p_nan = cf.fit_predict("lr", X_nan, y, X_nan)
    p_zero = cf.fit_predict("lr", X_zero, y, X_zero)
    assert np.isfinite(p_nan).all() and ((p_nan > 0) & (p_nan < 1)).all()
    assert np.allclose(p_nan, p_zero)  # an all-missing column is a constant: carries no label info
    assert np.isfinite(cf.fit_predict("gbm", X_nan, y, X_nan)).all()


def test_cross_page_agreement_and_disagreement() -> None:
    p1 = _page({"invoice_number": "A-1", "currency": "USD"})
    p2 = _page({"invoice_number": "B-2", "currency": "usd"}, img="x_p2.png")
    pred = _pred(invoice_number="A-1", currency="USD")
    rows = cf.build_doc_features("d_0000", pred, _trace([p1, p2], {"invoice_number": 1}), None)
    inv, cur = _feat(rows, "header", "invoice_number"), _feat(rows, "header", "currency")
    assert (inv["xp_agree"], inv["xp_disagree"], inv["xp_single"], inv["xp_n_pages"]) == (
        0,
        1,
        0,
        2,
    )
    assert (cur["xp_agree"], cur["xp_disagree"], cur["xp_agree_frac"]) == (1, 0, 1.0)
    # a value only on page 2 while the emitted field is null (the page-2 trap)
    p2b = _page({"buyer_name": "Late Buyer"}, img="x_p2.png")
    rows = cf.build_doc_features("d_0000", _pred(), _trace([p1, p2b], {}), None)
    assert _feat(rows, "header", "buyer_name")["xp_disagree"] == 1


def test_row_sources_mirror_merge_filters() -> None:
    hdr_row = {"supplier_part_number": "Part No", "quantity": "Qty"}
    null_row = {"supplier_part_number": None}
    pages = [
        _page({}, [dict(ROW), hdr_row, null_row]),
        _page({}, ["junk", dict(ROW)]),  # type: ignore[list-item]
    ]
    assert cf.row_sources(pages) == [(0, 0), (1, 1)]


def test_date_not_iso() -> None:
    assert not cf.date_not_iso("2026-02-17") and not cf.date_not_iso(None)
    assert cf.date_not_iso("17/02/2026") and cf.date_not_iso("2026-02-30")


# --------------------------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------------------------


@needs_scorer
def test_label_doc_follows_scorer_same_and_false_fill() -> None:
    gold_header = {k: None for k in cf.HEADER_FIELDS["invoice"]}
    gold_header.update(invoice_number="INV-1", currency="USD")
    gold = {"doc_type": "invoice", "header": gold_header, "line_items": [dict(ROW)]}
    pred = _pred(invoice_number="inv-1", currency="EUR", buyer_name="Filled")
    pred["line_items"] = [dict(ROW), {**ROW, "supplier_part_number": "ZZ-9"}]
    lab = cf.label_doc(pred, gold)
    assert lab[("header", "invoice_number", -1)] == (True, False)  # case-insensitive id
    assert lab[("header", "currency", -1)] == (False, False)
    assert lab[("header", "buyer_name", -1)] == (False, True)  # false fill on a gold null
    assert lab[("header", "awb_number", -1)] == (True, True)  # null on a gold null
    assert lab[("row", "quantity", 0)] == (True, False)
    assert lab[("row", "quantity", 1)] == (False, False)  # extra row: no gold to match
    wrong_type = dict(gold, doc_type="waybill")
    assert set(cf.label_doc(pred, wrong_type).values()) == {(False, False)}


# --------------------------------------------------------------------------------------------
# Cross-fit, novelty
# --------------------------------------------------------------------------------------------


def test_cross_fit_has_no_doc_or_group_in_both_sides() -> None:
    syn = cf.synthetic_data(n_docs=90, seed=1)
    ids = [k.doc_id for k in syn.table.keys]
    splits = cf.fold_splits(ids, syn.doc_fold, syn.groups)
    assert len(splits) == 3
    for fit, pred in splits:
        assert not {ids[i] for i in fit} & {ids[i] for i in pred}
        assert not {syn.groups[ids[i]] for i in fit} & {syn.groups[ids[i]] for i in pred}
    assert sorted(np.concatenate([p for _, p in splits]).tolist()) == list(range(len(ids)))
    bad_groups = {d: "same" for d in syn.groups}  # one supplier in every fold: must be rejected
    with pytest.raises(AssertionError, match="supplier group"):
        cf.fold_splits(ids, syn.doc_fold, bad_groups)


def test_oof_probabilities_are_out_of_fold() -> None:
    syn = cf.synthetic_data(n_docs=90, seed=2)
    oof = cf.cross_fit(syn.table, syn.y_null, np.ones(len(syn.y_null), bool), syn.doc_fold, "lr")
    assert np.isfinite(oof).all()
    # flipping the labels of one fold must not change that fold's own predictions
    fold0 = np.array([syn.doc_fold[k.doc_id] == 0 for k in syn.table.keys])
    y2 = syn.y_null.copy()
    y2[fold0] = ~y2[fold0]
    oof2 = cf.cross_fit(syn.table, y2, np.ones(len(y2), bool), syn.doc_fold, "lr")
    assert np.allclose(oof[fold0], oof2[fold0])
    assert not np.allclose(oof[~fold0], oof2[~fold0])


def test_novelty_never_uses_the_held_out_fold_as_reference() -> None:
    # signature = 100 * fold id: a doc can only be near its own fold, so a reference that leaked
    # the doc's own fold would give novelty 0
    doc_fold = {f"d{i}": i % 3 for i in range(30)}
    sigs = {d: np.array([100.0 * f]) for d, f in doc_fold.items()}
    table = cf.compute_novelty_table(sigs, doc_fold, reference_fn=lambda s: np.vstack(s))
    assert set(table) == {0, 1, 2}
    for held, per_doc in table.items():
        assert len(per_doc) == 30
        assert min(per_doc.values()) >= 100.0, f"fold {held}: reference contained the doc's fold"


def test_layout_novelty_is_distance_to_nearest_row() -> None:
    ref = np.array([[0.0, 0.0], [10.0, 0.0]])
    assert cf.layout_novelty(np.array([9.0, 0.0]), ref) == pytest.approx(1.0)
    assert cf.layout_novelty(np.array([0.0, 0.0]), ref) == 0.0


# --------------------------------------------------------------------------------------------
# Calibrator selection
# --------------------------------------------------------------------------------------------


def test_lr_gbm_selection_is_deterministic_and_margin_driven() -> None:
    syn = cf.synthetic_data(n_docs=90, seed=4)
    all_rows = np.ones(len(syn.y_null), bool)
    a = cf.select_model(syn.table, syn.y_null, all_rows, syn.doc_fold)
    b = cf.select_model(syn.table, syn.y_null, all_rows, syn.doc_fold)
    assert a.kind == b.kind and a.logloss == b.logloss and np.array_equal(a.oof, b.oof)
    # a huge margin can never pick GBM; a hugely negative one always does
    assert cf.select_model(syn.table, syn.y_null, all_rows, syn.doc_fold, margin=10.0).kind == "lr"
    assert (
        cf.select_model(syn.table, syn.y_null, all_rows, syn.doc_fold, margin=-10.0).kind == "gbm"
    )


def test_gbm_wins_on_an_interaction_lr_cannot_fit() -> None:
    rng = np.random.default_rng(0)
    n = 1500
    syn = cf.synthetic_data(n_docs=n // 12, seed=5)
    n = len(syn.y_null)
    X = syn.table.X.copy()
    a, b = rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)
    X[:, B.index("lp_min")], X[:, B.index("ocr_fuzzy")] = a, b
    table = cf.FieldTable(syn.table.keys, X, syn.table.emitted)
    y = (a * b > 0).astype(bool)  # XOR-like: linearly inseparable
    sel = cf.select_model(table, y, np.ones(n, bool), syn.doc_fold)
    assert sel.kind == "gbm" and sel.logloss["gbm"] < sel.logloss["lr"] - cf.GBM_MARGIN


def test_synthetic_calibrators_recover_the_known_truth() -> None:
    syn = cf.synthetic_data(n_docs=300, seed=42)
    oof = cf.calibrate_oof(syn.table, syn.y_correct, syn.y_null, syn.doc_fold, None, syn.groups)
    em = syn.table.emitted
    assert np.corrcoef(oof.p_correct[em], syn.true_p_correct[em])[0, 1] > 0.95
    assert np.isnan(oof.p_correct[~em]).all()
    assert cf.ece(oof.p_correct[em], syn.y_correct[em]) < 0.05


def test_constant_model_when_a_class_is_missing() -> None:
    X = np.zeros((10, 3))
    p = cf.fit_predict("lr", X, np.zeros(10, dtype=int), X)
    assert np.allclose(p, 0.0)


# --------------------------------------------------------------------------------------------
# Null policy
# --------------------------------------------------------------------------------------------


def test_null_policy_rule_and_never_fills_property() -> None:
    rng = np.random.default_rng(7)
    for trial in range(40):
        n_docs = int(rng.integers(1, 6))
        docs: dict[str, dict[str, Any]] = {}
        keys: list[cf.FieldKey] = []
        emitted: list[bool] = []
        for d in range(n_docs):
            did = f"d{d}"
            header = {
                f: (None if rng.random() < 0.4 else f"v{trial}{d}{f}")
                for f in cf.HEADER_FIELDS["invoice"]
            }
            rows = [
                {f: (None if rng.random() < 0.4 else "x") for f in cf.ROW_KEYS}
                for _ in range(int(rng.integers(0, 4)))
            ]
            docs[did] = {"doc_type": "invoice", "header": header, "line_items": rows}
            for f, v in header.items():
                keys.append(cf.FieldKey(did, "header", f, -1))
                emitted.append(v is not None)
            for ri, r in enumerate(rows):
                for f, v in r.items():
                    keys.append(cf.FieldKey(did, "row", f, ri))
                    emitted.append(v is not None)
        em = np.array(emitted)
        pc, pn = rng.random(len(em)), rng.random(len(em))
        nulled = cf.null_decisions(pc, pn, em)
        assert not (nulled & ~em).any()  # never touches a non-emitted field
        assert np.array_equal(nulled, em & (pn > pc))
        out = cf.apply_null_policy(docs, keys, nulled)
        cf.assert_never_fills(docs, out)  # also asserted inside apply_null_policy
        for k, hit in zip(keys, nulled, strict=True):
            got = (
                out[k.doc_id]["header"][k.field]
                if k.scope == "header"
                else out[k.doc_id]["line_items"][k.row_idx][k.field]
            )
            assert (got is None) if hit else True
        assert docs["d0"] is not out["d0"]  # input untouched (deep copy)


def test_assert_never_fills_rejects_a_fill_and_a_change() -> None:
    before = {"d": {"header": {"a": None, "b": "x"}, "line_items": [{"q": None}]}}
    filled = {"d": {"header": {"a": "NEW", "b": "x"}, "line_items": [{"q": None}]}}
    changed = {"d": {"header": {"a": None, "b": "y"}, "line_items": [{"q": None}]}}
    row_filled = {"d": {"header": {"a": None, "b": "x"}, "line_items": [{"q": "1"}]}}
    for bad in (filled, changed, row_filled):
        with pytest.raises(AssertionError):
            cf.assert_never_fills(before, bad)
    cf.assert_never_fills(
        before, {"d": {"header": {"a": None, "b": None}, "line_items": [{"q": None}]}}
    )


def test_false_fill_split_redaction_vs_absent_line() -> None:
    hdr = {k: "v" for k in cf.HEADER_FIELDS["invoice"]}
    gold = {
        "i1": {"doc_type": "invoice", "header": {**hdr, "awb_number": None, "currency": None}},
        "i2": {"doc_type": "invoice", "header": {**hdr, "awb_number": None}},
    }
    meta = {"i1": {"awb_absent": True}, "i2": {"awb_absent": True}}
    pred = {
        "i1": {"header": {**hdr, "awb_number": "999-12345678", "currency": "USD"}},  # 2 false fills
        "i2": {"header": {**hdr, "awb_number": None}},
    }
    out = cf.false_fill_split(pred, gold, meta)
    assert out == {
        "redaction": {"null_fields": 1, "filled": 1},
        "absent_line": {"null_fields": 2, "filled": 1},
    }


# --------------------------------------------------------------------------------------------
# Metrics: ECE, tau, review CIs
# --------------------------------------------------------------------------------------------


def test_ece_is_near_zero_when_calibrated_and_large_when_overconfident() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 200_000)
    y = (rng.uniform(0, 1, p.size) < p).astype(float)
    assert cf.ece(p, y) < 0.01 and cf.ece(p, y, scheme="mass") < 0.01
    over_p = np.full(10_000, 0.95)
    over_y = (rng.uniform(0, 1, over_p.size) < 0.6).astype(float)
    assert cf.ece(over_p, over_y) > 0.3 and cf.ece(over_p, over_y, scheme="mass") > 0.3
    assert np.isnan(cf.ece(np.array([]), np.array([])))
    with pytest.raises(ValueError):
        cf.ece(p, y, scheme="nope")


def test_reliability_and_coverage_shapes() -> None:
    p = np.array([0.05, 0.5, 0.95, 1.0])
    y = np.array([0, 1, 1, 1])
    rel = cf.reliability(p, y)
    assert len(rel) == cf.ECE_BINS and sum(r["n"] for r in rel) == 4
    assert rel[-1]["n"] == 2  # p = 1.0 falls in the last bin
    cov = cf.coverage_curve(p, y, points=4)
    assert [round(c["coverage"], 2) for c in cov] == [0.25, 0.5, 0.75, 1.0]
    assert cov[0]["accuracy"] == 1.0 and cov[-1]["accuracy"] == 0.75


def test_select_tau_hits_the_precision_target() -> None:
    rng = np.random.default_rng(1)
    conf = rng.uniform(0, 1, 20_000)
    correct = rng.uniform(0, 1, conf.size) < conf  # calibrated: precision of {conf >= t} ~ (1+t)/2
    tau = cf.select_tau(conf, correct, 0.98)
    assert tau is not None
    acc = conf >= tau
    assert correct[acc].mean() >= 0.98
    # lowest tau: no smaller candidate threshold also reaches the target (precision is not
    # monotone in tau, so this is a full scan, not a bisection)
    cands = np.unique(conf)
    qual = [t for t in cands if correct[conf >= t].mean() >= 0.98]
    assert tau == min(qual)


def test_select_tau_flags_everything_when_unattainable() -> None:
    conf = np.linspace(0.1, 0.9, 50)
    correct = np.arange(50) % 2 == 0  # 50% precision everywhere
    assert cf.select_tau(conf, correct, 0.98) is None
    assert cf.select_tau(np.array([]), np.array([], bool)) is None
    stats = cf.review_with_ci(conf, correct, np.array(["d"] * 50, dtype=object), None, n_boot=50)
    assert stats["review_rate"]["point"] == 1.0 and stats["accepted_share"]["point"] == 0.0
    assert stats["error_recall"]["point"] == 1.0


def test_select_tau_accepts_whole_ties() -> None:
    conf = np.array([0.9, 0.9, 0.9, 0.5])
    correct = np.array([True, True, False, True])
    assert cf.select_tau(conf, correct, 0.6) == 0.5  # 3/4 >= 0.6 only when the 0.5 row is in


def test_review_bootstrap_is_deterministic_and_resamples_documents() -> None:
    rng = np.random.default_rng(3)
    n = 600
    docs = np.array([f"d{i // 6}" for i in range(n)], dtype=object)
    conf = rng.uniform(0, 1, n)
    correct = rng.uniform(0, 1, n) < conf
    a = cf.review_with_ci(conf, correct, docs, 0.7, n_boot=300, seed=42)
    b = cf.review_with_ci(conf, correct, docs, 0.7, n_boot=300, seed=42)
    c = cf.review_with_ci(conf, correct, docs, 0.7, n_boot=300, seed=43)
    assert a == b and a != c
    for m, v in a.items():
        assert v["lo"] <= v["point"] <= v["hi"], m
    # point values against a direct computation
    acc = conf >= 0.7
    assert a["precision_accepted"]["point"] == pytest.approx(correct[acc].mean())
    assert a["review_rate"]["point"] == pytest.approx((~acc).mean())
    assert a["error_recall"]["point"] == pytest.approx((~acc & ~correct).sum() / (~correct).sum())
    flagged_docs = {d for d, f in zip(docs, ~acc, strict=True) if f}
    assert a["doc_flag_rate"]["point"] == pytest.approx(len(flagged_docs) / len(set(docs)))


def test_synthetic_data_is_seeded() -> None:
    a, b = cf.synthetic_data(30, seed=9), cf.synthetic_data(30, seed=9)
    assert np.array_equal(a.table.X, b.table.X) and np.array_equal(a.y_correct, b.y_correct)
    assert len(cf.DESIGN_FEATURES) == cf.design_matrix(a.table).shape[1]
