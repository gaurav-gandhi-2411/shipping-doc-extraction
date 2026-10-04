"""Tests for shipdoc.confidence_v3 and scripts/calibrate_v3.py (synthetic data, no real I/O)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from shipdoc import confidence as cf
from shipdoc import confidence_v2 as c2
from shipdoc import confidence_v3 as c3

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("calibrate_v3", ROOT / "scripts" / "calibrate_v3.py")
cal3 = importlib.util.module_from_spec(_spec)
sys.modules["calibrate_v3"] = cal3
_spec.loader.exec_module(cal3)
F = {n: i for i, n in enumerate(c3.AGREE_FEATURES)}
SPN, CPN, PO, QTY = c3.ROW_KEYS


def _row(spn: str | None, cpn: str | None = None, po: str | None = None, qty: str | None = "1"):
    return {SPN: spn, CPN: cpn, PO: po, QTY: qty}


# ---------------------------------------------------------------------------------------------
# value comparison
# ---------------------------------------------------------------------------------------------


def test_norm_equal_and_similarity() -> None:
    assert c3.norm_equal(SPN, "AB-12/3", "ab 123")
    assert not c3.norm_equal(SPN, "AB-12/3", "AB-124")
    assert c3.norm_equal(QTY, "1,000", "1000.00")  # numbers via plain-number parse
    assert not c3.norm_equal(QTY, "1,000", "1001")
    assert not c3.norm_equal(SPN, None, None)  # null/null is the both_null indicator, not norm
    assert c3.norm_equal(SPN, "--", "--") and not c3.norm_equal(SPN, "--", "//")
    assert c3.similarity(SPN, "AB-123", "ab123") == 1.0
    assert c3.similarity(SPN, "AB123", None) == 0.0
    s = c3.similarity(SPN, "ABCDEFGHIJ", "ABCDEFGHIK")
    assert 0.0 < s < 1.0 and s == c3.similarity(SPN, "ABCDEFGHIK", "ABCDEFGHIJ")
    assert c3.plain_number("nan") is None and c3.plain_number("1 234,5") == 12345.0


def test_pair_features_values() -> None:
    assert c3.pair_features(SPN, "A-1", "A-1") == (1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
    exact, norm, sim, bn, so, sv = c3.pair_features(SPN, "A-1", "a1")
    assert (exact, norm, sim, bn, so, sv) == (0.0, 1.0, 1.0, 0.0, 0.0, 0.0)
    assert c3.pair_features(SPN, None, None) == (0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    assert c3.pair_features(SPN, "", "x") == (0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    assert c3.pair_features(SPN, "x", "  ") == (0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    e, n, s, *_ = c3.pair_features(SPN, "ABCD", "WXYZ")
    assert (e, n) == (0.0, 0.0) and 0.0 <= s < 0.5


def test_num_fields_match_scorer() -> None:
    p = ROOT / "assignment" / "score.py"
    if not p.is_file():
        pytest.skip("assignment/score.py absent")
    from shipdoc import eval as ev

    kinds = ev.load_scorer().KIND
    assert {k for k, v in kinds.items() if v == "num"} == c3.NUM_FIELDS


# ---------------------------------------------------------------------------------------------
# row alignment
# ---------------------------------------------------------------------------------------------


def _pairs(ft: list[dict], zs: list[dict]) -> list[tuple[int, int, int]]:
    return [(p.ft, p.zs, p.stage) for p in c3.align_rows(ft, zs)]


def test_align_exact_and_reordered() -> None:
    ft = [_row("A1", "C1", "P1", "5"), _row("B2", None, None, "7"), _row("C3")]
    assert _pairs(ft, ft) == [(0, 0, 1), (1, 1, 1), (2, 2, 1)]
    zs = [ft[2], ft[0], ft[1]]  # reordered
    assert _pairs(ft, zs) == [(0, 1, 1), (1, 2, 1), (2, 0, 1)]


def test_align_spn_equal_then_fuzzy_then_position() -> None:
    ft = [_row("AB-100", qty="5"), _row("XYZ-9999-A", qty="2"), _row("QQ", qty="9")]
    zs = [_row("ab100", qty="6"), _row("XYZ-9999-B", qty="2"), _row("ZZ", qty="1")]
    got = _pairs(ft, zs)
    # row 0: same part number, other quantity -> pass 2; row 1: fuzzy (one char off) -> pass 3;
    # row 2: unrelated, same position -> pass 4
    assert got == [(0, 0, 2), (1, 1, 3), (2, 2, 4)]


def test_align_missing_extra_unaligned() -> None:
    ft = [_row("A1"), _row("B2"), _row("C3")]
    zs = [_row("C3"), _row("A1")]  # B2 missing in zs
    got = _pairs(ft, zs)
    assert (0, 1, 1) in got and (2, 0, 1) in got
    assert all(p[0] != 1 for p in got)  # the extra ft row has no partner
    # position never pairs a row that already has a partner, and pairs are one-to-one
    zs2 = [_row("A1")]
    assert _pairs(ft, zs2) == [(0, 0, 1)]
    assert c3.align_rows([], zs) == [] and c3.align_rows(ft, []) == []


def test_align_deterministic_and_one_to_one() -> None:
    rng = np.random.default_rng(0)
    pool = [f"PN{int(x):03d}" for x in rng.integers(0, 40, 30)]
    ft = [_row(p) for p in pool[:15]]
    zs = [_row(p) for p in pool[10:30]]
    a, b = c3.align_rows(ft, zs), c3.align_rows(ft, zs)
    assert a == b
    assert len({p.ft for p in a}) == len(a) == len({p.zs for p in a})


# ---------------------------------------------------------------------------------------------
# agreement features
# ---------------------------------------------------------------------------------------------


def _doc(header: dict[str, Any], rows: list[dict], dt: str = "invoice") -> dict[str, Any]:
    return {"doc_type": dt, "header": header, "line_items": rows}


def _keys(doc: dict[str, Any], d: str = "d1") -> list[cf.FieldKey]:
    ks = [cf.FieldKey(d, "header", n, -1) for n in cf.HEADER_FIELDS[doc["doc_type"]]]
    for ri in range(len(doc["line_items"])):
        ks += [cf.FieldKey(d, "row", n, ri) for n in c3.ROW_KEYS]
    return ks


def test_agreement_values_header_rows_and_doc_level() -> None:
    hf = cf.HEADER_FIELDS["invoice"]
    ft_h = dict.fromkeys(hf)
    zs_h = dict.fromkeys(hf)
    ft_h["invoice_number"], zs_h["invoice_number"] = "INV-1", "inv1"  # norm-equal, not exact
    ft_h["currency"], zs_h["currency"] = "USD", None  # self value, other null
    ft_h["buyer_name"], zs_h["buyer_name"] = None, "ACME"  # self null, other value
    ft = _doc(ft_h, [_row("A1", qty="5"), _row("ZZ9", qty="3")])
    zs = _doc(zs_h, [_row("a1", qty="5"), _row("Q", qty="3")])
    pairs = {"d1": c3.align_rows(c3.dict_rows(ft), c3.dict_rows(zs))}
    keys = _keys(ft)
    m = c3.agreement_matrix(keys, {"d1": ft}, {"d1": zs}, pairs, True)
    ix = {k: i for i, k in enumerate(keys)}
    inv = m[ix[cf.FieldKey("d1", "header", "invoice_number", -1)]]
    assert inv[F["ag_exact"]] == 0 and inv[F["ag_norm"]] == 1 and inv[F["ag_sim"]] == 1
    cur = m[ix[cf.FieldKey("d1", "header", "currency", -1)]]
    assert cur[F["ag_self_val_other_null"]] == 1 and cur[F["ag_norm"]] == 0
    buy = m[ix[cf.FieldKey("d1", "header", "buyer_name", -1)]]
    assert buy[F["ag_self_null_other_val"]] == 1
    both = m[ix[cf.FieldKey("d1", "header", "invoice_date", -1)]]
    assert both[F["ag_both_null"]] == 1 and both[F["ag_norm"]] == 0
    r0 = m[ix[cf.FieldKey("d1", "row", SPN, 0)]]
    assert r0[F["ag_row_unaligned"]] == 0 and r0[F["ag_norm"]] == 1 and r0[F["ag_exact"]] == 0
    r1 = m[ix[cf.FieldKey("d1", "row", SPN, 1)]]  # aligned by position only
    assert r1[F["ag_pos_aligned"]] == 1 and r1[F["ag_norm"]] == 0
    # doc level: 8 header fields + 8 row fields compared; agree = norm-equal or both-null
    rate = m[:, F["ag_doc_rate"]]
    assert np.all(rate == rate[0]) and 0 < rate[0] < 1
    n_agree = sum(
        float(m[i, F["ag_norm"]] == 1 or m[i, F["ag_both_null"]] == 1) for i in range(len(keys))
    )
    assert rate[0] == pytest.approx(n_agree / len(keys))  # every field is compared (all aligned)
    assert np.all(m[:, F["ag_type_differs"]] == 0)


def test_agreement_unaligned_row_and_symmetric_orientation() -> None:
    ft = _doc({}, [_row("A1"), _row("B2"), _row("C3")])
    zs = _doc({}, [_row("C3"), _row("A1")])
    pairs = {"d1": c3.align_rows(c3.dict_rows(ft), c3.dict_rows(zs))}
    k_ft, k_zs = _keys(ft), _keys(zs)
    m_ft = c3.agreement_matrix(k_ft, {"d1": ft}, {"d1": zs}, pairs, True)
    m_zs = c3.agreement_matrix(k_zs, {"d1": zs}, {"d1": ft}, pairs, False)
    un = [
        (k.row_idx, m_ft[i, F["ag_row_unaligned"]]) for i, k in enumerate(k_ft) if k.scope == "row"
    ]
    assert {r for r, u in un if u == 1} == {1}  # only B2 of ft is unaligned
    assert not any(m_zs[i, F["ag_row_unaligned"]] for i, k in enumerate(k_zs) if k.scope == "row")
    assert m_ft[0, F["ag_doc_unaligned_frac"]] == pytest.approx(1 / 3)
    assert m_zs[0, F["ag_doc_unaligned_frac"]] == 0
    # ft row 0 (A1) pairs zs row 1 and the same pair seen from zs is row 1 -> ft row 0
    i_ft = k_ft.index(cf.FieldKey("d1", "row", SPN, 0))
    i_zs = k_zs.index(cf.FieldKey("d1", "row", SPN, 1))
    assert m_ft[i_ft, F["ag_exact"]] == m_zs[i_zs, F["ag_exact"]] == 1


def test_agreement_doc_type_differs_and_identical_arms() -> None:
    ft = _doc({"carrier": "X"}, [], "waybill")
    zs = _doc({"invoice_number": "1"}, [], "invoice")
    m = c3.agreement_matrix(_keys(ft), {"d1": ft}, {"d1": zs}, {"d1": []}, True)
    assert np.all(m[:, F["ag_type_differs"]] == 1)
    same = c3.agreement_matrix(_keys(ft), {"d1": ft}, {"d1": ft}, {"d1": []}, True)
    assert np.all(same[:, F["ag_type_differs"]] == 0)
    assert np.all(same[:, F["ag_row_unaligned"]] == 0)


# ---------------------------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------------------------


def _scores(seed: int = 1, n: int = 400, n_docs: int = 40) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    y = (rng.uniform(size=n) < 0.7).astype(int)
    weak = y * 0.3 + rng.normal(size=n)
    strong = y * 1.5 + rng.normal(size=n)
    docs = np.array([f"d{i % n_docs}" for i in range(n)], dtype=object)
    return weak, strong, y, docs


def test_paired_boot_deterministic_and_signed() -> None:
    weak, strong, y, docs = _scores()
    a = c3.paired_auroc_boot(weak, strong, y, docs, n_boot=200, seed=42)
    b = c3.paired_auroc_boot(weak, strong, y, docs, n_boot=200, seed=42)
    assert a == b
    assert a["point"] == pytest.approx(c2.auroc(strong, y) - c2.auroc(weak, y))
    assert a["point"] > 0 and a["lo"] > 0 and a["lo"] <= a["point"] <= a["hi"]
    z = c3.paired_auroc_boot(weak, weak, y, docs, n_boot=200, seed=42)
    assert z["point"] == 0 and z["lo"] == 0 == z["hi"]
    c = c3.paired_auroc_boot(weak, strong, y, docs, n_boot=200, seed=7)
    assert c["lo"] != a["lo"]  # a different seed changes the resamples


def test_paired_boot_single_class_is_nan() -> None:
    weak, strong, y, docs = _scores()
    out = c3.paired_auroc_boot(weak, strong, np.ones_like(y), docs, n_boot=20)
    assert np.isnan(out["point"]) and np.isnan(out["lo"])


# ---------------------------------------------------------------------------------------------
# folds and cross-fit
# ---------------------------------------------------------------------------------------------


def _docs_groups(n_docs: int = 40, n_groups: int = 8) -> tuple[list[str], dict[str, str]]:
    ids = [f"train_{i:04d}" for i in range(n_docs)]
    return ids, {d: f"s{i % n_groups}" for i, d in enumerate(ids)}


def test_inner_group_folds_disjoint_deterministic() -> None:
    ids, groups = _docs_groups()
    f1, k, ng = c3.inner_group_folds(ids, groups)
    f2, _, _ = c3.inner_group_folds(list(reversed(ids)), groups)
    assert f1 == f2 and k == 5 and ng == 8
    by_group: dict[str, set[int]] = {}
    for d, f in f1.items():
        by_group.setdefault(groups[d], set()).add(f)
    assert all(len(v) == 1 for v in by_group.values())  # a supplier group lives in ONE fold
    assert set(f1.values()) == set(range(k))
    _, k3, _ = c3.inner_group_folds(ids, {d: f"s{i % 3}" for i, d in enumerate(ids)})
    assert k3 == 3
    with pytest.raises(ValueError, match="at least 2"):
        c3.inner_group_folds(ids, dict.fromkeys(ids, "one"))


def test_cross_fit_never_shares_a_supplier_group() -> None:
    ids, groups = _docs_groups()
    folds, _, _ = c3.inner_group_folds(ids, groups)
    rows_doc = [d for d in ids for _ in range(3)]
    n = len(rows_doc)
    seen: list[tuple[set[str], set[str]]] = []

    def recorder(kind: str, Xf: np.ndarray, yf: np.ndarray, Xp: np.ndarray) -> np.ndarray:
        fit_g = {groups[rows_doc[int(i)]] for i in Xf[:, 0]}
        pred_g = {groups[rows_doc[int(i)]] for i in Xp[:, 0]}
        seen.append((fit_g, pred_g))
        return np.full(len(Xp), 0.5)

    X = np.arange(n, dtype=float).reshape(-1, 1)
    y = np.tile([0, 1, 1], len(ids))
    out = c3.cross_fit_design(
        lambda h: X, y, np.ones(n, dtype=bool), rows_doc, folds, groups, "lr", fit_fn=recorder
    )
    assert len(seen) == 5 and not np.isnan(out).any()
    assert all(not (f & p) for f, p in seen)


# ---------------------------------------------------------------------------------------------
# gates and fail-closed behaviour
# ---------------------------------------------------------------------------------------------

FOLDS = {
    "folds": [
        {"fold": 0, "val_doc_ids": ["train_0", "train_1"]},
        {"fold": 1, "val_doc_ids": ["train_2"]},
        {"fold": 2, "val_doc_ids": ["dev_0", "dev_1"]},
    ]
}
ALL = ["train_0", "train_1", "train_2", "dev_0", "dev_1"]


def _runs() -> list[tuple[int, list[str]]]:
    return [(0, ["train_0", "train_1"]), (1, ["train_2"]), (2, ["dev_0", "dev_1"])]


def test_coverage_accepts_valid_and_single_fold() -> None:
    c3.check_fold_coverage(_runs(), FOLDS, ALL, [0, 1, 2])
    c3.check_fold_coverage(_runs()[:1], FOLDS, ALL, [0])


def test_coverage_refuses_missing_duplicate_and_wrong_docs() -> None:
    with pytest.raises(c3.OofCoverageError, match="required folds"):
        c3.check_fold_coverage(_runs()[:2], FOLDS, ALL, [0, 1, 2])  # fold 2 missing
    dup = [_runs()[0], _runs()[0], _runs()[1]]
    with pytest.raises(c3.OofCoverageError, match="required folds"):
        c3.check_fold_coverage(dup, FOLDS, ALL, [0, 1, 2])  # fold 0 twice
    bad = [(0, ["train_0"]), *_runs()[1:]]
    with pytest.raises(c3.OofCoverageError, match="not exactly"):
        c3.check_fold_coverage(bad, FOLDS, ALL, [0, 1, 2])
    twice = [(0, ["train_0", "train_0"]), *_runs()[1:]]
    with pytest.raises(c3.OofCoverageError, match="duplicate"):
        c3.check_fold_coverage(twice, FOLDS, ALL, [0, 1, 2])
    with pytest.raises(c3.OofCoverageError, match="union"):
        c3.check_fold_coverage(_runs(), FOLDS, [*ALL, "dev_9"], [0, 1, 2])


def test_coverage_and_features_abort_on_test_ids() -> None:
    bad = [(0, ["test_0001"]), *_runs()[1:]]
    with pytest.raises(c2.NoTestDataError):
        c3.check_fold_coverage(bad, FOLDS, ALL, [0, 1, 2])
    with pytest.raises(c2.NoTestDataError):
        c3.align_all({"test_0001": {}}, {"test_0001": {}}, ["test_0001"])
    with pytest.raises(c2.NoTestDataError):
        c3.agreement_matrix([cf.FieldKey("test_9", "header", "x", -1)], {}, {}, {}, True)
    with pytest.raises(c2.NoTestDataError):
        c3.inner_group_folds(["test_1", "train_1"], {"test_1": "a", "train_1": "b"})


def test_gate_args() -> None:
    p = Path("x")
    assert cal3.gate_args([0], [p], False) is None
    assert cal3.gate_args([0, 1, 2], [p, p, p], False) is None
    assert cal3.gate_args([0, 1, 2], [p], False)  # needs all three runs
    assert cal3.gate_args([0, 1], [p, p], False)  # two folds is neither mode
    assert cal3.gate_args([0, 0, 1], [p, p, p], False)  # repeated fold
    assert cal3.gate_args([0], [], False)  # one fold without a run
    assert cal3.gate_args([0], [p], True)  # plumbing takes no run
    assert cal3.gate_args([0], [], True) is None


def test_main_refuses_before_reading_anything(capsys: pytest.CaptureFixture[str]) -> None:
    assert cal3.main(["--folds", "0", "1", "2", "--oof-run", "a"]) == 2
    assert cal3.main(["--folds", "0", "1", "--oof-run", "a", "--oof-run", "b"]) == 2
    assert cal3.main(["--folds", "0"]) == 2
    assert "REFUSED" in capsys.readouterr().err


def test_load_oof_run_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(cal3.ArmError, match="manifest"):
        cal3.load_oof_run(tmp_path)
    (tmp_path / "manifest.json").write_text('{"oof": {"fold": 0}}', encoding="utf-8")
    with pytest.raises(cal3.ArmError, match="not complete"):
        cal3.load_oof_run(tmp_path)
    (tmp_path / "progress.json").write_text('{"status": "complete"}', encoding="utf-8")
    (tmp_path / "trace.jsonl").write_text('{"doc_id": "test_0001"}\n', encoding="utf-8")
    (tmp_path / "predictions.json").write_text('{"test_0001": {}}', encoding="utf-8")
    with pytest.raises(c2.NoTestDataError):
        cal3.load_oof_run(tmp_path)
    (tmp_path / "trace.jsonl").write_text('{"doc_id": "train_0001"}\n', encoding="utf-8")
    (tmp_path / "predictions.json").write_text('{"train_0002": {}}', encoding="utf-8")
    with pytest.raises(cal3.ArmError, match="different documents"):
        cal3.load_oof_run(tmp_path)
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(cal3.ArmError, match="oof.fold"):
        cal3.load_oof_run(tmp_path)


# ---------------------------------------------------------------------------------------------
# view analysis on synthetic data
# ---------------------------------------------------------------------------------------------


def _synthetic_view(n_docs: int = 90) -> tuple[c2.TableV2, np.ndarray, np.ndarray, dict, dict]:
    syn = cf.synthetic_data(n_docs=n_docs, fields_per_doc=12)
    tv = c2.TableV2(syn.table, np.zeros((len(syn.table.keys), len(c2.EXTRA_FEATURES))))
    rng = np.random.default_rng(3)
    y = syn.y_correct
    flip = rng.uniform(size=len(y)) < 0.1
    agree = np.zeros((len(y), len(c3.AGREE_FEATURES)))
    agree[:, F["ag_norm"]] = np.where(flip, 1 - y, y)  # an informative, noisy agreement flag
    return tv, y, agree, syn.doc_fold, syn.groups


def test_fold_view_exploratory_machinery() -> None:
    tv, y, agree, _, groups = _synthetic_view()
    sigs = dict.fromkeys(tv.table.doc_ids.tolist())
    r = c3.analyze_fold_view("judge_ft", tv, y, agree, groups, sigs, n_boot=40)
    assert r["mode"] == "fold_exploratory" and r["k"] == 5 and r["n_groups"] == 15
    assert r["kind"] in ("lr", "gbm") and set(r["oof"]) == set(c3.VARIANTS)
    rec = {x["group"]: x for x in r["records"]}
    assert set(rec) == {"header_all"}  # the synthetic table has header fields only
    h = rec["header_all"]
    assert h["auroc_flag"]["point"] > 0.6  # the flag carries signal by construction
    assert h["auroc_agree"]["point"] > 0.6
    assert h["lift"]["point"] == pytest.approx(
        h["auroc_v2_agree"]["point"] - h["auroc_v2"]["point"]
    )
    again = c3.analyze_fold_view("judge_ft", tv, y, agree, groups, sigs, n_boot=40)
    assert again["records"] == r["records"]  # deterministic
    reuse = c3.analyze_fold_view("judge_zs", tv, y, agree, groups, sigs, 40, ("pooled", "lr"))
    assert reuse["kind"] == "lr" and reuse["emitted_oof_log_loss"] == {}


def test_full_view_cross_fitted_machinery() -> None:
    tv, y, agree, doc_fold, groups = _synthetic_view()
    r = c3.analyze_full_view("judge_ft", tv, y, agree, doc_fold, groups, None, n_boot=30)
    assert r["mode"] == "cross_fitted_3_fold" and r["structure"] in ("pooled", "per_type")
    assert {t["variant"] for t in r["tau"]} == {"v2", "v2_agree"}
    assert {t["target"] for t in r["tau"]} == set(c2.TARGETS)
    assert not np.isnan(r["oof"]["v2_agree"][tv.emitted]).any()
    # the OOF probability of a row never came from a model fitted on its own supplier fold
    ids = tv.table.doc_ids.tolist()
    for (fit, pred), _h in zip(
        cf.fold_splits(ids, doc_fold, groups), sorted(set(doc_fold.values())), strict=True
    ):
        assert not {groups[ids[i]] for i in fit} & {groups[ids[i]] for i in pred}


def test_view_analysis_aborts_on_test_ids() -> None:
    tv, y, agree, doc_fold, groups = _synthetic_view(30)
    keys = [cf.FieldKey("test_0001", k.scope, k.field, k.row_idx) for k in tv.table.keys]
    tv.table.keys = keys
    with pytest.raises(c2.NoTestDataError):
        c3.analyze_fold_view("judge_ft", tv, y, agree, groups, {}, n_boot=10)
    with pytest.raises(c2.NoTestDataError):
        c3.analyze_full_view("judge_ft", tv, y, agree, doc_fold, groups, None, n_boot=10)


# ---------------------------------------------------------------------------------------------
# outputs hold no values
# ---------------------------------------------------------------------------------------------

SENTINELS = ("SECRETVALUE-777", "Acme Zebra Ltd", "PN-SENTINEL-42")


def test_written_files_hold_no_extracted_value(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    hf = cf.HEADER_FIELDS["invoice"]
    ids, groups = _docs_groups(30, 6)
    ft_post: dict[str, Any] = {}
    zs_post: dict[str, Any] = {}
    for d in ids:
        hdr = {f: (SENTINELS[0] if f != "supplier_name" else SENTINELS[1]) for f in hf}
        rows = [_row(SENTINELS[2] + str(i), SENTINELS[0], SENTINELS[0], "3") for i in range(2)]
        ft_post[d] = _doc(hdr, rows)
        zs_post[d] = _doc({**hdr, "currency": None}, rows[:1])
    traces = {d: {"pages": []} for d in ids}
    tv_ft, _ = c3.build_table_v2(ids, ft_post, traces, {}, {})
    tv_zs, _ = c3.build_table_v2(ids, zs_post, traces, {}, {})
    pairs = c3.align_all(ft_post, zs_post, ids)
    ag_ft = c3.agreement_matrix(tv_ft.keys, ft_post, zs_post, pairs, True)
    ag_zs = c3.agreement_matrix(tv_zs.keys, zs_post, ft_post, pairs, False)
    y_ft = rng.uniform(size=len(tv_ft.keys)) < 0.6
    y_zs = rng.uniform(size=len(tv_zs.keys)) < 0.6
    views = [
        c3.analyze_fold_view("judge_ft", tv_ft, y_ft, ag_ft, groups, dict.fromkeys(ids), 20),
        c3.analyze_fold_view("judge_zs", tv_zs, y_zs, ag_zs, groups, dict.fromkeys(ids), 20),
    ]
    res = {"mode": "fold_exploratory", "n_docs": len(ids), "folds": [0], "views": views}
    prov = {
        "oof_runs": "oof_fold0_x", "zs_run": "zs", "head": "abc", "sha_module": "0" * 64,
        "sha_script": "1" * 64, "command": "cmd", "wall_s": 1.0, "n_boot": 20,
        "shapes_equal": {0: True}, "rules_ft": {}, "rules_zs": {},
    }  # fmt: skip
    md = cal3.render_md(res, prov)
    assert "EXPLORATORY, NOT SHIPPABLE" in md
    cal3.write_outputs(
        res, {"judge_ft": (tv_ft, y_ft, ag_ft), "judge_zs": (tv_zs, y_zs, ag_zs)}, tmp_path
    )
    (tmp_path / "calibration_v3.md").write_text(md, encoding="utf-8")
    files = list(tmp_path.iterdir())
    assert {f.name for f in files} >= {"lift_table.csv", "oof_fields_v3.csv", "calibration_v3.md"}
    for f in files:
        text = f.read_text(encoding="utf-8")
        assert not any(s in text for s in SENTINELS), f.name


def test_full_mode_banner_is_not_the_exploratory_label() -> None:
    tv, y, agree, doc_fold, groups = _synthetic_view(60)
    v = c3.analyze_full_view("judge_ft", tv, y, agree, doc_fold, groups, None, n_boot=20)
    res = {"mode": v["mode"], "n_docs": 60, "folds": [0, 1, 2], "views": [v]}
    prov = {
        "oof_runs": "a, b, c", "zs_run": "zs", "head": "abc", "sha_module": "0" * 64,
        "sha_script": "1" * 64, "command": "cmd", "wall_s": 1.0, "n_boot": 20,
        "shapes_equal": {}, "rules_ft": {}, "rules_zs": {},
    }  # fmt: skip
    md = cal3.render_md(res, prov)
    assert "NOT SHIPPABLE" not in md and "cross-fitted" in md and "Nested per-field-type" in md
