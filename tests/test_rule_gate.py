"""Unit tests for shipdoc.rules and scripts/rule_gate.py on synthetic fixtures (no I/O)."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

from shipdoc import rules

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("rule_gate", ROOT / "scripts" / "rule_gate.py")
rg = importlib.util.module_from_spec(_spec)
sys.modules["rule_gate"] = rg  # dataclasses resolve annotations via sys.modules
_spec.loader.exec_module(rg)

CPN, PO = rules.CPN, rules.PO


class FakeScorer:
    """Stand-in for the official scorer's `same`: alphanumeric case-insensitive equality."""

    @staticmethod
    def same(field: str, a: Any, b: Any) -> bool:
        na = re.sub(r"[^0-9a-z]", "", str(a).lower())
        nb = re.sub(r"[^0-9a-z]", "", str(b).lower())
        return bool(na) and na == nb


def wb(**header: Any) -> dict[str, Any]:
    return {"doc_type": "waybill", "header": dict(header), "line_items": []}


def inv(*rows: tuple[Any, Any]) -> dict[str, Any]:
    items = [{"supplier_part_number": f"S{i}", CPN: c, PO: p} for i, (c, p) in enumerate(rows)]
    return {"doc_type": "invoice", "header": {}, "line_items": items}


SHAPES = rules.SlotShapes(cpn_only=frozenset({"AA-9999"}), po_only=frozenset({"9999999"}))


# ---- R1 ------------------------------------------------------------------------------------


def test_r1_fills_empty_carrier_from_supplier_slot() -> None:
    doc = wb(carrier=None, mawb="123-12345678")
    new, ch = rules.carrier_from_supplier(doc, "Acme Air")
    assert new["header"]["carrier"] == "Acme Air"
    assert [c.field for c in ch] == ["carrier"]
    assert doc["header"]["carrier"] is None  # input not mutated


def test_r1_never_overrides_model_carrier_or_fills_from_empty_supplier() -> None:
    new, ch = rules.carrier_from_supplier(wb(carrier="Model Cargo"), "Acme Air")
    assert new["header"]["carrier"] == "Model Cargo" and ch == []
    new, ch = rules.carrier_from_supplier(wb(carrier=None), "  ")
    assert new["header"]["carrier"] is None and ch == []
    new, ch = rules.carrier_from_supplier(inv((None, None)), "Acme Air")
    assert ch == []  # invoices are never touched


# ---- R2 ------------------------------------------------------------------------------------


def test_r2_unique_match_fills_mawb_and_hawb() -> None:
    text = "AWB 176-12345678 house AB12345678 end"
    new, ch = rules.pattern_backfill(wb(mawb=None, hawb=""), text)
    assert new["header"]["mawb"] == "176-12345678" and new["header"]["hawb"] == "AB12345678"
    assert {c.field for c in ch} == {"mawb", "hawb"}


def test_r2_ambiguous_or_missing_leaves_null_and_keeps_model_value() -> None:
    two = "176-12345678 and 180-87654321, no house number"
    new, ch = rules.pattern_backfill(wb(mawb=None, hawb=None), two)
    assert new["header"] == {"mawb": None, "hawb": None} and ch == []
    same_twice = "176-12345678 ... 176-12345678"  # one DISTINCT candidate: allowed
    new, _ = rules.pattern_backfill(wb(mawb=None), same_twice)
    assert new["header"]["mawb"] == "176-12345678"
    new, ch = rules.pattern_backfill(wb(mawb="999-00000000"), "176-12345678")
    assert new["header"]["mawb"] == "999-00000000" and ch == []


def test_r2_pattern_is_not_a_substring_match() -> None:
    new, ch = rules.pattern_backfill(wb(mawb=None, hawb=None), "1176-123456789 XAB123456789")
    assert ch == []


# ---- R3 ------------------------------------------------------------------------------------


def test_r3_moves_lone_value_by_exclusive_shape() -> None:
    new, ch = rules.slot_shape(inv(("1234567", None), (None, "AB-1234")), SHAPES)
    assert (new["line_items"][0][CPN], new["line_items"][0][PO]) == (None, "1234567")
    assert (new["line_items"][1][CPN], new["line_items"][1][PO]) == ("AB-1234", None)
    assert len(ch) == 2


def test_r3_swaps_only_when_both_present_and_both_shapes_point_across() -> None:
    new, ch = rules.slot_shape(inv(("1234567", "AB-1234")), SHAPES)
    assert (new["line_items"][0][CPN], new["line_items"][0][PO]) == ("AB-1234", "1234567")
    assert len(ch) == 1
    # only one of the two shapes is known: ambiguous, left alone
    new, ch = rules.slot_shape(inv(("1234567", "ZZZ")), SHAPES)
    assert ch == [] and new["line_items"][0][CPN] == "1234567"
    # already in the right slots
    assert rules.slot_shape(inv(("AB-1234", "1234567")), SHAPES)[1] == []


def test_r3_ambiguous_or_unknown_shape_and_both_empty_untouched() -> None:
    shapes = rules.learn_slot_shapes([inv(("AB-1234", "1234567"), ("1234567", "AB-1234"))])
    assert shapes.cpn_only == frozenset() and shapes.po_only == frozenset()  # shared shapes
    assert rules.slot_shape(inv(("1234567", None)), shapes)[1] == []
    assert rules.slot_shape(inv((None, None), ("Q9", None)), SHAPES)[1] == []


def test_learn_slot_shapes_uses_given_invoice_labels_only() -> None:
    labels = [inv(("AB-1234", "1234567")), wb(carrier="x"), inv((None, "12345678"))]
    got = rules.learn_slot_shapes(labels)
    assert got.cpn_only == frozenset({"AA-9999"})
    assert got.po_only == frozenset({"9999999", "99999999"})


def _fold_fixture() -> tuple[list[dict[str, Any]], dict[str, int], dict[str, str]]:
    """6 invoice docs, 3 folds, 1 supplier group per fold (2 docs each).

    The cpn shape "AA-9999" appears ONLY in the gold of the fold-1 docs.
    """
    gold = {
        "a1": inv((None, "1234567")),
        "a2": inv((None, "1234567")),
        "b1": inv(("AB-1234", None)),  # fold 1 only: cpn shape
        "b2": inv(("AB-1234", None)),
        "c1": inv((None, "12345678")),
        "c2": inv((None, "12345678")),
    }
    labels = [{**g, "doc_id": d} for d, g in gold.items()]
    doc_fold = {"a1": 0, "a2": 0, "b1": 1, "b2": 1, "c1": 2, "c2": 2}
    groups = {d: f"inv_g0{doc_fold[d]}" for d in gold}
    return labels, doc_fold, groups


def test_fold_shapes_never_learn_from_the_held_out_fold_or_its_suppliers() -> None:
    labels, doc_fold, groups = _fold_fixture()
    got = rg.fold_shapes(labels, doc_fold, groups)
    assert sorted(got) == [0, 1, 2]
    assert "AA-9999" not in got[1][0].cpn_only  # fold 1's own shape is invisible to fold 1
    assert "AA-9999" in got[0][0].cpn_only and "AA-9999" in got[2][0].cpn_only
    for k, (_, learn_ids) in got.items():
        held = {d for d, f in doc_fold.items() if f == k}
        assert learn_ids and learn_ids.isdisjoint(held)
        assert {groups[d] for d in learn_ids}.isdisjoint({groups[d] for d in held})
        assert learn_ids == {d for d, f in doc_fold.items() if f != k}


def test_fold_shapes_fail_closed_when_folds_are_not_supplier_grouped() -> None:
    labels, doc_fold, groups = _fold_fixture()
    groups = {**groups, "a1": "inv_g01"}  # a fold-0 doc now shares fold 1's supplier group
    with pytest.raises(ValueError, match="overlaps"):
        rg.fold_shapes(labels, doc_fold, groups)


def test_fold_shapes_ignore_labels_outside_the_folds() -> None:
    labels, doc_fold, groups = _fold_fixture()
    labels.append({**inv((None, "123")), "doc_id": "stray"})
    got = rg.fold_shapes(labels, doc_fold, groups)
    assert all("stray" not in ids for _, ids in got.values())


def test_r3_heldout_pools_each_doc_once_with_its_own_fold_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels, doc_fold, groups = _fold_fixture()
    per_fold = rg.fold_shapes(labels, doc_fold, groups)
    # the model puts the cpn-shaped value in the po slot for every doc; folds 0 and 2 learn the
    # cpn shape from fold 1 gold, fold 1 (its own gold held out) cannot and must leave its docs.
    pred = {d: inv((None, "AB-1234")) for d in doc_fold}
    gold = {d: inv(("AB-1234", None)) for d in doc_fold}
    inp = make_inputs(pred, gold, groups)
    monkeypatch.setattr(rg, "rescore", lambda i, p, n: {"lo": 0.01, "hi": 0.02, "delta": 0.015})
    seen: list[str] = []
    real = rg.apply_rule

    def spy(i: Any, fn: Any, doc_type: str, cells: int) -> Any:
        seen.extend(sorted(i.pred))
        return real(i, fn, doc_type, cells)

    monkeypatch.setattr(rg, "apply_rule", spy)
    # stand-in for the official scorer's pairing: row 0 pairs with row 0
    monkeypatch.setattr(rg.ev, "_pair_rows", lambda sc, pr, gr: ([(0, 0)], [], [], []))
    res = rg.evaluate_r3_heldout(inp, per_fold, doc_fold, 10)
    assert seen == sorted(doc_fold)  # one pooled pass: every doc evaluated exactly once
    assert res.outcome.eligible == 6
    assert res.outcome.groups["inv_g01"]["n"] == 2
    assert res.outcome.touched == 4  # folds 0 and 2 patch their 4 docs, fold 1's 2 stay as is
    assert "inv_g01" not in {g for g, v in res.outcome.groups.items() if v["fixed"]}


def test_require_fold_docs_fails_closed_on_missing_docs() -> None:
    _, doc_fold, _ = _fold_fixture()
    rg.require_fold_docs([*doc_fold, "extra"], doc_fold)  # a superset is fine
    with pytest.raises(SystemExit, match="1 missing"):
        rg.require_fold_docs([d for d in doc_fold if d != "c2"], doc_fold)


def test_r3_verdict_uses_pooled_group_nets() -> None:
    # one supplier group nets negative across the pooled folds: no ship despite a good CI
    pooled = {"inv_g01": 5, "inv_g02": 3, "inv_g03": -1}
    v = rg.ship_verdict(0.02, 0, pooled, require_group_clause=True)
    assert not v.ship and v.failed == (rg.CLAUSE_GROUP,) and "inv_g03" in v.reasons[-1]
    assert rg.ship_verdict(0.02, 0, {**pooled, "inv_g03": 0}, require_group_clause=True).ship


# ---- verdict -------------------------------------------------------------------------------


def test_ship_verdict_clauses() -> None:
    assert rg.ship_verdict(0.001, 0).ship
    v = rg.ship_verdict(0.0, 0)
    assert not v.ship and v.failed == (rg.CLAUSE_CI,)  # lower bound must be strictly > 0
    v = rg.ship_verdict(0.01, 1)
    assert not v.ship and v.failed == (rg.CLAUSE_BROKEN,)
    v = rg.ship_verdict(-0.01, 2)
    assert v.failed == (rg.CLAUSE_CI, rg.CLAUSE_BROKEN)


def test_ship_verdict_group_clause_is_r3_only() -> None:
    net = {"inv_g01": 5, "inv_g02": -1}
    assert rg.ship_verdict(0.01, 0, net).ship  # R1 / R2 ignore groups
    v = rg.ship_verdict(0.01, 0, net, require_group_clause=True)
    assert not v.ship and v.failed == (rg.CLAUSE_GROUP,) and "inv_g02" in v.reasons[-1]
    assert rg.ship_verdict(0.01, 0, {"inv_g01": 0}, require_group_clause=True).ship


# ---- unit accounting -----------------------------------------------------------------------


def make_inputs(pred: dict[str, Any], gold: dict[str, Any], groups: dict[str, str]) -> Any:
    return rg.Inputs(
        pred=pred, gold=gold, groups=groups, supplier={}, model_text={}, ocr_text={},
        sc=FakeScorer(), train_pool=[],
    )  # fmt: skip


def test_apply_rule_counts_fixed_broken_and_groups_for_header_cells() -> None:
    pred = {"d1": wb(carrier=None), "d2": wb(carrier=None), "d3": wb(carrier="Model")}
    gold = {
        "d1": wb(carrier="Acme Air"),  # fixed
        "d2": wb(carrier=None),  # gold null, rule fills: broken
        "d3": wb(carrier="Model"),  # untouched
    }
    groups = {"d1": "wb_g01", "d2": "wb_g01", "d3": "wb_g02"}
    inp = make_inputs(pred, gold, groups)
    sup = {"d1": "Acme Air", "d2": "Acme Air", "d3": "Acme Air"}
    patched, out = rg.apply_rule(
        inp, lambda d, p: rules.carrier_from_supplier(p, sup[d]), "waybill", 1
    )
    assert (out.eligible, out.fixed, out.broken, out.neutral, out.untouched) == (3, 1, 1, 0, 1)
    assert out.net == 0 and out.touched == 2
    assert out.groups["wb_g01"] == {"n": 2, "fixed": 1, "broken": 1, "neutral": 0}
    assert out.groups["wb_g02"]["n"] == 1
    assert pred["d1"]["header"]["carrier"] is None  # inputs untouched
    assert patched["d3"]["header"]["carrier"] == "Model"


def test_apply_rule_r3_uses_unpatched_pairing_and_unpaired_rows_are_neutral(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gold_doc = inv((None, "1234567"))
    pred = {"d1": inv(("1234567", None)), "d2": inv(("1234567", None))}
    pred["d2"]["line_items"][0]["supplier_part_number"] = "OTHER"  # unpaired by the pairing
    inp = make_inputs(pred, {"d1": gold_doc, "d2": gold_doc}, {"d1": "inv_g01", "d2": "inv_g01"})

    def pair_by_spn(sc: Any, pr: list[dict], gr: list[dict]) -> tuple[list, list, list, list]:
        # stand-in for the official scorer's pairing
        full = [
            (i, j)
            for i, a in enumerate(pr)
            for j, b in enumerate(gr)
            if a["supplier_part_number"] == b["supplier_part_number"]
        ]
        return full, [], [], []

    monkeypatch.setattr(rg.ev, "_pair_rows", pair_by_spn)
    _, out = rg.apply_rule(inp, lambda d, p: rules.slot_shape(p, SHAPES), "invoice", 0)
    assert (out.fixed, out.broken, out.neutral) == (1, 0, 1)


# ---- rendering and CLI guards ----------------------------------------------------------------


def _result(name: str = "R1") -> Any:
    out = rg.Outcome(eligible=2, fixed=1)
    out.groups["wb_g01"] = {"n": 2, "fixed": 1, "broken": 0, "neutral": 0}
    stats = {"base_overall": 0.5, "overall": 0.51, "delta": 0.01, "lo": 0.001, "hi": 0.02}
    return rg.RuleResult(name, "cells", "note", out, stats, rg.ship_verdict(0.001, 0))


def test_smoke_report_has_banner_and_never_a_verdict() -> None:
    text = rg.render_report("run", ("DEV SMOKE", 3, [_result()]), None, smoke=True)
    assert text.startswith("!")
    assert "NOT A GATE RESULT" in text
    assert "DO NOT SHIP" not in text.split("**Ship rule.**")[1].split("## ")[1]
    assert "smoke only" in text
    gate = rg.render_report("run", ("TRAIN verdict", 3, [_result()]), None, smoke=False)
    assert "NOT A GATE RESULT" not in gate and "| SHIP |" in gate


def test_confirmation_is_a_separate_labelled_section() -> None:
    text = rg.render_report(
        "run", ("TRAIN verdict", 3, [_result()]), ("Dev confirmation", 2, [_result()]), False
    )
    assert text.index("TRAIN verdict") < text.index("CONFIRMATION ONLY")


def test_cli_fails_closed_on_wrong_train_doc_count(tmp_path: Path) -> None:
    (tmp_path / "predictions.json").write_text(
        json.dumps({"train_0001": {}, "dev_0001": {}}), encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="expected 400 train docs"):
        rg.main(["--run-dir", str(tmp_path)])
    assert rg.doc_ids_with_prefix(tmp_path, "train_") == ["train_0001"]  # dev excluded
    with pytest.raises(SystemExit):
        rg.main(["--run-dir", str(tmp_path), "--allow-dev-smoke", "--confirm-dev"])


def test_cli_fails_closed_when_the_run_lacks_fold_docs(tmp_path: Path) -> None:
    folds = json.loads((ROOT / "splits" / "folds.json").read_text(encoding="utf-8"))
    ids = sorted(d for d in folds["doc_fold"] if d.startswith("train_"))
    pred = dict.fromkeys(ids, {})  # all 400 train docs, no dev docs
    (tmp_path / "predictions.json").write_text(json.dumps(pred), encoding="utf-8")
    with pytest.raises(SystemExit, match="R3 held-out needs all 500 docs"):
        rg.main(["--run-dir", str(tmp_path)])


def test_report_renders_r3_section_and_labels_the_in_sample_row() -> None:
    held = _result("R3 (supplier-held-out, folds pooled)")
    ref = _result("R3 (all-gold shapes)")
    ref.informational = True
    r3 = ("R3 supplier-held-out", 5, [held, ref])
    text = rg.render_report("run", ("R1 / R2 TRAIN", 3, [_result()]), None, False, r3)
    assert "# R3 supplier-held-out" in text and "no verdict (in-sample reference)" in text
    assert text.index("R1 / R2 TRAIN") < text.index("# R3 supplier-held-out")


# ---- OOF mode (synthetic mini run folder; no real OOF run exists) ----------------------------


def _oof_world() -> tuple[dict[str, Any], dict[str, int], dict[str, str], dict[str, Any]]:
    """3 folds x (2 invoice docs + 1 waybill doc); the cpn shape lives only in fold 1's gold."""
    gold: dict[str, Any] = {
        "a1": inv((None, "1234567")),
        "a2": inv((None, "1234567")),
        "b1": inv(("AB-1234", None)),
        "b2": inv(("AB-1234", None)),
        "c1": inv((None, "12345678")),
        "c2": inv((None, "12345678")),
        "wa": wb(carrier="Acme"),
        "wb": wb(carrier="Beta"),
        "wc": wb(carrier="Gamma"),
    }
    doc_fold = dict(a1=0, a2=0, wa=0, b1=1, b2=1, wb=1, c1=2, c2=2, wc=2)
    groups = {d: ("wb_g0" if d.startswith("w") else "inv_g0") + str(doc_fold[d]) for d in gold}
    folds = {
        "folds": [
            {"fold": k, "val_doc_ids": sorted(d for d, f in doc_fold.items() if f == k)}
            for k in range(3)
        ],
        "doc_fold": doc_fold,
    }
    return gold, doc_fold, groups, folds


def _oof_run(
    tmp: Path, ids: list[str], oof: Any = "ok", status: str = "complete",
    name: str = "oof_fold0_abc1234", config_hash: str | None = None,
) -> Path:  # fmt: skip
    """Write a mini run folder; the model emits the cpn-shaped value in the po slot / no carrier."""
    run = tmp / name
    run.mkdir()
    pred = {d: (inv((None, "AB-1234")) if d[0] != "w" else wb(carrier=None)) for d in ids}
    (run / "predictions.json").write_text(json.dumps(pred), encoding="utf-8")
    (run / "trace.jsonl").write_text(
        "".join(json.dumps({"doc_id": d, "pages": [{"raw_text": "{}"}]}) + "\n" for d in ids),
        encoding="utf-8",
    )
    (run / "progress.json").write_text(json.dumps({"status": status}), encoding="utf-8")
    man: dict[str, Any] = {}
    if config_hash is not None:
        man["config"] = {"name": "cfg", "hash": config_hash}
    if oof == "ok":
        man["oof"] = {
            "fold": 0, "adapter_sha256": "a" * 64, "train_code_sha": "b" * 40,
            "verification": {"ok": True}, "n_inference_docs": len(ids),
            "batch": {"used": 4}, "guard": {"ran": True, "fallback_to_1": False},
        }  # fmt: skip
    elif oof is not None:
        man["oof"] = oof
    (run / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    return run


@pytest.fixture
def oof_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    gold, doc_fold, groups, folds = _oof_world()
    root = tmp_path / "repo"
    (root / "splits").mkdir(parents=True)
    (root / "meta").mkdir()
    (root / "splits" / "folds.json").write_text(json.dumps(folds), encoding="utf-8")
    (root / "meta" / "supplier_groups.json").write_text(json.dumps(groups), encoding="utf-8")
    monkeypatch.setattr(rg, "ROOT", root)
    labels = [{**g, "doc_id": d} for d, g in gold.items()]
    monkeypatch.setattr(
        rg.meta_mod, "load_labels", lambda split: labels if split == "train" else []
    )
    seen: list[list[str]] = []

    def fake_load(run_dir: Path, ids: Any, ocr: Any, split_of: Any) -> Any:
        seen.append(list(ids))
        pred = json.loads((run_dir / "predictions.json").read_text(encoding="utf-8"))
        inp = make_inputs({d: pred[d] for d in ids}, {d: gold[d] for d in ids}, groups)
        inp.supplier = dict.fromkeys(ids, "Acme")
        return inp

    monkeypatch.setattr(rg, "load_inputs", fake_load)
    monkeypatch.setattr(rg, "require_ocr", lambda *a, **k: None)
    monkeypatch.setattr(rg, "rescore", lambda i, p, n: {"lo": 0.01, "hi": 0.02, "delta": 0.015})
    monkeypatch.setattr(rg.ev, "_pair_rows", lambda sc, pr, gr: ([(0, 0)], [], [], []))
    return {"seen": seen, "tmp": tmp_path, "gold": gold, "groups": groups, "doc_fold": doc_fold}


FOLD0 = ["a1", "a2", "wa"]


def test_oof_mode_evaluates_only_fold_docs_and_writes_the_report(oof_env: dict[str, Any]) -> None:
    run = _oof_run(oof_env["tmp"], FOLD0)
    out = oof_env["tmp"] / "rep.md"
    assert rg.main(["--oof-run-dir", str(run), "--fold", "0", "--out", str(out)]) == 0
    assert oof_env["seen"] == [FOLD0]
    text = out.read_text(encoding="utf-8")
    assert "fold 0 OOF" in text and "Docs evaluated: 3." in text
    assert "only if it passes on this fold's OOF outputs too" in text
    assert "lack power" in text and "adapter sha256" in text


def test_oof_default_report_path_is_per_fold(oof_env: dict[str, Any]) -> None:
    (rg.ROOT / "reports").mkdir()
    run = _oof_run(oof_env["tmp"], FOLD0)
    rg.main(["--oof-run-dir", str(run)])
    assert (rg.ROOT / "reports" / "rule_gate_ft_fold0.md").is_file()


def test_oof_r3_shapes_exclude_the_held_out_fold(oof_env: dict[str, Any]) -> None:
    labels = rg.meta_mod.load_labels("train") + rg.meta_mod.load_labels("dev")
    dfold, groups, gold = oof_env["doc_fold"], oof_env["groups"], oof_env["gold"]
    pred = {d: inv((None, "AB-1234")) for d in ("b1", "b2")}  # cpn-shaped value in the po slot
    inp1 = make_inputs(pred, {d: gold[d] for d in pred}, groups)
    r1 = rg.run_oof_rules(inp1, 1, labels, dfold, groups, 10)[-1]
    assert "excluding fold 1" in r1.name
    assert r1.outcome.touched == 0  # the cpn shape exists only in fold 1's own gold
    pred0 = {d: inv((None, "AB-1234")) for d in ("a1", "a2")}
    inp0 = make_inputs(pred0, {d: inv(("AB-1234", None)) for d in pred0}, groups)
    r0 = rg.run_oof_rules(inp0, 0, labels, dfold, groups, 10)[-1]
    assert r0.outcome.fixed == 2  # fold 0 learns it from fold 1 gold
    with pytest.raises(SystemExit, match="not in fold 1"):
        rg.run_oof_rules(inp0, 1, labels, dfold, groups, 10)


def test_oof_r3_fails_closed_when_folds_are_not_supplier_disjoint(oof_env: dict[str, Any]) -> None:
    labels = rg.meta_mod.load_labels("train")
    groups = {**oof_env["groups"], "a1": "inv_g01"}  # a fold-0 doc in fold 1's supplier group
    inp = make_inputs({"a1": inv((None, "1")), "a2": inv((None, "1"))}, oof_env["gold"], groups)
    with pytest.raises(SystemExit, match="overlaps"):
        rg.run_oof_rules(inp, 0, labels, oof_env["doc_fold"], groups, 10)


@pytest.mark.parametrize(
    ("kwargs", "args", "match"),
    [
        ({"oof": None}, [], "no `oof` section"),
        ({"oof": {"fold": 0}}, [], "verification is not ok"),
        ({"oof": {"fold": 0, "verification": {"ok": True}}}, [], "lacks the adapter sha"),
        ({}, ["--fold", "1"], "contradicts the manifest"),
        ({"status": "running"}, [], "not complete"),
        ({"ids": ["a1", "a2"]}, [], "not the fold's docs"),
        ({"ids": [*FOLD0, "b1"]}, [], "not the fold's docs"),
    ],
)
def test_oof_refusals_write_nothing(
    oof_env: dict[str, Any], kwargs: dict[str, Any], args: list[str], match: str
) -> None:
    ids = kwargs.pop("ids", FOLD0)
    run = _oof_run(oof_env["tmp"], ids, **kwargs)
    out = oof_env["tmp"] / "rep.md"
    with pytest.raises(SystemExit, match=match):
        rg.main(["--oof-run-dir", str(run), "--out", str(out), *args])
    assert not out.exists()


def test_oof_refuses_a_trace_that_lacks_docs(oof_env: dict[str, Any]) -> None:
    run = _oof_run(oof_env["tmp"], FOLD0)
    lines = (run / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    (run / "trace.jsonl").write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="trace.jsonl"):
        rg.main(["--oof-run-dir", str(run)])
    (run / "trace.jsonl").unlink()
    with pytest.raises(SystemExit, match="missing"):
        rg.main(["--oof-run-dir", str(run)])


def test_oof_refuses_when_gold_is_missing(
    oof_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _oof_run(oof_env["tmp"], FOLD0)

    def no_gold(*a: Any, **k: Any) -> Any:
        raise SystemExit("2 docs lack gold or prediction")  # what load_inputs raises

    monkeypatch.setattr(rg, "load_inputs", no_gold)
    with pytest.raises(SystemExit, match="lack gold"):
        rg.main(["--oof-run-dir", str(run)])


def test_require_ocr_refuses_missing_or_short_waybill_pages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run = _oof_run(tmp_path, ["a1", "wa"])
    inp = make_inputs({}, {"a1": inv(), "wa": wb(carrier="x")}, {})
    monkeypatch.setattr(rg, "doc_pages", lambda d, root=None: [object()])
    rg.require_ocr(run, inp, None)  # 1 trace page, 1 OCR page: fine
    monkeypatch.setattr(rg, "doc_pages", lambda d, root=None: [])
    with pytest.raises(SystemExit, match="OCR cache lacks waybill pages for 1 docs"):
        rg.require_ocr(run, inp, None)


def test_oof_option_conflicts_and_plumbing_needs_oof_dir(tmp_path: Path) -> None:
    for bad in (
        ["--oof-run-dir", str(tmp_path), "--run-dir", str(tmp_path)],
        ["--oof-run-dir", str(tmp_path), "--allow-dev-smoke"],
        ["--run-dir", str(tmp_path), "--plumbing-check"],
        ["--run-dir", str(tmp_path), "--zs-run-dir", str(tmp_path)],
        [],
    ):
        with pytest.raises(SystemExit):
            rg.main(bad)


def test_plumbing_check_skips_oof_section_prints_banner_and_never_writes(
    oof_env: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    ids = sorted(oof_env["gold"])  # a non-OOF run holding every doc: restricted to the fold
    run = _oof_run(oof_env["tmp"], ids, oof=None)
    out = oof_env["tmp"] / "rep.md"
    argv = ["--oof-run-dir", str(run), "--plumbing-check", "--out", str(out)]
    assert rg.main([*argv, "--zs-run-dir", str(run)]) == 0
    text = capsys.readouterr().out
    assert text.startswith("!") and "PLUMBING CHECK NOT A RESULT" in text
    assert "Same rules on the zero-shot run" in text and "no oof section" in text
    assert oof_env["seen"][0] == FOLD0 and not out.exists()


@pytest.mark.parametrize(
    ("ft_hash", "zs_hash", "ok"),
    [("hn", "hn", True), ("h1", "h1", True), ("hn", "h1", False), ("h1", "hn", False)],
)
def test_oof_side_by_side_zs_run_must_share_the_resolution(
    oof_env: dict[str, Any], ft_hash: str, zs_hash: str, ok: bool
) -> None:
    ft = _oof_run(oof_env["tmp"], FOLD0, config_hash=ft_hash)
    zs = _oof_run(
        oof_env["tmp"], sorted(oof_env["gold"]), oof=None, name="zs_run", config_hash=zs_hash
    )
    out = oof_env["tmp"] / "rep.md"
    argv = ["--oof-run-dir", str(ft), "--zs-run-dir", str(zs), "--out", str(out)]
    if ok:
        assert rg.main(argv) == 0 and out.is_file()
    else:
        with pytest.raises(SystemExit, match=r"mixed-resolution.*OOF run.*ZS run `zs_run`"):
            rg.main(argv)
        assert not out.exists()


def _res(broken: int, delta: float, lo: float, group_net: int = 0) -> Any:
    out = rg.Outcome(eligible=10, fixed=2, broken=broken)
    out.groups["g"] = {"n": 1, "fixed": 2, "broken": broken, "neutral": 0}
    stats = {"base_overall": 0.5, "overall": 0.5 + delta, "delta": delta, "lo": lo, "hi": 0.05}
    verdict = rg.ship_verdict(lo, broken, {"g": group_net}, True)
    return rg.RuleResult("R", "u", "n", out, stats, verdict)


def test_failure_wording_separates_insufficient_power_from_failed() -> None:
    assert rg.failure_kind(_res(0, 0.01, 0.001)) == ""  # ships
    power = _res(0, 0.01, -0.002)
    assert rg.failure_kind(power) == rg.INSUFFICIENT_POWER
    assert "insufficient power" in rg.failure_sentence(power)
    assert "broken 0" in rg.failure_sentence(power) and "-0.20 pts" in rg.failure_sentence(power)
    assert rg.failure_kind(_res(1, 0.01, -0.002)) == rg.FAILED_BROKEN
    assert rg.failure_kind(_res(1, 0.01, 0.002)) == rg.FAILED_BROKEN  # CI ok, still broken
    assert rg.failure_kind(_res(0, 0.0, -0.002)).startswith("FAILED (no positive effect")
    assert rg.failure_kind(_res(0, 0.01, 0.002, group_net=-1)).startswith("FAILED (a supplier")
    # the table verdict string stays DO NOT SHIP for both
    assert rg.verdict_text(power, smoke=False) == "DO NOT SHIP"
    assert rg.verdict_text(_res(1, 0.01, 0.002), smoke=False) == "DO NOT SHIP"
