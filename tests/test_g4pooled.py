"""Pooled 3-fold final-system decision: the pure rule, cross-fold validation, pooling, and the
driver on a synthetic miniature world (3 folds x (2 invoices + 1 waybill); no model, no document).

Helpers (gold, traces, runs) are reused from ``test_g4_fold.py``; the ZS arm gets the invoice
quantity wrong, the FT arm is right, so FT wins by construction unless a test tampers with it.
"""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest
from test_g4_fold import make_run, world  # noqa: F401  (pytest prepend mode puts tests/ on path)

from shipdoc import g4pooled, postrules
from shipdoc.g4pooled import FT_LABEL, ZS_LABEL, PoolError, decide

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import g4_pooled as g4p  # noqa: E402

rg = g4p.rg
g4f = g4p.g4f

# ---- decide(): truth table ---------------------------------------------------------------------


def counts(ff: Any, on: Any) -> dict[str, Any]:
    return {"false_fill_total": ff, "over_null_total": on}


@pytest.mark.parametrize(
    ("lo", "ft", "zs", "final", "failed"),
    [
        (0.01, counts(3, 5), counts(3, 5), FT_LABEL, []),  # equal counts pass
        (0.01, counts(2, 4), counts(3, 5), FT_LABEL, []),  # fewer pass
        (1e-9, counts(3, 5), counts(3, 5), FT_LABEL, []),  # any strictly positive bound passes
        (0.0, counts(3, 5), counts(3, 5), ZS_LABEL, ["C1"]),  # bound exactly 0 fails
        (-1e-9, counts(0, 0), counts(9, 9), ZS_LABEL, ["C1"]),
        (0.01, counts(4, 5), counts(3, 5), ZS_LABEL, ["C2"]),  # one more false fill fails
        (0.01, counts(3, 6), counts(3, 5), ZS_LABEL, ["C3"]),  # one more over-null fails
        (0.01, counts(4, 6), counts(3, 5), ZS_LABEL, ["C2", "C3"]),
        (0.0, counts(4, 6), counts(3, 5), ZS_LABEL, ["C1", "C2", "C3"]),  # all clauses reported
        (-0.02, counts(3, 5), counts(3, 5), ZS_LABEL, ["C1"]),
    ],
)
def test_decide_truth_table(lo: float, ft: Any, zs: Any, final: str, failed: list[str]) -> None:
    d = decide(lo, ft, zs)
    assert d.final == final and [c.key for c in d.failed] == failed
    assert d.ft_selected == (final == FT_LABEL)
    assert d.line.startswith(f"FINAL SYSTEM: {FT_LABEL}" if d.ft_selected else "FINAL SYSTEM: ZS+")
    assert all(f"{k} failed" in d.line for k in failed)
    assert [c.key for c in d.clauses] == ["C1", "C2", "C3"]  # every clause always evaluated


def test_decision_line_texts_are_exact_for_the_two_outcomes() -> None:
    assert decide(0.01, counts(1, 1), counts(1, 1)).line == "FINAL SYSTEM: FT+rules"
    line = decide(0.0, counts(1, 1), counts(1, 1)).line
    assert line.startswith("FINAL SYSTEM: ZS+rules (v1) [C1 failed:")
    assert "equal to 0" in line and "strict inequality" in line


@pytest.mark.parametrize(
    ("lo", "ft", "zs", "failed"),
    [
        (math.nan, counts(1, 1), counts(1, 1), ["C1"]),
        (math.inf, counts(1, 1), counts(1, 1), ["C1"]),  # not finite: closed, not a pass
        (None, counts(1, 1), counts(1, 1), ["C1"]),
        (0.01, counts(math.nan, 1), counts(1, 1), ["C2"]),
        (0.01, counts(1, 1), counts(1, math.nan), ["C3"]),
        (0.01, counts(1, 1), counts(math.nan, math.nan), ["C2", "C3"]),
        (0.01, {"false_fill_total": 1}, counts(1, 1), ["C3"]),  # missing key
        (0.01, counts(None, 1), counts(1, 1), ["C2"]),
        (0.01, counts(True, 1), counts(1, 1), ["C2"]),  # a bool is not a count
        (0.01, counts("0", 1), counts(1, 1), ["C2"]),
    ],
)
def test_decide_fails_closed_on_nan_missing_or_non_numeric(
    lo: Any, ft: Any, zs: Any, failed: list[str]
) -> None:
    d = decide(lo, ft, zs)
    assert d.final == ZS_LABEL and [c.key for c in d.failed] == failed
    assert "failed closed" in d.line


def test_equal_arms_never_select_ft() -> None:
    same = counts(7, 11)
    for lo in (0.0, -0.0, -1e-12):
        assert decide(lo, same, copy.deepcopy(same)).final == ZS_LABEL


def test_sign_convention_ft_minus_zs_via_compare_models() -> None:
    """decide_from_pair reads zero_shot = ZS, oof = FT: a worse FT must not pass C1."""
    w = world()
    good = {d: copy.deepcopy(g) for d, g in w["gold"].items()}
    bad = copy.deepcopy(good)
    for p in bad.values():
        if p["doc_type"] == "invoice":
            for r in p["line_items"]:
                r["quantity"] = "7"
    from shipdoc import oof

    ft_wins = oof.compare_models(bad, good, w["gold"], None, 20, 42)
    ft_loses = oof.compare_models(good, bad, w["gold"], None, 20, 42)
    assert g4pooled.decide_from_pair(ft_wins).ft_selected
    d = g4pooled.decide_from_pair(ft_loses)
    assert not d.ft_selected and [c.key for c in d.failed] == ["C1"]


def test_r3_broken_folds_ignore_fold_0_and_folds_where_r3_does_not_fire() -> None:
    def eff(touched: int, broken: int) -> dict[str, Any]:
        return {"ft": {"R3": {"touched_docs": touched, "broken": broken}}}

    assert g4pooled.r3_broken_folds({0: eff(5, 3), 1: eff(2, 1), 2: eff(0, 0)}) == [1]
    assert g4pooled.r3_broken_folds({0: eff(5, 3), 1: eff(2, 0), 2: eff(3, 0)}) == []


# ---- cross-fold validation and pooling -----------------------------------------------------------


def manifest(k: int, **over: Any) -> dict[str, Any]:
    sec = {
        "fold": k, "adapter_sha256": f"{k:064d}", "train_code_sha": "b" * 40,
        "verification": {"ok": True}, "batch": {"used": 4}, "zero_shot_run": "zs_run",
        "guard": {"ran": True, "fallback_to_1": False},
    }  # fmt: skip
    m = {
        "code_sha": "c" * 40, "batch_size": 4, "output_format": "compact",
        "config": {"hash": "h1"}, "model": {"revision": "r1"}, "oof": sec,
    }  # fmt: skip
    for key, val in over.items():
        if key in sec:
            sec[key] = val
        elif key == "batch_used":
            sec["batch"] = {"used": val}
        else:
            m[key] = val
    return m


def facts3(**over_by_fold: dict[str, Any]) -> list[g4pooled.RunFacts]:
    return [g4pooled.run_facts(manifest(k, **over_by_fold.get(f"f{k}", {}))) for k in range(3)]


def test_cross_fold_accepts_one_consistent_run_per_fold() -> None:
    g4pooled.check_cross_fold(facts3(), [0, 1, 2])


@pytest.mark.parametrize(
    ("over", "match"),
    [
        ({"f1": {"train_code_sha": "d" * 40}}, "training code SHA differs"),
        ({"f2": {"batch_used": 1}}, "batch size .*differs"),
        ({"f2": {"batch_size": 8}}, "batch size .*differs"),
        ({"f0": {"config": {"hash": "h2"}}}, "config_hash differs"),
        ({"f0": {"model": {"revision": "r2"}}}, "model_revision differs"),
        ({"f1": {"output_format": "json"}}, "output_format differs"),
        ({"f1": {"zero_shot_run": "other_zs"}}, "zero_shot_run differs"),
        ({"f1": {"adapter_sha256": f"{0:064d}"}}, "share one adapter"),
    ],
)
def test_cross_fold_refuses_mismatches(over: dict[str, Any], match: str) -> None:
    with pytest.raises(PoolError, match=match):
        g4pooled.check_cross_fold(facts3(**over), [0, 1, 2])


def test_cross_fold_refuses_wrong_or_duplicate_folds_and_different_inference_keys() -> None:
    f = facts3()
    with pytest.raises(PoolError, match="cover folds"):
        g4pooled.check_cross_fold(f[:2], [0, 1, 2])
    with pytest.raises(PoolError, match="cover folds"):
        g4pooled.check_cross_fold([f[0], f[0], f[1]], [0, 1, 2])

    def ver(v: str) -> dict[str, Any]:
        return {
            "rows": [{"name": "inference key max_pixels", "found": v}, {"name": "x", "found": 1}]
        }

    mans = [manifest(k) for k in range(3)]
    same = [g4pooled.run_facts(m, ver("1024")) for m in mans]
    g4pooled.check_cross_fold(same, [0, 1, 2])
    diff = [*same[:2], g4pooled.run_facts(mans[2], ver("2048"))]
    with pytest.raises(PoolError, match="inference keys differ"):
        g4pooled.check_cross_fold(diff, [0, 1, 2])
    mixed = [*same[:2], g4pooled.run_facts(mans[2], None)]
    with pytest.raises(PoolError, match="inference keys differ"):
        g4pooled.check_cross_fold(mixed, [0, 1, 2])


@pytest.mark.parametrize("gone", ["config", "model", "batch_size", "output_format"])
def test_run_facts_refuses_absent_comparison_fields(gone: str) -> None:
    m = manifest(0)
    m.pop(gone)
    with pytest.raises(PoolError, match="lacks"):
        g4pooled.run_facts(m)
    m = manifest(0)
    m["oof"].pop("train_code_sha")
    with pytest.raises(PoolError, match="train_code_sha"):
        g4pooled.run_facts(m)
    with pytest.raises(PoolError, match="no `oof` section"):
        g4pooled.run_facts({"code_sha": "c"})


def preds_by_fold(w: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], dict[int, list[str]]]:
    ids = {k: sorted(d for d, f in w["doc_fold"].items() if f == k) for k in range(3)}
    return {k: {d: {"doc": d} for d in v} for k, v in ids.items()}, ids


def test_pool_arms_concatenates_each_doc_exactly_once() -> None:
    w = world()
    per, ids = preds_by_fold(w)
    out = g4pooled.pool_arms(per, ids, w["doc_fold"], 9)
    assert sorted(out) == sorted(w["doc_fold"]) and len(out) == 9
    with pytest.raises(PoolError, match="expected 500"):
        g4pooled.pool_arms(per, ids, w["doc_fold"])  # default total is the 500 docs


def test_pool_arms_refuses_anything_but_a_partition() -> None:
    w = world()
    per, ids = preds_by_fold(w)
    missing = copy.deepcopy(per)
    missing[1].pop("b1")
    with pytest.raises(PoolError, match="fold 1: predictions are not exactly"):
        g4pooled.pool_arms(missing, ids, w["doc_fold"], 9)
    extra = copy.deepcopy(per)
    extra[1]["a1"] = {"doc": "a1"}  # a fold-0 doc smuggled into fold 1 (duplicate across folds)
    with pytest.raises(PoolError, match="fold 1: predictions are not exactly"):
        g4pooled.pool_arms(extra, ids, w["doc_fold"], 9)
    wrong_ids = copy.deepcopy(ids)
    wrong_ids[2] = [*ids[2][:2], "a1"]
    with pytest.raises(PoolError, match="doc list is not the fold membership"):
        g4pooled.pool_arms(per, wrong_ids, w["doc_fold"], 9)
    dup_ids = copy.deepcopy(ids)
    dup_ids[0] = [*ids[0], ids[0][0]]
    with pytest.raises(PoolError, match="doc list is not the fold membership"):
        g4pooled.pool_arms(per, dup_ids, w["doc_fold"], 9)
    with pytest.raises(PoolError, match="disagree"):
        g4pooled.pool_arms({0: per[0], 1: per[1]}, ids, w["doc_fold"], 9)
    moved = {**w["doc_fold"], "a1": 1}  # folds.json membership differs from the run's doc lists
    with pytest.raises(PoolError, match="not the fold membership"):
        g4pooled.pool_arms(per, ids, moved, 9)


# ---- the driver on the miniature world -----------------------------------------------------------


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    w = world()
    root = tmp_path / "repo"
    (root / "splits").mkdir(parents=True)
    (root / "meta").mkdir()
    (root / "splits" / "folds.json").write_text(json.dumps(w["folds"]), encoding="utf-8")
    (root / "meta" / "supplier_groups.json").write_text(json.dumps(w["groups"]), encoding="utf-8")
    for mod in (g4p, g4f, rg):
        monkeypatch.setattr(mod, "ROOT", root)
    monkeypatch.setattr(g4pooled, "EXPECTED_DOCS", 9)
    labels = [copy.deepcopy(g) for g in w["gold"].values()]
    monkeypatch.setattr(
        g4p.meta_mod, "load_labels", lambda split: labels if split == "train" else []
    )
    monkeypatch.setattr(g4f, "load_gold", lambda ids: {d: w["gold"][d] for d in ids})
    monkeypatch.setattr(
        g4f,
        "load_meta",
        lambda ids: [
            {"doc_id": d, "scanned": d.endswith("1"), "repeated_parts": False} for d in ids
        ],
    )
    monkeypatch.setattr(rg, "doc_pages", lambda d, root=None: [object()])
    monkeypatch.setattr(
        postrules, "ocr_text_for_doc", lambda d, n=None, root=None: ("no mawb", None)
    )
    zs = make_run(tmp_path, "zs_run", w["gold"], sorted(w["gold"]), "zs")
    ft = []
    for k in range(3):
        ids = sorted(d for d, f in w["doc_fold"].items() if f == k)
        run = make_run(tmp_path, f"oof_fold{k}_abc1234", w["gold"], ids, "ft", oof="ok")
        (run / "manifest.json").write_text(
            json.dumps({**manifest(k), "n_docs": len(ids)}), encoding="utf-8"
        )
        ft.append(run)
    return {**w, "tmp": tmp_path, "zs": zs, "ft": ft, "root": root}


def argv(env: dict[str, Any], *extra: str) -> list[str]:
    out: list[str] = []
    for run in env["ft"]:
        out += ["--oof-run-dir", str(run)]
    return [
        *out, "--zs-run-dir", str(env["zs"]), "--out", str(env["tmp"] / "pooled.md"),
        "--n-boot", "20", *extra,
    ]  # fmt: skip


def edit_manifest(run: Path, fn: Any) -> None:
    m = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    fn(m)
    (run / "manifest.json").write_text(json.dumps(m), encoding="utf-8")


def test_driver_selects_ft_on_a_clean_win_and_writes_the_report(
    env: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    js = env["tmp"] / "pooled.json"
    assert g4p.main(argv(env, "--json-out", str(js))) == 0
    assert capsys.readouterr().out.strip().endswith("FINAL SYSTEM: FT+rules")
    text = (env["tmp"] / "pooled.md").read_text(encoding="utf-8")
    for section in (
        "## Provenance", "## Pre-registered rule", "## Clause table", "## Over-nulls and false",
        "## Headline: FT + rules vs ZS + rules", "## Secondary: raw vs raw", "## Slices",
        "## Per-fold OVERALL", "## Rules per fold and arm", "## R3-off sensitivity", "## Limits",
    ):  # fmt: skip
        assert section in text
    assert "**FINAL SYSTEM: FT+rules**" in text and "UNVERIFIED" in text
    assert "invoice (HEADLINE slice)" in text and "493 of 502" in text
    res = json.loads(js.read_text(encoding="utf-8"))
    d = res["rules_pair"]["subsets"]["all"]["paired_delta_oof_minus_zero_shot"]["OVERALL"]
    assert d["delta"] > 0 and d["b"] > d["a"] and d["lo"] > 0  # FT minus ZS
    assert res["n_docs"] == 9 and res["decision"]["final"] == FT_LABEL
    assert sorted(res["fold_deltas"]) == ["0", "1", "2"]
    assert all(v["delta"] > 0 for v in res["fold_deltas"].values())


def test_report_holds_no_gold_value_and_no_doc_id(env: dict[str, Any]) -> None:
    g4p.main(argv(env))
    text = (env["tmp"] / "pooled.md").read_text(encoding="utf-8")
    for needle in ("INV-a1", "Acme Air", "176-12345678", "Pa10", "Sup a1"):
        assert needle not in text
    assert not any(f" {d} " in text or f"`{d}`" in text for d in env["gold"])


def test_the_same_rules_hit_both_arms_in_every_fold(env: dict[str, Any]) -> None:
    js = env["tmp"] / "pooled.json"
    g4p.main(argv(env, "--json-out", str(js)))
    res = json.loads(js.read_text(encoding="utf-8"))
    for k in ("0", "1", "2"):
        for arm in ("zs", "ft"):
            r1 = res["effects"][k][arm]["R1"]
            assert (r1["touched_docs"], r1["fixed"], r1["broken"]) == (1, 1, 0)
            assert res["effects"][k][arm]["R3"]["touched_docs"] == 0
    a = res["rules_pair"]["subsets"]["all"]
    ov = {x: a[x]["over_null"]["header_over_null_by_field"] for x in ("zero_shot", "oof")}
    assert "carrier" not in ov["zero_shot"] and "carrier" not in ov["oof"]  # R1 applied to both
    raw = res["raw_pair"]["subsets"]["all"]
    for x in ("zero_shot", "oof"):  # and the raw view has the carrier over-null in both
        assert raw[x]["over_null"]["header_over_null_by_field"].get("carrier") == 3


def test_an_extra_over_null_in_ft_fails_c3_and_selects_zs(
    env: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    run = env["ft"][1]
    lines = [json.loads(x) for x in (run / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    for t in lines:
        if t["doc_id"] == "b1":
            t["prediction"]["header"]["buyer_name"] = None
    (run / "trace.jsonl").write_text("".join(json.dumps(t) + "\n" for t in lines), encoding="utf-8")
    saved = g4f.rv1.finalize({t["doc_id"]: t["prediction"] for t in lines})
    (run / "predictions.json").write_text(json.dumps(saved), encoding="utf-8")
    js = env["tmp"] / "pooled.json"
    g4p.main(argv(env, "--json-out", str(js)))
    out = capsys.readouterr().out
    assert "FINAL SYSTEM: ZS+rules (v1) [C3 failed" in out
    res = json.loads(js.read_text(encoding="utf-8"))
    assert res["decision"]["clauses"][0]["passed"]  # CI clause still passes: the reason is C3 only


def test_r3_warning_and_r3_off_decision_when_r3_breaks_cells_on_fold_1(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    orig = g4p.rule_counts
    state = {"n": 0}

    def patched(sc: Any, raw: Any, final: Any, gold: Any, summary: Any) -> Any:
        out = orig(sc, raw, final, gold, summary)
        state["n"] += 1
        if state["n"] == 4:  # folds are processed in order, 2 arms each: 4th call = fold 1, FT
            out["R3"].update(touched_docs=2, broken=1)
        return out

    monkeypatch.setattr(g4p, "rule_counts", patched)
    g4p.main(argv(env))
    out = capsys.readouterr().out
    assert "WARNING: R3 breaks cells on the FT arm of fold(s) [1]" in out
    assert "R3-off (informational): FINAL SYSTEM" in out and out.strip().endswith("FT+rules")
    text = (env["tmp"] / "pooled.md").read_text(encoding="utf-8")
    assert "WARNING: R3 fires on the FT arm of fold(s) [1]" in text and "INFORMATIONAL" in text


# ---- fail-closed paths -------------------------------------------------------------------------


def refuses(env: dict[str, Any], match: str) -> None:
    with pytest.raises(SystemExit, match=match):
        g4p.main(argv(env))
    assert not (env["tmp"] / "pooled.md").exists()


def test_refuses_without_oof_section_unverified_adapter_or_incomplete_run(
    env: dict[str, Any],
) -> None:
    run = env["ft"][2]
    edit_manifest(run, lambda m: m["oof"].update(verification={"ok": False}))
    refuses(env, "verification is not ok")
    edit_manifest(run, lambda m: m.pop("oof"))
    refuses(env, "no `oof` section")
    edit_manifest(run, lambda m: m.update(oof=manifest(2)["oof"]))
    (run / "progress.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")
    refuses(env, "not complete")


def test_refuses_fewer_duplicate_or_swapped_folds(env: dict[str, Any]) -> None:
    run0 = env["ft"][0]
    env["ft"] = env["ft"][:2]
    refuses(env, "cover folds")
    env["ft"] = [run0, run0, env["ft"][1]]
    refuses(env, "fold 0 is given twice")


def test_refuses_an_oof_run_holding_another_folds_docs(env: dict[str, Any]) -> None:
    other = make_run(
        env["tmp"], "oof_wrong", env["gold"], ["a1", "a2", "wa"], "ft", oof="ok"
    )  # fold 0's docs under fold 1's manifest
    edit_manifest(other, lambda m: m.update(manifest(1)))
    env["ft"][1] = other
    refuses(env, "not the fold's docs")


@pytest.mark.parametrize(
    ("over", "match"),
    [
        ({"train_code_sha": "d" * 40}, "training code SHA differs"),
        ({"batch_used": 1}, "batch size .*differs"),
        ({"config": {"hash": "zzz"}}, "config_hash differs"),
        ({"adapter_sha256": f"{0:064d}"}, "share one adapter"),
    ],
)
def test_refuses_runs_that_differ_in_code_sha_batch_config_or_share_an_adapter(
    env: dict[str, Any], over: dict[str, Any], match: str
) -> None:
    edit_manifest(env["ft"][2], lambda m: m.update(manifest(2, **over)))
    refuses(env, match)


def test_refuses_a_manifest_without_the_fields_to_compare(env: dict[str, Any]) -> None:
    edit_manifest(env["ft"][1], lambda m: m.pop("config"))
    refuses(env, "fold 1: manifest lacks config.hash")


def _set_hash(run: Path, h: str) -> None:
    edit_manifest(run, lambda m: m.update(config={"name": "cfg", "hash": h}))


def test_all_native_folds_with_a_native_zs_run_pass(env: dict[str, Any]) -> None:
    for run in (env["zs"], *env["ft"]):
        _set_hash(run, "hn")  # miniature stand-in for the native config hash
    assert g4p.main(argv(env)) == 0


@pytest.mark.parametrize("which", ["zs", "ft"])
def test_refuses_native_folds_against_a_1260_zs_run_and_the_reverse(
    env: dict[str, Any], which: str
) -> None:
    runs = env["ft"] if which == "ft" else [env["zs"]]
    for run in runs:
        _set_hash(run, "hn")
    refuses(env, r"mixed-resolution.*ZS run `zs_run`")


def test_refuses_a_zs_run_without_a_config_hash(env: dict[str, Any]) -> None:
    edit_manifest(env["zs"], lambda m: m.pop("config"))
    refuses(env, "mixed-resolution")


def test_refuses_a_zero_shot_run_that_lacks_fold_docs(env: dict[str, Any]) -> None:
    env["zs"] = make_run(env["tmp"], "zs_partial", env["gold"], ["a1", "wa", "b1"], "zs")
    refuses(env, "zero-shot run")


def test_refuses_traces_that_already_carry_rule_records(env: dict[str, Any]) -> None:
    run = env["ft"][0]
    lines = [json.loads(x) for x in (run / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    lines[0]["rules"] = {"changes": []}
    (run / "trace.jsonl").write_text("".join(json.dumps(t) + "\n" for t in lines), encoding="utf-8")
    refuses(env, "not a raw arm")


def test_refuses_when_all_rules_off_does_not_reproduce_the_saved_predictions(
    env: dict[str, Any],
) -> None:
    run = env["ft"][2]
    pred = json.loads((run / "predictions.json").read_text(encoding="utf-8"))
    pred["c1"]["header"]["buyer_name"] = "tampered"
    (run / "predictions.json").write_text(json.dumps(pred), encoding="utf-8")
    refuses(env, "all rules OFF != predictions.json")


def test_refuses_when_r3_learner_overlaps_a_held_out_supplier(env: dict[str, Any]) -> None:
    groups = {**env["groups"], "a1": "inv_g01"}
    (env["root"] / "meta" / "supplier_groups.json").write_text(json.dumps(groups), encoding="utf-8")
    refuses(env, "overlaps")


def test_refuses_missing_ocr_pages_for_a_waybill(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rg, "doc_pages", lambda d, root=None: [])
    refuses(env, "OCR cache lacks waybill pages")


def test_refuses_a_total_other_than_500_docs(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(g4pooled, "EXPECTED_DOCS", 500)
    with pytest.raises(PoolError, match="expected 500"):
        g4p.main(argv(env))


# ---- plumbing check ----------------------------------------------------------------------------


def test_plumbing_check_is_not_a_result_picks_zs_and_writes_nothing(
    env: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    out = env["tmp"] / "never.md"
    rc = g4p.main(
        ["--zs-run-dir", str(env["zs"]), "--plumbing-check", "--out", str(out), "--n-boot", "20"]
    )
    text = capsys.readouterr().out
    assert rc == 0 and "PLUMBING CHECK NOT A RESULT" in text and "!!!!!!!!" in text
    assert not out.exists() and not (env["root"] / "reports" / "g4_pooled.md").exists()
    assert "+0.00 [+0.00, +0.00]" in text
    assert text.strip().splitlines()[-1].startswith("FINAL SYSTEM: ZS+rules (v1) [C1 failed")
    assert "equal to 0" in text  # delta 0 by construction: the bound is exactly 0, which fails


def test_oof_run_dir_is_required_without_plumbing(env: dict[str, Any]) -> None:
    with pytest.raises(SystemExit):
        g4p.main(["--zs-run-dir", str(env["zs"])])
