"""scripts/gate_eval.py on the synthetic miniature world of test_g4pooled (no model, no document).

Covers the fail-closed checks (missing / repeated fold, duplicate trace, mixed resolution, fewer
than three folds without --exploratory), the exploratory stamp and the absence of a decision in an
exploratory run, the plumbing check, the fit / apply disjointness wiring, and a full 3-fold run on
the miniature world (logprobs are not required there: `prepare_arm` is wrapped to relax that).
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from test_g4_fold import make_run  # noqa: F401  (also used by the env fixture)
from test_g4pooled import argv as pooled_argv
from test_g4pooled import edit_manifest, manifest
from test_g4pooled import env as pooled_env  # noqa: F401  (a fixture: pytest finds it by name)

from shipdoc import gate
from shipdoc.g4pooled import FT_LABEL, ZS_LABEL

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import gate_eval as ge  # noqa: E402

STAMP0 = (
    "EXPLORATORY, NOT A DECISION; fold 0 was already seen when G was pre-registered; "
    "within-fold cross-fit; fold 1/2 untouched"
)

# ---- pure checks ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("folds", "flag", "expected"),
    [
        ([0, 1, 2], False, False),  # the decision run
        ([2, 0, 1], False, False),  # order is irrelevant
        ([0, 1, 2], True, True),  # allowed, but stamped exploratory
        ([0], True, True),
        ([1, 2], True, True),
    ],
)
def test_check_folds_accepts(folds: list[int], flag: bool, expected: bool) -> None:
    assert ge.check_folds(folds, flag) is expected


@pytest.mark.parametrize(
    ("folds", "flag", "match"),
    [
        ([0], False, "without --exploratory"),  # fewer than 3 folds REQUIRES the flag
        ([0, 1], False, "without --exploratory"),
        ([], True, "distinct folds"),
        ([0, 0, 1], True, "distinct folds"),
        ([0, 1, 3], True, "distinct folds"),
        ([-1], True, "distinct folds"),
    ],
)
def test_check_folds_refuses(folds: list[int], flag: bool, match: str) -> None:
    with pytest.raises(ge.GateEvalError, match=match):
        ge.check_folds(folds, flag)


def test_exploratory_stamp_texts() -> None:
    assert ge.exploratory_stamp([0]) == STAMP0
    s = ge.exploratory_stamp([1, 2])
    assert s.startswith("EXPLORATORY, NOT A DECISION") and "fold 0 untouched" in s
    assert "fold 0 was already seen" in ge.exploratory_stamp([0, 1])


def ids_by_fold(doc_fold: dict[str, int]) -> dict[int, list[str]]:
    return {
        k: sorted(d for d, f in doc_fold.items() if f == k) for k in sorted(set(doc_fold.values()))
    }


def test_pool_folds_partial_and_full() -> None:
    from test_g4_fold import world

    w = world()
    ids = ids_by_fold(w["doc_fold"])
    per = {k: {d: {"doc": d} for d in v} for k, v in ids.items()}
    assert sorted(ge.pool_folds({0: per[0]}, {0: ids[0]}, w["doc_fold"], False)) == ids[0]
    full = ge.pool_folds(per, ids, w["doc_fold"], False)
    assert sorted(full) == sorted(w["doc_fold"])
    with pytest.raises(Exception, match="expected 500"):  # the decision run pools exactly 500 docs
        ge.pool_folds(per, ids, w["doc_fold"], True)


def test_pool_folds_refuses_wrong_membership_and_duplicates() -> None:
    from test_g4_fold import world

    w = world()
    ids = ids_by_fold(w["doc_fold"])
    per = {k: {d: {"doc": d} for d in v} for k, v in ids.items()}
    missing = copy.deepcopy(per)
    missing[1].pop("b1")
    with pytest.raises(ge.GateEvalError, match="fold 1"):
        ge.pool_folds(missing, ids, w["doc_fold"], False)
    smuggled = copy.deepcopy(per)
    smuggled[1]["a1"] = {"doc": "a1"}  # a fold-0 doc inside fold 1
    with pytest.raises(ge.GateEvalError, match="fold 1"):
        ge.pool_folds(smuggled, ids, w["doc_fold"], False)
    dup = copy.deepcopy(ids)
    dup[0] = [*ids[0], ids[0][0]]  # a duplicated doc id in a fold's doc list
    with pytest.raises(ge.GateEvalError, match="fold 0"):
        ge.pool_folds(per, dup, w["doc_fold"], False)
    with pytest.raises(ge.GateEvalError, match="disagree"):
        ge.pool_folds({0: per[0]}, ids, w["doc_fold"], False)


# ---- fit / apply wiring -------------------------------------------------------------------------


def test_cross_fit_asserts_supplier_disjointness_before_fitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = np.array(["a", "b", "c", "d"], dtype=object)
    arm = SimpleNamespace(tv=SimpleNamespace(table=SimpleNamespace(doc_ids=ids)), sigs={})
    groups = {"a": "g1", "b": "g1", "c": "g2", "d": "g2"}
    # a fold map that splits supplier g1 across two folds: a leak the assertion must catch
    monkeypatch.setattr(
        ge.c3, "inner_group_folds", lambda docs, g: ({"a": 0, "b": 1, "c": 0, "d": 1}, 2, 2)
    )
    with pytest.raises(gate.GateError, match="supplier group"):
        ge.cross_fit_gate_probs(arm, arm, np.zeros((4, 1)), np.zeros((4, 1)), [0], {}, groups)


def test_cross_fit_refuses_test_ids() -> None:
    ids = np.array(["test_0001"], dtype=object)
    arm = SimpleNamespace(tv=SimpleNamespace(table=SimpleNamespace(doc_ids=ids)), sigs={})
    with pytest.raises(Exception, match="test split"):
        ge.cross_fit_gate_probs(arm, arm, np.zeros((1, 1)), np.zeros((1, 1)), [0], {}, {})


# ---- fail-closed paths of the driver ------------------------------------------------------------


@pytest.fixture
def genv(pooled_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:  # noqa: F811
    monkeypatch.setattr(ge, "ROOT", pooled_env["root"])
    gold = pooled_env["gold"]
    monkeypatch.setattr(
        ge.v2s.v1cal,
        "load_corpus",
        lambda ids: (
            {d: gold[d] for d in ids},
            {d: {"supplier_group": pooled_env["groups"][d]} for d in ids},
        ),
    )
    real = ge.cal3.prepare_arm
    # the miniature traces carry no logprobs: relax only that requirement (kept by one test below)
    monkeypatch.setattr(
        ge.cal3, "prepare_arm", lambda *a, **k: real(*a[:-1], require_logprobs=False)
    )
    return {**pooled_env, "real_prepare": real}


def args(w: dict[str, Any], *extra: str, out: str = "gate.md") -> list[str]:
    return [*pooled_argv(w)[:-4], "--out", str(w["tmp"] / out), "--n-boot", "20", *extra]


def refuses(w: dict[str, Any], argv: list[str], match: str) -> None:
    with pytest.raises(SystemExit, match=match):
        ge.main(argv)
    assert not (w["tmp"] / "gate.md").exists()  # nothing is written on a refusal


def test_refuses_fewer_than_three_folds_without_exploratory(genv: dict[str, Any]) -> None:
    a = ["--folds", "0", "--oof-run-dir", str(genv["ft"][0]), "--zs-run-dir", str(genv["zs"])]
    refuses(genv, [*a, "--out", str(genv["tmp"] / "gate.md")], "without --exploratory")


def test_refuses_a_run_count_that_differs_from_the_folds(genv: dict[str, Any]) -> None:
    a = args(genv)
    del a[:2]  # drop the fold-0 run: two runs for three folds
    with pytest.raises(SystemExit) as e:
        ge.main(a)
    assert e.value.code == 2


def test_refuses_a_repeated_fold(genv: dict[str, Any]) -> None:
    run0, run1 = genv["ft"][0], genv["ft"][1]
    genv["ft"] = [run0, run0, run1]
    refuses(genv, args(genv), "fold 0 is given twice")


def test_validate_runs_refuses_a_missing_fold_and_an_unrequested_fold(genv: dict[str, Any]) -> None:
    ft, folds, doc_fold = genv["ft"], genv["folds"], genv["doc_fold"]
    with pytest.raises(SystemExit, match="cover folds"):  # the decision run needs all three
        ge.validate_runs(ft[:2], folds, doc_fold, [0, 1, 2])
    with pytest.raises(SystemExit, match="cover folds"):  # a fold that was not asked for
        ge.validate_runs(ft[:2], folds, doc_fold, [0])
    assert sorted(ge.validate_runs(ft[:2], folds, doc_fold, [0, 1])) == [0, 1]
    assert sorted(ge.validate_runs(ft, folds, doc_fold, [0, 1, 2])) == [0, 1, 2]


def test_refuses_a_duplicate_doc_in_a_run(genv: dict[str, Any]) -> None:
    run = genv["ft"][0]
    lines = (run / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    (run / "trace.jsonl").write_text("\n".join([*lines, lines[0]]) + "\n", encoding="utf-8")
    refuses(genv, args(genv), "exactly one trace per fold doc")


def test_refuses_a_mixed_resolution_between_ft_and_zs(genv: dict[str, Any]) -> None:
    edit_manifest(genv["zs"], lambda m: m.update(config={"name": "cfg", "hash": "hn"}))
    refuses(genv, args(genv), "mixed-resolution")


def test_refuses_runs_with_different_resolution_across_folds(genv: dict[str, Any]) -> None:
    edit_manifest(genv["ft"][2], lambda m: m.update(manifest(2, config={"hash": "hn"})))
    refuses(genv, args(genv), "config_hash differs")


def test_refuses_an_unverified_adapter(genv: dict[str, Any]) -> None:
    edit_manifest(genv["ft"][1], lambda m: m["oof"].update(verification={"ok": False}))
    refuses(genv, args(genv), "verification is not ok")


def test_requires_logprobs_when_not_relaxed(
    genv: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ge.cal3, "prepare_arm", genv["real_prepare"])  # drop the relaxation
    with pytest.raises(SystemExit, match=r"FAIL \(closed\): only 0/\d+ documents carry field_logp"):
        ge.main(args(genv))
    assert not (genv["tmp"] / "gate.md").exists()


# ---- the driver on the miniature world ----------------------------------------------------------


def run_text(w: dict[str, Any], *extra: str, out: str = "gate.md") -> tuple[str, dict[str, Any]]:
    js = w["tmp"] / "gate.json"
    assert ge.main(args(w, "--json-out", str(js), *extra, out=out)) == 0
    res = json.loads(js.read_text(encoding="utf-8"))
    return (w["tmp"] / out).read_text(encoding="utf-8"), res


def test_three_fold_run_decides_with_the_primary_rule_first(
    genv: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    text, res = run_text(genv)
    out = capsys.readouterr().out
    assert not text.startswith("# EXPLORATORY") and "pooled 3-fold decision" in text
    assert "## Decision" in text and "Primary rule (spec section 11 item 1, decided first)" in text
    assert res["exploratory"] is False and res["primary_decision"]["final"] in (FT_LABEL, ZS_LABEL)
    # the miniature world: FT is right where ZS is wrong, so FT wins the primary rule; G cannot
    # beat the FT winner (it can at best equal it), so G1 (strict CI lower bound > 0) must fail
    assert res["primary_decision"]["final"] == FT_LABEL
    assert (
        res["g_decision"]["clauses"][0]["key"] == "G1"
        and not res["g_decision"]["clauses"][0]["passed"]
    )
    assert "SECONDARY SYSTEM G DOES NOT REPLACE FT+rules" in out
    sections = (
        "## Cross-fit",
        "## Design decisions the spec does not define",
        "## What G took from which arm",
        "## ORACLE per-field selection",
        "## Limits",
    )
    assert all(s in text for s in sections)
    assert "UNVERIFIED" in text
    assert res["cross_fit"]["mode"].startswith("supplier-fold cross-fit")


def test_exploratory_run_is_stamped_and_prints_no_decision(
    genv: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    js = genv["tmp"] / "gate.json"
    a = ["--folds", "0", "--exploratory", "--oof-run-dir", str(genv["ft"][0]), "--zs-run-dir",
         str(genv["zs"]), "--out", str(genv["tmp"] / "gate.md"), "--json-out", str(js),
         "--n-boot", "20"]  # fmt: skip
    assert ge.main(a) == 0
    text = (genv["tmp"] / "gate.md").read_text(encoding="utf-8")
    assert text.splitlines()[0] == f"# {STAMP0}"  # the stamp is the file header
    assert "The replacement rule was NOT evaluated" in text
    assert "SECONDARY SYSTEM" not in text and "SECONDARY SYSTEM" not in capsys.readouterr().out
    res = json.loads(js.read_text(encoding="utf-8"))
    assert "g_decision" not in res and "primary_decision" not in res and res["exploratory"] is True
    assert res["cross_fit"]["mode"].startswith("within-fold supplier-grouped cross-fit")


def test_three_folds_with_the_flag_are_stamped_and_not_decided(genv: dict[str, Any]) -> None:
    text, res = run_text(genv, "--exploratory")
    assert text.startswith("# EXPLORATORY, NOT A DECISION") and "g_decision" not in res


def test_report_holds_no_gold_value_and_no_doc_id(genv: dict[str, Any]) -> None:
    text, _ = run_text(genv)
    for needle in ("INV-a1", "Acme Air", "176-12345678", "Pa10", "Sup a1"):
        assert needle not in text
    assert not any(f" {d} " in text or f"`{d}`" in text for d in genv["gold"])


def test_plumbing_check_writes_nothing_and_g_equals_zs(
    genv: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    out = genv["tmp"] / "never.md"
    rc = ge.main(["--plumbing-check", "--zs-run-dir", str(genv["zs"]), "--out", str(out),
                  "--n-boot", "20"])  # fmt: skip
    text = capsys.readouterr().out
    assert rc == 0 and "!!!!!!!!" in text and "PLUMBING CHECK NOT A RESULT" in text
    assert "PLUMBING OK: G equals ZS + rules" in text and "+0.00 [+0.00, +0.00]" in text
    assert not out.exists() and not (genv["root"] / "reports" / "gate_exploratory.md").exists()


def test_plumbing_check_needs_no_oof_run_but_a_real_run_does(genv: dict[str, Any]) -> None:
    with pytest.raises(SystemExit):
        ge.main(["--zs-run-dir", str(genv["zs"])])  # neither --oof-run-dir nor --plumbing-check
