"""Pooled 3-fold G4 final-system decision: FT + rules vs ZS + rules on all 500 train + dev docs.

Implements spec section 11 item 1 (pre-registered before any 3-fold result): the pure rule lives in
``shipdoc.g4pooled.decide``. This driver feeds it. For each fold k the FT arm is the notebook-05
OOF run of fold k and the ZS arm is the 02 run restricted to fold k's docs; BOTH go through the
production post-processing with the honest per-fold R3 shapes (``scripts/g4_fold.py``'s
``process_arm`` and ``honest_shapes``, imported, not copied). The three folds are validated like
``g4_fold.py`` / ``rule_gate.py`` do per fold, plus: same adapter training code SHA, same inference
setup and batch size across folds, distinct adapters. Then the predictions are pooled to one
500-doc dict per arm and compared once (``shipdoc.oof.compare_models``: paired doc-level bootstrap,
2000 resamples, seed 42, unmodified scorer).

    uv run python scripts/g4_pooled.py --oof-run-dir <f0> --oof-run-dir <f1> --oof-run-dir <f2> \
        --zs-run-dir <02 run folder> [--out reports/g4_pooled.md] [--json-out <path>]

Plumbing check (NOT A RESULT, writes nothing; the 02 run restricted to each fold's docs is BOTH
arms, so every delta is 0 by construction and the rule must pick ZS + rules)::

    uv run python scripts/g4_pooled.py --zs-run-dir <02 run folder> --plumbing-check
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

from shipdoc import eval as ev
from shipdoc import g4, g4pooled
from shipdoc import meta as meta_mod
from shipdoc import oof as oof_mod
from shipdoc.postrules import RuleConfig
from shipdoc.runmeta import read_manifest

ROOT = Path(__file__).resolve().parents[1]
PLUMBING_BANNER = (
    "!" * 78 + "\n"
    "PLUMBING CHECK NOT A RESULT: the 02 run restricted to each fold's docs is used as BOTH arms,\n"
    "so every delta is 0 by construction and nothing below says anything about the fine-tuned\n"
    "model. No report file is written.\n" + "!" * 78
)


def _script(name: str) -> ModuleType:
    """Import ``scripts/<name>.py`` (scripts are not a package); reuse an already loaded copy."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parent / f"{name}.py"
    )
    if spec is None or spec.loader is None:  # pragma: no cover - the file is next to this one
        raise ImportError(name)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses resolve annotations via sys.modules
    spec.loader.exec_module(mod)
    return mod


g4f = _script("g4_fold")
rg = g4f.rg
rv1 = g4f.rv1


# --------------------------------------------------------------------------------------------
# Per-fold inputs
# --------------------------------------------------------------------------------------------


def read_verification(run_dir: Path) -> dict[str, Any] | None:
    """The run's ``oof_verification.json`` (None when absent)."""
    path = run_dir / oof_mod.VERIFIED_NAME
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def validate_oof_runs(
    dirs: Sequence[Path], folds: Mapping[str, Any], doc_fold: Mapping[str, int]
) -> dict[int, tuple[Path, dict[str, Any], list[str]]]:
    """fold -> (run dir, oof section, fold doc ids) for exactly one verified run per fold.

    Per run: the manifest checks of ``rule_gate.check_oof_manifest`` (oof section, fold, adapter
    verification ok, adapter sha and training code sha), complete progress, the exact doc ids of
    its fold and the manifest's ``n_inference_docs``. Across runs: `g4pooled.check_cross_fold`.
    Any failure is a SystemExit.
    """
    runs: dict[int, tuple[Path, dict[str, Any], list[str]]] = {}
    facts = []
    for d in dirs:
        man = read_manifest(d) if d.is_dir() else None
        fold, sec = rg.check_oof_manifest(man, None, False)
        if fold in runs:
            raise SystemExit(f"fold {fold} is given twice ({runs[fold][0].name}, {d.name})")
        rg.require_complete(d, "OOF")
        ids = rg.fold_docs(folds, fold, doc_fold)
        rg.require_oof_inputs(d, ids, True, f"OOF run of fold {fold}")
        if sec.get("n_inference_docs") not in (None, len(ids)):
            raise SystemExit(f"fold {fold}: manifest n_inference_docs differs from the fold size")
        try:
            facts.append(g4pooled.run_facts(man or {}, read_verification(d)))
        except g4pooled.PoolError as e:
            raise SystemExit(f"fold {fold}: {e}") from e
        runs[fold] = (d, sec, ids)
    try:
        g4pooled.check_cross_fold(facts, sorted(set(doc_fold.values())))
    except g4pooled.PoolError as e:
        raise SystemExit(str(e)) from e
    return runs


def rule_counts(
    sc: Any, raw: Mapping[str, Any], final: Mapping[str, Any], gold: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> dict[str, dict[str, int]]:  # fmt: skip
    """Per rule: eligible / touched docs, cells changed, fixed / broken / neutral (fold, arm)."""
    counts = rv1.count_changes(sc, dict(raw), dict(final), dict(gold))
    out: dict[str, dict[str, int]] = {}
    for r in g4.RULES:
        tot = {k: sum(x[k] for x in counts[r].values()) for k in ("fixed", "broken", "neutral")}
        out[r] = {
            "eligible_docs": summary["eligible_docs"][r],
            "touched_docs": summary["touched_docs"][r],
            "cells_changed": summary["changes"][r],
            **tot,
        }
    return out


# --------------------------------------------------------------------------------------------
# The analysis
# --------------------------------------------------------------------------------------------


def analyse(
    fold_arms: Mapping[int, Mapping[str, Any]],
    fold_ids: Mapping[int, Sequence[str]],
    gold: Mapping[str, Any],
    meta_rows: Sequence[Mapping[str, Any]],
    doc_fold: Mapping[str, int],
    shapes: Mapping[int, Any],
    ocr_cache: Path | None,
    n_boot: int,
    plumbing: bool,
) -> dict[str, Any]:
    """All numbers of the report. `fold_arms[k]` is ``{"zs": Arm, "ft": Arm}`` (the same Arm
    object twice in plumbing mode, so the production path is computed once)."""
    sc = ev.load_scorer()
    pooled: dict[str, dict[int, Any]] = {n: {} for n in ("zs_raw", "ft_raw", "zs", "ft")}
    off3: dict[str, dict[int, Any]] = {"zs": {}, "ft": {}}
    effects: dict[int, Any] = {}
    fold_deltas: dict[int, Any] = {}
    for k in sorted(fold_arms):
        gold_k = {d: gold[d] for d in fold_ids[k]}
        done: dict[str, Any] = {}
        eff: dict[str, Any] = {}
        for name in ("zs", "ft"):
            arm = fold_arms[k][name]
            if plumbing and name == "ft":
                raw, final, summary = done["zs"]
                no_r3 = off3["zs"][k]
            else:
                raw, final, summary = g4f.process_arm(arm, shapes[k], ocr_cache)
                no_r3, _ = rv1.production(arm.traces, RuleConfig(r3=False), shapes[k], ocr_cache)
            done[name] = (raw, final, summary)
            pooled[f"{name}_raw"][k], pooled[name][k], off3[name][k] = raw, final, no_r3
            eff[name] = rule_counts(sc, raw, final, gold_k, summary)
        effects[k] = eff
        fold_deltas[k] = ev.paired_bootstrap(
            pooled["zs"][k], pooled["ft"][k], gold_k, n=n_boot, seed=g4.SEED
        )["OVERALL"]
    pool = {
        n: g4pooled.pool_arms(p, fold_ids, doc_fold, g4pooled.EXPECTED_DOCS)
        for n, p in (*pooled.items(), ("zs_off3", off3["zs"]), ("ft_off3", off3["ft"]))
    }
    if set(gold) != set(pool["zs"]):
        raise SystemExit("gold docs differ from the pooled prediction docs")
    meta = list(meta_rows)
    pair = oof_mod.compare_models(pool["zs"], pool["ft"], dict(gold), meta, n_boot, g4.SEED)
    raw_pair = oof_mod.compare_models(
        pool["zs_raw"], pool["ft_raw"], dict(gold), meta, n_boot, g4.SEED
    )
    off_pair = oof_mod.compare_models(
        pool["zs_off3"], pool["ft_off3"], dict(gold), meta, n_boot, g4.SEED
    )
    return {
        "n_docs": len(gold),
        "rules_pair": pair,
        "raw_pair": raw_pair,
        "r3_off_pair": off_pair,
        "decision": g4pooled.decide_from_pair(pair),
        "decision_r3_off": g4pooled.decide_from_pair(off_pair),
        "fold_deltas": fold_deltas,
        "fold_sizes": {k: len(fold_ids[k]) for k in fold_ids},
        "effects": effects,
        "r3_broken_folds": g4pooled.r3_broken_folds(effects),
        "n_boot": n_boot,
        "seed": g4.SEED,
    }


def provenance(
    a: argparse.Namespace,
    runs: Mapping[int, tuple[Path, dict[str, Any], list[str]]],
    zs_dir: Path,
    n_learn: Mapping[int, int],
    plumbing: bool,
) -> dict[str, str]:
    """Run folders, manifest code shas, repo HEAD, command, bootstrap settings (no doc ids)."""
    zs_man = read_manifest(zs_dir) or {}
    out: dict[str, str] = {}
    for k in sorted(runs):
        d, sec, _ids = runs[k]
        man = read_manifest(d) or {}
        out[f"FT fold {k}"] = (
            f"`{d.name}`, manifest code_sha `{man.get('code_sha')}`, adapter sha256 "
            f"`{sec.get('adapter_sha256')}`, batch used {(sec.get('batch') or {}).get('used')}, "
            f"guard fallback to batch 1 {(sec.get('guard') or {}).get('fallback_to_1')}"
        )
    if runs:
        any_sec = next(iter(runs.values()))[1]
        out["adapter training code SHA (same on all folds, checked)"] = (
            f"`{any_sec.get('train_code_sha')}`"
        )
        for k, (_d, sec, _i) in sorted(runs.items()):
            if sec.get("zero_shot_run") not in (None, zs_dir.name):
                out[f"WARNING fold {k}"] = (
                    f"the OOF manifest records zero-shot run `{sec.get('zero_shot_run')}`, but "
                    f"`--zs-run-dir` is `{zs_dir.name}`"
                )
    out["ZS run folder"] = f"`{zs_dir.name}`, manifest code_sha `{zs_man.get('code_sha')}`"
    out["repo state"] = rg.git_state()
    out["command"] = f"`uv run python scripts/g4_pooled.py {' '.join(a.argv)}`"
    out["bootstrap"] = (
        f"{a.n_boot} doc-level resamples, seed {g4.SEED}, unmodified scorer (`shipdoc.eval`), "
        "paired for deltas (FT minus ZS), pooled over all docs"
    )
    out["R3 shapes"] = (
        "per fold, learned from the gold of "
        + ", ".join(f"{n_learn[k]} (fold {k})" for k in sorted(n_learn))
        + " docs outside that fold (supplier-disjoint, checked)"
    )
    if plumbing:
        out["MODE"] = "PLUMBING CHECK NOT A RESULT (02 run used as both arms)"
    return out


def run(a: argparse.Namespace) -> int:
    """Validate, analyse, write (or, for the plumbing check, only print)."""
    plumbing: bool = a.plumbing_check
    zs_dir: Path = a.zs_run_dir
    folds = json.loads((ROOT / "splits" / "folds.json").read_text(encoding="utf-8"))
    doc_fold = rg.load_doc_fold()
    fold_list = sorted(set(doc_fold.values()))
    fold_ids = {k: rg.fold_docs(folds, k, doc_fold) for k in fold_list}
    runs: dict[int, tuple[Path, dict[str, Any], list[str]]] = {}
    if not plumbing:
        runs = validate_oof_runs(a.oof_run_dir, folds, doc_fold)
        # check_cross_fold already pins one config hash across the folds; ZS must match it too
        rg.require_same_resolution(
            {"ZS run": zs_dir, **{f"FT fold {k}": runs[k][0] for k in sorted(runs)}}
        )
    rg.require_complete(zs_dir, "zero-shot")
    all_ids = sorted(d for k in fold_list for d in fold_ids[k])
    gold = g4f.load_gold(all_ids)
    meta_rows = g4f.load_meta(all_ids)
    labels = meta_mod.load_labels("train") + meta_mod.load_labels("dev")
    groups = json.loads((ROOT / "meta" / "supplier_groups.json").read_text(encoding="utf-8"))
    try:
        per_fold_shapes = rg.fold_shapes(labels, doc_fold, groups)
    except ValueError as e:  # supplier overlap: fail closed
        raise SystemExit(str(e)) from e
    shapes = {k: per_fold_shapes[k][0] for k in fold_list}
    n_learn = {k: len(per_fold_shapes[k][1]) for k in fold_list}
    fold_arms: dict[int, dict[str, Any]] = {}
    for k in fold_list:
        ids = fold_ids[k]
        gold_k = {d: gold[d] for d in ids}
        rg.require_oof_inputs(zs_dir, ids, False, "zero-shot run")
        dirs = [zs_dir] if plumbing else [zs_dir, runs[k][0]]
        for d in dirs:  # R2 eligibility must not shrink: every waybill page is cached
            rg.require_ocr(d, SimpleNamespace(gold=gold_k), a.ocr_cache)
        zs = g4f.read_arm("ZS arm", zs_dir, ids)
        ft = zs if plumbing else g4f.read_arm(f"FT arm fold {k}", runs[k][0], ids)
        fold_arms[k] = {"zs": zs, "ft": ft}
    res = analyse(
        fold_arms, fold_ids, gold, meta_rows, doc_fold, shapes, a.ocr_cache, a.n_boot, plumbing
    )
    res["provenance"] = provenance(a, runs, zs_dir, n_learn, plumbing)
    text = g4pooled.render_report(res)
    if plumbing:
        print(PLUMBING_BANNER + "\n" + text)
        print(res["decision"].line)
        return 0
    out = a.out if a.out is not None else ROOT / "reports" / "g4_pooled.md"
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")
    if a.json_out is not None:
        a.json_out.write_text(json.dumps(_jsonable(res), indent=1, default=str), encoding="utf-8")
        print(f"wrote {a.json_out}")
    if res["r3_broken_folds"]:
        print(
            f"WARNING: R3 breaks cells on the FT arm of fold(s) {res['r3_broken_folds']}; "
            f"R3-off (informational): {res['decision_r3_off'].line}"
        )
    print(res["decision"].line)
    return 0


def _jsonable(res: Mapping[str, Any]) -> dict[str, Any]:
    """`res` with the Decision objects flattened (everything else is already plain data)."""
    out = dict(res)
    for key in ("decision", "decision_r3_off"):
        d = res[key]
        out[key] = {
            "final": d.final,
            "line": d.line,
            "clauses": [{"key": c.key, "passed": c.passed, "text": c.text} for c in d.clauses],
        }
    return out


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--oof-run-dir", type=Path, action="append", default=[],
        help="an OOF run folder (give it once per fold, three times)",
    )  # fmt: skip
    ap.add_argument("--zs-run-dir", type=Path, required=True, help="the 02 zero-shot run folder")
    ap.add_argument("--out", type=Path, default=None, help="default reports/g4_pooled.md")
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--plumbing-check", action="store_true", help="NOT A RESULT; writes nothing")
    ap.add_argument("--ocr-cache", type=Path, default=None)
    ap.add_argument("--n-boot", type=int, default=g4.N_BOOT)
    a = ap.parse_args(raw)
    a.argv = raw
    if not a.plumbing_check and not a.oof_run_dir:
        ap.error("--oof-run-dir (three times) is required, or --plumbing-check")
    if a.plumbing_check and a.oof_run_dir:
        print("note: --plumbing-check uses --zs-run-dir as both arms; --oof-run-dir is ignored")
    if a.plumbing_check and (a.out is not None or a.json_out is not None):
        print("note: --plumbing-check writes no report; --out / --json-out are ignored")
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
