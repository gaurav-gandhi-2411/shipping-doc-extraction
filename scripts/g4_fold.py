"""Fold-K G4 analysis (CPU only, no model): fine-tuned + rules vs zero-shot + rules.

Prepares the analysis of notebook 05's fold-K OOF output. BOTH arms are post-processed with the
PRODUCTION rules (R1 / R2 / R3, ``shipdoc.postrules.postprocess_traces`` on the TRACES, then
coerce + repair, exactly ``scripts/replay_v1_check.py::production``), with the same honest R3
shapes (learned from the gold of the other folds only, supplier-disjointness checked) and the OCR
cache for R2. For each arm the same path with every rule OFF is asserted to reproduce the arm's
``predictions.json`` on the fold's docs (fails closed otherwise). Then:

* headline: FT + rules vs ZS + rules, five metrics, paired doc-level bootstrap (2000, seed 42,
  unmodified scorer); secondary: raw vs raw (``shipdoc.oof.compare_models``, reused);
* slices scanned / digital and invoice / waybill (invoice-only is the headline slice);
* over-nulls per field for both arms; R1 / R2 / R3 touched / fixed / broken per arm; the
  row-error CAUSE table of ``scripts/row_error_diagnosis.py`` rerun on every arm's outputs;
* two verdict lines: the interim G4 verdict defined for 05 (notebooks/README) and the spec's G4
  clause for reference, both labelled interim / one fold. Aggregates only; all UNVERIFIED.

Validation reuses ``scripts/rule_gate.py``'s OOF-mode checks (imported, not copied): manifest
``oof`` section and adapter verification, complete progress, exact doc ids of fold K, traces,
OCR pages of every waybill; plus gold for every doc and traces free of rule records (a raw arm
must be raw). The ZS run may hold more docs (500) and is restricted to the fold.

    uv run python scripts/g4_fold.py --oof-run-dir <oof_fold0_sha7 folder> \
        --zs-run-dir <02 run folder> [--fold 0] [--out reports/g4_fold0.md] [--json-out <path>]

Plumbing check (NOT A RESULT, writes no report; the 02 run restricted to fold K's docs is BOTH
arms, so every delta is 0 by construction)::

    uv run python scripts/g4_fold.py --zs-run-dir <02 run folder> --plumbing-check [--fold 0]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

from shipdoc import eval as ev
from shipdoc import g4, locate, paths
from shipdoc import meta as meta_mod
from shipdoc import oof as oof_mod
from shipdoc.postrules import RuleConfig
from shipdoc.replay import read_trace
from shipdoc.runmeta import read_manifest

ROOT = Path(__file__).resolve().parents[1]
PLUMBING_BANNER = (
    "!" * 78 + "\n"
    "PLUMBING CHECK NOT A RESULT: the 02 run restricted to the fold's docs is used as BOTH arms,\n"
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


rg = _script("rule_gate")
rv1 = _script("replay_v1_check")
rde = _script("row_error_diagnosis")


# --------------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------------


@dataclass
class Arm:
    """One arm's run folder restricted to the fold's docs."""

    label: str
    run_dir: Path
    traces: list[dict[str, Any]]  # trace.jsonl lines of the fold's docs (doc_id order)
    saved: dict[str, Any]  # predictions.json restricted to the fold's docs


def read_arm(label: str, run_dir: Path, ids: Sequence[str]) -> Arm:
    """Traces and saved predictions of `run_dir` for exactly `ids`.

    Fails closed (SystemExit) if a trace already carries a ``rules`` record: that arm was
    post-processed at inference time, so neither its raw view nor the rules-off assertion hold.
    """
    want = set(ids)
    traces = sorted(
        (t for t in read_trace(run_dir / "trace.jsonl") if t["doc_id"] in want),
        key=lambda t: t["doc_id"],
    )
    if sorted(t["doc_id"] for t in traces) != sorted(ids):
        raise SystemExit(f"{label}: trace.jsonl does not hold exactly one trace per fold doc")
    if any("rules" in t for t in traces):
        raise SystemExit(f"{label}: traces carry post-processing records, not a raw arm")
    pred = json.loads((run_dir / "predictions.json").read_text(encoding="utf-8"))
    return Arm(label, run_dir, traces, {d: pred[d] for d in ids})


def load_gold(ids: Sequence[str]) -> dict[str, Any]:
    """Gold of `ids` from the train / dev label folders; SystemExit if any doc has none."""
    gold: dict[str, Any] = {}
    for split in sorted({d.split("_", 1)[0] for d in ids}):
        gold |= ev.load_gold(paths.data_dir() / split / "labels")
    missing = [d for d in ids if d not in gold]
    if missing:
        raise SystemExit(f"{len(missing)} fold docs have no gold label, e.g. {missing[:3]}")
    return {d: gold[d] for d in ids}


def load_meta(ids: Sequence[str]) -> list[dict[str, Any]]:
    """meta/{train,dev}.json entries of `ids` (scanned / repeated_parts / ...); all must exist."""
    rows = {}
    for split in ("train", "dev"):
        for m in json.loads((ROOT / "meta" / f"{split}.json").read_text(encoding="utf-8")):
            rows[m["doc_id"]] = m
    missing = [d for d in ids if d not in rows]
    if missing:
        raise SystemExit(f"{len(missing)} fold docs have no meta entry, e.g. {missing[:3]}")
    return [rows[d] for d in ids]


def honest_shapes(
    fold: int,
    labels: Sequence[Mapping[str, Any]],
    doc_fold: Mapping[str, int],
    groups: Mapping[str, str],
) -> tuple[Any, int]:
    """(R3 shapes learned from the gold of the other folds only, number of learner docs)."""
    try:
        per_fold = rg.fold_shapes(labels, doc_fold, groups)
    except ValueError as e:  # supplier overlap: fail closed
        raise SystemExit(str(e)) from e
    shapes, learn_ids = per_fold[fold]
    return shapes, len(learn_ids)


# --------------------------------------------------------------------------------------------
# Production path per arm
# --------------------------------------------------------------------------------------------


def process_arm(
    arm: Arm, shapes: Any, ocr_cache: Path | None
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """(raw predictions, rules-on predictions, rules-on summary) of the production path.

    "raw" is the production path with every rule OFF; it must equal the arm's saved
    ``predictions.json`` on the fold's docs, else SystemExit (the traces and the saved
    predictions would disagree, so no number could be trusted).
    """
    raw, _ = rv1.production(arm.traces, RuleConfig.all_off(), None, ocr_cache)
    if raw != arm.saved:
        raise SystemExit(f"{arm.label}: production path with all rules OFF != predictions.json")
    final, summary = rv1.production(arm.traces, RuleConfig(), shapes, ocr_cache)
    return raw, final, summary


# --------------------------------------------------------------------------------------------
# Cause table (row_error_diagnosis taxonomy, imported not edited)
# --------------------------------------------------------------------------------------------


def build_indexes(ids: Sequence[str], ocr_cache: Path | None) -> dict[str, Any]:
    """Locator index per doc from the OCR cache; SystemExit when any doc has no cached OCR."""
    ocr = ev.load_ocr_pages(list(ids), ocr_cache)
    missing = [d for d in ids if d not in ocr]
    if missing:
        raise SystemExit(f"no cached OCR for {len(missing)} fold docs, e.g. {missing[:3]}")
    return {d: locate.build_index(ocr[d]) for d in ids}


def cause_summary(
    pred: Mapping[str, Any],
    gold: Mapping[str, Any],
    traces: Mapping[str, Any],
    indexes: Mapping[str, Any],
    meta: Mapping[str, Mapping[str, Any]],
    groups: Mapping[str, str],
    truncated: set[str],
    sc: Any,
) -> dict[str, Any]:
    """Counts of ``row_error_diagnosis``' classification for one prediction set (no values)."""
    units = []
    n_inv = n_trunc = 0
    for d, g in gold.items():
        p = pred.get(d)
        if g["doc_type"] != "invoice" or not isinstance(p, dict) or p.get("doc_type") != "invoice":
            continue
        n_inv += 1
        n_trunc += d in truncated
        ctx = {
            "group": groups[d],
            "scanned": bool(meta[d]["scanned"]),
            "repeated": bool(meta[d]["repeated_parts"]),
        }
        units += rde.diagnose_doc(d, p, g, traces[d], indexes[d], ctx, sc, d in truncated)
    slots = Counter(s["kind"] for s in rde.slot_rows(dict(pred), dict(gold), sc))
    wb = rde.waybill_over_nulls(
        dict(pred),
        dict(gold),
        dict(traces),
        dict(indexes),
        {d: bool(meta[d]["scanned"]) for d in gold},
        sc,
    )
    fields = {
        f: {
            "cells": 0,
            "null_emitted": 0,
            "key_missing": 0,
            "alt_slot": 0,
            "pattern_unique_correct": 0,
        }
        for f in rde.WB_FIELDS
    }
    for c in wb["cells"]:
        x = fields[c["field"]]
        x["cells"] += 1
        x["null_emitted"] += c["raw_state"] == "null_emitted"
        x["key_missing"] += c["raw_state"] == "key_missing"
        x["alt_slot"] += bool(c["alt_slots"])
        x["pattern_unique_correct"] += bool(c["pattern_unique_correct"])
    return {
        "n_invoice_docs": n_inv,
        "n_truncated_invoice_docs": n_trunc,
        "side": {s: sum(u.side == s for u in units) for s in ("pair", "gold", "pred")},
        "causes": dict(Counter(u.cause for u in units)),
        "slots": dict(slots),
        "n_waybill_docs": sum(g["doc_type"] == "waybill" for g in gold.values()),
        "waybill": fields,
    }


# --------------------------------------------------------------------------------------------
# The analysis
# --------------------------------------------------------------------------------------------


def analyse(
    zs: Arm,
    ft: Arm,
    gold: Mapping[str, Any],
    meta_rows: Sequence[Mapping[str, Any]],
    shapes: Any,
    groups: Mapping[str, str],
    ocr_cache: Path | None,
    n_boot: int,
) -> dict[str, Any]:
    """All numbers of the report as one dict (aggregates only)."""
    sc = ev.load_scorer()
    meta = {m["doc_id"]: m for m in meta_rows}
    gold = dict(gold)
    zs_raw, zs_final, zs_sum = process_arm(zs, shapes, ocr_cache)
    ft_raw, ft_final, ft_sum = process_arm(ft, shapes, ocr_cache)
    rules_pair = oof_mod.compare_models(
        zs_final, ft_final, gold, list(meta_rows), n_boot=n_boot, seed=g4.SEED
    )
    raw_pair = oof_mod.compare_models(
        zs_raw, ft_raw, gold, list(meta_rows), n_boot=n_boot, seed=g4.SEED
    )
    effect = {
        "zs": g4.rules_effect(sc, zs_raw, zs_final, gold, zs_sum, rv1.count_changes, n_boot),
        "ft": g4.rules_effect(sc, ft_raw, ft_final, gold, ft_sum, rv1.count_changes, n_boot),
    }
    indexes = build_indexes(list(gold), ocr_cache)
    tr = {"zs": {t["doc_id"]: t for t in zs.traces}, "ft": {t["doc_id"]: t for t in ft.traces}}
    trunc = {
        k: {d for d, t in tr[k].items() if any(not p["json_valid"] for p in t["pages"])} for k in tr
    }
    causes = {
        name: cause_summary(pred, gold, tr[arm], indexes, meta, groups, trunc[arm], sc)
        for name, arm, pred in (
            ("zs_rules", "zs", zs_final),
            ("ft_rules", "ft", ft_final),
            ("zs_raw", "zs", zs_raw),
            ("ft_raw", "ft", ft_raw),
        )
    }
    all_a, raw_a = rules_pair["subsets"]["all"], raw_pair["subsets"]["all"]
    blocks = {
        "zs_rules": all_a["zero_shot"],
        "ft_rules": all_a["oof"],
        "zs_raw": raw_a["zero_shot"],
        "ft_raw": raw_a["oof"],
    }
    return {
        "n_docs": len(gold),
        "rules_pair": rules_pair,
        "raw_pair": raw_pair,
        "blocks": blocks,
        "rules_effect": effect,
        "causes": causes,
        "cause_names": [
            *rde.ALL_CAUSES,
            *sorted({c for s in causes.values() for c in s["causes"]} - set(rde.ALL_CAUSES)),
        ],
        "verdicts": g4.verdicts(rules_pair, raw_pair),
    }


LIMITS = (
    "One fold: the interim verdict and the spec clause are NOT the G4 decision.",
    "R3 shapes are learned from the other folds' gold; R3 may not fire on a fold, and is then "
    "not judged (it is judged only on folds where it fires).",
    "Row-error causes need the row page of each predicted row, replayed from the trace; where the "
    "replay does not give the final row count the pages are unknown (as in row_error_diagnosis).",
    "Waybill over-null facts use the model's raw page-1 output; R2 needs OCR at inference time.",
    "Over-null counts on rows use the scorer's pairing: a missing row is a missing row, not "
    "an over-null cell.",
    "Slices of one fold are small (see the docs column); wide CIs are expected.",
)


def provenance(
    a: argparse.Namespace,
    fold: int,
    ft: Arm,
    zs: Arm,
    sec: Mapping[str, Any],
    n_learn: int,
    n_boot: int,
    plumbing: bool,
) -> dict[str, str]:
    """Run folders, manifest code shas, repo HEAD, command, bootstrap settings (no doc ids)."""
    ft_man = read_manifest(ft.run_dir) or {}
    zs_man = read_manifest(zs.run_dir) or {}
    argv = " ".join(a.argv)
    out = {
        "FT run folder": ft.run_dir.name,
        "FT manifest code_sha": str(ft_man.get("code_sha")),
        "FT oof section": (
            "none (plumbing check)"
            if not sec
            else f"fold {sec.get('fold')}, adapter sha256 `{sec.get('adapter_sha256')}`, training "
            f"code SHA `{sec.get('train_code_sha')}`, batch used "
            f"{(sec.get('batch') or {}).get('used')}, guard fallback to batch 1 "
            f"{(sec.get('guard') or {}).get('fallback_to_1')}"
        ),
        "ZS run folder": zs.run_dir.name,
        "ZS manifest code_sha": str(zs_man.get("code_sha")),
        "repo state": rg.git_state(),
        "command": f"`uv run python scripts/g4_fold.py {argv}`",
        "bootstrap": f"{n_boot} doc-level resamples, seed {g4.SEED}, unmodified scorer "
        "(`shipdoc.eval`), paired for deltas (FT minus ZS)",
        "R3 shapes": f"learned from the gold of {n_learn} docs outside fold {fold} "
        "(supplier-disjoint, checked)",
    }
    if sec and sec.get("zero_shot_run") not in (None, zs.run_dir.name):
        out["WARNING"] = (
            f"the OOF manifest records zero-shot run `{sec.get('zero_shot_run')}`, but "
            f"`--zs-run-dir` is `{zs.run_dir.name}`"
        )
    if plumbing:
        out["MODE"] = "PLUMBING CHECK NOT A RESULT (02 run used as both arms)"
    return out


def run(a: argparse.Namespace) -> int:
    """Validate, analyse, write (or, for the plumbing check, only print)."""
    plumbing: bool = a.plumbing_check
    zs_dir: Path = a.zs_run_dir
    ft_dir: Path = zs_dir if plumbing else a.oof_run_dir
    man = read_manifest(ft_dir) if ft_dir.is_dir() else None
    fold, sec = rg.check_oof_manifest(man, a.fold, plumbing)
    rg.require_complete(ft_dir, "zero-shot" if plumbing else "OOF")
    if not plumbing:  # FT and ZS must be one input resolution, else the delta is confounded
        rg.require_same_resolution({"FT run": ft_dir, "ZS run": zs_dir})
    folds = json.loads((ROOT / "splits" / "folds.json").read_text(encoding="utf-8"))
    doc_fold = rg.load_doc_fold()
    ids = rg.fold_docs(folds, fold, doc_fold)
    rg.require_oof_inputs(ft_dir, ids, not plumbing, "zero-shot run" if plumbing else "OOF run")
    if sec and sec.get("n_inference_docs") not in (None, len(ids)):
        raise SystemExit("manifest n_inference_docs differs from the fold's doc count")
    if not plumbing:
        rg.require_complete(zs_dir, "zero-shot")
        rg.require_oof_inputs(zs_dir, ids, False, "zero-shot run")
    gold = load_gold(ids)
    meta_rows = load_meta(ids)
    for d in (ft_dir, zs_dir):  # R2 eligibility must not shrink: every waybill page is cached
        rg.require_ocr(d, SimpleNamespace(gold=gold), a.ocr_cache)
    ft = read_arm("FT arm", ft_dir, ids)
    zs = read_arm("ZS arm", zs_dir, ids)
    labels = meta_mod.load_labels("train") + meta_mod.load_labels("dev")
    groups = json.loads((ROOT / "meta" / "supplier_groups.json").read_text(encoding="utf-8"))
    shapes, n_learn = honest_shapes(fold, labels, doc_fold, groups)
    res = analyse(zs, ft, gold, meta_rows, shapes, groups, a.ocr_cache, a.n_boot)
    res |= {
        "fold": fold,
        "provenance": provenance(a, fold, ft, zs, sec, n_learn, a.n_boot, plumbing),
        "limits": list(LIMITS),
        "n_boot": a.n_boot,
        "seed": g4.SEED,
    }
    text = g4.render_report(res)
    if plumbing:
        print(PLUMBING_BANNER + "\n" + text)
        return 0
    out = a.out if a.out is not None else ROOT / "reports" / f"g4_fold{fold}.md"
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")
    if a.json_out is not None:
        a.json_out.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
        print(f"wrote {a.json_out}")
    print(g4.interim_line(res["verdicts"]["interim_rules"], "both arms with production rules"))
    print(g4.spec_line(res["verdicts"]["spec_rules"]))
    return 0


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--oof-run-dir", type=Path, default=None, help="fold-K OOF run folder")
    ap.add_argument("--zs-run-dir", type=Path, required=True, help="the 02 zero-shot run folder")
    ap.add_argument("--fold", type=int, default=None, help="cross-check vs the manifest")
    ap.add_argument("--out", type=Path, default=None, help="default reports/g4_fold<K>.md")
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--plumbing-check", action="store_true", help="NOT A RESULT; writes nothing")
    ap.add_argument("--ocr-cache", type=Path, default=None)
    ap.add_argument("--n-boot", type=int, default=g4.N_BOOT)
    a = ap.parse_args(raw)
    a.argv = raw
    if a.oof_run_dir is None and not a.plumbing_check:
        ap.error("--oof-run-dir is required (or --plumbing-check)")
    if a.plumbing_check and a.oof_run_dir is not None:
        print("note: --plumbing-check uses --zs-run-dir as both arms; --oof-run-dir is ignored")
    if a.plumbing_check and (a.out is not None or a.json_out is not None):
        print("note: --plumbing-check writes no report; --out / --json-out are ignored")
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
