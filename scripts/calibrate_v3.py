"""Calibration v3 PREP: model-agreement feature (fine-tuned vs zero-shot) and its AUROC lift.

CPU only, no model, no network. Three modes (the OOF run folders are produced by
``python -m shipdoc oof infer``; layout in ``src/shipdoc/oof.py``)::

    # exploratory, ONE fold, EXPLORATORY, NOT SHIPPABLE
    uv run python scripts/calibrate_v3.py --folds 0 --oof-run $SHIPDOC_RUNS_DIR/oof_fold0_<sha7> \
        --require-logprobs [--report-md <path>]
    # full, cross-fitted by all 3 supplier folds (refuses unless all three OOF runs are given)
    uv run python scripts/calibrate_v3.py --folds 0 1 2 --oof-run <fold0> --oof-run <fold1> \
        --oof-run <fold2> --require-logprobs
    # plumbing only: the zero-shot run is BOTH arms, fold 0, writes nothing
    uv run python scripts/calibrate_v3.py --plumbing-check

PRE-REGISTRATION (written before any real run; nothing below is changed after seeing a result):

Population and labels. The held-out documents of the OOF folds requested (fold k = the
``val_doc_ids`` of ``splits/folds.json``, train + dev only; any id starting with ``test_`` aborts:
test data has no labels and never enters). Arms: FT = the fold adapter's OOF run (the fine-tuned
model never saw these documents), ZS = the 02 zero-shot run restricted to the same documents. Both
arms go through the production post-rule path (``shipdoc.postrules.postprocess_traces`` with R1,
R2, R3 ON, then coerce / repair) with the HONEST per-fold R3 shapes (learned from the gold of the
OTHER supplier folds); with all rules OFF each arm must reproduce its own ``predictions.json``
(else abort). Labels ``y_correct`` = the unmodified scorer's ``same`` with the scorer's own row
pairing (``shipdoc.confidence.label_doc``) on the arm's post-rule output; they are never
redefined. Population of every metric = emitted non-null fields.

Agreement features (``shipdoc.confidence_v3.AGREE_FEATURES``) compare the post-rule output of the
judged arm with the other arm's, same document: header fields by name, row fields through the
label-free alignment ``align_rows`` (pass 1 all four row fields normalised-equal; pass 2 supplier
part number normalised-equal; pass 3 supplier part number fuzzy similarity >= 0.8, best first;
pass 4 same position, both rows free; leftovers are unaligned). Similarity = rapidfuzz
``fuzz.ratio`` of the casefolded alphanumeric keys / 100; numbers by plain-number parse with the
scorer's 0.005 tolerance. Features: exact, normalised-equal, similarity, both-null, self-null /
other-value, self-value / other-null, row-unaligned, position-aligned, doc-type-differs, document
agreement rate, document unaligned fraction. No gold, no supplier id, no layout cluster id.

Calibrators (three per judging view, all P(correct | emitted)): ``v2`` = every v2 feature of the
judged arm (``shipdoc.confidence_v2.DESIGN_V2_FEATURES``: its own logprobs, OCR support,
validators, cross-page agreement, header-column presence, rule-touched, shape agreement, row
position, scanned, layout novelty with the existing signature, field-type one-hot); ``v2_agree`` =
``v2`` + agreement features; ``agree`` = field-type one-hot + agreement features only. Views: (a)
judging FT correctness (agreement from the ZS arm), (b) the symmetric view, judging ZS correctness
(agreement from the FT arm).

Cross-fit. ``--folds k`` (exploratory): GroupKFold over supplier groups INSIDE fold k's documents,
k_inner = min(5, number of supplier groups) (printed with the group and document counts), sorted
document order, no shuffling; layout novelty from the fitting inner folds only. Pooled structure
only; LR by default, GBM only if better by 0.01 nats of OOF emitted log-loss; this choice is made
ONCE on the ``v2`` design of view (a) and reused for every variant and view (limits selection bias
toward the agreement features). ``--folds 0 1 2``: the v2 cross-fit by the 3 supplier folds (fit on
two folds' fine-tuned OOF outputs, predict the third); structure (pooled vs per_type) and kind
chosen once by the v2 rule on the ``v2`` design of view (a) and reused; nested tau per field type
(header, four row fields) at 95 / 98 / 99% precision for ``v2`` and ``v2_agree``.

Metrics (95% CI: document-level percentile bootstrap, 2000 resamples, seed 42, whole documents).
Per field group (header_all, supplier_part_number, customer_part_number, purchase_order, quantity)
and view: AUROC of the three variants, AUROC of the bare agreement flag (normalised-equal; an
unaligned row counts as disagreeing), the paired lift AUROC(v2_agree) - AUROC(v2) on the same rows
and resamples, base accuracy of the judged arm, n, errors, documents. A group with fewer than 5
errors is flagged. No threshold is selected and nothing is shipped from the exploratory mode.

Caveats printed in every report: one fold = few suppliers; CIs condition on the cross-fit and on
the kind choice; the fold's documents are the ONLY place fine-tuned outputs are out-of-fold, so a
calibrator for fine-tuned outputs cannot be trained on other folds until folds 1 and 2 exist. GG:
"Do not ship anything from this until it is cross-fitted across all 3 folds."

All numbers UNVERIFIED until a verifier recomputes them. Outputs are aggregates and numeric
per-field tables (doc id, field name, probabilities, labels, agreement flags): no gold or
predicted value.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from shipdoc import confidence as cf
from shipdoc import confidence_v2 as c2
from shipdoc import confidence_v3 as c3
from shipdoc import eval as ev
from shipdoc import meta as meta_mod
from shipdoc import paths, postrules
from shipdoc.coerce import coerce_predictions, repair_predictions
from shipdoc.postrules import RuleConfig
from shipdoc.replay import read_trace
from shipdoc.runcompat import ResolutionMismatchError, assert_runs_share_resolution

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("calibrate_v2", ROOT / "scripts" / "calibrate_v2.py")
assert _spec is not None and _spec.loader is not None
v2s = importlib.util.module_from_spec(_spec)
sys.modules["calibrate_v2"] = v2s
_spec.loader.exec_module(v2s)

ZS_RUN = Path(
    "D:/shipdoc/runs/zeroshot500/zeroshot500_qwen35_4b_img_only_keyed_42b812b"
)  # the 02 zero-shot run
FOLD_LABEL = "EXPLORATORY, NOT SHIPPABLE"
EXPLORATORY_BANNER = (
    f"{FOLD_LABEL}: single-fold measurement. GG: do not ship anything from this until it is "
    "cross-fitted across all 3 folds. UNVERIFIED: aggregates computed by scripts/calibrate_v3.py; "
    "a verifier has not recomputed them."
)
FULL_BANNER = (
    "UNVERIFIED: aggregates computed by scripts/calibrate_v3.py (3-fold cross-fitted); a verifier "
    "has not recomputed them."
)
PLUMBING = "PLUMBING CHECK NOT A RESULT"
VIEWS = ("judge_ft", "judge_zs")
CAVEATS = (
    "One fold holds few suppliers: the supplier-grouped cross-fit inside it trains each "
    "calibrator on a handful of suppliers and the group / document counts above are small.",
    "CIs resample documents but condition on the cross-fit and on the LR-vs-GBM choice.",
    "The held-out documents of one fold are the ONLY place the fine-tuned outputs are "
    "out-of-fold, so a calibrator for fine-tuned outputs cannot be trained on other folds until "
    "folds 1 and 2 exist.",
    "Labels are post-rule; population = emitted non-null fields; the zero-shot arm is the 02 run "
    "restricted to the same documents.",
)


class ArmError(RuntimeError):
    """An arm's run folder failed a fail-closed check (messages never quote a value)."""


@dataclass
class Arm:
    """One arm after the production post-rule path, with its v2 table and labels."""

    post: dict[str, Any]
    summary: dict[str, Any]
    tv: c2.TableV2
    sigs: dict[str, np.ndarray | None]
    y: np.ndarray
    n_logprob_docs: int


# ---------------------------------------------------------------------------------------------
# Loading (fail closed)
# ---------------------------------------------------------------------------------------------


def load_oof_run(run_dir: Path) -> tuple[int, list[str], dict[str, Any], dict[str, Any]]:
    """``(manifest oof.fold, trace doc ids, predictions, traces)`` of a COMPLETE OOF run folder.

    Test ids anywhere in the run abort; a run whose progress is not ``complete`` or whose
    predictions and traces disagree is refused.
    """
    run_dir = Path(run_dir)
    man_p, prog_p = run_dir / "manifest.json", run_dir / "progress.json"
    if not man_p.is_file():
        raise ArmError(f"{run_dir.name}: no manifest.json")
    man = json.loads(man_p.read_text(encoding="utf-8"))
    oof = man.get("oof")
    if (
        not isinstance(oof, dict)
        or isinstance(oof.get("fold"), bool)
        or not isinstance(oof.get("fold"), int)
    ):
        raise ArmError(f"{run_dir.name}: manifest has no integer oof.fold (not an OOF run)")
    status = (
        json.loads(prog_p.read_text(encoding="utf-8")).get("status") if prog_p.is_file() else None
    )
    if status != "complete":
        raise ArmError(f"{run_dir.name}: run is not complete (status {status!r})")
    traces_l = read_trace(run_dir / "trace.jsonl")
    ids = [t["doc_id"] for t in traces_l]
    preds = json.loads((run_dir / "predictions.json").read_text(encoding="utf-8"))
    c2.assert_no_test_ids(list(ids) + list(preds))
    if set(preds) != set(ids):
        raise ArmError(
            f"{run_dir.name}: predictions.json and trace.jsonl cover different documents"
        )
    return int(oof["fold"]), ids, preds, {t["doc_id"]: t for t in traces_l}


def load_zs_restricted(run_dir: Path, ids: Sequence[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The zero-shot run (predictions, traces) restricted to `ids`; every id must be present."""
    run_dir = Path(run_dir)
    preds = json.loads((run_dir / "predictions.json").read_text(encoding="utf-8"))
    traces = {t["doc_id"]: t for t in read_trace(run_dir / "trace.jsonl")}
    c2.assert_no_test_ids(list(preds) + list(traces))
    miss = [d for d in ids if d not in preds or d not in traces]
    if miss:
        raise ArmError(f"{run_dir.name}: {len(miss)} requested documents missing from the run")
    return {d: preds[d] for d in ids}, {d: traces[d] for d in ids}


def prepare_arm(
    ids: Sequence[str],
    preds: Mapping[str, Any],
    traces: Mapping[str, Any],
    shapes_fn: Any,
    ocr_cache: Path | None,
    gold: Mapping[str, Any],
    ocr: Mapping[str, list[Any]],
    require_logprobs: bool,
) -> Arm:
    """Post-rule predictions (production path), v2 table and labels of one arm."""
    c2.assert_no_test_ids(ids)
    tr = [traces[d] for d in ids]
    n_lp = sum(1 for t in tr if any(p.get("field_logprobs") for p in t["pages"]))
    if require_logprobs and n_lp < len(ids):
        raise ArmError(f"only {n_lp}/{len(ids)} documents carry field_logprobs")
    off_raw = postrules.postprocess_traces(tr, RuleConfig.all_off(), None, None)[0]
    off, _ = repair_predictions(coerce_predictions(dict(off_raw)))
    if off != {d: preds[d] for d in ids}:
        raise ArmError("all rules OFF does not reproduce the run's predictions.json")
    post, changes, summary = v2s.post_rule_predictions(tr, shapes_fn, ocr_cache)
    tv, sigs = c3.build_table_v2(ids, post, traces, ocr, changes)
    lab = {d: cf.label_doc(post[d], gold[d]) for d in ids}
    y, _ = cf.label_arrays(tv.keys, lab)
    return Arm(post, summary, tv, sigs, y, n_lp)


# ---------------------------------------------------------------------------------------------
# Argument gating
# ---------------------------------------------------------------------------------------------


def gate_args(folds: Sequence[int], oof_runs: Sequence[Path], plumbing: bool) -> str | None:
    """Refusal message for an invalid fold / run combination (None when acceptable).

    Plumbing takes no OOF run. One fold takes exactly one OOF run. The full mode takes folds
    0 1 2 and exactly three runs; any other combination is refused before any file is read.
    """
    if plumbing:
        return "--plumbing-check takes no --oof-run" if oof_runs else None
    fs = sorted(folds)
    if len(set(fs)) != len(fs):
        return f"--folds {fs} repeats a fold"
    if len(fs) == 1:
        return None if len(oof_runs) == 1 else "one fold needs exactly one --oof-run"
    if fs != [0, 1, 2]:
        return f"--folds must be one fold (exploratory) or exactly 0 1 2, got {fs}"
    if len(oof_runs) != 3:
        return f"the 3-fold mode needs all three OOF runs, got {len(oof_runs)}"
    return None


# ---------------------------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------------------------


def write_outputs(
    res: dict[str, Any],
    tables: Mapping[str, tuple[c2.TableV2, np.ndarray, np.ndarray]],
    out_dir: Path,
) -> None:
    """JSON aggregates plus CSV tables; no extracted value is ever written.

    `tables` maps a view to ``(v2 table, y_correct, agreement matrix)``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    oofs = {v["view"]: v.pop("oof") for v in res["views"]}
    (out_dir / "calibration_v3.json").write_text(
        json.dumps(res, indent=1, default=float), encoding="utf-8"
    )
    v2s.write_flat_csv(out_dir / "lift_table.csv", [r for v in res["views"] for r in v["records"]])
    taus = [r for v in res["views"] for r in v.get("tau", [])]
    if taus:
        v2s.write_flat_csv(out_dir / "tau_field_types_v3.csv", taus)
    flag = c3.AGREE_FEATURES.index("ag_norm")
    unal = c3.AGREE_FEATURES.index("ag_row_unaligned")
    rows: list[list[Any]] = []
    for view, (tv, y, agree) in tables.items():
        for i, k in enumerate(tv.keys):
            rows.append(
                [view, *k, int(tv.emitted[i]), int(y[i]), *(oofs[view][v][i] for v in c3.VARIANTS),
                 int(agree[i, flag]), int(agree[i, unal])]
            )  # fmt: skip
    v2s.write_csv(
        out_dir / "oof_fields_v3.csv",
        ["view", "doc_id", "scope", "field", "row_idx", "emitted", "y_correct",
         *(f"p_{v}" for v in c3.VARIANTS), "agree_norm", "row_unaligned"],
        rows,
    )  # fmt: skip


def _ci(d: Mapping[str, float], pct: bool = False) -> str:
    return v2s._ci(d, pct)


def _tab(head: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    return v2s._tab(head, rows)


def render_md(res: Mapping[str, Any], prov: Mapping[str, Any]) -> str:
    """Aggregate-only markdown report (banner depends on the mode)."""
    full = res["mode"] == "cross_fitted_3_fold"
    L = ["# Calibration v3 prep: fine-tuned vs zero-shot agreement\n"]
    L.append(f"> **{FULL_BANNER if full else EXPLORATORY_BANNER}**\n")
    L.append(
        f"Provenance: OOF run(s) `{prov['oof_runs']}`, zero-shot run `{prov['zs_run']}`, repo HEAD "
        f"`{prov['head']}` (confidence_v3.py sha256 `{prov['sha_module'][:12]}`, calibrate_v3.py "
        f"sha256 `{prov['sha_script'][:12]}`), command `{prov['command']}`, wall time "
        f"{prov['wall_s']:.0f} s. Bootstrap: document-level percentile, {prov['n_boot']} "
        "resamples, seed 42. Test data not used. Protocol: docstring of "
        "`scripts/calibrate_v3.py`.\n"
    )
    L.append(
        f"Documents {res['n_docs']}; R3 shapes per-fold honest, equal to the all-gold set per "
        f"fold: {prov['shapes_equal']}. Rules summary FT `{json.dumps(prov['rules_ft'])}`; ZS "
        f"`{json.dumps(prov['rules_zs'])}`.\n"
    )
    for v in res["views"]:
        title = {"judge_ft": "judging the FINE-TUNED arm", "judge_zs": "judging the ZERO-SHOT arm"}
        L.append(f"\n## View: {title[v['view']]}\n")
        ll = json.dumps({k: round(x, 4) for k, x in v["emitted_oof_log_loss"].items()})
        sel = f"emitted OOF log-loss {ll}" if v["emitted_oof_log_loss"] else "choice reused"
        head = (
            f"Structure **{v['structure']}**, kind **{v['kind']}** (chosen once on the v2 design "
            f"of the fine-tuned view; {sel})."
        )
        if not full:
            head = (
                f"{v['n_docs']} documents, {v['n_groups']} supplier groups, inner GroupKFold "
                f"k = {v['k']}. " + head
            )
        L.append(head + "\n")
        rows = [
            [r["group"], r["n"], r["n_wrong"], r["docs"], f"{r['base_acc']:.3f}",
             _ci(r["auroc_v2"]), _ci(r["auroc_v2_agree"]), _ci(r["auroc_agree"]),
             _ci(r["auroc_flag"]), _ci(r["lift"]), r["note"]]
            for r in v["records"]
        ]  # fmt: skip
        L += _tab(
            ["group", "fields", "wrong", "docs", "base acc", "AUROC v2", "AUROC v2+agree",
             "AUROC agree only", "AUROC flag", "lift (v2+agree - v2)", "note"],
            rows,
        )  # fmt: skip
        if full and v.get("tau"):
            L.append("\nNested per-field-type auto-accept (precision % / coverage %, 95% CI):\n")
            trows = []
            for r in v["tau"]:
                trows.append([
                    r["variant"], r["population"], f"{int(100 * r['target'])}%", r["n"],
                    _ci(r["precision_accepted"], True), _ci(r["accepted_share"], True),
                ])  # fmt: skip
            L += _tab(["variant", "type", "target", "fields", "precision", "coverage"], trows)
    L.append("\n## Caveats\n")
    L += [f"- {c}" for c in CAVEATS]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------------


def plumbing_report(res: Mapping[str, Any]) -> str:
    """Counts only (no metric value): what a plumbing run produced."""
    lines = [PLUMBING + " (zero-shot used as BOTH arms; agreement is trivially 1; nothing written)"]
    for v in res["views"]:
        n_cells = sum(r["n"] for r in v["records"])
        lines.append(
            f"view {v['view']}: docs {v['n_docs']}, supplier groups {v['n_groups']}, inner k "
            f"{v['k']}, kind {v['kind']}, field groups {len(v['records'])}, emitted fields in "
            f"groups {n_cells}, lift rows {len(v['records'])}, variants {list(v['oof'])}"
        )
    return "\n".join(lines)


def run(a: argparse.Namespace) -> int:
    """Real run, full run or plumbing check."""
    t0 = time.time()
    msg = gate_args(a.folds, a.oof_run, a.plumbing_check)
    if msg:
        print(f"REFUSED (closed): {msg}", file=sys.stderr)
        return 2
    ids500 = json.loads((ROOT / "splits" / "zeroshot500.json").read_text(encoding="utf-8"))
    c2.assert_no_test_ids(ids500)
    folds_json = json.loads((ROOT / "splits" / "folds.json").read_text(encoding="utf-8"))
    required = [0] if a.plumbing_check else sorted(a.folds)
    ft_preds: dict[str, Any] = {}
    ft_traces: dict[str, Any] = {}
    try:
        if a.plumbing_check:
            ids = sorted(next(f for f in folds_json["folds"] if f["fold"] == 0)["val_doc_ids"])
            c2.assert_no_test_ids(ids)
        else:
            loaded = [load_oof_run(p) for p in a.oof_run]
            # one input resolution across every FT fold run and the ZS run (default ZS_RUN is the
            # 1260-token run: at native it must be overridden, else this refuses)
            assert_runs_share_resolution(
                {"ZS run": a.zs_run_dir, **{f"FT run {i}": p for i, p in enumerate(a.oof_run)}}
            )
            c3.check_fold_coverage([(f, i) for f, i, _, _ in loaded], folds_json, ids500, required)
            for _, _ids, p, t in loaded:
                ft_preds.update(p)
                ft_traces.update(t)
            ids = sorted(ft_preds)
        zs_preds, zs_traces = load_zs_restricted(a.zs_run_dir, ids)
        if a.plumbing_check:
            ft_preds, ft_traces = zs_preds, zs_traces
    except (ArmError, c3.OofCoverageError, c2.NoTestDataError, ResolutionMismatchError) as exc:
        print(f"FAIL (closed): {exc}", file=sys.stderr)
        return 2
    gold, meta = v2s.v1cal.load_corpus(ids500)
    doc_fold = {d: int(folds_json["doc_fold"][d]) for d in ids500}
    groups = {d: meta[d]["supplier_group"] for d in ids500}
    labels = meta_mod.load_labels("train") + meta_mod.load_labels("dev")
    shapes_fn, shapes_equal = v2s.fold_shape_fn(labels, doc_fold, groups)
    ocr = ev.load_ocr_pages(list(ids))
    try:
        ft = prepare_arm(
            ids, ft_preds, ft_traces, shapes_fn, a.ocr_cache, gold, ocr, a.require_logprobs
        )
        zs = prepare_arm(
            ids, zs_preds, zs_traces, shapes_fn, a.ocr_cache, gold, ocr, a.require_logprobs
        )
    except ArmError as exc:
        print(f"FAIL (closed): {exc}", file=sys.stderr)
        return 2
    pairs = c3.align_all(ft.post, zs.post, ids)
    ag_ft = c3.agreement_matrix(ft.tv.keys, ft.post, zs.post, pairs, True)
    ag_zs = c3.agreement_matrix(zs.tv.keys, zs.post, ft.post, pairs, False)
    if a.plumbing_check:  # same run on both sides: every emitted field must agree, none unaligned
        nrm, una = c3.AGREE_FEATURES.index("ag_norm"), c3.AGREE_FEATURES.index("ag_row_unaligned")
        assert np.all(ag_ft[ft.tv.emitted, nrm] == 1) and not ag_ft[:, una].any(), "plumbing"
    views: list[dict[str, Any]] = []
    fixed: tuple[str, str] | None = None  # (structure, kind) chosen once, on the fine-tuned view
    for view, arm, agree in (("judge_ft", ft, ag_ft), ("judge_zs", zs, ag_zs)):
        if len(required) == 1:
            r = c3.analyze_fold_view(view, arm.tv, arm.y, agree, groups, arm.sigs, a.n_boot, fixed)
        else:
            nov = cf.compute_novelty_table(arm.sigs, doc_fold)
            r = c3.analyze_full_view(
                view, arm.tv, arm.y, agree, doc_fold, groups, nov, a.n_boot, fixed
            )
        fixed = (r["structure"], r["kind"])
        views.append(r)
    res: dict[str, Any] = {
        "mode": views[0]["mode"],
        "n_docs": len(ids),
        "folds": required,
        "label": FOLD_LABEL if len(required) == 1 else "cross-fitted across 3 folds",
        "views": views,
    }
    if a.plumbing_check:
        print(plumbing_report(res))
        return 0
    prov = {
        "oof_runs": ", ".join(p.name for p in a.oof_run),
        "zs_run": a.zs_run_dir.name,
        "head": v2s.git_head(),
        "sha_module": v2s.sha256_file(ROOT / "src" / "shipdoc" / "confidence_v3.py"),
        "sha_script": v2s.sha256_file(Path(__file__)),
        "command": "uv run python scripts/calibrate_v3.py " + " ".join(sys.argv[1:]),
        "wall_s": time.time() - t0,
        "n_boot": a.n_boot,
        "shapes_equal": shapes_equal,
        "rules_ft": ft.summary,
        "rules_zs": zs.summary,
    }
    res["provenance"] = prov
    md = render_md(res, prov)
    out_dir = a.out_dir or paths.runs_dir() / "calibration_v3" / (
        "_".join(p.name for p in a.oof_run)
    )
    write_outputs(
        res, {"judge_ft": (ft.tv, ft.y, ag_ft), "judge_zs": (zs.tv, zs.y, ag_zs)}, out_dir
    )
    (out_dir / "calibration_v3.md").write_text(md, encoding="utf-8")
    if a.report_md:
        a.report_md.write_text(md, encoding="utf-8")
    print(f"{res['label']}: outputs in {out_dir}; wall {time.time() - t0:.0f} s")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--folds", type=int, nargs="+", default=[0])
    ap.add_argument("--oof-run", type=Path, action="append", default=[])
    ap.add_argument("--zs-run-dir", type=Path, default=ZS_RUN)
    ap.add_argument("--plumbing-check", action="store_true")
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--ocr-cache", type=Path, default=None)
    ap.add_argument("--report-md", type=Path, default=None)
    ap.add_argument("--n-boot", type=int, default=c3.N_BOOT)
    ap.add_argument("--require-logprobs", action="store_true")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
