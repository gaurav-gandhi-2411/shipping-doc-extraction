"""Calibration v2 on the POST-RULE outputs of the 02 zero-shot run (CPU only, no model, no network).

    uv run python scripts/calibrate_v2.py --run-dir $SHIPDOC_RUNS_DIR/zeroshot500/<run folder> \
        --require-logprobs [--report-md reports/calibration_v2.md]

PRE-REGISTRATION (written before the first real run; nothing below is changed after seeing a
result):

Population and labels. The 500 train + dev docs of ``splits/zeroshot500.json`` (any id starting
with ``test_`` aborts: test data has no labels and never enters). Predictions = the production
post-processing path (``shipdoc.postrules.postprocess_traces`` with R1, R2, R3 ON, then
``coerce_predictions`` / ``repair_predictions``); R3 shapes are the per-fold HONEST shapes (learned
by ``rules.learn_slot_shapes`` from the gold of the OTHER supplier folds); the script records
whether they equal the all-gold set. Field label ``y_correct`` and ``y_null`` = unmodified scorer
``same`` semantics with the scorer's own row pairing (``shipdoc.confidence.label_doc``) applied to
the post-rule predictions. Document label = ``score_doc(..)["exact"]`` (documents_fully_correct:
right type, every header field right, exactly the right rows). Labels are never redefined.

Models. P(correct | emitted non-null) with all v1 features plus the v2 generic features
(``shipdoc.confidence_v2.EXTRA_FEATURES``; no supplier id, no layout cluster id), cross-fitted by
the 3 supplier folds (fit on 2, predict the third). Exactly two structural choices: ``pooled`` vs
``per_type`` (header, supplier_part_number, customer_part_number, quantity, purchase_order), each
with LR by default and GBM only if it is better by 0.01 nats of OOF log-loss; the structure with
the strictly lower emitted OOF log-loss wins (ties: pooled). No P(null) model, no null policy.
Document model: logistic regression on aggregated field P(correct) (OOF) + doc features,
cross-fitted by supplier fold.

Metrics (95% CIs: doc-level percentile bootstrap, 2000 resamples, seed 42; documents resampled
whole). Slices: all 500 docs (OOF), dev (the 100 dev docs' OOF predictions), train, invoice-only,
waybill-only, scanned, digital. (a) per-field AUROC of P(correct) on emitted fields: header-all,
each header field with >= 5 wrong in the slice (else "n<5 errors"), each row field; v1 (pre-rule
labels, as published), v1 scored on the post-rule labels (matched rows only) and v2 on the same
matched rows, plus v2 on all its rows. (b) per-field-type tau (header, four row fields) for a
precision target; targets 95 / 98 / 99%, primary 98%. NESTED selection: for fold k tau is chosen
(lowest tau whose accepted set has precision >= target) on the OOF predictions of the other two
folds and applied to fold k, decisions pooled; precision, coverage (accepted share), review rate,
error recall with CIs. In-sample tau (chosen on the evaluated rows) is reported only for
reference and labelled. "Coverage at target" = the nested accepted share at that target (its
nested precision is reported next to it and may fall short of the target); the in-sample
largest-accepted-share with precision >= target is given as an optimistic reference, point only.
Combined review rate = flagged share of all emitted fields when each type uses its own tau; a
single nested global tau is the reference. (c) document auto-accept at 95 / 98% with the same
nested protocol; attained iff nested precision >= target AND nested coverage >= 5%, otherwise
"NOT ATTAINABLE on this evidence" with the best in-sample precision at >= 5% coverage. (d)
risk-coverage CSV, ECE (15 equal-width, 15 equal-mass), reliability CSV per slice and group.
CIs condition on the selected taus (tau selection variance is not resampled) and the dev slice
has only 100 docs.

All numbers UNVERIFIED until a verifier recomputes them. Outputs are aggregates and numeric
per-field tables (doc id, field name, probabilities, labels): no gold or predicted value.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from shipdoc import confidence as cf
from shipdoc import confidence_v2 as c2
from shipdoc import eval as ev
from shipdoc import meta as meta_mod
from shipdoc import paths, postrules, rules
from shipdoc.cluster import layout_signature
from shipdoc.coerce import coerce_predictions, repair_predictions
from shipdoc.extract import ROW_KEYS
from shipdoc.postrules import RuleConfig

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("calibrate", ROOT / "scripts" / "calibrate.py")
assert _spec is not None and _spec.loader is not None
v1cal = importlib.util.module_from_spec(_spec)
sys.modules["calibrate"] = v1cal  # dataclasses / typing resolve annotations via sys.modules
_spec.loader.exec_module(v1cal)

SLICES = ("all", "dev", "train", "invoice_only", "waybill_only", "scanned", "digital")
DOC_TARGETS = (0.95, 0.98)
PRIMARY = 0.98
MIN_ERRORS = 5
BANNER = (
    "UNVERIFIED: aggregates computed by scripts/calibrate_v2.py from the 02 zero-shot run; a "
    "verifier has not recomputed them."
)

# ---------------------------------------------------------------------------------------------
# Post-rule predictions (production path)
# ---------------------------------------------------------------------------------------------


def fold_shape_fn(
    labels: Sequence[Mapping[str, Any]],
    doc_fold: Mapping[str, int],
    groups: Mapping[str, str],
) -> tuple[Callable[[str], rules.SlotShapes], dict[int, bool]]:
    """``doc_id -> shapes`` learned from the gold of the OTHER supplier folds, plus per fold
    whether the learned shapes equal the all-gold set. Raises on any doc / supplier overlap."""
    all_shapes = rules.learn_slot_shapes([x for x in labels if x["doc_id"] in doc_fold])
    per_fold: dict[int, rules.SlotShapes] = {}
    equal: dict[int, bool] = {}
    for k in sorted(set(doc_fold.values())):
        held = {d for d, f in doc_fold.items() if f == k}
        learn = [x for x in labels if x["doc_id"] in doc_fold and doc_fold[x["doc_id"]] != k]
        ids = {x["doc_id"] for x in learn}
        if ids & held or {groups[d] for d in ids} & {groups[d] for d in held}:
            raise SystemExit(f"fold {k}: shape learner overlaps the held-out docs / suppliers")
        per_fold[k] = rules.learn_slot_shapes(learn)
        equal[k] = per_fold[k] == all_shapes
    return (lambda d: per_fold[doc_fold[d]]), equal


def post_rule_predictions(
    traces: Sequence[Mapping[str, Any]],
    shapes: Any,
    ocr_root: Path | None,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """``(final predictions, {doc_id: rule change records}, summary)`` of the production path."""
    preds, records, summary = postrules.postprocess_traces(traces, RuleConfig(), shapes, ocr_root)
    final, _ = repair_predictions(coerce_predictions(dict(preds)))
    return final, {r["doc_id"]: r["changes"] for r in records}, summary


# ---------------------------------------------------------------------------------------------
# Slices and small helpers
# ---------------------------------------------------------------------------------------------


def slice_doc_sets(
    doc_ids: Sequence[str], meta: Mapping[str, Mapping[str, Any]], scanned: Mapping[str, bool]
) -> dict[str, set[str]]:
    """Slice name -> doc ids (gold-side doc type from meta, scanned from the image feature)."""
    ids = set(doc_ids)
    wb = {d for d in ids if meta[d].get("waybill")}
    return {
        "all": ids,
        "dev": {d for d in ids if d.startswith("dev_")},
        "train": {d for d in ids if d.startswith("train_")},
        "invoice_only": ids - wb,
        "waybill_only": wb,
        "scanned": {d for d in ids if scanned.get(d)},
        "digital": {d for d in ids if not scanned.get(d, True)},
    }


def doc_mask(ids: np.ndarray, docs: set[str]) -> np.ndarray:
    """Bool mask of the array entries whose doc id is in `docs`."""
    return np.fromiter((str(d) in docs for d in ids), dtype=bool, count=len(ids))


def _ci(d: Mapping[str, float], pct: bool = True) -> str:
    def f(x: float) -> str:
        if x is None or np.isnan(x):
            return "n/a"
        return f"{100 * x:.1f}" if pct else f"{x:.3f}"

    return f"{f(d['point'])} [{f(d['lo'])}, {f(d['hi'])}]"


def write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    """Write a CSV (numbers, ids and field names only: callers never pass values)."""
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def sha256_file(p: Path) -> str:
    """Hex sha256 of a file."""
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git_head() -> str:
    """Short HEAD sha (``unknown`` when git is unavailable). Read-only."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=ROOT, check=True,
        ).stdout.strip()  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# ---------------------------------------------------------------------------------------------
# v1 outputs (as published) for the side-by-side
# ---------------------------------------------------------------------------------------------


def load_v1_oof(path: Path) -> dict[tuple[str, str, str, int], tuple[float, int]]:
    """``(doc, scope, field, row_idx) -> (p_correct, y_correct)`` of v1's emitted fields."""
    out: dict[tuple[str, str, str, int], tuple[float, int]] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            p = float(r["p_correct"]) if r["p_correct"] not in ("", "nan") else float("nan")
            if r["emitted"] == "1" and not np.isnan(p):
                out[(r["doc_id"], r["scope"], r["field"], int(r["row_idx"]))] = (
                    p,
                    int(r["y_correct"]),
                )
    return out


class Rows:
    """Scored rows: doc id, scope, field name, score, binary label (all numpy, aligned)."""

    def __init__(self, doc: Any, scope: Any, field: Any, score: Any, y: Any) -> None:
        self.doc, self.scope, self.field = np.asarray(doc), np.asarray(scope), np.asarray(field)
        self.score, self.y = np.asarray(score, dtype=float), np.asarray(y, dtype=int)


def group_masks(r: Rows) -> dict[str, np.ndarray]:
    """``header_all``, ``header.<field>`` for each header field present, ``row.<field>``."""
    hdr = r.scope == "header"
    out = {"header_all": hdr}
    for f in sorted(set(r.field[hdr].tolist())):
        out[f"header.{f}"] = hdr & (r.field == f)
    for f in ROW_KEYS:
        out[f"row.{f}"] = (r.scope == "row") & (r.field == f)
    return out


def auroc_records(
    variant: str,
    r: Rows,
    slices: Mapping[str, set[str]],
    n_boot: int,
) -> list[dict[str, Any]]:
    """One record per slice x group: AUROC with CI, class counts, ``n<5 errors`` note."""
    recs: list[dict[str, Any]] = []
    gm = group_masks(r)
    for sname, docs in slices.items():
        sm = doc_mask(r.doc, docs)
        for g, m in gm.items():
            sel = sm & m
            n_wrong = int((r.y[sel] == 0).sum())
            n_docs = len(set(r.doc[sel].tolist()))
            rec: dict[str, Any] = {
                "variant": variant,
                "slice": sname,
                "group": g,
                "n": int(sel.sum()),
                "n_wrong": n_wrong,
                "docs": n_docs,
            }
            if not sel.any():
                continue
            if g.startswith("header.") and n_wrong < MIN_ERRORS:
                rec.update(note="n<5 errors", point=np.nan, lo=np.nan, hi=np.nan)
            else:
                rec.update(note="", **{
                    k: v
                    for k, v in c2.auroc_boot(r.score[sel], r.y[sel], r.doc[sel], n_boot).items()
                    if k in ("point", "lo", "hi")
                })  # fmt: skip
            recs.append(rec)
    return recs


# ---------------------------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------------------------


def threshold_records(
    name: str,
    conf: np.ndarray,
    correct: np.ndarray,
    ids: np.ndarray,
    fold_of: np.ndarray,
    base_mask: np.ndarray,
    targets: Sequence[float],
    slices: Mapping[str, set[str]],
    n_boot: int,
) -> tuple[list[dict[str, Any]], dict[float, np.ndarray], dict[float, np.ndarray]]:
    """Nested and in-sample accept metrics of one population over targets x slices.

    Returns ``(records, {target: nested accept}, {target: in-sample accept})``.
    """
    recs: list[dict[str, Any]] = []
    nested: dict[float, np.ndarray] = {}
    insamp: dict[float, np.ndarray] = {}
    for t in targets:
        acc, taus, _ = c2.nested_accept(conf, correct, fold_of, base_mask, t)
        acc_in, tau_in = c2.insample_accept(conf, correct, base_mask, t)
        nested[t], insamp[t] = acc, acc_in
        for sname, docs in slices.items():
            sel = base_mask & doc_mask(ids, docs)
            if not sel.any():
                continue
            for scheme, a, tau in (("nested", acc, taus), ("in_sample", acc_in, tau_in)):
                m = c2.accept_metrics(a[sel], correct[sel], ids[sel], n_boot)
                recs.append(
                    {
                        "population": name,
                        "target": t,
                        "slice": sname,
                        "scheme": scheme,
                        "tau": json.dumps(tau) if isinstance(tau, dict) else tau,
                        "n": int(sel.sum()),
                        "docs": len(set(ids[sel].tolist())),
                        "oracle_coverage_in_sample": c2.coverage_at_target(
                            conf[sel], correct[sel], t
                        ),
                        **{k: m[k] for k in cf.REVIEW_METRICS},
                    }
                )
    return recs, nested, insamp


def reliability_and_curves(
    groups: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    slices: Mapping[str, set[str]],
) -> tuple[list[dict[str, Any]], list[list[Any]], list[list[Any]]]:
    """(calibration summaries, reliability rows, risk-coverage rows) per slice x group.

    `groups` maps a name to ``(confidence, correct, doc ids)`` of its population.
    """
    summ: list[dict[str, Any]] = []
    rel: list[list[Any]] = []
    cur: list[list[Any]] = []
    for sname, docs in slices.items():
        for g, (conf, correct, ids) in groups.items():
            sel = doc_mask(ids, docs)
            if not sel.any():
                continue
            c, y = conf[sel], correct[sel].astype(float)
            summ.append({"slice": sname, "group": g, **cf.calibration_summary(c, y)})
            rel += [
                [sname, g, r["lo"], r["hi"], r["n"], r["mean_conf"], r["accuracy"]]
                for r in cf.reliability(c, y)
            ]
            cur += [
                [sname, g, p["coverage"], p["accuracy"], p["min_conf"]]
                for p in cf.coverage_curve(c, y, 100)
            ]
    return summ, rel, cur


def analyze(
    tv: c2.TableV2,
    y_correct: np.ndarray,
    doc_ids: Sequence[str],
    y_doc: np.ndarray,
    doc_fold: Mapping[str, int],
    groups: Mapping[str, str],
    meta: Mapping[str, Mapping[str, Any]],
    novelty: Mapping[int, Mapping[str, float]],
    v1_path: Path | None,
    n_boot: int,
) -> dict[str, Any]:
    """Whole evaluation from labelled features; returns aggregates plus the arrays to export."""
    c2.assert_no_test_ids(doc_ids)
    em = tv.emitted
    ids = tv.table.doc_ids
    scanned_col = tv.table.X[:, cf.BASE_FEATURES.index("scanned")] == 1.0
    scanned_doc = {d: bool(s) for d, s in zip(ids.tolist(), scanned_col.tolist(), strict=True)}
    slices = slice_doc_sets(doc_ids, meta, scanned_doc)
    types = tv.types()
    fold_of = np.array([doc_fold[d] for d in ids.tolist()])
    res: dict[str, Any] = {"slice_docs": {k: len(v) for k, v in slices.items()}}

    choice = c2.select_structure(tv, y_correct, doc_fold, novelty, groups)
    p = choice.oof
    res["model_selection"] = {
        "chosen_structure": choice.structure,
        "chosen_kind": choice.kind,
        "kind_by_structure": choice.kind_by_structure,
        "emitted_oof_log_loss": choice.logloss,
        "gbm_margin_nats": cf.GBM_MARGIN,
        "note": "chosen on the same OOF predictions that are reported (mild optimism)",
    }
    rt = {r: tv.extras[:, c2.EXTRA_FEATURES.index(f"rt_{r}")] == 1.0 for r in c2.RULE_NAMES}
    res["rule_touched_emitted_fields"] = {
        r: {"n": int((m & em).sum()), "correct": int((m & em & y_correct).sum())}
        for r, m in rt.items()
    }

    # (a) AUROC, v1 vs v2
    scope = np.array([k.scope for k in tv.keys])
    field = np.array([k.field for k in tv.keys])
    keyt = [(k.doc_id, k.scope, k.field, k.row_idx) for k in tv.keys]
    v2_rows = Rows(ids[em], scope[em], field[em], p[em], y_correct[em])
    au = auroc_records("v2_all_rows", v2_rows, slices, n_boot)
    # Ablations (diagnostic, not used in any selection): the default LR model on the v2 features,
    # and the chosen structure/kind on the v1 features only (extras zeroed) with post-rule labels.
    # They separate the model-family effect and the extra-feature effect from the rule effect.
    lr_oof = (choice.all_oof or {})["pooled/lr"]
    au += auroc_records(
        "abl_v2_features_pooled_lr",
        Rows(ids[em], scope[em], field[em], lr_oof[em], y_correct[em]),
        slices,
        n_boot,
    )
    tv_v1 = c2.TableV2(tv.table, np.zeros_like(tv.extras))
    v1f = c2.cross_fit_v2(
        tv_v1, y_correct, em, doc_fold, choice.kind, novelty, groups, choice.structure
    )
    au += auroc_records(
        "abl_v1_features_same_model",
        Rows(ids[em], scope[em], field[em], v1f[em], y_correct[em]),
        slices,
        n_boot,
    )
    res["ablation_emitted_oof_log_loss"] = {
        "v2_features_pooled_lr": cf.log_loss(lr_oof[em], y_correct[em].astype(float)),
        "v1_features_chosen_model": cf.log_loss(v1f[em], y_correct[em].astype(float)),
        "v2_features_chosen_model": cf.log_loss(p[em], y_correct[em].astype(float)),
    }
    if v1_path is not None and v1_path.is_file():
        v1 = load_v1_oof(v1_path)
        v1_ids = [k for k in v1 if k[0] in set(doc_ids)]
        au += auroc_records(
            "v1_prerule_labels_as_published",
            Rows(
                [k[0] for k in v1_ids], [k[1] for k in v1_ids], [k[2] for k in v1_ids],
                [v1[k][0] for k in v1_ids], [v1[k][1] for k in v1_ids],
            ),
            slices,
            n_boot,
        )  # fmt: skip
        match = np.array([em[i] and keyt[i] in v1 for i in range(len(keyt))])
        p1 = np.array([v1[keyt[i]][0] if match[i] else np.nan for i in range(len(keyt))])
        mrow = lambda s: Rows(ids[match], scope[match], field[match], s[match], y_correct[match])  # noqa: E731
        au += auroc_records("v1_postrule_labels_matched", mrow(p1), slices, n_boot)
        au += auroc_records("v2_matched_rows", mrow(p), slices, n_boot)
        res["v1_join"] = {
            "v1_emitted_rows_in_these_docs": len(v1_ids),
            "v2_emitted_rows": int(em.sum()),
            "matched_rows": int(match.sum()),
            "v2_emitted_without_v1_p": int((em & ~match).sum()),
        }
    else:
        res["v1_join"] = {"note": "v1 oof_fields.csv not found: v1 columns omitted"}
    res["auroc"] = au

    # (b) tau per field type
    tau_recs: list[dict[str, Any]] = []
    nested_by: dict[str, dict[float, np.ndarray]] = {}
    insamp_by: dict[str, dict[float, np.ndarray]] = {}
    for t_name in (*c2.FIELD_TYPES, "all_global_tau"):
        mask = em & ((types == t_name) if t_name != "all_global_tau" else True)
        rs, nested_by[t_name], insamp_by[t_name] = threshold_records(
            t_name, p, y_correct, ids, fold_of, mask, c2.TARGETS, slices, n_boot
        )
        tau_recs += rs
    for scheme, store in (("nested", nested_by), ("in_sample", insamp_by)):
        for t in c2.TARGETS:
            comb = np.zeros(len(em), dtype=bool)
            for t_name in c2.FIELD_TYPES:
                comb |= store[t_name][t]
            for sname, docs in slices.items():
                sel = em & doc_mask(ids, docs)
                if not sel.any():
                    continue
                m = c2.accept_metrics(comb[sel], y_correct[sel], ids[sel], n_boot)
                tau_recs.append(
                    {
                        "population": "all_per_type_tau", "target": t, "slice": sname,
                        "scheme": scheme, "tau": "per type", "n": int(sel.sum()),
                        "docs": len(set(ids[sel].tolist())),
                        "oracle_coverage_in_sample": float("nan"),
                        **{k: m[k] for k in cf.REVIEW_METRICS},
                    }
                )  # fmt: skip
    res["tau"] = tau_recs

    # (c) document level
    Xd, order = c2.doc_table(tv, p, doc_ids)
    p_doc = c2.cross_fit_docs(Xd, y_doc, order, doc_fold, novelty, groups)
    d_ids = np.array(order, dtype=object)
    d_fold = np.array([doc_fold[d] for d in order])
    doc_recs, d_nested, _ = threshold_records(
        "doc", p_doc, y_doc.astype(bool), d_ids, d_fold, np.ones(len(order), dtype=bool),
        DOC_TARGETS, slices, n_boot,
    )  # fmt: skip
    res["doc_level"] = doc_recs
    res["doc_model"] = {
        "features": list(c2.DOC_FEATURES),
        "oof_log_loss": cf.log_loss(p_doc, y_doc.astype(float)),
        "stacking_caveat": (
            "doc features use field-level OOF probabilities; the field models of fold A saw "
            "fold C labels, so doc-model training features carry mild second-level leakage"
        ),
    }
    base_rates: dict[str, Any] = {}
    for sname, docs in slices.items():
        sel = doc_mask(d_ids, docs)
        if sel.any():
            m = c2.accept_metrics(np.ones(int(sel.sum()), dtype=bool), y_doc[sel].astype(bool),
                                  d_ids[sel], n_boot)  # fmt: skip
            base_rates[sname] = {"docs": int(sel.sum()), **m["precision_accepted"]}
    res["doc_base_rate_fully_correct"] = base_rates
    res["doc_attainability"] = {}
    all_d = np.ones(len(order), dtype=bool)
    for t in DOC_TARGETS:
        rec = next(r for r in doc_recs if r["target"] == t and r["slice"] == "all"
                   and r["scheme"] == "nested")  # fmt: skip
        ok = (rec["precision_accepted"]["point"] >= t) and (rec["accepted_share"]["point"] >= 0.05)
        res["doc_attainability"][str(t)] = {
            "attained": bool(ok),
            "best_in_sample": c2.best_precision_at_coverage(
                p_doc[all_d], y_doc[all_d].astype(bool)
            ),
        }
    del d_nested

    # (d) calibration summaries, reliability, risk-coverage
    pops: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for t_name in (*c2.FIELD_TYPES, "all_emitted"):
        m = em & ((types == t_name) if t_name != "all_emitted" else True)
        pops[t_name] = (p[m], y_correct[m], ids[m])
    pops["doc"] = (p_doc, y_doc.astype(bool), d_ids)
    summ, rel, cur = reliability_and_curves(pops, slices)
    res["calibration"] = summ
    res["_reliability"], res["_curves"] = rel, cur
    res["_arrays"] = {"p": p, "p_doc": p_doc, "doc_order": order, "y_doc": y_doc}
    return res


# ---------------------------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------------------------


def _flat(rec: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in rec.items():
        if isinstance(v, dict) and set(v) == {"point", "lo", "hi"}:
            out.update({f"{k}": v["point"], f"{k}_lo": v["lo"], f"{k}_hi": v["hi"]})
        else:
            out[k] = v
    return out


def write_flat_csv(path: Path, recs: Sequence[Mapping[str, Any]]) -> None:
    """CSV of flat records (CI dicts expand to ``x``, ``x_lo``, ``x_hi``)."""
    flat = [_flat(r) for r in recs]
    cols = list(flat[0]) if flat else []
    write_csv(path, cols, [[r.get(c) for c in cols] for r in flat])


def write_outputs(
    res: dict[str, Any], tv: c2.TableV2, y_correct: np.ndarray, y_null: np.ndarray,
    doc_fold: Mapping[str, int], meta: Mapping[str, Mapping[str, Any]], out_dir: Path,
) -> None:  # fmt: skip
    """JSON aggregates plus CSV tables; no extracted value is ever written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = res.pop("_arrays")
    rel, cur = res.pop("_reliability"), res.pop("_curves")
    (out_dir / "calibration_v2.json").write_text(
        json.dumps(res, indent=1, default=float), encoding="utf-8"
    )
    p = arrays["p"]
    write_csv(
        out_dir / "oof_fields_v2.csv",
        ["doc_id", "scope", "field", "row_idx", "emitted", "p_correct_v2", "y_correct", "y_null",
         "rule_touched"],
        [
            [*k, int(tv.emitted[i]), p[i], int(y_correct[i]), int(y_null[i]),
             int(tv.extras[i, c2.EXTRA_FEATURES.index("rt_touched")])]
            for i, k in enumerate(tv.keys)
        ],
    )  # fmt: skip
    write_csv(
        out_dir / "oof_docs_v2.csv",
        ["doc_id", "fold", "waybill", "p_fully_correct", "y_fully_correct"],
        [
            [d, doc_fold[d], int(bool(meta[d].get("waybill"))), arrays["p_doc"][i],
             int(arrays["y_doc"][i])]
            for i, d in enumerate(arrays["doc_order"])
        ],
    )  # fmt: skip
    write_flat_csv(out_dir / "auroc_v1_v2.csv", res["auroc"])
    write_flat_csv(out_dir / "tau_field_types.csv", res["tau"])
    write_flat_csv(out_dir / "doc_level.csv", res["doc_level"])
    write_flat_csv(out_dir / "calibration_summary.csv", res["calibration"])
    write_csv(
        out_dir / "reliability.csv",
        ["slice", "group", "lo", "hi", "n", "mean_conf", "accuracy"],
        rel,
    )
    write_csv(
        out_dir / "risk_coverage.csv", ["slice", "group", "coverage", "precision", "min_conf"], cur
    )


def _tab(head: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + [
        "| " + " | ".join(str(c) for c in r) + " |" for r in rows
    ]


def render_md(res: Mapping[str, Any], prov: Mapping[str, Any]) -> str:
    """Aggregate-only markdown with the honest coverage statement."""
    L = ["# Calibration v2 on post-rule outputs\n", f"> **{BANNER}**\n"]
    L.append(
        f"Provenance: run `{prov['run']}`, repo HEAD `{prov['head']}` (scripts uncommitted at run "
        f"time: confidence_v2.py sha256 `{prov['sha_module'][:12]}`, calibrate_v2.py sha256 "
        f"`{prov['sha_script'][:12]}`), command `{prov['command']}`, wall time "
        f"{prov['wall_s']:.0f} s. Bootstrap: doc-level percentile, 2000 resamples, seed 42. All "
        "500 docs are OOF (supplier-fold cross-fit); `dev` = the 100 dev docs' OOF predictions "
        "(only 100 docs: wide CIs). Test data not used. Pre-registration: docstring of "
        "`scripts/calibrate_v2.py`.\n"
    )
    ms = res["model_selection"]
    L.append(
        f"Model: structure **{ms['chosen_structure']}**, kind **{ms['chosen_kind']}**; emitted "
        f"OOF log-loss {json.dumps({k: round(v, 4) for k, v in ms['emitted_oof_log_loss'].items()})}"  # noqa: E501
        f"; kind per structure {ms['kind_by_structure']}. R3 shapes: per-fold honest; equal to the "
        f"all-gold set per fold: {prov['shapes_equal']}.\n"
    )
    touched = json.dumps(res["rule_touched_emitted_fields"])
    L.append(
        f"Rules summary (production path): `{json.dumps(prov['rules_summary'])}`; rule-touched "
        f"emitted fields `{touched}`.\n"
    )
    L.append("## Honest coverage statement\n")
    L += prov["honest"]
    L.append("\n## (a) AUROC of P(correct), v1 vs v2 (95% CI)\n")
    L.append(
        "Variants: `v1_prerule` = v1 as published (pre-rule outputs and labels); `v1_post` = v1 "
        "probabilities scored on the post-rule labels, matched rows only (separates the rule "
        "effect from the feature effect; rule-filled fields have no v1 probability and drop "
        "out); `v2_matched` = v2 on the same rows; `v2_all` = v2 on all its emitted rows; "
        "`abl_v2feat_LR` = ablation, the default LR model on the v2 features (model-family "
        "effect); `abl_v1feat_samemodel` = ablation, v1 features only with the chosen v2 model "
        "on post-rule labels (extra-feature effect). Ablations are diagnostic and feed no "
        "selection.\n"
    )
    short = {
        "v1_prerule_labels_as_published": "v1_prerule",
        "v1_postrule_labels_matched": "v1_post",
        "v2_matched_rows": "v2_matched",
        "v2_all_rows": "v2_all",
        "abl_v2_features_pooled_lr": "abl_v2feat_LR",
        "abl_v1_features_same_model": "abl_v1feat_samemodel",
    }
    for sname in SLICES:
        recs = [r for r in res["auroc"] if r["slice"] == sname]
        if not recs:
            continue
        L.append(f"\n**slice `{sname}`** ({res['slice_docs'][sname]} docs)\n")
        groups = sorted({r["group"] for r in recs}, key=lambda g: (g.startswith("row"), g))
        rows = []
        for g in groups:
            cells = []
            n_wrong = 0
            note = ""
            for v in short:
                r = next((x for x in recs if x["group"] == g and x["variant"] == v), None)
                if r is None:
                    cells.append("n/a")
                    continue
                if v == "v2_all_rows":
                    n_wrong, note = r["n_wrong"], r["note"]
                cells.append(note if r["note"] else _ci(r, False))
            if note:
                cells = [note] * len(cells)
            rows.append([g, n_wrong, *cells])
        L += _tab(["group", "wrong (v2)", *short.values()], rows)
    L.append("\n## (b) Per-field-type tau, target 98% precision, NESTED (95% CI, %)\n")
    L.append(
        "tau of fold k chosen on the other two folds' OOF predictions, applied to fold k, pooled. "
        "`in-sample` = tau chosen on the evaluated rows (optimistic, reference only). CIs "
        "condition on the taus.\n"
    )
    for sname in SLICES:
        recs = [r for r in res["tau"] if r["slice"] == sname and r["target"] == PRIMARY]
        if not recs:
            continue
        L.append(f"\n**slice `{sname}`**\n")
        rows = []
        for pop in (*c2.FIELD_TYPES, "all_per_type_tau", "all_global_tau"):
            n = next((r for r in recs if r["population"] == pop and r["scheme"] == "nested"), None)
            i = next(
                (r for r in recs if r["population"] == pop and r["scheme"] == "in_sample"), None
            )
            if n is None or i is None:
                continue
            rows.append([
                pop, n["n"], _ci(n["precision_accepted"]), _ci(n["accepted_share"]),
                _ci(n["review_rate"]), _ci(n["error_recall"]),
                f"{_ci(i['precision_accepted'])} / cov {_ci(i['accepted_share'])}",
            ])  # fmt: skip
        L += _tab(["population", "fields", "precision", "coverage", "review rate", "error recall",
                   "in-sample precision / coverage"], rows)  # fmt: skip
    L.append("\n## Coverage at precision targets 95 / 98 / 99% (nested; all and dev)\n")
    L.append("Cell: nested coverage % [CI] ; nested precision % [CI] ; in-sample optimum coverage "
             "(point, optimistic).\n")  # fmt: skip
    for sname in ("all", "dev"):
        L.append(f"\n**slice `{sname}`**\n")
        rows = []
        for pop in (*c2.FIELD_TYPES, "all_global_tau"):
            cells = []
            for t in c2.TARGETS:
                n = next((r for r in res["tau"] if r["population"] == pop and r["slice"] == sname
                          and r["target"] == t and r["scheme"] == "nested"), None)  # fmt: skip
                cells.append("n/a" if n is None else
                             f"{_ci(n['accepted_share'])} ; {_ci(n['precision_accepted'])} ; "
                             f"{100 * n['oracle_coverage_in_sample']:.1f}")  # fmt: skip
            rows.append([pop, *cells])
        L += _tab(["population", "95%", "98%", "99%"], rows)
    L.append("\n## (c) Document auto-accept (fully correct document)\n")
    L.append("Base rate of fully-correct docs (95% CI, %):\n")
    base_rows = [[s, b["docs"], _ci(b)] for s, b in res["doc_base_rate_fully_correct"].items()]
    L += _tab(["slice", "docs", "base rate"], base_rows)
    for t in DOC_TARGETS:
        att = res["doc_attainability"][str(t)]
        L.append(f"\nTarget {int(100 * t)}%: " + (
            "attained (nested precision >= target and coverage >= 5%)." if att["attained"] else
            "**NOT ATTAINABLE on this evidence** (nested precision < target or coverage < 5%); "
            "best in-sample precision at >= 5% coverage: "
            f"{100 * att['best_in_sample']['precision']:.1f}% "
            f"at coverage {100 * att['best_in_sample']['coverage']:.1f}%.") + "\n")  # fmt: skip
        rows = []
        for sname in SLICES:
            n = next((r for r in res["doc_level"] if r["target"] == t and r["slice"] == sname
                      and r["scheme"] == "nested"), None)  # fmt: skip
            i = next((r for r in res["doc_level"] if r["target"] == t and r["slice"] == sname
                      and r["scheme"] == "in_sample"), None)  # fmt: skip
            if n is None or i is None:
                continue
            in_s = f"{_ci(i['precision_accepted'])} / cov {_ci(i['accepted_share'])}"
            rows.append(
                [sname, n["n"], _ci(n["precision_accepted"]), _ci(n["accepted_share"]), in_s]
            )
        L += _tab(["slice", "docs", "nested precision", "nested coverage", "in-sample"], rows)
    L.append("\n## (d) Calibration (ECE 15 equal-width / equal-mass bins)\n")
    rows = [[c["slice"], c["group"], c["n"], f"{c['base_rate']:.3f}", f"{c['log_loss']:.3f}",
             f"{c['ece_width15']:.3f}", f"{c['ece_mass15']:.3f}"]
            for c in res["calibration"] if c.get("n")]  # fmt: skip
    L += _tab(["slice", "group", "n", "base rate", "log-loss", "ECE w15", "ECE m15"], rows)
    L.append("\nReliability tables and risk-coverage curves: `reliability.csv`, "
             "`risk_coverage.csv` in the run folder (not reproduced here).\n")  # fmt: skip
    L.append("\n## Caveats\n")
    L += [
        "- CIs resample documents but condition on the selected taus; tau-selection variance is "
        "not in them.",
        "- Model structure / kind were chosen on the same OOF predictions that are reported.",
        "- The doc-level model stacks on field-level OOF probabilities (mild second-level "
        "leakage).",
        "- Labels are post-rule; v1 published numbers are pre-rule, hence the matched variants.",
        "- Population of field metrics = emitted non-null fields; null fields are not scored "
        "here (no null policy; the row null policy was rejected).",
    ]
    return "\n".join(L) + "\n"


def honest_statement(res: Mapping[str, Any]) -> list[str]:
    """Generated bullets: achievable coverage per target, CI widths, dev size."""
    out: list[str] = []
    for sname in ("all", "dev"):
        out.append(f"- slice `{sname}` ({res['slice_docs'][sname]} docs), nested, per type:")
        for pop in (*c2.FIELD_TYPES, "all_per_type_tau"):
            parts = []
            for t in c2.TARGETS:
                r = next((x for x in res["tau"] if x["population"] == pop and x["slice"] == sname
                          and x["target"] == t and x["scheme"] == "nested"), None)  # fmt: skip
                if r is None:
                    continue
                prec, cov = r["precision_accepted"], r["accepted_share"]
                met = prec["point"] >= t
                parts.append(
                    f"{int(100 * t)}%: coverage {_ci(cov)}, precision {_ci(prec)} "
                    + ("(target met at point estimate)" if met else "(TARGET MISSED)")
                    + f", CI width cov {100 * (cov['hi'] - cov['lo']):.1f} pts"
                )
            out.append(f"  - {pop}: " + "; ".join(parts))
    out.append(
        "- Document level: "
        + "; ".join(
            f"{int(100 * float(t))}% "
            + ("attained" if a["attained"] else "NOT ATTAINABLE on this evidence")
            for t, a in res["doc_attainability"].items()
        )
        + "."
    )
    out.append("- The dev slice has only 100 docs: its CIs are wide and it is not a separate test.")
    return out


# ---------------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------------


def run(a: argparse.Namespace) -> int:
    """Real run on the 02 outputs."""
    t0 = time.time()
    ids = json.loads((ROOT / "splits" / "zeroshot500.json").read_text(encoding="utf-8"))
    c2.assert_no_test_ids(ids)
    try:
        preds, traces = v1cal.load_run(a.run_dir, ids)
    except v1cal.RunNotFullError as exc:
        print(f"FAIL (closed): {exc}", file=sys.stderr)
        return 2
    n_lp = sum(1 for t in traces.values() if any(p.get("field_logprobs") for p in t["pages"]))
    if n_lp < len(ids) and a.require_logprobs:
        print(f"FAIL (closed): only {n_lp}/{len(ids)} docs carry field_logprobs", file=sys.stderr)
        return 2
    folds = json.loads((ROOT / "splits" / "folds.json").read_text(encoding="utf-8"))
    gold, meta = v1cal.load_corpus(ids)
    doc_fold = v1cal.restricted_folds(ids, folds, meta)
    groups = {d: meta[d]["supplier_group"] for d in ids}
    labels = meta_mod.load_labels("train") + meta_mod.load_labels("dev")
    shapes_fn, shapes_equal = fold_shape_fn(labels, doc_fold, groups)
    tr_list = [traces[d] for d in ids]
    off_raw = postrules.postprocess_traces(tr_list, RuleConfig.all_off(), None, None)[0]
    off_preds, _ = repair_predictions(coerce_predictions(dict(off_raw)))
    assert off_preds == {d: preds[d] for d in ids}, "all rules OFF does not reproduce predictions"
    post, changes, summary = post_rule_predictions(tr_list, shapes_fn, a.ocr_cache)
    print(f"rules summary: {json.dumps(summary)}", file=sys.stderr, flush=True)
    ocr = ev.load_ocr_pages(list(ids))
    rows: list[cf.FieldRow] = []
    extras: list[np.ndarray] = []
    sigs: dict[str, np.ndarray | None] = {}
    for n, d in enumerate(ids, 1):
        pages = ocr.get(d) or []
        r = cf.build_doc_features(d, post[d], traces[d], pages)
        rows.extend(r)
        extras.append(c2.build_extras(r, post[d], traces[d], pages, changes.get(d, [])))
        sigs[d] = layout_signature(pages[0]) if pages else None
        if n % 100 == 0:
            print(f"  features {n}/{len(ids)} docs", file=sys.stderr, flush=True)
    tv = c2.TableV2(cf.assemble(rows), np.vstack(extras))
    lab = {d: cf.label_doc(post[d], gold[d]) for d in ids}
    y_correct, y_null = cf.label_arrays(tv.keys, lab)
    sc = ev.load_scorer()
    y_doc = np.array([c2.doc_exact(sc, post[d], gold[d]) for d in ids])
    nov = cf.compute_novelty_table(sigs, doc_fold)
    v1_path = paths.runs_dir() / "calibration" / a.run_dir.name / "oof_fields.csv"
    res = analyze(tv, y_correct, ids, y_doc, doc_fold, groups, meta, nov, v1_path, a.n_boot)
    res["docs_with_logprobs"] = n_lp
    res["rules_summary"] = summary
    out_dir = a.out_dir or paths.runs_dir() / "calibration_v2" / a.run_dir.name
    prov = {
        "run": a.run_dir.name,
        "head": git_head(),
        "sha_module": sha256_file(ROOT / "src" / "shipdoc" / "confidence_v2.py"),
        "sha_script": sha256_file(Path(__file__)),
        "command": "uv run python scripts/calibrate_v2.py --run-dir <02 run> --require-logprobs",
        "wall_s": time.time() - t0,
        "shapes_equal": shapes_equal,
        "rules_summary": summary,
        "honest": honest_statement(res),
    }
    res["provenance"] = {k: v for k, v in prov.items() if k != "honest"}
    md = render_md(res, prov)
    write_outputs(res, tv, y_correct, y_null, doc_fold, meta, out_dir)
    (out_dir / "calibration_v2.md").write_text(md, encoding="utf-8")
    if a.report_md:
        a.report_md.write_text(md, encoding="utf-8")
    print(f"outputs in {out_dir}; wall {time.time() - t0:.0f} s")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--ocr-cache", type=Path, default=None)
    ap.add_argument("--report-md", type=Path, default=None)
    ap.add_argument("--n-boot", type=int, default=c2.N_BOOT)
    ap.add_argument("--require-logprobs", action="store_true")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
