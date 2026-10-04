"""Phase 5 confidence and calibration CLI (CPU only, no model, no network).

Real mode (needs the finished full zero-shot run, 500 docs = ``splits/zeroshot500.json``)::

    uv run python scripts/calibrate.py --run-dir $SHIPDOC_RUNS_DIR/zeroshot500/<run folder>

Fails closed when the run is not exactly those 500 docs. Consumes ``predictions.json`` and
``trace.jsonl`` (``field_logprobs`` when the run captured them), the OCR cache, gold labels and
``splits/folds.json``. Writes aggregates and a per-field OOF probability table (doc id, field name,
probabilities, no values) to ``--out-dir`` (default ``<runs>/calibration/<run folder>``).

Dry run (``--dry-run``): every output carries a DRY RUN label and is not a result.
(i) synthetic features with a known generative truth (mechanics of calibrator / ECE / tau / null
policy); (ii) the saved dev100 outputs when found locally (OCR + validator + cross-page + layout
features only, no logprobs). ``--report-md reports/calibration_dryrun.md`` writes the aggregate-only
markdown summary.

Definitions (spec Phase 5; see ``shipdoc.confidence``):

* P(correct) calibrator on emitted non-null fields, P(gold null) calibrator on all fields, both
  cross-fitted by supplier fold (fit on 2 folds, predict the third).
* Null policy: null iff P(gold null) > P(correct); only ever removes a value.
* Review flag: tau = lowest threshold whose auto-accepted set ({P(correct) >= tau}) has precision
  >= 0.98 on the out-of-fold TRAIN docs; reported on the OOF dev docs (a held-out slice for tau)
  and, labelled in-sample, with tau chosen on all docs. "Dev slice" = the dev docs' OUT-OF-FOLD
  predictions (each dev doc is held out with its supplier fold); the seen-layout dev evaluation
  needs the final fine-tuned model, which does not exist yet.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from shipdoc import confidence as cf
from shipdoc import eval as ev
from shipdoc import paths
from shipdoc.cluster import layout_signature
from shipdoc.replay import read_trace

ROOT = Path(__file__).resolve().parents[1]
DRY = "DRY RUN"
DRY_BANNER = (
    "DRY RUN: not a result. Synthetic data and/or saved dev100 outputs used to exercise the "
    "mechanics; no number here may be quoted as a Phase 5 result."
)
DEV100_NAME = "dev100_qwen35_4b_img_only_2abf481"
DEV100_SEARCH = (
    Path("D:/shipdoc/runs/spike_download2/x") / DEV100_NAME,
    Path("D:/shipdoc/runs/spike_download/x") / DEV100_NAME,
)


class RunNotFullError(RuntimeError):
    """The run folder does not cover exactly the expected document set."""


# ---------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------


def load_run(
    run_dir: Path, expected_ids: Sequence[str]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """``(predictions, {doc_id: trace line})`` of a run; raises `RunNotFullError` unless both
    cover exactly `expected_ids` (fail closed: a partial run would calibrate on a biased subset)."""
    run_dir = Path(run_dir)
    preds = json.loads((run_dir / "predictions.json").read_text(encoding="utf-8"))
    traces = {t["doc_id"]: t for t in read_trace(run_dir / "trace.jsonl")}
    want = set(expected_ids)
    for name, have in (("predictions.json", set(preds)), ("trace.jsonl", set(traces))):
        if have != want:
            raise RunNotFullError(
                f"{run_dir.name}: {name} has {len(have)} docs, expected exactly {len(want)} "
                f"({len(want - have)} missing, {len(have - want)} unexpected); "
                "refusing to calibrate on a partial or foreign run"
            )
    return preds, traces


def load_corpus(
    doc_ids: Sequence[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Gold labels and meta rows (by doc id) of train+dev docs."""
    gold: dict[str, dict[str, Any]] = {}
    meta: dict[str, dict[str, Any]] = {}
    for split in ("train", "dev"):
        gold.update(ev.load_gold(paths.data_dir() / split / "labels"))
        for m in json.loads((ROOT / "meta" / f"{split}.json").read_text(encoding="utf-8")):
            meta[m["doc_id"]] = m
    missing = [d for d in doc_ids if d not in gold or d not in meta]
    if missing:
        raise RuntimeError(f"{len(missing)} docs lack gold or meta, e.g. {missing[:3]}")
    return {d: gold[d] for d in doc_ids}, {d: meta[d] for d in doc_ids}


def restricted_folds(
    doc_ids: Sequence[str], folds: Mapping[str, Any], meta: Mapping[str, Mapping[str, Any]]
) -> dict[str, int]:
    """``doc_fold`` of `doc_ids` from ``folds.json``; if fewer than 2 folds are populated (a tiny
    subset), fall back to supplier-grouped 3-fold CV over the groups present (never splits a
    supplier group)."""
    df = {d: int(folds["doc_fold"][d]) for d in doc_ids}
    if len(set(df.values())) >= 2:
        return df
    from sklearn.model_selection import GroupKFold

    groups = [meta[d]["supplier_group"] for d in doc_ids]
    k = min(3, len(set(groups)))
    out: dict[str, int] = {}
    for i, (_, va) in enumerate(GroupKFold(n_splits=k).split(doc_ids, groups=groups)):
        out.update({doc_ids[j]: i for j in va})
    return out


# ---------------------------------------------------------------------------------------------
# Real / dev100 pipeline
# ---------------------------------------------------------------------------------------------


def build_table(
    doc_ids: Sequence[str],
    preds: Mapping[str, Any],
    traces: Mapping[str, Any],
    ocr: Mapping[str, list[Any]],
) -> tuple[cf.FieldTable, dict[str, np.ndarray | None]]:
    """Feature table of every doc (no gold) and the page-1 layout signatures."""
    rows: list[cf.FieldRow] = []
    sigs: dict[str, np.ndarray | None] = {}
    for n, d in enumerate(doc_ids, 1):
        pages = ocr.get(d) or []
        rows.extend(cf.build_doc_features(d, preds[d], traces[d], pages))
        sigs[d] = layout_signature(pages[0]) if pages else None
        if n % 100 == 0:
            print(f"  features {n}/{len(doc_ids)} docs", file=sys.stderr, flush=True)
    return cf.assemble(rows), sigs


def _slice_masks(
    table: cf.FieldTable, meta: Mapping[str, Mapping[str, Any]] | None
) -> dict[str, np.ndarray]:
    ids = table.doc_ids
    masks: dict[str, np.ndarray] = {
        "all": np.ones(len(ids), dtype=bool),
        "dev_oof": np.array([str(d).startswith("dev_") for d in ids]),
        "train_oof": np.array([str(d).startswith("train_") for d in ids]),
        "scanned": table.X[:, cf.BASE_FEATURES.index("scanned")] == 1.0,
        "digital": table.X[:, cf.BASE_FEATURES.index("scanned")] == 0.0,
    }
    if meta is not None:
        wb = np.array([bool(meta[d].get("waybill")) for d in ids])
        masks["invoice_only"] = ~wb
        masks["waybill_only"] = wb
    return {k: v for k, v in masks.items() if v.any()}


def _calib_block(
    p: np.ndarray, y: np.ndarray, pop: np.ndarray, masks: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, m in masks.items():
        sel = m & pop
        out[name] = cf.calibration_summary(p[sel], y[sel])
    return out


def _review_block(
    conf: np.ndarray,
    correct: np.ndarray,
    doc_ids: np.ndarray,
    pop: np.ndarray,
    masks: Mapping[str, np.ndarray],
    n_boot: int,
) -> dict[str, Any]:
    """tau on OOF train docs, evaluated on OOF dev docs (held-out for tau); plus in-sample."""
    out: dict[str, Any] = {
        "target_precision": cf.PRECISION_TARGET,
        "population_fields": int(pop.sum()),
    }
    schemes: dict[str, np.ndarray] = {"in_sample_all": masks["all"] & pop}
    if "train_oof" in masks and "dev_oof" in masks:
        schemes["train_tau_dev_eval"] = masks["train_oof"] & pop
    for name, sel_mask in schemes.items():
        tau = cf.select_tau(conf[sel_mask], correct[sel_mask])
        block: dict[str, Any] = {
            "tau": tau,
            "tau_note": "None = target unattainable, every field flagged",
            "tau_selected_on_fields": int(sel_mask.sum()),
        }
        evals = {"dev_oof": masks.get("dev_oof")} if name == "train_tau_dev_eval" else dict(masks)
        for ename, em in evals.items():
            sel = (em & pop) if em is not None else None
            if sel is None or not sel.any():
                continue
            block[f"eval_{ename}"] = {
                "fields": int(sel.sum()),
                "docs": len(set(doc_ids[sel].tolist())),
                **cf.review_with_ci(conf[sel], correct[sel], doc_ids[sel], tau, n_boot),
            }
        out[name] = block
    return out


def _field_ablation(
    y_correct: np.ndarray, y_null: np.ndarray, nulled: np.ndarray, emitted: np.ndarray
) -> dict[str, Any]:
    """Field-level effect of the null policy (counts; the doc-level OVERALL delta is separate)."""
    final = np.where(nulled, y_null, y_correct)
    return {
        "emitted_fields": int(emitted.sum()),
        "nulled_by_policy": int(nulled.sum()),
        "nulled_that_were_false_fills_fixed": int((nulled & y_null).sum()),
        "nulled_that_were_correct_values_lost": int((nulled & y_correct).sum()),
        "nulled_that_were_wrong_not_null_gold": int((nulled & ~y_null & ~y_correct).sum()),
        "field_accuracy_raw": float(y_correct.mean()),
        "field_accuracy_policy": float(final.mean()),
    }


def _doc_ablation(
    raw: Mapping[str, Any],
    keys: Sequence[cf.FieldKey],
    nulled: np.ndarray,
    gold: Mapping[str, Any],
    meta: Mapping[str, Mapping[str, Any]],
    n_boot: int,
) -> dict[str, Any]:
    """Paired doc-level bootstrap of OVERALL (raw nulls vs null policy) plus false-fill split."""
    out: dict[str, Any] = {}
    for scope_name, scopes in (
        ("header_and_rows", ("header", "row")),
        ("header_only", ("header",)),
    ):
        pol = cf.apply_null_policy(raw, keys, nulled, scopes)
        slices = {"all": list(gold), "dev_oof": [d for d in gold if d.startswith("dev_")]}
        block: dict[str, Any] = {}
        for sname, ids in slices.items():
            if not ids:
                continue
            g = {d: gold[d] for d in ids}
            r = ev.paired_bootstrap(
                {d: raw[d] for d in ids}, {d: pol[d] for d in ids}, g, n=n_boot, seed=cf.SEED
            )
            block[sname] = {
                "docs": len(ids),
                "OVERALL": r["OVERALL"],
                "false_fill_rate": r["false_fill_rate"],
                "false_fills_raw": cf.false_fill_split({d: raw[d] for d in ids}, g, meta),
                "false_fills_policy": cf.false_fill_split({d: pol[d] for d in ids}, g, meta),
            }
        out[scope_name] = block
    return out


def analyze(
    table: cf.FieldTable,
    y_correct: np.ndarray,
    y_null: np.ndarray,
    doc_fold: Mapping[str, int],
    novelty_table: Mapping[int, Mapping[str, float]] | None,
    groups: Mapping[str, str] | None,
    meta: Mapping[str, Mapping[str, Any]] | None,
    n_boot: int,
    final_correct_fn: Any = None,
) -> tuple[dict[str, Any], cf.OofResult, np.ndarray]:
    """Calibrators, calibration metrics, null policy and review flag from labelled features.

    `final_correct_fn(nulled) -> bool array` gives field correctness of the post-policy output
    (real runs re-score the policy documents); default is the field-level identity
    ``where(nulled, y_null, y_correct)``. Returns (results, OOF probabilities, nulled mask).
    """
    oof = cf.calibrate_oof(table, y_correct, y_null, doc_fold, novelty_table, groups)
    em = table.emitted
    masks = _slice_masks(table, meta)
    nulled = cf.null_decisions(oof.p_correct, oof.p_null, em)
    final_correct = (
        final_correct_fn(nulled) if final_correct_fn else np.where(nulled, y_null, y_correct)
    )
    conf = np.where(em & ~nulled, oof.p_correct, oof.p_null)
    final_nonnull = em & ~nulled
    ids = table.doc_ids
    res: dict[str, Any] = {
        "rows": int(len(em)),
        "docs": len(set(ids.tolist())),
        "emitted_rows": int(em.sum()),
        "features": list(cf.DESIGN_FEATURES),
        "rows_with_logprobs": int(
            (table.X[:, cf.BASE_FEATURES.index("has_logprobs")] == 1.0).sum()
        ),
        "model_selection": {
            "p_correct": {"chosen": oof.sel_correct.kind, "log_loss": oof.sel_correct.logloss},
            "p_null": {"chosen": oof.sel_null.kind, "log_loss": oof.sel_null.logloss},
            "gbm_margin_nats": cf.GBM_MARGIN,
            "note": "chosen on the same OOF predictions that are reported (mild optimism)",
        },
        "calibration_p_correct_emitted": _calib_block(oof.p_correct, y_correct, em, masks),
        "calibration_p_null_all_fields": _calib_block(oof.p_null, y_null, np.ones_like(em), masks),
        "reliability_p_correct_emitted": cf.reliability(oof.p_correct[em], y_correct[em]),
        "reliability_p_null_all": cf.reliability(oof.p_null, y_null),
        "coverage_curve_p_correct_emitted": cf.coverage_curve(oof.p_correct[em], y_correct[em]),
        "null_policy_field_level": _field_ablation(y_correct, y_null, nulled, em),
        "review": {
            "final_nonnull_fields": _review_block(
                conf, final_correct, ids, final_nonnull, masks, n_boot
            ),
            "all_final_fields_nulls_use_p_null": _review_block(
                conf, final_correct, ids, np.ones_like(em), masks, n_boot
            ),
        },
    }
    return res, oof, nulled


def run_real(
    doc_ids: Sequence[str],
    preds: Mapping[str, Any],
    traces: Mapping[str, Any],
    gold: Mapping[str, Any],
    meta: Mapping[str, Mapping[str, Any]],
    folds: Mapping[str, Any],
    n_boot: int,
) -> dict[str, Any]:
    """Whole Phase 5 evaluation of one finished run (also used for the dev100 dry run)."""
    ocr = ev.load_ocr_pages(list(doc_ids))
    table, sigs = build_table(doc_ids, preds, traces, ocr)
    labels = {d: cf.label_doc(preds[d], gold[d]) for d in doc_ids}
    y_correct, y_null = cf.label_arrays(table.keys, labels)
    doc_fold = restricted_folds(doc_ids, folds, meta)
    groups = {d: meta[d]["supplier_group"] for d in doc_ids}
    nov = cf.compute_novelty_table(sigs, doc_fold)

    def final_correct(nulled: np.ndarray) -> np.ndarray:
        pol = cf.apply_null_policy(preds, table.keys, nulled)
        fl = {d: cf.label_doc(pol[d], gold[d]) for d in doc_ids}
        return cf.label_arrays(table.keys, fl)[0]

    res, oof, nulled = analyze(
        table, y_correct, y_null, doc_fold, nov, groups, meta, n_boot, final_correct
    )
    res["docs_without_ocr"] = sum(1 for d in doc_ids if not ocr.get(d))
    res["doc_type_mismatches"] = sum(
        preds[d].get("doc_type") != gold[d]["doc_type"] for d in doc_ids
    )
    res["null_policy_doc_level"] = _doc_ablation(
        {d: preds[d] for d in doc_ids}, table.keys, nulled, gold, meta, n_boot
    )
    res["_arrays"] = {
        "table": table,
        "oof": oof,
        "nulled": nulled,
        "y_correct": y_correct,
        "y_null": y_null,
    }
    return res


# ---------------------------------------------------------------------------------------------
# Synthetic dry run
# ---------------------------------------------------------------------------------------------


def run_synthetic(n_boot: int) -> dict[str, Any]:
    """Mechanics check on seeded synthetic data whose true probabilities are known."""
    syn = cf.synthetic_data()
    res, oof, nulled = analyze(
        syn.table, syn.y_correct, syn.y_null, syn.doc_fold, None, syn.groups, None, n_boot
    )
    em = syn.table.emitted
    rng_err = np.abs(oof.p_correct[em] - syn.true_p_correct[em])
    res["recovery_vs_known_truth"] = {
        "mean_abs_err_p_correct": float(rng_err.mean()),
        "mean_abs_err_p_null": float(np.abs(oof.p_null - syn.true_p_null).mean()),
        "corr_p_correct": float(np.corrcoef(oof.p_correct[em], syn.true_p_correct[em])[0, 1]),
    }
    res["_arrays"] = {
        "table": syn.table,
        "oof": oof,
        "nulled": nulled,
        "y_correct": syn.y_correct,
        "y_null": syn.y_null,
    }
    return res


# ---------------------------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------------------------


def _csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def write_outputs(res: dict[str, Any], out_dir: Path, prefix: str, label: str | None) -> None:
    """JSON aggregates, reliability / coverage CSVs, per-field OOF table, optional plots."""
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = res.pop("_arrays")
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        have_plt = False
        res["plots"] = "matplotlib not installed: reliability / coverage data are the CSVs only"
    else:
        have_plt = True
    if label:
        res = {"label": label, **res}
    (out_dir / f"{prefix}calibration.json").write_text(
        json.dumps(res, indent=1, default=float), encoding="utf-8"
    )
    for name in ("reliability_p_correct_emitted", "reliability_p_null_all"):
        _csv(
            out_dir / f"{prefix}{name}.csv",
            ["lo", "hi", "n", "mean_conf", "accuracy"],
            [[r["lo"], r["hi"], r["n"], r["mean_conf"], r["accuracy"]] for r in res[name]],
        )
    cov = res["coverage_curve_p_correct_emitted"]
    _csv(
        out_dir / f"{prefix}coverage_curve.csv",
        ["coverage", "accuracy", "min_conf"],
        [[r["coverage"], r["accuracy"], r["min_conf"]] for r in cov],
    )
    t, oof = arrays["table"], arrays["oof"]
    _csv(
        out_dir / f"{prefix}oof_fields.csv",
        ["doc_id", "scope", "field", "row_idx", "emitted", "p_correct", "p_null", "nulled",
         "y_correct", "y_null"],
        [
            [*k, int(t.emitted[i]), oof.p_correct[i], oof.p_null[i], int(arrays["nulled"][i]),
             int(arrays["y_correct"][i]), int(arrays["y_null"][i])]
            for i, k in enumerate(t.keys)
        ],
    )  # fmt: skip
    if have_plt:
        _plot(res, out_dir, prefix, label)


def _plot(res: Mapping[str, Any], out_dir: Path, prefix: str, label: str | None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    r = [x for x in res["reliability_p_correct_emitted"] if x["n"]]
    ax[0].plot([0, 1], [0, 1], "k--")
    ax[0].plot([x["mean_conf"] for x in r], [x["accuracy"] for x in r], "o-")
    ax[0].set(xlabel="mean P(correct)", ylabel="observed accuracy", title="reliability")
    c = res["coverage_curve_p_correct_emitted"]
    ax[1].plot([x["coverage"] for x in c], [x["accuracy"] for x in c])
    ax[1].set(xlabel="coverage", ylabel="accuracy", title="coverage vs accuracy")
    if label:
        fig.suptitle(label, color="red")
    fig.savefig(out_dir / f"{prefix}reliability_coverage.png", dpi=110)
    plt.close(fig)


def _f(x: Any, pct: bool = True) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{100 * x:.2f}%" if pct else f"{x:.4f}"


def _ci(d: Mapping[str, float]) -> str:
    return f"{_f(d['point'])} [{_f(d['lo'])}, {_f(d['hi'])}]"


def render_md(sections: Mapping[str, dict[str, Any]], title: str, banner: str | None) -> str:
    """Aggregate-only markdown (no values, no names, no doc ids)."""
    L = [f"# {title}\n"]
    if banner:
        L += [f"> **{banner}**\n"]
    for sname, res in sections.items():
        L += [f"\n## {sname}\n"]
        if banner:
            L += [f"*{DRY}*\n"]
        L += [
            f"rows {res['rows']} (emitted non-null {res['emitted_rows']}), docs {res['docs']}, "
            f"rows with logprobs {res['rows_with_logprobs']}\n",
            f"calibrators chosen: P(correct) = {res['model_selection']['p_correct']['chosen']} "
            f"(log-loss {res['model_selection']['p_correct']['log_loss']}), P(null) = "
            f"{res['model_selection']['p_null']['chosen']} "
            f"(log-loss {res['model_selection']['p_null']['log_loss']})\n",
        ]
        for key, head in (
            ("calibration_p_correct_emitted", "P(correct | emitted), OOF"),
            ("calibration_p_null_all_fields", "P(gold null), all fields, OOF"),
        ):
            L += [
                f"\n**{head}**\n",
                "| slice | n | base rate | log-loss | Brier | ECE w15 | ECE m15 |",
                "|---|---|---|---|---|---|---|",
            ]
            for s, v in res[key].items():
                if v.get("n"):
                    L.append(
                        f"| {s} | {v['n']} | {_f(v['base_rate'])} | {_f(v['log_loss'], False)} | "
                        f"{_f(v['brier'], False)} | {_f(v['ece_width15'], False)} | "
                        f"{_f(v['ece_mass15'], False)} |"
                    )
        npf = res["null_policy_field_level"]
        L += ["\n**Null policy, field level**\n", "```", json.dumps(npf, indent=1), "```"]
        for scope, blocks in res.get("null_policy_doc_level", {}).items():
            for sl, b in blocks.items():
                o = b["OVERALL"]
                L.append(
                    f"\nnull policy `{scope}` on `{sl}` ({b['docs']} docs): OVERALL "
                    f"{_f(o['a'], False)} -> {_f(o['b'], False)}, delta {o['delta']:+.4f} "
                    f"[{o['lo']:+.4f}, {o['hi']:+.4f}] (paired doc bootstrap, 2000 resamples, "
                    "seed 42); "
                    f"false fills raw {json.dumps(b['false_fills_raw'])} -> policy "
                    f"{json.dumps(b['false_fills_policy'])}"
                )
        for rname, rb in res["review"].items():
            L += [
                f"\n**Review flag, population `{rname}`** "
                f"(target precision {rb['target_precision']:.2f})\n"
            ]
            for scheme, blk in rb.items():
                if not isinstance(blk, dict) or "tau" not in blk:
                    continue
                L.append(
                    f"\n- `{scheme}`: tau = {blk['tau']} "
                    f"(selected on {blk['tau_selected_on_fields']} fields)"
                )
                for k, v in blk.items():
                    if k.startswith("eval_"):
                        L.append(
                            f"  - {k[5:]} ({v['fields']} fields, {v['docs']} docs): "
                            + ", ".join(f"{m} {_ci(v[m])}" for m in cf.REVIEW_METRICS)
                        )
        if "recovery_vs_known_truth" in res:
            L += ["\n**Recovery of the known generative truth (synthetic only)**\n", "```",
                  json.dumps(res["recovery_vs_known_truth"], indent=1), "```"]  # fmt: skip
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


def find_dev100(explicit: Path | None) -> Path | None:
    """The saved dev100 run folder, or None if absent locally."""
    for p in ([explicit] if explicit else []) + list(DEV100_SEARCH):
        if p is not None and (p / "predictions.json").is_file() and (p / "trace.jsonl").is_file():
            return p
    return None


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", type=Path, help="full 500-doc zero-shot run folder")
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="synthetic + dev100 mechanics check")
    ap.add_argument("--dev100-dir", type=Path, help="dry run: dev100 run folder (default: search)")
    ap.add_argument("--report-md", type=Path, help="dry run: write the aggregate markdown here")
    ap.add_argument("--n-boot", type=int, default=cf.N_BOOT)
    ap.add_argument("--require-logprobs", action="store_true")
    a = ap.parse_args(argv)
    folds = json.loads((ROOT / "splits" / "folds.json").read_text(encoding="utf-8"))
    if a.dry_run:
        out_dir = a.out_dir or paths.tmp_dir() / "calibration_dryrun"
        sections = {"(i) synthetic features, known generative truth": run_synthetic(a.n_boot)}
        write_outputs(
            sections["(i) synthetic features, known generative truth"],
            out_dir,
            "DRYRUN_synthetic_",
            DRY_BANNER,
        )
        d100 = find_dev100(a.dev100_dir)
        if d100 is None:
            print("dev100 saved outputs not found locally; dry run (ii) skipped", file=sys.stderr)
        else:
            ids = json.loads((ROOT / "splits" / "dev100.json").read_text(encoding="utf-8"))
            preds, traces = load_run(d100, ids)
            gold, meta = load_corpus(ids)
            name = "(ii) dev100 saved outputs, no logprobs"
            sections[name] = run_real(ids, preds, traces, gold, meta, folds, a.n_boot)
            write_outputs(sections[name], out_dir, "DRYRUN_dev100_", DRY_BANNER)
        if a.report_md:
            a.report_md.write_text(
                render_md(sections, "Calibration dry run (DRY RUN, not results)", DRY_BANNER),
                encoding="utf-8",
            )
        print(f"{DRY}: outputs in {out_dir}")
        return 0
    if a.run_dir is None:
        ap.error("--run-dir is required unless --dry-run")
    ids = json.loads((ROOT / "splits" / "zeroshot500.json").read_text(encoding="utf-8"))
    try:
        preds, traces = load_run(a.run_dir, ids)
    except RunNotFullError as exc:
        print(f"FAIL (closed): {exc}", file=sys.stderr)
        return 2
    n_lp = sum(1 for t in traces.values() if any(p.get("field_logprobs") for p in t["pages"]))
    if n_lp < len(ids):
        msg = f"only {n_lp}/{len(ids)} docs carry field_logprobs (run without --logprobs?)"
        if a.require_logprobs:
            print(f"FAIL (closed): {msg}", file=sys.stderr)
            return 2
        print(f"WARNING: {msg}; logprob features are missing for the rest", file=sys.stderr)
    gold, meta = load_corpus(ids)
    res = run_real(ids, preds, traces, gold, meta, folds, a.n_boot)
    res["docs_with_logprobs"] = n_lp
    res["run_dir_name"] = a.run_dir.name
    out_dir = a.out_dir or paths.runs_dir() / "calibration" / a.run_dir.name
    sections = {f"zero-shot run {a.run_dir.name}": res}
    write_outputs(res, out_dir, "", None)
    (out_dir / "calibration.md").write_text(
        render_md(sections, "Phase 5 calibration (UNVERIFIED until recomputed)", None),
        encoding="utf-8",
    )
    print(f"outputs in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
