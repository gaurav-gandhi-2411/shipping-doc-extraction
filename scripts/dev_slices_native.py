"""Dev-slice table of the native ZS + rules system (v1.5) on the 100 dev docs (CPU only, no model).

Scores the native zero-shot run (``predictions.json``, rules OFF) and the same run through the
production rule path (``replay_v1_check.production``, R1 + R2 + R3, HONEST supplier-held-out R3
shapes) on the dev documents with the unmodified scorer, per slice of the doc meta tags (scanned,
multipage, repeated parts, illegible = gold redaction, awb / hawb absent = gold absent line, doc
type). Each slice carries n, the OVERALL point with a doc-level bootstrap 95% CI (2000 resamples,
seed 42), and the paired delta (rules minus raw). Header false fills are split by why the gold is
null (``confidence.false_fill_split``: redaction vs absent line), on dev and on all 500 docs.

The dev set holds SEEN layouts (its supplier groups also occur in train): this is NOT an
unseen-supplier estimate. Aggregates and counts only: no gold or predicted value, no document id
is printed or written. All numbers are UNVERIFIED until a verifier recomputes them.

Run: ``uv run python scripts/dev_slices_native.py --run-dir <native ZS run> --out
reports/dev_slices_native.md --json-out $SHIPDOC_TMP_DIR/report_out/dev_slices_native.json``
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay_v1_check as rv  # noqa: E402  (sibling script, not a package)
import report_inputs as ri  # noqa: E402  (for the Wilson interval)

from shipdoc import eval as ev  # noqa: E402
from shipdoc import meta as meta_mod  # noqa: E402
from shipdoc import paths  # noqa: E402
from shipdoc.confidence import false_fill_split  # noqa: E402
from shipdoc.postrules import RuleConfig  # noqa: E402
from shipdoc.replay import read_trace  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
N_BOOT = 2000
SEED = 42
# slice name -> label; names are those of ``eval.slice_doc_ids`` plus the derived ones below
SLICES: tuple[tuple[str, str], ...] = (
    ("all", "all dev"),
    ("invoices", "invoices"),
    ("waybills", "waybills"),
    ("scanned=yes", "scanned"),
    ("scanned=no", "digital"),
    ("multipage=yes", "multipage"),
    ("multipage=no", "single page"),
    ("repeated_parts=yes", "repeated part numbers"),
    ("illegible=yes", "illegible: gold redaction (a required header field is null)"),
    ("absent_line=yes", "absent line: awb (invoice) or hawb (waybill) not printed"),
    (
        "illegible_any=yes",
        "illegible, scorer sense: any gold header null (redaction or absent line)",
    ),
)
# supplier-held-out view of the 500 docs: each fold's suppliers are absent from the R3 shape fit
FOLD_SLICES: tuple[tuple[str, str], ...] = (
    ("fold=0", "OOF fold 0 (supplier-held-out)"),
    ("fold=1", "OOF fold 1 (supplier-held-out)"),
    ("fold=2", "OOF fold 2 (supplier-held-out)"),
)
MIN_N_FOR_CI = 10  # below this a doc-level bootstrap CI says little; the cell still prints


def derived_slices(
    ids: Sequence[str],
    meta: Mapping[str, Mapping[str, Any]],
    base: dict[str, list[str]],
    doc_fold: Mapping[str, int] | None = None,
) -> dict[str, list[str]]:
    """``eval.slice_doc_ids`` plus ``absent_line=yes`` (awb_absent or hawb_absent),
    ``illegible_any=yes`` (redaction or absent line: the scorer's "any gold header null") and,
    when `doc_fold` is given, ``fold=<k>`` (the supplier-held-out validation folds)."""
    out = dict(base)
    out["absent_line=yes"] = [
        d for d in ids if meta[d].get("awb_absent") or meta[d].get("hawb_absent")
    ]
    out["illegible_any=yes"] = [
        d
        for d in ids
        if meta[d].get("illegible") or meta[d].get("awb_absent") or meta[d].get("hawb_absent")
    ]
    if doc_fold is not None:
        for k in sorted(set(doc_fold.values())):
            out[f"fold={k}"] = [d for d in ids if doc_fold.get(d) == k]
    return out


def slice_rows(
    raw: dict[str, Any],
    new: dict[str, Any],
    gold: dict[str, dict[str, Any]],
    slices: Mapping[str, Sequence[str]],
    n_boot: int,
    spec: Sequence[tuple[str, str]] = SLICES,
) -> list[dict[str, Any]]:
    """One row per slice: n, OVERALL (rules) with CI, OVERALL (raw), paired delta, other metrics."""
    rows = []
    for name, label in spec:
        ids = list(slices.get(name, []))
        if not ids:
            rows.append({"slice": name, "label": label, "n": 0})
            continue
        g = {d: gold[d] for d in ids}
        pr = {d: raw[d] for d in ids}
        pn = {d: new[d] for d in ids}
        per = ev.per_doc_results(pn, g)
        ci = ev.bootstrap_metrics([per[d] for d in ids], n=n_boot, seed=SEED)
        pb = ev.paired_bootstrap(pr, pn, g, n=n_boot, seed=SEED)["OVERALL"]
        rows.append(
            {
                "slice": name,
                "label": label,
                "n": len(ids),
                "overall": ci["OVERALL"],
                "header_acc": ci["header_field_accuracy"]["point"],
                "row_f1": ci["row_f1"]["point"],
                "fully_correct": ci["documents_fully_correct"]["point"],
                "false_fill": ci["false_fill_rate"]["point"],
                "raw_overall": pb["a"],
                "delta": {"delta": pb["delta"], "lo": pb["lo"], "hi": pb["hi"]},
            }
        )
    return rows


def false_fill_rows(
    raw: dict[str, Any],
    new: dict[str, Any],
    gold: dict[str, dict[str, Any]],
    meta: Mapping[str, Mapping[str, Any]],
    ids: Sequence[str],
) -> dict[str, Any]:
    """Header false fills by gold-null kind for raw and post-rule predictions of `ids`."""
    g = {d: gold[d] for d in ids}
    out: dict[str, Any] = {}
    for arm, preds in (("raw", raw), ("rules", new)):
        split = false_fill_split({d: preds[d] for d in ids}, g, meta)
        out[arm] = {
            kind: {
                "null_fields": v["null_fields"],
                "filled": v["filled"],
                "rate": ri.wilson(v["filled"], v["null_fields"]) if v["null_fields"] else None,
            }
            for kind, v in split.items()
        }
    return out


def compute(run_dir: Path, ocr_cache: Path | None, n_boot: int) -> dict[str, Any]:
    """All numbers of the report as one dict (aggregates only)."""
    traces = read_trace(run_dir / "trace.jsonl")
    pred_all = json.loads((run_dir / "predictions.json").read_text(encoding="utf-8"))
    folds = json.loads((ROOT / "splits" / "folds.json").read_text("utf-8"))["doc_fold"]
    doc_fold = {str(d): int(f) for d, f in folds.items()}
    gold: dict[str, Any] = {}
    for split in ("train", "dev"):
        gold |= ev.load_gold(paths.data_dir() / split / "labels")
    gold = {d: g for d, g in gold.items() if d in pred_all}
    if len(gold) != 500:
        raise SystemExit(f"expected 500 scored docs, found {len(gold)}")
    traces = [t for t in traces if t["doc_id"] in gold]
    groups = json.loads((ROOT / "meta" / "supplier_groups.json").read_text(encoding="utf-8"))
    labels = meta_mod.load_labels("train") + meta_mod.load_labels("dev")
    honest = rv.fold_shape_fn(labels, doc_fold, groups)
    raw = {d: pred_all[d] for d in gold}
    off, _ = rv.production(traces, RuleConfig.all_off(), None, ocr_cache)
    if off != raw:
        raise SystemExit("rules OFF does not reproduce predictions.json")
    new, summary = rv.production(traces, RuleConfig(), honest, ocr_cache)
    meta_list = ev.load_json(ROOT / "meta" / "train.json") + ev.load_json(
        ROOT / "meta" / "dev.json"
    )
    meta = {m["doc_id"]: m for m in meta_list}
    dev_ids = [d for d in gold if d.startswith("dev_")]
    gdev = {d: gold[d] for d in dev_ids}
    slices = derived_slices(dev_ids, meta, ev.slice_doc_ids(gdev, meta_list, group_slices=False))
    all_ids = list(gold)
    slices_all = derived_slices(
        all_ids, meta, ev.slice_doc_ids(gold, meta_list, group_slices=False), doc_fold
    )
    train_groups = {groups[d] for d in all_ids if d.startswith("train_")}
    dev_groups = {groups[d] for d in dev_ids}
    return {
        "run": run_dir.name,
        "n_docs_all": len(gold),
        "n_docs_dev": len(dev_ids),
        "dev_groups": len(dev_groups),
        "dev_groups_also_in_train": len(dev_groups & train_groups),
        "rules_summary": summary,
        "dev_slices": slice_rows(raw, new, gold, slices, n_boot),
        "all500_slices": slice_rows(raw, new, gold, slices_all, n_boot),
        "fold_slices": slice_rows(raw, new, gold, slices_all, n_boot, FOLD_SLICES),
        "all500_overall": ev.score(new, gold)["all"]["OVERALL"],
        "false_fill_dev": false_fill_rows(raw, new, gold, meta, dev_ids),
        "false_fill_all500": false_fill_rows(raw, new, gold, meta, list(gold)),
    }


def _ci(est: Mapping[str, float]) -> str:
    return f"{100 * est['point']:.2f} [{100 * est['lo']:.2f}, {100 * est['hi']:.2f}]"


def _dci(d: Mapping[str, float]) -> str:
    return f"{100 * d['delta']:+.2f} [{100 * d['lo']:+.2f}, {100 * d['hi']:+.2f}]"


SLICE_DEFINITIONS: tuple[str, ...] = (
    "- scanned: any page of the doc is a `.jpg` (`src/shipdoc/meta.py:108`); digital otherwise.",
    "- multipage: more than one page image (`meta.py:109`).",
    "- repeated part numbers: a supplier_part_number appears on >= 2 gold rows (`meta.py:75-85`).",
    "- invoice / waybill: the gold `doc_type` (`src/shipdoc/eval.py:109-113`).",
    "- illegible = redaction: >= 1 gold header null in a REQUIRED field, i.e. a field whose "
    "pooled null rate inside its doc type is < 10% (`meta.py:47-72`, `OPTIONAL_NULL_RATE` at "
    "`meta.py:20`; `reports/recon.md:125` confirmed the rule by eye on 13 docs). The tag is "
    "derived from gold null rates, NOT inspected per document: the per-document redaction vs "
    "absent assignment is rule-based (UNVERIFIED as a visual fact).",
    "- absent line: an invoice whose gold awb_number is null or a waybill whose gold hawb is "
    "null, i.e. the line is not printed (`meta.py:88-95`).",
    "- illegible, scorer sense: `score.py`'s `illegible_fields` metric counts EVERY gold header "
    "null, so its illegible set is redaction + absent line; the slice here is their union "
    "(`meta.py` tags; `reports/recon.md:173`).",
    "- false fills split the same way by `src/shipdoc/confidence.py:741-755` (`false_fill_split`).",
)


def _slice_table(
    rows: Sequence[Mapping[str, Any]], what: str, arm: str = "ZS + rules"
) -> list[str]:
    """Markdown rows of a slice table; an empty slice says so instead of printing n/a."""
    L = [
        f"| slice | n | OVERALL % [95% CI], {arm} | OVERALL %, raw | paired delta, rules - raw "
        "(pts) [95% CI] | header acc % | row F1 % | docs fully correct % | false-fill % |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        label = "all" if r["slice"] == "all" and what != "dev" else r["label"]
        if r["n"] == 0:
            L.append(f"| {label} | 0 {what} docs | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        tiny = f" (n < {MIN_N_FOR_CI}: CI uninformative)" if r["n"] < MIN_N_FOR_CI else ""
        L.append(
            f"| {label} | {r['n']}{tiny} | {_ci(r['overall'])} | "
            f"{100 * r['raw_overall']:.2f} | "
            f"{_dci(r['delta'])} | {100 * r['header_acc']:.2f} | {100 * r['row_f1']:.2f} | "
            f"{100 * r['fully_correct']:.2f} | {100 * r['false_fill']:.2f} |"
        )
    return L


def _overlap_note(rows: Sequence[Mapping[str, Any]], what: str) -> list[str]:
    """One line: docs that are both redaction and absent-line (redaction + absent - union)."""
    n = {r["slice"]: r["n"] for r in rows}
    keys = ("illegible=yes", "absent_line=yes", "illegible_any=yes")
    if not all(k in n for k in keys):
        return []
    both = n[keys[0]] + n[keys[1]] - n[keys[2]]
    return [
        "",
        f"Redaction and absent-line overlap in {both} {what} docs "
        f"({n[keys[0]]} + {n[keys[1]]} - {n[keys[2]]} in the union).",
    ]


def render(res: Mapping[str, Any], cmd: str, state: str) -> str:
    """Markdown report (aggregates only)."""
    L = [
        "# Native ZS + rules (v1.5) on the dev documents, by slice",
        "",
        f"**Provenance.** Run `{res['run']}`, command `{cmd}` (repo state `{state}`; paired and "
        f"plain bootstrap {N_BOOT} doc-level resamples, seed {SEED}; unmodified scorer via "
        "`shipdoc.eval`). Rules: production path `replay_v1_check.production` (R1 + R2 + R3) with "
        "the HONEST supplier-held-out R3 shapes; rules OFF reproduces `predictions.json` exactly "
        "(asserted). **All numbers UNVERIFIED** until a verifier recomputes them. Aggregates and "
        "counts only. The dev documents are SEEN layouts (their supplier groups also occur in "
        "train): this is not an unseen-supplier estimate. Slices with n < "
        f"{MIN_N_FOR_CI} are printed but their intervals are not informative.",
        "",
        f"Docs: {res['n_docs_dev']} dev of {res['n_docs_all']} scored; native ZS + rules OVERALL "
        f"on all {res['n_docs_all']}: {100 * res['all500_overall']:.2f}.",
        "",
        "## Slice definitions (the repo's own; file:line of this checkout)",
        "",
        *SLICE_DEFINITIONS,
        "",
        f"Dev suppliers: {res.get('dev_groups', 'n/a')} supplier groups, of which "
        f"{res.get('dev_groups_also_in_train', 'n/a')} also occur in train (`meta/"
        "supplier_groups.json`).",
        "",
        "## Dev100 (seen layouts), by slice",
        "",
        *_slice_table(res["dev_slices"], "dev"),
    ]
    L += _overlap_note(res["dev_slices"], "dev")
    if res.get("all500_slices"):
        L += [
            "",
            f"## All {res['n_docs_all']} train+dev docs (supplier-held-out R3 shapes), same slices",
            "",
            "Every document is scored with R3 shapes learned from the OTHER supplier folds; the "
            "zero-shot model never trained on any of them, so this is the unseen-supplier view "
            "of the same system (the honest shapes equal the all-gold shapes in 3 of 3 folds, "
            "`reports/calibration_v2_native.md`).",
            "",
            *_slice_table(res["all500_slices"], "500-doc"),
        ]
        L += _overlap_note(res["all500_slices"], "500-doc")
    if res.get("fold_slices"):
        L += [
            "",
            "## Seen (dev100) next to supplier-held-out (OOF) for the SAME system",
            "",
            "| set | n | OVERALL % [95% CI], ZS + rules |",
            "|---|---|---|",
        ]
        all_by = {r["slice"]: r for r in res.get("all500_slices", [])}
        dev_by = {r["slice"]: r for r in res["dev_slices"]}
        picks = [
            ("dev100 (seen layouts)", dev_by.get("all")),
            ("dev100 invoices (seen)", dev_by.get("invoices")),
            (f"{res['n_docs_all']}-doc OOF, supplier-held-out", all_by.get("all")),
            (f"{res['n_docs_all']}-doc OOF invoices, supplier-held-out", all_by.get("invoices")),
            *[(r["label"], r) for r in res["fold_slices"]],
        ]
        for label, r in picks:
            if r and r["n"]:
                L.append(f"| {label} | {r['n']} | {_ci(r['overall'])} |")
    L += [
        "",
        "## Header false fills by why the gold is null (Wilson 95%)",
        "",
        "`redaction` = every gold header null except the awb / hawb lines flagged absent in the "
        "meta tags; `absent_line` = invoice awb_number or waybill hawb null and flagged absent "
        "(`shipdoc.confidence.false_fill_split`).",
        "",
        "| set | arm | kind | gold-null header fields | filled | rate % [95% CI] |",
        "|---|---|---|---|---|---|",
    ]
    for key, label in (("false_fill_dev", "dev"), ("false_fill_all500", "all 500")):
        for arm in ("raw", "rules"):
            for kind in ("redaction", "absent_line"):
                v = res[key][arm][kind]
                rate = _ci(v["rate"]) if v["rate"] else "n/a"
                L.append(
                    f"| {label} | {arm} | {kind} | {v['null_fields']} | {v['filled']} | {rate} |"
                )
    return "\n".join(L) + "\n"


def git_state() -> str:
    """Short HEAD sha, ``+dirty`` when the tracked tree differs."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=ROOT, check=True,
        ).stdout.strip()  # fmt: skip
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, cwd=ROOT, check=True,
        ).stdout.strip()  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return sha + ("+dirty" if dirty else "")


# --------------------------------------------------------------------------------------------
# v2 (fine-tuned) slices: the 08 dev run, or (fallback, no 08) the pooled supplier-held-out OOF
# --------------------------------------------------------------------------------------------

V2_SCHEMA = "shipdoc-v2-slices/1"
# the kinds of v2 slice JSON; wandb_log_public and fill_v2_registry map each to its own entry:
# dev_run (08, final adapter, dev100, L4 fp16 inference), dev_oof (dev100 from the pooled OOF
# predictions: each dev doc predicted by a fold model that never saw its supplier: the stricter dev
# evidence), pooled_oof (all 500 docs, supplier-held-out; the contingency if 08 fails)
# The artifact-derived label of 08's numbers. The GPU name is NOT in any 08 artifact (only a Colab
# banner shows it), so "L4" is added by GG's paste-back in fill_v2_registry.py and
# wandb_log_public.py, never here.
LABEL_DEV_RUN = "final adapter, dev100, fp16 inference"
LABEL_DEV_OOF = "supplier-held-out"  # dev100 from the pooled OOF predictions
V2_KINDS = ("dev_run", "dev_oof", "pooled_oof")
# the slices the v2 report line shows, in its order (labels = the report's own words)
V2_SLICES: tuple[tuple[str, str], ...] = (
    ("all", "all"),
    ("invoices", "invoices"),
    ("waybills", "waybills"),
    ("scanned=yes", "scanned"),
    ("scanned=no", "digital"),
    ("multipage=yes", "multipage"),
    ("repeated_parts=yes", "repeated parts"),
    ("illegible=yes", "redaction"),
    ("absent_line=yes", "absent line"),
)


def _script(name: str) -> Any:
    """Sibling script as a module (``g4_fold`` / ``g4_pooled`` load their own helpers lazily)."""
    import importlib

    return importlib.import_module(name)


def _devfinal_facts(man: Mapping[str, Any]) -> dict[str, Any]:
    """Training precision, merge dtype and batch size from an 08 manifest's ``devfinal`` section."""
    sec = man.get("devfinal") or {}
    return {
        "train_precision": sec.get("train_precision"),
        "merge_dtype": (sec.get("merge") or {}).get("merge_dtype"),
        "batch_used": (sec.get("batch") or {}).get("used"),
    }


def compute_dev_run(
    run_dir: Path, ocr_cache: Path | None, n_boot: int, stand_in: bool = False
) -> dict[str, Any]:
    """Slices of ONE run on exactly the 100 dev documents: FT (or any) arm + rules, production path.

    `run_dir` is the 08 run folder (``predictions.json`` = rules OFF, ``trace.jsonl``). The rules
    run through ``replay_v1_check.production`` (R1 + R2 + R3) exactly as 08's PRIMARY arm: R3
    shapes learned from the TRAIN gold only. Refuses (SystemExit) unless the run holds exactly the
    dev ids; `stand_in` instead accepts a larger run (e.g. the 500-doc ZS run), restricts it to the
    dev ids and stamps the result ``stand_in`` (plumbing rehearsal, not an FT result).
    """
    from shipdoc import rules as rules_mod

    g4f = _script("g4_fold")
    dev_ids = sorted(json.loads((ROOT / "splits" / "dev100.json").read_text(encoding="utf-8")))
    pred = json.loads((run_dir / "predictions.json").read_text(encoding="utf-8"))
    if not stand_in and sorted(pred) != dev_ids:
        raise SystemExit(
            f"{run_dir.name}: predictions.json is not exactly the {len(dev_ids)} dev documents"
        )
    if not stand_in:
        from shipdoc import devfinal
        from shipdoc.runmeta import read_manifest

        g4f.rg.require_complete(run_dir, "08 dev")
        man = read_manifest(run_dir) or {}
        if (man.get("devfinal") or {}).get("ids_sha256") != devfinal.ids_sha256(dev_ids):
            raise SystemExit(f"{run_dir.name}: not a 08 run of exactly the dev ids (manifest)")
    arm = g4f.read_arm("dev run", run_dir, dev_ids)
    gold = g4f.load_gold(dev_ids)
    train_gold = ev.load_gold(paths.data_dir() / "train" / "labels")
    shapes = rules_mod.learn_slot_shapes(list(train_gold.values()))
    raw, new, summary = g4f.process_arm(arm, shapes, ocr_cache)
    meta_list = ev.load_json(ROOT / "meta" / "dev.json")
    meta = {m["doc_id"]: m for m in meta_list}
    slices = derived_slices(dev_ids, meta, ev.slice_doc_ids(gold, meta_list, group_slices=False))
    return {
        "schema": V2_SCHEMA,
        "kind": "dev_run",
        "stand_in": stand_in,
        # what the 08 manifest records (None for a stand-in); the GPU name is NOT among it
        "facts": None if stand_in else _devfinal_facts(man),
        "run": run_dir.name,
        "n_docs": len(dev_ids),
        "rules_summary": summary,
        "slices": slice_rows(raw, new, gold, slices, n_boot),
        "false_fill": false_fill_rows(raw, new, gold, meta, dev_ids),
    }


def compute_pooled_oof(
    oof_dirs: Sequence[Path],
    zs_dir: Path,
    ocr_cache: Path | None,
    n_boot: int,
    stand_in: bool = False,
) -> dict[str, Any]:
    """Slices of the pooled supplier-held-out OOF of FT + rules (the no-08 fallback), 500 docs.

    The three OOF run folders are validated and pooled exactly as ``scripts/g4_pooled.py`` does
    (its ``validate_oof_runs``, ``process_arm`` with the honest per-fold R3 shapes, ``pool_arms``);
    the pooled rules-on predictions are then sliced like the 500-doc zero-shot table. `stand_in`
    uses `zs_dir` as every fold's arm (no OOF dirs needed): a plumbing rehearsal, not an FT result.
    """
    g4f = _script("g4_fold")
    g4p = _script("g4_pooled")
    from shipdoc import g4pooled

    rg = g4f.rg
    folds = json.loads((ROOT / "splits" / "folds.json").read_text(encoding="utf-8"))
    doc_fold = rg.load_doc_fold()
    fold_list = sorted(set(doc_fold.values()))
    fold_ids = {k: rg.fold_docs(folds, k, doc_fold) for k in fold_list}
    runs = {} if stand_in else g4p.validate_oof_runs(list(oof_dirs), folds, doc_fold)
    all_ids = sorted(d for k in fold_list for d in fold_ids[k])
    gold = g4f.load_gold(all_ids)
    meta_list = ev.load_json(ROOT / "meta" / "train.json") + ev.load_json(
        ROOT / "meta" / "dev.json"
    )
    meta = {m["doc_id"]: m for m in meta_list}
    labels = meta_mod.load_labels("train") + meta_mod.load_labels("dev")
    groups = json.loads((ROOT / "meta" / "supplier_groups.json").read_text(encoding="utf-8"))
    try:
        per_fold = rg.fold_shapes(labels, doc_fold, groups)
    except ValueError as e:  # supplier overlap: fail closed
        raise SystemExit(str(e)) from e
    raws: dict[int, Any] = {}
    news: dict[int, Any] = {}
    for k in fold_list:
        arm = g4f.read_arm(f"FT arm fold {k}", zs_dir if stand_in else runs[k][0], fold_ids[k])
        raws[k], news[k], _ = g4f.process_arm(arm, per_fold[k][0], ocr_cache)
    raw_all = g4pooled.pool_arms(raws, fold_ids, doc_fold, g4pooled.EXPECTED_DOCS)
    new_all = g4pooled.pool_arms(news, fold_ids, doc_fold, g4pooled.EXPECTED_DOCS)
    slices = derived_slices(
        all_ids, meta, ev.slice_doc_ids(gold, meta_list, group_slices=False), doc_fold
    )
    return {
        "schema": V2_SCHEMA,
        "kind": "pooled_oof",
        "stand_in": stand_in,
        "run": zs_dir.name if stand_in else ",".join(runs[k][0].name for k in fold_list),
        "n_docs": len(all_ids),
        "slices": slice_rows(raw_all, new_all, gold, slices, n_boot),
        "fold_slices": slice_rows(raw_all, new_all, gold, slices, n_boot, FOLD_SLICES),
        "false_fill": false_fill_rows(raw_all, new_all, gold, meta, all_ids),
    }


def compute_dev_oof(
    oof_dirs: Sequence[Path],
    zs_dir: Path,
    ocr_cache: Path | None,
    n_boot: int,
    folds: Sequence[int] | None = None,
) -> dict[str, Any]:
    """FT + rules on the 100 DEV docs from the pooled OOF runs (supplier-held-out), plus ZS + rules.

    Each dev document is predicted by the OOF adapter of the fold that held its supplier out and
    scored with R3 shapes learned from the other folds (honest, as ``g4_fold`` / ``g4_pooled``);
    the ZS arm is the 02 run through the same path on the same docs. All three folds are validated
    like ``g4_pooled`` does. `folds` (e.g. [0, 1]) restricts to those folds' dev docs: a PARTIAL
    plumbing result (``partial: true``, manifests checked per run only) that the registry filler
    refuses. Slices are the v1.5 dev table's; every FT row also carries the paired FT - ZS delta.
    """
    g4f = _script("g4_fold")
    g4p = _script("g4_pooled")
    rg = g4f.rg
    folds_json = json.loads((ROOT / "splits" / "folds.json").read_text(encoding="utf-8"))
    doc_fold = rg.load_doc_fold()
    fold_list = sorted(set(doc_fold.values()))
    want = sorted(set(folds)) if folds else fold_list
    partial = want != fold_list
    fold_ids = {k: rg.fold_docs(folds_json, k, doc_fold) for k in fold_list}
    if partial:
        runs: dict[int, Path] = {}
        for d in oof_dirs:
            k, _sec = rg.check_oof_manifest(g4p.read_manifest(d), None, False)
            rg.require_complete(d, "OOF")
            runs[k] = d
        if sorted(runs) != want:
            raise SystemExit(f"--folds {want} but the OOF runs are folds {sorted(runs)}")
    else:
        runs = {
            k: v[0] for k, v in g4p.validate_oof_runs(list(oof_dirs), folds_json, doc_fold).items()
        }
    dev_set = {d for d in doc_fold if d.startswith("dev_")}
    labels = meta_mod.load_labels("train") + meta_mod.load_labels("dev")
    groups = json.loads((ROOT / "meta" / "supplier_groups.json").read_text(encoding="utf-8"))
    try:
        per_fold = rg.fold_shapes(labels, doc_fold, groups)
    except ValueError as e:  # supplier overlap: fail closed
        raise SystemExit(str(e)) from e
    raws: dict[str, dict[str, Any]] = {"ft": {}, "zs": {}}
    news: dict[str, dict[str, Any]] = {"ft": {}, "zs": {}}
    for k in want:
        ids = [d for d in fold_ids[k] if d in dev_set]
        if not ids:
            continue
        for name, src in (("ft", runs[k]), ("zs", zs_dir)):
            arm = g4f.read_arm(f"{name} arm fold {k}", src, ids)
            raw, new, _ = g4f.process_arm(arm, per_fold[k][0], ocr_cache)
            raws[name] |= raw
            news[name] |= new
    dev_ids = sorted(news["ft"])
    if not partial and dev_ids != sorted(dev_set):
        raise SystemExit("the pooled dev docs are not exactly the dev documents")
    gold = g4f.load_gold(dev_ids)
    meta_list = ev.load_json(ROOT / "meta" / "dev.json")
    meta = {m["doc_id"]: m for m in meta_list}
    slices = derived_slices(dev_ids, meta, ev.slice_doc_ids(gold, meta_list, group_slices=False))
    ft_rows = slice_rows(raws["ft"], news["ft"], gold, slices, n_boot)
    zs_rows = slice_rows(raws["zs"], news["zs"], gold, slices, n_boot)
    for r in ft_rows:
        if r["n"]:
            ids = list(slices[r["slice"]])
            pb = ev.paired_bootstrap(
                {d: news["zs"][d] for d in ids},
                {d: news["ft"][d] for d in ids},
                {d: gold[d] for d in ids},
                n=n_boot,
                seed=SEED,
            )["OVERALL"]
            r["ft_minus_zs"] = {"delta": pb["delta"], "lo": pb["lo"], "hi": pb["hi"]}
    return {
        "schema": V2_SCHEMA,
        "kind": "dev_oof",
        "stand_in": False,
        "partial": partial,
        "folds": want,
        "run": ",".join(runs[k].name for k in want),
        "n_docs": len(dev_ids),
        "slices": ft_rows,
        "zs_slices": zs_rows,
        "false_fill": false_fill_rows(raws["ft"], news["ft"], gold, meta, dev_ids),
    }


def render_v2(res: Mapping[str, Any], cmd: str, state: str) -> str:
    """Markdown of a v2 slice result (aggregates only; states plainly which evidence it is)."""
    kind = res["kind"]
    dev = kind in ("dev_run", "dev_oof")
    title = {
        "dev_run": f"FT + rules, {LABEL_DEV_RUN}: by slice (dev evidence only)",
        "dev_oof": f"FT + rules on the 100 dev documents, {LABEL_DEV_OOF}: by slice, with ZS + "
        "rules on the same documents",
        "pooled_oof": "FT + rules, pooled 3-fold supplier-held-out OOF, by slice (contingency: "
        "the final-adapter dev run was not done)",
    }[kind]
    what = "dev" if dev else "500-doc"
    L = [f"# {title}", ""]
    if res.get("partial"):
        L += [
            f"**PARTIAL: folds {res.get('folds')} only, NOT THE RESULT** (plumbing; the dev "
            "documents of the other folds are missing and the OOF runs were not cross-validated "
            "like `g4_pooled.py` does).",
            "",
        ]
    if res["stand_in"]:
        L += [
            "**STAND-IN: PLUMBING REHEARSAL, NOT AN FT RESULT.** The arm below is the zero-shot "
            "run used in place of the fine-tuned one; no number says anything about the "
            "fine-tuned model.",
            "",
        ]
    seen = {
        "dev_run": "The dev documents are SEEN layouts (their supplier groups also occur in "
        "train): this is not an unseen-supplier estimate; the adapter itself trained on the 400 "
        "train documents only. R3 shapes: learned from the TRAIN gold only (08's primary arm). "
        "Dev evidence only: the test submission is T4. The GPU name is not recorded in any 08 "
        "artifact (the manifest has train precision, merge dtype and batch size only).",
        "dev_oof": "Each dev document is predicted by the OOF adapter of the fold that held its "
        "supplier out and scored with R3 shapes learned from the other folds (honest, "
        "supplier-disjoint): the stricter, unseen-supplier dev evidence.",
        "pooled_oof": "Every document is predicted by the adapter of the fold that held its "
        "supplier out and scored with R3 shapes learned from the other folds (honest, "
        "supplier-disjoint): the unseen-supplier view.",
    }[kind]
    L += [
        f"**Provenance.** Run `{res['run']}`, command `{cmd}` (repo state `{state}`; bootstrap "
        f"{N_BOOT} doc-level resamples, seed {SEED}; unmodified scorer via `shipdoc.eval`; "
        "production rule path `replay_v1_check.production`, R1 + R2 + R3; rules OFF reproduces "
        f"`predictions.json` exactly, asserted). **All numbers UNVERIFIED.** {seen} Slices with "
        f"n < {MIN_N_FOR_CI} are printed but their intervals are not informative.",
        "",
        f"## {what} slices ({res['n_docs']} docs)",
        "",
        *_slice_table(res["slices"], what, "FT + rules"),
    ]
    L += _overlap_note(res["slices"], what)
    if res.get("zs_slices"):
        L += ["", f"## ZS + rules on the same {what} documents (comparator)", ""]
        L += _slice_table(res["zs_slices"], what)
        L += [
            "",
            "## Paired FT - ZS delta (OVERALL, pts) per slice",
            "",
            "| slice | n | delta [95% CI] |",
        ]
        L += ["|---|---|---|"]
        L += [
            f"| {r['label']} | {r['n']} | {_dci(r['ft_minus_zs'])} |"
            for r in res["slices"]
            if r.get("ft_minus_zs")
        ]
    if res.get("fold_slices"):
        L += ["", "## By validation fold (supplier-held-out)", ""]
        L += _slice_table(res["fold_slices"], "fold", "FT + rules")
    L += [
        "",
        "## Header false fills by why the gold is null (Wilson 95%)",
        "",
        "| arm | kind | gold-null header fields | filled | rate % [95% CI] |",
        "|---|---|---|---|---|",
    ]
    for arm in ("raw", "rules"):
        for kind in ("redaction", "absent_line"):
            v = res["false_fill"][arm][kind]
            rate = _ci(v["rate"]) if v["rate"] else "n/a"
            L.append(f"| {arm} | {kind} | {v['null_fields']} | {v['filled']} | {rate} |")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", type=Path, default=None, help="the native ZS run folder")
    ap.add_argument(
        "--mode",
        choices=("zs500", *V2_KINDS),
        default="zs500",
        help="zs500 (default, the v1.5 table) | dev_run (08 run folder) | dev_oof (dev100 from "
        "3 OOF runs + ZS) | pooled_oof (all 500 from 3 OOF runs)",
    )
    ap.add_argument("--oof-run-dir", type=Path, action="append", default=[])
    ap.add_argument("--zs-run-dir", type=Path, default=None, help="pooled_oof / dev_oof: 02 ZS run")
    ap.add_argument(
        "--folds",
        type=int,
        action="append",
        default=[],
        help="dev_oof only: restrict to these folds (repeatable): a PARTIAL plumbing result",
    )
    ap.add_argument(
        "--stand-in",
        action="store_true",
        help="dev_run / pooled_oof rehearsal on a zero-shot run (stamped NOT A RESULT)",
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--ocr-cache", type=Path, default=None)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args(argv)
    cmd = "uv run python scripts/dev_slices_native.py " + " ".join(
        argv if argv is not None else sys.argv[1:]
    )
    if a.mode != "zs500":
        if a.mode == "dev_run":
            if a.run_dir is None:
                ap.error("--mode dev_run needs --run-dir (the 08 run folder)")
            v2 = compute_dev_run(a.run_dir, a.ocr_cache, a.n_boot, a.stand_in)
        elif a.mode == "dev_oof":
            zs = a.zs_run_dir or a.run_dir
            if zs is None or not a.oof_run_dir:
                ap.error("--mode dev_oof needs --zs-run-dir and the --oof-run-dir folders")
            if a.stand_in:
                ap.error("--stand-in does not apply to --mode dev_oof")
            if not a.folds and len(a.oof_run_dir) != 3:
                ap.error("--mode dev_oof needs three --oof-run-dir (or --folds for a partial run)")
            v2 = compute_dev_oof(a.oof_run_dir, zs, a.ocr_cache, a.n_boot, a.folds or None)
        else:
            zs = a.zs_run_dir or a.run_dir
            if zs is None:
                ap.error("--mode pooled_oof needs --zs-run-dir (and three --oof-run-dir)")
            if not a.stand_in and len(a.oof_run_dir) != 3:
                ap.error("--mode pooled_oof needs --oof-run-dir three times (or --stand-in)")
            v2 = compute_pooled_oof(a.oof_run_dir, zs, a.ocr_cache, a.n_boot, a.stand_in)
        out = a.out or ROOT / "reports" / "dev_slices_native_v2.md"
        out.write_text(render_v2(v2, cmd, git_state()), encoding="utf-8", newline="\n")
        if a.json_out:
            a.json_out.parent.mkdir(parents=True, exist_ok=True)
            a.json_out.write_text(json.dumps(v2, indent=1, sort_keys=True), encoding="utf-8")
        print(f"wrote {out}")
        return 0
    if a.run_dir is None:
        ap.error("--run-dir is required")
    a.out = a.out or ROOT / "reports" / "dev_slices_native.md"
    res = compute(a.run_dir, a.ocr_cache, a.n_boot)
    cmd = "uv run python scripts/dev_slices_native.py " + " ".join(
        argv if argv is not None else sys.argv[1:]
    )
    a.out.write_text(render(res, cmd, git_state()), encoding="utf-8", newline="\n")
    if a.json_out:
        a.json_out.parent.mkdir(parents=True, exist_ok=True)
        a.json_out.write_text(json.dumps(res, indent=1, sort_keys=True), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
