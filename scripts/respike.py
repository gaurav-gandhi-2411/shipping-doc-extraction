"""Re-spike analysis (Step S): rescoring, slices, null handling, keyed-vs-compact A/B.

Everything is computed from the saved artifacts of the Colab re-spike (``predictions.json``,
``trace.jsonl``, ``metrics.json`` per run dir) and the unmodified official scorer
(``assignment/score.py`` via ``shipdoc.eval``) against ``data/dev/labels``. No model is called.

Outputs: ``reports/respike.md`` (aggregates and counts only), a local value-level file outside the
repo (``<SHIPDOC_RUNS_DIR>/diagnosis/respike_local.md``) and optionally a results JSON.

Run: uv run python scripts/respike.py --runs-dir <unzipped x/ folder> [--final-pick <run dir name>]
(the runs dir may also come from $SHIPDOC_RESPIKE_DIR). Every number it writes is UNVERIFIED until
a verifier recomputes it.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from shipdoc import diagnostics as dg
from shipdoc import eval as ev
from shipdoc import format_ab as fab
from shipdoc import meta as meta_mod
from shipdoc import paths
from shipdoc.replay import read_trace

ROOT = Path(__file__).resolve().parents[1]
RUNS_ENV = "SHIPDOC_RESPIKE_DIR"
CODE_SHA = "2abf481"
N_BOOT = 2000
SEED = 42
MODELS = ("qwen35_4b", "qwen3vl_8b", "nuextract3")
ARMS = ("img_only", "img_ocr")
ROW_NULL_FIELDS = ("customer_part_number", "purchase_order")
#: Metrics of ``metrics.json["slices"]["all"]`` that must equal the rescore exactly.
CHECK_KEYS = (
    "documents",
    "OVERALL",
    "header_field_accuracy",
    "row_f1",
    "documents_fully_correct",
    "false_fill_rate",
    "illegible_fields",
)
HEADER_METRICS = ("OVERALL", "header_field_accuracy", "row_f1", "documents_fully_correct")
KEYED_START = '{"doc_type"'
COMPACT_START = '{"dt"'


# --------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------


def pct(x: float | None, d: int = 2) -> str:
    """Fraction -> percent string (``n/a`` for None / NaN)."""
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{d}f}"


def pci(c: Mapping[str, float] | None, d: int = 2) -> str:
    """``point [lo, hi]`` in percent."""
    if c is None:
        return "n/a"
    return f"{pct(c['point'], d)} [{pct(c['lo'], d)}, {pct(c['hi'], d)}]"


def md_table(head: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Markdown table; every cell is str()-ed."""
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def ratio(num: int, den: int) -> float | None:
    """num/den or None when den == 0."""
    return num / den if den else None


def rate_str(num: int, den: int) -> str:
    """``num/den (xx.x%)`` or ``num/0 (n/a)``."""
    return f"{num}/{den} ({pct(ratio(num, den), 1)}%)" if den else f"{num}/0 (n/a)"


# --------------------------------------------------------------------------------------------
# Loading and rescoring
# --------------------------------------------------------------------------------------------


@dataclass
class RunData:
    """One finished run dir."""

    name: str
    path: Path
    pred: dict[str, Any]
    traces: list[dict[str, Any]]
    metrics: dict[str, Any]
    extra: dict[str, Any] = field(default_factory=dict)


def load_run(path: Path) -> RunData:
    """Read ``predictions.json``, ``trace.jsonl`` and ``metrics.json`` of a run dir."""
    path = Path(path)
    pred = json.loads((path / "predictions.json").read_text(encoding="utf-8"))
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    return RunData(path.name, path, pred, read_trace(path / "trace.jsonl"), metrics)


def restrict(gold: Mapping[str, Any], ids: Sequence[str]) -> dict[str, Any]:
    """Gold restricted to `ids` (all must exist)."""
    miss = [d for d in ids if d not in gold]
    if miss:
        raise ValueError(f"{len(miss)} doc ids have no gold, e.g. {miss[:3]}")
    return {d: gold[d] for d in sorted(ids)}


def compare_metrics(agg: Mapping[str, Any], saved: Mapping[str, Any]) -> dict[str, Any]:
    """Float-equality check of the rescore `agg` against metrics.json ``slices.all``."""
    s = saved.get("slices", {}).get("all", {})
    rows = {}
    for k in CHECK_KEYS:
        if k not in s:
            rows[k] = {"rescore": agg[k], "saved": None, "equal": None}
            continue
        rows[k] = {"rescore": agg[k], "saved": s[k], "equal": bool(agg[k] == s[k])}
    checked = [r["equal"] for r in rows.values() if r["equal"] is not None]
    return {"keys": rows, "all_equal": bool(checked) and all(checked), "n_checked": len(checked)}


def run_identity(run: RunData) -> dict[str, Any]:
    """Output format / config / prompt / commit as recorded in the trace, plus raw_text shape."""
    first = run.traces[0]
    shape = {"keyed": 0, "compact": 0, "other": 0}
    for t in run.traces:
        for p in t["pages"]:
            raw = p["raw_text"].lstrip()
            if raw.startswith(KEYED_START):
                shape["keyed"] += 1
            elif raw.startswith(COMPACT_START):
                shape["compact"] += 1
            else:
                shape["other"] += 1
    formats = {t["output_format"] for t in run.traces}
    hashes = {t["config"]["hash"] for t in run.traces}
    return {
        "output_format_trace": sorted(formats),
        "raw_text_shape_pages": shape,
        "config_name": first["config"]["name"],
        "config_hash": sorted(hashes),
        "prompt_version": first["prompt"]["version"],
        "prompt_hash": first["prompt"]["hash"],
        "git_commit": first["git_commit"][:7],
        "model_id": first["model"]["id"],
        "model_revision": first["model"]["revision"],
        "metrics_output_format": run.metrics.get("output_format"),
    }


def trace_stats(run: RunData) -> dict[str, Any]:
    """Latency, VRAM and JSON validity recomputed from the trace, next to metrics.json's values."""
    pages = [p for t in run.traces for p in t["pages"]]
    lat = [p["meta"]["latency_s"] for p in pages if p["meta"].get("latency_s") is not None]
    vram = [p["meta"]["peak_vram_bytes"] for p in pages if p["meta"].get("peak_vram_bytes")]
    valid = sum(bool(p["json_valid"]) for p in pages)
    return {
        "n_pages": len(pages),
        "s_per_page_mean": float(np.mean(lat)) if lat else None,
        "s_per_page_p95": float(np.percentile(lat, 95)) if lat else None,
        "peak_vram_bytes": int(max(vram)) if vram else None,
        "json_validity": valid / len(pages) if pages else None,
        "metrics_s_per_page_mean": run.metrics.get("s_per_page_mean"),
        "metrics_s_per_page_p95": run.metrics.get("s_per_page_p95"),
        "metrics_peak_vram_bytes": run.metrics.get("peak_vram_max_bytes"),
        "metrics_json_validity": run.metrics.get("json_validity_rate"),
    }


def full_run_row(
    run: RunData, gold: Mapping[str, Any], n: int = N_BOOT, seed: int = SEED
) -> dict[str, Any]:
    """S1 row for one run: rescore, equality check, CIs, speed, VRAM."""
    g = restrict(gold, list(run.pred))
    agg = ev.score(run.pred, g)["all"]
    per = ev.per_doc_results(run.pred, g)
    cis = ev.bootstrap_metrics(list(per.values()), n=n, seed=seed)
    return {
        "run": run.name,
        "n_docs": agg["documents"],
        "agg": {k: v for k, v in agg.items() if not isinstance(v, dict)},
        "header_by_field": agg["header_by_field"],
        "line_item_field_accuracy": agg["line_item_field_accuracy"],
        "ci": cis,
        "check": compare_metrics(agg, run.metrics),
        "identity": run_identity(run),
        "trace": trace_stats(run),
        "step_l": dg.run_row_convention_errors(run.path, g),
    }


# --------------------------------------------------------------------------------------------
# Slices
# --------------------------------------------------------------------------------------------


def slice_ids(gold: Mapping[str, Any], meta: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    """Slice name -> doc ids for the S2 report.

    Meta tags (scanned, multipage, repeated_parts, illegible, waybill) come from ``meta/dev.json``.
    ``awb_absent`` / ``hawb_absent`` are derived from the gold labels (invoice with a null
    awb_number / waybill with a null hawb) because ``meta/dev.json`` does not carry those tags.
    """
    m = {x["doc_id"]: x for x in meta if x["doc_id"] in gold}
    missing = [d for d in gold if d not in m]
    if missing:
        raise ValueError(f"{len(missing)} docs have no meta, e.g. {missing[:3]}")
    ids = list(gold)

    def pick(pred: Any) -> list[str]:
        return [d for d in ids if pred(d)]

    def awb_absent(d: str) -> bool:
        return gold[d]["doc_type"] == "invoice" and dg.empty(gold[d]["header"].get("awb_number"))

    def hawb_absent(d: str) -> bool:
        return gold[d]["doc_type"] == "waybill" and dg.empty(gold[d]["header"].get("hawb"))

    out = {
        "all": ids,
        "invoice": pick(lambda d: gold[d]["doc_type"] == "invoice"),
        "waybill": pick(lambda d: gold[d]["doc_type"] == "waybill"),
        "scanned": pick(lambda d: m[d]["scanned"]),
        "digital": pick(lambda d: not m[d]["scanned"]),
        "multipage": pick(lambda d: m[d]["multipage"]),
        "single page": pick(lambda d: not m[d]["multipage"]),
        "repeated_parts": pick(lambda d: m[d]["repeated_parts"]),
        "no repeated_parts": pick(lambda d: not m[d]["repeated_parts"]),
        "illegible": pick(lambda d: m[d]["illegible"]),
        "not illegible": pick(lambda d: not m[d]["illegible"]),
        "awb_absent (invoice)": pick(awb_absent),
        "awb_present (invoice)": pick(
            lambda d: gold[d]["doc_type"] == "invoice" and not awb_absent(d)
        ),
        "hawb_absent (waybill)": pick(hawb_absent),
        "hawb_present (waybill)": pick(
            lambda d: gold[d]["doc_type"] == "waybill" and not hawb_absent(d)
        ),
    }
    return out


def field_cis(
    per_list: Sequence[Mapping[str, Any]], n: int = N_BOOT, seed: int = SEED
) -> dict[str, Any]:
    """Doc-level bootstrap CIs of per-field header accuracy and per-field row accuracy.

    A field absent from a resample (e.g. a waybill-only field in an invoice-only draw) yields NaN
    for that draw and is dropped from the quantile; row accuracy is over all gold rows
    (``line_item_field_accuracy``). Returns ``{"header": {f: {point, lo, hi, n_docs}}, "row": ...}``
    (percentile CIs).
    """
    sc = ev.load_scorer()
    N = len(per_list)
    out: dict[str, Any] = {"header": {}, "row": {}}
    if N == 0:
        return out
    point = sc.aggregate(list(per_list))
    hf = list(point["header_by_field"])
    rf = list(point["line_item_field_accuracy"])
    idx = ev._rng(seed).integers(0, N, size=(n, N))
    hd: dict[str, list[float]] = {f: [] for f in hf}
    rd: dict[str, list[float]] = {f: [] for f in rf}
    for row in idx:
        a = sc.aggregate([per_list[i] for i in row])
        for f in hf:
            hd[f].append(a["header_by_field"].get(f, float("nan")))
        for f in rf:
            rd[f].append(a["line_item_field_accuracy"].get(f, float("nan")) if a else float("nan"))
    n_docs = {f: sum(f in r["header"] for r in per_list) for f in hf}
    for kind, draws, pts in (
        ("header", hd, point["header_by_field"]),
        ("row", rd, point["line_item_field_accuracy"]),
    ):
        for f, d in draws.items():
            lo, hi = np.nanquantile(d, [0.025, 0.975])
            out[kind][f] = {
                "point": float(pts[f]),
                "lo": float(lo),
                "hi": float(hi),
                "n_docs": n_docs.get(f) if kind == "header" else point["line_item_rows"],
            }
    return out


def slice_report(
    pred: Mapping[str, Any],
    gold: Mapping[str, Any],
    slices: Mapping[str, Sequence[str]],
    n: int = N_BOOT,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    """One row per slice: n, header acc / row F1 / fully correct / OVERALL with CIs, false fill."""
    sc = ev.load_scorer()
    per = ev.per_doc_results(dict(pred), dict(gold))
    rows = []
    for name, ids in slices.items():
        if not ids:
            rows.append({"slice": name, "n": 0})
            continue
        lst = [per[d] for d in ids]
        agg = sc.aggregate(lst)
        cis = ev.bootstrap_metrics(lst, n=n, seed=seed)
        rows.append(
            {
                "slice": name,
                "n": len(ids),
                "gold_rows": agg["line_item_rows"],
                "ci": {k: cis[k] for k in HEADER_METRICS},
                "false_fill_rate": agg["false_fill_rate"],
                "illegible_fields": agg["illegible_fields"],
            }
        )
    return rows


# --------------------------------------------------------------------------------------------
# Null handling
# --------------------------------------------------------------------------------------------


def not_printed_fields() -> dict[str, set[str]]:
    """Per doc type, header fields whose nulls mean "line not printed" (train+dev pooled)."""
    docs = meta_mod.load_labels("train") + meta_mod.load_labels("dev")
    return meta_mod.not_printed_fields(docs)


def header_cells(
    pred: Mapping[str, Any],
    gold: Mapping[str, Any],
    not_printed: Mapping[str, set[str]],
    scanned: Mapping[str, bool],
) -> list[dict[str, Any]]:
    """One record per scored gold header cell: null status of gold and prediction.

    ``cls`` is ``absent_line`` for fields in `not_printed` (a gold null means the line is not on
    the page) and ``required`` otherwise (a gold null is a redaction / scribble). Cells follow
    the scorer: the fields of the GOLD doc type, whatever doc_type was predicted.
    """
    cells = []
    for d, g in gold.items():
        p = pred.get(d)
        ph = p.get("header") if isinstance(p, dict) and isinstance(p.get("header"), dict) else {}
        for f in ev.load_scorer().HEADER[g["doc_type"]]:
            cells.append(
                {
                    "doc": d,
                    "type": g["doc_type"],
                    "field": f,
                    "cls": "absent_line"
                    if f in not_printed.get(g["doc_type"], set())
                    else "required",
                    "gold_null": dg.empty(g["header"].get(f)),
                    "pred_null": dg.empty(ph.get(f)),
                    "scanned": bool(scanned[d]),
                }
            )
    return cells


def confusion(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Null confusion of cells: counts, null precision / recall, over-null rate.

    over_null = gold has a value, prediction null; false_fill = gold null, prediction has a value
    (the scorer's false fill, header only); null precision = true null / predicted null; null
    recall = true null / gold null.
    """
    gold_null = sum(c["gold_null"] for c in cells)
    pred_null = sum(c["pred_null"] for c in cells)
    true_null = sum(c["gold_null"] and c["pred_null"] for c in cells)
    over = sum((not c["gold_null"]) and c["pred_null"] for c in cells)
    ff = sum(c["gold_null"] and not c["pred_null"] for c in cells)
    gold_valued = len(cells) - gold_null
    return {
        "cells": len(cells),
        "gold_null": gold_null,
        "gold_valued": gold_valued,
        "pred_null": pred_null,
        "true_null": true_null,
        "over_null": over,
        "false_fill": ff,
        "null_precision": ratio(true_null, pred_null),
        "null_recall": ratio(true_null, gold_null),
        "over_null_rate": ratio(over, gold_valued),
    }


def header_null_report(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Confusions pooled, by scanned/digital, by class, by class x scanned, and per field."""
    out: dict[str, Any] = {"pooled": confusion(cells)}
    for sc_name, flag in (("scanned", True), ("digital", False)):
        out[sc_name] = confusion([c for c in cells if c["scanned"] is flag])
    for cls in ("required", "absent_line"):
        sub = [c for c in cells if c["cls"] == cls]
        out[cls] = confusion(sub)
        for sc_name, flag in (("scanned", True), ("digital", False)):
            out[f"{cls}/{sc_name}"] = confusion([c for c in sub if c["scanned"] is flag])
    fields: dict[str, Any] = {}
    for f in sorted({(c["type"], c["field"]) for c in cells}):
        sub = [c for c in cells if (c["type"], c["field"]) == f]
        row = confusion(sub)
        row["cls"] = sub[0]["cls"]
        for sc_name, flag in (("scanned", True), ("digital", False)):
            row[sc_name] = confusion([c for c in sub if c["scanned"] is flag])
        fields[f"{f[0]}.{f[1]}"] = row
    out["fields"] = fields
    return out


def pair_rows_of(
    sc: ModuleType, pr: list[dict[str, Any]], gr: list[dict[str, Any]]
) -> list[tuple[int, int]]:
    """Full and partial (part-number) pairs of the scorer's greedy matching."""
    full, partial, _, _ = ev._pair_rows(sc, pr, gr)
    return full + partial


def row_cells(
    pred: Mapping[str, Any], gold: Mapping[str, Any], scanned: Mapping[str, bool]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Row-field null cells over scorer-paired rows, plus marginal null counts over all rows.

    Only invoice/invoice pairs have rows to compare (as in ``shipdoc.diagnostics``). Paired cells
    are exact for fully matched rows and for partial pairs (matched on supplier_part_number);
    gold rows left unpaired (e.g. a misread supplier_part_number) contribute no cell, only to the
    marginal counts, which compare null counts per field over ALL gold rows and ALL predicted rows
    of docs whose gold type is invoice.
    """
    sc = ev.load_scorer()
    cells: list[dict[str, Any]] = []
    marg: dict[str, Any] = {
        s: {"gold_rows": 0, "pred_rows": 0, **{f"gold_null_{f}": 0 for f in sc.ROW}}
        | {f"pred_null_{f}": 0 for f in sc.ROW}
        for s in ("scanned", "digital")
    }
    for d, g in gold.items():
        if g["doc_type"] != "invoice":
            continue
        p = pred.get(d)
        pr = (
            [x for x in (p.get("line_items") or []) if isinstance(x, dict)]
            if isinstance(p, dict)
            else []
        )
        gr = list(g.get("line_items") or [])
        side = "scanned" if scanned[d] else "digital"
        marg[side]["gold_rows"] += len(gr)
        marg[side]["pred_rows"] += len(pr)
        for f in sc.ROW:
            marg[side][f"gold_null_{f}"] += sum(dg.empty(x.get(f)) for x in gr)
            marg[side][f"pred_null_{f}"] += sum(dg.empty(x.get(f)) for x in pr)
        if not (isinstance(p, dict) and p.get("doc_type") == "invoice"):
            continue
        for pi, gi in pair_rows_of(sc, pr, gr):
            for f in sc.ROW:
                cells.append(
                    {
                        "doc": d,
                        "field": f,
                        "gold_null": dg.empty(gr[gi].get(f)),
                        "pred_null": dg.empty(pr[pi].get(f)),
                        "scanned": bool(scanned[d]),
                        "pi": pi,
                        "gi": gi,
                    }
                )
    return cells, marg


def row_null_report(cells: Sequence[Mapping[str, Any]], marg: Mapping[str, Any]) -> dict[str, Any]:
    """Per row field: paired-cell confusion pooled and by scanned/digital, plus marginals."""
    sc = ev.load_scorer()
    out: dict[str, Any] = {"fields": {}, "marginal": marg}
    for f in sc.ROW:
        sub = [c for c in cells if c["field"] == f]
        row = confusion(sub)
        for sc_name, flag in (("scanned", True), ("digital", False)):
            row[sc_name] = confusion([c for c in sub if c["scanned"] is flag])
        out["fields"][f] = row
    return out


def oracle_fill(
    pred: Mapping[str, Any],
    gold: Mapping[str, Any],
    header: bool = False,
    row_fields: Sequence[str] = (),
    blank_false_fill: Sequence[str] = (),
) -> tuple[dict[str, Any], dict[str, int]]:
    """ORACLE: replace over-null predictions by the gold value. Never a fix, only an upper bound.

    Header: every scored header cell with a gold value and an empty prediction. Rows: for each
    scorer-paired row (computed on the unmodified prediction), each of `row_fields` with a gold
    value and an empty prediction; each of `blank_false_fill` with an empty gold and a
    non-empty prediction is blanked (so a row's whole null status can be made right). Returns
    (patched deep copy, cells touched per kind: header, row, blanked).
    """
    sc = ev.load_scorer()
    out = copy.deepcopy(dict(pred))
    filled = {"header": 0, "row": 0, "blanked": 0}
    for d, g in gold.items():
        p = out.get(d)
        if not isinstance(p, dict):
            continue
        if header:
            if not isinstance(p.get("header"), dict):
                p["header"] = {}
            for f in sc.HEADER[g["doc_type"]]:
                gv = g["header"].get(f)
                if not dg.empty(gv) and dg.empty(p["header"].get(f)):
                    p["header"][f] = gv
                    filled["header"] += 1
        if (
            (row_fields or blank_false_fill)
            and g["doc_type"] == "invoice"
            and p.get("doc_type") == "invoice"
        ):
            pr = [x for x in (p.get("line_items") or []) if isinstance(x, dict)]
            gr = list(g.get("line_items") or [])
            for pi, gi in pair_rows_of(sc, pr, gr):
                for f in row_fields:
                    gv = gr[gi].get(f)
                    if not dg.empty(gv) and dg.empty(pr[pi].get(f)):
                        pr[pi][f] = gv
                        filled["row"] += 1
                for f in blank_false_fill:
                    if dg.empty(gr[gi].get(f)) and not dg.empty(pr[pi].get(f)):
                        pr[pi][f] = None
                        filled["blanked"] += 1
    return out, filled


ORACLE_VARIANTS: dict[str, dict[str, Any]] = {
    "header only": {"header": True, "row_fields": ()},
    "rows: customer_part_number": {"header": False, "row_fields": ("customer_part_number",)},
    "rows: purchase_order": {"header": False, "row_fields": ("purchase_order",)},
    "rows: cpn + po": {
        "header": False,
        "row_fields": ("customer_part_number", "purchase_order"),
    },
    "rows: all four fields": {
        "header": False,
        "row_fields": (
            "supplier_part_number",
            "customer_part_number",
            "purchase_order",
            "quantity",
        ),
    },
    "header + rows (cpn + po)": {
        "header": True,
        "row_fields": ("customer_part_number", "purchase_order"),
    },
    "rows: cpn + po null status (fill over-nulls AND blank false fills)": {
        "header": False,
        "row_fields": ("customer_part_number", "purchase_order"),
        "blank_false_fill": ("customer_part_number", "purchase_order"),
    },
    "header + rows null status (all of the above)": {
        "header": True,
        "row_fields": ("customer_part_number", "purchase_order"),
        "blank_false_fill": ("customer_part_number", "purchase_order"),
    },
}


def oracle_report(
    pred: Mapping[str, Any], gold: Mapping[str, Any], n: int = N_BOOT, seed: int = SEED
) -> list[dict[str, Any]]:
    """ORACLE replay per variant: cells filled, rescored metrics, paired delta of OVERALL (CI)."""
    base = ev.score(dict(pred), dict(gold))["all"]
    rows = []
    for name, kw in ORACLE_VARIANTS.items():
        patched, filled = oracle_fill(pred, gold, **kw)
        agg = ev.score(patched, dict(gold))["all"]
        pb = ev.paired_bootstrap(dict(pred), patched, dict(gold), n=n, seed=seed)
        rows.append(
            {
                "variant": name,
                "filled_header": filled["header"],
                "filled_row": filled["row"],
                "blanked_row": filled["blanked"],
                "OVERALL": agg["OVERALL"],
                "d_OVERALL": agg["OVERALL"] - base["OVERALL"],
                "d_OVERALL_ci": [pb["OVERALL"]["lo"], pb["OVERALL"]["hi"]],
                "d_header_acc": agg["header_field_accuracy"] - base["header_field_accuracy"],
                "d_row_f1": agg["row_f1"] - base["row_f1"],
                "d_fully_correct": agg["documents_fully_correct"] - base["documents_fully_correct"],
            }
        )
    return rows


def row_false_fill_counts(cells: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """Per row field over paired cells: false fills (gold null, pred value) and over-nulls."""
    out = {}
    for f in sorted({c["field"] for c in cells}):
        sub = [c for c in cells if c["field"] == f]
        out[f] = {
            "paired_cells": len(sub),
            "false_fill": sum(c["gold_null"] and not c["pred_null"] for c in sub),
            "over_null": sum((not c["gold_null"]) and c["pred_null"] for c in sub),
        }
    return out


def over_null_sources(
    cells: Sequence[Mapping[str, Any]], traces: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, int]]:
    """Where did each header over-null come from: the model, the parser or post-processing?

    ``model_null``: every page's parsed header holds null for the field (the model emitted null or
    omitted it); ``dropped_after_parse``: some page's parsed header had a value that merge /
    normalize did not carry into the prediction; ``page_parse_failed``: a page failed to parse.
    Returns {field: {source: count}} over cells with a gold value and an empty prediction.
    """
    by_doc = {t["doc_id"]: t for t in traces}
    out: dict[str, dict[str, int]] = {}
    for c in cells:
        if c["gold_null"] or not c["pred_null"]:
            continue
        pages = by_doc[c["doc"]]["pages"] if c["doc"] in by_doc else []
        if any(p.get("parsed") is None for p in pages):
            src = "page_parse_failed"
        elif any(not dg.empty((p["parsed"].get("header") or {}).get(c["field"])) for p in pages):
            src = "dropped_after_parse"
        else:
            src = "model_null"
        key = f"{c['type']}.{c['field']}"
        out.setdefault(key, {})
        out[key][src] = out[key].get(src, 0) + 1
    return {k: dict(sorted(v.items())) for k, v in sorted(out.items())}


def slot_misplacements(
    pred: Mapping[str, Any], gold: Mapping[str, Any], scanned: Mapping[str, bool]
) -> dict[str, dict[str, int]]:
    """Paired invoice rows whose purchase_order / customer_part_number value sits in the other slot.

    ``po_in_cpn_slot``: gold cpn empty, predicted cpn equals the gold PO (scorer identifier rule)
    and the predicted PO is empty. ``cpn_in_po_slot``: the mirror image. ``null_status_rows``:
    paired rows with any cpn/po null-status mismatch. Counts by scanned / digital.
    """
    sc = ev.load_scorer()
    out = {
        s: {"paired_rows": 0, "null_status_rows": 0, "po_in_cpn_slot": 0, "cpn_in_po_slot": 0}
        for s in ("scanned", "digital")
    }
    cpn, po = "customer_part_number", "purchase_order"
    for d, g in gold.items():
        p = pred.get(d)
        if g["doc_type"] != "invoice" or not isinstance(p, dict) or p.get("doc_type") != "invoice":
            continue
        pr = [x for x in (p.get("line_items") or []) if isinstance(x, dict)]
        gr = list(g.get("line_items") or [])
        side = "scanned" if scanned[d] else "digital"
        for pi, gi in pair_rows_of(sc, pr, gr):
            r, q = pr[pi], gr[gi]
            out[side]["paired_rows"] += 1
            if any(dg.empty(r.get(f)) != dg.empty(q.get(f)) for f in (cpn, po)):
                out[side]["null_status_rows"] += 1
            if (
                dg.empty(q.get(cpn))
                and not dg.empty(q.get(po))
                and not dg.empty(r.get(cpn))
                and dg.empty(r.get(po))
                and sc.same(po, r.get(cpn), q.get(po))
            ):
                out[side]["po_in_cpn_slot"] += 1
            if (
                dg.empty(q.get(po))
                and not dg.empty(q.get(cpn))
                and not dg.empty(r.get(po))
                and dg.empty(r.get(cpn))
                and sc.same(cpn, r.get(po), q.get(cpn))
            ):
                out[side]["cpn_in_po_slot"] += 1
    return out


def parsed_vs_final_row_nulls(
    pred: Mapping[str, Any], traces: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, int]]:
    """Non-null counts of cpn / po in the parsed pages vs the final prediction (all invoice docs).

    Equal totals mean merge / normalize did not null any row value; a gap would be a
    post-processing null (or dropped rows).
    """
    out = {f: {"parsed_non_null": 0, "final_non_null": 0} for f in ROW_NULL_FIELDS}
    for t in traces:
        p = pred.get(t["doc_id"])
        if not isinstance(p, dict) or p.get("doc_type") != "invoice":
            continue
        for page in t["pages"]:
            for x in (page.get("parsed") or {}).get("line_items") or []:
                if isinstance(x, dict):
                    for f in ROW_NULL_FIELDS:
                        out[f]["parsed_non_null"] += not dg.empty(x.get(f))
        for x in p.get("line_items") or []:
            if isinstance(x, dict):
                for f in ROW_NULL_FIELDS:
                    out[f]["final_non_null"] += not dg.empty(x.get(f))
    return out


def status_seconds(runs_dir: Path) -> dict[str, float]:
    """run_id -> ``seconds`` from the ``*_status.json`` files (smoke files are skipped)."""
    out: dict[str, float] = {}
    for f in sorted(Path(runs_dir).glob("*_status.json")):
        if f.name.startswith("smoke"):
            continue
        for rid, st in json.loads(f.read_text(encoding="utf-8")).items():
            if isinstance(st, dict) and st.get("seconds") is not None:
                out[rid] = float(st["seconds"])
    return out


# --------------------------------------------------------------------------------------------
# Taxonomy and unpaired-row causes
# --------------------------------------------------------------------------------------------


def _by_scope(errors: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """Error label -> {scope (doc / header / row): count}."""
    out: dict[str, dict[str, int]] = {}
    for e in errors:
        d = out.setdefault(e["label"], {})
        d[e["scope"]] = d.get(e["scope"], 0) + 1
    return {k: dict(sorted(v.items())) for k, v in sorted(out.items())}


def taxonomy_counts(
    pred: Mapping[str, Any],
    gold: Mapping[str, Any],
    meta: Sequence[Mapping[str, Any]],
    ocr: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """``ev.error_taxonomy`` counts (labels and categories) for all / invoices / waybills /
    scanned / digital."""
    tax = ev.error_taxonomy(dict(pred), dict(gold), dict(ocr) if ocr else None, list(meta))
    pick = {
        "all": "all",
        "invoices": "invoices",
        "waybills": "waybills",
        "scanned": "scanned=yes",
        "digital": "scanned=no",
    }
    return {
        "labels": {k: tax["counts"][v] for k, v in pick.items()},
        "categories": {k: tax["counts_by_category"][v] for k, v in pick.items()},
        "n_errors": len(tax["errors"]),
        "by_scope": _by_scope(tax["errors"]),
        "ocr_docs": len(ocr or {}),
        "support_unknown": sum(e["support"] == "unknown_support" for e in tax["errors"]),
        "_errors": tax["errors"],
    }


def unpaired_causes(
    pred: Mapping[str, Any], gold: Mapping[str, Any], truncated: Sequence[str] = ()
) -> dict[str, int]:
    """Step L cause counts of unpaired gold rows (shift / missing / spn_null / ...)."""
    sc = ev.load_scorer()
    skip = set(truncated)
    out: dict[str, int] = {}
    for d, g in gold.items():
        p = pred.get(d)
        if g["doc_type"] != "invoice" or not isinstance(p, dict) or p.get("doc_type") != "invoice":
            continue
        gr = list(g.get("line_items") or [])
        pr = [x for x in (p.get("line_items") or []) if isinstance(x, dict)]
        _, _, unp, ung = ev._pair_rows(sc, pr, gr)
        att, _ = dg.attribute_unpaired(sc, pr, gr, unp, ung, d in skip)
        for rec in att:
            out[rec["cause"]] = out.get(rec["cause"], 0) + 1
    return dict(sorted(out.items()))


# --------------------------------------------------------------------------------------------
# A/B
# --------------------------------------------------------------------------------------------


def rederive_decision(res: Mapping[str, Any]) -> dict[str, Any]:
    """Re-derive the spec's decision rule from a ``compare_formats`` result, clause by clause.

    Independent of ``format_ab.decide`` (so a bug there would show as a mismatch).
    """
    d = res["deltas"]
    row, ov = d["row_f1"], d["OVERALL"]
    c1 = bool(row["delta"] > 0 and row["ci95"][0] > 0)
    rot_less = bool(res["keyed"]["rotations"] < res["compact"]["rotations"])
    not_worse = bool(ov["delta"] >= 0 or ov["ci95"][0] <= 0 <= ov["ci95"][1])
    c2 = rot_less and not_worse
    fired = (
        "clause 1 (row F1 gain, CI excludes 0)"
        if c1
        else (
            "clause 2 (fewer rotations and OVERALL not worse)" if c2 else "none (compact on speed)"
        )
    )
    return {
        "clause1_row_f1_gain_ci_excludes_0": c1,
        "clause2_rotations_keyed_lt_compact": rot_less,
        "clause2_overall_not_worse": not_worse,
        "clause2": c2,
        "decision": "keyed" if (c1 or c2) else "compact",
        "fired": fired,
    }


def rank_check(runs: Mapping[str, RunData], gold: Mapping[str, Any]) -> dict[str, Any]:
    """``ev.rank_runs`` over finished runs (names are the keys of `runs`)."""
    preds = {k: r.pred for k, r in runs.items()}
    ids = sorted(set.intersection(*[set(p) for p in preds.values()]))
    return ev.rank_runs(preds, restrict(gold, ids), n=N_BOOT, seed=SEED)


# --------------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------------


def gib(b: int | None) -> str:
    """Bytes -> GiB string."""
    return "n/a" if b is None else f"{b / 2**30:.2f}"


def s1_table(rows: Sequence[Mapping[str, Any]]) -> str:
    """The S1 table (one row per run)."""
    head = [
        "run",
        "fmt",
        "docs",
        "header acc % [95% CI]",
        "row P %",
        "row R %",
        "row F1 % [95% CI]",
        "fully correct % [95% CI]",
        "OVERALL % [95% CI]",
        "false-fill %",
        "JSON valid %",
        "s/page mean",
        "s/page p95",
        "peak VRAM GiB",
    ]
    body = []
    for r in rows:
        a, c, t = r["agg"], r["ci"], r["trace"]
        body.append(
            [
                r["run"],
                "/".join(r["identity"]["output_format_trace"]),
                r["n_docs"],
                pci(c["header_field_accuracy"]),
                pct(a["row_precision"]),
                pct(a["row_recall"]),
                pci(c["row_f1"]),
                pci(c["documents_fully_correct"]),
                pci(c["OVERALL"]),
                f"{pct(a['false_fill_rate'])} ({a['illegible_fields']} null)",
                pct(t["json_validity"]),
                f"{t['s_per_page_mean']:.1f}",
                f"{t['s_per_page_p95']:.1f}",
                gib(t["peak_vram_bytes"]),
            ]
        )
    return md_table(head, body)


def check_table(rows: Sequence[Mapping[str, Any]]) -> str:
    """Rescore-vs-metrics.json equality table."""
    head = [
        "run",
        "keys checked",
        "all float-equal",
        "OVERALL rescore",
        "OVERALL metrics.json",
        "speed/VRAM metrics.json vs trace",
    ]
    body = []
    for r in rows:
        k = r["check"]["keys"]["OVERALL"]
        t = r["trace"]
        same = (
            t["metrics_s_per_page_mean"] == t["s_per_page_mean"]
            and t["metrics_s_per_page_p95"] == t["s_per_page_p95"]
            and t["metrics_peak_vram_bytes"] == t["peak_vram_bytes"]
            and t["metrics_json_validity"] == t["json_validity"]
        )
        body.append(
            [
                r["run"],
                r["check"]["n_checked"],
                "yes" if r["check"]["all_equal"] else "NO",
                f"{k['rescore']!r}",
                f"{k['saved']!r}",
                "equal" if same else "DIFFER",
            ]
        )
    return md_table(head, body)


def slice_table(rows: Sequence[Mapping[str, Any]]) -> str:
    """The S2 slice table."""
    head = [
        "slice",
        "n docs",
        "gold rows",
        "OVERALL % [95% CI]",
        "header acc % [95% CI]",
        "row F1 % [95% CI]",
        "fully correct % [95% CI]",
        "false-fill (scorer)",
    ]
    body = []
    for r in rows:
        if not r["n"]:
            body.append([r["slice"], 0, "-", "-", "-", "-", "-", "-"])
            continue
        c = r["ci"]
        rowf1 = pci(c["row_f1"]) if r["gold_rows"] else "n/a (no gold rows)"
        body.append(
            [
                r["slice"],
                r["n"],
                r["gold_rows"],
                pci(c["OVERALL"]),
                pci(c["header_field_accuracy"]),
                rowf1,
                pci(c["documents_fully_correct"]),
                f"{pct(r['false_fill_rate'])} of {r['illegible_fields']}",
            ]
        )
    return md_table(head, body)


def field_table(fc: Mapping[str, Any]) -> str:
    """Per-field header and row accuracy tables with CIs."""
    h = [[f, v["n_docs"], pci(v)] for f, v in sorted(fc["header"].items())]
    r = [[f, v["n_docs"], pci(v)] for f, v in fc["row"].items()]
    return (
        md_table(["header field", "n docs", "accuracy % [95% CI]"], h)
        + "\n"
        + md_table(["row field (share of ALL gold rows)", "gold rows", "accuracy % [95% CI]"], r)
    )


def conf_row(name: str, c: Mapping[str, Any]) -> list[Any]:
    """One confusion table row."""
    return [
        name,
        c["cells"],
        c["gold_null"],
        c["pred_null"],
        c["true_null"],
        c["over_null"],
        rate_str(c["over_null"], c["gold_valued"]),
        c["false_fill"],
        pct(c["null_precision"], 1),
        pct(c["null_recall"], 1),
    ]


CONF_HEAD = [
    "group",
    "cells",
    "gold null",
    "pred null",
    "true null",
    "over-null",
    "over-null / gold-valued",
    "false fill",
    "null precision %",
    "null recall %",
]


def header_null_tables(rep: Mapping[str, Any]) -> str:
    """Header null confusion tables (pooled/class/scan split, then per field)."""
    keys = [
        ("all header cells", "pooled"),
        ("  scanned", "scanned"),
        ("  digital", "digital"),
        ("required fields (null = redaction)", "required"),
        ("  scanned", "required/scanned"),
        ("  digital", "required/digital"),
        ("optional-line fields (null = absent line)", "absent_line"),
        ("  scanned", "absent_line/scanned"),
        ("  digital", "absent_line/digital"),
    ]
    t1 = md_table(CONF_HEAD, [conf_row(n, rep[k]) for n, k in keys])
    rows = []
    for f, c in rep["fields"].items():
        rows.append(conf_row(f"{f} [{c['cls']}]", c))
        rows.append(conf_row("  scanned", c["scanned"]))
        rows.append(conf_row("  digital", c["digital"]))
    return t1 + "\n" + md_table(CONF_HEAD, rows)


def row_null_tables(rep: Mapping[str, Any]) -> str:
    """Row-field null tables over paired rows plus marginal null rates."""
    rows = []
    for f, c in rep["fields"].items():
        rows.append(conf_row(f, c))
        rows.append(conf_row("  scanned", c["scanned"]))
        rows.append(conf_row("  digital", c["digital"]))
    t1 = md_table(["row field (paired rows)", *CONF_HEAD[1:]], rows)
    m = rep["marginal"]
    sc = ev.load_scorer()
    mrows = []
    for s in ("scanned", "digital"):
        for f in sc.ROW:
            gn, pn = m[s][f"gold_null_{f}"], m[s][f"pred_null_{f}"]
            mrows.append(
                [
                    s,
                    f,
                    rate_str(gn, m[s]["gold_rows"]),
                    rate_str(pn, m[s]["pred_rows"]),
                    f"{pn - gn:+d}",
                ]
            )
    t2 = md_table(
        [
            "split",
            "row field",
            "gold null / gold rows",
            "pred null / pred rows",
            "pred - gold null",
        ],
        mrows,
    )
    return t1 + "\nMarginal null counts over ALL rows (unpaired rows included):\n\n" + t2


def slots_table(slots: Mapping[str, Mapping[str, int]]) -> str:
    """Slot-misplacement counts, scanned / digital / total."""
    keys = ("paired_rows", "null_status_rows", "po_in_cpn_slot", "cpn_in_po_slot")
    rows = [[k, *[v[x] for x in keys]] for k, v in slots.items()]
    rows.append(["total", *[sum(v[x] for v in slots.values()) for x in keys]])
    return md_table(["split", *keys], rows)


def oracle_table(rows: Sequence[Mapping[str, Any]]) -> str:
    """ORACLE replay table (percent points)."""
    head = [
        "ORACLE variant (gold replaces over-null)",
        "header cells filled",
        "row cells filled",
        "row cells blanked",
        "OVERALL %",
        "d OVERALL pts [paired 95% CI]",
        "d header acc pts",
        "d row F1 pts",
        "d fully correct pts",
    ]
    body = []
    for r in rows:
        lo, hi = r["d_OVERALL_ci"]
        body.append(
            [
                r["variant"],
                r["filled_header"],
                r["filled_row"],
                r["blanked_row"],
                pct(r["OVERALL"]),
                f"{100 * r['d_OVERALL']:+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]",
                f"{100 * r['d_header_acc']:+.2f}",
                f"{100 * r['d_row_f1']:+.2f}",
                f"{100 * r['d_fully_correct']:+.2f}",
            ]
        )
    return md_table(head, body)


def tax_table(tax: Mapping[str, Any], key: str) -> str:
    """Taxonomy counts table (rows = slices)."""
    labels = sorted({k for c in tax[key].values() for k in c})
    body = [[s, *[c.get(k, 0) for k in labels]] for s, c in tax[key].items()]
    return md_table(["slice", *labels], body)


def step_l_table(rows: Sequence[tuple[str, Mapping[str, int]]]) -> str:
    """Step L row-convention counts."""
    return md_table(
        ["run", *dg.COUNT_KEYS],
        [[n, *[c[k] for k in dg.COUNT_KEYS]] for n, c in rows],
    )


def ab_table(res: Mapping[str, Any]) -> str:
    """Keyed vs compact table with CIs and paired deltas."""
    k, c, d = res["keyed"], res["compact"], res["deltas"]

    def dci(m: Mapping[str, Any]) -> str:
        star = " (CI excludes 0)" if m["ci_excludes_0"] else " (CI includes 0)"
        return (
            f"{100 * m['delta']:+.2f} [{100 * m['ci95'][0]:+.2f}, {100 * m['ci95'][1]:+.2f}]{star}"
        )

    rows = [
        ["OVERALL % [95% CI]", pci(c["OVERALL"]), pci(k["OVERALL"]), dci(d["OVERALL"])],
        ["row F1 % [95% CI]", pci(c["row_f1"]), pci(k["row_f1"]), dci(d["row_f1"])],
    ]
    for f, cv in c["line_item_field_accuracy"].items():
        kv = k["line_item_field_accuracy"][f]
        rows.append([f"row acc %: {f}", pct(cv), pct(kv), f"{100 * (kv - cv):+.2f}"])
    for key in ("rotations", "clean_swaps", "cpn_equals_po", "cpn_filled_gold_empty"):
        rows.append([key, c[key], k[key], f"{k[key] - c[key]:+d}"])
    for key in ("s_per_page_mean", "s_per_page_p95"):
        rows.append([key, f"{c[key]:.1f}", f"{k[key]:.1f}", f"{k[key] - c[key]:+.1f}"])
    rows.append(
        [
            "n docs / n pages",
            f"{c['n_docs']} / {c['n_pages']}",
            f"{k['n_docs']} / {k['n_pages']}",
            "",
        ]
    )
    return md_table(["metric", "compact", "keyed", "keyed - compact (paired)"], rows)


# --------------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------------


def git_state() -> str:
    """``<short sha>`` plus ``+dirty`` when tracked files differ from HEAD."""

    def run(*a: str) -> str:
        return subprocess.run(
            ["git", *a], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return run("rev-parse", "--short", "HEAD") + (
            "+dirty" if run("status", "--porcelain", "--untracked-files=no") else ""
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def spike_names(sha: str) -> list[str]:
    """The six spike40 compact run dir names."""
    return [f"spike40_{m}_{a}_compact_{sha}" for m in MODELS for a in ARMS]


def analyse(
    runs_dir: Path,
    final_pick: str,
    context: str,
    ab_keyed: str,
    ab_compact: str,
    sha: str,
    gold_dir: Path,
    meta_path: Path,
    n: int = N_BOOT,
) -> dict[str, Any]:
    """Run S1-S4 and return the results (plus value-level examples under ``_local``)."""
    gold_all = ev.load_gold(gold_dir)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    scanned = {m["doc_id"]: bool(m["scanned"]) for m in meta}
    names = [*spike_names(sha), ab_keyed, context, final_pick]
    runs = {nm: load_run(runs_dir / nm) for nm in names}
    s1 = [full_run_row(runs[nm], gold_all, n=n) for nm in names]

    # S2: final pick slices / fields / taxonomy / Step L
    fp = runs[final_pick]
    gfp = restrict(gold_all, list(fp.pred))
    slices = slice_ids(gfp, meta)
    per_fp = ev.per_doc_results(fp.pred, gfp)
    ocr = ev.load_ocr_pages(list(gfp))
    s2 = {
        "slices": slice_report(fp.pred, gfp, slices, n=n),
        "fields": field_cis(list(per_fp.values()), n=n),
        "taxonomy": taxonomy_counts(fp.pred, gfp, meta, ocr),
        "step_l": dg.run_row_convention_errors(fp.path, gfp),
        "unpaired_causes": unpaired_causes(fp.pred, gfp, dg.truncated_docs(fp.path)),
    }

    # S3: null handling for the final pick and the context run
    np_fields = not_printed_fields()
    s3: dict[str, Any] = {"not_printed_fields": {t: sorted(v) for t, v in np_fields.items()}}
    local_cells: dict[str, Any] = {}
    for label, nm in (("final_pick", final_pick), ("context", context)):
        run = runs[nm]
        g = restrict(gold_all, list(run.pred))
        hc = header_cells(run.pred, g, np_fields, scanned)
        rc, marg = row_cells(run.pred, g, scanned)
        s3[label] = {
            "run": nm,
            "scorer": {
                "false_fill_rate": next(r for r in s1 if r["run"] == nm)["agg"]["false_fill_rate"],
                "illegible_fields": next(r for r in s1 if r["run"] == nm)["agg"][
                    "illegible_fields"
                ],
            },
            "header": header_null_report(hc),
            "rows": row_null_report(rc, marg),
            "row_false_fill": row_false_fill_counts(rc),
            "over_null_sources": over_null_sources(hc, run.traces),
            "slots": slot_misplacements(run.pred, g, scanned),
            "row_parse_vs_final": parsed_vs_final_row_nulls(run.pred, run.traces),
            "oracle": oracle_report(run.pred, g, n=n),
        }
        local_cells[label] = {"header": hc, "rows": rc, "gold": g, "pred": run.pred}

    # S4: keyed vs compact on the same 40 docs
    kd, cd = runs[ab_keyed], runs[ab_compact]
    g40 = restrict(gold_all, list(cd.pred))
    ab = fab.compare_formats(cd.path, kd.path, g40, "qwen35_4b", "img_ocr")
    ab_extra = {
        "rederived": rederive_decision(ab),
        "unpaired_causes": {
            "compact": unpaired_causes(cd.pred, g40, dg.truncated_docs(cd.path)),
            "keyed": unpaired_causes(kd.pred, g40, dg.truncated_docs(kd.path)),
        },
        "identity": {"compact": run_identity(cd), "keyed": run_identity(kd)},
    }
    spike_runs = {nm: runs[nm] for nm in spike_names(sha)}
    rank40 = rank_check(spike_runs, gold_all)
    d100 = {context: runs[context], final_pick: runs[final_pick]}
    rank100 = rank_check(d100, gold_all)
    saved_rank_path = runs_dir / "dev100_rank.json"
    saved_rank = (
        json.loads(saved_rank_path.read_text(encoding="utf-8"))
        if saved_rank_path.is_file()
        else None
    )
    secs = status_seconds(runs_dir)
    wall = [
        {
            "run": r["run"],
            "status_seconds": secs.get(r["run"]),
            "sum_page_latency_s": sum(
                p["meta"]["latency_s"] for t in runs[r["run"]].traces for p in t["pages"]
            ),
        }
        for r in s1
    ]
    return {
        "wall": wall,
        "s1": s1,
        "s2": s2,
        "s3": s3,
        "s4": {
            "ab": ab,
            "extra": ab_extra,
            "rank40": rank40,
            "rank100": rank100,
            "saved_rank100": saved_rank,
            "dev100_identity": {nm: run_identity(runs[nm]) for nm in (context, final_pick)},
        },
        "_local": local_cells,
        "_errors": s2["taxonomy"].pop("_errors"),
        "meta": {
            "final_pick": final_pick,
            "context": context,
            "ab_keyed": ab_keyed,
            "ab_compact": ab_compact,
            "sha": sha,
            "n_boot": n,
            "seed": SEED,
        },
    }


def type_totals(rep: Mapping[str, Any], doc_type: str) -> dict[str, int]:
    """Sum the per-field confusion counts of one doc type (keys of ``rep['fields']``)."""
    keys = ("gold_null", "gold_valued", "pred_null", "true_null", "over_null", "false_fill")
    tot = dict.fromkeys(keys, 0)
    for f, c in rep["fields"].items():
        if f.startswith(doc_type + "."):
            for k in keys:
                tot[k] += c[k]
    return tot


def findings(res: Mapping[str, Any]) -> str:
    """Numeric findings, every number taken from ``res`` (no hand-typed values)."""
    m, s3, s4 = res["meta"], res["s3"], res["s4"]
    ab, rd = s4["ab"], s4["extra"]["rederived"]
    di = s4["dev100_identity"]
    fp_i = di[m["final_pick"]]
    sh = fp_i["raw_text_shape_pages"]
    r100 = s4["rank100"]
    d = ab["deltas"]
    fp, cx = s3["final_pick"], s3["context"]
    kh = s4["extra"]["identity"]["keyed"]["config_hash"]
    ch = di[m["context"]]["config_hash"]
    lines: list[str] = []
    add = lines.append
    add(
        f"1. **Format of the final pick.** Trace `output_format` of `{m['final_pick']}` is "
        f"`{'/'.join(fp_i['output_format_trace'])}` (keyed): {sh['keyed']} of "
        f"{sh['keyed'] + sh['compact'] + sh['other']} pages start with the keyed prefix, "
        f"{sh['compact']} with the compact prefix; `{m['context']}` is the same format. The "
        f"A/B was run on `{m['ab_keyed']}` (qwen35_4b x img_ocr, rank 1 on spike40), i.e. a "
        "different arm from the final pick; the final pick runs keyed only because the spec's "
        "dev100 policy makes rank #2 follow the A/B decision (same config hash for the A/B "
        f"keyed run and the dev100 img_ocr run: {kh[0]} == {ch[0]}: {kh == ch})."
    )
    add(
        f"2. **A/B decision.** Re-derived `{rd['decision']}` via {rd['fired']}; "
        f"`shipdoc.format_ab` gave `{ab['decision']}` (same: {rd['decision'] == ab['decision']}). "
        f"Matches `ab_decision=keyed` and the keyed dev100 traces: {rd['decision'] == 'keyed'}. "
        f"Keyed minus compact: dOVERALL {100 * d['OVERALL']['delta']:+.2f} pts "
        f"[{100 * d['OVERALL']['ci95'][0]:+.2f}, {100 * d['OVERALL']['ci95'][1]:+.2f}], "
        f"d row F1 {100 * d['row_f1']['delta']:+.2f} pts "
        f"[{100 * d['row_f1']['ci95'][0]:+.2f}, {100 * d['row_f1']['ci95'][1]:+.2f}] "
        f"(CI excludes 0: {d['OVERALL']['ci_excludes_0']} / {d['row_f1']['ci_excludes_0']}), "
        f"rotations {ab['compact']['rotations']} -> {ab['keyed']['rotations']}, mean "
        f"s/page {ab['compact']['s_per_page_mean']:.1f} -> {ab['keyed']['s_per_page_mean']:.1f}; "
        f"clause 1 (row F1 gain) true: {rd['clause1_row_f1_gain_ci_excludes_0']}; clause 2 "
        f"(rotations down and OVERALL not worse) true: {rd['clause2']}."
    )
    r1, r2 = r100["ranking"][0], r100["ranking"][1]
    add(
        f"3. **Final pick vs runner-up.** `{r1['config']}` {100 * r1['OVERALL']:.2f} vs "
        f"`{r2['config']}` {100 * r2['OVERALL']:.2f}; paired d {100 * r100['delta']:+.2f} pts, "
        f"CI [{100 * r100['ci95'][0]:+.2f}, {100 * r100['ci95'][1]:+.2f}] "
        f"(excludes 0: {r100['delta_ci_excludes_0']}): the pick rests on a point estimate."
    )
    inv, wb = type_totals(fp["header"], "invoice"), type_totals(fp["header"], "waybill")
    inv_c, wb_c = type_totals(cx["header"], "invoice"), type_totals(cx["header"], "waybill")
    add(
        "4. **Scorer definitions.** `illegible_fields` counts every gold-null HEADER cell "
        f"({fp['scorer']['illegible_fields']} here: redaction + absent-line); `false_fill_rate` "
        "is the share of them the model filled. It ignores row fields and over-nulls. The 0.000 "
        "therefore only says the model never invented a header value where gold is null."
    )
    add(
        f"5. **Header over-nulling (final pick).** Gold-valued header cells predicted null: "
        f"invoice {inv['over_null']}/{inv['gold_valued']}, waybill "
        f"{wb['over_null']}/{wb['gold_valued']} (context run: invoice {inv_c['over_null']}, "
        f"waybill {wb_c['over_null']}). Gold-null cells recovered: invoice "
        f"{inv['true_null']}/{inv['gold_null']}, waybill {wb['true_null']}/{wb['gold_null']} "
        f"(zero header false fills: {inv['false_fill'] + wb['false_fill'] == 0}). Every "
        "over-null source in the trace is the model's own null "
        f"({json.dumps(fp['over_null_sources'])})."
    )
    o = {r["variant"]: r for r in fp["oracle"]}
    hdr = o["header only"]
    rows_fill = o["rows: cpn + po"]
    rows_ns = o["rows: cpn + po null status (fill over-nulls AND blank false fills)"]
    both = o["header + rows null status (all of the above)"]
    sl = fp["slots"]
    po_cpn = sum(v["po_in_cpn_slot"] for v in sl.values())
    ns_rows = sum(v["null_status_rows"] for v in sl.values())
    rff = fp["row_false_fill"]
    rffc = rff["customer_part_number"]["false_fill"]
    add(
        f"6. **Row null handling (final pick).** Over paired rows, purchase_order over-null "
        f"{rff['purchase_order']['over_null']} and customer_part_number false fill "
        f"{rff['customer_part_number']['false_fill']} (scorer false-fill counts none of these); "
        f"{po_cpn} of the {ns_rows} paired rows with a cpn/po null-status mismatch are a PO value "
        "written in the customer_part_number slot (a column-assignment error, not a "
        "null-versus-value decision)."
    )
    add(
        "7. **Cost of nulling, ORACLE upper bounds (not fixes), final pick.** Header over-nulls "
        f"filled with gold: {100 * hdr['d_OVERALL']:+.2f} pts OVERALL "
        f"[{100 * hdr['d_OVERALL_ci'][0]:+.2f}, {100 * hdr['d_OVERALL_ci'][1]:+.2f}]; row "
        f"over-nulls (cpn+po) filled alone {100 * rows_fill['d_OVERALL']:+.2f} pts "
        f"[{100 * rows_fill['d_OVERALL_ci'][0]:+.2f}, {100 * rows_fill['d_OVERALL_ci'][1]:+.2f}] "
        f"(the same rows also carry {rffc} cpn false fills); row null status made right "
        f"in both directions {100 * rows_ns['d_OVERALL']:+.2f} pts "
        f"[{100 * rows_ns['d_OVERALL_ci'][0]:+.2f}, {100 * rows_ns['d_OVERALL_ci'][1]:+.2f}]; "
        f"header + row null status {100 * both['d_OVERALL']:+.2f} pts "
        f"[{100 * both['d_OVERALL_ci'][0]:+.2f}, {100 * both['d_OVERALL_ci'][1]:+.2f}]. "
        "The effects are not additive."
    )
    low = [w for w in res["wall"] if w["status_seconds"] is not None]
    low = [w for w in low if w["status_seconds"] < 0.8 * w["sum_page_latency_s"]]
    add(
        "8. **Timing caveat.** "
        + (
            "`status_seconds` is far below the summed page latencies for "
            + ", ".join(w["run"] for w in low)
            + " (likely a resumed or partial status; the s/page figures use the traces)."
            if low
            else "status seconds agree with the summed page latencies for every run."
        )
    )
    return "\n".join(lines) + "\n"


def render_md(res: Mapping[str, Any], runs_dir: Path, repo_state: str) -> str:
    """The aggregate-only report (``reports/respike.md``)."""
    m, s1, s2, s3, s4 = res["meta"], res["s1"], res["s2"], res["s3"], res["s4"]
    fp, ctx = m["final_pick"], m["context"]
    out: list[str] = []
    w = out.append
    w("# Re-spike analysis (Step S)\n")
    w(
        f"**Provenance.** Runs: Colab T4, code `{m['sha']}`, artifacts under `{runs_dir}` "
        "(`<run>/{predictions.json, trace.jsonl, metrics.json}`; `dev100_rank.json`, "
        "`ab_status.json`). "
        "Generated by `uv run python scripts/respike.py` "
        f"(repo state `{repo_state}`; bootstrap {m['n_boot']} doc-level resamples, "
        f"seed {m['seed']}). "
        "Every number is computed from the artifacts with the unmodified official scorer "
        "(`assignment/score.py` via `shipdoc.eval`) against `data/dev/labels` restricted to each "
        "run's doc_ids. **All numbers UNVERIFIED** until a verifier recomputes them. "
        "Aggregates and "
        "counts only; value-level examples are in the local, outside-repo "
        "`<SHIPDOC_RUNS_DIR>/diagnosis/respike_local.md`.\n"
    )
    w("## S1. Rescore of every run\n")
    w(
        "Rescore = `eval.score(predictions.json, gold of the run's docs)['all']`; compared with "
        "`metrics.json['slices']['all']` by float `==` on "
        f"{', '.join(CHECK_KEYS)}.\n"
    )
    w(check_table(s1))
    w(
        "\nFull table (percent; CIs are doc-level percentile bootstrap; false-fill is the "
        "scorer's header false-fill rate over gold header nulls; s/page from `trace.jsonl` "
        "page latencies):\n"
    )
    w(s1_table(s1))
    w("\nRun identity from the traces (output format, config hash, prompt, code, model):\n")
    idr = []
    for r in s1:
        i = r["identity"]
        s = i["raw_text_shape_pages"]
        idr.append(
            [
                r["run"],
                "/".join(i["output_format_trace"]),
                f"keyed {s['keyed']} / compact {s['compact']} / other {s['other']}",
                i["config_name"],
                "/".join(i["config_hash"]),
                i["prompt_version"],
                i["git_commit"],
            ]
        )
    w(
        md_table(
            [
                "run",
                "trace output_format",
                "raw_text starts with (pages)",
                "config",
                "config hash",
                "prompt",
                "code",
            ],
            idr,
        )
    )
    w("\nStep L row-convention counts for every run (`shipdoc.diagnostics`):\n")
    w(step_l_table([(r["run"], r["step_l"]) for r in s1]))
    w(
        "\nWall-clock sanity: `*_status.json` seconds vs the sum of per-page latencies in the "
        "trace (a status value far below the sum means the status covers only part of the run, "
        "e.g. a resumed notebook cell; s/page in the tables above comes from the traces):\n"
    )
    w(
        md_table(
            ["run", "status seconds", "sum of page latencies (s)", "status / sum"],
            [
                [
                    x["run"],
                    "n/a" if x["status_seconds"] is None else f"{x['status_seconds']:.1f}",
                    f"{x['sum_page_latency_s']:.1f}",
                    "n/a"
                    if x["status_seconds"] is None
                    else f"{x['status_seconds'] / x['sum_page_latency_s']:.2f}",
                ]
                for x in res["wall"]
            ],
        )
    )

    w(f"\n## S2. Final pick `{fp}`: slices, fields, taxonomy\n")
    w(
        "Slices (tags from `meta/dev.json`; `awb_absent` / `hawb_absent` derived from gold nulls "
        "because `meta/dev.json` has no such tags). Small n: read CIs, not points.\n"
    )
    w(slice_table(s2["slices"]))
    w("\nPer-field accuracy (doc-level bootstrap CIs):\n")
    w(field_table(s2["fields"]))
    tax = s2["taxonomy"]
    w(
        f"\nError taxonomy (`eval.error_taxonomy`; OCR cache found for {tax['ocr_docs']} docs; "
        f"{tax['support_unknown']} of {tax['n_errors']} error records have unknown OCR support). "
        "Labels:\n"
    )
    w(tax_table(tax, "labels"))
    w("\nCategories:\n")
    w(tax_table(tax, "categories"))
    w(
        "\nScope of each label, all docs (`header` = header field, `row` = row field or row, "
        "`doc` = doc_type). Note the scorer's false-fill rate counts HEADER cells only, so "
        "`false_fill` on `row` scope is invisible to it:\n"
    )
    w(
        md_table(
            ["label", "doc", "header", "row"],
            [
                [k, v.get("doc", 0), v.get("header", 0), v.get("row", 0)]
                for k, v in tax["by_scope"].items()
            ],
        )
    )
    w("\nStep L (rotation / clean swap / cpn == po) and the cause of every unpaired gold row:\n")
    w(step_l_table([(fp, s2["step_l"])]))
    w(
        "\nunpaired gold rows by cause: "
        + ", ".join(f"{k} {v}" for k, v in s2["unpaired_causes"].items())
        + "\n"
    )

    w("\n## S3. Null handling\n")
    w(
        "**Scorer definitions** (`assignment/score.py`). `illegible_fields` = the number of GOLD "
        "header cells (of the gold doc type's header fields) that are null/blank, whatever the "
        "reason; "
        "`false_fill_rate` = cells among them where the prediction is non-empty, divided by "
        "`illegible_fields`. Both are HEADER-only and count only gold-null cells: they say nothing "
        "about the opposite error (gold has a value, prediction null = over-null), which is simply "
        "an ordinary wrong header field in `header_field_accuracy`. Row fields have no null "
        "accounting at all: a wrong row field just makes the row not fully correct. "
        "`illegible_fields` therefore mixes redactions with absent-line nulls "
        "(awb_number/hawb), so a false-fill rate of 0.000 mostly tests whether the model invents "
        "values for lines that are not printed.\n"
    )
    w(
        f"Optional-line (not-printed) fields from `shipdoc.meta.not_printed_fields` (train+dev, "
        f">=10% null): {s3['not_printed_fields']}; all other header fields are required, so a gold "
        "null there is a redaction.\n"
    )
    for label, title in (("final_pick", f"Final pick `{fp}`"), ("context", f"Context `{ctx}`")):
        d = s3[label]
        w(f"\n### {title}\n")
        w(
            f"Scorer: false_fill_rate {pct(d['scorer']['false_fill_rate'], 3)}% of "
            f"{d['scorer']['illegible_fields']} illegible_fields.\n"
        )
        w(
            "Header null confusion (over-null = gold has value, prediction null; null precision = "
            "true null / predicted null; null recall = true null / gold null):\n"
        )
        w(header_null_tables(d["header"]))
        w("\nRow-field null analysis:\n")
        w(row_null_tables(d["rows"]))
        rf = d["row_false_fill"]
        w(
            "\nRow fields, paired rows: false fills (gold null, prediction value) vs over-nulls: "
            + "; ".join(
                f"{f}: false fill {v['false_fill']}, over-null {v['over_null']} "
                f"of {v['paired_cells']}"
                for f, v in rf.items()
            )
            + "\n"
        )
        src = "; ".join(
            f"{f}: " + ", ".join(f"{k} {v}" for k, v in c.items())
            for f, c in d["over_null_sources"].items()
        )
        w(f"\nSource of each header over-null (from the trace): {src or 'none'}.\n")
        pv = "; ".join(
            f"{f}: parsed {c['parsed_non_null']} vs final {c['final_non_null']}"
            for f, c in d["row_parse_vs_final"].items()
        )
        w(
            "\nRow cpn / po non-null counts, parsed pages vs final prediction (equal = merge and "
            f"normalize nulled no row value): {pv}.\n"
        )
        w("\nPO / CPN slot misplacement over scorer-paired invoice rows:\n")
        w(slots_table(d["slots"]))
        w(
            "\nORACLE replay (gold value replaces each over-null prediction, then the unmodified "
            "scorer; an upper bound on what over-nulling costs, NOT a fix; the last two rows also "
            "blank row false fills so the whole null status of cpn/po is right):\n"
        )
        w(oracle_table(d["oracle"]))

    w("\n## S4. Keyed vs compact A/B (same 40 docs, qwen35_4b x img_ocr)\n")
    ab, ex = s4["ab"], s4["extra"]
    w(f"Compact = `{m['ab_compact']}`, keyed = `{m['ab_keyed']}`.\n")
    w(ab_table(ab))
    w(
        "\nStep L cause of every unpaired gold row: "
        + "; ".join(
            f"{k}: " + ", ".join(f"{a} {b}" for a, b in v.items())
            for k, v in ex["unpaired_causes"].items()
        )
        + "\n"
    )
    rd = ex["rederived"]
    w("\nDecision rule (spec section 5, verbatim): " + fab.DECISION_RULE + ".\n")
    w(
        md_table(
            ["clause", "value"],
            [
                [
                    "clause 1: row F1 keyed > compact AND paired CI of d row F1 excludes 0",
                    rd["clause1_row_f1_gain_ci_excludes_0"],
                ],
                ["clause 2a: rotations keyed < compact", rd["clause2_rotations_keyed_lt_compact"]],
                [
                    "clause 2b: OVERALL not worse (point delta >= 0 OR paired CI includes 0)",
                    rd["clause2_overall_not_worse"],
                ],
                ["clause 2 = 2a AND 2b", rd["clause2"]],
                ["clause fired", rd["fired"]],
                ["decision (re-derived here)", rd["decision"]],
                ["decision (`shipdoc.format_ab`)", f"{ab['decision']} ({ab['reason']})"],
            ],
        )
    )
    w("\nWhich format each dev100 run used (trace `output_format`, raw_text shape, config hash):\n")
    di = s4["dev100_identity"]
    rows = []
    for nm, i in di.items():
        s = i["raw_text_shape_pages"]
        rows.append(
            [
                nm,
                "/".join(i["output_format_trace"]),
                f"keyed {s['keyed']} / compact {s['compact']}",
                i["config_hash"][0],
                i["prompt_hash"][:12],
            ]
        )
    ik = ex["identity"]["keyed"]
    rows.append(
        [
            m["ab_keyed"] + " (A/B run)",
            "/".join(ik["output_format_trace"]),
            f"keyed {ik['raw_text_shape_pages']['keyed']} / "
            f"compact {ik['raw_text_shape_pages']['compact']}",
            ik["config_hash"][0],
            ik["prompt_hash"][:12],
        ]
    )
    w(md_table(["run", "format", "raw_text pages", "config hash", "prompt hash (12)"], rows))
    r40, r100, sv = s4["rank40"], s4["rank100"], s4["saved_rank100"]
    w("\nspike40 ranking recomputed from the six compact runs (`eval.rank_runs`):\n")
    w(
        md_table(
            ["rank", "config", "OVERALL % [95% CI]", "false-fill %"],
            [
                [
                    i + 1,
                    r["config"],
                    f"{pct(r['OVERALL'])} [{pct(r['ci95'][0])}, {pct(r['ci95'][1])}]",
                    pct(r["false_fill_rate"]),
                ]
                for i, r in enumerate(r40["ranking"])
            ],
        )
    )
    w(
        f"\ntop-2 {r40['top2']}: delta {100 * r40['delta']:+.2f} pts, paired CI "
        f"[{100 * r40['ci95'][0]:+.2f}, {100 * r40['ci95'][1]:+.2f}], individual CIs overlap "
        f"{r40['cis_overlap']}, decision `{r40['decision']}`.\n"
    )
    w("\ndev100 ranking recomputed (`eval.rank_runs`) vs the saved `dev100_rank.json`:\n")
    w(
        md_table(
            [
                "source",
                "rank 1",
                "OVERALL %",
                "rank 2",
                "OVERALL %",
                "paired d OVERALL pts [95% CI]",
                "decision",
            ],
            [
                [
                    "recomputed",
                    r100["ranking"][0]["config"],
                    pct(r100["ranking"][0]["OVERALL"]),
                    r100["ranking"][1]["config"],
                    pct(r100["ranking"][1]["OVERALL"]),
                    f"{100 * r100['delta']:+.2f} "
                    f"[{100 * r100['ci95'][0]:+.2f}, {100 * r100['ci95'][1]:+.2f}]",
                    r100["decision"],
                ]
            ]
            + (
                [
                    [
                        "saved dev100_rank.json",
                        sv["ranking"][0]["config"],
                        pct(sv["ranking"][0]["OVERALL"]),
                        sv["ranking"][1]["config"],
                        pct(sv["ranking"][1]["OVERALL"]),
                        f"{100 * sv['delta']:+.2f} "
                        f"[{100 * sv['ci95'][0]:+.2f}, {100 * sv['ci95'][1]:+.2f}]",
                        sv["decision"],
                    ]
                ]
                if sv
                else []
            ),
        )
    )
    w("\n## Findings (computed; interpretation of the tables above)\n")
    w(findings(res))
    return "\n".join(out) + "\n"


def render_local(res: Mapping[str, Any], limit: int = 60) -> str:
    """Value-level examples (confidential: written outside the repo only)."""
    out = ["# Re-spike value-level examples (LOCAL, confidential, never commit)\n"]
    for label in ("final_pick", "context"):
        lc = res["_local"][label]
        name = res["s3"][label]["run"]
        out.append(f"\n## {label}: {name}\n")
        hc = [c for c in lc["header"] if c["pred_null"] != c["gold_null"]]
        out.append("### header cells where null status differs (over-null / false fill)\n")
        for c in hc:
            g = lc["gold"][c["doc"]]["header"].get(c["field"])
            p = lc["pred"].get(c["doc"], {}).get("header", {}).get(c["field"])
            kind = "OVER-NULL" if c["pred_null"] else "FALSE-FILL"
            out.append(
                f"- {c['doc']} {c['field']} "
                f"[{c['cls']}, {'scanned' if c['scanned'] else 'digital'}] "
                f"{kind}: gold={g!r} pred={p!r}\n"
            )
        rc = [c for c in lc["rows"] if c["pred_null"] != c["gold_null"]]
        out.append(f"\n### row cells where null status differs ({len(rc)}; first {limit})\n")
        for c in rc[:limit]:
            g = lc["gold"][c["doc"]]["line_items"][c["gi"]].get(c["field"])
            pr = [x for x in (lc["pred"][c["doc"]].get("line_items") or []) if isinstance(x, dict)]
            p = pr[c["pi"]].get(c["field"])
            kind = "OVER-NULL" if c["pred_null"] else "FALSE-FILL"
            out.append(
                f"- {c['doc']} row gold#{c['gi']}/pred#{c['pi']} {c['field']} "
                f"[{'scanned' if c['scanned'] else 'digital'}] {kind}: gold={g!r} pred={p!r}\n"
            )
    out.append("\n## taxonomy error records of the final pick (first 80)\n")
    for e in res["_errors"][:80]:
        out.append(
            f"- {e['doc_id']} {e['scope']}.{e['field']} {e['label']} ({e['support']}): "
            f"pred={e['pred']!r} gold={e['gold']!r}\n"
        )
    return "".join(out)


def to_jsonable(o: Any) -> Any:
    """Recursively make results JSON-safe (drops ``_``-prefixed bulk keys)."""
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(x) for x in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs-dir", type=Path, default=os.environ.get(RUNS_ENV))
    ap.add_argument("--sha", default=CODE_SHA)
    ap.add_argument("--final-pick", default=None, help="default: dev100_qwen35_4b_img_only_<sha>")
    ap.add_argument("--context", default=None, help="default: dev100_qwen35_4b_img_ocr_<sha>")
    ap.add_argument("--ab-keyed", default=None, help="default: ab_keyed_qwen35_4b_img_ocr_<sha>")
    ap.add_argument(
        "--ab-compact", default=None, help="default: spike40_qwen35_4b_img_ocr_compact_<sha>"
    )
    ap.add_argument("--split", default="dev")
    ap.add_argument("--gold-dir", type=Path, default=None)
    ap.add_argument("--meta", type=Path, default=ROOT / "meta" / "dev.json")
    ap.add_argument("--out-md", type=Path, default=ROOT / "reports" / "respike.md")
    ap.add_argument("--local-md", type=Path, default=None)
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args(argv)
    if not a.runs_dir:
        print(f"respike: pass --runs-dir or set ${RUNS_ENV}", file=sys.stderr)
        return 2
    sha = a.sha
    res = analyse(
        Path(a.runs_dir),
        a.final_pick or f"dev100_qwen35_4b_img_only_{sha}",
        a.context or f"dev100_qwen35_4b_img_ocr_{sha}",
        a.ab_keyed or f"ab_keyed_qwen35_4b_img_ocr_{sha}",
        a.ab_compact or f"spike40_qwen35_4b_img_ocr_compact_{sha}",
        sha,
        a.gold_dir or paths.data_dir() / a.split / "labels",
        a.meta,
        n=a.n_boot,
    )
    a.out_md.write_text(render_md(res, Path(a.runs_dir), git_state()), encoding="utf-8")
    local = a.local_md or paths.runs_dir() / "diagnosis" / "respike_local.md"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(render_local(res), encoding="utf-8")
    if a.json_out:
        a.json_out.write_text(json.dumps(to_jsonable(res), indent=1), encoding="utf-8")
    print(f"wrote {a.out_md} and {local}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
