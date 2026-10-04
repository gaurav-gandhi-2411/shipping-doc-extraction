"""Fill ``reports/final/report.md.tmpl`` from run artifacts (the 2-page final report).

    uv run python scripts/build_report.py                       # no inputs: all-PENDING skeleton
    uv run python scripts/build_report.py --manifest reports/final/inputs.json --strict
    uv run python scripts/build_report.py --dummy --out $SHIPDOC_TMP_DIR/report_dummy.md
    uv run python scripts/build_report.py --final-system v1_5 --out $SHIPDOC_TMP_DIR/r.md  # or v2
    uv run python scripts/build_report.py --numbers-md   # reports/report_numbers.md

The final report (``reports/final/report_final.md.tmpl`` + ``report_numbers.json``) has its own
grammar, documented above the ``FINAL_SYSTEMS`` constant; the legacy template below is unchanged.

The template holds NO hand-typed number: every numeric token is a ``{{placeholder}}`` (or a table
built here). A placeholder that cannot be resolved renders as ``[[PENDING: name]]``; the builder
prints the count and ``--strict`` exits non-zero if any remain.

Placeholder grammar
-------------------
``{{ns.key.key|fmt}}``  value at a dotted path inside the namespace ``ns`` (a JSON file loaded from
the manifest); list items are addressed by index (``rungs.0.name``). ``{{values.name}}`` is a flat
scalar (links, dates, model state). ``{{auto:name}}`` / ``{{table:name}}`` are generated blocks.
Formats (``fmt``; scores are FRACTIONS in [0, 1], as the scorer and ``shipdoc.eval`` emit them):

======  ===========================================================================
(none)  strings and ints as is; floats with 3 significant digits
int     integer
pct     fraction -> percent, 1 decimal ("76.8")
pctp    same with a percent sign
ci      ``{point, lo, hi}`` -> "76.8 [72.2, 81.6]" (percent)
dci     ``{delta|point, lo, hi}`` -> "+0.5 [+0.0, +1.3]" (percentage points, signed)
ciw     ``{lo, hi}`` -> "8.5" (half-width of the CI in percentage points)
sec     float with 2 decimals;  usd  "$12.34";  f1  one decimal;  f3  three decimals
======  ===========================================================================

Status marks (suffix of every rendered number, set per source in the manifest; default
``unverified`` because spec section 10 says a number is unverified until re-computed from an
artifact): ``verified`` no mark, ``unverified`` dagger, ``estimated`` asterisk, ``dummy`` double
dagger (fixture only).

Manifest (JSON, paths relative to the manifest file or absolute)::

    {"schema": "shipdoc-report-inputs/1",
     "sources": {"<ns>": {"path": "...json", "status": "verified|unverified|estimated"}, ...},
     "trace":   {"path": "runs/<id>/trace.jsonl", "status": "..."},      # optional -> ns latency
     "values":  {"github_url": "...", "model_state": {"value": "...", "status": "..."}, ...}}

Expected namespaces (``reports/final/inputs.example.json`` lists them; keys the template reads):

* ``dev``     metrics.json of the dev run: ``slices.<name>.{documents, OVERALL,
  header_field_accuracy, row_f1, documents_fully_correct, false_fill_rate,
  ci95.<metric>.{point,lo,hi}}`` for names ``all, invoices, waybills, scanned=yes, scanned=no,
  multipage=yes, repeated_parts=yes``; and ``OVERALL_ci95``.
* ``oof``     ``oof_compare.json`` (``python -m shipdoc oof compare``): ``subsets.<all|invoices|
  waybills|scanned|digital>.{n_docs, zero_shot, oof, paired_delta_oof_minus_zero_shot.OVERALL}``.
* ``ablation`` ``{n_docs, split, rungs: [{name, overall: est, false_fill: est, delta_vs_prev:
  {delta,lo,hi} | null}]}`` (est = ``{point, lo, hi}``), cumulative ladder, one entry per rung.
* ``failure`` ``{run_id, n_docs, causes: {<cause>: {rows, docs, oracle: {delta,lo,hi}}}}`` with the
  causes ``spn_misread, column_shift, po_in_cpn_slot`` (names of reports/row_errors.md X1.3/X4).
* ``calibration`` (``scripts/report_inputs.py`` from calibrate_v2 + calibrate): ``{n_docs, dev_docs,
  shapes_equal_folds, shapes_folds, min_coverage, row_null_policy: {delta: {delta,lo,hi},
  header_only_delta}, types: {<pop>: TY},
  false_fill: {redaction: FF, absent_line: FF}, synthetic_redaction: {n_docs, null_rate: est}}``
  with ``pop`` in ``header, supplier_part_number, customer_part_number, purchase_order, quantity,
  all_per_type_tau, doc``; ``TY = {n, ece, t95: TG, t98: TG, t99: TG}`` (``doc`` has no ``t99``);
  ``TG = {status, coverage: est, precision: est | "n/a"}`` where ``status`` is ``attained``,
  ``near-miss`` or ``NOT ATTAINABLE`` (nested figures, slice ``all``); ``FF = {n, rate: est}``.
* ``rules`` (rule_gate + v1 replay): ``{train_docs, r3_docs, r1|r2|r2_ocrfree: {fixed, broken,
  delta}, r3: {fixed, broken, delta, total_fixed, top3_fixed, top_n, groups_total,
  groups_touched}, combined: {delta}}`` (deltas are ``{delta, lo, hi}`` fractions).
* ``clusters`` (layout_test_clusters): ``{n_test_docs, n_clusters, estimated_unseen_share,
  predicted_review_rate}`` (fractions; the share is the raw flagged share of invoice-like docs).
* ``cost``    MEASURED (``scripts/report_inputs.py``): ``{gpu, batch_size, timing_basis,
  pages_per_doc,
  test: {docs, pages, model_time_s, wall_clock_s, s_per_page}, zeroshot: {session_s, pages,
  s_per_page}, ocr: {engine, device, pages, compute_s, s_per_page_mean, s_per_page_p95, wall_s},
  submission: {docs, vlm_h, full_h, reuse_ocr_h, reuse_vlm_h}, train_gpu, train_gpu_source,
  train: {hours, steps, s_per_step, sessions, interrupted_sessions, peak_vram_gib, precision},
  oof: {hours, pages, load_and_merge_s}}``.
* ``cost_est`` ESTIMATES: ``{native_factor, native_s_per_page, native_vlm_h, hourly_usd,
  price_source, usd_per_1000_docs}`` (the price fields stay PENDING until a price source exists).
* ``latency`` derived from ``trace``: ``{pages, s_per_page_mean, s_per_page_median, s_per_page_p95,
  peak_vram_gib}`` (``page.meta.latency_s`` and ``peak_vram_bytes`` of ``trace.jsonl``).
* ``values``  ``github_url hf_url wandb_url report_date commit_sha model_state suppliers_per_fold``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "reports" / "final" / "report.md.tmpl"
DEFAULT_OUT = ROOT / "reports" / "final" / "report.md"
MANIFEST_SCHEMA = "shipdoc-report-inputs/1"

PLACEHOLDER_RE = re.compile(r"\{\{\s*([A-Za-z0-9_.:=\-]+)\s*(?:\|\s*([a-z0-9]+)\s*)?\}\}")
STATUS_MARK = {"verified": "", "unverified": "\u2020", "estimated": "*", "dummy": "\u2021"}
DEFAULT_STATUS = "unverified"

# Digit-bearing literals the template may contain; everything else numeric must be a placeholder.
# Kept explicit (and pinned in tests/test_build_report.py) so widening it is a reviewed change.
ALLOWED_LITERALS: tuple[str, ...] = (
    r"K=3",  # the fixed fold count of the CV protocol (spec Phase 4.4)
    r"\b2 pages\b",  # the page budget
    r"\u2265?98%",  # the auto-accept precision target (spec Phase 5.4)
    r"\b95%",  # the CI level
    r"\b99%",  # the stricter auto-accept precision target shown beside 95% and 98% (Section 7)
    r"\bp95\b",  # latency percentile name
    r"\bT4\b",  # GPU name
    r"\bG4\b",  # gate name
    r"\bR[1-8]b?\b",  # post-processing rule ids
    r"\bQwen3\.5-4B\b",  # model name
    r"\b1,000 docs\b",  # cost unit
    r"\b\d{4}-\d{2}-\d{2}\b",  # ISO dates
    r"^#{1,6} \d+\. ",  # numbered section headings
)


class ReportError(Exception):
    """A manifest or a format is malformed (distinct from a merely missing input)."""


# --------------------------------------------------------------------------------------------
# value lookup and formatting
# --------------------------------------------------------------------------------------------


class Context:
    """Namespaces (loaded JSON) with a status each, plus flat ``values``."""

    def __init__(
        self,
        data: Mapping[str, Any] | None = None,
        status: Mapping[str, str] | None = None,
        values: Mapping[str, Any] | None = None,
    ) -> None:
        self.data: dict[str, Any] = dict(data or {})
        self.status: dict[str, str] = dict(status or {})
        self.values: dict[str, Any] = {}
        self.value_status: dict[str, str] = {}
        for k, v in (values or {}).items():
            if isinstance(v, Mapping) and "value" in v:
                self.values[k] = v["value"]
                self.value_status[k] = str(v.get("status", "verified"))
            else:
                self.values[k] = v
                self.value_status[k] = "verified"

    def lookup(self, path: str) -> tuple[Any, str] | None:
        """(value, status) at a dotted path, or None when any part is missing."""
        head, _, rest = path.partition(".")
        if head == "values":
            if rest in self.values:
                return self.values[rest], self.value_status[rest]
            return None
        if head not in self.data:
            return None
        node: Any = self.data[head]
        for part in rest.split(".") if rest else []:
            if isinstance(node, Mapping) and part in node:
                node = node[part]
            elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
                node = node[int(part)]
            else:
                return None
        if node is None:
            return None
        return node, self.status.get(head, DEFAULT_STATUS)

    def render(self, path: str, fmt: str | None = None) -> str | None:
        """Formatted value with its status mark, or None when unresolved."""
        hit = self.lookup(path)
        if hit is None:
            return None
        value, status = hit
        if status not in STATUS_MARK:
            raise ReportError(f"unknown status {status!r} for {path}")
        text = format_value(value, fmt, path)
        return text + (STATUS_MARK[status] if _is_numeric_text(text) else "")


def _is_numeric_text(text: str) -> bool:
    return any(c.isdigit() for c in text)


def _num(x: Any, path: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ReportError(f"{path}: expected a number, got {type(x).__name__}")
    return float(x)


def _est(x: Any, path: str, point: tuple[str, ...] = ("point",)) -> tuple[float, float, float]:
    if not isinstance(x, Mapping):
        raise ReportError(f"{path}: expected an object with lo/hi, got {type(x).__name__}")
    key = next((k for k in point if k in x), None)
    if key is None or "lo" not in x or "hi" not in x:
        raise ReportError(f"{path}: needs keys {point[0]}, lo, hi; has {sorted(x)}")
    return _num(x[key], path), _num(x["lo"], path), _num(x["hi"], path)


def format_value(value: Any, fmt: str | None, path: str = "?") -> str:
    """Render one value; raises `ReportError` on a type mismatch (never silently)."""
    if fmt is None:
        if isinstance(value, float):
            return f"{value:.3g}"
        if isinstance(value, (str, int)) and not isinstance(value, bool):
            return str(value)
        raise ReportError(f"{path}: no format given for a {type(value).__name__}")
    if fmt == "int":
        return str(int(_num(value, path)))
    if fmt == "pct":
        return f"{100 * _num(value, path):.1f}"
    if fmt == "pctp":
        return f"{100 * _num(value, path):.1f}%"
    if fmt == "sec":
        return f"{_num(value, path):.2f}"
    if fmt == "f1":
        return f"{_num(value, path):.1f}"
    if fmt == "f3":
        return f"{_num(value, path):.3f}"
    if fmt == "usd":
        return f"${_num(value, path):.2f}"
    if fmt == "ci":
        p, lo, hi = _est(value, path)
        return f"{100 * p:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"
    if fmt == "dci":
        d, lo, hi = _est(value, path, ("delta", "point"))
        return f"{100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"
    if fmt == "ciw":
        _, lo, hi = _est({"point": 0, **value} if isinstance(value, Mapping) else value, path)
        return f"{50 * (hi - lo):.1f}"
    raise ReportError(f"{path}: unknown format {fmt!r}")


# --------------------------------------------------------------------------------------------
# generated blocks (tables); each returns markdown, cells are PENDING when an input is missing
# --------------------------------------------------------------------------------------------


class Pending:
    """Collects unresolved names while tables/placeholders render."""

    def __init__(self) -> None:
        self.names: list[str] = []

    def token(self, name: str) -> str:
        self.names.append(name)
        return f"[[PENDING: {name}]]"


def _cell(ctx: Context, pend: Pending, path: str, fmt: str | None) -> str:
    out = ctx.render(path, fmt)
    return out if out is not None else pend.token(path)


def _table(header: list[str], widths: list[int], rows: list[list[str]]) -> str:
    sep = "|" + "|".join("-" * w for w in widths) + "|"
    lines = ["| " + " | ".join(header) + " |", sep]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


DEV_SLICES = (
    ("all", "all"),
    ("invoices", "invoices"),
    ("waybills", "waybills"),
    ("scanned=yes", "scanned"),
    ("scanned=no", "digital"),
    ("multipage=yes", "multipage"),
    ("repeated_parts=yes", "repeated parts"),
)
OOF_SUBSETS = (
    ("invoices", "**invoice-only (HEADLINE)**"),
    ("all", "all (invoices + waybills)"),
    ("waybills", "waybills"),
    ("scanned", "scanned"),
    ("digital", "digital"),
)


def table_results(ctx: Context, pend: Pending) -> str:
    """Seen-layout dev slices, then unseen OOF slices, with n and 95% CIs."""
    if "dev" not in ctx.data:
        return pend.token("table:results")
    rows = []
    for key, label in DEV_SLICES:
        b = f"dev.slices.{key}"
        rows.append(
            [
                f"dev (seen), {label}",
                _cell(ctx, pend, f"{b}.documents", "int"),
                _cell(ctx, pend, f"{b}.ci95.OVERALL", "ci"),
                _cell(ctx, pend, f"{b}.header_field_accuracy", "pct"),
                _cell(ctx, pend, f"{b}.row_f1", "pct"),
                _cell(ctx, pend, f"{b}.false_fill_rate", "pct"),
            ]
        )
    for key, label in OOF_SUBSETS:
        b = f"oof.subsets.{key}"
        rows.append(
            [
                f"OOF (unseen), {label}",
                _cell(ctx, pend, f"{b}.n_docs", "int"),
                _cell(ctx, pend, f"{b}.oof.ci95.OVERALL", "ci"),
                _cell(ctx, pend, f"{b}.oof.header_field_accuracy", "pct"),
                _cell(ctx, pend, f"{b}.oof.row_f1", "pct"),
                _cell(ctx, pend, f"{b}.oof.false_fill_rate", "pct"),
            ]
        )
    return _table(
        ["slice", "n", "OVERALL % [95% CI]", "header acc %", "row F1 %", "false-fill %"],
        [22, 9, 19, 10, 9, 10],
        rows,
    )


def table_oof_vs_zero_shot(ctx: Context, pend: Pending) -> str:
    """OOF model against zero-shot on the same held-out documents, paired delta."""
    if "oof" not in ctx.data:
        return pend.token("table:oof_vs_zero_shot")
    rows = []
    for key, label in OOF_SUBSETS:
        b = f"oof.subsets.{key}"
        rows.append(
            [
                label,
                _cell(ctx, pend, f"{b}.zero_shot.ci95.OVERALL", "ci"),
                _cell(ctx, pend, f"{b}.oof.ci95.OVERALL", "ci"),
                _cell(ctx, pend, f"{b}.paired_delta_oof_minus_zero_shot.OVERALL", "dci"),
            ]
        )
    return _table(
        ["held-out slice", "zero-shot OVERALL %", "OOF OVERALL %", "paired delta, pts [95% CI]"],
        [24, 20, 20, 24],
        rows,
    )


def table_ablation(ctx: Context, pend: Pending) -> str:
    """Cumulative ladder, one row per rung, with the paired delta against the previous rung."""
    rungs = ctx.data.get("ablation", {}).get("rungs") if "ablation" in ctx.data else None
    if not rungs:
        return pend.token("table:ablation")
    rows = []
    for i, r in enumerate(rungs):
        b = f"ablation.rungs.{i}"
        has_prev = r.get("delta_vs_prev") is not None
        delta = _cell(ctx, pend, f"{b}.delta_vs_prev", "dci") if has_prev else "-"
        rows.append(
            [
                _cell(ctx, pend, f"{b}.name", None),
                _cell(ctx, pend, f"{b}.overall", "ci"),
                delta,
                _cell(ctx, pend, f"{b}.false_fill", "ci"),
            ]
        )
    return _table(
        ["rung (cumulative)", "OVERALL % [95% CI]", "paired delta vs prev, pts", "false-fill %"],
        [34, 20, 28, 20],
        rows,
    )


REVIEW_TYPES = (
    ("header", "header fields", True),
    ("supplier_part_number", "supplier part number", True),
    ("customer_part_number", "customer part number", True),
    ("purchase_order", "purchase order", True),
    ("quantity", "quantity", True),
    ("all_per_type_tau", "all fields (per-type thresholds)", True),
    ("doc", "document (fully correct)", False),  # the doc flag has no 99% target in calibrate_v2
)


def _ci_or_na(ctx: Context, pend: Pending, path: str) -> str:
    """``ci`` cell, or the converter's literal string ("n/a": nothing was auto-accepted)."""
    hit = ctx.lookup(path)
    if hit is not None and isinstance(hit[0], str):
        return hit[0]
    return _cell(ctx, pend, path, "ci")


def table_review(ctx: Context, pend: Pending) -> str:
    """Per-field-type review flag: nested coverage / precision at 98%, coverage at 95 / 99, ECE."""
    if "calibration" not in ctx.data:
        return pend.token("table:review")
    rows = []
    for key, label, has99 in REVIEW_TYPES:
        b = f"calibration.types.{key}"
        cov = _cell(ctx, pend, f"{b}.t95.coverage.point", "pct")
        cov += " / " + (_cell(ctx, pend, f"{b}.t99.coverage.point", "pct") if has99 else "-")
        rows.append(
            [
                label,
                _cell(ctx, pend, f"{b}.n", "int"),
                _cell(ctx, pend, f"{b}.t98.status", None),
                _cell(ctx, pend, f"{b}.t98.coverage", "ci"),
                _ci_or_na(ctx, pend, f"{b}.t98.precision"),
                cov,
                _cell(ctx, pend, f"{b}.ece", "f3"),
            ]
        )
    return _table(
        [
            "field type",
            "n",
            "98% target",
            "coverage % [CI] at 98%",
            "precision % [CI] at 98%",
            "coverage % at 95 / 99%",
            "ECE",
        ],
        [22, 6, 16, 22, 22, 14, 6],
        rows,
    )


def table_false_fill(ctx: Context, pend: Pending) -> str:
    """False fills split into redaction vs absent-line, plus the SYNTHETIC redaction eval."""
    if "calibration" not in ctx.data:
        return pend.token("table:false_fill")
    rows = []
    for key, label in (
        ("redaction", "redaction (gold null, field illegible)"),
        ("absent_line", "absent line (gold null, field not on the page)"),
    ):
        b = f"calibration.false_fill.{key}"
        rows.append([label, _cell(ctx, pend, f"{b}.n", "int"), _cell(ctx, pend, f"{b}.rate", "ci")])
    b = "calibration.synthetic_redaction"
    rows.append(
        [
            "**SYNTHETIC** redaction eval (boxes drawn on dev pages)",
            _cell(ctx, pend, f"{b}.n_docs", "int"),
            _cell(ctx, pend, f"{b}.null_rate", "ci"),
        ]
    )
    return _table(
        ["gold-null kind", "n", "rate % [95% CI] (SYNTHETIC row: share correctly nulled)"],
        [40, 10, 40],
        rows,
    )


def auto_legend(ctx: Context, pend: Pending) -> str:
    """One-line key to the status marks (always present)."""
    return (
        "Marks: \u2020 unverified (not yet re-computed by the verifier), * estimated, "
        "\u2021 DUMMY fixture value; unmarked = re-computed from the logged artifact. "
        "All scores are percent; CIs are doc-level bootstrap, seed 42."
    )


def auto_banner(ctx: Context, pend: Pending) -> str:
    """Loud banner when any source is a dummy fixture; empty otherwise."""
    if "dummy" in ctx.status.values():
        return "**DUMMY NUMBERS: layout fixture only, not results. Do not publish.**"
    return ""


BLOCKS: dict[str, Callable[[Context, Pending], str]] = {
    "table:results": table_results,
    "table:oof_vs_zero_shot": table_oof_vs_zero_shot,
    "table:ablation": table_ablation,
    "table:review": table_review,
    "table:false_fill": table_false_fill,
    "auto:legend": auto_legend,
    "auto:banner": auto_banner,
}


# --------------------------------------------------------------------------------------------
# fill / load
# --------------------------------------------------------------------------------------------


def fill(template: str, ctx: Context) -> tuple[str, list[str]]:
    """Resolve every placeholder; returns (markdown, unresolved names in order of occurrence)."""
    pend = Pending()

    def sub(m: re.Match[str]) -> str:
        name, fmt = m.group(1), m.group(2)
        if name in BLOCKS:
            return BLOCKS[name](ctx, pend)
        if name.startswith(("table:", "auto:")):
            raise ReportError(f"unknown generated block {name!r}")
        out = ctx.render(name, fmt)
        return out if out is not None else pend.token(name)

    return PLACEHOLDER_RE.sub(sub, template), pend.names


def find_bare_numbers(template: str) -> list[str]:
    """Lines of `template` with a digit that is neither in a placeholder nor an allowed literal."""
    bad = []
    for line in template.splitlines():
        s = PLACEHOLDER_RE.sub("", line)
        for pat in ALLOWED_LITERALS:
            s = re.sub(pat, "", s, flags=re.MULTILINE)
        if re.search(r"\d", s):
            bad.append(line)
    return bad


def trace_latency(path: Path) -> dict[str, Any]:
    """Per-page latency and peak VRAM from a run's ``trace.jsonl`` (``page.meta``)."""
    lat: list[float] = []
    peak = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        for page in json.loads(raw).get("pages", []):
            meta = page.get("meta", {})
            if meta.get("latency_s") is not None:
                lat.append(float(meta["latency_s"]))
            peak = max(peak, int(meta.get("peak_vram_bytes") or 0))
    if not lat:
        raise ReportError(f"{path}: no page.meta.latency_s found")
    lat.sort()
    p95 = lat[min(len(lat) - 1, math.ceil(0.95 * len(lat)) - 1)]
    return {
        "pages": len(lat),
        "s_per_page_mean": statistics.fmean(lat),
        "s_per_page_median": statistics.median(lat),
        "s_per_page_p95": p95,
        "peak_vram_gib": peak / 2**30,
    }


def load_manifest(path: Path) -> Context:
    """Load a manifest; a missing source FILE is an error (a missing key is merely PENDING)."""
    man = json.loads(path.read_text(encoding="utf-8"))
    if man.get("schema") != MANIFEST_SCHEMA:
        raise ReportError(f"{path}: schema must be {MANIFEST_SCHEMA!r}, got {man.get('schema')!r}")
    base = path.parent

    def resolve(p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else base / q

    data: dict[str, Any] = {}
    status: dict[str, str] = {}
    for ns, src in (man.get("sources") or {}).items():
        f = resolve(src["path"])
        if not f.is_file():
            raise ReportError(f"source {ns!r}: {f} does not exist")
        data[ns] = json.loads(f.read_text(encoding="utf-8"))
        status[ns] = src.get("status", DEFAULT_STATUS)
    if man.get("trace"):
        f = resolve(man["trace"]["path"])
        if not f.is_file():
            raise ReportError(f"trace: {f} does not exist")
        data["latency"] = trace_latency(f)
        status["latency"] = man["trace"].get("status", DEFAULT_STATUS)
    return Context(data, status, man.get("values"))


def dummy_context() -> Context:
    """Fixture with DUMMY numbers, realistic shape, for the page-fit check. Never a result."""

    def est(p: float, w: float = 0.05) -> dict[str, float]:
        return {"point": p, "lo": p - w, "hi": min(1.0, p + w)}

    def dl(d: float) -> dict[str, float]:
        return {"delta": d, "lo": d - 0.02, "hi": d + 0.03}

    def blk(p: float) -> dict[str, Any]:
        return {
            "documents": 100,
            "OVERALL": p,
            "header_field_accuracy": p,
            "row_f1": p,
            "documents_fully_correct": p,
            "false_fill_rate": 0.01,
            "ci95": {"OVERALL": est(p)},
        }

    subs = {
        k: {
            "n_docs": 150,
            "zero_shot": blk(0.6),
            "oof": blk(0.7),
            "paired_delta_oof_minus_zero_shot": {"OVERALL": dl(0.1)},
        }
        for k in ("all", "invoices", "waybills", "scanned", "digital")
    }

    def tgt(status: str, prec: Any = None) -> dict[str, Any]:
        return {
            "status": status,
            "coverage": est(0.7),
            "precision": est(0.98, 0.01) if prec is None else prec,
        }

    def ty(n: int, status: str, has99: bool = True) -> dict[str, Any]:
        out = {"n": n, "ece": 0.03, "t95": tgt(status), "t98": tgt(status)}
        if has99:
            out["t99"] = tgt(status)
        return out

    def gate(fixed: int, broken: int = 0) -> dict[str, Any]:
        return {"fixed": fixed, "broken": broken, "delta": dl(0.01)}

    data: dict[str, Any] = {
        "dev": {"slices": {k: blk(0.8) for k, _ in DEV_SLICES}, "OVERALL_ci95": est(0.8)},
        "oof": {"subsets": subs},
        "ablation": {
            "n_docs": 100,
            "split": "DUMMY split",
            "rungs": [
                {
                    "name": f"rung {n}",
                    "overall": est(0.6 + 0.02 * i),
                    "false_fill": est(0.02),
                    "delta_vs_prev": None if i == 0 else dl(0.02),
                }
                for i, n in enumerate("ABCDEF")
            ],
        },
        "failure": {
            "run_id": "dummy_run",
            "n_docs": 100,
            "causes": {
                c: {"rows": 66, "docs": 28, "oracle": dl(0.05)}
                for c in ("spn_misread", "column_shift", "po_in_cpn_slot")
            },
        },
        "rules": {
            "train_docs": 400,
            "r3_docs": 500,
            "r1": gate(8),
            "r2": gate(20),
            "r2_ocrfree": gate(0),
            "r3": {
                **gate(502),
                "total_fixed": 502,
                "top3_fixed": 493,
                "top_n": 3,
                "groups_total": 18,
                "groups_touched": 6,
            },
            "combined": {"delta": dl(0.07)},
        },
        "calibration": {
            "n_docs": 500,
            "dev_docs": 100,
            "shapes_equal_folds": 3,
            "shapes_folds": 3,
            "min_coverage": 0.05,
            "row_null_policy": {"delta": dl(-0.013), "header_only_delta": dl(0.0)},
            "types": {
                "header": ty(3907, "attained"),
                "supplier_part_number": ty(4925, "near-miss"),
                "customer_part_number": ty(2080, "NOT ATTAINABLE"),
                "purchase_order": {
                    **ty(2736, "NOT ATTAINABLE"),
                    "t98": tgt("NOT ATTAINABLE", "n/a"),
                },
                "quantity": ty(4925, "NOT ATTAINABLE"),
                "all_per_type_tau": ty(18573, "NOT ATTAINABLE"),
                "doc": ty(500, "NOT ATTAINABLE", has99=False),
            },
            "false_fill": {
                "redaction": {"n": 35, "rate": est(0.1)},
                "absent_line": {"n": 157, "rate": est(0.02)},
            },
            "synthetic_redaction": {"n_docs": 100, "null_rate": est(0.8)},
        },
        "clusters": {
            "n_test_docs": 200,
            "n_clusters": 24,
            "estimated_unseen_share": 0.5,
            "predicted_review_rate": 0.3,
        },
        "cost": {
            "gpu": "T4",
            "batch_size": 8,
            "timing_basis": "DUMMY timing basis, measured from run manifests and session logs",
            "pages_per_doc": 1.9,
            "test": {
                "docs": 200,
                "pages": 280,
                "model_time_s": 3431.5,
                "wall_clock_s": 4146.6,
                "s_per_page": 12.26,
            },
            "zeroshot": {"session_s": 8406.1, "pages": 671, "s_per_page": 12.53},
            "ocr": {
                "engine": "paddleocr",
                "device": "gpu",
                "pages": 280,
                "compute_s": 326.9,
                "s_per_page_mean": 1.17,
                "s_per_page_p95": 1.67,
                "wall_s": 624.6,
            },
            "submission": {
                "docs": 200,
                "vlm_h": 0.95,
                "full_h": 1.04,
                "reuse_ocr_h": 0.09,
                "reuse_vlm_h": 0.0,
            },
            "train_gpu": "L4",
            "train_gpu_source": "DUMMY label source, supplied by hand; not recorded in artifacts",
            "train": {
                "hours": 2.17,
                "steps": 112,
                "s_per_step": 69.85,
                "sessions": 3,
                "interrupted_sessions": 2,
                "peak_vram_gib": 11.4,
                "precision": "bf16",
            },
            "oof": {"hours": 0.9, "pages": 230, "load_and_merge_s": 52.6},
        },
        "cost_est": {
            "native_factor": 1.05,
            "native_s_per_page": 12.8,
            "native_vlm_h": 1.0,
            "hourly_usd": 0.35,
            "price_source": "DUMMY",
            "usd_per_1000_docs": 4.2,
        },
        "latency": {
            "pages": 380,
            "s_per_page_mean": 9.1,
            "s_per_page_median": 8.0,
            "s_per_page_p95": 15.0,
            "peak_vram_gib": 9.5,
        },
    }
    values = {
        "github_url": "DUMMY-github-url",
        "hf_url": "DUMMY-hf-url",
        "wandb_url": "DUMMY-wandb-url",
        "report_date": "2000-01-01",
        "commit_sha": "0000000",
        "model_state": "DUMMY",
        "suppliers_per_fold": 6,
    }
    return Context(data, dict.fromkeys(data, "dummy"), values)


# --------------------------------------------------------------------------------------------
# final report, one switch FINAL_SYSTEM = v1_5 | v2
# --------------------------------------------------------------------------------------------
#
# ``reports/final/report_final.md.tmpl`` is filled from ``reports/final/report_numbers.json``, a
# registry in which EVERY number of the report is one entry with its source artifact and a
# VERIFIED / UNVERIFIED status; ``reports/report_numbers.md`` is rendered from the same registry,
# so the PDF cannot claim more than the table. Grammar of the final template:
#
# * ``{{n:id}}``   a registry value (text, already formatted); an UNVERIFIED value gets a dagger.
# * ``{{ph:id}}``  a placeholder: renders as ``**[[TO FILL: id]]**`` until the registry entry gets
#                  a ``value`` (then it is a number like any other).
# * ``{{table:name}}``  a pipe table whose rows are registry entries with ``cells``.
# * a line starting with ``@v1_5 `` or ``@v2 `` exists only in that system's report.
#
# A report with at least one unresolved placeholder is a DRAFT: the PDF carries a watermark.

FINAL_SYSTEMS = ("v1_5", "v2")
FINAL_SYSTEM = os.environ.get("FINAL_SYSTEM", "v1_5")  # the switch; --final-system overrides it
FINAL_TEMPLATE = ROOT / "reports" / "final" / "report_final.md.tmpl"
FINAL_REGISTRY = ROOT / "reports" / "final" / "report_numbers.json"
NUMBERS_MD = ROOT / "reports" / "report_numbers.md"
REGISTRY_SCHEMA = "shipdoc-report-numbers/1"
REGISTRY_STATUSES = ("VERIFIED", "UNVERIFIED")
WATERMARK_TEXT = "DRAFT: placeholders"
FINAL_RE = re.compile(r"\{\{\s*(n|ph|table):([A-Za-z0-9_]+)\s*\}\}")
AUTO_RE = re.compile(r"\{\{\s*auto:([A-Za-z0-9_]+)\s*\}\}")
TAG_RE = re.compile(r"^@(" + "|".join(FINAL_SYSTEMS) + r")(?: |$)")  # a bare tag = a blank line
# Matches the shape of a document id (split name + index). The report may name supplier GROUP ids
# (inv_g03) but never a document id; a hit is a hard error, not a warning.
DOC_ID_RE = re.compile(r"\b(?:train|dev|test)_\d{3,5}\b")
# Digit-bearing identifiers (not results) the final template may contain besides ALLOWED_LITERALS.
FINAL_ALLOWED_LITERALS: tuple[str, ...] = ALLOWED_LITERALS + (
    r"\bv1\.5\b",  # system names
    r"\bv[12]\b",
    r"\b[0-9]{2}n?\b(?= (?:resolution sweep|header-hint))",  # notebook ids named in prose
    r"\b[Ff]old[ -][0-2]\b",  # fold names of the K=3 protocol
    r"\b(?:excludes|includes) 0\b",  # the CI-vs-zero wording
    r"\bpage 1\b",
    r"\bcausal_conv1d\b",  # a package name
    r"\bDay 0\b",
    r"\bG3\b",  # the post-processing gate name of the spec
    r"\bcalibrate_v2\b",  # a script name
    r"\b3-fold\b",
    r"\b1260-token\b",  # the production config before the native-resolution switch
    r"\b1260\b",
    r"\bQwen3\.5-4B\b",
    r"\bL4\b",
    r"\bR1a\b",
    r"\bSection [1-9]\b",
    r"\bC[1-3]\b",  # the three clauses of spec section 11 item 1
    r"\bspec section 11\b",
    r"\bseed 42\b",  # the fixed seed
    r"\bitem [1-9]\b",
    r"\b[0-9]{2}n\b",
)


@dataclass
class FinalResult:
    """A filled final report: markdown, unresolved placeholder ids, every registry id it used."""

    text: str
    unresolved: list[str] = field(default_factory=list)
    used: list[str] = field(default_factory=list)


def load_registry(path: Path = FINAL_REGISTRY) -> dict[str, Any]:
    """The numbers registry; a wrong schema, status or an entry without a source is an error."""
    reg = json.loads(path.read_text(encoding="utf-8"))
    if reg.get("schema") != REGISTRY_SCHEMA:
        raise ReportError(f"{path}: schema must be {REGISTRY_SCHEMA!r}, got {reg.get('schema')!r}")
    for key, e in reg.get("entries", {}).items():
        if e.get("status") not in REGISTRY_STATUSES:
            raise ReportError(f"registry entry {key!r}: status must be one of {REGISTRY_STATUSES}")
        if not e.get("source"):
            raise ReportError(f"registry entry {key!r}: a source artifact is required")
        if "cells" not in e and "value" not in e:
            raise ReportError(f"registry entry {key!r}: needs a value or cells")
    return reg


def filter_system(template: str, system: str) -> str:
    """Drop lines tagged for the other system and strip the tag from the lines kept."""
    if system not in FINAL_SYSTEMS:
        raise ReportError(f"FINAL_SYSTEM must be one of {FINAL_SYSTEMS}, got {system!r}")
    out = []
    for line in template.splitlines():
        m = TAG_RE.match(line)
        if m is None:
            out.append(line)
        elif m.group(1) == system:
            out.append(line[m.end() :])
    return "\n".join(out) + "\n"


def _mark(status: str) -> str:
    return STATUS_MARK["unverified"] if status == "UNVERIFIED" else ""


def _final_cell(reg: Mapping[str, Any], res: FinalResult, key: str) -> str:
    """One ``n:`` or ``ph:`` token. A placeholder with a value is a number; without, a marker."""
    e = reg["entries"].get(key)
    if e is None:
        raise ReportError(f"template names {key!r}, which is not in the registry")
    res.used.append(key)
    if e.get("placeholder") and e.get("value") in (None, ""):
        res.unresolved.append(key)
        return f"**[[TO FILL: {key.replace('_', ' ')}]]**"  # no underscore: LaTeX breaks the word
    if "value" not in e:
        raise ReportError(f"registry entry {key!r} is a table row, not a value")
    return str(e["value"]) + _mark(e["status"])


def _final_table(reg: Mapping[str, Any], res: FinalResult, name: str, system: str) -> str:
    tab = reg.get("tables", {}).get(name)
    if tab is None:
        raise ReportError(f"template names table {name!r}, which is not in the registry")
    # a system may have its own header / widths (``header_v2``) and per-row cells (``cells_v2``),
    # e.g. the v2 slice table has two more columns; every row it shows must then carry them
    sys_header = f"header_{system}" in tab
    header = tab[f"header_{system}"] if sys_header else tab["header"]
    widths = tab.get(f"widths_{system}" if sys_header else "widths", [12] * len(header))
    rows = []
    for key in tab["rows"]:
        e = reg["entries"].get(key)
        if e is None or "cells" not in e:
            raise ReportError(f"table {name!r}: row {key!r} is missing or has no cells")
        if e.get("systems") and system not in e["systems"]:
            continue
        res.used.append(key)
        if sys_header and f"cells_{system}" not in e:
            raise ReportError(f"table {name!r}: row {key!r} has no cells_{system}")
        src_cells = e[f"cells_{system}"] if f"cells_{system}" in e else e["cells"]
        cells = [
            FINAL_RE.sub(lambda m: _final_cell(reg, res, m.group(2)), str(c)) for c in src_cells
        ]
        if len(cells) != len(header):
            raise ReportError(
                f"table {name!r}: row {key!r} has {len(cells)} cells, not {len(header)}"
            )
        cells[0] += _mark(e["status"])
        rows.append("| " + " | ".join(cells) + " |")
    if len(header) != len(widths):
        raise ReportError(f"table {name!r}: widths and header differ in length")
    header = [FINAL_RE.sub(lambda m: _final_cell(reg, res, m.group(2)), h) for h in header]
    head = "| " + " | ".join(header) + " |"
    sep = "|" + "|".join("-" * w for w in widths) + "|"
    return "\n".join([head, sep, *rows])


def render_final(template: str, reg: Mapping[str, Any], system: str) -> FinalResult:
    """Fill the final template for `system` (v1_5 or v2)."""
    res = FinalResult("")
    body = filter_system(template, system)

    def sub(m: re.Match[str]) -> str:
        kind, key = m.group(1), m.group(2)
        if kind == "table":
            return _final_table(reg, res, key, system)
        return _final_cell(reg, res, key)

    body = FINAL_RE.sub(sub, body)
    body = AUTO_RE.sub(lambda m: _final_auto(m.group(1), system, res), body)
    res.text = body
    return res


def _final_auto(name: str, system: str, res: FinalResult) -> str:
    if name == "banner":
        if not res.unresolved:
            return ""
        return (
            f"**{WATERMARK_TEXT.upper()}: {len(set(res.unresolved))} unresolved placeholder(s) "
            f"([[TO FILL: ...]]); system {system}. NOT THE FINAL REPORT.**"
        )
    if name == "legend":
        return (
            "† = UNVERIFIED (source and status of every number: reports/report_numbers.md); "
            "unmarked = VERIFIED by a second path. Scores are percent; CIs are doc-level "
            "bootstraps (2000 resamples, seed 42, unmodified scorer)."
        )
    raise ReportError(f"unknown auto block {name!r}")


def needs_watermark(unresolved: Sequence[str]) -> bool:
    """True iff at least one placeholder is unresolved (the report is a draft)."""
    return len(unresolved) > 0


def find_doc_ids(text: str) -> list[str]:
    """Every string in `text` shaped like a document id (the report must contain none)."""
    return DOC_ID_RE.findall(text)


def find_bare_final_numbers(template: str) -> list[str]:
    """Lines of the final template with a digit outside any placeholder or allowed literal."""
    bad = []
    for raw in template.splitlines():
        line = TAG_RE.sub("", raw)
        s = AUTO_RE.sub("", FINAL_RE.sub("", line))
        for pat in FINAL_ALLOWED_LITERALS:
            s = re.sub(pat, "", s, flags=re.MULTILINE)
        if re.search(r"\d", s):
            bad.append(line)
    return bad


def fill_test_registry(reg: Mapping[str, Any], variant: str = "") -> dict[str, Any]:
    """Copy of `reg` in which every unresolved placeholder holds its realistic-length dummy.

    Each placeholder entry carries ``fill_test``: a dummy of the length the real value will have
    (``budget_chars`` is that length). Used only to measure the page-2 slack of the v2 report in a
    SCRATCH build (``report_to_pdf.py --fill-test``); the committed registry keeps its
    placeholders and this copy is never written back. A placeholder without ``fill_test`` is an
    error: a silent gap would make the measured slack optimistic.
    """
    out = json.loads(json.dumps(reg))
    for key, e in out["entries"].items():
        if e.get("placeholder") and e.get("value") in (None, ""):
            if not e.get("fill_test"):
                raise ReportError(f"placeholder {key!r} has no fill_test dummy")
            # a variant (e.g. "no08": the final-adapter dev run was not done) may override a dummy
            e["value"] = (
                e.get(f"fill_test_{variant}", e["fill_test"]) if variant else e["fill_test"]
            )
    return out


def over_budget(reg: Mapping[str, Any]) -> list[tuple[str, int, int]]:
    """Entries whose filled ``value`` is longer than ``budget_chars``: (id, length, budget)."""
    bad = []
    for key, e in reg["entries"].items():
        budget, val = e.get("budget_chars"), e.get("value")
        if budget and isinstance(val, str) and len(val) > budget:
            bad.append((key, len(val), int(budget)))
    return bad


def build_final(
    system: str,
    template_path: Path = FINAL_TEMPLATE,
    registry_path: Path = FINAL_REGISTRY,
    fill_test: bool = False,
    fill_variant: str = "",
) -> FinalResult:
    """Fill the final report; raises `ReportError` if the text contains a document id.

    `fill_test` fills every placeholder with its dummy (layout measurement only, never a report).
    """
    reg = load_registry(registry_path)
    if fill_test:
        reg = fill_test_registry(reg, fill_variant)
    res = render_final(template_path.read_text(encoding="utf-8"), reg, system)
    ids = find_doc_ids(res.text)
    if ids:
        raise ReportError(f"the report text contains {len(ids)} document-id-shaped string(s)")
    return res


def numbers_markdown(reg: Mapping[str, Any], template: str | None = None) -> str:
    """``reports/report_numbers.md``: every number of both reports with source and status.

    The "used by" column is computed by rendering the template for each system, so the table
    lists exactly the entries a PDF shows (an entry no system uses is flagged ORPHAN).
    """
    tpl = template if template is not None else FINAL_TEMPLATE.read_text(encoding="utf-8")
    used = {s: set(render_final(tpl, reg, s).used) for s in FINAL_SYSTEMS}
    entries: Mapping[str, Any] = reg["entries"]
    lines = [
        "# Report numbers: provenance of every number in the final report",
        "",
        reg.get("preamble", ""),
        "",
        "Generated by `uv run python scripts/build_report.py --numbers-md` from "
        "`reports/final/report_numbers.json`; do not edit by hand. VERIFIED = re-computed by a "
        "second path (stated in `check`); UNVERIFIED = read from the cited artifact, not "
        "re-computed. Aggregates only: no document value, no document id.",
        "",
        "## Numbers shown in the PDFs",
        "",
        "| id | shown as | source artifact (commit) | run / basis | status | check |",
        "|---|---|---|---|---|---|",
    ]
    todo = []
    for key, e in entries.items():
        where = ",".join(s for s in FINAL_SYSTEMS if key in used[s]) or "ORPHAN"
        if e.get("placeholder") and e.get("value") in (None, ""):
            todo.append((key, e, where))
            continue
        shown = " | ".join(e["cells"]) if "cells" in e else str(e["value"])
        cell = shown.replace("|", "/")
        lines.append(
            f"| {key} ({where}) | {cell} | {e['source']} | {e.get('run', '')} | {e['status']} "
            f"| {e.get('check', '')} |"
        )
    lines += [
        "",
        "## To fill (unresolved placeholders; each makes the PDF a DRAFT with a watermark)",
        "",
        "| id | used by | what is needed | blocked on |",
        "|---|---|---|---|",
    ]
    for key, e, where in todo:
        lines.append(f"| {key} | {where} | {e.get('needs', '')} | {e.get('blocked_on', '')} |")
    lines += ["", "## Gaps (requirements or inputs that could not be located)", ""]
    lines += [f"- {g}" for g in reg.get("gaps", [])]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--final-system", choices=FINAL_SYSTEMS, default=None, help="final report")
    ap.add_argument("--numbers-md", action="store_true", help=f"write {NUMBERS_MD}")
    ap.add_argument("--manifest", type=Path, help="artifacts manifest (default: none = skeleton)")
    ap.add_argument("--template", type=Path, default=TEMPLATE)
    ap.add_argument("--out", type=Path, default=None, help=f"default {DEFAULT_OUT}")
    ap.add_argument("--dummy", action="store_true", help="DUMMY fixture; requires an --out")
    ap.add_argument("--strict", action="store_true", help="exit 1 if any placeholder is unresolved")
    args = ap.parse_args(argv)

    if args.numbers_md:
        NUMBERS_MD.write_text(numbers_markdown(load_registry()), encoding="utf-8", newline="\n")
        print(f"wrote {NUMBERS_MD}")
        return 0
    if args.final_system:
        if args.out is None:
            ap.error("--final-system needs --out (the markdown goes outside the repo)")
        res = build_final(args.final_system)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(res.text, encoding="utf-8", newline="\n")
        print(f"wrote {args.out}")
        print(
            f"unresolved placeholders: {len(res.unresolved)} (watermark: "
            f"{needs_watermark(res.unresolved)})"
        )
        return 1 if args.strict and res.unresolved else 0

    if args.dummy and args.manifest:
        ap.error("--dummy and --manifest are mutually exclusive")
    if args.dummy:
        # dummy numbers must never reach the committed report.md
        if args.out is None or args.out.resolve() == DEFAULT_OUT.resolve():
            ap.error("--dummy needs an explicit --out other than reports/final/report.md")
        ctx = dummy_context()
    elif args.manifest:
        ctx = load_manifest(args.manifest)
    else:
        ctx = Context()
    out = args.out or DEFAULT_OUT

    text, unresolved = fill(args.template.read_text(encoding="utf-8"), ctx)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {out}")
    print(f"unresolved placeholders: {len(unresolved)} ({len(set(unresolved))} distinct)")
    if args.strict and unresolved:
        for n in sorted(set(unresolved)):
            print(f"  PENDING {n}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
