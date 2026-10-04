"""Turn the repo's evidence artifacts into the JSON inputs that ``build_report.py`` reads.

    uv run python scripts/report_inputs.py --out-dir $SHIPDOC_TMP_DIR/report_inputs \\
        --run-dir $SHIPDOC_RUNS_DIR/zeroshot500/<02 run> \\
        --calibration-v2-dir $SHIPDOC_RUNS_DIR/calibration_v2/<run> \\
        --calibration-dir $SHIPDOC_RUNS_DIR/calibration/<run> \\
        --ablation-json $SHIPDOC_TMP_DIR/g3_500/postproc_ablation_500.json
    uv run python scripts/build_report.py --manifest $SHIPDOC_TMP_DIR/report_inputs/inputs.json

Writes one JSON per namespace (``ablation failure calibration rules clusters cost cost_est``) plus
``inputs.json`` (the manifest, every status ``unverified`` unless ``--status ns=verified``). A
namespace whose input was not supplied is simply not written: its placeholders stay PENDING.
Aggregates only: no gold or predicted value is read into or written to any output.

``cost`` is MEASURED (``--submission-manifest --ocr-timing --train-dir --oof-dir``, explicit
paths; a given path that is missing or changed in shape raises); ``cost_est`` holds the ESTIMATES
(native-resolution scaling from ``--res-sweep-md``, and the price fields, which stay PENDING
unless ``--cost`` supplies ``hourly_usd`` and ``price_source``: no price source exists in the repo).

Structured sources (JSON) are parsed as JSON. ``rule_gate.md``, ``v1_replay.md``,
``layout_test_clusters.md`` and ``row_errors.md`` exist only as markdown; their tables are parsed
by `parse_md_tables` + `find_table`, which raise `MdFormatError` (never return a guess) when a
heading, a column or a row is missing or a cell does not parse, so a format change fails loudly.
Markdown deltas carry the markdown's resolution (0.01 points) and are converted points -> fractions.

Definitions that are choices made here (so they are stated, not implied):

* ``calibration.types.<pop>.<tNN>.status``: ``attained`` iff nested precision (point) >= target
  and nested coverage (point) >= ``MIN_COVERAGE`` (the rule of ``scripts/calibrate_v2.py``);
  ``near-miss`` iff coverage is enough, precision is under target and the CI upper bound reaches
  it; ``NOT ATTAINABLE`` otherwise (including "nothing auto-accepted"). The ``doc`` population
  takes calibrate_v2's own ``doc_attainability`` instead.
* ``rules.r3.groups_touched``: invoice groups with fixed + broken + neutral > 0 (the rule changed
  a value there, whether or not the score moved).
* ``clusters.estimated_unseen_share``: the RAW flagged share of the invoice-like test docs
  (count / docs of the invoice row), not the FPR/TPR-corrected figure, which the artifact itself
  calls extremely sensitive. ``clusters.n_clusters`` is the invoice k (waybill clusters are
  degenerate). ``predicted_review_rate`` is produced by nothing and stays PENDING.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_report as br  # noqa: E402  (sibling script, not a package)

ROOT = Path(__file__).resolve().parents[1]
MIN_COVERAGE = 0.05  # calibrate_v2's minimum nested coverage for a target to count as attained
TARGETS = ("0.95", "0.98", "0.99")  # keys in the tau rows; the template shows 98% with 95/99
CAL_POPS = (
    "header",
    "supplier_part_number",
    "customer_part_number",
    "purchase_order",
    "quantity",
    "all_per_type_tau",
)
ECE_GROUP = {"all_per_type_tau": "all_emitted"}  # calibration list names it differently
R3_TOP_N = 3  # how many top groups the concentration disclosure counts


class MdFormatError(br.ReportError):
    """A markdown source no longer has the structure the converter was written against."""


# --------------------------------------------------------------------------------------------
# markdown tables
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MdTable:
    """One pipe table: its nearest preceding heading text, header cells and body rows."""

    heading: str
    header: list[str]
    rows: list[list[str]]


_SEP_RE = re.compile(r"^\|(\s*:?-+:?\s*\|)+\s*$")


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def parse_md_tables(text: str) -> list[MdTable]:
    """All pipe tables of `text`, each with the heading above it.

    A table is a header line starting with ``|`` followed by a ``|---|`` separator and body rows
    starting with ``|``. A row whose cell count differs from the header raises `MdFormatError`.
    """
    lines = text.splitlines()
    tables: list[MdTable] = []
    heading = ""
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^#{1,6}\s+(.*\S)\s*$", line)
        if m:
            heading = m.group(1)
        elif line.lstrip().startswith("|") and i + 1 < len(lines) and _SEP_RE.match(lines[i + 1]):
            header = _cells(line)
            rows: list[list[str]] = []
            i += 2
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                row = _cells(lines[i])
                if len(row) != len(header):
                    raise MdFormatError(
                        f"table under {heading!r}: row has {len(row)} cells, header {len(header)}:"
                        f" {lines[i][:80]!r}"
                    )
                rows.append(row)
                i += 1
            tables.append(MdTable(heading, header, rows))
            continue
        i += 1
    return tables


def find_table(
    tables: list[MdTable], heading_re: str, columns: tuple[str, ...] | list[str]
) -> MdTable:
    """First table whose heading matches `heading_re` and whose header has every column."""
    for t in tables:
        if re.search(heading_re, t.heading) and all(c in t.header for c in columns):
            return t
    raise MdFormatError(f"no table under a heading matching {heading_re!r} with columns {columns}")


def row_where(table: MdTable, column: str, pred: str, *, prefix: bool = False) -> dict[str, str]:
    """The body row whose `column` equals (or starts with) `pred`, as {column: cell}."""
    ci = table.header.index(column)
    for r in table.rows:
        if r[ci] == pred or (prefix and r[ci].startswith(pred)):
            return dict(zip(table.header, r, strict=True))
    raise MdFormatError(f"table under {table.heading!r}: no row with {column} = {pred!r}")


_NUM = r"[+-]?\d+(?:\.\d+)?"


def to_int(cell: str, what: str) -> int:
    """Integer cell; anything else is a format error."""
    if not re.fullmatch(r"\d+", cell.strip()):
        raise MdFormatError(f"{what}: expected an integer, got {cell!r}")
    return int(cell)


def pts_to_delta(delta_cell: str, ci_cell: str, what: str) -> dict[str, float]:
    """'+0.35' and '[+0.10, +0.63]' (points) -> ``{delta, lo, hi}`` as fractions."""
    if not re.fullmatch(_NUM, delta_cell.strip()):
        raise MdFormatError(f"{what}: expected a signed number of points, got {delta_cell!r}")
    m = re.fullmatch(rf"\[\s*({_NUM})\s*,\s*({_NUM})\s*\]", ci_cell.strip())
    if not m:
        raise MdFormatError(f"{what}: expected '[lo, hi]' points, got {ci_cell!r}")
    d, lo, hi = float(delta_cell), float(m.group(1)), float(m.group(2))
    if not lo <= d <= hi:
        raise MdFormatError(f"{what}: point {d} outside its interval [{lo}, {hi}]")
    return {"delta": round(d / 100, 6), "lo": round(lo / 100, 6), "hi": round(hi / 100, 6)}


# --------------------------------------------------------------------------------------------
# rules (rule_gate.md, v1_replay.md)
# --------------------------------------------------------------------------------------------

GATE_COLS = ("rule", "fixed", "broken", "d OVERALL (pts)", "95% CI (pts)")


def _gate_row(row: dict[str, str], what: str) -> dict[str, Any]:
    return {
        "fixed": to_int(row["fixed"], f"{what} fixed"),
        "broken": to_int(row["broken"], f"{what} broken"),
        "delta": pts_to_delta(row["d OVERALL (pts)"], row["95% CI (pts)"], what),
    }


def rules_from_gate_md(text: str) -> dict[str, Any]:
    """R1 / R2 / R2-ocrfree (TRAIN verdict) and R3 (supplier-held-out) from ``rule_gate.md``."""
    tables = parse_md_tables(text)
    t12 = find_table(tables, r"^R1 / R2 TRAIN verdict", GATE_COLS)
    m = re.search(r"\((\d+) docs\)", t12.heading)
    if not m:
        raise MdFormatError(f"cannot read the doc count from heading {t12.heading!r}")
    out: dict[str, Any] = {"train_docs": int(m.group(1))}
    for key, name in (("r1", "R1"), ("r2", "R2"), ("r2_ocrfree", "R2-ocrfree")):
        out[key] = _gate_row(row_where(t12, "rule", name), name)

    t3 = find_table(tables, r"^R3 supplier-held-out, .*\d+ docs$", GATE_COLS)
    m = re.search(r"(\d+) docs$", t3.heading)
    if not m:
        raise MdFormatError(f"cannot read the doc count from heading {t3.heading!r}")
    out["r3_docs"] = int(m.group(1))
    r3 = _gate_row(row_where(t3, "rule", "R3 (supplier-held-out", prefix=True), "R3")
    groups = find_table(
        tables, r"^R3 \(supplier-held-out, folds pooled\)", ("group", "fixed", "broken", "neutral")
    )
    if not groups.rows:
        raise MdFormatError("R3 per-group table is empty")
    per = [
        (
            to_int(r[groups.header.index("fixed")], f"{r[0]} fixed"),
            to_int(r[groups.header.index("broken")], f"{r[0]} broken"),
            to_int(r[groups.header.index("neutral")], f"{r[0]} neutral"),
        )
        for r in groups.rows
    ]
    total = sum(f for f, _, _ in per)
    if total != r3["fixed"]:
        raise MdFormatError(
            f"R3 group fixed rows sum to {total}, the summary row says {r3['fixed']}"
        )
    fixed_sorted = sorted((f for f, _, _ in per), reverse=True)
    out["r3"] = {
        **r3,
        "total_fixed": total,
        "top3_fixed": sum(fixed_sorted[:R3_TOP_N]),
        "top_n": R3_TOP_N,
        "groups_total": len(per),
        "groups_touched": sum(1 for f, b, n in per if f + b + n > 0),
    }
    return out


def combined_delta_from_replay_md(text: str) -> dict[str, float]:
    """The HONEST 'all' slice delta of ``v1_replay.md`` (R1+R2+R3, supplier-held-out R3)."""
    t = find_table(
        parse_md_tables(text),
        r"^\(a\) HONEST",
        ("slice", "docs", "d OVERALL (pts)", "95% CI (pts)"),
    )
    row = row_where(t, "slice", "all")
    return pts_to_delta(row["d OVERALL (pts)"], row["95% CI (pts)"], "v1 replay combined")


# --------------------------------------------------------------------------------------------
# failure taxonomy (row_errors.md) and clusters (layout_test_clusters.md)
# --------------------------------------------------------------------------------------------

FAILURE_LABELS = {
    "spn_misread": "rows: spn_misread",
    "column_shift": "rows: column_shift",
    "po_in_cpn_slot": "paired-row slot: po_in_cpn_slot",
}


def failure_from_row_errors_md(text: str) -> dict[str, Any]:
    """``{run_id, causes}`` from the X4 table (rows, docs where given, ORACLE delta)."""
    m = re.search(r"\*\*Provenance\.\*\*\s+Run `([^`]+)`", text)
    if not m:
        raise MdFormatError("row_errors.md: no 'Provenance. Run `<id>`' line")
    tables = parse_md_tables(text)
    cols = ("cause", "n rows / cells")
    ocol = next(
        (c for t in tables for c in t.header if c.startswith("recoverable OVERALL points")), None
    )
    if ocol is None:
        raise MdFormatError("row_errors.md: no 'recoverable OVERALL points' column")
    t = find_table(tables, r"^X4\.", (*cols, ocol))
    causes: dict[str, Any] = {}
    for key, label in FAILURE_LABELS.items():
        row = row_where(t, "cause", label)
        n = re.match(r"(\d+)\b", row["n rows / cells"])
        if not n:
            raise MdFormatError(f"{label}: cannot read a row count from {row['n rows / cells']!r}")
        oracle = re.fullmatch(rf"({_NUM}) (\[.*\])", row[ocol].strip())
        if not oracle:
            raise MdFormatError(f"{label}: cannot read an oracle delta from {row[ocol]!r}")
        entry: dict[str, Any] = {
            "rows": int(n.group(1)),
            "oracle": pts_to_delta(oracle.group(1), oracle.group(2), f"{label} oracle"),
        }
        docs = re.search(r"\((\d+) docs\)", row["n rows / cells"])
        if docs:
            entry["docs"] = int(docs.group(1))
        causes[key] = entry
    # The diagnosed doc set is named by the run id (``dev<N>_...``); fail closed if it is not.
    n = re.match(r"dev(\d+)_", m.group(1))
    if not n:
        raise MdFormatError(f"row_errors.md: run id {m.group(1)!r} does not name its doc count")
    return {"run_id": m.group(1), "n_docs": int(n.group(1)), "causes": causes}


def clusters_from_layout_md(text: str) -> dict[str, Any]:
    """``{n_test_docs, n_clusters, estimated_unseen_share}``; see the module docstring."""
    m = re.search(r"Test split \((\d+) docs\)", text)
    if not m:
        raise MdFormatError("layout md: no 'Test split (<n> docs)' line")
    tables = parse_md_tables(text)
    ktab = find_table(tables, r"^3\. ", ("doc type", "k", "threshold"))
    k = to_int(row_where(ktab, "doc type", "invoice")["k"], "invoice k")
    pcol = next(
        (c for t in tables for c in t.header if c.startswith("primary: centre distance")), None
    )
    if pcol is None:
        raise MdFormatError("layout md: no 'primary: centre distance' column")
    stab = find_table(tables, r"^3\. ", ("doc type", "test docs", pcol))
    row = row_where(stab, "doc type", "invoice")
    docs = to_int(row["test docs"], "invoice test docs")
    c = re.match(r"(\d+) \(", row[pcol])
    if not c or docs == 0:
        raise MdFormatError(f"cannot read a flagged count from {row[pcol]!r}")
    flagged = int(c.group(1))
    if flagged > docs:
        raise MdFormatError(f"flagged {flagged} exceeds invoice-like docs {docs}")
    return {
        "n_test_docs": int(m.group(1)),
        "n_clusters": k,
        "estimated_unseen_share": flagged / docs,
    }


# --------------------------------------------------------------------------------------------
# JSON sources
# --------------------------------------------------------------------------------------------


def _isnan(x: Any) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _est(d: dict[str, Any]) -> dict[str, float]:
    return {"point": float(d["point"]), "lo": float(d["lo"]), "hi": float(d["hi"])}


def _status(target: float, prec: dict[str, Any], cov: dict[str, Any]) -> str:
    if _isnan(prec["point"]) or cov["point"] < MIN_COVERAGE:
        return "NOT ATTAINABLE"
    if prec["point"] >= target:
        return "attained"
    return "near-miss" if prec["hi"] >= target else "NOT ATTAINABLE"


def _target_block(row: dict[str, Any], target: float, status: str | None = None) -> dict[str, Any]:
    prec = row["precision_accepted"]
    cov = row["accepted_share"]
    return {
        "status": status or _status(target, prec, cov),
        "coverage": _est(cov),
        "precision": "n/a" if _isnan(prec["point"]) else _est(prec),
    }


def wilson(k: int, n: int, z: float = 1.959964) -> dict[str, float]:
    """Point and Wilson score interval of k / n (n > 0)."""
    if n <= 0 or not 0 <= k <= n:
        raise br.ReportError(f"wilson: bad counts k={k}, n={n}")
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return {"point": p, "lo": max(0.0, mid - half), "hi": min(1.0, mid + half)}


def calibration_from_v2(v2: dict[str, Any], v1: dict[str, Any] | None = None) -> dict[str, Any]:
    """Per-type nested tau results (slice ``all``), ECE, and (with `v1`) the null-policy delta.

    `v2` is ``calibration_v2.json``; `v1` is the older ``calibration.json`` (row null policy and
    the gold-null false-fill counts), optional.
    """
    rows = {
        (r["population"], f"{r['target']:.2f}"): r
        for r in [*v2["tau"], *v2["doc_level"]]
        if r["slice"] == "all" and r["scheme"] == "nested"
    }
    ece = {c["group"]: c["ece_width15"] for c in v2["calibration"] if c["slice"] == "all"}
    types: dict[str, Any] = {}
    for pop in CAL_POPS:
        entry: dict[str, Any] = {}
        for t in TARGETS:
            if (pop, t) not in rows:
                raise br.ReportError(f"calibration_v2.json: no nested 'all' row for {pop} @ {t}")
            entry[f"t{round(float(t) * 100)}"] = _target_block(rows[(pop, t)], float(t))
        entry["n"] = rows[(pop, TARGETS[0])]["n"]
        grp = ECE_GROUP.get(pop, pop)
        if grp not in ece:
            raise br.ReportError(f"calibration_v2.json: no ECE row for slice all / {grp}")
        entry["ece"] = float(ece[grp])
        types[pop] = entry
    doc: dict[str, Any] = {}
    for t in ("0.95", "0.98"):
        if ("doc", t) not in rows:
            raise br.ReportError(f"calibration_v2.json: no nested 'all' doc row @ {t}")
        att = v2["doc_attainability"][str(float(t))]["attained"]
        doc[f"t{round(float(t) * 100)}"] = _target_block(
            rows[("doc", t)], float(t), "attained" if att else "NOT ATTAINABLE"
        )
    doc["n"] = rows[("doc", "0.95")]["n"]
    doc["ece"] = float(ece["doc"])
    types["doc"] = doc
    out: dict[str, Any] = {
        "n_docs": v2["slice_docs"]["all"],
        "dev_docs": v2["slice_docs"]["dev"],
        "min_coverage": MIN_COVERAGE,
        "types": types,
    }
    # per-fold "learned R3 shapes equal the all-gold set" flags (shape strings only, no values)
    eq = v2["provenance"]["shapes_equal"]
    out["shapes_equal_folds"] = sum(1 for v in eq.values() if v)
    out["shapes_folds"] = len(eq)
    if v1 is not None:
        pol = v1["null_policy_doc_level"]
        out["row_null_policy"] = {
            "delta": {
                k: float(pol["header_and_rows"]["all"]["OVERALL"][k]) for k in ("delta", "lo", "hi")
            },
            "header_only_delta": {
                k: float(pol["header_only"]["all"]["OVERALL"][k]) for k in ("delta", "lo", "hi")
            },
        }
        raw = pol["header_and_rows"]["all"]["false_fills_raw"]
        out["false_fill"] = {
            kind: {
                "n": raw[kind]["null_fields"],
                "rate": wilson(raw[kind]["filled"], raw[kind]["null_fields"]),
            }
            for kind in ("redaction", "absent_line")
        }
    return out


def ablation_from_json(d: dict[str, Any]) -> dict[str, Any]:
    """Cumulative ladder from ``postproc_ablation_*.json`` (``ladder.rungs``)."""
    rungs = []
    for r in d["ladder"]["rungs"]:
        o, f = r["OVERALL_ci"], r["false_fill_ci"]
        vs = r.get("vs_prev")
        rungs.append(
            {
                "name": f"{r['id']} {r['label']}",
                "overall": {"point": r["OVERALL"], "lo": o[0], "hi": o[1]},
                "false_fill": {"point": r["false_fill"], "lo": f[0], "hi": f[1]},
                "delta_vs_prev": (
                    {k: vs["OVERALL"][k] for k in ("delta", "lo", "hi")} if vs else None
                ),
            }
        )
    if not rungs:
        raise br.ReportError("ablation json: no rungs")
    meta = d["meta"]
    if meta["n_train"] + meta["n_dev"] != meta["n_docs"]:
        raise br.ReportError("ablation json: n_train + n_dev != n_docs")
    split = f"{meta['n_train']} train + {meta['n_dev']} dev docs, zero-shot run"
    return {"n_docs": meta["n_docs"], "split": split, "rungs": rungs}


def cost_from_run(
    run_dir: Path, overrides: dict[str, Any] | None, latency: dict[str, Any] | None
) -> dict[str, Any]:
    """``cost`` namespace: pages per doc from the run's metrics, plus hand-supplied fields.

    ``usd_per_1000_docs`` is computed only when ``hourly_usd`` is supplied:
    mean s/page x pages/doc x 1000 / 3600 x hourly_usd. ``gpu``, ``timing_basis``,
    ``hourly_usd`` and ``price_source`` come only from ``--cost`` (nothing here knows the GPU).
    """
    out: dict[str, Any] = {}
    mpath = run_dir / "metrics.json"
    if mpath.is_file():
        m = json.loads(mpath.read_text(encoding="utf-8"))
        if m.get("n_docs"):
            out["pages_per_doc"] = m["n_pages"] / m["n_docs"]
    out.update(overrides or {})
    if (
        latency
        and "hourly_usd" in out
        and "pages_per_doc" in out
        and "usd_per_1000_docs" not in out
    ):
        out["usd_per_1000_docs"] = (
            latency["s_per_page_mean"] * out["pages_per_doc"] * 1000 / 3600 * out["hourly_usd"]
        )
    return out


# --------------------------------------------------------------------------------------------
# measured cost and latency (run manifests, session logs, timing files); aggregates only
# --------------------------------------------------------------------------------------------

TRAIN_GPU_SOURCE = (
    "label supplied by hand from notebooks/README.md (its recommended training runtime); "
    "the run artifacts do not record the GPU"
)


def _need(d: Any, *path: str, what: str) -> Any:
    """`d[path...]`, or a `ReportError` naming the file and the missing key (shape change)."""
    node = d
    for p in path:
        if not isinstance(node, dict) or p not in node:
            raise br.ReportError(f"{what}: missing key {'.'.join(path)}")
        node = node[p]
    return node


def _pos(x: Any, what: str) -> float:
    """A strictly positive finite number, else a `ReportError`."""
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0:
        raise br.ReportError(f"{what}: expected a positive number, got {x!r}")
    return float(x)


def cost_inference_from_manifest(m: dict[str, Any]) -> dict[str, Any]:
    """T4 inference cost of the test run from a submission ``manifest.json`` (v0)."""
    w = "submission manifest"
    pages = _pos(_need(m, "docs", "n_pages", what=w), f"{w} docs.n_pages")
    docs = _pos(_need(m, "docs", "n_docs", what=w), f"{w} docs.n_docs")
    model_s = _pos(_need(m, "timings", "model_time_s", what=w), f"{w} timings.model_time_s")
    gpu = _need(m, "dependencies", "gpu", what=w)
    if not isinstance(gpu, str) or not gpu:
        raise br.ReportError(f"{w}: dependencies.gpu is not a string")
    return {
        "gpu": gpu,
        "batch_size": int(_pos(_need(m, "batch_size", what=w), f"{w} batch_size")),
        "test": {
            "docs": int(docs),
            "pages": int(pages),
            "pages_per_doc": pages / docs,
            "model_time_s": model_s,
            "wall_clock_s": _pos(_need(m, "timings", "wall_clock_s", what=w), f"{w} wall_clock_s"),
            "smoke_s": _pos(_need(m, "timings", "smoke_s", what=w), f"{w} smoke_s"),
            "s_per_page": model_s / pages,
        },
    }


def cost_ocr_from_timing(t: dict[str, Any]) -> dict[str, Any]:
    """OCR cost from ``ocr_timing.json`` (the per-page figures are the file's own)."""
    w = "ocr_timing.json"
    return {
        "engine": str(_need(t, "engine", what=w)),
        "device": str(_need(t, "device", what=w)),
        "pages": int(_pos(_need(t, "pages", what=w), f"{w} pages")),
        "compute_s": _pos(_need(t, "seconds_total", what=w), f"{w} seconds_total"),
        "s_per_page_mean": _pos(_need(t, "seconds_per_page_mean", what=w), f"{w} mean"),
        "s_per_page_p95": _pos(_need(t, "seconds_per_page_p95", what=w), f"{w} p95"),
        "wall_s": _pos(_need(t, "wall_clock_s_all_sessions", what=w), f"{w} wall"),
    }


def cost_zeroshot_from_run(
    sessions: list[dict[str, Any]], metrics: dict[str, Any]
) -> dict[str, Any]:
    """Zero-shot (02) run: total session seconds over its pages (``sessions.json``, metrics)."""
    w = "zero-shot run"
    done = [s for s in sessions if s.get("seconds") is not None]
    if len(done) != len(sessions) or not sessions:
        raise br.ReportError(f"{w}: sessions.json has an unfinished or no session")
    total = _pos(sum(float(s["seconds"]) for s in done), f"{w} session seconds")
    pages = _pos(_need(metrics, "n_pages", what=f"{w} metrics.json"), f"{w} n_pages")
    return {"session_s": total, "pages": int(pages), "s_per_page": total / pages}


def cost_train_from_run(
    steps: list[dict[str, Any]], sessions: list[dict[str, Any]], status: dict[str, Any]
) -> dict[str, Any]:
    """Fine-tune cost: the SUM of per-step seconds (``metrics.jsonl``), not wall clock.

    Sessions without ``seconds`` were interrupted (Colab disconnects), so wall clock was longer
    than the step sum. Fails loudly unless the steps are exactly 1..``total_steps`` and the
    run's ``train_status.json`` says ``complete``.
    """
    w = "training run"
    if _need(status, "state", what=f"{w} train_status.json") != "complete":
        raise br.ReportError(f"{w}: train_status.json state is not 'complete'")
    total = int(_need(status, "total_steps", what=f"{w} train_status.json"))
    nums = sorted(int(_need(s, "step", what=f"{w} metrics.jsonl line")) for s in steps)
    if nums != list(range(1, total + 1)):
        raise br.ReportError(f"{w}: metrics.jsonl steps are not exactly 1..{total}")
    secs = sum(_pos(s.get("seconds"), f"{w} step seconds") for s in steps)
    interrupted = sum(1 for s in sessions if s.get("seconds") is None)
    if not sessions or sessions[-1].get("exit_code") != 0:
        raise br.ReportError(f"{w}: the last session did not exit 0")
    return {
        "steps": total,
        "step_seconds_sum": secs,
        "s_per_step": secs / total,
        "hours": secs / 3600,
        "sessions": len(sessions),
        "interrupted_sessions": interrupted,
        "peak_vram_gib": _pos(_need(status, "peak_vram_gib", what=f"{w} status"), "peak vram"),
        "n_train_pages": int(_need(status, "n_train_pages", what=f"{w} status")),
        "precision": str(_need(status, "precision", what=f"{w} status")),
    }


def cost_oof_from_run(
    sessions: list[dict[str, Any]], manifest: dict[str, Any], metrics: dict[str, Any]
) -> dict[str, Any]:
    """OOF inference cost of one fold: session seconds (incl. load, merge, guard) over pages."""
    w = "OOF run"
    z = cost_zeroshot_from_run(sessions, metrics)
    load = _pos(
        _need(manifest, "oof", "merge", "load_and_merge_s", what=w), f"{w} load_and_merge_s"
    )
    return {
        "session_s": z["session_s"],
        "hours": z["session_s"] / 3600,
        "pages": z["pages"],
        "s_per_page": z["s_per_page"],
        "load_and_merge_s": load,
    }


def res_sweep_factor(text: str) -> dict[str, float]:
    """Native-over-control s/page ratio from ``reports/res_sweep.md`` (batch-size-one sweep)."""
    t = find_table(parse_md_tables(text), r"^Per-resolution results", ("max_pixels", "s/page"))
    out: dict[str, float] = {}
    for key, name in (("control_s_per_page", "1310720"), ("native_s_per_page", "2196480")):
        cell = row_where(t, "max_pixels", name, prefix=True)["s/page"]
        if not re.fullmatch(r"\d+(?:\.\d+)?", cell):
            raise MdFormatError(f"res_sweep: cannot read s/page from {cell!r}")
        out[key] = float(cell)
    out["factor"] = out["native_s_per_page"] / out["control_s_per_page"]
    return out


def native_estimate(factor: dict[str, float], inf: dict[str, Any]) -> dict[str, Any]:
    """ESTIMATE of native-resolution test cost: measured batch-8 time x the batch-1 ratio."""
    t = inf["test"]
    return {
        "native_factor": factor["factor"],
        "native_s_per_page": t["s_per_page"] * factor["factor"],
        "native_vlm_h": t["model_time_s"] * factor["factor"] / 3600,
    }


def submission_cost(inf: dict[str, Any], ocr: dict[str, Any]) -> dict[str, Any]:
    """Per-submission GPU hours: VLM only, VLM + OCR, and OCR only (VLM output reused)."""
    vlm = inf["test"]["model_time_s"]
    return {
        "docs": inf["test"]["docs"],
        "vlm_h": vlm / 3600,
        "full_h": (vlm + ocr["compute_s"]) / 3600,
        "reuse_ocr_h": ocr["compute_s"] / 3600,
        "reuse_vlm_h": 0.0,
    }


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def _read_json(p: Path) -> Any:
    if not p.is_file():
        raise br.ReportError(f"{p} does not exist")
    return json.loads(p.read_text(encoding="utf-8"))


def _read_text(p: Path) -> str:
    if not p.is_file():
        raise br.ReportError(f"{p} does not exist")
    return p.read_text(encoding="utf-8")


def _kv(items: list[str], what: str) -> dict[str, str]:
    out = {}
    for it in items:
        k, sep, v = it.partition("=")
        if not sep or not k:
            raise br.ReportError(f"{what}: expected key=value, got {it!r}")
        out[k] = v
    return out


def _read_jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in _read_text(p).splitlines() if x.strip()]


def measured_cost(args: argparse.Namespace, est: dict[str, Any]) -> dict[str, Any]:
    """Measured ``cost`` fields from the explicit artifact paths; fills `est` with the ESTIMATE.

    Every group is optional (its placeholders stay PENDING when its path is not given); a path
    that is given but missing or of a changed shape raises.
    """
    out: dict[str, Any] = {}
    inf = None
    if args.submission_manifest:
        inf = cost_inference_from_manifest(_read_json(args.submission_manifest))
        zs_manifest = _read_json(args.run_dir / "manifest.json")
        if zs_manifest.get("batch_size") != inf["batch_size"]:
            raise br.ReportError("zero-shot and submission runs used different batch sizes")
        out.update({k: inf[k] for k in ("gpu", "batch_size", "test")})
        out["timing_basis"] = (
            f"measured from run manifests and session logs, one run per figure, "
            f"{inf['gpu']} inference at batch size {inf['batch_size']}"
        )
        out["zeroshot"] = cost_zeroshot_from_run(
            _read_json(args.run_dir / "sessions.json"), _read_json(args.run_dir / "metrics.json")
        )
    ocr = cost_ocr_from_timing(_read_json(args.ocr_timing)) if args.ocr_timing else None
    if ocr:
        out["ocr"] = ocr
    if inf and ocr:
        out["submission"] = submission_cost(inf, ocr)
    if args.train_dir:
        d = args.train_dir
        out["train"] = cost_train_from_run(
            _read_jsonl(d / "metrics.jsonl"),
            _read_json(d / "sessions.json"),
            _read_json(d / "train_status.json"),
        )
        out["train_gpu"] = args.train_gpu
        out["train_gpu_source"] = TRAIN_GPU_SOURCE
    if args.oof_dir:
        d = args.oof_dir
        out["oof"] = cost_oof_from_run(
            _read_json(d / "sessions.json"), _read_json(d / "manifest.json"),
            _read_json(d / "metrics.json"),
        )  # fmt: skip
    if inf and args.res_sweep_md:
        est.update(native_estimate(res_sweep_factor(_read_text(args.res_sweep_md)), inf))
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Write the namespaces and the manifest; returns the manifest dict."""
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    status = _kv(args.status, "--status")
    sources: dict[str, dict[str, str]] = {}

    def emit(ns: str, data: Any, default_status: str = br.DEFAULT_STATUS) -> None:
        (out_dir / f"{ns}.json").write_text(
            json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )
        sources[ns] = {"path": f"{ns}.json", "status": status.get(ns, default_status)}

    def abs_path(p: Path) -> str:
        return str(p.resolve()).replace("\\", "/")

    if args.ablation_json:
        emit("ablation", ablation_from_json(_read_json(args.ablation_json)))
    if args.row_errors_md:
        emit("failure", failure_from_row_errors_md(_read_text(args.row_errors_md)))
    if args.layout_md:
        emit("clusters", clusters_from_layout_md(_read_text(args.layout_md)))
    if args.calibration_v2_dir:
        v2 = _read_json(args.calibration_v2_dir / "calibration_v2.json")
        v1 = _read_json(args.calibration_dir / "calibration.json") if args.calibration_dir else None
        emit("calibration", calibration_from_v2(v2, v1))
    if args.rule_gate_md:
        rules = rules_from_gate_md(_read_text(args.rule_gate_md))
        if args.v1_replay_md:
            rules["combined"] = {
                "delta": combined_delta_from_replay_md(_read_text(args.v1_replay_md))
            }
        emit("rules", rules)
    for ns, p in (("dev", args.dev_metrics), ("oof", args.oof_compare)):
        if p:
            if not p.is_file():
                raise br.ReportError(f"{p} does not exist")
            sources[ns] = {"path": abs_path(p), "status": status.get(ns, br.DEFAULT_STATUS)}

    manifest: dict[str, Any] = {"schema": br.MANIFEST_SCHEMA, "sources": sources}
    latency = None
    if args.run_dir:
        trace = args.run_dir / "trace.jsonl"
        if not trace.is_file():
            raise br.ReportError(f"{trace} does not exist")
        manifest["trace"] = {"path": abs_path(trace), "status": status.get("latency", "unverified")}
        latency = br.trace_latency(trace)
        overrides = _read_json(args.cost) if args.cost else None
        full = cost_from_run(args.run_dir, overrides, latency)
        # price fields are an estimate (own namespace, own status mark); the rest is measured
        est = {
            k: full.pop(k) for k in ("hourly_usd", "price_source", "usd_per_1000_docs") if k in full
        }
        cost = {**full, **measured_cost(args, est)}
        if cost:
            emit("cost", cost)
        if est:
            emit("cost_est", est, "estimated")
    manifest["values"] = _kv(args.value, "--value")
    (out_dir / "inputs.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, help="the 02 run (trace.jsonl, metrics.json)")
    ap.add_argument("--calibration-v2-dir", type=Path, help="holds calibration_v2.json")
    ap.add_argument("--calibration-dir", type=Path, help="holds calibration.json (null policy)")
    ap.add_argument("--ablation-json", type=Path)
    ap.add_argument("--v1-replay-md", type=Path)
    ap.add_argument("--rule-gate-md", type=Path)
    ap.add_argument("--layout-md", type=Path)
    ap.add_argument("--row-errors-md", type=Path)
    ap.add_argument("--cost", type=Path, help="JSON: gpu, timing_basis, hourly_usd, price_source")
    ap.add_argument("--submission-manifest", type=Path, help="v0 submission manifest.json")
    ap.add_argument("--ocr-timing", type=Path, help="ocr_timing.json of the OCR submission")
    ap.add_argument("--train-dir", type=Path, help="fold-0 training run (metrics.jsonl, ...)")
    ap.add_argument("--train-gpu", default="L4", help="training GPU label (not in the artifacts)")
    ap.add_argument("--oof-dir", type=Path, help="OOF inference run (sessions.json, manifest)")
    ap.add_argument("--res-sweep-md", type=Path, help="reports/res_sweep.md (native s/page ratio)")
    ap.add_argument("--dev-metrics", type=Path, help="dev metrics.json, passed through")
    ap.add_argument("--oof-compare", type=Path, help="oof_compare.json, passed through")
    ap.add_argument("--status", action="append", default=[], metavar="NS=STATUS")
    ap.add_argument("--value", action="append", default=[], metavar="KEY=VALUE")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = run(args)
    except br.ReportError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"wrote {args.out_dir / 'inputs.json'}: sources {sorted(manifest['sources'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
