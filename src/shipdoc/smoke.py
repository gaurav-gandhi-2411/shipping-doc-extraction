"""Smoke gate for the re-spike notebook: pick 5 docs and assert a tiny GPU run is sane.

Stdlib only on purpose: the Colab notebook loads this file by path with the system Python (the
project venv is only used for ``python -m shipdoc``), so it must not import ``shipdoc`` or numpy.

Six assertions per model, computed from ``trace.jsonl`` + ``predictions.json`` of a 5-doc run:

(a) JSON-valid page rate == 1.0
(b) no page hit ``max_new_tokens`` and no page's raw output is unterminated
(c) every invoice has >= 1 emitted row and the total emitted rows over the invoices are within
    +-50% of the gold row count
(d) non-null ``quantity`` share among emitted invoice rows >= 0.8
(e) non-null ``total_amount`` on >= 2/3 of the invoices
(f) no all-null rows left after merge

Plus one non-blocking WARNING (``warn_cpn_equals_po``): the number of emitted rows whose
customer_part_number is non-null and equal, under the scorer's identifier rule (whitespace
ignored, case ignored), to the same row's purchase_order. Step L found the model copying the PO
into the customer-part column; the count is logged per model and never fails the gate.

With ``require_logprobs`` (the 02 zero-shot notebook) a seventh blocking assertion (g) checks that
every field of every valid page has finite ``field_logprobs`` (`check_logprobs`).

The thresholds are smoke-level tripwires for the four unverified fixes (declared key order in the
grammar, numbers as JSON numbers, prompt v2, all-null rows dropped), not quality targets.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SMOKE_N = 5
ROW_TOLERANCE = 0.5  # emitted rows must be within gold * (1 +- 0.5)
MIN_QUANTITY_SHARE = 0.8
MIN_TOTAL_SHARE = 2 / 3


@dataclass(frozen=True)
class Check:
    """One assertion's outcome; ``detail`` is a short human-readable measurement."""

    name: str
    passed: bool
    detail: str


def pick_smoke_docs(
    spike_ids: Sequence[str], meta: Sequence[Mapping[str, Any]], n: int = SMOKE_N
) -> list[str]:
    """Deterministic smoke set from the spike docs, by meta tags only (no labels needed).

    Composition for n == 5: the first waybill, the first multipage invoice, then three
    single-page non-illegible invoices (first scanned, first digital, next in spike order), so
    the set has >= 3 invoices, >= 1 multipage doc and >= 1 waybill and stays cheap in pages.
    Returned in spike order.
    """
    if n != SMOKE_N:
        raise ValueError(f"composition is defined for n == {SMOKE_N}, got {n}")
    by_id = {m["doc_id"]: m for m in meta}
    docs = [by_id[d] for d in spike_ids]
    picked: list[str] = []

    def take(pred: Any) -> None:
        for m in docs:
            if m["doc_id"] not in picked and pred(m):
                picked.append(m["doc_id"])
                return
        raise ValueError("spike set cannot satisfy the smoke composition")

    def simple_invoice(m: Mapping[str, Any]) -> bool:
        return not m["waybill"] and not m["multipage"] and not m["illegible"]

    take(lambda m: m["waybill"])
    take(lambda m: not m["waybill"] and m["multipage"])
    take(lambda m: simple_invoice(m) and m["scanned"])
    take(lambda m: simple_invoice(m) and not m["scanned"])
    take(simple_invoice)
    order = {d: i for i, d in enumerate(spike_ids)}
    return sorted(picked, key=order.__getitem__)


def _blank(v: Any) -> bool:
    return v is None or (isinstance(v, str) and not v.strip())


def _identifier(v: Any) -> str:
    """The scorer's identifier rule: whitespace removed, case ignored."""
    return "".join(str(v).split()).upper()


@dataclass(frozen=True)
class Warn:
    """A non-blocking observation: logged in the smoke table and smoke_status.json only."""

    name: str
    count: int | None
    detail: str


def warn_cpn_equals_po(predictions: Mapping[str, Mapping[str, Any]]) -> Warn:
    """Count emitted rows whose customer_part_number equals the same row's purchase_order."""
    rows = [
        r
        for p in predictions.values()
        for r in (p.get("line_items") or [])
        if isinstance(r, Mapping)
    ]
    n = sum(
        not _blank(r.get("customer_part_number"))
        and not _blank(r.get("purchase_order"))
        and _identifier(r["customer_part_number"]) == _identifier(r["purchase_order"])
        for r in rows
    )
    return Warn("cpn_equals_po", n, f"{n} of {len(rows)} rows have customer_part_number == PO")


def load_warnings(run_dir: Path) -> list[Warn]:
    """Warnings for a finished smoke run; never raises (a warning must not block the gate)."""
    try:
        preds = json.loads((Path(run_dir) / "predictions.json").read_text(encoding="utf-8"))
        return [warn_cpn_equals_po(preds)]
    except (OSError, ValueError, AttributeError, TypeError) as exc:
        return [Warn("cpn_equals_po", None, f"not computed: {type(exc).__name__}: {exc}")]


def check_smoke(
    traces: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]],
    gold: Mapping[str, Mapping[str, Any]],
    max_new_tokens: int,
    require_logprobs: bool = False,
) -> list[Check]:
    """Evaluate the six assertions; pure (no I/O). `gold` maps doc_id -> label dict.

    With `require_logprobs` a seventh one is added (``g_field_logprobs``), see `check_logprobs`.
    """
    pages = [p for t in traces for p in t["pages"]]
    valid = sum(bool(p["json_valid"]) for p in pages)
    rate = valid / len(pages) if pages else 0.0
    checks = [
        Check("a_json_valid_rate", bool(pages) and rate == 1.0, f"{rate:.3f} ({len(pages)} pages)")
    ]

    capped = [
        f"{t['doc_id']} p{p['page']}"
        for t in traces
        for p in t["pages"]
        if (p["meta"].get("n_output_tokens") or 0) >= max_new_tokens
    ]
    unterminated = [
        f"{t['doc_id']} p{p['page']}"
        for t in traces
        for p in t["pages"]
        if not str(p.get("raw_text", "")).rstrip().endswith("}")
    ]
    checks.append(
        Check(
            "b_no_truncation",
            not capped and not unterminated,
            f"at cap {capped or 0}, unterminated {unterminated or 0} "
            f"(max_new_tokens {max_new_tokens})",
        )
    )

    inv = [t["doc_id"] for t in traces if gold[t["doc_id"]]["doc_type"] == "invoice"]
    rows_by_doc = {d: list(predictions.get(d, {}).get("line_items") or []) for d in inv}
    empty = [d for d, r in rows_by_doc.items() if not r]
    emitted = sum(len(r) for r in rows_by_doc.values())
    gold_rows = sum(len(gold[d].get("line_items") or []) for d in inv)
    lo, hi = gold_rows * (1 - ROW_TOLERANCE), gold_rows * (1 + ROW_TOLERANCE)
    checks.append(
        Check(
            "c_row_counts",
            bool(inv) and not empty and lo <= emitted <= hi,
            f"emitted {emitted} vs gold {gold_rows} (allowed {lo:.1f}-{hi:.1f}); "
            f"invoices with 0 rows: {empty or 0}",
        )
    )

    rows = [r for rs in rows_by_doc.values() for r in rs]
    q_share = sum(not _blank(r.get("quantity")) for r in rows) / len(rows) if rows else 0.0
    checks.append(
        Check(
            "d_quantity_share", q_share >= MIN_QUANTITY_SHARE, f"{q_share:.3f} of {len(rows)} rows"
        )
    )

    n_total = sum(
        not _blank(predictions.get(d, {}).get("header", {}).get("total_amount")) for d in inv
    )
    t_share = n_total / len(inv) if inv else 0.0
    checks.append(
        Check(
            "e_total_amount_share",
            bool(inv) and t_share >= MIN_TOTAL_SHARE,
            f"{n_total}/{len(inv)} invoices",
        )
    )

    all_rows = [r for p in predictions.values() for r in (p.get("line_items") or [])]
    null_rows = sum(all(_blank(v) for v in r.values()) for r in all_rows)
    checks.append(
        Check("f_no_all_null_rows", null_rows == 0, f"{null_rows} of {len(all_rows)} rows")
    )
    if require_logprobs:
        checks.append(check_logprobs(traces))
    return checks


def _emitted_fields(parsed: Mapping[str, Any]) -> set[tuple[str, int | None, str]]:
    """(scope, row_idx, field) of every header / row field value of a parsed keyed page."""
    out: set[tuple[str, int | None, str]] = set()
    header = parsed.get("header")
    if isinstance(header, Mapping):
        out |= {("header", None, str(k)) for k in header}
    rows = parsed.get("line_items")
    if isinstance(rows, list):
        for i, row in enumerate(rows):
            if isinstance(row, Mapping):
                out |= {("row", i, str(k)) for k in row}
    return out


def check_logprobs(traces: Sequence[Mapping[str, Any]]) -> Check:
    """Every field of every valid page has a finite min and mean logprob, with >= 1 token.

    Fails closed: a page without ``field_logprobs`` (old trace, or capture failed), a missing
    field, a null / NaN / infinite value or a positive logprob all count as bad.
    """
    n_fields, bad = 0, []
    pages = [(t["doc_id"], p) for t in traces for p in t["pages"] if p.get("json_valid")]
    for doc_id, p in pages:
        entries = p.get("field_logprobs")
        if not isinstance(entries, list):
            bad.append(f"{doc_id} p{p['page']}: no field_logprobs")
            continue
        got = {(e["scope"], e["row_idx"], e["field"]): e for e in entries}
        for key in _emitted_fields(p["parsed"]):
            n_fields += 1
            e = got.get(key)
            ok = e is not None and e.get("n_tokens", 0) >= 1
            for stat in ("min", "mean"):
                v = e.get(stat) if e is not None else None
                ok = ok and isinstance(v, (int, float)) and math.isfinite(v) and v <= 1e-6
            if not ok:
                bad.append(f"{doc_id} p{p['page']} {key[0]}.{key[2]}")
    return Check(
        "g_field_logprobs",
        bool(pages) and not bad,
        f"{n_fields} fields on {len(pages)} pages; bad {len(bad)}"
        + (f" e.g. {bad[:3]}" if bad else ""),
    )


def load_and_check(
    run_dir: Path, labels_dir: Path, max_new_tokens: int, require_logprobs: bool = False
) -> list[Check]:
    """Read ``trace.jsonl`` + ``predictions.json`` of a run and the gold labels, then check."""
    lines = (Path(run_dir) / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    traces = [json.loads(ln) for ln in lines if ln.strip()]
    preds = json.loads((Path(run_dir) / "predictions.json").read_text(encoding="utf-8"))
    gold = {
        t["doc_id"]: json.loads(
            (Path(labels_dir) / f"{t['doc_id']}.json").read_text(encoding="utf-8")
        )
        for t in traces
    }
    return check_smoke(traces, preds, gold, max_new_tokens, require_logprobs)


def format_table(model: str, checks: Sequence[Check], warnings: Sequence[Warn] = ()) -> str:
    """Per-assertion PASS/FAIL table for one model, then any non-blocking WARN lines."""
    w = max(len(c.name) for c in [*checks, *warnings])
    lines = [f"smoke assertions: {model}"]
    lines += [f"  {c.name.ljust(w)}  {'PASS' if c.passed else 'FAIL'}  {c.detail}" for c in checks]
    lines += [f"  {x.name.ljust(w)}  WARN  {x.detail} (non-blocking)" for x in warnings]
    return "\n".join(lines)
