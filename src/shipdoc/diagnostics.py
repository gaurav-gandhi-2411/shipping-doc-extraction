"""Row-convention error counts shared by the Step L diagnosis and the keyed-vs-compact A/B.

The definitions are those of ``scripts/spike_diagnosis.py`` (Step L, ``reports/spike_diagnosis.md``
hypothesis (b)). Unpaired gold rows are the rows ``eval._pair_rows`` (a mirror of the official
scorer's greedy matching) leaves unmatched. Each is attributed, in order, to the first cause that
applies:

* the doc has a truncated page -> ``truncated`` (not counted as a column shift);
* no free (unpaired) predicted row shares any value with it -> ``missing``;
* the best-overlapping free predicted row holds as many *cross-field* matches (a predicted field
  equals a different gold field of the row, alphanumerics only, ``quantity`` excluded) as
  same-field ones -> a column shift (``rotation`` here, including the clean swap);
* otherwise ``spn_null`` (predicted supplier_part_number empty) or ``misread_spn``.

A *clean swap* is a column shift whose cross matches contain both (spn -> cpn) and (cpn -> spn).
``cpn_equals_po`` counts predicted rows whose customer_part_number is non-empty and equal, under
the scorer's identifier rule, to the same row's purchase_order. ``cpn_filled_gold_empty`` counts
predicted rows with a non-empty customer_part_number on invoices whose gold rows have none.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from shipdoc import eval as ev
from shipdoc.replay import read_trace

COUNT_KEYS = (
    "rotations",
    "clean_swaps",
    "cpn_equals_po",
    "cpn_filled_gold_empty",
    "unpaired_gold_rows",
)


def empty(v: Any) -> bool:
    """Blank value: None or whitespace-only string."""
    return v is None or (isinstance(v, str) and not v.strip())


def alnum(v: Any) -> str:
    """Lowercase alphanumerics only (the Step L comparison key)."""
    return re.sub(r"[^0-9a-z]", "", str(v).lower())


def attribute_unpaired(
    sc: ModuleType,
    pr: Sequence[Mapping[str, Any]],
    gr: Sequence[Mapping[str, Any]],
    unp: Sequence[int],
    ung: Sequence[int],
    truncated: bool,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Attribute every unpaired gold row to a cause (see module doc).

    Returns (one dict per unpaired gold row in `ung` order, the predicted rows left unused).
    Each dict has ``gi``, ``cause`` in {truncated, missing, shift, spn_null, misread_spn} and, for
    non-truncated rows with an overlapping prediction, ``pi``, ``same`` and ``cross`` lists;
    ``shift`` rows also carry ``swap``.
    """
    free = list(unp)
    out: list[dict[str, Any]] = []
    for gi in ung:
        if truncated:
            out.append({"gi": gi, "cause": "truncated"})
            continue
        best: tuple[int, int, list[str], list[tuple[str, str]]] | None = None
        for pi in free:
            same_f = [
                f
                for f in sc.ROW
                if not empty(gr[gi].get(f)) and sc.same(f, pr[pi].get(f), gr[gi].get(f))
            ]
            cross = [
                (f1, f2)
                for f1 in sc.ROW
                for f2 in sc.ROW
                if f1 != f2
                and "quantity" not in (f1, f2)
                and not empty(pr[pi].get(f1))
                and not empty(gr[gi].get(f2))
                and alnum(pr[pi].get(f1)) == alnum(gr[gi].get(f2))
            ]
            score = len(same_f) + len(cross)
            if score and (best is None or score > best[0]):
                best = (score, pi, same_f, cross)
        if best is None:
            out.append({"gi": gi, "cause": "missing"})
            continue
        _, pi, same_f, cross = best
        free.remove(pi)
        rec: dict[str, Any] = {"gi": gi, "pi": pi, "same": same_f, "cross": cross}
        if cross and len(cross) >= len(same_f):
            swap = ("supplier_part_number", "customer_part_number") in cross and (
                "customer_part_number",
                "supplier_part_number",
            ) in cross
            rec.update(cause="shift", swap=swap)
        elif empty(pr[pi].get("supplier_part_number")):
            rec["cause"] = "spn_null"
        else:
            rec["cause"] = "misread_spn"
        out.append(rec)
    return out, free


def row_convention_errors(
    pred: Mapping[str, Any],
    gold: Mapping[str, Any],
    truncated: bool = False,
    sc: ModuleType | None = None,
) -> dict[str, int]:
    """Row-convention error counts for one document.

    Only invoice-vs-invoice pairs have rows to compare; any other pair returns all zeros (as in
    Step L). `truncated` marks a doc with an invalid/truncated page, whose unpaired gold rows are
    never counted as rotations. Keys: see ``COUNT_KEYS``.
    """
    counts = dict.fromkeys(COUNT_KEYS, 0)
    if gold.get("doc_type") != "invoice" or pred.get("doc_type") != "invoice":
        return counts
    sc = sc or ev.load_scorer()
    gr = list(gold.get("line_items") or [])
    pr = [x for x in (pred.get("line_items") or []) if isinstance(x, dict)]
    _, _, unp, ung = ev._pair_rows(sc, pr, gr)
    counts["unpaired_gold_rows"] = len(ung)
    attributed, _ = attribute_unpaired(sc, pr, gr, unp, ung, truncated)
    for rec in attributed:
        if rec["cause"] == "shift":
            counts["rotations"] += 1
            counts["clean_swaps"] += bool(rec["swap"])
    for x in pr:
        cpn, po = x.get("customer_part_number"), x.get("purchase_order")
        if not empty(cpn) and not empty(po) and sc.same("purchase_order", cpn, po):
            counts["cpn_equals_po"] += 1
    if all(empty(x.get("customer_part_number")) for x in gr):
        counts["cpn_filled_gold_empty"] = sum(not empty(x.get("customer_part_number")) for x in pr)
    return counts


def sum_row_convention_errors(
    preds: Mapping[str, Mapping[str, Any]],
    gold: Mapping[str, Mapping[str, Any]],
    truncated_docs: Sequence[str] = (),
) -> dict[str, int]:
    """Per-doc ``row_convention_errors`` summed over the docs of `gold` (missing pred = empty)."""
    sc = ev.load_scorer()
    total = dict.fromkeys(COUNT_KEYS, 0)
    skip = set(truncated_docs)
    for did, g in gold.items():
        c = row_convention_errors(preds.get(did) or {}, g, did in skip, sc)
        for k in COUNT_KEYS:
            total[k] += c[k]
    return total


def truncated_docs(run_dir: Path) -> list[str]:
    """doc_ids of a run with at least one page whose JSON was invalid (from trace.jsonl)."""
    traces = read_trace(Path(run_dir) / "trace.jsonl")
    return [t["doc_id"] for t in traces if any(not p["json_valid"] for p in t["pages"])]


def run_row_convention_errors(
    run_dir: Path, gold: Mapping[str, Mapping[str, Any]]
) -> dict[str, int]:
    """Summed counts for a run dir, restricted to the docs in its predictions.json."""
    preds = json.loads((Path(run_dir) / "predictions.json").read_text(encoding="utf-8"))
    sub = {d: g for d, g in gold.items() if d in preds}
    return sum_row_convention_errors(preds, sub, truncated_docs(run_dir))
