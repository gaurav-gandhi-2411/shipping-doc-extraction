"""Candidate merge rules R1 / R2 / R3 as pure functions (GATE CANDIDATES, not in the pipeline).

None of this is wired into ``shipdoc.merge`` or ``shipdoc.predict``; ``scripts/rule_gate.py`` is
the only caller until a rule passes its gate. The rules mirror ``scripts/row_error_diagnosis.py``
(``rule_carrier_from_supplier``, ``rule_pattern_backfill``, ``rule_slot_shape``) but never look at
gold: that script's versions take the gold labels to record before / after correctness, which
makes them unusable at inference time.

Pipeline stage: they operate on a MERGED document dict (``{"doc_type", "header", "line_items"}``,
the output of ``merge.merge_pages``) with whatever value types the merge produced, strictly BEFORE
schema coercion (a separate module: this one does not depend on it and compares values only by
emptiness and by format shape, never by numeric value). Every rule returns a patched deep copy
plus the list of ``Change`` records it made; the input is never mutated.

* R1 ``carrier_from_supplier``: a waybill whose carrier is empty takes the model's OWN
  ``supplier_name`` slot. The merged doc does not carry ``supplier_name``, so the caller passes
  the page-1 value of the model's parsed header. A model-provided carrier is never overridden.
* R2 ``pattern_backfill``: an empty ``mawb`` / ``hawb`` takes the pattern match found in `text`
  when there is exactly ONE distinct candidate; zero or several candidates leave it null. The
  patterns are format facts from the spec / ``reports/recon.md`` section 7 (MAWB ``999-99999999``,
  HAWB two letters + 8 digits), not values. DEPLOYMENT COST: `text` is the OCR text of the doc, so
  R2 needs OCR (PaddleOCR) at inference time on the test set, a different deployment cost from the
  image-only pick. ``text`` may instead be the model's own raw output (the "OCR-free" variant the
  gate evaluates).
* R3 ``slot_shape``: cpn / po values whose format shape (digits -> 9, letters -> A) occurs ONLY in
  the other column of the TRAIN gold are moved there. A shape that occurs in both columns, or in
  neither, is ambiguous and never triggers a move. With one slot empty the lone value moves; with
  both present the two values swap only when BOTH shapes are unambiguous and point at the
  opposite slot.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from shipdoc.diagnostics import empty

CPN = "customer_part_number"
PO = "purchase_order"

#: MAWB ``999-99999999`` (recon section 7: 100/100 gold values) and HAWB ``AA99999999``.
PATTERNS: dict[str, re.Pattern[str]] = {
    "mawb": re.compile(r"(?<!\d)\d{3}-\d{8}(?!\d)"),
    "hawb": re.compile(r"(?<![A-Z0-9])[A-Z]{2}\d{8}(?![A-Z0-9])"),
}


@dataclass(frozen=True)
class Change:
    """One edit made by a rule. `row` is None for header cells; R3 edits whole rows."""

    rule: str
    field: str
    row: int | None
    before: Any
    after: Any


@dataclass(frozen=True)
class SlotShapes:
    """Format shapes that occur in exactly one of the cpn / po columns of the learning gold."""

    cpn_only: frozenset[str]
    po_only: frozenset[str]


def shape_of(value: Any) -> str:
    """Format shape of an identifier: digits -> 9, letters -> A, punctuation kept."""
    return re.sub(r"[A-Za-z]", "A", re.sub(r"\d", "9", str(value)))


def learn_slot_shapes(labels: Sequence[Mapping[str, Any]]) -> SlotShapes:
    """Shapes seen in the cpn and po columns of invoice `labels` (pass TRAIN gold only)."""
    cpn: set[str] = set()
    po: set[str] = set()
    for g in labels:
        if g.get("doc_type") != "invoice":
            continue
        for r in g.get("line_items") or []:
            if not empty(r.get(CPN)):
                cpn.add(shape_of(r[CPN]))
            if not empty(r.get(PO)):
                po.add(shape_of(r[PO]))
    return SlotShapes(cpn_only=frozenset(cpn - po), po_only=frozenset(po - cpn))


def carrier_from_supplier(
    doc: Mapping[str, Any], supplier_name: Any
) -> tuple[dict[str, Any], list[Change]]:
    """R1: empty waybill carrier <- `supplier_name` (the model's own slot, page 1)."""
    out = copy.deepcopy(dict(doc))
    if out.get("doc_type") != "waybill" or empty(supplier_name):
        return out, []
    hdr = out.setdefault("header", {})
    if not empty(hdr.get("carrier")):
        return out, []
    before = hdr.get("carrier")
    hdr["carrier"] = supplier_name
    return out, [Change("R1", "carrier", None, before, supplier_name)]


def pattern_candidates(field: str, text: str) -> list[str]:
    """Distinct pattern matches of `field` (mawb / hawb) in `text`, sorted."""
    return sorted(set(PATTERNS[field].findall(text or "")))


def pattern_backfill(doc: Mapping[str, Any], text: str) -> tuple[dict[str, Any], list[Change]]:
    """R2: empty mawb / hawb <- the unique pattern match in `text`; else untouched (null)."""
    out = copy.deepcopy(dict(doc))
    changes: list[Change] = []
    if out.get("doc_type") != "waybill":
        return out, changes
    hdr = out.setdefault("header", {})
    for f in ("mawb", "hawb"):
        if not empty(hdr.get(f)):
            continue
        found = pattern_candidates(f, text)
        if len(found) != 1:
            continue
        changes.append(Change("R2", f, None, hdr.get(f), found[0]))
        hdr[f] = found[0]
    return out, changes


def slot_shape(doc: Mapping[str, Any], shapes: SlotShapes) -> tuple[dict[str, Any], list[Change]]:
    """R3: move / swap cpn and po values by format shape (see module doc)."""
    out = copy.deepcopy(dict(doc))
    changes: list[Change] = []
    if out.get("doc_type") != "invoice":
        return out, changes
    for i, r in enumerate(x for x in (out.get("line_items") or []) if isinstance(x, dict)):
        c, p = r.get(CPN), r.get(PO)
        new: tuple[Any, Any] | None = None
        if not empty(c) and empty(p) and shape_of(c) in shapes.po_only:
            new = (None, c)
        elif not empty(p) and empty(c) and shape_of(p) in shapes.cpn_only:
            new = (p, None)
        elif (
            not empty(c)
            and not empty(p)
            and shape_of(c) in shapes.po_only
            and shape_of(p) in shapes.cpn_only
        ):
            new = (p, c)
        if new is None:
            continue
        changes.append(Change("R3", "cpn_po", i, (c, p), new))
        r[CPN], r[PO] = new
    return out, changes
