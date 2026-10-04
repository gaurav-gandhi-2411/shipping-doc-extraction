"""Multi-page merge (spec Phase 3.1).

Rules, all value-preserving (the merge picks WHICH page's value to trust, never edits a value):

* ``doc_type``: majority vote over pages; a tie goes to the earliest tied page (page 1 first).
* Identity header fields come from page 1 only. A value seen only on a continuation page (e.g. the
  "<TITLE> - continued <invoice number>" banner, recon section 9) must not fill a page-1 null.
* Totals (``total_amount``; waybill ``pieces`` / ``gross_weight_kg``): the generic rule is the
  page whose VLM output has a non-null value, preferring the last page (a labelled total can
  also sit on a page-1 running subtotal; the last page wins). There are no per-supplier-group
  overrides: reports/total_rule.md measures the generic rule against the old table. A
  `FieldProvenance` table may still pin a FIELD (not a group) to page 1.
* Rows are concatenated in page order. Repeated table-header rows are dropped; identical rows are
  NEVER deduplicated (2-3% of docs legitimately repeat a full row). All-null rows (no field set) are
  dropped (``drop_null_rows``, default on): in the spike40 traces nearly all sit in a trailing run
  at the end of a page (reports/spike_diagnosis.md), and a row with no value can never match a
  gold row, so it only lowers row precision.
* No PO propagation (Step H verdict, reports/po_check.md).
* Per-field cross-page disagreement is returned for a later confidence feature.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shipdoc.extract import INVOICE_KEYS, ROW_KEYS, WAYBILL_KEYS

ROOT = Path(__file__).resolve().parents[2]
PROVENANCE_PATH = ROOT / "meta" / "field_provenance.json"

#: Header keys that are totals (located by provenance); every other header key is identity.
TOTAL_FIELDS: dict[str, tuple[str, ...]] = {
    "invoice": ("total_amount",),
    "waybill": ("pieces", "gross_weight_kg"),
}
HEADER_FIELDS: dict[str, tuple[str, ...]] = {"invoice": INVOICE_KEYS, "waybill": WAYBILL_KEYS}

# Words printed in table-header rows. A row whose every non-null field is made only of these words
# is a repeated column-header row, not data.
HEADER_WORDS = frozenset(
    {
        "part",
        "parts",
        "no",
        "number",
        "num",
        "nr",
        "item",
        "items",
        "description",
        "desc",
        "qty",
        "quantity",
        "quantities",
        "qnty",
        "units",
        "unit",
        "po",
        "p",
        "o",
        "purchase",
        "order",
        "orders",
        "customer",
        "cust",
        "customers",
        "supplier",
        "suppliers",
        "seller",
        "sellers",
        "buyer",
        "buyers",
        "your",
        "our",
        "ref",
        "reference",
        "code",
        "pn",
        "n",
        "sku",
        "cpn",
        "spn",
        "mfr",
        "manufacturer",
    }
)

POLICY_FIRST = "first"
POLICY_LAST_NONNULL = "last_nonnull"
# Rule names as emitted by scripts/provenance.py (meta/field_provenance.json) -> merge policy.
RULE_TO_POLICY = {
    "page1_header": POLICY_FIRST,
    "page1": POLICY_FIRST,
    "first": POLICY_FIRST,
    "last_page_footer": POLICY_LAST_NONNULL,
    "last_page": POLICY_LAST_NONNULL,
    "last_nonnull": POLICY_LAST_NONNULL,
    "any_page": POLICY_LAST_NONNULL,
}


def _blank(v: Any) -> bool:
    return v is None or (isinstance(v, str) and not v.strip())


class FieldProvenance:
    """Which page a total field is read from. Drop-in point for the learned provenance table.

    Accepted JSON shapes (all tolerated; unknown rules fall back to the default and are listed in
    ``warnings``): ``{"header_fields": {<field>: {"recommended": {"rule": R} | null}}}`` as
    written by scripts/provenance.py (``fields`` also accepted as the wrapper key, or no
    wrapper), or a flat ``{<field>: R}`` / ``{"<doc_type>.<field>": R}``. A ``group_overrides``
    entry in the table is IGNORED: rules are per field, never per supplier group. Rules
    ``page1_header``/``page1`` pin a field to page 1; ``last_page``/``last_page_footer``/
    ``any_page`` read the last page with a value.
    """

    def __init__(
        self,
        rules: dict[str, str] | None = None,
        source: str = "default",
        warnings: list[str] | None = None,
    ) -> None:
        self.rules = rules or {}
        self.source = source
        self.warnings = warnings or []

    @classmethod
    def default(cls) -> FieldProvenance:
        """No table: every total uses "last page that has a non-null value"."""
        return cls()

    @classmethod
    def load(cls, path: str | Path | None = None) -> FieldProvenance:
        """Read `path` (default meta/field_provenance.json); the default policy if it is absent."""
        p = Path(path) if path else PROVENANCE_PATH
        if not p.is_file():
            return cls.default()
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls.from_dict(data, source=str(p.name))

    @classmethod
    def from_dict(cls, data: dict[str, Any], source: str = "dict") -> FieldProvenance:
        """Parse any accepted shape (see class docstring)."""
        table = {}
        if isinstance(data, dict):
            table = data.get("header_fields") or data.get("fields") or data
        rules: dict[str, str] = {}
        warnings: list[str] = []

        def policy(rule: Any, where: str) -> str | None:
            if isinstance(rule, str) and rule in RULE_TO_POLICY:
                return RULE_TO_POLICY[rule]
            warnings.append(f"{where}: unknown rule {rule!r}, using default")
            return None

        for name, entry in table.items():
            if not isinstance(entry, (dict, str)):
                continue
            rec = entry.get("recommended", entry) if isinstance(entry, dict) else entry
            rule = rec.get("rule") if isinstance(rec, dict) else rec
            if rule is None:
                continue
            pol = policy(rule, name)
            if pol:
                rules[name] = pol
        return cls(rules, source, warnings)

    def policy_for(self, doc_type: str, field_name: str) -> str:
        """Policy name for one total field (``first`` or ``last_nonnull``)."""
        for key in (f"{doc_type}.{field_name}", field_name):
            if key in self.rules:
                return self.rules[key]
        return POLICY_LAST_NONNULL

    def page_for(
        self,
        doc_type: str,
        field_name: str,
        values: Sequence[Any],
    ) -> int | None:
        """Index of the page the field is read from, or None when no page has a value."""
        if self.policy_for(doc_type, field_name) == POLICY_FIRST:
            return 0 if values and not _blank(values[0]) else None
        for i in range(len(values) - 1, -1, -1):
            if not _blank(values[i]):
                return i
        return None


# --------------------------------------------------------------------------------------------


def is_header_row(row: dict[str, Any]) -> bool:
    """True if every non-null field of `row` consists only of table-column-label words.

    A value with a digit is data, never a column label: ``PO5551230001`` or ``PN-100`` reduce to the
    label words ``po`` / ``pn`` once the digits are ignored and used to drop real rows.
    """
    vals = [str(v) for v in row.values() if not _blank(v)]
    if not vals:
        return False
    for v in vals:
        if re.search(r"\d", v):
            return False
        words = re.findall(r"[a-z]+", v.lower().replace("'", ""))
        if not words or any(w not in HEADER_WORDS for w in words):
            return False
    return True


def _norm_val(v: Any) -> str:
    return re.sub(r"\s+", "", str(v)).lower()


@dataclass
class MergeResult:
    """Merged document plus everything the trace and the confidence model need."""

    doc: dict[str, Any]
    diffs: dict[str, Any] = field(default_factory=dict)


def _vote_doc_type(pages: Sequence[dict[str, Any] | None]) -> tuple[str | None, dict[str, int]]:
    votes = [p.get("doc_type") for p in pages if p and p.get("doc_type") in HEADER_FIELDS]
    if not votes:
        return None, {}
    counts = Counter(votes)
    top = max(counts.values())
    tied = {k for k, n in counts.items() if n == top}
    winner = next(v for v in votes if v in tied)  # earliest page among the tied types
    return winner, dict(counts)


def merge_pages(
    pages: Sequence[dict[str, Any] | None],
    provenance: FieldProvenance | None = None,
    drop_null_rows: bool = True,
    *,
    provenance_aware: bool = True,
    drop_header_rows: bool = True,
    total_page_hints: Mapping[str, int] | None = None,
) -> MergeResult:
    """Merge per-page model outputs (``None`` = parse failure) into one schema document.

    The keyword switches exist for the Phase 3 ablation ladder (scripts/postproc_ablation.py);
    the defaults are the pipeline. ``provenance_aware=False`` is the "any-page" baseline with no
    Phase 3 merge logic: doc_type of the first parsed page, every header field = the first non-null
    value over the pages in page order. ``drop_header_rows=False`` keeps repeated column-header
    rows. ``total_page_hints`` maps a total field to the 0-based page an OCR labelled-total
    detector points at (`shipdoc.totals`): that page's value is used when it is non-null, else the
    provenance policy decides (a hint picks WHICH page's value to trust, it never supplies one).

    ``diffs`` keys: ``doc_type_votes``, ``page_kinds``, ``field_pages`` (which page each header
    value came from), ``disagreement`` (field -> {page_no: value} when >=2 pages hold different
    non-null values), ``ignored_elsewhere`` (identity fields null on page 1 but non-null on a later
    page: the page-2 trap), ``dropped_header_rows``, ``dropped_waybill_rows``,
    ``dropped_null_rows`` (page numbers of all-null rows dropped), ``failed_pages``.
    """
    prov = provenance or FieldProvenance.default()
    doc_type, votes = _vote_doc_type(pages)
    if not provenance_aware and votes:
        doc_type = next(p["doc_type"] for p in pages if p and p.get("doc_type") in HEADER_FIELDS)
    diffs: dict[str, Any] = {
        "doc_type_votes": votes,
        "page_kinds": [p.get("page_kind") if p else None for p in pages],
        "failed_pages": [i + 1 for i, p in enumerate(pages) if p is None],
        "field_pages": {},
        "disagreement": {},
        "ignored_elsewhere": {},
        "dropped_header_rows": [],
        "dropped_null_rows": [],
        "dropped_waybill_rows": 0,
        "provenance_source": prov.source,
    }
    if doc_type is None:
        # No usable page at all: fall back to an all-null invoice (the scorer then counts every
        # field wrong, never a false fill).
        doc_type = "invoice"
        diffs["doc_type_fallback"] = True

    headers = [(p.get("header") if p and isinstance(p.get("header"), dict) else {}) for p in pages]
    totals = TOTAL_FIELDS[doc_type]
    header: dict[str, Any] = {}
    for name in HEADER_FIELDS[doc_type]:
        values = [h.get(name) for h in headers]
        hint = (total_page_hints or {}).get(name)
        if not provenance_aware:
            page = next((i for i, v in enumerate(values) if not _blank(v)), None)
            header[name] = values[page] if page is not None else None
            diffs["field_pages"][name] = None if page is None else page + 1
        elif name in totals:
            if hint is not None and 0 <= hint < len(values) and not _blank(values[hint]):
                page = hint
            else:
                page = prov.page_for(doc_type, name, values)
            header[name] = values[page] if page is not None else None
            diffs["field_pages"][name] = None if page is None else page + 1
        else:
            header[name] = values[0] if values else None
            diffs["field_pages"][name] = 1 if not _blank(header[name]) else None
            later = [i + 1 for i, v in enumerate(values) if i > 0 and not _blank(v)]
            if _blank(header[name]) and later:
                diffs["ignored_elsewhere"][name] = later
        seen = {i + 1: v for i, v in enumerate(values) if not _blank(v)}
        if len({_norm_val(v) for v in seen.values()}) > 1:
            diffs["disagreement"][name] = seen

    rows: list[dict[str, Any]] = []
    for pno, p in enumerate(pages, start=1):
        for raw in (p.get("line_items") if p else None) or []:
            if not isinstance(raw, dict):
                continue
            row = {k: raw.get(k) for k in ROW_KEYS}
            if drop_header_rows and is_header_row(row):
                diffs["dropped_header_rows"].append(pno)
                continue
            if drop_null_rows and all(_blank(v) for v in row.values()):
                diffs["dropped_null_rows"].append(pno)
                continue
            rows.append(row)
    if doc_type == "waybill":
        diffs["dropped_waybill_rows"] = len(rows)
        rows = []  # schema: line_items is empty for waybills
    return MergeResult({"doc_type": doc_type, "header": header, "line_items": rows}, diffs)
