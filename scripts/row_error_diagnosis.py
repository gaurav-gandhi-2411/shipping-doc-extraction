"""Row-level error diagnosis of one finished run (CPU only, no model is called).

Breaks the structural row errors of a run (``row_missing`` = gold rows the official scorer leaves
unpaired, ``row_extra`` = predicted rows it leaves unpaired) into mutually exclusive causes, looks
at the cpn / po slot mix-ups of paired rows, at the waybill header over-nulls, and prices every
cause with an ORACLE replay (gold substituted, unmodified scorer, paired bootstrap).

Inputs: ``<run>/{predictions.json,trace.jsonl}``, gold labels, the OCR cache (``shipdoc.ocr``), the
locator (``shipdoc.locate``), ``meta/{dev,train}.json`` and ``meta/supplier_groups.json``.
Outputs: ``reports/row_errors.md`` (aggregates and counts only: no gold or predicted values, no
names, supplier groups as ``inv_gNN`` / ``wb_gNN``) and a local value-level file outside the repo
(``<SHIPDOC_RUNS_DIR>/diagnosis/row_errors_local.md``).

Cause decision order (each unpaired row gets exactly one cause; see ``classify_*``):

* Unpaired gold and predicted rows of a doc are first LINKED one-to-one (``link_unpaired``): by
  identifier evidence (a same-field or cross-field identifier match among supplier_part_number,
  customer_part_number, purchase_order), then by equal quantity on the same page, then by position
  when a page has the same number of leftovers on both sides.
* Linked pair: ``spn_copies_other_slot`` (the predicted part number is a copy of the same row's
  cpn / po) > ``column_shift`` (cross-field matches >= same-field matches) > ``spn_null`` >
  ``spn_misread`` (taxonomy definition: normalized edit distance <= 2 or similarity >= 0.8) >
  ``spn_other`` (linked by another identifier, part number far off) > ``weak_link_*`` (linked by
  quantity or position only, part number not a near misread).
* Unlinked gold row: ``truncated_page`` > ``page_dropped`` (no predicted row from its page at all)
  > ``merged`` (one predicted row carries the summed quantity of two gold rows) > ``page_short``
  (its page has fewer predicted than gold rows) > ``missing_other``.
* Unlinked predicted row: ``header_row`` (a value made only of column-label words) > ``split``
  (with another predicted row it carries the part number and the summed quantity of one gold
  row) > ``duplicate_boundary`` (same part number as a row of an adjacent page, both near a page
  boundary, more copies predicted than in gold) > ``duplicate_other`` > ``extra_other``.

Every number is UNVERIFIED until a verifier recomputes it. Run:
``uv run python scripts/row_error_diagnosis.py --run-dir <run dir>``.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rapidfuzz.distance import Levenshtein

from shipdoc import diagnostics as dg
from shipdoc import eval as ev
from shipdoc import layout, locate, merge, paths
from shipdoc import meta as meta_mod
from shipdoc.extract import ROW_KEYS
from shipdoc.ocr import page_items, page_text
from shipdoc.replay import read_trace, salvage_page_json
from shipdoc.trainset import resolve_row_pages

ROOT = Path(__file__).resolve().parents[1]
N_BOOT = 2000
SEED = 42
SPN, CPN, PO, QTY = "supplier_part_number", "customer_part_number", "purchase_order", "quantity"
IDENT = (SPN, CPN, PO)
#: "Near a page boundary" = first / last K rows of a page (by gold order for gold rows, by
#: predicted order for predicted rows).
BOUNDARY_K = 3
POSITIONS = ("only", "first", "middle", "last", "unknown")
LENGTH_BUCKETS = (("1-5", 1, 5), ("6-15", 6, 15), ("16-25", 16, 25), ("26+", 26, 10_000))
PAIR_CAUSES = (
    "spn_copies_other_slot",
    "column_shift",
    "spn_null",
    "spn_misread",
    "spn_other",
    "weak_link_qty",
    "weak_link_position",
)
GOLD_ONLY_CAUSES = ("truncated_page", "page_dropped", "merged", "page_short", "missing_other")
PRED_ONLY_CAUSES = ("header_row", "split", "duplicate_boundary", "duplicate_other", "extra_other")
ALL_CAUSES = PAIR_CAUSES + GOLD_ONLY_CAUSES + PRED_ONLY_CAUSES

Same = Callable[[str, Any, Any], bool]

#: Visually confusable character pairs (lowercase alphanumerics) for the misread edit profile.
#: A fixed descriptive list; s8 / wm / nm / nh were added after the first run showed them as the
#: residue of ``other_substitution``, so ``glyph_confusion`` is a description, not a test.
CONFUSABLE = frozenset(
    frozenset(p)
    for p in ("o0", "il", "i1", "l1", "s5", "b8", "z2", "g6", "q0", "d0", "uv", "ec")
    + ("s8", "wm", "nm", "nh")
)

# Column-label words that mark a customer-part / PO column in the table-header line (OCR text).
CPN_LABEL = re.compile(r"cust(?:omer|\.)?\s*(?:part|p/?n)|your\s+part|customer\s+p/?n", re.I)
PO_LABEL = re.compile(r"\bpo\b|p\.o\.|purchase\s+order|your\s+order|customer\s+po", re.I)
WB_LABEL = {
    "mawb": re.compile(r"\bmawb\b|master\s+(?:air\s*)?waybill", re.I),
    "hawb": re.compile(r"\bhawb\b|house\s+(?:air\s*)?waybill", re.I),
    "carrier": re.compile(r"\bcarrier\b|\bairline\b", re.I),
}
WB_PATTERN = {
    "mawb": re.compile(r"(?<!\d)\d{3}-\d{8}(?!\d)"),
    "hawb": re.compile(r"(?<![A-Z0-9])[A-Z]{2}\d{8}(?![A-Z0-9])"),
}


# --------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------


def md_table(head: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Markdown table; every cell is str()-ed."""
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def rate(num: int, den: int) -> str:
    """``num/den (xx.x%)`` or ``num/0 (n/a)``."""
    return f"{num}/{den} ({100 * num / den:.1f}%)" if den else f"{num}/0 (n/a)"


def pts(x: float, signed: bool = True) -> str:
    """Fraction -> percentage points with two decimals."""
    return f"{100 * x:+.2f}" if signed else f"{100 * x:.2f}"


def ranks(pages: Sequence[int | None]) -> tuple[list[int | None], dict[int, int]]:
    """Rank of every row among the rows of its page (list order) and the row count per page."""
    seen: Counter[int] = Counter()
    out: list[int | None] = []
    for p in pages:
        if p is None:
            out.append(None)
        else:
            out.append(seen[p])
            seen[p] += 1
    return out, dict(seen)


def near_boundary(rank: int | None, n_on_page: int, k: int = BOUNDARY_K) -> bool | None:
    """True when the row is one of the first / last `k` of its page (None = page unknown)."""
    if rank is None:
        return None
    return rank < k or rank >= n_on_page - k


def length_bucket(n_rows: int) -> str:
    """Bucket label of a doc's number of gold rows."""
    for name, lo, hi in LENGTH_BUCKETS:
        if lo <= n_rows <= hi:
            return name
    return "0"


def label_only(value: Any) -> bool:
    """True if `value` is made only of table-column-label words (and has no digit)."""
    if dg.empty(value):
        return False
    s = str(value)
    if re.search(r"\d", s):
        return False
    words = re.findall(r"[a-z]+", s.lower().replace("'", ""))
    return bool(words) and all(w in merge.HEADER_WORDS for w in words)


# --------------------------------------------------------------------------------------------
# Linking the unpaired rows of one doc
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Link:
    """One gold row linked to one predicted row; `kind` = ident / qty / position."""

    gi: int
    pi: int
    kind: str


def ident_evidence(
    g: Mapping[str, Any], p: Mapping[str, Any], same: Same
) -> tuple[list[str], list[tuple[str, str]]]:
    """(same-field identifier matches, cross-field identifier matches) of a predicted row vs gold.

    Same-field uses the scorer's rule; cross-field means a predicted field equals a DIFFERENT gold
    identifier field (alphanumerics only, as in ``shipdoc.diagnostics``). Quantity is excluded:
    it is too weak (many rows share a quantity) to link rows on its own.
    """
    same_f = [f for f in IDENT if not dg.empty(g.get(f)) and same(f, p.get(f), g.get(f))]
    cross = [
        (f1, f2)
        for f1 in IDENT
        for f2 in IDENT
        if f1 != f2
        and not dg.empty(p.get(f1))
        and not dg.empty(g.get(f2))
        and dg.alnum(p.get(f1)) == dg.alnum(g.get(f2))
    ]
    return same_f, cross


def link_unpaired(
    gr: Sequence[Mapping[str, Any]],
    pr: Sequence[Mapping[str, Any]],
    free_g: Sequence[int],
    free_p: Sequence[int],
    gpage: Sequence[int | None],
    ppage: Sequence[int | None],
    same: Same,
) -> tuple[list[Link], list[int], list[int]]:
    """Link unpaired gold rows to unpaired predicted rows one-to-one; returns the leftovers too.

    Pass 1 (``ident``): candidates with >= 1 identifier match, greedy by (evidence desc, same page
    first, small rank distance on the page). Pass 2 (``qty``): equal non-empty quantity on the same
    page, smallest rank distance first. Pass 3 (``position``): per page, if the same number of
    gold and predicted rows is left, pair them in order.
    """
    grank, _ = ranks(gpage)
    prank, _ = ranks(ppage)

    def dist(gi: int, pi: int) -> int:
        a, b = grank[gi], prank[pi]
        return abs(a - b) if a is not None and b is not None and gpage[gi] == ppage[pi] else 999

    links: list[Link] = []
    lg, lp = list(free_g), list(free_p)
    cands = []
    for gi in lg:
        for pi in lp:
            sf, cr = ident_evidence(gr[gi], pr[pi], same)
            if sf or cr:
                cands.append((-(len(sf) + len(cr)), gpage[gi] != ppage[pi], dist(gi, pi), gi, pi))
    for *_, gi, pi in sorted(cands):
        if gi in lg and pi in lp:
            lg.remove(gi)
            lp.remove(pi)
            links.append(Link(gi, pi, "ident"))
    qcands = []
    for gi in lg:
        for pi in lp:
            if (
                gpage[gi] is not None
                and gpage[gi] == ppage[pi]
                and not dg.empty(gr[gi].get(QTY))
                and same(QTY, pr[pi].get(QTY), gr[gi].get(QTY))
            ):
                qcands.append((dist(gi, pi), gi, pi))
    for _, gi, pi in sorted(qcands):
        if gi in lg and pi in lp:
            lg.remove(gi)
            lp.remove(pi)
            links.append(Link(gi, pi, "qty"))
    for page in sorted({gpage[i] for i in lg if gpage[i] is not None}):  # type: ignore[type-var]
        gs = [i for i in lg if gpage[i] == page]
        ps = [i for i in lp if ppage[i] == page]
        if gs and len(gs) == len(ps):
            for gi, pi in zip(gs, ps, strict=True):
                lg.remove(gi)
                lp.remove(pi)
                links.append(Link(gi, pi, "position"))
    return sorted(links, key=lambda x: (x.gi, x.pi)), lg, lp


# --------------------------------------------------------------------------------------------
# Cause classification
# --------------------------------------------------------------------------------------------


def classify_pair(g: Mapping[str, Any], p: Mapping[str, Any], kind: str, same: Same) -> str:
    """Cause of a linked (gold, predicted) pair (decision order in the module doc)."""
    sf, cr = ident_evidence(g, p, same)
    spn = p.get(SPN)
    if (
        not dg.empty(spn)
        and not same(SPN, spn, g.get(SPN))
        and any(not dg.empty(p.get(f)) and dg.alnum(p.get(f)) == dg.alnum(spn) for f in (CPN, PO))
    ):
        return "spn_copies_other_slot"
    if cr and len(cr) >= len(sf):
        return "column_shift"
    if dg.empty(spn):
        return "spn_null"
    if ev._is_misread(spn, g.get(SPN)):
        return "spn_misread"
    return {"qty": "weak_link_qty", "position": "weak_link_position"}.get(kind, "spn_other")


def _num(v: Any) -> float | None:
    """Quantity as a float (thousands commas tolerated), None if it is not a number."""
    try:
        return float(str(v).replace(",", "").strip())
    except ValueError:
        return None


def _qty_sum_matches(rows: Sequence[Mapping[str, Any]], total_row: Mapping[str, Any]) -> bool:
    """True if the quantities of `rows` add up to the quantity of `total_row`."""
    qs = [_num(r.get(QTY)) for r in rows]
    t = _num(total_row.get(QTY))
    return t is not None and all(q is not None for q in qs) and abs(sum(qs) - t) < 0.005  # type: ignore[arg-type]


def classify_gold_only(
    gi: int,
    gr: Sequence[Mapping[str, Any]],
    gpage: Sequence[int | None],
    pred_rows_on_page: Mapping[int, int],
    gold_rows_on_page: Mapping[int, int],
    linked_pred: Mapping[int, int],
    pr: Sequence[Mapping[str, Any]],
    truncated: bool,
    same: Same,
) -> str:
    """Cause of a gold row with no linked predicted row (decision order in the module doc).

    `linked_pred` maps a predicted row index to the gold row it is linked to (or paired with).
    ``merged``: a predicted row, already linked to another gold row with the same part number,
    whose quantity is the sum of the two gold quantities (one predicted row for two gold rows).
    """
    if truncated:
        return "truncated_page"
    page = gpage[gi]
    if page is not None and pred_rows_on_page.get(page, 0) == 0:
        return "page_dropped"
    for pi, other in linked_pred.items():
        if (
            other != gi
            and not dg.empty(gr[gi].get(SPN))
            and same(SPN, pr[pi].get(SPN), gr[gi].get(SPN))
            and same(SPN, gr[other].get(SPN), gr[gi].get(SPN))
            and _qty_sum_matches([gr[gi], gr[other]], pr[pi])
        ):
            return "merged"
    if page is not None and pred_rows_on_page.get(page, 0) < gold_rows_on_page.get(page, 0):
        return "page_short"
    return "missing_other"


def classify_pred_only(
    pi: int,
    pr: Sequence[Mapping[str, Any]],
    ppage: Sequence[int | None],
    gr: Sequence[Mapping[str, Any]],
    linked_gold: Mapping[int, int],
    same: Same,
) -> str:
    """Cause of a predicted row with no linked gold row (decision order in the module doc).

    `linked_gold` maps a gold row index to the predicted row it is linked to (or paired with).
    ``split``: the row has the part number of a gold row that is linked to another predicted row
    with that part number, and the two predicted quantities add up to the gold quantity.
    """
    row = pr[pi]
    if any(label_only(row.get(f)) for f in IDENT):
        return "header_row"
    spn = row.get(SPN)
    if dg.empty(spn):
        return "extra_other"
    for gi, other_pi in linked_gold.items():
        if (
            other_pi != pi
            and same(SPN, row.get(SPN), gr[gi].get(SPN))
            and same(SPN, pr[other_pi].get(SPN), gr[gi].get(SPN))
            and _qty_sum_matches([row, pr[other_pi]], gr[gi])
        ):
            return "split"
    twins = [
        j
        for j in range(len(pr))
        if j != pi and not dg.empty(pr[j].get(SPN)) and same(SPN, pr[j].get(SPN), spn)
    ]
    n_gold = sum(1 for g in gr if not dg.empty(g.get(SPN)) and same(SPN, spn, g.get(SPN)))
    if twins and len(twins) + 1 > n_gold:
        rk, cnt = ranks(ppage)
        for j in twins:
            pa, pb = ppage[pi], ppage[j]
            if (
                pa is not None
                and pb is not None
                and abs(pa - pb) == 1
                and near_boundary(rk[pi], cnt[pa])
                and near_boundary(rk[j], cnt[pb])
            ):
                return "duplicate_boundary"
        return "duplicate_other"
    return "extra_other"


# --------------------------------------------------------------------------------------------
# Per-document analysis
# --------------------------------------------------------------------------------------------


@dataclass
class Unit:
    """One diagnosed unit: a linked pair (side ``pair``), a gold-only or a predicted-only row."""

    doc: str
    group: str
    scanned: bool
    repeated: bool
    n_pages: int
    n_gold_rows: int
    side: str  # pair | gold | pred
    cause: str
    gi: int | None = None
    pi: int | None = None
    g_page: int | None = None
    p_page: int | None = None
    g_pos: str = "unknown"
    p_pos: str = "unknown"
    g_near: bool | None = None
    p_near: bool | None = None
    link_kind: str = ""
    gold_spn_level: str = ""  # locator row-assignment level of the gold row (line/page_fuzzy/...)
    gold_match: str = ""  # locator match level of the gold part number on its own line, if any
    extra: dict[str, Any] = field(default_factory=dict)


def predicted_row_pages(trace: Mapping[str, Any], n_final: int) -> list[int | None]:
    """0-based page of every row of the merged prediction, replaying ``merge_pages``' row loop.

    Rows are concatenated in page order after dropping repeated column-header rows and all-null
    rows (the merge defaults). If the replay does not give `n_final` rows the pages are unknown
    (all None), never guessed.
    """
    pages: list[int | None] = []
    for pno, pg in enumerate(trace["pages"]):
        parsed = pg.get("parsed")
        for raw in (parsed.get("line_items") if isinstance(parsed, dict) else None) or []:
            if not isinstance(raw, dict):
                continue
            row = {k: raw.get(k) for k in ROW_KEYS}
            if merge.is_header_row(row) or all(dg.empty(v) for v in row.values()):
                continue
            pages.append(pno)
    return pages if len(pages) == n_final else [None] * n_final


def gold_row_pages(
    gold: Mapping[str, Any], index: locate.DocIndex
) -> tuple[list[int | None], list[str], list[str]]:
    """(page, assignment level, part-number match level) of every gold row, via the locator.

    Pages follow ``trainset.resolve_row_pages`` on ``locate.assign_rows`` output (read-only use).
    The match level is the locator level (exact / normalized / fuzzy) of the part number on its
    own OCR line, empty when the row got no line.
    """
    rows = list(gold.get("line_items") or [])
    if not rows:
        return [], [], []
    got = locate.assign_rows(rows, index)
    levels = [a.level for a in got]
    res = resolve_row_pages(levels, [a.page for a in got], max(1, len(gold.get("pages") or [None])))
    return list(res.row_pages), levels, [a.match.level if a.match else "" for a in got]


def diagnose_doc(
    doc: str,
    pred: Mapping[str, Any],
    gold: Mapping[str, Any],
    trace: Mapping[str, Any],
    index: locate.DocIndex,
    ctx: Mapping[str, Any],
    sc: Any,
    truncated: bool = False,
) -> list[Unit]:
    """All units (pairs, gold-only rows, predicted-only rows) of one invoice document."""
    gr = list(gold.get("line_items") or [])
    pr = [x for x in (pred.get("line_items") or []) if isinstance(x, dict)]
    full, partial, free_p, free_g = ev._pair_rows(sc, pr, gr)
    n_pages = max(1, len(gold.get("pages") or [None]))
    gpage, levels, gmatch = gold_row_pages(gold, index)
    ppage = predicted_row_pages(trace, len(pr))
    grank, gcount = ranks(gpage)
    prank, pcount = ranks(ppage)
    links, left_g, left_p = link_unpaired(gr, pr, free_g, free_p, gpage, ppage, sc.same)
    paired_pred_to_gold = {pi: gi for pi, gi in full + partial}
    linked_pred = {l.pi: l.gi for l in links} | paired_pred_to_gold  # noqa: E741
    linked_gold = {l.gi: l.pi for l in links} | {gi: pi for pi, gi in full + partial}  # noqa: E741
    pred_on_page = Counter(p for p in ppage if p is not None)
    gold_on_page = Counter(p for p in gpage if p is not None)
    base = {
        "doc": doc,
        "group": ctx["group"],
        "scanned": ctx["scanned"],
        "repeated": ctx["repeated"],
        "n_pages": n_pages,
        "n_gold_rows": len(gr),
    }

    def pos(page: int | None) -> str:
        return "unknown" if page is None else layout.page_position(page, n_pages)

    def gfields(gi: int) -> dict[str, Any]:
        n = gcount.get(gpage[gi], 0) if gpage[gi] is not None else 0
        return {
            "gi": gi,
            "g_page": gpage[gi],
            "g_pos": pos(gpage[gi]),
            "g_near": near_boundary(grank[gi], n),
            "gold_spn_level": levels[gi],
            "gold_match": gmatch[gi],
        }

    def pfields(pi: int) -> dict[str, Any]:
        n = pcount.get(ppage[pi], 0) if ppage[pi] is not None else 0
        return {
            "pi": pi,
            "p_page": ppage[pi],
            "p_pos": pos(ppage[pi]),
            "p_near": near_boundary(prank[pi], n),
        }

    units: list[Unit] = []
    for lk in links:
        cause = classify_pair(gr[lk.gi], pr[lk.pi], lk.kind, sc.same)
        sf, cr = ident_evidence(gr[lk.gi], pr[lk.pi], sc.same)
        extra = {
            "same_ident": len(sf),
            "cross": cr,
            "qty_same": bool(sc.same(QTY, pr[lk.pi].get(QTY), gr[lk.gi].get(QTY))),
        }
        if cause in ("spn_misread", "spn_other", "spn_copies_other_slot", "weak_link_qty"):
            extra |= spn_evidence(pr[lk.pi].get(SPN), gr[lk.gi].get(SPN), index, sc.same)
        units.append(
            Unit(
                **base,
                side="pair",
                cause=cause,
                link_kind=lk.kind,
                extra=extra,
                **gfields(lk.gi),
                **pfields(lk.pi),
            )
        )
    for gi in left_g:
        cause = classify_gold_only(
            gi, gr, gpage, pred_on_page, gold_on_page, linked_pred, pr, truncated, sc.same
        )
        units.append(Unit(**base, side="gold", cause=cause, **gfields(gi)))
    for pi in left_p:
        cause = classify_pred_only(pi, pr, ppage, gr, linked_gold, sc.same)
        units.append(Unit(**base, side="pred", cause=cause, **pfields(pi)))
    return units


# --------------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------------


def scan_name(scanned: bool) -> str:
    """``scanned`` / ``digital``."""
    return "scanned" if scanned else "digital"


def cause_counts(units: Sequence[Unit]) -> dict[str, dict[str, int]]:
    """Per cause: units, gold rows, predicted rows, docs, and units by scanned / digital."""
    out: dict[str, dict[str, int]] = {}
    docs: dict[str, set[str]] = defaultdict(set)
    for u in units:
        d = out.setdefault(
            u.cause, {"units": 0, "gold": 0, "pred": 0, "scanned": 0, "digital": 0, "docs": 0}
        )
        d["units"] += 1
        d["gold"] += u.side in ("pair", "gold")
        d["pred"] += u.side in ("pair", "pred")
        d[scan_name(u.scanned)] += 1
        docs[u.cause].add(u.doc)
    for c, ds in docs.items():
        out[c]["docs"] = len(ds)
    return out


def cause_by_position(units: Sequence[Unit], side: str) -> dict[tuple[str, str, str], int]:
    """(cause, scanned/digital, page position) -> rows on the gold side (`gold`) or pred side."""
    out: Counter[tuple[str, str, str]] = Counter()
    for u in units:
        if side == "gold" and u.side in ("pair", "gold"):
            out[(u.cause, scan_name(u.scanned), u.g_pos)] += 1
        if side == "pred" and u.side in ("pair", "pred"):
            out[(u.cause, scan_name(u.scanned), u.p_pos)] += 1
    return dict(out)


def breakdown(
    units: Sequence[Unit],
    docs: Sequence[Mapping[str, Any]],
    side: str,
    key: Callable[[Mapping[str, Any]], str],
) -> dict[str, tuple[int, int]]:
    """Bucket -> (unpaired rows, all rows) for one side; `docs` are per-row records with a key.

    `docs` here are the per-ROW records of ``row_records`` (one per gold or predicted row).
    """
    num: Counter[str] = Counter()
    den: Counter[str] = Counter()
    for r in docs:
        if r["side"] == side:
            den[key(r)] += 1
    for u in units:
        if side == "gold" and u.side in ("pair", "gold"):
            num[key(unit_row_record(u, "gold"))] += 1
        if side == "pred" and u.side in ("pair", "pred"):
            num[key(unit_row_record(u, "pred"))] += 1
    return {k: (num.get(k, 0), den[k]) for k in sorted(den)}


def unit_row_record(u: Unit, side: str) -> dict[str, Any]:
    """The row-record view of one unit's gold or predicted row (same keys as ``row_records``)."""
    return {
        "side": side,
        "scanned": u.scanned,
        "repeated": u.repeated,
        "n_pages": u.n_pages,
        "n_gold_rows": u.n_gold_rows,
        "pos": u.g_pos if side == "gold" else u.p_pos,
        "near": u.g_near if side == "gold" else u.p_near,
        "group": u.group,
    }


def row_records(
    pred: Mapping[str, Any],
    gold: Mapping[str, Any],
    traces: Mapping[str, Any],
    indexes: Mapping[str, locate.DocIndex],
    ctxs: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """One record per gold row and per predicted row of the scored invoice docs (denominators)."""
    out: list[dict[str, Any]] = []
    for d, g in gold.items():
        p = pred.get(d)
        if g["doc_type"] != "invoice" or not isinstance(p, dict) or p.get("doc_type") != "invoice":
            continue
        n_pages = max(1, len(g.get("pages") or [None]))
        gr = list(g.get("line_items") or [])
        pr = [x for x in (p.get("line_items") or []) if isinstance(x, dict)]
        gpage, _, _ = gold_row_pages(g, indexes[d])
        ppage = predicted_row_pages(traces[d], len(pr))
        for side, pages in (("gold", gpage), ("pred", ppage)):
            rk, cnt = ranks(pages)
            for i, pg in enumerate(pages):
                out.append(
                    {
                        "side": side,
                        "scanned": ctxs[d]["scanned"],
                        "repeated": ctxs[d]["repeated"],
                        "n_pages": n_pages,
                        "n_gold_rows": len(gr),
                        "pos": "unknown" if pg is None else layout.page_position(pg, n_pages),
                        "near": None if pg is None else near_boundary(rk[i], cnt[pg]),
                        "group": ctxs[d]["group"],
                    }
                )
    return out


def page_records(
    units: Sequence[Unit],
    gold: Mapping[str, Any],
    pred: Mapping[str, Any],
    indexes: Mapping[str, locate.DocIndex],
    ctxs: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """One record per page that holds gold rows: layout method, gold rows, unpaired, shifts.

    Layout method (``layout.analyze_page``, OCR only): ``table_header`` = the page has a
    column-header row, ``continuation`` = a "continued" banner and no header row, ``y_fraction`` =
    neither. `shift` counts the column_shift pairs of the page.
    """
    bad: Counter[tuple[str, int]] = Counter()
    shift: Counter[tuple[str, int]] = Counter()
    for u in units:
        if u.side in ("pair", "gold") and u.g_page is not None:
            bad[(u.doc, u.g_page)] += 1
            shift[(u.doc, u.g_page)] += u.cause == "column_shift"
    out: list[dict[str, Any]] = []
    for d, g in gold.items():
        p = pred.get(d)
        if g["doc_type"] != "invoice" or not isinstance(p, dict) or p.get("doc_type") != "invoice":
            continue
        gpage, _, _ = gold_row_pages(g, indexes[d])
        per_page = Counter(pg for pg in gpage if pg is not None)
        for pg, n in sorted(per_page.items()):
            out.append(
                {
                    "doc": d,
                    "page": pg,
                    "group": ctxs[d]["group"],
                    "scanned": ctxs[d]["scanned"],
                    "method": layout.analyze_page(indexes[d].pages[pg]).method,
                    "gold_rows": n,
                    "unpaired": bad.get((d, pg), 0),
                    "shift": shift.get((d, pg), 0),
                }
            )
    return out


def page_method_summary(recs: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """Per layout method: pages, gold rows, pages / rows with unpaired rows, whole-page failures."""
    rows: dict[str, dict[str, int]] = {}
    for r in recs:
        m = rows.setdefault(
            r["method"],
            {
                "pages": 0,
                "gold_rows": 0,
                "pages_with_unpaired": 0,
                "unpaired_rows": 0,
                "whole_page_bad": 0,
                "whole_page_bad_rows": 0,
                "shift_rows": 0,
            },
        )
        m["pages"] += 1
        m["gold_rows"] += r["gold_rows"]
        if r["unpaired"]:
            m["pages_with_unpaired"] += 1
            m["unpaired_rows"] += r["unpaired"]
            m["shift_rows"] += r["shift"]
            if r["unpaired"] == r["gold_rows"]:
                m["whole_page_bad"] += 1
                m["whole_page_bad_rows"] += r["unpaired"]
    return rows


def group_method_summary(
    recs: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, tuple[int, int, int]]]:
    """Group -> layout method -> (gold rows, unpaired rows, column_shift rows), groups that have
    both header-less (``continuation``) and ``table_header`` pages on a page where they have gold
    rows only. Controls the continuation-page comparison for the supplier group."""
    agg: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    for r in recs:
        a = agg[r["group"]][r["method"]]
        a[0] += r["gold_rows"]
        a[1] += r["unpaired"]
        a[2] += r["shift"]
    return {
        g: {m: (v[0], v[1], v[2]) for m, v in ms.items()}
        for g, ms in sorted(agg.items())
        if "continuation" in ms and "table_header" in ms
    }


def doc_concentration(units: Sequence[Unit]) -> dict[str, Any]:
    """How concentrated the unpaired gold rows are in a few documents."""
    per_doc: Counter[str] = Counter(u.doc for u in units if u.side in ("pair", "gold"))
    total = sum(per_doc.values())
    ordered = sorted(per_doc.values(), reverse=True)
    top5 = sum(ordered[:5])
    return {
        "docs_with_unpaired": len(per_doc),
        "unpaired_gold": total,
        "top5_docs_rows": top5,
        "top5_share": top5 / total if total else 0.0,
    }


# --------------------------------------------------------------------------------------------
# cpn / po slot analysis (X2)
# --------------------------------------------------------------------------------------------


def group_columns(labels: Sequence[Mapping[str, Any]], groups: Mapping[str, str]) -> dict[str, Any]:
    """Per supplier group: rows, share of gold rows with a non-null cpn / po (train+dev pooled).

    A share of 0 means the layout does not print the column (recon: group-wide constants); a share
    strictly between 0 and 1 would be a mixed group.
    """
    out: dict[str, dict[str, int]] = {}
    for g in labels:
        if g["doc_type"] != "invoice":
            continue
        d = out.setdefault(groups[g["doc_id"]], {"docs": 0, "rows": 0, "cpn": 0, "po": 0})
        d["docs"] += 1
        for r in g.get("line_items") or []:
            d["rows"] += 1
            d["cpn"] += not dg.empty(r.get(CPN))
            d["po"] += not dg.empty(r.get(PO))
    return out


def column_state(n_nonnull: int, rows: int) -> str:
    """``printed`` / ``not printed`` / ``mixed`` from a non-null count."""
    if rows == 0:
        return "n/a"
    if n_nonnull == 0:
        return "not printed"
    return "printed" if n_nonnull == rows else "mixed"


def header_line_labels(pages: Sequence[Any]) -> dict[str, bool]:
    """Whether the table-header line of any page names a customer-part / PO column (OCR text)."""
    text = ""
    for page in pages:
        lay = layout.analyze_page(page)
        if lay.header_line is None:
            continue
        text += " ".join(it.text for it in page_items(page) if it.line_idx == lay.header_line)
        text += "\n"
    return {
        "has_header_line": bool(text.strip()),
        "cpn": bool(CPN_LABEL.search(text)),
        "po": bool(PO_LABEL.search(text)),
    }


def paired_rows_by_group(
    pred: Mapping[str, Any], gold: Mapping[str, Any], groups: Mapping[str, str], sc: Any
) -> dict[str, int]:
    """Scorer-paired (full + partial) invoice rows per supplier group (denominators for X2)."""
    out: Counter[str] = Counter()
    for d, g in gold.items():
        p = pred.get(d)
        if g["doc_type"] != "invoice" or not isinstance(p, dict) or p.get("doc_type") != "invoice":
            continue
        pr = [x for x in (p.get("line_items") or []) if isinstance(x, dict)]
        full, partial, _, _ = ev._pair_rows(sc, pr, list(g.get("line_items") or []))
        out[groups[d]] += len(full) + len(partial)
    return dict(out)


def slot_rows(pred: Mapping[str, Any], gold: Mapping[str, Any], sc: Any) -> list[dict[str, Any]]:
    """Scorer-paired invoice rows with a cpn / po null-status mismatch, classified.

    ``kind``: ``po_in_cpn_slot`` / ``cpn_in_po_slot`` (as ``respike.slot_misplacements``), else
    ``cpn_false_fill`` / ``cpn_over_null`` / ``po_false_fill`` / ``po_over_null`` / ``mixed``.
    """
    out: list[dict[str, Any]] = []
    for d, g in gold.items():
        p = pred.get(d)
        if g["doc_type"] != "invoice" or not isinstance(p, dict) or p.get("doc_type") != "invoice":
            continue
        pr = [x for x in (p.get("line_items") or []) if isinstance(x, dict)]
        gr = list(g.get("line_items") or [])
        full, partial, _, _ = ev._pair_rows(sc, pr, gr)
        for pi, gi in full + partial:
            r, q = pr[pi], gr[gi]
            bad = [f for f in (CPN, PO) if dg.empty(r.get(f)) != dg.empty(q.get(f))]
            if not bad:
                continue
            if (
                dg.empty(q.get(CPN))
                and not dg.empty(q.get(PO))
                and not dg.empty(r.get(CPN))
                and dg.empty(r.get(PO))
                and sc.same(PO, r.get(CPN), q.get(PO))
            ):
                kind = "po_in_cpn_slot"
            elif (
                dg.empty(q.get(PO))
                and not dg.empty(q.get(CPN))
                and not dg.empty(r.get(PO))
                and dg.empty(r.get(CPN))
                and sc.same(CPN, r.get(PO), q.get(CPN))
            ):
                kind = "cpn_in_po_slot"
            elif len(bad) == 1:
                f = "cpn" if bad[0] == CPN else "po"
                kind = f"{f}_over_null" if dg.empty(r.get(bad[0])) else f"{f}_false_fill"
            else:
                kind = "mixed"
            out.append({"doc": d, "pi": pi, "gi": gi, "kind": kind})
    return out


# --------------------------------------------------------------------------------------------
# Waybill header over-nulls (X3)
# --------------------------------------------------------------------------------------------

WB_FIELDS = ("mawb", "carrier", "hawb")


def raw_header_state(page: Mapping[str, Any], key: str) -> tuple[str, dict[str, Any] | None]:
    """(``null_emitted`` / ``key_missing`` / ``value_emitted`` / ``unparsed``, raw header dict)."""
    raw = page.get("raw_text") or ""
    obj: Any = None
    try:
        obj = json.loads(raw)
    except ValueError:
        obj = salvage_page_json(raw)
    hdr = obj.get("header") if isinstance(obj, dict) else None
    if not isinstance(hdr, dict):
        return "unparsed", None
    if key not in hdr:
        return "key_missing", hdr
    return ("null_emitted" if dg.empty(hdr[key]) else "value_emitted"), hdr


def waybill_over_nulls(
    pred: Mapping[str, Any],
    gold: Mapping[str, Any],
    traces: Mapping[str, Any],
    indexes: Mapping[str, locate.DocIndex],
    scanned: Mapping[str, bool],
    sc: Any,
) -> dict[str, Any]:
    """Facts about every waybill header cell with a gold value and an empty prediction.

    Everything here is independent of the page image: OCR presence of the gold value (locator
    levels exact / normalized), the field's label in the OCR text, a label on the same OCR line as
    the value, the model's raw output state (null emitted vs key missing), other raw header slots
    holding the value, and whether a pattern backfill from OCR would be unambiguous. A control
    row gives the label presence over ALL waybill docs.
    """
    cells: list[dict[str, Any]] = []
    control: dict[str, Counter[str]] = {f: Counter() for f in WB_FIELDS}
    for d, g in gold.items():
        if g["doc_type"] != "waybill":
            continue
        idx = indexes[d]
        text = "\n".join(page_text(p) for p in idx.pages)
        p = pred.get(d) or {}
        ph = p.get("header") if isinstance(p.get("header"), dict) else {}
        for f in WB_FIELDS:
            gv = g["header"].get(f)
            if dg.empty(gv):
                continue
            label_any = bool(WB_LABEL[f].search(text))
            control[f]["docs"] += 1
            control[f]["label_in_ocr"] += label_any
            if not dg.empty(ph.get(f)):
                control[f]["predicted"] += 1
                control[f]["predicted_label_in_ocr"] += label_any
                continue
            matches = locate.find_matches(gv, f, idx, max_level="normalized")
            # the label sits on the value's OCR line or on the line right above it (stacked
            # label / value cells in the waybill layout)
            on_label_line = any(
                WB_LABEL[f].search(idx.line_text(m.page, li))
                for m in matches
                for li in (m.line_idx, m.line_idx - 1)
                if li >= 0
            )
            page = traces[d]["pages"][0]
            state, hdr = raw_header_state(page, f)
            alt = sorted(
                k
                for k, v in (hdr or {}).items()
                if k != f and not dg.empty(v) and sc.same(f, v, gv)
            )
            pattern = WB_PATTERN.get(f)
            uniq = sorted(set(pattern.findall(text))) if pattern else []
            cells.append(
                {
                    "doc": d,
                    "field": f,
                    "scanned": bool(scanned[d]),
                    "ocr_exact": any(m.level == "exact" for m in matches),
                    "ocr_normalized": bool(matches),
                    "label_in_ocr": label_any,
                    "label_adjacent": on_label_line,
                    "raw_state": state,
                    "alt_slots": alt,
                    "pattern_applicable": pattern is not None,
                    "pattern_unique": len(uniq) == 1 if pattern else None,
                    "pattern_candidates": len(uniq),
                    "pattern_correct": (any(sc.same(f, u, gv) for u in uniq) if pattern else None),
                    "pattern_unique_correct": (
                        len(uniq) == 1 and bool(sc.same(f, uniq[0], gv)) if pattern else None
                    ),
                }
            )
    return {"cells": cells, "control": {f: dict(c) for f, c in control.items()}}


# --------------------------------------------------------------------------------------------
# ORACLE replay
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class OracleSpec:
    """What an ORACLE variant substitutes with gold."""

    causes: frozenset[str] = frozenset()
    slot_kinds: frozenset[str] = frozenset()
    header_fields: frozenset[str] = frozenset()


def apply_oracle(
    pred: Mapping[str, Any],
    gold: Mapping[str, Any],
    units: Sequence[Unit],
    slots: Sequence[Mapping[str, Any]],
    spec: OracleSpec,
) -> dict[str, Any]:
    """Patched deep copy of `pred`: gold substituted for the rows / cells named by `spec`.

    Pairs of a listed cause: the predicted row becomes the gold row. Gold-only rows of a listed
    cause are appended as gold rows. Predicted-only rows of a listed cause are removed. Slot rows
    of a listed kind get gold cpn / po. Header cells (waybill fields) with a gold value and an
    empty prediction get the gold value. An upper bound, never a fix.
    """
    out = copy.deepcopy(dict(pred))
    drop: dict[str, set[int]] = defaultdict(set)
    add: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for u in units:
        if u.cause not in spec.causes:
            continue
        p = out.get(u.doc)
        if not isinstance(p, dict):
            continue
        g = gold[u.doc]["line_items"]
        if u.side == "pair":
            rows = [x for x in p["line_items"] if isinstance(x, dict)]
            assert u.pi is not None and u.gi is not None
            rows[u.pi].clear()
            rows[u.pi].update({k: g[u.gi].get(k) for k in ROW_KEYS})
            p["line_items"] = rows
        elif u.side == "gold":
            assert u.gi is not None
            add[u.doc].append({k: g[u.gi].get(k) for k in ROW_KEYS})
        else:
            assert u.pi is not None
            drop[u.doc].add(u.pi)
    for s in slots:
        if s["kind"] not in spec.slot_kinds:
            continue
        p = out[s["doc"]]
        rows = [x for x in p["line_items"] if isinstance(x, dict)]
        q = gold[s["doc"]]["line_items"][s["gi"]]
        rows[s["pi"]][CPN] = q.get(CPN)
        rows[s["pi"]][PO] = q.get(PO)
        p["line_items"] = rows
    for d, p in out.items():
        if not isinstance(p, dict):
            continue
        if d in drop:
            rows = [x for x in p["line_items"] if isinstance(x, dict)]
            p["line_items"] = [r for i, r in enumerate(rows) if i not in drop[d]]
        if d in add:
            p["line_items"] = list(p["line_items"]) + add[d]
        if spec.header_fields and gold[d]["doc_type"] == "waybill" and isinstance(p, dict):
            hdr = p.setdefault("header", {})
            for f in spec.header_fields:
                gv = gold[d]["header"].get(f)
                if not dg.empty(gv) and dg.empty(hdr.get(f)):
                    hdr[f] = gv
    return out


def oracle_row(
    name: str,
    pred: Mapping[str, Any],
    gold: Mapping[str, Any],
    units: Sequence[Unit],
    slots: Sequence[Mapping[str, Any]],
    spec: OracleSpec,
    base: Mapping[str, Any],
    n: int = N_BOOT,
    seed: int = SEED,
) -> dict[str, Any]:
    """Rescore a patched prediction with the unmodified scorer; paired bootstrap of OVERALL."""
    patched = apply_oracle(pred, gold, units, slots, spec)
    agg = ev.score(patched, dict(gold))["all"]
    pb = ev.paired_bootstrap(dict(pred), patched, dict(gold), n=n, seed=seed)
    return {
        "name": name,
        "OVERALL": agg["OVERALL"],
        "d_OVERALL": agg["OVERALL"] - base["OVERALL"],
        "ci": [pb["OVERALL"]["lo"], pb["OVERALL"]["hi"]],
        "d_header_acc": agg["header_field_accuracy"] - base["header_field_accuracy"],
        "d_row_f1": agg["row_f1"] - base["row_f1"],
        "d_fully_correct": agg["documents_fully_correct"] - base["documents_fully_correct"],
    }


# --------------------------------------------------------------------------------------------
# Whole-run analysis
# --------------------------------------------------------------------------------------------


def analyse(
    run_dir: Path,
    gold_dir: Path,
    meta_path: Path,
    ocr_cache: Path | None = None,
    n_boot: int = N_BOOT,
    with_oracle: bool = True,
) -> dict[str, Any]:
    """Run the full diagnosis of one run dir; returns a plain-dict result (no rendering)."""
    sc = ev.load_scorer()
    pred = json.loads((run_dir / "predictions.json").read_text(encoding="utf-8"))
    gold_all = ev.load_gold(gold_dir)
    gold = {d: gold_all[d] for d in sorted(pred)}
    traces = {t["doc_id"]: t for t in read_trace(run_dir / "trace.jsonl")}
    meta = {m["doc_id"]: m for m in json.loads(meta_path.read_text(encoding="utf-8"))}
    groups = json.loads((ROOT / "meta" / "supplier_groups.json").read_text(encoding="utf-8"))
    ocr = ev.load_ocr_pages(list(gold), ocr_cache)
    missing = [d for d in gold if d not in ocr]
    if missing:
        raise FileNotFoundError(f"no cached OCR for {len(missing)} docs, e.g. {missing[:3]}")
    indexes = {d: locate.build_index(ocr[d]) for d in gold}
    ctxs = {
        d: {
            "group": groups[d],
            "scanned": bool(meta[d]["scanned"]),
            "repeated": bool(meta[d]["repeated_parts"]),
        }
        for d in gold
    }
    truncated = set(dg.truncated_docs(run_dir))
    units: list[Unit] = []
    for d, g in gold.items():
        p = pred.get(d)
        if g["doc_type"] != "invoice" or not isinstance(p, dict) or p.get("doc_type") != "invoice":
            continue
        units += diagnose_doc(d, p, g, traces[d], indexes[d], ctxs[d], sc, d in truncated)
    rows = row_records(pred, gold, traces, indexes, ctxs)
    tax = ev.error_taxonomy(dict(pred), dict(gold), ocr, list(meta.values()))
    step_l = _step_l_cross(units, pred, gold, sc, truncated)
    base = ev.score(dict(pred), dict(gold))["all"]
    slots = slot_rows(pred, gold, sc)
    labels = meta_mod.load_labels("train") + meta_mod.load_labels("dev")
    gcols = group_columns(labels, groups)
    hdr_labels = {
        d: header_line_labels(indexes[d].pages) for d in gold if gold[d]["doc_type"] == "invoice"
    }
    wb = waybill_over_nulls(pred, gold, traces, indexes, {d: ctxs[d]["scanned"] for d in gold}, sc)
    recs = page_records(units, gold, pred, indexes, ctxs)
    method_of = {(r["doc"], r["page"]): r["method"] for r in recs}
    for u in units:
        if u.g_page is not None:
            u.extra["page_method"] = method_of.get((u.doc, u.g_page), "?")
    wb_ocr = ev.load_ocr_pages(
        [g["doc_id"] for g in labels if g["doc_type"] == "waybill"], ocr_cache
    )
    shapes = learn_slot_shapes(meta_mod.load_labels("train"))
    res: dict[str, Any] = {
        "n_docs": len(gold),
        "n_invoice_docs": sum(g["doc_type"] == "invoice" for g in gold.values()),
        "gold_rows": sum(len(g.get("line_items") or []) for g in gold.values()),
        "pred_rows": sum(len(p.get("line_items") or []) for p in pred.values() if p),
        "taxonomy": {k: v for k, v in tax["counts"]["all"].items() if k.startswith("row_")},
        "units": units,
        "n_unpaired_gold": sum(u.side in ("pair", "gold") for u in units),
        "n_unpaired_pred": sum(u.side in ("pair", "pred") for u in units),
        "rows": rows,
        "step_l": step_l,
        "slots": slots,
        "group_cols": gcols,
        "hdr_labels": hdr_labels,
        "paired_by_group": paired_rows_by_group(pred, gold, groups, sc),
        "slot_docs_scan": {d: ctxs[d]["scanned"] for d in gold},
        "slot_docs_group": {d: ctxs[d]["group"] for d in gold},
        "page_method": page_method_summary(recs),
        "group_method": group_method_summary(recs),
        "backfill_check": pattern_backfill_check(labels, wb_ocr, sc.same),
        "shapes": {k: v for k, v in shapes.items() if k.startswith("n_")}
        | {k: len(v) for k, v in shapes.items() if k in ("cpn_only", "po_only")},
        "concentration": doc_concentration(units),
        "waybill": wb,
        "base": {k: v for k, v in base.items() if not isinstance(v, dict)},
        "truncated_docs": sorted(truncated),
        "n_boot": n_boot,
    }
    res["oracle"] = oracle_table(pred, gold, units, slots, wb, base, n_boot) if with_oracle else []
    res["rules"] = (
        candidate_rules(pred, gold, traces, indexes, shapes, sc, base, n_boot)
        if with_oracle
        else []
    )
    res["_pred"], res["_gold"] = pred, gold
    return res


def _step_l_cross(
    units: Sequence[Unit],
    pred: Mapping[str, Any],
    gold: Mapping[str, Any],
    sc: Any,
    truncated: set[str],
) -> dict[str, Any]:
    """Cross-tab of this script's cause (gold side) vs the Step L cause of the same gold row."""
    by_doc: dict[str, dict[int, str]] = {}
    for d, g in gold.items():
        p = pred.get(d)
        if g["doc_type"] != "invoice" or not isinstance(p, dict) or p.get("doc_type") != "invoice":
            continue
        gr = list(g.get("line_items") or [])
        pr = [x for x in (p.get("line_items") or []) if isinstance(x, dict)]
        _, _, unp, ung = ev._pair_rows(sc, pr, gr)
        att, _ = dg.attribute_unpaired(sc, pr, gr, unp, ung, d in truncated)
        by_doc[d] = {a["gi"]: a["cause"] for a in att}
    cross: Counter[tuple[str, str]] = Counter()
    for u in units:
        if u.side in ("pair", "gold") and u.gi is not None:
            cross[(u.cause, by_doc[u.doc][u.gi])] += 1
    totals = Counter(c for m in by_doc.values() for c in m.values())
    return {"cross": dict(cross), "totals": dict(sorted(totals.items()))}


def oracle_table(
    pred: Mapping[str, Any],
    gold: Mapping[str, Any],
    units: Sequence[Unit],
    slots: Sequence[Mapping[str, Any]],
    wb: Mapping[str, Any],
    base: Mapping[str, Any],
    n_boot: int,
) -> list[dict[str, Any]]:
    """ORACLE rows: every cause alone, slot kinds, waybill header fields, and combinations."""
    present = [c for c in ALL_CAUSES if any(u.cause == c for u in units)]
    rows = []
    for c in present:
        rows.append(("cause:" + c, OracleSpec(causes=frozenset({c}))))
    kinds = sorted({s["kind"] for s in slots})
    for k in kinds:
        rows.append(("slot:" + k, OracleSpec(slot_kinds=frozenset({k}))))
    rows.append(("slot:all_null_status_mismatch", OracleSpec(slot_kinds=frozenset(kinds))))
    for f in WB_FIELDS:
        if any(c["field"] == f for c in wb["cells"]):
            rows.append(("header:" + f, OracleSpec(header_fields=frozenset({f}))))
    rows.append(("header:all_waybill_over_nulls", OracleSpec(header_fields=frozenset(WB_FIELDS))))
    rows.append(("rows:all_unpaired_causes", OracleSpec(causes=frozenset(present))))
    rows.append(
        (
            "rows:unpaired + slot mismatches",
            OracleSpec(causes=frozenset(present), slot_kinds=frozenset(kinds)),
        )
    )
    rows.append(
        (
            "COMBINED: rows + slots + waybill over-nulls",
            OracleSpec(frozenset(present), frozenset(kinds), frozenset(WB_FIELDS)),
        )
    )
    return [oracle_row(name, pred, gold, units, slots, spec, base, n=n_boot) for name, spec in rows]


# --------------------------------------------------------------------------------------------
# Part-number evidence, candidate merge-rule replays
# --------------------------------------------------------------------------------------------


def edit_profile(pred: Any, gold: Any) -> str:
    """How a predicted part number differs from the gold one (alphanumerics, lowercase).

    ``punctuation_only`` (equal once separators are dropped), ``glyph_confusion`` (only
    substitutions between visually confusable characters, ``CONFUSABLE``), ``char_dropped_or_added``
    (only insertions / deletions), ``mixed`` (substitution plus insertion / deletion),
    ``other_substitution``.
    """
    a, b = dg.alnum(pred), dg.alnum(gold)
    if a == b:
        return "punctuation_only"
    ops = [op for op in Levenshtein.opcodes(a, b) if op.tag != "equal"]
    tags = {op.tag for op in ops}
    if tags == {"replace"}:
        pairs = [
            (x, y)
            for op in ops
            for x, y in zip(
                a[op.src_start : op.src_end], b[op.dest_start : op.dest_end], strict=False
            )
        ]
        same_len = all(op.src_end - op.src_start == op.dest_end - op.dest_start for op in ops)
        if same_len and all(frozenset((x, y)) in CONFUSABLE for x, y in pairs):
            return "glyph_confusion"
        return "other_substitution"
    return "char_dropped_or_added" if "replace" not in tags else "mixed"


def snap_spn(value: Any, index: locate.DocIndex) -> str | None:
    """Text of the nearest OCR span for a part number the OCR does not contain verbatim.

    None when `value` is empty, already found in the OCR at exact / normalized level, or has no
    fuzzy match at the locator's threshold. The candidate "OCR snap" merge rule.
    """
    if dg.empty(value):
        return None
    if locate.find_matches(value, SPN, index, max_level="normalized"):
        return None
    ms = locate.find_matches(value, SPN, index)
    return ms[0].text if ms else None


def spn_evidence(
    pred_spn: Any, gold_spn: Any, index: locate.DocIndex, same: Same
) -> dict[str, Any]:
    """OCR-side facts about one wrong part number (page-image independent)."""
    out: dict[str, Any] = {}
    both = not dg.empty(pred_spn) and not dg.empty(gold_spn)
    out["edit"] = edit_profile(pred_spn, gold_spn) if both else "n/a"
    gm = locate.find_matches(gold_spn, SPN, index) if not dg.empty(gold_spn) else []
    out["gold_in_ocr"] = gm[0].level if gm else "none"
    pred_in = (
        bool(locate.find_matches(pred_spn, SPN, index, max_level="normalized"))
        if not dg.empty(pred_spn)
        else False
    )
    out["pred_in_ocr"] = pred_in
    if pred_in:
        out["snap"] = "pred_verbatim_in_ocr"
    else:
        snapped = snap_spn(pred_spn, index)
        if snapped is None:
            out["snap"] = "no_candidate"
        else:
            out["snap"] = "correct" if same(SPN, snapped, gold_spn) else "wrong"
    return out


def shape_of(value: Any) -> str:
    """Format shape of an identifier: digits -> 9, letters -> A, punctuation kept."""
    return re.sub(r"[A-Za-z]", "A", re.sub(r"\d", "9", str(value)))


def learn_slot_shapes(labels: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Format shapes seen in the cpn and PO columns of `labels` (pass TRAIN gold only)."""
    cpn: set[str] = set()
    po: set[str] = set()
    for g in labels:
        if g["doc_type"] != "invoice":
            continue
        for r in g.get("line_items") or []:
            if not dg.empty(r.get(CPN)):
                cpn.add(shape_of(r[CPN]))
            if not dg.empty(r.get(PO)):
                po.add(shape_of(r[PO]))
    return {
        "cpn_only": sorted(cpn - po),
        "po_only": sorted(po - cpn),
        "n_cpn_shapes": len(cpn),
        "n_po_shapes": len(po),
        "n_overlap": len(cpn & po),
    }


def _doc_text(index: locate.DocIndex) -> str:
    return "\n".join(page_text(p) for p in index.pages)


def pattern_backfill_check(
    labels: Sequence[Mapping[str, Any]], ocr: Mapping[str, Any], same: Same
) -> dict[str, dict[str, int]]:
    """Gold-vs-OCR precision of the pattern backfill over waybills (no model involved).

    Per field: docs with a gold value split into unique pattern match right / wrong / several
    matches / none; docs with a gold null split into none / a unique match (a backfill would fill
    a value that should stay null) / several.
    """
    out: dict[str, Counter[str]] = {f: Counter() for f in WB_PATTERN}
    for g in labels:
        if g["doc_type"] != "waybill" or g["doc_id"] not in ocr:
            continue
        text = "\n".join(page_text(p) for p in ocr[g["doc_id"]])
        for f, pat in WB_PATTERN.items():
            found = sorted(set(pat.findall(text)))
            gv = g["header"].get(f)
            if dg.empty(gv):
                out[f]["gold_null_docs"] += 1
                out[f]["gold_null_unique_match"] += len(found) == 1
                out[f]["gold_null_several_matches"] += len(found) > 1
                continue
            out[f]["gold_docs"] += 1
            if len(found) == 1:
                out[f]["unique_right" if same(f, found[0], gv) else "unique_wrong"] += 1
            elif found:
                out[f]["several_matches"] += 1
            else:
                out[f]["no_match"] += 1
    return {f: dict(c) for f, c in out.items()}


def rule_carrier_from_supplier(
    pred: Mapping[str, Any], gold: Mapping[str, Any], traces: Mapping[str, Any], same: Same
) -> tuple[dict[str, Any], list[dict[str, bool]]]:
    """Candidate rule R1: waybill with an empty carrier takes the model's own supplier_name slot."""
    out = copy.deepcopy(dict(pred))
    rec: list[dict[str, bool]] = []
    for d, g in gold.items():
        p = out.get(d)
        if g["doc_type"] != "waybill" or not isinstance(p, dict):
            continue
        hdr = p.setdefault("header", {})
        if not dg.empty(hdr.get("carrier")):
            continue
        parsed = traces[d]["pages"][0].get("parsed") or {}
        sup = (parsed.get("header") or {}).get("supplier_name")
        if dg.empty(sup):
            continue
        gv = g["header"].get("carrier")
        rec.append({"before_ok": dg.empty(gv), "after_ok": bool(same("carrier", sup, gv))})
        hdr["carrier"] = sup
    return out, rec


def rule_pattern_backfill(
    pred: Mapping[str, Any],
    gold: Mapping[str, Any],
    indexes: Mapping[str, locate.DocIndex],
    same: Same,
) -> tuple[dict[str, Any], list[dict[str, bool]]]:
    """Candidate rule R2: empty mawb / hawb takes the unique OCR pattern match of the doc."""
    out = copy.deepcopy(dict(pred))
    rec: list[dict[str, bool]] = []
    for d, g in gold.items():
        p = out.get(d)
        if g["doc_type"] != "waybill" or not isinstance(p, dict):
            continue
        hdr = p.setdefault("header", {})
        text = _doc_text(indexes[d])
        for f in ("mawb", "hawb"):
            if not dg.empty(hdr.get(f)):
                continue
            found = sorted(set(WB_PATTERN[f].findall(text)))
            if len(found) != 1:
                continue
            gv = g["header"].get(f)
            rec.append({"before_ok": dg.empty(gv), "after_ok": bool(same(f, found[0], gv))})
            hdr[f] = found[0]
    return out, rec


def _cpn_po_right(same: Same, row: Mapping[str, Any], gold_row: Mapping[str, Any] | None) -> bool:
    """True if both cpn and po of `row` match the paired gold row (False when unpaired)."""
    return gold_row is not None and all(same(f, row.get(f), gold_row.get(f)) for f in (CPN, PO))


def rule_slot_shape(
    pred: Mapping[str, Any],
    gold: Mapping[str, Any],
    shapes: Mapping[str, Any],
    sc: Any,
) -> tuple[dict[str, Any], list[dict[str, bool]]]:
    """Candidate rule R3: move a cpn / po value to the other slot when its format shape occurs
    only in that slot's column in the (train) gold, and the other slot is empty."""
    out = copy.deepcopy(dict(pred))
    rec: list[dict[str, bool]] = []
    cpn_only, po_only = set(shapes["cpn_only"]), set(shapes["po_only"])
    for d, g in gold.items():
        p = out.get(d)
        if g["doc_type"] != "invoice" or not isinstance(p, dict) or p.get("doc_type") != "invoice":
            continue
        rows = [x for x in (p.get("line_items") or []) if isinstance(x, dict)]
        gr = list(g.get("line_items") or [])
        full, partial, _, _ = ev._pair_rows(sc, rows, gr)
        partner = {pi: gi for pi, gi in full + partial}
        for pi, r in enumerate(rows):
            move = None
            if not dg.empty(r.get(CPN)) and dg.empty(r.get(PO)) and shape_of(r[CPN]) in po_only:
                move = (CPN, PO)
            elif not dg.empty(r.get(PO)) and dg.empty(r.get(CPN)) and shape_of(r[PO]) in cpn_only:
                move = (PO, CPN)
            if move is None:
                continue
            before = dict(r)
            r[move[1]], r[move[0]] = r[move[0]], None
            gi = partner.get(pi)
            gold_row = gr[gi] if gi is not None else None
            rec.append(
                {
                    "before_ok": _cpn_po_right(sc.same, before, gold_row),
                    "after_ok": _cpn_po_right(sc.same, r, gold_row),
                }
            )
    return out, rec


def rule_snap_spn(
    pred: Mapping[str, Any],
    gold: Mapping[str, Any],
    indexes: Mapping[str, locate.DocIndex],
    same: Same,
) -> tuple[dict[str, Any], list[dict[str, bool]]]:
    """Candidate rule R4: replace a predicted part number the OCR lacks by its nearest OCR span."""
    out = copy.deepcopy(dict(pred))
    rec: list[dict[str, bool]] = []
    for d, g in gold.items():
        p = out.get(d)
        if g["doc_type"] != "invoice" or not isinstance(p, dict) or p.get("doc_type") != "invoice":
            continue
        gold_spns = [r.get(SPN) for r in g.get("line_items") or []]
        for r in (x for x in (p.get("line_items") or []) if isinstance(x, dict)):
            snapped = snap_spn(r.get(SPN), indexes[d])
            if snapped is None:
                continue
            rec.append(
                {
                    "before_ok": any(same(SPN, r.get(SPN), v) for v in gold_spns),
                    "after_ok": any(same(SPN, snapped, v) for v in gold_spns),
                }
            )
            r[SPN] = snapped
    return out, rec


def replay_row(
    name: str,
    pred: Mapping[str, Any],
    patched: Mapping[str, Any],
    rec: Sequence[Mapping[str, bool]],
    gold: Mapping[str, Any],
    base: Mapping[str, Any],
    n: int,
) -> dict[str, Any]:
    """Rescore a rule-patched prediction (unmodified scorer) with a paired bootstrap of OVERALL."""
    agg = ev.score(dict(patched), dict(gold))["all"]
    pb = ev.paired_bootstrap(dict(pred), dict(patched), dict(gold), n=n, seed=SEED)
    return {
        "name": name,
        "touched": len(rec),
        "right_after": sum(r["after_ok"] for r in rec),
        "wrong_after": sum(not r["after_ok"] for r in rec),
        "broken": sum(r["before_ok"] and not r["after_ok"] for r in rec),
        "fixed": sum(not r["before_ok"] and r["after_ok"] for r in rec),
        "OVERALL": agg["OVERALL"],
        "d_OVERALL": agg["OVERALL"] - base["OVERALL"],
        "ci": [pb["OVERALL"]["lo"], pb["OVERALL"]["hi"]],
        "d_row_f1": agg["row_f1"] - base["row_f1"],
        "d_header_acc": agg["header_field_accuracy"] - base["header_field_accuracy"],
    }


def candidate_rules(
    pred: Mapping[str, Any],
    gold: Mapping[str, Any],
    traces: Mapping[str, Any],
    indexes: Mapping[str, locate.DocIndex],
    shapes: Mapping[str, Any],
    sc: Any,
    base: Mapping[str, Any],
    n: int,
) -> list[dict[str, Any]]:
    """Replay the four candidate merge rules (R1..R4) on the run, one at a time."""
    out = []
    for name, fn in (
        (
            "R1 waybill carrier <- own supplier_name slot",
            lambda: rule_carrier_from_supplier(pred, gold, traces, sc.same),
        ),
        (
            "R2 waybill mawb / hawb <- unique OCR pattern match",
            lambda: rule_pattern_backfill(pred, gold, indexes, sc.same),
        ),
        (
            "R3 cpn <-> po slot swap by format shape (shapes from train gold)",
            lambda: rule_slot_shape(pred, gold, shapes, sc),
        ),
        (
            "R4 part number <- nearest OCR span when absent from OCR",
            lambda: rule_snap_spn(pred, gold, indexes, sc.same),
        ),
    ):
        patched, rec = fn()
        out.append(replay_row(name, pred, patched, rec, gold, base, n))
    return out


# --------------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------------


def git_state() -> str:
    """Short HEAD sha, ``+dirty`` when the tracked tree differs."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=ROOT,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            cwd=ROOT,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return sha + ("+dirty" if dirty else "")


def render_md(
    res: Mapping[str, Any], run_name: str, repo_state: str, mapping: Mapping[str, Any]
) -> str:
    """The aggregates-only report (no values, no names)."""
    units: list[Unit] = res["units"]
    out: list[str] = []
    w = out.append
    w("# Row-level error diagnosis (executor X)\n")
    w(
        f"**Provenance.** Run `{run_name}` (Qwen3.5-4B image-only, keyed JSON), generated by "
        f"`uv run python scripts/row_error_diagnosis.py` (repo state `{repo_state}`; ORACLE "
        f"bootstrap {res['n_boot']} doc-level paired resamples, seed {SEED}). Every number is "
        "computed from the run's `predictions.json` / `trace.jsonl`, the gold labels, the OCR "
        "cache and the unmodified official scorer (`assignment/score.py` via `shipdoc.eval`). "
        "**All numbers UNVERIFIED** until a verifier recomputes them. Aggregates and counts "
        "only: no gold or predicted values, no names; value-level examples are in the local, "
        "outside-repo `<SHIPDOC_RUNS_DIR>/diagnosis/row_errors_local.md`.\n"
    )
    out += render_x1(res, units)
    out += render_x2(res)
    out += render_x3(res)
    out += render_x4(res, units, mapping)
    return "\n".join(out)


def _pos_table(units: Sequence[Unit], side: str, causes: Sequence[str]) -> str:
    cbp = cause_by_position(units, side)
    rows = []
    for c in causes:
        for sc_name in ("scanned", "digital"):
            vals = [cbp.get((c, sc_name, p), 0) for p in POSITIONS]
            if any(vals):
                rows.append([c, sc_name, *vals, sum(vals)])
    tot = [sum(v for (c, s, p), v in cbp.items() if p == pos) for pos in POSITIONS]
    rows.append(["**total**", "", *tot, sum(tot)])
    return md_table(["cause", "scan", *POSITIONS, "total"], rows)


def _bucket_table(
    res: Mapping[str, Any],
    units: Sequence[Unit],
    key: Callable[[Mapping[str, Any]], str],
    title: str,
    order: Sequence[str] | None = None,
) -> str:
    g = breakdown(units, res["rows"], "gold", key)
    p = breakdown(units, res["rows"], "pred", key)
    names = list(order) if order else sorted(set(g) | set(p))
    rows = []
    for k in names:
        gn, gd = g.get(k, (0, 0))
        pn, pd = p.get(k, (0, 0))
        rows.append([k, rate(gn, gd), rate(pn, pd)])
    return md_table(
        [title, "unpaired gold rows (row_missing)", "unpaired pred rows (row_extra)"], rows
    )


def render_x1(res: Mapping[str, Any], units: Sequence[Unit]) -> list[str]:
    """Section X1."""
    out: list[str] = []
    w = out.append
    tax = res["taxonomy"]
    match = res["n_unpaired_gold"] == tax.get("row_missing")
    match = match and res["n_unpaired_pred"] == tax.get("row_extra")
    w("## X1. row_missing / row_extra\n")
    w("### X1.0 Recount\n")
    w(
        f"`eval.error_taxonomy` (the respike definition) gives row_missing "
        f"{tax.get('row_missing', 0)}, row_extra {tax.get('row_extra', 0)}, row_split_merge "
        f"{tax.get('row_split_merge', 0)}. Recomputed here from `eval._pair_rows` (the scorer's "
        f"own two greedy passes): unpaired gold rows {res['n_unpaired_gold']}, unpaired "
        f"predicted rows {res['n_unpaired_pred']}, over {res['n_invoice_docs']} invoice docs "
        f"({res['gold_rows']} gold rows, {res['pred_rows']} predicted rows). "
        f"Match: {match}. Docs with a truncated / invalid page: {len(res['truncated_docs'])}.\n"
    )
    c = res["concentration"]
    w(
        f"Concentration: the unpaired gold rows sit in {c['docs_with_unpaired']} docs; the 5 docs "
        f"with the most account for {c['top5_docs_rows']} of {c['unpaired_gold']} "
        f"({100 * c['top5_share']:.1f}%).\n"
    )
    w("### X1.1 Breakdown (unpaired rows / all rows of the bucket)\n")
    w(
        "`page position`: `only` = single-page doc, `first` / `middle` / `last` = page of a "
        "multipage doc (`middle` = a continuation page that is not the last). A gold row's page is "
        "the locator's row-to-page assignment (`locate.assign_rows` + "
        "`trainset.resolve_row_pages`, read-only); a predicted row's page is the trace page it "
        "was parsed from (replay of `merge_pages`' row loop). `near boundary` = first / last "
        f"{BOUNDARY_K} rows of the page (gold order for gold rows, predicted order for predicted "
        "rows). Unpaired = left unpaired by the scorer's matching; a gold row and a predicted "
        "row of one wrongly read line are both counted, so the two columns are not independent.\n"
    )
    w(_bucket_table(res, units, lambda r: r["pos"], "page position", POSITIONS))
    w(_bucket_table(res, units, lambda r: scan_name(r["scanned"]), "scan", ("scanned", "digital")))
    w(
        _bucket_table(
            res,
            units,
            lambda r: (
                "near boundary" if r["near"] else ("interior" if r["near"] is False else "unknown")
            ),
            "page boundary",
            ("near boundary", "interior", "unknown"),
        )
    )
    w(
        _bucket_table(
            res,
            units,
            lambda r: length_bucket(r["n_gold_rows"]),
            "gold rows per doc",
            [b[0] for b in LENGTH_BUCKETS],
        )
    )
    w(
        _bucket_table(
            res,
            units,
            lambda r: "repeated_parts" if r["repeated"] else "no repeated_parts",
            "doc tag",
            ("repeated_parts", "no repeated_parts"),
        )
    )
    w(_bucket_table(res, units, lambda r: r["group"], "supplier group"))
    w("Scanned x page position (unpaired gold rows / gold rows):\n")
    sp: dict[tuple[str, str], list[int]] = {}
    for r in res["rows"]:
        if r["side"] == "gold":
            sp.setdefault((scan_name(r["scanned"]), r["pos"]), [0, 0])[1] += 1
    for u in units:
        if u.side in ("pair", "gold"):
            sp.setdefault((scan_name(u.scanned), u.g_pos), [0, 0])[0] += 1
    w(
        md_table(
            ["scan", *POSITIONS],
            [
                [s, *[rate(*sp.get((s, p), [0, 0])) for p in POSITIONS]]
                for s in ("scanned", "digital")
            ],
        )
    )
    w("### X1.2 Page layout (OCR only): does the page have a column-header row?\n")
    w(
        "Layout method per page (`layout.analyze_page`): `table_header` = the page has a "
        "column-header row; `continuation` = a 'continued' banner and no header row; "
        "`y_fraction` = neither found. `whole page` = every gold row of the page is unpaired.\n"
    )
    rows = []
    for m, r in sorted(res["page_method"].items()):
        rows.append(
            [
                m,
                r["pages"],
                r["gold_rows"],
                rate(r["pages_with_unpaired"], r["pages"]),
                rate(r["unpaired_rows"], r["gold_rows"]),
                r["whole_page_bad"],
                r["whole_page_bad_rows"],
                r["shift_rows"],
            ]
        )
    w(
        md_table(
            [
                "method",
                "pages",
                "gold rows",
                "pages with an unpaired gold row",
                "unpaired gold rows",
                "whole-page-unpaired pages",
                "rows on them",
                "column_shift rows",
            ],
            rows,
        )
    )
    w(
        "Controlled for the supplier group (groups that have BOTH header-less and header pages "
        "with gold rows; cell = unpaired gold rows / gold rows, column_shift rows in brackets):\n"
    )
    rows = []
    for g, ms in res["group_method"].items():
        a, b = ms["continuation"], ms["table_header"]
        rows.append([g, f"{a[1]}/{a[0]} [{a[2]}]", f"{b[1]}/{b[0]} [{b[2]}]"])
    w(md_table(["group", "continuation pages (no header row)", "table_header pages"], rows))
    w("### X1.3 Causes (mutually exclusive; decision order in the script docstring)\n")
    cc = cause_counts(units)
    rows = []
    for cause in ALL_CAUSES:
        d = cc.get(cause, {"units": 0, "gold": 0, "pred": 0, "scanned": 0, "digital": 0, "docs": 0})
        kind = (
            "pair"
            if cause in PAIR_CAUSES
            else ("gold only" if cause in GOLD_ONLY_CAUSES else "pred only")
        )
        rows.append(
            [cause, kind, d["units"], d["gold"], d["pred"], d["scanned"], d["digital"], d["docs"]]
        )
    rows.append(
        [
            "**total**",
            "",
            sum(d["units"] for d in cc.values()),
            sum(d["gold"] for d in cc.values()),
            sum(d["pred"] for d in cc.values()),
            sum(d["scanned"] for d in cc.values()),
            sum(d["digital"] for d in cc.values()),
            len({u.doc for u in units}),
        ]
    )
    w(
        md_table(
            [
                "cause",
                "unit",
                "units",
                "gold rows (row_missing)",
                "pred rows (row_extra)",
                "scanned units",
                "digital units",
                "docs",
            ],
            rows,
        )
    )
    kinds = Counter(u.link_kind for u in units if u.side == "pair")
    w(
        f"Links: {sum(kinds.values())} gold-predicted pairs formed from the unpaired rows "
        f"(identifier evidence {kinds.get('ident', 0)}, equal quantity on the same page "
        f"{kinds.get('qty', 0)}, position {kinds.get('position', 0)}); gold-only "
        f"{sum(u.side == 'gold' for u in units)}, pred-only "
        f"{sum(u.side == 'pred' for u in units)}. A pair is one gold row (row_missing) AND one "
        "predicted row (row_extra) that are the same table line read wrongly, not a missing line.\n"
    )
    resid = [
        u
        for u in units
        if u.cause in ("missing_other", "extra_other", "spn_other")
        or u.cause.startswith("weak_link")
    ]
    w(
        f"Unclassified residue (`missing_other`, `extra_other`, `spn_other`, `weak_link_*`): "
        f"{len(resid)} units of {len(units)}.\n"
    )
    w("Gold side: cause x scan x page position (rows; pairs counted at the gold row's page):\n")
    w(_pos_table(units, "gold", [c for c in ALL_CAUSES if c not in PRED_ONLY_CAUSES]))
    w("Predicted side: cause x scan x page position (rows; pairs counted at the predicted page):\n")
    w(_pos_table(units, "pred", [c for c in ALL_CAUSES if c not in GOLD_ONLY_CAUSES]))
    w("Near a page boundary (first / last 3 rows) and page layout method, gold side:\n")
    rows = []
    for cause in ALL_CAUSES:
        sel = [u for u in units if u.cause == cause and u.side in ("pair", "gold")]
        if sel:
            pm = Counter(u.extra.get("page_method", "?") for u in sel)
            rows.append(
                [
                    cause,
                    sum(u.g_near is True for u in sel),
                    sum(u.g_near is False for u in sel),
                    sum(u.g_near is None for u in sel),
                    pm.get("table_header", 0),
                    pm.get("continuation", 0),
                    pm.get("y_fraction", 0),
                ]
            )
    w(
        md_table(
            [
                "cause",
                "near boundary",
                "interior",
                "page unknown",
                "on table_header page",
                "on continuation page",
                "on y_fraction page",
            ],
            rows,
        )
    )
    w("Locator evidence for the unpaired gold rows (the gold part number in the OCR of its doc):\n")
    lv: dict[str, Counter[str]] = defaultdict(Counter)
    for u in units:
        if u.side in ("pair", "gold"):
            lv[u.cause][u.gold_match or u.gold_spn_level] += 1
    names = ["exact", "normalized", "fuzzy", "page_fuzzy", "unassigned"]
    w(
        md_table(
            [
                "cause",
                "own line: exact",
                "own line: normalized",
                "own line: fuzzy",
                "page only (page_fuzzy)",
                "not located",
            ],
            [[c, *[lv[c][n] for n in names]] for c in ALL_CAUSES if c in lv],
        )
    )
    sp_units = [u for u in units if u.side == "pair" and "edit" in u.extra]
    if sp_units:
        w(
            "Part-number evidence on the pairs whose part number is wrong "
            "(`spn_misread`, `spn_other`, `spn_copies_other_slot`, `weak_link_qty`). `edit` = how "
            "the predicted part number differs from gold; `OCR snap` = would the nearest OCR "
            "span (locator fuzzy match, threshold from reports/provenance.md) replace the "
            "predicted value by the gold value (candidate rule R4):\n"
        )
        edits = ["glyph_confusion", "char_dropped_or_added", "mixed", "other_substitution",
                 "punctuation_only", "n/a"]  # fmt: skip
        snaps = ["correct", "wrong", "no_candidate", "pred_verbatim_in_ocr"]
        rows = []
        for cause in ("spn_misread", "weak_link_qty", "spn_other", "spn_copies_other_slot"):
            sel = [u for u in sp_units if u.cause == cause]
            if sel:
                ec = Counter(u.extra["edit"] for u in sel)
                sn = Counter(u.extra["snap"] for u in sel)
                gi = Counter(u.extra["gold_in_ocr"] for u in sel)
                rows.append(
                    [cause, len(sel)]
                    + [ec[e] for e in edits]
                    + [gi["exact"] + gi["normalized"]]
                    + [sn[s] for s in snaps]
                )
        w(
            md_table(
                ["cause", "n"]
                + [f"edit: {e}" for e in edits]
                + ["gold in OCR (exact / normalized)"]
                + [f"snap: {s}" for s in snaps],
                rows,
            )
        )
    mis = [u for u in units if u.cause == "spn_misread"]
    if mis:
        pure = sum(u.extra["same_ident"] >= 1 and u.extra["qty_same"] for u in mis)
        none_ident = sum(u.extra["same_ident"] == 0 for u in mis)
        w(
            f"`spn_misread` pairs: {len(mis)}; with another identifier field and the quantity "
            f"right: {pure}; with no other identifier field present (spn + qty only rows): "
            f"{none_ident}.\n"
        )
    shifts = [u for u in units if u.cause == "column_shift"]
    if shifts:
        slot = Counter(
            "spn slot holds po"
            if ("purchase_order", SPN) in u.extra["cross"]
            else "spn slot holds cpn"
            if (CPN, SPN) in u.extra["cross"]
            else "spn slot correct (other slots moved)"
            for u in shifts
        )
        w(
            "`column_shift` pairs by what the supplier_part_number slot holds: "
            + ", ".join(f"{k}: {v}" for k, v in sorted(slot.items()))
            + ".\n"
        )
    w(
        "Cross-tab against the Step L cause of the same gold row (`shipdoc.diagnostics`; Step L "
        "totals: " + ", ".join(f"{k} {v}" for k, v in res["step_l"]["totals"].items()) + "):\n"
    )
    cross = res["step_l"]["cross"]
    sl_names = sorted({b for (_, b) in cross})
    w(
        md_table(
            ["cause (this report)", *sl_names],
            [
                [c, *[cross.get((c, b), 0) for b in sl_names]]
                for c in ALL_CAUSES
                if any((c, b) in cross for b in sl_names)
            ],
        )
    )
    return out


def render_x2(res: Mapping[str, Any]) -> list[str]:
    """Section X2."""
    out: list[str] = []
    w = out.append
    slots = res["slots"]
    kinds = Counter(s["kind"] for s in slots)
    w("## X2. cpn / po null-status mismatches of paired rows\n")
    w(
        f"Scorer-paired invoice rows (full + partial pairs) with a customer_part_number / "
        f"purchase_order null-status mismatch: {len(slots)}. By kind: "
        + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items()))
        + ". Kinds: `po_in_cpn_slot` = gold cpn empty, predicted cpn equals the gold PO, predicted "
        "PO empty; `cpn_false_fill` / `po_false_fill` = gold empty, prediction filled; "
        "`cpn_over_null` / `po_over_null` = gold filled, prediction empty; `mixed` = both fields "
        "differ in null status in another way.\n"
    )
    gc = res["group_cols"]
    by_g: dict[str, Counter[str]] = defaultdict(Counter)
    for s in slots:
        g = res["slot_docs_group"][s["doc"]]
        by_g[g]["mismatch"] += 1
        by_g[g][s["kind"]] += 1
    rows = []
    shown = set(by_g) | {
        g
        for g, d in gc.items()
        if column_state(d["cpn"], d["rows"]) == "not printed"
        and column_state(d["po"], d["rows"]) == "printed"
        and g in res["paired_by_group"]
    }
    for g in sorted(shown):
        d = gc.get(g, {"docs": 0, "rows": 0, "cpn": 0, "po": 0})
        docs_g = [x for x in res["hdr_labels"] if res["slot_docs_group"][x] == g]
        lab = [res["hdr_labels"][x] for x in docs_g]
        n_hl = sum(x["has_header_line"] for x in lab)
        rows.append(
            [
                g,
                res["paired_by_group"].get(g, 0),
                by_g[g]["mismatch"],
                by_g[g]["po_in_cpn_slot"],
                by_g[g]["mismatch"] - by_g[g]["po_in_cpn_slot"],
                column_state(d["cpn"], d["rows"]),
                column_state(d["po"], d["rows"]),
                f"{sum(x['cpn'] for x in lab)}/{n_hl}",
                f"{sum(x['po'] for x in lab)}/{n_hl}",
            ]
        )
    w(
        "Per supplier group (every group with a mismatch, plus every group whose layout has a PO "
        "column but no customer-part column; layout columns from the gold of train+dev pooled: "
        "`not printed` = every gold row of the group has that column empty, a group-wide layout "
        "constant per recon; the last two columns = dev docs of the group whose OCR "
        "table-header line names a customer-part / a PO column, over dev docs with a detected "
        "header line):\n"
    )
    w(
        md_table(
            [
                "group",
                "paired rows",
                "mismatch rows",
                "po_in_cpn_slot",
                "other mismatches",
                "cpn column (gold)",
                "po column (gold)",
                "OCR header names cpn col",
                "OCR header names po col",
            ],
            rows,
        )
    )
    other: Counter[tuple[str, str]] = Counter()
    for s in slots:
        if s["kind"] != "po_in_cpn_slot":
            other[(res["slot_docs_group"][s["doc"]], s["kind"])] += 1
    w(
        "The mismatches that are not `po_in_cpn_slot`, by group and kind: "
        + (", ".join(f"{g} {k} {n}" for (g, k), n in sorted(other.items())) or "none")
        + ".\n"
    )
    pic = [s for s in slots if s["kind"] == "po_in_cpn_slot"]
    gs = sorted({res["slot_docs_group"][s["doc"]] for s in pic})
    no_cpn = [g for g in gs if column_state(gc[g]["cpn"], gc[g]["rows"]) == "not printed"]
    has_po = [g for g in gs if column_state(gc[g]["po"], gc[g]["rows"]) == "printed"]
    sc_n = sum(res["slot_docs_scan"][s["doc"]] for s in pic)
    w(
        f"`po_in_cpn_slot` rows: {len(pic)} ({sc_n} scanned, {len(pic) - sc_n} digital) in "
        f"{len(gs)} groups ({', '.join(gs)}); groups whose gold layout prints NO customer-part "
        f"column: {len(no_cpn)} of {len(gs)}; groups whose layout prints a PO column: "
        f"{len(has_po)} of {len(gs)}; rows in a group without a customer-part column: "
        f"{sum(res['slot_docs_group'][s['doc']] in no_cpn for s in pic)} of {len(pic)}. "
        "In these groups there is no customer-part column to read, so the model fills a column "
        "that does not exist: it writes the PO value into the cpn slot and leaves the PO slot "
        "empty.\n"
    )
    same_class = [
        g
        for g in sorted(shown)
        if column_state(gc[g]["cpn"], gc[g]["rows"]) == "not printed"
        and column_state(gc[g]["po"], gc[g]["rows"]) == "printed"
        and g not in gs
    ]
    w(
        "Groups with the SAME layout class (PO column, no customer-part column) in dev and no "
        f"`po_in_cpn_slot` row: {', '.join(same_class) or 'none'} "
        "("
        + ", ".join(f"{g}: {res['paired_by_group'].get(g, 0)} paired rows" for g in same_class)
        + "). So the missing customer-part column is necessary-looking but NOT sufficient: "
        "the error is concentrated in the groups listed above.\n"
    )
    mix = [
        g
        for g, d in gc.items()
        if "mixed" in (column_state(d["cpn"], d["rows"]), column_state(d["po"], d["rows"]))
    ]
    w(f"Mixed groups (column printed on some rows only): {len(mix)}.\n")
    sh = res["shapes"]
    w(
        f"Format shapes (digits -> 9, letters -> A) in the TRAIN gold: {sh['n_cpn_shapes']} cpn "
        f"shapes, {sh['n_po_shapes']} PO shapes, {sh['n_overlap']} shared; {sh['cpn_only']} "
        f"cpn-only and {sh['po_only']} PO-only shapes (basis of candidate rule R3).\n"
    )
    return out


def render_x3(res: Mapping[str, Any]) -> list[str]:
    """Section X3."""
    out: list[str] = []
    w = out.append
    cells = res["waybill"]["cells"]
    w("## X3. Waybill header over-nulls (gold has a value, final prediction empty)\n")
    w(
        f"{len(cells)} cells in {len({x['doc'] for x in cells})} waybill docs; by scan status: "
        f"scanned {sum(x['scanned'] for x in cells)}, digital "
        f"{sum(not x['scanned'] for x in cells)}. `label adjacent` = the field label is on the "
        "value's OCR line or the line right above it (the layout stacks label over value). `raw` "
        "= the model's own page output (`trace.jsonl` raw_text): `null emitted` = the key is "
        "present with a null value, `key missing` = the key is absent.\n"
    )
    rows = []
    for f in WB_FIELDS:
        c = [x for x in cells if x["field"] == f]
        if not c:
            continue
        rows.append(
            [
                f,
                len(c),
                sum(x["scanned"] for x in c),
                sum(not x["scanned"] for x in c),
                sum(x["ocr_exact"] for x in c),
                sum(x["ocr_normalized"] for x in c),
                sum(x["label_in_ocr"] for x in c),
                sum(x["label_adjacent"] for x in c),
                sum(x["raw_state"] == "null_emitted" for x in c),
                sum(x["raw_state"] == "key_missing" for x in c),
                sum(x["raw_state"] not in ("null_emitted", "key_missing") for x in c),
                sum(bool(x["alt_slots"]) for x in c),
            ]
        )
    w(
        md_table(
            [
                "field",
                "over-nulls",
                "scanned",
                "digital",
                "gold value in OCR (exact)",
                "gold value in OCR (normalized or better)",
                "field label in OCR text",
                "label adjacent to the value",
                "raw: null emitted",
                "raw: key missing",
                "raw: other",
                "value sits in another raw header slot",
            ],
            rows,
        )
    )
    alts: Counter[str] = Counter()
    for x in cells:
        for k in x["alt_slots"]:
            alts[f"{x['field']} -> {k}"] += 1
    w(
        "Other slots of the model's raw header that hold the over-nulled value: "
        + (", ".join(f"{k}: {v}" for k, v in sorted(alts.items())) or "none")
        + ". (The model emits the union schema, including the invoice keys, for waybill pages; "
        "`merge_pages` keeps only the waybill keys, so a value in an invoice key is dropped.)\n"
    )
    w(
        "Control, label presence over ALL waybill docs with a gold value (does the label "
        "explain the over-null?):\n"
    )
    ctl = res["waybill"]["control"]
    w(
        md_table(
            [
                "field",
                "docs",
                "label in OCR (all docs)",
                "predicted non-null",
                "label in OCR when predicted",
                "over-null",
                "label in OCR when over-null",
            ],
            [
                [
                    f,
                    c.get("docs", 0),
                    c.get("label_in_ocr", 0),
                    c.get("predicted", 0),
                    c.get("predicted_label_in_ocr", 0),
                    c.get("docs", 0) - c.get("predicted", 0),
                    c.get("label_in_ocr", 0) - c.get("predicted_label_in_ocr", 0),
                ]
                for f, c in ctl.items()
            ],
        )
    )
    w(
        "Pattern backfill from the OCR text (mawb `ddd-dddddddd`, hawb two letters + 8 digits). "
        "On the over-nulled cells of this run:\n"
    )
    rows = []
    for f in ("mawb", "hawb"):
        c = [x for x in cells if x["field"] == f]
        if c:
            rows.append(
                [
                    f,
                    len(c),
                    sum(x["pattern_unique"] is True for x in c),
                    sum(x["pattern_unique_correct"] is True for x in c),
                    sum(x["pattern_correct"] is True for x in c),
                ]
            )
    w(
        md_table(
            [
                "field",
                "over-nulls",
                "exactly one pattern match in OCR",
                "that unique match is the gold value",
                "gold value among the matches",
            ],
            rows,
        )
    )
    w(
        "Gold-vs-OCR precision of the same pattern over ALL train+dev waybills (no model "
        "involved):\n"
    )
    bc = res["backfill_check"]
    keys = [
        "gold_docs", "unique_right", "unique_wrong", "several_matches", "no_match",
        "gold_null_docs", "gold_null_unique_match", "gold_null_several_matches",
    ]  # fmt: skip
    w(md_table(["field", *keys], [[f, *[c.get(k, 0) for k in keys]] for f, c in bc.items()]))
    return out


def render_x4(
    res: Mapping[str, Any], units: Sequence[Unit], mapping: Mapping[str, Any]
) -> list[str]:
    """Section X4: the cause table, the combined ORACLEs, candidate-rule replays."""
    out: list[str] = []
    w = out.append
    w("## X4. Cause -> fix class -> ORACLE recoverable points\n")
    w(
        "ORACLE replay (**ORACLE upper bound, not achievable**): the rows / cells of the cause "
        "are replaced by gold (pairs: the predicted row becomes the gold row; gold-only rows are "
        "appended; predicted-only rows are removed; slot mismatches get the gold cpn / po; "
        "waybill over-nulls get the gold value), then the unmodified scorer rescores the 100 dev "
        f"docs. Delta = OVERALL points vs the run's {100 * res['base']['OVERALL']:.2f}; CI = "
        f"paired doc-level bootstrap ({res['n_boot']} resamples, seed {SEED}). Effects are NOT "
        "additive (OVERALL is a mean of per-doc scores and row F1 is a ratio); the combined rows "
        "are computed, not summed. Causes with zero rows are listed in X1.3 and have no ORACLE "
        "row.\n"
    )
    orc = {r["name"]: r for r in res["oracle"]}
    cc = cause_counts(units)

    def pt(name: str) -> str:
        r = orc.get(name)
        if not r:
            return "n/a"
        return f"{pts(r['d_OVERALL'])} [{pts(r['ci'][0])}, {pts(r['ci'][1])}]"

    rows = []
    for cause in ALL_CAUSES:
        if cause not in cc:
            continue
        d = cc[cause]
        fc, conf = mapping["cause"].get(cause, ("?", "?"))
        rows.append(
            [
                f"rows: {cause}",
                f"{d['gold']} gold / {d['pred']} pred ({d['docs']} docs)",
                f"{d['scanned']} / {d['digital']}",
                fc,
                pt("cause:" + cause),
                conf,
            ]
        )
    slots = Counter(s["kind"] for s in res["slots"])
    sc_by_kind: dict[str, Counter[bool]] = defaultdict(Counter)
    for s in res["slots"]:
        sc_by_kind[s["kind"]][res["slot_docs_scan"][s["doc"]]] += 1
    for k in sorted(slots):
        fc, conf = mapping["slot"].get(k, mapping["slot"]["_default"])
        rows.append(
            [
                f"paired-row slot: {k}",
                f"{slots[k]} paired rows",
                f"{sc_by_kind[k][True]} / {sc_by_kind[k][False]}",
                fc,
                pt("slot:" + k),
                conf,
            ]
        )
    wb = Counter(x["field"] for x in res["waybill"]["cells"])
    wb_scan: dict[str, Counter[bool]] = defaultdict(Counter)
    for x in res["waybill"]["cells"]:
        wb_scan[x["field"]][x["scanned"]] += 1
    for f in WB_FIELDS:
        if wb[f]:
            fc, conf = mapping["header"].get(f, ("?", "?"))
            rows.append(
                [
                    f"waybill header: {f} over-null",
                    f"{wb[f]} cells",
                    f"{wb_scan[f][True]} / {wb_scan[f][False]}",
                    fc,
                    pt("header:" + f),
                    conf,
                ]
            )
    w(
        md_table(
            [
                "cause",
                "n rows / cells",
                "scanned / digital",
                "fix class",
                "recoverable OVERALL points [95% CI] (ORACLE upper bound, not achievable)",
                "confidence in the classification",
            ],
            rows,
        )
    )
    w("Combined and group ORACLEs (computed jointly, so not the sum of the rows above):\n")
    crow = []
    for name in (
        "slot:all_null_status_mismatch",
        "header:all_waybill_over_nulls",
        "rows:all_unpaired_causes",
        "rows:unpaired + slot mismatches",
        "COMBINED: rows + slots + waybill over-nulls",
    ):
        r = orc.get(name)
        if r:
            crow.append(
                [
                    name,
                    f"{100 * r['OVERALL']:.2f}",
                    pt(name),
                    pts(r["d_header_acc"]),
                    pts(r["d_row_f1"]),
                    pts(r["d_fully_correct"]),
                ]
            )
    w(
        md_table(
            [
                "variant (ORACLE upper bound, not achievable)",
                "OVERALL %",
                "d OVERALL pts [95% CI]",
                "d header acc pts",
                "d row F1 pts",
                "d fully correct pts",
            ],
            crow,
        )
    )
    singles = [
        k
        for k in orc
        if k.startswith(("cause:", "header:", "slot:"))
        and k not in ("header:all_waybill_over_nulls", "slot:all_null_status_mismatch")
    ]
    total = sum(orc[k]["d_OVERALL"] for k in singles)
    comb = orc.get("COMBINED: rows + slots + waybill over-nulls")
    if comb:
        w(
            f"Sum of the {len(singles)} single-cause deltas above = {pts(total)} pts vs the "
            f"combined ORACLE {pts(comb['d_OVERALL'])} pts: the gap "
            f"({pts(comb['d_OVERALL'] - total)}) "
            "is the non-additivity (a doc's OVERALL has a fully-correct component, so fixing the "
            "last wrong cell of a doc is worth more than its cell share).\n"
        )
    w("### Candidate merge-rule replays (achieved on dev, unmodified scorer, NOT oracles)\n")
    w(
        "Four rules that follow from the findings above, replayed one at a time on the run's "
        "predictions. They are NOT upper bounds: they use only the model output, the OCR text "
        "and (R3) format shapes learned from TRAIN gold. R1 and R2 were designed after looking "
        "at the dev over-nulls, so their dev numbers are optimistic; R2's precision is also "
        "checked on all train+dev waybills in X3. `touched` = cells / rows the rule changed; "
        "`fixed` = wrong before, right after; `broken` = right before, wrong after.\n"
    )
    rows = []
    for r in res["rules"]:
        rows.append(
            [
                r["name"],
                r["touched"],
                r["fixed"],
                r["broken"],
                r["wrong_after"],
                f"{pts(r['d_OVERALL'])} [{pts(r['ci'][0])}, {pts(r['ci'][1])}]",
                pts(r["d_row_f1"]),
                pts(r["d_header_acc"]),
            ]
        )
    w(
        md_table(
            [
                "rule",
                "touched",
                "fixed",
                "broken",
                "wrong after",
                "d OVERALL pts [95% CI]",
                "d row F1 pts",
                "d header acc pts",
            ],
            rows,
        )
    )
    w("Mechanisms (one line per cause):\n")
    for key, text in mapping["mechanism"].items():
        w(f"- `{key}`: {text}")
    w("")
    return out


def render_local(res: Mapping[str, Any], limit: int = 400) -> str:
    """Value-level examples (local file only, never committed)."""
    pred, gold = res["_pred"], res["_gold"]
    out = ["# Row errors, value level (LOCAL ONLY, gitignored location, never commit)\n"]
    for u in res["units"][:limit]:
        g = gold[u.doc]["line_items"][u.gi] if u.gi is not None else None
        p = (
            [x for x in pred[u.doc]["line_items"] if isinstance(x, dict)][u.pi]
            if u.pi is not None
            else None
        )
        out.append(
            f"- {u.doc} {u.group} {scan_name(u.scanned)} {u.side}/{u.cause} "
            f"g_page={u.g_page}({u.g_pos}) p_page={u.p_page}({u.p_pos})\n"
            f"    gold: {json.dumps(g, ensure_ascii=False)}\n"
            f"    pred: {json.dumps(p, ensure_ascii=False)}"
        )
    out.append("\n## Waybill over-null cells\n")
    for c in res["waybill"]["cells"]:
        out.append(f"- {json.dumps({k: v for k, v in c.items()}, ensure_ascii=False)}")
    return "\n".join(out) + "\n"


#: Cause -> (fix class, confidence in the classification). Mechanisms are written next to the
#: data they rest on in ``render_x4``; see reports/row_errors.md for the evidence per cause.
#: Cause -> (fix class, confidence in the classification), and one mechanism line per cause. The
#: wording rests on the numbers printed in reports/row_errors.md (X1-X3) and on the candidate-rule
#: replays; edit it together with a re-run, never on its own.
MAPPING: dict[str, Any] = {
    "cause": {
        "spn_misread": (
            "fine-tune target (glyph-level part-number reading); a plain nearest-OCR-span merge "
            "rule does not pay (R4)",
            "high: edit distance <= 2 or similarity >= 0.8 by the taxonomy rule; other fields "
            "right in most pairs; 28 docs",
        ),
        "column_shift": (
            "multi-page context strategy (column order of page 1 given to continuation pages) "
            "+ fine-tune target (continuation-page rows)",
            "medium: most shifts sit on header-less continuation pages and the contrast holds "
            "inside supplier groups, but 7 docs, 13 shifts sit on pages with a header row, and "
            "the mechanism (missing column header) is inferred from OCR layout, not seen",
        ),
        "spn_copies_other_slot": (
            "fine-tune target (single doc; the gold part number is in the OCR text)",
            "low: one doc, mechanism unknown",
        ),
        "spn_other": ("not fixable at this n (single doc)", "low: 2 rows in one doc"),
        "page_short": (
            "multi-page context strategy / fine-tune target (rows dropped inside a page)",
            "low: one doc, 4 rows",
        ),
        "missing_other": ("not fixable at this n", "low"),
    },
    "slot": {
        "po_in_cpn_slot": (
            "prompt / fine-tune target (a layout may have no customer-part column; the PO "
            "column is not the customer part); merge rule R3 (format-shape swap) also fixes it "
            "but rests on shapes of synthetic formats",
            "high: all rows are in groups whose gold layout has no customer-part column and a "
            "PO column",
        ),
        "cpn_false_fill": (
            "fine-tune target (do not fill a customer-part slot the layout lacks)",
            "medium: one group",
        ),
        "_default": ("not fixable at this n (a few cells)", "low"),
    },
    "header": {
        "mawb": (
            "merge rule R2 (OCR pattern backfill) or fine-tune target (waybill-only schema)",
            "high: value and label are in the OCR text, the model emitted null",
        ),
        "carrier": (
            "merge rule R1 (carrier from the model's own supplier_name slot) or prompt "
            "(carrier = letterhead company)",
            "high: the value sits in the supplier_name slot in all cases",
        ),
        "hawb": (
            "merge rule R2 (OCR pattern backfill) or fine-tune target (waybill-only schema)",
            "medium: 2 cells",
        ),
    },
    "mechanism": {
        "spn_misread": "the part number is the right line read with 1-2 wrong characters "
        "(see the edit profile in X1.3); the model, not the OCR layout, is the weak reader here "
        "and the OCR text is wrong too in a part of the cases, so snapping to it breaks as many "
        "rows as it fixes.",
        "column_shift": "cpn / po / part-number values land one slot to the left or right; "
        "concentrated on continuation pages that carry no column-header row, where the model "
        "has to guess the column order.",
        "spn_copies_other_slot": "the part-number slot is filled with the PO value of the same "
        "row (one doc, 17 rows), although the gold part number is readable in the OCR text.",
        "page_short": "a page yields fewer rows than the gold page holds (one doc).",
        "slot:po_in_cpn_slot": "layouts without a customer-part column: the PO value is "
        "written into the customer_part_number slot and purchase_order is left empty.",
        "header:mawb": "the model emits a null mawb although MAWB value and label are on the "
        "page and in the OCR text; mostly scanned pages.",
        "header:carrier": "the carrier is the letterhead company name (no 'carrier' label on "
        "any waybill); the model writes it into the invoice-schema supplier_name slot, which "
        "the merge drops.",
        "header:hawb": "the model emits a null hawb although value and label are in the OCR "
        "text (scanned pages).",
    },
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--gold-dir", type=Path, default=None)
    ap.add_argument("--meta", type=Path, default=ROOT / "meta" / "dev.json")
    ap.add_argument("--ocr-cache", type=Path, default=None)
    ap.add_argument("--out-md", type=Path, default=ROOT / "reports" / "row_errors.md")
    ap.add_argument("--local-md", type=Path, default=None)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args(argv)
    res = analyse(
        a.run_dir,
        a.gold_dir or paths.data_dir() / a.split / "labels",
        a.meta,
        a.ocr_cache,
        a.n_boot,
    )
    a.out_md.write_text(render_md(res, a.run_dir.name, git_state(), MAPPING), encoding="utf-8")
    local = a.local_md or paths.runs_dir() / "diagnosis" / "row_errors_local.md"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(render_local(res), encoding="utf-8")
    print(f"wrote {a.out_md} and {local}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
