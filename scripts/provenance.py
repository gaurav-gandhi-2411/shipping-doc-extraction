"""Step G: page-provenance table for train+dev.

1. Choose the locator's fuzzy threshold T on data (scripts/_threshold.py) and check it equals
   ``shipdoc.locate.FUZZY_THRESHOLD``.
2. Locate every non-null header field and every row field in each doc's OCR pages; classify the
   page position and region (shipdoc.layout) of each hit. Per-doc output (boxes, matched text) goes
   to ``<SHIPDOC_RUNS_DIR>/provenance/locations.jsonl`` and never into the repo.
3. Aggregate per supplier_group x field into ``meta/field_provenance.json`` (counts and shares only).
4. Write ``reports/provenance.md`` (doc_ids, field names, aggregates; no label values, no OCR text).

Run: ``uv run python scripts/provenance.py``
"""

# ruff: noqa: E501  # long markdown table/prose literals in the report writer

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from typing import Any

from _corpus import ROOT, DocRec, header_fields, is_empty, load_corpus, row_fields
from _threshold import study

from shipdoc import locate as loc
from shipdoc import meta, paths
from shipdoc.layout import analyze_page, page_position
from shipdoc.ocr import page_items

META_OUT = ROOT / "meta" / "field_provenance.json"
REPORT_OUT = ROOT / "reports" / "provenance.md"
# Rule derivation: a rule applies when at least this share of located docs satisfies it, and a
# group overrides it when its own share is below GROUP_DEVIATION (with at least MIN_GROUP_N docs).
RULE_SHARE = 0.90
GROUP_DEVIATION = 0.80
MIN_GROUP_N = 5
MIN_MULTIPAGE_N = 10
REGIONS = ("header", "table", "footer", "banner")
# Gold-null header fields we look for in the OCR without using gold: label + value-shape detectors.
NULL_FIELDS = ("invoice_number", "invoice_date", "awb_number", "hawb")
# Label words that mark the line a field is printed on; only used to order otherwise equal matches.
LABEL_HINT = {
    "total_amount": r"total",
    "invoice_number": r"invoice",
    "invoice_date": r"date",
    "awb_number": r"awb|waybill",
    "currency": r"currency",
}
MAX_OCC = 40  # per (doc, field); numerics such as a currency code can match many lines
LABELS = {
    "invoice_number": r"invoice\s*(?:no|number|num|#)",
    "invoice_date": r"\bdate\b",
    "awb_number": r"awb|waybill",
    "hawb": r"hawb|house",
}
# Manual inspection of not-locatable cases (filled in after viewing the OCR text; categories only).
INSPECTION: list[dict[str, str]] = [
    {
        "doc_id": "train_0002",
        "field": "pieces",
        "scanned": "yes",
        "category": "OCR misread",
        "note": "image prints a different digit than the OCR read (viewed)",
    },
    {
        "doc_id": "train_0270",
        "field": "pieces",
        "scanned": "yes",
        "category": "OCR misread",
        "note": "value read as a letter",
    },
    {
        "doc_id": "train_0322",
        "field": "pieces",
        "scanned": "yes",
        "category": "OCR misread",
        "note": "image digit misread as another digit (viewed)",
    },
    {
        "doc_id": "dev_0011",
        "field": "pieces",
        "scanned": "yes",
        "category": "OCR misread",
        "note": "single digit misread",
    },
    {
        "doc_id": "dev_0011",
        "field": "origin_airport",
        "scanned": "yes",
        "category": "OCR misread",
        "note": "code garbled on a scan",
    },
    {
        "doc_id": "dev_0021",
        "field": "origin_airport",
        "scanned": "yes",
        "category": "OCR misread",
        "note": "code garbled on a scan",
    },
    {
        "doc_id": "train_0286",
        "field": "destination_airport",
        "scanned": "yes",
        "category": "OCR misread",
        "note": "one letter of the 3-letter code misread; ratio 67 cannot reach T",
    },
    {
        "doc_id": "train_0334",
        "field": "invoice_number",
        "scanned": "yes",
        "category": "OCR misread",
        "note": "image prints a clean id (viewed); small monospace font read as another digit string",
    },
    {
        "doc_id": "train_0107",
        "field": "total_amount",
        "scanned": "yes",
        "category": "off-page / absent from OCR",
        "note": "page 2 OCR returned only the page footer; totals line not in the OCR",
    },
    {
        "doc_id": "train_0115",
        "field": "total_amount",
        "scanned": "yes",
        "category": "formatting (separator confusion)",
        "note": "thousands separators read as a mix of . and ,",
    },
    {
        "doc_id": "train_0157",
        "field": "total_amount",
        "scanned": "no",
        "category": "formatting (separator confusion)",
        "note": "thousands separators read as .",
    },
    {
        "doc_id": "train_0158",
        "field": "total_amount",
        "scanned": "no",
        "category": "formatting (separator confusion)",
        "note": "thousands separators read as .",
    },
    {
        "doc_id": "train_0178",
        "field": "total_amount",
        "scanned": "no",
        "category": "formatting (separator confusion)",
        "note": "thousands separators read as a mix of . and ,",
    },
    {
        "doc_id": "train_0314",
        "field": "total_amount",
        "scanned": "no",
        "category": "formatting (separator confusion)",
        "note": "thousands separators read as a mix of . and ,",
    },
    {
        "doc_id": "dev_0037",
        "field": "total_amount",
        "scanned": "yes",
        "category": "formatting (separator confusion)",
        "note": "thousands separators read as a mix of . and ,",
    },
]


def shape(tok: str) -> str:
    """Letters -> A, digits -> 9, other characters kept."""
    return re.sub(r"[0-9]", "9", re.sub(r"[A-Za-z]", "A", tok))


def tokens(text: str) -> list[str]:
    """Whitespace tokens with surrounding punctuation stripped."""
    return [t for t in (x.strip(".,;:()[]") for x in text.split()) if t]


def learn_shapes(corpus: list[DocRec]) -> dict[str, set[str]]:
    """Value shapes of each null-prone field, from non-null gold of train+dev."""
    out: dict[str, set[str]] = {f: set() for f in NULL_FIELDS}
    for r in corpus:
        for f in NULL_FIELDS:
            v = r.gold["header"].get(f)
            if not is_empty(v) and f != "invoice_date":
                out[f].add(shape(str(v).strip()))
    return out


def null_candidates(
    rec: DocRec, pages: list[Any], layouts: list[Any], shapes: dict[str, set[str]]
) -> list[dict[str, Any]]:
    """Value-shaped tokens (or dates) printed in a doc whose gold header field is null.

    No gold value is used: a token qualifies when it has the learned shape of that field (or, for
    the date, parses as a date). ``banner`` = on a "continued" banner line; ``label`` = the line
    carries the field's label; else ``shape``.
    """
    out: list[dict[str, Any]] = []
    for f in header_fields(rec.doc_type):
        if f not in NULL_FIELDS or not is_empty(rec.gold["header"].get(f)):
            continue
        for pno, page in enumerate(pages):
            lines: dict[int, list[str]] = defaultdict(list)
            for it in page_items(page):
                lines[it.line_idx].append(it.text)
            for lidx in sorted(lines):
                text = " ".join(lines[lidx])
                if f == "invoice_date":
                    hits = loc.date_forms(text)
                else:
                    hits = [t for t in tokens(text) if shape(t) in shapes[f] and len(t) >= 6]
                if not hits:
                    continue
                region = layouts[pno].region(lidx)
                if region == "banner":
                    det = "banner"
                elif re.search(LABELS[f], text, re.IGNORECASE):
                    det = "label"
                else:
                    det = "shape"
                out.append(
                    {
                        "scope": "null_candidate",
                        "doc_id": rec.doc_id,
                        "field": f,
                        "detector": det,
                        "page": pno,
                        "line_idx": lidx,
                        "region": region,
                        "n_pages": len(pages),
                        "text": text,
                    }
                )
    return out


def header_records(
    rec: DocRec, index: loc.DocIndex, layouts: list[Any], n_pages: int
) -> list[dict[str, Any]]:
    """One record per non-null gold header field: best match, all per-page occurrences."""
    out: list[dict[str, Any]] = []
    sc_kind = loc.kind_of
    for f in header_fields(rec.doc_type):
        v = rec.gold["header"].get(f)
        if is_empty(v):
            continue
        ms = loc.find_matches(v, f, index)
        base = {
            "scope": "header",
            "doc_id": rec.doc_id,
            "split": rec.split,
            "doc_type": rec.doc_type,
            "group": rec.group,
            "scanned": rec.scanned,
            "n_pages": n_pages,
            "field": f,
            "kind": sc_kind(f),
        }
        if not ms:
            out.append({**base, "located": False, "level": None, "pages_found": [], "occ": []})
            continue
        # Tie-break equal-quality matches towards the line that carries the field's label, so a
        # single-row invoice's total is located at the "Total" line, not at the identical line total.
        hint = re.compile(LABEL_HINT.get(f, "$^"), re.IGNORECASE)
        ms = sorted(
            ms,
            key=lambda m: (
                loc.LEVEL_RANK[m.level],
                -m.score,
                0 if hint.search(index.line_text(m.page, m.line_idx)) else 1,
            ),
        )
        occ = [
            {
                "page": m.page,
                "line_idx": m.line_idx,
                "region": layouts[m.page].region(m.line_idx),
                "banner": layouts[m.page].region(m.line_idx) == "banner",
                "level": m.level,
                "score": m.score,
            }
            for m in ms[:MAX_OCC]
        ]
        pages_found = sorted({m.page for m in ms})
        best = ms[0]
        lay = layouts[best.page]
        region = lay.region(best.line_idx)
        out.append(
            {
                **base,
                "located": True,
                "level": best.level,
                "score": best.score,
                "page": best.page,
                "position": page_position(best.page, n_pages),
                "region": region,
                "region_method": lay.method,
                "on_continued_banner": region == "banner",
                "line_idx": best.line_idx,
                "box": list(best.box),
                "text": best.text,
                "pages_found": pages_found,
                "occ": occ,
            }
        )
    return out


def row_records(
    rec: DocRec, index: loc.DocIndex, layouts: list[Any], n_pages: int
) -> list[dict[str, Any]]:
    """One record per gold row: assigned line, region, and per-field on-line / doc-level hits."""
    rows = rec.gold.get("line_items") or []
    if not rows:
        return []
    assigned = loc.assign_rows(rows, index)
    fields = row_fields()
    out: list[dict[str, Any]] = []
    doc_level: dict[tuple[str, str], str | None] = {}
    for a in assigned:
        row = rows[a.row_idx]
        fl: dict[str, dict[str, Any]] = {}
        for f in fields:
            v = row.get(f)
            if is_empty(v):
                continue
            key = (f, str(v))
            if key not in doc_level:
                m = loc.locate(v, f, index)
                doc_level[key] = m.level if m else None
            fl[f] = {
                "doc_level": doc_level[key],
                "on_line": f == "supplier_part_number" and a.match is not None or f in a.on_line,
            }
        rec_out: dict[str, Any] = {
            "scope": "row",
            "doc_id": rec.doc_id,
            "split": rec.split,
            "group": rec.group,
            "scanned": rec.scanned,
            "n_pages": n_pages,
            "row_idx": a.row_idx,
            "assigned": a.match is not None,
            "reason": a.reason,
            "level": a.level,
            "assigned_page": a.page,
            "fields": fl,
        }
        if a.match is not None:
            lay = layouts[a.match.page]
            rec_out |= {
                "page": a.match.page,
                "position": page_position(a.match.page, n_pages),
                "line_idx": a.match.line_idx,
                "region": lay.region(a.match.line_idx),
                "region_method": lay.method,
                "box": list(a.match.box),
                "spn_level": a.match.level,
            }
        out.append(rec_out)
    return out


# ---------------------------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------------------------


def _share(n: int, d: int) -> float | None:
    return None if d == 0 else n / d


def stats(recs: list[dict[str, Any]]) -> dict[str, Any]:
    """Provenance statistics of one set of header-field records."""
    n = len(recs)
    located = [r for r in recs if r["located"]]
    multi = [r for r in recs if r["n_pages"] > 1]

    def has(r: dict[str, Any], pred: Any) -> bool:
        return any(pred(o, r["n_pages"]) for o in r["occ"])

    def p1(o: dict[str, Any], _: int) -> bool:
        return bool(o["page"] == 0)

    def last(o: dict[str, Any], n_pages: int) -> bool:
        return bool(o["page"] == n_pages - 1)

    def p1_hdr(o: dict[str, Any], _: int) -> bool:
        return bool(o["page"] == 0 and o["region"] == "header")

    def last_ftr(o: dict[str, Any], n_pages: int) -> bool:
        return bool(o["page"] == n_pages - 1 and o["region"] == "footer")

    reg = Counter(r["region"] for r in located)
    lvl = Counter(r["level"] if r["located"] else "none" for r in recs)
    return {
        "n": n,
        "n_multipage": len(multi),
        "not_locatable_share": _share(n - len(located), n),
        "found_page1_share": _share(sum(has(r, p1) for r in recs), n),
        "found_last_page_share": _share(sum(has(r, last) for r in recs), n),
        "found_page1_share_multipage": _share(sum(has(r, p1) for r in multi), len(multi)),
        "found_last_page_share_multipage": _share(sum(has(r, last) for r in multi), len(multi)),
        "page1_header_of_located": _share(sum(has(r, p1_hdr) for r in located), len(located)),
        "last_page_footer_of_located": _share(sum(has(r, last_ftr) for r in located), len(located)),
        "region_counts": {k: reg.get(k, 0) for k in REGIONS},
        "banner_primary_share": _share(reg.get("banner", 0), len(located)),
        "banner_any_share": _share(
            sum(any(o["banner"] for o in r["occ"]) for r in located), len(located)
        ),
        "match_level_counts": {k: lvl.get(k, 0) for k in ("exact", "normalized", "fuzzy", "none")},
    }


def _rule_shares(recs: list[dict[str, Any]]) -> dict[str, float | None]:
    """Candidate-rule satisfaction among located docs (multipage docs when enough, else all)."""
    located = [r for r in recs if r["located"]]
    multi = [r for r in located if r["n_pages"] > 1]
    base = multi if len(multi) >= MIN_MULTIPAGE_N else located

    def share(pred: Any) -> float | None:
        return _share(sum(any(pred(o, r["n_pages"]) for o in r["occ"]) for r in base), len(base))

    return {
        "page1_header": share(lambda o, n: o["page"] == 0 and o["region"] == "header"),
        "last_page_footer": share(lambda o, n: o["page"] == n - 1 and o["region"] == "footer"),
        "page1": share(lambda o, n: o["page"] == 0),
        "last_page": share(lambda o, n: o["page"] == n - 1),
        "n_basis": len(base),  # type: ignore[dict-item]
    }


RULE_ORDER = ("page1_header", "last_page_footer", "page1", "last_page")


def _pick_rule(shares: dict[str, float | None]) -> str:
    for r in RULE_ORDER:
        s = shares.get(r)
        if s is not None and s >= RULE_SHARE:
            return r
    return "any_page"


def recommend(recs: list[dict[str, Any]]) -> dict[str, Any]:
    """Recommended provenance rule for a field.

    No per-group overrides: shipdoc.merge ignores them (one rule per field for every supplier
    group), so they are not emitted.
    """
    shares = _rule_shares(recs)
    return {"rule": _pick_rule(shares), "shares": shares}


def aggregate(header: list[dict[str, Any]], corpus: list[DocRec]) -> dict[str, dict[str, Any]]:
    """Per field: overall, scanned/digital, per-group stats and the recommended rule."""
    by_field: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in header:
        by_field[r["field"]].append(r)
    out: dict[str, dict[str, Any]] = {}
    for f, recs in by_field.items():
        groups = sorted({r["group"] for r in recs})
        out[f] = {
            "doc_type": recs[0]["doc_type"],
            "overall": stats(recs),
            "by_scanned": {
                "scanned": stats([r for r in recs if r["scanned"]]),
                "digital": stats([r for r in recs if not r["scanned"]]),
            },
            "by_group": {g: stats([r for r in recs if r["group"] == g]) for g in groups},
            "recommended": recommend(recs) if recs[0]["doc_type"] == "invoice" else None,
        }
    return out


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Row-level stats: assignment, region/page of assigned lines, per-field hits."""

    def one(rs: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(rs)
        ass = [r for r in rs if r["assigned"]]
        fields: dict[str, Any] = {}
        for f in row_fields():
            fr = [r["fields"][f] for r in rs if f in r["fields"]]
            lv = Counter(x["doc_level"] or "none" for x in fr)
            fields[f] = {
                "n": len(fr),
                "doc_level_counts": {
                    k: lv.get(k, 0) for k in ("exact", "normalized", "fuzzy", "none")
                },
                "on_assigned_line_share": _share(sum(x["on_line"] for x in fr), len(fr)),
            }
        return {
            "n_rows": n,
            "assigned_share": _share(len(ass), n),
            "reasons": dict(Counter(r["reason"] for r in rs)),
            "level_counts": {
                k: sum(r["level"] == k for r in rs) for k in ("line", "page_fuzzy", "unassigned")
            },
            "unassigned_share": _share(sum(r["level"] == "unassigned" for r in rs), n),
            # Cause of the missing distinct line x what the page-level fallback did with the row.
            "no_line_by_cause_and_level": {
                f"{c}|{lv}": sum(r["reason"] == c and r["level"] == lv for r in rs)
                for c in ("no_free_line", "spn_not_located")
                for lv in ("page_fuzzy", "unassigned")
            },
            "docs_with_no_free_line": len(
                {r["doc_id"] for r in rs if r["reason"] == "no_free_line"}
            ),
            "docs_with_rows": len({r["doc_id"] for r in rs}),
            "region_counts": {k: sum(r["region"] == k for r in ass) for k in REGIONS},
            "region_method_counts": dict(Counter(r["region_method"] for r in ass)),
            "position_counts": dict(Counter(r["position"] for r in ass)),
            "fields": fields,
        }

    groups = sorted({r["group"] for r in rows})
    return {
        "overall": one(rows),
        "by_scanned": {
            "scanned": one([r for r in rows if r["scanned"]]),
            "digital": one([r for r in rows if not r["scanned"]]),
        },
        "by_group": {g: one([r for r in rows if r["group"] == g]) for g in groups},
    }


# ---------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------


def pct(x: float | None, nd: int = 1) -> str:
    """Percent string, '-' for None."""
    return "-" if x is None else f"{100 * x:.{nd}f}%"


def level_mix(s: dict[str, Any]) -> str:
    """exact/normalized/fuzzy/none as percentages of n."""
    c, n = s["match_level_counts"], s["n"]
    return (
        "/".join(f"{100 * c[k] / n:.1f}" for k in ("exact", "normalized", "fuzzy", "none"))
        if n
        else "-"
    )


def write_report(
    thr: dict[str, Any],
    agg: dict[str, Any],
    rows_agg: dict[str, Any],
    layout_methods: Counter[str],
    ff: list[dict[str, Any]],
    unloc: list[dict[str, Any]],
    n_docs: int,
) -> str:
    """Markdown for reports/provenance.md."""
    L: list[str] = []
    w = L.append
    w("# Page provenance of gold values (train+dev)\n")
    w(
        "**Every number in this report is UNVERIFIED** (computed by `scripts/provenance.py`; the "
        "verifier will recompute). Confidential data policy: only doc_ids, field names, anonymised "
        "group ids and aggregates appear here; per-doc boxes and matched text are in "
        "`<SHIPDOC_RUNS_DIR>/provenance/locations.jsonl` (gitignored, on D:).\n"
    )
    w(
        f"Corpus: {n_docs} train+dev docs. Locator: `src/shipdoc/locate.py`; regions: `src/shipdoc/layout.py`.\n"
    )
    w("## 1. Fuzzy threshold T (chosen on data)\n")
    w(
        f"Positives: {thr['n_pos']} non-null gold header values vs their own doc. Negatives: the same "
        f"values vs one random other doc of the same doc_type (seed {thr['seed']}). Fuzzy score = best "
        "rapidfuzz ratio over the scorer-normalized strings of every candidate span. A negative "
        f"already matching at exact/normalized level ({thr['n_neg_coincidental_ab']} of {thr['n_neg']}: "
        "same currency/supplier/buyer printed in both docs) is a coincidence T cannot influence, so "
        f"it is excluded from the false-match rate (FMR); the other {thr['n_neg'] - thr['n_neg_coincidental_ab']} "
        "negatives define FMR(T) = share with best ratio >= T. "
        f"{thr['n_pos_unresolved_at_normalized']} positives are unresolved at exact/normalized and are "
        "what fuzzy matching can rescue.\n"
    )
    w(f"**Chosen T = {thr['T']}** (smallest integer with FMR <= {100 * thr['fmr_target']:.0f}%).\n")
    w(
        "| T | positive recall, total (a+b+c) | fuzzy rescue of the unresolved positives | negative FMR | negative FMR incl. coincidental |"
    )
    w("|---:|---:|---:|---:|---:|")
    shown = {50, 60, 70, 75, 80, 83, 85, 86, 87, 88, 89, 90, 92, 95, 100}
    for r in thr["curve"]:
        if r["T"] in shown:
            mark = " **<- chosen**" if r["T"] == thr["T"] else ""
            w(
                f"| {r['T']}{mark} | {pct(r['positive_recall_total'], 2)} | "
                f"{pct(r['positive_recall_fuzzy_of_unresolved'])} | {pct(r['negative_fmr'], 2)} | "
                f"{pct(r['negative_fmr_incl_coincidental'], 2)} |"
            )
    w("\nPer kind at the pooled T (diagnostic; kinds have different score distributions):\n")
    w(
        "| kind | n pos | n neg | coincidental neg | FMR at T | fuzzy rescue at T | smallest T with FMR<=1% |"
    )
    w("|---|---:|---:|---:|---:|---:|---:|")
    for k, v in sorted(thr["per_kind"].items()):
        row = next(r for r in v["curve"] if r["T"] == thr["T"])
        w(
            f"| {k} | {v['n_pos']} | {v['n_neg']} | {v['n_neg_coincidental_ab']} | "
            f"{pct(row['negative_fmr'], 2)} | {pct(row['positive_recall_fuzzy_of_unresolved'])} | "
            f"{v['T_at_fmr_target']} |"
        )
    w(
        "\n**Date exception.** Dates share a narrow ISO range, so a different date one digit away scores "
        "90: date FMR is 8.1% at every T in 87..90 and no positive date was rescued by fuzzy matching. "
        "The locator therefore uses T=91 for dates (`FUZZY_THRESHOLD_BY_KIND`), i.e. dates are matched "
        "at exact/normalized level only in practice.\n"
    )
    w("## 2. Region detection\n")
    tot = sum(layout_methods.values())
    w(
        "Pages by region method: "
        + ", ".join(f"{k} {v} ({100 * v / tot:.1f}%)" for k, v in sorted(layout_methods.items()))
        + ". `table_header` = a line naming >=2 of qty/part/description/item; `continuation` = banner "
        "and no header row (table resumes under the banner; **deviation from the plain y-fraction "
        "fallback**, which labelled the first rows of every page 2 as `header`); `y_fraction` = "
        "neither (header < 25% of page height, footer > 85%). Waybills have no table, so their "
        "regions are y-fractions only and mean top/middle/bottom.\n"
    )
    for dt in ("invoice", "waybill"):
        w(f"## 3{'a' if dt == 'invoice' else 'b'}. Provenance per {dt} header field\n")
        w(
            "n = gold non-null values. page1 / last = value found (any match level) on page 1 / the "
            "last page (single-page docs count for both). Levels = exact/normalized/fuzzy/not "
            "locatable as % of n. Scanned vs digital split on the right.\n"
        )
        w(
            "| field | n | page1 | last | last (multipage only, n) | not locatable | levels e/n/f/none | scanned: page1 / last / not loc / levels (n) | digital: page1 / last / not loc / levels (n) |"
        )
        w("|---|---:|---:|---:|---|---:|---|---|---|")
        for f, a in agg.items():
            if a["doc_type"] != dt:
                continue
            o, sc, dg = a["overall"], a["by_scanned"]["scanned"], a["by_scanned"]["digital"]
            w(
                f"| {f} | {o['n']} | {pct(o['found_page1_share'])} | {pct(o['found_last_page_share'])} | "
                f"{pct(o['found_last_page_share_multipage'])} ({o['n_multipage']}) | "
                f"{pct(o['not_locatable_share'])} | {level_mix(o)} | "
                f"{pct(sc['found_page1_share'])} / {pct(sc['found_last_page_share'])} / {pct(sc['not_locatable_share'])} / {level_mix(sc)} ({sc['n']}) | "
                f"{pct(dg['found_page1_share'])} / {pct(dg['found_last_page_share'])} / {pct(dg['not_locatable_share'])} / {level_mix(dg)} ({dg['n']}) |"
            )
        w("")
    w("## 4. Region and banner of the best match (invoices)\n")
    w("| field | header | table | footer | banner | any occurrence on a banner (of located) |")
    w("|---|---:|---:|---:|---:|---:|")
    for f, a in agg.items():
        if a["doc_type"] != "invoice":
            continue
        o = a["overall"]
        rc = o["region_counts"]
        n = sum(rc.values())
        w(
            f"| {f} | "
            + " | ".join(pct(rc[k] / n if n else None) for k in REGIONS)
            + f" | {pct(o['banner_any_share'])} |"
        )
    w("\n## 5. Recommended provenance rule per invoice field\n")
    w(
        f"Rule = first of page1_header, last_page_footer, page1, last_page whose share of located "
        f"docs (multipage docs when >= {MIN_MULTIPAGE_N}, else all) is >= {int(100 * RULE_SHARE)}%; "
        f"else `any_page`. One rule per field for every supplier group (no per-group overrides).\n"
    )
    w("| field | rule | page1_header | last_page_footer | page1 | last_page | basis n |")
    w("|---|---|---:|---:|---:|---:|---:|")
    for f, a in agg.items():
        rec = a["recommended"]
        if a["doc_type"] != "invoice":
            continue
        s = rec["shares"]
        w(
            f"| {f} | {rec['rule']} | {pct(s['page1_header'])} | {pct(s['last_page_footer'])} | "
            f"{pct(s['page1'])} | {pct(s['last_page'])} | {s['n_basis']} |"
        )
    w(
        "\nWaybill fields: every waybill is a single page, so no per-field rule is emitted "
        "(`recommended` is null in `meta/field_provenance.json`); read them from page 1.\n"
    )
    w("\n## 6. Any-page rule would false-fill\n")
    w(
        "Docs whose gold header field is null although something with the field's printed shape (or "
        "a date, for invoice_date) is in the OCR. Found without gold values: tokens with the shape "
        "learned from non-null train+dev gold of that field. `banner` = on a 'continued' banner "
        "line, `label` = the line carries the field's label, `shape` = shape only (weaker evidence: "
        "it may be an unrelated token). Page is 1-based.\n"
    )
    red = [x for x in ff if x["null_class"] == "redaction"]
    ab = [x for x in ff if x["null_class"] == "absent_line"]
    w(
        f"Redaction nulls (invoice_number/invoice_date, 35 in gold): {len({(x['doc_id'], x['field']) for x in red})} "
        f"have a match somewhere in the doc; absent-line nulls (awb_number/hawb, 157 in gold): "
        f"{len({(x['doc_id'], x['field']) for x in ab})}.\n"
    )
    w("| doc_id | field | null class | detector | page (of n) | region |")
    w("|---|---|---|---|---|---|")
    for x in sorted(ff, key=lambda x: (x["null_class"], x["field"], x["doc_id"], x["page"])):
        w(
            f"| {x['doc_id']} | {x['field']} | {x['null_class']} | {x['detector']} | {x['page'] + 1} of {x['n_pages']} | {x['region']} |"
        )
    w("\n## 7. Not-locatable gold values\n")
    nl = Counter(u["field"] for u in unloc)
    w(
        f"{len(unloc)} of {sum(a['overall']['n'] for a in agg.values())} non-null header values are not locatable at any level (<= fuzzy at T):\n"
    )
    w("| field | not locatable | of n |")
    w("|---|---:|---:|")
    for f, a in agg.items():
        if nl.get(f):
            w(f"| {f} | {nl[f]} | {a['overall']['n']} |")
    if INSPECTION:
        w("\nManual inspection (categories only, no values):\n")
        w("| doc_id | field | scanned | category | note |")
        w("|---|---|---|---|---|")
        for i in INSPECTION:
            w(f"| {i['doc_id']} | {i['field']} | {i['scanned']} | {i['category']} | {i['note']} |")
        cats = Counter(i["category"] for i in INSPECTION)
        w("\nCategory counts: " + ", ".join(f"{k} {v}" for k, v in sorted(cats.items())) + ".")
    w("\n## 8. Row fields\n")
    o = rows_agg["overall"]
    w(
        f"{o['n_rows']} gold rows; {pct(o['assigned_share'])} assigned to a distinct OCR line "
        f"(reasons: {o['reasons']}; `no_free_line` = every candidate line already taken, typically a table "
        f"that OCR returned as one text blob: {o['docs_with_no_free_line']} of {o['docs_with_rows']} invoices). Assigned-line regions: {o['region_counts']}; positions: "
        f"{o['position_counts']}; region methods: {o['region_method_counts']}.\n"
    )
    lc = o["level_counts"]
    sc_ = rows_agg["by_scanned"]
    w(
        f"Row alignment level (`locate.assign_rows`): `line` {lc['line']}, `page_fuzzy` {lc['page_fuzzy']} "
        f"(no distinct line, but the part number is present on the page at fuzzy >= {loc.FUZZY_THRESHOLD:g}, "
        f"scorer-normalized, over that page's spans and line-crossing word runs), `unassigned` {lc['unassigned']} "
        f"({pct(o['unassigned_share'])} of rows). Rows without a distinct line, by cause|outcome: "
        f"{o['no_line_by_cause_and_level']}. By scan: "
        + "; ".join(
            f"{k} {sc_[k]['level_counts']} (unassigned {pct(sc_[k]['unassigned_share'])} of {sc_[k]['n_rows']}; "
            f"by cause|outcome {sc_[k]['no_line_by_cause_and_level']})"
            for k in ("scanned", "digital")
        )
        + ".\n"
    )
    w("| field | n non-null | exact | normalized | fuzzy | not locatable | on assigned line |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    for f, s in o["fields"].items():
        c = s["doc_level_counts"]
        n = s["n"]
        w(
            f"| {f} | {n} | "
            + " | ".join(
                pct(c[k] / n if n else None) for k in ("exact", "normalized", "fuzzy", "none")
            )
            + f" | {pct(s['on_assigned_line_share'])} |"
        )
    w(
        "\n`doc level` = best match anywhere in the doc. Short numerics (quantity) match by chance "
        "somewhere in a table-heavy page, so read quantity through the on-assigned-line column.\n"
    )
    w("## 9. Commands\n")
    w("```\nuv run python scripts/provenance.py\nuv run pytest -q\nuv run ruff check .\n```\n")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------------------------


def main() -> int:
    """Run the study, locate everything, write jsonl + meta + report."""
    corpus = load_corpus()
    thr = study(corpus)
    if thr["T"] != loc.FUZZY_THRESHOLD:
        print(f"chosen T={thr['T']} != locate.FUZZY_THRESHOLD={loc.FUZZY_THRESHOLD}; update it")
        return 1
    shapes = learn_shapes(corpus)
    not_printed = meta.not_printed_fields([r.gold for r in corpus])
    out_dir = paths.runs_dir() / "provenance"
    out_dir.mkdir(parents=True, exist_ok=True)
    header: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    ff: list[dict[str, Any]] = []
    unloc: list[dict[str, Any]] = []
    methods: Counter[str] = Counter()
    with (out_dir / "locations.jsonl").open("w", encoding="utf-8") as fh:
        for n, rec in enumerate(corpus):
            pages = rec.pages()
            index = loc.build_index(pages)
            layouts = [analyze_page(p) for p in pages]
            methods.update(lay.method for lay in layouts)
            hr = header_records(rec, index, layouts, len(pages))
            rr = row_records(rec, index, layouts, len(pages))
            nc = null_candidates(rec, pages, layouts, shapes)
            for x in nc:
                x["null_class"] = (
                    "absent_line" if x["field"] in not_printed[rec.doc_type] else "redaction"
                )
                ff.append(x)
            header += hr
            rows += rr
            unloc += [
                {"doc_id": r["doc_id"], "field": r["field"], "scanned": r["scanned"]}
                for r in hr
                if not r["located"]
            ]
            for r in [*hr, *rr, *nc]:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            if n % 100 == 0:
                print(f"locate {n}/{len(corpus)}", flush=True)
    agg = aggregate(header, corpus)
    rows_agg = aggregate_rows(rows)
    payload = {
        "threshold": {
            "T": thr["T"],
            "fmr_target": thr["fmr_target"],
            "seed": thr["seed"],
            "n_pos": thr["n_pos"],
            "n_neg": thr["n_neg"],
            "n_neg_coincidental_exact_or_normalized": thr["n_neg_coincidental_ab"],
            "n_pos_unresolved_at_normalized": thr["n_pos_unresolved_at_normalized"],
            "per_kind_override": loc.FUZZY_THRESHOLD_BY_KIND,
            "curve": thr["curve"],
            "per_kind": {
                k: {kk: vv for kk, vv in v.items() if kk != "curve"} | {"curve": v["curve"]}
                for k, v in thr["per_kind"].items()
            },
        },
        "region_methods": dict(methods),
        "rule_thresholds": {
            "rule_share": RULE_SHARE,
            "group_deviation": GROUP_DEVIATION,
            "min_group_n": MIN_GROUP_N,
        },
        "header_fields": agg,  # key read by shipdoc.merge.FieldProvenance
        "row_fields": rows_agg,
    }
    META_OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    REPORT_OUT.write_text(
        write_report(thr, agg, rows_agg, methods, ff, unloc, len(corpus)), encoding="utf-8"
    )
    (out_dir / "unlocatable.json").write_text(json.dumps(unloc, indent=1), encoding="utf-8")
    print("wrote", META_OUT, REPORT_OUT, out_dir / "locations.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
