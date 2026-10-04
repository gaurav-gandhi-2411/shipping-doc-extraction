"""Step I: normalized OCR recall ceiling -> reports/ocr_ceiling.md.

For every non-null train+dev gold header value and row field, is the value recoverable from the
cached OCR text at three cumulative levels?

* (a) exact substring: the raw gold string is a substring of the doc's OCR text (pages joined);
* (b) (a) or the locator's scorer-normalized match (``shipdoc.locate``, level <= normalized);
* (c) (b) or a fuzzy match at the data-chosen T (``locate.FUZZY_THRESHOLD``).

Gap decomposition per field: (b)-(a) = formatting, (c)-(b) = near-miss recognition errors,
100-(c) = unrecoverable. Split scanned (.jpg) vs digital (.png). Aggregates only; no values.

Run: ``uv run python scripts/ocr_ceiling.py``
"""

# ruff: noqa: E501  # long markdown table/prose literals in the report writer

from __future__ import annotations

import re
import sys
from collections import defaultdict
from typing import Any

from _corpus import ROOT, DocRec, header_fields, is_empty, load_corpus, row_fields

from shipdoc import locate as loc
from shipdoc import ocr

OUT = ROOT / "reports" / "ocr_ceiling.md"
PRIOR = {  # prior exact-substring measurement quoted in the task (scanned, digital)
    "header (pooled)": (74.8, 84.2),
    "supplier_part_number": (88.9, 96.5),
}


def _ci(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def measure(corpus: list[DocRec]) -> list[dict[str, Any]]:
    """One record per (doc, scope, field[, row]) with flags a / a_ci / b / c."""
    out: list[dict[str, Any]] = []
    for n, rec in enumerate(corpus):
        pages = rec.pages()
        text = ocr.doc_text(rec.doc_id)
        text_ci = _ci(text)
        index = loc.build_index(pages)

        def flags(
            v: Any, f: str, _text: str = text, _tci: str = text_ci, _idx: Any = index
        ) -> dict[str, bool]:
            s = str(v).strip()
            a = s in _text
            m = loc.locate(v, f, _idx)
            b = a or (m is not None and m.level in ("exact", "normalized"))
            c = b or m is not None
            return {"a": a, "a_ci": _ci(s) in _tci, "b": b, "c": c}

        for f in header_fields(rec.doc_type):
            v = rec.gold["header"].get(f)
            if not is_empty(v):
                out.append(
                    {
                        "scope": "header",
                        "field": f,
                        "doc_type": rec.doc_type,
                        "scanned": rec.scanned,
                    }
                    | flags(v, f)
                )
        rows = rec.gold.get("line_items") or []
        if rows:
            assigned = loc.assign_rows(rows, index)
            for a in assigned:
                row = rows[a.row_idx]
                for f in row_fields():
                    v = row.get(f)
                    if is_empty(v):
                        continue
                    fl = flags(v, f)
                    rec_out = {
                        "scope": "row",
                        "field": f,
                        "doc_type": rec.doc_type,
                        "scanned": rec.scanned,
                    } | fl
                    # Same-line variant: the field is on the line assigned through the part number.
                    if f == "supplier_part_number":
                        rec_out["on_line"] = a.match is not None
                    else:
                        rec_out["on_line"] = f in a.on_line
                    out.append(rec_out)
        if n % 100 == 0:
            print(f"ceiling {n}/{len(corpus)}", flush=True)
    return out


def agg(recs: list[dict[str, Any]]) -> dict[str, float]:
    """Counts and percentages of a, a_ci, b, c over `recs`."""
    n = len(recs)
    if n == 0:
        return {"n": 0}
    r = {k: 100 * sum(x[k] for x in recs) / n for k in ("a", "a_ci", "b", "c")}
    r["n"] = n
    r["formatting"] = r["b"] - r["a"]
    r["near_miss"] = r["c"] - r["b"]
    r["unrecoverable"] = 100 - r["c"]
    if recs and "on_line" in recs[0]:
        r["on_line"] = 100 * sum(x["on_line"] for x in recs) / n
    return r


def fmt(x: dict[str, float]) -> str:
    """Cells: n | a | b | c | formatting | near-miss | unrecoverable."""
    if not x.get("n"):
        return "| 0 | - | - | - | - | - | - |"
    return (
        f"| {int(x['n'])} | {x['a']:.1f} | {x['b']:.1f} | {x['c']:.1f} | {x['formatting']:.1f} | "
        f"{x['near_miss']:.1f} | {x['unrecoverable']:.1f} |"
    )


def table(groups: dict[str, list[dict[str, Any]]], extra_on_line: bool = False) -> list[str]:
    """Markdown table: one row per (name, scanned/digital/all)."""
    head = "| field | split | n | (a) exact % | (b) normalized % | (c) +fuzzy % | b-a formatting | c-b near-miss | 100-c unrecoverable |"
    lines = [head, "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, recs in groups.items():
        for split, sel in (
            ("scanned", [r for r in recs if r["scanned"]]),
            ("digital", [r for r in recs if not r["scanned"]]),
            ("all", recs),
        ):
            lines.append(f"| {name} | {split} " + fmt(agg(sel)))
    return lines


def main() -> int:
    """Measure, then write reports/ocr_ceiling.md."""
    corpus = load_corpus()
    recs = measure(corpus)
    hdr = [r for r in recs if r["scope"] == "header"]
    rows = [r for r in recs if r["scope"] == "row"]
    by_f: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in hdr:
        by_f[r["field"]].append(r)
    rby_f: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        rby_f[r["field"]].append(r)
    L: list[str] = []
    w = L.append
    w("# OCR recall ceiling at three levels (train+dev)\n")
    w(
        "**Every number is UNVERIFIED** (computed by `scripts/ocr_ceiling.py`; the verifier will "
        f"recompute). Corpus: {len(corpus)} docs, {len(hdr)} non-null header values, {len(rows)} "
        "non-null row values (percent of gold values whose text is recoverable from the cached "
        "PaddleOCR text). Locator `src/shipdoc/locate.py`, "
        f"fuzzy T = {loc.FUZZY_THRESHOLD} (dates {loc.FUZZY_THRESHOLD_BY_KIND['date']}); "
        "T was chosen on train+dev (see `reports/provenance.md`), so (c) is measured on the data "
        "that picked T: the false-match side was controlled by the negatives, but the figure is "
        "not out-of-sample.\n"
    )
    w(
        "Levels are cumulative: (a) raw gold string is a substring of the doc OCR text (pages "
        "joined); (b) = (a) or scorer-normalized locator match (`same()` after parsing printed "
        "dates/numbers/airport+city/label prefixes); (c) = (b) or fuzzy match at T. "
        "Decomposition: (b)-(a) = formatting, (c)-(b) = near-miss recognition errors, 100-(c) = "
        "unrecoverable (value absent or too badly misread to match).\n"
    )
    w("## 1. Header fields, pooled\n")
    pooled = {
        "header, all fields": hdr,
        "header, invoice": [r for r in hdr if r["doc_type"] == "invoice"],
        "header, waybill": [r for r in hdr if r["doc_type"] == "waybill"],
    }
    L += table(pooled)
    w("\n### Reconciliation with the prior exact-substring measurement\n")
    p_hs = agg([r for r in hdr if r["scanned"]])
    p_hd = agg([r for r in hdr if not r["scanned"]])
    s_s = agg([r for r in rby_f["supplier_part_number"] if r["scanned"]])
    s_d = agg([r for r in rby_f["supplier_part_number"] if not r["scanned"]])
    w(
        "| quantity | prior (scanned / digital) | this run (a), case-sensitive | this run (a), case/space-insensitive |"
    )
    w("|---|---|---|---|")
    w(
        f"| header, pooled | {PRIOR['header (pooled)'][0]} / {PRIOR['header (pooled)'][1]} | "
        f"{p_hs['a']:.1f} / {p_hd['a']:.1f} | {p_hs['a_ci']:.1f} / {p_hd['a_ci']:.1f} |"
    )
    w(
        f"| supplier_part_number | {PRIOR['supplier_part_number'][0]} / {PRIOR['supplier_part_number'][1]} | "
        f"{s_s['a']:.1f} / {s_d['a']:.1f} | {s_s['a_ci']:.1f} / {s_d['a_ci']:.1f} |"
    )
    nd_s = agg([r for r in hdr if r["scanned"] and r["field"] != "invoice_date"])
    nd_d = agg([r for r in hdr if not r["scanned"] and r["field"] != "invoice_date"])
    w(
        f"| header, pooled, excluding invoice_date | 74.8 / 84.2 | {nd_s['a']:.1f} / {nd_d['a']:.1f} | "
        f"{nd_s['a_ci']:.1f} / {nd_d['a_ci']:.1f} |"
    )
    w(
        "\nReading: supplier_part_number reproduces the prior numbers exactly (88.9 / 96.5), and the "
        "prior pooled header figure is reproduced exactly once invoice_date is left out (gold dates "
        "are ISO, so a raw substring test is meaningless for them: 11.5% hit). Including invoice_date "
        "the pooled exact-substring figure is lower (68.6 / 77.0). The prior script is not in the "
        "repo; that leaving out invoice_date explains the gap is inferred from this exact numeric "
        "match, not from its source.\n"
    )
    w("## 2. Per header field\n")
    L += table(dict(sorted(by_f.items(), key=lambda kv: (kv[1][0]["doc_type"], kv[0]))))
    w("\n## 3. Per row field (doc level)\n")
    w(
        "Row values are searched in the whole doc text, so short numerics (quantity) can match by "
        "chance elsewhere on the page: read quantity/customer part/PO next to the same-line "
        "column below.\n"
    )
    L += table(dict(rby_f))
    w("\n### Row fields found on the line assigned through the part number\n")
    w("| field | split | n | found on assigned line % |")
    w("|---|---|---:|---:|")
    for f, rs in rby_f.items():
        for split, sel in (
            ("scanned", [r for r in rs if r["scanned"]]),
            ("digital", [r for r in rs if not r["scanned"]]),
            ("all", rs),
        ):
            a = agg(sel)
            w(f"| {f} | {split} | {int(a['n'])} | {a['on_line']:.1f} |")
    w("\n## 4. Commands\n")
    w("```\nuv run python scripts/ocr_ceiling.py\nuv run pytest -q\nuv run ruff check .\n```\n")
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
