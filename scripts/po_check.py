"""Step H: does any no-PO-column group print a header-level PO in its OCR text?

Greps the reading-ordered OCR text of every page of every train/dev doc for PO labels, in the 8
groups whose gold purchase_order is always null and (as a false-positive control) in the 10
groups that print a PO column. Prints actual values nowhere: value-like tokens are masked.

Run: ``uv run python scripts/po_check.py`` (writes reports/po_check.json; the prose in
reports/po_check.md is hand-written from that output).
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shipdoc.ocr import doc_pages, page_items  # noqa: E402

# Boundaries: a label must not be glued to letters/digits on either side, so "POWER", "POR123"
# and PO-prefixed part numbers do not match. Bare "No." is never a label on its own.
B0 = r"(?<![A-Za-z0-9])"
B1 = r"(?![A-Za-z0-9])"
NO = r"(?:No\.?|Number|Num\.?|#)"
PATTERNS: dict[str, str] = {
    "po_no": rf"P\.?\s?O\.?\s*{NO}",
    # "P.O. Box" is an address, not an order number.
    "po": r"P\.?\s?O\.?(?!\s*Box)",
    "p_o_slash": r"P/O",
    "purchase_order": r"Purch(?:ase|\.)?\s*Order",
    "order_no": rf"Order\s*{NO}",
    "customer_order": r"Customer\s+Order",
    "cust_po": r"Cust\.?\s*P\.?O\.?",
    "buyers_order": r"Buyer[’']?s\s+Order",
    "your_order": r"Your\s+Order",
    "your_ref": r"Your\s+Ref\.?",
    "ref_po": r"Ref\.?\s*P\.?O\.?",
}
# Most specific first: at a given position the first alternative that matches wins.
PO_RE = re.compile(
    B0 + "(?:" + "|".join(f"(?P<{k}>{v})" for k, v in PATTERNS.items()) + ")" + B1,
    re.IGNORECASE,
)
VALUE_RE = re.compile(r"[A-Z0-9][A-Z0-9\-/]{3,}")
# Gold PO shapes from recon section 5: AA9999999999 and A99999999-999.
PO_SHAPE_RE = re.compile(r"(?<![A-Z0-9])(?:[A-Z]{2}\d{10}|[A-Z]\d{8}-\d{3})(?![A-Z0-9\-])")
TABLE_KEYS = ("qty", "quantity", "part", "description", "item")
FOOTER_RE = re.compile(r"\b(sub\s*-?total|grand\s*total|total|amount\s+due|bank|remarks?)\b", re.I)
NO_PO_GROUPS = [
    "inv_g03",
    "inv_g04",
    "inv_g08",
    "inv_g10",
    "inv_g13",
    "inv_g14",
    "inv_g15",
    "inv_g16",
]


@dataclass
class Hit:
    """One PO-label match on a page."""

    doc: str
    page: int
    pattern: str
    region: str  # header | table_header | table | footer
    region_method: str  # table_header_line | y_fraction
    value: str  # same_line | next_line | none
    snippet: str


def split_lines(page: Any) -> list[tuple[str, float]]:
    """Reading-ordered lines as (text, centre-y fraction of page height)."""
    lines: dict[int, list[str]] = defaultdict(list)
    ys: dict[int, float] = {}
    for it in page_items(page):
        lines[it.line_idx].append(it.text)
        ys.setdefault(it.line_idx, (it.box[1] + it.box[3]) / 2 / max(page.height, 1))
    return [(" ".join(lines[i]), ys[i]) for i in sorted(lines)]


def table_header_idx(lines: list[tuple[str, float]]) -> int | None:
    """Index of the first line containing >= 2 distinct table keywords, else None."""
    for i, (t, _) in enumerate(lines):
        low = t.lower()
        if sum(k in low for k in TABLE_KEYS) >= 2:
            return i
    return None


def region_of(lines: list[tuple[str, float]], i: int, th: int | None) -> tuple[str, str]:
    """Classify line `i` as header / table_header / table / footer (and how it was decided)."""
    y = lines[i][1]
    if th is None:
        return ("header" if y < 0.25 else "footer" if y > 0.85 else "table"), "y_fraction"
    if i < th:
        return "header", "table_header_line"
    if i == th:
        return "table_header", "table_header_line"
    if any(FOOTER_RE.search(lines[j][0]) for j in range(th + 1, i + 1)) or y > 0.85:
        return "footer", "table_header_line"
    return "table", "table_header_line"


def mask(text: str) -> str:
    """Replace every value-like token with <VALUE len=N>."""
    return VALUE_RE.sub(lambda m: f"<VALUE len={len(m.group())}>", text)


def scan_doc(doc: str) -> tuple[list[Hit], int, int]:
    """PO-label hits of a doc, its page count and its count of PO-shaped tokens (no values kept)."""
    hits: list[Hit] = []
    shaped = 0
    pages = doc_pages(doc)
    for pn, page in enumerate(pages, 1):
        lines = split_lines(page)
        th = table_header_idx(lines)
        for i, (text, _) in enumerate(lines):
            shaped += len(PO_SHAPE_RE.findall(text))
            for m in PO_RE.finditer(text):
                rest = text[m.end() :]
                nxt = lines[i + 1][0] if i + 1 < len(lines) else ""
                if VALUE_RE.search(rest):
                    val = "same_line"
                elif VALUE_RE.search(nxt):
                    val = "next_line"
                else:
                    val = "none"
                region, method = region_of(lines, i, th)
                snippet = f"{mask(text[: m.start()])}[[{m.group()}]]{mask(rest)}"
                if val == "next_line":
                    snippet += f"  // next line: {mask(nxt)}"
                hits.append(Hit(doc, pn, m.lastgroup or "?", region, method, val, snippet[:200]))
    return hits, len(pages), shaped


def main() -> None:
    """Scan train+dev docs and write reports/po_check.json."""
    groups = json.loads((ROOT / "meta" / "supplier_groups.json").read_text("utf-8"))
    docs = {d: g for d, g in groups.items() if d.split("_")[0] in ("train", "dev")}
    gold_null: dict[str, list[bool]] = defaultdict(list)
    for d, g in docs.items():
        f = ROOT / "data" / d.split("_")[0] / "labels" / f"{d}.json"
        lab = json.loads(f.read_text("utf-8"))
        gold_null[g].append(all(not r.get("purchase_order") for r in lab["line_items"]))
    all_groups = sorted(g for g in gold_null if g.startswith("inv_"))
    all_null = [g for g in all_groups if all(gold_null[g])]
    print("all-null-PO groups from labels:", all_null)
    print("matches recon list:", all_null == NO_PO_GROUPS)

    per: dict[str, dict[str, Any]] = {}
    examples: list[Hit] = []
    for g in all_groups:
        ds = sorted(d for d, gg in docs.items() if gg == g)
        c: dict[str, Any] = {
            "docs": len(ds),
            "docs_no_ocr": 0,
            "docs_hit": 0,
            "docs_hit_header": 0,
            "docs_value": 0,
            "docs_po_shaped_token": 0,
            "pages": 0,
            "patterns": Counter(),
            "regions": Counter(),
            "value": Counter(),
            "no_po_col": g in all_null,
        }
        for d in ds:
            hits, npages, shaped = scan_doc(d)
            c["pages"] += npages
            c["docs_no_ocr"] += npages == 0
            c["docs_po_shaped_token"] += shaped > 0
            if hits:
                c["docs_hit"] += 1
                c["docs_hit_header"] += any(h.region == "header" for h in hits)
                c["docs_value"] += any(h.value != "none" for h in hits)
            for h in hits:
                c["patterns"][h.pattern] += 1
                c["regions"][h.region] += 1
                c["value"][h.value] += 1
                if g in all_null and len(examples) < 40:
                    examples.append(h)
        per[g] = c
    out = {
        "all_null_groups": all_null,
        "regex": PO_RE.pattern,
        "per_group": {
            g: {k: dict(v) if isinstance(v, Counter) else v for k, v in c.items()}
            for g, c in per.items()
        },
        "examples": [h.__dict__ for h in examples],
    }
    (ROOT / "reports" / "po_check.json").write_text(json.dumps(out, indent=1), "utf-8")
    for g, c in per.items():
        print(
            g,
            "NOPO" if c["no_po_col"] else "PO  ",
            f"{c['docs_hit']}/{c['docs']}",
            "hdr",
            c["docs_hit_header"],
            "val",
            c["docs_value"],
            "shaped",
            c["docs_po_shaped_token"],
            dict(c["patterns"]),
            dict(c["regions"]),
            dict(c["value"]),
            "noocr",
            c["docs_no_ocr"],
        )
    for h in examples[:8]:
        print(h.doc, h.page, h.pattern, h.region, h.value, "|", h.snippet)


if __name__ == "__main__":
    main()
