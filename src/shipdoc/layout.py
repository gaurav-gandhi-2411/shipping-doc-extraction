"""Coarse page regions (header / table / footer / banner) from OCR lines, no gold needed.

The table-header line is found by keywords; lines above it are ``header``, lines from it to the
last item-like line are ``table`` and later lines are ``footer``. A page without a table-header
line falls back to fixed y-fractions and says so in ``method``; a page with a "continued" banner
but no header row is a ``continuation`` page (table from the banner down). A banner at the top of
a page is its own region so a value seen only there can be told apart from a header value.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from shipdoc.ocr import PageOcr, page_items

# A table-header line names at least two of these column families (case-insensitive).
HEADER_FAMILIES: dict[str, str] = {
    "qty": r"\bqty\b|\bquantity\b",
    "part": r"\bpart\b|\bp/n\b|\bpn\b",
    "description": r"\bdescription\b|\bdesc\b",
    "item": r"\bitem\b",
}
MIN_FAMILIES = 2
# Page fractions used when no table-header line and no banner is found: header above, footer below.
HEADER_FRACTION = 0.25
FOOTER_FRACTION = 0.85
# The "continued" banner is one of the first lines of a page.
BANNER_MAX_LINE = 1
_CONTINUED = re.compile(r"continued", re.IGNORECASE)
_TOTALS_OR_PAGE = re.compile(r"\btotal|\bpage\s*\d+\s*of\s*\d+", re.IGNORECASE)
_NUMERIC = re.compile(r"[\d.,]*\d[\d.,]*")


@dataclass(frozen=True)
class PageLayout:
    """Region label per reading-order line of one page."""

    region_by_line: dict[int, str]
    method: str  # "table_header" | "continuation" | "y_fraction"
    header_line: int | None
    last_item_line: int | None

    def region(self, line_idx: int) -> str:
        """Region of a line (``header``/``table``/``footer``/``banner``)."""
        return self.region_by_line.get(line_idx, "table")


def is_table_header(text: str) -> bool:
    """True if `text` names at least ``MIN_FAMILIES`` column families (qty, part, desc, item)."""
    low = text.lower()
    return sum(1 for p in HEADER_FAMILIES.values() if re.search(p, low)) >= MIN_FAMILIES


def is_item_like(text: str) -> bool:
    """A table-row-looking line: 4+ tokens, 2+ of them numeric, and not a totals line."""
    toks = text.split()
    if _TOTALS_OR_PAGE.search(text):  # totals line, or the "Page 1 of 2" footer
        return False
    return len(toks) >= 4 and sum(1 for t in toks if _NUMERIC.fullmatch(t)) >= 2


def analyze_page(page: PageOcr) -> PageLayout:
    """Assign a region to every reading-order line of `page`."""
    lines: dict[int, list[str]] = defaultdict(list)
    ys: dict[int, list[float]] = defaultdict(list)
    for it in page_items(page):
        lines[it.line_idx].append(it.text)
        ys[it.line_idx].append((it.box[1] + it.box[3]) / 2)
    order = sorted(lines)
    text = {i: " ".join(lines[i]) for i in order}
    banner = {i for i in order[: BANNER_MAX_LINE + 1] if _CONTINUED.search(text[i])}
    header = next((i for i in order if i not in banner and is_table_header(text[i])), None)
    region: dict[int, str] = {}
    last_item: int | None = None
    if header is not None:
        after = [i for i in order if i > header and is_item_like(text[i])]
        last_item = after[-1] if after else header
        for i in order:
            if i in banner:
                region[i] = "banner"
            elif i < header:
                region[i] = "header"
            elif i <= last_item:
                region[i] = "table"
            else:
                region[i] = "footer"
        return PageLayout(region, "table_header", header, last_item)
    if banner:
        # A continuation page (banner, no header row): the table resumes right under the banner,
        # so y-fractions would wrongly call its first rows "header". Deviation from plain
        # y-fraction fallback, reported in reports/provenance.md.
        rows = [i for i in order if i not in banner and is_item_like(text[i])]
        last_item = rows[-1] if rows else max(banner)
        for i in order:
            if i in banner:
                region[i] = "banner"
            else:
                region[i] = "table" if i <= last_item else "footer"
        return PageLayout(region, "continuation", None, last_item)
    h = float(page.height) or 1.0
    for i in order:
        cy = sum(ys[i]) / len(ys[i]) / h
        if i in banner:
            region[i] = "banner"
        elif cy < HEADER_FRACTION:
            region[i] = "header"
        elif cy <= FOOTER_FRACTION:
            region[i] = "table"
        else:
            region[i] = "footer"
    return PageLayout(region, "y_fraction", None, None)


def page_position(page_idx: int, n_pages: int) -> str:
    """``only`` / ``first`` / ``middle`` / ``last`` for a 0-based page index."""
    if n_pages == 1:
        return "only"
    if page_idx == 0:
        return "first"
    return "last" if page_idx == n_pages - 1 else "middle"
