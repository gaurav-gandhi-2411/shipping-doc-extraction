"""Labelled-total detector on OCR pages (evaluation of the generic ``total_amount`` rule).

The generic provenance rule is "the labelled total on any page, preferring the last page"
(`shipdoc.merge`: in the merge itself it is applied to the VLM outputs, not to OCR). This module
is the OCR-side check used by ``scripts/total_rule_eval.py`` to measure how often that rule points
at the gold value on train/dev. It reads the OCR only to decide WHICH printed number is the total;
it never changes a value.

Detection: a reading-order item whose text matches `TOTAL_LABEL` and not `TOTAL_EXCLUDE` is a
label; its number is the one printed after the label inside the same item, else the first item to
its right on the same line that holds digits.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from shipdoc.normalize import normalize_number
from shipdoc.ocr import PageOcr, ReadItem, page_items

#: Specified label shape. ``\s*`` (not ``\s+``) because the OCR often fuses the words
#: ("TotalAmount:USD123.00"; accuracy with and without in reports/total_rule.md, v0 vs v1); the
#: trailing look-ahead replaces a word boundary for the same reason.
TOTAL_LABEL = re.compile(r"\b(grand\s*)?total(\s*amount|\s*due|\s*value)?(?![a-z])", re.IGNORECASE)
#: Labels that are not the invoice total: quantity totals and sub/line totals (column header).
TOTAL_EXCLUDE = re.compile(
    r"total\s*(qty|quantity|pcs|pieces)|sub\s*-?\s*total|line\s*total", re.IGNORECASE
)


@dataclass(frozen=True)
class TotalCandidate:
    """One labelled total: where it is and the printed number text."""

    page: int  # 0-based page index
    line_idx: int
    raw: str  # printed number text, e.g. ``USD 1,234.50``
    value: str | None  # `normalize_number` output; None when the OCR number is not a clean number


def _number_of(text: str) -> tuple[bool, str | None]:
    """``(has_digits, plain number or None)`` for the text printed next to a label."""
    t = text.strip(" :=-")
    if not re.search(r"\d", t):
        return False, None
    value, flags = normalize_number(t)
    return True, (None if "number_unparsed" in flags else value)


def labelled_totals(
    page: PageOcr,
    page_idx: int = 0,
    label: re.Pattern[str] = TOTAL_LABEL,
    exclude: re.Pattern[str] = TOTAL_EXCLUDE,
    keep_unparsed: bool = True,
) -> list[TotalCandidate]:
    """Labelled totals of one page in reading order (top to bottom).

    A label whose number has digits but is not a clean number (an OCR separator or digit error such
    as ``999,99.99``) is kept with ``value=None`` when `keep_unparsed`: the label still locates the
    total, only the OCR digits are unusable.
    """
    by_line: dict[int, list[ReadItem]] = {}
    for it in page_items(page):
        by_line.setdefault(it.line_idx, []).append(it)
    out: list[TotalCandidate] = []
    for line_idx in sorted(by_line):
        line = by_line[line_idx]
        for k, it in enumerate(line):
            m = label.search(it.text)
            if not m or exclude.search(it.text):
                continue
            found: TotalCandidate | None = None
            for text in [it.text[m.end() :], *(n.text for n in line[k + 1 :])]:
                has_digits, value = _number_of(text)
                if value is not None or (has_digits and keep_unparsed):
                    found = TotalCandidate(page_idx, line_idx, text.strip(" :=-"), value)
                    break
            if found is not None:
                out.append(found)
                break  # one total per line
    return out


def pick_total(
    pages: list[PageOcr],
    label: re.Pattern[str] = TOTAL_LABEL,
    exclude: re.Pattern[str] = TOTAL_EXCLUDE,
    keep_unparsed: bool = True,
) -> TotalCandidate | None:
    """The labelled total of the LAST page that has one (bottom-most on that page)."""
    for idx in range(len(pages) - 1, -1, -1):
        cands = labelled_totals(pages[idx], idx, label, exclude, keep_unparsed)
        if cands:
            return cands[-1]
    return None
