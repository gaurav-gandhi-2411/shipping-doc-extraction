"""Labelled-total detector (OCR-side check of the generic total_amount rule)."""

from __future__ import annotations

import re

from shipdoc.ocr import OcrItem, PageOcr
from shipdoc.totals import TOTAL_EXCLUDE, TOTAL_LABEL, labelled_totals, pick_total


def _page(*lines: list[str]) -> PageOcr:
    """A page whose i-th line holds the given items left to right (one box each)."""
    items = []
    for row, texts in enumerate(lines):
        for col, text in enumerate(texts):
            x0, y0 = 100 + 300 * col, 100 + 60 * row
            box = [[x0, y0], [x0 + 250, y0], [x0 + 250, y0 + 30], [x0, y0 + 30]]
            items.append(OcrItem(text, [[float(x), float(y)] for x, y in box], 0.99))
    return PageOcr("p", 2000, 3000, "paddleocr", "line", 0, items=items)


def test_label_regex_variants() -> None:
    for ok in ("Total Amount: USD 10.00", "TotalAmount:EUR10.00", "GRAND TOTAL", "Total due"):
        assert TOTAL_LABEL.search(ok) and not TOTAL_EXCLUDE.search(ok), ok
    for bad in (
        "Total Qty: 12",
        "TotalQty:12",
        "Total Quantity",
        "Subtotal",
        "Sub-total",
        "Line Total",
    ):
        assert TOTAL_EXCLUDE.search(bad), bad
    assert not TOTAL_LABEL.search("Totalling")  # not a label word


def test_number_in_same_item_or_next_item_to_the_right() -> None:
    same_item = _page(["Total Amount: USD 1,234.50"])
    assert labelled_totals(same_item)[0].value == "1234.50"
    right = _page(["Grand Total", "EUR 99.00"])
    assert labelled_totals(right)[0].value == "99.00"
    fused = _page(["TotalAmount:JPY9,999,999.99"])
    assert labelled_totals(fused)[0].value == "9999999.99"


def test_total_qty_and_line_total_header_are_not_totals() -> None:
    page = _page(["Line No.", "Line Total"], ["Total Qty: 120"], ["Total Amount: USD 5.00"])
    assert [c.value for c in labelled_totals(page)] == ["5.00"]


def test_unparsable_ocr_number_is_kept_as_a_pointer_only_when_asked() -> None:
    page = _page(["Total Amount: USD 999,99.99"])
    assert labelled_totals(page, keep_unparsed=False) == []
    kept = labelled_totals(page)
    assert len(kept) == 1 and kept[0].value is None and "999,99.99" in kept[0].raw


def test_pick_prefers_last_page_then_bottom_most_line() -> None:
    p1 = _page(["Total Amount: USD 10.00"])
    p2 = _page(["Total Amount: USD 20.00"], ["Total Amount: USD 30.00"])
    pick = pick_total([p1, p2])
    assert pick is not None and pick.page == 1 and pick.value == "30.00"
    only_first = pick_total([p1, _page(["no total here"])])
    assert only_first is not None and only_first.page == 0 and only_first.value == "10.00"
    assert pick_total([_page(["nothing"])]) is None
    assert re.fullmatch(r"\d+\.\d\d", pick.value)
