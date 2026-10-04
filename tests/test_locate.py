"""Tests for the OCR locator and page-layout regions (synthetic pages; scorer needed for locate)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shipdoc import locate as L
from shipdoc.layout import analyze_page, is_item_like, is_table_header, page_position
from shipdoc.ocr import OcrItem, PageOcr

ROOT = Path(__file__).resolve().parents[1]
SCORER = ROOT / "assignment" / "score.py"
needs_scorer = pytest.mark.skipif(not SCORER.is_file(), reason="assignment/score.py absent")


def _item(text: str, x0: float, y0: float, x1: float, y1: float) -> OcrItem:
    return OcrItem(text, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], 0.9)


def _page(lines: list[list[str]], name: str = "d_0000_p1.png") -> PageOcr:
    """One item per cell; row r sits at y=100+50r, cells 300px apart."""
    items = [
        _item(t, 50 + 300 * c, 100 + 50 * r, 50 + 300 * c + 25 * len(t), 120 + 50 * r)
        for r, cells in enumerate(lines)
        for c, t in enumerate(cells)
    ]
    return PageOcr(name, 1240, 1754, "paddleocr", "line", 0, items=items)


def _doc() -> list[PageOcr]:
    p1 = _page(
        [
            ["Acme Trading Co., Ltd"],
            ["Invoice No.: INV-2026-0042"],
            ["Date: 17/02/2026"],
            ["Ship From", "ABCDEFVILLE"],
            ["Line Item Part Qty Description"],
            ["1 AB-100 12 widget 5.00 60.00"],
            ["2 AB-100 7 widget 5.00 35.00"],
            ["Total Amount: USD 1.234,50"],
        ]
    )
    p2 = _page(
        [["INVOICE - continued INV-2026-0042"], ["3 ZZ-9 4 gadget 2.00 8.00"]], "d_0000_p2.png"
    )
    return [p1, p2]


# ------------------------------------------------------------------ pure parsers


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("17/02/2026", {"2026-02-17"}),
        ("02/17/2026", {"2026-02-17"}),
        ("05/03/2026", {"2026-03-05", "2026-05-03"}),  # ambiguous: both readings
        ("17-FEB-2026", {"2026-02-17"}),
        ("Feb 17, 2026", {"2026-02-17"}),
        ("February 17, 2026", {"2026-02-17"}),
        ("17 February 2026", {"2026-02-17"}),
        ("2026-02-17", {"2026-02-17"}),
        ("Date:17.02.2026", {"2026-02-17"}),
        ("32/13/2026", set()),
        ("no date here", set()),
    ],
)
def test_date_forms(text: str, expected: set[str]) -> None:
    assert set(L.date_forms(text)) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1,234.50", [1234.5]),
        ("1.234,50", [1234.5]),
        ("1 234,50", [1234.5]),
        ("USD 123,456.78", [123456.78]),
        ("TotalAmount:USD123,456.78", [123456.78]),
        ("$1,000", [1000.0]),
        ("98.6kg", [98.6]),
        ("1.234", [1.234, 1234.0]),  # decimal point or EU thousands: both readings offered
        ("PO1234567890 x", []),  # label glued to digits is not a clean number
        ("12/3", []),
        ("B12345678-001", []),
    ],
)
def test_parse_printed_number(text: str, expected: list[float]) -> None:
    assert L.parse_printed_number(text) == expected


def test_repaired_number_readings() -> None:
    assert L.repaired_number_readings("USD 4.567.890.12") == [4567890.12]
    assert L.repaired_number_readings("CNY7.654,321.09") == [7654321.09]
    assert L.repaired_number_readings("1.234,50") == []  # a valid EU format needs no repair
    assert L.repaired_number_readings("1,234.50") == []


# ------------------------------------------------------------------ locator levels


@needs_scorer
def test_level_exact_substring_and_box() -> None:
    m = L.locate("INV-2026-0042", "invoice_number", _doc())
    assert m is not None and m.level == "exact" and m.page == 0 and m.line_idx == 1
    x0, y0, x1, y1 = m.box
    assert x1 > x0 and y1 > y0 and m.text == "INV-2026-0042"  # shortest span, not the whole line


@needs_scorer
def test_exact_requires_token_boundary_for_non_names() -> None:
    page = _page([["Order 1920 shipped"]])
    assert L.locate("9", "pieces", [page], max_level="exact") is None
    assert L.locate("1920", "pieces", [page], max_level="exact") is not None


@needs_scorer
def test_level_normalized_names_ignore_case_and_punctuation() -> None:
    m = L.locate("ACME TRADING CO LTD", "supplier_name", _doc())
    assert m is not None and m.level == "normalized" and m.score == 100.0


@needs_scorer
def test_level_normalized_id_label_prefix_and_spacing() -> None:
    page = _page([["Waybill #: 123 45678901"]])
    m = L.locate("123-45678901", "awb_number", [page], max_level="normalized")
    assert m is None  # '-' is kept by the scorer, so a space is not a hyphen
    m = L.locate("12345678901", "awb_number", [page], max_level="normalized")
    assert m is not None and m.level == "normalized"


@needs_scorer
@pytest.mark.parametrize(
    "printed",
    ["17/02/2026", "02/17/2026", "17-FEB-2026", "Feb 17, 2026", "17 February 2026"],
)
def test_level_normalized_dates(printed: str) -> None:
    page = _page([[f"Date: {printed}"]])
    m = L.locate("2026-02-17", "invoice_date", [page])
    assert m is not None and m.level == "normalized" and m.text.endswith(printed.split(":")[-1])


@needs_scorer
@pytest.mark.parametrize(
    "printed", ["1,234.50", "1.234,50", "1 234,50", "USD 1,234.50", "EUR1.234,50", "€ 1234.5"]
)
def test_level_normalized_numbers(printed: str) -> None:
    page = _page([[f"Total {printed}"]])
    m = L.locate("1234.50", "total_amount", [page], max_level="normalized")
    assert m is not None and m.level == "normalized"


@needs_scorer
def test_separator_confusion_is_fuzzy_not_normalized() -> None:
    page = _page([["Total Amount: USD 4.567.890.12"]])  # OCR read the thousands commas as dots
    assert L.locate("4567890.12", "total_amount", [page], max_level="normalized") is None
    m = L.locate("4567890.12", "total_amount", [page])
    assert m is not None and m.level == "fuzzy"


@needs_scorer
def test_airport_code_followed_by_city_and_glued() -> None:
    page = _page([["XYZ SAMPLETOWN 6"], ["ABCDEFVILLE QX12345678"]])
    a = L.locate("XYZ", "destination_airport", [page], max_level="normalized")
    b = L.locate("ABC", "origin_airport", [page], max_level="normalized")
    assert a is not None and a.line_idx == 0
    assert b is not None and b.line_idx == 1
    # a code word in free text must not satisfy an airport field
    assert L.locate("LTD", "origin_airport", [_page([["Acme Co LTD"]])], max_level="normalized")


@needs_scorer
def test_level_fuzzy_threshold_and_ordering() -> None:
    page = _page([["Invoice No.: INV-2O26-0042"]])  # OCR read a zero as the letter O
    assert L.locate("INV-2026-0042", "invoice_number", [page], max_level="normalized") is None
    m = L.locate("INV-2026-0042", "invoice_number", [page], threshold=87.0)
    assert m is not None and m.level == "fuzzy" and 87.0 <= m.score < 100.0
    assert L.locate("INV-2026-0042", "invoice_number", [page], threshold=99.0) is None
    far = _page([["Invoice No.: ZZZ-9999-1111"]])
    assert L.locate("INV-2026-0042", "invoice_number", [far], threshold=87.0) is None


@needs_scorer
def test_ngram_span_across_items_on_one_line() -> None:
    page = _page([["Orbitex", "Trading", "S.A. de C.V."]])
    m = L.locate("ORBITEX TRADING SA DE CV", "buyer_name", [page], max_level="normalized")
    assert m is not None and m.level == "normalized"
    # union box covers all three items
    assert m.box[0] <= 50 and m.box[2] >= 50 + 600


@needs_scorer
def test_empty_value_and_all_pages_searched() -> None:
    assert L.locate(None, "invoice_number", _doc()) is None
    assert L.locate("  ", "invoice_number", _doc()) is None
    ms = L.find_matches("INV-2026-0042", "invoice_number", _doc())
    assert sorted(m.page for m in ms) == [0, 1]  # page-1 header and page-2 banner


@needs_scorer
def test_threshold_matches_committed_study() -> None:
    f = ROOT / "meta" / "field_provenance.json"
    if not f.is_file():
        pytest.skip("meta/field_provenance.json not generated yet")
    assert json.loads(f.read_text(encoding="utf-8"))["threshold"]["T"] == L.FUZZY_THRESHOLD


# ------------------------------------------------------------------ row assignment


@needs_scorer
def test_assign_rows_repeated_part_gets_distinct_lines() -> None:
    rows = [
        {"supplier_part_number": "AB-100", "quantity": "7"},
        {"supplier_part_number": "AB-100", "quantity": "12"},
        {"supplier_part_number": "ZZ-9", "quantity": "4"},
        {"supplier_part_number": "QQ-1", "quantity": "1"},
    ]
    got = L.assign_rows(rows, _doc())
    lines = [(g.match.page, g.match.line_idx) for g in got[:3] if g.match]
    assert len(set(lines)) == 3
    assert lines[0] == (0, 6) and lines[1] == (0, 5)  # qty breaks the tie on the same line
    assert "quantity" in got[0].on_line and got[2].match.page == 1
    assert got[3].match is None and got[3].reason == "spn_not_located"


@needs_scorer
def test_assign_rows_no_free_line() -> None:
    rows = [{"supplier_part_number": "ZZ-9"}, {"supplier_part_number": "ZZ-9"}]
    got = L.assign_rows(rows, _doc())
    assert [g.reason for g in got] == ["ok", "no_free_line"]


# ------------------------------------------------------------------ layout


def test_table_header_and_item_like() -> None:
    assert is_table_header("Line No. Your Order Cust. Part Item Code Item Description Qty")
    assert not is_table_header("Invoice No.: 123 Item")
    assert is_item_like("3 B12345678-001 11-2233-44 XX/100/0AB Sensor 1,000 JP 1.2345 1,234.50")
    assert not is_item_like("Total Qty:12,345")
    assert not is_item_like("Total Qty: 12,345 Total Amount: USD 123,456.78")
    assert not is_item_like("Page 1 of 2")


def test_layout_regions_with_table_header_and_continuation_page() -> None:
    p1, p2 = _doc()
    a = analyze_page(p1)
    assert a.method == "table_header"
    assert [a.region(i) for i in range(8)] == [
        "header", "header", "header", "header", "table", "table", "table", "footer",
    ]  # fmt: skip
    b = analyze_page(p2)
    assert b.method == "continuation"
    assert [b.region(0), b.region(1)] == ["banner", "table"]


def test_layout_y_fraction_fallback_when_no_header_row() -> None:
    page = PageOcr(
        "w.png",
        1240,
        1000,
        "paddleocr",
        "line",
        0,
        items=[
            _item("top", 10, 10, 90, 30),
            _item("mid", 10, 500, 90, 520),
            _item("end", 10, 950, 90, 970),
        ],
    )
    a = analyze_page(page)
    assert a.method == "y_fraction"
    assert [a.region(i) for i in range(3)] == ["header", "table", "footer"]


def test_page_position() -> None:
    assert page_position(0, 1) == "only"
    assert [page_position(i, 3) for i in range(3)] == ["first", "middle", "last"]


def _blob_doc() -> list[PageOcr]:
    """Page 1: the whole table is ONE text item (a blob); page 2: a line with a qty."""
    p1 = _page(
        [
            ["Acme Trading Co., Ltd"],
            ["Line Item Part Qty Description"],
            ["1 AB-100 12 widget 2 CD-200 7 gadget 3 EF-300 5 gizmo"],
        ]
    )
    p2 = _page([["Continued"], ["4 CD-200 9 gadget"]], "d_0000_p2.png")
    return [p1, p2]


@needs_scorer
def test_assign_rows_page_fuzzy_fallback() -> None:
    rows = [
        {"supplier_part_number": "AB-100", "quantity": "12"},
        {"supplier_part_number": "EF-300", "quantity": "5"},  # only candidate is the taken blob
        {"supplier_part_number": "ZZ-999", "quantity": "1"},  # nowhere on any page
    ]
    got = L.assign_rows(rows, _blob_doc())
    assert [g.level for g in got] == ["line", "page_fuzzy", "unassigned"]
    assert got[0].match is not None and got[0].page == 0 and got[0].reason == "ok"
    # No box for a page-level placement; the cause of the missing line is kept.
    assert got[1].match is None and got[1].page == 0 and got[1].reason == "no_free_line"
    assert got[2].page is None and got[2].reason == "spn_not_located"


def _two_page_doc() -> list[PageOcr]:
    p1 = _page([["Line Item Part Qty"], ["1 CD-200 7 gadget"], ["Pack of 5"]])
    p2 = _page([["Continued"], ["2 CD-200 8 gadget"], ["Summary 9"]], "d_0000_p2.png")
    return [p1, p2]


@needs_scorer
def test_assign_rows_page_fuzzy_prefers_page_with_remaining_quantity() -> None:
    def third(qty: str) -> L.RowAssignment:
        rows = [{"supplier_part_number": "CD-200", "quantity": q} for q in ("7", "8", qty)]
        got = L.assign_rows(rows, _two_page_doc())
        assert [g.level for g in got[:2]] == ["line", "line"]  # both lines taken by rows 1, 2
        return got[2]

    # Both pages hold the part number, both lines are taken: the quantity picks the page, and the
    # outcome follows the quantity, not the earliest-page default.
    assert (third("9").level, third("9").page) == ("page_fuzzy", 1)
    assert (third("5").level, third("5").page) == ("page_fuzzy", 0)
    assert third("3").page == 0  # quantity nowhere: earliest page


@needs_scorer
def test_assign_rows_page_fuzzy_line_crossing_run() -> None:
    p = _page([["Line Item Part Qty"], ["1 AB-"], ["100 12 widget"]])
    got = L.assign_rows([{"supplier_part_number": "AB-100", "quantity": "12"}], [p])
    assert got[0].level == "page_fuzzy" and got[0].page == 0 and got[0].match is None


@needs_scorer
def test_assign_rows_fallback_leaves_line_assignments_untouched() -> None:
    rows = [{"supplier_part_number": "AB-100"}, {"supplier_part_number": "AB-100"}]
    got = L.assign_rows(rows, _doc())
    assert [(g.level, g.reason) for g in got] == [("line", "ok"), ("line", "ok")]
    assert all(g.page == g.match.page for g in got)
