"""Tests for the reading-order page loader in shipdoc.ocr (synthetic items, no cache needed)."""

from __future__ import annotations

import json
from pathlib import Path

from shipdoc.ocr import OcrItem, PageOcr, doc_pages, doc_text, load_page, page_items, page_text


def _item(text: str, x0: float, y0: float, x1: float, y1: float, conf: float = 0.9) -> OcrItem:
    return OcrItem(text, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], conf)


def _page(items: list[OcrItem], name: str = "dev_0001_p1.jpg") -> PageOcr:
    return PageOcr(name, 1000, 1000, "paddleocr", "line", 0, items=items)


def test_out_of_order_items_are_sorted_into_lines() -> None:
    page = _page(
        [
            _item("world", 120, 100, 200, 120),
            _item("second line", 10, 150, 150, 170),
            _item("hello", 10, 102, 100, 122),
        ]
    )
    assert page_text(page) == "hello world\nsecond line"
    items = page_items(page)
    assert [(i.text, i.line_idx) for i in items] == [("hello", 0), ("world", 0), ("second line", 1)]
    assert items[0].box == (10, 102, 100, 122) and items[0].conf == 0.9


def test_slanted_line_stays_together() -> None:
    # Height 20 each; centres drift 8px per item (tolerance is 10), left-to-right descending.
    page = _page(
        [
            _item("c", 220, 116, 300, 136),
            _item("a", 10, 100, 100, 120),
            _item("b", 110, 108, 200, 128),
            _item("next", 10, 200, 100, 220),
        ]
    )
    assert page_text(page) == "a b c\nnext"


def test_slanted_polygon_uses_axis_aligned_box() -> None:
    skew = OcrItem("x", [[10, 10], [100, 20], [100, 40], [10, 30]], 0.5)
    (it,) = page_items(_page([skew]))
    assert it.box == (10, 10, 100, 40)


def test_empty_text_items_dropped() -> None:
    page = _page([_item("", 0, 0, 50, 20), _item("  ", 0, 30, 50, 50), _item("ok", 0, 60, 50, 80)])
    assert [i.text for i in page_items(page)] == ["ok"]
    assert page_text(_page([_item("", 0, 0, 5, 5)])) == ""
    assert page_items(_page([])) == []


def test_cache_loaders_round_trip(tmp_path: Path) -> None:
    d = tmp_path / "paddleocr" / "dev"
    d.mkdir(parents=True)
    for n, text in ((10, "ten"), (2, "two"), (1, "one")):
        page = _page([_item(text, 0, 0, 40, 20)], f"dev_0001_p{n}.jpg")
        (d / f"dev_0001_p{n}.json").write_text(json.dumps(page.to_dict()), encoding="utf-8")
    other = _page([_item("other", 0, 0, 40, 20)], "dev_0010_p1.jpg")
    (d / "dev_0010_p1.json").write_text(json.dumps(other.to_dict()), encoding="utf-8")

    assert load_page("dev", "dev_0001_p2", tmp_path).items[0].text == "two"
    # numeric (not lexicographic) page order; dev_0010 must not leak into dev_0001
    assert [p.items[0].text for p in doc_pages("dev_0001", tmp_path)] == ["one", "two", "ten"]
    assert doc_text("dev_0001", tmp_path) == "one\n\ntwo\n\nten"
    assert doc_pages("dev_9999", tmp_path) == []
