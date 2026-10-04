"""Tests for occlusion / scan degradation (synthetic images and invented values only)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from shipdoc import augment as A
from shipdoc.ocr import OcrItem, PageOcr

ROOT = Path(__file__).resolve().parents[1]
needs_scorer = pytest.mark.skipif(
    not (ROOT / "assignment" / "score.py").is_file(), reason="assignment/score.py absent"
)
PAGE = (600, 400)
CENTRE = (200.0, 150.0, 320.0, 174.0)  # a 24px-high "word" away from every edge
LEFT = (20.0, 150.0, 140.0, 174.0)
RIGHT = (470.0, 150.0, 590.0, 174.0)
TOP = (200.0, 10.0, 320.0, 34.0)
BOTTOM = (200.0, 360.0, 320.0, 384.0)
OCR_PAGE = (1200, 800)


def _page_image(mode: str = "L") -> Image.Image:
    """White page with seeded dark vertical stripes standing in for text (high-frequency)."""
    rng = np.random.default_rng(1)
    arr = np.full((PAGE[1], PAGE[0]), 255, np.uint8)
    for y in range(0, PAGE[1], 30):
        cols = rng.integers(0, PAGE[0], size=200)
        arr[y + 2 : y + 22, cols] = 0
    return Image.fromarray(arr).convert(mode)


def _inside(shape: tuple[int, int], ab: tuple[int, int, int, int]) -> np.ndarray:
    m = np.zeros(shape, bool)
    m[ab[1] : ab[3], ab[0] : ab[2]] = True
    return m


def _grad(a: np.ndarray) -> float:
    return float(np.abs(np.diff(a, axis=1)).mean())


@pytest.mark.parametrize("method", ["black_box", "scribble", "smudge"])
@pytest.mark.parametrize("mode", ["L", "RGB"])
def test_geometry_and_outside_untouched(method: str, mode: str) -> None:
    img = _page_image(mode)
    out, ab = A.occlude(img, CENTRE, method, np.random.default_rng(3), PAGE)
    x0, y0, x1, y1 = A.padded_box(CENTRE)
    assert ab[0] <= x0 and ab[1] <= y0 and ab[2] >= x1 and ab[3] >= y1  # contains padded target
    assert ab[0] <= CENTRE[0] and ab[2] >= CENTRE[2]  # ... and so the raw target
    assert 0 <= ab[0] < ab[2] <= PAGE[0] and 0 <= ab[1] < ab[3] <= PAGE[1]
    a, b = np.asarray(img), np.asarray(out)
    inside = _inside(a.shape[:2], ab)
    assert np.array_equal(a[~inside], b[~inside])  # exact bound: smudge feathers inward
    assert not np.array_equal(a[inside], b[inside])
    assert out.size == img.size and out.mode == img.mode


def test_applied_box_clipped_at_page_border() -> None:
    rng = np.random.default_rng(0)
    _, ab = A.occlude(_page_image(), (0.0, 0.0, 50.0, 20.0), "black_box", rng, PAGE)
    assert ab[0] == 0 and ab[1] == 0


def test_methods_make_target_unreadable() -> None:
    img = _page_image()
    tx0, ty0, tx1, ty1 = (int(v) for v in CENTRE)
    base = np.asarray(img, float)[ty0:ty1, tx0:tx1]
    out, _ = A.occlude(img, CENTRE, "black_box", np.random.default_rng(1), PAGE)
    assert np.asarray(out)[ty0:ty1, tx0:tx1].max() <= A.BLACK_BOX_INK[1]
    out, _ = A.occlude(img, CENTRE, "scribble", np.random.default_rng(1), PAGE)
    assert (np.asarray(out)[ty0:ty1, tx0:tx1] <= 120).mean() > 0.8  # dense strokes cover it
    out, _ = A.occlude(img, CENTRE, "smudge", np.random.default_rng(1), PAGE)
    assert _grad(np.asarray(out, float)[ty0:ty1, tx0:tx1]) < 0.1 * _grad(base)  # detail destroyed


@pytest.mark.parametrize(
    ("box", "edge"), [(LEFT, "left"), (RIGHT, "right"), (TOP, "top"), (BOTTOM, "bottom")]
)
def test_edge_crop_fires_near_edges_only(box: tuple[float, ...], edge: str) -> None:
    img = _page_image()
    out, ab = A.occlude(img, box, "edge_crop", np.random.default_rng(2), PAGE)
    reached = {
        "left": ab[0] == 0,
        "right": ab[2] == PAGE[0],
        "top": ab[1] == 0,
        "bottom": ab[3] == PAGE[1],
    }
    assert reached[edge]
    assert ab[0] <= box[0] and ab[2] >= box[2] and ab[1] <= box[1] and ab[3] >= box[3]
    inside = _inside((PAGE[1], PAGE[0]), ab)
    assert (np.asarray(out)[inside] == 255).all()  # page background (white)
    assert np.array_equal(np.asarray(img)[~inside], np.asarray(out)[~inside])
    assert A.edge_crop_eligible(box, PAGE)


def test_edge_crop_rejects_central_box_and_bad_inputs() -> None:
    rng = np.random.default_rng(0)
    assert not A.edge_crop_eligible(CENTRE, PAGE)
    with pytest.raises(ValueError, match="edge_crop"):
        A.occlude(_page_image(), CENTRE, "edge_crop", rng, PAGE)
    with pytest.raises(ValueError, match="unknown"):
        A.occlude(_page_image(), CENTRE, "laser", rng, PAGE)
    with pytest.raises(ValueError, match="page_size"):
        A.occlude(_page_image(), CENTRE, "black_box", rng, (10, 10))


@pytest.mark.parametrize("method", A.METHODS)
def test_occlude_is_deterministic(method: str) -> None:
    img = _page_image()
    a, ba = A.occlude(img, LEFT, method, np.random.default_rng(42), PAGE)  # LEFT: all eligible
    b, bb = A.occlude(img, LEFT, method, np.random.default_rng(42), PAGE)
    c, _ = A.occlude(img, LEFT, method, np.random.default_rng(43), PAGE)
    assert ba == bb and A.png_bytes(a) == A.png_bytes(b)
    if method != "edge_crop":  # edge_crop's fill is fixed; only its box jitters
        assert A.png_bytes(a) != A.png_bytes(c)


def _item(text: str, x0: float, y0: float, x1: float, y1: float) -> OcrItem:
    return OcrItem(text, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], 0.9)


def _ocr(name: str = "dev_0000_p1.png") -> PageOcr:
    """OCR of the synthetic doc; line r sits at y=100+50r (invented values)."""
    lines = [
        "Acme Trading Co",
        "Invoice No.: INV-2026-0042",
        "Line Item Part Qty Description",
        "1 AB-100 12 widget 5.00 60.00",
        "2 CD-200 7 widget 5.00 35.00",
    ]
    items = [
        _item(t, 50, 100 + 50 * r, 50 + 20 * len(t), 124 + 50 * r) for r, t in enumerate(lines)
    ]
    return PageOcr(name, OCR_PAGE[0], OCR_PAGE[1], "paddleocr", "line", 0, items=items)


def _gold() -> dict:
    return {
        "doc_id": "dev_0000",
        "doc_type": "invoice",
        "header": {
            "invoice_number": "INV-2026-0042",
            "supplier_name": "Acme Trading Co",
            "awb_number": None,
        },
        "line_items": [
            {"supplier_part_number": "AB-100", "quantity": "12", "purchase_order": None},
            {"supplier_part_number": "CD-200", "quantity": "7", "purchase_order": None},
        ],
        "pages": ["dev_0000_p1.png"],
    }


@needs_scorer
def test_sample_nulls_only_the_target_header_field() -> None:
    gold, before = _gold(), json.dumps(_gold())
    img = Image.new("L", OCR_PAGE, 255)
    s = A.make_occluded_sample(
        gold, "invoice_number", None, "black_box", 7, images=[img], ocr_pages=[_ocr()]
    )
    expect = _gold()
    expect["header"]["invoice_number"] = None
    assert s.gold == expect  # every other field untouched
    assert json.dumps(gold) == before  # input not mutated
    x0, y0, x1, y1 = s.applied_box
    assert np.asarray(s.images[0])[y0:y1, x0:x1].max() < 30
    assert np.asarray(img).min() == 255  # source image not mutated


@needs_scorer
def test_sample_nulls_only_that_row_field() -> None:
    img = Image.new("L", OCR_PAGE, 255)
    s = A.make_occluded_sample(
        _gold(), "quantity", 1, "scribble", 5, images=[img], ocr_pages=[_ocr()]
    )
    expect = _gold()
    expect["line_items"][1]["quantity"] = None
    assert s.gold == expect  # row 0's quantity and row 1's other fields intact
    assert s.target_box[1] > 190  # drawn on row 1's line (y~200), not row 0's (y~150)


@needs_scorer
def test_sample_rejects_null_and_unlocatable() -> None:
    img = Image.new("L", OCR_PAGE, 255)
    kw = {"images": [img], "ocr_pages": [_ocr()]}
    with pytest.raises(ValueError, match="null"):
        A.make_occluded_sample(_gold(), "awb_number", None, "black_box", 1, **kw)
    g = _gold()
    g["header"]["supplier_name"] = "Not On The Page Ltd"
    with pytest.raises(ValueError, match="no reliable box"):
        A.make_occluded_sample(g, "supplier_name", None, "black_box", 1, **kw)


@needs_scorer
def test_materialize_is_idempotent_and_byte_identical(tmp_path: Path) -> None:
    data, cache, meta = tmp_path / "data", tmp_path / "cache", tmp_path / "meta"
    (data / "dev" / "labels").mkdir(parents=True)
    (data / "dev" / "images").mkdir(parents=True)
    (cache / "paddleocr" / "dev").mkdir(parents=True)
    meta.mkdir()
    (data / "dev" / "labels" / "dev_0000.json").write_text(json.dumps(_gold()), encoding="utf-8")
    Image.new("L", OCR_PAGE, 255).save(data / "dev" / "images" / "dev_0000_p1.png")
    (cache / "paddleocr" / "dev" / "dev_0000_p1.json").write_text(
        json.dumps(_ocr().to_dict()), encoding="utf-8"
    )
    tag = {"doc_id": "dev_0000", "scanned": False, "waybill": False}
    (meta / "dev.json").write_text(json.dumps([tag]), encoding="utf-8")
    (meta / "train.json").write_text("[]", encoding="utf-8")
    recipes = [
        {
            "variant_id": f"dev_0000__syn00{i}",
            "doc_id": "dev_0000",
            "field": "invoice_number",
            "row_idx": None,
            "page": 0,
            "method": m,
            "seed": 10 + i,
            "label": "synthetic",
        }
        for i, m in enumerate(["black_box", "scribble", "smudge"], start=1)
    ]
    out = tmp_path / "out"
    kw = {"data_root": data, "ocr_cache": cache, "meta_dir": meta}
    m1 = A.materialize_synthetic(recipes, out, **kw)
    img_path = out / "images" / "dev_0000__syn001_p1.png"
    mtime = img_path.stat().st_mtime_ns
    m2 = A.materialize_synthetic(recipes, out, **kw)
    assert m1 == m2 and len(m1["sha256"]) == 6  # 3 images + 3 labels
    assert img_path.stat().st_mtime_ns == mtime  # unchanged bytes are not rewritten
    label = json.loads((out / "labels" / "dev_0000__syn002.json").read_text("utf-8"))
    assert label["synthetic"] is True and label["doc_id"] == "dev_0000__syn002"
    assert label["header"]["invoice_number"] is None and label["source_doc_id"] == "dev_0000"


def _two_currency_ocr() -> PageOcr:
    """Currency code printed in the header AND next to the total (invented values)."""
    lines = ["Acme Trading Co", "Currency: USD", "Line Item Part Qty Description",
             "1 AB-100 12 widget 5.00 60.00", "Total Amount: USD 60.00"]  # fmt: skip
    items = [
        _item(t, 50, 100 + 50 * r, 50 + 20 * len(t), 124 + 50 * r) for r, t in enumerate(lines)
    ]
    return PageOcr("dev_0000_p1.png", OCR_PAGE[0], OCR_PAGE[1], "paddleocr", "line", 0, items=items)


@needs_scorer
def test_all_occurrences_hides_every_copy_and_is_deterministic() -> None:
    gold = _gold()
    gold["header"]["currency"] = "USD"
    img = Image.new("L", OCR_PAGE, 255)
    kw = {"images": [img], "ocr_pages": [_two_currency_ocr()], "all_occurrences": True}
    tgts = A.resolve_all_targets(gold, [_two_currency_ocr()], "currency")
    assert [t.line_idx for t in tgts] == [1, 4]  # reading order, one box per line
    s = A.make_occluded_sample(gold, "currency", None, "black_box", 9, n_boxes=2, **kw)
    assert len(s.applied_boxes) == 2 and s.gold["header"]["currency"] is None
    arr = np.asarray(s.images[0])
    for x0, y0, x1, y1 in s.applied_boxes:
        assert arr[y0:y1, x0:x1].max() < 30  # both copies covered
    again = A.make_occluded_sample(gold, "currency", None, "black_box", 9, n_boxes=2, **kw)
    assert A.png_bytes(again.images[0]) == A.png_bytes(s.images[0])
    # the recipe's n_boxes guards against the locator drifting
    with pytest.raises(ValueError, match="recipe says 3"):
        A.make_occluded_sample(gold, "currency", None, "black_box", 9, n_boxes=3, **kw)
    with pytest.raises(ValueError, match="header fields only"):
        A.make_occluded_sample(gold, "quantity", 0, "black_box", 9, **kw)


def test_box_rng_streams() -> None:
    a = A.box_rng(5, 0, False).integers(0, 10**9)
    assert a == np.random.default_rng(5).integers(0, 10**9)  # legacy single-box stream
    streams = {int(A.box_rng(5, k, True).integers(0, 10**9)) for k in range(4)}
    assert len(streams) == 4  # independent per box
