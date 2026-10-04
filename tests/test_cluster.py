"""Label-free layout signature and clusters."""

from __future__ import annotations

import numpy as np

from shipdoc.cluster import KEYWORDS, LayoutClusters, layout_signature
from shipdoc.ocr import OcrItem, PageOcr


def _page(*cells: tuple[str, float, float]) -> PageOcr:
    """Items given as (text, x, y) top-left corners on a 1000 x 2000 page."""
    items = [
        OcrItem(t, [[x, y], [x + 100, y], [x + 100, y + 20], [x, y + 20]], 0.99)
        for t, x, y in cells
    ]
    return PageOcr("p", 1000, 2000, "paddleocr", "line", 0, items=items)


def test_signature_has_presence_and_position_of_the_first_matching_item() -> None:
    sig = layout_signature(_page(("Invoice No", 50, 100), ("Date: 2026", 450, 100)))
    i = list(KEYWORDS).index("invoice")
    j = list(KEYWORDS).index("date")
    assert sig.shape == (3 * len(KEYWORDS),)
    assert (
        sig[3 * i] == 1.0
        and abs(sig[3 * i + 1] - 0.1) < 1e-9
        and abs(sig[3 * i + 2] - 0.055) < 1e-9
    )
    assert sig[3 * j] == 1.0 and abs(sig[3 * j + 1] - 0.5) < 1e-9
    k = list(KEYWORDS).index("awb")
    assert sig[3 * k : 3 * k + 3].tolist() == [0.0, 0.0, 0.0]  # absent keyword: all zeros


def test_signature_uses_the_first_occurrence_only() -> None:
    sig = layout_signature(_page(("Total", 100, 1500), ("Total", 100, 1800)))
    i = list(KEYWORDS).index("total")
    assert abs(sig[3 * i + 2] - (1510 / 2000)) < 1e-9


def test_signature_is_label_free_and_deterministic() -> None:
    page = _page(("Bill To", 50, 300), ("Qty", 600, 700))
    assert np.array_equal(layout_signature(page), layout_signature(page))


def test_clusters_separate_two_layouts_and_are_deterministic() -> None:
    a = [layout_signature(_page(("Invoice", 50, 100 + i), ("Total", 800, 1700))) for i in range(6)]
    b = [layout_signature(_page(("Invoice", 700, 900 + i), ("Total", 100, 200))) for i in range(6)]
    c1 = LayoutClusters.fit(a + b, k=2).predict(a + b)
    c2 = LayoutClusters.fit(a + b, k=2).predict(a + b)
    assert c1 == c2
    assert len(set(c1[:6])) == 1 and len(set(c1[6:])) == 1 and c1[0] != c1[-1]
