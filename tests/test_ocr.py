"""Unit tests for the model-free parts of shipdoc.ocr."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shipdoc.ocr import (
    OcrItem,
    PageOcr,
    cache_path,
    percentile,
    run_ocr,
    split_line_to_words,
    unrotate_polygon,
)


def test_cache_path_uses_stem_and_engine_split() -> None:
    p = cache_path(Path("cache/ocr"), "paddleocr", "dev", "dev_0003_p2.jpg")
    assert p == Path("cache/ocr/paddleocr/dev/dev_0003_p2.json")
    # png and jpg of the same stem map to the same cache file (stems are unique per page).
    assert cache_path(Path("c"), "e", "train", "x_p1.png") == cache_path(
        Path("c"), "e", "train", "x_p1.jpg"
    )


def test_page_ocr_json_round_trip() -> None:
    item = OcrItem("Invoice No.", [[0, 0], [10, 0], [10, 5], [0, 5]], 0.97)
    page = PageOcr(
        page="train_0001_p1.png",
        width=1240,
        height=1754,
        engine="paddleocr",
        granularity="line",
        rotation_deg=0,
        items=[item],
        words=split_line_to_words(item),
        seconds=1.5,
    )
    restored = PageOcr.from_dict(json.loads(json.dumps(page.to_dict())))
    assert restored == page
    assert all(w.derived for w in restored.words)
    assert not restored.items[0].derived


def test_split_line_to_words_proportional() -> None:
    # "ab cdef": 7 chars over width 70 -> "ab" spans x 0..20, "cdef" spans x 30..70.
    line = OcrItem("ab cdef", [[0, 0], [70, 0], [70, 10], [0, 10]], 0.9)
    words = split_line_to_words(line)
    assert [w.text for w in words] == ["ab", "cdef"]
    assert words[0].box[0] == [0.0, 0.0] and words[0].box[1] == pytest.approx([20.0, 0.0])
    assert words[1].box[0] == pytest.approx([30.0, 0.0])
    assert words[1].box[2] == pytest.approx([70.0, 10.0])
    assert split_line_to_words(OcrItem("", [[0, 0]] * 4, 1.0)) == []


@pytest.mark.parametrize("angle", [0, 90, 180, 270])
def test_unrotate_polygon_inverts_ccw_rotation(angle: int) -> None:
    w, h = 100, 60
    x, y = 70.0, 10.0
    # forward map of a CCW rotation by `angle` (expanded canvas), derived independently.
    fwd = {0: (x, y), 90: (y, w - x), 180: (w - x, h - y), 270: (h - y, x)}[angle]
    assert unrotate_polygon([fwd], angle, w, h) == [[x, y]]
    with pytest.raises(ValueError):
        unrotate_polygon([fwd], 45, w, h)


def test_percentile_nearest_rank() -> None:
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == 10
    assert percentile([5.0], 95) == 5.0


class _FakeEngine:
    """Model-free engine used to exercise the cache driver."""

    name = "fake"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def run(self, image_path: Path) -> PageOcr:
        self.calls.append(image_path.name)
        return PageOcr(image_path.name, 10, 10, self.name, "line", 0, seconds=0.5)

    def describe(self) -> dict[str, object]:
        return {"engine": self.name}


def test_run_ocr_is_resumable_and_idempotent(tmp_path: Path) -> None:
    images = tmp_path / "data" / "train" / "images"
    images.mkdir(parents=True)
    for name in ("a_p1.png", "a_p2.jpg"):
        (images / name).write_bytes(b"")
    engine = _FakeEngine()
    cache = tmp_path / "cache"
    first = run_ocr(engine, ["train"], tmp_path / "data", cache)
    assert first["pages_cached"] == 2 and first["failures"] == []
    assert (cache / "fake" / "train" / "a_p2.json").exists()
    second = run_ocr(engine, ["train"], tmp_path / "data", cache)
    assert engine.calls == ["a_p1.png", "a_p2.jpg"]  # nothing re-run on the second pass
    assert second["pages_cached"] == 2 and second["this_run"]["new_pages"] == 0
