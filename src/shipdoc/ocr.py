"""OCR and word/box cache.

An `OcrEngine` turns one page image into a `PageOcr`. `run_ocr` walks the split image folders,
writes one JSON per page under ``<ocr cache>/<engine>/<split>/<page_stem>.json`` (the cache
root comes from `shipdoc.paths`) and a run-level ``manifest.json``. Re-running skips pages that
are already cached (resumable and idempotent).

Honesty note: PaddleOCR returns text LINES, not words. Items are therefore recorded with
``granularity: "line"``; approximate word boxes derived by splitting line boxes proportionally to
character counts are stored separately under ``words`` and every one is marked ``derived: true``.
"""

from __future__ import annotations

import json
import platform
import statistics
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from shipdoc import paths

Point = tuple[float, float]

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
# A wedged engine (observed when the disk filled up) fails every following page; stop instead of
# recording hundreds of spurious failures. Failed pages stay uncached, so a rerun retries them.
MAX_CONSECUTIVE_FAILURES = 5
# Fraction of the rotated pass's line count the unrotated pass must reach to be trusted.
ROTATION_KEEP_RATIO = 0.5


@dataclass
class OcrItem:
    """One recognised text unit (a line, or a derived word) in original image pixel coords."""

    text: str
    box: list[list[float]]  # 4-point polygon, original image pixel coordinates
    confidence: float
    derived: bool = False


@dataclass
class PageOcr:
    """OCR result for one page."""

    page: str
    width: int
    height: int
    engine: str
    granularity: str  # "line" for PaddleOCR
    rotation_deg: int | None  # counter-clockwise degrees actually applied to the page
    items: list[OcrItem] = field(default_factory=list)
    words: list[OcrItem] = field(default_factory=list)  # derived approximations, derived=True
    seconds: float = 0.0
    # What the doc-orientation classifier predicted (may differ from rotation_deg when overridden).
    orientation_predicted_deg: int | None = None
    # Recovery steps taken for this page, e.g. "orientation_override", "det_limit_max1280".
    fallbacks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PageOcr:
        """Inverse of `to_dict`."""
        d = dict(data)
        d["items"] = [OcrItem(**i) for i in d.get("items", [])]
        d["words"] = [OcrItem(**i) for i in d.get("words", [])]
        return cls(**d)


class OcrEngine(Protocol):
    """Structural interface every OCR backend implements."""

    name: str

    def run(self, image_path: Path) -> PageOcr:
        """OCR one page image."""
        ...

    def describe(self) -> dict[str, Any]:
        """Versions, model names and parameters, for the manifest."""
        ...


# ---------------------------------------------------------------------------------------------
# Pure helpers (no OCR model needed)
# ---------------------------------------------------------------------------------------------


def cache_path(cache_root: Path, engine: str, split: str, page_name: str) -> Path:
    """Path of the cached JSON for `page_name` (any extension) of `split`."""
    return Path(cache_root) / engine / split / f"{Path(page_name).stem}.json"


def unrotate_polygon(
    poly: Sequence[Sequence[float]], angle_deg: int, orig_w: int, orig_h: int
) -> list[list[float]]:
    """Map a polygon from the orientation-corrected frame back to the original image frame.

    PaddleX corrects page orientation by rotating the image counter-clockwise by `angle_deg`
    (cv2 convention, canvas expanded). Boxes it reports live in that rotated frame; the cache
    stores original-frame pixels, so each point is mapped through the inverse rotation.
    """
    angle = angle_deg % 360
    out: list[list[float]] = []
    for x, y in poly:
        if angle == 0:
            ox, oy = x, y
        elif angle == 90:
            ox, oy = orig_w - y, x
        elif angle == 180:
            ox, oy = orig_w - x, orig_h - y
        elif angle == 270:
            ox, oy = y, orig_h - x
        else:
            raise ValueError(f"unsupported rotation angle: {angle_deg}")
        out.append([float(ox), float(oy)])
    return out


def _lerp(a: Sequence[float], b: Sequence[float], t: float) -> list[float]:
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]


def split_line_to_words(item: OcrItem) -> list[OcrItem]:
    """Approximate word boxes by cutting a line quad proportionally to character counts.

    The quad is assumed ordered TL, TR, BR, BL (PaddleOCR order). A word's span is its character
    range within the line over the total character count; no glyph widths are known, so this is
    an approximation (proportional fonts drift by a few percent) and every result is `derived`.
    """
    text = item.text
    n = len(text)
    if n == 0 or len(item.box) != 4:
        return []
    tl, tr, br, bl = item.box
    words: list[OcrItem] = []
    i = 0
    while i < n:
        if text[i].isspace():
            i += 1
            continue
        j = i
        while j < n and not text[j].isspace():
            j += 1
        t0, t1 = i / n, j / n
        box = [_lerp(tl, tr, t0), _lerp(tl, tr, t1), _lerp(bl, br, t1), _lerp(bl, br, t0)]
        words.append(OcrItem(text[i:j], box, item.confidence, derived=True))
        i = j
    return words


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile (q in [0, 100]) of a non-empty sequence."""
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(-(-q * len(s) // 100)) - 1))
    return s[k]


# ---------------------------------------------------------------------------------------------
# PaddleOCR engine
# ---------------------------------------------------------------------------------------------

PADDLE_VARIANTS: dict[str, tuple[str, str]] = {
    "server": ("PP-OCRv5_server_det", "PP-OCRv5_server_rec"),
    "mobile": ("PP-OCRv5_mobile_det", "PP-OCRv5_mobile_rec"),
}


def device_params(device: str, cpu_threads: int = 8) -> dict[str, Any]:
    """PaddleOCR device kwargs: ``cpu`` (oneDNN on) or ``gpu`` (``gpu:0``, oneDNN off)."""
    if device == "cpu":
        # oneDNN is ~10x faster on CPU. It only works on the pinned paddlepaddle 3.0.0; on
        # 3.3.1 it raises NotImplementedError (see reports/ocr.md).
        return {"device": "cpu", "enable_mkldnn": True, "cpu_threads": cpu_threads}
    if device == "gpu":
        return {"device": "gpu:0", "enable_mkldnn": False}
    raise ValueError(f"unsupported OCR device {device!r}; expected 'cpu' or 'gpu'")


class PaddleOcrEngine:
    """PaddleOCR 3.x (PP-OCRv5) with document + text-line orientation classification.

    Runs on CPU unless ``$SHIPDOC_OCR_DEVICE=gpu`` (set by scripts/colab_ocr_run.sh when the
    isolated Colab venv holds paddlepaddle-gpu).
    """

    name = "paddleocr"

    def __init__(self, variant: str = "server", cpu_threads: int = 8) -> None:
        import os

        device = os.environ.get("SHIPDOC_OCR_DEVICE", "cpu")
        paths.apply_env()  # before paddle is imported: PaddleX reads its model home at import
        # Skip the per-start connectivity probe to the model hoster.
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from paddleocr import PaddleOCR

        det, rec = PADDLE_VARIANTS[variant]
        self.params: dict[str, Any] = {
            "variant": variant,
            "text_detection_model_name": det,
            "text_recognition_model_name": rec,
            "use_doc_orientation_classify": True,
            "use_doc_unwarping": False,
            "use_textline_orientation": True,
            **device_params(device, cpu_threads),
        }
        kwargs = {k: v for k, v in self.params.items() if k != "variant"}
        self._ocr = PaddleOCR(**kwargs)

    def describe(self) -> dict[str, Any]:
        """Versions, models and params for the manifest."""
        import paddle
        import paddleocr
        import paddlex

        return {
            "engine": self.name,
            "granularity": "line",
            "paddleocr_version": paddleocr.__version__,
            "paddlepaddle_version": paddle.__version__,
            "paddlex_version": paddlex.__version__,
            "models": {
                "doc_orientation": "PP-LCNet_x1_0_doc_ori",
                "textline_orientation": "PP-LCNet_x1_0_textline_ori",
                "detection": self.params["text_detection_model_name"],
                "recognition": self.params["text_recognition_model_name"],
            },
            "params": self.params,
            "python": platform.python_version(),
            "platform": platform.platform(),
        }

    def run(self, image_path: Path) -> PageOcr:
        """OCR one page; boxes are returned in original-image pixel coordinates."""
        import numpy as np
        from PIL import Image

        t0 = time.perf_counter()
        # Decode with PIL (not by path) so the size we record and the pixels the model sees
        # come from the same decode, independent of cv2's EXIF handling.
        with Image.open(image_path) as im:
            rgb = im.convert("RGB")
        width, height = rgb.size
        bgr = np.ascontiguousarray(np.asarray(rgb)[:, :, ::-1])
        fallbacks: list[str] = []
        res = self._ocr.predict(bgr)[0]
        predicted = int(res["doc_preprocessor_res"]["angle"])
        if predicted > 0:
            # The classifier flags 25/951 pages as rotated, mostly sparse continuation pages.
            # 22 are upright (an unrotated pass finds >= 86% as many lines); 2 are sideways (an
            # unrotated pass finds <= 20%). Trust the unrotated pass only when it keeps up
            # (per-page counts in reports/ocr.md).
            upright = self._ocr.predict(bgr, use_doc_orientation_classify=False)[0]
            if len(upright["rec_texts"]) >= ROTATION_KEEP_RATIO * len(res["rec_texts"]):
                res = upright
                fallbacks.append("orientation_override")
            else:
                fallbacks.append("orientation_kept")
        if not res["rec_texts"]:
            # PP-OCRv5 server detection returns nothing on a few blurred scans at native size;
            # a 1280px max-side limit recovers them (verified on the 3 known cases).
            res = self._ocr.predict(
                bgr,
                use_doc_orientation_classify=False,
                text_det_limit_type="max",
                text_det_limit_side_len=1280,
            )[0]
            fallbacks.append("det_limit_max1280")
        # angle is absent from the result when orientation is switched off per call.
        angle = int(res["doc_preprocessor_res"].get("angle", -1))
        unrotated = "orientation_override" in fallbacks or "det_limit_max1280" in fallbacks
        rotation = 0 if unrotated else (None if angle < 0 else angle)
        items: list[OcrItem] = []
        for text, score, poly in zip(
            res["rec_texts"], res["rec_scores"], res["rec_polys"], strict=True
        ):
            box = unrotate_polygon(np.asarray(poly).tolist(), rotation or 0, width, height)
            items.append(OcrItem(text, box, float(score)))
        words = [w for it in items for w in split_line_to_words(it)]
        return PageOcr(
            page=Path(image_path).name,
            width=width,
            height=height,
            engine=self.name,
            granularity="line",
            rotation_deg=rotation,
            items=items,
            words=words,
            seconds=time.perf_counter() - t0,
            orientation_predicted_deg=None if predicted < 0 else predicted,
            fallbacks=fallbacks,
        )


# ---------------------------------------------------------------------------------------------
# Cache driver
# ---------------------------------------------------------------------------------------------


def list_pages(data_root: Path, split: str) -> list[Path]:
    """Sorted page images of a split."""
    folder = Path(data_root) / split / "images"
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON via temp file + rename so a killed run never leaves a half-written page."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def run_ocr(
    engine: OcrEngine,
    splits: Sequence[str],
    data_root: Path | None = None,
    cache_root: Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """OCR every page of `splits` that is not cached yet; write per-page JSON and manifest.

    Per-page seconds and failures accumulate across resumed runs (the manifest is rebuilt from
    the cached page files plus this run's failures), so the manifest always describes the cache.
    `data_root` / `cache_root` default to the `shipdoc.paths` locations.
    """
    data_root = paths.data_dir() if data_root is None else data_root
    cache_root = paths.ocr_cache_dir() if cache_root is None else cache_root
    started = time.time()
    failures: list[dict[str, str]] = []
    new_pages = 0
    consecutive = 0
    aborted = False
    for split in splits:
        if aborted:
            break
        pages = list_pages(data_root, split)
        if limit is not None:
            pages = pages[:limit]
        for n, img in enumerate(pages, 1):
            out = cache_path(cache_root, engine.name, split, img.name)
            if out.exists():
                continue
            try:
                page = engine.run(img)
            except Exception as exc:  # noqa: BLE001  # one bad page must not kill a 2h run
                failures.append({"split": split, "page": img.name, "error": repr(exc)})
                consecutive += 1
                print(f"[{split} {n}/{len(pages)}] FAIL {img.name}: {exc!r}", flush=True)
                if consecutive >= MAX_CONSECUTIVE_FAILURES:
                    aborted = True
                    break
                continue
            consecutive = 0
            write_json_atomic(out, page.to_dict())
            new_pages += 1
            print(
                f"[{split} {n}/{len(pages)}] {img.name} {page.seconds:.1f}s "
                f"{len(page.items)} lines rot={page.rotation_deg}",
                flush=True,
            )
    manifest = build_manifest(engine, splits, data_root, cache_root, failures)
    manifest["this_run"] = {
        "new_pages": new_pages,
        "wall_clock_seconds": time.time() - started,
        "started_unix": started,
        "aborted_after_consecutive_failures": aborted,
    }
    write_json_atomic(Path(cache_root) / engine.name / "manifest.json", manifest)
    return manifest


def build_manifest(
    engine: OcrEngine,
    splits: Sequence[str],
    data_root: Path,
    cache_root: Path,
    failures: list[dict[str, str]],
) -> dict[str, Any]:
    """Assemble the manifest from the page files currently in the cache."""
    per_page: dict[str, dict[str, float]] = {}
    expected = 0
    for split in splits:
        pages = list_pages(data_root, split)
        expected += len(pages)
        per_page[split] = {}
        for img in pages:
            p = cache_path(cache_root, engine.name, split, img.name)
            if p.exists():
                per_page[split][img.name] = json.loads(p.read_text(encoding="utf-8"))["seconds"]
    secs = [s for d in per_page.values() for s in d.values()]
    cached = len(secs)
    return {
        **engine.describe(),
        "splits": list(splits),
        "pages_expected": expected,
        "pages_cached": cached,
        "seconds_per_page": per_page,
        "seconds_mean": statistics.fmean(secs) if secs else None,
        "seconds_p95": percentile(secs, 95) if secs else None,
        "failures": failures,
    }


# ---------------------------------------------------------------------------------------------
# Cache readers: reading-order page loader
# ---------------------------------------------------------------------------------------------

DEFAULT_ENGINE = "paddleocr"
# An item joins the current line when its centre is within this fraction of the median item
# height from the previous item's centre.
LINE_JOIN_FRACTION = 0.5


@dataclass
class ReadItem:
    """One non-empty OCR item in reading order, with an axis-aligned box."""

    text: str
    box: tuple[float, float, float, float]  # x0, y0, x1, y1
    conf: float
    line_idx: int


def load_page(
    split: str,
    page_stem: str,
    cache_root: Path | None = None,
    engine: str = DEFAULT_ENGINE,
) -> PageOcr:
    """Load the cached `PageOcr` of ``<page_stem>`` (e.g. ``dev_0003_p2``) in `split`."""
    root = paths.ocr_cache_dir() if cache_root is None else Path(cache_root)
    path = cache_path(root, engine, split, page_stem)
    return PageOcr.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _aabb(box: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return min(xs), min(ys), max(xs), max(ys)


def page_items(page: PageOcr) -> list[ReadItem]:
    """Non-empty items in reading order.

    Items are grouped into lines by vertical overlap: walking items by centre-y, one joins the
    current line if its centre is within half the median item height of the previous item's
    centre (chaining, so a slightly slanted line stays together). Lines are ordered by y, items
    within a line by x. Boxes are in original-image pixels, so a page that is physically
    rotated is read in image order, not text order.
    """
    kept = [(it, _aabb(it.box)) for it in page.items if it.text.strip() and it.box]
    if not kept:
        return []
    median_h = statistics.median(b[3] - b[1] for _, b in kept)
    tol = LINE_JOIN_FRACTION * median_h
    kept.sort(key=lambda t: (t[1][1] + t[1][3]) / 2)
    lines: list[list[tuple[OcrItem, tuple[float, float, float, float]]]] = []
    last_cy = 0.0
    for it, b in kept:
        cy = (b[1] + b[3]) / 2
        if lines and cy - last_cy <= tol:
            lines[-1].append((it, b))
        else:
            lines.append([(it, b)])
        last_cy = cy
    out: list[ReadItem] = []
    for idx, ln in enumerate(lines):  # already ascending in y: items were walked by centre-y
        for it, b in sorted(ln, key=lambda t: t[1][0]):
            out.append(ReadItem(it.text, b, it.confidence, idx))
    return out


def page_text(page: PageOcr) -> str:
    """Page text: lines joined with newline, items within a line joined with a space."""
    lines: dict[int, list[str]] = {}
    for it in page_items(page):
        lines.setdefault(it.line_idx, []).append(it.text)
    return "\n".join(" ".join(parts) for _, parts in sorted(lines.items()))


def doc_pages(
    doc_id: str, cache_root: Path | None = None, engine: str = DEFAULT_ENGINE
) -> list[PageOcr]:
    """All cached pages of a doc (``<split>_<n>``), in page order. Empty list if none cached."""
    root = paths.ocr_cache_dir() if cache_root is None else Path(cache_root)
    split = doc_id.split("_", 1)[0]
    files = (root / engine / split).glob(f"{doc_id}_p*.json")

    def page_no(f: Path) -> int:
        return int(f.stem.rsplit("_p", 1)[1])

    return [load_page(split, f.stem, root, engine) for f in sorted(files, key=page_no)]


def doc_text(doc_id: str, cache_root: Path | None = None, engine: str = DEFAULT_ENGINE) -> str:
    """Text of every page of a doc, pages separated by a blank line."""
    return "\n\n".join(page_text(p) for p in doc_pages(doc_id, cache_root, engine))
