"""Tests for scan degradation and the image statistics used to fit it (synthetic images)."""

from __future__ import annotations

import io
import math

import numpy as np
import pytest
from PIL import Image

from shipdoc import augment as A
from shipdoc import imgstats as S

PAGE = (600, 400)


def _page_image(mode: str = "L") -> Image.Image:
    """White page with seeded dark vertical stripes standing in for text (high-frequency)."""
    rng = np.random.default_rng(1)
    arr = np.full((PAGE[1], PAGE[0]), 255, np.uint8)
    for y in range(0, PAGE[1], 30):
        cols = rng.integers(0, PAGE[0], size=200)
        arr[y + 2 : y + 22, cols] = 0
    return Image.fromarray(arr).convert(mode)


def test_scan_degrade_is_deterministic_and_keeps_size() -> None:
    p = A.ScanParams(2.0, (0.9, 1.0), 2.0, (0.0, 20.0), (0.5, 1.5), (1.0, 3.0), (30, 70))
    img = _page_image()
    a = A.scan_degrade(img, np.random.default_rng(42), p)
    b = A.scan_degrade(img, np.random.default_rng(42), p)
    c = A.scan_degrade(img, np.random.default_rng(43), p)
    assert A.png_bytes(a) == A.png_bytes(b) != A.png_bytes(c)
    assert a.size == img.size and a.mode == img.mode
    assert A.ScanParams.from_dict(p.to_dict()) == p


def test_scan_degrade_tone_mapping() -> None:
    p = A.ScanParams(0.0, (0.95, 0.95), 1.0, (10.0, 10.0), (0.0, 0.0), (0.0, 0.0), (50, 50))
    src = _page_image()
    out = np.asarray(A.scan_degrade(src, np.random.default_rng(0), p), float)
    assert 215 <= np.percentile(out, 90) <= 250  # paper no longer pure white
    ink = np.asarray(src) == 0
    assert np.median(out[ink]) >= 5  # ink lifted off black (jpeg ringing clips a few pixels)


def test_imgstats_known_values() -> None:
    flat = np.full((64, 64), 200, np.uint8)
    assert S.laplacian_variance(flat) == 0.0 and S.noise_mad(flat) == 0.0
    assert S.blockiness(flat) == 0.0
    noisy = (flat + np.random.default_rng(0).normal(0, 4, flat.shape)).clip(0, 255).astype(np.uint8)
    assert 3.0 < S.noise_mad(noisy) < 5.0  # recovers the injected sigma
    x = np.arange(20.0)
    assert S.ks_distance(x, x) == 0.0
    assert S.ks_distance(x, x + 100) == 1.0
    assert 0.0 < S.ks_distance(x, x + 5) < 1.0


def test_blockiness_detects_jpeg_grid() -> None:
    img = _page_image()
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=20)
    jpg = np.asarray(Image.open(io.BytesIO(buf.getvalue())))
    assert S.blockiness(jpg) > S.blockiness(np.asarray(img)) + 0.2


@pytest.mark.parametrize("angle", [-2.0, 0.0, 1.5])
def test_projection_skew_recovers_rotation(angle: float) -> None:
    arr = np.full((700, 500), 255, np.uint8)
    for y in range(60, 640, 25):
        arr[y : y + 8, 40:460] = 0  # ruled "text lines"
    rot = Image.fromarray(arr).rotate(angle, fillcolor=255, resample=Image.Resampling.BILINEAR)
    est = S.projection_skew(np.asarray(rot))
    assert abs(abs(est) - abs(angle)) < 0.4
    if angle:
        assert est * angle < 0  # the estimate is the correcting rotation, opposite the skew


def test_ocr_line_skew() -> None:
    t = math.tan(math.radians(1.0))
    poly = [[0, 100], [400, 100 - 400 * t], [400, 130], [0, 130]]  # rises to the right
    assert abs(S.ocr_line_skew([poly]) - 1.0) < 1e-6
    assert math.isnan(S.ocr_line_skew([[[0, 0], [10, 0], [10, 5], [0, 5]]]))


def test_committed_params_respect_spec_bounds() -> None:
    p = A.load_scan_params()
    assert 0 < p.rotate_deg <= 3.0  # spec: rotation within +-3 degrees
    assert A.JPEG_QUALITY_BOUNDS[0] <= p.jpeg_quality[0] <= p.jpeg_quality[1] <= 70
    assert 0 < p.paper_gain[0] <= p.paper_gain[1] <= 1.0
    out = A.scan_degrade(_page_image(), np.random.default_rng(42), p)
    assert out.size == _page_image().size
