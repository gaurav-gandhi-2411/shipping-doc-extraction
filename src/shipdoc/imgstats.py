"""Summary image statistics used to compare scanned pages with digital pages (numpy + PIL only).

All functions take a 2-D ``uint8`` grayscale array (rows x cols). `skew` is estimated from ink
projection profiles on a downsampled copy; the OCR-polygon estimate lives in `ocr_line_skew`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from PIL import Image

STAT_NAMES = ("mean", "std", "bg_p90", "lap_var", "noise_mad", "abs_skew_deg", "blockiness")
# Projection-profile search: coarse then fine angle grid (degrees). Pages are never skewed by more
# than a few degrees, so a +-6 degree window is enough and keeps the cost to ~60 small rotations.
SKEW_LIMIT_DEG = 6.0
SKEW_COARSE_STEP = 0.5
SKEW_FINE_STEP = 0.1
SKEW_DOWNSCALE = 0.3
# MAD -> sigma for Gaussian data.
MAD_TO_SIGMA = 1.4826
JPEG_BLOCK = 8


def to_gray(image: Image.Image) -> np.ndarray:
    """``uint8`` grayscale array of a PIL image."""
    return np.asarray(image.convert("L"), dtype=np.uint8)


def laplacian_variance(a: np.ndarray) -> float:
    """Variance of the 4-neighbour Laplacian (higher means sharper)."""
    f = a.astype(np.float32)
    lap = f[:-2, 1:-1] + f[2:, 1:-1] + f[1:-1, :-2] + f[1:-1, 2:] - 4.0 * f[1:-1, 1:-1]
    return float(lap.var())


def noise_mad(a: np.ndarray) -> float:
    """Noise sigma estimate: MAD of the residual after a 3x3 box blur (robust to sparse edges)."""
    f = a.astype(np.float32)
    box = sum(f[i : f.shape[0] - 2 + i, j : f.shape[1] - 2 + j] for i in range(3) for j in range(3))
    res = f[1:-1, 1:-1] - box / 9.0
    return float(MAD_TO_SIGMA * np.median(np.abs(res - np.median(res))))


def blockiness(a: np.ndarray) -> float:
    """JPEG 8x8 grid artefact: mean |step| across block boundaries minus elsewhere, both axes."""
    f = a.astype(np.float32)
    out = []
    for axis in (0, 1):
        d = np.abs(np.diff(f, axis=axis))
        idx = np.arange(d.shape[axis])
        on = (idx % JPEG_BLOCK) == JPEG_BLOCK - 1
        dm = d.mean(axis=1 - axis)  # mean over the other axis -> one value per step position
        out.append(float(dm[on].mean() - dm[~on].mean()))
    return float(np.mean(out))


def _row_profile_score(ink: Image.Image, angle: float) -> float:
    rot = ink.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)
    prof = np.asarray(rot, dtype=np.float32).sum(axis=1)
    return float(prof.var())


def projection_skew(a: np.ndarray) -> float:
    """Skew in degrees (positive = counter-clockwise rotation needed to straighten) by the
    projection-profile method: the angle maximising the variance of the ink row sums."""
    bg = float(np.percentile(a, 90))
    img = Image.fromarray(a)
    w, h = img.size
    img = img.resize((max(1, int(w * SKEW_DOWNSCALE)), max(1, int(h * SKEW_DOWNSCALE))))
    ink = Image.fromarray(((np.asarray(img, dtype=np.float32) < bg - 60.0) * 255).astype(np.uint8))
    coarse = np.arange(-SKEW_LIMIT_DEG, SKEW_LIMIT_DEG + 1e-9, SKEW_COARSE_STEP)
    best = max(coarse, key=lambda t: _row_profile_score(ink, float(t)))
    fine = np.arange(best - SKEW_COARSE_STEP, best + SKEW_COARSE_STEP + 1e-9, SKEW_FINE_STEP)
    return float(max(fine, key=lambda t: _row_profile_score(ink, float(t))))


def ocr_line_skew(polygons: Sequence[Sequence[Sequence[float]]], min_width: float = 80.0) -> float:
    """Median angle (degrees) of the top edge of OCR line polygons wider than `min_width` px.

    Sign convention: positive when the text baseline rises to the right in image coordinates
    (y decreasing), i.e. the same sense as `projection_skew`. NaN if no polygon qualifies.
    """
    angles = []
    for poly in polygons:
        (x0, y0), (x1, y1) = poly[0], poly[1]
        if abs(x1 - x0) >= min_width:
            angles.append(math.degrees(math.atan2(-(y1 - y0), x1 - x0)))
    return float(np.median(angles)) if angles else float("nan")


def image_stats(a: np.ndarray) -> dict[str, Any]:
    """The `STAT_NAMES` statistics of one grayscale page (skew reported as an absolute value)."""
    return {
        "mean": float(a.mean()),
        "std": float(a.std()),
        "bg_p90": float(np.percentile(a, 90)),
        "lap_var": laplacian_variance(a),
        "noise_mad": noise_mad(a),
        "abs_skew_deg": abs(projection_skew(a)),
        "blockiness": blockiness(a),
    }


def ks_distance(x: Sequence[float], y: Sequence[float]) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (max ECDF gap), no p-value."""
    xs = np.sort(np.asarray(x, dtype=np.float64))
    ys = np.sort(np.asarray(y, dtype=np.float64))
    grid = np.concatenate([xs, ys])
    cx = np.searchsorted(xs, grid, side="right") / len(xs)
    cy = np.searchsorted(ys, grid, side="right") / len(ys)
    return float(np.max(np.abs(cx - cy)))
