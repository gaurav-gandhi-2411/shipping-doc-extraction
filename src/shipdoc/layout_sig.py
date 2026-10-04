"""Test-set layout signature, seen/unseen calibration and a leak-proof fit path (spec Phase 6.4).

Builds on `shipdoc.cluster` (unchanged). Everything here is label-free for the test split: a test
page is only ever turned into a signature vector, assigned to a centre fitted on train+dev, and
reported as (doc id, cluster id, distance, flag). No OCR text leaves this module.

Fail-closed guards: `assert_fit_ids` rejects any doc id of the test split, and `write_assignments`
accepts only typed id/int/float/bool/doc-type fields (never text).
"""

from __future__ import annotations

import csv
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from shipdoc.cluster import K_RANGE, KEYWORDS, SEED, layout_signature
from shipdoc.ocr import PageOcr, page_items

DOC_ID_RE = re.compile(r"^(train|dev|test)_\d{4}$")
DOC_TYPES = ("invoice", "waybill")
SigFn = Callable[[PageOcr], np.ndarray]


# ---------------------------------------------------------------------------------------------
# Signatures (all label-free; the variants are compared on train+dev only)
# ---------------------------------------------------------------------------------------------


def sig_base(page: PageOcr) -> np.ndarray:
    """The unchanged `cluster.layout_signature` (20 keyword families x (present, x, y))."""
    return layout_signature(page)


def sig_coarse(page: PageOcr) -> np.ndarray:
    """Base signature with positions snapped to a 0.1 grid (robust to small OCR box jitter)."""
    s = layout_signature(page).reshape(len(KEYWORDS), 3).copy()
    s[:, 1:] = np.round(s[:, 1:] * 10.0) / 10.0
    return s.reshape(-1)


def geometry_features(page: PageOcr) -> np.ndarray:
    """Six page-geometry features in [0, 1]: aspect, log line count, text block extent.

    ``aspect = width/height / 2`` (clipped), ``log1p(#lines) / log(201)`` (clipped), 5th percentile
    of left edges, 95th percentile of right edges, top of the first and bottom of the last item
    (all normalised by page size). Constants are fixed, never fitted, so test pages need no stats.
    """
    w, h = float(page.width) or 1.0, float(page.height) or 1.0
    items = page_items(page)
    if not items:
        return np.zeros(6)
    n_lines = len({it.line_idx for it in items})
    x0 = np.array([it.box[0] for it in items]) / w
    x1 = np.array([it.box[2] for it in items]) / w
    y0 = np.array([it.box[1] for it in items]) / h
    y1 = np.array([it.box[3] for it in items]) / h
    return np.array(
        [
            min(w / h / 2.0, 1.0),
            min(math.log1p(n_lines) / math.log(201.0), 1.0),
            float(np.percentile(x0, 5)),
            float(np.percentile(x1, 95)),
            float(y0.min()),
            float(y1.max()),
        ]
    )


def sig_geom(page: PageOcr) -> np.ndarray:
    """Base signature plus `geometry_features` (66 floats)."""
    return np.concatenate([layout_signature(page), geometry_features(page)])


SIGNATURES: dict[str, SigFn] = {"base": sig_base, "coarse": sig_coarse, "geom": sig_geom}


def invoice_keyword_present(page: PageOcr) -> bool:
    """Label-free doc-type cue: page 1 carries the 'invoice' keyword family => invoice.

    Chosen on train+dev (a depth-1 decision tree over the 20 presence flags separates invoice
    from waybill with 5-fold CV accuracy 1.0; the 'awb' family does not, invoices mention AWBs).
    """
    i = list(KEYWORDS).index("invoice")
    return bool(layout_signature(page)[3 * i] > 0.5)


# ---------------------------------------------------------------------------------------------
# Leak guard
# ---------------------------------------------------------------------------------------------


def assert_fit_ids(doc_ids: Sequence[str]) -> None:
    """Fail closed: refuse to fit on anything that is not a well-formed train/dev doc id."""
    for d in doc_ids:
        m = DOC_ID_RE.match(d)
        if m is None:
            raise ValueError(f"refusing to fit: malformed doc id {d!r}")
        if m.group(1) == "test":
            raise ValueError(f"refusing to fit on test split doc {d!r}")


# ---------------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------------


def ari(labels: Sequence[str], clusters: Sequence[int]) -> float:
    """Adjusted Rand index of a clustering against reference labels."""
    return float(adjusted_rand_score(list(labels), list(clusters)))


def purity(labels: Sequence[str], clusters: Sequence[int]) -> float:
    """Share of docs that carry the majority label of their cluster."""
    by: dict[int, dict[str, int]] = {}
    for lab, c in zip(labels, clusters, strict=True):
        by.setdefault(int(c), {}).setdefault(lab, 0)
        by[int(c)][lab] += 1
    return sum(max(v.values()) for v in by.values()) / len(labels)


def bootstrap_ari_ci(
    labels: Sequence[str], clusters: Sequence[int], n_boot: int = 1000, seed: int = SEED
) -> tuple[float, float]:
    """Approximate 95% interval of ARI over docs, clusters held fixed (half-sample subsampling).

    A with-replacement bootstrap duplicates docs, which agree with themselves and shift ARI up
    (measured: the point estimate fell outside its own percentile interval). Instead draw `n_boot`
    half-samples WITHOUT replacement and rescale their deviation from the full-sample ARI by
    sqrt(m/n) (subsampling theory). It is a rough interval and ignores refit variance.
    """
    rng = np.random.default_rng(seed)
    lab, cl = np.asarray(labels), np.asarray(clusters)
    n = len(lab)
    m = n // 2
    vals = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=m, replace=False)
        vals.append(adjusted_rand_score(lab[idx], cl[idx]))
    q_lo, q_hi = np.percentile(vals, [2.5, 97.5])
    theta = adjusted_rand_score(lab, cl)
    scale = math.sqrt(m / n)
    return float(theta - (theta - q_lo) * scale), float(theta + (q_hi - theta) * scale)


def wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (q in [0, 100])."""
    return float(np.percentile(np.asarray(values, dtype=float), q))


# ---------------------------------------------------------------------------------------------
# Clustering, distances, calibration
# ---------------------------------------------------------------------------------------------


@dataclass
class Fit:
    """KMeans over reference signatures with the silhouette used to pick k."""

    model: KMeans
    k: int
    silhouette: float

    def labels(self) -> list[int]:
        """Training-set cluster ids."""
        return [int(c) for c in self.model.labels_]

    def distance(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(nearest centre id, Euclidean distance to it) per row of `x`."""
        d = self.model.transform(np.atleast_2d(x))
        return d.argmin(axis=1), d.min(axis=1)


def fit_reference(
    doc_ids: Sequence[str], x: np.ndarray, k: int | None = None, k_range: Sequence[int] = K_RANGE
) -> Fit:
    """Fit KMeans (seed 42) on train/dev signatures; k by silhouette over `k_range` if None.

    Raises ValueError for any test-split or malformed doc id (see `assert_fit_ids`).
    """
    assert_fit_ids(doc_ids)
    if len(doc_ids) != len(x):
        raise ValueError("doc_ids and signature rows differ in length")
    best: tuple[float, int] | None = None
    ks = [k] if k else [n for n in k_range if n < len(x)]
    for n in ks:
        labels = KMeans(n_clusters=n, random_state=SEED, n_init=10).fit_predict(x)
        sil = float(silhouette_score(x, labels)) if len(set(labels)) > 1 else -1.0
        if best is None or sil > best[0]:
            best = (sil, n)
    if best is None:
        raise ValueError("no admissible k")
    model = KMeans(n_clusters=best[1], random_state=SEED, n_init=10).fit(x)
    return Fit(model, best[1], best[0])


def seen_distances(doc_ids: Sequence[str], x: np.ndarray, k: int, n_folds: int = 5) -> np.ndarray:
    """Out-of-sample distance to the nearest centre of docs whose layout group IS in the fit.

    Random `n_folds`-fold split of the docs (seed 42): fit on the other folds, measure the held-out
    fold. Out-of-sample on purpose: in-sample distances are biased low and would shrink the
    threshold.
    """
    rng = np.random.default_rng(SEED)
    fold = rng.permutation(len(x)) % n_folds
    out = np.zeros(len(x))
    for f in range(n_folds):
        tr, te = fold != f, fold == f
        fit = fit_reference([d for d, m in zip(doc_ids, tr, strict=True) if m], x[tr], k=k)
        out[te] = fit.distance(x[te])[1]
    return out


def logo_distances(
    doc_ids: Sequence[str], groups: Sequence[str], x: np.ndarray, k: int
) -> np.ndarray:
    """Leave-one-group-out distance: each doc to the nearest centre fitted WITHOUT its group."""
    g = np.asarray(groups)
    out = np.zeros(len(x))
    for grp in sorted(set(groups)):
        te = g == grp
        fit = fit_reference([d for d, m in zip(doc_ids, ~te, strict=True) if m], x[~te], k=k)
        out[te] = fit.distance(x[te])[1]
    return out


@dataclass
class Calibration:
    """Threshold and its label-free-calibrated operating characteristics."""

    threshold: float
    percentile: float
    fpr: float  # seen docs flagged unseen (out-of-sample)
    tpr: float  # held-out-group docs flagged unseen (LOGO)


def calibrate_threshold(seen: np.ndarray, logo: np.ndarray, q: float = 95.0) -> Calibration:
    """Threshold = q-th percentile of the seen distances, fixed BEFORE test data is looked at."""
    t = percentile(seen, q)
    return Calibration(t, q, float(np.mean(seen > t)), float(np.mean(logo > t)))


def corrected_share(flagged: float, cal: Calibration) -> float | None:
    """Rogan-Gladen correction of a flagged share for the proxy's FPR/TPR (None if TPR <= FPR)."""
    if cal.tpr <= cal.fpr:
        return None
    return min(1.0, max(0.0, (flagged - cal.fpr) / (cal.tpr - cal.fpr)))


# ---------------------------------------------------------------------------------------------
# Writer that can only carry ids and numbers
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AssignmentRow:
    """One test doc's label-free layout assignment. No free-text field exists by construction."""

    doc_id: str
    doc_type: str
    cluster: int
    distance: float
    seen: bool


def write_assignments(rows: Sequence[AssignmentRow], path: Path) -> None:
    """Write rows to CSV; validates every field's type/shape so OCR text cannot slip through."""
    for r in rows:
        if DOC_ID_RE.match(r.doc_id) is None:
            raise ValueError(f"not a doc id: {r.doc_id!r}")
        if r.doc_type not in DOC_TYPES:
            raise ValueError(f"unknown doc_type {r.doc_type!r}")
        if not (isinstance(r.cluster, int) and isinstance(r.seen, bool)):
            raise TypeError("cluster must be int and seen bool")
        if not isinstance(r.distance, float):
            raise TypeError("distance must be float")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["doc_id", "doc_type", "cluster", "distance", "seen"])
        for r in rows:
            w.writerow([r.doc_id, r.doc_type, r.cluster, f"{r.distance:.6f}", int(r.seen)])
