"""Label-free layout clusters from an OCR keyword + position signature (spec Phase 3.3 / 6.4).

The signature of a page is, for each keyword family of ``KEYWORDS``: whether an OCR item of the
page contains it and, if so, the normalised centre ``(x / width, y / height)`` of the first such
item in reading order. No gold label and no image pixels are used, so the same function runs on
test pages. Clusters are fit on train+dev OCR only (never test); a new page is assigned to its
nearest centre.

Used for per-cluster date-format inference (`shipdoc.normalize.infer_date_order`): all-numeric
dates of one cluster are pooled, and one date with a number above 12 settles the day/month order
for the whole cluster.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from shipdoc.ocr import PageOcr, page_items

#: Keyword families whose presence and position characterise a layout (case-insensitive).
KEYWORDS: dict[str, str] = {
    "invoice": r"\binvoice\b",
    "date": r"\bdate\b",
    "total": r"\btotal\b",
    "bill_to": r"\bbill(ed)?\s*to\b",
    "ship_to": r"\bship(ped)?\s*to\b",
    "sold_to": r"\bsold\s*to\b",
    "supplier": r"\b(supplier|seller|vendor|from)\b",
    "customer": r"\b(buyer|customer|consignee)\b",
    "currency": r"\bcurrency\b",
    "awb": r"\b(awb|air\s*waybill|waybill)\b",
    "po": r"\b(po|p\.o\.|purchase\s*order)\b",
    "part": r"\b(part|p/n)\b",
    "description": r"\bdescription\b",
    "qty": r"\b(qty|quantity)\b",
    "unit_price": r"\bunit\s*price\b",
    "amount": r"\bamount\b",
    "terms": r"\bterms\b",
    "phone": r"\b(tel|phone|fax)\b",
    "tax": r"\b(vat|tax|gst)\b",
    "page": r"\bpage\b",
}
_COMPILED = {k: re.compile(p, re.IGNORECASE) for k, p in KEYWORDS.items()}
SEED = 42
K_RANGE = range(8, 31)  # label-free k search; the recon supplier count (18) is not used


def layout_signature(page: PageOcr) -> np.ndarray:
    """Vector of ``3 * len(KEYWORDS)`` floats: (present, x/width, y/height) per keyword family."""
    w, h = float(page.width) or 1.0, float(page.height) or 1.0
    sig = np.zeros(3 * len(KEYWORDS), dtype=float)
    pending = dict(_COMPILED)
    for it in page_items(page):
        for i, (name, pat) in enumerate(_COMPILED.items()):
            if name in pending and pat.search(it.text):
                sig[3 * i : 3 * i + 3] = (
                    1.0,
                    (it.box[0] + it.box[2]) / 2 / w,
                    (it.box[1] + it.box[3]) / 2 / h,
                )
                del pending[name]
    return sig


@dataclass
class LayoutClusters:
    """Fitted KMeans over page signatures; `predict` maps new signatures to cluster ids."""

    model: KMeans
    k: int
    silhouette: float

    @classmethod
    def fit(cls, sigs: Sequence[np.ndarray], k: int | None = None) -> LayoutClusters:
        """Fit on `sigs`; with ``k=None`` pick k in `K_RANGE` by silhouette (label-free)."""
        x = np.vstack(sigs)
        best: tuple[float, int] | None = None
        ks = [k] if k else [n for n in K_RANGE if n < len(x)]
        for n in ks:
            labels = KMeans(n_clusters=n, random_state=SEED, n_init=10).fit_predict(x)
            sil = float(silhouette_score(x, labels)) if len(set(labels)) > 1 else -1.0
            if best is None or sil > best[0]:
                best = (sil, n)
        assert best is not None
        model = KMeans(n_clusters=best[1], random_state=SEED, n_init=10).fit(x)
        return cls(model, best[1], best[0])

    def predict(self, sigs: Sequence[np.ndarray]) -> list[int]:
        """Cluster id of each signature."""
        return [int(c) for c in self.model.predict(np.vstack(sigs))]
