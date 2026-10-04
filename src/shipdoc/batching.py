"""Page batching for the batched runner: expected-length key, batch plan, doc windows, fallback.

Stdlib only. Everything here is a pure function of page-order / image-header / file-size
features that exist BEFORE generation (no gold, no OCR, no model output), so a plan can be
reproduced and tested.

Why sort at all: a batch decodes until its LONGEST row stops (finished rows are padded, not
dropped) and its prompt is padded to the LONGEST row. Putting pages of similar expected output
length and similar visual-token count in the same batch cuts both wastes.

Expected-length key (`expected_length_key`, ascending = scheduled first):

1. ``-multipage``: pages of multi-page documents first. On the corpus layout (recon: identity
   header on page 1, totals on the last page, table rows split over the pages) a page of a
   multi-page document carries rows PLUS a header or totals block, so it is the longest kind of
   page. Measured on the 671 train+dev pages with the mock layout (gold JSON characters; scratch
   script, not a model measurement): first 1.78k, last 1.65k, middle 1.46k, single 0.95k mean.
2. ``-n_pages``: more pages in the document = longer table.
3. ``extension``: sizes are only comparable within one extension (``.jpg`` = scanned photo,
   ``.png`` = digital render in this corpus, see `shipdoc.meta`).
4. ``-file_bytes``: the image FILE size. For the 212 single-page ``.png`` pages the Spearman
   correlation of file size with the gold page length is 0.973 (a denser page has more text,
   more rows, a bigger PNG); for the 136 single-page ``.jpg`` pages it is only 0.138 (scan noise
   dominates). Image-derived and known before generation; no label is read.
5. ``role``: first (0) before last (1) before middle (2) before single (3).
6. ``-visual_tokens``: estimated visual tokens from the image header, ``min(w*h, max_pixels) /
   1024`` (32x32 px per visual token, the model's cap): the prompt length, i.e. left-padding
   waste in prefill.
7. ``order``: the global input position (doc order, then page), so the plan is deterministic.

Effect, measured on the same proxy (671 pages, windows as below, decode-step inflation = sum of
batch max length x batch size / sum of lengths; 1.0 = no straggler waste): batch 2 / 4 / 8 gives
1.144 / 1.271 / 1.385 with this key and 8-batch windows, against 1.211 / 1.397 / 1.567 for the
same windows in input order (scratch script, 2026-10-02). The proxy is the gold length, not the
model output length, so this bounds the idea; it is not a measurement of the model. The bench
measures pages per hour end to end.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

#: Visual tokens cost 32 x 32 pixels each (patch 16 x spatial merge 2).
PIXELS_PER_VISUAL_TOKEN = 32 * 32
#: A window holds whole documents until it has at least this many batches' worth of pages.
WINDOW_BATCHES = 8
ROLE_RANK = {"first": 0, "last": 1, "middle": 2, "single": 3}


@dataclass(frozen=True)
class PageJob:
    """One page to run: where it sits in the input order and what is known without the model."""

    order: int  # global position in input order (doc order, then page order)
    doc_id: str
    page_idx: int  # 0-based
    n_pages: int  # pages of the document
    path: Path
    width: int = 0  # image header size in pixels; 0 = unknown
    height: int = 0
    size: int = 0  # image file size in bytes; 0 = unknown

    @property
    def role(self) -> str:
        """``single`` / ``first`` / ``last`` / ``middle`` (by page order only)."""
        if self.n_pages == 1:
            return "single"
        if self.page_idx == 0:
            return "first"
        return "last" if self.page_idx == self.n_pages - 1 else "middle"


def est_visual_tokens(job: PageJob, max_pixels: int) -> int:
    """Estimated visual tokens of a page: pixel area capped at `max_pixels`, 1024 px per token."""
    return min(job.width * job.height, max_pixels) // PIXELS_PER_VISUAL_TOKEN


def expected_length_key(job: PageJob, max_pixels: int) -> tuple[int, int, str, int, int, int, int]:
    """Sort key; see the module docstring for the definition and the reasons."""
    return (
        -int(job.n_pages > 1),
        -job.n_pages,
        job.path.suffix.lower(),
        -job.size,
        ROLE_RANK[job.role],
        -est_visual_tokens(job, max_pixels),
        job.order,
    )


def plan_batches(jobs: Sequence[PageJob], batch_size: int, max_pixels: int) -> list[list[PageJob]]:
    """Pages sorted by `expected_length_key`, cut into consecutive batches of `batch_size`."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    ordered = sorted(jobs, key=lambda j: expected_length_key(j, max_pixels))
    return [ordered[i : i + batch_size] for i in range(0, len(ordered), batch_size)]


def doc_windows(
    doc_ids: Sequence[str], pages_of: Callable[[str], int], batch_size: int
) -> list[list[str]]:
    """Consecutive runs of whole documents, each with >= WINDOW_BATCHES * batch_size pages.

    A window is the unit of sorting: pages are re-ordered inside a window only, so documents are
    still finished (and traced) in input order, window by window, and a kill loses at most one
    window. The last window may be smaller.
    """
    target = WINDOW_BATCHES * batch_size
    windows: list[list[str]] = []
    cur: list[str] = []
    pages = 0
    for d in doc_ids:
        cur.append(d)
        pages += pages_of(d)
        if pages >= target:
            windows.append(cur)
            cur, pages = [], 0
    if cur:
        windows.append(cur)
    return windows


def next_smaller_batch(n: int) -> int:
    """Batch size to retry with after a batch of `n` pages failed: the largest power of two < n."""
    if n < 2:
        raise ValueError("a batch of 1 has no smaller fallback")
    return 1 << ((n - 1).bit_length() - 1)


def split_for_fallback(items: Sequence[PageJob]) -> list[list[PageJob]]:
    """Cut a failed batch into consecutive chunks of `next_smaller_batch(len(items))`."""
    size = next_smaller_batch(len(items))
    return [list(items[i : i + size]) for i in range(0, len(items), size)]
