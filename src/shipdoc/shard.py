"""Document-level sharding of a run across Colab tabs (stdlib only; the notebook loads it by path).

``SHARD = "i/K"`` (0-based i). ``"0/1"`` is the unsharded run. Assignment (`assign_shards`),
deterministic and balanced by page count: sort the documents by ``(n_pages desc, doc_id)``, give
each to the shard with the fewest pages so far (ties: the lowest shard index), then list every
shard's documents in the INPUT order. Greedy longest-first keeps the page totals of the shards
within one document's page count of each other for this corpus (1-5 pages per document).

Each shard writes to its own folder ``<run_id>_shard<i>of<K>``; `shipdoc.shardmerge` combines them.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

_SHARD = re.compile(r"^(\d+)/(\d+)$")


def parse_shard(text: str) -> tuple[int, int]:
    """``"i/K"`` -> ``(i, K)``; raises ValueError unless ``0 <= i < K``."""
    m = _SHARD.match(str(text).strip())
    if not m:
        raise ValueError(f"bad shard {text!r}: expected 'i/K' such as '0/2'")
    i, k = int(m.group(1)), int(m.group(2))
    if k < 1 or not 0 <= i < k:
        raise ValueError(f"bad shard {text!r}: need 0 <= i < K and K >= 1")
    return i, k


def shard_run_id(run_id: str, shard: str) -> str:
    """Folder name of one shard: the base id for ``0/1``, else ``<run_id>_shard<i>of<K>``."""
    i, k = parse_shard(shard)
    return run_id if k == 1 else f"{run_id}_shard{i}of{k}"


def assign_shards(pages: Sequence[tuple[str, int]], k: int) -> list[list[str]]:
    """Doc ids per shard (input order inside each shard) from ``(doc_id, n_pages)`` pairs."""
    if k < 1:
        raise ValueError(f"K must be >= 1, got {k}")
    ids = [d for d, _ in pages]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate doc_id in the shard input")
    load = [0] * k
    owner: dict[str, int] = {}
    for doc, n in sorted(pages, key=lambda p: (-p[1], p[0])):
        s = min(range(k), key=lambda j: (load[j], j))
        owner[doc] = s
        load[s] += n
    return [[d for d in ids if owner[d] == s] for s in range(k)]
