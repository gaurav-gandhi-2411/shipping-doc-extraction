"""Write meta/train.json and meta/dev.json (slice tags per doc).

Run: uv run python scripts/make_meta.py
"""

from __future__ import annotations

import json
from collections import Counter

from shipdoc import meta as M

ROOT = M.ROOT


def main() -> None:
    groups = json.loads((ROOT / "meta" / "supplier_groups.json").read_text(encoding="utf-8"))
    labels = {s: M.load_labels(s) for s in ("train", "dev")}
    pooled = [d for s in labels for d in labels[s]]
    np_fields = M.not_printed_fields(pooled)
    print(
        "not-printed header fields (null = absent line):",
        {t: sorted(f) for t, f in np_fields.items()},
    )
    n_mixed = sum(M.is_mixed_scan(d) for d in pooled)
    for split, docs in labels.items():
        rows = [M.tag_doc(d, groups[d["doc_id"]], np_fields) for d in docs]
        if n_mixed:  # only add the tag if mixed docs exist (none in train+dev as of recon)
            for r, d in zip(rows, docs, strict=True):
                r["mixed_scan"] = M.is_mixed_scan(d)
        (ROOT / "meta" / f"{split}.json").write_text(
            json.dumps(rows, indent=1) + "\n", encoding="utf-8"
        )
        c = Counter(t for r in rows for t in M.BOOL_TAGS if r[t])
        mixed = sum(M.is_mixed_scan(d) for d in docs)
        counts = " ".join(f"{t}={c[t]}" for t in M.BOOL_TAGS)
        print(f"{split}: n={len(rows)} {counts} mixed_scan={mixed}")
    print(f"mixed_scan tag {'added' if n_mixed else 'omitted (0 docs)'}")


if __name__ == "__main__":
    main()
