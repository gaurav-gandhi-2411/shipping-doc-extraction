"""Write splits/folds.json: 3-fold, seed 42, grouped by supplier_group, over train+dev.

Run: uv run python scripts/make_folds.py   (needs meta/*.json from make_meta.py)
"""

from __future__ import annotations

import json

from shipdoc import meta as M

ROOT = M.ROOT


def main() -> None:
    rows = []
    for s in ("train", "dev"):
        rows += json.loads((ROOT / "meta" / f"{s}.json").read_text(encoding="utf-8"))
    out = M.make_folds(rows)  # asserts group-disjointness and both doc types per fold
    by = {r["doc_id"]: r for r in rows}
    (ROOT / "splits").mkdir(exist_ok=True)
    (ROOT / "splits" / "folds.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print("fold docs invoices waybills groups scanned multipage")
    for f in out["folds"]:
        m = [by[d] for d in f["val_doc_ids"]]
        wb = sum(x["waybill"] for x in m)
        print(
            f"{f['fold']:>4} {len(m):>4} {len(m) - wb:>8} {wb:>8} {len(f['val_groups']):>6} "
            f"{sum(x['scanned'] for x in m):>7} {sum(x['multipage'] for x in m):>9}"
        )


if __name__ == "__main__":
    main()
