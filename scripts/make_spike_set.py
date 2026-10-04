"""Write splits/spike40.json (stratified 40 dev docs) and splits/dev100.json (all dev docs).

Selection is a seeded greedy fill: all illegible docs first (dev has only 7; the spec wants >= 5),
then each step adds the doc that most reduces the remaining stratum deficits, with a large bonus
for a supplier group not yet covered, so the 40 docs reach as many supplier groups as possible.
Ties are broken by a seed-42 shuffle, so the output is deterministic.

Run: uv run python scripts/make_spike_set.py   (needs meta/dev.json from make_meta.py)
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
N_SPIKE = 40
STRATA = ("scanned", "multipage", "repeated_parts", "illegible", "waybill")
# Minimum count per stratum. illegible: 7 exist in dev, spec wants >= 5, take them all.
MINIMUMS = {
    "illegible": 7,
    "waybill": 10,
    "scanned": 15,
    "multipage": 10,
    "repeated_parts": 5,
}
GROUP_BONUS = 100.0  # dominates stratum deficits so group coverage is maximised first


def select_spike(
    meta: list[dict[str, Any]],
    n: int = N_SPIKE,
    minimums: dict[str, int] = MINIMUMS,
    seed: int = SEED,
) -> list[str]:
    """Greedy stratified selection of `n` doc_ids from dev meta rows; sorted result."""
    rng = random.Random(seed)
    pool = sorted(meta, key=lambda r: r["doc_id"])
    rng.shuffle(pool)  # tie-break order; sorting first makes the shuffle input-order independent
    chosen: list[dict[str, Any]] = []
    groups: set[str] = set()
    counts = dict.fromkeys(minimums, 0)

    def add(row: dict[str, Any]) -> None:
        pool.remove(row)
        chosen.append(row)
        groups.add(row["supplier_group"])
        for k in counts:
            counts[k] += bool(row[k])

    for row in [r for r in pool if r["illegible"]][: minimums.get("illegible", 0)]:
        add(row)
    while len(chosen) < n:

        def gain(r: dict[str, Any]) -> float:
            g = GROUP_BONUS if r["supplier_group"] not in groups else 0.0
            return g + sum(
                (counts[k] < minimums[k]) * (minimums[k] - counts[k]) for k in counts if r[k]
            )

        add(max(pool, key=gain))  # max() keeps the first (shuffled) element on ties
    return sorted(r["doc_id"] for r in chosen)


def composition(meta: list[dict[str, Any]], ids: list[str]) -> str:
    """Markdown composition table of the chosen docs (counts vs dev totals)."""
    by = {r["doc_id"]: r for r in meta}
    sel = [by[i] for i in ids]
    lines = ["| stratum | spike40 | dev |", "|---|---|---|"]
    for k in STRATA:
        lines.append(f"| {k} | {sum(bool(r[k]) for r in sel)} | {sum(bool(r[k]) for r in meta)} |")
    lines.append(
        f"| supplier groups | {len({r['supplier_group'] for r in sel})} "
        f"| {len({r['supplier_group'] for r in meta})} |"
    )
    lines.append(f"| docs | {len(sel)} | {len(meta)} |")
    return "\n".join(lines)


def main() -> None:
    """Write both split files and print the composition table."""
    meta = json.loads((ROOT / "meta" / "dev.json").read_text(encoding="utf-8"))
    spike = select_spike(meta)
    out = ROOT / "splits"
    out.mkdir(exist_ok=True)
    (out / "spike40.json").write_text(json.dumps(spike, indent=1) + "\n", encoding="utf-8")
    dev = sorted(r["doc_id"] for r in meta)
    (out / "dev100.json").write_text(json.dumps(dev, indent=1) + "\n", encoding="utf-8")
    print(composition(meta, spike))


if __name__ == "__main__":
    main()
