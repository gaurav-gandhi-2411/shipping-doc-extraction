"""Choose the fuzzy threshold T on data (Step G.1).

Positives: every non-null train/dev gold header value against its OWN doc. Negatives: the same
value against one random OTHER doc of the same doc_type (seed 42). The fuzzy score of a pair is
the best rapidfuzz ratio over the scorer-normalized strings of every candidate span.

A negative that already matches at the exact/normalized level is a genuine coincidence (the same
currency or supplier is printed in the other doc), not something T can influence, so it is counted
separately and excluded from the false-match rate; FMR(T) is the share of the remaining negatives
whose best ratio is >= T. T is the smallest integer with FMR(T) <= FMR_TARGET.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from _corpus import DocRec, header_fields, is_empty

from shipdoc.eval import load_scorer
from shipdoc.locate import best_fuzzy_score, build_index, locate

SEED = 42
FMR_TARGET = 0.01
T_GRID = list(range(50, 101))


def _pair(
    value: Any, field: str, idx: Any
) -> tuple[bool, float]:  # (matched at exact/normalized, best ratio)
    ab = locate(value, field, idx, max_level="normalized") is not None
    return ab, best_fuzzy_score(value, field, idx)


def collect(corpus: list[DocRec]) -> list[dict[str, Any]]:
    """One row per (doc, header field, side) with ab flag and best ratio."""
    rng = random.Random(SEED)
    sc = load_scorer()
    by_type: dict[str, list[DocRec]] = defaultdict(list)
    for r in corpus:
        by_type[r.doc_type].append(r)
    rows: list[dict[str, Any]] = []
    for n, rec in enumerate(corpus):
        other = rng.choice([o for o in by_type[rec.doc_type] if o.doc_id != rec.doc_id])
        own, oth = build_index(rec.pages()), build_index(other.pages())
        for f in header_fields(rec.doc_type):
            v = rec.gold["header"].get(f)
            if is_empty(v):
                continue
            kind = str(sc.KIND.get(f, "id"))
            for side, idx in (("pos", own), ("neg", oth)):
                ab, ratio = _pair(v, f, idx)
                rows.append({"side": side, "field": f, "kind": kind, "ab": ab, "ratio": ratio})
        if n % 100 == 0:
            print(f"threshold study {n}/{len(corpus)}", flush=True)
    return rows


def curve(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """T vs recall and false-match rate over `rows` (see module docstring for definitions)."""
    pos = [r for r in rows if r["side"] == "pos"]
    neg = [r for r in rows if r["side"] == "neg"]
    neg_fuzzy = [r for r in neg if not r["ab"]]
    pos_fuzzy = [r for r in pos if not r["ab"]]
    out = []
    for t in T_GRID:
        rec_f = sum(r["ratio"] >= t for r in pos_fuzzy)
        fm = sum(r["ratio"] >= t for r in neg_fuzzy)
        out.append(
            {
                "T": t,
                "positive_recall_total": (sum(r["ab"] for r in pos) + rec_f) / len(pos)
                if pos
                else 0,
                "positive_recall_fuzzy_of_unresolved": rec_f / len(pos_fuzzy) if pos_fuzzy else 0,
                "negative_fmr": fm / len(neg_fuzzy) if neg_fuzzy else 0,
                "negative_fmr_incl_coincidental": (sum(r["ab"] for r in neg) + fm) / len(neg)
                if neg
                else 0,
            }
        )
    return out


def choose(c: list[dict[str, Any]]) -> int | None:
    """Smallest T with negative_fmr <= FMR_TARGET, or None if no T on the grid achieves it."""
    for row in c:
        if row["negative_fmr"] <= FMR_TARGET:
            return int(row["T"])
    return None


def study(corpus: list[DocRec]) -> dict[str, Any]:
    """Full study: pooled curve and chosen T, plus per-kind curves/T as diagnostics."""
    rows = collect(corpus)
    pooled = curve(rows)
    kinds = sorted({r["kind"] for r in rows})
    per_kind = {}
    for k in kinds:
        sub = [r for r in rows if r["kind"] == k]
        c = curve(sub)
        per_kind[k] = {
            "n_pos": sum(r["side"] == "pos" for r in sub),
            "n_neg": sum(r["side"] == "neg" for r in sub),
            "n_neg_coincidental_ab": sum(r["side"] == "neg" and r["ab"] for r in sub),
            "T_at_fmr_target": choose(c),
            "curve": c,
        }
    pos = [r for r in rows if r["side"] == "pos"]
    neg = [r for r in rows if r["side"] == "neg"]
    return {
        "seed": SEED,
        "fmr_target": FMR_TARGET,
        "T": choose(pooled),
        "n_pos": len(pos),
        "n_neg": len(neg),
        "n_pos_unresolved_at_normalized": sum(not r["ab"] for r in pos),
        "n_neg_coincidental_ab": sum(r["ab"] for r in neg),
        "curve": pooled,
        "per_kind": per_kind,
    }
