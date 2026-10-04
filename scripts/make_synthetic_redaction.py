"""Write ``splits/synthetic_redaction_dev.json``: ~200 single-occlusion recipes over dev docs.

Each recipe hides ONE gold value (a header field, or one row's supplier_part_number / quantity /
purchase_order) of a dev doc, so a model that fills it anyway has false-filled a redaction. The
file holds recipes only (doc_id, field names, seed, method) plus aggregate counts; images are
regenerated on demand by ``python -m shipdoc synth-redaction --materialize``.

A (doc, field) pair is a candidate only when its box is reliable and the occlusion is clean:

* the gold value is non-null and the locator finds it at level exact/normalized (<= b);
* header fields: exactly ONE line of the document holds the value (if it is printed twice,
  hiding one copy would leave the value readable, so a null gold would be wrong);
* the widest occlusion (padding + margin + feather) does not overlap any other located gold
  value (collateral damage would make "all other fields untouched" false for the page).

Backfill (second pass, ``MULTI_FIELDS``): invoice ``currency`` and ``total_amount`` are printed on
several lines, so the "exactly one line" rule excluded nearly all of them. They are instead
occluded at EVERY occurrence the locator finds (level <= normalized); a doc is eligible only if all
occurrences sit in the header/footer region (none in the table or on a continuation banner), at
most one per line, and none of the boxes causes collateral. Then the gold value really is
unreadable and null. The recipe records ``n_boxes`` and ``all_occurrences``. The first 200 recipes
(no ``backfill`` key) are kept byte-for-byte; the backfill recipes are appended with fresh variant
indices, and re-running rebuilds them from the same base, so the file is reproducible.

Every row-target recipe also carries ``inferable_from_siblings``: True iff the occluded value also
appears in a sibling row of the doc's gold (scorer-equal) or elsewhere on the same page per the
OCR locator, i.e. a model could recover it without reading the occluded box.

Run: ``uv run python scripts/make_synthetic_redaction.py`` (needs the OCR cache and data dir).
"""

from __future__ import annotations

import json
import re
import sys
import zlib
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from shipdoc import augment as A
from shipdoc import locate as loc
from shipdoc import meta
from shipdoc.eval import load_scorer
from shipdoc.layout import analyze_page
from shipdoc.ocr import doc_pages

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "splits" / "synthetic_redaction_dev.json"
SEED = 42
N_TARGET = 200
ROW_FIELDS = ("supplier_part_number", "quantity", "purchase_order")
# Header fields printed on several lines: occluded at every occurrence (see module docstring).
MULTI_FIELDS = ("currency", "total_amount")
MULTI_REGIONS = ("header", "footer")
BACKFILL_FIELDS = ("awb_number", *MULTI_FIELDS)
# Backfill: minimum recipes per stratum, and minimum scanned recipes where scanned docs exist.
BACKFILL_MIN = 15
BACKFILL_STRATA = (
    "invoice_header:currency",
    "invoice_header:total_amount",
    "invoice_header:awb_number",
)
BACKFILL_MIN_SCANNED = {"invoice_header:awb_number": 5}
# Share of another value's box a candidate occlusion may cover before it counts as collateral.
COLLATERAL_MAX_FRAC = 0.10


def _overlap_frac(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    """Share of box `b`'s area covered by box `a`."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    area = max((b[2] - b[0]) * (b[3] - b[1]), 1e-9)
    return max(w, 0.0) * max(h, 0.0) / area


def widest_box(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Largest hard-edged applied box of any non-edge method (smudge adds a soft feather on top)."""
    x0, y0, x1, y1 = A.padded_box(box)
    m = A.MARGIN_MAX_LINE_FRAC * A.line_height(box)
    return (x0 - m, y0 - m, x1 + m, y1 + m)


def collateral(
    applied: tuple[float, float, float, float], page: int, others: list[tuple[int, tuple, str]],
    own: str,
) -> bool:  # fmt: skip
    """True iff `applied` covers more than ``COLLATERAL_MAX_FRAC`` of another value's box."""
    return any(
        p == page and tag != own and _overlap_frac(applied, b) > COLLATERAL_MAX_FRAC
        for p, b, tag in others
    )


def count_on_line(line: str, field: str, value: str) -> int:
    """Printed occurrences of a currency code / amount within one OCR line's text."""
    if field == "currency":
        return len(re.findall(rf"(?<![A-Za-z]){re.escape(value.strip())}(?![A-Za-z])", line, re.I))
    try:
        gold = float(re.sub(r"[,\s]", "", value))
    except ValueError:
        return 1
    return sum(
        any(abs(v - gold) < 0.005 for v in loc.parse_printed_number(tok)) for tok in line.split()
    )


def harmed_tags(
    applied: list[tuple[int, tuple[float, float, float, float]]],
    others: list[tuple[int, tuple, str]],
    own: str,
) -> set[str]:
    """Other values left unreadable by the applied boxes: ALL their located copies are hit.

    A copy is hit when a box covers more than ``COLLATERAL_MAX_FRAC`` of it. A value printed more
    than once keeps a clean copy when only some are hit, so (unlike the legacy ``collateral``
    rule, which flags any hit) it is not harmed. This matters for the currency code printed
    right next to the total: hiding the total always grazes one copy of the code.
    """
    total: Counter[str] = Counter()
    hit: Counter[str] = Counter()
    for p, b, tag in others:
        if tag == own:
            continue
        total[tag] += 1
        if any(ap == p and _overlap_frac(ab, b) > COLLATERAL_MAX_FRAC for ap, ab in applied):
            hit[tag] += 1
    return {t for t, n in hit.items() if n >= total[t]}


def feasible_seed(
    targets: list[A.Target],
    method: str,
    sizes: dict[int, tuple[int, int]],
    others: list[tuple[int, tuple, str]],
    own: str,
    multi: bool,
    rng: np.random.Generator,
    tries: int = 64,
) -> int | None:
    """First seed from `rng` whose APPLIED boxes (``plan_box``, exactly what ``occlude`` draws)
    harm no other value; None if `method` cannot be used on this doc.

    The legacy check bounds the applied box by its widest possible extent, which on a tightly
    packed scan (header lines ~6 px apart) rejects every occlusion although a seed with little
    jitter stays clear of the neighbours. Checking the real box per seed is exact, not looser.
    """
    for _ in range(tries):
        seed = int(rng.integers(0, 2**31 - 1))
        try:
            applied = [
                (t.page, A.plan_box(t.box, method, A.box_rng(seed, k, multi), sizes[t.page]))
                for k, t in enumerate(targets)
            ]
        except ValueError:  # edge_crop on a box far from every page edge
            return None
        if not harmed_tags(applied, others, own):
            return seed
    return None


def doc_backfill(
    gold: dict[str, Any], scanned: bool, excluded: Counter[str]
) -> list[dict[str, Any]]:
    """Backfill candidates of one invoice: awb_number (single copy) and the ``MULTI_FIELDS``.

    Unlike ``doc_candidates`` each candidate carries ``seeds`` (method -> a seed whose applied
    boxes are collateral-free) so the recipe's seed is chosen for cleanliness, and ``eligible`` is
    the methods that have one. Exclusion reasons are counted into `excluded`.
    """
    if gold["doc_type"] != "invoice":
        return []
    pages = doc_pages(gold["doc_id"])
    index = loc.build_index(pages)
    sc = load_scorer()
    others: list[tuple[int, tuple, str]] = []
    header_ms: dict[str, list[loc.Match]] = {}
    for f in sc.HEADER["invoice"]:
        if gold["header"].get(f) is None:
            continue
        header_ms[f] = loc.find_matches(gold["header"][f], f, index, max_level="normalized")
        others += [(m.page, m.box, f"h:{f}") for m in header_ms[f]]
    rows = gold.get("line_items") or []
    assigned = loc.assign_rows(rows, index) if rows else []
    for i, a in enumerate(assigned):
        for f in ROW_FIELDS:
            if rows[i].get(f) is None or not a.match:
                continue
            t = A.resolve_target(gold, pages, f, i, index=index, assigned=assigned)
            if t is not None:
                others.append((t.page, t.box, f"r{i}:{f}"))
    sizes = {k: (p.width, p.height) for k, p in enumerate(pages)}
    num = int(gold["doc_id"].split("_")[1])
    out: list[dict[str, Any]] = []
    for f in BACKFILL_FIELDS:
        if gold["header"].get(f) is None:
            continue
        key = f"invoice_header:{f}"
        targets = A.resolve_all_targets(gold, pages, f, index)
        if not targets:
            excluded[f"{key}|no_reliable_box"] += 1
            continue
        multi = f in MULTI_FIELDS
        if not multi and len(targets) > 1:
            excluded[f"{key}|multiple_occurrences"] += 1
            continue
        if multi:
            layouts = {t.page: analyze_page(pages[t.page]) for t in targets}
            if any(layouts[t.page].region(t.line_idx) not in MULTI_REGIONS for t in targets):
                excluded[f"{key}|occurrence_outside_header_total_region"] += 1
                continue
            value = str(gold["header"][f])
            if any(
                count_on_line(index.line_text(t.page, t.line_idx), f, value) > 1 for t in targets
            ):
                excluded[f"{key}|multiple_on_line"] += 1
                continue
        rng = np.random.default_rng([SEED + 2, num, zlib.crc32(f.encode())])
        seeds: dict[str, int] = {}
        for method in A.METHODS:
            if multi and method == "edge_crop":  # a cut to the page edge would hit many boxes
                continue
            s = feasible_seed(targets, method, sizes, others, f"h:{f}", multi, rng)
            if s is not None:
                seeds[method] = s
        if not seeds:
            excluded[f"{key}|collateral_all_seeds"] += 1
            continue
        cand: dict[str, Any] = {
            "stratum": key,
            "doc_id": gold["doc_id"],
            "field": f,
            "row_idx": None,
            "page": targets[0].page,
            "scanned": scanned,
            "eligible": sorted(seeds, key=A.METHODS.index),
            "seeds": seeds,
        }
        if multi:
            cand["n_boxes"] = len(targets)
        out.append(cand)
    return out


def select_backfill(
    pool: list[dict[str, Any]],
    base: list[dict[str, Any]],
    scanned_docs: dict[str, bool],
    share: float,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    """Choose backfill candidates so each ``BACKFILL_STRATA`` stratum has >= ``BACKFILL_MIN``
    recipes (base included) and >= ``BACKFILL_MIN_SCANNED`` scanned ones where scanned docs
    exist. Docs already used are avoided; method is the stratum's least-used eligible one.
    Deterministic in `seed`; never repeats a (doc, field) pair of `base`.
    """
    rng = np.random.default_rng(seed + 2)
    taken = {(r["doc_id"], r["field"], r["row_idx"]) for r in base}
    used: Counter[str] = Counter(r["doc_id"] for r in base)
    picked: list[dict[str, Any]] = []
    for st in BACKFILL_STRATA:
        field = st.split(":")[1]
        have = [r for r in base if r["field"] == field and r["row_idx"] is None]
        scanned_have = sum(scanned_docs[r["doc_id"]] for r in have)
        left = [
            c
            for c in sorted(pool, key=lambda c: c["doc_id"])
            if c["stratum"] == st and (c["doc_id"], c["field"], None) not in taken
        ]
        min_scanned = BACKFILL_MIN_SCANNED.get(st, round(share * BACKFILL_MIN))
        want_scan = max(0, min_scanned - scanned_have)
        need = max(BACKFILL_MIN - len(have), want_scan)
        chosen: list[dict[str, Any]] = []
        for flag, n in ((True, want_scan), (False, need - want_scan)):
            for _ in range(n):
                sub = [c for c in left if c["scanned"] is flag]
                if not sub:  # not enough of this kind: the other kind tops the quota up below
                    break
                tie = rng.random(len(sub))
                best = min(range(len(sub)), key=lambda k: (used[sub[k]["doc_id"]], tie[k]))
                c = sub[best]
                left.remove(c)
                used[c["doc_id"]] += 1
                chosen.append(c)
        while len(chosen) < need and left:
            c = left.pop(min(range(len(left)), key=lambda k: (used[left[k]["doc_id"]], k)))
            used[c["doc_id"]] += 1
            chosen.append(c)
        per_method: Counter[str] = Counter(r["method"] for r in have)
        for c in sorted(chosen, key=lambda c: (len(c["eligible"]), c["doc_id"])):
            c["method"] = min(c["eligible"], key=lambda m: (per_method[m], rng.random()))
            per_method[c["method"]] += 1
            taken.add((c["doc_id"], c["field"], None))
        picked += chosen
    return picked


def inferable_from_siblings(
    gold: dict[str, Any],
    recipe: dict[str, Any],
    pages: list[Any],
    index: loc.DocIndex,
    assigned: list[loc.RowAssignment],
) -> tuple[bool, bool]:
    """``(in_sibling_row, elsewhere_on_page)`` for one row-target recipe.

    Sibling: another row of the doc's gold holds a scorer-equal value of the same field.
    Page: the OCR locator (level <= normalized) finds the value on the recipe's page on a line
    other than the occluded one. Either makes the redaction recoverable without reading the box.
    """
    sc = load_scorer()
    f, i = recipe["field"], recipe["row_idx"]
    rows = gold["line_items"]
    v = rows[i][f]
    sib = any(
        sc.same(f, r.get(f), v) for j, r in enumerate(rows) if j != i and r.get(f) is not None
    )
    t = A.resolve_target(gold, pages, f, i, page=recipe["page"], index=index, assigned=assigned)
    if t is None:
        raise ValueError(f"{recipe['variant_id']}: occluded value no longer has a reliable box")
    page = any(
        m.page == t.page and m.line_idx != t.line_idx
        for m in loc.find_matches(v, f, index, max_level="normalized")
    )
    return sib, page


def doc_candidates(
    gold: dict[str, Any], scanned: bool, excluded: Counter[str]
) -> list[dict[str, Any]]:
    """Candidate recipes (without method/seed) for one doc; counts exclusions by reason."""
    pages = doc_pages(gold["doc_id"])
    index = loc.build_index(pages)
    wb = gold["doc_type"] == "waybill"
    group = "waybill_header" if wb else "invoice_header"
    sc = load_scorer()
    others: list[tuple[int, tuple, str]] = []
    header_ms: dict[str, list[loc.Match]] = {}
    for f in sc.HEADER[gold["doc_type"]]:
        if gold["header"].get(f) is None:
            continue
        header_ms[f] = loc.find_matches(gold["header"][f], f, index, max_level="normalized")
        others += [(m.page, m.box, f"h:{f}") for m in header_ms[f]]
    rows = gold.get("line_items") or []
    assigned = loc.assign_rows(rows, index) if rows else []
    row_ms: dict[tuple[int, str], A.Target | None] = {}
    for i, a in enumerate(assigned):
        for f in ROW_FIELDS:
            if rows[i].get(f) is None:
                continue
            t = A.resolve_target(gold, pages, f, i, index=index, assigned=assigned)
            row_ms[(i, f)] = t if a.match else None
            if t is not None:
                others.append((t.page, t.box, f"r{i}:{f}"))
    out: list[dict[str, Any]] = []
    cands = [(group, f, None, header_ms[f]) for f in header_ms]
    for (i, f), m in row_ms.items():
        cands.append(("row", f, i, [m] if m else []))
    for g, f, i, ms in cands:
        key = f"{g}:{f}"
        if not ms:
            excluded[f"{key}|no_reliable_box"] += 1
            continue
        if i is None and len(ms) > 1:
            excluded[f"{key}|multiple_occurrences"] += 1
            continue
        m = ms[0]
        own = f"h:{f}" if i is None else f"r{i}:{f}"
        if collateral(widest_box(m.box), m.page, others, own):
            excluded[f"{key}|collateral"] += 1
            continue
        eligible = [x for x in A.METHODS if x != "edge_crop"]
        w, h = pages[m.page].width, pages[m.page].height
        if A.edge_crop_eligible(m.box, (w, h)):
            trial = A.plan_box(m.box, "edge_crop", np.random.default_rng(0), (w, h))
            if not collateral(trial, m.page, others, own):
                eligible.append("edge_crop")
        out.append(
            {
                "stratum": key,
                "doc_id": gold["doc_id"],
                "field": f,
                "row_idx": i,
                "page": m.page,
                "scanned": scanned,
                "eligible": eligible,
            }
        )
    return out


def quotas(avail: dict[str, int], total: int) -> dict[str, int]:
    """Water-filling: raise the smallest stratum quota by one until `total` or capacity runs out."""
    q = dict.fromkeys(avail, 0)
    while sum(q.values()) < total:
        open_ = [s for s in sorted(avail) if q[s] < avail[s]]
        if not open_:
            break
        q[min(open_, key=lambda s: (q[s], s))] += 1
    return q


def select(
    cands: list[dict[str, Any]], total: int, scanned_share: float, seed: int = SEED
) -> list[dict[str, Any]]:
    """Pick ~`total` candidates: equal quota per stratum, scanned share kept, docs spread, and
    methods balanced (edge_crop only on eligible ones). Deterministic in `seed`."""
    rng = np.random.default_rng(seed)
    by: dict[str, list[dict[str, Any]]] = {}
    for c in sorted(cands, key=lambda c: (c["stratum"], c["doc_id"], c["row_idx"] or -1)):
        by.setdefault(c["stratum"], []).append(c)
    q = quotas({s: len(v) for s, v in by.items()}, total)
    used: Counter[str] = Counter()
    picked: list[dict[str, Any]] = []
    for s in sorted(by):
        pool = by[s]
        want_scan = round(q[s] * scanned_share)
        for flag, n in ((True, want_scan), (False, q[s] - want_scan)):
            sub = [c for c in pool if c["scanned"] is flag]
            if len(sub) < n:  # not enough of this kind: the other kind makes up the quota
                n = len(sub)
            for _ in range(n):
                tie = rng.random(len(sub))
                best = min(range(len(sub)), key=lambda k: (used[sub[k]["doc_id"]], tie[k]))
                c = sub.pop(best)
                used[c["doc_id"]] += 1
                picked.append(c)
        short = q[s] - sum(1 for c in picked if c["stratum"] == s)
        if short > 0:  # top up from whatever is left in the stratum
            left = [c for c in pool if c not in picked]
            for c in left[:short]:
                used[c["doc_id"]] += 1
                picked.append(c)
    # Methods: variants with fewer eligible methods first; each takes the method least used in
    # its own stratum (diversity per field), then globally (overall balance), then at random.
    count: Counter[str] = Counter()
    per_stratum: Counter[tuple[str, str]] = Counter()
    order = sorted(range(len(picked)), key=lambda k: (len(picked[k]["eligible"]), k))
    for k in order:
        s_key = picked[k]["stratum"]
        m = min(
            picked[k]["eligible"], key=lambda x: (per_stratum[(s_key, x)], count[x], rng.random())
        )
        picked[k]["method"] = m
        count[m] += 1
        per_stratum[(s_key, m)] += 1
    return picked


def to_recipes(picked: list[dict[str, Any]], seed: int = SEED) -> list[dict[str, Any]]:
    """Final recipes with variant ids (per-doc counter) and per-variant seeds."""
    rng = np.random.default_rng(seed + 1)
    per_doc: Counter[str] = Counter()
    out = []
    for c in sorted(picked, key=lambda c: (c["doc_id"], c["stratum"], c["row_idx"] or -1)):
        per_doc[c["doc_id"]] += 1
        out.append(
            {
                "variant_id": f"{c['doc_id']}__syn{per_doc[c['doc_id']]:03d}",
                "doc_id": c["doc_id"],
                "field": c["field"],
                "row_idx": c["row_idx"],
                "page": c["page"],
                "method": c["method"],
                "seed": int(rng.integers(0, 2**31 - 1)),
                "label": A.LABEL,
            }
        )
    return out


def to_backfill_recipes(
    picked: list[dict[str, Any]], base: list[dict[str, Any]], seed: int = SEED
) -> list[dict[str, Any]]:
    """Recipes for backfill picks: fresh per-doc variant indices after the base's, own seeds."""
    per_doc: Counter[str] = Counter()
    for r in base:
        per_doc[r["doc_id"]] = max(per_doc[r["doc_id"]], int(r["variant_id"].rsplit("syn", 1)[1]))
    order = {st: k for k, st in enumerate(BACKFILL_STRATA)}
    out = []
    for c in sorted(picked, key=lambda c: (order[c["stratum"]], c["doc_id"])):
        per_doc[c["doc_id"]] += 1
        r: dict[str, Any] = {
            "variant_id": f"{c['doc_id']}__syn{per_doc[c['doc_id']]:03d}",
            "doc_id": c["doc_id"],
            "field": c["field"],
            "row_idx": None,
            "page": c["page"],
            "method": c["method"],
            "seed": c["seeds"][c["method"]],  # chosen so the applied boxes are collateral-free
            "label": A.LABEL,
        }
        if "n_boxes" in c:
            r |= {"n_boxes": c["n_boxes"], "all_occurrences": True}
        r["backfill"] = True
        out.append(r)
    return out


def tag_inferable(
    recipes: list[dict[str, Any]], golds: dict[str, dict[str, Any]]
) -> Counter[tuple[str, bool, bool]]:
    """Set ``inferable_from_siblings`` on every row-target recipe; counts (field, sib, page)."""
    counts: Counter[tuple[str, bool, bool]] = Counter()
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for r in recipes:
        if r["row_idx"] is not None:
            by_doc.setdefault(r["doc_id"], []).append(r)
    for doc_id, rs in sorted(by_doc.items()):
        gold = golds[doc_id]
        pages = doc_pages(doc_id)
        index = loc.build_index(pages)
        assigned = loc.assign_rows(gold["line_items"], index)
        for r in rs:
            sib, page = inferable_from_siblings(gold, r, pages, index, assigned)
            r["inferable_from_siblings"] = sib or page
            counts[(r["field"], sib, page)] += 1
    return counts


def main() -> int:
    tags = {m["doc_id"]: m for m in json.loads((ROOT / "meta" / "dev.json").read_text("utf-8"))}
    golds = {g["doc_id"]: g for g in meta.load_labels("dev")}
    excluded: Counter[str] = Counter()
    cands: list[dict[str, Any]] = []
    for gold in golds.values():
        cands += doc_candidates(gold, bool(tags[gold["doc_id"]]["scanned"]), excluded)
    share = sum(t["scanned"] for t in tags.values()) / len(tags)
    legacy = to_recipes(select(cands, N_TARGET, share))
    avail = Counter(c["stratum"] for c in cands)
    # Base = the committed first batch, kept as is (rebuilt from the legacy selection only when no
    # file exists yet); the legacy rebuild is compared to it as a determinism check.
    prior = json.loads(OUT.read_text("utf-8"))["recipes"] if OUT.is_file() else None
    base = [r for r in prior if not r.get("backfill")] if prior else legacy
    stripped = [{k: v for k, v in r.items() if k != "inferable_from_siblings"} for r in base]
    base_matches_legacy = stripped == legacy
    # Second pass: backfill strata, with the applied-box collateral check.
    bf_excluded: Counter[str] = Counter()
    pool: list[dict[str, Any]] = []
    for gold in golds.values():
        pool += doc_backfill(gold, bool(tags[gold["doc_id"]]["scanned"]), bf_excluded)
    scanned_docs = {d: bool(t["scanned"]) for d, t in tags.items()}
    picked = select_backfill(pool, base, scanned_docs, share)
    recipes = base + to_backfill_recipes(picked, base)
    inferable = tag_inferable(recipes, golds)
    pool_by = Counter(c["stratum"] for c in pool)
    pool_scanned = Counter(c["stratum"] for c in pool if c["scanned"])
    payload = {
        "label": A.LABEL,
        "description": "SYNTHETIC redaction eval: one occluded value per variant of a dev doc",
        "seed": SEED,
        "n_recipes": len(recipes),
        "dev_scanned_doc_share": round(share, 4),
        "candidates_by_stratum": dict(sorted(avail.items())),
        "excluded_no_reliable_box_or_ambiguous": dict(sorted(excluded.items())),
        "backfill": {
            "min_per_stratum": BACKFILL_MIN,
            "min_scanned": BACKFILL_MIN_SCANNED,
            "pool_by_stratum": {k: pool_by[k] for k in BACKFILL_STRATA},
            "pool_scanned_by_stratum": {k: pool_scanned[k] for k in BACKFILL_STRATA},
            "excluded": dict(sorted(bf_excluded.items())),
            "n_added": len(recipes) - len(base),
        },
        "inferable_from_siblings": {
            "n_true": sum(1 for r in recipes if r.get("inferable_from_siblings") is True),
            "n_false": sum(1 for r in recipes if r.get("inferable_from_siblings") is False),
            "by_field_sibling_page": {
                f"{f}|sibling={s}|page={p}": n for (f, s, p), n in sorted(inferable.items())
            },
        },
        "recipes": recipes,
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"candidates {sum(avail.values())}; excluded {sum(excluded.values())}; "
          f"recipes {len(recipes)} ({len(base)} base + {len(recipes) - len(base)} backfill) "
          f"-> {OUT}")  # fmt: skip
    print("legacy rebuild equals committed base:", base_matches_legacy)
    by_reason: Counter[str] = Counter()
    for k, n in excluded.items():
        by_reason[k.split("|")[1]] += n
    print("excluded by reason:", dict(by_reason))
    print("backfill pool:", payload["backfill"]["pool_by_stratum"], "scanned:",
          payload["backfill"]["pool_scanned_by_stratum"])  # fmt: skip
    print("backfill exclusions:", payload["backfill"]["excluded"])
    print("inferable:", payload["inferable_from_siblings"])
    print(A.composition_markdown(recipes, tags))
    return 0


if __name__ == "__main__":
    sys.exit(main())
