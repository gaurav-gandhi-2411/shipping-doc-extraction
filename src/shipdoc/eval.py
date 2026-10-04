"""Scorer wrapper, slices, bootstrap, error taxonomy.

The official scorer (``assignment/score.py``) is loaded as a module and its ``score_doc``,
``aggregate``, ``same``, ``match_rows``, ``HEADER`` and ``ROW`` are reused unmodified. Nothing
here re-implements scoring; the taxonomy only *explains* diffs the scorer already counts as wrong.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import re
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from rapidfuzz.distance import Levenshtein

from shipdoc import locate, paths
from shipdoc.ocr import PageOcr, doc_pages

SCORER_ENV = "SHIPDOC_SCORER_PATH"

_SCORER_CACHE: dict[Path, ModuleType] = {}


def load_scorer(path: str | os.PathLike[str] | None = None) -> ModuleType:
    """Import the official scorer (path, else $SHIPDOC_SCORER_PATH, else score.py in
    SHIPDOC_ASSIGNMENT_DIR).

    Raises FileNotFoundError with an actionable message when the file is absent (it is gitignored,
    so a fresh checkout will not have it).
    """
    p = Path(path or os.environ.get(SCORER_ENV) or paths.scorer_path()).resolve()
    if p in _SCORER_CACHE:
        return _SCORER_CACHE[p]
    if not p.is_file():
        raise FileNotFoundError(
            f"official scorer not found at {p}; place score.py there or set {SCORER_ENV} "
            "(the assignment dir is gitignored and never committed)"
        )
    spec = importlib.util.spec_from_file_location("_official_score", p)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load scorer module from {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _SCORER_CACHE[p] = mod
    return mod


def _sc() -> ModuleType:
    return load_scorer()


# --------------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------------


def load_gold(directory: str | os.PathLike[str]) -> dict[str, dict[str, Any]]:
    """Load every ``*.json`` gold label in `directory`, keyed by doc_id (utf-8)."""
    gold: dict[str, dict[str, Any]] = {}
    for f in sorted(glob.glob(os.path.join(os.fspath(directory), "*.json"))):
        with open(f, encoding="utf-8") as fh:
            g = json.load(fh)
        gold[g["doc_id"]] = g
    if not gold:
        raise FileNotFoundError(f"no gold labels found in {directory}")
    return gold


def load_json(path: str | os.PathLike[str]) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------------------------
# Scoring and slices
# --------------------------------------------------------------------------------------------


def per_doc_results(
    pred: dict[str, Any], gold: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """score_doc for every gold document (missing predictions score as None -> zero)."""
    if not isinstance(pred, dict):
        raise ValueError("submission must be a JSON object keyed by doc_id")
    sc = _sc()
    return {d: sc.score_doc(pred.get(d), g) for d, g in gold.items()}


def slice_doc_ids(
    gold: dict[str, dict[str, Any]],
    meta: list[dict[str, Any]] | None,
    group_slices: bool = True,
) -> dict[str, list[str]]:
    """Slice name -> doc ids, built exactly like score.py main() for bool tags.

    Extra (not in the CLI): a string-valued ``supplier_group`` yields ``supplier_group=<value>``
    slices; pass ``group_slices=False`` for strict CLI parity.
    """
    out: dict[str, list[str]] = {
        "all": list(gold),
        "invoices": [d for d, g in gold.items() if g["doc_type"] == "invoice"],
        "waybills": [d for d, g in gold.items() if g["doc_type"] == "waybill"],
    }
    if meta:
        m = {x["doc_id"]: x for x in meta if x["doc_id"] in gold}
        tags = sorted({k for x in m.values() for k, v in x.items() if isinstance(v, bool)})
        for t in tags:
            out[f"{t}=yes"] = [d for d, x in m.items() if x.get(t)]
            out[f"{t}=no"] = [d for d, x in m.items() if not x.get(t)]
        if group_slices:
            groups = sorted(
                {
                    x["supplier_group"]
                    for x in m.values()
                    if isinstance(x.get("supplier_group"), str)
                }
            )
            for gname in groups:
                out[f"supplier_group={gname}"] = [
                    d for d, x in m.items() if x.get("supplier_group") == gname
                ]
    out.update(_redaction_slices(gold))
    return out


def _redaction_slices(gold: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    """Slices of a synthetic-redaction run, from each label's ``recipe`` (empty otherwise).

    ``target=header`` / ``target=row`` split by what was occluded. For row targets,
    ``inferable_from_siblings=yes|no``: yes means the hidden value also appears in a sibling row
    or elsewhere on the page, so a correct fill proves less than it would for a "no" variant.
    """
    recipes = {d: g.get("recipe") for d, g in gold.items()}
    if not gold or not all(g.get("synthetic") is True and recipes[d] for d, g in gold.items()):
        return {}
    out: dict[str, list[str]] = {
        "target=header": [d for d, r in recipes.items() if r["row_idx"] is None],
        "target=row": [d for d, r in recipes.items() if r["row_idx"] is not None],
        "inferable_from_siblings=yes": [
            d for d, r in recipes.items() if r.get("inferable_from_siblings") is True
        ],
        "inferable_from_siblings=no": [
            d for d, r in recipes.items() if r.get("inferable_from_siblings") is False
        ],
    }
    return {k: v for k, v in out.items() if v}


def score(
    pred: dict[str, Any],
    gold: dict[str, dict[str, Any]],
    meta: list[dict[str, Any]] | None = None,
    group_slices: bool = False,
) -> dict[str, Any]:
    """Report dict identical to ``score.py --out`` (all/invoices/waybills + ``<tag>=yes/no``).

    `group_slices=True` adds per-supplier_group slices (not part of the CLI report).
    """
    sc = _sc()
    per = per_doc_results(pred, gold)
    slices = slice_doc_ids(gold, meta, group_slices=group_slices)
    return {name: sc.aggregate([per[d] for d in ids]) for name, ids in slices.items()}


# --------------------------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------------------------

METRICS: dict[str, Callable[[dict[str, Any]], float]] = {
    "OVERALL": lambda a: a["OVERALL"],
    "header_field_accuracy": lambda a: a["header_field_accuracy"],
    "row_f1": lambda a: a["row_f1"],
    "documents_fully_correct": lambda a: a["documents_fully_correct"],
    "false_fill_rate": lambda a: a["false_fill_rate"],
}


def _rng(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed))


def bootstrap_metrics(
    per_doc: list[dict[str, Any]],
    metrics: dict[str, Callable[[dict[str, Any]], float]] | None = None,
    n: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, dict[str, float]]:
    """Doc-level percentile bootstrap for several metrics sharing the same resamples."""
    metrics = metrics or METRICS
    sc = _sc()
    N = len(per_doc)
    if N == 0:
        return {}
    idx = _rng(seed).integers(0, N, size=(n, N))
    point = sc.aggregate(per_doc)
    draws: dict[str, list[float]] = {k: [] for k in metrics}
    for row in idx:
        agg = sc.aggregate([per_doc[i] for i in row])
        for k, fn in metrics.items():
            draws[k].append(fn(agg))
    out = {}
    for k, fn in metrics.items():
        lo, hi = np.quantile(draws[k], [alpha / 2, 1 - alpha / 2])
        out[k] = {"point": float(fn(point)), "lo": float(lo), "hi": float(hi)}
    return out


def bootstrap_ci(
    per_doc_results: list[dict[str, Any]],
    metric_fn: Callable[[dict[str, Any]], float],
    n: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Percentile CI of `metric_fn(aggregate(resampled docs))`; returns point/lo/hi."""
    return bootstrap_metrics(per_doc_results, {"m": metric_fn}, n=n, seed=seed, alpha=alpha)["m"]


def confidence_intervals(
    pred: dict[str, Any],
    gold: dict[str, dict[str, Any]],
    meta: list[dict[str, Any]] | None = None,
    n: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, dict[str, dict[str, float]]]:
    """CIs for the five headline metrics on `all` and every slice (incl. supplier_group)."""
    per = per_doc_results(pred, gold)
    out = {}
    for name, ids in slice_doc_ids(gold, meta).items():
        if ids:
            out[name] = bootstrap_metrics([per[d] for d in ids], n=n, seed=seed, alpha=alpha)
    return out


def paired_bootstrap(
    pred_a: dict[str, Any],
    pred_b: dict[str, Any],
    gold: dict[str, dict[str, Any]],
    n: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, dict[str, float]]:
    """Paired doc-level bootstrap of B - A using the same resampled indices for both systems."""
    sc = _sc()
    pa, pb = per_doc_results(pred_a, gold), per_doc_results(pred_b, gold)
    ids = list(gold)
    ra, rb = [pa[d] for d in ids], [pb[d] for d in ids]
    N = len(ids)
    idx = _rng(seed).integers(0, N, size=(n, N))
    deltas: dict[str, list[float]] = {k: [] for k in METRICS}
    for row in idx:
        aa = sc.aggregate([ra[i] for i in row])
        ab = sc.aggregate([rb[i] for i in row])
        for k, fn in METRICS.items():
            deltas[k].append(fn(ab) - fn(aa))
    full_a, full_b = sc.aggregate(ra), sc.aggregate(rb)
    out = {}
    for k, fn in METRICS.items():
        d = np.asarray(deltas[k])
        lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
        out[k] = {
            "a": float(fn(full_a)),
            "b": float(fn(full_b)),
            "delta": float(fn(full_b) - fn(full_a)),
            "lo": float(lo),
            "hi": float(hi),
            "p_delta_le_0": float(np.mean(d <= 0)),
        }
    return out


def _cis_overlap(a: dict[str, float], b: dict[str, float]) -> bool:
    """Closed intervals [lo, hi] share at least one point."""
    return a["lo"] <= b["hi"] and b["lo"] <= a["hi"]


def rank_runs(
    preds: dict[str, dict[str, Any]],
    gold: dict[str, dict[str, Any]],
    n: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Rank finished runs by OVERALL and paired-bootstrap the top two (spec Phase 2.3).

    `preds` maps a run name to its predictions.json content; `gold` must already be restricted to
    the doc list every run covers. Ties on OVERALL break by lower false_fill_rate, then name.
    ``delta`` is top1 - top2 on the shared doc set (paired, same resamples); ``p_le_0`` is the
    bootstrap share of resamples where top1 is not better. ``cis_overlap`` compares the two runs'
    individual 95% OVERALL CIs (the spec's rule) and drives ``decision``: overlap -> both go to
    dev100, otherwise the leader wins. ``delta_ci_excludes_0`` is reported alongside, not used.
    """
    if len(preds) < 2:
        raise ValueError("need at least 2 runs to rank")
    rows = []
    for name, pred in preds.items():
        per = per_doc_results(pred, gold)
        ci = bootstrap_metrics(list(per.values()), n=n, seed=seed)
        agg = _sc().aggregate(list(per.values()))
        rows.append(
            {
                "config": name,
                "n_docs": len(per),
                "OVERALL": float(agg["OVERALL"]),
                "ci95": [ci["OVERALL"]["lo"], ci["OVERALL"]["hi"]],
                "false_fill_rate": float(agg["false_fill_rate"]),
            }
        )
    rows.sort(key=lambda r: (-r["OVERALL"], r["false_fill_rate"], r["config"]))
    a, b = rows[0], rows[1]
    pb = paired_bootstrap(preds[b["config"]], preds[a["config"]], gold, n=n, seed=seed)["OVERALL"]
    overlap = _cis_overlap(
        {"lo": a["ci95"][0], "hi": a["ci95"][1]}, {"lo": b["ci95"][0], "hi": b["ci95"][1]}
    )
    return {
        "ranking": rows,
        "top2": [a["config"], b["config"]],
        "delta": pb["delta"],
        "ci95": [pb["lo"], pb["hi"]],
        "p_le_0": pb["p_delta_le_0"],
        "cis_overlap": overlap,
        "delta_ci_excludes_0": bool(pb["lo"] > 0 or pb["hi"] < 0),
        "n_docs": len(gold),
        "n_resamples": n,
        "seed": seed,
        "decision": "run_dev100" if overlap else "pick_top1",
    }


# --------------------------------------------------------------------------------------------
# Error taxonomy
# --------------------------------------------------------------------------------------------

#: Precedence when several categories could apply to one error (first match wins).
PRECEDENCE = (
    "doc_type",
    "false_fill",
    "hallucination",
    "normalization",
    "convention",
    "misread",
    "other",
)
#: row_missing / row_extra / row_split_merge are structural and never compete with the above:
#: a row is either unmatched (structural) or matched (field-level errors, precedence applies).
ROW_CATEGORIES = ("row_missing", "row_extra", "row_split_merge")

#: misread: normalized (lowercase alphanumeric) Levenshtein distance <= 2 OR similarity >= 0.8.
MISREAD_MAX_DISTANCE = 2
MISREAD_MIN_SIMILARITY = 0.8
#: hallucination: the predicted value has no match in the doc's OCR pages at any level of
#: `shipdoc.locate` (exact, scorer-normalized, or fuzzy at the data-chosen threshold).

_DATE_FORMATS = (
    "%d/%m/%Y", "%m/%d/%Y", "%d.%m.%Y", "%d-%m-%Y", "%m-%d-%Y", "%d %b %Y", "%d %B %Y",
    "%b %d, %Y", "%B %d, %Y", "%d-%b-%Y", "%d-%b-%y", "%Y/%m/%d", "%Y.%m.%d", "%Y%m%d",
    "%d/%m/%y", "%m/%d/%y", "%d.%m.%y", "%b %d %Y", "%B %d %Y",
)  # fmt: skip


def _alnum(v: Any) -> str:
    return re.sub(r"[^0-9a-z]", "", str(v).lower())


def _is_empty(v: Any) -> bool:
    return v is None or (isinstance(v, str) and not v.strip())


def parse_number(v: Any) -> float | None:
    """Parse a number tolerating currency symbols/letters and European decimal commas."""
    s = re.sub(r"[^0-9,.\-]", "", str(v))
    if not re.search(r"\d", s):
        return None
    if "," in s and "." in s:
        dec = "," if s.rfind(",") > s.rfind(".") else "."
        thou = "." if dec == "," else ","
        s = s.replace(thou, "").replace(dec, ".")
    elif "," in s:
        s = s.replace(",", ".") if re.fullmatch(r"-?\d+,\d{1,2}", s) else s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def _date_matches(pred: str, gold: str) -> bool:
    p = pred.strip()
    if p == gold:
        return False  # identical text is not a normalization problem
    if p.startswith(gold):  # ISO date with trailing time component
        return True
    for fmt in _DATE_FORMATS:
        try:
            if datetime.strptime(p, fmt).strftime("%Y-%m-%d") == gold:
                return True
        except ValueError:
            continue
    return False


def _is_normalization(sc: ModuleType, field: str, pred: Any, gold: Any) -> bool:
    kind = sc.KIND.get(field, "id")
    if kind == "date":
        return _date_matches(str(pred), str(gold).strip())
    if kind == "num":
        a, b = parse_number(pred), sc._num(gold)
        return a is not None and b is not None and abs(a - b) < 0.005
    return False


def _is_misread(pred: Any, gold: Any) -> bool:
    p, g = _alnum(pred), _alnum(gold)
    if not p or not g:
        return False
    return (
        Levenshtein.distance(p, g) <= MISREAD_MAX_DISTANCE
        or Levenshtein.normalized_similarity(p, g) >= MISREAD_MIN_SIMILARITY
    )


def classify_value(
    sc: ModuleType,
    field: str,
    pred: Any,
    gold: Any,
    siblings: dict[str, Any],
    ocr: locate.DocIndex | None,
) -> tuple[str, str, str]:
    """Classify one wrong field value. Returns (category, label, support).

    Precedence: doc_type > false_fill > hallucination > normalization > convention > misread >
    other (doc_type is handled by the caller). `siblings` are the other gold fields of the same
    header/row, used for convention:wrong_field. `ocr` is the doc's OCR index; the value is
    'supported' when `locate.locate` finds it at any level, else 'unsupported' (hallucination).
    Without OCR the support is 'unknown_support' (hallucination is then not evaluated).
    """
    if _is_empty(gold):
        return "false_fill", "false_fill", "n/a"
    if _is_empty(pred):
        return "convention", "convention:missed", "n/a"
    support = "unknown_support"
    if ocr is not None:
        if locate.locate(pred, field, ocr) is None:
            return "hallucination", "hallucination", "unsupported"
        support = "supported"
    if _is_normalization(sc, field, pred, gold):
        return "normalization", "normalization", support
    pa = _alnum(pred)
    if pa and any(k != field and not _is_empty(v) and _alnum(v) == pa for k, v in siblings.items()):
        return "convention", "convention:wrong_field", support
    if _is_misread(pred, gold):
        return "misread", "misread", support
    return "other", "other", support


def _pair_rows(
    sc: ModuleType, pred_rows: list[dict[str, Any]], gold_rows: list[dict[str, Any]]
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[int], list[int]]:
    """Recover the pairs behind score.py's match_rows (which returns only counts).

    Mirrors its two greedy passes using the scorer's own `match_rows`/`same` as the predicates,
    then asserts that the full-match count and per-field counts equal `match_rows(pred, gold)`.
    Returns (full_pairs, partial_pairs, unmatched_pred, unmatched_gold) as index lists.
    """
    free_p, free_g = list(range(len(pred_rows))), list(range(len(gold_rows)))
    full: list[tuple[int, int]] = []
    partial: list[tuple[int, int]] = []
    for gi in list(free_g):
        for pi in free_p:
            if sc.match_rows([pred_rows[pi]], [gold_rows[gi]])[0] == 1:
                free_p.remove(pi)
                free_g.remove(gi)
                full.append((pi, gi))
                break
    for gi in list(free_g):
        for pi in free_p:
            if sc.same(
                "supplier_part_number",
                pred_rows[pi].get("supplier_part_number"),
                gold_rows[gi].get("supplier_part_number"),
            ):
                free_p.remove(pi)
                free_g.remove(gi)
                partial.append((pi, gi))
                break
    n_full, fields = sc.match_rows(pred_rows, gold_rows)
    mine = {
        k: sum(sc.same(k, pred_rows[pi].get(k), gold_rows[gi].get(k)) for pi, gi in full + partial)
        for k in sc.ROW
    }
    if n_full != len(full) or mine != fields:
        raise RuntimeError("row pairing diverged from score.match_rows; taxonomy would be wrong")
    return full, partial, free_p, free_g


def _split_merge(
    sc: ModuleType,
    pred_rows: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
    nonfull_p: list[int],
    nonfull_g: list[int],
) -> tuple[set[int], set[int]]:
    """Find split/merge rows among the not-fully-matched rows.

    For each part number (scorer-normalized), if one side has a single row whose quantity equals
    the sum of >=2 rows of the other side with that part number, all those rows are flagged.
    Not-fully-matched (not just unmatched) rows are used because score.py pairs a merged row
    with one of its constituents by part number, leaving only the rest unmatched.
    """
    bad_p: set[int] = set()
    bad_g: set[int] = set()

    def key(r: dict[str, Any]) -> str:
        return re.sub(r"\s", "", str(r.get("supplier_part_number") or "")).upper()

    def qty(r: dict[str, Any]) -> float | None:
        return None if _is_empty(r.get("quantity")) else sc._num(r.get("quantity"))

    keys = {key(pred_rows[i]) for i in nonfull_p} | {key(gold_rows[i]) for i in nonfull_g}
    for k in keys:
        if not k:
            continue
        ps = [i for i in nonfull_p if key(pred_rows[i]) == k]
        gs = [i for i in nonfull_g if key(gold_rows[i]) == k]
        for singles, many, rows_s, rows_m, bad_s, bad_m in (
            (ps, gs, pred_rows, gold_rows, bad_p, bad_g),
            (gs, ps, gold_rows, pred_rows, bad_g, bad_p),
        ):
            if len(many) < 2:
                continue
            q_many = [qty(rows_m[i]) for i in many]
            if any(q is None for q in q_many):
                continue
            total = sum(q_many)  # type: ignore[arg-type]
            for s in singles:
                q = qty(rows_s[s])
                if q is not None and abs(q - total) < 0.005:
                    bad_s.add(s)
                    bad_m.update(many)
    return bad_p, bad_g


def _rec(
    doc_id: str, scope: str, field: str, category: str, label: str, pred: Any, gold: Any,
    support: str = "n/a", row: dict[str, int | None] | None = None,
) -> dict[str, Any]:  # fmt: skip
    r = {
        "doc_id": doc_id, "scope": scope, "field": field, "category": category, "label": label,
        "pred": pred, "gold": gold, "support": support,
    }  # fmt: skip
    if row:
        r.update(row)
    return r


def doc_errors(
    doc_id: str, pred_doc: Any, gold_doc: dict[str, Any], ocr: list[PageOcr] | None = None
) -> list[dict[str, Any]]:
    """All error records for one document (see `error_taxonomy`); `ocr` = the doc's OCR pages."""
    sc = _sc()
    index = locate.build_index(ocr) if ocr else None
    pd_ = pred_doc if isinstance(pred_doc, dict) else {}
    recs: list[dict[str, Any]] = []
    type_bad = pd_.get("doc_type") != gold_doc["doc_type"]
    if type_bad:
        recs.append(
            _rec(doc_id, "doc", "doc_type", "doc_type", "doc_type", pd_.get("doc_type"),
                 gold_doc["doc_type"])
        )  # fmt: skip
    ph = pd_.get("header") if isinstance(pd_.get("header"), dict) else {}
    gh = gold_doc["header"]
    for f in sc.HEADER[gold_doc["doc_type"]]:
        if sc.same(f, ph.get(f), gh.get(f)):
            continue
        if type_bad:  # root cause outranks everything (precedence)
            recs.append(_rec(doc_id, "header", f, "doc_type", "doc_type", ph.get(f), gh.get(f)))
            continue
        cat, label, sup = classify_value(sc, f, ph.get(f), gh.get(f), gh, index)
        recs.append(_rec(doc_id, "header", f, cat, label, ph.get(f), gh.get(f), sup))

    pr = [r for r in (pd_.get("line_items") or []) if isinstance(r, dict)]
    gr = gold_doc["line_items"]
    if type_bad:
        for gi, g in enumerate(gr):
            recs.append(
                _rec(doc_id, "row", "row", "doc_type", "doc_type", None, g, row={"gold_row": gi})
            )
        for pi, p in enumerate(pr):
            recs.append(
                _rec(doc_id, "row", "row", "doc_type", "doc_type", p, None, row={"pred_row": pi})
            )
        return recs
    full, partial, un_p, un_g = _pair_rows(sc, pr, gr)
    nonfull_p = un_p + [pi for pi, _ in partial]
    nonfull_g = un_g + [gi for _, gi in partial]
    bad_p, bad_g = _split_merge(sc, pr, gr, nonfull_p, nonfull_g)
    for pi in sorted(bad_p):
        recs.append(
            _rec(doc_id, "row", "row", "row_split_merge", "row_split_merge", pr[pi], None,
                 row={"pred_row": pi})
        )  # fmt: skip
    for gi in sorted(bad_g):
        recs.append(
            _rec(doc_id, "row", "row", "row_split_merge", "row_split_merge", None, gr[gi],
                 row={"gold_row": gi})
        )  # fmt: skip
    for pi, gi in partial:
        if pi in bad_p or gi in bad_g:
            continue
        for f in sc.ROW:
            if sc.same(f, pr[pi].get(f), gr[gi].get(f)):
                continue
            cat, label, sup = classify_value(sc, f, pr[pi].get(f), gr[gi].get(f), gr[gi], index)
            recs.append(
                _rec(doc_id, "row", f, cat, label, pr[pi].get(f), gr[gi].get(f), sup,
                     row={"pred_row": pi, "gold_row": gi})
            )  # fmt: skip
    for gi in un_g:
        if gi not in bad_g:
            recs.append(
                _rec(doc_id, "row", "row", "row_missing", "row_missing", None, gr[gi],
                     row={"gold_row": gi})
            )  # fmt: skip
    for pi in un_p:
        if pi not in bad_p:
            recs.append(
                _rec(doc_id, "row", "row", "row_extra", "row_extra", pr[pi], None,
                     row={"pred_row": pi})
            )  # fmt: skip
    return recs


def error_taxonomy(
    pred: dict[str, Any],
    gold: dict[str, dict[str, Any]],
    ocr_pages: dict[str, list[PageOcr]] | None = None,
    meta: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Automatic error categories from diffs.

    Precedence (single-valued per error): doc_type > false_fill > hallucination > normalization >
    convention > misread > other. A wrong doc_type labels every other error in that doc as
    doc_type too (a missing doc counts as a wrong doc_type). Rows: unmatched gold = row_missing,
    unmatched pred = row_extra, split/merge as in `_split_merge`; matched rows with wrong fields
    go through the same field precedence. hallucination needs `ocr_pages` (doc_id -> cached OCR
    pages, see `load_ocr_pages`); a doc without OCR pages, or no `ocr_pages` at all, gives
    'unknown_support' on its records.

    Returns {"errors": [...records], "counts": {slice: {label: n}},
    "counts_by_category": {slice: {category: n}}}.
    """
    errors: list[dict[str, Any]] = []
    for d, g in gold.items():
        ocr = None if ocr_pages is None else ocr_pages.get(d)  # no OCR for a doc -> unknown
        errors.extend(doc_errors(d, pred.get(d), g, ocr))
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for e in errors:
        by_doc.setdefault(e["doc_id"], []).append(e)
    counts: dict[str, dict[str, int]] = {}
    counts_cat: dict[str, dict[str, int]] = {}
    for name, ids in slice_doc_ids(gold, meta).items():
        c: dict[str, int] = {}
        cc: dict[str, int] = {}
        for d in ids:
            for e in by_doc.get(d, []):
                c[e["label"]] = c.get(e["label"], 0) + 1
                cc[e["category"]] = cc.get(e["category"], 0) + 1
        counts[name] = dict(sorted(c.items()))
        counts_cat[name] = dict(sorted(cc.items()))
    return {"errors": errors, "counts": counts, "counts_by_category": counts_cat}


# --------------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------------


def _fmt_ci(ci: dict[str, float], pct: bool = True) -> str:
    k = 100.0 if pct else 1.0
    return f"{k * ci['point']:.2f} [{k * ci['lo']:.2f}, {k * ci['hi']:.2f}]"


def markdown_table(
    report: dict[str, Any], cis: dict[str, dict[str, dict[str, float]]] | None = None
) -> str:
    """Markdown table: one row per slice; metrics in percent, with 95% CI when `cis` is given."""
    cols = list(METRICS)
    lines = [
        "| slice | n | " + " | ".join(cols) + " |",
        "|---|---:|" + "---:|" * len(cols),
    ]
    for name, agg in report.items():
        if not agg:
            continue
        cells = []
        for c in cols:
            if cis and name in cis:
                cells.append(_fmt_ci(cis[name][c]))
            else:
                cells.append(f"{100 * METRICS[c](agg):.2f}")
        lines.append(f"| {name} | {agg['documents']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_report(
    out_json: str | os.PathLike[str],
    report: dict[str, Any],
    cis: dict[str, dict[str, dict[str, float]]] | None = None,
    taxonomy: dict[str, Any] | None = None,
    out_md: str | os.PathLike[str] | None = None,
    title: str = "Evaluation report",
) -> None:
    """Write a JSON bundle (report, ci, taxonomy) and a markdown summary (default: same stem)."""
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    bundle: dict[str, Any] = {"report": report}
    if cis is not None:
        bundle["ci"] = cis
    if taxonomy is not None:
        bundle["taxonomy"] = taxonomy
    out_json.write_text(json.dumps(bundle, indent=1), encoding="utf-8")
    md = [f"# {title}\n", "Metrics in percent; brackets are bootstrap 95% CIs.\n"]
    md.append(markdown_table(report, cis))
    if taxonomy is not None:
        labels = sorted({k for c in taxonomy["counts"].values() for k in c})
        md.append("\n## Error taxonomy (count per slice)\n\n")
        md.append("| slice | " + " | ".join(labels) + " |\n")
        md.append("|---|" + "---:|" * len(labels) + "\n")
        for name, c in taxonomy["counts"].items():
            md.append(f"| {name} | " + " | ".join(str(c.get(k, 0)) for k in labels) + " |\n")
    Path(out_md or out_json.with_suffix(".md")).write_text("".join(md), encoding="utf-8")


def load_ocr_pages(
    doc_ids: list[str], cache_root: str | os.PathLike[str] | None = None
) -> dict[str, list[PageOcr]]:
    """Cached OCR pages per doc (``shipdoc.ocr.doc_pages``); docs with no cached page are omitted.

    `cache_root` defaults to the configured ``SHIPDOC_OCR_CACHE`` (shipdoc.paths).
    """
    root = None if cache_root is None else Path(cache_root)
    out = {d: doc_pages(d, root) for d in doc_ids}
    return {d: p for d, p in out.items() if p}


SYNTHETIC_TITLE = "SYNTHETIC redaction eval"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m shipdoc.eval", description=__doc__.split("\n")[0])
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--meta")
    ap.add_argument(
        "--ocr-cache",
        help="OCR cache root, <root>/<engine>/<split>/<page>.json (default: SHIPDOC_OCR_CACHE "
        "from the paths config)",
    )
    ap.add_argument("--out")
    ap.add_argument("--bootstrap", type=int, default=2000, help="resamples (0 disables)")
    ap.add_argument("--scorer", help="path to score.py (default: SHIPDOC_ASSIGNMENT_DIR/score.py)")
    a = ap.parse_args(argv)
    if a.scorer:
        os.environ[SCORER_ENV] = a.scorer
    gold = load_gold(a.gold)
    pred = load_json(a.pred)
    meta = load_json(a.meta) if a.meta else None
    report = score(pred, gold, meta, group_slices=True)
    cis = confidence_intervals(pred, gold, meta, n=a.bootstrap) if a.bootstrap else None
    ocr = load_ocr_pages(list(gold), a.ocr_cache)
    tax = error_taxonomy(pred, gold, ocr, meta)
    A = report["all"]
    # Variants from `shipdoc synth-redaction` carry synthetic=true: label the run accordingly.
    synthetic = all(g.get("synthetic") is True for g in gold.values())
    title = SYNTHETIC_TITLE if synthetic else "Evaluation report"
    if synthetic:
        print(f"== {title} (not comparable to official or OOF numbers) ==")
    print(f"documents {A['documents']}  OVERALL {100 * A['OVERALL']:.2f}")
    print(markdown_table(report, cis))
    if a.out:
        write_report(a.out, report, cis, tax, title=title)
    return 0


if __name__ == "__main__":
    sys.exit(main())
