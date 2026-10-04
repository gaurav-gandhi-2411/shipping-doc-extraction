"""Calibration v2: post-rule features, per-field-type models, nested tau, document auto-accept.

Built beside ``shipdoc.confidence`` (v1 stays untouched and is reused: labels, v1 feature builder,
novelty, ECE, reliability, tau scan, review metrics). v2 differs from v1 in four ways:

1. it scores the POST-RULE predictions (R1 / R2 / R3 of ``shipdoc.postrules``), so every feature
   and label is computed on what would ship;
2. generic extra features (no supplier id, no layout cluster id): table-header column presence from
   the OCR header line, rule-touched flags from the value-free rule records, value-shape agreement
   within a column of a page, row position (`EXTRA_FEATURES`);
3. a pooled model vs per-field-type models choice (`select_structure`);
4. nested tau (`nested_accept`), a document-level auto-accept model (`doc_table`) and tie-aware
   AUROC with document-level bootstrap (`auroc_boot`).

Fail closed on test data: every public entry that takes document ids calls `assert_no_test_ids`.
No function here reads or prints an extracted value except to compute a shape or a regex flag
that is stored as a number.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from shipdoc import confidence as cf
from shipdoc import rules
from shipdoc.extract import ROW_KEYS
from shipdoc.headerhint import header_cells
from shipdoc.ocr import PageOcr

SEED = cf.SEED
N_BOOT = cf.N_BOOT
TARGETS: tuple[float, ...] = (0.95, 0.98, 0.99)
#: Minimum coverage at which "best achievable precision" is quoted when a target is not attainable.
MIN_COVERAGE = 0.05
TEST_PREFIX = "test_"
HEADER = "header"
FIELD_TYPES: tuple[str, ...] = (HEADER, *ROW_KEYS)


class NoTestDataError(RuntimeError):
    """A document id of the unlabeled test split reached the calibration code (fail closed)."""


def assert_no_test_ids(doc_ids: Sequence[str]) -> None:
    """Raise `NoTestDataError` if any id starts with ``test_`` (test data must never enter)."""
    bad = [d for d in doc_ids if str(d).startswith(TEST_PREFIX)]
    if bad:
        raise NoTestDataError(
            f"{len(bad)} document id(s) of the test split, e.g. {bad[0]!r}: test data has no "
            "labels and must not enter calibration; refusing"
        )


# ---------------------------------------------------------------------------------------------
# Extra features
# ---------------------------------------------------------------------------------------------

#: Label regexes of the table-header cells (lowercase text). Generic wording, no supplier names.
COL_PATTERNS: dict[str, re.Pattern[str]] = {
    "cpn": re.compile(r"customer|\bcust\b|cust\.|buyer|\bcpn\b|your\s+(part|item|ref)"),
    "po": re.compile(r"\bpo\b|p\.o\b|purchase\s*order|order\s*(no|num|#|ref)|\border\b"),
    "qty": re.compile(r"\bqty\b|quantity|\bpcs\b|\bunits?\b"),
    "spn": re.compile(r"supplier|vendor|seller|\bmfr\b|manufacturer|\bour\s+(part|item)"),
}
_PARTISH = re.compile(r"\bpart\b|p/n|\bpn\b|\bitem\b|\bsku\b")
#: Which `COL_PATTERNS` key is the column of each row field.
OWN_COL = {
    "supplier_part_number": "spn",
    "customer_part_number": "cpn",
    "purchase_order": "po",
    "quantity": "qty",
}
RULE_NAMES = ("R1", "R2", "R3")
EXTRA_FEATURES: tuple[str, ...] = (
    "hc_cpn",
    "hc_po",
    "hc_qty",
    "hc_spn",
    "hc_own",
    "hc_none",
    "hc_fallback_p1",
    "rt_touched",
    "rt_R1",
    "rt_R2",
    "rt_R3",
    "sh_frac_same",
    "sh_dom_share",
    "sh_lt2",
    "pos_first_page",
    "pos_cont_page",
    "pos_idx_in_page",
    "pos_n_rows",
    "pos_unmapped",
)
_XI = {n: i for i, n in enumerate(EXTRA_FEATURES)}
DESIGN_V2_FEATURES: tuple[str, ...] = cf.DESIGN_FEATURES + EXTRA_FEATURES


def column_presence(cells: Sequence[str]) -> dict[str, float]:
    """Binary presence of the customer-part / PO / quantity / supplier-part column labels.

    A part-like cell (``part``, ``p/n``, ``item``, ``sku``) that is not customer-like counts as a
    supplier-part column; an explicit supplier / vendor / manufacturer label does too.
    """
    out = dict.fromkeys(COL_PATTERNS, 0.0)
    for c in cells:
        low = c.lower()
        for k, pat in COL_PATTERNS.items():
            if pat.search(low):
                out[k] = 1.0
        if _PARTISH.search(low) and not COL_PATTERNS["cpn"].search(low):
            out["spn"] = 1.0
    return out


def rule_touch_map(changes: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, int], str]:
    """``(scope, field, row_idx) -> rule`` of the cells the value-free rule records touched.

    R1 / R2 touch a header field; R3 (field ``cpn_po``) touches both slot fields of the row.
    """
    out: dict[tuple[str, str, int], str] = {}
    for c in changes:
        rule, field, row = str(c.get("rule")), str(c.get("field")), c.get("row")
        if rule == "R3" and isinstance(row, int):
            out[("row", rules.CPN, row)] = rule
            out[("row", rules.PO, row)] = rule
        elif rule in ("R1", "R2"):
            out[(HEADER, field, -1)] = rule
    return out


def _page_cols(
    ocr_pages: Sequence[PageOcr], pno: int, cache: dict[int, dict[str, float] | None]
) -> dict[str, float] | None:
    """Column presence of the table-header line of page `pno`; None when it has no such line."""
    if pno not in cache:
        cache[pno] = None
        if 0 <= pno < len(ocr_pages):
            cells, _, status = header_cells(ocr_pages[pno])
            cache[pno] = column_presence(cells) if status == "ok" else None
    return cache[pno]


def build_extras(
    rows: Sequence[cf.FieldRow],
    pred_doc: Mapping[str, Any],
    trace: Mapping[str, Any],
    ocr_pages: Sequence[PageOcr] | None,
    changes: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    """``(len(rows), len(EXTRA_FEATURES))`` generic features of the fields of one document.

    `rows` are the v1 rows (`cf.build_doc_features`) of the same prediction. The page of a row is
    the page of its merge source (`cf.row_sources`); when the merged row count differs from the
    prediction's the page is unknown (``pos_unmapped = 1``, position NaN, shape groups over the
    whole document, header columns from page 1). A page without a table-header line falls back
    to page 1 (``hc_fallback_p1``); no header line on either gives ``hc_none = 1``.
    """
    pages = list(trace.get("pages") or [])
    ocr = list(ocr_pages or [])
    pred_rows = [r for r in (pred_doc.get("line_items") or []) if isinstance(r, dict)]
    src = cf.row_sources(pages)
    mapped = len(src) == len(pred_rows)
    page_of = [src[i][0] if mapped else -1 for i in range(len(pred_rows))]
    idx_in_page: list[int] = []
    seen: dict[int, int] = {}
    for p in page_of:
        idx_in_page.append(seen.get(p, 0))
        seen[p] = seen.get(p, 0) + 1
    touch = rule_touch_map(changes)
    cache: dict[int, dict[str, float] | None] = {}
    # shape groups: (field, page) -> {row index: shape} of EMITTED values
    groups: dict[tuple[str, int], dict[int, str]] = {}
    for ri, r in enumerate(pred_rows):
        for f in ROW_KEYS:
            if not cf._blank(r.get(f)):
                groups.setdefault((f, page_of[ri]), {})[ri] = rules.shape_of(r[f])
    out = np.full((len(rows), len(EXTRA_FEATURES)), np.nan)
    for i, fr in enumerate(rows):
        _, scope, name, ri = fr.key
        v = out[i]
        is_row = scope == "row"
        pno = page_of[ri] if is_row else 0
        cols = _page_cols(ocr, pno, cache) if pno >= 0 else None
        fallback = 0.0
        if cols is None and pno != 0:
            cols, fallback = _page_cols(ocr, 0, cache), 1.0
        if pno < 0:
            cols, fallback = _page_cols(ocr, 0, cache), 1.0
        if cols is None:
            v[_XI["hc_none"]] = 1.0
            for k in COL_PATTERNS:
                v[_XI[f"hc_{k}"]] = 0.0
            v[_XI["hc_own"]] = 0.0
            v[_XI["hc_fallback_p1"]] = 0.0
        else:
            v[_XI["hc_none"]] = 0.0
            for k in COL_PATTERNS:
                v[_XI[f"hc_{k}"]] = cols[k]
            v[_XI["hc_own"]] = cols[OWN_COL[name]] if is_row else 0.0
            v[_XI["hc_fallback_p1"]] = fallback
        rule = touch.get((scope, name, ri))
        v[_XI["rt_touched"]] = float(rule is not None)
        for r_name in RULE_NAMES:
            v[_XI[f"rt_{r_name}"]] = float(rule == r_name)
        v[_XI["pos_n_rows"]] = float(len(pred_rows))
        v[_XI["sh_lt2"]] = 1.0
        v[_XI["pos_unmapped"]] = 0.0
        if not is_row:
            continue
        if not mapped:
            v[_XI["pos_unmapped"]] = 1.0
        else:
            v[_XI["pos_first_page"]] = float(pno == 0)
            v[_XI["pos_cont_page"]] = float(pno > 0)
            v[_XI["pos_idx_in_page"]] = float(idx_in_page[ri])
        grp = groups.get((name, page_of[ri]), {})
        if ri in grp and len(grp) - 1 >= 2:
            mine = grp[ri]
            others = [s for j, s in grp.items() if j != ri]
            v[_XI["sh_frac_same"]] = float(np.mean([s == mine for s in others]))
            counts: dict[str, int] = {}
            for s in grp.values():
                counts[s] = counts.get(s, 0) + 1
            v[_XI["sh_dom_share"]] = max(counts.values()) / len(grp)
            v[_XI["sh_lt2"]] = 0.0
    return out


@dataclass
class TableV2:
    """v1 `FieldTable` plus the v2 extra-feature matrix (same row order)."""

    table: cf.FieldTable
    extras: np.ndarray

    @property
    def keys(self) -> list[cf.FieldKey]:
        """Field keys of the rows."""
        return self.table.keys

    @property
    def emitted(self) -> np.ndarray:
        """Bool (n,): the prediction holds a value."""
        return self.table.emitted

    def types(self) -> np.ndarray:
        """Field type of every row: ``header`` or the row field name."""
        return np.array(
            [HEADER if k.scope == "header" else k.field for k in self.keys], dtype=object
        )


def design_v2(tv: TableV2, novelty: Mapping[str, float] | None) -> np.ndarray:
    """v1 design matrix (`cf.DESIGN_FEATURES`) followed by `EXTRA_FEATURES`."""
    return np.hstack([cf.design_matrix(tv.table, novelty), tv.extras])


# ---------------------------------------------------------------------------------------------
# Field-level models: pooled vs per-type
# ---------------------------------------------------------------------------------------------


def cross_fit_v2(
    tv: TableV2,
    y: np.ndarray,
    fit_mask: np.ndarray,
    doc_fold: Mapping[str, int],
    kind: str,
    novelty_table: Mapping[int, Mapping[str, float]] | None,
    groups: Mapping[str, str] | None,
    structure: str,
) -> np.ndarray:
    """Out-of-fold P(y=1) of every row; `structure` is ``pooled`` or ``per_type``.

    ``per_type`` fits one model per field type (header, each row field) on the rows of that type
    in the fitting folds. Fit on K-1 supplier folds, predict the held-out one (`cf.fold_splits`).
    """
    assert structure in ("pooled", "per_type"), structure
    doc_ids = [k.doc_id for k in tv.keys]
    assert_no_test_ids(doc_ids)
    types = tv.types()
    oof = np.full(len(y), np.nan)
    folds = sorted(set(doc_fold[d] for d in doc_ids))
    for (fit, pred), h in zip(cf.fold_splits(doc_ids, doc_fold, groups), folds, strict=True):
        X = design_v2(tv, None if novelty_table is None else novelty_table[h])
        use = fit[fit_mask[fit]]
        if structure == "pooled":
            oof[pred] = cf.fit_predict(kind, X[use], y[use].astype(int), X[pred])
            continue
        for t in FIELD_TYPES:
            u, p = use[types[use] == t], pred[types[pred] == t]
            if len(p):
                oof[p] = cf.fit_predict(kind, X[u], y[u].astype(int), X[p])
    return oof


@dataclass
class StructureChoice:
    """Outcome of the (kind, structure) selection for P(correct | emitted)."""

    structure: str
    kind: str
    oof: np.ndarray
    logloss: dict[str, float]
    kind_by_structure: dict[str, str]
    all_oof: dict[str, np.ndarray] | None = None  # "structure/kind" -> OOF (diagnostics only)


def select_structure(
    tv: TableV2,
    y: np.ndarray,
    doc_fold: Mapping[str, int],
    novelty_table: Mapping[int, Mapping[str, float]] | None,
    groups: Mapping[str, str] | None,
    margin: float = cf.GBM_MARGIN,
) -> StructureChoice:
    """Pre-registered selection: per structure LR unless GBM wins by `margin` nats of emitted OOF
    log-loss; then the structure (``pooled`` vs ``per_type``) with the lower emitted OOF log-loss
    (strictly lower wins, ties keep ``pooled``). Four cross-fits, two decisions, all reported."""
    em = tv.emitted
    yf = y.astype(float)
    ll: dict[str, float] = {}
    oofs: dict[tuple[str, str], np.ndarray] = {}
    kinds: dict[str, str] = {}
    for st in ("pooled", "per_type"):
        for k in ("lr", "gbm"):
            o = cross_fit_v2(tv, y, em, doc_fold, k, novelty_table, groups, st)
            oofs[(st, k)] = o
            ll[f"{st}/{k}"] = cf.log_loss(o[em], yf[em])
        kinds[st] = "gbm" if ll[f"{st}/lr"] - ll[f"{st}/gbm"] >= margin else "lr"
    pick = (
        "per_type"
        if ll[f"per_type/{kinds['per_type']}"] < ll[f"pooled/{kinds['pooled']}"]
        else "pooled"
    )
    k = kinds[pick]
    allo = {f"{s}/{kd}": np.where(em, o, np.nan) for (s, kd), o in oofs.items()}
    return StructureChoice(pick, k, np.where(em, oofs[(pick, k)], np.nan), ll, kinds, allo)


# ---------------------------------------------------------------------------------------------
# AUROC with document-level bootstrap
# ---------------------------------------------------------------------------------------------


def _auc_weighted(yy: np.ndarray, starts: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Tie-aware AUROC for each weight row; rows sorted by score, `starts` = tie-group starts."""
    pos = np.add.reduceat(weights * yy, starts, axis=1)
    neg = np.add.reduceat(weights * (1.0 - yy), starts, axis=1)
    below = np.cumsum(neg, axis=1) - neg
    num = (pos * (below + 0.5 * neg)).sum(axis=1)
    den = pos.sum(axis=1) * neg.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.asarray(np.where(den > 0, num / den, np.nan))


def auroc(score: np.ndarray, y: np.ndarray) -> float:
    """Tie-aware AUROC of `score` for the positive class ``y == 1`` (NaN if one class only)."""
    if len(score) == 0:
        return float("nan")
    order = np.argsort(score, kind="stable")
    s = score[order]
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    return float(_auc_weighted(y[order].astype(float), starts, np.ones((1, len(s))))[0])


def auroc_boot(
    score: np.ndarray,
    y: np.ndarray,
    doc_ids: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = SEED,
    chunk: int = 100,
) -> dict[str, float]:
    """AUROC with a document-level percentile bootstrap 95% CI (documents resampled whole).

    Resamples where one class is absent are skipped. Returns ``{point, lo, hi, n, n_pos, n_neg}``
    (``y == 1`` is the positive class, here "value correct").
    """
    n = len(score)
    out = {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": n}
    out["n_pos"], out["n_neg"] = int((y == 1).sum()), int((y == 0).sum())
    if n == 0 or out["n_pos"] == 0 or out["n_neg"] == 0:
        return out
    docs, inv = np.unique(doc_ids, return_inverse=True)
    nd = len(docs)
    order = np.argsort(score, kind="stable")
    s, yy, inv_s = score[order], y[order].astype(float), inv[order]
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    out["point"] = float(_auc_weighted(yy, starts, np.ones((1, n)))[0])
    idx = np.random.Generator(np.random.PCG64(seed)).integers(0, nd, size=(n_boot, nd))
    draws: list[np.ndarray] = []
    for a in range(0, n_boot, chunk):
        counts = np.stack([np.bincount(r, minlength=nd) for r in idx[a : a + chunk]])
        draws.append(_auc_weighted(yy, starts, counts[:, inv_s].astype(float)))
    d = np.concatenate(draws)
    d = d[~np.isnan(d)]
    if len(d):
        out["lo"], out["hi"] = (float(x) for x in np.quantile(d, [0.025, 0.975]))
    return out


# ---------------------------------------------------------------------------------------------
# Nested tau
# ---------------------------------------------------------------------------------------------


def nested_accept(
    conf: np.ndarray,
    correct: np.ndarray,
    fold_of: np.ndarray,
    mask: np.ndarray,
    target: float,
) -> tuple[np.ndarray, dict[int, float | None], dict[int, np.ndarray]]:
    """Nested auto-accept decisions of the rows in `mask`.

    For fold k, tau is chosen (`cf.select_tau`, lowest tau with precision >= `target`) on the
    rows of the OTHER folds only, then applied to the rows of fold k; the decisions are pooled.
    A fold whose other folds reach no tau accepts nothing. Returns ``(accept, {fold: tau},
    {fold: row indices tau was chosen on})``; the last is returned so callers can assert that
    the evaluated fold never selects its own threshold.
    """
    accept = np.zeros(len(conf), dtype=bool)
    taus: dict[int, float | None] = {}
    used: dict[int, np.ndarray] = {}
    for k in sorted(set(fold_of[mask].tolist())):
        sel = np.flatnonzero(mask & (fold_of != k))
        ev = np.flatnonzero(mask & (fold_of == k))
        assert not set(sel.tolist()) & set(ev.tolist()), "tau selected on the evaluated rows"
        tau = cf.select_tau(conf[sel], correct[sel], target)
        taus[k], used[k] = tau, sel
        if tau is not None:
            accept[ev] = conf[ev] >= tau
    return accept, taus, used


def insample_accept(
    conf: np.ndarray, correct: np.ndarray, mask: np.ndarray, target: float
) -> tuple[np.ndarray, float | None]:
    """In-sample reference: tau chosen on all rows of `mask` and applied to them (optimistic)."""
    idx = np.flatnonzero(mask)
    tau = cf.select_tau(conf[idx], correct[idx], target)
    accept = np.zeros(len(conf), dtype=bool)
    if tau is not None:
        accept[idx] = conf[idx] >= tau
    return accept, tau


def best_precision_at_coverage(
    conf: np.ndarray, correct: np.ndarray, min_cov: float = MIN_COVERAGE
) -> dict[str, float]:
    """Highest precision among thresholds accepting at least `min_cov` of the rows (in-sample).

    Used only to say how close a not-attainable target gets. Whole tie groups are accepted.
    """
    n = len(conf)
    if n == 0:
        return {"precision": float("nan"), "coverage": float("nan")}
    order = np.argsort(-conf, kind="stable")
    c, ok = conf[order], np.cumsum(correct[order].astype(float))
    cnt = np.arange(1, n + 1)
    last = np.append(c[1:] != c[:-1], True) & (cnt / n >= min_cov)
    if not last.any():
        last = np.zeros(n, dtype=bool)
        last[-1] = True
    prec = np.where(last, ok / cnt, -1.0)
    j = int(np.flatnonzero(prec == prec.max())[-1])  # ties in precision: the larger accepted set
    return {"precision": float(prec[j]), "coverage": float(cnt[j] / n)}


def coverage_at_target(conf: np.ndarray, correct: np.ndarray, target: float) -> float:
    """In-sample largest accepted share with precision >= `target` (0.0 if none): optimistic."""
    if len(conf) == 0:
        return float("nan")
    tau = cf.select_tau(conf, correct, target)
    return 0.0 if tau is None else float((conf >= tau).mean())


def accept_metrics(
    accept: np.ndarray,
    correct: np.ndarray,
    doc_ids: np.ndarray,
    n_boot: int = N_BOOT,
) -> dict[str, dict[str, float]]:
    """Review metrics with doc-level bootstrap CIs of fixed accept decisions.

    Reuses `cf.review_with_ci` with the 0/1 decision as the confidence and tau 0.5 (flagged =
    not accepted). CIs condition on the thresholds that produced `accept` (selection variance is
    not resampled).
    """
    return cf.review_with_ci(accept.astype(float), correct, doc_ids, 0.5, n_boot, SEED)


# ---------------------------------------------------------------------------------------------
# Document-level auto-accept
# ---------------------------------------------------------------------------------------------

DOC_THRESHOLDS: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95, 0.98)
DOC_FEATURES: tuple[str, ...] = (
    (
        "h_min",
        "h_mean",
        "h_n",
        "h_null_n",
        "h_exp_err",
    )
    + tuple(f"h_below_{t}" for t in DOC_THRESHOLDS)
    + ("r_min", "r_mean", "r_n_fields", "r_exp_err", "has_rows")
    + tuple(f"r_below_{t}" for t in DOC_THRESHOLDS)
    + tuple(f"rowmin_{f}" for f in ROW_KEYS)
    + ("n_rows", "is_waybill", "scanned", "rule_touched_n", "layout_novelty")
)
_NOV_COL = DOC_FEATURES.index("layout_novelty")


def _agg(p: np.ndarray) -> tuple[float, float, float, list[float]]:
    """(min, mean, expected errors, counts below each threshold); empty -> (1, 1, 0, zeros)."""
    if len(p) == 0:
        return 1.0, 1.0, 0.0, [0.0] * len(DOC_THRESHOLDS)
    return (
        float(p.min()),
        float(p.mean()),
        float((1.0 - p).sum()),
        [float((p < t).sum()) for t in DOC_THRESHOLDS],
    )


def doc_table(
    tv: TableV2, p_correct: np.ndarray, doc_ids: Sequence[str]
) -> tuple[np.ndarray, list[str]]:
    """``(X (n_docs, len(DOC_FEATURES)), doc id order)`` aggregated from field-level P(correct).

    Uses the emitted fields' OOF probability only (a null field has no P(correct)); the number
    of null header fields is a feature. Novelty is left NaN: it depends on the held-out fold and
    is filled per fold by `cross_fit_docs`.
    """
    assert_no_test_ids(doc_ids)
    pos = {d: i for i, d in enumerate(doc_ids)}
    rows_of: dict[str, list[int]] = {d: [] for d in doc_ids}
    for i, k in enumerate(tv.keys):
        rows_of[k.doc_id].append(i)
    sc = cf.BASE_FEATURES.index("scanned")
    X = np.full((len(doc_ids), len(DOC_FEATURES)), np.nan)
    for d, idx_l in rows_of.items():
        idx = np.array(idx_l, dtype=int)
        keys = [tv.keys[i] for i in idx]
        hdr = np.array([k.scope == "header" for k in keys], dtype=bool)
        em = tv.emitted[idx]
        hp = p_correct[idx][hdr & em]
        rp = p_correct[idx][~hdr & em]
        hmin, hmean, hexp, hbelow = _agg(hp)
        rmin, rmean, rexp, rbelow = _agg(rp)
        rowmin = []
        for f in ROW_KEYS:
            sel = np.array([k.scope == "row" and k.field == f for k in keys], dtype=bool) & em
            rowmin.append(float(p_correct[idx][sel].min()) if sel.any() else 1.0)
        n_rows = len({k.row_idx for k in keys if k.scope == "row"})
        vals = [hmin, hmean, float(len(hp)), float((hdr & ~em).sum()), hexp, *hbelow]
        vals += [rmin, rmean, float(len(rp)), rexp, float(len(rp) > 0), *rbelow, *rowmin]
        wb = float(any(k.scope == "header" and k.field == "carrier" for k in keys))
        vals += [float(n_rows), wb, float(tv.table.X[idx[0], sc]) if len(idx) else 0.0]
        vals += [float(tv.extras[idx, _XI["rt_touched"]].sum()), np.nan]
        X[pos[d]] = vals
    return X, list(doc_ids)


def cross_fit_docs(
    X: np.ndarray,
    y: np.ndarray,
    doc_ids: Sequence[str],
    doc_fold: Mapping[str, int],
    novelty_table: Mapping[int, Mapping[str, float]] | None,
    groups: Mapping[str, str] | None,
) -> np.ndarray:
    """Out-of-fold P(document fully correct) by logistic regression, supplier-fold cross-fit.

    The novelty column of fold h's design is ``novelty_table[h]`` (reference from training folds
    only), exactly the field-level construction.
    """
    assert_no_test_ids(doc_ids)
    folds = sorted(set(doc_fold[d] for d in doc_ids))
    oof = np.full(len(y), np.nan)
    for (fit, pred), h in zip(cf.fold_splits(doc_ids, doc_fold, groups), folds, strict=True):
        Xh = X.copy()
        if novelty_table is not None:
            Xh[:, _NOV_COL] = [novelty_table[h].get(d, np.nan) for d in doc_ids]
        oof[pred] = cf.fit_predict("lr", Xh[fit], y[fit].astype(int), Xh[pred])
    return oof


def doc_exact(sc: Any, pred_doc: Any, gold_doc: Mapping[str, Any]) -> bool:
    """The scorer's ``documents_fully_correct`` cell of one document: ``score_doc(..)["exact"]``
    (right doc type, every header field right, exactly the right set of rows)."""
    return bool(sc.score_doc(pred_doc, gold_doc)["exact"])
