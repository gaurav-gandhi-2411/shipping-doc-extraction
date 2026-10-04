"""Per-field features, calibrators, null policy, review flag (spec Phase 5).

Everything here is CONFIDENCE ONLY: no function builds or edits a value except `apply_null_policy`,
which may turn an emitted value into null and can never fill one (`assert_never_fills`).

Pipeline (see ``scripts/calibrate.py`` for the CLI):

1. `build_doc_features` -> one `FieldRow` per field of a predicted document (header fields of the
   predicted doc type, and every field of every predicted row), emitted or null. It takes the
   prediction, the trace and the OCR pages only: no gold, no supplier id, so the same code path
   serves train, dev and test.
2. `label_doc` attaches gold-derived labels (scorer ``same`` semantics) AFTER features exist.
3. `calibrate_oof` cross-fits two calibrators by the supplier folds (fit on K-1 folds, predict the
   held-out one): (a) P(value correct | non-null emitted), (b) P(gold null) over all fields.
   Logistic regression by default; HistGradientBoosting only if it beats LR on cross-fitted
   log-loss by `GBM_MARGIN`.
4. `null_decisions` / `apply_null_policy`: emit null iff P(gold null) > P(value correct).
5. `select_tau` / `review_with_ci`: review threshold with a precision floor, document-level
   bootstrap CIs.

Missing-logprob contract: a page without ``field_logprobs`` gives NaN ``lp_min`` / ``lp_mean`` and
``has_logprobs = 0``; the LR pipeline mean-imputes the NaNs from the fitting rows and keeps the
indicator, so a run without logprobs (dev100 saved outputs) is handled without any value invented
for the label side.

Layout novelty is a doc-level feature supplied separately (`compute_novelty_table`) so that its
reference set (training-fold docs only) can follow the cross-fit; the function that scores it,
`layout_novelty(sig, reference)`, is a small swappable interface.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any, NamedTuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from shipdoc import locate, validate
from shipdoc.cluster import LayoutClusters
from shipdoc.extract import INVOICE_KEYS, ROW_KEYS, WAYBILL_KEYS
from shipdoc.logprobs import lookup
from shipdoc.merge import is_header_row
from shipdoc.normalize import normalize_value
from shipdoc.ocr import PageOcr, page_items

SEED = 42
N_BOOT = 2000
ECE_BINS = 15
#: Precision floor of the auto-accepted set (spec Phase 5.4).
PRECISION_TARGET = 0.98
#: GBM replaces LR only when its cross-fitted log-loss is lower by at least this many nats. 0.01 is
#: about 2% of a typical log-loss of 0.45: smaller gaps on ~30k correlated rows are noise-sized and
#: not worth the loss of a monotone, inspectable model.
GBM_MARGIN = 0.01
#: Min rows per class to fit a calibrator at all; below it a constant (prior) model is used.
MIN_PER_CLASS = 2

HEADER_FIELDS: dict[str, tuple[str, ...]] = {"invoice": INVOICE_KEYS, "waybill": WAYBILL_KEYS}
#: Scope-qualified field names, the categories of the field-type one-hot.
ALL_FIELDS: tuple[str, ...] = tuple(
    [f"header.{k}" for k in INVOICE_KEYS + WAYBILL_KEYS] + [f"row.{k}" for k in ROW_KEYS]
)
#: Features computed per field (NaN = missing). ``layout_novelty`` and the one-hot are added at
#: design time (`design_matrix`).
BASE_FEATURES: tuple[str, ...] = (
    "lp_min",
    "lp_mean",
    "has_logprobs",
    "ocr_exact_norm",
    "ocr_hit",
    "ocr_fuzzy",
    "ocr_conf",
    "has_ocr",
    "v_currency_bad",
    "v_airport_bad",
    "v_awb_bad",
    "v_hawb_bad",
    "v_date_bad",
    "xp_single",
    "xp_n_pages",
    "xp_agree",
    "xp_disagree",
    "xp_agree_frac",
    "scanned",
    "was_null",
)
NOVELTY = "layout_novelty"
DESIGN_FEATURES: tuple[str, ...] = BASE_FEATURES + (NOVELTY,) + tuple(f"ft_{f}" for f in ALL_FIELDS)
_N_BASE = len(BASE_FEATURES)
_IDX = {n: i for i, n in enumerate(BASE_FEATURES)}


class FieldKey(NamedTuple):
    """Identity of one field: ``row_idx`` is -1 for header fields, else the index in the
    prediction's (dict-only) ``line_items``."""

    doc_id: str
    scope: str
    field: str
    row_idx: int


@dataclass
class FieldRow:
    """One field of a predicted document with its label-free features."""

    key: FieldKey
    emitted: bool  # the prediction holds a non-blank value
    feats: dict[str, float] = field(default_factory=dict)


@dataclass
class FieldTable:
    """Feature matrix of many documents: ``X`` columns follow `BASE_FEATURES`."""

    keys: list[FieldKey]
    X: np.ndarray
    emitted: np.ndarray  # bool (n,)

    @property
    def doc_ids(self) -> np.ndarray:
        """Document id of every row, as a str array."""
        return np.array([k.doc_id for k in self.keys], dtype=object)


def assemble(rows: Sequence[FieldRow]) -> FieldTable:
    """Stack `FieldRow` features into a `FieldTable` (absent feature = NaN)."""
    X = np.full((len(rows), _N_BASE), np.nan, dtype=float)
    for i, r in enumerate(rows):
        for name, v in r.feats.items():
            X[i, _IDX[name]] = v
    return FieldTable([r.key for r in rows], X, np.array([r.emitted for r in rows], dtype=bool))


# ---------------------------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------------------------


def _blank(v: Any) -> bool:
    return v is None or (isinstance(v, str) and not v.strip())


def _cmp_key(v: Any) -> str:
    """Agreement key: lowercase alphanumerics (the scorer's identifier/name view of a value)."""
    return "" if _blank(v) else re.sub(r"[^0-9a-z]", "", str(v).lower())


def _page_value(name: str, v: Any) -> str:
    """Page-level raw value as the pipeline would normalise it (no cluster date order)."""
    if _blank(v):
        return ""
    return _cmp_key(normalize_value(name, v)[0])


def date_not_iso(value: Any) -> bool:
    """True iff a non-blank value is not a real ``YYYY-MM-DD`` date (what the scorer requires)."""
    if _blank(value):
        return False
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", str(value).strip())
    if m is None:
        return True
    try:
        date(int(m[1]), int(m[2]), int(m[3]))
    except ValueError:
        return True
    return False


def row_sources(pages: Sequence[Mapping[str, Any]]) -> list[tuple[int, int]]:
    """``(page index, raw row index)`` of every row `merge_pages` keeps, in merged order.

    Mirrors the merge defaults (repeated column-header rows and all-null rows are dropped), so
    the i-th kept source is the page row behind the i-th merged row. A run with other merge
    settings gives a different count; the caller then drops the row logprobs instead of guessing.
    """
    out: list[tuple[int, int]] = []
    for pno, p in enumerate(pages):
        parsed = p.get("parsed")
        raw_rows = (parsed.get("line_items") if isinstance(parsed, dict) else None) or []
        for ridx, raw in enumerate(raw_rows):
            if not isinstance(raw, dict):
                continue
            row = {k: raw.get(k) for k in ROW_KEYS}
            if is_header_row(row) or all(_blank(v) for v in row.values()):
                continue
            out.append((pno, ridx))
    return out


def _lp(entry: Mapping[str, Any] | None) -> tuple[float, float]:
    """(min, mean) of a ``field_logprobs`` entry; NaN when absent or non-finite (null)."""
    if not entry:
        return np.nan, np.nan
    mn, me = entry.get("min"), entry.get("mean")
    return (np.nan if mn is None else float(mn), np.nan if me is None else float(me))


def _match_conf(page: PageOcr, items_cache: dict[int, list[Any]], pno: int, box: Any) -> float:
    """Mean OCR confidence of the page items whose box overlaps the matched span."""
    if pno not in items_cache:
        items_cache[pno] = page_items(page)
    x0, y0, x1, y1 = box
    confs = [
        it.conf
        for it in items_cache[pno]
        if min(x1, it.box[2]) > max(x0, it.box[0]) and min(y1, it.box[3]) > max(y0, it.box[1])
    ]
    return float(np.mean(confs)) if confs else 0.0


def _ocr_feats(
    index: locate.DocIndex | None,
    pages: Sequence[PageOcr],
    items_cache: dict[int, list[Any]],
    name: str,
    value: str | None,
) -> dict[str, float]:
    """OCR support of a predicted value; label-free (it only reads the prediction and the page)."""
    out = dict.fromkeys(("ocr_exact_norm", "ocr_hit", "ocr_fuzzy", "ocr_conf"), 0.0)
    out["has_ocr"] = float(index is not None)
    if index is None or value is None:
        return out
    ms = locate.find_matches(value, name, index)
    best = ms[0] if ms else None
    exact_norm = best is not None and best.level in ("exact", "normalized")
    out["ocr_exact_norm"] = float(exact_norm)
    out["ocr_hit"] = float(best is not None)
    fuzzy = locate.best_fuzzy_score(value, name, index) / 100.0
    out["ocr_fuzzy"] = 1.0 if exact_norm else fuzzy
    if best is not None:
        out["ocr_conf"] = _match_conf(pages[best.page], items_cache, best.page, best.box)
    return out


def _validator_feats(name: str, value: str | None) -> dict[str, float]:
    flags = validate.validate_value(name, value) if value is not None else []
    return {
        "v_currency_bad": float(validate.FLAG_CURRENCY in flags),
        "v_airport_bad": float(validate.FLAG_AIRPORT in flags),
        "v_awb_bad": float(validate.FLAG_AWB in flags),
        "v_hawb_bad": float(validate.FLAG_HAWB in flags),
        "v_date_bad": float(name == "invoice_date" and date_not_iso(value)),
    }


def _xp_feats(n_pages: int, agree: int, disagree: int) -> dict[str, float]:
    return {
        "xp_single": float(n_pages <= 1),
        "xp_n_pages": float(n_pages),
        "xp_agree": float(agree),
        "xp_disagree": float(disagree),
        "xp_agree_frac": agree / (n_pages - 1) if n_pages > 1 else 0.0,
    }


def build_doc_features(
    doc_id: str,
    pred_doc: Mapping[str, Any],
    trace: Mapping[str, Any],
    ocr_pages: Sequence[PageOcr] | None,
) -> list[FieldRow]:
    """Features of every field of one predicted document.

    `pred_doc` is the final prediction (schema document), `trace` the doc's ``trace.jsonl`` line
    (page parses, optional ``field_logprobs``, ``merge.field_pages``) and `ocr_pages` its cached OCR
    pages (None/empty: ``has_ocr = 0`` and zero OCR support). One row per header field of the
    predicted doc type and per field of every predicted row, emitted or null. No gold is read.

    Logprob source: a non-null header value uses the page the merge took it from; a null header
    value uses the least confident page (min of mins, mean of means) because every page said null.
    Row fields use `row_sources`; when the merged row count differs from the prediction's, row
    logprobs are left missing.
    """
    pages = list(trace.get("pages") or [])
    n_pages = len(pages)
    scanned = float(any(str(p.get("image", "")).lower().endswith((".jpg", ".jpeg")) for p in pages))
    lps = [lookup(p["field_logprobs"]) if p.get("field_logprobs") else None for p in pages]
    ocr = list(ocr_pages or [])
    index = locate.build_index(ocr) if ocr else None
    items_cache: dict[int, list[Any]] = {}
    base = {"scanned": scanned}
    rows: list[FieldRow] = []

    def mk(
        scope: str,
        name: str,
        ridx: int,
        value: Any,
        lp: tuple[float, float],
        xp: dict[str, float],
    ) -> None:
        val = None if _blank(value) else str(value)
        feats: dict[str, float] = {
            **base,
            "lp_min": lp[0],
            "lp_mean": lp[1],
            "has_logprobs": float(not np.isnan(lp[0]) or not np.isnan(lp[1])),
            "was_null": float(val is None),
            **_ocr_feats(index, ocr, items_cache, name, val),
            **(_validator_feats(name, val) if scope == "header" else _validator_feats("", None)),
            **xp,
        }
        rows.append(FieldRow(FieldKey(doc_id, scope, name, ridx), val is not None, feats))

    header = pred_doc.get("header") if isinstance(pred_doc.get("header"), dict) else {}
    field_pages = (trace.get("merge") or {}).get("field_pages") or {}
    for name in HEADER_FIELDS.get(str(pred_doc.get("doc_type")), ()):
        value = header.get(name)
        emitted_key = _cmp_key(value)
        pg = field_pages.get(name)
        if emitted_key and pg and 1 <= pg <= n_pages and lps[pg - 1] is not None:
            lp = _lp(lps[pg - 1].get(("header", None, name)))
        elif not emitted_key:
            ents = [_lp(m.get(("header", None, name))) for m in lps if m is not None]
            ents = [e for e in ents if not np.isnan(e[0])]
            lp = (
                (min(e[0] for e in ents), float(np.mean([e[1] for e in ents])))
                if ents
                else (np.nan, np.nan)
            )
        else:
            lp = (np.nan, np.nan)
        page_vals = [
            _page_value(name, ((p.get("parsed") or {}).get("header") or {}).get(name))
            for p in pages
        ]
        if emitted_key:
            same = sum(v == emitted_key for v in page_vals)
            agree, disagree = max(same - 1, 0), sum(v not in ("", emitted_key) for v in page_vals)
        else:
            agree, disagree = 0, sum(v != "" for v in page_vals)
        mk("header", name, -1, value, lp, _xp_feats(n_pages, agree, disagree))

    pred_rows = [r for r in (pred_doc.get("line_items") or []) if isinstance(r, dict)]
    src = row_sources(pages)
    mapped = len(src) == len(pred_rows)
    page_row_keys: list[dict[str, set[str]]] = []
    for p in pages:
        parsed = p.get("parsed")
        raw_rows = [r for r in ((parsed or {}).get("line_items") or []) if isinstance(r, dict)]
        page_row_keys.append(
            {k: {_page_value(k, r.get(k)) for r in raw_rows} - {""} for k in ROW_KEYS}
        )
    for ri, row in enumerate(pred_rows):
        for name in ROW_KEYS:
            value = row.get(name)
            key = _cmp_key(value)
            lp = (np.nan, np.nan)
            if mapped and lps[src[ri][0]] is not None:
                lp = _lp(lps[src[ri][0]].get(("row", src[ri][1], name)))
            agree = sum(key in pk[name] for pk in page_row_keys) - 1 if key else 0
            mk("row", name, ri, value, lp, _xp_feats(n_pages, max(agree, 0), 0))
    return rows


# ---------------------------------------------------------------------------------------------
# Labels (gold side; never used by the feature builder)
# ---------------------------------------------------------------------------------------------


def label_doc(
    pred_doc: Mapping[str, Any], gold_doc: Mapping[str, Any]
) -> dict[tuple[str, str, int], tuple[bool, bool]]:
    """``(scope, field, row_idx) -> (correct, gold_null)`` of the fields features are built for.

    ``correct`` is the unmodified scorer's ``same(field, pred, gold)``, so a null prediction is
    correct iff gold is null and a value on a gold null (a false fill) is wrong. Rows are paired to
    gold rows with the scorer's own greedy matching (`shipdoc.eval._pair_rows`); a predicted row
    that stays unpaired (an extra row) has ``correct = False`` and ``gold_null = False`` for every
    field (there is no gold row to be null). When the predicted doc type is wrong nothing in the
    document can be right: every field is ``(False, False)``.
    """
    from shipdoc import eval as ev

    sc = ev.load_scorer()
    out: dict[tuple[str, str, int], tuple[bool, bool]] = {}
    ptype = str(pred_doc.get("doc_type"))
    type_ok = ptype == gold_doc["doc_type"]
    header = pred_doc.get("header") if isinstance(pred_doc.get("header"), dict) else {}
    for name in HEADER_FIELDS.get(ptype, ()):
        if not type_ok:
            out[("header", name, -1)] = (False, False)
            continue
        g = gold_doc["header"].get(name)
        out[("header", name, -1)] = (bool(sc.same(name, header.get(name), g)), _blank(g))
    pred_rows = [r for r in (pred_doc.get("line_items") or []) if isinstance(r, dict)]
    gold_rows = list(gold_doc.get("line_items") or [])
    paired: dict[int, int] = {}
    if type_ok and pred_rows:
        full, partial, _, _ = ev._pair_rows(sc, pred_rows, gold_rows)
        paired = dict(full + partial)
    for ri, row in enumerate(pred_rows):
        for name in ROW_KEYS:
            if ri in paired:
                g = gold_rows[paired[ri]].get(name)
                out[("row", name, ri)] = (bool(sc.same(name, row.get(name), g)), _blank(g))
            else:
                out[("row", name, ri)] = (False, False)
    return out


def label_arrays(
    keys: Sequence[FieldKey],
    labels: Mapping[str, Mapping[tuple[str, str, int], tuple[bool, bool]]],
) -> tuple[np.ndarray, np.ndarray]:
    """``(correct, gold_null)`` bool arrays aligned with `keys`; ``labels`` is per doc id."""
    corr = np.zeros(len(keys), dtype=bool)
    gnull = np.zeros(len(keys), dtype=bool)
    for i, k in enumerate(keys):
        corr[i], gnull[i] = labels[k.doc_id][(k.scope, k.field, k.row_idx)]
    return corr, gnull


# ---------------------------------------------------------------------------------------------
# Layout novelty (swappable)
# ---------------------------------------------------------------------------------------------


def layout_novelty(sig: np.ndarray, reference: np.ndarray) -> float:
    """Euclidean distance from a page-1 layout signature to the nearest reference row.

    `reference` is a ``(n, d)`` array of training-fold cluster centres (or training signatures).
    Large = a layout unlike any training layout. Swappable: any ``(sig, reference) -> float``.
    """
    ref = np.atleast_2d(np.asarray(reference, dtype=float))
    return float(np.sqrt(((ref - np.asarray(sig, dtype=float)) ** 2).sum(axis=1)).min())


def novelty_reference(train_sigs: Sequence[np.ndarray]) -> np.ndarray:
    """Cluster centres of the training signatures (`LayoutClusters`, label-free k search).

    With fewer than 10 signatures there is nothing to cluster: the signatures are the reference.
    """
    if len(train_sigs) < 10:
        return np.vstack(train_sigs)
    return np.asarray(LayoutClusters.fit(list(train_sigs)).model.cluster_centers_)


def compute_novelty_table(
    doc_sigs: Mapping[str, np.ndarray | None],
    doc_fold: Mapping[str, int],
    reference_fn: Callable[[Sequence[np.ndarray]], np.ndarray] = novelty_reference,
    novelty_fn: Callable[[np.ndarray, np.ndarray], float] = layout_novelty,
) -> dict[int, dict[str, float]]:
    """``{held_fold: {doc_id: novelty}}`` for the cross-fit that holds `held_fold` out.

    For a doc of the held-out fold the reference is built from the OTHER folds' docs (never from
    held-out docs). For a doc of a fitting fold the reference is built from the remaining fitting
    folds only (an inner leave-one-fold-out), so the calibrator is trained on novelties computed
    the same way it will be applied: against layouts of docs from other suppliers. A doc without a
    signature gets NaN. No supplier ids and no labels are read.
    """
    folds = sorted(set(doc_fold.values()))
    cache: dict[frozenset[int], np.ndarray | None] = {}

    def reference(fs: frozenset[int]) -> np.ndarray | None:
        if fs not in cache:
            sigs = [s for d, s in doc_sigs.items() if s is not None and doc_fold.get(d) in fs]
            cache[fs] = reference_fn(sigs) if sigs else None
        return cache[fs]

    table: dict[int, dict[str, float]] = {}
    for held in folds:
        fit_folds = [f for f in folds if f != held]
        per_doc: dict[str, float] = {}
        for d, s in doc_sigs.items():
            f = doc_fold.get(d)
            if f is None:
                continue
            ref_folds = frozenset(fit_folds) if f == held else frozenset(fit_folds) - {f}
            ref = reference(ref_folds)
            per_doc[d] = np.nan if s is None or ref is None else novelty_fn(s, ref)
        table[held] = per_doc
    return table


# ---------------------------------------------------------------------------------------------
# Calibrators
# ---------------------------------------------------------------------------------------------


def design_matrix(table: FieldTable, novelty: Mapping[str, float] | None = None) -> np.ndarray:
    """`BASE_FEATURES` + layout novelty + field-type one-hot, columns as `DESIGN_FEATURES`."""
    n = len(table.keys)
    nov = np.full((n, 1), np.nan)
    if novelty is not None:
        nov[:, 0] = [novelty.get(k.doc_id, np.nan) for k in table.keys]
    onehot = np.zeros((n, len(ALL_FIELDS)))
    col = {f: i for i, f in enumerate(ALL_FIELDS)}
    for i, k in enumerate(table.keys):
        onehot[i, col[f"{k.scope}.{k.field}"]] = 1.0
    return np.hstack([table.X, nov, onehot])


class ConstantModel:
    """Prior-probability fallback when a fit set lacks one of the classes."""

    def __init__(self, p: float) -> None:
        self.p = p

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """P(y=1) is the constant prior for every row."""
        return np.column_stack([np.full(len(X), 1 - self.p), np.full(len(X), self.p)])


def make_model(kind: str) -> Pipeline | HistGradientBoostingClassifier:
    """The calibrator: ``lr`` (impute, standardise, plain logistic) or ``gbm`` (shallow HGB)."""
    if kind == "lr":
        return make_pipeline(
            # keep_empty_features: a column that is NaN in every fitting row (e.g. no logprobs in a
            # whole run) becomes a constant 0 instead of vanishing, so the matrix width is stable.
            SimpleImputer(strategy="mean", keep_empty_features=True),
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2000, random_state=SEED),
        )
    if kind == "gbm":
        # Fixed small capacity and no early stopping (it would hold out a random validation
        # split): the fit is a deterministic function of the data.
        return HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=20,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=SEED,
        )
    raise ValueError(f"unknown calibrator kind {kind!r}; expected 'lr' or 'gbm'")


def fit_predict(kind: str, X_fit: np.ndarray, y_fit: np.ndarray, X_pred: np.ndarray) -> np.ndarray:
    """Fit a calibrator and return P(y=1) on `X_pred` (prior if a class is (nearly) missing)."""
    pos = int(y_fit.sum())
    if pos < MIN_PER_CLASS or len(y_fit) - pos < MIN_PER_CLASS:
        return np.asarray(ConstantModel(pos / max(len(y_fit), 1)).predict_proba(X_pred)[:, 1])
    model = make_model(kind)
    if kind == "gbm":
        # HGB's binning raises on an all-NaN column (no logprobs in a run, no novelty): drop it.
        keep = ~np.isnan(X_fit).all(axis=0)
        X_fit, X_pred = X_fit[:, keep], X_pred[:, keep]
    model.fit(X_fit, y_fit.astype(int))
    return np.asarray(model.predict_proba(X_pred)[:, 1])


def fold_splits(
    doc_ids: Sequence[str],
    doc_fold: Mapping[str, int],
    groups: Mapping[str, str] | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Row-index ``(fit, predict)`` pairs of the supplier-fold cross-fit.

    Fit on every fold but one, predict the held-out one. Asserts that no document and (when
    `groups` is given) no supplier group is on both sides, and that every row is predicted once.
    """
    fold_of = np.array([doc_fold[d] for d in doc_ids])
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for h in sorted(set(fold_of.tolist())):
        fit, pred = np.flatnonzero(fold_of != h), np.flatnonzero(fold_of == h)
        assert not {doc_ids[i] for i in fit} & {doc_ids[i] for i in pred}, "doc in fit and predict"
        if groups is not None:
            gf = {groups[doc_ids[i]] for i in fit}
            gp = {groups[doc_ids[i]] for i in pred}
            assert not gf & gp, "supplier group in fit and predict"
        out.append((fit, pred))
    allp = np.concatenate([p for _, p in out])
    assert len(allp) == len(set(allp.tolist())) == len(doc_ids), "rows not predicted exactly once"
    return out


def cross_fit(
    table: FieldTable,
    y: np.ndarray,
    fit_mask: np.ndarray,
    doc_fold: Mapping[str, int],
    kind: str,
    novelty_table: Mapping[int, Mapping[str, float]] | None = None,
    groups: Mapping[str, str] | None = None,
) -> np.ndarray:
    """Out-of-fold P(y=1) for every row: fit on rows of the other folds where `fit_mask`.

    The novelty column of fold ``h``'s design comes from ``novelty_table[h]`` (reference from
    training folds only). Rows outside `fit_mask` are still predicted (callers mask them).
    """
    doc_ids = [k.doc_id for k in table.keys]
    oof = np.full(len(y), np.nan)
    for (fit, pred), h in zip(
        fold_splits(doc_ids, doc_fold, groups),
        sorted(set(doc_fold[d] for d in doc_ids)),
        strict=True,
    ):
        X = design_matrix(table, None if novelty_table is None else novelty_table[h])
        use = fit[fit_mask[fit]]
        oof[pred] = fit_predict(kind, X[use], y[use].astype(int), X[pred])
    return oof


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    """Mean binary log-loss (probabilities clipped to [1e-12, 1 - 1e-12])."""
    q = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))


@dataclass
class Selection:
    """Outcome of the LR-vs-GBM comparison for one calibrator."""

    kind: str  # the chosen model
    oof: np.ndarray
    logloss: dict[str, float]
    margin: float


def select_model(
    table: FieldTable,
    y: np.ndarray,
    fit_mask: np.ndarray,
    doc_fold: Mapping[str, int],
    novelty_table: Mapping[int, Mapping[str, float]] | None = None,
    groups: Mapping[str, str] | None = None,
    margin: float = GBM_MARGIN,
) -> Selection:
    """LR by default; GBM only when its cross-fitted log-loss is lower by at least `margin`.

    Both models see the same folds and features; the comparison is on the rows in `fit_mask`.
    The choice is made on the same out-of-fold predictions that are then reported, which is a mild
    optimism for a single binary choice behind a conservative margin (stated in the report).
    """
    oofs = {
        k: cross_fit(table, y, fit_mask, doc_fold, k, novelty_table, groups) for k in ("lr", "gbm")
    }
    ll = {k: log_loss(v[fit_mask], y[fit_mask].astype(float)) for k, v in oofs.items()}
    kind = "gbm" if ll["lr"] - ll["gbm"] >= margin else "lr"
    return Selection(kind, oofs[kind], ll, margin)


@dataclass
class OofResult:
    """Out-of-fold probabilities of both calibrators (NaN of `p_correct` where not emitted)."""

    p_correct: np.ndarray
    p_null: np.ndarray
    sel_correct: Selection
    sel_null: Selection


def calibrate_oof(
    table: FieldTable,
    y_correct: np.ndarray,
    y_null: np.ndarray,
    doc_fold: Mapping[str, int],
    novelty_table: Mapping[int, Mapping[str, float]] | None = None,
    groups: Mapping[str, str] | None = None,
    margin: float = GBM_MARGIN,
) -> OofResult:
    """Cross-fit (a) P(correct | emitted non-null) on emitted rows, (b) P(gold null) on all rows."""
    sa = select_model(table, y_correct, table.emitted, doc_fold, novelty_table, groups, margin)
    sb = select_model(
        table, y_null, np.ones(len(y_null), dtype=bool), doc_fold, novelty_table, groups, margin
    )
    p_correct = np.where(table.emitted, sa.oof, np.nan)
    return OofResult(p_correct, sb.oof, sa, sb)


# ---------------------------------------------------------------------------------------------
# Null policy
# ---------------------------------------------------------------------------------------------


def null_decisions(p_correct: np.ndarray, p_null: np.ndarray, emitted: np.ndarray) -> np.ndarray:
    """True where an emitted value is replaced by null: ``P(gold null) > P(value correct)``.

    Expected-score optimal: nulling scores 1 with probability P(gold null), keeping the value
    scores 1 with probability P(correct). Rows that were not emitted are never touched (the policy
    only removes values), and a NaN probability never triggers a null.
    """
    with np.errstate(invalid="ignore"):
        return np.asarray(emitted & (p_null > p_correct))


def apply_null_policy(
    docs: Mapping[str, Mapping[str, Any]],
    keys: Sequence[FieldKey],
    nulled: np.ndarray,
    scopes: Sequence[str] = ("header", "row"),
) -> dict[str, dict[str, Any]]:
    """Copy of `docs` with the decided fields set to None (only `scopes`); asserts no fill."""
    out = copy.deepcopy({d: dict(v) for d, v in docs.items()})
    for k, hit in zip(keys, nulled, strict=True):
        if not hit or k.scope not in scopes:
            continue
        doc = out[k.doc_id]
        if k.scope == "header":
            doc["header"][k.field] = None
        else:
            rows = [r for r in doc["line_items"] if isinstance(r, dict)]
            rows[k.row_idx][k.field] = None
    assert_never_fills(docs, out)
    return out


def assert_never_fills(
    before: Mapping[str, Mapping[str, Any]], after: Mapping[str, Mapping[str, Any]]
) -> None:
    """Raise AssertionError unless every value in `after` is its `before` value or None.

    Also requires the same documents, header keys and row counts: the policy may only null.
    """
    assert set(before) == set(after), "documents added or removed"
    for d, b in before.items():
        a = after[d]
        bh, ah = b.get("header") or {}, a.get("header") or {}
        assert set(bh) == set(ah), f"{d}: header keys changed"
        for f, v in ah.items():
            assert _blank(v) or v == bh[f], f"{d}.{f}: value filled or changed"
        br = [r for r in (b.get("line_items") or []) if isinstance(r, dict)]
        ar = [r for r in (a.get("line_items") or []) if isinstance(r, dict)]
        assert len(br) == len(ar), f"{d}: row count changed"
        for rb, ra in zip(br, ar, strict=True):
            assert set(rb) == set(ra), f"{d}: row keys changed"
            for f, v in ra.items():
                assert _blank(v) or v == rb[f], f"{d}.{f}: row value filled or changed"


def false_fill_split(
    pred: Mapping[str, Mapping[str, Any]],
    gold: Mapping[str, Mapping[str, Any]],
    meta: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    """Header false fills split by why gold is null (scorer's definition: gold null, value given).

    ``absent_line``: invoice ``awb_number`` / waybill ``hawb`` null with the doc's ``awb_absent`` /
    ``hawb_absent`` meta tag (the line is not printed). ``redaction``: every other gold header
    null. Returns ``{category: {"null_fields": n, "filled": k}}``.
    """
    out = {c: {"null_fields": 0, "filled": 0} for c in ("redaction", "absent_line")}
    for d, g in gold.items():
        m = meta.get(d, {})
        p = pred.get(d) or {}
        ph = p.get("header") if isinstance(p.get("header"), dict) else {}
        fields = HEADER_FIELDS.get(g["doc_type"], ())
        for f in fields:
            if not _blank(g["header"].get(f)):
                continue
            absent = (f == "awb_number" and bool(m.get("awb_absent"))) or (
                f == "hawb" and bool(m.get("hawb_absent"))
            )
            cat = out["absent_line" if absent else "redaction"]
            cat["null_fields"] += 1
            cat["filled"] += int(not _blank(ph.get(f)))
    return out


# ---------------------------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------------------------


def reliability(p: np.ndarray, y: np.ndarray, bins: int = ECE_BINS) -> list[dict[str, float]]:
    """Equal-width reliability table: per bin n, mean confidence, observed accuracy."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    out = []
    for b in range(bins):
        sel = idx == b
        out.append(
            {
                "lo": float(edges[b]),
                "hi": float(edges[b + 1]),
                "n": int(sel.sum()),
                "mean_conf": float(p[sel].mean()) if sel.any() else float("nan"),
                "accuracy": float(y[sel].mean()) if sel.any() else float("nan"),
            }
        )
    return out


def ece(p: np.ndarray, y: np.ndarray, bins: int = ECE_BINS, scheme: str = "width") -> float:
    """Expected calibration error: ``sum_b n_b/n * |acc_b - conf_b|``.

    ``scheme="width"`` uses `bins` equal-width bins on [0, 1]; ``"mass"`` uses `bins` bins of equal
    count (sorted by confidence). NaN for an empty input.
    """
    if len(p) == 0:
        return float("nan")
    if scheme == "width":
        rows = [r for r in reliability(p, y, bins) if r["n"]]
        return float(sum(r["n"] / len(p) * abs(r["accuracy"] - r["mean_conf"]) for r in rows))
    if scheme == "mass":
        order = np.argsort(p, kind="stable")
        parts = [c for c in np.array_split(order, bins) if len(c)]
        return float(sum(len(c) / len(p) * abs(y[c].mean() - p[c].mean()) for c in parts))
    raise ValueError(f"unknown ECE scheme {scheme!r}; expected 'width' or 'mass'")


def coverage_curve(
    conf: np.ndarray, correct: np.ndarray, points: int = 50
) -> list[dict[str, float]]:
    """Accuracy of the most confident ``coverage`` share of fields, at `points` coverage levels."""
    order = np.argsort(-conf, kind="stable")
    c = correct[order].astype(float)
    cum = np.cumsum(c)
    n = len(conf)
    out = []
    for q in np.linspace(1 / points, 1.0, points):
        k = max(1, int(round(q * n)))
        out.append(
            {
                "coverage": k / n,
                "accuracy": float(cum[k - 1] / k),
                "min_conf": float(conf[order][k - 1]),
            }
        )
    return out


def calibration_summary(p: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """n, base rate, log-loss, Brier, ECE (15 equal-width and 15 equal-mass bins)."""
    n = len(p)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "base_rate": float(np.mean(y)),
        "log_loss": log_loss(p, y.astype(float)),
        "brier": float(np.mean((p - y) ** 2)),
        "ece_width15": ece(p, y, ECE_BINS, "width"),
        "ece_mass15": ece(p, y, ECE_BINS, "mass"),
    }


# ---------------------------------------------------------------------------------------------
# Review flag
# ---------------------------------------------------------------------------------------------


def select_tau(
    conf: np.ndarray, correct: np.ndarray, target: float = PRECISION_TARGET
) -> float | None:
    """Lowest tau whose auto-accepted set ``{conf >= tau}`` has precision >= `target`.

    Candidates are the observed confidence values (ties are accepted together). Returns None when
    no candidate reaches the target: the caller must then flag every field (tau = +inf). Precision
    is not monotone in tau, so the lowest qualifying tau is the largest accepted set that meets
    the floor, found by a full scan rather than a bisection.
    """
    if len(conf) == 0:
        return None
    order = np.argsort(-conf, kind="stable")
    c = conf[order]
    ok = np.cumsum(correct[order].astype(float))
    n = np.arange(1, len(c) + 1)
    last_of_tie = np.append(c[1:] != c[:-1], True)  # accept whole tie groups only
    good = last_of_tie & (ok / n >= target)
    if not good.any():
        return None
    return float(c[np.flatnonzero(good)[-1]])


REVIEW_COLS = ("n", "n_acc", "n_acc_ok", "n_wrong", "n_wrong_flagged", "n_flagged", "doc_flagged")


def _per_doc_stats(
    conf: np.ndarray, correct: np.ndarray, doc_ids: np.ndarray, tau: float, docs: Sequence[str]
) -> np.ndarray:
    pos = {d: i for i, d in enumerate(docs)}
    stats = np.zeros((len(docs), len(REVIEW_COLS)))
    flagged = conf < tau
    for i, d in enumerate(doc_ids):
        r = stats[pos[d]]
        r[0] += 1
        r[1] += not flagged[i]
        r[2] += (not flagged[i]) and correct[i]
        r[3] += not correct[i]
        r[4] += (not correct[i]) and flagged[i]
        r[5] += flagged[i]
        r[6] = max(r[6], float(flagged[i]))
    return stats


def _review_ratios(s: np.ndarray, n_docs: np.ndarray) -> np.ndarray:
    """Metric vectors from summed stats (rows = resamples): precision, accepted_share,
    review_rate, error_recall, doc_flag_rate, field_accuracy. 0/0 gives NaN."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.stack(
            [
                s[:, 2] / s[:, 1],
                s[:, 1] / s[:, 0],
                s[:, 5] / s[:, 0],
                s[:, 4] / s[:, 3],
                s[:, 6] / n_docs,
                (s[:, 0] - s[:, 3]) / s[:, 0],
            ],
            axis=1,
        )


REVIEW_METRICS = (
    "precision_accepted",
    "accepted_share",
    "review_rate",
    "error_recall",
    "doc_flag_rate",
    "field_accuracy",
)


def review_with_ci(
    conf: np.ndarray,
    correct: np.ndarray,
    doc_ids: np.ndarray,
    tau: float | None,
    n_boot: int = N_BOOT,
    seed: int = SEED,
    alpha: float = 0.05,
) -> dict[str, dict[str, float]]:
    """Review-flag metrics at `tau` with document-level percentile bootstrap 95% CIs.

    Flagged = ``conf < tau`` (None -> +inf, flag everything). Metrics: precision of the
    auto-accepted fields, share auto-accepted, review rate (share of fields flagged), error recall
    (share of WRONG fields that are flagged), document flag rate (a doc is flagged if any of its
    fields is), field accuracy. Documents are resampled whole (same resamples for every metric);
    a resample with an empty denominator is skipped for that metric.
    """
    t = float("inf") if tau is None else tau
    docs = sorted(set(doc_ids.tolist()))
    stats = _per_doc_stats(conf, correct, doc_ids, t, docs)
    idx = np.random.Generator(np.random.PCG64(seed)).integers(
        0, len(docs), size=(n_boot, len(docs))
    )
    sums = stats[idx].sum(axis=1)
    draws = _review_ratios(sums, np.full(n_boot, float(len(docs))))
    point = _review_ratios(stats.sum(axis=0, keepdims=True), np.array([float(len(docs))]))[0]
    out: dict[str, dict[str, float]] = {}
    for j, name in enumerate(REVIEW_METRICS):
        col = draws[:, j]
        col = col[~np.isnan(col)]
        lo, hi = np.quantile(col, [alpha / 2, 1 - alpha / 2]) if len(col) else (np.nan, np.nan)
        out[name] = {"point": float(point[j]), "lo": float(lo), "hi": float(hi)}
    return out


# ---------------------------------------------------------------------------------------------
# Synthetic data with known generative truth (dry-run mechanics check)
# ---------------------------------------------------------------------------------------------


@dataclass
class SyntheticData:
    """A seeded synthetic table with the true probabilities that generated its labels."""

    table: FieldTable
    y_correct: np.ndarray
    y_null: np.ndarray
    doc_fold: dict[str, int]
    groups: dict[str, str]
    true_p_correct: np.ndarray
    true_p_null: np.ndarray


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.asarray(1.0 / (1.0 + np.exp(-z)))


def synthetic_data(n_docs: int = 300, fields_per_doc: int = 12, seed: int = SEED) -> SyntheticData:
    """Fields whose labels are drawn from a known logistic model of the features.

    For an emitted value: gold is null with ``P_n = sigmoid(0.5 - 4*ocr_fuzzy + 0.8*v_awb_bad)``,
    else the value is correct with ``P_c = sigmoid(-0.5 + 1.5*lp_min + 5*ocr_fuzzy)``; the true
    P(correct) is ``(1 - P_n) * P_c``. A field emitted as null is gold-null with probability 0.7.
    Docs are spread over 3 supplier folds (one group per fold-local supplier).
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    n = n_docs * fields_per_doc
    X = np.zeros((n, _N_BASE))
    X[:, _IDX["has_logprobs"]] = 1.0
    X[:, _IDX["lp_min"]] = -np.abs(rng.normal(0.4, 0.6, n))
    X[:, _IDX["lp_mean"]] = X[:, _IDX["lp_min"]] * 0.5
    X[:, _IDX["has_ocr"]] = 1.0
    X[:, _IDX["ocr_fuzzy"]] = rng.uniform(0, 1, n)
    X[:, _IDX["ocr_hit"]] = (X[:, _IDX["ocr_fuzzy"]] > 0.87).astype(float)
    X[:, _IDX["v_awb_bad"]] = (rng.uniform(0, 1, n) < 0.05).astype(float)
    X[:, _IDX["xp_single"]] = 1.0
    X[:, _IDX["xp_n_pages"]] = 1.0
    was_null = rng.uniform(0, 1, n) < 0.15
    X[:, _IDX["was_null"]] = was_null.astype(float)
    p_n = _sigmoid(0.5 - 4.0 * X[:, _IDX["ocr_fuzzy"]] + 0.8 * X[:, _IDX["v_awb_bad"]])
    p_c = _sigmoid(-0.5 + 1.5 * X[:, _IDX["lp_min"]] + 5.0 * X[:, _IDX["ocr_fuzzy"]])
    gold_null = np.where(was_null, rng.uniform(0, 1, n) < 0.7, rng.uniform(0, 1, n) < p_n)
    correct = (~gold_null) & (rng.uniform(0, 1, n) < p_c) & ~was_null
    names = [f for f in ALL_FIELDS if f.startswith("header.")]
    keys: list[FieldKey] = []
    doc_fold: dict[str, int] = {}
    groups: dict[str, str] = {}
    for d in range(n_docs):
        did = f"syn_{d:04d}"
        doc_fold[did] = d % 3
        groups[did] = f"g{d % 3}_{(d // 3) % 5}"
        for j in range(fields_per_doc):
            sc, nm = names[j % len(names)].split(".", 1)
            keys.append(FieldKey(did, sc, nm, -1))
    table = FieldTable(keys, X, ~was_null)
    true_pc = np.where(was_null, np.nan, (1 - p_n) * p_c)
    true_pn = np.where(was_null, 0.7, p_n)
    return SyntheticData(
        table, correct.astype(bool), gold_null.astype(bool), doc_fold, groups, true_pc, true_pn
    )
