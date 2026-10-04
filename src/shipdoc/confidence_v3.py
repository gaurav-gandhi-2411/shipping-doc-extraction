"""Calibration v3 PREP: a fine-tuned-vs-zero-shot agreement feature and its AUROC lift.

Built beside ``shipdoc.confidence`` / ``shipdoc.confidence_v2`` (both untouched and reused: labels,
v2 feature builder, tie-aware AUROC with document bootstrap, nested tau, fold splits). New here:

1. **Agreement features** (`agreement_matrix`): per field of one arm's POST-RULE output, compared
   with the other arm's post-rule output on the same document. Header fields pair by name; row
   fields pair through a deterministic, label-free alignment (`align_rows`) that mirrors the
   scorer's two greedy passes between the two predictions (no gold) and adds a fuzzy and a
   positional pass. Pure functions, no gold, no supplier id.
2. **Paired AUROC lift** (`paired_auroc_boot`): AUROC(with agreement) - AUROC(without), both on the
   same rows, document-level bootstrap with shared resamples.
3. **Cross-fit machinery** over an arbitrary design matrix (`cross_fit_design`,
   `select_design`): the same fit-on-others / predict-held-out logic as v2, generalised so the
   design can carry the agreement columns. `inner_group_folds` builds supplier-grouped folds
   INSIDE one fold's documents (single-fold exploratory measurement).
4. **Gate for the full 3-fold version** (`check_fold_coverage`): refuses unless the OOF runs
   cover folds 0, 1, 2 exactly once each and their documents are the 500 train + dev documents.

Similarity function (documented once, used everywhere): both values non-blank, `alnum_key` =
casefold with every non-alphanumeric character removed; ``sim = rapidfuzz.fuzz.ratio(ka, kb) / 100``
(normalised Indel similarity in [0, 1], the same family as ``shipdoc.locate.best_fuzzy_score``);
1.0 when `norm_equal`. Null on either side: 0 (the null patterns have their own indicators).

Fail closed on test data: every public entry that takes document ids calls
``confidence_v2.assert_no_test_ids``. No function here reads or prints an extracted value except
to compute a number from it.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from rapidfuzz import fuzz
from sklearn.model_selection import GroupKFold

from shipdoc import confidence as cf
from shipdoc import confidence_v2 as c2
from shipdoc.cluster import layout_signature
from shipdoc.extract import ROW_KEYS
from shipdoc.ocr import PageOcr

SEED = cf.SEED
N_BOOT = cf.N_BOOT
#: Scorer ``KIND == "num"`` fields (``assignment/score.py``); duplicated so the module needs no
#: scorer file. A test compares it with the scorer's table when the scorer is present.
NUM_FIELDS = frozenset({"total_amount", "quantity", "pieces", "gross_weight_kg"})
#: Scorer tolerance for numbers (``abs(a - b) < 0.005``).
NUM_TOL = 0.005
#: Minimum similarity of the fuzzy supplier-part-number pass of the row alignment. 0.8 = a
#: 4-in-20-character difference; below it a shared part number is not credible.
FUZZY_ALIGN_MIN = 0.8
MIN_ERRORS = 5
#: Cross-fit folds inside one fold's documents are ``min(INNER_K_MAX, n supplier groups)``.
INNER_K_MAX = 5

AGREE_FEATURES: tuple[str, ...] = (
    "ag_exact",
    "ag_norm",
    "ag_sim",
    "ag_both_null",
    "ag_self_null_other_val",
    "ag_self_val_other_null",
    "ag_row_unaligned",
    "ag_pos_aligned",
    "ag_type_differs",
    "ag_doc_rate",
    "ag_doc_unaligned_frac",
)
_F = {n: i for i, n in enumerate(AGREE_FEATURES)}
#: Variant names of the three calibrators compared per field group.
VARIANTS: tuple[str, ...] = ("v2", "v2_agree", "agree")


# ---------------------------------------------------------------------------------------------
# Value comparison
# ---------------------------------------------------------------------------------------------


def alnum_key(v: Any) -> str:
    """Casefolded value with every non-alphanumeric character removed ('' for a blank value)."""
    if cf._blank(v):
        return ""
    return re.sub(r"[\W_]+", "", str(v).casefold())


def plain_number(v: Any) -> float | None:
    """The scorer's plain-number parse (commas and spaces removed); None if not a finite number."""
    try:
        x = float(re.sub(r"[,\s]", "", str(v)))
    except ValueError:
        return None
    return x if math.isfinite(x) else None


def norm_equal(field: str, a: Any, b: Any) -> bool:
    """Both values non-blank and equal after normalisation.

    Number fields: both parse (`plain_number`) and differ by less than `NUM_TOL`; if either does
    not parse, the alphanumeric keys are compared. Other fields: equal `alnum_key`; when both keys
    are empty (punctuation-only values) the stripped casefolded strings are compared instead.
    """
    if cf._blank(a) or cf._blank(b):
        return False
    if field in NUM_FIELDS:
        na, nb = plain_number(a), plain_number(b)
        if na is not None and nb is not None:
            return abs(na - nb) < NUM_TOL
    ka, kb = alnum_key(a), alnum_key(b)
    if not ka and not kb:
        return str(a).strip().casefold() == str(b).strip().casefold()
    return ka == kb


def similarity(field: str, a: Any, b: Any) -> float:
    """Fuzzy similarity in [0, 1]; see the module docstring. 0 when either value is blank."""
    if cf._blank(a) or cf._blank(b):
        return 0.0
    if norm_equal(field, a, b):
        return 1.0
    return float(fuzz.ratio(alnum_key(a), alnum_key(b))) / 100.0


def pair_features(field: str, a: Any, b: Any) -> tuple[float, float, float, float, float, float]:
    """``(exact, norm, sim, both_null, self_null_other_val, self_val_other_null)`` of one pair.

    ``exact`` is string equality after stripping; ``exact`` / ``norm`` / ``sim`` are 0 unless both
    values are non-blank (the null patterns are the last three indicators).
    """
    a_null, b_null = cf._blank(a), cf._blank(b)
    if a_null or b_null:
        return (0.0, 0.0, 0.0, float(a_null and b_null), float(a_null and not b_null),
                float(b_null and not a_null))  # fmt: skip
    exact = float(str(a).strip() == str(b).strip())
    return exact, float(norm_equal(field, a, b)), similarity(field, a, b), 0.0, 0.0, 0.0


# ---------------------------------------------------------------------------------------------
# Row alignment between two predictions (label-free)
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RowPair:
    """One aligned row pair: indices into the two arms' dict-only ``line_items`` lists.

    ``stage``: 1 = every row field agrees (normalised, a blank on both sides agrees), 2 = supplier
    part numbers normalised-equal, 3 = supplier part numbers fuzzy-similar (>= `FUZZY_ALIGN_MIN`),
    4 = same original position (both rows still free).
    """

    ft: int
    zs: int
    stage: int


def dict_rows(doc: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The dict-only ``line_items`` of a document (the indexing every feature builder uses)."""
    return [r for r in (doc.get("line_items") or []) if isinstance(r, dict)]


def _rows_agree(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    return all(
        (cf._blank(a.get(k)) and cf._blank(b.get(k))) or norm_equal(k, a.get(k), b.get(k))
        for k in ROW_KEYS
    )


def align_rows(
    ft_rows: Sequence[Mapping[str, Any]],
    zs_rows: Sequence[Mapping[str, Any]],
    fuzzy_min: float = FUZZY_ALIGN_MIN,
) -> list[RowPair]:
    """Deterministic one-to-one pairing of the rows of the two arms, no gold.

    Orientation is fixed (zero-shot rows play the scorer's "gold" outer loop, fine-tuned rows the
    "pred" inner loop) so both judging views share one alignment. Passes: (1) rows whose four
    fields all agree after normalisation (scorer pass 1), (2) supplier part numbers
    normalised-equal (scorer pass 2; a blank part number never pairs here), (3) supplier part
    numbers fuzzy-similar >= `fuzzy_min`, best similarity first, ties by (zero-shot, fine-tuned)
    index, (4) same original index with both rows still free. Rows left over are unaligned.
    Result sorted by fine-tuned index.
    """
    free_f, free_z = list(range(len(ft_rows))), list(range(len(zs_rows)))
    pairs: list[RowPair] = []

    def take(fi: int, zi: int, stage: int) -> None:
        free_f.remove(fi)
        free_z.remove(zi)
        pairs.append(RowPair(fi, zi, stage))

    for zi in list(free_z):  # pass 1
        for fi in free_f:
            if _rows_agree(ft_rows[fi], zs_rows[zi]):
                take(fi, zi, 1)
                break
    spn = "supplier_part_number"
    for zi in list(free_z):  # pass 2
        for fi in free_f:
            if norm_equal(spn, ft_rows[fi].get(spn), zs_rows[zi].get(spn)):
                take(fi, zi, 2)
                break
    cands = [
        (similarity(spn, ft_rows[fi].get(spn), zs_rows[zi].get(spn)), zi, fi)
        for zi in free_z
        for fi in free_f
    ]
    good = sorted((c for c in cands if c[0] >= fuzzy_min), key=lambda c: (-c[0], c[1], c[2]))
    for _sim, zi, fi in good:
        if fi in free_f and zi in free_z:  # pass 3
            take(fi, zi, 3)
    for i in range(min(len(ft_rows), len(zs_rows))):  # pass 4
        if i in free_f and i in free_z:
            take(i, i, 4)
    return sorted(pairs, key=lambda p: p.ft)


def align_all(
    ft_post: Mapping[str, Any], zs_post: Mapping[str, Any], doc_ids: Sequence[str]
) -> dict[str, list[RowPair]]:
    """`align_rows` of every document (post-rule predictions of the two arms)."""
    c2.assert_no_test_ids(doc_ids)
    return {d: align_rows(dict_rows(ft_post[d]), dict_rows(zs_post[d])) for d in doc_ids}


# ---------------------------------------------------------------------------------------------
# Agreement features
# ---------------------------------------------------------------------------------------------


def _header(doc: Mapping[str, Any]) -> Mapping[str, Any]:
    h = doc.get("header")
    return h if isinstance(h, dict) else {}


def doc_agreement(
    keys: Sequence[cf.FieldKey],
    self_doc: Mapping[str, Any],
    other_doc: Mapping[str, Any],
    partner: Mapping[int, int],
    pos_only: frozenset[int] = frozenset(),
) -> np.ndarray:
    """``(len(keys), len(AGREE_FEATURES))`` agreement of one document's fields, "self" vs "other".

    `keys` are the field keys of the SELF arm's feature rows; `partner` maps a self row index to
    its other-arm row index (`align_rows`, in the orientation of the view), `pos_only` = self
    rows aligned only by position. A row field of an unaligned row has ``ag_row_unaligned = 1``
    and zeros elsewhere. ``ag_doc_rate`` = share of compared fields (header fields of the self
    arm's doc type plus the fields of aligned rows) that are normalised-equal or null on both
    sides (NaN if none); ``ag_doc_unaligned_frac`` = share of self rows without a partner.
    """
    out = np.zeros((len(keys), len(AGREE_FEATURES)))
    s_hdr, o_hdr = _header(self_doc), _header(other_doc)
    s_rows, o_rows = dict_rows(self_doc), dict_rows(other_doc)
    differs = float(str(self_doc.get("doc_type")) != str(other_doc.get("doc_type")))
    n_cmp, n_agree = 0, 0.0
    for i, k in enumerate(keys):
        v = out[i]
        v[_F["ag_type_differs"]] = differs
        if k.scope == "header":
            pf = pair_features(k.field, s_hdr.get(k.field), o_hdr.get(k.field))
        else:
            j = partner.get(k.row_idx)
            if j is None:
                v[_F["ag_row_unaligned"]] = 1.0
                continue
            v[_F["ag_pos_aligned"]] = float(k.row_idx in pos_only)
            pf = pair_features(k.field, s_rows[k.row_idx].get(k.field), o_rows[j].get(k.field))
        v[:6] = pf
        n_cmp += 1
        n_agree += float(pf[1] == 1.0 or pf[3] == 1.0)
    rate = n_agree / n_cmp if n_cmp else float("nan")
    unal = (len(s_rows) - len(partner)) / len(s_rows) if s_rows else 0.0
    out[:, _F["ag_doc_rate"]] = rate
    out[:, _F["ag_doc_unaligned_frac"]] = unal
    return out


def agreement_matrix(
    keys: Sequence[cf.FieldKey],
    self_post: Mapping[str, Any],
    other_post: Mapping[str, Any],
    pairs: Mapping[str, Sequence[RowPair]],
    self_is_ft: bool,
) -> np.ndarray:
    """Agreement features aligned with `keys` (rows of several documents, any order).

    `self_is_ft` says which arm `keys` belong to, so the (ft, zs) pairs of `align_all` are
    oriented correctly for the zero-shot-judging view.
    """
    c2.assert_no_test_ids(sorted({k.doc_id for k in keys}))
    out = np.zeros((len(keys), len(AGREE_FEATURES)))
    by_doc: dict[str, list[int]] = {}
    for i, k in enumerate(keys):
        by_doc.setdefault(k.doc_id, []).append(i)
    for d, idx in by_doc.items():
        ps = pairs[d]
        partner = {(p.ft if self_is_ft else p.zs): (p.zs if self_is_ft else p.ft) for p in ps}
        pos = frozenset((p.ft if self_is_ft else p.zs) for p in ps if p.stage == 4)
        out[idx] = doc_agreement([keys[i] for i in idx], self_post[d], other_post[d], partner, pos)
    return out


def build_table_v2(
    doc_ids: Sequence[str],
    post: Mapping[str, Any],
    traces: Mapping[str, Any],
    ocr: Mapping[str, Sequence[PageOcr]],
    changes: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[c2.TableV2, dict[str, np.ndarray | None]]:
    """v2 feature table of one arm's post-rule output plus the page-1 layout signatures."""
    c2.assert_no_test_ids(doc_ids)
    rows: list[cf.FieldRow] = []
    extras: list[np.ndarray] = []
    sigs: dict[str, np.ndarray | None] = {}
    for d in doc_ids:
        pages = list(ocr.get(d) or [])
        r = cf.build_doc_features(d, post[d], traces[d], pages)
        rows.extend(r)
        extras.append(c2.build_extras(r, post[d], traces[d], pages, changes.get(d, [])))
        sigs[d] = layout_signature(pages[0]) if pages else None
    return c2.TableV2(cf.assemble(rows), np.vstack(extras)), sigs


# ---------------------------------------------------------------------------------------------
# Folds and cross-fit over an arbitrary design
# ---------------------------------------------------------------------------------------------


def inner_group_folds(
    doc_ids: Sequence[str], groups: Mapping[str, str], k_max: int = INNER_K_MAX
) -> tuple[dict[str, int], int, int]:
    """Supplier-grouped folds INSIDE a document set: ``(doc -> fold, k, n_groups)``.

    ``k = min(k_max, n_groups)``; sklearn ``GroupKFold`` without shuffling over the sorted
    document ids is a deterministic function of the groups. Raises if fewer than 2 groups.
    """
    c2.assert_no_test_ids(doc_ids)
    ids = sorted(set(doc_ids))
    grp = [groups[d] for d in ids]
    n_groups = len(set(grp))
    k = min(k_max, n_groups)
    if k < 2:
        raise ValueError(f"need at least 2 supplier groups to cross-fit, got {n_groups}")
    out: dict[str, int] = {}
    for f, (_, va) in enumerate(GroupKFold(n_splits=k).split(ids, groups=grp)):
        out.update({ids[j]: f for j in va})
    return out, k, n_groups


FitFn = Callable[[str, np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def cross_fit_design(
    design_fn: Callable[[int], np.ndarray],
    y: np.ndarray,
    fit_mask: np.ndarray,
    doc_ids: Sequence[str],
    doc_fold: Mapping[str, int],
    groups: Mapping[str, str] | None,
    kind: str,
    types: np.ndarray | None = None,
    fit_fn: FitFn = cf.fit_predict,
) -> np.ndarray:
    """Out-of-fold P(y=1) of every row for a design ``design_fn(held fold) -> X``.

    Fit on the rows (where `fit_mask`) of every other fold, predict the held-out fold
    (`cf.fold_splits`, which asserts that no document and no supplier group is on both sides).
    With `types` (one entry per row) one model is fitted per field type (``per_type``).
    """
    c2.assert_no_test_ids(doc_ids)
    oof = np.full(len(y), np.nan)
    folds = sorted({doc_fold[d] for d in doc_ids})
    for (fit, pred), h in zip(cf.fold_splits(doc_ids, doc_fold, groups), folds, strict=True):
        X = design_fn(h)
        use = fit[fit_mask[fit]]
        if types is None:
            oof[pred] = fit_fn(kind, X[use], y[use].astype(int), X[pred])
            continue
        for t in c2.FIELD_TYPES:
            u, p = use[types[use] == t], pred[types[pred] == t]
            if len(p):
                oof[p] = fit_fn(kind, X[u], y[u].astype(int), X[p])
    return oof


def select_design(
    design_fn: Callable[[int], np.ndarray],
    y: np.ndarray,
    emitted: np.ndarray,
    doc_ids: Sequence[str],
    doc_fold: Mapping[str, int],
    groups: Mapping[str, str] | None,
    types: np.ndarray,
    structures: Sequence[str] = ("pooled", "per_type"),
    margin: float = cf.GBM_MARGIN,
) -> c2.StructureChoice:
    """The v2 selection rule on one design: per structure LR unless GBM wins by `margin` nats of
    emitted OOF log-loss; then the structure with the strictly lower log-loss (ties: the first
    listed, ``pooled``). With ``structures=("pooled",)`` only the LR-vs-GBM choice is made."""
    yf = y.astype(float)
    ll: dict[str, float] = {}
    oofs: dict[tuple[str, str], np.ndarray] = {}
    kinds: dict[str, str] = {}
    for st in structures:
        for k in ("lr", "gbm"):
            o = cross_fit_design(
                design_fn, y, emitted, doc_ids, doc_fold, groups, k,
                types if st == "per_type" else None,
            )  # fmt: skip
            oofs[(st, k)] = o
            ll[f"{st}/{k}"] = cf.log_loss(o[emitted], yf[emitted])
        kinds[st] = "gbm" if ll[f"{st}/lr"] - ll[f"{st}/gbm"] >= margin else "lr"
    pick = structures[0]
    for st in structures[1:]:
        if ll[f"{st}/{kinds[st]}"] < ll[f"{pick}/{kinds[pick]}"]:
            pick = st
    allo = {f"{s}/{kd}": np.where(emitted, o, np.nan) for (s, kd), o in oofs.items()}
    return c2.StructureChoice(
        pick, kinds[pick], np.where(emitted, oofs[(pick, kinds[pick])], np.nan), ll, kinds, allo
    )


def design_variants(
    tv: c2.TableV2, agree: np.ndarray, novelty: Mapping[str, float] | None
) -> dict[str, np.ndarray]:
    """The three design matrices: ``v2`` (all v2 features), ``v2_agree`` (v2 + agreement),
    ``agree`` (field-type one-hot + agreement only)."""
    base = c2.design_v2(tv, novelty)
    n0 = len(cf.BASE_FEATURES) + 1  # base features + layout novelty, then the field-type one-hot
    onehot = base[:, n0 : n0 + len(cf.ALL_FIELDS)]
    return {"v2": base, "v2_agree": np.hstack([base, agree]), "agree": np.hstack([onehot, agree])}


# ---------------------------------------------------------------------------------------------
# Paired AUROC bootstrap
# ---------------------------------------------------------------------------------------------


def _boot_draws(
    score: np.ndarray, y: np.ndarray, inv: np.ndarray, counts: np.ndarray, chunk: int
) -> tuple[float, np.ndarray]:
    """``(point AUROC, AUROC of each resample)``; `counts` is ``(n_boot, n_docs)`` multiplicity."""
    order = np.argsort(score, kind="stable")
    s, yy, inv_s = score[order], y[order].astype(float), inv[order]
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    point = float(c2._auc_weighted(yy, starts, np.ones((1, len(s))))[0])
    draws = [
        c2._auc_weighted(yy, starts, counts[a : a + chunk][:, inv_s].astype(float))
        for a in range(0, len(counts), chunk)
    ]
    return point, np.concatenate(draws)


def paired_auroc_boot(
    s_base: np.ndarray,
    s_new: np.ndarray,
    y: np.ndarray,
    doc_ids: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = SEED,
    chunk: int = 100,
) -> dict[str, float]:
    """AUROC(new) - AUROC(base) with a paired document-level percentile bootstrap 95% CI.

    Both scores are evaluated on the same resamples of whole documents (the resample indices are
    those of `shipdoc.confidence_v2.auroc_boot` for the same seed). Resamples where one class is
    absent are dropped. Returns ``{point, lo, hi, base, new}``; NaN when one class only.
    """
    nan = float("nan")
    out = {"point": nan, "lo": nan, "hi": nan, "base": nan, "new": nan}
    if len(y) == 0 or (y == 1).sum() == 0 or (y == 0).sum() == 0:
        return out
    docs, inv = np.unique(doc_ids, return_inverse=True)
    nd = len(docs)
    idx = np.random.Generator(np.random.PCG64(seed)).integers(0, nd, size=(n_boot, nd))
    counts = np.stack([np.bincount(r, minlength=nd) for r in idx])
    p0, d0 = _boot_draws(s_base, y, inv, counts, chunk)
    p1, d1 = _boot_draws(s_new, y, inv, counts, chunk)
    out.update(point=p1 - p0, base=p0, new=p1)
    diff = (d1 - d0)[~(np.isnan(d0) | np.isnan(d1))]
    if len(diff):
        out["lo"], out["hi"] = (float(x) for x in np.quantile(diff, [0.025, 0.975]))
    return out


def _ci(d: Mapping[str, float]) -> dict[str, float]:
    return {"point": d["point"], "lo": d["lo"], "hi": d["hi"]}


# ---------------------------------------------------------------------------------------------
# Per-view analysis
# ---------------------------------------------------------------------------------------------

GROUPS: tuple[str, ...] = ("header_all", *(f"row.{f}" for f in ROW_KEYS))


def group_masks(tv: c2.TableV2) -> dict[str, np.ndarray]:
    """Emitted-field masks of ``header_all`` and each row field."""
    scope = np.array([k.scope for k in tv.keys])
    field = np.array([k.field for k in tv.keys])
    em = tv.emitted
    out = {"header_all": em & (scope == "header")}
    for f in ROW_KEYS:
        out[f"row.{f}"] = em & (scope == "row") & (field == f)
    return out


def lift_records(
    view: str,
    tv: c2.TableV2,
    y: np.ndarray,
    agree: np.ndarray,
    oof: Mapping[str, np.ndarray],
    n_boot: int,
) -> list[dict[str, Any]]:
    """One record per field group: n, errors, docs, base accuracy, AUROC of the three variants
    and of the agreement flag alone (``ag_norm``; an unaligned row counts as disagreeing) with
    document-bootstrap CIs, and the paired lift ``v2_agree - v2``."""
    ids = tv.table.doc_ids
    flag = agree[:, _F["ag_norm"]]
    recs: list[dict[str, Any]] = []
    for g, m in group_masks(tv).items():
        if not m.any():
            continue
        yy, d = y[m], ids[m]
        n_wrong = int((yy == 0).sum())
        rec: dict[str, Any] = {
            "view": view,
            "group": g,
            "n": int(m.sum()),
            "n_wrong": n_wrong,
            "docs": len(set(d.tolist())),
            "base_acc": float(yy.mean()),
            "note": f"n_wrong<{MIN_ERRORS}" if n_wrong < MIN_ERRORS else "",
        }
        for v in VARIANTS:
            rec[f"auroc_{v}"] = _ci(c2.auroc_boot(oof[v][m], yy, d, n_boot))
        rec["auroc_flag"] = _ci(c2.auroc_boot(flag[m], yy, d, n_boot))
        rec["lift"] = _ci(paired_auroc_boot(oof["v2"][m], oof["v2_agree"][m], yy, d, n_boot))
        recs.append(rec)
    return recs


def analyze_fold_view(
    view: str,
    tv: c2.TableV2,
    y: np.ndarray,
    agree: np.ndarray,
    groups: Mapping[str, str],
    sigs: Mapping[str, np.ndarray | None],
    n_boot: int = N_BOOT,
    fixed: tuple[str, str] | None = None,
) -> dict[str, Any]:
    """EXPLORATORY single-fold measurement of one judging view (cross-fit inside the fold).

    Supplier-grouped folds over the fold's own documents (`inner_group_folds`); layout novelty
    from the fitting inner folds only; pooled structure; LR-vs-GBM chosen ONCE on the ``v2`` design
    by the v2 margin rule and then used for all three variants. `fixed` = ``(structure, kind)``
    reuses the choice made on another view (no selection here). Returns the records plus the
    OOF probabilities (numbers only).
    """
    ids = tv.table.doc_ids.tolist()
    c2.assert_no_test_ids(ids)
    docs = sorted(set(ids))
    inner, k, n_groups = inner_group_folds(docs, groups)
    nov = cf.compute_novelty_table(dict(sigs), inner)
    cache: dict[int, dict[str, np.ndarray]] = {}

    def variants(h: int) -> dict[str, np.ndarray]:
        if h not in cache:
            cache[h] = design_variants(tv, agree, nov[h])
        return cache[h]

    types = tv.types()
    if fixed is None:
        choice = select_design(
            lambda h: variants(h)["v2"], y, tv.emitted, ids, inner, groups, types, ("pooled",)
        )
        kind, logloss = choice.kind, choice.logloss
    else:
        kind, logloss = fixed[1], {}
    oof = {
        v: cross_fit_design(lambda h, v=v: variants(h)[v], y, tv.emitted, ids, inner, groups, kind)
        for v in VARIANTS
    }
    return {
        "view": view,
        "mode": "fold_exploratory",
        "k": k,
        "n_groups": n_groups,
        "n_docs": len(docs),
        "structure": "pooled",
        "kind": kind,
        "emitted_oof_log_loss": logloss,
        "records": lift_records(view, tv, y, agree, oof, n_boot),
        "oof": oof,
    }


def tau_records(
    view: str,
    variant: str,
    p: np.ndarray,
    y: np.ndarray,
    tv: c2.TableV2,
    fold_of: np.ndarray,
    n_boot: int,
) -> list[dict[str, Any]]:
    """Nested per-field-type auto-accept at `c2.TARGETS` (tau chosen on the other folds only)."""
    ids, types = tv.table.doc_ids, tv.types()
    recs: list[dict[str, Any]] = []
    for t_name in c2.FIELD_TYPES:
        mask = tv.emitted & (types == t_name)
        for t in c2.TARGETS:
            acc, _, _ = c2.nested_accept(p, y, fold_of, mask, t)
            m = c2.accept_metrics(acc[mask], y[mask], ids[mask], n_boot)
            recs.append(
                {
                    "view": view, "variant": variant, "population": t_name, "target": t,
                    "n": int(mask.sum()),
                    **{k: m[k] for k in cf.REVIEW_METRICS},
                }
            )  # fmt: skip
    return recs


def analyze_full_view(
    view: str,
    tv: c2.TableV2,
    y: np.ndarray,
    agree: np.ndarray,
    doc_fold: Mapping[str, int],
    groups: Mapping[str, str],
    novelty: Mapping[int, Mapping[str, float]] | None,
    n_boot: int = N_BOOT,
    fixed: tuple[str, str] | None = None,
) -> dict[str, Any]:
    """Full version: the 3-supplier-fold cross-fit exactly like v2 (fit on two folds' fine-tuned
    OOF outputs, predict the third), structure / kind chosen once on the ``v2`` design by the v2
    rule, the same choice for all three variants, lift table, nested tau for ``v2`` and
    ``v2_agree``. `fixed` = ``(structure, kind)`` reuses the choice made on another view. Only
    call this with all three OOF folds (`check_fold_coverage`)."""
    ids = tv.table.doc_ids.tolist()
    c2.assert_no_test_ids(ids)
    cache: dict[int, dict[str, np.ndarray]] = {}

    def variants(h: int) -> dict[str, np.ndarray]:
        if h not in cache:
            cache[h] = design_variants(tv, agree, None if novelty is None else novelty[h])
        return cache[h]

    types = tv.types()
    if fixed is None:
        choice = select_design(
            lambda h: variants(h)["v2"], y, tv.emitted, ids, doc_fold, groups, types
        )
        structure, kind, logloss = choice.structure, choice.kind, choice.logloss
    else:
        (structure, kind), logloss = fixed, {}
    t_arg = types if structure == "per_type" else None
    oof = {
        v: cross_fit_design(
            lambda h, v=v: variants(h)[v], y, tv.emitted, ids, doc_fold, groups, kind, t_arg
        )
        for v in VARIANTS
    }
    fold_of = np.array([doc_fold[d] for d in ids])
    taus: list[dict[str, Any]] = []
    for v in ("v2", "v2_agree"):
        taus += tau_records(view, v, oof[v], y, tv, fold_of, n_boot)
    return {
        "view": view,
        "mode": "cross_fitted_3_fold",
        "structure": structure,
        "kind": kind,
        "emitted_oof_log_loss": logloss,
        "records": lift_records(view, tv, y, agree, oof, n_boot),
        "tau": taus,
        "oof": oof,
    }


# ---------------------------------------------------------------------------------------------
# Gate of the 3-fold mode
# ---------------------------------------------------------------------------------------------


class OofCoverageError(RuntimeError):
    """The OOF runs given do not cover the required folds / documents exactly (fail closed)."""


def check_fold_coverage(
    runs: Sequence[tuple[int, Sequence[str]]],
    folds: Mapping[str, Any],
    expected_ids: Sequence[str],
    required: Sequence[int],
) -> None:
    """Raise `OofCoverageError` unless `runs` = ``[(manifest oof.fold, trace doc ids), ...]`` is
    exactly one run per required fold, each run's documents are exactly that fold's held-out
    documents (no duplicates), and, when all three folds are required, the union is exactly
    `expected_ids` (the 500 train + dev documents). Test ids abort (`NoTestDataError`)."""
    for _, ids in runs:
        c2.assert_no_test_ids(list(ids))
    c2.assert_no_test_ids(list(expected_ids))
    got = sorted(f for f, _ in runs)
    if got != sorted(required):
        raise OofCoverageError(
            f"OOF run folds {got} (each once) do not equal the required folds {sorted(required)}"
        )
    held = {int(f["fold"]): set(f["val_doc_ids"]) for f in folds["folds"]}
    union: set[str] = set()
    for fold, ids in runs:
        ids = list(ids)
        if len(ids) != len(set(ids)):
            raise OofCoverageError(f"fold {fold}: duplicate document ids in the run")
        if set(ids) != held.get(fold):
            raise OofCoverageError(
                f"fold {fold}: run documents are not exactly the fold's held-out documents "
                f"({len(set(ids))} vs {len(held.get(fold, set()))})"
            )
        union |= set(ids)
    if sorted(required) == sorted(held) and union != set(expected_ids):
        raise OofCoverageError(
            f"union of the fold runs has {len(union)} documents, expected the "
            f"{len(set(expected_ids))} train + dev documents"
        )
