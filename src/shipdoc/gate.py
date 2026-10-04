"""Secondary system G: a per-field gate between FT + rules and ZS + rules (spec section 11 item 7).

Pure, typed, deterministic: no I/O, no model, no randomness, no gold. For every header field and
every aligned row field G emits the value of whichever arm has the higher calibrated P(correct);
it never writes a value neither arm produced (`GateError` if that would ever happen).

WHICH P(correct). The per-field probabilities come from the ``calibrate_v3`` agreement-feature
calibrators (variant ``v2_agree``: the v2 features of the judged arm plus the FT-vs-ZS agreement
features of ``shipdoc.confidence_v3``), one calibrator per judging view (judge_ft scores the FT
arm's emitted fields, judge_zs the ZS arm's). They are fitted ELSEWHERE (``scripts/gate_eval.py``
wires the cross-fit): the calibrator that scores a document is trained on OTHER SUPPLIER FOLDS
ONLY, never on the document it scores (`assert_fit_apply_disjoint` checks doc and supplier-group
disjointness; the cross-fit itself is ``confidence_v3.cross_fit_design``). This module only
consumes the resulting probabilities (`FieldProbs`, `probs_by_doc`).

THE RULES, one by one (each is a test):

1. Rows of the two arms are paired by ``confidence_v3.align_rows`` (reused, not reimplemented).
2. A slot that both arms fill with different values: the arm with the strictly higher P wins; a
   TIE goes to ZS + rules (the simpler system, spec section 11 item 1 tie rule).
3. A slot where both values are equal (exact string equality), or both are blank: no choice
   exists; the ZS value is written (identical anyway).
4. A slot filled by ONE arm only (the other is blank): the filled value is kept iff its P(correct)
   is >= `KEEP_MIN` (0.5), else the other arm's blank is written. DESIGN DECISION NOT IN THE SPEC:
   the spec defines the 0.5 rule for rows that exist in one arm only; the calibrators have no
   P(null) model (they score emitted fields only), so a blank has no P(correct) to compare with.
   The same >= 0.5 rule is applied to one-arm FIELDS (value vs blank in an aligned row or in the
   header) as the closest consistent reading of "select among the arms' values". GG must see this.
5. A row present in ONE arm only (`align_rows` left it unpaired): kept iff its row P >= 0.5, else
   dropped. DESIGN DECISION NOT IN THE SPEC (section 11.7 does not say how a row-level P is
   formed, and ``confidence_v3`` / ``calibrate_v3`` define none): the row P is the MINIMUM of the
   P(correct) of the row's emitted fields (a row is only as trustworthy as its least trustworthy
   cell; conservative). A row with no emitted field has no P and is dropped.
6. A document whose two arms disagree on ``doc_type`` has different header field sets and no
   calibrated P for the type itself: the ZS + rules document is written unchanged. DESIGN
   DECISION NOT IN THE SPEC; the count of such documents is reported (``doc_type_fallback``).
7. Output row order: aligned rows and ZS-only rows in ZS order, then FT-only rows in FT order
   (deterministic; the scorer pairs rows by content, not by position).

The output document has the production shape (``doc_type``, ``header`` with the arms' keys,
``line_items`` of dicts with the arms' keys); `gate_predictions` pushes the result through
``coerce_predictions`` / ``repair_predictions`` and fails if either changes a value (the inputs are
already post-rule, coerced and repaired, so both are the identity here).

All numbers any caller derives from this module are UNVERIFIED until a verifier recomputes them.
"""

from __future__ import annotations

import copy
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from shipdoc import confidence as cf
from shipdoc import confidence_v3 as c3
from shipdoc.coerce import coerce_predictions, repair_predictions
from shipdoc.g4pooled import FT_LABEL, ZS_LABEL, Clause

FT = "ft"
ZS = "zs"
#: One-arm rows and one-arm fields are kept iff their P(correct) is >= this (spec section 11.7).
KEEP_MIN = 0.5
COUNT_KEYS = ("false_fill_total", "over_null_total")
G_CLAUSE_NAMES = {
    "G1": "paired OVERALL delta (G - winner) CI lower bound strictly above 0",
    "G2": "false-fill count of G <= the winner's (raw counts, no tolerance)",
    "G3": "over-null count of G <= the winner's (raw counts, no tolerance)",
}

#: ``(scope, field, row_idx)`` of an arm's own field (row_idx -1 for header fields, else the index
#: into the arm's dict-only ``line_items``): `confidence.FieldKey` without the document id.
FieldId = tuple[str, str, int]
#: P(correct) of the EMITTED (non-blank) fields of one arm's document.
FieldProbs = Mapping[FieldId, float]


class GateError(RuntimeError):
    """A gate precondition or the no-synthesis invariant failed. Messages never quote a value."""


# --------------------------------------------------------------------------------------------
# Fit / apply separation and probability plumbing
# --------------------------------------------------------------------------------------------


def assert_fit_apply_disjoint(
    fit_docs: Sequence[str], apply_docs: Sequence[str], groups: Mapping[str, str]
) -> None:
    """Raise `GateError` unless no document and no supplier group is on both sides.

    A calibrator fitted on `fit_docs` may score `apply_docs` only if this passes: the gate is never
    fitted on the documents it is applied to. A document without a supplier group is an error
    (fail closed: an unknown group cannot be shown disjoint).
    """
    both = set(fit_docs) & set(apply_docs)
    if both:
        raise GateError(f"{len(both)} document(s) are in both the fit set and the apply set")
    unknown = [d for d in (*fit_docs, *apply_docs) if d not in groups]
    if unknown:
        raise GateError(f"{len(unknown)} document(s) have no supplier group: cannot show disjoint")
    shared = {groups[d] for d in fit_docs} & {groups[d] for d in apply_docs}
    if shared:
        raise GateError(f"{len(shared)} supplier group(s) are in both the fit and the apply set")


def probs_by_doc(
    keys: Sequence[cf.FieldKey], p: np.ndarray, emitted: np.ndarray
) -> dict[str, dict[FieldId, float]]:
    """``{doc_id: {(scope, field, row_idx): P}}`` of the emitted fields of one arm.

    `p` / `emitted` are aligned with `keys` (a v2 table's keys and ``emitted`` mask). An emitted
    field whose P is not a finite number in [0, 1] is an error (fail closed: a NaN would compare
    false against everything and silently pick ZS).
    """
    out: dict[str, dict[FieldId, float]] = {}
    for i, k in enumerate(keys):
        if not emitted[i]:
            continue
        v = float(p[i])
        if not math.isfinite(v) or not 0.0 <= v <= 1.0:
            raise GateError(f"an emitted field has no valid P(correct) (scope {k.scope})")
        out.setdefault(k.doc_id, {})[(k.scope, k.field, k.row_idx)] = v
    return out


# --------------------------------------------------------------------------------------------
# One slot and one row
# --------------------------------------------------------------------------------------------


def _same_value(a: Any, b: Any) -> bool:
    return bool((cf._blank(a) and cf._blank(b)) or (not cf._blank(a) and a == b))


def select_slot(v_ft: Any, v_zs: Any, p_ft: float | None, p_zs: float | None) -> tuple[str, Any]:
    """``(source, value)`` of one slot; source is ``same`` / ``ft`` / ``zs`` (rules 2 to 4).

    `p_ft` / `p_zs` are the P(correct) of the arm's value and are required (not None) exactly when
    that arm's value is non-blank. The returned value is always `v_ft` or `v_zs`, never a new one.
    """
    if _same_value(v_ft, v_zs):
        return "same", v_zs
    ft_blank, zs_blank = cf._blank(v_ft), cf._blank(v_zs)
    if (not ft_blank and p_ft is None) or (not zs_blank and p_zs is None):
        raise GateError("an emitted value has no P(correct)")
    if ft_blank:  # ZS alone filled the slot
        assert p_zs is not None
        return (ZS, v_zs) if p_zs >= KEEP_MIN else (FT, v_ft)
    if zs_blank:  # FT alone filled the slot
        assert p_ft is not None
        return (FT, v_ft) if p_ft >= KEEP_MIN else (ZS, v_zs)
    assert p_ft is not None and p_zs is not None
    return (FT, v_ft) if p_ft > p_zs else (ZS, v_zs)  # a tie goes to ZS + rules


def row_probability(row: Mapping[str, Any], row_idx: int, probs: FieldProbs) -> float | None:
    """P(row correct) = the MINIMUM P(correct) over the row's emitted fields (rule 5).

    None when the row has no emitted field. An emitted field without a P is an error.
    """
    ps: list[float] = []
    for name, v in row.items():
        if cf._blank(v):
            continue
        p = probs.get(("row", name, row_idx))
        if p is None:
            raise GateError("an emitted row field has no P(correct)")
        ps.append(p)
    return min(ps) if ps else None


# --------------------------------------------------------------------------------------------
# One document
# --------------------------------------------------------------------------------------------


@dataclass
class GateDoc:
    """The gated document plus how it was made (no values).

    ``row_sources[i]`` = ``(ft row index | None, zs row index | None)`` of output row i, indices
    into each arm's dict-only ``line_items``. ``stats`` counts, per field type
    (``header`` / ``row.<field>``), the slots won by each arm (``slot.<type>.ft`` / ``.zs`` /
    ``.same``), and ``rows.paired`` / ``rows.one_arm.<arm>.kept`` / ``.dropped``,
    ``slot.one_arm_field.<arm>.kept`` (a lone value kept) and ``.dropped`` (a lone value turned
    into the other arm's blank).
    """

    doc: dict[str, Any]
    row_sources: list[tuple[int | None, int | None]] = field(default_factory=list)
    stats: Counter[str] = field(default_factory=Counter)
    doc_type_fallback: bool = False


def _count_slot(stats: Counter[str], ftype: str, source: str, v_ft: Any, v_zs: Any) -> None:
    stats[f"slot.{ftype}.{source}"] += 1
    if source != "same" and (cf._blank(v_ft) != cf._blank(v_zs)):
        lone = ZS if cf._blank(v_ft) else FT  # the arm that alone held a value
        stats[f"slot.one_arm_field.{lone}.{'kept' if source == lone else 'dropped'}"] += 1


def _merge_keys(a: Mapping[str, Any], b: Mapping[str, Any]) -> list[str]:
    return [*a, *(k for k in b if k not in a)]


def _gate_slot(
    stats: Counter[str],
    ftype: str,
    v_ft: Any,
    v_zs: Any,
    p_ft: float | None,
    p_zs: float | None,
) -> Any:
    """`select_slot` plus the bookkeeping; `p_*` are the arm's P for a NON-blank value only."""
    src, val = select_slot(v_ft, v_zs, p_ft, p_zs)
    _count_slot(stats, ftype, src, v_ft, v_zs)
    if src != "same" and p_ft is not None and p_zs is not None and p_ft == p_zs:
        stats["slot.ties_to_zs"] += 1
    return val


def gate_document(
    ft_doc: Mapping[str, Any],
    zs_doc: Mapping[str, Any],
    p_ft: FieldProbs,
    p_zs: FieldProbs,
    pairs: Sequence[c3.RowPair] | None = None,
) -> GateDoc:
    """G of one document: both arms' post-rule documents and their per-field P(correct).

    `pairs` defaults to ``confidence_v3.align_rows`` of the two arms' dict-only rows. See the
    module docstring for the rules. Raises `GateError` on a value that is not one of the arms'
    (cannot happen by construction; checked anyway) or an emitted field without a P.
    """
    if str(ft_doc.get("doc_type")) != str(zs_doc.get("doc_type")):
        stats: Counter[str] = Counter({"doc_type_fallback": 1})
        return GateDoc(copy.deepcopy(dict(zs_doc)), [], stats, True)
    stats = Counter()
    ft_h, zs_h = c3._header(ft_doc), c3._header(zs_doc)
    header: dict[str, Any] = {}
    for name in _merge_keys(zs_h, ft_h):
        v_ft, v_zs = ft_h.get(name), zs_h.get(name)
        header[name] = _gate_slot(
            stats, "header", v_ft, v_zs,
            p_ft.get(("header", name, -1)) if not cf._blank(v_ft) else None,
            p_zs.get(("header", name, -1)) if not cf._blank(v_zs) else None,
        )  # fmt: skip

    ft_rows, zs_rows = c3.dict_rows(ft_doc), c3.dict_rows(zs_doc)
    ps = list(pairs) if pairs is not None else c3.align_rows(ft_rows, zs_rows)
    by_zs = {p.zs: p.ft for p in ps}
    paired_ft = set(by_zs.values())
    rows: list[dict[str, Any]] = []
    sources: list[tuple[int | None, int | None]] = []
    for zi, zrow in enumerate(zs_rows):
        if zi in by_zs:
            fi = by_zs[zi]
            frow = ft_rows[fi]
            out_row: dict[str, Any] = {}
            for name in _merge_keys(zrow, frow):
                v_ft, v_zs = frow.get(name), zrow.get(name)
                out_row[name] = _gate_slot(
                    stats, f"row.{name}", v_ft, v_zs,
                    p_ft.get(("row", name, fi)) if not cf._blank(v_ft) else None,
                    p_zs.get(("row", name, zi)) if not cf._blank(v_zs) else None,
                )  # fmt: skip
            rows.append(out_row)
            sources.append((fi, zi))
            stats["rows.paired"] += 1
        else:
            rp = row_probability(zrow, zi, p_zs)
            keep = rp is not None and rp >= KEEP_MIN
            stats[f"rows.one_arm.{ZS}.{'kept' if keep else 'dropped'}"] += 1
            if keep:
                rows.append(dict(zrow))
                sources.append((None, zi))
    for fi, frow in enumerate(ft_rows):
        if fi in paired_ft:
            continue
        rp = row_probability(frow, fi, p_ft)
        keep = rp is not None and rp >= KEEP_MIN
        stats[f"rows.one_arm.{FT}.{'kept' if keep else 'dropped'}"] += 1
        if keep:
            rows.append(dict(frow))
            sources.append((fi, None))

    out = copy.deepcopy(dict(zs_doc))
    out["header"] = header
    out["line_items"] = rows
    assert_no_synthesis(out, ft_doc, zs_doc, sources)
    return GateDoc(out, sources, stats, False)


def assert_no_synthesis(
    out_doc: Mapping[str, Any],
    ft_doc: Mapping[str, Any],
    zs_doc: Mapping[str, Any],
    row_sources: Sequence[tuple[int | None, int | None]],
) -> None:
    """Raise `GateError` unless every output value equals the same-slot value of one of the arms.

    Header: ``out.header[k]`` is the FT or the ZS header value of ``k`` (a missing key counts as
    blank). Rows: output row i derives from the arm rows ``row_sources[i]``; each of its values is
    the same-key value of one of them. Blank outputs must be blank in at least one arm.
    """
    h_ft, h_zs = c3._header(ft_doc), c3._header(zs_doc)
    for k, v in c3._header(out_doc).items():
        if not any(_same_value(v, a) for a in (h_ft.get(k), h_zs.get(k))):
            raise GateError(f"header field {k!r} holds a value neither arm produced")
    ft_rows, zs_rows = c3.dict_rows(ft_doc), c3.dict_rows(zs_doc)
    out_rows = c3.dict_rows(out_doc)
    if len(out_rows) != len(row_sources):
        raise GateError("output rows and row provenance differ in length")
    for row, (fi, zi) in zip(out_rows, row_sources, strict=True):
        srcs = []
        if fi is not None:
            srcs.append(ft_rows[fi])
        if zi is not None:
            srcs.append(zs_rows[zi])
        if not srcs:
            raise GateError("an output row has no source row")
        for k, v in row.items():
            if not any(_same_value(v, r.get(k)) for r in srcs):
                raise GateError(f"row field {k!r} holds a value neither arm produced")


# --------------------------------------------------------------------------------------------
# Many documents
# --------------------------------------------------------------------------------------------


def gate_predictions(
    ft_post: Mapping[str, Any],
    zs_post: Mapping[str, Any],
    p_ft: Mapping[str, FieldProbs],
    p_zs: Mapping[str, FieldProbs],
    doc_ids: Sequence[str],
) -> tuple[dict[str, Any], Counter[str]]:
    """``(G predictions, summed stats)`` for `doc_ids`.

    The output is pushed through ``coerce_predictions`` / ``repair_predictions`` (the production
    writers' choke points); it must come out unchanged (both are the identity on post-rule
    input), else `GateError`.
    """
    out: dict[str, Any] = {}
    total: Counter[str] = Counter()
    for d in doc_ids:
        gd = gate_document(ft_post[d], zs_post[d], p_ft.get(d, {}), p_zs.get(d, {}))
        out[d] = gd.doc
        total.update(gd.stats)
    final, _events = repair_predictions(coerce_predictions(out))
    if final != out:
        raise GateError("coerce / repair changed the gated output: it is not production-shaped")
    return final, total


# --------------------------------------------------------------------------------------------
# The pre-registered replacement rule (spec section 11 item 7)
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GDecision:
    """Outcome of `decide_g`: whether G replaces the winner of the primary rule."""

    winner: str
    replaces: bool
    clauses: tuple[Clause, ...]

    @property
    def final(self) -> str:
        """The system that ships: ``G`` if it replaces the winner, else the winner."""
        return "G (per-field gate)" if self.replaces else self.winner

    @property
    def line(self) -> str:
        """The one printed decision line, naming every failed clause."""
        if self.replaces:
            return f"SECONDARY SYSTEM G REPLACES {self.winner}: FINAL SYSTEM: {self.final}"
        why = "; ".join(f"{c.key} failed: {c.text}" for c in self.clauses if not c.passed)
        head = f"SECONDARY SYSTEM G DOES NOT REPLACE {self.winner}"
        return f"{head}: FINAL SYSTEM: {self.winner} [{why}]"


def _num(x: Any) -> float | None:
    """`x` as a finite float, or None (bool, None, str, NaN, inf: fail closed)."""
    if isinstance(x, bool) or not isinstance(x, int | float):
        return None
    return float(x) if math.isfinite(x) else None


def _count_clause(key: str, field_name: str, g: Mapping[str, Any], w: Mapping[str, Any]) -> Clause:
    g_v, w_v = _num(g.get(field_name)), _num(w.get(field_name))
    what = field_name.replace("_total", "").replace("_", "-")
    if g_v is None or w_v is None:
        return Clause(
            key, G_CLAUSE_NAMES[key], False,
            f"{what} count missing or not finite (G {g.get(field_name)!r}, winner "
            f"{w.get(field_name)!r}): failed closed",
        )  # fmt: skip
    ok = g_v <= w_v
    rel = "<=" if ok else ">"
    return Clause(key, G_CLAUSE_NAMES[key], ok, f"{what} count {g_v:g} (G) {rel} {w_v:g} (winner)")


def decide_g(
    winner: str,
    overall_delta_ci_lo: float,
    g_counts: Mapping[str, Any],
    winner_counts: Mapping[str, Any],
) -> GDecision:
    """Spec section 11 item 7: G replaces `winner` ONLY IF G1 and G2 and G3 all hold.

    G1: the lower bound of the paired OVERALL delta CI (G minus winner, fraction scale, pooled
    3-fold OOF) is strictly above 0 (a bound of exactly 0 fails). G2 / G3: G's ``false_fill_total``
    / ``over_null_total`` (``shipdoc.oof.over_null_counts``) are each <= the winner's, as raw
    counts, no tolerance (equal passes). Any non-finite or missing number fails closed. All three
    clauses are always evaluated. `winner` is the outcome of the primary rule
    (``g4pooled.decide``): `FT_LABEL` or `ZS_LABEL`; G is never a substitute for that rule.
    """
    if winner not in (FT_LABEL, ZS_LABEL):
        raise GateError(f"winner must be {FT_LABEL!r} or {ZS_LABEL!r}, got {winner!r}")
    lo = _num(overall_delta_ci_lo)
    if lo is None:
        c1 = Clause(
            "G1", G_CLAUSE_NAMES["G1"], False,
            f"CI lower bound {overall_delta_ci_lo!r} is not a finite number: failed closed",
        )  # fmt: skip
    elif lo > 0:
        c1 = Clause(
            "G1", G_CLAUSE_NAMES["G1"], True,
            f"CI lower bound {100 * lo:+.4f} points ({lo:+.3e}) is above 0",
        )  # fmt: skip
    else:
        rel = "equal to 0 (a strict inequality is required)" if lo == 0 else "below 0"
        c1 = Clause(
            "G1", G_CLAUSE_NAMES["G1"], False,
            f"CI lower bound {100 * lo:+.4f} points ({lo:+.3e}) is {rel}",
        )  # fmt: skip
    clauses = (
        c1,
        _count_clause("G2", "false_fill_total", g_counts, winner_counts),
        _count_clause("G3", "over_null_total", g_counts, winner_counts),
    )
    return GDecision(winner, all(c.passed for c in clauses), clauses)


def decide_g_from_pair(winner: str, pair: Mapping[str, Any]) -> GDecision:
    """`decide_g` on an ``oof.compare_models(winner_pred, g_pred, ...)`` result.

    The ``zero_shot`` slot of the pair holds the WINNER's predictions (whichever arm the primary
    rule picked) and the ``oof`` slot holds G's, so the paired delta is G minus winner.
    """
    a = pair["subsets"]["all"]
    return decide_g(
        winner,
        a["paired_delta_oof_minus_zero_shot"]["OVERALL"]["lo"],
        a["oof"]["over_null"],
        a["zero_shot"]["over_null"],
    )
