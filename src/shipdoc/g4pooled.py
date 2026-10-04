"""Pooled 3-fold final-system decision (spec section 11 item 1, pre-registered before any result).

Pure pieces of ``scripts/g4_pooled.py``; nothing here reads a run folder or a model:

* `decide`: the registered rule. The final system is FT + rules ONLY IF (C1) the lower bound of the
  paired doc-level OVERALL delta CI (FT minus ZS, both with the production rules, pooled over the
  500 train + dev docs) is strictly above 0 AND (C2) the false-fill count and (C3) the over-null
  count of FT + rules (``shipdoc.oof.over_null_counts``, summed over the 500 docs) are each
  less than or equal to ZS + rules' (raw counts, no tolerance, no CI). Anything else is
  ZS + rules (v1); a tie goes to the simpler system, so FT needs a strict CI-positive win. A lower
  bound of exactly 0 fails C1, equal counts pass C2 / C3, any non-finite number fails closed.
* `check_cross_fold`: the three OOF runs must come from the same training code and the same
  inference setup, with distinct adapters, one per fold.
* `pool_arms`: concatenate the folds' predictions into one 500-doc dict, each doc exactly once and
  the doc sets equal to the fold membership of ``splits/folds.json``.
* `render_report`: aggregates-only markdown (no gold or predicted value, no doc id).

Sign convention everywhere: delta = FT minus ZS (positive favours the fine-tuned model). All
numbers are UNVERIFIED until a verifier recomputes them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from shipdoc import g4

EXPECTED_DOCS = 500
FT_LABEL = "FT+rules"
ZS_LABEL = "ZS+rules (v1)"
COUNT_KEYS = ("false_fill_total", "over_null_total")
CLAUSE_NAMES = {
    "C1": "paired OVERALL delta (FT - ZS) CI lower bound strictly above 0",
    "C2": "false-fill count of FT+rules <= ZS+rules (raw counts, no tolerance)",
    "C3": "over-null count of FT+rules <= ZS+rules (raw counts, no tolerance)",
}


class PoolError(RuntimeError):
    """A precondition of the pooled comparison failed. Messages never quote a document value."""


# --------------------------------------------------------------------------------------------
# The registered rule
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Clause:
    """One clause of the rule: whether it passed and a one-sentence explanation with numbers."""

    key: str
    name: str
    passed: bool
    text: str


@dataclass(frozen=True)
class Decision:
    """Outcome of `decide`: ``final`` is `FT_LABEL` or `ZS_LABEL`; ``clauses`` is always C1..C3."""

    final: str
    clauses: tuple[Clause, ...]

    @property
    def ft_selected(self) -> bool:
        """True iff the final system is FT + rules."""
        return self.final == FT_LABEL

    @property
    def failed(self) -> tuple[Clause, ...]:
        """The clauses that failed (empty iff FT + rules is selected)."""
        return tuple(c for c in self.clauses if not c.passed)

    @property
    def line(self) -> str:
        """The one printed decision line, naming every failed clause."""
        if self.ft_selected:
            return f"FINAL SYSTEM: {FT_LABEL}"
        why = "; ".join(f"{c.key} failed: {c.text}" for c in self.failed)
        return f"FINAL SYSTEM: {ZS_LABEL} [{why}]"


def _num(x: Any) -> float | None:
    """`x` as a finite float, or None (bool, None, str, NaN and inf are all None: fail closed)."""
    if isinstance(x, bool) or not isinstance(x, int | float):
        return None
    return float(x) if math.isfinite(x) else None


def _count_clause(
    key: str, field: str, ft_counts: Mapping[str, Any], zs: Mapping[str, Any]
) -> Clause:
    """C2 / C3: FT count <= ZS count as raw numbers; missing or non-finite fails closed."""
    ft_v, zs_v = _num(ft_counts.get(field)), _num(zs.get(field))
    what = field.replace("_total", "").replace("_", "-")
    if ft_v is None or zs_v is None:
        text = (
            f"{what} count missing or not finite (FT {ft_counts.get(field)!r}, "
            f"ZS {zs.get(field)!r}): failed closed"
        )
        return Clause(key, CLAUSE_NAMES[key], False, text)
    ok = ft_v <= zs_v
    rel = "<=" if ok else ">"
    text = f"{what} count {ft_v:g} (FT) {rel} {zs_v:g} (ZS)"
    return Clause(key, CLAUSE_NAMES[key], ok, text)


def decide(
    overall_delta_ci_lo: float, ft_counts: Mapping[str, Any], zs_counts: Mapping[str, Any]
) -> Decision:
    """The pre-registered final-system rule (see the module docstring).

    `overall_delta_ci_lo` is the lower bound of the paired OVERALL delta CI as a FRACTION (the
    scorer's 0..1 scale, FT minus ZS); `ft_counts` / `zs_counts` are
    ``shipdoc.oof.over_null_counts`` dicts (only ``false_fill_total`` and ``over_null_total`` are
    read). All three clauses are always evaluated and reported, so the explanation names every
    clause that failed, not just the first.
    """
    lo = _num(overall_delta_ci_lo)
    if lo is None:
        c1 = Clause(
            "C1", CLAUSE_NAMES["C1"], False,
            f"CI lower bound {overall_delta_ci_lo!r} is not a finite number: failed closed",
        )  # fmt: skip
    elif lo > 0:
        c1 = Clause(
            "C1", CLAUSE_NAMES["C1"], True,
            f"CI lower bound {100 * lo:+.4f} points ({lo:+.3e}) is above 0",
        )  # fmt: skip
    else:
        rel = "equal to 0 (a strict inequality is required)" if lo == 0 else "below 0"
        c1 = Clause(
            "C1", CLAUSE_NAMES["C1"], False,
            f"CI lower bound {100 * lo:+.4f} points ({lo:+.3e}) is {rel}",
        )  # fmt: skip
    clauses = (
        c1,
        _count_clause("C2", "false_fill_total", ft_counts, zs_counts),
        _count_clause("C3", "over_null_total", ft_counts, zs_counts),
    )
    final = FT_LABEL if all(c.passed for c in clauses) else ZS_LABEL
    return Decision(final, clauses)


def decide_from_pair(pair: Mapping[str, Any]) -> Decision:
    """`decide` on an ``oof.compare_models`` result (arm ``zero_shot`` = ZS, ``oof`` = FT)."""
    a = pair["subsets"]["all"]
    return decide(
        a["paired_delta_oof_minus_zero_shot"]["OVERALL"]["lo"],
        a["oof"]["over_null"],
        a["zero_shot"]["over_null"],
    )


# --------------------------------------------------------------------------------------------
# Cross-fold validation and pooling
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RunFacts:
    """What must agree across the three OOF runs, read from one run's manifest (and verification).

    ``infer_keys`` is the ``found`` value of every ``inference key *`` row of the run's
    ``oof_verification.json`` (None when the run has no such file).
    """

    fold: int
    adapter_sha256: str
    train_code_sha: str
    batch_used: int
    batch_size: int
    config_hash: str
    model_revision: str
    output_format: str
    zero_shot_run: str | None
    infer_keys: tuple[tuple[str, str], ...] | None


def run_facts(
    manifest: Mapping[str, Any], verification: Mapping[str, Any] | None = None
) -> RunFacts:
    """`RunFacts` of a run; raises `PoolError` when a field that must be compared is absent
    (an absent field would compare equal to another absent field and hide a mismatch)."""
    sec = manifest.get("oof")
    if not isinstance(sec, Mapping):
        raise PoolError("manifest has no `oof` section: not an OOF run folder")
    batch = sec.get("batch") if isinstance(sec.get("batch"), Mapping) else {}
    cfg = manifest.get("config") if isinstance(manifest.get("config"), Mapping) else {}
    model = manifest.get("model") if isinstance(manifest.get("model"), Mapping) else {}
    vals = {
        "oof.fold": sec.get("fold"),
        "oof.adapter_sha256": sec.get("adapter_sha256"),
        "oof.train_code_sha": sec.get("train_code_sha"),
        "oof.batch.used": batch.get("used"),
        "batch_size": manifest.get("batch_size"),
        "config.hash": cfg.get("hash"),
        "model.revision": model.get("revision"),
        "output_format": manifest.get("output_format"),
    }
    absent = [k for k, v in vals.items() if v in (None, "") or isinstance(v, bool)]
    if absent:
        raise PoolError(f"manifest lacks {', '.join(absent)}: cannot compare folds")
    keys: tuple[tuple[str, str], ...] | None = None
    if verification is not None:
        keys = tuple(
            sorted(
                (str(r["name"]), repr(r.get("found")))
                for r in verification.get("rows", [])
                if str(r.get("name", "")).startswith("inference key ")
            )
        )
    return RunFacts(
        fold=int(vals["oof.fold"]),  # type: ignore[call-overload]
        adapter_sha256=str(vals["oof.adapter_sha256"]),
        train_code_sha=str(vals["oof.train_code_sha"]),
        batch_used=int(vals["oof.batch.used"]),  # type: ignore[call-overload]
        batch_size=int(vals["batch_size"]),  # type: ignore[call-overload]
        config_hash=str(vals["config.hash"]),
        model_revision=str(vals["model.revision"]),
        output_format=str(vals["output_format"]),
        zero_shot_run=sec.get("zero_shot_run"),
        infer_keys=keys,
    )


def check_cross_fold(facts: Sequence[RunFacts], expected_folds: Sequence[int]) -> None:
    """Refuse unless the runs are exactly one per expected fold, trained with the SAME adapter
    training code and inferred with the SAME inference setup (config hash, model revision, output
    format, inference keys, batch size, zero-shot run), and no adapter is shared between folds."""
    got = sorted(f.fold for f in facts)
    if got != sorted(expected_folds):
        raise PoolError(f"OOF runs cover folds {got}, expected exactly {sorted(expected_folds)}")
    for field in (
        "train_code_sha", "batch_used", "batch_size", "config_hash", "model_revision",
        "output_format", "zero_shot_run", "infer_keys",
    ):  # fmt: skip
        vals = {getattr(f, field) for f in facts}
        if len(vals) != 1:
            label = {
                "train_code_sha": "adapter training code SHA",
                "batch_used": "inference batch size (batch used)",
                "batch_size": "inference batch size (manifest)",
                "infer_keys": "inference keys",
            }.get(field, field)
            raise PoolError(f"{label} differs across folds: {len(vals)} distinct values")
    if len({f.adapter_sha256 for f in facts}) != len(facts):
        raise PoolError("two folds share one adapter sha256: each fold needs its own adapter")


def pool_arms(
    per_fold: Mapping[int, Mapping[str, Any]],
    fold_ids: Mapping[int, Sequence[str]],
    doc_fold: Mapping[str, int],
    expected_total: int | None = EXPECTED_DOCS,
) -> dict[str, Any]:
    """One prediction dict over all docs from the folds' dicts; refuses anything but a partition.

    Each fold's predictions must hold exactly that fold's doc ids, the ids must equal the
    fold membership of `doc_fold` (``splits/folds.json``), no doc may appear in two folds, and the
    total must be `expected_total` (None skips that last check; tests use miniatures).
    """
    if set(per_fold) != set(fold_ids) or set(per_fold) != set(doc_fold.values()):
        raise PoolError("folds of the predictions, the fold doc lists and doc_fold disagree")
    pooled: dict[str, Any] = {}
    for k in sorted(per_fold):
        member = {d for d, f in doc_fold.items() if f == k}
        ids = list(fold_ids[k])
        if len(ids) != len(set(ids)) or set(ids) != member:
            raise PoolError(f"fold {k}: doc list is not the fold membership of folds.json")
        if set(per_fold[k]) != member:
            raise PoolError(f"fold {k}: predictions are not exactly the fold's docs")
        dup = set(pooled) & set(per_fold[k])
        if dup:
            raise PoolError(f"fold {k}: {len(dup)} docs already appear in an earlier fold")
        pooled.update(per_fold[k])
    if set(pooled) != set(doc_fold) or len(pooled) != len(doc_fold):
        raise PoolError("pooled docs are not exactly the docs of folds.json")
    if expected_total is not None and len(pooled) != expected_total:
        raise PoolError(f"pooled {len(pooled)} docs, expected {expected_total}")
    return pooled


# --------------------------------------------------------------------------------------------
# R3 warning
# --------------------------------------------------------------------------------------------


def r3_broken_folds(effects: Mapping[int, Mapping[str, Any]]) -> list[int]:
    """Folds other than 0 where R3 fires on the FT arm and breaks at least one cell.

    R3 is judged only on folds where it fires; fold 0 can neither clear nor sink it (spec
    section 11 item 3), so fold 0 never appears here. `effects[k]["ft"]["R3"]` carries
    ``touched_docs`` and ``broken``.
    """
    return sorted(
        k
        for k, e in effects.items()
        if k != 0 and e["ft"]["R3"]["touched_docs"] > 0 and e["ft"]["R3"]["broken"] > 0
    )


# --------------------------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------------------------

LIMITS = (
    "The decision uses only C1 to C3 above; every other table is information.",
    "Both arms use the production rules (R1 / R2 / R3) and, for R3, the shapes learned from the "
    "gold of the other folds only (supplier-disjoint, checked). The R3-off sensitivity is "
    "informational: the registered decision uses all three rules.",
    "The CI is a doc-level paired bootstrap over the pooled docs; supplier clustering is not "
    "modelled, so the CI is optimistic when docs of one supplier move together.",
    "R3 concentration: 3 of 18 invoice groups carry 493 of 502 fixed rows (spec section 11 "
    "item 3, from the earlier R3 gate report); a pooled gain can be one or two suppliers.",
    "Over-null counts on rows use the scorer's pairing: a missing row is a missing row, not an "
    "over-null cell.",
)


def clause_table(decision: Decision, pair: Mapping[str, Any]) -> str:
    """The decision's clause table: the numbers each clause read and its verdict."""
    a = pair["subsets"]["all"]
    d = a["paired_delta_oof_minus_zero_shot"]["OVERALL"]
    ft, zs = a["oof"]["over_null"], a["zero_shot"]["over_null"]
    shown = {
        "C1": f"OVERALL delta {g4.pts(d['delta'])} pts, 95% CI "
        f"[{g4.pts(d['lo'])}, {g4.pts(d['hi'])}]",
        "C2": f"FT {ft['false_fill_total']} vs ZS {zs['false_fill_total']} (header "
        f"{ft['header_false_fill']} vs {zs['header_false_fill']}, row "
        f"{ft['row_false_fill']} vs {zs['row_false_fill']})",
        "C3": f"FT {ft['over_null_total']} vs ZS {zs['over_null_total']} (header "
        f"{ft['header_over_null']} vs {zs['header_over_null']}, row "
        f"{ft['row_over_null']} vs {zs['row_over_null']})",
    }
    rows = [
        [c.key, c.name, shown[c.key], "PASS" if c.passed else "FAIL", c.text]
        for c in decision.clauses
    ]
    return g4.md_table(["clause", "requirement", "numbers", "result", "reading"], rows)


def fold_delta_table(fold_deltas: Mapping[int, Mapping[str, Any]], sizes: Mapping[int, int]) -> str:
    """Per-fold OVERALL (FT+rules vs ZS+rules) for information; not part of the decision."""
    rows = []
    for k in sorted(fold_deltas):
        d = fold_deltas[k]
        rows.append(
            [k, sizes[k], g4.pct(d["a"]), g4.pct(d["b"]),
             f"{g4.pts(d['delta'])} [{g4.pts(d['lo'])}, {g4.pts(d['hi'])}]"]
        )  # fmt: skip
    head = ["fold", "docs", "OVERALL ZS+rules", "OVERALL FT+rules", "delta FT - ZS (pts) [95% CI]"]
    return g4.md_table(head, rows)


def r3_fire_table(effects: Mapping[int, Mapping[str, Any]]) -> str:
    """R1 / R2 / R3 per fold and arm: eligible / touched docs, cells changed, fixed / broken."""
    rows = []
    for k in sorted(effects):
        for arm, label in (("zs", "ZS"), ("ft", "FT")):
            for r in g4.RULES:
                c = effects[k][arm][r]
                rows.append(
                    [k, label, r, c["eligible_docs"], c["touched_docs"], c["cells_changed"],
                     c["fixed"], c["broken"], c["neutral"]]
                )  # fmt: skip
    head = ["fold", "arm", "rule", "eligible docs", "touched docs", "cells changed", "fixed",
            "broken", "neutral"]  # fmt: skip
    return g4.md_table(head, rows)


def render_report(res: Mapping[str, Any]) -> str:
    """The aggregates-only markdown report of the pooled comparison (no values, no doc ids)."""
    dec: Decision = res["decision"]
    pair, raw_pair = res["rules_pair"], res["raw_pair"]
    w: list[str] = ["# G4 pooled 3-fold: final-system decision (FT+rules vs ZS+rules)\n"]
    w.append(f"**{dec.line}**\n")
    w.append(
        "**All numbers UNVERIFIED** until a verifier recomputes them. Aggregates and counts only: "
        "no gold or predicted value, no document id.\n"
    )
    w.append("## Provenance\n")
    w.append("\n".join(f"- {k}: {v}" for k, v in res["provenance"].items()) + "\n")
    w.append("## Pre-registered rule (spec section 11 item 1)\n")
    w.append(
        "FT+rules is the final system only if C1, C2 and C3 all hold; otherwise ZS+rules (v1). "
        "A tie goes to ZS+rules: FT needs a strict CI-positive win. A CI lower bound of exactly 0 "
        "fails C1; equal counts pass C2 / C3; any count above ZS's fails.\n"
    )
    w.append("## Clause table\n")
    w.append(clause_table(dec, pair))
    for c in dec.clauses:
        w.append(f"- {c.key} {'PASS' if c.passed else 'FAIL'}: {c.text}")
    w.append("")
    w.append(
        f"Pooled over {res['n_docs']} docs. Delta = FT minus ZS, paired doc-level bootstrap "
        f"({res['n_boot']} resamples, seed {res['seed']}, unmodified scorer).\n"
    )
    w.append("## Over-nulls and false fills per field (pooled; C2 and C3 read the totals)\n")
    blocks = {
        "zs_rules": pair["subsets"]["all"]["zero_shot"],
        "ft_rules": pair["subsets"]["all"]["oof"],
        "zs_raw": raw_pair["subsets"]["all"]["zero_shot"],
        "ft_raw": raw_pair["subsets"]["all"]["oof"],
    }
    w.append(g4.over_null_table(blocks))
    w.append("## Headline: FT + rules vs ZS + rules (pooled, all docs)\n")
    w.append(g4.headline_table(pair, "ZS + rules (95% CI)", "FT + rules (95% CI)"))
    w.append("## Secondary: raw vs raw (no rules, pooled)\n")
    w.append(g4.headline_table(raw_pair, "ZS raw (95% CI)", "FT raw (95% CI)"))
    w.append("## Slices (FT + rules vs ZS + rules, pooled)\n")
    w.append(g4.slice_table(pair))
    w.append("## Per-fold OVERALL (information, not part of the decision)\n")
    w.append(fold_delta_table(res["fold_deltas"], res["fold_sizes"]))
    w.append("## Rules per fold and arm (R3 is judged only on folds where it fires)\n")
    w.append(
        "fixed / broken / neutral: cell diffs rules-off -> rules-on scored against gold, as in "
        "`reports/g4_fold0.md`. Shapes for R3 on fold k exclude fold k's suppliers.\n"
    )
    w.append(r3_fire_table(res["effects"]))
    bad = res["r3_broken_folds"]
    if bad:
        off = res["decision_r3_off"]
        w.append(
            f"WARNING: R3 fires on the FT arm of fold(s) {bad} and breaks at least one cell "
            "there: R3 should be switched off for the FT system. INFORMATIONAL (the registered "
            f"decision above uses all three rules). With R3 OFF for both arms the rule gives: "
            f"{off.line}\n"
        )
    else:
        w.append(
            "No R3 warning: R3 broke no cell on the FT arm of any fold other than fold 0 (or did "
            "not fire there). Fold 0 can neither clear nor sink R3.\n"
        )
    w.append("## R3-off sensitivity (informational)\n")
    w.append(
        f"Same rule on both arms with R1 and R2 on and R3 OFF: {res['decision_r3_off'].line}\n"
    )
    w.append(clause_table(res["decision_r3_off"], res["r3_off_pair"]))
    w.append("## Limits\n")
    w.extend(f"- {x}" for x in LIMITS)
    w.append("")
    return "\n".join(w)
