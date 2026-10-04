"""Secondary system G (per-field gate), pooled 3-fold evaluation (spec section 11 item 7).

CPU only, no model, no network. G picks, per header field and per aligned row field, the value of
the arm (FT + rules or ZS + rules) with the higher cross-fitted calibrated P(correct); the rules
of the selection live in ``shipdoc.gate`` (module docstring). This driver wires the cross-fit and
the evaluation:

* Both arms go through the production post-processing with the honest per-fold R3 shapes
  (``scripts/g4_fold.py::process_arm``, as ``scripts/g4_pooled.py`` does) and the same fail-closed
  checks as g4_pooled: one verified OOF run per requested fold, fold ids / doc counts equal to
  ``splits/folds.json``, one input resolution for the native FT runs and the native ZS run (config
  hash), the same adapter training code / inference setup / distinct adapters across folds,
  complete runs, exact doc ids with no duplicate trace, OCR for every waybill page, rules-off path
  reproducing each ``predictions.json``. The calibration side (``calibrate_v3.prepare_arm``)
  rebuilds the post-rule outputs a second time and they must equal the g4 path's, else abort.
* The PRIMARY rule (spec section 11 item 1, ``shipdoc.g4pooled.decide``) is evaluated first and
  names the WINNER; G is compared to that winner only (``shipdoc.gate.decide_g``): G replaces it
  only if the paired OVERALL CI lower bound (2000 resamples, seed 42, unmodified scorer) is > 0
  AND G's false_fill_total <= the winner's AND over_null_total <= the winner's (raw counts).
* Cross-fit. P(correct) is the ``calibrate_v3`` ``v2_agree`` calibrator of the judging view (one
  per arm), structure / kind chosen ONCE on the ``v2`` design of the FT view by the v2 rule
  (exactly ``calibrate_v3``). 3 folds: fold k's documents are scored by calibrators fitted on the
  OTHER folds' documents (supplier-disjoint; `shipdoc.gate.assert_fit_apply_disjoint` and the
  asserts inside ``cross_fit_design``). Fewer than 3 folds (EXPLORATORY): with one fold there is
  nothing to fit on, so the cross-fit is WITHIN the fold by supplier groups (GroupKFold over the
  fold's groups, k = min(5, groups), seed-free and deterministic): a weaker cross-fit, labelled in
  the report; with two folds, the other fold is the fit set.
* ``--exploratory`` is REQUIRED with fewer than 3 folds; such a run is stamped EXPLORATORY, NOT A
  DECISION in the file header and the replacement rule is neither called nor printed.

    # decision (all three folds, nothing else)
    uv run python scripts/gate_eval.py --oof-run-dir <f0> --oof-run-dir <f1> --oof-run-dir <f2> \
        --zs-run-dir <native ZS run> --out reports/gate_pooled.md --json-out <path>
    # exploratory, fold 0 only
    uv run python scripts/gate_eval.py --folds 0 --exploratory --oof-run-dir <f0> \
        --zs-run-dir <native ZS run> --out reports/gate_fold0_native_exploratory.md
    # plumbing (NOT A RESULT; the ZS run is BOTH arms, nothing is written, G must equal ZS+rules)
    uv run python scripts/gate_eval.py --plumbing-check --zs-run-dir <native ZS run>

Aggregates and counts only: no extracted value, no document id. All numbers UNVERIFIED until a
verifier recomputes them.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np

from shipdoc import confidence as cf
from shipdoc import confidence_v2 as c2
from shipdoc import confidence_v3 as c3
from shipdoc import eval as ev
from shipdoc import g4, g4pooled, gate
from shipdoc import meta as meta_mod
from shipdoc import oof as oof_mod
from shipdoc.runmeta import read_manifest

ROOT = Path(__file__).resolve().parents[1]
ALL_FOLDS = (0, 1, 2)
PLUMBING_BANNER = (
    "!" * 78 + "\n"
    "PLUMBING CHECK NOT A RESULT: the ZS run is used as BOTH arms; G must equal ZS + rules\n"
    "exactly and nothing below says anything about the fine-tuned model. Nothing is written.\n"
    + "!"
    * 78
)
DESIGN_DECISIONS = (
    "Variant: the per-field P(correct) is the `calibrate_v3` `v2_agree` calibrator (v2 features of "
    "the judged arm + FT-vs-ZS agreement features), one per judging view; the spec names the "
    "agreement features but not the variant.",
    "A slot filled by one arm only (the other blank), in the header or in an aligned row: the "
    "value is kept iff its P(correct) >= 0.5, else the other arm's blank is written. The "
    "calibrators have no P(null), so a blank has no P to compare with; this extends the spec's "
    "one-arm-row rule to fields. SPEC GAP, GG must see this.",
    "Row-level P (for a row present in one arm only): the MINIMUM P(correct) over the row's "
    "emitted fields; a row with no emitted field is dropped. `confidence_v3` / `calibrate_v3` "
    "define no row-level P and spec 11.7 does not say how to form one. SPEC GAP, GG must see "
    "this.",
    "A document on which the arms disagree about `doc_type` has different header field sets and "
    "no calibrated P for the type: the ZS + rules document is written unchanged. SPEC GAP.",
    "Ties (equal P, both arms filled and different) go to ZS + rules, as the spec's tie rule.",
)


class GateEvalError(ValueError):
    """An argument or input of the evaluation is refused (fail closed; no value is quoted)."""


def _script(name: str) -> ModuleType:
    """Import ``scripts/<name>.py`` (scripts are not a package); reuse an already loaded copy."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parent / f"{name}.py"
    )
    if spec is None or spec.loader is None:  # pragma: no cover - the file is next to this one
        raise ImportError(name)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses resolve annotations via sys.modules
    spec.loader.exec_module(mod)
    return mod


g4p = _script("g4_pooled")
g4f = g4p.g4f
rg = g4f.rg
cal3 = _script("calibrate_v3")
v2s = cal3.v2s


# --------------------------------------------------------------------------------------------
# Pure checks
# --------------------------------------------------------------------------------------------


def check_folds(folds: Sequence[int], exploratory: bool) -> bool:
    """Validate ``--folds`` / ``--exploratory``; returns whether the run is EXPLORATORY.

    Folds must be distinct members of {0, 1, 2}. Fewer than three folds REQUIRE ``exploratory``
    (otherwise `GateEvalError`: a decision needs all three folds). Three folds without the flag
    is the decision run; three folds WITH the flag is allowed and stamped exploratory.
    """
    fs = sorted(folds)
    if not fs or len(set(fs)) != len(fs) or not set(fs) <= set(ALL_FOLDS):
        raise GateEvalError(f"--folds must be distinct folds from {ALL_FOLDS}, got {list(folds)}")
    if len(fs) < len(ALL_FOLDS) and not exploratory:
        raise GateEvalError(
            f"folds {fs} are fewer than three: refusing without --exploratory (a decision needs "
            "all three folds)"
        )
    return exploratory or len(fs) < len(ALL_FOLDS)


def exploratory_stamp(folds: Sequence[int]) -> str:
    """The header stamp of an exploratory report."""
    fs = sorted(folds)
    if fs == [0]:
        return (
            "EXPLORATORY, NOT A DECISION; fold 0 was already seen when G was pre-registered; "
            "within-fold cross-fit; fold 1/2 untouched"
        )
    seen = "fold 0 was already seen when G was pre-registered" if 0 in fs else "fold 0 untouched"
    cross = "supplier-fold cross-fit" if len(fs) > 1 else "within-fold cross-fit"
    return f"EXPLORATORY, NOT A DECISION; folds {fs} given; {seen}; {cross}"


def pool_folds(
    per_fold: Mapping[int, Mapping[str, Any]],
    fold_ids: Mapping[int, Sequence[str]],
    doc_fold: Mapping[str, int],
    all_required: bool,
) -> dict[str, Any]:
    """One prediction dict over the given folds; refuses anything but exact fold membership.

    `all_required` (the decision run) delegates to ``g4pooled.pool_arms`` (all folds, exactly
    the 500 docs of folds.json). Otherwise each fold's predictions must be exactly that fold's
    docs (per ``doc_fold``) and no doc may appear twice; the folds not given are simply absent.
    """
    if all_required:
        return g4pooled.pool_arms(per_fold, fold_ids, doc_fold, g4pooled.EXPECTED_DOCS)
    if set(per_fold) != set(fold_ids):
        raise GateEvalError("folds of the predictions and of the fold doc lists disagree")
    pooled: dict[str, Any] = {}
    for k in sorted(per_fold):
        member = {d for d, f in doc_fold.items() if f == k}
        ids = list(fold_ids[k])
        if len(ids) != len(set(ids)) or set(ids) != member or set(per_fold[k]) != member:
            raise GateEvalError(f"fold {k}: docs are not the fold membership of folds.json")
        dup = set(pooled) & set(per_fold[k])
        if dup:
            raise GateEvalError(f"fold {k}: {len(dup)} docs already appear in an earlier fold")
        pooled.update(per_fold[k])
    return pooled


def validate_runs(
    dirs: Sequence[Path],
    folds: Mapping[str, Any],
    doc_fold: Mapping[str, int],
    required: Sequence[int],
) -> dict[int, tuple[Path, dict[str, Any], list[str]]]:
    """fold -> (run dir, oof section, fold doc ids): exactly one verified OOF run per required fold.

    The per-run checks of ``g4_pooled.validate_oof_runs`` (manifest oof section, adapter
    verification, complete, exact doc ids, n_inference_docs) and the cross-run checks of
    ``g4pooled.check_cross_fold`` against the REQUIRED folds (same training code / inference setup,
    distinct adapters). A missing or repeated fold, or a fold that was not asked for, is refused.
    """
    runs: dict[int, tuple[Path, dict[str, Any], list[str]]] = {}
    facts = []
    for d in dirs:
        man = read_manifest(d) if d.is_dir() else None
        fold, sec = rg.check_oof_manifest(man, None, False)
        if fold in runs:
            raise SystemExit(f"fold {fold} is given twice ({runs[fold][0].name}, {d.name})")
        rg.require_complete(d, "OOF")
        ids = rg.fold_docs(folds, fold, doc_fold)
        rg.require_oof_inputs(d, ids, True, f"OOF run of fold {fold}")
        if sec.get("n_inference_docs") not in (None, len(ids)):
            raise SystemExit(f"fold {fold}: manifest n_inference_docs differs from the fold size")
        try:
            facts.append(g4pooled.run_facts(man or {}, g4p.read_verification(d)))
        except g4pooled.PoolError as e:
            raise SystemExit(f"fold {fold}: {e}") from e
        runs[fold] = (d, sec, ids)
    try:
        g4pooled.check_cross_fold(facts, sorted(required))
    except g4pooled.PoolError as e:
        raise SystemExit(str(e)) from e
    return runs


# --------------------------------------------------------------------------------------------
# Cross-fit wiring (never fit on the docs it is applied to)
# --------------------------------------------------------------------------------------------


ProbMap = dict[gate.FieldId, float]


def cross_fit_gate_probs(
    ft: Any,
    zs: Any,
    ag_ft: np.ndarray,
    ag_zs: np.ndarray,
    folds_given: Sequence[int],
    doc_fold: Mapping[str, int],
    groups: Mapping[str, str],
) -> tuple[dict[str, ProbMap], dict[str, ProbMap], dict[str, Any]]:
    """Out-of-fold ``v2_agree`` P(correct) of both arms, as per-doc `gate.FieldProbs` maps.

    ``ft`` / ``zs`` are ``calibrate_v3.Arm`` objects, ``ag_*`` their agreement matrices. One fold
    given: supplier-grouped folds INSIDE it (``c3.inner_group_folds``, pooled structure only, LR vs
    GBM by the v2 margin rule), a within-fold cross-fit. Several folds: the supplier folds of
    ``doc_fold`` (structure and kind by the v2 rule). The structure / kind is chosen ONCE on the
    ``v2`` design of the FT view and reused for the ZS view (as ``calibrate_v3``). For every held
    fold the fit / apply sets are asserted doc- and supplier-disjoint BEFORE any model is fitted.
    Returns ``(p_ft, p_zs, info)``.
    """
    ids = ft.tv.table.doc_ids.tolist()
    docs = sorted(set(ids) | set(zs.tv.table.doc_ids.tolist()))
    c2.assert_no_test_ids(docs)
    if len(folds_given) == 1:
        fold_map, k, n_groups = c3.inner_group_folds(docs, groups)
        structures: tuple[str, ...] = ("pooled",)
        mode = f"within-fold supplier-grouped cross-fit (inner k = {k}, {n_groups} supplier groups)"
    else:
        fold_map = {d: int(doc_fold[d]) for d in docs}
        structures = ("pooled", "per_type")
        mode = f"supplier-fold cross-fit over folds {sorted(set(fold_map.values()))}"
    for h in sorted(set(fold_map.values())):
        gate.assert_fit_apply_disjoint(
            [d for d in docs if fold_map[d] != h], [d for d in docs if fold_map[d] == h], groups
        )
    views: dict[str, Any] = {}
    choice: tuple[str, str] | None = None
    info: dict[str, Any] = {"mode": mode, "structures_considered": list(structures)}
    for name, arm, agree in (("ft", ft, ag_ft), ("zs", zs, ag_zs)):
        nov = cf.compute_novelty_table(dict(arm.sigs), fold_map)
        cache: dict[int, dict[str, np.ndarray]] = {}

        def variants(
            h: int, arm: Any = arm, agree: np.ndarray = agree, nov: Any = nov,
            cache: dict[int, dict[str, np.ndarray]] = cache,
        ) -> dict[str, np.ndarray]:  # fmt: skip
            if h not in cache:
                cache[h] = c3.design_variants(arm.tv, agree, nov[h])
            return cache[h]

        rows_doc = arm.tv.table.doc_ids.tolist()
        types = arm.tv.types()
        if choice is None:
            sel = c3.select_design(
                lambda h: variants(h)["v2"], arm.y, arm.tv.emitted, rows_doc, fold_map, groups,
                types, structures,
            )  # fmt: skip
            choice = (sel.structure, sel.kind)
            info["structure"], info["kind"] = choice
            info["emitted_oof_log_loss"] = dict(sel.logloss)
        oof = c3.cross_fit_design(
            lambda h: variants(h)["v2_agree"], arm.y, arm.tv.emitted, rows_doc, fold_map, groups,
            choice[1], types if choice[0] == "per_type" else None,
        )  # fmt: skip
        views[name] = gate.probs_by_doc(arm.tv.keys, oof, arm.tv.emitted)
    return views["ft"], views["zs"], info


# --------------------------------------------------------------------------------------------
# Evaluation and report
# --------------------------------------------------------------------------------------------


def oracle_predictions(ft: Any, zs: Any, ids: Sequence[str]) -> tuple[dict[str, Any], Counter[str]]:
    """The ORACLE per-field selection: the same gate with P(correct) = the gold label (1 / 0).

    A ceiling for context, never a system: it reads gold. One-arm rows are kept only if every
    emitted field of the row is correct (row P = min), so for one-arm rows it is not a strict upper
    bound; for slots both arms fill it picks the arm that is right (ties, i.e. both right or both
    wrong, go to ZS).
    """
    p_ft = gate.probs_by_doc(ft.tv.keys, ft.y.astype(float), ft.tv.emitted)
    p_zs = gate.probs_by_doc(zs.tv.keys, zs.y.astype(float), zs.tv.emitted)
    return gate.gate_predictions(ft.post, zs.post, p_ft, p_zs, ids)


def evaluate_systems(
    zs_final: Mapping[str, Any],
    ft_final: Mapping[str, Any],
    g_pred: Mapping[str, Any],
    oracle_pred: Mapping[str, Any],
    gold: Mapping[str, Any],
    meta_rows: Sequence[Mapping[str, Any]],
    n_boot: int,
) -> dict[str, Any]:
    """Paired comparisons with ``oof.compare_models`` (the ``oof`` slot is the system under test).

    ``g_vs_zs`` / ``g_vs_ft``: G against each arm; ``oracle_vs_zs``: the oracle ceiling. The
    ``zero_shot`` slot of a pair holds the BASELINE named in the key (not necessarily ZS).
    """
    gd, meta = dict(gold), [dict(m) for m in meta_rows]
    return {
        "g_vs_zs": oof_mod.compare_models(zs_final, g_pred, gd, meta, n_boot, g4.SEED),
        "g_vs_ft": oof_mod.compare_models(ft_final, g_pred, gd, meta, n_boot, g4.SEED),
        "oracle_vs_zs": oof_mod.compare_models(zs_final, oracle_pred, gd, meta, n_boot, g4.SEED),
        "docs_g_differs_from_zs": sum(g_pred[d] != zs_final[d] for d in gold),
        "docs_g_differs_from_ft": sum(g_pred[d] != ft_final[d] for d in gold),
        "docs_arms_differ": sum(zs_final[d] != ft_final[d] for d in gold),
    }


def share_table(stats: Mapping[str, int]) -> str:
    """Slots won per arm by field type (``header`` and the row fields)."""
    types = ["header", *(t for t in c2.FIELD_TYPES if t != c2.HEADER)]
    rows = []
    for t in types:
        key = "header" if t == c2.HEADER else f"row.{t}"
        ft, zs, same = (stats.get(f"slot.{key}.{s}", 0) for s in ("ft", "zs", "same"))
        diff = ft + zs
        rows.append(
            [key, ft + zs + same, same, diff, ft, zs,
             f"{100 * ft / diff:.1f}%" if diff else "n/a"]
        )  # fmt: skip
    return g4.md_table(
        [
            "field type",
            "slots",
            "arms equal",
            "arms differ",
            "taken from FT",
            "taken from ZS",
            "FT share of differing",
        ],
        rows,
    )


def count_table(res: Mapping[str, Any]) -> str:
    """Over-null / false-fill counts of ZS + rules, FT + rules, G and the oracle (same docs)."""
    blocks = {
        "ZS + rules": res["g_vs_zs"]["subsets"]["all"]["zero_shot"]["over_null"],
        "FT + rules": res["g_vs_ft"]["subsets"]["all"]["zero_shot"]["over_null"],
        "G": res["g_vs_zs"]["subsets"]["all"]["oof"]["over_null"],
        "ORACLE (reads gold)": res["oracle_vs_zs"]["subsets"]["all"]["oof"]["over_null"],
    }
    keys = (
        ("false_fill_total", "false fills, header + row (G2 reads this)"),
        ("header_false_fill", "  header false fills"),
        ("row_false_fill", "  row false fills (scorer-paired rows)"),
        ("over_null_total", "over-null cells, header + row (G3 reads this)"),
        ("header_over_null", "  header over-nulls"),
        ("row_over_null", "  row over-nulls (scorer-paired rows)"),
        ("rows_unmatched_gold", "gold rows left unmatched (missing rows)"),
    )
    rows = [[label, *(b[k] for b in blocks.values())] for k, label in keys]
    return g4.md_table(["count", *blocks], rows)


def vs_table(pair: Mapping[str, Any], base: str, system: str) -> str:
    """Five metrics: baseline, system, paired delta (system - baseline) with its CI."""
    sub = pair["subsets"]["all"]
    rows = []
    for key, label in g4.METRIC_LABELS:
        d = sub["paired_delta_oof_minus_zero_shot"][key]
        rows.append(
            [label, g4._ci(sub["zero_shot"]["ci95"][key]), g4._ci(sub["oof"]["ci95"][key]),
             g4._delta(d), g4.excludes_zero(d["lo"], d["hi"])]
        )  # fmt: skip
    return g4.md_table(
        [
            "metric (percent)",
            f"{base} (95% CI)",
            f"{system} (95% CI)",
            f"delta {system} - {base} (pts) [95% CI]",
            "CI excludes 0",
        ],
        rows,
    )


def render_report(res: Mapping[str, Any]) -> str:
    """The aggregates-only markdown (no value, no doc id). Exploratory runs carry the stamp and no
    decision; the decision run carries the primary rule's line and G's."""
    exploratory: bool = res["exploratory"]
    w: list[str] = []
    w.append(
        f"# {res['stamp']}\n" if exploratory else "# G per-field gate: pooled 3-fold decision\n"
    )
    w.append("## G (per-field gate) vs ZS + rules and FT + rules\n")
    w.append(
        "**All numbers UNVERIFIED** until a verifier recomputes them. Aggregates and counts "
        "only: no gold or predicted value, no document id.\n"
    )
    if exploratory:
        w.append(
            "**No replacement decision is computed or printed**: this run does not satisfy "
            "spec section 11 item 7 (3 folds, cross-fitted across them).\n"
        )
    w.append("## Provenance\n")
    w.append("\n".join(f"- {k}: {v}" for k, v in res["provenance"].items()) + "\n")
    w.append("## Cross-fit (never fit on the docs it is applied to)\n")
    cf_info = res["cross_fit"]
    w.append(
        f"- mode: {cf_info['mode']}\n"
        f"- calibrator: `v2_agree`, structure **{cf_info['structure']}**, "
        f"kind **{cf_info['kind']}** "
        f"(chosen once on the v2 design of the FT view; emitted OOF log-loss "
        f"{json.dumps({k: round(v, 4) for k, v in cf_info['emitted_oof_log_loss'].items()})})\n"
        "- before any model is fitted, the fit and apply sets are asserted doc- and "
        "supplier-group-disjoint for every held-out fold (`gate.assert_fit_apply_disjoint`; "
        "`cross_fit_design` asserts it again)\n"
    )
    w.append("## Design decisions the spec does not define\n")
    w.extend(f"- {x}" for x in DESIGN_DECISIONS)
    w.append("")
    n = res["n_docs"]
    w.append(f"## Systems on {n} docs\n")
    w.append(
        f"Docs where G differs from ZS + rules: {res['docs_g_differs_from_zs']}; from FT + rules: "
        f"{res['docs_g_differs_from_ft']}; where the two arms differ: {res['docs_arms_differ']}.\n"
    )
    w.append("### G vs ZS + rules\n")
    w.append(vs_table(res["g_vs_zs"], "ZS + rules", "G"))
    w.append("### G vs FT + rules\n")
    w.append(vs_table(res["g_vs_ft"], "FT + rules", "G"))
    w.append("### Over-nulls and false fills (raw counts)\n")
    w.append(count_table(res))
    st = res["gate_stats"]
    w.append("## What G took from which arm\n")
    w.append(share_table(st))
    w.append(
        f"Ties (equal P, both arms filled, values differ) sent to ZS + rules: "
        f"{st.get('slot.ties_to_zs', 0)}. One-arm FIELDS (value vs blank): FT-only value kept "
        f"{st.get('slot.one_arm_field.ft.kept', 0)}, dropped "
        f"{st.get('slot.one_arm_field.ft.dropped', 0)}; ZS-only value kept "
        f"{st.get('slot.one_arm_field.zs.kept', 0)}, dropped "
        f"{st.get('slot.one_arm_field.zs.dropped', 0)}. Documents with different `doc_type` "
        f"(ZS document written): {st.get('doc_type_fallback', 0)}.\n"
    )
    w.append("### Rows\n")
    w.append(
        g4.md_table(
            ["rows", "count"],
            [
                ["aligned between the arms (per-field selection)", st.get("rows.paired", 0)],
                ["FT-only, kept (row P >= 0.5)", st.get("rows.one_arm.ft.kept", 0)],
                ["FT-only, dropped", st.get("rows.one_arm.ft.dropped", 0)],
                ["ZS-only, kept (row P >= 0.5)", st.get("rows.one_arm.zs.kept", 0)],
                ["ZS-only, dropped", st.get("rows.one_arm.zs.dropped", 0)],
            ],
        )
    )
    w.append("## ORACLE per-field selection (reads gold; a ceiling for context, not a system)\n")
    w.append(
        "The same gate with P(correct) replaced by the gold label of each arm's field. For slots "
        "both arms fill it picks the arm that is right; one-arm rows are kept only if every "
        "emitted field is right. It bounds what ANY per-field selection between these two arms "
        "could reach.\n"
    )
    w.append(vs_table(res["oracle_vs_zs"], "ZS + rules", "ORACLE"))
    w.append("## Decision\n")
    if exploratory:
        w.append(f"{res['stamp']}. The replacement rule was NOT evaluated.\n")
    else:
        dec = res["primary_decision"]
        gd: gate.GDecision = res["g_decision"]
        w.append(f"Primary rule (spec section 11 item 1, decided first): **{dec.line}**\n")
        w.append(f"**{gd.line}**\n")
        w.append(
            g4.md_table(
                ["clause", "requirement", "result", "reading"],
                [[c.key, c.name, "PASS" if c.passed else "FAIL", c.text] for c in gd.clauses],
            )
        )
    w.append("## Limits\n")
    w.extend(f"- {x}" for x in LIMITS)
    w.append("")
    return "\n".join(w)


LIMITS = (
    "The CI is a doc-level paired bootstrap; supplier clustering is not modelled, so it is "
    "optimistic when documents of one supplier move together.",
    "The calibrators are cross-fitted but the structure / kind choice used all folds' labels once "
    "(as `calibrate_v3`); CIs condition on the cross-fit.",
    "G is a post-hoc ensemble of two arms: any gain over the winner has to survive the three "
    "pre-registered clauses; a fold-0-only run is exploratory by construction.",
    "Rows aligned only by position (`align_rows` pass 4) can pair rows that are not the same row; "
    "G then mixes fields of two rows. Counted in `rows.paired`, not separated.",
)


# --------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------


def provenance(
    a: argparse.Namespace,
    runs: Mapping[int, tuple[Path, dict[str, Any], list[str]]],
    zs_dir: Path,
    n_learn: Mapping[int, int],
    plumbing: bool,
    wall_s: float,
) -> dict[str, str]:
    """Run folders, manifest code shas, repo HEAD, command, bootstrap settings (no doc ids)."""
    zs_man = read_manifest(zs_dir) or {}
    out: dict[str, str] = {}
    for k in sorted(runs):
        d, sec, _ids = runs[k]
        man = read_manifest(d) or {}
        out[f"FT fold {k}"] = (
            f"`{d.name}`, manifest code_sha `{man.get('code_sha')}`, adapter sha256 "
            f"`{sec.get('adapter_sha256')}`, training code SHA `{sec.get('train_code_sha')}`"
        )
    out["ZS run folder"] = f"`{zs_dir.name}`, manifest code_sha `{zs_man.get('code_sha')}`"
    out["repo state"] = rg.git_state()
    out["command"] = f"`uv run python scripts/gate_eval.py {' '.join(a.argv)}`"
    out["bootstrap"] = (
        f"{a.n_boot} doc-level resamples, seed {g4.SEED}, unmodified scorer (`shipdoc.eval`), "
        "paired (system minus baseline)"
    )
    out["R3 shapes"] = (
        "per fold, learned from the gold of "
        + ", ".join(f"{n_learn[k]} (fold {k})" for k in sorted(n_learn))
        + " docs outside that fold (supplier-disjoint, checked)"
    )
    out["wall time"] = f"{wall_s:.0f} s"
    if plumbing:
        out["MODE"] = "PLUMBING CHECK NOT A RESULT (ZS run used as both arms)"
    return out


def run(a: argparse.Namespace) -> int:
    """Validate, cross-fit, evaluate, write (or, for the plumbing check, only print)."""
    t0 = time.time()
    plumbing: bool = a.plumbing_check
    zs_dir: Path = a.zs_run_dir
    folds_given = [0] if plumbing else sorted(a.folds)
    try:
        exploratory = True if plumbing else check_folds(folds_given, a.exploratory)
    except GateEvalError as e:
        raise SystemExit(f"REFUSED (closed): {e}") from e
    folds_json = json.loads((ROOT / "splits" / "folds.json").read_text(encoding="utf-8"))
    doc_fold = rg.load_doc_fold()
    fold_ids = {k: rg.fold_docs(folds_json, k, doc_fold) for k in folds_given}
    runs: dict[int, tuple[Path, dict[str, Any], list[str]]] = {}
    if not plumbing:
        runs = validate_runs(a.oof_run_dir, folds_json, doc_fold, folds_given)
        # native FT runs and the native ZS run: one config hash, else the delta is confounded
        rg.require_same_resolution(
            {"ZS run": zs_dir, **{f"FT fold {k}": runs[k][0] for k in sorted(runs)}}
        )
    rg.require_complete(zs_dir, "zero-shot")
    ids = sorted(d for k in folds_given for d in fold_ids[k])
    c2.assert_no_test_ids(ids)
    gold = g4f.load_gold(ids)
    meta_rows = g4f.load_meta(ids)
    labels = meta_mod.load_labels("train") + meta_mod.load_labels("dev")
    groups = json.loads((ROOT / "meta" / "supplier_groups.json").read_text(encoding="utf-8"))
    try:
        per_fold_shapes = rg.fold_shapes(labels, doc_fold, groups)
    except ValueError as e:  # supplier overlap: fail closed
        raise SystemExit(str(e)) from e
    n_learn = {k: len(per_fold_shapes[k][1]) for k in folds_given}
    zs_fin: dict[int, Any] = {}
    ft_fin: dict[int, Any] = {}
    ft_preds: dict[str, Any] = {}
    ft_traces: dict[str, Any] = {}
    zs_preds: dict[str, Any] = {}
    zs_traces: dict[str, Any] = {}
    for k in folds_given:
        ids_k = fold_ids[k]
        gold_k = {d: gold[d] for d in ids_k}
        rg.require_oof_inputs(zs_dir, ids_k, False, "zero-shot run")
        for d in [zs_dir] if plumbing else [zs_dir, runs[k][0]]:
            rg.require_ocr(d, SimpleNamespace(gold=gold_k), a.ocr_cache)
        zs_arm = g4f.read_arm("ZS arm", zs_dir, ids_k)
        ft_arm = zs_arm if plumbing else g4f.read_arm(f"FT arm fold {k}", runs[k][0], ids_k)
        shapes = per_fold_shapes[k][0]
        _, zs_fin[k], _ = g4f.process_arm(zs_arm, shapes, a.ocr_cache)
        ft_fin[k] = zs_fin[k] if plumbing else g4f.process_arm(ft_arm, shapes, a.ocr_cache)[1]
        zs_preds.update(zs_arm.saved)
        zs_traces.update({t["doc_id"]: t for t in zs_arm.traces})
        ft_preds.update(ft_arm.saved)
        ft_traces.update({t["doc_id"]: t for t in ft_arm.traces})
    all_req = not exploratory
    zs_pool = pool_folds(zs_fin, fold_ids, doc_fold, all_req)
    ft_pool = pool_folds(ft_fin, fold_ids, doc_fold, all_req)
    if set(zs_pool) != set(gold) or set(ft_pool) != set(gold):
        raise SystemExit("gold docs differ from the pooled prediction docs")
    # calibration side: the same production path, rebuilt through calibrate_v3 (must agree exactly)
    shapes_fn, _eq = v2s.fold_shape_fn(labels, doc_fold, groups)
    cal_gold, _cal_meta = v2s.v1cal.load_corpus(ids)
    ocr = ev.load_ocr_pages(list(ids), a.ocr_cache)
    try:
        ft = cal3.prepare_arm(ids, ft_preds, ft_traces, shapes_fn, a.ocr_cache, cal_gold, ocr, True)
        zs = (
            ft
            if plumbing
            else cal3.prepare_arm(
                ids, zs_preds, zs_traces, shapes_fn, a.ocr_cache, cal_gold, ocr, True
            )
        )
    except cal3.ArmError as e:
        raise SystemExit(f"FAIL (closed): {e}") from e
    if any(zs.post[d] != zs_pool[d] or ft.post[d] != ft_pool[d] for d in ids):
        raise SystemExit("calibrate_v3's post-rule outputs differ from the g4 production path")
    pairs = c3.align_all(ft.post, zs.post, ids)
    ag_ft = c3.agreement_matrix(ft.tv.keys, ft.post, zs.post, pairs, True)
    ag_zs = c3.agreement_matrix(zs.tv.keys, zs.post, ft.post, pairs, False)
    p_ft, p_zs, cf_info = cross_fit_gate_probs(
        ft, zs, ag_ft, ag_zs, folds_given, doc_fold, {d: groups[d] for d in ids}
    )
    g_pred, g_stats = gate.gate_predictions(ft.post, zs.post, p_ft, p_zs, ids)
    if plumbing and g_pred != zs_pool:
        raise SystemExit("PLUMBING FAILED: G differs from ZS + rules although both arms are equal")
    oracle, _ = oracle_predictions(ft, zs, ids)
    res = evaluate_systems(zs_pool, ft_pool, g_pred, oracle, gold, meta_rows, a.n_boot)
    res |= {
        "n_docs": len(ids),
        "folds": folds_given,
        "exploratory": exploratory,
        "stamp": exploratory_stamp(folds_given) if not plumbing else "PLUMBING CHECK NOT A RESULT",
        "gate_stats": dict(g_stats),
        "cross_fit": cf_info,
        "provenance": provenance(a, runs, zs_dir, n_learn, plumbing, time.time() - t0),
    }
    decision_line = None
    if not exploratory:
        pair_primary = oof_mod.compare_models(
            zs_pool, ft_pool, dict(gold), [dict(m) for m in meta_rows], a.n_boot, g4.SEED
        )
        primary = g4pooled.decide_from_pair(pair_primary)
        winner_pair = res["g_vs_ft"] if primary.ft_selected else res["g_vs_zs"]
        res["primary_decision"] = primary
        res["g_decision"] = gate.decide_g_from_pair(primary.final, winner_pair)
        decision_line = res["g_decision"].line
    text = render_report(res)
    if plumbing:
        print(PLUMBING_BANNER + "\n" + text)
        print("PLUMBING OK: G equals ZS + rules on every document (arms identical)")
        return 0
    out = a.out or ROOT / "reports" / ("gate_exploratory.md" if exploratory else "gate_pooled.md")
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")
    if a.json_out is not None:
        a.json_out.write_text(json.dumps(_jsonable(res), indent=1, default=str), encoding="utf-8")
        print(f"wrote {a.json_out}")
    print(res["stamp"] if exploratory else f"{res['primary_decision'].line}\n{decision_line}")
    return 0


def _jsonable(res: Mapping[str, Any]) -> dict[str, Any]:
    """`res` with the Decision objects flattened (everything else is already plain data)."""
    out = dict(res)
    for key in ("primary_decision", "g_decision"):
        if key in res:
            d = res[key]
            out[key] = {
                "final": d.final,
                "line": d.line,
                "clauses": [{"key": c.key, "passed": c.passed, "text": c.text} for c in d.clauses],
            }
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entry."""
    raw = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--oof-run-dir", type=Path, action="append", default=[],
        help="an OOF run folder (once per fold, in any order)",
    )  # fmt: skip
    ap.add_argument("--zs-run-dir", type=Path, required=True, help="the native ZS run folder")
    ap.add_argument("--folds", type=int, nargs="+", default=list(ALL_FOLDS))
    ap.add_argument("--exploratory", action="store_true", help="required with fewer than 3 folds")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--plumbing-check", action="store_true", help="NOT A RESULT; writes nothing")
    ap.add_argument("--ocr-cache", type=Path, default=None)
    ap.add_argument("--n-boot", type=int, default=g4.N_BOOT)
    a = ap.parse_args(raw)
    a.argv = raw
    if a.plumbing_check:
        if a.oof_run_dir:
            print("note: --plumbing-check uses --zs-run-dir as both arms; --oof-run-dir is ignored")
        if a.out is not None or a.json_out is not None:
            print("note: --plumbing-check writes no report; --out / --json-out are ignored")
    elif len(a.oof_run_dir) != len(a.folds):
        ap.error("give exactly one --oof-run-dir per fold in --folds (or --plumbing-check)")
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
