"""Gate for the candidate merge rules R1 / R2 / R3 (CPU only, no model is called).

The rules were designed after looking at dev100 errors (``reports/row_errors.md``), so their dev
numbers are optimistic. R1 / R2 have no learned parameters: they are gated on the 400 TRAIN
documents of the zero-shot 02 run only; the 100 dev documents stay a held-out confirmation
(``--confirm-dev``, evaluated AFTER the train verdict with the same frozen rules, printed
separately, never part of the verdict).

R3 learns its slot shapes from gold (``rules.learn_slot_shapes``), so gating it on the docs its
shapes came from is circular. R3 is therefore gated SUPPLIER-HELD-OUT: for each fold k of
``splits/folds.json`` (supplier-grouped, train + dev = 500 docs) shapes are learned ONLY from the
gold of that fold's training docs (``doc_fold != k``), R3 is applied to fold k's docs, and the
folds are pooled (every doc evaluated exactly once) into one outcome and one paired CI. That
pooled row is the R3 verdict basis; dev docs are inside the pooling, so there is no separate R3
dev confirmation. The "all-gold shapes" row (shapes from all 500 gold docs = the shipping
artefact; in-sample on the evaluated docs) is shown for reference and is never a verdict. The
run therefore needs predictions for all 500 docs (train_* and dev_*).

Inputs: ``<run>/{predictions.json,trace.jsonl}`` (raw model output per page), gold labels, the OCR
cache (R2 only), ``meta/supplier_groups.json``. Rules: ``shipdoc.rules`` (pure functions on merged
doc dicts, applied BEFORE schema coercion; this script touches no pipeline default).

Per rule on the evaluated docs: ``fixed`` (cell / row wrong, becomes right), ``broken`` (right,
becomes wrong), ``neutral`` (touched, correctness unchanged), ``untouched``, ``net`` = fixed -
broken; paired bootstrap (2000 doc-level resamples, seed 42) of delta OVERALL with the UNMODIFIED
scorer via ``shipdoc.eval``; per supplier group (``inv_gNN`` / ``wb_gNN``) fixed / broken / net.
Units: R1 = waybill carrier cell, R2 = waybill mawb / hawb cell, R3 = predicted invoice row
(paired to gold by the scorer's pairing on the UNPATCHED prediction; unpaired rows are neutral).

R2 needs OCR at inference time (Paddle on Colab for test): a different deployment cost from the
image-only pick. The ``R2-ocrfree`` variant searches the model's own raw output instead of OCR.

Aggregates and counts only: no gold or predicted values. Numbers are UNVERIFIED until a verifier
recomputes them. Run when the 02 run lands::

    uv run python scripts/rule_gate.py --run-dir <path to 02 run folder> --out reports/rule_gate.md

OOF mode (the fine-tuned model's fold-K outputs, ``oof`` section in the run's manifest)::

    uv run python scripts/rule_gate.py --oof-run-dir <oof_fold0_sha7 folder> [--fold 0] \
        [--zs-run-dir <02 run folder>] [--out reports/rule_gate_ft_fold0.md]

The SAME frozen rules are applied to fold K's held-out docs only: R1 / R2 (no learned parameters)
on those docs, R3 with shapes learned from the gold of the OTHER folds (supplier-disjoint,
checked). Same ship rule and scorer; policy: a rule ships with the fine-tuned model only if it
also passes on this fold's OOF outputs. With ~170 docs R1 / R2 may lack power, so a DO NOT SHIP
caused only by a CI that includes 0 (broken == 0, point delta > 0) is worded "NOT SHIPPABLE ON
THIS EVIDENCE (insufficient power)", distinct from "FAILED (broken > 0)". Fails closed (no oof
section, incomplete run, doc ids != fold K, missing gold / trace / OCR page); with `--zs-run-dir`
the 02 run's effect on the same docs is printed side by side.

Smoke (NOT A GATE RESULT, never writes a report). R1 / R2 run on the dev docs present; R3
held-out runs on the same docs, each evaluated with shapes learned from the gold of the OTHER
folds' docs (the full folds.json assignment, present in the run or not)::

    uv run python scripts/rule_gate.py --run-dir <dev run> --allow-dev-smoke
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from shipdoc import eval as ev
from shipdoc import meta as meta_mod
from shipdoc import paths, rules
from shipdoc.diagnostics import empty
from shipdoc.ocr import doc_pages, page_text
from shipdoc.replay import read_trace, salvage_page_json
from shipdoc.runcompat import ResolutionMismatchError, assert_runs_share_resolution
from shipdoc.runmeta import read_manifest

ROOT = Path(__file__).resolve().parents[1]
N_BOOT = 2000
SEED = 42
EXPECTED_TRAIN_DOCS = 400
PLUMBING_BANNER = (
    "!" * 78 + "\n"
    "PLUMBING CHECK NOT A RESULT: the run is NOT an OOF run (no oof section is required); the\n"
    "numbers below say nothing about the fine-tuned model. No report file is written.\n" + "!" * 78
)
INSUFFICIENT_POWER = "NOT SHIPPABLE ON THIS EVIDENCE (insufficient power)"
FAILED_BROKEN = "FAILED (broken > 0)"
SMOKE_BANNER = (
    "!" * 78 + "\n"
    "NOT A GATE RESULT: dev smoke mode. The rules were designed on these docs; nothing here may\n"
    "be cited as a ship decision. No report file is written.\n" + "!" * 78
)
CLAUSE_CI = "paired CI of delta OVERALL excludes 0 (lower bound > 0)"
CLAUSE_BROKEN = "broken == 0"
CLAUSE_GROUP = "no group with net breaks (net < 0 in any group)"


# --------------------------------------------------------------------------------------------
# Verdict
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Verdict:
    """SHIP / DO NOT SHIP of one rule with the clause results."""

    ship: bool
    failed: tuple[str, ...]
    reasons: tuple[str, ...]


def ship_verdict(
    ci_lo: float,
    broken: int,
    group_net: Mapping[str, int] | None = None,
    require_group_clause: bool = False,
) -> Verdict:
    """SHIP iff the paired CI of delta OVERALL excludes 0 (lower bound > 0) AND broken == 0.

    `require_group_clause=True` (R3) additionally demands that no group has net < 0
    (`group_net` group -> fixed - broken). Otherwise DO NOT SHIP, naming the failed clause(s).
    """
    reasons: list[str] = []
    failed: list[str] = []
    if ci_lo > 0:
        reasons.append(f"{CLAUSE_CI}: PASS (lower bound {100 * ci_lo:+.2f} pts)")
    else:
        failed.append(CLAUSE_CI)
        reasons.append(f"{CLAUSE_CI}: FAIL (lower bound {100 * ci_lo:+.2f} pts)")
    if broken == 0:
        reasons.append(f"{CLAUSE_BROKEN}: PASS")
    else:
        failed.append(CLAUSE_BROKEN)
        reasons.append(f"{CLAUSE_BROKEN}: FAIL (broken = {broken})")
    if require_group_clause:
        bad = sorted(g for g, n in (group_net or {}).items() if n < 0)
        if bad:
            failed.append(CLAUSE_GROUP)
            reasons.append(f"{CLAUSE_GROUP}: FAIL (groups {', '.join(bad)})")
        else:
            reasons.append(f"{CLAUSE_GROUP}: PASS")
    return Verdict(ship=not failed, failed=tuple(failed), reasons=tuple(reasons))


# --------------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------------


@dataclass
class Inputs:
    """Everything the rules and the scorer need for one evaluated doc set."""

    pred: dict[str, Any]
    gold: dict[str, Any]
    groups: dict[str, str]
    supplier: dict[str, Any]  # doc_id -> page-1 parsed header supplier_name
    model_text: dict[str, str]  # doc_id -> raw model output of all pages
    ocr_text: dict[str, str]  # doc_id -> OCR text (waybills only; may be missing)
    sc: Any = None
    train_pool: list[dict[str, Any]] = field(default_factory=list)


def page_parsed(page: Mapping[str, Any]) -> dict[str, Any] | None:
    """Parsed page output; falls back to salvaging ``raw_text`` when ``parsed`` is absent."""
    parsed = page.get("parsed")
    if isinstance(parsed, dict):
        return parsed
    got = salvage_page_json(page.get("raw_text") or "")
    return got if isinstance(got, dict) else None


def trace_facts(trace: Mapping[str, Any]) -> tuple[Any, str]:
    """(supplier_name of the page-1 parsed header, concatenated raw model text of all pages)."""
    pages = trace.get("pages") or []
    first = page_parsed(pages[0]) if pages else None
    hdr = (first or {}).get("header") if isinstance(first, dict) else None
    sup = hdr.get("supplier_name") if isinstance(hdr, dict) else None
    return sup, "\n".join(str(p.get("raw_text") or "") for p in pages)


def load_inputs(
    run_dir: Path, ids: Sequence[str], ocr_cache: Path | None, split_of: Callable[[str], str]
) -> Inputs:
    """Load predictions, gold, traces, groups and (waybill) OCR text for the doc ids `ids`."""
    sc = ev.load_scorer()
    pred_all = json.loads((run_dir / "predictions.json").read_text(encoding="utf-8"))
    gold: dict[str, Any] = {}
    for split in sorted({split_of(d) for d in ids}):
        gold |= ev.load_gold(paths.data_dir() / split / "labels")
    missing = [d for d in ids if d not in gold or d not in pred_all]
    if missing:
        raise SystemExit(f"{len(missing)} docs lack gold or prediction, e.g. {missing[:3]}")
    traces = {t["doc_id"]: t for t in read_trace(run_dir / "trace.jsonl")}
    if any(d not in traces for d in ids):
        raise SystemExit("trace.jsonl lacks some selected docs")
    groups = json.loads((ROOT / "meta" / "supplier_groups.json").read_text(encoding="utf-8"))
    wb_ids = [d for d in ids if gold[d]["doc_type"] == "waybill"]
    ocr = ev.load_ocr_pages(wb_ids, ocr_cache)
    facts = {d: trace_facts(traces[d]) for d in ids}
    return Inputs(
        pred={d: pred_all[d] for d in ids},
        gold={d: gold[d] for d in ids},
        groups={d: groups[d] for d in ids},
        supplier={d: facts[d][0] for d in ids},
        model_text={d: facts[d][1] for d in ids},
        ocr_text={d: "\n".join(page_text(p) for p in pages) for d, pages in ocr.items()},
        sc=sc,
        train_pool=meta_mod.load_labels("train"),
    )


# --------------------------------------------------------------------------------------------
# Unit correctness and rule application
# --------------------------------------------------------------------------------------------


def cell_ok(sc: Any, name: str, value: Any, gold_value: Any) -> bool:
    """Header / row cell right under the scorer: both empty, or both set and ``sc.same``."""
    if empty(value) or empty(gold_value):
        return empty(value) and empty(gold_value)
    return bool(sc.same(name, value, gold_value))


@dataclass
class Outcome:
    """Unit-level result of one rule: counts overall and per group."""

    eligible: int = 0
    fixed: int = 0
    broken: int = 0
    neutral: int = 0
    groups: dict[str, dict[str, int]] = field(default_factory=dict)

    def bump(self, group: str, key: str) -> None:
        """Count one touched unit of `key` (fixed / broken / neutral) in `group`."""
        setattr(self, key, getattr(self, key) + 1)
        self.groups.setdefault(group, {"n": 0, "fixed": 0, "broken": 0, "neutral": 0})[key] += 1

    @property
    def touched(self) -> int:
        return self.fixed + self.broken + self.neutral

    @property
    def untouched(self) -> int:
        return self.eligible - self.touched

    @property
    def net(self) -> int:
        return self.fixed - self.broken


def classify(before_ok: bool, after_ok: bool) -> str:
    """``fixed`` / ``broken`` / ``neutral`` from correctness before and after a change."""
    if not before_ok and after_ok:
        return "fixed"
    if before_ok and not after_ok:
        return "broken"
    return "neutral"


RuleFn = Callable[[str, Mapping[str, Any]], tuple[dict[str, Any], list[rules.Change]]]


def apply_rule(
    inp: Inputs, fn: RuleFn, doc_type: str, n_cells_per_doc: int
) -> tuple[dict[str, Any], Outcome]:
    """Apply `fn` doc by doc; returns (patched predictions, unit outcome).

    Only docs of `doc_type` (by GOLD type) are passed to the rule; the rest stay untouched.
    `n_cells_per_doc` is the number of eligible header cells of a waybill doc (R1 1, R2 2); for
    invoices eligibility is the number of predicted rows.
    """
    sc = inp.sc
    patched = dict(inp.pred)
    out = Outcome()
    for d in sorted(inp.pred):
        g = inp.gold[d]
        grp = inp.groups[d]
        p = inp.pred[d]
        if g["doc_type"] != doc_type:
            continue
        out.groups.setdefault(grp, {"n": 0, "fixed": 0, "broken": 0, "neutral": 0})["n"] += 1
        if not isinstance(p, dict):
            continue
        new, changes = fn(d, p)
        patched[d] = new
        if doc_type == "waybill":
            out.eligible += n_cells_per_doc
            for c in changes:
                gv = g["header"].get(c.field)
                k = classify(cell_ok(sc, c.field, c.before, gv), cell_ok(sc, c.field, c.after, gv))
                out.bump(grp, k)
            continue
        rows = [x for x in (p.get("line_items") or []) if isinstance(x, dict)]
        out.eligible += len(rows)
        gr = list(g.get("line_items") or [])
        full, partial, _, _ = ev._pair_rows(sc, rows, gr)
        partner = dict(full + partial)
        for c in changes:
            gi = partner.get(c.row)
            if gi is None:
                out.bump(grp, "neutral")
                continue
            bc, bp = c.before
            ac, ap = c.after
            before_ok = cell_ok(sc, rules.CPN, bc, gr[gi].get(rules.CPN)) and cell_ok(
                sc, rules.PO, bp, gr[gi].get(rules.PO)
            )
            after_ok = cell_ok(sc, rules.CPN, ac, gr[gi].get(rules.CPN)) and cell_ok(
                sc, rules.PO, ap, gr[gi].get(rules.PO)
            )
            out.bump(grp, classify(before_ok, after_ok))
    return patched, out


def rescore(inp: Inputs, patched: Mapping[str, Any], n_boot: int) -> dict[str, Any]:
    """Delta OVERALL and its paired-bootstrap 95% CI (unmodified scorer, doc level, seed 42)."""
    base = ev.score(dict(inp.pred), dict(inp.gold))["all"]["OVERALL"]
    new = ev.score(dict(patched), dict(inp.gold))["all"]["OVERALL"]
    pb = ev.paired_bootstrap(dict(inp.pred), dict(patched), dict(inp.gold), n=n_boot, seed=SEED)
    return {
        "base_overall": base,
        "overall": new,
        "delta": new - base,
        "lo": pb["OVERALL"]["lo"],
        "hi": pb["OVERALL"]["hi"],
    }


@dataclass
class RuleResult:
    """One evaluated rule variant."""

    name: str
    unit: str
    note: str
    outcome: Outcome
    stats: dict[str, Any]
    verdict: Verdict
    informational: bool = False  # shown for reference, never a ship decision


def evaluate(
    name: str,
    unit: str,
    note: str,
    inp: Inputs,
    fn: RuleFn,
    doc_type: str,
    cells: int,
    n_boot: int,
    group_clause: bool = False,
) -> RuleResult:
    """Apply, rescore and judge one rule variant on `inp`."""
    patched, out = apply_rule(inp, fn, doc_type, cells)
    stats = rescore(inp, patched, n_boot)
    net = {g: v["fixed"] - v["broken"] for g, v in out.groups.items()}
    verdict = ship_verdict(stats["lo"], out.broken, net, group_clause)
    return RuleResult(name, unit, note, out, stats, verdict)


# --------------------------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------------------------


def fold_shapes(
    labels: Sequence[Mapping[str, Any]],
    doc_fold: Mapping[str, int],
    groups: Mapping[str, str],
) -> dict[int, tuple[rules.SlotShapes, frozenset[str]]]:
    """Per fold k: (shapes learned from gold of docs with ``doc_fold != k``, those doc ids).

    `labels` are gold dicts carrying ``doc_id``; labels of docs absent from `doc_fold` are never
    used. Fails closed (ValueError) if a learner doc shares a doc id or a supplier group with
    fold k's own docs, i.e. if the folds were not supplier-grouped.
    """
    out: dict[int, tuple[rules.SlotShapes, frozenset[str]]] = {}
    for k in sorted(set(doc_fold.values())):
        held = {d for d, f in doc_fold.items() if f == k}
        held_groups = {groups[d] for d in held}
        learn = [x for x in labels if x["doc_id"] in doc_fold and doc_fold[x["doc_id"]] != k]
        ids = frozenset(x["doc_id"] for x in learn)
        if ids & held or {groups[d] for d in ids} & held_groups:
            raise ValueError(f"fold {k}: shape learner input overlaps the held-out docs/suppliers")
        out[k] = (rules.learn_slot_shapes(learn), ids)
    return out


def evaluate_r3_heldout(
    inp: Inputs,
    per_fold: Mapping[int, tuple[rules.SlotShapes, frozenset[str]]],
    doc_fold: Mapping[str, int],
    n_boot: int,
) -> RuleResult:
    """R3 with each doc patched by the shapes of ITS fold's learner; all folds pooled once."""
    return evaluate(
        "R3 (supplier-held-out, folds pooled)",
        "predicted invoice rows",
        "each doc patched with shapes learned only from the gold of the other folds' docs "
        "(supplier-disjoint); VERDICT BASIS",
        inp,
        lambda d, p: rules.slot_shape(p, per_fold[doc_fold[d]][0]),
        "invoice",
        0,
        n_boot,
        group_clause=True,
    )


def run_r3(
    inp: Inputs,
    labels: Sequence[Mapping[str, Any]],
    doc_fold: Mapping[str, int],
    groups: Mapping[str, str],
    n_boot: int,
) -> list[RuleResult]:
    """R3 section: the held-out pooled verdict row plus the in-sample shipping-artefact row.

    Shapes are learned from the full `doc_fold` assignment; `inp` may hold only a subset of its
    docs (smoke), in which case only those docs are evaluated.
    """
    unknown = sorted(d for d in inp.pred if d not in doc_fold)
    if unknown:
        raise SystemExit(f"{len(unknown)} evaluated docs are not in folds.json, e.g. {unknown[:3]}")
    per_fold = fold_shapes(labels, doc_fold, groups)
    held = evaluate_r3_heldout(inp, per_fold, doc_fold, n_boot)
    shipped = rules.learn_slot_shapes([x for x in labels if x["doc_id"] in doc_fold])
    ship_row = evaluate(
        "R3 (all-gold shapes, shipping artefact, in-sample)",
        "predicted invoice rows",
        "shapes from all 500 gold docs, the version that would ship; includes the evaluated "
        "docs so it is optimistic and NOT a verdict",
        inp,
        lambda d, p: rules.slot_shape(p, shipped),
        "invoice",
        0,
        n_boot,
        group_clause=True,
    )
    return [held, replace(ship_row, informational=True)]


def run_rules(inp: Inputs, n_boot: int) -> list[RuleResult]:
    """Evaluate R1, R2 and R2-ocrfree (no learned parameters) on `inp`; R3 is `run_r3`."""
    out: list[RuleResult] = []
    out.append(
        evaluate(
            "R1",
            "waybill carrier cells",
            "carrier <- model's own supplier_name slot; never overrides a model carrier",
            inp,
            lambda d, p: rules.carrier_from_supplier(p, inp.supplier[d]),
            "waybill",
            1,
            n_boot,
        )
    )
    out.append(
        evaluate(
            "R2",
            "waybill mawb/hawb cells",
            "unique OCR pattern match (REQUIRES OCR at inference time)",
            inp,
            lambda d, p: rules.pattern_backfill(p, inp.ocr_text.get(d, "")),
            "waybill",
            2,
            n_boot,
        )
    )
    out.append(
        evaluate(
            "R2-ocrfree",
            "waybill mawb/hawb cells",
            "same rule, pattern searched in the model's own raw output (no OCR); variant",
            inp,
            lambda d, p: rules.pattern_backfill(p, inp.model_text.get(d, "")),
            "waybill",
            2,
            n_boot,
        )
    )
    return out


def ocr_dependence(results: Sequence[RuleResult]) -> str:
    """Honest R2 deployment sentence from the OCR vs OCR-free outcomes."""
    by = {r.name: r for r in results}
    if "R2" not in by or "R2-ocrfree" not in by:
        return ""
    a, b = by["R2"].outcome, by["R2-ocrfree"].outcome
    return (
        f"R2 with OCR touches {a.touched} cells (fixed {a.fixed}, broken {a.broken}); the "
        f"OCR-free variant (pattern searched only in the model's own raw output) touches "
        f"{b.touched} cells (fixed {b.fixed}, broken {b.broken}). A model that emitted null "
        "for a field has usually not printed the value anywhere else in its output, so the "
        f"OCR-free variant recovers {b.fixed} of the {a.fixed} cells the OCR variant fixes: "
        "R2's gain is not available to the image-only pipeline without OCR at inference time."
    )


# --------------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------------


def md_table(head: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Markdown table; every cell is str()-ed."""
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def pts(x: float) -> str:
    """Fraction -> signed percentage points, two decimals."""
    return f"{100 * x:+.2f}"


def verdict_text(r: RuleResult, smoke: bool) -> str:
    """SHIP / DO NOT SHIP; in smoke mode the clause outcome is shown but never as a verdict."""
    if r.informational:
        return "no verdict (in-sample reference)"
    if smoke:
        return "smoke only: clauses " + ("pass" if r.verdict.ship else "fail")
    return "SHIP" if r.verdict.ship else "DO NOT SHIP"


def render_section(
    title: str, n_docs: int, results: Sequence[RuleResult], smoke: bool = False
) -> str:
    """Markdown of one evaluated doc set: summary, verdicts, per-group tables."""
    w: list[str] = [f"## {title}\n", f"Docs evaluated: {n_docs}.\n"]
    head = [
        "rule", "unit", "eligible", "fixed", "broken", "neutral", "untouched", "net",
        "d OVERALL (pts)", "95% CI (pts)", "verdict",
    ]  # fmt: skip
    rows = []
    for r in results:
        o, s = r.outcome, r.stats
        rows.append(
            [
                r.name, r.unit, o.eligible, o.fixed, o.broken, o.neutral, o.untouched, o.net,
                pts(s["delta"]), f"[{pts(s['lo'])}, {pts(s['hi'])}]",
                verdict_text(r, smoke),
            ]
        )  # fmt: skip
    w.append(md_table(head, rows))
    for r in results:
        w.append(f"### {r.name}: {verdict_text(r, smoke)}\n")
        w.append(f"{r.note}.\n")
        w += [f"- {x}" for x in r.verdict.reasons]
        if r.verdict.failed:
            w.append(f"- failed clause(s): {'; '.join(r.verdict.failed)}")
        w.append("")
        w.append(
            md_table(
                ["group", "group n (docs)", "fixed", "broken", "neutral", "net"],
                [
                    [g, v["n"], v["fixed"], v["broken"], v["neutral"], v["fixed"] - v["broken"]]
                    for g, v in sorted(r.outcome.groups.items())
                ],
            )
        )
    return "\n".join(w)


def git_state() -> str:
    """Short HEAD sha, ``+dirty`` when the tracked tree differs."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=ROOT, check=True,
        ).stdout.strip()  # fmt: skip
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, cwd=ROOT, check=True,
        ).stdout.strip()  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return sha + ("+dirty" if dirty else "")


def render_report(
    run_name: str,
    main: tuple[str, int, Sequence[RuleResult]],
    confirm: tuple[str, int, Sequence[RuleResult]] | None,
    smoke: bool,
    r3: tuple[str, int, Sequence[RuleResult]] | None = None,
) -> str:
    """The full aggregates-only report (`r3`: the supplier-held-out R3 section)."""
    w: list[str] = []
    if smoke:
        w.append(SMOKE_BANNER + "\n")
    w.append("# Merge-rule gate (R1 / R2 / R3)\n")
    w.append(
        f"**Provenance.** Run `{run_name}`, generated by `uv run python scripts/rule_gate.py` "
        f"(repo state `{git_state()}`; paired bootstrap {N_BOOT} doc-level resamples, seed "
        f"{SEED}; unmodified scorer via `shipdoc.eval`). Rules: `src/shipdoc/rules.py`, applied "
        "to merged doc dicts before schema coercion. **All numbers UNVERIFIED** until a "
        "verifier recomputes them. Aggregates and counts only.\n"
    )
    w.append(
        "**Ship rule.** SHIP iff the paired 95% CI of delta OVERALL excludes 0 (lower bound > 0) "
        "AND broken == 0. R3 additionally requires no group with net < 0. Otherwise DO NOT "
        "SHIP, naming the failed clause. R1 / R2 (no learned parameters) are decided on the "
        "TRAIN docs only; R3 (shapes learned from gold) is decided supplier-held-out, folds "
        "pooled, on the train + dev docs of `splits/folds.json`.\n"
    )
    w.append(render_section(*main, smoke=smoke))
    note = ocr_dependence(main[2])
    if note:
        w.append("## R2 deployment cost\n")
        w.append(
            "R2 needs OCR (PaddleOCR, run on Colab for the test set) at inference time, a "
            "different deployment cost from the image-only pick. " + note + "\n"
        )
    if r3 is not None:
        w.append(
            "# R3 supplier-held-out\n\n"
            "Shapes for each fold come only from the gold of the OTHER folds' docs (supplier "
            "groups are disjoint across folds, checked at run time); the folds are pooled so "
            "every doc is evaluated once. The dev docs are inside this pooling, so R3 has no "
            "separate dev confirmation.\n"
        )
        w.append(render_section(*r3, smoke=smoke))
    if confirm is not None:
        w.append(
            "# CONFIRMATION ONLY (held-out dev, frozen R1 / R2)\n\n"
            "Evaluated after the train verdict above. It never changes that verdict. R3 is not "
            "repeated here: its dev docs are already inside the held-out pooled section.\n"
        )
        w.append(render_section(*confirm))
    return "\n".join(w)


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def doc_ids_with_prefix(run_dir: Path, prefix: str) -> list[str]:
    """Sorted doc ids of the run's predictions that start with `prefix`."""
    pred = json.loads((run_dir / "predictions.json").read_text(encoding="utf-8"))
    return sorted(d for d in pred if d.startswith(prefix))


def load_doc_fold() -> dict[str, int]:
    """doc_id -> fold of ``splits/folds.json`` (supplier-grouped, train + dev)."""
    raw = json.loads((ROOT / "splits" / "folds.json").read_text(encoding="utf-8"))
    return {str(d): int(f) for d, f in raw["doc_fold"].items()}


def require_fold_docs(present: Sequence[str], doc_fold: Mapping[str, int]) -> None:
    """Fail closed unless the run holds exactly the docs of `doc_fold` (R3 needs all of them)."""
    missing = sorted(set(doc_fold) - set(present))
    if missing:
        raise SystemExit(
            f"R3 held-out needs all {len(doc_fold)} docs of folds.json in the run; "
            f"{len(missing)} missing, e.g. {missing[:3]}"
        )


# --------------------------------------------------------------------------------------------
# OOF mode: the same frozen rules on the fine-tuned model's fold-K held-out outputs
# --------------------------------------------------------------------------------------------


def failure_kind(r: RuleResult) -> str:
    """Why a rule did not ship: '' (it ships), FAILED (...) or the insufficient-power wording.

    Insufficient power = nothing broke, the point delta is positive, and only the CI clause fails
    (lower bound <= 0): the evidence cannot separate the gain from 0, it does not show harm.
    """
    v = r.verdict
    if v.ship:
        return ""
    if r.outcome.broken > 0:
        return FAILED_BROKEN
    if CLAUSE_GROUP in v.failed:
        return "FAILED (a supplier group has net < 0)"
    if r.stats["delta"] > 0:
        return INSUFFICIENT_POWER
    return "FAILED (no positive effect: point delta <= 0 and CI lower bound <= 0)"


def failure_sentence(r: RuleResult) -> str:
    """One distinguishing sentence from the numbers of a non-shipping rule ('' if it ships)."""
    kind = failure_kind(r)
    if not kind:
        return ""
    o, s = r.outcome, r.stats
    nums = (
        f"broken {o.broken}, fixed {o.fixed}, point delta {pts(s['delta'])} pts, CI lower bound "
        f"{pts(s['lo'])} pts"
    )
    if kind == INSUFFICIENT_POWER:
        return (
            f"{kind}: {nums}. Nothing broke and the point estimate is positive, but with this "
            "few docs the CI includes 0; this is a lack of evidence, not evidence of harm."
        )
    return f"{kind}: {nums}."


def fold_docs(folds: Mapping[str, Any], fold: int, doc_fold: Mapping[str, int]) -> list[str]:
    """Sorted held-out doc ids of fold `fold`, cross-checked against the ``doc_fold`` map."""
    got = [f for f in folds["folds"] if int(f["fold"]) == fold]
    if len(got) != 1:
        raise SystemExit(f"fold {fold} is not in splits/folds.json")
    ids = sorted(got[0]["val_doc_ids"])
    if set(ids) != {d for d, f in doc_fold.items() if f == fold}:
        raise SystemExit(f"folds.json is inconsistent for fold {fold} (folds vs doc_fold)")
    return ids


def require_complete(run_dir: Path, label: str) -> None:
    """Fail closed unless ``progress.json`` of `run_dir` says ``complete``."""
    prog = run_dir / "progress.json"
    status = None
    if prog.is_file():
        status = json.loads(prog.read_text(encoding="utf-8")).get("status")
    if status != "complete":
        raise SystemExit(f"{label} {run_dir.name}: run is not complete (status {status!r})")


def require_same_resolution(runs: Mapping[str, Path]) -> None:
    """Fail closed unless all run folders share one config hash (one input resolution)."""
    try:
        assert_runs_share_resolution(runs)
    except ResolutionMismatchError as e:
        raise SystemExit(str(e)) from e


def check_oof_manifest(
    man: Mapping[str, Any] | None, fold_arg: int | None, plumbing: bool
) -> tuple[int, dict[str, Any]]:
    """(fold K, oof section) of a verified OOF manifest; K is cross-checked against `fold_arg`.

    Refuses a manifest without the ``oof`` section, a failed adapter verification, or a missing
    adapter / training-code sha. `plumbing` skips all of that (K = `fold_arg`, default 0).
    """
    if plumbing:
        return (0 if fold_arg is None else fold_arg), {}
    sec = (man or {}).get("oof")
    if not isinstance(sec, dict):
        raise SystemExit("manifest.json has no `oof` section: not an OOF run folder")
    if not isinstance(sec.get("fold"), int):
        raise SystemExit("manifest oof section has no integer `fold`")
    fold = int(sec["fold"])
    if fold_arg is not None and fold_arg != fold:
        raise SystemExit(f"--fold {fold_arg} contradicts the manifest's oof fold {fold}")
    if not (sec.get("verification") or {}).get("ok"):
        raise SystemExit("manifest oof verification is not ok: refusing an unverified adapter")
    if not sec.get("adapter_sha256") or not sec.get("train_code_sha"):
        raise SystemExit("manifest oof section lacks the adapter sha or the training code sha")
    return fold, sec


def require_exact_docs(found: Sequence[str], want: Sequence[str], label: str, exact: bool) -> None:
    """Fail closed unless `found` holds exactly `want` (`exact`) or at least all of `want`."""
    miss, extra = set(want) - set(found), set(found) - set(want)
    if miss or (exact and extra):
        raise SystemExit(
            f"{label}: doc ids are not the fold's docs ({len(miss)} missing, "
            f"{len(extra) if exact else 0} extra, fold has {len(want)})"
        )


def require_oof_inputs(run_dir: Path, ids: Sequence[str], exact: bool, label: str) -> None:
    """Fail closed on doc ids / traces; OCR page completeness is checked by `require_ocr`."""
    pred_path = run_dir / "predictions.json"
    if not pred_path.is_file() or not (run_dir / "trace.jsonl").is_file():
        raise SystemExit(f"{label}: predictions.json or trace.jsonl is missing")
    pred = json.loads(pred_path.read_text(encoding="utf-8"))
    require_exact_docs(list(pred), ids, f"{label} predictions.json", exact)
    traces = [t["doc_id"] for t in read_trace(run_dir / "trace.jsonl")]
    require_exact_docs(traces, ids, f"{label} trace.jsonl", exact)


def require_ocr(run_dir: Path, inp: Inputs, ocr_cache: Path | None) -> None:
    """Fail closed unless every waybill doc has all its pages in the OCR cache (R2 eligibility).

    Page count must equal the trace's page count; a doc with no cached page, or fewer pages than
    the model saw, would silently shrink R2's eligibility.
    """
    pages = {t["doc_id"]: len(t.get("pages") or []) for t in read_trace(run_dir / "trace.jsonl")}
    bad = [
        d
        for d, g in sorted(inp.gold.items())
        if g["doc_type"] == "waybill" and len(doc_pages(d, ocr_cache)) != pages[d]
    ]
    if bad:
        raise SystemExit(
            f"OCR cache lacks waybill pages for {len(bad)} docs (R2 would shrink), e.g. {bad[:3]}"
        )


def run_oof_rules(
    inp: Inputs,
    fold: int,
    labels: Sequence[Mapping[str, Any]],
    doc_fold: Mapping[str, int],
    groups: Mapping[str, str],
    n_boot: int,
) -> list[RuleResult]:
    """R1 / R2 / R2-ocrfree on `inp` plus R3 with shapes learned EXCLUDING fold `fold`'s suppliers.

    `inp` must hold only fold `fold` docs. The learner is `fold_shapes` (gold of the other folds
    only, supplier-disjointness checked fail-closed -> SystemExit).
    """
    stray = sorted(d for d in inp.pred if doc_fold.get(d) != fold)
    if stray:
        raise SystemExit(f"{len(stray)} evaluated docs are not in fold {fold}, e.g. {stray[:3]}")
    try:
        per_fold = fold_shapes(labels, doc_fold, groups)
    except ValueError as e:
        raise SystemExit(str(e)) from e
    r3 = evaluate_r3_heldout(inp, {fold: per_fold[fold]}, doc_fold, n_boot)
    r3 = replace(
        r3,
        name=f"R3 (shapes learned excluding fold {fold}'s suppliers)",
        note="shapes learned only from the gold of the other folds' docs (supplier-disjoint); "
        "VERDICT BASIS",
    )
    return [*run_rules(inp, n_boot), r3]


def render_side_by_side(zs_name: str, ft: Sequence[RuleResult], zs: Sequence[RuleResult]) -> str:
    """Zero-shot vs fine-tuned effect of the same rules on the same docs (counts, no values)."""
    by = {r.name: r for r in zs}
    rows = []
    for r in ft:
        z = by.get(r.name)
        if z is None:
            continue
        rows.append(
            [
                r.name, z.outcome.fixed, r.outcome.fixed, z.outcome.broken, r.outcome.broken,
                z.outcome.net, r.outcome.net, pts(z.stats["delta"]), pts(r.stats["delta"]),
            ]
        )  # fmt: skip
    head = [
        "rule", "fixed ZS", "fixed FT", "broken ZS", "broken FT", "net ZS", "net FT",
        "d OVERALL ZS (pts)", "d OVERALL FT (pts)",
    ]  # fmt: skip
    return (
        f"## Same rules on the zero-shot run `{zs_name}` (same docs)\n\n"
        "Side by side: ZS = zero-shot 02 run, FT = fine-tuned fold-adapter outputs. A lower FT "
        "`fixed` count than ZS means the fine-tuned model already produces what the rule would "
        "have repaired. Counts only; the ZS column is context, never part of the verdict.\n\n"
        + md_table(head, rows)
    )


def render_oof_report(
    run_name: str,
    fold: int,
    oof_meta: Mapping[str, Any],
    n_docs: int,
    ft: Sequence[RuleResult],
    zs: tuple[str, Sequence[RuleResult]] | None = None,
) -> str:
    """Aggregates-only report of the OOF-mode gate (verdicts on the fine-tuned outputs only)."""
    w: list[str] = [f"# Merge-rule gate on the fine-tuned model, fold {fold} OOF outputs\n"]
    w.append(
        f"**Provenance.** OOF run `{run_name}`, generated by `uv run python "
        f"scripts/rule_gate.py --oof-run-dir` (repo state `{git_state()}`; paired bootstrap "
        f"{N_BOOT} doc-level resamples, seed {SEED}; unmodified scorer via `shipdoc.eval`). "
        "**All numbers UNVERIFIED** until a verifier recomputes them. Aggregates and counts "
        "only.\n"
    )
    batch = oof_meta.get("batch") or {}
    guard = oof_meta.get("guard") or {}
    w.append(
        "**Run.** no oof section (plumbing check).\n"
        if not oof_meta
        else f"**Run.** fold {oof_meta.get('fold')}; "
        f"adapter sha256 `{oof_meta.get('adapter_sha256')}`; "
        f"training code SHA `{oof_meta.get('train_code_sha')}`; batch used {batch.get('used')}; "
        f"guard ran {guard.get('ran')}, fallback to batch 1 {guard.get('fallback_to_1')}; "
        f"adapter verification ok {(oof_meta.get('verification') or {}).get('ok')}.\n"
    )
    w.append(
        "**Policy.** The frozen rules are unchanged (no re-tuning on these outputs). A rule ships "
        "with the fine-tuned model only if it passes on this fold's OOF outputs too (in addition "
        "to its zero-shot gate). Ship rule as before: paired 95% CI lower bound > 0 AND broken == "
        "0 (R3 also no group with net < 0). R1 / R2 have no learned parameters and run on this "
        f"fold's {n_docs} held-out docs; R3 shapes come only from the other folds' gold.\n"
    )
    w.append(
        f"**Power caveat.** With {n_docs} docs R1 / R2 may lack power. A rule whose CI includes 0 "
        f"with 0 broken (point delta > 0) is reported '{INSUFFICIENT_POWER}'; one with broken > 0 "
        f"is '{FAILED_BROKEN}'. The verdict string is DO NOT SHIP either way.\n"
    )
    w.append(render_section(f"Fold {fold} OOF verdict ({n_docs} docs)", n_docs, ft))
    w.append("## Verdict detail\n")
    for r in ft:
        w.append(f"- **{r.name}**: " + (failure_sentence(r) or "SHIP on this fold's OOF outputs."))
    w.append("")
    if zs is not None:
        w.append(render_side_by_side(zs[0], ft, zs[1]))
    return "\n".join(w)


def main_oof(a: argparse.Namespace) -> int:
    """OOF mode entry; writes ``reports/rule_gate_ft_fold<K>.md`` only when every check passed."""
    run_dir: Path = a.oof_run_dir
    man = read_manifest(run_dir) if run_dir.is_dir() else None
    fold, sec = check_oof_manifest(man, a.fold, a.plumbing_check)
    require_complete(run_dir, "OOF")
    if a.zs_run_dir is not None:  # before any rule run: a mixed-resolution side-by-side is refused
        require_same_resolution({"OOF run": run_dir, "ZS run": a.zs_run_dir})
    folds = json.loads((ROOT / "splits" / "folds.json").read_text(encoding="utf-8"))
    doc_fold = load_doc_fold()
    ids = fold_docs(folds, fold, doc_fold)
    exact = not a.plumbing_check  # plumbing: the 02 run holds all 500 docs, restrict to the fold
    require_oof_inputs(run_dir, ids, exact, "OOF run")
    if sec and sec.get("n_inference_docs") not in (None, len(ids)):
        raise SystemExit("manifest n_inference_docs differs from the fold's doc count")
    split_of = lambda d: d.split("_", 1)[0]  # noqa: E731
    inp = load_inputs(run_dir, ids, a.ocr_cache, split_of)
    require_ocr(run_dir, inp, a.ocr_cache)
    labels = meta_mod.load_labels("train") + meta_mod.load_labels("dev")
    groups = json.loads((ROOT / "meta" / "supplier_groups.json").read_text(encoding="utf-8"))
    ft = run_oof_rules(inp, fold, labels, doc_fold, groups, a.n_boot)
    zs = None
    if a.zs_run_dir is not None:
        require_complete(a.zs_run_dir, "zero-shot")
        require_oof_inputs(a.zs_run_dir, ids, False, "zero-shot run")
        zs_inp = load_inputs(a.zs_run_dir, ids, a.ocr_cache, split_of)
        require_ocr(a.zs_run_dir, zs_inp, a.ocr_cache)
        zs = (
            a.zs_run_dir.name,
            run_oof_rules(zs_inp, fold, labels, doc_fold, groups, a.n_boot),
        )
    text = render_oof_report(run_dir.name, fold, sec, len(ids), ft, zs)
    if a.plumbing_check:
        print(PLUMBING_BANNER + "\n" + text)
        return 0
    out = a.out if a.out is not None else ROOT / "reports" / f"rule_gate_ft_fold{fold}.md"
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", type=Path, required=False)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--oof-run-dir", type=Path, default=None, help="OOF mode: fold-K run folder")
    ap.add_argument("--fold", type=int, default=None, help="OOF mode: cross-check vs manifest")
    ap.add_argument("--zs-run-dir", type=Path, default=None, help="OOF mode: the 02 run folder")
    ap.add_argument("--plumbing-check", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--ocr-cache", type=Path, default=None)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--confirm-dev", action="store_true", help="also evaluate frozen rules on dev")
    ap.add_argument(
        "--allow-dev-smoke",
        action="store_true",
        help="evaluate the DEV docs instead of train (NOT A GATE RESULT; writes no report)",
    )
    a = ap.parse_args(argv)
    if a.oof_run_dir is not None:
        if a.run_dir is not None or a.allow_dev_smoke or a.confirm_dev:
            ap.error("--oof-run-dir excludes --run-dir / --allow-dev-smoke / --confirm-dev")
        return main_oof(a)
    if a.run_dir is None:
        ap.error("--run-dir is required (or --oof-run-dir)")
    if a.fold is not None or a.zs_run_dir is not None or a.plumbing_check:
        ap.error("--fold / --zs-run-dir / --plumbing-check need --oof-run-dir")
    if a.out is None:
        a.out = ROOT / "reports" / "rule_gate.md"
    if a.allow_dev_smoke and a.confirm_dev:
        ap.error("--confirm-dev is meaningless with --allow-dev-smoke")
    prefix = "dev_" if a.allow_dev_smoke else "train_"
    ids = doc_ids_with_prefix(a.run_dir, prefix)
    if not a.allow_dev_smoke and len(ids) != EXPECTED_TRAIN_DOCS:
        # fail closed: a partial run must not produce a verdict
        raise SystemExit(f"expected {EXPECTED_TRAIN_DOCS} train docs in the run, found {len(ids)}")
    if not ids:
        raise SystemExit(f"no {prefix}* docs in {a.run_dir / 'predictions.json'}")
    split_of = lambda d: d.split("_", 1)[0]  # noqa: E731  (train_0001 -> train)
    doc_fold = load_doc_fold()
    if a.allow_dev_smoke:
        r3_ids = ids  # smoke: whatever dev docs the run holds, other-fold gold still learned from
    else:
        r3_ids = sorted(doc_fold)
        require_fold_docs(doc_ids_with_prefix(a.run_dir, ""), doc_fold)
    inp = load_inputs(a.run_dir, ids, a.ocr_cache, split_of)
    r3_inp = inp if r3_ids == ids else load_inputs(a.run_dir, r3_ids, a.ocr_cache, split_of)
    title = (
        "DEV SMOKE (NOT A GATE RESULT)" if a.allow_dev_smoke else "R1 / R2 TRAIN verdict (400 docs)"
    )
    main_sec = (title, len(ids), run_rules(inp, a.n_boot))
    labels = meta_mod.load_labels("train") + meta_mod.load_labels("dev")
    r3_title = (
        "R3 supplier-held-out, smoke subset (NOT A GATE RESULT)"
        if a.allow_dev_smoke
        else f"R3 supplier-held-out, {len(set(doc_fold.values()))} folds pooled, {len(r3_ids)} docs"
    )
    groups = json.loads((ROOT / "meta" / "supplier_groups.json").read_text(encoding="utf-8"))
    r3_res = run_r3(r3_inp, labels, doc_fold, groups, a.n_boot)
    r3_sec = (r3_title, len(r3_ids), r3_res)
    confirm = None
    if a.confirm_dev:
        dev_ids = doc_ids_with_prefix(a.run_dir, "dev_")
        if not dev_ids:
            raise SystemExit("--confirm-dev: no dev_* docs in the run")
        dev_inp = load_inputs(a.run_dir, dev_ids, a.ocr_cache, split_of)
        confirm = ("Dev confirmation", len(dev_ids), run_rules(dev_inp, a.n_boot))
    text = render_report(a.run_dir.name, main_sec, confirm, a.allow_dev_smoke, r3_sec)
    if a.allow_dev_smoke:
        print(text)
        return 0
    a.out.write_text(text, encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
