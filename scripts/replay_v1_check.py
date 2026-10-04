"""Replay check of post-processing v1 (R1 + R2 + R3) on the 02 zero-shot run (CPU only, no model).

Runs the PRODUCTION code path (``shipdoc.postrules.postprocess_traces`` on the run's
``trace.jsonl``, then the same ``coerce_predictions`` / ``repair_predictions`` that
``predict.assemble_submission`` applies), NOT ``scripts/rule_gate.py``, and scores it against the
run's own baseline ``predictions.json`` on all 500 train + dev docs with the unmodified scorer
(``shipdoc.eval``) and the paired doc-level bootstrap (2000 resamples, seed 42).

Two variants of the R3 shapes:
  (a) HONEST: shapes of fold k learned only from the gold of the other folds (supplier-held-out);
  (b) SHIPPING: the frozen ``meta/slot_shapes.json`` (learned from all 500 docs: IN-SAMPLE here,
      optimistic).

Sanity asserts: the baseline OVERALL recomputed from ``predictions.json`` equals ``metrics.json``
(0.7658), and the production path with every rule OFF reproduces ``predictions.json`` exactly.
Per-rule fixed / broken counts come from diffing the patched against the baseline predictions
cell by cell (R1 carrier, R2 mawb / hawb, R3 cpn / po of predicted rows) and are compared with the
rule-gate numbers (``reports/rule_gate.md``). Aggregates and counts only: no gold or predicted
value is printed or written. All numbers UNVERIFIED until a verifier recomputes them.

Run: ``uv run python scripts/replay_v1_check.py --run-dir <02 run> --out reports/v1_replay.md``
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from shipdoc import eval as ev
from shipdoc import meta as meta_mod
from shipdoc import paths, postrules, rules
from shipdoc.coerce import coerce_predictions, repair_predictions
from shipdoc.diagnostics import empty
from shipdoc.postrules import RuleConfig
from shipdoc.replay import read_trace

ROOT = Path(__file__).resolve().parents[1]
N_BOOT = 2000
SEED = 42
EXPECTED_DOCS = 500
EXPECTED_BASE_OVERALL = 0.7658  # metrics.json of the 02 run, asserted to 4 decimals
# Rule-gate numbers (reports/rule_gate.md, cited, not recomputed here): fixed counts.
GATE_FIXED = {"R1": {"train": 8, "dev": 4}, "R2": {"train": 20, "dev": 12}, "R3": {"all": 502}}
SHAPES_ALL = "frozen"


def finalize(preds: Mapping[str, Any]) -> dict[str, Any]:
    """The coerce + repair stage of ``predict.assemble_submission``."""
    out, _ = repair_predictions(coerce_predictions(dict(preds)))
    return out


def production(
    traces: Sequence[Mapping[str, Any]],
    cfg: RuleConfig,
    shapes: Any,
    ocr_root: Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """(final predictions, rules summary) of the production post-processing path."""
    preds, _records, summary = postrules.postprocess_traces(traces, cfg, shapes, ocr_root)
    return finalize(preds), summary


def fold_shape_fn(
    labels: Sequence[Mapping[str, Any]],
    doc_fold: Mapping[str, int],
    groups: Mapping[str, str],
) -> Callable[[str], rules.SlotShapes]:
    """doc_id -> shapes learned from the gold of the OTHER folds only (supplier-disjoint)."""
    per_fold: dict[int, rules.SlotShapes] = {}
    for k in sorted(set(doc_fold.values())):
        held = {d for d, f in doc_fold.items() if f == k}
        learn = [x for x in labels if x["doc_id"] in doc_fold and doc_fold[x["doc_id"]] != k]
        ids = {x["doc_id"] for x in learn}
        if ids & held or {groups[d] for d in ids} & {groups[d] for d in held}:
            raise SystemExit(f"fold {k}: shape learner overlaps the held-out docs / suppliers")
        per_fold[k] = rules.learn_slot_shapes(learn)
    return lambda d: per_fold[doc_fold[d]]


def cell_ok(sc: Any, name: str, value: Any, gold_value: Any) -> bool:
    """Cell right under the scorer: both empty, or both set and ``sc.same``."""
    if empty(value) or empty(gold_value):
        return empty(value) and empty(gold_value)
    return bool(sc.same(name, value, gold_value))


def classify(before_ok: bool, after_ok: bool) -> str:
    """fixed / broken / neutral from correctness before and after."""
    if not before_ok and after_ok:
        return "fixed"
    if before_ok and not after_ok:
        return "broken"
    return "neutral"


def count_changes(
    sc: Any, base: Mapping[str, Any], new: Mapping[str, Any], gold: Mapping[str, Any]
) -> dict[str, dict[str, dict[str, int]]]:
    """Per rule and split (train / dev): fixed / broken / neutral from cell diffs base -> new."""
    out: dict[str, dict[str, dict[str, int]]] = {r: {} for r in ("R1", "R2", "R3")}

    def bump(rule: str, doc_id: str, kind: str) -> None:
        split = doc_id.split("_", 1)[0]
        cell = out[rule].setdefault(split, {"fixed": 0, "broken": 0, "neutral": 0})
        cell[kind] += 1

    for d in sorted(base):
        b, n, g = base[d], new[d], gold[d]
        if not isinstance(b, dict) or not isinstance(n, dict):
            continue
        if g["doc_type"] == "waybill":
            for rule, fields in (("R1", ("carrier",)), ("R2", ("mawb", "hawb"))):
                for f in fields:
                    bv, nv = (b.get("header") or {}).get(f), (n.get("header") or {}).get(f)
                    if bv == nv:
                        continue
                    gv = g["header"].get(f)
                    bump(rule, d, classify(cell_ok(sc, f, bv, gv), cell_ok(sc, f, nv, gv)))
            continue
        rows_b = [x for x in (b.get("line_items") or []) if isinstance(x, dict)]
        rows_n = [x for x in (n.get("line_items") or []) if isinstance(x, dict)]
        gr = list(g.get("line_items") or [])
        full, partial, _, _ = ev._pair_rows(sc, rows_b, gr)
        partner = dict(full + partial)
        for i, (rb, rn) in enumerate(zip(rows_b, rows_n, strict=True)):
            pair_b = (rb.get(rules.CPN), rb.get(rules.PO))
            pair_n = (rn.get(rules.CPN), rn.get(rules.PO))
            if pair_b == pair_n:
                continue
            gi = partner.get(i)
            if gi is None:
                bump("R3", d, "neutral")
                continue
            ok_b = cell_ok(sc, rules.CPN, pair_b[0], gr[gi].get(rules.CPN)) and cell_ok(
                sc, rules.PO, pair_b[1], gr[gi].get(rules.PO)
            )
            ok_n = cell_ok(sc, rules.CPN, pair_n[0], gr[gi].get(rules.CPN)) and cell_ok(
                sc, rules.PO, pair_n[1], gr[gi].get(rules.PO)
            )
            bump("R3", d, classify(ok_b, ok_n))
    return out


def delta_stats(
    base: dict[str, Any], new: dict[str, Any], gold: dict[str, Any], n_boot: int
) -> dict[str, float]:
    """OVERALL before / after, delta and its paired bootstrap CI on the docs of `gold`."""
    b = ev.score(base, gold)["all"]
    a = ev.score(new, gold)["all"]
    pb = ev.paired_bootstrap(base, new, gold, n=n_boot, seed=SEED)["OVERALL"]
    return {
        "base": b["OVERALL"],
        "new": a["OVERALL"],
        "delta": a["OVERALL"] - b["OVERALL"],
        "lo": pb["lo"],
        "hi": pb["hi"],
        "ff_base": b["false_fill_rate"],
        "ff_new": a["false_fill_rate"],
    }


def pts(x: float) -> str:
    """Fraction -> signed percentage points, two decimals."""
    return f"{100 * x:+.2f}"


def md_table(head: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Markdown table."""
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def slice_ids(gold: Mapping[str, Any], meta: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    """Slices of the combined delta: doc type, scanned / digital, train / dev."""
    sc_of = {m["doc_id"]: bool(m.get("scanned")) for m in meta}
    ids = list(gold)
    return {
        "all": ids,
        "invoice": [d for d in ids if gold[d]["doc_type"] == "invoice"],
        "waybill": [d for d in ids if gold[d]["doc_type"] == "waybill"],
        "scanned": [d for d in ids if sc_of.get(d)],
        "digital": [d for d in ids if not sc_of.get(d, True)],
        "train": [d for d in ids if d.startswith("train_")],
        "dev": [d for d in ids if d.startswith("dev_")],
    }


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


def run_check(
    run_dir: Path,
    ocr_cache: Path | None,
    n_boot: int = N_BOOT,
    check_baseline: bool = True,
    expected_base_overall: float = EXPECTED_BASE_OVERALL,
) -> dict[str, Any]:
    """All numbers of the report as one dict (aggregates only).

    `expected_base_overall` is the 4-decimal OVERALL of the run's own metrics.json (default: the
    1260-token 02 run; a native run passes its own value, never a copy of the 1260 one).
    """
    sc = ev.load_scorer()
    traces = read_trace(run_dir / "trace.jsonl")
    pred_all = json.loads((run_dir / "predictions.json").read_text(encoding="utf-8"))
    doc_fold = {
        str(d): int(f)
        for d, f in json.loads((ROOT / "splits" / "folds.json").read_text("utf-8"))[
            "doc_fold"
        ].items()
    }
    gold: dict[str, Any] = {}
    for split in ("train", "dev"):
        gold |= ev.load_gold(paths.data_dir() / split / "labels")
    gold = {d: g for d, g in gold.items() if d in pred_all}
    if check_baseline and len(gold) != EXPECTED_DOCS:
        raise SystemExit(f"expected {EXPECTED_DOCS} scored docs, found {len(gold)}")
    traces = [t for t in traces if t["doc_id"] in gold]
    groups = json.loads((ROOT / "meta" / "supplier_groups.json").read_text(encoding="utf-8"))
    base = {d: pred_all[d] for d in gold}
    base_overall = ev.score(base, gold)["all"]["OVERALL"]
    if check_baseline:
        saved = json.loads((run_dir / "metrics.json").read_text("utf-8"))["slices"]["all"]
        assert abs(base_overall - saved["OVERALL"]) < 1e-12, "baseline != metrics.json"
        assert round(base_overall, 4) == expected_base_overall, round(base_overall, 4)
    off, off_sum = production(traces, RuleConfig.all_off(), None, ocr_cache)
    assert off == base, "all rules OFF does not reproduce predictions.json"
    labels = meta_mod.load_labels("train") + meta_mod.load_labels("dev")
    frozen, frozen_sha = postrules.load_slot_shapes()
    honest_fn = fold_shape_fn(labels, doc_fold, groups)
    meta = ev.load_json(ROOT / "meta" / "train.json") + ev.load_json(ROOT / "meta" / "dev.json")
    slices = slice_ids(gold, meta)
    variants: dict[str, Any] = {}
    for name, shapes in (("honest", honest_fn), (SHAPES_ALL, frozen)):
        new, summ = production(traces, RuleConfig(), shapes, ocr_cache)
        singles = {}
        for r in ("r1", "r2", "r3"):
            only = RuleConfig(**{k: k == r for k in ("r1", "r2", "r3")})
            s_new, _ = production(traces, only, shapes, ocr_cache)
            singles[r.upper()] = ev.score(s_new, gold)["all"]["OVERALL"] - base_overall
        p12, _ = production(traces, RuleConfig(r1=True, r2=True, r3=False), shapes, ocr_cache)
        singles["R1+R2"] = ev.score(p12, gold)["all"]["OVERALL"] - base_overall
        per_slice = {}
        for sname, ids in slices.items():
            g_s = {d: gold[d] for d in ids}
            per_slice[sname] = (
                len(ids),
                delta_stats({d: base[d] for d in ids}, {d: new[d] for d in ids}, g_s, n_boot),
            )
        variants[name] = {
            "summary": summ,
            "counts": count_changes(sc, base, new, gold),
            "singles": singles,
            "slices": per_slice,
        }
    return {
        "run": run_dir.name,
        "n_docs": len(gold),
        "base_overall": base_overall,
        "frozen_sha": frozen_sha,
        "variants": variants,
    }


def parse_gate_fixed(items: Sequence[str]) -> dict[str, dict[str, int]]:
    """``R1=train:8,dev:4`` items -> {rule: {split: fixed}} (the rule-gate counts to cite)."""
    out: dict[str, dict[str, int]] = {}
    for it in items:
        rule, _, rest = it.partition("=")
        if rule not in GATE_FIXED or not rest:
            raise SystemExit(f"--gate-fixed {it!r}: expected R1|R2|R3=<split>:<n>[,<split>:<n>]")
        out[rule] = {k: int(v) for k, _, v in (x.partition(":") for x in rest.split(","))}
    if set(out) != set(GATE_FIXED):
        raise SystemExit("--gate-fixed must give all of R1, R2, R3")
    return out


def render(
    res: Mapping[str, Any],
    n_boot: int,
    gate_fixed: Mapping[str, Mapping[str, int]] = GATE_FIXED,
    command: str = "uv run python scripts/replay_v1_check.py",
) -> str:
    """The aggregates-only markdown report."""
    w = [
        "# Post-processing v1 replay check (R1 + R2 + R3)\n",
        f"**Provenance.** Run `{res['run']}`, `{command}` (repo "
        f"state `{git_state()}`; paired bootstrap {n_boot} doc-level resamples, seed {SEED}; "
        "unmodified scorer via `shipdoc.eval`). Production path: `shipdoc.postrules` + the "
        "coerce / repair stage of `predict assemble`; not `scripts/rule_gate.py`. **All numbers "
        "UNVERIFIED** until a verifier recomputes them. Aggregates and counts only.\n",
        f"Docs: {res['n_docs']}. Baseline OVERALL (predictions.json, equals metrics.json to 4 "
        f"decimals): {100 * res['base_overall']:.2f}. The production path with all rules OFF "
        "reproduces predictions.json exactly (asserted). Frozen shapes sha256 "
        f"`{res['frozen_sha']}`.\n",
        "(a) HONEST = R3 shapes of fold k learned from the other folds' gold "
        "(supplier-held-out). (b) SHIPPING = frozen `meta/slot_shapes.json` (learned from these "
        "500 docs: in-sample, optimistic).\n",
    ]
    for name, title in (("honest", "(a) HONEST"), (SHAPES_ALL, "(b) SHIPPING (in-sample)")):
        v = res["variants"][name]
        w.append(f"## {title}\n")
        w.append(
            md_table(
                [
                    "slice",
                    "docs",
                    "base OVERALL",
                    "v1 OVERALL",
                    "d OVERALL (pts)",
                    "95% CI (pts)",
                    "false-fill before",
                    "false-fill after",
                ],
                [
                    [
                        s,
                        n,
                        f"{100 * d['base']:.2f}",
                        f"{100 * d['new']:.2f}",
                        pts(d["delta"]),
                        f"[{pts(d['lo'])}, {pts(d['hi'])}]",
                        f"{d['ff_base']:.4f}",
                        f"{d['ff_new']:.4f}",
                    ]
                    for s, (n, d) in v["slices"].items()
                ],
            )  # fmt: skip
        )
        rows = []
        for r in ("R1", "R2", "R3"):
            c = v["counts"][r]
            tot = {k: sum(x[k] for x in c.values()) for k in ("fixed", "broken", "neutral")}
            gate = gate_fixed[r]
            gate_s = " + ".join(f"{k} {n}" for k, n in gate.items())
            split_s = ", ".join(f"{k} {x['fixed']}" for k, x in sorted(c.items()))
            rows.append([r, tot["fixed"], tot["broken"], tot["neutral"], split_s, gate_s,
                         pts(v["singles"][r])])  # fmt: skip
        w.append("Per rule from the production path (cell diffs vs the baseline):\n")
        w.append(
            md_table(
                [
                    "rule",
                    "fixed",
                    "broken",
                    "neutral",
                    "fixed by split",
                    "gate fixed (cited)",
                    "single-rule d OVERALL (pts)",
                ],
                rows,
            )  # fmt: skip
        )
        sg = v["singles"]
        total_single = sg["R1"] + sg["R2"] + sg["R3"]
        comb = v["slices"]["all"][1]["delta"]
        w.append(
            f"Sum of single-rule deltas {pts(total_single)} pts vs combined {pts(comb)} pts "
            f"(difference {pts(comb - total_single)} pts). R1+R2 together (both only touch "
            f"waybill headers): {pts(sg['R1+R2'])} pts vs R1 + R2 alone {pts(sg['R1'] + sg['R2'])}"
            f" pts: the excess is within the waybill pair (consistent with a doc score that is not "
            "linear in the cell count; mechanism not isolated); R3 (invoices) adds on top "
            f"without interaction (combined minus R1+R2 = {pts(comb - sg['R1+R2'])} vs R3 alone "
            f"{pts(sg['R3'])}).\n"
        )
        s = v["summary"]
        w.append(
            f"Rule touches (docs): {s['touched_docs']}; eligible docs: {s['eligible_docs']}; "
            f"skipped: {s['skipped'] or 'none'}.\n"
        )
    return "\n".join(w)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "v1_replay.md")
    ap.add_argument("--ocr-cache", type=Path, default=None)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument(
        "--expected-base-overall",
        type=float,
        default=EXPECTED_BASE_OVERALL,
        help="4-decimal OVERALL of the run's metrics.json (default: the 1260-token 02 run)",
    )
    ap.add_argument(
        "--gate-fixed",
        action="append",
        default=[],
        metavar="RULE=SPLIT:N,..",
        help="rule-gate fixed counts of THIS run to cite, e.g. R1=train:4,dev:3 (x3, R1 R2 R3)",
    )
    a = ap.parse_args(argv)
    gate = parse_gate_fixed(a.gate_fixed) if a.gate_fixed else GATE_FIXED
    res = run_check(a.run_dir, a.ocr_cache, a.n_boot, expected_base_overall=a.expected_base_overall)
    cmd = "uv run python scripts/replay_v1_check.py " + " ".join(
        argv if argv is not None else sys.argv[1:]
    )
    a.out.write_text(render(res, a.n_boot, gate, cmd), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
