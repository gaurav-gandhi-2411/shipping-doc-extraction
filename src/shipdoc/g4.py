"""Fold-K G4 analysis: pure helpers (verdicts, rule-effect tables, markdown) for ``g4_fold.py``.

The G4 comparison puts BOTH arms through the PRODUCTION post-processing (R1 / R2 / R3, default
on): fine-tuned + rules vs zero-shot + rules on fold K's held-out documents. Raw vs raw is the
secondary view. The paired comparison itself is ``shipdoc.oof.compare_models`` (doc-level paired
bootstrap, seed 42, unmodified scorer): this module does not re-implement it, it only adds

* `spec_g4_clause`: the spec's G4 keep clause (reference only, one fold, never the decision);
* `rules_effect`: per rule eligible / touched docs and fixed / broken / neutral cells per arm;
* `render_report`: the aggregates-only markdown (no gold or predicted value, no doc id).

Sign convention everywhere: delta = fine-tuned (FT) minus zero-shot (ZS), so a positive OVERALL
delta favours the fine-tuned model; for ``false_fill_rate`` lower is better (a positive delta is
worse). The oof module calls the fine-tuned arm ``oof``; here that key is the FT arm.
All numbers are UNVERIFIED until a verifier recomputes them.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from shipdoc import eval as ev
from shipdoc import oof

SEED = 42
N_BOOT = 2000
RULES = ("R1", "R2", "R3")
SPEC_LABEL = "interim, one fold, not the G4 decision"
METRIC_LABELS = (
    ("OVERALL", "OVERALL"),
    ("header_field_accuracy", "header accuracy"),
    ("row_f1", "row F1"),
    ("documents_fully_correct", "documents fully correct"),
    ("false_fill_rate", "false-fill rate (lower is better)"),
)
SLICES = (
    ("invoices", "invoice (HEADLINE slice)"),
    ("waybills", "waybill"),
    ("scanned", "scanned"),
    ("digital", "digital"),
    ("all", "all docs"),
)
ARMS = ("zs_rules", "ft_rules", "zs_raw", "ft_raw")
ARM_LABELS = {
    "zs_rules": "ZS + rules",
    "ft_rules": "FT + rules",
    "zs_raw": "ZS raw",
    "ft_raw": "FT raw",
}


# --------------------------------------------------------------------------------------------
# Verdicts
# --------------------------------------------------------------------------------------------


def interim_verdict(
    delta_ci_lo: float, ft_over_null: int, zs_over_null: int, delta_ci_hi: float | None = None
) -> dict[str, Any]:
    """The interim G4 verdict defined for notebook 05 (``shipdoc.oof.g4_interim_verdict``).

    NO REGRESSION iff the paired OVERALL delta (FT - ZS) CI lower bound is strictly above -1.0
    point AND the FT over-null cell count (header + row) is <= the ZS one. A lower bound of
    exactly -0.01 is REGRESSION (strict inequality), equal over-null counts pass.
    """
    return oof.g4_interim_verdict(delta_ci_lo, ft_over_null, zs_over_null, delta_ci_hi)


def interim_line(v: Mapping[str, Any], basis: str) -> str:
    """The printed interim verdict line, with the arms it was computed on appended."""
    return f"{oof.verdict_line(v)} [basis: {basis}]"


def spec_g4_clause(
    delta_ci_lo: float, ft_false_fill: float, zs_false_fill: float
) -> dict[str, Any]:
    """The spec's G4 clause: fine-tuned KEPT only if OVERALL beats ZS + rules with the paired CI
    excluding 0 (lower bound strictly > 0) AND false-fill is not worse (FT rate <= ZS rate).

    Reference only: one fold is not the G4 decision (`SPEC_LABEL`).
    """
    failed = []
    if not delta_ci_lo > 0:
        failed.append(
            f"paired OVERALL delta CI lower bound {100 * delta_ci_lo:+.2f} points is not above 0"
        )
    if ft_false_fill > zs_false_fill:
        failed.append(
            f"false-fill rate {100 * ft_false_fill:.2f}% (FT) > {100 * zs_false_fill:.2f}% (ZS)"
        )
    return {
        "verdict": "FINE-TUNED KEPT" if not failed else "FINE-TUNED NOT KEPT",
        "failed_clauses": failed,
        "delta_ci_lo": delta_ci_lo,
        "ft_false_fill": ft_false_fill,
        "zs_false_fill": zs_false_fill,
        "label": SPEC_LABEL,
    }


def spec_line(v: Mapping[str, Any]) -> str:
    """The printed spec-clause line (labelled interim, not the G4 decision)."""
    why = "" if v["verdict"] == "FINE-TUNED KEPT" else " (" + "; ".join(v["failed_clauses"]) + ")"
    return f"G4 SPEC CLAUSE ({v['label']}): {v['verdict']}{why}"


def verdicts(rules_pair: Mapping[str, Any], raw_pair: Mapping[str, Any]) -> dict[str, Any]:
    """Interim verdict on the rules arms (PRIMARY) and the raw arms, plus the spec clause."""
    out: dict[str, Any] = {}
    for key, pair in (("interim_rules", rules_pair), ("interim_raw", raw_pair)):
        a = pair["subsets"]["all"]
        d = a["paired_delta_oof_minus_zero_shot"]["OVERALL"]
        out[key] = interim_verdict(
            d["lo"],
            a["oof"]["over_null"]["over_null_total"],
            a["zero_shot"]["over_null"]["over_null_total"],
            d["hi"],
        )
    a = rules_pair["subsets"]["all"]
    out["spec_rules"] = spec_g4_clause(
        a["paired_delta_oof_minus_zero_shot"]["OVERALL"]["lo"],
        a["oof"]["false_fill_rate"],
        a["zero_shot"]["false_fill_rate"],
    )
    return out


# --------------------------------------------------------------------------------------------
# Rule effect (per arm)
# --------------------------------------------------------------------------------------------


def rules_effect(
    sc: Any,
    raw: Mapping[str, Any],
    final: Mapping[str, Any],
    gold: Mapping[str, Any],
    summary: Mapping[str, Any],
    count_changes: Callable[[Any, Any, Any, Any], Mapping[str, Mapping[str, Mapping[str, int]]]],
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict[str, Any]:
    """Per rule: eligible / touched docs, changed cells, fixed / broken / neutral; plus the paired
    OVERALL delta of rules-on minus rules-off on this arm.

    `raw` is the production path with every rule OFF, `final` with the rules ON (same arm, same
    docs); `summary` is ``postprocess_traces``' summary of the rules-ON run; `count_changes` is
    ``replay_v1_check.count_changes`` (cell diffs raw -> final scored against `gold`).
    """
    counts = count_changes(sc, dict(raw), dict(final), dict(gold))
    per_rule = {}
    for r in RULES:
        by_split = counts[r]
        tot = {k: sum(x[k] for x in by_split.values()) for k in ("fixed", "broken", "neutral")}
        per_rule[r] = {
            "eligible_docs": summary["eligible_docs"][r],
            "touched_docs": summary["touched_docs"][r],
            "cells_changed": summary["changes"][r],
            **tot,
            "net": tot["fixed"] - tot["broken"],
        }
    pb = ev.paired_bootstrap(dict(raw), dict(final), dict(gold), n=n_boot, seed=seed)["OVERALL"]
    return {
        "per_rule": per_rule,
        "skipped": dict(summary.get("skipped") or {}),
        "overall_off": pb["a"],
        "overall_on": pb["b"],
        "delta": pb["delta"],
        "lo": pb["lo"],
        "hi": pb["hi"],
    }


# --------------------------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------------------------


def md_table(head: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Markdown table; every cell is str()-ed."""
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def pct(x: float) -> str:
    """Fraction -> percent, two decimals."""
    return f"{100 * x:.2f}"


def pts(x: float) -> str:
    """Fraction -> signed percentage points, two decimals."""
    return f"{100 * x:+.2f}"


def excludes_zero(lo: float, hi: float) -> str:
    """CI-excludes-0 flag with the sign of the side it excludes (FT - ZS)."""
    if lo > 0:
        return "yes (positive)"
    if hi < 0:
        return "yes (negative)"
    return "no"


def _ci(ci: Mapping[str, float]) -> str:
    return f"{pct(ci['point'])} [{pct(ci['lo'])}, {pct(ci['hi'])}]"


def _delta(d: Mapping[str, float]) -> str:
    return f"{pts(d['delta'])} [{pts(d['lo'])}, {pts(d['hi'])}]"


def headline_table(pair: Mapping[str, Any], zs_name: str, ft_name: str) -> str:
    """Per metric: each arm's point + 95% CI, paired delta (FT - ZS) + CI, CI-excludes-0 flag."""
    sub = pair["subsets"]["all"]
    rows = []
    for key, label in METRIC_LABELS:
        d = sub["paired_delta_oof_minus_zero_shot"][key]
        rows.append(
            [
                label,
                _ci(sub["zero_shot"]["ci95"][key]),
                _ci(sub["oof"]["ci95"][key]),
                _delta(d),
                excludes_zero(d["lo"], d["hi"]),
            ]
        )
    head = ["metric (percent)", zs_name, ft_name, "delta FT - ZS (pts) [95% CI]", "CI excludes 0"]
    return md_table(head, rows)


def slice_table(pair: Mapping[str, Any]) -> str:
    """Per slice and metric: ZS, FT (points), paired delta FT - ZS with its CI."""
    rows = []
    for sname, slabel in SLICES:
        sub = pair["subsets"].get(sname)
        if sub is None:
            rows.append([slabel, 0, "(no docs)", "", "", "", ""])
            continue
        for key, label in METRIC_LABELS:
            d = sub["paired_delta_oof_minus_zero_shot"][key]
            rows.append(
                [slabel, sub["n_docs"], label, pct(d["a"]), pct(d["b"]), _delta(d),
                 excludes_zero(d["lo"], d["hi"])]
            )  # fmt: skip
    head = ["slice", "docs", "metric", "ZS", "FT", "delta FT - ZS (pts) [95% CI]", "CI excludes 0"]
    return md_table(head, rows)


def over_null_table(blocks: Mapping[str, Mapping[str, Any]]) -> str:
    """Over-null / false-fill counts per field for the four arms (`blocks` keyed by `ARMS`)."""
    ov = {a: blocks[a]["over_null"] for a in ARMS}
    hdr_fields = sorted({f for a in ARMS for f in ov[a]["header_over_null_by_field"]})
    row_fields = sorted({f for a in ARMS for f in ov[a]["row_over_null_by_field"]})

    def line(label: str, get: Callable[[Mapping[str, Any]], int]) -> list[Any]:
        vals = {a: get(ov[a]) for a in ARMS}
        return [label, *(vals[a] for a in ARMS), vals["ft_rules"] - vals["zs_rules"]]

    rows = [
        line(f"header over-null: {f}", lambda o, f=f: o["header_over_null_by_field"].get(f, 0))
        for f in hdr_fields
    ]
    rows.append(line("header over-nulls, total", lambda o: o["header_over_null"]))
    rows += [
        line(
            f"row over-null (scorer-paired rows): {f}",
            lambda o, f=f: o["row_over_null_by_field"].get(f, 0),
        )
        for f in row_fields
    ]
    rows.append(line("row over-nulls, total", lambda o: o["row_over_null"]))
    rows.append(
        line("OVER-NULL CELLS, header + row (verdict clause)", lambda o: o["over_null_total"])
    )
    rows.append(line("header false fills", lambda o: o["header_false_fill"]))
    rows.append(line("row false fills (scorer-paired rows)", lambda o: o["row_false_fill"]))
    rows.append(line("gold rows left unmatched (missing rows)", lambda o: o["rows_unmatched_gold"]))
    head = ["count", *(ARM_LABELS[a] for a in ARMS), "FT - ZS (rules arms)"]
    return md_table(head, rows)


def rules_table(effect: Mapping[str, Mapping[str, Any]]) -> str:
    """R1 / R2 / R3 touched / fixed / broken per arm (`effect` keyed ``zs`` / ``ft``)."""
    rows = []
    for arm, label in (("zs", "ZS"), ("ft", "FT")):
        for r in RULES:
            c = effect[arm]["per_rule"][r]
            rows.append(
                [label, r, c["eligible_docs"], c["touched_docs"], c["cells_changed"], c["fixed"],
                 c["broken"], c["neutral"], c["net"]]
            )  # fmt: skip
    head = ["arm", "rule", "eligible docs", "touched docs", "cells changed", "fixed", "broken",
            "neutral", "net"]  # fmt: skip
    return md_table(head, rows)


def rules_overall_table(effect: Mapping[str, Mapping[str, Any]]) -> str:
    """OVERALL with rules off vs on per arm, paired delta and CI."""
    rows = []
    for arm, label in (("zs", "ZS"), ("ft", "FT")):
        e = effect[arm]
        rows.append(
            [label, pct(e["overall_off"]), pct(e["overall_on"]),
             f"{pts(e['delta'])} [{pts(e['lo'])}, {pts(e['hi'])}]", excludes_zero(e["lo"], e["hi"])]
        )  # fmt: skip
    return md_table(
        ["arm", "OVERALL rules off", "OVERALL rules on", "delta on - off (pts) [95% CI]",
         "CI excludes 0"],
        rows,
    )  # fmt: skip


def r3_note(effect: Mapping[str, Mapping[str, Any]], fold: int) -> str:
    """Plain statement of whether R3 fired on this fold, per arm."""
    parts = []
    for arm, label in (("zs", "ZS"), ("ft", "FT")):
        c = effect[arm]["per_rule"]["R3"]
        fired = f"touched {c['touched_docs']} docs ({c['cells_changed']} row edits)"
        parts.append(f"{label}: " + ("R3 did NOT fire" if c["touched_docs"] == 0 else fired))
    return (
        f"R3 on fold {fold} (shapes learned excluding this fold's suppliers): "
        + "; ".join(parts)
        + ". R3 is judged only on folds where it fires (GG decision); a fold where it does not "
        "fire says nothing about it.\n"
    )


def cause_table(summ: Mapping[str, Mapping[str, Any]], causes: Sequence[str]) -> str:
    """Row-error causes and waybill over-nulls, four arms side by side (counts)."""

    def line(label: str, get: Callable[[Mapping[str, Any]], int]) -> list[Any]:
        return [label, *(get(summ[a]) for a in ARMS)]

    rows = [
        line("invoice docs diagnosed", lambda s: s["n_invoice_docs"]),
        line("invoice docs with a truncated page", lambda s: s["n_truncated_invoice_docs"]),
        line("unpaired rows linked one-to-one (side pair), total", lambda s: s["side"]["pair"]),
        line("gold rows with no link (row_missing side), total", lambda s: s["side"]["gold"]),
        line("predicted rows with no link (row_extra side), total", lambda s: s["side"]["pred"]),
    ]
    rows += [line(f"cause: {c}", lambda s, c=c: s["causes"].get(c, 0)) for c in causes]
    kinds = sorted({k for a in ARMS for k in summ[a]["slots"]})
    rows += [
        line(f"slot (scorer-paired rows): {k}", lambda s, k=k: s["slots"].get(k, 0)) for k in kinds
    ]
    rows.append(line("waybill docs", lambda s: s["n_waybill_docs"]))
    fields = sorted({f for a in ARMS for f in summ[a]["waybill"]})
    for f in fields:
        for key, what in (
            ("cells", "over-null cells"),
            ("null_emitted", "of which the model emitted a null"),
            ("key_missing", "of which the key is missing from the raw output"),
            ("alt_slot", "of which the value sits in another raw header slot"),
            ("pattern_unique_correct", "of which a unique OCR pattern match is correct"),
        ):
            rows.append(line(f"waybill {f}: {what}", lambda s, f=f, key=key: s["waybill"][f][key]))
    head = ["count", *(ARM_LABELS[a] for a in ARMS)]
    return md_table(head, rows)


def render_report(res: Mapping[str, Any]) -> str:
    """The aggregates-only markdown report of one fold (no values, no doc ids)."""
    fold, n = res["fold"], res["n_docs"]
    prov = res["provenance"]
    v = res["verdicts"]
    w: list[str] = [f"# G4 fold {fold} analysis: fine-tuned + rules vs zero-shot + rules\n"]
    w.append(
        "**All numbers UNVERIFIED** until a verifier recomputes them. Aggregates and counts "
        "only: no gold or predicted value, no document id.\n"
    )
    w.append("## Provenance\n")
    w.append("\n".join(f"- {k}: {val}" for k, val in prov.items()) + "\n")
    w.append(
        f"Docs: {n} held-out docs of fold {fold}. Both arms go through the PRODUCTION path "
        "(`postprocess_traces` -> coerce -> repair on the traces) with the same honest R3 shapes "
        "(learned excluding this fold's suppliers) and the OCR cache for R2. For each arm the "
        "same path with every rule OFF was asserted to reproduce the arm's `predictions.json` on "
        "these docs. Delta = FT minus ZS, paired doc-level bootstrap, unmodified scorer.\n"
    )
    w.append("## Verdicts\n")
    w.append(
        "**1.** " + interim_line(v["interim_rules"], "both arms with production rules, PRIMARY")
    )
    w.append("")
    w.append(
        "   Same verdict on the RAW arms (secondary): " + interim_line(v["interim_raw"], "raw arms")
    )
    w.append("")
    w.append("**2.** " + spec_line(v["spec_rules"]) + " (reference only)\n")
    w.append(
        "Interim verdict: NO REGRESSION iff the paired OVERALL delta CI lower bound > -1.0 point "
        "AND over-null cells (header + row) of FT <= ZS. Spec clause: FT kept only if OVERALL "
        "beats ZS + rules with the paired CI excluding 0 AND false-fill not worse (FT <= ZS).\n"
    )
    w.append("## Headline: FT + rules vs ZS + rules (all docs)\n")
    w.append(headline_table(res["rules_pair"], "ZS + rules (95% CI)", "FT + rules (95% CI)"))
    w.append("## Secondary: raw vs raw (no rules)\n")
    w.append(headline_table(res["raw_pair"], "ZS raw (95% CI)", "FT raw (95% CI)"))
    w.append("## Slices (FT + rules vs ZS + rules)\n")
    w.append(slice_table(res["rules_pair"]))
    w.append("## Over-nulls and false fills (all docs)\n")
    w.append(
        "Over-null = gold has a value, prediction empty. Header per field; row fields on the "
        "rows the scorer pairs (gold rows nobody matched are counted as missing rows). Fields "
        "with a zero count in every column are omitted.\n"
    )
    w.append(over_null_table(res["blocks"]))
    w.append("## Rules effect per arm (production path, rules on vs off)\n")
    w.append(
        "fixed / broken / neutral: cell diffs rules-off -> rules-on scored against gold (R1 "
        "carrier, R2 mawb / hawb, R3 cpn / po of scorer-paired rows; an R3 row the scorer cannot "
        "pair is neutral).\n"
    )
    w.append(rules_table(res["rules_effect"]))
    w.append(rules_overall_table(res["rules_effect"]))
    w.append(r3_note(res["rules_effect"], fold))
    skipped = {a: res["rules_effect"][a]["skipped"] for a in ("zs", "ft")}
    w.append(
        f"Rule skips (reason counts): ZS {skipped['zs'] or 'none'}; FT {skipped['ft'] or 'none'}.\n"
    )
    w.append("## Row-error cause table (taxonomy of `scripts/row_error_diagnosis.py`)\n")
    w.append(
        "Same classification, rerun on each arm's outputs for the fold's docs: invoice rows by "
        "cause, cpn / po slot mix-ups on scorer-paired rows, waybill header over-nulls. 'rules' "
        "columns are the production-processed outputs, 'raw' the model outputs. Counts only.\n"
    )
    w.append(cause_table(res["causes"], res["cause_names"]))
    w.append("## Limits\n")
    w.extend(f"- {x}" for x in res["limits"])
    w.append("")
    return "\n".join(w)
