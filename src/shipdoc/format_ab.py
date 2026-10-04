"""Keyed vs compact output-format A/B for one model x arm on the same docs (spec Phase 2.3b).

Inputs are two finished spike run dirs (``predictions.json`` + ``trace.jsonl``). Everything is
computed from those artifacts and the official scorer; nothing is re-run on a GPU.

Decision rule (verbatim, GG-approved; motivated by Step L root cause 2, rotations)::

    choose KEYED if (row F1 keyed > compact AND the paired CI of delta row F1 excludes 0)
    OR (rotations keyed < rotations compact AND OVERALL keyed >= compact, where "not worse"
    means ...); otherwise choose COMPACT on speed.

"OVERALL not worse" is operationalised as: point delta OVERALL >= 0 OR the paired CI of delta
OVERALL includes 0. Deltas are keyed - compact, paired doc-level bootstrap (2000 resamples,
seed 42, the same resampled docs for both formats).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from shipdoc import diagnostics as dg
from shipdoc import eval as ev
from shipdoc.replay import read_trace

N_RESAMPLES = 2000
SEED = 42
NO_GAIN_REASON = "no gain; compact on speed"
DECISION_RULE = (
    "choose KEYED if (row F1 keyed > compact AND the paired CI of delta row F1 excludes 0) "
    "OR (rotations keyed < rotations compact AND OVERALL keyed >= OVERALL compact, where "
    '"OVERALL not worse" = point delta OVERALL >= 0 OR the paired CI of delta OVERALL '
    "includes 0); otherwise COMPACT on speed"
)


def decide(
    deltas: Mapping[str, Any], rotations_keyed: int, rotations_compact: int
) -> tuple[str, str]:
    """Apply the decision rule to paired deltas (keyed - compact); returns (decision, reason)."""
    row, overall = deltas["row_f1"], deltas["OVERALL"]
    row_gain = row["delta"] > 0 and row["ci95"][0] > 0  # gain, and the CI excludes 0
    overall_not_worse = overall["delta"] >= 0 or overall["ci95"][0] <= 0 <= overall["ci95"][1]
    if row_gain:
        return "keyed", (
            f"row F1 gain {row['delta']:+.4f}, paired 95% CI "
            f"[{row['ci95'][0]:+.4f}, {row['ci95'][1]:+.4f}] excludes 0"
        )
    if rotations_keyed < rotations_compact and overall_not_worse:
        return "keyed", (
            f"rotations {rotations_keyed} < {rotations_compact} and OVERALL not worse "
            f"(delta {overall['delta']:+.4f}, CI [{overall['ci95'][0]:+.4f}, "
            f"{overall['ci95'][1]:+.4f}])"
        )
    return "compact", NO_GAIN_REASON


def _load_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pred = json.loads((Path(run_dir) / "predictions.json").read_text(encoding="utf-8"))
    return pred, read_trace(Path(run_dir) / "trace.jsonl")


def _speed(traces: list[dict[str, Any]]) -> dict[str, float | None]:
    lat = [
        p["meta"]["latency_s"]
        for t in traces
        for p in t["pages"]
        if p["meta"].get("latency_s") is not None
    ]
    return {
        "s_per_page_mean": float(np.mean(lat)) if lat else None,
        "s_per_page_p95": float(np.percentile(lat, 95)) if lat else None,
    }


def _summary(
    run_dir: Path,
    pred: dict[str, Any],
    traces: list[dict[str, Any]],
    gold: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    per = ev.per_doc_results(pred, gold)
    cis = ev.bootstrap_metrics(list(per.values()), n=N_RESAMPLES, seed=SEED)
    rep = ev.score(pred, gold)["all"]
    return {
        "OVERALL": cis["OVERALL"],
        "row_f1": cis["row_f1"],
        "line_item_field_accuracy": rep["line_item_field_accuracy"],
        **dg.run_row_convention_errors(run_dir, gold),
        **_speed(traces),
        "n_docs": len(gold),
        "n_pages": sum(len(t["pages"]) for t in traces),
    }


def compare_formats(
    compact_dir: Path,
    keyed_dir: Path,
    gold: dict[str, dict[str, Any]],
    model: str,
    arm: str,
) -> dict[str, Any]:
    """The A/B result ``{model, arm, keyed, compact, deltas, decision, reason}``.

    Both runs must cover exactly the same doc ids and each must have gold; anything else raises
    ValueError (a partial run must never win on fewer documents).
    """
    cp, ct = _load_run(compact_dir)
    kp, kt = _load_run(keyed_dir)
    if set(cp) != set(kp):
        raise ValueError(
            f"runs cover different docs: compact {len(cp)}, keyed {len(kp)}, "
            f"only-compact {len(set(cp) - set(kp))}, only-keyed {len(set(kp) - set(cp))}"
        )
    missing = [d for d in cp if d not in gold]
    if missing:
        raise ValueError(f"{len(missing)} doc ids have no gold, e.g. {missing[:3]}")
    g = {d: gold[d] for d in sorted(cp)}
    compact = _summary(Path(compact_dir), cp, ct, g)
    keyed = _summary(Path(keyed_dir), kp, kt, g)
    pb = ev.paired_bootstrap(cp, kp, g, n=N_RESAMPLES, seed=SEED)
    deltas: dict[str, Any] = {}
    for k in ("OVERALL", "row_f1"):
        deltas[k] = {
            "delta": pb[k]["delta"],
            "ci95": [pb[k]["lo"], pb[k]["hi"]],
            "ci_excludes_0": bool(pb[k]["lo"] > 0 or pb[k]["hi"] < 0),
        }
    for k in ("rotations", "clean_swaps", "cpn_equals_po"):
        deltas[k] = keyed[k] - compact[k]
    deltas["line_item_field_accuracy"] = {
        f: keyed["line_item_field_accuracy"][f] - compact["line_item_field_accuracy"][f]
        for f in compact["line_item_field_accuracy"]
    }
    decision, reason = decide(deltas, keyed["rotations"], compact["rotations"])
    return {
        "model": model,
        "arm": arm,
        "n_resamples": N_RESAMPLES,
        "seed": SEED,
        "keyed": keyed,
        "compact": compact,
        "deltas": deltas,
        "decision": decision,
        "reason": reason,
    }


def format_table(res: Mapping[str, Any]) -> str:
    """Readable keyed-vs-compact table plus the decision and its reason."""
    k, c, d = res["keyed"], res["compact"], res["deltas"]

    def ci(m: Mapping[str, float]) -> str:
        return f"{m['point']:.4f} [{m['lo']:.4f}, {m['hi']:.4f}]"

    def dci(m: Mapping[str, Any]) -> str:
        star = " *" if m["ci_excludes_0"] else ""
        return f"{m['delta']:+.4f} [{m['ci95'][0]:+.4f}, {m['ci95'][1]:+.4f}]{star}"

    rows = [
        ("OVERALL (95% CI)", ci(c["OVERALL"]), ci(k["OVERALL"]), dci(d["OVERALL"])),
        ("row F1 (95% CI)", ci(c["row_f1"]), ci(k["row_f1"]), dci(d["row_f1"])),
    ]
    for f, cv in c["line_item_field_accuracy"].items():
        kv = k["line_item_field_accuracy"][f]
        rows.append((f"row acc: {f}", f"{cv:.4f}", f"{kv:.4f}", f"{kv - cv:+.4f}"))
    for key in ("rotations", "clean_swaps", "cpn_equals_po"):
        rows.append((key, str(c[key]), str(k[key]), f"{d[key]:+d}"))
    for key in ("s_per_page_mean", "s_per_page_p95"):
        cs, ks = c[key], k[key]
        rows.append(
            (
                key,
                "n/a" if cs is None else f"{cs:.1f}",
                "n/a" if ks is None else f"{ks:.1f}",
                "n/a" if cs is None or ks is None else f"{ks - cs:+.1f}",
            )
        )
    head = ("metric", "compact", "keyed", "keyed - compact")
    w = [max(len(r[i]) for r in [head, *rows]) for i in range(4)]
    n = c["n_docs"]
    lines = [f"format A/B: {res['model']} x {res['arm']} ({n} docs; * = paired CI excludes 0)"]
    lines += ["  " + "  ".join(r[i].ljust(w[i]) for i in range(4)) for r in [head, *rows]]
    lines.append(f"decision: {res['decision']}  reason: {res['reason']}")
    return "\n".join(lines)
