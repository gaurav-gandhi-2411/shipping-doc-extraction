"""GPU-hour estimate for the spike (used by the notebook's estimates cell; runnable locally).

Formula per page: ``s_per_page = prefill_s + n_out / decode_tok_s`` with ``prefill_s`` and
``decode_tok_s`` per config from configs/spike_speed.json and ``n_out`` the compact-format gold
tokens per page (mean, with an upper bound). Config names carry a ``_compact`` suffix; the speed
file is keyed without it. Hours are T4 hours; a compute-unit figure is printed only when
``t4_cu_per_hour`` in the speed file is set (never invented). The smoke stage (3 models x the
smoke5 pages, img_only configs) and ``model_load_s`` per model load (an ESTIMATE) are included
in the stage table, plus the keyed-vs-compact A/B (one keyed spike40 run) and the dev100 top-2 at
both compact and keyed speed. Keyed output length is the json-format gold ``n_out_keyed``
(mean; p99 is the upper bound); the model of the A/B is unknown before the run, so the stage
uses the costliest config's measured speed (worst case).

Run: uv run python scripts/gpu_estimate.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SUFFIX = "_compact"


def base_config(config: str) -> str:
    """Speed-file key of a config name (drops the ``_compact`` suffix)."""
    return config.removesuffix(SUFFIX)


def model_key(config: str) -> str:
    """Model part of ``<model>_<arm>`` (key into ``n_out_compact``)."""
    return base_config(config).rsplit("_img_", 1)[0]


def est_s_per_page(prefill_s: float, decode_tok_s: float, n_out: float) -> float:
    """``prefill_s + n_out / decode_tok_s``."""
    if decode_tok_s <= 0:
        raise ValueError(f"decode_tok_s must be > 0, got {decode_tok_s}")
    return prefill_s + n_out / decode_tok_s


def count_pages(doc_ids: list[str], labels_dir: Path) -> int:
    """Total pages of the docs, from ``pages`` in each gold label file."""
    total = 0
    for d in doc_ids:
        gold = json.loads((labels_dir / f"{d}.json").read_text(encoding="utf-8"))
        total += len(gold["pages"])
    return total


def estimate_config(config: str, speed: dict[str, Any], pages: int) -> dict[str, Any]:
    """Per-page seconds and hours for one config, at mean and upper-bound output length."""
    m = speed["models"][base_config(config)]
    n = speed["n_out_compact"][model_key(config)]
    s_mean = est_s_per_page(m["prefill_s"], m["decode_tok_s"], n["mean"])
    s_ub = est_s_per_page(m["prefill_s"], m["decode_tok_s"], n["upper_bound"])
    return {
        "config": config,
        "s_page": s_mean,
        "s_page_ub": s_ub,
        "hours": s_mean * pages / 3600,
        "hours_ub": s_ub * pages / 3600,
        "source": m["source"],
    }


def estimate_keyed(config: str, speed: dict[str, Any], pages: int) -> dict[str, Any]:
    """Like ``estimate_config`` but for the keyed (json) format: ``n_out_keyed`` tokens/page."""
    m = speed["models"][base_config(config)]
    n = speed["n_out_keyed"][model_key(config)]
    s_mean = est_s_per_page(m["prefill_s"], m["decode_tok_s"], n["mean"])
    s_ub = est_s_per_page(m["prefill_s"], m["decode_tok_s"], n["upper_bound"])
    return {
        "config": base_config(config),
        "s_page": s_mean,
        "s_page_ub": s_ub,
        "hours": s_mean * pages / 3600,
        "hours_ub": s_ub * pages / 3600,
        "source": m["source"],
    }


def ab_cost(configs: list[str], speed: dict[str, Any], pages_spike: int) -> dict[str, float]:
    """T4 hours of the A/B stage: one keyed run on spike40 plus one model load.

    The top model is unknown before the run, so the costliest config's measured prefill_s and
    decode_tok_s are used (worst case), at mean and at p99 keyed output length.
    """
    load_h = float(speed.get("model_load_s", 0)) / 3600
    rows = [estimate_keyed(c, speed, pages_spike) for c in configs]
    return {
        "hours": max(r["hours"] for r in rows) + load_h,
        "hours_ub": max(r["hours_ub"] for r in rows) + load_h,
    }


def dev100_keyed_cost(configs: list[str], speed: dict[str, Any], pages: int) -> dict[str, float]:
    """T4 hours of the dev100 top-2 at keyed speed (worst case: the 2 costliest), with 2 loads."""
    load_h = float(speed.get("model_load_s", 0)) / 3600
    rows = [estimate_keyed(c, speed, pages) for c in configs]
    s = sorted(r["hours"] for r in rows)
    u = sorted(r["hours_ub"] for r in rows)
    return {"hours": sum(s[-2:]) + 2 * load_h, "hours_ub": sum(u[-2:]) + 2 * load_h}


def order_by_estimate(configs: list[str], speed: dict[str, Any]) -> list[str]:
    """Configs sorted cheapest first by estimated s/page (ties keep the given order)."""
    return sorted(configs, key=lambda c: estimate_config(c, speed, 1)["s_page"])


def dev100_cost(rows: list[dict[str, Any]], pages: int) -> dict[str, float]:
    """Hours for running the top-2 on `pages`: cheapest pair to costliest pair (mean and ub)."""
    s = sorted(r["s_page"] for r in rows)
    u = sorted(r["s_page_ub"] for r in rows)
    return {
        "hours_min": sum(s[:2]) * pages / 3600,
        "hours_max": sum(s[-2:]) * pages / 3600,
        "hours_max_ub": sum(u[-2:]) * pages / 3600,
    }


def smoke_configs(configs: list[str], speed: dict[str, Any]) -> list[str]:
    """One img_only compact config per model (the smoke stage), cheapest model first."""
    models = dict.fromkeys(model_key(c) for c in configs)
    return order_by_estimate([f"{m}_img_only{SUFFIX}" for m in models], speed)


def smoke_cost(speed: dict[str, Any], configs: list[str], smoke_pages: int) -> dict[str, float]:
    """T4 hours of the smoke stage: per model, smoke pages x s/page plus one model load."""
    load = float(speed.get("model_load_s", 0))
    rows = [estimate_config(c, speed, smoke_pages) for c in smoke_configs(configs, speed)]
    n = len(rows)
    return {
        "models": float(n),
        "hours": sum(r["hours"] for r in rows) + n * load / 3600,
        "hours_ub": sum(r["hours_ub"] for r in rows) + n * load / 3600,
    }


def stage_table(
    configs: list[str],
    speed: dict[str, Any],
    pages_spike: int,
    pages_dev100: int,
    smoke_pages: int,
) -> list[str]:
    """Stage rows in T4 hours and compute units: smoke, spike40, the keyed-vs-compact A/B, and
    the conditional dev100 top-2 at compact speed and at keyed speed (if the A/B picks keyed).

    Load time (``model_load_s``, an ESTIMATE) is added once per config run: 3 in smoke, one per
    config in spike40, 1 in the A/B, 2 in dev100. The A/B and dev100 use the costliest config(s)
    (worst case; the top model is not known before the run).
    """
    load_h = float(speed.get("model_load_s", 0)) / 3600
    rows = [estimate_config(c, speed, pages_spike) for c in configs]
    sm = smoke_cost(speed, configs, smoke_pages)
    spike = (
        sum(r["hours"] for r in rows) + len(rows) * load_h,
        sum(r["hours_ub"] for r in rows) + len(rows) * load_h,
    )
    d = dev100_cost(rows, pages_dev100)
    dev = (d["hours_max"] + 2 * load_h, d["hours_max_ub"] + 2 * load_h)
    ab = ab_cost(configs, speed, pages_spike)
    devk = dev100_keyed_cost(configs, speed, pages_dev100)
    base = (sm["hours"] + spike[0] + ab["hours"], sm["hours_ub"] + spike[1] + ab["hours_ub"])
    r1, r2 = speed.get("t4_cu_per_hour"), speed.get("t4_cu_per_hour_conservative")

    def cu(rate: float | None, hours: float) -> str:
        return "n/a" if rate is None else f"{hours * rate:.1f}"

    def line(name: str, h: float, ub: float) -> str:
        return (
            f"{name.ljust(42)}  {h:8.2f}  {cu(r1, h):>9}  {cu(r2, h):>9}  "
            f"{ub:9.2f}  {cu(r2, ub):>13}"
        )

    head = (
        f"{'stage'.ljust(42)}  {'T4 h':>8}  {'CU@' + str(r1):>9}  {'CU@' + str(r2):>9}  "
        f"{'T4 h (ub)':>9}  {'CU@' + str(r2) + ' (ub)':>13}"
    )
    n_smoke = int(sm["models"])
    return [
        head,
        line(f"smoke ({smoke_pages} pages x {n_smoke} models)", sm["hours"], sm["hours_ub"]),
        line(f"spike40 ({len(rows)} configs)", spike[0], spike[1]),
        line(f"A/B keyed spike40 (1 run, {pages_spike} pages)", ab["hours"], ab["hours_ub"]),
        line("TOTAL without dev100", base[0], base[1]),
        line("dev100 top-2, compact speed (cond.)", dev[0], dev[1]),
        line("dev100 top-2, keyed speed (cond., worst)", devk["hours"], devk["hours_ub"]),
        line("TOTAL with dev100 (compact)", base[0] + dev[0], base[1] + dev[1]),
        line(
            "TOTAL with dev100 (keyed, worst)", base[0] + devk["hours"], base[1] + devk["hours_ub"]
        ),
    ]


def gpu_warning(gpu_name: str | None) -> str | None:
    """Warning line when the runtime GPU is not a T4 (or could not be read); None for a T4."""
    if gpu_name is None:
        return "WARNING: runtime GPU name could not be read; speeds and CU rates are for a T4."
    if "T4" in gpu_name.upper():
        return None
    return (
        f"WARNING: runtime GPU is {gpu_name!r}, not a T4. The speeds and the CU rates below are "
        "T4 figures; for this GPU they are NOT valid (it will be faster or slower and burn CU at "
        "a different rate)."
    )


def _cu(hours: float, rate: float | None) -> str:
    return "" if rate is None else f" (~{hours * rate:.1f} compute units)"


def report(
    configs: list[str],
    speed: dict[str, Any],
    pages_spike: int,
    pages_dev100: int,
    n_docs_spike: int | None = None,
    smoke_pages: int | None = None,
    gpu_name: str | None = "T4",
) -> str:
    """Printable estimate: per-config table, spike40 total, conditional dev100 top-2 cost.

    With `smoke_pages` it also prints the stage table (smoke + spike40 + conditional dev100) in T4
    hours and compute units at both known rates; `gpu_name` None/non-T4 adds a warning line.
    """
    rows = [estimate_config(c, speed, pages_spike) for c in configs]
    rate = speed.get("t4_cu_per_hour")
    placeholder = any(
        "placeholder" in str(m.get("source", "")).lower() for m in speed["models"].values()
    )
    title = (
        "PLACEHOLDER speeds: see configs/spike_speed.json"
        if placeholder
        else "trace-measured speeds: see configs/spike_speed.json"
    )
    w = max(len(r["config"]) for r in rows)
    head = f"{'config'.ljust(w)}  s/page  s/page(ub)  T4 h (spike40)  T4 h (ub)  source"
    lines = [
        f"GPU-hour estimate ({title})",
        "s_per_page = prefill_s + n_out / decode_tok_s; ub = upper-bound n_out (compact gold p99)",
        f"spike40: {pages_spike} pages" + (f" ({n_docs_spike} docs)" if n_docs_spike else ""),
        head,
    ]
    for r in rows:
        lines.append(
            f"{r['config'].ljust(w)}  {r['s_page']:6.1f}  {r['s_page_ub']:10.1f}  "
            f"{r['hours']:14.2f}  {r['hours_ub']:9.2f}  {r['source']}"
        )
    tot, tot_ub = sum(r["hours"] for r in rows), sum(r["hours_ub"] for r in rows)
    lines.append(
        f"TOTAL spike40: {tot:.2f} T4 h{_cu(tot, rate)}; upper bound {tot_ub:.2f} T4 h"
        f"{_cu(tot_ub, rate)}"
    )
    d = dev100_cost(rows, pages_dev100)
    lines.append(
        f"IF dev100 top-2 runs ({pages_dev100} pages each): {d['hours_min']:.2f} T4 h (2 cheapest)"
        f" to {d['hours_max']:.2f} T4 h (2 costliest), upper bound {d['hours_max_ub']:.2f} T4 h"
        f"{_cu(d['hours_max_ub'], rate)}"
    )
    lines.append("Per-config rows exclude model download/load time and any OOM retry.")
    if rate is None:
        lines.append("Compute units: not shown (t4_cu_per_hour unset in spike_speed.json).")
    if smoke_pages is not None:
        warn = gpu_warning(gpu_name)
        bar = "=" * 100
        lines += ["", bar, "ESTIMATED COMPUTE-UNIT BURN (T4 rates; read before the run)", bar]
        if warn:
            lines.append(warn)
        lines += stage_table(configs, speed, pages_spike, pages_dev100, smoke_pages)
        lines += [
            f"model load: {speed.get('model_load_s', 0)} s per model load "
            f"({speed.get('model_load_s_label', 'ESTIMATE')})",
            f"CU rate {speed.get('t4_cu_per_hour')}/h: {speed.get('t4_cu_source', 'source unset')}",
            f"CU rate {speed.get('t4_cu_per_hour_conservative')}/h: "
            f"{speed.get('t4_cu_conservative_source', 'source unset')}",
            "CU balance: Colab exposes no programmatic balance; check Runtime -> Manage sessions "
            "/ Resources for your balance.",
            bar,
        ]
    return "\n".join(lines)


def zeroshot_scenarios(
    config: str, speed: dict[str, Any], max_new_tokens: int
) -> list[tuple[str, float, float]]:
    """(label, n_out tokens/page, s/page) for one keyed config: gold mean, observed mean, p99, cap.

    "observed" is the mean ``n_output_tokens`` of the spike40 traces of that config (the model
    emits more than the gold length); "cap" is every page running to ``max_new_tokens``.
    """
    base = base_config(config)
    m, n = speed["models"][base], speed["n_out_keyed"][model_key(config)]
    rows = [
        ("gold-length mean", n["mean"]),
        ("observed spike40 mean", m["n_output_tokens_mean"]),
        ("p99 gold length (upper bound)", n["upper_bound"]),
        (f"every page at the cap, {max_new_tokens} tok", float(max_new_tokens)),
    ]
    return [(lab, tok, est_s_per_page(m["prefill_s"], m["decode_tok_s"], tok)) for lab, tok in rows]


def zeroshot_report(
    speed: dict[str, Any],
    config: str,
    pages_full: int,
    pages_smoke: int,
    gpu_name: str | None = "T4",
    max_new_tokens: int = 1536,
) -> str:
    """Printable T4-hour / compute-unit estimate of the zero-shot run (smoke, then full).

    Smoke and the full run are separate subprocesses, so each pays one model load. A resumed
    session pays one more load (``model_load_s``, an ESTIMATE). Logprob capture is assumed free
    (not measured on a GPU). Both CU rates of the speed file are shown; neither is Google-published.
    """
    load_h = float(speed.get("model_load_s", 0)) / 3600
    r1, r2 = speed.get("t4_cu_per_hour"), speed.get("t4_cu_per_hour_conservative")

    def cu(rate: float | None, hours: float) -> str:
        return "n/a" if rate is None else f"{hours * rate:.1f}"

    bar = "=" * 100
    lines = [
        bar,
        f"ESTIMATED T4 HOURS / COMPUTE UNITS: zero-shot {config} (keyed), UNVERIFIED",
        bar,
    ]
    warn = gpu_warning(gpu_name)
    if warn:
        lines.append(warn)
    head = (
        f"{'scenario'.ljust(40)}  {'tok/page':>8}  {'s/page':>7}  {'smoke h':>7}  {'full h':>7}  "
        f"{'total h':>7}  {'CU@' + str(r1):>9}  {'CU@' + str(r2):>9}"
    )
    lines.append(head)
    for label, tok, s_page in zeroshot_scenarios(config, speed, max_new_tokens):
        smoke_h = s_page * pages_smoke / 3600 + load_h
        full_h = s_page * pages_full / 3600 + load_h
        total = smoke_h + full_h
        lines.append(
            f"{label.ljust(40)}  {tok:8.1f}  {s_page:7.1f}  {smoke_h:7.2f}  {full_h:7.2f}  "
            f"{total:7.2f}  {cu(r1, total):>9}  {cu(r2, total):>9}"
        )
    lines += [
        f"pages: {pages_full} full run, {pages_smoke} smoke. s/page = prefill_s + tok/decode_tok_s "
        f"(configs/spike_speed.json, {base_config(config)}).",
        f"model load: {speed.get('model_load_s', 0)} s per load, charged once for smoke and once "
        f"for the full run, plus once per resumed session "
        f"({speed.get('model_load_s_label', 'ESTIMATE')}).",
        "Logprob capture adds a few vector ops per decode step; assumed 0 s here, NOT measured "
        "on a GPU.",
        f"CU rate {r1}/h: {speed.get('t4_cu_source', 'source unset')}",
        f"CU rate {r2}/h: {speed.get('t4_cu_conservative_source', 'source unset')}",
        "CU balance: Colab exposes no programmatic balance; check Runtime -> Manage sessions "
        "/ Resources for your balance.",
        bar,
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# Batch-size scenarios (ESTIMATE until the bench measures the real pages per hour)
# --------------------------------------------------------------------------------------------

BATCH_SIZES = (1, 2, 4, 8)
#: Scaling efficiency of batched decode: the decode step takes 1/eff times as long as at batch 1,
#: so B pages per step give a throughput of eff * B (1.0 = the step time does not grow with the
#: batch, the best case for overhead-bound decode; 0.5 = batching gains half of the ideal).
EFFICIENCIES = (1.0, 0.75, 0.5)
#: Decode-step inflation of length-sorted batches (a batch runs until its longest row stops):
#: sum over batches of max length x size / sum of lengths, gold-length proxy on the 671 train+dev
#: pages, shipdoc.batching key and 8-batch windows. Measured with $SHIPDOC_TMP_DIR/straggler.py on
#: 2026-10-02 (not a model measurement: the real output lengths are unknown until a GPU run).
STRAGGLER = {1: 1.0, 2: 1.144, 4: 1.271, 8: 1.385}
SHARD_COUNTS = (1, 2)
TEST_DOCS, TEST_PAGES = 200, 280  # reports/recon.md: the test split, file level
BENCH_PAGES, BENCH_SIZES = 12, (1, 2, 4, 8)  # shipdoc.bench: 12 dev pages at each size


def batch_s_per_page(
    prefill_s: float, decode_tok_s: float, n_out: float, batch: int, eff: float
) -> float:
    """Seconds per page at `batch` pages per generate call and scaling efficiency `eff`.

    ``prefill_s + n_out * straggler(B) / (decode_tok_s * eff * B)``: prefill is linear in pages
    (a batch of B costs B times one page; the fixed per-call overhead inside ``prefill_s`` is not
    amortised, which is conservative); decode runs ``n_out * straggler`` steps per batch at a step
    time of ``1 / (decode_tok_s * eff)``. Batch 1 is exactly `est_s_per_page`.
    """
    if batch == 1:
        return est_s_per_page(prefill_s, decode_tok_s, n_out)
    if not 0 < eff <= 1:
        raise ValueError(f"eff must be in (0, 1], got {eff}")
    if batch not in STRAGGLER:
        raise ValueError(f"no straggler factor for batch {batch}; known: {sorted(STRAGGLER)}")
    return prefill_s + n_out * STRAGGLER[batch] / (decode_tok_s * eff * batch)


def batch_scenarios() -> list[tuple[int, float]]:
    """(batch, efficiency) rows: batch 1 once, then every batch >= 2 at each efficiency."""
    return [(1, 1.0)] + [(b, e) for b in BATCH_SIZES if b > 1 for e in EFFICIENCIES]


def tab_hours(
    speed: dict[str, Any], config: str, n_out: float, batch: int, eff: float, pages: float,
    smoke_pages: int = 0, with_bench: bool = False,
) -> float:  # fmt: skip
    """T4 hours of ONE tab: smoke (batch 1) + bench + full run, a model load for each stage.

    `smoke_pages` 0 means no smoke stage. The bench (`with_bench`) is one load, one warm-up page
    and BENCH_PAGES pages at each of BENCH_SIZES, all at the scenario efficiency.
    """
    m = speed["models"][base_config(config)]
    load = float(speed.get("model_load_s", 0))

    def s(b: int) -> float:
        return batch_s_per_page(m["prefill_s"], m["decode_tok_s"], n_out, b, eff)

    secs = load + pages * s(batch)
    if smoke_pages:
        secs += load + smoke_pages * s(1)
    if with_bench:
        secs += load + s(1) + BENCH_PAGES * sum(s(b) for b in BENCH_SIZES)
    return secs / 3600


def zeroshot_batch_report(
    speed: dict[str, Any],
    config: str,
    pages_full: int,
    pages_smoke: int,
    gpu_name: str | None = "T4",
    pages_test: int = TEST_PAGES,
    docs_full: int | None = None,
) -> str:
    """Printable T4-hour / compute-unit table per batch-size scenario, 1 and 2 parallel shards.

    ESTIMATE, UNVERIFIED: the decode throughput gain of batching is an assumption (scaling
    efficiency 100 / 75 / 50 % of ideal) until the bench measures it on a GPU. Output length is
    the observed spike40 mean (``n_output_tokens_mean``). Rows with batch >= 2 include the bench;
    batch 1 is the manual ``BATCH_SIZE = 1`` (no bench). With K = 2 shards (two Colab tabs) the
    wall-clock is one tab time while the compute units add up: K tabs burn K times the hours.
    Each tab pays its own loads, smoke and bench (worst case: a later tab may reuse the bench
    result). Shard page counts are assumed balanced (greedy by page count, within one document).
    """
    m = speed["models"][base_config(config)]
    n_out = float(m["n_output_tokens_mean"])
    r1, r2 = speed.get("t4_cu_per_hour"), speed.get("t4_cu_per_hour_conservative")

    def cu(rate: float | None, hours: float) -> str:
        return "n/a" if rate is None else f"{hours * rate:.1f}"

    bar = "=" * 100
    lines = [
        bar,
        f"ESTIMATE (UNVERIFIED) per batch size: zero-shot {config}, T4. Batching gain ASSUMED "
        "until the bench measures it",
        bar,
    ]
    warn = gpu_warning(gpu_name)
    if warn:
        lines.append(warn)
    lines += [
        f"s/page(B, eff) = prefill_s {m['prefill_s']} + {n_out} tok x straggler(B) / "
        f"({m['decode_tok_s']} tok/s x eff x B); straggler(B) = {STRAGGLER} (sorted batches run to "
        "their longest row); eff = scaling efficiency of the decode step.",
        f"prefill linear in pages; {speed.get('model_load_s', 0)} s per model load "
        f"({speed.get('model_load_s_label', 'ESTIMATE')}); smoke {pages_smoke} pages at batch 1.",
    ]
    head = (
        f"{'scenario':<24}  {'K':>1}  {'s/page':>6}  {'wall h':>7}  {'CU h':>7}  "
        f"{'CU@' + str(r1):>9}  {'CU@' + str(r2):>9}"
    )

    def table(title: str, pages: int, full: bool) -> None:
        lines.extend(["", title, head])
        for b, eff in batch_scenarios():
            name = "B=1 (no bench)" if b == 1 else f"B={b} @ {int(eff * 100)}% scaling"
            s_page = batch_s_per_page(m["prefill_s"], m["decode_tok_s"], n_out, b, eff)
            for k in SHARD_COUNTS:
                h = tab_hours(
                    speed, config, n_out, b, eff, pages / k,
                    smoke_pages=pages_smoke if full else 0, with_bench=full and b > 1,
                )  # fmt: skip
                lines.append(
                    f"{name:<24}  {k:>1}  {s_page:6.1f}  {h:7.2f}  {k * h:7.2f}  "
                    f"{cu(r1, k * h):>9}  {cu(r2, k * h):>9}"
                )

    docs = f"{docs_full} docs / " if docs_full else ""
    table(
        f"{docs}{pages_full} pages: smoke + bench (B >= 2) + full run, per tab; CU = K tabs",
        pages_full,
        True,
    )
    table(
        f"test {TEST_DOCS} docs / {pages_test} pages: load + run only (the dev bench result is "
        "reused, no smoke)",
        pages_test,
        False,
    )
    lines += [
        "",
        "wall h = one tab hours (the tabs run in parallel); CU h = hours summed over the K tabs.",
        f"CU rate {r1}/h: {speed.get('t4_cu_source', 'source unset')}",
        f"CU rate {r2}/h: {speed.get('t4_cu_conservative_source', 'source unset')}",
        "Not modelled: logprob capture overhead (assumed 0), Drive I/O, session restarts, OOM "
        "fallbacks. Replace the assumed scaling by the bench pages/hour once it has run.",
        bar,
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=ROOT, help="repo root (docs/labels resolved here)")
    ap.add_argument("--labels-dir", type=Path, default=None, help="default: <root>/data/dev/labels")
    ap.add_argument("--spike-docs", default="splits/spike40.json")
    ap.add_argument("--dev100-docs", default="splits/dev100.json")
    ap.add_argument("--smoke-docs", default="splits/smoke5.json")
    ap.add_argument("--gpu-name", default="T4", help="runtime GPU name (nvidia-smi); default T4")
    ap.add_argument(
        "--zeroshot-docs",
        default="splits/zeroshot500.json",
        help="doc list for the per-batch-size zero-shot table ('' = skip it)",
    )
    args = ap.parse_args(argv)
    speed = json.loads((args.root / "configs" / "spike_speed.json").read_text(encoding="utf-8"))
    order = json.loads((args.root / "configs" / "spike_order.json").read_text(encoding="utf-8"))
    labels = args.labels_dir or args.root / "data" / "dev" / "labels"
    spike = json.loads((args.root / args.spike_docs).read_text(encoding="utf-8"))
    dev100 = json.loads((args.root / args.dev100_docs).read_text(encoding="utf-8"))
    smoke = json.loads((args.root / args.smoke_docs).read_text(encoding="utf-8"))
    print(
        report(
            order["order"], speed, count_pages(spike, labels), count_pages(dev100, labels),
            len(spike), count_pages(smoke, labels), args.gpu_name,
        )
    )  # fmt: skip
    est = order_by_estimate(order["order"], speed)
    if est != order["order"]:
        print(f"NOTE: spike_order.json order differs from the estimated order {est}")
    if args.zeroshot_docs:
        ids = json.loads((args.root / args.zeroshot_docs).read_text(encoding="utf-8"))
        data = args.root / "data"
        full = sum(count_pages([d], data / d.split("_")[0] / "labels") for d in ids)
        print()
        print(
            zeroshot_batch_report(
                speed, "qwen35_4b_img_only", full, count_pages(smoke, labels), args.gpu_name,
                docs_full=len(ids),
            )
        )  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
