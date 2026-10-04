"""Resolution sweep of the production extractor (notebook 06_res_sweep): paired analysis + decision.

Question: does a larger ``max_pixels`` (more visual tokens per page) fix part-number misreads on
the 40 documents of ``splits/spike40.json`` at an acceptable VRAM and speed cost? The production
config (``configs/spike_qwen35_4b_img_only.yaml``, the CONTROL) caps a page at 1,310,720 pixels.

Visual-token arithmetic (Qwen3.5-4B: patch 16, spatial merge 2): the image is resized by
``smart_resize`` to multiples of 32 px, then ``tokens = (H / 16) * (W / 16) / 4 = H * W / 1024``.
Every page of the corpus is 1240 x 1754 px (671 / 671 pages, measured by the ``estimate`` command
on its documents), so ``max_pixels`` above the native 1760 x 1248 grid (2,196,480 px, 2,145
tokens) changes NOTHING: the processor only ever shrinks to the cap, it never upscales. 2,500 and
3,500 tokens are therefore unreachable through ``max_pixels`` alone (they would need a raised
``min_pixels``, which the extractor does not set). ``check_reachable`` refuses a resolution list
whose token counts are not strictly increasing, so a no-op config cannot be run by mistake.

Analysis (CPU, no model): for every non-control resolution, a PAIRED document-level bootstrap
(2000 resamples, seed 42, ``shipdoc.eval.paired_bootstrap``, the unmodified scorer) of
candidate - control for OVERALL and row F1, overall and split scanned / digital; the part-number
misread count (``spn_misread`` of ``scripts/row_error_diagnosis.py``: a linked gold/predicted row
pair whose supplier_part_number is a near misread, edit distance <= 2 or similarity >= 0.8; the
definition behind "66 part-number misreads, 41 on scans" in reports/row_errors.md) with a paired
bootstrap CI of the COUNT difference; seconds per page and peak VRAM from the trace.

DECISION RULE (``decide``, verbatim): adopt the HIGHEST resolution whose paired OVERALL delta CI
lower bound is > 0 AND whose peak VRAM is <= 14.5 GiB; ties (several qualify with an equal OVERALL
point estimate) go to the LOWER resolution; if none qualifies keep the control. The CI is on 40
documents (wide); the decision uses OVERALL only; every other delta is reported, not decisive.

Batch contract: outputs are only comparable at equal batch size; a candidate run is paired with
the control only when their manifests record the same batch size.

CLI (``python -m shipdoc.ressweep``): ``estimate`` (T4 hours / CU, printed before anything runs),
``check-control`` (strict reuse check of an existing control run), ``analyse`` (paired analysis,
decision, report files). Every number is UNVERIFIED until the notebook has run on a GPU.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

PATCH_SIZE = 16  # Qwen3.5 vision config: patch_size 16 (transformers qwen3_5 configuration)
MERGE_SIZE = 2  # spatial_merge_size 2: 2 x 2 patches become one visual token
FACTOR = PATCH_SIZE * MERGE_SIZE  # smart_resize snaps both sides to a multiple of 32 px
#: Qwen2VL image-processor default; the checkpoint's own shortest_edge is kept by the extractor
#: and is far below any page here, so it never decides the grid (the value itself: UNVERIFIED).
MIN_PIXELS = 3136
PAGE_HW = (1754, 1240)  # (height, width) of every page of the corpus (measured, see module doc)
PRODUCTION_MAX_PIXELS = 1_310_720  # configs/spike_qwen35_4b_img_only.yaml: 1,260 tokens per page
CONTROL_CONFIG = "qwen35_4b_img_only"
#: (max_pixels, config name). The control is the production config itself; the others are
#: configs/spike_<name>.yaml with only name and max_pixels changed.
RESOLUTIONS: tuple[tuple[int, str], ...] = (
    (PRODUCTION_MAX_PIXELS, CONTROL_CONFIG),
    (1_843_200, "qwen35_4b_img_only_keyed_px1843200"),
    (2_196_480, "qwen35_4b_img_only_keyed_px2196480"),
)
VRAM_LIMIT_GIB = 14.5  # T4 15 GiB minus headroom: the same bound as shipdoc.bench
N_BOOT = 2000
SEED = 42
STATES = ("complete", "oom", "failed", "pending", "incomplete")
CI_WARNING = (
    "The CI is on 40 documents (wide). The decision uses OVERALL as its only criterion; every "
    "other delta is reported, not decisive."
)


# --------------------------------------------------------------------------------------------
# Visual-token arithmetic
# --------------------------------------------------------------------------------------------


def smart_resize(
    height: int, width: int, max_pixels: int, factor: int = FACTOR, min_pixels: int = MIN_PIXELS
) -> tuple[int, int]:
    """(H, W) after the Qwen2VL ``smart_resize`` (mirror of transformers, tested against it)."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError("absolute aspect ratio must be smaller than 200")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def visual_tokens(max_pixels: int, hw: tuple[int, int] = PAGE_HW) -> int:
    """Visual tokens of one page under ``max_pixels``: (H/16) * (W/16) / 2^2 after resizing."""
    h, w = smart_resize(hw[0], hw[1], max_pixels)
    return (h // PATCH_SIZE) * (w // PATCH_SIZE) // (MERGE_SIZE**2)


def check_reachable(
    resolutions: Sequence[int], hw: tuple[int, int] = PAGE_HW
) -> list[tuple[int, tuple[int, int], int]]:
    """(max_pixels, grid, tokens) per resolution; raises when the tokens are not strictly rising.

    Two resolutions with the same token count are the same experiment (the cap does not bind),
    so a sweep over them would only measure run-to-run noise.
    """
    rows = []
    for px in resolutions:
        grid = smart_resize(hw[0], hw[1], px)
        rows.append((px, grid, visual_tokens(px, hw)))
    toks = [r[2] for r in rows]
    if any(b <= a for a, b in zip(toks, toks[1:], strict=False)):
        raise ValueError(
            "the resolutions do not give strictly increasing visual tokens for a "
            f"{hw[1]}x{hw[0]} page ({[(r[0], r[2]) for r in rows]}): max_pixels above the native "
            "grid changes nothing (the processor never upscales)"
        )
    return rows


# --------------------------------------------------------------------------------------------
# Decision rule
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """One non-control resolution as the decision sees it."""

    max_pixels: int
    state: str  # complete | oom | failed | pending | incomplete
    delta_overall: float | None = None  # candidate - control, paired point estimate
    delta_overall_lo: float | None = None  # lower bound of its 95% paired bootstrap CI
    peak_vram_bytes: int | None = None


@dataclass(frozen=True)
class Decision:
    """Result of ``decide``: the adopted resolution (None = keep the control) and why."""

    adopted: int | None
    reasons: dict[int, str] = field(default_factory=dict)

    @property
    def label(self) -> str:
        """The adopted ``max_pixels`` as text, or ``control``."""
        return "control" if self.adopted is None else str(self.adopted)


def decide(cands: Sequence[Candidate], vram_limit_gib: float = VRAM_LIMIT_GIB) -> Decision:
    """Adopt the HIGHEST resolution with a paired OVERALL delta CI lower bound > 0 and peak VRAM
    <= `vram_limit_gib`; ties go to the LOWER resolution; none qualifies -> keep the control.

    A candidate qualifies only if it is ``complete`` and BOTH numbers were measured: a missing
    lower bound or VRAM is a failed check, never a pass (an OOM, failed or unmeasured run is
    excluded). "Ties": qualifying resolutions whose OVERALL point estimates are equal to the
    highest qualifying one's; the lowest of those is adopted (no gain from the extra pixels).
    The CI is on 40 documents (wide) and OVERALL is the only criterion.
    """
    limit = vram_limit_gib * 2**30
    reasons: dict[int, str] = {}
    qualifying: list[Candidate] = []
    for c in sorted(cands, key=lambda x: x.max_pixels):
        if c.state != "complete":
            reasons[c.max_pixels] = f"excluded: run {c.state}"
        elif c.delta_overall_lo is None or c.delta_overall is None:
            reasons[c.max_pixels] = "excluded: OVERALL delta CI not measured"
        elif c.peak_vram_bytes is None:
            reasons[c.max_pixels] = "excluded: peak VRAM not measured"
        elif c.peak_vram_bytes > limit:
            reasons[c.max_pixels] = (
                f"rejected: peak VRAM {c.peak_vram_bytes / 2**30:.2f} GiB > {vram_limit_gib} GiB"
            )
        elif not c.delta_overall_lo > 0:
            reasons[c.max_pixels] = (
                f"rejected: OVERALL delta CI lower bound {c.delta_overall_lo:+.4f} is not > 0"
            )
        else:
            qualifying.append(c)
            reasons[c.max_pixels] = "qualifies"
    if not qualifying:
        return Decision(None, reasons)
    top = qualifying[-1]  # qualifying is ascending by max_pixels
    tied = [c for c in qualifying if abs(c.delta_overall - top.delta_overall) <= 1e-12]  # type: ignore[operator]
    best = tied[0]
    for c in qualifying:
        if c is not best:
            tail = (
                "tie with a lower resolution (equal OVERALL): the lower is preferred"
                if c in tied
                else "qualifies but a higher resolution also qualifies"
            )
            reasons[c.max_pixels] = tail
    reasons[best.max_pixels] = "ADOPTED" + (" (tie: lowest)" if len(tied) > 1 else "")
    return Decision(best.max_pixels, reasons)


# --------------------------------------------------------------------------------------------
# Strict reuse of an existing control run
# --------------------------------------------------------------------------------------------


def check_control(
    manifest: Mapping[str, Any] | None,
    progress: Mapping[str, Any] | None,
    trace_doc_ids: Sequence[str],
    expected: Mapping[str, Any],
) -> list[str]:
    """Reasons a control run folder may NOT be reused (empty list = reusable).

    `expected`: ``config_hash``, ``code_sha`` (the notebook's full pin), ``model_revision``,
    ``doc_ids`` (the ordered spike40 list), ``batch_size``. Strict: every key must match exactly,
    the run must be complete, keyed + logprobs, shard 0/1 and carry exactly the expected
    documents. A missing manifest (a pre-manifest run) is a refusal, never an assumption.
    """
    if manifest is None:
        return ["no manifest.json: the run's config, code and batch size are unknown"]
    from shipdoc.runmeta import docs_sha

    why: list[str] = []

    def need(name: str, got: Any, want: Any) -> None:
        if got != want:
            why.append(f"{name}: run has {got!r}, expected {want!r}")

    cfg = manifest.get("config") or {}
    need("config hash", cfg.get("hash"), expected["config_hash"])
    need("code SHA", manifest.get("code_sha"), expected["code_sha"])
    need(
        "model revision", (manifest.get("model") or {}).get("revision"), expected["model_revision"]
    )
    need("batch size", manifest.get("batch_size"), expected["batch_size"])
    need("doc list sha", manifest.get("docs_sha"), docs_sha(list(expected["doc_ids"])))
    need("shard", manifest.get("shard"), "0/1")
    need("logprobs", manifest.get("logprobs"), True)
    need("output format", manifest.get("output_format"), "json")
    need("progress status", (progress or {}).get("status"), "complete")
    want_ids = sorted(expected["doc_ids"])
    if sorted(trace_doc_ids) != want_ids:
        why.append(
            f"trace holds {len(set(trace_doc_ids))} documents, expected exactly {len(want_ids)}"
        )
    return why


# --------------------------------------------------------------------------------------------
# Reading runs
# --------------------------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_run(run_dir: Path) -> dict[str, Any]:
    """predictions, trace lines, manifest, progress of a run folder (missing parts -> None)."""
    run_dir = Path(run_dir)

    def opt(name: str) -> Any:
        p = run_dir / name
        return _read_json(p) if p.is_file() else None

    trace_path = run_dir / "trace.jsonl"
    traces = (
        [json.loads(ln) for ln in trace_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if trace_path.is_file()
        else []
    )
    return {
        "predictions": opt("predictions.json") or {},
        "traces": traces,
        "manifest": opt("manifest.json"),
        "progress": opt("progress.json"),
    }


def run_stats(traces: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """s/page (mean of call latencies), peak VRAM (max over pages), visual tokens, page count."""
    pages = [p for t in traces for p in t["pages"]]
    lat = [p["meta"]["latency_s"] for p in pages if p["meta"].get("latency_s") is not None]
    vram = [p["meta"]["peak_vram_bytes"] for p in pages if p["meta"].get("peak_vram_bytes")]
    vis = [p["meta"]["n_visual_tokens"] for p in pages if p["meta"].get("n_visual_tokens")]
    oom_pages = sum(
        any(isinstance(v, str) and "out of memory" in v.lower() for v in p["meta"].values())
        for p in pages
    )
    return {
        "n_pages": len(pages),
        "oom_pages": oom_pages,  # a swallowed CUDA OOM surfaced in a page's meta
        "s_per_page_mean": sum(lat) / len(lat) if lat else None,
        "peak_vram_bytes": max(vram) if vram else None,
        "n_visual_tokens_mean": sum(vis) / len(vis) if vis else None,
    }


def load_row_diagnosis() -> ModuleType:
    """``scripts/row_error_diagnosis.py`` loaded by path (read-only reuse of its definitions)."""
    name = "row_error_diagnosis"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


def doc_spn_misreads(
    pred: Any,
    gold: Mapping[str, Any],
    trace: Mapping[str, Any],
    index: Any | None,
    sc: ModuleType,
    rd: ModuleType,
) -> int:
    """Part-number misreads (``spn_misread`` linked pairs) of one document.

    Same pipeline as ``row_error_diagnosis.diagnose_doc``: the rows the official scorer leaves
    unpaired are linked (identifier evidence, then equal quantity on the same page, then page
    position) and each linked pair is classified by ``classify_pair``. `index` (the document's
    OCR index) places the gold rows on pages; without it every gold page is unknown, the
    quantity / position links cannot form and the count is a LOWER BOUND (``basis`` in the report).
    """
    from shipdoc import eval as ev

    if (
        gold.get("doc_type") != "invoice"
        or not isinstance(pred, dict)
        or pred.get("doc_type") != "invoice"
    ):
        return 0
    gr = list(gold.get("line_items") or [])
    pr = [x for x in (pred.get("line_items") or []) if isinstance(x, dict)]
    _, _, free_p, free_g = ev._pair_rows(sc, pr, gr)
    gpage = rd.gold_row_pages(gold, index)[0] if index is not None else [None] * len(gr)
    ppage = rd.predicted_row_pages(trace, len(pr))
    links, _, _ = rd.link_unpaired(gr, pr, free_g, free_p, gpage, ppage, sc.same)
    return sum(
        rd.classify_pair(gr[k.gi], pr[k.pi], k.kind, sc.same) == "spn_misread" for k in links
    )


def paired_count_delta(
    ctrl: Sequence[int], cand: Sequence[int], n: int = N_BOOT, seed: int = SEED
) -> dict[str, float]:
    """Paired document-level bootstrap of the summed count difference (candidate - control).

    Uses the same resample indices as ``eval.paired_bootstrap`` (same rng, same document order),
    so the count CI and the score CIs resample the same documents. Negative = fewer misreads.
    """
    import numpy as np

    from shipdoc import eval as ev

    if len(ctrl) != len(cand) or not ctrl:
        raise ValueError("count vectors must be non-empty and the same length")
    a, b = np.asarray(ctrl, dtype=float), np.asarray(cand, dtype=float)
    idx = ev._rng(seed).integers(0, len(a), size=(n, len(a)))
    d = b[idx].sum(axis=1) - a[idx].sum(axis=1)
    lo, hi = np.quantile(d, [0.025, 0.975])
    return {
        "control": float(a.sum()),
        "candidate": float(b.sum()),
        "delta": float(b.sum() - a.sum()),
        "lo": float(lo),
        "hi": float(hi),
    }


# --------------------------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------------------------


def _slice_scores(
    pa: dict[str, Any], pb: dict[str, Any], gold: dict[str, Any], n: int
) -> dict[str, dict[str, float]]:
    """OVERALL and row F1 paired bootstrap of B - A on `gold` (the unmodified scorer)."""
    from shipdoc import eval as ev

    if not gold:
        return {}
    out = ev.paired_bootstrap(pa, pb, gold, n=n, seed=SEED)
    return {k: out[k] for k in ("OVERALL", "row_f1")}


def analyse(
    status: Mapping[str, Any],
    doc_ids: Sequence[str],
    gold_dir: Path,
    meta_path: Path,
    ocr_cache: Path | None,
    n_boot: int = N_BOOT,
) -> dict[str, Any]:
    """Paired analysis of every non-control resolution vs the control, plus the decision.

    `status` maps ``str(max_pixels)`` to ``{"state", "run_dir", "source"}``; the control's key is
    ``str(PRODUCTION_MAX_PIXELS)``. A run counts as ``complete`` only if its progress file says so
    and its trace holds exactly `doc_ids`; anything else is listed and excluded, never skipped.
    """
    from shipdoc import eval as ev
    from shipdoc import locate

    ctl_key = str(PRODUCTION_MAX_PIXELS)
    ctl = status.get(ctl_key)
    if not ctl or ctl.get("state") != "complete":
        raise ValueError(f"the control run is not complete ({ctl}): nothing to compare against")
    want = sorted(doc_ids)
    gold_all = ev.load_gold(gold_dir)
    gold = {d: gold_all[d] for d in doc_ids}
    meta = {m["doc_id"]: m for m in _read_json(meta_path)}
    scanned = {d: bool(meta[d]["scanned"]) for d in doc_ids}
    sc = ev.load_scorer()
    rd = load_row_diagnosis()

    ocr_pages = ev.load_ocr_pages(list(doc_ids), ocr_cache) if ocr_cache else {}
    have_ocr = all(d in ocr_pages for d in doc_ids)
    indexes = {d: locate.build_index(ocr_pages[d]) for d in doc_ids} if have_ocr else {}
    basis = (
        "exact: gold rows placed on pages with the OCR cache, as in reports/row_errors.md"
        if have_ocr
        else "LOWER BOUND: no OCR cache, gold row pages unknown, only identifier links form"
    )

    def load(entry: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
        run = read_run(Path(entry["run_dir"]))
        ids = sorted(t["doc_id"] for t in run["traces"])
        if entry.get("state") != "complete":
            return run, entry.get("state", "pending")
        if (run["progress"] or {}).get("status") != "complete" or ids != want:
            return run, "incomplete"
        if run_stats(run["traces"])["oom_pages"]:
            return run, "oom"
        return run, "complete"

    ctl_run, ctl_state = load(ctl)
    if ctl_state != "complete":
        raise ValueError(f"the control run is {ctl_state} (progress / documents): cannot compare")
    ctl_bs = (ctl_run["manifest"] or {}).get("batch_size")
    traces_of = lambda run: {t["doc_id"]: t for t in run["traces"]}  # noqa: E731

    def misreads(run: Mapping[str, Any]) -> list[int]:
        tr = traces_of(run)
        return [
            doc_spn_misreads(run["predictions"].get(d), gold[d], tr[d], indexes.get(d), sc, rd)
            for d in doc_ids
        ]

    ctl_mis = misreads(ctl_run)
    ctl_stats = run_stats(ctl_run["traces"])
    result: dict[str, Any] = {
        "n_docs": len(doc_ids),
        "n_boot": n_boot,
        "seed": SEED,
        "control": {
            "max_pixels": PRODUCTION_MAX_PIXELS,
            "source": ctl.get("source"),
            "batch_size": ctl_bs,
            "misreads": sum(ctl_mis),
            "misreads_scanned": sum(m for d, m in zip(doc_ids, ctl_mis, strict=True) if scanned[d]),
            "ocr_basis": basis,
            **ctl_stats,
            "OVERALL": ev.score(ctl_run["predictions"], gold)["all"]["OVERALL"],
        },
        "candidates": [],
        "ci_warning": CI_WARNING,
    }
    cands: list[Candidate] = []
    for px in sorted(int(k) for k in status if int(k) != PRODUCTION_MAX_PIXELS):
        entry = status[str(px)]
        row: dict[str, Any] = {"max_pixels": px, "source": entry.get("source")}
        run, state = load(entry) if entry.get("run_dir") else ({}, entry.get("state", "pending"))
        bs = (run.get("manifest") or {}).get("batch_size") if run else None
        if state == "complete" and bs != ctl_bs:
            state = "incomplete"
            row["note"] = f"batch size {bs} != control's {ctl_bs}: not comparable"
        row["state"] = state
        if state != "complete":
            result["candidates"].append(row)
            cands.append(Candidate(px, state, peak_vram_bytes=None))
            continue
        pa, pb = ctl_run["predictions"], run["predictions"]
        subsets = {
            "all": gold,
            "scanned": {d: g for d, g in gold.items() if scanned[d]},
            "digital": {d: g for d, g in gold.items() if not scanned[d]},
        }
        row["scores"] = {k: _slice_scores(pa, pb, g, n_boot) for k, g in subsets.items()}
        cand_mis = misreads(run)
        row["misreads"] = paired_count_delta(ctl_mis, cand_mis, n_boot)
        sc_idx = [i for i, d in enumerate(doc_ids) if scanned[d]]
        row["misreads_scanned"] = (
            paired_count_delta([ctl_mis[i] for i in sc_idx], [cand_mis[i] for i in sc_idx], n_boot)
            if sc_idx
            else None
        )
        row.update(run_stats(run["traces"]))
        d_all = row["scores"]["all"]["OVERALL"]
        cands.append(Candidate(px, "complete", d_all["delta"], d_all["lo"], row["peak_vram_bytes"]))
        result["candidates"].append(row)
    decision = decide(cands)
    for row in result["candidates"]:
        row["verdict"] = decision.reasons.get(row["max_pixels"], "")
    result["adopted"] = decision.label
    result["tokens"] = {
        str(px): visual_tokens(px) for px in [PRODUCTION_MAX_PIXELS, *(c.max_pixels for c in cands)]
    }
    return result


def format_result(res: Mapping[str, Any]) -> str:
    """Printable report (aggregates only, no document values) ending with the DONE line."""
    c = res["control"]
    gib = lambda b: "n/a" if b is None else f"{b / 2**30:.2f}"  # noqa: E731
    sp = lambda v: "n/a" if v is None else f"{v:.1f}"  # noqa: E731
    pct = lambda x: f"{100 * x:+.2f}"  # noqa: E731
    bar = "=" * 78
    out = [
        bar,
        f"RESOLUTION SWEEP on {res['n_docs']} documents: paired vs the control (max_pixels "
        f"{c['max_pixels']}, {res['tokens'][str(c['max_pixels'])]} visual tokens)",
        res["ci_warning"],
        f"batch size (control): {c['batch_size']}; outputs are only comparable at equal batch size",
        bar,
        f"control: OVERALL {100 * c['OVERALL']:.2f}  s/page {sp(c['s_per_page_mean'])}  "
        f"peak VRAM {gib(c['peak_vram_bytes'])} GiB  misreads {c['misreads']} "
        f"({c['misreads_scanned']} on scans)  [{c['source']}]",
        f"misread basis: {c['ocr_basis']}",
    ]
    for r in res["candidates"]:
        px = r["max_pixels"]
        out.append(f"--- max_pixels {px} ({res['tokens'].get(str(px))} tokens): {r['state']}")
        if r["state"] != "complete":
            out.append(f"    EXCLUDED ({r['state']}){': ' + r['note'] if r.get('note') else ''}")
            out.append(f"    verdict: {r.get('verdict', '')}")
            continue
        for sub in ("all", "scanned", "digital"):
            for m, label in (("OVERALL", "OVERALL"), ("row_f1", "row F1")):
                d = r["scores"][sub].get(m)
                if d:
                    out.append(
                        f"    {sub:<8}{label:<8} paired delta {pct(d['delta'])} "
                        f"[{pct(d['lo'])}, {pct(d['hi'])}] pts"
                    )
        m = r["misreads"]
        out.append(
            f"    misreads {m['control']:.0f} -> {m['candidate']:.0f}: delta {m['delta']:+.0f} "
            f"[{m['lo']:+.0f}, {m['hi']:+.0f}] (paired count bootstrap)"
        )
        ms = r["misreads_scanned"]
        if ms:
            out.append(
                f"    misreads on scans {ms['control']:.0f} -> {ms['candidate']:.0f}: "
                f"delta {ms['delta']:+.0f} [{ms['lo']:+.0f}, {ms['hi']:+.0f}]"
            )
        out.append(
            f"    s/page {sp(r['s_per_page_mean'])} (ctl {sp(c['s_per_page_mean'])}), peak VRAM "
            f"{gib(r['peak_vram_bytes'])} GiB (limit {VRAM_LIMIT_GIB}), visual tokens "
            f"{sp(r['n_visual_tokens_mean'])}"
        )
        out.append(f"    verdict: {r['verdict']}")
    out += [
        bar,
        f"DECISION RULE: adopt the highest resolution with OVERALL paired-delta CI lower bound > 0 "
        f"and peak VRAM <= {VRAM_LIMIT_GIB} GiB; ties go to the lower; none -> control.",
        res["ci_warning"],
        f"adopted: {res['adopted']}",
        bar,
        f"DONE ressweep px={','.join(str(p) for p in res['tokens'])} adopted={res['adopted']}",
    ]
    return "\n".join(out)


# --------------------------------------------------------------------------------------------
# T4-hour / compute-unit estimate
# --------------------------------------------------------------------------------------------


def scaled_prefill_s(prefill_s: float, n_in_ref: float, vis_ref: float, vis: float) -> float:
    """ASSUMED scaling: prefill seconds grow linearly with input tokens (text + visual).

    ``prefill_s`` of ``configs/spike_speed.json`` was fitted at `n_in_ref` input tokens of which
    `vis_ref` were visual; at `vis` visual tokens the input has ``n_in_ref - vis_ref + vis``. The
    fitted intercept also holds fixed per-call overhead, so scaling all of it overstates the cost
    (conservative). Vision-tower and attention costs are not modelled (UNVERIFIED).
    """
    return prefill_s * (n_in_ref - vis_ref + vis) / n_in_ref


def estimate_rows(
    speed: Mapping[str, Any],
    pages: int,
    smoke_pages: int,
    batch_size: int,
    reuse_control: bool,
    resolutions: Sequence[int],
) -> dict[str, Any]:
    """T4 hours per stage and in total (mean and p99-output upper bound), nothing measured."""
    m = speed["models"]["qwen35_4b_img_only"]
    n = speed["n_out_keyed"]["qwen35_4b"]
    load_h = float(speed["model_load_s"]) / 3600
    vis_ref = float(visual_tokens(PRODUCTION_MAX_PIXELS))
    rows = []
    for px in resolutions:
        pf = scaled_prefill_s(m["prefill_s"], m["n_input_tokens_mean"], vis_ref, visual_tokens(px))
        s_mean = pf + n["mean"] / m["decode_tok_s"]
        s_ub = pf + n["upper_bound"] / m["decode_tok_s"]
        run = not (reuse_control and px == PRODUCTION_MAX_PIXELS)
        rows.append(
            {
                "max_pixels": px,
                "tokens": visual_tokens(px),
                "prefill_s": pf,
                "s_page": s_mean,
                "s_page_ub": s_ub,
                "runs": run,
                "hours": (s_mean * pages / 3600 + load_h) if run else 0.0,
                "hours_ub": (s_ub * pages / 3600 + load_h) if run else 0.0,
            }
        )
    big = rows[-1]  # the smoke runs the largest resolution
    smoke_h = big["s_page"] * smoke_pages / 3600 + load_h
    smoke_ub = big["s_page_ub"] * smoke_pages / 3600 + load_h
    return {
        "rows": rows,
        "smoke_hours": smoke_h,
        "smoke_hours_ub": smoke_ub,
        "hours": smoke_h + sum(r["hours"] for r in rows),
        "hours_ub": smoke_ub + sum(r["hours_ub"] for r in rows),
        "batch_size": batch_size,
        "pages": pages,
        "smoke_pages": smoke_pages,
    }


def format_estimate(
    est: Mapping[str, Any],
    speed: Mapping[str, Any],
    gpu_warning: str | None,
    reused: Mapping[str, Any] | None = None,
) -> str:
    """ESTIMATE text: per resolution tokens, s/page, T4 hours; totals with both CU rates.

    `est` is the plan where every resolution runs; `reused` (optional) the same plan with the
    control reused (only if CONTROL_RUN_DIR passes the strict check; otherwise `est` applies).
    """
    lo, hi = float(speed["t4_cu_per_hour"]), float(speed["t4_cu_per_hour_conservative"])
    out = [
        "ESTIMATE (UNVERIFIED: nothing here is measured on this sweep)",
        f"{est['pages']} pages (spike40), smoke {est['smoke_pages']} pages at the largest "
        f"resolution, batch size {est['batch_size']}.",
        "Speed model: s/page = prefill_s + n_out / decode_tok_s from configs/spike_speed.json "
        "(spike40 traces, T4), keyed output length (mean; p99 as upper bound), model load "
        f"{speed['model_load_s']} s per run (ESTIMATE).",
        "ASSUMED: prefill grows linearly with input tokens (text + visual); vision tower, "
        "attention and KV-cache growth are not modelled; batch size > 1 is not credited with "
        "any speed-up.",
        "",
        f"{'max_pixels':>11} {'tokens':>7} {'s/page':>7} {'s/page p99':>10} {'T4 h':>6} "
        f"{'T4 h p99':>9}  run",
    ]
    for r in est["rows"]:
        out.append(
            f"{r['max_pixels']:>11} {r['tokens']:>7} {r['s_page']:>7.1f} {r['s_page_ub']:>10.1f} "
            f"{r['hours']:>6.2f} {r['hours_ub']:>9.2f}  "
            + ("this notebook" if r["runs"] else "reused control")
        )
    out.append(
        f"{'smoke':>11} {'':>7} {'':>7} {'':>10} {est['smoke_hours']:>6.2f} "
        f"{est['smoke_hours_ub']:>9.2f}  largest resolution, incl. one model load"
    )
    out.append(
        f"TOTAL: {est['hours']:.2f} T4 hours (p99 output {est['hours_ub']:.2f}) = "
        f"{est['hours'] * lo:.1f}-{est['hours'] * hi:.1f} CU "
        f"(p99 {est['hours_ub'] * lo:.1f}-{est['hours_ub'] * hi:.1f}) at {lo} / {hi} CU per T4 hour"
    )
    if reused is not None:
        out.append(
            f"IF the control is reused (CONTROL_RUN_DIR passes the strict check): "
            f"{reused['hours']:.2f} T4 hours = "
            f"{reused['hours'] * lo:.1f}-{reused['hours'] * hi:.1f} "
            f"CU (p99 {reused['hours_ub'] * lo:.1f}-{reused['hours_ub'] * hi:.1f})"
        )
    out.append("CU rates are third-party observations, not published by Google; check Colab.")
    if gpu_warning:
        out.append(gpu_warning)
    return "\n".join(out)


def page_sizes(doc_ids: Sequence[str], data_dir: Path) -> dict[tuple[int, int], int]:
    """Count of pages per image size (width, height) over the documents' images."""
    from collections import Counter

    from PIL import Image

    sizes: Counter[tuple[int, int]] = Counter()
    for d in doc_ids:
        split = d.split("_")[0]
        for p in sorted((data_dir / split / "images").glob(f"{d}_p*")) or sorted(
            (data_dir / split / "images").glob(f"{d}*")
        ):
            with Image.open(p) as im:
                sizes[im.size] += 1
    return dict(sizes)


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def _load_ge() -> ModuleType:
    spec = importlib.util.spec_from_file_location("gpu_estimate", ROOT / "scripts/gpu_estimate.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _gpu_name() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=False,
        ).stdout.strip().splitlines()  # fmt: skip
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out[0].strip() if out else None


def _doc_ids(path: str) -> list[str]:
    ids = _read_json(ROOT / path if not Path(path).is_absolute() else Path(path))
    if not isinstance(ids, list) or len(ids) != len(set(ids)):
        raise ValueError(f"{path}: expected a JSON list of unique doc ids")
    return ids


def main_estimate(a: argparse.Namespace) -> int:
    """Print the page-size check, the token table and the T4-hour / CU estimate."""
    ge = _load_ge()
    speed = _read_json(ROOT / "configs" / "spike_speed.json")
    docs, smoke = _doc_ids(a.docs), _doc_ids(a.smoke_docs)
    if a.data_dir is None:
        from shipdoc import paths

        a.data_dir = paths.data_dir()
    data = a.data_dir
    sizes = page_sizes(docs, data)
    print(f"page sizes of the {len(docs)} documents (width, height): {sizes}")
    if set(sizes) != {(PAGE_HW[1], PAGE_HW[0])}:
        print(
            f"WARNING: pages are not all {PAGE_HW[1]}x{PAGE_HW[0]}: the token counts below do "
            "not apply to them."
        )
    pxs = (
        [int(x) for x in a.resolutions.split(",")] if a.resolutions else [r[0] for r in RESOLUTIONS]
    )
    print("max_pixels -> resized grid (H x W) -> visual tokens (patch 16, merge 2):")
    for px, grid, tok in check_reachable(pxs):
        print(f"  {px:>9} -> {grid[0]}x{grid[1]} -> {tok} tokens")
    pages = ge.count_pages(docs, data / "dev" / "labels")
    smoke_pages = ge.count_pages(smoke, data / "dev" / "labels")
    gpu = a.gpu_name if a.gpu_name is not None else _gpu_name()
    print("Runtime GPU (nvidia-smi):", gpu)
    est = estimate_rows(speed, pages, smoke_pages, a.batch_size, False, pxs)
    reused = estimate_rows(speed, pages, smoke_pages, a.batch_size, True, pxs)
    print(format_estimate(est, speed, ge.gpu_warning(gpu), reused))
    return 0


def main_check_control(a: argparse.Namespace) -> int:
    """Exit 0 when the control run folder passes the strict reuse check, else 1 (reasons shown)."""
    from shipdoc.spike import load_config

    cfg = load_config(ROOT / a.config if not Path(a.config).is_absolute() else a.config)
    run = read_run(a.run_dir)
    expected = {
        "config_hash": cfg.config_hash,
        "code_sha": a.pin,
        "model_revision": cfg.backend.revision,
        "doc_ids": _doc_ids(a.docs),
        "batch_size": a.batch_size,
    }
    why = check_control(
        run["manifest"], run["progress"], [t["doc_id"] for t in run["traces"]], expected
    )
    if why:
        print(f"CONTROL NOT REUSED ({a.run_dir}):")
        for w in why:
            print("  -", w)
        return 1
    print(
        f"CONTROL REUSED: {a.run_dir} matches config hash {cfg.config_hash}, code {a.pin[:7]}, "
        f"model revision, doc list and batch size {a.batch_size}."
    )
    return 0


def main_analyse(a: argparse.Namespace) -> int:
    """Run the analysis from a status file, print the report, write result json + markdown."""
    from shipdoc import paths

    status = _read_json(a.status)
    ocr = None if a.no_ocr else (a.ocr_cache or paths.ocr_cache_dir())
    res = analyse(
        status,
        _doc_ids(a.docs),
        a.gold_dir or paths.data_dir() / "dev" / "labels",
        ROOT / "meta" / "dev.json",
        ocr,
    )
    text = format_result(res)
    print(text)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "ressweep_result.json").write_text(json.dumps(res, indent=1) + "\n", "utf-8")
    (a.out_dir / "ressweep_result.md").write_text("```\n" + text + "\n```\n", "utf-8")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m shipdoc.ressweep estimate | check-control | analyse``."""
    ap = argparse.ArgumentParser(prog="shipdoc.ressweep", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("estimate")
    e.add_argument("--docs", default="splits/spike40.json")
    e.add_argument("--smoke-docs", default="splits/smoke5.json")
    e.add_argument("--batch-size", type=int, default=1)
    e.add_argument("--data-dir", type=Path, default=None, help="default: SHIPDOC_DATA_DIR")
    e.add_argument("--resolutions", default=None, help="comma-separated max_pixels, control first")
    e.add_argument("--gpu-name", default=None, help="default: read from nvidia-smi")
    c = sub.add_parser("check-control")
    c.add_argument("--run-dir", type=Path, required=True)
    c.add_argument("--config", default=f"configs/spike_{CONTROL_CONFIG}.yaml")
    c.add_argument("--pin", required=True, help="full 40-char SHA the notebook is pinned to")
    c.add_argument("--docs", default="splits/spike40.json")
    c.add_argument("--batch-size", type=int, required=True)
    n = sub.add_parser("analyse")
    n.add_argument("--status", type=Path, required=True)
    n.add_argument("--docs", default="splits/spike40.json")
    n.add_argument("--out-dir", type=Path, required=True)
    n.add_argument("--gold-dir", type=Path, default=None)
    n.add_argument("--ocr-cache", type=Path, default=None)
    n.add_argument(
        "--no-ocr", action="store_true", help="skip the OCR cache (misreads: lower bound)"
    )
    a = ap.parse_args(argv)
    return {
        "estimate": main_estimate,
        "check-control": main_check_control,
        "analyse": main_analyse,
    }[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
