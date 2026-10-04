"""Batch-size bench: pick the pages-per-generate-call for a run by measuring it on 12 dev pages.

Run after the smoke gate (notebook 02, and any later notebook that calls `run_bench` / the
``bench`` CLI subcommand). On 12 dev pages chosen from the meta tags (`pick_bench_docs`,
written to ``splits/bench12.json``; no test pages) it runs the real pipeline at batch sizes
1, 2, 4, 8 and records per size: pages per hour, peak VRAM, the byte-identical page-JSON rate vs
batch 1, the field-level agreement rate vs batch 1 and whether the official scorer gives the
identical per-document result on those pages.

Selection rule (`choose_batch_size`, verbatim in ``bench_result.json``):

    The largest batch size with 100% byte-identical outputs and peak VRAM <= 14.5 GiB. If none
    above 1 is byte-identical, take the largest with >= 99.5% field agreement AND identical scorer
    result on those pages, and log the deviation explicitly in the manifest and printed banner. If
    both fail, use batch 1.

Readings (the rule leaves them open): the VRAM limit is a hard safety bound in every step, also
the second; "none above 1 is byte-identical" means none above 1 passes step one (identical AND
within the VRAM limit); a size whose peak VRAM was not measured (CPU / mock) or that raised counts
as failing. Selection never looks at pages per hour: throughput is reported, the rule decides on
identity and memory only, so two tabs benching the same code on the same GPU model pick the same
size.

Why byte identity is the criterion: batched fp16 matmul / attention kernels reduce in a different
order than batch 1, and the model is a hybrid (24 Gated-DeltaNet linear-attention layers + 8
full-attention layers, mrope positions) whose left-padding behaviour can only be settled on the
GPU, so batched greedy decoding is not guaranteed to reproduce the batch-1 text.

Scope of that identity: batch N is compared with batch 1 BY DESIGN, so every page is decoded in a
different composition (other companions, other left-padding) than in the reference. The question
is "does batching change the text", and the answer selects the batch size. It is not a determinism
check and its rate must never be quoted as one: determinism (same input, same composition, same
bytes) is `shipdoc.predict.run_determinism`, which replays the first pass's exact generate calls.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from shipdoc import eval as ev
from shipdoc import paths
from shipdoc.extract import VlmBackend
from shipdoc.spike import (
    SpikeConfig,
    atomic_write,
    doc_page_images,
    git_commit,
    load_config,
    load_doc_ids,
    load_gold_and_meta,
    make_backend,
    run_spike,
)

BENCH_SCHEMA = 1
BATCH_SIZES = (1, 2, 4, 8)
BENCH_PAGES = 12
BENCH_SPLIT = "dev"
VRAM_LIMIT_GIB = 14.5
VRAM_LIMIT_BYTES = int(VRAM_LIMIT_GIB * 2**30)
MIN_FIELD_AGREEMENT = 0.995
SELECTION_RULE = (
    "The largest batch size with 100% byte-identical outputs and peak VRAM <= 14.5 GiB. If none "
    "above 1 is byte-identical, take the largest with >= 99.5% field agreement AND identical "
    "scorer result on those pages, and log the deviation explicitly in the manifest and printed "
    "banner. If both fail, use batch 1."
)
RESULT_NAME = "bench_result.json"
MULTI_RESERVE = 4  # pages kept for single-page docs so the set is always mixed


# --------------------------------------------------------------------------------------------
# Page selection
# --------------------------------------------------------------------------------------------


def pick_bench_docs(
    meta: Sequence[Mapping[str, Any]], page_counts: Mapping[str, int], n_pages: int = BENCH_PAGES
) -> dict[str, str]:
    """Deterministic bench set of whole dev documents totalling exactly `n_pages` pages.

    From the meta tags only: up to 2 multipage scanned and 2 multipage digital invoices (the
    long tables; documents with repeated parts first, then dev order) within `n_pages -
    MULTI_RESERVE` pages, then single-page documents round robin over scanned invoice, digital
    invoice, waybill (illegible ones skipped) until the total is exact. Returns ``{doc_id:
    group}`` in dev order. Raises ValueError if the split cannot fill the budget.
    """
    order = {m["doc_id"]: i for i, m in enumerate(meta)}
    chosen: dict[str, str] = {}
    left = n_pages

    def pool(pred: Any) -> list[Mapping[str, Any]]:
        return sorted(
            (m for m in meta if pred(m)),
            key=lambda m: (not m["repeated_parts"], order[m["doc_id"]]),
        )

    multi_left = n_pages - MULTI_RESERVE
    multis = {
        name: pool(lambda m, s=scanned: m["multipage"] and m["scanned"] == s and not m["waybill"])
        for name, scanned in (("multi_scanned", True), ("multi_digital", False))
    }
    for _round in range(2):  # alternate scanned / digital so neither starves the other
        for name, docs in multis.items():
            for m in docs:
                n = page_counts[m["doc_id"]]
                if m["doc_id"] not in chosen and n <= multi_left:
                    chosen[m["doc_id"]] = name
                    multi_left -= n
                    left -= n
                    break
    singles = {
        "single_scanned_invoice": pool(
            lambda m: (
                not m["multipage"] and m["scanned"] and not m["waybill"] and not m["illegible"]
            )
        ),
        "single_digital_invoice": pool(
            lambda m: (
                not m["multipage"] and not m["scanned"] and not m["waybill"] and not m["illegible"]
            )
        ),
        "single_waybill": pool(
            lambda m: not m["multipage"] and m["waybill"] and not m["illegible"]
        ),
    }
    cursor = dict.fromkeys(singles, 0)
    while left > 0:
        progressed = False
        for name, docs in singles.items():
            while cursor[name] < len(docs) and page_counts[docs[cursor[name]]["doc_id"]] != 1:
                cursor[name] += 1  # a "single" doc always has 1 page; skip anything else
            if left > 0 and cursor[name] < len(docs):
                chosen[docs[cursor[name]]["doc_id"]] = name
                cursor[name] += 1
                left -= 1
                progressed = True
        if not progressed:
            raise ValueError(
                f"the split cannot fill the bench set ({left} of {n_pages} pages short)"
            )
    return dict(sorted(chosen.items(), key=lambda kv: order[kv[0]]))


# --------------------------------------------------------------------------------------------
# Agreement with batch 1
# --------------------------------------------------------------------------------------------


def _flat(parsed: Mapping[str, Any] | None) -> dict[tuple[str, int | None, str], Any]:
    """(scope, row_idx, field) -> value of a keyed parsed page; {} for an unparsed page."""
    out: dict[tuple[str, int | None, str], Any] = {}
    if not isinstance(parsed, Mapping):
        return out
    header = parsed.get("header")
    if isinstance(header, Mapping):
        out.update({("header", None, str(k)): v for k, v in header.items()})
    rows = parsed.get("line_items")
    if isinstance(rows, list):
        for i, row in enumerate(rows):
            if isinstance(row, Mapping):
                out.update({("row", i, str(k)): v for k, v in row.items()})
    for key in ("doc_type", "page_kind"):
        if key in parsed:
            out[(key, None, key)] = parsed[key]
    return out


def page_agreement(
    ref: Sequence[Mapping[str, Any]], other: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Byte-identical page-JSON rate and field-level agreement rate of `other` vs `ref` traces.

    Pages are matched by (doc_id, page). Byte-identical = equal ``raw_text``. Field agreement =
    equal values over the UNION of the (scope, row_idx, field) keys of both pages (a row or field
    present in only one counts as a disagreement); a page pair with no fields on either side
    adds nothing, and the rate is 1.0 if no page has fields at all and all pages are identical.
    """
    pages = {(t["doc_id"], p["page"]): p for t in ref for p in t["pages"]}
    got = {(t["doc_id"], p["page"]): p for t in other for p in t["pages"]}
    if set(pages) != set(got):
        raise ValueError("the two runs do not cover the same pages")
    same = fields = agree = 0
    for key, a in pages.items():
        b = got[key]
        same += a["raw_text"] == b["raw_text"]
        fa, fb = _flat(a["parsed"]), _flat(b["parsed"])
        for k in fa.keys() | fb.keys():
            fields += 1
            agree += k in fa and k in fb and fa[k] == fb[k]
    n = len(pages)
    if not n:
        raise ValueError("no pages to compare")
    return {
        "n_pages": n,
        "byte_identical_pages": same,
        "byte_identical_rate": same / n,
        "n_fields": fields,
        "field_agreement_rate": (agree / fields) if fields else float(same == n),
    }


def scorer_identical(
    ref: Sequence[Mapping[str, Any]],
    other: Sequence[Mapping[str, Any]],
    gold: Mapping[str, Mapping[str, Any]] | None,
) -> bool:
    """True iff the official scorer gives the identical per-document result for both runs.

    Fails closed: without gold the answer is False.
    """
    if not gold:
        return False
    docs = {t["doc_id"] for t in ref}
    if docs != {t["doc_id"] for t in other}:
        return False
    g = {d: gold[d] for d in docs if d in gold}
    if set(g) != docs:
        return False
    a = ev.per_doc_results({t["doc_id"]: t["prediction"] for t in ref}, g)
    b = ev.per_doc_results({t["doc_id"]: t["prediction"] for t in other}, g)
    return bool(a == b)


# --------------------------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------------------------


def _vram_ok(r: Mapping[str, Any]) -> bool:
    v = r.get("peak_vram_bytes")
    return isinstance(v, (int, float)) and v <= VRAM_LIMIT_BYTES


def choose_batch_size(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply `SELECTION_RULE` to per-size results; returns ``{chosen, deviation, reasons}``.

    Each result needs ``batch_size``, ``ok``, ``peak_vram_bytes``, ``byte_identical_rate``,
    ``field_agreement_rate``, ``scorer_identical`` (the entry of size 1 is the reference and is
    never a candidate: it is what is used when nothing else qualifies). ``deviation`` is None
    unless step two chose a size that is not byte-identical.
    """
    reasons: dict[str, str] = {}
    step1: list[int] = []
    step2: list[int] = []
    for r in sorted(results, key=lambda r: r["batch_size"]):
        b = r["batch_size"]
        if b == 1:
            continue
        if not r.get("ok"):
            reasons[str(b)] = f"failed: {r.get('error', 'no result')}"
        elif not _vram_ok(r):
            reasons[str(b)] = f"peak VRAM {r.get('peak_vram_bytes')} not measured or > 14.5 GiB"
        elif r["byte_identical_rate"] == 1.0:
            step1.append(b)
            reasons[str(b)] = "100% byte-identical, VRAM ok"
        elif r["field_agreement_rate"] >= MIN_FIELD_AGREEMENT and r["scorer_identical"]:
            step2.append(b)
            reasons[str(b)] = (
                f"not byte-identical ({r['byte_identical_rate']:.4f}); field agreement "
                f"{r['field_agreement_rate']:.4f} >= 0.995 and identical scorer result"
            )
        else:
            reasons[str(b)] = (
                f"rejected: byte-identical {r['byte_identical_rate']:.4f}, field agreement "
                f"{r['field_agreement_rate']:.4f}, scorer identical {r['scorer_identical']}"
            )
    if step1:
        return {"chosen": max(step1), "deviation": None, "reasons": reasons}
    if step2:
        b = max(step2)
        r = next(r for r in results if r["batch_size"] == b)
        dev = {
            "kind": "not_byte_identical",
            "batch_size": b,
            "byte_identical_rate": r["byte_identical_rate"],
            "field_agreement_rate": r["field_agreement_rate"],
            "scorer_identical": True,
            "text": (
                f"DEVIATION: batch size {b} is NOT byte-identical to batch 1 "
                f"({100 * r['byte_identical_rate']:.1f}% of pages); chosen because field "
                f"agreement is {100 * r['field_agreement_rate']:.2f}% (>= 99.5%) and the scorer "
                "result on the bench pages is identical"
            ),
        }
        return {"chosen": b, "deviation": dev, "reasons": reasons}
    return {"chosen": 1, "deviation": None, "reasons": reasons}


# --------------------------------------------------------------------------------------------
# Running the bench
# --------------------------------------------------------------------------------------------


def _read_traces(run_dir: Path) -> list[dict[str, Any]]:
    text = (run_dir / "trace.jsonl").read_text(encoding="utf-8")
    return [json.loads(ln) for ln in text.splitlines() if ln.strip()]


def _speed(traces: Sequence[Mapping[str, Any]]) -> tuple[float | None, int | None]:
    """(pages per hour from the summed call latencies, max peak VRAM bytes)."""
    pages = [p for t in traces for p in t["pages"]]
    lat = sum(p["meta"].get("latency_s") or 0.0 for p in pages)
    vram = [p["meta"]["peak_vram_bytes"] for p in pages if p["meta"].get("peak_vram_bytes")]
    return (3600.0 * len(pages) / lat if lat > 0 else None), (max(vram) if vram else None)


def run_bench(
    cfg: SpikeConfig,
    backend: VlmBackend,
    doc_ids: Sequence[str],
    out_dir: Path,
    batch_sizes: Sequence[int] = BATCH_SIZES,
    data_root: Path | None = None,
    logprobs: bool = True,
) -> dict[str, Any]:
    """Run the bench; writes ``<out_dir>/bench_result.json`` and returns it.

    One run per batch size under ``<out_dir>/b<N>/`` (resumable: a finished size is not redone).
    Runs with logprobs on by default because the production run captures them. No batch fallback
    here: a size that raises (OOM) is recorded as failed. A throwaway batch-1 run of one
    single-page document first absorbs the CUDA / allocator warm-up. Throughput counts model call
    time only (not model load, image decoding, merging); peak VRAM is the maximum over the size's
    calls.
    """
    if 1 not in batch_sizes:
        raise ValueError("the bench needs batch size 1 as the reference")
    root = paths.data_dir() if data_root is None else Path(data_root)
    out_dir = Path(out_dir)
    gold, _meta = load_gold_and_meta([BENCH_SPLIT], root)
    counts = {d: len(doc_page_images(BENCH_SPLIT, d, root)) for d in doc_ids}
    warm = next((d for d in doc_ids if counts[d] == 1), None)
    if warm is not None and not (out_dir / "warmup" / "trace.jsonl").is_file():
        run_spike(cfg, [warm], BENCH_SPLIT, "warmup", backend, runs_root=out_dir, data_root=root,
                  logprobs=logprobs, batch_size=1)  # fmt: skip
    results: list[dict[str, Any]] = []
    ref: list[dict[str, Any]] | None = None
    for b in sorted(batch_sizes):
        run_id = f"b{b}"
        entry: dict[str, Any] = {"batch_size": b, "ok": False}
        t0 = time.perf_counter()
        try:
            resume = (out_dir / run_id / "trace.jsonl").is_file()
            run_spike(cfg, doc_ids, BENCH_SPLIT, run_id, backend, runs_root=out_dir,
                      data_root=root, logprobs=logprobs, batch_size=b, fallback=False,
                      resume=resume)  # fmt: skip
            traces = _read_traces(out_dir / run_id)
        except Exception as exc:  # OOM and everything else: this size is out, the others go on
            if b == 1:
                raise
            entry["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            results.append(entry)
            print(f"bench: batch size {b} FAILED ({entry['error']})", file=sys.stderr)
            continue
        pph, vram = _speed(traces)
        entry.update(ok=True, wall_s=round(time.perf_counter() - t0, 1), pages_per_hour=pph,
                     peak_vram_bytes=vram)  # fmt: skip
        if b == 1:
            ref = traces
            entry.update(byte_identical_rate=1.0, field_agreement_rate=1.0, scorer_identical=True)
        else:
            assert ref is not None
            entry.update(page_agreement(ref, traces))
            entry["scorer_identical"] = scorer_identical(ref, traces, gold)
        results.append(entry)
    decision = choose_batch_size(results)
    result = {
        "schema": BENCH_SCHEMA,
        "code_sha": git_commit(),
        "config_hash": cfg.config_hash,
        "config": cfg.name,
        "model": {"id": backend.model_id, "revision": backend.revision},
        "seed": cfg.backend.seed,
        "docs": list(doc_ids),
        "n_pages": sum(counts.values()),
        "logprobs": logprobs,
        "batch_sizes": sorted(batch_sizes),
        "vram_limit_bytes": VRAM_LIMIT_BYTES,
        "min_field_agreement": MIN_FIELD_AGREEMENT,
        "rule": SELECTION_RULE,
        "results": results,
        "chosen_batch_size": decision["chosen"],
        "deviation": decision["deviation"],
        "reasons": decision["reasons"],
    }
    atomic_write(out_dir / RESULT_NAME, json.dumps(result, indent=1))
    return result


def load_bench_result(path: Path, cfg: SpikeConfig, backend: VlmBackend) -> dict[str, Any]:
    """Read and validate a bench result for THIS code, config and model; raises ValueError if it
    belongs to something else (a stale bench must not size a different run)."""
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    problems = []
    if result.get("schema") != BENCH_SCHEMA:
        problems.append(f"schema {result.get('schema')!r}")
    if result.get("config_hash") != cfg.config_hash:
        problems.append("config hash differs")
    if (result.get("model") or {}).get("revision") != backend.revision:
        problems.append("model revision differs")
    if result.get("code_sha") != git_commit():
        problems.append(f"code sha differs ({result.get('code_sha')} vs {git_commit()})")
    if not isinstance(result.get("chosen_batch_size"), int) or result["chosen_batch_size"] < 1:
        problems.append("no valid chosen_batch_size")
    if problems:
        raise ValueError(f"{path} does not apply to this run: {'; '.join(problems)}")
    return result


def chosen_from_result(result: Mapping[str, Any]) -> int:
    """The chosen batch size of a validated bench result."""
    return int(result["chosen_batch_size"])


def format_banner(result: Mapping[str, Any]) -> str:
    """Printable bench table, the decision, any deviation and the test-submission reminder."""
    bar = "=" * 100
    lines = [
        bar,
        f"BATCH-SIZE BENCH: {result['n_pages']} dev pages, config {result['config']}",
        bar,
        f"{'batch':>5}  {'pages/h':>9}  {'peak VRAM GiB':>13}  {'byte-identical':>14}  "
        f"{'field agree':>11}  {'scorer same':>11}  note",
    ]
    for r in result["results"]:
        if not r.get("ok"):
            lines.append(f"{r['batch_size']:>5}  FAILED  {r.get('error', '')}")
            continue
        v = r.get("peak_vram_bytes")
        pph = r.get("pages_per_hour")
        lines.append(
            f"{r['batch_size']:>5}  {pph if pph is None else format(pph, '9.1f'):>9}  "
            f"{'n/a' if v is None else format(v / 2**30, '13.2f'):>13}  "
            f"{100 * r['byte_identical_rate']:>13.1f}%  "
            f"{100 * r['field_agreement_rate']:>10.2f}%  {str(r['scorer_identical']):>11}  "
            f"{result['reasons'].get(str(r['batch_size']), 'reference')}"
        )
    lines.append(f"rule: {result['rule']}")
    lines.append(f"CHOSEN BATCH SIZE: {result['chosen_batch_size']}")
    if result.get("deviation"):
        lines.append(result["deviation"]["text"])
    lines.append(
        "The bench result (bench_result.json) must be used for the TEST submission too: greedy "
        "outputs are only comparable at the same batch size "
        "(shipdoc.runmeta.require_same_batch_size)."
    )
    lines.append(bar)
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def write_bench_list(out: Path, root: Path | None = None) -> dict[str, str]:
    """Write the bench doc list (a JSON list of dev doc_ids) from meta/dev.json; returns groups."""
    data = paths.data_dir() if root is None else Path(root)
    meta = ev.load_json(paths.REPO_ROOT / "meta" / f"{BENCH_SPLIT}.json")
    counts = {m["doc_id"]: len(doc_page_images(BENCH_SPLIT, m["doc_id"], data)) for m in meta}
    groups = pick_bench_docs(meta, counts)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(list(groups), indent=1) + "\n", encoding="utf-8", newline="\n")
    return groups


def main_bench(args: Any) -> int:
    """CLI glue for ``python -m shipdoc bench`` (see cli.py)."""
    if args.write_list:
        groups = write_bench_list(Path(args.write_list))
        print(f"wrote {args.write_list}: {len(groups)} docs {groups}")
        return 0
    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)
    result_path = out_dir / RESULT_NAME
    backend = make_backend(args.backend, cfg, BENCH_SPLIT)
    if args.reuse and result_path.is_file():
        try:
            result = load_bench_result(result_path, cfg, backend)
            print(f"bench: reusing {result_path}")
            print(format_banner(result))
            return 0
        except ValueError as exc:
            print(f"bench: not reusing {result_path}: {exc}", file=sys.stderr)
    sizes = tuple(int(x) for x in args.batch_sizes.split(","))
    doc_ids = load_doc_ids(args.docs)
    result = run_bench(cfg, backend, doc_ids, out_dir, sizes)
    print(format_banner(result))
    return 0
