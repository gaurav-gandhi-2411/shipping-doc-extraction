"""M2: output-token budget per model from the gold targets (train pages).

Every train page's gold target is rendered in the per-page output format (``--format json`` is the
current `page_schema` format, ``compact`` the short-key format of M3), counted with the REAL model
tokenizers, and ``max_new_tokens`` is derived per model:

    max_new_tokens = roundup64(ceil(p99_page_tokens * HEADROOM))      HEADROOM = 1.25

where ``p99_page_tokens`` is the 99th percentile (numpy linear interpolation) of the per-page
target token counts of the 400 train docs, plus one token for the end-of-turn stop.

Row-to-page assignment: ``<SHIPDOC_RUNS_DIR>/provenance/locations.jsonl`` (row records with an
assigned OCR line); a row without a page goes on the page where its supplier part number is found
in the OCR, else on the last page. Tokenizer files are fetched with ``huggingface_hub`` into
HF_HOME (tokenizer.json / tokenizer_config.json / vocab / merges only).

Run (throwaway deps, nothing is added to the project):

    uv run --with tokenizers --with huggingface_hub python scripts/token_budget.py \
        --format json --out reports/token_budget.md
"""

# ruff: noqa: E501  # markdown table literals

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from _corpus import DocRec, load_corpus

from shipdoc import locate as loc
from shipdoc import paths
from shipdoc.extract import compact_page
from shipdoc.targets import gold_page_payloads, render_target

HEADROOM = 1.25
ROUND_TO = 64
# model key -> (repo, pinned revision); the revisions equal the ones in configs/spike_*.yaml.
TOKENIZERS: dict[str, tuple[str, str]] = {
    "qwen35_4b": ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
    "nuextract3": ("numind/NuExtract3", "c99dc8f5641b866aa0192b6ea78f84bf9f3535f1"),
    "qwen3vl_8b": ("Qwen/Qwen3-VL-8B-Instruct", "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"),
}
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
OLD_CAP = 1536


def round_up(n: float, multiple: int = ROUND_TO) -> int:
    """Smallest multiple of `multiple` that is >= n."""
    return int(math.ceil(n / multiple) * multiple)


def budget_from_p99(p99: float, headroom: float = HEADROOM) -> int:
    """``max_new_tokens`` = roundup64(ceil(p99 * headroom))."""
    return round_up(math.ceil(p99 * headroom))


def dist(values: list[int]) -> dict[str, float]:
    """mean / p50 / p95 / p99 / max of a token-count list."""
    a = np.asarray(values, dtype=float)
    return {
        "n": len(values),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
    }


ROW_STATS: dict[str, Any] = {}


def row_pages_of(docs: list[DocRec]) -> dict[str, list[int]]:
    """Page (0-based) of every gold row of every doc, with the fallbacks described above.

    Source counts and the sha256 of locations.jsonl land in `ROW_STATS` (report provenance).
    """
    assigned: dict[str, dict[int, int]] = defaultdict(dict)
    path = paths.runs_dir() / "provenance" / "locations.jsonl"
    blob = path.read_bytes()
    ROW_STATS.update(sha256=hashlib.sha256(blob).hexdigest(), from_locations=0, from_ocr=0, last=0)
    for line in blob.decode("utf-8").splitlines():
        rec = json.loads(line)
        if rec["scope"] == "row" and rec.get("assigned") and rec.get("page") is not None:
            assigned[rec["doc_id"]][rec["row_idx"]] = int(rec["page"])
    out: dict[str, list[int]] = {}
    for d in docs:
        n_pages = len(d.gold.get("pages") or [None])
        rows = d.gold.get("line_items") or []
        pages: list[int] | None = None
        result: list[int] = []
        for i, row in enumerate(rows):
            if i in assigned[d.doc_id]:
                result.append(assigned[d.doc_id][i])
                ROW_STATS["from_locations"] += 1
                continue
            if pages is None:
                pages = d.pages()
            spn = row.get("supplier_part_number")
            m = loc.locate(spn, "supplier_part_number", pages) if spn and pages else None
            ROW_STATS["from_ocr" if m is not None else "last"] += 1
            result.append(m.page if m is not None else n_pages - 1)
        out[d.doc_id] = result
    return out


def fetch_tokenizers() -> dict[str, Any]:
    """Download tokenizer files only (into HF_HOME) and load the `tokenizers` objects."""
    paths.apply_env()
    from huggingface_hub import hf_hub_download, list_repo_files
    from tokenizers import Tokenizer

    out = {}
    for key, (repo, rev) in TOKENIZERS.items():
        have = set(list_repo_files(repo, revision=rev))
        got = {f: hf_hub_download(repo, f, revision=rev) for f in TOKENIZER_FILES if f in have}
        out[key] = Tokenizer.from_file(got["tokenizer.json"])
    return out


def measure(
    docs: list[DocRec],
    fmt: str,
    tokenizers: dict[str, Any],
    row_pages: dict[str, list[int]] | None = None,
) -> dict[str, Any]:
    """Per-model token distribution of the gold page targets of `docs` in format `fmt`."""
    row_pages = row_pages or row_pages_of(docs)
    texts: list[tuple[str, int, int, str]] = []  # doc_id, page, n_rows, text
    for d in docs:
        n_pages = len(d.gold.get("pages") or [None])
        for p, payload in enumerate(gold_page_payloads(d.gold, n_pages, row_pages[d.doc_id])):
            body = payload
            if fmt == "compact":
                body = compact_page(payload)
            texts.append((d.doc_id, p, len(payload["line_items"]), render_target(body)))
    res: dict[str, Any] = {"n_pages": len(texts), "models": {}}
    for key, tok in tokenizers.items():
        counts = [len(tok.encode(t, add_special_tokens=False).ids) + 1 for *_, t in texts]
        top = max(range(len(texts)), key=lambda i: counts[i])
        d = dist(counts)
        cap = budget_from_p99(d["p99"])
        res["models"][key] = {
            **d,
            "chars_mean": statistics.fmean(len(t) for *_, t in texts),
            "largest_page": {
                "doc_id": texts[top][0],
                "page": texts[top][1] + 1,
                "rows": texts[top][2],
                "tokens": counts[top],
            },
            "max_new_tokens": cap,
            "pages_over_cap": sum(c > cap for c in counts),
            "pages_over_old_cap": sum(c > OLD_CAP for c in counts),
            "max_cap_alt": budget_from_p99(d["max"]),
        }
        res["models"][key]["counts"] = counts
    return res


def render_md(res: dict[str, Any], fmt: str, cmd: str) -> str:
    """Markdown report (doc_ids and aggregates only; no label values)."""
    lines = [
        f"# Output-token budget from gold ({fmt} format, train pages)",
        "",
        "**Every number in this report is UNVERIFIED** (produced by `scripts/token_budget.py`, not yet independently recomputed).",
        "",
        f"Command: `{cmd}`",
        "",
        f"Train pages: {res['n_pages']}. Targets rendered by `shipdoc.targets.gold_page_payloads` + `render_target` "
        "(identity header on page 1, totals on the last page, union header with nulls, "
        'single-line JSON with `", "`/`": "` separators, +1 token for the stop token). '
        "Row-to-page: `<SHIPDOC_RUNS_DIR>/provenance/locations.jsonl`, else the page where the part number is found in the OCR, else the last page.",
        "",
        f"locations.jsonl sha256 `{ROW_STATS['sha256'][:16]}`; rows placed from it: {ROW_STATS['from_locations']}, "
        f"by part-number OCR search: {ROW_STATS['from_ocr']}, on the last page (fallback): {ROW_STATS['last']}.",
        "",
        f"Formula: `max_new_tokens = roundup64(ceil(p99_page_tokens * {HEADROOM}))`, "
        "p99 over per-page target tokens (numpy linear percentile).",
        "",
        "| model (tokenizer) | mean | p50 | p95 | p99 | max | largest page (doc, page, rows) | max_new_tokens | train pages over it | pages over old cap 1536 |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for key, m in res["models"].items():
        lp = m["largest_page"]
        lines.append(
            f"| {key} | {m['mean']:.1f} | {m['p50']:.0f} | {m['p95']:.0f} | {m['p99']:.0f} | {m['max']:.0f} "
            f"| {lp['doc_id']} p{lp['page']}, {lp['rows']} rows | {m['max_new_tokens']} "
            f"| {m['pages_over_cap']} | {m['pages_over_old_cap']} |"
        )
    worst = max(m["max"] for m in res["models"].values())
    verdict = (
        "the old cap was BELOW the observed max, so gold-length pages were being truncated."
        if worst > OLD_CAP
        else "the old cap was NOT below the observed max, so it does not explain truncation of "
        "a gold-length page (a model that rambles past the gold length still can be truncated)."
    )
    lines += [
        "",
        f"Old cap {OLD_CAP} vs the largest observed gold page ({worst:.0f} tokens): {verdict}",
    ]
    return "\n".join(lines) + "\n"


def render_compare_md(res: dict[str, dict[str, Any]], cmd: str) -> str:
    """reports/output_format.md: json vs compact token counts, budgets, s/page projection."""
    n = res["json"]["n_pages"]
    lines = [
        "# Output format: json vs compact (gold-rendered targets, train pages)",
        "",
        "**Every number in this report is UNVERIFIED** (produced by `scripts/token_budget.py --format compare`, not yet independently recomputed).",
        "",
        f"Command: `{cmd}`",
        "",
        f"{n} train pages; same row-to-page assignment for both formats (locations.jsonl sha256 "
        f"`{ROW_STATS['sha256'][:16]}`); +1 stop token; `max_new_tokens = roundup64(ceil(p99 * {HEADROOM}))` "
        "as in `reports/token_budget.md`.",
        "",
        "## Formats",
        "",
        '- `json` (current): `{"doc_type": .., "header": {17 long keys, other type null}, "line_items": [{4 long keys}, ..], "page_kind": ..}`.',
        '- `compact`: `{"dt": .., "h": {inv_no, inv_date, sup, buy, ship_to, cur, total, awb, car, mawb, hawb, orig, dest, shp, cns, pcs, gw}, "r": [[spn, cpn, po, qty], ..], "pk": ..}`; nulls are JSON null; `shipdoc.extract.expand_page` restores the schema JSON deterministically.',
        "",
        "## Schema-constrained decoding: `prefixItems`",
        "",
        "xgrammar 0.2.8 supports `prefixItems` (read from GitHub tag v0.2.8): `cpp/json_schema_converter.cc` "
        "`SchemaParser::ParseArray` parses `prefixItems` into positional item rules and the grammar builder "
        'emits them in order (comment "prefixItems entries are positional", issue #824 fix); '
        "`tests/python/test_json_schema_converter.py` has `test_array_schema_min_max` cases for "
        "`prefixItems` with `minItems`/`maxItems`/`items: false` and the #824 regression tests. "
        'Used schema for a row: `{type: array, prefixItems: [4 x ["string","null"]], items: false, minItems: 4, maxItems: 4}`. '
        "The 1-char-key object fallback (`s`,`c`,`p`,`q`) is therefore NOT used. "
        "Compiled and checked with the real xgrammar 0.2.8 (throwaway env, CPU): "
        "`uv run --with xgrammar==0.2.8 pytest -q tests/test_compact.py -k grammar` accepts a valid page "
        "and rejects 3-item rows, 5-item rows and an integer in a row "
        "(`test_compact_schema_grammar_pins_row_arity`; skipped when xgrammar is absent).",
        "",
        "## Token reduction (tokens per page)",
        "",
        "| model (tokenizer) | json mean | compact mean | mean reduction | json p99 | compact p99 | p99 reduction | json max | compact max | json max_new_tokens | compact max_new_tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in res["json"]["models"]:
        j, c = res["json"]["models"][key], res["compact"]["models"][key]
        lines.append(
            f"| {key} | {j['mean']:.1f} | {c['mean']:.1f} | {100 * (1 - c['mean'] / j['mean']):.1f}% "
            f"| {j['p99']:.0f} | {c['p99']:.0f} | {100 * (1 - c['p99'] / j['p99']):.1f}% "
            f"| {j['max']:.0f} | {c['max']:.0f} | {j['max_new_tokens']} | {c['max_new_tokens']} |"
        )
    lines += [
        "",
        "Compact pages over their own cap: "
        + ", ".join(f"{k} {m['pages_over_cap']}" for k, m in res["compact"]["models"].items())
        + ".",
        "",
        "## Seconds-per-page projection (UNVERIFIED)",
        "",
        "Decode time scales with output tokens:",
        "",
        "    s/page ~= prefill_s + n_out / decode_tok_s",
        "",
        "`prefill_s` (image + prompt tokens) and `decode_tok_s` are PARAMETERS to be measured from the "
        "Step L traces (`latency_s`, `n_input_tokens`, `n_output_tokens` per page in trace.jsonl); "
        "no speed is assumed here. Predicted saving per page = (n_out_json - n_out_compact) / decode_tok_s "
        "(prefill unchanged apart from the shorter prompt text). Mean n_out from the table above: "
        + ", ".join(
            f"{k}: {res['json']['models'][k]['mean']:.0f} -> {m['mean']:.0f}"
            for k, m in res["compact"]["models"].items()
        )
        + ".",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--format", choices=("json", "compact", "compare"), default="json")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--raw-out", type=Path, default=None, help="write the full result JSON here")
    args = ap.parse_args(argv)
    docs = [d for d in load_corpus() if d.split == "train"]
    toks = fetch_tokenizers()
    if args.format == "compare":
        rows = row_pages_of(docs)
        both = {f: measure(docs, f, toks, rows) for f in ("json", "compact")}
        res = both
    else:
        res = measure(docs, args.format, toks)
    cmd = " ".join(
        [
            "uv run --with tokenizers --with huggingface_hub python scripts/token_budget.py",
            *(argv or sys.argv[1:]),
        ]
    )
    md = (
        render_compare_md(res, cmd)
        if args.format == "compare"
        else render_md(res, args.format, cmd)
    )
    if args.out:
        args.out.write_text(md, encoding="utf-8")
    if args.raw_out:
        args.raw_out.write_text(json.dumps(res), encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
