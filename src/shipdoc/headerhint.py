"""Continuation-page header hint: the page-1 table header row, put in the prompt of pages >= 2.

Why: a continuation page of a multi-page invoice often has no header row, so the model has to
guess which column is which (the column-shift / rotation errors of ``shipdoc.diagnostics``). The
page-1 table header names the columns. This module builds ONE prompt line from it::

    Column headers from page 1: Item Part No. Description Qty Unit Price Amount

OPT-IN, default OFF. Only a config with ``header_hint: true`` (``configs/spike_qwen35_4b_img_only_
keyed_hdrhint.yaml``) turns it on; every other config, the production prompt v2 and its hashes
are byte-for-byte unchanged (``tests/test_headerhint.py`` pins them).

DEPLOYMENT COST: the hint is OCR OUTPUT of page 1, so it needs OCR text at inference time (Paddle,
the ``ocr`` dependency group, a GPU/CPU pass over page 1 of every multi-page document). This is a
deployment cost like R2's, which the image-only production path does not pay. The A/B notebook
(07_hdrhint_ab) does NOT run OCR: it reads the precomputed ``ocr_cache.zip`` as 01 and 03 do.
FALLBACK: when OCR for page 1 is absent (no cache file, unreadable file) the hint is simply
absent and the prompt is the production prompt; the reason is recorded per page.

Generic by construction: no supplier id, no cluster, no gold. The text is what the OCR read, in
the order the page shows it (left to right by box x inside the header line).

Rules (all pure, ``tests/test_headerhint.py``):

* page 1 and one-page documents get no hint;
* the header line is ``shipdoc.layout.analyze_page(page1).header_line`` and only when its method
  is ``table_header``; any other method -> no hint (reason ``no_table_header``);
* a header printed on two OCR lines (``Part`` over ``Number``) is joined: the line directly above
  or below is merged when it is close (vertical gap <= ``JOIN_GAP_FACTOR`` x the header's median
  box height), sits inside the header's horizontal span, is not a banner, not item-like and holds
  no numeric token. UNVERIFIED on real pages beyond the unit tests: the rate of joined hints is
  reported by the A/B (``joined_lines``);
* control characters are stripped, whitespace collapsed;
* length cap ``HINT_MAX_CHARS`` characters (about ``HINT_MAX_TOKENS`` tokens at the repo's
  4-chars-per-token approximation): cells are kept left to right until the next one would exceed
  the cap (a cell is cut only when it alone exceeds the cap), the trace records ``truncated``.

Also here (pure, tested): the A/B bookkeeping of notebook 07: doc subsets, the strict check that
the 02 run may be the control (``control_check``: it refuses, nothing is decoded, unless config /
prompt hash, revision, batch size, seed, logprobs, output format and full doc coverage match), the
page-1 NOISE FLOOR, the decision rule, and the comparison itself. CLI:
``python -m shipdoc.headerhint plan | compare`` (see ``scripts/colab_build_hdrhint.py``).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from shipdoc.layout import analyze_page, is_item_like
from shipdoc.ocr import PageOcr, ReadItem, load_page, page_items

HINT_PREFIX = "Column headers from page 1: "
CHARS_PER_TOKEN = 4  # the repo's token approximation (prompts.ocr_block without a tokenizer)
HINT_MAX_TOKENS = 64
HINT_MAX_CHARS = HINT_MAX_TOKENS * CHARS_PER_TOKEN  # cap of the hint text, prefix excluded
JOIN_GAP_FACTOR = 0.75
JOIN_MAX_TOKENS = 8  # a joined neighbour line is a few header words, not a paragraph
_MISSING = object()  # a leaf absent on one side (distinct from a None value)
_NUMERIC = re.compile(r"[\d.,]*\d[\d.,]*")

REASONS = ("ok", "page_1", "single_page", "no_ocr", "no_table_header", "empty_header")


# --------------------------------------------------------------------------------------------
# Prompt assembly (pure)
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class HeaderHint:
    """The hint of one page, or why there is none (recorded in the trace page)."""

    applied: bool
    reason: str  # one of REASONS
    text: str = ""
    n_cells: int = 0
    truncated: bool = False
    joined_lines: int = 0  # OCR lines merged into the header (1 = a plain one-line header)

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible dict for ``trace.jsonl``."""
        return asdict(self)


def sanitize(text: str) -> str:
    """`text` without control / format characters (newlines and tabs become spaces), collapsed."""
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat == "Cc":
            out.append(" ")
        elif cat != "Cf":
            out.append(ch)
    return " ".join("".join(out).split())


def cap_cells(cells: Sequence[str], max_chars: int = HINT_MAX_CHARS) -> tuple[str, bool]:
    """Cells joined by one space, kept left to right while the text fits `max_chars`.

    Returns (text, truncated). A cell is never cut in half unless the FIRST cell alone exceeds the
    cap (then it is cut, so a header is never silently reduced to nothing).
    """
    kept: list[str] = []
    used = 0
    for cell in cells:
        add = len(cell) + (1 if kept else 0)
        if used + add > max_chars:
            if not kept:
                return cell[:max_chars].rstrip(), True
            return " ".join(kept), True
        kept.append(cell)
        used += add
    return " ".join(kept), False


def _box_h(items: Sequence[ReadItem]) -> float:
    return float(statistics.median(it.box[3] - it.box[1] for it in items))


def _joins_header(header: Sequence[ReadItem], other: Sequence[ReadItem]) -> bool:
    """True if `other` (an adjacent OCR line) reads as the second row of a two-row header."""
    h = _box_h(header)
    x0, x1 = min(it.box[0] for it in header), max(it.box[2] for it in header)
    top_h, bot_h = min(it.box[1] for it in header), max(it.box[3] for it in header)
    top_o, bot_o = min(it.box[1] for it in other), max(it.box[3] for it in other)
    gap = top_o - bot_h if top_o >= top_h else top_h - bot_o
    text = " ".join(it.text for it in other)
    if gap > JOIN_GAP_FACTOR * h or len(text.split()) > JOIN_MAX_TOKENS:
        return False
    if min(it.box[0] for it in other) < x0 - h or max(it.box[2] for it in other) > x1 + h:
        return False
    return not (is_item_like(text) or any(_NUMERIC.fullmatch(t) for t in text.split()))


def header_cells(page: PageOcr) -> tuple[list[str], int, str]:
    """(cell texts left to right, OCR lines joined, status) of the table header of `page`.

    status is ``ok``, ``no_table_header`` (``analyze_page`` found no table-header line) or
    ``empty_header`` (the line has no text left after sanitising).
    """
    layout = analyze_page(page)
    if layout.method != "table_header" or layout.header_line is None:
        return [], 0, "no_table_header"
    by_line: dict[int, list[ReadItem]] = defaultdict(list)
    for it in page_items(page):
        by_line[it.line_idx].append(it)
    h = layout.header_line
    stacked: list[ReadItem] = []
    joined = 1
    for n in (h - 1, h + 1):
        if n in by_line and layout.region(n) != "banner" and _joins_header(by_line[h], by_line[n]):
            stacked += by_line[n]
            joined += 1
    cells = [c for c in (sanitize(t) for t in _columns(by_line[h], stacked)) if c]
    return (cells, joined, "ok") if cells else ([], joined, "empty_header")


def _columns(primary: Sequence[ReadItem], stacked: Sequence[ReadItem]) -> list[str]:
    """Column texts left to right.

    Each cell of a joined second row goes under the header cell it overlaps most horizontally (top
    row first), so ``Part`` over ``No.`` reads ``Part No.`` and the column order stays the page's
    even when the stacked cells start at slightly different x.
    """
    cols: list[list[ReadItem]] = [[it] for it in sorted(primary, key=lambda i: i.box[0])]
    for it in sorted(stacked, key=lambda i: i.box[0]):
        best: list[ReadItem] | None = None
        overlap = 0.0
        for col in cols:
            right = min(it.box[2], max(c.box[2] for c in col))
            ov = right - max(it.box[0], min(c.box[0] for c in col))
            if ov > overlap:
                best, overlap = col, ov
        if best is None:
            cols.append([it])  # overlaps no header cell: its own column, placed by x below
        else:
            best.append(it)
    cols.sort(key=lambda col: min(c.box[0] for c in col))
    return [" ".join(c.text for c in sorted(col, key=lambda i: i.box[1])) for col in cols]


def page1_header_hint(page1: PageOcr | None) -> HeaderHint:
    """The hint built from the OCR of page 1 (None = OCR absent -> no hint, reason ``no_ocr``)."""
    if page1 is None:
        return HeaderHint(False, "no_ocr")
    cells, joined, status = header_cells(page1)
    if status != "ok":
        return HeaderHint(False, status, joined_lines=joined)
    text, truncated = cap_cells(cells)
    return HeaderHint(True, "ok", text, len(cells), truncated, joined)


def hint_for_page(page_idx: int, n_pages: int, page1: PageOcr | None) -> HeaderHint:
    """Hint of the 0-based page `page_idx` of an `n_pages`-page document."""
    if n_pages <= 1:
        return HeaderHint(False, "single_page")
    if page_idx == 0:
        return HeaderHint(False, "page_1")
    return page1_header_hint(page1)


def hinted_prompt(prompt: str, hint: HeaderHint) -> str:
    """`prompt` followed by the hint line; `prompt` itself (same object) when there is no hint."""
    if not hint.applied:
        return prompt
    return f"{prompt}\n{HINT_PREFIX}{hint.text}"


def load_page1(split: str, doc_id: str, ocr_root: Path) -> PageOcr | None:
    """Cached OCR of page 1 of `doc_id`; None when it is absent or unreadable (the fallback)."""
    try:
        return load_page(split, f"{doc_id}_p1", ocr_root)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def hinted_prompt_for_job(
    prompt: str,
    split: str,
    doc_id: str,
    page_idx: int,
    n_pages: int,
    ocr_root: Path,
    cache: dict[str, PageOcr | None],
) -> tuple[str, dict[str, Any]]:
    """(prompt with the hint, trace record) of one page: the spike hook.

    `cache` holds the page-1 OCR per doc, so it is loaded once per document.
    """
    page1: PageOcr | None = None
    if n_pages > 1 and page_idx > 0:
        if doc_id not in cache:
            cache[doc_id] = load_page1(split, doc_id, ocr_root)
        page1 = cache[doc_id]
    hint = hint_for_page(page_idx, n_pages, page1)
    return hinted_prompt(prompt, hint), hint.to_dict()


def require_supported(arm: str) -> None:
    """The hint is for the image-only arm (the img_ocr arm already carries OCR text)."""
    if arm != "img_only":
        raise ValueError(f"header_hint needs arm img_only, got {arm!r}")


# --------------------------------------------------------------------------------------------
# A/B bookkeeping (pure)
# --------------------------------------------------------------------------------------------

BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 42
DECISION_RULE = (
    "ADOPT iff the paired OVERALL-delta 95% CI lower bound (hint - control) > 0 on the pooled "
    "multipage docs AND the row F1 delta is not negative at its point estimate AND over-null "
    "cells (header + row, on the pooled docs) do not increase; otherwise DO NOT ADOPT"
)
CONTROL_NOTE = (
    "the 02 zero-shot run restricted to the pooled doc ids (no control rerun); batch size equal "
    "(8), but the batch composition differs (the 02 run had 500 docs, this arm runs only the "
    "pooled ones) and byte-identical outputs across batch compositions were verified by the 02 "
    "bench on 12 pages only: a small difference from batching alone is possible, bounded by the "
    "NOISE FLOOR above."
)
CAVEATS = (
    "Train docs are zero-shot (unseen by construction), so pooling them with dev100 is "
    "legitimate. The dev100 multipage subset is small (n printed above): its interval is wide.",
    "The hint is OCR output of page 1: adopting it makes inference depend on OCR (Paddle) at "
    "run time, a deployment cost like R2. This A/B reads the precomputed OCR cache.",
)


def select_docs(
    meta: Sequence[Mapping[str, Any]],
    dev100: Sequence[str],
    zeroshot500: Sequence[str],
) -> dict[str, list[str]]:
    """The multipage docs of the A/B: ``dev`` (dev100 and multipage), ``train`` (the 02 run's
    train docs that are multipage), ``pooled`` (dev then train, each in `splits` order)."""
    multi = {m["doc_id"] for m in meta if m.get("multipage")}
    dev = [d for d in dev100 if d in multi]
    train = [d for d in zeroshot500 if d.startswith("train_") and d in multi]
    return {"dev": dev, "train": train, "pooled": dev + train}


def trace_matches(
    trace: Mapping[str, Any], config_hash: str, prompt_hash: str, revision: str
) -> bool:
    """A trace line was written by this config, this prompt and this model revision."""
    return (
        (trace.get("config") or {}).get("hash") == config_hash
        and (trace.get("prompt") or {}).get("hash") == prompt_hash
        and (trace.get("model") or {}).get("revision") == revision
    )


class ControlRefused(ValueError):
    """The existing control run may not stand in for a control arm; nothing may be decoded."""

    def __init__(self, reasons: Sequence[str]) -> None:
        self.reasons = list(reasons)
        super().__init__("control run REFUSED: " + "; ".join(self.reasons))


def control_check(
    manifest: Mapping[str, Any] | None,
    progress: Mapping[str, Any] | None,
    traces: Sequence[Mapping[str, Any]],
    *,
    needed: Sequence[str],
    config_hash: str,
    prompt_hash: str,
    revision: str,
    batch_size: int,
    seed: int,
    output_format: str,
    logprobs: bool = True,
    hint_flag_off: bool = True,
) -> list[str]:
    """Reasons an existing run may NOT be the control of the hint arm (empty list = it may).

    Strict, fail closed (a missing field fails). The control is the 02 run restricted to `needed`.
    The code SHA is deliberately NOT compared (the pinned code legitimately differs from the one
    that wrote the control; the notebook prints ``git diff --stat`` of ``src configs`` instead),
    but an unknown or ``+dirty`` SHA is refused: its diff cannot be shown. What must match:

    * config hash (of the production config computed from the CHECKED-OUT code, which must have
      ``header_hint`` off: `hint_flag_off`), prompt hash (every needed trace records
      ``prompt.hash``, the manifest has no prompt field; compared with the checked-out code's
      ``prompts.prompt_hash``), model revision (manifest and every needed trace);
    * batch size, seed, logprobs, output format (the hint arm runs with the same values), arm
      ``img_only``, unsharded;
    * `progress` status ``complete`` with done == total, and a trace for EVERY needed doc.
    """
    if not manifest:
        return ["no manifest.json"]
    why: list[str] = []
    sha = str(manifest.get("code_sha"))
    checks = [
        (hint_flag_off, "production config of the checked-out code has header_hint on"),
        ((manifest.get("config") or {}).get("hash") == config_hash, "config hash differs"),
        ((manifest.get("model") or {}).get("revision") == revision, "model revision differs"),
        (manifest.get("arm") == "img_only", "arm is not img_only"),
        (manifest.get("output_format") == output_format, "output format differs"),
        (manifest.get("logprobs") is logprobs, f"logprobs differ (hint arm: {logprobs})"),
        (manifest.get("seed") == seed, f"seed is not {seed}"),
        (manifest.get("shard") == "0/1", "run is a shard"),
        (
            manifest.get("batch_size") == batch_size,
            f"batch size {manifest.get('batch_size')} != {batch_size}",
        ),
        ("dirty" not in sha and sha != "None", f"code SHA {sha[:12]} is unknown or dirty"),
    ]
    why += [msg for ok, msg in checks if not ok]
    prog = progress or {}
    if not (
        prog.get("status") == "complete"
        and isinstance(prog.get("done"), int)
        and prog.get("done") == prog.get("total")
    ):
        why.append("progress is not complete (status/done/total)")
    by_doc = {t["doc_id"]: t for t in traces}
    absent = [d for d in needed if d not in by_doc]
    if absent:
        why.append(f"{len(absent)} needed docs not in the run")
    bad = [
        d
        for d in needed
        if d in by_doc and not trace_matches(by_doc[d], config_hash, prompt_hash, revision)
    ]
    if bad:
        why.append(f"{len(bad)} traces differ in config hash / prompt hash / revision")
    return why


def require_control(
    manifest: Mapping[str, Any] | None,
    progress: Mapping[str, Any] | None,
    traces: Sequence[Mapping[str, Any]],
    **kw: Any,
) -> None:
    """`control_check`, raising ``ControlRefused`` (with every reason) unless it passes."""
    why = control_check(manifest, progress, traces, **kw)
    if why:
        raise ControlRefused(why)


def hint_config_reasons(prod: Mapping[str, Any], hint: Mapping[str, Any]) -> list[str]:
    """Why the hint config is not 'production + the one flag' (empty = it is)."""
    diff = {k for k in set(prod) | set(hint) if prod.get(k) != hint.get(k)}
    why = []
    if diff != {"name", "header_hint"}:
        why.append(f"hint config differs from production in {sorted(diff)}, not name+header_hint")
    if hint.get("header_hint") is not True:
        why.append("hint config does not set header_hint: true")
    return why


def _leaves(obj: Any, path: tuple[Any, ...] = ()) -> dict[tuple[Any, ...], Any]:
    """Scalar leaves of a nested dict / list, keyed by their path."""
    if isinstance(obj, Mapping):
        out: dict[tuple[Any, ...], Any] = {}
        for k, v in obj.items():
            out.update(_leaves(v, (*path, k)))
        return out
    if isinstance(obj, list):
        out = {}
        for i, v in enumerate(obj):
            out.update(_leaves(v, (*path, i)))
        return out
    return {path: obj}


def noise_floor(
    control_traces: Mapping[str, Mapping[str, Any]],
    hint_traces: Mapping[str, Mapping[str, Any]],
    ids: Sequence[str],
) -> dict[str, Any]:
    """How much page 1 differs between the arms, where the hint cannot act (page 1 never hints).

    Any difference is batching / session noise (batch composition differs: the control ran inside
    the 500-doc 02 run), not the hint. ``raw_text`` of page 1 should be byte-identical; the share
    of changed parsed cells (union of leaf paths) bounds the noise in every other number.
    Counts and shares only, never values.
    """
    raw_diff = parsed_diff = either = cells = changed = 0
    for d in ids:
        a = (control_traces[d]["pages"] or [{}])[0]
        b = (hint_traces[d]["pages"] or [{}])[0]
        r = a.get("raw_text") != b.get("raw_text")
        la, lb = _leaves(a.get("parsed")), _leaves(b.get("parsed"))
        paths = set(la) | set(lb)
        n_ch = sum(1 for k in paths if la.get(k, _MISSING) != lb.get(k, _MISSING))
        raw_diff += int(r)
        parsed_diff += int(n_ch > 0)
        either += int(r or n_ch > 0)
        cells += len(paths)
        changed += n_ch
    n = len(ids)
    return {
        "n_docs": n,
        "docs_page1_raw_text_differs": raw_diff,
        "raw_text_diff_rate": raw_diff / n if n else math.nan,
        "docs_page1_parsed_differs": parsed_diff,
        "docs_page1_any_differs": either,
        "n_cells": cells,
        "n_cells_changed": changed,
        "cell_change_share": changed / cells if cells else math.nan,
    }


def format_noise_floor(nf: Mapping[str, Any]) -> str:
    """Printable NOISE FLOOR block (counts and shares only)."""
    lines = [
        "NOISE FLOOR (page 1 only: the hint never touches page 1, so any difference between "
        "control and hint arm is batching / session noise, not the hint)",
        f"  docs with page-1 raw_text NOT byte-identical: {nf['docs_page1_raw_text_differs']} of "
        f"{nf['n_docs']} ({100 * nf['raw_text_diff_rate']:.2f}%)",
        f"  docs with any page-1 difference (raw_text or parsed): {nf['docs_page1_any_differs']}",
        f"  page-1 parsed cells changed: {nf['n_cells_changed']} of {nf['n_cells']} "
        f"({100 * nf['cell_change_share']:.2f}%)",
    ]
    if nf["docs_page1_raw_text_differs"]:
        lines.append(
            "  WARNING: page-1 raw_text differs, so the hint is NOT cleanly attributable; read "
            "every delta below against this noise floor."
        )
    return "\n".join(lines)


def decide(
    overall_ci_lo: float, row_f1_delta: float, over_null_hint: int, over_null_control: int
) -> tuple[str, list[str]]:
    """The A/B decision (see ``DECISION_RULE``): ("ADOPT" | "DO NOT ADOPT", failed clauses).

    Fail closed: a NaN anywhere is a failed clause.
    """
    failed: list[str] = []
    if (
        not (isinstance(overall_ci_lo, float) and math.isfinite(overall_ci_lo))
        or overall_ci_lo <= 0
    ):
        failed.append(f"paired OVERALL delta CI lower bound {overall_ci_lo} is not > 0")
    if not (isinstance(row_f1_delta, float) and math.isfinite(row_f1_delta)) or row_f1_delta < 0:
        failed.append(f"row F1 delta {row_f1_delta} is negative (or not finite)")
    if over_null_hint > over_null_control:
        failed.append(f"over-null cells increased: {over_null_control} -> {over_null_hint}")
    return ("DO NOT ADOPT" if failed else "ADOPT"), failed


def row_counts(
    pred: Mapping[str, Any], gold: Mapping[str, Any], truncated: bool, sc: Any
) -> dict[str, int]:
    """Row-level counts of one doc, with ``shipdoc.diagnostics`` definitions.

    ``row_missing``: unpaired gold rows attributed ``missing``; ``row_extra``: predicted rows left
    unused after the attribution; ``rotations`` / ``cpn_equals_po``: as ``row_convention_errors``.
    """
    from shipdoc import diagnostics as dg
    from shipdoc import eval as ev

    counts = {"row_missing": 0, "row_extra": 0, "rotations": 0, "cpn_equals_po": 0}
    if gold.get("doc_type") != "invoice" or pred.get("doc_type") != "invoice":
        return counts
    gr = list(gold.get("line_items") or [])
    pr = [x for x in (pred.get("line_items") or []) if isinstance(x, dict)]
    _, _, unp, ung = ev._pair_rows(sc, pr, gr)
    attributed, free = dg.attribute_unpaired(sc, pr, gr, unp, ung, truncated)
    counts["row_missing"] = sum(r["cause"] == "missing" for r in attributed)
    counts["row_extra"] = len(free)
    base = dg.row_convention_errors(pred, gold, truncated, sc)
    counts["rotations"], counts["cpn_equals_po"] = base["rotations"], base["cpn_equals_po"]
    return counts


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _s_per_page(traces: Iterable[Mapping[str, Any]]) -> float | None:
    lat = [
        p["meta"]["latency_s"]
        for t in traces
        for p in t["pages"]
        if (p.get("meta") or {}).get("latency_s") is not None
    ]
    return float(sum(lat) / len(lat)) if lat else None


def compare_runs(
    control: Mapping[str, Mapping[str, Any]],
    control_traces: Mapping[str, Mapping[str, Any]],
    hint_pred: Mapping[str, Mapping[str, Any]],
    hint_traces: Mapping[str, Mapping[str, Any]],
    gold: Mapping[str, Mapping[str, Any]],
    subsets: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Paired comparison hint - control over ``subsets`` (``pooled`` decides). Refuses (ValueError)
    when either arm lacks a doc of `subsets["pooled"]`. Unmodified official scorer and
    ``shipdoc.eval`` bootstrap (2000 resamples, seed 42, doc level)."""
    from shipdoc import eval as ev
    from shipdoc.oof import over_null_counts

    pooled = list(subsets["pooled"])
    if not pooled:
        raise ValueError("no pooled docs to compare")
    for name, pred in (("control", control), ("hint", hint_pred)):
        gone = [d for d in pooled if d not in pred]
        if gone:
            raise ValueError(f"{name} arm lacks {len(gone)} pooled docs, e.g. {gone[:3]}")
    sc = ev.load_scorer()
    out: dict[str, Any] = {
        "n_resamples": BOOTSTRAP_N,
        "seed": BOOTSTRAP_SEED,
        "noise_floor": noise_floor(control_traces, hint_traces, pooled),
        "subsets": {},
    }
    for name, ids in subsets.items():
        if not ids:  # e.g. a smoke run without train docs: nothing to bootstrap
            continue
        g = {d: dict(gold[d]) for d in ids}
        a = {d: dict(control[d]) for d in ids}
        b = {d: dict(hint_pred[d]) for d in ids}
        pb = ev.paired_bootstrap(a, b, g, n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)

        def rows(
            pred: Mapping[str, Any], traces: Mapping[str, Any], ids: Sequence[str] = ids
        ) -> Any:
            tot = dict.fromkeys(("row_missing", "row_extra", "rotations", "cpn_equals_po"), 0)
            for d in ids:
                trunc = any(not p["json_valid"] for p in traces[d]["pages"])
                for k, v in row_counts(pred[d], gold[d], trunc, sc).items():
                    tot[k] += v
            return tot

        pages = sum(len(hint_traces[d]["pages"]) for d in ids)
        out["subsets"][name] = {
            "n_docs": len(ids),
            "n_pages": pages,
            "OVERALL": {k: pb["OVERALL"][k] for k in ("a", "b", "delta", "lo", "hi")},
            "row_f1": {k: pb["row_f1"][k] for k in ("a", "b", "delta", "lo", "hi")},
            "rows_control": rows(a, control_traces),
            "rows_hint": rows(b, hint_traces),
            "over_null_control": over_null_counts(a, g),
            "over_null_hint": over_null_counts(b, g),
            "s_per_page_control": _s_per_page(control_traces[d] for d in ids),
            "s_per_page_hint": _s_per_page(hint_traces[d] for d in ids),
        }
    p = out["subsets"]["pooled"]
    decision, failed = decide(
        float(p["OVERALL"]["lo"]),
        float(p["row_f1"]["delta"]),
        int(p["over_null_hint"]["over_null_total"]),
        int(p["over_null_control"]["over_null_total"]),
    )
    out["decision"], out["failed_clauses"], out["rule"] = decision, failed, DECISION_RULE
    reasons: dict[str, int] = defaultdict(int)
    joined = 0
    for d in pooled:
        for page in hint_traces[d]["pages"]:
            h = page.get("header_hint")
            if h and h["reason"] not in ("page_1", "single_page"):
                reasons[h["reason"]] += 1
                joined += int(h.get("joined_lines", 1) > 1)
    out["hint_page_reasons"] = dict(sorted(reasons.items()))
    out["hint_joined_two_line_headers"] = joined
    return out


def format_report(res: Mapping[str, Any]) -> str:
    """Readable table of ``compare_runs`` output: counts and aggregates only, no document values."""
    lines = [
        format_noise_floor(res["noise_floor"]),
        "header-hint A/B: hint - control (paired doc-level bootstrap, 2000, seed 42)",
    ]
    for name, s in res["subsets"].items():
        o, r = s["OVERALL"], s["row_f1"]
        lines.append(f"[{name}] {s['n_docs']} docs, {s['n_pages']} pages")
        lines.append(
            f"  OVERALL control {o['a']:.4f} hint {o['b']:.4f} delta {o['delta']:+.4f} "
            f"[{o['lo']:+.4f}, {o['hi']:+.4f}]"
        )
        lines.append(
            f"  row F1  control {r['a']:.4f} hint {r['b']:.4f} delta {r['delta']:+.4f} "
            f"[{r['lo']:+.4f}, {r['hi']:+.4f}]"
        )
        c, h = s["rows_control"], s["rows_hint"]
        for k in c:
            lines.append(f"  {k:<14} control {c[k]:>4} hint {h[k]:>4} ({h[k] - c[k]:+d})")
        cn, hn = s["over_null_control"], s["over_null_hint"]
        lines.append(
            f"  over-null cells control {cn['over_null_total']} hint {hn['over_null_total']}; "
            f"false fills control {cn['false_fill_total']} hint {hn['false_fill_total']}"
        )
        sc_, sh = s["s_per_page_control"], s["s_per_page_hint"]
        fmt = lambda v: "n/a" if v is None else f"{v:.2f}"  # noqa: E731
        lines.append(
            f"  s/page control {fmt(sc_)} hint {fmt(sh)} (control may come from another session)"
        )
    lines.append(f"CONTROL: {res.get('control_note', CONTROL_NOTE)}")
    lines.append(
        f"hint pages by reason (pooled, pages >= 2): {res['hint_page_reasons']}; "
        f"two-line headers joined: {res['hint_joined_two_line_headers']}"
    )
    lines.append(f"RULE: {res['rule']}")
    for c in res["failed_clauses"]:
        lines.append(f"FAILED CLAUSE: {c}")
    lines.append(f"DECISION: {res['decision']}")
    lines += [f"CAVEAT: {c}" for c in CAVEATS]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# CLI (called by notebook 07)
# --------------------------------------------------------------------------------------------


def _load_run(run_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    pred = json.loads((run_dir / "predictions.json").read_text(encoding="utf-8"))
    traces = {t["doc_id"]: t for t in _read_jsonl(run_dir / "trace.jsonl")}
    return pred, traces


def cmd_plan(args: argparse.Namespace) -> int:
    """Select the docs, require that the control run is valid (``ControlRefused`` otherwise),
    write ``plan.json``."""
    from shipdoc import paths
    from shipdoc.prompts import prompt_hash
    from shipdoc.runmeta import read_manifest
    from shipdoc.spike import load_config

    root = Path(args.repo)
    meta = [
        m
        for s in ("train", "dev")
        for m in json.loads((root / "meta" / f"{s}.json").read_text("utf-8"))
    ]
    docs = select_docs(
        meta,
        json.loads((root / "splits" / "dev100.json").read_text("utf-8")),
        json.loads((root / "splits" / "zeroshot500.json").read_text("utf-8")),
    )
    data = paths.data_dir()

    def n_pages(d: str) -> int:
        label = data / d.split("_")[0] / "labels" / f"{d}.json"
        return len(json.loads(label.read_text(encoding="utf-8"))["pages"])

    counts = {k: {"docs": len(v), "pages": sum(n_pages(d) for d in v)} for k, v in docs.items()}
    cfg = load_config(root / args.control_config)
    hint_raw = yaml.safe_load((root / args.hint_config).read_text(encoding="utf-8"))
    p_hash = prompt_hash(cfg.backend.output_format)
    for k, c in counts.items():
        print(f"{k:<7}: {c['docs']:>4} multipage docs, {c['pages']:>4} pages")
    run_dir = Path(args.control_run)
    manifest = read_manifest(run_dir)
    try:
        traces = _read_jsonl(run_dir / "trace.jsonl")
        progress = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:  # fail closed: a control that cannot be read is refused
        raise ControlRefused([f"cannot read the control run: {type(exc).__name__}"]) from exc
    why = hint_config_reasons(cfg.raw, hint_raw)
    why += control_check(
        manifest,
        progress,
        traces,
        needed=docs["pooled"],
        config_hash=cfg.config_hash,
        prompt_hash=p_hash,
        revision=cfg.backend.revision,
        batch_size=args.batch_size,
        seed=cfg.backend.seed,
        output_format=cfg.backend.output_format,
        hint_flag_off=not cfg.raw.get("header_hint", False),
    )
    print(f"control run: {run_dir}")
    print(
        f"  code SHA of the control run: {(manifest or {}).get('code_sha')} (NOT required to equal "
        "the pin); prompt hash: the manifest has no prompt field, so it is taken from the "
        "control's traces and compared with prompts.prompt_hash of the checked-out code"
    )
    print(f"  checked-out code: prompt hash {p_hash}, production config hash {cfg.config_hash}")
    print(f"  batch size {args.batch_size}, seed {cfg.backend.seed}, logprobs True")
    if why:
        print("CONTROL REFUSED, nothing is decoded: " + "; ".join(why))
        raise ControlRefused(why)
    print("CONTROL ACCEPTED: header_hint off reproduces the control's prompt hash + config hash")
    plan = {
        "docs": docs,
        "counts": counts,
        "control_config_hash": cfg.config_hash,
        "control_prompt_hash": p_hash,
        "control_run": str(run_dir),
        "control_code_sha": (manifest or {}).get("code_sha"),
        "control_batch_size": args.batch_size,
        "control_sources": dict.fromkeys(docs["pooled"], str(run_dir)),
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps(plan, indent=1), encoding="utf-8")
    for name in ("pooled", "dev", "train"):
        (out / f"docs_{name}.json").write_text(json.dumps(docs[name]), encoding="utf-8")
    pooled = counts["pooled"]
    print(
        f"control arm: the 02 run restricted to {pooled['docs']} docs ({pooled['pages']} pages), "
        f"no rerun; hint arm: the same {pooled['docs']} docs (whole docs)"
    )
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare the hint run with the control (the 02 run), print and save the report."""
    from shipdoc import eval as ev
    from shipdoc import paths

    out = Path(args.out_dir)
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    pooled = plan["docs"]["pooled"]
    ctrl_pred: dict[str, Any] = {}
    ctrl_tr: dict[str, dict[str, Any]] = {}
    sources = dict(plan["control_sources"])
    for run in sorted(set(sources.values())):
        pred, traces = _load_run(Path(run))
        for d, src in sources.items():
            if src == run and d in pred and d in traces:
                ctrl_pred[d], ctrl_tr[d] = pred[d], traces[d]
    hint_pred, hint_tr = _load_run(Path(args.hint_run_dir))
    data = paths.data_dir()
    gold: dict[str, Any] = {}
    for split in ("train", "dev"):
        if (data / split / "labels").is_dir():
            gold.update(ev.load_gold(data / split / "labels"))
    no_gold = [d for d in pooled if d not in gold]
    if no_gold:
        raise ValueError(f"{len(no_gold)} pooled docs have no gold label, e.g. {no_gold[:3]}")
    subsets = {"pooled": pooled, "dev": plan["docs"]["dev"], "train": plan["docs"]["train"]}
    missing = [d for d in pooled if d not in ctrl_pred or d not in hint_pred]
    if missing:
        raise ValueError(f"{len(missing)} pooled docs missing in an arm, e.g. {missing[:3]}")
    # printed BEFORE any analysis: what differs where the hint cannot act
    print(format_noise_floor(noise_floor(ctrl_tr, hint_tr, pooled)), flush=True)
    res = compare_runs(ctrl_pred, ctrl_tr, hint_pred, hint_tr, gold, subsets)
    res["control_note"] = CONTROL_NOTE
    res["control_sources"] = {d: Path(s).name for d, s in sources.items()}
    text = format_report(res)
    print(text)
    (out / "hdrhint_compare.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (out / "hdrhint_compare.md").write_text("```\n" + text + "\n```\n", encoding="utf-8")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m shipdoc.headerhint plan|compare``."""
    ap = argparse.ArgumentParser(prog="shipdoc.headerhint", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="doc subsets, counts, control reuse decision")
    p.add_argument("--repo", default=".")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--control-config", default="configs/spike_qwen35_4b_img_only.yaml")
    p.add_argument("--hint-config", default="configs/spike_qwen35_4b_img_only_keyed_hdrhint.yaml")
    p.add_argument("--control-run", required=True, help="the 02 run folder (the control)")
    p.add_argument("--batch-size", type=int, required=True)
    p.set_defaults(fn=cmd_plan)
    c = sub.add_parser("compare", help="paired comparison and the decision")
    c.add_argument("--out-dir", required=True)
    c.add_argument("--hint-run-dir", required=True)
    c.set_defaults(fn=cmd_compare)
    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
