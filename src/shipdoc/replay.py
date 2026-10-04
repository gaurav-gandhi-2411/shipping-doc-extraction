"""Replay parse -> merge -> normalize -> score on the SAVED raw model outputs of a spike run.

No model is called: every page's ``raw_text`` from ``trace.jsonl`` goes back through the current
parser, merge and normalizer, with post-processing fixes switched on or off by flags, and the result
is scored with the official scorer on the run's own doc_ids. Every number it prints is a "replay on
saved outputs".

What replay CANNOT fix: anything the model decided at decode time. A value the model emitted as
null (or as a wrong value, or in the wrong field) is already baked into ``raw_text``; only changes
to how that text is parsed, merged or normalized can move the score. Prompt, schema and key-order
changes need a new GPU run.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shipdoc import eval as ev
from shipdoc import paths
from shipdoc.extract import COMPACT_TOP, parse_output
from shipdoc.merge import FieldProvenance, merge_pages
from shipdoc.normalize import normalize_doc
from shipdoc.spike import load_config

LABEL = "replay on saved outputs"


@dataclass(frozen=True)
class Fixes:
    """Post-processing switches. ``drop_null_rows=None`` keeps the merge default."""

    drop_null_rows: bool | None = None
    salvage_truncated: bool = False


def salvage_page_json(raw: str) -> dict[str, Any] | None:
    """Rebuild a page cut off at ``max_new_tokens`` from its last COMPLETE container.

    Scans the text (string/escape aware), remembers every position just after a ``}`` or ``]`` that
    closes a nested container, and returns the parse of ``raw[:pos]`` plus the closers still open,
    taking the last position that parses. So a half-written row or value is dropped whole and no
    value is ever completed or altered: the result holds exactly the complete rows/fields the model
    emitted. Returns None when nothing is recoverable or the text is malformed beyond truncation.
    """
    stack: list[str] = []
    cuts: list[tuple[int, str]] = []
    in_str = esc = False
    for i, ch in enumerate(raw):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not stack or stack.pop() != ch:
                return None
            if stack:
                cuts.append((i + 1, "".join(reversed(stack))))
    for pos, closers in reversed(cuts):
        try:
            obj = json.loads(raw[:pos] + closers)
        except json.JSONDecodeError:
            continue
        return obj if isinstance(obj, dict) else None
    return None


def parse_page(raw: str, output_format: str, fixes: Fixes) -> dict[str, Any] | None:
    """`parse_output`, then (with ``salvage_truncated``) the salvage path for a failed parse."""
    parsed = parse_output(raw, output_format)
    if parsed is not None or not fixes.salvage_truncated:
        return parsed
    obj = salvage_page_json(raw)
    if obj is None:
        return None
    if output_format == "compact":
        # expand_page needs every top-level key; a cut page lacks the tail ones (page kind).
        for short in COMPACT_TOP.values():
            obj.setdefault(short, {} if short == "h" else [] if short == "r" else None)
        return parse_output(json.dumps(obj), output_format)
    return obj


def replay_doc(
    trace: dict[str, Any],
    output_format: str,
    fixes: Fixes,
    provenance: FieldProvenance | None = None,
) -> dict[str, Any]:
    """One trace line -> the prediction the pipeline would produce today from the saved raw text."""
    pages = [parse_page(p["raw_text"], output_format, fixes) for p in trace["pages"]]
    kwargs = {} if fixes.drop_null_rows is None else {"drop_null_rows": fixes.drop_null_rows}
    merged = merge_pages(pages, provenance, **kwargs)
    doc, _ = normalize_doc(merged.doc)
    return doc


def read_trace(path: Path) -> list[dict[str, Any]]:
    """All trace lines of a run."""
    text = path.read_text(encoding="utf-8")
    return [json.loads(ln) for ln in text.splitlines() if ln.strip()]


def replay_traces(
    traces: Sequence[dict[str, Any]],
    gold: dict[str, dict[str, Any]],
    output_format: str,
    fixes: Fixes,
) -> dict[str, Any]:
    """Score the replayed predictions of `traces` against `gold` (restricted to their doc_ids)."""
    prov = FieldProvenance.load()
    pred = {t["doc_id"]: replay_doc(t, output_format, fixes, prov) for t in traces}
    gold = {d: g for d, g in gold.items() if d in pred}
    agg = ev.score(pred, gold)["all"]
    keys = (
        "documents",
        "OVERALL",
        "header_field_accuracy",
        "row_precision",
        "row_recall",
        "row_f1",
        "documents_fully_correct",
        "false_fill_rate",
        "header_by_field",
        "line_item_field_accuracy",
    )
    return {k: agg[k] for k in keys}


def replay_run(run_dir: Path, config: Path, labels: Path, fixes: Fixes) -> dict[str, Any]:
    """Replay a run dir; also reports the run's saved ``metrics.json`` OVERALL when present."""
    cfg = load_config(config)
    traces = read_trace(Path(run_dir) / "trace.jsonl")
    result = replay_traces(traces, ev.load_gold(labels), cfg.backend.output_format, fixes)
    result["label"] = LABEL
    result["fixes"] = {
        "drop_null_rows": fixes.drop_null_rows,
        "salvage_truncated": fixes.salvage_truncated,
    }
    metrics = Path(run_dir) / "metrics.json"
    if metrics.is_file():
        result["saved_OVERALL"] = json.loads(metrics.read_text(encoding="utf-8")).get("OVERALL")
    return result


def main_replay(args: Any) -> int:
    """CLI glue for ``python -m shipdoc replay`` (see cli.py for the argument list)."""
    labels = args.labels or paths.data_dir() / args.split / "labels"
    fixes = Fixes(drop_null_rows=args.drop_null_rows, salvage_truncated=args.salvage_truncated)
    res = replay_run(args.run_dir, args.config, labels, fixes)
    saved = res.get("saved_OVERALL")
    print(f"[{LABEL}] {Path(args.run_dir).name}  fixes={res['fixes']}")
    print(
        f"  docs {res['documents']}  OVERALL {100 * res['OVERALL']:.2f}"
        + (f"  (saved run: {100 * saved:.2f})" if saved is not None else "")
    )
    print(
        f"  header {100 * res['header_field_accuracy']:.2f}%  row P/R/F1 "
        f"{100 * res['row_precision']:.2f}/{100 * res['row_recall']:.2f}/{100 * res['row_f1']:.2f}"
        f"  fully correct {100 * res['documents_fully_correct']:.2f}%"
    )
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0
