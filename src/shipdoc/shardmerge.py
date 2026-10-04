"""Merge the K shard folders of a sharded run into the folder of the unsharded run id.

Inputs: ``<runs>/<run_id>_shard<i>of<K>/`` for i in 0..K-1 (written by ``shipdoc spike --shard``).
Output: ``<runs>/<run_id>/`` with the same ``trace.jsonl`` / ``predictions.json`` /
``metrics.json`` / ``progress.json`` an unsharded run writes: documents in the order of the doc
list, metrics recomputed by the same code (`spike.build_metrics`) over the merged traces. With a
deterministic backend (latency pinned) the three files are byte-identical to an unsharded run's
(tests/test_shards.py). A ``manifest.json`` (``shard`` = ``merged``) and a ``sessions.json``
(the shards' sessions, each tagged with its shard) are added.

Refuses (ValueError, nothing written) unless: every shard folder exists, finished
(``progress.json`` status ``complete``) and has a manifest; code SHA, config hash, model
revision, seed, batch size, split, arm, output format, logprobs flag and the full doc-list hash
agree across shards; each manifest names its own shard; no document appears twice; the merged
documents are exactly the doc list; the document and page counts equal the expected ones.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from shipdoc import paths
from shipdoc.coerce import coerce_predictions, repair_predictions
from shipdoc.prompts import prompt_hash
from shipdoc.runmeta import docs_sha, read_manifest, write_manifest
from shipdoc.shard import shard_run_id
from shipdoc.spike import (
    SpikeConfig,
    _read_trace,
    atomic_write,
    build_metrics,
    doc_page_images,
    doc_splits,
    load_config,
    load_doc_ids,
    load_gold_and_meta,
    split_names,
)

#: Manifest fields that must be identical in every shard.
SAME_KEYS = (
    ("code_sha",),
    ("config", "hash"),
    ("model", "revision"),
    ("model", "id"),
    ("seed",),
    ("batch_size",),
    ("split",),
    ("arm",),
    ("output_format",),
    ("logprobs",),
    ("docs_sha",),
    ("base_run_id",),
)


def _get(m: dict[str, Any], path: tuple[str, ...]) -> Any:
    cur: Any = m
    for k in path:
        cur = cur.get(k) if isinstance(cur, dict) else None
    return cur


def merge_shards(
    cfg: SpikeConfig,
    doc_ids: Sequence[str],
    split: str,
    run_id: str,
    k: int,
    runs_root: Path | None = None,
    data_root: Path | None = None,
    expected_docs: int | None = None,
    expected_pages: int | None = None,
) -> dict[str, Any]:
    """Merge the shards of `run_id`; returns the merged metrics (also written to metrics.json)."""
    runs = paths.runs_dir() if runs_root is None else Path(runs_root)
    root = paths.data_dir() if data_root is None else Path(data_root)
    ids = list(doc_ids)
    dirs = [runs / shard_run_id(run_id, f"{i}/{k}") for i in range(k)]
    manifests: list[dict[str, Any]] = []
    for i, d in enumerate(dirs):
        m = read_manifest(d)
        if m is None:
            raise ValueError(f"{d}: no manifest.json (shard {i}/{k} missing or unfinished)")
        prog_path = d / "progress.json"
        status = (
            json.loads(prog_path.read_text(encoding="utf-8")).get("status")
            if (prog_path.is_file())
            else None
        )
        if status != "complete":
            raise ValueError(f"{d}: shard {i}/{k} is not complete (progress status {status!r})")
        if m.get("shard") != f"{i}/{k}":
            raise ValueError(f"{d}: manifest says shard {m.get('shard')!r}, expected {i}/{k}")
        manifests.append(m)
    for path in SAME_KEYS:
        values = {json.dumps(_get(m, path)) for m in manifests}
        if len(values) != 1:
            raise ValueError(
                f"shards disagree on {'.'.join(path)}: "
                f"{[_get(m, path) for m in manifests]}; refusing to merge"
            )
    if manifests[0]["docs_sha"] != docs_sha(ids):
        raise ValueError("the doc list differs from the one the shards were run on (docs_sha)")
    if manifests[0]["config"]["hash"] != cfg.config_hash:
        raise ValueError("the config differs from the one the shards were run with (config hash)")
    traces: dict[str, dict[str, Any]] = {}
    for d in dirs:
        for t in _read_trace(d / "trace.jsonl"):
            if t["doc_id"] in traces:
                raise ValueError(f"duplicate document {t['doc_id']} across shards")
            traces[t["doc_id"]] = t
    missing = [d for d in ids if d not in traces]
    extra = sorted(set(traces) - set(ids))
    if missing or extra:
        raise ValueError(
            f"documents do not match the doc list: {len(missing)} missing {missing[:5]}, "
            f"{len(extra)} unexpected {extra[:5]}"
        )
    want_docs = len(ids) if expected_docs is None else expected_docs
    if len(traces) != want_docs:
        raise ValueError(f"{len(traces)} documents merged, expected {want_docs}")
    n_pages = sum(len(t["pages"]) for t in traces.values())
    if expected_pages is None:
        split_of = doc_splits(ids, split_names(split), root)
        expected_pages = sum(len(doc_page_images(split_of[d], d, root)) for d in ids)
    if n_pages != expected_pages:
        raise ValueError(f"{n_pages} pages merged, expected {expected_pages}")

    final = [traces[d] for d in ids]
    out_dir = runs / run_id
    gold, slice_meta = load_gold_and_meta(split_names(split), root)
    m0 = manifests[0]
    metrics = build_metrics(
        cfg, final, gold, slice_meta, split_names(split), split, run_id, m0["model"],
        m0["code_sha"], prompt_hash(cfg.backend.output_format),
    )  # fmt: skip
    # LF like append_jsonl (atomic_write would write CRLF on Windows and break byte-identity)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "trace.jsonl.tmp"
    tmp.write_text("".join(json.dumps(t) + "\n" for t in final), encoding="utf-8", newline="\n")
    os.replace(tmp, out_dir / "trace.jsonl")
    # the same choke point as run_spike: plain-number strings, whatever a shard trace holds, and
    # the schema-invalid date repair (a no-op on traces that already carry it)
    merged_preds, late_repairs = repair_predictions(
        coerce_predictions({t["doc_id"]: t["prediction"] for t in final})
    )
    atomic_write(out_dir / "predictions.json", json.dumps(merged_preds, indent=1))
    atomic_write(out_dir / "metrics.json", json.dumps(metrics, indent=1))
    atomic_write(
        out_dir / "progress.json",
        json.dumps(
            {
                "status": "complete",
                "done": len(final),
                "total": len(final),
                "last_doc": final[-1]["doc_id"] if final else None,
                "run_id": run_id,
            },
            indent=1,
        ),
    )
    sessions: list[dict[str, Any]] = []
    for i, d in enumerate(dirs):
        p = d / "sessions.json"
        if p.is_file():
            sessions += [{**s, "shard": f"{i}/{k}"} for s in json.loads(p.read_text("utf-8"))]
    if sessions:
        atomic_write(out_dir / "sessions.json", json.dumps(sessions, indent=1))
    write_manifest(
        out_dir,
        {
            **m0,
            "run_id": run_id,
            "shard": "merged",
            "merged_from": [d.name for d in dirs],
            "n_docs": len(final),
            "schema_repairs": {
                "n": sum(len(t.get("schema_repairs") or []) for t in final) + len(late_repairs),
                "reason": "schema_invalid_date",
            },
        },
    )
    return metrics


def main_merge(args: Any) -> int:
    """CLI glue for ``python -m shipdoc merge-shards`` (see cli.py)."""
    cfg = load_config(args.config)
    metrics = merge_shards(
        cfg,
        load_doc_ids(args.docs),
        args.split,
        args.run_id,
        args.shards,
        expected_docs=args.expected_docs,
        expected_pages=args.expected_pages,
    )
    print(
        f"merged {args.shards} shards into {args.run_id}: {metrics['n_docs']} docs, "
        f"{metrics['n_pages']} pages"
    )
    return 0
