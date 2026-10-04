"""Zero-shot VLM spike runner (spec Phase 2.3).

Pipeline per document: per-page extract -> merge -> normalize -> one trace line + one prediction.
Outputs land in ``<SHIPDOC_RUNS_DIR>/<run_id>/``: ``predictions.json``, ``trace.jsonl``,
``metrics.json``, ``progress.json``. After each document ``trace.jsonl`` gets ONE appended line
(fsync'd; a torn last line from a kill is dropped on resume) and the other files are rewritten
atomically, so a Colab disconnect never leaves a half-written file and ``--resume`` skips doc_ids
already traced.

``--split train+dev`` runs documents of several splits as ONE run (each doc is looked up in its own
split; gold and meta are the union; ``metrics.json`` gains a ``per_split`` block). ``--logprobs``
adds per-field token logprobs to every trace page (`shipdoc.logprobs`, keyed format only).

Scoring uses the official scorer through `shipdoc.eval`; CIs are doc-level percentile bootstraps
(2,000 resamples, seed 42). The ``scanned`` slice is reported explicitly (test is 47.1% scanned vs
38.1% in train/dev, recon section 1).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from shipdoc import eval as ev
from shipdoc import headerhint, ocr, paths
from shipdoc.batching import (
    PageJob,
    doc_windows,
    next_smaller_batch,
    plan_batches,
    split_for_fallback,
)
from shipdoc.coerce import (
    coerce_doc,
    coerce_predictions,
    page_for_merge,
    repair_doc,
    repair_predictions,
)
from shipdoc.extract import (
    OUTPUT_FORMATS,
    BackendConfig,
    HfBackend,
    MockBackend,
    PageRequest,
    VlmBackend,
    schema_errors,
    schema_for,
    seed_everything,
)
from shipdoc.logprobs import field_logprobs
from shipdoc.merge import FieldProvenance, merge_pages
from shipdoc.normalize import normalize_doc
from shipdoc.prompts import PROMPT_VERSION, build_prompt, prompt_hash
from shipdoc.runmeta import check_resume, docs_sha, read_manifest, write_manifest
from shipdoc.shard import assign_shards, parse_shard, shard_run_id

ARMS = ("img_only", "img_ocr")
WANDB_PROJECT = "shipdoc-extract-debug"
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 42
SCORE_KEYS = (
    "documents",
    "OVERALL",
    "header_field_accuracy",
    "row_f1",
    "documents_fully_correct",
    "false_fill_rate",
    "illegible_fields",
)
_SHA = re.compile(r"^[0-9a-f]{40}$")
MANIFEST_SCHEMA = 1


@dataclass(frozen=True)
class SpikeConfig:
    """A parsed ``configs/spike_<model>_<arm>.yaml``."""

    name: str
    arm: str
    backend: BackendConfig
    prompt_version: str
    raw: dict[str, Any]

    @property
    def config_hash(self) -> str:
        """sha256 (first 16 hex) of the canonical JSON of the whole YAML mapping."""
        blob = json.dumps(self.raw, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def load_config(path: str | os.PathLike[str]) -> SpikeConfig:
    """Parse and validate a spike YAML; raises ValueError naming the offending key."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    missing = [
        k
        for k in ("name", "arm", "model", "max_pixels", "max_new_tokens", "prompt_version", "seed")
        if k not in raw
    ]
    if missing:
        raise ValueError(f"{path}: missing keys {missing}")
    if raw["arm"] not in ARMS:
        raise ValueError(f"{path}: arm must be one of {ARMS}, got {raw['arm']!r}")
    if raw["prompt_version"] != PROMPT_VERSION:
        raise ValueError(
            f"{path}: prompt_version {raw['prompt_version']!r} != code PROMPT_VERSION "
            f"{PROMPT_VERSION!r}; update the config or the prompt deliberately"
        )
    fmt = raw.get("output_format", "json")
    if fmt not in OUTPUT_FORMATS:
        raise ValueError(f"{path}: output_format must be one of {OUTPUT_FORMATS}, got {fmt!r}")
    m = raw["model"]
    if not _SHA.match(str(m.get("revision", ""))):
        raise ValueError(f"{path}: model.revision must be a 40-hex commit sha (pinned)")
    backend = BackendConfig(
        key=m["key"],
        repo=m["repo"],
        revision=m["revision"],
        adapter=m["adapter"],
        quant=m.get("quant", "none"),
        dtype=m.get("dtype", "float16"),
        attn_implementation=m.get("attn_implementation"),
        trust_remote_code=bool(m.get("trust_remote_code", False)),
        max_pixels=int(raw["max_pixels"]),
        max_new_tokens=int(raw["max_new_tokens"]),
        output_format=fmt,
        ocr_token_budget=int(raw.get("ocr_token_budget", 1200)),
        seed=int(raw["seed"]),
    )
    return SpikeConfig(raw["name"], raw["arm"], backend, raw["prompt_version"], raw)


# --------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------


def atomic_write(path: Path, text: str) -> None:
    """Write `text` to `path` via a temp file in the same folder and ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def git_commit() -> str:
    """HEAD sha (``+dirty`` when the tree has changes), or ``unknown`` outside a git checkout."""
    root = paths.REPO_ROOT
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return sha + ("+dirty" if dirty else "")


def doc_page_images(split: str, doc_id: str, data_root: Path | None = None) -> list[Path]:
    """Page images of a doc in page order (``<doc_id>_p<n>.<ext>``)."""
    root = paths.data_dir() if data_root is None else Path(data_root)
    found = list((root / split / "images").glob(f"{doc_id}_p*.*"))
    found = [p for p in found if re.fullmatch(rf"{re.escape(doc_id)}_p\d+", p.stem)]
    if not found:
        raise FileNotFoundError(f"no page images for {doc_id} under {root / split / 'images'}")
    return sorted(found, key=lambda p: int(p.stem.rsplit("_p", 1)[1]))


def load_doc_ids(arg: str) -> list[str]:
    """``--docs`` value: a path to a JSON list file, or an inline JSON list."""
    text = arg if arg.lstrip().startswith("[") else Path(arg).read_text(encoding="utf-8")
    ids = json.loads(text)
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        raise ValueError("--docs must be a JSON list of doc_id strings")
    if len(set(ids)) != len(ids):
        raise ValueError("--docs contains duplicate doc_ids")
    return ids


def _read_trace(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _load_trace_for_resume(path: Path) -> list[dict[str, Any]]:
    """`_read_trace`, but a torn LAST line (the process was killed mid-append) is dropped.

    The file is then rewritten atomically with the intact lines only, so the next append starts on a
    clean line. A corrupt line anywhere else still raises: that is not a kill artefact.
    """
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    traces: list[dict[str, Any]] = []
    torn = bool(text) and not text.endswith("\n")  # a complete line missing only its newline
    for i, ln in enumerate(lines):
        try:
            traces.append(json.loads(ln))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
            print(f"{path}: dropped a torn last line (killed mid-write)", file=sys.stderr)
            torn = True
    if torn:
        atomic_write(path, "".join(json.dumps(t) + "\n" for t in traces))
    return traces


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON line and fsync: a kill loses at most the line being written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


SPLIT_SEP = "+"


def split_names(split: str) -> list[str]:
    """``"dev"`` -> ``["dev"]``; ``"train+dev"`` -> ``["train", "dev"]`` (no empty or repeated)."""
    names = split.split(SPLIT_SEP)
    if not all(names) or len(set(names)) != len(names):
        raise ValueError(f"bad --split {split!r}: use one name or names joined by {SPLIT_SEP!r}")
    return names


def doc_splits(ids: Sequence[str], splits: Sequence[str], root: Path) -> dict[str, str]:
    """doc_id -> the split whose ``images/`` folder holds it (one split: that split, unchecked)."""
    if len(splits) == 1:
        return dict.fromkeys(ids, splits[0])
    found: dict[str, str] = {}
    for s in splits:
        for p in (root / s / "images").glob("*_p*.*"):
            doc = p.stem.rsplit("_p", 1)[0]
            if found.setdefault(doc, s) != s:
                raise ValueError(f"doc {doc} has images in both {found[doc]} and {s}")
    missing = [d for d in ids if d not in found]
    if missing:
        raise FileNotFoundError(
            f"no page images in {list(splits)} for {len(missing)} docs: {missing[:5]}"
        )
    return {d: found[d] for d in ids}


# --------------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------------


def compute_metrics(
    traces: Sequence[dict[str, Any]],
    gold: dict[str, dict[str, Any]] | None,
    slice_meta: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Run-level metrics from the trace lines (and gold, when the split has labels)."""
    pages = [p for t in traces for p in t["pages"]]
    lat = [p["meta"]["latency_s"] for p in pages if p["meta"].get("latency_s") is not None]
    vram = [p["meta"]["peak_vram_bytes"] for p in pages if p["meta"].get("peak_vram_bytes")]
    out: dict[str, Any] = {
        "n_docs": len(traces),
        "n_pages": len(pages),
        "json_validity_rate": (sum(p["json_valid"] for p in pages) / len(pages)) if pages else None,
        "schema_validity_rate": (
            sum(not p["schema_errors"] and p["json_valid"] for p in pages) / len(pages)
        )
        if pages
        else None,
        "s_per_page_mean": float(np.mean(lat)) if lat else None,
        "s_per_page_p95": float(np.percentile(lat, 95)) if lat else None,
        "peak_vram_max_bytes": max(vram) if vram else None,
        "n_visual_tokens_mean": _mean([p["meta"].get("n_visual_tokens") for p in pages]),
        "n_input_tokens_mean": _mean([p["meta"].get("n_input_tokens") for p in pages]),
        "n_output_tokens_mean": _mean([p["meta"].get("n_output_tokens") for p in pages]),
    }
    if any("ocr_truncated" in p["meta"] for p in pages):
        out["ocr_truncation_rate"] = _mean([p["meta"].get("ocr_truncated") for p in pages])
    if gold is None:
        out["scored"] = False
        return out
    pred = {t["doc_id"]: t["prediction"] for t in traces}
    gold = {d: g for d, g in gold.items() if d in pred}
    meta = [m for m in (slice_meta or []) if m["doc_id"] in gold]
    per = ev.per_doc_results(pred, gold)
    report = ev.score(pred, gold, meta or None, group_slices=False)
    cis: dict[str, Any] = {}
    for name, ids in ev.slice_doc_ids(gold, meta or None, group_slices=False).items():
        if ids:
            cis[name] = ev.bootstrap_metrics(
                [per[d] for d in ids], n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED
            )
    out["scored"] = True
    out["slices"] = {
        name: {
            **{k: agg[k] for k in SCORE_KEYS if k in agg},
            "ci95": cis.get(name, {}),
        }
        for name, agg in report.items()
    }
    out["OVERALL"] = report["all"]["OVERALL"]
    out["OVERALL_ci95"] = cis["all"]["OVERALL"]
    out["false_fill_rate"] = report["all"]["false_fill_rate"]
    # The test split is 47.1% scanned vs 38.1% in train/dev: always show both sides.
    out["scanned"] = out["slices"].get("scanned=yes")
    out["digital"] = out["slices"].get("scanned=no")
    return out


def per_split_metrics(
    traces: Sequence[dict[str, Any]],
    gold: dict[str, dict[str, Any]],
    slice_meta: list[dict[str, Any]] | None,
    splits: Sequence[str],
) -> dict[str, Any]:
    """Headline numbers of each split's docs (official scorer, doc-level bootstrap CI)."""
    out: dict[str, Any] = {}
    for s in splits:
        sub = [t for t in traces if t["split"] == s]
        if not sub:
            continue
        m = compute_metrics(sub, gold, slice_meta)
        if not m.get("scored"):
            continue
        agg = m["slices"]["all"]
        out[s] = {
            "documents": agg["documents"],
            "n_pages": m["n_pages"],
            "OVERALL": m["OVERALL"],
            "OVERALL_ci95": m["OVERALL_ci95"],
            "header_field_accuracy": agg["header_field_accuracy"],
            "row_f1": agg["row_f1"],
            "false_fill_rate": agg["false_fill_rate"],
            "json_validity_rate": m["json_validity_rate"],
        }
    return out


def load_gold_and_meta(
    splits: Sequence[str], root: Path
) -> tuple[dict[str, dict[str, Any]] | None, list[dict[str, Any]] | None]:
    """Gold labels (None when no split has any) and slice meta (None when absent) of `splits`."""
    gold_parts = [
        ev.load_gold(root / s / "labels")
        for s in splits
        if (root / s / "labels").is_dir() and any((root / s / "labels").glob("*.json"))
    ]
    gold = {d: g for part in gold_parts for d, g in part.items()} if gold_parts else None
    slice_meta = [
        m
        for s in splits
        if (paths.REPO_ROOT / "meta" / f"{s}.json").is_file()
        for m in ev.load_json(paths.REPO_ROOT / "meta" / f"{s}.json")
    ] or None
    return gold, slice_meta


def build_metrics(
    cfg: SpikeConfig,
    final: Sequence[dict[str, Any]],
    gold: dict[str, dict[str, Any]] | None,
    slice_meta: list[dict[str, Any]] | None,
    splits: Sequence[str],
    split: str,
    run_id: str,
    model: dict[str, Any],
    commit: str,
    p_hash: str,
) -> dict[str, Any]:
    """The contents of metrics.json for the traces `final` (shared with the shard merge)."""
    metrics = compute_metrics(final, gold, slice_meta)
    if len(splits) > 1 and gold is not None:
        metrics["per_split"] = per_split_metrics(final, gold, slice_meta, splits)
    metrics.update(
        run_id=run_id,
        config=cfg.name,
        config_hash=cfg.config_hash,
        model=model,
        prompt={"version": PROMPT_VERSION, "hash": p_hash},
        git_commit=commit,
        split=split,
        arm=cfg.arm,
        output_format=cfg.backend.output_format,
    )
    return metrics


def _mean(values: Sequence[Any]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def _log_wandb(metrics: dict[str, Any], cfg: SpikeConfig, run_id: str) -> None:
    """Metrics + config only (no images, no extracted values) to the private debug project."""
    import wandb

    run = wandb.init(
        project=WANDB_PROJECT,
        entity=os.environ.get("WANDB_ENTITY"),
        name=run_id,
        config={
            **cfg.raw,
            "config_hash": cfg.config_hash,
            "prompt_hash": prompt_hash(cfg.backend.output_format),
        },
    )
    flat: dict[str, float] = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            flat[k] = v
    for name, s in (metrics.get("slices") or {}).items():
        for k in ("OVERALL", "header_field_accuracy", "row_f1", "false_fill_rate", "documents"):
            if k in s:
                flat[f"{name}/{k}"] = s[k]
    run.log(flat)
    run.finish()


# --------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------


def _apply_post_rules(
    doc: dict[str, Any],
    doc_id: str,
    page_records: Sequence[dict[str, Any]],
    rule_cfg: Any,
    ocr_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Post-processing v1 on one merged doc: (patched doc, the trace's value-free ``rules``).

    Lazy import: ``postrules`` imports ``replay`` which imports this module. Shapes are the
    frozen ``meta/slot_shapes.json`` (read once; an unusable file raises when R3 is on).
    """
    from shipdoc import postrules

    shapes = postrules.default_shapes() if rule_cfg.r3 else None
    text, why = (None, None)
    if rule_cfg.r2 and doc.get("doc_type") == "waybill":
        text, why = postrules.ocr_text_for_doc(doc_id, len(page_records), ocr_root)
    out, changes, skipped = postrules.apply_rules(
        doc,
        supplier_name=postrules.supplier_from_trace({"pages": page_records}),
        ocr_text=text,
        shapes=shapes,
        cfg=rule_cfg,
        ocr_reason=why,
    )
    return out, {"changes": changes, "skipped": skipped}


def run_spike(
    cfg: SpikeConfig,
    doc_ids: Sequence[str],
    split: str,
    run_id: str,
    backend: VlmBackend,
    resume: bool = False,
    limit: int | None = None,
    use_wandb: bool = False,
    runs_root: Path | None = None,
    data_root: Path | None = None,
    logprobs: bool = False,
    batch_size: int | None = None,
    shard: str = "0/1",
    base_run_id: str | None = None,
    bench_info: dict[str, Any] | None = None,
    fallback: bool = True,
    replay_batches: Sequence[Sequence[tuple[str, int]]] | None = None,
    rule_cfg: Any | None = None,
) -> dict[str, Any]:
    """Run the pipeline over `doc_ids`; returns the metrics dict (also written to metrics.json).

    `rule_cfg` (a ``postrules.RuleConfig``) opts the run's writer into post-processing v1: the
    rules R1 / R2 / R3 run on each merged + normalised document before coercion and each trace
    gets a value-free ``rules`` record. The default None keeps the model run exactly as it was
    (traces, predictions and the baseline the rule gate compares against are unchanged); the
    submission path applies the rules at ``predict assemble`` instead (default ON there).

    `split` may join several splits with ``+`` (``train+dev``). `logprobs` makes the backend
    capture per-token logprobs and stores ``field_logprobs`` / ``token_logprobs`` on every page.

    `batch_size` is the number of pages per generate call. 1 (the default for a new run) is the
    original unbatched path, untouched; None on a resume means "the stored size" (`runmeta`). Above
    1, the pages of a window of whole documents (`batching.doc_windows`) are sorted by the
    expected-length key and run in batches; a document is traced only when ALL its pages are done,
    documents leave in INPUT order (so trace.jsonl and predictions.json are ordered exactly like an
    unbatched run) and a kill loses at most one window. If a batch raises (OOM or anything else)
    and `fallback` is on, it is rerun in chunks of the next smaller size down to 1; the reason is
    recorded in each page's ``meta.batch_fallback``. `shard` ``i/K`` runs the i-th document-level
    shard (`shipdoc.shard`); `bench_info` is recorded in the manifest.

    Every generate call with SEVERAL pages stamps them with ``meta.exec_batch = {"id", "members"}``:
    the ordered ``[doc_id, page_idx]`` list of the call that decoded them (after any fallback
    split) and an id (the first member, ``doc_id:page_idx``), which is what a determinism re-run
    needs to rebuild the same batch composition. A page decoded alone carries no stamp (the batch
    size 1 path stays byte for byte what it was); the manifest says ``exec_batch_recorded``.

    `replay_batches` (the determinism pass) runs exactly those batches, in that member order, and
    nothing else: `doc_ids` are the documents they touch, only documents whose EVERY page is in
    a replayed batch get a trace, and the pages of the other documents are appended to
    ``replay_pages.jsonl`` (``doc_id``, ``page``, ``raw_text``, ``exec_batch``). Pair it with
    ``fallback=False`` so a replay never silently changes the composition.
    """
    seed_everything(cfg.backend.seed)
    if logprobs and cfg.backend.output_format != "json":
        raise ValueError("--logprobs supports the keyed (json) output format only")
    if batch_size is not None and batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    shard_i, shard_k = parse_shard(shard)
    out_dir = (paths.runs_dir() if runs_root is None else Path(runs_root)) / run_id
    trace_path, pred_path = out_dir / "trace.jsonl", out_dir / "predictions.json"
    traces = _load_trace_for_resume(trace_path)
    if traces and not resume:
        raise FileExistsError(
            f"{trace_path} already has {len(traces)} docs; pass --resume to continue or use a new "
            "--run-id"
        )
    root = paths.data_dir() if data_root is None else Path(data_root)
    splits = split_names(split)
    all_ids = list(doc_ids)[:limit] if limit else list(doc_ids)
    split_of = doc_splits(all_ids, splits, root)
    ids = all_ids
    if shard_k > 1:
        sizes = [(d, len(doc_page_images(split_of[d], d, root))) for d in all_ids]
        ids = assign_shards(sizes, shard_k)[shard_i]
    replay_images: dict[str, list[Path]] = {}
    if replay_batches is not None:
        replay_images = {d: doc_page_images(split_of[d], d, root) for d in all_ids}
        covered = {(d, i) for b in replay_batches for d, i in b}
        ids = [d for d in all_ids if all((d, i) in covered for i in range(len(replay_images[d])))]
    commit = git_commit()
    expected_manifest = {
        "schema": MANIFEST_SCHEMA,
        "run_id": run_id,
        "base_run_id": base_run_id or run_id,
        "code_sha": commit,
        "model": {"id": backend.model_id, "revision": backend.revision},
        "config": {"name": cfg.name, "hash": cfg.config_hash},
        "seed": cfg.backend.seed,
        "batch_size": batch_size,
        "batch_size_source": "default" if batch_size is None else "manual",
        "bench": bench_info,
        "split": split,
        "arm": cfg.arm,
        "output_format": cfg.backend.output_format,
        "logprobs": logprobs,
        "shard": shard,
        "docs_sha": docs_sha(all_ids),
        "n_docs": len(ids),
    }
    if replay_batches is not None:
        expected_manifest["replay"] = {
            "n_batches": len(replay_batches),
            "members_sha": docs_sha([f"{d}:{i}" for b in replay_batches for d, i in b]),
        }
    stored_manifest = read_manifest(out_dir) if resume else None
    manifest = check_resume(stored_manifest, expected_manifest, has_trace=bool(traces))
    # a run resumed from a trace written before the stamp existed has unstamped batched pages
    manifest["exec_batch_recorded"] = (
        stored_manifest is None or not traces or stored_manifest.get("exec_batch_recorded") is True
    )
    if bench_info is not None and manifest["batch_size_source"] != "bench":
        manifest["batch_size_source"] = "bench"
    batch_size = int(manifest["batch_size"])
    if batch_size > 1 and not hasattr(backend, "extract_pages"):
        raise ValueError(f"backend {type(backend).__name__} has no extract_pages (batch_size > 1)")
    write_manifest(out_dir, manifest)
    done = {t["doc_id"] for t in traces}
    todo = [d for d in ids if d not in done]
    gold, slice_meta = load_gold_and_meta(splits, root)
    provenance = FieldProvenance.load()
    fmt = cfg.backend.output_format
    if logprobs:
        backend.capture_logprobs = True  # optional backend attribute, not part of the Protocol
    schema, p_hash = schema_for(fmt), prompt_hash(fmt)
    ocr_root = paths.ocr_cache_dir()
    hint_on = bool(cfg.raw.get("header_hint", False))
    hint_log: dict[str, dict[str, Any]] = {}  # page image name -> header_hint trace record
    hint_page1: dict[str, ocr.PageOcr | None] = {}
    if hint_on:
        headerhint.require_supported(cfg.arm)

    def write_progress(status: str, last: str | None) -> None:
        atomic_write(
            out_dir / "progress.json",
            json.dumps(
                {
                    "status": status,
                    "done": len(traces),
                    "total": len(ids),
                    "last_doc": last,
                    "run_id": run_id,
                },
                indent=1,
            ),
        )

    def write_predictions() -> list[dict[str, str]]:
        # the choke point: numbers become plain-number strings here whatever the trace holds
        # (a resumed run may carry traces written before `coerce` existed); idempotent. The
        # schema repair is the same: a no-op on traces that already carry it, so the returned
        # events are only those of old traces (counted into the manifest)
        preds = coerce_predictions({t["doc_id"]: t["prediction"] for t in traces})
        preds, late = repair_predictions(preds)
        atomic_write(pred_path, json.dumps(preds, indent=1))
        return late

    def make_record(
        idx: int, img_path: Path, result: tuple[str, dict[str, Any] | None, dict[str, Any]]
    ) -> dict[str, Any]:
        raw, parsed, meta = result
        trace_lp = meta.pop("logprob_trace", None)  # bulky: lives beside meta, not in it
        record = {
            "page": idx + 1,
            "image": img_path.name,
            "raw_text": raw,
            "parsed": parsed,
            "json_valid": parsed is not None,
            "schema_errors": schema_errors(parsed) if parsed is not None else [],
            "meta": meta,
        }
        if hint_on:
            record["header_hint"] = hint_log.get(img_path.name)
        if logprobs:  # None = capture failed (meta carries `logprob_error`); smoke fails closed
            record["field_logprobs"] = (
                field_logprobs(raw, trace_lp["ends"], trace_lp["lp"], trace_lp["lp_c"])
                if trace_lp
                else None
            )
            record["token_logprobs"] = trace_lp
        return record

    def finish_doc(doc_id: str, page_records: list[dict[str, Any]]) -> None:
        pages = [page_for_merge(p["raw_text"], p["parsed"], fmt) for p in page_records]
        merged = merge_pages(pages, provenance)
        doc, flags = normalize_doc(merged.doc)
        rules_rec: dict[str, Any] | None = None
        if rule_cfg is not None:  # opt-in post-processing v1, before coerce / repair
            doc, rules_rec = _apply_post_rules(doc, doc_id, page_records, rule_cfg, ocr_root)
        doc, repairs = repair_doc(coerce_doc(doc))  # events: field + reason, never the value
        trace = {
            "doc_id": doc_id,
            "split": split_of[doc_id],
            "arm": cfg.arm,
            "output_format": fmt,
            "pages": page_records,
            "merge": merged.diffs,
            "normalize_flags": flags,
            "prediction": doc,
            "schema_repairs": repairs,
            "model": {"id": backend.model_id, "revision": backend.revision},
            "prompt": {"version": PROMPT_VERSION, "hash": p_hash},
            "config": {"name": cfg.name, "hash": cfg.config_hash},
            "git_commit": commit,
        }
        if rules_rec is not None:
            trace["rules"] = rules_rec
        traces.append(trace)
        append_jsonl(trace_path, trace)  # O(1) per doc: a full rewrite is O(n^2) bytes on Drive
        write_predictions()
        write_progress("running", doc_id)

    def page_input(job: PageJob) -> tuple[Any, str, str | None]:
        """(image, prompt, ocr text) of one page."""
        ocr_text = None
        if cfg.arm == "img_ocr":
            ocr_text = ocr.page_text(ocr.load_page(split_of[job.doc_id], job.path.stem, ocr_root))
        with Image.open(job.path) as im:
            image = im.convert("RGB")
        prompt = build_prompt(job.page_idx, job.n_pages, fmt)
        if hint_on:  # opt-in `header_hint` config key (headerhint.py); absent OCR = no hint
            prompt, hint_log[job.path.name] = headerhint.hinted_prompt_for_job(
                prompt, split_of[job.doc_id], job.doc_id, job.page_idx, job.n_pages, ocr_root,
                hint_page1,
            )  # fmt: skip
        return image, prompt, ocr_text

    stats = {"batches": 0, "fallbacks": 0}

    def stamp(
        batch: list[PageJob], results: list[tuple[str, dict[str, Any] | None, dict[str, Any]]]
    ) -> list[tuple[str, dict[str, Any] | None, dict[str, Any]]]:
        """Record which generate call (and which companions, in which order) decoded each page."""
        if len(batch) < 2:
            return results
        members = [[j.doc_id, j.page_idx] for j in batch]
        for _raw, _parsed, meta in results:
            meta["exec_batch"] = {
                "id": f"{batch[0].doc_id}:{batch[0].page_idx}",
                "members": members,
            }
        return results

    def run_batch(batch: list[PageJob]) -> list[tuple[str, dict[str, Any] | None, dict[str, Any]]]:
        """Results of `batch` in order; a failing batch is split down the fallback ladder."""
        try:
            if len(batch) == 1:  # the unbatched path, also the end of every fallback chain
                job = batch[0]
                image, prompt, ocr_text = page_input(job)
                if hasattr(backend, "set_context"):
                    backend.set_context(job.doc_id, job.page_idx, job.n_pages)
                stats["batches"] += 1
                return stamp(batch, [backend.extract_page(image, prompt, schema, ocr_text)])
            reqs = []
            for job in batch:
                image, prompt, ocr_text = page_input(job)
                ctx = (job.doc_id, job.page_idx, job.n_pages)
                reqs.append(PageRequest(image, prompt, schema, ocr_text, ctx))
            stats["batches"] += 1
            return stamp(batch, list(backend.extract_pages(reqs)))
        except Exception as exc:
            if len(batch) == 1 or not fallback:
                raise
            note = {
                "from": len(batch),
                "to": next_smaller_batch(len(batch)),
                "error": f"{type(exc).__name__}: {str(exc)[:200]}",
            }
            stats["fallbacks"] += 1
            print(
                f"batch of {len(batch)} failed ({note['error']}); retrying smaller", file=sys.stderr
            )
            release = getattr(backend, "release_memory", None)
            if callable(release):
                release()
            out = [r for chunk in split_for_fallback(batch) for r in run_batch(chunk)]
            for _raw, _parsed, meta in out:
                meta.setdefault("batch_fallback", []).insert(0, note)
            return out

    write_progress("running", None)
    if replay_batches is not None:
        jobs_by_key: dict[tuple[str, int], PageJob] = {}
        for d, imgs in replay_images.items():
            for idx, img_path in enumerate(imgs):
                jobs_by_key[(d, idx)] = PageJob(len(jobs_by_key), d, idx, len(imgs), img_path)
        slots_r: dict[str, list[dict[str, Any] | None]] = {
            d: [None] * len(replay_images[d]) for d in ids if d not in done
        }
        for members in replay_batches:
            missing = [m for m in members if tuple(m) not in jobs_by_key]
            if missing:
                raise ValueError(f"replay batch names {len(missing)} page(s) outside doc_ids")
            batch = [jobs_by_key[(d, i)] for d, i in members]
            if all(j.doc_id in done for j in batch):
                continue  # resume: every page of this batch already belongs to a traced document
            for job, result in zip(batch, run_batch(batch), strict=True):
                rec = make_record(job.page_idx, job.path, result)
                append_jsonl(
                    out_dir / "replay_pages.jsonl",
                    {
                        "doc_id": job.doc_id,
                        "page": job.page_idx + 1,
                        "raw_text": rec["raw_text"],
                        "exec_batch": rec["meta"].get("exec_batch"),
                    },
                )
                if job.doc_id in slots_r:
                    slots_r[job.doc_id][job.page_idx] = rec
            for d in list(slots_r):
                if all(slots_r[d]):
                    finish_doc(d, slots_r.pop(d))  # type: ignore[arg-type]
    elif batch_size == 1:
        for doc_id in todo:
            images = doc_page_images(split_of[doc_id], doc_id, root)
            page_records: list[dict[str, Any]] = []
            for idx, img_path in enumerate(images):
                job = PageJob(idx, doc_id, idx, len(images), img_path)
                page_records.append(make_record(idx, img_path, run_batch([job])[0]))
            finish_doc(doc_id, page_records)
    else:
        images_of = {d: doc_page_images(split_of[d], d, root) for d in todo}
        for window in doc_windows(todo, lambda d: len(images_of[d]), batch_size):
            jobs: list[PageJob] = []
            for d in window:
                for idx, img_path in enumerate(images_of[d]):
                    with Image.open(img_path) as im:
                        w, h = im.size  # header only: no pixel decode
                    n = len(images_of[d])
                    size = img_path.stat().st_size
                    jobs.append(PageJob(len(jobs), d, idx, n, img_path, w, h, size))
            slots: dict[str, list[dict[str, Any] | None]] = {
                d: [None] * len(images_of[d]) for d in window
            }
            next_doc = 0
            for batch in plan_batches(jobs, batch_size, cfg.backend.max_pixels):
                for job, result in zip(batch, run_batch(batch), strict=True):
                    slots[job.doc_id][job.page_idx] = make_record(job.page_idx, job.path, result)
                while next_doc < len(window) and all(slots[window[next_doc]]):
                    finish_doc(window[next_doc], slots[window[next_doc]])  # type: ignore[arg-type]
                    next_doc += 1

    late_repairs = write_predictions()  # also heals a kill between the trace append and the rewrite
    manifest["schema_repairs"] = {
        "n": sum(len(t.get("schema_repairs") or []) for t in traces) + len(late_repairs),
        "reason": "schema_invalid_date",
    }
    write_manifest(out_dir, manifest)
    wanted = set(ids)
    final = [t for t in traces if t["doc_id"] in wanted]
    metrics = build_metrics(
        cfg, final, gold, slice_meta, splits, split, run_id,
        {"id": backend.model_id, "revision": backend.revision}, commit, p_hash,
    )  # fmt: skip
    atomic_write(out_dir / "metrics.json", json.dumps(metrics, indent=1))
    if batch_size > 1 or stats["fallbacks"]:
        print(
            f"{run_id}: batch_size {batch_size}, {stats['batches']} generate calls, "
            f"{stats['fallbacks']} fallbacks",
            file=sys.stderr,
        )
    write_progress("complete", final[-1]["doc_id"] if final else None)
    if use_wandb:
        if os.environ.get("WANDB_API_KEY"):
            _log_wandb(metrics, cfg, run_id)
        else:
            print("--wandb given but WANDB_API_KEY is not set: skipping W&B", file=sys.stderr)
    return metrics


def make_backend(kind: str, cfg: SpikeConfig, split: str) -> VlmBackend:
    """``hf`` -> HfBackend; ``mock`` -> MockBackend replaying the split's gold labels."""
    if kind == "hf":
        return HfBackend(cfg.backend)
    if kind == "mock":
        gold: dict[str, dict[str, Any]] = {}
        for s in split_names(split):
            gold.update(ev.load_gold(paths.data_dir() / s / "labels"))
        return MockBackend(
            gold,
            ocr_token_budget=cfg.backend.ocr_token_budget,
            output_format=cfg.backend.output_format,
        )
    raise ValueError(f"unknown backend {kind!r} (use 'hf' or 'mock')")


def main_spike(args: Any) -> int:
    """CLI glue for ``python -m shipdoc spike`` (see cli.py for the argument list)."""
    cfg = load_config(args.config)
    doc_ids = load_doc_ids(args.docs)
    backend = make_backend(args.backend, cfg, args.split)
    batch_size, bench_info = args.batch_size, None
    if getattr(args, "bench_result", None):
        from shipdoc.bench import chosen_from_result, load_bench_result

        result = load_bench_result(Path(args.bench_result), cfg, backend)
        chosen = chosen_from_result(result)
        if batch_size is not None and batch_size != chosen:
            raise ValueError(
                f"--batch-size {batch_size} contradicts the bench result's {chosen} "
                f"({args.bench_result}); pass one of them"
            )
        batch_size = chosen
        bench_info = {
            "path": str(args.bench_result),
            "chosen": chosen,
            "deviation": result.get("deviation"),
        }
    metrics = run_spike(
        cfg,
        doc_ids,
        args.split,
        shard_run_id(args.run_id, args.shard),
        backend,
        resume=args.resume,
        limit=args.limit,
        use_wandb=args.wandb,
        logprobs=args.logprobs,
        batch_size=batch_size,
        shard=args.shard,
        base_run_id=args.run_id,
        bench_info=bench_info,
    )
    if metrics.get("scored"):
        lo, hi = metrics["OVERALL_ci95"]["lo"], metrics["OVERALL_ci95"]["hi"]
        print(
            f"{args.run_id}: {metrics['n_docs']} docs  OVERALL {100 * metrics['OVERALL']:.2f} "
            f"[{100 * lo:.2f}, {100 * hi:.2f}]  JSON-valid {metrics['json_validity_rate']:.3f}"
        )
    else:
        print(f"{args.run_id}: {metrics['n_docs']} docs (split has no labels; not scored)")
    return 0
