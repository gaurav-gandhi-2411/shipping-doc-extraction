"""Log aggregate evidence to the PUBLIC W&B project ``shipdoc-extract`` (dry-run by default).

    uv run python scripts/wandb_log_public.py --dry-run          # default: nothing leaves the disk
    uv run python scripts/wandb_log_public.py --sync --i-have-gg-approval   # GG ONLY, see below

GG MUST APPROVE ``--sync``. It refuses unless BOTH the flag ``--i-have-gg-approval`` and the
environment variable ``SHIPDOC_PUBLISH_OK=1`` are present, and even then it only logs online to the
project name below (it never runs ``wandb sync``, never logs in, never reads or sets an API key).
The private debug project ``shipdoc-extract-debug`` is not touched by this script.

What is logged (aggregates only; the data is confidential, spec section 2): training and
inference configs, seeds, pinned SHAs, model revision, dependency versions from ``uv.lock``;
the loss curves of the fine-tune runs (``metrics.jsonl``: step, loss, grad_norm, lr, held-out eval
loss); headline metrics and slice tables with CIs from the 02 run's ``metrics.json``; the ablation
ladder, rule gate and v1 replay numbers (parsed by ``scripts/report_inputs.py``); the calibration v2
summary and the reliability / risk-coverage aggregate tables. It never opens ``trace.jsonl``,
``predictions.json`` or ``oof_fields*.csv`` and never logs a per-document or per-field row.

Safety net: every payload object passes `check_runs` before anything is written or logged: keys
must be short ASCII identifiers (never doc-id shaped), values must be numbers, booleans, null, or
strings from an explicit vocabulary / hash / version / run-name pattern; strings over 80 chars and
anything path-like (data/, assignment/, drive letters, backslashes) are refused. The check is
structural: it cannot know a gold or predicted value, so it refuses every string it does not
recognise instead (fail closed). Violation messages carry the path and the reason, never the value.

Native-run options (additive, defaults unchanged): ``--train-run-dir`` is repeatable,
``--train-run-dir-optional`` takes folders or globs that are included only when complete (fold 2 and
the final run may not exist yet), and ``--slice-json NAME=PATH`` logs the dev slice breakdown of
``scripts/dev_slices_native.py`` as a summary-only run (``v1_5_dev_slices``). v2 has two reserved
entries, one of which is logged: ``v2_dev_slices`` (08 final-adapter dev run) or, when 08 was not
done, ``v2_oof_slices`` (pooled supplier-held-out OOF of FT + rules); each takes only the v2 slice
JSON of its own kind, and a stand-in rehearsal JSON is refused for ``--sync``.

Dry-run: the exact payload goes to ``<out-dir>/<run>/payload.json`` (default
``$SHIPDOC_TMP_DIR/wandb_dryrun``; any directory inside the repo or outside ``$SHIPDOC_TMP_DIR`` is
refused) and, only if ``wandb`` is importable, to a local OFFLINE run (``WANDB_MODE=offline``,
``WANDB_DIR=<out-dir>``). The manifest printed at the end holds counts and names only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_inputs as ri  # noqa: E402  (sibling script, not a package)

ROOT = Path(__file__).resolve().parents[1]
PROJECT = "shipdoc-extract"  # the PUBLIC project; the private one is shipdoc-extract-debug
TMP_ROOT = Path("D:/shipdoc/tmp")
DEFAULT_OUT = TMP_ROOT / "wandb_dryrun"
DEFAULT_RUNS = Path("D:/shipdoc/runs")
DEFAULT_ABLATION = TMP_ROOT / "g3_500" / "postproc_ablation_500.json"
APPROVAL_FLAG = "--i-have-gg-approval"
APPROVAL_ENV = "SHIPDOC_PUBLISH_OK"

MAX_STR = 80
MAX_KEY = 64
MAX_LIST = 100_000
KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.=\-]*$")
DOC_ID_RE = re.compile(r"(^|[_.=\-])(train|dev|test)_\d{2,}", re.IGNORECASE)
HEX_RE = re.compile(r"^[0-9a-f]{7,64}$")
VERSION_RE = re.compile(r"^\d+(\.\d+)+([+\-.][A-Za-z0-9.]+)?$")
# A run directory name: lowercase words ending in a 7-hex code SHA (+ optional precision tag).
RUN_RE = re.compile(r"^[a-z0-9]+(_[a-z0-9]+)*_[0-9a-f]{7}(_[a-z0-9]+)?$")
RUNG_RE = re.compile(r"^R\d+[a-z]?$")
STAGE_RE = re.compile(r"^(fold\d+|final|smoke)$")
PATH_HINTS = ("data/", "assignment/", "data\\", "assignment\\", "cache/", "runs/")


def _words(s: str) -> list[str]:
    """Whitespace-separated vocabulary words (keeps the long lists readable at 100 columns)."""
    return s.split()


SLICE_BASES = _words(
    "all invoices waybills awb_absent hawb_absent illegible multipage repeated_parts scanned "
    "digital waybill train dev invoice_only waybill_only absent_line illegible_any"
)
SLICE_NAMES = (
    frozenset(SLICE_BASES)
    | {f"{b}={v}" for b in SLICE_BASES for v in ("yes", "no")}
    | {f"fold={k}" for k in range(3)}  # the per-fold OOF rows of the slice JSON
)
# Metric keys of one row of the slice JSON (``dev_slices_native.py --json-out``) -> logged key.
SLICE_JSON_SETS = {
    "dev_slices": "dev100",
    "all500_slices": "oof500",
    "fold_slices": "oof_fold",
    "zs_slices": "zs_dev100",  # dev_oof only: the ZS + rules counterpart on the same documents
}
SLICE_JSON_SCALARS = {
    "n": "n",
    "raw_overall": "raw_overall",
    "header_acc": "header_acc",
    "row_f1": "row_f1",
    "fully_correct": "fully_correct",
    "false_fill": "false_fill",
}
# v2 entries (one per kind of v2 slice JSON): v2_dev_slices = 08, final adapter, dev100, L4 fp16
# inference; v2_dev_oof_slices = dev100 from the pooled OOF predictions (supplier-held-out, with the
# ZS + rules counterpart); v2_oof_slices = all 500 docs, pooled OOF (reserved contingency).
SLICE_RUN_NAMES = ("v1_5_dev_slices", "v2_dev_slices", "v2_dev_oof_slices", "v2_oof_slices")
V2_SLICE_SCHEMA = "shipdoc-v2-slices/1"  # scripts/dev_slices_native.py V2_SCHEMA
V2_KIND_RUN = {
    "dev_run": "v2_dev_slices",
    "dev_oof": "v2_dev_oof_slices",
    "pooled_oof": "v2_oof_slices",
}  # kind -> entry name
V2_KIND_SRC = {
    "dev_run": "dev_slices",
    "dev_oof": "dev_slices",
    "pooled_oof": "all500_slices",
}  # kind -> SLICE_JSON_SETS key
# the evidence label of each entry: logged as the summary value `evidence` (a fixed vocabulary
# string, never read from the JSON). The 08 numbers carry GG's wording 'final adapter, dev100, L4
# fp16 inference' ONLY when GG pasted the Colab banner showing an L4 (--dev-run-gpu-l4): no 08
# artifact records the GPU, so the default label has no GPU word.
V2_LABEL_DEV_RUN_L4 = "final adapter, dev100, L4 fp16 inference"
V2_KIND_LABEL = {
    "dev_run": "final adapter, dev100, fp16 inference",
    "dev_oof": "dev100, supplier-held-out",
    "pooled_oof": "pooled OOF 500, supplier-held-out",
}
GROUP_NAMES = frozenset(
    _words(
        "header supplier_part_number customer_part_number purchase_order quantity all_emitted doc "
        "all_per_type_tau"
    )
)
VOCAB = frozenset(
    {
        # run labels and groups (code-defined, see build_runs)
        "config_pins",
        "eval_zeroshot500",
        "ablation_ladder",
        "rule_gate",
        "calibration_v2",
        "oof_compare",
        *SLICE_RUN_NAMES,
        "config",
        "training",
        "eval",
        # config / manifest enumerations
        "Qwen/Qwen3.5-4B",
        "qwen35",
        "qwen35_4b",
        "qwen35_4b_img_only",
        "qwen35_4b_img_only_native",
        "img_only",
        "float16",
        "none",
        "auto",
        "bf16",
        "fp16",
        "fp32",
        "header_only",
        "json",
        "v2",
        "train+dev",
        "complete",
        "nested",
        # statuses of the calibration summary
        "attained",
        "near-miss",
        "NOT ATTAINABLE",
        "n/a",
        *V2_KIND_LABEL.values(),  # the evidence labels of the v2 slice entries
        V2_LABEL_DEV_RUN_L4,
    }
    | SLICE_NAMES
    | GROUP_NAMES
)
# Optional artifacts of unknown shape (oof_compare.json, g4 json): only leaves whose every key is
# in this vocabulary are taken; everything else is dropped and counted.
OPTIONAL_KEYS = frozenset(
    _words(
        "point lo hi delta n docs documents OVERALL header_field_accuracy row_f1 "
        "documents_fully_correct false_fill_rate ft zs zero_shot finetuned paired invoices "
        "waybills all ci95 auroc rows header_acc"
    )
) | set(SLICE_NAMES)
HISTORY_KEYS = _words(
    "step epoch loss grad_norm lr n_micro scaler_skipped seconds eval_loss eval_loss_epoch "
    "epoch_done"
)
SLICE_METRICS = _words(
    "OVERALL header_field_accuracy row_f1 documents_fully_correct false_fill_rate"
)
LOCK_PACKAGES = _words(
    "transformers peft torch accelerate huggingface-hub safetensors pillow numpy bitsandbytes "
    "xgrammar wandb scikit-learn pyyaml"
)
MANIFEST_KEYS = _words(
    "stage fold manifest_hash n_heldout_eval_pages code_sha train_config_signature "
    "n_train_docs steps total_steps n_train_pages base_repo base_revision precision adapter_sha256"
)
EVAL_SUMMARY_KEYS = _words(
    "n_docs n_pages json_validity_rate schema_validity_rate s_per_page_mean "
    "s_per_page_p95 peak_vram_max_bytes n_visual_tokens_mean n_input_tokens_mean "
    "n_output_tokens_mean false_fill_rate OVERALL"
)


class PublishRefused(RuntimeError):
    """``--sync`` was requested without the explicit approval flag and environment variable."""


class AllowlistError(ValueError):
    """The payload holds something the allowlist does not recognise (never carries the value)."""


# --------------------------------------------------------------------------------------------
# payload model
# --------------------------------------------------------------------------------------------


@dataclass
class Table:
    """One aggregate table: column names and rows of allowlisted scalars."""

    columns: list[str]
    rows: list[list[Any]]


@dataclass
class RunPayload:
    """Everything one W&B run would receive."""

    name: str
    group: str
    config: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    tables: dict[str, Table] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "project": PROJECT,
            "name": self.name,
            "group": self.group,
            "config": self.config,
            "summary": self.summary,
            "history": self.history,
            "tables": {k: {"columns": t.columns, "rows": t.rows} for k, t in self.tables.items()},
        }


# --------------------------------------------------------------------------------------------
# allowlist
# --------------------------------------------------------------------------------------------


def check_key(key: Any) -> str | None:
    """Reason a key is refused, or None."""
    if not isinstance(key, str):
        return "key_not_string"
    if len(key) > MAX_KEY or not KEY_RE.match(key) or not key.isascii():
        return "key_not_short_ascii_identifier"
    if DOC_ID_RE.search(key):
        return "key_looks_like_doc_id"
    return None


def check_string(s: str) -> str | None:
    """Reason a string value is refused, or None. Order matters: path checks beat vocabulary."""
    if len(s) > MAX_STR:
        return "string_longer_than_80"
    low = s.lower()
    if any(h in low for h in PATH_HINTS) or "\\" in s or re.match(r"^[A-Za-z]:", s):
        return "path_like"
    if DOC_ID_RE.search(s):
        return "looks_like_doc_id"
    if (
        s in VOCAB
        or HEX_RE.match(s)
        or VERSION_RE.match(s)
        or RUN_RE.match(s)
        or RUNG_RE.match(s)
        or STAGE_RE.match(s)
    ):
        return None
    return "string_not_in_allowlist"


def check_value(obj: Any, path: str = "$") -> list[str]:
    """All violations in `obj` as ``path: reason`` (the offending value is never included)."""
    out: list[str] = []
    if obj is None or isinstance(obj, bool):
        return out
    if isinstance(obj, int):
        return out
    if isinstance(obj, float):
        return out if math.isfinite(obj) else [f"{path}: non_finite_number"]
    if isinstance(obj, str):
        why = check_string(obj)
        return [f"{path}: {why}"] if why else out
    if isinstance(obj, list | tuple):
        if len(obj) > MAX_LIST:
            return [f"{path}: list_too_long"]
        for i, v in enumerate(obj):
            out += check_value(v, f"{path}[{i}]")
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            why = check_key(k)
            if why:
                out.append(f"{path}.<key #{list(obj).index(k)}>: {why}")
                continue
            out += check_value(v, f"{path}.{k}")
        return out
    return [f"{path}: type_{type(obj).__name__}_not_allowed"]


def check_runs(runs: list[RunPayload]) -> list[str]:
    """Violations over every run (name, group, config, summary, history, tables)."""
    out: list[str] = []
    for i, r in enumerate(runs):
        base = f"run[{i}]"
        out += check_value(r.name, f"{base}.name")
        out += check_value(r.group, f"{base}.group")
        out += check_value(r.config, f"{base}.config")
        out += check_value(r.summary, f"{base}.summary")
        out += check_value(r.history, f"{base}.history")
        for tname, t in r.tables.items():
            tbase = f"{base}.tables.{tname}"
            why = check_key(tname)
            if why:
                out.append(f"{tbase}: table_name_{why}")
            for c in t.columns:
                why = check_key(c)
                if why:
                    out.append(f"{tbase}.columns: {why}")
            ncol = len(t.columns)
            for j, row in enumerate(t.rows):
                if len(row) != ncol:
                    out.append(f"{tbase}.rows[{j}]: width_mismatch")
                out += check_value(row, f"{tbase}.rows[{j}]")
    return out


def assert_allowed(runs: list[RunPayload]) -> None:
    """Raise `AllowlistError` (paths and reasons only) if any payload object is refused."""
    bad = check_runs(runs)
    if bad:
        shown = "; ".join(bad[:10])
        raise AllowlistError(f"{len(bad)} payload objects refused by the allowlist: {shown}")


# --------------------------------------------------------------------------------------------
# gates and paths
# --------------------------------------------------------------------------------------------


def require_publish_approval(approved: bool, env: Mapping[str, str]) -> None:
    """Both the explicit flag and ``SHIPDOC_PUBLISH_OK=1`` are required; else `PublishRefused`."""
    missing = []
    if not approved:
        missing.append(APPROVAL_FLAG)
    if env.get(APPROVAL_ENV) != "1":
        missing.append(f"{APPROVAL_ENV}=1")
    if missing:
        raise PublishRefused(
            f"--sync refused: missing {' and '.join(missing)}. GG must approve publishing."
        )


def safe_out_dir(path: Path, allowed_roots: tuple[Path, ...] | None = None) -> Path:
    """Resolve `path`; refuse anything inside the repo or outside the allowed scratch roots."""
    allowed_roots = allowed_roots or (TMP_ROOT,)  # read at call time (tests override it)
    p = path.resolve()
    if p == ROOT or ROOT in p.parents:
        raise ValueError(f"refusing to write inside the repository: {p}")
    if not any(p == r.resolve() or r.resolve() in p.parents for r in allowed_roots):
        roots = ", ".join(str(r) for r in allowed_roots)
        raise ValueError(f"refusing output dir {p}: not under an allowed scratch root ({roots})")
    return p


# --------------------------------------------------------------------------------------------
# payload builders (aggregates only)
# --------------------------------------------------------------------------------------------


def flatten(d: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Nested dict -> ``{"a.b.c": leaf}``; lists of scalars are kept, other lists dropped."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, Mapping):
            out.update(flatten(v, f"{key}."))
        elif not isinstance(v, list) or all(not isinstance(x, Mapping | list) for x in v):
            out[key] = v
    return out


def _sha256_text(s: str) -> str:
    import hashlib

    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def read_yaml_config(path: Path) -> dict[str, Any]:
    """A training / inference YAML as a flat dict; over-long strings (regexes) become a sha256."""
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    flat = flatten(raw)
    for k, v in list(flat.items()):
        if isinstance(v, str) and len(v) > MAX_STR:
            flat[f"{k}_sha256"] = _sha256_text(v)
            del flat[k]
    return flat


def lock_versions(lock_path: Path) -> dict[str, str]:
    """``deps.<name>`` -> version from ``uv.lock`` (first non-local-version entry per package)."""
    pkgs = tomllib.loads(lock_path.read_text(encoding="utf-8")).get("package", [])
    out: dict[str, str] = {}
    for p in pkgs:
        name, ver = p.get("name"), p.get("version")
        if name in LOCK_PACKAGES and isinstance(ver, str) and "+" not in ver:
            out.setdefault(f"deps.{name}", ver)
    return out


def config_run(finetune_yaml: Path, infer_yaml: Path, lock_path: Path) -> RunPayload:
    """Training config, production inference config, seeds / pins and dependency versions."""
    cfg: dict[str, Any] = {}
    cfg.update({f"finetune.{k}": v for k, v in read_yaml_config(finetune_yaml).items()})
    cfg.update({f"inference.{k}": v for k, v in read_yaml_config(infer_yaml).items()})
    cfg.update(lock_versions(lock_path))
    return RunPayload("config_pins", "config", config=cfg)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _num(x: Any) -> bool:
    return isinstance(x, int | float) and not isinstance(x, bool)


def training_run(run_dir: Path) -> RunPayload:
    """One fine-tune run: loss curves + whitelisted manifest / train_status / sessions fields."""
    rows = _jsonl(run_dir / "metrics.jsonl")
    history = [
        {k: r[k] for k in HISTORY_KEYS if k in r and (_num(r[k]) or isinstance(r[k], bool))}
        for r in rows
    ]
    cfg: dict[str, Any] = {}
    man_path = run_dir / "final" / "manifest.json"
    if man_path.is_file():
        m = json.loads(man_path.read_text(encoding="utf-8"))
        for k in MANIFEST_KEYS:
            if k in m:
                cfg[k] = m[k]
        cfg["n_heldout_docs"] = len(m.get("heldout_doc_ids", []))  # a count, never the ids
        cfg.update(flatten(m.get("lora", {}), "lora."))
        cfg.update(flatten(m.get("inference_keys", {}), "inference_keys."))
        cfg.update({f"peft_sha256.{k}": v for k, v in m.get("peft_sha256", {}).items()})
    summary: dict[str, Any] = {"n_history_rows": len(history)}
    status = run_dir / "train_status.json"
    if status.is_file():
        s = json.loads(status.read_text(encoding="utf-8"))
        for k in ("state", "peak_vram_gib", "n_train_pages", "total_steps"):
            if k in s:
                summary[f"status.{k}"] = s[k]
    sess = run_dir / "sessions.json"
    if sess.is_file():
        ss = json.loads(sess.read_text(encoding="utf-8"))
        timed = [x["seconds"] for x in ss if _num(x.get("seconds"))]
        summary["sessions"] = len(ss)
        summary["sessions_without_duration"] = len(ss) - len(timed)
        if timed:
            summary["last_session_seconds"] = timed[-1]
    losses = [r["loss"] for r in history if "loss" in r]
    evals = [r["eval_loss"] for r in history if "eval_loss" in r]
    if losses:
        summary["final_loss"] = losses[-1]
    if evals:
        summary["final_eval_loss"] = evals[-1]
    summary["step_seconds_sum"] = sum(r["seconds"] for r in history if "seconds" in r)
    return RunPayload(run_dir.name, "training", config=cfg, summary=summary, history=history)


def _v2_slice_view(
    name: str, d: Mapping[str, Any], allow_stand_in: bool, gpu_l4: bool = False
) -> dict[str, Any]:
    """A v2 slice JSON (``dev_slices_native.py --mode dev_run|pooled_oof``) in the layout
    `slice_run` reads. The entry NAME must match the JSON's kind (08 dev run -> v2_dev_slices,
    pooled OOF -> v2_oof_slices) so the two can never be swapped; a stand-in rehearsal is refused
    unless `allow_stand_in` (never for an online sync)."""
    kind = d.get("kind")
    if kind not in V2_KIND_RUN:
        raise ValueError(f"v2 slice JSON: unknown kind (one of {', '.join(V2_KIND_RUN)})")
    if V2_KIND_RUN[kind] != name:
        raise ValueError(f"v2 slice JSON of kind {kind} belongs to the entry {V2_KIND_RUN[kind]}")
    if d.get("stand_in") and not allow_stand_in:
        raise ValueError("v2 slice JSON is a stand-in rehearsal: refused for a sync")
    if d.get("partial"):
        raise ValueError("v2 slice JSON is PARTIAL (a --folds plumbing run): never logged")
    view: dict[str, Any] = {
        "n_docs_all" if kind == "pooled_oof" else "n_docs_dev": d.get("n_docs"),
        "stand_in": bool(d.get("stand_in")),
        "evidence": V2_LABEL_DEV_RUN_L4 if gpu_l4 and kind == "dev_run" else V2_KIND_LABEL[kind],
        V2_KIND_SRC[kind]: d.get("slices", []),
        "fold_slices": d.get("fold_slices", []),
        "zs_slices": d.get("zs_slices", []),
    }
    run = d.get("run")
    if isinstance(run, str) and RUN_RE.match(run):  # pooled OOF holds three names: not logged
        view["run"] = run
    return view


def slice_run(
    name: str, path: Path, allow_stand_in: bool = True, gpu_l4: bool = False
) -> RunPayload:
    """A dev slice breakdown (``dev_slices_native.py --json-out``) as one summary-only run.

    Per set (``dev100``, ``oof500``, ``oof_fold``) and slice: scalars ``<set>.<slice>.<metric>``
    plus one table per set. Values stay fractions in [0, 1] exactly as in the artifact. Only the
    slice KEY is read (never the free-text ``label``); a slice key outside the vocabulary raises
    (fail closed), so no free text can reach the payload. A JSON of the v2 schema goes through
    `_v2_slice_view` (entry names ``v2_dev_slices`` / ``v2_oof_slices``).
    """
    d = json.loads(path.read_text(encoding="utf-8"))
    if d.get("schema") == V2_SLICE_SCHEMA:
        d = _v2_slice_view(name, d, allow_stand_in, gpu_l4)
    elif name != "v1_5_dev_slices":
        raise ValueError(f"the entry {name} takes a v2 slice JSON ({V2_SLICE_SCHEMA})")
    summary: dict[str, Any] = {}
    for k in ("n_docs_dev", "n_docs_all", "dev_groups", "dev_groups_also_in_train"):
        if _num(d.get(k)):
            summary[k] = d[k]
    if d.get("stand_in") is True:
        summary["stand_in"] = True
    if d.get("evidence") in (*V2_KIND_LABEL.values(), V2_LABEL_DEV_RUN_L4):  # fixed labels
        summary["evidence"] = d["evidence"]
    if isinstance(d.get("run"), str):
        summary["source_run"] = d["run"]
    cols = [
        "slice",
        "n",
        "overall",
        "overall_lo",
        "overall_hi",
        "raw_overall",
        "delta_rules",
        "delta_lo",
        "delta_hi",
        "header_acc",
        "row_f1",
        "fully_correct",
        "false_fill",
    ]
    tables: dict[str, Table] = {}
    for src_key, prefix in SLICE_JSON_SETS.items():
        rows: list[list[Any]] = []
        for s in d.get(src_key, []):
            sname = s["slice"]
            if sname not in SLICE_NAMES:
                raise ValueError(f"{src_key}: slice key not in the vocabulary (index {len(rows)})")
            ov, de = s.get("overall") or {}, s.get("delta") or {}
            vals: dict[str, Any] = {
                "overall": ov.get("point"),
                "overall_lo": ov.get("lo"),
                "overall_hi": ov.get("hi"),
                "delta_rules": de.get("delta"),
                "delta_lo": de.get("lo"),
                "delta_hi": de.get("hi"),
            }
            vals.update({out: s.get(src) for src, out in SLICE_JSON_SCALARS.items()})
            fz = s.get("ft_minus_zs") or {}  # dev_oof only: paired FT - ZS delta (summary only)
            extra = {"ft_minus_zs": fz.get("delta"), "ft_minus_zs_lo": fz.get("lo")}
            extra["ft_minus_zs_hi"] = fz.get("hi")
            for m, v in {**vals, **extra}.items():
                if _num(v):
                    summary[f"{prefix}.{sname}.{m}"] = v
            rows.append([sname, *(vals.get(c) for c in cols[1:])])
        if rows:
            tables[prefix] = Table(cols, rows)
    return RunPayload(name, "eval", summary=summary, tables=tables)


def parse_slice_json_arg(arg: str) -> tuple[str, Path]:
    """``NAME=PATH`` -> (NAME, PATH); NAME must be one of `SLICE_RUN_NAMES`."""
    name, sep, p = arg.partition("=")
    if not sep or name not in SLICE_RUN_NAMES or not p:
        raise ValueError(f"--slice-json wants NAME=PATH with NAME in {', '.join(SLICE_RUN_NAMES)}")
    return name, Path(p)


def complete_training_dir(run_dir: Path) -> bool:
    """A fine-tune folder that finished: metrics, manifest and ``train_status`` state complete."""
    status = run_dir / "train_status.json"
    if (
        not (run_dir / "metrics.jsonl").is_file()
        or not (run_dir / "final" / "manifest.json").is_file()
    ):
        return False
    if not status.is_file():
        return False
    try:
        return json.loads(status.read_text(encoding="utf-8")).get("state") == "complete"
    except (OSError, ValueError):
        return False


def optional_training_dirs(patterns: list[str], status: dict[str, str]) -> list[Path]:
    """Folders matching the (glob) patterns that are complete; absent or unfinished are skipped.

    Never raises for a missing run (fold 2 / the final run may not exist yet); the skip is
    recorded in `status` under the pattern's last component (names only, never a path).
    """
    import glob

    found: list[Path] = []
    for pat in patterns:
        label = f"optional_training_run[{Path(pat).name}]"
        hits = [Path(p) for p in sorted(glob.glob(pat)) if Path(p).is_dir()]
        ok = [p for p in hits if complete_training_dir(p)]
        status[label] = (
            f"{len(ok)} included"
            if ok
            else ("skipped (present but incomplete)" if hits else "skipped (absent)")
        )
        found += ok
    return found


def _est_cols(est: Mapping[str, Any] | None) -> list[Any]:
    e = est or {}
    return [e.get("point"), e.get("lo"), e.get("hi")]


def eval_run(metrics_path: Path) -> RunPayload:
    """Headline numbers, slice table and per-split table of the 02 (zero-shot, 500 docs) run."""
    m = json.loads(metrics_path.read_text(encoding="utf-8"))
    summary: dict[str, Any] = {k: m[k] for k in EVAL_SUMMARY_KEYS if k in m}
    summary.update(flatten(m.get("OVERALL_ci95", {}), "OVERALL_ci95."))
    pins = {
        "run_id": m.get("run_id"),
        "config": m.get("config"),
        "config_hash": m.get("config_hash"),
        "git_commit": m.get("git_commit"),
        "split": m.get("split"),
        "arm": m.get("arm"),
        "output_format": m.get("output_format"),
        "model.revision": (m.get("model") or {}).get("revision"),
        "prompt.version": (m.get("prompt") or {}).get("version"),
        "prompt.hash": (m.get("prompt") or {}).get("hash"),
    }
    summary.update({k: v for k, v in pins.items() if v is not None})
    cols = ["slice", "documents", "illegible_fields"]
    for met in SLICE_METRICS:
        cols += [met, f"{met}_lo", f"{met}_hi"]
    rows: list[list[Any]] = []
    for sname, s in m["slices"].items():
        row: list[Any] = [sname, s.get("documents"), s.get("illegible_fields")]
        ci = s.get("ci95", {})
        for met in SLICE_METRICS:
            e = ci.get(met) or {"point": s.get(met), "lo": None, "hi": None}
            row += _est_cols(e)
        rows.append(row)
    tables = {"slices": Table(cols, rows)}
    scols = ["split", "documents", "n_pages", "json_validity_rate", "false_fill_rate"]
    for met in ("OVERALL", "header_field_accuracy", "row_f1"):
        scols.append(met)
    scols += ["OVERALL_lo", "OVERALL_hi"]
    srows = []
    for sp, d in m.get("per_split", {}).items():
        ci = d.get("OVERALL_ci95", {})
        srows.append([sp] + [d.get(c) for c in scols[1:8]] + [ci.get("lo"), ci.get("hi")])
    tables["per_split"] = Table(scols, srows)
    return RunPayload("eval_zeroshot500", "eval", summary=summary, tables=tables)


def ablation_run(path: Path) -> RunPayload:
    """The cumulative post-processing ladder (rung ids only, no free-text labels)."""
    ab = ri.ablation_from_json(json.loads(path.read_text(encoding="utf-8")))
    cols = [
        "rung",
        "overall",
        "overall_lo",
        "overall_hi",
        "false_fill",
        "false_fill_lo",
        "false_fill_hi",
        "delta_vs_prev",
        "delta_lo",
        "delta_hi",
    ]
    rows = []
    for r in ab["rungs"]:
        d = r["delta_vs_prev"] or {}
        rows.append(
            [r["name"].split(" ")[0]]
            + _est_cols(r["overall"])
            + _est_cols(r["false_fill"])
            + [d.get("delta"), d.get("lo"), d.get("hi")]
        )
    return RunPayload(
        "ablation_ladder",
        "eval",
        summary={"n_docs": ab["n_docs"], "n_rungs": len(rows)},
        tables={"ladder": Table(cols, rows)},
    )


def rules_run(gate_md: Path, replay_md: Path | None) -> RunPayload:
    """Rule gate (R1/R2/R3) and v1 replay combined delta, via report_inputs' md parsers."""
    rules = ri.rules_from_gate_md(gate_md.read_text(encoding="utf-8"))
    if replay_md is not None and replay_md.is_file():
        rules["combined"] = {
            "delta": ri.combined_delta_from_replay_md(replay_md.read_text(encoding="utf-8"))
        }
    return RunPayload("rule_gate", "eval", summary=flatten(rules))


def _csv_table(path: Path, text_cols: tuple[str, ...], drop_empty: str | None) -> Table:
    """CSV aggregate -> Table; rows with a non-finite cell (or empty bins) are dropped."""
    import csv

    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        cols = list(reader.fieldnames or [])
        rows: list[list[Any]] = []
        for rec in reader:
            if drop_empty and float(rec[drop_empty]) == 0:
                continue
            row: list[Any] = []
            ok = True
            for c in cols:
                if c in text_cols:
                    row.append(rec[c])
                    continue
                v = float(rec[c])
                if not math.isfinite(v):
                    ok = False
                    break
                row.append(int(v) if c == "n" else v)
            if ok:
                rows.append(row)
    return Table(cols, rows)


def calibration_run(cal_dir: Path, v1_json: Path | None) -> RunPayload:
    """Calibration v2 summary plus the reliability and risk-coverage aggregate tables."""
    v2 = json.loads((cal_dir / "calibration_v2.json").read_text(encoding="utf-8"))
    v1 = json.loads(v1_json.read_text(encoding="utf-8")) if v1_json else None
    summary = flatten(ri.calibration_from_v2(v2, v1))
    tables = {
        "reliability": _csv_table(cal_dir / "reliability.csv", ("slice", "group"), "n"),
        "risk_coverage": _csv_table(cal_dir / "risk_coverage.csv", ("slice", "group"), None),
    }
    return RunPayload("calibration_v2", "eval", summary=summary, tables=tables)


def optional_numeric_run(path: Path, dropped: list[int]) -> RunPayload:
    """oof_compare.json / g4 json of unknown shape: only numeric leaves under vocabulary keys."""
    leaves: dict[str, Any] = {}

    def walk(d: Any, trail: tuple[str, ...]) -> None:
        if isinstance(d, Mapping):
            for k, v in d.items():
                walk(v, (*trail, str(k)))
        elif _num(d) and trail and all(t in OPTIONAL_KEYS for t in trail):
            leaves[".".join(trail)] = d
        else:
            dropped[0] += 1

    walk(json.loads(path.read_text(encoding="utf-8")), ())
    return RunPayload("oof_compare", "eval", summary=leaves)


def find_training_runs(runs_root: Path) -> list[Path]:
    """Run directories (holding metrics.jsonl) of fold and final fine-tunes, sorted by name."""
    found = [
        p.parent
        for pat in ("ft_fold*/*/metrics.jsonl", "ft_final*/*/metrics.jsonl")
        for p in runs_root.glob(pat)
    ]
    return sorted(found, key=lambda p: p.name)


def build_runs(args: argparse.Namespace) -> tuple[list[RunPayload], dict[str, str]]:
    """All run payloads plus ``{source: found|missing}`` for the manifest."""
    status: dict[str, str] = {}
    runs: list[RunPayload] = [config_run(args.finetune_config, args.infer_config, ROOT / "uv.lock")]
    rr: Path = args.runs_root
    tdirs = list(args.train_run_dir) or find_training_runs(rr)
    for extra in optional_training_dirs(list(args.train_run_dir_optional), status):
        if extra.resolve() not in {d.resolve() for d in tdirs}:
            tdirs.append(extra)
    status["training_runs"] = f"{len(tdirs)} found"
    runs += [training_run(d) for d in tdirs]
    for spec in args.slice_json:
        sname, spath = parse_slice_json_arg(spec)
        runs.append(
            slice_run(sname, spath, allow_stand_in=not args.sync, gpu_l4=args.dev_run_gpu_l4)
        )
        status[f"slice_json[{sname}]"] = "found"
    if args.training_only:
        # Training curves + configs only: the eval / rule / calibration runs below describe other
        # (1260-token) runs and must not sit next to a native-resolution fold model.
        status["eval_runs"] = "skipped (--training-only)"
        return runs, status

    def have(label: str, p: Path | None) -> bool:
        ok = p is not None and p.is_file()
        status[label] = "found" if ok else "missing"
        return ok

    zs = rr / "zeroshot500" / "zeroshot500_qwen35_4b_img_only_keyed_42b812b" / "metrics.json"
    if have("zeroshot500_metrics.json", zs):
        runs.append(eval_run(zs))
    if have("ablation_json", args.ablation_json):
        runs.append(ablation_run(args.ablation_json))
    gate = ROOT / "reports" / "rule_gate.md"
    if have("rule_gate.md", gate):
        runs.append(rules_run(gate, ROOT / "reports" / "v1_replay.md"))
    cal = rr / "calibration_v2" / "zeroshot500_qwen35_4b_img_only_keyed_42b812b"
    if have("calibration_v2.json", cal / "calibration_v2.json"):
        v1 = rr / "calibration" / cal.name / "calibration.json"
        runs.append(calibration_run(cal, v1 if v1.is_file() else None))
    dropped = [0]
    cands = sorted([*rr.glob("*/oof_compare.json"), *rr.glob("*/g4*.json")])
    status["oof_compare_or_g4_json"] = f"{len(cands)} found"
    for c in cands[:1]:
        runs.append(optional_numeric_run(c, dropped))
    status["optional_dropped_leaves"] = str(dropped[0])
    return runs, status


# --------------------------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------------------------


def write_payloads(runs: list[RunPayload], out_dir: Path) -> list[Path]:
    """Write ``<out_dir>/<run>/payload.json`` for every run (the exact would-be payload)."""
    paths = []
    for r in runs:
        d = out_dir / r.name
        d.mkdir(parents=True, exist_ok=True)
        p = d / "payload.json"
        p.write_text(json.dumps(r.to_json(), indent=1, sort_keys=True) + "\n", encoding="utf-8")
        paths.append(p)
    return paths


def manifest_lines(runs: list[RunPayload], status: Mapping[str, str]) -> list[str]:
    """Counts and names only; never a value."""
    lines = [f"project: {PROJECT} (public)"]
    for k, v in status.items():
        lines.append(f"source {k}: {v}")
    tot = {"configs": 0, "scalars": 0, "history_rows": 0, "tables": 0, "table_rows": 0}
    for r in runs:
        nrows = sum(len(t.rows) for t in r.tables.values())
        tabs = ", ".join(f"{k}({len(t.columns)}c x {len(t.rows)}r)" for k, t in r.tables.items())
        lines.append(
            f"run {r.name} [{r.group}]: config {len(r.config)}, summary {len(r.summary)}, "
            f"history {len(r.history)} rows, tables: {tabs or 'none'}"
        )
        tot["configs"] += len(r.config)
        tot["scalars"] += len(r.summary)
        tot["history_rows"] += len(r.history)
        tot["tables"] += len(r.tables)
        tot["table_rows"] += nrows
    lines.append(
        f"TOTAL runs {len(runs)}: configs {tot['configs']}, scalars {tot['scalars']}, "
        f"history rows {tot['history_rows']}, tables {tot['tables']}, "
        f"table rows {tot['table_rows']}"
    )
    return lines


def _import_wandb() -> Any | None:
    """The wandb module, or None when it is not installed (it is a ``vlm``-group dependency)."""
    try:
        import wandb  # noqa: PLC0415  (lazy: the dry-run must work without the library)
    except ImportError:
        return None
    return wandb


def log_with_wandb(wandb: Any, runs: list[RunPayload]) -> None:
    """Log every run with the wandb client; mode comes from the environment (offline in dry-run)."""
    for r in runs:
        run = wandb.init(
            project=PROJECT,
            name=r.name,
            id=r.name,
            group=r.group,
            config=r.config,
            resume="allow",
            reinit=True,
        )
        for row in r.history:
            wandb.log(row, step=int(row["step"]) if "step" in row else None)
        for k, v in r.summary.items():
            run.summary[k] = v
        for tname, t in r.tables.items():
            wandb.log({tname: wandb.Table(columns=t.columns, data=t.rows)})
        run.finish()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="default: stage locally only")
    mode.add_argument("--sync", action="store_true", help=f"online log; needs {APPROVAL_FLAG}")
    ap.add_argument(APPROVAL_FLAG, dest="approved", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS)
    ap.add_argument("--ablation-json", type=Path, default=DEFAULT_ABLATION)
    ap.add_argument(
        "--train-run-dir",
        type=Path,
        action="append",
        default=[],
        help="a fine-tune run folder (metrics.jsonl + final/manifest.json); repeatable; "
        "default: discover ft_fold*/ and ft_final*/ under --runs-root",
    )
    ap.add_argument(
        "--train-run-dir-optional",
        action="append",
        default=[],
        metavar="PATTERN",
        help="a fine-tune run folder or glob that is included only if it exists and its "
        "train_status.json says complete; absent or unfinished is skipped, never an error",
    )
    ap.add_argument(
        "--slice-json",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="dev slice breakdown JSON of dev_slices_native.py as a summary-only run; NAME is "
        f"one of {', '.join(SLICE_RUN_NAMES)}; repeatable",
    )
    ap.add_argument(
        "--dev-run-gpu-l4",
        action="store_true",
        help="GG pasted the Colab banner of the 08 run and it shows an L4: the v2_dev_slices "
        "evidence label then says 'L4' (no 08 artifact records the GPU; default: no GPU word)",
    )
    ap.add_argument(
        "--training-only",
        action="store_true",
        help="log only the config run and the training runs (no eval / rule / calibration runs)",
    )
    ap.add_argument(
        "--finetune-config",
        type=Path,
        default=ROOT / "configs" / "finetune_qwen35_4b.yaml",
        help="fine-tune YAML logged as config (use the native one for a native run)",
    )
    ap.add_argument(
        "--infer-config",
        type=Path,
        default=ROOT / "configs" / "spike_qwen35_4b_img_only.yaml",
        help="inference YAML logged as config",
    )
    return ap


def main(argv: list[str] | None = None, env: Mapping[str, str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    env = os.environ if env is None else env
    if args.sync:
        try:
            require_publish_approval(args.approved, env)
        except PublishRefused as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 3
    try:
        out_dir = safe_out_dir(args.out_dir)
        runs, status = build_runs(args)
        assert_allowed(runs)
    except (ValueError, OSError, KeyError, ri.br.ReportError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 4
    for line in manifest_lines(runs, status):
        print(line)
    wandb = _import_wandb()
    if args.sync:
        if wandb is None:
            print("error: wandb is not importable (uv sync --group vlm needed)", file=sys.stderr)
            return 5
        os.environ.pop("WANDB_MODE", None)
        log_with_wandb(wandb, runs)
        print(f"logged online to project {PROJECT}")
        return 0
    paths = write_payloads(runs, out_dir)
    print(f"dry-run: wrote {len(paths)} payload.json files under {out_dir}")
    if wandb is None:
        print("dry-run: wandb not importable; payload.json only (no offline run written)")
    else:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["WANDB_DIR"] = str(out_dir)
        log_with_wandb(wandb, runs)
        print(f"dry-run: offline wandb runs under {out_dir} (never synced)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
