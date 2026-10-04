"""Production pipeline v1 (notebook 04b): reuse of the v0 model outputs, assembly, extra checks.

Notebook 04 (safety submission v0) decoded the 200 test documents once. v1 = the same model
outputs + OCR + post-processing rules R1 / R2 / R3. Re-decoding is only needed when something that
changes the model's output changed. This module decides that, loudly and fail closed:

``decide_reuse``     a pure refusal function. It lists EVERY reason reuse is not allowed
                     (`ReuseDecision.reasons`); one reason is enough to fall back to full inference.
                     The notebook then runs the full path and never reuses silently.
``assemble_reuse``   the CPU assembly over the v0 ``trace.jsonl`` (R1-R3 default on, the OCR cache
                     passed), writing the same files as ``predict assemble`` into the v1 folder.
``finalize``         the v1-only blocking checks, added to ``validation_report.json`` of BOTH the
                     reuse and the full-inference path (rule switches, R2 not skipped, shapes
                     sha256, OCR cache complete, OCR re-check, assemble-twice in reuse mode).

CLI: ``python -m shipdoc.reuse check | assemble | finalize | compare``.

What replaces code identity. The v0 code SHA (42b812b...) cannot equal the v1 pin (there are newer
commits), so the contract "reuse when code, config, model revision and batch size match" cannot
hold literally. It is replaced by (1) a DECODE-PATH FINGERPRINT: the git blob hash of every file
on the model-output path (`DECODE_PATH_FILES`) is identical at the v0 SHA and at the checked-out
HEAD, and `spike.py` (changed by opt-in code since v0) differs only by the allowlisted opt-in
additions (`SPIKE_ALLOWED_*`, byte-exact by hash, plus the config must not enable
``header_hint`` and the notebook never passes ``rule_cfg`` to the decoder); (2) the config hash and
the prompt hash of the checked-out code equal v0's; (3) the model revision, seed, output format,
arm, shard and the BATCH SIZE equal v1's. The notebook prints ``git diff --stat <v0> <pin>``.

In reuse mode the VLM is not run: VLM determinism rests on v0's own determinism check (re-attached
to the manifest with the sha256 of its source). Test data policy: nothing here prints or stores an
extracted value; reports carry counts, ids, hashes and key paths only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shipdoc import paths, postrules, prompts, spike
from shipdoc import predict as pr
from shipdoc.coerce import CoerceError, coerce_predictions, repair_predictions
from shipdoc.postrules import RuleConfig

SUBMISSION_VERSION = "v1"
OUT_FILES = (*pr.OUT_FILES, "ocr_timing.json")
SPIKE_PATH = "src/shipdoc/spike.py"
# Every file whose content can change what the model emits for a page or how the trace's merged,
# normalised, coerced ``prediction`` is built from it (call graph: predict.run_test ->
# spike.run_spike -> {extract (backend, parse), prompts, logprobs, batching, shard, merge,
# normalize, coerce, runmeta (manifest/resume), eval (imported by spike), paths (config)} and
# merge-shards -> shardmerge). Plus the merge provenance table, the production config, and the
# dependency pins (a different torch / transformers lock is a different decode).
DECODE_PATH_FILES = (
    "src/shipdoc/__init__.py",
    "src/shipdoc/extract.py",
    "src/shipdoc/prompts.py",
    "src/shipdoc/logprobs.py",
    "src/shipdoc/batching.py",
    "src/shipdoc/shard.py",
    "src/shipdoc/shardmerge.py",
    "src/shipdoc/merge.py",
    "src/shipdoc/normalize.py",
    "src/shipdoc/coerce.py",
    "src/shipdoc/runmeta.py",
    "src/shipdoc/eval.py",
    "src/shipdoc/paths.py",
    "meta/field_provenance.json",
    "configs/spike_qwen35_4b_img_only.yaml",
    "uv.lock",
    "pyproject.toml",
)
# `spike.py` between 42b812b and b20b391 changed only by (a) the opt-in `header_hint` hook
# (default off: needs the config key), (b) the opt-in `rule_cfg` post-rules hook (default None:
# the submission applies the rules at assembly instead) and (c) the import line / `return` line
# those two touch. `git diff -U0` gives 2 removed and 56 added lines; their sha256 (the lines
# joined with LF plus a trailing LF) are pinned here, so ANY other edit to spike.py breaks reuse.
SPIKE_ALLOWED_REMOVED_SHA256 = "65bb893c07fad6063e8991988ca9e2174bed30096d9ad5ab80e14cc50a002147"
SPIKE_ALLOWED_ADDED_SHA256 = "f08e61faecea4823f0199814398653b7ef92201325253bf87b01de31b69dc6f5"
SHAPES_REL = "meta/slot_shapes.json"
BAD_SKIPS = ("R2/no_ocr", "R2/ocr_incomplete", "R3/no_shapes")
MAX_SAMPLE = pr.MAX_SAMPLE


class ReuseError(RuntimeError):
    """A stage refused; the message carries counts, ids and hashes only."""


# --------------------------------------------------------------------------------------------
# Decode-path fingerprint (git blob hashes)
# --------------------------------------------------------------------------------------------

BlobFn = Callable[[str, str], "str | None"]
DiffFn = Callable[[str, str, str], "str | None"]


def git_blob(sha: str, path: str, repo: Path | None = None) -> str | None:
    """``git rev-parse <sha>:<path>`` (the blob hash), None when git cannot resolve it."""
    res = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{sha}:{path}"],
        cwd=repo or paths.REPO_ROOT, capture_output=True, text=True, check=False,
    )  # fmt: skip
    out = res.stdout.strip()
    return out if res.returncode == 0 and out else None


def git_diff_u0(a: str, b: str, path: str, repo: Path | None = None) -> str | None:
    """``git diff -U0 a b -- path`` text, None when git fails (unknown sha)."""
    res = subprocess.run(
        ["git", "diff", "-U0", "--no-color", "--no-ext-diff", a, b, "--", path],
        cwd=repo or paths.REPO_ROOT, capture_output=True, text=True, check=False,
    )  # fmt: skip
    return res.stdout if res.returncode == 0 else None


def git_diff_stat(a: str, b: str, repo: Path | None = None) -> str:
    """``git diff --stat a b -- src configs`` (file names and counts), or an error line."""
    res = subprocess.run(
        ["git", "diff", "--stat", a, b, "--", "src", "configs"],
        cwd=repo or paths.REPO_ROOT, capture_output=True, text=True, check=False,
    )  # fmt: skip
    return (
        res.stdout.strip()
        if res.returncode == 0
        else f"git diff --stat failed: {res.stderr[-200:]}"
    )


def split_diff_lines(diff_text: str) -> tuple[list[str], list[str]]:
    """``(removed, added)`` line texts of a ``git diff -U0`` (file headers excluded)."""
    removed, added = [], []
    for ln in diff_text.splitlines():
        if ln.startswith("---") or ln.startswith("+++"):
            continue
        if ln.startswith("-"):
            removed.append(ln[1:])
        elif ln.startswith("+"):
            added.append(ln[1:])
    return removed, added


def _lines_sha(lines: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def spike_diff_check(diff_text: str | None, ack: bool = False) -> dict[str, Any]:
    """Is the spike.py diff exactly the allowlisted opt-in additions?

    ``ok`` when the removed / added lines hash to the pinned constants (or the diff is empty).
    `ack` (the notebook's ``REUSE_ACK_SPIKE_DIFF``) accepts ANY other diff, recorded as such.
    An unavailable diff (None) is never ok, not even with `ack`: that is a git failure.
    """
    if diff_text is None:
        return {"ok": False, "state": "diff unavailable", "acked": False}
    removed, added = split_diff_lines(diff_text)
    if not removed and not added:
        return {"ok": True, "state": "spike.py identical", "acked": False, "removed": 0, "added": 0}
    exact = (
        _lines_sha(removed) == SPIKE_ALLOWED_REMOVED_SHA256
        and _lines_sha(added) == SPIKE_ALLOWED_ADDED_SHA256
    )
    base = {"removed": len(removed), "added": len(added),
            "removed_sha256": _lines_sha(removed), "added_sha256": _lines_sha(added)}  # fmt: skip
    if exact:
        return {
            "ok": True,
            "state": "only the allowlisted opt-in additions",
            "acked": False,
            **base,
        }
    if ack:
        return {"ok": True, "state": "NOT the allowlisted diff: accepted by REUSE_ACK_SPIKE_DIFF",
                "acked": True, **base}  # fmt: skip
    return {"ok": False, "state": "spike.py differs from the allowlisted opt-in additions",
            "acked": False, **base}  # fmt: skip


def decode_path_report(
    v0_sha: str, head_sha: str, blob_fn: BlobFn = git_blob, diff_fn: DiffFn = git_diff_u0,
    files: Sequence[str] = DECODE_PATH_FILES, ack_spike: bool = False,
) -> dict[str, Any]:  # fmt: skip
    """Compare the git blob of every decode-path file at `v0_sha` and `head_sha`.

    ``differing`` = the blobs differ; ``unresolved`` = a blob could not be read at either SHA (a
    missing file or an unknown SHA is never treated as equal). ``spike`` = `spike_diff_check`.
    ``ok`` only when nothing differs, nothing is unresolved and the spike diff is allowed.
    """
    blobs: dict[str, dict[str, str | None]] = {}
    differing, unresolved = [], []
    for f in files:
        a, b = blob_fn(v0_sha, f), blob_fn(head_sha, f)
        blobs[f] = {"v0": a, "head": b}
        if a is None or b is None:
            unresolved.append(f)
        elif a != b:
            differing.append(f)
    spike_rep = spike_diff_check(diff_fn(v0_sha, head_sha, SPIKE_PATH), ack_spike)
    return {
        "ok": not differing and not unresolved and bool(spike_rep["ok"]),
        "v0_sha": v0_sha, "head_sha": head_sha, "files": len(files),
        "differing": differing, "unresolved": unresolved, "blobs": blobs, "spike": spike_rep,
    }  # fmt: skip


# --------------------------------------------------------------------------------------------
# The reuse decision
# --------------------------------------------------------------------------------------------


@dataclass
class ReuseDecision:
    """``ok`` iff `reasons` is empty. Reasons are ``<code>: <counts / hashes only>``."""

    ok: bool
    reasons: list[str] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)


def production_facts(cfg: spike.SpikeConfig, batch_size: int | None, shard: str) -> dict[str, Any]:
    """What v1 would decode with, read from the CHECKED-OUT code (never typed)."""
    return {
        "config_hash": cfg.config_hash,
        "prompt_hash": prompts.prompt_hash(cfg.backend.output_format),
        "model_revision": cfg.backend.revision,
        "seed": cfg.backend.seed,
        "output_format": cfg.backend.output_format,
        "arm": cfg.arm,
        "shard": shard,
        "batch_size": batch_size,
        "header_hint": bool(cfg.raw.get("header_hint", False)),
    }


def decide_reuse(
    v0_manifest: Mapping[str, Any] | None,
    v0_report: Mapping[str, Any] | None,
    v0_traces: Sequence[Mapping[str, Any]] | None,
    expected_pages: Mapping[str, int],
    prod: Mapping[str, Any],
    decode: Mapping[str, Any] | None,
    head_sha: str,
) -> ReuseDecision:
    """May the v0 model outputs be reused for v1? Pure: every input is passed in.

    Refuses (a reason per failed condition, all listed) unless ALL hold: the v0 manifest is
    present with a clean 40-hex code SHA and its determinism ok; the v0 validation report says
    ok; the v0 config hash and prompt hash (manifest AND every trace) equal `prod`'s; model
    revision, seed, output format, arm, shard and batch size equal `prod`'s (a batch size of None
    means "bench decides": reuse refused); logprobs recorded on every page; the trace holds
    exactly the expected doc ids once each, each with its expected page count and raw text;
    `head_sha` is clean; `header_hint` is off in the config; and the decode-path fingerprint
    (`decode_path_report`) is ok.
    """
    why: list[str] = []
    facts: dict[str, Any] = {"v0_code_sha": None, "head_sha": head_sha}
    if v0_manifest is None:
        why.append("v0_manifest_missing: no manifest.json in the v0 folder")
        v0_manifest = {}
    if v0_report is None:
        why.append("v0_report_missing: no validation_report.json in the v0 folder")
    elif v0_report.get("ok") is not True or not all(
        c.get("ok") for c in (v0_report.get("checks") or {}).values()
    ):
        why.append(
            "v0_not_validated: the v0 validation_report is not ok (REJECTED or a failed check)"
        )
    v0_sha = str(v0_manifest.get("code_sha", "")) if v0_manifest else ""
    facts["v0_code_sha"] = v0_sha or None
    if v0_manifest and (len(v0_sha) != 40 or "+dirty" in v0_sha):
        why.append(f"v0_code_sha: {v0_sha!r} is not a clean 40-hex commit (decode path unknowable)")
    if "+dirty" in head_sha or len(head_sha.removesuffix("+dirty")) != 40:
        why.append(f"head_dirty: checked-out code is {head_sha!r}, not a clean commit")
    if v0_manifest and not (v0_manifest.get("determinism") or {}).get("ok"):
        why.append("v0_determinism: the v0 manifest has no passing determinism record")

    def need(name: str, got: Any, want: Any) -> None:
        facts[name] = {"v0": got, "v1": want}
        if got != want or want is None:
            why.append(f"{name}: v0 {got!r} != v1 {want!r}")

    man_prompt = (v0_manifest.get("prompt") or {}).get("hash")
    trace_prompts = {(t.get("prompt") or {}).get("hash") for t in (v0_traces or [])}
    prompt_set = trace_prompts | ({man_prompt} if man_prompt else set())
    need("config_hash", (v0_manifest.get("config") or {}).get("hash"), prod["config_hash"])
    if v0_traces:
        trace_cfg = {(t.get("config") or {}).get("hash") for t in v0_traces}
        if trace_cfg != {prod["config_hash"]}:
            why.append(f"config_hash_traces: trace config hashes {sorted(map(str, trace_cfg))}")
    facts["prompt_hash"] = {"v0": sorted(map(str, prompt_set)), "v1": prod["prompt_hash"]}
    if prompt_set != {prod["prompt_hash"]}:
        why.append(f"prompt_hash: v0 {sorted(map(str, prompt_set))} != v1 {prod['prompt_hash']!r}")
    need("model_revision", (v0_manifest.get("model") or {}).get("revision"), prod["model_revision"])
    need("seed", v0_manifest.get("seed"), prod["seed"])
    need("shard", v0_manifest.get("shard"), prod["shard"])
    if prod["batch_size"] is None:
        why.append("batch_size_unset: BATCH_SIZE is None (the bench would decide): not reusable")
    need("batch_size", v0_manifest.get("batch_size"), prod["batch_size"])
    if prod.get("header_hint"):
        why.append("header_hint: the config enables header_hint (a different prompt)")
    traces = list(v0_traces or [])
    if v0_traces is None:
        why.append("trace_missing: no trace.jsonl in the v0 folder")
    else:
        fmt = {t.get("output_format") for t in traces}
        arm = {t.get("arm") for t in traces}
        facts["output_format"] = {"v0": sorted(map(str, fmt)), "v1": prod["output_format"]}
        facts["arm"] = {"v0": sorted(map(str, arm)), "v1": prod["arm"]}
        if fmt != {prod["output_format"]}:
            why.append(f"output_format: v0 {sorted(map(str, fmt))} != v1 {prod['output_format']!r}")
        if arm != {prod["arm"]}:
            why.append(f"arm: v0 {sorted(map(str, arm))} != v1 {prod['arm']!r}")
        revs = {(t.get("model") or {}).get("revision") for t in traces}
        if revs != {prod["model_revision"]}:
            why.append(f"model_revision_traces: {sorted(map(str, revs))}")
        ids = [t.get("doc_id") for t in traces]
        rep = pr.check_ids(ids, list(expected_pages))
        facts["trace_ids"] = {k: rep[k] for k in ("n_expected", "n_found", "n_missing", "n_extra",
                                                  "n_duplicate")}  # fmt: skip
        if not rep["ok"]:
            why.append(f"trace_ids: {pr._ids_detail(rep)}")
        bad_pages = [
            t.get("doc_id") for t in traces
            if len(t.get("pages") or []) != expected_pages.get(t.get("doc_id"))
            or not all(isinstance(p.get("raw_text"), str) for p in t.get("pages") or [])
        ]  # fmt: skip
        facts["trace_incomplete_docs"] = len(bad_pages)
        if bad_pages:
            why.append(f"trace_pages: {len(bad_pages)} docs with a wrong page count or no raw "
                       f"text {sorted(map(str, bad_pages))[:MAX_SAMPLE]}")  # fmt: skip
        pages = [p for t in traces for p in t.get("pages") or []]
        n_lp = sum(isinstance(p.get("field_logprobs"), list) for p in pages)
        facts["logprobs_pages"] = {"with": n_lp, "pages": len(pages)}
        if not pages or n_lp != len(pages):
            why.append(f"logprobs: field logprobs on {n_lp}/{len(pages)} pages")
        git_commits = {t.get("git_commit") for t in traces}
        if v0_sha and git_commits != {v0_sha}:
            why.append(f"trace_code_sha: traces were written by {sorted(map(str, git_commits))}")
    if decode is None:
        why.append("decode_path: not computed")
    else:
        keys = ("ok", "v0_sha", "head_sha", "files", "differing", "unresolved", "spike")
        facts["decode_path"] = {k: decode[k] for k in keys}
        if decode["differing"]:
            why.append(f"decode_path: blob differs for {decode['differing']}")
        if decode["unresolved"]:
            why.append(f"decode_path: blob unresolved for {decode['unresolved']}")
        if not decode["spike"]["ok"]:
            why.append(f"spike_diff: {decode['spike']['state']}")
    return ReuseDecision(ok=not why, reasons=why, facts=facts)


# --------------------------------------------------------------------------------------------
# Loading the v0 folder
# --------------------------------------------------------------------------------------------


def load_v0(v0_dir: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[Any] | None]:
    """``(manifest, validation_report, traces)`` of the v0 folder; None for a missing file."""
    d = Path(v0_dir)

    def js(name: str) -> dict[str, Any] | None:
        p = d / name
        if not p.is_file():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            return None
        return data if isinstance(data, dict) else None

    trace_p = d / "trace.jsonl"
    traces = spike._read_trace(trace_p) if trace_p.is_file() else None
    return js("manifest.json"), js("validation_report.json"), traces


def evaluate_reuse(
    cfg: spike.SpikeConfig, v0_dir: Path, data_root: Path, batch_size: int | None,
    shard: str = "0/1", ack_spike: bool = False, blob_fn: BlobFn = git_blob,
    diff_fn: DiffFn = git_diff_u0, head_sha: str | None = None,
) -> ReuseDecision:  # fmt: skip
    """Load everything `decide_reuse` needs from disk / git and decide."""
    man, rep, traces = load_v0(v0_dir)
    head = head_sha or spike.git_commit()
    v0_sha = str((man or {}).get("code_sha", ""))
    decode = None
    if len(v0_sha) == 40:
        decode = decode_path_report(v0_sha, head.removesuffix("+dirty"), blob_fn, diff_fn,
                                    ack_spike=ack_spike)  # fmt: skip
    return decide_reuse(
        man,
        rep,
        traces,
        pr.discover_test_docs(data_root),
        production_facts(cfg, batch_size, shard),
        decode,
        head,
    )


# --------------------------------------------------------------------------------------------
# Assembly over the v0 traces
# --------------------------------------------------------------------------------------------


def _check(ok: bool, detail: str) -> dict[str, Any]:
    return {"ok": bool(ok), "detail": detail}


def assemble_reuse(
    cfg: spike.SpikeConfig, v0_dir: Path, out_dir: Path, data_root: Path, schema_path: Path,
    ocr_cache: Path, batch_size: int, shard: str = "0/1", dev_run_dir: Path | None = None,
    expect_docs: int = pr.EXPECTED_DOCS, expect_pages: int = pr.EXPECTED_PAGES,
    expect_code_sha: str | None = None, sample_submission: Path | None = None,
    shapes_file: Path | None = None, rule_cfg: RuleConfig | None = None,
    ack_spike: bool = False, decision: ReuseDecision | None = None,
) -> dict[str, Any]:  # fmt: skip
    """The v1 submission folder from the v0 traces (R1-R3 default on). Returns the report.

    `decision` is recomputed from disk when None; a refusal raises `ReuseError` and writes nothing
    (the caller falls back to full inference). Blocking checks: counts, ids (image folder and
    sample submission), post rules, coercion, JSON Schema, the v0 determinism record, the batch
    contract against the dev run, code SHA clean and equal to the pin, the reuse decision
    (config / prompt hashes, model revision, seed, ..., decode-path fingerprint), dependencies
    recorded. On failure the predictions go to ``test_predictions.REJECTED.json``.
    """
    out_dir = Path(out_dir)
    pr.assert_outside_repo(out_dir)
    dec = decision or evaluate_reuse(cfg, v0_dir, data_root, batch_size, shard, ack_spike)
    if not dec.ok:
        raise ReuseError("REUSE REFUSED: " + " | ".join(dec.reasons))
    man, v0_report, traces = load_v0(v0_dir)
    assert man is not None and traces is not None  # decide_reuse guaranteed both
    checks: dict[str, dict[str, Any]] = {}
    pages_on_disk = pr.discover_test_docs(data_root)
    expected_ids = list(pages_on_disk)
    found_ids = [t["doc_id"] for t in traces]
    n_pages = sum(len(t["pages"]) for t in traces)
    checks["run_complete"] = _check(True, "v0 run complete (validated v0 submission, reused)")
    checks["counts"] = _check(
        len(pages_on_disk) == expect_docs and sum(pages_on_disk.values()) == expect_pages
        and len(traces) == expect_docs and n_pages == expect_pages,
        f"image folder {len(pages_on_disk)} docs / {sum(pages_on_disk.values())} pages; trace "
        f"{len(traces)} docs / {n_pages} pages; expected {expect_docs} / {expect_pages}",
    )  # fmt: skip
    ids_report = pr.check_ids(found_ids, expected_ids)
    checks["doc_ids"] = {**ids_report, "detail": pr._ids_detail(ids_report)}
    if sample_submission is not None and Path(sample_submission).is_file():
        s = pr.check_ids(list(pr._read_json(sample_submission)), expected_ids)
        checks["ids_match_sample_submission"] = _check(
            s["ok"], f"assignment sample submission: {pr._ids_detail(s)}"
        )
    rcfg = RuleConfig() if rule_cfg is None else rule_cfg
    shapes, shapes_sha, shapes_err = None, None, None
    if rcfg.r3:
        try:
            shapes, shapes_sha = postrules.load_slot_shapes(shapes_file)
        except (OSError, ValueError) as exc:  # fail closed
            shapes_err = type(exc).__name__
    predictions, rule_records, rules_summary = postrules.postprocess_traces(
        traces, rcfg, shapes, ocr_cache
    )
    note = (f"slot shapes sha256 {shapes_sha[:12]}" if shapes_sha
            else (f"slot shapes UNUSABLE ({shapes_err})" if shapes_err else "R3 off"))  # fmt: skip
    checks["post_rules"] = _check(
        shapes_err is None,
        f"switches {rcfg.as_dict()}; {note}; touched docs {rules_summary['touched_docs']}; "
        f"skipped {rules_summary['skipped']}",
    )
    try:
        predictions = coerce_predictions(predictions)
        checks["number_coercion"] = _check(True, "numeric fields are plain-number strings")
    except CoerceError as exc:
        checks["number_coercion"] = _check(False, str(exc))
    predictions, at_assembly = repair_predictions(predictions)
    in_trace = sum(len(t.get("schema_repairs") or []) for t in traces)
    repairs = {"n": in_trace + len(at_assembly), "in_trace": in_trace,
               "at_assembly": len(at_assembly), "reason": "schema_invalid_date",
               "events": at_assembly}  # fmt: skip
    checks["date_repair"] = _check(
        True, f"{repairs['n']} schema-invalid date(s) set to null ({in_trace} in the v0 traces, "
        f"{len(at_assembly)} at assembly); values are never stored",
    )  # fmt: skip
    sch = pr.validate_schema(predictions, pr._read_json(schema_path))
    checks["json_schema"] = {**sch, "detail": f"{sch['n_errors']} schema errors"}

    v0_det = man.get("determinism") or {}
    checks["determinism"] = {
        **_check(
            bool(v0_det.get("ok")) and int(v0_det.get("n_docs") or 0) > 0,
            f"REUSE: VLM determinism rests on v0's own check, re-attached: "
            f"{v0_det.get('n_identical')}/{v0_det.get('n_docs')} documents byte-identical on the "
            "second decode (see manifest.determinism_v0); assemble determinism and OCR "
            "determinism are checked by `finalize`",
        ),
        "n_docs": v0_det.get("n_docs"),
    }
    try:
        contract = pr.check_batch_contract(dev_run_dir, batch_size)
        checks["batch_size_contract"] = _check(
            True, f"v0 and v1 both batch size {batch_size}; dev contract {contract}"
        )
    except pr.BatchContractError as exc:
        checks["batch_size_contract"] = _check(False, str(exc))
    code_sha = spike.git_commit()
    checks["code_sha_clean"] = _check("+dirty" not in code_sha and code_sha != "unknown",
                                      f"code_sha {code_sha}")  # fmt: skip
    if expect_code_sha:
        checks["code_sha_is_pin"] = _check(
            code_sha.removesuffix("+dirty") == expect_code_sha,
            f"{code_sha} vs pin {expect_code_sha}",
        )
    checks["production_config"] = _check(
        True,
        "reuse conditions all hold: config hash, prompt hash, model revision, seed, output "
        "format, arm, shard, batch size equal v1's; decode-path blobs identical; spike.py "
        f"{dec.facts['decode_path']['spike']['state']}",
    )  # fmt: skip
    deps = man.get("dependencies") or {}
    missing_pk = [k for k, v in (deps.get("packages") or {}).items() if v is None]
    checks["dependencies_recorded"] = _check(
        bool(deps.get("packages")) and not missing_pk,
        f"v0 manifest dependencies present, missing packages {missing_pk}",
    )
    checks["real_model_and_stack"] = _check(
        (man.get("model") or {}).get("id") not in (None, "mock") and not missing_pk,
        f"v0 model id {(man.get('model') or {}).get('id')}",
    )
    ok = all(c["ok"] for c in checks.values())
    report = {
        "schema": 1, "ok": ok, "mode": "reuse",
        "schema_ok": checks["json_schema"]["ok"],
        "docs_found": f"{len(set(found_ids) & set(expected_ids))}/{len(expected_ids)}",
        "determinism_ok": checks["determinism"]["ok"],
        "checks": checks,
        "page_stats": pr._page_stats(traces, cfg.backend.max_new_tokens),
        "schema_repairs": repairs, "post_rules": rules_summary,
        "note": "counts, ids and key paths only; no extracted value is stored in this report",
    }  # fmt: skip
    v0_manifest_path = Path(v0_dir) / "manifest.json"
    manifest = {
        "schema": 1,
        "submission": f"{SUBMISSION_VERSION}_{code_sha[:7]}",
        "mode": "reuse",
        "run_id": man.get("run_id"),
        "code_sha": code_sha,
        "model": man["model"],
        "config": {**man["config"], "file": f"configs/spike_{cfg.name}.yaml"},
        "seed": man["seed"],
        "prompt": man.get("prompt"),
        "batch_size": man["batch_size"],
        "batch_size_source": f"v0 ({man.get('batch_size_source')})",
        "batch_size_contract": checks["batch_size_contract"]["detail"],
        "bench": man.get("bench"),
        "shard": man["shard"],
        "docs": {
            "n_docs": len(traces),
            "n_pages": n_pages,
            "docs_sha": (man.get("docs") or {}).get("docs_sha"),
        },  # fmt: skip
        "schema_file": {"name": Path(schema_path).name, "sha256": pr.sha256_file(schema_path)},
        "sample_submission_checked": bool(sample_submission and Path(sample_submission).is_file()),
        "phase3_rules": pr.phase3_rules(),
        "post_rules": {
            **rules_summary,
            "slot_shapes_file": postrules.SHAPES_FILE,
            "slot_shapes_sha256": shapes_sha,
        },  # fmt: skip
        "schema_repairs": {k: repairs[k] for k in ("n", "in_trace", "at_assembly", "reason")},
        "dependencies": man.get("dependencies"),
        "determinism": {
            k: v0_det.get(k) for k in ("ok", "n_docs", "n_identical", "doc_ids", "rule", "seed")
        },  # fmt: skip
        "determinism_v0": {
            "source": f"{v0_manifest_path.parent.name}/manifest.json (determinism)",
            "source_sha256": pr.sha256_file(v0_manifest_path),
            "validation_report_sha256": pr.sha256_file(Path(v0_dir) / "validation_report.json"),
            "trace_sha256": pr.sha256_file(Path(v0_dir) / "trace.jsonl"),
            "report": v0_det,
            "note": "VLM determinism in reuse mode rests on v0's own check; not re-run here.",
        },
        "timings": {
            **{
                k: (man.get("timings") or {}).get(k)
                for k in ("model_time_s", "wall_clock_s", "sessions", "smoke_s")
            },
            "from_v0_run": True,
        },  # fmt: skip
        "smoke": man.get("smoke"),
        "reuse": {
            "v0_dir": Path(v0_dir).name,
            "v0_code_sha": man["code_sha"],
            "decision": dec.facts,
            "code_identity": "replaced by the decode-path blob fingerprint + config/prompt hashes",
            "spike_diff_acked": bool(dec.facts["decode_path"]["spike"].get("acked")),
        },
        "output_dir": out_dir.name,
        "files": list(OUT_FILES),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in (pr.REJECTED_NAME, "test_predictions.json"):
        (out_dir / stale).unlink(missing_ok=True)
    pr._write_json(out_dir / (pr.OUT_FILES[0] if ok else pr.REJECTED_NAME), predictions)
    shutil.copyfile(Path(v0_dir) / "trace.jsonl", out_dir / "trace.jsonl")
    pr._write_json(out_dir / "manifest.json", manifest)
    pr._write_json(out_dir / "validation_report.json", report)
    (out_dir / "rules.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rule_records), encoding="utf-8", newline="\n"
    )
    return report


# --------------------------------------------------------------------------------------------
# finalize: the v1-only blocking checks (both paths) and the assemble-twice comparison
# --------------------------------------------------------------------------------------------


def compare_outputs(a: Path, b: Path) -> dict[str, Any]:
    """Byte comparison of the deterministic outputs of two assemblies in fresh processes.

    Compared: the predictions file (same name in both: submittable or REJECTED), ``trace.jsonl``
    and ``rules.jsonl``. The report and manifest are left out on purpose: `finalize` rewrites
    them in place, so a later re-run of it would compare its own edits.
    """
    a, b = Path(a), Path(b)
    pred = next(
        (n for n in (pr.OUT_FILES[0], pr.REJECTED_NAME) if (a / n).is_file() and (b / n).is_file()),
        None,
    )
    names = [n for n in (pred, "trace.jsonl", "rules.jsonl") if n]
    same, diff = [], []
    for n in names:
        if (a / n).is_file() and (b / n).is_file() and (a / n).read_bytes() == (b / n).read_bytes():
            same.append(n)
        else:
            diff.append(n)
    if pred is None:
        diff.append("predictions (different names or missing)")
    return {
        "ok": not diff,
        "identical": same,
        "different_or_missing": diff,
        "sha256": {n: pr.sha256_file(a / n) for n in same},
    }


def finalize(
    out_dir: Path, shapes_file: Path, mode: str, expect_pages: int = pr.EXPECTED_PAGES,
    rerun_dir: Path | None = None, recheck_path: Path | None = None,
) -> dict[str, Any]:  # fmt: skip
    """Add the v1-only checks to ``validation_report.json``; withhold the name if one fails.

    Checks: ``v1_rule_switches`` (R1, R2, R3 all True), ``v1_no_rule_skipped`` (no R2/no_ocr,
    R2/ocr_incomplete, R3/no_shapes, nor a disabled rule), ``v1_shapes_sha256`` (the manifest's
    shapes hash equals ``meta/slot_shapes.json``), ``ocr_cache_complete`` (``ocr_timing.json``:
    all pages), ``ocr_determinism`` (the recheck report ok), and in reuse mode
    ``assemble_twice`` (the rerun folder is byte-identical). A failing check flips ``ok``, moves
    ``test_predictions.json`` to ``test_predictions.REJECTED.json`` and the manifest's
    ``submission`` is set to the folder's v1 name (``predict assemble`` labels it v0).
    """
    out = Path(out_dir)
    man = pr._read_json(out / "manifest.json")
    rep = pr._read_json(out / "validation_report.json")
    post = man.get("post_rules") or {}
    skipped = dict(post.get("skipped") or {})
    checks = rep["checks"]
    checks["v1_rule_switches"] = _check(
        post.get("switches") == {"r1": True, "r2": True, "r3": True},
        f"switches {post.get('switches')}",
    )
    bad = sorted(k for k in skipped if k in BAD_SKIPS or k.endswith("/disabled"))
    checks["v1_no_rule_skipped"] = _check(
        not bad, f"skipped {skipped}" + (f"; BLOCKING {bad}" if bad else "")
    )
    want_sha = pr.sha256_file(shapes_file)
    checks["v1_shapes_sha256"] = _check(
        post.get("slot_shapes_sha256") == want_sha,
        f"manifest {str(post.get('slot_shapes_sha256'))[:12]} vs {SHAPES_REL} {want_sha[:12]}",
    )
    timing_p = out / "ocr_timing.json"
    timing = pr._read_json(timing_p) if timing_p.is_file() else None
    checks["ocr_cache_complete"] = _check(
        bool(timing) and timing.get("pages") == expect_pages,
        f"ocr_timing.json pages {(timing or {}).get('pages')} (expected {expect_pages}), device "
        f"{(timing or {}).get('device')}",
    )
    recheck = pr._read_json(recheck_path) if recheck_path and Path(recheck_path).is_file() else None
    checks["ocr_determinism"] = _check(
        bool(recheck) and recheck.get("ok") is True,
        f"{(recheck or {}).get('n_text_identical')}/{(recheck or {}).get('n_pages')} pages "
        f"text-identical on a fresh OCR, {(recheck or {}).get('n_content_identical')} also "
        "content-identical (all but seconds)",
    )
    if mode == "reuse":
        cmp = compare_outputs(out, rerun_dir) if rerun_dir and Path(rerun_dir).is_dir() else None
        checks["assemble_twice"] = _check(
            bool(cmp) and cmp["ok"],
            "no rerun folder"
            if cmp is None
            else f"identical {cmp['identical']}; different {cmp['different_or_missing']}",
        )
    ok = all(c["ok"] for c in checks.values())
    rep["ok"], rep["mode"] = ok, mode
    man["submission"] = f"{SUBMISSION_VERSION}_{out.name.split('_', 1)[-1]}"
    man["mode"] = mode
    man["files"] = list(OUT_FILES)
    man["ocr"] = {"timing": timing, "recheck": recheck}
    pred, rejected = out / pr.OUT_FILES[0], out / pr.REJECTED_NAME
    if not ok and pred.is_file():
        pred.replace(rejected)  # never leave the submittable name on an unchecked file
    pr._write_json(out / "manifest.json", man)
    pr._write_json(out / "validation_report.json", rep)
    return rep


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Argument parser of ``python -m shipdoc.reuse``."""
    p = argparse.ArgumentParser(prog="python -m shipdoc.reuse")
    sub = p.add_subparsers(dest="stage", required=True)

    def common(s: argparse.ArgumentParser) -> None:
        s.add_argument("--config", default="configs/spike_qwen35_4b_img_only.yaml")
        s.add_argument("--v0-dir", type=Path, required=True)
        s.add_argument("--batch-size", type=int, required=True)
        s.add_argument("--shard", default="0/1")
        s.add_argument("--ack-spike-diff", action="store_true")
        s.add_argument("--data-root", type=Path, default=None)

    s = sub.add_parser("check", help="decide reuse; exit 0 = allowed, 3 = refused (full path)")
    common(s)
    s.add_argument("--out", type=Path, default=None, help="decision JSON to write")
    s = sub.add_parser("assemble", help="assemble the v1 folder from the v0 traces (CPU)")
    common(s)
    s.add_argument("--out-dir", type=Path, required=True)
    s.add_argument("--schema", type=Path, required=True)
    s.add_argument("--ocr-cache", type=Path, required=True)
    s.add_argument("--dev-run-dir", type=Path, default=None)
    s.add_argument("--expect-docs", type=int, default=pr.EXPECTED_DOCS)
    s.add_argument("--expect-pages", type=int, default=pr.EXPECTED_PAGES)
    s.add_argument("--expect-code-sha", default=None)
    s.add_argument("--sample-submission", type=Path, default=None)
    s.add_argument("--shapes-file", type=Path, default=None)
    s = sub.add_parser("finalize", help="the v1-only checks on a submission folder")
    s.add_argument("--out-dir", type=Path, required=True)
    s.add_argument("--mode", choices=["reuse", "full"], required=True)
    s.add_argument("--shapes-file", type=Path, default=paths.REPO_ROOT / SHAPES_REL)
    s.add_argument("--expect-pages", type=int, default=pr.EXPECTED_PAGES)
    s.add_argument("--rerun-dir", type=Path, default=None)
    s.add_argument("--recheck", type=Path, default=None)
    s = sub.add_parser("compare", help="byte comparison of two assembled folders")
    s.add_argument("--a", type=Path, required=True)
    s.add_argument("--b", type=Path, required=True)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """CLI. Exit 0 ok, 1 a check failed, 3 reuse refused (not an error: run full inference)."""
    args = build_parser().parse_args(argv)
    try:
        if args.stage == "compare":
            c = compare_outputs(args.a, args.b)
            print(
                f"compare: ok={c['ok']} identical {c['identical']} "
                f"different {c['different_or_missing']}"
            )
            return 0 if c["ok"] else 1
        if args.stage == "finalize":
            rep = finalize(args.out_dir, args.shapes_file, args.mode, args.expect_pages,
                           args.rerun_dir, args.recheck)  # fmt: skip
            failed = [k for k, c in rep["checks"].items() if not c["ok"]]
            print(f"finalize ({args.mode}): ok={rep['ok']} failed checks {failed}")
            return 0 if rep["ok"] else 1
        cfg = spike.load_config(args.config)
        data = Path(args.data_root) if args.data_root else paths.data_dir()
        dec = evaluate_reuse(cfg, args.v0_dir, data, args.batch_size, args.shard,
                             args.ack_spike_diff)  # fmt: skip
        if args.stage == "check":
            if args.out:
                pr._write_json(args.out, {"ok": dec.ok, "reasons": dec.reasons, "facts": dec.facts})
            head = dec.facts.get("head_sha")
            v0 = dec.facts.get("v0_code_sha")
            if v0 and head:
                print(git_diff_stat(str(v0), str(head).removesuffix("+dirty")))
            print(f"reuse decision: {'ALLOWED' if dec.ok else 'REFUSED'}")
            for r in dec.reasons:
                print(f"  refusal: {r}")
            return 0 if dec.ok else 3
        rep = assemble_reuse(
            cfg, args.v0_dir, args.out_dir, data, args.schema, args.ocr_cache, args.batch_size,
            args.shard, args.dev_run_dir, args.expect_docs, args.expect_pages,
            args.expect_code_sha, args.sample_submission, args.shapes_file,
            ack_spike=args.ack_spike_diff, decision=dec,
        )  # fmt: skip
        failed = [k for k, c in rep["checks"].items() if not c["ok"]]
        print(f"assemble (reuse): ok={rep['ok']} docs {rep['docs_found']} failed checks {failed} "
              f"-> {args.out_dir}")  # fmt: skip
        return 0 if rep["ok"] else 1
    except (ReuseError, pr.PredictError) as exc:
        print(f"shipdoc.reuse {args.stage}: REFUSED: {exc}", file=sys.stderr)
        return 3 if isinstance(exc, ReuseError) else 1


if __name__ == "__main__":
    raise SystemExit(main())
