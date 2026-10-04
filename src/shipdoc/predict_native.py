"""Notebook 04c_predict_test_native: both pre-registered final-system paths at NATIVE resolution.

GG decision 2026-10-03: native (``configs/spike_qwen35_4b_img_only_native.yaml``, ``max_pixels``
2,196,480) is the production resolution, and GG wants both deliverable paths of the pre-registered
final-system rule (spec section 11) ready. ``MODEL`` selects one:

``zs``   ('v1.5') the BASE model, no adapter, on the 200 test documents / 280 pages. The v0 traces
         are at 1260 tokens and can NOT be reused (their config hash differs), so this path runs
         the VLM: smoke gate, batch size = the 02n bench's stored size (the batch contract of 04),
         resumable run, determinism pass, then OCR (the 04b cache), R1-R3 and the same blocking
         checks as 04b. Output folder ``v15_<sha7>``.
``ft``   ('v2') the FINAL native adapter (verified as in 04c + the resolution gate), merged
         inference at native, the same checks as 04c. Its review flags need the ZS native test
         traces of the ``zs`` path (``ZS_TEST_DIR``, accepted only when VALIDATED, at the native
         config hash and strictly reusable); no zero-shot inference ever runs on this path.
         Output folder ``v2n_<sha7>``.

Everything that is shared is imported unchanged (``predict`` stages, ``predict_ft``, ``flags``,
``reuse``, ``ocr_stage``, ``nativerun``); this module adds the refusals and the glue:

``check``        CPU, before anything runs: the config is the native one; the 02n run carries the
                 native config hash and its batch size resolves (the ZS test batch must EQUAL it);
                 ``ft``: the resolution gate on the adapter and the ``ZS_TEST_DIR`` conditions;
                 ``zs``: an optional v0 folder MUST be refused by the reuse decision.
``estimate``     ESTIMATE of T4 hours / CU for the chosen path (nothing in it is measured).
``verify`` / ``infer`` / ``determinism``   (ft) the resolution gate, then the unchanged
                 ``predict_ft`` stage; ``infer`` stamps the gate into ``ft_run.json``.
``finalize``     the 04b (zs) or 04c (ft) blocking checks + the native checks; the manifest says
                 ``submission = v15_<sha7>`` / ``v2n_<sha7>``.
``flags`` / ``check-flags``   ``shipdoc.flags`` with the calibrator refused unless it carries the
                 native config hash (``run_config_hash``, written by
                 ``scripts/freeze_calibrator.py``), then the value-free / structure checks of
                 ``review_flags.json``.

Test data policy: nothing here prints or stores an extracted value; reports carry counts, ids,
hashes and key paths only. What is UNVERIFIED on a GPU: the whole of both paths at native (the
batch size, VRAM, the merge, every timing); only the CPU stages and the mock-backend pipeline are
tested locally.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from shipdoc import flags, nativerun, oof, paths, predict_ft, reuse, runcompat, spike
from shipdoc import predict as pr
from shipdoc.runmeta import read_manifest

MODELS = ("zs", "ft")
PREFIX = {"zs": "v15", "ft": "v2n"}  # submission folder prefix of each path
NATIVE_CONFIG = nativerun.NATIVE_CONFIG
ALLOWED_DOC_KEYS = frozenset({"doc_type", "header", "line_items"})
RUN_FILES = (*reuse.OUT_FILES, flags.FLAGS_NAME)
MODEL_KIND = {
    "zs": "zero-shot (base model, no adapter, native resolution) + R1-R3",
    "ft": "fine-tuned (final native adapter, merged into fp16 weights) + R1-R3",
}


class NativeTestError(RuntimeError):
    """A native-resolution precondition of the test notebook failed. No value in a message."""


def _check(ok: bool, detail: str) -> dict[str, Any]:
    return {"ok": bool(ok), "detail": detail}


def _need_model(model: str) -> str:
    if model not in MODELS:
        raise NativeTestError(f"MODEL must be one of {MODELS}, got {model!r}")
    return model


# --------------------------------------------------------------------------------------------
# The native config, the 02n run and the batch contract
# --------------------------------------------------------------------------------------------


def load_native_config(path: Path | str) -> spike.SpikeConfig:
    """The inference config, refused unless it is the native production config.

    ``max_pixels`` must be 2,196,480 and the name the native one: the 1260-token config (the
    default of every other notebook) is refused with the reason.
    """
    cfg = spike.load_config(Path(path))
    if cfg.backend.max_pixels != nativerun.NATIVE_MAX_PIXELS or cfg.name != (
        nativerun.NATIVE_CONFIG_NAME
    ):
        raise NativeTestError(
            f"config {cfg.name!r} has max_pixels {cfg.backend.max_pixels}: this notebook runs only "
            f"{nativerun.NATIVE_CONFIG_NAME!r} (max_pixels {nativerun.NATIVE_MAX_PIXELS}); pass "
            f"--config {NATIVE_CONFIG}"
        )
    return cfg


def assert_native_hash(
    manifest: Mapping[str, Any] | None, cfg: spike.SpikeConfig, label: str
) -> str:
    """The manifest's config hash, refused unless it equals the native config's (fail closed)."""
    native = {"config": {"name": cfg.name, "hash": cfg.config_hash}}
    try:
        runcompat.assert_same_resolution(native, manifest, "native config", label)
    except runcompat.ResolutionMismatchError as exc:
        raise NativeTestError(str(exc)) from exc
    return str(runcompat.manifest_config_hash(manifest))


def check_zs_run(zs_run_dir: Path, cfg: spike.SpikeConfig) -> dict[str, Any]:
    """The 02n zero-shot run: complete, at the native config hash; returns its manifest."""
    d = Path(zs_run_dir)
    if not d.is_dir():
        raise NativeTestError(f"ZS_RUN_DIR {d.name}: not a folder (the 02n run, e.g. from ZS_SHA7)")
    assert_native_hash(read_manifest(d), cfg, f"zero-shot run `{d.name}`")
    try:
        man, _ = oof.load_zero_shot(d)
    except oof.OofError as exc:
        raise NativeTestError(str(exc)) from exc
    return man


def resolve_batch(model: str, zs_run_dir: Path, requested: int | None) -> dict[str, Any]:
    """The batch size of the test run: the one stored with the 02n run (``None``), or an override.

    ``zs``: an override that differs from the stored size is REFUSED (the batch contract of 04:
    greedy outputs are only comparable at equal batch size). ``ft``: allowed and said loudly (the
    guard on the merged model decides, and may fall back to 1). Refused when the 02n run stores no
    size and none is given (`oof.resolve_batch_size`).
    """
    _need_model(model)
    try:
        chosen = oof.resolve_batch_size(Path(zs_run_dir), requested)
    except oof.OofError as exc:
        raise NativeTestError(str(exc)) from exc
    if chosen["differs_from_zero_shot"] and model == "zs":
        raise NativeTestError(
            f"BATCH_SIZE {chosen['batch_size']} differs from the 02n run's {chosen['stored']}: the "
            "batch-size contract refuses it (set BATCH_SIZE = None)"
        )
    return chosen


# --------------------------------------------------------------------------------------------
# ZS_TEST_DIR (ft path) and the v0 traces (zs path)
# --------------------------------------------------------------------------------------------

EvaluateFn = Callable[..., reuse.ReuseDecision]


def check_zs_test_dir(
    zs_test_dir: Path,
    cfg: spike.SpikeConfig,
    batch_size: int | None,
    *,
    data_root: Path | None = None,
    ack_spike_diff: bool = False,
    evaluate: EvaluateFn | None = None,
    skip_decision: bool = False,
) -> dict[str, Any]:
    """Accept the ZS native test folder (``v15_<sha7>``) for the agreement features, or refuse.

    Required: the folder and its manifest exist; the manifest's config hash is the native config's;
    the manifest names a ``v15_`` full-mode submission of a NON-fine-tuned model; the validation
    report is VALIDATED (``ok`` and every check ok); ``test_predictions.json`` exists (a REJECTED
    file is not enough); its batch size equals `batch_size` (the 02n run's; None = not compared);
    and the strict reuse decision (`reuse.evaluate_reuse` at the native config: prompt hash, model
    revision, seed, decode-path blobs, ...) allows it (`evaluate` None = `reuse.evaluate_reuse`
    looked up now; `skip_decision` skips it, for a re-check after it was made). Anything else
    raises `NativeTestError` naming every problem. No zero-shot inference ever runs on the ft path.
    """
    d = Path(zs_test_dir)
    if not d.is_dir():
        raise NativeTestError(
            f"ZS_TEST_DIR {d.name} is not a folder: run this notebook with MODEL = 'zs' first "
            f"(it writes {PREFIX['zs']}_<sha7>) and set ZS_TEST_DIR to it. No zero-shot inference "
            "runs on the ft path."
        )
    man = pr._read_json(d / "manifest.json") if (d / "manifest.json").is_file() else None
    rep = (
        pr._read_json(d / "validation_report.json")
        if (d / "validation_report.json").is_file()
        else None
    )
    if man is None or rep is None:
        raise NativeTestError(f"ZS_TEST_DIR {d.name}: no manifest.json / validation_report.json")
    problems: list[str] = []
    h = runcompat.manifest_config_hash(man)
    if h != cfg.config_hash:
        problems.append(
            f"config hash {h} is not the native {cfg.config_hash} (a v0 / v1 folder at 1260 "
            "tokens is refused)"
        )
    if not str(man.get("submission", "")).startswith(f"{PREFIX['zs']}_"):
        problems.append(
            f"manifest submission {man.get('submission')!r} is not a {PREFIX['zs']}_ folder"
        )
    if man.get("mode") != "full":
        problems.append(f"manifest mode {man.get('mode')!r} is not 'full' (the VLM must have run)")
    model_id = str((man.get("model") or {}).get("id") or "")
    if flags.LORA_MARK in model_id or man.get("ft"):
        problems.append("the folder is a fine-tuned run, not the zero-shot one")
    failed = [k for k, c in (rep.get("checks") or {}).items() if not c.get("ok")]
    if rep.get("ok") is not True or failed:
        problems.append(f"not VALIDATED (report ok {rep.get('ok')!r}, failed checks {failed})")
    if not (d / "test_predictions.json").is_file():
        problems.append("no test_predictions.json (a REJECTED folder is not accepted)")
    if not isinstance(man.get("batch_size"), int):
        problems.append("the manifest records no batch size")
    elif batch_size is not None and man["batch_size"] != batch_size:
        problems.append(f"batch size {man.get('batch_size')} is not the 02n run's {batch_size}")
    if problems:
        raise NativeTestError(f"ZS_TEST_DIR {d.name} refused: " + "; ".join(problems))
    facts: dict[str, Any] = {}
    if not skip_decision:
        data = Path(data_root) if data_root else paths.data_dir()
        decide = evaluate or reuse.evaluate_reuse
        dec = decide(cfg, d, data, int(man["batch_size"]), "0/1", ack_spike_diff)
        if not dec.ok:
            raise NativeTestError(
                f"ZS_TEST_DIR {d.name} refused by the strict reuse decision: "
                + " | ".join(dec.reasons)
            )
        facts = dec.facts
    return {
        "dir": d.name,
        "config_hash": h,
        "batch_size": man.get("batch_size"),
        "code_sha": man.get("code_sha"),
        "trace_sha256": pr.sha256_file(d / "trace.jsonl"),
        "predictions_sha256": pr.sha256_file(d / "test_predictions.json"),
        "decode_path_files": (facts.get("decode_path") or {}).get("files"),
    }


def assert_v0_refused(
    v0_dir: Path,
    cfg: spike.SpikeConfig,
    batch_size: int,
    *,
    data_root: Path | None = None,
    evaluate: EvaluateFn | None = None,
) -> list[str]:
    """The reuse decision on a v0 folder at the NATIVE config must REFUSE; returns its reasons.

    The v0 traces are at 1260 tokens (config hash differs), so the zs path runs the VLM. If the
    decision ever allowed them this raises: reusing 1260-token traces as native output is exactly
    the mistake this notebook exists to prevent.
    """
    data = Path(data_root) if data_root else paths.data_dir()
    dec = (evaluate or reuse.evaluate_reuse)(cfg, Path(v0_dir), data, batch_size, "0/1", False)
    if dec.ok or not any(r.startswith("config_hash") for r in dec.reasons):
        raise NativeTestError(
            "the reuse decision did not refuse the v0 traces on the config hash: it must (they are "
            f"at another resolution); reasons {dec.reasons}"
        )
    return list(dec.reasons)


# --------------------------------------------------------------------------------------------
# The adapter (ft path)
# --------------------------------------------------------------------------------------------


def check_adapter_native(
    adapter_dir: Path, cfg: spike.SpikeConfig, out: Callable[[str], None] = print
) -> None:
    """The resolution gate (`nativerun.assert_adapter_resolution`, fails closed)."""
    try:
        nativerun.assert_adapter_resolution(Path(adapter_dir), cfg, out=out)
    except nativerun.NativeError as exc:
        raise NativeTestError(str(exc)) from exc


# --------------------------------------------------------------------------------------------
# The calibrator
# --------------------------------------------------------------------------------------------


def check_calibrator(path: Path | str | None, cfg: spike.SpikeConfig, model: str) -> dict[str, Any]:
    """Refuse a calibrator that is missing, of the other arm, or not frozen for the native config.

    ``None`` = no CALIBRATOR_FILE: the message says how to freeze one at native. The artifact must
    carry ``run_config_hash`` equal to the native config hash (``scripts/freeze_calibrator.py``
    records it from the run manifest): the 1260-token ``meta/calibrator_zs.json`` carries none and
    is refused here. The full load (sklearn version, parity probe) is `flags.load_calibrator`.
    """
    _need_model(model)
    if path is None:
        raise NativeTestError(
            f"no CALIBRATOR_FILE: the review flags are not computed. Refreeze at native with "
            f"`scripts/freeze_calibrator.py --arm {model}` "
            + (
                "after 02n and calibrate_v2 on the native run"
                if model == "zs"
                else "(three native OOF runs + calibrate_v3 on them)"
            )
            + ", commit or upload the artifact and set CALIBRATOR_FILE."
        )
    p = Path(path)
    if not p.is_file():
        raise NativeTestError(f"CALIBRATOR_FILE {p} does not exist")
    data = json.loads(p.read_text(encoding="utf-8"))
    if data.get("artifact") != flags.ARTIFACT_KIND:
        raise NativeTestError(f"{p.name} is not a calibrator artifact")
    if data.get("arm") != model:
        raise NativeTestError(f"{p.name} is the {data.get('arm')!r} calibrator, MODEL is {model!r}")
    frozen_for = data.get("run_config_hash")
    if frozen_for != cfg.config_hash:
        raise NativeTestError(
            f"{p.name} was frozen for config hash {frozen_for!r}, not the native "
            f"{cfg.config_hash}: its probabilities are for another resolution (a calibrator with "
            "no recorded hash, like meta/calibrator_zs.json, is refused). Refreeze at native "
            f"with `scripts/freeze_calibrator.py --arm {model}`."
        )
    return {"file": p.name, "arm": model, "run_config_hash": frozen_for}


# --------------------------------------------------------------------------------------------
# finalize: the 04b / 04c checks + the native checks
# --------------------------------------------------------------------------------------------


def finalize(
    model: str,
    out_dir: Path,
    *,
    cfg: spike.SpikeConfig,
    run_dir: Path,
    shapes_file: Path,
    recheck_path: Path | None,
    expect_pages: int = pr.EXPECTED_PAGES,
    zs_test_dir: Path | None = None,
) -> dict[str, Any]:
    """`reuse.finalize` (zs) / `predict_ft.finalize_v2` (ft) plus the native checks.

    Added checks, all blocking: ``native_config`` (the run's config hash and name are the native
    config's), ``native_predictions_strict`` (no key outside doc_type / header / line_items),
    and per path ``native_zero_shot_model`` (no adapter anywhere) or ``native_adapter_resolution``
    (the adapter passed the resolution gate in the infer stage, stamped in ``ft_run.json``) and
    ``native_zero_shot_test_traces`` (the ZS test folder is VALIDATED at the native hash; its
    hashes are recorded). The manifest's ``submission`` is the folder name (``v15_`` / ``v2n_``).
    """
    _need_model(model)
    out = Path(out_dir)
    if model == "ft":
        if zs_test_dir is None:
            raise NativeTestError("ft finalize needs --zs-test-dir (the v15 folder)")
        predict_ft.finalize_v2(
            out, run_dir=run_dir, shapes_file=shapes_file, v0_dir=zs_test_dir,
            recheck_path=recheck_path, expect_pages=expect_pages,
        )  # fmt: skip
    else:
        reuse.finalize(out, shapes_file, "full", expect_pages, None, recheck_path)
    man = pr._read_json(out / "manifest.json")
    rep = pr._read_json(out / "validation_report.json")
    checks = rep["checks"]
    got_hash = (man.get("config") or {}).get("hash")
    checks["native_config"] = _check(
        got_hash == cfg.config_hash
        and (man.get("config") or {}).get("name") == cfg.name
        and man.get("seed") == 42,
        f"run config {(man.get('config') or {}).get('name')} hash {got_hash} vs native "
        f"{cfg.name} {cfg.config_hash}",
    )
    native: dict[str, Any] = {
        "notebook": "04c_predict_test_native",
        "model": model,
        "config": cfg.name,
        "config_hash": cfg.config_hash,
        "max_pixels": cfg.backend.max_pixels,
    }
    if model == "zs":
        model_id = str((man.get("model") or {}).get("id") or "")
        checks["native_zero_shot_model"] = _check(
            flags.LORA_MARK not in model_id and not man.get("ft"),
            f"model id {model_id}; no adapter in the manifest: {not man.get('ft')}",
        )
    else:
        ft_path = Path(run_dir) / predict_ft.FT_RUN_FILE
        ft = pr._read_json(ft_path) if ft_path.is_file() else {}
        nr = ft.get("native_resolution") or {}
        checks["native_adapter_resolution"] = _check(
            nr.get("ok") is True and nr.get("max_pixels") == cfg.backend.max_pixels,
            f"resolution gate stamped in ft_run.json: ok {nr.get('ok')}, max_pixels "
            f"{nr.get('max_pixels')}",
        )
        try:
            # the strict reuse decision was made by `check` / `flags` (it needs git); not repeated
            native["zs_test"] = check_zs_test_dir(Path(zs_test_dir), cfg, None, skip_decision=True)
            checks["native_zero_shot_test_traces"] = _check(
                True, f"{native['zs_test']['dir']} VALIDATED at the native config hash"
            )
        except NativeTestError as exc:
            checks["native_zero_shot_test_traces"] = _check(False, str(exc))
    pred_name = pr.OUT_FILES[0] if (out / pr.OUT_FILES[0]).is_file() else pr.REJECTED_NAME
    preds = pr._read_json(out / pred_name)
    bad = [d for d, v in preds.items() if not isinstance(v, dict) or set(v) - ALLOWED_DOC_KEYS]
    checks["native_predictions_strict"] = _check(
        not bad, f"documents with a key outside doc_type / header / line_items: {len(bad)}"
    )
    ok = all(c["ok"] for c in checks.values())
    rep["ok"], rep["mode"] = ok, "full"
    man["submission"] = out.name  # the underlying finalizers label every folder v1_ / v2_
    man["model_kind"] = MODEL_KIND[model]
    man["native"] = native
    man["files"] = list(RUN_FILES)
    pred = out / pr.OUT_FILES[0]
    if not ok and pred.is_file():
        pred.replace(
            out / pr.REJECTED_NAME
        )  # never leave the submittable name on an unchecked file
    pr._write_json(out / "manifest.json", man)
    pr._write_json(out / "validation_report.json", rep)
    return rep


# --------------------------------------------------------------------------------------------
# flags
# --------------------------------------------------------------------------------------------


def run_flags(
    model: str,
    *,
    cfg: spike.SpikeConfig,
    config_path: Path,
    submission_dir: Path,
    calibrator: Path | str | None,
    ocr_cache: Path,
    zs_test_dir: Path | None,
    batch_size: int,
    field_target: float = flags.PRIMARY_TARGET,
    doc_target: float = flags.PRIMARY_TARGET,
    expect_docs: int = pr.EXPECTED_DOCS,
    ack_spike_diff: bool = False,
    data_root: Path | None = None,
    shapes_file: Path | None = None,
    say: Callable[[str], None] = print,
) -> dict[str, Any]:
    """`flags.run_stage` after the native refusals; writes ``review_flags.json`` only.

    Refused before anything is computed: no / wrong-arm / wrong-hash calibrator
    (`check_calibrator`); a submission manifest that is not at the native config hash;
    ``ft``: a ``ZS_TEST_DIR`` that fails `check_zs_test_dir`. The submission's predictions file is
    never touched (`flags.run_stage` only reads it).
    """
    _need_model(model)
    cal = check_calibrator(calibrator, cfg, model)
    sub = Path(submission_dir)
    man_path = sub / "manifest.json"
    if not man_path.is_file():
        raise NativeTestError(f"{sub.name}: no manifest.json")
    assert_native_hash(pr._read_json(man_path), cfg, f"submission `{sub.name}`")
    zs_dir: Path | None = None
    if model == "ft":
        if zs_test_dir is None:
            raise NativeTestError("the ft flags need ZS_TEST_DIR (the v15 folder)")
        check_zs_test_dir(zs_test_dir, cfg, batch_size, data_root=data_root,
                          ack_spike_diff=ack_spike_diff)  # fmt: skip
        zs_dir = Path(zs_test_dir)
    say(f"calibrator {cal['file']} ({model}) frozen for the native config hash {cfg.config_hash}")
    return flags.run_stage(
        submission_dir=sub, calibrator=Path(str(calibrator)), ocr_cache=Path(ocr_cache),
        out=sub / flags.FLAGS_NAME, arm=model, zs_v0_dir=zs_dir, batch_size=batch_size,
        ack_spike_diff=ack_spike_diff, config=Path(config_path), shapes_file=shapes_file,
        data_root=data_root, field_target=field_target, doc_target=doc_target,
        expect_docs=expect_docs, say=say,
    )  # fmt: skip


def check_flags(
    out_dir: Path,
    *,
    calibrator: Path,
    cfg: spike.SpikeConfig,
    field_target: float,
    doc_target: float,
    expect_docs: int = pr.EXPECTED_DOCS,
) -> dict[str, Any]:
    """`predict_ft.check_flags` + the native checks (exact id count, calibrator hash, strict file).

    Added: ``flags_exact_docs`` (the flags file holds exactly `expect_docs` ids and the predictions
    exactly the same ids), ``flags_calibrator_native`` (the calibrator carries the native config
    hash) and ``flags_not_in_predictions`` (``test_predictions.json`` has no key outside
    doc_type / header / line_items: the flags live ONLY in ``review_flags.json``).
    """
    out = Path(out_dir)
    rep = predict_ft.check_flags(
        out, calibrator=Path(calibrator), field_target=field_target, doc_target=doc_target
    )
    fpath, ppath = out / flags.FLAGS_NAME, out / pr.OUT_FILES[0]
    doc = pr._read_json(fpath) if fpath.is_file() else {}
    preds = pr._read_json(ppath) if ppath.is_file() else {}
    f_ids, p_ids = sorted(doc.get("docs") or {}), sorted(preds)
    cks = rep["flags_checks"]
    cks["flags_exact_docs"] = _check(
        len(f_ids) == expect_docs and f_ids == p_ids,
        f"{len(f_ids)} flagged / {len(p_ids)} predicted ids (expected {expect_docs}, equal sets)",
    )
    data = json.loads(Path(calibrator).read_text(encoding="utf-8"))
    cks["flags_calibrator_native"] = _check(
        data.get("run_config_hash") == cfg.config_hash,
        f"calibrator run_config_hash {data.get('run_config_hash')!r} vs native {cfg.config_hash}",
    )
    bad = [d for d, v in preds.items() if not isinstance(v, dict) or set(v) - ALLOWED_DOC_KEYS]
    cks["flags_not_in_predictions"] = _check(
        not bad and bool(preds), f"documents with a key outside the schema: {len(bad)}"
    )
    inp = doc.get("inputs") or {}
    rpath = out / "rules.jsonl"
    rules_ok = (
        rpath.is_file() and inp.get("rules_jsonl_sha256") == pr.sha256_file(rpath)
        if rpath.is_file() or inp.get("rules_jsonl_sha256")
        else flags._all_rules_off(out / "manifest.json")
    )
    cks["flags_rules_jsonl"] = _check(
        rules_ok,
        "rules.jsonl is in the folder and equals the one the flags were computed against "
        "(inputs.rules_jsonl_sha256); absent only if every rule is recorded OFF. A flags file "
        "written by an older stage (no such key) fails this check",
    )
    man = pr._read_json(out / "manifest.json")
    sub_sha = inp.get("submission_code_sha")
    cks["flags_submission_code_sha"] = _check(
        isinstance(sub_sha, str) and bool(sub_sha) and sub_sha == man.get("code_sha"),
        f"inputs.submission_code_sha {str(sub_sha)[:12]} vs manifest code_sha "
        f"{str(man.get('code_sha'))[:12]}; inputs.local_head_sha "
        f"{str(inp.get('local_head_sha'))[:12]} is the checkout that ran the stage (informational)",
    )
    rep["flags_ok"] = all(c["ok"] for c in cks.values())
    if isinstance(man.get("review_flags"), dict):
        man["review_flags"]["ok"] = rep["flags_ok"]
        pr._write_json(out / "manifest.json", man)
    pr._write_json(out / "validation_report.json", rep)
    return rep


# --------------------------------------------------------------------------------------------
# Estimates (stdlib arithmetic on nativerun's two measured pace figures; every figure ESTIMATE)
# --------------------------------------------------------------------------------------------


def estimate_rows(
    speed: Mapping[str, Any],
    model: str,
    n_pages: int,
    smoke_pages: int,
    det_pages: int,
    batch: int,
) -> list[dict[str, Any]]:
    """ESTIMATE rows (low / high pace) of one test GPU session on a T4 at the native resolution.

    Pace as `nativerun`: low = the 02 batch-8 pace (MEASURED at 1260 tokens) x1.047 (the 06
    sweep's native cost ratio), valid only if `batch` fits; high = the sweep's MEASURED native
    batch-1 pace (at batch 1 there is one row). ``zs``: three processes (smoke at batch 1, the test
    run, the determinism pass), each one model load, no bench (the batch is the 02n bench's).
    ``ft``: one process (load + merge + guard of 12 bench pages at batch 1 and `batch` + smoke at
    batch 1 + the test run) and a second process for the determinism pass (load + merge again).
    The replayed determinism calls may add companion pages.
    """
    _need_model(model)
    load = float(speed.get("model_load_s", 0))
    s1 = nativerun.NATIVE_B1_S_PER_PAGE
    rows: list[dict[str, Any]] = []
    for scenario in ("low", "high") if batch > 1 else ("high",):
        sb = nativerun.batched_pace(scenario) if batch > 1 else s1
        if model == "zs":
            parts = {
                "load_merge_s": 0.0,
                "guard_s": 0.0,
                "smoke_s": load + smoke_pages * s1,
                "run_s": load + n_pages * sb,
                "det_s": load + det_pages * sb,
            }
        else:
            guard = 0.0 if batch == 1 else s1 + nativerun.BENCH_PAGES * (s1 + sb)
            parts = {
                "load_merge_s": load + nativerun.MERGE_S,
                "guard_s": guard,
                "smoke_s": smoke_pages * s1,
                "run_s": n_pages * sb,
                "det_s": load + nativerun.MERGE_S + det_pages * sb,
            }
        hours = sum(parts.values()) / 3600
        r1, r2 = speed.get("t4_cu_per_hour"), speed.get("t4_cu_per_hour_conservative")
        rows.append(
            {
                "scenario": scenario if batch > 1 else "batch 1",
                "s_per_page": sb,
                **parts,
                "hours": hours,
                "cu_central": None if r1 is None else hours * r1,
                "cu_conservative": None if r2 is None else hours * r2,
            }
        )
    return rows


def format_estimate(
    rows: Sequence[Mapping[str, Any]],
    speed: Mapping[str, Any],
    model: str,
    n_pages: int,
    batch: int,
) -> str:
    """The printed estimate (every figure labelled ESTIMATE, UNVERIFIED on a GPU)."""
    bar = "=" * 110
    r1, r2 = speed.get("t4_cu_per_hour"), speed.get("t4_cu_per_hour_conservative")
    name = "zero-shot (base model)" if model == "zs" else "fine-tuned (merged final adapter)"
    out = [
        bar,
        f"ESTIMATE (UNVERIFIED on a GPU) T4 hours / CU: native test inference, {name}, "
        f"{n_pages} pages, batch {batch}",
        bar,
        f"{'pace':>8} {'s/page':>7} {'load+merge':>10} {'guard s':>8} {'smoke s':>8} {'run h':>6} "
        f"{'det s':>6} {'T4 h':>6} {'CU@' + str(r1):>9} {'CU@' + str(r2):>9}",
    ]
    fmt = lambda v: "n/a" if v is None else f"{v:.1f}"  # noqa: E731
    for r in rows:
        out.append(
            f"{r['scenario']:>8} {r['s_per_page']:7.1f} {r['load_merge_s']:10.0f} "
            f"{r['guard_s']:8.0f} {r['smoke_s']:8.0f} {r['run_s'] / 3600:6.2f} {r['det_s']:6.0f} "
            f"{r['hours']:6.2f} {fmt(r['cu_central']):>9} {fmt(r['cu_conservative']):>9}"
        )
    out += [
        "\n".join(nativerun._basis_lines(speed)),
        "Not modelled: the OCR stage (reused from the 04b cache when valid: nothing to run; the "
        "estimate cell above covers a rerun), the CPU assembly and flags (seconds), restarts. "
        "The batch is the 02n bench's: VRAM at that size at native was measured there, not here.",
        bar,
    ]
    return "\n".join(out)


def run_estimate(
    *,
    model: str,
    speed: Mapping[str, Any],
    data_root: Path,
    batch: int,
    smoke_docs: Sequence[str],
    out: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Print the ESTIMATE (reads image FILE NAMES and dev labels only; runs no model)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "gpu_estimate", paths.REPO_ROOT / "scripts" / "gpu_estimate.py"
    )
    if spec is None or spec.loader is None:
        raise NativeTestError("scripts/gpu_estimate.py not found")
    ge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ge)
    plan = pr.plan(Path(data_root))
    by_doc = pr.discover_test_docs(Path(data_root))
    det_pages = sum(by_doc[d] for d in plan["determinism_docs"])
    smoke_pages = ge.count_pages(list(smoke_docs), Path(data_root) / "dev" / "labels")
    rows = estimate_rows(speed, model, plan["n_pages"], smoke_pages, det_pages, batch)
    out(format_estimate(rows, speed, model, plan["n_pages"], batch))
    return {"n_pages": plan["n_pages"], "smoke_pages": smoke_pages, "det_pages": det_pages,
            "rows": rows}  # fmt: skip


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Argument parser of ``python -m shipdoc.predict_native``."""
    p = argparse.ArgumentParser(
        prog="python -m shipdoc.predict_native", description=__doc__.split("\n")[0]
    )
    sub = p.add_subparsers(dest="stage", required=True)

    def cfg_arg(s: argparse.ArgumentParser) -> None:
        s.add_argument("--config", type=Path, default=Path(NATIVE_CONFIG))

    def model_arg(s: argparse.ArgumentParser) -> None:
        s.add_argument("--model", choices=MODELS, required=True)

    s = sub.add_parser("check", help="CPU refusals before anything runs; writes the resolved batch")
    model_arg(s)
    cfg_arg(s)
    s.add_argument("--zs-run-dir", type=Path, required=True)
    s.add_argument("--batch-size", type=int, default=None)
    s.add_argument("--adapter-dir", type=Path, default=None)
    s.add_argument("--zs-test-dir", type=Path, default=None)
    s.add_argument("--v0-dir", type=Path, default=None, help="zs: must be REFUSED for reuse")
    s.add_argument("--ack-spike-diff", action="store_true")
    s.add_argument("--data-root", type=Path, default=None)
    s.add_argument("--out", type=Path, default=None)
    s = sub.add_parser("estimate", help="ESTIMATE of T4 hours and CU (prints only)")
    model_arg(s)
    cfg_arg(s)
    s.add_argument("--zs-run-dir", type=Path, default=None)
    s.add_argument("--batch-size", type=int, default=None)
    s.add_argument("--smoke-docs", default="splits/smoke5.json")
    s.add_argument("--data-root", type=Path, default=None)
    for name, help_ in (
        ("verify", "(ft) resolution gate + final adapter verification"),
        ("infer", "(ft) resolution gate + verify, merge, guard, smoke, resumable test run"),
        ("determinism", "(ft) resolution gate + the second pass in a fresh process"),
    ):
        sub.add_parser(name, help=help_)  # the flags are predict_ft's own (passed through)
    s = sub.add_parser("finalize", help="the 04b / 04c checks + the native checks")
    model_arg(s)
    cfg_arg(s)
    s.add_argument("--out-dir", type=Path, required=True)
    s.add_argument("--run-dir", type=Path, required=True)
    s.add_argument("--zs-test-dir", type=Path, default=None)
    s.add_argument("--recheck", type=Path, default=None)
    s.add_argument("--shapes-file", type=Path, default=paths.REPO_ROOT / reuse.SHAPES_REL)
    s.add_argument("--expect-pages", type=int, default=pr.EXPECTED_PAGES)
    s = sub.add_parser("flags", help="the review flags after the native refusals")
    model_arg(s)
    cfg_arg(s)
    s.add_argument("--submission-dir", type=Path, required=True)
    s.add_argument("--calibrator", type=Path, default=None)
    s.add_argument("--ocr-cache", type=Path, required=True)
    s.add_argument("--zs-test-dir", type=Path, default=None)
    s.add_argument("--batch-size", type=int, required=True, help="the size the ZS test run used")
    s.add_argument("--field-target", type=float, default=flags.PRIMARY_TARGET)
    s.add_argument("--doc-target", type=float, default=flags.PRIMARY_TARGET)
    s.add_argument("--expect-docs", type=int, default=pr.EXPECTED_DOCS)
    s.add_argument("--ack-spike-diff", action="store_true")
    s.add_argument("--data-root", type=Path, default=None)
    s.add_argument("--shapes-file", type=Path, default=None)
    s = sub.add_parser("check-flags", help="structure / value-free checks of review_flags.json")
    cfg_arg(s)
    s.add_argument("--out-dir", type=Path, required=True)
    s.add_argument("--calibrator", type=Path, required=True)
    s.add_argument("--field-target", type=float, default=flags.PRIMARY_TARGET)
    s.add_argument("--doc-target", type=float, default=flags.PRIMARY_TARGET)
    s.add_argument("--expect-docs", type=int, default=pr.EXPECTED_DOCS)
    return p


def _argv_config(rest: Sequence[str]) -> Path:
    """The ``--config`` of a pass-through command line (the native config when absent)."""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--config", type=Path, default=Path(NATIVE_CONFIG))
    return ap.parse_known_args(list(rest))[0].config  # type: ignore[no-any-return]


def _gated_ft(stage: str, rest: list[str]) -> int:
    """Resolution gate, then the unchanged ``predict_ft`` stage; ``infer`` stamps the gate."""
    gate = argparse.ArgumentParser(add_help=False)
    gate.add_argument("--adapter-dir", type=Path, required=True)
    gate.add_argument("--run-id", default=None)
    gate.add_argument("--runs-root", type=Path, default=None)
    g, _ = gate.parse_known_args(rest)
    cfg = load_native_config(_argv_config(rest))
    nativerun.assert_adapter_resolution(g.adapter_dir, cfg)  # prints the PASS row, or raises
    args = list(rest)
    if not any(a == "--config" or a.startswith("--config=") for a in args):
        args += ["--config", NATIVE_CONFIG]
    rc = int(predict_ft.main([stage, *args]))
    if rc == 0 and stage == "infer" and g.run_id:
        ft_path = (Path(g.runs_root) if g.runs_root else paths.runs_dir()) / g.run_id
        ft_path = ft_path / predict_ft.FT_RUN_FILE
        if ft_path.is_file():
            rec = pr._read_json(ft_path)
            rec["native_resolution"] = {
                "ok": True,
                "max_pixels": cfg.backend.max_pixels,
                "config_hash": cfg.config_hash,
                "check": "shipdoc.resmatch.training_resolution_matches",
            }
            pr._write_json(ft_path, rec)
    return rc


def main(argv: Sequence[str] | None = None) -> int:
    """Stage CLI. Exit 0 = passed; 1 = refused / a check failed; 3 = ZS traces not reusable."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in ("verify", "infer", "determinism"):
        try:
            return _gated_ft(args[0], args[1:])
        except (NativeTestError, nativerun.NativeError, predict_ft.FtError) as exc:
            print(f"predict_native {args[0]}: REFUSED: {exc}", file=sys.stderr)
            return 1
    a = build_parser().parse_args(args)
    try:
        return _dispatch(a)
    except flags.ZsReuseRefused as exc:
        print(f"predict_native {a.stage}: REFUSED: {exc}", file=sys.stderr)
        return 3
    except (NativeTestError, nativerun.NativeError, predict_ft.FtError, pr.PredictError,
            flags.FlagsError, oof.OofError) as exc:  # fmt: skip
        print(f"predict_native {a.stage}: REFUSED: {exc}", file=sys.stderr)
        return 1


def _dispatch(a: argparse.Namespace) -> int:
    cfg = load_native_config(a.config)
    if a.stage == "check":
        return _check_stage(a, cfg)
    if a.stage == "estimate":
        return _estimate_stage(a, cfg)
    if a.stage == "finalize":
        rep = finalize(a.model, a.out_dir, cfg=cfg, run_dir=a.run_dir, shapes_file=a.shapes_file,
                       recheck_path=a.recheck, expect_pages=a.expect_pages,
                       zs_test_dir=a.zs_test_dir)  # fmt: skip
        failed = [k for k, c in rep["checks"].items() if not c["ok"]]
        print(f"finalize ({a.model}, native): ok={rep['ok']} failed checks {failed}")
        return 0 if rep["ok"] else 1
    if a.stage == "flags":
        run_flags(a.model, cfg=cfg, config_path=a.config, submission_dir=a.submission_dir,
                  calibrator=a.calibrator, ocr_cache=a.ocr_cache, zs_test_dir=a.zs_test_dir,
                  batch_size=a.batch_size, field_target=a.field_target, doc_target=a.doc_target,
                  expect_docs=a.expect_docs, ack_spike_diff=a.ack_spike_diff,
                  data_root=a.data_root, shapes_file=a.shapes_file)  # fmt: skip
        return 0
    rep = check_flags(a.out_dir, calibrator=a.calibrator, cfg=cfg, field_target=a.field_target,
                      doc_target=a.doc_target, expect_docs=a.expect_docs)  # fmt: skip
    failed = [k for k, c in rep["flags_checks"].items() if not c["ok"]]
    print(f"check-flags (native): flags_ok={rep['flags_ok']} failed checks {failed}")
    return 0 if rep["flags_ok"] else 1


def _check_stage(a: argparse.Namespace, cfg: spike.SpikeConfig) -> int:
    paths.apply_env()
    model = a.model
    zs_man = check_zs_run(a.zs_run_dir, cfg)
    chosen = resolve_batch(model, a.zs_run_dir, a.batch_size)
    batch = int(chosen["batch_size"])
    result: dict[str, Any] = {
        "model": model,
        "config": cfg.name,
        "config_hash": cfg.config_hash,
        "max_pixels": cfg.backend.max_pixels,
        "batch_size": batch,
        "batch_source": chosen["source"],
        "zs_run": Path(a.zs_run_dir).name,
        "zs_run_config_hash": runcompat.manifest_config_hash(zs_man),
    }
    print(
        f"PASS  native config {cfg.name} (max_pixels {cfg.backend.max_pixels}, hash "
        f"{cfg.config_hash}); 02n run {Path(a.zs_run_dir).name} carries the same hash"
    )
    print(
        f"PASS  batch size {batch} ({chosen['source']}; stored with the 02n run {chosen['stored']})"
    )
    if chosen["differs_from_zero_shot"]:
        print(f"WARNING: BATCH_SIZE {batch} differs from the 02n run's {chosen['stored']}.")
    if model == "ft":
        if a.adapter_dir is None:
            raise NativeTestError("ft needs --adapter-dir (the native final run's final/ folder)")
        check_adapter_native(a.adapter_dir, cfg)
        if a.zs_test_dir is None:
            raise NativeTestError("ft needs --zs-test-dir (the v15 folder of the zs path)")
        zs_batch = next(iter(chosen["stored"].values()), batch)  # what the v15 run must have used
        zs_test = check_zs_test_dir(a.zs_test_dir, cfg, zs_batch, data_root=a.data_root,
                                    ack_spike_diff=a.ack_spike_diff)  # fmt: skip
        result["zs_test"] = zs_test
        result["zs_test_batch_size"] = zs_test["batch_size"]
        print(
            f"PASS  ZS_TEST_DIR {zs_test['dir']}: VALIDATED, native config hash, strictly reusable"
        )
    elif a.v0_dir is not None and Path(a.v0_dir).is_dir():
        reasons = assert_v0_refused(a.v0_dir, cfg, batch, data_root=a.data_root)
        result["v0_refused"] = reasons
        print(f"PASS  the v0 traces are REFUSED for reuse ({len(reasons)} reasons, config hash "
              "among them): the VLM runs")  # fmt: skip
    if a.out:
        pr._write_json(Path(a.out), result)
    return 0


def _estimate_stage(a: argparse.Namespace, cfg: spike.SpikeConfig) -> int:
    paths.apply_env()
    if a.zs_run_dir is not None:
        check_zs_run(a.zs_run_dir, cfg)
        batch = int(resolve_batch(a.model, a.zs_run_dir, a.batch_size)["batch_size"])
    elif a.batch_size is not None:
        batch = int(a.batch_size)
    else:
        raise NativeTestError("pass --zs-run-dir (the 02n run) or --batch-size")
    speed = json.loads((paths.REPO_ROOT / "configs" / "spike_speed.json").read_text("utf-8"))
    data = Path(a.data_root) if a.data_root else paths.data_dir()
    run_estimate(model=a.model, speed=speed, data_root=data, batch=batch,
                 smoke_docs=spike.load_doc_ids(a.smoke_docs))  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
