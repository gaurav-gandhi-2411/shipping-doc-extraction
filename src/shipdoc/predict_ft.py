"""Production pipeline v2 (notebook 04c): the FINAL fine-tuned adapter + R1-R3 + review flags.

04c = the 04 inference path with the fine-tuned model: the ``final`` adapter of notebook 03 (trained
on the 400 ``train_*`` documents, never on a dev or test document) is verified, merged into the fp16
base weights, guarded (the 12 bench pages at batch 1 and at the chosen batch size on the
MERGED model must be byte-identical, else the run uses batch 1), smoke-checked on dev documents,
and run on the 200
test documents with logprobs on through the same ``spike.run_spike`` machinery as 04 (resumable,
determinism pass in a fresh process). OCR and the post-rules R1-R3 are those of 04b
(``shipdoc.ocr_stage``, ``shipdoc.postrules`` via ``predict assemble``). The review flags are a
separate stage (``shipdoc.flags``). Stages of ``python -m shipdoc.predict_ft``:

``estimate``      ESTIMATE of T4 hours / compute units (labelled; nothing in it is measured).
``verify``        `verify_final_adapter`: the printed refusal table (fails closed).
``infer``         VERIFY -> MERGE -> GUARD -> SMOKE -> TEST RUN in one process (the merged
                  model never leaves it), resumable per document.
``determinism``   a second pass over 5 seeded test documents in a fresh process (merges again).
``finalize``      `reuse.finalize` (the v1 blocking checks) + the v2 checks; the manifest says
                  ``submission = v2_<sha7>`` (`reuse.finalize` labels every folder ``v1_...``).
``check-flags``   the value-free / structure checks of ``review_flags.json`` and of the predictions.
``adopt-ocr``     take the OCR timing and re-check record of a 04b folder (same Drive OCR cache).

Test data policy: nothing here prints or stores an extracted value; reports carry counts, ids,
hashes and key paths only. The submission folder is outside the repo
(`predict.assert_outside_repo`).

What is UNVERIFIED on a GPU: the merge itself (peft on the real model, `oof.MergedHfBackend`), the
fp16 byte-identity of batch 8 on the merged model, and every timing here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from shipdoc import confidence_v2 as c2
from shipdoc import flags, oof, paths, reuse, spike
from shipdoc import predict as pr
from shipdoc import trainset as ts
from shipdoc.bench import BENCH_PAGES, RESULT_NAME, run_bench
from shipdoc.train import EXPECTED_LORA_MODULES, expected_lora_params

SUBMISSION_VERSION = "v2"
FT_RUN_FILE = "ft_run.json"
FT_SCHEMA = 1
EXPECTED_LORA_R = oof.EXPECTED_LORA_R
OUT_FILES = (*reuse.OUT_FILES, flags.FLAGS_NAME)
GUARD_SOURCE = "guard on the merged model (12 dev bench pages, batch 1 vs the chosen size)"
CONTRACT_NOTE = "n/a: a fine-tuned model has no dev-run batch contract; the guard decides"


class FtError(RuntimeError):
    """A precondition of the fine-tuned pipeline failed. Messages never quote a value."""


# --------------------------------------------------------------------------------------------
# Final adapter verification
# --------------------------------------------------------------------------------------------


def _row(name: str, expected: Any, found: Any, ok: bool, note: str = "") -> dict[str, Any]:
    return {"check": name, "expected": expected, "found": found, "ok": bool(ok), "note": note}


def _is_sha40(sha: Any) -> bool:
    return (
        isinstance(sha, str)
        and len(sha) == 40
        and all(c in "0123456789abcdef" for c in sha)
        and "+dirty" not in sha
    )


def verify_final_adapter(
    adapter_dir: Path,
    *,
    cfg: spike.SpikeConfig,
    folds: Mapping[str, Any],
    pin_sha: str | None = None,
    reachable: Callable[[str], bool] | None = None,
    expected_r: int = EXPECTED_LORA_R,
) -> dict[str, Any]:
    """Check a ``final/`` adapter folder of notebook 03; never raises on a failed check.

    Refused (rows failing): any stage but ``final`` (a smoke / fold adapter), a fold id, a missing
    or dirty or unreachable training code SHA, a base repo / revision other than the production
    config's, LoRA bookkeeping other than r=16 / 200 modules / 30,474,240 trainable parameters, a
    manifest without its training doc ids, a training set that is not exactly the ``train_*``
    documents of ``splits/folds.json`` (a dev or test id in it is a refusal of its own), held-out
    ids that are not the dev documents, a ``manifest_hash`` that does not match the ids,
    inference keys (model repo / revision, adapter name, max_pixels, prompt version, output
    format) that differ from the production config, weight files whose sha256 differs from the
    manifest, no precision. A training SHA that differs from `pin_sha` is a WARNING.
    """
    d = Path(adapter_dir)
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    mpath = d / "manifest.json"
    if not mpath.is_file():
        rows.append(_row("manifest.json", "present", "missing", False))
        return {"ok": False, "rows": rows, "warnings": warnings}
    m = json.loads(mpath.read_text(encoding="utf-8"))
    split = ts.stage_split("final", folds)
    want_train, want_held = sorted(split.train_ids), sorted(split.heldout_ids)
    stage = m.get("stage")
    note = "a fold / smoke adapter is not the production model" if stage != "final" else ""
    rows.append(_row("stage", "final", stage, stage == "final", note))
    rows.append(_row("fold id", None, m.get("fold"), m.get("fold") is None))

    train_sha = m.get("code_sha")
    sha_ok = _is_sha40(train_sha)
    rows.append(_row("training code sha", "40-hex, clean", train_sha, sha_ok))
    if sha_ok and reachable is not None:
        ok = bool(reachable(str(train_sha)))
        rows.append(_row("training code sha reachable", True, ok, ok))
    if sha_ok and pin_sha and train_sha != pin_sha:
        warnings.append(
            f"training code {str(train_sha)[:7]} differs from this notebook's pin {pin_sha[:7]}: "
            "check `git diff --stat <train>..<pin> -- src/shipdoc/{train,prompts,extract}.py`"
        )

    b = cfg.backend
    rows.append(_row("base revision", b.revision, m.get("base_revision"),
                     m.get("base_revision") == b.revision))  # fmt: skip
    rows.append(_row("base repo", b.repo, m.get("base_repo"), m.get("base_repo") == b.repo))
    lora = m.get("lora") if isinstance(m.get("lora"), dict) else {}
    rows.append(_row("lora r", expected_r, lora.get("r"), lora.get("r") == expected_r))
    rows.append(_row("lora modules", EXPECTED_LORA_MODULES, lora.get("n_modules"),
                     lora.get("n_modules") == EXPECTED_LORA_MODULES))  # fmt: skip
    want_params = expected_lora_params(expected_r)
    rows.append(_row("lora trainable params", want_params, lora.get("n_trainable"),
                     lora.get("n_trainable") == want_params))  # fmt: skip

    ids = m.get("train_doc_ids")
    n_train: int | None = None
    if not isinstance(ids, list) or not ids:
        rows.append(_row("training doc ids", "recorded in the manifest", "missing", False,
                         "adapter predates doc-id recording: retrain"))  # fmt: skip
    else:
        n_train = len(ids)
        n_test = sum(str(i).startswith("test_") for i in ids)
        n_dev = sum(str(i).startswith("dev_") for i in ids)
        rows.append(_row("no test document in training", 0, n_test, n_test == 0))
        rows.append(_row("no dev document in training", 0, n_dev, n_dev == 0))
        rows.append(_row("training set = the train_* documents", len(want_train), len(ids),
                         sorted(ids) == want_train and len(set(ids)) == len(ids)))  # fmt: skip
        mh = m.get("manifest_hash")
        calc = hashlib.sha256(("final|" + ",".join(sorted(ids))).encode()).hexdigest()[:16]
        rows.append(_row("manifest_hash matches the ids", calc, mh, mh == calc))
    held = m.get("heldout_doc_ids")
    if isinstance(held, list):
        rows.append(_row("recorded held-out ids = the dev documents", len(want_held), len(held),
                         sorted(held) == want_held))  # fmt: skip

    want_keys, got_keys = oof.inference_keys_of_config(cfg), m.get("inference_keys")
    got_keys = got_keys if isinstance(got_keys, dict) else {}
    for k in oof.INFERENCE_KEYS:
        rows.append(_row(f"inference key {k}", want_keys[k], got_keys.get(k),
                         k in got_keys and got_keys[k] == want_keys[k]))  # fmt: skip

    ap = d / "adapter.pt"
    found = oof.sha256_file(ap) if ap.is_file() else "missing"
    rows.append(_row("adapter.pt sha256", m.get("adapter_sha256"), found,
                     ap.is_file() and found == m.get("adapter_sha256")))  # fmt: skip
    peft = d / "peft"
    recorded = m.get("peft_sha256") if isinstance(m.get("peft_sha256"), dict) else {}
    mf = peft / "adapter_model.safetensors"
    found = oof.sha256_file(mf) if mf.is_file() else "missing"
    rows.append(_row("peft/adapter_model.safetensors sha256", recorded.get(mf.name), found,
                     mf.is_file() and found == recorded.get(mf.name)))  # fmt: skip
    cfg_file = peft / "adapter_config.json"
    rows.append(_row("peft/adapter_config.json", "present",
                     "present" if cfg_file.is_file() else "missing",
                     cfg_file.is_file()))  # fmt: skip
    rows.append(_row("training precision", "recorded", m.get("precision"),
                     m.get("precision") in ("bf16", "fp16", "fp32")))  # fmt: skip
    return {
        "ok": all(r["ok"] for r in rows),
        "rows": rows,
        "warnings": warnings,
        "train_sha": train_sha,
        "pin_sha": pin_sha,
        "n_train_docs": n_train,
        "train_precision": m.get("precision"),
        "adapter_sha256": m.get("adapter_sha256"),
        "peft_sha256": recorded,
        "lora": lora,
    }


def format_verification(report: Mapping[str, Any]) -> str:
    """The printed table (counts, hashes and SHAs only)."""
    bar = "=" * 100
    lines = [
        bar,
        "FINAL ADAPTER MANIFEST VERIFICATION: "
        + ("PASSED" if report["ok"] else "FAILED (refused)"),
        bar,
    ]
    w = max(len(r["check"]) for r in report["rows"])
    for r in report["rows"]:
        e, f = str(r["expected"]), str(r["found"])
        e, f = (e[:24] + "..") if len(e) > 26 else e, (f[:24] + "..") if len(f) > 26 else f
        note = f"  <- {r['note']}" if r.get("note") else ""
        lines.append(
            f"{'PASS' if r['ok'] else 'FAIL':<5} {r['check']:<{w}}  expected {e:<28} "
            f"found {f}{note}"
        )
    if report.get("train_sha"):
        lines.append(
            f"training code {report['train_sha']}  |  this notebook's pin {report.get('pin_sha')}"
        )
    if report.get("n_train_docs") is not None:
        lines.append(f"training docs {report['n_train_docs']}")
    lines += [f"WARNING: {w_}" for w_ in report.get("warnings", [])]
    lines.append(bar)
    return "\n".join(lines)


def assert_verified(report: Mapping[str, Any]) -> None:
    """Raise `FtError` naming every failed check when the verification did not pass."""
    if not report["ok"]:
        failed = [r["check"] for r in report["rows"] if not r["ok"]]
        raise FtError(f"final adapter verification FAILED, refused: {failed}")


def verify_stage(
    *,
    adapter_dir: Path,
    cfg: spike.SpikeConfig,
    folds: Mapping[str, Any],
    pin_sha: str | None,
    reachable: Callable[[str], bool] | None,
    out: Callable[[str], None] = print,
) -> dict[str, Any]:
    """VERIFY: print the table and raise `FtError` when anything fails."""
    report = verify_final_adapter(
        adapter_dir, cfg=cfg, folds=folds, pin_sha=pin_sha, reachable=reachable
    )
    out("== VERIFY ==")
    if report.get("rows"):
        out(format_verification(report))
    assert_verified(report)
    return report


# --------------------------------------------------------------------------------------------
# Merge, guard, batch decision
# --------------------------------------------------------------------------------------------


def merge_stage(backend: Any, out: Callable[[str], None] = print) -> dict[str, Any]:
    """Load the backend (= merge the adapter) and check that `EXPECTED_LORA_MODULES` were merged."""
    out("== MERGE ==")
    t0 = time.perf_counter()
    load = getattr(backend, "load", None)
    if callable(load):
        load()
    load_s = round(time.perf_counter() - t0, 1)
    info = getattr(backend, "merge_info", None)
    if not isinstance(info, dict) or info.get("n_lora_modules_merged") != EXPECTED_LORA_MODULES:
        raise FtError(f"merge did not merge {EXPECTED_LORA_MODULES} LoRA modules: {info}")
    out(
        f"merged {info['n_lora_modules_merged']} LoRA modules into the base weights "
        f"({info.get('merge_dtype')}), model load + merge {load_s} s"
    )
    return {**info, "load_and_merge_s": load_s}


def guard_stage(
    cfg: spike.SpikeConfig,
    backend: Any,
    bench_docs: Sequence[str],
    batch: int,
    guard_dir: Path,
    data_root: Path,
    out: Callable[[str], None] = print,
) -> dict[str, Any]:
    """The 05 guard on the MERGED model: byte-identical at `batch` and at 1, else batch 1."""
    guard: dict[str, Any] = {"ran": False, "batch_size": batch, "fallback_to_1": False}
    if batch <= 1:
        out("== GUARD == skipped (batch 1)")
        return guard
    out(
        f"== GUARD == {len(bench_docs)} bench documents at batch 1 and batch {batch} on the "
        "MERGED model (byte-identical outputs required)"
    )
    res = run_bench(cfg, backend, list(bench_docs), guard_dir, (1, batch), data_root, logprobs=True)
    entry = next(r for r in res["results"] if r["batch_size"] == batch)
    same = bool(entry.get("ok")) and entry.get("byte_identical_rate") == 1.0
    guard.update(
        ran=True,
        ok=same,
        byte_identical_rate=entry.get("byte_identical_rate"),
        error=entry.get("error"),
        n_pages=res["n_pages"],
    )
    if same:
        out(f"GUARD PASSED: batch {batch} byte-identical to batch 1 on {res['n_pages']} pages.")
    else:
        guard["fallback_to_1"] = True
        out(
            f"GUARD FAILED: batch {batch} is not byte-identical to batch 1 on the merged model: "
            "FALLING BACK TO BATCH 1 (recorded in the manifest)."
        )
    return guard


def make_decision(
    cfg: spike.SpikeConfig,
    backend: Any,
    guard: Mapping[str, Any],
    requested: int,
    guard_dir: Path,
    adapter_sha256: str,
) -> dict[str, Any]:
    """The batch decision in the shape `predict.run_test` / `run_determinism` read."""
    batch = 1 if guard.get("fallback_to_1") else int(requested)
    bench_path = Path(guard_dir) / RESULT_NAME
    info = None
    if guard.get("ran"):
        info = {
            "path": str(bench_path),
            "chosen": batch,
            "deviation": dict(guard) if guard.get("fallback_to_1") else None,
            "sha256": pr.sha256_file(bench_path) if bench_path.is_file() else None,
        }
    return {
        "batch_size": batch,
        "source": GUARD_SOURCE,
        "requested_batch_size": int(requested),
        "bench": info,
        "bench_reused": False,
        "note": None,
        "dev_run_dir": None,
        "dev_batch_size": None,
        "contract": CONTRACT_NOTE,
        "adapter_sha256": adapter_sha256,
        **pr._identity(cfg, backend),
    }


def load_decision(
    path: Path, cfg: spike.SpikeConfig, backend: Any, adapter_sha256: str
) -> dict[str, Any] | None:
    """The stored decision when it was made for THIS adapter, code, config and model, else None."""
    p = Path(path)
    if not p.is_file():
        return None
    dec = json.loads(p.read_text(encoding="utf-8"))
    want = {**pr._identity(cfg, backend), "adapter_sha256": adapter_sha256}
    return dec if all(dec.get(k) == v for k, v in want.items()) else None


# --------------------------------------------------------------------------------------------
# The test run (VERIFY -> MERGE -> GUARD -> SMOKE -> RUN) and the determinism pass
# --------------------------------------------------------------------------------------------


def run_ft_infer(
    *,
    cfg: spike.SpikeConfig,
    adapter_dir: Path,
    folds: Mapping[str, Any],
    run_id: str,
    runs_root: Path,
    data_root: Path,
    decision_path: Path,
    smoke_status_path: Path,
    smoke_docs: Sequence[str],
    bench_docs: Sequence[str],
    batch_size: int,
    backend_factory: Callable[[Path, str], Any],
    pin_sha: str | None = None,
    reachable: Callable[[str], bool] | None = None,
    expect_docs: int | None = None,
    expect_pages: int | None = None,
    out: Callable[[str], None] = print,
) -> dict[str, Any]:
    """VERIFY -> MERGE -> GUARD -> SMOKE -> TEST RUN. Resumable; returns the ``ft_run`` record.

    `backend_factory(adapter_dir, adapter_sha256)` builds the backend (`oof.MergedHfBackend` for
    real, a mock in tests). A stored batch decision made for the same adapter / code / config /
    model is reused on a resume (the guard is not repeated, so the batch size cannot change under
    a half-finished run); otherwise the guard runs. The smoke gate (the 5 dev documents of
    `smoke_docs`, same seven checks as 04) runs on the merged model and must pass before the first
    test document is decoded.
    """
    if isinstance(batch_size, bool) or batch_size < 1:
        raise FtError(f"batch size must be an int >= 1, got {batch_size!r}")
    report = verify_stage(
        adapter_dir=adapter_dir, cfg=cfg, folds=folds, pin_sha=pin_sha, reachable=reachable, out=out
    )
    sha = str(report.get("adapter_sha256") or "")
    backend = backend_factory(Path(adapter_dir), sha)
    merge = merge_stage(backend, out)
    run_dir = Path(runs_root) / run_id
    guard_dir = run_dir / "guard"
    decision = (
        load_decision(decision_path, cfg, backend, sha)
        if (run_dir / "trace.jsonl").is_file()
        else None
    )
    if decision is not None:
        out(
            "== GUARD == resumed run: the stored batch decision stands "
            f"(batch {decision['batch_size']})"
        )
        guard = {"ran": False, "reused_decision": True, "batch_size": decision["batch_size"],
                 "fallback_to_1": decision["batch_size"]
                 != decision["requested_batch_size"]}  # fmt: skip
    else:
        guard = guard_stage(cfg, backend, bench_docs, batch_size, guard_dir, data_root, out)
        decision = make_decision(cfg, backend, guard, batch_size, guard_dir, sha)
        pr._write_json(Path(decision_path), decision)
    out("== SMOKE == 5 dev documents on the merged model (the gate of 04)")
    status = pr.run_smoke_gate(
        cfg, backend, list(smoke_docs), f"smoke_{run_id}", Path(runs_root) / f"{run_id}_meta",
        smoke_status_path, data_root,
    )  # fmt: skip
    if status["state"] != pr.SMOKE_STATE_PASSED:
        raise FtError(
            f"smoke gate {status['state']!r} on the merged model: the test run is NOT started "
            f"({status.get('error') or [c['name'] for c in status['checks'] if not c['passed']]})"
        )
    out(f"== RUN == test documents at batch {decision['batch_size']}")
    metrics = pr.run_test(
        cfg, backend, run_id, runs_root, data_root, decision, smoke_status_path, "0/1",
        expect_docs, expect_pages,
    )  # fmt: skip
    record = {
        "schema": FT_SCHEMA,
        "adapter_dir": Path(adapter_dir).name,
        "adapter_sha256": sha,
        "peft_sha256": report.get("peft_sha256"),
        "train_code_sha": report.get("train_sha"),
        "pin_sha": pin_sha,
        "train_precision": report.get("train_precision"),
        "n_train_docs": report.get("n_train_docs"),
        "lora": report.get("lora"),
        "verification": {
            "ok": report["ok"],
            "warnings": report["warnings"],
            "n_checks": len(report["rows"]),
        },  # fmt: skip
        "merge": merge,
        "merge_precision_note": "LoRA update added to fp16 base weights (fp32 sum, then fp16)",
        "guard": guard,
        "decision": {k: decision.get(k) for k in ("batch_size", "requested_batch_size", "source")},
        "n_docs": metrics.get("n_docs"),
        "n_pages": metrics.get("n_pages"),
    }
    pr._write_json(run_dir / FT_RUN_FILE, record)
    return record


def run_ft_determinism(
    *,
    cfg: spike.SpikeConfig,
    adapter_dir: Path,
    folds: Mapping[str, Any],
    run_id: str,
    runs_root: Path,
    data_root: Path,
    decision_path: Path,
    backend_factory: Callable[[Path, str], Any],
    pin_sha: str | None = None,
    reachable: Callable[[str], bool] | None = None,
    out: Callable[[str], None] = print,
) -> dict[str, Any]:
    """The determinism pass: verify and merge AGAIN (a fresh process), replay the first pass."""
    report = verify_stage(
        adapter_dir=adapter_dir, cfg=cfg, folds=folds, pin_sha=pin_sha, reachable=reachable, out=out
    )
    sha = str(report.get("adapter_sha256") or "")
    backend = backend_factory(Path(adapter_dir), sha)
    merge_stage(backend, out)
    decision = load_decision(decision_path, cfg, backend, sha)
    if decision is None:
        raise FtError(
            "no batch decision for this adapter / code / config: run the inference stage first "
            "(the determinism pass replays its batch composition)"
        )
    return pr.run_determinism(cfg, backend, run_id, runs_root, data_root, decision, "0/1")


# --------------------------------------------------------------------------------------------
# finalize: the v1 checks + the v2 checks
# --------------------------------------------------------------------------------------------


def structure_signature(preds: Mapping[str, Any]) -> dict[str, Any]:
    """Key structure of a predictions file: ids, the union of document / header / row keys.

    Header keys are unioned per ``doc_type`` value (the type name itself is not part of the
    signature: a document may legitimately change type between two models). No value is stored.
    """
    top: set[str] = set()
    hdr: dict[str, set[str]] = {}
    row: set[str] = set()
    for d in preds.values():
        top |= set(d)
        header = d.get("header") if isinstance(d.get("header"), dict) else {}
        hdr.setdefault(str(d.get("doc_type")), set()).update(header)
        for r in d.get("line_items") or []:
            if isinstance(r, dict):
                row |= set(r)
    return {
        "n_docs": len(preds),
        "ids_sha256": hashlib.sha256(",".join(sorted(preds)).encode()).hexdigest(),
        "doc_keys": sorted(top),
        "header_keys": {k: sorted(v) for k, v in sorted(hdr.items())},
        "row_keys": sorted(row),
    }


def _check(ok: bool, detail: str) -> dict[str, Any]:
    return {"ok": bool(ok), "detail": detail}


def finalize_v2(
    out_dir: Path,
    *,
    run_dir: Path,
    shapes_file: Path,
    v0_dir: Path | None,
    recheck_path: Path | None,
    expect_pages: int = pr.EXPECTED_PAGES,
) -> dict[str, Any]:
    """`reuse.finalize` (mode ``full``) plus the v2 checks; withholds the name if one fails.

    Added checks (all blocking): ``ft_adapter_verified``, ``ft_model_is_merged_adapter`` (the run's
    model id carries the adapter hash), ``ft_merge_recorded`` (200 modules merged, none left),
    ``ft_smoke_on_merged_model``, ``ft_batch_decision`` (the manifest's batch size is the
    decision's), ``predictions_structure_vs_v0`` (same ids, same document / header / row key sets
    as the 04 file: nothing was added to the predictions) and ``predictions_schema_strict`` (the
    file validates and every document has only the schema's top-level keys). The manifest's
    ``submission`` is ``v2_<sha7>``.
    """
    out = Path(out_dir)
    rep = reuse.finalize(out, shapes_file, "full", expect_pages, None, recheck_path)
    man = pr._read_json(out / "manifest.json")
    rep = pr._read_json(out / "validation_report.json")
    checks = rep["checks"]
    ft_path = Path(run_dir) / FT_RUN_FILE
    ft = pr._read_json(ft_path) if ft_path.is_file() else None
    sha12 = str((ft or {}).get("adapter_sha256") or "")[:12]
    model_id = str((man.get("model") or {}).get("id") or "")
    checks["ft_adapter_verified"] = _check(
        bool(ft) and ft["verification"]["ok"] is True,
        f"{(ft or {}).get('verification', {}).get('n_checks')} checks passed" if ft
        else f"{FT_RUN_FILE} missing in the run folder",
    )  # fmt: skip
    checks["ft_model_is_merged_adapter"] = _check(
        bool(sha12) and model_id.endswith(f"{flags.LORA_MARK}{sha12}"),
        f"model id {model_id}",
    )
    merge = (ft or {}).get("merge") or {}
    checks["ft_merge_recorded"] = _check(
        merge.get("n_lora_modules_merged") == EXPECTED_LORA_MODULES
        and merge.get("n_lora_modules_left") == 0,
        f"merged {merge.get('n_lora_modules_merged')}, left {merge.get('n_lora_modules_left')}, "
        f"{merge.get('merge_dtype')}",
    )
    smoke_state = (man.get("smoke") or {}).get("state")
    checks["ft_smoke_on_merged_model"] = _check(
        smoke_state == pr.SMOKE_STATE_PASSED, f"smoke state {smoke_state!r}"
    )
    dec = (ft or {}).get("decision") or {}
    guard = (ft or {}).get("guard") or {}
    checks["ft_batch_decision"] = _check(
        bool(dec) and man.get("batch_size") == dec.get("batch_size"),
        f"manifest batch {man.get('batch_size')} vs decision {dec.get('batch_size')} "
        f"(requested {dec.get('requested_batch_size')}; guard ran {guard.get('ran')}, "
        f"fallback_to_1 {guard.get('fallback_to_1')})",
    )
    pred_name = pr.OUT_FILES[0] if (out / pr.OUT_FILES[0]).is_file() else pr.REJECTED_NAME
    preds = pr._read_json(out / pred_name)
    sig = structure_signature(preds)
    v0_pred = Path(v0_dir) / "test_predictions.json" if v0_dir else None
    if v0_pred is not None and v0_pred.is_file():
        want = structure_signature(pr._read_json(v0_pred))
        diff = sorted(k for k in want if want[k] != sig[k])
        checks["predictions_structure_vs_v0"] = _check(
            not diff,
            f"same structure as the v0 file: {not diff}" + (f"; differs in {diff}" if diff else ""),
        )  # fmt: skip
    else:
        checks["predictions_structure_vs_v0"] = _check(
            False, "the v0 test_predictions.json is missing"
        )
    bad = [
        d
        for d, v in preds.items()
        if not isinstance(v, dict) or set(v) - {"doc_type", "header", "line_items"}
    ]
    checks["predictions_schema_strict"] = _check(
        checks["json_schema"]["ok"] and not bad,
        f"schema ok {checks['json_schema']['ok']}; documents with a key outside "
        f"doc_type / header / line_items: {len(bad)}",
    )
    ok = all(c["ok"] for c in checks.values())
    rep["ok"], rep["mode"] = ok, "full"
    man["submission"] = f"{SUBMISSION_VERSION}_{out.name.split('_', 1)[-1]}"
    man["mode"] = "full"
    man["model_kind"] = "fine-tuned (final adapter, merged into fp16 weights) + R1-R3"
    man["ft"] = ft
    man["files"] = list(OUT_FILES)
    man["structure"] = sig
    pred, rejected = out / pr.OUT_FILES[0], out / pr.REJECTED_NAME
    if not ok and pred.is_file():
        pred.replace(rejected)  # never leave the submittable name on an unchecked file
    pr._write_json(out / "manifest.json", man)
    pr._write_json(out / "validation_report.json", rep)
    return rep


# --------------------------------------------------------------------------------------------
# check-flags
# --------------------------------------------------------------------------------------------


def _all_strings(x: Any) -> set[str]:
    out: set[str] = set()
    if isinstance(x, str):
        out.add(x)
    elif isinstance(x, dict):
        for k, v in x.items():
            out.add(str(k))
            out |= _all_strings(v)
    elif isinstance(x, list):
        for v in x:
            out |= _all_strings(v)
    return out


def prediction_values(preds: Mapping[str, Any], min_len: int = 4) -> set[str]:
    """Every header / row value string of the predictions with at least `min_len` characters."""
    vals: set[str] = set()
    for d in preds.values():
        cells = list((d.get("header") or {}).values())
        for r in d.get("line_items") or []:
            cells += list(r.values()) if isinstance(r, dict) else []
        vals |= {str(v) for v in cells if v is not None and len(str(v)) >= min_len}
    return vals


def check_flags(
    out_dir: Path, *, calibrator: Path, field_target: float, doc_target: float
) -> dict[str, Any]:
    """The checks of ``review_flags.json`` (value-free, complete, thresholds and hashes recorded).

    Written to ``validation_report.json`` as ``flags_checks`` / ``flags_ok`` (the predictions'
    own ``ok`` and name are not touched: the flags file is an advisory side file) and summarised in
    the manifest. Returns the report.
    """
    out = Path(out_dir)
    rep = pr._read_json(out / "validation_report.json")
    man = pr._read_json(out / "manifest.json")
    checks: dict[str, dict[str, Any]] = {}
    fpath = out / flags.FLAGS_NAME
    pred_path = out / pr.OUT_FILES[0]
    checks["flags_file_present"] = _check(fpath.is_file(), f"{flags.FLAGS_NAME} in the folder")
    doc: dict[str, Any] = pr._read_json(fpath) if fpath.is_file() else {}
    preds: dict[str, Any] = pr._read_json(pred_path) if pred_path.is_file() else {}
    ids = sorted(doc.get("docs") or {})
    ids_rep = pr.check_ids(ids, sorted(preds))
    checks["flags_ids"] = _check(
        bool(preds) and ids_rep["ok"] and len(ids) == man.get("docs", {}).get("n_docs"),
        f"{len(ids)} flagged documents; {pr._ids_detail(ids_rep)}",
    )
    strings = doc_string_values_safe(doc)
    stray = sorted(set(strings) - flags.DOC_STRING_VALUES)
    leak = sorted(_all_strings(doc) & prediction_values(preds))
    checks["flags_no_values"] = _check(
        not stray and not leak,
        f"string values under docs outside accept/review: {len(stray)}; extracted values found "
        f"in the file: {len(leak)}",
    )
    thr = doc.get("thresholds") or {}
    tau = thr.get("field_tau") or {}
    checks["flags_thresholds_recorded"] = _check(
        set(tau) == set(c2.FIELD_TYPES)
        and "doc_tau" in thr
        and thr.get("field_target") == field_target
        and thr.get("doc_target") == doc_target
        and bool(thr.get("nested_estimates")),
        f"field tau for {sorted(tau)}, doc tau {thr.get('doc_tau')!r}; targets "
        f"{thr.get('field_target')} / {thr.get('doc_target')}; nested estimates recorded "
        f"{bool(thr.get('nested_estimates'))}",
    )
    want = flags.sha256_file(Path(calibrator)) if Path(calibrator).is_file() else None
    got = (doc.get("calibrator") or {}).get("sha256")
    checks["flags_calibrator_sha256"] = _check(
        bool(want) and got == want,
        f"file {str(got)[:12]} vs {Path(calibrator).name} {str(want)[:12]}",
    )
    inp = doc.get("inputs") or {}
    checks["flags_match_the_predictions"] = _check(
        pred_path.is_file()
        and inp.get("test_predictions_sha256") == pr.sha256_file(pred_path)
        and inp.get("trace_sha256") == pr.sha256_file(out / "trace.jsonl"),
        "inputs.test_predictions_sha256 / trace_sha256 equal the files in this folder",
    )
    rep["flags_checks"] = checks
    rep["flags_ok"] = all(c["ok"] for c in checks.values())
    man["review_flags"] = {
        "file": flags.FLAGS_NAME,
        "ok": rep["flags_ok"],
        "sha256": pr.sha256_file(fpath) if fpath.is_file() else None,
        "calibrator_sha256": got,
        "arm": doc.get("arm"),
        "thresholds": {
            k: thr.get(k) for k in ("field_target", "field_tau", "doc_target", "doc_tau")
        },
        "summary": doc.get("summary"),
    }
    pr._write_json(out / "manifest.json", man)
    pr._write_json(out / "validation_report.json", rep)
    return rep


def doc_string_values_safe(doc: Mapping[str, Any]) -> list[str]:
    """`flags.doc_string_values` of the ``docs`` block (empty when the file has none)."""
    return flags.doc_string_values(doc.get("docs") or {})


# --------------------------------------------------------------------------------------------
# adopt-ocr
# --------------------------------------------------------------------------------------------


def adopt_ocr(
    from_dir: Path, timing_out: Path, recheck_out: Path, expect_pages: int = pr.EXPECTED_PAGES
) -> dict[str, Any]:
    """Take ``ocr.timing`` and ``ocr.recheck`` from a finished 04b folder's manifest.

    Refused unless the 04b manifest holds a timing record of `expect_pages` pages and a PASSED
    re-check. Existing files are never overwritten (the notebook may be a resume). The OCR cache
    itself is the one on Drive, checked again by the notebook's own OCR stage (all pages valid).
    """
    man_path = Path(from_dir) / "manifest.json"
    if not man_path.is_file():
        raise FtError(f"{from_dir}: no manifest.json to adopt the OCR record from")
    ocr = pr._read_json(man_path).get("ocr") or {}
    timing, recheck = ocr.get("timing"), ocr.get("recheck")
    if not isinstance(timing, dict) or timing.get("pages") != expect_pages:
        raise FtError(f"{Path(from_dir).name}: no OCR timing record of {expect_pages} pages")
    if not isinstance(recheck, dict) or recheck.get("ok") is not True:
        raise FtError(f"{Path(from_dir).name}: the OCR re-check did not pass there")
    note = {"adopted_from": Path(from_dir).name}
    done = []
    for path, obj in ((Path(timing_out), timing), (Path(recheck_out), recheck)):
        if not path.is_file():
            pr._write_json(path, {**obj, **note})
            done.append(path.name)
    return {"adopted": done, "from": Path(from_dir).name}


# --------------------------------------------------------------------------------------------
# estimate
# --------------------------------------------------------------------------------------------


def estimate_rows(
    ge: Any,
    speed: Mapping[str, Any],
    config: str,
    n_pages: int,
    batch: int,
    smoke_pages: int,
    det_pages: int,
    merge_s: float = oof.MERGE_S_ESTIMATE,
) -> list[dict[str, Any]]:
    """ESTIMATE rows of one 04c GPU session on a T4, one per scaling efficiency.

    Hours = model load + merge + guard (12 bench pages at batch 1 and `batch`, after a warm-up
    page) + smoke (`smoke_pages` at batch 1) + the test run (`n_pages` at `batch`) + the
    determinism pass in a fresh process (another load + merge + `det_pages` at `batch`).
    """
    m = speed["models"][ge.base_config(config)]
    n_out = float(m["n_output_tokens_mean"])
    load_s = float(speed.get("model_load_s", 0))
    effs = [1.0] if batch == 1 else [e for _b, e in ge.batch_scenarios() if _b == 2]
    rows = []
    for eff in effs:
        s1 = ge.batch_s_per_page(m["prefill_s"], m["decode_tok_s"], n_out, 1, eff)
        sb = ge.batch_s_per_page(m["prefill_s"], m["decode_tok_s"], n_out, batch, eff)
        guard_s = 0.0 if batch == 1 else s1 + BENCH_PAGES * (s1 + sb)
        smoke_s = smoke_pages * s1
        run_s = n_pages * sb
        det_s = load_s + merge_s + det_pages * sb
        total = load_s + merge_s + guard_s + smoke_s + run_s + det_s
        rows.append(
            {
                "efficiency": eff, "s_per_page": sb, "load_s": load_s, "merge_s": merge_s,
                "guard_s": guard_s, "smoke_s": smoke_s, "run_s": run_s, "det_s": det_s,
                "hours": total / 3600,
                "cu_central": None if speed.get("t4_cu_per_hour") is None
                else total / 3600 * speed["t4_cu_per_hour"],
                "cu_conservative": None if speed.get("t4_cu_per_hour_conservative") is None
                else total / 3600 * speed["t4_cu_per_hour_conservative"],
            }
        )  # fmt: skip
    return rows


def format_estimate(
    rows: Sequence[Mapping[str, Any]], n_pages: int, batch: int, det_pages: int
) -> str:
    """The printed estimate table (every figure labelled ESTIMATE)."""
    bar = "=" * 120
    out = [
        bar,
        f"ESTIMATE (UNVERIFIED) T4 hours / CU: 04c fine-tuned test inference, {n_pages} pages at "
        f"batch {batch}. Batching gain, merge time and model load ASSUMED, not measured",
        bar,
        f"{'scaling eff':>11} {'s/page':>7} {'load s':>7} {'merge s':>8} {'guard s':>8} "
        f"{'smoke s':>8} {'run h':>6} {'det s':>6} {'T4 h':>6} {'CU@low':>7} {'CU@high':>8}",
    ]
    for r in rows:
        cu = lambda v: "n/a" if v is None else f"{v:.1f}"  # noqa: E731
        out.append(
            f"{int(r['efficiency'] * 100):>10}% {r['s_per_page']:7.1f} {r['load_s']:7.0f} "
            f"{r['merge_s']:8.0f} {r['guard_s']:8.0f} {r['smoke_s']:8.0f} "
            f"{r['run_s'] / 3600:6.2f} {r['det_s']:6.0f} {r['hours']:6.2f} "
            f"{cu(r['cu_central']):>7} {cu(r['cu_conservative']):>8}"
        )
    out += [
        f"det s = a second load + merge in a fresh process + {det_pages} pages (the replayed "
        "calls may add companion pages). Not modelled: logprob overhead, Drive I/O, restarts, OOM "
        "fallbacks, the CPU assembly (seconds). The guard falls back to batch 1 on any difference "
        "(then the run costs batch-1 time).",
        bar,
    ]
    return "\n".join(out)


def run_estimate(
    *,
    cfg: spike.SpikeConfig,
    data_root: Path,
    batch_size: int,
    smoke_docs: Sequence[str],
    n_pages: int | None = None,
    out: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Print the ESTIMATE table (reads image FILE NAMES and dev labels only; runs no model)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "gpu_estimate", paths.REPO_ROOT / "scripts" / "gpu_estimate.py"
    )
    if spec is None or spec.loader is None:
        raise FtError("scripts/gpu_estimate.py not found")
    ge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ge)
    speed = json.loads((paths.REPO_ROOT / "configs" / "spike_speed.json").read_text("utf-8"))
    plan = pr.plan(Path(data_root))
    pages = plan["n_pages"] if n_pages is None else n_pages
    by_doc = pr.discover_test_docs(Path(data_root))
    det_pages = sum(by_doc[d] for d in plan["determinism_docs"])
    smoke_pages = ge.count_pages(list(smoke_docs), Path(data_root) / "dev" / "labels")
    rows = estimate_rows(ge, speed, cfg.name, pages, batch_size, smoke_pages, det_pages)
    out(format_estimate(rows, pages, batch_size, det_pages))
    return {"n_pages": pages, "smoke_pages": smoke_pages, "det_pages": det_pages, "rows": rows}


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Argument parser of ``python -m shipdoc.predict_ft``."""
    p = argparse.ArgumentParser(
        prog="python -m shipdoc.predict_ft", description=__doc__.split("\n")[0]
    )
    sub = p.add_subparsers(dest="stage", required=True)

    def common(s: argparse.ArgumentParser) -> None:
        s.add_argument("--config", type=Path, default=Path("configs/spike_qwen35_4b_img_only.yaml"))
        s.add_argument("--runs-root", type=Path, default=None, help="default: SHIPDOC_RUNS_DIR")
        s.add_argument("--data-root", type=Path, default=None, help="default: SHIPDOC_DATA_DIR")
        s.add_argument("--folds", type=Path, default=paths.REPO_ROOT / "splits" / "folds.json")

    def adapter(s: argparse.ArgumentParser) -> None:
        s.add_argument("--adapter-dir", type=Path, required=True, help="<final run>/final")
        s.add_argument("--pin", default=None, help="this notebook's pinned 40-hex code SHA")

    s = sub.add_parser("estimate", help="ESTIMATE of T4 hours and CU (prints only)")
    common(s)
    s.add_argument("--batch-size", type=int, default=8)
    s.add_argument("--smoke-docs", default="splits/smoke5.json")
    s = sub.add_parser("verify", help="final adapter manifest verification (fails closed)")
    common(s)
    adapter(s)
    for name, help_ in (
        ("infer", "verify, merge, guard, smoke, resumable test run"),
        ("determinism", "second pass over 5 seeded test documents (fresh process)"),
    ):
        s = sub.add_parser(name, help=help_)
        common(s)
        adapter(s)
        s.add_argument("--run-id", required=True)
        s.add_argument("--decision", type=Path, required=True)
        if name == "infer":
            s.add_argument("--smoke-status", type=Path, required=True)
            s.add_argument("--smoke-docs", default="splits/smoke5.json")
            s.add_argument("--bench-docs", default="splits/bench12.json")
            s.add_argument("--batch-size", type=int, default=8)
            s.add_argument("--expect-docs", type=int, default=None)
            s.add_argument("--expect-pages", type=int, default=None)
    s = sub.add_parser("finalize", help="v1 checks + v2 checks on a submission folder")
    s.add_argument("--out-dir", type=Path, required=True)
    s.add_argument("--run-dir", type=Path, required=True)
    s.add_argument("--v0-dir", type=Path, default=None)
    s.add_argument("--recheck", type=Path, default=None)
    s.add_argument("--shapes-file", type=Path, default=paths.REPO_ROOT / reuse.SHAPES_REL)
    s.add_argument("--expect-pages", type=int, default=pr.EXPECTED_PAGES)
    s = sub.add_parser("check-flags", help="checks of review_flags.json and the predictions")
    s.add_argument("--out-dir", type=Path, required=True)
    s.add_argument("--calibrator", type=Path, required=True)
    s.add_argument("--field-target", type=float, default=flags.PRIMARY_TARGET)
    s.add_argument("--doc-target", type=float, default=flags.PRIMARY_TARGET)
    s = sub.add_parser("adopt-ocr", help="take the OCR timing / re-check record of a 04b folder")
    s.add_argument("--from-dir", type=Path, required=True)
    s.add_argument("--timing-out", type=Path, required=True)
    s.add_argument("--recheck-out", type=Path, required=True)
    s.add_argument("--expect-pages", type=int, default=pr.EXPECTED_PAGES)
    return p


def _load_folds(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    """Stage CLI. Exit 0 = the stage passed; 1 = refused / a check failed (counts only)."""
    a = build_parser().parse_args(argv)
    try:
        return _dispatch(a)
    except (FtError, pr.PredictError, flags.FlagsError) as exc:
        print(f"predict_ft {a.stage}: REFUSED: {exc}", file=sys.stderr)
        return 1


def _dispatch(a: argparse.Namespace) -> int:
    if a.stage == "adopt-ocr":
        res = adopt_ocr(a.from_dir, a.timing_out, a.recheck_out, a.expect_pages)
        print(f"adopt-ocr: {res}")
        return 0
    if a.stage == "check-flags":
        rep = check_flags(a.out_dir, calibrator=a.calibrator, field_target=a.field_target,
                          doc_target=a.doc_target)  # fmt: skip
        failed = [k for k, c in rep["flags_checks"].items() if not c["ok"]]
        print(f"check-flags: flags_ok={rep['flags_ok']} failed checks {failed}")
        return 0 if rep["flags_ok"] else 1
    if a.stage == "finalize":
        rep = finalize_v2(a.out_dir, run_dir=a.run_dir, shapes_file=a.shapes_file,
                          v0_dir=a.v0_dir, recheck_path=a.recheck,
                          expect_pages=a.expect_pages)  # fmt: skip
        failed = [k for k, c in rep["checks"].items() if not c["ok"]]
        print(f"finalize (v2): ok={rep['ok']} failed checks {failed}")
        return 0 if rep["ok"] else 1
    paths.apply_env()  # before anything can import transformers (the cli does the same)
    cfg = spike.load_config(a.config)
    data = Path(a.data_root) if a.data_root else paths.data_dir()
    runs = Path(a.runs_root) if a.runs_root else paths.runs_dir()
    folds = _load_folds(a.folds)
    if a.stage == "estimate":
        run_estimate(cfg=cfg, data_root=data, batch_size=a.batch_size,
                     smoke_docs=spike.load_doc_ids(a.smoke_docs))  # fmt: skip
        return 0

    def factory(adapter_dir: Path, sha: str) -> Any:
        return oof.MergedHfBackend(cfg.backend, adapter_dir, sha)

    if a.stage == "verify":
        verify_stage(adapter_dir=a.adapter_dir, cfg=cfg, folds=folds, pin_sha=a.pin,
                     reachable=oof.git_reachable)  # fmt: skip
        return 0
    if a.stage == "infer":
        rec = run_ft_infer(
            cfg=cfg, adapter_dir=a.adapter_dir, folds=folds, run_id=a.run_id, runs_root=runs,
            data_root=data, decision_path=a.decision, smoke_status_path=a.smoke_status,
            smoke_docs=spike.load_doc_ids(a.smoke_docs),
            bench_docs=spike.load_doc_ids(a.bench_docs),
            batch_size=a.batch_size, backend_factory=factory, pin_sha=a.pin,
            reachable=oof.git_reachable, expect_docs=a.expect_docs, expect_pages=a.expect_pages,
        )  # fmt: skip
        print(f"test run {a.run_id}: {rec['n_docs']} docs, {rec['n_pages']} pages, batch "
              f"{rec['decision']['batch_size']} (unscored: no test labels)")  # fmt: skip
        return 0
    rep = run_ft_determinism(
        cfg=cfg, adapter_dir=a.adapter_dir, folds=folds, run_id=a.run_id, runs_root=runs,
        data_root=data, decision_path=a.decision, backend_factory=factory, pin_sha=a.pin,
        reachable=oof.git_reachable,
    )  # fmt: skip
    print(f"determinism ok: {rep['n_identical']}/{rep['n_docs']} documents byte-identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
