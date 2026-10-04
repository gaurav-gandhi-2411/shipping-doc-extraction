"""Test-set prediction: the stages behind notebooks/04_predict_test.ipynb (safety submission v0).

The model run itself is `shipdoc.spike.run_spike` (split ``test``, images only, keyed format,
prompt v2, greedy, seed 42, the default Phase 3 merge + normalisation). This module adds what a
SUBMISSION needs around it, as stages of ``python -m shipdoc predict <stage>``:

``plan``         doc ids and page counts from the test image FOLDER NAMES (never labels), the
                 shard's documents and the determinism documents.
``smoke``        the 5-dev-doc smoke gate (same run + same seven checks as notebook 02), status
                 file written; the ``run`` stage refuses unless it passed for this code/config.
``batch``        the batch size: the dev run's stored bench result when it applies to this code,
                 config and model (else the bench is run on dev pages), then the batch-size
                 contract against the dev run (`shipdoc.runmeta`): a different size is refused.
``run``          the resumable per-document test run (a ``spike`` run, logprobs on, as the bench).
``determinism``  a second pass over 5 seeded test documents in a fresh process; the serialized
                 per-document predictions must be byte-identical to the first pass.
``assemble``     validate (JSON Schema, exactly the expected doc ids, determinism, provenance) and
                 write ``test_predictions.json``, ``trace.jsonl``, ``manifest.json`` and
                 ``validation_report.json`` into the submission folder.
``validate``     the schema + id-set check alone, on any predictions file.

Test data policy: test IMAGES are read only through the inference loaders; there are no test
labels. Nothing in this module prints, logs or stores an extracted value outside the submission
folder: every message and report carries counts, ids and key paths only. The submission folder
must be outside the repo tree (or under the gitignored ``submissions/``); `assert_outside_repo`.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import os
import random
import re
import shutil
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from shipdoc import paths, postrules, runmeta, smoke
from shipdoc import shard as shard_mod
from shipdoc.coerce import CoerceError, coerce_predictions, repair_predictions
from shipdoc.merge import FieldProvenance, merge_pages
from shipdoc.normalize import normalize_doc
from shipdoc.postrules import RuleConfig
from shipdoc.prompts import PROMPT_VERSION
from shipdoc.spike import (
    SpikeConfig,
    _read_trace,
    git_commit,
    load_config,
    load_doc_ids,
    make_backend,
    run_spike,
)

TEST_SPLIT = "test"
EXPECTED_DOCS = 200  # reports/recon.md: the test split, file level
EXPECTED_PAGES = 280
DET_N = 5  # documents re-decoded by the determinism pass
DET_MULTI = 2  # of which multi-page (so batching/merge paths are exercised), when available
DET_SEED = 42
DET_REPLAY_RULE = (
    "The second pass replays EXACTLY the generate calls of the first pass that decoded any page "
    "of the determinism documents: same pages, same order inside each call (left-padding order), "
    "recorded per page in meta.exec_batch. Compared byte for byte: the serialized prediction of "
    "each determinism document, and the raw text of every page of every replayed call (the "
    "companions of the determinism pages included)."
)
DET_RULE = (
    "From the documents of this run (sorted by doc_id) draw 2 multi-page and 3 single-page "
    "documents with random.Random(42).sample (fewer of a kind: topped up from the other kind); "
    "ids come from the image folder names, so the selection never looks at labels or output."
)
SUBMISSION_VERSION = "v0"
SMOKE_STATE_PASSED = "passed"
STACK_PACKAGES = ("transformers", "torch", "xgrammar", "pycountry")
PAGE_STEM = re.compile(r"^(?P<doc>.+)_p(?P<page>\d+)$")
OUT_FILES = (
    "test_predictions.json",
    "trace.jsonl",
    "manifest.json",
    "validation_report.json",
    "rules.jsonl",
)
REJECTED_NAME = "test_predictions.REJECTED.json"
MAX_SAMPLE = 5  # ids listed in a report line; counts carry the rest


class PredictError(RuntimeError):
    """A stage refused or a check failed; the message carries counts and ids only."""


class GateError(PredictError):
    """The smoke gate has not passed for this code/config: the test run must not start."""


class BatchContractError(PredictError):
    """The test batch size differs from the dev run's: outputs are not comparable."""


class DeterminismError(PredictError):
    """The second decode of the determinism documents differs from the first."""


class ReplayPlanError(PredictError):
    """The batch composition of the first pass cannot be rebuilt from its trace (old run)."""


# --------------------------------------------------------------------------------------------
# Documents (ids come from image folder names only)
# --------------------------------------------------------------------------------------------


def discover_test_docs(data_root: Path, split: str = TEST_SPLIT) -> dict[str, int]:
    """``{doc_id: n_pages}`` from the file names in ``<data_root>/<split>/images``, sorted by id.

    Only names are read (no image decode, no labels). Raises PredictError for a missing or empty
    folder, or for a document whose page numbers are not 1..n.
    """
    folder = Path(data_root) / split / "images"
    if not folder.is_dir():
        raise PredictError(f"no image folder {folder}")
    pages: dict[str, list[int]] = {}
    for p in folder.iterdir():
        m = PAGE_STEM.match(p.stem)
        if p.is_file() and m:
            pages.setdefault(m["doc"], []).append(int(m["page"]))
    if not pages:
        raise PredictError(f"no page images (<doc>_p<n>.<ext>) in {folder}")
    bad = [d for d, ns in pages.items() if sorted(ns) != list(range(1, len(ns) + 1))]
    if bad:
        raise PredictError(f"{len(bad)} documents with missing/duplicate page numbers {bad[:5]}")
    return {d: len(pages[d]) for d in sorted(pages)}


def select_determinism_docs(
    pages: Mapping[str, int], n: int = DET_N, seed: int = DET_SEED, multi: int = DET_MULTI
) -> list[str]:
    """The seeded determinism documents (`DET_RULE`), sorted. All of them when there are <= n."""
    ids = sorted(pages)
    if len(ids) <= n:
        return ids
    rng = random.Random(seed)  # noqa: S311 - a reproducible selection, not security
    multis = [d for d in ids if pages[d] > 1]
    singles = [d for d in ids if pages[d] == 1]
    take_m = min(multi, len(multis), n)
    take_s = min(n - take_m, len(singles))
    chosen = rng.sample(multis, take_m) + rng.sample(singles, take_s)
    short = n - len(chosen)
    if short > 0:  # not enough singles: top up from the remaining multi-page documents
        rest = [d for d in multis if d not in chosen]
        chosen += rng.sample(rest, short)
    return sorted(chosen)


def plan(data_root: Path, shard: str = "0/1") -> dict[str, Any]:
    """Counts, shard documents and determinism documents for the test split (ids only)."""
    pages = discover_test_docs(data_root)
    i, k = shard_mod.parse_shard(shard)
    mine = shard_mod.assign_shards(list(pages.items()), k)[i]
    return {
        "n_docs": len(pages),
        "n_pages": sum(pages.values()),
        "docs": list(pages),
        "docs_sha": runmeta.docs_sha(list(pages)),
        "shard": shard,
        "shard_docs": mine,
        "shard_n_docs": len(mine),
        "shard_n_pages": sum(pages[d] for d in mine),
        "determinism_rule": DET_RULE,
        "determinism_docs": select_determinism_docs({d: pages[d] for d in mine}),
    }


# --------------------------------------------------------------------------------------------
# Provenance helpers
# --------------------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def phase3_rules() -> dict[str, Any]:
    """The Phase 3 rule switches the run applies, READ from the code defaults it uses.

    `run_spike` calls ``merge_pages(parsed_pages, provenance)`` and ``normalize_doc(merged)`` with
    no keyword overrides, so the signature defaults ARE the pipeline (tests replay the traces to
    prove it). Rung labels are those of reports/ablations.md section G3.
    """
    mp = {n: p.default for n, p in inspect.signature(merge_pages).parameters.items()}
    nd = {n: p.default for n, p in inspect.signature(normalize_doc).parameters.items()}
    prov_path = paths.REPO_ROOT / "meta" / "field_provenance.json"
    prov = FieldProvenance.load()
    return {
        "merge_pages": {
            k: mp[k] for k in ("provenance_aware", "drop_header_rows", "drop_null_rows")
        }
        | {"total_page_hints": mp["total_page_hints"]},
        "normalize_doc": {k: nd[k] for k in ("dates", "numbers", "codes", "date_order")},
        "ladder": {
            "R1a_dates_to_iso": bool(nd["dates"]),
            "R1b_cluster_date_order": nd["date_order"] is not None,
            "R2_number_normalisation": bool(nd["numbers"]),
            "R2b_code_normalisation": bool(nd["codes"]),
            "R5_provenance_aware_merge": bool(mp["provenance_aware"]),
            "R6_ocr_total_pointer": mp["total_page_hints"] is not None,
            "R7_repeated_header_row_filter": bool(mp["drop_header_rows"]),
            "R8_drop_all_null_rows": bool(mp["drop_null_rows"]),
            "R3_R4_validators": "flags only, never applied to values (not run here)",
        },
        "field_provenance": {
            "source": prov.source,
            "sha256": sha256_file(prov_path) if prov_path.is_file() else None,
        },
    }


def collect_env() -> dict[str, Any]:
    """Python and dependency versions (None = not installed) and the GPU name, when visible."""
    versions: dict[str, str | None] = {}
    for name in STACK_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    gpu = None
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
    return {"python": sys.version.split()[0], "packages": versions, "gpu": gpu}


def assert_outside_repo(out_dir: Path) -> None:
    """Refuse a submission folder inside the repo tree, except under the gitignored submissions/."""
    out = Path(out_dir).resolve()
    repo = paths.REPO_ROOT.resolve()
    if out != repo and repo not in out.parents:
        return
    ignored = repo / "submissions"
    if out == ignored or ignored in out.parents:
        return
    raise PredictError(
        f"{out} is inside the repository tree: test predictions never enter git. Use a folder "
        "outside the repo (SHIPDOC_SUBMISSIONS_DIR) or the gitignored submissions/."
    )


def _write_json(path: Path, obj: Any) -> None:
    """Atomic, LF-terminated, ascii-escaped JSON (identical bytes on every OS)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------------
# Smoke gate
# --------------------------------------------------------------------------------------------


def _identity(cfg: SpikeConfig, backend: Any) -> dict[str, Any]:
    """What a smoke status / decision is bound to: code, config and model revision."""
    return {
        "code_sha": git_commit(),
        "config_hash": cfg.config_hash,
        "model_revision": backend.revision,
    }


def run_smoke_gate(
    cfg: SpikeConfig,
    backend: Any,
    smoke_ids: Sequence[str],
    run_id: str,
    runs_root: Path,
    status_path: Path,
    data_root: Path | None = None,
) -> dict[str, Any]:
    """Run the 5-dev-doc smoke (same call as notebook 02) and check it; writes the status file.

    State ``passed`` / ``failed`` (an assertion failed) / ``error`` (the run or the check raised:
    fail closed, could not verify = fail). A passed status for the same run id, code, config and
    model revision is not repeated.
    """
    ident = {**_identity(cfg, backend), "run_id": run_id}
    status_path = Path(status_path)
    if status_path.is_file():
        old = _read_json(status_path)
        if old.get("state") == SMOKE_STATE_PASSED and all(
            old.get(k) == v for k, v in ident.items()
        ):
            return {**old, "skipped": True}
    root = paths.data_dir() if data_root is None else Path(data_root)
    smoke_root = Path(runs_root) / "smoke"
    t0 = time.time()
    checks: list[smoke.Check] = []
    warns: list[smoke.Warn] = []
    error = None
    try:
        run_spike(
            cfg, list(smoke_ids), "dev", run_id, backend, resume=True, limit=smoke.SMOKE_N,
            runs_root=smoke_root, data_root=root, logprobs=True,
        )  # fmt: skip
        run_dir = smoke_root / run_id
        checks = smoke.load_and_check(
            run_dir, root / "dev" / "labels", cfg.backend.max_new_tokens, require_logprobs=True
        )
        warns = smoke.load_warnings(run_dir)
        state = SMOKE_STATE_PASSED if all(c.passed for c in checks) else "failed"
    except Exception as exc:  # noqa: BLE001 - any failure to verify is a failed gate
        state, error = "error", f"{type(exc).__name__}: {str(exc)[:300]}"
    status = {
        **ident,
        "state": state,
        "seconds": round(time.time() - t0, 1),
        "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail} for c in checks],
        "warnings": [{"name": w.name, "count": w.count, "detail": w.detail} for w in warns],
        "error": error,
    }
    _write_json(status_path, status)
    if checks:
        print(smoke.format_table(cfg.name, checks, warns))
    return status


def require_smoke_passed(status_path: Path, cfg: SpikeConfig, backend: Any) -> dict[str, Any]:
    """The smoke status, or GateError unless it passed for THIS code, config and model."""
    status_path = Path(status_path)
    if not status_path.is_file():
        raise GateError(f"smoke gate not run ({status_path} missing): the test run is not started")
    status = _read_json(status_path)
    if status.get("state") != SMOKE_STATE_PASSED:
        raise GateError(f"smoke gate state {status.get('state')!r}: the test run is not started")
    for key, want in _identity(cfg, backend).items():
        if status.get(key) != want:
            raise GateError(
                f"smoke status was recorded for a different {key}; rerun the smoke gate "
                "(the test run is not started)"
            )
    return status


# --------------------------------------------------------------------------------------------
# Batch size and the dev contract
# --------------------------------------------------------------------------------------------


def dev_batch_size(dev_run_dir: Path | None) -> int | None:
    """Batch size stored in the dev run's manifest, None if there is no dev run to compare."""
    if dev_run_dir is None or not Path(dev_run_dir).is_dir():
        return None
    m = runmeta.read_manifest(Path(dev_run_dir))
    if m is None or not isinstance(m.get("batch_size"), int):
        return None
    return int(m["batch_size"])


def check_batch_contract(dev_run_dir: Path | None, batch_size: int) -> str:
    """'checked' when equal to the dev run's size, 'unchecked: ...' when there is none.

    Raises BatchContractError when the sizes differ: batched greedy decoding is only comparable
    at equal batch size (`runmeta.require_same_batch_size`).
    """
    dev = dev_batch_size(dev_run_dir)
    if dev is None:
        return "unchecked: no dev run manifest to compare with"
    if dev != batch_size:
        raise BatchContractError(
            f"batch size contract: the dev run used {dev} but the test run would use "
            f"{batch_size}. Greedy outputs are only comparable at equal batch size: set "
            f"BATCH_SIZE = {dev} (manual) or rerun the dev run with {batch_size}."
        )
    return "checked"


def decide_batch_size(
    cfg: SpikeConfig,
    backend: Any,
    bench_dir: Path,
    bench_docs: Sequence[str],
    decision_path: Path,
    dev_run_dir: Path | None = None,
    manual: int | None = None,
    data_root: Path | None = None,
) -> dict[str, Any]:
    """Pick the test batch size, enforce the dev contract, write the decision file.

    ``manual`` (an int): used as is, the bench is skipped (recorded as source ``manual``).
    Otherwise the bench result in `bench_dir` is reused when it applies to this code, config and
    model (`bench.load_bench_result`); when there is none, the dev run's stored result is copied
    there first and verified the same way; when that does not apply either, the bench is run on
    the 12 dev pages exactly as notebook 02 does. The chosen size must equal the dev run's.
    """
    from shipdoc import bench

    bench_dir = Path(bench_dir)
    note: str | None = None
    reused = False
    info: dict[str, Any] | None = None
    if manual is not None:
        if isinstance(manual, bool) or manual < 1:
            raise PredictError(f"batch size must be an int >= 1, got {manual!r}")
        size, source = int(manual), "manual"
    else:
        source = "bench"
        result_path = bench_dir / bench.RESULT_NAME
        dev_manifest = runmeta.read_manifest(Path(dev_run_dir)) if dev_run_dir else None
        dev_bench = ((dev_manifest or {}).get("bench") or {}).get("path")
        if not result_path.is_file() and dev_bench and Path(dev_bench).is_file():
            bench_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(dev_bench, result_path)  # verified below, not trusted
            note = "copied the dev run's stored bench result"
        result: dict[str, Any] | None = None
        if result_path.is_file():
            try:
                result = bench.load_bench_result(result_path, cfg, backend)
                reused = True
            except ValueError as exc:
                note = f"stored bench result does not apply ({exc}); bench rerun on dev pages"
        if result is None:
            root = paths.data_dir() if data_root is None else Path(data_root)
            result = bench.run_bench(cfg, backend, list(bench_docs), bench_dir, data_root=root)
        size = bench.chosen_from_result(result)
        info = {
            "path": str(result_path),
            "chosen": size,
            "deviation": result.get("deviation"),
            "sha256": sha256_file(result_path),
        }
    contract = check_batch_contract(dev_run_dir, size)
    decision = {
        "batch_size": size,
        "source": source,
        "bench": info,
        "bench_reused": reused,
        "note": note,
        "dev_run_dir": str(dev_run_dir) if dev_run_dir else None,
        "dev_batch_size": dev_batch_size(dev_run_dir),
        "contract": contract,
        **_identity(cfg, backend),
    }
    _write_json(Path(decision_path), decision)
    return decision


# --------------------------------------------------------------------------------------------
# The test run and the determinism pass
# --------------------------------------------------------------------------------------------


def _spike_kwargs(decision: Mapping[str, Any]) -> dict[str, Any]:
    bench_info = decision.get("bench")
    return {
        "batch_size": int(decision["batch_size"]),
        "bench_info": (
            {k: bench_info[k] for k in ("path", "chosen", "deviation")} if bench_info else None
        ),
    }


def run_test(
    cfg: SpikeConfig,
    backend: Any,
    run_id: str,
    runs_root: Path,
    data_root: Path,
    decision: Mapping[str, Any],
    smoke_status_path: Path,
    shard: str = "0/1",
    expect_docs: int | None = None,
    expect_pages: int | None = None,
) -> dict[str, Any]:
    """The resumable test run of this shard (a ``spike`` run on split ``test``).

    Refuses (nothing decoded) unless the smoke gate passed for this code/config/model and the
    batch size still equals the dev run's. Document ids come from the image folder names.
    Returns the run's metrics (unscored: the test split has no labels).
    """
    require_smoke_passed(smoke_status_path, cfg, backend)
    check_batch_contract(
        Path(decision["dev_run_dir"]) if decision.get("dev_run_dir") else None,
        int(decision["batch_size"]),
    )
    pages = discover_test_docs(data_root)
    if expect_docs is not None and len(pages) != expect_docs:
        raise PredictError(f"{len(pages)} test documents found, expected {expect_docs}")
    if expect_pages is not None and sum(pages.values()) != expect_pages:
        raise PredictError(f"{sum(pages.values())} test pages found, expected {expect_pages}")
    out_id = shard_mod.shard_run_id(run_id, shard)
    metrics = run_spike(
        cfg, list(pages), TEST_SPLIT, out_id, backend, resume=True, runs_root=Path(runs_root),
        data_root=Path(data_root), logprobs=True, shard=shard, base_run_id=run_id,
        **_spike_kwargs(decision),
    )  # fmt: skip
    _write_json(
        Path(runs_root) / out_id / "env.json", {**collect_env(), "backend": backend.model_id}
    )
    return metrics


def _docs_json(trace: Mapping[str, Any]) -> str:
    """Serialized prediction of one traced document (key order is part of the bytes)."""
    return json.dumps(trace["prediction"])


def compare_runs(first: Sequence[Mapping[str, Any]], second: Sequence[Mapping[str, Any]],
                 ids: Sequence[str]) -> dict[str, Any]:  # fmt: skip
    """Counts of identical / different / missing documents (and pages) between two passes.

    Compares the serialized per-document predictions byte for byte; raw page texts are compared
    as a second, informational count. Never returns a value.
    """
    a = {t["doc_id"]: t for t in first}
    b = {t["doc_id"]: t for t in second}
    missing = [d for d in ids if d not in a or d not in b]
    both = [d for d in ids if d in a and d in b]
    same = [d for d in both if _docs_json(a[d]) == _docs_json(b[d])]
    pages = [(p["raw_text"], q["raw_text"]) for d in both
             for p, q in zip(a[d]["pages"], b[d]["pages"], strict=False)]  # fmt: skip
    page_mismatch = sum(len(a[d]["pages"]) != len(b[d]["pages"]) for d in both)
    return {
        "n_docs": len(ids),
        "n_identical": len(same),
        "n_different": len(both) - len(same),
        "n_missing": len(missing),
        "pages_compared": len(pages),
        "pages_raw_text_identical": sum(x == y for x, y in pages),
        "docs_with_page_count_change": page_mismatch,
        "different_doc_ids": [d for d in both if d not in same][:MAX_SAMPLE],
        "ok": bool(ids) and len(same) == len(ids) and not page_mismatch,
    }


def replay_plan(
    first: Sequence[Mapping[str, Any]], ids: Sequence[str], recorded: bool = True
) -> tuple[list[list[tuple[str, int]]], list[str]]:
    """(batches, documents) the second pass must run to rebuild the first pass's composition.

    A batch is the ordered ``(doc_id, page_idx)`` member list of one generate call
    (``meta.exec_batch``) that decoded at least one page of `ids`; batches come in the order of
    their call id. A page without a stamp was decoded alone (the runner stamps multi-page calls
    only) when `recorded` is true (``manifest.exec_batch_recorded``, or batch size 1). `documents`
    are all documents those batches touch (sorted). Raises `ReplayPlanError` when the trace was
    written before the stamp existed (`recorded` false and a page without a stamp), when a
    call members are not all found with the same stamp, or when a document id is not traced.
    """
    Call = tuple[str, tuple[tuple[str, int], ...]]
    pages: dict[tuple[str, int], Call] = {}
    for t in first:
        for k, p in enumerate(t["pages"]):
            eb = (p.get("meta") or {}).get("exec_batch")
            if not eb and recorded:  # decoded alone
                pages[(t["doc_id"], k)] = (f"{t['doc_id']}:{k}", ((t["doc_id"], k),))
                continue
            if not eb:
                raise ReplayPlanError(
                    f"{t['doc_id']} page {k + 1} has no meta.exec_batch: this trace predates the "
                    "batch-composition record; re-run the main pass with the current code"
                )
            pages[(t["doc_id"], k)] = (eb["id"], tuple((d, int(i)) for d, i in eb["members"]))
    wanted: dict[Call, None] = {}
    for d in ids:
        found = [key for key in pages if key[0] == d]
        if not found:
            raise ReplayPlanError(f"determinism document {d} is not in the first-pass trace")
        for key in found:
            wanted[pages[key]] = None
    for call_id, members in wanted:
        stray = [m for m in members if pages.get(m) != (call_id, members)]
        if stray:
            raise ReplayPlanError(
                f"generate call {call_id}: {len(stray)} of its {len(members)} member pages are "
                "not stamped with the same call in the trace; cannot rebuild its composition"
            )
    batches = [list(members) for _id, members in sorted(wanted, key=lambda w: w[0])]
    return batches, sorted({d for b in batches for d, _ in b})


def compare_replayed_pages(
    first: Sequence[Mapping[str, Any]],
    second: Sequence[Mapping[str, Any]],
    extra_pages: Sequence[Mapping[str, Any]],
    batches: Sequence[Sequence[tuple[str, int]]],
) -> dict[str, Any]:
    """Counts over every page of every replayed call: raw text equal, composition equal, missing.

    `second` are the traces of the replay run, `extra_pages` its ``replay_pages.jsonl`` lines
    (pages of documents the replay could not complete). Never returns a value.
    """
    mine: dict[tuple[str, int], tuple[str, Any]] = {}
    for t in first:
        for k, p in enumerate(t["pages"]):
            mine[(t["doc_id"], k)] = (p["raw_text"], (p.get("meta") or {}).get("exec_batch"))
    redo: dict[tuple[str, int], tuple[str, Any]] = {}
    for t in second:
        for k, p in enumerate(t["pages"]):
            redo[(t["doc_id"], k)] = (p["raw_text"], (p.get("meta") or {}).get("exec_batch"))
    for r in extra_pages:
        redo[(r["doc_id"], int(r["page"]) - 1)] = (r["raw_text"], r.get("exec_batch"))
    keys = [m for b in batches for m in b]
    missing = [k for k in keys if k not in redo]
    got = [k for k in keys if k in redo]
    same = {k for k in got if redo[k][0] == mine[k][0]}
    comp = [
        k for k in got if (redo[k][1] or {}).get("members") == (mine[k][1] or {}).get("members")
    ]
    return {
        "replayed_batches": len(batches),
        "replayed_pages": len(keys),
        "replayed_pages_missing": len(missing),
        "replayed_pages_identical": len(same),
        "replayed_composition_identical": len(comp),
        "composition_replicated": bool(keys) and not missing and len(comp) == len(keys),
        "different_pages": [f"{d}:p{i + 1}" for d, i in got if (d, i) not in same][:MAX_SAMPLE],
    }


def run_determinism(
    cfg: SpikeConfig,
    backend: Any,
    run_id: str,
    runs_root: Path,
    data_root: Path,
    decision: Mapping[str, Any],
    shard: str = "0/1",
) -> dict[str, Any]:
    """Second pass over the shard's determinism documents; writes ``determinism.json``.

    Run it as its own process after the main run so the decode is fresh (new model load, new CUDA
    context). The same batch size and bench record as the main run. Raises DeterminismError (after
    writing the file) when any serialized prediction differs; the message has counts only.
    """
    out_id = shard_mod.shard_run_id(run_id, shard)
    main_dir = Path(runs_root) / out_id
    prog = main_dir / "progress.json"
    if not prog.is_file() or _read_json(prog).get("status") != "complete":
        raise PredictError(f"{main_dir}: the main run is not complete; run it before this pass")
    first = _read_trace(main_dir / "trace.jsonl")
    pages = {t["doc_id"]: len(t["pages"]) for t in first}
    ids = select_determinism_docs(pages)
    man = runmeta.read_manifest(main_dir) or {}
    recorded = man.get("exec_batch_recorded") is True or man.get("batch_size") == 1
    batches, touched = replay_plan(first, ids, recorded)
    det_id = f"{out_id}_det"
    det_dir = Path(runs_root) / det_id
    run_spike(
        cfg, touched, TEST_SPLIT, det_id, backend, resume=True, runs_root=Path(runs_root),
        data_root=Path(data_root), logprobs=True, replay_batches=batches, fallback=False,
        **_spike_kwargs(decision),
    )  # fmt: skip
    second = _read_trace(det_dir / "trace.jsonl")
    extra_file = det_dir / "replay_pages.jsonl"
    extra = (
        [json.loads(ln) for ln in extra_file.read_text(encoding="utf-8").splitlines() if ln]
        if extra_file.is_file()
        else []
    )
    result = compare_runs(first, second, ids)
    replayed = compare_replayed_pages(first, second, extra, batches)
    result["ok"] = bool(
        result["ok"]
        and replayed["composition_replicated"]
        and replayed["replayed_pages_identical"] == replayed["replayed_pages"]
    )
    report = {
        **result,
        **replayed,
        "replay_rule": DET_REPLAY_RULE,
        "rule": DET_RULE,
        "seed": DET_SEED,
        "doc_ids": ids,
        "batch_size": int(decision["batch_size"]),
        "second_pass_run": det_id,
        "shard": shard,
    }
    _write_json(main_dir / "determinism.json", report)
    if not result["ok"]:
        raise DeterminismError(
            f"determinism FAILED on {out_id}: {result['n_different']} of {result['n_docs']} "
            f"documents differ, {result['n_missing']} missing, "
            f"{result['docs_with_page_count_change']} with a changed page count "
            f"(raw page text identical on {result['pages_raw_text_identical']}/"
            f"{result['pages_compared']} pages; ids {result['different_doc_ids']}); replayed "
            f"{replayed['replayed_batches']} calls / {replayed['replayed_pages']} pages: "
            f"{replayed['replayed_pages_identical']} identical, composition replicated "
            f"{replayed['composition_replicated']}, different {replayed['different_pages']}"
        )
    return report


# --------------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------------


def load_strict_json(path: Path) -> tuple[Any, list[str]]:
    """Parse a JSON file and also return the duplicate object keys met anywhere (names only)."""
    dups: list[str] = []

    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        seen: set[str] = set()
        for k, _ in pairs:
            if k in seen:
                dups.append(k)
            seen.add(k)
        return dict(pairs)

    data = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=hook)
    return data, dups


def _path_pattern(parts: Sequence[Any], root_is_docs: bool = True) -> str:
    """``*/header/invoice_date`` style path: the doc id and row index are replaced by wildcards."""
    out = []
    for i, p in enumerate(parts):
        if isinstance(p, int):
            out.append("[]")
        elif i == 0 and root_is_docs:
            out.append("*")
        else:
            out.append(str(p))
    return "/".join(out) or "<root>"


def validate_schema(predictions: Any, schema: Mapping[str, Any]) -> dict[str, Any]:
    """JSON Schema validation of a whole submission; the report has no instance values.

    ``by_kind`` counts ``<keyword> @ <path pattern>`` (a ``oneOf`` failure also lists its
    branches as ``oneOf> <keyword> @ <path>``); messages are never copied because jsonschema
    messages quote the offending value.
    """
    import jsonschema

    validator = jsonschema.Draft202012Validator(dict(schema))
    kinds: Counter[str] = Counter()
    n = 0
    for err in validator.iter_errors(predictions):
        n += 1
        kinds[f"{err.validator} @ {_path_pattern(list(err.absolute_path))}"] += 1
        for sub in err.context or ():
            kinds[f"oneOf> {sub.validator} @ {_path_pattern(list(sub.absolute_path))}"] += 1
    return {"ok": n == 0, "n_errors": n, "by_kind": dict(sorted(kinds.items()))}


def check_ids(found: Sequence[str], expected: Sequence[str]) -> dict[str, Any]:
    """Exactly the expected ids, no extra, no duplicate (counts, plus up to 5 ids of each kind)."""
    exp = set(expected)
    counts = Counter(found)
    missing = sorted(exp - set(counts))
    extra = sorted(set(counts) - exp)
    dup = sorted(d for d, c in counts.items() if c > 1)
    return {
        "ok": not (missing or extra or dup) and len(found) == len(expected),
        "n_expected": len(expected),
        "n_found": len(found),
        "n_missing": len(missing),
        "n_extra": len(extra),
        "n_duplicate": len(dup),
        "missing": missing[:MAX_SAMPLE],
        "extra": extra[:MAX_SAMPLE],
        "duplicate": dup[:MAX_SAMPLE],
    }


def validate_file(
    pred_path: Path, schema_path: Path, expected_ids: Sequence[str]
) -> dict[str, Any]:
    """Schema + id-set + duplicate-key validation of a predictions file (the ``validate`` stage)."""
    data, dups = load_strict_json(pred_path)
    schema = _read_json(schema_path)
    is_map = isinstance(data, dict)
    report = {
        "schema": validate_schema(data, schema),
        "ids": check_ids(list(data) if is_map else [], expected_ids),
        "duplicate_keys": {"ok": not dups, "n": len(dups), "keys": sorted(set(dups))[:MAX_SAMPLE]},
        "schema_sha256": sha256_file(schema_path),
    }
    report["ok"] = all(report[k]["ok"] for k in ("schema", "ids", "duplicate_keys"))
    return report


# --------------------------------------------------------------------------------------------
# Assemble the submission folder
# --------------------------------------------------------------------------------------------


def _run_components(runs_root: Path, run_id: str) -> tuple[Path, dict[str, Any], list[Path]]:
    """(run dir, its manifest, the folders that hold the sessions / determinism / env files)."""
    run_dir = Path(runs_root) / run_id
    m = runmeta.read_manifest(run_dir)
    if m is None:
        raise PredictError(f"{run_dir}: no manifest.json (run not started?)")
    if m.get("shard") == "merged":
        return run_dir, m, [Path(runs_root) / n for n in m["merged_from"]]
    if m.get("shard") != "0/1":
        raise PredictError(
            f"{run_dir} is shard {m.get('shard')}: merge the shards first "
            "(python -m shipdoc merge-shards ... --split test)"
        )
    return run_dir, m, [run_dir]


def _check(ok: bool, detail: str) -> dict[str, Any]:
    return {"ok": bool(ok), "detail": detail}


def assemble_submission(
    cfg: SpikeConfig,
    run_id: str,
    runs_root: Path,
    data_root: Path,
    out_dir: Path,
    schema_path: Path,
    dev_run_dir: Path | None = None,
    expect_docs: int = EXPECTED_DOCS,
    expect_pages: int = EXPECTED_PAGES,
    expect_code_sha: str | None = None,
    allow_dirty: bool = False,
    require_stack: bool = False,
    smoke_status_path: Path | None = None,
    sample_submission: Path | None = None,
    rule_cfg: RuleConfig | None = None,
    shapes_file: Path | None = None,
    ocr_cache: Path | None = None,
) -> dict[str, Any]:
    """Validate a finished (or merged) test run and write the submission folder.

    Post-processing v1 (``shipdoc.postrules``: R1, R2, R3) runs on every traced document BEFORE
    coercion / repair, DEFAULT ON (`rule_cfg` None = all three on; ``RuleConfig.all_off()`` gives
    the pre-v1 predictions byte for byte). R3 reads the frozen ``meta/slot_shapes.json``
    (`shapes_file`; an unreadable file is a blocking ``post_rules`` check), R2 reads the OCR cache
    (`ocr_cache`, default the configured one; a waybill without cached OCR keeps the model's
    value and is counted as skipped ``no_ocr`` in the manifest and the report). The per-document
    rule records (field names and row indexes only) go to ``rules.jsonl``.

    Blocking checks (``ok`` false if any fails): run complete; exactly the ids of the image folder
    (and of the assignment sample submission, when given); no duplicate documents; expected doc and
    page counts; JSON Schema; determinism passes of every component; same batch size as the dev
    run (when a dev run exists); code SHA clean and equal to ``expect_code_sha``; model revision,
    config hash, seed, split, logprobs and prompt version as the production config; and with
    ``require_stack`` the real model and the recorded dependency versions.
    On failure the predictions go to ``test_predictions.REJECTED.json`` so the submittable name
    never holds an unchecked file. Returns the validation report (also written to disk).
    """
    out_dir = Path(out_dir)
    assert_outside_repo(out_dir)
    run_dir, man, comps = _run_components(runs_root, run_id)
    checks: dict[str, dict[str, Any]] = {}

    prog = run_dir / "progress.json"
    status = _read_json(prog).get("status") if prog.is_file() else None
    checks["run_complete"] = _check(status == "complete", f"progress status {status!r}")

    pages_on_disk = discover_test_docs(data_root)
    expected_ids = list(pages_on_disk)
    traces = _read_trace(run_dir / "trace.jsonl")
    found_ids = [t["doc_id"] for t in traces]
    n_pages = sum(len(t["pages"]) for t in traces)
    checks["counts"] = _check(
        len(pages_on_disk) == expect_docs
        and sum(pages_on_disk.values()) == expect_pages
        and len(traces) == expect_docs
        and n_pages == expect_pages,
        f"image folder {len(pages_on_disk)} docs / {sum(pages_on_disk.values())} pages; trace "
        f"{len(traces)} docs / {n_pages} pages; expected {expect_docs} / {expect_pages}",
    )
    ids_report = check_ids(found_ids, expected_ids)
    checks["doc_ids"] = {**ids_report, "detail": _ids_detail(ids_report)}
    if sample_submission is not None and Path(sample_submission).is_file():
        sample_ids = list(_read_json(sample_submission))
        s = check_ids(sample_ids, expected_ids)
        checks["ids_match_sample_submission"] = _check(
            s["ok"], f"assignment sample submission: {_ids_detail(s)}"
        )
    rcfg = RuleConfig() if rule_cfg is None else rule_cfg
    shapes, shapes_sha, shapes_err = None, None, None
    if rcfg.r3:
        try:
            shapes, shapes_sha = postrules.load_slot_shapes(shapes_file)
        except (OSError, ValueError) as exc:  # fail closed: R3 is on but its artefact is unusable
            shapes_err = type(exc).__name__
    predictions, rule_records, rules_summary = postrules.postprocess_traces(
        traces, rcfg, shapes, ocr_cache
    )
    if shapes_sha:
        shapes_note = f"slot shapes sha256 {shapes_sha[:12]}"
    else:
        shapes_note = f"slot shapes UNUSABLE ({shapes_err})" if shapes_err else "R3 off"
    checks["post_rules"] = _check(
        shapes_err is None,
        f"switches {rcfg.as_dict()}; {shapes_note}; touched docs {rules_summary['touched_docs']}; "
        f"skipped {rules_summary['skipped']}",
    )
    try:  # numbers -> plain-number strings (the schema types every field string|null)
        predictions = coerce_predictions(predictions)
        checks["number_coercion"] = _check(True, "numeric fields are plain-number strings")
    except CoerceError as exc:  # fail closed: the unchanged predictions go to the REJECTED name
        checks["number_coercion"] = _check(False, str(exc))
    # the one schema repair: a date that can never satisfy the schema pattern becomes null (counted;
    # traces written by the repair-aware runner already carry theirs, so both are summed)
    predictions, at_assembly = repair_predictions(predictions)
    in_trace = sum(len(t.get("schema_repairs") or []) for t in traces)
    repairs = {
        "n": in_trace + len(at_assembly),
        "in_trace": in_trace,
        "at_assembly": len(at_assembly),
        "reason": "schema_invalid_date",
        "events": at_assembly,
    }
    checks["date_repair"] = _check(
        True, f"{repairs['n']} schema-invalid date(s) set to null ({in_trace} in the run traces, "
        f"{len(at_assembly)} at assembly); values are never stored"
    )  # fmt: skip
    schema = _read_json(schema_path)
    sch = validate_schema(predictions, schema)
    checks["json_schema"] = {**sch, "detail": f"{sch['n_errors']} schema errors"}

    det = _determinism(comps)
    checks["determinism"] = det
    contract = _dev_contract(dev_run_dir, run_dir)
    checks["batch_size_contract"] = contract

    code_sha = str(man.get("code_sha"))
    clean = (allow_dirty or "+dirty" not in code_sha) and code_sha != "unknown"
    checks["code_sha_clean"] = _check(clean, f"code_sha {code_sha}")
    if expect_code_sha:
        checks["code_sha_is_pin"] = _check(
            code_sha.removesuffix("+dirty") == expect_code_sha,
            f"{code_sha} vs pin {expect_code_sha}",
        )
    prompt_versions = {(t.get("prompt") or {}).get("version") for t in traces}
    # a mock run has revision "mock"; `require_stack` below refuses a mock run for a real submission
    model_rev = (man.get("model") or {}).get("revision")
    mismatches = [
        name
        for name, ok in (
            ("model.revision", model_rev in (cfg.backend.revision, "mock")),
            ("config.hash", (man.get("config") or {}).get("hash") == cfg.config_hash),
            ("seed", man.get("seed") == cfg.backend.seed == 42),
            ("split", man.get("split") == TEST_SPLIT),
            ("arm", man.get("arm") == cfg.arm == "img_only"),
            ("output_format", man.get("output_format") == cfg.backend.output_format == "json"),
            ("logprobs", man.get("logprobs") is True),
            ("prompt.version", prompt_versions == {PROMPT_VERSION}),
        )
        if not ok
    ]
    checks["production_config"] = _check(
        not mismatches, "matches" if not mismatches else f"differs: {mismatches}"
    )
    envs = [_read_json(c / "env.json") if (c / "env.json").is_file() else None for c in comps]
    env = envs[0] if envs and all(e == envs[0] for e in envs) and envs[0] else None
    checks["dependencies_recorded"] = _check(
        env is not None,
        "env.json present and identical in every component"
        if env
        else "env.json missing or differing between shards",
    )
    if require_stack:
        missing_pk = [k for k, v in (env or {}).get("packages", {}).items() if v is None]
        real = (man.get("model") or {}).get("id") != "mock"
        checks["real_model_and_stack"] = _check(
            real and env is not None and not missing_pk,
            f"model id {(man.get('model') or {}).get('id')}, missing packages {missing_pk}",
        )

    blocking_ok = all(c["ok"] for c in checks.values())
    stats = _page_stats(traces, cfg.backend.max_new_tokens)
    report = {
        "schema": 1,
        "ok": blocking_ok,
        "schema_ok": checks["json_schema"]["ok"],
        "docs_found": f"{len(set(found_ids) & set(expected_ids))}/{len(expected_ids)}",
        "determinism_ok": det["ok"],
        "checks": checks,
        "page_stats": stats,
        "schema_repairs": repairs,
        "post_rules": rules_summary,
        "note": "counts, ids and key paths only; no extracted value is stored in this report",
    }
    manifest = _submission_manifest(
        cfg, man, comps, run_id, out_dir, schema_path, traces, n_pages, env, det, contract,
        smoke_status_path, sample_submission, repairs,
        {
            **rules_summary,
            "slot_shapes_file": postrules.SHAPES_FILE,
            "slot_shapes_sha256": shapes_sha,
        },
    )  # fmt: skip
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in (REJECTED_NAME, "test_predictions.json"):
        (out_dir / stale).unlink(missing_ok=True)
    _write_json(out_dir / (OUT_FILES[0] if blocking_ok else REJECTED_NAME), predictions)
    shutil.copyfile(run_dir / "trace.jsonl", out_dir / "trace.jsonl")
    _write_json(out_dir / "manifest.json", manifest)
    _write_json(out_dir / "validation_report.json", report)
    (out_dir / "rules.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rule_records), encoding="utf-8", newline="\n"
    )
    return report


def _ids_detail(r: Mapping[str, Any]) -> str:
    return (
        f"{r['n_found']} found / {r['n_expected']} expected; missing {r['n_missing']} "
        f"{r['missing']}, extra {r['n_extra']} {r['extra']}, duplicate {r['n_duplicate']} "
        f"{r['duplicate']}"
    )


def _determinism(comps: Sequence[Path]) -> dict[str, Any]:
    reports = [_read_json(c / "determinism.json") if (c / "determinism.json").is_file() else None
               for c in comps]  # fmt: skip
    if any(r is None for r in reports):
        return {**_check(False, "determinism.json missing in a component: run the pass"),
                "n_docs": 0}  # fmt: skip
    n_docs = sum(r["n_docs"] for r in reports)
    n_same = sum(r["n_identical"] for r in reports)
    replicated = all(r.get("composition_replicated") is True for r in reports)
    return {
        **_check(
            all(r["ok"] for r in reports) and n_docs > 0 and replicated,
            f"{n_same}/{n_docs} documents byte-identical on the second decode "
            f"({len(reports)} component(s)); first-pass batch composition replicated: {replicated}",
        ),
        "n_docs": n_docs,
        "n_identical": n_same,
        "doc_ids": sorted(d for r in reports for d in r["doc_ids"]),
        "rule": DET_RULE,
        "seed": DET_SEED,
    }


def _dev_contract(dev_run_dir: Path | None, run_dir: Path) -> dict[str, Any]:
    if dev_run_dir is None or dev_batch_size(Path(dev_run_dir)) is None:
        return {**_check(True, "not checked: no dev run manifest"), "checked": False}
    try:
        size = runmeta.require_same_batch_size(Path(dev_run_dir), run_dir)
    except ValueError as exc:
        return {**_check(False, str(exc)), "checked": True}
    return {**_check(True, f"dev and test both batch size {size}"), "checked": True}


def _page_stats(traces: Sequence[Mapping[str, Any]], max_new_tokens: int) -> dict[str, Any]:
    pages = [p for t in traces for p in t["pages"]]
    return {
        "pages": len(pages),
        "json_invalid": sum(not p["json_valid"] for p in pages),
        "page_schema_errors": sum(bool(p["schema_errors"]) for p in pages),
        "at_max_new_tokens": sum(
            (p["meta"].get("n_output_tokens") or 0) >= max_new_tokens for p in pages
        ),
        "with_field_logprobs": sum(isinstance(p.get("field_logprobs"), list) for p in pages),
    }


def _submission_manifest(
    cfg: SpikeConfig,
    man: Mapping[str, Any],
    comps: Sequence[Path],
    run_id: str,
    out_dir: Path,
    schema_path: Path,
    traces: Sequence[Mapping[str, Any]],
    n_pages: int,
    env: Mapping[str, Any] | None,
    det: Mapping[str, Any],
    contract: Mapping[str, Any],
    smoke_status_path: Path | None,
    sample_submission: Path | None,
    repairs: Mapping[str, Any] | None = None,
    post_rules: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    sessions = [s for c in comps for s in (
        _read_json(c / "sessions.json") if (c / "sessions.json").is_file() else [])]  # fmt: skip
    smoke_status = (
        _read_json(smoke_status_path) if smoke_status_path and Path(smoke_status_path).is_file()
        else {}
    )  # fmt: skip
    lat = [p["meta"].get("latency_s") or 0.0 for t in traces for p in t["pages"]]
    bench = man.get("bench") or None
    return {
        "schema": 1,
        "submission": f"{SUBMISSION_VERSION}_{str(man['code_sha'])[:7]}",
        "run_id": run_id,
        "code_sha": man["code_sha"],
        "model": man["model"],
        "config": {**man["config"], "file": f"configs/spike_{cfg.name}.yaml"},
        "seed": man["seed"],
        "prompt": {"version": PROMPT_VERSION, **(traces[0].get("prompt") or {})}
        if traces
        else None,
        "batch_size": man["batch_size"],
        "batch_size_source": man.get("batch_size_source"),
        "batch_size_contract": contract.get("detail"),
        "bench": bench,
        "shard": man["shard"],
        "merged_from": man.get("merged_from"),
        "docs": {"n_docs": len(traces), "n_pages": n_pages, "docs_sha": man.get("docs_sha")},
        "schema_file": {"name": Path(schema_path).name, "sha256": sha256_file(schema_path)},
        "sample_submission_checked": bool(sample_submission and Path(sample_submission).is_file()),
        "phase3_rules": phase3_rules(),
        "post_rules": dict(post_rules) if post_rules is not None else None,
        "schema_repairs": {
            k: (repairs or {}).get(k) for k in ("n", "in_trace", "at_assembly", "reason")
        },
        "dependencies": env,
        "determinism": {
            k: det.get(k) for k in ("ok", "n_docs", "n_identical", "doc_ids", "rule", "seed")
        },  # fmt: skip
        "timings": {
            "model_time_s": round(sum(lat), 1),
            "wall_clock_s": round(sum(s.get("seconds") or 0 for s in sessions), 1),
            "sessions": len(sessions),
            "unfinished_sessions": sum(s.get("seconds") is None for s in sessions),
            "smoke_s": smoke_status.get("seconds"),
        },
        "smoke": {k: smoke_status.get(k) for k in ("state", "run_id", "code_sha", "seconds")},
        "output_dir": out_dir.name,
        "files": list(OUT_FILES),
    }


# --------------------------------------------------------------------------------------------
# CLI glue
# --------------------------------------------------------------------------------------------


def _backend(args: Any, cfg: SpikeConfig, split: str) -> Any:
    """hf -> HfBackend; mock -> gold from ``--mock-gold DIR`` (labels of FAKE test docs) or dev."""
    if args.backend == "mock" and getattr(args, "mock_gold", None):
        from shipdoc import eval as ev
        from shipdoc.extract import MockBackend

        return MockBackend(
            ev.load_gold(args.mock_gold),
            ocr_token_budget=cfg.backend.ocr_token_budget,
            output_format=cfg.backend.output_format,
        )
    return make_backend(args.backend, cfg, split)


def _paths(args: Any) -> tuple[Path, Path]:
    runs = Path(args.runs_root) if args.runs_root else paths.runs_dir()
    data = Path(args.data_root) if args.data_root else paths.data_dir()
    return runs, data


def main_predict(args: Any) -> int:
    """CLI glue for ``python -m shipdoc predict <stage>`` (see cli.py). Exit 0 = stage passed."""
    try:
        return _dispatch(args)
    except PredictError as exc:
        print(f"predict {args.stage}: REFUSED: {exc}", file=sys.stderr)
        return 1


def _dispatch(args: Any) -> int:
    stage = args.stage
    runs, data = _paths(args)
    if stage == "plan":
        out = plan(data, args.shard)
        if args.out:
            _write_json(Path(args.out), out)
        print(f"plan: {out['n_docs']} docs / {out['n_pages']} pages; shard {args.shard}: "
              f"{out['shard_n_docs']} docs / {out['shard_n_pages']} pages; determinism docs "
              f"{out['determinism_docs']}")  # fmt: skip
        return 0
    if stage == "validate":
        rep = validate_file(args.pred, args.schema, list(discover_test_docs(data)))
        if args.report_out:
            _write_json(Path(args.report_out), rep)
        print(f"validate: ok={rep['ok']} schema_errors={rep['schema']['n_errors']} "
              f"{_ids_detail(rep['ids'])} duplicate_keys={rep['duplicate_keys']['n']}")  # fmt: skip
        return 0 if rep["ok"] else 1
    cfg = load_config(args.config)
    if stage == "smoke":
        status = run_smoke_gate(
            cfg, _backend(args, cfg, "dev"), load_doc_ids(args.docs), args.run_id, runs,
            args.status, data,
        )  # fmt: skip
        print(f"smoke gate: {status['state']}")
        return 0 if status["state"] == SMOKE_STATE_PASSED else 1
    if stage == "batch":
        from shipdoc import bench

        backend = _backend(args, cfg, "dev")
        dec = decide_batch_size(
            cfg, backend, args.bench_dir, load_doc_ids(args.bench_docs), args.decision,
            Path(args.dev_run_dir) if args.dev_run_dir else None, args.batch_size, data,
        )  # fmt: skip
        if dec["bench"]:  # the table and any deviation notice, as notebook 02 prints them
            print(bench.format_banner(_read_json(Path(dec["bench"]["path"]))))
        print(f"batch size {dec['batch_size']} ({dec['source']}; bench reused: "
              f"{dec['bench_reused']}; dev contract: {dec['contract']})")  # fmt: skip
        return 0
    if stage in ("run", "determinism"):
        backend = _backend(args, cfg, TEST_SPLIT)
        decision = _read_json(args.decision)
        if stage == "run":
            m = run_test(cfg, backend, args.run_id, runs, data, decision, args.smoke_status,
                         args.shard, args.expect_docs, args.expect_pages)  # fmt: skip
            print(f"test run {args.run_id} shard {args.shard}: {m['n_docs']} docs, "
                  f"{m['n_pages']} pages (unscored: the test split has no labels)")  # fmt: skip
            return 0
        rep = run_determinism(cfg, backend, args.run_id, runs, data, decision, args.shard)
        print(f"determinism ok: {rep['n_identical']}/{rep['n_docs']} documents byte-identical")
        return 0
    if stage == "assemble":
        if not args.out_dir and not args.sha7:
            raise PredictError("assemble needs --out-dir, or --sha7 for the default folder")
        sub = Path(args.out_dir or paths.submissions_dir() / f"{SUBMISSION_VERSION}_{args.sha7}")
        rep = assemble_submission(
            cfg, args.run_id, runs, data, sub, args.schema,
            Path(args.dev_run_dir) if args.dev_run_dir else None,
            args.expect_docs, args.expect_pages, args.expect_code_sha, args.allow_dirty,
            args.require_stack, args.smoke_status, args.sample_submission,
            RuleConfig(**{r.lower(): False for r in args.no_rule}), args.shapes_file,
            args.ocr_cache,
        )  # fmt: skip
        failed = [k for k, c in rep["checks"].items() if not c["ok"]]
        print(
            f"assemble: ok={rep['ok']} docs {rep['docs_found']} schema_ok={rep['schema_ok']} "
            f"determinism_ok={rep['determinism_ok']} failed checks {failed} -> {sub}\n"
            f"post-processing v1: touched docs {rep['post_rules']['touched_docs']}, "
            f"skipped {rep['post_rules']['skipped']}"
        )
        return 0 if rep["ok"] else 1
    raise PredictError(f"unknown stage {stage!r}")
