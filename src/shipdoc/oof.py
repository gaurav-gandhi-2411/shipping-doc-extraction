"""Out-of-fold (OOF) inference of a fold adapter on its held-out documents; the interim G4 check.

Notebook 05 (T4, fp16) wraps ``python -m shipdoc oof verify | infer | compare``. For fold k the
adapter was trained on every train+dev document OUTSIDE fold k (``splits/folds.json``); this module
runs it on the fold's held-out documents (invoices AND waybills) with the production inference
config (keyed output, prompt v2, greedy, xgrammar, logprobs on, the shared coerce / schema-repair
writer in `shipdoc.spike.run_spike`) and compares it, on exactly the same documents, with the 02
zero-shot run.

Stages, in this order (each fails closed; the order is asserted by tests)::

    VERIFY   `verify_adapter_manifest`: fold id, training code SHA, base revision, LoRA bookkeeping,
             the training doc ids (present, disjoint from the held-out ids, equal to everything
             outside the fold), inference-relevant keys, file hashes. Printed as a table.
    MERGE    `MergedHfBackend.load`: the fp16 base model + the peft adapter, ``merge_and_unload``.
    GUARD    the 12 bench pages at batch 1 and at the chosen batch size on the MERGED model: byte
             identical outputs or the run falls back to batch 1 (said in the manifest).
    INFER    `shipdoc.spike.run_spike` over the held-out ids, resumable per document.
    SCORE    official scorer vs the gold of those documents (train / dev labels only).
    COMPARE  paired comparison with the zero-shot run, over-null counts, the interim verdict.

Precision caveat (MERGE): the adapter was trained with bf16 (L4) or fp16 + fp32 LoRA weights on a
bf16 / fp16 base; here its low-rank update is added to the fp16 base weights (computed in fp32, then
rounded to fp16). Updates smaller than the fp16 rounding step of a weight are lost, so the merged
model is NOT bit-identical to base + unmerged adapter. This is the numeric path production
inference uses on a T4, which is why OOF is measured on it. UNVERIFIED on a GPU.

Precision of statements: every number a run prints is measured by that run; estimates are labelled.
Reports carry aggregates only: no document id, no extracted value.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from shipdoc import eval as ev
from shipdoc import paths
from shipdoc.bench import BENCH_PAGES, run_bench
from shipdoc.extract import BackendConfig, HfBackend
from shipdoc.runmeta import read_manifest, write_manifest
from shipdoc.spike import (
    SCORE_KEYS,
    SpikeConfig,
    atomic_write,
    load_gold_and_meta,
    run_spike,
)
from shipdoc.train import EXPECTED_LORA_MODULES, expected_lora_params

OOF_SCHEMA = 1
EXPECTED_LORA_R = 16
#: Gate G4 interim: the paired OVERALL delta's CI lower bound must stay above -1.0 point.
G4_DELTA_FLOOR = -0.01
#: Inference-relevant keys: what a training run and an inference run must share (see
#: `shipdoc.train.inference_keys`). No tuning knobs: only what changes the model's input / output.
INFERENCE_KEYS = (
    "model_repo",
    "model_revision",
    "adapter",
    "max_pixels",
    "prompt_version",
    "output_format",
)
INTERIM_LABEL = "interim, one fold, not the final G4 decision"
ROW_FIELDS = ("supplier_part_number", "customer_part_number", "purchase_order", "quantity")
#: ESTIMATE (not measured): seconds to read the peft folder, wrap the model and merge 200 modules.
MERGE_S_ESTIMATE = 60.0
VERIFIED_NAME = "oof_verification.json"
COMPARE_NAME = "oof_compare.json"
COMPARE_MD_NAME = "oof_compare.md"


class OofError(RuntimeError):
    """A precondition of the OOF run failed. Messages never quote a document value."""


# --------------------------------------------------------------------------------------------
# Folds and inference keys
# --------------------------------------------------------------------------------------------


def fold_heldout_ids(folds: Mapping[str, Any], fold: int) -> list[str]:
    """Sorted ``val_doc_ids`` of fold `fold` (the documents its adapter never trained on)."""
    for f in folds["folds"]:
        if f["fold"] == fold:
            return sorted(f["val_doc_ids"])
    raise OofError(f"fold {fold} is not in splits/folds.json")


def fold_train_ids(folds: Mapping[str, Any], fold: int) -> list[str]:
    """Sorted ids of every fold document outside fold `fold` (what its adapter must have used)."""
    held = set(fold_heldout_ids(folds, fold))
    return sorted(d for f in folds["folds"] for d in f["val_doc_ids"] if d not in held)


def inference_keys_of_config(cfg: SpikeConfig) -> dict[str, Any]:
    """The `INFERENCE_KEYS` values of a production inference config."""
    b = cfg.backend
    return {
        "model_repo": b.repo,
        "model_revision": b.revision,
        "adapter": b.adapter,
        "max_pixels": b.max_pixels,
        "prompt_version": cfg.prompt_version,
        "output_format": b.output_format,
    }


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_reachable(sha: str, repo: Path | None = None) -> bool:
    """True iff commit `sha` exists in the repo (a full clone has every pushed commit)."""
    res = subprocess.run(
        ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
        cwd=repo or paths.REPO_ROOT,
        capture_output=True,
        check=False,
    )
    return res.returncode == 0


# --------------------------------------------------------------------------------------------
# Batch size of the zero-shot run
# --------------------------------------------------------------------------------------------


def resolve_batch_size(zs_run_dir: Path, requested: int | None) -> dict[str, Any]:
    """The batch size of the OOF run.

    ``None`` = the size the 02 zero-shot run was decoded at, read from ``bench_result.json``
    (``chosen_batch_size``) in the run folder and from its ``manifest.json`` (``batch_size``); both
    must agree when both exist. Refused when neither exists. An explicit int overrides (said in
    the result: the pairing with the zero-shot run is then at a different batch size).
    """
    zs = Path(zs_run_dir)
    stored: dict[str, int] = {}
    bench = zs / "bench_result.json"
    if bench.is_file():
        v = json.loads(bench.read_text(encoding="utf-8")).get("chosen_batch_size")
        if isinstance(v, int) and not isinstance(v, bool) and v >= 1:
            stored["bench_result.json"] = v
    man = read_manifest(zs) if (zs / "manifest.json").is_file() else None
    if man is not None and isinstance(man.get("batch_size"), int) and man["batch_size"] >= 1:
        stored["manifest.json"] = int(man["batch_size"])
    if len(set(stored.values())) > 1:
        raise OofError(f"{zs.name}: stored batch sizes disagree {stored}; refusing to guess")
    if requested is not None:
        if isinstance(requested, bool) or requested < 1:
            raise OofError(f"BATCH_SIZE must be None or an int >= 1, got {requested!r}")
        diff = bool(stored) and requested not in stored.values()
        return {
            "batch_size": int(requested),
            "source": "manual",
            "stored": stored,
            "differs_from_zero_shot": diff,
        }
    if not stored:
        raise OofError(
            f"{zs.name}: no bench_result.json and no manifest batch_size: the zero-shot batch size "
            "is unknown. Copy bench_result.json into that folder or pass an explicit BATCH_SIZE."
        )
    return {
        "batch_size": next(iter(stored.values())),
        "source": "zero_shot_run",
        "stored": stored,
        "differs_from_zero_shot": False,
    }


# --------------------------------------------------------------------------------------------
# Adapter manifest verification
# --------------------------------------------------------------------------------------------


def _row(name: str, expected: Any, found: Any, ok: bool, note: str = "") -> dict[str, Any]:
    return {"check": name, "expected": expected, "found": found, "ok": bool(ok), "note": note}


def verify_adapter_manifest(
    adapter_dir: Path,
    *,
    fold: int,
    cfg: SpikeConfig,
    zs_manifest: Mapping[str, Any],
    folds: Mapping[str, Any],
    pin_sha: str | None = None,
    reachable: Callable[[str], bool] | None = None,
    inference_ids: Sequence[str] | None = None,
    expected_r: int = EXPECTED_LORA_R,
) -> dict[str, Any]:
    """Check an adapter folder (``final/``) against fold `fold`; never raises on a failed check.

    Returns ``{"ok", "rows", "warnings", "train_sha", "pin_sha", "n_train_docs", ...}``; each row is
    one check (expected, found, ok). `inference_ids` default to the fold's held-out ids. Use
    `assert_verified` to turn a failed report into an `OofError`. Refused outright (rows failing):
    stage ``final`` / ``smoke`` adapters, a fold id other than `fold`, an adapter whose manifest
    does not record its training doc ids (trained before they were recorded), any overlap between
    the training and inference ids, a training set that is not exactly the other folds' ids, a base
    revision or inference key that differs from the production config, an unreachable code SHA.
    """
    d = Path(adapter_dir)
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    mpath = d / "manifest.json"
    if not mpath.is_file():
        rows.append(_row("manifest.json", "present", "missing", False))
        return {"ok": False, "rows": rows, "warnings": warnings}
    m = json.loads(mpath.read_text(encoding="utf-8"))
    held = fold_heldout_ids(folds, fold)
    infer_ids = sorted(inference_ids) if inference_ids is not None else held
    want_stage = f"fold{fold}"
    stage = m.get("stage")
    if stage in ("final", "smoke"):
        rows.append(_row("stage", want_stage, stage, False, "OOF applies to fold adapters only"))
    else:
        rows.append(_row("stage", want_stage, stage, stage == want_stage))
    rows.append(_row("fold id", fold, m.get("fold"), m.get("fold") == fold))

    # code SHA of the TRAINING run
    train_sha = m.get("code_sha")
    sha_ok = isinstance(train_sha, str) and len(train_sha) == 40 and "+dirty" not in train_sha
    sha_ok = sha_ok and all(c in "0123456789abcdef" for c in str(train_sha))
    rows.append(_row("training code sha", "40-hex, clean", train_sha, sha_ok))
    if sha_ok and reachable is not None:
        is_reachable = bool(reachable(str(train_sha)))
        rows.append(_row("training code sha reachable", True, is_reachable, is_reachable))
    if sha_ok and pin_sha and train_sha != pin_sha:
        warnings.append(
            f"training code {str(train_sha)[:7]} differs from this notebook's pin {pin_sha[:7]}: "
            "check `git diff --stat <train>..<pin> -- src/shipdoc/{train,prompts,extract}.py`"
        )

    # base model + LoRA bookkeeping
    b = cfg.backend
    rows.append(
        _row(
            "base revision",
            b.revision,
            m.get("base_revision"),
            m.get("base_revision") == b.revision,
        )
    )
    rows.append(_row("base repo", b.repo, m.get("base_repo"), m.get("base_repo") == b.repo))
    lora = m.get("lora") if isinstance(m.get("lora"), dict) else {}
    r = lora.get("r")
    rows.append(_row("lora r", expected_r, r, r == expected_r))
    rows.append(
        _row(
            "lora modules",
            EXPECTED_LORA_MODULES,
            lora.get("n_modules"),
            lora.get("n_modules") == EXPECTED_LORA_MODULES,
        )
    )
    want_params = expected_lora_params(expected_r)
    rows.append(
        _row(
            "lora trainable params",
            want_params,
            lora.get("n_trainable"),
            lora.get("n_trainable") == want_params,
        )
    )

    # training doc ids
    ids = m.get("train_doc_ids")
    if not isinstance(ids, list) or not ids:
        rows.append(
            _row(
                "training doc ids",
                "recorded in the manifest",
                "missing",
                False,
                "adapter predates doc-id recording: retrain; refused for OOF",
            )
        )
        n_train = None
    else:
        n_train = len(ids)
        want = fold_train_ids(folds, fold)
        overlap = sorted(set(ids) & set(infer_ids))
        rows.append(
            _row(
                "train/inference ids disjoint",
                0,
                len(overlap),
                not overlap,
                f"{len(ids)} training ids vs {len(infer_ids)} inference ids",
            )
        )
        rows.append(
            _row(
                "training set = all docs outside the fold",
                len(want),
                len(ids),
                sorted(ids) == want and len(set(ids)) == len(ids),
            )
        )
        mh = m.get("manifest_hash")
        calc = hashlib.sha256((want_stage + "|" + ",".join(sorted(ids))).encode()).hexdigest()[:16]
        rows.append(_row("manifest_hash matches the ids", calc, mh, mh == calc))
    if isinstance(m.get("heldout_doc_ids"), list):
        rows.append(
            _row(
                "recorded held-out ids = fold",
                len(held),
                len(m["heldout_doc_ids"]),
                sorted(m["heldout_doc_ids"]) == held,
            )
        )

    # inference-relevant keys vs the production config and the zero-shot run
    want_keys, got_keys = inference_keys_of_config(cfg), m.get("inference_keys")
    got_keys = got_keys if isinstance(got_keys, dict) else {}
    for k in INFERENCE_KEYS:
        rows.append(
            _row(
                f"inference key {k}",
                want_keys[k],
                got_keys.get(k),
                k in got_keys and got_keys[k] == want_keys[k],
            )
        )
    zs_cfg = (zs_manifest.get("config") or {}).get("hash")
    rows.append(
        _row("zero-shot run config hash", cfg.config_hash, zs_cfg, zs_cfg == cfg.config_hash)
    )
    zs_rev = (zs_manifest.get("model") or {}).get("revision")
    rows.append(_row("zero-shot run model revision", b.revision, zs_rev, zs_rev == b.revision))
    zs_fmt = zs_manifest.get("output_format")
    rows.append(
        _row("zero-shot run output format", b.output_format, zs_fmt, zs_fmt == b.output_format)
    )

    # files, bound to the manifest by hash
    ap = d / "adapter.pt"
    rows.append(
        _row(
            "adapter.pt sha256",
            m.get("adapter_sha256"),
            sha256_file(ap) if ap.is_file() else "missing",
            ap.is_file() and sha256_file(ap) == m.get("adapter_sha256"),
        )
    )
    peft = d / "peft"
    recorded = m.get("peft_sha256") if isinstance(m.get("peft_sha256"), dict) else {}
    model_file = peft / "adapter_model.safetensors"
    found = sha256_file(model_file) if model_file.is_file() else "missing"
    rows.append(
        _row(
            "peft/adapter_model.safetensors sha256",
            recorded.get(model_file.name),
            found,
            model_file.is_file() and found == recorded.get(model_file.name),
        )
    )
    rows.append(
        _row(
            "peft/adapter_config.json",
            "present",
            "present" if (peft / "adapter_config.json").is_file() else "missing",
            (peft / "adapter_config.json").is_file(),
        )
    )
    rows.append(
        _row(
            "training precision",
            "recorded",
            m.get("precision"),
            m.get("precision") in ("bf16", "fp16", "fp32"),
        )
    )

    return {
        "ok": all(r_["ok"] for r_ in rows),
        "rows": rows,
        "warnings": warnings,
        "fold": fold,
        "train_sha": train_sha,
        "pin_sha": pin_sha,
        "n_train_docs": n_train,
        "n_inference_docs": len(infer_ids),
        "n_overlap": len(set(ids or []) & set(infer_ids)),
        "train_precision": m.get("precision"),
        "adapter_sha256": m.get("adapter_sha256"),
    }


def format_verification(report: Mapping[str, Any]) -> str:
    """The printed verification table (counts, hashes and SHAs only; no document ids)."""
    bar = "=" * 100
    lines = [
        bar,
        f"ADAPTER MANIFEST VERIFICATION, fold {report.get('fold')}: "
        f"{'PASSED' if report['ok'] else 'FAILED (refused)'}",
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
        lines.append(
            f"training docs {report['n_train_docs']}  inference docs "
            f"{report['n_inference_docs']}  overlap {report['n_overlap']}"
        )
    lines += [f"WARNING: {w_}" for w_ in report.get("warnings", [])]
    lines.append(bar)
    return "\n".join(lines)


def assert_verified(report: Mapping[str, Any]) -> None:
    """Raise `OofError` naming every failed check when the verification did not pass."""
    if not report["ok"]:
        failed = [r["check"] for r in report["rows"] if not r["ok"]]
        raise OofError(f"adapter verification FAILED, refused: {failed}")


# --------------------------------------------------------------------------------------------
# The merged backend (the only part that needs peft + a GPU; UNVERIFIED on a GPU)
# --------------------------------------------------------------------------------------------


class MergedHfBackend(HfBackend):
    """`HfBackend` whose fp16 base model has the fold adapter merged into its weights.

    Everything else (processor, visual-token cap, xgrammar, stop ids, batching) is the production
    path, so the OOF run differs from zero-shot only in the weights. ``load`` loads the base model
    exactly as `HfBackend.load` does, wraps it with ``PeftModel.from_pretrained(<final>/peft)``,
    checks that `EXPECTED_LORA_MODULES` modules carry an adapter, and ``merge_and_unload``s them.
    ``merge_info`` records what was merged (counts only).
    """

    def __init__(self, cfg: BackendConfig, adapter_dir: Path, adapter_sha256: str = "") -> None:
        super().__init__(cfg)
        self.adapter_dir = Path(adapter_dir)
        self.merge_info: dict[str, Any] | None = None
        self.model_id = f"{cfg.repo}+lora:{adapter_sha256[:12]}" if adapter_sha256 else cfg.repo

    def load(self) -> None:
        """Load the base model, then merge the adapter into it; idempotent."""
        if self._loaded:
            return
        super().load()
        from peft import PeftModel

        t0 = time.perf_counter()
        wrapped = PeftModel.from_pretrained(self.model, str(self.adapter_dir / "peft"))
        n_before = sum(
            1 for m in wrapped.modules() if hasattr(m, "lora_A") and hasattr(m, "lora_B")
        )
        if n_before != EXPECTED_LORA_MODULES:
            raise OofError(
                f"adapter attached to {n_before} modules, expected {EXPECTED_LORA_MODULES}"
            )
        merged = wrapped.merge_and_unload()
        n_after = sum(1 for m in merged.modules() if hasattr(m, "lora_A") or hasattr(m, "lora_B"))
        if n_after:
            raise OofError(f"{n_after} LoRA modules left after merge_and_unload")
        self.model = merged.eval()
        self.merge_info = {
            "n_lora_modules_merged": n_before,
            "n_lora_modules_left": n_after,
            "merge_dtype": str(next(self.model.parameters()).dtype),
            "merge_s": round(time.perf_counter() - t0, 1),
        }


# --------------------------------------------------------------------------------------------
# Over-null counts, paired comparison, interim verdict
# --------------------------------------------------------------------------------------------


def over_null_counts(
    pred: Mapping[str, Any], gold: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Cells where the gold has a value and the prediction is empty (over-nulls), and false fills.

    Header cells are compared field by field. Row cells are compared on the rows the scorer pairs
    (`shipdoc.eval._pair_rows`: full and partial matches); a gold row nobody matched is counted
    in ``rows_unmatched_gold`` (a missing row), not as cell over-nulls. Empty = the scorer's
    notion (None or blank string). False fill = gold empty, prediction non-empty.
    """
    sc = ev._sc()
    h_over: Counter[str] = Counter()
    r_over: Counter[str] = Counter()
    h_ff = r_ff = unmatched = 0
    for doc_id, g in gold.items():
        p = pred.get(doc_id)
        pd_ = p if isinstance(p, dict) else {}
        ph = pd_.get("header") if isinstance(pd_.get("header"), dict) else {}
        for f in sc.HEADER[g["doc_type"]]:
            gv, pv = g["header"].get(f), ph.get(f)
            if sc._empty(pv) and not sc._empty(gv):
                h_over[f] += 1
            elif sc._empty(gv) and not sc._empty(pv):
                h_ff += 1
        pr = [x for x in (pd_.get("line_items") or []) if isinstance(x, dict)]
        gr = g["line_items"]
        full, partial, _un_p, un_g = ev._pair_rows(sc, pr, gr)
        unmatched += len(un_g)
        for pi, gi in full + partial:
            for f in sc.ROW:
                gv, pv = gr[gi].get(f), pr[pi].get(f)
                if sc._empty(pv) and not sc._empty(gv):
                    r_over[f] += 1
                elif sc._empty(gv) and not sc._empty(pv):
                    r_ff += 1
    return {
        "header_over_null_by_field": dict(sorted(h_over.items())),
        "header_over_null": sum(h_over.values()),
        "row_over_null_by_field": {f: r_over.get(f, 0) for f in sc.ROW},
        "row_over_null": sum(r_over.values()),
        "over_null_total": sum(h_over.values()) + sum(r_over.values()),
        "header_false_fill": h_ff,
        "row_false_fill": r_ff,
        "false_fill_total": h_ff + r_ff,
        "rows_unmatched_gold": unmatched,
    }


def g4_interim_verdict(
    delta_ci_lo: float, oof_over_null: int, zs_over_null: int, delta_ci_hi: float | None = None
) -> dict[str, Any]:
    """The interim G4 verdict for one fold (see `INTERIM_LABEL`).

    NO REGRESSION iff the paired OVERALL delta's CI lower bound is above `G4_DELTA_FLOOR`
    (-1.0 point) AND the OOF model's over-null cell count (header + row) is not higher than
    zero-shot's. Otherwise REGRESSION, naming the clause(s) that failed. Whether the CI excludes 0
    on the positive side is reported but is not part of the verdict.
    """
    failed = []
    if not delta_ci_lo > G4_DELTA_FLOOR:
        failed.append(
            f"paired OVERALL delta CI lower bound {100 * delta_ci_lo:+.2f} points is not "
            f"above {100 * G4_DELTA_FLOOR:+.1f}"
        )
    if oof_over_null > zs_over_null:
        failed.append(f"over-null cells {oof_over_null} (OOF) > {zs_over_null} (zero-shot)")
    return {
        "verdict": "NO REGRESSION" if not failed else "REGRESSION",
        "failed_clauses": failed,
        "delta_ci_lo": delta_ci_lo,
        "delta_ci_hi": delta_ci_hi,
        "oof_over_null": oof_over_null,
        "zs_over_null": zs_over_null,
        "delta_excludes_zero_positive": bool(delta_ci_lo > 0),
        "label": INTERIM_LABEL,
    }


def verdict_line(v: Mapping[str, Any]) -> str:
    """The one explicit printed line of the interim verdict."""
    why = "" if v["verdict"] == "NO REGRESSION" else " (" + "; ".join(v["failed_clauses"]) + ")"
    return f"G4 INTERIM VERDICT ({v['label']}): {v['verdict']}{why}"


def _model_block(
    pred: Mapping[str, Any], gold: dict[str, dict[str, Any]], n_boot: int, seed: int
) -> dict[str, Any]:
    per = ev.per_doc_results(dict(pred), gold)
    agg = ev._sc().aggregate(list(per.values()))
    ci = ev.bootstrap_metrics(list(per.values()), n=n_boot, seed=seed)
    return {
        "documents": agg["documents"],
        **{k: agg[k] for k in SCORE_KEYS if k in agg},
        "ci95": ci,
        "over_null": over_null_counts(pred, gold),
    }


def compare_models(
    zs_pred: Mapping[str, Any],
    oof_pred: Mapping[str, Any],
    gold: Mapping[str, Mapping[str, Any]],
    meta: list[dict[str, Any]] | None = None,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Zero-shot vs OOF on exactly the documents of `gold` (aggregates only).

    Subsets: all, invoices, waybills, scanned, digital. Per subset and model: the five headline
    metrics with a 95% bootstrap CI (doc level, `n_boot` resamples, `seed`), plus the PAIRED
    bootstrap delta (OOF - zero-shot, same resamples) with its CI, plus over-null / false-fill
    counts. The verdict uses the ``all`` subset. Raises `OofError` when either prediction set
    misses a gold document (nothing is scored as an empty prediction silently).
    """
    gold = dict(gold)
    for name, pred in (("zero-shot", zs_pred), ("OOF", oof_pred)):
        missing = [d for d in gold if d not in pred]
        if missing:
            raise OofError(
                f"{name} predictions miss {len(missing)} of {len(gold)} held-out documents"
            )
    names = {
        "all": "all",
        "invoices": "invoices",
        "waybills": "waybills",
        "scanned=yes": "scanned",
        "scanned=no": "digital",
    }
    slices = ev.slice_doc_ids(gold, meta or None, group_slices=False)
    out: dict[str, Any] = {"n_boot": n_boot, "seed": seed, "subsets": {}}
    for key, label in names.items():
        ids = slices.get(key) or []
        if not ids:
            continue
        sub = {d: gold[d] for d in ids}
        out["subsets"][label] = {
            "n_docs": len(ids),
            "zero_shot": _model_block(zs_pred, sub, n_boot, seed),
            "oof": _model_block(oof_pred, sub, n_boot, seed),
            "paired_delta_oof_minus_zero_shot": ev.paired_bootstrap(
                dict(zs_pred), dict(oof_pred), sub, n=n_boot, seed=seed
            ),
        }
    allb = out["subsets"]["all"]
    d = allb["paired_delta_oof_minus_zero_shot"]["OVERALL"]
    out["verdict"] = g4_interim_verdict(
        d["lo"],
        allb["oof"]["over_null"]["over_null_total"],
        allb["zero_shot"]["over_null"]["over_null_total"],
        d["hi"],
    )
    return out


def _pct(x: float) -> str:
    return f"{100 * x:.2f}"


def _ci(ci: Mapping[str, float]) -> str:
    return f"{_pct(ci['point'])} [{_pct(ci['lo'])}, {_pct(ci['hi'])}]"


def format_compare(cmp: Mapping[str, Any], markdown: bool = False) -> str:
    """Printable (or markdown) comparison: aggregates only, no document id or value."""
    lines: list[str] = []
    title = f"OOF fine-tuned vs zero-shot, same held-out documents ({INTERIM_LABEL})"
    lines += [f"# {title}", ""] if markdown else ["=" * 110, title, "=" * 110]
    lines.append(
        f"percent; 95% CI = doc-level bootstrap, {cmp['n_boot']} resamples, seed "
        f"{cmp['seed']}; delta = OOF - zero-shot (paired, same resamples)"
    )
    metrics = (
        "OVERALL",
        "header_field_accuracy",
        "row_f1",
        "documents_fully_correct",
        "false_fill_rate",
    )
    for label, sub in cmp["subsets"].items():
        zs, oo, pd_ = sub["zero_shot"], sub["oof"], sub["paired_delta_oof_minus_zero_shot"]
        lines.append("")
        lines.append(("## " if markdown else "--- ") + f"{label} ({sub['n_docs']} documents)")
        if markdown:
            lines += [
                "",
                "| metric | zero-shot | OOF fine-tuned | paired delta |",
                "|---|---|---|---|",
            ]
        for k in metrics:
            row = (
                f"{_ci(zs['ci95'][k])}",
                f"{_ci(oo['ci95'][k])}",
                f"{100 * pd_[k]['delta']:+.2f} [{_pct(pd_[k]['lo'])}, {_pct(pd_[k]['hi'])}]",
            )
            lines.append(
                f"| {k} | {row[0]} | {row[1]} | {row[2]} |"
                if markdown
                else f"  {k:<26} zero-shot {row[0]:<26} OOF {row[1]:<26} delta {row[2]}"
            )
        zn, on = zs["over_null"], oo["over_null"]
        for text, key in (
            ("header over-nulls", "header_over_null"),
            ("row over-nulls", "row_over_null"),
            ("over-nulls total", "over_null_total"),
            ("header false fills", "header_false_fill"),
            ("row false fills", "row_false_fill"),
        ):
            lines.append(
                f"| {text} | {zn[key]} | {on[key]} | {on[key] - zn[key]:+d} |"
                if markdown
                else f"  {text:<26} zero-shot {zn[key]:<8} OOF {on[key]:<8} "
                f"diff {on[key] - zn[key]:+d}"
            )
        if label == "all":
            for per, key in (
                ("header", "header_over_null_by_field"),
                ("row", "row_over_null_by_field"),
            ):
                fields = sorted(set(zn[key]) | set(on[key]))
                if fields:
                    cells = ", ".join(
                        f"{f}: {zn[key].get(f, 0)} -> {on[key].get(f, 0)}" for f in fields
                    )
                    lines.append(f"  {per} over-nulls by field (zero-shot -> OOF): {cells}")
    lines.append("")
    v = cmp["verdict"]
    d = cmp["subsets"]["all"]["paired_delta_oof_minus_zero_shot"]["OVERALL"]
    lines.append(
        f"OVERALL paired delta CI [{_pct(d['lo'])}, {_pct(d['hi'])}] excludes 0 on the "
        f"positive side: {v['delta_excludes_zero_positive']} (reported, not part of the "
        "verdict)"
    )
    lines.append(
        f"over-null cells (header+row, matched rows): zero-shot {v['zs_over_null']}, "
        f"OOF {v['oof_over_null']}"
    )
    lines.append(verdict_line(v))
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------------
# Estimate (ESTIMATE labels: nothing here is measured)
# --------------------------------------------------------------------------------------------


def estimate_rows(
    ge: Any,
    speed: Mapping[str, Any],
    config: str,
    n_pages: int,
    batch: int,
    merge_s: float = MERGE_S_ESTIMATE,
) -> list[dict[str, Any]]:
    """ESTIMATE rows for one OOF run on a T4: one per scaling efficiency (1 row at batch 1).

    `ge` is the ``scripts/gpu_estimate.py`` module (``batch_s_per_page`` and ``STRAGGLER``).
    Hours = model load + merge + guard (the bench pages at batch 1 and at `batch`, after one
    warm-up page) + `n_pages` at `batch`; the guard is skipped at batch 1.
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
        infer_s = n_pages * sb
        total = load_s + merge_s + guard_s + infer_s
        rows.append(
            {
                "efficiency": eff,
                "s_per_page": sb,
                "load_s": load_s,
                "merge_s": merge_s,
                "guard_s": guard_s,
                "infer_s": infer_s,
                "hours": total / 3600,
                "cu_central": None
                if speed.get("t4_cu_per_hour") is None
                else total / 3600 * speed["t4_cu_per_hour"],
                "cu_conservative": None
                if speed.get("t4_cu_per_hour_conservative") is None
                else total / 3600 * speed["t4_cu_per_hour_conservative"],
            }
        )
    return rows


def format_estimate(rows: Sequence[Mapping[str, Any]], n_pages: int, batch: int, fold: int) -> str:
    """The printed estimate table (every figure labelled ESTIMATE)."""
    bar = "=" * 110
    out = [
        bar,
        f"ESTIMATE (UNVERIFIED) T4 hours / CU: OOF fold {fold}, {n_pages} held-out pages at "
        f"batch {batch}. Batching gain and merge time ASSUMED, not measured",
        bar,
        f"{'scaling eff':>11} {'s/page':>7} {'load s':>7} {'merge s':>8} {'guard s':>8} "
        f"{'infer h':>8} {'T4 h':>6} {'CU@low':>7} {'CU@high':>8}",
    ]
    for r in rows:
        cu = lambda v: "n/a" if v is None else f"{v:.1f}"  # noqa: E731
        out.append(
            f"{int(r['efficiency'] * 100):>10}% {r['s_per_page']:7.1f} {r['load_s']:7.0f} "
            f"{r['merge_s']:8.0f} {r['guard_s']:8.0f} {r['infer_s'] / 3600:8.2f} "
            f"{r['hours']:6.2f} {cu(r['cu_central']):>7} {cu(r['cu_conservative']):>8}"
        )
    out += [
        "load/merge/guard are included; not modelled: logprob overhead, Drive I/O, restarts, "
        "OOM fallbacks.",
        bar,
    ]
    return "\n".join(out)


# --------------------------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_zero_shot(zs_run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """(manifest, predictions) of a COMPLETE zero-shot run; refuses an unfinished one."""
    zs = Path(zs_run_dir)
    man = read_manifest(zs)
    if man is None:
        raise OofError(f"{zs}: no manifest.json (not a zero-shot run folder?)")
    prog = zs / "progress.json"
    status = _read_json(prog).get("status") if prog.is_file() else None
    if status != "complete":
        raise OofError(f"{zs.name}: zero-shot run is not complete (status {status!r})")
    if not (zs / "predictions.json").is_file():
        raise OofError(f"{zs}: no predictions.json")
    return man, _read_json(zs / "predictions.json")


def verify_stage(
    *,
    fold: int,
    adapter_dir: Path,
    zs_run_dir: Path,
    cfg: SpikeConfig,
    folds: Mapping[str, Any],
    pin_sha: str | None = None,
    reachable: Callable[[str], bool] | None = None,
    out: Callable[[str], None] = print,
) -> dict[str, Any]:
    """VERIFY: print the table and raise `OofError` when anything fails."""
    zs_man, _ = load_zero_shot(zs_run_dir)
    report = verify_adapter_manifest(
        adapter_dir,
        fold=fold,
        cfg=cfg,
        zs_manifest=zs_man,
        folds=folds,
        pin_sha=pin_sha,
        reachable=reachable,
    )
    out("== VERIFY ==")
    if report.get("rows"):
        out(format_verification(report))
    assert_verified(report)
    return report


def run_infer(
    *,
    fold: int,
    adapter_dir: Path,
    zs_run_dir: Path,
    cfg: SpikeConfig,
    folds: Mapping[str, Any],
    out_dir: Path,
    backend_factory: Callable[[Path, str], Any],
    bench_docs: Sequence[str],
    data_root: Path,
    batch_size: int | None = None,
    split: str = "train+dev",
    pin_sha: str | None = None,
    reachable: Callable[[str], bool] | None = None,
    use_wandb: bool = False,
    out: Callable[[str], None] = print,
) -> dict[str, Any]:
    """VERIFY -> MERGE -> GUARD -> INFER. Resumable; returns the OOF manifest section.

    `backend_factory(adapter_dir, adapter_sha256)` builds the backend (`MergedHfBackend` for real;
    a mock in tests). A backend with a ``load()`` is loaded here (that is the MERGE stage) and must
    expose ``merge_info`` with ``n_lora_modules_merged == EXPECTED_LORA_MODULES``. The guard runs
    `bench_docs` at batch 1 and at the chosen size on the merged model: any difference in the
    page outputs, or a failed size, makes the run use batch 1 (recorded in the manifest).
    """
    out_dir = Path(out_dir)
    report = verify_stage(
        fold=fold,
        adapter_dir=adapter_dir,
        zs_run_dir=zs_run_dir,
        cfg=cfg,
        folds=folds,
        pin_sha=pin_sha,
        reachable=reachable,
        out=out,
    )
    held = fold_heldout_ids(folds, fold)
    chosen = resolve_batch_size(zs_run_dir, batch_size)
    if chosen["differs_from_zero_shot"]:
        out(
            f"WARNING: BATCH_SIZE {chosen['batch_size']} differs from the zero-shot run's "
            f"{chosen['stored']}: greedy outputs are only strictly comparable at the same size."
        )
    backend = backend_factory(Path(adapter_dir), str(report.get("adapter_sha256") or ""))

    out("== MERGE ==")
    t0 = time.perf_counter()
    load = getattr(backend, "load", None)
    if callable(load):
        load()
    load_s = round(time.perf_counter() - t0, 1)
    info = getattr(backend, "merge_info", None)
    if not isinstance(info, dict) or info.get("n_lora_modules_merged") != EXPECTED_LORA_MODULES:
        raise OofError(f"merge did not merge {EXPECTED_LORA_MODULES} LoRA modules: {info}")
    out(
        f"merged {info['n_lora_modules_merged']} LoRA modules into the base weights "
        f"({info.get('merge_dtype')}), model load + merge {load_s} s"
    )

    batch = int(chosen["batch_size"])
    guard: dict[str, Any] = {"ran": False, "batch_size": batch, "fallback_to_1": False}
    if batch > 1:
        out(
            f"== GUARD == {len(bench_docs)} bench documents at batch 1 and batch {batch} on the "
            "MERGED model (byte-identical outputs required)"
        )
        res = run_bench(
            cfg, backend, list(bench_docs), out_dir / "guard", (1, batch), data_root, logprobs=True
        )
        entry = next(r for r in res["results"] if r["batch_size"] == batch)
        same = bool(entry.get("ok")) and entry.get("byte_identical_rate") == 1.0
        guard.update(
            ran=True,
            ok=same,
            byte_identical_rate=entry.get("byte_identical_rate"),
            error=entry.get("error"),
            n_pages=res["n_pages"],
        )
        if not same:
            guard["fallback_to_1"] = True
            batch = 1
            out(
                f"GUARD FAILED: batch {chosen['batch_size']} is not byte-identical to batch 1 on "
                "the merged model: FALLING BACK TO BATCH 1 (recorded in the manifest)."
            )
        else:
            out(f"GUARD PASSED: batch {batch} byte-identical to batch 1 on {res['n_pages']} pages.")
    else:
        out("== GUARD == skipped (batch 1)")

    out(f"== INFER == {len(held)} held-out documents at batch {batch}")
    resume = (out_dir / "trace.jsonl").is_file()
    bench_info = {
        "path": str(Path(zs_run_dir) / "bench_result.json"),
        "chosen": batch,
        "deviation": guard if guard["fallback_to_1"] else None,
    }
    run_spike(
        cfg,
        held,
        split,
        out_dir.name,
        backend,
        resume=resume,
        use_wandb=use_wandb,
        runs_root=out_dir.parent,
        data_root=data_root,
        logprobs=True,
        batch_size=batch,
        bench_info=bench_info,
    )
    man = read_manifest(out_dir) or {}
    section = {
        "schema": OOF_SCHEMA,
        "fold": fold,
        "adapter_dir": str(adapter_dir),
        "adapter_sha256": report.get("adapter_sha256"),
        "train_code_sha": report.get("train_sha"),
        "pin_sha": pin_sha,
        "train_precision": report.get("train_precision"),
        "n_train_docs": report.get("n_train_docs"),
        "n_inference_docs": len(held),
        "n_train_inference_overlap": report.get("n_overlap"),
        "zero_shot_run": Path(zs_run_dir).name,
        "batch": {**chosen, "used": batch},
        "guard": guard,
        "merge": {**info, "load_and_merge_s": load_s},
        "merge_precision_note": "LoRA update added to fp16 base weights (fp32 sum, then fp16)",
        "verification": {"ok": report["ok"], "warnings": report["warnings"]},
    }
    write_manifest(out_dir, {**man, "oof": section})
    atomic_write(out_dir / VERIFIED_NAME, json.dumps(report, indent=1))
    return section


def run_compare(
    *,
    fold: int,
    zs_run_dir: Path,
    out_dir: Path,
    folds: Mapping[str, Any],
    data_root: Path,
    split: str = "train+dev",
    n_boot: int = 2000,
    out: Callable[[str], None] = print,
) -> dict[str, Any]:
    """SCORE + COMPARE: official scorer vs the gold of the fold's documents; paired comparison.

    Writes ``oof_compare.json`` and ``oof_compare.md`` (aggregates only) next to the OOF run's
    files and prints the table and the interim verdict line.
    """
    out_dir = Path(out_dir)
    held = fold_heldout_ids(folds, fold)
    _zs_man, zs_pred = load_zero_shot(zs_run_dir)
    prog = out_dir / "progress.json"
    status = _read_json(prog).get("status") if prog.is_file() else None
    if status != "complete":
        raise OofError(f"{out_dir.name}: the OOF run is not complete (status {status!r})")
    oof_pred = _read_json(out_dir / "predictions.json")
    gold_all, meta = load_gold_and_meta(split.split("+"), Path(data_root))
    if gold_all is None:
        raise OofError(f"no gold labels for {split} under {data_root}")
    missing = [d for d in held if d not in gold_all]
    if missing:
        raise OofError(f"{len(missing)} held-out documents have no gold label")
    gold = {d: gold_all[d] for d in held}
    extra = sorted(set(oof_pred) - set(held))
    if extra:
        raise OofError(f"the OOF predictions hold {len(extra)} documents outside the fold")
    out("== SCORE / COMPARE ==")
    cmp = compare_models(zs_pred, oof_pred, gold, meta, n_boot=n_boot)
    cmp["fold"], cmp["zero_shot_run"], cmp["n_docs"] = fold, Path(zs_run_dir).name, len(held)
    atomic_write(out_dir / COMPARE_NAME, json.dumps(cmp, indent=1))
    atomic_write(out_dir / COMPARE_MD_NAME, format_compare(cmp, markdown=True))
    out(format_compare(cmp))
    return cmp


def run_estimate(
    *,
    fold: int,
    zs_run_dir: Path,
    cfg: SpikeConfig,
    folds: Mapping[str, Any],
    data_root: Path,
    batch_size: int | None = None,
    out: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Print the ESTIMATE table for the fold (hours and CU on a T4); reads labels, runs no model.

    Pages come from the ``pages`` of the gold label files of the fold's held-out documents; the
    batch size is the one the OOF run will use (`resolve_batch_size`).
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "gpu_estimate", paths.REPO_ROOT / "scripts" / "gpu_estimate.py"
    )
    if spec is None or spec.loader is None:
        raise OofError("scripts/gpu_estimate.py not found")
    ge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ge)
    speed = json.loads((paths.REPO_ROOT / "configs" / "spike_speed.json").read_text("utf-8"))
    held = fold_heldout_ids(folds, fold)
    n_pages = 0
    for prefix in sorted({d.split("_")[0] for d in held}):
        ids = [d for d in held if d.startswith(prefix + "_")]
        n_pages += ge.count_pages(ids, Path(data_root) / prefix / "labels")
    chosen = resolve_batch_size(zs_run_dir, batch_size)
    rows = estimate_rows(ge, speed, cfg.name, n_pages, chosen["batch_size"])
    out(format_estimate(rows, n_pages, chosen["batch_size"], fold))
    out(
        f"documents {len(held)}, pages {n_pages} (counted from the label files), batch size "
        f"{chosen['batch_size']} ({chosen['source']})."
    )
    return {"n_docs": len(held), "n_pages": n_pages, "batch": chosen, "rows": rows}


# --------------------------------------------------------------------------------------------
# CLI glue (python -m shipdoc oof ...)
# --------------------------------------------------------------------------------------------


def main_oof(args: Any) -> int:
    """``python -m shipdoc oof verify | infer | compare`` (see cli.py)."""
    from shipdoc.spike import load_config

    cfg = load_config(args.config)
    folds = json.loads(Path(args.folds).read_text(encoding="utf-8"))
    data_root = Path(args.data_root) if args.data_root else paths.data_dir()
    out_dir = Path(args.out_dir) if getattr(args, "out_dir", None) else None
    common = {"fold": args.fold, "zs_run_dir": Path(args.zs_run_dir), "folds": folds}
    try:
        if args.stage == "estimate":
            run_estimate(
                fold=args.fold,
                zs_run_dir=Path(args.zs_run_dir),
                cfg=cfg,
                folds=folds,
                data_root=data_root,
                batch_size=args.batch_size,
            )
            return 0
        if args.stage == "compare":
            run_compare(out_dir=out_dir, data_root=data_root, split=args.split, **common)
            return 0
        if args.stage == "verify":
            verify_stage(
                adapter_dir=Path(args.adapter_dir),
                cfg=cfg,
                pin_sha=args.pin,
                reachable=git_reachable,
                **common,
            )
            return 0

        def factory(adapter_dir: Path, sha: str) -> Any:
            return MergedHfBackend(cfg.backend, adapter_dir, sha)

        bench_docs = json.loads(Path(args.bench_docs).read_text(encoding="utf-8"))
        run_infer(
            adapter_dir=Path(args.adapter_dir),
            cfg=cfg,
            out_dir=out_dir,
            backend_factory=factory,
            bench_docs=bench_docs,
            data_root=data_root,
            batch_size=args.batch_size,
            split=args.split,
            pin_sha=args.pin,
            reachable=git_reachable,
            use_wandb=args.wandb,
            **common,
        )
    except OofError as exc:
        print(f"OOF REFUSED: {exc}")
        return 1
    return 0
