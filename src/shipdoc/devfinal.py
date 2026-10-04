"""Notebook 05b: the FINAL adapter on the 100 DEV documents (the official seen-layout evaluation).

Spec Phase 4 item 6 and section 8: the final model is trained on the 400 ``train_*`` documents; the
100 ``dev_*`` documents are held out of its training and are the official seen-layout evaluation.
Notebook 05 refuses ``final`` adapters (it is the OOF check of FOLD adapters) and 04c covers only
the test documents, so this module runs the same production pipeline on the dev documents and
scores it with the official scorer against the 02 zero-shot run restricted to the same documents.

Stages of ``python -m shipdoc.devfinal``:

``plan``       CPU. The 100 ids of ``splits/dev100.json`` must equal the dev ids of
               ``splits/zeroshot500.json``, the held-out ids of the ``final`` stage of
               ``splits/folds.json`` and the dev label files; counts documents and pages.
``estimate``   ESTIMATE of T4 hours / compute units (labelled; nothing in it is measured).
``verify``     `predict_ft.verify_final_adapter`: the printed refusal table (fails closed).
``infer``      VERIFY -> MERGE -> GUARD -> INFER (`spike.run_spike` on the 100 dev ids, logprobs
               on),
               resumable per document; the batch decision of a half-finished run stands.
``compare``    CPU. Dev documents only. FT vs ZS under three arms: R1-R3 with R3 shapes learned from
               the TRAIN gold only (primary: no dev gold anywhere), R1-R3 with the frozen shipping
               shapes (learned from all 500 train+dev docs: IN-SAMPLE on dev, optimistic), and raw
               (no rules, secondary). Official scorer, paired doc-level bootstrap (2000, seed 42).
               ``--plumbing-check`` uses the 02 run's dev docs as BOTH arms (NOT A RESULT).

Reports carry aggregates only: no document id, no extracted value. Reused unchanged by import:
`predict_ft` (verify / merge / guard / decision), `oof` (batch size, over-null counts, the paired
comparison), `postrules` and ``scripts/replay_v1_check.production`` (the production rule path).

What is UNVERIFIED on a GPU: the merge, the batch-8 byte identity on the merged model and every
timing; only the CPU stages and the mock-backend pipeline are tested locally.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from shipdoc import eval as ev
from shipdoc import oof, paths, postrules, predict_ft, rules, spike
from shipdoc import trainset as ts
from shipdoc.postrules import RuleConfig
from shipdoc.replay import read_trace
from shipdoc.runmeta import read_manifest, write_manifest

DEVFINAL_SCHEMA = 1
EXPECTED_DEV_DOCS = 100
SPLIT = "dev"
PLAN_NAME = "devfinal_plan.json"
DECISION_NAME = "batch_decision.json"
COMPARE_NAME = "devfinal_compare.json"
COMPARE_MD_NAME = "devfinal_compare.md"
SEED = 42
N_BOOT = 2000
#: The five headline metrics (`shipdoc.eval.METRICS`), in print order.
METRICS = (
    "OVERALL",
    "header_field_accuracy",
    "row_f1",
    "documents_fully_correct",
    "false_fill_rate",
)
ARM_TRAIN = "rules_train_shapes"
ARM_FROZEN = "rules_frozen_shapes"
ARM_RAW = "raw"
ARMS = {
    ARM_TRAIN: "R1-R3, R3 shapes learned from the TRAIN gold only (primary)",
    ARM_FROZEN: "R1-R3, frozen shipping shapes (learned from all 500 train+dev docs: IN-SAMPLE)",
    ARM_RAW: "raw model output, no rules (secondary)",
}
RESULT_LABEL = "dev, 100 documents, seen layouts (the final adapter never trained on a dev doc)"
PLUMBING_LABEL = "PLUMBING CHECK, NOT A RESULT: the 02 run's dev documents are BOTH arms"
SUBSET_NAMES = ("all", "invoices", "waybills", "scanned", "digital")


class DevFinalError(RuntimeError):
    """A precondition of the dev run failed. Messages never quote a document value."""


# --------------------------------------------------------------------------------------------
# The dev ids
# --------------------------------------------------------------------------------------------


def ids_sha256(ids: Sequence[str]) -> str:
    """sha256 of the sorted, comma-joined ids (what the manifests record instead of the ids)."""
    return hashlib.sha256(",".join(sorted(ids)).encode()).hexdigest()


def check_dev_ids(
    dev_ids: Sequence[str],
    zs500_ids: Sequence[str],
    folds: Mapping[str, Any],
    label_ids: Sequence[str] | None = None,
    expected: int | None = None,
) -> list[str]:
    """The sorted dev ids, after asserting they are THE dev documents; raises `DevFinalError`.

    Checked: `expected` (default `EXPECTED_DEV_DOCS`) unique ids, all ``dev_*``; equal to the dev
    ids of the 500-doc zero-shot split; equal to the held-out ids of the ``final`` stage of
    ``folds`` (the documents the final adapter is documented not to have trained on); equal to
    the dev label files when given.
    """
    ids = sorted(dev_ids)
    want_n = EXPECTED_DEV_DOCS if expected is None else expected  # read at call time (tests)
    if len(set(ids)) != len(ids):
        raise DevFinalError("the dev document list has duplicate ids")
    if len(ids) != want_n:
        raise DevFinalError(f"expected {want_n} dev documents, found {len(ids)}")
    bad = [d for d in ids if not d.startswith("dev_")]
    if bad:
        raise DevFinalError(f"{len(bad)} ids of the dev list are not dev_* documents")
    zs_dev = sorted(d for d in zs500_ids if d.startswith("dev_"))
    if zs_dev != ids:
        raise DevFinalError(
            f"the dev list is not the dev ids of splits/zeroshot500.json "
            f"({len(ids)} vs {len(zs_dev)} ids)"
        )
    held = sorted(ts.stage_split("final", folds).heldout_ids)
    if held != ids:
        raise DevFinalError(
            f"the dev list is not the held-out ids of the final stage of folds.json "
            f"({len(ids)} vs {len(held)} ids)"
        )
    if label_ids is not None and sorted(label_ids) != ids:
        raise DevFinalError(
            f"the dev list is not the dev label files ({len(ids)} vs {len(label_ids)} ids)"
        )
    return ids


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_dev_ids(
    dev_docs: Path, zs500_docs: Path, folds: Mapping[str, Any], data_root: Path
) -> list[str]:
    """`check_dev_ids` on the split files and the dev label folder under `data_root`."""
    labels = Path(data_root) / SPLIT / "labels"
    label_ids = sorted(p.stem for p in labels.glob("*.json")) if labels.is_dir() else []
    return check_dev_ids(
        spike.load_doc_ids(str(dev_docs)), spike.load_doc_ids(str(zs500_docs)), folds, label_ids
    )


def _gpu_estimate() -> Any:
    spec = importlib.util.spec_from_file_location(
        "gpu_estimate", paths.REPO_ROOT / "scripts" / "gpu_estimate.py"
    )
    if spec is None or spec.loader is None:
        raise DevFinalError("scripts/gpu_estimate.py not found")
    ge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ge)
    return ge


def count_pages(ids: Sequence[str], data_root: Path) -> int:
    """Pages of the dev documents, counted from ``pages`` in their label files."""
    return int(_gpu_estimate().count_pages(list(ids), Path(data_root) / SPLIT / "labels"))


def run_plan(
    *,
    dev_docs: Path,
    zs500_docs: Path,
    folds: Mapping[str, Any],
    data_root: Path,
    out_path: Path | None = None,
    out: Callable[[str], None] = print,
) -> dict[str, Any]:
    """PLAN: assert the ids, count documents and pages (label files only), write the plan."""
    ids = load_dev_ids(dev_docs, zs500_docs, folds, data_root)
    plan = {
        "schema": DEVFINAL_SCHEMA,
        "split": SPLIT,
        "n_docs": len(ids),
        "n_pages": count_pages(ids, data_root),
        "ids_sha256": ids_sha256(ids),
        "ids_equal": ["splits/zeroshot500.json dev ids", "folds.json final held-out", "dev labels"],
    }
    out(
        f"plan: {plan['n_docs']} dev documents, {plan['n_pages']} pages (counted from the label "
        f"files); ids equal the dev ids of zeroshot500, the final stage's held-out ids and the "
        f"dev label files; ids sha256 {plan['ids_sha256'][:16]}"
    )
    if out_path is not None:
        spike.atomic_write(Path(out_path), json.dumps(plan, indent=1))
    return plan


# --------------------------------------------------------------------------------------------
# Estimate
# --------------------------------------------------------------------------------------------


def run_estimate(
    *,
    cfg: spike.SpikeConfig,
    zs_run_dir: Path,
    n_pages: int,
    batch_size: int | None,
    out: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Print the ESTIMATE table of the dev run (hours and CU on a T4); runs no model."""
    ge = _gpu_estimate()
    speed = json.loads((paths.REPO_ROOT / "configs" / "spike_speed.json").read_text("utf-8"))
    chosen = oof.resolve_batch_size(zs_run_dir, batch_size)
    rows = oof.estimate_rows(ge, speed, cfg.name, n_pages, chosen["batch_size"])
    text = oof.format_estimate(rows, n_pages, chosen["batch_size"], fold=0)
    head = "OOF fold 0, "
    if head not in text:
        raise DevFinalError("oof.format_estimate changed: the 05b title cannot be derived")
    out(text.replace(head, "05b final adapter on the dev documents, "))
    out(
        f"documents {EXPECTED_DEV_DOCS}, pages {n_pages} (counted from the label files), batch "
        f"size {chosen['batch_size']} ({chosen['source']}). ESTIMATE, UNVERIFIED."
    )
    return {"n_pages": n_pages, "batch": chosen, "rows": rows}


# --------------------------------------------------------------------------------------------
# Inference: VERIFY -> MERGE -> GUARD -> INFER
# --------------------------------------------------------------------------------------------


def run_infer(
    *,
    cfg: spike.SpikeConfig,
    adapter_dir: Path,
    zs_run_dir: Path,
    folds: Mapping[str, Any],
    dev_ids: Sequence[str],
    out_dir: Path,
    data_root: Path,
    backend_factory: Callable[[Path, str], Any],
    bench_docs: Sequence[str],
    batch_size: int | None = None,
    pin_sha: str | None = None,
    reachable: Callable[[str], bool] | None = None,
    use_wandb: bool = False,
    out: Callable[[str], None] = print,
) -> dict[str, Any]:
    """VERIFY -> MERGE -> GUARD -> INFER on the dev documents. Resumable; returns the record.

    The adapter must pass `predict_ft.verify_final_adapter` (a fold / smoke adapter is refused,
    the held-out ids must be the dev documents). `dev_ids` are re-checked against the ``final``
    stage of `folds`. A stored batch decision made for the same adapter / code / config / model is
    reused on a resume, so the batch size cannot change under a half-finished run.
    """
    out_dir = Path(out_dir)
    ids = sorted(dev_ids)
    held = sorted(ts.stage_split("final", folds).heldout_ids)
    if ids != held:
        raise DevFinalError("the documents to infer are not the dev documents of the final stage")
    report = predict_ft.verify_stage(
        adapter_dir=adapter_dir, cfg=cfg, folds=folds, pin_sha=pin_sha, reachable=reachable,
        out=out,
    )  # fmt: skip
    sha = str(report.get("adapter_sha256") or "")
    chosen = oof.resolve_batch_size(zs_run_dir, batch_size)
    if chosen["differs_from_zero_shot"]:
        out(
            f"WARNING: BATCH_SIZE {chosen['batch_size']} differs from the zero-shot run's "
            f"{chosen['stored']}: greedy outputs are only strictly comparable at the same size."
        )
    backend = backend_factory(Path(adapter_dir), sha)
    merge = predict_ft.merge_stage(backend, out)
    decision_path = out_dir / DECISION_NAME
    resume = (out_dir / "trace.jsonl").is_file()
    decision = predict_ft.load_decision(decision_path, cfg, backend, sha) if resume else None
    if decision is not None:
        out(
            "== GUARD == resumed run: the stored batch decision stands "
            f"(batch {decision['batch_size']})"
        )
        guard = {
            "ran": False,
            "reused_decision": True,
            "batch_size": decision["batch_size"],
            "fallback_to_1": decision["batch_size"] != decision["requested_batch_size"],
        }
    else:
        guard = predict_ft.guard_stage(
            cfg, backend, bench_docs, int(chosen["batch_size"]), out_dir / "guard", data_root, out
        )
        decision = predict_ft.make_decision(
            cfg, backend, guard, int(chosen["batch_size"]), out_dir / "guard", sha
        )
        spike.atomic_write(decision_path, json.dumps(decision, indent=1))
    batch = int(decision["batch_size"])
    out(f"== INFER == {len(ids)} dev documents at batch {batch}")
    bench_info = {
        "path": str(Path(zs_run_dir) / "bench_result.json"),
        "chosen": batch,
        "deviation": guard if guard.get("fallback_to_1") else None,
    }
    spike.run_spike(
        cfg, ids, SPLIT, out_dir.name, backend, resume=resume, use_wandb=use_wandb,
        runs_root=out_dir.parent, data_root=data_root, logprobs=True, batch_size=batch,
        bench_info=bench_info,
    )  # fmt: skip
    preds = _read_json(out_dir / "predictions.json")
    if sorted(preds) != ids:
        raise DevFinalError(
            f"the predictions hold {len(preds)} documents, not exactly the {len(ids)} dev ids"
        )
    record = {
        "schema": DEVFINAL_SCHEMA,
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
        },
        "merge": merge,
        "merge_precision_note": "LoRA update added to fp16 base weights (fp32 sum, then fp16)",
        "guard": guard,
        "batch": {**chosen, "used": batch},
        "decision": {k: decision.get(k) for k in ("batch_size", "requested_batch_size", "source")},
        "n_docs": len(ids),
        "n_pages": count_pages(ids, data_root),
        "ids_sha256": ids_sha256(ids),
        "zero_shot_run": Path(zs_run_dir).name,
    }
    man = read_manifest(out_dir) or {}
    write_manifest(out_dir, {**man, "devfinal": record})
    return record


# --------------------------------------------------------------------------------------------
# Compare (CPU)
# --------------------------------------------------------------------------------------------


def _replay_module() -> Any:
    """``scripts/replay_v1_check.py`` (its `production` is THE production rule path)."""
    spec = importlib.util.spec_from_file_location(
        "replay_v1_check", paths.REPO_ROOT / "scripts" / "replay_v1_check.py"
    )
    if spec is None or spec.loader is None:
        raise DevFinalError("scripts/replay_v1_check.py not found")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def shapes_sha256(shapes: rules.SlotShapes) -> str:
    """sha256 of the sorted shape lists (counts of shapes only are ever printed)."""
    blob = json.dumps(
        {"cpn_only": sorted(shapes.cpn_only), "po_only": sorted(shapes.po_only)}, sort_keys=True
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def run_arms(
    traces: Sequence[Mapping[str, Any]],
    raw_preds: Mapping[str, Any],
    ids: Sequence[str],
    *,
    ocr_root: Path | None,
    train_shapes: rules.SlotShapes,
    frozen_shapes: rules.SlotShapes,
    production: Callable[..., tuple[dict[str, Any], dict[str, Any]]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """``({arm: predictions}, {arm: rules summary})`` of one model on the dev documents.

    Checks (fail closed): every trace is a dev document of `ids` and each dev id has one; the
    production path with every rule OFF reproduces the run's ``predictions.json`` exactly; no rule
    was skipped for a missing / incomplete OCR cache (R2 needs the OCR of every dev waybill page).
    """
    want = set(ids)
    tr = [t for t in traces if t["doc_id"] in want]
    if sorted(t["doc_id"] for t in tr) != sorted(ids):
        raise DevFinalError(f"the trace covers {len(tr)} of the {len(ids)} dev documents")
    raw = {d: raw_preds[d] for d in sorted(want)}
    off, _ = production(tr, RuleConfig.all_off(), None, None)
    if off != raw:
        raise DevFinalError("the production path with all rules OFF does not reproduce predictions")
    preds: dict[str, dict[str, Any]] = {ARM_RAW: raw}
    summaries: dict[str, dict[str, Any]] = {ARM_RAW: {"switches": RuleConfig.all_off().as_dict()}}
    for arm, shapes in ((ARM_TRAIN, train_shapes), (ARM_FROZEN, frozen_shapes)):
        preds[arm], summaries[arm] = production(tr, RuleConfig(), shapes, ocr_root)
        skipped = {k: v for k, v in summaries[arm]["skipped"].items() if k.startswith("R2/")}
        if skipped:
            raise DevFinalError(f"R2 was skipped on dev waybills (OCR cache incomplete): {skipped}")
    return preds, summaries


def _block(raw: Mapping[str, Any]) -> dict[str, Any]:
    return dict(raw)


def compare_arm(
    zs_pred: Mapping[str, Any],
    ft_pred: Mapping[str, Any],
    gold: Mapping[str, Mapping[str, Any]],
    meta: list[dict[str, Any]] | None,
    n_boot: int,
    seed: int = SEED,
) -> dict[str, Any]:
    """FT vs ZS on the documents of `gold` under one arm: `oof.compare_models`, relabelled.

    ``oof.compare_models`` calls them ``zero_shot`` / ``oof`` and adds a fold-0 G4 verdict that has
    no meaning here; this returns ``{subset: {n_docs, zs, ft, paired_delta_ft_minus_zs}}`` only.
    """
    try:
        cmp = oof.compare_models(zs_pred, ft_pred, gold, meta, n_boot=n_boot, seed=seed)
    except oof.OofError as exc:
        raise DevFinalError(str(exc)) from exc
    return {
        label: {
            "n_docs": sub["n_docs"],
            "zs": _block(sub["zero_shot"]),
            "ft": _block(sub["oof"]),
            "paired_delta_ft_minus_zs": sub["paired_delta_oof_minus_zero_shot"],
        }
        for label, sub in cmp["subsets"].items()
    }


def _is_complete(run_dir: Path) -> bool:
    prog = Path(run_dir) / "progress.json"
    return prog.is_file() and _read_json(prog).get("status") == "complete"


def plumbing_cells(arms: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    """Counts for the plumbing check: paired-delta cells examined and how many are not exactly 0."""
    total = nonzero = 0
    for sub_by_arm in arms.values():
        for sub in sub_by_arm["subsets"].values():
            for k in METRICS:
                d = sub["paired_delta_ft_minus_zs"][k]
                total += 1
                nonzero += any(d[x] != 0.0 for x in ("delta", "lo", "hi"))
    return {"cells": total, "nonzero": nonzero}


def run_compare(
    *,
    zs_run_dir: Path,
    ft_run_dir: Path | None,
    out_dir: Path,
    dev_docs: Path,
    zs500_docs: Path,
    folds: Mapping[str, Any],
    data_root: Path,
    ocr_root: Path | None,
    n_boot: int = N_BOOT,
    plumbing: bool = False,
    out: Callable[[str], None] = print,
) -> dict[str, Any]:
    """SCORE + COMPARE on the dev documents; writes ``devfinal_compare.json`` / ``.md``.

    `plumbing` takes the 02 run's dev documents as BOTH arms (`ft_run_dir` is ignored): the result
    is stamped NOT A RESULT and every paired delta must be exactly 0.
    """
    out_dir = Path(out_dir)
    ids = load_dev_ids(dev_docs, zs500_docs, folds, data_root)
    try:
        _zs_man, zs_pred = oof.load_zero_shot(zs_run_dir)
    except oof.OofError as exc:
        raise DevFinalError(str(exc)) from exc
    ft_dir = Path(zs_run_dir) if plumbing else ft_run_dir
    if ft_dir is None:
        raise DevFinalError("compare needs --ft-run-dir (or --plumbing-check)")
    if not plumbing:
        if not _is_complete(ft_dir):
            raise DevFinalError(f"{Path(ft_dir).name}: the dev run is not complete")
        man = read_manifest(ft_dir) or {}
        if "devfinal" not in man:
            raise DevFinalError(
                f"{Path(ft_dir).name}: not a 05b run (no devfinal manifest section)"
            )
        if (man["devfinal"].get("ids_sha256")) != ids_sha256(ids):
            raise DevFinalError("the run's dev ids differ from the dev list")
        ft_pred = _read_json(Path(ft_dir) / "predictions.json")
        if sorted(ft_pred) != ids:
            raise DevFinalError(f"the dev run's predictions are not exactly the {len(ids)} dev ids")
    else:
        ft_pred = zs_pred
    missing = [d for d in ids if d not in zs_pred]
    if missing:
        raise DevFinalError(f"the zero-shot run misses {len(missing)} of the dev documents")
    gold, meta = spike.load_gold_and_meta([SPLIT], Path(data_root))
    if gold is None or sorted(gold) != ids:
        raise DevFinalError("the dev gold labels are not exactly the dev ids")
    train_gold = ev.load_gold(Path(data_root) / "train" / "labels")
    if any(not d.startswith("train_") for d in train_gold):
        raise DevFinalError("the R3 shape learner got a non-train document")
    train_shapes = rules.learn_slot_shapes(list(train_gold.values()))
    frozen_shapes, frozen_sha = postrules.load_slot_shapes()
    production = _replay_module().production
    zs_arms, zs_sum = run_arms(
        read_trace(Path(zs_run_dir) / "trace.jsonl"), zs_pred, ids, ocr_root=ocr_root,
        train_shapes=train_shapes, frozen_shapes=frozen_shapes, production=production,
    )  # fmt: skip
    ft_arms, ft_sum = run_arms(
        read_trace(Path(ft_dir) / "trace.jsonl"), ft_pred, ids, ocr_root=ocr_root,
        train_shapes=train_shapes, frozen_shapes=frozen_shapes, production=production,
    )  # fmt: skip
    out("== SCORE / COMPARE ==")
    arms: dict[str, Any] = {}
    for arm, title in ARMS.items():
        arms[arm] = {
            "title": title,
            "subsets": compare_arm(zs_arms[arm], ft_arms[arm], gold, meta, n_boot),
            "rules": {"zs": zs_sum[arm], "ft": ft_sum[arm]},
        }
    result: dict[str, Any] = {
        "schema": DEVFINAL_SCHEMA,
        "kind": "plumbing_check" if plumbing else "dev_final",
        "label": PLUMBING_LABEL if plumbing else RESULT_LABEL,
        "n_docs": len(ids),
        "n_boot": n_boot,
        "seed": SEED,
        "ids_sha256": ids_sha256(ids),
        "zero_shot_run": Path(zs_run_dir).name,
        "ft_run": None if plumbing else Path(str(ft_dir)).name,
        "shapes": {
            "train": {
                "sha256": shapes_sha256(train_shapes),
                "n_cpn_only": len(train_shapes.cpn_only),
                "n_po_only": len(train_shapes.po_only),
                "learned_from": f"{len(train_gold)} train gold documents",
            },
            "frozen": {"sha256": frozen_sha, "learned_from": "all 500 train+dev docs (in-sample)"},
        },
        "arms": arms,
        "official_dev": official_dev(arms),
    }
    if plumbing:
        cells = plumbing_cells(arms)
        result["plumbing"] = {**cells, "all_zero": cells["nonzero"] == 0}
    spike.atomic_write(out_dir / COMPARE_NAME, json.dumps(result, indent=1))
    spike.atomic_write(out_dir / COMPARE_MD_NAME, format_compare(result, markdown=True))
    out(format_compare(result))
    return result


def official_dev(arms: Mapping[str, Any]) -> dict[str, Any]:
    """The final model's official dev scores with 95% CI (the 'seen layouts' number), per arm."""
    res: dict[str, Any] = {}
    for arm, a in arms.items():
        s = a["subsets"]["all"]
        res[arm] = {
            "ft_OVERALL": s["ft"]["ci95"]["OVERALL"],
            "zs_OVERALL": s["zs"]["ci95"]["OVERALL"],
            "paired_delta_OVERALL": s["paired_delta_ft_minus_zs"]["OVERALL"],
        }
    return res


# --------------------------------------------------------------------------------------------
# Formatting: aggregates only
# --------------------------------------------------------------------------------------------


def _pct(x: float) -> str:
    return f"{100 * x:.2f}"


def _ci(ci: Mapping[str, float]) -> str:
    return f"{_pct(ci['point'])} [{_pct(ci['lo'])}, {_pct(ci['hi'])}]"


def format_banner(res: Mapping[str, Any]) -> str:
    """The completion banner: the seen-layout OVERALL of the final model and the verdict inputs."""
    bar = "=" * 78
    lines = [bar, str(res["label"]), bar]
    for arm, o in res["official_dev"].items():
        d = o["paired_delta_OVERALL"]
        lines.append(
            f"{arm:<20} FT OVERALL {_ci(o['ft_OVERALL'])}   ZS {_ci(o['zs_OVERALL'])}   "
            f"paired delta {100 * d['delta']:+.2f} [{_pct(d['lo'])}, {_pct(d['hi'])}]"
        )
    p = res["arms"][ARM_TRAIN]["subsets"]["all"]
    zn, fn = p["zs"]["over_null"], p["ft"]["over_null"]
    lines.append(
        f"{ARM_TRAIN}: false fills ZS {zn['false_fill_total']} -> FT {fn['false_fill_total']}, "
        f"over-nulls ZS {zn['over_null_total']} -> FT {fn['over_null_total']} (header+row cells)"
    )
    if res["kind"] == "plumbing_check":
        pl = res["plumbing"]
        lines.append(
            f"PLUMBING: {pl['cells']} paired-delta cells examined, {pl['nonzero']} not exactly 0; "
            "NOT A RESULT"
        )
    else:
        lines.append(
            "seen layouts: the official dev OVERALL; it does NOT measure unseen suppliers (that "
            "is the out-of-fold / G4 comparison)"
        )
    lines.append(bar)
    return "\n".join(lines)


def format_compare(res: Mapping[str, Any], markdown: bool = False) -> str:
    """Printable (or markdown) comparison: aggregates only, no document id or value."""
    lines: list[str] = []
    title = f"05b final adapter vs zero-shot, {res['label']}"
    lines += [f"# {title}", ""] if markdown else ["=" * 110, title, "=" * 110]
    lines.append(
        f"percent; 95% CI = doc-level bootstrap, {res['n_boot']} resamples, seed {res['seed']}; "
        "delta = FT - zero-shot (paired, same resamples); official scorer, unmodified. R3 shapes: "
        f"train-only sha256 {res['shapes']['train']['sha256'][:12]} "
        f"({res['shapes']['train']['learned_from']}), frozen "
        f"{str(res['shapes']['frozen']['sha256'])[:12]}."
    )
    for arm, a in res["arms"].items():
        lines.append("")
        lines.append(("## " if markdown else "=== ") + f"arm {arm}: {a['title']}")
        for label, sub in a["subsets"].items():
            zs, ft, pd_ = sub["zs"], sub["ft"], sub["paired_delta_ft_minus_zs"]
            lines.append("")
            lines.append(("### " if markdown else "--- ") + f"{label} ({sub['n_docs']} documents)")
            if markdown:
                lines += [
                    "",
                    "| metric | zero-shot | final FT | paired delta |",
                    "|---|---|---|---|",
                ]
            for k in METRICS:
                cells = (
                    _ci(zs["ci95"][k]),
                    _ci(ft["ci95"][k]),
                    f"{100 * pd_[k]['delta']:+.2f} [{_pct(pd_[k]['lo'])}, {_pct(pd_[k]['hi'])}]",
                )
                lines.append(
                    f"| {k} | {cells[0]} | {cells[1]} | {cells[2]} |"
                    if markdown
                    else f"  {k:<26} ZS {cells[0]:<24} FT {cells[1]:<24} delta {cells[2]}"
                )
            zn, fn = zs["over_null"], ft["over_null"]
            for text, key in (
                ("header over-nulls", "header_over_null"),
                ("row over-nulls", "row_over_null"),
                ("over-nulls total", "over_null_total"),
                ("header false fills", "header_false_fill"),
                ("row false fills", "row_false_fill"),
                ("false fills total", "false_fill_total"),
            ):
                lines.append(
                    f"| {text} | {zn[key]} | {fn[key]} | {fn[key] - zn[key]:+d} |"
                    if markdown
                    else f"  {text:<26} ZS {zn[key]:<8} FT {fn[key]:<8} diff {fn[key] - zn[key]:+d}"
                )
            if label == "all":
                for per, key in (
                    ("header", "header_over_null_by_field"),
                    ("row", "row_over_null_by_field"),
                ):
                    fields = sorted(set(zn[key]) | set(fn[key]))
                    if fields:
                        cells_s = ", ".join(
                            f"{f}: {zn[key].get(f, 0)} -> {fn[key].get(f, 0)}" for f in fields
                        )
                        lines.append(f"  {per} over-nulls by field (ZS -> FT): {cells_s}")
        touched = a["rules"]["ft"].get("touched_docs")
        if touched is not None:
            lines.append(
                f"  rules touched docs (ZS): {a['rules']['zs'].get('touched_docs')}, "
                f"(FT): {touched}"
            )
    lines += ["", format_banner(res)]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Argument parser of ``python -m shipdoc.devfinal``."""
    p = argparse.ArgumentParser(
        prog="python -m shipdoc.devfinal", description=__doc__.split("\n")[0]
    )
    sub = p.add_subparsers(dest="stage", required=True)

    def common(s: argparse.ArgumentParser) -> None:
        s.add_argument("--config", type=Path, default=Path("configs/spike_qwen35_4b_img_only.yaml"))
        s.add_argument("--data-root", type=Path, default=None, help="default: SHIPDOC_DATA_DIR")
        s.add_argument("--folds", type=Path, default=paths.REPO_ROOT / "splits" / "folds.json")
        s.add_argument("--dev-docs", type=Path, default=paths.REPO_ROOT / "splits" / "dev100.json")
        s.add_argument(
            "--zs500-docs", type=Path, default=paths.REPO_ROOT / "splits" / "zeroshot500.json"
        )

    def zs(s: argparse.ArgumentParser, required: bool = True) -> None:
        s.add_argument("--zs-run-dir", type=Path, required=required)

    def adapter(s: argparse.ArgumentParser) -> None:
        s.add_argument("--adapter-dir", type=Path, required=True, help="<final run>/final")
        s.add_argument("--pin", default=None, help="this notebook's pinned 40-hex code SHA")

    s = sub.add_parser("plan", help="assert the dev ids; count documents and pages")
    common(s)
    s.add_argument("--out", type=Path, default=None)
    s = sub.add_parser("estimate", help="ESTIMATE of T4 hours and CU (prints only)")
    common(s)
    zs(s)
    s.add_argument("--batch-size", type=int, default=None)
    s = sub.add_parser("verify", help="final adapter manifest verification (fails closed)")
    common(s)
    adapter(s)
    s = sub.add_parser("infer", help="verify, merge, guard, resumable dev run")
    common(s)
    zs(s)
    adapter(s)
    s.add_argument("--out-dir", type=Path, required=True)
    s.add_argument("--bench-docs", default="splits/bench12.json")
    s.add_argument("--batch-size", type=int, default=None)
    s.add_argument("--wandb", action="store_true")
    s = sub.add_parser("compare", help="CPU: score and compare on the dev documents")
    common(s)
    zs(s)
    s.add_argument("--ft-run-dir", type=Path, default=None)
    s.add_argument("--out-dir", type=Path, required=True)
    s.add_argument("--ocr-cache", type=Path, default=None, help="default: SHIPDOC_OCR_CACHE")
    s.add_argument("--n-boot", type=int, default=N_BOOT)
    s.add_argument(
        "--plumbing-check",
        action="store_true",
        help="the 02 run's dev docs are BOTH arms: NOT A RESULT, every delta must be 0",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Stage CLI. Exit 0 = the stage passed; 1 = refused / a check failed (counts only)."""
    a = build_parser().parse_args(argv)
    try:
        return _dispatch(a)
    except (DevFinalError, predict_ft.FtError, oof.OofError) as exc:
        print(f"devfinal {a.stage}: REFUSED: {exc}", file=sys.stderr)
        return 1


def _dispatch(a: argparse.Namespace) -> int:
    paths.apply_env()  # before anything can import transformers
    data = Path(a.data_root) if a.data_root else paths.data_dir()
    folds = _read_json(a.folds)
    if a.stage == "plan":
        run_plan(dev_docs=a.dev_docs, zs500_docs=a.zs500_docs, folds=folds, data_root=data,
                 out_path=a.out)  # fmt: skip
        return 0
    if a.stage == "compare":
        ocr = Path(a.ocr_cache) if a.ocr_cache else paths.ocr_cache_dir()
        res = run_compare(
            zs_run_dir=a.zs_run_dir, ft_run_dir=a.ft_run_dir, out_dir=a.out_dir,
            dev_docs=a.dev_docs, zs500_docs=a.zs500_docs, folds=folds, data_root=data,
            ocr_root=ocr, n_boot=a.n_boot, plumbing=a.plumbing_check,
        )  # fmt: skip
        if res["kind"] == "plumbing_check":
            print(
                f"plumbing check: docs {res['n_docs']}, paired-delta cells "
                f"{res['plumbing']['cells']}, not exactly 0: {res['plumbing']['nonzero']}"
            )
            return 0 if res["plumbing"]["all_zero"] else 1
        return 0
    cfg = spike.load_config(a.config)
    ids = load_dev_ids(a.dev_docs, a.zs500_docs, folds, data)
    if a.stage == "estimate":
        run_estimate(cfg=cfg, zs_run_dir=a.zs_run_dir, n_pages=count_pages(ids, data),
                     batch_size=a.batch_size)  # fmt: skip
        return 0
    if a.stage == "verify":
        predict_ft.verify_stage(adapter_dir=a.adapter_dir, cfg=cfg, folds=folds, pin_sha=a.pin,
                                reachable=oof.git_reachable)  # fmt: skip
        return 0

    def factory(adapter_dir: Path, sha: str) -> Any:
        return oof.MergedHfBackend(cfg.backend, adapter_dir, sha)

    rec = run_infer(
        cfg=cfg, adapter_dir=a.adapter_dir, zs_run_dir=a.zs_run_dir, folds=folds, dev_ids=ids,
        out_dir=a.out_dir, data_root=data, backend_factory=factory,
        bench_docs=spike.load_doc_ids(a.bench_docs), batch_size=a.batch_size, pin_sha=a.pin,
        reachable=oof.git_reachable, use_wandb=a.wandb,
    )  # fmt: skip
    print(f"dev run {Path(a.out_dir).name}: {rec['n_docs']} docs, {rec['n_pages']} pages, batch "
          f"{rec['batch']['used']}")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
