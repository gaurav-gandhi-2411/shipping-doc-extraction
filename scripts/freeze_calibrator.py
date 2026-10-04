"""Freeze the production calibrator (CPU, no model, no network) into a portable artifact.

    # zero-shot arm: the v1 fallback and the ZS+rules arm of the final system rule
    uv run python scripts/freeze_calibrator.py --arm zs \
        --run-dir $SHIPDOC_RUNS_DIR/zeroshot500/<02 run folder> \
        --calibration-dir $SHIPDOC_RUNS_DIR/calibration_v2/<run folder> --out meta/calibrator_zs.json
    # fine-tuned arm (needs the three OOF runs and the 3-fold calibrate_v3 outputs: NOT YET THERE)
    uv run python scripts/freeze_calibrator.py --arm ft \
        --oof-run <fold0> --oof-run <fold1> --oof-run <fold2> \
        --zs-run-dir <02 run folder> --calibration-dir <calibrate_v3 3-fold output folder> \
        --out meta/calibrator_ft.json

WHAT IS FROZEN WHERE (the artifact is ``meta/calibrator_<arm>.json``, read by ``shipdoc.flags``):

* Model structure and kind (pooled vs per_type, LR vs GBM): copied from the calibration run that
  produced the OOF outputs (``calibration_v2.json`` ``model_selection`` for ZS, the judge_ft view of
  ``calibration_v3.json`` for FT). Nothing is re-selected here.
* Field model(s): fitted on ALL emitted rows of the 500 train + dev documents (the production
  calibrator is fit on everything), with the same fit rules as the cross-fit
  (``shipdoc.flags.fit_model`` mirrors ``confidence.fit_predict``). The layout-novelty column of a
  training row is computed against the signatures of the OTHER supplier folds (what the cross-fit
  does for its fitting rows); at inference it is computed against the production reference
  (cluster centres of all 500 signatures, stored in the artifact).
* Document model: logistic regression on the aggregated OUT-OF-FOLD field probabilities
  (``confidence_v2.doc_table``) and document label ``documents_fully_correct``, fitted on all 500
  documents. At inference its input features come from the final field model's probabilities, which
  are in-sample-fitted and so slightly sharper than the OOF ones it was trained on (stacking
  caveat, stated in the artifact).
* tau (the stored thresholds): for every field type (header + four row fields) and target 95 / 98 /
  99%: ``confidence.select_tau`` on ALL POOLED OUT-OF-FOLD predictions of that type (the lowest
  confidence whose accepted set has precision >= target); for documents at 95 / 98%: the same on
  the pooled OOF document probabilities (recomputed here by the supplier-fold document cross-fit).
  ``null`` = the target is not reached anywhere on the pooled OOF rows: nothing of that type is
  auto-accepted. WHY pooled and not the per-fold taus of the nested protocol: the nested protocol
  (tau of fold k from the other folds, applied to fold k) is the honest EVALUATION of a threshold
  rule; its per-fold taus are three different numbers, none of which is "the" production threshold.
  The production rule is the same rule applied to all the OOF evidence at once. The nested
  estimates (precision, coverage, review rate, error recall with document-bootstrap CIs) are stored
  beside the thresholds as the expected performance; the pooled in-sample precision / coverage at
  tau is stored too and labelled optimistic. The final model's probabilities differ a little from
  the cross-fitted ones tau was chosen on (more data, in-sample fit): another reason to read the
  nested numbers, not the in-sample ones.
* Parity: the artifact records the feature names and a digest of the design matrix the current
  feature builder makes of a fixed synthetic document (``shipdoc.flags.parity_probe``); the flags
  stage refuses the artifact if either differs.
* sklearn: LR models are plain numbers; a GBM is a pickle that ``shipdoc.flags`` loads only under
  the sklearn version recorded here (and after a sha256 check), else REFUSES.

Test data never enters: every document id here must be train / dev (``test_`` aborts).
No extracted value is written: the artifact holds numbers, field names and hashes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from shipdoc import confidence as cf
from shipdoc import confidence_v2 as c2
from shipdoc import confidence_v3 as c3
from shipdoc import eval as ev
from shipdoc import flags, paths
from shipdoc import meta as meta_mod
from shipdoc.runcompat import (
    ResolutionMismatchError,
    assert_runs_share_resolution,
    manifest_config_hash,
)
from shipdoc.runmeta import read_manifest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_VERSION_NOTE = "meta/calibrator_<arm>.json, schema 1 (shipdoc.flags.ARTIFACT_SCHEMA)"
MEANING = (
    "tau of a field type = lowest confidence whose accepted set has precision >= the target on ALL "
    "POOLED OUT-OF-FOLD predictions of that type (confidence.select_tau); null = the target is "
    "never reached there, so every field of that type goes to review. Document tau likewise on the "
    "pooled OOF document probabilities. The production models are fitted on all rows, so their "
    "probabilities differ a little from the cross-fitted ones tau was chosen on; read "
    "nested_estimates (tau of fold k from the other folds, applied to fold k) as the expected "
    "performance, not in_sample_at_tau."
)
STACKING_CAVEAT = (
    "the document model is fitted on OUT-OF-FOLD field probabilities and applied to the final "
    "field "
    "model's (in-sample-fitted, slightly sharper) probabilities"
)


class FreezeError(RuntimeError):
    """An input failed a fail-closed check (messages never quote a value)."""


def _load_script(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses / typing resolve annotations via sys.modules
    spec.loader.exec_module(mod)
    return mod


def _scripts() -> tuple[Any, Any]:
    """``(calibrate_v3 module, calibrate_v2 module)``, loaded as calibrate_v3 itself does."""
    v3s = _load_script("calibrate_v3", ROOT / "scripts" / "calibrate_v3.py")
    return v3s, v3s.v2s


@dataclass
class Inputs:
    """Everything `fit_artifact` needs, from either arm's calibration run."""

    arm: str
    name: str
    tv: c2.TableV2
    y: np.ndarray  # bool (n rows): the field label
    p_oof: np.ndarray  # (n rows): OOF P(correct), NaN where not emitted
    y_doc: np.ndarray  # bool (n docs), aligned with `doc_ids`
    doc_ids: list[str]
    doc_fold: dict[str, int]
    groups: dict[str, str]
    sigs: dict[str, np.ndarray | None]
    structure: str
    kind: str
    nested_field: list[dict[str, Any]]
    agree: np.ndarray | None = None
    source: dict[str, Any] = field(default_factory=dict)
    p_doc_csv: dict[str, float] | None = None  # ZS only: calibrate_v2's own OOF document p


# --------------------------------------------------------------------------------------------
# The fit (pure given the arrays: tested on synthetic data)
# --------------------------------------------------------------------------------------------


def _lf_sha(path: Path) -> str:
    """sha256 of a text file with CRLF normalised (identical on Windows and Linux checkouts)."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def pooled_taus(
    p: np.ndarray, y: np.ndarray, em: np.ndarray, types: np.ndarray
) -> dict[str, dict[str, Any]]:
    """Per target and field type: tau (pooled OOF), n rows, in-sample precision / coverage."""
    out: dict[str, dict[str, Any]] = {}
    for t in flags.FIELD_TARGETS:
        per: dict[str, Any] = {}
        for ft in c2.FIELD_TYPES:
            m = em & (types == ft)
            tau = cf.select_tau(p[m], y[m], t) if m.any() else None
            per[ft] = _tau_record(tau, p[m], y[m])
        out[flags.tkey(t)] = per
    return out


#: Minimum accepted share of documents for the document-level target to count as attained
#: (``scripts/calibrate_v2.py``: nested precision >= target AND coverage >= 5%).
MIN_DOC_COVERAGE = 0.05


def _ge(value: Any, bound: float) -> bool:
    """``value >= bound`` that is False for None / NaN (a type with nothing accepted)."""
    return value is not None and not np.isnan(value) and bool(value >= bound)


def nested_target_met(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, bool | None]]:
    """``{target: {field type: nested precision point >= target}}`` (None = no nested record)."""
    out: dict[str, dict[str, bool | None]] = {}
    for t in flags.FIELD_TARGETS:
        per: dict[str, bool | None] = {}
        for ft in c2.FIELD_TYPES:
            rec = next((r for r in records if r["population"] == ft and r["target"] == t), None)
            per[ft] = None if rec is None else _ge(rec["precision_accepted"]["point"], t)
        out[flags.tkey(t)] = per
    return out


def _tau_record(tau: float | None, p: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    if tau is None or len(p) == 0:
        return {"tau": None, "n": int(len(p)), "precision": None, "coverage": None}
    acc = p >= tau
    return {
        "tau": float(tau),
        "n": int(len(p)),
        "precision": float(y[acc].mean()) if acc.any() else None,
        "coverage": float(acc.mean()),
    }


def fit_artifact(
    inp: Inputs, n_boot: int = c2.N_BOOT, atol_doc_csv: float = 1e-6
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit the production models and thresholds; returns ``(artifact dict, report dict)``.

    The report holds checks that are not part of the artifact (document OOF vs the calibrate_v2
    CSV, sizes). Raises `FreezeError` on a failed fail-closed check.
    """
    c2.assert_no_test_ids(inp.doc_ids)
    tv, ids = inp.tv, list(inp.doc_ids)
    em = tv.emitted
    types = tv.types()
    if np.isnan(inp.p_oof[em]).any():
        raise FreezeError("an emitted row has no OOF probability")

    # novelty: training rows against the OTHER folds, inference against all (production reference)
    nov_table = cf.compute_novelty_table(inp.sigs, inp.doc_fold)
    nov_train = {d: nov_table[inp.doc_fold[d]][d] for d in ids}
    present = [s for s in inp.sigs.values() if s is not None]
    reference = cf.novelty_reference(present)

    tables = flags.Tables(tv, inp.sigs, inp.agree, ids)
    X = flags.design(inp.arm, tables, nov_train)
    if inp.structure == "pooled":
        models = {"all": flags.fit_model(inp.kind, X[em], inp.y[em])}
    else:
        models = {}
        for ft in c2.FIELD_TYPES:
            m = em & (types == ft)
            # a type without a single emitted training row gets the constant prior 0.0 (never
            # accepted; its tau is null too) instead of no model: inference must not crash on it
            models[ft] = flags.fit_model(inp.kind, X[m], inp.y[m])

    # document model on the OOF field probabilities
    Xd_raw, order = c2.doc_table(tv, inp.p_oof, ids)
    if order != ids:
        raise FreezeError("document order changed")
    p_doc_oof = c2.cross_fit_docs(Xd_raw, inp.y_doc, ids, inp.doc_fold, nov_table, inp.groups)
    report: dict[str, Any] = {}
    if inp.p_doc_csv is not None:
        ref = np.array([inp.p_doc_csv[d] for d in ids])
        diff = float(np.max(np.abs(ref - p_doc_oof)))
        report["doc_oof_max_abs_diff_vs_csv"] = diff
        if diff > atol_doc_csv:
            raise FreezeError(
                f"recomputed OOF document probabilities differ from the calibration CSV by "
                f"{diff:.3g}"
                f" (> {atol_doc_csv}): the run folder is not the one the calibration used"
            )
    Xd = flags.doc_design(tv, inp.p_oof, ids, nov_train)
    doc_model = flags.fit_model("lr", Xd, inp.y_doc)

    # thresholds: pooled OOF
    field_tau = pooled_taus(inp.p_oof, inp.y, em, types)
    doc_tau: dict[str, Any] = {}
    for t in flags.DOC_TARGETS:
        tau = cf.select_tau(p_doc_oof, inp.y_doc, t)
        doc_tau[flags.tkey(t)] = tau if tau is None else float(tau)
    # nested document-level estimates (field-level nested records come from the calibration run)
    fold_of = np.array([inp.doc_fold[d] for d in ids])
    d_ids = np.array(ids, dtype=object)
    nested_doc = []
    for t in flags.DOC_TARGETS:
        acc, _, _ = c2.nested_accept(p_doc_oof, inp.y_doc, fold_of, np.ones(len(ids), bool), t)
        mt = c2.accept_metrics(acc, inp.y_doc, d_ids, n_boot)
        nested_doc.append({"population": "doc", "target": t, "n": len(ids),
                           **{k: mt[k] for k in ("precision_accepted", "accepted_share",
                                                 "review_rate", "error_recall")}})  # fmt: skip
    doc_in_sample = {
        flags.tkey(t): _tau_record(doc_tau[flags.tkey(t)], p_doc_oof, inp.y_doc)
        for t in flags.DOC_TARGETS
    }
    met_field = nested_target_met(inp.nested_field)
    met_doc = {
        flags.tkey(r["target"]): bool(
            _ge(r["precision_accepted"]["point"], r["target"])
            and _ge(r["accepted_share"]["point"], MIN_DOC_COVERAGE)
        )
        for r in nested_doc
    }

    art: dict[str, Any] = {
        "schema": flags.ARTIFACT_SCHEMA,
        "artifact": flags.ARTIFACT_KIND,
        "arm": inp.arm,
        "variant": "v2" if inp.arm == "zs" else "v2_agree",
        "sklearn_version": flags.sklearn_version(),
        "numpy_version": np.__version__,
        "feature_names": flags.feature_names(inp.arm),
        "field_model": {"structure": inp.structure, "kind": inp.kind, "models": models},
        "doc_model": doc_model,
        "doc_features": list(c2.DOC_FEATURES),
        "doc_model_note": STACKING_CAVEAT,
        "novelty": {
            "reference": reference.tolist(),
            "n_signatures": len(present),
            "train_construction": (
                "a training row's novelty = distance to the nearest cluster centre of the "
                "signatures of the OTHER supplier folds; inference: all 500 signatures"
            ),
        },
        "thresholds": {
            "meaning": MEANING,
            "primary_target": flags.PRIMARY_TARGET,
            "field": {
                "tau": {t: {ft: r["tau"] for ft, r in per.items()} for t, per in field_tau.items()},
                "in_sample_at_tau": field_tau,
            },
            "doc": {"tau": doc_tau, "in_sample_at_tau": doc_in_sample},
            "nested_target_met": {
                "note": (
                    "does the NESTED (honest) precision point estimate reach the target? Fields: "
                    "per type; documents: also needs >= 5% of the documents accepted (the "
                    "calibrate_v2 attainability rule). false = the stored tau exists but the "
                    "expected precision misses the target: treat its accepts as unproven"
                ),
                "field": met_field,
                "doc": met_doc,
            },
        },
        "nested_estimates": {
            "note": (
                "tau of fold k chosen on the other folds' OOF predictions and applied to fold k, "
                "pooled; CIs = document-level bootstrap conditioned on the taus (the calibration "
                "run's own numbers for fields; computed here for documents)"
            ),
            "field": inp.nested_field,
            "doc": nested_doc,
        },
        "parity_probe": flags.parity_probe(inp.arm),
        "source": {
            "name": inp.name,
            **inp.source,
            "n_docs": len(ids),
            "n_emitted_fields": int(em.sum()),
            "code_sha256_lf": {
                p: _lf_sha(ROOT / p)
                for p in (
                    "src/shipdoc/confidence.py",
                    "src/shipdoc/confidence_v2.py",
                    "src/shipdoc/confidence_v3.py",
                    "src/shipdoc/flags.py",
                    "scripts/freeze_calibrator.py",
                )
            },
            "git_head": _git_head(),
        },
    }
    report["n_models"] = len(models)
    report["tau_field_primary"] = {
        ft: field_tau[flags.tkey(flags.PRIMARY_TARGET)][ft]["tau"] for ft in c2.FIELD_TYPES
    }
    report["tau_doc"] = doc_tau
    return art, report


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def recorded_run_config_hash(run_dir: Path | None) -> str | None:
    """``config.hash`` of the zero-shot run folder the calibrator is frozen from, else None.

    The hash covers the whole inference config (``max_pixels`` included), so it is the resolution
    the calibrator's probabilities belong to. ``shipdoc.flags.run_stage`` refuses a submission
    whose run used another hash. None (no folder, no manifest, no hash) records nothing: the
    artifact then carries no ``run_config_hash``, as the 1260-token ``meta/calibrator_zs.json``.
    """
    if run_dir is None or not Path(run_dir).is_dir():
        return None
    return manifest_config_hash(read_manifest(Path(run_dir)))


def write_artifact(art: Mapping[str, Any], out: Path) -> str:
    """Write the artifact (one JSON line, sorted keys, LF) and return its sha256."""
    text = json.dumps(art, sort_keys=True, separators=(",", ":")) + "\n"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(text.encode("utf-8"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------------
# Loading the calibration runs (real data)
# --------------------------------------------------------------------------------------------

Key = tuple[str, str, str, int]


def read_oof_csv(
    path: Path, p_col: str, view: str | None = None
) -> dict[Key, tuple[bool, float, bool]]:
    """``{(doc, scope, field, row_idx): (emitted, p, y_correct)}`` of a calibration OOF CSV."""
    out: dict[Key, tuple[bool, float, bool]] = {}
    with Path(path).open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if view is not None and r.get("view") != view:
                continue
            raw = r[p_col]
            p = float("nan") if raw in ("", "nan", "NaN") else float(raw)
            out[(r["doc_id"], r["scope"], r["field"], int(r["row_idx"]))] = (
                r["emitted"] == "1",
                p,
                r["y_correct"] == "1",
            )
    return out


def align_oof(
    tv: c2.TableV2, y: np.ndarray, oof: Mapping[Key, tuple[bool, float, bool]]
) -> np.ndarray:
    """OOF probabilities aligned with `tv.keys`; every key, emitted flag and label must agree."""
    keys = [(k.doc_id, k.scope, k.field, k.row_idx) for k in tv.keys]
    if set(keys) != set(oof) or len(keys) != len(set(keys)):
        raise FreezeError(
            f"the OOF CSV has {len(oof)} fields, the rebuilt feature table {len(keys)}: "
            "the run folder is not the one the calibration used"
        )
    p = np.full(len(keys), np.nan)
    bad_em = bad_y = 0
    for i, k in enumerate(keys):
        e, pi, yi = oof[k]
        bad_em += bool(e) != bool(tv.emitted[i])
        bad_y += bool(yi) != bool(y[i])
        p[i] = pi if e else np.nan
    if bad_em or bad_y:
        raise FreezeError(
            f"{bad_em} emitted flags and {bad_y} labels differ between the OOF CSV and the "
            "rebuilt table: refusing (wrong run, rules or shapes)"
        )
    return p


def _sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _nested_field_records(
    recs: Sequence[Mapping[str, Any]], keep: Mapping[str, Any]
) -> list[dict[str, Any]]:
    cols = ("precision_accepted", "accepted_share", "review_rate", "error_recall")
    return [
        {
            "population": r["population"],
            "target": r["target"],
            "n": r["n"],
            **{c: r[c] for c in cols},
        }  # fmt: skip
        for r in recs
        if all(r.get(k) == v for k, v in keep.items()) and r["population"] in c2.FIELD_TYPES
    ]


def build_zs_inputs(run_dir: Path, calibration_dir: Path, ocr_cache: Path | None = None) -> Inputs:
    """Rebuild the ZS calibration's tables (as ``calibrate_v2.run`` does), join its OOF output."""
    v3s, v2s = _scripts()
    v1cal = v2s.v1cal
    cal_dir = Path(calibration_dir)
    ids = json.loads((ROOT / "splits" / "zeroshot500.json").read_text(encoding="utf-8"))
    c2.assert_no_test_ids(ids)
    preds, traces = v1cal.load_run(Path(run_dir), ids)
    n_lp = sum(1 for t in traces.values() if any(p.get("field_logprobs") for p in t["pages"]))
    if n_lp < len(ids):
        raise FreezeError(f"only {n_lp}/{len(ids)} documents carry field_logprobs")
    folds = json.loads((ROOT / "splits" / "folds.json").read_text(encoding="utf-8"))
    gold, meta = v1cal.load_corpus(ids)
    doc_fold = v1cal.restricted_folds(ids, folds, meta)
    groups = {d: meta[d]["supplier_group"] for d in ids}
    labels = meta_mod.load_labels("train") + meta_mod.load_labels("dev")
    shapes_fn, _eq = v2s.fold_shape_fn(labels, doc_fold, groups)
    tr_list = [traces[d] for d in ids]
    _check_rules_off(v3s, tr_list, {d: preds[d] for d in ids})
    post, changes, _summary = v2s.post_rule_predictions(tr_list, shapes_fn, ocr_cache)
    ocr = ev.load_ocr_pages(list(ids), ocr_cache)
    tables = flags.build_tables("zs", ids, post, traces, ocr, changes)
    lab = {d: cf.label_doc(post[d], gold[d]) for d in ids}
    y, _ = cf.label_arrays(tables.tv.keys, lab)
    sc = ev.load_scorer()
    y_doc = np.array([c2.doc_exact(sc, post[d], gold[d]) for d in ids])
    cal_json = json.loads((cal_dir / "calibration_v2.json").read_text(encoding="utf-8"))
    sel = cal_json["model_selection"]
    oof = read_oof_csv(cal_dir / "oof_fields_v2.csv", "p_correct_v2")
    p = align_oof(tables.tv, y, oof)
    p_doc = {}
    with (cal_dir / "oof_docs_v2.csv").open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            p_doc[r["doc_id"]] = float(r["p_fully_correct"])
    nested = _nested_field_records(cal_json["tau"], {"slice": "all", "scheme": "nested"})
    return Inputs(
        "zs", f"{Path(run_dir).name} + {cal_dir.name}", tables.tv, y, p, y_doc, list(ids),
        doc_fold, groups, tables.sigs, sel["chosen_structure"], sel["chosen_kind"], nested,
        None,
        {
            "kind": "calibrate_v2",
            "run_dir": Path(run_dir).name,
            "calibration_dir": cal_dir.name,
            "calibration_json_sha256": _sha_file(cal_dir / "calibration_v2.json"),
            "oof_fields_sha256": _sha_file(cal_dir / "oof_fields_v2.csv"),
            "oof_docs_sha256": _sha_file(cal_dir / "oof_docs_v2.csv"),
            "model_selection": sel,
        },
        p_doc,
    )  # fmt: skip


def _check_rules_off(
    v3s: Any, tr_list: Sequence[Mapping[str, Any]], preds: Mapping[str, Any]
) -> None:
    from shipdoc import postrules
    from shipdoc.coerce import coerce_predictions, repair_predictions
    from shipdoc.postrules import RuleConfig

    raw = postrules.postprocess_traces(tr_list, RuleConfig.all_off(), None, None)[0]
    off, _ = repair_predictions(coerce_predictions(dict(raw)))
    if off != dict(preds):
        raise FreezeError("all rules OFF does not reproduce the run's predictions.json")


def build_ft_inputs(
    oof_runs: Sequence[Path], zs_run_dir: Path, calibration_dir: Path,
    ocr_cache: Path | None = None,
) -> Inputs:  # fmt: skip
    """Rebuild the FT calibration's tables (as ``calibrate_v3.run`` does) and join its OOF outputs.

    UNVERIFIED on real data: the three fine-tuned OOF runs and the 3-fold ``calibrate_v3`` outputs
    do not exist yet. Refuses unless the runs cover folds 0, 1, 2 exactly (`check_fold_coverage`).
    """
    v3s, v2s = _scripts()
    v1cal = v2s.v1cal
    cal_dir = Path(calibration_dir)
    ids500 = json.loads((ROOT / "splits" / "zeroshot500.json").read_text(encoding="utf-8"))
    c2.assert_no_test_ids(ids500)
    folds = json.loads((ROOT / "splits" / "folds.json").read_text(encoding="utf-8"))
    msg = v3s.gate_args([0, 1, 2], list(oof_runs), False)
    if msg:
        raise FreezeError(msg)
    try:  # the FT calibrator sees ZS agreement features: one input resolution on all four runs
        assert_runs_share_resolution(
            {"ZS run": Path(zs_run_dir), **{f"FT run {i}": Path(p) for i, p in enumerate(oof_runs)}}
        )
    except ResolutionMismatchError as e:
        raise FreezeError(str(e)) from e
    loaded = [v3s.load_oof_run(Path(p)) for p in oof_runs]
    c3.check_fold_coverage([(f, i) for f, i, _, _ in loaded], folds, ids500, [0, 1, 2])
    ft_preds: dict[str, Any] = {}
    ft_traces: dict[str, Any] = {}
    for _, _ids, p, t in loaded:
        ft_preds.update(p)
        ft_traces.update(t)
    ids = sorted(ft_preds)
    zs_preds, zs_traces = v3s.load_zs_restricted(Path(zs_run_dir), ids)
    gold, meta = v1cal.load_corpus(ids500)
    doc_fold = {d: int(folds["doc_fold"][d]) for d in ids500}
    groups = {d: meta[d]["supplier_group"] for d in ids500}
    labels = meta_mod.load_labels("train") + meta_mod.load_labels("dev")
    shapes_fn, _eq = v2s.fold_shape_fn(labels, doc_fold, groups)
    ocr = ev.load_ocr_pages(list(ids), ocr_cache)
    arms: dict[str, Any] = {}
    for name, pr_, tr_ in (("ft", ft_preds, ft_traces), ("zs", zs_preds, zs_traces)):
        tr_list = [tr_[d] for d in ids]
        n_lp = sum(1 for t in tr_list if any(p.get("field_logprobs") for p in t["pages"]))
        if n_lp < len(ids):
            raise FreezeError(f"{name}: only {n_lp}/{len(ids)} documents carry field_logprobs")
        _check_rules_off(v3s, tr_list, {d: pr_[d] for d in ids})
        arms[name] = v2s.post_rule_predictions(tr_list, shapes_fn, ocr_cache)
    ft_post, ft_changes, _ = arms["ft"]
    zs_post, _zc, _ = arms["zs"]
    tables = flags.build_tables("ft", ids, ft_post, ft_traces, ocr, ft_changes, zs_post)
    lab = {d: cf.label_doc(ft_post[d], gold[d]) for d in ids}
    y, _ = cf.label_arrays(tables.tv.keys, lab)
    sc = ev.load_scorer()
    y_doc = np.array([c2.doc_exact(sc, ft_post[d], gold[d]) for d in ids])
    cal_json = json.loads((cal_dir / "calibration_v3.json").read_text(encoding="utf-8"))
    view = next(v for v in cal_json["views"] if v["view"] == "judge_ft")
    if cal_json.get("mode") != "cross_fitted_3_fold":
        raise FreezeError("calibration_v3.json is not the 3-fold cross-fitted run")
    oof = read_oof_csv(cal_dir / "oof_fields_v3.csv", "p_v2_agree", view="judge_ft")
    p = align_oof(tables.tv, y, oof)
    nested = _nested_field_records(view.get("tau", []), {"variant": "v2_agree"})
    return Inputs(
        "ft", f"{'+'.join(Path(p).name for p in oof_runs)} + {cal_dir.name}", tables.tv, y, p,
        y_doc, ids, {d: doc_fold[d] for d in ids}, {d: groups[d] for d in ids}, tables.sigs,
        view["structure"], view["kind"], nested, tables.agree,
        {
            "kind": "calibrate_v3",
            "oof_runs": [Path(p).name for p in oof_runs],
            "zs_run_dir": Path(zs_run_dir).name,
            "calibration_dir": cal_dir.name,
            "calibration_json_sha256": _sha_file(cal_dir / "calibration_v3.json"),
            "oof_fields_sha256": _sha_file(cal_dir / "oof_fields_v3.csv"),
            "view": "judge_ft",
            "variant": "v2_agree",
        },
    )  # fmt: skip


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def run(a: argparse.Namespace) -> int:
    """Build the inputs of the chosen arm, fit, write, then re-load the artifact as a check."""
    try:
        if a.arm == "zs":
            if not a.run_dir:
                raise FreezeError("--arm zs needs --run-dir (the 02 zero-shot run folder)")
            inp = build_zs_inputs(a.run_dir, a.calibration_dir, a.ocr_cache)
        else:
            if not a.oof_run or not a.zs_run_dir:
                raise FreezeError("--arm ft needs three --oof-run and --zs-run-dir")
            inp = build_ft_inputs(a.oof_run, a.zs_run_dir, a.calibration_dir, a.ocr_cache)
        art, report = fit_artifact(inp, a.n_boot)
    except (FreezeError, c2.NoTestDataError, c3.OofCoverageError) as exc:
        print(f"FAIL (closed): {exc}", file=sys.stderr)
        return 2
    # the FT arm's runs share one resolution (build_ft_inputs asserted it): the 02 run stands in
    run_hash = recorded_run_config_hash(a.run_dir if a.arm == "zs" else a.zs_run_dir)
    if run_hash:
        art["run_config_hash"] = run_hash  # additive: flags.run_stage refuses another config hash
    print(f"run_config_hash: {run_hash or 'NOT RECORDED (the run manifest has no config hash)'}")
    out = a.out or paths.REPO_ROOT / "meta" / f"calibrator_{a.arm}.json"
    sha = write_artifact(art, out)
    cal = flags.load_calibrator(out, a.arm)  # the artifact must load and pass its own probe
    assert cal.sha256 == sha
    size = Path(out).stat().st_size
    print(
        f"froze {a.arm} calibrator: structure {inp.structure}, kind {inp.kind}, "
        f"{report['n_models']} field model(s), {art['source']['n_emitted_fields']} emitted fields, "
        f"{art['source']['n_docs']} docs; {size} bytes, sha256 {sha}"
    )
    print(f"tau field @{flags.PRIMARY_TARGET}: {json.dumps(report['tau_field_primary'])}")
    print(f"tau doc: {json.dumps(report['tau_doc'])}; wrote {out}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", choices=flags.ARMS, required=True)
    ap.add_argument("--run-dir", type=Path, default=None, help="zs: the 02 zero-shot run folder")
    ap.add_argument("--oof-run", type=Path, action="append", default=[], help="ft: x3")
    ap.add_argument("--zs-run-dir", type=Path, default=None, help="ft: the 02 run folder")
    ap.add_argument("--calibration-dir", type=Path, required=True)
    ap.add_argument("--ocr-cache", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None, help="default: meta/calibrator_<arm>.json")
    ap.add_argument("--n-boot", type=int, default=c2.N_BOOT)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
