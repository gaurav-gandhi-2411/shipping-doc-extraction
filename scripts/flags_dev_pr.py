"""Review-flag precision and recall on the 100 dev documents (CPU only, no model, no network).

The brief asks for "a per-field confidence or needs-human-review flag, with its precision and recall
on dev". This script computes exactly that for the submitted system (v1.5 = native zero-shot +
rules R1-R3) at the PRODUCTION operating point: field target 98% precision, one tau per field type,
document target 98% (``shipdoc.flags.PRIMARY_TARGET``).

Inputs are the numeric outputs of ``scripts/calibrate_v2.py`` on the native run
(``oof_fields_v2.csv``:
one P(correct) per emitted field, cross-fitted by the 3 supplier folds, plus the label ``y_correct``
from the unmodified scorer; ``oof_docs_v2.csv``: the same per document) and the frozen calibrator
``meta/calibrator_zs_native.json``. Nothing is refitted here.

Three operating schemes, none of which lets a document choose its own threshold:

* ``dev100_nested`` (PRIMARY, labelled "dev100, cross-fitted"): the dev docs' OUT-OF-FOLD
  probabilities; for fold k the tau of each field type is chosen on the OTHER two folds' OOF rows
  (``confidence.select_tau``, lowest tau with precision >= 98%) and applied to fold k
  (``confidence_v2.nested_accept``, the protocol of calibrate_v2 section (b)); the decisions of the
  100 dev docs are then scored.
* ``dev100_frozen`` (REFERENCE, optimistic): the same dev OOF probabilities against the frozen
  production taus. Those taus were chosen on the pooled OOF rows of all 500 docs, which include
  these 100, so they are in-sample for dev; the production calibrator itself was also fitted on
  all 500 docs, so scoring its own probabilities on dev would be in-sample twice (not done).
* ``all500_nested`` (SECONDARY): the same nested protocol pooled over all 500 docs.

Metrics per field type and overall (each type with its own tau): auto-accept precision (accepted
fields that are correct), error recall (wrong emitted fields that are sent to review), review rate
(emitted fields sent to review) and accepted share, with doc-level percentile bootstrap 95% CIs
(2000 resamples, seed 42; documents resampled whole, ``confidence.review_with_ci``). The CIs
condition on the thresholds (tau-selection variance is not resampled). Population = EMITTED
(non-null) fields: a null prediction carries no probability and is never flagged; the wrong cells
of non-emitted fields (gold has a value, the prediction is null: over-nulls) are counted and
disclosed separately, as are gold rows that no predicted row matches (not in the table at all).

Document level: the document tau (98%) of the frozen calibrator and the nested per-fold document
tau applied to the dev documents' OOF document probabilities; counts of accepted documents.

Aggregates and counts only: no gold or predicted value, no document id. All numbers UNVERIFIED
until a second path recomputes them.

Run: ``uv run python scripts/flags_dev_pr.py --calibration-dir $SHIPDOC_TMP_DIR/calibration_native
--out reports/flags_dev100_native.md --json-out $SHIPDOC_TMP_DIR/report_out/flags_dev100_native.json``
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from shipdoc import confidence_v2 as c2

ROOT = Path(__file__).resolve().parents[1]
TARGET = 0.98  # shipdoc.flags.PRIMARY_TARGET: the field and the document precision target
SEED = c2.SEED
N_BOOT = c2.N_BOOT
TYPES: tuple[str, ...] = c2.FIELD_TYPES
ALL = "all fields (per-type tau)"
SCHEMES: tuple[tuple[str, str], ...] = (
    ("dev100_nested", "dev100, cross-fitted (nested supplier-fold tau): PRIMARY"),
    ("dev100_frozen", "dev100, OOF probabilities vs the frozen production tau (tau in-sample)"),
    ("all500_nested", "pooled 500 docs, nested: SECONDARY"),
)
METRICS = ("precision_accepted", "error_recall", "review_rate", "accepted_share")


@dataclass
class OofFields:
    """Columns of ``oof_fields_v2.csv`` as arrays (no value, no text)."""

    doc_ids: np.ndarray
    types: np.ndarray
    emitted: np.ndarray
    p: np.ndarray
    correct: np.ndarray


def field_type(scope: str, field: str) -> str:
    """``header`` for every header field, else the row field name (the calibrator's field type)."""
    return c2.HEADER if scope == "header" else field


def load_oof_fields(path: Path) -> OofFields:
    """Read ``oof_fields_v2.csv``; ``p`` is NaN for a non-emitted field."""
    ids: list[str] = []
    types: list[str] = []
    em: list[bool] = []
    p: list[float] = []
    ok: list[bool] = []
    with path.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            ids.append(r["doc_id"])
            types.append(field_type(r["scope"], r["field"]))
            em.append(r["emitted"] == "1")
            p.append(float(r["p_correct_v2"]) if r["p_correct_v2"] not in ("", "nan") else np.nan)
            ok.append(r["y_correct"] == "1")
    return OofFields(
        np.array(ids, dtype=object),
        np.array(types, dtype=object),
        np.array(em, dtype=bool),
        np.array(p, dtype=float),
        np.array(ok, dtype=bool),
    )


def load_doc_table(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """``(doc ids, P(fully correct), fully correct)`` of ``oof_docs_v2.csv``."""
    ids: list[str] = []
    p: list[float] = []
    y: list[bool] = []
    with path.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            ids.append(r["doc_id"])
            p.append(float(r["p_fully_correct"]))
            y.append(r["y_fully_correct"] == "1")
    return ids, np.array(p, dtype=float), np.array(y, dtype=bool)


def type_decisions(
    f: OofFields,
    fold_of: np.ndarray,
    target: float,
    frozen_tau: Mapping[str, float | None] | None,
) -> dict[str, np.ndarray]:
    """Accept decision (bool per CSV row) of every field type.

    ``frozen_tau`` None: nested (fold k's tau chosen on the other folds); else accept iff
    ``p >= tau`` of the type (None tau accepts nothing). Non-emitted rows are never accepted.
    """
    out: dict[str, np.ndarray] = {}
    for t in TYPES:
        mask = f.emitted & (f.types == t)
        if frozen_tau is None:
            acc, _, _ = c2.nested_accept(f.p, f.correct, fold_of, mask, target)
        else:
            tau = frozen_tau.get(t)
            acc = np.zeros(len(f.p), dtype=bool)
            if tau is not None:
                acc[mask] = f.p[mask] >= tau
        out[t] = acc
    return out


def summarize(
    accept: np.ndarray, f: OofFields, rows: np.ndarray, n_boot: int = N_BOOT
) -> dict[str, Any]:
    """Counts and bootstrap CIs of the accept decisions over the emitted rows selected by `rows`."""
    sel = rows & f.emitted
    a, y, ids = accept[sel], f.correct[sel], f.doc_ids[sel]
    n_wrong = int((~y).sum())
    out: dict[str, Any] = {
        "emitted": int(sel.sum()),
        "accepted": int(a.sum()),
        "accepted_correct": int((a & y).sum()),
        "wrong": n_wrong,
        "wrong_sent_to_review": int((~a & ~y).sum()),
        "docs": len(set(ids.tolist())),
    }
    if sel.sum() == 0:
        return out | {m: None for m in METRICS}
    m = c2.accept_metrics(a, y, ids, n_boot)
    return out | {k: m[k] for k in METRICS}


def evaluate(
    f: OofFields,
    fold_of: np.ndarray,
    dev: set[str],
    frozen_tau: Mapping[str, float | None],
    n_boot: int = N_BOOT,
    target: float = TARGET,
) -> dict[str, dict[str, dict[str, Any]]]:
    """``{scheme: {population: summary}}`` for the three schemes (see the module docstring)."""
    is_dev = np.fromiter((str(d) in dev for d in f.doc_ids), dtype=bool, count=len(f.doc_ids))
    everyone = np.ones(len(f.doc_ids), dtype=bool)
    nested = type_decisions(f, fold_of, target, None)
    frozen = type_decisions(f, fold_of, target, frozen_tau)
    res: dict[str, dict[str, dict[str, Any]]] = {}
    for scheme, dec, rows in (
        ("dev100_nested", nested, is_dev),
        ("dev100_frozen", frozen, is_dev),
        ("all500_nested", nested, everyone),
    ):
        pops: dict[str, dict[str, Any]] = {}
        comb = np.zeros(len(f.p), dtype=bool)
        for t in TYPES:
            pops[t] = summarize(dec[t], f, rows & (f.types == t), n_boot)
            comb |= dec[t]
        pops[ALL] = summarize(comb, f, rows, n_boot)
        res[scheme] = pops
    return res


def over_null_cells(f: OofFields, dev: set[str]) -> dict[str, dict[str, int]]:
    """Wrong NON-emitted cells (gold has a value, prediction null) per type, dev and all docs."""
    is_dev = np.fromiter((str(d) in dev for d in f.doc_ids), dtype=bool, count=len(f.doc_ids))
    wrong_null = ~f.emitted & ~f.correct
    return {
        t: {
            "dev": int((wrong_null & is_dev & (f.types == t)).sum()),
            "all": int((wrong_null & (f.types == t)).sum()),
        }
        for t in TYPES
    }


def doc_level(
    ids: Sequence[str],
    p: np.ndarray,
    y: np.ndarray,
    fold_of: np.ndarray,
    dev: set[str],
    frozen_doc_tau: float | None,
    target: float = TARGET,
) -> dict[str, Any]:
    """Accepted documents (count, correct) on dev: nested per-fold tau and the frozen tau."""
    everyone = np.ones(len(ids), dtype=bool)
    acc_n, taus, _ = c2.nested_accept(p, y, fold_of, everyone, target)
    is_dev = np.array([d in dev for d in ids])
    acc_f = np.zeros(len(ids), dtype=bool) if frozen_doc_tau is None else p >= frozen_doc_tau
    out: dict[str, Any] = {
        "dev_docs": int(is_dev.sum()),
        "dev_fully_correct": int((is_dev & y).sum()),
        "all_docs": len(ids),
        "all_fully_correct": int(y.sum()),
        "frozen_tau_is_none": frozen_doc_tau is None,
        "nested_taus_none": sum(t is None for t in taus.values()),
    }
    for name, acc in (("nested", acc_n), ("frozen", acc_f)):
        for scope, mask in (("dev", is_dev), ("all", everyone)):
            m = mask & acc
            out[f"{name}_{scope}"] = {
                "accepted": int(m.sum()),
                "accepted_correct": int((m & y).sum()),
            }
    return out


def crosscheck_calibrate_v2(res: Mapping[str, Any], csv_path: Path) -> dict[str, Any]:
    """Max absolute difference to ``tau_field_types.csv`` (calibrate_v2, nested, 98%, dev / all).

    Compares precision, coverage (accepted share), review rate and error recall, point and CI,
    for the five field types and the combined population. NaN pairs (nothing accepted) match.
    """
    keymap = {"dev": "dev100_nested", "all": "all500_nested"}
    cols = {
        "precision_accepted": "precision_accepted",
        "accepted_share": "accepted_share",
        "review_rate": "review_rate",
        "error_recall": "error_recall",
    }
    worst = 0.0
    n = 0
    with csv_path.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["scheme"] != "nested" or float(r["target"]) != TARGET or r["slice"] not in keymap:
                continue
            pop = ALL if r["population"] == "all_per_type_tau" else r["population"]
            if pop not in res[keymap[r["slice"]]]:
                continue
            mine = res[keymap[r["slice"]]][pop]
            for k, col in cols.items():
                for suffix, key in (("", "point"), ("_lo", "lo"), ("_hi", "hi")):
                    theirs = float(r[col + suffix])
                    ours = float("nan") if mine[k] is None else float(mine[k][key])
                    if np.isnan(theirs) and np.isnan(ours):
                        continue
                    worst = max(worst, abs(theirs - ours) if not np.isnan(ours) else 1.0)
                    n += 1
    return {"max_abs_diff": worst, "n_compared": n}


def _ci(d: Mapping[str, float] | None) -> str:
    if d is None or np.isnan(d["point"]):
        return "n/a"
    return f"{100 * d['point']:.1f} [{100 * d['lo']:.1f}, {100 * d['hi']:.1f}]"


def table(pops: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Markdown rows of one scheme."""
    L = [
        "| field type | emitted fields | auto-accepted | auto-accept precision % [95% CI] | "
        "error recall % [95% CI] | review rate % [95% CI] | wrong fields |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in (*TYPES, ALL):
        s = pops[name]
        prec = _ci(s["precision_accepted"])
        if s["accepted"] == 0:
            prec = "n/a (none accepted)"
        share = f"{100 * s['accepted'] / s['emitted']:.1f}%" if s["emitted"] else "n/a"
        L.append(
            f"| {name} | {s['emitted']} | {s['accepted']} ({share}) | {prec} | "
            f"{_ci(s['error_recall'])} | {_ci(s['review_rate'])} | {s['wrong']} |"
        )
    return L


def render(res: Mapping[str, Any], cmd: str, state: str) -> str:
    """Markdown report (aggregates only)."""
    d = res["doc_level"]
    L = [
        "# Review flags on dev100: auto-accept precision and error recall (native ZS + rules)",
        "",
        f"**Provenance.** Command `{cmd}` (repo state `{state}`; doc-level percentile bootstrap "
        f"{N_BOOT} resamples, seed {SEED}). Inputs: `oof_fields_v2.csv` (sha256 "
        f"`{res['oof_fields_sha256'][:16]}`, equals the sha recorded in `meta/calibrator_zs_native"
        f".json` source: {res['oof_fields_sha_matches']}), `oof_docs_v2.csv`, frozen taus of "
        f"`meta/calibrator_zs_native.json` (sha256 `{res['calibrator_sha256'][:16]}`, run config "
        f"hash `{res['run_config_hash']}`). **All numbers UNVERIFIED** here; the report registry "
        "records the second path. Aggregates and counts only.",
        "",
        "## Operating point and protocol",
        "",
        f"Production thresholds: field target {TARGET:.2f} (one tau per field type, lowest tau "
        f"whose accepted set has >= {100 * TARGET:.0f}% precision, `confidence.select_tau`), "
        f"document target {TARGET:.2f}. The production calibrator was fitted on all 500 docs, so "
        "its probabilities on dev are in-sample. The PRIMARY table therefore uses the cross-fitted "
        "OUT-OF-FOLD probabilities of calibrate_v2 (supplier-fold cross-fit, `reports/"
        "calibration_v2_native.md`) with NESTED taus (fold k's tau chosen on the other two folds, "
        "applied to fold k): the dev docs never influence their own probabilities or thresholds. "
        "Population: EMITTED (non-null) fields. CIs are conditional on the thresholds. "
        f"Frozen production taus: {json.dumps(res['frozen_tau'], sort_keys=True)} "
        "(null = nothing of that type is auto-accepted).",
    ]
    for key, label in SCHEMES:
        L += ["", f"## {label}", "", *table(res["schemes"][key])]
    L += [
        "",
        "## Wrong cells the flag does not see",
        "",
        "Not in the tables above (the flag only judges emitted fields): over-nulls (gold has a "
        "value, the prediction is null) and gold rows that no predicted row pairs with.",
        "",
        "| field type | over-null cells, dev100 | over-null cells, 500 docs |",
        "|---|---|---|",
    ]
    for t in TYPES:
        L.append(f"| {t} | {res['over_null'][t]['dev']} | {res['over_null'][t]['all']} |")
    n, f_ = d["nested_dev"], d["frozen_dev"]
    na, fa = d["nested_all"], d["frozen_all"]
    L += [
        "",
        "## Document level (target 0.98)",
        "",
        f"Fully correct documents: {d['dev_fully_correct']} of {d['dev_docs']} on dev, "
        f"{d['all_fully_correct']} of {d['all_docs']} on all. Auto-accepted documents on dev: "
        f"nested tau {n['accepted']} accepted, {n['accepted_correct']} of them fully correct; "
        f"frozen production tau {f_['accepted']} accepted, {f_['accepted_correct']} fully "
        f"correct. On all 500 docs: nested {na['accepted']} accepted, {na['accepted_correct']} "
        f"correct; frozen {fa['accepted']}, {fa['accepted_correct']} "
        "correct. The pooled nested document precision is below the 98% target (reports/"
        "calibration_v2_native.md section (c): NOT ATTAINABLE), so the document-level flag is not "
        "a usable auto-accept; the shipped test flags accept 0 of 200 documents "
        "(`reports/v1_5_submission.md`).",
    ]
    cc = res.get("crosscheck")
    if cc:
        L += [
            "",
            "## Cross-check against calibrate_v2's own output",
            "",
            f"`tau_field_types.csv` (nested, target 0.98, slices dev and all): {cc['n_compared']} "
            f"values compared, max absolute difference {cc['max_abs_diff']:.2e}.",
        ]
    return "\n".join(L) + "\n"


def sha256_file(p: Path) -> str:
    """Hex sha256 of a file."""
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git_state() -> str:
    """Short HEAD sha, ``+dirty`` when the tracked tree differs."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=ROOT, check=True,
        ).stdout.strip()  # fmt: skip
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, cwd=ROOT, check=True,
        ).stdout.strip()  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return sha + ("+dirty" if dirty else "")


def compute(calibration_dir: Path, calibrator: Path, n_boot: int = N_BOOT) -> dict[str, Any]:
    """Everything the report prints, as one dict (aggregates only)."""
    cal = json.loads(calibrator.read_text(encoding="utf-8"))
    fields_csv = calibration_dir / "oof_fields_v2.csv"
    sha = sha256_file(fields_csv)
    f = load_oof_fields(fields_csv)
    doc_fold = {
        str(d): int(k)
        for d, k in json.loads((ROOT / "splits" / "folds.json").read_text("utf-8"))[
            "doc_fold"
        ].items()
    }
    if any(str(d).startswith("test_") for d in f.doc_ids):
        raise SystemExit("test ids in the calibration table: refusing")
    fold_of = np.array([doc_fold[str(d)] for d in f.doc_ids])
    dev = {d for d in doc_fold if d.startswith("dev_")}
    tau = cal["thresholds"]["field"]["tau"][f"{TARGET:.2f}"]
    res: dict[str, Any] = {
        "oof_fields_sha256": sha,
        "oof_fields_sha_matches": sha == cal.get("source", {}).get("oof_fields_sha256"),
        "calibrator_sha256": sha256_file(calibrator),
        "run_config_hash": cal.get("run_config_hash"),
        "frozen_tau": tau,
        "n_docs": len(set(f.doc_ids.tolist())),
        "n_dev_docs": len(dev),
        "schemes": evaluate(f, fold_of, dev, tau, n_boot),
        "over_null": over_null_cells(f, dev),
    }
    ids, p, y = load_doc_table(calibration_dir / "oof_docs_v2.csv")
    d_fold = np.array([doc_fold[d] for d in ids])
    res["doc_level"] = doc_level(
        ids, p, y, d_fold, dev, cal["thresholds"]["doc"]["tau"][f"{TARGET:.2f}"]
    )
    tau_csv = calibration_dir / "tau_field_types.csv"
    if tau_csv.is_file():
        res["crosscheck"] = crosscheck_calibrate_v2(res["schemes"], tau_csv)
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--calibration-dir", type=Path, required=True)
    ap.add_argument("--calibrator", type=Path, default=ROOT / "meta" / "calibrator_zs_native.json")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "flags_dev100_native.md")
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args(argv)
    res = compute(a.calibration_dir, a.calibrator, a.n_boot)
    cmd = "uv run python scripts/flags_dev_pr.py " + " ".join(
        argv if argv is not None else sys.argv[1:]
    )
    a.out.write_text(render(res, cmd, git_state()), encoding="utf-8", newline="\n")
    if a.json_out:
        a.json_out.parent.mkdir(parents=True, exist_ok=True)
        a.json_out.write_text(json.dumps(res, indent=1, sort_keys=True), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
