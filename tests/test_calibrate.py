"""Tests for scripts/calibrate.py (fail-closed run check, dry-run labelling, fold fallback)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("calibrate", ROOT / "scripts" / "calibrate.py")
cal = importlib.util.module_from_spec(_spec)
sys.modules["calibrate"] = cal  # dataclasses / typing resolve annotations via sys.modules
_spec.loader.exec_module(cal)


def _run(tmp_path: Path, pred_ids: list[str], trace_ids: list[str]) -> Path:
    (tmp_path / "predictions.json").write_text(json.dumps({d: {} for d in pred_ids}), "utf-8")
    lines = [json.dumps({"doc_id": d, "pages": []}) for d in trace_ids]
    (tmp_path / "trace.jsonl").write_text("\n".join(lines) + "\n", "utf-8")
    return tmp_path


def test_load_run_accepts_exactly_the_expected_docs(tmp_path: Path) -> None:
    run = _run(tmp_path, ["a", "b"], ["b", "a"])
    preds, traces = cal.load_run(run, ["a", "b"])
    assert set(preds) == set(traces) == {"a", "b"}


@pytest.mark.parametrize(
    ("pred_ids", "trace_ids"),
    [(["a"], ["a", "b"]), (["a", "b"], ["a"]), (["a", "b", "x"], ["a", "b"]), ([], [])],
)
def test_load_run_fails_closed_on_a_partial_or_foreign_run(
    tmp_path: Path, pred_ids: list[str], trace_ids: list[str]
) -> None:
    run = _run(tmp_path, pred_ids, trace_ids)
    with pytest.raises(cal.RunNotFullError, match="refusing"):
        cal.load_run(run, ["a", "b"])


def test_cli_refuses_a_run_that_is_not_the_full_500(tmp_path: Path, capsys: Any) -> None:
    run = _run(tmp_path, ["train_0000", "train_0001"], ["train_0000", "train_0001"])
    assert cal.main(["--run-dir", str(run), "--out-dir", str(tmp_path / "out")]) == 2
    assert "FAIL (closed)" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()  # nothing was written


def test_restricted_folds_never_splits_a_supplier_group() -> None:
    ids = [f"d{i}" for i in range(6)]
    meta = {d: {"supplier_group": f"g{i // 2}"} for i, d in enumerate(ids)}
    one_fold = {"doc_fold": dict.fromkeys(ids, 0)}  # a subset that kept a single fold
    out = cal.restricted_folds(ids, one_fold, meta)
    assert len(set(out.values())) >= 2
    by_group: dict[str, set[int]] = {}
    for d, f in out.items():
        by_group.setdefault(meta[d]["supplier_group"], set()).add(f)
    assert all(len(v) == 1 for v in by_group.values())
    kept = {"doc_fold": {d: i % 3 for i, d in enumerate(ids)}}
    assert cal.restricted_folds(ids, kept, meta) == kept["doc_fold"]


def test_synthetic_dry_run_is_labelled_everywhere(tmp_path: Path) -> None:
    res = cal.run_synthetic(n_boot=30)
    title = "Calibration dry run (DRY RUN, not results)"
    md = cal.render_md({"(i) synthetic": dict(res)}, title, cal.DRY_BANNER)
    assert md.count("DRY RUN") >= 3 and "syn_0000" not in md  # aggregates only, no doc ids
    cal.write_outputs(res, tmp_path, "DRYRUN_synthetic_", cal.DRY_BANNER)
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files and all(f.startswith("DRYRUN_") for f in files)
    data = json.loads((tmp_path / "DRYRUN_synthetic_calibration.json").read_text("utf-8"))
    assert "DRY RUN" in data["label"]
    assert data["model_selection"]["p_correct"]["chosen"] in ("lr", "gbm")
    assert "_arrays" not in data
