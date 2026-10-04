"""``run_config_hash`` of a frozen calibrator: recorded by the freezer, enforced by the flags stage.

The gap this closes (commit c6d8a37): the frozen 1260-token calibrator recorded no config hash, so
``shipdoc.flags`` could not refuse it on a native submission. New artifacts carry the hash of the
zero-shot run they were frozen from; ``flags.run_stage`` refuses a submission whose run used
another hash and still accepts an artifact without the field (the committed
``meta/calibrator_zs.json``, which is not modified). Synthetic corpus and calibrators only.
"""

from __future__ import annotations

# ruff: noqa: F811  (the `world` fixture is imported, then requested by name, as pytest requires)
import argparse
import json
from pathlib import Path
from typing import Any

import pytest
from test_flags import calibrator_for, freeze, synthetic_inputs, world_submission
from test_predict import (  # noqa: F401 - `_clean_sha` is an autouse fixture
    TEST_IDS,
    World,
    _clean_sha,
    needs_schema,
    world,
)

from shipdoc import flags

ROOT = Path(__file__).resolve().parents[1]


def run_folder(tmp: Path, name: str, manifest: dict[str, Any] | None) -> Path:
    d = tmp / name
    d.mkdir()
    if manifest is not None:
        (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return d


# -------------------- the freezer


def test_the_recorded_hash_is_the_run_manifests_or_none(tmp_path: Path) -> None:
    ok = run_folder(tmp_path, "ok", {"config": {"name": "c", "hash": "e4b84ec2809625d5"}})
    assert freeze.recorded_run_config_hash(ok) == "e4b84ec2809625d5"
    assert freeze.recorded_run_config_hash(run_folder(tmp_path, "nohash", {"config": {}})) is None
    assert freeze.recorded_run_config_hash(run_folder(tmp_path, "noman", None)) is None
    assert freeze.recorded_run_config_hash(tmp_path / "nowhere") is None
    assert freeze.recorded_run_config_hash(None) is None


def freeze_with(tmp: Path, monkeypatch: pytest.MonkeyPatch, run_dir: Path, out: Path) -> int:
    inp = synthetic_inputs("zs", n_docs=45)
    monkeypatch.setattr(freeze, "build_zs_inputs", lambda *a, **k: inp)
    args = argparse.Namespace(arm="zs", run_dir=run_dir, oof_run=[], zs_run_dir=None,
                              calibration_dir=tmp, ocr_cache=None, out=out, n_boot=10)  # fmt: skip
    return int(freeze.run(args))


def test_freezing_records_the_run_config_hash_additively(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run = run_folder(tmp_path, "zs", {"config": {"name": "c", "hash": "e4b84ec2809625d5"}})
    out = tmp_path / "cal.json"
    assert freeze_with(tmp_path, monkeypatch, run, out) == 0
    art = json.loads(out.read_text())
    assert art["run_config_hash"] == "e4b84ec2809625d5"
    assert "run_config_hash: e4b84ec2809625d5" in capsys.readouterr().out
    cal = flags.load_calibrator(out, "zs")  # still loads and passes its parity probe
    assert cal.run_config_hash == "e4b84ec2809625d5"
    # a run without a hash records nothing (the artifact stays what it was before this change)
    bare = run_folder(tmp_path, "bare", {"config": {}})
    out2 = tmp_path / "cal2.json"
    assert freeze_with(tmp_path, monkeypatch, bare, out2) == 0
    art2 = json.loads(out2.read_text())
    assert "run_config_hash" not in art2
    assert "NOT RECORDED" in capsys.readouterr().out
    assert flags.load_calibrator(out2, "zs").run_config_hash is None
    assert sorted(set(art) - set(art2)) == ["run_config_hash"]  # the ONLY difference


def test_the_committed_1260_calibrator_is_untouched_and_hashless() -> None:
    data = json.loads((ROOT / "meta" / "calibrator_zs.json").read_text(encoding="utf-8"))
    assert "run_config_hash" not in data and data["arm"] == "zs"


# -------------------- the Calibrator property


def test_run_config_hash_property_ignores_malformed_values() -> None:
    def cal(**d: Any) -> flags.Calibrator:
        return flags.Calibrator({"arm": "zs", **d}, "0" * 64, {}, None, None)  # type: ignore[arg-type]

    assert cal(run_config_hash="abc").run_config_hash == "abc"
    for bad in ({}, {"run_config_hash": ""}, {"run_config_hash": 7}, {"run_config_hash": None}):
        assert cal(**bad).run_config_hash is None


# -------------------- the flags stage


def with_hash(cal_path: Path, h: str | None, tmp: Path) -> Path:
    data = json.loads(cal_path.read_text())
    if h is None:
        data.pop("run_config_hash", None)
    else:
        data["run_config_hash"] = h
    out = tmp / f"cal_{h}.json"
    freeze.write_artifact(data, out)
    return out


@needs_schema
def test_the_flags_stage_refuses_a_calibrator_frozen_for_another_config(
    world: World, tmp_path: Path
) -> None:
    sub, ocr_root, shapes = world_submission(world, tmp_path)
    base = calibrator_for(sub, ocr_root, "zs", tmp_path)
    kw: dict[str, Any] = {
        "submission_dir": sub, "ocr_cache": ocr_root, "shapes_file": shapes,
        "expect_docs": len(TEST_IDS), "say": lambda m: None,
    }  # fmt: skip
    run_hash = json.loads((sub / "manifest.json").read_text())["config"]["hash"]
    assert run_hash == world.cfg.config_hash
    # the same hash: accepted; no hash recorded (older artifact): accepted, as before
    for h in (run_hash, None):
        doc = flags.run_stage(calibrator=with_hash(base, h, tmp_path), **kw)
        assert sorted(doc["docs"]) == sorted(TEST_IDS)
    # another config / resolution: refused with both hashes in the message
    other = with_hash(base, "e4b84ec2809625d5", tmp_path)
    with pytest.raises(
        flags.FlagsError, match=rf"frozen for config hash e4b84ec2809625d5.*{run_hash}"
    ):
        flags.run_stage(calibrator=other, **kw)
    assert flags.main([
        "--submission-dir", str(sub), "--calibrator", str(other), "--ocr-cache", str(ocr_root),
        "--shapes-file", str(shapes), "--expect-docs", str(len(TEST_IDS)),
    ]) == 1  # fmt: skip
    # a submission whose manifest records no hash cannot satisfy a calibrator that has one
    man = json.loads((sub / "manifest.json").read_text())
    man["config"].pop("hash")
    (sub / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(flags.FlagsError, match="frozen for config hash"):
        flags.run_stage(calibrator=with_hash(base, run_hash, tmp_path), **kw)
