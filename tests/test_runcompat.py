"""shipdoc.runcompat: refuse mixed-resolution (1260-token vs native) run comparisons."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from shipdoc import runcompat as rc
from shipdoc.spike import load_config

ROOT = Path(__file__).resolve().parents[1]
LEGACY_YAML = ROOT / "configs" / "spike_qwen35_4b_img_only.yaml"
NATIVE_YAML = ROOT / "configs" / "spike_qwen35_4b_img_only_native.yaml"


def man(name: str, h: str | None) -> dict[str, Any]:
    return {"config": {"name": name, "hash": h}} if h is not None else {}


def write_run(tmp: Path, folder: str, manifest: dict[str, Any]) -> Path:
    d = tmp / folder
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return d


def test_real_native_and_1260_configs_hash_differently_and_resolve_max_pixels() -> None:
    legacy, native = load_config(LEGACY_YAML), load_config(NATIVE_YAML)
    assert legacy.config_hash != native.config_hash
    assert (legacy.backend.max_pixels, native.backend.max_pixels) == (1310720, 2196480)
    m_legacy = man(legacy.name, legacy.config_hash)
    m_native = man(native.name, native.config_hash)
    assert rc.resolve_max_pixels(m_legacy) == 1310720
    assert rc.resolve_max_pixels(m_native) == 2196480
    with pytest.raises(rc.ResolutionMismatchError, match="1310720.*2196480"):
        rc.assert_same_resolution(m_legacy, m_native, "FT", "ZS")


def test_same_hash_passes_and_different_or_missing_hash_is_refused() -> None:
    rc.assert_same_resolution(man("a", "h1"), man("a", "h1"))
    for a, b in ((man("a", "h1"), man("b", "h2")), (man("a", "h1"), {}), ({}, {})):
        with pytest.raises(rc.ResolutionMismatchError, match="mixed-resolution"):
            rc.assert_same_resolution(a, b, "run X", "run Y")


def test_message_names_both_runs_and_an_unknown_config_is_unresolved() -> None:
    with pytest.raises(rc.ResolutionMismatchError) as e:
        rc.assert_same_resolution(man("cfgA", "h1"), man("cfgB", "h2"), "FT run `f`", "ZS run `z`")
    msg = str(e.value)
    assert "FT run `f`" in msg and "ZS run `z`" in msg and "cfgA" in msg and "cfgB" in msg
    assert "unresolved" in msg


def test_a_manifest_that_records_max_pixels_is_used_directly() -> None:
    assert rc.resolve_max_pixels({"config": {"name": "n", "hash": "h", "max_pixels": 5}}) == 5
    assert rc.resolve_max_pixels({"config": {"name": "n", "hash": "h", "max_pixels": True}}) is None
    assert rc.resolve_max_pixels(None) is None


def test_runs_share_resolution_folders(tmp_path: Path) -> None:
    a = write_run(tmp_path, "zs", man("n", "h1"))
    b = write_run(tmp_path, "ft0", man("n", "h1"))
    c = write_run(tmp_path, "ft1", man("n", "h2"))
    nomanifest = tmp_path / "bare"
    nomanifest.mkdir()
    rc.assert_runs_share_resolution({"ZS": a, "FT 0": b})
    rc.assert_runs_share_resolution({"ZS": a})  # nothing to compare
    rc.assert_runs_share_resolution({"ZS": nomanifest, "same": nomanifest})  # plumbing: one folder
    with pytest.raises(rc.ResolutionMismatchError, match="FT 1 `ft1`"):
        rc.assert_runs_share_resolution({"ZS": a, "FT 0": b, "FT 1": c})
    with pytest.raises(rc.ResolutionMismatchError, match="mixed-resolution"):
        rc.assert_runs_share_resolution({"ZS": a, "missing": tmp_path / "nope"})


def test_calibrate_v3_refuses_a_mixed_resolution_pair(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = ROOT / "scripts" / "calibrate_v3.py"
    spec = importlib.util.spec_from_file_location("calibrate_v3", path)
    assert spec is not None and spec.loader is not None
    cal3 = importlib.util.module_from_spec(spec)
    sys.modules["calibrate_v3"] = cal3
    spec.loader.exec_module(cal3)

    def run(folder: str, h: str, oof: bool) -> Path:
        m = {**man("n", h), **({"oof": {"fold": 0}} if oof else {})}
        d = write_run(tmp_path, folder, m)
        (d / "progress.json").write_text('{"status": "complete"}', encoding="utf-8")
        (d / "trace.jsonl").write_text('{"doc_id": "train_0001"}\n', encoding="utf-8")
        (d / "predictions.json").write_text('{"train_0001": {}}', encoding="utf-8")
        return d

    ft, zs = run("oof_fold0_x", "hn", True), run("zs_run", "h1", False)
    rc_code = cal3.main(["--folds", "0", "--oof-run", str(ft), "--zs-run-dir", str(zs)])
    err = capsys.readouterr().err
    assert rc_code == 2 and "FAIL (closed)" in err and "mixed-resolution" in err
    assert "FT run 0" in err and "ZS run" in err
