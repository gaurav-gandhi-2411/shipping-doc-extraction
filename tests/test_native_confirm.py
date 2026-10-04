"""scripts/native_confirm.py: the deliberate native-vs-1260 comparison (no run folders needed)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from shipdoc import runcompat as rc
from shipdoc.spike import load_config

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "native_confirm", ROOT / "scripts" / "native_confirm.py"
)
nc = importlib.util.module_from_spec(spec)
sys.modules["native_confirm"] = nc  # dataclasses resolve annotations through sys.modules
spec.loader.exec_module(nc)

LEGACY_YAML = ROOT / "configs" / "spike_qwen35_4b_img_only.yaml"
NATIVE_YAML = ROOT / "configs" / "spike_qwen35_4b_img_only_native.yaml"


def manifest(cfg_path: Path, **over: Any) -> dict[str, Any]:
    c = load_config(cfg_path)
    m = {
        "config": {"name": c.name, "hash": c.config_hash},
        "model": {"id": "m", "revision": "r"},
        "seed": 42,
        "docs_sha": "abc",
    }
    return m | over


def test_cross_resolution_accepts_exactly_the_native_vs_1260_pair() -> None:
    nc.assert_cross_resolution(manifest(NATIVE_YAML), manifest(LEGACY_YAML))


def test_cross_resolution_refuses_same_hash_swapped_and_non_resolution_differences() -> None:
    nat, leg = manifest(NATIVE_YAML), manifest(LEGACY_YAML)
    with pytest.raises(ValueError, match="DIFFERENT resolutions"):
        nc.assert_cross_resolution(nat, nat)  # equal hashes: that is runcompat's job, not ours
    with pytest.raises(ValueError, match="max_pixels"):
        nc.assert_cross_resolution(leg, nat)  # swapped
    with pytest.raises(ValueError, match="config hash"):
        nc.assert_cross_resolution({}, leg)
    for key, val in (("docs_sha", "other"), ("seed", 7), ("model", {"id": "x", "revision": "y"})):
        with pytest.raises(ValueError, match=key):
            nc.assert_cross_resolution(manifest(NATIVE_YAML, **{key: val}), leg)


def test_cross_resolution_does_not_weaken_the_default_guard() -> None:
    with pytest.raises(rc.ResolutionMismatchError):
        rc.assert_same_resolution(manifest(NATIVE_YAML), manifest(LEGACY_YAML))


def test_slices_of_partitions_the_documents() -> None:
    gold = {
        "train_1": {"doc_type": "invoice"},
        "train_2": {"doc_type": "waybill"},
        "dev_1": {"doc_type": "invoice"},
    }
    meta = [
        {"doc_id": "train_1", "scanned": True, "multipage": True},
        {"doc_id": "train_2", "scanned": False, "multipage": False},
        {"doc_id": "dev_1", "scanned": False, "multipage": True},
    ]
    s = nc.slices_of(gold, meta)
    assert list(s["all"]) == ["train_1", "train_2", "dev_1"]  # gold order is kept (bootstrap order)
    assert s["scanned"] == ["train_1"] and s["digital"] == ["train_2", "dev_1"]
    assert s["multipage"] == ["train_1", "dev_1"] and s["single-page"] == ["train_2"]
    assert s["waybill"] == ["train_2"] and s["dev"] == ["dev_1"] and len(s["train"]) == 2


def test_merge_block_appends_then_replaces_without_touching_the_text_above() -> None:
    old = "# T\n\ntext above\n"
    once = nc.merge_block(old, "BLOCK1")
    assert once.startswith(old.rstrip("\n")) and f"{nc.BEGIN}\nBLOCK1\n{nc.END}\n" in once
    twice = nc.merge_block(once, "BLOCK2")
    assert twice.count(nc.BEGIN) == 1 and "BLOCK2" in twice and "BLOCK1" not in twice
    assert twice.startswith(old.rstrip("\n"))
    assert nc.merge_block(twice, "BLOCK2") == twice  # idempotent


def test_formatting_helpers() -> None:
    assert nc.pts(0.0729) == "+7.29" and nc.pts(-0.0012) == "-0.12"
    assert nc.fmt_ci({"delta": 0.05, "lo": 0.01, "hi": 0.09}) == "+5.00 [+1.00, +9.00]"
    assert nc.count_row(
        "x", {"control": 26.0, "candidate": 12.0, "delta": -14.0, "lo": -24.0, "hi": -5.0}
    ) == [
        "x",
        "26 -> 12",
        "-14 [-24, -5]",
    ]
