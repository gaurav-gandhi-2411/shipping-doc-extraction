"""scripts/calibration_compare_native.py on synthetic calibration_v2 dicts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "calibration_compare_native", ROOT / "scripts" / "calibration_compare_native.py"
)
cc = importlib.util.module_from_spec(spec)
sys.modules["calibration_compare_native"] = cc
spec.loader.exec_module(cc)


def fake(share: float, prec: float, auroc: float | None, ece: float) -> dict[str, Any]:
    return {
        "tau": [
            {
                "population": "header",
                "target": 0.98,
                "slice": "all",
                "scheme": "nested",
                "accepted_share": {"point": share},
                "precision_accepted": {"point": prec},
            },
            {  # in-sample rows must never be picked up as the nested cell
                "population": "header",
                "target": 0.98,
                "slice": "all",
                "scheme": "in_sample",
                "accepted_share": {"point": 0.0},
                "precision_accepted": {"point": 0.0},
            },
        ],
        "auroc": [
            {"variant": "v2_all_rows", "slice": "all", "group": "header_all", "point": auroc},
            {"variant": "v1", "slice": "all", "group": "header_all", "point": 0.1},
        ],
        "calibration": [{"slice": "all", "group": "header", "ece_width15": ece}],
        "model_selection": {"chosen_structure": "pooled", "chosen_kind": "lr"},
        "doc_attainability": {"0.95": {"attained": False}, "0.98": {"attained": False}},
    }


def test_cells_pick_nested_all_slice_and_handle_missing_and_nan() -> None:
    r = fake(1.0, 0.998, 0.964, 0.0006)
    assert cc.tau_cell(r, "header", 0.98) == "100.0 / 99.8"
    assert cc.tau_cell(r, "header", 0.95) == "n/a" and cc.tau_cell(r, "quantity", 0.98) == "n/a"
    assert cc.auroc_cell(r, "header_all") == "0.964"
    assert cc.auroc_cell(fake(1, 1, float("nan"), 0), "header_all") == "n/a"
    assert cc.auroc_cell(r, "row.quantity") == "n/a"
    assert cc.ece_cell(r, "header") == "0.001" and cc.ece_cell(r, "doc") == "n/a"


def test_render_lists_both_calibrators_and_merge_block_is_idempotent() -> None:
    text = cc.render(fake(1.0, 0.99, 0.9, 0.01), fake(0.5, 0.98, 0.8, 0.02), "run1260", "runnat")
    assert "`meta/calibrator_zs.json`" in text and "`meta/calibrator_zs_native.json`" in text
    assert "50.0 / 98.0" in text and "e4b84ec2809625d5" in text
    old = "# T\n\nbody\n"
    once = cc.merge_block(old, text)
    assert once.startswith(old.rstrip("\n")) and cc.merge_block(once, text) == once
    assert cc.merge_block(once, "OTHER").count(cc.BEGIN) == 1
