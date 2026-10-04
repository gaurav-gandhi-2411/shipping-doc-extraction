from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from shipdoc import cli
from shipdoc import replay as rp
from shipdoc.extract import HEADER_KEYS, INVOICE_KEYS
from shipdoc.merge import merge_pages

ROOT = Path(__file__).resolve().parents[1]
SCORER = ROOT / "assignment" / "score.py"
needs_scorer = pytest.mark.skipif(not SCORER.is_file(), reason="assignment/score.py absent")
CFG = ROOT / "configs" / "spike_qwen3vl_8b_img_only.yaml"

HEADER_NULL = dict.fromkeys(HEADER_KEYS)


def _row(spn: str | None, cpn: str | None, po: str | None, qty: str | None) -> dict[str, Any]:
    return {
        "supplier_part_number": spn,
        "customer_part_number": cpn,
        "purchase_order": po,
        "quantity": qty,
    }


def _page(rows: list[dict[str, Any]], **header: str) -> dict[str, Any]:
    return {
        "doc_type": "invoice",
        "header": {**HEADER_NULL, **header},
        "line_items": rows,
        "page_kind": "single",
    }


GOOD_ROWS = [_row("ZZ-1", None, "PO-9", "5"), _row("ZZ-2", "C-2", "PO-9", "7")]


def test_salvage_cuts_at_last_complete_row() -> None:
    full = json.dumps(_page(GOOD_ROWS, invoice_number="INV-X"))
    cut = full[: full.index("ZZ-2") + 6]  # inside the second row's supplier_part_number value
    assert rp.parse_page(cut, "json", rp.Fixes()) is None  # default: no repair
    got = rp.parse_page(cut, "json", rp.Fixes(salvage_truncated=True))
    assert got is not None
    assert got["header"]["invoice_number"] == "INV-X"
    assert got["line_items"] == [GOOD_ROWS[0]]  # the half-written row is dropped, never completed


def test_salvage_never_alters_values_and_handles_braces_inside_strings() -> None:
    rows = [_row("A}]{B", None, None, "1"), _row("C-3", None, None, "2")]
    full = json.dumps(_page(rows))
    cut = full[: full.index("C-3")]
    got = rp.salvage_page_json(cut)
    assert got is not None and got["line_items"] == [rows[0]]


@pytest.mark.parametrize("raw", ["", "not json", '{"doc_type": "inv', "[1, 2]", '{"a": 1}}'])
def test_salvage_returns_none_when_nothing_recoverable(raw: str) -> None:
    assert rp.salvage_page_json(raw) is None


def test_salvage_compact_fills_missing_tail_keys() -> None:
    compact = (
        '{"dt": "invoice", "h": {"inv_no": "INV-C"}, "r": [["S-1", null, null, "3"], ["S-2", nu'
    )
    got = rp.parse_page(compact, "compact", rp.Fixes(salvage_truncated=True))
    assert got is not None
    assert got["header"]["invoice_number"] == "INV-C"
    assert got["line_items"] == [_row("S-1", None, None, "3")]


def test_merge_drop_null_rows_flag() -> None:
    pages = [_page([*GOOD_ROWS, _row(None, None, None, None), _row(None, "", None, " ")])]
    kept = merge_pages(pages, drop_null_rows=False)
    assert len(kept.doc["line_items"]) == 4 and kept.diffs["dropped_null_rows"] == []
    dropped = merge_pages(pages)  # default: on
    assert dropped.doc["line_items"] == GOOD_ROWS
    assert dropped.diffs["dropped_null_rows"] == [1, 1]


def _write_run(tmp_path: Path, raws: list[str]) -> tuple[Path, Path]:
    run = tmp_path / "run"
    run.mkdir(parents=True)
    trace = {"doc_id": "dev_9001", "pages": [{"page": 1, "raw_text": r} for r in raws]}
    (run / "trace.jsonl").write_text(json.dumps(trace) + "\n", encoding="utf-8")
    labels = tmp_path / "labels"
    labels.mkdir()
    gold = {
        "doc_id": "dev_9001",
        "doc_type": "invoice",
        "header": {**dict.fromkeys(INVOICE_KEYS), "invoice_number": "INV-X"},
        "line_items": GOOD_ROWS,
    }
    (labels / "dev_9001.json").write_text(json.dumps(gold), encoding="utf-8")
    return run, labels


@needs_scorer
def test_replay_run_scores_baseline_and_fixes(tmp_path: Path) -> None:
    full = json.dumps(_page([*GOOD_ROWS, _row(None, None, None, None)], invoice_number="INV-X"))
    run, labels = _write_run(tmp_path / "a", [full])
    base = rp.replay_run(run, CFG, labels, rp.Fixes(drop_null_rows=False))
    assert base["label"] == rp.LABEL and base["documents"] == 1
    assert base["row_precision"] < 1.0  # the all-null row is an extra row
    fixed = rp.replay_run(run, CFG, labels, rp.Fixes(drop_null_rows=True))
    assert fixed["row_f1"] == 1.0 and fixed["OVERALL"] > base["OVERALL"]

    cut = json.dumps(_page(GOOD_ROWS, invoice_number="INV-X"))
    cut = cut[: cut.index("ZZ-2") + 6]
    run2, labels2 = _write_run(tmp_path / "b", [cut])
    assert rp.replay_run(run2, CFG, labels2, rp.Fixes())["row_recall"] == 0.0
    salvaged = rp.replay_run(run2, CFG, labels2, rp.Fixes(salvage_truncated=True))
    assert salvaged["row_recall"] == 0.5  # one of two rows recoverable


@needs_scorer
def test_cli_replay_prints_label(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run, labels = _write_run(tmp_path, [json.dumps(_page(GOOD_ROWS, invoice_number="INV-X"))])
    out = tmp_path / "o.json"
    argv = ["replay", "--run-dir", str(run), "--config", str(CFG), "--labels", str(labels)]
    rc = cli.main([*argv, "--drop-null-rows", "--salvage-truncated", "--out", str(out)])
    assert rc == 0
    assert "replay on saved outputs" in capsys.readouterr().out
    assert json.loads(out.read_text(encoding="utf-8"))["fixes"]["salvage_truncated"] is True
