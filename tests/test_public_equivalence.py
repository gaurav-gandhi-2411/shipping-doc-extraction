"""scripts/public_equivalence.py: the re-assembly hash check (the post path is stubbed)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "public_equivalence", ROOT / "scripts" / "public_equivalence.py"
)
pe = importlib.util.module_from_spec(spec)
sys.modules["public_equivalence"] = pe
spec.loader.exec_module(pe)

FINAL = {"test_0002": {"doc_type": "invoice"}, "test_0001": {"doc_type": "waybill"}}
EXPECTED_BYTES = (json.dumps(FINAL, indent=1) + "\n").encode("utf-8")  # insertion order, LF
EXPECTED = hashlib.sha256(EXPECTED_BYTES).hexdigest()


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    from shipdoc import flags

    trace = tmp_path / "trace.jsonl"
    trace.write_text(json.dumps({"doc_id": "test_0002", "pages": []}) + "\n", encoding="utf-8")

    def fake(traces: Any, shapes: Any, ocr: Any) -> tuple[dict, dict]:
        assert [t["doc_id"] for t in traces] == ["test_0002"]
        return FINAL, {}

    monkeypatch.setattr(flags, "post_rule_output", fake)
    return trace


def test_hash_is_that_of_the_production_writer_in_insertion_order(stub: Path) -> None:
    digest, n = pe.assemble_sha256(stub, stub.parent)
    assert (digest, n) == (EXPECTED, 2)  # test_0002 before test_0001: not sorted


def test_main_reports_match_and_mismatch_with_exit_codes(
    stub: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["--trace", str(stub), "--ocr-cache", str(stub.parent)]
    assert pe.main([*args, "--expect-sha256", EXPECTED.upper()]) == 0
    assert "MATCH" in capsys.readouterr().out
    assert pe.main([*args, "--expect-sha256", "0" * 64]) == 1
    assert "MISMATCH" in capsys.readouterr().out
    assert pe.main(args) == 0


def test_an_empty_trace_is_an_input_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    empty = tmp_path / "trace.jsonl"
    empty.write_text("", encoding="utf-8")
    assert pe.main(["--trace", str(empty), "--ocr-cache", str(tmp_path)]) == 2
    assert "no traced document" in capsys.readouterr().err
