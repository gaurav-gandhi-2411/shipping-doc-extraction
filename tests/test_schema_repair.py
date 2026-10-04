"""Schema repair (shipdoc.coerce.repair_*): a date that can never satisfy the schema becomes null.

Unit tests of the rule, the saved runs scored before / after with the official scorer, and the
writers (spike runner, merge-shards, ``shipdoc predict`` assembly) driven by a mock backend that
misreads one date. The assignment folder is gitignored: tests needing it skip cleanly without it.
"""

# ruff: noqa: F811

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import pytest
from test_predict import (
    DEV_IDS,
    N_PAGES,
    RUN,
    TEST_IDS,
    World,
    _clean_sha,  # noqa: F401 - autouse fixture, must be visible in this module
    assemble,
    needs_schema,
    run_all,
    world,  # noqa: F401 - fixture
)

from shipdoc import eval as ev
from shipdoc import shardmerge, spike
from shipdoc.coerce import (
    REPAIR_REASON,
    coerce_predictions,
    date_rules,
    default_date_rules,
    repair_doc,
    repair_predictions,
)
from shipdoc.extract import MockBackend

ROOT = Path(__file__).resolve().parents[1]
SCORER = ROOT / "assignment" / "score.py"
SAVED = [Path("D:/shipdoc/runs/spike_download/x"), Path("D:/shipdoc/runs/spike_download2/x")]
needs_scorer = pytest.mark.skipif(not SCORER.is_file(), reason="assignment/score.py absent")
needs_saved = pytest.mark.skipif(not all(p.is_dir() for p in SAVED), reason="saved runs absent")

BAD = "2470"


def _doc(date: Any) -> dict[str, Any]:
    return {"doc_type": "invoice", "header": {"invoice_date": date, "invoice_number": "A-1"},
            "line_items": []}  # fmt: skip


# ---------------------------------------------------------------------- the rule


@needs_schema
def test_rules_are_read_from_the_schema_not_hardcoded() -> None:
    assert list(default_date_rules()) == ["invoice_date"]
    custom = {"$defs": {"h": {"properties": {
        "ship_date": {"type": ["string", "null"], "pattern": "^\\d{2}/\\d{2}$"},
        "po_code": {"type": ["string", "null"], "pattern": "^[A-Z]+$"},  # a pattern, not a date
        "note": {"type": ["string", "null"]},
    }}}}  # fmt: skip
    rules = date_rules(custom)
    assert list(rules) == ["ship_date"]
    out, ev_ = repair_doc({"header": {"ship_date": "2026-01-01", "po_code": "abc"}}, rules)
    assert out["header"] == {"ship_date": None, "po_code": "abc"}  # po_code is never repaired
    assert ev_ == [{"field": "ship_date", "reason": REPAIR_REASON}]


@needs_schema
@pytest.mark.parametrize("value", ["2026-01-31", "1999-12-01", "2026-13-45", "0000-00-00"])
def test_values_that_satisfy_the_pattern_are_never_touched(value: str) -> None:
    # "2026-13-45" matches the schema pattern, so the schema does not call it wrong: left alone
    doc = _doc(value)
    out, events = repair_doc(doc)
    assert out == doc and events == []


@needs_schema
@pytest.mark.parametrize("value", [BAD, "03/04/2026", "2026-1-5", "2026-01-01T00:00", "", " ",
                                   "2026-01-01\n", "２０２６-01-01", "ten"])  # fmt: skip
def test_pattern_violations_become_null_with_an_event_that_has_no_value(value: str) -> None:
    out, events = repair_doc(_doc(value))
    assert out["header"]["invoice_date"] is None and out["header"]["invoice_number"] == "A-1"
    assert events == [{"field": "invoice_date", "reason": "schema_invalid_date"}]
    assert value.strip() not in json.dumps(events) or value.strip() == ""


@needs_schema
def test_null_stays_null_waybills_and_other_fields_are_untouched() -> None:
    doc = _doc(None)
    assert repair_doc(doc) == (doc, [])
    way = {
        "doc_type": "waybill",
        "header": {"carrier": "XY", "mawb": "garbage 12"},
        "line_items": [],
    }
    assert repair_doc(way) == (way, [])
    odd = {"doc_type": "invoice", "header": {"invoice_number": "2470", "currency": "usd??"},
           "line_items": [{"purchase_order": "2470"}]}  # fmt: skip
    assert repair_doc(odd) == (odd, [])  # no other field is ever repaired


@needs_schema
def test_repair_is_idempotent_pure_and_per_document() -> None:
    preds = {"a": _doc(BAD), "b": _doc("2026-02-02"), "c": _doc(None)}
    frozen = copy.deepcopy(preds)
    once, events = repair_predictions(preds)
    assert preds == frozen  # input untouched
    assert events == [{"doc_id": "a", "field": "invoice_date", "reason": REPAIR_REASON}]
    assert once["a"]["header"]["invoice_date"] is None and once["b"] == preds["b"]
    twice, again = repair_predictions(once)
    assert twice == once and again == []
    assert repair_predictions(coerce_predictions(preds))[0] == once  # commutes with coerce


# ---------------------------------------------------------------------- the saved runs


def _saved_runs() -> list[Path]:
    return sorted(p for root in SAVED for p in root.glob("**/predictions.json"))


def _gold_for(preds: dict[str, Any]) -> dict[str, dict[str, Any]]:
    gold: dict[str, dict[str, Any]] = {}
    for split in ("train", "dev"):
        d = ROOT / "data" / split / "labels"
        if d.is_dir():
            gold.update(ev.load_gold(d))
    return {k: v for k, v in gold.items() if k in preds}


@needs_saved
@needs_scorer
@needs_schema
def test_saved_runs_score_the_same_or_better_and_the_known_misreads_are_repaired() -> None:
    sc, scored, repaired = ev.load_scorer(SCORER), 0, 0
    for path in _saved_runs():
        preds = json.loads(path.read_text(encoding="utf-8"))
        fixed, events = repair_predictions(preds)
        repaired += len(events)
        gold = _gold_for(preds)
        if not gold:
            continue
        scored += 1
        a = sc.aggregate(list(ev.per_doc_results(preds, gold).values()))
        b = sc.aggregate(list(ev.per_doc_results(fixed, gold).values()))
        assert b["OVERALL"] >= a["OVERALL"], (path, a["OVERALL"], b["OVERALL"])
        assert b["header_field_accuracy"] >= a["header_field_accuracy"], path
        for e in events:  # the event names the doc and field, never the value
            assert set(e) == {"doc_id", "field", "reason"}
        # after the repair no saved run has a date-pattern failure
        for doc in fixed.values():
            v = doc["header"].get("invoice_date")
            assert v is None or re.fullmatch(r"\d{4}-\d{2}-\d{2}", v)
    assert scored >= 9 and repaired >= 1


# ---------------------------------------------------------------------- the writers


class MisreadBackend(MockBackend):
    """Replays gold but misreads the invoice date of one document (all its pages)."""

    def __init__(self, *a: Any, bad_doc: str, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.bad_doc = bad_doc

    def page_payload(self, doc_id: str, page_index: int, n_pages: int) -> dict[str, Any]:
        out = super().page_payload(doc_id, page_index, n_pages)
        if doc_id == self.bad_doc and out["header"].get("invoice_date"):
            out["header"]["invoice_date"] = BAD
        return out


def _misread(w: World) -> MisreadBackend:
    return MisreadBackend(w.gold, fixed_latency_s=0.5, bad_doc="test_0000")


def test_spike_writer_repairs_traces_predictions_and_manifest(world: World) -> None:
    be = MisreadBackend(world.gold, fixed_latency_s=0.5, bad_doc=DEV_IDS[0])
    spike.run_spike(world.cfg, DEV_IDS, "dev", "r0", be, runs_root=world.runs, data_root=world.data)
    run = world.runs / "r0"
    preds = json.loads((run / "predictions.json").read_text())
    assert preds[DEV_IDS[0]]["header"]["invoice_date"] is None
    traces = spike._read_trace(run / "trace.jsonl")
    assert traces[0]["schema_repairs"] == [{"field": "invoice_date", "reason": REPAIR_REASON}]
    assert all(t["schema_repairs"] == [] for t in traces[1:])
    assert BAD not in json.dumps([t["schema_repairs"] for t in traces])
    man = json.loads((run / "manifest.json").read_text())
    assert man["schema_repairs"] == {"n": 1, "reason": REPAIR_REASON}
    assert BAD not in (run / "manifest.json").read_text()
    # resume after completion keeps the count (idempotent)
    spike.run_spike(world.cfg, DEV_IDS, "dev", "r0", be, resume=True, runs_root=world.runs,
                    data_root=world.data)  # fmt: skip
    assert json.loads((run / "manifest.json").read_text())["schema_repairs"]["n"] == 1


@needs_schema
def test_assemble_accepts_a_run_with_a_misread_date_and_counts_it(world: World) -> None:
    be = _misread(world)
    run_all(world, be=be)
    out = world.runs.parent / "sub"
    rep = assemble(world, out)
    assert rep["ok"], {k: c["detail"] for k, c in rep["checks"].items() if not c["ok"]}
    assert rep["schema_repairs"]["n"] == 1 and rep["schema_repairs"]["in_trace"] == 1
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["schema_repairs"]["n"] == 1
    preds = json.loads((out / "test_predictions.json").read_text())
    assert preds["test_0000"]["header"]["invoice_date"] is None
    assert BAD not in (out / "validation_report.json").read_text()
    assert BAD not in (out / "manifest.json").read_text()


@needs_schema
def test_assemble_repairs_a_trace_written_before_the_repair_existed(world: World) -> None:
    run_all(world)
    trace_path = world.runs / RUN / "trace.jsonl"
    lines = [json.loads(ln) for ln in trace_path.read_text().splitlines()]
    for t in lines:
        t.pop("schema_repairs")  # an old trace
    lines[0]["prediction"]["header"]["invoice_date"] = BAD
    trace_path.write_text("".join(json.dumps(t) + "\n" for t in lines), newline="\n")
    rep = assemble(world, world.runs.parent / "sub")
    assert rep["ok"] and rep["schema_repairs"] == {
        "n": 1, "in_trace": 0, "at_assembly": 1, "reason": REPAIR_REASON,
        "events": [
            {"doc_id": lines[0]["doc_id"], "field": "invoice_date", "reason": REPAIR_REASON}
        ],
    }  # fmt: skip


@needs_schema
def test_other_schema_failures_still_reject(world: World) -> None:
    run_all(world)
    trace_path = world.runs / RUN / "trace.jsonl"
    lines = [json.loads(ln) for ln in trace_path.read_text().splitlines()]
    lines[0]["prediction"]["header"]["invoice_number"] = 123  # not a string: not repaired
    trace_path.write_text("".join(json.dumps(t) + "\n" for t in lines), newline="\n")
    rep = assemble(world, world.runs.parent / "sub")
    assert not rep["ok"] and not rep["schema_ok"]


@needs_schema
def test_merge_shards_repairs_and_sums_the_counts(world: World, tmp_path: Path) -> None:
    sharded = World(world.data, tmp_path / "sharded", world.gold, world.cfg)
    for i in (0, 1):
        run_all(sharded, shard=f"{i}/2", be=_misread(world))
    shardmerge.merge_shards(world.cfg, TEST_IDS, "test", RUN, 2, runs_root=sharded.runs,
                            data_root=world.data, expected_docs=len(TEST_IDS),
                            expected_pages=N_PAGES)  # fmt: skip
    man = json.loads((sharded.runs / RUN / "manifest.json").read_text())
    assert man["schema_repairs"] == {"n": 1, "reason": REPAIR_REASON}
    preds = json.loads((sharded.runs / RUN / "predictions.json").read_text())
    assert preds["test_0000"]["header"]["invoice_date"] is None
    rep = assemble(sharded, tmp_path / "sub")
    assert rep["ok"] and rep["schema_repairs"]["n"] == 1
