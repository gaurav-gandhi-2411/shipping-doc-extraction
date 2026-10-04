"""The determinism pass replays the first pass's batch composition (shipdoc.predict.replay_plan).

A real batched decode (left-padded, fp16 kernels) can give a page different text depending on which
other pages share its generate call, so a second pass that decodes the 5 check documents alone
proves nothing. The `CompositionBackend` below reproduces that sensitivity on the mock: a page-1
field carries a signature of the ORDERED members of the call that decoded it (still schema-valid).
"""

# ruff: noqa: F811

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from test_predict import (
    DEV_IDS,
    RUN,
    TEST_IDS,
    FlakyBackend,
    World,
    _clean_sha,  # noqa: F401 - autouse fixture, must be visible in this module
    decide,
    mock,
    run_all,
    smoke_ok,
    world,  # noqa: F401 - fixture
)

from shipdoc import bench, predict, spike
from shipdoc.extract import MockBackend


class CompositionBackend(MockBackend):
    """Output of a page depends on the ordered (doc, page) list of its generate call."""

    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self._sig: str | None = None

    def set_context(self, doc_id: str, page_index: int, n_pages: int) -> None:
        super().set_context(doc_id, page_index, n_pages)
        self._sig = None  # a single-page call: the signature is the page itself

    def page_payload(self, doc_id: str, page_index: int, n_pages: int) -> dict[str, Any]:
        out = super().page_payload(doc_id, page_index, n_pages)
        sig = self._sig or f"{doc_id}:{page_index}"
        tag = hashlib.sha1(sig.encode()).hexdigest()[:8]  # noqa: S324 - a test signature
        if page_index == 0:
            key = "ship_to_name" if out["doc_type"] == "invoice" else "shipper_name"
            out["header"][key] = f"{out['header'].get(key)} #{tag}"
        return out

    def extract_pages(self, requests: Any) -> Any:
        sig = "|".join(f"{r.context[0]}:{r.context[1]}" for r in requests)
        out = []
        for r in requests:
            self.set_context(*r.context)
            self._sig = sig
            out.append(self.extract_page(r.image, r.prompt, r.schema, r.ocr_text))
        wall = self._latency(0.0)
        for _raw, _parsed, meta in out:
            meta.update(batch_size=len(out), batch_latency_s=wall, latency_s=wall / len(out))
        return out


def comp(w: World) -> CompositionBackend:
    return CompositionBackend(w.gold, fixed_latency_s=0.5)


def _trace(w: World, run: str = RUN) -> list[dict[str, Any]]:
    return spike._read_trace(w.runs / run / "trace.jsonl")


# ---------------------------------------------------------------------- the record


@pytest.mark.parametrize("batch", [1, 4])
def test_every_page_records_the_call_that_decoded_it(world: World, batch: int) -> None:
    spike.run_spike(world.cfg, DEV_IDS, "dev", "r", comp(world), runs_root=world.runs,
                    data_root=world.data, batch_size=batch)  # fmt: skip
    seen: dict[tuple[str, int], tuple[str, list[list[Any]]]] = {}
    for t in _trace(world, "r"):
        for k, p in enumerate(t["pages"]):
            eb = p["meta"].get("exec_batch")
            if batch == 1:
                assert eb is None  # the unbatched path is untouched: no new key
                continue
            if eb is None:  # decoded alone (a call of one page is not stamped)
                continue
            assert [t["doc_id"], k] in eb["members"] and 1 < len(eb["members"]) <= batch
            seen[(t["doc_id"], k)] = (eb["id"], eb["members"])
    for _key, (call_id, members) in seen.items():  # members agree about the call
        assert all(seen[(d, i)] == (call_id, members) for d, i in members)
    assert (batch == 1) == (not seen)
    man = json.loads((world.runs / "r" / "manifest.json").read_text())
    assert man["exec_batch_recorded"] is True


def test_replay_plan_covers_every_call_of_the_check_docs_in_member_order(
    world: World,
) -> None:
    spike.run_spike(world.cfg, TEST_IDS, "test", "r", comp(world), runs_root=world.runs,
                    data_root=world.data, batch_size=4)  # fmt: skip
    first = _trace(world, "r")
    ids = predict.select_determinism_docs({t["doc_id"]: len(t["pages"]) for t in first})
    batches, touched = predict.replay_plan(first, ids)
    by_page = {(t["doc_id"], k): p["meta"].get("exec_batch") for t in first
               for k, p in enumerate(t["pages"])}  # fmt: skip
    for d in ids:
        for k in range(next(len(t["pages"]) for t in first if t["doc_id"] == d)):
            members = (by_page[(d, k)] or {"members": [[d, k]]})["members"]
            hit = [[list(x) for x in b] for b in batches if (d, k) in [tuple(x) for x in b]]
            assert hit == [members]
    assert set(touched) >= set(ids)
    assert any(len(b) > 1 for b in batches)


# ---------------------------------------------------------------------- the pass


@pytest.mark.parametrize("batch", [2, 4, 8])
def test_composition_dependent_backend_passes_when_the_composition_is_replicated(
    world: World,
    batch: int,
) -> None:
    run_all(world, batch=batch, be=comp(world))
    rep = json.loads((world.runs / RUN / "determinism.json").read_text())
    assert rep["ok"] and rep["n_docs"] == rep["n_identical"] == 5
    assert rep["composition_replicated"] is True
    assert rep["replayed_pages"] == rep["replayed_pages_identical"] > 0
    assert rep["replayed_composition_identical"] == rep["replayed_pages"]
    assert "exec_batch" in rep["replay_rule"]


def test_the_old_second_pass_would_have_failed_this_backend(world: World) -> None:
    """Control: decoding only the 5 check docs (the former pass) changes a composition-dependent
    backend's output, so the test above is not vacuous."""
    run_all(world, batch=4, be=comp(world), det=False)
    first = _trace(world)
    ids = predict.select_determinism_docs({t["doc_id"]: len(t["pages"]) for t in first})
    spike.run_spike(world.cfg, ids, "test", "old_style", comp(world), runs_root=world.runs,
                    data_root=world.data, batch_size=4)  # fmt: skip
    res = predict.compare_runs(first, _trace(world, "old_style"), ids)
    assert not res["ok"] and res["n_different"] > 0


@pytest.mark.parametrize("how", ["reversed_order", "reshuffled_composition"])
def test_a_shuffled_replay_fails(
    world: World,
    monkeypatch: pytest.MonkeyPatch,
    how: str,
) -> None:
    real = predict.replay_plan

    def shuffled(first: Any, ids: Any, recorded: bool = True) -> Any:
        batches, touched = real(first, ids, recorded)
        if how == "reversed_order":
            return [list(reversed(b)) for b in batches], touched
        flat = [m for b in batches for m in b]  # same pages, different groups
        cut = [flat[i : i + 3] for i in range(0, len(flat), 3)]
        return cut, touched

    monkeypatch.setattr(predict, "replay_plan", shuffled)
    with pytest.raises(predict.DeterminismError) as exc:
        run_all(world, batch=4, be=comp(world))
    rep = json.loads((world.runs / RUN / "determinism.json").read_text())
    assert rep["ok"] is False
    assert "replayed" in str(exc.value) and "#" not in str(exc.value)  # counts, never values


def test_a_nondeterministic_backend_fails_even_with_the_composition_replicated(
    world: World,
) -> None:
    with pytest.raises(predict.DeterminismError):
        run_all(world, batch=4, be=FlakyBackend(world.gold, fixed_latency_s=0.5))
    rep = json.loads((world.runs / RUN / "determinism.json").read_text())
    assert rep["ok"] is False and rep["composition_replicated"] is True
    assert rep["replayed_pages_identical"] < rep["replayed_pages"]


def test_a_trace_without_the_composition_record_raises_a_clear_error(
    world: World,
) -> None:
    be = mock(world)
    status = smoke_ok(world, be)
    dec = decide(world, 4)
    predict.run_test(world.cfg, be, RUN, world.runs, world.data, dec, status)
    tp = world.runs / RUN / "trace.jsonl"
    lines = [json.loads(ln) for ln in tp.read_text().splitlines()]
    for t in lines:
        for p in t["pages"]:
            p["meta"].pop("exec_batch", None)
    tp.write_text("".join(json.dumps(t) + "\n" for t in lines), encoding="utf-8", newline="\n")
    mp = world.runs / RUN / "manifest.json"  # what a run from before the stamp looks like
    man = json.loads(mp.read_text())
    man.pop("exec_batch_recorded")
    mp.write_text(json.dumps(man), encoding="utf-8")
    with pytest.raises(predict.ReplayPlanError, match="exec_batch"):
        predict.run_determinism(world.cfg, be, RUN, world.runs, world.data, dec)
    assert not (world.runs / RUN / "determinism.json").exists()


def test_a_call_whose_members_disagree_in_the_trace_is_refused(world: World) -> None:
    run_all(world, batch=4, be=comp(world), det=False)
    first = _trace(world)
    ids = predict.select_determinism_docs({t["doc_id"]: len(t["pages"]) for t in first})
    victim = next(
        p
        for t in first
        for p in t["pages"]
        if len(p["meta"]["exec_batch"]["members"]) > 1 and t["doc_id"] in ids
    )
    victim["meta"]["exec_batch"]["id"] += "-other"
    with pytest.raises(predict.ReplayPlanError, match="cannot rebuild"):
        predict.replay_plan(first, ids)


def test_an_old_determinism_report_without_the_replication_flag_is_not_accepted(
    world: World,
) -> None:
    run_all(world, batch=4)
    p = world.runs / RUN / "determinism.json"
    rep = json.loads(p.read_text())
    rep.pop("composition_replicated")
    p.write_text(json.dumps(rep), encoding="utf-8")
    assert predict._determinism([world.runs / RUN])["ok"] is False


def test_replay_with_resume_does_not_duplicate_documents(world: World) -> None:
    run_all(world, batch=4, be=comp(world))
    be = comp(world)
    dec = json.loads((world.runs / "decision.json").read_text())
    predict.run_determinism(world.cfg, be, RUN, world.runs, world.data, dec)  # second run, resume
    det = _trace(world, f"{RUN}_det")
    ids = [t["doc_id"] for t in det]
    assert len(ids) == len(set(ids)) and set(ids) >= set(
        json.loads((world.runs / RUN / "determinism.json").read_text())["doc_ids"]
    )


# ---------------------------------------------------------------------- bench is not determinism


def test_bench_compares_different_compositions_by_design_and_says_so() -> None:
    doc = bench.__doc__ or ""
    assert "different composition" in doc and "not a determinism" in doc.replace("\n", " ")
    assert Path(bench.__file__).is_file()


def test_a_batch_size_one_run_from_before_the_stamp_is_still_replayable(
    world: World,
) -> None:
    """Batch 1 decodes every page alone: nothing to rebuild, so no stamp is needed."""
    run_all(world, batch=1, det=False)
    mp = world.runs / RUN / "manifest.json"
    man = json.loads(mp.read_text())
    man.pop("exec_batch_recorded")
    mp.write_text(json.dumps(man), encoding="utf-8")
    dec = json.loads((world.runs / "decision.json").read_text())
    rep = predict.run_determinism(world.cfg, mock(world), RUN, world.runs, world.data, dec)
    assert rep["ok"] and rep["composition_replicated"] and rep["replayed_pages"] > 0
