"""CPU tests for the test-prediction stages (shipdoc.predict): the logic behind notebook 04.

Everything runs on the mock backend over a SYNTHETIC corpus (tests/_synth.py) whose dev images are
copied to a fake ``test`` folder with renamed ids. No real test image, label or prediction is
read or written; the schema check uses the (gitignored) assignment schema and skips without it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from _synth import SPECS, make_corpus

from shipdoc import bench, cli, ocr, paths, postrules, predict, runmeta, shardmerge, spike
from shipdoc.coerce import coerce_predictions, repair_predictions
from shipdoc.extract import MockBackend, MockCorruption
from shipdoc.merge import FieldProvenance, merge_pages
from shipdoc.normalize import normalize_doc
from shipdoc.postrules import RuleConfig

ROOT = Path(__file__).resolve().parents[1]
KEYED = ROOT / "configs" / "spike_qwen35_4b_img_only.yaml"
SCHEMA = ROOT / "assignment" / "schema.json"
SAMPLE = ROOT / "assignment" / "sample_submission.json"
needs_schema = pytest.mark.skipif(not SCHEMA.is_file(), reason="assignment/schema.json absent")
SHA = "a" * 40
DEV_IDS = [s[0] for s in SPECS]
TEST_IDS = [d.replace("dev_", "test_") for d in DEV_IDS]
N_PAGES = sum(s[1] for s in SPECS)
SMOKE_IDS = DEV_IDS[:5]
RUN = "t0"
SENTINEL = "Supplier 3 Ltd"  # a value the mock replays for dev_0003 / test_0003


@dataclass
class World:
    data: Path
    runs: Path
    gold: dict[str, dict[str, Any]]
    cfg: spike.SpikeConfig


@pytest.fixture(autouse=True)
def _clean_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fixed clean code SHA, so the tests do not depend on the state of the working tree."""
    for mod in (spike, predict, bench):
        monkeypatch.setattr(mod, "git_commit", lambda: SHA)


@pytest.fixture()
def world(tmp_path: Path) -> World:
    data = tmp_path / "data"
    gold = make_corpus(data)
    (data / "test" / "images").mkdir(parents=True)
    for img in (data / "dev" / "images").iterdir():  # fake test folder = renamed dev copies
        shutil.copyfile(img, data / "test" / "images" / img.name.replace("dev_", "test_"))
    both = dict(gold)
    for d, g in gold.items():
        t = d.replace("dev_", "test_")
        both[t] = {**g, "doc_id": t}
    return World(data, tmp_path / "runs", both, spike.load_config(KEYED))


def mock(w: World, **kw: Any) -> MockBackend:
    return MockBackend(w.gold, fixed_latency_s=0.5, **kw)


class Kill(BaseException):
    """Simulates a kill: not an Exception, so no fallback or handler in the runner catches it."""


class KillingBackend(MockBackend):
    def __init__(self, *a: Any, after: int, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.after = after

    def extract_page(self, *a: Any, **kw: Any) -> Any:
        if self._calls >= self.after:
            raise Kill
        return super().extract_page(*a, **kw)


class FlakyBackend(MockBackend):
    """Nondeterministic on purpose: a page decoded a second time gets a different invoice date."""

    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.seen: dict[tuple[str, int], int] = {}

    def page_payload(self, doc_id: str, page_index: int, n_pages: int) -> dict[str, Any]:
        out = super().page_payload(doc_id, page_index, n_pages)
        n = self.seen[(doc_id, page_index)] = self.seen.get((doc_id, page_index), 0) + 1
        if n > 1 and out["header"].get("invoice_date"):
            out["header"]["invoice_date"] = "2031-12-31"
        return out


def smoke_ok(w: World, be: Any = None, tag: str = RUN) -> Path:
    status = w.runs / "smoke_status.json"
    st = predict.run_smoke_gate(w.cfg, be or mock(w), SMOKE_IDS, f"smoke_{tag}", w.runs, status,
                                w.data)  # fmt: skip
    assert st["state"] == "passed", st
    return status


def decide(w: World, batch: int | None = 1, dev: Path | None = None) -> dict[str, Any]:
    return predict.decide_batch_size(
        w.cfg, mock(w), w.runs / "bench", DEV_IDS, w.runs / "decision.json", dev, batch, w.data
    )


def run_all(w: World, batch: int = 1, shard: str = "0/1", be: Any = None, det: bool = True,
            dev: Path | None = None) -> dict[str, Any]:  # fmt: skip
    be = be or mock(w)
    status = smoke_ok(w, be)
    dec = decide(w, batch, dev)
    predict.run_test(w.cfg, be, RUN, w.runs, w.data, dec, status, shard)
    if det:
        predict.run_determinism(w.cfg, be, RUN, w.runs, w.data, dec, shard)
    return dec


def assemble(w: World, out: Path, run_id: str = RUN, **kw: Any) -> dict[str, Any]:
    kw.setdefault("expect_docs", len(TEST_IDS))
    kw.setdefault("expect_pages", N_PAGES)
    # an empty OCR cache: the configured one holds real test_XXXX pages whose ids collide with
    # the synthetic ones, and a test must not depend on it
    kw.setdefault("ocr_cache", w.runs.parent / "no_ocr_cache")
    return predict.assemble_submission(w.cfg, run_id, w.runs, w.data, out, SCHEMA, **kw)


# ---------------------------------------------------------------------- documents


def test_ids_come_from_image_file_names_and_never_from_labels(world: World) -> None:
    pages = predict.discover_test_docs(world.data)
    assert list(pages) == TEST_IDS and sum(pages.values()) == N_PAGES
    assert not (world.data / "test" / "labels").exists()  # the fixture has no test labels
    (world.data / "test" / "images" / "test_9999_p1.png").write_bytes(b"")
    assert "test_9999" in predict.discover_test_docs(world.data)


def test_discover_refuses_missing_folder_and_page_gaps(world: World) -> None:
    with pytest.raises(predict.PredictError, match="no image folder"):
        predict.discover_test_docs(world.data, "nope")
    (world.data / "test" / "images" / "test_0001_p2.png").unlink()  # test_0001 has pages 1..3
    with pytest.raises(predict.PredictError, match="missing/duplicate page numbers"):
        predict.discover_test_docs(world.data)


def test_determinism_selection_is_seeded_documented_and_mixed() -> None:
    pages = {f"d{i:03d}": (2 if i % 4 == 0 else 1) for i in range(200)}
    sel = predict.select_determinism_docs(pages)
    assert sel == predict.select_determinism_docs(dict(reversed(pages.items())))  # order-free
    assert len(sel) == 5 and sel == sorted(sel)
    assert sum(pages[d] > 1 for d in sel) == 2 and sum(pages[d] == 1 for d in sel) == 3
    assert sel != predict.select_determinism_docs(pages, seed=7)  # the seed matters
    assert "random.Random(42)" in predict.DET_RULE
    assert predict.select_determinism_docs({"a": 1, "b": 2}) == ["a", "b"]  # fewer than n: all
    only_multi = {f"m{i}": 2 for i in range(9)}
    assert len(predict.select_determinism_docs(only_multi)) == 5  # topped up from the other kind


def test_plan_partitions_documents_over_shards(world: World) -> None:
    p = predict.plan(world.data, "0/2")
    q = predict.plan(world.data, "1/2")
    assert p["n_docs"] == len(TEST_IDS) and p["n_pages"] == N_PAGES
    assert sorted(p["shard_docs"] + q["shard_docs"]) == TEST_IDS
    assert p["shard_n_pages"] + q["shard_n_pages"] == N_PAGES
    assert set(p["determinism_docs"]) <= set(p["shard_docs"])


# ---------------------------------------------------------------------- smoke gate


def test_smoke_gate_blocks_the_test_run_when_it_fails(world: World) -> None:
    bad = mock(world, corruption=MockCorruption(invalid_json_every=1))  # every page truncated
    status = world.runs / "smoke_status.json"
    st = predict.run_smoke_gate(
        world.cfg, bad, SMOKE_IDS, "smoke_x", world.runs, status, world.data
    )
    assert st["state"] == "failed" and not all(c["passed"] for c in st["checks"])
    dec = decide(world)
    with pytest.raises(predict.GateError, match="state 'failed'"):
        predict.run_test(world.cfg, mock(world), RUN, world.runs, world.data, dec, status)
    assert not (world.runs / RUN).exists()  # nothing was decoded


def test_smoke_gate_missing_status_or_other_code_blocks_the_run(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    dec = decide(world)
    status = world.runs / "smoke_status.json"
    with pytest.raises(predict.GateError, match="not run"):
        predict.run_test(world.cfg, mock(world), RUN, world.runs, world.data, dec, status)
    smoke_ok(world)
    monkeypatch.setattr(predict, "git_commit", lambda: "b" * 40)  # other code than the smoke saw
    with pytest.raises(predict.GateError, match="code_sha"):
        predict.run_test(world.cfg, mock(world), RUN, world.runs, world.data, dec, status)
    assert not (world.runs / RUN).exists()


def test_smoke_gate_error_is_a_failed_gate_and_a_passed_one_is_not_repeated(world: World) -> None:
    status = world.runs / "smoke_status.json"
    st = predict.run_smoke_gate(world.cfg, mock(world), ["dev_nope"], "s", world.runs, status,
                                world.data)  # fmt: skip
    assert st["state"] == "error" and st["error"]  # could not verify = fail
    smoke_ok(world, tag="s2")
    again = predict.run_smoke_gate(world.cfg, mock(world), SMOKE_IDS, "smoke_s2", world.runs,
                                   status, world.data)  # fmt: skip
    assert again["skipped"] is True


# ---------------------------------------------------------------------- run, resume, determinism


@pytest.mark.parametrize("batch", [1, 4])
def test_resume_after_a_kill_is_byte_identical(world: World, tmp_path: Path, batch: int) -> None:
    status = smoke_ok(world)
    dec = decide(world, batch)
    killed = KillingBackend(world.gold, fixed_latency_s=0.5, after=9)
    with pytest.raises(Kill):
        predict.run_test(world.cfg, killed, RUN, world.runs, world.data, dec, status)
    trace = world.runs / RUN / "trace.jsonl"
    partial = trace.read_text(encoding="utf-8").splitlines() if trace.is_file() else []
    if batch == 1:
        assert 0 < len(partial) < len(TEST_IDS)  # a per-document trace survived the kill
    else:  # the 20 pages are one window of 8 batches: a kill loses the whole window, never more
        assert len(partial) < len(TEST_IDS)
    predict.run_test(world.cfg, mock(world), RUN, world.runs, world.data, dec, status)
    other = World(world.data, tmp_path / "runs2", world.gold, world.cfg)
    smoke_ok(other)
    predict.run_test(world.cfg, mock(world), RUN, other.runs, world.data, decide(other, batch),
                     other.runs / "smoke_status.json")  # fmt: skip
    for name in ("trace.jsonl", "predictions.json"):
        assert (world.runs / RUN / name).read_bytes() == (other.runs / RUN / name).read_bytes()
    assert runmeta.read_manifest(world.runs / RUN)["batch_size"] == batch


def test_determinism_passes_for_a_deterministic_backend_and_fails_for_a_flaky_one(
    world: World, tmp_path: Path
) -> None:
    run_all(world)
    rep = json.loads((world.runs / RUN / "determinism.json").read_text())
    assert rep["ok"] and rep["n_docs"] == rep["n_identical"] == 5 and rep["seed"] == 42
    flaky = World(world.data, tmp_path / "flaky", world.gold, world.cfg)
    with pytest.raises(predict.DeterminismError) as exc:
        run_all(flaky, be=FlakyBackend(world.gold, fixed_latency_s=0.5))
    msg = str(exc.value)
    assert "documents differ" in msg and "2031" not in msg  # counts, never values
    rep = json.loads((flaky.runs / RUN / "determinism.json").read_text())
    assert rep["ok"] is False and rep["n_different"] > 0


def test_second_pass_uses_its_own_run_folder_with_the_same_batch_size(world: World) -> None:
    run_all(world, batch=4)
    det = runmeta.read_manifest(world.runs / f"{RUN}_det")
    assert det["batch_size"] == 4 and det["split"] == "test"
    planned = json.loads((world.runs / RUN / "determinism.json").read_text())["doc_ids"]
    redone = spike._read_trace(world.runs / f"{RUN}_det" / "trace.jsonl")
    # the replay also completes companion documents whose every page rode in a replayed call
    assert set(planned) <= {t["doc_id"] for t in redone}


def test_determinism_needs_a_complete_main_run(world: World) -> None:
    dec = decide(world)
    with pytest.raises(predict.PredictError, match="not complete"):
        predict.run_determinism(world.cfg, mock(world), RUN, world.runs, world.data, dec)


# ---------------------------------------------------------------------- batch size contract


def _dev_run(w: World, batch: int, bench_path: Path | None = None) -> Path:
    info = {"path": str(bench_path), "chosen": batch, "deviation": None} if bench_path else None
    root = w.runs.parent / "dev_runs"
    spike.run_spike(w.cfg, DEV_IDS, "dev", "devrun", mock(w), runs_root=root, data_root=w.data,
                    logprobs=True, batch_size=batch, bench_info=info)  # fmt: skip
    return root / "devrun"


def test_batch_contract_mismatch_refuses(world: World) -> None:
    dev = _dev_run(world, 4)
    with pytest.raises(predict.BatchContractError, match="dev run used 4"):
        decide(world, 2, dev)
    assert not (world.runs / "decision.json").exists()
    assert decide(world, 4, dev)["contract"] == "checked"
    assert predict.check_batch_contract(None, 8).startswith("unchecked")


def test_run_refuses_when_the_decision_no_longer_matches_the_dev_run(world: World) -> None:
    dev = _dev_run(world, 4)
    status = smoke_ok(world)
    dec = decide(world, 4, dev)
    dec["batch_size"] = 2  # a hand-edited decision file
    with pytest.raises(predict.BatchContractError):
        predict.run_test(world.cfg, mock(world), RUN, world.runs, world.data, dec, status)
    assert not (world.runs / RUN).exists()


def test_stored_dev_bench_is_reused_when_it_applies_else_the_bench_reruns(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    dev_bench = world.runs.parent / "dev_bench"
    first = predict.decide_batch_size(
        world.cfg,
        mock(world),
        dev_bench,
        DEV_IDS,
        world.runs.parent / "d0.json",
        None,
        None,
        world.data,
    )
    assert first["source"] == "bench" and not first["bench_reused"]  # no stored result: ran it
    chosen = first["batch_size"]
    dev = _dev_run(world, chosen, dev_bench / bench.RESULT_NAME)
    dec = decide(world, None, dev)  # test bench dir is empty: the dev result is copied + verified
    assert dec["bench_reused"] and dec["batch_size"] == chosen and dec["contract"] == "checked"
    assert dec["bench"]["sha256"] == predict.sha256_file(dev_bench / bench.RESULT_NAME)
    monkeypatch.setattr(
        bench, "git_commit", lambda: "c" * 40
    )  # re-pinned code: stored one is stale
    shutil.rmtree(world.runs / "bench")
    again = decide(world, None, dev)
    assert not again["bench_reused"] and "does not apply" in again["note"]


def test_manual_batch_size_skips_the_bench_and_is_validated(world: World) -> None:
    dec = decide(world, 2)
    assert dec["source"] == "manual" and dec["bench"] is None
    assert not (world.runs / "bench").exists()
    for bad in (0, -1, True):
        with pytest.raises(predict.PredictError, match="int >= 1"):
            decide(world, bad)  # type: ignore[arg-type]


# ---------------------------------------------------------------------- validation


@needs_schema
def test_schema_validation_accepts_a_correct_submission_and_reports_no_values(world: World) -> None:
    run_all(world)
    out = world.runs.parent / "sub"
    rep = assemble(world, out)
    assert rep["ok"], {k: c["detail"] for k, c in rep["checks"].items() if not c["ok"]}
    preds = json.loads((out / "test_predictions.json").read_text())
    assert list(preds) == TEST_IDS and not (out / predict.REJECTED_NAME).exists()
    assert predict.validate_schema(preds, json.loads(SCHEMA.read_text()))["ok"]


@needs_schema
@pytest.mark.parametrize(
    ("mutate", "kind"),
    [
        (lambda d: d["header"].pop("invoice_number"), "required @ */header"),
        (lambda d: d["header"].update(invoice_number=123), "type @ */header/invoice_number"),
        (lambda d: d["header"].update(extra_field="x"), "additionalProperties @ */header"),
        (lambda d: d.update(doc_type="receipt"), "enum @ */doc_type"),
        (lambda d: d["line_items"][0].update(quantity=5), "type @ */line_items/[]/quantity"),
        (lambda d: d.pop("line_items"), "required @ *"),
        (
            lambda d: d["header"].update(invoice_date="03/04/2026"),
            "pattern @ */header/invoice_date",
        ),
    ],
)
def test_schema_validation_rejects_broken_documents_without_leaking_values(
    world: World, mutate: Any, kind: str
) -> None:
    run_all(world)
    preds = json.loads((world.runs / RUN / "predictions.json").read_text())
    mutate(preds["test_0000"])  # an invoice with rows
    rep = predict.validate_schema(preds, json.loads(SCHEMA.read_text()))
    assert not rep["ok"] and rep["n_errors"] >= 1
    assert any(k.endswith(kind) or k.split("> ")[-1] == kind for k in rep["by_kind"]), rep
    blob = json.dumps(rep)
    assert "INV-" not in blob and "Supplier" not in blob and "03/04/2026" not in blob


@needs_schema
def test_assemble_rejects_a_schema_failure_and_withholds_the_submittable_name(
    world: World,
) -> None:
    run_all(world)
    trace_path = world.runs / RUN / "trace.jsonl"
    lines = [json.loads(ln) for ln in trace_path.read_text().splitlines()]
    lines[0]["prediction"]["doc_type"] = (
        "receipt"  # not in the enum (a bare number is now repaired)
    )
    trace_path.write_text("".join(json.dumps(t) + "\n" for t in lines), newline="\n")
    out = world.runs.parent / "sub"
    rep = assemble(world, out)
    assert not rep["ok"] and not rep["schema_ok"]
    assert (out / predict.REJECTED_NAME).is_file() and not (out / "test_predictions.json").exists()


def test_id_set_check_missing_extra_and_duplicate_ids_fail() -> None:
    exp = ["a", "b", "c"]
    assert predict.check_ids(["c", "a", "b"], exp)["ok"]
    r = predict.check_ids(["a", "b"], exp)
    assert not r["ok"] and r["n_missing"] == 1 and r["missing"] == ["c"]
    r = predict.check_ids(["a", "b", "c", "x"], exp)
    assert not r["ok"] and r["n_extra"] == 1 and r["extra"] == ["x"]
    r = predict.check_ids(["a", "b", "c", "c"], exp)
    assert not r["ok"] and r["n_duplicate"] == 1 and r["duplicate"] == ["c"]
    r = predict.check_ids(["a", "b", "c", "x", "y"], exp + ["z"])
    assert not r["ok"] and (r["n_missing"], r["n_extra"]) == (1, 2)


@needs_schema
def test_assemble_fails_on_missing_extra_and_duplicate_documents(world: World) -> None:
    run_all(world)
    trace_path = world.runs / RUN / "trace.jsonl"
    original = trace_path.read_text().splitlines()
    cases = {
        "missing": original[:-1],
        "extra": [*original, json.dumps({**json.loads(original[0]), "doc_id": "test_0777"})],
        "duplicate": [*original, original[0]],
    }
    for name, lines in cases.items():
        trace_path.write_text("".join(ln + "\n" for ln in lines), newline="\n")
        rep = assemble(world, world.runs.parent / f"sub_{name}")
        assert not rep["ok"] and not rep["checks"]["doc_ids"]["ok"], name
        assert not rep["checks"]["counts"]["ok"], name


@needs_schema
def test_validate_file_catches_duplicate_keys_in_the_json_text(
    world: World, tmp_path: Path
) -> None:
    run_all(world)
    preds = json.loads((world.runs / RUN / "predictions.json").read_text())
    good = tmp_path / "good.json"
    good.write_text(json.dumps(preds))
    assert predict.validate_file(good, SCHEMA, TEST_IDS)["ok"]
    body = json.dumps(preds)
    dup = tmp_path / "dup.json"
    dup.write_text(body[:-1] + ', "test_0000": ' + json.dumps(preds["test_0000"]) + "}")
    rep = predict.validate_file(dup, SCHEMA, TEST_IDS)
    assert (
        not rep["ok"]
        and rep["duplicate_keys"]["n"] > 0
        and "test_0000" in rep["duplicate_keys"]["keys"]
    )
    notmap = tmp_path / "list.json"
    notmap.write_text("[]")
    assert not predict.validate_file(notmap, SCHEMA, TEST_IDS)["ok"]


@pytest.mark.skipif(
    not (SCHEMA.is_file() and SAMPLE.is_file() and (ROOT / "data" / "test" / "images").is_dir()),
    reason="assignment/ or data/test/images absent (gitignored)",
)
def test_validator_accepts_the_assignment_sample_and_rejects_broken_copies(tmp_path: Path) -> None:
    """The sample is the assignment's own empty-but-valid file for the 200 real test ids (only
    the file NAMES of data/test/images are read)."""
    ids = list(predict.discover_test_docs(ROOT / "data"))
    assert len(ids) == predict.EXPECTED_DOCS
    sample = json.loads(SAMPLE.read_text(encoding="utf-8"))
    ok = tmp_path / "ok.json"
    ok.write_text(json.dumps(sample))
    assert predict.validate_file(ok, SCHEMA, ids)["ok"]
    victim = ids[3]
    broken: dict[str, Any] = {}
    missing_field = json.loads(json.dumps(sample))
    missing_field[victim]["header"].pop(next(iter(missing_field[victim]["header"])))
    broken["missing_field"] = missing_field
    wrong_type = json.loads(json.dumps(sample))
    wrong_type[victim]["header"]["currency"] = 7
    broken["wrong_type"] = wrong_type
    broken["missing_doc"] = {k: v for k, v in sample.items() if k != victim}
    broken["extra_doc"] = {**sample, "test_9999": sample[victim]}
    for name, content in broken.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(content))
        assert not predict.validate_file(path, SCHEMA, ids)["ok"], name


# ---------------------------------------------------------------------- manifest and rules


@needs_schema
def test_manifest_is_complete_and_records_no_values(world: World) -> None:
    run_all(world, batch=4)
    (world.runs / RUN / "sessions.json").write_text(
        json.dumps([{"start": "t", "end": "t", "seconds": 12.5, "exit_code": 0}])
    )
    out = world.runs.parent / "sub"
    rep = assemble(world, out, smoke_status_path=world.runs / "smoke_status.json")
    assert rep["ok"]
    m = json.loads((out / "manifest.json").read_text())
    assert m["code_sha"] == SHA and m["submission"] == f"v0_{SHA[:7]}"
    assert m["model"] == {"id": "mock", "revision": "mock"}
    assert m["config"]["hash"] == world.cfg.config_hash and m["seed"] == 42
    assert (m["batch_size"], m["batch_size_source"], m["shard"]) == (4, "manual", "0/1")
    assert m["docs"]["n_docs"] == len(TEST_IDS) and m["docs"]["n_pages"] == N_PAGES
    assert m["schema_file"]["sha256"] == predict.sha256_file(SCHEMA)
    assert m["prompt"]["version"] == "v2"
    assert set(m["dependencies"]["packages"]) == {"transformers", "torch", "xgrammar", "pycountry"}
    assert m["determinism"]["ok"] and m["determinism"]["n_docs"] == 5
    assert m["timings"]["wall_clock_s"] == 12.5 and m["timings"]["model_time_s"] > 0
    assert m["timings"]["smoke_s"] is not None
    assert m["phase3_rules"]["ladder"]["R5_provenance_aware_merge"] is True
    assert m["batch_size_contract"].startswith("not checked")
    assert sorted(p.name for p in out.iterdir()) == sorted(predict.OUT_FILES)
    for name in ("manifest.json", "validation_report.json"):  # no extracted value outside the data
        assert SENTINEL not in (out / name).read_text()
    assert SENTINEL in (out / "test_predictions.json").read_text()  # ... but the data has them


def test_phase3_switches_are_the_documented_pipeline_defaults() -> None:
    r = predict.phase3_rules()
    assert r["merge_pages"] == {
        "provenance_aware": True, "drop_header_rows": True, "drop_null_rows": True,
        "total_page_hints": None,
    }  # fmt: skip
    assert r["normalize_doc"] == {"dates": True, "numbers": True, "codes": True, "date_order": None}
    on = {k for k, v in r["ladder"].items() if v is True}
    assert on == {"R1a_dates_to_iso", "R2_number_normalisation", "R2b_code_normalisation",
                  "R5_provenance_aware_merge", "R7_repeated_header_row_filter",
                  "R8_drop_all_null_rows"}  # fmt: skip
    assert r["ladder"]["R1b_cluster_date_order"] is False
    assert r["ladder"]["R6_ocr_total_pointer"] is False  # no OCR at inference


@needs_schema
def test_recorded_rule_switches_reproduce_every_prediction(world: World) -> None:
    run_all(world)
    out = world.runs.parent / "sub"
    assert assemble(world, out)["ok"]
    manifest = json.loads((out / "manifest.json").read_text())
    rules = manifest["phase3_rules"]
    post = manifest["post_rules"]
    assert post["switches"] == {"r1": True, "r2": True, "r3": True}  # post-processing v1 default
    shapes, sha = postrules.load_slot_shapes()
    assert post["slot_shapes_sha256"] == sha
    cfg = RuleConfig(**post["switches"])
    preds = json.loads((out / "test_predictions.json").read_text())
    prov = FieldProvenance.load()
    for t in spike._read_trace(out / "trace.jsonl"):
        merged = merge_pages([p["parsed"] for p in t["pages"]], prov, **{
            "drop_null_rows": rules["merge_pages"]["drop_null_rows"],
            "provenance_aware": rules["merge_pages"]["provenance_aware"],
            "drop_header_rows": rules["merge_pages"]["drop_header_rows"],
        })  # fmt: skip
        doc, _flags = normalize_doc(merged.doc, **rules["normalize_doc"])
        doc, _changes, _skipped = postrules.apply_rules(
            doc, supplier_name=postrules.supplier_from_trace(t), ocr_text=None, shapes=shapes,
            cfg=cfg,
        )  # fmt: skip
        assert doc == preds[t["doc_id"]], t["doc_id"]


def _edit_trace(w: World) -> None:
    """Make the synthetic run give every rule something to do (R1, R2 and R3), in place."""
    path = w.runs / RUN / "trace.jsonl"
    lines = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln]
    for t in lines:
        if t["doc_id"] == "test_0002":  # a waybill: empty carrier, the model's own supplier slot
            t["prediction"]["header"]["carrier"] = None
            t["pages"][0]["parsed"]["header"]["supplier_name"] = "Acme Air Sentinel"
        if t["doc_id"] == "test_0000":  # an invoice: a po-shaped value in the cpn slot
            t["prediction"]["line_items"][0]["customer_part_number"] = "1234567"
            t["prediction"]["line_items"][0]["purchase_order"] = None
    path.write_text("".join(json.dumps(t) + "\n" for t in lines), encoding="utf-8")


def _ocr_and_shapes(tmp: Path) -> tuple[Path, Path]:
    page = ocr.PageOcr("test_0002_p1", 10, 5, "paddleocr", "line", 0, [
        ocr.OcrItem("MAWB 176-12345678", [[0, 0], [10, 0], [10, 5], [0, 5]], 0.9)])  # fmt: skip
    cache = ocr.cache_path(tmp / "ocr", ocr.DEFAULT_ENGINE, "test", "test_0002_p1")
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps(page.to_dict()), encoding="utf-8")
    shapes = tmp / "shapes.json"
    labels = [{"doc_type": "invoice", "line_items": [{"customer_part_number": "AB-12"},
                                                      {"purchase_order": "7654321"}]}]  # fmt: skip
    shapes.write_text(postrules.dump_shapes(postrules.shapes_payload(labels, "0" * 64)))
    return tmp / "ocr", shapes


@needs_schema
def test_rules_default_on_change_the_predictions_and_are_recorded(
    world: World, tmp_path: Path
) -> None:
    run_all(world)
    _edit_trace(world)
    ocr_root, shapes = _ocr_and_shapes(tmp_path)
    on, off = tmp_path / "on", tmp_path / "off"
    assert assemble(world, on, ocr_cache=ocr_root, shapes_file=shapes)["ok"]
    assert assemble(world, off, rule_cfg=RuleConfig.all_off(), ocr_cache=ocr_root)["ok"]
    a = json.loads((on / "test_predictions.json").read_text())
    b = json.loads((off / "test_predictions.json").read_text())
    assert a["test_0002"]["header"]["carrier"] == "Acme Air Sentinel"
    assert a["test_0002"]["header"]["mawb"] == "176-12345678"
    assert a["test_0000"]["line_items"][0]["purchase_order"] == "1234567"
    assert b["test_0002"]["header"]["carrier"] is None
    assert b["test_0000"]["line_items"][0]["customer_part_number"] == "1234567"
    post = json.loads((on / "manifest.json").read_text())["post_rules"]
    assert post["touched_docs"] == {"R1": 1, "R2": 1, "R3": 1}
    assert post["skipped"] == {"R2/no_ocr": 1}  # the second waybill has no cached OCR
    assert post["slot_shapes_sha256"] == predict.sha256_file(shapes)
    recs = [json.loads(ln) for ln in (on / "rules.jsonl").read_text().splitlines()]
    assert len(recs) == len(TEST_IDS)
    for name in ("rules.jsonl", "manifest.json", "validation_report.json"):
        text = (on / name).read_text()
        assert "Acme Air Sentinel" not in text and "176-12345678" not in text  # no values


@needs_schema
def test_rules_off_reproduces_the_pre_v1_output_byte_for_byte(
    world: World, tmp_path: Path
) -> None:
    run_all(world)
    _edit_trace(world)  # rules WOULD fire here, so "off == old" is not vacuous
    out = tmp_path / "off"
    assert assemble(world, out, rule_cfg=RuleConfig.all_off())["ok"]
    traces = spike._read_trace(world.runs / RUN / "trace.jsonl")
    old = {t["doc_id"]: t["prediction"] for t in sorted(traces, key=lambda t: t["doc_id"])}
    old, _ = repair_predictions(coerce_predictions(old))  # the pre-v1 assemble stage, verbatim
    expected = tmp_path / "expected.json"
    predict._write_json(expected, old)
    assert (out / "test_predictions.json").read_bytes() == expected.read_bytes()
    assert (out / "trace.jsonl").read_bytes() == (world.runs / RUN / "trace.jsonl").read_bytes()
    recs = [json.loads(ln) for ln in (out / "rules.jsonl").read_text().splitlines()]
    assert all(r["changes"] == [] for r in recs)


@needs_schema
def test_unusable_shapes_file_blocks_assembly_when_r3_is_on(world: World, tmp_path: Path) -> None:
    run_all(world)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": 99}))
    rep = assemble(world, tmp_path / "a", shapes_file=bad)
    assert not rep["ok"] and not rep["checks"]["post_rules"]["ok"]
    assert assemble(world, tmp_path / "b", shapes_file=bad, rule_cfg=RuleConfig(r3=False))["ok"]


# ---------------------------------------------------------------------- provenance refusals


@needs_schema
def test_assemble_refuses_unmerged_shard_incomplete_run_and_unreal_stack(world: World) -> None:
    run_all(world, shard="0/2")
    with pytest.raises(predict.PredictError, match="merge the shards first"):
        assemble(world, world.runs.parent / "s1", run_id=f"{RUN}_shard0of2")
    full = World(world.data, world.runs.parent / "full", world.gold, world.cfg)
    run_all(full)
    rep = assemble(full, full.runs.parent / "s2", require_stack=True)
    assert not rep["ok"] and not rep["checks"]["real_model_and_stack"]["ok"]  # mock is not real
    (full.runs / RUN / "progress.json").write_text(json.dumps({"status": "running"}))
    assert not assemble(full, full.runs.parent / "s3")["checks"]["run_complete"]["ok"]


@needs_schema
def test_assemble_checks_pin_dirty_tree_and_the_config(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_all(world)
    assert assemble(world, world.runs.parent / "a", expect_code_sha=SHA)["ok"]
    rep = assemble(world, world.runs.parent / "b", expect_code_sha="d" * 40)
    assert not rep["ok"] and not rep["checks"]["code_sha_is_pin"]["ok"]
    m = runmeta.read_manifest(world.runs / RUN)
    runmeta.write_manifest(world.runs / RUN, {**m, "code_sha": SHA + "+dirty"})
    rep = assemble(world, world.runs.parent / "c")
    assert not rep["checks"]["code_sha_clean"]["ok"]
    assert assemble(world, world.runs.parent / "d", allow_dirty=True)["checks"]["code_sha_clean"][
        "ok"
    ]
    runmeta.write_manifest(world.runs / RUN, {**m, "seed": 7})
    assert not assemble(world, world.runs.parent / "e")["checks"]["production_config"]["ok"]


@needs_schema
def test_assemble_flags_a_batch_size_that_differs_from_the_dev_run(world: World) -> None:
    run_all(world, batch=4)
    dev = _dev_run(world, 1)
    rep = assemble(world, world.runs.parent / "sub", dev_run_dir=dev)
    assert not rep["ok"] and "batch size differs" in rep["checks"]["batch_size_contract"]["detail"]
    dev4 = world.runs.parent / "dev4"
    shutil.copytree(dev, dev4)
    runmeta.write_manifest(dev4, {**runmeta.read_manifest(dev4), "batch_size": 4})
    assert assemble(world, world.runs.parent / "sub2", dev_run_dir=dev4)["ok"]


@needs_schema
def test_sample_submission_id_cross_check_is_blocking(world: World, tmp_path: Path) -> None:
    run_all(world)
    sample = tmp_path / "sample.json"
    sample.write_text(json.dumps(dict.fromkeys(TEST_IDS)))
    assert assemble(world, tmp_path / "a", sample_submission=sample)["ok"]
    sample.write_text(json.dumps(dict.fromkeys(TEST_IDS[:-1])))
    rep = assemble(world, tmp_path / "b", sample_submission=sample)
    assert not rep["ok"] and not rep["checks"]["ids_match_sample_submission"]["ok"]


# ---------------------------------------------------------------------- shards


@needs_schema
def test_two_shard_merge_assembles_to_the_unsharded_submission(
    world: World, tmp_path: Path
) -> None:
    sharded = World(world.data, tmp_path / "sharded", world.gold, world.cfg)
    for i in (0, 1):
        run_all(sharded, shard=f"{i}/2")
    shardmerge.merge_shards(
        world.cfg, TEST_IDS, "test", RUN, 2, runs_root=sharded.runs, data_root=world.data,
        expected_docs=len(TEST_IDS), expected_pages=N_PAGES,
    )  # fmt: skip
    plain = World(world.data, tmp_path / "plain", world.gold, world.cfg)
    run_all(plain)
    a, b = tmp_path / "sub_sharded", tmp_path / "sub_plain"
    ra, rb = assemble(sharded, a), assemble(plain, b)
    assert ra["ok"] and rb["ok"], ra
    for name in ("test_predictions.json", "trace.jsonl"):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
    ma = json.loads((a / "manifest.json").read_text())
    assert ma["shard"] == "merged" and len(ma["merged_from"]) == 2
    assert ma["determinism"]["n_docs"] == 10  # 5 per shard
    assert json.loads((b / "manifest.json").read_text())["determinism"]["n_docs"] == 5


@needs_schema
def test_merge_with_a_shard_that_failed_determinism_is_not_submittable(
    world: World, tmp_path: Path
) -> None:
    w = World(world.data, tmp_path / "sh", world.gold, world.cfg)
    run_all(w, shard="0/2")
    with pytest.raises(predict.DeterminismError):
        run_all(w, shard="1/2", be=FlakyBackend(world.gold, fixed_latency_s=0.5))
    shardmerge.merge_shards(world.cfg, TEST_IDS, "test", RUN, 2, runs_root=w.runs,
                            data_root=world.data)  # fmt: skip
    rep = assemble(w, tmp_path / "sub")
    assert not rep["ok"] and not rep["checks"]["determinism"]["ok"]
    assert (tmp_path / "sub" / predict.REJECTED_NAME).is_file()


# ---------------------------------------------------------------------- never in git


def test_submission_folder_inside_the_repo_is_refused_except_under_submissions() -> None:
    with pytest.raises(predict.PredictError, match="inside the repository"):
        predict.assert_outside_repo(paths.REPO_ROOT / "reports" / "x")
    with pytest.raises(predict.PredictError, match="inside the repository"):
        predict.assert_outside_repo(paths.REPO_ROOT)
    predict.assert_outside_repo(paths.REPO_ROOT / "submissions" / "v0_abc1234")
    predict.assert_outside_repo(ROOT.parent / "elsewhere")


def test_assemble_never_writes_into_the_repo_tree(world: World) -> None:
    run_all(world)
    with pytest.raises(predict.PredictError, match="inside the repository"):
        assemble(world, paths.REPO_ROOT / "reports" / "oops")
    assert not (paths.REPO_ROOT / "reports" / "oops").exists()


def test_submissions_folder_is_gitignored() -> None:
    if shutil.which("git") is None:
        pytest.skip("git not available")
    for rel in ("submissions/x", "submissions/v0_abc1234/test_predictions.json"):
        res = subprocess.run(["git", "check-ignore", "-q", rel], cwd=ROOT, check=False)
        assert res.returncode == 0, f"{rel} is not ignored"
    res = subprocess.run(["git", "check-ignore", "-q", "reports/x.md"], cwd=ROOT, check=False)
    assert res.returncode == 1  # the check is not vacuous: other paths are not ignored


def test_submissions_dir_is_a_configured_path_in_every_profile() -> None:
    assert "SHIPDOC_SUBMISSIONS_DIR" in paths.PATH_KEYS
    assert paths._profile_defaults("colab")["SHIPDOC_SUBMISSIONS_DIR"].endswith(
        "/MyDrive/shipdoc-extract/submissions"
    )
    assert paths._profile_defaults("local")["SHIPDOC_SUBMISSIONS_DIR"] == "submissions"


# ---------------------------------------------------------------------- CLI


def test_cli_registers_the_predict_stages() -> None:
    parser = cli.build_parser()
    a = parser.parse_args(["predict", "plan", "--shard", "1/2"])
    assert (a.command, a.stage, a.shard) == ("predict", "plan", "1/2")
    a = parser.parse_args(["predict", "run", "--config", "c.yaml", "--run-id", "r",
                           "--decision", "d.json", "--smoke-status", "s.json"])  # fmt: skip
    assert a.stage == "run" and a.shard == "0/1" and a.backend == "hf"
    with pytest.raises(SystemExit):
        parser.parse_args(["predict"])  # a stage is required


def test_cli_plan_prints_counts_only(world: World, capsys: pytest.CaptureFixture[str]) -> None:
    args = cli.build_parser().parse_args(
        ["predict", "plan", "--data-root", str(world.data), "--shard", "0/2"]
    )
    assert predict.main_predict(args) == 0
    out = capsys.readouterr().out
    assert f"{len(TEST_IDS)} docs / {N_PAGES} pages" in out


def test_cli_refusal_is_exit_1_with_a_message(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    args = cli.build_parser().parse_args(
        ["predict", "plan", "--data-root", str(world.data / "nope")]
    )
    assert predict.main_predict(args) == 1
    assert "REFUSED" in capsys.readouterr().err
