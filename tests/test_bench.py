"""The batch-size bench: page selection, agreement metrics, the selection rule, a mocked run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from _synth import make_corpus

from shipdoc import bench, cli, runmeta, spike
from shipdoc.extract import MockBackend, parse_output

ROOT = Path(__file__).resolve().parents[1]
KEYED = ROOT / "configs" / "spike_qwen35_4b_img_only.yaml"
GIB = 2**30
needs_scorer = pytest.mark.skipif(
    not (ROOT / "assignment" / "score.py").is_file(), reason="assignment/score.py absent"
)


# ---------------------------------------------------------------------- page selection


def _m(doc: str, scanned: bool, multi: bool, waybill: bool = False, rep: bool = False,
       ill: bool = False) -> dict[str, Any]:  # fmt: skip
    return {"doc_id": doc, "scanned": scanned, "multipage": multi, "waybill": waybill,
            "repeated_parts": rep, "illegible": ill}  # fmt: skip


def _meta() -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows = [
        _m("d00", True, True), _m("d01", True, True, rep=True), _m("d02", True, True),
        _m("d03", False, True), _m("d04", False, True, rep=True), _m("d05", False, True),
        _m("d06", True, False), _m("d07", False, False), _m("d08", True, False, waybill=True),
        _m("d09", True, False), _m("d10", False, False), _m("d11", False, False, waybill=True),
        _m("d12", True, False), _m("d13", False, False), _m("d14", True, False, ill=True),
    ]  # fmt: skip
    pages = {r["doc_id"]: (3 if r["multipage"] else 1) for r in rows}
    return rows, pages


def test_pick_bench_docs_is_exactly_12_pages_mixed_and_deterministic() -> None:
    meta, pages = _meta()
    picked = bench.pick_bench_docs(meta, pages)
    assert sum(pages[d] for d in picked) == 12
    assert list(picked) == sorted(picked)  # dev order
    groups = list(picked.values())
    assert groups.count("multi_scanned") == 1 and groups.count("multi_digital") == 1
    # 8 pages for multipage docs: one 3-page doc of each kind, a second would not fit (6+3 > 8)
    assert {"single_scanned_invoice", "single_digital_invoice", "single_waybill"} <= set(groups)
    assert bench.pick_bench_docs(meta, pages) == picked
    assert "d14" not in picked  # illegible docs are skipped
    # repeated_parts (the long tables) are preferred within a group
    assert "d01" in picked and "d04" in picked


def test_pick_bench_docs_fails_loudly_when_the_split_is_too_small() -> None:
    meta, pages = _meta()
    with pytest.raises(ValueError, match="cannot fill"):
        bench.pick_bench_docs(meta[:8], pages)


def test_committed_bench12_list_matches_the_selection_rule() -> None:
    ids = json.loads((ROOT / "splits" / "bench12.json").read_text(encoding="utf-8"))
    assert len(ids) == len(set(ids)) and all(d.startswith("dev_") for d in ids)  # no test pages
    dev = ROOT / "data" / "dev" / "images"
    if not dev.is_dir():
        pytest.skip("data/dev absent (gitignored)")
    meta = json.loads((ROOT / "meta" / "dev.json").read_text(encoding="utf-8"))
    counts = {
        m["doc_id"]: len(spike.doc_page_images("dev", m["doc_id"], ROOT / "data")) for m in meta
    }
    assert list(bench.pick_bench_docs(meta, counts)) == ids
    assert sum(counts[d] for d in ids) == 12
    by_id = {m["doc_id"]: m for m in meta}
    assert any(by_id[d]["multipage"] for d in ids)
    assert {by_id[d]["scanned"] for d in ids} == {True, False}
    assert any(by_id[d]["waybill"] for d in ids)


# ---------------------------------------------------------------------- agreement


def _trace(doc: str, raws: list[str]) -> dict[str, Any]:
    pages = []
    for i, raw in enumerate(raws):
        parsed = json.loads(raw)
        pages.append({"page": i + 1, "raw_text": raw, "parsed": parsed})
    return {"doc_id": doc, "pages": pages, "prediction": {}}


RAW = json.dumps(
    {"doc_type": "invoice", "header": {"a": "1", "b": None},
     "line_items": [{"q": "5"}, {"q": "6"}], "page_kind": "single"}
)  # fmt: skip


def test_page_agreement_counts_pages_and_fields() -> None:
    ref = [_trace("d", [RAW, RAW])]
    same = bench.page_agreement(ref, [_trace("d", [RAW, RAW])])
    assert same["byte_identical_rate"] == 1.0 and same["field_agreement_rate"] == 1.0
    assert same["n_fields"] == 2 * (2 + 2 + 2)  # header a,b + doc_type/page_kind + 2 rows

    one_field = RAW.replace('"a": "1"', '"a": "2"')
    got = bench.page_agreement(ref, [_trace("d", [RAW, one_field])])
    assert got["byte_identical_rate"] == 0.5 and got["byte_identical_pages"] == 1
    assert got["field_agreement_rate"] == pytest.approx(1 - 1 / got["n_fields"])

    fewer_rows = json.dumps({**json.loads(RAW), "line_items": [{"q": "5"}]})
    got = bench.page_agreement(ref, [_trace("d", [RAW, fewer_rows])])
    assert got["field_agreement_rate"] == pytest.approx(1 - 1 / got["n_fields"])  # lost field

    good = {"page": 2, "raw_text": RAW, "parsed": json.loads(RAW)}
    junk = {"doc_id": "d", "pages": [{"page": 1, "raw_text": "{", "parsed": None}, good]}
    got = bench.page_agreement(ref, [junk])
    assert got["byte_identical_rate"] == 0.5 and got["field_agreement_rate"] == 0.5


def test_page_agreement_rejects_different_page_sets_and_empty_input() -> None:
    with pytest.raises(ValueError, match="same pages"):
        bench.page_agreement([_trace("d", [RAW])], [_trace("e", [RAW])])
    with pytest.raises(ValueError, match="no pages"):
        bench.page_agreement([], [])


def test_scorer_identity_fails_closed_without_gold() -> None:
    assert bench.scorer_identical([_trace("d", [RAW])], [_trace("d", [RAW])], None) is False
    assert bench.scorer_identical([_trace("d", [RAW])], [_trace("e", [RAW])], {"d": {}}) is False


# ---------------------------------------------------------------------- the selection rule


def _r(b: int, *, ok: bool = True, vram: float | None = 8.0, byte: float = 1.0,
       field: float = 1.0, scorer: bool = True) -> dict[str, Any]:  # fmt: skip
    return {
        "batch_size": b, "ok": ok, "peak_vram_bytes": None if vram is None else int(vram * GIB),
        "byte_identical_rate": byte, "field_agreement_rate": field, "scorer_identical": scorer,
        "error": None if ok else "OutOfMemoryError",
    }  # fmt: skip


def test_rule_step_one_takes_the_largest_byte_identical_size_within_vram() -> None:
    d = bench.choose_batch_size([_r(1), _r(2), _r(4), _r(8)])
    assert d["chosen"] == 8 and d["deviation"] is None
    d = bench.choose_batch_size([_r(1), _r(2), _r(4), _r(8, vram=14.51)])
    assert d["chosen"] == 4 and "14.5 GiB" in d["reasons"]["8"]
    d = bench.choose_batch_size([_r(1), _r(2), _r(4, vram=14.5), _r(8, ok=False)])
    assert d["chosen"] == 4 and d["reasons"]["8"].startswith("failed")
    d = bench.choose_batch_size([_r(1), _r(2), _r(4, byte=0.9167), _r(8, byte=0.5)])
    assert d["chosen"] == 2 and d["deviation"] is None  # a non-identical 4 does not block a clean 2


def test_rule_step_two_accepts_near_identical_with_the_same_score_and_flags_a_deviation() -> None:
    results = [_r(1), _r(2, byte=0.9167, field=0.9990), _r(4, byte=0.9167, field=0.9950),
               _r(8, byte=0.6, field=0.9949)]  # fmt: skip
    d = bench.choose_batch_size(results)
    assert d["chosen"] == 4  # >= 99.5% is inclusive; 8 misses by a hair
    dev = d["deviation"]
    assert dev["kind"] == "not_byte_identical" and dev["batch_size"] == 4
    assert (
        "DEVIATION" in dev["text"]
        and "99.5%" in dev["text"]
        and "not byte-identical" in d["reasons"]["4"]
    )
    assert "rejected" in d["reasons"]["8"]
    # the scorer must also agree, and VRAM still binds in step two
    d = bench.choose_batch_size([_r(1), _r(2, byte=0.9, field=0.999, scorer=False)])
    assert d["chosen"] == 1 and d["deviation"] is None
    d = bench.choose_batch_size([_r(1), _r(2, byte=0.9, field=0.999, vram=15.0)])
    assert d["chosen"] == 1


def test_rule_step_three_is_batch_one_when_everything_fails() -> None:
    both = [_r(1), _r(2, byte=0.5, field=0.9), _r(4, ok=False), _r(8, byte=0.9, field=0.99)]
    d = bench.choose_batch_size(both)
    assert d["chosen"] == 1 and d["deviation"] is None and set(d["reasons"]) == {"2", "4", "8"}
    assert bench.choose_batch_size([_r(1)])["chosen"] == 1


def test_an_unmeasured_peak_vram_counts_as_failing_not_as_passing() -> None:
    d = bench.choose_batch_size([_r(1), _r(2, vram=None)])
    assert d["chosen"] == 1 and "not measured" in d["reasons"]["2"]


def test_the_rule_text_is_the_one_in_the_task() -> None:
    assert bench.SELECTION_RULE.startswith("The largest batch size with 100% byte-identical")
    assert (
        "<= 14.5 GiB" in bench.SELECTION_RULE and ">= 99.5% field agreement" in bench.SELECTION_RULE
    )
    assert bench.SELECTION_RULE.endswith("If both fail, use batch 1.")
    assert int(14.5 * GIB) == bench.VRAM_LIMIT_BYTES


# ---------------------------------------------------------------------- a mocked bench run


class BenchMock(MockBackend):
    """Reports a VRAM peak that grows with the batch and can damage one value per batch."""

    def __init__(self, *a: Any, vram_gib_per_page: float = 1.0, damage_sizes: tuple[int, ...] = (),
                 damage: tuple[str, str] = ("Supplier", "Suplier"), **k: Any) -> None:  # fmt: skip
        super().__init__(*a, fixed_latency_s=0.5, **k)
        self.vram, self.damage_sizes, self.damage = vram_gib_per_page, damage_sizes, damage
        self.damaged: set[int] = set()

    def extract_pages(self, requests: Any) -> Any:
        out = super().extract_pages(requests)
        n = len(requests)
        for k, (raw, _parsed, meta) in enumerate(out):
            meta["peak_vram_bytes"] = int(self.vram * n * GIB)
            if n in self.damage_sizes and n not in self.damaged and self.damage[0] in raw:
                raw = raw.replace(*self.damage, 1)  # once per batch size: the first page with it
                out[k] = (raw, parse_output(raw, self.output_format), meta)
                self.damaged.add(n)
        return out

    def extract_page(self, *a: Any, **k: Any) -> Any:
        raw, parsed, meta = super().extract_page(*a, **k)
        meta["peak_vram_bytes"] = int(self.vram * GIB)
        return raw, parsed, meta


BENCH_DOCS = ["dev_0000", "dev_0001", "dev_0003", "dev_0005", "dev_0009", "dev_0002"]


def _run_bench(tmp: Path, backend_kw: dict[str, Any], sizes: tuple[int, ...] = (1, 2, 4, 8)) -> Any:
    root = tmp / "data"
    gold = make_corpus(root, labels=True)
    return bench.run_bench(
        spike.load_config(KEYED), BenchMock(gold, **backend_kw), BENCH_DOCS, tmp / "bench",
        sizes, data_root=root,
    )  # fmt: skip


@needs_scorer
def test_run_bench_picks_the_largest_clean_size_and_writes_the_result(tmp_path: Path) -> None:
    result = _run_bench(tmp_path, {"vram_gib_per_page": 1.0})  # 8 pages * 1 GiB = 8 GiB peak
    assert result["chosen_batch_size"] == 8 and result["deviation"] is None
    assert result["docs"] == BENCH_DOCS and result["n_pages"] == 14
    by = {r["batch_size"]: r for r in result["results"]}
    assert set(by) == {1, 2, 4, 8}
    assert all(r["byte_identical_rate"] == 1.0 and r["scorer_identical"] for r in by.values())
    assert all(r["pages_per_hour"] > 0 for r in by.values())
    # fixed 0.5 s per call: a call of N pages costs the same, so pages/hour grows with N
    assert by[1]["pages_per_hour"] < by[2]["pages_per_hour"] < by[4]["pages_per_hour"]
    assert by[8]["peak_vram_bytes"] == pytest.approx(8 * GIB, rel=0.01) and by[1]["ok"]
    assert result["rule"] == bench.SELECTION_RULE
    on_disk = json.loads((tmp_path / "bench" / "bench_result.json").read_text())
    assert on_disk == result
    banner = bench.format_banner(result)
    assert "CHOSEN BATCH SIZE: 8" in banner and "TEST submission" in banner
    assert "DEVIATION" not in banner and "byte-identical" in banner


@needs_scorer
def test_run_bench_vram_limit_and_failed_sizes_cap_the_choice(tmp_path: Path) -> None:
    result = _run_bench(tmp_path, {"vram_gib_per_page": 2.0})  # 4 -> 8 GiB ok, 8 -> 16 GiB over
    assert result["chosen_batch_size"] == 4
    assert "14.5 GiB" in result["reasons"]["8"]


@needs_scorer
def test_run_bench_logs_a_deviation_for_a_near_identical_size(tmp_path: Path) -> None:
    # One stray space in one value for each of the sizes 2, 4, 8: not byte-identical, 1 of ~390
    # fields differs, and the scorer normalises whitespace.
    result = _run_bench(tmp_path, {"damage_sizes": (2, 4, 8), "damage": ('Ltd"', 'Ltd "')})
    by = {r["batch_size"]: r for r in result["results"]}
    assert all(by[b]["byte_identical_rate"] < 1.0 and by[b]["scorer_identical"] for b in (2, 4, 8))
    assert all(by[b]["field_agreement_rate"] >= 0.995 for b in (2, 4, 8))
    assert result["chosen_batch_size"] == 8  # the largest that clears 99.5%
    assert result["deviation"]["batch_size"] == 8 and "DEVIATION" in result["deviation"]["text"]
    banner = bench.format_banner(result)
    assert (
        "DEVIATION: batch size 8 is NOT byte-identical" in banner
        and "CHOSEN BATCH SIZE: 8" in banner
    )


@needs_scorer
def test_run_bench_falls_back_to_batch_one_when_the_scorer_sees_the_difference(
    tmp_path: Path,
) -> None:
    result = _run_bench(tmp_path, {"damage_sizes": (2, 4, 8)})  # a typo in a supplier name: scored
    by = {r["batch_size"]: r for r in result["results"]}
    assert all(
        by[b]["byte_identical_rate"] < 1.0 and not by[b]["scorer_identical"] for b in (2, 4, 8)
    )
    assert result["chosen_batch_size"] == 1 and result["deviation"] is None


@needs_scorer
def test_run_bench_records_an_oom_size_as_failed_and_goes_on(tmp_path: Path) -> None:
    class Oom(BenchMock):
        def extract_pages(self, requests: Any) -> Any:
            if len(requests) >= 8:
                raise RuntimeError("CUDA out of memory")
            return super().extract_pages(requests)

    root = tmp_path / "data"
    gold = make_corpus(root, labels=True)
    result = bench.run_bench(
        spike.load_config(KEYED), Oom(gold), BENCH_DOCS, tmp_path / "bench", data_root=root
    )
    by = {r["batch_size"]: r for r in result["results"]}
    assert by[8]["ok"] is False and "out of memory" in by[8]["error"]
    assert result["chosen_batch_size"] == 4 and "out of memory" in bench.format_banner(result)


def test_without_gold_the_bench_cannot_use_step_two_and_still_runs(tmp_path: Path) -> None:
    root = tmp_path / "data"
    gold = make_corpus(root, labels=False)
    result = bench.run_bench(
        spike.load_config(KEYED), BenchMock(gold), BENCH_DOCS, tmp_path / "bench", data_root=root
    )
    assert result["chosen_batch_size"] == 8  # byte-identical needs no scorer
    assert all(r["scorer_identical"] in (True, False) for r in result["results"])


def test_the_bench_needs_batch_size_one_as_reference(tmp_path: Path) -> None:
    root = tmp_path / "data"
    gold = make_corpus(root, labels=False)
    with pytest.raises(ValueError, match="reference"):
        bench.run_bench(spike.load_config(KEYED), BenchMock(gold), BENCH_DOCS, tmp_path / "b",
                        (2, 4), data_root=root)  # fmt: skip


# ---------------------------------------------------------------------- result file -> run


def _result(**over: Any) -> dict[str, Any]:
    cfg = spike.load_config(KEYED)
    base = {
        "schema": bench.BENCH_SCHEMA, "code_sha": spike.git_commit(),
        "config_hash": cfg.config_hash,
        "config": cfg.name, "model": {"id": "mock", "revision": "mock"}, "seed": 42, "docs": [],
        "n_pages": 12, "logprobs": True, "batch_sizes": [1, 2], "vram_limit_bytes": 1,
        "min_field_agreement": 0.995, "rule": bench.SELECTION_RULE, "results": [],
        "chosen_batch_size": 4, "deviation": None, "reasons": {},
    }  # fmt: skip
    return {**base, **over}


def test_a_bench_result_only_applies_to_the_code_config_and_model_it_was_measured_on(
    tmp_path: Path,
) -> None:
    cfg, backend = spike.load_config(KEYED), MockBackend({})
    path = tmp_path / "bench_result.json"
    path.write_text(json.dumps(_result()))
    assert bench.chosen_from_result(bench.load_bench_result(path, cfg, backend)) == 4
    for over, msg in (
        ({"config_hash": "0" * 16}, "config hash"),
        ({"model": {"id": "x", "revision": "other"}}, "model revision"),
        ({"code_sha": "elsewhere"}, "code sha"),
        ({"chosen_batch_size": 0}, "chosen_batch_size"),
        ({"schema": 99}, "schema"),
    ):
        path.write_text(json.dumps(_result(**over)))
        with pytest.raises(ValueError, match=msg):
            bench.load_bench_result(path, cfg, backend)


def test_spike_cli_uses_the_bench_choice_and_records_it_in_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "data"
    gold = make_corpus(root, labels=False)
    monkeypatch.setenv("SHIPDOC_DATA_DIR", str(root))
    monkeypatch.setenv("SHIPDOC_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(
        spike, "make_backend", lambda kind, cfg, split: MockBackend(gold, fixed_latency_s=0.5)
    )
    docs = tmp_path / "docs.json"
    docs.write_text(json.dumps(["dev_0000", "dev_0001", "dev_0002"]))
    res = tmp_path / "bench_result.json"
    dev = {"kind": "not_byte_identical", "batch_size": 4, "text": "DEVIATION: x"}
    res.write_text(json.dumps(_result(deviation=dev)))
    base = ["spike", "--config", str(KEYED), "--docs", str(docs), "--split", "dev",
            "--backend", "mock", "--bench-result", str(res)]  # fmt: skip
    assert cli.main([*base, "--run-id", "b"]) == 0
    m = runmeta.read_manifest(tmp_path / "runs" / "b")
    assert m is not None and m["batch_size"] == 4 and m["batch_size_source"] == "bench"
    assert m["bench"] == {"path": str(res), "chosen": 4, "deviation": dev}
    with pytest.raises(ValueError, match="contradicts the bench"):
        cli.main([*base, "--run-id", "c", "--batch-size", "2"])
    assert cli.main([*base, "--run-id", "d", "--batch-size", "4"]) == 0
    res.write_text(json.dumps(_result(code_sha="elsewhere")))
    with pytest.raises(ValueError, match="does not apply"):
        cli.main([*base, "--run-id", "e"])


@needs_scorer
def test_bench_cli_runs_reuses_and_refuses_a_stale_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "data"
    gold = make_corpus(root, labels=True)
    monkeypatch.setenv("SHIPDOC_DATA_DIR", str(root))
    monkeypatch.setenv("SHIPDOC_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(
        bench, "make_backend", lambda kind, cfg, split: BenchMock(gold, vram_gib_per_page=1.0)
    )
    docs = tmp_path / "docs.json"
    docs.write_text(json.dumps(BENCH_DOCS))
    out = tmp_path / "bench"
    args = ["bench", "--config", str(KEYED), "--docs", str(docs), "--out-dir", str(out),
            "--backend", "mock", "--reuse"]  # fmt: skip
    assert cli.main(args) == 0
    first = capsys.readouterr().out
    assert "reusing" not in first, first
    assert "CHOSEN BATCH SIZE: 8" in first, first
    stamp = (out / "bench_result.json").read_bytes()
    assert cli.main(args) == 0  # a valid result is reused, not measured again
    assert "bench: reusing" in capsys.readouterr().out
    assert (out / "bench_result.json").read_bytes() == stamp
    monkeypatch.setattr(bench, "git_commit", lambda: "another-commit")  # new code: bench again
    assert cli.main(args) == 0
    captured = capsys.readouterr()
    assert "not reusing" in captured.err and "CHOSEN BATCH SIZE" in captured.out
    with pytest.raises(SystemExit):  # neither --config/--out-dir nor --write-list
        cli.main(["bench"])
