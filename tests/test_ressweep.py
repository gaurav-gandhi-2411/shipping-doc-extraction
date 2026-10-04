"""shipdoc.ressweep: visual-token arithmetic, the decision rule, the strict control-reuse check,
the paired count bootstrap, the misread count and the analysis on synthetic runs (CPU only).

No model, GPU or network. The synthetic corpus of tests/_synth.py carries the gold; predictions
are copies of it with controlled errors. Tests that need the official scorer skip without it.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _synth  # noqa: E402

from shipdoc import eval as ev  # noqa: E402
from shipdoc import ressweep as rs  # noqa: E402
from shipdoc.runmeta import docs_sha  # noqa: E402
from shipdoc.spike import load_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GIB = 2**30
needs_scorer = pytest.mark.skipif(
    not (ROOT / "assignment" / "score.py").is_file(), reason="assignment/score.py absent"
)

# ---------------------------------------------------------------------- visual-token arithmetic


def test_tokens_of_the_production_cap_and_the_native_page() -> None:
    assert rs.visual_tokens(1_310_720) == 1260  # the production config; spike40 trace: 1260.0
    assert rs.smart_resize(1754, 1240, 1_310_720) == (1344, 960)
    assert rs.visual_tokens(1_843_200) == 1750
    assert rs.visual_tokens(2_196_480) == 2145  # the native page, 1760 x 1248
    assert rs.visual_tokens(1_048_576) == 988  # 1024 * 32 * 32, quoted in the production config


def test_a_cap_above_the_native_page_changes_nothing() -> None:
    """The reason 2,500 / 3,500 tokens are unreachable: the processor never upscales."""
    native = rs.visual_tokens(2_196_480)
    assert rs.visual_tokens(2_560_000) == native == rs.visual_tokens(3_584_000)
    assert rs.visual_tokens(2**31) == native


def test_smart_resize_equals_transformers() -> None:
    pytest.importorskip("transformers")
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize as ref

    for hw in [(1754, 1240), (1240, 1754), (600, 800), (100, 3000), (40, 40)]:
        for mp in [200_000, 1_048_576, 1_310_720, 1_843_200, 2_196_480, 3_584_000]:
            assert rs.smart_resize(*hw, mp) == ref(
                *hw, factor=32, min_pixels=rs.MIN_PIXELS, max_pixels=mp
            ), (hw, mp)


def test_check_reachable_accepts_the_sweep_and_refuses_a_noop_resolution() -> None:
    rows = rs.check_reachable([px for px, _ in rs.RESOLUTIONS])
    assert [r[2] for r in rows] == [1260, 1750, 2145]
    with pytest.raises(ValueError, match="strictly increasing.*never upscales"):
        rs.check_reachable([1_310_720, 2_560_000, 3_584_000])  # 2145 twice
    with pytest.raises(ValueError, match="strictly increasing"):
        rs.check_reachable([2_196_480, 1_310_720])


def test_sweep_configs_differ_from_production_only_in_name_and_max_pixels() -> None:
    prod = load_config(ROOT / "configs" / "spike_qwen35_4b_img_only.yaml")
    assert rs.RESOLUTIONS[0] == (prod.backend.max_pixels, "qwen35_4b_img_only")
    assert prod.backend.max_pixels == rs.PRODUCTION_MAX_PIXELS
    for px, name in rs.RESOLUTIONS[1:]:
        cfg = load_config(ROOT / "configs" / f"spike_{name}.yaml")
        assert cfg.raw["max_pixels"] == px and cfg.raw["name"] == name
        a = {k: v for k, v in prod.raw.items() if k not in ("name", "max_pixels")}
        b = {k: v for k, v in cfg.raw.items() if k not in ("name", "max_pixels")}
        assert a == b  # same model revision, prompt, output format, seed, max_new_tokens
        assert cfg.config_hash != prod.config_hash  # distinct runs never look like the control


# ---------------------------------------------------------------------- decision rule


def cand(px: int, lo: float | None, delta: float | None = None, vram_gib: float | None = 10.0,
         state: str = "complete") -> rs.Candidate:  # fmt: skip
    return rs.Candidate(
        px, state, delta if delta is not None else lo,
        lo, None if vram_gib is None else int(vram_gib * GIB),
    )  # fmt: skip


def test_decide_adopts_the_highest_qualifying_resolution() -> None:
    d = rs.decide([cand(1000, 0.01, 0.02), cand(2000, 0.005, 0.03), cand(3000, 0.001, 0.05)])
    assert d.adopted == 3000 and d.label == "3000"
    assert d.reasons[3000] == "ADOPTED" and "higher resolution also qualifies" in d.reasons[1000]


def test_decide_skips_a_higher_resolution_whose_ci_includes_zero() -> None:
    d = rs.decide([cand(1000, 0.01, 0.02), cand(2000, -0.002, 0.04)])
    assert d.adopted == 1000 and "not > 0" in d.reasons[2000]
    assert rs.decide([cand(1000, 0.0, 0.03)]).adopted is None  # a lower bound of exactly 0 fails


def test_decide_keeps_the_control_when_nothing_qualifies() -> None:
    d = rs.decide([cand(1000, -0.01, 0.02), cand(2000, 0.0, 0.01)])
    assert d.adopted is None and d.label == "control"
    assert rs.decide([]).label == "control"


def test_decide_vram_limit_is_inclusive_at_14_5_gib() -> None:
    assert rs.decide([cand(2000, 0.01, vram_gib=14.5)]).adopted == 2000
    over = rs.decide([cand(1000, 0.01, vram_gib=10), cand(2000, 0.05, vram_gib=14.51)])
    assert over.adopted == 1000 and "14.5 GiB" in over.reasons[2000]


def test_decide_ties_on_the_point_estimate_go_to_the_lower_resolution() -> None:
    d = rs.decide([cand(1000, 0.01, 0.03), cand(2000, 0.02, 0.03), cand(3000, 0.015, 0.03)])
    assert d.adopted == 1000 and d.reasons[1000] == "ADOPTED (tie: lowest)"
    assert "tie" in d.reasons[3000]
    # not a tie: a lower resolution with a smaller estimate does not beat the highest
    assert rs.decide([cand(1000, 0.01, 0.02), cand(2000, 0.01, 0.03)]).adopted == 2000
    # a tie only among the top: the lower resolution with an equal estimate wins
    d2 = rs.decide([cand(1000, 0.01, 0.01), cand(2000, 0.02, 0.05), cand(3000, 0.02, 0.05)])
    assert d2.adopted == 2000


def test_decide_excludes_oom_failed_and_unmeasured_runs_never_silently() -> None:
    d = rs.decide([
        cand(1000, 0.01, 0.02),
        rs.Candidate(2000, "oom"),
        rs.Candidate(3000, "failed"),
        cand(4000, None, None),  # CI not measured
        cand(5000, 0.05, 0.06, vram_gib=None),  # peak VRAM not measured: a failed check
    ])  # fmt: skip
    assert d.adopted == 1000
    assert d.reasons[2000] == "excluded: run oom" and d.reasons[3000] == "excluded: run failed"
    assert "CI not measured" in d.reasons[4000] and "VRAM not measured" in d.reasons[5000]
    assert set(d.reasons) == {1000, 2000, 3000, 4000, 5000}  # every candidate has a verdict


def test_decide_does_not_depend_on_input_order() -> None:
    cs = [cand(1000, 0.01, 0.02), cand(2000, 0.02, 0.03), cand(3000, -0.1, 0.0)]
    assert rs.decide(cs).adopted == rs.decide(cs[::-1]).adopted == 2000


# ---------------------------------------------------------------------- strict control reuse

DOCS = ["dev_0001", "dev_0006", "dev_0007"]
EXPECTED: dict[str, Any] = {
    "config_hash": "abc123", "code_sha": "a" * 40, "model_revision": "r" * 40,
    "doc_ids": DOCS, "batch_size": 1,
}  # fmt: skip


def good_manifest() -> dict[str, Any]:
    return {
        "config": {"name": "qwen35_4b_img_only", "hash": "abc123"}, "code_sha": "a" * 40,
        "model": {"id": "m", "revision": "r" * 40}, "batch_size": 1, "docs_sha": docs_sha(DOCS),
        "shard": "0/1", "logprobs": True, "output_format": "json",
    }  # fmt: skip


def test_control_reuse_accepts_an_exact_match() -> None:
    assert rs.check_control(good_manifest(), {"status": "complete"}, DOCS[::-1], EXPECTED) == []


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [
        (lambda m: m["config"].update(hash="zzz"), "config hash"),
        (lambda m: m.update(code_sha="b" * 40), "code SHA"),
        (lambda m: m.update(code_sha="a" * 40 + "+dirty"), "code SHA"),
        (lambda m: m["model"].update(revision="q" * 40), "model revision"),
        (lambda m: m.update(batch_size=2), "batch size"),
        (lambda m: m.update(docs_sha="0" * 16), "doc list sha"),
        (lambda m: m.update(shard="0/2"), "shard"),
        (lambda m: m.update(logprobs=False), "logprobs"),
        (lambda m: m.update(output_format="compact"), "output format"),
    ],
)
def test_control_reuse_refuses_every_mismatch(mutate: Any, fragment: str) -> None:
    m = good_manifest()
    mutate(m)
    why = rs.check_control(m, {"status": "complete"}, DOCS, EXPECTED)
    assert any(fragment in w for w in why), why


def test_control_reuse_refuses_missing_manifest_incomplete_run_and_wrong_documents() -> None:
    assert "no manifest" in rs.check_control(None, {"status": "complete"}, DOCS, EXPECTED)[0]
    why = rs.check_control(good_manifest(), {"status": "running"}, DOCS, EXPECTED)
    assert any("progress status" in w for w in why)
    assert rs.check_control(good_manifest(), None, DOCS, EXPECTED)  # no progress file at all
    why = rs.check_control(good_manifest(), {"status": "complete"}, DOCS[:2], EXPECTED)
    assert any("documents" in w for w in why)
    why = rs.check_control(good_manifest(), {"status": "complete"}, [*DOCS, "dev_0099"], EXPECTED)
    assert any("documents" in w for w in why)


def test_the_500_document_zero_shot_run_is_not_a_reusable_control() -> None:
    m = good_manifest()
    m["docs_sha"] = docs_sha([f"dev_{i:04d}" for i in range(500)])
    assert rs.check_control(m, {"status": "complete"}, DOCS, EXPECTED)


# ---------------------------------------------------------------------- count bootstrap


def test_paired_count_delta_identical_and_uniform_improvement() -> None:
    same = rs.paired_count_delta([1, 0, 2, 0], [1, 0, 2, 0])
    assert (same["delta"], same["lo"], same["hi"]) == (0.0, 0.0, 0.0)
    better = rs.paired_count_delta([1, 1, 1, 1], [0, 0, 0, 0])
    assert better["delta"] == -4.0 and better["lo"] == better["hi"] == -4.0
    assert (better["control"], better["candidate"]) == (4.0, 0.0)


def test_paired_count_delta_is_seeded_and_ci_brackets_the_point_estimate() -> None:
    a, b = [3, 0, 1, 2, 0, 1, 4, 0], [1, 0, 1, 0, 0, 2, 1, 0]
    r1, r2 = rs.paired_count_delta(a, b), rs.paired_count_delta(a, b)
    assert r1 == r2
    assert r1["delta"] == sum(b) - sum(a) == -6
    assert r1["lo"] <= r1["delta"] <= r1["hi"]
    import numpy as np

    idx = ev._rng(42).integers(0, len(a), size=(rs.N_BOOT, len(a)))  # paired_bootstrap's draw
    d = np.asarray(b, float)[idx].sum(axis=1) - np.asarray(a, float)[idx].sum(axis=1)
    assert (r1["lo"], r1["hi"]) == tuple(float(x) for x in np.quantile(d, [0.025, 0.975]))
    with pytest.raises(ValueError, match="same length"):
        rs.paired_count_delta([1, 2], [1])
    with pytest.raises(ValueError, match="non-empty"):
        rs.paired_count_delta([], [])


# ---------------------------------------------------------------------- misreads


@needs_scorer
def test_part_number_misread_count_follows_the_diagnosis_definition(tmp_path: Path) -> None:
    gold = _synth.make_corpus(tmp_path)
    g = gold["dev_0001"]
    sc, rd = ev.load_scorer(), rs.load_row_diagnosis()

    def count(pred: dict[str, Any], gold_doc: dict[str, Any] = g) -> int:
        trace = {"pages": [{"parsed": {"line_items": pred["line_items"]}}]}
        return rs.doc_spn_misreads(pred, gold_doc, trace, None, sc, rd)

    assert count(copy.deepcopy(g)) == 0  # perfect
    p = copy.deepcopy(g)
    p["line_items"][1]["supplier_part_number"] = "P-1-X"  # near misread, row linked by its PO
    assert count(p) == 1
    p2 = copy.deepcopy(g)
    p2["line_items"][1]["supplier_part_number"] = "COMPLETELY-OTHER-9999"  # not a near misread
    assert count(p2) == 0
    p3 = copy.deepcopy(g)
    p3["line_items"][1]["supplier_part_number"] = "P-1-X"
    p3["line_items"][1]["purchase_order"] = None  # nothing to link the row on without OCR pages
    assert count(p3) == 0  # documented LOWER BOUND without an OCR index
    way = copy.deepcopy(gold["dev_0002"])
    assert count(way, way) == 0 and rs.doc_spn_misreads(None, g, {"pages": []}, None, sc, rd) == 0


# ---------------------------------------------------------------------- analysis


def write_run(
    root: Path, name: str, preds: dict[str, Any], vram_gib: float, batch: int = 1,
    complete: bool = True, drop_last_doc: bool = False,
) -> Path:  # fmt: skip
    run = root / name
    run.mkdir(parents=True)
    ids = list(preds)[:-1] if drop_last_doc else list(preds)
    meta = {"latency_s": 40.0 + vram_gib, "peak_vram_bytes": int(vram_gib * GIB),
            "n_visual_tokens": 1260}  # fmt: skip
    lines = "".join(
        json.dumps({"doc_id": d, "pages": [{"parsed": {"line_items": preds[d]["line_items"]},
                                            "meta": meta}]}) + "\n"
        for d in ids
    )  # fmt: skip
    (run / "trace.jsonl").write_text(lines)
    (run / "predictions.json").write_text(json.dumps({d: preds[d] for d in ids}))
    (run / "manifest.json").write_text(json.dumps({"batch_size": batch}))
    (run / "progress.json").write_text(
        json.dumps({"status": "complete" if complete else "running", "done": len(ids)})
    )
    return run


class Scenario:
    """Six synthetic invoices (3 scanned); the control misreads one part number in every doc."""

    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        self.gold = _synth.make_corpus(tmp / "data")
        self.ids = ["dev_0000", "dev_0001", "dev_0003", "dev_0004", "dev_0005", "dev_0006"]
        self.meta_path = tmp / "meta.json"
        self.meta_path.write_text(json.dumps(
            [{"doc_id": d, "scanned": i % 2 == 0} for i, d in enumerate(self.ids)]
        ))  # fmt: skip
        self.good = {d: copy.deepcopy(self.gold[d]) for d in self.ids}
        self.bad = copy.deepcopy(self.good)
        for p in self.bad.values():
            p["line_items"][1]["supplier_part_number"] += "X"  # a one-character misread
        self.status: dict[str, Any] = {}

    def add(self, px: int, preds: dict[str, Any], vram: float, **kw: Any) -> None:
        run = write_run(self.tmp / "runs", f"run{px}", preds, vram, **kw)
        self.status[str(px)] = {"state": "complete", "run_dir": str(run), "source": "ran"}

    def analyse(self) -> dict[str, Any]:
        return rs.analyse(
            self.status, self.ids, self.tmp / "data" / "dev" / "labels", self.meta_path,
            None, n_boot=200,
        )  # fmt: skip


@needs_scorer
def test_analysis_adopts_the_best_resolution_that_fits_and_reports_every_delta(
    tmp_path: Path,
) -> None:
    s = Scenario(tmp_path)
    s.add(1_310_720, s.bad, 9.0)
    s.add(1_843_200, s.good, 10.0)  # fixes every misread, fits
    s.add(2_196_480, s.good, 15.0)  # same gain but 15 GiB > 14.5
    res = s.analyse()
    assert res["adopted"] == "1843200"
    ctl = res["control"]
    assert (ctl["misreads"], ctl["misreads_scanned"]) == (6, 3) and ctl["source"] == "ran"
    assert "LOWER BOUND" in ctl["ocr_basis"] and ctl["batch_size"] == 1
    r1, r2 = res["candidates"]
    assert r1["state"] == r2["state"] == "complete"
    assert r1["verdict"] == "ADOPTED" and "14.5 GiB" in r2["verdict"]
    d = r1["scores"]["all"]["OVERALL"]
    assert d["delta"] > 0 and d["lo"] > 0
    assert r1["scores"]["scanned"]["OVERALL"]["delta"] > 0
    assert r1["scores"]["digital"]["OVERALL"]["delta"] > 0
    assert r1["misreads"]["delta"] == -6 and r1["misreads_scanned"]["delta"] == -3
    assert r1["misreads"]["lo"] == r1["misreads"]["hi"] == -6
    assert r1["peak_vram_bytes"] == 10 * GIB and r2["peak_vram_bytes"] == 15 * GIB
    assert r1["s_per_page_mean"] == pytest.approx(50.0) and r1["n_visual_tokens_mean"] == 1260
    assert res["tokens"] == {"1310720": 1260, "1843200": 1750, "2196480": 2145}
    text = rs.format_result(res)
    assert "40 documents (wide)" in text and "OVERALL" in text and "only criterion" in text
    assert "misreads 6 -> 0" in text and "verdict: ADOPTED" in text
    assert text.splitlines()[-1] == "DONE ressweep px=1310720,1843200,2196480 adopted=1843200"


@needs_scorer
def test_analysis_keeps_the_control_when_the_gain_is_not_significant(tmp_path: Path) -> None:
    s = Scenario(tmp_path)
    s.add(1_310_720, s.good, 9.0)
    s.add(1_843_200, s.good, 10.0)  # identical outputs: delta 0, CI [0, 0]
    res = s.analyse()
    assert res["adopted"] == "control"
    assert rs.format_result(res).splitlines()[-1].endswith("adopted=control")


@needs_scorer
def test_analysis_lists_oom_failed_incomplete_and_mismatched_batch_runs_as_excluded(
    tmp_path: Path,
) -> None:
    s = Scenario(tmp_path)
    s.add(1_310_720, s.bad, 9.0)
    s.status["1500000"] = {"state": "oom", "source": "smoke"}  # no run folder at all
    s.add(1_600_000, s.good, 10.0, drop_last_doc=True)  # trace misses a document
    s.add(1_700_000, s.good, 10.0, complete=False)  # progress says running
    s.add(1_800_000, s.good, 10.0, batch=4)  # not comparable with the batch-1 control
    s.status["1900000"] = {"state": "failed", "run_dir": str(tmp_path / "nope"), "source": "ran"}
    res = s.analyse()
    states = {c["max_pixels"]: c["state"] for c in res["candidates"]}
    assert states == {1_500_000: "oom", 1_600_000: "incomplete", 1_700_000: "incomplete",
                      1_800_000: "incomplete", 1_900_000: "failed"}  # fmt: skip
    assert res["adopted"] == "control"
    assert all(c["verdict"].startswith("excluded") for c in res["candidates"])
    assert (
        "batch size 4" in next(c for c in res["candidates"] if c["max_pixels"] == 1_800_000)["note"]
    )
    text = rs.format_result(res)
    assert text.count("EXCLUDED") == 5


@needs_scorer
def test_analysis_refuses_a_missing_or_incomplete_control(tmp_path: Path) -> None:
    s = Scenario(tmp_path)
    s.add(1_843_200, s.good, 10.0)
    with pytest.raises(ValueError, match="control run is not complete"):
        s.analyse()
    s.add(1_310_720, s.bad, 9.0, complete=False)
    with pytest.raises(ValueError, match="control run is incomplete"):
        s.analyse()


@needs_scorer
def test_a_swallowed_cuda_oom_in_a_trace_marks_the_run_oom(tmp_path: Path) -> None:
    s = Scenario(tmp_path)
    s.add(1_310_720, s.bad, 9.0)
    s.add(1_843_200, s.good, 10.0)
    path = Path(s.status["1843200"]["run_dir"]) / "trace.jsonl"
    path.write_text(path.read_text().replace('"n_visual_tokens": 1260',
                                             '"n_visual_tokens": 1260, "logprob_error": '
                                             '"CUDA out of memory"', 1))  # fmt: skip
    res = s.analyse()
    assert res["candidates"][0]["state"] == "oom" and res["adopted"] == "control"


def test_run_stats_reads_latency_vram_and_tokens() -> None:
    pages = [{"meta": {"latency_s": 10.0, "peak_vram_bytes": 5, "n_visual_tokens": 100}},
             {"meta": {"latency_s": 20.0, "peak_vram_bytes": 9, "n_visual_tokens": 300}},
             {"meta": {"latency_s": None, "peak_vram_bytes": None}}]  # fmt: skip
    st = rs.run_stats([{"doc_id": "a", "pages": pages}])
    assert st["s_per_page_mean"] == 15.0 and st["peak_vram_bytes"] == 9
    assert st["n_visual_tokens_mean"] == 200.0 and st["n_pages"] == 3 and st["oom_pages"] == 0
    assert rs.run_stats([])["s_per_page_mean"] is None


# ---------------------------------------------------------------------- estimate


SPEED = json.loads((ROOT / "configs" / "spike_speed.json").read_text(encoding="utf-8"))


def test_prefill_scaling_is_linear_in_input_tokens_and_exact_at_the_reference() -> None:
    assert rs.scaled_prefill_s(2.0, 1875, 1260, 1260) == 2.0
    assert rs.scaled_prefill_s(2.0, 1875, 1260, 2145) == pytest.approx(2.0 * (615 + 2145) / 1875)
    assert rs.scaled_prefill_s(2.0, 1875, 1260, 1750) < rs.scaled_prefill_s(2.0, 1875, 1260, 2145)


def test_estimate_follows_the_speed_model_and_control_reuse_removes_its_hours() -> None:
    pxs = [px for px, _ in rs.RESOLUTIONS]
    est = rs.estimate_rows(SPEED, 55, 6, 1, False, pxs)
    m, n = SPEED["models"]["qwen35_4b_img_only"], SPEED["n_out_keyed"]["qwen35_4b"]
    load_h = SPEED["model_load_s"] / 3600
    ctl = est["rows"][0]
    assert ctl["s_page"] == pytest.approx(m["prefill_s"] + n["mean"] / m["decode_tok_s"])
    assert ctl["hours"] == pytest.approx(ctl["s_page"] * 55 / 3600 + load_h)
    assert [r["s_page"] for r in est["rows"]] == sorted(r["s_page"] for r in est["rows"])
    assert est["hours"] == pytest.approx(sum(r["hours"] for r in est["rows"]) + est["smoke_hours"])
    reused = rs.estimate_rows(SPEED, 55, 6, 1, True, pxs)
    assert reused["rows"][0]["runs"] is False and reused["rows"][0]["hours"] == 0.0
    assert reused["hours"] == pytest.approx(est["hours"] - ctl["hours"])
    assert est["smoke_hours"] == pytest.approx(
        est["rows"][-1]["s_page"] * 6 / 3600 + load_h  # the smoke runs the largest resolution
    )
    assert est["hours_ub"] > est["hours"]


def test_estimate_text_shows_both_cu_rates_and_labels_the_assumptions() -> None:
    pxs = [px for px, _ in rs.RESOLUTIONS]
    est = rs.estimate_rows(SPEED, 55, 6, 1, False, pxs)
    text = rs.format_estimate(
        est, SPEED, "WARNING: not a T4", rs.estimate_rows(SPEED, 55, 6, 1, True, pxs)
    )
    lo, hi = SPEED["t4_cu_per_hour"], SPEED["t4_cu_per_hour_conservative"]
    assert f"{est['hours'] * lo:.1f}-{est['hours'] * hi:.1f} CU" in text
    assert "UNVERIFIED" in text and "ASSUMED" in text and "IF the control is reused" in text
    assert "1260" in text and "1750" in text and "2145" in text and "WARNING: not a T4" in text
