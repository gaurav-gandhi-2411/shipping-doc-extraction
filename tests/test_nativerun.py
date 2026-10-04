"""shipdoc.nativerun: native estimates, the resolution gate, and the unchanged oof / bench logic
exercised with the native config (max_pixels 2,196,480). CPU only, mock backends, synthetic corpus.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from _synth import make_corpus
from test_bench import BENCH_DOCS, BenchMock, needs_scorer
from test_oof import FOLDS, HELD0, MergedMock, World
from test_predict import _clean_sha  # noqa: F401 - autouse fixture, must be visible here

from shipdoc import bench, oof, spike
from shipdoc import nativerun as nr

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / nr.NATIVE_CONFIG
LEGACY = ROOT / "configs" / "spike_qwen35_4b_img_only.yaml"
SPEED = json.loads((ROOT / "configs" / "spike_speed.json").read_text(encoding="utf-8"))
needs_data = pytest.mark.skipif(
    not (ROOT / "data" / "train" / "labels").is_dir(), reason="data/ absent (gitignored)"
)


def ok_matcher(manifest: Any, cfg: Any) -> tuple[bool, str]:
    return True, "match"


# --------------------------------------------------------------------------------------------
# Config and constants
# --------------------------------------------------------------------------------------------


def test_native_config_is_the_legacy_config_with_only_name_and_max_pixels_changed() -> None:
    new, old = (yaml.safe_load(p.read_text(encoding="utf-8")) for p in (NATIVE, LEGACY))
    assert new["max_pixels"] == nr.NATIVE_MAX_PIXELS == 2_196_480 == 2145 * 32 * 32
    assert old["max_pixels"] == nr.LEGACY_MAX_PIXELS == 1_310_720 == 1280 * 32 * 32
    assert new["name"] == nr.NATIVE_CONFIG_NAME and old["name"] != new["name"]
    assert {k: v for k, v in new.items() if k not in ("name", "max_pixels")} == {
        k: v for k, v in old.items() if k not in ("name", "max_pixels")
    }
    cfg = spike.load_config(NATIVE)
    assert cfg.backend.max_pixels == nr.NATIVE_MAX_PIXELS and cfg.config_hash
    assert cfg.config_hash != spike.load_config(LEGACY).config_hash


def test_pace_constants_trace_to_the_measured_figures() -> None:
    zs_pace, native_pace = nr.ZS_B8_S_PER_PAGE, nr.NATIVE_B8_S_PER_PAGE
    assert zs_pace == pytest.approx(8406.1 / 671)
    assert native_pace == pytest.approx(8406.1 / 671 * 1.047)
    # the factor lies between the two ratios the sweep's numbers give
    assert 38.3 / 36.6 - 0.001 <= nr.NATIVE_FACTOR <= 2133.4 / 2036.5 + 0.001
    assert nr.batched_pace("low") < nr.batched_pace("high") == nr.NATIVE_B1_S_PER_PAGE
    with pytest.raises(ValueError, match="scenario"):
        nr.batched_pace("mid")


# --------------------------------------------------------------------------------------------
# Estimates
# --------------------------------------------------------------------------------------------


def test_zs_estimate_rows_follow_the_stated_formula() -> None:
    load, s1, sb = 120.0, 38.3, nr.NATIVE_B8_S_PER_PAGE
    low, high = nr.zs_estimate_rows(SPEED, 671, 6, None)
    assert (low["scenario"], high["scenario"]) == ("low", "high")
    smoke = load + 6 * s1
    bench_s = load + s1 + 12 * (s1 + 3 * sb)
    full = load + 671 * sb
    assert low["hours"] == pytest.approx((smoke + bench_s + full) / 3600)
    assert low["cu_central"] == pytest.approx(low["hours"] * 1.19)
    assert low["cu_conservative"] == pytest.approx(low["hours"] * 1.58)
    assert high["hours"] > low["hours"] and high["s_per_page"] == s1
    assert high["full_s"] == pytest.approx(load + 671 * s1)


def test_zs_estimate_manual_batch_has_no_bench_and_batch_one_is_one_row() -> None:
    low, high = nr.zs_estimate_rows(SPEED, 671, 6, 8)
    assert low["bench_s"] == high["bench_s"] == 0.0
    (only,) = nr.zs_estimate_rows(SPEED, 671, 6, 1)
    assert only["scenario"] == "batch 1" and only["s_per_page"] == 38.3
    assert only["full_s"] == pytest.approx(120 + 671 * 38.3)


def test_oof_estimate_rows_follow_the_stated_formula() -> None:
    s1, sb = 38.3, nr.NATIVE_B8_S_PER_PAGE
    low, high = nr.oof_estimate_rows(SPEED, 230, 8)
    guard = s1 + 12 * (s1 + sb)
    assert low["guard_s"] == pytest.approx(guard) and low["merge_s"] == 60.0
    assert low["hours"] == pytest.approx((120 + 60 + guard + 230 * sb) / 3600)
    assert high["hours"] == pytest.approx((120 + 60 + s1 + 12 * (2 * s1) + 230 * s1) / 3600)
    (one,) = nr.oof_estimate_rows(SPEED, 230, 1)
    assert one["guard_s"] == 0.0 and one["hours"] == pytest.approx((120 + 60 + 230 * s1) / 3600)


def test_estimate_text_is_labelled_and_shows_both_cu_rates() -> None:
    txt = nr.format_zs_estimate(SPEED, 671, 6, None, "Tesla T4")
    assert "ESTIMATE (UNVERIFIED on a GPU)" in txt and "CU@1.19" in txt and "CU@1.58" in txt
    assert "UNMEASURED at native" in txt and "not a T4" not in txt
    assert "not a T4" in nr.format_zs_estimate(SPEED, 671, 6, None, "NVIDIA L4")
    assert "not a T4" in nr.format_zs_estimate(SPEED, 671, 6, None, None)
    rows = nr.oof_estimate_rows(SPEED, 230, 8)
    oof_txt = nr.format_oof_estimate(rows, SPEED, 230, 8, 0)
    assert "ESTIMATE (UNVERIFIED on a GPU)" in oof_txt and "OOF fold 0" in oof_txt
    assert "low" in oof_txt and "high" in oof_txt


@needs_data
def test_estimate_oof_cli_counts_the_pages_of_the_fold(capsys: pytest.CaptureFixture[str]) -> None:
    assert nr.main(["estimate-oof", "--fold", "0", "--batch-size", "8"]) == 0
    out = capsys.readouterr().out
    assert "documents 171, pages 230" in out and "batch size 8 (manual)" in out
    with pytest.raises(nr.NativeError, match="--zs-run-dir"):
        nr.main(["estimate-oof", "--fold", "0"])


# --------------------------------------------------------------------------------------------
# The resolution gate (fails closed)
# --------------------------------------------------------------------------------------------


def test_gate_accepts_a_matching_adapter_under_the_native_config() -> None:
    ok, why = nr.check_adapter_resolution({"stage": "fold0"}, spike.load_config(NATIVE), ok_matcher)
    assert ok is True and why == "match"


def test_gate_refuses_a_matcher_that_says_no_and_passes_its_reason_on() -> None:
    ok, why = nr.check_adapter_resolution(
        {"stage": "fold0"}, spike.load_config(NATIVE), lambda m, c: (False, "trained at 1310720")
    )
    assert ok is False and "1310720" in why


@pytest.mark.parametrize("answer", [True, None, (True,), ("yes", "x"), (True, 3), [True, "x"]])
def test_gate_refuses_a_malformed_matcher_answer(answer: Any) -> None:
    ok, why = nr.check_adapter_resolution(
        {"stage": "fold0"}, spike.load_config(NATIVE), lambda m, c: answer
    )
    assert ok is False and "unexpected answer" in why


def test_gate_refuses_when_the_matcher_raises() -> None:
    def boom(manifest: Any, cfg: Any) -> tuple[bool, str]:
        raise KeyError("max_pixels")

    ok, why = nr.check_adapter_resolution({"stage": "fold0"}, spike.load_config(NATIVE), boom)
    assert ok is False and "KeyError" in why


def test_gate_fails_closed_when_resmatch_is_not_importable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "shipdoc.resmatch", None)  # import raises ImportError
    ok, why = nr.check_adapter_resolution({"stage": "fold0"}, spike.load_config(NATIVE))
    assert ok is False and "resmatch is not available" in why and "fail closed" in why


def test_gate_with_the_real_resmatch_accepts_2196480_and_refuses_1310720_or_unknown() -> None:
    native, legacy = spike.load_config(NATIVE), spike.load_config(LEGACY)

    def manifest(px: Any) -> dict[str, Any]:
        return {"inference_keys": {"max_pixels": px}}

    ok, why = nr.check_adapter_resolution(manifest(2_196_480), native)
    assert ok is True and "2196480" in why
    ok, why = nr.check_adapter_resolution(manifest(1_310_720), native)  # a 1260-token adapter
    assert ok is False and "resolution mismatch" in why and "1310720" in why
    for bad in (None, "2196480", True, 0):  # unknown / malformed training resolution: closed
        assert nr.check_adapter_resolution(manifest(bad), native)[0] is False
    assert nr.check_adapter_resolution({"stage": "fold0"}, native)[0] is False  # no keys at all
    # the matcher is handed a mapping with the inference max_pixels (what resmatch reads)
    seen: list[Any] = []
    nr.check_adapter_resolution({"x": 1}, native, lambda m, c: (seen.append(c), (True, ""))[1])
    assert seen == [{"max_pixels": 2_196_480}]
    # the reverse: a native-trained adapter under the 1260 config never reaches resmatch here
    assert nr.check_adapter_resolution(manifest(2_196_480), legacy)[0] is False


def test_gate_uses_the_real_resmatch_when_it_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    import types

    fake = types.ModuleType("shipdoc.resmatch")
    fake.training_resolution_matches = lambda m, c: (True, "stub")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "shipdoc.resmatch", fake)
    assert nr.check_adapter_resolution({"x": 1}, spike.load_config(NATIVE)) == (True, "stub")


def test_gate_refuses_a_non_native_config_and_a_missing_manifest() -> None:
    ok, why = nr.check_adapter_resolution({"stage": "fold0"}, spike.load_config(LEGACY), ok_matcher)
    assert ok is False and "1310720" in why and "native" in why
    for manifest in (None, {}):
        ok, why = nr.check_adapter_resolution(manifest, spike.load_config(NATIVE), ok_matcher)
        assert ok is False and "manifest" in why


def test_assert_adapter_resolution_prints_the_row_and_raises(tmp_path: Path) -> None:
    final = tmp_path / "final"
    final.mkdir()
    msgs: list[str] = []
    cfg = spike.load_config(NATIVE)
    with pytest.raises(nr.NativeError, match="manifest"):
        nr.assert_adapter_resolution(final, cfg, ok_matcher, msgs.append)  # no manifest.json
    assert msgs[-1].startswith("FAIL  RESOLUTION CHECK")
    (final / "manifest.json").write_text("{not json")
    with pytest.raises(nr.NativeError):
        nr.assert_adapter_resolution(final, cfg, ok_matcher, msgs.append)
    (final / "manifest.json").write_text(json.dumps({"stage": "fold0"}))
    nr.assert_adapter_resolution(final, cfg, ok_matcher, msgs.append)
    assert msgs[-1].startswith("PASS  RESOLUTION CHECK")


def test_cli_gate_runs_before_the_oof_stage_and_a_refusal_never_reaches_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from shipdoc import cli

    final = tmp_path / "final"
    final.mkdir()
    (final / "manifest.json").write_text(json.dumps({"stage": "fold0"}))
    seen: list[list[str]] = []
    monkeypatch.setattr(cli, "main", lambda argv: (seen.append(list(argv)), 0)[1])
    argv = ["--fold", "0", "--config", str(NATIVE), "--adapter-dir", str(final), "--pin", "f" * 40]
    monkeypatch.setattr(nr, "_load_resmatch", lambda: ok_matcher)
    assert nr.main(["verify", *argv]) == 0
    assert seen == [["oof", "verify", *argv]]  # the oof stage gets exactly the same flags
    monkeypatch.setattr(nr, "_load_resmatch", lambda: lambda m, c: (False, "trained at 1310720"))
    assert nr.main(["infer", *argv]) == 1
    assert len(seen) == 1  # refused: shipdoc oof infer was NOT called
    assert "OOF REFUSED" in capsys.readouterr().out


# --------------------------------------------------------------------------------------------
# oof.py with the native config: max_pixels 1310720 refused, 2196480 accepted, and the reverse
# --------------------------------------------------------------------------------------------


def _native_world(tmp: Path) -> World:
    w = World(tmp)
    w.cfg = spike.load_config(NATIVE)  # zero-shot run, adapters and inference all at native
    return w


def _failed(report: dict[str, Any]) -> list[str]:
    return [r["check"] for r in report["rows"] if not r["ok"]]


def test_a_1310720_adapter_is_refused_by_the_native_config_and_2196480_is_accepted(
    tmp_path: Path,
) -> None:
    w = _native_world(tmp_path)
    w.zero_shot()
    keys = oof.inference_keys_of_config(w.cfg)
    assert keys["max_pixels"] == 2_196_480
    old = w.adapter(_name="old", inference_keys={**keys, "max_pixels": 1_310_720})
    with pytest.raises(oof.OofError, match="inference key max_pixels"):
        w.infer(old, MergedMock(w.gold, w.cfg))
    assert not (w.out_dir / "trace.jsonl").exists()  # refused before anything ran
    section = w.infer(w.adapter(_name="new"), MergedMock(w.gold, w.cfg))
    assert section["verification"]["ok"] and section["n_inference_docs"] == len(HELD0)
    # max_pixels is the ONLY failing row for the old adapter
    zs_man, _ = oof.load_zero_shot(w.runs / "zs")
    rep = oof.verify_adapter_manifest(old, fold=0, cfg=w.cfg, zs_manifest=zs_man, folds=FOLDS)
    assert _failed(rep) == ["inference key max_pixels"]


def test_a_native_adapter_and_a_native_zero_shot_run_are_refused_by_the_1260_config(
    tmp_path: Path,
) -> None:
    """The reverse (the old notebook 05's semantics): its 1260 config refuses native inputs."""
    w = _native_world(tmp_path)
    w.zero_shot()
    adapter = w.adapter()
    zs_man, _ = oof.load_zero_shot(w.runs / "zs")
    legacy = spike.load_config(LEGACY)
    rep = oof.verify_adapter_manifest(adapter, fold=0, cfg=legacy, zs_manifest=zs_man, folds=FOLDS)
    assert not rep["ok"]
    assert "inference key max_pixels" in _failed(rep)
    assert "zero-shot run config hash" in _failed(rep)  # the 02n run is not a 1260 run either
    ok, _why = nr.check_adapter_resolution({"stage": "fold0"}, legacy, ok_matcher)
    assert ok is False  # and 05n's own gate refuses the 1260 config outright


def test_run_ids_of_the_native_notebooks_never_collide_with_the_1260_runs() -> None:
    sha7 = "abcdef1"
    zs_native = f"zeroshot500_{nr.NATIVE_CONFIG_NAME}_{sha7}"
    zs_1260 = f"zeroshot500_qwen35_4b_img_only_keyed_{sha7}"
    assert zs_native != zs_1260 and "keyed" not in zs_native
    assert "native" in zs_native


# --------------------------------------------------------------------------------------------
# The bench at native: an out-of-memory size is recorded as failed, the bench goes on
# --------------------------------------------------------------------------------------------


@needs_scorer
def test_native_bench_records_oom_sizes_as_failed_and_picks_the_largest_that_fits(
    tmp_path: Path,
) -> None:
    class OomAt4(BenchMock):
        def extract_pages(self, requests: Any) -> Any:
            if len(requests) >= 4:  # a native batch 4 / 8 that does not fit
                raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
            return super().extract_pages(requests)

    root = tmp_path / "data"
    gold = make_corpus(root, labels=True)
    result = bench.run_bench(
        spike.load_config(NATIVE), OomAt4(gold), BENCH_DOCS, tmp_path / "bench", data_root=root
    )
    by = {r["batch_size"]: r for r in result["results"]}
    assert by[4]["ok"] is False and by[8]["ok"] is False and by[2]["ok"] and by[1]["ok"]
    assert "out of memory" in by[8]["error"]
    assert result["chosen_batch_size"] == 2 and result["config"] == nr.NATIVE_CONFIG_NAME
    assert "FAILED" in bench.format_banner(result)
