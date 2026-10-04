"""shipdoc.predict_native (notebook 04c_predict_test_native): refusals, estimates, both pipelines.

Everything runs on the mock backend over the SYNTHETIC corpus of tests/_synth.py (the fake ``test``
folder of tests/test_predict.py) with the NATIVE config swapped in; the calibrators are fitted on
synthetic labels. No real image, OCR text, label, prediction or calibrator run is used; the merge
itself (peft on the real model) is UNVERIFIED on a GPU. The unchanged stages are tested in
tests/test_predict.py, test_reuse_v0.py, test_notebook_predict_v2.py and test_flags.py.
"""

from __future__ import annotations

# ruff: noqa: F811  (the `world` fixture is imported, then requested by name, as pytest requires)
import dataclasses
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from test_flags import calibrator_for, freeze
from test_notebook_predict_v2 import ft_world_run, make_final_adapter
from test_predict import (  # noqa: F401 - `_clean_sha` is an autouse fixture
    N_PAGES,
    SCHEMA,
    TEST_IDS,
    World,
    _clean_sha,
    _edit_trace,
    assemble,
    needs_schema,
    run_all,
    world,
)
from test_reuse_v0 import HEAD, SHAPES, V0, blobs, fake_v0, write_ocr_cache

from shipdoc import flags, nativerun, oof, predict, predict_ft, predict_native, reuse, spike
from shipdoc import ocr_stage as ocr_stage_mod

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / nativerun.NATIVE_CONFIG
LEGACY = ROOT / "configs" / "spike_qwen35_4b_img_only.yaml"
NATIVE_HASH = "e4b84ec2809625d5"  # the hash GG recorded for the native config (2026-10-03)
SPEED = json.loads((ROOT / "configs" / "spike_speed.json").read_text(encoding="utf-8"))
NErr = predict_native.NativeTestError


@pytest.fixture()
def cfg() -> spike.SpikeConfig:
    return predict_native.load_native_config(NATIVE)


@pytest.fixture()
def nworld(world: World) -> World:
    """The synthetic world at the NATIVE config (the runs' manifests carry its hash)."""
    return dataclasses.replace(world, cfg=spike.load_config(NATIVE))


# -------------------- the native config, the 02n run, the batch contract


def test_the_native_config_is_accepted_and_the_1260_config_refused() -> None:
    c = predict_native.load_native_config(NATIVE)
    assert c.backend.max_pixels == 2_196_480 and c.config_hash == NATIVE_HASH
    with pytest.raises(NErr, match="max_pixels 1310720"):
        predict_native.load_native_config(LEGACY)


def zs_run(tmp: Path, h: str, name: str = "zs02n", batch: int | None = 2, status: str = "complete"):
    d = tmp / name
    d.mkdir(parents=True)
    man: dict[str, Any] = {"config": {"name": "c", "hash": h}, "shard": "0/1"}
    if batch is not None:
        man["batch_size"] = batch
        (d / "bench_result.json").write_text(json.dumps({"chosen_batch_size": batch}))
    (d / "manifest.json").write_text(json.dumps(man))
    (d / "progress.json").write_text(json.dumps({"status": status}))
    (d / "predictions.json").write_text("{}")
    return d


def test_the_02n_run_must_be_complete_and_at_the_native_hash(tmp_path: Path, cfg: Any) -> None:
    ok = zs_run(tmp_path, NATIVE_HASH)
    assert predict_native.check_zs_run(ok, cfg)["batch_size"] == 2
    with pytest.raises(NErr, match="mixed-resolution"):
        predict_native.check_zs_run(zs_run(tmp_path, "01d87878679ca0fc", "old"), cfg)
    with pytest.raises(NErr, match="not complete"):
        predict_native.check_zs_run(zs_run(tmp_path, NATIVE_HASH, "half", status="running"), cfg)
    with pytest.raises(NErr, match="not a folder"):
        predict_native.check_zs_run(tmp_path / "nowhere", cfg)


def test_the_batch_is_the_02n_runs_and_the_zs_path_refuses_any_other(tmp_path: Path) -> None:
    d = zs_run(tmp_path, NATIVE_HASH, batch=4)
    for model in ("zs", "ft"):
        c = predict_native.resolve_batch(model, d, None)
        assert c["batch_size"] == 4 and c["source"] == "zero_shot_run"
    assert predict_native.resolve_batch("zs", d, 4)["batch_size"] == 4  # equal is fine
    with pytest.raises(NErr, match="batch-size contract"):
        predict_native.resolve_batch("zs", d, 8)
    ft = predict_native.resolve_batch("ft", d, 8)  # the guard decides on the merged model
    assert ft["batch_size"] == 8 and ft["differs_from_zero_shot"] is True
    with pytest.raises(NErr, match="batch size is unknown"):
        predict_native.resolve_batch("zs", zs_run(tmp_path, NATIVE_HASH, "nobatch", None), None)
    with pytest.raises(NErr, match="MODEL must be"):
        predict_native.resolve_batch("both", d, None)


# -------------------- ZS_TEST_DIR (ft path) and the v0 traces (zs path)


def v15(tmp: Path, cfg: Any, name: str = "v15_abcdef0", **edit: Any) -> Path:
    """A synthetic validated v15 folder (files only; the strict reuse decision is injected)."""
    d = tmp / name
    d.mkdir(parents=True)
    man: dict[str, Any] = {
        "submission": name,
        "mode": "full",
        "config": {"name": cfg.name, "hash": cfg.config_hash},
        "model": {"id": "Qwen/Qwen3.5-4B", "revision": "r"},
        "batch_size": 2,
        "code_sha": "a" * 40,
    }
    rep: dict[str, Any] = {"ok": True, "checks": {"x": {"ok": True}, "y": {"ok": True}}}
    man.update(edit.pop("man", {}))
    rep.update(edit.pop("rep", {}))
    (d / "manifest.json").write_text(json.dumps(man))
    (d / "validation_report.json").write_text(json.dumps(rep))
    (d / "trace.jsonl").write_text("{}\n")
    if not edit.pop("no_predictions", False):
        (d / "test_predictions.json").write_text("{}")
    return d


def allow(*a: Any, **k: Any) -> reuse.ReuseDecision:
    return reuse.ReuseDecision(True, [], {"decode_path": {"files": 17}})


def test_a_validated_native_v15_folder_is_accepted(tmp_path: Path, cfg: Any) -> None:
    got = predict_native.check_zs_test_dir(v15(tmp_path, cfg), cfg, 2, evaluate=allow)
    assert got["dir"] == "v15_abcdef0" and got["batch_size"] == 2
    assert got["decode_path_files"] == 17 and len(got["trace_sha256"]) == 64


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        ({"man": {"config": {"name": "c", "hash": "01d87878679ca0fc"}}}, "config hash"),
        ({"man": {"config": {"name": "c"}}}, "config hash None"),
        ({"man": {"submission": "v1_abcdef0"}}, "not a v15_ folder"),
        ({"man": {"submission": "v0_abcdef0"}}, "not a v15_ folder"),
        ({"man": {"mode": "reuse"}}, "not 'full'"),
        ({"man": {"model": {"id": "Qwen/x+lora:abcdef012345"}}}, "fine-tuned"),
        ({"man": {"ft": {"adapter_sha256": "x"}}}, "fine-tuned"),
        ({"rep": {"ok": False}}, "not VALIDATED"),
        ({"rep": {"checks": {"x": {"ok": True}, "y": {"ok": False}}}}, "failed checks \\['y'\\]"),
        ({"no_predictions": True}, "no test_predictions.json"),
        ({"man": {"batch_size": 8}}, "not the 02n run's 2"),
        ({"man": {"batch_size": None}}, "no batch size"),
    ],
)
def test_every_unacceptable_zs_test_folder_is_refused(
    tmp_path: Path, cfg: Any, edit: dict[str, Any], message: str
) -> None:
    d = v15(tmp_path, cfg, **edit)
    with pytest.raises(NErr, match=message):
        predict_native.check_zs_test_dir(d, cfg, 2, evaluate=allow)


def test_a_missing_folder_a_missing_manifest_and_a_refused_reuse_decision(
    tmp_path: Path, cfg: Any
) -> None:
    with pytest.raises(NErr, match=r"MODEL = 'zs' first.*No zero-shot inference"):
        predict_native.check_zs_test_dir(tmp_path / "v15_none", cfg, 2, evaluate=allow)
    empty = tmp_path / "v15_empty"
    empty.mkdir()
    with pytest.raises(NErr, match="no manifest.json"):
        predict_native.check_zs_test_dir(empty, cfg, 2, evaluate=allow)
    refuse = lambda *a, **k: reuse.ReuseDecision(False, ["seed: 7 != 42"], {})  # noqa: E731
    with pytest.raises(NErr, match=r"strict reuse decision: seed: 7 != 42"):
        predict_native.check_zs_test_dir(v15(tmp_path, cfg), cfg, 2, evaluate=refuse)


@needs_schema
def test_the_v0_1260_traces_are_refused_by_the_real_reuse_decision_at_native(
    nworld: World, world: World, tmp_path: Path, cfg: Any
) -> None:
    v0 = fake_v0(world, tmp_path)  # a v0 folder from a run at the 1260 config

    def real(c: Any, d: Path, data: Path, b: int, shard: str, ack: bool) -> reuse.ReuseDecision:
        return reuse.evaluate_reuse(c, d, data, b, shard, ack, blob_fn=blobs(),
                                    diff_fn=lambda *a: "", head_sha=HEAD)  # fmt: skip

    reasons = predict_native.assert_v0_refused(v0, cfg, 1, data_root=world.data, evaluate=real)
    assert any(r.startswith("config_hash: v0 '") for r in reasons)
    # the same folder IS reusable at its own (1260) config: the refusal is the resolution's
    assert reuse.evaluate_reuse(world.cfg, v0, world.data, 1, "0/1", blob_fn=blobs(),
                                diff_fn=lambda *a: "", head_sha=HEAD).ok  # fmt: skip
    # and ZS_TEST_DIR refuses a v0 folder for the ft path (config hash, name, mode)
    with pytest.raises(NErr, match="config hash"):
        predict_native.check_zs_test_dir(v0, cfg, 1, evaluate=allow)


def test_assert_v0_refused_raises_when_the_decision_would_allow_or_names_no_hash(
    tmp_path: Path, cfg: Any
) -> None:
    with pytest.raises(NErr, match="did not refuse"):
        predict_native.assert_v0_refused(tmp_path, cfg, 1, evaluate=allow)
    other = lambda *a, **k: reuse.ReuseDecision(False, ["seed: 7 != 42"], {})  # noqa: E731
    with pytest.raises(NErr, match="did not refuse the v0 traces on the config hash"):
        predict_native.assert_v0_refused(tmp_path, cfg, 1, evaluate=other)
    hashy = lambda *a, **k: reuse.ReuseDecision(False, ["config_hash: v0 'a' != v1 'b'"], {})  # noqa: E731
    assert predict_native.assert_v0_refused(tmp_path, cfg, 1, evaluate=hashy)


# -------------------- the adapter (ft path)


def test_the_adapter_resolution_gate(tmp_path: Path, cfg: Any) -> None:
    good = make_final_adapter(tmp_path / "ok" / "final", cfg)
    msgs: list[str] = []
    predict_native.check_adapter_native(good, cfg, msgs.append)
    assert msgs[0].startswith("PASS") and "RESOLUTION CHECK" in msgs[0]
    old = make_final_adapter(tmp_path / "old" / "final", spike.load_config(LEGACY))
    with pytest.raises(NErr, match="resolution mismatch"):
        predict_native.check_adapter_native(old, cfg, msgs.append)
    (good / "manifest.json").unlink()
    with pytest.raises(NErr, match="missing or unreadable"):
        predict_native.check_adapter_native(good, cfg, msgs.append)


# -------------------- the calibrator


def cal_file(tmp: Path, arm: str = "zs", **edit: Any) -> Path:
    data: dict[str, Any] = {"artifact": flags.ARTIFACT_KIND, "arm": arm,
                            "run_config_hash": NATIVE_HASH}  # fmt: skip
    data.update(edit)
    p = tmp / f"cal_{len(list(tmp.glob('cal_*')))}.json"
    p.write_text(json.dumps({k: v for k, v in data.items() if v is not None}))
    return p


def test_no_calibrator_says_how_to_refreeze_at_native(tmp_path: Path, cfg: Any) -> None:
    with pytest.raises(NErr, match=r"freeze_calibrator.py --arm zs.*after 02n and calibrate_v2"):
        predict_native.check_calibrator(None, cfg, "zs")
    with pytest.raises(NErr, match=r"--arm ft.*calibrate_v3"):
        predict_native.check_calibrator(None, cfg, "ft")
    with pytest.raises(NErr, match="does not exist"):
        predict_native.check_calibrator(tmp_path / "nope.json", cfg, "zs")


def test_a_calibrator_without_or_with_another_config_hash_is_refused(
    tmp_path: Path, cfg: Any
) -> None:
    assert predict_native.check_calibrator(cal_file(tmp_path), cfg, "zs")["arm"] == "zs"
    with pytest.raises(NErr, match="frozen for config hash None"):
        predict_native.check_calibrator(cal_file(tmp_path, run_config_hash=None), cfg, "zs")
    with pytest.raises(NErr, match="frozen for config hash '01d87878679ca0fc'"):
        predict_native.check_calibrator(
            cal_file(tmp_path, run_config_hash="01d87878679ca0fc"), cfg, "zs"
        )
    with pytest.raises(NErr, match="is the 'ft' calibrator"):
        predict_native.check_calibrator(cal_file(tmp_path, "ft"), cfg, "zs")
    with pytest.raises(NErr, match="not a calibrator artifact"):
        predict_native.check_calibrator(cal_file(tmp_path, artifact="other"), cfg, "zs")


def test_the_committed_1260_calibrator_is_refused_at_native(cfg: Any) -> None:
    committed = ROOT / "meta" / "calibrator_zs.json"
    assert "run_config_hash" not in json.loads(committed.read_text())  # the known gap, unmodified
    with pytest.raises(NErr, match="no recorded hash"):
        predict_native.check_calibrator(committed, cfg, "zs")


# -------------------- estimates


def test_the_estimate_rows_follow_the_measured_native_pace_and_are_labelled() -> None:
    s1, b8 = nativerun.NATIVE_B1_S_PER_PAGE, nativerun.NATIVE_B8_S_PER_PAGE
    load = SPEED["model_load_s"]
    rows = predict_native.estimate_rows(SPEED, "zs", 280, 6, 7, 8)
    assert [r["scenario"] for r in rows] == ["low", "high"]
    low, high = rows
    assert low["run_s"] == pytest.approx(load + 280 * b8) and high["run_s"] == pytest.approx(
        load + 280 * s1
    )
    assert low["smoke_s"] == pytest.approx(load + 6 * s1) and low["guard_s"] == 0.0
    assert low["hours"] == pytest.approx((3 * load + 6 * s1 + 280 * b8 + 7 * b8) / 3600)
    assert low["hours"] < high["hours"] and low["cu_central"] == pytest.approx(
        low["hours"] * SPEED["t4_cu_per_hour"]
    )
    ft = predict_native.estimate_rows(SPEED, "ft", 280, 6, 7, 8)
    guard = s1 + nativerun.BENCH_PAGES * (s1 + b8)
    assert ft[0]["guard_s"] == pytest.approx(guard)
    assert ft[0]["load_merge_s"] == pytest.approx(load + nativerun.MERGE_S)
    assert ft[0]["det_s"] == pytest.approx(load + nativerun.MERGE_S + 7 * b8)
    one = predict_native.estimate_rows(SPEED, "ft", 280, 6, 7, 1)
    assert [r["scenario"] for r in one] == ["batch 1"] and one[0]["guard_s"] == 0.0
    text = predict_native.format_estimate(rows, SPEED, "zs", 280, 8)
    assert "ESTIMATE (UNVERIFIED on a GPU)" in text and "zero-shot" in text and "CU@" in text
    with pytest.raises(NErr, match="MODEL must be"):
        predict_native.estimate_rows(SPEED, "x", 1, 1, 1, 1)


def test_the_estimate_stage_prints_for_both_paths(
    nworld: World, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    smoke = tmp_path / "smoke.json"
    smoke.write_text(json.dumps([f"dev_{i:04d}" for i in range(5)]))
    for model in ("zs", "ft"):
        rc = predict_native.main(
            ["estimate", "--model", model, "--batch-size", "8", "--data-root", str(nworld.data),
             "--smoke-docs", str(smoke)]
        )  # fmt: skip
        assert rc == 0
        out = capsys.readouterr().out
        assert f"{len(TEST_IDS) and N_PAGES} pages, batch 8" in out and "ESTIMATE" in out
    assert predict_native.main(["estimate", "--model", "zs"]) == 1
    assert "pass --zs-run-dir" in capsys.readouterr().err


# -------------------- check (CLI)


def test_check_stage_resolves_the_batch_refuses_and_writes_the_result(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    d = zs_run(tmp_path, NATIVE_HASH, batch=4)
    out = tmp_path / "check.json"
    base = ["check", "--model", "zs", "--config", str(NATIVE), "--zs-run-dir", str(d)]
    assert predict_native.main([*base, "--out", str(out)]) == 0
    res = json.loads(out.read_text())
    assert res["batch_size"] == 4 and res["config_hash"] == NATIVE_HASH and res["model"] == "zs"
    assert "PASS  batch size 4" in capsys.readouterr().out
    assert predict_native.main([*base, "--batch-size", "8"]) == 1  # the contract
    assert "batch-size contract" in capsys.readouterr().err
    old = zs_run(tmp_path, "01d87878679ca0fc", "old")
    argv = ["check", "--model", "zs", "--zs-run-dir", str(old)]
    assert predict_native.main(argv) == 1 and "mixed-resolution" in capsys.readouterr().err
    argv = ["check", "--model", "zs", "--config", str(LEGACY), "--zs-run-dir", str(d)]
    assert predict_native.main(argv) == 1 and "this notebook runs only" in capsys.readouterr().err
    # ft needs its adapter and the v15 folder; each missing input is a refusal naming it
    ft = ["check", "--model", "ft", "--zs-run-dir", str(d)]
    assert predict_native.main(ft) == 1 and "--adapter-dir" in capsys.readouterr().err


def test_check_stage_ft_refuses_a_1260_adapter_and_a_missing_zs_test_dir(
    tmp_path: Path, cfg: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    d = zs_run(tmp_path, NATIVE_HASH, batch=2)
    good = make_final_adapter(tmp_path / "ok" / "final", cfg)
    old = make_final_adapter(tmp_path / "old" / "final", spike.load_config(LEGACY))
    base = ["check", "--model", "ft", "--zs-run-dir", str(d)]
    assert predict_native.main([*base, "--adapter-dir", str(old)]) == 1
    assert "resolution mismatch" in capsys.readouterr().err
    assert predict_native.main([*base, "--adapter-dir", str(good)]) == 1
    assert "--zs-test-dir" in capsys.readouterr().err
    argv = [*base, "--adapter-dir", str(good), "--zs-test-dir", str(tmp_path / "v15_none")]
    assert predict_native.main(argv) == 1
    assert "MODEL = 'zs' first" in capsys.readouterr().err


# -------------------- gated ft stages (the resolution gate before predict_ft)


def test_the_ft_stages_run_the_gate_first_and_infer_stamps_it(
    tmp_path: Path, cfg: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[list[str]] = []
    runs = tmp_path / "runs"
    (runs / "r1").mkdir(parents=True)
    (runs / "r1" / predict_ft.FT_RUN_FILE).write_text(json.dumps({"adapter_sha256": "x"}))

    def fake_main(argv: list[str]) -> int:
        calls.append(list(argv))
        return 0

    monkeypatch.setattr(predict_ft, "main", fake_main)
    good = make_final_adapter(tmp_path / "ok" / "final", cfg)
    old = make_final_adapter(tmp_path / "old" / "final", spike.load_config(LEGACY))
    rest = ["--run-id", "r1", "--runs-root", str(runs), "--decision", "d.json"]
    assert predict_native.main(["infer", "--adapter-dir", str(old), *rest]) == 1
    assert not calls and "resolution mismatch" in capsys.readouterr().err  # nothing delegated
    assert predict_native.main(["infer", "--adapter-dir", str(good), *rest]) == 0
    assert calls[0][0] == "infer" and calls[0][-2:] == ["--config", nativerun.NATIVE_CONFIG]
    rec = json.loads((runs / "r1" / predict_ft.FT_RUN_FILE).read_text())
    assert rec["native_resolution"]["ok"] is True
    assert rec["native_resolution"]["max_pixels"] == 2_196_480
    for stage in ("verify", "determinism"):  # no stamp, but the same gate
        assert predict_native.main([stage, "--adapter-dir", str(good), *rest]) == 0
    assert [c[0] for c in calls] == ["infer", "verify", "determinism"]
    monkeypatch.setattr(predict_ft, "main", lambda argv: 1)  # a failed stage is not stamped
    (runs / "r1" / predict_ft.FT_RUN_FILE).write_text(json.dumps({"adapter_sha256": "x"}))
    assert predict_native.main(["infer", "--adapter-dir", str(good), *rest]) == 1
    assert "native_resolution" not in json.loads((runs / "r1" / predict_ft.FT_RUN_FILE).read_text())
    assert predict_native.main(["infer", "--adapter-dir", str(good), "--config", str(LEGACY)]) == 1
    assert "this notebook runs only" in capsys.readouterr().err


# -------------------- the ZS path end to end (mock backend)


def finish_v15(w: World, tmp: Path, name: str = "v15_abcdef0") -> tuple[Path, Path, Path]:
    """ZS run + assemble (04b's full path) + OCR records: ``(submission, ocr root, recheck)``."""
    ocr_root = tmp / "ocr"
    write_ocr_cache(ocr_root, w.data / "test" / "images")
    run_all(w, batch=1)
    _edit_trace(w)  # every rule has something to do
    sub = tmp / name
    rep = assemble(w, sub, ocr_cache=ocr_root, shapes_file=SHAPES,
                   smoke_status_path=w.runs / "smoke_status.json")  # fmt: skip
    assert rep["ok"], {k: c for k, c in rep["checks"].items() if not c["ok"]}
    stems = ocr_stage_mod.expected_stems(w.data / "test" / "images")
    timing = ocr_stage_mod.timing_summary(ocr_root, stems, "gpu", 12.0, len(stems))
    predict._write_json(sub / "ocr_timing.json", timing)
    recheck = tmp / "recheck.json"
    recheck.write_text(
        json.dumps({"ok": True, "n_pages": 5, "n_text_identical": 5, "n_content_identical": 5})
    )
    return sub, ocr_root, recheck


def fin_zs(w: World, sub: Path, recheck: Path, cfg: Any) -> dict[str, Any]:
    return predict_native.finalize(
        "zs", sub, cfg=cfg, run_dir=w.runs / "t0", shapes_file=SHAPES, recheck_path=recheck,
        expect_pages=N_PAGES,
    )  # fmt: skip


@needs_schema
def test_the_zs_path_finalizes_labels_v15_and_records_the_native_config(
    nworld: World, tmp_path: Path, cfg: Any
) -> None:
    sub, _, recheck = finish_v15(nworld, tmp_path)
    rep = fin_zs(nworld, sub, recheck, cfg)
    bad = {k: c["detail"] for k, c in rep["checks"].items() if not c["ok"]}
    assert rep["ok"] and not bad, bad
    for k in ("native_config", "native_zero_shot_model", "native_predictions_strict",
              "v1_rule_switches", "v1_no_rule_skipped", "v1_shapes_sha256", "json_schema",
              "ocr_cache_complete", "ocr_determinism", "production_config"):  # fmt: skip
        assert rep["checks"][k]["ok"], k
    man = json.loads((sub / "manifest.json").read_text())
    assert man["submission"] == "v15_abcdef0" and man["mode"] == "full"  # not v1_ / v0_
    assert man["native"]["config_hash"] == NATIVE_HASH and man["native"]["model"] == "zs"
    assert man["config"]["hash"] == NATIVE_HASH and man["files"][-1] == "review_flags.json"
    assert man["post_rules"]["switches"] == {"r1": True, "r2": True, "r3": True}
    assert (sub / "test_predictions.json").is_file()
    preds = json.loads((sub / "test_predictions.json").read_text())
    assert predict.validate_schema(preds, json.loads(SCHEMA.read_text()))["ok"]
    for d in preds.values():
        assert set(d) <= {"doc_type", "header", "line_items"}
    assert man["post_rules"]["skipped"] == {}  # R2 skipped nowhere: the OCR cache is complete


@needs_schema
def test_a_zs_run_at_1260_or_with_an_adapter_is_rejected_and_the_name_withheld(
    world: World, tmp_path: Path, cfg: Any
) -> None:
    sub, _, recheck = finish_v15(world, tmp_path)  # a complete, valid run, but at 1260 tokens
    rep = predict_native.finalize("zs", sub, cfg=cfg, run_dir=world.runs / "t0",
                                  shapes_file=SHAPES, recheck_path=recheck,
                                  expect_pages=N_PAGES)  # fmt: skip
    assert not rep["ok"] and not rep["checks"]["native_config"]["ok"]
    assert not (sub / "test_predictions.json").exists()
    assert (sub / "test_predictions.REJECTED.json").is_file()
    assert json.loads((sub / "validation_report.json").read_text())["ok"] is False


@needs_schema
def test_a_zs_manifest_with_an_adapter_model_id_is_rejected(
    nworld: World, tmp_path: Path, cfg: Any
) -> None:
    sub, _, recheck = finish_v15(nworld, tmp_path)
    man = json.loads((sub / "manifest.json").read_text())
    man["model"]["id"] = "Qwen/Qwen3.5-4B+lora:abcdef012345"
    (sub / "manifest.json").write_text(json.dumps(man))
    rep = fin_zs(nworld, sub, recheck, cfg)
    assert not rep["ok"] and not rep["checks"]["native_zero_shot_model"]["ok"]


@needs_schema
def test_a_prediction_with_a_flag_key_is_rejected(nworld: World, tmp_path: Path, cfg: Any) -> None:
    sub, _, recheck = finish_v15(nworld, tmp_path)
    p = json.loads((sub / "test_predictions.json").read_text())
    p["test_0000"]["flags"] = {"p": 1}
    (sub / "test_predictions.json").write_text(json.dumps(p))
    rep = fin_zs(nworld, sub, recheck, cfg)
    assert not rep["ok"] and not rep["checks"]["native_predictions_strict"]["ok"]


def native_calibrator(tmp: Path, sub: Path, ocr_root: Path, arm: str, **kw: Any) -> Path:
    """`calibrator_for` (synthetic labels) with the native config hash recorded in the file."""
    path = calibrator_for(sub, ocr_root, arm, tmp, **kw)
    data = json.loads(path.read_text())
    data["run_config_hash"] = NATIVE_HASH
    out = tmp / f"native_{path.name}"
    freeze.write_artifact(data, out)
    return out


def flags_run(
    model: str, sub: Path, cal: Path | None, ocr_root: Path, cfg: Any, **kw: Any
) -> dict[str, Any]:
    return predict_native.run_flags(
        model, cfg=cfg, config_path=NATIVE, submission_dir=sub, calibrator=cal,
        ocr_cache=ocr_root, zs_test_dir=kw.pop("zs_test_dir", None),
        batch_size=kw.pop("batch_size", 1), expect_docs=len(TEST_IDS), shapes_file=SHAPES,
        say=lambda m: None, **kw,
    )  # fmt: skip


@needs_schema
def test_the_zs_flags_need_the_native_calibrator_and_write_only_the_flags_file(
    nworld: World, tmp_path: Path, cfg: Any
) -> None:
    sub, ocr_root, recheck = finish_v15(nworld, tmp_path)
    assert fin_zs(nworld, sub, recheck, cfg)["ok"]
    before = (sub / "test_predictions.json").read_bytes()
    with pytest.raises(NErr, match="no CALIBRATOR_FILE"):
        flags_run("zs", sub, None, ocr_root, cfg)  # the default: refuses, says how to refreeze
    assert not (sub / flags.FLAGS_NAME).exists()
    plain = calibrator_for(sub, ocr_root, "zs", tmp_path)  # no run_config_hash: like the 1260 one
    with pytest.raises(NErr, match="frozen for config hash None"):
        flags_run("zs", sub, plain, ocr_root, cfg)
    cal = native_calibrator(tmp_path, sub, ocr_root, "zs")
    doc = flags_run("zs", sub, cal, ocr_root, cfg)
    assert sorted(doc["docs"]) == sorted(TEST_IDS) and doc["arm"] == "zs"
    assert (sub / "test_predictions.json").read_bytes() == before  # never touched
    text = (sub / flags.FLAGS_NAME).read_text()
    assert set(flags.doc_string_values(doc["docs"])) <= flags.DOC_STRING_VALUES
    preds = json.loads((sub / "test_predictions.json").read_text())
    for d in preds.values():
        for v in [*d["header"].values(), *(c for r in d["line_items"] for c in r.values())]:
            if isinstance(v, str) and len(v) >= 4:
                assert f'"{v}"' not in text, "an extracted value leaked into the flags file"
    rep = predict_native.check_flags(sub, calibrator=cal, cfg=cfg, field_target=0.98,
                                     doc_target=0.98, expect_docs=len(TEST_IDS))  # fmt: skip
    bad = {k: c["detail"] for k, c in rep["flags_checks"].items() if not c["ok"]}
    assert rep["flags_ok"] and not bad, bad
    for k in ("flags_exact_docs", "flags_calibrator_native", "flags_not_in_predictions"):
        assert rep["flags_checks"][k]["ok"], k
    man = json.loads((sub / "manifest.json").read_text())
    assert man["review_flags"]["ok"] is True and man["review_flags"]["arm"] == "zs"
    # the committed 1260 calibrator carries no hash: refused before anything is computed
    with pytest.raises(NErr, match="no recorded hash"):
        flags_run("zs", sub, ROOT / "meta" / "calibrator_zs.json", ocr_root, cfg)


@needs_schema
def test_the_flags_stage_refuses_a_submission_at_another_hash_and_the_check_catches_drift(
    nworld: World, tmp_path: Path, cfg: Any
) -> None:
    sub, ocr_root, recheck = finish_v15(nworld, tmp_path)
    assert fin_zs(nworld, sub, recheck, cfg)["ok"]
    cal = native_calibrator(tmp_path, sub, ocr_root, "zs")
    man = json.loads((sub / "manifest.json").read_text())
    man["config"]["hash"] = "01d87878679ca0fc"
    (sub / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(NErr, match="mixed-resolution"):
        flags_run("zs", sub, cal, ocr_root, cfg)
    man["config"]["hash"] = NATIVE_HASH
    (sub / "manifest.json").write_text(json.dumps(man))
    flags_run("zs", sub, cal, ocr_root, cfg)
    doc = json.loads((sub / flags.FLAGS_NAME).read_text())
    doc["docs"].pop(next(iter(doc["docs"])))  # a flags file with 199 documents
    (sub / flags.FLAGS_NAME).write_text(json.dumps(doc))
    rep = predict_native.check_flags(sub, calibrator=cal, cfg=cfg, field_target=0.98,
                                     doc_target=0.98, expect_docs=len(TEST_IDS))  # fmt: skip
    assert not rep["flags_ok"] and not rep["flags_checks"]["flags_exact_docs"]["ok"]


@needs_schema
def test_check_flags_catches_a_missing_rules_file_and_reports_the_submission_code_sha(
    nworld: World, tmp_path: Path, cfg: Any
) -> None:
    sub, ocr_root, recheck = finish_v15(nworld, tmp_path)
    assert fin_zs(nworld, sub, recheck, cfg)["ok"]
    cal = native_calibrator(tmp_path, sub, ocr_root, "zs")
    doc = flags_run("zs", sub, cal, ocr_root, cfg)
    man = json.loads((sub / "manifest.json").read_text())
    assert doc["inputs"]["submission_code_sha"] == man["code_sha"]
    assert doc["inputs"]["local_head_sha"] == doc["inputs"]["code_sha"]

    def check() -> dict[str, Any]:
        return predict_native.check_flags(sub, calibrator=cal, cfg=cfg, field_target=0.98,
                                          doc_target=0.98, expect_docs=len(TEST_IDS))  # fmt: skip

    assert check()["flags_ok"]
    # a flags file produced against another / no recorded code SHA fails the check
    saved = (sub / flags.FLAGS_NAME).read_text()
    for bad in (None, "f" * 40):
        d2 = json.loads(saved)
        d2["inputs"]["submission_code_sha"] = bad
        (sub / flags.FLAGS_NAME).write_text(json.dumps(d2))
        assert not check()["flags_checks"]["flags_submission_code_sha"]["ok"]
    (sub / flags.FLAGS_NAME).write_text(saved)
    # the rules file removed after the fact: the flags no longer describe a checkable folder
    (sub / "rules.jsonl").rename(sub / "rules.jsonl.moved")
    rep = check()
    assert not rep["flags_ok"] and not rep["flags_checks"]["flags_rules_jsonl"]["ok"]


# -------------------- the FT path end to end (mock merged backend)


def ft_native_world(
    nworld: World, tmp: Path, cfg: Any, v15_sub: Path, ocr_root: Path, recheck: Path
) -> tuple[Path, Path, Path]:
    """FT run (mock merged backend, native config) + assemble: ``(v2n dir, run dir, adapter)``."""
    rec, _, ad, _ = ft_world_run(nworld, tmp, cfg, det=True, batch=1)
    run = nworld.runs / "ft0"
    ftp = run / predict_ft.FT_RUN_FILE
    ft = json.loads(ftp.read_text())
    ft["native_resolution"] = {"ok": True, "max_pixels": cfg.backend.max_pixels}  # the gate's stamp
    ftp.write_text(json.dumps(ft))
    sub = tmp / "v2n_abcdef0"
    rep = assemble(nworld, sub, run_id="ft0", ocr_cache=ocr_root, shapes_file=SHAPES,
                   smoke_status_path=tmp / "meta" / "smoke.json")  # fmt: skip
    assert rep["ok"], {k: c for k, c in rep["checks"].items() if not c["ok"]}
    shutil.copyfile(v15_sub / "ocr_timing.json", sub / "ocr_timing.json")
    return sub, run, ad


def fin_ft(sub: Path, run: Path, v15_sub: Path, recheck: Path, cfg: Any) -> dict[str, Any]:
    return predict_native.finalize(
        "ft", sub, cfg=cfg, run_dir=run, shapes_file=SHAPES, recheck_path=recheck,
        expect_pages=N_PAGES, zs_test_dir=v15_sub,
    )  # fmt: skip


@needs_schema
def test_the_ft_path_end_to_end_finalize_flags_and_every_refusal(
    nworld: World, tmp_path: Path, cfg: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    v15_sub, ocr_root, recheck = finish_v15(nworld, tmp_path)
    assert fin_zs(nworld, v15_sub, recheck, cfg)["ok"]
    sub, run, _ = ft_native_world(nworld, tmp_path, cfg, v15_sub, ocr_root, recheck)
    # an ft finalize without the v15 folder is refused outright
    with pytest.raises(NErr, match="needs --zs-test-dir"):
        predict_native.finalize("ft", sub, cfg=cfg, run_dir=run, shapes_file=SHAPES,
                                recheck_path=recheck)  # fmt: skip
    rep = fin_ft(sub, run, v15_sub, recheck, cfg)
    bad = {k: c["detail"] for k, c in rep["checks"].items() if not c["ok"]}
    assert rep["ok"] and not bad, bad
    for k in ("native_config", "native_adapter_resolution", "native_zero_shot_test_traces",
              "native_predictions_strict", "ft_adapter_verified", "ft_model_is_merged_adapter",
              "ft_merge_recorded", "predictions_structure_vs_v0"):  # fmt: skip
        assert rep["checks"][k]["ok"], k
    man = json.loads((sub / "manifest.json").read_text())
    assert man["submission"] == "v2n_abcdef0" and man["ft"]["merge"]["n_lora_modules_merged"] == 200
    assert man["native"]["zs_test"]["dir"] == "v15_abcdef0" and man["native"]["model"] == "ft"
    zs_hash = man["native"]["zs_test"]["trace_sha256"]
    assert zs_hash == predict.sha256_file(v15_sub / "trace.jsonl")
    # flags: the ft calibrator needs the v15 folder; the strict reuse decision is the real one
    cal = native_calibrator(tmp_path, sub, ocr_root, "ft", zs_sub=v15_sub)
    with pytest.raises(NErr, match="need ZS_TEST_DIR"):
        flags_run("ft", sub, cal, ocr_root, cfg)
    decisions: list[tuple[Any, ...]] = []

    def allow_reuse(*a: Any, **k: Any) -> reuse.ReuseDecision:
        decisions.append(a)
        return reuse.ReuseDecision(True, [], {"v0_code_sha": "a" * 40,
                                              "decode_path": {"files": 17}})  # fmt: skip

    monkeypatch.setattr(reuse, "evaluate_reuse", allow_reuse)  # flags._zs_arm and the check use it
    flagged = flags_run("ft", sub, cal, ocr_root, cfg, zs_test_dir=v15_sub, batch_size=1)
    assert flagged["arm"] == "ft" and sorted(flagged["docs"]) == sorted(TEST_IDS)
    assert flagged["inputs"]["zs_reuse"]["decode_path_files"] == 17
    rep2 = predict_native.check_flags(sub, calibrator=cal, cfg=cfg, field_target=0.98,
                                      doc_target=0.98, expect_docs=len(TEST_IDS))  # fmt: skip
    assert rep2["flags_ok"], {k: c["detail"] for k, c in rep2["flags_checks"].items()}
    assert decisions  # the real decision function was consulted, not skipped
    # a v15 folder that is not validated / at another hash is refused before any flags
    man15 = json.loads((v15_sub / "manifest.json").read_text())
    man15["config"]["hash"] = "01d87878679ca0fc"
    (v15_sub / "manifest.json").write_text(json.dumps(man15))
    with pytest.raises(NErr, match="ZS_TEST_DIR v15_abcdef0 refused"):
        flags_run("ft", sub, cal, ocr_root, cfg, zs_test_dir=v15_sub, batch_size=1)


@needs_schema
@pytest.mark.parametrize(
    ("what", "check"),
    [
        ("no_stamp", "native_adapter_resolution"),
        ("wrong_pixels", "native_adapter_resolution"),
        ("zs_test_hash", "native_zero_shot_test_traces"),
        ("zs_test_rejected", "native_zero_shot_test_traces"),
    ],
)
def test_the_ft_finalize_rejects_an_unstamped_adapter_and_an_unfit_zs_folder(
    nworld: World, tmp_path: Path, cfg: Any, what: str, check: str
) -> None:
    v15_sub, ocr_root, recheck = finish_v15(nworld, tmp_path)
    assert fin_zs(nworld, v15_sub, recheck, cfg)["ok"]
    sub, run, _ = ft_native_world(nworld, tmp_path, cfg, v15_sub, ocr_root, recheck)
    ftp = run / predict_ft.FT_RUN_FILE
    ft = json.loads(ftp.read_text())
    if what == "no_stamp":
        ft.pop("native_resolution")
    elif what == "wrong_pixels":
        ft["native_resolution"]["max_pixels"] = 1_310_720
    ftp.write_text(json.dumps(ft))
    if what == "zs_test_hash":
        m = json.loads((v15_sub / "manifest.json").read_text())
        m["config"]["hash"] = "01d87878679ca0fc"
        (v15_sub / "manifest.json").write_text(json.dumps(m))
    elif what == "zs_test_rejected":
        (v15_sub / "test_predictions.json").rename(v15_sub / "test_predictions.REJECTED.json")
    rep = fin_ft(sub, run, v15_sub, recheck, cfg)
    assert not rep["ok"] and not rep["checks"][check]["ok"], rep["checks"][check]
    assert not (sub / "test_predictions.json").exists()
    assert (sub / "test_predictions.REJECTED.json").is_file()


def test_the_cli_parser_knows_every_stage() -> None:
    p = predict_native.build_parser()
    for stage in ("check", "estimate", "verify", "infer", "determinism", "finalize", "flags",
                  "check-flags"):  # fmt: skip
        assert stage in next(a for a in p._actions if getattr(a, "choices", None)).choices
    assert hashlib.sha256(b"").hexdigest() and oof and V0  # imports used by fixtures above
