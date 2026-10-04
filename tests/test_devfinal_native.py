"""shipdoc.devfinal_native (notebook 08): the native-resolution gates around the unchanged devfinal.

Runs on the SYNTHETIC corpus and the mock merged backend of tests/test_devfinal.py with the NATIVE
config swapped in (the manifests then carry the native config hash and ``max_pixels``). No GPU,
no peft, no real document, no network. The unchanged devfinal stages are tested in
tests/test_devfinal.py; only the new gates, the native estimate, the stamp and the CLI wiring are
tested here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from test_devfinal import (  # noqa: F401 - `_eleven_dev_docs` is an autouse fixture
    DEV_IDS,
    FOLDS_F,
    Dev,
    MergedMock,
    _eleven_dev_docs,
    finished,
    make_final_adapter,
)
from test_predict import _clean_sha  # noqa: F401 - autouse fixture, must be visible here

from shipdoc import devfinal, devfinal_native, nativerun, oof, runcompat, spike

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / nativerun.NATIVE_CONFIG
LEGACY = ROOT / "configs" / "spike_qwen35_4b_img_only.yaml"
NATIVE_HASH = "e4b84ec2809625d5"  # the hash GG recorded for the native config (2026-10-03)


@pytest.fixture()
def nd(tmp_path: Path) -> Dev:
    """A synthetic dev world whose zero-shot run is at the NATIVE config."""
    dev = Dev(tmp_path)
    dev.w.cfg = spike.load_config(NATIVE)
    dev.w.zero_shot(null_header=("buyer_name",))
    return dev


@pytest.fixture()
def legacy(tmp_path: Path) -> Dev:
    """The same world at the 1260-token config (what 05b makes)."""
    dev = Dev(tmp_path / "legacy")
    dev.w.zero_shot(null_header=("buyer_name",))
    return dev


def cli_args(d: Dev, stage: str, *extra: str) -> list[str]:
    return [
        stage, "--config", str(NATIVE), "--data-root", str(d.w.data),
        "--folds", str(d.tmp / "folds.json"), "--dev-docs", str(d.dev_docs),
        "--zs500-docs", str(d.zs500), *extra,
    ]  # fmt: skip


@pytest.fixture(autouse=True)
def _files(tmp_path: Path) -> None:
    (tmp_path / "folds.json").write_text(json.dumps(FOLDS_F))


# -------------------- the native config gate


def test_the_native_config_is_accepted_and_the_1260_config_is_refused() -> None:
    cfg = devfinal_native.load_native_config(NATIVE)
    assert cfg.backend.max_pixels == 2_196_480 and cfg.config_hash == NATIVE_HASH
    with pytest.raises(devfinal_native.NativeDevError, match="max_pixels 1310720"):
        devfinal_native.load_native_config(LEGACY)


def test_main_adds_the_native_config_and_never_devfinals_own_1260_default(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert devfinal_native._with_config(["verify"])[-2:] == [
        "--config",
        str(ROOT / nativerun.NATIVE_CONFIG),
    ]
    assert devfinal_native._with_config(["verify", "--config", "x"]) == ["verify", "--config", "x"]
    assert devfinal_native._with_config(["verify", "--config=x"]) == ["verify", "--config=x"]
    rc = devfinal_native.main(["verify", "--config", str(LEGACY), "--adapter-dir", "nowhere"])
    assert rc == 1 and "this notebook runs only" in capsys.readouterr().err


# -------------------- a run folder must carry the native config hash


def test_a_native_run_passes_and_a_1260_run_is_refused(nd: Dev, legacy: Dev) -> None:
    cfg = devfinal_native.load_native_config(NATIVE)
    h = devfinal_native.check_run_resolution(nd.w.runs / "zs", cfg, "zero-shot run")
    assert h == NATIVE_HASH
    with pytest.raises(devfinal_native.NativeDevError, match="mixed-resolution"):
        devfinal_native.check_run_resolution(legacy.w.runs / "zs", cfg, "zero-shot run")


def test_a_missing_folder_a_missing_manifest_and_a_hashless_manifest_are_refused(
    nd: Dev, tmp_path: Path
) -> None:
    cfg = devfinal_native.load_native_config(NATIVE)
    with pytest.raises(devfinal_native.NativeDevError, match="not a folder"):
        devfinal_native.check_run_resolution(tmp_path / "nowhere", cfg, "dev run")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(devfinal_native.NativeDevError, match="mixed-resolution"):
        devfinal_native.check_run_resolution(empty, cfg, "dev run")
    man = json.loads((nd.w.runs / "zs" / "manifest.json").read_text())
    man["config"].pop("hash")
    (empty / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(devfinal_native.NativeDevError, match="mixed-resolution"):
        devfinal_native.check_run_resolution(empty, cfg, "dev run")
    assert runcompat.manifest_config_hash(man) is None


# -------------------- the adapter resolution gate (on top of the unchanged verification)


def test_a_native_adapter_passes_the_resolution_gate_and_prints_its_row(nd: Dev) -> None:
    cfg = devfinal_native.load_native_config(NATIVE)
    msgs: list[str] = []
    devfinal_native.check_adapter_native(nd.adapter(), cfg, msgs.append)
    assert msgs and msgs[0].startswith("PASS") and "RESOLUTION CHECK" in msgs[0]


def test_an_adapter_trained_at_1260_is_refused_by_the_gate_before_verification(
    nd: Dev,
) -> None:
    cfg = devfinal_native.load_native_config(NATIVE)
    legacy_cfg = spike.load_config(LEGACY)
    old = make_final_adapter(nd.tmp / "old" / "final", legacy_cfg)  # inference_keys at 1310720
    msgs: list[str] = []
    with pytest.raises(devfinal_native.NativeDevError, match="resolution mismatch"):
        devfinal_native.check_adapter_native(old, cfg, msgs.append)
    assert msgs[0].startswith("FAIL")
    # the unchanged verification refuses it too (its own inference-key row), as a second layer
    rep = oof_free_verify(nd, old)
    assert "inference key max_pixels" in [r["check"] for r in rep["rows"] if not r["ok"]]


def oof_free_verify(d: Dev, adapter: Path) -> dict[str, Any]:
    from shipdoc import predict_ft

    return predict_ft.verify_final_adapter(
        adapter, cfg=d.w.cfg, folds=FOLDS_F, pin_sha="b" * 40, reachable=lambda s: True
    )


def test_an_adapter_without_a_readable_training_resolution_is_refused(nd: Dev) -> None:
    cfg = devfinal_native.load_native_config(NATIVE)
    ad = nd.adapter()
    man = json.loads((ad / "manifest.json").read_text())
    man["inference_keys"].pop("max_pixels")
    (ad / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(devfinal_native.NativeDevError, match="no usable inference_keys"):
        devfinal_native.check_adapter_native(ad, cfg, lambda m: None)
    (ad / "manifest.json").unlink()
    with pytest.raises(devfinal_native.NativeDevError, match="missing or unreadable"):
        devfinal_native.check_adapter_native(ad, cfg, lambda m: None)


# -------------------- estimate


def test_the_estimate_is_labelled_scales_from_the_native_pace_and_names_08(nd: Dev) -> None:
    cfg = devfinal_native.load_native_config(NATIVE)
    msgs: list[str] = []
    res = devfinal_native.run_estimate(
        cfg=cfg, zs_run_dir=nd.w.runs / "zs", n_pages=135, batch_size=None, out=msgs.append
    )
    text = "\n".join(msgs)
    assert "ESTIMATE (UNVERIFIED on a GPU)" in text and "08 final adapter on the dev" in text
    assert "OOF fold" not in text and res["batch"]["batch_size"] == 2  # the 02n run's size
    want = nativerun.oof_estimate_rows({"model_load_s": 0}, 135, 2)
    assert [r["scenario"] for r in res["rows"]] == [r["scenario"] for r in want] == ["low", "high"]
    assert res["rows"][0]["infer_s"] == pytest.approx(135 * nativerun.NATIVE_B8_S_PER_PAGE)
    assert res["rows"][1]["infer_s"] == pytest.approx(135 * nativerun.NATIVE_B1_S_PER_PAGE)
    one = devfinal_native.run_estimate(
        cfg=cfg, zs_run_dir=nd.w.runs / "zs", n_pages=135, batch_size=1, out=lambda m: None
    )
    assert [r["scenario"] for r in one["rows"]] == ["batch 1"]


def test_the_estimate_refuses_a_1260_zero_shot_run(legacy: Dev) -> None:
    cfg = devfinal_native.load_native_config(NATIVE)
    with pytest.raises(devfinal_native.NativeDevError, match="mixed-resolution"):
        devfinal_native.run_estimate(
            cfg=cfg, zs_run_dir=legacy.w.runs / "zs", n_pages=135, batch_size=None
        )


# -------------------- the CLI end to end (mock merged backend through the real devfinal stages)


@pytest.fixture()
def mock_backend(nd: Dev, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(oof, "MergedHfBackend", lambda b, ad, sha: MergedMock(nd.w.gold, nd.w.cfg))
    monkeypatch.setattr(oof, "git_reachable", lambda sha: True)


def test_cli_plan_estimate_verify_infer_compare_at_native(
    nd: Dev,
    mock_backend: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ad = nd.adapter()
    zs = str(nd.w.runs / "zs")
    bench = nd.tmp / "bench.json"
    bench.write_text(json.dumps(DEV_IDS[:3]))
    assert devfinal_native.main(cli_args(nd, "plan")) == 0
    assert devfinal_native.main(cli_args(nd, "estimate", "--zs-run-dir", zs)) == 0
    out = capsys.readouterr().out
    assert "08 final adapter on the dev documents at native" in out and "ESTIMATE" in out
    assert devfinal_native.main(cli_args(nd, "verify", "--adapter-dir", str(ad))) == 0
    assert "PASS  RESOLUTION CHECK" in capsys.readouterr().out
    rc = devfinal_native.main(
        cli_args(
            nd,
            "infer",
            "--adapter-dir",
            str(ad),
            "--zs-run-dir",
            zs,
            "--out-dir",
            str(nd.run_dir),
            "--bench-docs",
            str(bench),
        )  # fmt: skip
    )
    assert rc == 0
    man = json.loads((nd.run_dir / "manifest.json").read_text())
    assert man["config"]["hash"] == NATIVE_HASH and "devfinal" in man
    rc = devfinal_native.main(
        cli_args(
            nd,
            "compare",
            "--zs-run-dir",
            zs,
            "--ft-run-dir",
            str(nd.run_dir),
            "--out-dir",
            str(nd.run_dir),
            "--ocr-cache",
            str(nd.ocr),
            "--n-boot",
            "50",
        )  # fmt: skip
    )
    assert rc == 0
    res = json.loads((nd.run_dir / devfinal.COMPARE_NAME).read_text())
    assert res["native_resolution"]["config_hash"] == NATIVE_HASH
    assert res["native_resolution"]["zero_shot_run_config_hash"] == NATIVE_HASH
    assert res["native_resolution"]["max_pixels"] == 2_196_480 and res["kind"] == "dev_final"
    md = (nd.run_dir / devfinal.COMPARE_MD_NAME).read_text()
    assert (
        md.startswith("> NATIVE RESOLUTION (notebook 08, max_pixels 2196480")
        and "dev_0000" not in md
    )
    assert "dev_0000" not in json.dumps(res)  # aggregates only


def test_cli_infer_refuses_a_1260_adapter_and_a_1260_zero_shot_run_before_any_model(
    nd: Dev,
    legacy: Dev,
    mock_backend: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bench = nd.tmp / "bench.json"
    bench.write_text(json.dumps(DEV_IDS[:3]))
    common = ["--out-dir", str(nd.run_dir), "--bench-docs", str(bench)]
    old = make_final_adapter(nd.tmp / "old" / "final", spike.load_config(LEGACY))
    rc = devfinal_native.main(
        cli_args(
            nd, "infer", "--adapter-dir", str(old), "--zs-run-dir", str(nd.w.runs / "zs"), *common
        )  # fmt: skip
    )
    assert rc == 1 and "resolution mismatch" in capsys.readouterr().err
    rc = devfinal_native.main(
        cli_args(
            nd,
            "infer",
            "--adapter-dir",
            str(nd.adapter()),
            "--zs-run-dir",
            str(legacy.w.runs / "zs"),
            *common,
        )  # fmt: skip
    )
    assert rc == 1 and "mixed-resolution" in capsys.readouterr().err
    assert not (nd.run_dir / "trace.jsonl").exists()  # nothing was decoded


def test_cli_compare_refuses_mixed_resolutions_and_a_run_without_the_hash(
    nd: Dev, legacy: Dev, capsys: pytest.CaptureFixture[str]
) -> None:
    finished(nd)  # an FT dev run at native
    zs_native = str(nd.w.runs / "zs")
    base = ["--out-dir", str(nd.tmp / "cmp"), "--ocr-cache", str(nd.ocr), "--n-boot", "20"]
    rc = devfinal_native.main(
        cli_args(
            nd,
            "compare",
            "--zs-run-dir",
            str(legacy.w.runs / "zs"),
            "--ft-run-dir",
            str(nd.run_dir),
            *base,
        )  # fmt: skip
    )
    assert rc == 1 and "mixed-resolution" in capsys.readouterr().err
    ft_old = legacy.run_dir  # a dev run made at 1260 (05b)
    finished(legacy)
    rc = devfinal_native.main(
        cli_args(nd, "compare", "--zs-run-dir", zs_native, "--ft-run-dir", str(ft_old), *base)
    )
    assert rc == 1 and "mixed-resolution" in capsys.readouterr().err
    rc = devfinal_native.main(cli_args(nd, "compare", "--zs-run-dir", zs_native, *base))
    assert rc == 1 and "needs --ft-run-dir" in capsys.readouterr().err
    assert not (nd.tmp / "cmp" / devfinal.COMPARE_NAME).exists()  # nothing was scored


def test_the_plumbing_check_still_works_and_is_stamped_at_native(nd: Dev) -> None:
    rc = devfinal_native.main(
        cli_args(
            nd,
            "compare",
            "--zs-run-dir",
            str(nd.w.runs / "zs"),
            "--plumbing-check",
            "--out-dir",
            str(nd.tmp / "pl"),
            "--ocr-cache",
            str(nd.ocr),
            "--n-boot",
            "20",
        )  # fmt: skip
    )
    assert rc == 0
    res = json.loads((nd.tmp / "pl" / devfinal.COMPARE_NAME).read_text())
    assert res["kind"] == "plumbing_check" and res["plumbing"]["all_zero"] is True
    assert res["native_resolution"]["config"] == "qwen35_4b_img_only_native"
