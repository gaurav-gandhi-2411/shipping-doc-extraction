"""Tests for scripts/wandb_log_public.py: allowlist, refusals, safe paths. No network, no wandb."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import wandb_log_public as wl  # noqa: E402

PLANTED = "ACME-4711-XQ"  # stands in for a gold / predicted value: unknown strings are refused


# ---- allowlist ---------------------------------------------------------------------------------


def test_allowlist_accepts_numbers_bools_null_and_known_strings() -> None:
    ok = {
        "loss": 0.12,
        "step": 3,
        "flag": True,
        "none": None,
        "sha": "42b812b5b09d6e4bff0df12564017f71ffad5fc9",
        "ver": "5.18.0",
        "model": "Qwen/Qwen3.5-4B",
        "run": "ft_fold0_42b812b_bf16",
        "slice": "awb_absent=yes",
        "rung": "R1a",
        "stage": "fold0",
        "lst": [1, 2.5, "bf16"],
    }
    assert wl.check_value(ok) == []


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        (PLANTED, "string_not_in_allowlist"),
        ("x" * 81, "string_longer_than_80"),
        ("data/train/labels/a.json", "path_like"),
        ("assignment/spec.pdf", "path_like"),
        ("D:\\shipdoc\\runs\\x", "path_like"),
        ("C:/Users/someone", "path_like"),
        ("dev_0002", "looks_like_doc_id"),
        (float("nan"), "non_finite_number"),
        (float("inf"), "non_finite_number"),
        (b"bytes", "type_bytes_not_allowed"),
    ],
)
def test_allowlist_refuses_and_never_echoes_the_value(value: Any, reason: str) -> None:
    bad = wl.check_value({"k": value})
    assert len(bad) == 1 and reason in bad[0]
    assert PLANTED not in bad[0] and "dev_0002" not in bad[0]


@pytest.mark.parametrize("key", ["dev_0002", "has space", "ünï", "a" * 65, "", 7])
def test_allowlist_refuses_bad_keys(key: Any) -> None:
    assert wl.check_value({key: 1})


def test_a_long_string_is_refused_even_if_it_is_a_known_word_prefix() -> None:
    assert wl.check_value("bf16" + " " * 80)


def test_planted_value_in_a_table_cell_or_config_is_refused() -> None:
    run = wl.RunPayload(
        "config_pins",
        "config",
        config={"a": 1},
        tables={"t": wl.Table(["slice", "n"], [["all", 1], [PLANTED, 2]])},
    )
    bad = wl.check_runs([run])
    assert len(bad) == 1 and "tables.t.rows[1]" in bad[0] and PLANTED not in bad[0]
    with pytest.raises(wl.AllowlistError):
        wl.assert_allowed([run])
    run2 = wl.RunPayload("config_pins", "config", config={"x": PLANTED})
    assert wl.check_runs([run2])


def test_table_width_mismatch_and_bad_column_are_refused() -> None:
    run = wl.RunPayload("config_pins", "config", tables={"t": wl.Table(["a b"], [[1, 2]])})
    reasons = " ".join(wl.check_runs([run]))
    assert "width_mismatch" in reasons and "key_not_short_ascii_identifier" in reasons


# ---- gates and paths ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("approved", "env", "missing"),
    [
        (False, {"SHIPDOC_PUBLISH_OK": "1"}, "--i-have-gg-approval"),
        (True, {}, "SHIPDOC_PUBLISH_OK=1"),
        (True, {"SHIPDOC_PUBLISH_OK": "yes"}, "SHIPDOC_PUBLISH_OK=1"),
        (False, {}, "--i-have-gg-approval and SHIPDOC_PUBLISH_OK=1"),
    ],
)
def test_sync_refused_without_both_flag_and_env(
    approved: bool, env: dict[str, str], missing: str
) -> None:
    with pytest.raises(wl.PublishRefused, match=missing):
        wl.require_publish_approval(approved, env)
    wl.require_publish_approval(True, {"SHIPDOC_PUBLISH_OK": "1"})


def test_main_sync_refuses_before_reading_any_artifact(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def boom(_: Any) -> Any:
        raise AssertionError("artifacts must not be read before the approval gate")

    monkeypatch.setattr(wl, "build_runs", boom)
    assert wl.main(["--sync"], env={}) == 3
    assert wl.main(["--sync", "--i-have-gg-approval"], env={}) == 3
    assert wl.main(["--sync"], env={"SHIPDOC_PUBLISH_OK": "1"}) == 3
    assert "GG must approve" in capsys.readouterr().err


def test_main_sync_with_approval_never_logs_without_the_library(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run = wl.RunPayload("config_pins", "config", config={"seed": 42})
    monkeypatch.setattr(wl, "build_runs", lambda _a: ([run], {}))
    monkeypatch.setattr(wl, "TMP_ROOT", tmp_path)
    monkeypatch.setattr(wl, "_import_wandb", lambda: None)
    rc = wl.main(
        ["--sync", "--i-have-gg-approval", "--out-dir", str(tmp_path / "o")],
        env={"SHIPDOC_PUBLISH_OK": "1"},
    )
    assert rc == 5


def test_safe_out_dir_refuses_repo_and_foreign_dirs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="inside the repository"):
        wl.safe_out_dir(wl.ROOT / "reports" / "x", (wl.ROOT,))  # repo is refused even if allowed
    with pytest.raises(ValueError, match="inside the repository"):
        wl.safe_out_dir(wl.ROOT)
    with pytest.raises(ValueError, match="not under an allowed scratch root"):
        wl.safe_out_dir(tmp_path / "x", (tmp_path / "other",))
    assert wl.safe_out_dir(tmp_path / "x", (tmp_path,)) == (tmp_path / "x").resolve()


def test_main_refuses_a_repo_output_dir_and_writes_nothing(tmp_path: Path) -> None:
    target = wl.ROOT / "wandb_dryrun_should_not_exist"
    assert wl.main(["--out-dir", str(target)]) == 4
    assert not target.exists()


def test_dry_run_writes_only_payload_json_under_the_out_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run = wl.RunPayload(
        "config_pins",
        "config",
        config={"seed": 42},
        history=[{"step": 1, "loss": 0.5}],
        tables={"t": wl.Table(["slice", "n"], [["all", 1]])},
    )
    monkeypatch.setattr(wl, "build_runs", lambda _a: ([run], {"x": "found"}))
    monkeypatch.setattr(wl, "TMP_ROOT", tmp_path)
    monkeypatch.setattr(wl, "_import_wandb", lambda: None)
    out = tmp_path / "out"
    assert wl.main(["--dry-run", "--out-dir", str(out)]) == 0
    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    assert files == ["config_pins/payload.json"]
    payload = json.loads((out / "config_pins" / "payload.json").read_text(encoding="utf-8"))
    assert payload["project"] == "shipdoc-extract" and payload["history"][0]["step"] == 1


def test_main_refuses_a_payload_the_allowlist_rejects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run = wl.RunPayload("config_pins", "config", config={"v": PLANTED})
    monkeypatch.setattr(wl, "build_runs", lambda _a: ([run], {}))
    monkeypatch.setattr(wl, "TMP_ROOT", tmp_path)
    assert wl.main(["--dry-run", "--out-dir", str(tmp_path / "o")]) == 4
    assert not (tmp_path / "o").exists()


def test_manifest_lines_have_counts_and_names_only() -> None:
    run = wl.RunPayload(
        "eval_zeroshot500",
        "eval",
        summary={"a": 1.0},
        tables={"slices": wl.Table(["slice", "n"], [["all", 1], ["dev", 2]])},
    )
    text = "\n".join(wl.manifest_lines([run], {"src": "found"}))
    assert "slices(2c x 2r)" in text and "TOTAL runs 1" in text and "table rows 2" in text


# ---- builders ----------------------------------------------------------------------------------


def test_flatten_and_long_strings_become_hashes(tmp_path: Path) -> None:
    assert wl.flatten({"a": {"b": 1, "c": [1, 2]}, "d": [{"x": 1}]}) == {"a.b": 1, "a.c": [1, 2]}
    y = tmp_path / "c.yaml"
    y.write_text("seed: 42\nlora:\n  target_regex: '" + "r" * 120 + "'\n", encoding="utf-8")
    cfg = wl.read_yaml_config(y)
    assert cfg["seed"] == 42 and "lora.target_regex" not in cfg
    assert len(cfg["lora.target_regex_sha256"]) == 64
    assert wl.check_value(cfg) == []


def test_training_run_logs_counts_never_doc_ids(tmp_path: Path) -> None:
    rd = tmp_path / "ft_fold0_42b812b_bf16"
    (rd / "final").mkdir(parents=True)
    rows = [
        {
            "step": 1,
            "epoch": 0,
            "loss": 0.5,
            "grad_norm": 1.0,
            "lr": 1e-5,
            "seconds": 2.0,
            "x": "s",
        },
        {"step": 2, "epoch": 0, "loss": 0.4, "eval_loss": 0.3, "scaler_skipped": False},
    ]
    (rd / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    manifest = {
        "stage": "fold0",
        "fold": 0,
        "code_sha": "42b812b5b09d6e4bff0df12564017f71ffad5fc9",
        "n_train_docs": 2,
        "train_doc_ids": ["dev_0002", "train_0001"],
        "heldout_doc_ids": ["dev_0003"],
        "lora": {"r": 16},
        "peft_sha256": {"adapter_config.json": "ab" * 32},
    }
    (rd / "final" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    run = wl.training_run(rd)
    assert run.config["n_heldout_docs"] == 1 and "train_doc_ids" not in run.config
    assert "x" not in run.history[0] and run.summary["final_eval_loss"] == 0.3
    assert "dev_0002" not in json.dumps(run.to_json())
    assert wl.check_runs([run]) == []


def test_eval_run_slice_table_and_csv_drops_non_finite_rows(tmp_path: Path) -> None:
    est = {"point": 0.5, "lo": 0.4, "hi": 0.6}
    m = {
        "n_docs": 2,
        "OVERALL": 0.5,
        "OVERALL_ci95": est,
        "slices": {
            "all": {"documents": 2, "illegible_fields": 0, "ci95": {"OVERALL": est}},
            "awb_absent=yes": {"documents": 1, "OVERALL": 0.7},
        },
        "per_split": {"train": {"documents": 1, "OVERALL_ci95": est}},
    }
    p = tmp_path / "metrics.json"
    p.write_text(json.dumps(m), encoding="utf-8")
    run = wl.eval_run(p)
    assert len(run.tables["slices"].rows) == 2 and wl.check_runs([run]) == []
    csv_path = tmp_path / "r.csv"
    csv_path.write_text(
        "slice,group,lo,hi,n,mean_conf,accuracy\n"
        "all,header,0,0.5,0,nan,nan\nall,header,0.5,1,4,0.9,0.75\n",
        encoding="utf-8",
    )
    t = wl._csv_table(csv_path, ("slice", "group"), "n")
    assert t.rows == [["all", "header", 0.5, 1.0, 4, 0.9, 0.75]]


def test_optional_numeric_run_drops_unknown_keys(tmp_path: Path) -> None:
    p = tmp_path / "oof_compare.json"
    p.write_text(
        json.dumps({"all": {"delta": {"point": 0.1, "lo": 0.0}}, "SupplierName": {"point": 1}}),
        encoding="utf-8",
    )
    dropped = [0]
    run = wl.optional_numeric_run(p, dropped)
    assert run.summary == {"all.delta.point": 0.1, "all.delta.lo": 0.0}
    assert dropped[0] == 1


def test_training_only_logs_the_config_and_the_named_run_dir_and_nothing_else(
    tmp_path: Path,
) -> None:
    rd = tmp_path / "ft_native_fold0_4c17aa3_bf16"
    rd.mkdir()
    (rd / "metrics.jsonl").write_text(
        json.dumps({"step": 1, "loss": 0.5, "eval_loss": 0.4}) + "\n", encoding="utf-8"
    )
    args = wl.build_parser().parse_args(
        ["--runs-root", str(tmp_path / "no_runs"), "--train-run-dir", str(rd), "--training-only"]
    )
    runs, status = wl.build_runs(args)
    assert [r.name for r in runs] == ["config_pins", rd.name]
    assert status["eval_runs"].startswith("skipped") and "rule_gate.md" not in status
    assert runs[1].summary["final_eval_loss"] == 0.4
    assert wl.check_runs(runs) == []


def test_finetune_config_option_changes_the_logged_resolution() -> None:
    native = wl.ROOT / "configs" / "finetune_qwen35_4b_native.yaml"
    args = wl.build_parser().parse_args(["--finetune-config", str(native)])
    assert args.finetune_config == native
    infer = wl.ROOT / "configs" / "spike_qwen35_4b_img_only_native.yaml"
    run = wl.config_run(args.finetune_config, infer, wl.ROOT / "uv.lock")
    assert run.config["finetune.max_pixels"] == 2196480
    assert run.config["inference.name"] == "qwen35_4b_img_only_native"
    assert wl.check_runs([run]) == []  # the native config name is in the allowlist vocabulary


# ---- slice JSON and optional training runs ------------------------------------------------------


def _slice_json(tmp_path: Path, slice_key: str = "scanned=yes") -> Path:
    row = {
        "slice": slice_key,
        "label": "free text that must never be logged: ACME-4711-XQ",
        "n": 43,
        "overall": {"point": 0.87, "lo": 0.81, "hi": 0.93},
        "delta": {"delta": 0.05, "lo": 0.02, "hi": 0.08},
        "raw_overall": 0.82,
        "header_acc": 0.997,
        "row_f1": 0.83,
        "fully_correct": 0.7,
        "false_fill": 0.0,
    }
    d = {
        "run": "zeroshot500_qwen35_4b_img_only_native_4c17aa3",
        "n_docs_dev": 100,
        "dev_slices": [row],
        "all500_slices": [{**row, "slice": "all"}],
        "fold_slices": [{**row, "slice": "fold=2"}],
    }
    p = tmp_path / "slices.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    return p


def test_slice_run_logs_scalars_and_tables_and_never_the_label(tmp_path: Path) -> None:
    run = wl.slice_run("v1_5_dev_slices", _slice_json(tmp_path))
    assert run.group == "eval"
    assert run.summary["dev100.scanned=yes.overall"] == 0.87
    assert run.summary["dev100.scanned=yes.delta_lo"] == 0.02
    assert run.summary["oof500.all.row_f1"] == 0.83 and "oof_fold.fold=2.n" in run.summary
    assert run.summary["n_docs_dev"] == 100
    assert set(run.tables) == {"dev100", "oof500", "oof_fold"}
    assert "ACME-4711-XQ" not in json.dumps(run.to_json())
    assert wl.check_runs([run]) == []


def test_slice_run_refuses_an_unknown_slice_key(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="vocabulary"):
        wl.slice_run("v1_5_dev_slices", _slice_json(tmp_path, slice_key=PLANTED))


def _v2_slice_json(tmp_path: Path, kind: str, **over: Any) -> Path:
    row = {
        "slice": "scanned=yes",
        "label": "free text that must never be logged: ACME-4711-XQ",
        "n": 43,
        "overall": {"point": 0.87, "lo": 0.81, "hi": 0.93},
        "delta": {"delta": 0.05, "lo": 0.02, "hi": 0.08},
        "raw_overall": 0.82,
        "header_acc": 0.997,
        "row_f1": 0.83,
        "fully_correct": 0.7,
        "false_fill": 0.0,
    }
    d = {
        "schema": "shipdoc-v2-slices/1",
        "kind": kind,
        "stand_in": False,
        "run": "oof_native_fold0_4c17aa3" if kind == "dev_run" else "a_4c17aa3,b_4c17aa3",
        "n_docs": 500 if kind == "pooled_oof" else 100,
        "slices": [{**row, "ft_minus_zs": {"delta": 0.012, "lo": -0.01, "hi": 0.03}}]
        if kind == "dev_oof"
        else [row],
        "zs_slices": [{**row, "overall": {"point": 0.85, "lo": 0.8, "hi": 0.9}}]
        if kind == "dev_oof"
        else [],
        "fold_slices": [{**row, "slice": "fold=1"}] if kind == "pooled_oof" else [],
        **over,
    }
    p = tmp_path / f"v2_{kind}.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    return p


def test_v2_dev_run_json_goes_to_v2_dev_slices(tmp_path: Path) -> None:
    run = wl.slice_run("v2_dev_slices", _v2_slice_json(tmp_path, "dev_run"))
    assert run.summary["dev100.scanned=yes.overall"] == 0.87 and run.summary["n_docs_dev"] == 100
    assert run.summary["source_run"] == "oof_native_fold0_4c17aa3"
    assert "stand_in" not in run.summary and set(run.tables) == {"dev100"}
    assert "ACME-4711-XQ" not in json.dumps(run.to_json()) and wl.check_runs([run]) == []


def test_v2_oof_json_replaces_it_as_v2_oof_slices(tmp_path: Path) -> None:
    run = wl.slice_run("v2_oof_slices", _v2_slice_json(tmp_path, "pooled_oof"))
    assert run.name == "v2_oof_slices" and run.summary["n_docs_all"] == 500
    assert run.summary["oof500.scanned=yes.n"] == 43 and "oof_fold.fold=1.n" in run.summary
    assert "source_run" not in run.summary  # three run names: not allowlisted, not logged
    assert set(run.tables) == {"oof500", "oof_fold"} and wl.check_runs([run]) == []


def test_v2_entries_carry_the_exact_evidence_label(tmp_path: Path) -> None:
    dev = wl.slice_run("v2_dev_slices", _v2_slice_json(tmp_path, "dev_run"))
    # no 08 artifact records the GPU: the L4 word needs GG's Colab banner paste (gpu_l4)
    assert dev.summary["evidence"] == "final adapter, dev100, fp16 inference"
    l4 = wl.slice_run("v2_dev_slices", _v2_slice_json(tmp_path, "dev_run"), gpu_l4=True)
    assert l4.summary["evidence"] == "final adapter, dev100, L4 fp16 inference"
    assert wl.check_runs([l4]) == []
    only_dev_run = wl.slice_run(
        "v2_dev_oof_slices", _v2_slice_json(tmp_path, "dev_oof"), gpu_l4=True
    )
    assert only_dev_run.summary["evidence"] == "dev100, supplier-held-out"  # L4 only for 08
    oof = wl.slice_run("v2_dev_oof_slices", _v2_slice_json(tmp_path, "dev_oof"))
    assert oof.summary["evidence"] == "dev100, supplier-held-out"
    pooled = wl.slice_run("v2_oof_slices", _v2_slice_json(tmp_path, "pooled_oof"))
    assert pooled.summary["evidence"] == "pooled OOF 500, supplier-held-out"
    assert wl.check_runs([dev, oof, pooled]) == []  # the labels are in the allowlist vocabulary


def test_dev_oof_json_logs_ft_zs_pair_and_the_zs_counterpart(tmp_path: Path) -> None:
    run = wl.slice_run("v2_dev_oof_slices", _v2_slice_json(tmp_path, "dev_oof"))
    assert run.summary["dev100.scanned=yes.overall"] == 0.87
    assert run.summary["dev100.scanned=yes.ft_minus_zs"] == 0.012
    assert run.summary["dev100.scanned=yes.ft_minus_zs_hi"] == 0.03
    assert run.summary["zs_dev100.scanned=yes.overall"] == 0.85
    assert set(run.tables) == {"dev100", "zs_dev100"} and run.summary["n_docs_dev"] == 100
    assert "ACME-4711-XQ" not in json.dumps(run.to_json()) and wl.check_runs([run]) == []


def test_a_partial_dev_oof_json_is_never_logged(tmp_path: Path) -> None:
    p = _v2_slice_json(tmp_path, "dev_oof", partial=True, folds=[0, 1])
    with pytest.raises(ValueError, match="PARTIAL"):
        wl.slice_run("v2_dev_oof_slices", p)


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("v2_oof_slices", "dev_run"),
        ("v2_dev_slices", "pooled_oof"),
        ("v2_dev_slices", "dev_oof"),
        ("v2_dev_oof_slices", "dev_run"),
    ],
)
def test_v2_entries_cannot_be_swapped(tmp_path: Path, name: str, kind: str) -> None:
    with pytest.raises(ValueError, match="belongs to the entry"):
        wl.slice_run(name, _v2_slice_json(tmp_path, kind))


def test_v2_entries_refuse_a_v1_5_json_and_v1_5_takes_no_v2_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="takes a v2 slice JSON"):
        wl.slice_run("v2_dev_slices", _slice_json(tmp_path))
    with pytest.raises(ValueError, match="unknown kind"):
        wl.slice_run("v2_dev_slices", _v2_slice_json(tmp_path, "bogus"))


def test_stand_in_slice_json_is_refused_for_a_sync_but_marked_in_a_dry_run(tmp_path: Path) -> None:
    p = _v2_slice_json(tmp_path, "dev_run", stand_in=True)
    with pytest.raises(ValueError, match="stand-in"):
        wl.slice_run("v2_dev_slices", p, allow_stand_in=False)
    assert wl.slice_run("v2_dev_slices", p).summary["stand_in"] is True


def test_build_runs_refuses_a_stand_in_only_when_syncing(tmp_path: Path) -> None:
    p = _v2_slice_json(tmp_path, "dev_run", stand_in=True)
    args = wl.build_parser().parse_args(
        ["--slice-json", f"v2_dev_slices={p}", "--train-run-dir-optional", str(tmp_path / "none")]
    )
    runs, _ = wl.build_runs(args)  # dry run: allowed, marked
    assert any(r.name == "v2_dev_slices" and r.summary.get("stand_in") for r in runs)
    args.sync = True
    with pytest.raises(ValueError, match="stand-in"):
        wl.build_runs(args)


@pytest.mark.parametrize("arg", ["x=a.json", "v1_5_dev_slices", "v1_5_dev_slices=", "=a.json"])
def test_slice_json_arg_needs_a_known_name_and_a_path(arg: str) -> None:
    with pytest.raises(ValueError, match="NAME=PATH"):
        wl.parse_slice_json_arg(arg)
    assert wl.parse_slice_json_arg("v2_dev_slices=D:/x/y.json") == (
        "v2_dev_slices",
        Path("D:/x/y.json"),
    )


def _fake_run_dir(root: Path, name: str, state: str | None) -> Path:
    rd = root / name
    (rd / "final").mkdir(parents=True)
    (rd / "metrics.jsonl").write_text(json.dumps({"step": 1, "loss": 0.5}) + "\n", encoding="utf-8")
    (rd / "final" / "manifest.json").write_text(json.dumps({"stage": "fold2"}), encoding="utf-8")
    if state is not None:
        (rd / "train_status.json").write_text(json.dumps({"state": state}), encoding="utf-8")
    return rd


def test_optional_training_dirs_include_complete_and_skip_the_rest(tmp_path: Path) -> None:
    done = _fake_run_dir(tmp_path, "ft_native_fold2_4c17aa3_bf16", "complete")
    _fake_run_dir(tmp_path, "ft_native_final_4c17aa3_bf16", "running")
    status: dict[str, str] = {}
    pats = [
        str(tmp_path / "ft_native_fold2_*"),
        str(tmp_path / "ft_native_final_*"),
        str(tmp_path / "ft_native_nothing_*"),
    ]
    assert wl.optional_training_dirs(pats, status) == [done]
    assert status["optional_training_run[ft_native_fold2_*]"] == "1 included"
    assert "incomplete" in status["optional_training_run[ft_native_final_*]"]
    assert status["optional_training_run[ft_native_nothing_*]"] == "skipped (absent)"
    assert all(str(tmp_path) not in v for v in status.values())


def test_build_runs_takes_fold_runs_slices_and_dedupes_optional(tmp_path: Path) -> None:
    f0 = _fake_run_dir(tmp_path, "ft_native_fold0_4c17aa3_bf16", "complete")
    f2 = _fake_run_dir(tmp_path, "ft_native_fold2_4c17aa3_bf16", "complete")
    args = wl.build_parser().parse_args(
        [
            "--runs-root",
            str(tmp_path / "no_runs"),
            "--train-run-dir",
            str(f0),
            "--train-run-dir-optional",
            str(f0),
            "--train-run-dir-optional",
            str(f2),
            "--train-run-dir-optional",
            str(tmp_path / "ft_native_final_*"),
            "--slice-json",
            f"v1_5_dev_slices={_slice_json(tmp_path)}",
            "--training-only",
        ]
    )
    runs, status = wl.build_runs(args)
    assert [r.name for r in runs] == ["config_pins", f0.name, f2.name, "v1_5_dev_slices"]
    assert status["slice_json[v1_5_dev_slices]"] == "found"
    assert wl.check_runs(runs) == []
