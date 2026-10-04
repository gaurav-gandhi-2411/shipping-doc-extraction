"""Tests for scripts/hf_upload.py: staging allowlist, sha256, leak check, card fill, refusals."""

from __future__ import annotations

import hashlib
import json
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import hf_upload as hf  # noqa: E402

PLANTED = "Zq Planted Supplier GmbH"  # a fake gold value, planted in the synthetic labels


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@pytest.fixture
def final_dir(tmp_path: Path) -> Path:
    """A synthetic ``<run>/final`` (adapter files, manifest, junk that must never be staged)."""
    run = tmp_path / "run"
    final = run / "final"
    (final / "peft").mkdir(parents=True)
    cfg, weights = b'{"r": 16}', b"\x00weights" * 50
    (final / "peft" / "adapter_config.json").write_bytes(cfg)
    (final / "peft" / "adapter_model.safetensors").write_bytes(weights)
    (final / "peft" / "README.md").write_text("generic peft card", encoding="utf-8")
    (final / "adapter.pt").write_bytes(b"pickle")
    (run / "trainer_state.pt").write_bytes(b"t" * 10)
    (run / "optimizer.pt").write_bytes(b"o" * 10)
    (run / "ckpt").mkdir()
    (run / "ckpt" / "step_000010.pt").write_bytes(b"c")
    manifest = {
        "stage": "fold0",
        "fold": 0,
        "code_sha": "42b812b5b09d6e4bff0df12564017f71ffad5fc9",
        "n_train_docs": 3,
        "train_doc_ids": ["dev_0001", "dev_0002", "dev_0003"],
        "heldout_doc_ids": ["dev_0004"],
        "lora": {"r": 16, "alpha": 32, "n_modules": 200, "n_trainable": 30474240},
        "inference_keys": {"prompt_version": "v2", "max_pixels": 1310720},
        "steps": 4,
        "peft_sha256": {
            "adapter_config.json": _sha(cfg),
            "adapter_model.safetensors": _sha(weights),
        },
    }
    (final / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run / "metrics.jsonl").write_text(json.dumps({"step": 1, "seconds": 3600.0}), encoding="utf-8")
    return final


@pytest.fixture
def labels_root(tmp_path: Path) -> Path:
    lab = tmp_path / "labels_root" / "train" / "labels"
    lab.mkdir(parents=True)
    (lab / "a.json").write_text(
        json.dumps({"header": {"supplier_name": PLANTED}, "line_items": []}), encoding="utf-8"
    )
    return tmp_path / "labels_root"


@pytest.fixture
def matcher(labels_root: Path) -> Any:
    return hf.gold_matcher(labels_root)


@pytest.fixture(autouse=True)
def _scratch_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    monkeypatch.setattr(hf, "TMP_ROOT", tmp_path)
    yield


# ---- sha256 verification ----------------------------------------------------------------------


def test_verify_adapter_accepts_the_manifest_hashes(final_dir: Path) -> None:
    got = hf.verify_adapter(final_dir)
    assert set(got) == set(hf.ADAPTER_FILES)


def test_verify_adapter_catches_a_tampered_adapter(final_dir: Path) -> None:
    f = final_dir / "peft" / "adapter_model.safetensors"
    f.write_bytes(f.read_bytes() + b"x")
    with pytest.raises(hf.StagingError, match="sha256 mismatch for adapter_model.safetensors"):
        hf.verify_adapter(final_dir)


def test_verify_adapter_needs_a_manifest_hash_and_the_file(final_dir: Path) -> None:
    man = json.loads((final_dir / "manifest.json").read_text(encoding="utf-8"))
    del man["peft_sha256"]["adapter_config.json"]
    (final_dir / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    with pytest.raises(hf.StagingError, match="no sha256"):
        hf.verify_adapter(final_dir)
    (final_dir / "peft" / "adapter_config.json").unlink()
    with pytest.raises(hf.StagingError, match="missing"):
        hf.verify_adapter(final_dir)


# ---- staging allowlist and leak check ---------------------------------------------------------


def test_staging_copies_only_allowlisted_files(
    final_dir: Path, tmp_path: Path, matcher: Any
) -> None:
    stage = tmp_path / "stage"
    staged = hf.stage_tree(final_dir, stage, "card text")
    names = sorted(p.name for p in staged)
    assert names == ["README.md", "adapter_config.json", "adapter_model.safetensors"]
    on_disk = sorted(p.relative_to(stage).as_posix() for p in stage.rglob("*") if p.is_file())
    assert on_disk == names
    for junk in (
        "trainer_state.pt",
        "optimizer.pt",
        "step_000010.pt",
        "adapter.pt",
        "manifest.json",
    ):
        assert not list(stage.rglob(junk))
    assert hf.check_staging(stage, matcher, allow_merged=False) == []


def test_staging_refuses_a_stale_different_file_and_keeps_identical_ones(
    final_dir: Path, tmp_path: Path
) -> None:
    stage = tmp_path / "stage"
    hf.stage_tree(final_dir, stage, "card")
    hf.stage_tree(final_dir, stage, "card")  # identical files are kept
    (stage / "adapter_config.json").write_bytes(b"different")
    with pytest.raises(hf.StagingError, match="already exists"):
        hf.stage_tree(final_dir, stage, "card")


@pytest.mark.parametrize(
    ("name", "content", "needle"),
    [
        ("extra.png", b"\x89PNG....", "not in the staging allowlist"),
        ("bundle.zip", b"PK\x03\x04....", "content is a zip"),
        ("doc.pdf", b"%PDF-1.4", "forbidden path or type"),
        ("trainer_state.pt", b"x", "forbidden path or type"),
        ("notes.txt", b"hello", "not in the staging allowlist"),
    ],
)
def test_check_staging_flags_images_zips_pdfs_and_strays(
    final_dir: Path, tmp_path: Path, matcher: Any, name: str, content: bytes, needle: str
) -> None:
    stage = tmp_path / "stage"
    hf.stage_tree(final_dir, stage, "card")
    (stage / name).write_bytes(content)
    problems = hf.check_staging(stage, matcher, allow_merged=False)
    assert any(name in p and needle in p for p in problems), problems


def test_check_staging_catches_renamed_zip_by_magic_bytes(
    final_dir: Path, tmp_path: Path, matcher: Any
) -> None:
    stage = tmp_path / "stage"
    hf.stage_tree(final_dir, stage, "PK\x03\x04 pretending to be text")
    problems = hf.check_staging(stage, matcher, allow_merged=False)
    assert any("README.md: content is a zip" in p for p in problems)


def test_card_with_a_planted_label_value_is_detected(
    final_dir: Path, tmp_path: Path, matcher: Any
) -> None:
    assert hf.leak_kinds(f"trained on {PLANTED} invoices", matcher) == {"header.supplier_name"}
    stage = tmp_path / "stage"
    hf.stage_tree(final_dir, stage, f"supplier: {PLANTED}")
    problems = hf.check_staging(stage, matcher, allow_merged=False)
    assert problems and "gold label value(s) present [header.supplier_name]" in problems[0]
    assert PLANTED not in " ".join(problems)  # kinds and paths only, never the value


def test_leak_check_fails_closed_without_gold(tmp_path: Path) -> None:
    empty = tmp_path / "no_gold"
    empty.mkdir()
    with pytest.raises(hf.StagingError, match="cannot run the leak check"):
        hf.gold_matcher(empty)


def test_merged_files_use_a_suffix_allowlist(final_dir: Path, tmp_path: Path, matcher: Any) -> None:
    merged = tmp_path / "merged_src"
    merged.mkdir()
    (merged / "model-00001.safetensors").write_bytes(b"w")
    (merged / "config.json").write_text("{}", encoding="utf-8")
    (merged / "optimizer.pt").write_bytes(b"o")
    stage = tmp_path / "stage"
    staged = hf.stage_tree(final_dir, stage, "card", merged)
    assert sorted(p.name for p in staged if p.parent.name == "merged") == [
        "config.json",
        "model-00001.safetensors",
    ]
    assert hf.check_staging(stage, matcher, allow_merged=True) == []
    assert hf.check_staging(stage, matcher, allow_merged=False)  # merged/ without the flag


# ---- paths ------------------------------------------------------------------------------------


def test_safe_paths_refuse_the_repo(tmp_path: Path) -> None:
    with pytest.raises(hf.StagingError, match="inside the repository"):
        hf.safe_stage_dir(hf.ROOT / "docs" / "stage")
    with pytest.raises(hf.StagingError, match="not under an allowed scratch root"):
        hf.safe_stage_dir(tmp_path.parent / "elsewhere")
    assert hf.safe_stage_dir(tmp_path / "ok") == (tmp_path / "ok").resolve()
    assert hf.assert_repo_write_allowed(hf.DRAFT_CARD) == hf.DRAFT_CARD.resolve()
    with pytest.raises(hf.StagingError):
        hf.assert_repo_write_allowed(hf.ROOT / "docs" / "model_card.md")
    assert hf.assert_repo_write_allowed(tmp_path / "x.md") == (tmp_path / "x.md").resolve()
    with pytest.raises(hf.StagingError, match="data/"):
        hf.assert_source_allowed(hf.ROOT / "data" / "train")


# ---- card -------------------------------------------------------------------------------------


def test_fill_card_resolves_known_and_marks_the_rest_pending() -> None:
    tmpl = hf.TEMPLATE.read_text(encoding="utf-8")
    text, resolved, pending = hf.fill_card(tmpl, {"model_name": "m", "epochs": "2"})
    assert "# m\n" in text and "2 epochs" in text
    assert {"model_name", "epochs"} <= set(resolved)
    assert "ft_overall" in pending and "[[PENDING: ft_overall]]" in text
    assert "{{" not in text.replace("{ {{", "")  # no unresolved placeholder survives
    assert "TEMPLATE:" not in text and "one of: fold K" not in text
    assert "[[PENDING: PINNED_SHA]]" in text and "<PINNED_SHA>" not in text
    assert "license: apache-2.0" in text and "TBD by GG" not in text
    assert "Adapter licence: Apache-2.0" in text
    assert "base_model_license" in pending  # only build_values supplies the cited base licence
    assert hf.BANNER.strip() in text
    assert text.count("[[PENDING") == len(
        [m for m in hf.PENDING_RE.finditer(text)]
    )  # the banner itself carries no marker


def test_fill_card_numbers_only_come_from_values() -> None:
    tmpl = hf.TEMPLATE.read_text(encoding="utf-8")
    text, _, pending = hf.fill_card(tmpl, {"ft_overall": "0.9"})
    assert "ft_overall" not in pending and "| 0.9 |" in text and "ft_header" in pending


def test_build_values_from_synthetic_artifacts(final_dir: Path, tmp_path: Path) -> None:
    hashes = hf.verify_adapter(final_dir)
    v = hf.build_values(final_dir, tmp_path / "no_runs", "My Model", None, hashes)
    assert v["lora_rank"] == "16" and v["trainable_params"] == "30,474,240"
    assert v["optimizer_steps"] == "4" and v["train_gpu_hours"].startswith("1.00")
    assert v["adapter_sha256"] == hashes["adapter_model.safetensors"]
    assert "fold 0 only (3 documents; 1 held out)" in v["released_adapter_scope"]
    assert v["citation_key"] == "my_model" and "adapter_repo_or_path" not in v
    assert "ft_overall" not in v and "zs_overall" not in v  # no source, so no number
    assert v["base_model_license"] == hf.BASE_LICENSE and "PENDING" not in hf.BASE_LICENSE
    assert v["carbon_estimate_or_not_measured"] == "NOT MEASURED"


# ---- refusals and main ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("approved", "env", "repo", "missing"),
    [
        (False, {"SHIPDOC_PUBLISH_OK": "1"}, "o/n", "--i-have-gg-approval"),
        (True, {}, "o/n", "SHIPDOC_PUBLISH_OK=1"),
        (True, {"SHIPDOC_PUBLISH_OK": "1"}, None, "--repo-id"),
        (False, {}, None, "--i-have-gg-approval, SHIPDOC_PUBLISH_OK=1, --repo-id"),
    ],
)
def test_upload_refused_unless_flag_env_and_repo_id(
    approved: bool, env: dict[str, str], repo: str | None, missing: str
) -> None:
    with pytest.raises(hf.PublishRefused, match=missing):
        hf.require_upload_approval(approved, env, repo)
    hf.require_upload_approval(True, {"SHIPDOC_PUBLISH_OK": "1"}, "o/n")


def test_main_upload_refuses_before_touching_anything(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    called: list[str] = []
    monkeypatch.setattr(hf, "hub_upload", lambda *a, **k: called.append("up"))
    monkeypatch.setattr(hf, "verify_adapter", lambda *_: called.append("read"))  # type: ignore[arg-type]
    assert hf.main(["--upload"], env={}) == 3
    assert hf.main(["--upload", "--i-have-gg-approval", "--repo-id", "o/n"], env={}) == 3
    assert hf.main(["--upload", "--repo-id", "o/n"], env={"SHIPDOC_PUBLISH_OK": "1"}) == 3
    assert hf.main(["--upload", "--i-have-gg-approval"], env={"SHIPDOC_PUBLISH_OK": "1"}) == 3
    full = ["--upload", "--i-have-gg-approval", "--repo-id", "o/n", "--merged-dir", str(tmp_path)]
    assert hf.main(full, env={"SHIPDOC_PUBLISH_OK": "1"}) == 3  # merged needs its own repo id
    assert called == []


class _NoEnv(dict):  # type: ignore[type-arg]
    def get(self, *a: Any, **k: Any) -> Any:
        raise AssertionError("the dry-run must not read credentials or approval env vars")

    __getitem__ = get  # type: ignore[assignment]


def _argv(final: Path, tmp_path: Path, labels: Path, *extra: str) -> list[str]:
    return [
        "--final-dir",
        str(final),
        "--runs-root",
        str(tmp_path / "no_runs"),
        "--stage-dir",
        str(tmp_path / "stage"),
        "--labels-root",
        str(labels),
        "--draft-card",
        str(tmp_path / "draft.md"),
        *extra,
    ]


def test_dry_run_end_to_end_reads_no_env_and_writes_only_scratch(
    final_dir: Path, tmp_path: Path, labels_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert hf.main(_argv(final_dir, tmp_path, labels_root), env=_NoEnv()) == 0
    out = capsys.readouterr().out
    assert "leak check: clean" in out and "PENDING: ft_overall" in out
    assert "adapter_model.safetensors" in out and _sha(b"\x00weights" * 50) in out
    assert "trainer_state" not in out and "optimizer" not in out
    assert (tmp_path / "draft.md").read_text(encoding="utf-8").startswith("---")
    staged = sorted(p.name for p in (tmp_path / "stage").iterdir())
    assert staged == ["README.md", "adapter_config.json", "adapter_model.safetensors"]


def test_dry_run_refuses_a_tampered_adapter_and_a_repo_stage_dir(
    final_dir: Path, tmp_path: Path, labels_root: Path
) -> None:
    (final_dir / "peft" / "adapter_config.json").write_bytes(b"tampered")
    assert hf.main(_argv(final_dir, tmp_path, labels_root), env={}) == 4
    assert not (tmp_path / "draft.md").exists()
    ok = ["--stage-dir", str(hf.ROOT / "docs" / "stage_should_not_exist")]
    assert hf.main(_argv(final_dir, tmp_path, labels_root) + ok, env={}) == 4
    assert not (hf.ROOT / "docs" / "stage_should_not_exist").exists()


def test_dry_run_fails_closed_without_gold(final_dir: Path, tmp_path: Path) -> None:
    empty = tmp_path / "empty_labels"
    empty.mkdir()
    assert hf.main(_argv(final_dir, tmp_path, empty), env={}) == 4
    assert not (tmp_path / "draft.md").exists()


def test_upload_refuses_while_the_card_has_pending_markers(
    final_dir: Path, tmp_path: Path, labels_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called: list[Any] = []
    monkeypatch.setattr(hf, "hub_upload", lambda *a: called.append(a))
    argv = _argv(final_dir, tmp_path, labels_root, "--upload", "--i-have-gg-approval")
    argv += ["--repo-id", "o/n"]
    assert hf.main(argv, env={"SHIPDOC_PUBLISH_OK": "1"}) == 7
    assert called == []


def test_upload_with_all_gates_and_a_clean_card_calls_hub_privately(
    final_dir: Path, tmp_path: Path, labels_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called: list[Any] = []
    monkeypatch.setattr(hf, "hub_upload", lambda *a: called.append(a))
    monkeypatch.setattr(hf, "fill_card", lambda t, v: ("clean card", [], []))
    argv = _argv(final_dir, tmp_path, labels_root, "--upload", "--i-have-gg-approval")
    argv += ["--repo-id", "o/n"]
    assert hf.main(argv, env={"SHIPDOC_PUBLISH_OK": "1"}) == 0
    assert len(called) == 1 and called[0][1] == "o/n"


def test_hub_upload_creates_a_private_repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    class FakeApi:
        def create_repo(self, repo_id: str, **kw: Any) -> None:
            calls.append(("create_repo", {"repo_id": repo_id, **kw}))

        def upload_folder(self, **kw: Any) -> None:
            calls.append(("upload_folder", kw))

    fake = types.ModuleType("huggingface_hub")
    fake.HfApi = FakeApi  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    hf.hub_upload(tmp_path, "o/n", None)
    assert calls[0][0] == "create_repo" and calls[0][1]["private"] is True
    assert calls[1][1]["ignore_patterns"] == ["merged/*"]
    assert len(calls) == 2


def test_card_file_is_staged_as_is_and_writes_no_draft(
    final_dir: Path, tmp_path: Path, labels_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    card = tmp_path / "card.md"
    card.write_text("# finished card\n\nno placeholders here\n", encoding="utf-8")
    argv = _argv(final_dir, tmp_path, labels_root, "--card-file", str(card))
    assert hf.main(argv, env=_NoEnv()) == 0
    assert (tmp_path / "stage" / "README.md").read_text(encoding="utf-8") == card.read_text(
        encoding="utf-8"
    )
    assert not (tmp_path / "draft.md").exists()
    assert "card: 0 placeholders resolved, 0 PENDING" in capsys.readouterr().out


def test_card_file_with_a_template_placeholder_or_a_gold_value_is_refused(
    final_dir: Path, tmp_path: Path, labels_root: Path
) -> None:
    card = tmp_path / "card.md"
    card.write_text("value {{ft_overall}}", encoding="utf-8")
    argv = _argv(final_dir, tmp_path, labels_root, "--card-file", str(card))
    assert hf.main(argv, env={}) == 4
    card.write_text(f"leaked {PLANTED}", encoding="utf-8")
    assert hf.main(argv, env={}) == 4
    assert not (tmp_path / "stage" / "README.md").exists()


def test_card_file_pending_marker_still_blocks_the_upload(
    monkeypatch: pytest.MonkeyPatch, final_dir: Path, tmp_path: Path, labels_root: Path
) -> None:
    called: list[Any] = []
    monkeypatch.setattr(hf, "hub_upload", lambda *a: called.append(a))
    card = tmp_path / "card.md"
    card.write_text("dev score [[PENDING: dev_score]]", encoding="utf-8")
    argv = _argv(final_dir, tmp_path, labels_root, "--card-file", str(card))
    argv += ["--upload", "--i-have-gg-approval", "--repo-id", "o/n"]
    assert hf.main(argv, env={"SHIPDOC_PUBLISH_OK": "1"}) == 7
    assert called == []
