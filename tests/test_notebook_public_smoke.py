"""notebooks/public_smoke.ipynb: structure, the pin refusal, the data extraction, the banner.

Cells are exec'd on the CPU with ``/content`` redirected into a tmp dir and a fake ``run_stream``;
nothing here touches Colab, Drive, a GPU or the network (the git clone uses a local repo).
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "public_smoke.ipynb"
spec = importlib.util.spec_from_file_location(
    "colab_build_public_smoke", ROOT / "scripts" / "colab_build_public_smoke.py"
)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

NB = json.loads(NB_PATH.read_text(encoding="utf-8"))
TITLE, PARAMS, ACCOUNT, MOUNT, CLONE, VERIFY, UNZIP, INSTALL, SMOKE, DETERMINISM, BANNER = range(11)
FAKE_SHA = "abcdef1" + "0" * 33
SMOKE_IDS = ["dev_0001", "dev_0006", "dev_0009", "dev_0010", "dev_0045"]


def _src(i: int) -> str:
    return "".join(NB["cells"][i]["source"])


def _redirect(src: str, root: Path) -> str:
    """Point every hard-coded /content path of a cell into ``root``."""
    return src.replace('"/content', f'"{root.as_posix()}')


def test_committed_notebook_matches_builder() -> None:
    assert NB_PATH.read_text(encoding="utf-8") == builder.render()


def test_cells_compile_outputs_cleared_and_lines_fit_100_columns() -> None:
    assert len(NB["cells"]) == 11
    for i, c in enumerate(NB["cells"]):
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None
            ast.parse(_src(i))
            assert all(len(ln) <= 100 for ln in _src(i).splitlines()), i


def test_no_token_secret_or_local_path_anywhere() -> None:
    text = "\n".join(_src(i) for i in range(len(NB["cells"])))
    drive, home = "D:" + "\\shipdoc", "C:" + "\\Users"  # joined here so no scan finds them as text
    account = "gauravgandhi" + "429"  # joined: the public tree must not contain the literal
    for word in ("GH_TOKEN", "x-access-token", "Authorization", "ANTHROPIC", "ghp_", "github_pat",
                 drive, account, home):  # fmt: skip
        assert word not in text, word
    assert "userdata" not in text  # no Colab secret is read at all
    assert "USE_WANDB = False" in _src(PARAMS) and "get_secret" not in text  # nothing is read


def test_pin_and_url_follow_the_config_and_a_placeholder_refuses() -> None:
    """No config (the private repo): placeholders. Public tree: whatever its config says."""
    cfg = builder.base.load_public_config()
    pin = cfg["pinned_sha"] if cfg else "FILL_PINNED_SHA"
    url = cfg["repo_url"] if cfg else "FILL_PUBLIC_REPO_URL"
    assert f'PINNED_SHA = "{pin}"' in _src(PARAMS) and f'REPO_URL = "{url}"' in _src(CLONE)
    if "FILL" in pin:
        with pytest.raises(ValueError, match="PINNED_SHA"):
            exec(_src(PARAMS), {})
    else:
        exec(_src(PARAMS), {})
    assert '"FILL" in PINNED_SHA' in _src(CLONE) and "assert head == pinned_full" in _src(CLONE)


def _params(sha: str = FAKE_SHA, **overrides: Any) -> dict[str, Any]:
    src = re.sub(r'(?m)^PINNED_SHA = ".*"$', f'PINNED_SHA = "{sha}"', _src(PARAMS))
    for name, value in overrides.items():
        src, n = re.subn(rf"(?m)^{name} = .*$", f"{name} = {value!r}", src)
        assert n == 1, name
    ns: dict[str, Any] = {}
    exec(src, ns)
    return ns


def test_parameters_cell_contract() -> None:
    ns = _params()
    assert ns["RUN_ID"] == "publicsmoke_qwen35_4b_img_only_native_abcdef1"
    assert ns["MODE"] == "run" and ns["USE_WANDB"] is False and ns["HF_TOKEN"] is None
    assert ns["SMOKE_LIMIT"] == 5 and ns["SMOKE_DOCS"] == "splits/smoke5.json"
    assert ns["DETERMINISM_DOCS"] == 2 and ns["CONFIG"] == "qwen35_4b_img_only_native"
    smoke5 = json.loads((ROOT / "splits" / "smoke5.json").read_text(encoding="utf-8"))
    assert smoke5 == SMOKE_IDS  # the ids this test's synthetic zip is built from
    for bad in (-1, 6, 2.5, "2"):
        with pytest.raises(ValueError, match="DETERMINISM_DOCS"):
            _params(DETERMINISM_DOCS=bad)


def test_the_smoke_and_install_cells_are_the_02n_cells() -> None:
    zs = builder.zs
    assert _src(SMOKE) == zs.SMOKE and _src(INSTALL) == zs.INSTALL
    assert 'uv", "sync", "--frozen"' in _src(INSTALL) and "check_smoke" not in _src(SMOKE)
    assert "SMOKE GATE FAILED" in _src(SMOKE) and "load_and_check" in _src(SMOKE)
    assert "require_logprobs=True" in _src(SMOKE)


def _synthetic_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for d in (*SMOKE_IDS, "dev_0002", "train_0001"):
            split = d.split("_")[0]
            zf.writestr(f"data/{split}/images/{d}_p1.jpg", b"img")
            zf.writestr(f"data/{split}/labels/{d}.json", json.dumps({"pages": [1]}))
        zf.writestr("data/dev/images/dev_0001_p2.jpg", b"img2")


def _ns_with_drive(tmp_path: Path) -> dict[str, Any]:
    drive = tmp_path / "drive" / "MyDrive" / "shipdoc-extract"
    drive.mkdir(parents=True)
    _synthetic_zip(drive / "data.zip")
    ns = _params()
    ns.update(DRIVE_DIR=drive, REPO=ROOT, Path=Path, json=json)
    import shutil

    ns["shutil"], ns["zipfile"] = shutil, zipfile
    return ns


def test_unzip_extracts_only_the_five_smoke_documents(tmp_path: Path) -> None:
    ns = _ns_with_drive(tmp_path)
    exec(_redirect(_src(UNZIP), tmp_path), ns)
    imgs = sorted(p.name for p in (tmp_path / "data" / "dev" / "images").iterdir())
    labels = sorted(p.name for p in (tmp_path / "data" / "dev" / "labels").iterdir())
    assert imgs == sorted([f"{d}_p1.jpg" for d in SMOKE_IDS] + ["dev_0001_p2.jpg"])
    assert labels == sorted(f"{d}.json" for d in SMOKE_IDS)  # dev_0002 / train_0001 not read
    assert not (tmp_path / "data" / "train").exists()


def test_unzip_refuses_a_data_zip_without_a_smoke_document(tmp_path: Path) -> None:
    ns = _ns_with_drive(tmp_path)
    with zipfile.ZipFile(ns["DRIVE_DIR"] / "data.zip", "w") as zf:
        zf.writestr("data/dev/labels/dev_0001.json", "{}")
    with pytest.raises(AssertionError, match="no label file|no images"):
        exec(_redirect(_src(UNZIP), tmp_path), ns)


def _git(cwd: Path, *args: str) -> str:
    env = ["-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false"]
    return subprocess.run(["git", *env, *args], cwd=cwd, check=True, capture_output=True,
                          text=True).stdout.strip()  # fmt: skip


def _clone(tmp_path: Path, pin_of: str = "first") -> dict[str, Any]:
    remote = tmp_path / "remote"
    remote.mkdir()
    _git(remote, "init", "-q", "-b", "main")
    (remote / "uv.lock").write_text("a\n", encoding="utf-8")
    (remote / "splits").mkdir()
    (remote / "splits" / "smoke5.json").write_text("[]\n", encoding="utf-8")
    _git(remote, "add", ".")
    _git(remote, "commit", "-q", "-m", "one")
    first = _git(remote, "rev-parse", "HEAD")
    (remote / "uv.lock").write_text("b\n", encoding="utf-8")
    _git(remote, "commit", "-q", "-am", "two")
    pin = first if pin_of == "first" else "c" * 40
    (tmp_path / "content").mkdir()
    cell = re.sub(r'(?m)^REPO_URL = ".*"$', f'REPO_URL = "{remote.as_posix()}"', _src(CLONE))
    cell = _redirect(cell, tmp_path)
    ns: dict[str, Any] = {"PINNED_SHA": pin}
    exec(cell, ns)
    return ns


def test_clone_pins_the_commit_and_verify_accepts_a_clean_tree(tmp_path: Path) -> None:
    ns = _clone(tmp_path)
    ns.update(Path=Path)
    exec(_src(VERIFY), ns)
    assert ns["TREE_SHA"] and len(ns["TREE_SHA"]) == 40


def test_verify_refuses_a_modified_tree(tmp_path: Path) -> None:
    ns = _clone(tmp_path)
    (ns["REPO"] / "uv.lock").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="differs from the pinned commit"):
        exec(_src(VERIFY), ns)


def test_clone_refuses_a_pin_that_is_not_a_commit_of_the_repository(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="git checkout failed"):
        _clone(tmp_path, pin_of="missing")


def _write_trace(run_dir: Path, texts: dict[str, list[str]]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"doc_id": d, "pages": [{"raw_text": t} for t in ts]})
             for d, ts in texts.items()]  # fmt: skip
    (run_dir / "trace.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _det_ns(tmp_path: Path, second: dict[str, list[str]]) -> dict[str, Any]:
    first = {"dev_0001": ["a", "b"], "dev_0006": ["c"], "dev_0009": ["d"]}
    runs = tmp_path / "runs" / "smoke"
    ns = _params()
    ns.update(json=json, time=__import__("time"), SMOKE_PASSED=True, PY="python",
              SMOKE_RUNS=runs, SMOKE_RUN_ID="smoke_x", SMOKE_ENV={})  # fmt: skip
    _write_trace(runs / "smoke_x", first)

    def fake(cmd: list[str], tail: int = 40, env: dict | None = None) -> Any:
        assert cmd[cmd.index("--limit") + 1] == "2" and "--logprobs" in cmd
        _write_trace(runs / cmd[cmd.index("--run-id") + 1], second)
        return 0, ["ok"]

    ns["run_stream"] = fake
    return ns


def test_determinism_replay_reports_identical_and_different(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    same = _det_ns(tmp_path / "a", {"dev_0001": ["a", "b"], "dev_0006": ["c"]})
    exec(_src(DETERMINISM), same)
    assert (
        same["DET"]["identical"] is True
        and "byte-identical raw text: True" in capsys.readouterr().out
    )
    other = _det_ns(tmp_path / "b", {"dev_0001": ["a", "X"], "dev_0006": ["c"]})
    exec(_src(DETERMINISM), other)
    assert other["DET"]["identical"] is False
    off = _det_ns(tmp_path / "c", {})
    off["DETERMINISM_DOCS"] = 0
    exec(_src(DETERMINISM), off)
    assert off["DET"]["ran"] is False and "skipped" in capsys.readouterr().out


def _banner_ns(tmp_path: Path, state: str) -> dict[str, Any]:
    ns = _params()
    runs = tmp_path / "runs"
    runs.mkdir(parents=True)
    checks = [{"name": "json_valid_rate", "passed": True, "detail": "1.0"},
              {"name": "no_truncation", "passed": state == "passed", "detail": "x"}]  # fmt: skip
    status = {"state": state, "seconds": 301.5, "checks": checks}
    (runs / "publicsmoke_smoke_status.json").write_text(json.dumps(status), encoding="utf-8")
    ns.update(json=json, time=__import__("time"), RUNS_DIR=runs, PY=sys.executable,
              REPO_URL="https://example.invalid/r.git", TREE_SHA="f" * 40,
              DET={"ran": True, "identical": True, "seconds": 90.0})  # fmt: skip
    return ns


def test_banner_prints_the_pin_versions_gpu_result_and_no_value(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ns = _banner_ns(tmp_path, "passed")
    exec(_src(BANNER), ns)
    out = capsys.readouterr().out
    for needle in ("PUBLIC SMOKE BANNER", FAKE_SHA, "torch / transformers / xgrammar", "GPU",
                   "smoke gate   : PASSED - 2/2 assertions passed, 301.5 s", "identical=True",
                   "DONE publicsmoke_qwen35_4b_img_only_native_abcdef1 smoke=passed"):  # fmt: skip
        assert needle in out, needle


def test_banner_fails_loudly_when_the_smoke_gate_did_not_pass(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ns = _banner_ns(tmp_path, "failed")
    with pytest.raises(AssertionError, match="SMOKE GATE DID NOT PASS"):
        exec(_src(BANNER), ns)
    assert "FAIL  no_truncation" in capsys.readouterr().out
