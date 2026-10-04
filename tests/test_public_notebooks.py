"""The public build of the Colab notebooks (configs/public_notebooks.json): no token, own pin.

The private repo has no such file, so its notebooks must not change (the per-notebook tests assert
that); here: the config loader, the per-cell rewrites, the public clone cell on real git
repositories (local paths stand in for the public URL: only the git mechanics are exercised, not
https) and an end-to-end build of the native notebooks from a copy of ``scripts/`` + ``configs/``.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "colab_build_notebook_pub", ROOT / "scripts" / "colab_build_notebook.py"
)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

SHA = "a" * 40
URL = "https://github.com/example-owner/example-repo.git"
NATIVE = ("02n_zeroshot500_native", "03n_finetune_native", "05n_oof_infer_native", "08_dev_final",
          "04c_predict_test_native")  # fmt: skip
BUILDERS = ("colab_build_zs_native", "colab_build_finetune_native", "colab_build_oof_native",
            "colab_build_dev_final_native", "colab_build_predict_native",
            "colab_build_public_smoke")  # fmt: skip


def _cfg(path: Path, **kw: Any) -> Path:
    path.write_text(json.dumps({"repo_url": URL, "pinned_sha": SHA, **kw}), encoding="utf-8")
    return path


def test_the_private_repo_has_no_public_config_and_keeps_its_clone_cell() -> None:
    """Holds in both trees: the private one has no config, the public one always has it."""
    if not base.PUBLIC_CONFIG_PATH.exists():
        assert base.PUBLIC is None
        assert base.public_pin("1" * 40) == "1" * 40
        assert "GH_TOKEN" in base.CLONE and "x-access-token" in base.CLONE
    else:
        assert base.load_public_config() == base.PUBLIC
        assert base.public_pin("1" * 40) == base.PUBLIC["pinned_sha"]
        assert "GH_TOKEN" not in base.CLONE and base.PUBLIC["repo_url"] in base.CLONE


def test_config_loader_validates_and_fails_closed(tmp_path: Path) -> None:
    assert base.load_public_config(tmp_path / "absent.json") is None
    ok = base.load_public_config(_cfg(tmp_path / "ok.json"))
    assert ok == {"repo_url": URL, "pinned_sha": SHA}
    placeholder = _cfg(tmp_path / "ph.json", pinned_sha=base.PIN_PLACEHOLDER)
    assert base.load_public_config(placeholder)["pinned_sha"] == "FILL_PINNED_SHA"
    for bad in ({"pinned_sha": "abc123"}, {"pinned_sha": "A" * 40}, {"repo_url": ""},
                {"repo_url": " x "}, {"extra": 1}):  # fmt: skip
        with pytest.raises(ValueError):
            base.load_public_config(_cfg(tmp_path / "bad.json", **bad))
    (tmp_path / "list.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly the keys"):
        base.load_public_config(tmp_path / "list.json")


def test_public_cell_drops_the_token_and_the_account_check_and_the_local_paths() -> None:
    sec = base.public_cell("code", base.ZS_SECRETS)
    assert "GH_TOKEN" not in sec and 'print("HF_TOKEN:"' in sec and "get_secret" in sec
    mount = base.public_cell("code", base.MOUNT)
    assert "EXPECTED_ACCOUNT" not in mount and "gcloud" not in mount and "drive.mount" in mount
    assert "Public build" in mount
    assert "Drive must be mounted" not in base.public_cell("markdown", base.ACCOUNT_MD)
    banner = 'LOCAL_DEST = BS.join(["D:", "shipdoc", "submissions", SUB_NAME]) + BS'
    assert base.public_cell("code", banner).count("$SHIPDOC_SUBMISSIONS_DIR") == 1
    assert base.public_cell("markdown", "into D:\\shipdoc\\runs\\x\\").startswith(
        "into $SHIPDOC_RUNS_DIR"
    )
    other = "x = 1\n"
    assert base.public_cell("code", other) == other


def test_native_parameter_defaults_share_the_pin_but_other_cells_keep_their_values() -> None:
    params = (
        'ZS_SHA7 = "FILL_ZS_SHA7"\nTRAIN_SHA7 = "FILL_TRAIN_SHA7"\n'
        'REUSE_V0_DIR = "submissions/v0_42b812b"  # x\nREUSE_OCR_FROM = "submissions/v1_8833c73"\n'
    )
    out = base.public_cell("code", params)
    assert out.count("PINNED_SHA[:7]") == 2
    assert "REUSE_V0_DIR = None" in out and "REUSE_OCR_FROM = None" in out
    old = 'REUSE_V0_DIR = "submissions/v0_42b812b"\n'  # a 1260-token (archived) notebook
    assert base.public_cell("code", old) == old


def _git(cwd: Path, *args: str) -> str:
    env = ["-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false"]
    return subprocess.run(["git", *env, *args], cwd=cwd, check=True, capture_output=True,
                          text=True).stdout.strip()  # fmt: skip


def _remote(tmp_path: Path) -> tuple[Path, str, str]:
    """A local 'public' repo with two commits; returns (path, first sha, second sha)."""
    r = tmp_path / "remote"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    (r / "uv.lock").write_text("lock\n", encoding="utf-8")
    (r / "splits").mkdir()
    (r / "splits" / "smoke5.json").write_text("[]\n", encoding="utf-8")
    _git(r, "add", ".")
    _git(r, "commit", "-q", "-m", "one")
    first = _git(r, "rev-parse", "HEAD")
    (r / "uv.lock").write_text("lock2\n", encoding="utf-8")
    _git(r, "commit", "-q", "-am", "two")
    return r, first, _git(r, "rev-parse", "HEAD")


def _run_clone(tmp_path: Path, url: Path, pin: str) -> dict[str, Any]:
    cell = base.PUBLIC_CLONE.replace("@@REPO_URL@@", url.as_posix())
    work = (tmp_path / "content" / "shipdoc-extract").as_posix()
    cell = cell.replace("/content/shipdoc-extract", work).replace(
        '"/content"', f'"{tmp_path.as_posix()}"'
    )
    ns: dict[str, Any] = {"PINNED_SHA": pin}
    exec(cell, ns)
    return ns


def test_the_public_clone_cell_clones_without_credentials_and_pins_the_commit(
    tmp_path: Path,
) -> None:
    remote, first, second = _remote(tmp_path)
    (tmp_path / "content").mkdir()
    ns = _run_clone(tmp_path, remote, first)  # an OLDER commit than the remote's tip
    assert ns["head"] == first != second
    assert (tmp_path / "content" / "shipdoc-extract" / "uv.lock").read_text() == "lock\n"
    assert "token" not in base.PUBLIC_CLONE.lower() and "Authorization" not in base.PUBLIC_CLONE


def test_the_public_clone_cell_refuses_a_missing_pin_and_a_placeholder(tmp_path: Path) -> None:
    remote, _first, _second = _remote(tmp_path)
    (tmp_path / "content").mkdir()
    with pytest.raises(RuntimeError, match="git checkout failed"):
        _run_clone(tmp_path, remote, "b" * 40)  # not a commit of that repository
    with pytest.raises(ValueError, match="PINNED_SHA"):
        _run_clone(tmp_path, remote, base.PIN_PLACEHOLDER)
    placeholder = base.PUBLIC_CLONE.replace("@@REPO_URL@@", base.URL_PLACEHOLDER)
    with pytest.raises(ValueError, match="REPO_URL"):
        exec(placeholder, {"PINNED_SHA": SHA})


@pytest.fixture(scope="module")
def public_tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A copy of scripts/ + configs/ with a public config; the native builders are run in it."""
    tree = tmp_path_factory.mktemp("public_tree")
    shutil.copytree(
        ROOT / "scripts", tree / "scripts", ignore=shutil.ignore_patterns("__pycache__")
    )
    shutil.copytree(ROOT / "configs", tree / "configs")
    shutil.copytree(ROOT / "splits", tree / "splits")  # the smoke / bench lists the builders read
    _cfg(tree / "configs" / "public_notebooks.json")
    for name in BUILDERS:
        res = subprocess.run([sys.executable, f"scripts/{name}.py"], cwd=tree, capture_output=True,
                             text=True, check=False)  # fmt: skip
        assert res.returncode == 0, (name, res.stderr[-600:])
    return tree


def _cells(path: Path) -> list[str]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"]]


def test_native_notebooks_of_the_public_tree_clone_the_public_url_at_the_public_pin(
    public_tree: Path,
) -> None:
    for name in (*NATIVE, "public_smoke"):
        text = "\n".join(_cells(public_tree / "notebooks" / f"{name}.ipynb"))
        assert f'PINNED_SHA = "{SHA}"' in text, name
        assert f'REPO_URL = "{URL}"' in text, name
        for word in base.PUBLIC_FORBIDDEN + ("ghp_",):
            assert word not in text, (name, word)
        assert "assert head == pinned_full" in text and '"FILL" in PINNED_SHA' in text, name


def test_the_private_pin_is_gone_from_every_public_native_notebook(public_tree: Path) -> None:
    for name in (*NATIVE, "public_smoke"):
        text = "\n".join(_cells(public_tree / "notebooks" / f"{name}.ipynb"))
        assert "4c17aa3" not in text.replace("4c17aa3 vs", ""), name  # no private sha in the cells


def test_public_default_parameters_need_no_manual_edit(public_tree: Path) -> None:
    ns: dict[str, Any] = {}
    src = next(s for s in _cells(public_tree / "notebooks" / "04c_predict_test_native.ipynb")
               if s.startswith("# Parameters."))  # fmt: skip
    exec(src, ns)
    assert ns["ZS_SHA7"] == ns["TRAIN_SHA7"] == SHA[:7] and ns["MODEL"] == "zs"
    assert ns["REUSE_V0_DIR"] is None and ns["REUSE_OCR_FROM"] is None
    assert ns["ZS_RUN_DIR"] == f"zeroshot500_qwen35_4b_img_only_native_{SHA[:7]}"


def test_a_placeholder_pin_makes_every_public_notebook_refuse(
    public_tree: Path, tmp_path: Path
) -> None:
    tree = tmp_path / "ph"
    shutil.copytree(public_tree, tree)
    _cfg(tree / "configs" / "public_notebooks.json", pinned_sha=base.PIN_PLACEHOLDER)
    for name in BUILDERS:
        res = subprocess.run([sys.executable, f"scripts/{name}.py"], cwd=tree, capture_output=True,
                             text=True, check=False)  # fmt: skip
        assert res.returncode == 0, (name, res.stderr[-400:])
    for nb in (*NATIVE, "public_smoke"):
        cells = _cells(tree / "notebooks" / f"{nb}.ipynb")
        params = next(s for s in cells if s.startswith(("# Parameters", "import json\nimport os")))
        with pytest.raises(ValueError, match="PINNED_SHA"):
            exec(params, {})


def test_every_public_notebook_cell_compiles_and_fits_100_columns(public_tree: Path) -> None:
    for name in (*NATIVE, "public_smoke"):
        nb = json.loads((public_tree / "notebooks" / f"{name}.ipynb").read_text(encoding="utf-8"))
        for c in nb["cells"]:
            src = "".join(c["source"])
            if c["cell_type"] == "code":
                assert c["outputs"] == [] and c["execution_count"] is None
                compile(src, f"{name}", "exec")
                assert all(len(ln) <= 100 for ln in src.splitlines()), (name, src[:50])
    assert re.fullmatch(r"[0-9a-f]{40}", SHA)
