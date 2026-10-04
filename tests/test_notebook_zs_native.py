"""notebooks/02n_zeroshot500_native.ipynb: structure, native-specific cells, behaviour on the CPU.

Same approach as tests/test_notebook_zeroshot.py: cells are exec'd in one namespace with a fake
``run_stream`` that returns canned results or runs the REAL CLI on the mock backend. Nothing here
talks to Colab, Drive, a GPU or the network.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "02n_zeroshot500_native.ipynb"
spec = importlib.util.spec_from_file_location(
    "colab_build_zs_native", ROOT / "scripts" / "colab_build_zs_native.py"
)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

NB = json.loads(NB_PATH.read_text(encoding="utf-8"))
# Cell indices: the same 15 cells, in the same order, as notebook 02
(TITLE, PARAMS, ACCOUNT, MOUNT, SECRETS, CLONE, UNZIP, SHARDS, ESTIMATE, INSTALL, SMOKE, BENCH,
 RUN, MERGE, BANNER) = range(15)  # fmt: skip
FAKE_SHA = "abcdef1" + "0" * 33
PLACEHOLDER = "FILL_PINNED_SHA"
NATIVE_CFG = "qwen35_4b_img_only_native"
needs_data = pytest.mark.skipif(
    not (
        (ROOT / "assignment" / "score.py").is_file()
        and (ROOT / "data" / "train" / "labels").is_dir()
        and (ROOT / "data" / "dev" / "labels").is_dir()
    ),
    reason="assignment/score.py or data/{train,dev} absent (gitignored)",
)


def _src(i: int) -> str:
    return "".join(NB["cells"][i]["source"])


def _code_cells() -> list[str]:
    return [_src(i) for i, c in enumerate(NB["cells"]) if c["cell_type"] == "code"]


# ---------------------------------------------------------------------- structure


def test_committed_notebook_matches_builder() -> None:
    assert NB_PATH.read_text(encoding="utf-8") == builder.render()


def test_outputs_cleared_cells_compile_and_lines_fit_100_columns() -> None:
    assert len(NB["cells"]) == 15 and [c["cell_type"] for c in NB["cells"]].count("code") == 13
    for i, c in enumerate(NB["cells"]):
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None
            ast.parse(_src(i))
            for ln in _src(i).splitlines():
                assert len(ln) <= 100, (i, ln)


def test_cells_match_notebook_02_apart_from_the_declared_native_edits() -> None:
    """Unchanged 02 cells are byte-identical; only the cells the builder edits differ."""
    zs = json.loads((ROOT / "notebooks" / "archive" / "02_zeroshot500.ipynb").read_text(encoding="utf-8"))
    same = [i for i in range(15) if zs["cells"][i]["source"] == NB["cells"][i]["source"]]
    assert same == [ACCOUNT, MOUNT, SECRETS, CLONE, UNZIP, SHARDS, SMOKE, RUN, MERGE, BANNER]
    differ = sorted(set(range(15)) - set(same))
    assert differ == [TITLE, PARAMS, ESTIMATE, INSTALL, BENCH]


def test_pin_placeholder_blocks_the_notebook_until_the_pin_commit_fills_it() -> None:
    pin = builder.PINNED_SHA
    if pin == PLACEHOLDER:
        with pytest.raises(ValueError, match="PINNED_SHA"):
            exec(_src(PARAMS), {})  # fails on the first code cell, before Drive is touched
        assert f'PINNED_SHA = "{PLACEHOLDER}"' in _src(PARAMS)
    else:  # after the pin commit: a full SHA that exists in this repo
        assert re.fullmatch(r"[0-9a-f]{40}", pin)
        subprocess.run(["git", "cat-file", "-e", f"{pin}^{{commit}}"], cwd=ROOT, check=True)
    assert '"FILL" in PINNED_SHA' in _src(CLONE) and "assert head == pinned_full" in _src(CLONE)


def _params(sha: str = FAKE_SHA, **overrides: Any) -> dict[str, Any]:
    src = _src(PARAMS).replace(builder.PINNED_SHA, sha)
    for name, value in overrides.items():
        src, n = re.subn(rf"(?m)^{name} = .*$", f"{name} = {value!r}", src)
        assert n == 1, name
    ns: dict[str, Any] = {}
    exec(src, ns)
    return ns


def test_parameters_cell_contract_and_native_config() -> None:
    ns = _params()
    assert ns["CONFIG"] == NATIVE_CFG
    assert ns["RUN_ID"] == "zeroshot500_qwen35_4b_img_only_native_abcdef1"
    assert ns["SPLIT"] == "train+dev" and ns["DOCS"] == "splits/zeroshot500.json"
    assert ns["EXPECTED_DOCS"] == 500 and ns["EXPECTED_PAGES"] == 671
    assert ns["BATCH_SIZE"] is None and ns["SHARD"] == "0/1" and ns["MODE"] == "run"
    assert ns["BENCH_DOCS"] == "splits/bench12.json" and ns["USE_WANDB"] is False
    cfg = yaml.safe_load((ROOT / "configs" / f"spike_{ns['CONFIG']}.yaml").read_text())
    assert cfg["max_pixels"] == 2_196_480  # native: 2145 tokens per page
    assert (cfg["output_format"], cfg["prompt_version"], cfg["arm"]) == ("json", "v2", "img_only")
    assert cfg["name"] == ns["CONFIG"]


def test_run_ids_are_distinct_from_every_1260_run() -> None:
    ns = _params()
    old = f"zeroshot500_qwen35_4b_img_only_keyed_{FAKE_SHA[:7]}"  # notebook 02's run id
    assert ns["RUN_ID"] != old and "keyed" not in ns["RUN_ID"]
    sharded = _params(SHARD="1/2")
    assert sharded["SHARD_RUN_ID"] == ns["RUN_ID"] + "_shard1of2" != old + "_shard1of2"
    # smoke / bench folders are derived from the config name and the sha: native too
    assert f"smoke_{ns['RUN_ID']}" != f"smoke_{old}"
    assert f"bench_{ns['CONFIG']}_{ns['SHA7']}" != f"bench_qwen35_4b_img_only_{ns['SHA7']}"


def test_parameter_validation() -> None:
    for bad in ("1", "2/2", "0/0", "a/b", "-1/2", "1/2/3", ""):
        with pytest.raises(ValueError, match="SHARD"):
            _params(SHARD=bad)
    for bad in (0, -1, 2.5, "4", True):
        with pytest.raises(ValueError, match="BATCH_SIZE"):
            _params(BATCH_SIZE=bad)
    with pytest.raises(ValueError, match="MODE"):
        _params(MODE="merged")
    with pytest.raises(ValueError, match="K >= 2"):
        _params(MODE="merge")
    assert _params(BATCH_SIZE=4)["BATCH_SIZE"] == 4 and _params(MODE="merge", SHARD="0/2")


def test_estimate_precedes_install_smoke_and_run_and_spends_nothing() -> None:
    est = _src(ESTIMATE)
    for forbidden in ("run_stream(", '-m", "shipdoc', "uv sync", "from_pretrained"):
        assert forbidden not in est
    assert "nativerun" in est and "format_zs_estimate" in est and "zeroshot_report" not in est
    assert UNZIP < SHARDS < ESTIMATE < INSTALL < SMOKE < BENCH < RUN < MERGE < BANNER
    for i in range(ESTIMATE):
        assert 'shipdoc", "spike' not in _src(i) and 'uv", "sync' not in _src(i)


def test_native_install_sets_the_allocator_and_nothing_else_changes() -> None:
    inst = _src(INSTALL)
    assert '"PYTORCH_ALLOC_CONF": "expandable_segments:True"' in inst
    assert "--frozen" in inst and '"--group", "vlm"' in inst and "MODE" in inst


def test_every_command_uses_the_native_config_and_never_the_1260_one() -> None:
    text = "\n".join(_code_cells())
    assert "configs/spike_{CONFIG}.yaml" in text  # derived from CONFIG, which is native
    assert "spike_qwen35_4b_img_only.yaml" not in text
    assert "keyed_" not in text.replace("KEYED", "")  # no run id with the 1260 'keyed' infix


def test_no_secret_is_printed() -> None:
    joined = "\n".join(_code_cells())
    for name in ("GH_TOKEN", "WANDB_API_KEY", "HF_TOKEN", "_basic"):
        for line in joined.splitlines():
            if "print(" in line and name in line:
                assert f"{{{name}" not in line and f", {name}" not in line, line


# ---------------------------------------------------------------------- behaviour


class Env:
    """A tmp Drive-like folder plus the namespace the cells expect (REPO = this checkout)."""

    def __init__(self, tmp_path: Path, *, real_cli: bool, **params: Any) -> None:
        self.tmp, self.calls = tmp_path, []
        self.ns = _params(**params)
        runs = tmp_path / "runs"
        runs.mkdir(exist_ok=True)
        in_merge = self.ns["MODE"] == "merge"
        self.ns.update(
            json=json,
            REPO=ROOT,
            RUNS_DIR=runs,
            RUN_DIR=runs / (self.ns["RUN_ID"] if in_merge else self.ns["SHARD_RUN_ID"]),
            PY=sys.executable,
            ENV={**os.environ, "SHIPDOC_RUNS_DIR": str(runs)},
            DATA_DIR=ROOT / "data",
            ASSIGNMENT_DIR=ROOT / "assignment",
            TRAIN_LABELS=ROOT / "data" / "train" / "labels",
            DEV_LABELS=ROOT / "data" / "dev" / "labels",
            run_stream=self.mock_cli if real_cli else self.canned,
        )
        self.results: list[tuple[int, list[str]]] = []

    def exec(self, i: int) -> None:
        exec(_src(i), self.ns)

    def mock_cli(self, cmd: list[str], tail: int = 40, env: dict | None = None) -> Any:
        """Run the REAL `shipdoc spike` / `bench` on the mock backend."""
        self.calls.append(cmd)
        takes_backend = cmd[3] in ("spike", "bench")
        res = subprocess.run(
            [*cmd, *(["--backend", "mock"] if takes_backend else [])], cwd=ROOT,
            env=env or self.ns["ENV"], capture_output=True, text=True, check=False,
        )  # fmt: skip
        return res.returncode, (res.stdout + res.stderr).splitlines()[-tail:]

    def canned(self, cmd: list[str], tail: int = 40, env: dict | None = None) -> Any:
        self.calls.append(cmd)
        return self.results.pop(0)


@needs_data
def test_estimate_cell_prints_the_native_estimate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env = Env(tmp_path, real_cli=False)
    ids = json.loads((ROOT / "splits" / "zeroshot500.json").read_text())
    env.ns.update(SHARD_EXPECTED_PAGES=671, doc_ids=ids)
    env.exec(ESTIMATE)
    out = capsys.readouterr().out
    assert "ESTIMATE (UNVERIFIED on a GPU)" in out and "zero-shot qwen35_4b_img_only_native" in out
    assert "671 pages, smoke 6 pages, batch from the bench" in out
    assert "CU@1.19" in out and "CU@1.58" in out and "UNMEASURED at native" in out
    assert "low" in out and "high" in out and "THIS TAB: shard 0/1 (671 pages)" in out
    assert env.calls == []  # the estimate spends nothing


def test_smoke_error_blocks_the_bench_and_the_run(tmp_path: Path) -> None:
    env = Env(tmp_path, real_cli=False)
    env.ns.update(smoke_ids=["a"], DEV_LABELS=tmp_path, yaml=yaml)
    env.ns["run_stream"] = lambda cmd, tail=40, env=None: (1, ["CUDA out of memory"])
    with pytest.raises(RuntimeError, match="SMOKE GATE FAILED"):
        env.exec(SMOKE)
    for i in (BENCH, RUN):
        with pytest.raises(RuntimeError, match="not (been )?passed|has not passed"):
            env.exec(i)


def test_run_cell_builds_the_native_command_and_logs_oom(tmp_path: Path) -> None:
    env = Env(tmp_path, real_cli=False)
    bench_file = env.ns["RUNS_DIR"] / "bench_x" / "bench_result.json"
    env.ns.update(SMOKE_PASSED=True, BATCH=4, BENCH_RESULT=bench_file)
    env.results = [(1, ["torch.OutOfMemoryError: CUDA out of memory"])]
    with pytest.raises(RuntimeError, match="out of memory.*Run all again resumes"):
        env.exec(RUN)
    cmd = env.calls[0]
    assert cmd[1:4] == ["-m", "shipdoc", "spike"] and "--resume" in cmd and "--logprobs" in cmd
    assert cmd[cmd.index("--config") + 1] == f"configs/spike_{NATIVE_CFG}.yaml"
    assert cmd[cmd.index("--run-id") + 1] == env.ns["RUN_ID"]
    assert cmd[cmd.index("--batch-size") + 1] == "4"
    assert cmd[cmd.index("--bench-result") + 1] == str(bench_file)  # stored with the run
    assert cmd[cmd.index("--split") + 1] == "train+dev"


def _write_bench(env: Env, result: dict[str, Any]) -> Any:
    def fake(cmd: list[str], tail: int = 40, env_: dict | None = None) -> Any:
        env.calls.append(cmd)
        out = Path(cmd[cmd.index("--out-dir") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "bench_result.json").write_text(json.dumps(result))
        return 0, ["ok"]

    return lambda cmd, tail=40, env=None: fake(cmd, tail, env)


def test_bench_cell_uses_the_native_config_and_names_failed_sizes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env = Env(tmp_path, real_cli=False)
    env.ns.update(SMOKE_PASSED=True)
    result = {
        "chosen_batch_size": 2,
        "deviation": None,
        "results": [
            {"batch_size": 1, "ok": True},
            {"batch_size": 2, "ok": True},
            {"batch_size": 4, "ok": False, "error": "RuntimeError: CUDA out of memory"},
            {"batch_size": 8, "ok": False, "error": "RuntimeError: CUDA out of memory"},
        ],
    }
    env.ns["run_stream"] = _write_bench(env, result)
    env.exec(BENCH)
    cmd = env.calls[0]
    assert cmd[1:4] == ["-m", "shipdoc", "bench"] and "--reuse" in cmd
    assert cmd[cmd.index("--config") + 1] == f"configs/spike_{NATIVE_CFG}.yaml"
    assert cmd[cmd.index("--docs") + 1] == "splits/bench12.json"
    assert env.ns["BATCH"] == 2
    assert env.ns["BENCH_RESULT"].parent.name == f"bench_{NATIVE_CFG}_abcdef1"
    out = capsys.readouterr().out
    assert "bench sizes that FAILED" in out and "[4, 8]" in out
    assert "BATCH SIZE FOR THE FULL RUN: 2 (bench)" in out and "TEST submission MUST use" in out


def test_bench_failure_stops_before_the_full_run_and_says_how_to_resume(tmp_path: Path) -> None:
    env = Env(tmp_path, real_cli=False)
    env.ns.update(SMOKE_PASSED=True)
    env.ns["run_stream"] = lambda cmd, tail=40, env=None: (1, ["CUDA out of memory"])
    with pytest.raises(RuntimeError, match="BENCH FAILED.*NOT started.*Run all again resumes"):
        env.exec(BENCH)
    assert "BATCH" not in env.ns


def test_manual_batch_size_skips_the_bench(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env = Env(tmp_path, real_cli=False, BATCH_SIZE=8)
    env.ns.update(SMOKE_PASSED=True)
    env.exec(BENCH)
    assert env.calls == [] and env.ns["BATCH"] == 8 and env.ns["BENCH_RESULT"] is None
    assert "MANUAL BATCH SIZE 8" in capsys.readouterr().out


@needs_data
def test_end_to_end_on_the_mock_backend(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """real smoke5 run -> real bench (native config) -> 6-doc run -> banner, on the mock backend."""
    from shipdoc import spike

    env = Env(tmp_path, real_cli=True)
    meta = {s: json.loads((ROOT / "meta" / f"{s}.json").read_text()) for s in ("train", "dev")}
    ids = [m["doc_id"] for s in ("train", "dev") for m in meta[s][:3]]
    ns = env.ns
    docs_file = tmp_path / "docs.json"
    docs_file.write_text(json.dumps(ids), encoding="utf-8")
    pages = sum(len(spike.doc_page_images(d.split("_")[0], d)) for d in ids)
    ns.update(DOCS=str(docs_file), EXPECTED_DOCS=len(ids), EXPECTED_PAGES=pages, doc_ids=ids,
              yaml=yaml)  # fmt: skip
    ns["smoke_ids"] = json.loads((ROOT / "splits" / "smoke5.json").read_text())
    env.exec(SMOKE)
    assert ns["SMOKE_PASSED"] is True, capsys.readouterr().out
    env.exec(BENCH)  # mock backend: no peak VRAM is measured, so the rule picks batch 1
    result = json.loads(ns["BENCH_RESULT"].read_text())
    assert ns["BATCH"] == result["chosen_batch_size"] == 1
    assert result["config"] == NATIVE_CFG
    env.exec(RUN)
    run_dir = ns["RUN_DIR"]
    assert run_dir.name == f"zeroshot500_{NATIVE_CFG}_abcdef1"
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["batch_size"] == 1 and manifest["bench"]["chosen"] == 1
    assert manifest["config"]["name"] == NATIVE_CFG
    assert (manifest.get("model") or {}).get("revision")
    capsys.readouterr()
    ns.update(SMOKE_RUNS=ns["RUNS_DIR"] / "smoke", SMOKE_RUN_ID=f"smoke_{ns['RUN_ID']}",
              smoke_status={}, SESSIONS_PATH=run_dir / "sessions.json")  # fmt: skip
    env.exec(BANNER)
    out = capsys.readouterr().out
    assert f"DONE {run_dir.name} docs=6" in out and "complete=True" in out
    # the stored batch size is what 05n / 04c read
    from shipdoc import oof

    assert oof.resolve_batch_size(run_dir, None)["batch_size"] == 1
