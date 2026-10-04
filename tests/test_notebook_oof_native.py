"""notebooks/05n_oof_infer_native.ipynb: structure, parameters, the commands its cells build.

Nothing here talks to Colab, Drive, a GPU or the network. Cells are exec'd in a fake namespace with
the heavy helpers replaced. The resolution gate itself is tested in tests/test_nativerun.py; the
unchanged oof stages in tests/test_oof.py.
"""

from __future__ import annotations

import ast
import importlib.util
import io
import json
import re
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "05n_oof_infer_native.ipynb"
spec = importlib.util.spec_from_file_location(
    "colab_build_oof_native", ROOT / "scripts" / "colab_build_oof_native.py"
)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

NB = json.loads(NB_PATH.read_text(encoding="utf-8"))
CODE = [i for i, c in enumerate(NB["cells"]) if c["cell_type"] == "code"]
FAKE_SHA = "0123456789abcdef0123456789abcdef01234567"
PARAMS = 1
NATIVE_CFG = "qwen35_4b_img_only_native"


def src(i: int) -> str:
    return "".join(NB["cells"][i]["source"])


def find(marker: str) -> int:
    """Index of the one code cell containing `marker`."""
    hits = [i for i in CODE if marker in src(i)]
    assert len(hits) == 1, (marker, hits)
    return hits[0]


def params_ns(sha: str = FAKE_SHA, **override: Any) -> dict[str, Any]:
    """Run the parameters cell with a fake pin, ZS_SHA7 and ADAPTER_DIR set like GG sets them."""
    text = src(PARAMS).replace(builder.PINNED_SHA, sha)
    defaults = {"ZS_SHA7": '"abcdef1"', "TRAIN_SHA7": '"9999999"'}
    edits = {k: (v if isinstance(v, str) else repr(v)) for k, v in override.items()}
    for k, v in {**defaults, **edits}.items():
        text, n = re.subn(rf"(?m)^{k} = .*$", f"{k} = {v}", text)
        assert n == 1, k
    ns: dict[str, Any] = {}
    exec(text, ns)
    return ns


def run_cell(i: int, ns: dict[str, Any]) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        exec(compile(src(i), f"cell{i}", "exec"), ns)
    return buf.getvalue()


# ---------------------------------------------------------------------- structure


def test_committed_notebook_matches_builder() -> None:
    assert NB_PATH.read_text(encoding="utf-8") == builder.render()
    assert len(NB["cells"]) == 13


def test_outputs_cleared_cells_compile_and_lines_fit_100_columns() -> None:
    for i, c in enumerate(NB["cells"]):
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None
            ast.parse(src(i))
            for ln in src(i).splitlines():
                assert len(ln) <= 100, (i, ln)


def test_cells_match_notebook_05_apart_from_the_declared_native_edits() -> None:
    old = json.loads((ROOT / "notebooks" / "archive" / "05_oof_infer.ipynb").read_text(encoding="utf-8"))
    same = [i for i in range(13) if old["cells"][i]["source"] == NB["cells"][i]["source"]]
    differ = sorted(set(range(13)) - set(same))
    # title, parameters, mount (messages), verify, estimate, infer differ; the rest is byte-equal
    assert differ == [0, 1, 3, 8, 9, 10]
    assert same == [2, 4, 5, 6, 7, 11, 12]


def test_pin_placeholder_blocks_the_notebook_until_the_pin_commit_fills_it() -> None:
    pin = builder.PINNED_SHA
    if pin == "FILL_PINNED_SHA":
        with pytest.raises(ValueError, match="PINNED_SHA"):
            exec(src(PARAMS), {})
        assert 'PINNED_SHA = "FILL_PINNED_SHA"' in src(PARAMS)
    else:
        assert re.fullmatch(r"[0-9a-f]{40}", pin)
    assert '"FILL" in PINNED_SHA' in src(find("def git("))


def test_params_cell_native_defaults_and_run_ids() -> None:
    ns = params_ns()
    assert ns["CONFIG"] == NATIVE_CFG and ns["SPLIT"] == "train+dev"
    assert ns["FOLD"] == 0 and ns["BATCH_SIZE"] is None and ns["USE_WANDB"] is False
    assert ns["ZS_RUN_DIR"] == "zeroshot500_qwen35_4b_img_only_native_abcdef1"
    assert ns["RUN_ID"] == "oof_native_fold0_0123456" and ns["RUN_NAME"] == "oof_native_fold0"
    assert not ns["RUN_ID"].startswith("oof_fold")  # never a 05 (1260) folder name
    assert ns["BENCH_DOCS"] == "splits/bench12.json"
    assert ns["ADAPTER_DIR"] == "runs/ft_native_fold0_9999999_bf16/final"  # the 03n run naming
    cfg = yaml.safe_load((ROOT / "configs" / f"spike_{ns['CONFIG']}.yaml").read_text())
    assert cfg["max_pixels"] == 2_196_480 and cfg["name"] == NATIVE_CFG


def test_zs_run_dir_follows_zs_sha7_and_fold_follows_the_fold() -> None:
    ns = params_ns(ZS_SHA7='"1234abc"', FOLD=2)
    assert ns["ZS_RUN_DIR"] == "zeroshot500_qwen35_4b_img_only_native_1234abc"
    assert ns["RUN_ID"] == "oof_native_fold2_0123456"
    assert ns["ADAPTER_DIR"] == "runs/ft_native_fold2_9999999_bf16/final"
    ns = params_ns(TRAIN_SHA7='"abcdef0"', PRECISION='"fp16"')
    assert ns["ADAPTER_DIR"] == "runs/ft_native_fold0_abcdef0_fp16/final"


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"PINNED_SHA": '"FILL_ME"'}, "PINNED_SHA"),
        ({"FOLD": 3}, "FOLD"),
        ({"FOLD": True}, "FOLD"),
        ({"ZS_SHA7": '"FILL_ZS_SHA7"'}, "ZS_SHA7"),  # the placeholder refuses
        ({"ZS_SHA7": '"ABCDEF1"'}, "ZS_SHA7"),
        ({"ZS_SHA7": '"abc"'}, "ZS_SHA7"),
        ({"TRAIN_SHA7": '"FILL_TRAIN_SHA7"'}, "TRAIN_SHA7"),  # no default: refuses
        ({"TRAIN_SHA7": '"abc"'}, "TRAIN_SHA7"),
        ({"PRECISION": '"fp32"'}, "PRECISION"),
        ({"ADAPTER_DIR": '"FILL_ADAPTER_DIR"'}, "ADAPTER_DIR"),
        ({"ADAPTER_DIR": '""'}, "ADAPTER_DIR"),
        ({"BATCH_SIZE": 0}, "BATCH_SIZE"),
        ({"BATCH_SIZE": True}, "BATCH_SIZE"),
        ({"BATCH_SIZE": '"4"'}, "BATCH_SIZE"),
        ({"USE_WANDB": '"yes"'}, "USE_WANDB"),
    ],
)
def test_params_validation_refuses_bad_values(override: dict[str, Any], message: str) -> None:
    sha = override.pop("PINNED_SHA", None)
    with pytest.raises(ValueError, match=message):
        if sha is not None:
            params_ns(sha.strip('"'))
        else:
            params_ns(**override)


def test_unedited_placeholders_refuse_even_with_a_valid_pin() -> None:
    text = src(PARAMS).replace(builder.PINNED_SHA, FAKE_SHA)
    ns = {}
    exec(text, ns)  # public build: ZS_SHA7 and TRAIN_SHA7 default to the pin's sha7
    assert ns["ZS_SHA7"] == ns["TRAIN_SHA7"] == FAKE_SHA[:7]


def test_the_missing_zero_shot_folder_is_refused_with_the_exact_value_to_set(
    tmp_path: Path,
) -> None:
    """The mount cell's tail (the part after drive.mount, which needs Colab)."""
    text = src(find("def _under_runs"))
    tail = text[text.index("def _under_runs") :]
    runs = tmp_path / "runs"
    adapter = runs / "ft_native_fold0" / "final"
    adapter.mkdir(parents=True)
    (adapter / "manifest.json").write_text("{}")
    ns = {**params_ns(), "Path": Path, "DRIVE_DIR": tmp_path, "RUNS_DIR": runs}
    ns["ADAPTER_DIR"] = "runs/ft_native_fold0/final"  # as if set by hand
    with pytest.raises(FileNotFoundError, match="zero-shot run .*predictions.json.*ZS_SHA7"):
        run_cell_text(tail, ns)
    zs = runs / ns["ZS_RUN_DIR"]
    zs.mkdir()
    (zs / "predictions.json").write_text("{}")
    out = run_cell_text(tail, ns)
    assert "zero-shot run   :" in out and "(02n, native)" in out and ns["ZS_RUN_DIR"] in out
    (adapter / "manifest.json").unlink()
    with pytest.raises(FileNotFoundError, match="adapter folder .*manifest.json"):
        run_cell_text(tail, ns)


def run_cell_text(text: str, ns: dict[str, Any]) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        exec(compile(text, "tail", "exec"), ns)
    return buf.getvalue()


def test_cell_order_install_verify_estimate_infer_compare_banner() -> None:
    order = [
        find("uv sync"),
        find('"shipdoc.nativerun", "verify"'),
        find('"estimate-oof"'),
        find('"shipdoc.nativerun", "infer"'),
        find('"oof", "compare"'),
        find("OOF FOLD"),
    ]
    assert order == sorted(order) and len(set(order)) == len(order)
    infer = src(find('"shipdoc.nativerun", "infer"'))
    steps = ["VERIFY (again", "MERGE (fp16", "GUARD (12", "INFER (the fold"]
    assert [infer.index(s) for s in steps] == sorted(infer.index(s) for s in steps)
    assert '"compare"' not in infer


def test_install_is_the_05_install_and_the_notebook_reads_no_test_files() -> None:
    inst = src(find("uv sync"))
    assert '"--group", "vlm", "--group", "train"' in inst and "'peft'" in inst
    text = "\n".join(src(i) for i in CODE)
    assert "test_predictions" not in text and '/ "test"' not in text and "ocr_cache.zip" not in text
    assert "spike_qwen35_4b_img_only.yaml" not in text  # the 1260 config is never named


def test_no_secrets_in_the_notebook() -> None:
    text = json.dumps(NB)
    for pattern in (r"ghp_[A-Za-z0-9]{20,}", r"github_pat_", r"hf_[A-Za-z0-9]{20,}",
                    r"sk-[A-Za-z0-9]{20,}", r"WANDB_API_KEY\s*=\s*[\"'][^\"']"):  # fmt: skip
        assert not re.search(pattern, text), pattern
    assert "ANTHROPIC_API_KEY" not in text


# ---------------------------------------------------------------------- commands


def _ns(tmp_path: Path, calls: list[list[str]], rc: dict[str, int]) -> dict[str, Any]:
    adapter = tmp_path / "final"
    adapter.mkdir(exist_ok=True)
    (adapter / "manifest.json").write_text(json.dumps({"code_sha": FAKE_SHA, "precision": "bf16"}))

    def run_stream(cmd: list[str], *a: Any, **k: Any) -> tuple[int, list[str]]:
        calls.append(cmd)
        return rc["value"], ["last line"]

    return {
        "PY": "py", "FOLD": 1, "CONFIG": NATIVE_CFG, "ZS_RUN": tmp_path / "zs",
        "OUT_DIR": tmp_path / "out", "ADAPTER": adapter, "PINNED_SHA": "f" * 40, "json": json,
        "run_stream": run_stream, "BATCH_SIZE": None, "USE_WANDB": False,
        "RUN_ID": "oof_native_fold1_x", "HELD": ["a", "b"], "BENCH_DOCS": "splits/bench12.json",
        "Path": Path,
    }  # fmt: skip


def test_cells_build_the_native_commands(tmp_path: Path) -> None:
    calls: list[list[str]] = []
    rc = {"value": 0}
    ns = _ns(tmp_path, calls, rc)
    cfg = f"configs/spike_{NATIVE_CFG}.yaml"
    out = run_cell(find('"shipdoc.nativerun", "verify"'), ns)
    v = calls[0]
    assert v[:4] == ["py", "-m", "shipdoc.nativerun", "verify"]
    assert v[v.index("--config") + 1] == cfg and v[v.index("--pin") + 1] == "f" * 40
    assert "--adapter-dir" in v and "--zs-run-dir" in v
    assert "training code 0123456 != pin fffffff" in out and "VERIFIED: fold 1 adapter" in out
    rc["value"] = 1
    with pytest.raises(RuntimeError, match="ADAPTER REFUSED"):
        run_cell(find('"shipdoc.nativerun", "verify"'), ns)
    rc["value"] = 0
    calls.clear()
    run_cell(find('"estimate-oof"'), ns)
    e = calls[0]
    assert e[:4] == ["py", "-m", "shipdoc.nativerun", "estimate-oof"] and "--batch-size" not in e
    assert "--adapter-dir" not in e and e[e.index("--zs-run-dir") + 1] == str(tmp_path / "zs")
    ns["BATCH_SIZE"] = 4
    calls.clear()
    run_cell(find('"estimate-oof"'), ns)
    assert calls[0][calls[0].index("--batch-size") + 1] == "4"
    calls.clear()
    out = run_cell(find('"shipdoc.nativerun", "infer"'), ns)
    i = calls[0]
    assert i[:4] == ["py", "-m", "shipdoc.nativerun", "infer"] and i[i.index("--config") + 1] == cfg
    assert i[i.index("--bench-docs") + 1] == "splits/bench12.json"
    assert i[i.index("--batch-size") + 1] == "4" and "MANUAL BATCH SIZE 4" in out
    rc["value"] = 1
    ns["BATCH_SIZE"] = None
    with pytest.raises(RuntimeError, match="OOF inference stopped"):
        run_cell(find('"shipdoc.nativerun", "infer"'), ns)
    rc["value"] = 0
    calls.clear()
    run_cell(find('"oof", "compare"'), ns)
    c = calls[0]
    assert c[:5] == ["py", "-m", "shipdoc", "oof", "compare"] and c[c.index("--config") + 1] == cfg
    assert "--adapter-dir" not in c  # the raw OOF compare, unchanged
