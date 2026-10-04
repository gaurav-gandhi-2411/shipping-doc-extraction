"""notebooks/08_dev_final.ipynb: structure, parameters, refusals, the cells that run locally.

Nothing here talks to Colab, Drive, a GPU, Paddle or the network. Cells are exec'd in a fake
namespace with `run_stream` replaced; the banner runs on a mock-backend run of the synthetic corpus
at the NATIVE config (tests/test_devfinal_native.py). The gates themselves are tested in
tests/test_devfinal_native.py, the unchanged stages in tests/test_devfinal.py.
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
from test_devfinal import FOLDS_F, Dev, _eleven_dev_docs, finished  # noqa: F401
from test_devfinal import RUN as _OLD_RUN  # noqa: F401
from test_devfinal_native import NATIVE, nd  # noqa: F401 - `nd` is a fixture
from test_predict import _clean_sha  # noqa: F401 - autouse fixture, must be visible here

from shipdoc import devfinal, devfinal_native

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "08_dev_final.ipynb"
spec = importlib.util.spec_from_file_location(
    "colab_build_dev_final_native", ROOT / "scripts" / "colab_build_dev_final_native.py"
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
    """Run the parameters cell with a fake pin, ZS_SHA7 and TRAIN_SHA7 set like GG sets them."""
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


# -------------------- structure


def test_committed_notebook_matches_builder() -> None:
    assert NB_PATH.read_text(encoding="utf-8") == builder.render()
    assert len(NB["cells"]) == 14


def test_outputs_cleared_cells_compile_lines_fit_100_columns_and_no_magics() -> None:
    for i, c in enumerate(NB["cells"]):
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None
            ast.parse(src(i))
            for ln in src(i).splitlines():
                assert len(ln) <= 100, (i, ln)
                assert not re.match(r"\s*[%!]", ln), (i, ln)  # no IPython magics / shell escapes


def test_cells_match_notebook_05b_apart_from_the_declared_native_edits() -> None:
    old = json.loads((ROOT / "notebooks" / "archive" / "05b_dev_final.ipynb").read_text(encoding="utf-8"))
    same = [i for i in range(14) if old["cells"][i]["source"] == NB["cells"][i]["source"]]
    differ = sorted(set(range(14)) - set(same))
    # account, secrets, clone, unzip, install are byte-equal; the rest carries a native edit
    assert same == [2, 4, 5, 6, 7]
    assert differ == [0, 1, 3, 8, 9, 10, 11, 12, 13]


def test_pin_placeholder_blocks_the_notebook_until_the_pin_commit_fills_it() -> None:
    pin = builder.PINNED_SHA  # unpinned until GG pins; a real SHA is accepted after the pin commit
    if pin == "FILL_PINNED_SHA":
        with pytest.raises(ValueError, match="PINNED_SHA"):
            exec(src(PARAMS), {})
        assert 'PINNED_SHA = "FILL_PINNED_SHA"' in src(PARAMS)
    else:
        assert re.fullmatch(r"[0-9a-f]{40}", pin)
    assert "FILL" in src(find("assert head == pinned_full"))  # the clone cell refuses it too


def test_params_cell_native_defaults_and_run_ids() -> None:
    ns = params_ns()
    assert ns["CONFIG"] == NATIVE_CFG and ns["SPLIT"] == "dev" and ns["EXPECTED_DOCS"] == 100
    assert ns["BATCH_SIZE"] is None and ns["USE_WANDB"] is False and ns["PRECISION"] == "bf16"
    assert ns["ZS_RUN_DIR"] == "zeroshot500_qwen35_4b_img_only_native_abcdef1"
    assert ns["ADAPTER_DIR"] == "runs/ft_native_final_9999999_bf16/final"  # the 03n run naming
    assert ns["RUN_ID"] == "devfinal_native_0123456" and ns["RUN_NAME"] == "devfinal_native"
    assert not ns["RUN_ID"].startswith("devfinal_0")  # never a 05b folder name
    assert (ns["DEV_DOCS"], ns["ZS500_DOCS"]) == ("splits/dev100.json", "splits/zeroshot500.json")
    assert ns["BENCH_DOCS"] == "splits/bench12.json"
    cfg = yaml.safe_load((ROOT / "configs" / f"spike_{ns['CONFIG']}.yaml").read_text())
    assert cfg["max_pixels"] == 2_196_480 and cfg["name"] == NATIVE_CFG


def test_adapter_dir_and_zs_dir_follow_their_parameters() -> None:
    ns = params_ns(TRAIN_SHA7='"abcdef0"', PRECISION='"fp16"', ZS_SHA7='"1234abc"')
    assert ns["ADAPTER_DIR"] == "runs/ft_native_final_abcdef0_fp16/final"
    assert ns["ZS_RUN_DIR"] == "zeroshot500_qwen35_4b_img_only_native_1234abc"


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"PINNED_SHA": '"FILL_ME"'}, "PINNED_SHA"),
        ({"ZS_SHA7": '"FILL_ZS_SHA7"'}, "ZS_SHA7"),  # the placeholder refuses
        ({"ZS_SHA7": '"ABCDEF1"'}, "ZS_SHA7"),
        ({"ZS_SHA7": '"abc"'}, "ZS_SHA7"),
        ({"TRAIN_SHA7": '"FILL_TRAIN_SHA7"'}, "TRAIN_SHA7"),  # no default: refuses
        ({"TRAIN_SHA7": '"abc"'}, "TRAIN_SHA7"),
        ({"PRECISION": '"fp32"'}, "PRECISION"),
        ({"ADAPTER_DIR": '"FILL_ADAPTER_DIR"'}, "ADAPTER_DIR"),
        ({"ADAPTER_DIR": '""'}, "ADAPTER_DIR"),
        ({"BATCH_SIZE": "0"}, "BATCH_SIZE"),
        ({"BATCH_SIZE": "True"}, "BATCH_SIZE"),
        ({"BATCH_SIZE": '"4"'}, "BATCH_SIZE"),
        ({"USE_WANDB": '"yes"'}, "USE_WANDB"),
    ],
)
def test_params_validation_refuses_bad_values(override: dict[str, Any], message: str) -> None:
    sha = "FILL_PINNED_SHA" if override.get("PINNED_SHA") else FAKE_SHA
    with pytest.raises(ValueError, match=message):
        params_ns(sha=sha, **{k: v for k, v in override.items() if k != "PINNED_SHA"})


def test_cell_order_install_verify_plan_estimate_infer_compare_banner() -> None:
    order = [
        find("drive.mount"),
        find("uv sync"),
        find('"shipdoc.devfinal_native", "verify"'),
        find('"shipdoc.devfinal_native", "plan"'),
        find('"shipdoc.devfinal_native", "estimate"'),
        find('"shipdoc.devfinal_native", "infer"'),
        find('"shipdoc.devfinal_native", "compare"'),
        find("FINAL ADAPTER ON DEV, NATIVE"),
    ]
    assert order == sorted(order) and len(set(order)) == len(order)
    assert find("zipfile.ZipFile") < find('"shipdoc.devfinal_native", "verify"')
    infer = src(find('"shipdoc.devfinal_native", "infer"'))
    steps = ["VERIFY (again", "MERGE (fp16", "GUARD (12", "INFER (the 100"]
    assert [infer.index(s) for s in steps] == sorted(infer.index(s) for s in steps)
    assert '"compare"' not in infer  # scoring comes after the inference cell, never inside it


def test_every_stage_goes_through_the_native_module_never_the_1260_one_directly() -> None:
    text = "\n".join(src(i) for i in CODE)
    assert '"shipdoc.devfinal"' not in text and '"shipdoc.predict_ft"' not in text
    assert text.count('"shipdoc.devfinal_native"') == 5
    assert "finetune_qwen35_4b_native.yaml" in src(find('"shipdoc.devfinal_native", "verify"'))
    assert 'CFG = f"configs/spike_{CONFIG}.yaml"' in src(
        find('"shipdoc.devfinal_native", "verify"')
    )
    # the 02n zero-shot run, not the 1260 one, is the comparison arm
    assert "zeroshot500_qwen35_4b_img_only_keyed" not in text


def test_unzip_reads_the_ocr_cache_but_only_dev_pages_are_checked_and_no_test_file_is_read() -> (
    None
):
    mount = src(find("drive.mount"))
    assert 'REQUIRED_ZIPS = ["data.zip", "assignment.zip", "ocr_cache.zip"]' in mount
    unzip = src(find("zipfile.ZipFile"))
    assert 'rel.startswith("paddleocr/dev/")' in unzip and "SHA256SUMS" in unzip
    text = "\n".join(src(i) for i in CODE)
    assert "test_predictions" not in text and "/images" not in text and '"images"' not in text
    assert "paddleocr/test" not in text and '/ "test"' not in text
    assert not re.search(r"test_\d{4}", json.dumps(NB))


def test_install_uses_the_locked_vlm_and_train_groups_and_no_ocr_stack() -> None:
    inst = src(find("uv sync"))
    assert '"--group", "vlm", "--group", "train"' in inst and "--frozen" in inst
    assert "'peft', 'pycountry'" in inst and "find_spec('paddle') is None" in inst
    text = "\n".join(src(i) for i in CODE)
    assert "--group ocr" not in text and 'shipdoc", "ocr"' not in text


def test_no_secrets_in_the_notebook() -> None:
    text = json.dumps(NB)
    for pattern in (
        r"ghp_[A-Za-z0-9]{20,}",
        r"github_pat_",
        r"hf_[A-Za-z0-9]{20,}",
        r"AKIA[0-9A-Z]{12}",
        r"sk-[A-Za-z0-9]{20,}",
        r"WANDB_API_KEY\s*=\s*[\"'][^\"']",
    ):
        assert not re.search(pattern, text), pattern
    assert "ANTHROPIC_API_KEY" not in text


def test_the_notebook_never_writes_under_the_repo_tree() -> None:
    writes = re.compile(
        r"write_text|write_bytes|\.mkdir\(|copyfile|copytree|shutil\.move|\.replace\("
        r"|json\.dump\(|open\([^)]*[\"']w"
    )
    for i in CODE:
        for ln in src(i).splitlines():
            if writes.search(ln):
                assert "REPO" not in ln, (i, ln)


def _flags(text: str) -> set[str]:
    return set(re.findall(r"\"(--[a-z0-9-]+)\"", text))


def _subparser(parser: Any, name: str) -> Any:
    act = next(a for a in parser._actions if getattr(a, "choices", None) and name in a.choices)
    return act.choices[name]


def test_every_flag_the_notebook_passes_exists() -> None:
    parser = devfinal.build_parser()  # devfinal_native parses with the very same parser
    for stage in ("verify", "plan", "estimate", "infer", "compare"):
        marker = f'"shipdoc.devfinal_native", "{stage}"'
        text = src(find(marker))
        chunk = text[text.index(marker) :].split("]")[0]
        known = set(_subparser(parser, stage)._option_string_actions)
        assert _flags(chunk) <= known, (stage, _flags(chunk) - known)
    assert devfinal_native.devfinal is devfinal


# -------------------- cells that run locally


class Recorder:
    """A fake `run_stream`: records commands, answers by stage name."""

    def __init__(self, answers: dict[str, list[int]] | None = None) -> None:
        self.calls: list[list[str]] = []
        self.answers = {k: list(v) for k, v in (answers or {}).items()}

    def __call__(self, cmd: list[str], tail: int = 40, env: dict | None = None) -> tuple[int, Any]:
        self.calls.append([str(c) for c in cmd])
        key = next((k for k in self.answers if k in cmd), None)
        rc = self.answers[key].pop(0) if key and self.answers[key] else 0
        return rc, ["last line"]


def cell_ns(tmp_path: Path, rec: Recorder, **extra: Any) -> dict[str, Any]:
    adapter = tmp_path / "ad"
    adapter.mkdir(exist_ok=True)
    (adapter / "manifest.json").write_text(json.dumps({"code_sha": FAKE_SHA, "precision": "bf16"}))
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    return {
        "Path": Path,
        "json": json,
        "PY": "py",
        "run_stream": rec,
        "REPO": tmp_path / "repo",
        "ADAPTER": adapter,
        "ZS_RUN": tmp_path / "zs",
        "OUT_DIR": out,
        "OCR_CACHE": tmp_path / "ocr",
        "PINNED_SHA": FAKE_SHA,
        "RUN_ID": "devfinal_native_x",
        "CONFIG": NATIVE_CFG,
        "CFG": f"configs/spike_{NATIVE_CFG}.yaml",
        "BATCH_SIZE": None,
        "USE_WANDB": False,
        "EXPECTED_DOCS": 100,
        "DEV_DOCS": "splits/dev100.json",
        "ZS500_DOCS": "splits/zeroshot500.json",
        "BENCH_DOCS": "splits/bench12.json",
        **extra,
    }


def test_verify_cell_runs_the_native_gate_prints_the_table_and_refuses(tmp_path: Path) -> None:
    rec = Recorder()
    ns = cell_ns(tmp_path, rec)
    out = run_cell(find('"shipdoc.devfinal_native", "verify"'), ns)
    args = devfinal.build_parser().parse_args(rec.calls[0][3:])
    assert rec.calls[0][2] == "shipdoc.devfinal_native"
    assert args.stage == "verify" and args.pin == FAKE_SHA and args.adapter_dir == ns["ADAPTER"]
    assert args.config == Path(ns["CFG"]) and "VERIFIED: final adapter" in out
    ns["PINNED_SHA"] = "f" * 40
    rec.calls.clear()
    out = run_cell(find('"shipdoc.devfinal_native", "verify"'), ns)
    assert "training code 0123456 != pin fffffff" in out
    git = next(c for c in rec.calls if c[0] == "git")
    assert "configs/finetune_qwen35_4b_native.yaml" in git  # the NATIVE training config is diffed
    with pytest.raises(RuntimeError, match="ADAPTER REFUSED"):
        run_cell(
            find('"shipdoc.devfinal_native", "verify"'),
            cell_ns(tmp_path, Recorder({"verify": [1]})),
        )


def test_plan_and_estimate_cells_build_the_documented_commands(tmp_path: Path) -> None:
    rec = Recorder()
    ns = cell_ns(tmp_path, rec)
    run_cell(find('"shipdoc.devfinal_native", "plan"'), ns)
    a = devfinal.build_parser().parse_args(rec.calls[0][3:])
    assert a.stage == "plan" and a.out == ns["OUT_DIR"] / "devfinal_plan.json"
    run_cell(find('"shipdoc.devfinal_native", "estimate"'), ns)
    est = rec.calls[1]
    a = devfinal.build_parser().parse_args(est[3:])
    assert a.stage == "estimate" and a.zs_run_dir == ns["ZS_RUN"] and a.config == Path(ns["CFG"])
    assert "--batch-size" not in est and "--adapter-dir" not in est  # the estimate loads no model
    ns["BATCH_SIZE"] = 4
    run_cell(find('"shipdoc.devfinal_native", "estimate"'), ns)
    assert rec.calls[2][rec.calls[2].index("--batch-size") + 1] == "4"
    with pytest.raises(RuntimeError, match="PLAN REFUSED"):
        run_cell(
            find('"shipdoc.devfinal_native", "plan"'), cell_ns(tmp_path, Recorder({"plan": [1]}))
        )


def test_infer_cell_logs_the_session_announces_a_manual_batch_and_raises_on_failure(
    tmp_path: Path,
) -> None:
    rec = Recorder()
    ns = cell_ns(tmp_path, rec, BATCH_SIZE=4)
    out = run_cell(find('"shipdoc.devfinal_native", "infer"'), ns)
    args = devfinal.build_parser().parse_args(rec.calls[0][3:])
    assert args.stage == "infer" and args.batch_size == 4 and not args.wandb
    assert args.bench_docs == "splits/bench12.json" and args.out_dir == ns["OUT_DIR"]
    assert args.adapter_dir == ns["ADAPTER"] and args.zs_run_dir == ns["ZS_RUN"]
    assert args.config == Path(ns["CFG"]) and "MANUAL BATCH SIZE 4" in out
    sessions = json.loads((ns["OUT_DIR"] / "sessions.json").read_text())
    assert sessions[-1]["exit_code"] == 0 and sessions[-1]["seconds"] is not None
    rec = Recorder({"infer": [1]})
    ns = cell_ns(tmp_path, rec, USE_WANDB=True)
    with pytest.raises(RuntimeError, match="Dev inference stopped"):
        run_cell(find('"shipdoc.devfinal_native", "infer"'), ns)
    assert "--wandb" in rec.calls[0] and "--batch-size" not in rec.calls[0]


def test_compare_cell_passes_the_run_and_the_ocr_cache_and_raises_on_failure(
    tmp_path: Path,
) -> None:
    rec = Recorder()
    ns = cell_ns(tmp_path, rec)
    run_cell(find('"shipdoc.devfinal_native", "compare"'), ns)
    a = devfinal.build_parser().parse_args(rec.calls[0][3:])
    assert a.stage == "compare" and not a.plumbing_check and a.ft_run_dir == ns["OUT_DIR"]
    assert a.ocr_cache == ns["OCR_CACHE"] and a.out_dir == ns["OUT_DIR"]
    with pytest.raises(RuntimeError, match="COMPARE FAILED"):
        run_cell(
            find('"shipdoc.devfinal_native", "compare"'),
            cell_ns(tmp_path, Recorder({"compare": [2]})),
        )


def test_banner_prints_the_native_seen_layout_number_and_the_files(
    nd: Dev,  # noqa: F811
) -> None:
    finished(nd)
    (nd.tmp / "folds.json").write_text(json.dumps(FOLDS_F))
    rc = devfinal_native.main(
        [
            "compare", "--config", str(NATIVE), "--data-root", str(nd.w.data),
            "--folds", str(nd.tmp / "folds.json"), "--dev-docs", str(nd.dev_docs),
            "--zs500-docs", str(nd.zs500), "--zs-run-dir", str(nd.w.runs / "zs"),
            "--ft-run-dir", str(nd.run_dir), "--out-dir", str(nd.run_dir),
            "--ocr-cache", str(nd.ocr), "--n-boot", "50",
        ]
    )  # fmt: skip
    assert rc == 0
    ns = {"OUT_DIR": nd.run_dir, "RUN_ID": "devfinal_native_x", "PINNED_SHA": "a" * 40}
    out = run_cell(find("FINAL ADAPTER ON DEV, NATIVE"), ns)
    assert "SEEN LAYOUTS" in out and "not an unseen-supplier number" in out
    assert "native config qwen35_4b_img_only_native hash e4b84ec2809625d5 max_pixels 2196480" in out
    assert "DONE devfinal_native_x compared=True" in out and "MISSING" not in out
    assert "rules_train_shapes" in out and "raw" in out and "paired delta" in out
    assert "guard ran=True ok=True" in out and "dev_0000" not in out and "Supplier" not in out
    assert "devfinal_native_x" in out  # the local destination names the native run
    (nd.run_dir / "devfinal_compare.json").unlink()
    assert "NOT COMPARED" in run_cell(find("FINAL ADAPTER ON DEV, NATIVE"), ns)
