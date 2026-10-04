"""notebooks/04c_predict_test_native.ipynb: structure, parameters, refusals, local cells.

Nothing here talks to Colab, Drive, a GPU, Paddle or the network. Cells are exec'd in a fake
namespace with `run_stream` replaced. The refusals themselves and both pipelines (mock backend over
the synthetic corpus) are tested in tests/test_predict_native.py; the unchanged stages in
tests/test_predict.py, test_reuse_v0.py, test_notebook_predict_v2.py and test_flags.py.
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

from shipdoc import cli, flags, ocr_stage, predict_ft, predict_native

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "04c_predict_test_native.ipynb"
spec = importlib.util.spec_from_file_location(
    "colab_build_predict_native", ROOT / "scripts" / "colab_build_predict_native.py"
)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

NB = json.loads(NB_PATH.read_text(encoding="utf-8"))
CODE = [i for i, c in enumerate(NB["cells"]) if c["cell_type"] == "code"]
FAKE_SHA = "0123456789abcdef0123456789abcdef01234567"
S7 = FAKE_SHA[:7]
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
    """Run the parameters cell with a fake pin and the placeholders set like GG sets them."""
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
    assert len(NB["cells"]) == 24


def test_outputs_cleared_cells_compile_lines_fit_100_columns_and_no_magics() -> None:
    for i, c in enumerate(NB["cells"]):
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None
            ast.parse(src(i))
            for ln in src(i).splitlines():
                assert len(ln) <= 100, (i, ln)
                assert not re.match(r"\s*[%!]", ln), (i, ln)  # no IPython magics / shell escapes


def test_the_shared_cells_are_byte_equal_to_04c() -> None:
    old = json.loads((ROOT / "notebooks" / "archive" / "04c_predict_test_v2.ipynb").read_text("utf-8"))
    # account, secrets, clone, unzip, OCR estimate, install, plan, adopt-ocr, OCR stage
    pairs = [(2, 2), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (11, 10), (12, 11), (13, 12)]
    for mine, theirs in pairs:
        assert NB["cells"][mine]["source"] == old["cells"][theirs]["source"], (mine, theirs)


def test_pin_placeholder_blocks_the_notebook_until_the_pin_commit_fills_it() -> None:
    pin = builder.PINNED_SHA  # unpinned until GG pins; a real SHA is accepted after the pin commit
    if pin == "FILL_PINNED_SHA":
        with pytest.raises(ValueError, match="PINNED_SHA"):
            exec(src(PARAMS), {})
        assert 'PINNED_SHA = "FILL_PINNED_SHA"' in src(PARAMS)
    else:
        assert re.fullmatch(r"[0-9a-f]{40}", pin)
    assert "FILL" in src(find("assert head == pinned_full"))  # the clone cell refuses it too


def test_params_cell_native_defaults_and_ids() -> None:
    ns = params_ns()
    assert ns["MODEL"] == "zs" and ns["CONFIG"] == NATIVE_CFG and ns["SPLIT"] == "test"
    assert (ns["EXPECTED_DOCS"], ns["EXPECTED_PAGES"], ns["SHARD"]) == (200, 280, "0/1")
    assert ns["BATCH_SIZE"] is None and ns["CALIBRATOR_FILE"] is None  # the flags refuse by default
    assert ns["ZS_RUN_DIR"] == "zeroshot500_qwen35_4b_img_only_native_abcdef1"
    assert ns["ADAPTER_DIR"] == "runs/ft_native_final_9999999_bf16/final"  # the 03n run naming
    assert ns["ZS_TEST_DIR"] == f"submissions/v15_{S7}"  # the zs run of the same pin
    assert ns["RUN_ID"] == f"testnative_zs_{S7}" and ns["SUB_NAME"] == f"v15_{S7}"
    assert ns["CFG"] == f"configs/spike_{NATIVE_CFG}.yaml" and ns["SUB_PREFIX"] == "v15"
    assert ns["REUSE_V0_DIR"] is None and ns["REUSE_ACK_SPIKE_DIFF"] is False
    assert ns["REUSE_OCR_FROM"] is None
    assert (ns["FIELD_TARGET"], ns["DOC_TARGET"]) == (0.98, 0.98)
    assert not ns["RUN_ID"].startswith(("test_", "testft_"))  # never a 04 / 04c run folder name
    ft = params_ns(MODEL='"ft"')
    assert ft["SUB_NAME"] == f"v2n_{S7}" and ft["RUN_ID"] == f"testnative_ft_{S7}"
    cfg = yaml.safe_load((ROOT / "configs" / f"spike_{ns['CONFIG']}.yaml").read_text())
    assert cfg["max_pixels"] == 2_196_480 and cfg["name"] == NATIVE_CFG


def test_zs_dir_adapter_dir_and_zs_test_dir_follow_their_parameters() -> None:
    ns = params_ns(ZS_SHA7='"1234abc"', MODEL='"ft"', TRAIN_SHA7='"abcdef0"', PRECISION='"fp16"')
    assert ns["ZS_RUN_DIR"] == "zeroshot500_qwen35_4b_img_only_native_1234abc"
    assert ns["ADAPTER_DIR"] == "runs/ft_native_final_abcdef0_fp16/final"


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"MODEL": '"both"'}, "MODEL"),
        ({"ZS_SHA7": '"FILL_ZS_SHA7"'}, "ZS_SHA7"),  # the placeholder refuses, for both models
        ({"ZS_SHA7": '"ABCDEF1"'}, "ZS_SHA7"),
        ({"ZS_SHA7": '"abc"'}, "ZS_SHA7"),
        ({"BATCH_SIZE": "0"}, "BATCH_SIZE"),
        ({"BATCH_SIZE": "True"}, "BATCH_SIZE"),
        ({"BATCH_SIZE": '"4"'}, "BATCH_SIZE"),
        ({"PRECISION": '"fp32"'}, "PRECISION"),
        ({"REUSE_V0_DIR": "7"}, "REUSE_V0_DIR"),
        ({"REUSE_ACK_SPIKE_DIFF": '"yes"'}, "REUSE_ACK_SPIKE_DIFF"),
        ({"CALIBRATOR_FILE": "7"}, "CALIBRATOR_FILE"),
        ({"CALIBRATOR_FILE": '""'}, "CALIBRATOR_FILE"),
        ({"FIELD_TARGET": "0.9"}, "FIELD_TARGET"),
        ({"DOC_TARGET": "0.99"}, "DOC_TARGET"),
        ({"REUSE_OCR_FROM": "7"}, "REUSE_OCR_FROM"),
        ({"MODEL": '"ft"', "TRAIN_SHA7": '"FILL_TRAIN_SHA7"'}, "TRAIN_SHA7"),  # ft: no default
        ({"MODEL": '"ft"', "TRAIN_SHA7": '"abc"'}, "TRAIN_SHA7"),
        ({"MODEL": '"ft"', "ADAPTER_DIR": '"FILL_ADAPTER_DIR"'}, "ADAPTER_DIR"),
        ({"MODEL": '"ft"', "ADAPTER_DIR": '""'}, "ADAPTER_DIR"),
        ({"MODEL": '"ft"', "ZS_TEST_DIR": '"submissions/v15_FILL"'}, "ZS_TEST_DIR"),
        ({"MODEL": '"ft"', "ZS_TEST_DIR": '""'}, "ZS_TEST_DIR"),
    ],
)
def test_params_validation_refuses_bad_values(override: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        params_ns(**override)


def test_the_ft_inputs_are_not_required_by_the_zs_path_and_the_placeholder_refuses_the_pin() -> (
    None
):
    assert params_ns(TRAIN_SHA7='"FILL_TRAIN_SHA7"')["MODEL"] == "zs"  # zs never reads them
    with pytest.raises(ValueError, match="PINNED_SHA"):
        params_ns(sha="FILL_PINNED_SHA")


# -------------------- the mount cell: every required input is checked


def mount_tail(tmp: Path, **override: Any) -> dict[str, Any]:
    """Run the part of the mount cell after the base mount (paths, required-input checks)."""
    text = src(find("drive.mount"))
    tail = text[text.index("SUBMISSIONS_DIR = ") :]
    ns = params_ns(**override)
    ns.update(Path=Path, DRIVE_DIR=tmp / "drive", RUNS_DIR=tmp / "drive" / "runs")
    exec(compile(tail, "mount-tail", "exec"), ns)
    return ns


def make(tmp: Path, rel: str, *files: str) -> None:
    d = tmp / "drive" / rel
    d.mkdir(parents=True, exist_ok=True)
    for f in files:
        (d / f).write_text("{}")


def test_zs_needs_the_02n_run_and_says_which_parameter_to_set(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"zero-shot run \(02n\).*ZS_SHA7"):
        mount_tail(tmp_path)
    make(tmp_path, "runs/zeroshot500_qwen35_4b_img_only_native_abcdef1", "manifest.json")
    with pytest.raises(FileNotFoundError, match="predictions.json"):
        mount_tail(tmp_path)
    make(tmp_path, "runs/zeroshot500_qwen35_4b_img_only_native_abcdef1", "predictions.json")
    ns = mount_tail(tmp_path)
    assert ns["ADAPTER"] is None and ns["ZS_TEST"] is None  # the zs path reads no adapter
    assert ns["SUBMISSION_DIR"] == tmp_path / "drive" / "submissions" / f"v15_{S7}"
    assert ns["DEV_RUN"] == ns["ZS_RUN"] and ns["V0_DIR"] is None
    assert ns["OCR_ROOT"] == tmp_path / "drive" / "ocr_cache_test"  # the cache 04b filled
    assert ns["RUN_DIR"] == tmp_path / "drive" / "runs" / f"testnative_zs_{S7}"
    assert ns["SUBMISSION_DIR"].is_dir() and ns["META_DIR"].is_dir()


def test_ft_needs_the_adapter_and_the_zs_test_folder(tmp_path: Path) -> None:
    zs = "runs/zeroshot500_qwen35_4b_img_only_native_abcdef1"
    make(tmp_path, zs, "manifest.json", "predictions.json")
    with pytest.raises(FileNotFoundError, match=r"final adapter folder.*TRAIN_SHA7.*ZS_TEST_DIR"):
        mount_tail(tmp_path, MODEL='"ft"')
    make(tmp_path, "runs/ft_native_final_9999999_bf16/final", "manifest.json")
    with pytest.raises(
        FileNotFoundError, match=r"zs test submission \(ZS_TEST_DIR\).*MODEL = 'zs'"
    ):
        mount_tail(tmp_path, MODEL='"ft"')
    make(tmp_path, f"submissions/v15_{S7}", "manifest.json")
    ns = mount_tail(tmp_path, MODEL='"ft"')
    assert ns["ADAPTER"] == tmp_path / "drive/runs/ft_native_final_9999999_bf16/final"
    assert ns["ZS_TEST"] == tmp_path / "drive" / "submissions" / f"v15_{S7}"
    assert ns["SUBMISSION_DIR"].name == f"v2n_{S7}" and ns["V0_DIR"] is None


# -------------------- cell order and the path guards


def test_cell_order() -> None:
    order = [
        find("drive.mount"),
        find("ESTIMATE (UNVERIFIED) OCR stage"),
        find('"uv", "sync", "--frozen"'),
        find('"shipdoc.predict_native", "check"'),
        find('"shipdoc.predict_native", "estimate"'),
        find('"predict", "plan"'),
        find('"adopt-ocr"'),
        find("def ocr_device"),
        find('"predict", "smoke"'),
        find('"predict", "batch"'),
        find('"predict", "run"'),
        find('"predict", "determinism"'),
        find('"shipdoc.predict_native", "verify"'),
        find('"shipdoc.predict_native", "infer"'),
        find('"shipdoc.predict_native", "determinism"'),
        find('"predict", "assemble"'),
        find('"shipdoc.predict_native", "flags"'),
        find("TEST PREDICTIONS"),
    ]
    assert order == sorted(order) and len(set(order)) == len(order)
    assert find('"shipdoc.predict_native", "finalize"') == find('"predict", "assemble"')
    assert find('"check-flags"') == find('"shipdoc.predict_native", "flags"')
    # the native checks run before the OCR stage and before any model stage
    assert find('"shipdoc.predict_native", "check"') < find("def ocr_device")
    assert find('"shipdoc.predict_native", "check"') < find('"predict", "smoke"')
    assert find('"shipdoc.predict_native", "check"') < find('"shipdoc.predict_native", "verify"')
    assert find('"shipdoc.predict_native", "verify"') < find('"shipdoc.predict_native", "infer"')
    assert find('"shipdoc.predict_native", "estimate"') < find('"predict", "smoke"')


ZS_ONLY = (
    '"predict", "smoke"',
    '"predict", "batch"',
    '"predict", "run"',
    '"predict", "determinism"',
)
FT_ONLY = (
    '"shipdoc.predict_native", "verify"',
    '"shipdoc.predict_native", "infer"',
    '"shipdoc.predict_native", "determinism"',
)


def test_every_path_specific_cell_is_guarded_by_the_model_switch() -> None:
    for marker in ZS_ONLY:
        assert src(find(marker)).startswith('if MODEL == "zs":\n'), marker
    for marker in FT_ONLY:
        assert src(find(marker)).startswith('if MODEL == "ft":\n'), marker
    text = "\n".join(src(i) for i in CODE)
    # no zero-shot inference on the ft path: the zs stages appear only in the guarded cells
    for marker in ZS_ONLY:
        assert text.count(marker) == 1, marker
    assert '"shipdoc.predict_ft"' in text and text.count('"shipdoc.predict_ft"') == 1  # adopt-ocr
    assert '"shipdoc", "spike"' not in text and '"oof", "compare"' not in text
    for i in CODE:  # nothing but the 04 stages and the native stages is ever started
        for mod in re.findall(r'"-m", "(shipdoc[a-z_.]*)"', src(i)):
            assert mod in ("shipdoc", "shipdoc.predict_native", "shipdoc.predict_ft",
                           "shipdoc.ocr_stage"), mod  # fmt: skip


def test_asserts_and_guards_are_present() -> None:
    text = "\n".join(src(i) for i in CODE)
    assert "assert head == pinned_full" in text
    assert 'assert not (DATA_DIR / SPLIT / "labels").exists()' in text
    assert "assert len(test_pages_by_doc) == EXPECTED_DOCS" in text
    assert 'assert after["valid"] == EXPECTED_PAGES' in text  # 280/280 OCR pages
    assert text.count("find_spec('paddle') is None") >= 2  # install, after the OCR stage
    assert 'OCR_VENV.resolve() != (REPO / ".venv").resolve()' in text
    assert '--group", "vlm", "--group", "train"' in text and "'peft'" in text
    assert "--require-stack" in text and "--expect-code-sha" in text


def test_the_notebook_never_writes_under_the_repo_tree_and_has_no_secrets_or_test_ids() -> None:
    writes = re.compile(
        r"write_text|write_bytes|\.mkdir\(|copyfile|copytree|shutil\.move|\.replace\("
        r"|json\.dump\(|open\([^)]*[\"']w"
    )
    for i in CODE:
        for ln in src(i).splitlines():
            if writes.search(ln):
                assert "REPO" not in ln, (i, ln)
    env = src(find('"uv", "sync", "--frozen"'))
    for key in ("SHIPDOC_RUNS_DIR", "SHIPDOC_SUBMISSIONS_DIR", "SHIPDOC_OCR_CACHE", "HF_HOME"):
        line = next(ln for ln in env.splitlines() if f'"{key}"' in ln)
        assert "REPO" not in line, line
    text = json.dumps(NB)
    for pattern in (r"ghp_[A-Za-z0-9]{20,}", r"github_pat_", r"hf_[A-Za-z0-9]{20,}",
                    r"sk-[A-Za-z0-9]{20,}", r"AKIA[0-9A-Z]{12}"):  # fmt: skip
        assert not re.search(pattern, text), pattern
    assert "ANTHROPIC_API_KEY" not in text and "wandb" not in text.lower()
    assert not re.search(r"test_\d{4}", text)  # no test doc id


def _flags(text: str) -> set[str]:
    return set(re.findall(r"\"(--[a-z0-9-]+)\"", text))


def _subparser(parser: Any, *names: str) -> Any:
    cur = parser
    for n in names:
        act = next(a for a in cur._actions if getattr(a, "choices", None) and n in a.choices)
        cur = act.choices[n]
    return cur


def _chunk(marker: str) -> str:
    text = src(find(marker))
    return text[text.index(marker) :].split("]")[0]


def test_every_flag_the_notebook_passes_exists() -> None:
    pn = predict_native.build_parser()
    for stage in ("check", "estimate", "finalize", "flags", "check-flags"):
        marker = f'"shipdoc.predict_native", "{stage}"'
        known = set(_subparser(pn, stage)._option_string_actions)
        assert _flags(_chunk(marker)) <= known, (stage, _flags(_chunk(marker)) - known)
    # ft stages are passed through to predict_ft: its parser must know their flags
    pf = predict_ft.build_parser()
    for stage in ("verify", "infer", "determinism"):
        marker = f'"shipdoc.predict_native", "{stage}"'
        known = set(_subparser(pf, stage)._option_string_actions)
        assert _flags(_chunk(marker)) <= known, (stage, _flags(_chunk(marker)) - known)
    cp = cli.build_parser()
    for stage in ("smoke", "batch", "run", "determinism", "assemble", "plan"):
        marker = f'"predict", "{stage}"'
        known = set(_subparser(cp, "predict", stage)._option_string_actions)
        assert _flags(_chunk(marker)) <= known | {"--out-dir"}, (stage, _flags(_chunk(marker)))
    assert set(flags.build_parser()._option_string_actions)  # (the flags stage is predict_native)
    op = ocr_stage.build_parser()
    ocr_cell = src(find("def ocr_device"))
    for stage in ("check", "finalize", "recheck", "device"):
        assert set(_subparser(op, stage)._option_string_actions) & _flags(ocr_cell)


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

    def has(self, *words: str) -> list[list[str]]:
        return [c for c in self.calls if all(w in c for w in words)]


def cell_ns(tmp_path: Path, rec: Recorder, model: str = "zs", **extra: Any) -> dict[str, Any]:
    meta, sub, run = tmp_path / "meta", tmp_path / "sub", tmp_path / "run"
    for d in (meta, sub, run):
        d.mkdir(exist_ok=True)
    ns = {
        "Path": Path, "json": json, "PY": "py", "run_stream": rec, "MODEL": model,
        "REPO": tmp_path / "repo", "DRIVE_DIR": tmp_path / "drive", "RUNS_DIR": tmp_path / "runs",
        "META_DIR": meta, "SUBMISSION_DIR": sub, "RUN_DIR": run,
        "ZS_RUN": tmp_path / "zs02n", "DEV_RUN": tmp_path / "zs02n",
        "ADAPTER": tmp_path / "ad", "ZS_TEST": tmp_path / "v15", "V0_DIR": None,
        "OCR_ROOT": tmp_path / "ocr", "OCR_FROM": None, "REUSE_OCR_FROM": None,
        "PINNED_SHA": FAKE_SHA, "SHA7": S7, "RUN_ID": "run", "SHARD_RUN_ID": "run",
        "SUB_NAME": f"v15_{S7}" if model == "zs" else f"v2n_{S7}",
        "SUB_PREFIX": "v15" if model == "zs" else "v2n",
        "CONFIG": NATIVE_CFG, "CFG": f"configs/spike_{NATIVE_CFG}.yaml",
        "EXPECTED_DOCS": 200, "EXPECTED_PAGES": 280, "SHARD": "0/1", "MODE": "run",
        "BATCH_SIZE": None, "BATCH": 4, "FLAGS_ZS_BATCH": 4,
        "SMOKE_DOCS": "splits/smoke5.json", "BENCH_DOCS": "splits/bench12.json",
        "DECISION_PATH": meta / "d.json", "SMOKE_STATUS_PATH": meta / "s.json",
        "CHECK_PATH": meta / "native_check.json",
        "SCHEMA_PATH": tmp_path / "schema.json", "SAMPLE_PATH": tmp_path / "nosample.json",
        "REUSE_ACK_SPIKE_DIFF": False, "FIELD_TARGET": 0.98, "DOC_TARGET": 0.98,
        "CALIBRATOR_FILE": None, "SPLIT": "test",
        **extra,
    }  # fmt: skip
    return ns


def test_check_cell_builds_the_command_reads_the_batch_and_stops_on_a_refusal(
    tmp_path: Path,
) -> None:
    marker = '"shipdoc.predict_native", "check"'

    class Writer(Recorder):
        def __call__(self, cmd: list[str], tail: int = 40, env: dict | None = None) -> Any:
            rc, t = super().__call__(cmd, tail, env)
            Path(cmd[cmd.index("--out") + 1]).write_text(json.dumps(
                {"batch_size": 4, "batch_source": "zero_shot_run", "config_hash": "h",
                 "zs_test_batch_size": 4}))  # fmt: skip
            return rc, t

    rec = Writer()
    ns = cell_ns(tmp_path, rec)
    out = run_cell(find(marker), ns)
    a = predict_native.build_parser().parse_args(rec.calls[0][3:])
    assert a.stage == "check" and a.model == "zs" and a.zs_run_dir == ns["ZS_RUN"]
    assert a.config == Path(ns["CFG"]) and a.batch_size is None and a.adapter_dir is None
    assert ns["BATCH"] == 4 and "NATIVE CHECKS PASSED" in out
    assert "no v0 folder to prove" in out  # V0_DIR None
    v0 = tmp_path / "v0"
    v0.mkdir()
    rec = Writer()
    run_cell(find(marker), cell_ns(tmp_path, rec, V0_DIR=v0, BATCH_SIZE=4))
    a = predict_native.build_parser().parse_args(rec.calls[0][3:])
    assert a.v0_dir == v0 and a.batch_size == 4
    rec = Writer()
    ns = cell_ns(tmp_path, rec, "ft", ADAPTER=tmp_path / "ad", ZS_TEST=tmp_path / "v15")
    run_cell(find(marker), ns)
    a = predict_native.build_parser().parse_args(rec.calls[0][3:])
    assert a.model == "ft" and a.adapter_dir == ns["ADAPTER"] and a.zs_test_dir == ns["ZS_TEST"]
    with pytest.raises(RuntimeError, match=r"NATIVE CHECK REFUSED \(exit 1\)"):
        run_cell(find(marker), cell_ns(tmp_path, Recorder({"check": [1]})))
    ns = cell_ns(tmp_path, Writer(), REUSE_ACK_SPIKE_DIFF=True)
    run_cell(find(marker), ns)
    assert "--ack-spike-diff" in ns["run_stream"].calls[0]


def test_estimate_cell(tmp_path: Path) -> None:
    rec = Recorder()
    ns = cell_ns(tmp_path, rec, "ft")
    run_cell(find('"shipdoc.predict_native", "estimate"'), ns)
    a = predict_native.build_parser().parse_args(rec.calls[0][3:])
    assert a.stage == "estimate" and a.model == "ft" and a.zs_run_dir == ns["ZS_RUN"]
    assert a.batch_size is None
    ns = cell_ns(tmp_path, rec, BATCH_SIZE=2)
    run_cell(find('"shipdoc.predict_native", "estimate"'), ns)
    assert rec.calls[1][rec.calls[1].index("--batch-size") + 1] == "2"


def test_the_zs_cells_run_for_zs_only_and_build_the_documented_commands(tmp_path: Path) -> None:
    cp = cli.build_parser()
    # skipped for ft: nothing is started, no zero-shot inference on the ft path
    rec = Recorder()
    ns = cell_ns(tmp_path, rec, "ft", SMOKE_PASSED=True)
    for marker in ZS_ONLY:
        out = run_cell(find(marker), ns)
        assert "is false" in out and "skipped" in out, marker
    assert rec.calls == []
    # smoke
    rec = Recorder()
    ns = cell_ns(tmp_path, rec)
    run_cell(find('"predict", "smoke"'), ns)
    a = cp.parse_args(rec.calls[0][3:])
    assert a.stage == "smoke" and a.status == ns["SMOKE_STATUS_PATH"] and a.run_id == "smoke_run"
    assert a.config == Path(ns["CFG"]) and ns["SMOKE_PASSED"] is True
    with pytest.raises(RuntimeError, match="SMOKE GATE FAILED"):
        run_cell(find('"predict", "smoke"'), cell_ns(tmp_path, Recorder({"smoke": [1]})))
    # batch: manual = the 02n size, with the 02n run as the dev contract
    rec = Recorder()
    ns = cell_ns(tmp_path, rec, SMOKE_PASSED=True)
    ns["DECISION_PATH"].write_text(
        json.dumps({"batch_size": 4, "source": "manual", "contract": "c"})
    )
    out = run_cell(find('"predict", "batch"'), ns)
    a = cp.parse_args(rec.calls[0][3:])
    assert a.stage == "batch" and a.batch_size == 4 and a.dev_run_dir == ns["DEV_RUN"]
    assert a.decision == ns["DECISION_PATH"] and "BATCH SIZE FOR THE TEST RUN: 4" in out
    ns["DECISION_PATH"].write_text(
        json.dumps({"batch_size": 1, "source": "manual", "contract": "c"})
    )
    with pytest.raises(AssertionError, match="02n run's"):  # a decision that is not the 02n size
        run_cell(find('"predict", "batch"'), ns)
    with pytest.raises(RuntimeError, match="The smoke gate has not passed"):
        run_cell(find('"predict", "batch"'), cell_ns(tmp_path, Recorder(), SMOKE_PASSED=False))
    with pytest.raises(RuntimeError, match="BATCH SIZE REFUSED"):
        run_cell(
            find('"predict", "batch"'),
            cell_ns(tmp_path, Recorder({"batch": [1]}), SMOKE_PASSED=True),
        )
    # run + determinism
    rec = Recorder()
    ns = cell_ns(tmp_path, rec, SMOKE_PASSED=True, plan={"shard_n_docs": 200})
    run_cell(find('"predict", "run"'), ns)
    a = cp.parse_args(rec.calls[0][3:])
    assert a.stage == "run" and (a.expect_docs, a.expect_pages) == (200, 280) and a.shard == "0/1"
    assert a.run_id == "run" and a.decision == ns["DECISION_PATH"]
    sessions = json.loads((ns["RUN_DIR"] / "sessions.json").read_text())
    assert sessions[-1]["exit_code"] == 0 and sessions[-1]["seconds"] is not None
    run_cell(find('"predict", "determinism"'), ns)
    a = cp.parse_args(rec.calls[1][3:])
    assert (
        a.stage == "determinism"
        and a.shard == "0/1"
        and ns["sessions"][-1]["stage"] == "determinism"
    )
    with pytest.raises(RuntimeError, match="Test run stopped"):
        run_cell(
            find('"predict", "run"'),
            cell_ns(tmp_path, Recorder({"run": [1]}), SMOKE_PASSED=True, plan={"shard_n_docs": 1}),
        )
    ns["run_stream"] = Recorder({"determinism": [1]})
    with pytest.raises(RuntimeError, match="DETERMINISM CHECK FAILED"):
        run_cell(find('"predict", "determinism"'), ns)


def test_the_ft_cells_run_for_ft_only_and_go_through_the_native_gate(tmp_path: Path) -> None:
    rec = Recorder()
    ns = cell_ns(tmp_path, rec, "zs")
    for marker in FT_ONLY:
        assert "is false" in run_cell(find(marker), ns), marker
    assert rec.calls == []  # an ft stage never starts on the zs path
    adapter = tmp_path / "ad"
    adapter.mkdir()
    (adapter / "manifest.json").write_text(json.dumps({"code_sha": FAKE_SHA, "precision": "bf16"}))
    pf = predict_ft.build_parser()
    rec = Recorder()
    ns = cell_ns(tmp_path, rec, "ft", ADAPTER=adapter)
    out = run_cell(find('"shipdoc.predict_native", "verify"'), ns)
    assert (
        rec.calls[0][1:3] == ["-m", "shipdoc.predict_native"] and "VERIFIED: final adapter" in out
    )
    args = pf.parse_args(rec.calls[0][3:])
    assert args.stage == "verify" and args.pin == FAKE_SHA and args.adapter_dir == adapter
    assert args.config == Path(ns["CFG"])
    ns["PINNED_SHA"] = "f" * 40  # the training SHA differs from the pin: the NATIVE diff is shown
    rec.calls.clear()
    out = run_cell(find('"shipdoc.predict_native", "verify"'), ns)
    assert "training code 0123456 != pin fffffff" in out
    git = next(c for c in rec.calls if c[0] == "git")
    assert "configs/finetune_qwen35_4b_native.yaml" in git
    with pytest.raises(RuntimeError, match="ADAPTER REFUSED"):
        run_cell(
            find('"shipdoc.predict_native", "verify"'),
            cell_ns(tmp_path, Recorder({"verify": [1]}), "ft", ADAPTER=adapter),
        )
    rec = Recorder()
    ns = cell_ns(tmp_path, rec, "ft", ADAPTER=adapter)
    run_cell(find('"shipdoc.predict_native", "infer"'), ns)
    args = pf.parse_args(rec.calls[0][3:])
    assert args.stage == "infer" and (args.batch_size, args.expect_docs) == (4, 200)  # BATCH
    assert args.decision == ns["DECISION_PATH"] and args.smoke_status == ns["SMOKE_STATUS_PATH"]
    assert args.config == Path(ns["CFG"]) and args.expect_pages == 280
    sessions = json.loads((ns["RUN_DIR"] / "sessions.json").read_text())
    assert sessions[-1]["exit_code"] == 0
    with pytest.raises(RuntimeError, match="Inference stopped"):
        run_cell(
            find('"shipdoc.predict_native", "infer"'),
            cell_ns(tmp_path, Recorder({"infer": [1]}), "ft", ADAPTER=adapter),
        )
    ns["sessions"], ns["SESSIONS_PATH"] = [], tmp_path / "s.json"
    ns["save_json"], ns["time"] = (lambda p, o: None), __import__("time")
    rec2 = Recorder()
    ns["run_stream"] = rec2
    run_cell(find('"shipdoc.predict_native", "determinism"'), ns)
    assert pf.parse_args(rec2.calls[0][3:]).stage == "determinism"
    ns["run_stream"] = Recorder({"determinism": [1]})
    with pytest.raises(RuntimeError, match="DETERMINISM CHECK FAILED"):
        run_cell(find('"shipdoc.predict_native", "determinism"'), ns)


def test_assemble_cell_per_path(tmp_path: Path) -> None:
    cp, pn = cli.build_parser(), predict_native.build_parser()
    rec = Recorder()
    ns = cell_ns(tmp_path, rec, "zs")
    run_cell(find('"predict", "assemble"'), ns)
    asm, fin = rec.calls
    args = cp.parse_args(asm[3:])
    assert args.expect_code_sha == FAKE_SHA and args.require_stack and args.no_rule == []
    assert args.dev_run_dir == ns["DEV_RUN"]  # zs: the batch contract against the 02n run
    assert args.ocr_cache == ns["OCR_ROOT"] and args.config == Path(ns["CFG"])
    fa = pn.parse_args(fin[3:])
    assert fa.stage == "finalize" and fa.model == "zs" and fa.zs_test_dir is None
    assert (
        fa.out_dir == ns["SUBMISSION_DIR"]
        and fa.run_dir == ns["RUN_DIR"]
        and fa.expect_pages == 280
    )
    assert fa.recheck == ns["META_DIR"] / "ocr_recheck.json"
    rec = Recorder()
    ns = cell_ns(tmp_path, rec, "ft")
    ns["SAMPLE_PATH"] = tmp_path / "sample.json"
    ns["SAMPLE_PATH"].write_text("{}")
    run_cell(find('"predict", "assemble"'), ns)
    asm, fin = rec.calls
    args = cp.parse_args(asm[3:])
    assert args.dev_run_dir is None and args.sample_submission  # ft: no dev contract
    fa = pn.parse_args(fin[3:])
    assert fa.model == "ft" and fa.zs_test_dir == ns["ZS_TEST"]
    with pytest.raises(RuntimeError, match="NOT submittable"):
        run_cell(
            find('"predict", "assemble"'), cell_ns(tmp_path, Recorder({"finalize": [1]}), "zs")
        )


def test_flags_cell_refuses_by_default_resolves_the_calibrator_and_builds_the_commands(
    tmp_path: Path,
) -> None:
    marker = '"shipdoc.predict_native", "flags"'
    rec = Recorder()
    repo, drive = tmp_path / "repo", tmp_path / "drive"
    (repo / "meta").mkdir(parents=True)
    drive.mkdir()
    ns = cell_ns(tmp_path, rec)  # CALIBRATOR_FILE None: the default
    with pytest.raises(
        RuntimeError, match=r"freeze_calibrator.py --arm zs.*after 02n and calibrate_v2"
    ):
        run_cell(find(marker), ns)
    with pytest.raises(RuntimeError, match=r"VALIDATED and on Drive"):
        run_cell(find(marker), ns)
    assert rec.calls == []  # nothing was run
    ns = cell_ns(tmp_path, rec, CALIBRATOR_FILE="meta/calibrator_zs_native.json")
    with pytest.raises(FileNotFoundError, match="neither the clone nor Drive"):
        run_cell(find(marker), ns)
    (repo / "meta" / "calibrator_zs_native.json").write_text("{}")
    with pytest.raises(RuntimeError, match="no validated test_predictions.json"):
        run_cell(find(marker), ns)
    (ns["SUBMISSION_DIR"] / "test_predictions.json").write_text("{}")
    out = run_cell(find(marker), ns)
    flag_cmd, check_cmd = rec.calls
    pn = predict_native.build_parser()
    fa = pn.parse_args(flag_cmd[3:])
    assert (
        fa.stage == "flags"
        and fa.model == "zs"
        and fa.calibrator == repo / "meta/calibrator_zs_native.json"
    )
    assert fa.batch_size == 4 and fa.zs_test_dir is None and fa.expect_docs == 200
    assert (fa.field_target, fa.doc_target) == (0.98, 0.98) and not fa.ack_spike_diff
    ca = pn.parse_args(check_cmd[3:])
    assert ca.stage == "check-flags" and ca.calibrator == fa.calibrator and "FAILED" not in out
    # the calibrator on Drive (not in the clone) is found second; ft passes the v15 folder
    (drive / "native").mkdir()
    (drive / "native" / "cal_ft.json").write_text("{}")
    rec = Recorder()
    ns = cell_ns(tmp_path, rec, "ft", CALIBRATOR_FILE="native/cal_ft.json", FLAGS_ZS_BATCH=2,
                 REUSE_ACK_SPIKE_DIFF=True)  # fmt: skip
    (ns["SUBMISSION_DIR"] / "test_predictions.json").write_text("{}")
    run_cell(find(marker), ns)
    fa = pn.parse_args(rec.calls[0][3:])
    assert fa.model == "ft" and fa.calibrator == drive / "native" / "cal_ft.json"
    assert fa.zs_test_dir == ns["ZS_TEST"] and fa.batch_size == 2 and fa.ack_spike_diff
    # a refusal raises but says the predictions are fine; failing checks only print
    with pytest.raises(RuntimeError, match="VALIDATED and on Drive"):
        run_cell(find(marker), cell_ns(tmp_path, Recorder({"flags": [1]}),
                 CALIBRATOR_FILE="meta/calibrator_zs_native.json"))  # fmt: skip
    ns = cell_ns(tmp_path, Recorder({"check-flags": [1]}),
                 CALIBRATOR_FILE="meta/calibrator_zs_native.json")  # fmt: skip
    out = run_cell(find(marker), ns)
    assert "REVIEW FLAGS CHECKS FAILED" in out and "predictions are not affected" in out


def banner(tmp_path: Path, manifest: dict, report: dict | None, model: str) -> str:
    sub = tmp_path / ("v15_0123456" if model == "zs" else "v2n_0123456")
    sub.mkdir(exist_ok=True)
    for f in ("validation_report.json", "manifest.json"):
        (sub / f).unlink(missing_ok=True)
    if report is not None:
        (sub / "validation_report.json").write_text(json.dumps(report))
    (sub / "manifest.json").write_text(json.dumps(manifest))
    ns = {"SUBMISSION_DIR": sub, "SHA7": "0123456", "RUN_ID": "r", "PINNED_SHA": FAKE_SHA,
          "MODEL": model, "SUB_NAME": sub.name, "SUB_PREFIX": sub.name.split("_")[0]}  # fmt: skip
    return run_cell(find("TEST PREDICTIONS"), ns)


def test_banner_for_both_paths_rejected_and_unassembled(tmp_path: Path) -> None:
    report = {
        "ok": True, "schema_ok": True, "docs_found": "200/200", "determinism_ok": True,
        "checks": {"native_config": {"ok": True, "detail": "x"}},
        "flags_ok": True, "flags_checks": {"flags_ids": {"ok": True, "detail": "200 flagged"}},
    }  # fmt: skip
    manifest: dict[str, Any] = {
        "code_sha": FAKE_SHA, "model": {"id": "Qwen/Qwen3.5-4B", "revision": FAKE_SHA},
        "config": {"hash": "e4b84ec2809625d5"}, "batch_size": 4, "batch_size_source": "manual",
        "batch_size_contract": "checked", "smoke": {"state": "passed"},
        "native": {"config": NATIVE_CFG, "config_hash": "e4b84ec2809625d5",
                   "max_pixels": 2196480, "model": "zs"},
        "ocr": {"timing": {"pages": 280, "seconds_total": 2.0, "seconds_per_page_mean": 0.1,
                           "seconds_per_page_p95": 0.2, "device": "gpu"}},
        "post_rules": {"touched_docs": {"R1": 1}, "skipped": {}}, "timings": {"wall_clock_s": 1.0},
        "review_flags": {"arm": "zs", "calibrator_sha256": "cd" * 32,
                         "summary": {"n_docs": 200, "n_docs_auto_accept": 3, "n_accept": 9,
                                     "n_review": 4},
                         "thresholds": {"field_tau": {"header": 0.2}, "doc_tau": None}},
    }  # fmt: skip
    out = banner(tmp_path, manifest, report, "zs")
    assert (
        "TEST PREDICTIONS v15 (zero-shot, base model, NATIVE + R1-R3) VALIDATED: SUBMITTABLE" in out
    )
    assert "native config qwen35_4b_img_only_native hash e4b84ec2809625d5 max_pixels 2196480" in out
    assert "REVIEW FLAGS OK" in out and "$SHIPDOC_SUBMISSIONS_DIR\\v15_0123456\\" in out
    assert (
        "final adapter" not in out and "DONE r assembled=True submittable=True flags_ok=True" in out
    )
    ft = {**manifest, "native": {**manifest["native"], "model": "ft",
                                 "zs_test": {"dir": "v15_0123456", "batch_size": 4,
                                             "code_sha": FAKE_SHA}},
          "ft": {"adapter_sha256": "ab" * 32, "train_code_sha": FAKE_SHA, "train_precision": "bf16",
                 "merge": {"n_lora_modules_merged": 200, "merge_dtype": "torch.float16",
                           "load_and_merge_s": 3.0},
                 "guard": {"ran": True, "ok": True, "fallback_to_1": False}}}  # fmt: skip
    out = banner(tmp_path, ft, report, "ft")
    assert "TEST PREDICTIONS v2n (fine-tuned, final native adapter, NATIVE + R1-R3)" in out
    assert (
        "abababababab" in out
        and "merge: 200 modules" in out
        and "zero-shot traces: v15_0123456" in out
    )
    assert "$SHIPDOC_SUBMISSIONS_DIR\\v2n_0123456\\" in out and "review_flags.json" in out
    assert "REVIEW FLAGS CHECKS FAILED" in banner(
        tmp_path, manifest, {**report, "flags_ok": False}, "zs"
    )
    nof = {k: v for k, v in manifest.items() if k != "review_flags"}
    assert "REVIEW FLAGS: not computed" in banner(tmp_path, nof, {**report, "flags_ok": None}, "zs")
    rej = banner(tmp_path, manifest, {**report, "ok": False}, "zs")
    assert "REJECTED: DO NOT SUBMIT" in rej and "test_predictions.REJECTED.json" in rej
    assert "NOT ASSEMBLED" in banner(tmp_path, {}, None, "zs")
