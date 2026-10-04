"""notebooks/04c_predict_test_v2.ipynb and shipdoc.predict_ft: structure, refusals, the pipeline.

Nothing here talks to Colab, Drive, a GPU, Paddle or the network. Cells are exec'd in a fake
namespace with `run_stream` replaced; the pipeline runs on the mock backend over the SYNTHETIC
corpus of tests/_synth.py (a stub peft is not even needed: the merged backend is a mock with the
`MergedHfBackend` interface). No real image, OCR text, label or prediction is used. The merge
itself (peft on the real model) is UNVERIFIED on a GPU.
"""

from __future__ import annotations

# ruff: noqa: F811  (the `world` fixture is imported, then requested by name, as pytest requires)
import ast
import hashlib
import importlib.util
import io
import json
import re
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import pytest
from _synth import SPECS
from test_flags import calibrator_for
from test_oof import DROP, MergedMock
from test_predict import (  # noqa: F401 - `_clean_sha` is an autouse fixture
    N_PAGES,
    SCHEMA,
    SMOKE_IDS,
    TEST_IDS,
    World,
    _clean_sha,
    assemble,
    needs_schema,
    world,
)
from test_reuse_v0 import SHAPES, fake_v0, write_ocr_cache

from shipdoc import cli, flags, oof, predict, predict_ft, reuse
from shipdoc import confidence_v2 as c2
from shipdoc import ocr_stage as ocr_stage_mod
from shipdoc import trainset as ts
from shipdoc.train import EXPECTED_LORA_MODULES

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "archive" / "04c_predict_test_v2.ipynb"
spec = importlib.util.spec_from_file_location(
    "colab_build_predict_v2", ROOT / "scripts" / "colab_build_predict_v2.py"
)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

NB = json.loads(NB_PATH.read_text(encoding="utf-8"))
CODE = [i for i, c in enumerate(NB["cells"]) if c["cell_type"] == "code"]
SHA = "0123456789abcdef0123456789abcdef01234567"
PARAMS = 1


def src(i: int) -> str:
    return "".join(NB["cells"][i]["source"])


def find(marker: str) -> int:
    """Index of the one code cell containing `marker`."""
    hits = [i for i in CODE if marker in src(i)]
    assert len(hits) == 1, (marker, hits)
    return hits[0]


def params_ns(**override: str) -> dict[str, Any]:
    text = src(PARAMS)
    for k, v in override.items():
        text = re.sub(rf"^{k} = .*$", f"{k} = {v}", text, flags=re.MULTILINE)
    ns: dict[str, Any] = {}
    exec(text, ns)
    return ns


def pinned(**override: str) -> dict[str, Any]:
    return params_ns(PINNED_SHA=f'"{SHA}"', **override)


def run_cell(i: int, ns: dict[str, Any]) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        exec(compile(src(i), f"cell{i}", "exec"), ns)
    return buf.getvalue()


# -------------------- structure


def test_committed_notebook_matches_builder() -> None:
    assert NB_PATH.read_text(encoding="utf-8") == builder.render()
    assert len(NB["cells"]) == 20


def test_outputs_cleared_cells_compile_lines_fit_100_columns_and_no_magics() -> None:
    for i, c in enumerate(NB["cells"]):
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None
            ast.parse(src(i))
            for ln in src(i).splitlines():
                assert len(ln) <= 100, (i, ln)
                assert not re.match(r"\s*[%!]", ln), (i, ln)  # no IPython magics / shell escapes


def test_the_pin_is_a_placeholder_that_raises_until_gg_pins() -> None:
    assert builder.PINNED_SHA == "SUPERSEDED-NOT-PINNED"
    assert 'PINNED_SHA = "SUPERSEDED-NOT-PINNED"' in src(PARAMS)
    with pytest.raises(ValueError, match="PINNED_SHA"):
        params_ns()
    ns = pinned()
    assert ns["PINNED_SHA"] == SHA and ns["SHA7"] == SHA[:7]
    clone = src(find("assert head == pinned_full"))
    assert "FILL" in clone  # the clone cell refuses a placeholder too


def test_params_cell_defaults() -> None:
    ns = pinned()
    assert (ns["BATCH_SIZE"], ns["ZS_BATCH_SIZE"], ns["SHARD"]) == (8, 8, "0/1")
    assert ns["REUSE_V0_DIR"] == "submissions/v0_42b812b" and ns["REUSE_ACK_SPIKE_DIFF"] is False
    assert (ns["EXPECTED_DOCS"], ns["EXPECTED_PAGES"]) == (200, 280)
    assert ns["CONFIG"] == "qwen35_4b_img_only" and ns["SPLIT"] == "test"
    assert ns["TRAIN_SHA7"] == SHA[:7]
    assert ns["ADAPTER_DIR"] == f"runs/ft_final_{SHA[:7]}_bf16/final"
    assert ns["RUN_ID"] == f"testft_qwen35_4b_img_only_keyed_{SHA[:7]}"
    assert ns["CALIBRATOR_FILE"] == "meta/calibrator_ft.json"
    assert (ns["FIELD_TARGET"], ns["DOC_TARGET"], ns["REUSE_OCR_FROM"]) == (0.98, 0.98, None)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"BATCH_SIZE": "0"}, "BATCH_SIZE"),
        ({"BATCH_SIZE": "None"}, "BATCH_SIZE"),  # no bench: an explicit size is required
        ({"BATCH_SIZE": "True"}, "BATCH_SIZE"),
        ({"ZS_BATCH_SIZE": "0"}, "ZS_BATCH_SIZE"),
        ({"TRAIN_SHA7": '"ABCDEF1"'}, "TRAIN_SHA7"),
        ({"TRAIN_SHA7": '"abc"'}, "TRAIN_SHA7"),
        ({"REUSE_V0_DIR": "None"}, "REUSE_V0_DIR"),
        ({"REUSE_V0_DIR": '""'}, "REUSE_V0_DIR"),
        ({"REUSE_ACK_SPIKE_DIFF": '"yes"'}, "REUSE_ACK_SPIKE_DIFF"),
        ({"FIELD_TARGET": "0.9"}, "FIELD_TARGET"),
        ({"DOC_TARGET": "0.99"}, "DOC_TARGET"),
        ({"REUSE_OCR_FROM": "7"}, "REUSE_OCR_FROM"),
    ],
)
def test_params_validation_refuses_bad_values(override: dict[str, str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        pinned(**override)


def test_adapter_dir_follows_the_training_sha() -> None:
    ns = pinned(TRAIN_SHA7='"abcdef1"')
    assert ns["ADAPTER_DIR"] == "runs/ft_final_abcdef1_bf16/final"


def test_cell_order() -> None:
    order = [
        find("drive.mount"),
        find("ESTIMATE (UNVERIFIED) OCR stage"),
        find('"uv", "sync", "--frozen"'),
        find('"shipdoc.predict_ft", "estimate"'),
        find('"predict", "plan"'),
        find('"adopt-ocr"'),
        find("def ocr_device"),
        find('"shipdoc.reuse", "check"'),
        find('"shipdoc.predict_ft", "verify"'),
        find('"shipdoc.predict_ft", "infer"'),
        find('"shipdoc.predict_ft", "determinism"'),
        find('"predict", "assemble"'),
        find('"shipdoc.flags"'),
        find("TEST PREDICTIONS v2"),
    ]
    assert order == sorted(order) and len(set(order)) == len(order)
    assert find('"shipdoc.predict_ft", "finalize"') == find('"predict", "assemble"')
    assert find('"check-flags"') == find('"shipdoc.flags"')
    # nothing touches the model before the adapter is verified and the ZS traces are cleared
    assert find('"shipdoc.reuse", "check"') < find('"shipdoc.predict_ft", "verify"')
    assert find('"shipdoc.predict_ft", "verify"') < find('"shipdoc.predict_ft", "infer"')
    # the estimate is printed before the first GPU stage
    assert find('"shipdoc.predict_ft", "estimate"') < find('"shipdoc.predict_ft", "infer"')


def test_infer_cell_documents_the_stage_order_and_the_pipeline_runs_it_in_that_order() -> None:
    import inspect

    text = src(find('"shipdoc.predict_ft", "infer"'))
    steps = ["VERIFY (again", "MERGE (fp16", "GUARD (12", "SMOKE (the gate", "RUN (the 200"]
    assert [text.index(s) for s in steps] == sorted(text.index(s) for s in steps)
    code = inspect.getsource(predict_ft.run_ft_infer)
    marks = [
        "= verify_stage(",
        "= merge_stage(",
        "= guard_stage(",
        "run_smoke_gate(",
        "pr.run_test(",
    ]
    pos = [code.index(m) for m in marks]
    assert pos == sorted(pos), dict(zip(marks, pos, strict=True))
    printed = [code.index(m) for m in ("== SMOKE ==", "== RUN ==")]
    assert printed == sorted(printed)


def test_no_zero_shot_inference_and_no_scoring_in_the_notebook() -> None:
    text = "\n".join(src(i) for i in CODE)
    assert '"predict", "run"' not in text and '"predict", "smoke"' not in text
    assert '"shipdoc", "spike"' not in text and '"oof", "compare"' not in text
    assert "full inference" not in text.lower() and "REUSE_V0" in text
    zs = src(find('"shipdoc.reuse", "check"'))
    assert "raise RuntimeError" in zs and "no zero-shot inference runs here" in zs
    assert "ZERO-SHOT TRACES NOT REUSABLE" in zs


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


def test_ocr_runs_only_in_its_own_venv_by_subprocess() -> None:
    text = "\n".join(src(i) for i in CODE)
    assert not re.search(r"^\s*(import|from)\s+paddle", text, flags=re.MULTILINE)
    assert "--group ocr" not in text and '"shipdoc", "ocr"' not in text
    ocr_cell = src(find("def ocr_device"))
    assert "/content/ocr-venv" in ocr_cell and "colab_ocr_test.sh" in ocr_cell


def test_the_notebook_never_writes_under_the_repo_tree() -> None:
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
    mount = src(find("drive.mount"))
    assert 'SUBMISSION_DIR = SUBMISSIONS_DIR / f"v2_{SHA7}"' in mount
    assert 'OCR_ROOT = DRIVE_DIR / "ocr_cache_test"' in mount


def test_no_secrets_no_test_ids_no_test_labels_no_wandb() -> None:
    text = json.dumps(NB)
    for pattern in (
        r"ghp_[A-Za-z0-9]{20,}",
        r"github_pat_",
        r"hf_[A-Za-z0-9]{20,}",
        r"sk-[A-Za-z0-9]{20,}",
        r"AKIA[0-9A-Z]{12}",
    ):
        assert not re.search(pattern, text), pattern
    assert "ANTHROPIC_API_KEY" not in text and "wandb" not in text.lower()
    assert not re.search(r"test_\d{4}", text)  # no test doc id
    code = "\n".join(src(i) for i in CODE)
    for ln in code.splitlines():
        if "labels" in ln and not ln.lstrip().startswith("#"):
            assert "DEV_LABELS" in ln or "dev" in ln or 'SPLIT / "labels"' in ln, ln


def _flags(text: str) -> set[str]:
    return set(re.findall(r"\"(--[a-z0-9-]+)\"", text))


def _subparser(parser: Any, *names: str) -> Any:
    cur = parser
    for n in names:
        act = next(a for a in cur._actions if getattr(a, "choices", None) and n in a.choices)
        cur = act.choices[n]
    return cur


def test_every_flag_the_notebook_passes_exists() -> None:
    pp = predict_ft.build_parser()
    for stage in (
        "estimate",
        "verify",
        "infer",
        "determinism",
        "finalize",
        "check-flags",
        "adopt-ocr",
    ):
        text = src(find(f'"shipdoc.predict_ft", "{stage}"'))
        chunk = text[text.index(f'"shipdoc.predict_ft", "{stage}"') :].split("]")[0]
        known = set(_subparser(pp, stage)._option_string_actions)
        assert _flags(chunk) <= known, (stage, _flags(chunk) - known)
    fl = set(flags.build_parser()._option_string_actions)
    chunk = src(find('"shipdoc.flags"'))
    chunk = chunk[chunk.index('"shipdoc.flags"') :].split("]")[0]
    assert _flags(chunk) <= fl, _flags(chunk) - fl
    rp = reuse.build_parser()
    text = src(find('"shipdoc.reuse", "check"'))
    chunk = text[text.index('"shipdoc.reuse", "check"') :].split("]")[0]
    assert _flags(chunk) <= set(_subparser(rp, "check")._option_string_actions)
    asm = src(find('"predict", "assemble"'))
    chunk = asm[asm.index('"predict", "assemble"') :].split("]")[0]
    known = set(_subparser(cli.build_parser(), "predict", "assemble")._option_string_actions)
    assert _flags(chunk) <= known | {"--out-dir"}, _flags(chunk) - known
    op = ocr_stage_mod.build_parser()
    ocr_cell = src(find("def ocr_device"))
    for stage in ("check", "finalize", "recheck", "device"):
        assert set(_subparser(op, stage)._option_string_actions) & _flags(ocr_cell)


# -------------------- cells that run locally


class Recorder:
    """A fake `run_stream`: records commands, answers by stage name."""

    def __init__(self, answers: dict[str, list[int]] | None = None, **ns: Any) -> None:
        self.calls: list[list[str]] = []
        self.answers = {k: list(v) for k, v in (answers or {}).items()}
        self.ns = ns

    def __call__(self, cmd: list[str], tail: int = 40, env: dict | None = None) -> tuple[int, Any]:
        self.calls.append([str(c) for c in cmd])
        key = next((k for k in self.answers if k in cmd), None)
        rc = self.answers[key].pop(0) if key and self.answers[key] else 0
        return rc, ["last line"]

    def has(self, *words: str) -> list[list[str]]:
        return [c for c in self.calls if all(w in c for w in words)]


def cell_ns(tmp_path: Path, rec: Recorder, **extra: Any) -> dict[str, Any]:
    meta, sub, run = tmp_path / "meta", tmp_path / "sub", tmp_path / "run"
    for d in (meta, sub, run):
        d.mkdir(exist_ok=True)
    ns = {
        "Path": Path,
        "json": json,
        "PY": "py",
        "run_stream": rec,
        "REPO": tmp_path / "repo",
        "META_DIR": meta,
        "SUBMISSION_DIR": sub,
        "RUN_DIR": run,
        "ADAPTER": tmp_path / "ad",
        "V0_DIR": tmp_path / "v0",
        "OCR_ROOT": tmp_path / "ocr",
        "OCR_FROM": None,
        "PINNED_SHA": SHA,
        "SHA7": SHA[:7],
        "RUN_ID": "run",
        "CONFIG": "qwen35_4b_img_only",
        "EXPECTED_DOCS": 200,
        "EXPECTED_PAGES": 280,
        "BATCH_SIZE": 8,
        "ZS_BATCH_SIZE": 8,
        "SMOKE_DOCS": "splits/smoke5.json",
        "BENCH_DOCS": "splits/bench12.json",
        "DECISION_PATH": meta / "d.json",
        "SMOKE_STATUS_PATH": meta / "s.json",
        "SCHEMA_PATH": tmp_path / "schema.json",
        "SAMPLE_PATH": tmp_path / "nosample.json",
        "REUSE_ACK_SPIKE_DIFF": False,
        "FIELD_TARGET": 0.98,
        "DOC_TARGET": 0.98,
        "CALIBRATOR_FILE": "meta/calibrator_ft.json",
        "REUSE_OCR_FROM": None,
        **extra,
    }
    return ns


def test_estimate_and_adopt_cells(tmp_path: Path) -> None:
    rec = Recorder()
    ns = cell_ns(tmp_path, rec)
    run_cell(find('"shipdoc.predict_ft", "estimate"'), ns)
    cmd = rec.calls[0]
    predict_ft.build_parser().parse_args(cmd[3:])
    assert cmd[cmd.index("--batch-size") + 1] == "8"
    out = run_cell(find('"adopt-ocr"'), ns)  # OCR_FROM None: nothing adopted
    assert "REUSE_OCR_FROM is None" in out and not rec.has("adopt-ocr")
    ns2 = cell_ns(
        tmp_path, Recorder({"adopt-ocr": [1]}), OCR_FROM=tmp_path / "v1", REUSE_OCR_FROM="x"
    )
    out = run_cell(find('"adopt-ocr"'), ns2)  # a refusal is not fatal
    assert "NOT ADOPTED" in out
    cmd = ns2["run_stream"].calls[0]
    args = predict_ft.build_parser().parse_args(cmd[3:])
    assert args.timing_out == ns2["SUBMISSION_DIR"] / "ocr_timing.json"
    assert args.recheck_out == ns2["META_DIR"] / "ocr_recheck.json"


def test_ocr_estimate_cell_labels_unmeasured_numbers_and_the_ocr_reuse() -> None:
    ns: dict[str, Any] = {"REPO": ROOT, "EXPECTED_PAGES": 280, "REUSE_OCR_FROM": None}
    out = run_cell(find("ESTIMATE (UNVERIFIED) OCR stage"), ns)
    assert "ESTIMATE (UNVERIFIED) OCR stage" in out and "ASSUMED" in out and "CU" in out
    assert "REUSE_OCR_FROM = None" in out and "cost nothing" in out and "V0" not in out


def test_zs_check_cell_stops_the_notebook_when_the_traces_are_not_reusable(tmp_path: Path) -> None:
    rec = Recorder({"check": [0]})
    ns = cell_ns(tmp_path, rec)
    out = run_cell(find('"shipdoc.reuse", "check"'), ns)
    cmd = rec.calls[0]
    reuse.build_parser().parse_args(cmd[3:])
    assert cmd[cmd.index("--batch-size") + 1] == "8" and "--ack-spike-diff" not in cmd
    assert "ZERO-SHOT TRACES REUSABLE" in out
    rec = Recorder({"check": [3]})
    ns = cell_ns(tmp_path, rec, REUSE_ACK_SPIKE_DIFF=True)
    (ns["META_DIR"] / "reuse_decision.json").write_text(json.dumps({"reasons": ["seed: 7 != 42"]}))
    with pytest.raises(RuntimeError, match=r"NOT REUSABLE.*seed: 7 != 42.*no zero-shot inference"):
        run_cell(find('"shipdoc.reuse", "check"'), ns)
    assert "--ack-spike-diff" in rec.calls[0]


def test_verify_infer_determinism_cells_build_the_documented_commands(tmp_path: Path) -> None:
    adapter = tmp_path / "ad"
    adapter.mkdir()
    (adapter / "manifest.json").write_text(json.dumps({"code_sha": SHA, "precision": "bf16"}))
    rec = Recorder()
    ns = cell_ns(tmp_path, rec, ADAPTER=adapter)
    out = run_cell(find('"shipdoc.predict_ft", "verify"'), ns)
    cmd = rec.calls[0]
    args = predict_ft.build_parser().parse_args(cmd[3:])
    assert args.pin == SHA and args.adapter_dir == adapter and "VERIFIED: final adapter" in out
    ns["PINNED_SHA"] = "f" * 40  # the training SHA differs from the pin: the diff is shown
    rec.calls.clear()
    out = run_cell(find('"shipdoc.predict_ft", "verify"'), ns)
    assert "training code 0123456 != pin fffffff" in out and any(c[0] == "git" for c in rec.calls)
    ns["PINNED_SHA"] = SHA
    rec_bad = Recorder({"verify": [1]})
    ns_bad = cell_ns(tmp_path, rec_bad, ADAPTER=adapter)
    with pytest.raises(RuntimeError, match="ADAPTER REFUSED"):
        run_cell(find('"shipdoc.predict_ft", "verify"'), ns_bad)
    # infer: the session is logged, the command carries every stage input, a failure raises
    rec = Recorder()
    ns = cell_ns(tmp_path, rec, ADAPTER=adapter, CFG="configs/spike_qwen35_4b_img_only.yaml")
    run_cell(find('"shipdoc.predict_ft", "infer"'), ns)
    cmd = rec.calls[0]
    args = predict_ft.build_parser().parse_args(cmd[3:])
    assert (args.batch_size, args.expect_docs, args.expect_pages) == (8, 200, 280)
    assert args.decision == ns["DECISION_PATH"] and args.smoke_status == ns["SMOKE_STATUS_PATH"]
    sessions = json.loads((ns["RUN_DIR"] / "sessions.json").read_text())
    assert sessions[-1]["exit_code"] == 0 and sessions[-1]["seconds"] is not None
    rec = Recorder({"infer": [1]})
    ns = cell_ns(tmp_path, rec, ADAPTER=adapter, CFG="c")
    with pytest.raises(RuntimeError, match="Inference stopped"):
        run_cell(find('"shipdoc.predict_ft", "infer"'), ns)
    # determinism: appended to the sessions, a difference raises
    rec = Recorder()
    ns = cell_ns(tmp_path, rec, ADAPTER=adapter, CFG="c")
    ns["sessions"] = []
    ns["SESSIONS_PATH"] = tmp_path / "sessions.json"
    ns["save_json"] = lambda p, o: None
    ns["time"] = __import__("time")
    run_cell(find('"shipdoc.predict_ft", "determinism"'), ns)
    predict_ft.build_parser().parse_args(rec.calls[0][3:])
    assert ns["sessions"][-1]["stage"] == "determinism"
    ns["run_stream"] = Recorder({"determinism": [1]})
    with pytest.raises(RuntimeError, match="DETERMINISM CHECK FAILED"):
        run_cell(find('"shipdoc.predict_ft", "determinism"'), ns)


def test_assemble_cell_runs_assemble_then_both_finalizers_and_raises_on_failure(
    tmp_path: Path,
) -> None:
    rec = Recorder()
    ns = cell_ns(tmp_path, rec, CFG="configs/spike_qwen35_4b_img_only.yaml")
    ns["SAMPLE_PATH"] = tmp_path / "sample.json"
    ns["SAMPLE_PATH"].write_text("{}")
    run_cell(find('"predict", "assemble"'), ns)
    asm, fin = rec.calls
    args = cli.build_parser().parse_args(asm[3:])
    assert args.expect_code_sha == SHA and args.require_stack and args.no_rule == []
    assert args.ocr_cache == ns["OCR_ROOT"] and args.dev_run_dir is None  # no dev contract
    assert args.smoke_status == ns["SMOKE_STATUS_PATH"] and args.sample_submission
    fa = predict_ft.build_parser().parse_args(fin[3:])
    assert fa.stage == "finalize" and fa.v0_dir == ns["V0_DIR"] and fa.expect_pages == 280
    assert fa.run_dir == ns["RUN_DIR"] and fa.recheck == ns["META_DIR"] / "ocr_recheck.json"
    ns_bad = cell_ns(tmp_path, Recorder({"finalize": [1]}), CFG="c")
    with pytest.raises(RuntimeError, match="NOT submittable"):
        run_cell(find('"predict", "assemble"'), ns_bad)


def test_flags_cell_needs_the_calibrator_and_validated_predictions(tmp_path: Path) -> None:
    rec = Recorder()
    repo = tmp_path / "repo"
    (repo / "meta").mkdir(parents=True)
    ns = cell_ns(tmp_path, rec, REPO=repo)
    with pytest.raises(FileNotFoundError, match="calibrator_ft.json"):
        run_cell(find('"shipdoc.flags"'), ns)
    (repo / "meta" / "calibrator_ft.json").write_text("{}")
    with pytest.raises(RuntimeError, match="no validated test_predictions.json"):
        run_cell(find('"shipdoc.flags"'), ns)
    (ns["SUBMISSION_DIR"] / "test_predictions.json").write_text("{}")
    out = run_cell(find('"shipdoc.flags"'), ns)
    flag_cmd, check_cmd = rec.calls
    fa = flags.build_parser().parse_args(flag_cmd[3:])
    assert fa.arm == "ft" and fa.zs_v0_dir == ns["V0_DIR"] and fa.batch_size == 8
    assert fa.out == ns["SUBMISSION_DIR"] / "review_flags.json" and fa.expect_docs == 200
    assert (fa.field_target, fa.doc_target) == (0.98, 0.98) and not fa.ack_spike_diff
    ca = predict_ft.build_parser().parse_args(check_cmd[3:])
    assert ca.stage == "check-flags" and "FAILED" not in out
    # a refusal raises, but says the predictions are fine; failing checks only print
    ns_ref = cell_ns(tmp_path, Recorder({"shipdoc.flags": [3]}), REPO=repo)
    with pytest.raises(RuntimeError, match="VALIDATED and on Drive"):
        run_cell(find('"shipdoc.flags"'), ns_ref)
    ns_chk = cell_ns(tmp_path, Recorder({"check-flags": [1]}), REPO=repo)
    out = run_cell(find('"shipdoc.flags"'), ns_chk)
    assert "REVIEW FLAGS CHECKS FAILED" in out and "predictions are not affected" in out


def banner(tmp_path: Path, manifest: dict, report: dict | None) -> str:
    sub = tmp_path / "v2_0123456"
    sub.mkdir(exist_ok=True)
    for f in ("validation_report.json", "manifest.json"):
        (sub / f).unlink(missing_ok=True)
    if report is not None:
        (sub / "validation_report.json").write_text(json.dumps(report))
    (sub / "manifest.json").write_text(json.dumps(manifest))
    ns = {"SUBMISSION_DIR": sub, "SHA7": "0123456", "RUN_ID": "r", "PINNED_SHA": SHA}
    return run_cell(find("TEST PREDICTIONS v2"), ns)


def test_banner_for_validated_with_flags_rejected_and_unassembled(tmp_path: Path) -> None:
    report = {
        "ok": True,
        "schema_ok": True,
        "docs_found": "200/200",
        "determinism_ok": True,
        "checks": {"ft_adapter_verified": {"ok": True, "detail": "x"}},
        "flags_ok": True,
        "flags_checks": {"flags_ids": {"ok": True, "detail": "200 flagged"}},
    }
    manifest = {
        "code_sha": SHA,
        "model": {"id": "Qwen/x+lora:abcdef012345", "revision": SHA},
        "config": {"hash": "h"},
        "batch_size": 8,
        "batch_size_source": "bench",
        "smoke": {"state": "passed"},
        "ft": {
            "adapter_sha256": "ab" * 32,
            "train_code_sha": SHA,
            "train_precision": "bf16",
            "merge": {
                "n_lora_modules_merged": 200,
                "merge_dtype": "torch.float16",
                "load_and_merge_s": 3.0,
            },
            "guard": {"ran": True, "ok": True, "fallback_to_1": False},
        },
        "ocr": {
            "timing": {
                "pages": 280,
                "seconds_total": 2.0,
                "seconds_per_page_mean": 0.1,
                "seconds_per_page_p95": 0.2,
                "device": "gpu",
            }
        },
        "post_rules": {"touched_docs": {"R1": 1}, "skipped": {}},
        "timings": {"wall_clock_s": 1.0},
        "review_flags": {
            "arm": "ft",
            "calibrator_sha256": "cd" * 32,
            "summary": {"n_docs": 200, "n_docs_auto_accept": 3, "n_accept": 9, "n_review": 4},
            "thresholds": {"field_tau": {"header": 0.2}, "doc_tau": None},
        },
    }
    out = banner(tmp_path, manifest, report)
    assert "VALIDATED: SUBMITTABLE" in out and "REVIEW FLAGS OK" in out
    assert "abababababab" in out and "merge: 200 modules" in out
    assert "$SHIPDOC_SUBMISSIONS_DIR\\v2_0123456\\" in out and "review_flags.json" in out
    assert "DONE r assembled=True submittable=True flags_ok=True" in out
    bad = banner(
        tmp_path,
        {**manifest, "review_flags": {**manifest["review_flags"]}},
        {**report, "flags_ok": False},
    )
    assert "REVIEW FLAGS CHECKS FAILED" in bad
    nof = {k: v for k, v in manifest.items() if k != "review_flags"}
    assert "REVIEW FLAGS: not computed" in banner(tmp_path, nof, {**report, "flags_ok": None})
    rej = banner(tmp_path, manifest, {**report, "ok": False})
    assert "REJECTED: DO NOT SUBMIT" in rej and "test_predictions.REJECTED.json" in rej
    assert "NOT ASSEMBLED" in banner(tmp_path, {}, None)


# -------------------- final adapter verification

FOLDS = {
    "k": 3,
    "folds": [
        {"fold": 0, "val_doc_ids": ["train_0000", "train_0001", "dev_0000"]},
        {"fold": 1, "val_doc_ids": ["train_0002", "train_0003", "dev_0001"]},
        {"fold": 2, "val_doc_ids": ["train_0004", "dev_0002"]},
    ],
}
SPLIT = ts.stage_split("final", FOLDS)
SHA40 = "b" * 40
KEYED = ROOT / "configs" / "spike_qwen35_4b_img_only.yaml"


def make_final_adapter(final: Path, cfg: Any, **override: Any) -> Path:
    """A fake `final/` folder: dummy weight files + the manifest `train.py` writes (final)."""
    (final / "peft").mkdir(parents=True, exist_ok=True)
    (final / "adapter.pt").write_bytes(b"fake adapter state")
    (final / "peft" / "adapter_model.safetensors").write_bytes(b"fake safetensors")
    (final / "peft" / "adapter_config.json").write_text("{}")
    ids = sorted(SPLIT.train_ids)
    m: dict[str, Any] = {
        "stage": "final",
        "fold": None,
        "manifest_hash": hashlib.sha256(("final|" + ",".join(ids)).encode()).hexdigest()[:16],
        "n_heldout_eval_pages": 24,
        "code_sha": SHA40,
        "train_config_signature": "sig",
        "train_doc_ids": ids,
        "n_train_docs": len(ids),
        "heldout_doc_ids": sorted(SPLIT.heldout_ids),
        "lora": {"r": 16, "alpha": 32, "n_modules": 200, "n_trainable": 30_474_240},
        "inference_keys": oof.inference_keys_of_config(cfg),
        "signature": "s",
        "steps": 10,
        "total_steps": 10,
        "n_train_pages": 99,
        "base_repo": cfg.backend.repo,
        "base_revision": cfg.backend.revision,
        "precision": "bf16",
        "adapter_sha256": hashlib.sha256(b"fake adapter state").hexdigest(),
        "peft_sha256": {
            "adapter_model.safetensors": hashlib.sha256(b"fake safetensors").hexdigest()
        },
    }
    for k, v in override.items():
        if v is DROP:
            m.pop(k, None)
        else:
            m[k] = v
    (final / "manifest.json").write_text(json.dumps(m, indent=1))
    return final


@pytest.fixture()
def cfg() -> Any:
    from shipdoc import spike

    return spike.load_config(KEYED)


def verify(tmp_path: Path, cfg: Any, name: str = "ad", **override: Any) -> dict[str, Any]:
    d = make_final_adapter(tmp_path / name / "final", cfg, **override)
    return predict_ft.verify_final_adapter(
        d, cfg=cfg, folds=FOLDS, pin_sha=SHA40, reachable=lambda sha: True
    )


def failed(rep: dict[str, Any]) -> list[str]:
    return [r["check"] for r in rep["rows"] if not r["ok"]]


def test_a_good_final_adapter_passes_and_the_table_prints(tmp_path: Path, cfg: Any) -> None:
    rep = verify(tmp_path, cfg)
    assert rep["ok"] and not rep["warnings"] and rep["n_train_docs"] == 5
    text = predict_ft.format_verification(rep)
    assert "FINAL ADAPTER MANIFEST VERIFICATION: PASSED" in text and "train_0000" not in text
    predict_ft.assert_verified(rep)
    names = {r["check"] for r in rep["rows"]}
    assert {
        "stage",
        "fold id",
        "no dev document in training",
        "no test document in training",
        "training set = the train_* documents",
        "recorded held-out ids = the dev documents",
        "inference key prompt_version",
        "adapter.pt sha256",
        "lora modules",
    } <= names


def spike_cfg() -> Any:
    from shipdoc import spike

    return spike.load_config(KEYED)


REFUSALS = {
    "a smoke adapter": ({"stage": "smoke"}, ["stage"]),
    "a fold adapter": ({"stage": "fold0", "fold": 0}, ["stage", "fold id"]),
    "a fold id only": ({"fold": 1}, ["fold id"]),
    "dirty training code": ({"code_sha": SHA40 + "+dirty"}, ["training code sha"]),
    "short training code sha": ({"code_sha": "abc"}, ["training code sha"]),
    "no training code sha": ({"code_sha": DROP}, ["training code sha"]),
    "wrong base revision": ({"base_revision": "c" * 40}, ["base revision"]),
    "wrong base repo": ({"base_repo": "other/model"}, ["base repo"]),
    "wrong lora rank": (
        {"lora": {"r": 8, "n_modules": 200, "n_trainable": 15_237_120}},
        ["lora r", "lora trainable params"],
    ),
    "wrong module count": (
        {"lora": {"r": 16, "n_modules": 196, "n_trainable": 30_474_240}},
        ["lora modules"],
    ),
    "no training ids": ({"train_doc_ids": DROP}, ["training doc ids"]),
    "empty training ids": ({"train_doc_ids": []}, ["training doc ids"]),
    "a dev document in training": (
        {"train_doc_ids": sorted([*SPLIT.train_ids, "dev_0000"])},
        [
            "no dev document in training",
            "training set = the train_* documents",
            "manifest_hash matches the ids",
        ],
    ),
    "a test document in training": (
        {"train_doc_ids": sorted([*SPLIT.train_ids, "test_0001"])},
        [
            "no test document in training",
            "training set = the train_* documents",
            "manifest_hash matches the ids",
        ],
    ),
    "training set too small": (
        {"train_doc_ids": sorted(SPLIT.train_ids)[:-1]},
        ["training set = the train_* documents", "manifest_hash matches the ids"],
    ),
    "wrong held-out ids": (
        {"heldout_doc_ids": ["dev_0000"]},
        ["recorded held-out ids = the dev documents"],
    ),
    "manifest hash": ({"manifest_hash": "0" * 16}, ["manifest_hash matches the ids"]),
    "prompt version": (
        {"inference_keys": {**oof.inference_keys_of_config(spike_cfg()), "prompt_version": "v1"}},
        ["inference key prompt_version"],
    ),
    "max pixels": (
        {"inference_keys": {**oof.inference_keys_of_config(spike_cfg()), "max_pixels": 1}},
        ["inference key max_pixels"],
    ),
    "no inference keys": ({"inference_keys": DROP}, ["inference key model_repo"]),
    "adapter hash": ({"adapter_sha256": "0" * 64}, ["adapter.pt sha256"]),
    "peft hash missing": ({"peft_sha256": DROP}, ["peft/adapter_model.safetensors sha256"]),
    "no precision": ({"precision": DROP}, ["training precision"]),
}


@pytest.mark.parametrize("name", sorted(REFUSALS))
def test_each_refusal_fails_its_check_and_assert_verified_raises(
    tmp_path: Path, cfg: Any, name: str
) -> None:
    override, want = REFUSALS[name]
    rep = verify(tmp_path, cfg, **override)
    assert not rep["ok"]
    assert set(want) <= set(failed(rep)), (name, failed(rep))
    with pytest.raises(predict_ft.FtError, match="refused"):
        predict_ft.assert_verified(rep)


def test_missing_files_unreachable_sha_and_pin_difference(tmp_path: Path, cfg: Any) -> None:
    d = make_final_adapter(tmp_path / "ad" / "final", cfg)
    (d / "peft" / "adapter_config.json").unlink()
    (d / "adapter.pt").unlink()
    rep = predict_ft.verify_final_adapter(d, cfg=cfg, folds=FOLDS)
    assert {"adapter.pt sha256", "peft/adapter_config.json"} <= set(failed(rep))
    d2 = make_final_adapter(tmp_path / "ad2" / "final", cfg)
    rep = predict_ft.verify_final_adapter(d2, cfg=cfg, folds=FOLDS, reachable=lambda s: False)
    assert failed(rep) == ["training code sha reachable"]
    rep = predict_ft.verify_final_adapter(d2, cfg=cfg, folds=FOLDS, pin_sha="d" * 40)
    assert rep["ok"] and "differs from this notebook's pin" in rep["warnings"][0]
    rep = predict_ft.verify_final_adapter(tmp_path / "nowhere", cfg=cfg, folds=FOLDS)
    assert not rep["ok"] and failed(rep) == ["manifest.json"]


# -------------------- the pipeline (mock)


def ft_world_run(
    w: World,
    tmp_path: Path,
    cfg: Any,
    *,
    backend: Any = None,
    batch: int = 2,
    run_id: str = "ft0",
    adapter: Path | None = None,
    det: bool = False,
    **kw: Any,
) -> tuple[dict[str, Any], MergedMock, Path, list[str]]:
    ad = adapter or make_final_adapter(tmp_path / "ad" / "final", cfg)
    be = backend or MergedMock(w.gold, w.cfg, null_header=("buyer_name",))
    msgs: list[str] = []
    made: list[Any] = []

    def factory(d: Path, sha: str) -> Any:
        be.model_id = f"{w.cfg.backend.repo}+lora:{sha[:12]}"
        made.append(sha)
        return be

    rec = predict_ft.run_ft_infer(
        cfg=w.cfg,
        adapter_dir=ad,
        folds=FOLDS,
        run_id=run_id,
        runs_root=w.runs,
        data_root=w.data,
        decision_path=tmp_path / "meta" / "decision.json",
        smoke_status_path=tmp_path / "meta" / "smoke.json",
        smoke_docs=SMOKE_IDS,
        bench_docs=[s[0] for s in SPECS][:3],
        batch_size=batch,
        backend_factory=factory,
        pin_sha=SHA40,
        reachable=lambda s: True,
        expect_docs=len(TEST_IDS),
        expect_pages=N_PAGES,
        out=msgs.append,
        **kw,
    )
    if det:  # the determinism pass in a 'fresh process': another factory call merges again
        predict_ft.run_ft_determinism(
            cfg=w.cfg,
            adapter_dir=ad,
            folds=FOLDS,
            run_id=run_id,
            runs_root=w.runs,
            data_root=w.data,
            decision_path=tmp_path / "meta" / "decision.json",
            backend_factory=factory,
            pin_sha=SHA40,
            reachable=lambda s: True,
            out=msgs.append,
        )
    return rec, be, ad, msgs


def test_ft_pipeline_verifies_merges_guards_smokes_runs_and_records_everything(
    world: World, tmp_path: Path, cfg: Any
) -> None:
    rec, be, ad, msgs = ft_world_run(world, tmp_path, cfg)
    heads = [m.split(" ==")[0] + " ==" for m in msgs if m.startswith("== ")]
    assert heads == ["== VERIFY ==", "== MERGE ==", "== GUARD ==", "== SMOKE ==", "== RUN =="]
    assert be.loads == 1 and rec["merge"]["n_lora_modules_merged"] == EXPECTED_LORA_MODULES
    assert rec["guard"]["ran"] and rec["guard"]["ok"] and not rec["guard"]["fallback_to_1"]
    assert rec["decision"]["batch_size"] == 2 and rec["verification"]["ok"]
    run = world.runs / "ft0"
    traces = [json.loads(x) for x in (run / "trace.jsonl").read_text().splitlines()]
    assert sorted(t["doc_id"] for t in traces) == sorted(TEST_IDS)
    assert all(isinstance(p.get("field_logprobs"), list) for t in traces for p in t["pages"])
    man = json.loads((run / "manifest.json").read_text())
    assert man["batch_size"] == 2 and man["logprobs"] is True and man["split"] == "test"
    assert man["model"]["id"].endswith("+lora:" + rec["adapter_sha256"][:12])
    assert (run / "env.json").is_file() and (run / "guard" / "bench_result.json").is_file()
    assert json.loads((run / "ft_run.json").read_text())["adapter_sha256"] == rec["adapter_sha256"]
    dec = json.loads((tmp_path / "meta" / "decision.json").read_text())
    assert dec["source"] == predict_ft.GUARD_SOURCE and dec["bench"]["chosen"] == 2
    smoke = json.loads((tmp_path / "meta" / "smoke.json").read_text())
    assert smoke["state"] == "passed"
    text = "\n".join(msgs) + json.dumps(rec)
    assert "Supplier" not in text and "INV-" not in text  # no values


def test_a_failed_guard_falls_back_to_batch_1_and_a_resume_keeps_the_decision(
    world: World, tmp_path: Path, cfg: Any
) -> None:
    flip = MergedMock(world.gold, world.cfg, null_header=("buyer_name",), flip_in_batches=True)
    rec, _, ad, msgs = ft_world_run(world, tmp_path, cfg, backend=flip)
    assert rec["guard"]["fallback_to_1"] and rec["decision"]["batch_size"] == 1
    assert rec["decision"]["requested_batch_size"] == 2
    assert any("FALLING BACK TO BATCH 1" in m for m in msgs)
    man = json.loads((world.runs / "ft0" / "manifest.json").read_text())
    assert man["batch_size"] == 1 and man["bench"]["deviation"]["fallback_to_1"] is True
    # a resume with a backend that would now pass the guard keeps the stored decision (batch 1)
    again = MergedMock(world.gold, world.cfg, null_header=("buyer_name",))
    rec2, _, _, msgs2 = ft_world_run(world, tmp_path, cfg, backend=again, adapter=ad)
    assert rec2["decision"]["batch_size"] == 1 and rec2["guard"]["reused_decision"]
    assert any("stored batch decision stands" in m for m in msgs2)


def test_the_pipeline_refuses_before_any_backend_is_built(
    world: World, tmp_path: Path, cfg: Any
) -> None:
    bad = make_final_adapter(tmp_path / "bad" / "final", cfg, stage="fold0", fold=0)
    built: list[int] = []

    def factory(d: Path, s: str) -> Any:
        built.append(1)
        return MergedMock(world.gold, world.cfg)

    with pytest.raises(predict_ft.FtError, match="verification FAILED"):
        predict_ft.run_ft_infer(
            cfg=world.cfg,
            adapter_dir=bad,
            folds=FOLDS,
            run_id="x",
            runs_root=world.runs,
            data_root=world.data,
            decision_path=tmp_path / "d.json",
            smoke_status_path=tmp_path / "s.json",
            smoke_docs=SMOKE_IDS,
            bench_docs=SMOKE_IDS[:2],
            batch_size=2,
            backend_factory=factory,
            out=lambda m: None,
        )
    assert built == [] and not (world.runs / "x").exists()
    with pytest.raises(predict_ft.FtError, match="batch size"):
        ft_world_run(world, tmp_path, cfg, batch=0)


def test_a_merge_of_the_wrong_module_count_stops_the_run(
    world: World, tmp_path: Path, cfg: Any
) -> None:
    short = MergedMock(world.gold, world.cfg, n_merged=196)
    with pytest.raises(predict_ft.FtError, match="did not merge 200"):
        ft_world_run(world, tmp_path, cfg, backend=short)
    assert not (world.runs / "ft0" / "trace.jsonl").exists()  # nothing was decoded


def test_a_failed_smoke_gate_on_the_merged_model_stops_before_the_test_run(
    world: World, tmp_path: Path, cfg: Any
) -> None:
    broken = MergedMock(world.gold, world.cfg, null_row=("quantity",))  # smoke check (d) fails
    with pytest.raises(predict_ft.FtError, match="smoke gate .* on the merged model"):
        ft_world_run(world, tmp_path, cfg, backend=broken)
    assert not (world.runs / "ft0" / "trace.jsonl").exists()


def test_determinism_pass_merges_again_and_needs_the_first_pass_decision(
    world: World, tmp_path: Path, cfg: Any
) -> None:
    _, _, ad, _ = ft_world_run(world, tmp_path, cfg)
    be2 = MergedMock(world.gold, world.cfg, null_header=("buyer_name",))

    def factory(d: Path, sha: str) -> Any:
        be2.model_id = f"{world.cfg.backend.repo}+lora:{sha[:12]}"
        return be2

    kw: dict[str, Any] = dict(
        cfg=world.cfg,
        adapter_dir=ad,
        folds=FOLDS,
        run_id="ft0",
        runs_root=world.runs,
        data_root=world.data,
        backend_factory=factory,
        pin_sha=SHA40,
        reachable=lambda s: True,
        out=lambda m: None,
    )
    rep = predict_ft.run_ft_determinism(decision_path=tmp_path / "meta" / "decision.json", **kw)
    assert rep["ok"] and be2.loads == 1 and rep["batch_size"] == 2
    with pytest.raises(predict_ft.FtError, match="no batch decision"):
        predict_ft.run_ft_determinism(decision_path=tmp_path / "nowhere.json", **kw)
    bad_ad = make_final_adapter(tmp_path / "bad2" / "final", cfg, code_sha="x")
    with pytest.raises(predict_ft.FtError, match="verification FAILED"):
        predict_ft.run_ft_determinism(
            **{**kw, "adapter_dir": bad_ad}, decision_path=tmp_path / "meta" / "decision.json"
        )


# -------------------- finalize + flags checks


def full_submission(
    world: World, tmp_path: Path, cfg: Any, *, v0_edit: Any = None
) -> tuple[Path, Path, Path, Path]:
    """FT run + assemble + OCR files: ``(submission, run dir, v0 dir, ocr root)``."""
    ocr_root = tmp_path / "ocr"
    write_ocr_cache(ocr_root, world.data / "test" / "images")
    v0 = fake_v0(world, tmp_path / "v0")
    if v0_edit:
        v0_edit(v0)
    ft_world_run(world, tmp_path, cfg, det=True)
    sub = tmp_path / "v2_abcdef0"
    rep = assemble(
        world,
        sub,
        run_id="ft0",
        ocr_cache=ocr_root,
        shapes_file=SHAPES,
        smoke_status_path=tmp_path / "meta" / "smoke.json",
    )
    assert rep["ok"], {k: c for k, c in rep["checks"].items() if not c["ok"]}
    stems = ocr_stage_mod.expected_stems(world.data / "test" / "images")
    timing = ocr_stage_mod.timing_summary(ocr_root, stems, "gpu", 12.0, len(stems))
    predict._write_json(sub / "ocr_timing.json", timing)
    (tmp_path / "recheck.json").write_text(
        json.dumps({"ok": True, "n_pages": 5, "n_text_identical": 5, "n_content_identical": 5})
    )
    return sub, world.runs / "ft0", v0, ocr_root


def finalize(sub: Path, run: Path, v0: Path, tmp_path: Path) -> dict[str, Any]:
    return predict_ft.finalize_v2(
        sub,
        run_dir=run,
        shapes_file=SHAPES,
        v0_dir=v0,
        recheck_path=tmp_path / "recheck.json",
        expect_pages=N_PAGES,
    )


@needs_schema
def test_finalize_accepts_a_complete_v2_submission_and_labels_it_v2(
    world: World, tmp_path: Path, cfg: Any
) -> None:
    sub, run, v0, _ = full_submission(world, tmp_path, cfg)
    rep = finalize(sub, run, v0, tmp_path)
    bad = {k: c["detail"] for k, c in rep["checks"].items() if not c["ok"]}
    assert rep["ok"] and not bad, bad
    for k in (
        "ft_adapter_verified",
        "ft_model_is_merged_adapter",
        "ft_merge_recorded",
        "ft_smoke_on_merged_model",
        "ft_batch_decision",
        "predictions_structure_vs_v0",
        "predictions_schema_strict",
        "v1_rule_switches",
        "v1_no_rule_skipped",
        "v1_shapes_sha256",
        "ocr_cache_complete",
        "ocr_determinism",
        "json_schema",
    ):
        assert rep["checks"][k]["ok"], k
    assert (sub / "test_predictions.json").is_file()
    man = json.loads((sub / "manifest.json").read_text())
    assert man["submission"] == "v2_abcdef0" and man["mode"] == "full"  # not reuse.finalize's v1_
    assert (
        man["ft"]["merge"]["n_lora_modules_merged"] == 200
        and man["files"][-1] == "review_flags.json"
    )
    assert man["structure"]["n_docs"] == len(TEST_IDS) and "ocr" in man
    schema = json.loads(SCHEMA.read_text())
    preds = json.loads((sub / "test_predictions.json").read_text())
    assert predict.validate_schema(preds, schema)["ok"]
    for d in preds.values():  # nothing but the schema's keys: the flags are not in here
        assert set(d) <= {"doc_type", "header", "line_items"}


@needs_schema
@pytest.mark.parametrize(
    ("what", "check"),
    [
        ("no_ft_run", "ft_adapter_verified"),
        ("other_model", "ft_model_is_merged_adapter"),
        ("merge_short", "ft_merge_recorded"),
        ("smoke", "ft_smoke_on_merged_model"),
        ("batch", "ft_batch_decision"),
        ("v0_structure", "predictions_structure_vs_v0"),
        ("no_v0", "predictions_structure_vs_v0"),
        ("extra_key", "predictions_schema_strict"),
    ],
)
def test_finalize_rejects_and_withholds_the_name(
    world: World, tmp_path: Path, cfg: Any, what: str, check: str
) -> None:
    def v0_edit(v0: Path) -> None:
        if what == "v0_structure":
            p = json.loads((v0 / "test_predictions.json").read_text())
            p["test_0000"]["extra"] = 1
            (v0 / "test_predictions.json").write_text(json.dumps(p))

    sub, run, v0, _ = full_submission(world, tmp_path, cfg, v0_edit=v0_edit)
    ft = json.loads((run / "ft_run.json").read_text())
    man = json.loads((sub / "manifest.json").read_text())
    if what == "no_ft_run":
        (run / "ft_run.json").unlink()
    elif what == "other_model":
        man["model"]["id"] = "Qwen/Qwen3.5-4B"
        (sub / "manifest.json").write_text(json.dumps(man))
    elif what == "merge_short":
        ft["merge"]["n_lora_modules_merged"] = 196
        (run / "ft_run.json").write_text(json.dumps(ft))
    elif what == "smoke":
        man["smoke"] = {"state": "failed"}
        (sub / "manifest.json").write_text(json.dumps(man))
    elif what == "batch":
        ft["decision"]["batch_size"] = 8
        (run / "ft_run.json").write_text(json.dumps(ft))
    elif what == "no_v0":
        (v0 / "test_predictions.json").unlink()
    elif what == "extra_key":
        p = json.loads((sub / "test_predictions.json").read_text())
        p["test_0000"]["flags"] = {"p": 1}
        (sub / "test_predictions.json").write_text(json.dumps(p))
    rep = finalize(sub, run, v0, tmp_path)
    assert not rep["ok"] and not rep["checks"][check]["ok"], rep["checks"][check]
    assert not (sub / "test_predictions.json").exists()
    assert (sub / "test_predictions.REJECTED.json").is_file()
    assert json.loads((sub / "validation_report.json").read_text())["ok"] is False


def test_structure_signature_is_value_free_and_compares_key_sets() -> None:
    a = {
        "d1": {"doc_type": "invoice", "header": {"x": "SECRETVALUE"}, "line_items": [{"q": "1"}]},
        "d2": {"doc_type": "waybill", "header": {"y": None}, "line_items": []},
    }
    b = json.loads(json.dumps(a))
    b["d1"]["header"]["x"] = "OTHER"
    sig = predict_ft.structure_signature(a)
    assert sig == predict_ft.structure_signature(b) and "SECRETVALUE" not in json.dumps(sig)
    b["d1"]["header"]["z"] = "x"
    assert predict_ft.structure_signature(b)["header_keys"] != sig["header_keys"]
    b = json.loads(json.dumps(a))
    b["d1"]["flags"] = {}
    assert predict_ft.structure_signature(b)["doc_keys"] != sig["doc_keys"]


@needs_schema
def test_check_flags_passes_on_the_real_stage_output_and_fails_on_each_defect(
    world: World, tmp_path: Path, cfg: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    sub, run, v0, ocr_root = full_submission(world, tmp_path, cfg)
    assert finalize(sub, run, v0, tmp_path)["ok"]
    cal = calibrator_for(sub, ocr_root, "ft", tmp_path, zs_sub=v0)
    monkeypatch.setattr(
        reuse,
        "evaluate_reuse",
        lambda *a, **k: reuse.ReuseDecision(True, [], {"v0_code_sha": "a" * 40}),
    )
    flags.run_stage(
        submission_dir=sub,
        calibrator=cal,
        ocr_cache=ocr_root,
        zs_v0_dir=v0,
        data_root=world.data,
        shapes_file=SHAPES,
        expect_docs=len(TEST_IDS),
        say=lambda m: None,
    )
    flags_path = sub / "review_flags.json"
    good = flags_path.read_text()
    rep = predict_ft.check_flags(sub, calibrator=cal, field_target=0.98, doc_target=0.98)
    assert rep["flags_ok"], {k: c for k, c in rep["flags_checks"].items() if not c["ok"]}
    assert rep["ok"] is True and (sub / "test_predictions.json").is_file()  # predictions untouched
    man = json.loads((sub / "manifest.json").read_text())
    assert man["review_flags"]["ok"] and man["review_flags"][
        "calibrator_sha256"
    ] == flags.sha256_file(cal)
    assert man["review_flags"]["thresholds"]["field_tau"] and man["review_flags"]["arm"] == "ft"

    def check(mutate: Any) -> dict[str, Any]:
        doc = json.loads(good)
        mutate(doc)
        flags_path.write_text(json.dumps(doc))
        return predict_ft.check_flags(sub, calibrator=cal, field_target=0.98, doc_target=0.98)

    leaked = json.loads((sub / "test_predictions.json").read_text())["test_0000"]["header"]
    value = next(v for v in leaked.values() if isinstance(v, str) and len(v) >= 4)
    r = check(lambda d: d["docs"]["test_0000"].update(note=value))  # a value under docs
    assert not r["flags_checks"]["flags_no_values"]["ok"] and not r["flags_ok"]
    r = check(lambda d: d.update(oops=value))  # a value anywhere else in the file
    assert not r["flags_checks"]["flags_no_values"]["ok"]
    r = check(lambda d: d["docs"].pop("test_0003"))
    assert not r["flags_checks"]["flags_ids"]["ok"]
    r = check(lambda d: d["docs"].update(test_9999=d["docs"]["test_0000"]))
    assert not r["flags_checks"]["flags_ids"]["ok"]
    r = check(lambda d: d["thresholds"].pop("field_tau"))
    assert not r["flags_checks"]["flags_thresholds_recorded"]["ok"]
    r = check(lambda d: d["thresholds"].update(nested_estimates=None))
    assert not r["flags_checks"]["flags_thresholds_recorded"]["ok"]
    r = check(lambda d: d["calibrator"].update(sha256="0" * 64))
    assert not r["flags_checks"]["flags_calibrator_sha256"]["ok"]
    r = check(lambda d: d["inputs"].update(test_predictions_sha256="0" * 64))
    assert not r["flags_checks"]["flags_match_the_predictions"]["ok"]
    assert r["ok"] is True  # the predictions' own verdict never changes because of the flags
    flags_path.unlink()
    r = predict_ft.check_flags(sub, calibrator=cal, field_target=0.98, doc_target=0.98)
    assert not r["flags_checks"]["flags_file_present"]["ok"]
    assert predict_ft.main(["check-flags", "--out-dir", str(sub), "--calibrator", str(cal)]) == 1


# -------------------- OCR record and estimate


def test_adopt_ocr_copies_the_records_once_and_refuses_an_unproven_folder(tmp_path: Path) -> None:
    v1 = tmp_path / "v1_x"
    v1.mkdir()
    timing = {"pages": 280, "device": "gpu", "seconds_total": 3.0}
    man = {"ocr": {"timing": timing, "recheck": {"ok": True, "n_text_identical": 5}}}
    (v1 / "manifest.json").write_text(json.dumps(man))
    t_out, r_out = tmp_path / "sub" / "ocr_timing.json", tmp_path / "meta" / "ocr_recheck.json"
    res = predict_ft.adopt_ocr(v1, t_out, r_out)
    assert res["adopted"] == ["ocr_timing.json", "ocr_recheck.json"]
    assert json.loads(t_out.read_text())["adopted_from"] == "v1_x"
    assert json.loads(r_out.read_text())["ok"] is True
    t_out.write_text(json.dumps({"mine": 1}))  # never overwritten
    assert predict_ft.adopt_ocr(v1, t_out, r_out)["adopted"] == []
    assert json.loads(t_out.read_text()) == {"mine": 1}
    for bad, msg in (
        ({"ocr": {"timing": {"pages": 3}, "recheck": {"ok": True}}}, "OCR timing"),
        ({"ocr": {"timing": timing, "recheck": {"ok": False}}}, "re-check"),
        ({}, "OCR timing"),
    ):
        (v1 / "manifest.json").write_text(json.dumps(bad))
        with pytest.raises(predict_ft.FtError, match=msg):
            predict_ft.adopt_ocr(v1, tmp_path / "t2.json", tmp_path / "r2.json")
    with pytest.raises(predict_ft.FtError, match="no manifest"):
        predict_ft.adopt_ocr(tmp_path / "none", tmp_path / "t3.json", tmp_path / "r3.json")
    assert (
        predict_ft.main(
            [
                "adopt-ocr",
                "--from-dir",
                str(tmp_path / "none"),
                "--timing-out",
                str(tmp_path / "t4"),
                "--recheck-out",
                str(tmp_path / "r4"),
            ]
        )
        == 1
    )


def test_the_estimate_is_labelled_and_includes_every_gpu_stage(
    world: World, cfg: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = predict_ft.run_estimate(
        cfg=world.cfg, data_root=world.data, batch_size=8, smoke_docs=SMOKE_IDS
    )
    out = capsys.readouterr().out
    assert "ESTIMATE (UNVERIFIED)" in out and "ASSUMED" in out and "CU@low" in out
    assert rows["n_pages"] == N_PAGES and rows["smoke_pages"] > 0 and rows["det_pages"] > 0
    r = rows["rows"][0]
    parts = r["load_s"] + r["merge_s"] + r["guard_s"] + r["smoke_s"] + r["run_s"] + r["det_s"]
    assert abs(r["hours"] * 3600 - parts) < 1e-6 and r["guard_s"] > 0 and len(rows["rows"]) == 3
    one = predict_ft.run_estimate(
        cfg=world.cfg, data_root=world.data, batch_size=1, smoke_docs=SMOKE_IDS, out=lambda m: None
    )
    assert len(one["rows"]) == 1 and one["rows"][0]["guard_s"] == 0.0  # no guard at batch 1


def test_cli_stages_exist_and_the_cli_file_is_untouched() -> None:
    pp = predict_ft.build_parser()
    for stage in (
        "estimate",
        "verify",
        "infer",
        "determinism",
        "finalize",
        "check-flags",
        "adopt-ocr",
    ):
        assert _subparser(pp, stage) is not None
    assert "predict_ft" not in (ROOT / "src" / "shipdoc" / "cli.py").read_text()  # no cli change
    assert c2.TEST_PREFIX == "test_" and flags.LORA_MARK == "+lora:"
