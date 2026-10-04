"""notebooks/03n_finetune_native.ipynb: structure, native isolation and the cells that run locally.

Nothing here talks to Colab, Drive, a GPU or the network (cells are exec'd in a fake namespace).
"""

from __future__ import annotations

import ast
import importlib.util
import io
import json
import re
import subprocess
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "03n_finetune_native.ipynb"
spec = importlib.util.spec_from_file_location(
    "colab_build_finetune_native", ROOT / "scripts" / "colab_build_finetune_native.py"
)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

NB = json.loads(NB_PATH.read_text(encoding="utf-8"))
SHA = "0123456789abcdef0123456789abcdef01234567"
# Cell indices (see build_native): title, params, account, mount, secrets, clone, unzip, estimate,
# install, prepare, smoke, train, rescheck, banner.
PARAMS, ESTIMATE, INSTALL, PREPARE, SMOKE, TRAIN, RESCHECK, BANNER = 1, 7, 8, 9, 10, 11, 12, 13
needs_data = pytest.mark.skipif(
    not (ROOT / "data" / "train" / "labels").is_dir(), reason="data/ absent"
)


def src(i: int) -> str:
    return "".join(NB["cells"][i]["source"])


def params_ns(**override: str) -> dict[str, Any]:
    ns: dict[str, Any] = {}
    text = src(PARAMS).replace(builder.PINNED_SHA, override.pop("sha", SHA))
    for k, v in override.items():
        text = text.replace(f'{k} = "smoke"', f'{k} = "{v}"')
    exec(text, ns)
    return ns


def run_cell(i: int, ns: dict[str, Any]) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        exec(src(i), ns)
    return buf.getvalue()


def test_committed_notebook_matches_builder_and_has_14_cells() -> None:
    assert NB_PATH.read_text(encoding="utf-8") == builder.render_native()
    assert len(NB["cells"]) == 14


def test_outputs_cleared_cells_compile_no_magics_and_at_most_100_columns() -> None:
    for i, c in enumerate(NB["cells"]):
        text = "".join(c["source"])
        for line in text.splitlines():
            assert len(line) <= 100, (i, len(line), line[:50])
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None
            ast.parse(text)
            assert not re.search(r"^\s*[!%]", text, re.M), i  # no shell / IPython magics
    assert NB["metadata"]["colab"]["gpuType"] == "L4"


def test_pin_is_a_placeholder_that_refuses_to_run_or_a_full_sha() -> None:
    pin = builder.PINNED_SHA
    assert f'PINNED_SHA = "{pin}"' in src(PARAMS)
    if pin == "FILL_PINNED_SHA":
        with pytest.raises(ValueError, match="PINNED_SHA"):
            exec(src(PARAMS), {})
    else:
        assert re.fullmatch(r"[0-9a-f]{40}", pin)
    ns = params_ns()
    assert re.fullmatch(r"[0-9a-f]{40}", ns["PINNED_SHA"]) and ns["SHA7"] == SHA[:7]


def test_parameters_default_to_the_smoke_stage_and_the_native_config_and_run_ids() -> None:
    ns = params_ns()
    assert ns["STAGE"] == "smoke" and ns["STAGES"] == list(builder.STAGES)
    assert ns["CONFIG"] == "configs/finetune_qwen35_4b_native.yaml"
    assert ns["NATIVE_MAX_PIXELS"] == 2_196_480
    assert ns["RUN_BASE"] == f"ft_native_smoke_{SHA[:7]}"
    assert params_ns(STAGE="fold2")["RUN_BASE"] == f"ft_native_fold2_{SHA[:7]}"
    assert ns["USE_WANDB"] is False and params_ns(STAGE="fold0")["USE_WANDB"] is True
    with pytest.raises(ValueError, match="STAGE must be one of"):
        params_ns(STAGE="fold9")


def test_run_ids_never_start_like_a_1260_run() -> None:
    install = src(INSTALL)
    assert 'SMOKE_RUN_ID = f"ft_native_smoke_{SHA7}_{PRECISION}"' in install
    assert 'RUN_ID = f"{RUN_BASE}_{PRECISION}"' in install
    for stage in builder.STAGES:
        rid = params_ns(STAGE=stage)["RUN_BASE"]
        assert rid.startswith("ft_native_") and not re.match(r"ft_(smoke|fold\d|final)", rid)


def test_stage_order_reminder_is_the_new_plan_and_is_printed_for_the_later_stages() -> None:
    for stage, expected in (("smoke", False), ("fold0", False), ("fold1", True),
                            ("fold2", True), ("final", True)):  # fmt: skip
        buf = io.StringIO()
        with redirect_stdout(buf):
            ns = params_ns(STAGE=stage)
        assert ("REMINDER" in buf.getvalue()) is expected, stage
        text = ns["AFTER_FOLD0"]
        assert "fold0 (native) runs first" in text and "never mix in a 1260 adapter" in text
        assert "no regression" not in text  # the 03 wording is the old plan
    assert "AFTER_FOLD0" in src(TRAIN) and "AFTER_FOLD0" in src(ESTIMATE)
    assert "AFTER_FOLD0" in src(BANNER)
    assert "no code gate" in src(0) and "spec.md section 11" in src(0)


def test_cell_order_estimate_before_install_smoke_before_stage_then_resolution_check() -> None:
    assert ESTIMATE < INSTALL < PREPARE < SMOKE < TRAIN < RESCHECK < BANNER
    assert "ESTIMATE of VRAM" in src(ESTIMATE) and "MANDATORY smoke stage" in src(SMOKE)
    assert "The chosen stage" in src(TRAIN) and "Resolution record" in src(RESCHECK)
    for i in range(ESTIMATE):
        if NB["cells"][i]["cell_type"] != "code":
            continue
        text = src(i)
        assert "uv sync" not in text and "shipdoc.train" not in text
    launching = [i for i in range(len(NB["cells"])) if '"shipdoc.train"' in src(i)]
    assert launching == [SMOKE, TRAIN]
    assert "if not SMOKE_PASSED" in src(TRAIN) and "SMOKE GATE FAILED" in src(SMOKE)
    assert "Do not edit thresholds" in src(SMOKE)


def test_smoke_thresholds_are_not_touched_by_the_builder() -> None:
    text = builder.ft.FT_SMOKE
    assert src(SMOKE) == text  # reused verbatim from notebook 03


def test_unreplaced_text_is_asserted_so_a_change_in_03_breaks_the_build() -> None:
    with pytest.raises(AssertionError, match="colab_build_finetune changed"):
        builder._swap("abc", "xyz", "q")


def test_no_secret_is_printed() -> None:
    joined = "\n".join(src(i) for i, c in enumerate(NB["cells"]) if c["cell_type"] == "code")
    for name in ("GH_TOKEN", "WANDB_API_KEY", "HF_TOKEN", "_basic"):
        for line in joined.splitlines():
            if "print(" in line and name in line:
                assert f"{{{name}" not in line and f", {name}" not in line, line


# --------------------------------------------------------------------------------------------
# Executing cells locally
# --------------------------------------------------------------------------------------------


def fake_gpu(monkeypatch: pytest.MonkeyPatch, line: str | None) -> None:
    def run(cmd: list[str], **kw: Any) -> SimpleNamespace:
        assert cmd[0] == "nvidia-smi"
        return SimpleNamespace(stdout=line or "", returncode=0 if line else 1)

    monkeypatch.setattr(subprocess, "run", run)


@needs_data
@pytest.mark.parametrize(
    ("gpu_line", "precision", "warns"),
    [("NVIDIA L4, 8.9", "bf16", False), ("Tesla T4, 7.5", "fp16", True)],
)
def test_estimate_cell_prints_native_vram_hours_and_cu(
    monkeypatch: pytest.MonkeyPatch, gpu_line: str, precision: str, warns: bool
) -> None:
    fake_gpu(monkeypatch, gpu_line)
    ns = {**params_ns(), "REPO": ROOT}
    out = run_cell(ESTIMATE, ns)
    assert ns["PREVIEW_PRECISION"] == precision
    assert "max_pixels 2196480 (2145 visual tokens per page)" in out
    assert "NATIVE RESOLUTION ESTIMATE / UNVERIFIED" in out and "MEASURED at 1260" in out
    for stage in builder.STAGES:
        assert re.search(rf"^{stage}\s", out, re.M), stage
    assert ("expect the smoke to fail" in out) is warns
    assert "4 parallel tabs" in out and "1.19" in out and "1.58" in out


def test_estimate_cell_refuses_a_config_that_is_not_native(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True)
    (repo / "scripts").mkdir()
    for name in ("finetune_estimate.py", "finetune_native_estimate.py"):
        (repo / "scripts" / name).write_text((ROOT / "scripts" / name).read_text(encoding="utf-8"))
    old = (ROOT / "configs" / "finetune_qwen35_4b.yaml").read_text(encoding="utf-8")
    (repo / "configs" / "finetune_qwen35_4b_native.yaml").write_text(old)  # the 1260 values
    with pytest.raises(AssertionError, match="this notebook is the native one"):
        run_cell(ESTIMATE, {**params_ns(), "REPO": repo})


def test_resolution_check_cell_passes_native_and_refuses_1260_or_missing(tmp_path: Path) -> None:
    run = tmp_path / "run"
    (run / "final").mkdir(parents=True)

    def ns(stage: str) -> dict[str, Any]:
        return {**params_ns(STAGE=stage), "REPO": ROOT, "RUN_DIR": run, "json": json}

    assert "no adapter manifest" in run_cell(RESCHECK, ns("smoke"))
    with pytest.raises(RuntimeError, match="manifest.json is missing"):
        run_cell(RESCHECK, ns("fold0"))
    manifest = run / "final" / "manifest.json"
    manifest.write_text(json.dumps({"inference_keys": {"max_pixels": 2_196_480}}))
    assert "RESOLUTION OK" in run_cell(RESCHECK, ns("fold0"))
    for bad in ({"inference_keys": {"max_pixels": 1_310_720}}, {"inference_keys": {}}):
        manifest.write_text(json.dumps(bad))
        with pytest.raises(RuntimeError, match="does not record the native resolution"):
            run_cell(RESCHECK, ns("fold0"))


def test_notebooks_readme_documents_03n() -> None:
    text = (ROOT / "notebooks" / "DEVELOPMENT_NOTES.md").read_text(encoding="utf-8")
    assert "## 03n_finetune_native" in text and "scripts/colab_build_finetune_native.py" in text
    assert "FILL_PINNED_SHA" in text and "ft_native_<stage>_<sha7>_<precision>" in text
    assert "ESTIMATE, UNVERIFIED" in text and "What GG does when it is time to pin and run" in text
