from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

from shipdoc import ocr

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ["colab_ocr_setup.sh", "colab_ocr_run.sh"]


def test_device_params() -> None:
    cpu = ocr.device_params("cpu", cpu_threads=4)
    assert cpu == {"device": "cpu", "enable_mkldnn": True, "cpu_threads": 4}
    gpu = ocr.device_params("gpu")
    assert gpu["device"] == "gpu:0" and gpu["enable_mkldnn"] is False
    with pytest.raises(ValueError, match="unsupported OCR device"):
        ocr.device_params("tpu")


@pytest.mark.skipif(importlib.util.find_spec("paddleocr") is None, reason="paddleocr absent")
def test_engine_construction_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SHIPDOC_OCR_DEVICE", raising=False)
    engine = ocr.PaddleOcrEngine(variant="mobile", cpu_threads=1)  # builds models; no inference
    assert engine.params["device"] == "cpu"
    assert engine.params["text_detection_model_name"] == "PP-OCRv5_mobile_det"


def _bash_works() -> bool:
    """True when bash can run a command (Windows' System32 bash.exe fails without a WSL distro)."""
    if shutil.which("bash") is None:
        return False
    try:
        res = subprocess.run(["bash", "-c", "echo ok"], capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return res.stdout.strip() == b"ok"


@pytest.mark.skipif(not _bash_works(), reason="no working bash")
@pytest.mark.parametrize("name", SCRIPTS)
def test_shell_scripts_parse(name: str) -> None:
    res = subprocess.run(
        ["bash", "-n", f"scripts/{name}"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert res.returncode == 0, res.stderr


def test_setup_script_pins_and_fallback() -> None:
    text = (ROOT / "scripts" / "colab_ocr_setup.sh").read_text(encoding="utf-8")
    assert 'PADDLE_VERSION="3.0.0"' in text
    assert "https://www.paddlepaddle.org.cn/packages/stable" in text
    assert "install_cpu" in text and "FORCE_CPU" in text
    assert "--only-group ocr" in text  # pins come from uv.lock, not duplicated here
