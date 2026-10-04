#!/usr/bin/env bash
# Build an ISOLATED PaddleOCR environment for Colab (Linux) at $OCR_VENV (default /content/ocr-venv).
#
# Why a separate venv: Colab ships its own torch/CUDA stack, and the VLM environment adds a pinned
# torch + transformers + bitsandbytes on top of it. paddlepaddle-gpu bundles its own CUDA/cuDNN
# runtime wheels (nvidia-*-cu12) with exact pins, and the langchain 0.3.x / stringzilla / protobuf
# pins needed by paddlex 3.0.3 differ from what the VLM group resolves. Mixing them in one
# environment means one side's pins silently win and break the other. The OCR venv shares nothing
# with the VLM venv; the repo is put on PYTHONPATH instead of being installed (see
# scripts/colab_ocr_run.sh).
#
# Pins are reused from the repo's uv.lock via `uv export --only-group ocr` (the same versions the
# local CPU cache was produced with), minus paddlepaddle, which is installed separately:
#   GPU: paddlepaddle-gpu==3.0.0 from Paddle's official index for the Colab CUDA version
#        (https://www.paddlepaddle.org.cn/packages/stable/cu126/ or .../cu118/, per Paddle's 3.0
#        install docs: docs/install/pip/linux-pip_en.md in github.com/PaddlePaddle/docs, branch
#        release/3.0), only if the wheel is listed on that index.
#   CPU: paddlepaddle==3.0.0 from PyPI. Used when there is no GPU, no listed wheel, or the GPU
#        install / smoke test fails (the venv is rebuilt from scratch, then).
#
# Usage: bash scripts/colab_ocr_setup.sh        (env: OCR_VENV, FORCE_CPU=1, PY_VERSION)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OCR_VENV="${OCR_VENV:-/content/ocr-venv}"
PY_VERSION="${PY_VERSION:-3.11}"
PADDLE_VERSION="3.0.0"
PADDLE_INDEX_BASE="https://www.paddlepaddle.org.cn/packages/stable"
PY="$OCR_VENV/bin/python"
REQS="${TMPDIR:-/tmp}/ocr-requirements.txt"

log() { echo "[colab_ocr_setup] $*"; }

command -v uv >/dev/null 2>&1 || { log "installing uv"; python3 -m pip install -q uv; }

# Pick the Paddle CUDA index from the CUDA version nvidia-smi reports (the driver's max CUDA).
# Prints the index suffix (cu126 or cu118), or nothing.
pick_cuda_tag() {
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  local ver major minor
  ver="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9]*\.[0-9]*\).*/\1/p' | head -n1)"
  [ -n "$ver" ] || return 0
  major="${ver%%.*}"
  minor="${ver##*.}"
  if [ "$major" -gt 12 ] || { [ "$major" -eq 12 ] && [ "$minor" -ge 6 ]; }; then
    echo cu126
  elif [ "$major" -eq 12 ] || { [ "$major" -eq 11 ] && [ "$minor" -ge 8 ]; }; then
    echo cu118
  fi
}

# True when the index lists the cp311 linux wheel of paddlepaddle-gpu==3.0.0 (checked, not guessed).
wheel_listed() {
  curl -fsSL "$1paddlepaddle-gpu/" 2>/dev/null \
    | grep -Eq "paddlepaddle_gpu-${PADDLE_VERSION}-cp311-cp311-(manylinux1_|linux_)x86_64\.whl"
}

build_base() {
  rm -rf "$OCR_VENV"
  uv venv --python "$PY_VERSION" "$OCR_VENV"
  (cd "$REPO" && uv export --frozen --only-group ocr --no-hashes --no-emit-project) \
    | grep -Ev '^paddlepaddle(-gpu)?==' > "$REQS"
  uv pip install --python "$PY" -r "$REQS"
}

install_cpu() {
  uv pip install --python "$PY" "paddlepaddle==${PADDLE_VERSION}"
  echo cpu > "$OCR_VENV/.paddle_flavor"
}

try_gpu() {
  local tag idx
  tag="$(pick_cuda_tag)"
  [ -n "$tag" ] || { log "no usable GPU/CUDA version from nvidia-smi"; return 1; }
  idx="${PADDLE_INDEX_BASE}/${tag}/"
  wheel_listed "$idx" || { log "paddlepaddle-gpu==${PADDLE_VERSION} not listed at $idx"; return 1; }
  log "installing paddlepaddle-gpu==${PADDLE_VERSION} from $idx"
  # Paddle's index mirrors many packages; PyPI is added and best-match is used so the pinned
  # versions above are not displaced by older mirror copies.
  uv pip install --python "$PY" "paddlepaddle-gpu==${PADDLE_VERSION}" \
    --default-index "$idx" --index https://pypi.org/simple --index-strategy unsafe-best-match
  # Smoke test: a real kernel on the GPU, not just an import.
  "$PY" -c '
import paddle
assert paddle.device.is_compiled_with_cuda(), "paddle built without CUDA"
paddle.set_device("gpu:0")
x = paddle.ones([4, 4])
assert float((x @ x).sum()) == 64.0
print("paddle", paddle.__version__, "gpu smoke ok")
'
  echo "gpu-${tag}" > "$OCR_VENV/.paddle_flavor"
}

build_base
flavor=cpu
if [ "${FORCE_CPU:-0}" != "1" ] && try_gpu; then
  flavor="$(cat "$OCR_VENV/.paddle_flavor")"
else
  if [ "${FORCE_CPU:-0}" != "1" ]; then
    log "GPU path unavailable or failed; rebuilding the venv with CPU paddlepaddle"
    build_base
  fi
  install_cpu
fi

# Import smoke for the whole stack (no inference, no model download).
"$PY" -c '
import paddle, paddleocr, paddlex
print("paddlepaddle", paddle.__version__, "| paddleocr", paddleocr.__version__,
      "| paddlex", paddlex.__version__)
'
log "done: flavor=$flavor venv=$OCR_VENV"
log "run OCR with: bash scripts/colab_ocr_run.sh --splits test"
