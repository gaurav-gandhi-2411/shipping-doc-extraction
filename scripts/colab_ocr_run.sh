#!/usr/bin/env bash
# Run `python -m shipdoc ocr ...` inside the isolated OCR venv built by colab_ocr_setup.sh.
# The repo is on PYTHONPATH (not installed) so the venv stays free of the VLM stack.
# Usage: bash scripts/colab_ocr_run.sh --splits test
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OCR_VENV="${OCR_VENV:-/content/ocr-venv}"
if [ ! -x "$OCR_VENV/bin/python" ]; then
  echo "missing $OCR_VENV; run scripts/colab_ocr_setup.sh first" >&2
  exit 2
fi

export SHIPDOC_PROFILE="${SHIPDOC_PROFILE:-colab}"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
case "$(cat "$OCR_VENV/.paddle_flavor" 2>/dev/null || echo cpu)" in
  gpu-*) export SHIPDOC_OCR_DEVICE=gpu ;;
  *) export SHIPDOC_OCR_DEVICE=cpu ;;
esac
cd "$REPO"
exec "$OCR_VENV/bin/python" -m shipdoc ocr "$@"
