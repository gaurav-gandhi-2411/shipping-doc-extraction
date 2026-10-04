#!/usr/bin/env bash
# OCR of the 280 TEST pages for notebook 04b, in the isolated Paddle venv ($OCR_VENV, default
# /content/ocr-venv; never the venv that decodes). Thin wrapper around the existing scripts:
#   setup                      build the venv unless it is already usable (GPU wheel with a CPU
#                              fallback inside colab_ocr_setup.sh; FORCE_CPU=1 forces CPU)
#   run <data_root> <cache>    python -m shipdoc ocr --splits test (resumable: cached pages are
#                              skipped by shipdoc.ocr.run_ocr) via colab_ocr_run.sh
#   module <args...>           python -m shipdoc.ocr_stage <args...> inside the OCR venv (the
#                              re-OCR determinism check needs Paddle)
# OCR text is never printed: the CLI prints page names, line counts and seconds.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OCR_VENV="${OCR_VENV:-/content/ocr-venv}"
cmd="${1:-}"
[ -n "$cmd" ] || { echo "usage: $0 setup | run <data_root> <cache_root> | module <args...>" >&2; exit 2; }
shift

usable() {
  [ -x "$OCR_VENV/bin/python" ] && [ -f "$OCR_VENV/.paddle_flavor" ] \
    && "$OCR_VENV/bin/python" -c 'import paddle, paddleocr, paddlex' >/dev/null 2>&1
}

case "$cmd" in
  setup)
    if usable; then
      echo "[colab_ocr_test] reusing $OCR_VENV (flavor $(cat "$OCR_VENV/.paddle_flavor"))"
    else
      bash "$REPO/scripts/colab_ocr_setup.sh"
    fi
    ;;
  run)
    [ "$#" -eq 2 ] || { echo "run needs <data_root> <cache_root>" >&2; exit 2; }
    exec bash "$REPO/scripts/colab_ocr_run.sh" --splits test --data-root "$1" --cache-root "$2"
    ;;
  module)
    [ -x "$OCR_VENV/bin/python" ] || { echo "missing $OCR_VENV; run: $0 setup" >&2; exit 2; }
    export SHIPDOC_PROFILE="${SHIPDOC_PROFILE:-colab}"
    export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
    case "$(cat "$OCR_VENV/.paddle_flavor" 2>/dev/null || echo cpu)" in
      gpu-*) export SHIPDOC_OCR_DEVICE=gpu ;;
      *) export SHIPDOC_OCR_DEVICE=cpu ;;
    esac
    cd "$REPO"
    exec "$OCR_VENV/bin/python" -m shipdoc.ocr_stage "$@"
    ;;
  *)
    echo "unknown command $cmd" >&2
    exit 2
    ;;
esac
