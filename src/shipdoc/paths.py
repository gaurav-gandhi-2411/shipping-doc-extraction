"""Single source of truth for every location-dependent path.

Resolution order for each key: real environment variable > repo-root ``.env`` (gitignored) >
profile defaults in ``configs/paths.yaml``. The profile is ``$SHIPDOC_PROFILE`` if set, else
``colab`` when running on Colab, else ``local``. Relative values resolve against the repo root.

`apply_env` exports the cache locations (HF, PaddleX, temp) into *this process only*, via
``setdefault`` so an explicit variable always wins. It never touches user/machine-global settings.
"""

from __future__ import annotations

import os
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "paths.yaml"
DOTENV_PATH = REPO_ROOT / ".env"
PROFILE_ENV = "SHIPDOC_PROFILE"

# PaddleX reads this at import time (paddlex/utils/cache.py:29), so apply_env must run before
# paddle/paddleocr is imported.
PADDLE_HOME_ENV = "PADDLE_PDX_CACHE_HOME"

PATH_KEYS = (
    "SHIPDOC_DATA_DIR",
    "SHIPDOC_ASSIGNMENT_DIR",
    "SHIPDOC_RUNS_DIR",
    "SHIPDOC_SUBMISSIONS_DIR",
    "SHIPDOC_OCR_CACHE",
    "SHIPDOC_CKPT_DIR",
    "SHIPDOC_TMP_DIR",
    "HF_HOME",
    "UV_CACHE_DIR",
    PADDLE_HOME_ENV,
)


def parse_dotenv(text: str) -> dict[str, str]:
    """Parse ``KEY=VALUE`` lines; ``#`` comments and blank lines ignored, quotes stripped."""
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[key.strip().removeprefix("export ").strip()] = value
    return out


def detect_profile() -> str:
    """``$SHIPDOC_PROFILE``, else ``colab`` when on Colab, else ``local``."""
    forced = os.environ.get(PROFILE_ENV)
    if forced:
        return forced
    if "google.colab" in sys.modules:
        return "colab"
    if sys.platform.startswith("linux"):
        if Path("/content").is_dir():
            return "colab"
        try:
            import importlib.util

            if importlib.util.find_spec("google.colab") is not None:
                return "colab"
        except (ImportError, ValueError):
            pass
    return "local"


@lru_cache(maxsize=1)
def _dotenv() -> dict[str, str]:
    return parse_dotenv(DOTENV_PATH.read_text(encoding="utf-8")) if DOTENV_PATH.is_file() else {}


def _profile_defaults(profile: str) -> dict[str, str]:
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if profile not in cfg:
        raise KeyError(f"unknown profile {profile!r} in {CONFIG_PATH}; have {sorted(cfg)}")
    return {k: str(v) for k, v in cfg[profile].items()}


def get(key: str, profile: str | None = None) -> str:
    """Raw string value of `key`: environment > .env > profile default."""
    if key in os.environ:
        return os.environ[key]
    env_file = _dotenv()
    if key in env_file:
        return env_file[key]
    defaults = _profile_defaults(profile or detect_profile())
    if key not in defaults:
        raise KeyError(f"{key} not set in environment, .env or configs/paths.yaml")
    return defaults[key]


def get_path(key: str, profile: str | None = None) -> Path:
    """`get` as an absolute Path; relative values are anchored at the repo root."""
    p = Path(get(key, profile))
    return p if p.is_absolute() else REPO_ROOT / p


def data_dir() -> Path:
    """Root of the split folders (``<split>/images``, ``<split>/labels``)."""
    return get_path("SHIPDOC_DATA_DIR")


def assignment_dir() -> Path:
    """Folder holding the brief, schema and official scorer."""
    return get_path("SHIPDOC_ASSIGNMENT_DIR")


def runs_dir() -> Path:
    """Where run outputs (predictions, reports) are written."""
    return get_path("SHIPDOC_RUNS_DIR")


def submissions_dir() -> Path:
    """Where test-prediction submissions are assembled (gitignored: never enters the repo)."""
    return get_path("SHIPDOC_SUBMISSIONS_DIR")


def ocr_cache_dir() -> Path:
    """OCR cache root: the folder that contains ``<engine>/<split>/<page_stem>.json``."""
    return get_path("SHIPDOC_OCR_CACHE")


def ckpt_dir() -> Path:
    """Where training checkpoints are written."""
    return get_path("SHIPDOC_CKPT_DIR")


def tmp_dir() -> Path:
    """Scratch space for this project."""
    return get_path("SHIPDOC_TMP_DIR")


def scorer_path() -> Path:
    """Default path of the official scorer."""
    return assignment_dir() / "score.py"


def apply_env() -> dict[str, str]:
    """Export HF_HOME, the PaddleX model home and TMP/TEMP/TMPDIR for this process only.

    Uses ``setdefault`` so a variable already set in the real environment wins. Must be called
    before importing paddle or transformers. Returns the values now in effect.
    """
    tmp = str(tmp_dir())
    wanted = {
        "HF_HOME": str(get_path("HF_HOME")),
        PADDLE_HOME_ENV: str(get_path(PADDLE_HOME_ENV)),
        "TMP": tmp,
        "TEMP": tmp,
        "TMPDIR": tmp,
    }
    effective = {k: os.environ.setdefault(k, v) for k, v in wanted.items()}
    # tempfile caches its choice on first use; drop it so the new TMP/TEMP/TMPDIR is honoured.
    tempfile.tempdir = None
    Path(effective["TMP"]).mkdir(parents=True, exist_ok=True)
    return effective
