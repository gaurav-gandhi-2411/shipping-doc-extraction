"""Tests for shipdoc.paths and the no-hardcoded-paths rule for src/."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import pytest

from shipdoc import paths

SRC = Path(paths.__file__).resolve().parent


def test_parse_dotenv_handles_comments_quotes_export() -> None:
    text = "# c\n\nA=1\nB = \"x y\"\nexport C='z'\nbad line\n"
    assert paths.parse_dotenv(text) == {"A": "1", "B": "x y", "C": "z"}


def test_env_beats_dotenv_beats_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHIPDOC_PROFILE", "local")
    monkeypatch.delenv("SHIPDOC_RUNS_DIR", raising=False)
    monkeypatch.setattr(paths, "_dotenv", lambda: {"SHIPDOC_RUNS_DIR": "/from/dotenv"})
    assert paths.get("SHIPDOC_RUNS_DIR") == "/from/dotenv"
    monkeypatch.setenv("SHIPDOC_RUNS_DIR", "/from/env")
    assert paths.get("SHIPDOC_RUNS_DIR") == "/from/env"
    monkeypatch.setattr(paths, "_dotenv", lambda: {})
    monkeypatch.delenv("SHIPDOC_RUNS_DIR")
    assert paths.get("SHIPDOC_RUNS_DIR") == "runs"
    assert paths.get_path("SHIPDOC_RUNS_DIR") == paths.REPO_ROOT / "runs"


def test_profiles_and_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths, "_dotenv", lambda: {})
    monkeypatch.delenv("SHIPDOC_OCR_CACHE", raising=False)
    monkeypatch.setenv("SHIPDOC_PROFILE", "colab")
    assert paths.detect_profile() == "colab"
    assert paths.get("SHIPDOC_OCR_CACHE") == "/content/ocr_cache"
    assert paths.get("SHIPDOC_RUNS_DIR", "colab").startswith("/content/drive/MyDrive/shipdoc-")
    monkeypatch.setenv("SHIPDOC_PROFILE", "nope")
    with pytest.raises(KeyError):
        paths.get("SHIPDOC_OCR_CACHE")


def test_every_key_defined_for_every_profile() -> None:
    for profile in ("local", "colab"):
        defaults = paths._profile_defaults(profile)
        assert set(paths.PATH_KEYS) <= set(defaults), profile


def test_apply_env_setdefault_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SHIPDOC_PROFILE", "local")
    monkeypatch.setattr(paths, "_dotenv", lambda: {"SHIPDOC_TMP_DIR": str(tmp_path / "t")})
    monkeypatch.setenv("HF_HOME", "/explicit/hf")
    for k in ("PADDLE_PDX_CACHE_HOME", "TMP", "TEMP", "TMPDIR"):
        monkeypatch.delenv(k, raising=False)
    saved = tempfile.tempdir
    try:
        eff = paths.apply_env()
        assert eff["HF_HOME"] == "/explicit/hf"
        assert eff["TMP"] == eff["TEMP"] == eff["TMPDIR"] == str(tmp_path / "t")
        assert (tmp_path / "t").is_dir()
    finally:
        tempfile.tempdir = saved


def test_src_has_no_hardcoded_storage_paths() -> None:
    drive = re.compile(r"""["'\s(][A-Za-z]:[\\/]""")
    literal = re.compile(r"cache/ocr|runs/|cache\\ocr|runs\\")
    bad: list[str] = []
    for f in sorted(SRC.rglob("*.py")):
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if drive.search(line) or literal.search(line):
                bad.append(f"{f.name}:{n}: {line.strip()}")
    assert not bad, "hardcoded paths in src/:\n" + "\n".join(bad)
