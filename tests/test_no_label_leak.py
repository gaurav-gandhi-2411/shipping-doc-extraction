"""Regression guard: real gold label values (confidential) must never appear in tracked files.

Failures report path:line and the field kind only -- never the value itself.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
# Dates are excluded: ISO dates collide with lockfile / upload timestamps.
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# >=8 chars keeps the false-positive rate low; 6-7 is reported by the variant below.
MIN_LEN = 8
SKIP_FILES = {"uv.lock"}


def collect_values(min_len: int, max_len: int | None = None) -> dict[str, set[str]]:
    """Map distinct train+dev gold value -> set of field kinds (header.x / row.x)."""
    out: dict[str, set[str]] = {}
    for split in ("train", "dev"):
        for f in sorted((DATA / split / "labels").glob("*.json")):
            d = json.loads(f.read_text(encoding="utf-8"))
            pairs = [(f"header.{k}", v) for k, v in d.get("header", {}).items()]
            for row in d.get("line_items", []):
                pairs += [(f"row.{k}", v) for k, v in row.items()]
            for kind, v in pairs:
                if not isinstance(v, str):
                    continue
                v = v.strip()
                if len(v) < min_len or (max_len is not None and len(v) > max_len):
                    continue
                if ISO_DATE.match(v):
                    continue
                out.setdefault(v, set()).add(kind)
    return out


def tracked_text_files() -> list[Path]:
    """Tracked, non-binary files excluding lockfiles."""
    names = (
        subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True)
        .stdout.decode("utf-8")
        .split("\0")
    )
    files = []
    for n in names:
        p = ROOT / n
        if n and p.name not in SKIP_FILES and p.is_file():
            files.append(p)
    return files


def scan(
    values: dict[str, set[str]], files: list[Path] | None = None
) -> list[tuple[str, int, str]]:
    """Return (relpath, lineno, kinds) hits; never includes the value."""
    hits: list[tuple[str, int, str]] = []
    for p in files if files is not None else tracked_text_files():
        raw = p.read_bytes()
        if b"\0" in raw:
            continue  # binary
        for i, line in enumerate(raw.decode("utf-8", errors="ignore").splitlines(), 1):
            for v, kinds in values.items():
                if v in line:
                    hits.append(
                        (
                            p.name if files is not None else p.relative_to(ROOT).as_posix(),
                            i,
                            "/".join(sorted(kinds)),
                        )
                    )
    return hits


pytestmark = pytest.mark.skipif(not DATA.is_dir(), reason="data/ absent")


def test_no_real_label_values_in_tracked_files() -> None:
    hits = scan(collect_values(MIN_LEN))
    assert not hits, "real label values found (values withheld): " + "; ".join(
        f"{p}:{n} [{k}]" for p, n, k in hits
    )


def test_scan_detects_a_planted_value_without_echoing_it(tmp_path: Path) -> None:
    probe = tmp_path / "probe.txt"
    probe.write_text("ok\nline with ZZPLANT-12345 inside\n", encoding="utf-8")
    hits = scan({"ZZPLANT-12345": {"header.x"}}, files=[probe])
    assert hits == [("probe.txt", 2, "header.x")]


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def _history_repo(tmp_path: Path, seeded: bool) -> tuple[Path, Path]:
    """Synthetic repo, 3 commits; if `seeded`, a fake value is added in c1 and deleted in c2."""
    labels = tmp_path / "labels_root" / "train" / "labels"
    labels.mkdir(parents=True)
    doc = {"header": {"supplier_name": "ZZFAKE Supplier 4711"}, "line_items": []}
    (labels / "t.json").write_text(json.dumps(doc), encoding="utf-8")
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    note = repo / "notes.txt"
    note.write_text("keep\n" + ("quoted ZZFAKE Supplier 4711\n" if seeded else "nothing\n"))
    _git(repo, "add", "notes.txt")
    _git(repo, "commit", "-q", "-m", "c1")
    note.write_text("keep\n")  # the seeded line is now only in history
    _git(repo, "commit", "-q", "-am", "c2")
    (repo / "more.txt").write_text("harmless\n")
    _git(repo, "add", "more.txt")
    _git(repo, "commit", "-q", "-m", "c3")
    return repo, tmp_path / "labels_root"


def test_history_scan_finds_a_value_only_present_in_a_deleted_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = importlib.util.spec_from_file_location(
        "history_leak_scan", ROOT / "scripts" / "history_leak_scan.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["history_leak_scan"] = mod
    spec.loader.exec_module(mod)
    repo, labels_root = _history_repo(tmp_path, seeded=True)
    first = subprocess.run(
        ["git", "rev-list", "--max-parents=0", "HEAD"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()
    assert mod.main(["--repo", str(repo), "--labels-root", str(labels_root)]) == 1
    out = capsys.readouterr().out
    assert "label-value hits: 2" in out  # added in c1, deleted in c2
    assert first[:12] in out and "notes.txt" in out and "header.supplier_name" in out
    assert "ZZFAKE" not in out and "4711" not in out  # the value is never echoed


def test_history_scan_is_clean_without_the_seeded_value(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = importlib.util.spec_from_file_location(
        "history_leak_scan", ROOT / "scripts" / "history_leak_scan.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    repo, labels_root = _history_repo(tmp_path, seeded=False)
    assert mod.main(["--repo", str(repo), "--labels-root", str(labels_root)]) == 0
    assert "label-value hits: 0" in capsys.readouterr().out
