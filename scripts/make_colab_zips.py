"""Build the Colab input zips (assignment.zip, ocr_cache.zip) deterministically.

Output goes to SHIPDOC_RUNS_DIR (never inside the repo; the script refuses a repo-internal
target). Both zips have sorted entries, fixed 1980-01-01 timestamps, fixed permissions and
ZIP_DEFLATED, so a re-run on unchanged inputs yields an identical SHA256.

- assignment.zip: contents of SHIPDOC_ASSIGNMENT_DIR under a top-level ``assignment/`` folder
  (``__pycache__`` excluded).
- ocr_cache.zip: contents of SHIPDOC_OCR_CACHE (``paddleocr/`` tree incl. manifest.json, plus
  SHA256SUMS) under a top-level ``ocr_cache/`` folder. ``logs/`` is excluded. SHA256SUMS is
  verified against the files first, so a stale checksum list is never shipped.

data.zip is not built here. Its size and SHA256 are reported if it exists next to the repo root.
ZIPS_SHA256.txt (sha256sum format) is written next to the zips.

Run: uv run python scripts/make_colab_zips.py
"""

from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

from shipdoc import paths

FIXED_DATE = (1980, 1, 1, 0, 0, 0)  # earliest timestamp a zip can store
FILE_MODE = 0o100644 << 16  # regular file, rw-r--r--, as the zip external_attr
EXCLUDED_DIRS = frozenset({"logs", "__pycache__"})
SUMS_NAME = "SHA256SUMS"


def sha256_file(path: Path) -> str:
    """Hex SHA256 of a file."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(root: Path, excluded: frozenset[str] = EXCLUDED_DIRS) -> list[Path]:
    """Files under `root` (no path component in `excluded`), sorted by POSIX relative path."""
    files = [
        p
        for p in root.rglob("*")
        if p.is_file() and not (excluded & set(p.relative_to(root).parts[:-1]))
    ]
    return sorted(files, key=lambda p: p.relative_to(root).as_posix())


def write_zip(out: Path, root: Path, top: str, files: list[Path]) -> None:
    """Write `files` (under `root`) to `out` as ``<top>/<relative path>``, deterministically."""
    tmp = out.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in files:
            info = zipfile.ZipInfo(f"{top}/{f.relative_to(root).as_posix()}", FIXED_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3  # fixed (unix) so the bytes do not depend on the host OS
            info.external_attr = FILE_MODE
            zf.writestr(info, f.read_bytes(), compresslevel=6)
    tmp.replace(out)


def verify_sums(cache_root: Path) -> list[str]:
    """Problems found checking ``SHA256SUMS`` against the files it lists (empty list = ok)."""
    sums = cache_root / SUMS_NAME
    if not sums.is_file():
        return [f"{SUMS_NAME} missing in {cache_root}"]
    problems = []
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, _, rel = line.partition("  ")
        target = cache_root / rel
        if not target.is_file():
            problems.append(f"listed but missing: {rel}")
        elif sha256_file(target) != digest:
            problems.append(f"checksum mismatch: {rel}")
    return problems


def main() -> int:
    """Build both zips, print sizes and SHA256, write ZIPS_SHA256.txt."""
    out_dir = paths.runs_dir().resolve()
    repo = paths.REPO_ROOT.resolve()
    if out_dir == repo or repo in out_dir.parents:
        print(f"refusing to write zips inside the repo: {out_dir}", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_root = paths.ocr_cache_dir()
    problems = verify_sums(cache_root)
    if problems:
        print(
            "OCR cache SHA256SUMS check failed:\n  " + "\n  ".join(problems[:20]), file=sys.stderr
        )
        return 1

    jobs = [
        ("assignment.zip", paths.assignment_dir(), "assignment"),
        ("ocr_cache.zip", cache_root, "ocr_cache"),
    ]
    lines = []
    for name, root, top in jobs:
        files = collect(root)
        out = out_dir / name
        write_zip(out, root, top, files)
        digest = sha256_file(out)
        print(f"{name}: {len(files)} files, {out.stat().st_size} bytes, sha256 {digest}")
        lines.append(f"{digest}  {name}")
    data_zip = repo / "data.zip"
    if data_zip.is_file():
        digest = sha256_file(data_zip)
        print(f"data.zip (not built here): {data_zip.stat().st_size} bytes, sha256 {digest}")
        lines.append(f"{digest}  data.zip")
    (out_dir / "ZIPS_SHA256.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_dir / 'ZIPS_SHA256.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
