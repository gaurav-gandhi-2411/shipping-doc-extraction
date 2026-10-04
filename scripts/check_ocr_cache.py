"""Verify the OCR cache: per-split page counts vs. images, manifest present, SHA256SUMS intact.

Reads the cache dir from `shipdoc.paths` (SHIPDOC_OCR_CACHE) and the images from
SHIPDOC_DATA_DIR. Exits nonzero on any problem.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from shipdoc import paths
from shipdoc.ocr import list_pages

ENGINE = "paddleocr"
SPLITS = ("train", "dev", "test")
SUMS_NAME = "SHA256SUMS"


def sha256_file(path: Path) -> str:
    """Hex SHA256 of a file."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(cache_root: Path, data_root: Path) -> list[str]:
    """Return a list of problems (empty means the cache is healthy)."""
    problems: list[str] = []
    engine_dir = cache_root / ENGINE
    if not (engine_dir / "manifest.json").is_file():
        problems.append(f"missing {engine_dir / 'manifest.json'}")
    for split in SPLITS:
        try:
            expected = {p.stem for p in list_pages(data_root, split)}
        except FileNotFoundError:
            problems.append(f"no images folder for split {split!r} under {data_root}")
            continue
        cached = {p.stem for p in (engine_dir / split).glob("*.json")}
        print(f"{split}: {len(cached)} cached / {len(expected)} images")
        if missing := sorted(expected - cached):
            problems.append(f"{split}: {len(missing)} pages not cached, e.g. {missing[:3]}")
        if extra := sorted(cached - expected):
            problems.append(f"{split}: {len(extra)} cached without an image, e.g. {extra[:3]}")
    sums = cache_root / SUMS_NAME
    if not sums.is_file():
        problems.append(f"missing {sums}")
        return problems
    listed: set[str] = set()
    for n, line in enumerate(sums.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        digest, _, rel = line.partition("  ")
        listed.add(rel)
        f = cache_root / rel
        if not f.is_file():
            problems.append(f"{SUMS_NAME}:{n}: {rel} missing")
        elif sha256_file(f) != digest:
            problems.append(f"{SUMS_NAME}:{n}: {rel} hash mismatch")
    on_disk = {p.relative_to(cache_root).as_posix() for p in engine_dir.rglob("*") if p.is_file()}
    if unlisted := sorted(on_disk - listed):
        problems.append(f"{len(unlisted)} files not in {SUMS_NAME}, e.g. {unlisted[:3]}")
    print(f"{SUMS_NAME}: {len(listed)} entries checked")
    return problems


def main() -> int:
    """CLI entry; prints problems and returns the exit code."""
    cache_root = paths.ocr_cache_dir()
    print(f"cache: {cache_root}")
    problems = check(cache_root, paths.data_dir())
    for p in problems:
        print("PROBLEM:", p)
    print("OK" if not problems else f"FAILED ({len(problems)} problems)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
