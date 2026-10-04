"""Summarise the OCR cache and manifest: pages, failures, s/page mean and p95, rotations."""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

from shipdoc import paths
from shipdoc.ocr import percentile


def main(engine: str = "paddleocr", root: Path | None = None) -> None:
    """Print cache statistics computed from the per-page JSON files and the manifest."""
    root = paths.ocr_cache_dir() if root is None else root
    manifest = json.loads((root / engine / "manifest.json").read_text(encoding="utf-8"))
    secs: list[float] = []
    rot: Counter[str] = Counter()
    lines = 0
    by_split: Counter[str] = Counter()
    for f in sorted((root / engine).glob("*/*.json")):
        page = json.loads(f.read_text(encoding="utf-8"))
        secs.append(page["seconds"])
        rot[str(page["rotation_deg"])] += 1
        lines += len(page["items"])
        by_split[f.parent.name] += 1
    print(
        "engine",
        manifest["engine"],
        "paddleocr",
        manifest["paddleocr_version"],
        "paddlepaddle",
        manifest["paddlepaddle_version"],
        "paddlex",
        manifest["paddlex_version"],
    )
    print("models", manifest["models"])
    print("pages expected", manifest["pages_expected"], "cached", len(secs), dict(by_split))
    print("failures in manifest (last run)", len(manifest["failures"]), manifest["failures"][:5])
    print(
        "s/page mean",
        round(statistics.fmean(secs), 2),
        "median",
        round(statistics.median(secs), 2),
        "p95",
        round(percentile(secs, 95), 2),
        "sum_h",
        round(sum(secs) / 3600, 2),
    )
    print("rotation_deg counts", dict(rot), "total lines", lines)
    print("this_run", manifest.get("this_run"))


if __name__ == "__main__":
    main(*sys.argv[1:2])
