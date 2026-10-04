"""OCR stage of notebook 04b (production pipeline v1): checks, timing and the re-OCR check.

The OCR itself is ``python -m shipdoc ocr --splits test`` run in the isolated Paddle venv by
``scripts/colab_ocr_test.sh`` (paddlepaddle-gpu clashes with the vlm group, so Paddle is never
installed into, or imported by, the environment that decodes). This module is the thin, testable
part around it:

``check``     are all expected TEST pages cached and valid? (counts only; ``--isolate-invalid``
              renames a corrupt page file to ``*.invalid`` so a rerun redoes it: nothing is deleted)
``device``    GPU / CPU decision from nvidia-smi and the venv's paddle flavour, loud on a fallback
``finalize``  ``ocr_timing.json`` (pages, seconds total, seconds/page mean and p95, device)
``recheck``   5 seeded pages OCR'd again in a fresh process into a scratch root and compared

Importing this module never imports Paddle (``PaddleOcrEngine`` imports it lazily, and only the
``recheck`` stage builds an engine, inside the OCR venv).

Test data policy: OCR text is never printed, logged or stored here. Reports carry counts, page
names and timings only; the recheck compares texts in memory and reports booleans.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from shipdoc import ocr

SPLIT = "test"
ENGINE = ocr.DEFAULT_ENGINE
RECHECK_N = 5
RECHECK_SEED = 42
INVALID_SUFFIX = ".invalid"
MAX_LISTED = 5  # page names listed in a message; counts carry the rest
RECHECK_RULE = (
    "random.Random(42).sample of 5 test pages (sorted page stems) are OCR'd again in a fresh "
    "process into a scratch cache. The cached JSON always differs in `seconds`, so 'identical' "
    "means: (1) text-identical = ocr.page_text equal, the exact text R2 reads (BLOCKING); "
    "(2) content-identical = the whole page JSON equal once `seconds` is dropped (reported, not "
    "blocking: a GPU kernel may move a box by a pixel without changing a character)."
)


class OcrStageError(RuntimeError):
    """The OCR cache is incomplete / invalid or a recheck failed; messages carry counts only."""


# --------------------------------------------------------------------------------------------
# Pages and cache validity
# --------------------------------------------------------------------------------------------


def expected_stems(images_dir: Path) -> list[str]:
    """Sorted page stems (``test_0001_p1``) of the image folder, from file names only."""
    folder = Path(images_dir)
    if not folder.is_dir():
        raise OcrStageError(f"no image folder {folder}")
    stems = sorted(p.stem for p in folder.iterdir() if p.suffix.lower() in ocr.IMAGE_SUFFIXES)
    if not stems:
        raise OcrStageError(f"no page images in {folder}")
    return stems


def page_is_valid(path: Path, stem: str) -> bool:
    """A cached page file parses as a ``PageOcr`` of this engine and page, with numeric seconds."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("items"), list):
            return False
        secs = data.get("seconds")
        if isinstance(secs, bool) or not isinstance(secs, int | float) or secs < 0:
            return False
        page = ocr.PageOcr.from_dict(data)
    except (OSError, ValueError, TypeError, KeyError):
        return False
    return Path(page.page).stem == stem and page.engine == ENGINE


def cache_status(cache_root: Path, stems: Sequence[str]) -> dict[str, Any]:
    """Counts of valid / missing / invalid cached pages of `stems` (page names, never text)."""
    valid, missing, invalid = [], [], []
    for stem in stems:
        p = ocr.cache_path(cache_root, ENGINE, SPLIT, stem)
        if not p.is_file():
            missing.append(stem)
        elif page_is_valid(p, stem):
            valid.append(stem)
        else:
            invalid.append(stem)
    return {
        "expected": len(stems),
        "valid": len(valid),
        "missing": len(missing),
        "invalid": len(invalid),
        "missing_pages": missing[:MAX_LISTED],
        "invalid_pages": invalid[:MAX_LISTED],
        "invalid_all": invalid,
        "complete": len(valid) == len(stems) and bool(stems),
    }


def isolate_invalid(cache_root: Path, stems: Sequence[str]) -> list[str]:
    """Rename every invalid page file to ``<name>.invalid`` (kept for inspection, never deleted).

    ``ocr.run_ocr`` skips any page whose cache file exists, so an invalid file must move away for
    a resumed run to redo the page. Returns the stems moved.
    """
    moved = []
    for stem in cache_status(cache_root, stems)["invalid_all"]:
        p = ocr.cache_path(cache_root, ENGINE, SPLIT, stem)
        target = p.with_name(p.name + INVALID_SUFFIX)
        n = 1
        while target.exists():  # never overwrite an earlier quarantined copy
            target = p.with_name(f"{p.name}{INVALID_SUFFIX}{n}")
            n += 1
        p.rename(target)
        moved.append(stem)
    return moved


def require_complete(cache_root: Path, stems: Sequence[str], expect: int) -> dict[str, Any]:
    """The cache status, or OcrStageError unless exactly `expect` pages exist and all are valid."""
    st = cache_status(cache_root, stems)
    if len(stems) != expect or not st["complete"]:
        raise OcrStageError(
            f"OCR cache incomplete: {st['valid']}/{len(stems)} valid pages (expected {expect}); "
            f"missing {st['missing']} {st['missing_pages']}, invalid {st['invalid']} "
            f"{st['invalid_pages']}"
        )
    return st


# --------------------------------------------------------------------------------------------
# Device
# --------------------------------------------------------------------------------------------


def decide_device(has_gpu: bool, flavor: str | None) -> dict[str, Any]:
    """OCR device from the runtime (nvidia-smi) and the venv's paddle flavour (``.paddle_flavor``).

    ``flavor`` is ``gpu-cu126`` / ``gpu-cu118`` / ``cpu`` (None = unknown, treated as cpu: that is
    what ``scripts/colab_ocr_run.sh`` does). A CPU run on a GPU runtime is a FALLBACK and says so
    in `warnings`; a GPU paddle build without a visible GPU cannot run and is `ok` false.
    """
    gpu_build = bool(flavor) and str(flavor).startswith("gpu-")
    warnings: list[str] = []
    ok = True
    if gpu_build and has_gpu:
        device = "gpu"
    elif gpu_build:
        device, ok = "gpu", False
        warnings.append(
            f"paddle flavour {flavor} needs a GPU but nvidia-smi shows none: switch the runtime "
            "to a GPU or rebuild the OCR venv with FORCE_CPU=1"
        )
    else:
        device = "cpu"
        if has_gpu:
            warnings.append(
                f"CPU FALLBACK on a GPU runtime (paddle flavour {flavor or 'unknown'}): the GPU "
                "wheel was not installed (not listed for this CUDA, or its smoke test failed; see "
                "the setup log). OCR will take roughly 5-10x longer."
            )
        else:
            warnings.append("CPU OCR: no GPU runtime (nvidia-smi found none); expect a long run.")
    return {
        "device": device, "flavor": flavor, "has_gpu": has_gpu, "ok": ok,
        "fallback": device == "cpu", "warnings": warnings,
    }  # fmt: skip


# --------------------------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------------------------


def timing_summary(
    cache_root: Path, stems: Sequence[str], device: str, wall_clock_s: float | None = None,
    new_pages: int | None = None, prior: Mapping[str, Any] | None = None,
) -> dict[str, Any]:  # fmt: skip
    """``ocr_timing.json`` content: per-page seconds are those stored in each page JSON.

    ``seconds_total`` is the sum over the cached pages (compute time, across resumed sessions);
    ``wall_clock_s`` is this session's elapsed time including model load (summed with the
    sessions recorded in `prior`). Raises OcrStageError when a page is missing or invalid.
    """
    require_complete(cache_root, stems, len(stems))
    secs = [
        float(
            json.loads(ocr.cache_path(cache_root, ENGINE, SPLIT, s).read_text("utf-8"))["seconds"]
        )
        for s in stems
    ]
    sessions = list((prior or {}).get("sessions") or [])
    if wall_clock_s is not None:
        sessions.append({"wall_clock_s": round(wall_clock_s, 1), "new_pages": new_pages,
                         "device": device})  # fmt: skip
    return {
        "schema": 1,
        "engine": ENGINE,
        "split": SPLIT,
        "pages": len(secs),
        "seconds_total": round(sum(secs), 1),
        "seconds_per_page_mean": round(statistics.fmean(secs), 3),
        "seconds_per_page_p95": round(ocr.percentile(secs, 95), 3),
        "device": device,
        "wall_clock_s_all_sessions": round(sum(s.get("wall_clock_s") or 0 for s in sessions), 1),
        "sessions": sessions,
        "note": (
            "seconds_* = sum/mean/p95 of the per-page `seconds` stored in the page JSON (compute "
            "time of the session that produced each page; pages resumed from an earlier session "
            "keep that session's seconds and device). No OCR text is stored here."
        ),
    }


# --------------------------------------------------------------------------------------------
# Re-OCR determinism check
# --------------------------------------------------------------------------------------------


def select_recheck_pages(
    stems: Sequence[str], n: int = RECHECK_N, seed: int = RECHECK_SEED
) -> list[str]:
    """The seeded pages (`RECHECK_RULE`), sorted; all pages when there are <= n."""
    ids = sorted(stems)
    if len(ids) <= n:
        return ids
    return sorted(random.Random(seed).sample(ids, n))  # noqa: S311 - reproducible, not security


def _load(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare_pages(first: Path, second: Path) -> dict[str, bool]:
    """``text_identical`` (page_text equal) and ``content_identical`` (JSON equal but `seconds`)."""
    a, b = _load(first), _load(second)
    text = ocr.page_text(ocr.PageOcr.from_dict(dict(a))) == ocr.page_text(
        ocr.PageOcr.from_dict(dict(b))
    )
    drop = {"seconds"}
    content = {k: v for k, v in a.items() if k not in drop} == {
        k: v for k, v in b.items() if k not in drop
    }
    return {"text_identical": text, "content_identical": content}


def recheck(
    engine: ocr.OcrEngine, images_dir: Path, cache_root: Path, scratch_root: Path,
    stems: Sequence[str] | None = None,
) -> dict[str, Any]:  # fmt: skip
    """OCR the seeded pages again with `engine` into `scratch_root` and compare with the cache.

    Report: counts and page names only. ``ok`` = every page text-identical. The caller runs it in
    a FRESH process (the notebook launches it as its own subprocess in the OCR venv).
    """
    images = {
        p.stem: p for p in Path(images_dir).iterdir() if p.suffix.lower() in ocr.IMAGE_SUFFIXES
    }
    chosen = list(stems) if stems is not None else select_recheck_pages(sorted(images))
    t0 = time.perf_counter()
    per_page: dict[str, dict[str, bool]] = {}
    for stem in chosen:
        out = ocr.cache_path(scratch_root, ENGINE, SPLIT, stem)
        page = engine.run(images[stem])
        ocr.write_json_atomic(out, page.to_dict())
        per_page[stem] = compare_pages(ocr.cache_path(cache_root, ENGINE, SPLIT, stem), out)
    n_text = sum(v["text_identical"] for v in per_page.values())
    n_content = sum(v["content_identical"] for v in per_page.values())
    return {
        "schema": 1,
        "rule": RECHECK_RULE,
        "seed": RECHECK_SEED,
        "pages": chosen,
        "n_pages": len(chosen),
        "n_text_identical": n_text,
        "n_content_identical": n_content,
        "per_page": per_page,
        "seconds": round(time.perf_counter() - t0, 1),
        "ok": bool(chosen) and n_text == len(chosen),
    }


# --------------------------------------------------------------------------------------------
# CLI: python -m shipdoc.ocr_stage <check|device|finalize|recheck>
# --------------------------------------------------------------------------------------------


def _write_json(path: Path, obj: Any) -> None:
    ocr.write_json_atomic(Path(path), obj)


def build_parser() -> argparse.ArgumentParser:
    """Argument parser of the stage CLI."""
    p = argparse.ArgumentParser(prog="python -m shipdoc.ocr_stage")
    sub = p.add_subparsers(dest="stage", required=True)
    for name in ("check", "finalize", "recheck"):
        s = sub.add_parser(name)
        s.add_argument("--images-dir", type=Path, required=True)
        s.add_argument("--cache-root", type=Path, required=True)
        s.add_argument("--expect-pages", type=int, default=280)
        s.add_argument("--out", type=Path, default=None)
        if name == "check":
            s.add_argument("--isolate-invalid", action="store_true")
        if name == "finalize":
            s.add_argument("--device", required=True)
            s.add_argument("--wall-s", type=float, default=None)
            s.add_argument("--new-pages", type=int, default=None)
        if name == "recheck":
            s.add_argument("--scratch-root", type=Path, required=True)
    s = sub.add_parser("device")
    s.add_argument("--has-gpu", choices=["0", "1"], required=True)
    s.add_argument("--flavor", default=None)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Stage CLI. Exit 0 = the stage passed; messages are counts only."""
    args = build_parser().parse_args(argv)
    try:
        if args.stage == "device":
            d = decide_device(args.has_gpu == "1", args.flavor or None)
            print(json.dumps(d))
            return 0 if d["ok"] else 1
        stems = expected_stems(args.images_dir)
        if args.stage == "check":
            moved = isolate_invalid(args.cache_root, stems) if args.isolate_invalid else []
            st = cache_status(args.cache_root, stems)
            if args.out:
                _write_json(args.out, {k: v for k, v in st.items() if k != "invalid_all"})
            print(
                f"ocr cache: {st['valid']}/{st['expected']} valid, {st['missing']} missing, "
                f"{st['invalid']} invalid, {len(moved)} isolated; complete={st['complete']}"
            )
            return 0 if st["complete"] and st["expected"] == args.expect_pages else 1
        if args.stage == "finalize":
            require_complete(args.cache_root, stems, args.expect_pages)
            prior = _load(args.out) if args.out and args.out.is_file() else None
            summary = timing_summary(
                args.cache_root, stems, args.device, args.wall_s, args.new_pages, prior
            )
            if args.out:
                _write_json(args.out, summary)
            print(
                f"ocr timing: {summary['pages']} pages, {summary['seconds_total']} s total, "
                f"{summary['seconds_per_page_mean']} s/page mean, "
                f"{summary['seconds_per_page_p95']} s/page p95, device {summary['device']}"
            )
            return 0
        require_complete(args.cache_root, stems, args.expect_pages)  # recheck
        engine = ocr.PaddleOcrEngine()  # lazy paddle import: only inside the OCR venv
        rep = recheck(engine, args.images_dir, args.cache_root, args.scratch_root)
        if args.out:
            _write_json(args.out, rep)
        print(
            f"ocr recheck: {rep['n_text_identical']}/{rep['n_pages']} pages text-identical, "
            f"{rep['n_content_identical']}/{rep['n_pages']} content-identical (all but seconds)"
        )
        return 0 if rep["ok"] else 1
    except OcrStageError as exc:
        print(f"ocr_stage {args.stage}: REFUSED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
