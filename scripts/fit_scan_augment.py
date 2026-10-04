"""Fit the scan-degradation parameter ranges to the train scans; write the report and config.

Groups (train split only; sampled with seed 42): scans = ``.jpg`` pages, digital = ``.png`` pages.
Fit data and report data are disjoint pages:

* fit    : 100 scans (target distribution) and 40 digital pages (what gets augmented),
* report : the other 100 scans, 100 other digital pages raw, and those same 100 digital pages
           augmented with the fitted parameters.

Search: random search over the `ScanParams` ranges (objective = mean over the 7 statistics of the
two-sample KS distance between the fit scans and the augmented fit digital pages), then a local
refinement around the best. Outputs: ``configs/augment_scan.yaml`` and ``reports/augmentation.md``.

Run: ``uv run python scripts/fit_scan_augment.py`` (CPU, ~10 min with 12 worker processes).
"""

# ruff: noqa: E501  # long markdown literals in the report writer

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from shipdoc import augment as A
from shipdoc import imgstats as S
from shipdoc import paths
from shipdoc.ocr import doc_pages

ROOT = Path(__file__).resolve().parents[1]
CONFIG_OUT = ROOT / "configs" / "augment_scan.yaml"
REPORT_OUT = ROOT / "reports" / "augmentation.md"
SEED = 42
N_FIT_SCANS = 100
N_FIT_DIGITAL = 40
N_REPORT = 100
N_RANDOM = 96
N_REFINE = 64
WORKERS = 12
# Untuned reference: the generic "looks scanned" recipe (spec ranges at their widest).
NAIVE = A.ScanParams(3.0, (0.9, 1.0), 1.0, (0.0, 30.0), (0.0, 1.5), (0.0, 5.0), (30, 70))

# Best of an earlier run of this same search before `paper_gain_skew` existed (fit-page mean KS
# 0.211, skew 1.0 = plain uniform gain). Added as a starting candidate because the enlarged random
# space with the same 96-candidate budget landed worse (0.254) by search noise alone; seeding makes
# the result at least as good as that run on the fit pages.
WARM_START = {
    "rotate_deg": 1.484,
    "paper_gain": [0.906, 1.0],
    "paper_gain_skew": 1.0,
    "ink_lift": [13.083, 24.281],
    "blur_sigma": [0.07, 1.093],
    "noise_sigma": [2.147, 6.756],
    "jpeg_quality": [49, 50],
}

_FIT_DIGITAL: list[np.ndarray] = []


def _init_worker(paths_: list[str]) -> None:
    _FIT_DIGITAL.extend(S.to_gray(Image.open(p)) for p in paths_)


def _augmented_stats(arrs: list[np.ndarray], params: A.ScanParams, seed: int) -> list[dict]:
    out = []
    for i, a in enumerate(arrs):
        rng = np.random.default_rng([SEED, seed, i])
        img = A.scan_degrade(Image.fromarray(a), rng, params)
        out.append(S.image_stats(S.to_gray(img)))
    return out


def _ks_by_stat(target: list[dict], cand: list[dict]) -> dict[str, float]:
    return {k: S.ks_distance([r[k] for r in target], [r[k] for r in cand]) for k in S.STAT_NAMES}


def _eval_candidate(args: tuple[dict, int, list[dict]]) -> dict[str, Any]:
    pd, seed, target = args
    params = A.ScanParams.from_dict(pd)
    ks = _ks_by_stat(target, _augmented_stats(_FIT_DIGITAL, params, seed))
    return {"params": pd, "ks": ks, "objective": float(np.mean(list(ks.values())))}


def _sample(rng: np.random.Generator) -> dict[str, Any]:
    def pair(lo_hi: tuple[float, float], top: float) -> list[float]:
        lo = rng.uniform(*lo_hi)
        return [round(float(lo), 3), round(float(rng.uniform(lo, top)), 3)]

    q_lo = int(rng.integers(30, 61))
    return {
        "rotate_deg": round(float(rng.uniform(0.0, 3.0)), 3),
        "paper_gain": pair((0.85, 1.0), 1.0),
        "paper_gain_skew": round(float(rng.uniform(1.0, 5.0)), 2),
        "ink_lift": pair((0.0, 15.0), 40.0),
        "blur_sigma": pair((0.0, 1.0), 2.5),
        "noise_sigma": pair((0.0, 2.0), 6.0),
        "jpeg_quality": [q_lo, int(rng.integers(q_lo, 71))],
    }


def _perturb(pd: dict[str, Any], rng: np.random.Generator, scale: float) -> dict[str, Any]:
    """Gaussian jitter of every range endpoint, clipped to valid/spec bounds."""

    def pr(v: list[float], lo: float, hi: float, s: float) -> list[float]:
        a = float(np.clip(v[0] + rng.normal(0, s * scale), lo, hi))
        b = float(np.clip(v[1] + rng.normal(0, s * scale), a, hi))
        return [round(a, 3), round(b, 3)]

    q = pd["jpeg_quality"]
    qa = int(np.clip(round(q[0] + rng.normal(0, 6 * scale)), 30, 70))
    qb = int(np.clip(round(q[1] + rng.normal(0, 6 * scale)), qa, 70))
    return {
        "rotate_deg": round(float(np.clip(pd["rotate_deg"] + rng.normal(0, 0.5 * scale), 0, 3)), 3),
        "paper_gain": pr(pd["paper_gain"], 0.8, 1.0, 0.03),
        "paper_gain_skew": round(
            float(np.clip(pd["paper_gain_skew"] + rng.normal(0, 0.6 * scale), 1.0, 8.0)), 2
        ),
        "ink_lift": pr(pd["ink_lift"], 0.0, 60.0, 5.0),
        "blur_sigma": pr(pd["blur_sigma"], 0.0, 3.0, 0.3),
        "noise_sigma": pr(pd["noise_sigma"], 0.0, 8.0, 0.8),
        "jpeg_quality": [qa, qb],
    }


def _pool_stats(files: list[Path]) -> list[dict]:
    with ProcessPoolExecutor(WORKERS) as ex:
        return list(ex.map(_file_stats, [str(f) for f in files], chunksize=4))


def _file_stats(path: str) -> dict:
    return S.image_stats(S.to_gray(Image.open(path)))


def _aug_file_stats(args: tuple[str, dict, int]) -> dict:
    path, pd, i = args
    rng = np.random.default_rng([SEED, 999, i])
    img = A.scan_degrade(Image.open(path).convert("L"), rng, A.ScanParams.from_dict(pd))
    return S.image_stats(S.to_gray(img))


def _ocr_skews(files: list[Path]) -> list[float]:
    """|OCR-line skew| per page from the cached polygons (NaN pages dropped)."""
    out = []
    for f in files:
        pages = doc_pages(f.stem.rsplit("_p", 1)[0])
        page = next((p for p in pages if Path(p.page).stem == f.stem), None)
        if page is None or not page.items:
            continue
        v = S.ocr_line_skew([it.box for it in page.items])
        if v == v:
            out.append(abs(v))
    return out


def _fmt(vals: list[float], k: str) -> str:
    a = np.asarray(vals, float)
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    d = 4 if k == "abs_skew_deg" else 2
    return f"{med:.{d}f} [{q1:.{d}f}, {q3:.{d}f}]"


def _reading(raw: dict[str, float], aug: dict[str, float], naive: dict[str, float]) -> str:
    """Numbers-derived summary plus the standing caveats of the method."""
    m = lambda d: float(np.mean(list(d.values())))  # noqa: E731
    bad = [f"{k} ({v:.2f})" for k, v in aug.items() if v > 0.2]
    worse = [k for k in aug if aug[k] > raw[k] + 0.02]
    return "\n".join(
        [
            f"- Mean KS to the scans: raw digital {m(raw):.3f}, fitted augmentation {m(aug):.3f}, "
            f"untuned ranges {m(naive):.3f}.",
            "- Statistics still above KS 0.2 after fitting: " + (", ".join(bad) or "none") + ".",
            "- Statistics where augmentation is worse than raw digital by more than 0.02 KS: "
            + (", ".join(worse) or "none")
            + ".",
            "- Caveats: KS is per statistic (marginals), so the joint distribution is not "
            "guaranteed to match; the search space is a product of independent uniform ranges, so "
            "multi-modal scan quality (e.g. sharp and soft scans) is covered only by wide ranges; "
            "rotation is capped at 3 degrees by spec; JPEG quality is capped to 30-70 by spec; "
            "the noise and skew estimators are crude (a robust MAD and a projection profile), "
            "so a match on them is a match on those estimators, not on physical scanner noise.",
        ]
    )


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:  # noqa: BLE001  # provenance is best-effort; the report says so
        return "unknown"


def main() -> int:
    t0 = time.time()
    img_dir = paths.data_dir() / "train" / "images"
    scans = sorted(img_dir.glob("*.jpg"))
    digital = sorted(img_dir.glob("*.png"))
    rng = np.random.default_rng(SEED)
    scans = [scans[i] for i in rng.permutation(len(scans))]
    digital = [digital[i] for i in rng.permutation(len(digital))]
    fit_scans, rep_scans = scans[:N_FIT_SCANS], scans[N_FIT_SCANS : N_FIT_SCANS + N_REPORT]
    fit_dig = digital[:N_FIT_DIGITAL]
    rep_dig = digital[N_FIT_DIGITAL : N_FIT_DIGITAL + N_REPORT]
    assert len(rep_scans) >= N_REPORT and len(rep_dig) >= N_REPORT
    print(f"scans {len(scans)} digital {len(digital)}; stats of fit scans ...", flush=True)
    target = _pool_stats(fit_scans)

    cands = [WARM_START] + [_sample(rng) for _ in range(N_RANDOM)]
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        WORKERS, initializer=_init_worker, initargs=([str(p) for p in fit_dig],)
    ) as ex:
        results += list(ex.map(_eval_candidate, [(c, 1, target) for c in cands]))
        print(f"random search done {time.time() - t0:.0f}s; best "
              f"{min(r['objective'] for r in results):.4f}", flush=True)  # fmt: skip
        for rnd, scale in enumerate((1.0, 0.5)):
            best = min(results, key=lambda r: r["objective"])["params"]
            batch = [_perturb(best, rng, scale) for _ in range(N_REFINE // 2)]
            results += list(ex.map(_eval_candidate, [(c, 2 + rnd, target) for c in batch]))
            print(f"refine {rnd} done {time.time() - t0:.0f}s; best "
                  f"{min(r['objective'] for r in results):.4f}", flush=True)  # fmt: skip
    best = min(results, key=lambda r: r["objective"])
    params = A.ScanParams.from_dict(best["params"])
    naive_fit = _ks_by_stat(target, _augmented_stats(
        [S.to_gray(Image.open(p)) for p in fit_dig], NAIVE, 1))  # fmt: skip
    print("best", best, flush=True)

    # Held-out report: disjoint scans and digital pages.
    rep_scan_stats = _pool_stats(rep_scans)
    rep_dig_stats = _pool_stats(rep_dig)
    with ProcessPoolExecutor(WORKERS) as ex:
        aug_stats = list(
            ex.map(
                _aug_file_stats,
                [(str(p), best["params"], i) for i, p in enumerate(rep_dig)],
                chunksize=4,
            )
        )
        naive_stats = list(
            ex.map(
                _aug_file_stats,
                [(str(p), NAIVE.to_dict(), i) for i, p in enumerate(rep_dig)],
                chunksize=4,
            )
        )
    ks_raw = _ks_by_stat(rep_scan_stats, rep_dig_stats)
    ks_aug = _ks_by_stat(rep_scan_stats, aug_stats)
    ks_naive = _ks_by_stat(rep_scan_stats, naive_stats)
    ocr_scan, ocr_dig = _ocr_skews(rep_scans), _ocr_skews(rep_dig)

    CONFIG_OUT.write_text(
        "# Fitted by scripts/fit_scan_augment.py (seed 42); see reports/augmentation.md.\n"
        + yaml.safe_dump(
            {
                "seed": SEED,
                "params": params.to_dict(),
                "fit": {
                    "objective_mean_ks_on_fit_pages": round(best["objective"], 4),
                    "n_fit_scans": N_FIT_SCANS,
                    "n_fit_digital": N_FIT_DIGITAL,
                    "n_candidates": len(results),
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    lines = [
        "# Scan-degradation augmentation, fitted to train scans",
        "",
        f"Generated by `scripts/fit_scan_augment.py` (seed {SEED}, base commit `{_git_head()}`, "
        f"{time.time() - t0:.0f}s on CPU). Params: `configs/augment_scan.yaml`. Confidential data "
        "policy: only aggregate statistics appear here; no page content.",
        "",
        "## Setup",
        "",
        f"- Groups from the **train** split: scans = `.jpg` pages ({len(scans)} available), digital = "
        f"`.png` pages ({len(digital)} available). All pages are 1240x1754 grayscale.",
        f"- Fit pages (disjoint from the report pages below): {N_FIT_SCANS} scans as target, "
        f"{N_FIT_DIGITAL} digital pages augmented per candidate.",
        f"- Search: one warm-start candidate (best of an earlier run without the `paper_gain_skew` parameter), {N_RANDOM} random candidates, plus {N_REFINE} local refinements ({len(results)} "
        "total); objective = mean over the 7 statistics of the two-sample KS distance. "
        "Best objective on the fit pages: "
        f"{best['objective']:.3f} (untuned `NAIVE` ranges: {np.mean(list(naive_fit.values())):.3f}).",
        f"- Report pages (held out from the fit): {len(rep_scan_stats)} scans, {len(rep_dig_stats)} "
        f"raw digital, the same {len(aug_stats)} digital pages augmented with the fitted params "
        f"(seed {SEED}), and augmented with the untuned `NAIVE` ranges for reference.",
        "- Statistics (`src/shipdoc/imgstats.py`): mean / std of grey level, background level "
        "(90th percentile), Laplacian variance (sharpness), noise (MAD of the residual after a 3x3 "
        "box blur, x1.4826), absolute skew (projection profile, degrees), JPEG blockiness "
        "(8x8 boundary step minus interior step, grey levels).",
        "- Cells: median [Q1, Q3]. KS = two-sample Kolmogorov-Smirnov distance to the held-out "
        "scans (0 = identical distributions, 1 = disjoint). Stats of each group are over 100 pages.",
        "",
        "## Fitted parameters",
        "",
        "```yaml",
        yaml.safe_dump(params.to_dict(), sort_keys=False).strip(),
        "```",
        "",
        "## Before / after",
        "",
        "| statistic | scans | raw digital | KS raw | augmented digital | KS aug | naive-range aug | KS naive |",
        "|---|---|---|---:|---|---:|---|---:|",
    ]
    for k in S.STAT_NAMES:
        lines.append(
            f"| {k} | {_fmt([r[k] for r in rep_scan_stats], k)} | {_fmt([r[k] for r in rep_dig_stats], k)} "
            f"| {ks_raw[k]:.3f} | {_fmt([r[k] for r in aug_stats], k)} | {ks_aug[k]:.3f} "
            f"| {_fmt([r[k] for r in naive_stats], k)} | {ks_naive[k]:.3f} |"
        )
    lines.append(
        f"| **mean KS** | | | **{np.mean(list(ks_raw.values())):.3f}** | | "
        f"**{np.mean(list(ks_aug.values())):.3f}** | | **{np.mean(list(ks_naive.values())):.3f}** |"
    )
    lines += [
        "",
        "Cross-check of the skew estimator on pages that have cached OCR (median absolute angle of "
        "OCR line polygons, degrees; augmented pages have no OCR so none is given): "
        f"scans {_fmt(ocr_scan, 'abs_skew_deg')} (n={len(ocr_scan)}), raw digital "
        f"{_fmt(ocr_dig, 'abs_skew_deg')} (n={len(ocr_dig)}).",
        "",
        "## Reading the table",
        "",
        _reading(ks_raw, ks_aug, ks_naive),
        "",
    ]
    REPORT_OUT.write_text("\n".join(lines), encoding="utf-8")
    json.dump(
        {"ks_raw": ks_raw, "ks_aug": ks_aug, "ks_naive": ks_naive},
        open(paths.runs_dir() / "_scratch" / "augment_ks.json", "w"),  # noqa: SIM115
    )
    print("wrote", CONFIG_OUT, REPORT_OUT, f"{time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    sys.exit(main())
