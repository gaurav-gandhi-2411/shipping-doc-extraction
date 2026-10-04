"""Native-resolution runs (notebooks 02n and 05n): estimates and the resolution gate.

GG decision 2026-10-03: the production resolution is native (``configs/spike_qwen35_4b_img_only_
native.yaml``: ``max_pixels`` 2,196,480 = 2,145 visual tokens per 1240 x 1754 page). The 1260-token
config stays for v0 / v1. Two things in the 02 / 05 notebooks cannot simply be re-pointed at it:

* the speed table ``configs/spike_speed.json`` has no entry for the native config (and must not be
  edited), so the T4 hour / CU ESTIMATE is computed here from the two pace figures that exist:
  the measured 02 batch-8 pace at 1260 tokens (8406.1 s for 671 pages, see
  ``docs/run_sheet_closing.md``) scaled by the sweep's native / control cost ratio, and the
  sweep's measured native batch-1 pace (``reports/res_sweep.md``). Everything printed is an
  ESTIMATE, UNVERIFIED on a GPU;
* an adapter trained at another resolution must never be run at the native resolution (or the
  reverse): ``check_adapter_resolution`` is that gate. It FAILS CLOSED: a missing
  ``shipdoc.resmatch`` module, an exception inside it, a malformed answer, an unreadable manifest
  or a config that is not the native one all refuse.

CLI (``python -m shipdoc.nativerun``): ``verify`` / ``infer`` (resolution gate, then the unchanged
``shipdoc oof`` stage), ``estimate-oof`` (T4 hours / CU of an OOF fold). The estimate functions are
stdlib only so a notebook can load this file with importlib before the venv exists.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
NATIVE_CONFIG = "configs/spike_qwen35_4b_img_only_native.yaml"
NATIVE_CONFIG_NAME = "qwen35_4b_img_only_native"
#: 1240 x 1754 page, no upscaling: 1760 x 1248 grid = 2,145 tokens (reports/res_sweep.md).
NATIVE_MAX_PIXELS = 2_196_480
LEGACY_MAX_PIXELS = 1_310_720  # configs/spike_qwen35_4b_img_only.yaml, 1,260 tokens (v0 / v1)

#: MEASURED by the 02 run (sessions.json, one session, exit 0): 8406.1 s for 671 pages at batch 8,
#: 1260 tokens (docs/run_sheet_closing.md). The GPU is not recorded in its manifest (assumed T4).
ZS_B8_SECONDS, ZS_PAGES = 8406.1, 671
ZS_B8_S_PER_PAGE = ZS_B8_SECONDS / ZS_PAGES
#: MEASURED by the 06 sweep at batch 1: native 2133.4 s vs control 2036.5 s per 40-doc run = x1.0476
#: (the table's rounded 38.3 / 36.6 s/page gives x1.0464); x1.047 is used (reports/res_sweep.md).
NATIVE_FACTOR = 1.047
NATIVE_B1_S_PER_PAGE = 38.3  # MEASURED by the sweep at batch 1, native (reports/res_sweep.md)
#: ASSUMED: the 02 batch-8 pace scaled by the sweep's cost ratio. Valid only if the native bench
#: accepts batch 8 (VRAM at batch 8 is UNMEASURED at native; the 1260-token peak was 13.6 GiB).
NATIVE_B8_S_PER_PAGE = ZS_B8_S_PER_PAGE * NATIVE_FACTOR
BENCH_PAGES = 12  # shipdoc.bench: 12 dev pages at each size, plus one warm-up page at batch 1
BENCH_SIZES_ABOVE_1 = 3  # sizes 2, 4, 8
MERGE_S = 60.0  # ASSUMED, as in shipdoc.oof.MERGE_S_ESTIMATE

SCENARIOS = ("low", "high")


class NativeError(RuntimeError):
    """The native-resolution preconditions failed. Messages never quote a document value."""


# --------------------------------------------------------------------------------------------
# Estimates (stdlib only; every figure is an ESTIMATE)
# --------------------------------------------------------------------------------------------


def batched_pace(scenario: str) -> float:
    """Seconds per page at a batch above 1: ``low`` = batch-8 pace x1.047, ``high`` = batch 1.

    The bounds of a pace nobody has measured at native: the true value for the batch size the
    bench picks lies between them (batch 1 is exactly the ``high`` figure).
    """
    if scenario == "low":
        return NATIVE_B8_S_PER_PAGE
    if scenario == "high":
        return NATIVE_B1_S_PER_PAGE
    raise ValueError(f"scenario must be one of {SCENARIOS}, got {scenario!r}")


def _cu(speed: Mapping[str, Any], hours: float) -> tuple[float | None, float | None]:
    r1, r2 = speed.get("t4_cu_per_hour"), speed.get("t4_cu_per_hour_conservative")
    return (None if r1 is None else hours * r1, None if r2 is None else hours * r2)


def zs_estimate_rows(
    speed: Mapping[str, Any], full_pages: int, smoke_pages: int, batch: int | None
) -> list[dict[str, Any]]:
    """ESTIMATE rows (low / high pace) of one 02n tab: smoke, bench, full run.

    `batch` None = the bench decides (bench cost included, pace between the bounds); an int =
    manual (no bench; batch 1 is exactly the batch-1 pace). Each stage pays one model load
    (``model_load_s``, an ESTIMATE); the smoke runs at batch 1; the bench is a warm-up page at
    batch 1 plus 12 pages at batch 1 and at the three larger sizes, whose pace is the scenario's.
    """
    load = float(speed.get("model_load_s", 0))
    s1 = NATIVE_B1_S_PER_PAGE
    rows = []
    for scenario in ("low", "high") if batch != 1 else ("high",):
        sb = batched_pace(scenario)
        bench_s = (
            0.0 if batch is not None else load + s1 + BENCH_PAGES * (s1 + BENCH_SIZES_ABOVE_1 * sb)
        )
        smoke_s = load + smoke_pages * s1
        full_s = load + full_pages * (s1 if batch == 1 else sb)
        total = smoke_s + bench_s + full_s
        hours = total / 3600
        cu1, cu2 = _cu(speed, hours)
        rows.append(
            {
                "scenario": scenario if batch != 1 else "batch 1",
                "s_per_page": s1 if batch == 1 else sb,
                "smoke_s": smoke_s,
                "bench_s": bench_s,
                "full_s": full_s,
                "hours": hours,
                "cu_central": cu1,
                "cu_conservative": cu2,
            }
        )
    return rows


def oof_estimate_rows(speed: Mapping[str, Any], n_pages: int, batch: int) -> list[dict[str, Any]]:
    """ESTIMATE rows (low / high pace) of one OOF fold run on a T4 at the native resolution.

    Hours = model load + merge + guard (a warm-up page and 12 bench pages at batch 1 and at
    `batch`, skipped at batch 1) + `n_pages` at `batch`. At batch 1 there is one row (the measured
    sweep pace); otherwise a low row (batch-8 pace x1.047) and a high row (batch-1 pace).
    """
    load = float(speed.get("model_load_s", 0))
    s1 = NATIVE_B1_S_PER_PAGE
    rows = []
    for scenario in ("low", "high") if batch > 1 else ("high",):
        sb = batched_pace(scenario) if batch > 1 else s1
        guard_s = 0.0 if batch == 1 else s1 + BENCH_PAGES * (s1 + sb)
        infer_s = n_pages * sb
        total = load + MERGE_S + guard_s + infer_s
        hours = total / 3600
        cu1, cu2 = _cu(speed, hours)
        rows.append(
            {
                "scenario": scenario if batch > 1 else "batch 1",
                "s_per_page": sb,
                "load_s": load,
                "merge_s": MERGE_S,
                "guard_s": guard_s,
                "infer_s": infer_s,
                "hours": hours,
                "cu_central": cu1,
                "cu_conservative": cu2,
            }
        )
    return rows


def _fmt_cu(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.1f}"


def _basis_lines(speed: Mapping[str, Any]) -> list[str]:
    return [
        f"pace basis: 02 batch-8 pace MEASURED at 1260 tokens {ZS_B8_SECONDS} s / {ZS_PAGES} "
        f"pages = {ZS_B8_S_PER_PAGE:.2f} s/page (GPU not recorded, assumed T4) "
        f"x{NATIVE_FACTOR} (06 sweep native / control cost, MEASURED at batch 1) = "
        f"{NATIVE_B8_S_PER_PAGE:.2f} s/page, ASSUMED "
        "valid only if the native bench accepts batch 8 (VRAM at batch 8 is UNMEASURED at native).",
        f"high = the sweep's MEASURED native batch-1 pace {NATIVE_B1_S_PER_PAGE} s/page (the "
        "fallback when batching is refused); a batch of 2 or 4 lies between low and high.",
        f"model load {speed.get('model_load_s', 0)} s per load "
        f"({speed.get('model_load_s_label', 'ESTIMATE')}); not modelled: logprob overhead, Drive "
        "I/O, restarts, OOM retries.",
        f"CU rate {speed.get('t4_cu_per_hour')}/h: {speed.get('t4_cu_source', 'source unset')}",
        f"CU rate {speed.get('t4_cu_per_hour_conservative')}/h: "
        f"{speed.get('t4_cu_conservative_source', 'source unset')}",
        "CU balance: Colab exposes no programmatic balance; check Runtime -> Manage sessions / "
        "Resources.",
    ]


def format_zs_estimate(
    speed: Mapping[str, Any],
    full_pages: int,
    smoke_pages: int,
    batch: int | None,
    gpu_name: str | None = "T4",
) -> str:
    """Printable 02n estimate (every figure ESTIMATE, UNVERIFIED on a GPU)."""
    rows = zs_estimate_rows(speed, full_pages, smoke_pages, batch)
    bar = "=" * 100
    r1, r2 = speed.get("t4_cu_per_hour"), speed.get("t4_cu_per_hour_conservative")
    out = [
        bar,
        f"ESTIMATE (UNVERIFIED on a GPU) T4 hours / CU: zero-shot {NATIVE_CONFIG_NAME}, "
        f"{full_pages} pages, smoke {smoke_pages} pages, batch "
        f"{'from the bench' if batch is None else batch}",
        bar,
    ]
    if gpu_name is None or "T4" not in gpu_name.upper():
        out.append(
            f"WARNING: runtime GPU is {gpu_name!r}, not a T4 (or unreadable): these are T4 figures."
        )
    out.append(
        f"{'pace':>8} {'s/page':>7} {'smoke s':>8} {'bench s':>8} {'full s':>8} {'T4 h':>6} "
        f"{'CU@' + str(r1):>9} {'CU@' + str(r2):>9}"
    )
    for r in rows:
        out.append(
            f"{r['scenario']:>8} {r['s_per_page']:7.1f} {r['smoke_s']:8.0f} {r['bench_s']:8.0f} "
            f"{r['full_s']:8.0f} {r['hours']:6.2f} {_fmt_cu(r['cu_central']):>9} "
            f"{_fmt_cu(r['cu_conservative']):>9}"
        )
    out += _basis_lines(speed)
    out.append(bar)
    return "\n".join(out)


def format_oof_estimate(
    rows: Sequence[Mapping[str, Any]],
    speed: Mapping[str, Any],
    n_pages: int,
    batch: int,
    fold: int,
) -> str:
    """Printable 05n estimate for one fold (every figure ESTIMATE, UNVERIFIED on a GPU)."""
    bar = "=" * 100
    r1, r2 = speed.get("t4_cu_per_hour"), speed.get("t4_cu_per_hour_conservative")
    out = [
        bar,
        f"ESTIMATE (UNVERIFIED on a GPU) T4 hours / CU: OOF fold {fold} at the native resolution, "
        f"{n_pages} held-out pages, batch {batch}",
        bar,
        f"{'pace':>8} {'s/page':>7} {'load s':>7} {'merge s':>8} {'guard s':>8} {'infer h':>8} "
        f"{'T4 h':>6} {'CU@' + str(r1):>9} {'CU@' + str(r2):>9}",
    ]
    for r in rows:
        out.append(
            f"{r['scenario']:>8} {r['s_per_page']:7.1f} {r['load_s']:7.0f} {r['merge_s']:8.0f} "
            f"{r['guard_s']:8.0f} {r['infer_s'] / 3600:8.2f} {r['hours']:6.2f} "
            f"{_fmt_cu(r['cu_central']):>9} {_fmt_cu(r['cu_conservative']):>9}"
        )
    out += _basis_lines(speed)
    out.append(f"merge {MERGE_S:.0f} s ASSUMED; the guard at batch > 1 is skipped at batch 1.")
    out.append(bar)
    return "\n".join(out)


# --------------------------------------------------------------------------------------------
# The resolution gate (fails closed)
# --------------------------------------------------------------------------------------------


def _load_resmatch() -> Callable[[Any, Any], tuple[bool, str]]:
    """``shipdoc.resmatch.training_resolution_matches``; ImportError when it does not exist."""
    mod = importlib.import_module("shipdoc.resmatch")
    return mod.training_resolution_matches  # type: ignore[no-any-return]


def check_adapter_resolution(
    manifest: Mapping[str, Any] | None,
    cfg: Any,
    matcher: Callable[[Any, Any], tuple[bool, str]] | None = None,
) -> tuple[bool, str]:
    """(ok, reason): may this adapter run under the native inference config?

    Refuses, with the reason, when (1) the inference config is not the native resolution (05n is
    native only; the 1260-token notebook is 05), (2) the manifest is missing, (3)
    ``shipdoc.resmatch`` cannot be imported (FAIL CLOSED: an unavailable check is a refusal),
    (4) the matcher raises, returns something that is not ``(bool, str)``, or says no.
    `matcher(manifest, {"max_pixels": <inference value>})` is injectable for tests; by default
    ``shipdoc.resmatch.training_resolution_matches``.
    """
    got = getattr(getattr(cfg, "backend", None), "max_pixels", None)
    if got != NATIVE_MAX_PIXELS:
        return False, (
            f"inference config max_pixels is {got}, not the native {NATIVE_MAX_PIXELS}: this "
            "notebook runs only the native config"
        )
    if not isinstance(manifest, Mapping) or not manifest:
        return False, "adapter manifest missing or unreadable"
    if matcher is None:
        try:
            matcher = _load_resmatch()
        except ImportError as exc:
            return False, f"shipdoc.resmatch is not available ({exc}); refusing (fail closed)"
    try:
        res = matcher(manifest, {"max_pixels": got})  # resmatch reads a mapping, not a config
    except Exception as exc:  # fail closed: ANY error in the check is a refusal
        return False, f"resolution check raised {type(exc).__name__}: {str(exc)[:200]}"
    if not (
        isinstance(res, tuple)
        and len(res) == 2
        and isinstance(res[0], bool)
        and isinstance(res[1], str)
    ):
        return False, f"resolution check returned an unexpected answer ({type(res).__name__})"
    ok, reason = res
    return ok, reason or ("training resolution matches" if ok else "resolution mismatch")


def assert_adapter_resolution(
    adapter_dir: Path,
    cfg: Any,
    matcher: Callable[[Any, Any], tuple[bool, str]] | None = None,
    out: Callable[[str], None] = print,
) -> None:
    """Print the resolution check row and raise `NativeError` when it refuses."""
    mpath = Path(adapter_dir) / "manifest.json"
    manifest: Any = None
    if mpath.is_file():
        try:
            manifest = json.loads(mpath.read_text(encoding="utf-8"))
        except ValueError:
            manifest = None
    ok, reason = check_adapter_resolution(manifest, cfg, matcher)
    out(f"{'PASS' if ok else 'FAIL'}  RESOLUTION CHECK (adapter training vs inference): {reason}")
    if not ok:
        raise NativeError(f"resolution check refused the adapter: {reason}")


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def _estimate_oof_main(a: argparse.Namespace) -> int:
    from shipdoc import oof, paths

    speed = json.loads((paths.REPO_ROOT / "configs" / "spike_speed.json").read_text("utf-8"))
    folds = json.loads(Path(a.folds).read_text(encoding="utf-8"))
    data_root = Path(a.data_root) if a.data_root else paths.data_dir()
    spec = importlib.util.spec_from_file_location(
        "gpu_estimate", paths.REPO_ROOT / "scripts" / "gpu_estimate.py"
    )
    if spec is None or spec.loader is None:
        raise NativeError("scripts/gpu_estimate.py not found")
    ge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ge)
    held = oof.fold_heldout_ids(folds, a.fold)
    n_pages = 0
    for prefix in sorted({d.split("_")[0] for d in held}):
        ids = [d for d in held if d.startswith(prefix + "_")]
        n_pages += ge.count_pages(ids, data_root / prefix / "labels")
    if a.zs_run_dir is not None:
        chosen = oof.resolve_batch_size(Path(a.zs_run_dir), a.batch_size)
    elif a.batch_size is not None:
        chosen = {"batch_size": a.batch_size, "source": "manual"}
    else:
        raise NativeError("pass --zs-run-dir (the 02n run) or --batch-size")
    rows = oof_estimate_rows(speed, n_pages, int(chosen["batch_size"]))
    print(format_oof_estimate(rows, speed, n_pages, int(chosen["batch_size"]), a.fold))
    print(
        f"documents {len(held)}, pages {n_pages} (counted from the label files), batch size "
        f"{chosen['batch_size']} ({chosen['source']})."
    )
    return 0


def _gated_oof_main(stage: str, rest: list[str]) -> int:
    """Resolution gate, then the unchanged ``shipdoc oof <stage>`` (verify | infer)."""
    from shipdoc import cli
    from shipdoc.spike import load_config

    gate = argparse.ArgumentParser(add_help=False)
    gate.add_argument("--config", type=Path, required=True)
    gate.add_argument("--adapter-dir", type=Path, required=True)
    g, _ = gate.parse_known_args(rest)
    try:
        assert_adapter_resolution(g.adapter_dir, load_config(g.config))
    except NativeError as exc:
        print(f"OOF REFUSED: {exc}")
        return 1
    return int(cli.main(["oof", stage, *rest]))


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m shipdoc.nativerun verify|infer ...`` or ``estimate-oof ...``."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in ("verify", "infer"):
        return _gated_oof_main(args[0], args[1:])
    ap = argparse.ArgumentParser(prog="shipdoc.nativerun", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("estimate-oof", help="ESTIMATE of T4 hours / CU of one native OOF fold.")
    p.add_argument("--fold", type=int, required=True, choices=[0, 1, 2])
    p.add_argument("--zs-run-dir", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--folds", type=Path, default=ROOT / "splits" / "folds.json")
    p.add_argument("--data-root", type=Path, default=None)
    sub.add_parser("verify", help="resolution gate + shipdoc oof verify (same flags).")
    sub.add_parser("infer", help="resolution gate + shipdoc oof infer (same flags).")
    a = ap.parse_args(args)
    return _estimate_oof_main(a)


if __name__ == "__main__":
    sys.exit(main())
