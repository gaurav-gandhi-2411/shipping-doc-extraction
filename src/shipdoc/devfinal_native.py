"""Notebook 08: the FINAL adapter at the NATIVE resolution on the 100 dev documents.

The native sibling of ``shipdoc.devfinal`` (notebook 05b). GG decision 2026-10-03: native
(``configs/spike_qwen35_4b_img_only_native.yaml``, ``max_pixels`` 2,196,480) is the production
resolution, so the official seen-layout dev number of the final adapter is taken at that
resolution. ``devfinal`` already takes the production config as ``--config`` and is NOT changed;
this module only adds what a native run needs around it, as ``nativerun`` does for the OOF notebook:

* the resolution gate, FAIL CLOSED, on top of ``predict_ft.verify_final_adapter`` (which
  ``devfinal verify`` / ``infer`` still run): `nativerun.assert_adapter_resolution`
  (``shipdoc.resmatch``) refuses an adapter trained at another resolution, with a readable reason;
* the zero-shot run (the 02n run) and, at compare time, the dev run itself must carry the native
  config hash (`shipdoc.runcompat`): a 1260-token zero-shot run is refused, so FT and ZS are never
  compared across resolutions;
* the estimate: ``configs/spike_speed.json`` has no native entry, so the ESTIMATE is
  `nativerun.oof_estimate_rows` (measured 1260-token pace scaled by the measured sweep ratio);
* the compare outputs are stamped with the native config (the shared formatter in ``devfinal``
  still titles them 05b).

Stages (``python -m shipdoc.devfinal_native``): ``plan``, ``estimate``, ``verify``, ``infer``,
``compare``: exactly the flags of ``python -m shipdoc.devfinal`` (it builds on its parser);
``--config`` defaults to the native config and anything else is refused. Reports carry aggregates
only: no document id, no extracted value.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from shipdoc import devfinal, nativerun, oof, paths, predict_ft, runcompat, spike
from shipdoc.runmeta import read_manifest

NOTEBOOK = "08_dev_final"
NATIVE_CONFIG = nativerun.NATIVE_CONFIG
RESOLUTION_NOTE = (
    "NATIVE RESOLUTION (notebook 08, max_pixels {px}, config {name} {hash}): the title below is "
    "the shared devfinal formatter's (05b); the numbers are at native resolution only."
)


class NativeDevError(devfinal.DevFinalError):
    """A native-resolution precondition of the dev run failed. No document value in a message."""


# --------------------------------------------------------------------------------------------
# Gates (all refuse; none is advisory)
# --------------------------------------------------------------------------------------------


def load_native_config(path: Path | str) -> spike.SpikeConfig:
    """The inference config, refused unless it is the native production config.

    Checked: ``max_pixels`` is 2,196,480 and the config name is the native one. The 1260-token
    config (devfinal's own default) is refused with the reason.
    """
    cfg = spike.load_config(Path(path))
    if cfg.backend.max_pixels != nativerun.NATIVE_MAX_PIXELS or cfg.name != (
        nativerun.NATIVE_CONFIG_NAME
    ):
        raise NativeDevError(
            f"config {cfg.name!r} has max_pixels {cfg.backend.max_pixels}: this notebook runs only "
            f"{nativerun.NATIVE_CONFIG_NAME!r} (max_pixels {nativerun.NATIVE_MAX_PIXELS}); pass "
            f"--config {NATIVE_CONFIG}"
        )
    return cfg


def check_run_resolution(run_dir: Path, cfg: spike.SpikeConfig, label: str) -> str:
    """Refuse a run folder whose manifest config hash is not the native config's.

    `runcompat.assert_same_resolution` between the native config (name + hash) and the run's
    manifest; a missing folder, a missing manifest or a manifest without a hash is a refusal too.
    Returns the run's config hash.
    """
    d = Path(run_dir)
    if not d.is_dir():
        raise NativeDevError(f"{label}: {d.name} is not a folder")
    man = read_manifest(d)
    native = {"config": {"name": cfg.name, "hash": cfg.config_hash}}
    try:
        runcompat.assert_same_resolution(native, man, "native config", f"{label} `{d.name}`")
    except runcompat.ResolutionMismatchError as exc:
        raise NativeDevError(str(exc)) from exc
    return str(runcompat.manifest_config_hash(man))


def check_adapter_native(
    adapter_dir: Path, cfg: spike.SpikeConfig, out: Callable[[str], None] = print
) -> None:
    """The resolution gate of `nativerun` (fails closed), printing its PASS / FAIL row."""
    try:
        nativerun.assert_adapter_resolution(Path(adapter_dir), cfg, out=out)
    except nativerun.NativeError as exc:
        raise NativeDevError(str(exc)) from exc


# --------------------------------------------------------------------------------------------
# Estimate
# --------------------------------------------------------------------------------------------


def run_estimate(
    *,
    cfg: spike.SpikeConfig,
    zs_run_dir: Path,
    n_pages: int,
    batch_size: int | None,
    out: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Print the ESTIMATE (T4 hours / CU) of the native dev run; runs no model.

    Load + merge + guard + `n_pages` at the batch size of the 02n run (or `batch_size`), the
    low / high pace rows of `nativerun.oof_estimate_rows`. Every figure is an ESTIMATE.
    """
    check_run_resolution(zs_run_dir, cfg, "zero-shot run")
    speed = json.loads((paths.REPO_ROOT / "configs" / "spike_speed.json").read_text("utf-8"))
    chosen = oof.resolve_batch_size(Path(zs_run_dir), batch_size)
    batch = int(chosen["batch_size"])
    rows = nativerun.oof_estimate_rows(speed, n_pages, batch)
    text = nativerun.format_oof_estimate(rows, speed, n_pages, batch, fold=0)
    head = f"OOF fold 0 at the native resolution, {n_pages} held-out pages"
    if head not in text:
        raise NativeDevError(
            "nativerun.format_oof_estimate changed: the 08 title cannot be derived"
        )
    out(text.replace(head, f"08 final adapter on the dev documents at native, {n_pages} pages"))
    out(
        f"documents {devfinal.EXPECTED_DEV_DOCS}, pages {n_pages} (counted from the label files), "
        f"batch size {batch} ({chosen['source']}). ESTIMATE, UNVERIFIED."
    )
    return {"n_pages": n_pages, "batch": chosen, "rows": rows}


# --------------------------------------------------------------------------------------------
# Compare: gates, the unchanged devfinal compare, then the native stamp
# --------------------------------------------------------------------------------------------


def stamp_compare(out_dir: Path, cfg: spike.SpikeConfig, zs_run_dir: Path) -> None:
    """Add the native config to ``devfinal_compare.json`` and a note to the markdown."""
    out_dir = Path(out_dir)
    jpath = out_dir / devfinal.COMPARE_NAME
    mpath = out_dir / devfinal.COMPARE_MD_NAME
    res = json.loads(jpath.read_text(encoding="utf-8"))
    res["native_resolution"] = {
        "notebook": NOTEBOOK,
        "config": cfg.name,
        "config_hash": cfg.config_hash,
        "max_pixels": cfg.backend.max_pixels,
        "zero_shot_run_config_hash": check_run_resolution(zs_run_dir, cfg, "zero-shot run"),
    }
    spike.atomic_write(jpath, json.dumps(res, indent=1))
    note = RESOLUTION_NOTE.format(px=cfg.backend.max_pixels, name=cfg.name, hash=cfg.config_hash)
    spike.atomic_write(mpath, f"> {note}\n\n" + mpath.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def _with_config(args: Sequence[str]) -> list[str]:
    """`args` with ``--config <native>`` appended when absent (devfinal's own default is 1260)."""
    out = list(args)
    if not any(a == "--config" or a.startswith("--config=") for a in out):
        out += ["--config", str(paths.REPO_ROOT / NATIVE_CONFIG)]
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Stage CLI. Exit 0 = the stage passed; 1 = refused / a check failed (counts only)."""
    args = _with_config(sys.argv[1:] if argv is None else argv)
    a = devfinal.build_parser().parse_args(args)
    try:
        return _dispatch(a, args)
    except (devfinal.DevFinalError, predict_ft.FtError, oof.OofError, nativerun.NativeError) as exc:
        print(f"devfinal_native {a.stage}: REFUSED: {exc}", file=sys.stderr)
        return 1


def _dispatch(a: Any, args: Sequence[str]) -> int:
    if a.stage == "plan":  # no model, no resolution: the unchanged plan
        return int(devfinal.main(args))
    cfg = load_native_config(a.config)
    if a.stage == "estimate":
        paths.apply_env()
        data = Path(a.data_root) if a.data_root else paths.data_dir()
        folds: Mapping[str, Any] = json.loads(Path(a.folds).read_text(encoding="utf-8"))
        ids = devfinal.load_dev_ids(a.dev_docs, a.zs500_docs, folds, data)
        run_estimate(cfg=cfg, zs_run_dir=a.zs_run_dir, n_pages=devfinal.count_pages(ids, data),
                     batch_size=a.batch_size)  # fmt: skip
        return 0
    if a.stage == "compare":
        check_run_resolution(a.zs_run_dir, cfg, "zero-shot run")
        if not a.plumbing_check:
            if a.ft_run_dir is None:
                raise NativeDevError("compare needs --ft-run-dir (or --plumbing-check)")
            check_run_resolution(a.ft_run_dir, cfg, "dev run")
        rc = int(devfinal.main(args))
        if rc == 0:
            stamp_compare(a.out_dir, cfg, a.zs_run_dir)
        return rc
    check_adapter_native(a.adapter_dir, cfg)  # verify | infer: the resolution gate first
    if a.stage == "infer":
        check_run_resolution(a.zs_run_dir, cfg, "zero-shot run")
    return int(devfinal.main(args))


if __name__ == "__main__":
    raise SystemExit(main())
