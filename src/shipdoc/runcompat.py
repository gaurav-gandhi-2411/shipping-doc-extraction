"""Refuse to compare runs made at different input resolutions (1260-token vs native pages).

A fine-tuned OOF run at native resolution (``max_pixels`` 2196480) compared with a zero-shot run
at 1260 tokens (``max_pixels`` 1310720) measures the resolution change, not the fine-tune. The run
manifest records ``config`` {name, hash}; the hash covers the whole config YAML (``max_pixels``
included), so equal hashes imply equal resolution and different hashes are refused (fail closed:
a manifest without a hash is refused too). ``max_pixels`` itself is not in the manifest; it is
resolved from the matching ``configs/spike_*.yaml`` only to make the refusal message readable.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from shipdoc import paths
from shipdoc.runmeta import read_manifest


class ResolutionMismatchError(ValueError):
    """Two runs do not share one config hash (resolution); messages never quote a value."""


def manifest_config_hash(manifest: Mapping[str, Any] | None) -> str | None:
    """``config.hash`` of a run manifest, or None when absent / malformed."""
    cfg = (manifest or {}).get("config")
    h = cfg.get("hash") if isinstance(cfg, Mapping) else None
    return h if isinstance(h, str) and h else None


def resolve_max_pixels(
    manifest: Mapping[str, Any] | None, configs_dir: Path | None = None
) -> int | None:
    """``max_pixels`` of the config a manifest names (matched by name AND hash), else None."""
    cfg = (manifest or {}).get("config")
    if not isinstance(cfg, Mapping):
        return None
    mp = cfg.get("max_pixels")  # a future manifest may record it directly
    if isinstance(mp, int) and not isinstance(mp, bool):
        return mp
    from shipdoc.spike import load_config  # lazy: heavy imports only when a message is built

    for p in sorted((configs_dir or paths.REPO_ROOT / "configs").glob("spike_*.yaml")):
        try:
            c = load_config(p)
        except (ValueError, OSError, KeyError, TypeError):
            continue
        if c.name == cfg.get("name") and c.config_hash == cfg.get("hash"):
            return c.backend.max_pixels
    return None


def _describe(label: str, manifest: Mapping[str, Any] | None, configs_dir: Path | None) -> str:
    cfg = (manifest or {}).get("config")
    cfg = cfg if isinstance(cfg, Mapping) else {}
    mp = resolve_max_pixels(manifest, configs_dir)
    return (
        f"{label}: config `{cfg.get('name')}` hash `{cfg.get('hash')}` "
        f"max_pixels {mp if mp is not None else 'unresolved'}"
    )


def assert_same_resolution(
    run_a_manifest: Mapping[str, Any] | None,
    run_b_manifest: Mapping[str, Any] | None,
    label_a: str = "run A",
    label_b: str = "run B",
    configs_dir: Path | None = None,
) -> None:
    """Raise `ResolutionMismatchError` unless both manifests carry the same ``config.hash``."""
    ha, hb = manifest_config_hash(run_a_manifest), manifest_config_hash(run_b_manifest)
    if ha is None or hb is None or ha != hb:
        raise ResolutionMismatchError(
            "refusing a mixed-resolution comparison (config hash must be equal and present): "
            f"{_describe(label_a, run_a_manifest, configs_dir)}; "
            f"{_describe(label_b, run_b_manifest, configs_dir)}"
        )


def assert_runs_share_resolution(runs: Mapping[str, Path], configs_dir: Path | None = None) -> None:
    """`assert_same_resolution` of every run folder in `runs` (label -> folder) against the first.

    A folder identical to the first is skipped (plumbing checks use one run on both sides).
    Raises `ResolutionMismatchError` (also when a folder has no manifest).
    """
    items = list(runs.items())
    if len(items) < 2:
        return
    first_label, first_dir = items[0]
    first = read_manifest(Path(first_dir)) if Path(first_dir).is_dir() else None
    for label, d in items[1:]:
        if Path(d).resolve() == Path(first_dir).resolve():
            continue  # the same folder on both sides (a plumbing check) is trivially one resolution
        man = read_manifest(Path(d)) if Path(d).is_dir() else None
        assert_same_resolution(
            first, man, f"{first_label} `{Path(first_dir).name}`", f"{label} `{Path(d).name}`",
            configs_dir,
        )  # fmt: skip
