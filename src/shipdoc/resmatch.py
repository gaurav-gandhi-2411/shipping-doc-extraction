"""Training-resolution vs inference-resolution check for an adapter manifest (stdlib only).

An adapter fine-tuned at ``max_pixels`` X must be run at X: the visual-token count per page is the
only thing the resolution changes, and a LoRA fitted to 2,145 image tokens is not the model that
was validated when it is fed 1,260 (or the reverse). ``shipdoc.train.inference_keys`` records the
training ``max_pixels`` in ``<run>/final/manifest.json`` under ``inference_keys``; the notebooks
that load an adapter (05 OOF, the dev / test predictors) call `training_resolution_matches` with
that manifest and the inference config and refuse to run on a mismatch.

Fails closed: a manifest without a readable ``max_pixels`` (an adapter written before the key was
recorded, a hand-edited file, a bool or a string) is a mismatch, never a pass.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

KEY = "max_pixels"


def _as_pixels(value: Any) -> int | None:
    """`value` as a positive int, or None (bool and non-integral floats are not pixel counts)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float) and value.is_integer() and value > 0:
        return int(value)
    return None


def training_resolution_matches(
    manifest: Mapping[str, Any] | None, inference_cfg: Mapping[str, Any] | None
) -> tuple[bool, str]:
    """``(ok, reason)``: does the adapter's training ``max_pixels`` equal the inference one?

    `manifest` is the parsed ``final/manifest.json`` (``inference_keys.max_pixels`` is read);
    `inference_cfg` is any mapping with a top-level ``max_pixels`` (the production inference config
    as a dict, or ``{"max_pixels": cfg.max_pixels}``). Equal passes; any difference fails with both
    values in the message; a missing or malformed value on EITHER side fails closed.
    """
    keys = (manifest or {}).get("inference_keys")
    trained_raw = keys.get(KEY) if isinstance(keys, Mapping) else None
    trained = _as_pixels(trained_raw)
    if trained is None:
        return False, (
            f"manifest has no usable inference_keys.{KEY} (found {trained_raw!r}): the training "
            "resolution of this adapter is unknown, so it cannot be checked against inference"
        )
    run_raw = (inference_cfg or {}).get(KEY)
    run = _as_pixels(run_raw)
    if run is None:
        return False, f"inference config has no usable {KEY} (found {run_raw!r})"
    if trained != run:
        return False, (
            f"resolution mismatch: the adapter was trained at {KEY} {trained} but inference is "
            f"configured at {KEY} {run}; refusing (run it at {trained} or use the adapter "
            f"trained at {run})"
        )
    return True, f"{KEY} {trained} matches between training and inference"
