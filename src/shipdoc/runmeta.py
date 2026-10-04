"""Run manifest (``<run>/manifest.json``): what a run was, so resume, merge and comparisons can
refuse to mix runs that are not the same experiment. Stdlib only.

Keys: ``schema``, ``run_id``, ``base_run_id`` (the unsharded id), ``code_sha`` (HEAD, ``+dirty``
when the tree has tracked changes), ``model`` {id, revision}, ``config`` {name, hash}, ``seed``,
``batch_size`` (pages per generate call; 1 = the unbatched path), ``batch_size_source``
(``bench`` | ``manual`` | ``default``), ``bench`` (null, or {path, chosen, deviation}), ``split``,
``arm``, ``output_format``, ``logprobs``, ``shard`` (``i/K``), ``docs_sha`` (sha256 of the full,
pre-shard doc list) and ``n_docs`` (docs of THIS run).

Rules (`check_resume`): a resume must reuse the stored batch size, config hash, model revision and
shard; a different value raises instead of silently changing the experiment. A folder with a
trace but no manifest is a pre-manifest run: those were all batch size 1.

Determinism note (`require_same_batch_size`): batched greedy decoding is not guaranteed
bit-identical across batch sizes, so a test submission is only comparable to a dev run produced
with the SAME batch size; 04_predict_test must call it before comparing.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

MANIFEST_NAME = "manifest.json"
SCHEMA = 1
#: Fields that must agree between a run and its resume.
RESUME_KEYS = (("config", "hash"), ("model", "revision"), ("shard",), ("batch_size",))


def docs_sha(doc_ids: Sequence[str]) -> str:
    """sha256 (first 16 hex) of the ordered doc list."""
    return hashlib.sha256("\n".join(doc_ids).encode("utf-8")).hexdigest()[:16]


def _get(m: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    cur: Any = m
    for k in path:
        cur = cur.get(k) if isinstance(cur, Mapping) else None
    return cur


def write_manifest(run_dir: Path, manifest: Mapping[str, Any]) -> None:
    """Atomic write (temp file + replace)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / MANIFEST_NAME
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def read_manifest(run_dir: Path) -> dict[str, Any] | None:
    """The manifest, or None when the folder has none."""
    path = Path(run_dir) / MANIFEST_NAME
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: manifest must be a JSON object")
    return data


def check_resume(
    stored: Mapping[str, Any] | None, new: Mapping[str, Any], has_trace: bool
) -> dict[str, Any]:
    """Manifest to continue with. Raises ValueError when `new` contradicts `stored`.

    `new["batch_size"]` may be None ("use the stored one"): the result then carries the stored size
    (or 1 for a pre-manifest run). A pre-manifest folder with a trace counts as batch size 1.
    """
    out = dict(new)
    if stored is None:
        if out.get("batch_size") is None:
            out["batch_size"] = 1
        if has_trace and out["batch_size"] != 1:
            raise ValueError(
                f"the existing trace has no manifest (a pre-manifest run, batch size 1) but "
                f"batch_size {out['batch_size']} was requested; resume with batch size 1"
            )
        return out
    if out.get("batch_size") is None:
        out["batch_size"] = stored.get("batch_size", 1)
        out["batch_size_source"] = stored.get("batch_size_source", "default")
        out["bench"] = stored.get("bench")
    for path in RESUME_KEYS:
        a, b = _get(stored, path), _get(out, path)
        if a != b:
            name = ".".join(path)
            raise ValueError(
                f"cannot resume: {name} is {a!r} in the stored manifest but {b!r} now; the batch "
                "size, config, model revision and shard of a run never change on resume "
                "(use a new --run-id for a different experiment)"
            )
    return out


def require_same_batch_size(dev_run_dir: Path, test_run_dir: Path) -> int:
    """Assert that two runs used the same batch size; returns it.

    Raises ValueError when they differ or either manifest is missing (a missing manifest means
    the batch size is unknown, which is refused, not assumed). Call before comparing a test
    submission with a dev run.
    """
    sizes = []
    for d in (dev_run_dir, test_run_dir):
        m = read_manifest(Path(d))
        if m is None or not isinstance(m.get("batch_size"), int):
            raise ValueError(f"{d}: no manifest with a batch_size; cannot compare runs")
        sizes.append(m["batch_size"])
    if sizes[0] != sizes[1]:
        raise ValueError(
            f"batch size differs (dev {sizes[0]}, test {sizes[1]}): greedy outputs are only "
            "comparable at the same batch size; rerun one of them with the other's"
        )
    return sizes[0]
