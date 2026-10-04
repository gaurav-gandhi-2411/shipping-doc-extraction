"""Step-keyed ``metrics.jsonl`` of a training run: idempotent writer, strict reader, repair.

Why this exists: the first L4 smoke (``ft_smoke_b10d810_bf16``) reported "optimizer steps 29 / 20".
Session 1 was killed after step 9, before the first checkpoint (step 10), so session 2 correctly
started at step 0 - but the trainer only rewrote ``metrics.jsonl`` when a checkpoint was restored,
so session 2's 20 lines were appended to session 1's 9 (steps 1..9, then 1..20). The training state
was right; the file was not. The invariant enforced here: the file holds exactly the steps
``1..n`` once each, in order, whatever the kill history.

Stdlib only, so the notebook's banner cell can load this single file without installing anything.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class MetricsAnomaly(ValueError):
    """``metrics.jsonl`` breaks the "steps 1..n, each once, in order" invariant."""


@dataclass(frozen=True)
class MetricsRead:
    """Result of `read_metrics`: the valid rows plus what had to be dropped to get them."""

    rows: list[dict[str, Any]]
    n_lines: int
    problems: list[str] = field(default_factory=list)

    @property
    def repaired(self) -> bool:
        """True when the file on disk was NOT already clean (rows were dropped or reordered)."""
        return bool(self.problems)


def parse_lines(text: str) -> list[dict[str, Any]]:
    """JSON objects of the non-blank lines; a line without an integer ``step`` is an anomaly."""
    rows = []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise MetricsAnomaly(f"line {n} is not JSON: {exc}") from exc
        if not isinstance(row, dict) or not isinstance(row.get("step"), int):
            raise MetricsAnomaly(f"line {n} has no integer 'step' key")
        rows.append(row)
    return rows


def problems_of(rows: Sequence[Mapping[str, Any]], total: int | None = None) -> list[str]:
    """Every violation of "steps 1..n once each in order" (and ``n <= total``), as text."""
    out: list[str] = []
    steps = [int(r["step"]) for r in rows]
    for i, s in enumerate(steps):
        if s != i + 1:
            out.append(f"line {i + 1} has step {s}, expected {i + 1}")
            break
    dup = sorted({s for s in steps if steps.count(s) > 1})
    if dup:
        out.append(f"{len(dup)} step id(s) occur more than once, e.g. {dup[:5]}")
    if total is not None and len(rows) > total:
        out.append(f"{len(rows)} lines for a {total}-step run")
    return out


def repair_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Last writer wins: a row whose step is <= the last kept step discards that step and all
    later ones first (a restart replays from there), then is kept. For the real anomaly (steps
    1..9, then 1..20) this returns the 20 rows of the second session."""
    kept: list[dict[str, Any]] = []
    for r in rows:
        s = int(r["step"])
        while kept and int(kept[-1]["step"]) >= s:
            kept.pop()
        kept.append(dict(r))
    return kept


def read_metrics(path: Path, total: int | None = None, *, repair: bool = False) -> MetricsRead:
    """Read ``metrics.jsonl``. Strict by default: a duplicated, out-of-order or surplus step raises
    `MetricsAnomaly`. ``repair=True`` applies `repair_rows` (and still raises if the result is not
    steps 1..n, e.g. a gap, or if it still has more than `total` rows). A missing file is empty."""
    if not path.is_file():
        return MetricsRead([], 0)
    rows = parse_lines(path.read_text(encoding="utf-8"))
    problems = problems_of(rows, total)
    if not problems:
        return MetricsRead(rows, len(rows))
    if not repair:
        raise MetricsAnomaly(
            f"{path.name}: {'; '.join(problems)} (a restart without checkpoint appended to an old "
            "file? read it with repair=True)"
        )
    fixed = repair_rows(rows)
    left = problems_of(fixed, total)
    if left:
        raise MetricsAnomaly(f"{path.name}: not repairable: {'; '.join(left)}")
    return MetricsRead(fixed, len(rows), problems)


def _dumps(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(dict(r)) + "\n" for r in rows)


def _atomic_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


def reset_metrics(
    path: Path, rows: Sequence[Mapping[str, Any]], discard_log: Path | None = None
) -> list[dict[str, Any]]:
    """Make the file hold exactly `rows` (steps 1..n). Whatever was on disk and is not identical
    to `rows` is appended to `discard_log` (when given) with ``discarded_at_step`` = n, so a
    restart never silently destroys evidence. Returns the discarded rows."""
    keep = {int(r["step"]): dict(r) for r in rows}
    old: list[dict[str, Any]] = []
    if path.is_file():
        try:
            old = parse_lines(path.read_text(encoding="utf-8"))
        except MetricsAnomaly:
            old = []  # unparseable debris: replaced, not preserved row by row
    n = len(rows)
    dropped = [r for i, r in enumerate(old) if i >= n or r != keep.get(int(r["step"]))]
    if discard_log is not None and dropped:
        with discard_log.open("a", encoding="utf-8", newline="\n") as f:
            f.write(_dumps({**r, "discarded_at_step": n} for r in dropped))
    _atomic_text(path, _dumps(rows))
    return dropped


def append_metric(path: Path, rec: Mapping[str, Any]) -> None:
    """Append step ``rec["step"]`` idempotently: writing the same step again replaces it (and
    drops any later ones); the next step appends; a gap raises `MetricsAnomaly`."""
    step = int(rec["step"])
    rows = parse_lines(path.read_text(encoding="utf-8")) if path.is_file() else []
    last = int(rows[-1]["step"]) if rows else 0
    if step == last + 1:
        with path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(_dumps([rec]))
        return
    if step <= last:
        _atomic_text(path, _dumps([*[r for r in rows if int(r["step"]) < step], rec]))
        return
    raise MetricsAnomaly(f"step {step} after step {last}: a gap would break the 1..n invariant")
