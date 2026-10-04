"""Scan the WHOLE git history for confidential label values (publish gate, spec Phase 8).

``tests/test_no_label_leak.py`` only looks at the files tracked right now. A value that was
committed once and deleted later still sits in the history, so before anything is published this
script reads every commit of the given revisions (``git log -p``: added AND deleted lines) and
reports:

* label values from ``<labels-root>/*/labels/*.json`` (train + dev): the header kinds
  ``supplier_name carrier shipper_name consignee_name buyer_name ship_to_name`` and the identifier
  kinds ``invoice_number awb_number mawb hawb supplier_part_number customer_part_number
  purchase_order``, only values of length >= ``--min-len`` (default 6), same normalisation as the
  test (``strip()``, strings only, ISO dates dropped, exact case-sensitive substring match);
* any path under ``assignment/ data/ cache/ runs/`` and any ``zip png jpg jpeg pdf`` file that
  appears in any commit's diff (a file present in a tree was added by some commit of the range, so
  scanning the diff headers covers every tree).

Output: counts and ``(commit sha, path)`` pairs only. The matched VALUES are never printed (nor the
matched line). Exit code 1 if anything is found, 0 if clean, 2 on a usage or git error.

Complexity: one ``git log -p`` stream; an Aho-Corasick automaton over all values is built once in
O(total pattern characters) and each diff line is scanned in O(line length + matches), so the whole
scan is O(total diff bytes) independent of the number of values (the naive per-value ``in`` check
is O(bytes x values)).

Run: uv run python scripts/history_leak_scan.py [--repo DIR] [--labels-root DIR] [REV ...]
(no REV: ``--all``). Scanning a repo other than this one (e.g. the fresh squashed repo) keeps
``--labels-root`` pointing at this repo's ``data/``.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import deque
from collections.abc import Iterable, Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
HEADER_KINDS = (
    "supplier_name",
    "carrier",
    "shipper_name",
    "consignee_name",
    "buyer_name",
    "ship_to_name",
    "invoice_number",
    "awb_number",
    "mawb",
    "hawb",
)
ROW_KINDS = ("supplier_part_number", "customer_part_number", "purchase_order")
DEFAULT_MIN_LEN = 6
FORBIDDEN_DIRS = ("assignment/", "data/", "cache/", "runs/")
FORBIDDEN_SUFFIXES = (".zip", ".png", ".jpg", ".jpeg", ".pdf")
COMMIT_MARK = "@@COMMIT@@"


def collect_values(labels_root: Path, min_len: int = DEFAULT_MIN_LEN) -> dict[str, set[str]]:
    """Distinct train+dev gold value -> set of kinds (``header.x`` / ``row.x``)."""
    out: dict[str, set[str]] = {}
    for split in ("train", "dev"):
        for f in sorted((labels_root / split / "labels").glob("*.json")):
            d = json.loads(f.read_text(encoding="utf-8"))
            pairs = [(f"header.{k}", d.get("header", {}).get(k)) for k in HEADER_KINDS]
            for row in d.get("line_items", []):
                pairs += [(f"row.{k}", row.get(k)) for k in ROW_KINDS]
            for kind, v in pairs:
                if not isinstance(v, str):
                    continue
                v = v.strip()
                if len(v) < min_len or ISO_DATE.match(v):
                    continue
                out.setdefault(v, set()).add(kind)
    return out


class AhoCorasick:
    """Multi-pattern substring matcher; ``kinds_in(text)`` returns the kinds that occur."""

    def __init__(self, patterns: dict[str, set[str]]) -> None:
        self._goto: list[dict[str, int]] = [{}]
        self._fail: list[int] = [0]
        self._out: list[frozenset[str]] = [frozenset()]
        for pat, kinds in patterns.items():
            node = 0
            for ch in pat:
                nxt = self._goto[node].get(ch)
                if nxt is None:
                    nxt = len(self._goto)
                    self._goto[node][ch] = nxt
                    self._goto.append({})
                    self._fail.append(0)
                    self._out.append(frozenset())
                node = nxt
            self._out[node] = self._out[node] | frozenset(kinds)
        queue: deque[int] = deque(self._goto[0].values())
        while queue:
            node = queue.popleft()
            for ch, child in self._goto[node].items():
                f = self._fail[node]
                while f and ch not in self._goto[f]:
                    f = self._fail[f]
                self._fail[child] = self._goto[f].get(ch, 0)
                self._out[child] = self._out[child] | self._out[self._fail[child]]
                queue.append(child)

    def kinds_in(self, text: str) -> frozenset[str]:
        """Kinds of every pattern occurring in ``text`` (empty frozenset: none)."""
        found: set[str] = set()
        goto, fail, out = self._goto, self._fail, self._out
        node = 0
        for ch in text:
            while node and ch not in goto[node]:
                node = fail[node]
            node = goto[node].get(ch, 0)
            if out[node]:
                found |= out[node]
        return frozenset(found)


def flagged_path_kind(path: str) -> str | None:
    """Why a path must never be in a published history (None: allowed)."""
    low = path.lower()
    if low.startswith(FORBIDDEN_DIRS):
        return "forbidden_dir"
    if low.endswith(FORBIDDEN_SUFFIXES):
        return "forbidden_type"
    return None


def stream_log(repo: Path, revs: list[str]) -> Iterator[bytes]:
    """Lines of ``git log -p`` (zero context, no rename detection) over the revisions."""
    cmd = [
        "git",
        "-c",
        "core.quotepath=off",
        "log",
        "-p",
        "-U0",
        "--no-color",
        "--no-renames",
        "--no-ext-diff",
        f"--format={COMMIT_MARK}%H",
        *(revs or ["--all"]),
        "--",
    ]
    proc = subprocess.Popen(cmd, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None and proc.stderr is not None
    yield from proc.stdout
    err = proc.stderr.read().decode("utf-8", errors="replace").strip()
    if proc.wait():
        raise RuntimeError(f"git log failed (exit {proc.returncode}): {err}")


def scan_lines(
    lines: Iterable[bytes], matcher: AhoCorasick
) -> tuple[dict[tuple[str, str], set[str]], dict[tuple[str, str], str], int]:
    """Scan a ``git log -p`` stream.

    Returns (value hits: (sha, path) -> kinds, path flags: (sha, path) -> reason, commits seen).
    """
    value_hits: dict[tuple[str, str], set[str]] = {}
    path_hits: dict[tuple[str, str], str] = {}
    sha = path = ""
    commits = 0
    for raw in lines:
        line = raw.decode("utf-8", errors="ignore").rstrip("\r\n")
        if line.startswith(COMMIT_MARK):
            sha, path = line[len(COMMIT_MARK) :], ""
            commits += 1
        elif line.startswith("diff --git "):
            path = line.rsplit(" b/", 1)[-1]
            reason = flagged_path_kind(path)
            if reason:
                path_hits[(sha, path)] = reason
        elif line[:1] in ("+", "-") and not line.startswith(("+++ ", "--- ")):
            kinds = matcher.kinds_in(line[1:])
            if kinds:
                value_hits.setdefault((sha, path), set()).update(kinds)
    return value_hits, path_hits, commits


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("revs", nargs="*", help="revisions / refs to scan (default: --all)")
    ap.add_argument("--repo", type=Path, default=ROOT, help="repository to scan")
    ap.add_argument("--labels-root", type=Path, default=ROOT / "data")
    ap.add_argument("--min-len", type=int, default=DEFAULT_MIN_LEN)
    args = ap.parse_args(argv)

    values = collect_values(args.labels_root, args.min_len)
    if not values:
        print(f"no label values found under {args.labels_root}: cannot scan", file=sys.stderr)
        return 2  # fail closed: an empty value set would pass everything
    try:
        value_hits, path_hits, commits = scan_lines(
            stream_log(args.repo, args.revs), AhoCorasick(values)
        )
    except (RuntimeError, OSError) as exc:
        print(f"scan failed: {exc}", file=sys.stderr)
        return 2
    if commits == 0:
        print("no commits scanned: wrong repo or revisions", file=sys.stderr)
        return 2

    print(f"commits scanned: {commits}; distinct label values: {len(values)} (values withheld)")
    print(f"label-value hits: {len(value_hits)} (commit, path) pairs")
    for (sha, path), kinds in sorted(value_hits.items()):
        print(f"  {sha[:12]} {path} [{'/'.join(sorted(kinds))}]")
    print(f"forbidden-path hits: {len(path_hits)} (commit, path) pairs")
    for (sha, path), reason in sorted(path_hits.items()):
        print(f"  {sha[:12]} {path} [{reason}]")
    return 1 if value_hits or path_hits else 0


if __name__ == "__main__":
    sys.exit(main())
