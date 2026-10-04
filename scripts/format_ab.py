"""Keyed vs compact format A/B for one model x arm (spec Phase 2.3b).

Reads two finished run dirs on the same docs, prints (and writes with ``--out``) one JSON object
``{model, arm, keyed, compact, deltas, decision, reason}``; ``--table`` prints a readable
table (instead of the JSON) on stdout. Exit code 2 when the runs are not comparable (different
docs / missing artifacts): fail closed, never a silent default decision.

Run: python -m scripts.format_ab --compact-dir <run> --keyed-dir <run> --model qwen35_4b \
         --arm img_only [--split dev] [--out ab.json] [--table]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from shipdoc import eval as ev
from shipdoc import paths
from shipdoc.format_ab import compare_formats, format_table


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--compact-dir", type=Path, required=True)
    ap.add_argument("--keyed-dir", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--gold-dir", type=Path, default=None, help="default: <data>/<split>/labels")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--table", action="store_true", help="print the readable table instead of the JSON"
    )
    args = ap.parse_args(argv)
    gold = ev.load_gold(args.gold_dir or paths.data_dir() / args.split / "labels")
    try:
        res = compare_formats(args.compact_dir, args.keyed_dir, gold, args.model, args.arm)
    except (ValueError, FileNotFoundError) as e:
        print(f"format_ab: cannot compare: {e}", file=sys.stderr)
        return 2
    text = json.dumps(res, indent=1)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(format_table(res) if args.table else text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
