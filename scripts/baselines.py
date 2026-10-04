"""Sanity baselines: write predictions under runs/baselines/, score with wrapper AND CLI.

Run: uv run python scripts/baselines.py   (writes reports/baselines.md)
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from shipdoc import eval as ev

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "baselines"
SPLITS = ("dev", "train")
NAMES = ("gold", "empty", "all_null", "doctype_only")


def make_baselines(gold: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The four baseline submissions for one split."""
    sc = ev.load_scorer()
    return {
        "gold": {
            d: {"doc_type": g["doc_type"], "header": g["header"], "line_items": g["line_items"]}
            for d, g in gold.items()
        },
        "empty": {},
        "all_null": {
            d: {
                "doc_type": g["doc_type"],
                "header": {k: None for k in sc.HEADER[g["doc_type"]]},
                "line_items": [],
            }
            for d, g in gold.items()
        },
        "doctype_only": {
            d: {"doc_type": g["doc_type"], "header": {}, "line_items": []} for d, g in gold.items()
        },
    }


def run_cli(split: str, pred_path: Path, out_json: Path) -> tuple[str, float]:
    """Run the unmodified CLI; returns (stdout, printed OVERALL)."""
    cmd = [
        sys.executable, "assignment/score.py", "--gold", f"data/{split}/labels",
        "--pred", str(pred_path.relative_to(ROOT)), "--out", str(out_json),
    ]  # fmt: skip
    r = subprocess.run(
        cmd, cwd=ROOT, env={**os.environ, "PYTHONUTF8": "1"}, capture_output=True, text=True,
        encoding="utf-8", check=True,
    )  # fmt: skip
    m = re.search(r"^OVERALL\s+([\d.]+)", r.stdout, re.M)
    assert m, r.stdout
    return r.stdout, float(m.group(1))


def non_ascii_labels(split: str) -> int:
    n = 0
    for f in (ROOT / "data" / split / "labels").glob("*.json"):
        if not f.read_bytes().isascii():
            n += 1
    return n


def main() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    md: list[str] = [
        "# Sanity baselines\n\n",
        "Produced by `uv run python scripts/baselines.py`. Metrics in percent; brackets are "
        "doc-level bootstrap 95% CIs (n=2000, seed 42). `CLI OVERALL` is the printed value of the "
        "unmodified `assignment/score.py` (run with `PYTHONUTF8=1`); `parity` = wrapper report "
        "equals the CLI `--out` JSON exactly.\n",
    ]
    facts: list[str] = []
    for split in SPLITS:
        gold = ev.load_gold(ROOT / "data" / split / "labels")
        na = non_ascii_labels(split)
        facts.append(f"{split}: {na}/{len(gold)} label files contain non-ASCII bytes")
        md.append(f"\n## {split} ({len(gold)} docs)\n\n")
        md.append(
            "| baseline | OVERALL | header acc | row F1 | fully correct | false-fill | "
            "CLI OVERALL | parity |\n|---|---|---|---|---|---|---:|---|\n"
        )
        extra: list[str] = []
        for name, pred in make_baselines(gold).items():
            p = RUNS / f"{split}_{name}.json"
            p.write_text(json.dumps(pred), encoding="utf-8")
            stdout, cli_overall = run_cli(split, p, RUNS / f"{split}_{name}.cli.json")
            cli_report = json.loads((RUNS / f"{split}_{name}.cli.json").read_text(encoding="utf-8"))
            rep = ev.score(pred, gold)
            cis = ev.confidence_intervals(pred, gold)["all"]
            cells = [ev._fmt_ci(cis[k]) for k in ev.METRICS]
            parity = "exact" if rep == cli_report else "MISMATCH"
            md.append(f"| {name} | {' | '.join(cells)} | {cli_overall:.2f} | {parity} |\n")
            print(f"[{split}/{name}] wrapper OVERALL {100 * rep['all']['OVERALL']:.2f} "
                  f"CLI {cli_overall:.2f} parity {parity}")  # fmt: skip
            A = rep["all"]
            extra.append(
                f"- {name}: doc_type acc {100 * A['doc_type_accuracy']:.2f}, "
                f"illegible fields {A['illegible_fields']}, rows {A['line_item_rows']}"
            )
        a_null = ev.score(make_baselines(gold)["all_null"], gold)["all"]
        n_fields = sum(len(ev.load_scorer().HEADER[g["doc_type"]]) for g in gold.values())
        md.append(
            f"\nGold null rate = {a_null['illegible_fields']}/{n_fields} header fields = "
            f"{100 * a_null['illegible_fields'] / n_fields:.2f}%; all-null header accuracy = "
            f"{100 * a_null['header_field_accuracy']:.2f}%.\n\n" + "\n".join(extra) + "\n"
        )
    md.append(
        "\n## Why each baseline scores what it does\n\n"
        "- **gold**: identical to the labels, so every component is 100 (harness sanity check).\n"
        "- **empty `{}`**: every document is missing, so doc_type is wrong and rows are all "
        "missing (row F1 0); header accuracy is nonzero only because null-gold fields are "
        '"correct" when empty, and no document is fully correct unless the type matches.\n'
        "- **all-null**: right doc_type, every header field null, no rows: it is right exactly "
        "on the gold-null fields, so header accuracy equals the gold null rate, a useful "
        '"always abstain" floor; row F1 is 0 (no rows predicted) and nothing is a false fill.\n'
        "- **doc_type-only**: header `{}` reads as all-null to the scorer, so it scores "
        "identically to all-null; the doc_type is not itself a scored component beyond "
        "the fully-correct condition.\n"
    )
    md.append("\n## Encoding\n\n" + "\n".join(f"- {f}" for f in facts) + "\n")
    out = ROOT / "reports" / "baselines.md"
    out.write_text("".join(md), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
