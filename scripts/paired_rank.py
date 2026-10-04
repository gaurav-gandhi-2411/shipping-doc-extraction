"""Rank finished spike runs by OVERALL and paired-bootstrap the top two (spec Phase 2.3).

Reads ``<runs-dir>/<group>_<config>_<sha7>/predictions.json`` for each config, scores it against
the dev gold restricted to ``--docs``, and prints (and writes with ``--out``) one JSON object:
``{ranking, top2, delta, ci95, p_le_0, cis_overlap, delta_ci_excludes_0, decision, ...}``.

A run whose predictions.json is missing or does not cover every doc in ``--docs`` is listed under
``excluded`` and not ranked (a partial run must never compete on fewer documents). With fewer than
2 usable runs the decision is ``insufficient_runs`` and the exit code is 2 (fail closed).

Run: python -m scripts.paired_rank --runs-dir <dir> --group spike40 --sha7 <sha7> \
         --docs splits/spike40.json --split dev [--configs a b ...] [--out file.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from shipdoc import eval as ev
from shipdoc import paths

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORDER_FILE = ROOT / "configs" / "spike_order.json"


def load_runs(
    runs_dir: Path, group: str, sha7: str, configs: list[str], doc_ids: list[str]
) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    """(config -> predictions restricted to `doc_ids`, excluded runs with a reason)."""
    preds: dict[str, dict[str, Any]] = {}
    excluded: list[dict[str, str]] = []
    for config in configs:
        path = runs_dir / f"{group}_{config}_{sha7}" / "predictions.json"
        if not path.is_file():
            excluded.append({"config": config, "reason": "no predictions.json"})
            continue
        pred = json.loads(path.read_text(encoding="utf-8"))
        missing = [d for d in doc_ids if d not in pred]
        if missing:
            excluded.append(
                {"config": config, "reason": f"incomplete: {len(missing)}/{len(doc_ids)} missing"}
            )
            continue
        preds[config] = {d: pred[d] for d in doc_ids}
    return preds, excluded


def rank(
    runs_dir: Path,
    group: str,
    sha7: str,
    configs: list[str],
    doc_ids: list[str],
    gold: dict[str, dict[str, Any]],
    n: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Load, score and rank; the result always carries ``excluded`` and ``decision``."""
    unknown = [d for d in doc_ids if d not in gold]
    if unknown:
        raise ValueError(f"{len(unknown)} doc ids have no dev gold, e.g. {unknown[:3]}")
    preds, excluded = load_runs(runs_dir, group, sha7, configs, doc_ids)
    if len(preds) < 2:
        return {"decision": "insufficient_runs", "excluded": excluded, "usable": sorted(preds)}
    result = ev.rank_runs(preds, {d: gold[d] for d in doc_ids}, n=n, seed=seed)
    result["excluded"] = excluded
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs-dir", type=Path, default=None, help="default: SHIPDOC_RUNS_DIR")
    ap.add_argument("--group", required=True, help="run-id prefix, e.g. spike40 or dev100")
    ap.add_argument("--sha7", required=True)
    ap.add_argument("--docs", required=True, help="JSON list of doc_ids (the run's doc list)")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--gold-dir", type=Path, default=None, help="default: <data>/<split>/labels")
    ap.add_argument("--configs", nargs="+", default=None, help="default: configs/spike_order.json")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    configs = args.configs or json.loads(DEFAULT_ORDER_FILE.read_text(encoding="utf-8"))["order"]
    runs_dir = args.runs_dir or paths.runs_dir()
    gold = ev.load_gold(args.gold_dir or paths.data_dir() / args.split / "labels")
    doc_ids = json.loads(Path(args.docs).read_text(encoding="utf-8"))
    result = rank(runs_dir, args.group, args.sha7, configs, doc_ids, gold)
    text = json.dumps(result, indent=1)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 2 if result["decision"] == "insufficient_runs" else 0


if __name__ == "__main__":
    sys.exit(main())
