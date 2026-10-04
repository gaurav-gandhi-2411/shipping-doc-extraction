"""RTX 3070 (8 GB, native Windows) 5-doc smoke for the spike configs. DEBUG ONLY.

Runs ``python -m shipdoc spike --limit 5`` on splits/spike40.json for each config, trying fp16
first and falling back to nf4 (fp16 compute) on CUDA OOM, then prints one summary per run.
Outputs go to ``<SHIPDOC_RUNS_DIR>/debug_3070/``. Nothing here is reportable: it is a
local 5-doc crash/format check, not an evaluation (never copy these numbers into reports/).

    uv run python scripts/smoke_local.py [--backend hf|mock] [--config CFG ...]
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from shipdoc import paths  # noqa: E402

LABEL = "RTX 3070 debug - not reportable"
DEFAULT_CONFIGS = (
    "configs/spike_nuextract3_img_only_compact.yaml",
    "configs/spike_qwen3vl_4b_img_only_compact.yaml",  # the compact variant of the 4B config
)
OOM_MARKERS = ("OutOfMemoryError", "out of memory", "CUDA error: out of memory")


def mean(xs: list[float]) -> float | None:
    """Arithmetic mean, None when empty."""
    return sum(xs) / len(xs) if xs else None


def summarize(run_dir: Path, gold_dir: Path, max_new_tokens: int) -> dict[str, Any]:
    """Debug metrics from a run's trace.jsonl and the dev gold labels."""
    traces = [
        json.loads(ln)
        for ln in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    pages = [p for t in traces for p in t["pages"]]
    rows_pred, rows_gold, qty_nonnull, qty_total = [], [], 0, 0
    per_doc = []
    for t in traces:
        pred_rows = (t["prediction"] or {}).get("line_items") or []
        gold = json.loads((gold_dir / f"{t['doc_id']}.json").read_text(encoding="utf-8"))
        gold_rows = gold.get("line_items") or []
        rows_pred.append(len(pred_rows))
        rows_gold.append(len(gold_rows))
        per_doc.append(f"{t['doc_id']}:{len(pred_rows)}/{len(gold_rows)}")
        qty_total += len(pred_rows)
        qty_nonnull += sum(1 for r in pred_rows if r.get("quantity") not in (None, ""))
    meta = [p["meta"] for p in pages]
    vram = [m["peak_vram_bytes"] for m in meta if m.get("peak_vram_bytes")]
    n_out = [m["n_output_tokens"] for m in meta if m.get("n_output_tokens") is not None]
    return {
        "label": LABEL,
        "n_docs": len(traces),
        "n_pages": len(pages),
        "json_valid_rate": mean([float(p["json_valid"]) for p in pages]),
        "rows_pred_mean": mean(rows_pred),
        "rows_gold_mean": mean(rows_gold),
        "rows_pred_over_gold_per_doc": per_doc,
        "quantity_nonnull_share": (qty_nonnull / qty_total) if qty_total else None,
        "pages_at_max_new_tokens": sum(n >= max_new_tokens for n in n_out),
        "s_per_page_mean": mean([m["latency_s"] for m in meta if m.get("latency_s") is not None]),
        "n_output_tokens_mean": mean([float(n) for n in n_out]),
        "peak_vram_gib": (max(vram) / 2**30) if vram else None,
        "git_commit": traces[0].get("git_commit") if traces else None,
    }


def run_one(cfg_path: Path, run_id: str, backend: str, limit: int) -> tuple[int, str]:
    """Run the spike CLI in a subprocess; returns (exit code, combined output)."""
    cmd = [
        sys.executable, "-m", "shipdoc", "spike", "--config", str(cfg_path),
        "--docs", str(REPO / "splits" / "spike40.json"), "--split", "dev",
        "--run-id", run_id, "--backend", backend, "--limit", str(limit),
    ]  # fmt: skip
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, check=False)
    out = proc.stdout + proc.stderr
    print(out[-3000:])
    return proc.returncode, out


def main() -> int:
    """Smoke each config (fp16, then nf4 on OOM) and print the debug summaries."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", action="append", help="repeatable; default: the two spike configs")
    ap.add_argument("--backend", choices=["hf", "mock"], default="hf")
    ap.add_argument("--limit", type=int, default=5)
    args = ap.parse_args()
    paths.apply_env()  # before anything imports transformers (HF_HOME on D:)
    runs = paths.runs_dir() / "debug_3070"
    gold_dir = paths.data_dir() / "dev" / "labels"
    (runs / "configs").mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {}
    for cfg_arg in args.config or DEFAULT_CONFIGS:
        cfg_path = REPO / cfg_arg
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        attempts = [("fp16", cfg_path, raw)]
        nf4 = copy.deepcopy(raw)
        nf4["name"] = raw["name"] + "_nf4"
        nf4["model"]["quant"] = "nf4"
        nf4_path = runs / "configs" / f"{nf4['name']}.yaml"
        nf4_path.write_text(yaml.safe_dump(nf4, sort_keys=False), encoding="utf-8")
        attempts.append(("nf4", nf4_path, nf4))
        for tag, path, cfg in attempts:
            run_id = f"debug_3070/{cfg['name']}_{tag}"
            print(f"=== {run_id} ({LABEL})")
            code, out = run_one(path, run_id, args.backend, args.limit)
            if code == 0:
                s = summarize(runs / f"{cfg['name']}_{tag}", gold_dir, int(cfg["max_new_tokens"]))
                s["quant"] = tag
                summaries[run_id] = s
                break
            summaries[run_id] = {
                "label": LABEL,
                "failed": True,
                "oom": any(m in out for m in OOM_MARKERS),
            }
            if not summaries[run_id]["oom"]:
                break  # only an OOM justifies the nf4 fallback
    (runs / "smoke_summary.json").write_text(json.dumps(summaries, indent=1), encoding="utf-8")
    print(json.dumps(summaries, indent=1))
    return 0 if all(not s.get("failed") for s in summaries.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
