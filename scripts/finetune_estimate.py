"""Fine-tune token / step / wall-clock / compute-unit arithmetic (reports/finetune_plan.md).

Every output is an ESTIMATE (UNVERIFIED): the page counts come from the gold labels and
splits/folds.json, the token lengths from reports/token_budget.md and the spike traces, the
training throughput from a FLOP model divided by an ASSUMED effective TFLOP/s (derived from the
measured inference prefill rate, see ``prefill_rate_tflops``). Nothing here was measured on a
training run. Hours are GPU hours; compute units (CU) use the T4 / L4 rates in ``CU_PER_HOUR``.

The ``measured`` scenario replaces the throughput ASSUMPTION for the L4 with one smoke run's
seconds per step (``MEASURED_*`` constants below, from ``smoke_status.json`` / ``metrics.jsonl``
of ``ft_smoke_b10d810_bf16``). It is still an ESTIMATE for the folds: one 20-step run, 40 pages,
no checkpoint or evaluation time inside the measured steps, the FLOP model used only to spread the
measured time over forward-only evaluation.

Run: uv run python scripts/finetune_estimate.py [--epochs 2] [--accum 8]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# --- model constants: sums of tensor shapes read from the safetensors headers of Qwen/Qwen3.5-4B
# --- at revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a (HTTP range request, 2026-10-02).
N_LM_MATMUL = 3_569_090_560  # 2-D weights under model.language_model.layers (all matmul params)
N_LM_HEAD = 635_699_200  # embed_tokens, tied with lm_head (248320 x 2560)
N_VISION_BLOCKS = 302_309_376  # model.visual.blocks.* (24 blocks)
N_VISION_MERGER = 27_271_680
N_LOADED = 4_539_265_536  # every tensor except mtp.* (ignored on load by transformers 5.18.0)
LORA_R16_ALL_LINEAR = 30_474_240  # r=16 on q/k/v/o, gate/up/down, in_proj_qkv/in_proj_z/out_proj
N_ATTN_LAYERS = 8  # full_attention layers (layer_types, every 4th of 32)
ATTN_WIDTH = 4096  # 16 heads x head_dim 256
VISION_WIDTH = 1024
VISION_DEPTH = 24

# --- token constants (artifacts named in reports/finetune_plan.md)
VISUAL_TOKENS = 1260  # max_pixels 1280*32*32 on a 1240x1754 page (spike traces: n_visual_tokens)
PATCHES = VISUAL_TOKENS * 4  # spatial merge 2x2
INPUT_TOKENS = 2100  # n_input_tokens mean, prompt v2 img_only (2093 dev100, 2136 spike40 compact)
TARGET_MEAN = 531.3  # reports/token_budget.md, keyed json format, qwen35_4b tokenizer
TARGET_P99 = 1190.0
TARGET_MAX = 1244.0

# --- measured inference anchors (spike traces at 2abf481, recomputed 2026-10-02)
PREFILL_S = 2.05  # OLS intercept, dev100 qwen35_4b img_only (includes fixed per-call overhead)
INFER_S_PER_PAGE = 36.0  # mean latency_s, dev100 qwen35_4b img_only keyed, T4 fp16, 135 pages
INFER_S_PER_PAGE_HIGH = 42.5  # configs/spike_speed.json formula at n_out = 531.3 keyed gold mean

CU_PER_HOUR = {"T4": (1.19, 1.58), "L4": (3.00, 3.00)}  # (low, high); L4: varlog.info only
MODEL_LOAD_S = 120.0  # same ESTIMATE as configs/spike_speed.json

# --- MEASURED on a Colab L4, bf16, run ft_smoke_b10d810_bf16 (GG, 2026-10-02): 20 optimizer
# --- steps of accumulation 2 (2 pages per step, micro-batch 1) over 40 of the 80 smoke pages.
MEASURED_GPU = "L4"
MEASURED_S_PER_STEP = 17.3  # mean of the 20 steps of session 2 = 17.29 s (all 29 file lines: 17.27)
MEASURED_STEP_PAGES = 2  # smoke accumulation 2 x micro-batch 1 (metrics.jsonl: n_micro 2)
MEASURED_S_PER_PAGE = MEASURED_S_PER_STEP / MEASURED_STEP_PAGES  # 8.65 s per training page
MEASURED_PEAK_VRAM_GIB = 11.28  # peak_vram_gib of the 20 steps; the longest-page preflight is <= it
L4_VRAM_GIB = (
    24.0  # NVIDIA datasheet; Colab shows ~22 GiB (varlog.info), VRAM budget in the config is 20
)
# --- evaluation (configs/finetune_qwen35_4b.yaml): informational held-out loss on a fixed subset
EVAL_PAGES = 24
EVAL_EVERY = 10
# --- OOF inference: T4 fp16, UNBATCHED (dev100 keyed trace at 2abf481); batched speed is a
# --- placeholder until the bench exists, so the rows are labelled hypotheticals.
BATCH_SPEEDUPS = (1.0, 2.0, 3.0)


def lm_flops_per_token() -> float:
    """Forward matmul FLOPs per token through the language-model layers (2 * params)."""
    return 2.0 * N_LM_MATMUL


def vision_flops() -> float:
    """Forward FLOPs of the vision tower for one page (matmuls + full self-attention)."""
    mm = 2.0 * (N_VISION_BLOCKS + N_VISION_MERGER / 4) * PATCHES
    attn = 4.0 * PATCHES**2 * VISION_WIDTH * VISION_DEPTH
    return mm + attn


def train_flops_per_page(in_tokens: float, tgt_tokens: float) -> dict[str, float]:
    """FLOPs of one training page (LoRA on a frozen base, gradient checkpointing).

    Frozen base: backward needs only the input gradient (1x forward, no weight-gradient matmuls);
    checkpointing adds one recompute forward, so the layers cost 3x forward. The vision tower is
    frozen and upstream of no trainable parameter: forward only. Logits are computed on the
    target positions only (forward + input gradient = 2x), outside the checkpointed layers.
    """
    s = in_tokens + tgt_tokens
    layers = 3.0 * lm_flops_per_token() * s
    attn = 3.0 * N_ATTN_LAYERS * 4.0 * s**2 * ATTN_WIDTH
    head = 2.0 * 2.0 * N_LM_HEAD * tgt_tokens
    vis = vision_flops()
    return {"layers": layers, "attn_scores": attn, "lm_head": head, "vision": vis,
            "total": layers + attn + head + vis}  # fmt: skip


def prefill_rate_tflops(in_tokens: float = INPUT_TOKENS, prefill_s: float = PREFILL_S) -> float:
    """Effective TFLOP/s implied by the measured inference prefill (a LOWER bound: the intercept
    also holds fixed per-call overhead and the first decode step)."""
    flops = vision_flops() + lm_flops_per_token() * in_tokens
    return flops / prefill_s / 1e12


@dataclass(frozen=True)
class Throughput:
    """Assumed effective training TFLOP/s per GPU for a scenario (UNVERIFIED)."""

    gpu: str
    label: str
    tflops: float


def measured_throughput() -> Throughput:
    """The L4 scenario whose seconds per page equal the measured smoke value (at the mean
    sequence): the effective TFLOP/s that the FLOP model implies for 8.65 s/page."""
    flops = train_flops_per_page(INPUT_TOKENS, TARGET_MEAN)["total"]
    return Throughput(MEASURED_GPU, "measured", flops / MEASURED_S_PER_PAGE / 1e12)


def scenarios(base: float | None = None, include_measured: bool = False) -> list[Throughput]:
    """Low / central / high effective rates. T4 central = 0.75 x the measured inference prefill
    rate (training adds fp32 LoRA paths and the fp32 torch gated-delta fallback); L4 = T4 x
    {1.2, 1.6, 1.9} (peak ratio 121/65 = 1.86 dense FP16, bandwidth equal at 300 GB/s)."""
    base = prefill_rate_tflops() if base is None else base
    t4 = {"low": 0.5 * base, "central": 0.75 * base, "high": base}
    ratio = {"low": 1.2, "central": 1.6, "high": 1.9}
    out = [Throughput("T4", k, v) for k, v in t4.items()]
    out += [Throughput("L4", k, t4[k] * ratio[k]) for k in t4]
    if include_measured:
        out.append(measured_throughput())
    return out


def seconds_per_page(tp: Throughput, in_tokens: float, tgt_tokens: float) -> float:
    """Training seconds for one page at the scenario's effective rate."""
    if tp.tflops <= 0:
        raise ValueError(f"tflops must be > 0, got {tp.tflops}")
    return train_flops_per_page(in_tokens, tgt_tokens)["total"] / (tp.tflops * 1e12)


def steps(pages: int, epochs: int, micro_bs: int, accum: int) -> int:
    """Optimizer steps for ``epochs`` passes over ``pages`` (a partial last batch is a step)."""
    per_step = micro_bs * accum
    return epochs * -(-pages // per_step)


def forward_fraction(in_tokens: float = INPUT_TOKENS, tgt_tokens: float = TARGET_MEAN) -> float:
    """Share of a training page's FLOPs that a forward-only evaluation page needs: layers and
    attention scores once instead of 3x, logits forward only (half), vision unchanged."""
    f = train_flops_per_page(in_tokens, tgt_tokens)
    fwd = f["layers"] / 3 + f["attn_scores"] / 3 + f["lm_head"] / 2 + f["vision"]
    return fwd / f["total"]


def eval_count(total_steps: int, eval_every: int, steps_per_epoch: int) -> int:
    """Distinct steps with a held-out evaluation: every `eval_every` steps and every epoch end
    (one evaluation serves both when they coincide), as ``Trainer._record_eval`` does."""
    every = set(range(eval_every, total_steps + 1, eval_every)) if eval_every else set()
    ends = set(range(steps_per_epoch, total_steps + 1, steps_per_epoch))
    return len(every | ends)


def hours_cu(seconds: float, gpu: str) -> tuple[float, float, float]:
    """(hours, CU at the low rate, CU at the high rate)."""
    h = seconds / 3600
    lo, hi = CU_PER_HOUR[gpu]
    return h, h * lo, h * hi


def count_pages(doc_ids: list[str], root: Path) -> int:
    """Total pages of the docs (``pages`` in each gold label file under data/<split>/labels)."""
    total = 0
    for d in doc_ids:
        p = root / "data" / d.split("_")[0] / "labels" / f"{d}.json"
        total += len(json.loads(p.read_text(encoding="utf-8"))["pages"])
    return total


def fold_pages(root: Path) -> dict[str, list[int] | int]:
    """Train / held-out page counts per fold, and the all-train page count."""
    folds = json.loads((root / "splits" / "folds.json").read_text(encoding="utf-8"))
    every = [d for f in folds["folds"] for d in f["val_doc_ids"]]
    val = [count_pages(f["val_doc_ids"], root) for f in folds["folds"]]
    return {
        "val": val,
        "train": [count_pages(every, root) - v for v in val],
        "all_train": count_pages([d for d in every if d.startswith("train_")], root),
        "all": count_pages(every, root),
    }


def kfold_seconds(train_pages: list[int], epochs: int, s_page: float) -> float:
    """Training seconds of a K-fold run: one model load plus ``epochs`` passes per fold."""
    return sum(MODEL_LOAD_S + epochs * n * s_page for n in train_pages)


def fold_balance(meta: dict[str, dict], folds: list[dict]) -> list[dict]:
    """Per fold: docs, invoice / waybill docs, held-out groups by type, train/val group overlap.

    ``meta`` maps doc_id -> a meta row (``waybill``, ``supplier_group``); ``folds`` is the
    ``folds`` list of splits/folds.json.
    """
    out = []
    for f in folds:
        val = set(f["val_doc_ids"])
        inv = [d for d in val if not meta[d]["waybill"]]
        wb = [d for d in val if meta[d]["waybill"]]
        train_groups = {m["supplier_group"] for d, m in meta.items() if d not in val}
        val_groups = {meta[d]["supplier_group"] for d in val}
        per_group: dict[str, int] = {}
        for d in inv:
            g = meta[d]["supplier_group"]
            per_group[g] = per_group.get(g, 0) + 1
        out.append(
            {
                "fold": f["fold"],
                "docs": len(val),
                "invoice_docs": len(inv),
                "waybill_docs": len(wb),
                "invoice_groups": len(per_group),
                "waybill_groups": len({meta[d]["supplier_group"] for d in wb}),
                "invoice_docs_per_group": sorted(per_group.values()),
                "group_overlap_with_train": len(val_groups & train_groups),
            }
        )
    return out


def stage_rows(
    pages: dict[str, list[int] | int], epochs: int, accum: int, tp: Throughput
) -> list[tuple[str, float, float]]:
    """(stage, seconds, pages) training rows for one scenario (one model load per run)."""
    s_page = seconds_per_page(tp, INPUT_TOKENS, TARGET_MEAN)
    tr = pages["train"]
    smoke_pages = 20 * 2  # 20 optimizer steps x accum 2
    return [
        ("smoke 20 steps (accum 2)", MODEL_LOAD_S + smoke_pages * s_page, smoke_pages),
        ("1 fold", MODEL_LOAD_S + epochs * tr[0] * s_page, epochs * tr[0]),
        ("3 folds", 3 * MODEL_LOAD_S + epochs * sum(tr) * s_page, epochs * sum(tr)),
        (
            "final all-train",
            MODEL_LOAD_S + epochs * pages["all_train"] * s_page,
            epochs * pages["all_train"],
        ),
    ]  # fmt: skip


def infer_rows(pages: dict[str, list[int] | int]) -> list[tuple[str, float, float]]:
    """T4 fp16 inference of the merged adapter: OOF on all 500 docs, dev100 (final), per page."""
    dev_pages = pages["all"] - pages["all_train"]
    return [
        ("OOF inference, 3 folds (all 500 docs)", pages["all"] * INFER_S_PER_PAGE, pages["all"]),
        ("final model on dev100", dev_pages * INFER_S_PER_PAGE, dev_pages),
    ]  # fmt: skip


def measured_rows(
    pages: dict[str, list[int] | int],
    epochs: int,
    accum: int,
    eval_every: int = EVAL_EVERY,
    eval_pages: int = EVAL_PAGES,
) -> list[dict[str, float | int | str]]:
    """Per stage on the measured L4: visits, steps, evaluations, seconds (train / eval / load),
    hours and CU. Training seconds = visits x 8.65 s; evaluation = `eval_pages` forward-only pages
    (FLOP-model fraction of the measured time); one model load per run."""
    s_page = MEASURED_S_PER_PAGE
    s_eval_page = s_page * forward_fraction()
    tr_pages = pages["train"]
    stages = [("smoke", None, 20 * 2)]
    stages += [(f"fold{k}", n, epochs * n) for k, n in enumerate(tr_pages)]
    stages += [("final", pages["all_train"], epochs * pages["all_train"])]
    out: list[dict[str, float | int | str]] = []
    for name, n_pages, visits in stages:
        if n_pages is None:
            n_steps, n_eval = 20, 0
        else:
            n_steps = steps(n_pages, epochs, 1, accum)
            n_eval = eval_count(n_steps, eval_every, -(-n_pages // accum))
        train_s, eval_s = visits * s_page, n_eval * eval_pages * s_eval_page
        h, lo, hi = hours_cu(MODEL_LOAD_S + train_s + eval_s, MEASURED_GPU)
        out.append({
            "stage": name, "visits": visits, "steps": n_steps, "evals": n_eval,
            "train_h": train_s / 3600, "eval_h": eval_s / 3600, "hours": h, "cu": lo,
        })  # fmt: skip
    folds = [r for r in out if str(r["stage"]).startswith("fold")]
    out.append({
        "stage": "3 folds", "visits": sum(int(r["visits"]) for r in folds),
        "steps": sum(int(r["steps"]) for r in folds), "evals": sum(int(r["evals"]) for r in folds),
        "train_h": sum(float(r["train_h"]) for r in folds),
        "eval_h": sum(float(r["eval_h"]) for r in folds),
        "hours": sum(float(r["hours"]) for r in folds),  # each fold is its own run: 3 loads
        "cu": sum(float(r["cu"]) for r in folds),
    })  # fmt: skip
    return out


def oof_rows(
    pages: dict[str, list[int] | int], speedups: tuple[float, ...] = BATCH_SPEEDUPS
) -> list[dict[str, float | str]]:
    """OOF inference on T4 fp16: fold 0 held-out and all folds, at the current UNBATCHED 36.0
    s/page divided by a HYPOTHETICAL batch speedup (1x = today; 2x / 3x are placeholders)."""
    sets = [("fold0 held-out", int(pages["val"][0])), ("all folds (OOF)", int(pages["all"]))]
    out: list[dict[str, float | str]] = []
    for label, n in sets:
        for k in speedups:
            h = n * INFER_S_PER_PAGE / k / 3600
            lo, hi = CU_PER_HOUR["T4"]
            out.append({"set": label, "pages": n, "speedup": k, "hours": h,
                        "cu_low": h * lo, "cu_high": h * hi})  # fmt: skip
    return out


def measured_lines(
    pages: dict[str, list[int] | int], epochs: int, accum: int,
    eval_every: int = EVAL_EVERY, eval_pages: int = EVAL_PAGES,
) -> list[str]:  # fmt: skip
    """Text lines of the measured-L4 table and the OOF table (printed by the notebook too)."""
    lines = [
        f"MEASURED L4 bf16 (smoke ft_smoke_b10d810_bf16): {MEASURED_S_PER_STEP} s/step of "
        f"{MEASURED_STEP_PAGES} pages = {MEASURED_S_PER_PAGE:.2f} s/page, peak "
        f"{MEASURED_PEAK_VRAM_GIB} GiB of ~22-24; ESTIMATE for the stages below "
        f"(CU at {CU_PER_HOUR['L4'][0]:.2f}/h, third-party UNVERIFIED; load {MODEL_LOAD_S:.0f} s; "
        f"eval = {eval_pages} held-out pages every {eval_every} steps + each epoch end):",
        f"{'stage':<8}{'visits':>7}{'steps':>7}{'evals':>6}{'train h':>9}{'eval h':>8}"
        f"{'hours':>7}{'CU':>7}",
    ]
    for r in measured_rows(pages, epochs, accum, eval_every, eval_pages):
        lines.append(
            f"{r['stage']:<8}{r['visits']:>7}{r['steps']:>7}{r['evals']:>6}"
            f"{r['train_h']:>9.2f}{r['eval_h']:>8.2f}{r['hours']:>7.2f}{r['cu']:>7.1f}"
        )
    lines += [
        "",
        f"OOF inference on T4 fp16, UNBATCHED {INFER_S_PER_PAGE:.0f} s/page; the 2x / 3x rows are "
        "HYPOTHETICAL batch speedups (no batched measurement exists), "
        "CU at 1.19 / 1.58 per T4 hour:",
        f"{'set':<17}{'pages':>6}{'speedup':>9}{'hours':>7}{'CU @1.19':>10}{'CU @1.58':>10}",
    ]
    for r in oof_rows(pages):
        lines.append(
            f"{r['set']:<17}{r['pages']:>6}{str(r['speedup']).rstrip('0').rstrip('.') + 'x':>9}"
            f"{r['hours']:>7.2f}{r['cu_low']:>10.1f}{r['cu_high']:>10.1f}"
        )
    return lines


def render(
    pages: dict[str, list[int] | int], epochs: int, accum: int, balance: list[dict] | None = None
) -> str:
    """Markdown report of the arithmetic (pages, tokens, steps, wall-clock, CU)."""
    tr, val = pages["train"], pages["val"]
    lines = [
        f"pages: all-train {pages['all_train']}, all 500 docs {pages['all']}, fold train {tr}, "
        f"fold held-out {val}",
        f"tokens/page: input {INPUT_TOKENS} (visual {VISUAL_TOKENS}) + target {TARGET_MEAN}"
        f" = {INPUT_TOKENS + TARGET_MEAN:.0f} mean; p99 {INPUT_TOKENS + TARGET_P99:.0f};"
        f" max {INPUT_TOKENS + TARGET_MAX:.0f}",
        f"tokens/epoch: all-train {pages['all_train'] * (INPUT_TOKENS + TARGET_MEAN):,.0f}"
        f" (supervised {pages['all_train'] * TARGET_MEAN:,.0f}); one fold (mean train pages "
        f"{sum(tr) / len(tr):.0f}) {sum(tr) / len(tr) * (INPUT_TOKENS + TARGET_MEAN):,.0f}",
        f"optimizer steps (micro-batch 1, accum {accum}, {epochs} epochs): all-train "
        f"{steps(pages['all_train'], epochs, 1, accum)}, fold "
        f"{[steps(t, epochs, 1, accum) for t in tr]}",
    ]
    f = train_flops_per_page(INPUT_TOKENS, TARGET_MEAN)
    lines.append(
        "FLOPs/page (TF): " + ", ".join(f"{k} {v / 1e12:.1f}" for k, v in f.items())
        + f"; inference-prefill anchor {prefill_rate_tflops():.1f} TFLOP/s"
    )  # fmt: skip
    lines += ["", "| gpu | scenario | TFLOP/s | s/page | stage | hours | CU low | CU high |"]
    lines.append("|---|---|---:|---:|---|---:|---:|---:|")
    for tp in scenarios():
        s_page = seconds_per_page(tp, INPUT_TOKENS, TARGET_MEAN)
        for name, sec, _ in stage_rows(pages, epochs, accum, tp):
            h, lo, hi = hours_cu(sec, tp.gpu)
            lines.append(
                f"| {tp.gpu} | {tp.label} | {tp.tflops:.1f} | {s_page:.1f} | {name} | "
                f"{h:.2f} | {lo:.1f} | {hi:.1f} |"
            )
    k2 = [pages["all"] // 2, pages["all"] - pages["all"] // 2]  # K=2: each fold trains on half
    for tp in scenarios():
        if tp.label != "central":
            continue
        s_page = seconds_per_page(tp, INPUT_TOKENS, TARGET_MEAN)
        k3_h = hours_cu(kfold_seconds(tr, epochs, s_page), tp.gpu)[0]
        k2_h = hours_cu(kfold_seconds(k2, epochs, s_page), tp.gpu)[0]
        lines.append(
            f"K=3 vs K=2 fold-training hours, {tp.gpu} central: {k3_h:.2f} vs {k2_h:.2f} "
            f"(K=2 trains on ~{k2[0]} pages/fold; OOF inference is {pages['all']} pages either way)"
        )
    for b in balance or []:
        lines.append(f"fold balance: {json.dumps(b)}")
    lines += ["", "| T4 fp16 inference stage (merged adapter) | pages | hours | CU low | CU high |"]
    lines.append("|---|---:|---:|---:|---:|")
    for name, sec, n in infer_rows(pages):
        h, lo, hi = hours_cu(sec, "T4")
        lines.append(f"| {name} | {n:.0f} | {h:.2f} | {lo:.1f} | {hi:.1f} |")
    lines += ["", *measured_lines(pages, epochs, accum)]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    args = ap.parse_args(argv)
    folds = json.loads((args.root / "splits" / "folds.json").read_text(encoding="utf-8"))
    meta: dict[str, dict] = {}
    for split in ("train", "dev"):
        for row in json.loads((args.root / "meta" / f"{split}.json").read_text(encoding="utf-8")):
            meta[row["doc_id"]] = row
    bal = fold_balance(meta, folds["folds"])
    print(render(fold_pages(args.root), args.epochs, args.accum, bal))
    return 0


if __name__ == "__main__":
    sys.exit(main())
