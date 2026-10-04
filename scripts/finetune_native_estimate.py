"""Native-resolution (max_pixels 2196480, 2145 tokens) fine-tune estimates: VRAM, hours, CU.

A thin wrapper over ``scripts/finetune_estimate.py`` (imported, not edited). Every output is an
ESTIMATE (UNVERIFIED): the only measurement at 1260 tokens is the fold-0 training run
($SHIPDOC_RUNS_DIR\\ft_fold0\\ft_fold0_42b812b_bf16: 112 steps, 882 page visits, mean 69.85 s/step,
peak VRAM 11.416 GiB, L4 bf16) and the 06 sweep's inference peaks (9.14 GiB at 1260 tokens,
9.41 GiB at 2145, T4 fp16 batch 1, reports/res_sweep.md). Nothing here was measured at native
resolution on a training run: the smoke stage measures it first.

Two models, both stated in the output:

* Time: seconds per page = measured 8.87 x a factor. ASSUMED linear: ``LOW`` scales only with the
  total sequence (the ~840 text-prompt tokens and the target do not grow: 1.336x), ``HIGH`` with the
  image tokens alone (2145 / 1260 = 1.70x), ``FLOP`` is the repo's FLOP model evaluated at both
  resolutions (vision attention quadratic in patches).
* Memory: peak = fixed (weights + LoRA state + loss chunk, resolution independent) + the
  resolution-dependent remainder, which is anchored to the fold-0 measurement (peak 11.416 GiB minus
  the fixed part) and scaled by the longest page's sequence growth (low), by the modeled split with
  the measured inference vision-transient growth (central), or by the image-token ratio (high).

Run: uv run python scripts/finetune_native_estimate.py
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "finetune_estimate", ROOT / "scripts" / "finetune_estimate.py"
)
fe = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("finetune_estimate", fe)  # dataclasses resolve their module via sys.modules
_spec.loader.exec_module(fe)

# --- resolution (shipdoc.ressweep.visual_tokens on a 1240x1754 page; tests assert both)
MAX_PIXELS_1260 = 1_310_720
MAX_PIXELS_NATIVE = 2_196_480
TOKENS_1260 = 1260
TOKENS_NATIVE = 2145
TOKEN_RATIO = TOKENS_NATIVE / TOKENS_1260  # 1.7024
TEXT_PROMPT_TOKENS = fe.INPUT_TOKENS - TOKENS_1260  # 840: schema prompt, not resolution dependent
INPUT_NATIVE = TEXT_PROMPT_TOKENS + TOKENS_NATIVE  # 2985

# --- MEASURED: fold-0 training run on a Colab L4, bf16, 1260 tokens (metrics.jsonl, train_status)
FOLD0_STEPS = 112
FOLD0_VISITS = 882  # 441 pages x 2 epochs (sum of n_micro over the 112 steps)
FOLD0_MEAN_STEP_S = 69.85  # mean of the per-step `seconds` (optimizer step only: no eval / ckpt)
FOLD0_S_PER_PAGE = FOLD0_MEAN_STEP_S * FOLD0_STEPS / FOLD0_VISITS  # 8.870
FOLD0_PEAK_GIB = 11.416  # train_status.json peak_vram_gib 11.415850639343262 (the whole run)

# --- memory model constants (reports/finetune_plan.md "Memory", chunked-loss column)
WEIGHTS_GIB = 8.46  # bf16 / fp16 base, 4.54B loaded parameters
LORA_STATE_GIB = 0.45  # fp32 LoRA weights + grads + Adam m, v
LOSS_CHUNK_GIB = 0.71  # chunked loss, 256 positions, independent of S
CKPT_INPUT_BYTES_PER_TOKEN = 32 * 2560 * 2  # checkpointed layer inputs, bf16
LAYER_TRANSIENT_GIB_AT_3344 = (
    1.6  # one-layer recompute + backward at the longest 1260 page (ASSUMED)
)
VISION_MISC_GIB = 0.66  # inference peak 9.12 minus 8.455 weights (measured at inference)
INFER_PEAK_1260_GIB = 9.14  # reports/res_sweep.md, control, T4 fp16 batch 1 (measured)
INFER_PEAK_NATIVE_GIB = 9.41  # reports/res_sweep.md, 2196480 (measured)
LONGEST_S_1260 = fe.INPUT_TOKENS + 1244  # 3344: longest gold target (fe.TARGET_MAX) + mean prompt
LONGEST_S_NATIVE = INPUT_NATIVE + 1244  # 4229
BUDGET_GIB = {"bf16": 20.0, "fp16": 13.5, "fp32": 20.0}  # configs/finetune_qwen35_4b*.yaml smoke
MARGIN = 0.15  # "within 15% of the budget" = estimate above 0.85 x budget

CU_RATES = (1.19, 1.58, 3.00)  # 1.19 / 1.58 as asked (third-party); 3.00 = fe's L4 rate (varlog)
STAGES = ("smoke", "fold0", "fold1", "fold2", "final")
GIB = 2**30


@dataclass(frozen=True)
class VramEstimate:
    """One scenario's native peak (GiB) with the growth factor applied to the remainder."""

    label: str
    peak_gib: float
    remainder_gib: float
    factor: float


def fixed_gib() -> float:
    """Resolution-independent part of the peak: weights + LoRA state + loss chunk."""
    return WEIGHTS_GIB + LORA_STATE_GIB + LOSS_CHUNK_GIB


def remainder_1260_gib() -> float:
    """The part of the measured 1260 peak that depends on the sequence (peak - fixed)."""
    return FOLD0_PEAK_GIB - fixed_gib()


def modeled_remainder_1260_gib() -> float:
    """The plan's modeled sequence-dependent total at the longest page (it over-predicts)."""
    ckpt = LONGEST_S_1260 * CKPT_INPUT_BYTES_PER_TOKEN / GIB
    return ckpt + LAYER_TRANSIENT_GIB_AT_3344 + VISION_MISC_GIB


def vram_estimates() -> list[VramEstimate]:
    """low / central / high native peaks plus a 3x gross-error bound (all UNVERIFIED)."""
    rem = remainder_1260_gib()
    s_ratio = LONGEST_S_NATIVE / LONGEST_S_1260
    vision_share = VISION_MISC_GIB / modeled_remainder_1260_gib()
    vision_growth = INFER_PEAK_NATIVE_GIB - INFER_PEAK_1260_GIB  # measured, inference, batch 1
    central_rem = rem * (1 - vision_share) * s_ratio + rem * vision_share * (
        (VISION_MISC_GIB + vision_growth) / VISION_MISC_GIB
    )
    rows = [
        ("low (remainder x longest-page sequence growth)", rem * s_ratio, s_ratio),
        ("central (layers x seq growth, vision x measured inference growth)", central_rem,
         central_rem / rem),
        ("high (remainder x image-token ratio)", rem * TOKEN_RATIO, TOKEN_RATIO),
        ("gross-error bound (remainder x 3)", rem * 3.0, 3.0),
    ]  # fmt: skip
    return [VramEstimate(lab, fixed_gib() + r, r, f) for lab, r, f in rows]


def vram_verdict(peak_gib: float, precision: str = "bf16") -> str:
    """How an estimate sits against the smoke budget (``within 15%`` = above 0.85 x budget)."""
    budget = BUDGET_GIB[precision]
    pct = 100 * peak_gib / budget
    flag = (
        "WITHIN 15% of the budget" if peak_gib > (1 - MARGIN) * budget else "clear of the 15% band"
    )
    return f"{peak_gib:.2f} GiB = {pct:.0f}% of the {budget:.1f} GiB {precision} budget ({flag})"


# --- time model --------------------------------------------------------------------------------


def native_vision_flops(patches: int) -> float:
    """``fe.vision_flops`` at an arbitrary patch count (matmuls linear, attention quadratic)."""
    mm = 2.0 * (fe.N_VISION_BLOCKS + fe.N_VISION_MERGER / 4) * patches
    attn = 4.0 * patches**2 * fe.VISION_WIDTH * fe.VISION_DEPTH
    return mm + attn


def flops_per_page(in_tokens: float, tgt_tokens: float, patches: int) -> dict[str, float]:
    """``fe.train_flops_per_page`` with the vision term recomputed for `patches`."""
    f = fe.train_flops_per_page(in_tokens, tgt_tokens)
    vis = native_vision_flops(patches)
    return {**f, "vision": vis, "total": f["total"] - f["vision"] + vis}


def time_factors() -> dict[str, float]:
    """Seconds-per-page multipliers 1260 -> native (ASSUMED linear): low, flop, high."""
    s0 = fe.INPUT_TOKENS + fe.TARGET_MEAN
    s1 = INPUT_NATIVE + fe.TARGET_MEAN
    f0 = flops_per_page(fe.INPUT_TOKENS, fe.TARGET_MEAN, TOKENS_1260 * 4)["total"]
    f1 = flops_per_page(INPUT_NATIVE, fe.TARGET_MEAN, TOKENS_NATIVE * 4)["total"]
    return {"low": s1 / s0, "flop": f1 / f0, "high": TOKEN_RATIO}


def eval_fraction(patches: int, in_tokens: float) -> float:
    """Forward-only evaluation page cost as a share of a training page (as fe.forward_fraction)."""
    f = flops_per_page(in_tokens, fe.TARGET_MEAN, patches)
    fwd = f["layers"] / 3 + f["attn_scores"] / 3 + f["lm_head"] / 2 + f["vision"]
    return fwd / f["total"]


@dataclass(frozen=True)
class StageEstimate:
    """One stage at one time factor: visits, steps, evals, hours."""

    stage: str
    visits: int
    steps: int
    evals: int
    hours: float


def stage_estimate(
    stage: str,
    n_pages: int | None,
    factor: float,
    epochs: int = 2,
    accum: int = 8,
    eval_every: int = 10,
    eval_pages: int = 24,
) -> StageEstimate:
    """Hours of one stage: model load + training visits x s/page x factor + held-out evals.

    ``n_pages`` None is the smoke (20 steps x 2 pages, no evaluation). Checkpoint writes and the
    longest-page preflight are not modeled (unmeasured).
    """
    s_page = FOLD0_S_PER_PAGE * factor
    if n_pages is None:
        visits, steps, evals = 40, 20, 0
    else:
        visits = epochs * n_pages
        steps = fe.steps(n_pages, epochs, 1, accum)
        evals = fe.eval_count(steps, eval_every, -(-n_pages // accum))
    frac = eval_fraction(TOKENS_NATIVE * 4, INPUT_NATIVE)
    seconds = fe.MODEL_LOAD_S + visits * s_page + evals * eval_pages * s_page * frac
    return StageEstimate(stage, visits, steps, evals, seconds / 3600)


def stage_pages(pages: dict) -> dict[str, int | None]:
    """Stage -> training pages (``pages`` as returned by ``fe.fold_pages``)."""
    tr = pages["train"]
    return {"smoke": None, "fold0": tr[0], "fold1": tr[1], "fold2": tr[2],
            "final": pages["all_train"]}  # fmt: skip


def totals(hours: dict[str, float]) -> dict[str, float]:
    """Sequential hours, all-four-parallel wall hours, and fold0-first-then-three-parallel wall.

    ``hours`` is stage -> hours (smoke included once, run before anything else). Parallel assumes
    one L4 per tab and that Colab grants them (UNVERIFIED); CU is the same in every arrangement.
    """
    run = [hours["fold0"], hours["fold1"], hours["fold2"], hours["final"]]
    return {
        "sequential_h": hours["smoke"] + sum(run),
        "parallel4_wall_h": hours["smoke"] + max(run),
        "fold0_first_wall_h": hours["smoke"] + hours["fold0"] + max(run[1:]),
        "cu_hours": hours["smoke"] + sum(run),
    }


def native_lines(pages: dict, epochs: int = 2, accum: int = 8, eval_every: int = 10,
                 eval_pages: int = 24) -> list[str]:  # fmt: skip
    """Printable estimate (the notebook prints these before anything runs)."""
    f = time_factors()
    out = [
        "NATIVE RESOLUTION ESTIMATE / UNVERIFIED (max_pixels 2196480: 2145 visual tokens per page "
        f"vs 1260 = x{TOKEN_RATIO:.2f}; input tokens {INPUT_NATIVE} vs {fe.INPUT_TOKENS}).",
        f"Anchor, MEASURED at 1260 on an L4 (fold 0): {FOLD0_S_PER_PAGE:.2f} s/page "
        f"({FOLD0_MEAN_STEP_S} s/step x {FOLD0_STEPS} steps / {FOLD0_VISITS} visits), peak "
        f"{FOLD0_PEAK_GIB} GiB.",
        f"s/page factors, ASSUMED linear: low x{f['low']:.2f} (total sequence), FLOP model "
        f"x{f['flop']:.2f}, high x{f['high']:.2f} (image tokens). Load {fe.MODEL_LOAD_S:.0f} s, "
        f"eval = {eval_pages} held-out pages every {eval_every} steps + each epoch end; "
        "checkpoint writes not modeled.",
        "",
        "VRAM at the longest page, native (budget check is the smoke's):",
    ]
    for v in vram_estimates():
        out.append(f"  {v.label}: {vram_verdict(v.peak_gib)}")
    out.append(f"  fp16 / T4 (budget {BUDGET_GIB['fp16']} GiB, NOT the plan): high = "
               + vram_verdict(vram_estimates()[2].peak_gib, "fp16"))  # fmt: skip
    sp = stage_pages(pages)
    hours: dict[str, dict[str, float]] = {k: {} for k in f}
    rates = " / ".join(f"{r:.2f}" for r in CU_RATES)
    out += [
        "",
        f"{'stage':<7}{'visits':>7}{'steps':>6}{'evals':>6}   hours low / flop / high"
        f"     CU at {rates} per h (central = flop)",
    ]
    for stage in STAGES:
        est = {k: stage_estimate(stage, sp[stage], fac, epochs, accum, eval_every, eval_pages)
               for k, fac in f.items()}  # fmt: skip
        for k, e in est.items():
            hours[k][stage] = e.hours
        e = est["flop"]
        cu = " / ".join(f"{e.hours * r:5.1f}" for r in CU_RATES)
        out.append(f"{stage:<7}{e.visits:>7}{e.steps:>6}{e.evals:>6}   "
                   f"{est['low'].hours:5.2f} / {e.hours:5.2f} / {est['high'].hours:5.2f}"
                   f"          {cu}")  # fmt: skip
    out.append("")
    for k in ("low", "flop", "high"):
        t = totals(hours[k])
        cu = " / ".join(f"{t['cu_hours'] * r:.1f}" for r in CU_RATES)
        out.append(
            f"{k:<5} total {t['sequential_h']:.2f} h sequential; wall "
            f"{t['parallel4_wall_h']:.2f} h with fold0/fold1/fold2/final in 4 parallel tabs, "
            f"{t['fold0_first_wall_h']:.2f} h fold0 first then 3 tabs; CU {cu} "
            "(same for any arrangement)"
        )
    out.append(
        "CU rates: 1.19 / 1.58 as asked (third-party, UNVERIFIED), 3.00 = the repo's L4 rate "
        "(varlog.info 2024-10-01, UNVERIFIED). Parallel tabs assume one L4 each is granted."
    )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--root", type=Path, default=ROOT)
    args = ap.parse_args(argv)
    print("\n".join(native_lines(fe.fold_pages(args.root))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
