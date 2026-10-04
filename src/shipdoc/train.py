"""LoRA fine-tuning of Qwen3.5-4B on per-page targets (spec Phase 4, reports/finetune_plan.md).

Method (V1 plan): LoRA (r=16) on the language-model linears only (``LORA_TARGET_REGEX``, 200
modules, 30,474,240 trainable parameters), vision tower frozen, gradient checkpointing with
``use_reentrant=False``, ``use_cache=False``, loss on the assistant JSON tokens only and computed
in 256-position chunks from the final hidden states (the full-sequence logits never exist),
micro-batch 1 with gradient accumulation, bf16 on an L4 or fp16 + ``GradScaler`` + fp32 LoRA
weights on a T4, a FIXED 2-epoch schedule.

Departure from spec Phase 4.3 (recorded in reports/finetune_plan.md "Leakage note"): there is NO
early stopping on held-out loss. The held-out loss is logged for information only, because
selecting a checkpoint on the fold's own held-out docs would bias the out-of-fold score that
decides gate G4.

Everything torch-dependent imports torch lazily and takes the model, the sample source, the hidden
state function and the logger as arguments, so the whole loop (including checkpoint and exact
resume) runs on a tiny CPU model in the tests. The Hugging Face path (``load_model_and_processor``,
``hf_hidden_states``, ``PageDataset``) is UNVERIFIED on the real model: no GPU was used to write it.

Run: ``python -m shipdoc.train --stage smoke|fold0|fold1|fold2|final --run-dir <dir>``.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import math
import os
import random
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml

from shipdoc import metricslog as ml
from shipdoc import paths
from shipdoc import trainset as ts
from shipdoc.augment import load_scan_params
from shipdoc.extract import SEED, get_adapter, page_schema

DEFAULT_CONFIG = paths.REPO_ROOT / "configs" / "finetune_qwen35_4b.yaml"
#: Anchored to ``model.language_model`` so the vision tower can never match (V1a). A ``str`` is
#: matched with ``re.fullmatch`` by peft. Checked against the checkpoint key names: 200 matches.
LORA_TARGET_REGEX = (
    r"model\.language_model\.layers\.\d+\."
    r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)"
    r"|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)"
    r"|mlp\.(gate_proj|up_proj|down_proj))"
)
#: 32 layers: 8 full attention (4 linears), 24 gated-delta (3 linears), 32 MLPs (3 linears).
EXPECTED_LORA_MODULES = 8 * 4 + 24 * 3 + 32 * 3
#: LoRA parameters per unit of rank over those 200 modules (30,474,240 at r=16).
LORA_PARAMS_PER_RANK = 1_904_640
PRIVATE_WANDB_PROJECT = "shipdoc-extract-debug"
WANDB_ENV_FLAG = "SHIPDOC_WANDB"
IGNORE_INDEX = -100
IM_END = "<|im_end|>"


class TrainError(RuntimeError):
    """A training-time invariant failed (token budget, divergence, bad checkpoint)."""


class SmokeFailure(TrainError):
    """The smoke-train stage failed: the full run must not start (`checks` has the table)."""

    checks: list[Check]


# --------------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LoraCfg:
    """LoRA hyper-parameters (``target_regex`` must stay the language-model-only regex)."""

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"
    target_regex: str = LORA_TARGET_REGEX


@dataclass(frozen=True)
class OptimCfg:
    """AdamW + warmup/cosine schedule."""

    lr: float = 2.0e-4
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    grad_clip: float = 1.0
    warmup_ratio: float = 0.1
    min_lr_ratio: float = 0.0


@dataclass(frozen=True)
class SmokeCfg:
    """Smoke-train stage: 20 optimizer steps and the abort criteria."""

    steps: int = 20
    grad_accum: int = 2
    first_last_k: int = 5
    min_rel_decrease: float = 0.05  # mean(last k) < mean(first k) * (1 - this)
    vram_budget_gib: Mapping[str, float] = field(
        default_factory=lambda: {"bf16": 20.0, "fp16": 13.5, "fp32": 20.0}
    )
    max_scaler_skips: int = 5
    preflight_longest: bool = True


@dataclass(frozen=True)
class TrainConfig:
    """Everything that defines a run; the whole dict is hashed into the run signature."""

    seed: int = SEED
    repo: str = "Qwen/Qwen3.5-4B"
    revision: str = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    adapter: str = "qwen35"
    max_pixels: int = 1280 * 32 * 32
    expected_visual_tokens: int | None = 1260
    precision: str = "auto"  # auto | bf16 | fp16 | fp32
    lora: LoraCfg = field(default_factory=LoraCfg)
    optim: OptimCfg = field(default_factory=OptimCfg)
    epochs: int = 2
    grad_accum: int = 8
    loss_chunk: int = 256
    ckpt_every: int = 10
    ckpt_keep: int = 2
    eval_every: int = 10
    eval_pages: int = 24
    early_stopping: bool = False
    ambiguous_policy: str = "header_only"
    monotone_boundary: bool = True
    occlusion_rate: float = 0.15
    occlusion_field_weights: Mapping[str, float] = field(
        default_factory=lambda: {"invoice_number": 4.0, "invoice_date": 4.0}
    )
    scan_prob: float = 0.5
    smoke: SmokeCfg = field(default_factory=SmokeCfg)

    def signature(self) -> str:
        """sha256 prefix of the config content (a checkpoint only resumes under the same one)."""
        blob = json.dumps(dataclasses.asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    @property
    def augment(self) -> ts.AugmentConfig:
        """Augmentation settings for `trainset`."""
        return ts.AugmentConfig(
            self.occlusion_rate, self.scan_prob, self.seed, dict(self.occlusion_field_weights)
        )


def load_config(path: Path | str = DEFAULT_CONFIG) -> TrainConfig:
    """Read the YAML config. Unknown keys raise TypeError (typos must not pass silently)."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    subs = {"lora": LoraCfg, "optim": OptimCfg, "smoke": SmokeCfg}
    kw = {k: (subs[k](**v) if k in subs else v) for k, v in raw.items()}
    cfg = TrainConfig(**kw)
    if cfg.early_stopping:
        raise ValueError("early stopping on held-out loss is disabled by design (G4 leakage)")
    if cfg.lora.target_regex != LORA_TARGET_REGEX:
        raise ValueError("lora.target_regex must be the language-model-only regex")
    return cfg


def expected_lora_params(r: int) -> int:
    """Trainable parameters of the plan's LoRA at rank `r` (200 modules)."""
    return LORA_PARAMS_PER_RANK * r


def inference_keys(cfg: TrainConfig) -> dict[str, Any]:
    """The settings of a training run that an inference run must share with it (no tuning knobs).

    Recorded in the adapter manifest; ``shipdoc.oof.verify_adapter_manifest`` compares them with
    the production inference config. The targets are the KEYED json format (`render_page_text`).
    """
    from shipdoc.prompts import PROMPT_VERSION

    return {
        "model_repo": cfg.repo,
        "model_revision": cfg.revision,
        "adapter": cfg.adapter,
        "max_pixels": cfg.max_pixels,
        "prompt_version": PROMPT_VERSION,
        "output_format": "json",
    }


def build_run_meta(
    stage: str,
    split: ts.StageSplit,
    cfg: TrainConfig,
    *,
    n_heldout_eval_pages: int,
    n_lora_modules: int,
    n_trainable: int,
    code_sha: str,
) -> dict[str, Any]:
    """What ``final/manifest.json`` records about the run (merged into it by `Trainer.save_final`).

    The OOF inference refuses an adapter whose manifest lacks the training doc ids (the proof that
    no held-out document was trained on), the fold, the code SHA or the LoRA bookkeeping.
    `train_doc_ids` is the stage's training POOL (`stage_split`); the smoke stage trains on a
    prefix of its pages only and is never used for OOF.
    """
    fold = int(stage[4:]) if stage.startswith("fold") else None
    return {
        "stage": stage,
        "fold": fold,
        "manifest_hash": split.manifest_hash(),
        "n_heldout_eval_pages": n_heldout_eval_pages,
        "code_sha": code_sha,
        "train_config_signature": cfg.signature(),
        "train_doc_ids": sorted(split.train_ids),
        "n_train_docs": len(split.train_ids),
        "heldout_doc_ids": sorted(split.heldout_ids),
        "lora": {
            "r": cfg.lora.r,
            "alpha": cfg.lora.alpha,
            "n_modules": n_lora_modules,
            "n_trainable": n_trainable,
        },
        "inference_keys": inference_keys(cfg),
    }


# --------------------------------------------------------------------------------------------
# Model preparation (LoRA bookkeeping, gradient checkpointing)
# --------------------------------------------------------------------------------------------


def match_target_modules(model: Any, regex: str = LORA_TARGET_REGEX) -> list[str]:
    """Names of the ``nn.Linear`` modules `regex` selects (``re.fullmatch``, as peft does)."""
    from torch import nn

    pat = re.compile(regex)
    return [n for n, m in model.named_modules() if isinstance(m, nn.Linear) and pat.fullmatch(n)]


def count_lora_modules(model: Any) -> int:
    """Modules that carry LoRA adapters (peft layers expose ``lora_A``)."""
    return sum(1 for m in model.modules() if hasattr(m, "lora_A") and hasattr(m, "lora_B"))


def count_trainable(model: Any) -> int:
    """Number of parameters with ``requires_grad``."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_lora_config(cfg: LoraCfg) -> Any:
    """peft ``LoraConfig`` for the plan (imported lazily: peft is in the locked ``train`` group)."""
    try:
        from peft import LoraConfig
    except ImportError as exc:
        raise ImportError(
            "peft is not installed. It is in the locked `train` dependency group "
            "(uv sync --frozen --group train); the default local environment omits it."
        ) from exc
    return LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        bias=cfg.bias,
        task_type="CAUSAL_LM",
        target_modules=cfg.target_regex,
    )


def ensure_fp32_trainable(model: Any) -> int:
    """Cast every trainable parameter to fp32 (peft usually does; this makes it a guarantee).

    A GradScaler cannot unscale fp16 gradients, and bf16 LoRA weights would lose small updates.
    Returns the number of parameters that had to be converted.
    """
    import torch

    n = 0
    for p in model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
            n += p.numel()
    return n


def prepare_model(model: Any) -> None:
    """Gradient checkpointing (non-reentrant) and ``use_cache=False`` before LoRA is attached.

    Non-reentrant checkpointing needs no ``enable_input_require_grads`` (V1a); with
    ``use_reentrant=True`` the frozen embeddings would leave LoRA without gradients.
    """
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False


def resolve_precision(requested: str, cuda_major: int | None) -> str:
    """``auto`` -> bf16 on compute capability >= 8 (L4), fp16 on Turing (T4); no CUDA -> fp32."""
    if requested != "auto":
        if requested not in ("bf16", "fp16", "fp32"):
            raise ValueError(f"precision {requested!r} not in auto/bf16/fp16/fp32")
        return requested
    if cuda_major is None:
        return "fp32"
    return "bf16" if cuda_major >= 8 else "fp16"


# --------------------------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------------------------


def chunked_target_loss(hidden: Any, head_weight: Any, labels: Any, chunk: int = 256) -> Any:
    """Mean cross-entropy over the target positions, computed ``chunk`` positions at a time.

    `hidden` is ``(B, S, H)`` (final hidden states), `head_weight` ``(V, H)`` (the tied
    ``lm_head``), `labels` ``(B, S)`` with ``-100`` outside the targets. Position t predicts
    token t+1. Only target positions are gathered and each chunk's logits are rebuilt in
    backward (``checkpoint``), so peak memory is one chunk of logits (V x 256 x 4 B x ~3), not
    S x V. Returns a scalar (0.0 with no gradient path if there is no target).
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    h = hidden[:, :-1, :]
    y = labels[:, 1:]
    keep = y != IGNORE_INDEX
    n = int(keep.sum())
    if n == 0:
        raise TrainError("sample has no supervised target tokens")
    h_sel, y_sel = h[keep], y[keep]

    def part(hc: Any, yc: Any, w: Any) -> Any:
        return F.cross_entropy((hc @ w.t()).float(), yc, reduction="sum")

    total = hidden.new_zeros((), dtype=torch.float32)
    for i in range(0, n, chunk):
        total = total + checkpoint(part, h_sel[i : i + chunk], y_sel[i : i + chunk], head_weight,
                                   use_reentrant=False)  # fmt: skip
    return total / n


def lr_scale(step: int, total: int, warmup: int, min_ratio: float) -> float:
    """Linear warmup then cosine decay to `min_ratio` of the peak (step 0-based, post-warmup)."""
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    span = max(1, total - warmup)
    progress = min(1.0, (step - warmup) / span)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


# --------------------------------------------------------------------------------------------
# Deterministic sampling
# --------------------------------------------------------------------------------------------


def epoch_order(n: int, seed: int, epoch: int) -> list[int]:
    """Permutation of ``range(n)`` for an epoch; identical for the same (n, seed, epoch)."""
    return [int(i) for i in np.random.default_rng([seed, epoch]).permutation(n)]


def eval_subset(specs: Sequence[Any], seed: int, n: int) -> list[Any]:
    """The held-out pages used for the informational eval loss: a FIXED seeded subset of at most
    `n` (every evaluation of a run, mid-epoch or epoch end, scores the same pages, so the curve
    is comparable across steps and the evaluation time is bounded)."""
    return [specs[i] for i in epoch_order(len(specs), seed, 0)[:n]]


def steps_per_epoch(n: int, accum: int) -> int:
    """Optimizer steps per epoch; a partial last batch counts as a step (the V1 arithmetic)."""
    return -(-n // accum)


def step_indices(n: int, accum: int, seed: int, step: int) -> tuple[int, list[int]]:
    """``(epoch, sample indices)`` of optimizer step `step` (0-based) - a pure function."""
    spe = steps_per_epoch(n, accum)
    epoch, s = divmod(step, spe)
    order = epoch_order(n, seed, epoch)
    return epoch, order[s * accum : (s + 1) * accum]


class SampleSource(Protocol):
    """A dataset of tokenised samples (dicts of tensors); epoch drives the augmentation."""

    def __len__(self) -> int: ...

    def get(self, index: int, epoch: int) -> dict[str, Any]: ...


# --------------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------------


class MetricsLogger(Protocol):
    """Scalar-metrics sink (W&B in production, a list in tests)."""

    def log(self, metrics: Mapping[str, float], step: int) -> None: ...

    def finish(self) -> None: ...


class NullLogger:
    """Default logger: drops everything."""

    def log(self, metrics: Mapping[str, float], step: int) -> None:
        """Ignore."""

    def finish(self) -> None:
        """Ignore."""


class ListLogger:
    """Collects ``(step, metrics)``; used by tests."""

    def __init__(self) -> None:
        self.records: list[tuple[int, dict[str, float]]] = []
        self.finished = False

    def log(self, metrics: Mapping[str, float], step: int) -> None:
        """Store a copy."""
        self.records.append((step, dict(metrics)))

    def finish(self) -> None:
        """Mark finished."""
        self.finished = True


class WandbLogger:
    """W&B logger for the PRIVATE debug project only; scalars only (no images, no values).

    Curves only: nothing read back from W&B ever influences training. Raises when wandb is not
    installed or ``wandb.init`` fails (no login, no network); `make_logger` turns that into a no-op.
    """

    def __init__(self, run_name: str, config: Mapping[str, Any]) -> None:
        import wandb

        # A dead network must cost a minute at start-up, not the default 90 s per attempt.
        os.environ.setdefault("WANDB_INIT_TIMEOUT", "60")
        self._run = wandb.init(
            project=PRIVATE_WANDB_PROJECT, name=run_name, config=dict(config), reinit=True
        )

    def log(self, metrics: Mapping[str, float], step: int) -> None:
        """Forward scalars."""
        self._run.log(dict(metrics), step=step)

    def finish(self) -> None:
        """Close the run."""
        self._run.finish()


class SafeLogger:
    """Wraps a logger so that it can never fail training: the first exception switches it off
    (one printed warning) and every later call is a no-op."""

    def __init__(self, inner: MetricsLogger) -> None:
        self.inner: MetricsLogger | None = inner

    def log(self, metrics: Mapping[str, float], step: int) -> None:
        """Forward, or drop silently once disabled."""
        self._call("log", metrics, step)

    def finish(self) -> None:
        """Close, or drop silently once disabled."""
        self._call("finish")

    def _call(self, name: str, *args: Any) -> None:
        if self.inner is None:
            return
        try:
            getattr(self.inner, name)(*args)
        except Exception as exc:  # noqa: BLE001  # W&B (network, auth) must never stop a fold
            print(f"WARNING: W&B logging disabled for the rest of the run: {exc!r}"[:500])
            self.inner = None


def make_logger(
    run_name: str, config: Mapping[str, Any], env: Mapping[str, str] | None = None
) -> MetricsLogger:
    """W&B only when ``SHIPDOC_WANDB=1`` (default off); otherwise a no-op logger.

    Never raises: a missing wandb package, a missing login or an unreachable server prints one
    warning and returns the no-op logger, and a failure later in the run disables the logger
    (`SafeLogger`). Training must not depend on W&B.
    """
    env = os.environ if env is None else env
    if env.get(WANDB_ENV_FLAG) != "1":
        return NullLogger()
    try:
        return SafeLogger(WandbLogger(run_name, config))
    except Exception as exc:  # noqa: BLE001  # ImportError, UsageError (no login), timeouts, ...
        print(f"WARNING: {WANDB_ENV_FLAG}=1 but W&B is unavailable ({exc!r}); training continues "
              "without it."[:500])  # fmt: skip
        return NullLogger()


# --------------------------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------------------------


def _atomic_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


def rng_state(device: str) -> dict[str, Any]:
    """Python / numpy / torch (and CUDA when used) generator states."""
    import torch

    st: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if device.startswith("cuda"):
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st: Mapping[str, Any]) -> None:
    """Inverse of `rng_state`."""
    import torch

    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st:
        torch.cuda.set_rng_state_all(st["cuda"])


def save_checkpoint(
    ckpt_root: Path,
    step: int,
    adapter: Mapping[str, Any],
    state: Mapping[str, Any],
    keep: int = 2,
) -> Path:
    """Write ``step_<n>/{adapter.pt, trainer_state.pt}`` atomically and point ``LATEST`` at it.

    The directory is built under a ``.tmp`` name and renamed, so a kill (Colab disconnect) never
    leaves a half-written checkpoint as the newest one. Older checkpoints beyond `keep` are
    pruned only AFTER the new one is in place.
    """
    import torch

    ckpt_root.mkdir(parents=True, exist_ok=True)
    name = f"step_{step:06d}"
    tmp, final = ckpt_root / (name + ".tmp"), ckpt_root / name
    tmp.mkdir(exist_ok=True)
    torch.save(dict(adapter), tmp / "adapter.pt")
    torch.save(dict(state), tmp / "trainer_state.pt")
    if final.exists():
        for f in final.iterdir():
            f.unlink()
        final.rmdir()
    tmp.replace(final)
    _atomic_text(ckpt_root / "LATEST", name)
    old = sorted(p for p in ckpt_root.iterdir() if p.is_dir() and p.name.startswith("step_")
                 and not p.name.endswith(".tmp"))  # fmt: skip
    for p in old[:-keep] if keep > 0 else []:
        for f in p.iterdir():
            f.unlink()
        p.rmdir()
    return final


def latest_checkpoint(ckpt_root: Path) -> Path | None:
    """Directory ``LATEST`` points at, or None (no checkpoint yet / pointer dangling)."""
    ptr = ckpt_root / "LATEST"
    if not ptr.is_file():
        return None
    d = ckpt_root / ptr.read_text(encoding="utf-8").strip()
    return d if (d / "trainer_state.pt").is_file() and (d / "adapter.pt").is_file() else None


# --------------------------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------------------------

HiddenFn = Callable[[Any, dict[str, Any]], tuple[Any, Any]]


class StepHook(Protocol):
    """Called after every optimizer step with the step record (tests raise to simulate a kill)."""

    def __call__(self, record: dict[str, Any]) -> None: ...


@dataclass
class Trainer:
    """The training loop: accumulation, schedule, checkpoint/exact resume, eval, logging."""

    model: Any
    source: SampleSource
    cfg: TrainConfig
    hidden_fn: HiddenFn
    run_dir: Path
    device: str = "cpu"
    precision: str = "fp32"
    total_steps: int | None = None
    grad_accum: int | None = None
    eval_source: SampleSource | None = None
    eval_each_epoch: bool = True  # curves only: also eval at every epoch end (see `evaluate`)
    run_meta: Mapping[str, Any] = field(default_factory=dict)  # stage, manifest hash -> manifest
    logger: MetricsLogger = field(default_factory=NullLogger)
    vram_fn: Callable[[], int | None] | None = None
    on_step: StepHook | None = None
    signature: str = ""
    history: list[dict[str, Any]] = field(default_factory=list)
    step: int = 0

    def __post_init__(self) -> None:
        import torch

        self.torch = torch
        self.accum = self.grad_accum or self.cfg.grad_accum
        n = len(self.source)
        if n == 0:
            raise TrainError("empty training set")
        self.spe = steps_per_epoch(n, self.accum)
        self.total = self.total_steps or self.cfg.epochs * self.spe
        self.params = {k: p for k, p in self.model.named_parameters() if p.requires_grad}
        if not self.params:
            raise TrainError("model has no trainable parameters")
        if self.precision == "fp16" and any(p.dtype != torch.float32 for p in self.params.values()):
            # GradScaler.unscale_ refuses fp16 gradients: the LoRA weights must be fp32 masters.
            raise TrainError("fp16 training needs fp32 trainable (LoRA) weights; see ensure_fp32")
        o = self.cfg.optim
        self.opt = torch.optim.AdamW(
            list(self.params.values()),
            lr=o.lr,
            betas=(o.beta1, o.beta2),
            weight_decay=o.weight_decay,
        )
        warm = max(1, round(o.warmup_ratio * self.total))
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lambda s: lr_scale(s, self.total, warm, o.min_lr_ratio)
        )
        dev_type = "cuda" if self.device.startswith("cuda") else "cpu"
        self.scaler = torch.amp.GradScaler(dev_type, enabled=self.precision == "fp16")
        self.ckpt_root = self.run_dir / "ckpt"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.peak_vram: int | None = None
        self.preflight_peak: int | None = None
        self.metrics_path = self.run_dir / "metrics.jsonl"

    # -- helpers ------------------------------------------------------------------------------

    def _to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        dt = {"bf16": self.torch.bfloat16, "fp16": self.torch.float16}.get(self.precision)
        out = {}
        for k, v in batch.items():
            if hasattr(v, "to"):
                v = v.to(self.device, dt) if (dt and v.is_floating_point()) else v.to(self.device)
            out[k] = v
        return out

    def _loss(self, batch: dict[str, Any]) -> Any:
        hidden, head = self.hidden_fn(self.model, batch)
        return chunked_target_loss(hidden, head, batch["labels"], self.cfg.loss_chunk)

    def _vram(self) -> int | None:
        if self.vram_fn is not None:
            return self.vram_fn()
        if self.device.startswith("cuda"):
            return int(self.torch.cuda.max_memory_allocated())
        return None

    def adapter_state(self) -> dict[str, Any]:
        """Trainable parameters (the LoRA adapter) as CPU tensors."""
        return {k: p.detach().cpu().clone() for k, p in self.params.items()}

    def _state(self, epoch: int, pos: int) -> dict[str, Any]:
        return {
            "step": self.step,
            "epoch": epoch,
            "sampler_pos": pos,
            "optimizer": self.opt.state_dict(),
            "scheduler": self.sched.state_dict(),
            "scaler": self.scaler.state_dict(),
            "rng": rng_state(self.device),
            "history": self.history,
            "signature": self.signature,
            "total": self.total,
        }

    # -- checkpoint / resume ------------------------------------------------------------------

    def save(self) -> Path:
        """Checkpoint now (adapter + optimizer + scheduler + scaler + RNG + step + sampler)."""
        epoch, s = divmod(self.step, self.spe)
        return save_checkpoint(self.ckpt_root, self.step, self.adapter_state(),
                               self._state(epoch, s * self.accum), self.cfg.ckpt_keep)  # fmt: skip

    def resume(self) -> bool:
        """Restore the newest checkpoint; returns False when there is none. Exact: weights,
        optimizer moments, schedule, scaler, RNG and the position in the epoch order.

        ``metrics.jsonl`` is reset to the restored history in BOTH cases: with a checkpoint at step
        S it keeps lines 1..S, without one it is emptied (the run restarts at step 0). Skipping the
        second case appended a whole new run to the dead session's lines (29 lines for a 20-step
        smoke). Rows removed are kept in ``metrics_discarded.jsonl``.
        """
        d = latest_checkpoint(self.ckpt_root)
        if d is None:
            if self.step:
                raise TrainError(
                    f"in-memory step {self.step} but no checkpoint on disk: build a new Trainer"
                )
            self.history = []
            self._reset_metrics()
            return False
        state = self.torch.load(d / "trainer_state.pt", weights_only=False)
        if state.get("signature") != self.signature or state.get("total") != self.total:
            raise TrainError(
                f"checkpoint {d.name} was written by a different run (signature/total differ); "
                "use a new run dir or delete the old checkpoints"
            )
        step, hist = int(state["step"]), list(state["history"])
        epoch, pos = divmod(step, self.spe)
        if (
            step > self.total
            or (state["epoch"], state["sampler_pos"]) != (epoch, pos * self.accum)
            or len(hist) != step
            or ml.problems_of(hist)
        ):
            raise TrainError(
                f"checkpoint {d.name} is inconsistent: step {step}, epoch/sampler "
                f"{state['epoch']}/{state['sampler_pos']} (expected {epoch}/{pos * self.accum}), "
                f"{len(hist)} history rows, total {self.total}"
            )
        adapter = self.torch.load(d / "adapter.pt", weights_only=False)
        if set(adapter) != set(self.params):
            raise TrainError("checkpoint adapter keys do not match the model's trainable params")
        with self.torch.no_grad():
            for k, p in self.params.items():
                p.copy_(adapter[k])
        self.opt.load_state_dict(state["optimizer"])
        self.sched.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        set_rng_state(state["rng"])
        self.step = step
        self.history = hist
        self._reset_metrics()
        return True

    def _reset_metrics(self) -> None:
        """Make ``metrics.jsonl`` hold exactly ``self.history`` (steps 1..step)."""
        ml.reset_metrics(self.metrics_path, self.history,
                         self.run_dir / "metrics_discarded.jsonl")  # fmt: skip

    # -- steps --------------------------------------------------------------------------------

    def preflight(self, index: int) -> int | None:
        """One forward+backward on sample `index` WITHOUT an optimizer step (peak-memory probe)."""
        batch = self._to_device(self.source.get(index, 0))
        self.model.train()
        loss = self._loss(batch)
        self.scaler.scale(loss).backward()
        for p in self.params.values():
            p.grad = None
        self.preflight_peak = self._vram()
        return self.preflight_peak

    def _optimizer_step(self) -> dict[str, Any]:
        torch = self.torch
        t0 = time.perf_counter()
        epoch, idxs = step_indices(len(self.source), self.accum, self.cfg.seed, self.step)
        self.model.train()
        losses = []
        scale_before = float(self.scaler.get_scale())
        lr_used = float(self.opt.param_groups[0]["lr"])  # the rate of THIS update
        for i in idxs:
            batch = self._to_device(self.source.get(i, epoch))
            loss = self._loss(batch)
            losses.append(float(loss.detach()))
            self.scaler.scale(loss / len(idxs)).backward()
        self.scaler.unscale_(self.opt)
        gnorm = float(torch.nn.utils.clip_grad_norm_(list(self.params.values()),
                                                     self.cfg.optim.grad_clip))  # fmt: skip
        self.scaler.step(self.opt)
        self.scaler.update()
        self.sched.step()
        self.opt.zero_grad(set_to_none=True)
        self.step += 1
        skipped = bool(self.scaler.is_enabled() and float(self.scaler.get_scale()) < scale_before)
        peak = self._vram()
        if peak is not None:
            self.peak_vram = max(self.peak_vram or 0, peak)
        return {
            "step": self.step,
            "epoch": epoch,
            "loss": sum(losses) / len(losses),
            "grad_norm": gnorm,
            "lr": lr_used,
            "n_micro": len(idxs),
            "scaler_skipped": skipped,
            "seconds": time.perf_counter() - t0,
        }

    def evaluate(self) -> float | None:
        """Mean target loss over the eval slice (information only: NO early stopping).

        The result goes into the step record and the logger and NOWHERE else: no stopping rule,
        checkpoint choice or schedule reads it (asserted by tests/test_train.py). RNG states are
        saved and restored around it, so switching evaluation on or off cannot change the
        training trajectory.
        """
        if self.eval_source is None or len(self.eval_source) == 0:
            return None
        saved = rng_state(self.device)
        self.model.eval()
        total = 0.0
        try:
            with self.torch.no_grad():
                for i in range(len(self.eval_source)):
                    total += float(self._loss(self._to_device(self.eval_source.get(i, 0))))
        finally:
            self.model.train()
            set_rng_state(saved)
        return total / len(self.eval_source)

    def _record_eval(self, rec: dict[str, Any]) -> None:
        """Held-out loss every ``eval_every`` steps and at every epoch end (curves for W&B).

        ``eval_loss`` is set at either kind of step; ``eval_loss_epoch`` / ``epoch_done`` only at
        epoch ends (one evaluation serves both when they coincide).
        """
        if self.eval_source is None:
            return
        step = rec["step"]
        every = bool(self.cfg.eval_every) and step % self.cfg.eval_every == 0
        epoch_end = self.eval_each_epoch and step % self.spe == 0
        if not (every or epoch_end):
            return
        value = self.evaluate()
        if value is None:
            return
        rec["eval_loss"] = value
        if epoch_end:
            rec["eval_loss_epoch"] = value
            rec["epoch_done"] = step // self.spe

    def fit(self) -> list[dict[str, Any]]:
        """Run (or resume) to exactly `total` steps; returns the per-step records.

        Control flow depends on the step counter only: never on a loss value.
        """
        if self.cfg.early_stopping:
            raise TrainError("early stopping is disabled by design")
        self.resume()
        if self.device.startswith("cuda") and self.vram_fn is None:
            self.torch.cuda.reset_peak_memory_stats()
        while self.step < self.total:
            rec = self._optimizer_step()
            self._record_eval(rec)
            self.history.append(rec)
            ml.append_metric(self.metrics_path, rec)  # idempotent, keyed by rec["step"]
            self.logger.log({k: v for k, v in rec.items() if isinstance(v, (int, float))
                             and v is not None}, rec["step"])  # fmt: skip
            if self.cfg.ckpt_every and (self.step % self.cfg.ckpt_every == 0
                                        or self.step == self.total):  # fmt: skip
                self.save()
            if self.on_step is not None:
                self.on_step(rec)
        if len(self.history) != self.total:
            raise TrainError(f"{len(self.history)} history rows after a {self.total}-step run")
        self.logger.finish()
        return self.history

    def save_final(self) -> Path:
        """Write ``final/adapter.pt`` (and peft's ``save_pretrained`` format when available)."""
        out = self.run_dir / "final"
        out.mkdir(parents=True, exist_ok=True)
        self.torch.save(self.adapter_state(), out / "adapter.pt")
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(out / "peft")
        # the peft folder is what the OOF / production inference loads: bind it to this manifest
        peft_sha256 = {
            f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted((out / "peft").glob("*"))
            if f.is_file()
        }
        manifest = {
            **self.run_meta,
            "signature": self.signature,
            "steps": self.step,
            "total_steps": self.total,
            "n_train_pages": len(self.source),
            "base_repo": self.cfg.repo,
            "base_revision": self.cfg.revision,
            "precision": self.precision,
            "adapter_sha256": hashlib.sha256((out / "adapter.pt").read_bytes()).hexdigest(),
            "peft_sha256": peft_sha256,
        }
        _atomic_text(out / "manifest.json", json.dumps(manifest, indent=1))
        return out


# --------------------------------------------------------------------------------------------
# Smoke-train stage
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Check:
    """One smoke criterion: ``pass`` / ``fail`` / ``skip`` (not applicable off the real model) /
    ``warn`` (reported, never blocking)."""

    name: str
    status: str
    detail: str

    @property
    def passed(self) -> bool:
        """True unless the check failed (a skip is not a failure)."""
        return self.status != "fail"


def smoke_checks(
    history: Sequence[Mapping[str, Any]],
    *,
    first_last_k: int,
    min_rel_decrease: float,
    peak_vram: int | None,
    vram_budget: int | None,
    n_modules: int | None,
    expected_modules: int | None,
    n_trainable: int | None,
    expected_trainable: int | None,
    params_finite: bool,
    max_scaler_skips: int,
    require_vram: bool,
) -> list[Check]:
    """Evaluate the four smoke criteria (i)-(iv); pure, so it is tested without any model."""
    out: list[Check] = []
    losses = [float(r["loss"]) for r in history]
    k = first_last_k
    if len(losses) < 2 * k:
        out.append(Check("loss_decreases", "fail", f"{len(losses)} steps < 2*{k}"))
    else:
        first, last = sum(losses[:k]) / k, sum(losses[-k:]) / k
        ok = last < first * (1.0 - min_rel_decrease)
        out.append(Check("loss_decreases", "pass" if ok else "fail",
                         f"mean first {k} = {first:.4f}, mean last {k} = {last:.4f}, "
                         f"need last < first*(1-{min_rel_decrease})"))  # fmt: skip
    if peak_vram is None or vram_budget is None:
        out.append(Check("peak_vram", "fail" if require_vram else "skip", "no VRAM accounting"))
    else:
        ok = peak_vram <= vram_budget
        detail = f"peak {peak_vram / 2**30:.2f} GiB vs budget {vram_budget / 2**30:.2f} GiB"
        out.append(Check("peak_vram", "pass" if ok else "fail", detail))
    if expected_modules is None or expected_trainable is None:
        out.append(Check("lora_tripwire", "skip", "no expectation (not the real model)"))
    else:
        ok = n_modules == expected_modules and n_trainable == expected_trainable
        out.append(Check("lora_tripwire", "pass" if ok else "fail",
                         f"modules {n_modules} (want {expected_modules}), trainable "
                         f"{n_trainable} (want {expected_trainable})"))  # fmt: skip
    # An fp16 GradScaler overflow step has an inf grad norm by design: it is skipped, not a NaN.
    finite = all(
        math.isfinite(float(r["loss"]))
        and (math.isfinite(float(r["grad_norm"])) or bool(r.get("scaler_skipped")))
        for r in history
    )
    skips = sum(bool(r.get("scaler_skipped")) for r in history)
    ok = finite and params_finite and skips <= max_scaler_skips
    out.append(Check("finite", "pass" if ok else "fail",
                     f"losses/grad norms finite={finite}, params finite={params_finite}, "
                     f"scaler skips {skips} (max {max_scaler_skips})"))  # fmt: skip
    return out


def format_checks(checks: Sequence[Check]) -> str:
    """Printable per-check table."""
    return "\n".join(f"{c.status.upper():<5} {c.name:<16} {c.detail}" for c in checks)


def run_smoke(
    trainer: Trainer,
    *,
    expected_modules: int | None,
    expected_trainable: int | None,
    vram_budget: int | None,
    require_vram: bool,
    min_rel_decrease: float | None = None,
    longest_index: int | None = None,
    extra_checks: Sequence[Check] = (),
) -> list[Check]:
    """Preflight the longest sample, train ``total`` steps, check; raises `SmokeFailure`.

    Also writes and re-reads a checkpoint (the Drive path on Colab) and fails if the adapter
    does not come back bit-identical.
    """
    sm = trainer.cfg.smoke
    if longest_index is not None and sm.preflight_longest:
        trainer.preflight(longest_index)
    trainer.fit()
    reload_ok = _checkpoint_roundtrip_ok(trainer)
    checks = smoke_checks(
        trainer.history,
        first_last_k=sm.first_last_k,
        min_rel_decrease=sm.min_rel_decrease if min_rel_decrease is None else min_rel_decrease,
        peak_vram=max(filter(None, [trainer.peak_vram, trainer.preflight_peak]), default=None),
        vram_budget=vram_budget,
        n_modules=count_lora_modules(trainer.model),
        expected_modules=expected_modules,
        n_trainable=count_trainable(trainer.model),
        expected_trainable=expected_trainable,
        params_finite=all(bool(trainer.torch.isfinite(p).all()) for p in trainer.params.values()),
        max_scaler_skips=sm.max_scaler_skips,
        require_vram=require_vram,
    )
    checks.append(Check("checkpoint_roundtrip", "pass" if reload_ok else "fail",
                        "adapter.pt reloads bit-identical"))  # fmt: skip
    checks.append(metrics_file_check(trainer.metrics_path, trainer.history, trainer.total))
    checks.extend(extra_checks)
    failed = [c for c in checks if not c.passed]
    if failed:
        exc = SmokeFailure("SMOKE FAILED, full training must not start:\n" + format_checks(checks))
        exc.checks = checks
        raise exc
    return checks


def metrics_file_check(path: Path, history: Sequence[Mapping[str, Any]], total: int) -> Check:
    """``metrics.jsonl`` must hold steps 1..total once each and match the in-memory history.

    This is the check that would have flagged the first L4 smoke ("29 lines for a 20-step
    target": a restart without checkpoint appended to a dead session's lines).
    """
    try:
        read = ml.read_metrics(path, total)
    except ml.MetricsAnomaly as exc:
        return Check("metrics_file", "fail", str(exc))
    steps = [r["step"] for r in read.rows]
    ok = steps == [int(r["step"]) for r in history] == list(range(1, total + 1))
    if ok:
        return Check(
            "metrics_file",
            "pass",
            f"{len(read.rows)} lines for a {total}-step run, steps 1..{total} once each",
        )
    return Check("metrics_file", "fail",
                 f"{len(read.rows)} lines do not match the {total}-step history")  # fmt: skip


def _checkpoint_roundtrip_ok(trainer: Trainer) -> bool:
    d = latest_checkpoint(trainer.ckpt_root)
    if d is None:
        return False
    saved = trainer.torch.load(d / "adapter.pt", weights_only=False)
    return set(saved) == set(trainer.params) and all(
        trainer.torch.equal(saved[k].cpu(), p.detach().cpu()) for k, p in trainer.params.items()
    )


# --------------------------------------------------------------------------------------------
# Tokenisation (chat template identical to inference) and the real-model path (UNVERIFIED)
# --------------------------------------------------------------------------------------------


def line_items_token_mask(offsets: Sequence[tuple[int, int]], span: tuple[int, int]) -> list[bool]:
    """True for tokens whose character range overlaps the line_items span."""
    s, e = span
    return [a < e and b > s for a, b in offsets]


def encode_page(
    rendered: ts.RenderedPage,
    processor: Any,
    *,
    adapter_name: str,
    image_token_id: int,
    expected_visual_tokens: int | None,
) -> dict[str, Any]:
    """Tokenise one rendered page: inference-identical prefix + target JSON + ``<|im_end|>``.

    The prefix is ``apply_chat_template(user[image, prompt], add_generation_prompt=True, ...)``
    with the adapter's kwargs (``enable_thinking=False`` for Qwen3.5), exactly as
    ``HfBackend.extract_page`` builds it. Labels are -100 on the prefix; the target tokens and
    the stop token are supervised, except the ``line_items`` tokens of header-only documents.
    Raises `TrainError` unless the page has exactly `expected_visual_tokens` image tokens.
    """
    import torch

    messages, chat_kwargs = get_adapter(adapter_name).build(
        rendered.image, rendered.prompt, page_schema(), "json"
    )
    enc = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True,
        return_tensors="pt", **chat_kwargs,
    )  # fmt: skip
    prefix = enc["input_ids"][0]
    n_vis = int((prefix == image_token_id).sum())
    if expected_visual_tokens is not None and n_vis != expected_visual_tokens:
        raise TrainError(
            f"{rendered.spec.doc_id} p{rendered.spec.page_index}: {n_vis} visual tokens, "
            f"expected {expected_visual_tokens} (image size / max_pixels drifted)"
        )
    tok = processor.tokenizer
    t = tok(rendered.target.text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(t["input_ids"])
    keep = [True] * len(ids)
    if not rendered.target.supervise_line_items:
        masked = line_items_token_mask(t["offset_mapping"], rendered.target.line_items_span)
        keep = [not m for m in masked]
    target_labels = [i if k else IGNORE_INDEX for i, k in zip(ids, keep, strict=True)]
    im_end = int(tok.convert_tokens_to_ids(IM_END))
    input_ids = torch.cat([prefix, torch.tensor(ids + [im_end], dtype=prefix.dtype)])
    labels = torch.cat(
        [torch.full_like(prefix, IGNORE_INDEX), torch.tensor(target_labels + [im_end],
                                                              dtype=prefix.dtype)]
    )  # fmt: skip
    mm = enc.get("mm_token_type_ids")
    mm_prefix = mm[0] if mm is not None else (prefix == image_token_id).to(prefix.dtype)
    zeros = torch.zeros(len(ids) + 1, dtype=mm_prefix.dtype)
    out = {
        "input_ids": input_ids[None],
        "attention_mask": torch.ones_like(input_ids)[None],
        "labels": labels[None],
        "mm_token_type_ids": torch.cat([mm_prefix, zeros])[None],
    }
    for k in ("pixel_values", "image_grid_thw"):
        if k in enc:
            out[k] = enc[k]
    return out


def template_matches_inference(
    render_text: Callable[[list[dict[str, Any]]], str],
    messages: list[dict[str, Any]],
    target_text: str,
) -> bool:
    """True iff the template's own rendering of a final assistant turn starts with our
    ``prefix + target + <|im_end|>`` (V1d question 7). `render_text(messages)` must call
    ``apply_chat_template(..., tokenize=False)`` with the same kwargs, adding the generation
    prompt only when the last message is not from the assistant."""
    prefix = render_text(messages)
    reply = {"role": "assistant", "content": target_text}
    full = render_text([*messages, reply])
    return full.startswith(prefix + target_text + IM_END)


def template_check(processor: Any, adapter_name: str, rendered: ts.RenderedPage) -> Check:
    """Smoke check that our prefix + target + stop token is what the chat template renders for a
    final assistant turn. A mismatch is a WARNING, not a failure: the check itself (string content
    for the assistant turn) is unverified against the real template."""
    messages, kw = get_adapter(adapter_name).build(
        rendered.image, rendered.prompt, page_schema(), "json"
    )

    def render_text(msgs: list[dict[str, Any]]) -> str:
        reply_last = msgs[-1]["role"] == "assistant"
        return str(
            processor.apply_chat_template(
                msgs, add_generation_prompt=not reply_last, tokenize=False, **kw
            )
        )

    ok = template_matches_inference(render_text, messages, rendered.target.text)
    detail = (
        "chat template renders prefix + target + <|im_end|>"
        if ok
        else "template rendering differs from prefix + target + <|im_end|>: compare before trusting"
    )
    return Check("template_matches", "pass" if ok else "warn", detail)


class PageDataset:
    """`SampleSource` over page specs of a `PreparedSet` (render, augment, tokenise on demand)."""

    def __init__(self, prepared: ts.PreparedSet, specs: Sequence[ts.PageSpec], cfg: TrainConfig,
                 processor: Any, image_token_id: int, augment: bool = True) -> None:  # fmt: skip
        self.prepared, self.specs, self.cfg = prepared, list(specs), cfg
        self.processor, self.image_token_id = processor, image_token_id
        self.aug = cfg.augment if augment else ts.AugmentConfig(0.0, 0.0, cfg.seed)

    def __len__(self) -> int:
        return len(self.specs)

    def get(self, index: int, epoch: int) -> dict[str, Any]:
        """Tokenised sample `index` as it looks in `epoch`."""
        rendered = self.prepared.render(self.specs[index], epoch, self.aug)
        return encode_page(rendered, self.processor, adapter_name=self.cfg.adapter,
                           image_token_id=self.image_token_id,
                           expected_visual_tokens=self.cfg.expected_visual_tokens)  # fmt: skip

    def longest_index(self) -> int:
        """Index of the page with the longest target text (the memory worst case)."""

        def length(sp: ts.PageSpec) -> int:
            plan, gold = self.prepared.plans[sp.doc_id], self.prepared.golds[sp.doc_id]
            return len(ts.page_targets(gold, plan)[sp.page_index].text)

        return max(range(len(self.specs)), key=lambda i: length(self.specs[i]))


def hf_hidden_states(model: Any, batch: dict[str, Any]) -> tuple[Any, Any]:
    """Final hidden states of ``Qwen3_5ForConditionalGeneration`` and the tied head weight.

    Calls the inner ``model.model`` (not the wrapper), so the ``lm_head`` is never applied to the
    whole sequence; LoRA layers are injected in place, so this still runs the adapters.
    """
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    keys = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids")
    out = inner.model(**{k: batch[k] for k in keys if k in batch}, use_cache=False)
    return out.last_hidden_state, inner.lm_head.weight


def load_model_and_processor(cfg: TrainConfig, precision: str) -> tuple[Any, Any]:
    """Pinned processor (visual-token cap) + model + LoRA. UNVERIFIED: needs a GPU and peft."""
    import torch
    import transformers
    from peft import get_peft_model

    proc = transformers.AutoProcessor.from_pretrained(cfg.repo, revision=cfg.revision)
    size = dict(proc.image_processor.size)
    size["longest_edge"] = cfg.max_pixels
    proc.image_processor.size = size
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
    cls = getattr(transformers, get_adapter(cfg.adapter).model_class_name)
    model = cls.from_pretrained(cfg.repo, revision=cfg.revision, dtype=dtype, device_map={"": 0})
    prepare_model(model)
    matched = match_target_modules(model, cfg.lora.target_regex)
    if len(matched) != EXPECTED_LORA_MODULES:
        raise TrainError(f"regex matches {len(matched)} modules, expected {EXPECTED_LORA_MODULES}")
    peft_model = get_peft_model(model, build_lora_config(cfg.lora))
    ensure_fp32_trainable(peft_model)
    return peft_model, proc


def seed_all(seed: int) -> None:
    """Seed python / numpy / torch (CUDA only when it is already initialised by the caller)."""
    import torch

    random.seed(seed)
    np.random.seed(seed)  # legacy global seed on purpose, as extract.seed_everything
    torch.manual_seed(seed)


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m shipdoc.train``: build the stage's data, train (or smoke), write the summary."""
    ap = argparse.ArgumentParser(description=(main.__doc__ or "").strip())
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--stage", choices=ts.STAGES, default="smoke")
    ap.add_argument("--run-dir", type=Path, required=True, help="checkpoints/metrics (Drive)")
    ap.add_argument("--precision", default=None, help="override config: bf16 | fp16 | fp32")
    args = ap.parse_args(argv)

    import torch

    from shipdoc.spike import git_commit

    cfg = load_config(args.config)
    if not torch.cuda.is_available():
        raise TrainError("fine-tuning needs a CUDA GPU (Colab L4/T4)")
    major = torch.cuda.get_device_capability(0)[0]
    precision = resolve_precision(args.precision or cfg.precision, major)
    seed_all(cfg.seed)
    split = ts.stage_split(args.stage, ts.load_folds())
    train_ids = list(split.train_ids)
    # One cache for all stages: plans + occlusion candidates of the 500 train+dev docs (no values).
    all_ids = sorted(d for f in ts.load_folds()["folds"] for d in f["val_doc_ids"])
    cache = args.run_dir.parent / "prepared_all.json"
    scan = load_scan_params()
    if cache.is_file():
        prepared = ts.load_prepared(cache, all_ids, scan_params=scan)
    else:
        prepared = ts.build_prepared(
            all_ids, ambiguous_policy=cfg.ambiguous_policy,
            monotone_boundary=cfg.monotone_boundary, scan_params=scan,
        )  # fmt: skip
        ts.save_prepared(prepared, cache)
        ts.write_audit(prepared, cache.parent)
    prepared.scan_params = scan
    model, proc = load_model_and_processor(cfg, precision)
    image_token_id = int(model.config.image_token_id)
    smoke = args.stage == "smoke"
    specs = prepared.page_specs(train_ids)
    if smoke:
        # 2 * steps * accum pages: a deterministic prefix of the training pages.
        specs = specs[: cfg.smoke.steps * cfg.smoke.grad_accum * 2]
    held = prepared.page_specs(split.heldout_ids)
    held = eval_subset(held, cfg.seed, cfg.eval_pages)
    train_ds = PageDataset(prepared, specs, cfg, proc, image_token_id)
    eval_ds = PageDataset(prepared, held, cfg, proc, image_token_id, augment=False)
    sig = f"{cfg.signature()}|{split.manifest_hash()}|{precision}"
    trainer = Trainer(
        model, train_ds, cfg, hf_hidden_states, args.run_dir, device="cuda", precision=precision,
        total_steps=cfg.smoke.steps if smoke else None,
        grad_accum=cfg.smoke.grad_accum if smoke else None,
        eval_source=None if smoke else eval_ds,  # the smoke is 20 steps, not an eval run
        logger=make_logger(f"{args.stage}_{split.manifest_hash()}", dataclasses.asdict(cfg)),
        signature=sig,
        run_meta=build_run_meta(
            args.stage, split, cfg, n_heldout_eval_pages=len(eval_ds),
            n_lora_modules=count_lora_modules(model), n_trainable=count_trainable(model),
            code_sha=git_commit(),
        ),
    )  # fmt: skip
    summary: dict[str, Any] = {
        "stage": args.stage,
        "precision": precision,
        "signature": sig,
        "n_train_pages": len(train_ds),
        "total_steps": trainer.total,
    }
    status = args.run_dir / ("smoke_status.json" if smoke else "train_status.json")
    try:
        if smoke:
            budget = int(cfg.smoke.vram_budget_gib[precision] * 2**30)
            checks = run_smoke(
                trainer, expected_modules=EXPECTED_LORA_MODULES,
                expected_trainable=expected_lora_params(cfg.lora.r), vram_budget=budget,
                require_vram=True, longest_index=train_ds.longest_index(),
                extra_checks=[template_check(proc, cfg.adapter, prepared.render(
                    specs[0], 0, ts.AugmentConfig(0.0, 0.0, cfg.seed)))],
            )  # fmt: skip
            summary["checks"] = [dataclasses.asdict(c) for c in checks]
        else:
            trainer.fit()
            summary["final"] = str(trainer.save_final())
        summary["state"] = "passed" if smoke else "complete"
        summary["peak_vram_gib"] = (trainer.peak_vram or 0) / 2**30
        if trainer.preflight_peak is not None:
            # fit() resets the CUDA peak counter, so the longest-page probe is only visible here
            summary["preflight_peak_gib"] = trainer.preflight_peak / 2**30
    except SmokeFailure as exc:
        summary["state"] = "failed"
        summary["error"] = str(exc)
        summary["checks"] = [dataclasses.asdict(c) for c in getattr(exc, "checks", [])]
        _atomic_text(status, json.dumps(summary, indent=1))
        print(summary["error"])
        return 2
    except Exception as exc:  # OOM, bad token budget, ...: leave a status file, then fail loudly
        summary["state"] = "error"
        summary["error"] = f"{type(exc).__name__}: {exc}"[:2000]
        _atomic_text(status, json.dumps(summary, indent=1))
        raise
    _atomic_text(status, json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("stage", "state", "precision", "peak_vram_gib")}))
    return 0


if __name__ == "__main__":
    with contextlib.suppress(BrokenPipeError):
        raise SystemExit(main())
