"""Fine-tune loop on a tiny CPU model (no GPU, no transformers, no peft, no real weights).

What is real here: the chunked loss, the schedule, the sampler, accumulation, the checkpoint
format, exact resume (bit-identical adapter, optimizer and history), the smoke criteria, the
tokenisation / label-masking code and the data path up to the tokenizer. What is MOCKED: the model
(a few ``nn.Linear`` with the real module names), LoRA (a minimal wrapper with peft's attribute
names), the processor / tokenizer, and VRAM accounting. The Hugging Face loading code
(``load_model_and_processor``, ``hf_hidden_states``) is not exercised.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

# No GPU, by rule: hide every device BEFORE torch can initialise CUDA (even torch's Adam step
# touches the accelerator on a machine that has one).
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # "" still reports is_available() on Windows (NVML)
torch = pytest.importorskip("torch")
nn = torch.nn

from shipdoc import train as tr  # noqa: E402
from shipdoc import trainset as ts  # noqa: E402

VOCAB, HID, SEQ = 64, 16, 12
IMG_ID, END_ID, PAD_ID = 1, 2, 3


# --------------------------------------------------------------------------------------------
# Tiny model with the real module names, and a minimal LoRA (peft attribute names)
# --------------------------------------------------------------------------------------------


class _Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(HID, HID)
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(HID, 2 * HID)
        self.mlp.down_proj = nn.Linear(2 * HID, HID)

    def forward(self, x: Any) -> Any:
        return (
            x
            + self.mlp.down_proj(torch.nn.functional.gelu(self.mlp.gate_proj(x)))
            + (self.self_attn.q_proj(x))
        )


class TinyLM(nn.Module):
    def __init__(self, n_layers: int = 2) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.embed = nn.Embedding(VOCAB, HID)
        self.model.language_model.layers = nn.ModuleList([_Layer() for _ in range(n_layers)])
        self.visual = nn.Module()
        self.visual.proj = nn.Linear(HID, HID)  # must never be matched by the LoRA regex
        self.lm_head = nn.Linear(HID, VOCAB, bias=False)
        self.config = types.SimpleNamespace(use_cache=True)
        self.calls: list[Any] = []

    def hidden(self, input_ids: Any) -> Any:
        x = self.model.language_model.embed(input_ids)
        for layer in self.model.language_model.layers:
            x = layer(x)
        return x

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: Any = None) -> None:
        self.calls.append(gradient_checkpointing_kwargs)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float) -> None:
        super().__init__()
        self.base = base
        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)
        nn.init.zeros_(self.lora_B.weight)
        self.drop = nn.Dropout(dropout)
        self.scale = alpha / r

    def forward(self, x: Any) -> Any:
        return self.base(x) + self.lora_B(self.lora_A(self.drop(x))) * self.scale


def build_model(seed: int = 0, r: int = 16, n_layers: int = 2) -> TinyLM:
    torch.manual_seed(seed)
    model = TinyLM(n_layers)
    for p in model.parameters():
        p.requires_grad_(False)
    for name in tr.match_target_modules(model):
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        setattr(parent, leaf, LoRALinear(getattr(parent, leaf), r, 2 * r, 0.1))
    return model


def hidden_fn(model: TinyLM, batch: dict[str, Any]) -> tuple[Any, Any]:
    return model.hidden(batch["input_ids"]), model.lm_head.weight


class ListSource:
    """Samples whose target is next = (token + 1) % 40 + 4: learnable by a bigram model."""

    def __init__(self, n: int = 7, seed: int = 1) -> None:
        rng = np.random.default_rng(seed)
        self.items = []
        for _ in range(n):
            start = int(rng.integers(4, 44))
            ids = [4 + (start - 4 + k) % 40 for k in range(SEQ)]
            labels = [-100] * 4 + ids[4:]
            self.items.append({"input_ids": torch.tensor([ids]), "labels": torch.tensor([labels])})

    def __len__(self) -> int:
        return len(self.items)

    def get(self, index: int, epoch: int) -> dict[str, Any]:
        return {k: v.clone() for k, v in self.items[index].items()}


def make_cfg(**kw: Any) -> tr.TrainConfig:
    optim = tr.OptimCfg(lr=3e-2, grad_clip=1.0, warmup_ratio=0.25)
    base = dict(optim=optim, epochs=2, grad_accum=3, loss_chunk=5, ckpt_every=3, ckpt_keep=2,
                eval_every=2, seed=42)  # fmt: skip
    base.update(kw)
    return tr.TrainConfig(**base)


def make_trainer(run_dir: Path, **kw: Any) -> tr.Trainer:
    cfg = kw.pop("cfg", make_cfg())
    model = kw.pop("model", None) or build_model()
    return tr.Trainer(model, kw.pop("source", ListSource()), cfg, hidden_fn, run_dir,
                      signature="sig", eval_source=kw.pop("eval_source", ListSource(3, seed=9)),
                      **kw)  # fmt: skip


def snapshot(t: tr.Trainer) -> dict[str, Any]:
    opt = {
        i: {k: (v.clone() if torch.is_tensor(v) else v) for k, v in st.items()}
        for i, st in enumerate(t.opt.state.values())
    }
    return {"adapter": t.adapter_state(), "opt": opt, "step": t.step}


def assert_same(a: dict[str, Any], b: dict[str, Any]) -> None:
    assert a["step"] == b["step"]
    assert set(a["adapter"]) == set(b["adapter"]) and a["adapter"]
    for k in a["adapter"]:
        assert torch.equal(a["adapter"][k], b["adapter"][k]), k
    assert set(a["opt"]) == set(b["opt"])
    for i, st in a["opt"].items():
        for k, v in st.items():
            assert torch.equal(v, b["opt"][i][k]) if torch.is_tensor(v) else v == b["opt"][i][k]


# --------------------------------------------------------------------------------------------
# LoRA bookkeeping
# --------------------------------------------------------------------------------------------


def real_module_names() -> list[str]:
    """Every linear module name of the Qwen3.5-4B checkpoint, from the V1a table."""
    names = ["lm_head"]
    for n in range(32):
        p = f"model.language_model.layers.{n}"
        if n % 4 == 3:  # full attention
            names += [f"{p}.self_attn.{x}_proj" for x in "qkvo"]
        else:  # gated delta net
            gdn = ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj")
            names += [f"{p}.linear_attn.{x}" for x in gdn]
        names += [f"{p}.mlp.{x}_proj" for x in ("gate", "up", "down")]
    for n in range(24):
        v = f"model.visual.blocks.{n}"
        names += [f"{v}.attn.qkv", f"{v}.attn.proj", f"{v}.mlp.linear_fc1", f"{v}.mlp.linear_fc2"]
    names += ["model.visual.merger.linear_fc1", "model.visual.merger.linear_fc2"]
    return names


def test_regex_matches_the_200_language_model_linears_and_nothing_else() -> None:
    pat = re.compile(tr.LORA_TARGET_REGEX)
    hit = [n for n in real_module_names() if pat.fullmatch(n)]
    assert len(hit) == tr.EXPECTED_LORA_MODULES == 200
    assert not [n for n in hit if "visual" in n or "in_proj_a" in n or "in_proj_b" in n]
    assert sum("mlp" in n for n in hit) == 96 and sum("linear_attn" in n for n in hit) == 72


def test_expected_trainable_parameter_count_matches_the_shapes() -> None:
    shapes = {  # (in, out) per module kind, V1a table
        "q": (2560, 8192), "k": (2560, 1024), "v": (2560, 1024), "o": (4096, 2560),
        "qkv": (2560, 8192), "z": (2560, 4096), "out": (4096, 2560),
        "gate": (2560, 9216), "up": (2560, 9216), "down": (9216, 2560),
    }  # fmt: skip
    per_rank = 8 * sum(sum(shapes[k]) for k in "qkvo")
    per_rank += 24 * sum(sum(shapes[k]) for k in ("qkv", "z", "out"))
    per_rank += 32 * sum(sum(shapes[k]) for k in ("gate", "up", "down"))
    assert per_rank == tr.LORA_PARAMS_PER_RANK
    assert tr.expected_lora_params(16) == 30_474_240
    assert tr.expected_lora_params(8) == 15_237_120


def test_tiny_model_regex_attaches_lora_to_language_layers_only() -> None:
    model = build_model(r=4, n_layers=2)
    assert tr.count_lora_modules(model) == 2 * 3  # q_proj, gate_proj, down_proj per layer
    assert tr.match_target_modules(model) == []  # wrapped: the base linears moved one level down
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in n and "language_model" in n for n in trainable)
    # r=4: q_proj 16->16 (128), gate 16->32 (192), down 32->16 (192) per layer, 2 layers
    assert tr.count_trainable(model) == 2 * (128 + 192 + 192)


def test_prepare_model_requests_non_reentrant_checkpointing_and_no_cache() -> None:
    model = TinyLM()
    tr.prepare_model(model)
    assert model.calls == [{"use_reentrant": False}] and model.config.use_cache is False


def test_fp16_training_requires_fp32_lora_weights(tmp_path: Path) -> None:
    model = build_model().half()  # LoRA weights fp16 too: a GradScaler could not unscale them
    with pytest.raises(tr.TrainError, match="fp32 trainable"):
        make_trainer(tmp_path, model=model, precision="fp16")
    n = tr.ensure_fp32_trainable(model)
    assert n == tr.count_trainable(model) > 0
    assert all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad)
    assert tr.ensure_fp32_trainable(model) == 0  # idempotent
    assert model.lm_head.weight.dtype == torch.float16  # frozen weights keep their dtype
    make_trainer(tmp_path, model=model, precision="fp16")  # now accepted


def test_resolve_precision() -> None:
    assert tr.resolve_precision("auto", 8) == "bf16"  # L4
    assert tr.resolve_precision("auto", 7) == "fp16"  # T4
    assert tr.resolve_precision("auto", None) == "fp32"
    assert tr.resolve_precision("fp16", 9) == "fp16"
    with pytest.raises(ValueError):
        tr.resolve_precision("fp8", 8)


def test_build_lora_config_without_peft_is_an_actionable_error() -> None:
    if "peft" in sys.modules or _has("peft"):
        pytest.skip("peft installed")
    with pytest.raises(ImportError, match="peft"):
        tr.build_lora_config(tr.LoraCfg())


def _has(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


# --------------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------------


def test_committed_config_matches_the_plan() -> None:
    cfg = tr.load_config()
    assert cfg.seed == 42 and cfg.lora.r == 16 and cfg.epochs == 2 and cfg.grad_accum == 8
    assert cfg.lora.target_regex == tr.LORA_TARGET_REGEX
    assert cfg.loss_chunk == 256 and cfg.expected_visual_tokens == 1260
    assert cfg.smoke.steps == 20 and cfg.smoke.first_last_k == 5
    assert cfg.early_stopping is False and cfg.ambiguous_policy == "header_only"
    assert cfg.revision == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    assert cfg.signature() == tr.load_config().signature()
    aug = cfg.augment
    assert (aug.occlusion_rate, aug.scan_prob, aug.seed) == (0.15, 0.5, 42)
    assert aug.field_weights == {"invoice_number": 4.0, "invoice_date": 4.0}


def test_config_rejects_unknown_keys_and_early_stopping(tmp_path: Path) -> None:
    text = tr.DEFAULT_CONFIG.read_text(encoding="utf-8")
    bad = tmp_path / "bad.yaml"
    bad.write_text(text + "\nlearning_rate: 1\n", encoding="utf-8")
    with pytest.raises(TypeError):
        tr.load_config(bad)
    es = tmp_path / "es.yaml"
    es.write_text(text.replace("early_stopping: false", "early_stopping: true"), encoding="utf-8")
    with pytest.raises(ValueError, match="early stopping"):
        tr.load_config(es)
    t = make_trainer(tmp_path / "run", cfg=make_cfg(early_stopping=True))
    with pytest.raises(tr.TrainError, match="early stopping"):
        t.fit()


# --------------------------------------------------------------------------------------------
# Loss, schedule, sampler
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("chunk", [1, 3, 7, 256])
def test_chunked_loss_equals_full_cross_entropy_and_gradients(chunk: int) -> None:
    g = torch.Generator().manual_seed(0)
    h = torch.randn(2, 9, 6, generator=g, requires_grad=True)
    w = torch.randn(11, 6, generator=g)
    labels = torch.randint(0, 11, (2, 9), generator=g)
    labels[:, :3] = -100
    labels[1, 6] = -100
    loss = tr.chunked_target_loss(h, w, labels, chunk)
    loss.backward()
    h2 = h.detach().clone().requires_grad_(True)
    logits = (h2[:, :-1] @ w.t()).reshape(-1, 11)
    ref = torch.nn.functional.cross_entropy(logits, labels[:, 1:].reshape(-1), ignore_index=-100)
    ref.backward()
    assert torch.allclose(loss, ref, atol=1e-6)
    assert torch.allclose(h.grad, h2.grad, atol=1e-6)


def test_chunked_loss_supervises_only_labelled_positions_and_rejects_empty() -> None:
    h = torch.randn(1, 6, 4)
    w = torch.randn(9, 4)
    labels = torch.tensor([[-100, -100, 5, 6, -100, 7]])
    full = tr.chunked_target_loss(h, w, labels, 2)
    changed = h.clone()
    changed[0, 0] += 100.0  # position 0 predicts token 1 (-100): must not matter
    assert torch.allclose(full, tr.chunked_target_loss(changed, w, labels, 2))
    with pytest.raises(tr.TrainError, match="no supervised"):
        tr.chunked_target_loss(h, w, torch.full((1, 6), -100), 2)


def test_lr_scale_warmup_then_cosine_to_min() -> None:
    vals = [tr.lr_scale(s, 20, 2, 0.0) for s in range(20)]
    assert vals[0] == 0.5 and vals[1] == 1.0 and vals[2] == pytest.approx(1.0)
    assert all(a >= b for a, b in zip(vals[1:], vals[2:], strict=False))
    assert vals[-1] < 0.01 and tr.lr_scale(20, 20, 2, 0.1) == pytest.approx(0.1)


def test_epoch_order_is_a_deterministic_permutation() -> None:
    a = tr.epoch_order(10, 42, 0)
    assert sorted(a) == list(range(10)) and a == tr.epoch_order(10, 42, 0)
    assert a != tr.epoch_order(10, 42, 1) and a != tr.epoch_order(10, 43, 0)


def test_step_indices_cover_each_sample_once_per_epoch_with_a_partial_last_batch() -> None:
    n, accum = 7, 3
    assert tr.steps_per_epoch(n, accum) == 3
    seen = [i for s in range(3) for i in tr.step_indices(n, accum, 42, s)[1]]
    assert sorted(seen) == list(range(7)) and len(tr.step_indices(n, accum, 42, 2)[1]) == 1
    assert tr.step_indices(n, accum, 42, 3)[0] == 1  # step 3 starts epoch 1
    assert tr.step_indices(n, accum, 42, 5) == tr.step_indices(n, accum, 42, 5)


# --------------------------------------------------------------------------------------------
# Trainer: learning, checkpoints, exact resume
# --------------------------------------------------------------------------------------------


def test_training_decreases_loss_and_writes_metrics(tmp_path: Path) -> None:
    t = make_trainer(tmp_path, cfg=make_cfg(epochs=4))
    hist = t.fit()
    assert len(hist) == t.total == 4 * 3  # ceil(7/3) = 3 steps per epoch, 4 epochs
    first, last = [np.mean([r["loss"] for r in x]) for x in (hist[:3], hist[-3:])]
    assert last < 0.7 * first
    assert all(np.isfinite(r["loss"]) and np.isfinite(r["grad_norm"]) for r in hist)
    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == list(range(1, 13))
    assert [r["n_micro"] for r in rows[:3]] == [3, 3, 1]  # partial last batch of the epoch
    assert hist[0]["lr"] < max(r["lr"] for r in hist)  # warmup


def test_checkpoint_contains_everything_needed_and_prunes_old_ones(tmp_path: Path) -> None:
    t = make_trainer(tmp_path, cfg=make_cfg(ckpt_every=1, ckpt_keep=2))
    t.fit()
    root = tmp_path / "ckpt"
    names = sorted(p.name for p in root.iterdir() if p.is_dir())
    assert names == ["step_000005", "step_000006"]  # keep=2, nothing half-written left behind
    assert (root / "LATEST").read_text() == "step_000006"
    state = torch.load(root / "step_000006" / "trainer_state.pt", weights_only=False)
    for key in ("step", "epoch", "sampler_pos", "optimizer", "scheduler", "scaler", "rng",
                "history", "signature", "total"):  # fmt: skip
        assert key in state
    assert state["step"] == 6 and state["epoch"] == 2 and state["sampler_pos"] == 0
    adapter = torch.load(root / "step_000006" / "adapter.pt", weights_only=False)
    assert set(adapter) == set(t.params) and all(v.dtype == torch.float32 for v in adapter.values())
    assert {"python", "numpy", "torch"} <= set(state["rng"])
    mid = torch.load(root / "step_000005" / "trainer_state.pt", weights_only=False)
    assert mid["epoch"] == 1 and mid["sampler_pos"] == 6  # step 5 = epoch 1, third batch next


class Kill(Exception):
    pass


def kill_at(n: int):
    def hook(rec: dict[str, Any]) -> None:
        if rec["step"] == n:
            raise Kill

    return hook


@pytest.mark.parametrize("kill_step", [3, 4, 5])
def test_resume_after_a_kill_is_bit_identical_to_an_uninterrupted_run(
    tmp_path: Path, kill_step: int
) -> None:
    ref = make_trainer(tmp_path / "ref")
    ref.fit()
    run = tmp_path / "killed"
    first = make_trainer(run, on_step=kill_at(kill_step))
    with pytest.raises(Kill):
        first.fit()
    assert first.step == kill_step
    again = make_trainer(run, model=build_model())  # a fresh process: fresh model, same seeds
    again.fit()
    assert_same(snapshot(ref), snapshot(again))
    key = ("step", "loss", "grad_norm", "lr", "n_micro", "eval_loss")
    assert [{k: r.get(k) for k in key} for r in ref.history] == [
        {k: r.get(k) for k in key} for r in again.history
    ]
    on_disk = [json.loads(x) for x in (run / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in on_disk] == [1, 2, 3, 4, 5, 6]  # no duplicated steps after resume


def test_resume_of_a_finished_run_does_nothing(tmp_path: Path) -> None:
    t = make_trainer(tmp_path)
    t.fit()
    done = snapshot(t)
    t2 = make_trainer(tmp_path)
    assert t2.fit() == t.history and t2.step == t2.total
    assert_same(done, snapshot(t2))


def test_resume_refuses_a_checkpoint_from_a_different_run(tmp_path: Path) -> None:
    make_trainer(tmp_path).fit()
    other = make_trainer(tmp_path)
    other.signature = "another config"
    with pytest.raises(tr.TrainError, match="different run"):
        other.resume()


def test_eval_loss_is_logged_but_never_stops_training(tmp_path: Path) -> None:
    logger = tr.ListLogger()
    t = make_trainer(tmp_path, logger=logger)
    hist = t.fit()
    evals = [(r["step"], r["eval_loss"]) for r in hist if "eval_loss" in r]
    # every 2 steps (2, 4, 6) plus the epoch ends (3 steps per epoch: 3, 6)
    assert [s for s, _ in evals] == [2, 3, 4, 6] and all(np.isfinite(v) for _, v in evals)
    assert len(hist) == t.total  # no early stop, whatever the eval loss did
    logged = {s: m for s, m in logger.records}
    assert "loss" in logged[1] and "eval_loss" in logged[2] and logger.finished
    assert [(r["step"], r["epoch_done"]) for r in hist if "eval_loss_epoch" in r] == [
        (3, 1),
        (6, 2),
    ]
    assert logged[3]["eval_loss_epoch"] == logged[3]["eval_loss"] and "eval_loss_epoch" in logged[6]
    assert "eval_loss_epoch" not in logged[2]


def test_fp16_scaler_path_runs_and_resumes(tmp_path: Path) -> None:
    ref = make_trainer(tmp_path / "ref", precision="fp16")
    ref.fit()
    assert ref.scaler.is_enabled()
    run = tmp_path / "k"
    k = make_trainer(run, precision="fp16", on_step=kill_at(4))
    with pytest.raises(Kill):
        k.fit()
    again = make_trainer(run, precision="fp16")
    again.fit()
    assert_same(snapshot(ref), snapshot(again))
    assert float(again.scaler.get_scale()) == float(ref.scaler.get_scale())


# --------------------------------------------------------------------------------------------
# W&B logger (injected / faked)
# --------------------------------------------------------------------------------------------


def test_logger_is_off_by_default_and_only_uses_the_private_project(monkeypatch) -> None:
    assert isinstance(tr.make_logger("r", {}, env={}), tr.NullLogger)
    inits: list[dict[str, Any]] = []
    logged: list[tuple[dict[str, float], int]] = []

    class FakeRun:
        def log(self, metrics: dict[str, float], step: int) -> None:
            logged.append((metrics, step))

        def finish(self) -> None:
            logged.append(({}, -1))

    fake = types.SimpleNamespace(init=lambda **kw: inits.append(kw) or FakeRun())
    monkeypatch.setitem(sys.modules, "wandb", fake)
    lg = tr.make_logger("run1", {"lr": 1}, env={tr.WANDB_ENV_FLAG: "1"})
    lg.log({"loss": 0.5}, 3)
    lg.finish()
    assert inits[0]["project"] == tr.PRIVATE_WANDB_PROJECT == "shipdoc-extract-debug"
    assert logged == [({"loss": 0.5}, 3), ({}, -1)]


def test_wandb_failures_never_stop_training(monkeypatch, capsys) -> None:
    # (1) flag set but the package is missing / (2) init fails (no login, no network): no-op logger
    monkeypatch.setitem(sys.modules, "wandb", None)  # makes `import wandb` raise ImportError
    assert isinstance(tr.make_logger("r", {}, env={tr.WANDB_ENV_FLAG: "1"}), tr.NullLogger)
    assert "WARNING" in capsys.readouterr().out
    assert isinstance(tr.make_logger("r", {}, env={tr.WANDB_ENV_FLAG: "0"}), tr.NullLogger)

    def no_login(**kw: Any) -> None:
        raise RuntimeError("api_key not configured (no-tty)")

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(init=no_login))
    assert isinstance(tr.make_logger("r", {}, env={tr.WANDB_ENV_FLAG: "1"}), tr.NullLogger)

    # (3) the server dies mid-run: the logger switches itself off, the run is bit-identical
    class Dying:
        def __init__(self) -> None:
            self.calls = 0

        def log(self, metrics: dict[str, float], step: int) -> None:
            self.calls += 1
            if step >= 2:
                raise ConnectionError("wandb unreachable")

        def finish(self) -> None:
            raise ConnectionError("wandb unreachable")

    dying = Dying()
    safe = tr.SafeLogger(dying)
    for step in (1, 2, 3):
        safe.log({"loss": 1.0}, step)  # step 2 raises inside: swallowed, logger switched off
    safe.finish()
    assert dying.calls == 2 and safe.inner is None  # step 3 and finish() never reached it


def test_training_is_identical_with_a_dying_wandb_logger(tmp_path: Path) -> None:
    class Dying:
        def log(self, metrics: dict[str, float], step: int) -> None:
            raise ConnectionError("down")

        def finish(self) -> None:
            raise ConnectionError("down")

    ref = make_trainer(tmp_path / "ref")
    ref.fit()
    run = make_trainer(tmp_path / "run", logger=tr.SafeLogger(Dying()))
    run.fit()
    assert_same(snapshot(ref), snapshot(run))
    assert len(run.history) == run.total


# --------------------------------------------------------------------------------------------
# Smoke stage
# --------------------------------------------------------------------------------------------


def smoke_trainer(tmp_path: Path, **kw: Any) -> tr.Trainer:
    smoke = tr.SmokeCfg(steps=12, grad_accum=2, first_last_k=3, min_rel_decrease=0.2)
    cfg = make_cfg(smoke=smoke, ckpt_every=5, eval_every=0)
    return make_trainer(tmp_path, cfg=cfg, total_steps=12, grad_accum=2,
                        source=ListSource(10), **kw)  # fmt: skip


def run_ok(t: tr.Trainer, **kw: Any) -> list[tr.Check]:
    args: dict[str, Any] = {
        "expected_modules": 6,
        "expected_trainable": tr.count_trainable(t.model),
        "vram_budget": 4 * 2**30,
        "require_vram": True,
        "longest_index": 0,
    }
    args.update(kw)
    return tr.run_smoke(t, **args)


def test_smoke_passes_on_a_healthy_tiny_run(tmp_path: Path) -> None:
    t = smoke_trainer(tmp_path, vram_fn=lambda: 2**30)
    checks = run_ok(t)
    assert {c.name for c in checks} == {"loss_decreases", "peak_vram", "lora_tripwire", "finite",
                                        "checkpoint_roundtrip", "metrics_file"}  # fmt: skip
    assert all(c.status == "pass" for c in checks), tr.format_checks(checks)
    assert t.preflight_peak == 2**30 and t.step == 12


def test_smoke_aborts_when_the_loss_does_not_decrease(tmp_path: Path) -> None:
    cfg = make_cfg(optim=tr.OptimCfg(lr=0.0, warmup_ratio=0.0),
                   smoke=tr.SmokeCfg(steps=12, grad_accum=2, first_last_k=3, min_rel_decrease=0.2),
                   ckpt_every=5, eval_every=0)  # fmt: skip
    t = make_trainer(tmp_path, cfg=cfg, total_steps=12, grad_accum=2, source=ListSource(10),
                     vram_fn=lambda: 1)  # fmt: skip
    with pytest.raises(tr.SmokeFailure, match="loss_decreases"):
        run_ok(t)


def test_smoke_aborts_on_nan(tmp_path: Path) -> None:
    calls = {"n": 0}

    def nan_hidden(model: TinyLM, batch: dict[str, Any]) -> tuple[Any, Any]:
        h, w = hidden_fn(model, batch)
        calls["n"] += 1
        return (h * float("nan") if calls["n"] > 8 else h), w

    t = smoke_trainer(tmp_path, vram_fn=lambda: 1)
    t.hidden_fn = nan_hidden
    with pytest.raises(tr.SmokeFailure) as exc:
        run_ok(t)
    assert "finite" in str(exc.value) and "FAIL" in str(exc.value)


def test_smoke_aborts_over_the_vram_budget_or_without_accounting(tmp_path: Path) -> None:
    t = smoke_trainer(tmp_path / "a", vram_fn=lambda: 5 * 2**30)
    with pytest.raises(tr.SmokeFailure, match="peak_vram"):
        run_ok(t)
    t = smoke_trainer(tmp_path / "b")  # CPU, no accounting injected, real run requires it
    with pytest.raises(tr.SmokeFailure, match="no VRAM accounting"):
        run_ok(t, vram_budget=2**30)
    t = smoke_trainer(tmp_path / "c")  # mock run: the criterion is skipped, not passed
    checks = run_ok(t, vram_budget=None, require_vram=False)
    assert next(c for c in checks if c.name == "peak_vram").status == "skip"


def test_smoke_aborts_on_a_wrong_module_or_parameter_count(tmp_path: Path) -> None:
    t = smoke_trainer(tmp_path / "a", vram_fn=lambda: 1)
    with pytest.raises(tr.SmokeFailure, match="lora_tripwire"):
        run_ok(t, expected_modules=200)
    t = smoke_trainer(tmp_path / "b", vram_fn=lambda: 1)
    with pytest.raises(tr.SmokeFailure, match="lora_tripwire"):
        run_ok(t, expected_trainable=30_474_240)
    t = smoke_trainer(tmp_path / "c", vram_fn=lambda: 1)  # off the real model: skipped
    checks = run_ok(t, expected_modules=None, expected_trainable=None)
    assert next(c for c in checks if c.name == "lora_tripwire").status == "skip"


def test_smoke_checks_are_pure_and_robust_to_a_scaler_skip() -> None:
    hist = [{"loss": 1.0 - 0.05 * i, "grad_norm": 1.0, "scaler_skipped": False} for i in range(20)]
    hist[1] = {"loss": 0.95, "grad_norm": float("inf"), "scaler_skipped": True}  # fp16 overflow
    common: dict[str, Any] = {
        "first_last_k": 5,
        "min_rel_decrease": 0.05,
        "peak_vram": 1,
        "vram_budget": 2,
        "n_modules": 200,
        "expected_modules": 200,
        "n_trainable": 30_474_240,
        "expected_trainable": 30_474_240,
        "params_finite": True,
        "max_scaler_skips": 5,
        "require_vram": True,
    }
    ok = tr.smoke_checks(hist, **common)
    assert all(c.status == "pass" for c in ok), tr.format_checks(ok)
    hist[2] = {"loss": float("nan"), "grad_norm": 1.0, "scaler_skipped": False}
    assert next(c for c in tr.smoke_checks(hist, **common) if c.name == "finite").status == "fail"
    short = tr.smoke_checks(hist[:8], **common)
    assert next(c for c in short if c.name == "loss_decreases").status == "fail"


# --------------------------------------------------------------------------------------------
# Tokenisation and label masking with a fake processor
# --------------------------------------------------------------------------------------------


class FakeTok:
    def __call__(self, text: str, add_special_tokens: bool = False,
                 return_offsets_mapping: bool = False) -> dict[str, Any]:  # fmt: skip
        out: dict[str, Any] = {"input_ids": [4 + ord(c) % 40 for c in text]}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out

    def convert_tokens_to_ids(self, tok: str) -> int:
        assert tok == tr.IM_END
        return END_ID


class FakeProcessor:
    def __init__(self, n_visual: int = 1260, with_mm: bool = True) -> None:
        self.tokenizer = FakeTok()
        self.n_visual, self.with_mm = n_visual, with_mm
        self.kwargs: list[dict[str, Any]] = []

    def apply_chat_template(self, messages: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
        self.kwargs.append(kw)
        ids = [PAD_ID] * 3 + [IMG_ID] * self.n_visual + [PAD_ID] * 4
        out = {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
            "pixel_values": torch.zeros(4, 3),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
        }
        if self.with_mm:
            out["mm_token_type_ids"] = torch.tensor([[1 if i == IMG_ID else 0 for i in ids]])
        return out


def rendered(supervise_line_items: bool = True) -> ts.RenderedPage:
    gold = {
        "doc_id": "train_9700", "doc_type": "invoice", "pages": ["a.png"],
        "header": {"invoice_number": "N-1", "invoice_date": None, "supplier_name": "S",
                   "buyer_name": None, "ship_to_name": None, "currency": "USD",
                   "total_amount": "5", "awb_number": None},
        "line_items": [{"supplier_part_number": "P1", "customer_part_number": None,
                        "purchase_order": None, "quantity": "2"}],
    }  # fmt: skip
    plan = ts.plan_doc(gold, ["line"], [0])
    target = ts.page_targets(gold, plan)[0]
    target = dataclasses.replace(target, supervise_line_items=supervise_line_items)
    spec = ts.PageSpec("train_9700", "train", 0, 1, "a.png")
    return ts.RenderedPage(spec, Image.new("RGB", (8, 8)), "PROMPT", target)


def encode(r: ts.RenderedPage, proc: FakeProcessor | None = None, **kw: Any) -> dict[str, Any]:
    args: dict[str, Any] = dict(adapter_name="qwen35", image_token_id=IMG_ID,
                                expected_visual_tokens=1260)  # fmt: skip
    args.update(kw)
    return tr.encode_page(r, proc or FakeProcessor(), **args)


def test_encode_page_labels_only_the_target_and_the_stop_token() -> None:
    r = rendered()
    proc = FakeProcessor()
    b = encode(r, proc)
    n_prefix = 3 + 1260 + 4
    text = r.target.text
    ids, labels = b["input_ids"][0], b["labels"][0]
    assert ids.shape == labels.shape == (n_prefix + len(text) + 1,)
    assert (labels[:n_prefix] == -100).all()
    assert labels[n_prefix:].tolist() == ids[n_prefix:].tolist()  # supervised tokens = the inputs
    assert int(ids[-1]) == END_ID and int(labels[-1]) == END_ID
    assert int((labels != -100).sum()) == len(text) + 1  # token_budget counts target + stop
    assert b["mm_token_type_ids"].shape == b["input_ids"].shape
    assert int(b["mm_token_type_ids"][0, n_prefix:].sum()) == 0  # target tokens are text
    assert proc.kwargs[0]["add_generation_prompt"] is True
    assert proc.kwargs[0]["enable_thinking"] is False  # Qwen3.5 adapter kwargs, as at inference
    assert {"pixel_values", "image_grid_thw"} <= set(b)


def test_encode_page_derives_mm_token_types_when_the_processor_omits_them() -> None:
    b = encode(rendered(), FakeProcessor(with_mm=False))
    assert int(b["mm_token_type_ids"].sum()) == 1260


def test_encode_page_asserts_exactly_1260_visual_tokens() -> None:
    with pytest.raises(tr.TrainError, match="1259 visual tokens, expected 1260"):
        encode(rendered(), FakeProcessor(n_visual=1259))
    assert encode(rendered(), FakeProcessor(n_visual=5), expected_visual_tokens=None)


def test_header_only_docs_mask_exactly_the_line_items_tokens() -> None:
    r = rendered(supervise_line_items=False)
    b = encode(r)
    n_prefix = 3 + 1260 + 4
    s, e = r.target.line_items_span
    labels = b["labels"][0, n_prefix:-1]
    assert (labels[s:e] == -100).all()  # the row array
    assert (labels[:s] != -100).all() and (labels[e:] != -100).all()  # header, page_kind, braces
    assert int(b["labels"][0, -1]) == END_ID


def test_template_consistency_check() -> None:
    def render(messages: list[dict[str, Any]]) -> str:
        text = (
            "<|im_start|>user\nIMG PROMPT<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
        if messages[-1]["role"] == "assistant":
            return text + messages[-1]["content"] + "<|im_end|>\n"
        return text

    msgs = [{"role": "user", "content": []}]
    assert tr.template_matches_inference(render, msgs, '{"a": null}')

    def drifted(messages: list[dict[str, Any]]) -> str:
        return render(messages).replace("<think>\n\n</think>\n\n", "<think>\n\n</think>\n")

    assert not tr.template_matches_inference(
        lambda m: drifted(m) if m[-1]["role"] == "assistant" else render(m), msgs, '{"a": null}'
    )


# --------------------------------------------------------------------------------------------
# End to end: data -> tokens -> smoke -> train -> checkpoint -> kill -> resume
# --------------------------------------------------------------------------------------------


def write_corpus(root: Path, n_docs: int = 6) -> ts.PreparedSet:
    golds, asg, cands = {}, {}, {}
    (root / "train" / "images").mkdir(parents=True)
    for i in range(n_docs):
        d = f"train_{9800 + i}"
        n_pages = 1 + i % 2
        gold = {
            "doc_id": d, "doc_type": "invoice",
            "pages": [f"{d}_p{k + 1}.png" for k in range(n_pages)],
            "header": {"invoice_number": f"N-{i}", "invoice_date": "2026-01-02",
                       "supplier_name": "S", "buyer_name": "B", "ship_to_name": "T",
                       "currency": "USD", "total_amount": "5", "awb_number": None},
            "line_items": [{"supplier_part_number": f"P{i}{j}", "customer_part_number": None,
                            "purchase_order": f"PO{j}", "quantity": str(j + 1)}
                           for j in range(3)],
        }  # fmt: skip
        golds[d] = gold
        base, extra = divmod(3, n_pages)
        asg[d] = (["line"] * 3, [p for p in range(n_pages) for _ in range(base + (p < extra))])
        cands[d] = [ts.OcclusionCandidate("invoice_number", 0, ((0, (20.0, 20.0, 80.0, 34.0)),))]
        for name in gold["pages"]:
            Image.new("L", (160, 120), 240).save(root / "train" / "images" / name)
    return ts.prepare(golds, asg, candidates=cands, data_root=root, scan_params=None)


def test_end_to_end_smoke_then_train_with_kill_and_exact_resume(tmp_path: Path) -> None:
    cuda_before = torch.cuda.is_initialized()  # other test modules may have initialised it
    prepared = write_corpus(tmp_path / "data")
    specs = prepared.page_specs(prepared.plans)
    assert len(specs) == 9  # pages 1,2,1,2,1,2
    smoke = tr.SmokeCfg(steps=8, grad_accum=2, first_last_k=2, min_rel_decrease=0.1)
    cfg = make_cfg(smoke=smoke, occlusion_rate=0.5, scan_prob=0.0, grad_accum=2, ckpt_every=4,
                   eval_every=4, epochs=2, loss_chunk=64)  # fmt: skip

    def dataset(augment: bool = True) -> tr.PageDataset:
        return tr.PageDataset(prepared, specs, cfg, FakeProcessor(), IMG_ID, augment=augment)

    ds = dataset()
    sample = ds.get(0, 0)
    assert sample["input_ids"].shape == sample["labels"].shape
    assert ds.longest_index() in range(len(specs))
    # the same sample twice is identical, a new epoch re-draws the augmentation
    assert torch.equal(ds.get(3, 1)["labels"], ds.get(3, 1)["labels"])

    # 1) smoke (20-step stage in miniature): must pass before the real run starts
    smoke_run = tmp_path / "smoke"
    t = tr.Trainer(build_model(), ds, cfg, hidden_fn, smoke_run, total_steps=8, grad_accum=2,
                   signature="s", vram_fn=lambda: 10, eval_source=None)  # fmt: skip
    checks = tr.run_smoke(t, expected_modules=6, expected_trainable=tr.count_trainable(t.model),
                          vram_budget=2**20, require_vram=True,
                          longest_index=ds.longest_index())  # fmt: skip
    assert all(c.passed for c in checks), tr.format_checks(checks)

    # 2) full run, uninterrupted
    full = tmp_path / "full"
    ref = tr.Trainer(build_model(), dataset(), cfg, hidden_fn, full / "ref", signature="f",
                     eval_source=dataset(False))  # fmt: skip
    ref.fit()
    ref.save_final()
    assert (full / "ref" / "final" / "adapter.pt").is_file()
    assert ref.total == 2 * 5  # ceil(9 / 2) = 5 steps per epoch, 2 epochs

    # 3) the same run killed after step 5 (last checkpoint: step 4), then resumed
    run = full / "killed"
    k = tr.Trainer(build_model(), dataset(), cfg, hidden_fn, run, signature="f",
                   eval_source=dataset(False), on_step=kill_at(5))  # fmt: skip
    with pytest.raises(Kill):
        k.fit()
    resumed = tr.Trainer(build_model(), dataset(), cfg, hidden_fn, run, signature="f",
                         eval_source=dataset(False))  # fmt: skip
    resumed.fit()
    assert_same(snapshot(ref), snapshot(resumed))
    assert resumed.history[-1]["loss"] == ref.history[-1]["loss"]
    # the trainer code must not initialise CUDA. Relative, not absolute: when the whole suite runs,
    # test_batched_decode / extract.py (is_available + peak-memory calls) get there first
    assert torch.cuda.is_initialized() == cuda_before


def test_held_out_samples_are_never_augmented_or_nulled(tmp_path: Path) -> None:
    prepared = write_corpus(tmp_path / "data")
    specs = prepared.page_specs(prepared.plans)
    hot = make_cfg(occlusion_rate=1.0, scan_prob=1.0)
    off = make_cfg(occlusion_rate=0.0, scan_prob=0.0)
    held_out = tr.PageDataset(prepared, specs, hot, FakeProcessor(), IMG_ID, augment=False)
    clean = tr.PageDataset(prepared, specs, off, FakeProcessor(), IMG_ID, augment=True)
    train = tr.PageDataset(prepared, specs, hot, FakeProcessor(), IMG_ID, augment=True)
    nulled = 0
    for i in range(len(specs)):
        assert torch.equal(held_out.get(i, 3)["labels"], clean.get(i, 3)["labels"])
        nulled += not torch.equal(train.get(i, 3)["labels"], clean.get(i, 3)["labels"])
    assert nulled > 0  # the train side does change targets (the null for a hidden field)


class TextProcessor:
    """Renders conversations like the Qwen template (generation prompt vs final assistant turn)."""

    def __init__(self, think: str = "<think>\n\n</think>\n\n") -> None:
        self.think = think
        self.kwargs: list[dict[str, Any]] = []

    def apply_chat_template(self, messages: list[dict[str, Any]], **kw: Any) -> str:
        self.kwargs.append(kw)
        assert kw["tokenize"] is False
        text = "<|im_start|>user\nIMG PROMPT<|im_end|>\n<|im_start|>assistant\n"
        if messages[-1]["role"] == "assistant":
            assert kw["add_generation_prompt"] is False
            return text + "<think>\n\n</think>\n\n" + messages[-1]["content"] + "<|im_end|>\n"
        assert kw["add_generation_prompt"] is True
        return text + self.think


def test_template_check_passes_on_a_matching_template_and_warns_otherwise() -> None:
    r = rendered()
    ok = tr.template_check(TextProcessor(), "qwen35", r)
    assert ok.status == "pass" and ok.passed
    bad = tr.template_check(TextProcessor(think=""), "qwen35", r)  # inference prefix lacks think
    assert bad.status == "warn" and bad.passed  # reported, never blocking
    proc = TextProcessor()
    tr.template_check(proc, "qwen35", r)
    assert all(k["enable_thinking"] is False for k in proc.kwargs)  # the adapter's kwargs


def test_smoke_failure_carries_the_check_table_and_extra_checks_are_reported(
    tmp_path: Path,
) -> None:
    warn = tr.Check("template_matches", "warn", "differs")
    t = smoke_trainer(tmp_path / "a", vram_fn=lambda: 1)
    checks = run_ok(t, extra_checks=[warn])
    assert checks[-1] == warn
    t = smoke_trainer(tmp_path / "b", vram_fn=lambda: 1)
    with pytest.raises(tr.SmokeFailure) as exc:
        run_ok(t, expected_modules=200, extra_checks=[warn])
    names = {c.name: c.status for c in exc.value.checks}
    assert names["lora_tripwire"] == "fail" and names["template_matches"] == "warn"
