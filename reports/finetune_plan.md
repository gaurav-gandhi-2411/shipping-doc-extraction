# Fine-tune plan: Qwen3.5-4B LoRA, compute, CV protocol, data-builder questions (STEP V1)

**Every estimate in this report is UNVERIFIED.** Nothing here was run on a training GPU. Items marked RECOMPUTED were re-derived from a named artifact in this session with the command shown; items marked CITED come from a source fetched in this session (Sources section). Aggregates only: no page content, no extracted values. Arithmetic is reproducible with `uv run python scripts/finetune_estimate.py` (tests: `tests/test_finetune_estimate.py`).

Branch `wave-3-respike-prep`, base commit 2160954. Model: `Qwen/Qwen3.5-4B` revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` (`configs/spike_qwen35_4b_img_only.yaml`, `src/shipdoc/extract.py` `Qwen35Adapter`), transformers `5.18.0` (`uv.lock`; `pyproject.toml` group `vlm`), bitsandbytes `0.50.2`. `peft==0.21.2` is in `uv.lock` via the `train` dependency group (GG-approved after this plan was written).

## Decision summary

| Item | Recommendation |
|---|---|
| Method | bf16 LoRA r=16 on all language-model linears (200 modules, 30,474,240 trainable params), vision tower + merger frozen, gradient checkpointing (`use_reentrant=False`), loss on target positions only |
| Training GPU | **Colab L4, bf16** (permitted by spec section 2, which constrains inference only; but spec section 3 lists T4 for "all GPU training", so this needs GG's OK, see Compute) |
| Inference / reported numbers | Colab T4, fp16, adapter merged in memory (`merge_and_unload()` after loading the fp16 base), unchanged from the spike |
| Fallback 1 | T4 fp16 LoRA (fp32 LoRA weights, GradScaler, micro-batch 1, chunked loss); fits only with chunked loss and ~2 GiB headroom (UNVERIFIED) |
| Fallback 2 | T4 NF4 QLoRA (frozen base in 4-bit, vision skipped from quantization); trains on a different base than inference runs on |
| CV | K=3 as in `splits/folds.json`; K=2 saves only the fold-training part (~half of it) and none of the OOF inference, so it is a poor lever |
| Epochs / schedule | fixed 2 epochs, effective batch 8 pages, no early stopping on OOF loss (see G4 leakage note) |

---

## V1a. Qwen3.5-4B LoRA feasibility

### Architecture (CITED: HF `config.json` and `model.safetensors.index.json` at the pinned revision; transformers 5.18.0 `modeling_qwen3_5.py`)

- `architectures: ["Qwen3_5ForConditionalGeneration"]`, `model_type: qwen3_5`. Hybrid: `text_config.layer_types` has 32 entries, **24 `linear_attention` (Gated DeltaNet) and 8 `full_attention`** (`full_attention_interval: 4`, pattern L L L F). `hidden_size 2560`, `intermediate_size 9216`, `vocab_size 248320`, `tie_word_embeddings: true`, 16 heads x head_dim 256, 4 KV heads, `attn_output_gate: true`, `mtp_num_hidden_layers: 1`.
- Vision tower: `vision_config` depth 24, hidden 1024, patch 16, spatial merge 2, `out_hidden_size 2560`. Class `Qwen3_5VisionModel` (`modeling_qwen3_5.py` line 1123), blocks are `GradientCheckpointingLayer` (line 1093).
- Parameter counts (RECOMPUTED from the safetensors headers, HTTP range request to both shards at the pinned revision): total 4,659,865,088; `mtp.*` 120,599,552 (ignored on load: `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`, line 924); **loaded 4,539,265,536 = 8.455 GiB in fp16**; language-model layers 3,570,049,536 (2-D matmul weights 3,569,090,560); embeddings (tied lm_head) 635,699,200; vision blocks 302,309,376; vision merger 27,271,680. The checkpoint is BF16 (690 tensors) + 48 F32 tensors.

### Exact linear module names (checkpoint key minus `.weight`; each is an `nn.Linear`, source lines in parentheses)

| Where | Module path under `model.language_model.layers.N.` | Layers | Shape (out x in) |
|---|---|---|---|
| full attention (`Qwen3_5Attention`, 748-773) | `self_attn.q_proj` / `k_proj` / `v_proj` / `o_proj` | 8 | q 8192x2560 (2x because of the output gate), k,v 1024x2560, o 2560x4096 |
| linear attention (`Qwen3_5GatedDeltaNet`, 503-546) | `linear_attn.in_proj_qkv`, `in_proj_z`, `out_proj` | 24 | 8192x2560, 4096x2560, 2560x4096 |
| linear attention, small | `linear_attn.in_proj_a`, `in_proj_b` | 24 | 32x2560 (decay / beta gates) |
| linear attention, non-linear | `linear_attn.conv1d` (depthwise `nn.Conv1d`), `A_log`, `dt_bias`, `norm.weight` | 24 | not LoRA targets |
| MLP (`Qwen3_5MLP`, 823-831) | `mlp.gate_proj`, `up_proj`, `down_proj` | 32 | 9216x2560, 9216x2560, 2560x9216 |

Vision modules (must not be hit): `model.visual.blocks.N.attn.qkv`, `attn.proj`, `mlp.linear_fc1`, `mlp.linear_fc2`, `model.visual.merger.linear_fc1`, `linear_fc2`, `model.visual.patch_embed.proj` (Conv3d), `pos_embed`. Note the vision tower's `attn.proj` ends in `.proj` and `attn.qkv` in `.qkv`: a loose suffix list such as `["proj", "qkv", "out_proj"]` WOULD hit them; the names `q_proj`/`k_proj`/`v_proj`/`o_proj`/`gate_proj`/`up_proj`/`down_proj` do not collide, but only the anchored regex below is checked.

### Excluded modules and why

| Module | Decision | Reason |
|---|---|---|
| `model.visual.*` blocks | **frozen** | (1) The failure mode seen in the spike is LLM-side field mapping (rotations / column shifts among part number, customer part number, PO: `reports/spike_diagnosis.md`), not glyph reading; 190/190 qwen35_4b img_only pages parsed as valid JSON (RECOMPUTED below). (2) LoRA on the tower would put 5,040 patches x 24 blocks into the autograd graph: vision forward is ~5.6 TFLOP/page and would add recompute + backward (~+11 TFLOP, ~+17% per page by the script's FLOP model) plus the largest activations. (3) 500 docs / 18 invoice layouts: more trainable capacity raises overfitting risk, which G4 exists to catch. Revisit only as a separate ablation if G4 passes and misreads remain. |
| `model.visual.merger.*` | **frozen** | Same reasons; it is the projector, small (27M), a tempting v2 ablation (`linear_fc1`/`linear_fc2` are plain `nn.Linear`, peft-compatible), not a v1 default. |
| `lm_head` / `embed_tokens` | **frozen** | Tied (`_tied_weights_keys`, line 1766). Training them forces `modules_to_save` and untie/duplicate handling on 636M params; no gain expected for fixed JSON vocabulary. |
| `in_proj_a`, `in_proj_b` | **excluded (choice, not a measured hazard)** | They produce the decay `g` and write-gate `beta` of the recurrence (modeling lines 570-618); perturbing them changes dynamics in a numerically delicate path for 0.4M params of capacity. They are `nn.Linear`, so adding them is possible later. |
| `conv1d`, `A_log`, `dt_bias`, norms | frozen | Not matmul adapters; `A_log` is deliberately upcast to float because "if the model is loaded in fp16, without the .float() here, A might be -inf" (modeling line 617). |
| `mtp.*` | not loaded | Ignored by the loader. A merge-and-save path would drop them; do not save a merged checkpoint (see Risks). |

### Ready-to-use peft config (checked against the real checkpoint key names; NOT instantiated, because the throwaway venv has no torch)

```python
from peft import LoraConfig

LORA = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM",
    # str => re.fullmatch against the module path (peft/utils/other.py:1516). Anchored to
    # model.language_model so the vision tower can never match.
    target_modules=(
        r"model\.language_model\.layers\.\d+\."
        r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)"
        r"|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)"
        r"|mlp\.(gate_proj|up_proj|down_proj))"
    ),
)
```

RECOMPUTED (python, `re.fullmatch` over every non-`mtp` module name derived from the weight index `$SHIPDOC_TMP_DIR\hf\model.safetensors.index.json`): **200 matches** = 32 each of `mlp.{gate,up,down}_proj`, 24 each of `linear_attn.{in_proj_qkv,in_proj_z,out_proj}`, 8 each of `self_attn.{q,k,v,o}_proj`; **0 matches under `model.visual`**. LoRA parameters (from the tensor shapes): r=8 15,237,120; **r=16 30,474,240**; r=32 60,948,480; the attention-only subset (no MLP) is 12,386,304 at r=16. Smoke-train tripwires: `len(lora modules) == 200` and `trainable == 30,474,240`.

peft facts (CITED, peft 0.21.2 source): there is no default `target_modules` entry for `qwen3_5` (grep of the package finds no `qwen3_5`/`qwen3_vl`), so `target_modules` is mandatory. A `str` is matched with `re.fullmatch`, a list by exact key or `endswith("." + name)` (`tuners/tuners_utils.py:2337-2376`). `get_peft_model(..., autocast_adapter_dtype=True)` is the default (`mapping_func.py:110`) and upcasts fp16/bf16 adapter weights to fp32 (`tuners_utils.py:2705-2730`), which is what fp16 + GradScaler needs. All targeted modules are `nn.Linear`, so the standard LoRA `Linear` and, for NF4, `Linear4bit` (`tuners/lora/bnb.py:313`) paths apply; `Conv1d` is not targeted.

### Gradient checkpointing (CITED, transformers 5.18.0)

- `supports_gradient_checkpointing = True` (line 919); decoder layers and vision blocks subclass `GradientCheckpointingLayer`. `gradient_checkpointing_enable()` defaults to `{"use_reentrant": False}` (`modeling_utils.py:3139`). With non-reentrant checkpointing the "frozen embeddings => inputs have no grad => LoRA gets no gradient" trap does not apply, so `enable_input_require_grads()` is not needed (peft only calls it for reentrant + quantized models: `peft/utils/other.py:229-233`; `peft_model.py:734` calls it when preparing a non-quantized model, harmless). If anyone passes `use_reentrant=True`, call `model.enable_input_require_grads()`.
- Layers must receive `hidden_states` positionally under reentrant mode (`modeling_layers.py:60-74`); `Qwen3_5TextModel.forward` already calls `decoder_layer(hidden_states, ...)` positionally (line 1291).
- Under checkpointing the layer sets `past_key_values=None` and `use_cache=False` with a warning (`modeling_layers.py:81-110`): train with `use_cache=False` explicitly.
- Linear-attention kernels: `chunk_gated_delta_rule`, `fused_recurrent_gated_delta_rule`, `causal_conv1d_*` use the optional `fla` / `causal_conv1d` packages and otherwise fall back to PyTorch references, with a logged warning that "this is correct but much slower" (`integrations/hub_kernels.py:984-1044`; reference chunk loop at `modeling_qwen3_5.py:299-433`, fp32 math, Python loop over 64-token chunks). Neither package is in `uv.lock`, so the spike inference already ran on the fallback; the fallback is differentiable (`torch.linalg.solve_triangular`, plain ops). Whether `flash-linear-attention` compiles and is faster on T4 (sm75) is **UNVERIFIED** (its README claims NVIDIA/AMD/Intel support, no T4 statement); do not add it without a timed smoke comparison.
- Vision tower under training: parameters frozen and `pixel_values` need no grad, so no autograd graph is built for the tower; its forward still costs ~5.6 TFLOP/page. `get_image_features` requires `image_grid_thw` and the processor's `mm_token_type_ids` (`compute_3d_position_ids`, modeling lines 1548-1555, raises `ValueError` without `mm_token_type_ids`); pass both from `processor(...)`.
- **Full-sequence logits are the largest avoidable allocation.** `Qwen3_5ForConditionalGeneration.forward` computes `lm_head` on all positions when `logits_to_keep=0` (lines 1878-1885) and then upcasts for the loss. At 2,631 tokens x 248,320 vocab that path is ~7.3 GiB (fp16 logits + fp32 upcast + their gradients, script arithmetic), at the longest page (3,344 tokens) ~9.3 GiB. Compute the loss from `model.model(...)` hidden states gathered at target positions only, in chunks of 256 positions with per-chunk checkpointing (0.71 GiB, arithmetic below).

### fp16 vs bf16 on T4 / L4

- T4 (Turing) has no native bf16 (spec section 2, `configs/spike_qwen35_4b_img_only.yaml` comment); fp16 training needs loss scaling (`torch.amp.GradScaler`) and fp32 trainable weights (peft upcasts them by default, above).
- What is known about Qwen-family fp16 activations: **(a) measured here:** fp16 *inference* of this exact model is clean on our data: 55/55 (spike40 compact) and 135/135 (dev100 keyed) qwen35_4b img_only pages produced valid JSON at 2abf481 (RECOMPUTED, `json_valid` over `$SHIPDOC_RUNS_DIR\spike_download2\x\*\trace.jsonl`). **(b) vendor claim, not verified by us:** unsloth lists `qwen3_5` among families whose "activations [are] outside float16's finite range" and force-loads them in bf16/fp32 (unslothai/unsloth PR #11533, open, body text; and merged PR #5880 `de3c745fab08e14a1cd7a825c434f048a53ec5ef` states FORCE_FLOAT32 models incl. `qwen3_5` "fell through to fp16 autocast and produced NaNs" on V100). Neither gives a measurement or a model size, so for the 4B they are a risk flag, not a result. **(c)** The transformers code already guards one known fp16 hazard (`A_log`, line 617). Training adds backward + loss scaling, so inference cleanliness does not prove training stability.
- Plan: the 20-step smoke must log `max |residual stream|` per layer (forward hooks), loss, grad-norm and scaler skips; any NaN/inf in the first 20 steps on T4 fp16 closes the fp16 path. bf16 on L4 has fp32's exponent range and needs neither scaler nor fp32 masters beyond the peft default.
- T4-trained vs L4-trained adapters are not interchangeable bit-for-bit, but all reported numbers come from T4 fp16 inference with the adapter merged in memory, so the evaluation numerics are the same in both routes.

### Memory (GiB; UNVERIFIED estimates; weights and the vision/inference transient are anchored to the measured inference peak)

Measured anchor (RECOMPUTED, `peak_vram_bytes` over the 2abf481 traces): inference peak **9.12 GiB** (spike40 compact, 55 pages; dev100 9.11; ab_keyed 9.41 max) = 8.455 GiB weights + ~0.66 GiB transient (vision + prefill + KV).

| Component | fp16 LoRA, micro-batch 1, mean page (S=2,631, T=531) | longest gold page (S=3,344, T=1,244) | assumption |
|---|---:|---:|---|
| base weights fp16 (4.54B) | 8.46 | 8.46 | script constant `N_LOADED` |
| LoRA r16 fp32 weights + grads + Adam m,v | 0.45 | 0.45 | 30.47M x 4 B x 4 |
| checkpointed layer inputs (32 x S x 2560 x 2 B) | 0.40 | 0.51 | |
| one-layer recompute + backward transient | ~1.3 | ~1.6 | MLP 0.36/0.46 computed; GDN fp32 intermediates ~0.9-1.1 ASSUMED |
| vision + misc transient (measured at inference) | 0.66 | 0.66 | inference peak minus weights |
| loss, target positions only, unchunked | 1.47 | 3.45 | fp16 logits + fp32 upcast + 2 grads |
| loss, chunks of 256 positions | 0.71 | 0.71 | |
| **total, unchunked loss** | **12.7** | **15.1** | |
| **total, chunked loss** | **12.0** | **12.4** | |

Reading: a T4 is a "16 GB" part (NVIDIA datasheet); Colab shows it as 15 GB (varlog table); the usable figure is ~14.5 GiB (UNVERIFIED). So **fp16 LoRA fits the T4 only at micro-batch 1 with the chunked loss (~2 GiB headroom, allocator fragmentation can eat it; the unchunked loss OOMs on the longest pages)**; the default HF full-sequence-logits path (~+7 GiB) does not fit. On an L4 (24 GB, Colab shows 22 GB per the varlog table) bf16 weights are the same 8.455 GiB and everything above fits with ~8-10 GiB spare, even with the unchunked target-only loss. NF4 QLoRA on T4: linear weights 3.57B x ~0.516 B = 1.72 GiB, embeddings 1.18 GiB, vision kept fp16 0.62 GiB = **~3.5 GiB** (saves ~4.9 GiB, so micro-batch 2-3 fits), but dequantization overhead (unmeasured) slows steps, and the adapter is fitted to the quantized base while inference uses the fp16 base. If used, skip quantizing `visual`, `lm_head` and the two tiny `in_proj_a/b` (the `llm_int8_skip_modules` matching semantics were not checked here).

---

## V1b. Compute decision and estimates

### Is training on L4 allowed?

Spec section 2 ("Hard constraints") limits **inference** to one GPU of at most 24 GB or CPU, and names the free Colab T4 as the primary *target*. It says nothing that binds training hardware. Spec section 3's compute table lists "Colab T4 (primary): all GPU training and inference whose results appear in the main report". So: **allowed by section 2, but a deviation from the section 3 plan.** Reported accuracy numbers would be produced by T4 fp16 inference in either case (an L4-trained adapter is evaluated on T4); I recommend recording the L4-training decision in PLAN.md and getting GG's explicit OK (it changes which Colab tier is needed: the FAQ says premium GPUs are available only on paid plans "subject to availability").

### Inputs (RECOMPUTED unless marked)

| Quantity | Value | Source |
|---|---:|---|
| Visual tokens / page | 1,260 | `n_visual_tokens` in all four traces; 1240x1754 pages, `max_pixels=1280*32*32` |
| Input tokens / page (visual + prompt v2 text) | 2,093-2,136, use 2,100 | `n_input_tokens` mean: dev100 img_only 2,093 (135 pages), spike40 compact 2,136 (55 pages), traces at 2abf481. The text prompt is therefore ~840 tokens |
| Target tokens / page (keyed json, train pages) | mean 531.3, p99 1,190, max 1,244 | `reports/token_budget.md` (UNVERIFIED there; re-derivation not repeated here) |
| Sequence / page | mean ~2,631, p99 ~3,290, max ~3,344 | sum |
| Pages | all-train 536; train+dev 671; fold train 441 / 454 / 447; fold held-out 230 / 217 / 224 | `scripts/finetune_estimate.py` over `data/*/labels` and `splits/folds.json` (sha256 prefix `03b5a8b44ea9bac8`) |
| Train pages minus exclusions | 536 minus 0-44 | see V1d (strict worst case excludes the 22 multipage train docs with an unassigned row = 44 pages; the recommended policy excludes ~5 docs) |
| Tokens / epoch | all-train 1,410,377 (supervised 284,777 = 20%); a fold (mean 447 pages) 1,177,068 | script |
| Steps (micro-batch 1, accumulation 8 => effective batch 8, 2 epochs) | all-train 134; folds 112 / 114 / 112 (3 epochs: 1.5x) | script |

### Throughput model (UNVERIFIED, every parameter labelled)

1. **FLOPs per training page** (script `train_flops_per_page`): frozen base + LoRA + gradient checkpointing => language layers cost 3x forward (forward, recompute, input-gradient only; no weight-gradient matmuls for frozen weights) = 56.3 TFLOP; attention scores (8 full-attention layers, same 3x) 2.7; logits on target positions only (forward + input gradient) 1.4; vision tower forward only 5.6 (5,040 patches; matmuls 3.0 + full attention 2.5 modeled); **total 66.0 TFLOP/page** at S=2,631. The gated-delta recurrence FLOPs are not modeled (small next to the matmuls, but the PyTorch fallback is probably launch- and bandwidth-bound (hypothesis, unmeasured); this is why the low scenario exists).
2. **Anchor:** measured inference prefill intercept 2.05 s/page (OLS of `latency_s` on `n_output_tokens`, dev100 qwen35_4b img_only, R^2 0.9997; it also holds fixed per-call overhead and the first decode step) for a forward of ~20.6 TFLOP (vision 5.6 + 2,100 tokens x 7.14 GFLOP) => **>= 10.1 TFLOP/s effective on T4, 16% of the 65 TFLOP/s datasheet peak** (a lower bound on the prefill rate).
3. **T4 training effective rate:** low 5.0 / **central 7.5** / high 10.1 TFLOP/s = 0.5 / 0.75 / 1.0 x the inference anchor (training adds fp32 LoRA branches, optimizer, scaler, and the differentiated fp32 gated-delta fallback). This factor is a judgment, not a measurement.
4. **L4 training rate:** T4 rate x 1.2 / 1.6 / 1.9. Basis: L4 dense FP16/BF16 tensor peak is 121 TFLOP/s (NVIDIA lists 242 "shown with sparsity; one-half lower without") vs T4 65, ratio 1.86; memory bandwidth is the same 300 GB/s on both, so bandwidth-bound ops (norms, the fp32 recurrence) do not speed up, hence the lower low/central.
5. **CU rates:** T4 1.19 (mccormickml.com via `configs/spike_speed.json`, not re-fetched; URL not recorded in the repo) and 1.58 (varlog.info, measured 2024-10-01, re-fetched: row "T4 ... 1.58"); **L4 3.00 (same varlog.info table, same date, Colab Pro)**. The Colab pricing page returned HTTP 403 to curl and the FAQ lists no rates, so there is no first-party L4 rate; third-party, 2024-10-01, may have changed. Bracket: L4 = 3.00 CU/h is the only citable value; if the T4 rate has dropped to 1.19 since 2024, the L4 rate may also be lower (unknown).
6. Model load 120 s per run (same ESTIMATE as `configs/spike_speed.json`); the smoke = 20 optimizer steps x accumulation 2 = 40 pages (an assumption about what "20-step" means).

### Estimate table (2 epochs, effective batch 8; hours = GPU hours incl. load; `scripts/finetune_estimate.py` output)

| Stage | T4 low / central / high h | T4 CU @1.19-1.58, central | L4 low / central / high h | L4 CU @3.00, low / central / high |
|---|---|---:|---|---|
| smoke, 20 steps | 0.18 / 0.13 / 0.11 | 0.2 | 0.16 / 0.09 / 0.07 | 0.5 / 0.3 / 0.2 |
| one fold (441 pages x 2 epochs) | 3.25 / 2.18 / 1.64 | 2.6-3.4 | 2.72 / 1.37 / 0.88 | 8.1 / 4.1 / 2.6 |
| all 3 folds | 9.90 / 6.63 / 5.00 | 7.9-10.5 | 8.26 / 4.18 / 2.68 | 24.8 / 12.5 / 8.0 |
| final all-train (536 pages x 2 epochs) | 3.95 / 2.64 / 1.99 | 3.1-4.2 | 3.29 / 1.66 / 1.06 | 9.9 / 5.0 / 3.2 |
| s/page (training) | 13.1 / 8.8 / 6.6 | | 11.0 / 5.5 / 3.5 | |

Inference stages (always on T4 fp16, adapter merged in memory; mean 36.0 s/page RECOMPUTED from the dev100 keyed trace at 2abf481, 135 pages; high case 42.5 s from the `spike_speed.json` formula at 531.3 tokens):

| Stage | Pages | T4 hours (36.0-42.5 s/page) | CU @1.19-1.58 (at 36.0 s) |
|---|---:|---:|---:|
| OOF inference, 3 folds (all 500 docs) | 671 | 6.71-7.92 | 8.0-10.6 |
| final model on dev100 | 135 | 1.35-1.59 | 1.6-2.1 |

**The OOF and dev inference (8.06 h at 36 s/page) cost about as much as the whole training in the recommended route.** Compute-unit totals for Phase 4 (training + these inference stages, no reruns, no zero-shot OOF which is budgeted elsewhere): L4 route central 5.93 h x 3.00 = 17.8 CU + 9.6-12.7 CU = **27.4-30.5 CU**, bracket 21.0-24.1 (L4 high) to 44.7-47.8 (L4 low); T4 route central 9.40 h => 11.2-14.9 CU + 9.6-12.7 = **20.8-27.6 CU**, bracket 18.0-23.9 (T4 high) to 26.3-34.9 (T4 low). Wall-clock: L4 route ~14.0 GPU hours (5.9 L4 + 8.1 T4), T4 route ~17.5 GPU hours. No Colab CU balance was available to compare against (Colab exposes none programmatically; `configs/spike_speed.json` note). Sessions are capped at 12 h on paid plans (FAQ: 12 h, 24 h on Pro+), so every stage above fits one session, one fold per session.

### Recommendation

**Train on L4 bf16 LoRA r=16 (config above), 2 epochs, effective batch 8, fixed schedule; infer on T4 fp16 with the adapter merged in memory.** Reasons: (1) the L4 route is ~37% shorter in training wall-clock at central (5.93 vs 9.40 h) for +20% (vs the 1.58 T4 rate) to +59% (vs 1.19) CU on the training part (17.8 vs 14.9-11.2), and the all-in CU difference is within the bracket's noise because T4 inference dominates both; (2) memory: L4 removes the T4's ~2 GiB headroom risk (OOM on the longest pages is a mid-fold failure, the most expensive kind); (3) bf16 avoids the vendor-flagged fp16 activation-range risk and the scaler; (4) the inference numerics that reach the report are identical either way. **Do first, before any fold:** the 20-step smoke on L4 (~0.1 h, 0.3 CU) to turn the table into measurements (s/step, peak VRAM, the two tripwires 200 / 30,474,240, no NaN), plus the same smoke once on T4 (~0.13 h, ~0.2 CU) to decide whether the fallback is real; then one fold on L4, OOF-infer it on T4, and **stop if fold-0 invoice-only point ΔOVERALL <= 0** (a futility stop; it can only prevent a G4 pass, it cannot create one, so it does not bias the decision), saving ~2.8 h L4 + ~4.4 h T4 inference (folds 1-2: 441 held-out pages x 36 s).

**Fallback:** T4 fp16 LoRA exactly as above (chunked loss, micro-batch 1, GradScaler), if the smoke shows no NaN and peak VRAM <= ~13.5 GiB; else T4 NF4 QLoRA (micro-batch 2). Use the fallback also if L4 is unavailable (availability is not guaranteed).

### K=3 vs K=2 if compute forces fewer folds

K=2 would need a new `splits/folds.json` (re-run `scripts/make_folds.py` with K=2, seed 42; not done here). Fold-training hours (script): K=3 vs K=2 = 6.63 vs 3.33 (T4 central), 4.18 vs 2.11 (L4 central). That saves ~2.1 L4 hours (~6 CU) out of ~14 GPU hours: **OOF inference stays 671 pages either way**, so K=2 is a weak lever; prefer 1 epoch or fold-0 futility stopping first. What G4 loses statistically: (a) all 500 docs still get one OOF prediction, so the paired-CI width, which is driven by n=500 docs and 18 invoice-supplier clusters, barely changes; (b) **bias**: each K=2 fold model trains on ~9 of 18 invoice layouts (K=3: 12 of 18; final model: 18), so OOF understates the final model's unseen-supplier accuracy more and G4 becomes harder to pass (conservative, not anti-conservative); (c) the number of independent training runs behind the verdict falls from 3 to 2, so run-to-run variance (never captured by a document-level bootstrap, see V1c) is even less observable; (d) held-out groups per fold become 9 invoice suppliers (instead of 6).

---

## V1c. CV protocol text

**Folds** (`splits/folds.json`, sha256 prefix `03b5a8b44ea9bac8`, last changed in commit `d1112ecd207b47fba02c26065ee5bd3242fac5ef`; K=3, seed 42, `group_key: supplier_group`, over train+dev = 500 docs). RECOMPUTED with `uv run python scripts/finetune_estimate.py` (`fold_balance`, reads `meta/{train,dev}.json`):

| Fold | Held-out docs | Invoice docs | Waybill docs | Held-out invoice suppliers | Held-out waybill groups (carrier brand) | Invoice docs per held-out supplier | Held-out pages (invoice-only) | Dev docs in held-out | Train/val group overlap |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 0 | 171 | 134 | 37 | 6 | 4 | 16, 19, 22, 22, 23, 32 | 230 (193) | 33 | 0 |
| 1 | 165 | 134 | 31 | 6 | 3 | 18, 19, 22, 23, 25, 27 | 217 (186) | 31 | 0 |
| 2 | 164 | 132 | 32 | 6 | 3 | 16, 19, 22, 23, 24, 28 | 224 (192) | 36 | 0 |

Totals: 400 invoice docs over 18 suppliers (6 per fold), 100 waybills over 10 carrier groups. Every fold has both types (balance ok). **Invoice-only OOF headline: n = 400 docs (571 pages) from 18 suppliers; each fold holds out 6 genuinely unseen layouts.** Waybill groups are carrier brands that do not track layout (spec Phase 0: best cluster ARI 0.133), so the waybill part of "unseen" is weaker than for invoices; this is why the invoice-only number is the headline and overall OOF is secondary. (Invoice pages per fold sum 193+186+192 = 571; the pairing of pages to doc type uses `meta`.)

**Procedure.**
1. For fold k in {0,1,2}: train a fold model on the other two folds' docs (441 / 454 / 447 pages before exclusions; contains dev docs; held-out suppliers never appear in training, overlap column above). Fixed schedule, hyper-parameters pre-registered in `configs/` before fold 0 is run. Predict fold k on T4 fp16 with greedy decoding and the Phase 3 post-processing, producing one OOF prediction per doc.
2. Compare, per doc and paired, **fine-tuned model + Phase 3** against **zero-shot + Phase 3** (G3 output) on the same 500 docs. Post-processing parameters learned from labels (page-provenance table, date-format inference) are re-fit inside each fold's training docs for both arms; otherwise the comparison leaks held-out suppliers into one or both arms.
3. Paired bootstrap, 2,000 resamples at document level, seed 42 (spec section 5), for OVERALL, its components, and the false-fill / over-null rates, for (a) invoice-only OOF (headline) and (b) all 500 docs. Add, as a sensitivity check that is not a gate, a **supplier-cluster bootstrap** (resample the 18 invoice suppliers): a document-level bootstrap treats the ~22 docs of one supplier as independent although a fine-tune's gain is correlated within a layout, so it is optimistic about the CI width.

**G4 rule (verbatim):** keep the fine-tune only if OOF OVERALL beats zero-shot + post-processing OOF with a paired CI excluding 0 AND the unseen-supplier slice false-fill/over-null is not worse.

Operational definitions proposed for GG to confirm (they are not in the spec; chosen to mirror the A/B rule's style in spec Phase 2.3): "beats" = point ΔOVERALL > 0 AND the 95% paired CI of ΔOVERALL excludes 0, evaluated on invoice-only OOF (headline) and required not to be contradicted on all-500 OOF; "unseen-supplier slice" = every OOF prediction (all are held-out suppliers by construction), false-fill rate split redaction vs absent-line per spec Phase 5.6, and over-null = predicted null where gold is non-null; "not worse" = point Δ(fine-tuned − zero-shot) <= 0 OR the paired CI of Δ includes 0, for each of the two rates. Over-null must be included because a model can lower false fills just by nulling more, which costs OVERALL on fields it could read.

**Leakage note (departs from spec Phase 4.3).** The spec says "early stopping on held-out-supplier eval loss". Selecting an epoch count or checkpoint on the fold's own held-out loss uses the very docs whose OOF score decides G4, which biases OOF upward. The final all-train model has no held-out supplier at all (dev shares its suppliers with train). I therefore recommend a **fixed 2-epoch schedule**, logging held-out loss for information only; any tuning must use a split nested inside the training folds.

**Which model produces which number** (the folds cover train+dev, so the roles must not be mixed):

| Number | Model | Data scored | Meaning |
|---|---|---|---|
| OOF OVERALL / invoice-only OOF / unseen false-fill / G4 | the 3 fold models (each scored only on its own held-out fold) | all 500 docs, held-out suppliers | unseen-supplier accuracy; headline = invoice-only |
| Official dev OVERALL (seen layouts) | the final model, trained on all 400 train docs only | the official 100 dev docs | seen-layout accuracy |
| Test predictions | the final model | 200 test docs | ~half unseen suppliers (spec Phase 6.4 estimate) |

Consequences: the final model **cannot** be scored on dev as unseen (dev suppliers are all in train); the fold models **must not** be scored on the official dev set (each has seen about two thirds of the dev docs, 33/31/36 dev docs held out per fold vs 100 total); and the final model is trained on 400 docs while the fold models use ~333, so OOF is a (mildly) pessimistic proxy for the final model's unseen-layout accuracy. Expected test performance is roughly a mixture of the OOF number (unseen half) and the dev number (seen half); that mixture is an expectation for the report, not a measurement.

---

## V1d. Risks and open questions for the training-data builder

### Facts the builder starts from (RECOMPUTED from `$SHIPDOC_RUNS_DIR\provenance\locations.jsonl`, sha256 prefix `6e1961394178486a`, row records only)

- 4,930 gold rows in 400 invoice docs (train 4,006, dev 924). Alignment level: `line` 4,545; `page_fuzzy` 299 (page known, no distinct OCR line); `unassigned` **86 = 1.7%** (train 70, dev 16) (`reports/provenance.md`). Unassigned rows: train 20 in single-page docs (trivially page 0) and **50 in multipage docs (22 docs, 44 pages)**; dev 16 in 5 multipage docs (13 pages). 68 of the 69 docs with any non-`line` row are scanned (provenance.md: scanned unassigned 85/1,853 = 4.6% vs digital 1/3,077).
- Multipage docs: train 124 (112 two-page, 12 three-page), all invoices; waybills are single-page. Rows of a multipage doc keep a non-decreasing page along gold row order in 151 of 152 multipage docs (on assigned rows).
- Applying "an unassigned row takes the page of its neighbours when the previous and next assigned rows are on the same page" resolves 38 of 50 train and 15 of 16 dev multipage rows; **12 train rows (5 docs) and 1 dev row (1 doc) remain ambiguous**. This relies on the monotone-order observation, which is itself derived from the same alignment: UNVERIFIED until the label-audit viewer checks those ~6 docs and a sample of the interpolated rows against the images.
- `reports/token_budget.md` placed the 70 unassigned train rows on the **last page** (and 229 `page_fuzzy` rows by part-number OCR search). That is right for single-page docs (20 rows) and a guess for the 50 multipage ones; the token statistics are insensitive to it (+/- a few rows per page) but training labels are not.

### Open questions (each needs an owner decision before the builder is written)

1. **Row-to-page policy.** Proposed: `line` and `page_fuzzy` rows use their assigned page; unassigned rows in single-page docs go to page 0; unassigned multipage rows use neighbour interpolation; docs with a remaining ambiguous row (5 train, 1 dev) are **excluded** (or trained header-only: supervise `doc_type`, `header`, `page_kind` and mask the `line_items` tokens, which keeps their layouts in training; `page_schema` orders keys doc_type, header, line_items, page_kind, so masking the middle block is well defined). The spec's "whole-document sample if it fits the token budget" is not recommended: a 2-3-page sample is ~5.3k-7.9k tokens (2-3 x 2,100 input + 2-3 x 531 target), needs a different output schema than the per-page inference, and is not memory-safe on T4. Exclusions are a logged artifact (doc id, reason, pages), aggregated counts only in this report.
2. **Exclusion bias.** The exclusions and 68/69 non-line docs concentrate in scanned multipage docs of a few suppliers (the groups with most unassigned rows are `inv_g06`, `inv_g05`, `inv_g07`, `inv_g11`, by row count), so training under-represents the layouts the model finds hardest. Quantify per supplier before accepting; the header-only option in 1 is the mitigation.
3. **Which docs does each fold train on?** Fold training docs include dev docs; confirm the builder takes `splits/folds.json` as the only source and that the builder emits per-fold manifests (doc ids) hashed into the run id.
4. **Keyed per-page targets.** Use `shipdoc.targets.gold_page_payloads(gold, n_pages, row_pages)` + `render_target` exactly (identity header on page 1, totals on the last page, union header with nulls, `", "` / `": "` separators, gold row order within a page) so the training text equals what the xgrammar decoder emits; add a test that every training target round-trips `parse_page_json` and `schema_errors`. Confirm the output format (keyed vs compact) is the A/B decision before building.
5. **Whole-doc fallback for multipage docs?** Per-page training is consistent with per-page inference and the merge rules of Phase 3; recommend no whole-doc samples at all (question 1).
6. **Prompt.** The ~840-token text prompt (prompt v2 rules) is 40% of every sequence. A fine-tuned model may need a one-line prompt; shortening it by ~800 tokens would cut roughly 25-30% of training FLOPs (hand arithmetic from the script's per-token terms; vision cost is unchanged) but changes the inference prompt of the fine-tuned arm, which must then be the one used for OOF inference. Decide explicitly; the estimates assume the full v2 prompt (conservative).
7. **Label masking.** Build `input_ids` as `processor.apply_chat_template(user[image,text], add_generation_prompt=True, enable_thinking=False)` (ends with `<|im_start|>assistant\n<think>\n\n</think>\n\n`, per the pinned `chat_template.jinja`, last 8 lines) followed by the tokenized target and `<|im_end|>`; set `labels=-100` on everything before the target; supervise the target tokens and `<|im_end|>` (the stop id the backend uses). Test: the prefix tokens equal the inference-time `apply_chat_template` output byte for byte, and the count of supervised tokens equals `reports/token_budget.md`'s per-page count (it already includes the +1 stop token). The template also renders a final assistant message as `<think>\n\n</think>\n\n` + content (`chat_template.jinja` lines 99-102), so the two constructions should agree; assert it.
8. **Image budget.** Use the identical processor path as `HfBackend.load` (set `image_processor.size["longest_edge"] = max_pixels = 1,310,720`, keep `shortest_edge`), so every 1240x1754 page yields exactly 1,260 visual tokens; assert `n_visual_tokens == 1260` for every sample (augmentation must not change the canvas: `scan_degrade` returns the input size, rotation keeps the canvas). Pass `mm_token_type_ids` and `image_grid_thw` to the model.
9. **Augmentation.** `shipdoc.augment.scan_degrade` (fitted params in `configs/augment_scan.yaml`; mean KS to real scans 0.667 -> 0.199, `reports/augmentation.md`) on digital pages only, deterministic per (doc, epoch, seed 42); occlusion via `make_occluded_sample(...)` whose returned gold is already `null_gold`-ed, so the page target must be built from the **nulled gold**, never the original. Questions: occlusion rate (gold header nulls are 35/192 redactions, spec recon section 3) and whether it is applied per epoch or pre-rendered; keep the recipes disjoint from the synthetic-redaction **dev** eval recipes (`splits/synthetic_redaction_dev.json`), and never occlude a field whose locator level is `fuzzy` (the existing `resolve_target` already returns None for it).
10. **Exclusion logging and determinism.** Log (doc id, page, reason) for every exclusion; seeds, manifest hashes, tokenizer revision and processor settings in the run config; same `seed_everything` as the backend.
11. **Dependencies.** `peft` (inspected: 0.21.2) is in the locked `train` dependency group (added with GG's approval after this plan was written). `trl`/`datasets` are not needed (a ~100-line custom loop gives the chunked target-only loss; HF `Trainer` would materialize full logits).

### Risks

| Risk | Consequence | Mitigation / owner |
|---|---|---|
| Throughput is a model (66 TFLOP/page / assumed 5-10 TFLOP/s on T4, x1.2-1.9 on L4) | stage hours off by up to ~2x either way | measured smoke replaces it; rerun `finetune_estimate.py` with the measured s/page |
| Gated-delta PyTorch fallback (fp32, Python chunk loop) dominates a layer's time or memory in backward | slower than the FLOP model; peak VRAM above the table | time forward/backward per layer type in the smoke; test `flash-linear-attention` only behind a timing + equality check |
| fp16 NaN/inf on T4 (vendor-flagged for qwen3_5) | T4 fallback unusable | bf16 on L4; NaN gate in the smoke |
| L4 unavailable / CU balance unknown | route falls back to T4 | fallback above; check Runtime -> Manage sessions before each stage |
| Colab 12 h cap | a long fold killed | checkpoint to Drive every N steps; resumable (spec section 3) |
| OOM mid-fold on the longest pages (p99 ~3,290 tokens) | lost hours | chunked loss, `PYTORCH_ALLOC_CONF=expandable_segments:True`, sort/preflight the longest page first in the smoke |
| Single training seed per fold | G4 judges one run per fold; run-to-run variance unobserved | optional second seed on fold 0 (~1.4 h L4) if the G4 margin is thin; report as such |
| OOF inference cost (8 T4 h) understated if the fine-tuned model rambles to `max_new_tokens` | +50% inference time | `max_new_tokens` 1536 stays; report truncation rate |
| Adapter merge dropping tensors | silent quality change | merge in memory only; do not `save_pretrained` a merged model for the report path. Third-party report (2U1/Qwen-VL-Series-Finetune PR #259 `456a207d`) that Qwen3.5 merge paths drop the `mtp.*` tensors and downcast fp32 tensors (`A_log`, norm) to fp16; the spike already loads in fp16, so only `mtp.*` (unused for inference) is lost |
| Spec text says "QLoRA fine-tune" (`train.py` docstring, section 4/5) | plan deviates (bf16/fp16 LoRA primary, NF4 fallback) | record in PLAN.md / ADR |
| Post-processing parameters leak across folds | G4 comparison biased | re-fit per fold (V1c step 2) |

---

## Smoke measurement (L4 bf16)

**Source:** GG's smoke run `ft_smoke_b10d810_bf16` on a Colab L4, 2026-10-02 (`$SHIPDOC_RUNS_DIR\ft_smoke\`: `smoke_status.json`, `metrics.jsonl`, `sessions.json`; train.py pinned `b10d810`). Everything below that is derived from it is an **ESTIMATE** (one 20-step run, 40 pages), computed by `scripts/finetune_estimate.py` (the `measured` scenario; tests `tests/test_finetune_estimate.py`).

**Measured.** 20 optimizer steps, each of 2 pages (smoke accumulation 2, micro-batch 1; `n_micro` 2 in every line), over 40 of the 80 smoke pages, all in epoch 0. Mean **17.29 s/step** over the 20 steps of the second session (17.27 over all 29 file lines) = **8.65 s per training page**. Peak VRAM **11.28 GiB** (the longest-page preflight is at most that: the check's printed peak is the max of both), against 20 GiB budget / ~22-24 GiB card. Mean loss of the first 5 steps 0.0551 (second session; 0.0557 over the file's first five lines), last 5 steps 0.0056, min 0.0025; 0 scaler skips (bf16: no scaler); 200 modules / 30,474,240 trainable parameters as expected. Wall-clock of the surviving session 424.6 s; 345.8 s of it inside the 20 steps, so ~79 s per session went to model load, preflight and two checkpoint writes (not separable), consistent with the 120 s `MODEL_LOAD_S` assumption. The 40 smoke pages average 520 supervised target tokens against 522 over all 536 train pages (`reports/supervised_tokens.md`), so the page mix is representative on target length.

**Against the assumption.** The FLOP model (66.0 TFLOP/page) predicted 10.95 / 5.48 / 3.46 s/page on the L4 (low / central / high). The measurement, 8.65 s/page, is 58% slower than central and between central and low; it implies **7.63 effective TFLOP/s** (the T4 central assumption was 7.5). The L4 gain over the T4 may therefore be smaller than the plan assumed: the plan's T4 central (8.76 s/page) came from the same unmeasured factor, and no T4 smoke exists, so the L4-vs-T4 training comparison above is **not** settled by this run.

**What a step is.** Smoke: 2 pages / step. Folds and final: `grad_accum` 8 = 8 pages / step, i.e. ~69 s/step (8 x 8.65 s) before evaluation and checkpoint time. Checkpoint writes (every 10 steps, adapter + optimizer ~0.4 GB) are not in the 17.3 s and are unmeasured on a long run.

**Held-out evaluation inside the runs.** `eval_pages` = 24 fixed held-out pages (N), forward-only: 0.393 of a training page's FLOPs by the model = ~3.4 s/page = **~82 s per evaluation**, 13 evaluations per fold (every 10 steps + each of the 2 epoch ends) = ~0.29 h per fold, 15 for the final stage (0.34 h). Time per page is a FLOP-model scaling of the measured value, not measured.

| Stage (2 epochs, batch 8; ESTIMATE; L4 bf16 measured 8.65 s/page, load 120 s, 3.00 CU/h third-party UNVERIFIED) | visits | steps | evals | train h | eval h | hours | CU |
|---|---:|---:|---:|---:|---:|---:|---:|
| smoke (measured stage shape) | 40 | 20 | 0 | 0.10 | 0.00 | 0.13 | 0.4 |
| fold0 (441 pages) | 882 | 112 | 13 | 2.12 | 0.29 | 2.45 | 7.3 |
| fold1 (454 pages) | 908 | 114 | 13 | 2.18 | 0.29 | 2.51 | 7.5 |
| fold2 (447 pages) | 894 | 112 | 13 | 2.15 | 0.29 | 2.48 | 7.4 |
| all 3 folds | 2684 | 338 | 39 | 6.45 | 0.88 | 7.43 | 22.3 |
| final (536 pages) | 1072 | 134 | 15 | 2.58 | 0.34 | 2.95 | 8.8 |

For comparison the pre-measurement L4 central estimate above was 1.37 h (4.1 CU) per fold and 1.66 h (5.0 CU) for the final stage: **the measured projection is ~1.8x higher for a fold (incl. evaluation) and ~1.8x for the final stage**. Training part of Phase 4 on this route (smoke + 3 folds + final): 10.51 h, 31.5 CU, against 5.93 h / 17.8 CU before. Fold 0 alone: 2.45 h, 7.3 CU. Each stage fits one Colab session (cap 12 h).

OOF inference on T4 fp16, **unbatched 36.0 s/page** (dev100 keyed trace at 2abf481). The batched-decoding bench (Step Y) is not available yet, so the 2x / 3x rows are **HYPOTHETICAL speedups**, not measurements. CU at 1.19 / 1.58 per T4 hour (third-party); all ESTIMATE.

| OOF set | pages | batch speedup (hypothetical except 1x) | T4 hours | CU @1.19 | CU @1.58 |
|---|---:|---:|---:|---:|---:|
| fold0 held-out | 230 | 1x (today) | 2.30 | 2.7 | 3.6 |
| fold0 held-out | 230 | 2x | 1.15 | 1.4 | 1.8 |
| fold0 held-out | 230 | 3x | 0.77 | 0.9 | 1.2 |
| all folds | 671 | 1x (today) | 6.71 | 8.0 | 10.6 |
| all folds | 671 | 2x | 3.35 | 4.0 | 5.3 |
| all folds | 671 | 3x | 2.24 | 2.7 | 3.5 |

**The "29 / 20" optimizer steps.** `metrics.jsonl` held steps 1..9 and then 1..20 (29 lines). `sessions.json` shows session 1 with no end time and no exit code (killed or disconnected, not finished) and session 2 with exit code 0 and 424.6 s. Session 1 died after step 9 and before the first checkpoint (step 10), so there was no checkpoint to resume from; session 2 correctly restarted at step 0 and ran exactly 20 steps (its `total` was 20, not recomputed from the 40-step epoch), but the trainer only rewrote `metrics.jsonl` when a checkpoint was restored, so session 2 **appended** to session 1's lines. The training state, the smoke checks (they read the in-memory history of 20) and the loss means were right; the banner counted file lines. Evidence: step ids 1..9 occur twice; step 1 has the identical loss to 17 digits in both (0.08626615256071091, same data, same seed, same weights before any update) while its grad norm differs in the 4th digit (0.84862 vs 0.84882), so the backward pass is not bit-deterministic on the GPU; steps 2-9 then differ by 0.5% to 9.4% in loss (not a different data order: the sequence of large and small losses repeats step for step). What the data cannot distinguish: why session 1 died (Colab disconnect, manual stop, OOM would have left a status file), exactly when between step 9 and 10, and whether nondeterministic kernels or something else makes the replayed steps differ (a CUDA nondeterminism test was not run). Fixed in `src/shipdoc/train.py` (reset in both resume cases, idempotent step-keyed writer, new `metrics_file` smoke check) with kill/resume tests, see commit `150bc3b`.

## What I could NOT verify

- Any training speed, peak training VRAM, or fp16/bf16 stability, beyond the one 20-step L4 bf16 smoke measured by GG (section above); T4 fp16 training is still unmeasured. All other stage times are the model above.
- That the regex yields exactly 200 modules on the *instantiated* `Qwen3_5ForConditionalGeneration`: validated against the checkpoint key names and the modeling source, not by building the model (the throwaway venv has `transformers==5.18.0` and `peft==0.21.2` installed with `--no-deps`, no torch). The smoke tripwire (200 modules, 30,474,240 params) closes this.
- That peft 0.21.2 is compatible with transformers 5.18.0 at runtime (only inspected statically; peft declares an unpinned `transformers` requirement).
- Whether `flash-linear-attention` works or helps on T4 (sm75) or L4.
- Current Colab CU rates: no first-party source reachable (pricing page 403, FAQ has none). 1.19 is from `configs/spike_speed.json` (mccormickml.com, URL not recorded in the repo, not re-fetched); 1.58 and 3.00 are one third-party table dated 2024-10-01.
- The T4 usable-memory figure (~14.5 GiB) and whether Colab's L4 is the 24 GB part (varlog shows "L4 22 GB").
- `reports/token_budget.md` numbers (target tokens) were taken as given; only the train page count (536) and the row total (4,006 = 3,707 + 229 + 70) were re-derived.
- The unsloth statements about qwen3_5 fp16 are vendor PR text (no measurements shown).
- The neighbour-interpolation result depends on the alignment it is derived from (circular until image-audited).

## Sources (all fetched 2026-10-02 unless noted)

- HF model repo `Qwen/Qwen3.5-4B` at `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`: `https://huggingface.co/Qwen/Qwen3.5-4B/resolve/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/{config.json, model.safetensors.index.json, chat_template.jinja, preprocessor_config.json, tokenizer_config.json, README.md}` and the two `model.safetensors-0000N-of-00002.safetensors` shards (header bytes only). File sha256 prefixes: config.json `ddc63e1c717a`, index `cf3f798ee02b`, chat_template `a4aee8afcf2e`. The repo has no `generation_config.json` (404) and no `processor_config.json` (404).
- transformers `5.18.0` (PyPI wheel, installed in `$SHIPDOC_TMP_DIR\venv_peft` with `--no-deps`): `models/qwen3_5/modeling_qwen3_5.py` (sha256 prefix `d0c8561d91e3`; lines cited above), `integrations/hub_kernels.py` (`a58d58697a46`, lines 984-1044, `_PACKAGE_TO_DISTRIBUTION` line 80), `modeling_utils.py` (`3b615a8d21cb`, line 3139), `modeling_layers.py` (`ee84c7229582`, lines 53-110).
- peft `0.21.2` (PyPI, same venv): `tuners/tuners_utils.py` (`e7f99b2ac576`, lines 2337-2376, 2705-2730), `utils/other.py` (`6b87957915ba`, lines 215-240, 1516-1521), `mapping_func.py:110`, `peft_model.py:722-740`, `tuners/lora/bnb.py:313`.
- NVIDIA T4 datasheet `https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf` (sha256 prefix `57f6b54378a8`): "Mixed-Precision (FP16/FP32) 65 TFLOPS", "16 GB GDDR6, 300 GB/sec". NVIDIA L4 page `https://www.nvidia.com/en-us/data-center/l4/` (`983001236473`): "FP16 Tensor Core 242 teraFLOPS*, BFLOAT16 242 teraFLOPS*, GPU memory 24GB, 300GB/s, 72W ... * Shown with sparsity. Specifications are one-half lower without sparsity"; L4 product brief `https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/l4/PB-11316-001_v01.pdf` (memory 24 GB GDDR6, 300 GB/s).
- Colab compute units: `https://varlog.info/colab-credit-consumption-comparison-2024/` (`b964f280bd05`, "measurement was made on October 1, 2024 with the Colab Pro tariff": T4 1.58, T4 High-RAM 1.67, L4 3.00, A100 10.59 compute units/h; L4 "GPU RAM 22 GB"). Colab FAQ `https://research.google.com/colaboratory/faq.html` (`a7e35a0f2a1a`): premium GPUs "subject to availability" on paid plans; notebooks run at most 12 h (Pro+ up to 24 h with sufficient units). `https://colab.research.google.com/pricing` returned HTTP 403.
- Flash-linear-attention README `https://raw.githubusercontent.com/fla-org/flash-linear-attention/main/README.md` (branch head, commit not recorded): "verified on NVIDIA, AMD, and Intel hardware", Triton-based.
- unsloth PR #11533 `https://github.com/unslothai/unsloth/pull/11533` (open; head `05236b959bc51fc27357a2bfd5c4b192ddfaaf04`); unsloth PR #5880 `https://github.com/unslothai/unsloth/pull/5880` (merged 2026-06-29, `de3c745fab08e14a1cd7a825c434f048a53ec5ef`); 2U1/Qwen-VL-Series-Finetune PR #259 `https://github.com/2U1/Qwen-VL-Series-Finetune/pull/259` (merged, `456a207dfbf8e3981f9ccff1148a1c4c8b098a50`). PR bodies read via the GitHub REST API.
- Repo artifacts: `spec.md` sections 2, 3, 5 (Phase 4), `reports/token_budget.md`, `reports/augmentation.md`, `reports/provenance.md`, `reports/spike_diagnosis.md`, `configs/spike_qwen35_4b_img_only.yaml`, `configs/spike_speed.json`, `splits/folds.json`, `meta/{train,dev}.json`, `src/shipdoc/{extract,targets,augment}.py`; spike traces `$SHIPDOC_RUNS_DIR\spike_download2\x\{spike40_qwen35_4b_img_only_compact,dev100_qwen35_4b_img_only,dev100_qwen35_4b_img_ocr,ab_keyed_qwen35_4b_img_ocr}_2abf481\trace.jsonl` (`git_commit` field 2abf481); `$SHIPDOC_RUNS_DIR\provenance\locations.jsonl`.
