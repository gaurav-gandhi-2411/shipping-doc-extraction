# Fold-2 native fine-tune: local verification

Run `ft_native_fold2_4c17aa3_bf16`, read in place at `$SHIPDOC_RUNS_DIR\ft_native_fold2\ft_native_fold2_4c17aa3_bf16\` (a folder, no extraction needed). Scripts: `$SHIPDOC_TMP_DIR\verify_native_fold2.py` (output `$SHIPDOC_TMP_DIR\verify_native_fold2.out`), `$SHIPDOC_TMP_DIR\verify_native_oof05n_f2.py`, `$SHIPDOC_TMP_DIR\native_extract\evalslice_f2.py` (adapted copies of the fold-1 scripts, run with `uv run --frozen python`; originals untouched). Aggregates, counts, hashes and supplier-group ids only; no document values or document ids. Numbers are VERIFIED (computed by those scripts) unless marked.

## Result: all 7 checks PASS. No deviation from the banner.

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | Manifest | PASS | code_sha = 4c17aa3c33c09f0cda7bb1f625947a1143a8cb28; max_pixels 2196480; precision bf16; stage fold2, fold 2; steps = total_steps = 112; n_train_pages 447. Equal to fold 0 AND fold 1: train_config_signature e68d9961faade76d, lora (r16, 200 modules, 30,474,240 trainable), inference_keys, base revision, peft README and adapter_config sha. Differ (expected): adapter_sha256, peft safetensors sha, fold, stage, manifest_hash (fold 2 1727889837a0ae7f; fold 1 4d32a97038e1c578; fold 0 f4bcc661a36211f7), doc-id lists, n_train_docs, n_train_pages, signature (last part); steps/total_steps differ from fold 1 only (112 vs 114). Config therefore equals folds 0/1 except fold-specific fields. |
| 2 | Doc sets | PASS | all 500; fold-2 val = 164 (equals doc_fold==2); train 336 = all minus val; heldout == val; union == all; overlap 0. Supplier groups (meta/supplier_groups.json): train 19, held-out 9, on both sides 0; the 9 held-out groups equal folds.json val_groups. |
| 3 | Hashes / weights | PASS | adapter.pt, safetensors, adapter_config.json, README.md sha256 all equal manifest; peft folder holds exactly the 3 manifest files. adapter.pt vs peft: 400 tensors each, key sets equal after name normalisation, 400/400 bit-equal, fp32, finite, 30,474,240 params. Fold-2 adapter sha differs from folds 0 and 1. |
| 4 | oof.verify_adapter_manifest (fold=2, native cfg, native ZS run `zeroshot500_qwen35_4b_img_only_native_4c17aa3`) | PASS | 27 PASS lines (26 manifest rows + the resolution check; same count as fold 1), 0 FAIL, warnings [] (ok: True); assert_adapter_resolution OK (2196480); manifest_hash matches ids; ZS config hash e4b84ec2809625d5. `resolve_batch_size(zs, None)` = 4 (source zero_shot_run); explicit 8 flagged differs_from_zero_shot. |
| 5 | Metrics | PASS | below |
| 6 | train_status / lr | PASS | below |
| 7 | vs folds 0/1 | table below | |

## 5. Metrics
- metrics.jsonl: 112 rows, steps exactly 1..112 in order, 0 duplicates, 0 gaps. 894 micro-batches (n_micro 8 on 110 steps, 7 on 2 steps = each epoch's short tail). 0 scaler skips, 0 non-finite losses. No metrics_discarded.jsonl (no restart).
- Timing: 12,074.3 s summed over step rows = 107.81 s/step (banner 107.8: match), median 107.83; peak VRAM 12.0101 GiB (banner 12.01: match).
- sessions.json: exactly 1 session, start 2026-10-03T19:03:22Z, 12,894.8 s, exit 0 = 3.582 h (banner 3.58: match). It exceeds the step sum by 820.5 s (model load, 13 held-out evals, checkpoints). Step-sum hours: 3.354.
- train_status.json: state complete, final = /content/drive/MyDrive/shipdoc-extract/runs/ft_native_fold2_4c17aa3_bf16/final, total_steps 112, n_train_pages 447, stage fold2, signature equals manifest.
- Held-out loss (24 pages, every 10 steps and epoch ends): step 10 0.00991, 20 0.00811, 30 0.00700, 40 0.00625 (the minimum), 50 0.00657, 56 (epoch 1) 0.00727, 60 0.00760, 70 0.00757, 80 0.00707, 90 0.00683, 100 0.00680, 110 0.00673, 112 (epoch 2) 0.00675. Final eval equals the last row's eval_loss (step 112): yes. Banner: 0.0099 -> 0.0068, epoch 1 0.0073, epoch 2 0.0068, low 0.0063 at step 40: all match; 0.0099 is the step-10 eval (no step-0 eval exists), so it is the first recorded value, not an untrained baseline (same wording point as fold 1).
- Unlike folds 0/1 the curve is not monotone after step 40: it rises from 0.00625 (step 40) to 0.00760 (step 60), then falls to 0.00675. Epoch-1 end 0.00727 is above the step-40 value. The final value is 0.00675, 7.4% above the step-40 minimum (0.00625).
- Train loss: first 0.1834, last 0.00025, mean of last 10 steps 0.00094 (fold 0 0.00026, fold 1 0.00029). Step 40 train loss 0.00046, steps 35-45 range 0.00040-0.00215; grad_norm steps 35-45 range 0.0074-0.0423 (no jump). Max train loss in steps 21-60 is 0.00664. So the step-40 low is a held-out-eval value, not linked to a train-loss event.

## 6. lr schedule
Expected per-step lr recomputed from the code's `lr_scale` (peak 2e-4, spe 56 = ceil(447/8), total 112, warmup round(0.1*112)=11, cosine to 0): max abs deviation 6.8e-21 (max relative 2.2e-16, at step 87). The schedule is followed exactly. Checkpoints left: step_000110 and step_000112 (LATEST = step_000112); the step-112 trainer_state is consistent (step 112, epoch 2, sampler_pos 0, history 112 rows with steps 1..112 and losses equal to metrics.jsonl, scheduler last_epoch 112).

## 7. Fold 0 / fold 1 / fold 2 (native, bf16)
| | fold 0 | fold 1 | fold 2 |
|---|---|---|---|
| train docs / pages | 329 / 441 | 335 / 454 | 336 / 447 |
| held-out docs / pages | 171 / 230 | 165 / 217 | 164 / 224 |
| steps | 112 | 114 | 112 |
| s/step (mean of step rows) | 109.49 | 108.89 | 107.81 |
| peak VRAM GiB | 12.0735 | 12.0735 | 12.0101 |
| first held-out eval (step 10) | 0.00340 | 0.00382 | 0.00991 |
| epoch 1 / final held-out loss | 0.00255 / 0.00175 | 0.00091 / 0.00069 | 0.00727 / 0.00675 |
| min held-out loss (step) | 0.00175 (112) | 0.00069 (114) | 0.00625 (40) |
| mean train loss all steps / last 10 | 0.00695 / 0.00026 | 0.00633 / 0.00029 | 0.00704 / 0.00094 |
| hours (step sum / sessions) | 3.406 / 3.616 (1) | 3.448 / 3.667 (session 2; plus unclosed session 1) | 3.354 / 3.582 (1) |
| metrics_discarded.jsonl | none | 8 rows (restart at step 0) | none |

(Held-out page counts 230/217/224 and train page counts 441/454/447 from `trainset.load_prepared($SHIPDOC_RUNS_DIR\trainset\prepared_final.json)`; the train counts equal the manifests' n_train_pages, a cross-check of the plans used.)

## Why is fold 2's held-out loss 4-10x the other folds: what the artifacts show
The held-out loss is on a FIXED seeded subset of 24 pages of that fold's held-out pages (`train.eval_subset`: `epoch_order(n, seed 42, 0)[:24]`), so the three folds are scored on different 24 pages. Measurable facts only:
- Fold-2 final 0.00675 vs fold 0 0.00175 (3.9x) and fold 1 0.00069 (9.8x). Its train loss over all steps is the same as the others (0.00704 vs 0.00695 / 0.00633; dominated by the first steps), and its last-10-step train loss is 0.00094 vs 0.00026 / 0.00029 (about 3.3x higher). So the held-out gap is large while the train-side level is not.
- Supplier groups are disjoint across folds (each group sits wholly in one fold, 0 on both sides), so no group of fold 2's held-out set has ANY train-side share in this run: train-side share is 0 for all 9 held-out groups, as for folds 0 and 1. Fold-2 held-out groups (docs, = the group's total in the 500): inv_g07 28, inv_g09 24, inv_g17 23, inv_g11 22, inv_g14 19, inv_g02 16, wb_g08 14, wb_g03 11, wb_g09 7. Held-out is 6 invoice groups (132 docs) and 3 waybill groups (32 docs).
- Reconstruction of the 24 eval pages (replica of the trainer's selection from the same plans and seed; UNVERIFIED against the live run, which does not record the page list; the plans cross-check above supports it): fold 2's slice has 23 docs, by group inv_g02 5, inv_g07 3, inv_g09 3, inv_g11 2, inv_g14 5, inv_g17 5, wb_g08 1 pages; 6 of the 24 are `dev`-split pages (fold 0: 2, fold 1: 3); 17 first pages, 6 second, 1 third page. Fold 1's slice is 8 pages from inv_g16 and 2 from wb_g01 plus 14 others; fold 0's slice has 6 pages each from inv_g04 and inv_g10.
- Nothing in the artifacts identifies which pages or fields carry the loss, so a cause is NOT established. What the data allows: the three slices differ in group composition and in the share of `dev`-split pages (6/24 vs 2/24 and 3/24); that this share drives the loss is a hypothesis, not a finding. The per-page eval losses are not saved. The OOF scoring of the fold-2 adapter (held-out accuracy on all 164 docs) is the proper measure; the 24-page eval loss is informational only (see `train.py` docstring).

## Adapter for 05n fold 2
- Local: `$SHIPDOC_RUNS_DIR\ft_native_fold2\ft_native_fold2_4c17aa3_bf16\final` (adapter.pt, peft\, manifest.json).
- Drive (per train_status.json, not checked from here): `/content/drive/MyDrive/shipdoc-extract/runs/ft_native_fold2_4c17aa3_bf16/final`.
- 05n needs: this `final/` folder, fold=2, native config `configs/spike_qwen35_4b_img_only_native.yaml`, pin 4c17aa3c33c09f0cda7bb1f625947a1143a8cb28, native ZS run `zeroshot500_qwen35_4b_img_only_native_4c17aa3` (stored batch 4; use 4), inference on the 164 fold-2 held-out docs, max_pixels 2196480.

## Open risks
- Cause of the high fold-2 held-out loss unestablished (above); the non-monotone eval curve (low at step 40, rise to step 60) is a fact, its reason is not.
- Eval-page reconstruction UNVERIFIED against the live run.
- Drive copy not hash-compared to the local copy beyond the manifest hashes, which match the local files.
