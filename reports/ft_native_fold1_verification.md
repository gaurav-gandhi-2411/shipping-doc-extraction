# Fold-1 native fine-tune: local verification

Run `ft_native_fold1_4c17aa3_bf16`, read in place at `$SHIPDOC_RUNS_DIR\ft_native_fold1\ft_native_fold1_4c17aa3_bf16\` (already a folder, no extraction needed). Scripts: `$SHIPDOC_TMP_DIR\verify_native_fold1.py` and `$SHIPDOC_TMP_DIR\verify_native_oof05n_f1.py` (adapted copies of the fold-0 scripts, run with `uv run --frozen python`). Aggregates only; no document values or ids. All numbers below are VERIFIED (computed by those scripts) unless marked UNVERIFIED.

## Result: all 7 checks PASS. Two banner wording points need correcting (see Inconsistencies).

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | Manifest | PASS | code_sha = 4c17aa3c33c09f0cda7bb1f625947a1143a8cb28; max_pixels 2196480; precision bf16; stage fold1, fold 1; steps = total_steps = 114; n_train_pages 454. Equal to fold 0: train_config_signature e68d9961faade76d, lora (r16, alpha32, 200 modules, 30,474,240 trainable), inference_keys, base revision, peft README and adapter_config sha. Differ (expected): adapter_sha256, peft safetensors sha, fold, manifest_hash (4d32a97038e1c578 vs f4bcc661a36211f7), doc-id lists and counts, n_train_pages (454 vs 441), steps (114 vs 112), signature (last part), stage. |
| 2 | Doc sets | PASS | all 500; fold-1 val = 165 (equals doc_fold==1); train 335 = all minus val; heldout == val; union == all 500; overlap 0; no duplicates. Supplier groups: train 19, held-out 9, on both sides 0; the 9 held-out groups equal folds.json val_groups. Manifest ids equal independently computed sets. |
| 3 | Hashes / weights | PASS | adapter.pt, safetensors, adapter_config.json, README.md sha256 all equal manifest; peft folder holds exactly the 3 manifest files. adapter.pt vs peft: 400 tensors each, key sets equal after name normalisation, 400/400 bit-equal, fp32, all finite, 30,474,240 params. Fold-1 adapter sha differs from fold 0. |
| 4 | oof.verify_adapter_manifest (fold=1, native cfg, native ZS run) | PASS | 27 PASS rows, 0 FAIL, warnings [] (ok: True); assert_adapter_resolution OK; manifest_hash matches ids; ZS config hash e4b84ec2809625d5. `resolve_batch_size(zs, None)` = 4 (source zero_shot_run); an explicit 8 is flagged differs_from_zero_shot. |
| 5 | Metrics | PASS | see below |
| 6 | Resume | PASS (see caveat) | see below |
| 7 | vs fold 0 | see table | |

## 5. Metrics detail
- metrics.jsonl: 114 rows, steps exactly 1..114, in order, 0 duplicates, 0 gaps. 908 micro-batches (n_micro 8 except 6 on the last step of each epoch's short tail). 0 scaler skips, 0 non-finite losses.
- metrics_discarded.jsonl EXISTS: 8 rows, steps 1..8, all `discarded_at_step: 0`, 900.7 s of step time. Fold 0 had no such file.
- Timing: 12,413.8 s summed over kept rows = 108.89 s/step (banner 108.9: match), median 108.6; peak VRAM 12.0735 GiB (banner 12.07: match).
- Sessions: session 1 started 2026-10-03T14:59:54Z, `seconds: null`, `exit_code: null` (never closed = the disconnect); session 2 started 15:17:12Z (1,038 s later), 13,199.7 s, exit 0 = 3.667 h (banner 3.67: match). 3.667 h is session 2 only; it includes model load, 13 held-out evals and checkpoints (785.9 s over the step sum 12,413.8 s). The disconnected session's time is NOT in the 3.67 h; its 8 steps cost about 900 s of step time plus load, so total wall/compute spent was about 3.67 h + roughly 0.25-0.29 h (the 1,038 s gap between starts bounds it; exact end of session 1 is UNVERIFIED, not recorded).
- train_status.json: state complete, final = /content/drive/MyDrive/shipdoc-extract/runs/ft_native_fold1_4c17aa3_bf16/final, total_steps 114, n_train_pages 454, stage fold1, signature equals manifest.
- Held-out loss (24 pages, every 10 steps and at epoch ends): step 10 0.0038, 20 0.0016, 30 0.0012, 40 0.0030, 50 0.0012, 57 (epoch 1) 0.00091, 60 0.00086, 70 0.00077, 80 0.00078, 90 0.00076, 100 0.00072, 110 0.00070, 114 (epoch 2) 0.00069. Final eval equals the last row's eval_loss (step 114): yes.
- Step-40 blip is in the HELD-OUT eval only. Train loss around step 40: step 40 = 0.00102; steps 35-45 range 0.00049-0.00282 (the 0.00282 is step 35, 0.0017 at step 43); mean steps 31-39 = 0.00153, mean 41-49 = 0.00119. No train-loss spike at 40, grad_norm steps 35-45 range 0.016-0.067 (no jump), lr smooth. Largest train loss in steps 21-60 is step 22 (0.00649).

## 6. Resume
- Finding: the "resume" was a RESTART FROM STEP 0, not a resume from a checkpoint. Evidence: all 8 discarded rows carry `discarded_at_step: 0` (no checkpoint existed; the first checkpoint is due at step 10 and session 1 died after step 8). The trainer reset metrics.jsonl, kept the 8 rows in metrics_discarded.jsonl, and recomputed all 114 steps in session 2. Steps recomputed after the "resume": 8 (steps 1..8 are the ones recomputed; no checkpoint-restored step counter, optimizer, scheduler, scaler or RNG state was exercised on the live path).
- Consequently the model trained in session 2 alone, uninterrupted from step 0 to 114. Discontinuity risk is low, but note the rerun is not bit-reproducible: for steps 1..8, kept vs discarded loss differ by 0 (step 1), then 5e-5 to 4.8e-4 (steps 2..8); n_micro and lr identical. Normal GPU/bf16 nondeterminism or a different GPU; cause UNVERIFIED.
- lr: expected per-step lr recomputed from the code's `lr_scale` (peak 2e-4, spe 57, total 114, warmup round(0.1*114)=11, cosine to 0): max abs deviation 1.0e-20 (max relative 3.4e-16, at step 89). The schedule is followed exactly at every step.
- Checkpoints left: step_000110 and step_000114 (LATEST = step_000114). The step-114 trainer_state is consistent: step 114, epoch 2, sampler_pos 0, history 114 rows with steps 1..114 and losses equal to metrics.jsonl, scheduler last_epoch 114.
- Bit-identical resume from a checkpoint is not provable from logs and was not exercised here.

## 7. Fold 0 vs fold 1 (native, bf16)
| | fold 0 | fold 1 |
|---|---|---|
| train docs / pages | 329 / 441 | 335 / 454 |
| held-out docs | 171 | 165 |
| steps | 112 | 114 |
| s/step (mean of step rows) | 109.49 | 108.89 |
| peak VRAM GiB | 12.0735 | 12.0735 |
| first held-out eval (step 10) | 0.00340 | 0.00382 |
| epoch 1 / final held-out loss | 0.00255 / 0.00175 | 0.00091 / 0.00069 |
| hours (step sum / sessions) | 3.406 / 3.616 (1 session) | 3.448 / 3.667 (session 2 only; plus an unclosed session 1) |
| metrics_discarded.jsonl | none | 8 rows (restart at step 0) |

Held-out loss is on each fold's own 24 eval pages, so the fold-0 and fold-1 numbers are not directly comparable.

## Inconsistencies with the banner
1. "held-out loss 0.0038 -> 0.0007": 0.0038 is the step-10 eval, there is no step-0 eval, so it is the first recorded value, not an untrained baseline. Epoch 1 0.0009 and epoch 2 0.0007 match (0.00091, 0.00069).
2. "one blip at step 40 = 0.0030": matches (0.00297), eval loss only; not a train-loss event.
3. "3.67 h plus 1 disconnected session (resumed)": numbers match, but it was a restart from step 0 (8 steps lost), not a checkpoint resume.
No other deviations: 114/114 steps, 454 pages, 108.9 s/step, 12.07 GiB all match.

## Adapter for 05n fold 1
- Local: `$SHIPDOC_RUNS_DIR\ft_native_fold1\ft_native_fold1_4c17aa3_bf16\final` (adapter.pt, peft\, manifest.json).
- Drive (per train_status.json, not checked from here): `/content/drive/MyDrive/shipdoc-extract/runs/ft_native_fold1_4c17aa3_bf16/final`.
- 05n needs: this `final/` folder, fold=1, native config `configs/spike_qwen35_4b_img_only_native.yaml`, pin 4c17aa3c33c09f0cda7bb1f625947a1143a8cb28, the native zero-shot run `zeroshot500_qwen35_4b_img_only_native_4c17aa3` (stored batch size 4; use 4 unless a differing batch is deliberate, 8 is flagged), inference on the 165 fold-1 held-out docs, max_pixels 2196480.

## Open risks
- Session-1 end time and cause of the disconnect are not recorded (UNVERIFIED).
- The replayed steps 2..8 differ slightly from the discarded ones (nondeterminism); not a correctness issue, but the run is not reproducible bit-for-bit.
- Drive copy was not compared to the local copy by hash here beyond the manifest hashes, which match the local files.
