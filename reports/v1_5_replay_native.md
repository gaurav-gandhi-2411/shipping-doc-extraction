# Post-processing v1 replay check (R1 + R2 + R3)

**Provenance.** Run `zeroshot500_qwen35_4b_img_only_native_4c17aa3`, `uv run python scripts/replay_v1_check.py --run-dir $SHIPDOC_TMP_DIR\native_extract\zs\zeroshot500_qwen35_4b_img_only_native_4c17aa3 --out reports/v1_5_replay_native.md --expected-base-overall 0.8386 --gate-fixed R1=train:4,dev:3 --gate-fixed R2=train:27,dev:14 --gate-fixed R3=all:278` (repo state `139186f+dirty`; paired bootstrap 2000 doc-level resamples, seed 42; unmodified scorer via `shipdoc.eval`). Production path: `shipdoc.postrules` + the coerce / repair stage of `predict assemble`; not `scripts/rule_gate.py`. **All numbers UNVERIFIED** until a verifier recomputes them. Aggregates and counts only.

Docs: 500. Baseline OVERALL (predictions.json, equals metrics.json to 4 decimals): 83.86. The production path with all rules OFF reproduces predictions.json exactly (asserted). Frozen shapes sha256 `920088327fcfe95dada188292bb4966f2b08a928624e34e34553ea633a7afd99`.

(a) HONEST = R3 shapes of fold k learned from the other folds' gold (supplier-held-out). (b) SHIPPING = frozen `meta/slot_shapes.json` (learned from these 500 docs: in-sample, optimistic).

## (a) HONEST

| slice | docs | base OVERALL | v1 OVERALL | d OVERALL (pts) | 95% CI (pts) | false-fill before | false-fill after |
|---|---|---|---|---|---|---|---|
| all | 500 | 83.86 | 88.51 | +4.64 | [+3.43, +6.00] | 0.0000 | 0.0000 |
| invoice | 400 | 83.98 | 86.99 | +3.01 | [+1.89, +4.17] | 0.0000 | 0.0000 |
| waybill | 100 | 88.84 | 99.71 | +10.87 | [+7.93, +14.11] | 0.0000 | 0.0000 |
| scanned | 193 | 82.27 | 86.92 | +4.64 | [+2.80, +6.69] | 0.0000 | 0.0000 |
| digital | 307 | 84.86 | 89.48 | +4.62 | [+3.11, +6.38] | 0.0000 | 0.0000 |
| train | 400 | 83.90 | 88.31 | +4.40 | [+3.11, +5.80] | 0.0000 | 0.0000 |
| dev | 100 | 83.72 | 89.31 | +5.59 | [+2.75, +9.26] | 0.0000 | 0.0000 |

Per rule from the production path (cell diffs vs the baseline):

| rule | fixed | broken | neutral | fixed by split | gate fixed (cited) | single-rule d OVERALL (pts) |
|---|---|---|---|---|---|---|
| R1 | 7 | 0 | 0 | dev 3, train 4 | train 4 + dev 3 | +0.23 |
| R2 | 41 | 0 | 0 | dev 14, train 27 | train 27 + dev 14 | +1.44 |
| R3 | 278 | 0 | 9 | dev 50, train 228 | all 278 | +2.86 |

Sum of single-rule deltas +4.52 pts vs combined +4.64 pts (difference +0.12 pts). R1+R2 together (both only touch waybill headers): +1.79 pts vs R1 + R2 alone +1.67 pts: the excess is within the waybill pair (consistent with a doc score that is not linear in the cell count; mechanism not isolated); R3 (invoices) adds on top without interaction (combined minus R1+R2 = +2.86 vs R3 alone +2.86).

Rule touches (docs): {'R1': 7, 'R2': 29, 'R3': 25}; eligible docs: {'R1': 100, 'R2': 100, 'R3': 400}; skipped: none.

## (b) SHIPPING (in-sample)

| slice | docs | base OVERALL | v1 OVERALL | d OVERALL (pts) | 95% CI (pts) | false-fill before | false-fill after |
|---|---|---|---|---|---|---|---|
| all | 500 | 83.86 | 88.51 | +4.64 | [+3.43, +6.00] | 0.0000 | 0.0000 |
| invoice | 400 | 83.98 | 86.99 | +3.01 | [+1.89, +4.17] | 0.0000 | 0.0000 |
| waybill | 100 | 88.84 | 99.71 | +10.87 | [+7.93, +14.11] | 0.0000 | 0.0000 |
| scanned | 193 | 82.27 | 86.92 | +4.64 | [+2.80, +6.69] | 0.0000 | 0.0000 |
| digital | 307 | 84.86 | 89.48 | +4.62 | [+3.11, +6.38] | 0.0000 | 0.0000 |
| train | 400 | 83.90 | 88.31 | +4.40 | [+3.11, +5.80] | 0.0000 | 0.0000 |
| dev | 100 | 83.72 | 89.31 | +5.59 | [+2.75, +9.26] | 0.0000 | 0.0000 |

Per rule from the production path (cell diffs vs the baseline):

| rule | fixed | broken | neutral | fixed by split | gate fixed (cited) | single-rule d OVERALL (pts) |
|---|---|---|---|---|---|---|
| R1 | 7 | 0 | 0 | dev 3, train 4 | train 4 + dev 3 | +0.23 |
| R2 | 41 | 0 | 0 | dev 14, train 27 | train 27 + dev 14 | +1.44 |
| R3 | 278 | 0 | 9 | dev 50, train 228 | all 278 | +2.86 |

Sum of single-rule deltas +4.52 pts vs combined +4.64 pts (difference +0.12 pts). R1+R2 together (both only touch waybill headers): +1.79 pts vs R1 + R2 alone +1.67 pts: the excess is within the waybill pair (consistent with a doc score that is not linear in the cell count; mechanism not isolated); R3 (invoices) adds on top without interaction (combined minus R1+R2 = +2.86 vs R3 alone +2.86).

Rule touches (docs): {'R1': 7, 'R2': 29, 'R3': 25}; eligible docs: {'R1': 100, 'R2': 100, 'R3': 400}; skipped: none.
