# Post-processing v1 replay check (R1 + R2 + R3)

**Provenance.** Run `zeroshot500_qwen35_4b_img_only_keyed_42b812b`, `uv run python scripts/replay_v1_check.py` (repo state `8f5ba7f`; paired bootstrap 2000 doc-level resamples, seed 42; unmodified scorer via `shipdoc.eval`). Production path: `shipdoc.postrules` + the coerce / repair stage of `predict assemble`; not `scripts/rule_gate.py`. **All numbers UNVERIFIED** until a verifier recomputes them. Aggregates and counts only.

Docs: 500. Baseline OVERALL (predictions.json, equals metrics.json to 4 decimals): 76.58. The production path with all rules OFF reproduces predictions.json exactly (asserted). Frozen shapes sha256 `920088327fcfe95dada188292bb4966f2b08a928624e34e34553ea633a7afd99`.

(a) HONEST = R3 shapes of fold k learned from the other folds' gold (supplier-held-out). (b) SHIPPING = frozen `meta/slot_shapes.json` (learned from these 500 docs: in-sample, optimistic).

## (a) HONEST

| slice | docs | base OVERALL | v1 OVERALL | d OVERALL (pts) | 95% CI (pts) | false-fill before | false-fill after |
|---|---|---|---|---|---|---|---|
| all | 500 | 76.58 | 83.60 | +7.02 | [+5.42, +8.90] | 0.0000 | 0.0000 |
| invoice | 400 | 76.12 | 81.84 | +5.73 | [+4.07, +7.56] | 0.0000 | 0.0000 |
| waybill | 100 | 88.36 | 98.27 | +9.91 | [+7.07, +13.11] | 0.0000 | 0.0000 |
| scanned | 193 | 71.05 | 80.09 | +9.05 | [+5.95, +12.37] | 0.0000 | 0.0000 |
| digital | 307 | 79.99 | 85.75 | +5.77 | [+3.90, +7.77] | 0.0000 | 0.0000 |
| train | 400 | 76.59 | 83.98 | +7.39 | [+5.47, +9.26] | 0.0000 | 0.0000 |
| dev | 100 | 76.54 | 81.97 | +5.43 | [+2.74, +8.88] | 0.0000 | 0.0000 |

Per rule from the production path (cell diffs vs the baseline):

| rule | fixed | broken | neutral | fixed by split | gate fixed (cited) | single-rule d OVERALL (pts) |
|---|---|---|---|---|---|---|
| R1 | 12 | 0 | 0 | dev 4, train 8 | train 8 + dev 4 | +0.40 |
| R2 | 32 | 0 | 0 | dev 12, train 20 | train 20 + dev 12 | +1.07 |
| R3 | 502 | 0 | 11 | dev 52, train 450 | all 502 | +5.40 |

Sum of single-rule deltas +6.86 pts vs combined +7.02 pts (difference +0.16 pts). R1+R2 together (both only touch waybill headers): +1.63 pts vs R1 + R2 alone +1.47 pts: the excess is within the waybill pair (consistent with a doc score that is not linear in the cell count; mechanism not isolated); R3 (invoices) adds on top without interaction (combined minus R1+R2 = +5.40 vs R3 alone +5.40).

Rule touches (docs): {'R1': 12, 'R2': 26, 'R3': 46}; eligible docs: {'R1': 100, 'R2': 100, 'R3': 400}; skipped: none.

## (b) SHIPPING (in-sample)

| slice | docs | base OVERALL | v1 OVERALL | d OVERALL (pts) | 95% CI (pts) | false-fill before | false-fill after |
|---|---|---|---|---|---|---|---|
| all | 500 | 76.58 | 83.60 | +7.02 | [+5.42, +8.90] | 0.0000 | 0.0000 |
| invoice | 400 | 76.12 | 81.84 | +5.73 | [+4.07, +7.56] | 0.0000 | 0.0000 |
| waybill | 100 | 88.36 | 98.27 | +9.91 | [+7.07, +13.11] | 0.0000 | 0.0000 |
| scanned | 193 | 71.05 | 80.09 | +9.05 | [+5.95, +12.37] | 0.0000 | 0.0000 |
| digital | 307 | 79.99 | 85.75 | +5.77 | [+3.90, +7.77] | 0.0000 | 0.0000 |
| train | 400 | 76.59 | 83.98 | +7.39 | [+5.47, +9.26] | 0.0000 | 0.0000 |
| dev | 100 | 76.54 | 81.97 | +5.43 | [+2.74, +8.88] | 0.0000 | 0.0000 |

Per rule from the production path (cell diffs vs the baseline):

| rule | fixed | broken | neutral | fixed by split | gate fixed (cited) | single-rule d OVERALL (pts) |
|---|---|---|---|---|---|---|
| R1 | 12 | 0 | 0 | dev 4, train 8 | train 8 + dev 4 | +0.40 |
| R2 | 32 | 0 | 0 | dev 12, train 20 | train 20 + dev 12 | +1.07 |
| R3 | 502 | 0 | 11 | dev 52, train 450 | all 502 | +5.40 |

Sum of single-rule deltas +6.86 pts vs combined +7.02 pts (difference +0.16 pts). R1+R2 together (both only touch waybill headers): +1.63 pts vs R1 + R2 alone +1.47 pts: the excess is within the waybill pair (consistent with a doc score that is not linear in the cell count; mechanism not isolated); R3 (invoices) adds on top without interaction (combined minus R1+R2 = +5.40 vs R3 alone +5.40).

Rule touches (docs): {'R1': 12, 'R2': 26, 'R3': 46}; eligible docs: {'R1': 100, 'R2': 100, 'R3': 400}; skipped: none.
