# EXPLORATORY, NOT A DECISION; fold 0 was already seen when G was pre-registered; within-fold cross-fit; fold 1/2 untouched

## G (per-field gate) vs ZS + rules and FT + rules

**All numbers UNVERIFIED** until a verifier recomputes them. Aggregates and counts only: no gold or predicted value, no document id.

**No replacement decision is computed or printed**: this run does not satisfy spec section 11 item 7 (3 folds, cross-fitted across them).

## Provenance

- FT fold 0: `oof_native_fold0_4c17aa3`, manifest code_sha `4c17aa3c33c09f0cda7bb1f625947a1143a8cb28`, adapter sha256 `ef66090cf11f46576550dd6d019e59d3f0d202c617cecb7cd2f2a82bb09de89c`, training code SHA `4c17aa3c33c09f0cda7bb1f625947a1143a8cb28`
- ZS run folder: `zeroshot500_qwen35_4b_img_only_native_4c17aa3`, manifest code_sha `4c17aa3c33c09f0cda7bb1f625947a1143a8cb28`
- repo state: 10dbbb8+dirty
- command: `uv run python scripts/gate_eval.py --folds 0 --exploratory --oof-run-dir $SHIPDOC_TMP_DIR\native_extract\oof0\oof_native_fold0_4c17aa3 --zs-run-dir $SHIPDOC_TMP_DIR\native_extract\zs\zeroshot500_qwen35_4b_img_only_native_4c17aa3 --out reports/gate_fold0_native_exploratory.md --json-out $SHIPDOC_TMP_DIR\native_fold0\gate_fold0_native_exploratory.json`
- bootstrap: 2000 doc-level resamples, seed 42, unmodified scorer (`shipdoc.eval`), paired (system minus baseline)
- R3 shapes: per fold, learned from the gold of 329 (fold 0) docs outside that fold (supplier-disjoint, checked)
- wall time: 244 s

## Cross-fit (never fit on the docs it is applied to)

- mode: within-fold supplier-grouped cross-fit (inner k = 5, 10 supplier groups)
- calibrator: `v2_agree`, structure **pooled**, kind **gbm** (chosen once on the v2 design of the FT view; emitted OOF log-loss {"pooled/lr": 0.8561, "pooled/gbm": 0.15})
- before any model is fitted, the fit and apply sets are asserted doc- and supplier-group-disjoint for every held-out fold (`gate.assert_fit_apply_disjoint`; `cross_fit_design` asserts it again)

## Design decisions the spec does not define

- Variant: the per-field P(correct) is the `calibrate_v3` `v2_agree` calibrator (v2 features of the judged arm + FT-vs-ZS agreement features), one per judging view; the spec names the agreement features but not the variant.
- A slot filled by one arm only (the other blank), in the header or in an aligned row: the value is kept iff its P(correct) >= 0.5, else the other arm's blank is written. The calibrators have no P(null), so a blank has no P to compare with; this extends the spec's one-arm-row rule to fields. SPEC GAP, GG must see this.
- Row-level P (for a row present in one arm only): the MINIMUM P(correct) over the row's emitted fields; a row with no emitted field is dropped. `confidence_v3` / `calibrate_v3` define no row-level P and spec 11.7 does not say how to form one. SPEC GAP, GG must see this.
- A document on which the arms disagree about `doc_type` has different header field sets and no calibrated P for the type: the ZS + rules document is written unchanged. SPEC GAP.
- Ties (equal P, both arms filled and different) go to ZS + rules, as the spec's tie rule.

## Systems on 171 docs

Docs where G differs from ZS + rules: 74; from FT + rules: 24; where the two arms differ: 94.

### G vs ZS + rules

| metric (percent) | ZS + rules (95% CI) | G (95% CI) | delta G - ZS + rules (pts) [95% CI] | CI excludes 0 |
|---|---|---|---|---|
| OVERALL | 91.62 [89.76, 93.37] | 93.60 [91.91, 95.19] | +1.98 [+0.37, +3.49] | yes (positive) |
| header accuracy | 99.79 [99.50, 100.00] | 100.00 [100.00, 100.00] | +0.21 [+0.00, +0.50] | no |
| row F1 | 93.58 [91.78, 95.21] | 95.40 [93.98, 96.68] | +1.81 [+0.22, +3.40] | yes (positive) |
| documents fully correct | 71.35 [64.33, 77.78] | 77.19 [70.76, 83.04] | +5.85 [+0.58, +10.54] | yes (positive) |
| false-fill rate (lower is better) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] | no |

### G vs FT + rules

| metric (percent) | FT + rules (95% CI) | G (95% CI) | delta G - FT + rules (pts) [95% CI] | CI excludes 0 |
|---|---|---|---|---|
| OVERALL | 93.53 [91.88, 95.08] | 93.60 [91.91, 95.19] | +0.07 [-0.43, +0.66] | no |
| header accuracy | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | +0.00 [+0.00, +0.00] | no |
| row F1 | 95.53 [94.27, 96.71] | 95.40 [93.98, 96.68] | -0.13 [-0.62, +0.36] | no |
| documents fully correct | 76.61 [70.18, 82.46] | 77.19 [70.76, 83.04] | +0.58 [-1.17, +2.92] | no |
| false-fill rate (lower is better) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] | no |

### Over-nulls and false fills (raw counts)

| count | ZS + rules | FT + rules | G | ORACLE (reads gold) |
|---|---|---|---|---|
| false fills, header + row (G2 reads this) | 0 | 5 | 5 | 0 |
|   header false fills | 0 | 0 | 0 | 0 |
|   row false fills (scorer-paired rows) | 0 | 5 | 5 | 0 |
| over-null cells, header + row (G3 reads this) | 11 | 1 | 1 | 0 |
|   header over-nulls | 0 | 0 | 0 | 0 |
|   row over-nulls (scorer-paired rows) | 11 | 1 | 1 | 0 |
| gold rows left unmatched (missing rows) | 69 | 60 | 64 | 49 |

## What G took from which arm

| field type | slots | arms equal | arms differ | taken from FT | taken from ZS | FT share of differing |
|---|---|---|---|---|---|---|
| header | 1405 | 1338 | 67 | 53 | 14 | 79.1% |
| row.supplier_part_number | 1543 | 1510 | 33 | 22 | 11 | 66.7% |
| row.customer_part_number | 1543 | 1528 | 15 | 14 | 1 | 93.3% |
| row.purchase_order | 1543 | 1516 | 27 | 26 | 1 | 96.3% |
| row.quantity | 1543 | 1543 | 0 | 0 | 0 | n/a |

Ties (equal P, both arms filled, values differ) sent to ZS + rules: 0. One-arm FIELDS (value vs blank): FT-only value kept 17, dropped 0; ZS-only value kept 0, dropped 1. Documents with different `doc_type` (ZS document written): 0.

### Rows

| rows | count |
|---|---|
| aligned between the arms (per-field selection) | 1543 |
| FT-only, kept (row P >= 0.5) | 0 |
| FT-only, dropped | 0 |
| ZS-only, kept (row P >= 0.5) | 0 |
| ZS-only, dropped | 0 |

## ORACLE per-field selection (reads gold; a ceiling for context, not a system)

The same gate with P(correct) replaced by the gold label of each arm's field. For slots both arms fill it picks the arm that is right; one-arm rows are kept only if every emitted field is right. It bounds what ANY per-field selection between these two arms could reach.

| metric (percent) | ZS + rules (95% CI) | ORACLE (95% CI) | delta ORACLE - ZS + rules (pts) [95% CI] | CI excludes 0 |
|---|---|---|---|---|
| OVERALL | 91.62 [89.76, 93.37] | 94.99 [93.47, 96.43] | +3.37 [+2.09, +4.69] | yes (positive) |
| header accuracy | 99.79 [99.50, 100.00] | 100.00 [100.00, 100.00] | +0.21 [+0.00, +0.50] | no |
| row F1 | 93.58 [91.78, 95.21] | 96.82 [95.79, 97.78] | +3.24 [+1.89, +4.74] | yes (positive) |
| documents fully correct | 71.35 [64.33, 77.78] | 81.29 [75.44, 87.13] | +9.94 [+5.26, +14.62] | yes (positive) |
| false-fill rate (lower is better) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] | no |

## Decision

EXPLORATORY, NOT A DECISION; fold 0 was already seen when G was pre-registered; within-fold cross-fit; fold 1/2 untouched. The replacement rule was NOT evaluated.

## Limits

- The CI is a doc-level paired bootstrap; supplier clustering is not modelled, so it is optimistic when documents of one supplier move together.
- The calibrators are cross-fitted but the structure / kind choice used all folds' labels once (as `calibrate_v3`); CIs condition on the cross-fit.
- G is a post-hoc ensemble of two arms: any gain over the winner has to survive the three pre-registered clauses; a fold-0-only run is exploratory by construction.
- Rows aligned only by position (`align_rows` pass 4) can pair rows that are not the same row; G then mixes fields of two rows. Counted in `rows.paired`, not separated.
