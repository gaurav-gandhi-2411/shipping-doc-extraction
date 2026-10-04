# G per-field gate: pooled 3-fold decision

## G (per-field gate) vs ZS + rules and FT + rules

**All numbers UNVERIFIED** until a verifier recomputes them. Aggregates and counts only: no gold or predicted value, no document id.

## Provenance

- FT fold 0: `oof_native_fold0_4c17aa3`, manifest code_sha `4c17aa3c33c09f0cda7bb1f625947a1143a8cb28`, adapter sha256 `ef66090cf11f46576550dd6d019e59d3f0d202c617cecb7cd2f2a82bb09de89c`, training code SHA `4c17aa3c33c09f0cda7bb1f625947a1143a8cb28`
- FT fold 1: `oof_native_fold1_4c17aa3`, manifest code_sha `4c17aa3c33c09f0cda7bb1f625947a1143a8cb28`, adapter sha256 `4cbb2ffe885028e2246f3d53d8267ff6fb541b7111c603d234af672a2f605cad`, training code SHA `4c17aa3c33c09f0cda7bb1f625947a1143a8cb28`
- FT fold 2: `oof_native_fold2_4c17aa3`, manifest code_sha `4c17aa3c33c09f0cda7bb1f625947a1143a8cb28`, adapter sha256 `ba6cc2755ed26489734f37cb332cfd19a1d0df2a678b31d3af4d28eb35a21d51`, training code SHA `4c17aa3c33c09f0cda7bb1f625947a1143a8cb28`
- ZS run folder: `zeroshot500_qwen35_4b_img_only_native_4c17aa3`, manifest code_sha `4c17aa3c33c09f0cda7bb1f625947a1143a8cb28`
- repo state: 0c16e3a+dirty
- command: `uv run python scripts/gate_eval.py --folds 0 1 2 --oof-run-dir $SHIPDOC_TMP_DIR\native_extract\oof0\oof_native_fold0_4c17aa3 --oof-run-dir $SHIPDOC_RUNS_DIR\oof_native_fold1_4c17aa3 --oof-run-dir $SHIPDOC_RUNS_DIR\oof_native_fold2_4c17aa3 --zs-run-dir $SHIPDOC_TMP_DIR\native_extract\zs\zeroshot500_qwen35_4b_img_only_native_4c17aa3 --out reports/gate_pooled_native.md --json-out $SHIPDOC_TMP_DIR\pooled\gate_pooled_native.json`
- bootstrap: 2000 doc-level resamples, seed 42, unmodified scorer (`shipdoc.eval`), paired (system minus baseline)
- R3 shapes: per fold, learned from the gold of 329 (fold 0), 335 (fold 1), 336 (fold 2) docs outside that fold (supplier-disjoint, checked)
- wall time: 854 s

## Cross-fit (never fit on the docs it is applied to)

- mode: supplier-fold cross-fit over folds [0, 1, 2]
- calibrator: `v2_agree`, structure **pooled**, kind **lr** (chosen once on the v2 design of the FT view; emitted OOF log-loss {"pooled/lr": 0.1818, "pooled/gbm": 0.2158, "per_type/lr": 0.2087, "per_type/gbm": 0.2435})
- before any model is fitted, the fit and apply sets are asserted doc- and supplier-group-disjoint for every held-out fold (`gate.assert_fit_apply_disjoint`; `cross_fit_design` asserts it again)

## Design decisions the spec does not define

- Variant: the per-field P(correct) is the `calibrate_v3` `v2_agree` calibrator (v2 features of the judged arm + FT-vs-ZS agreement features), one per judging view; the spec names the agreement features but not the variant.
- A slot filled by one arm only (the other blank), in the header or in an aligned row: the value is kept iff its P(correct) >= 0.5, else the other arm's blank is written. The calibrators have no P(null), so a blank has no P to compare with; this extends the spec's one-arm-row rule to fields. SPEC GAP, GG must see this.
- Row-level P (for a row present in one arm only): the MINIMUM P(correct) over the row's emitted fields; a row with no emitted field is dropped. `confidence_v3` / `calibrate_v3` define no row-level P and spec 11.7 does not say how to form one. SPEC GAP, GG must see this.
- A document on which the arms disagree about `doc_type` has different header field sets and no calibrated P for the type: the ZS + rules document is written unchanged. SPEC GAP.
- Ties (equal P, both arms filled and different) go to ZS + rules, as the spec's tie rule.

## Systems on 500 docs

Docs where G differs from ZS + rules: 243; from FT + rules: 76; where the two arms differ: 295.

### G vs ZS + rules

| metric (percent) | ZS + rules (95% CI) | G (95% CI) | delta G - ZS + rules (pts) [95% CI] | CI excludes 0 |
|---|---|---|---|---|
| OVERALL | 88.51 [86.86, 90.06] | 92.38 [91.00, 93.69] | +3.87 [+2.88, +4.95] | yes (positive) |
| header accuracy | 99.83 [99.68, 99.95] | 99.98 [99.93, 100.00] | +0.15 [+0.02, +0.29] | yes (positive) |
| row F1 | 87.04 [84.16, 89.66] | 91.87 [89.61, 93.91] | +4.83 [+3.35, +6.54] | yes (positive) |
| documents fully correct | 68.80 [64.60, 72.80] | 78.20 [74.40, 81.60] | +9.40 [+6.40, +12.60] | yes (positive) |
| false-fill rate (lower is better) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] | no |

### G vs FT + rules

| metric (percent) | FT + rules (95% CI) | G (95% CI) | delta G - FT + rules (pts) [95% CI] | CI excludes 0 |
|---|---|---|---|---|
| OVERALL | 93.31 [92.05, 94.50] | 92.38 [91.00, 93.69] | -0.93 [-1.78, -0.23] | yes (negative) |
| header accuracy | 99.98 [99.93, 100.00] | 99.98 [99.93, 100.00] | +0.00 [+0.00, +0.00] | no |
| row F1 | 93.39 [91.40, 95.15] | 91.87 [89.61, 93.91] | -1.52 [-3.07, -0.37] | yes (negative) |
| documents fully correct | 79.80 [76.20, 83.00] | 78.20 [74.40, 81.60] | -1.60 [-3.20, +0.00] | no |
| false-fill rate (lower is better) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] | no |

### Over-nulls and false fills (raw counts)

| count | ZS + rules | FT + rules | G | ORACLE (reads gold) |
|---|---|---|---|---|
| false fills, header + row (G2 reads this) | 76 | 5 | 76 | 0 |
|   header false fills | 0 | 0 | 0 | 0 |
|   row false fills (scorer-paired rows) | 76 | 5 | 76 | 0 |
| over-null cells, header + row (G3 reads this) | 33 | 7 | 15 | 6 |
|   header over-nulls | 0 | 0 | 0 | 0 |
|   row over-nulls (scorer-paired rows) | 33 | 7 | 15 | 6 |
| gold rows left unmatched (missing rows) | 421 | 293 | 290 | 261 |

## What G took from which arm

| field type | slots | arms equal | arms differ | taken from FT | taken from ZS | FT share of differing |
|---|---|---|---|---|---|---|
| header | 4100 | 3908 | 192 | 156 | 36 | 81.2% |
| row.supplier_part_number | 4930 | 4686 | 244 | 223 | 21 | 91.4% |
| row.customer_part_number | 4930 | 4669 | 261 | 186 | 75 | 71.3% |
| row.purchase_order | 4930 | 4658 | 272 | 253 | 19 | 93.0% |
| row.quantity | 4930 | 4866 | 64 | 62 | 2 | 96.9% |

Ties (equal P, both arms filled, values differ) sent to ZS + rules: 0. One-arm FIELDS (value vs blank): FT-only value kept 139, dropped 14; ZS-only value kept 75, dropped 157. Documents with different `doc_type` (ZS document written): 0.

### Rows

| rows | count |
|---|---|
| aligned between the arms (per-field selection) | 4930 |
| FT-only, kept (row P >= 0.5) | 0 |
| FT-only, dropped | 0 |
| ZS-only, kept (row P >= 0.5) | 0 |
| ZS-only, dropped | 0 |

## ORACLE per-field selection (reads gold; a ceiling for context, not a system)

The same gate with P(correct) replaced by the gold label of each arm's field. For slots both arms fill it picks the arm that is right; one-arm rows are kept only if every emitted field is right. It bounds what ANY per-field selection between these two arms could reach.

| metric (percent) | ZS + rules (95% CI) | ORACLE (95% CI) | delta ORACLE - ZS + rules (pts) [95% CI] | CI excludes 0 |
|---|---|---|---|---|
| OVERALL | 88.51 [86.86, 90.06] | 94.30 [93.09, 95.48] | +5.80 [+4.63, +7.10] | yes (positive) |
| header accuracy | 99.83 [99.68, 99.95] | 100.00 [100.00, 100.00] | +0.17 [+0.05, +0.32] | yes (positive) |
| row F1 | 87.04 [84.16, 89.66] | 94.26 [92.24, 95.97] | +7.22 [+5.36, +9.45] | yes (positive) |
| documents fully correct | 68.80 [64.60, 72.80] | 83.00 [79.60, 86.20] | +14.20 [+11.20, +17.40] | yes (positive) |
| false-fill rate (lower is better) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] | no |

## Decision

Primary rule (spec section 11 item 1, decided first): **FINAL SYSTEM: FT+rules**

**SECONDARY SYSTEM G DOES NOT REPLACE FT+rules: FINAL SYSTEM: FT+rules [G1 failed: CI lower bound -1.7842 points (-1.784e-02) is below 0; G2 failed: false-fill count 76 (G) > 5 (winner); G3 failed: over-null count 15 (G) > 7 (winner)]**

| clause | requirement | result | reading |
|---|---|---|---|
| G1 | paired OVERALL delta (G - winner) CI lower bound strictly above 0 | FAIL | CI lower bound -1.7842 points (-1.784e-02) is below 0 |
| G2 | false-fill count of G <= the winner's (raw counts, no tolerance) | FAIL | false-fill count 76 (G) > 5 (winner) |
| G3 | over-null count of G <= the winner's (raw counts, no tolerance) | FAIL | over-null count 15 (G) > 7 (winner) |

## Limits

- The CI is a doc-level paired bootstrap; supplier clustering is not modelled, so it is optimistic when documents of one supplier move together.
- The calibrators are cross-fitted but the structure / kind choice used all folds' labels once (as `calibrate_v3`); CIs condition on the cross-fit.
- G is a post-hoc ensemble of two arms: any gain over the winner has to survive the three pre-registered clauses; a fold-0-only run is exploratory by construction.
- Rows aligned only by position (`align_rows` pass 4) can pair rows that are not the same row; G then mixes fields of two rows. Counted in `rows.paired`, not separated.
