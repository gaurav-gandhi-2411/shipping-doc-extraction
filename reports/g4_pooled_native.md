# G4 pooled 3-fold: final-system decision (FT+rules vs ZS+rules)

**FINAL SYSTEM: FT+rules**

**All numbers UNVERIFIED** until a verifier recomputes them. Aggregates and counts only: no gold or predicted value, no document id.

## Provenance

- FT fold 0: `oof_native_fold0_4c17aa3`, manifest code_sha `4c17aa3c33c09f0cda7bb1f625947a1143a8cb28`, adapter sha256 `ef66090cf11f46576550dd6d019e59d3f0d202c617cecb7cd2f2a82bb09de89c`, batch used 4, guard fallback to batch 1 False
- FT fold 1: `oof_native_fold1_4c17aa3`, manifest code_sha `4c17aa3c33c09f0cda7bb1f625947a1143a8cb28`, adapter sha256 `4cbb2ffe885028e2246f3d53d8267ff6fb541b7111c603d234af672a2f605cad`, batch used 4, guard fallback to batch 1 False
- FT fold 2: `oof_native_fold2_4c17aa3`, manifest code_sha `4c17aa3c33c09f0cda7bb1f625947a1143a8cb28`, adapter sha256 `ba6cc2755ed26489734f37cb332cfd19a1d0df2a678b31d3af4d28eb35a21d51`, batch used 4, guard fallback to batch 1 False
- adapter training code SHA (same on all folds, checked): `4c17aa3c33c09f0cda7bb1f625947a1143a8cb28`
- ZS run folder: `zeroshot500_qwen35_4b_img_only_native_4c17aa3`, manifest code_sha `4c17aa3c33c09f0cda7bb1f625947a1143a8cb28`
- repo state: 0c16e3a
- command: `uv run python scripts/g4_pooled.py --oof-run-dir $SHIPDOC_TMP_DIR\native_extract\oof0\oof_native_fold0_4c17aa3 --oof-run-dir $SHIPDOC_RUNS_DIR\oof_native_fold1_4c17aa3 --oof-run-dir $SHIPDOC_RUNS_DIR\oof_native_fold2_4c17aa3 --zs-run-dir $SHIPDOC_TMP_DIR\native_extract\zs\zeroshot500_qwen35_4b_img_only_native_4c17aa3 --out reports/g4_pooled_native.md --json-out $SHIPDOC_TMP_DIR\pooled\g4_pooled_native.json`
- bootstrap: 2000 doc-level resamples, seed 42, unmodified scorer (`shipdoc.eval`), paired for deltas (FT minus ZS), pooled over all docs
- R3 shapes: per fold, learned from the gold of 329 (fold 0), 335 (fold 1), 336 (fold 2) docs outside that fold (supplier-disjoint, checked)

## Pre-registered rule (spec section 11 item 1)

FT+rules is the final system only if C1, C2 and C3 all hold; otherwise ZS+rules (v1). A tie goes to ZS+rules: FT needs a strict CI-positive win. A CI lower bound of exactly 0 fails C1; equal counts pass C2 / C3; any count above ZS's fails.

## Clause table

| clause | requirement | numbers | result | reading |
|---|---|---|---|---|
| C1 | paired OVERALL delta (FT - ZS) CI lower bound strictly above 0 | OVERALL delta +4.80 pts, 95% CI [+3.59, +6.14] | PASS | CI lower bound +3.5929 points (+3.593e-02) is above 0 |
| C2 | false-fill count of FT+rules <= ZS+rules (raw counts, no tolerance) | FT 5 vs ZS 76 (header 0 vs 0, row 5 vs 76) | PASS | false-fill count 5 (FT) <= 76 (ZS) |
| C3 | over-null count of FT+rules <= ZS+rules (raw counts, no tolerance) | FT 7 vs ZS 33 (header 0 vs 0, row 7 vs 33) | PASS | over-null count 7 (FT) <= 33 (ZS) |

- C1 PASS: CI lower bound +3.5929 points (+3.593e-02) is above 0
- C2 PASS: false-fill count 5 (FT) <= 76 (ZS)
- C3 PASS: over-null count 7 (FT) <= 33 (ZS)

Pooled over 500 docs. Delta = FT minus ZS, paired doc-level bootstrap (2000 resamples, seed 42, unmodified scorer).

## Over-nulls and false fills per field (pooled; C2 and C3 read the totals)

| count | ZS + rules | FT + rules | ZS raw | FT raw | FT - ZS (rules arms) |
|---|---|---|---|---|---|
| header over-null: carrier | 0 | 0 | 7 | 0 | 0 |
| header over-null: hawb | 0 | 0 | 12 | 0 | 0 |
| header over-null: mawb | 0 | 0 | 29 | 0 | 0 |
| header over-nulls, total | 0 | 0 | 48 | 0 | 0 |
| row over-null (scorer-paired rows): customer_part_number | 12 | 1 | 12 | 14 | -11 |
| row over-null (scorer-paired rows): purchase_order | 21 | 6 | 299 | 22 | -15 |
| row over-null (scorer-paired rows): quantity | 0 | 0 | 0 | 0 | 0 |
| row over-null (scorer-paired rows): supplier_part_number | 0 | 0 | 0 | 0 | 0 |
| row over-nulls, total | 33 | 7 | 311 | 36 | -26 |
| OVER-NULL CELLS, header + row (verdict clause) | 33 | 7 | 359 | 36 | -26 |
| header false fills | 0 | 0 | 0 | 0 | 0 |
| row false fills (scorer-paired rows) | 76 | 5 | 354 | 34 | -71 |
| gold rows left unmatched (missing rows) | 421 | 293 | 421 | 293 | -128 |

## Headline: FT + rules vs ZS + rules (pooled, all docs)

| metric (percent) | ZS + rules (95% CI) | FT + rules (95% CI) | delta FT - ZS (pts) [95% CI] | CI excludes 0 |
|---|---|---|---|---|
| OVERALL | 88.51 [86.86, 90.06] | 93.31 [92.05, 94.50] | +4.80 [+3.59, +6.14] | yes (positive) |
| header accuracy | 99.83 [99.68, 99.95] | 99.98 [99.93, 100.00] | +0.15 [+0.02, +0.29] | yes (positive) |
| row F1 | 87.04 [84.16, 89.66] | 93.39 [91.40, 95.15] | +6.35 [+4.37, +8.64] | yes (positive) |
| documents fully correct | 68.80 [64.60, 72.80] | 79.80 [76.20, 83.00] | +11.00 [+7.80, +14.40] | yes (positive) |
| false-fill rate (lower is better) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] | no |

## Secondary: raw vs raw (no rules, pooled)

| metric (percent) | ZS raw (95% CI) | FT raw (95% CI) | delta FT - ZS (pts) [95% CI] | CI excludes 0 |
|---|---|---|---|---|
| OVERALL | 83.86 [81.88, 85.72] | 92.83 [91.53, 94.11] | +8.97 [+7.36, +10.71] | yes (positive) |
| header accuracy | 98.66 [98.22, 99.10] | 99.98 [99.93, 100.00] | +1.32 [+0.90, +1.76] | yes (positive) |
| row F1 | 81.40 [77.88, 84.72] | 92.80 [90.75, 94.64] | +11.40 [+8.41, +14.65] | yes (positive) |
| documents fully correct | 59.20 [54.80, 63.40] | 78.60 [75.00, 82.00] | +19.40 [+15.60, +23.60] | yes (positive) |
| false-fill rate (lower is better) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] | no |

## Slices (FT + rules vs ZS + rules, pooled)

| slice | docs | metric | ZS | FT | delta FT - ZS (pts) [95% CI] | CI excludes 0 |
|---|---|---|---|---|---|---|
| invoice (HEADLINE slice) | 400 | OVERALL | 86.99 | 92.29 | +5.30 [+3.87, +6.75] | yes (positive) |
| invoice (HEADLINE slice) | 400 | header accuracy | 99.81 | 99.97 | +0.16 [+0.00, +0.34] | no |
| invoice (HEADLINE slice) | 400 | row F1 | 87.04 | 93.39 | +6.35 [+4.29, +8.46] | yes (positive) |
| invoice (HEADLINE slice) | 400 | documents fully correct | 61.25 | 74.75 | +13.50 [+9.50, +17.75] | yes (positive) |
| invoice (HEADLINE slice) | 400 | false-fill rate (lower is better) | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | no |
| waybill | 100 | OVERALL | 99.71 | 100.00 | +0.29 [+0.00, +0.87] | no |
| waybill | 100 | header accuracy | 99.89 | 100.00 | +0.11 [+0.00, +0.33] | no |
| waybill | 100 | row F1 | 100.00 | 100.00 | +0.00 [+0.00, +0.00] | no |
| waybill | 100 | documents fully correct | 99.00 | 100.00 | +1.00 [+0.00, +3.00] | no |
| waybill | 100 | false-fill rate (lower is better) | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | no |
| scanned | 193 | OVERALL | 86.92 | 91.46 | +4.55 [+2.45, +6.68] | yes (positive) |
| scanned | 193 | header accuracy | 99.87 | 100.00 | +0.13 [+0.00, +0.32] | no |
| scanned | 193 | row F1 | 84.78 | 91.10 | +6.31 [+3.14, +10.00] | yes (positive) |
| scanned | 193 | documents fully correct | 65.28 | 75.13 | +9.84 [+4.15, +16.06] | yes (positive) |
| scanned | 193 | false-fill rate (lower is better) | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | no |
| digital | 307 | OVERALL | 89.48 | 94.44 | +4.96 [+3.39, +6.49] | yes (positive) |
| digital | 307 | header accuracy | 99.80 | 99.96 | +0.16 [-0.00, +0.36] | no |
| digital | 307 | row F1 | 88.40 | 94.77 | +6.37 [+3.96, +8.98] | yes (positive) |
| digital | 307 | documents fully correct | 71.01 | 82.74 | +11.73 [+7.82, +15.64] | yes (positive) |
| digital | 307 | false-fill rate (lower is better) | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | no |
| all docs | 500 | OVERALL | 88.51 | 93.31 | +4.80 [+3.59, +6.14] | yes (positive) |
| all docs | 500 | header accuracy | 99.83 | 99.98 | +0.15 [+0.02, +0.29] | yes (positive) |
| all docs | 500 | row F1 | 87.04 | 93.39 | +6.35 [+4.37, +8.64] | yes (positive) |
| all docs | 500 | documents fully correct | 68.80 | 79.80 | +11.00 [+7.80, +14.40] | yes (positive) |
| all docs | 500 | false-fill rate (lower is better) | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | no |

## Per-fold OVERALL (information, not part of the decision)

| fold | docs | OVERALL ZS+rules | OVERALL FT+rules | delta FT - ZS (pts) [95% CI] |
|---|---|---|---|---|
| 0 | 171 | 91.62 | 93.53 | +1.92 [+0.36, +3.38] |
| 1 | 165 | 90.14 | 97.03 | +6.89 [+4.35, +9.72] |
| 2 | 164 | 83.98 | 89.46 | +5.49 [+3.56, +7.56] |

## Rules per fold and arm (R3 is judged only on folds where it fires)

fixed / broken / neutral: cell diffs rules-off -> rules-on scored against gold, as in `reports/g4_fold0.md`. Shapes for R3 on fold k exclude fold k's suppliers.

| fold | arm | rule | eligible docs | touched docs | cells changed | fixed | broken | neutral |
|---|---|---|---|---|---|---|---|---|
| 0 | ZS | R1 | 37 | 2 | 2 | 2 | 0 | 0 |
| 0 | ZS | R2 | 37 | 8 | 12 | 12 | 0 | 0 |
| 0 | ZS | R3 | 134 | 0 | 0 | 0 | 0 | 0 |
| 0 | FT | R1 | 37 | 0 | 0 | 0 | 0 | 0 |
| 0 | FT | R2 | 37 | 0 | 0 | 0 | 0 | 0 |
| 0 | FT | R3 | 134 | 3 | 8 | 8 | 0 | 0 |
| 1 | ZS | R1 | 31 | 2 | 2 | 2 | 0 | 0 |
| 1 | ZS | R2 | 31 | 9 | 11 | 11 | 0 | 0 |
| 1 | ZS | R3 | 134 | 18 | 221 | 212 | 0 | 9 |
| 1 | FT | R1 | 31 | 0 | 0 | 0 | 0 | 0 |
| 1 | FT | R2 | 31 | 0 | 0 | 0 | 0 | 0 |
| 1 | FT | R3 | 134 | 1 | 7 | 7 | 0 | 0 |
| 2 | ZS | R1 | 32 | 3 | 3 | 3 | 0 | 0 |
| 2 | ZS | R2 | 32 | 12 | 18 | 18 | 0 | 0 |
| 2 | ZS | R3 | 132 | 7 | 66 | 66 | 0 | 0 |
| 2 | FT | R1 | 32 | 0 | 0 | 0 | 0 | 0 |
| 2 | FT | R2 | 32 | 0 | 0 | 0 | 0 | 0 |
| 2 | FT | R3 | 132 | 3 | 19 | 14 | 0 | 5 |

No R3 warning: R3 broke no cell on the FT arm of any fold other than fold 0 (or did not fire there). Fold 0 can neither clear nor sink R3.

## R3-off sensitivity (informational)

Same rule on both arms with R1 and R2 on and R3 OFF: FINAL SYSTEM: FT+rules

| clause | requirement | numbers | result | reading |
|---|---|---|---|---|
| C1 | paired OVERALL delta (FT - ZS) CI lower bound strictly above 0 | OVERALL delta +7.18 pts, 95% CI [+5.59, +8.81] | PASS | CI lower bound +5.5933 points (+5.593e-02) is above 0 |
| C2 | false-fill count of FT+rules <= ZS+rules (raw counts, no tolerance) | FT 34 vs ZS 354 (header 0 vs 0, row 34 vs 354) | PASS | false-fill count 34 (FT) <= 354 (ZS) |
| C3 | over-null count of FT+rules <= ZS+rules (raw counts, no tolerance) | FT 36 vs ZS 311 (header 0 vs 0, row 36 vs 311) | PASS | over-null count 36 (FT) <= 311 (ZS) |

## Limits

- The decision uses only C1 to C3 above; every other table is information.
- Both arms use the production rules (R1 / R2 / R3) and, for R3, the shapes learned from the gold of the other folds only (supplier-disjoint, checked). The R3-off sensitivity is informational: the registered decision uses all three rules.
- The CI is a doc-level paired bootstrap over the pooled docs; supplier clustering is not modelled, so the CI is optimistic when docs of one supplier move together.
- R3 concentration: 3 of 18 invoice groups carry 493 of 502 fixed rows (spec section 11 item 3, from the earlier R3 gate report); a pooled gain can be one or two suppliers.
- Over-null counts on rows use the scorer's pairing: a missing row is a missing row, not an over-null cell.
