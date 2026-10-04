# G4 fold 1 analysis: fine-tuned + rules vs zero-shot + rules

**All numbers UNVERIFIED** until a verifier recomputes them. Aggregates and counts only: no gold or predicted value, no document id.

## Provenance

- FT run folder: oof_native_fold1_4c17aa3
- FT manifest code_sha: 4c17aa3c33c09f0cda7bb1f625947a1143a8cb28
- FT oof section: fold 1, adapter sha256 `4cbb2ffe885028e2246f3d53d8267ff6fb541b7111c603d234af672a2f605cad`, training code SHA `4c17aa3c33c09f0cda7bb1f625947a1143a8cb28`, batch used 4, guard fallback to batch 1 False
- ZS run folder: zeroshot500_qwen35_4b_img_only_native_4c17aa3
- ZS manifest code_sha: 4c17aa3c33c09f0cda7bb1f625947a1143a8cb28
- repo state: f725575+dirty
- command: `uv run python scripts/g4_fold.py --oof-run-dir $SHIPDOC_RUNS_DIR/oof_native_fold1_4c17aa3 --zs-run-dir $SHIPDOC_TMP_DIR/native_extract/zs/zeroshot500_qwen35_4b_img_only_native_4c17aa3 --fold 1 --out reports/g4_fold1_native.md --json-out $SHIPDOC_TMP_DIR/native_fold1/g4_fold1_native.json`
- bootstrap: 2000 doc-level resamples, seed 42, unmodified scorer (`shipdoc.eval`), paired for deltas (FT minus ZS)
- R3 shapes: learned from the gold of 335 docs outside fold 1 (supplier-disjoint, checked)

Docs: 165 held-out docs of fold 1. Both arms go through the PRODUCTION path (`postprocess_traces` -> coerce -> repair on the traces) with the same honest R3 shapes (learned excluding this fold's suppliers) and the OCR cache for R2. For each arm the same path with every rule OFF was asserted to reproduce the arm's `predictions.json` on these docs. Delta = FT minus ZS, paired doc-level bootstrap, unmodified scorer.

## Verdicts

**1.** G4 INTERIM VERDICT (interim, one fold, not the final G4 decision): NO REGRESSION [basis: both arms with production rules, PRIMARY]

   Same verdict on the RAW arms (secondary): G4 INTERIM VERDICT (interim, one fold, not the final G4 decision): NO REGRESSION [basis: raw arms]

**2.** G4 SPEC CLAUSE (interim, one fold, not the G4 decision): FINE-TUNED KEPT (reference only)

Interim verdict: NO REGRESSION iff the paired OVERALL delta CI lower bound > -1.0 point AND over-null cells (header + row) of FT <= ZS. Spec clause: FT kept only if OVERALL beats ZS + rules with the paired CI excluding 0 AND false-fill not worse (FT <= ZS).

## Headline: FT + rules vs ZS + rules (all docs)

| metric (percent) | ZS + rules (95% CI) | FT + rules (95% CI) | delta FT - ZS (pts) [95% CI] | CI excludes 0 |
|---|---|---|---|---|
| OVERALL | 90.14 [87.13, 92.90] | 97.03 [95.75, 98.15] | +6.89 [+4.35, +9.72] | yes (positive) |
| header accuracy | 99.85 [99.63, 100.00] | 99.93 [99.78, 100.00] | +0.07 [-0.15, +0.30] | no |
| row F1 | 87.92 [82.82, 92.49] | 98.70 [98.12, 99.23] | +10.78 [+6.23, +15.89] | yes (positive) |
| documents fully correct | 75.15 [67.88, 81.82] | 87.88 [82.42, 92.73] | +12.73 [+6.67, +18.79] | yes (positive) |
| false-fill rate (lower is better) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] | no |

## Secondary: raw vs raw (no rules)

| metric (percent) | ZS raw (95% CI) | FT raw (95% CI) | delta FT - ZS (pts) [95% CI] | CI excludes 0 |
|---|---|---|---|---|
| OVERALL | 82.43 [78.30, 86.27] | 96.74 [95.40, 97.94] | +14.31 [+10.65, +18.07] | yes (positive) |
| header accuracy | 98.89 [98.24, 99.48] | 99.93 [99.78, 100.00] | +1.04 [+0.45, +1.69] | yes (positive) |
| row F1 | 75.37 [67.31, 82.63] | 98.28 [97.23, 99.07] | +22.91 [+15.72, +30.79] | yes (positive) |
| documents fully correct | 63.64 [56.36, 70.30] | 87.27 [81.82, 92.12] | +23.64 [+16.95, +30.30] | yes (positive) |
| false-fill rate (lower is better) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] | no |

## Slices (FT + rules vs ZS + rules)

| slice | docs | metric | ZS | FT | delta FT - ZS (pts) [95% CI] | CI excludes 0 |
|---|---|---|---|---|---|---|
| invoice (HEADLINE slice) | 134 | OVERALL | 88.97 | 96.46 | +7.48 [+4.70, +10.22] | yes (positive) |
| invoice (HEADLINE slice) | 134 | header accuracy | 99.81 | 99.91 | +0.09 [-0.19, +0.47] | no |
| invoice (HEADLINE slice) | 134 | row F1 | 87.92 | 98.70 | +10.78 [+6.34, +15.34] | yes (positive) |
| invoice (HEADLINE slice) | 134 | documents fully correct | 69.40 | 85.07 | +15.67 [+8.21, +23.13] | yes (positive) |
| invoice (HEADLINE slice) | 134 | false-fill rate (lower is better) | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | no |
| waybill | 31 | OVERALL | 100.00 | 100.00 | +0.00 [+0.00, +0.00] | no |
| waybill | 31 | header accuracy | 100.00 | 100.00 | +0.00 [+0.00, +0.00] | no |
| waybill | 31 | row F1 | 100.00 | 100.00 | +0.00 [+0.00, +0.00] | no |
| waybill | 31 | documents fully correct | 100.00 | 100.00 | +0.00 [+0.00, +0.00] | no |
| waybill | 31 | false-fill rate (lower is better) | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | no |
| scanned | 56 | OVERALL | 90.17 | 96.97 | +6.80 [+2.29, +11.63] | yes (positive) |
| scanned | 56 | header accuracy | 100.00 | 100.00 | +0.00 [+0.00, +0.00] | no |
| scanned | 56 | row F1 | 87.93 | 98.68 | +10.74 [+3.23, +19.15] | yes (positive) |
| scanned | 56 | documents fully correct | 75.00 | 87.50 | +12.50 [+1.79, +23.26] | yes (positive) |
| scanned | 56 | false-fill rate (lower is better) | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | no |
| digital | 109 | OVERALL | 90.12 | 97.05 | +6.93 [+4.02, +10.52] | yes (positive) |
| digital | 109 | header accuracy | 99.78 | 99.89 | +0.11 [-0.23, +0.56] | no |
| digital | 109 | row F1 | 87.92 | 98.71 | +10.79 [+5.49, +17.13] | yes (positive) |
| digital | 109 | documents fully correct | 75.23 | 88.07 | +12.84 [+6.42, +20.18] | yes (positive) |
| digital | 109 | false-fill rate (lower is better) | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | no |
| all docs | 165 | OVERALL | 90.14 | 97.03 | +6.89 [+4.35, +9.72] | yes (positive) |
| all docs | 165 | header accuracy | 99.85 | 99.93 | +0.07 [-0.15, +0.30] | no |
| all docs | 165 | row F1 | 87.92 | 98.70 | +10.78 [+6.23, +15.89] | yes (positive) |
| all docs | 165 | documents fully correct | 75.15 | 87.88 | +12.73 [+6.67, +18.79] | yes (positive) |
| all docs | 165 | false-fill rate (lower is better) | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | no |

## Over-nulls and false fills (all docs)

Over-null = gold has a value, prediction empty. Header per field; row fields on the rows the scorer pairs (gold rows nobody matched are counted as missing rows). Fields with a zero count in every column are omitted.

| count | ZS + rules | FT + rules | ZS raw | FT raw | FT - ZS (rules arms) |
|---|---|---|---|---|---|
| header over-null: carrier | 0 | 0 | 2 | 0 | 0 |
| header over-null: hawb | 0 | 0 | 2 | 0 | 0 |
| header over-null: mawb | 0 | 0 | 9 | 0 | 0 |
| header over-nulls, total | 0 | 0 | 13 | 0 | 0 |
| row over-null (scorer-paired rows): customer_part_number | 0 | 0 | 0 | 0 | 0 |
| row over-null (scorer-paired rows): purchase_order | 21 | 1 | 233 | 8 | -20 |
| row over-null (scorer-paired rows): quantity | 0 | 0 | 0 | 0 | 0 |
| row over-null (scorer-paired rows): supplier_part_number | 0 | 0 | 0 | 0 | 0 |
| row over-nulls, total | 21 | 1 | 233 | 8 | -20 |
| OVER-NULL CELLS, header + row (verdict clause) | 21 | 1 | 246 | 8 | -20 |
| header false fills | 0 | 0 | 0 | 0 | 0 |
| row false fills (scorer-paired rows) | 76 | 0 | 288 | 7 | -76 |
| gold rows left unmatched (missing rows) | 103 | 17 | 103 | 17 | -86 |

## Rules effect per arm (production path, rules on vs off)

fixed / broken / neutral: cell diffs rules-off -> rules-on scored against gold (R1 carrier, R2 mawb / hawb, R3 cpn / po of scorer-paired rows; an R3 row the scorer cannot pair is neutral).

| arm | rule | eligible docs | touched docs | cells changed | fixed | broken | neutral | net |
|---|---|---|---|---|---|---|---|---|
| ZS | R1 | 31 | 2 | 2 | 2 | 0 | 0 | 2 |
| ZS | R2 | 31 | 9 | 11 | 11 | 0 | 0 | 11 |
| ZS | R3 | 134 | 18 | 221 | 212 | 0 | 9 | 212 |
| FT | R1 | 31 | 0 | 0 | 0 | 0 | 0 | 0 |
| FT | R2 | 31 | 0 | 0 | 0 | 0 | 0 | 0 |
| FT | R3 | 134 | 1 | 7 | 7 | 0 | 0 | 7 |

| arm | OVERALL rules off | OVERALL rules on | delta on - off (pts) [95% CI] | CI excludes 0 |
|---|---|---|---|---|
| ZS | 82.43 | 90.14 | +7.71 [+5.04, +10.58] | yes (positive) |
| FT | 96.74 | 97.03 | +0.29 [+0.00, +0.90] | no |

R3 on fold 1 (shapes learned excluding this fold's suppliers): ZS: touched 18 docs (221 row edits); FT: touched 1 docs (7 row edits). R3 is judged only on folds where it fires (GG decision); a fold where it does not fire says nothing about it.

Rule skips (reason counts): ZS none; FT none.

## Row-error cause table (taxonomy of `scripts/row_error_diagnosis.py`)

Same classification, rerun on each arm's outputs for the fold's docs: invoice rows by cause, cpn / po slot mix-ups on scorer-paired rows, waybill header over-nulls. 'rules' columns are the production-processed outputs, 'raw' the model outputs. Counts only.

| count | ZS + rules | FT + rules | ZS raw | FT raw |
|---|---|---|---|---|
| invoice docs diagnosed | 134 | 134 | 134 | 134 |
| invoice docs with a truncated page | 0 | 0 | 0 | 0 |
| unpaired rows linked one-to-one (side pair), total | 103 | 17 | 103 | 17 |
| gold rows with no link (row_missing side), total | 0 | 0 | 0 | 0 |
| predicted rows with no link (row_extra side), total | 0 | 0 | 0 | 0 |
| cause: spn_copies_other_slot | 6 | 0 | 6 | 0 |
| cause: column_shift | 67 | 0 | 76 | 0 |
| cause: spn_null | 0 | 0 | 0 | 0 |
| cause: spn_misread | 30 | 17 | 21 | 17 |
| cause: spn_other | 0 | 0 | 0 | 0 |
| cause: weak_link_qty | 0 | 0 | 0 | 0 |
| cause: weak_link_position | 0 | 0 | 0 | 0 |
| cause: truncated_page | 0 | 0 | 0 | 0 |
| cause: page_dropped | 0 | 0 | 0 | 0 |
| cause: merged | 0 | 0 | 0 | 0 |
| cause: page_short | 0 | 0 | 0 | 0 |
| cause: missing_other | 0 | 0 | 0 | 0 |
| cause: header_row | 0 | 0 | 0 | 0 |
| cause: split | 0 | 0 | 0 | 0 |
| cause: duplicate_boundary | 0 | 0 | 0 | 0 |
| cause: duplicate_other | 0 | 0 | 0 | 0 |
| cause: extra_other | 0 | 0 | 0 | 0 |
| slot (scorer-paired rows): cpn_false_fill | 75 | 0 | 75 | 0 |
| slot (scorer-paired rows): mixed | 1 | 0 | 1 | 0 |
| slot (scorer-paired rows): po_in_cpn_slot | 0 | 0 | 212 | 7 |
| slot (scorer-paired rows): po_over_null | 20 | 1 | 20 | 1 |
| waybill docs | 31 | 31 | 31 | 31 |
| waybill carrier: over-null cells | 0 | 0 | 2 | 0 |
| waybill carrier: of which the model emitted a null | 0 | 0 | 2 | 0 |
| waybill carrier: of which the key is missing from the raw output | 0 | 0 | 0 | 0 |
| waybill carrier: of which the value sits in another raw header slot | 0 | 0 | 2 | 0 |
| waybill carrier: of which a unique OCR pattern match is correct | 0 | 0 | 0 | 0 |
| waybill hawb: over-null cells | 0 | 0 | 2 | 0 |
| waybill hawb: of which the model emitted a null | 0 | 0 | 2 | 0 |
| waybill hawb: of which the key is missing from the raw output | 0 | 0 | 0 | 0 |
| waybill hawb: of which the value sits in another raw header slot | 0 | 0 | 0 | 0 |
| waybill hawb: of which a unique OCR pattern match is correct | 0 | 0 | 2 | 0 |
| waybill mawb: over-null cells | 0 | 0 | 9 | 0 |
| waybill mawb: of which the model emitted a null | 0 | 0 | 9 | 0 |
| waybill mawb: of which the key is missing from the raw output | 0 | 0 | 0 | 0 |
| waybill mawb: of which the value sits in another raw header slot | 0 | 0 | 0 | 0 |
| waybill mawb: of which a unique OCR pattern match is correct | 0 | 0 | 9 | 0 |

## Limits

- One fold: the interim verdict and the spec clause are NOT the G4 decision.
- R3 shapes are learned from the other folds' gold; R3 may not fire on a fold, and is then not judged (it is judged only on folds where it fires).
- Row-error causes need the row page of each predicted row, replayed from the trace; where the replay does not give the final row count the pages are unknown (as in row_error_diagnosis).
- Waybill over-null facts use the model's raw page-1 output; R2 needs OCR at inference time.
- Over-null counts on rows use the scorer's pairing: a missing row is a missing row, not an over-null cell.
- Slices of one fold are small (see the docs column); wide CIs are expected.
