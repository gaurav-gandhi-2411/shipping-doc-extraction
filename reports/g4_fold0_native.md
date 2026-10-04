# G4 fold 0 analysis: fine-tuned + rules vs zero-shot + rules

**All numbers UNVERIFIED** until a verifier recomputes them. Aggregates and counts only: no gold or predicted value, no document id.

## Provenance

- FT run folder: oof_native_fold0_4c17aa3
- FT manifest code_sha: 4c17aa3c33c09f0cda7bb1f625947a1143a8cb28
- FT oof section: fold 0, adapter sha256 `ef66090cf11f46576550dd6d019e59d3f0d202c617cecb7cd2f2a82bb09de89c`, training code SHA `4c17aa3c33c09f0cda7bb1f625947a1143a8cb28`, batch used 4, guard fallback to batch 1 False
- ZS run folder: zeroshot500_qwen35_4b_img_only_native_4c17aa3
- ZS manifest code_sha: 4c17aa3c33c09f0cda7bb1f625947a1143a8cb28
- repo state: ab92591+dirty
- command: `uv run python scripts/g4_fold.py --oof-run-dir $SHIPDOC_TMP_DIR\native_extract\oof0\oof_native_fold0_4c17aa3 --zs-run-dir $SHIPDOC_TMP_DIR\native_extract\zs\zeroshot500_qwen35_4b_img_only_native_4c17aa3 --fold 0 --out reports/g4_fold0_native.md --json-out $SHIPDOC_TMP_DIR\native_fold0\g4_fold0_native.json`
- bootstrap: 2000 doc-level resamples, seed 42, unmodified scorer (`shipdoc.eval`), paired for deltas (FT minus ZS)
- R3 shapes: learned from the gold of 329 docs outside fold 0 (supplier-disjoint, checked)

Docs: 171 held-out docs of fold 0. Both arms go through the PRODUCTION path (`postprocess_traces` -> coerce -> repair on the traces) with the same honest R3 shapes (learned excluding this fold's suppliers) and the OCR cache for R2. For each arm the same path with every rule OFF was asserted to reproduce the arm's `predictions.json` on these docs. Delta = FT minus ZS, paired doc-level bootstrap, unmodified scorer.

## Verdicts

**1.** G4 INTERIM VERDICT (interim, one fold, not the final G4 decision): NO REGRESSION [basis: both arms with production rules, PRIMARY]

   Same verdict on the RAW arms (secondary): G4 INTERIM VERDICT (interim, one fold, not the final G4 decision): NO REGRESSION [basis: raw arms]

**2.** G4 SPEC CLAUSE (interim, one fold, not the G4 decision): FINE-TUNED KEPT (reference only)

Interim verdict: NO REGRESSION iff the paired OVERALL delta CI lower bound > -1.0 point AND over-null cells (header + row) of FT <= ZS. Spec clause: FT kept only if OVERALL beats ZS + rules with the paired CI excluding 0 AND false-fill not worse (FT <= ZS).

## Headline: FT + rules vs ZS + rules (all docs)

| metric (percent) | ZS + rules (95% CI) | FT + rules (95% CI) | delta FT - ZS (pts) [95% CI] | CI excludes 0 |
|---|---|---|---|---|
| OVERALL | 91.62 [89.76, 93.37] | 93.53 [91.88, 95.08] | +1.92 [+0.36, +3.38] | yes (positive) |
| header accuracy | 99.79 [99.50, 100.00] | 100.00 [100.00, 100.00] | +0.21 [+0.00, +0.50] | no |
| row F1 | 93.58 [91.78, 95.21] | 95.53 [94.27, 96.71] | +1.94 [+0.35, +3.48] | yes (positive) |
| documents fully correct | 71.35 [64.33, 77.78] | 76.61 [70.18, 82.46] | +5.26 [+0.00, +10.53] | no |
| false-fill rate (lower is better) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] | no |

## Secondary: raw vs raw (no rules)

| metric (percent) | ZS raw (95% CI) | FT raw (95% CI) | delta FT - ZS (pts) [95% CI] | CI excludes 0 |
|---|---|---|---|---|
| OVERALL | 90.17 [88.21, 92.14] | 92.97 [91.30, 94.60] | +2.81 [+1.13, +4.46] | yes (positive) |
| header accuracy | 98.79 [98.01, 99.43] | 100.00 [100.00, 100.00] | +1.21 [+0.57, +1.99] | yes (positive) |
| row F1 | 93.58 [91.78, 95.21] | 95.01 [93.69, 96.22] | +1.43 [+0.00, +2.94] | no |
| documents fully correct | 66.08 [59.06, 73.10] | 74.85 [68.41, 80.70] | +8.77 [+2.92, +14.62] | yes (positive) |
| false-fill rate (lower is better) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] | no |

## Slices (FT + rules vs ZS + rules)

| slice | docs | metric | ZS | FT | delta FT - ZS (pts) [95% CI] | CI excludes 0 |
|---|---|---|---|---|---|---|
| invoice (HEADLINE slice) | 134 | OVERALL | 90.19 | 92.24 | +2.05 [+0.34, +3.83] | yes (positive) |
| invoice (HEADLINE slice) | 134 | header accuracy | 99.81 | 100.00 | +0.19 [+0.00, +0.47] | no |
| invoice (HEADLINE slice) | 134 | row F1 | 93.58 | 95.53 | +1.94 [+0.43, +3.60] | yes (positive) |
| invoice (HEADLINE slice) | 134 | documents fully correct | 64.18 | 70.15 | +5.97 [+0.00, +12.69] | no |
| invoice (HEADLINE slice) | 134 | false-fill rate (lower is better) | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | no |
| waybill | 37 | OVERALL | 99.22 | 100.00 | +0.78 [+0.00, +2.34] | no |
| waybill | 37 | header accuracy | 99.70 | 100.00 | +0.30 [+0.00, +0.90] | no |
| waybill | 37 | row F1 | 100.00 | 100.00 | +0.00 [+0.00, +0.00] | no |
| waybill | 37 | documents fully correct | 97.30 | 100.00 | +2.70 [+0.00, +8.11] | no |
| waybill | 37 | false-fill rate (lower is better) | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | no |
| scanned | 66 | OVERALL | 92.02 | 91.86 | -0.15 [-3.06, +2.60] | no |
| scanned | 66 | header accuracy | 99.82 | 100.00 | +0.18 [+0.00, +0.56] | no |
| scanned | 66 | row F1 | 93.11 | 93.30 | +0.19 [-3.18, +3.35] | no |
| scanned | 66 | documents fully correct | 74.24 | 72.73 | -1.52 [-10.61, +7.58] | no |
| scanned | 66 | false-fill rate (lower is better) | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | no |
| digital | 105 | OVERALL | 91.35 | 94.50 | +3.15 [+1.53, +4.86] | yes (positive) |
| digital | 105 | header accuracy | 99.77 | 100.00 | +0.23 [+0.00, +0.58] | no |
| digital | 105 | row F1 | 93.84 | 96.72 | +2.88 [+1.37, +4.61] | yes (positive) |
| digital | 105 | documents fully correct | 69.52 | 79.05 | +9.52 [+3.81, +16.19] | yes (positive) |
| digital | 105 | false-fill rate (lower is better) | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | no |
| all docs | 171 | OVERALL | 91.62 | 93.53 | +1.92 [+0.36, +3.38] | yes (positive) |
| all docs | 171 | header accuracy | 99.79 | 100.00 | +0.21 [+0.00, +0.50] | no |
| all docs | 171 | row F1 | 93.58 | 95.53 | +1.94 [+0.35, +3.48] | yes (positive) |
| all docs | 171 | documents fully correct | 71.35 | 76.61 | +5.26 [+0.00, +10.53] | no |
| all docs | 171 | false-fill rate (lower is better) | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | no |

## Over-nulls and false fills (all docs)

Over-null = gold has a value, prediction empty. Header per field; row fields on the rows the scorer pairs (gold rows nobody matched are counted as missing rows). Fields with a zero count in every column are omitted.

| count | ZS + rules | FT + rules | ZS raw | FT raw | FT - ZS (rules arms) |
|---|---|---|---|---|---|
| header over-null: carrier | 0 | 0 | 2 | 0 | 0 |
| header over-null: hawb | 0 | 0 | 4 | 0 | 0 |
| header over-null: mawb | 0 | 0 | 8 | 0 | 0 |
| header over-nulls, total | 0 | 0 | 14 | 0 | 0 |
| row over-null (scorer-paired rows): customer_part_number | 11 | 1 | 11 | 9 | -10 |
| row over-null (scorer-paired rows): purchase_order | 0 | 0 | 0 | 0 | 0 |
| row over-null (scorer-paired rows): quantity | 0 | 0 | 0 | 0 | 0 |
| row over-null (scorer-paired rows): supplier_part_number | 0 | 0 | 0 | 0 | 0 |
| row over-nulls, total | 11 | 1 | 11 | 9 | -10 |
| OVER-NULL CELLS, header + row (verdict clause) | 11 | 1 | 25 | 9 | -10 |
| header false fills | 0 | 0 | 0 | 0 | 0 |
| row false fills (scorer-paired rows) | 0 | 5 | 0 | 13 | 5 |
| gold rows left unmatched (missing rows) | 69 | 60 | 69 | 60 | -9 |

## Rules effect per arm (production path, rules on vs off)

fixed / broken / neutral: cell diffs rules-off -> rules-on scored against gold (R1 carrier, R2 mawb / hawb, R3 cpn / po of scorer-paired rows; an R3 row the scorer cannot pair is neutral).

| arm | rule | eligible docs | touched docs | cells changed | fixed | broken | neutral | net |
|---|---|---|---|---|---|---|---|---|
| ZS | R1 | 37 | 2 | 2 | 2 | 0 | 0 | 2 |
| ZS | R2 | 37 | 8 | 12 | 12 | 0 | 0 | 12 |
| ZS | R3 | 134 | 0 | 0 | 0 | 0 | 0 | 0 |
| FT | R1 | 37 | 0 | 0 | 0 | 0 | 0 | 0 |
| FT | R2 | 37 | 0 | 0 | 0 | 0 | 0 | 0 |
| FT | R3 | 134 | 3 | 8 | 8 | 0 | 0 | 8 |

| arm | OVERALL rules off | OVERALL rules on | delta on - off (pts) [95% CI] | CI excludes 0 |
|---|---|---|---|---|
| ZS | 90.17 | 91.62 | +1.45 [+0.61, +2.41] | yes (positive) |
| FT | 92.97 | 93.53 | +0.56 [+0.00, +1.23] | no |

R3 on fold 0 (shapes learned excluding this fold's suppliers): ZS: R3 did NOT fire; FT: touched 3 docs (8 row edits). R3 is judged only on folds where it fires (GG decision); a fold where it does not fire says nothing about it.

Rule skips (reason counts): ZS none; FT none.

## Row-error cause table (taxonomy of `scripts/row_error_diagnosis.py`)

Same classification, rerun on each arm's outputs for the fold's docs: invoice rows by cause, cpn / po slot mix-ups on scorer-paired rows, waybill header over-nulls. 'rules' columns are the production-processed outputs, 'raw' the model outputs. Counts only.

| count | ZS + rules | FT + rules | ZS raw | FT raw |
|---|---|---|---|---|
| invoice docs diagnosed | 134 | 134 | 134 | 134 |
| invoice docs with a truncated page | 0 | 0 | 0 | 0 |
| unpaired rows linked one-to-one (side pair), total | 69 | 60 | 69 | 60 |
| gold rows with no link (row_missing side), total | 0 | 0 | 0 | 0 |
| predicted rows with no link (row_extra side), total | 0 | 0 | 0 | 0 |
| cause: spn_copies_other_slot | 0 | 0 | 0 | 0 |
| cause: column_shift | 0 | 0 | 0 | 0 |
| cause: spn_null | 0 | 0 | 0 | 0 |
| cause: spn_misread | 69 | 60 | 69 | 60 |
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
| slot (scorer-paired rows): cpn_in_po_slot | 0 | 0 | 0 | 8 |
| slot (scorer-paired rows): cpn_over_null | 11 | 1 | 11 | 1 |
| slot (scorer-paired rows): po_false_fill | 0 | 5 | 0 | 5 |
| waybill docs | 37 | 37 | 37 | 37 |
| waybill carrier: over-null cells | 0 | 0 | 2 | 0 |
| waybill carrier: of which the model emitted a null | 0 | 0 | 2 | 0 |
| waybill carrier: of which the key is missing from the raw output | 0 | 0 | 0 | 0 |
| waybill carrier: of which the value sits in another raw header slot | 0 | 0 | 2 | 0 |
| waybill carrier: of which a unique OCR pattern match is correct | 0 | 0 | 0 | 0 |
| waybill hawb: over-null cells | 0 | 0 | 4 | 0 |
| waybill hawb: of which the model emitted a null | 0 | 0 | 4 | 0 |
| waybill hawb: of which the key is missing from the raw output | 0 | 0 | 0 | 0 |
| waybill hawb: of which the value sits in another raw header slot | 0 | 0 | 0 | 0 |
| waybill hawb: of which a unique OCR pattern match is correct | 0 | 0 | 4 | 0 |
| waybill mawb: over-null cells | 0 | 0 | 8 | 0 |
| waybill mawb: of which the model emitted a null | 0 | 0 | 8 | 0 |
| waybill mawb: of which the key is missing from the raw output | 0 | 0 | 0 | 0 |
| waybill mawb: of which the value sits in another raw header slot | 0 | 0 | 0 | 0 |
| waybill mawb: of which a unique OCR pattern match is correct | 0 | 0 | 8 | 0 |

## Limits

- One fold: the interim verdict and the spec clause are NOT the G4 decision.
- R3 shapes are learned from the other folds' gold; R3 may not fire on a fold, and is then not judged (it is judged only on folds where it fires).
- Row-error causes need the row page of each predicted row, replayed from the trace; where the replay does not give the final row count the pages are unknown (as in row_error_diagnosis).
- Waybill over-null facts use the model's raw page-1 output; R2 needs OCR at inference time.
- Over-null counts on rows use the scorer's pairing: a missing row is a missing row, not an over-null cell.
- Slices of one fold are small (see the docs column); wide CIs are expected.
