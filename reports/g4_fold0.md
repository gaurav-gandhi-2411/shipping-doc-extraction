# G4 fold 0 analysis: fine-tuned + rules vs zero-shot + rules

**All numbers UNVERIFIED** until a verifier recomputes them. Aggregates and counts only: no gold or predicted value, no document id.

## Provenance

- FT run folder: oof_fold0_42b812b
- FT manifest code_sha: 42b812b5b09d6e4bff0df12564017f71ffad5fc9
- FT oof section: fold 0, adapter sha256 `80286fba73049dac2c363d1e12f921a0800b5d8acfdbca77fc7cad26c7575c3f`, training code SHA `42b812b5b09d6e4bff0df12564017f71ffad5fc9`, batch used 8, guard fallback to batch 1 False
- ZS run folder: zeroshot500_qwen35_4b_img_only_keyed_42b812b
- ZS manifest code_sha: 42b812b5b09d6e4bff0df12564017f71ffad5fc9
- repo state: fd9b946
- command: `uv run python scripts/g4_fold.py --oof-run-dir $SHIPDOC_RUNS_DIR/oof_fold0/oof_fold0_42b812b --zs-run-dir $SHIPDOC_RUNS_DIR/zeroshot500/zeroshot500_qwen35_4b_img_only_keyed_42b812b --out reports/g4_fold0.md`
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
| OVERALL | 86.05 [83.62, 88.36] | 89.95 [88.01, 91.87] | +3.91 [+2.37, +5.63] | yes (positive) |
| header accuracy | 99.00 [98.44, 99.50] | 99.79 [99.50, 100.00] | +0.78 [+0.35, +1.29] | yes (positive) |
| row F1 | 86.58 [83.58, 89.46] | 92.06 [90.26, 93.77] | +5.47 [+3.29, +7.96] | yes (positive) |
| documents fully correct | 59.06 [51.46, 66.67] | 66.08 [58.48, 73.10] | +7.02 [+2.34, +12.28] | yes (positive) |
| false-fill rate (lower is better) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] | no |

## Secondary: raw vs raw (no rules)

| metric (percent) | ZS raw (95% CI) | FT raw (95% CI) | delta FT - ZS (pts) [95% CI] | CI excludes 0 |
|---|---|---|---|---|
| OVERALL | 84.22 [81.75, 86.57] | 89.82 [87.87, 91.77] | +5.61 [+3.84, +7.54] | yes (positive) |
| header accuracy | 97.94 [97.10, 98.65] | 99.79 [99.50, 100.00] | +1.85 [+1.14, +2.69] | yes (positive) |
| row F1 | 86.58 [83.58, 89.46] | 91.73 [89.94, 93.52] | +5.15 [+3.08, +7.60] | yes (positive) |
| documents fully correct | 52.05 [44.44, 59.65] | 66.08 [58.48, 73.10] | +14.04 [+8.19, +20.47] | yes (positive) |
| false-fill rate (lower is better) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] | no |

## Slices (FT + rules vs ZS + rules)

| slice | docs | metric | ZS | FT | delta FT - ZS (pts) [95% CI] | CI excludes 0 |
|---|---|---|---|---|---|---|
| invoice (HEADLINE slice) | 134 | OVERALL | 84.22 | 88.24 | +4.02 [+2.31, +5.84] | yes (positive) |
| invoice (HEADLINE slice) | 134 | header accuracy | 98.97 | 99.81 | +0.84 [+0.28, +1.49] | yes (positive) |
| invoice (HEADLINE slice) | 134 | row F1 | 86.58 | 92.06 | +5.47 [+3.34, +8.03] | yes (positive) |
| invoice (HEADLINE slice) | 134 | documents fully correct | 50.00 | 57.46 | +7.46 [+1.49, +13.43] | yes (positive) |
| invoice (HEADLINE slice) | 134 | false-fill rate (lower is better) | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | no |
| waybill | 37 | OVERALL | 97.66 | 99.22 | +1.56 [+0.00, +3.90] | no |
| waybill | 37 | header accuracy | 99.10 | 99.70 | +0.60 [+0.00, +1.50] | no |
| waybill | 37 | row F1 | 100.00 | 100.00 | +0.00 [+0.00, +0.00] | no |
| waybill | 37 | documents fully correct | 91.89 | 97.30 | +5.41 [+0.00, +13.51] | no |
| waybill | 37 | false-fill rate (lower is better) | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | no |
| scanned | 66 | OVERALL | 84.07 | 88.14 | +4.07 [+1.21, +7.04] | yes (positive) |
| scanned | 66 | header accuracy | 98.71 | 99.63 | +0.92 [+0.18, +1.83] | yes (positive) |
| scanned | 66 | row F1 | 82.68 | 89.66 | +6.97 [+3.07, +11.06] | yes (positive) |
| scanned | 66 | documents fully correct | 57.58 | 62.12 | +4.55 [-4.55, +13.64] | no |
| scanned | 66 | false-fill rate (lower is better) | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | no |
| digital | 105 | OVERALL | 87.14 | 91.00 | +3.86 [+2.16, +5.72] | yes (positive) |
| digital | 105 | header accuracy | 99.19 | 99.88 | +0.70 [+0.12, +1.38] | yes (positive) |
| digital | 105 | row F1 | 88.67 | 93.34 | +4.67 [+2.42, +7.70] | yes (positive) |
| digital | 105 | documents fully correct | 60.00 | 68.57 | +8.57 [+2.86, +14.29] | yes (positive) |
| digital | 105 | false-fill rate (lower is better) | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | no |
| all docs | 171 | OVERALL | 86.05 | 89.95 | +3.91 [+2.37, +5.63] | yes (positive) |
| all docs | 171 | header accuracy | 99.00 | 99.79 | +0.78 [+0.35, +1.29] | yes (positive) |
| all docs | 171 | row F1 | 86.58 | 92.06 | +5.47 [+3.29, +7.96] | yes (positive) |
| all docs | 171 | documents fully correct | 59.06 | 66.08 | +7.02 [+2.34, +12.28] | yes (positive) |
| all docs | 171 | false-fill rate (lower is better) | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | no |

## Over-nulls and false fills (all docs)

Over-null = gold has a value, prediction empty. Header per field; row fields on the rows the scorer pairs (gold rows nobody matched are counted as missing rows). Fields with a zero count in every column are omitted.

| count | ZS + rules | FT + rules | ZS raw | FT raw | FT - ZS (rules arms) |
|---|---|---|---|---|---|
| header over-null: carrier | 0 | 0 | 4 | 0 | 0 |
| header over-null: hawb | 0 | 0 | 1 | 0 | 0 |
| header over-null: mawb | 0 | 0 | 10 | 0 | 0 |
| header over-nulls, total | 0 | 0 | 15 | 0 | 0 |
| row over-null (scorer-paired rows): customer_part_number | 18 | 1 | 18 | 5 | -17 |
| row over-null (scorer-paired rows): purchase_order | 0 | 0 | 0 | 1 | 0 |
| row over-null (scorer-paired rows): quantity | 0 | 0 | 0 | 0 | 0 |
| row over-null (scorer-paired rows): supplier_part_number | 0 | 0 | 0 | 0 | 0 |
| row over-nulls, total | 18 | 1 | 18 | 6 | -17 |
| OVER-NULL CELLS, header + row (verdict clause) | 18 | 1 | 33 | 6 | -17 |
| header false fills | 0 | 0 | 0 | 0 | 0 |
| row false fills (scorer-paired rows) | 0 | 0 | 0 | 5 | 0 |
| gold rows left unmatched (missing rows) | 154 | 120 | 154 | 120 | -34 |

## Rules effect per arm (production path, rules on vs off)

fixed / broken / neutral: cell diffs rules-off -> rules-on scored against gold (R1 carrier, R2 mawb / hawb, R3 cpn / po of scorer-paired rows; an R3 row the scorer cannot pair is neutral).

| arm | rule | eligible docs | touched docs | cells changed | fixed | broken | neutral | net |
|---|---|---|---|---|---|---|---|---|
| ZS | R1 | 37 | 4 | 4 | 4 | 0 | 0 | 4 |
| ZS | R2 | 37 | 10 | 11 | 11 | 0 | 0 | 11 |
| ZS | R3 | 134 | 0 | 0 | 0 | 0 | 0 | 0 |
| FT | R1 | 37 | 0 | 0 | 0 | 0 | 0 | 0 |
| FT | R2 | 37 | 0 | 0 | 0 | 0 | 0 | 0 |
| FT | R3 | 134 | 3 | 6 | 5 | 0 | 1 | 5 |

| arm | OVERALL rules off | OVERALL rules on | delta on - off (pts) [95% CI] | CI excludes 0 |
|---|---|---|---|---|
| ZS | 84.22 | 86.05 | +1.83 [+0.93, +2.85] | yes (positive) |
| FT | 89.82 | 89.95 | +0.13 [+0.00, +0.34] | no |

R3 on fold 0 (shapes learned excluding this fold's suppliers): ZS: R3 did NOT fire; FT: touched 3 docs (6 row edits). R3 is judged only on folds where it fires (GG decision); a fold where it does not fire says nothing about it.

Rule skips (reason counts): ZS none; FT none.

## Row-error cause table (taxonomy of `scripts/row_error_diagnosis.py`)

Same classification, rerun on each arm's outputs for the fold's docs: invoice rows by cause, cpn / po slot mix-ups on scorer-paired rows, waybill header over-nulls. 'rules' columns are the production-processed outputs, 'raw' the model outputs. Counts only.

| count | ZS + rules | FT + rules | ZS raw | FT raw |
|---|---|---|---|---|
| invoice docs diagnosed | 134 | 134 | 134 | 134 |
| invoice docs with a truncated page | 0 | 0 | 0 | 0 |
| unpaired rows linked one-to-one (side pair), total | 153 | 119 | 153 | 119 |
| gold rows with no link (row_missing side), total | 1 | 1 | 1 | 1 |
| predicted rows with no link (row_extra side), total | 1 | 0 | 1 | 0 |
| cause: spn_copies_other_slot | 0 | 0 | 0 | 0 |
| cause: column_shift | 1 | 0 | 1 | 0 |
| cause: spn_null | 0 | 0 | 0 | 0 |
| cause: spn_misread | 152 | 119 | 152 | 119 |
| cause: spn_other | 0 | 0 | 0 | 0 |
| cause: weak_link_qty | 0 | 0 | 0 | 0 |
| cause: weak_link_position | 0 | 0 | 0 | 0 |
| cause: truncated_page | 0 | 0 | 0 | 0 |
| cause: page_dropped | 0 | 0 | 0 | 0 |
| cause: merged | 0 | 0 | 0 | 0 |
| cause: page_short | 0 | 1 | 0 | 1 |
| cause: missing_other | 1 | 0 | 1 | 0 |
| cause: header_row | 0 | 0 | 0 | 0 |
| cause: split | 0 | 0 | 0 | 0 |
| cause: duplicate_boundary | 0 | 0 | 0 | 0 |
| cause: duplicate_other | 0 | 0 | 0 | 0 |
| cause: extra_other | 1 | 0 | 1 | 0 |
| slot (scorer-paired rows): cpn_in_po_slot | 0 | 0 | 0 | 5 |
| slot (scorer-paired rows): cpn_over_null | 18 | 1 | 18 | 0 |
| slot (scorer-paired rows): po_over_null | 0 | 0 | 0 | 1 |
| waybill docs | 37 | 37 | 37 | 37 |
| waybill carrier: over-null cells | 0 | 0 | 4 | 0 |
| waybill carrier: of which the model emitted a null | 0 | 0 | 4 | 0 |
| waybill carrier: of which the key is missing from the raw output | 0 | 0 | 0 | 0 |
| waybill carrier: of which the value sits in another raw header slot | 0 | 0 | 4 | 0 |
| waybill carrier: of which a unique OCR pattern match is correct | 0 | 0 | 0 | 0 |
| waybill hawb: over-null cells | 0 | 0 | 1 | 0 |
| waybill hawb: of which the model emitted a null | 0 | 0 | 1 | 0 |
| waybill hawb: of which the key is missing from the raw output | 0 | 0 | 0 | 0 |
| waybill hawb: of which the value sits in another raw header slot | 0 | 0 | 0 | 0 |
| waybill hawb: of which a unique OCR pattern match is correct | 0 | 0 | 1 | 0 |
| waybill mawb: over-null cells | 0 | 0 | 10 | 0 |
| waybill mawb: of which the model emitted a null | 0 | 0 | 10 | 0 |
| waybill mawb: of which the key is missing from the raw output | 0 | 0 | 0 | 0 |
| waybill mawb: of which the value sits in another raw header slot | 0 | 0 | 0 | 0 |
| waybill mawb: of which a unique OCR pattern match is correct | 0 | 0 | 10 | 0 |

## Limits

- One fold: the interim verdict and the spec clause are NOT the G4 decision.
- R3 shapes are learned from the other folds' gold; R3 may not fire on a fold, and is then not judged (it is judged only on folds where it fires).
- Row-error causes need the row page of each predicted row, replayed from the trace; where the replay does not give the final row count the pages are unknown (as in row_error_diagnosis).
- Waybill over-null facts use the model's raw page-1 output; R2 needs OCR at inference time.
- Over-null counts on rows use the scorer's pairing: a missing row is a missing row, not an over-null cell.
- Slices of one fold are small (see the docs column); wide CIs are expected.
