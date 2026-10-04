# Fold 0: row false fills of the fine-tuned arm (aggregates and categories only)

- FT run: oof_native_fold0_4c17aa3
- ZS run (restricted to the fold): zeroshot500_qwen35_4b_img_only_native_4c17aa3
- R3 shapes: learned from 329 docs outside fold 0 (supplier-disjoint)
- script: scripts/false_fill_diag.py
- documents: 171

## 1. The cells (FT arm, raw)

| # | doc | field | doc type | scanned | group | pairing |
|---|---|---|---|---|---|---|
| 1 | doc1 | purchase_order | invoice | yes | inv_g03 | partial |
| 2 | doc2 | purchase_order | invoice | yes | inv_g03 | partial |
| 3 | doc2 | purchase_order | invoice | yes | inv_g03 | partial |
| 4 | doc2 | purchase_order | invoice | yes | inv_g03 | partial |
| 5 | doc3 | purchase_order | invoice | no | inv_g03 | partial |
| 6 | doc3 | purchase_order | invoice | no | inv_g03 | partial |
| 7 | doc3 | purchase_order | invoice | no | inv_g03 | partial |
| 8 | doc4 | purchase_order | invoice | no | inv_g03 | partial |
| 9 | doc4 | purchase_order | invoice | no | inv_g03 | partial |
| 10 | doc5 | purchase_order | invoice | no | inv_g04 | partial |
| 11 | doc5 | purchase_order | invoice | no | inv_g04 | partial |
| 12 | doc6 | purchase_order | invoice | yes | inv_g04 | partial |
| 13 | doc6 | purchase_order | invoice | yes | inv_g04 | partial |

13 cells; 6 distinct docs; 2 distinct supplier groups ({'inv_g03': 9, 'inv_g04': 4}); by field {'purchase_order': 13}; scanned 6 / digital 7.

## 2. Zero-shot arm on the same gold row and field, and the effect of the rules

| # | ZS raw | ZS+rules | FT+rules |
|---|---|---|---|
| 1 | null | null | value |
| 2 | null | null | null |
| 3 | null | null | null |
| 4 | null | null | null |
| 5 | null | null | null |
| 6 | null | null | null |
| 7 | null | null | null |
| 8 | null | null | null |
| 9 | null | null | null |
| 10 | null | null | value |
| 11 | null | null | value |
| 12 | null | null | value |
| 13 | null | null | value |

State: null = row paired and the cell empty; value = paired and non-empty;
unpaired = the scorer pairs no prediction row with that gold row.

## 3. Counts by the spec section 11 definitions (all docs of the fold)

| count | ZS-raw | FT-raw | ZS+rules | FT+rules |
|---|---|---|---|---|
| header_false_fill | 0 | 0 | 0 | 0 |
| row_false_fill | 0 | 13 | 0 | 5 |
| false_fill_total | 0 | 13 | 0 | 5 |
| header_over_null | 14 | 0 | 0 | 0 |
| row_over_null | 11 | 9 | 11 | 1 |
| over_null_total | 25 | 9 | 11 | 1 |
| rows_unmatched_gold | 69 | 60 | 69 | 60 |

False-fill cell set, raw -> after rules (same cell = same doc, gold row, field):

| arm | before | kept | cleared | new |
|---|---|---|---|---|
| ZS | 0 | 0 | 0 | 0 |
| FT | 13 | 5 | 8 | 0 |

Rules change the FT arm's false fills: 8 of 13 cleared, 0 new; FT row false fills 13 -> 5, FT over-nulls 9 -> 1.

## 4. Category of each FT value (decided on strings inside the script)

| # | category | equals gold other slot | also in pred other slot | in OCR | level |
|---|---|---|---|---|---|
| 1 | b | - | - | True | exact |
| 2 | a | customer_part_number | - | True | exact |
| 3 | a | customer_part_number | - | True | exact |
| 4 | a | customer_part_number | - | True | exact |
| 5 | a | customer_part_number | - | True | exact |
| 6 | a | customer_part_number | - | True | exact |
| 7 | a | customer_part_number | - | True | exact |
| 8 | a | customer_part_number | - | True | exact |
| 9 | a | customer_part_number | - | True | exact |
| 10 | b | - | - | True | exact |
| 11 | b | - | - | True | exact |
| 12 | b | - | - | True | exact |
| 13 | b | - | - | True | exact |

a (value equals the gold of another slot of the same row): 8; b (on the page, gold leaves the slot null): 5; c (other): 0.

## 5. What the scorer penalises, and which metric C2 uses

- `false_fill_rate` (score.py `score_doc` / `aggregate`) counts only gold-null HEADER fields that the prediction fills, divided by the gold-null header fields. Row cells never enter it, so it is 0.00 for both arms in `oof_compare.md` while row false fills are non-zero.
- Row false fills are still penalised by the scorer, through other metrics: `same` returns False for (gold empty, prediction non-empty), so `row_ok` fails and the row cannot be a fully correct row in pass 1 of `match_rows`; it can only pair in pass 2 (by supplier_part_number, the `partial` pairing above), where it loses that field's credit in `line_item_field_accuracy`. The row is then not counted in row F1's true positives and the document is not `exact` (documents_fully_correct).
- The pre-registered C2 (spec section 11 item 1) is `shipdoc.oof.over_null_counts(...)['false_fill_total']` = header false fills + row false fills on scorer-paired rows, so it includes these cells; it is not `false_fill_rate`.


## 6. Executor annotation: per fill, what R1-R3 do (production path, honest fold-0 R3 shapes)

Added by hand after the script output above (rows are the numbered cells of section 1; no ids). R1 and R2 are waybill-only rules and touched 0 FT docs on this fold (`reports/g4_fold0_native.md`, rules table), so every change below is R3 (cpn <-> po swap by format shape). All numbers UNVERIFIED unless marked; the 13 -> 5 count is VERIFIED by a second path (section 3 here, `reports/g4_fold0_native.md` over-null table, and `over_null_counts` rerun on the emitted post-rule predictions, all 13 raw / 5 with rules).

| # | field | supplier group | category | scan | R1-R3 effect |
|---|---|---|---|---|---|
| 1 | purchase_order | inv_g03 | b | scanned | leave (still a value) |
| 2 | purchase_order | inv_g03 | a | scanned | R3 clears |
| 3 | purchase_order | inv_g03 | a | scanned | R3 clears |
| 4 | purchase_order | inv_g03 | a | scanned | R3 clears |
| 5 | purchase_order | inv_g03 | a | digital | R3 clears |
| 6 | purchase_order | inv_g03 | a | digital | R3 clears |
| 7 | purchase_order | inv_g03 | a | digital | R3 clears |
| 8 | purchase_order | inv_g03 | a | digital | R3 clears |
| 9 | purchase_order | inv_g03 | a | digital | R3 clears |
| 10 | purchase_order | inv_g04 | b | digital | leave (still a value) |
| 11 | purchase_order | inv_g04 | b | digital | leave (still a value) |
| 12 | purchase_order | inv_g04 | b | scanned | leave (still a value) |
| 13 | purchase_order | inv_g04 | b | scanned | leave (still a value) |

Totals: R3 clears 8 (3 scanned, 5 digital; all category a, all inv_g03), breaks 0, creates 0 new false fills; 5 remain (category b: 1 in inv_g03, 4 in inv_g04; 3 scanned, 2 digital). **With rules: FT false_fill_total = 5, ZS+rules = 0: this is the C2 input on fold 0 (FT 5 > ZS 0).**

### Against the 1260-token fold 0 (`reports/fold0_false_fills.md`, a different input resolution)

| | 1260 fold 0 (oof_fold0_42b812b) | native fold 0 (oof_native_fold0_4c17aa3) |
|---|---|---|
| FT raw row false fills | 5 (2 docs, 1 group inv_g03; all category a) | 13 (6 docs, 2 groups: inv_g03 9, inv_g04 4; a 8, b 5) |
| FT + rules false fills | 0 (R3 cleared 5 of 5) | 5 (R3 cleared 8 of 13; the 5 left are category b) |
| ZS + rules false fills | 0 | 0 |
| C2 on the fold (FT <= ZS) | holds (0 <= 0) | FAILS on this fold (5 > 0) |

The two inv_g03 documents of the 1260 list recur in the native list with the same cell counts (3 and 2), all cleared by R3 in both. What is new at native resolution is category b (value on the page, gold leaves the slot null): 5 cells, none of them touched by R3. The mechanism of category b is not diagnosed here (UNVERIFIED).
