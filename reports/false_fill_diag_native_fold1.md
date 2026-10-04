# Fold 1: row false fills of the fine-tuned arm (aggregates and categories only)

- FT run: oof_native_fold1_4c17aa3
- ZS run (restricted to the fold): zeroshot500_qwen35_4b_img_only_native_4c17aa3
- R3 shapes: learned from 335 docs outside fold 1 (supplier-disjoint)
- script: scripts/false_fill_diag.py
- documents: 165

## 1. The cells (FT arm, raw)

| # | doc | field | doc type | scanned | group | pairing |
|---|---|---|---|---|---|---|
| 1 | doc1 | customer_part_number | invoice | yes | inv_g05 | partial |
| 2 | doc1 | customer_part_number | invoice | yes | inv_g05 | partial |
| 3 | doc1 | customer_part_number | invoice | yes | inv_g05 | partial |
| 4 | doc1 | customer_part_number | invoice | yes | inv_g05 | partial |
| 5 | doc1 | customer_part_number | invoice | yes | inv_g05 | partial |
| 6 | doc1 | customer_part_number | invoice | yes | inv_g05 | partial |
| 7 | doc1 | customer_part_number | invoice | yes | inv_g05 | partial |

7 cells; 1 distinct docs; 1 distinct supplier groups ({'inv_g05': 7}); by field {'customer_part_number': 7}; scanned 7 / digital 0.

## 2. Zero-shot arm on the same gold row and field, and the effect of the rules

| # | ZS raw | ZS+rules | FT+rules |
|---|---|---|---|
| 1 | unpaired | unpaired | null |
| 2 | unpaired | unpaired | null |
| 3 | unpaired | unpaired | null |
| 4 | unpaired | unpaired | null |
| 5 | unpaired | unpaired | null |
| 6 | unpaired | unpaired | null |
| 7 | unpaired | unpaired | null |

State: null = row paired and the cell empty; value = paired and non-empty;
unpaired = the scorer pairs no prediction row with that gold row.

## 3. Counts by the spec section 11 definitions (all docs of the fold)

| count | ZS-raw | FT-raw | ZS+rules | FT+rules |
|---|---|---|---|---|
| header_false_fill | 0 | 0 | 0 | 0 |
| row_false_fill | 288 | 7 | 76 | 0 |
| false_fill_total | 288 | 7 | 76 | 0 |
| header_over_null | 13 | 0 | 0 | 0 |
| row_over_null | 233 | 8 | 21 | 1 |
| over_null_total | 246 | 8 | 21 | 1 |
| rows_unmatched_gold | 103 | 17 | 103 | 17 |

False-fill cell set, raw -> after rules (same cell = same doc, gold row, field):

| arm | before | kept | cleared | new |
|---|---|---|---|---|
| ZS | 288 | 76 | 212 | 0 |
| FT | 7 | 0 | 7 | 0 |

Rules change the FT arm's false fills: 7 of 7 cleared, 0 new; FT row false fills 7 -> 0, FT over-nulls 8 -> 1.

## 4. Category of each FT value (decided on strings inside the script)

| # | category | equals gold other slot | also in pred other slot | in OCR | level |
|---|---|---|---|---|---|
| 1 | a | purchase_order | - | True | exact |
| 2 | a | purchase_order | - | True | exact |
| 3 | a | purchase_order | - | True | exact |
| 4 | a | purchase_order | - | True | exact |
| 5 | a | purchase_order | - | True | exact |
| 6 | a | purchase_order | - | True | exact |
| 7 | a | purchase_order | - | True | exact |

a (value equals the gold of another slot of the same row): 7; b (on the page, gold leaves the slot null): 0; c (other): 0.

## 5. What the scorer penalises, and which metric C2 uses

- `false_fill_rate` (score.py `score_doc` / `aggregate`) counts only gold-null HEADER fields that the prediction fills, divided by the gold-null header fields. Row cells never enter it, so it is 0.00 for both arms in `oof_compare.md` while row false fills are non-zero.
- Row false fills are still penalised by the scorer, through other metrics: `same` returns False for (gold empty, prediction non-empty), so `row_ok` fails and the row cannot be a fully correct row in pass 1 of `match_rows`; it can only pair in pass 2 (by supplier_part_number, the `partial` pairing above), where it loses that field's credit in `line_item_field_accuracy`. The row is then not counted in row F1's true positives and the document is not `exact` (documents_fully_correct).
- The pre-registered C2 (spec section 11 item 1) is `shipdoc.oof.over_null_counts(...)['false_fill_total']` = header false fills + row false fills on scorer-paired rows, so it includes these cells; it is not `false_fill_rate`.

## 6. Added after generation (counts only; scratch `$SHIPDOC_TMP_DIR\native_fold1\verify_fold1.py`)

Document ids were replaced by the ordinal `doc1` in sections 1-2 after the script ran (the script prints the real id; the id is not kept in this file). Everything below was computed by a second path (`over_null_counts` per doc on the emitted post-rule predictions of each arm, grouped by `meta/supplier_groups.json`), and its totals equal section 3 (VERIFIED).

The 76 ZS+rules row false fills (R3 had already cleared 212 of the 288 raw ones), by supplier group: inv_g18 68 (6 docs), inv_g12 6 (1 doc), inv_g05 2 (1 doc); 0 in inv_g06, inv_g13, inv_g16 and the 3 waybill groups. Field: all are scorer-paired row cells (`header_false_fill` 0); the row-error cause table (`row_errors_native_fold1.md`) types them as cpn_false_fill 75 + mixed 1 (a customer_part_number filled where the gold is empty). The 21 ZS+rules over-nulls are all purchase_order cells in inv_g18 (3 docs).

The 7 FT raw false fills: 1 doc, 1 supplier group (inv_g05), field customer_part_number, scanned, category a (the value equals the gold purchase_order of the same row; the purchase_order cell is empty in the raw output = 7 of the 8 FT raw over-nulls, the other 1 is inv_g18). R3 clears 7 of 7 with 0 broken and 0 new: FT+rules false fills 0, FT+rules over-nulls 1 (inv_g18, purchase_order). All 7 are in the same document, so there are not 7 independent events.

