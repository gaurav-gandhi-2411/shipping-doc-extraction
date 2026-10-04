# Where the submitted system fails (native ZS + rules, v1.5): wrong row cells by class

**Provenance.** Command `uv run python scripts/failure_classes_native.py --run-dir $SHIPDOC_TMP_DIR\native_extract\zs\zeroshot500_qwen35_4b_img_only_native_4c17aa3 --out reports/failure_classes_native.md --json-out $SHIPDOC_TMP_DIR\report_out\failure_classes_native.json` (repo state `389dc5c+dirty`). System: native zero-shot + rules R1-R3, honest supplier-held-out R3 shapes, unmodified scorer; classifier: `scripts/row_error_diagnosis.py` (`analyse`, no oracle) plus the cell counts of `scripts/failure_classes_native.py`. **All numbers UNVERIFIED** here; the report registry records the second path. Aggregates and counts only; dev100 = the 500-doc diagnosis filtered to the dev docs.

Unit: a wrong ROW is a gold row that is not fully right; a wrong CELL is a row field (supplier part number, customer part number, purchase order, quantity) the scorer does not count right. Classes are mutually exclusive (definitions in the script docstring).

Totals: 639 wrong rows / 1069 wrong cells of 4930 gold rows on the 500 docs; 120 / 238 on dev100 (924 gold rows). Wrong header cells (not row cells): 7 on 500 docs, 2 on dev100. Waybills have no rows.

Cross-check with the official scorer, no classifier: wrong rows 639 on 500 docs and 120 on dev100 (gold rows - `rows_full`); in the scorer's own sense every cell of an unpaired row counts, which gives 1905 / 388 wrong cells (the tables here count only fields whose value differs, so a shifted row's empty-empty slots are not counted).

| class | wrong rows, 500 | wrong cells, 500 | docs, 500 | wrong rows, dev100 | wrong cells, dev100 | docs, dev100 |
|---|---|---|---|---|---|---|
| column_shift | 252 | 666 | 30 | 65 | 181 | 6 |
| spn_misread | 152 | 159 | 98 | 24 | 25 | 12 |
| other_field_in_paired_row | 110 | 110 | 34 | 26 | 26 | 9 |
| cpn_false_fill | 75 | 77 | 7 | 0 | 0 | 0 |
| spn_copies_other_slot | 17 | 23 | 2 | 0 | 0 | 0 |
| po_over_null | 20 | 20 | 2 | 0 | 0 | 0 |
| cpn_over_null | 12 | 12 | 5 | 4 | 4 | 2 |
| mixed | 1 | 2 | 1 | 1 | 2 | 1 |

Top 3 by wrong cells, 500 docs: column_shift, spn_misread, other_field_in_paired_row. Top 3 on dev100: column_shift, other_field_in_paired_row, spn_misread. The two lists agree (order may differ).

## Top 3 (500 docs) with the concrete fix and the evidence that exists

### 1. column_shift: 252 rows, 666 cells, 30 docs (dev100: 65 rows, 181 cells)

- cells by field (500 docs): {"customer_part_number": 169, "purchase_order": 212, "quantity": 33, "supplier_part_number": 252}
- scanned rows (500 docs): 101 of 252
- wrong-cell kinds (500 docs): {"customer_part_number:gold_null_filled": 67, "customer_part_number:mixed": 56, "customer_part_number:other_substitution": 8, "customer_part_number:pred_null": 38, "purchase_order:char_dropped_or_added": 1, "purchase_order:mixed": 58, "purchase_order:other_substitution": 3, "purchase_order:pred_null": 150, "quantity:qty_differs": 33, "supplier_part_number:glyph_confusion": 1, "supplier_part_number:mixed": 234, "supplier_part_number:other_substitution": 17}
- rows by supplier group, top 6 (500 docs): {"inv_g07": 136, "inv_g05": 66, "inv_g02": 49, "inv_g18": 1}
- fix: continuation-page column-header hint: give pages without a header row the column order read on page 1 (07n; built as a design, NOT run and cancelled for the deadline: future work)
- evidence: ORACLE ceiling only: +2.80 pts [+1.77, +3.96] if all 252 rows were fixed, +2.46 pts [+1.52, +3.56] for the 220 continuation-page rows (reports/hdrhint_estimate_native.md section 1; reports/row_errors_native.md X4). All 220 sit on header-less continuation pages whose page 1 has a header line (27 docs).

### 2. spn_misread: 152 rows, 159 cells, 98 docs (dev100: 24 rows, 25 cells)

- cells by field (500 docs): {"customer_part_number": 1, "purchase_order": 5, "quantity": 1, "supplier_part_number": 152}
- scanned rows (500 docs): 77 of 152
- wrong-cell kinds (500 docs): {"customer_part_number:pred_null": 1, "purchase_order:glyph_confusion": 5, "quantity:qty_differs": 1, "supplier_part_number:char_dropped_or_added": 17, "supplier_part_number:glyph_confusion": 119, "supplier_part_number:other_substitution": 16}
- rows by supplier group, top 6 (500 docs): {"inv_g01": 25, "inv_g10": 24, "inv_g11": 19, "inv_g07": 12, "inv_g09": 10, "inv_g13": 9}
- fix: glyph-level part-number reading via fine-tuning (LoRA); an OCR-snap merge rule was tried and REJECTED
- evidence: 119 of 152 are glyph confusions (reports/row_errors_native.md X1.3); rule R4 (snap to the nearest OCR span): 92 fixed, 140 broken, -1.30 pts [-2.04, -0.52] (same report, X4 candidate rules). Fine-tune, INTERIM fold 0 only (171 docs): 60 misread pairs against 69 for zero-shot (reports/row_errors_native_fold0_ft.md and _zs.md, X1.3); the pooled 3-fold result is pending, so no fine-tune gain is claimed.

### 3. other_field_in_paired_row: 110 rows, 110 cells, 34 docs (dev100: 26 rows, 26 cells)

- cells by field (500 docs): {"customer_part_number": 1, "purchase_order": 80, "quantity": 29}
- scanned rows (500 docs): 62 of 110
- wrong-cell kinds (500 docs): {"customer_part_number:other_substitution": 1, "purchase_order:char_dropped_or_added": 4, "purchase_order:glyph_confusion": 74, "purchase_order:other_substitution": 2, "quantity:qty_differs": 29}
- rows by supplier group, top 6 (500 docs): {"inv_g11": 47, "inv_g07": 28, "inv_g01": 19, "inv_g02": 9, "inv_g12": 2, "inv_g06": 2}
- fix: same remedy as the part-number misreads for the purchase-order values (glyph-level reading, fine-tune); the quantity differences have no diagnosed cause
- evidence: in rows whose part number matched, 74 of the 80 wrong purchase-order cells are glyph confusions and 29 quantity values differ (kinds line, reports/failure_classes_native.md); no artifact measures a fix for either, so none is claimed.

### 4. cpn_false_fill (outside the top 3): 75 rows, 77 cells, 7 docs (dev100: 0 rows, 0 cells)

- cells by field (500 docs): {"customer_part_number": 75, "purchase_order": 2}
- scanned rows (500 docs): 19 of 75
- wrong-cell kinds (500 docs): {"customer_part_number:gold_null_filled": 75, "purchase_order:char_dropped_or_added": 1, "purchase_order:mixed": 1}
- rows by supplier group, top 6 (500 docs): {"inv_g18": 67, "inv_g12": 6, "inv_g05": 2}
- fix: do not fill a customer-part slot the layout lacks: prompt or fine-tune target that leaves it null (untested)
- evidence: ORACLE ceiling: +0.85 pts [+0.26, +1.59] (reports/row_errors_native.md X4). 67 of the 75 rows sit in one supplier group (inv_g18, fold 1, whose fine-tune arm has not run); no artifact measures any fix, and the R3 rules do not touch false fills.

### 5. spn_copies_other_slot (outside the top 3): 17 rows, 23 cells, 2 docs (dev100: 0 rows, 0 cells)

- cells by field (500 docs): {"customer_part_number": 6, "supplier_part_number": 17}
- scanned rows (500 docs): 17 of 17
- wrong-cell kinds (500 docs): {"customer_part_number:gold_null_filled": 6, "supplier_part_number:mixed": 17}
- rows by supplier group, top 6 (500 docs): {"inv_g07": 11, "inv_g05": 6}
- fix: fine-tune target (single document pair; the part number is readable in the OCR text)
- evidence: ORACLE ceiling +0.14 pts [+0.00, +0.36] (reports/row_errors_native.md X4); 2 documents, mechanism unknown.

