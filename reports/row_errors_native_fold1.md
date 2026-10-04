# Row-error causes, native fold 1: FT+rules vs ZS+rules (165 held-out docs)

Aggregates and counts only. Per-arm detail: `row_errors_native_fold1_ft.md` and `row_errors_native_fold1_zs.md` (their tables are valid; the "100 dev docs" prose in their X4 section is stale script text and must be ignored, as for fold 0).

Provenance: native FT OOF run `oof_native_fold1_4c17aa3` (adapter `ft_native_fold1_4c17aa3_bf16`) and native ZS run `zeroshot500_qwen35_4b_img_only_native_4c17aa3`, restricted to the 165 fold-1 docs; both arms through the production rules path with honest per-fold R3 shapes (learned from the 335 docs outside fold 1, supplier-disjoint); scripts `$SHIPDOC_TMP_DIR\native_fold1\make_rowerr_f1.py` then `scripts/row_error_diagnosis.py` (`--split train --gold-dir $SHIPDOC_TMP_DIR\native_fold1\rowerr\gold165 --meta ...\meta165.json`); repo state in the per-arm files `c5c1f50+dirty` (other work uncommitted in the tree). Both columns cover the same 1689 gold rows over 134 invoice docs.

| cause (rows) | ZS+rules | FT+rules |
|---|---|---|
| spn_misread | 30 | 17 |
| column_shift | 67 | 0 |
| spn_copies_other_slot | 6 | 0 |
| other causes | 0 | 0 |
| total unpaired | 103 | 17 |
| slot mismatches, paired rows: cpn_false_fill / mixed / po_over_null | 75 / 1 / 20 | 0 / 0 / 1 |
| slot mismatches (FT raw): po_in_cpn_slot | 212 (ZS raw) | 7 (FT raw) |

(The slot rows are the g4_fold X2 kinds; in terms of the C2/C3 counts: false fills = cpn_false_fill + mixed = 76 / 0, over-nulls = po_over_null + mixed = 21 / 1.)

VERIFIED: identical counts to `g4_fold`'s own cause table in `g4_fold1_native.md` (both drivers call the same taxonomy code, so this checks the driver, not the taxonomy). The unpaired totals equal `over_null_counts` `rows_unmatched_gold` (103 / 17), recomputed by a second path.

## column_shift versus the spec section 11 item 4 threshold (30)

Item 4: the header-hint A/B (07) runs only if the fine-tuned row-error cause table still shows at least 30 column-shift rows.
- FT+rules column_shift on fold 1: **0** (fold 0: 0). Partial FT tally folds 0+1: 0. Threshold 30 is for the FT table over the whole evidence; folds 0 and 1 contribute 0, so fold 2 alone would need 30 or more for the trigger to fire (arithmetic; fold 2 not run).
- ZS+rules column_shift on fold 1: 67 rows in 8 docs, 66 of them in one supplier group (inv_g05; on its continuation pages with no table-header row, 73 of 85 gold rows are unpaired and 66 of those are column_shift, per the controlled table in the ZS file) and 1 in inv_g18. FT removes all of them (67 -> 0).
- ZS+rules: the 103 unpaired rows are concentrated in inv_g05 (81 of 287 gold rows); FT+rules leaves 17 unpaired, all `spn_misread`.

Note for 07's trigger: on this evidence the trigger measured on FT output is not met (0 of the required 30 on two folds).
