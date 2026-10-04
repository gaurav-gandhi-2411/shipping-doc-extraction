# Row-error causes, native fold 0: FT+rules vs ZS+rules (171 held-out docs)

Aggregates and counts only. Per-arm detail: `row_errors_native_fold0_ft.md` and
`row_errors_native_fold0_zs.md` (their tables are valid; the "100 dev docs" prose in their X4 section
is stale script text and must be ignored).

Provenance: native FT OOF run `oof_native_fold0_4c17aa3` (adapter `ft_native_fold0_4c17aa3_bf16`) and
native ZS run `zeroshot500_qwen35_4b_img_only_native_4c17aa3`, restricted to the 171 fold-0 docs; both
arms through the production rules path with honest per-fold R3 shapes; method as in
`row_errors_native.md`; `scripts/row_error_diagnosis.py`; repo state `10dbbb8+dirty` (other work was
uncommitted in the tree). Both columns cover the same 1543 gold rows over 134 invoice docs.

| cause (rows) | ZS+rules | FT+rules |
|---|---|---|
| spn_misread | 69 | 60 |
| column_shift | 0 | 0 |
| spn_copies_other_slot | 0 | 0 |
| other causes | 0 | 0 |
| total unpaired | 69 | 60 |
| cpn_over_null slot rows | 11 | 1 |
| po_false_fill slot rows | 0 | 5 |

VERIFIED: identical counts to `g4_fold`'s own cause table (both drivers call the same taxonomy code,
so this checks the driver, not the taxonomy).

## Spec section 11.4

FT+rules column_shift on fold 0 is 0 (threshold 30). Fold 0 cannot settle the 07 header-hint gate in
either direction: the ZS arm also has 0 here, and `row_errors_native.md` places the 500-doc column
shifts (252 rows) in groups inv_g02, inv_g05 and inv_g07, none of which are in fold 0. The gate
should be read on the FT-native row-error table of the pooled folds, not on this one.
