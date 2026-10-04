# Review flags on dev100: auto-accept precision and error recall (native ZS + rules)

**Provenance.** Command `uv run python scripts/flags_dev_pr.py --calibration-dir $SHIPDOC_TMP_DIR\calibration_native --out reports/flags_dev100_native.md --json-out $SHIPDOC_TMP_DIR\report_out\flags_dev100_native.json` (repo state `389dc5c+dirty`; doc-level percentile bootstrap 2000 resamples, seed 42). Inputs: `oof_fields_v2.csv` (sha256 `28c039f9ae90c08f`, equals the sha recorded in `meta/calibrator_zs_native.json` source: True), `oof_docs_v2.csv`, frozen taus of `meta/calibrator_zs_native.json` (sha256 `2148c84a5b1d61fd`, run config hash `e4b84ec2809625d5`). **All numbers UNVERIFIED** here; the report registry records the second path. Aggregates and counts only.

## Operating point and protocol

Production thresholds: field target 0.98 (one tau per field type, lowest tau whose accepted set has >= 98% precision, `confidence.select_tau`), document target 0.98. The production calibrator was fitted on all 500 docs, so its probabilities on dev are in-sample. The PRIMARY table therefore uses the cross-fitted OUT-OF-FOLD probabilities of calibrate_v2 (supplier-fold cross-fit, `reports/calibration_v2_native.md`) with NESTED taus (fold k's tau chosen on the other two folds, applied to fold k): the dev docs never influence their own probabilities or thresholds. Population: EMITTED (non-null) fields. CIs are conditional on the thresholds. Frozen production taus: {"customer_part_number": 0.9996099555247, "header": 0.5131490608074835, "purchase_order": 0.9999993560311028, "quantity": null, "supplier_part_number": 0.9179919997981086} (null = nothing of that type is auto-accepted).

## dev100, cross-fitted (nested supplier-fold tau): PRIMARY

| field type | emitted fields | auto-accepted | auto-accept precision % [95% CI] | error recall % [95% CI] | review rate % [95% CI] | wrong fields |
|---|---|---|---|---|---|---|
| header | 789 | 789 (100.0%) | 99.7 [99.4, 100.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 2 |
| supplier_part_number | 924 | 787 (85.2%) | 98.1 [94.3, 99.9] | 83.1 [53.6, 99.1] | 14.8 [9.7, 20.8] | 89 |
| customer_part_number | 469 | 0 (0.0%) | n/a (none accepted) | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 72 |
| purchase_order | 536 | 56 (10.4%) | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 89.6 [78.1, 98.2] | 68 |
| quantity | 924 | 0 (0.0%) | n/a (none accepted) | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 89 |
| all fields (per-type tau) | 3642 | 1632 (44.8%) | 99.0 [97.3, 99.9] | 94.7 [86.3, 99.4] | 55.2 [49.6, 59.6] | 320 |

## dev100, OOF probabilities vs the frozen production tau (tau in-sample)

| field type | emitted fields | auto-accepted | auto-accept precision % [95% CI] | error recall % [95% CI] | review rate % [95% CI] | wrong fields |
|---|---|---|---|---|---|---|
| header | 789 | 789 (100.0%) | 99.7 [99.4, 100.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 2 |
| supplier_part_number | 924 | 780 (84.4%) | 98.2 [94.7, 99.9] | 84.3 [57.1, 99.1] | 15.6 [10.6, 21.4] | 89 |
| customer_part_number | 469 | 0 (0.0%) | n/a (none accepted) | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 72 |
| purchase_order | 536 | 19 (3.5%) | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 96.5 [89.7, 100.0] | 68 |
| quantity | 924 | 0 (0.0%) | n/a (none accepted) | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 89 |
| all fields (per-type tau) | 3642 | 1588 (43.6%) | 99.0 [97.5, 99.9] | 95.0 [87.3, 99.4] | 56.4 [51.1, 60.5] | 320 |

## pooled 500 docs, nested: SECONDARY

| field type | emitted fields | auto-accepted | auto-accept precision % [95% CI] | error recall % [95% CI] | review rate % [95% CI] | wrong fields |
|---|---|---|---|---|---|---|
| header | 3908 | 3907 (100.0%) | 99.8 [99.7, 99.9] | 14.3 [0.0, 50.0] | 0.0 [0.0, 0.1] | 7 |
| supplier_part_number | 4930 | 4219 (85.6%) | 98.1 [97.0, 99.0] | 80.5 [70.7, 89.1] | 14.4 [12.3, 16.6] | 421 |
| customer_part_number | 2032 | 74 (3.6%) | 90.5 [80.9, 97.6] | 98.3 [95.6, 99.8] | 96.4 [93.2, 99.1] | 403 |
| purchase_order | 2804 | 513 (18.3%) | 96.9 [95.3, 98.3] | 94.5 [90.5, 97.4] | 81.7 [75.6, 87.3] | 292 |
| quantity | 4930 | 2 (0.0%) | 0.0 [0.0, 0.0] | 99.6 [98.7, 100.0] | 100.0 [99.9, 100.0] | 450 |
| all fields (per-type tau) | 18604 | 8715 (46.8%) | 98.7 [98.1, 99.2] | 92.8 [89.6, 95.4] | 53.2 [51.1, 55.1] | 1573 |

## Wrong cells the flag does not see

Not in the tables above (the flag only judges emitted fields): over-nulls (gold has a value, the prediction is null) and gold rows that no predicted row pairs with.

| field type | over-null cells, dev100 | over-null cells, 500 docs |
|---|---|---|
| header | 0 | 0 |
| supplier_part_number | 0 | 0 |
| customer_part_number | 22 | 107 |
| purchase_order | 48 | 232 |
| quantity | 0 | 0 |

## Document level (target 0.98)

Fully correct documents: 73 of 100 on dev, 344 of 500 on all. Auto-accepted documents on dev: nested tau 7 accepted, 6 of them fully correct; frozen production tau 1 accepted, 1 fully correct. On all 500 docs: nested 38 accepted, 28 correct; frozen 1, 1 correct. The pooled nested document precision is below the 98% target (reports/calibration_v2_native.md section (c): NOT ATTAINABLE), so the document-level flag is not a usable auto-accept; the shipped test flags accept 0 of 200 documents (`reports/v1_5_submission.md`).

## Cross-check against calibrate_v2's own output

`tau_field_types.csv` (nested, target 0.98, slices dev and all): 138 values compared, max absolute difference 0.00e+00.
