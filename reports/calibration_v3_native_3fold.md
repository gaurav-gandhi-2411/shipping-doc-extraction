# Calibration v3 prep: fine-tuned vs zero-shot agreement

> **UNVERIFIED: aggregates computed by scripts/calibrate_v3.py (3-fold cross-fitted); a verifier has not recomputed them.**

Provenance: OOF run(s) `oof_native_fold0_4c17aa3, oof_native_fold1_4c17aa3, oof_native_fold2_4c17aa3`, zero-shot run `zeroshot500_qwen35_4b_img_only_native_4c17aa3`, repo HEAD `0c16e3a` (confidence_v3.py sha256 `4adc9ced79f8`, calibrate_v3.py sha256 `64ae0916e81a`), command `uv run python scripts/calibrate_v3.py --folds 0 1 2 --oof-run $SHIPDOC_TMP_DIR\native_extract\oof0\oof_native_fold0_4c17aa3 --oof-run $SHIPDOC_RUNS_DIR\oof_native_fold1_4c17aa3 --oof-run $SHIPDOC_RUNS_DIR\oof_native_fold2_4c17aa3 --zs-run-dir $SHIPDOC_TMP_DIR\native_extract\zs\zeroshot500_qwen35_4b_img_only_native_4c17aa3 --require-logprobs --out-dir $SHIPDOC_TMP_DIR\pooled\calv3_3fold --report-md reports/calibration_v3_native_3fold.md`, wall time 652 s. Bootstrap: document-level percentile, 2000 resamples, seed 42. Test data not used. Protocol: docstring of `scripts/calibrate_v3.py`.

Documents 500; R3 shapes per-fold honest, equal to the all-gold set per fold: {0: True, 1: True, 2: True}. Rules summary FT `{"n_docs": 500, "switches": {"r1": true, "r2": true, "r3": true}, "eligible_docs": {"R1": 100, "R2": 100, "R3": 400}, "touched_docs": {"R1": 0, "R2": 0, "R3": 7}, "changes": {"R1": 0, "R2": 0, "R3": 34}, "skipped": {}}`; ZS `{"n_docs": 500, "switches": {"r1": true, "r2": true, "r3": true}, "eligible_docs": {"R1": 100, "R2": 100, "R3": 400}, "touched_docs": {"R1": 7, "R2": 29, "R3": 25}, "changes": {"R1": 7, "R2": 41, "R3": 287}, "skipped": {}}`.


## View: judging the FINE-TUNED arm

Structure **pooled**, kind **lr** (chosen once on the v2 design of the fine-tuned view; emitted OOF log-loss {"pooled/lr": 0.1818, "pooled/gbm": 0.2158, "per_type/lr": 0.2087, "per_type/gbm": 0.2435}).

| group | fields | wrong | docs | base acc | AUROC v2 | AUROC v2+agree | AUROC agree only | AUROC flag | lift (v2+agree - v2) | note |
|---|---|---|---|---|---|---|---|---|---|---|
| header_all | 3908 | 1 | 500 | 1.000 | 0.994 [0.992, 0.996] | 0.997 [0.995, 0.998] | 0.931 [0.925, 0.937] | 0.999 [0.998, 1.000] | 0.003 [0.001, 0.005] | n_wrong<5 |
| row.supplier_part_number | 4929 | 292 | 400 | 0.941 | 0.827 [0.782, 0.871] | 0.781 [0.708, 0.860] | 0.371 [0.299, 0.462] | 0.625 [0.561, 0.690] | -0.046 [-0.114, 0.018] |  |
| row.customer_part_number | 1835 | 162 | 151 | 0.912 | 0.600 [0.488, 0.687] | 0.644 [0.542, 0.722] | 0.521 [0.448, 0.605] | 0.503 [0.486, 0.523] | 0.044 [0.013, 0.078] |  |
| row.purchase_order | 2923 | 215 | 232 | 0.926 | 0.725 [0.645, 0.790] | 0.609 [0.497, 0.728] | 0.421 [0.313, 0.552] | 0.654 [0.565, 0.738] | -0.116 [-0.235, 0.000] |  |
| row.quantity | 4930 | 294 | 400 | 0.940 | 0.709 [0.648, 0.759] | 0.640 [0.541, 0.737] | 0.354 [0.275, 0.449] | 0.556 [0.512, 0.606] | -0.069 [-0.176, 0.030] |  |

Nested per-field-type auto-accept (precision % / coverage %, 95% CI):

| variant | type | target | fields | precision | coverage |
|---|---|---|---|---|---|
| v2 | header | 95% | 3908 | 100.0 [99.9, 100.0] | 99.8 [99.7, 99.9] |
| v2 | header | 98% | 3908 | 100.0 [99.9, 100.0] | 99.8 [99.7, 99.9] |
| v2 | header | 99% | 3908 | 100.0 [99.9, 100.0] | 99.8 [99.7, 99.9] |
| v2 | supplier_part_number | 95% | 4929 | 94.5 [92.4, 96.5] | 86.8 [84.0, 89.5] |
| v2 | supplier_part_number | 98% | 4929 | 94.5 [92.3, 96.6] | 80.4 [76.6, 83.9] |
| v2 | supplier_part_number | 99% | 4929 | 95.2 [92.9, 97.2] | 70.1 [65.8, 74.0] |
| v2 | customer_part_number | 95% | 1835 | 94.7 [90.4, 97.9] | 17.3 [12.0, 23.2] |
| v2 | customer_part_number | 98% | 1835 | 95.1 [90.9, 98.3] | 16.7 [11.4, 22.5] |
| v2 | customer_part_number | 99% | 1835 | 95.1 [90.9, 98.3] | 16.7 [11.4, 22.5] |
| v2 | purchase_order | 95% | 2923 | 92.0 [89.0, 94.5] | 85.5 [81.1, 89.4] |
| v2 | purchase_order | 98% | 2923 | 96.2 [93.3, 98.1] | 32.1 [26.2, 38.3] |
| v2 | purchase_order | 99% | 2923 | 98.2 [97.1, 99.1] | 22.7 [17.1, 28.9] |
| v2 | quantity | 95% | 4930 | 93.6 [91.5, 95.5] | 86.5 [83.2, 89.5] |
| v2 | quantity | 98% | 4930 | 97.3 [95.8, 98.4] | 49.7 [45.1, 54.5] |
| v2 | quantity | 99% | 4930 | 98.7 [97.2, 99.8] | 13.7 [10.1, 17.4] |
| v2_agree | header | 95% | 3908 | 100.0 [99.9, 100.0] | 99.7 [99.5, 99.8] |
| v2_agree | header | 98% | 3908 | 100.0 [99.9, 100.0] | 99.7 [99.5, 99.8] |
| v2_agree | header | 99% | 3908 | 100.0 [99.9, 100.0] | 99.7 [99.5, 99.8] |
| v2_agree | supplier_part_number | 95% | 4929 | 94.6 [92.5, 96.4] | 91.1 [88.7, 93.2] |
| v2_agree | supplier_part_number | 98% | 4929 | 93.2 [90.4, 95.8] | 60.9 [55.7, 66.0] |
| v2_agree | supplier_part_number | 99% | 4929 | 94.4 [91.7, 96.9] | 53.3 [48.4, 58.3] |
| v2_agree | customer_part_number | 95% | 1835 | 93.9 [91.5, 95.9] | 36.4 [28.7, 44.8] |
| v2_agree | customer_part_number | 98% | 1835 | 94.8 [91.2, 97.6] | 21.0 [15.1, 27.3] |
| v2_agree | customer_part_number | 99% | 1835 | 94.8 [91.2, 97.6] | 21.0 [15.1, 27.3] |
| v2_agree | purchase_order | 95% | 2923 | 88.8 [84.7, 92.4] | 57.5 [49.5, 65.2] |
| v2_agree | purchase_order | 98% | 2923 | 71.9 [51.8, 90.2] | 7.2 [4.4, 10.3] |
| v2_agree | purchase_order | 99% | 2923 | 71.1 [25.0, 100.0] | 1.5 [0.4, 3.0] |
| v2_agree | quantity | 95% | 4930 | 91.5 [88.7, 94.1] | 62.4 [56.4, 68.0] |
| v2_agree | quantity | 98% | 4930 | 89.6 [81.8, 96.2] | 13.7 [10.1, 17.8] |
| v2_agree | quantity | 99% | 4930 | 90.6 [76.6, 98.7] | 4.1 [2.2, 6.6] |

## View: judging the ZERO-SHOT arm

Structure **pooled**, kind **lr** (chosen once on the v2 design of the fine-tuned view; choice reused).

| group | fields | wrong | docs | base acc | AUROC v2 | AUROC v2+agree | AUROC agree only | AUROC flag | lift (v2+agree - v2) | note |
|---|---|---|---|---|---|---|---|---|---|---|
| header_all | 3908 | 7 | 500 | 0.998 | 0.969 [0.896, 1.000] | 1.000 [0.999, 1.000] | 1.000 [0.999, 1.000] | 1.000 [1.000, 1.000] | 0.030 [0.000, 0.103] |  |
| row.supplier_part_number | 4930 | 421 | 400 | 0.915 | 0.932 [0.907, 0.954] | 0.923 [0.892, 0.953] | 0.668 [0.574, 0.768] | 0.748 [0.683, 0.811] | -0.009 [-0.042, 0.024] |  |
| row.customer_part_number | 2032 | 403 | 167 | 0.802 | 0.718 [0.594, 0.824] | 0.834 [0.788, 0.873] | 0.797 [0.711, 0.872] | 0.797 [0.726, 0.859] | 0.116 [0.009, 0.237] |  |
| row.purchase_order | 2804 | 292 | 229 | 0.896 | 0.813 [0.760, 0.859] | 0.850 [0.801, 0.889] | 0.727 [0.625, 0.818] | 0.729 [0.660, 0.796] | 0.037 [0.001, 0.082] |  |
| row.quantity | 4930 | 450 | 400 | 0.909 | 0.780 [0.719, 0.828] | 0.825 [0.773, 0.863] | 0.659 [0.570, 0.748] | 0.570 [0.527, 0.614] | 0.045 [0.005, 0.086] |  |

Nested per-field-type auto-accept (precision % / coverage %, 95% CI):

| variant | type | target | fields | precision | coverage |
|---|---|---|---|---|---|
| v2 | header | 95% | 3908 | 99.8 [99.7, 99.9] | 100.0 [99.9, 100.0] |
| v2 | header | 98% | 3908 | 99.8 [99.7, 99.9] | 100.0 [99.9, 100.0] |
| v2 | header | 99% | 3908 | 99.8 [99.7, 99.9] | 100.0 [99.9, 100.0] |
| v2 | supplier_part_number | 95% | 4930 | 92.8 [90.7, 94.7] | 97.7 [97.1, 98.3] |
| v2 | supplier_part_number | 98% | 4930 | 98.1 [97.1, 99.0] | 84.2 [82.0, 86.4] |
| v2 | supplier_part_number | 99% | 4930 | 99.0 [98.5, 99.4] | 72.8 [69.7, 75.8] |
| v2 | customer_part_number | 95% | 2032 | 43.4 [17.3, 71.3] | 7.8 [3.9, 12.3] |
| v2 | customer_part_number | 98% | 2032 | 90.4 [80.7, 97.5] | 3.6 [0.8, 6.8] |
| v2 | customer_part_number | 99% | 2032 | 90.4 [80.7, 97.5] | 3.6 [0.8, 6.8] |
| v2 | purchase_order | 95% | 2804 | 90.3 [87.3, 93.1] | 87.4 [83.4, 91.0] |
| v2 | purchase_order | 98% | 2804 | 97.1 [95.6, 98.4] | 18.4 [12.8, 24.5] |
| v2 | purchase_order | 99% | 2804 | 97.1 [95.6, 98.4] | 18.4 [12.8, 24.5] |
| v2 | quantity | 95% | 4930 | 92.8 [90.7, 94.6] | 84.3 [81.0, 87.2] |
| v2 | quantity | 98% | 4930 | 96.8 [95.0, 98.4] | 13.3 [9.9, 17.2] |
| v2 | quantity | 99% | 4930 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.1] |
| v2_agree | header | 95% | 3908 | 99.8 [99.7, 99.9] | 100.0 [99.9, 100.0] |
| v2_agree | header | 98% | 3908 | 99.8 [99.7, 99.9] | 100.0 [99.9, 100.0] |
| v2_agree | header | 99% | 3908 | 99.8 [99.7, 99.9] | 100.0 [99.9, 100.0] |
| v2_agree | supplier_part_number | 95% | 4930 | 95.2 [93.4, 96.7] | 95.1 [93.6, 96.5] |
| v2_agree | supplier_part_number | 98% | 4930 | 96.2 [94.3, 97.9] | 79.7 [76.2, 83.0] |
| v2_agree | supplier_part_number | 99% | 4930 | 97.0 [95.2, 98.6] | 75.2 [71.6, 78.6] |
| v2_agree | customer_part_number | 95% | 2032 | 94.0 [91.6, 96.0] | 33.5 [25.4, 42.0] |
| v2_agree | customer_part_number | 98% | 2032 | 95.9 [92.4, 98.7] | 12.0 [7.6, 17.0] |
| v2_agree | customer_part_number | 99% | 2032 | 95.9 [92.4, 98.7] | 12.0 [7.6, 17.0] |
| v2_agree | purchase_order | 95% | 2804 | 93.1 [90.8, 95.2] | 93.4 [91.0, 95.5] |
| v2_agree | purchase_order | 98% | 2804 | 97.1 [95.9, 98.2] | 36.3 [29.3, 43.6] |
| v2_agree | purchase_order | 99% | 2804 | 97.5 [96.2, 98.5] | 26.8 [20.4, 33.9] |
| v2_agree | quantity | 95% | 4930 | 94.0 [92.3, 95.6] | 90.9 [88.6, 93.1] |
| v2_agree | quantity | 98% | 4930 | 97.0 [94.2, 98.9] | 8.0 [5.2, 11.3] |
| v2_agree | quantity | 99% | 4930 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.1] |

## Caveats

- One fold holds few suppliers: the supplier-grouped cross-fit inside it trains each calibrator on a handful of suppliers and the group / document counts above are small.
- CIs resample documents but condition on the cross-fit and on the LR-vs-GBM choice.
- The held-out documents of one fold are the ONLY place the fine-tuned outputs are out-of-fold, so a calibrator for fine-tuned outputs cannot be trained on other folds until folds 1 and 2 exist.
- Labels are post-rule; population = emitted non-null fields; the zero-shot arm is the 02 run restricted to the same documents.
