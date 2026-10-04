# Calibration v3 prep: fine-tuned vs zero-shot agreement

> **EXPLORATORY, NOT SHIPPABLE: single-fold measurement. GG: do not ship anything from this until it is cross-fitted across all 3 folds. UNVERIFIED: aggregates computed by scripts/calibrate_v3.py; a verifier has not recomputed them.**

Provenance: OOF run(s) `oof_native_fold0_4c17aa3`, zero-shot run `zeroshot500_qwen35_4b_img_only_native_4c17aa3`, repo HEAD `ab92591` (confidence_v3.py sha256 `4adc9ced79f8`, calibrate_v3.py sha256 `64ae0916e81a`), command `uv run python scripts/calibrate_v3.py --folds 0 --oof-run $SHIPDOC_TMP_DIR\native_extract\oof0\oof_native_fold0_4c17aa3 --zs-run-dir $SHIPDOC_TMP_DIR\native_extract\zs\zeroshot500_qwen35_4b_img_only_native_4c17aa3 --require-logprobs --out-dir $SHIPDOC_TMP_DIR\native_fold0\calv3 --report-md reports/calibration_v3_native_fold0_exploratory.md`, wall time 202 s. Bootstrap: document-level percentile, 2000 resamples, seed 42. Test data not used. Protocol: docstring of `scripts/calibrate_v3.py`.

Documents 171; R3 shapes per-fold honest, equal to the all-gold set per fold: {0: True, 1: True, 2: True}. Rules summary FT `{"n_docs": 171, "switches": {"r1": true, "r2": true, "r3": true}, "eligible_docs": {"R1": 37, "R2": 37, "R3": 134}, "touched_docs": {"R1": 0, "R2": 0, "R3": 3}, "changes": {"R1": 0, "R2": 0, "R3": 8}, "skipped": {}}`; ZS `{"n_docs": 171, "switches": {"r1": true, "r2": true, "r3": true}, "eligible_docs": {"R1": 37, "R2": 37, "R3": 134}, "touched_docs": {"R1": 2, "R2": 8, "R3": 0}, "changes": {"R1": 2, "R2": 12, "R3": 0}, "skipped": {}}`.


## View: judging the FINE-TUNED arm

171 documents, 10 supplier groups, inner GroupKFold k = 5. Structure **pooled**, kind **gbm** (chosen once on the v2 design of the fine-tuned view; emitted OOF log-loss {"pooled/lr": 0.8561, "pooled/gbm": 0.15}).

| group | fields | wrong | docs | base acc | AUROC v2 | AUROC v2+agree | AUROC agree only | AUROC flag | lift (v2+agree - v2) | note |
|---|---|---|---|---|---|---|---|---|---|---|
| header_all | 1338 | 0 | 171 | 1.000 | n/a [n/a, n/a] | n/a [n/a, n/a] | n/a [n/a, n/a] | n/a [n/a, n/a] | n/a [n/a, n/a] | n_wrong<5 |
| row.supplier_part_number | 1543 | 60 | 134 | 0.961 | 0.688 [0.580, 0.790] | 0.706 [0.606, 0.799] | 0.453 [0.330, 0.570] | 0.602 [0.536, 0.676] | 0.019 [-0.010, 0.049] |  |
| row.customer_part_number | 770 | 48 | 64 | 0.938 | 0.401 [0.319, 0.484] | 0.423 [0.334, 0.515] | 0.430 [0.342, 0.526] | 0.513 [0.489, 0.549] | 0.022 [-0.023, 0.084] |  |
| row.purchase_order | 334 | 26 | 26 | 0.922 | 0.410 [0.305, 0.529] | 0.537 [0.389, 0.681] | 0.590 [0.451, 0.721] | 0.623 [0.519, 0.745] | 0.127 [0.042, 0.224] |  |
| row.quantity | 1543 | 60 | 134 | 0.961 | 0.401 [0.316, 0.489] | 0.408 [0.320, 0.501] | 0.401 [0.311, 0.491] | 0.500 [0.500, 0.500] | 0.007 [-0.036, 0.047] |  |

## View: judging the ZERO-SHOT arm

171 documents, 10 supplier groups, inner GroupKFold k = 5. Structure **pooled**, kind **gbm** (chosen once on the v2 design of the fine-tuned view; choice reused).

| group | fields | wrong | docs | base acc | AUROC v2 | AUROC v2+agree | AUROC agree only | AUROC flag | lift (v2+agree - v2) | note |
|---|---|---|---|---|---|---|---|---|---|---|
| header_all | 1338 | 3 | 171 | 0.998 | 0.801 [0.408, 0.996] | 1.000 [0.999, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.198 [0.004, 0.591] | n_wrong<5 |
| row.supplier_part_number | 1543 | 69 | 134 | 0.955 | 0.816 [0.736, 0.879] | 0.790 [0.699, 0.864] | 0.549 [0.444, 0.646] | 0.656 [0.604, 0.710] | -0.026 [-0.065, 0.010] |  |
| row.customer_part_number | 759 | 50 | 64 | 0.934 | 0.391 [0.301, 0.472] | 0.380 [0.295, 0.459] | 0.410 [0.305, 0.510] | 0.498 [0.494, 0.500] | -0.011 [-0.081, 0.054] |  |
| row.purchase_order | 329 | 44 | 23 | 0.866 | 0.682 [0.513, 0.826] | 0.727 [0.550, 0.861] | 0.714 [0.522, 0.871] | 0.750 [0.613, 0.853] | 0.044 [-0.053, 0.147] |  |
| row.quantity | 1543 | 69 | 134 | 0.955 | 0.401 [0.330, 0.467] | 0.368 [0.294, 0.436] | 0.430 [0.336, 0.526] | 0.500 [0.500, 0.500] | -0.034 [-0.093, 0.026] |  |

## Caveats

- One fold holds few suppliers: the supplier-grouped cross-fit inside it trains each calibrator on a handful of suppliers and the group / document counts above are small.
- CIs resample documents but condition on the cross-fit and on the LR-vs-GBM choice.
- The held-out documents of one fold are the ONLY place the fine-tuned outputs are out-of-fold, so a calibrator for fine-tuned outputs cannot be trained on other folds until folds 1 and 2 exist.
- Labels are post-rule; population = emitted non-null fields; the zero-shot arm is the 02 run restricted to the same documents.
