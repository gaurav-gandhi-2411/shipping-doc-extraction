# Calibration v2 on post-rule outputs

> **UNVERIFIED: aggregates computed by scripts/calibrate_v2.py from the 02 zero-shot run; a verifier has not recomputed them.**

Provenance: run `zeroshot500_qwen35_4b_img_only_native_4c17aa3`, repo HEAD `139186f` (scripts uncommitted at run time: confidence_v2.py sha256 `fe88372b4f9e`, calibrate_v2.py sha256 `87bf80baf539`), command `uv run python scripts/calibrate_v2.py --run-dir <02 run> --require-logprobs`, wall time 180 s. Bootstrap: doc-level percentile, 2000 resamples, seed 42. All 500 docs are OOF (supplier-fold cross-fit); `dev` = the 100 dev docs' OOF predictions (only 100 docs: wide CIs). Test data not used. Pre-registration: docstring of `scripts/calibrate_v2.py`.

Model: structure **pooled**, kind **lr**; emitted OOF log-loss {"pooled/lr": 0.2111, "pooled/gbm": 0.2244, "per_type/lr": 0.2798, "per_type/gbm": 0.2462}; kind per structure {'pooled': 'lr', 'per_type': 'gbm'}. R3 shapes: per-fold honest; equal to the all-gold set per fold: {0: True, 1: True, 2: True}.

Rules summary (production path): `{"n_docs": 500, "switches": {"r1": true, "r2": true, "r3": true}, "eligible_docs": {"R1": 100, "R2": 100, "R3": 400}, "touched_docs": {"R1": 7, "R2": 29, "R3": 25}, "changes": {"R1": 7, "R2": 41, "R3": 287}, "skipped": {}}`; rule-touched emitted fields `{"R1": {"n": 7, "correct": 7}, "R2": {"n": 41, "correct": 41}, "R3": {"n": 287, "correct": 278}}`.

## Honest coverage statement

- slice `all` (500 docs), nested, per type:
  - header: 95%: coverage 100.0 [99.9, 100.0], precision 99.8 [99.7, 99.9] (target met at point estimate), CI width cov 0.1 pts; 98%: coverage 100.0 [99.9, 100.0], precision 99.8 [99.7, 99.9] (target met at point estimate), CI width cov 0.1 pts; 99%: coverage 100.0 [99.9, 100.0], precision 99.8 [99.7, 99.9] (target met at point estimate), CI width cov 0.1 pts
  - supplier_part_number: 95%: coverage 97.7 [97.1, 98.3], precision 92.8 [90.7, 94.7] (TARGET MISSED), CI width cov 1.2 pts; 98%: coverage 85.6 [83.4, 87.7], precision 98.1 [97.0, 99.0] (target met at point estimate), CI width cov 4.3 pts; 99%: coverage 73.0 [69.9, 75.9], precision 99.1 [98.7, 99.4] (target met at point estimate), CI width cov 6.0 pts
  - customer_part_number: 95%: coverage 7.9 [4.0, 12.4], precision 43.8 [18.1, 71.5] (TARGET MISSED), CI width cov 8.4 pts; 98%: coverage 3.6 [0.9, 6.8], precision 90.5 [80.9, 97.6] (TARGET MISSED), CI width cov 5.9 pts; 99%: coverage 3.6 [0.9, 6.8], precision 90.5 [80.9, 97.6] (TARGET MISSED), CI width cov 5.9 pts
  - purchase_order: 95%: coverage 87.4 [83.4, 91.0], precision 90.3 [87.3, 93.1] (TARGET MISSED), CI width cov 7.6 pts; 98%: coverage 18.3 [12.7, 24.4], precision 96.9 [95.3, 98.3] (TARGET MISSED), CI width cov 11.6 pts; 99%: coverage 18.3 [12.7, 24.4], precision 96.9 [95.3, 98.3] (TARGET MISSED), CI width cov 11.6 pts
  - quantity: 95%: coverage 84.4 [81.2, 87.3], precision 92.8 [90.7, 94.6] (TARGET MISSED), CI width cov 6.1 pts; 98%: coverage 0.0 [0.0, 0.1], precision 0.0 [0.0, 0.0] (TARGET MISSED), CI width cov 0.1 pts; 99%: coverage 0.0 [0.0, 0.1], precision 0.0 [0.0, 0.0] (TARGET MISSED), CI width cov 0.1 pts
  - all_per_type_tau: 95%: coverage 83.3 [81.8, 84.7], precision 93.7 [92.0, 95.2] (TARGET MISSED), CI width cov 2.9 pts; 98%: coverage 46.8 [44.9, 48.9], precision 98.7 [98.1, 99.2] (target met at point estimate), CI width cov 3.9 pts; 99%: coverage 43.5 [41.6, 45.5], precision 99.2 [99.0, 99.4] (target met at point estimate), CI width cov 3.9 pts
- slice `dev` (100 docs), nested, per type:
  - header: 95%: coverage 100.0 [100.0, 100.0], precision 99.7 [99.4, 100.0] (target met at point estimate), CI width cov 0.0 pts; 98%: coverage 100.0 [100.0, 100.0], precision 99.7 [99.4, 100.0] (target met at point estimate), CI width cov 0.0 pts; 99%: coverage 100.0 [100.0, 100.0], precision 99.7 [99.4, 100.0] (target met at point estimate), CI width cov 0.0 pts
  - supplier_part_number: 95%: coverage 97.0 [95.1, 98.6], precision 92.2 [86.2, 97.0] (TARGET MISSED), CI width cov 3.4 pts; 98%: coverage 85.2 [79.2, 90.3], precision 98.1 [94.3, 99.9] (target met at point estimate), CI width cov 11.0 pts; 99%: coverage 73.3 [66.5, 79.6], precision 99.0 [97.4, 99.9] (TARGET MISSED), CI width cov 13.1 pts
  - customer_part_number: 95%: coverage 0.0 [0.0, 0.0], precision n/a [n/a, n/a] (TARGET MISSED), CI width cov 0.0 pts; 98%: coverage 0.0 [0.0, 0.0], precision n/a [n/a, n/a] (TARGET MISSED), CI width cov 0.0 pts; 99%: coverage 0.0 [0.0, 0.0], precision n/a [n/a, n/a] (TARGET MISSED), CI width cov 0.0 pts
  - purchase_order: 95%: coverage 87.3 [78.0, 94.7], precision 88.7 [80.6, 95.6] (TARGET MISSED), CI width cov 16.6 pts; 98%: coverage 10.4 [1.8, 21.9], precision 100.0 [100.0, 100.0] (target met at point estimate), CI width cov 20.1 pts; 99%: coverage 10.4 [1.8, 21.9], precision 100.0 [100.0, 100.0] (target met at point estimate), CI width cov 20.1 pts
  - quantity: 95%: coverage 87.8 [81.3, 93.3], precision 92.0 [86.3, 96.6] (TARGET MISSED), CI width cov 12.0 pts; 98%: coverage 0.0 [0.0, 0.0], precision n/a [n/a, n/a] (TARGET MISSED), CI width cov 0.0 pts; 99%: coverage 0.0 [0.0, 0.0], precision n/a [n/a, n/a] (TARGET MISSED), CI width cov 0.0 pts
  - all_per_type_tau: 95%: coverage 81.4 [78.6, 84.6], precision 93.6 [89.7, 97.1] (TARGET MISSED), CI width cov 6.0 pts; 98%: coverage 44.8 [40.4, 50.4], precision 99.0 [97.3, 99.9] (target met at point estimate), CI width cov 10.0 pts; 99%: coverage 41.8 [37.6, 47.1], precision 99.4 [98.8, 99.9] (target met at point estimate), CI width cov 9.5 pts
- Document level: 95% NOT ATTAINABLE on this evidence; 98% NOT ATTAINABLE on this evidence.
- The dev slice has only 100 docs: its CIs are wide and it is not a separate test.

## (a) AUROC of P(correct), v1 vs v2 (95% CI)

Variants: `v1_prerule` = v1 as published (pre-rule outputs and labels); `v1_post` = v1 probabilities scored on the post-rule labels, matched rows only (separates the rule effect from the feature effect; rule-filled fields have no v1 probability and drop out); `v2_matched` = v2 on the same rows; `v2_all` = v2 on all its emitted rows; `abl_v2feat_LR` = ablation, the default LR model on the v2 features (model-family effect); `abl_v1feat_samemodel` = ablation, v1 features only with the chosen v2 model on post-rule labels (extra-feature effect). Ablations are diagnostic and feed no selection.


**slice `all`** (500 docs)

| group | wrong (v2) | v1_prerule | v1_post | v2_matched | v2_all | abl_v2feat_LR | abl_v1feat_samemodel |
|---|---|---|---|---|---|---|---|
| header.awb_number | 1 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.buyer_name | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.carrier | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.consignee_name | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.currency | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.destination_airport | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.gross_weight_kg | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.hawb | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.invoice_date | 2 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.invoice_number | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.mawb | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.origin_airport | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.pieces | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.ship_to_name | 2 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.shipper_name | 1 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.supplier_name | 1 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.total_amount | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header_all | 7 | n/a | n/a | n/a | 0.964 [0.878, 1.000] | 0.964 [0.878, 1.000] | 0.965 [0.882, 1.000] |
| row.customer_part_number | 403 | n/a | n/a | n/a | 0.718 [0.595, 0.824] | 0.718 [0.595, 0.824] | 0.481 [0.367, 0.587] |
| row.purchase_order | 292 | n/a | n/a | n/a | 0.813 [0.760, 0.860] | 0.813 [0.760, 0.860] | 0.633 [0.542, 0.715] |
| row.quantity | 450 | n/a | n/a | n/a | 0.779 [0.718, 0.827] | 0.779 [0.718, 0.827] | 0.504 [0.443, 0.563] |
| row.supplier_part_number | 421 | n/a | n/a | n/a | 0.933 [0.907, 0.954] | 0.933 [0.907, 0.954] | 0.641 [0.569, 0.723] |

**slice `dev`** (100 docs)

| group | wrong (v2) | v1_prerule | v1_post | v2_matched | v2_all | abl_v2feat_LR | abl_v1feat_samemodel |
|---|---|---|---|---|---|---|---|
| header.awb_number | 1 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.buyer_name | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.carrier | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.consignee_name | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.currency | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.destination_airport | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.gross_weight_kg | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.hawb | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.invoice_date | 1 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.invoice_number | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.mawb | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.origin_airport | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.pieces | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.ship_to_name | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.shipper_name | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.supplier_name | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.total_amount | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header_all | 2 | n/a | n/a | n/a | 0.876 [0.721, 1.000] | 0.876 [0.721, 1.000] | 0.893 [0.773, 1.000] |
| row.customer_part_number | 72 | n/a | n/a | n/a | 0.883 [0.716, 0.961] | 0.883 [0.716, 0.961] | 0.724 [0.542, 0.892] |
| row.purchase_order | 68 | n/a | n/a | n/a | 0.852 [0.726, 0.955] | 0.852 [0.726, 0.955] | 0.697 [0.469, 0.888] |
| row.quantity | 89 | n/a | n/a | n/a | 0.833 [0.646, 0.942] | 0.833 [0.646, 0.942] | 0.535 [0.327, 0.738] |
| row.supplier_part_number | 89 | n/a | n/a | n/a | 0.942 [0.877, 0.983] | 0.942 [0.877, 0.983] | 0.609 [0.401, 0.852] |

**slice `train`** (400 docs)

| group | wrong (v2) | v1_prerule | v1_post | v2_matched | v2_all | abl_v2feat_LR | abl_v1feat_samemodel |
|---|---|---|---|---|---|---|---|
| header.awb_number | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.buyer_name | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.carrier | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.consignee_name | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.currency | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.destination_airport | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.gross_weight_kg | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.hawb | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.invoice_date | 1 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.invoice_number | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.mawb | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.origin_airport | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.pieces | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.ship_to_name | 2 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.shipper_name | 1 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.supplier_name | 1 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.total_amount | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header_all | 5 | n/a | n/a | n/a | 0.998 [0.995, 1.000] | 0.998 [0.995, 1.000] | 0.998 [0.994, 1.000] |
| row.customer_part_number | 331 | n/a | n/a | n/a | 0.685 [0.557, 0.805] | 0.685 [0.557, 0.805] | 0.433 [0.316, 0.553] |
| row.purchase_order | 224 | n/a | n/a | n/a | 0.803 [0.743, 0.854] | 0.803 [0.743, 0.854] | 0.615 [0.525, 0.707] |
| row.quantity | 361 | n/a | n/a | n/a | 0.766 [0.703, 0.818] | 0.766 [0.703, 0.818] | 0.495 [0.436, 0.550] |
| row.supplier_part_number | 332 | n/a | n/a | n/a | 0.930 [0.900, 0.955] | 0.930 [0.900, 0.955] | 0.649 [0.573, 0.728] |

**slice `invoice_only`** (400 docs)

| group | wrong (v2) | v1_prerule | v1_post | v2_matched | v2_all | abl_v2feat_LR | abl_v1feat_samemodel |
|---|---|---|---|---|---|---|---|
| header.awb_number | 1 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.buyer_name | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.currency | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.invoice_date | 2 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.invoice_number | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.ship_to_name | 2 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.supplier_name | 1 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.total_amount | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header_all | 6 | n/a | n/a | n/a | 0.957 [0.859, 1.000] | 0.957 [0.859, 1.000] | 0.954 [0.849, 1.000] |
| row.customer_part_number | 403 | n/a | n/a | n/a | 0.718 [0.595, 0.824] | 0.718 [0.595, 0.824] | 0.481 [0.367, 0.587] |
| row.purchase_order | 292 | n/a | n/a | n/a | 0.813 [0.760, 0.860] | 0.813 [0.760, 0.860] | 0.633 [0.542, 0.715] |
| row.quantity | 450 | n/a | n/a | n/a | 0.779 [0.718, 0.827] | 0.779 [0.718, 0.827] | 0.504 [0.443, 0.563] |
| row.supplier_part_number | 421 | n/a | n/a | n/a | 0.933 [0.907, 0.954] | 0.933 [0.907, 0.954] | 0.641 [0.569, 0.723] |

**slice `waybill_only`** (100 docs)

| group | wrong (v2) | v1_prerule | v1_post | v2_matched | v2_all | abl_v2feat_LR | abl_v1feat_samemodel |
|---|---|---|---|---|---|---|---|
| header.carrier | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.consignee_name | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.destination_airport | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.gross_weight_kg | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.hawb | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.mawb | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.origin_airport | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.pieces | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.shipper_name | 1 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header_all | 1 | n/a | n/a | n/a | 0.998 [0.994, 1.000] | 0.998 [0.994, 1.000] | 0.998 [0.994, 1.000] |

**slice `scanned`** (193 docs)

| group | wrong (v2) | v1_prerule | v1_post | v2_matched | v2_all | abl_v2feat_LR | abl_v1feat_samemodel |
|---|---|---|---|---|---|---|---|
| header.awb_number | 1 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.buyer_name | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.carrier | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.consignee_name | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.currency | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.destination_airport | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.gross_weight_kg | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.hawb | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.invoice_date | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.invoice_number | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.mawb | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.origin_airport | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.pieces | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.ship_to_name | 1 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.shipper_name | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.supplier_name | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.total_amount | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header_all | 2 | n/a | n/a | n/a | 0.849 [0.673, 1.000] | 0.849 [0.673, 1.000] | 0.877 [0.741, 1.000] |
| row.customer_part_number | 186 | n/a | n/a | n/a | 0.754 [0.601, 0.879] | 0.754 [0.601, 0.879] | 0.510 [0.356, 0.659] |
| row.purchase_order | 152 | n/a | n/a | n/a | 0.836 [0.767, 0.890] | 0.836 [0.767, 0.890] | 0.607 [0.499, 0.705] |
| row.quantity | 224 | n/a | n/a | n/a | 0.785 [0.703, 0.847] | 0.785 [0.703, 0.847] | 0.467 [0.389, 0.557] |
| row.supplier_part_number | 195 | n/a | n/a | n/a | 0.929 [0.886, 0.959] | 0.929 [0.886, 0.959] | 0.655 [0.563, 0.766] |

**slice `digital`** (307 docs)

| group | wrong (v2) | v1_prerule | v1_post | v2_matched | v2_all | abl_v2feat_LR | abl_v1feat_samemodel |
|---|---|---|---|---|---|---|---|
| header.awb_number | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.buyer_name | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.carrier | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.consignee_name | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.currency | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.destination_airport | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.gross_weight_kg | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.hawb | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.invoice_date | 2 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.invoice_number | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.mawb | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.origin_airport | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.pieces | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.ship_to_name | 1 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.shipper_name | 1 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.supplier_name | 1 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.total_amount | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header_all | 5 | n/a | n/a | n/a | 0.998 [0.996, 1.000] | 0.998 [0.996, 1.000] | 0.998 [0.995, 1.000] |
| row.customer_part_number | 217 | n/a | n/a | n/a | 0.680 [0.505, 0.832] | 0.680 [0.505, 0.832] | 0.427 [0.275, 0.598] |
| row.purchase_order | 140 | n/a | n/a | n/a | 0.783 [0.700, 0.854] | 0.783 [0.700, 0.854] | 0.607 [0.466, 0.759] |
| row.quantity | 226 | n/a | n/a | n/a | 0.766 [0.678, 0.837] | 0.766 [0.678, 0.837] | 0.501 [0.409, 0.588] |
| row.supplier_part_number | 226 | n/a | n/a | n/a | 0.935 [0.901, 0.961] | 0.935 [0.901, 0.961] | 0.608 [0.507, 0.721] |

## (b) Per-field-type tau, target 98% precision, NESTED (95% CI, %)

tau of fold k chosen on the other two folds' OOF predictions, applied to fold k, pooled. `in-sample` = tau chosen on the evaluated rows (optimistic, reference only). CIs condition on the taus.


**slice `all`**

| population | fields | precision | coverage | review rate | error recall | in-sample precision / coverage |
|---|---|---|---|---|---|---|
| header | 3908 | 99.8 [99.7, 99.9] | 100.0 [99.9, 100.0] | 0.0 [0.0, 0.1] | 14.3 [0.0, 50.0] | 99.8 [99.7, 99.9] / cov 100.0 [100.0, 100.0] |
| supplier_part_number | 4930 | 98.1 [97.0, 99.0] | 85.6 [83.4, 87.7] | 14.4 [12.3, 16.6] | 80.5 [70.7, 89.1] | 98.0 [96.9, 98.9] / cov 84.9 [82.7, 87.1] |
| customer_part_number | 2032 | 90.5 [80.9, 97.6] | 3.6 [0.9, 6.8] | 96.4 [93.2, 99.1] | 98.3 [95.6, 99.8] | 100.0 [100.0, 100.0] / cov 0.6 [0.0, 1.6] |
| purchase_order | 2804 | 96.9 [95.3, 98.3] | 18.3 [12.7, 24.4] | 81.7 [75.6, 87.3] | 94.5 [90.5, 97.4] | 98.0 [95.2, 100.0] / cov 1.8 [0.3, 3.7] |
| quantity | 4930 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.1] | 100.0 [99.9, 100.0] | 99.6 [98.7, 100.0] | n/a [n/a, n/a] / cov 0.0 [0.0, 0.0] |
| all_per_type_tau | 18604 | 98.7 [98.1, 99.2] | 46.8 [44.9, 48.9] | 53.2 [51.1, 55.1] | 92.8 [89.6, 95.4] | 98.9 [98.3, 99.3] / cov 43.9 [42.1, 45.8] |
| all_global_tau | 18604 | 98.1 [97.4, 98.8] | 57.5 [54.1, 61.0] | 42.5 [39.0, 45.9] | 87.3 [81.6, 91.8] | 98.0 [97.3, 98.6] / cov 60.8 [57.8, 63.9] |

**slice `dev`**

| population | fields | precision | coverage | review rate | error recall | in-sample precision / coverage |
|---|---|---|---|---|---|---|
| header | 789 | 99.7 [99.4, 100.0] | 100.0 [100.0, 100.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 99.7 [99.4, 100.0] / cov 100.0 [100.0, 100.0] |
| supplier_part_number | 924 | 98.1 [94.3, 99.9] | 85.2 [79.2, 90.3] | 14.8 [9.7, 20.8] | 83.1 [53.6, 99.1] | 98.2 [94.7, 99.9] / cov 84.4 [78.6, 89.4] |
| customer_part_number | 469 | n/a [n/a, n/a] | 0.0 [0.0, 0.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | n/a [n/a, n/a] / cov 0.0 [0.0, 0.0] |
| purchase_order | 536 | 100.0 [100.0, 100.0] | 10.4 [1.8, 21.9] | 89.6 [78.1, 98.2] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] / cov 3.5 [0.0, 10.3] |
| quantity | 924 | n/a [n/a, n/a] | 0.0 [0.0, 0.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | n/a [n/a, n/a] / cov 0.0 [0.0, 0.0] |
| all_per_type_tau | 3642 | 99.0 [97.3, 99.9] | 44.8 [40.4, 50.4] | 55.2 [49.6, 59.6] | 94.7 [86.3, 99.4] | 99.0 [97.5, 99.9] / cov 43.6 [39.5, 48.9] |
| all_global_tau | 3642 | 99.3 [98.8, 99.8] | 52.9 [45.5, 62.0] | 47.1 [38.0, 54.5] | 95.9 [90.6, 98.6] | 98.8 [97.5, 99.7] / cov 57.6 [51.5, 65.0] |

**slice `train`**

| population | fields | precision | coverage | review rate | error recall | in-sample precision / coverage |
|---|---|---|---|---|---|---|
| header | 3119 | 99.9 [99.7, 100.0] | 100.0 [99.9, 100.0] | 0.0 [0.0, 0.1] | 20.0 [0.0, 66.7] | 99.8 [99.7, 100.0] / cov 100.0 [100.0, 100.0] |
| supplier_part_number | 4006 | 98.0 [96.8, 98.9] | 85.7 [83.2, 87.9] | 14.3 [12.1, 16.8] | 79.8 [68.5, 88.4] | 98.0 [96.7, 98.9] / cov 85.0 [82.5, 87.5] |
| customer_part_number | 1563 | 90.5 [81.1, 97.4] | 4.7 [1.2, 9.1] | 95.3 [90.9, 98.8] | 97.9 [94.7, 99.7] | 100.0 [100.0, 100.0] / cov 0.8 [0.1, 2.1] |
| purchase_order | 2268 | 96.5 [94.7, 98.1] | 20.1 [13.8, 26.6] | 79.9 [73.4, 86.2] | 92.9 [87.8, 96.6] | 96.9 [94.3, 100.0] / cov 1.4 [0.1, 3.3] |
| quantity | 4006 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.1] | 100.0 [99.9, 100.0] | 99.4 [98.5, 100.0] | n/a [n/a, n/a] / cov 0.0 [0.0, 0.0] |
| all_per_type_tau | 14962 | 98.6 [98.0, 99.1] | 47.3 [45.2, 49.5] | 52.7 [50.5, 54.8] | 92.3 [88.7, 95.1] | 98.9 [98.2, 99.3] / cov 43.9 [42.0, 46.0] |
| all_global_tau | 14962 | 97.9 [97.1, 98.6] | 58.6 [54.8, 62.3] | 41.4 [37.7, 45.2] | 85.1 [77.5, 90.5] | 97.8 [97.1, 98.5] / cov 61.6 [58.1, 65.0] |

**slice `invoice_only`**

| population | fields | precision | coverage | review rate | error recall | in-sample precision / coverage |
|---|---|---|---|---|---|---|
| header | 3029 | 99.8 [99.7, 100.0] | 100.0 [99.9, 100.0] | 0.0 [0.0, 0.1] | 16.7 [0.0, 50.0] | 99.8 [99.6, 99.9] / cov 100.0 [100.0, 100.0] |
| supplier_part_number | 4930 | 98.1 [97.0, 99.0] | 85.6 [83.4, 87.7] | 14.4 [12.3, 16.6] | 80.5 [70.7, 89.1] | 98.0 [96.9, 98.9] / cov 84.9 [82.7, 87.1] |
| customer_part_number | 2032 | 90.5 [80.9, 97.6] | 3.6 [0.9, 6.8] | 96.4 [93.2, 99.1] | 98.3 [95.6, 99.8] | 100.0 [100.0, 100.0] / cov 0.6 [0.0, 1.6] |
| purchase_order | 2804 | 96.9 [95.3, 98.3] | 18.3 [12.7, 24.4] | 81.7 [75.6, 87.3] | 94.5 [90.5, 97.4] | 98.0 [95.2, 100.0] / cov 1.8 [0.3, 3.7] |
| quantity | 4930 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.1] | 100.0 [99.9, 100.0] | 99.6 [98.7, 100.0] | n/a [n/a, n/a] / cov 0.0 [0.0, 0.0] |
| all_per_type_tau | 17725 | 98.6 [97.9, 99.1] | 44.2 [42.4, 46.2] | 55.8 [53.8, 57.6] | 92.9 [89.8, 95.5] | 98.8 [98.1, 99.3] / cov 41.1 [39.5, 42.8] |
| all_global_tau | 17725 | 98.0 [97.2, 98.6] | 55.4 [51.9, 59.1] | 44.6 [40.9, 48.1] | 87.3 [81.6, 91.6] | 97.8 [97.1, 98.5] / cov 58.9 [55.8, 62.2] |

**slice `waybill_only`**

| population | fields | precision | coverage | review rate | error recall | in-sample precision / coverage |
|---|---|---|---|---|---|---|
| header | 879 | 99.9 [99.7, 100.0] | 100.0 [100.0, 100.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 99.9 [99.7, 100.0] / cov 100.0 [100.0, 100.0] |
| all_per_type_tau | 879 | 99.9 [99.7, 100.0] | 100.0 [100.0, 100.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 99.9 [99.7, 100.0] / cov 100.0 [100.0, 100.0] |
| all_global_tau | 879 | 100.0 [100.0, 100.0] | 99.3 [98.7, 99.8] | 0.7 [0.2, 1.3] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] / cov 99.3 [98.7, 99.8] |

**slice `scanned`**

| population | fields | precision | coverage | review rate | error recall | in-sample precision / coverage |
|---|---|---|---|---|---|---|
| header | 1511 | 99.9 [99.7, 100.0] | 100.0 [100.0, 100.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 99.9 [99.7, 100.0] / cov 100.0 [100.0, 100.0] |
| supplier_part_number | 1853 | 98.2 [96.9, 99.1] | 82.2 [78.5, 85.9] | 17.8 [14.1, 21.5] | 85.6 [75.0, 93.1] | 98.1 [96.5, 99.1] / cov 81.1 [77.1, 84.9] |
| customer_part_number | 851 | 100.0 [100.0, 100.0] | 0.2 [0.0, 0.6] | 99.8 [99.4, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] / cov 0.2 [0.0, 0.6] |
| purchase_order | 1076 | 96.9 [93.9, 99.2] | 15.1 [7.4, 23.4] | 84.9 [76.6, 92.6] | 96.7 [92.5, 99.3] | 100.0 [100.0, 100.0] / cov 0.5 [0.0, 1.6] |
| quantity | 1853 | 0.0 [0.0, 0.0] | 0.1 [0.0, 0.3] | 99.9 [99.7, 100.0] | 99.1 [97.5, 100.0] | n/a [n/a, n/a] / cov 0.0 [0.0, 0.0] |
| all_per_type_tau | 7144 | 98.8 [98.1, 99.4] | 44.8 [41.7, 48.4] | 55.2 [51.6, 58.3] | 95.1 [91.6, 97.4] | 99.0 [98.2, 99.5] / cov 42.3 [39.5, 45.6] |
| all_global_tau | 7144 | 98.0 [96.7, 99.0] | 55.8 [50.3, 61.9] | 44.2 [38.1, 49.7] | 89.5 [81.2, 94.8] | 97.8 [96.6, 98.8] / cov 58.7 [53.8, 64.2] |

**slice `digital`**

| population | fields | precision | coverage | review rate | error recall | in-sample precision / coverage |
|---|---|---|---|---|---|---|
| header | 2397 | 99.8 [99.7, 100.0] | 100.0 [99.9, 100.0] | 0.0 [0.0, 0.1] | 20.0 [0.0, 66.7] | 99.8 [99.6, 100.0] / cov 100.0 [100.0, 100.0] |
| supplier_part_number | 3077 | 98.0 [96.4, 99.1] | 87.6 [84.9, 89.9] | 12.4 [10.1, 15.1] | 76.1 [60.2, 88.9] | 98.0 [96.4, 99.1] / cov 87.2 [84.5, 89.7] |
| customer_part_number | 1181 | 90.3 [80.4, 97.4] | 6.1 [1.4, 11.4] | 93.9 [88.6, 98.6] | 96.8 [91.5, 99.6] | 100.0 [100.0, 100.0] / cov 0.8 [0.0, 2.5] |
| purchase_order | 1728 | 96.9 [94.7, 98.6] | 20.3 [12.8, 28.9] | 79.7 [71.1, 87.2] | 92.1 [84.3, 97.3] | 97.8 [94.7, 100.0] / cov 2.7 [0.2, 6.0] |
| quantity | 3077 | n/a [n/a, n/a] | 0.0 [0.0, 0.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | n/a [n/a, n/a] / cov 0.0 [0.0, 0.0] |
| all_per_type_tau | 11460 | 98.6 [97.8, 99.3] | 48.1 [45.6, 50.8] | 51.9 [49.2, 54.4] | 90.7 [85.7, 94.8] | 98.8 [98.0, 99.5] / cov 44.8 [42.6, 47.3] |
| all_global_tau | 11460 | 98.2 [97.3, 99.0] | 58.6 [54.1, 63.2] | 41.4 [36.8, 45.9] | 85.3 [76.0, 92.0] | 98.1 [97.3, 98.8] / cov 62.2 [58.4, 66.3] |

## Coverage at precision targets 95 / 98 / 99% (nested; all and dev)

Cell: nested coverage % [CI] ; nested precision % [CI] ; in-sample optimum coverage (point, optimistic).


**slice `all`**

| population | 95% | 98% | 99% |
|---|---|---|---|
| header | 100.0 [99.9, 100.0] ; 99.8 [99.7, 99.9] ; 100.0 | 100.0 [99.9, 100.0] ; 99.8 [99.7, 99.9] ; 100.0 | 100.0 [99.9, 100.0] ; 99.8 [99.7, 99.9] ; 100.0 |
| supplier_part_number | 97.7 [97.1, 98.3] ; 92.8 [90.7, 94.7] ; 94.8 | 85.6 [83.4, 87.7] ; 98.1 [97.0, 99.0] ; 84.9 | 73.0 [69.9, 75.9] ; 99.1 [98.7, 99.4] ; 73.1 |
| customer_part_number | 7.9 [4.0, 12.4] ; 43.8 [18.1, 71.5] ; 11.3 | 3.6 [0.9, 6.8] ; 90.5 [80.9, 97.6] ; 0.6 | 3.6 [0.9, 6.8] ; 90.5 [80.9, 97.6] ; 0.6 |
| purchase_order | 87.4 [83.4, 91.0] ; 90.3 [87.3, 93.1] ; 87.9 | 18.3 [12.7, 24.4] ; 96.9 [95.3, 98.3] ; 1.8 | 18.3 [12.7, 24.4] ; 96.9 [95.3, 98.3] ; 0.4 |
| quantity | 84.4 [81.2, 87.3] ; 92.8 [90.7, 94.6] ; 87.1 | 0.0 [0.0, 0.1] ; 0.0 [0.0, 0.0] ; 0.0 | 0.0 [0.0, 0.1] ; 0.0 [0.0, 0.0] ; 0.0 |
| all_global_tau | 96.3 [95.4, 97.1] ; 93.8 [92.3, 95.1] ; 94.8 | 57.5 [54.1, 61.0] ; 98.1 [97.4, 98.8] ; 60.8 | 43.3 [39.9, 46.8] ; 99.0 [98.6, 99.3] ; 43.6 |

**slice `dev`**

| population | 95% | 98% | 99% |
|---|---|---|---|
| header | 100.0 [100.0, 100.0] ; 99.7 [99.4, 100.0] ; 100.0 | 100.0 [100.0, 100.0] ; 99.7 [99.4, 100.0] ; 100.0 | 100.0 [100.0, 100.0] ; 99.7 [99.4, 100.0] ; 100.0 |
| supplier_part_number | 97.0 [95.1, 98.6] ; 92.2 [86.2, 97.0] ; 93.4 | 85.2 [79.2, 90.3] ; 98.1 [94.3, 99.9] ; 87.8 | 73.3 [66.5, 79.6] ; 99.0 [97.4, 99.9] ; 71.5 |
| customer_part_number | 0.0 [0.0, 0.0] ; n/a [n/a, n/a] ; 85.9 | 0.0 [0.0, 0.0] ; n/a [n/a, n/a] ; 6.6 | 0.0 [0.0, 0.0] ; n/a [n/a, n/a] ; 6.6 |
| purchase_order | 87.3 [78.0, 94.7] ; 88.7 [80.6, 95.6] ; 86.4 | 10.4 [1.8, 21.9] ; 100.0 [100.0, 100.0] ; 39.6 | 10.4 [1.8, 21.9] ; 100.0 [100.0, 100.0] ; 30.6 |
| quantity | 87.8 [81.3, 93.3] ; 92.0 [86.3, 96.6] ; 93.7 | 0.0 [0.0, 0.0] ; n/a [n/a, n/a] ; 39.3 | 0.0 [0.0, 0.0] ; n/a [n/a, n/a] ; 0.0 |
| all_global_tau | 96.1 [93.6, 98.3] ; 94.0 [90.5, 97.2] ; 94.8 | 52.9 [45.5, 62.0] ; 99.3 [98.8, 99.8] ; 65.1 | 37.8 [30.4, 46.8] ; 99.3 [98.6, 99.9] ; 56.2 |

## (c) Document auto-accept (fully correct document)

Base rate of fully-correct docs (95% CI, %):

| slice | docs | base rate |
|---|---|---|
| all | 500 | 68.8 [64.6, 72.8] |
| dev | 100 | 73.0 [64.0, 81.0] |
| train | 400 | 67.8 [63.0, 72.5] |
| invoice_only | 400 | 61.3 [56.5, 66.2] |
| waybill_only | 100 | 99.0 [97.0, 100.0] |
| scanned | 193 | 65.3 [58.5, 72.0] |
| digital | 307 | 71.0 [65.8, 76.2] |

Target 95%: **NOT ATTAINABLE on this evidence** (nested precision < target or coverage < 5%); best in-sample precision at >= 5% coverage: 90.5% at coverage 38.0%.

| slice | docs | nested precision | nested coverage | in-sample |
|---|---|---|---|---|
| all | 500 | 79.6 [67.4, 90.2] | 9.8 [7.4, 12.4] | 100.0 [100.0, 100.0] / cov 0.2 [0.0, 0.6] |
| dev | 100 | 88.9 [66.7, 100.0] | 9.0 [4.0, 15.0] | 100.0 [100.0, 100.0] / cov 1.0 [0.0, 3.0] |
| train | 400 | 77.5 [63.0, 90.0] | 10.0 [7.2, 13.0] | n/a [n/a, n/a] / cov 0.0 [0.0, 0.0] |
| invoice_only | 400 | 47.4 [24.0, 70.6] | 4.8 [2.8, 7.0] | 100.0 [100.0, 100.0] / cov 0.2 [0.0, 0.8] |
| waybill_only | 100 | 100.0 [100.0, 100.0] | 30.0 [21.0, 40.0] | n/a [n/a, n/a] / cov 0.0 [0.0, 0.0] |
| scanned | 193 | 88.2 [70.0, 100.0] | 8.8 [5.2, 13.0] | n/a [n/a, n/a] / cov 0.0 [0.0, 0.0] |
| digital | 307 | 75.0 [59.3, 89.7] | 10.4 [7.2, 14.0] | 100.0 [100.0, 100.0] / cov 0.3 [0.0, 1.0] |

Target 98%: **NOT ATTAINABLE on this evidence** (nested precision < target or coverage < 5%); best in-sample precision at >= 5% coverage: 90.5% at coverage 38.0%.

| slice | docs | nested precision | nested coverage | in-sample |
|---|---|---|---|---|
| all | 500 | 73.7 [58.8, 87.0] | 7.6 [5.4, 10.0] | 100.0 [100.0, 100.0] / cov 0.2 [0.0, 0.6] |
| dev | 100 | 85.7 [50.0, 100.0] | 7.0 [3.0, 13.0] | 100.0 [100.0, 100.0] / cov 1.0 [0.0, 3.0] |
| train | 400 | 71.0 [52.9, 86.4] | 7.8 [5.2, 10.2] | n/a [n/a, n/a] / cov 0.0 [0.0, 0.0] |
| invoice_only | 400 | 37.5 [14.3, 62.5] | 4.0 [2.2, 6.0] | 100.0 [100.0, 100.0] / cov 0.2 [0.0, 0.8] |
| waybill_only | 100 | 100.0 [100.0, 100.0] | 22.0 [14.0, 31.0] | n/a [n/a, n/a] / cov 0.0 [0.0, 0.0] |
| scanned | 193 | 86.7 [66.7, 100.0] | 7.8 [4.1, 11.4] | n/a [n/a, n/a] / cov 0.0 [0.0, 0.0] |
| digital | 307 | 65.2 [45.0, 84.6] | 7.5 [4.6, 10.4] | 100.0 [100.0, 100.0] / cov 0.3 [0.0, 1.0] |

## (d) Calibration (ECE 15 equal-width / equal-mass bins)

| slice | group | n | base rate | log-loss | ECE w15 | ECE m15 |
|---|---|---|---|---|---|---|
| all | header | 3908 | 0.998 | 0.006 | 0.001 | 0.001 |
| all | supplier_part_number | 4930 | 0.915 | 0.166 | 0.027 | 0.029 |
| all | customer_part_number | 2032 | 0.802 | 0.474 | 0.134 | 0.118 |
| all | purchase_order | 2804 | 0.896 | 0.278 | 0.038 | 0.032 |
| all | quantity | 4930 | 0.909 | 0.272 | 0.031 | 0.028 |
| all | all_emitted | 18604 | 0.915 | 0.211 | 0.029 | 0.029 |
| all | doc | 500 | 0.688 | 0.606 | 0.088 | 0.089 |
| dev | header | 789 | 0.997 | 0.011 | 0.001 | 0.002 |
| dev | supplier_part_number | 924 | 0.904 | 0.170 | 0.042 | 0.048 |
| dev | customer_part_number | 469 | 0.846 | 0.289 | 0.112 | 0.101 |
| dev | purchase_order | 536 | 0.873 | 0.268 | 0.054 | 0.062 |
| dev | quantity | 924 | 0.904 | 0.239 | 0.047 | 0.055 |
| dev | all_emitted | 3642 | 0.912 | 0.183 | 0.037 | 0.033 |
| dev | doc | 100 | 0.730 | 0.344 | 0.078 | 0.059 |
| train | header | 3119 | 0.998 | 0.005 | 0.001 | 0.001 |
| train | supplier_part_number | 4006 | 0.917 | 0.165 | 0.027 | 0.027 |
| train | customer_part_number | 1563 | 0.788 | 0.530 | 0.143 | 0.131 |
| train | purchase_order | 2268 | 0.901 | 0.280 | 0.039 | 0.036 |
| train | quantity | 4006 | 0.910 | 0.280 | 0.028 | 0.026 |
| train | all_emitted | 14962 | 0.916 | 0.218 | 0.028 | 0.027 |
| train | doc | 400 | 0.677 | 0.672 | 0.106 | 0.113 |
| invoice_only | header | 3029 | 0.998 | 0.007 | 0.001 | 0.001 |
| invoice_only | supplier_part_number | 4930 | 0.915 | 0.166 | 0.027 | 0.029 |
| invoice_only | customer_part_number | 2032 | 0.802 | 0.474 | 0.134 | 0.118 |
| invoice_only | purchase_order | 2804 | 0.896 | 0.278 | 0.038 | 0.032 |
| invoice_only | quantity | 4930 | 0.909 | 0.272 | 0.031 | 0.028 |
| invoice_only | all_emitted | 17725 | 0.911 | 0.221 | 0.031 | 0.029 |
| invoice_only | doc | 400 | 0.613 | 0.742 | 0.109 | 0.114 |
| waybill_only | header | 879 | 0.999 | 0.004 | 0.002 | 0.001 |
| waybill_only | all_emitted | 879 | 0.999 | 0.004 | 0.002 | 0.001 |
| waybill_only | doc | 100 | 0.990 | 0.063 | 0.002 | 0.019 |
| scanned | header | 1511 | 0.999 | 0.007 | 0.001 | 0.001 |
| scanned | supplier_part_number | 1853 | 0.895 | 0.186 | 0.034 | 0.032 |
| scanned | customer_part_number | 851 | 0.781 | 0.473 | 0.123 | 0.108 |
| scanned | purchase_order | 1076 | 0.859 | 0.302 | 0.053 | 0.044 |
| scanned | quantity | 1853 | 0.879 | 0.340 | 0.053 | 0.044 |
| scanned | all_emitted | 7144 | 0.894 | 0.240 | 0.036 | 0.027 |
| scanned | doc | 193 | 0.653 | 0.477 | 0.104 | 0.079 |
| digital | header | 2397 | 0.998 | 0.006 | 0.001 | 0.001 |
| digital | supplier_part_number | 3077 | 0.927 | 0.154 | 0.028 | 0.028 |
| digital | customer_part_number | 1181 | 0.816 | 0.475 | 0.151 | 0.146 |
| digital | purchase_order | 1728 | 0.919 | 0.263 | 0.032 | 0.027 |
| digital | quantity | 3077 | 0.927 | 0.232 | 0.027 | 0.032 |
| digital | all_emitted | 11460 | 0.929 | 0.193 | 0.029 | 0.029 |
| digital | doc | 307 | 0.710 | 0.688 | 0.104 | 0.105 |

Reliability tables and risk-coverage curves: `reliability.csv`, `risk_coverage.csv` in the run folder (not reproduced here).


## Caveats

- CIs resample documents but condition on the selected taus; tau-selection variance is not in them.
- Model structure / kind were chosen on the same OOF predictions that are reported.
- The doc-level model stacks on field-level OOF probabilities (mild second-level leakage).
- Labels are post-rule; v1 published numbers are pre-rule, hence the matched variants.
- Population of field metrics = emitted non-null fields; null fields are not scored here (no null policy; the row null policy was rejected).

<!-- native-vs-1260-begin (generated by scripts/calibration_compare_native.py) -->
## Native calibrator vs the 1260-token calibrator

Sources: 1260 = `zeroshot500_qwen35_4b_img_only_keyed_42b812b`, native = `zeroshot500_qwen35_4b_img_only_native_4c17aa3` (`calibration_v2.json` of each).

Nested coverage % / nested precision % (point estimates, slice `all`, supplier-fold nested tau; CIs are in the two calibration_v2 reports):

| population | 95% 1260 | 95% native | 98% 1260 | 98% native | 99% 1260 | 99% native |
|---|---|---|---|---|---|---|
| header | 100.0 / 99.4 | 100.0 / 99.8 | 100.0 / 99.4 | 100.0 / 99.8 | 100.0 / 99.4 | 100.0 / 99.8 |
| supplier_part_number | 88.0 / 95.0 | 97.7 / 92.8 | 74.3 / 98.0 | 85.6 / 98.1 | 45.6 / 98.8 | 73.0 / 99.1 |
| customer_part_number | 3.5 / 0.0 | 7.9 / 43.8 | 3.5 / 0.0 | 3.6 / 90.5 | 3.5 / 0.0 | 3.6 / 90.5 |
| purchase_order | 0.0 / n/a | 87.4 / 90.3 | 0.0 / n/a | 18.3 / 96.9 | 0.0 / n/a | 18.3 / 96.9 |
| quantity | 1.3 / 72.3 | 84.4 / 92.8 | 1.3 / 72.3 | 0.0 / 0.0 | 1.3 / 72.3 | 0.0 / 0.0 |
| all_per_type_tau | 45.1 / 96.0 | 83.3 / 93.7 | 41.5 / 97.5 | 46.8 / 98.7 | 33.9 / 97.7 | 43.5 / 99.2 |
| all_global_tau | 64.0 / 95.1 | 96.3 / 93.8 | 40.6 / 97.8 | 57.5 / 98.1 | 28.4 / 98.8 | 43.3 / 99.0 |

AUROC of P(correct), v2 model on all its rows, slice `all` (n/a = fewer than 5 wrong):

| group | 1260 | native |
|---|---|---|
| header_all | 0.976 | 0.964 |
| row.supplier_part_number | 0.935 | 0.933 |
| row.customer_part_number | 0.645 | 0.718 |
| row.purchase_order | 0.700 | 0.813 |
| row.quantity | 0.642 | 0.779 |

ECE, 15 equal-width bins, slice `all`:

| group | 1260 | native |
|---|---|---|
| header | 0.010 | 0.001 |
| supplier_part_number | 0.039 | 0.027 |
| customer_part_number | 0.094 | 0.134 |
| purchase_order | 0.060 | 0.038 |
| quantity | 0.034 | 0.031 |
| all_emitted | 0.019 | 0.029 |
| doc | 0.056 | 0.088 |

Model structure / kind: 1260 pooled / gbm, native pooled / lr. Document auto-accept at 95 / 98%: 1260 attained False / False, native attained False / False.

## Which calibrator serves which submission

| submission | model run | calibrator file | records `run_config_hash` |
|---|---|---|---|
| v1 (1260-token ZS + rules) | max_pixels 1310720, config hash `01d87878679ca0fc` | `meta/calibrator_zs.json` (untouched) | no (frozen before the guard existed) |
| v1.5 native (ZS + rules at native resolution) | max_pixels 2196480, config hash `e4b84ec2809625d5` | `meta/calibrator_zs_native.json` | yes, `e4b84ec2809625d5` |

`shipdoc.flags.run_stage` refuses a submission whose manifest config hash differs from the calibrator's recorded `run_config_hash`; the native notebooks must name `meta/calibrator_zs_native.json` explicitly (`CALIBRATOR_FILE`) and refuse the 1260 file, which carries no hash.
<!-- native-vs-1260-end -->
