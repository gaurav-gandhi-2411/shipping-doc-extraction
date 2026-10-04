# Calibration v2 on post-rule outputs

> **UNVERIFIED: aggregates computed by scripts/calibrate_v2.py from the 02 zero-shot run; a verifier has not recomputed them.**

Provenance: run `zeroshot500_qwen35_4b_img_only_keyed_42b812b`, repo HEAD `95fb653` (scripts uncommitted at run time: confidence_v2.py sha256 `fe88372b4f9e`, calibrate_v2.py sha256 `87bf80baf539`), command `uv run python scripts/calibrate_v2.py --run-dir <02 run> --require-logprobs`, wall time 358 s. Bootstrap: doc-level percentile, 2000 resamples, seed 42. All 500 docs are OOF (supplier-fold cross-fit); `dev` = the 100 dev docs' OOF predictions (only 100 docs: wide CIs). Test data not used. Pre-registration: docstring of `scripts/calibrate_v2.py`.

Model: structure **pooled**, kind **gbm**; emitted OOF log-loss {"pooled/lr": 0.3243, "pooled/gbm": 0.285, "per_type/lr": 0.3742, "per_type/gbm": 0.3191}; kind per structure {'pooled': 'gbm', 'per_type': 'gbm'}. R3 shapes: per-fold honest; equal to the all-gold set per fold: {0: True, 1: True, 2: True}.

Rules summary (production path): `{"n_docs": 500, "switches": {"r1": true, "r2": true, "r3": true}, "eligible_docs": {"R1": 100, "R2": 100, "R3": 400}, "touched_docs": {"R1": 12, "R2": 26, "R3": 46}, "changes": {"R1": 12, "R2": 32, "R3": 513}, "skipped": {}}`; rule-touched emitted fields `{"R1": {"n": 12, "correct": 12}, "R2": {"n": 32, "correct": 32}, "R3": {"n": 516, "correct": 505}}`.

## Honest coverage statement

- slice `all` (500 docs), nested, per type:
  - header: 95%: coverage 100.0 [99.9, 100.0], precision 99.4 [99.1, 99.6] (target met at point estimate), CI width cov 0.1 pts; 98%: coverage 100.0 [99.9, 100.0], precision 99.4 [99.1, 99.6] (target met at point estimate), CI width cov 0.1 pts; 99%: coverage 100.0 [99.9, 100.0], precision 99.4 [99.1, 99.6] (target met at point estimate), CI width cov 0.1 pts
  - supplier_part_number: 95%: coverage 88.0 [85.8, 90.2], precision 95.0 [93.3, 96.5] (target met at point estimate), CI width cov 4.4 pts; 98%: coverage 74.3 [71.4, 77.1], precision 98.0 [96.8, 98.8] (TARGET MISSED), CI width cov 5.7 pts; 99%: coverage 45.6 [41.4, 49.9], precision 98.8 [98.2, 99.3] (TARGET MISSED), CI width cov 8.5 pts
  - customer_part_number: 95%: coverage 3.5 [1.0, 6.4], precision 0.0 [0.0, 0.0] (TARGET MISSED), CI width cov 5.4 pts; 98%: coverage 3.5 [1.0, 6.4], precision 0.0 [0.0, 0.0] (TARGET MISSED), CI width cov 5.4 pts; 99%: coverage 3.5 [1.0, 6.4], precision 0.0 [0.0, 0.0] (TARGET MISSED), CI width cov 5.4 pts
  - purchase_order: 95%: coverage 0.0 [0.0, 0.0], precision n/a [n/a, n/a] (TARGET MISSED), CI width cov 0.0 pts; 98%: coverage 0.0 [0.0, 0.0], precision n/a [n/a, n/a] (TARGET MISSED), CI width cov 0.0 pts; 99%: coverage 0.0 [0.0, 0.0], precision n/a [n/a, n/a] (TARGET MISSED), CI width cov 0.0 pts
  - quantity: 95%: coverage 1.3 [0.4, 2.5], precision 72.3 [46.7, 92.0] (TARGET MISSED), CI width cov 2.2 pts; 98%: coverage 1.3 [0.4, 2.5], precision 72.3 [46.7, 92.0] (TARGET MISSED), CI width cov 2.2 pts; 99%: coverage 1.3 [0.4, 2.5], precision 72.3 [46.7, 92.0] (TARGET MISSED), CI width cov 2.2 pts
  - all_per_type_tau: 95%: coverage 45.1 [43.4, 47.0], precision 96.0 [94.6, 97.3] (target met at point estimate), CI width cov 3.5 pts; 98%: coverage 41.5 [39.7, 43.4], precision 97.5 [96.4, 98.5] (TARGET MISSED), CI width cov 3.7 pts; 99%: coverage 33.9 [32.0, 36.0], precision 97.7 [96.6, 98.7] (TARGET MISSED), CI width cov 4.1 pts
- slice `dev` (100 docs), nested, per type:
  - header: 95%: coverage 100.0 [100.0, 100.0], precision 99.4 [98.8, 99.9] (target met at point estimate), CI width cov 0.0 pts; 98%: coverage 100.0 [100.0, 100.0], precision 99.4 [98.8, 99.9] (target met at point estimate), CI width cov 0.0 pts; 99%: coverage 100.0 [100.0, 100.0], precision 99.4 [98.8, 99.9] (target met at point estimate), CI width cov 0.0 pts
  - supplier_part_number: 95%: coverage 86.6 [80.5, 91.9], precision 94.0 [89.8, 97.1] (TARGET MISSED), CI width cov 11.4 pts; 98%: coverage 73.3 [64.8, 80.9], precision 98.5 [96.7, 99.8] (target met at point estimate), CI width cov 16.1 pts; 99%: coverage 41.6 [32.7, 50.5], precision 98.2 [95.6, 100.0] (TARGET MISSED), CI width cov 17.8 pts
  - customer_part_number: 95%: coverage 1.4 [0.0, 5.2], precision 0.0 [0.0, 0.0] (TARGET MISSED), CI width cov 5.2 pts; 98%: coverage 1.4 [0.0, 5.2], precision 0.0 [0.0, 0.0] (TARGET MISSED), CI width cov 5.2 pts; 99%: coverage 1.4 [0.0, 5.2], precision 0.0 [0.0, 0.0] (TARGET MISSED), CI width cov 5.2 pts
  - purchase_order: 95%: coverage 0.0 [0.0, 0.0], precision n/a [n/a, n/a] (TARGET MISSED), CI width cov 0.0 pts; 98%: coverage 0.0 [0.0, 0.0], precision n/a [n/a, n/a] (TARGET MISSED), CI width cov 0.0 pts; 99%: coverage 0.0 [0.0, 0.0], precision n/a [n/a, n/a] (TARGET MISSED), CI width cov 0.0 pts
  - quantity: 95%: coverage 1.8 [0.1, 5.4], precision 88.2 [85.7, 100.0] (TARGET MISSED), CI width cov 5.3 pts; 98%: coverage 1.8 [0.1, 5.4], precision 88.2 [85.7, 100.0] (TARGET MISSED), CI width cov 5.3 pts; 99%: coverage 1.8 [0.1, 5.4], precision 88.2 [85.7, 100.0] (TARGET MISSED), CI width cov 5.3 pts
  - all_per_type_tau: 95%: coverage 44.2 [40.2, 49.6], precision 96.1 [94.0, 98.1] (target met at point estimate), CI width cov 9.4 pts; 98%: coverage 40.8 [36.3, 46.3], precision 98.4 [97.0, 99.4] (target met at point estimate), CI width cov 10.0 pts; 99%: coverage 32.8 [28.3, 38.7], precision 98.2 [96.6, 99.5] (TARGET MISSED), CI width cov 10.3 pts
- Document level: 95% NOT ATTAINABLE on this evidence; 98% NOT ATTAINABLE on this evidence.
- The dev slice has only 100 docs: its CIs are wide and it is not a separate test.

## (a) AUROC of P(correct), v1 vs v2 (95% CI)

Variants: `v1_prerule` = v1 as published (pre-rule outputs and labels); `v1_post` = v1 probabilities scored on the post-rule labels, matched rows only (separates the rule effect from the feature effect; rule-filled fields have no v1 probability and drop out); `v2_matched` = v2 on the same rows; `v2_all` = v2 on all its emitted rows; `abl_v2feat_LR` = ablation, the default LR model on the v2 features (model-family effect); `abl_v1feat_samemodel` = ablation, v1 features only with the chosen v2 model on post-rule labels (extra-feature effect). Ablations are diagnostic and feed no selection.


**slice `all`** (500 docs)

| group | wrong (v2) | v1_prerule | v1_post | v2_matched | v2_all | abl_v2feat_LR | abl_v1feat_samemodel |
|---|---|---|---|---|---|---|---|
| header.awb_number | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.buyer_name | 1 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.carrier | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.consignee_name | 2 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.currency | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.destination_airport | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.gross_weight_kg | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.hawb | 2 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.invoice_date | 6 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| header.invoice_number | 4 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.mawb | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.origin_airport | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.pieces | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.ship_to_name | 3 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.shipper_name | 2 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.supplier_name | 6 | 0.512 [0.265, 0.825] | 0.512 [0.265, 0.825] | 0.911 [0.849, 0.974] | 0.911 [0.849, 0.974] | 0.212 [0.039, 0.449] | 0.765 [0.561, 0.966] |
| header.total_amount | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header_all | 26 | 0.876 [0.785, 0.959] | 0.876 [0.785, 0.959] | 0.976 [0.949, 0.993] | 0.976 [0.949, 0.993] | 0.798 [0.652, 0.925] | 0.920 [0.856, 0.973] |
| row.customer_part_number | 645 | 0.334 [0.271, 0.406] | 0.460 [0.391, 0.535] | 0.645 [0.553, 0.727] | 0.645 [0.553, 0.727] | 0.617 [0.543, 0.688] | 0.455 [0.381, 0.534] |
| row.purchase_order | 468 | 0.707 [0.642, 0.765] | 0.712 [0.649, 0.768] | 0.719 [0.642, 0.787] | 0.700 [0.622, 0.770] | 0.743 [0.678, 0.801] | 0.739 [0.678, 0.794] |
| row.quantity | 727 | 0.578 [0.531, 0.625] | 0.578 [0.531, 0.625] | 0.642 [0.569, 0.710] | 0.642 [0.569, 0.710] | 0.673 [0.617, 0.726] | 0.640 [0.584, 0.690] |
| row.supplier_part_number | 675 | 0.721 [0.670, 0.780] | 0.721 [0.670, 0.780] | 0.935 [0.913, 0.954] | 0.935 [0.913, 0.954] | 0.892 [0.854, 0.925] | 0.785 [0.735, 0.835] |

**slice `dev`** (100 docs)

| group | wrong (v2) | v1_prerule | v1_post | v2_matched | v2_all | abl_v2feat_LR | abl_v1feat_samemodel |
|---|---|---|---|---|---|---|---|
| header.awb_number | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.buyer_name | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.carrier | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.consignee_name | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.currency | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.destination_airport | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.gross_weight_kg | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.hawb | 2 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.invoice_date | 2 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.invoice_number | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.mawb | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.origin_airport | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.pieces | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.ship_to_name | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.shipper_name | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.supplier_name | 1 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.total_amount | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header_all | 5 | 0.775 [0.488, 0.999] | 0.775 [0.488, 0.999] | 0.928 [0.777, 1.000] | 0.929 [0.779, 1.000] | 0.767 [0.388, 0.998] | 0.924 [0.758, 1.000] |
| row.customer_part_number | 146 | 0.427 [0.267, 0.622] | 0.566 [0.417, 0.730] | 0.670 [0.464, 0.826] | 0.670 [0.464, 0.826] | 0.651 [0.479, 0.785] | 0.509 [0.348, 0.686] |
| row.purchase_order | 121 | 0.768 [0.675, 0.844] | 0.768 [0.675, 0.844] | 0.806 [0.688, 0.891] | 0.802 [0.692, 0.890] | 0.789 [0.699, 0.853] | 0.794 [0.716, 0.863] |
| row.quantity | 172 | 0.613 [0.511, 0.706] | 0.613 [0.511, 0.706] | 0.739 [0.606, 0.838] | 0.739 [0.606, 0.838] | 0.696 [0.567, 0.794] | 0.719 [0.627, 0.804] |
| row.supplier_part_number | 150 | 0.763 [0.665, 0.885] | 0.763 [0.665, 0.885] | 0.951 [0.903, 0.980] | 0.951 [0.903, 0.980] | 0.918 [0.847, 0.961] | 0.824 [0.755, 0.893] |

**slice `train`** (400 docs)

| group | wrong (v2) | v1_prerule | v1_post | v2_matched | v2_all | abl_v2feat_LR | abl_v1feat_samemodel |
|---|---|---|---|---|---|---|---|
| header.awb_number | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.buyer_name | 1 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.carrier | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.consignee_name | 2 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.currency | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.destination_airport | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.gross_weight_kg | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.hawb | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.invoice_date | 4 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.invoice_number | 4 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.mawb | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.origin_airport | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.pieces | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.ship_to_name | 3 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.shipper_name | 2 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.supplier_name | 5 | 0.557 [0.268, 0.953] | 0.557 [0.268, 0.953] | 0.914 [0.841, 1.000] | 0.914 [0.841, 1.000] | 0.225 [0.006, 0.512] | 0.740 [0.536, 0.994] |
| header.total_amount | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header_all | 21 | 0.900 [0.805, 0.978] | 0.900 [0.805, 0.978] | 0.986 [0.976, 0.995] | 0.987 [0.976, 0.995] | 0.805 [0.637, 0.947] | 0.920 [0.854, 0.979] |
| row.customer_part_number | 499 | 0.317 [0.250, 0.391] | 0.429 [0.353, 0.513] | 0.636 [0.537, 0.742] | 0.636 [0.537, 0.742] | 0.610 [0.532, 0.694] | 0.443 [0.364, 0.530] |
| row.purchase_order | 347 | 0.684 [0.602, 0.754] | 0.690 [0.612, 0.759] | 0.691 [0.601, 0.770] | 0.669 [0.580, 0.752] | 0.727 [0.657, 0.799] | 0.717 [0.646, 0.785] |
| row.quantity | 555 | 0.573 [0.514, 0.623] | 0.573 [0.514, 0.623] | 0.613 [0.529, 0.693] | 0.613 [0.529, 0.693] | 0.666 [0.602, 0.726] | 0.614 [0.544, 0.675] |
| row.supplier_part_number | 525 | 0.709 [0.649, 0.776] | 0.709 [0.649, 0.776] | 0.931 [0.904, 0.952] | 0.931 [0.904, 0.952] | 0.885 [0.841, 0.921] | 0.772 [0.709, 0.830] |

**slice `invoice_only`** (400 docs)

| group | wrong (v2) | v1_prerule | v1_post | v2_matched | v2_all | abl_v2feat_LR | abl_v1feat_samemodel |
|---|---|---|---|---|---|---|---|
| header.awb_number | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.buyer_name | 1 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.currency | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.invoice_date | 6 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| header.invoice_number | 4 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.ship_to_name | 3 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.supplier_name | 6 | 0.512 [0.265, 0.825] | 0.512 [0.265, 0.825] | 0.911 [0.849, 0.974] | 0.911 [0.849, 0.974] | 0.212 [0.039, 0.449] | 0.765 [0.561, 0.966] |
| header.total_amount | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header_all | 20 | 0.852 [0.729, 0.960] | 0.852 [0.729, 0.960] | 0.987 [0.977, 0.996] | 0.987 [0.977, 0.996] | 0.754 [0.572, 0.912] | 0.913 [0.825, 0.983] |
| row.customer_part_number | 645 | 0.334 [0.271, 0.406] | 0.460 [0.391, 0.535] | 0.645 [0.553, 0.727] | 0.645 [0.553, 0.727] | 0.617 [0.543, 0.688] | 0.455 [0.381, 0.534] |
| row.purchase_order | 468 | 0.707 [0.642, 0.765] | 0.712 [0.649, 0.768] | 0.719 [0.642, 0.787] | 0.700 [0.622, 0.770] | 0.743 [0.678, 0.801] | 0.739 [0.678, 0.794] |
| row.quantity | 727 | 0.578 [0.531, 0.625] | 0.578 [0.531, 0.625] | 0.642 [0.569, 0.710] | 0.642 [0.569, 0.710] | 0.673 [0.617, 0.726] | 0.640 [0.584, 0.690] |
| row.supplier_part_number | 675 | 0.721 [0.670, 0.780] | 0.721 [0.670, 0.780] | 0.935 [0.913, 0.954] | 0.935 [0.913, 0.954] | 0.892 [0.854, 0.925] | 0.785 [0.735, 0.835] |

**slice `waybill_only`** (100 docs)

| group | wrong (v2) | v1_prerule | v1_post | v2_matched | v2_all | abl_v2feat_LR | abl_v1feat_samemodel |
|---|---|---|---|---|---|---|---|
| header.carrier | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.consignee_name | 2 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.destination_airport | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.gross_weight_kg | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.hawb | 2 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.mawb | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.origin_airport | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.pieces | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.shipper_name | 2 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header_all | 6 | 0.936 [0.843, 0.996] | 0.936 [0.843, 0.996] | 0.950 [0.858, 0.999] | 0.950 [0.860, 0.999] | 0.962 [0.897, 0.997] | 0.974 [0.927, 0.999] |

**slice `scanned`** (193 docs)

| group | wrong (v2) | v1_prerule | v1_post | v2_matched | v2_all | abl_v2feat_LR | abl_v1feat_samemodel |
|---|---|---|---|---|---|---|---|
| header.awb_number | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.buyer_name | 1 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.carrier | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.consignee_name | 1 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.currency | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.destination_airport | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.gross_weight_kg | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.hawb | 1 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.invoice_date | 5 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| header.invoice_number | 2 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.mawb | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.origin_airport | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.pieces | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.ship_to_name | 1 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.shipper_name | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.supplier_name | 3 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.total_amount | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header_all | 14 | 0.837 [0.682, 0.963] | 0.837 [0.682, 0.963] | 0.963 [0.908, 0.995] | 0.964 [0.909, 0.995] | 0.817 [0.630, 0.965] | 0.890 [0.761, 0.995] |
| row.customer_part_number | 313 | 0.294 [0.212, 0.389] | 0.400 [0.299, 0.507] | 0.565 [0.433, 0.684] | 0.565 [0.433, 0.684] | 0.577 [0.463, 0.688] | 0.479 [0.371, 0.591] |
| row.purchase_order | 243 | 0.656 [0.585, 0.722] | 0.656 [0.585, 0.722] | 0.730 [0.609, 0.827] | 0.734 [0.619, 0.832] | 0.730 [0.638, 0.825] | 0.730 [0.650, 0.796] |
| row.quantity | 354 | 0.505 [0.444, 0.568] | 0.505 [0.444, 0.568] | 0.643 [0.539, 0.734] | 0.643 [0.539, 0.734] | 0.645 [0.555, 0.721] | 0.661 [0.580, 0.732] |
| row.supplier_part_number | 328 | 0.700 [0.628, 0.783] | 0.700 [0.628, 0.783] | 0.934 [0.904, 0.958] | 0.934 [0.904, 0.958] | 0.864 [0.790, 0.926] | 0.810 [0.759, 0.856] |

**slice `digital`** (307 docs)

| group | wrong (v2) | v1_prerule | v1_post | v2_matched | v2_all | abl_v2feat_LR | abl_v1feat_samemodel |
|---|---|---|---|---|---|---|---|
| header.awb_number | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.buyer_name | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.carrier | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.consignee_name | 1 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.currency | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.destination_airport | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.gross_weight_kg | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.hawb | 1 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.invoice_date | 1 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.invoice_number | 2 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.mawb | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.origin_airport | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.pieces | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.ship_to_name | 2 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.shipper_name | 2 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.supplier_name | 3 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header.total_amount | 0 | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors | n<5 errors |
| header_all | 12 | 0.907 [0.780, 0.996] | 0.907 [0.780, 0.996] | 0.987 [0.974, 0.998] | 0.987 [0.974, 0.998] | 0.768 [0.520, 0.995] | 0.944 [0.886, 0.991] |
| row.customer_part_number | 332 | 0.306 [0.234, 0.386] | 0.470 [0.375, 0.576] | 0.678 [0.550, 0.802] | 0.678 [0.550, 0.802] | 0.644 [0.531, 0.743] | 0.424 [0.328, 0.534] |
| row.purchase_order | 225 | 0.691 [0.580, 0.784] | 0.701 [0.594, 0.787] | 0.669 [0.566, 0.754] | 0.642 [0.545, 0.727] | 0.748 [0.669, 0.812] | 0.716 [0.624, 0.794] |
| row.quantity | 373 | 0.580 [0.515, 0.644] | 0.580 [0.515, 0.644] | 0.598 [0.488, 0.697] | 0.598 [0.488, 0.697] | 0.680 [0.603, 0.745] | 0.597 [0.518, 0.670] |
| row.supplier_part_number | 347 | 0.704 [0.633, 0.789] | 0.704 [0.633, 0.789] | 0.935 [0.904, 0.959] | 0.935 [0.904, 0.959] | 0.916 [0.886, 0.942] | 0.755 [0.676, 0.832] |

## (b) Per-field-type tau, target 98% precision, NESTED (95% CI, %)

tau of fold k chosen on the other two folds' OOF predictions, applied to fold k, pooled. `in-sample` = tau chosen on the evaluated rows (optimistic, reference only). CIs condition on the taus.


**slice `all`**

| population | fields | precision | coverage | review rate | error recall | in-sample precision / coverage |
|---|---|---|---|---|---|---|
| header | 3907 | 99.4 [99.1, 99.6] | 100.0 [99.9, 100.0] | 0.0 [0.0, 0.1] | 3.8 [0.0, 12.5] | 99.3 [99.1, 99.6] / cov 100.0 [100.0, 100.0] |
| supplier_part_number | 4925 | 98.0 [96.8, 98.8] | 74.3 [71.4, 77.1] | 25.7 [22.9, 28.6] | 88.9 [82.9, 93.3] | 98.0 [96.9, 98.8] / cov 76.5 [73.7, 79.2] |
| customer_part_number | 2080 | 0.0 [0.0, 0.0] | 3.5 [1.0, 6.4] | 96.5 [93.6, 99.0] | 88.8 [80.4, 96.5] | n/a [n/a, n/a] / cov 0.0 [0.0, 0.0] |
| purchase_order | 2736 | n/a [n/a, n/a] | 0.0 [0.0, 0.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] / cov 0.1 [0.0, 0.5] |
| quantity | 4925 | 72.3 [46.7, 92.0] | 1.3 [0.4, 2.5] | 98.7 [97.5, 99.6] | 97.5 [94.2, 99.7] | 100.0 [100.0, 100.0] / cov 0.2 [0.1, 0.4] |
| all_per_type_tau | 18573 | 97.5 [96.4, 98.5] | 41.5 [39.7, 43.4] | 58.5 [56.6, 60.3] | 92.5 [89.3, 95.2] | 98.7 [98.1, 99.1] / cov 41.4 [39.6, 43.4] |
| all_global_tau | 18573 | 97.8 [96.4, 98.8] | 40.6 [38.2, 43.2] | 59.4 [56.8, 61.8] | 93.4 [89.2, 96.4] | 98.0 [96.7, 98.9] / cov 39.1 [37.0, 41.6] |

**slice `dev`**

| population | fields | precision | coverage | review rate | error recall | in-sample precision / coverage |
|---|---|---|---|---|---|---|
| header | 789 | 99.4 [98.8, 99.9] | 100.0 [100.0, 100.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 99.4 [98.8, 99.9] / cov 100.0 [100.0, 100.0] |
| supplier_part_number | 920 | 98.5 [96.7, 99.8] | 73.3 [64.8, 80.9] | 26.7 [19.1, 35.2] | 93.3 [83.2, 99.3] | 98.5 [96.7, 99.8] / cov 73.8 [65.5, 81.4] |
| customer_part_number | 483 | 0.0 [0.0, 0.0] | 1.4 [0.0, 5.2] | 98.6 [94.8, 100.0] | 95.2 [83.0, 100.0] | n/a [n/a, n/a] / cov 0.0 [0.0, 0.0] |
| purchase_order | 532 | n/a [n/a, n/a] | 0.0 [0.0, 0.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | n/a [n/a, n/a] / cov 0.0 [0.0, 0.0] |
| quantity | 920 | 88.2 [85.7, 100.0] | 1.8 [0.1, 5.4] | 98.2 [94.6, 99.9] | 98.8 [95.6, 100.0] | 100.0 [100.0, 100.0] / cov 0.5 [0.1, 1.2] |
| all_per_type_tau | 3644 | 98.4 [97.0, 99.4] | 40.8 [36.3, 46.3] | 59.2 [53.7, 63.7] | 96.0 [91.3, 98.6] | 99.0 [98.0, 99.7] / cov 40.4 [36.0, 46.1] |
| all_global_tau | 3644 | 98.4 [97.1, 99.6] | 39.4 [34.0, 45.8] | 60.6 [54.2, 66.0] | 96.1 [91.7, 99.0] | 98.4 [97.0, 99.7] / cov 38.5 [33.2, 44.8] |

**slice `train`**

| population | fields | precision | coverage | review rate | error recall | in-sample precision / coverage |
|---|---|---|---|---|---|---|
| header | 3118 | 99.4 [99.1, 99.6] | 100.0 [99.9, 100.0] | 0.0 [0.0, 0.1] | 4.8 [0.0, 15.8] | 99.3 [99.0, 99.6] / cov 100.0 [100.0, 100.0] |
| supplier_part_number | 4005 | 97.8 [96.4, 98.7] | 74.6 [71.4, 77.6] | 25.4 [22.4, 28.6] | 87.6 [79.9, 93.0] | 97.9 [96.6, 98.8] / cov 77.1 [74.1, 80.0] |
| customer_part_number | 1597 | 0.0 [0.0, 0.0] | 4.1 [0.8, 7.8] | 95.9 [92.2, 99.2] | 87.0 [77.4, 96.9] | n/a [n/a, n/a] / cov 0.0 [0.0, 0.0] |
| purchase_order | 2204 | n/a [n/a, n/a] | 0.0 [0.0, 0.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] / cov 0.2 [0.0, 0.6] |
| quantity | 4005 | 66.7 [33.3, 100.0] | 1.2 [0.2, 2.4] | 98.8 [97.6, 99.8] | 97.1 [92.9, 100.0] | 100.0 [100.0, 100.0] / cov 0.2 [0.0, 0.3] |
| all_per_type_tau | 14929 | 97.3 [96.0, 98.4] | 41.6 [39.9, 43.6] | 58.4 [56.4, 60.1] | 91.5 [88.0, 94.7] | 98.6 [97.9, 99.1] / cov 41.6 [39.7, 43.8] |
| all_global_tau | 14929 | 97.6 [96.0, 98.8] | 40.9 [38.3, 43.7] | 59.1 [56.3, 61.7] | 92.6 [87.5, 96.3] | 97.9 [96.4, 99.0] / cov 39.3 [36.8, 41.8] |

**slice `invoice_only`**

| population | fields | precision | coverage | review rate | error recall | in-sample precision / coverage |
|---|---|---|---|---|---|---|
| header | 3028 | 99.4 [99.1, 99.6] | 100.0 [99.9, 100.0] | 0.0 [0.0, 0.1] | 5.0 [0.0, 16.7] | 99.3 [99.0, 99.6] / cov 100.0 [100.0, 100.0] |
| supplier_part_number | 4925 | 98.0 [96.8, 98.8] | 74.3 [71.4, 77.1] | 25.7 [22.9, 28.6] | 88.9 [82.9, 93.3] | 98.0 [96.9, 98.8] / cov 76.5 [73.7, 79.2] |
| customer_part_number | 2080 | 0.0 [0.0, 0.0] | 3.5 [1.0, 6.4] | 96.5 [93.6, 99.0] | 88.8 [80.4, 96.5] | n/a [n/a, n/a] / cov 0.0 [0.0, 0.0] |
| purchase_order | 2736 | n/a [n/a, n/a] | 0.0 [0.0, 0.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] / cov 0.1 [0.0, 0.5] |
| quantity | 4925 | 72.3 [46.7, 92.0] | 1.3 [0.4, 2.5] | 98.7 [97.5, 99.6] | 97.5 [94.2, 99.7] | 100.0 [100.0, 100.0] / cov 0.2 [0.1, 0.4] |
| all_per_type_tau | 17694 | 97.3 [96.1, 98.4] | 38.6 [37.0, 40.2] | 61.4 [59.8, 63.0] | 92.7 [89.8, 95.5] | 98.6 [98.0, 99.1] / cov 38.5 [36.9, 40.3] |
| all_global_tau | 17694 | 97.5 [95.9, 98.7] | 37.7 [35.3, 40.2] | 62.3 [59.8, 64.7] | 93.5 [89.3, 96.5] | 97.8 [96.3, 98.9] / cov 36.2 [34.0, 38.5] |

**slice `waybill_only`**

| population | fields | precision | coverage | review rate | error recall | in-sample precision / coverage |
|---|---|---|---|---|---|---|
| header | 879 | 99.3 [98.8, 99.8] | 100.0 [100.0, 100.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 99.3 [98.8, 99.8] / cov 100.0 [100.0, 100.0] |
| all_per_type_tau | 879 | 99.3 [98.8, 99.8] | 100.0 [100.0, 100.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 99.3 [98.8, 99.8] / cov 100.0 [100.0, 100.0] |
| all_global_tau | 879 | 99.8 [99.4, 100.0] | 97.8 [96.8, 98.8] | 2.2 [1.2, 3.2] | 66.7 [24.1, 100.0] | 99.8 [99.4, 100.0] / cov 98.2 [97.2, 99.1] |

**slice `scanned`**

| population | fields | precision | coverage | review rate | error recall | in-sample precision / coverage |
|---|---|---|---|---|---|---|
| header | 1510 | 99.1 [98.6, 99.5] | 100.0 [100.0, 100.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 99.1 [98.6, 99.5] / cov 100.0 [100.0, 100.0] |
| supplier_part_number | 1848 | 98.0 [96.9, 98.9] | 67.2 [62.3, 72.0] | 32.8 [28.0, 37.7] | 92.4 [87.1, 96.2] | 97.9 [96.8, 98.8] / cov 69.3 [64.7, 73.8] |
| customer_part_number | 876 | 0.0 [0.0, 0.0] | 5.3 [0.9, 11.2] | 94.7 [88.8, 99.1] | 85.3 [71.7, 97.1] | n/a [n/a, n/a] / cov 0.0 [0.0, 0.0] |
| purchase_order | 1045 | n/a [n/a, n/a] | 0.0 [0.0, 0.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | n/a [n/a, n/a] / cov 0.0 [0.0, 0.0] |
| quantity | 1848 | 100.0 [100.0, 100.0] | 0.2 [0.0, 0.5] | 99.8 [99.5, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] / cov 0.2 [0.0, 0.5] |
| all_per_type_tau | 7127 | 97.0 [95.3, 98.5] | 39.3 [36.5, 42.5] | 60.7 [57.5, 63.5] | 93.2 [89.7, 96.3] | 98.5 [97.9, 99.1] / cov 39.2 [36.3, 42.5] |
| all_global_tau | 7127 | 99.2 [98.6, 99.6] | 34.3 [31.1, 38.0] | 65.7 [62.0, 68.9] | 98.4 [97.0, 99.3] | 99.1 [98.5, 99.6] / cov 32.3 [29.0, 36.0] |

**slice `digital`**

| population | fields | precision | coverage | review rate | error recall | in-sample precision / coverage |
|---|---|---|---|---|---|---|
| header | 2397 | 99.5 [99.2, 99.8] | 100.0 [99.9, 100.0] | 0.0 [0.0, 0.1] | 8.3 [0.0, 28.6] | 99.5 [99.2, 99.8] / cov 100.0 [100.0, 100.0] |
| supplier_part_number | 3077 | 97.9 [96.3, 99.0] | 78.6 [75.1, 82.1] | 21.4 [17.9, 24.9] | 85.6 [74.8, 93.2] | 98.1 [96.4, 99.1] / cov 80.8 [77.5, 84.1] |
| customer_part_number | 1204 | 0.0 [0.0, 0.0] | 2.2 [0.0, 5.7] | 97.8 [94.3, 100.0] | 92.2 [81.3, 100.0] | n/a [n/a, n/a] / cov 0.0 [0.0, 0.0] |
| purchase_order | 1691 | n/a [n/a, n/a] | 0.0 [0.0, 0.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] / cov 0.2 [0.0, 0.8] |
| quantity | 3077 | 70.5 [41.2, 90.5] | 2.0 [0.5, 3.9] | 98.0 [96.1, 99.5] | 95.2 [89.1, 99.5] | 100.0 [100.0, 100.0] / cov 0.3 [0.1, 0.5] |
| all_per_type_tau | 11446 | 97.9 [96.3, 99.0] | 42.8 [40.6, 45.3] | 57.2 [54.7, 59.4] | 91.9 [86.7, 95.9] | 98.8 [97.9, 99.3] / cov 42.8 [40.5, 45.3] |
| all_global_tau | 11446 | 97.1 [95.0, 98.6] | 44.5 [41.3, 48.3] | 55.5 [51.7, 58.7] | 88.5 [80.8, 94.3] | 97.5 [95.6, 98.9] / cov 43.4 [40.4, 46.7] |

## Coverage at precision targets 95 / 98 / 99% (nested; all and dev)

Cell: nested coverage % [CI] ; nested precision % [CI] ; in-sample optimum coverage (point, optimistic).


**slice `all`**

| population | 95% | 98% | 99% |
|---|---|---|---|
| header | 100.0 [99.9, 100.0] ; 99.4 [99.1, 99.6] ; 100.0 | 100.0 [99.9, 100.0] ; 99.4 [99.1, 99.6] ; 100.0 | 100.0 [99.9, 100.0] ; 99.4 [99.1, 99.6] ; 100.0 |
| supplier_part_number | 88.0 [85.8, 90.2] ; 95.0 [93.3, 96.5] ; 88.7 | 74.3 [71.4, 77.1] ; 98.0 [96.8, 98.8] ; 76.5 | 45.6 [41.4, 49.9] ; 98.8 [98.2, 99.3] ; 49.8 |
| customer_part_number | 3.5 [1.0, 6.4] ; 0.0 [0.0, 0.0] ; 0.0 | 3.5 [1.0, 6.4] ; 0.0 [0.0, 0.0] ; 0.0 | 3.5 [1.0, 6.4] ; 0.0 [0.0, 0.0] ; 0.0 |
| purchase_order | 0.0 [0.0, 0.0] ; n/a [n/a, n/a] ; 0.1 | 0.0 [0.0, 0.0] ; n/a [n/a, n/a] ; 0.1 | 0.0 [0.0, 0.0] ; n/a [n/a, n/a] ; 0.1 |
| quantity | 1.3 [0.4, 2.5] ; 72.3 [46.7, 92.0] ; 0.2 | 1.3 [0.4, 2.5] ; 72.3 [46.7, 92.0] ; 0.2 | 1.3 [0.4, 2.5] ; 72.3 [46.7, 92.0] ; 0.2 |
| all_global_tau | 64.0 [61.2, 66.8] ; 95.1 [93.7, 96.4] ; 63.8 | 40.6 [38.2, 43.2] ; 97.8 [96.4, 98.8] ; 39.1 | 28.4 [26.4, 30.6] ; 98.8 [97.5, 99.7] ; 28.6 |

**slice `dev`**

| population | 95% | 98% | 99% |
|---|---|---|---|
| header | 100.0 [100.0, 100.0] ; 99.4 [98.8, 99.9] ; 100.0 | 100.0 [100.0, 100.0] ; 99.4 [98.8, 99.9] ; 100.0 | 100.0 [100.0, 100.0] ; 99.4 [98.8, 99.9] ; 100.0 |
| supplier_part_number | 86.6 [80.5, 91.9] ; 94.0 [89.8, 97.1] ; 85.0 | 73.3 [64.8, 80.9] ; 98.5 [96.7, 99.8] ; 77.4 | 41.6 [32.7, 50.5] ; 98.2 [95.6, 100.0] ; 43.9 |
| customer_part_number | 1.4 [0.0, 5.2] ; 0.0 [0.0, 0.0] ; 0.0 | 1.4 [0.0, 5.2] ; 0.0 [0.0, 0.0] ; 0.0 | 1.4 [0.0, 5.2] ; 0.0 [0.0, 0.0] ; 0.0 |
| purchase_order | 0.0 [0.0, 0.0] ; n/a [n/a, n/a] ; 0.4 | 0.0 [0.0, 0.0] ; n/a [n/a, n/a] ; 0.4 | 0.0 [0.0, 0.0] ; n/a [n/a, n/a] ; 0.4 |
| quantity | 1.8 [0.1, 5.4] ; 88.2 [85.7, 100.0] ; 1.2 | 1.8 [0.1, 5.4] ; 88.2 [85.7, 100.0] ; 1.2 | 1.8 [0.1, 5.4] ; 88.2 [85.7, 100.0] ; 1.2 |
| all_global_tau | 60.7 [54.5, 67.7] ; 95.6 [92.9, 97.6] ; 68.4 | 39.4 [34.0, 45.8] ; 98.4 [97.1, 99.6] ; 48.0 | 27.8 [23.0, 34.1] ; 99.2 [98.0, 100.0] ; 30.7 |

## (c) Document auto-accept (fully correct document)

Base rate of fully-correct docs (95% CI, %):

| slice | docs | base rate |
|---|---|---|
| all | 500 | 58.8 [54.4, 63.0] |
| dev | 100 | 59.0 [50.0, 68.0] |
| train | 400 | 58.8 [54.0, 63.3] |
| invoice_only | 400 | 50.0 [45.2, 55.0] |
| waybill_only | 100 | 94.0 [89.0, 98.0] |
| scanned | 193 | 51.8 [44.6, 59.1] |
| digital | 307 | 63.2 [57.7, 68.7] |

Target 95%: **NOT ATTAINABLE on this evidence** (nested precision < target or coverage < 5%); best in-sample precision at >= 5% coverage: 94.9% at coverage 11.8%.

| slice | docs | nested precision | nested coverage | in-sample |
|---|---|---|---|---|
| all | 500 | 90.2 [80.4, 97.8] | 8.2 [6.0, 10.6] | 100.0 [100.0, 100.0] / cov 1.0 [0.2, 2.0] |
| dev | 100 | 100.0 [100.0, 100.0] | 8.0 [3.0, 14.0] | n/a [n/a, n/a] / cov 0.0 [0.0, 0.0] |
| train | 400 | 87.9 [75.9, 97.3] | 8.2 [5.5, 11.0] | 100.0 [100.0, 100.0] / cov 1.2 [0.2, 2.5] |
| invoice_only | 400 | 84.6 [62.5, 100.0] | 3.2 [1.5, 5.2] | 100.0 [100.0, 100.0] / cov 1.2 [0.2, 2.5] |
| waybill_only | 100 | 92.9 [82.8, 100.0] | 28.0 [19.0, 37.0] | n/a [n/a, n/a] / cov 0.0 [0.0, 0.0] |
| scanned | 193 | 100.0 [100.0, 100.0] | 4.1 [1.6, 6.7] | 100.0 [100.0, 100.0] / cov 1.6 [0.0, 3.6] |
| digital | 307 | 87.9 [75.8, 97.2] | 10.7 [7.5, 14.7] | 100.0 [100.0, 100.0] / cov 0.7 [0.0, 1.6] |

Target 98%: **NOT ATTAINABLE on this evidence** (nested precision < target or coverage < 5%); best in-sample precision at >= 5% coverage: 94.9% at coverage 11.8%.

| slice | docs | nested precision | nested coverage | in-sample |
|---|---|---|---|---|
| all | 500 | 87.5 [57.1, 100.0] | 1.6 [0.6, 2.8] | 100.0 [100.0, 100.0] / cov 1.0 [0.2, 2.0] |
| dev | 100 | n/a [n/a, n/a] | 0.0 [0.0, 0.0] | n/a [n/a, n/a] / cov 0.0 [0.0, 0.0] |
| train | 400 | 87.5 [60.0, 100.0] | 2.0 [0.8, 3.5] | 100.0 [100.0, 100.0] / cov 1.2 [0.2, 2.5] |
| invoice_only | 400 | 87.5 [60.0, 100.0] | 2.0 [0.8, 3.5] | 100.0 [100.0, 100.0] / cov 1.2 [0.2, 2.5] |
| waybill_only | 100 | n/a [n/a, n/a] | 0.0 [0.0, 0.0] | n/a [n/a, n/a] / cov 0.0 [0.0, 0.0] |
| scanned | 193 | 100.0 [100.0, 100.0] | 1.6 [0.0, 3.6] | 100.0 [100.0, 100.0] / cov 1.6 [0.0, 3.6] |
| digital | 307 | 80.0 [33.3, 100.0] | 1.6 [0.3, 3.3] | 100.0 [100.0, 100.0] / cov 0.7 [0.0, 1.6] |

## (d) Calibration (ECE 15 equal-width / equal-mass bins)

| slice | group | n | base rate | log-loss | ECE w15 | ECE m15 |
|---|---|---|---|---|---|---|
| all | header | 3907 | 0.993 | 0.024 | 0.010 | 0.009 |
| all | supplier_part_number | 4925 | 0.863 | 0.202 | 0.039 | 0.030 |
| all | customer_part_number | 2080 | 0.690 | 0.576 | 0.094 | 0.083 |
| all | purchase_order | 2736 | 0.829 | 0.389 | 0.060 | 0.051 |
| all | quantity | 4925 | 0.852 | 0.393 | 0.034 | 0.040 |
| all | all_emitted | 18573 | 0.863 | 0.285 | 0.019 | 0.017 |
| all | doc | 500 | 0.588 | 0.404 | 0.056 | 0.047 |
| dev | header | 789 | 0.994 | 0.026 | 0.009 | 0.009 |
| dev | supplier_part_number | 920 | 0.837 | 0.203 | 0.056 | 0.048 |
| dev | customer_part_number | 483 | 0.698 | 0.523 | 0.083 | 0.118 |
| dev | purchase_order | 532 | 0.773 | 0.407 | 0.084 | 0.105 |
| dev | quantity | 920 | 0.813 | 0.406 | 0.048 | 0.076 |
| dev | all_emitted | 3644 | 0.837 | 0.288 | 0.030 | 0.041 |
| dev | doc | 100 | 0.590 | 0.422 | 0.127 | 0.115 |
| train | header | 3118 | 0.993 | 0.024 | 0.010 | 0.009 |
| train | supplier_part_number | 4005 | 0.869 | 0.202 | 0.040 | 0.026 |
| train | customer_part_number | 1597 | 0.688 | 0.592 | 0.107 | 0.102 |
| train | purchase_order | 2204 | 0.843 | 0.385 | 0.061 | 0.048 |
| train | quantity | 4005 | 0.861 | 0.391 | 0.043 | 0.047 |
| train | all_emitted | 14929 | 0.870 | 0.284 | 0.020 | 0.016 |
| train | doc | 400 | 0.588 | 0.400 | 0.058 | 0.051 |
| invoice_only | header | 3028 | 0.993 | 0.024 | 0.010 | 0.009 |
| invoice_only | supplier_part_number | 4925 | 0.863 | 0.202 | 0.039 | 0.030 |
| invoice_only | customer_part_number | 2080 | 0.690 | 0.576 | 0.094 | 0.083 |
| invoice_only | purchase_order | 2736 | 0.829 | 0.389 | 0.060 | 0.051 |
| invoice_only | quantity | 4925 | 0.852 | 0.393 | 0.034 | 0.040 |
| invoice_only | all_emitted | 17694 | 0.857 | 0.298 | 0.021 | 0.020 |
| invoice_only | doc | 400 | 0.500 | 0.456 | 0.068 | 0.053 |
| waybill_only | header | 879 | 0.993 | 0.026 | 0.008 | 0.008 |
| waybill_only | all_emitted | 879 | 0.993 | 0.026 | 0.008 | 0.008 |
| waybill_only | doc | 100 | 0.940 | 0.194 | 0.032 | 0.065 |
| scanned | header | 1510 | 0.991 | 0.031 | 0.011 | 0.009 |
| scanned | supplier_part_number | 1848 | 0.823 | 0.237 | 0.043 | 0.046 |
| scanned | customer_part_number | 876 | 0.643 | 0.663 | 0.123 | 0.156 |
| scanned | purchase_order | 1045 | 0.767 | 0.448 | 0.046 | 0.057 |
| scanned | quantity | 1848 | 0.808 | 0.455 | 0.047 | 0.053 |
| scanned | all_emitted | 7127 | 0.824 | 0.333 | 0.028 | 0.035 |
| scanned | doc | 193 | 0.518 | 0.409 | 0.104 | 0.070 |
| digital | header | 2397 | 0.995 | 0.020 | 0.009 | 0.009 |
| digital | supplier_part_number | 3077 | 0.887 | 0.181 | 0.042 | 0.022 |
| digital | customer_part_number | 1204 | 0.724 | 0.513 | 0.101 | 0.115 |
| digital | purchase_order | 1691 | 0.867 | 0.354 | 0.079 | 0.065 |
| digital | quantity | 3077 | 0.879 | 0.356 | 0.061 | 0.053 |
| digital | all_emitted | 11446 | 0.887 | 0.255 | 0.025 | 0.025 |
| digital | doc | 307 | 0.632 | 0.401 | 0.060 | 0.056 |

Reliability tables and risk-coverage curves: `reliability.csv`, `risk_coverage.csv` in the run folder (not reproduced here).


## Caveats

- CIs resample documents but condition on the selected taus; tau-selection variance is not in them.
- Model structure / kind were chosen on the same OOF predictions that are reported.
- The doc-level model stacks on field-level OOF probabilities (mild second-level leakage).
- Labels are post-rule; v1 published numbers are pre-rule, hence the matched variants.
- Population of field metrics = emitted non-null fields; null fields are not scored here (no null policy; the row null policy was rejected).
