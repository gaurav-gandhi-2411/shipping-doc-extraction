# Calibration dry run (DRY RUN, not results)

> **DRY RUN: not a result. Synthetic data and/or saved dev100 outputs used to exercise the mechanics; no number here may be quoted as a Phase 5 result.**


## (i) synthetic features, known generative truth

*DRY RUN*

rows 3600 (emitted non-null 3083), docs 300, rows with logprobs 3600

calibrators chosen: P(correct) = lr (log-loss {'lr': 0.5037542628372296, 'gbm': 0.509535338316646}), P(null) = lr (log-loss {'lr': 0.5037526166349288, 'gbm': 0.5000347975469621})


**P(correct | emitted), OOF**

| slice | n | base rate | log-loss | Brier | ECE w15 | ECE m15 |
|---|---|---|---|---|---|---|
| all | 3083 | 56.44% | 0.5038 | 0.1659 | 0.0161 | 0.0220 |
| digital | 3083 | 56.44% | 0.5038 | 0.1659 | 0.0161 | 0.0220 |

**P(gold null), all fields, OOF**

| slice | n | base rate | log-loss | Brier | ECE w15 | ECE m15 |
|---|---|---|---|---|---|---|
| all | 3600 | 30.44% | 0.5038 | 0.1665 | 0.0389 | 0.0393 |
| digital | 3600 | 30.44% | 0.5038 | 0.1665 | 0.0389 | 0.0393 |

**Null policy, field level**

```
{
 "emitted_fields": 3083,
 "nulled_by_policy": 871,
 "nulled_that_were_false_fills_fixed": 412,
 "nulled_that_were_correct_values_lost": 164,
 "nulled_that_were_wrong_not_null_gold": 295,
 "field_accuracy_raw": 0.48333333333333334,
 "field_accuracy_policy": 0.5522222222222222
}
```

**Review flag, population `final_nonnull_fields`** (target precision 0.98)


- `in_sample_all`: tau = 0.9788561399611376 (selected on 2212 fields)
  - all (2212 fields, 300 docs): precision_accepted 100.00% [100.00%, 100.00%], accepted_share 0.18% [0.04%, 0.37%], review_rate 99.82% [99.63%, 99.96%], error_recall 100.00% [100.00%, 100.00%], doc_flag_rate 100.00% [100.00%, 100.00%], field_accuracy 71.25% [69.39%, 73.12%]
  - digital (2212 fields, 300 docs): precision_accepted 100.00% [100.00%, 100.00%], accepted_share 0.18% [0.04%, 0.37%], review_rate 99.82% [99.63%, 99.96%], error_recall 100.00% [100.00%, 100.00%], doc_flag_rate 100.00% [100.00%, 100.00%], field_accuracy 71.25% [69.39%, 73.12%]

**Review flag, population `all_final_fields_nulls_use_p_null`** (target precision 0.98)


- `in_sample_all`: tau = 0.9788561399611376 (selected on 3600 fields)
  - all (3600 fields, 300 docs): precision_accepted 100.00% [100.00%, 100.00%], accepted_share 0.11% [0.03%, 0.22%], review_rate 99.89% [99.78%, 99.97%], error_recall 100.00% [100.00%, 100.00%], doc_flag_rate 100.00% [100.00%, 100.00%], field_accuracy 55.22% [53.47%, 56.94%]
  - digital (3600 fields, 300 docs): precision_accepted 100.00% [100.00%, 100.00%], accepted_share 0.11% [0.03%, 0.22%], review_rate 99.89% [99.78%, 99.97%], error_recall 100.00% [100.00%, 100.00%], doc_flag_rate 100.00% [100.00%, 100.00%], field_accuracy 55.22% [53.47%, 56.94%]

**Recovery of the known generative truth (synthetic only)**

```
{
 "mean_abs_err_p_correct": 0.03251236759465131,
 "mean_abs_err_p_null": 0.05616946680973702,
 "corr_p_correct": 0.989396206257153
}
```

## (ii) dev100 saved outputs, no logprobs

*DRY RUN*

rows 4503 (emitted non-null 3629), docs 100, rows with logprobs 0

calibrators chosen: P(correct) = lr (log-loss {'lr': 0.41115983733338296, 'gbm': 0.4797347139108398}), P(null) = lr (log-loss {'lr': 0.19012263395813986, 'gbm': 0.25112780732455237})


**P(correct | emitted), OOF**

| slice | n | base rate | log-loss | Brier | ECE w15 | ECE m15 |
|---|---|---|---|---|---|---|
| all | 3629 | 82.20% | 0.4112 | 0.1264 | 0.0617 | 0.0543 |
| dev_oof | 3629 | 82.20% | 0.4112 | 0.1264 | 0.0617 | 0.0543 |
| scanned | 1426 | 75.95% | 0.4566 | 0.1489 | 0.0816 | 0.0857 |
| digital | 2203 | 86.25% | 0.3818 | 0.1118 | 0.0944 | 0.0826 |
| invoice_only | 3442 | 81.29% | 0.4301 | 0.1326 | 0.0646 | 0.0660 |
| waybill_only | 187 | 98.93% | 0.0624 | 0.0115 | 0.0114 | 0.0166 |

**P(gold null), all fields, OOF**

| slice | n | base rate | log-loss | Brier | ECE w15 | ECE m15 |
|---|---|---|---|---|---|---|
| all | 4503 | 17.43% | 0.1901 | 0.0517 | 0.0387 | 0.0321 |
| dev_oof | 4503 | 17.43% | 0.1901 | 0.0517 | 0.0387 | 0.0321 |
| scanned | 1716 | 14.04% | 0.1699 | 0.0465 | 0.0508 | 0.0272 |
| digital | 2787 | 19.52% | 0.2026 | 0.0549 | 0.0536 | 0.0507 |
| invoice_only | 4296 | 18.18% | 0.1984 | 0.0539 | 0.0407 | 0.0334 |
| waybill_only | 207 | 1.93% | 0.0189 | 0.0057 | 0.0086 | 0.0029 |

**Null policy, field level**

```
{
 "emitted_fields": 3629,
 "nulled_by_policy": 11,
 "nulled_that_were_false_fills_fixed": 0,
 "nulled_that_were_correct_values_lost": 0,
 "nulled_that_were_wrong_not_null_gold": 11,
 "field_accuracy_raw": 0.8232289584721297,
 "field_accuracy_policy": 0.8232289584721297
}
```

null policy `header_and_rows` on `all` (100 docs): OVERALL 0.7678 -> 0.7678, delta +0.0000 [+0.0000, +0.0000] (paired doc bootstrap, 2000 resamples, seed 42); false fills raw {"redaction": {"null_fields": 7, "filled": 0}, "absent_line": {"null_fields": 27, "filled": 0}} -> policy {"redaction": {"null_fields": 7, "filled": 0}, "absent_line": {"null_fields": 27, "filled": 0}}

null policy `header_and_rows` on `dev_oof` (100 docs): OVERALL 0.7678 -> 0.7678, delta +0.0000 [+0.0000, +0.0000] (paired doc bootstrap, 2000 resamples, seed 42); false fills raw {"redaction": {"null_fields": 7, "filled": 0}, "absent_line": {"null_fields": 27, "filled": 0}} -> policy {"redaction": {"null_fields": 7, "filled": 0}, "absent_line": {"null_fields": 27, "filled": 0}}

null policy `header_only` on `all` (100 docs): OVERALL 0.7678 -> 0.7678, delta +0.0000 [+0.0000, +0.0000] (paired doc bootstrap, 2000 resamples, seed 42); false fills raw {"redaction": {"null_fields": 7, "filled": 0}, "absent_line": {"null_fields": 27, "filled": 0}} -> policy {"redaction": {"null_fields": 7, "filled": 0}, "absent_line": {"null_fields": 27, "filled": 0}}

null policy `header_only` on `dev_oof` (100 docs): OVERALL 0.7678 -> 0.7678, delta +0.0000 [+0.0000, +0.0000] (paired doc bootstrap, 2000 resamples, seed 42); false fills raw {"redaction": {"null_fields": 7, "filled": 0}, "absent_line": {"null_fields": 27, "filled": 0}} -> policy {"redaction": {"null_fields": 7, "filled": 0}, "absent_line": {"null_fields": 27, "filled": 0}}

**Review flag, population `final_nonnull_fields`** (target precision 0.98)


- `in_sample_all`: tau = 0.9686394703964417 (selected on 3618 fields)
  - all (3618 fields, 100 docs): precision_accepted 98.09% [96.17%, 99.56%], accepted_share 25.98% [20.94%, 32.54%], review_rate 74.02% [67.46%, 79.06%], error_recall 97.17% [92.84%, 99.40%], doc_flag_rate 87.00% [80.00%, 93.00%], field_accuracy 82.45% [75.98%, 88.61%]
  - dev_oof (3618 fields, 100 docs): precision_accepted 98.09% [96.17%, 99.56%], accepted_share 25.98% [20.94%, 32.54%], review_rate 74.02% [67.46%, 79.06%], error_recall 97.17% [92.84%, 99.40%], doc_flag_rate 87.00% [80.00%, 93.00%], field_accuracy 82.45% [75.98%, 88.61%]
  - scanned (1418 fields, 43 docs): precision_accepted 98.52% [97.22%, 99.69%], accepted_share 23.77% [18.04%, 33.06%], review_rate 76.23% [66.94%, 81.96%], error_recall 98.51% [96.60%, 99.60%], doc_flag_rate 90.70% [81.40%, 97.67%], field_accuracy 76.38% [64.75%, 87.04%]
  - digital (2200 fields, 57 docs): precision_accepted 97.84% [95.09%, 100.00%], accepted_share 27.41% [20.43%, 36.64%], review_rate 72.59% [63.36%, 79.57%], error_recall 95.67% [87.24%, 100.00%], doc_flag_rate 84.21% [73.68%, 92.98%], field_accuracy 86.36% [79.32%, 92.75%]
  - invoice_only (3431 fields, 77 docs): precision_accepted 97.91% [95.65%, 99.69%], accepted_share 22.27% [17.59%, 28.54%], review_rate 77.73% [71.46%, 82.41%], error_recall 97.47% [93.76%, 99.61%], doc_flag_rate 98.70% [96.10%, 100.00%], field_accuracy 81.55% [74.18%, 87.76%]
  - waybill_only (187 fields, 23 docs): precision_accepted 98.86% [97.13%, 100.00%], accepted_share 94.12% [91.58%, 96.74%], review_rate 5.88% [3.26%, 8.42%], error_recall 0.00% [0.00%, 0.00%], doc_flag_rate 47.83% [26.09%, 69.57%], field_accuracy 98.93% [97.27%, 100.00%]

**Review flag, population `all_final_fields_nulls_use_p_null`** (target precision 0.98)


- `in_sample_all`: tau = 0.9708276403939791 (selected on 4503 fields)
  - all (4503 fields, 100 docs): precision_accepted 98.01% [97.10%, 98.90%], accepted_share 24.56% [20.43%, 29.74%], review_rate 75.44% [70.26%, 79.57%], error_recall 97.24% [95.25%, 98.60%], doc_flag_rate 97.00% [93.00%, 100.00%], field_accuracy 82.32% [76.19%, 88.23%]
  - dev_oof (4503 fields, 100 docs): precision_accepted 98.01% [97.10%, 98.90%], accepted_share 24.56% [20.43%, 29.74%], review_rate 75.44% [70.26%, 79.57%], error_recall 97.24% [95.25%, 98.60%], doc_flag_rate 97.00% [93.00%, 100.00%], field_accuracy 82.32% [76.19%, 88.23%]
  - scanned (1716 fields, 43 docs): precision_accepted 97.23% [94.97%, 99.06%], accepted_share 23.14% [18.03%, 30.37%], review_rate 76.86% [69.63%, 81.97%], error_recall 97.29% [93.82%, 99.09%], doc_flag_rate 100.00% [100.00%, 100.00%], field_accuracy 76.34% [65.16%, 86.13%]
  - digital (2787 fields, 57 docs): precision_accepted 98.45% [97.56%, 99.36%], accepted_share 25.44% [19.79%, 32.27%], review_rate 74.56% [67.73%, 80.21%], error_recall 97.18% [94.12%, 98.93%], doc_flag_rate 94.74% [87.72%, 100.00%], field_accuracy 86.01% [78.95%, 92.56%]
  - invoice_only (4296 fields, 77 docs): precision_accepted 97.85% [96.74%, 98.87%], accepted_share 21.65% [17.73%, 26.27%], review_rate 78.35% [73.73%, 82.27%], error_recall 97.43% [95.45%, 98.81%], doc_flag_rate 100.00% [100.00%, 100.00%], field_accuracy 81.89% [74.88%, 87.78%]
  - waybill_only (207 fields, 23 docs): precision_accepted 98.86% [97.13%, 100.00%], accepted_share 85.02% [81.64%, 88.41%], review_rate 14.98% [11.59%, 18.36%], error_recall 88.89% [76.47%, 100.00%], doc_flag_rate 86.96% [73.91%, 100.00%], field_accuracy 91.30% [86.96%, 95.17%]
