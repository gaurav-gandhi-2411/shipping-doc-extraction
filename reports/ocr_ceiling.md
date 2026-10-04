# OCR recall ceiling at three levels (train+dev)

**Every number is UNVERIFIED** (computed by `scripts/ocr_ceiling.py`; the verifier will recompute). Corpus: 500 docs, 3908 non-null header values, 14769 non-null row values (percent of gold values whose text is recoverable from the cached PaddleOCR text). Locator `src/shipdoc/locate.py`, fuzzy T = 87.0 (dates 91.0); T was chosen on train+dev (see `reports/provenance.md`), so (c) is measured on the data that picked T: the false-match side was controlled by the negatives, but the figure is not out-of-sample.

Levels are cumulative: (a) raw gold string is a substring of the doc OCR text (pages joined); (b) = (a) or scorer-normalized locator match (`same()` after parsing printed dates/numbers/airport+city/label prefixes); (c) = (b) or fuzzy match at T. Decomposition: (b)-(a) = formatting, (c)-(b) = near-miss recognition errors, 100-(c) = unrecoverable (value absent or too badly misread to match).

## 1. Header fields, pooled

| field | split | n | (a) exact % | (b) normalized % | (c) +fuzzy % | b-a formatting | c-b near-miss | 100-c unrecoverable |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| header, all fields | scanned | 1511 | 68.6 | 97.7 | 99.6 | 29.1 | 1.9 | 0.4 |
| header, all fields | digital | 2397 | 77.0 | 98.3 | 100.0 | 21.3 | 1.7 | 0.0 |
| header, all fields | all | 3908 | 73.8 | 98.1 | 99.8 | 24.3 | 1.8 | 0.2 |
| header, invoice | scanned | 1210 | 64.7 | 97.7 | 99.8 | 33.0 | 2.1 | 0.2 |
| header, invoice | digital | 1819 | 73.9 | 98.1 | 100.0 | 24.2 | 1.9 | 0.0 |
| header, invoice | all | 3029 | 70.2 | 98.0 | 99.9 | 27.7 | 2.0 | 0.1 |
| header, waybill | scanned | 301 | 84.4 | 98.0 | 98.7 | 13.6 | 0.7 | 1.3 |
| header, waybill | digital | 578 | 86.9 | 98.8 | 100.0 | 11.9 | 1.2 | 0.0 |
| header, waybill | all | 879 | 86.0 | 98.5 | 99.5 | 12.5 | 1.0 | 0.5 |

### Reconciliation with the prior exact-substring measurement

| quantity | prior (scanned / digital) | this run (a), case-sensitive | this run (a), case/space-insensitive |
|---|---|---|---|
| header, pooled | 74.8 / 84.2 | 68.6 / 77.0 | 69.1 / 79.8 |
| supplier_part_number | 88.9 / 96.5 | 88.9 / 96.5 | 89.6 / 96.5 |
| header, pooled, excluding invoice_date | 74.8 / 84.2 | 74.8 / 84.2 | 75.3 / 87.2 |

Reading: supplier_part_number reproduces the prior numbers exactly (88.9 / 96.5), and the prior pooled header figure is reproduced exactly once invoice_date is left out (gold dates are ISO, so a raw substring test is meaningless for them: 11.5% hit). Including invoice_date the pooled exact-substring figure is lower (68.6 / 77.0). The prior script is not in the repo; that leaving out invoice_date explains the gap is inferred from this exact numeric match, not from its source.

## 2. Per header field

| field | split | n | (a) exact % | (b) normalized % | (c) +fuzzy % | b-a formatting | c-b near-miss | 100-c unrecoverable |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| awb_number | scanned | 111 | 99.1 | 99.1 | 100.0 | 0.0 | 0.9 | 0.0 |
| awb_number | digital | 153 | 100.0 | 100.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| awb_number | all | 264 | 99.6 | 99.6 | 100.0 | 0.0 | 0.4 | 0.0 |
| buyer_name | scanned | 159 | 85.5 | 98.1 | 100.0 | 12.6 | 1.9 | 0.0 |
| buyer_name | digital | 241 | 97.5 | 100.0 | 100.0 | 2.5 | 0.0 | 0.0 |
| buyer_name | all | 400 | 92.8 | 99.2 | 100.0 | 6.5 | 0.8 | 0.0 |
| currency | scanned | 159 | 100.0 | 100.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| currency | digital | 241 | 100.0 | 100.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| currency | all | 400 | 100.0 | 100.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| invoice_date | scanned | 153 | 13.7 | 100.0 | 100.0 | 86.3 | 0.0 | 0.0 |
| invoice_date | digital | 231 | 10.0 | 100.0 | 100.0 | 90.0 | 0.0 | 0.0 |
| invoice_date | all | 384 | 11.5 | 100.0 | 100.0 | 88.5 | 0.0 | 0.0 |
| invoice_number | scanned | 151 | 99.3 | 99.3 | 99.3 | 0.0 | 0.0 | 0.7 |
| invoice_number | digital | 230 | 100.0 | 100.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| invoice_number | all | 381 | 99.7 | 99.7 | 99.7 | 0.0 | 0.0 | 0.3 |
| ship_to_name | scanned | 159 | 86.2 | 98.7 | 100.0 | 12.6 | 1.3 | 0.0 |
| ship_to_name | digital | 241 | 99.2 | 99.2 | 100.0 | 0.0 | 0.8 | 0.0 |
| ship_to_name | all | 400 | 94.0 | 99.0 | 100.0 | 5.0 | 1.0 | 0.0 |
| supplier_name | scanned | 159 | 43.4 | 91.2 | 100.0 | 47.8 | 8.8 | 0.0 |
| supplier_name | digital | 241 | 91.7 | 92.5 | 100.0 | 0.8 | 7.5 | 0.0 |
| supplier_name | all | 400 | 72.5 | 92.0 | 100.0 | 19.5 | 8.0 | 0.0 |
| total_amount | scanned | 159 | 0.6 | 95.6 | 99.4 | 95.0 | 3.8 | 0.6 |
| total_amount | digital | 241 | 0.8 | 94.2 | 100.0 | 93.4 | 5.8 | 0.0 |
| total_amount | all | 400 | 0.8 | 94.8 | 99.8 | 94.0 | 5.0 | 0.2 |
| carrier | scanned | 34 | 0.0 | 100.0 | 100.0 | 100.0 | 0.0 | 0.0 |
| carrier | digital | 66 | 0.0 | 100.0 | 100.0 | 100.0 | 0.0 | 0.0 |
| carrier | all | 100 | 0.0 | 100.0 | 100.0 | 100.0 | 0.0 | 0.0 |
| consignee_name | scanned | 34 | 85.3 | 94.1 | 100.0 | 8.8 | 5.9 | 0.0 |
| consignee_name | digital | 66 | 95.5 | 97.0 | 100.0 | 1.5 | 3.0 | 0.0 |
| consignee_name | all | 100 | 92.0 | 96.0 | 100.0 | 4.0 | 4.0 | 0.0 |
| destination_airport | scanned | 34 | 97.1 | 97.1 | 97.1 | 0.0 | 0.0 | 2.9 |
| destination_airport | digital | 66 | 100.0 | 100.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| destination_airport | all | 100 | 99.0 | 99.0 | 99.0 | 0.0 | 0.0 | 1.0 |
| gross_weight_kg | scanned | 34 | 100.0 | 100.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| gross_weight_kg | digital | 66 | 100.0 | 100.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| gross_weight_kg | all | 100 | 100.0 | 100.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| hawb | scanned | 29 | 100.0 | 100.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| hawb | digital | 50 | 100.0 | 100.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| hawb | all | 79 | 100.0 | 100.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| mawb | scanned | 34 | 100.0 | 100.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| mawb | digital | 66 | 100.0 | 100.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| mawb | all | 100 | 100.0 | 100.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| origin_airport | scanned | 34 | 94.1 | 94.1 | 94.1 | 0.0 | 0.0 | 5.9 |
| origin_airport | digital | 66 | 100.0 | 100.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| origin_airport | all | 100 | 98.0 | 98.0 | 98.0 | 0.0 | 0.0 | 2.0 |
| pieces | scanned | 34 | 97.1 | 97.1 | 97.1 | 0.0 | 0.0 | 2.9 |
| pieces | digital | 66 | 100.0 | 100.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| pieces | all | 100 | 99.0 | 99.0 | 99.0 | 0.0 | 0.0 | 1.0 |
| shipper_name | scanned | 34 | 88.2 | 100.0 | 100.0 | 11.8 | 0.0 | 0.0 |
| shipper_name | digital | 66 | 89.4 | 92.4 | 100.0 | 3.0 | 7.6 | 0.0 |
| shipper_name | all | 100 | 89.0 | 95.0 | 100.0 | 6.0 | 5.0 | 0.0 |

## 3. Per row field (doc level)

Row values are searched in the whole doc text, so short numerics (quantity) can match by chance elsewhere on the page: read quantity/customer part/PO next to the same-line column below.

| field | split | n | (a) exact % | (b) normalized % | (c) +fuzzy % | b-a formatting | c-b near-miss | 100-c unrecoverable |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| supplier_part_number | scanned | 1853 | 88.9 | 89.6 | 95.4 | 0.6 | 5.8 | 4.6 |
| supplier_part_number | digital | 3077 | 96.5 | 96.5 | 100.0 | 0.0 | 3.5 | 0.0 |
| supplier_part_number | all | 4930 | 93.6 | 93.9 | 98.3 | 0.3 | 4.4 | 1.7 |
| quantity | scanned | 1853 | 42.3 | 99.3 | 99.3 | 57.0 | 0.0 | 0.7 |
| quantity | digital | 3077 | 39.9 | 100.0 | 100.0 | 60.1 | 0.0 | 0.0 |
| quantity | all | 4930 | 40.8 | 99.7 | 99.7 | 59.0 | 0.0 | 0.3 |
| customer_part_number | scanned | 803 | 98.5 | 98.5 | 98.8 | 0.0 | 0.2 | 1.2 |
| customer_part_number | digital | 1131 | 100.0 | 100.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| customer_part_number | all | 1934 | 99.4 | 99.4 | 99.5 | 0.0 | 0.1 | 0.5 |
| purchase_order | scanned | 1127 | 94.9 | 94.9 | 98.1 | 0.0 | 3.2 | 1.9 |
| purchase_order | digital | 1848 | 99.2 | 99.2 | 100.0 | 0.0 | 0.8 | 0.0 |
| purchase_order | all | 2975 | 97.6 | 97.6 | 99.3 | 0.0 | 1.7 | 0.7 |

### Row fields found on the line assigned through the part number

| field | split | n | found on assigned line % |
|---|---|---:|---:|
| supplier_part_number | scanned | 1853 | 79.3 |
| supplier_part_number | digital | 3077 | 100.0 |
| supplier_part_number | all | 4930 | 92.2 |
| quantity | scanned | 1853 | 77.3 |
| quantity | digital | 3077 | 99.9 |
| quantity | all | 4930 | 91.4 |
| customer_part_number | scanned | 803 | 73.8 |
| customer_part_number | digital | 1131 | 99.9 |
| customer_part_number | all | 1934 | 89.1 |
| purchase_order | scanned | 1127 | 71.5 |
| purchase_order | digital | 1848 | 99.1 |
| purchase_order | all | 2975 | 88.7 |

## 4. Commands

```
uv run python scripts/ocr_ceiling.py
uv run pytest -q
uv run ruff check .
```

