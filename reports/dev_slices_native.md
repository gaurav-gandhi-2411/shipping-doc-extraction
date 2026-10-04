# Native ZS + rules (v1.5) on the dev documents, by slice

**Provenance.** Run `zeroshot500_qwen35_4b_img_only_native_4c17aa3`, command `uv run python scripts/dev_slices_native.py --run-dir $SHIPDOC_TMP_DIR\native_extract\zs\zeroshot500_qwen35_4b_img_only_native_4c17aa3 --out reports/dev_slices_native.md --json-out $SHIPDOC_TMP_DIR\report_out\dev_slices_native.json` (repo state `389dc5c+dirty`; paired and plain bootstrap 2000 doc-level resamples, seed 42; unmodified scorer via `shipdoc.eval`). Rules: production path `replay_v1_check.production` (R1 + R2 + R3) with the HONEST supplier-held-out R3 shapes; rules OFF reproduces `predictions.json` exactly (asserted). **All numbers UNVERIFIED** until a verifier recomputes them. Aggregates and counts only. The dev documents are SEEN layouts (their supplier groups also occur in train): this is not an unseen-supplier estimate. Slices with n < 10 are printed but their intervals are not informative.

Docs: 100 dev of 500 scored; native ZS + rules OVERALL on all 500: 88.51.

## Slice definitions (the repo's own; file:line of this checkout)

- scanned: any page of the doc is a `.jpg` (`src/shipdoc/meta.py:108`); digital otherwise.
- multipage: more than one page image (`meta.py:109`).
- repeated part numbers: a supplier_part_number appears on >= 2 gold rows (`meta.py:75-85`).
- invoice / waybill: the gold `doc_type` (`src/shipdoc/eval.py:109-113`).
- illegible = redaction: >= 1 gold header null in a REQUIRED field, i.e. a field whose pooled null rate inside its doc type is < 10% (`meta.py:47-72`, `OPTIONAL_NULL_RATE` at `meta.py:20`; `reports/recon.md:125` confirmed the rule by eye on 13 docs). The tag is derived from gold null rates, NOT inspected per document: the per-document redaction vs absent assignment is rule-based (UNVERIFIED as a visual fact).
- absent line: an invoice whose gold awb_number is null or a waybill whose gold hawb is null, i.e. the line is not printed (`meta.py:88-95`).
- illegible, scorer sense: `score.py`'s `illegible_fields` metric counts EVERY gold header null, so its illegible set is redaction + absent line; the slice here is their union (`meta.py` tags; `reports/recon.md:173`).
- false fills split the same way by `src/shipdoc/confidence.py:741-755` (`false_fill_split`).

Dev suppliers: 27 supplier groups, of which 27 also occur in train (`meta/supplier_groups.json`).

## Dev100 (seen layouts), by slice

| slice | n | OVERALL % [95% CI], ZS + rules | OVERALL %, raw | paired delta, rules - raw (pts) [95% CI] | header acc % | row F1 % | docs fully correct % | false-fill % |
|---|---|---|---|---|---|---|---|---|
| all dev | 100 | 89.31 [86.00, 92.77] | 83.72 | +5.59 [+2.75, +9.26] | 99.76 | 87.01 | 73.00 | 0.00 |
| invoices | 77 | 87.66 [83.59, 91.61] | 84.72 | +2.94 [+0.35, +6.19] | 99.68 | 87.01 | 64.94 | 0.00 |
| waybills | 23 | 100.00 [100.00, 100.00] | 84.73 | +15.27 [+8.70, +22.32] | 100.00 | 100.00 | 100.00 | 0.00 |
| scanned | 43 | 86.98 [81.23, 92.95] | 82.26 | +4.72 [+1.85, +8.12] | 99.72 | 82.85 | 69.77 | 0.00 |
| digital | 57 | 90.80 [86.59, 94.96] | 84.90 | +5.90 [+1.56, +11.27] | 99.79 | 89.48 | 75.44 | 0.00 |
| multipage | 28 | 80.45 [75.02, 85.90] | 79.37 | +1.08 [+0.00, +3.39] | 99.55 | 85.50 | 32.14 | 0.00 |
| single page | 72 | 94.10 [88.95, 98.15] | 84.34 | +9.77 [+3.64, +17.93] | 99.83 | 90.98 | 88.89 | 0.00 |
| repeated part numbers | 17 | 92.55 [86.23, 98.23] | 86.73 | +5.82 [+0.00, +16.04] | 100.00 | 93.15 | 76.47 | 0.00 |
| illegible: gold redaction (a required header field is null) | 7 (n < 10: CI uninformative) | 85.85 [67.52, 95.37] | 85.85 | +0.00 [+0.00, +0.00] | 100.00 | 86.05 | 57.14 | 0.00 |
| absent line: awb (invoice) or hawb (waybill) not printed | 27 | 85.96 [78.61, 93.29] | 82.35 | +3.61 [+0.00, +10.20] | 99.55 | 82.02 | 66.67 | 0.00 |
| illegible, scorer sense: any gold header null (redaction or absent line) | 30 | 84.91 [77.17, 92.00] | 81.52 | +3.40 [+0.00, +9.41] | 99.59 | 81.02 | 63.33 | 0.00 |

Redaction and absent-line overlap in 4 dev docs (7 + 27 - 30 in the union).

## All 500 train+dev docs (supplier-held-out R3 shapes), same slices

Every document is scored with R3 shapes learned from the OTHER supplier folds; the zero-shot model never trained on any of them, so this is the unseen-supplier view of the same system (the honest shapes equal the all-gold shapes in 3 of 3 folds, `reports/calibration_v2_native.md`).

| slice | n | OVERALL % [95% CI], ZS + rules | OVERALL %, raw | paired delta, rules - raw (pts) [95% CI] | header acc % | row F1 % | docs fully correct % | false-fill % |
|---|---|---|---|---|---|---|---|---|
| all | 500 | 88.51 [86.99, 90.07] | 83.86 | +4.64 [+3.43, +6.00] | 99.83 | 87.04 | 68.80 | 0.00 |
| invoices | 400 | 86.99 [85.35, 88.76] | 83.98 | +3.01 [+1.89, +4.17] | 99.81 | 87.04 | 61.25 | 0.00 |
| waybills | 100 | 99.71 [99.13, 100.00] | 88.84 | +10.87 [+7.93, +14.11] | 99.89 | 100.00 | 99.00 | 0.00 |
| scanned | 193 | 86.92 [84.11, 89.63] | 82.27 | +4.64 [+2.80, +6.69] | 99.87 | 84.78 | 65.28 | 0.00 |
| digital | 307 | 89.48 [87.56, 91.40] | 84.86 | +4.62 [+3.11, +6.38] | 99.80 | 88.40 | 71.01 | 0.00 |
| multipage | 152 | 81.76 [79.14, 84.14] | 78.21 | +3.55 [+2.00, +5.23] | 99.84 | 85.81 | 37.50 | 0.00 |
| single page | 348 | 92.46 [90.28, 94.39] | 87.30 | +5.17 [+3.27, +7.37] | 99.83 | 90.10 | 82.47 | 0.00 |
| repeated part numbers | 82 | 88.15 [84.71, 91.40] | 85.60 | +2.55 [+0.55, +5.14] | 100.00 | 89.28 | 62.20 | 0.00 |
| illegible: gold redaction (a required header field is null) | 35 | 88.77 [83.92, 93.27] | 83.94 | +4.83 [+0.95, +10.00] | 99.64 | 90.86 | 62.86 | 0.00 |
| absent line: awb (invoice) or hawb (waybill) not printed | 157 | 86.81 [83.86, 89.76] | 84.40 | +2.41 [+0.96, +3.99] | 99.77 | 85.41 | 63.69 | 0.00 |
| illegible, scorer sense: any gold header null (redaction or absent line) | 180 | 87.26 [84.52, 89.85] | 84.47 | +2.79 [+1.38, +4.49] | 99.79 | 86.41 | 63.89 | 0.00 |

Redaction and absent-line overlap in 12 500-doc docs (35 + 157 - 180 in the union).

## Seen (dev100) next to supplier-held-out (OOF) for the SAME system

| set | n | OVERALL % [95% CI], ZS + rules |
|---|---|---|
| dev100 (seen layouts) | 100 | 89.31 [86.00, 92.77] |
| dev100 invoices (seen) | 77 | 87.66 [83.59, 91.61] |
| 500-doc OOF, supplier-held-out | 500 | 88.51 [86.99, 90.07] |
| 500-doc OOF invoices, supplier-held-out | 400 | 86.99 [85.35, 88.76] |
| OOF fold 0 (supplier-held-out) | 171 | 91.62 [89.86, 93.51] |
| OOF fold 1 (supplier-held-out) | 165 | 90.14 [87.35, 92.87] |
| OOF fold 2 (supplier-held-out) | 164 | 83.98 [80.90, 87.08] |

## Header false fills by why the gold is null (Wilson 95%)

`redaction` = every gold header null except the awb / hawb lines flagged absent in the meta tags; `absent_line` = invoice awb_number or waybill hawb null and flagged absent (`shipdoc.confidence.false_fill_split`).

| set | arm | kind | gold-null header fields | filled | rate % [95% CI] |
|---|---|---|---|---|---|
| dev | raw | redaction | 7 | 0 | 0.00 [0.00, 35.43] |
| dev | raw | absent_line | 27 | 0 | 0.00 [0.00, 12.46] |
| dev | rules | redaction | 7 | 0 | 0.00 [0.00, 35.43] |
| dev | rules | absent_line | 27 | 0 | 0.00 [0.00, 12.46] |
| all 500 | raw | redaction | 35 | 0 | 0.00 [0.00, 9.89] |
| all 500 | raw | absent_line | 157 | 0 | 0.00 [0.00, 2.39] |
| all 500 | rules | redaction | 35 | 0 | 0.00 [0.00, 9.89] |
| all 500 | rules | absent_line | 157 | 0 | 0.00 [0.00, 2.39] |
