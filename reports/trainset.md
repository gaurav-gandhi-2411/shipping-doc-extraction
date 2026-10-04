# Fine-tune training set (STEP V2): row-to-page plan, round trip, augmentation rates

**Every number here is UNVERIFIED** (computed on this machine at commit `9a4b2a4`, nothing was trained). Counts only: no page content, no values, no names. Reproduce with `uv run python scripts/trainset_report.py` (needs `data/`, the OCR cache and `assignment/score.py`); tests: `tests/test_trainset.py`.

## Row -> page plan (`trainset.resolve_row_pages`, default policy)

| split | docs | pages | rows | line | page_fuzzy | single_page_default | interpolated | monotone_boundary | ambiguous |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dev | 100 | 135 | 924 | 838 | 70 | 0 | 10 | 5 | 1 |
| train | 400 | 536 | 4006 | 3707 | 229 | 20 | 37 | 1 | 12 |

| split | ok | header_only | excluded |
|---|---:|---:|---:|
| dev | 99 | 1 | 0 |
| train | 395 | 5 | 0 |

Sources: `line` / `page_fuzzy` keep the locator's page; `single_page_default` = unassigned row of a one-page doc (page 0); `interpolated` = both assigned neighbours on the same page; `monotone_boundary` = leading/trailing row whose neighbour is on the first/last page; `ambiguous` = page unknown. Documents with an `ambiguous` row are `header_only`: their `line_items` tokens are masked and the unplaceable rows are left out of the target text; with `ambiguous_policy: exclude` they would be dropped instead (below).

Audit log (`python -m shipdoc.trainset` writes `<SHIPDOC_RUNS_DIR>/trainset/audit.jsonl`, one line per decision: doc id, row, page, event; no values): {'ambiguous': 13, 'doc_header_only': 6, 'interpolated': 47, 'monotone_boundary': 6, 'single_page_default': 20} = 92 lines.

Ambiguous rows / header-only docs by rule (V1 estimated 12 train rows in 5 docs and 1 dev row in 1 doc with neighbour interpolation):

| rule | dev | train |
|---|---|---|
| boundary rule on (default, = V1 counts) | 1 rows / 1 docs | 12 rows / 5 docs |
| boundary rule off (strict neighbours only) | 6 rows / 3 docs | 13 rows / 6 docs |

Excluded documents under `ambiguous_policy: exclude`: dev 1, train 5.

Cross-check: the plan's assignments come from `locate.assign_rows` on the OCR cache; compared with `provenance/locations.jsonl` row records, 0 of 400 documents with rows differ.

Header-only documents by supplier group (train+dev): {'inv_g04': 1, 'inv_g05': 1, 'inv_g06': 2, 'inv_g07': 1, 'inv_g14': 1}. (Exclusion bias, V1d question 2: these are scanned multipage layouts; the header-only mode keeps their header supervision.)

## Round trip through `merge_pages` + `normalize_doc` + the official scorer

| split | tested docs | round-trip exact | OVERALL == 1.0 | header-only (skipped) |
|---|---|---|---|---|
| dev | 99 | 99 | 99 | 1 |
| train | 395 | 395 | 395 | 5 |

A document passes when its per-page targets, merged with the defaults of `merge_pages` and normalised, equal the gold header and rows AND score OVERALL 1.0 under `assignment/score.py` (document level).

## Stage splits (`splits/folds.json` is the only source)

| stage | train docs | train pages | held-out docs | held-out pages | manifest hash |
|---|---|---|---|---|---|
| smoke | 400 | 536 | 100 | 135 | bcc3c718b319390d |
| fold0 | 329 | 441 | 171 | 230 | f4bcc661a36211f7 |
| fold1 | 335 | 454 | 165 | 217 | 4d32a97038e1c578 |
| fold2 | 336 | 447 | 164 | 224 | 1727889837a0ae7f |
| final | 400 | 536 | 100 | 135 | d1d09bdfc4ba6008 |

`assert_no_leakage` passed for every stage (no train/held-out overlap, no test docs, no dev docs in `final`). The `smoke` stage trains on the first 80 pages (doc-id order) of the `final` training pages.

## Augmentation (train docs of `final`, seed 42)

Occlusion candidates over all 500 train+dev docs, by outcome per (doc, header field) with a non-null value or not: {'collateral': 2053, 'eligible': 1762, 'extra_fuzzy': 9, 'no_box': 78, 'not_on_target_page': 6, 'null_gold': 192}.

- Train pages: 536; pages whose target carries at least one provably hidable header field: 391 (72.9%).
- Stated rate: P(hide one field) = 0.15 per page and epoch -> expected 10.9% of train pages per epoch; realised over 20 sampled epochs: 1157/10720 = 10.8%.
- Natural prevalence for comparison (recon section 3): 35 redacted header cells in 500 train+dev docs (35 of 671 pages, 5.2%), all invoice_number / invoice_date.
- Field draw weights {'invoice_number': 4.0, 'invoice_date': 4.0} (others 1). Hidden field histogram (all 20 epochs): {'awb_number': 10, 'buyer_name': 115, 'carrier': 26, 'consignee_name': 30, 'currency': 2, 'destination_airport': 21, 'gross_weight_kg': 25, 'hawb': 18, 'invoice_date': 63, 'invoice_number': 154, 'mawb': 27, 'origin_airport': 22, 'pieces': 23, 'ship_to_name': 106, 'shipper_name': 29, 'supplier_name': 486}; invoice_number + invoice_date = 217 of 1157 (18.8%).
- Scan degradation: 336 of 536 train pages are digital (.png); each is degraded with p = 0.5 per epoch (expected 31.3% of all train pages; with the 200 real scans, 68.7% of pages look scanned).

## Not covered / limits

- Row-field occlusion is not done (header fields only); `edge_crop` is not used (its reach cannot be bounded by the collateral check).
- A field is hidden only if every printed copy is located (level <= normalized), no fuzzy-only copy remains, a copy sits on the page that carries the target, and the widest possible applied box overlaps no other located value by more than 10% (strict: any other value counts, so values that are printed twice side by side are rarely candidates).
- The interpolated rows rely on the monotone row order observed in 151 of 152 multipage docs; it is circular until the label-audit viewer checks a sample (V1d).
- Tokenised sample lengths and the 1,260-visual-token assertion need the real processor and were only exercised with a fake one.

locations.jsonl sha256 prefix (cross-check source): 6e1961394178486a.
