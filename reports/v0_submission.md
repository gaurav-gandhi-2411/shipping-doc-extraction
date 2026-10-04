# Safety submission v0 (notebook 04, code 42b812b)

Aggregates, hashes and structure only: no extracted value appears here. The submission file itself
(`$SHIPDOC_SUBMISSIONS_DIR\v0_42b812b\test_predictions.json`, outside the repo, gitignored) is never
committed. Copies of the run's `manifest.json` and `validation_report.json` are in
`reports/v0_manifest.json` and `reports/v0_validation_report.json` (unmodified; both state they hold
no extracted values).

## What it is

Qwen3.5-4B image-only, keyed format, prompt v2, greedy, seed 42, fp16 on a Tesla T4, batch size 8
(chosen by the 02 bench, contract "dev and test both batch size 8"), Phase 3 repo defaults, **no
R1/R2/R3 rules and no OCR** (those arrive in v1). Code `42b812b5b09d6e4bff0df12564017f71ffad5fc9`,
model revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, config hash `01d87878679ca0fc`, prompt hash
`cabc7bd9116664db00f6cd8e8b184970e0a186f82062ee55835275883061c64c`. Timing as reported by GG's banner
and the manifest: wall clock 4146.6 s over 2 sessions, model time 3431.5 s (280 pages), smoke 334.0 s.

## Local validation (this session, CPU, VERIFIED by running the commands)

| check | command | result |
|---|---|---|
| JSON Schema + 200 ids | `uv run python -m shipdoc predict validate --pred <file> --schema assignment/schema.json` | `ok=True schema_errors=0 200 found / 200 expected; missing 0, extra 0, duplicate 0` |
| `assignment/schema.json` sha256 | `sha256sum` | `4232741f...5143`, equals the manifest's `schema_file.sha256` |
| `test_predictions.json` sha256 | `sha256sum` | `f0f9e9452678f3716516d6f130c892379deb8fd44b44223b9c8869f479464c34` |
| structure vs `assignment/sample_submission.json` | `$SHIPDOC_TMP_DIR\v0_struct.py` (keys, order, value kinds only) | see below |
| the run's own validation report | `reports/v0_validation_report.json` | `ok: true`; run complete; 200 docs / 280 pages; ids match image folder and sample; schema 0 errors; determinism 5/5 byte-identical; batch-size contract ok; code SHA clean and equal to the pin; production config matches; 280/280 pages with field logprobs, 0 invalid JSON, 0 pages at max_new_tokens; 0 schema repairs |

## Structure against the sample submission (no values)

- Top level: both objects keyed by doc id; the 200 ids are identical to the sample's and in the same
  (sorted) order.
- Every doc has exactly the keys `doc_type`, `header`, `line_items`, as in the sample.
- Line items have exactly `supplier_part_number`, `customer_part_number`, `purchase_order`,
  `quantity`; numeric fields are strings, as in the sample.
- The sample is a placeholder with 200 invoices and no waybills; v0 predicts 163 invoices and 37
  waybills. Waybill docs carry 9 header keys the sample never shows (`carrier`, `consignee_name`,
  `destination_airport`, `gross_weight_kg`, `hawb`, `mawb`, `origin_airport`, `pieces`,
  `shipper_name`) and an empty `line_items` list, which is also how train/dev gold stores all 100
  waybills (100/100 have `line_items == []`). Schema validation accepts it.
- Rows: 2124 predicted rows over 163 invoices.
- Null counts among predicted invoices (of 163): `awb_number` 48, `invoice_date` 4,
  `invoice_number` 11; waybills (of 37): `hawb` 11, `mawb` 6, `carrier` 2. `customer_part_number`
  null on 867 of 2124 rows, `purchase_order` on 1063. These are descriptive; there are no test
  labels, so they say nothing about accuracy.

## Open items

- Predicted waybill share 18.5% (37/200); the label-free `'invoice'`-keyword split in
  `reports/layout_test_clusters.md` gave 38 waybill-like docs: consistent within one doc.
- v0 has no R1/R2/R3. The replay on the 02 run (`reports/v1_replay.md`, UNVERIFIED) puts those rules
  at +7.02 pts [+5.42, +8.90] OVERALL on 500 train+dev docs; v1 (notebook 04b) is not run yet.
