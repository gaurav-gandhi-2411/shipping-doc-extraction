# Submission v1: v0 VLM outputs + OCR + rules R1-R3 (notebook 04b, code 8833c73)

Aggregates, hashes and structure only: no extracted value appears here. The submission file
(`$SHIPDOC_SUBMISSIONS_DIR\v1_8833c73\test_predictions.json`, outside the repo, gitignored) is never
committed. Unmodified copies of the run's `manifest.json` and `validation_report.json` are in
`reports/v1_manifest.json` and `reports/v1_validation_report.json` (neither stores extracted values).

## What it is

The VLM outputs are the v0 traces, reused (the notebook's reuse mode): Qwen3.5-4B image-only, keyed
format, prompt v2, greedy, seed 42, fp16 T4, batch size 8, max_pixels 1310720 (1260 visual tokens).
Reuse was allowed because the config hash `01d87878679ca0fc`, the prompt hash
`cabc7bd9116664db00f6cd8e8b184970e0a186f82062ee55835275883061c64c`, the model revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, seed, shard and batch size equal v0's, the v0 report was
VALIDATED, and the decode-path blob hashes at `8833c73` equal those at v0's code `42b812b` (code
identity cannot hold literally; the fingerprint replaces it, see `notebooks/README.md` 04b).
On top: PaddleOCR on the 280 test pages (GPU, 326.9 s of compute = 1.168 s/page mean, 1.669 p95;
wall clock 624.6 s in one session) and rules R1, R2, R3 at assembly.

## Local validation (this session, CPU; VERIFIED by running the commands)

| check | command | result |
|---|---|---|
| JSON Schema + 200 ids | `uv run python -m shipdoc predict validate --pred <file> --schema assignment/schema.json` | `ok=True schema_errors=0 200 found / 200 expected; missing 0, extra 0, duplicate 0` |
| `schema.json` sha256 | `sha256sum` | `4232741f...5143` = the v0 manifest's value |
| `test_predictions.json` sha256 | `sha256sum` | `fd7ec9bf4ed90c4f6167385a07e075d9bb4fe313acdfff161762e945ccafa8d7` |
| structure vs `assignment/sample_submission.json` | `$SHIPDOC_TMP_DIR\v0_struct.py` | same ids and order, same keys per doc, row keys and string-typed numbers as the sample; 163 invoices + 37 waybills; 2124 rows; waybills carry 9 header keys and `line_items: []` (as the gold does) |
| the run's own report | `reports/v1_validation_report.json` | `ok: true`; every check true, incl. `v1_rule_switches`, `v1_no_rule_skipped` (`skipped: {}`), `v1_shapes_sha256` (b95e43...), `ocr_cache_complete` (280/280, gpu), `ocr_determinism` (5/5 pages text-identical on a fresh OCR), `assemble_twice` (byte-identical test_predictions, trace, rules) |

Determinism: VLM determinism rests on v0's own check (5/5 documents byte-identical, re-attached in
`manifest.determinism_v0`); v1 adds assemble-twice and OCR re-check.

## Rules and effect on the file (v0 -> v1, aggregate counts from `$SHIPDOC_TMP_DIR\v0_v1_diff.py`)

- Rules touched docs: R1 2, R2 6, R3 24 (eligible: R1 37, R2 37, R3 163); changes R1 2, R2 6, R3 341;
  0 skips (no `R2/no_ocr`, no `R2/ocr_incomplete`).
- 32 of 200 docs differ from v0. Changed header cells: `carrier` 2, `mawb` 6. Changed row cells:
  `customer_part_number` 341 and `purchase_order` 341 (R3 moves a value between the two slots, so the
  counts are equal). Row counts, doc types and all other cells are identical to v0.
- There are no test labels, so none of this says anything about accuracy. The evidence for the rules
  is `reports/rule_gate.md` and `reports/v1_replay.md` (500 train+dev docs, UNVERIFIED label there;
  independently re-derived by the verifier). R3's gain there is concentrated in 3 of 18 invoice groups
  (493 of 502 fixed rows), and R2 depends on OCR at inference (approved by GG).

## Open items

- v1 is the safety submission with rules; the final-system rule of `spec.md` section 11 may replace it
  by FT+rules (v2), and the native-resolution decision (`reports/res_sweep.md`) adds a v1.5.
- Review flags for v1 need the frozen ZS calibrator: `meta/calibrator_zs.json` is now committed
  (frozen from the 1260-token run; it records no config hash, so it is valid only for the 1260 v1 path).
- The manifest records the R3 shapes file sha256 `b95e4340...` (the file at code `8833c73`). The shapes
  file at later commits hashes to `92008832...`: commit 473bd7d changed how `splits/folds.json` is
  hashed (CRLF read as LF), which rewrites only the file's `folds_sha256` metadata line. The shape
  sets are identical, and the current file reproduces v1 cell for cell (second verifier, 0 of 200 docs
  differ), so the submission is unaffected.
