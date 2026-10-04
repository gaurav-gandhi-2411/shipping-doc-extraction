# v1.5 submission: native zero-shot + rules R1-R3 + review flags

> Aggregates and counts only. No document value and no document id appears in this file.
> There are NO test labels, so no test score exists for this submission (see section 6).
> Items marked VERIFIED were run on this machine on 2026-10-04 against a copy of the Colab output
> (the original folder was never modified: sha256 of its 6 files identical before and after).
> Items marked UNVERIFIED are claims recorded by the Colab run (manifest / validation_report /
> banner) that this machine cannot reproduce.

## 1. Provenance

| item | value |
|---|---|
| run (Colab, notebook `04c_predict_test_native`, MODEL `zs`) | `testnative_zs_4c17aa3` |
| original folder | `$SHIPDOC_SUBMISSIONS_DIR\v15_4c17aa3` (untouched) |
| working copy with flags | `$SHIPDOC_TMP_DIR\v15_final\copy` |
| submission code SHA (manifest `code_sha`) | `4c17aa3c33c09f0cda7bb1f625947a1143a8cb28` (= `PINNED_SHA` in the notebook and in `scripts/colab_build_dev_final_native.py`) |
| flags stage code | local HEAD `44c110e0f8d36297c8618545c5767c1ae26c1d6c` (recorded as `+dirty`: unrelated uncommitted report / publish files were in the tree; the flags code itself is committed in 44c110e), not yet pushed |
| model | `Qwen/Qwen3.5-4B` @ `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, no adapter |
| config | `qwen35_4b_img_only_native`, hash `e4b84ec2809625d5`, max_pixels 2196480, seed 42 |
| rule slot shapes | `meta/slot_shapes.json` sha256 `920088327fcfe95dada188292bb4966f2b08a928624e34e34553ea633a7afd99` |
| calibrator | `meta/calibrator_zs_native.json` sha256 `2148c84a5b1d61fd6577475f41d40417e7e905fbb617b0e28e91bea4c74ce81b` (pooled lr, frozen for hash e4b84ec2809625d5) |

sha256 of each file (the original six, unchanged; `review_flags.json` and the two files the flags
stage updates are from the copy):

| file | sha256 |
|---|---|
| test_predictions.json (original = copy) | `d20f69d286581f783a1e85a9e1dd3581fd9ed35e5248364082674c63a7ec67b5` |
| trace.jsonl (original = copy) | `1255288fe650cf6e80670891bfb32dbe387c87e6317983f19655c48bf721b4ba` |
| rules.jsonl (original = copy) | `c9eed7ada1311367b7d8eecb1f8f154457aff1e990fb61517d7b15aae9273f5d` |
| ocr_timing.json (original = copy) | `3e16a4d02809ebda70de77e9388dfa853dc08668479775848d516273e9ede039` |
| manifest.json, original (Colab) | `14892c1bd5dfdf5a8f8c4a302cb6c0b9b12831ace14710316d7c1e4233d14ada` |
| manifest.json, copy (adds the `review_flags` block) | `b369a252042699497e352b082149573d275a95042e603c8f7c85c4088a09e9d3` |
| validation_report.json, original (Colab) | `402a5813bdf23fa0ffb1dbd1536e88f052a0a946c173c648ded55c74c4795181` |
| validation_report.json, copy (adds `flags_checks`) | `48b2f2095edc2531f8a9ff4a7a85d95b420df1b494644ad84e96fa4d3e9b58a9` |
| review_flags.json (copy only) | `5e04501394711bf57f27015ba211d1af9574fd6fcae1a27dbfe982e8ee5c9ecc` |

## 2. How the flags were produced (exact commands, repo root, HEAD 44c110e)

```
Copy-Item (recursive) $SHIPDOC_SUBMISSIONS_DIR\v15_4c17aa3 -> $SHIPDOC_TMP_DIR\v15_final\copy
uv run python -m shipdoc.predict_native flags --model zs --submission-dir $SHIPDOC_TMP_DIR/v15_final/copy --calibrator meta/calibrator_zs_native.json --ocr-cache $SHIPDOC_OCR_CACHE --batch-size 4
uv run python -m shipdoc.predict_native check-flags --out-dir $SHIPDOC_TMP_DIR/v15_final/copy --calibrator meta/calibrator_zs_native.json
```

Output: `flags: 200 docs, auto-accept 0, emitted fields 8134 (accept 3198, review 4936), null fields
2003`; `check-flags (native): flags_ok=True failed checks []`. The stage refuses (and did not)
unless `test_predictions.json` equals the production post-processing of `trace.jsonl` (with the
frozen shapes and the local OCR cache) and `rules.jsonl` equals the rule records recomputed there.
Both held, so the flags describe exactly the submitted predictions. The OCR cache used is the local
CPU PaddleOCR cache (280 test pages); the Colab run used GPU OCR. Predictions assembled from the
Colab OCR equal those recomputed from the local OCR, so no Colab OCR file is needed to reproduce the
assembly locally.

## 3. Validation table

| # | check | result | evidence |
|---|---|---|---|
| a | JSON Schema (repo validator `shipdoc.predict.validate_file`, `assignment/schema.json` sha256 `4232741fabd7...`) | PASS (VERIFIED) | 0 schema errors, 0 duplicate keys |
| a | `predict_native finalize` re-run | NOT RUN | needs the Colab run folder (`--run-dir`: progress, session components, env.json) and the 280-page fresh-OCR recheck; neither exists locally. The `validation_report.json` from the Colab run records all its checks `ok` (read, not re-derived) |
| b | 200 ids | PASS (VERIFIED) | 200 found, 0 missing, 0 extra, 0 duplicate against the `data/test/images` folder ids; set equal to `assignment/sample_submission.json` ids; 200 unique |
| c | manifest pins | PASS (VERIFIED, read) | code_sha = pin = notebook pin; config hash e4b84ec2809625d5; max_pixels 2196480 (native config); model revision 851bf6e8...; `ft` absent; batch_size 4 (`manual`, contract "dev and test both batch size 4"); post_rules switches R1/R2/R3 true, touched docs 4/6/15, skipped {}; slot shapes sha 920088327fca...; prompt v2 |
| d | determinism and OCR determinism | recorded only (UNVERIFIED here) | `determinism`: ok true, 5/5 identical, batch composition replicated: True; `ocr.recheck`: 5/5 text-identical, 5/5 content-identical; `ocr_timing.json`: 280 pages, 1.168 s/page mean, p95 1.669, device gpu. FINDING: `ocr.recheck.adopted_from` = `v1_8833c73`, i.e. the OCR recheck record was adopted from the earlier v1 submission, not run fresh in this run (`ocr_timing.json` itself carries no such note) |
| e | review_flags.json structure | PASS (VERIFIED) | 200 docs (id set equals the predictions' ids); accept 3198 / review 4936 / null-not-scored 2003 / emitted 8134; calibrator sha256 recorded and equal to the file (computed here); run_config_hash e4b84ec2809625d5; `inputs.local_head_sha` 44c110e0f8d3...+dirty, `inputs.submission_code_sha` = pin, `inputs.rules_jsonl_sha256` = recomputed sha of `rules.jsonl`; `inputs.test_predictions_sha256` / `trace_sha256` equal the files |
| e | thresholds | recorded | field target 0.98 and doc target 0.98 (the `PRIMARY_TARGET` default); field tau: header 0.513, supplier_part_number 0.918, customer_part_number 0.99961, purchase_order 0.9999994, quantity null (target never reached, all review); doc tau 0.9999942. Nested target met: header and supplier_part_number yes; customer_part_number, purchase_order, quantity, document no |
| e | no document values in the flags | PASS (VERIFIED) | string leaves under `docs`: only "accept" (3198) and "review" (4936); 0 of 4095 distinct string values (len >= 4) of `test_predictions.json` equal any string leaf; the 230 raw-substring coincidences are all digits inside probability floats (228), one inside a key name and one inside note text. Shape: doc ids as the 200 keys of `docs` (allowed), per doc `p_fully_correct`, `auto_accept`, header and per-row `{emitted, flag, p_correct}` objects, field names and row indexes only |
| f | flags in a separate file; predictions untouched | PASS (VERIFIED) | `test_predictions.json` sha256 identical to the original; every doc has exactly doc_type / header / line_items |
| g | accept / review counts | see section 4 | document-level auto-accept 0 of 200 |
| h | drift sanity check | no large drift (section 5) | aggregates only |
| i | headline | see section 6 | 88.51 reproduced as a 500-doc labelled number; no test score |

## 4. What the flags say (field and document level)

Per field type on the 200 test documents (emitted, non-null fields only):

| field type | accept | review | realized accept share | calibrator coverage at 98% target (nested OOF, 500 docs) |
|---|---|---|---|---|
| header | 1561 | 1 | 99.9% | 100.0% |
| supplier_part_number | 1637 | 488 | 77.0% | 85.6% |
| customer_part_number | 0 | 901 | 0.0% | 3.6% |
| purchase_order | 0 | 1422 | 0.0% | 18.3% |
| quantity | 0 | 2124 | 0.0% | 0.0% |
| all emitted | 3198 | 4936 | 39.3% | 46.8% (dev-100 slice: 44.8%) |

The calibrator's own coverage claims for all per-type taus (`reports/calibration_v2_native.md`,
honest coverage statement, 500 docs nested): 83.3% at the 95 target, 46.8% at 98, 43.5% at 99; dev
slice 81.4 / 44.8 / 41.8. The flags use the 98 target, so the comparable figure is 46.8% against the
realized 39.3%. These are accept SHARES, not accuracies: no test labels exist, so the precision of
the accepted fields on test is unmeasured. The calibrator's precision claim (98.7% accepted-field
precision at the 98 target, nested OOF, 500 docs) is a train/dev estimate.

Document level: 0 of 200 documents are auto-accepted. The document tau is 0.9999942 and the highest
document `p_fully_correct` on test is 0.992 (median 0.77). Honestly: this submission carries no
document that the system is confident enough to pass without a human; the document-level 95% and
98% targets are NOT ATTAINABLE on the 500-doc evidence (nested document estimate at the 98 target:
7.6% accepted share at 73.7% precision, i.e. it misses the target). The useful business claim is
field-level only: about 39% of emitted fields (mainly headers and supplier part numbers) can skip
review; customer part number, purchase order and quantity are always reviewed.

## 5. Drift sanity check (aggregates; the null shares compare predictions with gold, different things)

| quantity | test (200) | train (400) gold | dev (100) gold |
|---|---|---|---|
| null share, all fields | 19.76% (2003 of 10137, predicted) | 22.30% | 18.57% |
| null share, header fields | 4.58% | 4.82% | 4.13% |
| null share, row fields | 22.68% | 25.87% | 21.78% |
| waybill share | 18.5% (37) | 19.3% (77) | 23% (23) |

Rule activity (`rules.jsonl`; 186 changes: R1 4 fills of `carrier`, R2 6 `mawb` + 2 `hawb` fills,
R3 174 `cpn_po` moves; 0 skipped):

| rule | docs touched / eligible, test | touched / eligible, 500 labelled docs (`reports/calibration_v2_native.md`) | changes, test vs 500 docs |
|---|---|---|---|
| R1 | 4 / 37 (10.8%) | 7 / 100 (7.0%) | 4 vs 7 |
| R2 | 6 / 37 (16.2%) | 29 / 100 (29.0%) | 8 vs 41 |
| R3 | 15 / 163 (9.2%) | 25 / 400 (6.3%) | 174 vs 287 (11.6 vs 11.5 per touched doc) |

No large drift. Mildest finding: R2 touches a smaller share of eligible waybills on test (16%)
than on the labelled set (29%), on 37 vs 100 documents; the cause is not diagnosed here.

## 6. The headline number

88.51 OVERALL is a score on the 500 LABELLED documents (400 train + 100 dev) with the official
scorer, native zero-shot plus R1-R3, not a test score. It was recomputed here with
`uv run python scripts/replay_v1_check.py --run-dir $SHIPDOC_TMP_DIR/native_extract/zs/zeroshot500_qwen35_4b_img_only_native_4c17aa3 --out $SHIPDOC_TMP_DIR/v15_final/replay_recompute.md --expected-base-overall 0.8386 --gate-fixed R1=train:4,dev:3 --gate-fixed R2=train:27,dev:14 --gate-fixed R3=all:278`
(repo HEAD 44c110e): all-500 OVERALL 83.86 -> 88.51, paired gain +4.64 [+3.43, +6.00], in both the
honest (supplier-held-out R3 shapes) and the shipping (in-sample shapes) variants, matching
`reports/v1_5_replay_native.md`. No test labels exist: the test score of this submission is
unknown and must not be quoted.

## 7. Limitations (honest)

- Document-level auto-accept is 0 of 200 (section 4).
- Native calibrator calibration is weaker than the 1260-resolution one in places: ECE (15 equal-width
  bins, 500 docs) all_emitted 0.029 vs 0.019, customer_part_number 0.134 vs 0.094; document-level ECE
  0.088. customer_part_number AUROC about 0.72. The P(correct) values are advisory.
- The shipping R3 slot shapes were learned from the same 500 labelled documents (in-sample,
  optimistic); the honest supplier-held-out variant also gives 88.51 on those 500, but the shapes'
  generalization to test suppliers is untested (3 of 18 invoice groups carry most of the R3 fixes).
- R1 is marginal: 4 fills on test (7 on 500 labelled docs).
- The flags thresholds come from pooled out-of-fold probabilities; the production calibrator is
  fitted on all rows, so its probabilities differ slightly from the cross-fitted ones.
- Local flags used CPU OCR text; the Colab run used GPU OCR. Assembly equality was checked
  (section 2), but the OCR determinism record in the manifest is adopted from v1.

## 8. VERIFIED vs UNVERIFIED

VERIFIED here: schema, id set and uniqueness, all manifest pins read, the predictions equal the
production post-processing of `trace.jsonl`, `rules.jsonl` equals the recomputed rule records, flags
produced and `check-flags` ok, calibrator / rules / predictions / trace hashes, no document values in
the flags file, predictions byte-identical to the original, the 88.51 replay on the 500 labelled docs.

UNVERIFIED (Colab-only, taken from the manifest / banner): determinism 5/5 and batch-composition
replication, OCR timing 1.168 s/page and wall-clock 5664.5 s over 2 sessions (model 5096.4 s, smoke
348.6 s), OCR determinism 5/5 (adopted from `v1_8833c73`), that trace.jsonl was produced by the
recorded model revision, `code_sha_clean`, and a `finalize` re-run.

## 9. Files that constitute the submission

Required by the brief (`brief.txt`: "What to submit"): 1. `test_predictions.json` (200 documents,
format of `sample_submission.json`); 2. `report.pdf` (2 pages at most; the optional review-flag
precision / recall on dev belongs there); 3. the GitHub repository link (code and reproduction
instructions; the brief also asks for the Hugging Face and Weights & Biases links inside report.pdf).

From this run:
- `test_predictions.json` sha256 `d20f69d286581f783a1e85a9e1dd3581fd9ed35e5248364082674c63a7ec67b5`
  (the file to hand in).
- `review_flags.json` sha256 `5e04501394711bf57f27015ba211d1af9574fd6fcae1a27dbfe982e8ee5c9ecc`, a
  SEPARATE advisory file (spec section 11 item 6: never inside `test_predictions.json`, whose schema
  has `additionalProperties: false`). The brief does not require it.
- Supporting provenance (for the repository / Hugging Face, not required by the brief):
  `manifest.json` and `validation_report.json` from the copy, `rules.jsonl`, `ocr_timing.json`,
  `trace.jsonl`.

The exact list with hashes is `$SHIPDOC_TMP_DIR\v15_final\SUBMISSION_FILES.txt`.
