# Running G (the per-field gate) on the test set

Status: PROCEDURE ONLY. Nothing below has been run on test data and the test-set inference script
does not exist yet (see "What is NOT built"). Spec section 11 item 7 makes G a secondary system:
it is built for the test set only if G replaces the pre-registered winner on the pooled 3-fold OOF
(`scripts/gate_eval.py`, decision line `SECONDARY SYSTEM G REPLACES ...`). Until then this file is
a plan, not a runbook. Statements marked VERIFIED were checked against the code in this repo;
anything else is a design proposal.

## 0. Gate before anything else

1. Fold 1 and fold 2 native OOF runs exist, `scripts/g4_pooled.py` has decided the primary rule
   (spec 11 item 1), and `scripts/gate_eval.py` (all three folds, no `--exploratory`) printed
   `SECONDARY SYSTEM G REPLACES <winner>`. If it printed `DOES NOT REPLACE`, stop: the winner of the
   primary rule is the final system and nothing in this file is run.
2. The report `reports/gate_pooled.md` is committed with its provenance (VERIFIED: the script writes
   it with the run folders, code shas, command and bootstrap settings).

## 1. Inputs needed (all CPU after the two 04c runs)

| input | what | where it comes from |
|---|---|---|
| v1.5 (ZS arm) | the 04c `MODEL = "zs"` submission folder: `test_predictions.json`, `trace.jsonl` (with `field_logprobs`), `manifest.json`, `rules.jsonl` | `notebooks/04c_predict_test_native.ipynb`, folder `submissions/v15_<sha7>/` (VERIFIED in `notebooks/README.md`) |
| v2 (FT arm) | the 04c `MODEL = "ft"` submission folder, same files | same notebook, `submissions/v2n_<sha7>/`; its flags stage already needs the VALIDATED v15 folder as `ZS_TEST_DIR` (VERIFIED), so both folders exist together |
| OCR cache of the test pages | for R2 and the OCR features | `ocr_cache_test` (the 04c OCR stage) |
| R3 shapes | the frozen all-gold shapes file | `meta/slot_shapes.json` (the production default, as in `flags.post_rule_output`) |
| calibrator, ZS view | `v2_agree` calibrator that scores the ZS arm's fields with the FT-vs-ZS agreement features | DOES NOT EXIST (see 2) |
| calibrator, FT view | `v2_agree` calibrator that scores the FT arm's fields | code exists, artifact does not (see 2) |
| already existing | `meta/calibrator_zs_native.json` is arm `zs`, variant `v2` (no agreement features), `run_config_hash` e4b84ec2809625d5 (VERIFIED by reading its keys). It is the ZS+rules (v1.5) review-flag calibrator, NOT the G calibrator | -- |

## 2. Freezing the calibrators from the three OOF folds

G's P(correct) is the `calibrate_v3` `v2_agree` variant of each judging view. The test-set
calibrators are the production versions of the cross-fitted ones: fitted on ALL emitted rows of the
500 train + dev documents (the same convention as `scripts/freeze_calibrator.py`: "the production
calibrator is fit on everything"), then applied to test documents that were never in the fit set.

Steps and what exists:

1. EXISTS: `scripts/calibrate_v3.py --folds 0 1 2 --oof-run <f0> --oof-run <f1> --oof-run <f2>
   --zs-run-dir <native 02n run> --require-logprobs --out-dir <dir>` writes `calibration_v3.json`
   (both views, `v2_agree` OOF probabilities in `oof_fields_v3.csv`) and refuses unless all three
   OOF runs are given. It is the source of the structure / kind choice.
2. EXISTS (FT view): `scripts/freeze_calibrator.py --arm ft --oof-run <f0> --oof-run <f1> --oof-run
   <f2> --zs-run-dir <native 02n run> --calibration-dir <step 1 dir> --out
   meta/calibrator_ft_native.json`. Its docstring marks the three native OOF runs and the 3-fold
   `calibrate_v3` output as "NOT YET THERE" (VERIFIED in the script's docstring); the code path is
   covered by `tests/test_freeze_calibrator.py`. `flags.design("ft", ...)` is exactly
   `design_variants(...)["v2_agree"]`, i.e. the FT view of G (VERIFIED).
3. DOES NOT EXIST (ZS view): a frozen `v2_agree` calibrator for the ZS arm's fields. `flags.ARMS`
   is `("zs", "ft")`; `zs` is the `v2` variant without agreement features, and `flags.build_tables`
   builds agreement features only for `ft` (VERIFIED). TODO: add an arm `zs_agree` (judging view
   `judge_zs`, agreement computed with `agreement_matrix(..., self_is_ft=False)`) to
   `src/shipdoc/flags.py` (`ARMS`, `feature_names`, `build_tables`, `design`, `parity_probe`) and to
   `scripts/freeze_calibrator.py` (`--arm zs_agree`, reading the `judge_zs` view of
   `calibration_v3.json`), with tests in `tests/test_flags.py` / `tests/test_freeze_calibrator.py`.
4. Consistency check to do when step 1 exists: the (structure, kind) that `gate_eval` reports under
   "Cross-fit" must equal the `judge_ft` choice stored in `calibration_v3.json`, since both apply
   the same rule on the same data. UNVERIFIED until the 3-fold run exists.

## 3. Proposed CLI (does not exist)

```
uv run python scripts/gate_infer.py \
    --zs-dir submissions/v15_<sha7> --ft-dir submissions/v2n_<sha7> \
    --calibrator-ft meta/calibrator_ft_native.json \
    --calibrator-zs-agree meta/calibrator_zs_agree_native.json \
    --ocr-cache <ocr_cache_test> --out-dir submissions/g_<sha7>
```

Steps the script would perform, each reusing existing code:

1. Refuse unless both submission folders are VALIDATED, at the native config hash that both
   calibrators were frozen for (`flags.run_stage` already implements this refusal pattern for one
   arm), complete, and cover the same 200 test ids.
2. Rebuild each arm's post-rule output from its `trace.jsonl` (`flags.post_rule_output`) and require
   it to equal that folder's `test_predictions.json` (as `flags.run_stage` does).
3. Build the tables and agreement matrices of both views under `flags.id_guard(ids)` (the
   calibration code refuses `test_` ids by design; `id_guard` swaps the guard for its inverse and
   restores it) and score them with the frozen models (`flags.Model.predict`).
4. `gate.probs_by_doc` per view, then `gate.gate_predictions(ft_post, zs_post, p_ft, p_zs, ids)`:
   pure, deterministic, raises if a value is not one of the two arms' values or if coerce / repair
   would change the output (VERIFIED by `tests/test_gate.py`).
5. Validate `test_predictions.json` with the submission validator and the schema (additionalProperties
   false), write it to `submissions/g_<sha7>/` with a manifest that records both input folders, both
   calibrator sha256, the `gate_eval` report sha256 and the repo sha. Never write anything but
   `doc_type` / `header` / `line_items` into the predictions file.

## 4. Review flags (spec section 11 item 6)

Flags go to a SEPARATE file, `review_flags.json`, never into `test_predictions.json` (its schema has
`additionalProperties: false`); the existing stage already works this way (VERIFIED: `shipdoc.flags`,
`flags.FLAGS_NAME`). For G the open design question, which is NOT decided here and needs GG:

* The P(correct) of a field G emitted is the P of the arm it was taken from, i.e. the MAX of two
  correlated estimates, so it is optimistically biased as a confidence and the tau thresholds
  stored in `calibrator_ft_native.json` / `calibrator_zs_native.json` (chosen on a single arm's OOF
  probabilities) do not describe G's output.
* Proposed fix (TODO, not built): choose G's own tau per field type on G's pooled OUT-OF-FOLD
  selected-field probabilities (`gate_eval` has them in memory but does not write them: TODO add an
  `--oof-probs-out` flag writing `doc, scope, field, row_idx, source, p, y_correct` with no values),
  store them in a G flags artifact, and apply to the test selection. Fields of one-arm rows and
  one-arm fields carry the P that kept them (>= 0.5).
* Until that exists, the safe fallback is no auto-accept: every emitted field goes to review.

## 5. What is NOT built, and why

* `scripts/gate_infer.py` (section 3): the test-set inference CLI. Not built because G is only a
  secondary system and is built for the test set only if it wins the pre-registered rule (task
  brief; the spec text of section 11 item 7 itself says only that G replaces the winner under the
  three clauses). The pure core (`shipdoc.gate`) is built and tested; the I/O shell around it is
  thin but needs the artifacts below.
* The ZS-view calibrator arm and its freeze (section 2, step 3): `src/shipdoc/flags.py`,
  `scripts/freeze_calibrator.py`, tests.
* G's review-flag thresholds (section 4): needs a decision and `gate_eval` OOF probability output.
* No notebook: a Colab notebook is not needed, G is CPU-only given the two 04c submissions. Do not
  build one unless G wins.
* Nothing here changes `predict.py`, the 04c notebooks, the pins, or the primary final-system rule.

## 6. Known limits to carry into the report

* Three design decisions the spec does not define (also listed in every `gate_eval` report): the
  one-arm FIELD rule (value kept iff P >= 0.5), the row-level P (minimum over the row's emitted
  fields), and the `doc_type`-disagreement fallback (the ZS document). A different choice is a
  different system; changing any of them after fold 1 / 2 results are seen is a deviation from the
  pre-registration and must be disclosed.
* Rows that exist in both arms but were aligned only by position (`align_rows` stage 4) can pair
  rows that are not the same row; G then mixes fields of two rows.
* All G numbers in any report are UNVERIFIED until a verifier recomputes them.
