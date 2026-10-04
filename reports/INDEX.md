# Reports index

The ten reports that carry the results quoted in the README, in the order of the README. All are
aggregates and counts only (no document value). Every other file in `reports/` is a working
report; its header names the script and the inputs that produced it. Paths named inside the
reports as `$SHIPDOC_RUNS_DIR`, `$SHIPDOC_SUBMISSIONS_DIR`, `$SHIPDOC_OCR_CACHE` or
`$SHIPDOC_TMP_DIR` are the folders of the author's machine (the README, "Paths and environment
variables").

| # | Report | What it holds |
|---|---|---|
| 1 | [`v1_5_submission.md`](v1_5_submission.md) | the submitted system: provenance, file hashes, the validation table, the review flags of the 200 test documents; why no test score exists |
| 2 | [`public_code_equivalence.md`](public_code_equivalence.md) | this repository's code against the original pin: the differing files, the import graph, and the sha256 of `test_predictions.json` re-assembled from the committed trace |
| 3 | [`v1_5_replay_native.md`](v1_5_replay_native.md) | the rules R1 to R3 replayed on the native zero-shot run: 83.86 to 88.51, paired CI, honest and shipping variants |
| 4 | [`res_sweep.md`](res_sweep.md) | the resolution decision (1,260 against 1,750 against 2,145 visual tokens) and the 500-document native confirmation |
| 5 | [`rule_gate_native.md`](rule_gate_native.md) | the pre-registered ship rule for R1, R2 and R3 at the native resolution, rule by rule |
| 6 | [`calibration_v2_native.md`](calibration_v2_native.md) | the out-of-fold calibration of the review flags: coverage and precision per field, what is and is not attainable |
| 7 | [`g4_fold0_native.md`](g4_fold0_native.md) and [`g4_native_fold0_verdict.md`](g4_native_fold0_verdict.md) | fold 0: fine-tuned plus rules against zero-shot plus rules, paired bootstrap, and the clause-by-clause verdict (one fold, not the decision) |
| 8 | [`gate_fold0_native_exploratory.md`](gate_fold0_native_exploratory.md) | an exploratory per-field gate on fold 0 (not a decision; folds 1 and 2 untouched) |
| 9 | [`row_errors_native.md`](row_errors_native.md) | where the remaining row errors are: missing and extra rows, part-number confusions, the oracle ceiling of the column-shift fix |
| 10 | [`baselines.md`](baselines.md) | the naive baselines (empty, all-null, document-type only) with the official scorer's own value next to ours |

Related, not in the ten: [`ft_native_fold1_verification.md`](ft_native_fold1_verification.md)
(checks of the fold-1 adapter), [`hdrhint_estimate_native.md`](hdrhint_estimate_native.md) (the
header-hint estimate behind the cancelled A/B), and the earlier 1,260-token reports
(`v0_submission.md`, `v1_submission.md`, `v1_replay.md`, `respike.md`, `ablations.md`).
