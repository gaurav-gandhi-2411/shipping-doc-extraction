# Public code against the original pin

> Aggregates, hashes and counts only. No document value and no document id appears in this file.
> Measured on 2026-10-04 (IST) on CPU, against a copy of the Colab output of the v1.5 test run (the
> original folder was only read). Every number below was produced by the command printed with it.

**Question.** The submission was produced by code pinned at commit `4c17aa3` of the private
development history. This repository is a squashed history, so that commit does not exist here and
the public notebooks are pinned to this repository's own first commit. Is the post-processing that
turns the Colab trace into `test_predictions.json` the same code?

**Answer, with its limits.** Yes for everything after the model: re-assembling the predictions from
the committed `trace.jsonl` and the OCR cache with this repository's code gives a file with exactly
the sha256 of the submission (and so does the original pin's code). The decoding itself (the T4 run
that wrote `trace.jsonl`) is NOT re-proved here: it needs a GPU run (README, "Reproduce v1.5").

## 1. What differs between the pin and this repository

`git diff --numstat 4c17aa3 <public code commit> -- src configs meta pyproject.toml uv.lock`:

| File | Added / removed lines | Role |
|---|---|---|
| `src/shipdoc/gate.py` | 465 / 0 | new offline-evaluation module (a per-field gate); nothing imports it |
| `src/shipdoc/flags.py` | 33 / 1 | review-flag stage: a missing `rules.jsonl` now refuses; two more provenance keys |
| `src/shipdoc/predict_native.py` | 21 / 1 | `check-flags`: two more checks (rules file hash, submission code sha) |
| `meta/calibrator_zs_native.json` | new | the frozen review-flag calibrator (never read by decoding or assembly) |
| `meta/README.md` | new | describes the calibrator |

`configs/`, `uv.lock` and `pyproject.toml` are byte-identical (the git blobs of `uv.lock` and
`pyproject.toml` are `5e412720f6d2` and `d303fa75b595` in both). The git tree of `src/` is
`62942f8f8d35` here and `cace564d879d` at the pin; `meta/slot_shapes.json` (the frozen R3 shapes)
is unchanged (sha256 `920088327fcfe95dada188292bb4966f2b08a928624e34e34553ea633a7afd99`).

## 2. Can the differences reach `test_predictions.json`? Import graph

Two independent views, both on the public tree's `src/`.

**Static** (`scripts/import_closure.py`: every `import` statement of every module, function-level
and relative ones included, followed inside the package):

```
python scripts/import_closure.py --changed shipdoc.gate shipdoc.flags shipdoc.predict_native \
    --entry shipdoc.__main__ shipdoc.predict shipdoc.spike shipdoc.bench shipdoc.postrules \
            shipdoc.ocr_stage shipdoc.predict_native
```

| Entry module | Modules in its closure | Changed modules it reaches |
|---|---|---|
| `shipdoc.__main__` (every `python -m shipdoc ...` stage) | 33 | none |
| `shipdoc.predict` (smoke, batch, run, assemble, validate) | 30 | none |
| `shipdoc.spike` (decoding) | 28 | none |
| `shipdoc.bench` | 28 | none |
| `shipdoc.postrules` (R1 to R3) | 28 | none |
| `shipdoc.ocr_stage` | 4 | none |
| `shipdoc.predict_native` (04c `check`, `estimate`, `finalize`, `flags`) | 43 | `shipdoc.flags`, `shipdoc.predict_native` |

`shipdoc.gate` is reached by no entry point at all.

**Dynamic** (each entry module imported in a fresh interpreter, every loaded `shipdoc` file's git
blob compared with the blob at `4c17aa3`): 44 modules are loaded by the seven entry modules
together; 42 are byte-identical to the pin, the 2 that differ are `shipdoc.flags` and
`shipdoc.predict_native`. Per entry module the number of loaded modules that differ from the pin is
0 for `__main__` (4 loaded), `predict` (23), `spike` (17), `bench` (18), `postrules` (21) and
`ocr_stage` (4).

So the decoding, merge, rules, OCR and assembly code that wrote and re-writes
`test_predictions.json` is byte-identical to the pin. The two files that differ are reached only by
the CPU stages of `04c_predict_test_native` that read a finished folder (`check`, `estimate`,
`finalize`) and by the flags stage; the changed lines are in `run_stage` and `check_flags`, and
`post_rule_output`, the function that applies the post-processing, is unchanged. Those stages write
no prediction.

## 3. The check: re-assemble and compare sha256

`scripts/public_equivalence.py` runs the production post path (`shipdoc.flags.post_rule_output`:
page merge, R1 to R3 with the frozen shapes and the OCR cache, coercion, the one date repair) and
the production writer on a trace, and compares the bytes with an expected hash. Inputs: the
`trace.jsonl` of the v1.5 submission (200 documents) and the local PaddleOCR cache of the 280 test
pages; `SHIPDOC_ASSIGNMENT_DIR` points at the evaluators' `schema.json` (the date repair reads it).

| Code under test | Command | sha256 of the re-assembled file | Equals the submission's |
|---|---|---|---|
| the original pin `4c17aa3` (`src/` and `meta/` exported with `git archive`) | the same logic as the script, run with `PYTHONPATH` at the export | `d20f69d286581f783a1e85a9e1dd3581fd9ed35e5248364082674c63a7ec67b5` | yes |
| this repository (the public tree built by the overlay, `shipdoc` imported from it) | `python scripts/public_equivalence.py --trace <v15 folder>/trace.jsonl --ocr-cache $SHIPDOC_OCR_CACHE --expect-sha256 d20f69d2...` | `d20f69d286581f783a1e85a9e1dd3581fd9ed35e5248364082674c63a7ec67b5` (`MATCH`, 200 documents) | yes |

The submission's `test_predictions.json` is `d20f69d286581f783a1e85a9e1dd3581fd9ed35e5248364082674c63a7ec67b5`
(`reports/v1_5_submission.md`, section 1).

The review-flag stage of this repository (README step 5) was also run on a copy of the same folder:
`flags` wrote 200 documents (8,134 emitted fields: accept 3,198, review 4,936; 2,003 null fields),
`check-flags` printed `flags_ok=True failed checks []`, and the `review_flags.json` it wrote equals
the one written at the previous commit of the private repository in every key except
`inputs.code_sha`, `inputs.local_head_sha` and `inputs.submission` (the checkout and the folder name).

## 4. What this does not show

- The decoding (model, prompt, xgrammar, batch size 4, fp16, T4) was not run: the trace is read as
  data. A fresh run in this repository has not been performed (UNVERIFIED); its predictions can
  differ if the GPU stack or the OCR text of the 6 documents R2 touches differs.
- The OCR cache read here is the local CPU PaddleOCR cache; the Colab run used GPU OCR. The
  predictions assembled from the Colab OCR equal those assembled from the local cache
  (`reports/v1_5_submission.md`, section 2), so the hash above does not depend on that difference
  for these 200 documents.
- `src/` was compared by git tree, import graph and the hash above; no claim is made about files
  the test run never imports (training, the analysis scripts).

Re-run at any commit of this repository: `python scripts/public_equivalence.py --trace
<folder>/trace.jsonl --ocr-cache <ocr_cache_test folder> --expect-sha256
d20f69d286581f783a1e85a9e1dd3581fd9ed35e5248364082674c63a7ec67b5` (exit 0 = identical).
