# shipdoc-extract

![python](https://img.shields.io/badge/python-3.11-blue)
![base model](https://img.shields.io/badge/base%20model-Qwen3.5--4B-informational)
![code licence](https://img.shields.io/badge/code%20licence-Apache--2.0-green)
![adapter licence](https://img.shields.io/badge/adapter%20licence-Apache--2.0-green)

Structured JSON extraction from page images of commercial invoices and air waybills, using only
the images (no OCR text goes into the model). The model is `Qwen/Qwen3.5-4B` read at its native
page resolution (2,145 visual tokens per page), its output is constrained to the task schema, and
three deterministic post-processing rules plus a calibrated per-field review flag sit on top.
Two systems are evaluated here: **v1.5**, the zero-shot base model plus the rules plus review
flags, and **v2**, the LoRA-fine-tuned model plus the same rules and flags. The section "Final
system" below states which one was submitted. The zero-shot baseline with the rules scores 88.51
OVERALL on the 500 labelled documents (83.86 without the rules); an all-empty baseline scores 1.93
(train) and 1.65 (dev).

**No data is included.** There is no page image, no label, no OCR text and no model output on a
test document in this repository. The document ids that appear in it (`splits/`, `meta/`, a few
tests) are the file names of the evaluators' package and anonymised supplier-group names: they
refer to that package, they are not values, and without the package they point at nothing. To
reproduce anything that reads images you need the evaluators' package (see step 1 below). The tree
and its history were scanned against every train and dev label value of eight or more characters
(no hit). The reports are aggregates only.

## Final system

<!--FINAL-SYSTEM:BEGIN-->
The submitted system is stated in this block once the pre-registered pooled rule has been
evaluated on all three folds and the test submission has been validated. The v1.5 reproduction
steps (Part 1) and the fine-tuning evidence (Part 2) below are complete and do not depend on it.
<!--FINAL-SYSTEM:END-->

# Part 1. The v1.5 path (zero-shot base model, rules, review flags)

## Reproduce v1.5 in 5 steps

Reproduction runs on Google Colab (T4) from this repository at a pinned commit; nothing needs a
token or a private repository. The code of the submission is the first commit of this repository,
`3e88cc8` (`3e88cc8aa4bc2c33ac676c4c52395271f570b9cd`): the notebooks clone this repository and assert that
`HEAD` equals it. A commit cannot name its own SHA, so the second commit only writes that pin into
the notebooks, the configuration `configs/public_notebooks.json` and this README; the tag
`submission-v1.5` marks it. Nothing else differs between the two commits (`git diff --stat <pin>
submission-v1.5`). A later commit, tagged `submission-v1.5.1`, fixes the smoke notebook (it did
not unpack `assignment.zip`, so its post-processing could not read `schema.json`) and makes every
notebook stop right after the unzip step, before any install or GPU work, if `assignment/schema.json`
or `assignment/score.py` is missing; it changes only this README, the notebooks and their builder
and test files, and the pin stays the first commit. The Colab badges below open each notebook at
`submission-v1.5.1`. Both long notebooks are resumable: after a disconnect, Run all again (finished
stages are skipped).

1. **Put the evaluators' package into Google Drive**, folder `MyDrive/shipdoc-extract/` of the
   account you sign in with in Colab (no account is checked). Upload BOTH `data.zip` and
   `assignment.zip` (your own copy of the evaluator package): every notebook needs both, the data
   for the documents and the assignment folder for `schema.json` (the post-processing rules read it)
   and `score.py`. Two files, both made from the evaluators' package and NOT included here:
   - `data.zip`: a zip whose top-level folder is `data/` with `train/`, `dev/` and `test/`, each
     holding `images/<doc_id>_p<page>.jpg` and (train and dev only) `labels/<doc_id>.json`;
   - `assignment.zip`: a zip whose top-level folder is `assignment/` holding the evaluators' five
     files `README.md`, `schema.json`, `sample_submission.json`, `score.py` (the official scorer)
     and the task PDF.

   `ocr_cache.zip` (built by `scripts/make_colab_zips.py`) is not needed: the v1.5 notebook runs
   the OCR stage itself, on the 280 test pages only.
2. **Run `02n_zeroshot500_native`** (zero-shot over the 500 labelled documents; it sets the batch
   size the test run must reuse and is the baseline of every comparison below).
   [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-gandhi-2411/shipdoc-extract-public/blob/submission-v1.5.1/notebooks/02n_zeroshot500_native.ipynb)
   Runtime type T4 GPU, then Run all with every parameter at its default (the sha7 of the run
   folders is the pin's). Measured basis for the duration: 12,022 s of summed per-page decoding
   time over its 671 pages in the original run (about 3.3 T4 hours; wall-clock is longer by the
   model load, the smoke gate and the batch bench, which are not separately measured). The bench
   picks the largest batch size whose outputs are byte-identical to batch 1 and that fits the GPU;
   it chose 4 in the original run (8 ran out of memory).
3. **Run `04c_predict_test_native`** (the 200 test documents, 280 pages).
   [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-gandhi-2411/shipdoc-extract-public/blob/submission-v1.5.1/notebooks/04c_predict_test_native.ipynb)
   Same runtime, Run all, defaults (`MODEL = "zs"`). It runs a 5-document smoke gate, the test
   documents at the 02n batch size, a determinism pass, the OCR stage (PaddleOCR in its own
   environment) and the validated assembly with the rules R1 to R3. The original run took 5,664.5 s
   of wall-clock (model time 5,096.4 s, smoke 348.6 s, OCR 1.168 s per page; `manifest.json` of
   that run, reported in `reports/v1_5_submission.md`).
4. **Read the banner.** The last cell of 04c ends with `TEST PREDICTIONS v15 (zero-shot, base model,
   NATIVE + R1-R3) VALIDATED: SUBMITTABLE`. The review-flags cell raises on purpose before it
   (`CALIBRATOR_FILE = None` stops it with an instruction, after the predictions are validated):
   that is expected, not a failure; run the banner cell alone. Download `test_predictions.json`,
   `trace.jsonl`, `manifest.json`, `validation_report.json`, `rules.jsonl` and `ocr_timing.json`
   from `MyDrive/shipdoc-extract/submissions/v15_3e88cc8/` and the folder
   `MyDrive/shipdoc-extract/ocr_cache_test/` (its layout is `paddleocr/test/<page>.json`).
5. **Compute the review flags on CPU** (a local clone at the pin; copy the submission folder first
   because the second command rewrites its manifest and report):

   ```
   git clone https://github.com/gaurav-gandhi-2411/shipdoc-extract-public.git shipdoc-extract && cd shipdoc-extract && git checkout 3e88cc8aa4bc2c33ac676c4c52395271f570b9cd
   uv sync --frozen
   uv run python -m shipdoc.predict_native flags --model zs --submission-dir <copy of v15 folder> --calibrator meta/calibrator_zs_native.json --ocr-cache <ocr_cache_test folder> --batch-size 4 --field-target 0.98 --doc-target 0.98 --expect-docs 200
   uv run python -m shipdoc.predict_native check-flags --out-dir <copy of v15 folder> --calibrator meta/calibrator_zs_native.json --field-target 0.98 --doc-target 0.98 --expect-docs 200
   uv run python -m shipdoc predict validate --pred <copy of v15 folder>/test_predictions.json --schema <your assignment folder>/schema.json
   ```

   The first command writes `review_flags.json`. It also recomputes the predictions from
   `trace.jsonl` with the production post-processing and refuses if `test_predictions.json` is not
   what that produces. The second must print `check-flags (native): flags_ok=True failed checks []`.

### Expected outputs

| What | Expected |
|---|---|
| `test_predictions.json` sha256 (the byte-identity check) | `d20f69d286581f783a1e85a9e1dd3581fd9ed35e5248364082674c63a7ec67b5` |
| documents / pages | 200 / 280; all 200 ids of the sample submission; `validation_report.json` `ok` true, `schema_ok` true |
| `manifest.json` config | `qwen35_4b_img_only_native`, config hash `e4b84ec2809625d5`, `max_pixels` 2196480, seed 42, prompt `v2` (hash `cabc7bd9116664db00f6cd8e8b184970e0a186f82062ee55835275883061c64c`) |
| model | `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, fp16, no adapter |
| batch size | 4, the size stored with the 02n run (the test run refuses any other) |
| rule slot shapes (`meta/slot_shapes.json`) sha256 | `920088327fcfe95dada188292bb4966f2b08a928624e34e34553ea633a7afd99` |
| rules on the 200 test documents | R1 4 changes in 4 documents, R2 8 changes in 6 documents, R3 174 changes in 15 documents, none skipped |
| `manifest.json` `code_sha` | `3e88cc8aa4bc2c33ac676c4c52395271f570b9cd` (this repository's pin) |
| determinism pass | 5 of 5 documents byte-identical |
| flags | `flags_ok=True`, 200 documents |

What may legitimately differ from the original submission: the manifest's `code_sha` (the original
run was pinned to a commit of the private development history, `4c17aa3`; this repository is a
single squashed history, so its pin is a different commit), the submission folder and run names
(they carry the pin's first seven characters), every timing, and the OCR timing record. The
predictions file's hash is the byte-identity check; the manifest is not. The hash can only be
expected to match on the same decoding stack (the T4 and the library versions locked in `uv.lock`,
batch size 4) and the same OCR text for the 6 documents R2 touches; a fresh end-to-end run in this
repository has NOT been performed (UNVERIFIED), see "Which code the pin runs" below for what was
verified.

### Smoke test of the repository on a T4 (about 15 to 20 minutes)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-gandhi-2411/shipdoc-extract-public/blob/submission-v1.5.1/notebooks/public_smoke.ipynb)
`public_smoke.ipynb` clones this repository without a token, refuses unless `HEAD` is the pin,
installs from `uv.lock`, decodes 5 documents (the same smoke gate as 02n) and prints a banner. It
needs BOTH `data.zip` (only the 5 smoke documents are read out of it) and `assignment.zip` (your
evaluator package: `schema.json` and `score.py`, neither is part of this repository) in
`MyDrive/shipdoc-extract/`, and stops before any GPU work, with a message, if either is missing. It
does not reproduce anything: it shows that the clone, the pin, the install and the GPU decoding
path work.

### Which code the pin runs

The pin of this repository is not the commit of the original run. The code is the same where it
matters: `configs/`, `uv.lock` and `pyproject.toml` are byte-identical to the original pin, and
`src/` differs in three files: the added offline-evaluation module `src/shipdoc/gate.py`, and the
review-flag stage (`src/shipdoc/flags.py`, `src/shipdoc/predict_native.py`: a stricter check that
`rules.jsonl` is present and which code version produced the submission). The decoding, merge,
rules and OCR modules (`spike`, `predict`, `postrules`, `ocr_stage`, the prompt and the
configuration) are unchanged and none of them imports a changed file. Re-assembling
`test_predictions.json` from the original `trace.jsonl` and the OCR cache with this repository's
code gives the same sha256 as the submission (`reports/public_code_equivalence.md`; re-run it on
your own download with `uv run python scripts/public_equivalence.py --trace <v15 folder>/trace.jsonl
--ocr-cache <ocr_cache_test folder> --expect-sha256 d20f69d286581f783a1e85a9e1dd3581fd9ed35e5248364082674c63a7ec67b5`,
with `SHIPDOC_ASSIGNMENT_DIR` pointing at the folder that holds the evaluators' `schema.json`); the
decoding itself (the Colab run) is the part this repository cannot re-prove without a GPU run.

## Quick start (CPU, no data)

```
uv sync --frozen
uv run python -m shipdoc --help
uv run pytest -q tests/test_normalize.py tests/test_merge.py tests/test_totals.py tests/test_targets.py tests/test_train_masking.py tests/test_validate.py tests/test_locate.py
```

The whole suite runs on a fresh clone and passes: `uv run pytest -q -rs` must end with
`N passed, M skipped in T s`, no failed and no error, exit code 0 (`N` and `M` depend on the
optional packages installed; `-rs` prints the reason of every skip). The tests that need the
evaluators' package (`assignment/score.py` and `assignment/schema.json`, not redistributed: put
them in `assignment/`), the confidential data (`data/`), the OCR cache or a GPU stack are skipped
with that reason; with the inputs in place they run, and they pass.

Load the released LoRA adapter (a fold model, see "Hugging Face" below) on the pinned base model:

```python
import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

BASE, REVISION = "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
processor = AutoProcessor.from_pretrained(BASE, revision=REVISION)
base = Qwen3_5ForConditionalGeneration.from_pretrained(
    BASE, revision=REVISION, torch_dtype=torch.float16  # fp16 on a T4 (no bf16); bf16 on an L4
)
# the `peft/` folder of the Hugging Face repo (it holds adapter_config.json and
# adapter_model.safetensors; the repo mirrors the training run's final/ folder)
model = PeftModel.from_pretrained(base, "./adapter/peft")
```

## Architecture

| Stage | What it does | Where |
|---|---|---|
| Extraction | `Qwen/Qwen3.5-4B` (revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`), image only, keyed JSON output decoded under a grammar (xgrammar), greedy, thinking off, seed 42, prompt `v2`, `max_pixels` 2,196,480 (2,145 visual tokens for a 1240 x 1754 page; the earlier 1,260-token setting is kept for comparison) | `src/shipdoc/`, `configs/spike_qwen35_4b_img_only_native.yaml` |
| Page merge | Page-1 identity fields, last-page totals, repeated header-row filter, all-null rows dropped, dates to ISO, number and code normalisation | `src/shipdoc/` (Phase 3 post-processing) |
| Rules R1, R2, R3 | R1: an empty waybill carrier takes the model's own page-1 supplier name; R2: an empty MAWB / HAWB is filled from the PaddleOCR text of the waybill when exactly one candidate matches the number format (the only rule that uses OCR); R3: customer part number / purchase order values whose format shape occurs only in the other column of the training labels are moved there (shapes frozen in `meta/slot_shapes.json`) | `src/shipdoc/postrules.py` |
| Review flags | Per-field confidence from the token logprobs, a pooled logistic-regression calibrator (`meta/calibrator_zs_native.json`) and per-field-type thresholds at a 98% precision target; written to `review_flags.json`, never into `test_predictions.json` (the schema forbids extra keys) | `src/shipdoc/flags.py` |
| Fine-tune (evaluated) | LoRA r=16, alpha 32, language model only (200 modules, 30,474,240 trainable parameters), 2 epochs, effective batch 8 pages, learning rate 2e-4 cosine, seed 42 | `configs/finetune_qwen35_4b_native.yaml` |

Everything is thin wrappers around the `shipdoc` package: the Colab notebooks mount Drive, clone
this repository at a pinned commit, install from `uv.lock` and call `python -m shipdoc ...`.

# Part 2. Evaluation

## Results

All numbers are percentages from the unmodified official scorer (OVERALL = the task's composite).
95% confidence intervals are document-level bootstrap (2,000 resamples, seed 42). The 500 labelled
documents (671 pages) are 400 train and 100 dev documents of 18 invoice suppliers and 10 waybill
carrier groups. Dev contains no unseen supplier, so it measures seen layouts. Zero-shot arms never
trained on these documents, so their scores are out of sample by construction.

Model and naive baselines side by side (500 documents unless stated):

| System | OVERALL [95% CI] | Notes |
|---|---|---|
| All-empty / all-null / doc-type-only baselines | train (400 docs) 1.93 [1.68, 2.19]; dev (100 docs) 1.65 [1.12, 2.19] | the three trivial baselines score identically; no pooled 500-document figure exists |
| Zero-shot, 1,260 visual tokens, no rules | 76.58 [74.44, 78.66] | earlier resolution |
| v1: zero-shot, 1,260 tokens, + rules | 83.60 | paired gain over its own no-rules run +7.02 |
| Zero-shot, native, no rules | 83.86 [81.88, 85.71] | |
| **v1.5: zero-shot, native, + rules** | **88.51** | paired gain over native no-rules +4.64 [+3.43, +6.00] |

"No rules" means the predictions as the notebook writes them (page merge and normalisation on,
R1 to R3 off). For v1.5 the slices are: header accuracy 99.83, row F1 87.04, documents fully
correct 68.80, scorer false-fill rate 0.00. The rules replay on the 02n traces reproduces 88.51
offline (CPU):

```
uv run python scripts/replay_v1_check.py --run-dir <downloaded 02n folder> --out replay_native.md --expected-base-overall 0.8386 --gate-fixed R1=train:4,dev:3 --gate-fixed R2=train:27,dev:14 --gate-fixed R3=all:278
```

What those numbers hide, for v1.5:

- The scorer's false-fill rate (header fields only) is 0.00, but counting row cells where the gold
  is empty and the prediction is not, zero-shot + rules has 76 such cells pooled over the 500
  documents, all in one supplier fold, and 33 row or header cells where the gold has a value and
  the prediction is empty (definitions of spec section 11).
- R3 changes 287 cells: 278 fixed, 0 broken, 9 neutral. On the earlier 1,260-token run its gains
  sat in 3 of the 18 invoice supplier groups (493 of 502 fixed rows); that concentration was not
  re-measured at the native resolution. The "shipping" shapes are learned from these same 500
  documents (in-sample); the supplier-held-out variant ("honest") also gives 88.51.
- Review flags (out of fold, calibration v2 on the native zero-shot run, 500 documents): header
  fields 100.0% coverage at 99.8% precision; over all fields at the 98% target, 46.8% of fields
  are auto-accepted at 98.7% precision. Of the four row fields only supplier_part_number meets the
  98% target at its point estimate, and a document-level 98% target is not attainable on this
  evidence.
- There are no test labels, so no test score exists for the submission; its validation is the
  schema, the id set, the determinism pass and the flag checks (`reports/v1_5_submission.md`).

Fine-tuning, one fold only (fold 0: 329 documents train, 171 held-out documents of held-out
suppliers; native resolution; both arms with the same rules; paired bootstrap):

| Fold 0, 171 documents | Zero-shot + rules | Fine-tuned + rules | Paired difference |
|---|---|---|---|
| OVERALL | 91.62 [89.76, 93.37] | 93.53 [91.88, 95.08] | +1.92 [+0.36, +3.38] |
| Row cells with a value where gold is empty (false fills) | 0 | 5 | +5 |
| Cells empty where gold has a value (over-nulls) | 11 | 1 | -10 |

The fine-tune improves OVERALL on this fold and loses on the false-fill clause of the
pre-registered rule (spec section 11, item 1: the fine-tuned system is chosen only if the paired
OVERALL lower bound is above 0 AND its pooled false-fill and over-null counts are not worse).
One fold is not that decision.
The pooled three-fold result of that rule, and the choice it leads to, are stated in the section
"Final system".

Not done, stated plainly: the header-hint A/B (07n) was cancelled before it was built (future
work; an oracle ceiling of +2.80 OVERALL for column-shift rows was computed before any build); the
zoom re-read is future work; `causal_conv1d` and `flash-linear-attention` are absent from the
inference environment (a possible latency lever, not measured).

[`reports/INDEX.md`](reports/INDEX.md) lists the ten reports that carry these results, one line
each. The numbers above come from those reports; the header of each names the script and the
inputs that produced it.

## The fine-tuned path

These steps produced the fine-tuned evidence above (whether that system is the submitted one is
stated in the section "Final system"). Run the v1.5 steps 1 to 4 first (the zero-shot run is the
baseline and its test traces are the fine-tuned path's flag input), then the steps below; every
notebook shares the one pin and every parameter default already holds the right value, so only
`STAGE`, `FOLD` and `MODEL` are edited.

1. **`03n_finetune_native.ipynb`**, L4 (a T4 would train in fp16 under a different run id: stop).
   Edit only `STAGE`: `"smoke"` first (the default, 20 steps, must pass), then `"fold0"`,
   `"fold1"`, `"fold2"` and `"final"`. Fold stages train on the other folds' documents (fold 0: 329
   documents, 441 pages, 112 steps); `final` trains on the 400 train documents and holds out the
   100 dev documents. Output `runs/ft_native_<stage>_<sha7>_bf16/` with `final/` (adapter).
2. **`05n_oof_infer_native.ipynb`**, T4, once per fold. Set `FOLD` (0, 1, 2). Output
   `runs/oof_native_fold<K>_<sha7>/`.
3. **Compare offline** (CPU), per fold, then pooled:

   ```
   uv run python scripts/g4_fold.py --oof-run-dir <oof_native_fold0 folder> --zs-run-dir <02n folder> --fold 0 --out g4_fold0_native.md --json-out g4_fold0_native.json
   uv run python scripts/g4_pooled.py --oof-run-dir <fold 0 OOF folder> --oof-run-dir <fold 1 OOF folder> --oof-run-dir <fold 2 OOF folder> --zs-run-dir <02n folder> --out g4_pooled_native.md --json-out g4_pooled_native.json
   ```

   The pooled script applies the pre-registered rule of `spec.md` section 11, item 1.
4. **`08_dev_final.ipynb`**, T4: official dev score of the final adapter at the native resolution
   (seen layouts). Output `runs/devfinal_native_<sha7>/`.
5. **`04c_predict_test_native.ipynb`**, T4, `MODEL = "ft"`; `ZS_TEST_DIR` stays at the validated
   v1.5 folder. Output `submissions/v2n_<sha7>/`.
   Review flags for the fine-tuned arm need a native fine-tuned calibrator and the zero-shot test
   directory as input; neither is part of the v1.5 flow and their local command is not verified.

Further analysis scripts (all CPU, aggregates only): `scripts/rule_gate.py`,
`scripts/false_fill_diag.py`, `scripts/calibrate_v2.py`, `scripts/calibrate_v3.py`,
`scripts/g4_pooled.py`; the reports they wrote are in `reports/`.

# Part 3. History

The development history is not published: this repository is a squashed history (the code, then
the notebooks pinned to it). What it keeps of the path:

- `notebooks/archive/`: notebooks of the earlier 1,260-token path, its spike and the resolution
  sweep that led to the native setting (`reports/res_sweep.md`), superseded by the native set
  above. They are kept as a record and are NOT runnable from this repository (they were pinned to
  commits of the private history that do not exist here). `notebooks/archive/README.md` says what
  each was.
- `notebooks/DEVELOPMENT_NOTES.md`: the full notes of the development workflow, written for the
  private repository (they mention a GitHub token and a Drive account that the public notebooks do
  not use).
- `spec.md`: the written specification and the pre-registered decision rules.
- `reports/`: every working report; the index names the ten that matter.

## Repository map

| Path | What |
|---|---|
| `src/shipdoc/` | the package: extraction, merge, rules, OCR stage, flags, evaluation |
| `configs/` | model and run configurations (YAML), `paths.yaml` (path defaults per profile) |
| `meta/` | frozen artefacts: rule slot shapes, review-flag calibrator, supplier groups |
| `splits/` | document lists and the three folds (ids only) |
| `scripts/` | notebook builders (`colab_build_*.py`), analysis and evaluation scripts |
| `notebooks/` | the Colab notebooks (generated from the builders; edit the builder, not the JSON) |
| `reports/` | aggregate reports, `reports/INDEX.md` |
| `tests/` | the test suite (part of it needs the evaluators' package) |

## Paths and environment variables

Scripts read their locations from environment variables (then a gitignored `.env`, then
`configs/paths.yaml`): `SHIPDOC_DATA_DIR`, `SHIPDOC_ASSIGNMENT_DIR`, `SHIPDOC_RUNS_DIR`,
`SHIPDOC_SUBMISSIONS_DIR`, `SHIPDOC_OCR_CACHE`, `SHIPDOC_CKPT_DIR`, `SHIPDOC_TMP_DIR`, and
`SHIPDOC_PROFILE` (`local` or `colab`). The reports and run sheets of this repository name those
variables where the author's machine used a local drive path. What `git grep -n "D:/shipdoc"`
still finds are defaults of development helpers outside the v1.5 path, left as they are because
changing a default would change what the script does: `scripts/calibrate.py`,
`calibrate_v3.py`, `final_checklist.py` (flag `--ext`), `hf_upload.py` (flags `--final-dir`,
`--runs-root`), `spike_diagnosis.py` (flag `--runs`), `supervised_share.py` (a Hugging Face cache
glob; without it that analysis skips the tokenizer part) and `wandb_log_public.py` (flag
`--runs-root`); the tests that read such folders (`tests/test_coerce.py`,
`test_schema_repair.py`, `test_freeze_calibrator.py`, `test_report_inputs.py`,
`test_train_masking.py`) skip when they are absent. The notebook builders and
`tests/test_public_notebooks.py` quote the pattern in order to remove it from the notebooks.

## Hugging Face and Weights & Biases

- Hugging Face: a LoRA adapter for `Qwen/Qwen3.5-4B`, with a model card (no data, metrics only):
  [[PUBLISH:hf_url]]
  The section "Final system" and the model card say which adapter it is: the fold-0
  cross-validation model (trained on 329 of the 500 documents) or the final adapter (trained on the
  400 train documents).
- Weights & Biases: public project `shipdoc-extract` with the fine-tuning configuration, loss
  curves and held-out loss only (no images, no document ids, no values): [[PUBLISH:wandb_url]]

## Licence

The code in this repository is released under the Apache License 2.0 (`LICENSE`, copyright 2026
Gaurav Gandhi). The adapter weights are released under Apache-2.0, the licence of the base model
`Qwen/Qwen3.5-4B` as read from its model card on 2026-10-04. The evaluators'
package (images, labels, scorer, task text) is not covered by this licence and is not included.
