# shipdoc-extract — Build Spec (v1.1, 2026-10-01)

v1.1 (2026-10-01): recon-driven changes 1–9; approved by GG

Owner: GG · Executor: Claude Code (Opus 5.5 orchestrator → executor + verifier subagents at `<agents folder>\`)
Root: `<repo root>` · Deadline: 7 days

---

## 1. Objective and priority order

Extract structured JSON from shipping-document page images (commercial invoices, air waybills). Output must follow `assignment/schema.json` and is scored by `assignment/score.py`, unmodified.

Optimise in this order:
1. **Unseen-layout accuracy**: OVERALL under leave-supplier-out cross-validation. The headline unseen-supplier metric is **invoice-only OOF**, because waybill groups (carrier brand) do not track layout (recon §2b, §2e). Overall OOF is reported alongside it.
2. **False-fill rate** on illegible fields, reported split into redaction vs absent-line nulls (recon §3: 157/192 gold header nulls are absent-line, 35/192 are redactions).
3. **Calibrated per-field confidence and a `needs_review` flag**, with precision and recall reported on dev.
4. **Official dev OVERALL**, which measures seen layouts.
5. **Cost and latency** per page on a free Colab T4.

Deliverables:
- `test_predictions.json` covering all 200 test docs.
- `report.pdf`, 2 pages maximum.
- A public GitHub repo, a Hugging Face model with model card, and a W&B project. All three are published only after GG approves the final text and numbers.

---

## 2. Hard constraints

- **Inference** must run on one GPU with at most 24 GB, or on CPU.
  - Open-weight models only. No paid APIs at inference.
  - The primary target is free Colab T4: 16 GB, fp16, no bf16.
- **No hand-labelling of test data, and no tuning on test.** Test images are touched only by:
  - final inference, and
  - label-free layout clustering, used for date-format inference and as a sanity check on the unseen-supplier share.
- **Extraction rules:**
  - Output exactly what is printed. Use null when a value is missing or illegible.
  - Never correct, complete, or infer a value.
  - Validators and verifiers may change *confidence* and *null decisions*. They never change a value.
- **Confidentiality:** the brief says "do not distribute."
  - `assignment/` and `data/` are gitignored and never committed.
  - No data, images, or extracted values go to Hugging Face or to the public W&B project.
  - A separate **private** W&B project holds per-doc debug tables.
- **Determinism:** fixed seeds, greedy decoding, pinned dependencies (`uv.lock`), pinned model revisions (commit hashes).

---

## 3. Compute plan

| Where | Used for |
|---|---|
| Colab T4 (primary) | All GPU inference whose results appear in the main report (T4 fp16, adapter merged) and GPU training |
| Colab L4 (optional) | GPU training only (spec §2 constrains inference, not training); reported inference numbers stay T4 fp16 with the adapter merged |
| Local Windows, CPU (32 GB RAM) | Data recon, OCR cache, post-processing, calibrator, evaluation, unit tests |
| Local RTX 3070, 8 GB (conditional) | Branch `exp/local-rtx3070`; see Phase 7 |

**Note on the 3070:** it has *less* VRAM than the T4. It adds bf16 support and throughput, not capacity, so expect a speed gain rather than an accuracy gain.

**How Colab runs are handled:**
- Claude Code (CC) writes thin notebooks under `notebooks/`. Each notebook:
  - clones the repo at a pinned commit,
  - copies `data.zip` from Drive to `/content` and unzips it (reading directly from Drive is slow),
  - runs `python -m shipdoc <cmd>`,
  - logs to W&B, and
  - saves checkpoints and outputs to `MyDrive/shipdoc-extract/runs/<run_id>/`.
- GG runs each notebook with **Runtime → Run all**.
- CC reads the results back through the W&B API. CC never asks GG to copy numbers by hand.
- Training checkpoints are written to Drive every N steps, and every job is resumable, because Colab sessions disconnect.

**Local GPU:** runs inside WSL2 Ubuntu, because Windows support for bitsandbytes and inference servers is unreliable. WSL2 is set up only if Phase 7 triggers.

---

## 4. Repo layout

```
shipdoc-extract/
  assignment/            # gitignored: brief, schema, scorer, sample submission
  data/                  # gitignored: train/dev/test
  spec.md
  README.md              # setup, one-command reproduce (Colab + local), results table, links
  pyproject.toml, uv.lock
  configs/               # YAML per experiment (model, prompt, image res, LoRA, thresholds)
  src/shipdoc/
    io.py                # page loading and ordering, deskew/rotation
    ocr.py               # OCR + cache
    extract.py           # VLM prompting + constrained JSON decoding
    merge.py             # multi-page merge
    normalize.py         # dates, numbers, codes
    validate.py          # ISO 4217, IATA, AWB shape (NNN-NNNNNNNN), schema
    paths.py             # storage path resolution (see Storage layout)
    confidence.py        # features, calibrators, null policy, review flag
    train.py             # LoRA fine-tune (bf16 on L4 or fp16 on T4; NF4 4-bit as fallback)
    eval.py              # scorer wrapper, slices, bootstrap, error taxonomy
    trace.py             # per-doc trace JSONL
    cli.py
  notebooks/             # 01_spike, 02_finetune, 03_infer_test (thin wrappers)
  splits/folds.json      # supplier-grouped CV folds
  meta/                  # slice meta for train+dev
  reports/               # recon.md, ablations.md, error_analysis.md, report.pdf
  tests/                 # pytest: normalizers, merge, scorer parity, schema
```

### Storage layout

Large artifacts live outside the repo, under the folders named by the `SHIPDOC_*` variables (HF cache, uv cache, Paddle models, OCR cache, runs, checkpoints, tmp; the README, "Paths and environment variables").
- Paths are resolved by `src/shipdoc/paths.py` with this precedence: environment variables > gitignored `.env` > `configs/paths.yaml` profiles (`local` / `colab`).
- No drive letters appear anywhere in `src/`.

---

## 5. Phases and gates

Each gate is decided on **out-of-fold (OOF) leave-supplier-out** results, compared with a **paired bootstrap** (2,000 resamples at document level).
- A change is kept only if the 95% CI of ΔOVERALL excludes 0, **or** OVERALL holds while the false-fill rate improves.
- Each gate's decision, numbers, and CIs are logged to `reports/ablations.md`.

### Phase 0 — Data recon (CPU) → `reports/recon.md`

Report every item below with counts and percentages:

1. **Basic shape:** docs by split and type, pages per doc, scanned share (`.jpg`), image resolutions.
2. **Supplier grouping key:**
   - For invoices, use `supplier_name`.
   - For waybills, test whether `carrier` or `shipper_name` tracks the layout.
   - The original target was to confirm about 20 groups. Result (recon §2): 18 invoice groups (by `supplier_name`) + 10 waybill groups (by carrier brand). Waybill groups by carrier brand do not track layout (best cluster ARI 0.133), so the waybill part of leave-supplier-out holds out printed carrier titles only, not layouts.
3. **Per-field null rates:**
   - Separate "illegible" nulls from "not printed" nulls (e.g. `customer_part_number`).
4. **Formats:**
   - Date format per supplier, and how many dates are ambiguous between DD/MM and MM/DD.
   - Number format (decimal comma, thousands separators).
   - Currency on the page (symbol vs code) versus the gold value. Specifically: is "$" alone ever labelled as USD?
5. **Purchase order:** printed per row or once per invoice, and how gold represents each case.
6. **Rows:**
   - rows-per-doc distribution,
   - rate of duplicate part numbers,
   - tables that continue across pages,
   - whether page 2 repeats the table header row.
7. **Waybill numbers:** MAWB/HAWB formats, and what share of gold MAWBs pass the mod-7 check digit (the IATA convention).
8. **Cross-links:** how often an invoice's `awb_number` matches a waybill's `mawb`/`hawb` within the same split.
9. **Label audit:** the verifier views 20 random docs (stratified) and records any label/page disagreements. This audit is limited to train and dev.

### Phase 1 — Evaluation harness (built before any model)

1. **Slice meta** for train and dev, with these tags:
   - `scanned`, `multipage`, `repeated_parts`, `illegible` (a gold header null that is not a "not printed" case),
   - `waybill`, and `supplier_group`.
   - `awb_absent` / `hawb_absent`, so false fills can be split by redaction vs absent-line (recon §3).
   - `unseen` is assigned per CV fold.
2. **CV folds:** GroupKFold, K=3, grouped by supplier over train+dev (500 docs), with every fold containing both invoices and waybills. Seed fixed; saved to `splits/folds.json`.
3. **Scorer wrapper** imports the vendored `score.py` functions without modification.
   - **Parity test:** wrapper output must equal the CLI output on 3 synthetic prediction files.
4. **Sanity baselines:**
   - gold submitted as predictions must score 100,
   - empty submission,
   - all-null submission,
   - doc-type-only submission.
5. **Bootstrap:** 95% CIs for OVERALL, its components, and every slice; plus a paired-bootstrap A/B utility.
6. **Error taxonomy**, assigned automatically from diffs, with counts per slice:
   - `doc_type`
   - `misread` (small edit distance)
   - `convention` (wrong field, or null vs value)
   - `normalization` (date or number format)
   - `row_missing` / `row_extra` / `row_split_merge`
   - `false_fill`
   - `hallucination` (value has no support in the OCR text)
7. **Synthetic-redaction dev eval:** overlay black boxes on header fields of dev pages to measure false fills on redactions. It is always labelled "synthetic" and never mixed into official or OOF numbers.

### Phase 2 — OCR cache and zero-shot VLM spike

1. **OCR:** run PaddleOCR (fallback docTR) on every page once on CPU. Cache words, boxes, and confidences.
2. **Candidate VLMs:** verify availability, license, and transformers support at run time. VRAM figures below are estimates until measured on T4.
   - About 4B instruct model, fp16.
   - About 7–8B instruct model, 4-bit.
   - One document-specialised open VLM, if one fits.
   - **Spike arms** for each candidate: (i) image only; (ii) image + OCR text in the prompt (reading order, token-budgeted).
3. **Spike:** 40 dev docs, stratified across all slices. For each candidate, measure:
   - OVERALL, false-fill rate,
   - seconds per page, peak VRAM on T4,
   - JSON validity rate.
   - Pick one model, breaking ties by accuracy → false fill → latency.
   - **Ranking uses a paired bootstrap.** If the top-two candidates' 95% CIs overlap, run both on all 100 dev docs before choosing.
   - Report the scanned slice explicitly, because test is 47.1% scanned vs 38.1% in train/dev (recon §1).
   - **Keyed vs compact format A/B (3a; GG-approved change, motivated by the Step L root cause #2: rotations/column shifts among supplier_part_number, customer_part_number and purchase_order, `reports/spike_diagnosis.md` hypothesis (b)).** The six spike40 configs stay compact (prompt v2). After the spike40 ranking and before the dev100 decision, the notebook reruns rank #1 in the keyed format (`configs/spike_<model>_<arm>.yaml`: `output_format: json`, full field names in declared order, numeric fields typed string|number|null, prompt v2, `max_new_tokens` 1536) on spike40 as run_id `ab_keyed_<keyed config>_<sha7>` with its own `ab_status.json`, then runs `scripts/format_ab.py`. It reports, for each format, OVERALL and row F1 with their own 95% CIs, a paired bootstrap (2000 resamples, seed 42) of the difference keyed − compact for OVERALL and row F1 (delta, CI, whether the CI excludes 0), per-field row accuracy (`line_item_field_accuracy`), rotations, clean swaps and customer_part_number == purchase_order rows (`shipdoc.diagnostics.row_convention_errors`, the Step L definitions), and mean and p95 s/page from `trace.jsonl`.
   - **Decision rule (verbatim):** choose KEYED if (row F1 keyed > compact AND the paired CI of Δrow F1 excludes 0) OR (rotations keyed < rotations compact AND OVERALL keyed ≥ compact, where **"OVERALL not worse" = point ΔOVERALL ≥ 0 OR the paired CI of ΔOVERALL includes 0**; the operational definition is stated explicitly rather than left as a lower-bound threshold). Otherwise choose COMPACT on speed (reason "no gain; compact on speed").
   - **dev100 format policy:** rank #1 uses the A/B decision. Rank #2 has no A/B of its own, so it uses compact, unless the A/B chose keyed: then rank #2 also runs keyed, because the effect is format-level. If the keyed run fails or OOMs, everything falls back to compact and the reason is recorded in `ab_status.json`. The final notebook line also prints `ab_decision=<keyed|compact>`.
   - **Evidence ambiguity on the production config (GG decision, no further format spend).** Production config = qwen35_4b img_only KEYED. The evidence is ambiguous: keyed vs compact on img_ocr (40 docs) gave ΔOVERALL -2.82 pts [-9.12, +2.64], Δrow F1 -4.55 [-18.85, +7.48], +25% s/page, and the rule chose keyed via the rotation clause (34 vs 42 rotations); img_only vs img_ocr on dev100 is +1.34 pts [-3.23, +6.08]; keyed was never measured on img_only. Numbers copied from S4 and Findings of `reports/respike.md`, which marks all of them UNVERIFIED until a verifier recomputes them.
   - Cost: the A/B adds one keyed spike40 run (55 pages at keyed speed, n_out = JSON-format gold mean tokens/page from `reports/token_budget.md`, p99 as the upper bound, costliest model's measured speed as the worst case); `scripts/gpu_estimate.py` and the notebook's estimate cell show it, and the dev100 top-2 at both compact and keyed speed, in T4 hours and compute units at 1.19 and 1.58 CU/h.
   - Smoke gate addition: a non-blocking WARNING counts emitted rows whose customer_part_number equals (identifier rule) the row's purchase_order, logged per model in the smoke table and `smoke_status.json`.
4. **Prompting:**
   - Extract per page, with schema-constrained JSON decoding.
   - The prompt states the extraction rules verbatim (copy exactly, use null when illegible).
   - Tune `max_pixels` for the accuracy/latency trade-off.
5. **G2:** the chosen model runs zero-shot over all 500 docs. Zero-shot is unseen by construction, so this becomes the baseline OOF.

### Phase 3 — Deterministic post-processing (unit-tested)

1. **Multi-page merge:**
   - Header fields use a per-field page-provenance table learned from train. Identity fields come from the page-1 header; totals come from wherever provenance says they are. A value seen only in a continuation banner never fills a field. (Recon §9 page-2 trap: train_0091, train_0253, train_0282 print a legible invoice number in the page-2 banner while gold is null; see also recon §3.)
   - If pages disagree on a field, lower its confidence.
   - Concatenate rows across pages and drop repeated table-header rows.
   - **Never** deduplicate identical rows: repeated rows are real.
2. **PO propagation** is conditional on the Step H result (OCR check of the 8 no-PO-column groups); default: no propagation.
   - Step H verdict: **no propagation.** An OCR grep of all train/dev pages of the 8 no-PO-column groups (171 docs) found 0 PO labels and 0 PO-shaped tokens, and no PO-printing group prints a header-level PO either (PO is only a per-row column). The "header PO printed, gold null" branch does not apply; the "propagate only if a header-level PO is detected and the table has no PO column (low-confidence)" branch never fires, so `purchase_order` stays null for those groups.
   - Evidence and caveats in `reports/po_check.md` (counts UNVERIFIED); script `scripts/po_check.py`.
3. **Dates:**
   - Normalise every date to ISO.
   - Infer the date format per layout cluster: any date in the cluster with day > 12 resolves it.
   - If a date stays ambiguous, keep the value and apply a confidence penalty.
4. **Numbers:** emit plain numbers (handle decimal commas, strip symbols). The scorer strips only commas, so we must not hand it European-format numbers.
5. **Codes:**
   - Validate currency against ISO 4217 and airports against a bundled IATA list (open dataset; cite its license).
   - AWB numbers: check the `NNN-NNNNNNNN` shape only. The MAWB mod-7 validator is removed: 16/100 gold MAWBs pass, which is chance level (recon §7).
6. **G3:** paired bootstrap of Phase 3 versus G2.

### Phase 4 — LoRA fine-tune (bf16 on L4 or fp16 on T4; NF4 4-bit as fallback)

1. **Targets:** per-page JSON.
   - Gold doesn't record which page each row is on, so assign rows to pages by fuzzy-matching OCR text to part numbers and quantities.
   - Docs whose rows can't be aligned are trained as whole-document samples if they fit the token budget. Otherwise they are excluded, and the exclusion is logged.
2. **Augmentation:** apply scan-style degradation to digital pages: rotation ±3°, blur, JPEG compression, noise.
   - Add synthetic occlusion augmentation (black boxes and scribbles) over all field types, to teach null on illegible values. Gold redactions are rare (35/192 header nulls, recon §3).
3. **Hyper-parameters:** low LoRA rank, a fixed 2-epoch schedule, NO early stopping on the CV folds (selecting on held-out loss would leak into the OOF score that decides G4; held-out loss is logged for information only). Everything lives in `configs/`.
4. **Timing first:** train one fold and record wall-clock time. Then train all K=3 folds to produce OOF predictions on held-out suppliers.
   - If K=3 doesn't fit Colab quota, drop to K=2 and document why.
5. **G4:** the fine-tuned model is kept only if its OOF OVERALL beats Phase 3 (CI excludes 0) **and** the unseen-slice false-fill rate is not worse.
   - If G4 fails, ship zero-shot + post-processing. The report states that nothing was trained, so HF and W&B are not required.
6. **Final model:** trained on all of train. Dev is the official seen-layout evaluation.
   - Log to W&B: train/eval loss curves, full config, dev OVERALL and slices.
   - Push the adapter (plus a merged model if it fits) to Hugging Face with a model card covering training data, intended use, metrics with CIs, and limitations.

### Phase 5 — Verifier and confidence

1. **Per-field features:**
   - VLM token log-probabilities (min and mean),
   - OCR fuzzy-support score and OCR confidence,
   - whether validators pass (AWB shape only; no mod-7 feature),
   - cross-page agreement,
   - field type,
   - a seen/unseen proxy (distance to the nearest training layout cluster).
   - The invoice↔waybill cross-link feature is removed: 0 links in train/dev (recon §8).
   - Self-consistency (a second pass on an enhanced image) is included **only if** its accuracy gain justifies roughly doubling latency. Measure this.
2. **Two calibrators**, trained on OOF data only with logistic regression (GBM if it beats it under CV):
   - (a) P(value correct)
   - (b) P(gold is null)
3. **Null policy:** emit null if and only if P(gold null) > P(value correct). This is the expected-score-optimal rule.
4. **Review flag:** a field is flagged when P(correct) < τ.
   - τ is chosen on OOF so that auto-accepted fields reach at least 98% precision.
   - On dev, report the achieved precision, recall, and review rate with CIs.
   - A document is flagged if any of its fields is flagged.
5. **Calibration metrics:** ECE, reliability diagram, coverage-vs-accuracy curve.
6. **False-fill reporting:** report false fills split into redaction vs absent-line (recon §3: 35 vs 157 of 192 gold header nulls), plus the synthetic-redaction dev eval from Phase 1.7, always labelled "synthetic".
7. **G5:** false-fill rate goes down, and OVERALL does not drop.

### Phase 6 — Test inference and submission

1. Run end-to-end on Colab T4, starting from the images. The final pipeline runs OCR from images on Colab; the local OCR cache is for development only.
2. Validate the submission with `jsonschema` and confirm all 200 doc IDs are present.
3. **Determinism check:** a second run must produce a byte-identical diff of 0.
4. **Label-free test diagnostics:**
   - layout clusters, using an OCR keyword+position layout signature (not image embeddings), and the estimated unseen share (expected about 50%),
   - the predicted review rate.
5. Record seconds per page and peak VRAM, and estimate $ per 1,000 docs using a cited public T4 hourly price (labelled as an estimate).

### Phase 7 — Local RTX 3070 branch (conditional)

1. On branch `exp/local-rtx3070`, under WSL2, run the same pipeline using bf16 where the model fits in 8 GB.
2. **Show the results only if** one of these is measured:
   - an accuracy gain where the paired-bootstrap CI excludes 0, or
   - at least a 1.5× throughput gain at equal accuracy.
3. If neither holds, the branch stays unmerged and does **not** appear in the report or README (GG's hard rule).

### Phase 8 — Report and publish (GG approval gate)

`report.pdf`, 2 pages maximum:
1. **Approach and why:** generalisation over templates, and the safety architecture (verbatim extraction, verifier, calibrated nulls).
2. **Results table:**
   - official dev score (seen layouts) and leave-supplier-out OOF score (unseen layouts), with **invoice-only OOF as the headline** and overall OOF alongside,
   - false fills split into redaction vs absent-line, plus the synthetic-redaction eval (labelled "synthetic"),
   - every slice, with 95% CIs and n.
3. **Ablation ladder** with paired deltas.
4. **Failure analysis:** taxonomy counts plus 2–3 concrete examples, each with root cause and the fix that would address it.
5. **New supplier arriving tomorrow:**
   - Day 0: zero-config extraction; low-confidence fields go to a human reviewer.
   - Reviewer corrections feed an active-learning queue.
   - LoRA is refreshed periodically, gated on supplier-held-out evaluation.
   - Drift is monitored per layout cluster (review rate, null rate).
6. **Review flag:** precision, recall, review rate, and the business framing (share of fields auto-accepted at ≥98% precision).
7. **Cost and latency on T4.**
8. **Links:** GitHub, Hugging Face, W&B.

**Publish only after GG approves** the final text and every number in it, after each number has been re-verified against the logged artifacts.

---

## 6. Logging and tracing

- **Public W&B project `shipdoc-extract`:** config, loss curves, OVERALL and slices with CIs, ablation table, calibration plots. No images, no extracted values.
- **Private W&B project `shipdoc-extract-debug`:** per-doc tables (images, gold, predictions, flags) for train and dev only.
- **`runs/<run_id>/trace.jsonl`, one line per doc**, containing:
  - raw VLM output and parsed JSON,
  - post-processing diffs,
  - validator and OCR flags,
  - confidences and decisions,
  - latency per page, peak VRAM,
  - model revision, git commit, config hash.
- Every reported number must trace back to a run ID and artifact.

---

## 7. Reproducibility

- `uv` with Python 3.11; pinned lockfile; pinned model commit hashes.
- Global seed set in Python, NumPy, and torch. Deterministic flags on where supported. Greedy decoding.
- The README provides one command per stage plus a Colab "Run all" path from raw images to `test_predictions.json`.
- **Final check:** run a clean-clone reproduction on a fresh Colab session and diff the output against the submitted file.

---

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Fine-tune overfits to about 20 layouts | Supplier-grouped CV gate (G4); low LoRA rank; augmentation |
| Ambiguous DD/MM dates on unseen suppliers | Per-cluster format inference; confidence penalty |
| VLM hallucinates values on illegible fields | OCR-support feature; P(null) calibrator; expected-score null rule |
| Rows missed or duplicated at page breaks | Per-page extraction + merge rules; row-count consistency checks; `row_*` taxonomy |
| Colab disconnects or quota limits | Checkpoint to Drive; resumable jobs; small spike before full runs |
| Small dev n=100 (about ±8 pts CI) | Bootstrap CIs everywhere; headline on 500-doc OOF; no over-reading of slices |
| Confidential material leaks | gitignore; public/private W&B split; pre-publish audit of repo contents |

---

## 9. Schedule

| Day | Work |
|---|---|
| 1 | Phase 0 recon; Phase 1 harness, folds, meta, parity tests; OCR cache |
| 2 | Phase 2 spike → model choice; zero-shot OOF on 500 docs |
| 3 | Phase 3 post-processing; error analysis; fine-tune data prep and one timed fold |
| 4 | Phase 4: K folds → G4 decision; final model on train; dev evaluation |
| 5 | Phase 5: confidence, null policy, review flag; G5 |
| 6 | Phase 6 test inference + determinism check; Phase 7 (conditional); README, model card |
| 7 | Phase 8 report; clean-room reproduction; GG approval → publish |

---

## 10. Working agreements for Claude Code

- Every phase ends with a **detailed per-item report**: what was done, evidence (command, file path, numbers), what changed, what remains, and open risks. No compressed summaries.
- The verifier subagent independently re-runs evaluations and spot-checks 10 docs against their images at every gate.
- Each number is marked **verified** (re-computed from an artifact) or **unverified**.
- Tasks that need GG (running Colab notebooks, approvals) are listed explicitly as numbered steps with exact URLs, buttons, and field values. CC then waits for GG's confirmation before continuing.
- Nothing is published, made public, or pushed to Hugging Face without GG's explicit approval of the final content.

---

## 11. Closing plan, pre-registered (GG, 2026-10-03, before any 05 / 06 result)

Written before the fold-1 / fold-2 OOF runs, the 06 resolution sweep and the 07 header-hint A/B have
produced a number, so none of the rules below can have been fitted to those results. Evidence
already in hand when this was written (all in `reports/`, see the audit trail there): the 02
zero-shot run, the R1/R2/R3 gate, the G3 500-doc re-gate, calibration v2, the v0 submission and the
fold-0 training run (loss curves only; no fold-0 OOF result).

1. **Final-system rule.** The final system is **FT + rules** only if, on the 3-fold out-of-fold
   comparison on all 500 train+dev docs, FT+rules beats ZS+rules in OVERALL with the lower bound of
   the paired doc-level 95% CI (2000 resamples, seed 42, unmodified scorer) above 0, AND the
   false-fill count and the over-null count (header + row cells, as defined in
   `shipdoc.oof.over_null_counts`) of FT+rules, pooled over the 3 folds, are not worse than
   ZS+rules'. Otherwise the final system is **ZS + rules (v1)**. A tie goes to the simpler system
   (ZS + rules). Both arms use the same production rules and the same honest per-fold R3 shapes.
   This is the spec section 5 Phase 4.5 G4 gate with the baseline made explicit (ZS+rules, not raw
   ZS), over-nulls added, and the tie-break stated.
   **Clarification, still before any fold-1/2 OOF result (2026-10-03):** the "false-fill count" is
   `shipdoc.oof.over_null_counts(...)["false_fill_total"]` (header cells plus scorer-paired row
   cells where the gold is empty and the prediction is not) and the "over-null count" is
   `over_null_counts(...)["over_null_total"]` (the reverse), each summed over the 500 docs of the
   three folds. "Not worse" means the FT+rules count is less than or equal to the ZS+rules count,
   compared as raw counts with no tolerance and no CI. The rule is implemented as a pure function
   with tests before any 3-fold result exists.
2. **Resolution.** A higher input resolution is adopted only by the 06 rule: the highest swept
   resolution whose paired OVERALL delta CI lower bound versus the production resolution is above 0
   and whose peak VRAM is at most 14.5 GiB (ties to the lower resolution). Fine-tuned retraining at
   a new resolution happens only if a resolution is adopted AND fold 0 passes G4 (the interim
   verdict of `reports/g4_fold0.md`).
   Result (2026-10-03): the 06 rule adopted max_pixels 2196480 (2145 tokens); see reports/res_sweep.md.
3. **R3.** R3 is judged only on folds where it fires (fold 1 / fold 2 OOF); fold 0 can neither
   clear nor sink it. Its concentration (3 of 18 invoice groups carry 493 of 502 fixed rows) is
   disclosed in the report.
4. **Scope cuts.** The zoom re-read (`src/shipdoc/zoom.py`) is future work and appears in the report
   only as such. The header-hint A/B (07) runs only if the fine-tuned row-error cause table still
   shows at least 30 column-shift rows.
5. **Kernel note.** `causal_conv1d` and `flash-linear-attention` are absent from the inference
   environment (reference fallback is used). Not changed for this submission. The report records it
   as a production latency lever, marked UNVERIFIED (no measurement exists).
6. **Review flags.** Calibration review flags for the test set are written to a separate file, never
   into `test_predictions.json` (the schema has `additionalProperties: false`).
7. **Secondary system G (per-field gate), pre-registered 2026-10-03 before any fold-1 / fold-2 native
   05n result exists.** Disclosure: the fold-0 native 05n result (FT vs ZS, both at 2196480) had
   ALREADY been seen when this item was written, so G was not designed blind to fold 0; fold 1 and
   fold 2 are the untouched evidence. For each header field and each aligned row field, G outputs the
   value from whichever arm (FT+rules or ZS+rules) has the higher cross-fitted calibrated P(correct),
   using the `calibrate_v3` agreement features. Calibrators are fitted out-of-fold by supplier fold,
   and the gate is never fitted on the docs it is applied to. Rows are aligned between the arms with
   `confidence_v3.align_rows`; a row that exists in only one arm is kept if that arm's P(row correct)
   is >= 0.5 and dropped otherwise. No value is ever synthesized: G only selects among values the two
   arms produced.
   G replaces the pre-registered winner ONLY IF, on pooled 3-fold OOF: the paired OVERALL CI lower
   bound (2000 resamples, seed 42, unmodified scorer) is > 0 versus the winner, AND G's
   `false_fill_total` <= the winner's, AND G's `over_null_total` <= the winner's (same definitions
   and raw-count comparison as item 1). The primary rule (item 1) is unchanged and is decided first;
   G is never evaluated as a substitute for it. Any fold-0-only run of G is EXPLORATORY and NOT A
   DECISION.
   **Clarification of item 7 (2026-10-03, still before any fold-1 / fold-2 native 05n result; fixes
   choices the text above left open, as implemented in `src/shipdoc/gate.py` and
   `scripts/gate_eval.py`, which were written after item 7 and run on fold 0 only, EXPLORATORY):**
   (a) calibrator variant: `v2_agree` (v2 features plus the `calibrate_v3` agreement features), pooled
   structure, one calibrator per judging view; (b) a field present in only one arm (header or aligned
   row slot) is kept if that arm's P(correct) >= 0.5, else the other arm's value (blank) is written;
   (c) row-level P(correct) for the one-arm-row rule is the minimum P over the row's emitted fields, and
   a row with no emitted field is dropped; (d) ties between arms go to ZS+rules; (e) `doc_type`
   disagreement: the ZS document's `doc_type` is written; (f) output rows: aligned and ZS-only rows in
   ZS order, then FT-only rows in FT order. The pooled run cross-fits across the three supplier folds
   (fold k's gate calibrators are fitted on the other two folds only). The replacement conditions
   in item 7 are unchanged.
   **Addendum to item 7, review flags (2026-10-03, before any fold-1 / fold-2 native 05n result):** if
   G replaces the winner, G's review flags come from a calibrator fitted on G's own pooled OOF output
   (the same `v2_agree` cross-fitted pipeline, supplier folds, nested tau) and frozen as
   `meta/calibrator_g_native.json`. There is no max-of-two-P shortcut. If G does not win, nothing is
   built.
8. **Native header-hint A/B (07n), pre-registered 2026-10-03 before it runs.** Hint arm versus control
   (the 02n native ZS run, same config and batch size 4 except the hint) on the multipage documents,
   both arms with the production rules. SHIP into the ZS+rules system only if the paired OVERALL CI
   lower bound (2000 resamples, seed 42, unmodified scorer) is > 0 AND `false_fill_total` of the hint
   arm <= the control's AND `over_null_total` of the hint arm <= the control's (same definitions as
   item 1, raw counts). It applies only if ZS+rules is the final system; otherwise it is reported as
   exploratory. The notebook refuses to run unless the control's prompt and config hashes match with
   the hint switched off. Supersedes the 1260-config scope cut of item 4 for the native build only;
   item 4's column-shift trigger is met by the native ZS+rules run (252 rows, see
   `reports/hdrhint_estimate_native.md`), whose oracle ceiling (+2.80 OVERALL, +2.46 for the
   hint-addressable rows) was computed before the build.
   **Item 8 status (2026-10-03, same day, before any fold-1 / fold-2 native 05n result): CANCELLED, not
   built, never run.** The 07n notebook, builder and config were not completed and none was committed
   (a partial build was removed from the tree; no result exists for it). 07n is future work; its
   pre-registered gate above is kept for the record and applies only if it is revived. Oracle ceilings
   computed before any build: +2.80 OVERALL for all 252 column-shift rows and +2.46 for the 220
   continuation-page rows a hint can address (`reports/hdrhint_estimate_native.md`).
   Dated note (2026-10-04, GG): 07n cancelled before build completion due to the submission deadline;
   no hint arm was run; reported as future work with the pre-build oracle ceiling (+2.46
   continuation-page subset, +2.80 all 252 shift rows).
9. **Closing note: which system is submitted (2026-10-03, dated before any fold-1 / fold-2 native 05n
   result; submission due Sunday 2026-10-04 17:00 IST, internal target 16:00).**
   - **v1.5** = native zero-shot + production rules + review flags (`meta/calibrator_zs_native.json`,
     flags stage run locally on CPU) is ALWAYS built and validated. It is the default submission.
   - **v2** = fine-tuned + rules replaces v1.5 ONLY IF (a) the pre-registered pooled rule (item 1)
     selects FT+rules, AND (b) the 04c `MODEL = "ft"` submission is validated (schema, 200 of 200 ids,
     determinism, flags) by 13:00 IST on Sunday 2026-10-04. Otherwise v1.5 ships.
   - **G** (item 7) gets no test-set inference path; it is reported as evaluated only. Nothing is built
     for its review flags unless it wins, and even then no submission path is built for this deadline.
   - Decision order when the 05n fold-1 / fold-2 results land: the pooled rule (item 1) first, then G
     (item 7); each gets its own report. Unchanged: nothing public without GG's explicit approval.
   - **Deviation note (2026-10-04, GG; written before any pooled 3-fold result exists; 05n fold 2 and
     the final adapter were still running):** operational cutoff moved 13:00 -> 14:30 IST because
     final training started at 07:30 instead of ~04:20. The selection rule (item 1, pooled 3-fold) is
     unchanged; there are no test labels, so the move cannot bias selection. If v2 is not validated
     by 14:30 IST, v1.5 ships. The 08 dev run of the final adapter is now optional (run only if the
     T4 is free after 04c ft and it lands by 14:30 IST); otherwise the report uses the pooled
     supplier-held-out OOF slices for the v2 system and states that the final-adapter dev run was not
     done. Seen before this note: the fold-0 and fold-1 native results (fold 1: the raw banner).
