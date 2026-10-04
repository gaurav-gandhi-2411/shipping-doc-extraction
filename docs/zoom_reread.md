# Zoom re-read of low-confidence fields

Status: DESIGN + CPU PROTOTYPE. Nothing here has run on a GPU, so there is no accuracy or latency
result. **The zoom re-read ships only if it passes the held-out paired gate below; otherwise DO NOT
SHIP.** It is opt-in and default OFF (`ZoomConfig.enabled = False`): with it off, `zoom_document`
returns a copy of the prediction without touching the backend, the images or the OCR index
(`tests/test_zoom.py::test_default_off_touches_nothing`).

Code: `src/shipdoc/zoom.py`. Cost estimate: `scripts/zoom_estimate.py`. Tests: `tests/test_zoom.py`.

## 1. What it does

For each field with calibrated P(correct) < tau (the Phase 5.4 review flag):

1. **Crop** the region at native resolution from the original page image (PIL, no resize unless the
   crop exceeds a pixel cap, see 3). Regions come from OCR only, never from gold:
   - *row field*: the row's OCR line (`locate.assign_rows` over all predicted rows, so repeated part
     numbers get distinct lines), taken as the union of the line's OCR item boxes, widened to the
     table width (`layout.analyze_page` table lines; `row_band_width="page"` for the full page
     width), padded 2% horizontally and 50% of the band height vertically, grown to at least
     256 x 64 px. With `include_table_header` (default) the table's column-heading line is cropped
     the same way and stacked on top, because the column labels disambiguate customer part number,
     PO and quantity. A row whose line cannot be found is skipped, with no call.
   - *header field*: the whole OCR line of the best match of the emitted value
     (`locate.find_matches`, exact < normalized < fuzzy; the line keeps the label in the crop),
     padded 10% horizontally and one line height vertically. With no OCR support (null value or no
     match) the fallback is the layout region from `layout.analyze_page`: `footer` for total_amount,
     pieces, gross_weight_kg, `header` otherwise. If that region is empty the field is skipped.
2. **Re-query the same backend** (`extract_page`, same model, greedy, xgrammar) with a tiny keyed
   schema and a new prompt (module constants in `zoom.py`; `prompts.py` is untouched):
   - header: `{"header": {<field>: value}}`;
   - row: `{"line_items": [ {4 row keys} ]}`, exactly one object. Flagged fields of one row share one
     call. If the row's supplier_part_number is itself UNFLAGGED it is named in the prompt
     ("the row whose supplier_part_number is X") and a re-read that returns a different part number
     is discarded whole (`reread_identity_mismatch`). A flagged part number is never given as a hint:
     it would anchor the model on the value we doubt.
   - The prompt carries the extraction rules verbatim (`prompts.RULES`, `NULL_RULE`,
     `PROVENANCE_RULE`, `CONVENTION_RULES`) plus "Read only this crop". No OCR hint is passed, no
     supplier or layout identifiers appear anywhere.
3. **Accept only if more confident** (section 4), only for FLAGGED fields (a row re-read returns all
   four keys, the unflagged ones are discarded), under the null policy of section 5.

## 2. Interfaces

### 2.1 Input from the calibrator (defined in `zoom.py`)

```python
FieldScore(doc_id, page, field, row_index | None, p_correct, value, logprob_conf=None)
select_for_reread(scores, tau) -> list[FieldScore]   # p_correct < tau, strict; NaN skipped
```

- `page`: 0-based page of the document the field was read from.
- `row_index`: index into the DOCUMENT-level `line_items` list of the prediction passed to
  `zoom_document` (None for header fields). `confidence.FieldKey.row_idx` is the same index space
  (-1 for header).
- `p_correct`: calibrated P(current value is right). For a NULL field it is the calibrator's
  P(gold is null) (`p_null`), i.e. the probability the null is right.
- `logprob_conf`: raw MIN token logprob of the value (`field_logprobs[...]["min"]`, <= 0), needed
  only by the interim `LOGPROB_MIN` comparison.

### 2.2 Adapter expected from `scripts/calibrate.py` output (NOT yet wired)

`calibrate.py` writes `<prefix>_oof_fields.csv` with columns
`doc_id, scope, field, row_idx, emitted, p_correct, p_null, nulled, y_correct, y_null`
(read-only look at the dry-run file `$SHIPDOC_TMP_DIR\calib_scratch\dry\DRYRUN_dev100_oof_fields.csv`).
`zoom.scores_from_oof_rows(rows, resolve)` turns csv rows into `FieldScore`s (score = `p_correct` if
`emitted` else `p_null`; `row_idx` -1 becomes None). The CSV has no value, page or logprob, so the
caller supplies `resolve(doc_id, scope, field, row_idx) -> (value, page, logprob_min)` from the
prediction JSON and its `field_logprobs` (`confidence.row_sources` gives the page of a row). For
production (non-OOF) use the same function on the final calibrator's per-field output, with the
final tau.

### 2.3 Confidence on one scale (injected)

`ConfidenceModel(original(score), reread(score, reread))`:

- `LOGPROB_MIN` (interim): original = `score.logprob_conf`, re-read = the re-read value's min token
  logprob, both computed by `field_logprobs` on the same keyed output format. Like with like, but
  uncalibrated.
- `calibrated(predict)` (**the gate must use this**): original = `score.p_correct`; re-read =
  `predict(score, reread)`, the SAME fitted calibrator applied to the re-read's features. The
  re-read supplies `lp_min`, `lp_mean`, `n_tokens`; the OCR support / validator features are
  recomputed for the re-read value (`locate`), `was_null` from the re-read value. Features that need
  page context (`xp_*` cross-page agreement, layout novelty) are copied from the original field;
  that is a known approximation to be stated in the gate report.

## 3. Parameters (`ZoomConfig`, all frozen before the gate)

| parameter | default | meaning |
|---|---|---|
| `enabled` | False | master switch (off = byte-for-byte unchanged prediction, zero calls) |
| `tau` | 0.0 | select `p_correct < tau`; the calibrator's review tau on the training folds |
| `null_fill` | `ocr_supported` | null -> value policy, section 5 |
| `accept_margin` | 0.0 | re-read must beat the original by MORE than this (same scale) |
| `allow_value_to_null` | False | accept a null re-read of a non-null value |
| `max_calls_per_doc` | None | cap; lowest-confidence tasks first |
| `header_pad_{x,y}_frac` | 0.10 / 1.0 | padding of the matched line (fraction of its width / height) |
| `row_pad_{x,y}_frac` | 0.02 / 0.5 | padding of the row band |
| `row_band_width` | `table` | `table` or `page` |
| `include_table_header` | True | stack the column-heading line above the row |
| `min_crop_w`, `min_crop_h` | 256, 64 px | minimum crop size |
| `max_pixels` | 1280*32*32 | pixel cap per crop; None = never resize in zoom.py |

Pixel policy: the crop is cut at the source image's native resolution. `zoom.py` resizes only a
crop larger than `max_pixels` (default = the page cap of `BackendConfig`, 1,310,720 px; one visual
token per 32x32 px), uniformly with LANCZOS. Beyond that the model processor applies its own
min/max pixel rules. Honest limit (arithmetic from stated constants, not a measurement): a full
1240 x 1754 page (2,174,960 px) is downscaled by sqrt(1,310,720 / 2,174,960) = 0.776 under the
default cap, so a native-resolution crop is at most 1/0.776 = 1.29x sharper linearly unless the
processor's `min_pixels` upscales small crops. Whether zoom helps at all is exactly what the gate
measures; the prototype does not upscale (a possible lever for a later, separately gated variant).

## 4. Acceptance rule

Keep the original unless ALL hold, evaluated per flagged field (`zoom.decide`, pure, fails closed):

1. the re-read call parsed, and carries logprobs (`reread_unparsable`, `reread_no_logprobs`);
2. for a row call, identity matches when an unflagged part number was supplied;
3. the re-read value differs from the original after alphanumeric normalisation (equal = `agree`);
4. both confidences are finite and `new > old + accept_margin` (`no_confidence`,
   `not_more_confident`);
5. the null policy of section 5 allows it.

A re-read is never accepted on its own say-so: a missing or NaN confidence on either side is a
reject.

## 5. null -> value: explicit design choice, FLAGGED FOR GG

Spec rule: "never fill a value the model did not emit". The zoom re-read IS a model emission, so
filling a null from it does not literally break the rule, but the largest false-fill source is a
null that the page really lacks (157 of 192 gold header nulls are absent lines, recon section 3), and
a second, tighter-prompted read of an empty crop can still hallucinate. Default therefore:

- `null_fill="ocr_supported"`: null -> value only if the re-read passes the same confidence rule
  AND the value is found in the document's OCR by `locate.find_matches` (fuzzy threshold of the
  locator). Otherwise the null stays null (`no_ocr_support`).
- `"never"` forbids it; `"any"` allows it on confidence alone.
- The gate's false-fill criterion is the backstop: if this policy increases false fills the pipeline
  does not ship, whichever mode was chosen.
- value -> null is ignored by default: the calibrator is trained on non-null emissions, so a null
  re-read is not on the same footing as the original value.

## 6. Cost

`uv run python scripts/zoom_estimate.py --fields-per-doc 45` prints the table below (ESTIMATE, not
measured). Inputs: `configs/spike_speed.json` (qwen35_4b_img_only: prefill 2.071 s, decode 13.132
tok/s, 45.87 s/page), ASSUMED crops 1240x128 (header) and 1240x192 (row), 600 prompt tokens, 30 / 70
output tokens, half the flagged fields in rows, 1.5 flagged fields per row call, 1.375 pages/doc
(UNVERIFIED). `linear..fixed` bounds the per-call prefill: the trace speeds cannot separate prefill
from fixed per-call overhead (see the config), so the bounds are lower (prefill scales with tokens)
and upper (the whole intercept is paid per call).

| flag rate (ESTIMATE) | flagged fields/doc | extra calls/doc | extra T4 s/doc | pages-equivalent | overhead vs baseline |
|---|---|---|---|---|---|
| 5% | 2.25 | 1.88 | 8.2..10.5 | 0.18..0.23 | 13%..17% |
| 10% | 4.50 | 3.75 | 16.4..20.9 | 0.36..0.46 | 26%..33% |
| 20% | 9.01 | 7.51 | 32.8..41.8 | 0.72..0.91 | 52%..66% |
| 30% | 13.51 | 11.26 | 49.2..62.7 | 1.07..1.37 | 78%..99% |
| 74.4% (DRY RUN) | 33.51 | 9.86 | 59.6..71.0 | 1.30..1.55 | 94%..113% |

The DRY RUN row uses the flag counts of `DRYRUN_dev100_oof_fields.csv` at tau 0.96864 (the dry-run
in-sample tau of `DRYRUN_dev100_calibration.json`). It is a mechanics check of the script, not a
Phase 5 result: the dry-run tau is near the unattainable end, which is why its rate is so high. Only
the real calibrator's flag rate on OOF data is a usable input. Because each re-read costs a large
share of a page, `max_calls_per_doc` is the budget lever, and the latency overhead is a gate output.

## 7. Gate protocol (ships only if ALL pass)

Systems compared on the SAME documents: A = pipeline without zoom, B = A + zoom. Predictions are
OUT-OF-FOLD: zero-shot, or the fold adapter's OOF inference (`oof.py`), under supplier-held-out
folds (`splits/folds.json`, the same folds as the calibration).

1. **Freeze on other folds.** For each evaluated fold k, the calibrator, tau and EVERY zoom
   parameter of section 3 are fitted or chosen using folds other than k only. The zoom parameter
   grid is pre-registered and small (`null_fill` in {never, ocr_supported}, `accept_margin` in
   {0, one calibrated-scale value fixed before looking}, `row_band_width` in {table, page}); the
   choice per fold k uses the same criteria as 2-3 below, computed on folds other than k. Fold k is
   scored once with the frozen setting. No parameter is tuned on the data it is scored on.
2. **OVERALL improves.** Paired doc-level bootstrap of B - A with `eval.paired_bootstrap(pred_A,
   pred_B, gold, n=2000, seed=42)`, pooled over all evaluated folds: the 95% CI lower bound of
   OVERALL delta must be > 0.
3. **False fills not worse (margin 0, GG decision 2026-10-03).** The false-fill COUNT of B must
   not exceed that of A (pooled over the evaluated folds; no tolerance, no CI-based slack), which
   also means the `false_fill_rate` point delta is <= 0. Report the paired-bootstrap CI of the
   rate delta for information. Report the false-fill split (redaction vs absent line) for A and B,
   and the count of null -> value replacements and how many were correct.
4. **Latency overhead reported:** extra model calls per doc, extra T4 s per doc and the ratio to
   the baseline s/doc, measured on the T4 from `ZoomResult.n_calls / latency_s / n_input_tokens`,
   not from `zoom_estimate.py`. There is no pass threshold fixed in advance; a CI-positive gain that
   costs, say, a doubling of latency is reported and decided by GG, but a failure of 2 or 3 is
   final.
5. **Provenance:** every number carries the commit SHA, the config (`ZoomConfig`, `zoom_prompt_hash`,
   model revision) and the report artifact. The replacement log (`ZoomResult.decisions`, one row per
   flagged field with reason and both confidences) is stored per document.
6. **Determinism:** two runs give a byte-identical prediction (greedy, seed 42, canonical call
   order; `test_input_is_never_mutated_and_run_is_deterministic` covers the CPU path).

If any of 2 or 3 fails, or the folds do not allow freezing parameters out-of-fold, DO NOT SHIP, and
write the negative result up (spec section 10). One clean measurement is needed before calling the
idea dead: crop quality (does the crop contain the field, per located-line rate) should be reported
so a failure is attributable to the lever and not to the plumbing.

## 8. What remains to run the gate

- **GPU:** the re-read calls need the real backend (`HfBackend`, T4) with `capture_logprobs=True`
  (zoom reads the logprob trace from `meta["logprob_trace"]`; without it nothing is accepted). The
  keyed `json` output format is required (`ZoomReader` raises on `compact`). NuExtract3's adapter
  builds its template from the page schema and ignores the zoom schema: UNVERIFIED, test first on
  Qwen3.5-4B (the adapter that takes the schema through the grammar and the prompt).
- **Calibrator output:** `scripts/calibrate.py` OOF fields (per-field P(correct), P(null)), the fitted
  calibrator object to build `calibrated(predict)`, and tau per fold. Wire: OOF CSV ->
  `scores_from_oof_rows` -> `zoom_document` per document with the page images and cached OCR
  (`ocr.doc_pages`).
- **A driver** (notebook or script) running A and B over folds, and the gate computation (the
  bootstrap itself exists: `eval.paired_bootstrap`). Not written: out of scope for this prototype.
- OOF predictions of A on all folds (zero-shot, or fold adapters from `oof.py`).

## 9. Risks

- The re-read is the SAME model under greedy decoding on a similar input: its errors are correlated
  with the original's, and its logprob is often high even when wrong. The gate, not the logic, decides.
- Re-read features are partly out of distribution for the calibrator (crop has no page context).
- Crop quality: OCR misses or a wrong row assignment give a wrong crop; counted as `no_region`,
  `row_not_located`, `identity_mismatch` in the accounting, but a wrong-yet-plausible crop is silent.
- Cost: at a 20% flag rate the estimate is roughly 0.7 to 0.9 extra page-equivalents per doc.
- Tested only with a scripted backend and the repo mock; the xgrammar `minItems/maxItems` on the
  one-row schema is covered by the compiler's documented support (extract.py compact schema notes),
  not by a run here.
