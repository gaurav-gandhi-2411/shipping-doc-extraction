# Header-hint (07) at the native resolution: ORACLE upper bound and T4-hour estimate

**Provenance.** Written 2026-10-03 at repo state `ab92591+dirty` (uncommitted edits elsewhere in the tree; nothing here changed any code under `src/`). Scratch scripts (not committed): `$SHIPDOC_TMP_DIR\hdrhint\est_oracle.py`, `est_oracle2.py`, `est_time.py`, `est_queue.py`, run as `uv run python <script>` (raw outputs next to them). Unmodified official scorer (`assignment/score.py` via `shipdoc.eval`), paired doc-level bootstrap 2000 resamples, seed 42. Aggregates and counts only: no document value, no doc id. **No decision is taken here; the 07 deferral stays open** until the FT-native row-error table and the pooled decision (spec section 11). Every number is UNVERIFIED until a verifier recomputes it, unless marked VERIFIED (a command was run and its output read this session), ESTIMATE (derived, inputs stated).

## 0. The run behind the numbers (provenance check)

The scratch run dirs under `$SHIPDOC_TMP_DIR\native_work\rowerr\` hold only `predictions.json` + `trace.jsonl` (no `manifest.json`), so their provenance cannot be read from a manifest. It was re-derived instead:

| check | result |
|---|---|
| scratch `..._postrules_500/predictions.json` scored by the unmodified scorer, 500 docs | OVERALL 88.5071 (matches the 88.51 of `reports/v1_5_replay_native.md` HONEST, `v1` column), row F1 87.0385, fully-correct docs 68.80% = 344 of 500, header accuracy 99.8293. VERIFIED |
| rebuilt from the native ZS trace (`$SHIPDOC_TMP_DIR\native_extract\zs\zeroshot500_qwen35_4b_img_only_native_4c17aa3\trace.jsonl`) with the production path `replay_v1_check.production` (`postrules.postprocess_traces` + coerce / repair), `RuleConfig(r1=True, r2=True, r3=True)`, HONEST (fold-held-out, supplier-disjoint) R3 shapes, default OCR cache | equal to the scratch predictions for all 500 docs (`True`). VERIFIED |
| native raw `predictions.json` (rules off), same scorer | OVERALL 83.8633 (the replay report's 83.86 baseline). VERIFIED |
| `reports/row_errors_native.md` (X4, ORACLE `rows: column_shift`) | 252 rows / 30 docs, +2.80 [+1.77, +3.96]: identical to section 1 below, so that report and this one describe the same run. VERIFIED (grep + recompute) |

Command that documents the post-rules path: `scripts/replay_v1_check.py --run-dir <native ZS run> --out reports/v1_5_replay_native.md ...` (the report's provenance line). The ORACLE below used the shipping-path variant with HONEST shapes, i.e. the (a) HONEST table of that report, not the in-sample (b) table (both give 88.51 overall).

## 1. ORACLE upper bound: every `column_shift` row scored as correct (500 docs, native ZS + rules)

Definition: `scripts/row_error_diagnosis.py` `classify_pair` (a linked gold/predicted row pair whose cross-field identifier matches are >= its same-field matches, after the `spn_copies_other_slot` test). `apply_oracle` replaces each such predicted row by its gold row; the unmodified scorer rescores. This is an UPPER BOUND that assumes a perfect fix; no hint arm exists at native.

| | OVERALL % | row F1 % | fully-correct docs | header acc % |
|---|---|---|---|---|
| native ZS + rules (baseline) | 88.51 | 87.04 | 344 / 500 (68.80%) | 99.83 |
| ORACLE: 252 `column_shift` rows fixed | 91.31 | 92.15 | 363 / 500 (72.60%) | 99.83 |
| delta, paired bootstrap 95% CI | **+2.80 pts [+1.77, +3.96]** | +5.11 pts [+3.19, +7.34] | +3.80 pts [+2.20, +5.60] (+19 docs) | +0.00 |

VERIFIED (command above; `$SHIPDOC_TMP_DIR\hdrhint\est_oracle.out`). All 252 are `pair` units (rows present on both sides, wrong slot).

Where the 252 rows are:

| facet | value |
|---|---|
| docs with at least one `column_shift` row | 30 (28 multipage, 2 single-page) |
| rows on multipage docs / on single-page docs | 235 / 17 |
| rows on page 1 (locator page, equal to the trace page) | 32 (15 on page 1 of a multipage doc + 17 single-page) |
| **rows on pages > 1 of multipage docs** | **220 of 252** (70 `middle`, 150 `last`); the same 220 by the gold-side locator page and by the predicted-side trace page |
| layout of those pages (`layout.analyze_page`, OCR) | all 220 on `continuation` pages (no column-header row); the 32 page-1 rows are on `table_header` pages |
| docs behind the 220 rows | 27; page 1 of all 27 has a table-header line, so the hint text would exist for every one of them |

VERIFIED. Only the 220 continuation-page rows are addressable by the 07 hint (it adds `Column headers from page 1: ...` on page >= 2 only; page 1 is never hinted). ORACLE restricted to those 220 rows (same method, 2000 resamples, seed 42):

| | OVERALL % | row F1 % | fully-correct docs |
|---|---|---|---|
| ORACLE: 220 continuation-page `column_shift` rows fixed | 90.97 | 91.50 | 361 / 500 (72.20%) |
| delta vs 88.51 [95% CI] | **+2.46 pts [+1.52, +3.56]** | +4.46 pts [+2.71, +6.49] | +3.40 pts [+2.00, +5.00] |

VERIFIED (`est_oracle2.out`). A real hint would fix some fraction of these, not all of them, and may break other rows: the numbers are a ceiling for the benefit, not a forecast. The ceiling is on the 500 train+dev docs (the native ZS run is zero-shot, so unseen by construction); the 07 A/B itself would run on the 152 multipage docs of those, where the same fixed rows are a larger share. For reference the spec 11.4 gate is ">= 30 column-shift rows remain": 252 are present (VERIFIED), so the gate is satisfied by this ZS+rules run; whether it still holds after FT is the open question the deferral waits on.

## 2. Cost of a native 07 variant (T4 hours, ESTIMATE)

What 07 runs (read from `notebooks/README.md` section 07 and `scripts/colab_build_hdrhint.py`, VERIFIED by reading): ONE arm, the hint arm, over the 152 pooled multipage docs (28 dev + 124 train, 323 pages) plus a smoke gate (first 3 multipage dev docs, 7 pages, batch 1). The control is NOT rerun: it is the existing 02 run restricted to those docs (plan cell REFUSES unless config hash, prompt hash, batch size, revision and seed equal). The 1260 batch size is 8.

Measured pace (sum of per-page `meta.latency_s` over the two ZS traces, `$SHIPDOC_TMP_DIR\hdrhint\est_time.py`; VERIFIED):

| run | all 671 pages | the 323 pooled 07 pages | pooled pages >= 2 (171) |
|---|---|---|---|
| 1260-token 02 (batch 8) | 8234.1 s = 2.287 h, 12.27 s/page | 4710.0 s = 1.308 h, 14.58 s/page | 13.89 s/page |
| native 02n (batch 4, some pages at 2) | 12022.4 s = 3.340 h, 17.92 s/page | 6983.4 s = 1.940 h, 21.62 s/page | 20.13 s/page |
| native / 1260 | 1.46 | 1.49 | 1.45 |

The 3.340 h and 671 pages equal the figures quoted in the task. The 1260 README figure (14.58 s/page over the same 323 pages, 1.31 h) is reproduced.

| native 07 hint arm (323 pages + smoke) | T4 hours |
|---|---|
| central, ESTIMATE: 323 pages at the native pace measured on exactly these pages (21.62 s/page) = 1.94 h; smoke 7 pages x 38.3 s (native batch-1 pace, `nativerun.NATIVE_B1_S_PER_PAGE`, measured by the 06 sweep) = 0.07 h; 2 model loads x 120 s (ASSUMED, as in 07) = 0.07 h | **about 2.1 h** |
| high, ESTIMATE: every page at the native batch-1 pace, 323 x 38.3 s = 3.44 h + 0.14 h | about 3.6 h |
| for comparison, the 1260 07 estimate table (ASSUMED) and measured pace | 1.02 - 1.70 h assumed; 1.31 h of generation measured (README, reproduced above) |

The hint adds at most 64 prompt tokens to pages >= 2 against a mean input of 2978 tokens at native (VERIFIED, trace mean), about 2%: ignored. Control pages are not rerun (no cost). Running only pages >= 2 (171 pages, 3442 s = 0.96 h at the pooled pace) would need custom merge code and is NOT what the notebook does. The OCR dependency at inference is unchanged (a deployment cost, not part of the GPU hours).

What a native 07 variant needs (NOT done; no decision):

1. A new notebook (or a re-pin of a copy), not an edit of `07_hdrhint_ab`: `07_hdrhint_ab` is pinned to `ebf96614eb8f077cbc6743d3f9e5d5e1ead691e7` and its `CONFIG` / `HINT_CONFIG` / `CONTROL_RUN_DIR` / `CONTROL_CODE_SHA` / `BATCH_SIZE` are the 1260 ones (VERIFIED in `scripts/colab_build_hdrhint.py`). A native builder (like `colab_build_*_native.py`) would set `CONFIG = qwen35_4b_img_only_native`, `HINT_CONFIG` = a new `configs/spike_qwen35_4b_img_only_native_hdrhint.yaml` (the native config plus `header_hint: true`, so `max_pixels` 2196480 and a new config hash), `CONTROL_RUN_DIR = zeroshot500_qwen35_4b_img_only_native_4c17aa3`, `CONTROL_CODE_SHA = 4c17aa3c33c09f0cda7bb1f625947a1143a8cb28`, `BATCH_SIZE = 4` (the control's batch: the plan cell requires equal batch sizes), plus the native resolution refusals the other `*_native` notebooks carry.
2. `headerhint.py` hard-codes the 1260 config paths as CLI defaults (`--control-config`, `--hint-config`, lines 810-811); the native notebook passes the native ones, so no logic change is needed there, only the new yaml and builder (the plan cell's hash checks compare the control manifest with the checked-out config, so the native control passes only with the native control config).
3. A new pin: the new config yaml and builder are new committed files, so the notebook pins a NEW code commit (not `4c17aa3`, which lacks them); sha7 of that commit names the run folders (`hdrhint_ab_<sha7>`, `hdrhint_hint_<sha7>`). Nothing about the existing native pin (`4c17aa3`) or the pinned native notebooks changes.
4. The 12 of 152 pooled docs with no page-1 table-header line (1260 OCR cache) get no hint; the OCR text is resolution-independent, so that count carries over (UNVERIFIED at native beyond the OCR cache being the same files).

Not measured: any native hint-arm accuracy (nothing has run), whether the hint helps FT output (the FT-native row-error table is pending), the native batch-4 VRAM with the longer prompt (UNVERIFIED; the 64 extra tokens are small against the 12.09 GiB batch-4 peak recorded in the 02n bench, `docs/run_sheet_native.md` section 2).
