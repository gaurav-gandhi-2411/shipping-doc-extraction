# Native fold 1: C1/C2/C3 verdict, running tally with fold 0 (ONE FOLD, NOT THE G4 DECISION)

Source: `reports/g4_fold1_native.md` (FT+rules vs ZS+rules, 165 docs, 2000 doc-level resamples, seed 42, unmodified scorer, native resolution 2196480 both arms), `false_fill_diag_native_fold1.md`, `rule_gate_native_fold1.md`, `row_errors_native_fold1.md`. Spec section 11.1 clauses; the pre-registered rule is unchanged and is decided only on all three folds. Aggregates and counts only. Each number is marked VERIFIED (second path named) or UNVERIFIED.

## Clauses on fold 1 with rules

| clause (spec s11.1) | ZS+rules | FT+rules | result on fold 1 |
|---|---|---|---|
| C1: paired OVERALL delta CI lower bound > 0 | 90.14 | 97.03 | **holds**: +6.89 [+4.35, +9.72] |
| C2: false_fill_total FT <= ZS (raw counts) | 76 | 0 | **holds** (0 <= 76) |
| C3: over_null_total FT <= ZS (raw counts) | 21 | 1 | **holds** (1 <= 21) |

VERIFIED (second path): `shipdoc.oof.over_null_counts` re-run on the emitted post-rule predictions of both arms (and on the independent 500-doc ZS post-rules file for the ZS arm) gives false fills 76 / 0 and over-nulls 21 / 1 (raw: 288 / 7 and 246 / 8, of which row over-nulls 233 / 8); a hand-rolled bootstrap loop over `score_doc` results reproduces OVERALL +6.89 [+4.35, +9.72] (seed 42; seed 7 gives [+4.26, +9.64]) and the raw +14.31 [+10.65, +18.07]. Script `$SHIPDOC_TMP_DIR\native_fold1\verify_fold1.py`, output `verify_fold1.out`.

Banner check against `oof_compare.json` / `metrics.json` (VERIFIED, read): raw OVERALL ZS 82.43 -> FT 96.74, +14.31 [10.65, 18.07]; invoices +13.38 [9.53, 17.17]; waybills +10.82 (FT 100.00); raw row false fills 288 -> 7, raw row over-nulls 233 -> 8 (all purchase_order); batch 4 (source zero_shot_run, no override), guard ran, ok, byte-identical rate 1.0 on 12 pages, no fallback; sessions.json one session 4,769.4 s = 1.325 h (banner 1.32); adapter sha256 in the OOF manifest equals the local fold-1 adapter manifest (4cbb2ffe...605cad); `oof_verification.json` ok True, no warnings; 165 of 165 docs complete. No inconsistency with the banner.

Header vs row: all false fills and nearly all over-nulls are row cells (header false fills 0 in all four columns; header over-nulls 13 ZS raw, 0 after rules and 0 FT). Scanned vs digital (OVERALL, FT+rules vs ZS+rules): scanned 56 docs +6.80 [+2.29, +11.63]; digital 109 docs +6.93 [+4.02, +10.52]; invoices 134 docs +7.48 [+4.70, +10.22]; waybills 31 docs 100.00 vs 100.00 (+0.00). The 7 FT raw false fills are all scanned (one document, inv_g05).

Interim lines printed by `g4_fold.py`: `NO REGRESSION` and `FINE-TUNED KEPT` (exit 0). The raw arms also give NO REGRESSION. One fold is not the decision.

## R3 on fold 1 (the fold where it fires, spec s11.3)

| arm | R3 touched docs | cells changed | fixed | broken | neutral | d OVERALL on-off (pts) | rule_gate verdict |
|---|---|---|---|---|---|---|---|
| ZS | 18 | 221 | 212 | 0 | 9 | +5.99 (rule_gate table; R3 alone) | ZS side is context only in this report |
| FT | 1 | 7 | 7 | 0 | 0 | +0.29 [+0.00, +0.90] | NOT SHIPPABLE ON THIS EVIDENCE (insufficient power): broken 0, no group net < 0, CI lower bound +0.00 so the "CI excludes 0" clause fails |

Reading: R3 does not hurt the fine-tuned model on this fold (0 broken, all 7 fixed cells in one group, inv_g05, net +7; 0 groups with net < 0), but its benefit on FT is 7 cells, too few to clear the paired CI. R3's effect is concentrated in the ZS arm (212 fixed). R1 / R2 / R2-ocrfree on the FT arm: 0 fixed, 0 broken (waybills need no repair after FT; ZS fixed 2 / 11 / 0). Per spec s11.3 R3 is judged only where it fires: both folds that exercise it are 1 and 2; fold 2 is pending.

## Row-error causes (spec s11.4)

FT+rules column_shift 0 (ZS+rules 67, 66 in inv_g05); threshold for the 07 A/B is 30 FT rows. Detail in `row_errors_native_fold1.md`.

## Running partial tally, folds 0+1 (NOT the decision)

Counts from `over_null_counts` on the emitted post-rule predictions (honest per-fold R3 shapes), VERIFIED by the per-fold values reproduced from the separate 500-doc ZS post-rules file (fold 0 0/11, fold 1 76/21, fold 2 0/1) and by the fold-0 verdict's published values.

| | fold 0 | fold 1 | folds 0+1 | fold 2 |
|---|---|---|---|---|
| ZS+rules false fills | 0 | 76 | 76 | 0 (measured, ZS arm only) |
| FT+rules false fills | 5 | 0 | 5 | pending |
| ZS+rules over-nulls | 11 | 21 | 32 | 1 (measured, ZS arm only) |
| FT+rules over-nulls | 1 | 1 | 2 | pending |

PROJECTION (arithmetic only, not a result): the pooled rule compares raw FT+rules <= ZS+rules summed over 500 docs. ZS+rules pooled = 76 false fills, 33 over-nulls. FT is at 5 and 2 after two folds, so FT+rules may add at most 71 false fills (76 - 5) and 31 over-nulls (33 - 2) on fold 2 before C2 / C3 fail. Fold 0 and fold 1 FT+rules produced 5 and 0 false fills, 1 and 1 over-nulls; for reference the fold-2 ZS+rules arm has 0 false fills and 1 over-null, but that is the ZS arm and says nothing about how FT does there. Fold 2's held-out loss was much higher than folds 0/1 (`ft_native_fold2_verification.md`; cause not established), which is a reason not to extrapolate from folds 0 and 1.

Paired pooled-so-far OVERALL, folds 0+1 combined (336 docs), FT+rules vs ZS+rules, paired bootstrap 2000 resamples, seed 42 (`ev.paired_bootstrap`): ZS 90.82, FT 95.29, delta **+4.47 [+2.93, +6.03]**. VERIFIED (second path): a hand-rolled bootstrap loop gives the identical numbers. PARTIAL, folds 0+1 only; it is not the C1 decision, which needs the 500-doc paired bootstrap with fold 2. Arithmetic only: the pooled 500-doc point delta is (336 x 4.47 + 164 x d2) / 500 for a fold-2 delta d2, which is 0 only for d2 = -9.16 (CI lower bound not projected; it depends on the fold-2 per-doc variance).

## Open risks
- Fold 2 OOF (05n with the fold-2 adapter) is the only missing input; the pooled rule must be applied to the 500-doc counts, not to this tally.
- FT C2 depends on R3 on fold 0 (5 raw false fills remain after rules, inv_g03/inv_g04, mechanism UNVERIFIED) and is 0 on fold 1; the one fold-1 FT residual over-null (inv_g18, purchase_order) is real but small.
- R3 cannot be shown to help FT on fold 1 by its own paired CI (lower bound +0.00); the shipping case for R3 on FT then rests on fold 2 or on the ZS-arm evidence.
- All report files carry `+dirty` repo state because other executors have uncommitted work in the tree; no generated number depends on those edits (scripts used: `g4_fold.py`, `false_fill_diag.py`, `rule_gate.py`, `row_error_diagnosis.py` unchanged), but the stamps are not clean.
