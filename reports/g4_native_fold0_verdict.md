# Native fold 0: C1/C2/C3 verdict and pooled projection (ONE FOLD, NOT THE G4 DECISION)

Source: `reports/g4_fold0_native.md` (FT+rules vs ZS+rules, 171 docs, 2000 doc-level resamples, seed 42,
unmodified scorer, native resolution 2196480 for both arms), `false_fill_diag_native_fold0.md`,
`rule_gate_native_fold0.md`. Spec section 11.1 clauses; the pre-registered rule is unchanged. Numbers
UNVERIFIED unless a second path is named. Aggregates and counts only.

## Clauses on fold 0 with rules

| clause (spec s11.1) | ZS+rules | FT+rules | result on fold 0 |
|---|---|---|---|
| C1: paired OVERALL delta CI lower bound > 0 | 91.62 | 93.53 | **holds**: +1.92 [+0.36, +3.38] |
| C2: false_fill_total FT <= ZS (raw counts) | 0 | 5 | **fails on this fold** |
| C3: over_null_total FT <= ZS (raw counts) | 11 | 1 | **holds** |

VERIFIED (second path): `over_null_counts` re-run on the emitted post-rule predictions reproduces 5/0
and 1/11; a hand-rolled paired bootstrap reproduces +1.916 [0.36, 3.38].

Raw arms: OVERALL +2.81 [+1.13, +4.46]; raw row false fills 0 -> 13 (6 scanned, 7 digital).
`false_fill_diag`: all 13 are purchase_order cells; 8 (all inv_g03) are the value of the other slot and
are cleared by R3 with 0 broken; 5 remain (inv_g03 1, inv_g04 4; 3 docs): the value is on the page but
the gold leaves the slot null; R1/R2 touch none. Mechanism of that second category is UNVERIFIED.

The `g4_fold` verdict lines (NO REGRESSION, FINE-TUNED KEPT) are weaker than C2: the first uses
over-nulls only, the second the header-only false-fill rate. Under the s11 clarification C2 on this fold
FAILS (5 vs 0). One fold is not the decision.

## What the pooled decision needs (arithmetic, from measured ZS+rules counts)

Pooled ZS+rules native, honest per-fold R3 shapes, computed over all 500 docs (not a projection):

| ZS+rules | fold 0 | fold 1 | fold 2 | pooled |
|---|---|---|---|---|
| false fills | 0 | 76 | 0 | 76 |
| over-nulls | 11 | 21 | 1 | 33 |
| OVERALL | 91.62 | 90.14 | 83.98 | 88.51 |

The 76 fold-1 false fills agree with the cpn_false_fill + mixed tally in `row_errors_native.md` X2
(75 + 1): VERIFIED by a second path. So FT folds 1+2 may add at most 71 false fills (76 - 5) and at
most 32 over-nulls (33 - 1) before FT+rules loses C2 / C3 on raw pooled counts.

## PROJECTION, not a result

Assumption: FT on folds 1 and 2 behaves like FT on fold 0. This is weak: fold 1 has a very different
layout mix (ZS has 288 raw false fills there), so group-level ranges matter.

| FT+rules, folds 1+2 (329 docs) | fold-0 rate scaled | doc bootstrap 95% | supplier-group bootstrap 95% (10 groups) | room before losing |
|---|---|---|---|---|
| false fills | 9.6 | 2 to 18 | 0 to 22.5 | 71 |
| over-nulls | 1.9 | 0 to 5 | 0 to 6.1 | 32 |

On those rates C2 and C3 pass with a large margin. C2 fails only if FT behaves nearly like ZS in unseen
layouts (more than 71 false fills in folds 1+2).

C1: fold-0 per-doc SE of the paired delta is 0.771 pts (n=171); scaled to 500 docs 0.445 pts (CI
half-width about 0.87). Pooled point = (171 x 1.916 + 329 x d) / 500 with d the folds 1+2 mean delta.
For the lower bound to clear 0: d >= +0.33 with fold-0 noise; >= +0.99 if 1.5x noisier; >= +1.66 if 2x
noisier. ZS+rules is weaker on folds 1 and 2 than on fold 0 (90.14 and 83.98 vs 91.62), so a delta at
or above fold 0's is plausible (UNVERIFIED).

## Recommendation for GG (not a decision; the pre-registered rule is unchanged)

Start `final` (03n, L4, about 4.4 h, ESTIMATE UNVERIFIED) as soon as fold 2 finishes; do not wait for
05n fold 1 / fold 2. Cost of starting early: if FT loses the pooled decision, about 4.4 L4 hours are
wasted and nothing else. Cost of waiting: the L4 idles roughly 1.4 to 3 h (ESTIMATE; depends on whether
the two 05n folds run back to back on the single T4). Wait instead only if compute units are tight,
since waiting costs wall-clock only.
