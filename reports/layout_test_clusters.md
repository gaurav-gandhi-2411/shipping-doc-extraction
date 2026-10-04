# Test-set layout clusters and unseen-layout share (Phase 6.4)

**All numbers UNVERIFIED** (computed by `scripts/layout_test_clusters.py`, seed 42; a verifier must recompute). Aggregates only: no OCR text and no test labels were used or created; per-doc assignments are kept outside the repo.

## Provenance

- Command: `uv run python scripts/layout_test_clusters.py`
- Git HEAD: `6c2fa30` (cluster.py/layout_sig.py uncommitted at run time: True)
- OCR cache: `paddleocr`, SHA256SUMS sha256 prefix `4d05e5a8ab84ff75`, manifest pages_cached 951; test cache has 280 pages / 200 docs
- Signature: page 1 only, `shipdoc.cluster.layout_signature` (unchanged) or a variant from `shipdoc.layout_sig`

## 1. Signature validation on train+dev (ARI vs supplier_group, page 1)

Selection criterion fixed in advance: highest ARI vs `supplier_group` on train+dev invoices; a variant replaces `base` only if it beats it by more than 0.01. Three signatures were tried (base + 2 variants) and all are listed, so the winner's ARI is mildly optimistic (multiple-comparison). CI = 95% reverse-percentile bootstrap over docs (seed 42, 1000 resamples, clusters held fixed: does not include refit variance). k by silhouette over 8..30, label-free.

| signature | doc type | docs | k | silhouette | ARI [95% CI] | purity |
|---|---|---|---|---|---|---|
| base | invoice | 400 | 28 | 0.502 | 0.346 [0.317, 0.379] | 0.645 |
| base | waybill | 100 | 9 | 0.913 | -0.007 [-0.036, 0.031] | 0.250 |
| coarse | invoice | 400 | 30 | 0.502 | 0.363 [0.329, 0.397] | 0.667 |
| coarse | waybill | 100 | 9 | 0.980 | 0.009 [-0.025, 0.049] | 0.270 |
| geom | invoice | 400 | 30 | 0.499 | 0.387 [0.354, 0.421] | 0.672 |
| geom | waybill | 100 | 18 | 0.610 | -0.004 [-0.034, 0.031] | 0.330 |

Chosen: **geom** (invoice ARI 0.387 vs base 0.346). Earlier recorded diagnostic (reports/ablations.md R1 detail): base, ARI 0.346, purity 0.645 on the 400 train+dev invoice pages; this run's base row is the same fit if the numbers match. Waybill groups are carrier brand, which recon.md found does not track layout, so waybill ARI is reported, not interpreted as layout quality.

Variants: `coarse` = positions snapped to a 0.1 grid; `geom` = base + 6 fixed-constant page-geometry features (aspect, log line count, 5th pct left edge, 95th pct right edge, top, bottom of the text block).

## 2. Test doc type (label-free)

Rule: page-1 signature lacks the 'invoice' keyword family => waybill. A first try with the 'awb' family scored accuracy 0.440 on train+dev (invoices mention AWBs) and was dropped; a depth-1 tree over the 20 presence flags found 'invoice' (5-fold CV accuracy 1.0). On train+dev vs gold doc_type: TP 100, FP 0, FN 0, TN 400 (accuracy 1.000, n=500). The pipeline has no label-free doc-type convention other than the model's own prediction, so this rule is used.

Test split (200 docs): 162 invoice-like, 38 waybill-like by the rule.

## 3. Unseen-layout share of the test split

Reference = train+dev docs of the same doc type (400 invoice pages, 100 waybill pages), signature `geom`. Threshold = 95th percentile of out-of-sample seen distances (5-fold within train+dev), fixed before test was assigned.

| doc type | k | threshold | seen-dist median | LOGO-dist median | FPR (seen flagged) | TPR (held-out group flagged) |
|---|---|---|---|---|---|---|
| invoice | 30 | 1.437 | 0.497 | 1.139 | 0.050 | 0.255 |
| waybill | 18 | 0.148 | 0.021 | 0.021 | 0.050 | 0.010 |

| doc type | test docs | primary: centre distance > threshold, count (share) [Wilson 95%] | check 1: 1-NN distance > its own 95th pct | check 2: joint KMeans, docs in test-only clusters | primary corrected for FPR/TPR (Rogan-Gladen) |
|---|---|---|---|---|---|
| invoice | 162 | 39 (24.1%) [0.181, 0.312] | 56 (34.6%) | 0 (0.0%), 0 test-only of k=29 | 93.0% [64.1%, 100.0%] |
| waybill | 38 | 1 (2.6%) [0.005, 0.135] | 3 (7.9%) | 1 (2.6%), 1 test-only of k=19 | n/a |

1-NN check thresholds: invoice 1.106 (FPR 0.050, TPR 0.375), waybill 0.071 (FPR 0.050, TPR 0.050).

Distinct reference signatures (geom): invoice 400/400, waybill 100/100; under the keyword-only `base` signature: invoice 395/400, waybill 71/100. Waybill page-1 signatures are near-constant (under `base` almost none of the 20 keyword families vary), so waybill clusters, distances and any 'unseen' count are degenerate and not evidence about layout.

## 4. Comparison with the brief's ~50% and caveats

Brief: about 50% of the test docs have unseen layouts. What this proxy says (all UNVERIFIED, and only for the 162 invoice-like test docs, because waybill layout signatures are degenerate and waybill 'unseen' by layout is not meaningful per recon.md):

- Raw flagged share (distance to nearest train+dev centre above the 95th-percentile seen threshold): 39/162 = 24.1%, Wilson 95% [18.1%, 31.2%]. The 1-NN variant flags 56/162 = 34.6%. Joint KMeans finds 0 test-only clusters (0.0%), but that check is insensitive: k=29 centres absorb a new layout unless several test docs share it.
- These raw shares are NOT the unseen share. Under leave-one-supplier-group-out the same threshold flags only 25.5% of docs whose supplier group was held out (TPR), at 5.0% false positives on seen docs. The proxy misses about three quarters of truly unseen suppliers, partly because different suppliers share template geometry and keyword placement (invoice ARI is only 0.387, purity 0.672). Correcting the flagged share for that, (0.241 - 0.050) / (TPR - 0.050), gives 93% at the measured TPR 0.255 (range [64%, 100%] from the Wilson bounds alone) and is extremely sensitive to the TPR: the same arithmetic gives 54.5% at TPR 0.40, 42.4% at 0.50, 34.7% at 0.60, 20.1% at 1.00.
- Reading: the data are consistent with the brief's ~50% (a TPR of about 0.43 reproduces 50%), but also with anything from about 20% to nearly all, so this proxy neither confirms nor refutes 50%. The one robust statement is a lower-bound-ish one: roughly 19 percentage points of the invoice-like test docs (24.1% flagged minus the 5% false-positive rate) sit farther from every train+dev layout centre than 95% of seen documents do, i.e. the test split is not distributed like train+dev layouts.
- The LOGO TPR is itself a pessimistic yardstick for 'unseen layout' (a held-out supplier can reuse another supplier's template); real unseen layouts in test may be more separable than held-out suppliers, which would pull the corrected share toward the lower end of the sensitivity list.
- Doc-type split (label-free rule validated 500/500 on train+dev): 162 invoice-like, 38 waybill-like of 200 test docs; train+dev is 400/100, so 81%/19% vs 80%/20%.
- Selection and calibration used train+dev only; the percentile (95) and tie margin (0.01) were fixed before test was assigned. Disclosure: the script ran twice on test; the first run used the 'awb' doc-type rule, whose train+dev accuracy (0.440) and a 44/156 test split exposed it as wrong, so the rule was replaced by 'invoice' keyword absence (train+dev-validated) and the ARI interval method was changed; neither change touches the threshold or the unseen criterion, and the unseen shares of the first run were not used to pick anything. Three signatures were tried and one chosen by invoice ARI (0.387 vs 0.346 base; the intervals overlap, so the multiple-comparison optimism is not negligible). Per-doc assignments (ids, cluster, distance, seen flag) are in `$SHIPDOC_TMP_DIR\test_layout_assignments.csv`, outside the repo.

