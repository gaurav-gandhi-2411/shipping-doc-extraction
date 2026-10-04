# Sanity baselines

Produced by `uv run python scripts/baselines.py`. Metrics in percent; brackets are doc-level bootstrap 95% CIs (n=2000, seed 42). `CLI OVERALL` is the printed value of the unmodified `assignment/score.py` (run with `PYTHONUTF8=1`); `parity` = wrapper report equals the CLI `--out` JSON exactly.

## dev (100 docs)

| baseline | OVERALL | header acc | row F1 | fully correct | false-fill | CLI OVERALL | parity |
|---|---|---|---|---|---|---:|---|
| gold | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 0.00 [0.00, 0.00] | 100.00 | exact |
| empty | 1.65 [1.12, 2.19] | 4.13 [2.81, 5.47] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 1.65 | exact |
| all_null | 1.65 [1.12, 2.19] | 4.13 [2.81, 5.47] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 1.65 | exact |
| doctype_only | 1.65 [1.12, 2.19] | 4.13 [2.81, 5.47] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 1.65 | exact |

Gold null rate = 34/823 header fields = 4.13%; all-null header accuracy = 4.13%.

- gold: doc_type acc 100.00, illegible fields 34, rows 924
- empty: doc_type acc 0.00, illegible fields 34, rows 924
- all_null: doc_type acc 100.00, illegible fields 34, rows 924
- doctype_only: doc_type acc 100.00, illegible fields 34, rows 924

## train (400 docs)

| baseline | OVERALL | header acc | row F1 | fully correct | false-fill | CLI OVERALL | parity |
|---|---|---|---|---|---|---:|---|
| gold | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 0.00 [0.00, 0.00] | 100.00 | exact |
| empty | 1.93 [1.68, 2.19] | 4.82 [4.21, 5.47] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 1.93 | exact |
| all_null | 1.93 [1.68, 2.19] | 4.82 [4.21, 5.47] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 1.93 | exact |
| doctype_only | 1.93 [1.68, 2.19] | 4.82 [4.21, 5.47] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 1.93 | exact |

Gold null rate = 158/3277 header fields = 4.82%; all-null header accuracy = 4.82%.

- gold: doc_type acc 100.00, illegible fields 158, rows 4006
- empty: doc_type acc 0.00, illegible fields 158, rows 4006
- all_null: doc_type acc 100.00, illegible fields 158, rows 4006
- doctype_only: doc_type acc 100.00, illegible fields 158, rows 4006

## Why each baseline scores what it does

- **gold**: identical to the labels, so every component is 100 (harness sanity check).
- **empty `{}`**: every document is missing, so doc_type is wrong and rows are all missing (row F1 0); header accuracy is nonzero only because null-gold fields are "correct" when empty, and no document is fully correct unless the type matches.
- **all-null**: right doc_type, every header field null, no rows: it is right exactly on the gold-null fields, so header accuracy equals the gold null rate, a useful "always abstain" floor; row F1 is 0 (no rows predicted) and nothing is a false fill.
- **doc_type-only**: header `{}` reads as all-null to the scorer, so it scores identically to all-null; the doc_type is not itself a scored component beyond the fully-correct condition.

## Encoding

- dev: 0/100 label files contain non-ASCII bytes
- train: 0/400 label files contain non-ASCII bytes
