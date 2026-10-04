# Supervised tokens of the fine-tune training pages (STEP Z2)

**Every number here is UNVERIFIED** (computed at commit `53746d1`; nothing was trained for this report). Tokenizer: **real Qwen3.5-4B tokenizer + chat template (revision 851bf6e8, local cache)**. Counts only: no page content, no values, no names. Reproduce with `uv run python scripts/supervised_share.py` (needs `data/`, `prepared_final.json` and the cached Qwen3.5 tokenizer files; tests: `tests/test_train_masking.py`).

Sample = `train.encode_page` output for the clean epoch-0 page (no augmentation; an occlusion changes one header value to `null`, a handful of tokens): inference-identical chat-template prefix (system/user turn, image placeholder expanded to 1,260 `<|image_pad|>`, generation prompt + empty think block) + keyed JSON target + `<|im_end|>`. **Supervised** = `labels != -100` = target JSON tokens + the stop token; the prefix, the image tokens and the masked `line_items` of header-only documents are `-100`. Kinds: *structure* = object keys, braces, brackets, colons, commas, separators and the quotes around values; *null* = a `null` literal; *value* = a token containing any character of a string/number value; *stop* = `<|im_end|>`. Shares in the last four columns are shares of the supervised tokens.

| page type | pages | supervised share of sequence, mean | min | max | supervised tokens per page, mean (min-max) | supervised tokens per epoch | structure | null | value | stop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **all train pages** | 536 | 18.9% | 5.2% | 37.3% | 522 (141-1244) | 279,992 | 55.9% | 3.6% | 40.3% | 0.19% |
| invoice / continuation | 136 | 18.2% | 5.5% | 31.8% | 485 (141-975) | 65,996 | 58.9% | 4.9% | 35.9% | 0.21% |
| invoice / first | 124 | 28.7% | 5.2% | 37.3% | 862 (166-1244) | 106,910 | 53.2% | 2.9% | 43.7% | 0.12% |
| invoice / single | 199 | 17.6% | 9.2% | 37.1% | 469 (211-1237) | 93,316 | 55.1% | 3.3% | 41.4% | 0.21% |
| waybill / single | 77 | 7.9% | 7.4% | 8.2% | 179 (167-186) | 13,770 | 67.3% | 4.6% | 27.6% | 0.56% |
| header-only docs (line_items masked) | 10 | 5.9% | 5.2% | 6.4% | 157 (141-183) | 1,568 | 72.8% | 8.5% | 18.0% | 0.64% |
| the 40 pages the smoke visited (epoch 0) | 40 | 18.9% | 7.6% | 34.1% | 520 (171-1083) | 20,800 | 55.8% | 3.6% | 40.4% | 0.19% |

## What this says about the smoke loss (0.0557 -> 0.0056, L4 bf16)

- The loss is the mean cross-entropy over the supervised tokens ONLY (`chunked_target_loss`: numerator and denominator both count labelled positions). Masked prefix tokens neither add to nor dilute it, so the supervised share of the sequence (18.9% of the tokens) does NOT explain a low loss; what matters is the composition of the supervised tokens.
- Composition of the supervised tokens of the 40 pages the smoke visited: 59.6% structure / `null` / stop and 40.4% value-bearing (all 536 pages: 59.7% and 40.3%; per-sample mean value share 36.1%).
- Arithmetic only (not a model measurement): if every structure / null / stop token cost exactly 0 nats, the observed means 0.0557 and 0.0056 mean 0.138 and 0.014 nats per value-bearing token (geometric-mean token probability 0.871 and 0.986). If they cost more than 0, the value tokens average LESS than that. Zero loss on all non-value tokens lowers the mean by at most their share of the tokens; the remaining mean has to come from the value-bearing tokens.
- All 20 smoke steps are in epoch 0 (`metrics.jsonl`: `epoch` 0 throughout) and every page is visited once, so each step's loss is computed on pages the model has not yet been updated on: the curve is a progressive validation over the 40 pages, not memorisation of a repeated set. Supplier layouts can still recur among those pages (doc-id-sorted prefix), so it is not a held-out-supplier loss.

## Not covered

- Whether the model's per-token losses are actually concentrated on value tokens: this needs a per-token loss dump from a GPU run (not available); only the composition is reported.
- Augmented epochs: occlusion adds `null` targets; scan degradation does not change the target.
