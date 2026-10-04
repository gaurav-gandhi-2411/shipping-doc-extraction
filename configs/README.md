# configs/

`paths.yaml` holds storage profiles (see `src/shipdoc/paths.py`). `spike_<model>_<arm>.yaml` are the
zero-shot spike configs (spec Phase 2.3): one file per candidate model x prompt arm.

## Spike run order (cheapest first)

Order is by estimated cost per run (T4 time, then quota risk). **The cost ranking is an estimate from
model size and prompt length; no s/page has been measured yet** (the first Colab run produces the
real numbers in `metrics.json`: `s_per_page_mean`, `s_per_page_p95`, `peak_vram_max_bytes`).

| # | config | model (revision pinned in the file) | precision | arm | why here |
|---|---|---|---|---|---|
| 1 | `spike_qwen35_4b_img_only.yaml` | Qwen/Qwen3.5-4B | fp16 | image only | 4B, shortest prompt |
| 2 | `spike_nuextract3_img_only.yaml` | numind/NuExtract3 | fp16 | image only | same size/arch as #1 |
| 3 | `spike_qwen35_4b_img_ocr.yaml` | Qwen/Qwen3.5-4B | fp16 | image + OCR hint | adds up to 1,200 prompt tokens |
| 4 | `spike_nuextract3_img_ocr.yaml` | numind/NuExtract3 | fp16 | image + OCR hint | same |
| 5 | `spike_qwen3vl_8b_img_only.yaml` | Qwen/Qwen3-VL-8B-Instruct | nf4 | image only | 8B 4-bit, slowest |
| 6 | `spike_qwen3vl_8b_img_ocr.yaml` | Qwen/Qwen3-VL-8B-Instruct | nf4 | image + OCR hint | slowest |
| alt | `spike_qwen3vl_4b_img_only.yaml` | Qwen/Qwen3-VL-4B-Instruct | fp16 | image only | only if #1-#4 do not run on the T4 |

The `img_ocr` arm needs the OCR cache (`python -m shipdoc ocr`), unzipped next to the data on Colab.

```bash
python -m shipdoc spike --config configs/spike_qwen35_4b_img_only.yaml \
    --docs splits/spike40.json --split dev --run-id spike40_qwen35_4b_img_only
# resume after a disconnect (skips doc_ids already in trace.jsonl):
python -m shipdoc spike ... --resume
# CPU-only dry run with gold replayed through the full merge/normalize/score path:
python -m shipdoc spike ... --backend mock --limit 5
```

Outputs: `<SHIPDOC_RUNS_DIR>/<run_id>/{predictions.json,trace.jsonl,metrics.json,progress.json}`.
`--wandb` logs metrics and config only (no images, no extracted values) to the private
`shipdoc-extract-debug` project, and only when `WANDB_API_KEY` is set.

## Config keys

| key | meaning |
|---|---|
| `model.repo`, `model.revision` | Hugging Face repo and the pinned 40-hex commit (loading fails config validation otherwise) |
| `model.adapter` | `qwen35`, `qwen3vl` or `nuextract3`: chat-template differences |
| `model.dtype`, `model.quant` | fp16 everywhere (T4 has no bf16); `nf4` = bitsandbytes 4-bit |
| `max_pixels` | visual-token cap, 1280*32*32 = 1,310,720 px -> 1,260 tokens on a 1240x1754 page (no cap: 2,145; 1024*32*32: 988) |
| `max_new_tokens` | greedy generation limit (1,536); a page that hits it is counted as invalid JSON, never retried |
| `ocr_token_budget` | `img_ocr` only: OCR hint truncated line-wise to this many model tokens (1,200); `ocr_truncated` is recorded per page |
| `prompt_version` | must equal `shipdoc.prompts.PROMPT_VERSION`; the prompt hash is recorded in every trace line |
| `seed` | 42 (random / numpy / torch; deterministic kernels where supported) |
