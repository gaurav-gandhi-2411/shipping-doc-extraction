# Output format: json vs compact (gold-rendered targets, train pages)

**Every number in this report is UNVERIFIED** (produced by `scripts/token_budget.py --format compare`, not yet independently recomputed).

Command: `uv run --with tokenizers --with huggingface_hub python scripts/token_budget.py --format compare --out reports/output_format.md`

536 train pages; same row-to-page assignment for both formats (locations.jsonl sha256 `6e1961394178486a`); +1 stop token; `max_new_tokens = roundup64(ceil(p99 * 1.25))` as in `reports/token_budget.md`.

## Formats

- `json` (current): `{"doc_type": .., "header": {17 long keys, other type null}, "line_items": [{4 long keys}, ..], "page_kind": ..}`.
- `compact`: `{"dt": .., "h": {inv_no, inv_date, sup, buy, ship_to, cur, total, awb, car, mawb, hawb, orig, dest, shp, cns, pcs, gw}, "r": [[spn, cpn, po, qty], ..], "pk": ..}`; nulls are JSON null; `shipdoc.extract.expand_page` restores the schema JSON deterministically.

## Schema-constrained decoding: `prefixItems`

xgrammar 0.2.8 supports `prefixItems` (read from GitHub tag v0.2.8): `cpp/json_schema_converter.cc` `SchemaParser::ParseArray` parses `prefixItems` into positional item rules and the grammar builder emits them in order (comment "prefixItems entries are positional", issue #824 fix); `tests/python/test_json_schema_converter.py` has `test_array_schema_min_max` cases for `prefixItems` with `minItems`/`maxItems`/`items: false` and the #824 regression tests. Used schema for a row: `{type: array, prefixItems: [4 x ["string","null"]], items: false, minItems: 4, maxItems: 4}`. The 1-char-key object fallback (`s`,`c`,`p`,`q`) is therefore NOT used. Compiled and checked with the real xgrammar 0.2.8 (throwaway env, CPU): `uv run --with xgrammar==0.2.8 pytest -q tests/test_compact.py -k grammar` accepts a valid page and rejects 3-item rows, 5-item rows and an integer in a row (`test_compact_schema_grammar_pins_row_arity`; skipped when xgrammar is absent).

## Token reduction (tokens per page)

| model (tokenizer) | json mean | compact mean | mean reduction | json p99 | compact p99 | p99 reduction | json max | compact max | json max_new_tokens | compact max_new_tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen35_4b | 531.3 | 386.2 | 27.3% | 1190 | 883 | 25.8% | 1244 | 920 | 1536 | 1152 |
| nuextract3 | 531.3 | 386.2 | 27.3% | 1190 | 883 | 25.8% | 1244 | 920 | 1536 | 1152 |
| qwen3vl_8b | 532.7 | 387.7 | 27.2% | 1191 | 884 | 25.8% | 1246 | 922 | 1536 | 1152 |

Compact pages over their own cap: qwen35_4b 0, nuextract3 0, qwen3vl_8b 0.

## Seconds-per-page projection (UNVERIFIED)

Decode time scales with output tokens:

    s/page ~= prefill_s + n_out / decode_tok_s

`prefill_s` (image + prompt tokens) and `decode_tok_s` are PARAMETERS to be measured from the Step L traces (`latency_s`, `n_input_tokens`, `n_output_tokens` per page in trace.jsonl); no speed is assumed here. Predicted saving per page = (n_out_json - n_out_compact) / decode_tok_s (prefill unchanged apart from the shorter prompt text). Mean n_out from the table above: qwen35_4b: 531 -> 386, nuextract3: 531 -> 386, qwen3vl_8b: 533 -> 388.
