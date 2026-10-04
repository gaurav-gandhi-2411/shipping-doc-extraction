"""Per-document trace JSONL (``<run>/trace.jsonl``, one JSON object per document).

Writer: `shipdoc.spike.run_spike` (appended after each document). Document line keys: ``doc_id``,
``split``, ``arm``, ``output_format``, ``pages``, ``merge``, ``normalize_flags``, ``prediction``,
``model``, ``prompt``, ``config``, ``git_commit``.

Page record (``pages[i]``): ``page`` (1-based), ``image``, ``raw_text``, ``parsed``,
``json_valid``, ``schema_errors``, ``meta`` (latency_s, peak_vram_bytes, n_input_tokens,
n_visual_tokens, n_output_tokens, ...).

Logprob extension (only when the run used ``--logprobs``; keyed output format only). Two extra
page keys, both ABSENT in older traces, so every reader must use ``page.get("field_logprobs")``:

``field_logprobs`` -- list (text order), or null if capture failed (``meta.logprob_error``)::

    {"scope": "header" | "row",   # which object the value sits in
     "row_idx": int | null,       # index in the page's emitted line_items; null for header
     "field": str,                # key, e.g. "invoice_number", "quantity"
     "min": float | null,         # min token logprob over the VALUE tokens (raw model logits)
     "mean": float | null,        # mean token logprob over the same tokens
     "n_tokens": int,             # tokens overlapping the value (>= 1 for every emitted value)
     "min_c": float | null,       # same, under the grammar-constrained distribution
     "mean_c": float | null,
     "truncated": true}           # only present when the page text ends inside this value

Value tokens: every token overlapping the span from the colon after the key to the end of the JSON
literal (quotes included; ``null`` for nulls, digits for numbers). The colon is included because
BPE merges it with the neighbouring quotes (``":"``, ``":``) and that token decides string versus
null versus number; a merged ``","`` is attributed to the value it closes. Keys and structure that
no value overlaps are not scored. Null (JSON ``null``) means
"missing / not finite", never 0. Duplicate keys: the last occurrence wins. Details and rationale:
`shipdoc.logprobs`.

``token_logprobs`` -- ``{"ends": [int], "lp": [float|null], "lp_c": [float|null]}``, one entry per
generated token (including the stop token, which has a zero-length span): ``ends`` is the
character end offset of the token in ``raw_text``, ``lp`` / ``lp_c`` the logprob of the chosen
(greedy) token under the raw / grammar-constrained distribution, rounded to 6 decimals. It is the
source `field_logprobs` is derived from, so features can be recomputed offline without the model.

Batch extension (only runs with ``--batch-size`` above 1; ABSENT at batch size 1, so a batch-1
trace is unchanged). Extra ``meta`` keys of a page, all absent in older traces (use ``.get``):

- ``batch_size`` -- pages in the generate call this page was in (after any fallback);
- ``batch_latency_s`` -- wall time of that call; ``latency_s`` is then ``batch_latency_s /
  batch_size`` (the page's share), so sums of ``latency_s`` stay the model time of the run;
- ``peak_vram_bytes`` -- peak of the whole call, the same for every page of it;
- ``batch_fallback`` -- list of ``{"from", "to", "error"}``, outermost first: the call of
  ``from`` pages raised (``error``), the pages were rerun in chunks of ``to``. Pages are never
  lost; the unbatched path is the end of the chain.

``n_input_tokens`` / ``n_visual_tokens`` / ``n_output_tokens`` are per page without padding.
A document is written once ALL its pages are done, in input order; ``manifest.json`` next to the
trace stores code SHA, config hash, model revision, seed, batch size (and its source), shard and
the bench result used (`shipdoc.runmeta`).
"""

from __future__ import annotations
