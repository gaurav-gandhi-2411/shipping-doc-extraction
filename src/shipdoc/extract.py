"""VLM prompting and constrained JSON decoding.

`VlmBackend` is the seam: the spike pipeline only sees ``extract_page``. `HfBackend` runs a
Hugging Face model with xgrammar-constrained greedy decoding; `MockBackend` replays gold for tests.
torch / transformers / xgrammar are imported lazily inside `HfBackend`, so importing this module
(and the mock path) needs none of the ``vlm`` dependency group.

Per-page output schema (`page_schema`): ``doc_type``, a ``header`` holding the UNION of the invoice
and waybill fields (all nullable strings; the other type's keys are null), ``line_items`` (the 4
nullable row fields) and ``page_kind``.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from shipdoc.logprobs import LogprobRecorder, build_trace, mock_logprob_trace
from shipdoc.prompts import (
    INVOICE_HEADER,
    PAGE_KINDS,
    ROW_FIELDS,
    WAYBILL_HEADER,
    ocr_block,
)

NULLABLE_STRING: dict[str, Any] = {"type": ["string", "null"]}
# Numeric fields also accept a bare JSON number. Under a string-or-null grammar the digit tokens are
# masked; the spike40 models (asked for "a plain number") put null, or a junk string made of
# closing braces, in nearly every quantity / total_amount / pieces / gross_weight_kg cell
# (reports/spike_diagnosis.md). Normalisation reads str(number).
NUMERIC_VALUE: dict[str, Any] = {"type": ["string", "number", "null"]}
NUMERIC_KEYS = frozenset({"total_amount", "pieces", "gross_weight_kg", "quantity"})
DOC_TYPES = ("invoice", "waybill")
INVOICE_KEYS = tuple(k for k, _ in INVOICE_HEADER)
WAYBILL_KEYS = tuple(k for k, _ in WAYBILL_HEADER)
TOTAL_KEYS = ("total_amount", "pieces", "gross_weight_kg")
# Declared (= decoding) order of the union header: identity fields first in reading order, totals
# last (they sit at the bottom of the page). The two key sets are disjoint (asserted in the tests).
HEADER_KEYS = tuple(k for k in INVOICE_KEYS + WAYBILL_KEYS if k not in TOTAL_KEYS) + TOTAL_KEYS
ROW_KEYS = tuple(k for k, _ in ROW_FIELDS)
SEED = 42


def _value_schema(key: str) -> dict[str, Any]:
    """Schema of one field's value: numeric fields allow a JSON number, the rest string-or-null."""
    return dict(NUMERIC_VALUE if key in NUMERIC_KEYS else NULLABLE_STRING)


def page_schema() -> dict[str, Any]:
    """JSON schema of one page's output (strict: no extra keys, every key required).

    Property order IS the decoding order: xgrammar emits the keys of an object in the order the
    schema declares them, so ``doc_type`` comes first (the model commits to a type before filling
    the header), rows read ``supplier_part_number, customer_part_number, purchase_order, quantity``
    like the table columns, and the totals come last.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["doc_type", "header", "line_items", "page_kind"],
        "properties": {
            "doc_type": {"enum": list(DOC_TYPES)},
            "header": {
                "type": "object",
                "additionalProperties": False,
                "required": list(HEADER_KEYS),
                "properties": {k: _value_schema(k) for k in HEADER_KEYS},
            },
            "line_items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": list(ROW_KEYS),
                    "properties": {k: _value_schema(k) for k in ROW_KEYS},
                },
            },
            "page_kind": {"enum": list(PAGE_KINDS)},
        },
    }


# --------------------------------------------------------------------------------------------
# Compact output format (config ``output_format: compact``)
# --------------------------------------------------------------------------------------------

OUTPUT_FORMATS = ("json", "compact")
#: Full header key -> short key. Order of `HEADER_KEYS` is kept so the expander is deterministic.
SHORT_HEADER_KEYS: dict[str, str] = {
    "invoice_number": "inv_no",
    "invoice_date": "inv_date",
    "supplier_name": "sup",
    "buyer_name": "buy",
    "ship_to_name": "ship_to",
    "currency": "cur",
    "total_amount": "total",
    "awb_number": "awb",
    "carrier": "car",
    "mawb": "mawb",
    "hawb": "hawb",
    "origin_airport": "orig",
    "destination_airport": "dest",
    "shipper_name": "shp",
    "consignee_name": "cns",
    "pieces": "pcs",
    "gross_weight_kg": "gw",
}
#: Top-level compact keys: doc_type, header, line_items, page_kind.
COMPACT_TOP = {"doc_type": "dt", "header": "h", "line_items": "r", "page_kind": "pk"}
#: Row array positions follow `ROW_KEYS`: [spn, cpn, po, qty].
_LONG_HEADER_KEYS = {v: k for k, v in SHORT_HEADER_KEYS.items()}


def compact_page_schema() -> dict[str, Any]:
    """JSON schema of one page in the compact format (strict, every key required).

    Rows are positional arrays ``[spn, cpn, po, qty]``. xgrammar 0.2.8 implements ``prefixItems``
    (cpp/json_schema_converter.cc ParseArray, tests/python/test_json_schema_converter.py), so the
    arrays are length-pinned by the grammar itself: ``prefixItems`` x4, ``minItems``/``maxItems`` 4
    and ``items: false`` (no additional items; strict mode already implies it).
    """
    row: dict[str, Any] = {
        "type": "array",
        "prefixItems": [_value_schema(k) for k in ROW_KEYS],
        "items": False,
        "minItems": len(ROW_KEYS),
        "maxItems": len(ROW_KEYS),
    }
    short = [SHORT_HEADER_KEYS[k] for k in HEADER_KEYS]
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(COMPACT_TOP.values()),
        "properties": {
            "dt": {"enum": list(DOC_TYPES)},
            "h": {
                "type": "object",
                "additionalProperties": False,
                "required": short,
                "properties": {SHORT_HEADER_KEYS[k]: _value_schema(k) for k in HEADER_KEYS},
            },
            "r": {"type": "array", "items": row},
            "pk": {"enum": list(PAGE_KINDS)},
        },
    }


def compact_page(page: dict[str, Any]) -> dict[str, Any]:
    """Schema-format page dict -> compact page dict (key order = the compact schema's)."""
    header = page["header"]
    return {
        "dt": page["doc_type"],
        "h": {SHORT_HEADER_KEYS[k]: header.get(k) for k in HEADER_KEYS},
        "r": [[row.get(k) for k in ROW_KEYS] for row in page["line_items"]],
        "pk": page["page_kind"],
    }


def expand_page(compact: dict[str, Any]) -> dict[str, Any]:
    """Deterministic inverse of `compact_page`; raises ValueError on a malformed compact page.

    Missing short header keys expand to null (the grammar always emits them; this only matters for
    hand-built input). Nulls stay null and strings are never touched.
    """
    try:
        header = compact["h"]
        rows = compact["r"]
        doc_type, page_kind = compact["dt"], compact["pk"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"compact page lacks {exc}") from exc
    if not isinstance(header, dict) or not isinstance(rows, list):
        raise ValueError("compact page: 'h' must be an object and 'r' an array")
    unknown = set(header) - set(_LONG_HEADER_KEYS)
    if unknown:
        raise ValueError(f"unknown compact header keys {sorted(unknown)}")
    items = []
    for row in rows:
        if not isinstance(row, list) or len(row) != len(ROW_KEYS):
            raise ValueError(f"compact row must be an array of {len(ROW_KEYS)} items")
        items.append(dict(zip(ROW_KEYS, row, strict=True)))
    return {
        "doc_type": doc_type,
        "header": {k: header.get(SHORT_HEADER_KEYS[k]) for k in HEADER_KEYS},
        "line_items": items,
        "page_kind": page_kind,
    }


def parse_output(
    raw_text: str, output_format: str = "json", *, number_text: bool = False
) -> dict[str, Any] | None:
    """`parse_page_json`, then (compact only) `expand_page`; any failure gives None."""
    obj = parse_page_json(raw_text, number_text=number_text)
    if obj is None or output_format == "json":
        return obj
    try:
        return expand_page(obj)
    except ValueError:
        return None


def schema_text(schema: dict[str, Any]) -> str:
    """The JSON string handed to the grammar compiler, keys in DECLARED order.

    Never ``sort_keys=True``: xgrammar 0.2.8 keeps the property order of the JSON string (picojson
    ``object_with_ordered_keys``, ``properties_obj.ordered_keys()`` in json_schema_converter.cc),
    so sorting the keys silently forces the model to emit alphabetical keys (quantity before the
    part number). The spike runs at 51cf560 did exactly that (reports/spike_diagnosis.md).
    """
    return json.dumps(schema)


def schema_for(output_format: str) -> dict[str, Any]:
    """The decoding schema of an output format."""
    if output_format == "compact":
        return compact_page_schema()
    if output_format == "json":
        return page_schema()
    raise ValueError(f"unknown output_format {output_format!r}; valid: {list(OUTPUT_FORMATS)}")


def nuextract_compact_template() -> dict[str, Any]:
    """NuExtract3 input template for the compact format (UNVERIFIED on the real model: the card
    documents arrays of objects; an array of one positional 4-string array is the closest
    equivalent). If NuExtract3 ignores it the xgrammar schema still pins the output."""
    return {
        "dt": list(DOC_TYPES),
        "h": {SHORT_HEADER_KEYS[k]: "verbatim-string" for k in HEADER_KEYS},
        "r": [["verbatim-string"] * len(ROW_KEYS)],
        "pk": list(PAGE_KINDS),
    }


def nuextract_template() -> dict[str, Any]:
    """NuExtract3 input template with the same structure as `page_schema` (model card format).

    Leaves name the output TYPE. Everything is ``verbatim-string`` (text exactly as printed; dates
    and numbers are normalised later by shipdoc.normalize, never by the model); enums are lists.
    """
    return {
        "doc_type": list(DOC_TYPES),
        "header": {k: "verbatim-string" for k in HEADER_KEYS},
        "line_items": [{k: "verbatim-string" for k in ROW_KEYS}],
        "page_kind": list(PAGE_KINDS),
    }


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not valid JSON")


def parse_page_json(raw_text: str, *, number_text: bool = False) -> dict[str, Any] | None:
    """Parse model output. Any failure (invalid JSON, truncation, non-object) gives None.

    The caller keeps `raw_text`; there is deliberately no repair and no retry with sampling.
    `number_text` returns every JSON number as its token text (a str: ``1234.50``, ``1E+6``) and
    treats NaN / Infinity as a parse failure; `shipdoc.coerce.page_for_merge` uses it so the
    merge sees what the model printed, not a float.
    """
    try:
        if number_text:
            obj = json.loads(
                raw_text, parse_float=str, parse_int=str, parse_constant=_reject_constant
            )
        else:
            obj = json.loads(raw_text)
    except (ValueError, TypeError):  # JSONDecodeError is a ValueError
        return None
    return obj if isinstance(obj, dict) else None


def schema_errors(parsed: dict[str, Any]) -> list[str]:
    """jsonschema errors of `parsed` against `page_schema` (empty list when valid)."""
    import jsonschema

    validator = jsonschema.Draft202012Validator(page_schema())
    return [e.message for e in validator.iter_errors(parsed)][:5]


@dataclass(frozen=True)
class BackendConfig:
    """Everything that defines a model run (all of it lands in the config hash)."""

    key: str  # e.g. "qwen35_4b"
    repo: str
    revision: str
    adapter: str  # "qwen35" | "qwen3vl" | "nuextract3"
    quant: str = "none"  # "none" (fp16) | "nf4"
    dtype: str = "float16"  # T4 has no bf16
    attn_implementation: str | None = None
    # Default Qwen cap is none -> 2145 visual tokens for a 1240x1754 page; 1280*32*32 -> 1260;
    # 1024*32*32 -> 988 (patch 16 x merge 2 = 32 px per visual token).
    max_pixels: int = 1280 * 32 * 32
    max_new_tokens: int = 1536
    output_format: str = "json"  # json | compact (see `compact_page_schema`)
    ocr_token_budget: int = 1200
    seed: int = SEED
    trust_remote_code: bool = False  # the NuExtract3 repo has no .py files; not needed


@dataclass(frozen=True)
class PageRequest:
    """One page of a batched call (`extract_pages`). `context` is the mock's (doc_id, page_index,
    n_pages); the HF backend ignores it."""

    image: Any
    prompt: str
    schema: dict[str, Any]
    ocr_text: str | None = None
    context: tuple[str, int, int] | None = None


class VlmBackend(Protocol):
    """One-page extractor. ``meta`` keys: latency_s, peak_vram_bytes (None off-GPU),
    n_input_tokens, n_visual_tokens, n_output_tokens, plus ocr_tokens/ocr_budget/ocr_truncated
    when OCR text was supplied.

    A backend may also offer ``extract_pages(requests) -> list[(raw, parsed, meta)]`` (same order
    as the requests): N pages in one generate call. Its metas carry ``batch_size`` (N),
    ``batch_latency_s`` (wall time of the call) and ``latency_s`` = batch_latency_s / N. The
    runner only uses it for batch sizes above 1; batch size 1 always goes through `extract_page`.
    """

    model_id: str
    revision: str

    def extract_page(
        self,
        image: Any,
        prompt: str,
        schema: dict[str, Any],
        ocr_text: str | None,
    ) -> tuple[str, dict[str, Any] | None, dict[str, Any]]:
        """Return ``(raw_text, parsed_json_or_None, meta)``."""
        ...


# --------------------------------------------------------------------------------------------
# Per-model adapters: chat-template differences
# --------------------------------------------------------------------------------------------


class Adapter(Protocol):
    """Builds (messages, apply_chat_template kwargs) for one page."""

    model_class_name: str

    def build(
        self, image: Any, prompt: str, schema: dict[str, Any], output_format: str = "json"
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]: ...


class _UserImageText:
    model_class_name = ""
    chat_kwargs: dict[str, Any] = {}  # class constant, never mutated

    def build(
        self, image: Any, prompt: str, schema: dict[str, Any], output_format: str = "json"
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        content = [{"type": "image", "image": image}, {"type": "text", "text": prompt}]
        return [{"role": "user", "content": content}], dict(self.chat_kwargs)


class Qwen35Adapter(_UserImageText):
    """Qwen3.5-4B (native Qwen3_5ForConditionalGeneration). Thinking is on by default in its
    chat template, so ``enable_thinking=False`` is mandatory (the card's non-thinking switch)."""

    model_class_name = "Qwen3_5ForConditionalGeneration"
    chat_kwargs = {"enable_thinking": False}


class Qwen3VLAdapter(_UserImageText):
    """Qwen3-VL Instruct (4B / 8B): plain image + text user turn, no template switches."""

    model_class_name = "Qwen3VLForConditionalGeneration"


class NuExtract3Adapter:
    """NuExtract3 (Qwen3.5-4B fine-tune): the user turn holds only the image; the JSON template
    goes in the ``template`` chat-template kwarg (a JSON string, indent=4 as in the card) and the
    rules + OCR hint in the optional ``instructions`` kwarg."""

    model_class_name = "Qwen3_5ForConditionalGeneration"

    def build(
        self, image: Any, prompt: str, schema: dict[str, Any], output_format: str = "json"
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        messages = [{"role": "user", "content": [{"type": "image", "image": image}]}]
        kwargs = {
            "template": json.dumps(
                nuextract_compact_template()
                if output_format == "compact"
                else nuextract_template(),
                indent=4,
            ),
            "instructions": prompt,
            "enable_thinking": False,
        }
        return messages, kwargs


ADAPTERS: dict[str, Callable[[], Adapter]] = {
    "qwen35": Qwen35Adapter,
    "qwen3vl": Qwen3VLAdapter,
    "nuextract3": NuExtract3Adapter,
}


def get_adapter(name: str) -> Adapter:
    """Adapter instance by config name; raises KeyError listing the valid names."""
    if name not in ADAPTERS:
        raise KeyError(f"unknown adapter {name!r}; valid: {sorted(ADAPTERS)}")
    return ADAPTERS[name]()


# --------------------------------------------------------------------------------------------
# Hugging Face backend
# --------------------------------------------------------------------------------------------


class HfBackend:
    """Transformers + xgrammar greedy extractor. Heavy imports happen in `load`, not here."""

    def __init__(self, cfg: BackendConfig) -> None:
        self.cfg = cfg
        self.model_id = cfg.repo
        self.revision = cfg.revision
        self.adapter = get_adapter(cfg.adapter)
        self._loaded = False
        self._grammar_cache: dict[str, Any] = {}
        # Set by the runner (`spike --logprobs`). Off by default so spike runs are untouched.
        self.capture_logprobs = False

    # -- loading ------------------------------------------------------------------------------

    def load(self) -> None:
        """Load processor + model (pinned revision) onto the GPU; idempotent."""
        if self._loaded:
            return
        import torch
        import transformers
        import xgrammar as xgr

        if not torch.cuda.is_available():
            raise RuntimeError(
                "HfBackend needs a CUDA GPU (Colab T4). For a CPU-only dry run use "
                "`--backend mock`."
            )
        cfg = self.cfg
        seed_everything(cfg.seed)
        proc = transformers.AutoProcessor.from_pretrained(
            cfg.repo, revision=cfg.revision, trust_remote_code=cfg.trust_remote_code
        )
        # Cap the visual tokens. In transformers 5.18 the Qwen2VL-family image processor reads
        # size={"shortest_edge": min_pixels, "longest_edge": max_pixels}; the per-call max_pixels
        # kwarg is deprecated. Keep the checkpoint's own shortest_edge.
        size = dict(proc.image_processor.size)
        size["longest_edge"] = cfg.max_pixels
        proc.image_processor.size = size

        kwargs: dict[str, Any] = {
            "revision": cfg.revision,
            "dtype": getattr(torch, cfg.dtype),
            "device_map": {"": 0},
            "trust_remote_code": cfg.trust_remote_code,
        }
        if cfg.attn_implementation:
            kwargs["attn_implementation"] = cfg.attn_implementation
        if cfg.quant == "nf4":
            kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        elif cfg.quant != "none":
            raise ValueError(f"unknown quant {cfg.quant!r} (use 'none' or 'nf4')")
        model_cls = getattr(transformers, self.adapter.model_class_name)
        model = model_cls.from_pretrained(cfg.repo, **kwargs).eval()

        tok = proc.tokenizer
        # The repo may ship no generation_config (Qwen3.5), so stop ids are explicit: the chat
        # end-of-turn token plus the tokenizer's own eos.
        stops = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
        self.stop_ids = sorted(i for i in stops if isinstance(i, int) and i >= 0)
        self.tokenizer_info = xgr.TokenizerInfo.from_huggingface(
            tok,
            vocab_size=model.config.get_text_config().vocab_size,
            stop_token_ids=self.stop_ids,
        )
        self.compiler = xgr.GrammarCompiler(self.tokenizer_info)
        self.torch, self.xgr, self.processor, self.model = torch, xgr, proc, model
        self.image_token_id = getattr(model.config, "image_token_id", None)
        self._loaded = True

    # -- per-page -----------------------------------------------------------------------------

    def _count_tokens(self, text: str) -> int:
        return len(self.processor.tokenizer(text, add_special_tokens=False)["input_ids"])

    def _compiled(self, schema: dict[str, Any]) -> Any:
        text = schema_text(schema)
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if key not in self._grammar_cache:
            # any_whitespace=False: single-line JSON, no whitespace-padding loops to burn tokens.
            self._grammar_cache[key] = self.compiler.compile_json_schema(text, any_whitespace=False)
        return self._grammar_cache[key]

    def extract_page(
        self,
        image: Any,
        prompt: str,
        schema: dict[str, Any],
        ocr_text: str | None,
    ) -> tuple[str, dict[str, Any] | None, dict[str, Any]]:
        """One constrained greedy generation. Never retries; a failed parse returns parsed=None."""
        self.load()
        torch, cfg = self.torch, self.cfg
        meta: dict[str, Any] = {}
        full = prompt
        if ocr_text is not None:
            block, info = ocr_block(ocr_text, cfg.ocr_token_budget, self._count_tokens)
            meta.update(info)
            if block:
                full = f"{prompt}\n\n{block}"
        messages, chat_kwargs = self.adapter.build(
            image.convert("RGB"), full, schema, cfg.output_format
        )
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **chat_kwargs,
        ).to(self.model.device)
        n_in = int(inputs["input_ids"].shape[1])
        n_vis = (
            int((inputs["input_ids"] == self.image_token_id).sum())
            if self.image_token_id is not None
            else None
        )
        # A new processor per call: xgrammar's LogitsProcessor is single-use.
        processor = self.xgr.contrib.hf.LogitsProcessor(self._compiled(schema))
        procs: list[Any] = [processor]
        rec = LogprobRecorder() if self.capture_logprobs else None
        if rec is not None:  # pure observers around the grammar mask; output is unchanged
            procs = [rec.pre, processor, rec.post]
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        seed_everything(cfg.seed)
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=False,
                num_beams=1,
                temperature=None,
                top_p=None,
                top_k=None,
                repetition_penalty=1.0,
                use_cache=True,
                eos_token_id=self.stop_ids,
                pad_token_id=self.stop_ids[0],
                logits_processor=procs,
            )
        torch.cuda.synchronize()
        latency = time.perf_counter() - t0
        gen = out[0, n_in:]
        tokenizer = self.processor.tokenizer
        raw = tokenizer.decode(gen, skip_special_tokens=True).strip()
        meta.update(
            latency_s=latency,
            peak_vram_bytes=int(torch.cuda.max_memory_allocated()),
            n_input_tokens=n_in,
            n_visual_tokens=n_vis,
            n_output_tokens=int(gen.shape[0]),
        )
        if rec is not None:
            try:  # after generation: a failure here can only lose the logprobs, never the output
                ids, lp, lp_c = rec.finish()
                meta["logprob_trace"] = build_trace(
                    gen.tolist(),
                    ids,
                    lp,
                    lp_c,
                    lambda seqs: tokenizer.batch_decode(seqs, skip_special_tokens=True),
                    raw,
                )
            except Exception as exc:  # surfaced in meta; the smoke gate fails closed
                meta["logprob_error"] = f"{type(exc).__name__}: {exc}"
        return raw, parse_output(raw, cfg.output_format), meta

    def release_memory(self) -> None:
        """After a failed batch (the runner calls it): free the dead call tensors (OOM)."""
        import gc

        gc.collect()
        if self._loaded:
            self.torch.cuda.empty_cache()

    def extract_pages(
        self, requests: Sequence[PageRequest]
    ) -> list[tuple[str, dict[str, Any] | None, dict[str, Any]]]:
        """N pages in ONE constrained greedy generate call (left padding, one matcher per row).

        Same decoding as `extract_page` per row; only the batch dimension differs. NOT bit-exact
        with batch 1 in general (fp16 matmul/attention kernels reduce in a different order for
        another batch shape; padded rows add masked work), which is why the bench selects the batch
        size by byte-identity against batch 1 on a real GPU. Nothing is retried here: any failure
        (OOM included) propagates and the runner falls back to a smaller batch.
        """
        self.load()
        cfg = self.cfg
        if not requests:
            return []
        schema = requests[0].schema
        if any(schema_text(r.schema) != schema_text(schema) for r in requests):
            raise ValueError("all pages of one batch must share one schema")
        metas: list[dict[str, Any]] = [{} for _ in requests]
        conversations: list[list[dict[str, Any]]] = []
        chat_kwargs: dict[str, Any] = {}
        for r, meta in zip(requests, metas, strict=True):
            full = r.prompt
            if r.ocr_text is not None:
                block, info = ocr_block(r.ocr_text, cfg.ocr_token_budget, self._count_tokens)
                meta.update(info)
                if block:
                    full = f"{r.prompt}\n\n{block}"
            messages, chat_kwargs = self.adapter.build(
                r.image.convert("RGB"), full, r.schema, cfg.output_format
            )
            conversations.append(messages)
        inputs = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            # processor kwargs, not **kwargs: transformers 5.18 treats unknown **kwargs as processor
            # kwargs with a warning and then DROPS the explicit dict. Generation appends on the
            # right, so pad on the left.
            processor_kwargs={"padding": True, "padding_side": "left"},
            **chat_kwargs,
        ).to(self.model.device)
        mask = inputs["attention_mask"]
        if not bool(mask[:, -1].all()):
            raise ValueError("batched inputs are not left-padded (a row ends in padding)")
        n_in = mask.sum(dim=1).tolist()
        n_vis = (
            (inputs["input_ids"] == self.image_token_id).sum(dim=1).tolist()
            if self.image_token_id is not None
            else [None] * len(requests)
        )
        dec = decode_batch(
            self.model,
            self.torch,
            self.xgr,
            self._compiled(schema),
            self.processor.tokenizer,
            inputs,
            self.stop_ids,
            cfg.max_new_tokens,
            self.capture_logprobs,
            cfg.seed,
        )
        out = []
        for row, meta, n, v in zip(dec.rows, metas, n_in, n_vis, strict=True):
            meta.update(
                latency_s=dec.latency_s / len(requests),
                batch_latency_s=dec.latency_s,
                batch_size=len(requests),
                peak_vram_bytes=dec.peak_vram_bytes,
                n_input_tokens=int(n),
                n_visual_tokens=v,
                n_output_tokens=len(row.gen_ids),
            )
            if row.logprob_trace is not None:
                meta["logprob_trace"] = row.logprob_trace
            if row.logprob_error is not None:
                meta["logprob_error"] = row.logprob_error
            out.append((row.raw, parse_output(row.raw, cfg.output_format), meta))
        return out


@dataclass
class RowDecode:
    """One row of a batched decode: text, its generated ids (up to and incl. the stop token) and
    the logprob trace (or the reason there is none)."""

    raw: str
    gen_ids: list[int]
    logprob_trace: dict[str, Any] | None = None
    logprob_error: str | None = None


@dataclass
class BatchDecode:
    """Result of `decode_batch`: rows in input order plus the call's wall time and peak VRAM."""

    rows: list[RowDecode]
    latency_s: float
    peak_vram_bytes: int | None


def trim_at_stop(ids: Sequence[int], stop_ids: Sequence[int]) -> list[int]:
    """`ids` up to and including the first stop id (HF pads the rest of a finished row)."""
    stops = set(stop_ids)
    for k, t in enumerate(ids):
        if t in stops:
            return list(ids[: k + 1])
    return list(ids)


def decode_batch(
    model: Any,
    torch: Any,
    xgr: Any,
    compiled: Any,
    tokenizer: Any,
    inputs: Any,
    stop_ids: Sequence[int],
    max_new_tokens: int,
    capture_logprobs: bool,
    seed: int = SEED,
) -> BatchDecode:
    """Constrained greedy decode of a LEFT-padded batch (rows of `inputs`) in one generate call.

    - One xgrammar matcher and one bitmask row per sequence: ``xgrammar.contrib.hf.LogitsProcessor``
      (0.2.8) replicates the single compiled grammar over ``input_ids.shape[0]`` rows, accepts each
      row's last token and refills each row's bitmask separately; a row whose matcher terminated
      (it accepted its stop token) is skipped, so the pad HF forces into finished rows never
      reaches a matcher. A fresh processor per call (it is single-use).
    - Per-row stopping is HF's: a row that picked a stop id is padded with ``pad_token_id`` while
      the others go on; every row has the same ``max_new_tokens`` cap. Each row's ids are cut at
      its first stop id, so ``n_output_tokens`` and the decoded text match the batch-1 path.
    - Logprobs: `LogprobRecorder(stop_ids)` records per row and never records padding.
    """
    n, n_in = inputs["input_ids"].shape
    is_cuda = model.device.type == "cuda"
    processor = xgr.contrib.hf.LogitsProcessor(compiled)
    rec = LogprobRecorder(stop_ids) if capture_logprobs else None
    procs: list[Any] = [rec.pre, processor, rec.post] if rec is not None else [processor]
    if is_cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    seed_everything(seed)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=1.0,
            use_cache=True,
            eos_token_id=list(stop_ids),
            pad_token_id=stop_ids[0],
            logits_processor=procs,
        )
    if is_cuda:
        torch.cuda.synchronize()
    latency = time.perf_counter() - t0
    peak = int(torch.cuda.max_memory_allocated()) if is_cuda else None
    gens = [trim_at_stop(out[i, n_in:].tolist(), stop_ids) for i in range(n)]
    rows = [
        RowDecode(raw=tokenizer.decode(g, skip_special_tokens=True).strip(), gen_ids=g)
        for g in gens
    ]
    if rec is not None:
        try:  # after generation: a failure here can only lose the logprobs, never the output
            recorded = rec.finish_rows()
            if len(recorded) != n:
                raise ValueError(f"recorder saw {len(recorded)} rows, expected {n}")
        except Exception as exc:  # surfaced in meta; the smoke gate fails closed
            for row in rows:
                row.logprob_error = f"{type(exc).__name__}: {exc}"
            recorded = []
        for row, (ids, lp, lp_c) in zip(rows, recorded, strict=False):
            try:
                row.logprob_trace = build_trace(
                    row.gen_ids,
                    ids,
                    lp,
                    lp_c,
                    lambda seqs: tokenizer.batch_decode(seqs, skip_special_tokens=True),
                    row.raw,
                )
            except Exception as exc:  # one bad row loses only its own logprobs
                row.logprob_error = f"{type(exc).__name__}: {exc}"
    return BatchDecode(rows=rows, latency_s=latency, peak_vram_bytes=peak)


def seed_everything(seed: int = SEED) -> None:
    """Seed random / numpy / torch (when importable) and request deterministic kernels."""
    import contextlib
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)  # legacy global seed on purpose: also covers third-party code
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    with contextlib.suppress(AttributeError, RuntimeError):  # torch build without support
        torch.use_deterministic_algorithms(True, warn_only=True)


# --------------------------------------------------------------------------------------------
# Mock backend
# --------------------------------------------------------------------------------------------


@dataclass
class MockCorruption:
    """Deterministic ways the mock can go wrong (all off by default)."""

    banner_leak: bool = False  # continuation pages carry the invoice number (page-2 trap)
    blank_fields: tuple[str, ...] = ()  # header fields forced to null on page 1
    drop_last_rows: int = 0  # rows removed from the end of the last page
    invalid_json_every: int = 0  # every k-th call returns truncated text (0 = never)
    wrong_doc_type_pages: tuple[int, ...] = ()  # page indices that report the other doc type


class MockBackend:
    """Replays gold as per-page JSON. Call `set_context` before `extract_page` (the spike does).

    Layout mimics the real corpus (recon): identity header only on page 1, totals only on the
    last page of a multi-page document, rows split into contiguous chunks across pages.
    """

    model_id = "mock"
    revision = "mock"

    TOTALS = {"invoice": ("total_amount",), "waybill": ("pieces", "gross_weight_kg")}

    def __init__(
        self,
        gold: dict[str, dict[str, Any]],
        corruption: MockCorruption | None = None,
        ocr_token_budget: int = 1200,
        output_format: str = "json",
        fixed_latency_s: float | None = None,
    ) -> None:
        self.output_format = output_format
        # Tests that compare whole traces byte for byte pin the (otherwise wall-clock) latency.
        self.fixed_latency_s = fixed_latency_s
        self.ocr_token_budget = ocr_token_budget
        self.gold = gold
        self.corruption = corruption or MockCorruption()
        self._ctx: tuple[str, int, int] | None = None
        self._calls = 0
        self.capture_logprobs = False  # set by the runner; adds `logprob_trace` to meta

    def set_context(self, doc_id: str, page_index: int, n_pages: int) -> None:
        """Tell the mock which page of which document the next call is about."""
        self._ctx = (doc_id, page_index, n_pages)

    def page_payload(self, doc_id: str, page_index: int, n_pages: int) -> dict[str, Any]:
        """The (uncorrupted-by-JSON-damage) page dict for a page of a gold document."""
        g, c = self.gold[doc_id], self.corruption
        dt = g["doc_type"]
        last = page_index == n_pages - 1
        totals = self.TOTALS[dt]
        header: dict[str, str | None] = dict.fromkeys(HEADER_KEYS)
        for k, v in g["header"].items():
            keep = last if k in totals else page_index == 0
            if keep:
                header[k] = v
        if page_index == 0:
            for k in c.blank_fields:
                header[k] = None
        elif c.banner_leak and dt == "invoice":
            header["invoice_number"] = g["header"].get("invoice_number")
        rows = g["line_items"]
        base, extra = divmod(len(rows), n_pages)
        start = page_index * base + min(page_index, extra)
        stop = start + base + (1 if page_index < extra else 0)
        chunk = [dict(r) for r in rows[start:stop]]
        if last and c.drop_last_rows:
            chunk = chunk[: max(0, len(chunk) - c.drop_last_rows)]
        kind = "single" if n_pages == 1 else ("first" if page_index == 0 else "continuation")
        doc_type = dt
        if page_index in c.wrong_doc_type_pages:
            doc_type = "waybill" if dt == "invoice" else "invoice"
        return {"doc_type": doc_type, "header": header, "line_items": chunk, "page_kind": kind}

    def extract_page(
        self,
        image: Any,
        prompt: str,
        schema: dict[str, Any],
        ocr_text: str | None,
    ) -> tuple[str, dict[str, Any] | None, dict[str, Any]]:
        """Deterministic page JSON from gold; meta mirrors the HF backend (4 chars/token)."""
        if self._ctx is None:
            raise RuntimeError("MockBackend.set_context(doc_id, page_index, n_pages) not called")
        self._calls += 1
        t0 = time.perf_counter()
        payload = self.page_payload(*self._ctx)
        body = compact_page(payload) if self.output_format == "compact" else payload
        raw = json.dumps(body, separators=(", ", ": "), ensure_ascii=False)
        k = self.corruption.invalid_json_every
        if k and self._calls % k == 0:
            raw = raw[: len(raw) // 2]
        meta: dict[str, Any] = {
            "latency_s": self._latency(t0),
            "peak_vram_bytes": None,
            "n_input_tokens": -(-len(prompt) // 4),
            "n_visual_tokens": 0,
            "n_output_tokens": -(-len(raw) // 4),
        }
        if ocr_text is not None:
            _, info = ocr_block(ocr_text, self.ocr_token_budget)
            meta.update(info)
        if self.capture_logprobs:
            meta["logprob_trace"] = mock_logprob_trace(raw)
        return raw, parse_output(raw, self.output_format), meta

    def _latency(self, t0: float) -> float:
        return time.perf_counter() - t0 if self.fixed_latency_s is None else self.fixed_latency_s

    def extract_pages(
        self, requests: Sequence[PageRequest]
    ) -> list[tuple[str, dict[str, Any] | None, dict[str, Any]]]:
        """N pages per call. Each page is answered exactly as `extract_page` would, independent
        of the other pages of the call (the property a real batched backend must also have)."""
        t0 = time.perf_counter()
        out = []
        for r in requests:
            if r.context is None:
                raise RuntimeError("MockBackend.extract_pages needs PageRequest.context")
            self.set_context(*r.context)
            out.append(self.extract_page(r.image, r.prompt, r.schema, r.ocr_text))
        wall = self._latency(t0)
        for _raw, _parsed, meta in out:
            meta.update(batch_size=len(out), batch_latency_s=wall, latency_s=wall / len(out))
        return out
