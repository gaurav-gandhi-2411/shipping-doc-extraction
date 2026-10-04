"""Batched constrained decoding on the CPU: real xgrammar + Qwen vocabulary + tiny random models.

What this proves (and what it cannot): the batching MACHINERY is right (left padding, one grammar
matcher and bitmask row per sequence, per-row stopping, per-row logprob capture, padding never
recorded), and the transformers 5.18 Qwen3.5 code path (linear-attention layers + mrope) does not
let left padding change a row's output in fp32 on the PyTorch reference kernels. It cannot show
anything about fp16, CUDA kernels, the real 4B weights or the real processor: that is the GPU
bench's job (shipdoc.bench).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from shipdoc.extract import decode_batch, trim_at_stop

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
xgr = pytest.importorskip("xgrammar")

from test_logprobs import _qwen_tokenizer_json  # noqa: E402  (shared HF-cache lookup)

VOCAB = 248_320  # Qwen3.5 text vocab_size (padded), as in HfBackend.load
#: Random weights pick "]" or "," at random: rows stop after different numbers of tokens.
ARRAY = {"type": "array", "items": {"enum": [1, 2, 3]}, "maxItems": 12}
LONG = {"type": "array", "items": {"enum": [1, 2, 3]}, "minItems": 40}


@pytest.fixture(scope="module")
def tok() -> Any:
    path = _qwen_tokenizer_json()
    if path is None:
        pytest.skip("Qwen3.5-4B tokenizer.json not in the local HF cache")
    return transformers.PreTrainedTokenizerFast(tokenizer_file=path)


@pytest.fixture(scope="module")
def stop(tok: Any) -> int:
    return int(tok.convert_tokens_to_ids("<|im_end|>"))


@pytest.fixture(scope="module")
def model() -> Any:
    torch.manual_seed(42)
    cfg = transformers.LlamaConfig(
        vocab_size=VOCAB, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=4096,
    )  # fmt: skip
    m = transformers.LlamaForCausalLM(cfg).eval()
    for p in m.parameters():
        p.data.normal_(0, 0.5)
    return m


@pytest.fixture(scope="module")
def compiler(tok: Any, stop: int) -> Any:
    info = xgr.TokenizerInfo.from_huggingface(tok, vocab_size=VOCAB, stop_token_ids=[stop])
    return xgr.GrammarCompiler(info)


def _grammar(compiler: Any, schema: dict[str, Any]) -> Any:
    return compiler.compile_json_schema(json.dumps(schema), any_whitespace=False)


def _batch(rows: list[list[int]]) -> dict[str, Any]:
    """Left-pad prompts with 0 and mask the padding, like the processor with padding_side=left."""
    n = max(len(r) for r in rows)
    return {
        "input_ids": torch.tensor([[0] * (n - len(r)) + r for r in rows]),
        "attention_mask": torch.tensor([[0] * (n - len(r)) + [1] * len(r) for r in rows]),
    }


PROMPTS = [[1, 5, 9, 13], [7, 8], [3, 4, 5, 6, 7, 8], [2, 2, 2]]


def _decode(model: Any, tok: Any, grammar: Any, stop: int, rows: list[list[int]], cap: int = 60):
    return decode_batch(model, torch, xgr, grammar, tok, _batch(rows), [stop], cap, True)


def test_trim_at_stop_keeps_the_stop_token_and_drops_the_padding() -> None:
    assert trim_at_stop([5, 6, 9, 9, 9], [9]) == [5, 6, 9]
    assert trim_at_stop([5, 6, 7], [9]) == [5, 6, 7]  # never stopped: all tokens
    assert trim_at_stop([5, 8, 9], [8, 9]) == [5, 8]  # any stop id
    assert trim_at_stop([], [9]) == []


def test_batched_rows_equal_the_same_rows_alone(model: Any, tok: Any, stop: int, compiler: Any):
    g = _grammar(compiler, ARRAY)
    singles = [_decode(model, tok, g, stop, [p]).rows[0] for p in PROMPTS]
    together = _decode(model, tok, g, stop, PROMPTS)
    lengths = [len(r.gen_ids) for r in together.rows]
    assert len(set(lengths)) > 1, "rows must stop at different steps for this test to mean anything"
    for alone, row in zip(singles, together.rows, strict=True):
        assert row.gen_ids == alone.gen_ids and row.raw == alone.raw
        assert row.gen_ids[-1] == stop  # the stop token is kept, like the batch-1 path
        assert isinstance(json.loads(row.raw), list)  # grammar-valid JSON for every row
        assert row.logprob_error is None and row.logprob_trace is not None
        lp, lp_alone = row.logprob_trace["lp"], alone.logprob_trace["lp"]
        # one entry per generated token: nothing recorded for the padding of a finished row
        assert len(lp) == len(row.gen_ids) == len(row.logprob_trace["lp_c"])
        assert lp == pytest.approx(lp_alone, abs=1e-4)
        assert row.logprob_trace["ends"] == alone.logprob_trace["ends"]


def test_a_rows_output_does_not_depend_on_its_companions_or_position(
    model: Any, tok: Any, stop: int, compiler: Any
) -> None:
    g = _grammar(compiler, ARRAY)
    base = _decode(model, tok, g, stop, PROMPTS).rows
    rev = _decode(model, tok, g, stop, PROMPTS[::-1]).rows[::-1]
    pair = _decode(model, tok, g, stop, [PROMPTS[2], PROMPTS[0]]).rows
    assert [r.gen_ids for r in rev] == [r.gen_ids for r in base]
    assert pair[0].gen_ids == base[2].gen_ids and pair[1].gen_ids == base[0].gen_ids


def test_every_row_gets_the_same_max_new_tokens_cap(
    model: Any, tok: Any, stop: int, compiler: Any
) -> None:
    g = _grammar(compiler, LONG)  # the grammar forbids stopping before 40 items
    out = _decode(model, tok, g, stop, PROMPTS, cap=7)
    assert [len(r.gen_ids) for r in out.rows] == [7] * len(PROMPTS)
    assert all(r.gen_ids[-1] != stop for r in out.rows)
    assert all(len(r.logprob_trace["lp"]) == 7 for r in out.rows)


def test_logprob_capture_does_not_change_the_batched_output(
    model: Any, tok: Any, stop: int, compiler: Any
) -> None:
    g = _grammar(compiler, ARRAY)
    on = decode_batch(model, torch, xgr, g, tok, _batch(PROMPTS), [stop], 60, True)
    off = decode_batch(model, torch, xgr, g, tok, _batch(PROMPTS), [stop], 60, False)
    assert [r.gen_ids for r in on.rows] == [r.gen_ids for r in off.rows]
    assert all(r.logprob_trace is None for r in off.rows)
    assert on.peak_vram_bytes is None  # CPU: no CUDA peak to report


def test_recorder_failure_loses_only_the_logprobs(
    model: Any, tok: Any, stop: int, compiler: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from shipdoc import logprobs

    def boom(self: Any) -> Any:
        raise RuntimeError("recorder broke")

    monkeypatch.setattr(logprobs.LogprobRecorder, "finish_rows", boom)
    out = _decode(model, tok, _grammar(compiler, ARRAY), stop, PROMPTS)
    assert all(r.raw and r.logprob_trace is None for r in out.rows)
    assert all("recorder broke" in (r.logprob_error or "") for r in out.rows)


def test_recorder_marks_rows_inactive_after_their_stop_token() -> None:
    rec_mod = pytest.importorskip("shipdoc.logprobs")
    rec = rec_mod.LogprobRecorder(stop_ids=[7])
    ids = torch.zeros(2, 1, dtype=torch.long)
    # step 0: row 0 picks the stop token 7, row 1 token 3; step 1: row 0 is finished (padding)
    for picks in ([7, 3], [5, 3], [5, 7]):
        scores = torch.full((2, 8), -5.0)
        scores[0, picks[0]] = scores[1, picks[1]] = 0.0
        rec.pre(ids, scores)
        rec.post(ids, scores)
    rows = rec.finish_rows()
    assert rows[0][0] == [7]  # row 0: only up to and including its stop token
    assert rows[1][0] == [3, 3, 7]
    assert [len(r[1]) for r in rows] == [1, 3] and [len(r[2]) for r in rows] == [1, 3]


# ------------------------------------------------------------------ Qwen3.5 hybrid, left padding


def _tiny_qwen35() -> Any:
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration, Qwen3_5TextConfig
    from transformers.models.qwen3_5 import configuration_qwen3_5 as c

    torch.manual_seed(0)
    text = Qwen3_5TextConfig(
        vocab_size=2000, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, linear_key_head_dim=8,
        linear_value_head_dim=8, linear_num_key_heads=2, linear_num_value_heads=4,
        max_position_embeddings=512,
    )  # fmt: skip
    vis = c.Qwen3_5VisionConfig(
        depth=1,
        hidden_size=32,
        num_heads=2,
        intermediate_size=64,
        out_hidden_size=32,
        num_position_embeddings=64,
    )
    cfg = Qwen3_5Config(
        text_config=text.to_dict(), vision_config=vis.to_dict(),
        image_token_id=1990, vision_start_token_id=1991, vision_end_token_id=1992,
    )  # fmt: skip
    assert text.layer_types.count("linear_attention") == 3  # 3 Gated-DeltaNet + 1 full attention
    model = Qwen3_5ForConditionalGeneration(cfg).eval()
    for p in model.parameters():
        p.data.normal_(0, 0.2)
    return model


def _vl_row(grid: tuple[int, int], text_ids: list[int]) -> tuple[list[int], Any, Any]:
    h, w = grid
    ids = [1991] + [1990] * (h * w // 4) + [1992] + text_ids  # 2x2 patches merge into 1 token
    return ids, torch.tensor([[1, h, w]]), torch.randn(h * w, 3 * 2 * 16 * 16)


def _vl_inputs(rows: list[Any], mask_padding: bool = True) -> dict[str, Any]:
    n = max(len(r[0]) for r in rows)
    ids = torch.tensor([[0] * (n - len(r[0])) + r[0] for r in rows])
    mask = torch.tensor([[0] * (n - len(r[0])) + [1] * len(r[0]) for r in rows])
    return {
        "input_ids": ids,
        "attention_mask": mask if mask_padding else torch.ones_like(mask),
        "pixel_values": torch.cat([r[2] for r in rows]),
        "image_grid_thw": torch.cat([r[1] for r in rows]),
    }


def _vl_generate(model: Any, inputs: dict[str, Any], steps: int = 8) -> Any:
    with torch.inference_mode():
        return model.generate(
            **inputs, max_new_tokens=steps, do_sample=False, num_beams=1, temperature=None,
            top_p=None, top_k=None, pad_token_id=0, eos_token_id=[1999],
            output_logits=True, return_dict_in_generate=True,
        )  # fmt: skip


def test_qwen35_left_padding_leaks_nothing_into_linear_attention_or_mrope() -> None:
    """Rows with different image sizes and text lengths (so different padding) decode exactly as
    they do alone: the Gated-DeltaNet state and the mrope positions ignore the padded slots.

    fp32, PyTorch reference kernels (flash-linear-attention / causal-conv1d are not installed, and
    neither are they in the Colab `vlm` group). A negative control with the padding mask switched
    off must differ, which shows the comparison is sensitive to a leak.
    """
    model = _tiny_qwen35()
    rows = [_vl_row((4, 4), [11, 12, 13, 14, 15]), _vl_row((2, 4), [21, 22]), _vl_row((4, 6), [31])]
    steps = 8
    batched = _vl_generate(model, _vl_inputs(rows), steps)
    width = batched.sequences.shape[1] - steps
    for i, r in enumerate(rows):
        alone = _vl_generate(model, _vl_inputs([r]), steps)
        assert alone.sequences[0, -steps:].tolist() == batched.sequences[i, width:].tolist()
        diff = max(
            (a[0] - b[i]).abs().max().item()
            for a, b in zip(alone.logits, batched.logits, strict=True)
        )
        assert diff < 1e-3, f"row {i}: logits differ by {diff}"
    leaky = _vl_generate(model, _vl_inputs(rows, mask_padding=False), steps)
    solo = _vl_generate(model, _vl_inputs([rows[1]]), steps)
    gap = max(
        (a[0] - b[1]).abs().max().item() for a, b in zip(solo.logits, leaky.logits, strict=True)
    )
    assert gap > 1e-2, "control: ignoring the padding mask should change the logits"


# ------------------------------------------------------------------ HfBackend.extract_pages glue


class _Enc(dict):  # stands in for transformers.BatchEncoding
    def to(self, _device: Any) -> _Enc:
        return self


class _StubProcessor:
    """Tokenizes the text of each conversation, prefixes one image token per 64 px of width and
    pads on the side the caller asks for (so the left-padding guard can be exercised)."""

    IMG = 248_000  # an id the ASCII prompts never produce

    def __init__(self, tokenizer: Any, side: str | None = None) -> None:
        self.tokenizer, self.side = tokenizer, side

    def apply_chat_template(self, conversations: Any, **kw: Any) -> _Enc:
        pk = kw["processor_kwargs"]
        assert pk["padding"] is True and kw["add_generation_prompt"] is True
        side = self.side or pk["padding_side"]
        rows = []
        for conv in conversations:
            parts = conv[0]["content"]
            image = next(p["image"] for p in parts if p["type"] == "image")
            text = next(p["text"] for p in parts if p["type"] == "text")
            ids = [self.IMG] * (image.width // 64)
            ids += self.tokenizer(text, add_special_tokens=False)["input_ids"]
            rows.append(ids)
        n = max(map(len, rows))
        ids_t = [
            ([0] * (n - len(r)) + r) if side == "left" else (r + [0] * (n - len(r))) for r in rows
        ]
        mask = [
            ([0] * (n - len(r)) + [1] * len(r))
            if side == "left"
            else ([1] * len(r) + [0] * (n - len(r)))
            for r in rows
        ]
        return _Enc(input_ids=torch.tensor(ids_t), attention_mask=torch.tensor(mask))


def _backend(model: Any, tok: Any, stop: int, compiler: Any, side: str | None = None) -> Any:
    from shipdoc.extract import BackendConfig, HfBackend

    cfg = BackendConfig(
        "tiny", "tiny/tiny", "0" * 40, "qwen35", max_new_tokens=60, ocr_token_budget=50
    )
    b = HfBackend(cfg)
    b.torch, b.xgr, b.model, b.compiler = torch, xgr, model, compiler
    b.processor, b.stop_ids, b.image_token_id = (
        _StubProcessor(tok, side),
        [stop],
        _StubProcessor.IMG,
    )
    b._loaded = True
    b.capture_logprobs = True
    return b


def _requests() -> list[Any]:
    from PIL import Image

    from shipdoc.extract import PageRequest

    texts = ["read page one", "p2", "a much longer prompt for the third page of this batch"]
    return [
        PageRequest(Image.new("RGB", (64 * (i + 1), 8)), t, ARRAY, "ocr text " * (i + 1))
        for i, t in enumerate(texts)
    ]


def test_hf_extract_pages_matches_one_page_calls_and_fills_the_meta(
    model: Any, tok: Any, stop: int, compiler: Any
) -> None:
    b = _backend(model, tok, stop, compiler)
    reqs = _requests()
    together = b.extract_pages(reqs)
    alone = [b.extract_pages([r])[0] for r in reqs]
    for (raw, parsed, meta), (raw1, parsed1, meta1), n_img in zip(
        together, alone, (1, 2, 3), strict=True
    ):
        assert raw == raw1 and parsed == parsed1 and isinstance(json.loads(raw), list)
        assert meta["batch_size"] == 3 and meta1["batch_size"] == 1
        assert meta["latency_s"] == pytest.approx(meta["batch_latency_s"] / 3)
        assert meta["n_visual_tokens"] == n_img  # per row, padding excluded
        assert meta["n_input_tokens"] == meta1["n_input_tokens"]  # unpadded length of the row
        assert meta["n_output_tokens"] == meta1["n_output_tokens"] > 0
        assert meta["peak_vram_bytes"] is None and "ocr_tokens" in meta
        assert meta["logprob_trace"]["lp"] == pytest.approx(meta1["logprob_trace"]["lp"], abs=1e-4)
    assert len({m["n_input_tokens"] for _, _, m in together}) == 3  # rows really were padded


def test_hf_extract_pages_refuses_inputs_that_are_not_left_padded(
    model: Any, tok: Any, stop: int, compiler: Any
) -> None:
    b = _backend(model, tok, stop, compiler, side="right")  # a processor that ignores padding_side
    with pytest.raises(ValueError, match="not left-padded"):
        b.extract_pages(_requests())
    assert len(b.extract_pages(_requests()[:1])) == 1  # nothing to pad: fine


def test_hf_extract_pages_rejects_mixed_schemas(
    model: Any, tok: Any, stop: int, compiler: Any
) -> None:
    from dataclasses import replace

    reqs = _requests()
    reqs[1] = replace(reqs[1], schema=LONG)
    with pytest.raises(ValueError, match="one schema"):
        _backend(model, tok, stop, compiler).extract_pages(reqs)
