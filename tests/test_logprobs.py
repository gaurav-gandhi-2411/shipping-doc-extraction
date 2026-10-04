"""CPU tests for shipdoc.logprobs: value spans, token offsets, aggregation, the HF recorder."""

from __future__ import annotations

import glob
import json
import math
from typing import Any

import pytest

from shipdoc import logprobs as L

PAGE: dict[str, Any] = {
    "doc_type": "invoice",
    "header": {
        "invoice_number": 'INV-"7"\\x',  # escaped quote and backslash inside the value
        "invoice_date": None,
        "total_amount": 12.5,  # a bare JSON number
        "supplier_name": "A, {B} [C]: é中\U0001f600",  # structure chars + non-ASCII
    },
    "line_items": [
        {"supplier_part_number": "X1", "customer_part_number": None, "purchase_order": "PO\n7",
         "quantity": 3},
        {"supplier_part_number": "", "customer_part_number": "C", "purchase_order": None,
         "quantity": None},
    ],
    "page_kind": "single",
}  # fmt: skip
COMPACT = (",", ":")
KEYED_MOCK = (", ", ": ")  # what MockBackend emits (the real grammar emits COMPACT)


def dump(page: dict[str, Any], seps: tuple[str, str] = COMPACT) -> str:
    return json.dumps(page, separators=seps, ensure_ascii=False)


def spans(text: str) -> dict[tuple[str, int | None, str], str]:
    """(scope, row_idx, field) -> the literal text of the value (from the colon, stripped)."""
    return {
        (s.scope, s.row_idx, s.field): text[s.start : s.end].lstrip(":").strip()
        for s in L.value_spans(text)
    }


# ---------------------------------------------------------------------- value spans


@pytest.mark.parametrize("seps", [COMPACT, KEYED_MOCK])
def test_value_spans_recover_every_value_literal(seps: tuple[str, str]) -> None:
    text = dump(PAGE, seps)
    got = spans(text)
    assert len(got) == 4 + 2 * 4
    assert got[("header", None, "invoice_number")] == json.dumps(PAGE["header"]["invoice_number"])
    assert got[("header", None, "invoice_date")] == "null"
    assert got[("header", None, "total_amount")] == "12.5"
    assert got[("header", None, "supplier_name")] == json.dumps(
        PAGE["header"]["supplier_name"], ensure_ascii=False
    )
    assert got[("row", 0, "purchase_order")] == json.dumps("PO\n7")
    assert got[("row", 0, "quantity")] == "3"
    assert got[("row", 1, "supplier_part_number")] == '""'
    assert got[("row", 1, "quantity")] == "null"
    for s in L.value_spans(text):  # the span starts at the colon and the value ends it
        assert text[s.start] == ":" and s.complete


def test_value_spans_ignore_other_top_level_keys_and_nested_junk() -> None:
    text = '{"doc_type":"invoice","extra":{"header":{"x":1}},"header":{"a":"b"},"line_items":[]}'
    assert list(spans(text)) == [("header", None, "a")]


def test_value_spans_row_idx_counts_array_elements_as_emitted() -> None:
    text = '{"header":{},"line_items":[{"a":1},{"a":2},{"a":3}]}'
    assert [(s.row_idx, s.field) for s in L.value_spans(text)] == [(0, "a"), (1, "a"), (2, "a")]


def test_duplicate_keys_last_occurrence_wins() -> None:
    text = '{"header":{"a":"first","b":"x","a":"second"},"line_items":[]}'
    got = spans(text)
    assert got[("header", None, "a")] == '"second"' and got[("header", None, "b")] == '"x"'
    dup_header = '{"header":{"a":"1"},"line_items":[],"header":{"z":"2"}}'
    assert list(spans(dup_header)) == [("header", None, "z")]
    # duplicate keys inside a row
    assert spans('{"line_items":[{"q":1,"q":22}]}')[("row", 0, "q")] == "22"


def test_truncated_text_scores_everything_before_the_cut() -> None:
    text = dump(PAGE)
    full = spans(text)
    counts = []
    for cut in range(len(text)):
        part = text[:cut]
        found = L.value_spans(part)
        counts.append(len(found))
        for s in found:  # every span lies inside the text
            assert 0 <= s.start < s.end <= len(part)
            if s.complete:  # a value that ended before the cut keeps exactly its full-text literal
                assert (
                    part[s.start : s.end].lstrip(":").strip() == full[(s.scope, s.row_idx, s.field)]
                )
    assert counts == sorted(counts)  # a longer prefix never loses a field
    assert counts[-1] >= len(full) - 1  # only the very last value (page_kind follows) is complete


def test_truncation_flag_on_the_cut_value_only() -> None:
    text = '{"header":{"a":"done","b":"cut off here'
    sp = {s.field: s for s in L.value_spans(text)}
    assert sp["a"].complete and not sp["b"].complete and sp["b"].end == len(text)
    num = '{"header":{"a":"x","b":12'  # a number at the very end may continue: not complete
    assert not {s.field: s for s in L.value_spans(num)}["b"].complete
    entries = L.field_logprobs(text, [len(text)], [-1.0])
    assert [bool(e.get("truncated")) for e in entries] == [False, True]


@pytest.mark.parametrize(
    "text",
    ["", "   ", "[]", "null", '"s"', "{", '{"', '{"header', '{"header":', '{"header":{"a"', "{{{{",
     "garbage }{ ][", '{"header":{"a":"x\\', '{"header":{"a":"\\u00', '{"a":1 "b":2}', "\x00\x01"],
)  # fmt: skip
def test_hostile_text_never_raises(text: str) -> None:
    L.value_spans(text)
    L.field_logprobs(text, [len(text)] if text else [], [-1.0] if text else [])


def test_deep_nesting_is_cut_off_not_recursed() -> None:
    text = '{"header":{"a":' + "[" * 5000 + "1" + "]" * 5000 + "}}"
    (s,) = L.value_spans(text)
    assert s.field == "a"


def test_nested_container_as_a_field_value_is_one_span() -> None:
    text = '{"header":{"a":{"k":[1,2,{"z":"}"}],"m":"]"},"b":"after"}}'
    got = spans(text)
    assert got[("header", None, "a")] == '{"k":[1,2,{"z":"}"}],"m":"]"}'
    assert got[("header", None, "b")] == '"after"'


# ---------------------------------------------------------------------- token offsets


def _byte_tokens(text: str, sizes: list[int]) -> list[bytes]:
    """Cut the UTF-8 bytes of `text` into tokens of the given sizes (cycled): may split chars."""
    data, out, i, k = text.encode("utf-8"), [], 0, 0
    while i < len(data):
        n = sizes[k % len(sizes)]
        out.append(data[i : i + n])
        i, k = i + n, k + 1
    return out


class FakeByteTokenizer:
    """Token id = index into a list of byte chunks; decode like a byte-level BPE decoder."""

    def __init__(self, chunks: list[bytes], special: dict[int, str] | None = None) -> None:
        self.chunks, self.special = chunks, special or {}

    def decode_many(self, seqs: list[list[int]]) -> list[str]:
        out = []
        for seq in seqs:
            data = b"".join(self.chunks[i] for i in seq if i not in self.special)
            out.append(data.decode("utf-8", errors="replace"))
        return out


@pytest.mark.parametrize("sizes", [[1], [2], [3, 1, 5], [4, 7, 2, 1], [64]])
def test_token_char_ends_survive_tokens_that_split_characters(sizes: list[int]) -> None:
    text = dump(PAGE)
    tok = FakeByteTokenizer(_byte_tokens(text, sizes))
    ids = list(range(len(tok.chunks)))
    got, ends = L.token_char_ends(ids, tok.decode_many)
    assert got == text and len(ends) == len(ids)
    assert ends == sorted(ends) and ends[-1] == len(text)
    # a character is never split: every non-zero token span, cut from the text, re-encodes to bytes
    # that contain the bytes of its own chunk (the completing token owns the whole character)
    gs, ge, members = L._groups(ends)
    assert gs[0] == 0 and all(a == b for a, b in zip(gs[1:], ge[:-1], strict=True))
    assert sorted(i for m in members for i in m) == ids


def test_split_character_logprobs_follow_the_completing_token() -> None:
    # "中" is 3 bytes; cut it as 1+2: both tokens belong to the value that contains it
    text = '{"header":{"a":"中","b":"z"}}'
    data = text.encode("utf-8")
    cut = data.index("中".encode()) + 1
    chunks = [data[:cut], data[cut:]]
    tok = FakeByteTokenizer(chunks)
    _, ends = L.token_char_ends([0, 1], tok.decode_many)
    assert ends[0] == text.index("中")  # the half character is not decodable yet: zero length
    assert ends[1] == len(text)
    fl = {e["field"]: e for e in L.field_logprobs(text, ends, [-1.0, -3.0])}
    # token 0 stops before the half character; both tokens overlap the value of "a"
    assert fl["a"]["n_tokens"] == 2 and fl["a"]["min"] == -3.0


def test_stop_token_and_leading_whitespace_are_handled() -> None:
    body = '{"header":{"a":"x"}}'
    chunks = [b"  \n", body[:10].encode(), body[10:].encode(), b"<STOP>"]
    tok = FakeByteTokenizer(chunks, special={3: "<STOP>"})
    text, ends = L.token_char_ends([0, 1, 2, 3], tok.decode_many)
    assert text == body  # stripped, exactly like HfBackend's raw_text
    assert ends[0] == 0 and ends[-1] == len(body) and ends[-2] == len(body)
    fl = L.field_logprobs(text, ends, [-9.0, -1.0, -2.0, -0.5])
    assert (
        fl[0]["n_tokens"] >= 1 and fl[0]["min"] >= -2.0
    )  # the whitespace token (-9) is not a value


def test_token_char_ends_empty() -> None:
    assert L.token_char_ends([], lambda s: []) == ("", [])


# ---------------------------------------------------------------------- aggregation


def test_field_logprobs_min_mean_and_n_tokens_by_hand() -> None:
    text = '{"header":{"a":"xy","b":null},"line_items":[{"q":7}]}'
    # tokens: {"header":{"a  ->  ":"  xy  "," b  ":  null  },"line_items":[{"q  ":  7  }]}
    pieces = [
        '{"header":{"a',
        '":"',
        "xy",
        '","',
        "b",
        '":',
        "null",
        '},"line_items":[{"q',
        '":',
        "7",
    ]
    pieces.append("}]}")
    assert "".join(pieces) == text
    ends = [sum(len(p) for p in pieces[: i + 1]) for i in range(len(pieces))]
    lp = [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0, -1.1]
    fl = {e["field"]: e for e in L.field_logprobs(text, ends, lp, [x / 2 for x in lp])}
    # a: colon is the 1st char of '":"' -> tokens '":"', 'xy', '","' (closing quote)
    assert fl["a"]["n_tokens"] == 3
    assert fl["a"]["min"] == -0.4 and fl["a"]["mean"] == pytest.approx(-0.3)
    assert fl["a"]["min_c"] == -0.2 and fl["a"]["mean_c"] == pytest.approx(-0.15)
    # b: null value: '":' and 'null' (the '","' belongs to a, "b" key token to nobody)
    assert fl["b"]["n_tokens"] == 2 and fl["b"]["min"] == -0.7
    assert fl["b"]["mean"] == pytest.approx(-0.65)
    assert fl["q"]["n_tokens"] == 2 and fl["q"]["mean"] == pytest.approx(-0.95)


def test_non_finite_logprob_gives_null_not_zero_and_length_mismatch_raises() -> None:
    text = '{"header":{"a":"x"}}'
    n = len(text)
    (e,) = L.field_logprobs(text, [n], [float("-inf")])
    assert e["min"] is None and e["mean"] is None and e["n_tokens"] == 1
    (e,) = L.field_logprobs(text, [n], [None])  # type: ignore[list-item]
    assert e["min"] is None
    with pytest.raises(ValueError, match="same length"):
        L.field_logprobs(text, [n], [-1.0, -2.0])


def test_value_with_no_token_has_zero_tokens_and_null_stats() -> None:
    text = '{"header":{"a":"x"}}'
    (e,) = L.field_logprobs(text, [], [])
    assert e["n_tokens"] == 0 and e["min"] is None and e["mean"] is None


def test_build_trace_checks_ids_and_text() -> None:
    text = '{"header":{"a":"x"}}'
    tok = FakeByteTokenizer([text.encode()[:7], text.encode()[7:]])
    tr = L.build_trace([0, 1], [0, 1], [-0.5, float("nan")], [-0.4, -0.3], tok.decode_many, text)
    assert tr["lp"] == [-0.5, None] and tr["lp_c"] == [-0.4, -0.3] and tr["ends"][-1] == len(text)
    with pytest.raises(ValueError, match="differ from the generated"):
        L.build_trace([0, 1], [0, 0], [0.0, 0.0], [0.0, 0.0], tok.decode_many, text)
    with pytest.raises(ValueError, match="differs from raw_text"):
        L.build_trace([0, 1], [0, 1], [0.0, 0.0], [0.0, 0.0], tok.decode_many, text + " ")
    json.dumps(tr)  # storable: no -Infinity / NaN


def test_mock_trace_is_deterministic_and_covers_every_value() -> None:
    text = dump(PAGE, KEYED_MOCK)
    a, b = L.mock_logprob_trace(text), L.mock_logprob_trace(text)
    assert a == b and a["ends"][-1] == len(text) and a["ends"][-2] == len(text)  # stop token
    fl = L.field_logprobs(text, a["ends"], a["lp"], a["lp_c"])
    assert len(fl) == 12 and all(e["n_tokens"] >= 1 and math.isfinite(e["min"]) for e in fl)
    assert all(e["min"] <= e["mean"] <= 0 and e["min_c"] >= e["min"] for e in fl)


# ---------------------------------------------------------------------- real tokenizer


def _qwen_tokenizer_json() -> str | None:
    """Local Qwen3.5-4B tokenizer.json (the pinned model's), or None on a machine without it."""
    from shipdoc import paths

    try:
        home = paths.get_path("HF_HOME")
    except KeyError:
        return None
    found = glob.glob(
        str(home / "hub" / "models--Qwen--Qwen3.5-4B" / "snapshots" / "*" / "tokenizer.json")
    )
    return found[0] if found else None


@pytest.fixture(scope="module")
def qwen_tok() -> Any:
    tokenizers = pytest.importorskip("tokenizers")
    path = _qwen_tokenizer_json()
    if path is None:
        pytest.skip("Qwen3.5-4B tokenizer.json not in the local HF cache")
    return tokenizers.Tokenizer.from_file(path)


@pytest.mark.parametrize("seps", [COMPACT, KEYED_MOCK])
def test_real_qwen_tokenizer_ends_match_the_tokenizers_offsets(qwen_tok: Any, seps: Any) -> None:
    text = dump(PAGE, seps)
    enc = qwen_tok.encode(text)
    got, ends = L.token_char_ends(
        enc.ids, lambda seqs: [qwen_tok.decode(s, skip_special_tokens=True) for s in seqs]
    )
    assert got == text
    assert ends == [o[1] for o in enc.offsets]  # independent method: encoder-side char offsets
    # the emoji and the CJK character are byte-fallback / multi-byte: at least one token pair shares
    assert len(ends) == len(enc.ids)


def test_real_qwen_tokens_attribute_the_decision_token_to_null_and_string(qwen_tok: Any) -> None:
    text = dump(PAGE)
    enc = qwen_tok.encode(text)
    lp = [-float(i + 1) for i in range(len(enc.ids))]  # unique, increasingly negative
    _, ends = L.token_char_ends(
        enc.ids, lambda seqs: [qwen_tok.decode(s, skip_special_tokens=True) for s in seqs]
    )
    fl = L.lookup(L.field_logprobs(text, ends, lp))
    toks = [qwen_tok.decode([i]) for i in enc.ids]
    first_null = toks.index("null")
    e = fl[("header", None, "invoice_date")]
    # the colon-bearing token before `null` (":) and `null` itself; nothing else
    assert e["n_tokens"] == 2 and e["min"] == lp[first_null]
    assert fl[("header", None, "invoice_date")]["mean"] == pytest.approx(
        (lp[first_null - 1] + lp[first_null]) / 2
    )
    # every field of the page got at least its decision token
    assert all(v["n_tokens"] >= 1 for v in fl.values()) and len(fl) == 12


# ---------------------------------------------------------------------- HF recorder (tiny model)


def _tiny_model() -> Any:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(42)
    cfg = transformers.LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=128,
        bos_token_id=1, eos_token_id=2, pad_token_id=0,
    )  # fmt: skip
    return transformers.LlamaForCausalLM(cfg).eval()


class MaskInPlace:
    """Stand-in for xgrammar: forbids half of the vocabulary by writing -inf INTO the scores."""

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        scores[:, ::2] = float("-inf")
        return scores


def _generate(model: Any, procs: list[Any], **kw: Any) -> Any:
    import torch

    prompt = torch.tensor([[1, 5, 9, 13]])
    with torch.inference_mode():
        return model.generate(
            prompt, max_new_tokens=12, do_sample=False, num_beams=1, temperature=None,
            top_p=None, top_k=None, logits_processor=procs, eos_token_id=[2], pad_token_id=0,
            **kw,
        )  # fmt: skip


def test_recorder_matches_reference_logprobs_and_never_changes_the_output() -> None:
    import torch

    model = _tiny_model()
    base = _generate(model, [MaskInPlace()])[0]
    rec = L.LogprobRecorder()
    out = _generate(model, [rec.pre, MaskInPlace(), rec.post])[0]
    assert out.tolist() == base.tolist()  # observers do not change greedy decoding
    gen = out[4:].tolist()

    ref = _generate(model, [MaskInPlace()], output_scores=True, return_dict_in_generate=True)
    assert ref.sequences[0].tolist() == base.tolist()
    ids, lp, lp_c = rec.finish()
    assert ids == gen and len(lp) == len(lp_c) == len(gen)
    # HF's `output_logits` alias the tensor that the (in-place) mask then edits, so the raw
    # reference is a teacher-forced forward pass over prompt + generation instead.
    with torch.inference_mode():
        full = model(out[None, :-1]).logits[0].float()
    for t, tok in enumerate(gen):
        raw = torch.log_softmax(full[3 + t], dim=-1)[tok].item()
        con = torch.log_softmax(ref.scores[t][0].float(), dim=-1)[tok].item()
        assert lp[t] == pytest.approx(raw, abs=1e-4)
        assert lp_c[t] == pytest.approx(con, abs=1e-5)
        assert lp_c[t] >= lp[t] - 1e-6  # renormalising over fewer tokens can only raise it
    assert all(math.isfinite(x) for x in lp + lp_c)


def test_recorder_rejects_missing_pre_and_batches_without_stop_ids() -> None:
    torch = pytest.importorskip("torch")
    rec = L.LogprobRecorder()
    with pytest.raises(ValueError, match="pre must run"):
        rec.post(torch.zeros(2, 3, dtype=torch.long), torch.zeros(2, 8))
    rec.pre(torch.zeros(2, 3, dtype=torch.long), torch.zeros(2, 8))
    with pytest.raises(ValueError, match="needs stop_ids"):
        rec.post(torch.zeros(2, 3, dtype=torch.long), torch.zeros(2, 8))
    assert L.LogprobRecorder().finish() == ([], [], [])
    assert L.LogprobRecorder().finish_rows() == []


def test_recorder_finish_is_for_batch_one_only() -> None:
    torch = pytest.importorskip("torch")
    rec = L.LogprobRecorder(stop_ids=[1])
    ids, scores = torch.zeros(2, 3, dtype=torch.long), torch.randn(2, 8)
    rec.pre(ids, scores)
    rec.post(ids, scores)
    assert len(rec.finish_rows()) == 2
    with pytest.raises(ValueError, match="batch size 1"):
        rec.finish()


def test_recorder_overhead_per_step_on_a_realistic_vocabulary() -> None:
    """Not an assertion on speed (CPU != T4); documents that the work is a few vector ops."""
    torch = pytest.importorskip("torch")
    import time

    rec = L.LogprobRecorder()
    scores = torch.randn(1, 248_320)
    ids = torch.zeros(1, 1, dtype=torch.long)
    t0 = time.perf_counter()
    for _ in range(50):
        rec.pre(ids, scores)
        rec.post(ids, scores)
    per_step = (time.perf_counter() - t0) / 50
    assert per_step < 0.5  # generous: ~ms on CPU; the T4 decode step is ~76 ms
    assert len(rec.finish()[0]) == 50


def test_xgrammar_tiny_model_end_to_end_with_the_real_qwen_tokenizer(qwen_tok: Any) -> None:
    """Real xgrammar mask (in place) + real Qwen vocabulary + random 1-layer model on CPU.

    The text is random but grammar-valid (cut at max_new_tokens): the recorder must not change the
    greedy ids, the incremental offsets must reproduce `raw_text`, and every complete value the
    scanner finds must be a valid JSON literal with finite logprobs on >= 1 token.
    """
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    xgr = pytest.importorskip("xgrammar")
    from shipdoc.extract import page_schema, schema_text

    path = _qwen_tokenizer_json()
    assert path is not None
    tok = transformers.PreTrainedTokenizerFast(tokenizer_file=path)
    vocab = 248_320  # Qwen3.5 text vocab_size (padded), as in HfBackend.load
    stop = tok.convert_tokens_to_ids("<|im_end|>")
    torch.manual_seed(42)
    cfg = transformers.LlamaConfig(
        vocab_size=vocab, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=4096,
    )  # fmt: skip
    model = transformers.LlamaForCausalLM(cfg).eval()
    for p in model.parameters():
        p.data.normal_(0, 0.5)
    info = xgr.TokenizerInfo.from_huggingface(tok, vocab_size=vocab, stop_token_ids=[stop])
    grammar = xgr.GrammarCompiler(info).compile_json_schema(
        schema_text(page_schema()), any_whitespace=False
    )

    def run(use_rec: bool) -> tuple[list[int], Any]:
        proc = xgr.contrib.hf.LogitsProcessor(grammar)
        rec = L.LogprobRecorder() if use_rec else None
        procs = [rec.pre, proc, rec.post] if rec else [proc]
        with torch.inference_mode():
            out = model.generate(
                torch.tensor([[1, 5, 9, 13]]), max_new_tokens=300, do_sample=False, num_beams=1,
                temperature=None, top_p=None, top_k=None, logits_processor=procs,
                eos_token_id=[stop], pad_token_id=0,
            )  # fmt: skip
        return out[0, 4:].tolist(), rec

    plain, _ = run(False)
    gen, rec = run(True)
    assert gen == plain  # identical greedy tokens with and without the recorders
    raw = tok.decode(gen, skip_special_tokens=True).strip()
    ids, lp, lp_c = rec.finish()
    trace = L.build_trace(
        gen, ids, lp, lp_c, lambda s: tok.batch_decode(s, skip_special_tokens=True), raw
    )
    entries = L.field_logprobs(raw, trace["ends"], trace["lp"], trace["lp_c"])
    assert entries, raw[:80]
    for e in entries:
        assert e["n_tokens"] >= 1 and math.isfinite(e["min"]) and e["min"] <= e["mean"] <= 0
    for s in L.value_spans(raw):
        if s.complete:
            json.loads(raw[s.start : s.end].lstrip(":").strip())  # a real JSON literal
