"""Loss-masking proof (STEP Z2): the loss is computed ONLY on assistant target tokens.

Two layers. (1) Structure, with the fake tokenizer: ``encode_page`` labels, then
``chunked_target_loss`` fed with hidden states that are PERFECT at every position that is supposed
to be supervised and ADVERSARIAL everywhere else - the loss is ~0 iff nothing else is scored, and
it jumps when any single supervised position (first target token, last target token, the final
``<|im_end|>``) is spoiled, which also proves the label shift (position t scores token t+1).
(2) The same with the REAL Qwen3.5 tokenizer and chat template when the tokenizer files are in the
local HF cache (skipped otherwise): prompt-token labels are all -100 and the supervised span
decodes to exactly the keyed JSON + ``<|im_end|>``. Only the tokenizer files are needed; the
image is the 1,260 ``<|image_pad|>`` tokens the processor expands its placeholder to.
"""

from __future__ import annotations

import glob
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import pytest

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
torch = pytest.importorskip("torch")
from test_train import (  # noqa: E402
    END_ID,
    IMG_ID,
    FakeProcessor,
    encode,
    rendered,
)

from shipdoc import train as tr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCALE = 30.0  # logit margin: a correctly scored token costs ~e^-30, a spoiled one ~30 nats


def compact(ids: Any, labels: Any) -> tuple[Any, Any, Any]:
    """Map the (<= a few hundred) distinct token ids of a sample to 0..V'-1 so a one-hot hidden
    state can address the vocabulary: returns (mapped ids, mapped labels with -100 kept, V')."""
    uniq = sorted({int(x) for x in ids[0]} | {int(x) for x in labels[0] if int(x) != -100})
    lut = {t: i for i, t in enumerate(uniq)}
    m_ids = torch.tensor([[lut[int(x)] for x in ids[0]]])
    m_lab = torch.tensor([[lut[int(x)] if int(x) != -100 else -100 for x in labels[0]]])
    return m_ids, m_lab, len(uniq)


def perfect_hidden(labels: Any, vocab: int, spoil: tuple[int, ...] = ()) -> Any:
    """Hidden states (S, V'): at the position that predicts a supervised token, a one-hot of that
    token; at every other position a one-hot of a WRONG token (the one after the label's, or 0).
    Positions in `spoil` (indices of the PREDICTING position) are made wrong as well."""
    s = labels.shape[1]
    h = torch.zeros(1, s, vocab)
    for t in range(s):
        nxt = int(labels[0, t + 1]) if t + 1 < s else -100
        wrong = (max(nxt, 0) + 1) % vocab
        right = nxt != -100 and t not in spoil
        h[0, t, nxt if right else wrong] = SCALE
    return h


def loss_of(enc: dict[str, Any], spoil: tuple[int, ...] = ()) -> tuple[float, int]:
    ids, lab, vocab = compact(enc["input_ids"], enc["labels"])
    head = torch.eye(vocab)  # logits = hidden: position t predicts the one-hot's token
    n = int((lab[:, 1:] != -100).sum())
    return float(tr.chunked_target_loss(perfect_hidden(lab, vocab, spoil), head, lab, 7)), n


# --------------------------------------------------------------------------------------------
# (1) structure, fake tokenizer
# --------------------------------------------------------------------------------------------


def test_only_target_tokens_and_the_stop_token_are_scored_adversarial_everywhere_else() -> None:
    enc = encode(rendered())
    loss, n = loss_of(enc)
    assert n == int((enc["labels"] != -100).sum())  # every label is scored exactly once
    assert loss < 1e-6  # prompt / image / pad positions are wrong on purpose and cost nothing


def test_label_shift_position_t_scores_token_t_plus_1() -> None:
    enc = encode(rendered())
    labels = enc["labels"][0]
    first = int((labels != -100).nonzero()[0])  # first target token
    last = int(labels.numel()) - 1  # <|im_end|>
    assert int(labels[last]) == END_ID
    n_prefix = 3 + 1260 + 4
    assert first == n_prefix
    base, n = loss_of(enc)
    # spoil the position that PREDICTS the first target token (= the last prompt token)
    first_spoiled, _ = loss_of(enc, spoil=(first - 1,))
    assert first_spoiled == pytest.approx(SCALE / n, rel=0.05)
    # the predictor of the final <|im_end|> is the last target token's position
    stop_spoiled, _ = loss_of(enc, spoil=(last - 1,))
    assert stop_spoiled == pytest.approx(SCALE / n, rel=0.05)  # the stop token IS supervised
    # spoiling the position that holds the first target token itself scores token t+1, not t:
    self_spoiled, _ = loss_of(enc, spoil=(first,))
    assert self_spoiled == pytest.approx(SCALE / n, rel=0.05) and base < 1e-6
    # the last position has no successor: it predicts nothing and is never scored
    assert int(enc["labels"][0, 0]) == -100
    ids, lab, v = compact(enc["input_ids"], enc["labels"])
    h = perfect_hidden(lab, v)
    h[0, -1] = torch.randn(v) * 1e3
    assert float(tr.chunked_target_loss(h, torch.eye(v), lab, 7)) < 1e-6


def test_prompt_system_image_and_padding_labels_are_all_minus_100() -> None:
    enc = encode(rendered(), FakeProcessor(n_visual=1260))
    labels, ids = enc["labels"][0], enc["input_ids"][0]
    n_prefix = 3 + 1260 + 4
    assert (labels[:n_prefix] == -100).all()
    assert int((ids[:n_prefix] == IMG_ID).sum()) == 1260  # the image tokens are inside the mask
    assert int((labels == IMG_ID).sum()) == 0
    # padding: attention_mask is all ones (no padded batch exists: micro-batch 1), so nothing to
    # mask there; a padded token would carry label -100 by construction of the dataset
    assert (enc["attention_mask"] == 1).all()
    sup = labels[labels != -100]
    assert sup.tolist() == ids[n_prefix:].tolist() and int(sup[-1]) == END_ID


def test_header_only_pages_score_nothing_in_line_items_and_the_stop_token_stays() -> None:
    r = rendered(supervise_line_items=False)
    enc = encode(r)
    n_prefix = 3 + 1260 + 4
    s, e = r.target.line_items_span
    lab = enc["labels"][0]
    assert (lab[n_prefix + s : n_prefix + e] == -100).all()
    loss, n = loss_of(enc)
    assert loss < 1e-6 and n == len(r.target.text) - (e - s) + 1


def test_a_sample_without_any_supervised_token_is_an_error_not_a_zero_loss() -> None:
    with pytest.raises(tr.TrainError, match="no supervised"):
        tr.chunked_target_loss(torch.zeros(1, 5, 4), torch.zeros(7, 4), torch.full((1, 5), -100))


def test_loss_is_the_mean_over_supervised_tokens_only_not_diluted_by_the_prompt() -> None:
    """The denominator is the number of supervised positions: a 100x longer masked prompt leaves
    the loss bit-identical (why the supervised share does not explain a low loss)."""
    g = torch.Generator().manual_seed(0)
    w = torch.randn(11, 6, generator=g)
    predictor = torch.randn(1, 1, 6, generator=g)  # last prompt position: predicts token 0
    tail_h = torch.randn(1, 9, 6, generator=g)  # 8 predictors + the last position (unscored)
    tail_y = torch.randint(0, 11, (1, 9), generator=g)  # 9 supervised tokens

    def loss(prompt_len: int) -> Any:
        prompt = torch.randn(1, prompt_len - 1, 6, generator=g)  # arbitrary, never scored
        h = torch.cat([prompt, predictor, tail_h], 1)
        y = torch.cat([torch.full((1, prompt_len), -100), tail_y], 1)
        return tr.chunked_target_loss(h, w, y, 4)

    short, long = loss(3), loss(300)
    assert torch.equal(short, long)
    # and it equals the plain mean of the 9 per-token cross-entropies
    logits = torch.cat([predictor, tail_h[:, :8]], 1)[0] @ w.t()
    ref = torch.nn.functional.cross_entropy(logits, tail_y[0])
    assert torch.allclose(short, ref, atol=1e-6)


# --------------------------------------------------------------------------------------------
# the stats script: char classes and token kinds
# --------------------------------------------------------------------------------------------


def _script() -> Any:
    spec = importlib.util.spec_from_file_location(
        "supervised_share", ROOT / "scripts" / "supervised_share.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["supervised_share"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_char_classes_separate_keys_values_and_nulls() -> None:
    ss = _script()
    text = '{"a": "xy", "b": null, "c": [{"k": "z"}], "n": 12}'
    cls = ss.char_classes(text)
    assert len(cls) == len(text)
    assert [c for c, ch in zip(cls, text, strict=True) if c == "V" and ch in "xyz"] == list("VVV")
    assert cls[text.index("xy") : text.index("xy") + 2] == "VV"
    assert cls[text.index("null") : text.index("null") + 4] == "NNNN"
    assert cls[text.index("12") : text.index("12") + 2] == "VV"
    for key in ("a", "b", "c", "k", "n"):  # keys, with their quotes, are structure
        i = text.index(f'"{key}"')
        assert cls[i : i + 3] == "SSS"
    assert cls[text.index('"xy"')] == "S" and cls[text.index('"xy"') + 3] == "S"  # value quotes


def test_token_kinds_value_wins_over_null_wins_over_structure() -> None:
    ss = _script()
    text = '{"a": "xy", "b": null}'
    cls = ss.char_classes(text)
    offsets = [(0, 1), (1, 4), (4, 7), (7, 9), (9, 17), (17, 21), (21, 22)]
    masked = [False] * 6 + [True]
    kinds = ss.token_kinds(cls, offsets, masked)
    assert kinds == ["structure", "structure", "structure", "value", "structure", "null", None]


# --------------------------------------------------------------------------------------------
# (2) real Qwen3.5 tokenizer + chat template
# --------------------------------------------------------------------------------------------

SNAP = glob.glob(r"D:\shipdoc\hf_cache\hub\models--Qwen--Qwen3.5-4B\snapshots\*")
needs_tokenizer = pytest.mark.skipif(not SNAP, reason="Qwen3.5 tokenizer files not in the HF cache")


@pytest.fixture(scope="module")
def real() -> Any:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(SNAP[0], local_files_only=True)
    ss = _script()
    return ss.RealShim(tok), tok


@needs_tokenizer
def test_real_template_prompt_labels_are_masked_and_target_decodes_exactly(real: Any) -> None:
    shim, tok = real
    r = rendered()
    enc = tr.encode_page(r, shim, adapter_name="qwen35",
                         image_token_id=int(tok.convert_tokens_to_ids("<|image_pad|>")),
                         expected_visual_tokens=1260)  # fmt: skip
    ids, labels = enc["input_ids"][0], enc["labels"][0]
    sup = labels != -100
    n_prefix = int((~sup).sum())
    assert (labels[:n_prefix] == -100).all() and sup[n_prefix:].all()  # contiguous tail
    # the prefix is the real template: system/user turn, vision tokens, generation prompt
    prefix_text = tok.decode(ids[:n_prefix])
    assert prefix_text.startswith("<|im_start|>user\n<|vision_start|><|image_pad|>")
    assert prefix_text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert int((ids[:n_prefix] == tok.convert_tokens_to_ids("<|image_pad|>")).sum()) == 1260
    # the supervised span decodes to exactly the keyed JSON + <|im_end|>
    assert tok.decode(ids[sup]) == r.target.text + "<|im_end|>"
    assert int(labels[-1]) == tok.convert_tokens_to_ids("<|im_end|>")
    assert labels[sup].tolist() == ids[sup].tolist()


@needs_tokenizer
def test_real_tokenizer_whole_conversation_tokenises_like_prefix_plus_target(real: Any) -> None:
    """The boundary between the generation prompt and the target does not merge differently when
    the template renders the full conversation: our concatenation == tokenising the final text."""
    shim, tok = real
    from shipdoc.extract import get_adapter, page_schema

    r = rendered()
    msgs, kw = get_adapter("qwen35").build(r.image, r.prompt, page_schema(), "json")
    full = tok.apply_chat_template([*msgs, {"role": "assistant", "content": r.target.text}],
                                   add_generation_prompt=False, tokenize=False, **kw)  # fmt: skip
    full = full.replace("<|image_pad|>", "<|image_pad|>" * 1260)
    whole = tok(full, add_special_tokens=False)["input_ids"]
    enc = tr.encode_page(r, shim, adapter_name="qwen35",
                         image_token_id=int(tok.convert_tokens_to_ids("<|image_pad|>")),
                         expected_visual_tokens=1260)  # fmt: skip
    ours = enc["input_ids"][0].tolist()
    assert whole[: len(ours)] == ours  # then only the template's trailing "\n" after <|im_end|>
    assert tok.decode(whole[len(ours) :]) == "\n"


@needs_tokenizer
def test_real_template_header_only_masks_the_row_array_and_keeps_the_stop_token(
    real: Any,
) -> None:
    shim, tok = real
    r = rendered(supervise_line_items=False)
    enc = tr.encode_page(r, shim, adapter_name="qwen35",
                         image_token_id=int(tok.convert_tokens_to_ids("<|image_pad|>")),
                         expected_visual_tokens=1260)  # fmt: skip
    ids, labels = enc["input_ids"][0], enc["labels"][0]
    sup_text = tok.decode(ids[labels != -100])
    head = r.target.text[: r.target.line_items_span[0]]
    tail = r.target.text[r.target.line_items_span[1] :]
    assert sup_text.startswith(head.rstrip()[:20])
    # tokens that straddle the span edge (e.g. "]," or ', "') overlap it and are masked too: the
    # supervised tail is the target's end, possibly minus those separator characters
    assert sup_text.endswith(tail.lstrip(", ") + "<|im_end|>")
    assert '"supplier_part_number"' not in sup_text  # the masked rows
    assert '"supplier_part_number"' in tok.decode(ids)  # but they are in the context


@needs_tokenizer
def test_real_template_end_to_end_loss_scores_only_the_target(real: Any) -> None:
    shim, tok = real
    r = rendered()
    enc = tr.encode_page(r, shim, adapter_name="qwen35",
                         image_token_id=int(tok.convert_tokens_to_ids("<|image_pad|>")),
                         expected_visual_tokens=1260)  # fmt: skip
    loss, n = loss_of(enc)
    assert loss < 1e-6 and n == int((enc["labels"] != -100).sum())
    assert n < 400 and enc["input_ids"].shape[1] > 1260 + n  # the prompt is scored 0 times
