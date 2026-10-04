"""Per-field token logprobs for keyed-JSON page outputs (Phase 5 calibrator features, gate G4).

Three independent pieces, so each is testable without a GPU:

1. `LogprobRecorder` -- two HF logits processors that wrap the xgrammar processor
   (``[recorder.pre, xgrammar, recorder.post]``). They never modify the scores, so the greedy
   output is bit-identical with and without them. Per generated token they record

   - ``lp``   : log-softmax of the chosen token under the RAW model logits (before the grammar
                mask; the model's own confidence), and
   - ``lp_c`` : the same under the grammar-CONSTRAINED logits (renormalised over the tokens the
                grammar allows).

   Cost: one clone of the logits row, two ``logsumexp`` and one ``argmax`` over the vocabulary per
   step, all on the device, plus no host sync (0-d tensors are stacked once at the end). This is
   NOT a second forward pass. xgrammar's own processor already syncs every step (``.item()``).
2. `token_char_ends` -- character end offsets of every generated token in the decoded text, by
   incremental (prefix) decoding, robust to byte-level BPE tokens that split a multi-byte
   character (the incomplete tokens get a zero-length span and are grouped with the token that
   completes the character).
3. `field_logprobs` -- scan the raw JSON text (tolerating truncation) for the character span of
   every header / row field VALUE and aggregate the logprobs of the tokens overlapping it.

Attribution choices (stated so the calibrator features are interpretable):

- The value span runs from the COLON after the key to the end of the JSON literal (including its
  quotes). The colon is in because real BPE merges it with the neighbours (``":"`` = key-closing
  quote + colon + opening quote, ``":`` before ``null`` or a digit), and that token is the one that
  decides string versus null versus number. A token is attributed to every value whose span it
  overlaps, so a merged ``","`` (closing quote + comma + next key's opening quote) counts for the
  value it closes. Keys, braces and commas that no value overlaps are not attributed.
- ``null`` values are scored (the logprob of the ``null`` token(s)), numbers likewise.
- Duplicate keys: the LAST occurrence wins (what ``json.loads`` keeps); earlier ones are dropped.
- Truncated JSON: every value that starts before the cut is scored; the one cut mid-value carries
  ``"truncated": true``.
- ``row_idx`` is the index of the object in the page's ``line_items`` array as emitted (before
  merge drops all-null rows or other pages are concatenated).
- Only the keyed (``json``) output format is supported; the compact format has positional rows.
"""

from __future__ import annotations

import bisect
import json
import math
import re
import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

HEADER, ROW = "header", "row"
MAX_DEPTH = 32  # nesting guard for hostile / garbage text; real pages nest 3 deep
_WS = " \t\r\n"
_LITERAL_END = ",}]" + _WS
INCOMPLETE = "�"  # decode() output for a byte-level token cut inside a UTF-8 character


# --------------------------------------------------------------------------------------------
# 1. JSON value spans
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ValueSpan:
    """Character span ``[start, end)`` of one field's value in the raw text.

    `start` is the colon that precedes the value (see the module docstring), `end` the end of the
    literal (the closing quote for a string).
    """

    scope: str  # "header" | "row"
    row_idx: int | None
    field: str
    start: int
    end: int
    complete: bool  # False when the text ends inside (or right at the end of) this value


class _Scanner:
    """Tolerant recursive-descent scan of a page object; collects the header / row value spans."""

    def __init__(self, text: str) -> None:
        self.t, self.n = text, len(text)
        self.spans: dict[tuple[str, int | None, str], ValueSpan] = {}
        self.colon = 0  # index of the colon of the member being visited

    def ws(self, i: int) -> int:
        while i < self.n and self.t[i] in _WS:
            i += 1
        return i

    def string(self, i: int) -> tuple[int, bool]:
        """End (exclusive) of the string starting at the quote ``t[i]``; handles escapes."""
        j = i + 1
        while j < self.n:
            c = self.t[j]
            if c == "\\":
                j += 2
                continue
            if c == '"':
                return j + 1, True
            j += 1
        return self.n, False

    def literal(self, i: int) -> tuple[int, bool]:
        j = i
        while j < self.n and self.t[j] not in _LITERAL_END:
            j += 1
        return j, j < self.n  # a literal that runs into the end of the text may be cut

    def key(self, start: int, end: int) -> str:
        raw = self.t[start:end]
        try:
            return str(json.loads(raw))
        except ValueError:
            return raw[1:-1]

    def value(self, i: int, depth: int = 0) -> tuple[int, bool]:
        if depth > MAX_DEPTH:
            return self.n, False
        c = self.t[i]
        if c == '"':
            return self.string(i)
        if c == "{":
            return self.members(i, lambda _k, s: self.value(s, depth + 1))
        if c == "[":
            return self.elements(i, lambda _idx, s: self.value(s, depth + 1))
        return self.literal(i)

    def members(self, i: int, handler: Callable[[str, int], tuple[int, bool]]) -> tuple[int, bool]:
        """Walk the object at ``t[i] == '{'``; ``handler(key, value_start)`` returns its end."""
        j = i + 1
        while True:
            j = self.ws(j)
            if j >= self.n:
                return self.n, False
            c = self.t[j]
            if c == "}":
                return j + 1, True
            if c == ",":
                j += 1
                continue
            if c != '"':
                return self.n, False  # not JSON: stop, everything before is still scored
            kend, ok = self.string(j)
            if not ok:
                return self.n, False
            key = self.key(j, kend)
            j = self.ws(kend)
            if j >= self.n or self.t[j] != ":":
                return self.n, False
            self.colon = j
            j = self.ws(j + 1)
            if j >= self.n:
                return self.n, False
            vend, ok = handler(key, j)
            if not ok:
                return self.n, False
            j = vend

    def elements(self, i: int, handler: Callable[[int, int], tuple[int, bool]]) -> tuple[int, bool]:
        """Walk the array at ``t[i] == '['``; ``handler(index, element_start)`` returns the end."""
        j, idx = i + 1, 0
        while True:
            j = self.ws(j)
            if j >= self.n:
                return self.n, False
            c = self.t[j]
            if c == "]":
                return j + 1, True
            if c == ",":
                j += 1
                continue
            vend, ok = handler(idx, j)
            idx += 1
            if not ok:
                return self.n, False
            j = vend

    def field(self, scope: str, row_idx: int | None, key: str, s: int) -> tuple[int, bool]:
        colon = self.colon  # read before value() recurses into nested members
        end, ok = self.value(s)
        self.spans[(scope, row_idx, key)] = ValueSpan(scope, row_idx, key, colon, end, ok)
        return end, ok

    def drop(self, scope: str) -> None:
        for k in [k for k in self.spans if k[0] == scope]:
            del self.spans[k]

    def page_member(self, key: str, s: int) -> tuple[int, bool]:
        if key == "header" and self.t[s] == "{":
            self.drop(HEADER)  # duplicate "header" key: the last object wins, like json.loads
            return self.members(s, lambda k, vs: self.field(HEADER, None, k, vs))
        if key == "line_items" and self.t[s] == "[":
            self.drop(ROW)
            return self.elements(s, self.row_element)
        return self.value(s)

    def row_element(self, idx: int, s: int) -> tuple[int, bool]:
        if self.t[s] == "{":
            return self.members(s, lambda k, vs: self.field(ROW, idx, k, vs))
        return self.value(s)


def value_spans(text: str) -> list[ValueSpan]:
    """Spans of every header / row field value in a keyed page text, in text order."""
    sc = _Scanner(text)
    i = sc.ws(0)
    if i < sc.n and text[i] == "{":
        sc.members(i, sc.page_member)
    return sorted(sc.spans.values(), key=lambda s: s.start)


# --------------------------------------------------------------------------------------------
# 2. Token -> character offsets
# --------------------------------------------------------------------------------------------


def token_char_ends(
    token_ids: Sequence[int], decode_many: Callable[[list[list[int]]], list[str]]
) -> tuple[str, list[int]]:
    """Decoded text (``.strip()``-ed, exactly what the backend stores as ``raw_text``) and the
    character END offset of every token.

    `decode_many` maps a list of id lists to their decoded strings with special tokens skipped
    (``tokenizer.batch_decode(seqs, skip_special_tokens=True)``). The end of token i is the length
    of the decode of ``ids[:i+1]``. A prefix that ends in U+FFFD is a byte-level token cut inside
    a character: its end stops just before that character and the next token that completes it
    carries it. A token with no complete character of its own, and skipped special tokens (the
    stop token), come out zero length and `_groups` joins them to the next token. Ends are clamped
    monotone and to the stripped text.
    """
    ids = list(token_ids)
    if not ids:
        return "", []
    prefixes = decode_many([ids[: i + 1] for i in range(len(ids))])
    full = prefixes[-1]
    lead = len(full) - len(full.lstrip())
    text = full.strip()
    ends: list[int] = []
    prev = 0
    for k, pre in enumerate(prefixes):
        if pre.endswith(INCOMPLETE) and k < len(prefixes) - 1:
            end = min(max(len(pre) - 1 - lead, prev), len(text))  # drop the half character
        else:
            end = min(max(len(pre) - lead, prev), len(text))
        ends.append(end)
        prev = end
    return text, ends


def _groups(ends: Sequence[int]) -> tuple[list[int], list[int], list[list[int]]]:
    """Group tokens by character span: zero-length tokens join the next non-empty token.

    Returns (group starts, group ends, token indices per group). Trailing zero-length tokens (the
    stop token) belong to no group.
    """
    starts: list[int] = []
    gends: list[int] = []
    members: list[list[int]] = []
    prev, gstart, pending = 0, 0, []
    for i, e in enumerate(ends):
        if not pending:
            gstart = prev
        pending.append(i)
        if e > prev:
            starts.append(gstart)
            gends.append(e)
            members.append(pending)
            pending = []
            prev = e
    return starts, gends, members


# --------------------------------------------------------------------------------------------
# 3. Field aggregation
# --------------------------------------------------------------------------------------------


def _agg(vals: list[float | None]) -> tuple[float | None, float | None]:
    """(min, mean) rounded to 6 decimals, or (None, None) when empty or any value is not finite."""
    if not vals or any(v is None or not math.isfinite(v) for v in vals):
        return None, None
    nums = [float(v) for v in vals if v is not None]
    return round(min(nums), 6), round(sum(nums) / len(nums), 6)


def field_logprobs(
    text: str,
    ends: Sequence[int],
    lp: Sequence[float | None],
    lp_c: Sequence[float | None] | None = None,
) -> list[dict[str, Any]]:
    """Per header / row field value: min and mean token logprob (raw, and constrained if given).

    Entry: ``{scope, row_idx, field, min, mean, n_tokens[, min_c, mean_c][, truncated]}``. ``min``
    / ``mean`` are null if the value overlaps no token or a token's logprob is not finite (a
    consumer must treat that as missing, never as 0). Output order: text order.
    """
    if len(ends) != len(lp) or (lp_c is not None and len(lp_c) != len(lp)):
        raise ValueError("ends, lp and lp_c must have the same length")
    starts, gends, members = _groups(ends)
    out: list[dict[str, Any]] = []
    for sp in value_spans(text):
        g = bisect.bisect_right(gends, sp.start)  # first group ending after the value starts
        toks: list[int] = []
        while g < len(starts) and starts[g] < sp.end:
            toks.extend(members[g])
            g += 1
        mn, mean = _agg([lp[i] for i in toks])
        entry: dict[str, Any] = {
            "scope": sp.scope,
            "row_idx": sp.row_idx,
            "field": sp.field,
            "min": mn,
            "mean": mean,
            "n_tokens": len(toks),
        }
        if lp_c is not None:
            entry["min_c"], entry["mean_c"] = _agg([lp_c[i] for i in toks])
        if not sp.complete:
            entry["truncated"] = True
        out.append(entry)
    return out


def _clean(x: float) -> float | None:
    """Round for storage; non-finite becomes None (JSON has no -Infinity)."""
    return round(x, 6) if math.isfinite(x) else None


def build_trace(
    gen_ids: Sequence[int],
    rec_ids: Sequence[int],
    rec_lp: Sequence[float],
    rec_lp_c: Sequence[float],
    decode_many: Callable[[list[list[int]]], list[str]],
    raw_text: str,
) -> dict[str, Any]:
    """The per-page ``logprob_trace`` stored in the trace: ``{ends, lp, lp_c}`` per token.

    Raises ValueError if the recorder saw different tokens than ``generate`` returned (then the
    logprobs cannot be trusted) or the incremental decode disagrees with `raw_text`.
    """
    gen = [int(t) for t in gen_ids]
    if [int(t) for t in rec_ids] != gen:
        raise ValueError("recorded token ids differ from the generated ids")
    text, ends = token_char_ends(gen, decode_many)
    if text != raw_text:
        raise ValueError("incremental decode differs from raw_text")
    return {
        "ends": ends,
        "lp": [_clean(x) for x in rec_lp],
        "lp_c": [_clean(x) for x in rec_lp_c],
    }


# --------------------------------------------------------------------------------------------
# HF logits processors
# --------------------------------------------------------------------------------------------


class _Pre:
    """Runs BEFORE the grammar processor: keeps a copy of the raw logits and their logsumexp."""

    def __init__(self, rec: LogprobRecorder) -> None:
        self.rec = rec

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        import torch

        raw = scores.detach().to(torch.float32, copy=True)  # xgrammar masks `scores` in place
        self.rec.raw = raw
        self.rec.raw_lse = torch.logsumexp(raw, dim=-1)
        return scores


class _Post:
    """Runs AFTER the grammar processor: the greedy token is the argmax of the final scores.

    Batched: row i records one entry per step while it is unfinished. A row is finished from the
    step AFTER it picked one of ``rec.stop_ids`` (the stop token itself is recorded, like the
    batch-1 path); from then on HF generate forces padding into that row and the entry is marked
    inactive, so padding is never recorded. All of it stays on the device (no host sync per step).
    """

    def __init__(self, rec: LogprobRecorder) -> None:
        self.rec = rec

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        import torch

        rec = self.rec
        n = scores.shape[0]
        if rec.raw is None:
            raise ValueError("LogprobRecorder.pre must run before the grammar processor")
        if n > 1 and rec.stop_ids is None:
            raise ValueError("LogprobRecorder needs stop_ids to track finished rows when batched")
        con = scores.detach().to(torch.float32)
        tok = con.argmax(dim=-1, keepdim=True)  # same op and tie rule as HF greedy decoding
        rec.ids.append(tok[:, 0])
        rec.lp_c.append(con.gather(-1, tok)[:, 0] - torch.logsumexp(con, dim=-1))
        rec.lp.append(rec.raw.gather(-1, tok)[:, 0] - rec.raw_lse)
        if rec.finished is None:
            rec.finished = torch.zeros(n, dtype=torch.bool, device=scores.device)
        rec.active.append(~rec.finished)
        if rec.stop_ids is not None:
            stop = torch.as_tensor(list(rec.stop_ids), device=scores.device)
            rec.finished = rec.finished | torch.isin(tok[:, 0], stop)
        rec.raw = None
        return scores


class LogprobRecorder:
    """``logits_processor=[rec.pre, <xgrammar processor>, rec.post]``; single use.

    Both processors return their input untouched, so generation is unchanged. Batch size 1 needs
    no arguments and `finish` returns that row. A batch needs ``stop_ids`` (the ids at which HF
    generate stops a row) and `finish_rows` returns one ``(ids, lp, lp_c)`` per row, each holding
    exactly the tokens that row generated up to and including its stop token.
    """

    def __init__(self, stop_ids: Sequence[int] | None = None) -> None:
        self.stop_ids = None if stop_ids is None else tuple(int(i) for i in stop_ids)
        self.raw: Any = None
        self.raw_lse: Any = None
        self.finished: Any = None
        self.ids: list[Any] = []
        self.lp: list[Any] = []
        self.lp_c: list[Any] = []
        self.active: list[Any] = []
        self.pre = _Pre(self)
        self.post = _Post(self)

    def finish_rows(self) -> list[tuple[list[int], list[float], list[float]]]:
        """Per row ``(token ids, raw logprobs, constrained logprobs)``; one host sync in total."""
        if not self.ids:
            return []
        import torch

        ids = torch.stack(self.ids).tolist()  # [step][row]
        lp = torch.stack(self.lp).tolist()
        lp_c = torch.stack(self.lp_c).tolist()
        act = torch.stack(self.active).tolist()
        rows = []
        for r in range(len(ids[0])):
            keep = [t for t in range(len(ids)) if act[t][r]]
            rows.append(
                (
                    [int(ids[t][r]) for t in keep],
                    [float(lp[t][r]) for t in keep],
                    [float(lp_c[t][r]) for t in keep],
                )
            )
        return rows

    def finish(self) -> tuple[list[int], list[float], list[float]]:
        """(token ids, raw logprobs, constrained logprobs) of a batch-1 generation."""
        rows = self.finish_rows()
        if not rows:
            return [], [], []
        if len(rows) != 1:
            raise ValueError(f"finish() is for batch size 1, got {len(rows)} rows; use finish_rows")
        return rows[0]


# --------------------------------------------------------------------------------------------
# Mock tokenisation (CPU tests and the mock backend)
# --------------------------------------------------------------------------------------------

_MOCK_TOKEN = re.compile(r"\w+|\W{1,3}")  # 1-3 punctuation chars per token: merged `","` shapes


def mock_logprob_trace(raw_text: str) -> dict[str, Any]:
    """Deterministic fake tokenisation + logprobs of `raw_text` (no model involved).

    Tokens are word runs or 1-3 non-word characters (so tokens straddle quote/comma/colon like a
    real BPE). Logprobs are a pure function of the token text; the constrained value is half the
    raw one (closer to 0, as renormalisation over fewer tokens would give). A trailing
    zero-length token stands in for the stop token.
    """
    toks = list(_MOCK_TOKEN.finditer(raw_text))
    ends = [m.end() for m in toks] + [len(raw_text)]
    lp = [-((zlib.crc32(m.group().encode("utf-8")) % 2000) + 1) / 1000 for m in toks] + [-0.0001]
    return {"ends": ends, "lp": lp, "lp_c": [x / 2 for x in lp]}


def lookup(entries: Sequence[dict[str, Any]]) -> dict[tuple[str, int | None, str], dict[str, Any]]:
    """``(scope, row_idx, field) -> entry`` for tests and consumers."""
    return {(e["scope"], e["row_idx"], e["field"]): e for e in entries}
