"""Supervised-token share of the fine-tune training pages (STEP Z2): aggregates only.

Answers "what fraction of each training sequence is supervised, and what kind of tokens are they":
for every page of the ``final`` training set (536 pages) it builds the sample with the SAME code
the trainer uses (``train.encode_page``, clean epoch-0 target, no augmentation) and counts

* supervised tokens (labels != -100: target JSON + the final ``<|im_end|>``) vs all tokens,
* the supervised tokens by kind: structure (keys, braces, quotes, separators), null literals,
  value-bearing tokens (any character of a string/number value) and the stop token,
* all of that per page type (waybill / invoice x page_kind) and per sample (mean / min / max).

The tokenizer is the REAL Qwen3.5 tokenizer + chat template from the local HF cache when present
(label ``real``); otherwise the character-level FAKE tokenizer of the unit tests (label ``fake``,
token counts then only illustrate the arithmetic). The image is represented by its 1,260
``<|image_pad|>`` tokens (what the processor expands the single placeholder to); the pixels are not
needed. Output: markdown with counts only, never a page, a value or a name.

Run: uv run python scripts/supervised_share.py [--out reports/supervised_tokens.md]
"""

# ruff: noqa: E501  # markdown prose literals

from __future__ import annotations

import argparse
import glob
import re
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from shipdoc import paths
from shipdoc import train as tr
from shipdoc import trainset as ts

ROOT = Path(__file__).resolve().parents[1]
VISUAL_TOKENS = 1260
HF_CACHE_GLOB = r"D:\shipdoc\hf_cache\hub\models--Qwen--Qwen3.5-4B\snapshots\*"
OBSERVED = {"first 5 steps": 0.0557, "last 5 steps": 0.0056}  # smoke_status.json, L4 bf16

_STRING = re.compile(r'"(?:[^"\\]|\\.)*"')
_SCALAR = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null")


def char_classes(text: str) -> str:
    """One class letter per character of a JSON text: ``S`` structure (object keys with their
    quotes, braces, brackets, colons, commas, spaces, the quotes around values), ``V`` the content
    of a string value or a number / boolean, ``N`` a ``null`` literal."""
    out = ["S"] * len(text)
    pos = 0
    while pos < len(text):
        ms, mc = _STRING.match(text, pos), _SCALAR.match(text, pos)
        if ms:
            tail = text[ms.end() :].lstrip(" ")
            if not tail.startswith(":"):  # a value string (a key is followed by ':')
                for i in range(ms.start() + 1, ms.end() - 1):
                    out[i] = "V"
            pos = ms.end()
        elif mc:
            kind = "N" if mc.group() == "null" else "V"
            for i in range(mc.start(), mc.end()):
                out[i] = kind
            pos = mc.end()
        else:
            pos += 1
    return "".join(out)


def token_kinds(
    classes: str, offsets: list[tuple[int, int]], masked: list[bool]
) -> list[str | None]:
    """Kind of each target token: ``value`` (any V character), ``null`` (any N, no V),
    ``structure`` (all S), or None for a masked token (header-only line_items)."""
    kinds: list[str | None] = []
    for (a, b), m in zip(offsets, masked, strict=True):
        if m:
            kinds.append(None)
            continue
        cs = set(classes[a:b])
        kinds.append("value" if "V" in cs else "null" if "N" in cs else "structure")
    return kinds


class RealShim:
    """Processor stand-in: the real tokenizer and chat template, the image placeholder expanded to
    `VISUAL_TOKENS` pad tokens as the processor does. No pixels (they are not tokenised)."""

    def __init__(self, tokenizer: Any) -> None:
        import torch

        self.tokenizer, self._torch = tokenizer, torch

    def apply_chat_template(self, messages: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
        text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=kw["add_generation_prompt"], tokenize=False,
            enable_thinking=kw.get("enable_thinking", True),
        )  # fmt: skip
        text = text.replace("<|image_pad|>", "<|image_pad|>" * VISUAL_TOKENS)
        ids = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
        return {
            "input_ids": ids,
            "attention_mask": self._torch.ones_like(ids),
            "pixel_values": self._torch.zeros(4, 3),
            "image_grid_thw": self._torch.tensor([[1, 2, 2]]),
        }


def load_real_tokenizer() -> Any | None:
    """The cached Qwen3.5 tokenizer (offline), or None when the cache is absent."""
    snaps = glob.glob(HF_CACHE_GLOB)
    if not snaps:
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(snaps[0], local_files_only=True)


def sample_stats(
    spec: ts.PageSpec, prepared: ts.PreparedSet, shim: Any, image_token_id: int
) -> dict[str, Any]:
    """Counts of one clean (epoch 0, no augmentation) sample."""
    rendered = prepared.render(spec, 0, ts.AugmentConfig(0.0, 0.0, 42))
    enc = tr.encode_page(rendered, shim, adapter_name="qwen35", image_token_id=image_token_id,
                         expected_visual_tokens=VISUAL_TOKENS)  # fmt: skip
    labels = enc["labels"][0]
    n_total = int(labels.numel())
    n_sup = int((labels != tr.IGNORE_INDEX).sum())
    tok = shim.tokenizer
    t = tok(rendered.target.text, add_special_tokens=False, return_offsets_mapping=True)
    masked = (
        [False] * len(t["input_ids"])
        if rendered.target.supervise_line_items
        else tr.line_items_token_mask(t["offset_mapping"], rendered.target.line_items_span)
    )
    kinds = Counter(token_kinds(char_classes(rendered.target.text), t["offset_mapping"], masked))
    n_prefix = n_total - len(t["input_ids"]) - 1
    return {
        "doc_id": spec.doc_id,
        "waybill": prepared.golds[spec.doc_id].get("doc_type") != "invoice",
        "page_kind": rendered.target.payload["page_kind"],
        "header_only": not rendered.target.supervise_line_items,
        "n_total": n_total,
        "n_prefix": n_prefix,
        "n_target_text": len(t["input_ids"]),
        "n_sup": n_sup,
        "structure": kinds["structure"],
        "null": kinds["null"],
        "value": kinds["value"],
        "masked_in_target": kinds[None],
        "stop": 1,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean / min / max of the supervised share and the kind shares over `rows`."""
    share = [r["n_sup"] / r["n_total"] for r in rows]
    sup = sum(r["n_sup"] for r in rows)

    def kind_share(k: str) -> float:
        return sum(r[k] for r in rows) / sup

    return {
        "pages": len(rows),
        "share_mean": statistics.fmean(share),
        "share_min": min(share),
        "share_max": max(share),
        "sup_per_page_mean": sup / len(rows),
        "sup_per_page_min": min(r["n_sup"] for r in rows),
        "sup_per_page_max": max(r["n_sup"] for r in rows),
        "sup_total": sup,
        "total_tokens": sum(r["n_total"] for r in rows),
        "structure": kind_share("structure"),
        "null": kind_share("null"),
        "value": kind_share("value"),
        "stop": kind_share("stop"),
        # per-sample mean of the value share: the unit the trainer's loss averages over
        "value_share_sample_mean": statistics.fmean(r["value"] / r["n_sup"] for r in rows),
    }


def row(label: str, s: dict[str, Any]) -> str:
    """One markdown table row."""
    return (
        f"| {label} | {s['pages']} | {100 * s['share_mean']:.1f}% | {100 * s['share_min']:.1f}% | "
        f"{100 * s['share_max']:.1f}% | {s['sup_per_page_mean']:.0f} "
        f"({s['sup_per_page_min']}-{s['sup_per_page_max']}) | {s['sup_total']:,} | "
        f"{100 * s['structure']:.1f}% | {100 * s['null']:.1f}% | {100 * s['value']:.1f}% | "
        f"{100 * s['stop']:.2f}% |"
    )


def git_head() -> str:
    """Short SHA of HEAD."""
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                         text=True, check=True)  # fmt: skip
    return out.stdout.strip()


def smoke_visited(
    rows: list[dict[str, Any]], steps: int = 20, accum: int = 2
) -> list[dict[str, Any]]:
    """The pages the 20-step smoke trained on: the first 80 pages, `step_indices` order (the
    smoke stage's ``specs[: steps * accum * 2]`` and epoch-0 permutation)."""
    pool = steps * accum * 2
    seen = [i for s in range(steps) for i in tr.step_indices(pool, accum, 42, s)[1]]
    return [rows[i] for i in seen]


def render_md(rows: list[dict[str, Any]], label: str, prepared_path: Path) -> str:
    """The report."""
    allp = summarize(rows)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[f"{'waybill' if r['waybill'] else 'invoice'} / {r['page_kind']}"].append(r)
    head = (
        "| page type | pages | supervised share of sequence, mean | min | max | supervised tokens "
        "per page, mean (min-max) | supervised tokens per epoch | structure | null | value | stop |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    tab = [head, row("**all train pages**", allp)]
    tab += [row(k, summarize(v)) for k, v in sorted(groups.items())]
    tab.append(row("header-only docs (line_items masked)",
                   summarize([r for r in rows if r["header_only"]])))  # fmt: skip
    smoke = summarize(smoke_visited(rows))
    tab.append(row("the 40 pages the smoke visited (epoch 0)", smoke))
    pv = smoke["value"]
    implied = {k: v / pv for k, v in OBSERVED.items()}
    lines = [
        "# Supervised tokens of the fine-tune training pages (STEP Z2)",
        "",
        f"**Every number here is UNVERIFIED** (computed at commit `{git_head()}`; nothing was trained "
        f"for this report). Tokenizer: **{label}**. Counts only: no page content, no values, no names. "
        f"Reproduce with `uv run python scripts/supervised_share.py` (needs `data/`, `{prepared_path.name}` "
        "and the cached Qwen3.5 tokenizer files; tests: `tests/test_train_masking.py`).",
        "",
        "Sample = `train.encode_page` output for the clean epoch-0 page (no augmentation; an occlusion "
        "changes one header value to `null`, a handful of tokens): inference-identical chat-template "
        "prefix (system/user turn, image placeholder expanded to 1,260 `<|image_pad|>`, "
        "generation prompt + empty think block) + keyed JSON target + `<|im_end|>`. "
        "**Supervised** = `labels != -100` = target JSON tokens + the stop token; the prefix, the "
        "image tokens and the masked `line_items` of header-only documents are `-100`. "
        "Kinds: *structure* = object keys, braces, brackets, colons, commas, separators and the "
        "quotes around values; *null* = a `null` literal; *value* = a token containing any "
        "character of a string/number value; *stop* = `<|im_end|>`. Shares in the last four columns "
        "are shares of the supervised tokens.",
        "",
        *tab,
        "",
        "## What this says about the smoke loss (0.0557 -> 0.0056, L4 bf16)",
        "",
        "- The loss is the mean cross-entropy over the supervised tokens ONLY (`chunked_target_loss`: "
        "numerator and denominator both count labelled positions). Masked prefix tokens neither add "
        "to nor dilute it, so the supervised share of the sequence "
        f"({100 * allp['share_mean']:.1f}% of the tokens) does NOT explain a low loss; what matters is "
        "the composition of the supervised tokens.",
        f"- Composition of the supervised tokens of the 40 pages the smoke visited: "
        f"{100 * (smoke['structure'] + smoke['null'] + smoke['stop']):.1f}% structure / `null` / stop "
        f"and {100 * pv:.1f}% value-bearing (all 536 pages: {100 * (allp['structure'] + allp['null'] + allp['stop']):.1f}% "
        f"and {100 * allp['value']:.1f}%; per-sample mean value share {100 * allp['value_share_sample_mean']:.1f}%).",
        "- Arithmetic only (not a model measurement): if every structure / null / stop token cost "
        f"exactly 0 nats, the observed means {OBSERVED['first 5 steps']} and "
        f"{OBSERVED['last 5 steps']} mean {implied['first 5 steps']:.3f} and "
        f"{implied['last 5 steps']:.3f} nats per value-bearing token (geometric-mean token "
        f"probability {math_exp(-implied['first 5 steps']):.3f} and "
        f"{math_exp(-implied['last 5 steps']):.3f}). If they cost more than 0, the value tokens "
        "average LESS than that. Zero loss on all non-value tokens lowers the mean by at most "
        "their share of the tokens; the remaining mean has to come from the value-bearing tokens.",
        "- All 20 smoke steps are in epoch 0 (`metrics.jsonl`: `epoch` 0 throughout) and every page "
        "is visited once, so each step's loss is computed on pages the model has not yet been "
        "updated on: the curve is a progressive validation over the 40 pages, not memorisation of "
        "a repeated set. Supplier layouts can still recur among those pages (doc-id-sorted prefix), "
        "so it is not a held-out-supplier loss.",
        "",
        "## Not covered",
        "",
        "- Whether the model's per-token losses are actually concentrated on value tokens: this needs a "
        "per-token loss dump from a GPU run (not available); only the composition is reported.",
        "- Augmented epochs: occlusion adds `null` targets; scan degradation does not change the "
        "target.",
    ]
    return "\n".join(lines) + "\n"


def math_exp(x: float) -> float:
    """exp(x) (kept separate so the report text stays readable)."""
    import math

    return math.exp(x)


def main(argv: list[str] | None = None) -> int:
    """Compute the statistics for the 536 final-stage training pages and print / write the report."""
    ap = argparse.ArgumentParser(description=(main.__doc__ or "").strip())
    ap.add_argument(
        "--prepared", type=Path, default=paths.runs_dir() / "trainset" / "prepared_final.json"
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--fake", action="store_true", help="use the fake char-level tokenizer")
    args = ap.parse_args(argv)
    split = ts.stage_split("final", ts.load_folds())
    prepared = ts.load_prepared(args.prepared, list(split.train_ids))
    tok = None if args.fake else load_real_tokenizer()
    if tok is None:
        import sys

        sys.path.insert(0, str(ROOT / "tests"))
        from test_train import IMG_ID, FakeProcessor  # type: ignore[import-not-found]

        shim, image_token_id, label = (
            FakeProcessor(),
            IMG_ID,
            "FAKE (character-level, illustration only)",
        )
    else:
        shim, image_token_id = RealShim(tok), int(tok.convert_tokens_to_ids("<|image_pad|>"))
        label = "real Qwen3.5-4B tokenizer + chat template (revision 851bf6e8, local cache)"
    specs = prepared.page_specs(split.train_ids)
    rows = [sample_stats(sp, prepared, shim, image_token_id) for sp in specs]
    text = render_md(rows, label, args.prepared)
    if args.out:
        args.out.write_text(text, encoding="utf-8", newline="\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
