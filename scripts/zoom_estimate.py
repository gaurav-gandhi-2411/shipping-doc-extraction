"""Extra model calls and T4 seconds of the zoom re-read, as a function of the review flag rate.

EVERY NUMBER HERE IS AN ESTIMATE (UNVERIFIED): no zoom call has been run on a GPU. Inputs:

* the flag rate: ``--flag-rate`` (one or more fractions, default 5/10/20/30%) with
  ``--fields-per-doc``, or a flag-rate source with per-document flagged counts
  (``--flag-source``: a JSON ``{"docs": {doc_id: {"fields": n, "flagged_header": a,
  "flagged_rows": r}}}``, or the OOF fields CSV of ``scripts/calibrate.py`` together with
  ``--tau``; a source whose file name contains ``DRYRUN`` is labelled DRY RUN);
* the speeds of ``configs/spike_speed.json`` (OLS of per-page latency on output tokens measured on
  the spike40 traces: ``prefill_s`` intercept, ``decode_tok_s``).

Cost of one re-read call = prefill of (prompt + crop visual tokens) + decode of a short JSON.
The trace speeds cannot separate prefill from fixed per-call overhead (the config says so), so two
bounds are reported, both ASSUMED:

* ``fixed``  : the whole ``prefill_s`` intercept is paid per call whatever the crop size (upper);
* ``linear`` : prefill scales with input tokens, ``prefill_s / n_input_tokens_mean`` per token,
  with zero fixed overhead (lower).

Crop sizes, prompt length and output lengths are ASSUMED (flags below); one visual token per 32x32
px is `shipdoc.extract`'s own figure (patch 16, merge 2).

Run: uv run python scripts/zoom_estimate.py --fields-per-doc 45
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SPEED_CONFIG = ROOT / "configs" / "spike_speed.json"
PX_PER_VISUAL_TOKEN = 32 * 32  # patch 16 x merge 2 (shipdoc.extract.BackendConfig)
DEFAULT_RATES = (0.05, 0.10, 0.20, 0.30)


@dataclass(frozen=True)
class Assumptions:
    """Every ASSUMED quantity of the estimate (none of them is measured)."""

    header_crop_px: tuple[int, int] = (1240, 128)  # matched line +- one line, full width
    row_crop_px: tuple[int, int] = (1240, 192)  # table headings strip + row band
    prompt_tokens: int = 600  # rules + field definitions; the page prompt is longer
    header_out_tokens: int = 30  # {"header":{"field":"value"}} plus the stop token
    row_out_tokens: int = 70  # one row object with 4 keys
    row_share: float = 0.5  # share of flagged fields that are row fields
    flagged_per_row: float = 1.5  # flagged fields sharing one row call
    pages_per_doc: float = 1.375  # UNVERIFIED: spike40 config n_pages 55 over 40 docs


@dataclass(frozen=True)
class Speeds:
    """Speeds of one model entry of ``configs/spike_speed.json``."""

    model: str
    prefill_s: float
    decode_tok_s: float
    n_input_tokens_mean: float
    s_per_page_mean: float


def load_speeds(model: str, path: Path = SPEED_CONFIG) -> Speeds:
    """Speeds of `model` (e.g. ``qwen35_4b_img_only``); KeyError lists the valid names."""
    models = json.loads(path.read_text(encoding="utf-8"))["models"]
    if model not in models:
        raise KeyError(f"unknown model {model!r}; valid: {sorted(models)}")
    m = models[model]
    return Speeds(
        model,
        float(m["prefill_s"]),
        float(m["decode_tok_s"]),
        float(m["n_input_tokens_mean"]),
        float(m["s_per_page_mean"]),
    )


def crop_tokens(px: tuple[int, int]) -> float:
    """Visual tokens of a crop of ``(w, h)`` px (one per 32x32 px, unrounded)."""
    return px[0] * px[1] / PX_PER_VISUAL_TOKEN


def call_seconds(
    sp: Speeds, crop_px: tuple[int, int], out_tokens: int, a: Assumptions, bound: str
) -> float:
    """Seconds of one re-read call under `bound` (``fixed`` upper, ``linear`` lower)."""
    decode = out_tokens / sp.decode_tok_s
    if bound == "fixed":
        return sp.prefill_s + decode
    if bound == "linear":
        per_token = sp.prefill_s / sp.n_input_tokens_mean
        return per_token * (a.prompt_tokens + crop_tokens(crop_px)) + decode
    raise ValueError(f"bound must be 'fixed' or 'linear', got {bound!r}")


def calls_per_doc(
    flagged_header: float, flagged_row_fields: float, a: Assumptions, flagged_rows: float | None
) -> tuple[float, float]:
    """(header calls, row calls) per doc: one per flagged header field; flagged row fields share
    a call per row (`flagged_rows` when measured, else ``fields / a.flagged_per_row``)."""
    rows = flagged_rows if flagged_rows is not None else flagged_row_fields / a.flagged_per_row
    return flagged_header, rows


def estimate(
    flagged_header: float,
    flagged_row_fields: float,
    sp: Speeds,
    a: Assumptions,
    flagged_rows: float | None = None,
) -> dict[str, float]:
    """Per-document extra calls, extra T4 seconds (lower / upper), pages-equivalent and overhead."""
    h_calls, r_calls = calls_per_doc(flagged_header, flagged_row_fields, a, flagged_rows)
    out: dict[str, float] = {"header_calls": h_calls, "row_calls": r_calls}
    out["calls"] = h_calls + r_calls
    for bound in ("linear", "fixed"):
        s = h_calls * call_seconds(sp, a.header_crop_px, a.header_out_tokens, a, bound)
        s += r_calls * call_seconds(sp, a.row_crop_px, a.row_out_tokens, a, bound)
        out[f"seconds_{bound}"] = s
        out[f"pages_eq_{bound}"] = s / sp.s_per_page_mean
        out[f"overhead_{bound}"] = s / (a.pages_per_doc * sp.s_per_page_mean)
    return out


def sweep(
    rates: Sequence[float], fields_per_doc: float, sp: Speeds, a: Assumptions
) -> list[dict[str, float]]:
    """One estimate per flag rate; flagged fields split by ``a.row_share``."""
    rows = []
    for r in rates:
        flagged = r * fields_per_doc
        rows.append(
            {
                "flag_rate": r,
                "flagged_fields": flagged,
                **estimate(flagged * (1 - a.row_share), flagged * a.row_share, sp, a),
            }
        )
    return rows


def load_flag_source(path: Path, tau: float | None) -> dict[str, float]:
    """Mean per-doc counts from a flag-rate source (JSON docs or the OOF fields CSV + tau).

    CSV rows are scored as ``zoom.scores_from_oof_rows`` does: ``p_correct`` if emitted else
    ``p_null``; flagged = score < tau. Returns fields, flagged fields, header / row-field counts
    and distinct flagged rows per document.
    """
    if path.suffix.lower() == ".csv":
        if tau is None:
            raise ValueError("a CSV flag source needs --tau")
        docs: dict[str, dict[str, Any]] = {}
        with path.open(encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                emitted = r["emitted"].strip() in {"1", "true", "True", "1.0"}
                raw = r["p_correct"] if emitted else r["p_null"]
                try:
                    p = float(raw)
                except ValueError:
                    continue  # unscored: never selected (zoom.select_for_reread skips it too)
                d = docs.setdefault(
                    r["doc_id"], {"fields": 0, "flagged_header": 0, "flagged_rows": set(), "rf": 0}
                )
                d["fields"] += 1
                if p < tau:
                    ridx = int(float(r["row_idx"]))
                    if ridx < 0:
                        d["flagged_header"] += 1
                    else:
                        d["flagged_rows"].add(ridx)
                        d["rf"] += 1
        per = [
            {
                "fields": d["fields"],
                "flagged_header": d["flagged_header"],
                "flagged_rows": len(d["flagged_rows"]),
                "flagged_row_fields": d["rf"],
            }
            for d in docs.values()
        ]
    else:
        raw_docs = json.loads(path.read_text(encoding="utf-8"))["docs"]
        per = [
            {
                "fields": int(d["fields"]),
                "flagged_header": int(d["flagged_header"]),
                "flagged_rows": int(d["flagged_rows"]),
                "flagged_row_fields": int(d.get("flagged_row_fields", d["flagged_rows"])),
            }
            for d in raw_docs.values()
        ]
    if not per:
        raise ValueError(f"no documents in flag source {path}")
    n = len(per)
    return {
        "n_docs": float(n),
        **{k: sum(d[k] for d in per) / n for k in per[0]},
    }


def render(
    rows: list[dict[str, float]],
    sp: Speeds,
    a: Assumptions,
    measured: dict[str, Any] | None,
    fields_per_doc: float,
) -> str:
    """Markdown report (every table labelled ESTIMATE)."""
    lines = [
        f"# Zoom re-read cost: ESTIMATE (UNVERIFIED) | model `{sp.model}`",
        "",
        f"Speeds (configs/spike_speed.json): prefill_s {sp.prefill_s}, decode_tok_s "
        f"{sp.decode_tok_s}, n_input_tokens_mean {sp.n_input_tokens_mean}, s_per_page_mean "
        f"{sp.s_per_page_mean}.",
        "",
        "ASSUMED: " + ", ".join(f"{k}={v}" for k, v in a.__dict__.items()) + ".",
        f"Fields per doc used in the sweep: {fields_per_doc:g}.",
        "`linear` / `fixed` = lower / upper bound of the per-call prefill "
        "(see the script docstring).",
        "",
        "| flag rate | flagged fields/doc | header calls | row calls | extra calls/doc | "
        "extra T4 s/doc (linear..fixed) | pages-equivalent (linear..fixed) | "
        "overhead vs baseline (linear..fixed) |",
        "|---|---|---|---|---|---|---|---|",
    ]

    def row(label: str, r: dict[str, float], flagged: float) -> str:
        return (
            f"| {label} | {flagged:.2f} | {r['header_calls']:.2f} | {r['row_calls']:.2f} | "
            f"{r['calls']:.2f} | {r['seconds_linear']:.1f}..{r['seconds_fixed']:.1f} | "
            f"{r['pages_eq_linear']:.2f}..{r['pages_eq_fixed']:.2f} | "
            f"{100 * r['overhead_linear']:.0f}%..{100 * r['overhead_fixed']:.0f}% |"
        )

    for r in rows:
        lines.append(row(f"{100 * r['flag_rate']:g}% (ESTIMATE)", r, r["flagged_fields"]))
    if measured is not None:
        m = measured
        flagged = m["flagged_header"] + m["flagged_row_fields"]
        est = estimate(m["flagged_header"], m["flagged_row_fields"], sp, a, m["flagged_rows"])
        rate = flagged / m["fields"]
        lines += [
            row(f"{100 * rate:.1f}% ({m['label']}, ESTIMATE of cost)", est, flagged),
            "",
            f"{m['label']}: flag counts from `{m['path']}` at tau={m['tau']}: "
            f"{m['n_docs']:.0f} docs, "
            f"{m['fields']:.2f} fields/doc, {flagged:.2f} flagged/doc, "
            f"{m['flagged_rows']:.2f} distinct flagged rows/doc. The flag rate is "
            f"{'a DRY RUN value, not a result' if m['label'] == 'DRY RUN' else 'as labelled'}.",
        ]
    lines += [
        "",
        f"Baseline = {a.pages_per_doc} pages/doc x s_per_page_mean (the pages/doc is UNVERIFIED).",
    ]
    return "\n".join(lines) + "\n"


def parse_px(text: str) -> tuple[int, int]:
    """``"1240x128"`` -> (1240, 128)."""
    w, h = text.lower().split("x")
    return int(w), int(h)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="qwen35_4b_img_only")
    ap.add_argument("--flag-rate", type=float, nargs="+", default=list(DEFAULT_RATES))
    ap.add_argument("--fields-per-doc", type=float, default=None)
    ap.add_argument("--flag-source", type=Path, default=None)
    ap.add_argument("--tau", type=float, default=None, help="for a CSV flag source")
    ap.add_argument("--label", default=None, help="label of the flag source (default by name)")
    ap.add_argument("--speed-config", type=Path, default=SPEED_CONFIG)
    ap.add_argument("--header-crop-px", type=parse_px, default=Assumptions.header_crop_px)
    ap.add_argument("--row-crop-px", type=parse_px, default=Assumptions.row_crop_px)
    ap.add_argument("--prompt-tokens", type=int, default=Assumptions.prompt_tokens)
    ap.add_argument("--row-share", type=float, default=Assumptions.row_share)
    ap.add_argument("--flagged-per-row", type=float, default=Assumptions.flagged_per_row)
    ap.add_argument("--pages-per-doc", type=float, default=Assumptions.pages_per_doc)
    ap.add_argument("--out", type=Path, default=None, help="also write the markdown here")
    args = ap.parse_args(argv)
    a = Assumptions(
        header_crop_px=args.header_crop_px,
        row_crop_px=args.row_crop_px,
        prompt_tokens=args.prompt_tokens,
        row_share=args.row_share,
        flagged_per_row=args.flagged_per_row,
        pages_per_doc=args.pages_per_doc,
    )
    sp = load_speeds(args.model, args.speed_config)
    measured: dict[str, Any] | None = None
    fields = args.fields_per_doc
    if args.flag_source is not None:
        label = args.label or ("DRY RUN" if "DRYRUN" in args.flag_source.name else "UNLABELLED")
        measured = {
            **load_flag_source(args.flag_source, args.tau),
            "label": label,
            "path": args.flag_source.as_posix(),
            "tau": args.tau,
        }
        fields = fields if fields is not None else measured["fields"]
    if fields is None:
        ap.error("give --fields-per-doc or --flag-source")
    text = render(sweep(args.flag_rate, fields, sp, a), sp, a, measured, fields)
    sys.stdout.write(text)
    if args.out is not None:
        args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
