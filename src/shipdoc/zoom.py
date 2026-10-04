"""Zoom re-read of low-confidence fields (spec Phase 5 extension; design in docs/zoom_reread.md).

For a field whose calibrated P(correct) is below tau, crop the region of the page image at its
native resolution, ask the SAME backend for only that field (or that table row) under a tiny
constrained schema, and keep the re-read only if it is MORE confident than the original on the same
scale. Everything here is opt-in: `ZoomConfig.enabled` defaults to False and `zoom_document` then
returns the prediction unchanged without touching the backend, the images or the OCR index.

Label-free by construction: no function takes labels or gold, nothing reads ``data/*/labels``.
Crop regions come from the OCR (`shipdoc.locate`, `shipdoc.layout`), never from gold boxes.

Scope of the interface to the calibrator (written by another step): `FieldScore` and
`select_for_reread`. `scores_from_oof_rows` is the adapter from ``scripts/calibrate.py``'s OOF
fields CSV (columns ``doc_id, scope, field, row_idx, emitted, p_correct, p_null``).

Design choices flagged for review:

* null -> value is allowed only when the re-read passes the same confidence rule AND the value has
  OCR support (`ZoomConfig.null_fill`, default ``"ocr_supported"``). The spec rule "never fill a
  value the model did not emit" is read as being about the model's own emission; the re-read IS a
  model emission, but a null that the page genuinely lacks is the main false-fill source, so the
  extra OCR evidence is required.
* value -> null is ignored by default (`allow_value_to_null=False`): the calibrator was fitted on
  non-null emissions and does not score a null re-read on the same footing.
* Only fields that were flagged are ever replaced. A row re-read returns all four row keys, but the
  unflagged ones are discarded.
"""

from __future__ import annotations

import copy
import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from shipdoc.extract import NULLABLE_STRING, NUMERIC_KEYS, NUMERIC_VALUE, ROW_KEYS, parse_page_json
from shipdoc.layout import analyze_page
from shipdoc.logprobs import field_logprobs, lookup
from shipdoc.ocr import PageOcr, page_items
from shipdoc.prompts import (
    CONVENTION_RULES,
    INVOICE_HEADER,
    NULL_RULE,
    PROVENANCE_RULE,
    ROW_FIELDS,
    RULES,
    WAYBILL_HEADER,
)

ZOOM_VERSION = "zoom-v1"
Box = tuple[float, float, float, float]
NullFill = Literal["never", "ocr_supported", "any"]
NULL_FILL_MODES: tuple[str, ...] = ("never", "ocr_supported", "any")
RowBandWidth = Literal["page", "table"]
#: The Qwen3.5 / Qwen3-VL processors spend one visual token per 32x32 px (patch 16, merge 2; see
#: `shipdoc.extract.BackendConfig`); this is also the default per-crop pixel cap, the same cap the
#: full page gets, so a crop is never downscaled by us unless it is as large as a capped page.
DEFAULT_MAX_PIXELS = 1280 * 32 * 32
#: Header keys in canonical (declared) order, used for deterministic task ordering.
_HEADER_ORDER: dict[str, int] = {k: i for i, (k, _) in enumerate(INVOICE_HEADER + WAYBILL_HEADER)}
_FIELD_DEFS: dict[str, str] = dict(INVOICE_HEADER + WAYBILL_HEADER + ROW_FIELDS)
_FOOTER_FIELDS = frozenset({"total_amount", "pieces", "gross_weight_kg"})

# --------------------------------------------------------------------------------------------
# Prompt text (new constants; prompts.py is untouched). Rules are the page prompt's, verbatim.
# --------------------------------------------------------------------------------------------

ZOOM_RULES_TEXT = "\n".join(
    f"- {r}" for r in (*RULES, NULL_RULE, PROVENANCE_RULE, *CONVENTION_RULES)
)
ZOOM_CROP_RULE = (
    "The image is a crop of one region of a page, cut at the page's original resolution. Read only "
    "this crop. Do not use anything you may remember about the rest of the page. If the value is "
    "not fully visible in the crop, output null."
)
ZOOM_HEADER_PROMPT = (
    "You read one cropped region of a shipping document page (a commercial invoice or an air "
    "waybill) and return JSON.\n\n"
    f"Rules:\n{ZOOM_RULES_TEXT}\n\n"
    f"{ZOOM_CROP_RULE}\n\n"
    'Output exactly one key, "header", holding exactly one key:\n'
    "- {field}: {definition}"
)
ZOOM_ROW_PROMPT = (
    "You read one cropped region of a shipping document page (a commercial invoice) holding ONE "
    "table row, and return JSON.\n\n"
    f"Rules:\n{ZOOM_RULES_TEXT}\n\n"
    f"{ZOOM_CROP_RULE}\n\n"
    'Output exactly one key, "line_items", an array with exactly one object, the row, with keys:\n'
    + "\n".join(f"- {k}: {d}" for k, d in ROW_FIELDS)
    + "\n\n{row_hint}"
)
ZOOM_ROW_IDENTITY_HINT = (
    "The row to read is the one whose supplier_part_number is {spn}. Other rows may be partly "
    "visible at the edges of the crop; ignore them."
)
ZOOM_ROW_CENTRE_HINT = (
    "The row to read is the one in the vertical middle of the crop. Other rows may be partly "
    "visible at the edges of the crop; ignore them."
)
ZOOM_HEADING_STRIP_NOTE = (
    " The top strip of the crop is the table's column headings, copied above the row."
)


def zoom_prompt_hash() -> str:
    """sha256 of every static zoom prompt string (recorded with the zoom config)."""
    import hashlib

    text = "\n".join(
        (
            ZOOM_VERSION,
            ZOOM_HEADER_PROMPT,
            ZOOM_ROW_PROMPT,
            ZOOM_ROW_IDENTITY_HINT,
            ZOOM_ROW_CENTRE_HINT,
            ZOOM_HEADING_STRIP_NOTE,
        )
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def header_prompt(field_name: str) -> str:
    """Zoom prompt for one header field."""
    return ZOOM_HEADER_PROMPT.format(field=field_name, definition=_FIELD_DEFS[field_name])


def row_prompt(spn: str | None, with_heading_strip: bool) -> str:
    """Zoom prompt for one table row; `spn` is an UNFLAGGED supplier_part_number (identity hint)."""
    hint = ZOOM_ROW_IDENTITY_HINT.format(spn=spn) if spn else ZOOM_ROW_CENTRE_HINT
    if with_heading_strip:
        hint += ZOOM_HEADING_STRIP_NOTE
    return ZOOM_ROW_PROMPT.format(row_hint=hint)


def _value_schema(key: str) -> dict[str, Any]:
    return dict(NUMERIC_VALUE if key in NUMERIC_KEYS else NULLABLE_STRING)


def header_zoom_schema(field_name: str) -> dict[str, Any]:
    """Schema of a one-field header re-read: ``{"header": {<field>: value}}`` (strict)."""
    inner = {
        "type": "object",
        "additionalProperties": False,
        "required": [field_name],
        "properties": {field_name: _value_schema(field_name)},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["header"],
        "properties": {"header": inner},
    }


def row_zoom_schema() -> dict[str, Any]:
    """Schema of a one-row re-read: ``{"line_items": [ {4 row keys} ]}`` (exactly one object)."""
    row = {
        "type": "object",
        "additionalProperties": False,
        "required": list(ROW_KEYS),
        "properties": {k: _value_schema(k) for k in ROW_KEYS},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["line_items"],
        "properties": {"line_items": {"type": "array", "items": row, "minItems": 1, "maxItems": 1}},
    }


# --------------------------------------------------------------------------------------------
# Interface to the calibrator
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldScore:
    """One emitted-or-null field with its calibrated confidence (the calibrator's output).

    `page` is the 0-based page of the document the field was read from. `row_index` indexes the
    document-level ``line_items`` list of the prediction handed to `zoom_document` (None for a
    header field). `p_correct` is P(the current value is right): for a null field the calibrator's
    P(gold is null). `logprob_conf` is the raw MINIMUM token logprob of the value (<= 0, higher is
    more confident; the ``min`` of `shipdoc.logprobs.field_logprobs`), None when unavailable.
    """

    doc_id: str
    page: int
    field: str
    row_index: int | None
    p_correct: float
    value: str | None
    logprob_conf: float | None = None

    @property
    def scope(self) -> str:
        """``header`` or ``row``."""
        return "header" if self.row_index is None else "row"


def _finite(x: float | None) -> bool:
    return x is not None and math.isfinite(x)


def _canon_key(s: FieldScore) -> tuple[str, int, int, int, int]:
    order = _HEADER_ORDER.get(s.field, ROW_KEYS.index(s.field) if s.field in ROW_KEYS else 99)
    return (s.doc_id, s.page, 0 if s.row_index is None else 1, s.row_index or -1, order)


def select_for_reread(scores: Iterable[FieldScore], tau: float) -> list[FieldScore]:
    """Fields with ``p_correct < tau`` (strict; the review flag of spec Phase 5.4), canonical order.

    A field whose p_correct is NaN / None is not selected (it could never pass the acceptance rule
    either); `count_unscored` reports how many there were.
    """
    return sorted((s for s in scores if _finite(s.p_correct) and s.p_correct < tau), key=_canon_key)


def count_unscored(scores: Iterable[FieldScore]) -> int:
    """Number of scores whose p_correct is not a finite number."""
    return sum(1 for s in scores if not _finite(s.p_correct))


def scores_from_oof_rows(
    rows: Iterable[Mapping[str, Any]],
    resolve: Callable[[str, str, str, int], tuple[str | None, int, float | None]],
) -> list[FieldScore]:
    """Adapter from ``scripts/calibrate.py``'s OOF fields rows to `FieldScore`.

    Each row is a mapping with ``doc_id, scope, field, row_idx (-1 for header), emitted, p_correct,
    p_null`` (a csv.DictReader row works: values may be strings). The score is ``p_correct`` for an
    emitted field and ``p_null`` (P(null is right)) for a null one. `resolve(doc_id, scope, field,
    row_idx)` returns ``(value, page, logprob_min)`` from the prediction + its logprob trace; the
    OOF CSV carries none of the three.
    """
    out: list[FieldScore] = []
    for r in rows:
        emitted = str(r["emitted"]).strip().lower() in {"1", "true", "1.0"}
        raw_p = r["p_correct"] if emitted else r["p_null"]
        try:
            p = float(raw_p)
        except (TypeError, ValueError):
            p = math.nan
        ridx = int(float(r["row_idx"]))
        value, page, lp = resolve(str(r["doc_id"]), str(r["scope"]), str(r["field"]), ridx)
        out.append(
            FieldScore(
                str(r["doc_id"]), page, str(r["field"]), None if ridx < 0 else ridx, p, value, lp
            )
        )
    return out


# --------------------------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------------------------


def union_boxes(boxes: Sequence[Box]) -> Box:
    """Smallest box containing all `boxes` (ValueError when empty)."""
    if not boxes:
        raise ValueError("union_boxes needs at least one box")
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def pad_box(
    box: Box,
    pad_x_frac: float,
    pad_y_frac: float,
    page_w: float,
    page_h: float,
    min_w: float = 0.0,
    min_h: float = 0.0,
) -> Box:
    """Grow `box` by a fraction of its own size on each side, enforce a minimum size around its
    centre, then clip to ``[0, page_w] x [0, page_h]`` (clipping never shifts the box inwards, so a
    box at the page edge just gets less padding)."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    x0, x1 = x0 - w * pad_x_frac, x1 + w * pad_x_frac
    y0, y1 = y0 - h * pad_y_frac, y1 + h * pad_y_frac
    if x1 - x0 < min_w:
        c = (x0 + x1) / 2
        x0, x1 = c - min_w / 2, c + min_w / 2
    if y1 - y0 < min_h:
        c = (y0 + y1) / 2
        y0, y1 = c - min_h / 2, c + min_h / 2
    return (max(0.0, x0), max(0.0, y0), min(page_w, x1), min(page_h, y1))


def line_box(page: PageOcr, line_idx: int) -> Box | None:
    """Union of the OCR items of reading-order line `line_idx` (None if the line is absent)."""
    boxes = [it.box for it in page_items(page) if it.line_idx == line_idx]
    return union_boxes(boxes) if boxes else None


def table_x_extent(page: PageOcr) -> tuple[float, float]:
    """Horizontal extent of the table lines (`layout.analyze_page`); the page width if none."""
    layout = analyze_page(page)
    xs = [
        it.box
        for it in page_items(page)
        if layout.region(it.line_idx) == "table" and it.line_idx != layout.header_line
    ]
    if not xs:
        return (0.0, float(page.width))
    u = union_boxes(xs)
    return (u[0], u[2])


def region_box(page: PageOcr, region: str) -> Box | None:
    """Union box of every OCR line of `layout.analyze_page` region `region` (None if empty)."""
    layout = analyze_page(page)
    boxes = [it.box for it in page_items(page) if layout.region(it.line_idx) == region]
    return union_boxes(boxes) if boxes else None


@dataclass(frozen=True)
class LineHit:
    """An OCR line a value was located on: its page, reading-order line and the matched box."""

    page: int
    line_idx: int
    box: Box


@dataclass(frozen=True)
class CropPlan:
    """What to cut from one page image. `boxes` are in OCR page pixels, already padded and
    clipped; two boxes (table headings strip, then the row band) are stacked vertically."""

    page: int
    boxes: tuple[Box, ...]
    source: str  # "ocr_match" | "region_fallback" | "row_line"
    heading_strip: bool = False


@dataclass(frozen=True)
class ZoomConfig:
    """All zoom parameters. Frozen on non-evaluated folds before the held-out gate."""

    enabled: bool = False  # default OFF: the production path is unchanged unless this is True
    tau: float = 0.0  # select p_correct < tau; 0.0 selects nothing
    null_fill: NullFill = "ocr_supported"
    accept_margin: float = 0.0  # re-read must beat the original by MORE than this
    allow_value_to_null: bool = False
    max_calls_per_doc: int | None = None  # None = unlimited; else lowest-confidence tasks first
    header_pad_x_frac: float = 0.10  # padding of the matched line, fraction of its width
    header_pad_y_frac: float = 1.0  # ... fraction of its height (one line above and below)
    row_pad_x_frac: float = 0.02
    row_pad_y_frac: float = 0.5
    row_band_width: RowBandWidth = "table"
    include_table_header: bool = True
    min_crop_w: float = 256.0  # px; tiny crops are grown around their centre
    min_crop_h: float = 64.0
    max_pixels: int | None = DEFAULT_MAX_PIXELS  # None = never resize in zoom.py

    def __post_init__(self) -> None:
        if self.null_fill not in NULL_FILL_MODES:
            raise ValueError(f"null_fill must be one of {NULL_FILL_MODES}, got {self.null_fill!r}")
        if self.row_band_width not in ("page", "table"):
            raise ValueError("row_band_width must be 'page' or 'table'")


def plan_header_crop(
    pages: Sequence[PageOcr], hit: LineHit | None, page_idx: int, field_name: str, cfg: ZoomConfig
) -> CropPlan | None:
    """Crop plan of a header field: its matched OCR line (label context included) padded, else the
    layout region (``footer`` for totals, ``header`` otherwise) of the field's page, else None."""
    if hit is not None:
        page = pages[hit.page]
        base = line_box(page, hit.line_idx) or hit.box
        box = pad_box(
            base, cfg.header_pad_x_frac, cfg.header_pad_y_frac, page.width, page.height,
            cfg.min_crop_w, cfg.min_crop_h,
        )  # fmt: skip
        return CropPlan(hit.page, (box,), "ocr_match")
    page = pages[page_idx]
    region = region_box(page, "footer" if field_name in _FOOTER_FIELDS else "header")
    if region is None:
        return None
    box = pad_box(region, 0.0, 0.05, page.width, page.height, cfg.min_crop_w, cfg.min_crop_h)
    return CropPlan(page_idx, (box,), "region_fallback")


def plan_row_crop(
    pages: Sequence[PageOcr], hit: LineHit | None, cfg: ZoomConfig
) -> CropPlan | None:
    """Crop plan of a table row: the union of the OCR items of the row's line, widened to the page
    or table width, padded vertically; plus the table's heading line stacked on top if enabled."""
    if hit is None:
        return None
    page = pages[hit.page]
    band = line_box(page, hit.line_idx) or hit.box
    x0, x1 = (0.0, float(page.width)) if cfg.row_band_width == "page" else table_x_extent(page)
    band = (min(x0, band[0]), band[1], max(x1, band[2]), band[3])
    box = pad_box(
        band, cfg.row_pad_x_frac, cfg.row_pad_y_frac, page.width, page.height,
        cfg.min_crop_w, cfg.min_crop_h,
    )  # fmt: skip
    boxes = [box]
    strip = False
    layout = analyze_page(page)
    if cfg.include_table_header and layout.header_line is not None:
        hb = line_box(page, layout.header_line)
        if hb is not None and hb[3] <= box[1]:  # only when the headings sit above the row crop
            hbox = pad_box(
                (box[0], hb[1], box[2], hb[3]), 0.0, cfg.row_pad_y_frac, page.width, page.height,
                0.0, cfg.min_crop_h,
            )  # fmt: skip
            boxes = [(hbox[0], hbox[1], hbox[2], min(hbox[3], box[1])), box]
            strip = True
    return CropPlan(hit.page, tuple(boxes), "row_line", strip)


def render_crop(
    image: Any, plan: CropPlan, page_w: float, page_h: float, max_pixels: int | None
) -> Any:
    """Cut `plan` from `image` (PIL) at native resolution; boxes are scaled if the image size
    differs from the OCR page size. Stacks multiple boxes vertically on a white canvas. Only if the
    result exceeds `max_pixels` is it downscaled uniformly (LANCZOS) to that area; None = never."""
    from PIL import Image

    sx, sy = image.width / page_w, image.height / page_h
    parts = []
    for x0, y0, x1, y1 in plan.boxes:
        px = (
            max(0, math.floor(x0 * sx)),
            max(0, math.floor(y0 * sy)),
            min(image.width, math.ceil(x1 * sx)),
            min(image.height, math.ceil(y1 * sy)),
        )
        parts.append(image.crop(px))
    if len(parts) == 1:
        out = parts[0]
    else:
        out = Image.new("RGB", (max(p.width for p in parts), sum(p.height for p in parts)), "white")
        y = 0
        for p in parts:
            out.paste(p.convert("RGB"), (0, y))
            y += p.height
    if max_pixels is not None and out.width * out.height > max_pixels:
        k = math.sqrt(max_pixels / (out.width * out.height))
        out = out.resize((max(1, int(out.width * k)), max(1, int(out.height * k))), Image.LANCZOS)
    return out


# --------------------------------------------------------------------------------------------
# OCR locator (label-free)
# --------------------------------------------------------------------------------------------


class Locator(Protocol):
    """Where a prediction sits on the OCR pages (`OcrLocator` is real; tests inject fakes)."""

    def header_line(self, field_name: str, value: str | None, page: int) -> LineHit | None:
        """Best OCR line of the emitted `value` on `page` (None for null / not found)."""
        ...

    def row_line(self, rows: Sequence[dict[str, Any]], row_index: int) -> LineHit | None:
        """OCR line of predicted row `row_index` (None if it has no line of its own)."""
        ...

    def supported(self, field_name: str, value: str) -> bool:
        """True if `value` is found anywhere in the document's OCR (fuzzy threshold of locate)."""
        ...


class OcrLocator:
    """`shipdoc.locate` over a document's pages (needs the official scorer, like locate does)."""

    def __init__(self, pages: Sequence[PageOcr]) -> None:
        from shipdoc import locate as L

        self._L = L
        self.pages = list(pages)
        self.index = L.build_index(self.pages)
        self._assign: list[Any] | None = None

    def header_line(self, field_name: str, value: str | None, page: int) -> LineHit | None:
        """First `find_matches` hit on `page` (exact < normalized < fuzzy, then reading order)."""
        if value is None or not str(value).strip():
            return None
        for m in self._L.find_matches(value, field_name, self.index):
            if m.page == page:
                return LineHit(m.page, m.line_idx, m.box)
        return None

    def row_line(self, rows: Sequence[dict[str, Any]], row_index: int) -> LineHit | None:
        """Line from `locate.assign_rows` over ALL predicted rows (distinct lines per row)."""
        if self._assign is None:
            self._assign = self._L.assign_rows(list(rows), self.index)
        a = self._assign[row_index]
        if a.match is None:
            return None
        return LineHit(a.match.page, a.match.line_idx, a.match.box)

    def supported(self, field_name: str, value: str) -> bool:
        """Any fuzzy-or-better OCR match for `value` anywhere in the document."""
        return bool(self._L.find_matches(value, field_name, self.index))


# --------------------------------------------------------------------------------------------
# Re-read
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Reread:
    """One field as re-read from a crop, with the same logprob statistics as the original."""

    field: str
    value: str | None
    lp_min: float | None  # min token logprob of the value (raw), same definition as FieldScore
    lp_mean: float | None
    n_tokens: int


@dataclass
class ReadOutcome:
    """Result of one backend call: re-reads per field (empty if unparsable) and cost accounting."""

    rereads: dict[str, Reread]
    status: str  # "ok" | "unparsable" | "identity_mismatch" | "no_logprobs"
    latency_s: float = 0.0
    n_input_tokens: int = 0
    n_output_tokens: int = 0


def _blank(v: Any) -> bool:
    return v is None or not str(v).strip()


def _text(v: Any) -> str | None:
    return None if _blank(v) else str(v)


def _norm(v: Any) -> str:
    return "".join(ch for ch in str(v) if ch.isalnum()).upper()


def _rereads_from(
    raw: str, meta: Mapping[str, Any], scope: str, fields: Sequence[str]
) -> dict[str, Reread] | None:
    """Parse the keyed zoom output and attach logprob stats; None when the JSON is unusable."""
    obj = parse_page_json(raw)
    if obj is None:
        return None
    try:
        if scope == "header":
            values = {f: obj["header"][f] for f in fields}
            row_idx: int | None = None
        else:
            values = {f: obj["line_items"][0][f] for f in fields}
            row_idx = 0
    except (KeyError, IndexError, TypeError):
        return None
    trace = meta.get("logprob_trace")
    entries: dict[Any, dict[str, Any]] = {}
    if trace:
        entries = lookup(field_logprobs(raw, trace["ends"], trace["lp"], trace.get("lp_c")))
    out: dict[str, Reread] = {}
    for f, v in values.items():
        e = entries.get((scope, row_idx, f))
        out[f] = Reread(
            f,
            _text(v),
            None if e is None else e["min"],
            None if e is None else e["mean"],
            0 if e is None else int(e["n_tokens"]),
        )
    return out


class ZoomReader:
    """Issues the constrained re-read calls on the SAME backend (one call per task)."""

    def __init__(self, backend: Any, cfg: ZoomConfig) -> None:
        fmt = getattr(
            getattr(backend, "cfg", None),
            "output_format",
            getattr(backend, "output_format", "json"),
        )
        if fmt != "json":
            raise ValueError(
                f"zoom needs the keyed 'json' output format (got {fmt!r}): the logprob statistics "
                "and the one-field schemas are keyed"
            )
        self.backend, self.cfg = backend, cfg
        self.calls = 0

    def _call(
        self, doc_id: str, page: int, n_pages: int, image: Any, prompt: str, schema: Any
    ) -> tuple[str, dict[str, Any] | None, dict[str, Any]]:
        if hasattr(self.backend, "set_context"):  # the mock needs it; mirrors spike.py
            self.backend.set_context(doc_id, page, n_pages)
        self.calls += 1
        return self.backend.extract_page(image, prompt, schema, None)  # no OCR hint: crop-only

    def read_header(
        self, doc_id: str, page: int, n_pages: int, image: Any, field_name: str
    ) -> ReadOutcome:
        """Re-read one header field from `image`."""
        raw, _parsed, meta = self._call(
            doc_id, page, n_pages, image, header_prompt(field_name), header_zoom_schema(field_name)
        )
        return self._outcome(raw, meta, "header", (field_name,))

    def read_row(
        self,
        doc_id: str,
        page: int,
        n_pages: int,
        image: Any,
        spn_hint: str | None,
        strip: bool,
    ) -> ReadOutcome:
        """Re-read one table row (all 4 keys) from `image`; the caller keeps only flagged fields."""
        raw, _parsed, meta = self._call(
            doc_id, page, n_pages, image, row_prompt(spn_hint, strip), row_zoom_schema()
        )
        out = self._outcome(raw, meta, "row", ROW_KEYS)
        got = out.rereads["supplier_part_number"].value if out.rereads else None
        if out.status == "ok" and spn_hint and _norm(got or "") != _norm(spn_hint):
            out.status = "identity_mismatch"  # it read another row: discard the whole re-read
        return out

    @staticmethod
    def _outcome(
        raw: str, meta: Mapping[str, Any], scope: str, fields: Sequence[str]
    ) -> ReadOutcome:
        rr = _rereads_from(raw, meta, scope, fields)
        base = {
            "latency_s": float(meta.get("latency_s") or 0.0),
            "n_input_tokens": int(meta.get("n_input_tokens") or 0),
            "n_output_tokens": int(meta.get("n_output_tokens") or 0),
        }
        if rr is None:
            return ReadOutcome({}, "unparsable", **base)
        if all(r.lp_min is None for r in rr.values()):
            return ReadOutcome(rr, "no_logprobs", **base)
        return ReadOutcome(rr, "ok", **base)


# --------------------------------------------------------------------------------------------
# Acceptance
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfidenceModel:
    """How the original and the re-read are put on ONE scale.

    `original(score)` and `reread(score, reread)` must be comparable numbers (higher = more
    likely correct). The final gate must use `calibrated`: the same calibrator for both sides.
    `LOGPROB_MIN` (raw min token logprob on both sides) is the stand-in until it exists.
    """

    name: str
    original: Callable[[FieldScore], float | None]
    reread: Callable[[FieldScore, Reread], float | None]


LOGPROB_MIN = ConfidenceModel("logprob_min", lambda s: s.logprob_conf, lambda _s, r: r.lp_min)


def calibrated(predict: Callable[[FieldScore, Reread], float | None]) -> ConfidenceModel:
    """Calibrated comparison: the original side is the calibrator's `p_correct` for the original;
    `predict(score, reread)` applies the SAME calibrator to the re-read's features (lp_min /
    lp_mean from `Reread`, OCR / validator / cross-page / novelty features from the original
    field's row, recomputed for the re-read value)."""
    return ConfidenceModel("calibrated", lambda s: s.p_correct, predict)


@dataclass(frozen=True)
class FieldDecision:
    """Outcome for one flagged field. `action` is ``replace`` or ``keep``; `reason` says why."""

    doc_id: str
    page: int
    field: str
    row_index: int | None
    action: str
    reason: str
    old_value: str | None
    new_value: str | None
    old_conf: float | None = None
    new_conf: float | None = None


def decide(
    score: FieldScore,
    reread: Reread | None,
    model: ConfidenceModel,
    cfg: ZoomConfig,
    supported: Callable[[str, str], bool],
) -> FieldDecision:
    """Accept-or-keep rule for one field (see module docstring). Pure; fails closed."""

    def out(
        action: str,
        reason: str,
        new: str | None = None,
        oc: float | None = None,
        nc: float | None = None,
    ) -> FieldDecision:
        return FieldDecision(
            score.doc_id, score.page, score.field, score.row_index, action, reason,
            score.value, new, oc, nc,
        )  # fmt: skip

    if reread is None:
        return out("keep", "reread_failed")
    old_blank, new_blank = _blank(score.value), reread.value is None
    if new_blank and old_blank:
        return out("keep", "both_null")
    if new_blank and not cfg.allow_value_to_null:
        return out("keep", "reread_null_ignored")
    if not new_blank and not old_blank and _norm(reread.value) == _norm(score.value):
        return out("keep", "agree")
    oc, nc = model.original(score), model.reread(score, reread)
    if not (_finite(oc) and _finite(nc)):
        return out("keep", "no_confidence", oc=oc, nc=nc)
    if nc <= oc + cfg.accept_margin:  # type: ignore[operator]
        return out("keep", "not_more_confident", oc=oc, nc=nc)
    if old_blank:  # null -> value: the explicit design choice
        if cfg.null_fill == "never":
            return out("keep", "null_fill_disabled", oc=oc, nc=nc)
        if cfg.null_fill == "ocr_supported" and not supported(score.field, reread.value or ""):
            return out("keep", "no_ocr_support", oc=oc, nc=nc)
    return out("replace", "more_confident", reread.value, oc, nc)


def apply_decisions(pred: Mapping[str, Any], decisions: Iterable[FieldDecision]) -> dict[str, Any]:
    """Deep copy of `pred` with every ``replace`` decision applied (input never mutated)."""
    out = copy.deepcopy(dict(pred))
    for d in decisions:
        if d.action != "replace":
            continue
        if d.row_index is None:
            out["header"][d.field] = d.new_value
        else:
            if not 0 <= d.row_index < len(out["line_items"]):
                raise ValueError(f"row_index {d.row_index} out of range for {d.doc_id}")
            out["line_items"][d.row_index][d.field] = d.new_value
    return out


# --------------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------------


@dataclass
class ZoomResult:
    """`pred` after the zoom pass plus accounting (calls, decisions, skips)."""

    pred: dict[str, Any]
    decisions: list[FieldDecision] = field(default_factory=list)
    n_calls: int = 0
    n_header_calls: int = 0
    n_row_calls: int = 0
    n_selected_fields: int = 0
    n_unscored: int = 0
    skipped: Counter[str] = field(default_factory=Counter)  # tasks never sent (no crop region...)
    latency_s: float = 0.0
    n_input_tokens: int = 0
    n_output_tokens: int = 0

    @property
    def n_replaced(self) -> int:
        """Fields actually changed."""
        return sum(1 for d in self.decisions if d.action == "replace")


@dataclass
class _Task:
    scope: str
    page: int
    row_index: int | None
    scores: list[FieldScore]

    @property
    def priority(self) -> tuple[float, tuple[str, int, int, int, int]]:
        """Lowest member p_correct first, ties in canonical order."""
        return (min(s.p_correct for s in self.scores), _canon_key(self.scores[0]))


def plan_tasks(selected: Sequence[FieldScore]) -> list[_Task]:
    """Group flagged fields into backend calls: one per header field, one per flagged row (all
    its flagged fields share the call). Order: lowest p_correct first, ties canonical."""
    tasks: list[_Task] = []
    rows: dict[int, _Task] = {}
    for s in selected:
        if s.row_index is None:
            tasks.append(_Task("header", s.page, None, [s]))
        elif s.row_index in rows:
            rows[s.row_index].scores.append(s)
        else:
            rows[s.row_index] = _Task("row", s.page, s.row_index, [s])
            tasks.append(rows[s.row_index])
    return sorted(tasks, key=lambda t: t.priority)


def zoom_document(
    doc_id: str,
    pred: Mapping[str, Any],
    scores: Sequence[FieldScore],
    cfg: ZoomConfig,
    *,
    backend: Any,
    ocr_pages: Sequence[PageOcr],
    image_for_page: Callable[[int], Any],
    confidence: ConfidenceModel = LOGPROB_MIN,
    locator: Locator | None = None,
) -> ZoomResult:
    """Zoom re-read of one document's flagged fields; returns the updated prediction.

    `pred` is the document-level prediction (``header`` dict, ``line_items`` list of dicts);
    `scores` its `FieldScore`s (other documents' scores are ignored); `image_for_page(i)` returns
    the ORIGINAL page image (PIL) of 0-based page `i`. With ``cfg.enabled`` False nothing runs.
    """
    if not cfg.enabled:
        return ZoomResult(pred=copy.deepcopy(dict(pred)))
    mine = [s for s in scores if s.doc_id == doc_id]
    res = ZoomResult(pred=copy.deepcopy(dict(pred)), n_unscored=count_unscored(mine))
    selected = select_for_reread(mine, cfg.tau)
    res.n_selected_fields = len(selected)
    if not selected:
        return res
    loc: Locator = locator or OcrLocator(ocr_pages)
    reader = ZoomReader(backend, cfg)
    rows = [r if isinstance(r, dict) else {} for r in pred.get("line_items", [])]
    by_row_spn = {
        s.row_index: s
        for s in mine
        if s.row_index is not None and s.field == "supplier_part_number"
    }
    tasks = plan_tasks(selected)
    if cfg.max_calls_per_doc is not None:
        for t in tasks[cfg.max_calls_per_doc :]:
            res.skipped["call_cap"] += 1
            res.decisions += [_keep(s, "call_cap") for s in t.scores]
        tasks = tasks[: cfg.max_calls_per_doc]
    images: dict[int, Any] = {}
    decisions: list[FieldDecision] = []
    for t in sorted(tasks, key=lambda t: _canon_key(t.scores[0])):  # canonical call order
        if t.scope == "header":
            s0 = t.scores[0]
            hit = loc.header_line(s0.field, s0.value, s0.page)
            plan = plan_header_crop(ocr_pages, hit, s0.page, s0.field, cfg)
        else:
            hit = loc.row_line(rows, t.row_index or 0)
            plan = plan_row_crop(ocr_pages, hit, cfg)
        if plan is None:
            reason = "no_region" if t.scope == "header" else "row_not_located"
            res.skipped[reason] += 1
            decisions += [_keep(s, reason) for s in t.scores]
            continue
        if plan.page not in images:
            images[plan.page] = image_for_page(plan.page)
        pg = ocr_pages[plan.page]
        crop = render_crop(images[plan.page], plan, pg.width, pg.height, cfg.max_pixels)
        if t.scope == "header":
            out = reader.read_header(doc_id, plan.page, len(ocr_pages), crop, t.scores[0].field)
            res.n_header_calls += 1
        else:
            spn = by_row_spn.get(t.row_index)
            # identity hint only from an UNFLAGGED part number: a flagged one would anchor the model
            trusted = spn is not None and not _blank(spn.value) and spn.p_correct >= cfg.tau
            hint = spn.value if spn is not None and trusted else None
            out = reader.read_row(doc_id, plan.page, len(ocr_pages), crop, hint, plan.heading_strip)
            res.n_row_calls += 1
        res.latency_s += out.latency_s
        res.n_input_tokens += out.n_input_tokens
        res.n_output_tokens += out.n_output_tokens
        for s in t.scores:
            if out.status in ("unparsable", "identity_mismatch", "no_logprobs"):
                decisions.append(_keep(s, f"reread_{out.status}"))
            else:
                decisions.append(
                    decide(s, out.rereads.get(s.field), confidence, cfg, loc.supported)
                )
    res.n_calls = reader.calls
    res.decisions = sorted(
        decisions + res.decisions,
        key=lambda d: (d.page, d.row_index is not None, d.row_index or -1, d.field),
    )
    res.pred = apply_decisions(pred, res.decisions)
    return res


def _keep(s: FieldScore, reason: str) -> FieldDecision:
    return FieldDecision(s.doc_id, s.page, s.field, s.row_index, "keep", reason, s.value, None)
