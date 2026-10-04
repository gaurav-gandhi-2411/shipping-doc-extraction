"""Seeded image augmentation: field occlusion (synthetic redactions) and scan degradation.

Occlusion
---------
`occlude` hides one box on a page with one of `METHODS`. The locator's word boxes are proportional
splits of an OCR line item (see ``shipdoc.locate``), so their horizontal edges are approximate; the
box is therefore padded before anything is drawn (`padded_box`), using the box height as the OCR
line height. Every method works inside its *applied box*, which always contains the padded target
box and stays inside the page, and leaves every pixel outside the applied box untouched. Smudge
feathers its alpha inward from the applied-box edge over ``FEATHER_PX`` pixels, so its outside
bound is exact too (zero change beyond the applied box) while the padded target is covered at full
strength.

`make_occluded_sample` builds one synthetic variant of a gold document (images plus a gold whose
occluded field is null). `materialize_synthetic` regenerates a whole recipe list deterministically.
Everything is seeded: the same seed gives byte-identical PNG output.

Scan degradation
----------------
`scan_degrade` makes a digital page look scanned (rotation, tone/paper level, blur, noise, JPEG).
Its parameter ranges are fitted to the train scans by ``scripts/fit_scan_augment.py`` and stored in
``configs/augment_scan.yaml`` (see ``reports/augmentation.md``).
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFilter

from shipdoc import locate as loc
from shipdoc import paths
from shipdoc.ocr import PageOcr, doc_pages

Box = tuple[float, float, float, float]
IntBox = tuple[int, int, int, int]
METHODS = ("black_box", "scribble", "smudge", "edge_crop")
LABEL = "synthetic"
SYNTH_DIRNAME = "synthetic_redaction_dev"

# --- occlusion geometry (all fractions are of the OCR line height = target box height) ---------
PAD_X_LINE_FRAC = 0.6  # horizontal padding: word splits are proportional, so edges can be off
PAD_X_WIDTH_FRAC = 0.2  # ... by a share of the value's own width as well; the larger one is used
PAD_Y_LINE_FRAC = 0.25  # vertical padding for ascenders/descenders and OCR box slack
MARGIN_MAX_LINE_FRAC = 0.15  # extra random per-side margin on top of the padding (jitter)
MIN_LINE_PX = 8.0  # floor on the line height, so a degenerate OCR box still gets real padding
FEATHER_PX = 6  # smudge alpha ramp, inside the applied box
EDGE_MAX_DIST_PX = 120  # edge_crop only fires when the target is this close to a page edge
SMUDGE_BLUR_LINE_FRAC = 0.7  # gaussian radius as a share of line height (text unreadable)
BLACK_BOX_INK = (0, 25)  # fill level range of black_box (slightly off pure black)
SCRIBBLE_INK = (0, 50)
SCRIBBLE_WIDTH_LINE_FRAC = 0.28
SCRIBBLE_ROW_STEP = 0.6  # zigzag row spacing as a share of the stroke width (<1: overlapping)


def _color(image: Image.Image, level: float) -> Any:
    """`level` (0-255) as a fill value valid for the image mode (L or RGB)."""
    v = int(round(level))
    if image.mode == "L":
        return v
    if image.mode == "RGB":
        return (v, v, v)
    raise ValueError(f"unsupported image mode {image.mode!r} (need L or RGB)")


def _clip(box: Sequence[float], page_size: tuple[int, int]) -> IntBox:
    """Integer box (floor/ceil) clipped to the page; raises if it ends up empty."""
    w, h = page_size
    x0 = max(0, math.floor(box[0]))
    y0 = max(0, math.floor(box[1]))
    x1 = min(w, math.ceil(box[2]))
    y1 = min(h, math.ceil(box[3]))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"box {tuple(box)} is empty or outside the {w}x{h} page")
    return (x0, y0, x1, y1)


def line_height(box: Sequence[float]) -> float:
    """OCR line height implied by a locator box (its height, floored at ``MIN_LINE_PX``)."""
    return max(float(box[3]) - float(box[1]), MIN_LINE_PX)


def padded_box(box: Sequence[float]) -> tuple[float, float, float, float]:
    """`box` grown to cover the full printed value (unclipped)."""
    h = line_height(box)
    px = max(PAD_X_LINE_FRAC * h, PAD_X_WIDTH_FRAC * (box[2] - box[0]))
    py = PAD_Y_LINE_FRAC * h
    return (box[0] - px, box[1] - py, box[2] + px, box[3] + py)


def edge_distances(box: Sequence[float], page_size: tuple[int, int]) -> dict[str, float]:
    """Distance in px from `box` to each page edge (left/top/right/bottom)."""
    w, h = page_size
    return {"left": box[0], "top": box[1], "right": w - box[2], "bottom": h - box[3]}


def edge_crop_eligible(box: Sequence[float], page_size: tuple[int, int]) -> bool:
    """True iff `box` lies within ``EDGE_MAX_DIST_PX`` of some page edge."""
    return min(edge_distances(box, page_size).values()) <= EDGE_MAX_DIST_PX


def plan_box(
    box: Sequence[float], method: str, rng: np.random.Generator, page_size: tuple[int, int]
) -> IntBox:
    """The applied box of `method`: contains the padded `box`, clipped to the page.

    Consumes `rng` exactly as `occlude` does, so planning and drawing agree for one seed.
    Raises ValueError for an unknown method or an ineligible ``edge_crop``.
    """
    if method not in METHODS:
        raise ValueError(f"unknown occlusion method {method!r}; choose from {METHODS}")
    x0, y0, x1, y1 = padded_box(box)
    h = line_height(box)
    jitter = rng.uniform(0.0, MARGIN_MAX_LINE_FRAC * h, size=4)  # always drawn: stable rng use
    if method == "smudge":
        return _clip(
            (x0 - FEATHER_PX, y0 - FEATHER_PX, x1 + FEATHER_PX, y1 + FEATHER_PX), page_size
        )
    if method == "edge_crop":
        dist = edge_distances(box, page_size)
        edge = min(dist, key=lambda k: dist[k])
        if dist[edge] > EDGE_MAX_DIST_PX:
            raise ValueError(
                f"edge_crop needs a box within {EDGE_MAX_DIST_PX}px of a page edge "
                f"(nearest: {edge} at {dist[edge]:.0f}px)"
            )
        w, ph = page_size
        if edge == "left":
            return _clip((0, y0 - jitter[1], x1 + jitter[2], y1 + jitter[3]), page_size)
        if edge == "right":
            return _clip((x0 - jitter[0], y0 - jitter[1], w, y1 + jitter[3]), page_size)
        if edge == "top":
            return _clip((x0 - jitter[0], 0, x1 + jitter[2], y1 + jitter[3]), page_size)
        return _clip((x0 - jitter[0], y0 - jitter[1], x1 + jitter[2], ph), page_size)
    return _clip((x0 - jitter[0], y0 - jitter[1], x1 + jitter[2], y1 + jitter[3]), page_size)


def _page_background(image: Image.Image) -> Any:
    """Page background colour: the 90th percentile of the grey levels (ink is sparse)."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        return int(np.percentile(arr, 90))
    return tuple(int(np.percentile(arr[..., c], 90)) for c in range(arr.shape[2]))


def _scribble_patch(crop: Image.Image, h: float, rng: np.random.Generator) -> Image.Image:
    """`crop` with dense strokes drawn over it (strokes are clipped to the crop by construction)."""
    w, ph = crop.size
    patch = crop.copy()
    draw = ImageDraw.Draw(patch)
    sw = max(2, round(SCRIBBLE_WIDTH_LINE_FRAC * h))
    ink = float(rng.uniform(*SCRIBBLE_INK))
    fill = _color(patch, ink)
    step = SCRIBBLE_ROW_STEP * sw
    y = -0.5 * sw
    while y < ph + sw:  # zigzag sweeps across the box, overlapping rows -> dense coverage
        pts, x = [], -float(sw)
        while x < w + sw:
            pts.append((x, y + rng.uniform(-0.6 * sw, 0.6 * sw)))
            x += rng.uniform(0.6, 1.6) * sw
        draw.line(pts, fill=fill, width=sw, joint="curve")
        y += step * rng.uniform(0.8, 1.2)
    for _ in range(max(2, int(w / (2 * h)))):  # a few free diagonal strokes for the look
        a = (rng.uniform(0, w), rng.uniform(0, ph))
        b = (rng.uniform(0, w), rng.uniform(0, ph))
        draw.line([a, b], fill=fill, width=sw)
    return patch


def _smudge_patch(crop: Image.Image, h: float, rng: np.random.Generator) -> Image.Image:
    w, ph = crop.size
    blurred = np.asarray(crop.filter(ImageFilter.GaussianBlur(SMUDGE_BLUR_LINE_FRAC * h)), float)
    xs = np.arange(w)[None, :]
    ys = np.arange(ph)[:, None]
    cx = w / 2 + rng.uniform(-0.15, 0.15) * w
    cy = ph / 2 + rng.uniform(-0.15, 0.15) * ph
    blot = np.exp(-0.5 * (((xs - cx) / (w / 3)) ** 2 + ((ys - cy) / (ph / 2.5)) ** 2))
    dark = 1.0 - rng.uniform(0.35, 0.6) * blot  # darkening blotch, centred in the box
    ramp_x = np.minimum(np.arange(w) + 1, w - np.arange(w)) / FEATHER_PX
    ramp_y = np.minimum(np.arange(ph) + 1, ph - np.arange(ph)) / FEATHER_PX
    alpha = np.clip(np.minimum(ramp_y[:, None], ramp_x[None, :]), 0.0, 1.0)  # 0 at the border
    orig = np.asarray(crop, float)
    if orig.ndim == 3:
        dark, alpha = dark[..., None], alpha[..., None]
    out = orig * (1.0 - alpha) + blurred * dark * alpha
    return Image.fromarray(np.clip(np.rint(out), 0, 255).astype(np.uint8), crop.mode)


def occlude(
    image: Image.Image,
    box: Sequence[float],
    method: str,
    rng: np.random.Generator,
    page_size: tuple[int, int],
) -> tuple[Image.Image, IntBox]:
    """Hide `box` on a copy of `image`; returns ``(new_image, applied_box)``.

    `box` is the locator box of the value (page pixels). The applied box contains
    ``padded_box(box)`` clipped to the page; pixels outside it are never modified.

    * ``black_box``: near-black fill (level jittered), per-side jittered margin.
    * ``scribble``: dense overlapping zigzag strokes, width ~0.28 line heights.
    * ``smudge``: gaussian blur (radius 0.7 line heights) plus a darkening blotch, alpha feathered
      inward over ``FEATHER_PX`` px; the padded box is at full strength.
    * ``edge_crop``: fill with the page background from the target to the nearest page edge
      (a cut-off). Only for boxes within ``EDGE_MAX_DIST_PX`` of an edge, else ValueError.
    """
    if image.size != tuple(page_size):
        raise ValueError(f"image size {image.size} != page_size {tuple(page_size)}")
    ab = plan_box(box, method, rng, page_size)
    out = image.copy()
    h = line_height(box)
    if method == "black_box":
        out.paste(_color(out, rng.uniform(*BLACK_BOX_INK)), ab)
    elif method == "edge_crop":
        out.paste(_page_background(image), ab)
    elif method == "scribble":
        out.paste(_scribble_patch(image.crop(ab), h, rng), ab[:2])
    else:  # smudge
        out.paste(_smudge_patch(image.crop(ab), h, rng), ab[:2])
    return out, ab


# ---------------------------------------------------------------------------------------------
# Target resolution and sample construction
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Target:
    """A reliably located value: where to draw, and how ambiguous the location is."""

    page: int  # 0-based
    box: Box
    level: str
    line_idx: int
    n_occurrences: int  # header: lines holding the value at level <= normalized, whole doc


def resolve_target(
    gold: dict[str, Any],
    ocr_pages: list[PageOcr],
    field: str,
    row_idx: int | None,
    page: int | None = None,
    index: loc.DocIndex | None = None,
    assigned: list[loc.RowAssignment] | None = None,
) -> Target | None:
    """Locator box of a gold value at level <= normalized, else None.

    Header fields: the best match (on `page` if given); ``n_occurrences`` counts every line of
    the document holding the value. Row fields: the match on the line ``assign_rows`` gave the
    row (the part number's line), so the same quantity in another row is not confused with it.
    """
    index = index or loc.build_index(ocr_pages)
    if row_idx is None:
        value = gold["header"].get(field)
        ms = loc.find_matches(value, field, index, max_level="normalized")
        if page is not None:
            ms = [m for m in ms if m.page == page]
        if not ms:
            return None
        m = ms[0]
        return Target(m.page, m.box, m.level, m.line_idx, len(ms))
    rows = gold.get("line_items") or []
    a = (assigned or loc.assign_rows(rows, index))[row_idx]  # callers may pass both to reuse them
    if a.match is None or (page is not None and a.match.page != page):
        return None
    if field == "supplier_part_number":
        m = a.match
    else:
        same = [
            x
            for x in loc.find_matches(
                rows[row_idx].get(field), field, index, max_level="normalized"
            )
            if (x.page, x.line_idx) == (a.match.page, a.match.line_idx)
        ]
        if not same:
            return None
        m = same[0]
    if m.level == "fuzzy":
        return None
    return Target(m.page, m.box, m.level, m.line_idx, 1)


def resolve_all_targets(
    gold: dict[str, Any],
    ocr_pages: list[PageOcr],
    field: str,
    index: loc.DocIndex | None = None,
) -> list[Target]:
    """Every line of the document holding a header value at level <= normalized, reading order.

    For fields printed several times (currency, total_amount): the boxes ``occlude`` must all hide
    for the value to be unreadable. One box per line (the locator keeps the best span per line).
    """
    index = index or loc.build_index(ocr_pages)
    ms = loc.find_matches(gold["header"].get(field), field, index, max_level="normalized")
    ms = sorted(ms, key=lambda m: (m.page, m.line_idx))
    return [Target(m.page, m.box, m.level, m.line_idx, len(ms)) for m in ms]


def box_rng(seed: int, k: int, all_occurrences: bool) -> np.random.Generator:
    """Generator for the k-th occluded box of a recipe.

    A single-box recipe uses ``default_rng(seed)`` (as every recipe did before multi-box ones
    existed); an all-occurrences recipe gives each box its own stream ``default_rng([seed, k])``,
    so one box's drawing cannot shift the jitter of the next and the applied boxes can be
    planned ahead of time (``plan_box`` alone) by the recipe builder.
    """
    return np.random.default_rng([seed, k] if all_occurrences else seed)


@dataclass
class OccludedSample:
    """One synthetic variant: modified page images, modified gold, and what was drawn."""

    images: list[Image.Image]
    gold: dict[str, Any]
    page: int
    target_box: Box
    applied_box: IntBox
    method: str
    seed: int
    applied_boxes: list[IntBox] = field(default_factory=list)  # all boxes (first == applied_box)


def null_gold(gold: dict[str, Any], field: str, row_idx: int | None) -> dict[str, Any]:
    """Deep copy of `gold` with exactly one value set to null (header field, or one row's field)."""
    out = copy.deepcopy(gold)
    holder = out["header"] if row_idx is None else out["line_items"][row_idx]
    if field not in holder or holder[field] is None:
        raise ValueError(f"gold field {field!r} (row {row_idx}) is already null or absent")
    holder[field] = None
    return out


def load_doc_images(gold: dict[str, Any], data_root: Path | None = None) -> list[Image.Image]:
    """Page images of a gold doc from ``<data dir>/<split>/images`` (fully decoded)."""
    split = gold["doc_id"].split("_", 1)[0]
    root = (paths.data_dir() if data_root is None else Path(data_root)) / split / "images"
    out = []
    for name in gold["pages"]:
        with Image.open(root / name) as im:
            out.append(im.copy())
    return out


def make_occluded_sample(
    doc: dict[str, Any],
    field: str,
    row_idx: int | None,
    method: str,
    seed: int,
    *,
    page: int | None = None,
    images: list[Image.Image] | None = None,
    ocr_pages: list[PageOcr] | None = None,
    all_occurrences: bool = False,
    n_boxes: int | None = None,
) -> OccludedSample:
    """Occlude one gold value of `doc` (a gold label dict) with `method`; fully seeded.

    Returns the page images (only the target pages change) and a gold where that field is null
    (for a row field only that row's field). `images`/`ocr_pages` default to the configured
    data dir and OCR cache. Raises ValueError if the value is null or has no reliable box.

    ``all_occurrences`` (header fields only) hides every line holding the value (see
    ``resolve_all_targets``), one box each with its own ``box_rng`` stream; `n_boxes`, when given,
    must equal how many were found (guards against the locator drifting under a recipe).
    """
    gold = null_gold(doc, field, row_idx)  # validates non-null before any heavy work
    pages = ocr_pages if ocr_pages is not None else doc_pages(doc["doc_id"])
    if all_occurrences:
        if row_idx is not None:
            raise ValueError("all_occurrences applies to header fields only")
        tgts = resolve_all_targets(doc, pages, field)
        if not tgts or (n_boxes is not None and len(tgts) != n_boxes):
            raise ValueError(
                f"{doc['doc_id']} {field}: found {len(tgts)} occurrences, recipe says {n_boxes}"
            )
        imgs = list(images) if images is not None else load_doc_images(doc)
        applied: list[IntBox] = []
        for k, t in enumerate(tgts):
            imgs[t.page], ab = occlude(
                imgs[t.page], t.box, method, box_rng(seed, k, True), imgs[t.page].size
            )
            applied.append(ab)
        return OccludedSample(
            imgs, gold, tgts[0].page, tgts[0].box, applied[0], method, seed, applied
        )
    tgt = resolve_target(doc, pages, field, row_idx, page)
    if tgt is None:
        raise ValueError(f"{doc['doc_id']} {field} row={row_idx}: no reliable box (level<=b)")
    imgs = list(images) if images is not None else load_doc_images(doc)
    rng = box_rng(seed, 0, False)
    img = imgs[tgt.page]
    imgs[tgt.page], applied_one = occlude(img, tgt.box, method, rng, img.size)
    return OccludedSample(imgs, gold, tgt.page, tgt.box, applied_one, method, seed, [applied_one])


def png_bytes(image: Image.Image) -> bytes:
    """Deterministic PNG encoding (fixed compression, no metadata chunks)."""
    buf = io.BytesIO()
    image.save(buf, format="PNG", compress_level=6, optimize=False)
    return buf.getvalue()


def sha256_hex(data: bytes) -> str:
    """SHA-256 of `data` as lowercase hex."""
    return hashlib.sha256(data).hexdigest()


def _write_if_changed(path: Path, data: bytes) -> None:
    if path.is_file() and path.read_bytes() == data:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def _json_bytes(obj: Any) -> bytes:
    return (json.dumps(obj, indent=1, ensure_ascii=False) + "\n").encode("utf-8")


def materialize_synthetic(
    recipes: list[dict[str, Any]],
    out_dir: Path | str | None = None,
    *,
    data_root: Path | None = None,
    ocr_cache: Path | None = None,
    meta_dir: Path | None = None,
) -> dict[str, Any]:
    """Regenerate every recipe's images and label under `out_dir`; returns the manifest.

    Default `out_dir` is ``<SHIPDOC_RUNS_DIR>/synthetic_redaction_dev``. Layout:
    ``images/<variant_id>_p<k>.png``, ``labels/<variant_id>.json`` (gold with the field nulled,
    ``"synthetic": true``, ``doc_id`` = variant_id), ``meta.json`` (slice tags per variant) and
    ``manifest.json`` (SHA-256 of every generated file). Idempotent: unchanged files are not
    rewritten, and a second run yields an identical manifest.
    """
    out = Path(out_dir) if out_dir is not None else paths.runs_dir() / SYNTH_DIRNAME
    cache = Path(ocr_cache) if ocr_cache is not None else None
    meta_root = Path(meta_dir) if meta_dir is not None else paths.REPO_ROOT / "meta"
    tags = {
        sp: {m["doc_id"]: m for m in json.loads((meta_root / f"{sp}.json").read_text("utf-8"))}
        for sp in ("train", "dev")
    }
    sha: dict[str, str] = {}
    meta_rows: list[dict[str, Any]] = []
    for r in recipes:
        doc_id, vid = r["doc_id"], r["variant_id"]
        split = doc_id.split("_", 1)[0]
        root = (paths.data_dir() if data_root is None else Path(data_root)) / split
        gold = json.loads((root / "labels" / f"{doc_id}.json").read_text(encoding="utf-8"))
        ocr = doc_pages(doc_id, cache)
        s = make_occluded_sample(
            gold,
            r["field"],
            r["row_idx"],
            r["method"],
            r["seed"],
            page=r["page"],
            images=load_doc_images(gold, data_root),
            ocr_pages=ocr,
            all_occurrences=bool(r.get("all_occurrences")),
            n_boxes=r.get("n_boxes"),
        )
        names = []
        for k, im in enumerate(s.images, start=1):
            name = f"{vid}_p{k}.png"
            data = png_bytes(im)
            _write_if_changed(out / "images" / name, data)
            sha[f"images/{name}"] = sha256_hex(data)
            names.append(name)
        label = {
            **s.gold,
            "doc_id": vid,
            "source_doc_id": doc_id,
            "pages": names,
            "synthetic": True,
            "recipe": {
                k: r[k]
                for k in (
                    "field",
                    "row_idx",
                    "page",
                    "method",
                    "seed",
                    "n_boxes",
                    "all_occurrences",
                    "inferable_from_siblings",
                )
                if k in r  # optional keys exist only on the recipes they apply to
            },
        }
        data = _json_bytes(label)
        _write_if_changed(out / "labels" / f"{vid}.json", data)
        sha[f"labels/{vid}.json"] = sha256_hex(data)
        meta_rows.append({**tags[split][doc_id], "doc_id": vid, "synthetic": True})
    _write_if_changed(out / "meta.json", _json_bytes(meta_rows))
    manifest = {
        "label": LABEL,
        "n_variants": len(recipes),
        "sha256": dict(sorted(sha.items())),
    }
    _write_if_changed(out / "manifest.json", _json_bytes(manifest))
    return manifest


# ---------------------------------------------------------------------------------------------
# Composition of a recipe set
# ---------------------------------------------------------------------------------------------


def field_group(recipe: dict[str, Any], tag: dict[str, Any]) -> str:
    """``invoice_header`` / ``waybill_header`` / ``row`` for a recipe, given its doc's meta tag."""
    if recipe["row_idx"] is not None:
        return "row"
    return "waybill_header" if tag["waybill"] else "invoice_header"


def composition_markdown(recipes: list[dict[str, Any]], tags: dict[str, dict[str, Any]]) -> str:
    """Markdown table: field x method, each cell ``scanned/digital`` counts, plus totals."""
    cell: dict[tuple[str, str], list[int]] = {}
    for r in recipes:
        tag = tags[r["doc_id"]]
        key = (f"{field_group(r, tag)}:{r['field']}", r["method"])
        cell.setdefault(key, [0, 0])[0 if tag["scanned"] else 1] += 1
    lines = [
        "| field | " + " | ".join(METHODS) + " | total (scanned/digital) |",
        "|---|" + "---:|" * (len(METHODS) + 1),
    ]
    tot = {m: [0, 0] for m in METHODS}
    for f in sorted({k[0] for k in cell}):
        cells, ft = [], [0, 0]
        for m in METHODS:
            sd = cell.get((f, m), [0, 0])
            cells.append(f"{sd[0]}/{sd[1]}" if any(sd) else "-")
            for i in (0, 1):
                ft[i] += sd[i]
                tot[m][i] += sd[i]
        lines.append(f"| {f} | " + " | ".join(cells) + f" | {ft[0] + ft[1]} ({ft[0]}/{ft[1]}) |")
    n_s, n_d = (sum(v[i] for v in tot.values()) for i in (0, 1))
    lines.append(
        "| **all** | "
        + " | ".join(f"{tot[m][0] + tot[m][1]} ({tot[m][0]}/{tot[m][1]})" for m in METHODS)
        + f" | {n_s + n_d} ({n_s}/{n_d}) |"
    )
    return "\n".join(lines) + "\n"


RECIPES_PATH = paths.REPO_ROOT / "splits" / "synthetic_redaction_dev.json"


def load_recipes(path: Path | str | None = None) -> list[dict[str, Any]]:
    """Recipes of ``splits/synthetic_redaction_dev.json`` (or `path`)."""
    return list(json.loads(Path(path or RECIPES_PATH).read_text(encoding="utf-8"))["recipes"])


def main_synth_redaction(args: Any) -> int:
    """CLI body of ``python -m shipdoc synth-redaction`` (--materialize and/or --table)."""
    recipes = load_recipes(args.recipes)
    print(f"SYNTHETIC redaction eval: {len(recipes)} recipes")
    if args.table:
        tags = {
            m["doc_id"]: m
            for m in json.loads((paths.REPO_ROOT / "meta" / "dev.json").read_text("utf-8"))
        }
        print(composition_markdown(recipes, tags))
    if args.materialize:
        out = Path(args.out_dir) if args.out_dir else paths.runs_dir() / SYNTH_DIRNAME
        manifest = materialize_synthetic(recipes, out)
        print(f"materialized {manifest['n_variants']} variants -> {out}")
        print(f"manifest: {out / 'manifest.json'} ({len(manifest['sha256'])} files hashed)")
    return 0


# ---------------------------------------------------------------------------------------------
# Scan degradation
# ---------------------------------------------------------------------------------------------

SCAN_CONFIG_PATH = paths.REPO_ROOT / "configs" / "augment_scan.yaml"
JPEG_QUALITY_BOUNDS = (30, 70)  # spec range; fitted sub-ranges must stay inside


@dataclass(frozen=True)
class ScanParams:
    """Ranges of the scan-degradation pipeline (each pair is a uniform (lo, hi) range)."""

    rotate_deg: float  # angle ~ U(-rotate_deg, +rotate_deg)
    paper_gain: tuple[float, float]  # white level = 255 * gain (greyish paper)
    # gain = hi - (hi - lo) * u**skew, u~U(0,1): skew > 1 piles mass near `hi` (most scans are
    # near-white with a long tail of greyer paper, which a plain uniform range cannot express).
    paper_gain_skew: float
    ink_lift: tuple[float, float]  # black level (ink is never pure black on a scan)
    blur_sigma: tuple[float, float]  # gaussian radius in px
    noise_sigma: tuple[float, float]  # additive gaussian noise, grey levels
    jpeg_quality: tuple[int, int]  # inclusive

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ScanParams:
        """Build from the ``params`` mapping of ``configs/augment_scan.yaml``."""
        return cls(
            rotate_deg=float(d["rotate_deg"]),
            paper_gain=(float(d["paper_gain"][0]), float(d["paper_gain"][1])),
            paper_gain_skew=float(d.get("paper_gain_skew", 1.0)),
            ink_lift=(float(d["ink_lift"][0]), float(d["ink_lift"][1])),
            blur_sigma=(float(d["blur_sigma"][0]), float(d["blur_sigma"][1])),
            noise_sigma=(float(d["noise_sigma"][0]), float(d["noise_sigma"][1])),
            jpeg_quality=(int(d["jpeg_quality"][0]), int(d["jpeg_quality"][1])),
        )

    def to_dict(self) -> dict[str, Any]:
        """Inverse of `from_dict` (plain lists, YAML friendly)."""
        return {
            "rotate_deg": self.rotate_deg,
            "paper_gain": list(self.paper_gain),
            "paper_gain_skew": self.paper_gain_skew,
            "ink_lift": list(self.ink_lift),
            "blur_sigma": list(self.blur_sigma),
            "noise_sigma": list(self.noise_sigma),
            "jpeg_quality": list(self.jpeg_quality),
        }


def load_scan_params(path: Path | str | None = None) -> ScanParams:
    """Fitted parameters from ``configs/augment_scan.yaml`` (or `path`)."""
    cfg = yaml.safe_load(Path(path or SCAN_CONFIG_PATH).read_text(encoding="utf-8"))
    return ScanParams.from_dict(cfg["params"])


def scan_degrade(image: Image.Image, rng: np.random.Generator, params: ScanParams) -> Image.Image:
    """Make a digital page look scanned: rotate, tone, blur, noise, then a JPEG round trip.

    The JPEG round trip is last so the 8x8 block grid is aligned, as on a real scan. Output has
    the input's size and mode (L or RGB) and is deterministic for a given `rng` state.
    """
    if image.mode not in ("L", "RGB"):
        raise ValueError(f"unsupported image mode {image.mode!r} (need L or RGB)")
    angle = rng.uniform(-params.rotate_deg, params.rotate_deg) if params.rotate_deg > 0 else 0.0
    g_lo, g_hi = params.paper_gain
    gain = g_hi - (g_hi - g_lo) * rng.uniform() ** params.paper_gain_skew
    lift = rng.uniform(*params.ink_lift)
    sigma = rng.uniform(*params.blur_sigma)
    noise = rng.uniform(*params.noise_sigma)
    quality = int(rng.integers(params.jpeg_quality[0], params.jpeg_quality[1] + 1))
    bg = _page_background(image)
    img = image.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=bg)
    arr = np.asarray(img, dtype=np.float32)
    white = 255.0 * gain
    arr = lift + arr * ((white - lift) / 255.0)
    img = Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8), image.mode)
    if sigma > 0.05:
        img = img.filter(ImageFilter.GaussianBlur(sigma))
    if noise > 0.0:
        arr = np.asarray(img, dtype=np.float32)
        arr = arr + rng.normal(0.0, noise, size=arr.shape).astype(np.float32)
        img = Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8), image.mode)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    with Image.open(io.BytesIO(buf.getvalue())) as dec:
        return dec.copy()
