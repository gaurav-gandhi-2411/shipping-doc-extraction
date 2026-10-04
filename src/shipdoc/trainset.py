"""Fine-tuning data builder (spec Phase 4.1-4.2, reports/finetune_plan.md V1d).

Gold document -> per-PAGE keyed targets (prompt v2, the declared key order the extraction path
emits), with the page of every gold row taken from the locator (``locate.assign_rows``), the
seeded training augmentations, and the stage splits (CV folds / final) from ``splits/folds.json``.

Row-to-page policy (``resolve_row_pages``), applied to the rows ``assign_rows`` could not place on
a distinct OCR line:

* ``line`` / ``page_fuzzy`` rows keep their assigned page;
* an unassigned row of a single-page document goes to page 0;
* an unassigned row of a multipage document takes the page of its neighbours when the previous
  and the next ASSIGNED row (in gold row order) sit on the same page ("interpolated"). Gold rows
  keep a non-decreasing page in 151 of 152 multipage docs (reports/finetune_plan.md), so two
  equal neighbours pin the page. ``monotone_boundary`` (default on, it reproduces the V1 counts:
  38 of 50 train and 15 of 16 dev multipage rows resolved) also pins a leading row whose next
  neighbour is on page 0 and a trailing row whose previous neighbour is on the last page;
* any other row stays AMBIGUOUS. A document with an ambiguous row is, per ``ambiguous_policy``,
  trained header-only (``header_only``: ``doc_type``, header and ``page_kind`` tokens are
  supervised, the ``line_items`` tokens are masked and the unplaceable rows left out of the
  text) or excluded from training (``exclude``). Every decision is written to an audit log.

Header fields per page follow ``shipdoc.targets.gold_page_payloads`` (identity fields on page 1,
totals on the last page, everything else null), so a model trained on these targets reproduces
``shipdoc.merge`` (the page-2 banner trap: continuation pages have ``invoice_number`` null).

Occlusion augmentation (``select_occlusions``): with probability ``occlusion_rate`` per page and
epoch, ONE header field that the page's target carries is hidden at EVERY printed copy and its
target is set to null. A field is a candidate only when the occlusion provably makes the value
unreadable: all copies are located at level <= normalized, none is only fuzzy-matched, at least
one sits on the page that carries the target, and the widest possible applied box harms no other
located value. Row fields are not occluded here.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from shipdoc import augment as A
from shipdoc import locate as loc
from shipdoc import paths
from shipdoc.extract import HEADER_KEYS, ROW_KEYS
from shipdoc.prompts import build_prompt
from shipdoc.targets import LAST_PAGE_FIELDS, gold_page_payloads

SEED = 42
AMBIGUOUS_POLICIES = ("header_only", "exclude")
#: Occlusion methods used in training. ``edge_crop`` is left out: it extends the box to the page
#: edge, so its collateral reach cannot be bounded by the widest-box check below.
OCCLUSION_METHODS = ("black_box", "scribble", "smudge")
#: Stage names accepted by `stage_split`.
STAGES = ("smoke", "fold0", "fold1", "fold2", "final")
#: Another value's box may be covered by at most this share before the occlusion counts as harm.
COLLATERAL_MAX_FRAC = 0.10
ROW_SOURCES = (
    "line",
    "page_fuzzy",
    "single_page_default",
    "interpolated",
    "monotone_boundary",
    "ambiguous",
)

Box = tuple[float, float, float, float]


# --------------------------------------------------------------------------------------------
# Row -> page
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RowPageResolution:
    """Page (0-based, None = ambiguous) and provenance of every gold row of one document."""

    row_pages: tuple[int | None, ...]
    sources: tuple[str, ...]

    @property
    def n_ambiguous(self) -> int:
        """Rows whose page could not be pinned."""
        return sum(p is None for p in self.row_pages)


def resolve_row_pages(
    levels: Sequence[str],
    pages: Sequence[int | None],
    n_pages: int,
    *,
    monotone_boundary: bool = True,
) -> RowPageResolution:
    """Apply the row-to-page policy (module docstring) to ``assign_rows`` output.

    `levels[i]` is ``line`` / ``page_fuzzy`` / ``unassigned`` and `pages[i]` the assigned page
    (None for unassigned rows). Raises ValueError on an assigned page outside the document.
    """
    if len(levels) != len(pages):
        raise ValueError(f"levels ({len(levels)}) and pages ({len(pages)}) differ in length")
    n = len(levels)
    known = [i for i in range(n) if levels[i] != "unassigned"]
    for i in known:
        if pages[i] is None or not 0 <= int(pages[i]) < max(1, n_pages):  # type: ignore[arg-type]
            raise ValueError(f"row {i}: assigned page {pages[i]} outside a {n_pages}-page doc")
    out_pages: list[int | None] = [None] * n
    sources = ["ambiguous"] * n
    for i in known:
        out_pages[i] = int(pages[i])  # type: ignore[arg-type]
        sources[i] = levels[i]
    for i in range(n):
        if levels[i] != "unassigned":
            continue
        if n_pages <= 1:
            out_pages[i], sources[i] = 0, "single_page_default"
            continue
        prev = next((j for j in range(i - 1, -1, -1) if levels[j] != "unassigned"), None)
        nxt = next((j for j in range(i + 1, n) if levels[j] != "unassigned"), None)
        p_prev = out_pages[prev] if prev is not None else None
        p_next = out_pages[nxt] if nxt is not None else None
        if p_prev is not None and p_prev == p_next:
            out_pages[i], sources[i] = p_prev, "interpolated"
        elif monotone_boundary and prev is None and p_next == 0:
            out_pages[i], sources[i] = 0, "monotone_boundary"
        elif monotone_boundary and nxt is None and p_prev == n_pages - 1:
            out_pages[i], sources[i] = n_pages - 1, "monotone_boundary"
    return RowPageResolution(tuple(out_pages), tuple(sources))


def assignments_from_ocr(
    gold: Mapping[str, Any], ocr_pages: list[Any]
) -> tuple[list[str], list[int | None]]:
    """``(levels, pages)`` of the gold rows from the locator (``assign_rows``), as in provenance."""
    rows = list(gold.get("line_items") or [])
    if not rows:
        return [], []
    got = loc.assign_rows(rows, loc.build_index(ocr_pages))
    return [a.level for a in got], [a.page for a in got]


def assignments_from_locations(path: Path) -> dict[str, tuple[list[str], list[int | None]]]:
    """Same pair per doc from ``provenance/locations.jsonl`` row records (cross-check source)."""
    by_doc: dict[str, dict[int, tuple[str, int | None]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("scope") == "row":
            by_doc.setdefault(rec["doc_id"], {})[int(rec["row_idx"])] = (
                rec["level"],
                rec.get("assigned_page"),
            )
    out = {}
    for d, rows in by_doc.items():
        order = sorted(rows)
        out[d] = ([rows[i][0] for i in order], [rows[i][1] for i in order])
    return out


@dataclass(frozen=True)
class DocPlan:
    """How one document enters training."""

    doc_id: str
    n_pages: int
    row_pages: tuple[int | None, ...]
    row_sources: tuple[str, ...]
    status: str  # "ok" | "header_only" | "excluded"

    @property
    def n_rows(self) -> int:
        """Gold rows of the document."""
        return len(self.row_pages)

    @property
    def supervise_line_items(self) -> bool:
        """False for header-only documents (line_items tokens are masked)."""
        return self.status == "ok"

    def events(self) -> list[dict[str, Any]]:
        """Audit events (no values): interpolations, defaults, ambiguous rows, doc decision."""
        out = [
            {"doc_id": self.doc_id, "event": src, "row_idx": i, "page": p}
            for i, (src, p) in enumerate(zip(self.row_sources, self.row_pages, strict=True))
            if src in ("interpolated", "monotone_boundary", "single_page_default", "ambiguous")
        ]
        if self.status != "ok":
            out.append({"doc_id": self.doc_id, "event": f"doc_{self.status}", "row_idx": None,
                        "page": None})  # fmt: skip
        return out


def plan_doc(
    gold: Mapping[str, Any],
    levels: Sequence[str],
    pages: Sequence[int | None],
    *,
    ambiguous_policy: str = "header_only",
    monotone_boundary: bool = True,
) -> DocPlan:
    """`DocPlan` of one gold document from its row assignments."""
    if ambiguous_policy not in AMBIGUOUS_POLICIES:
        raise ValueError(f"ambiguous_policy {ambiguous_policy!r} not in {AMBIGUOUS_POLICIES}")
    n_pages = max(1, len(gold.get("pages") or [None]))
    n_rows = len(gold.get("line_items") or [])
    if len(levels) != n_rows:
        raise ValueError(f"{gold['doc_id']}: {len(levels)} row assignments for {n_rows} gold rows")
    res = resolve_row_pages(levels, pages, n_pages, monotone_boundary=monotone_boundary)
    status = "ok"
    if res.n_ambiguous:
        status = "header_only" if ambiguous_policy == "header_only" else "excluded"
    return DocPlan(gold["doc_id"], n_pages, res.row_pages, res.sources, status)


# --------------------------------------------------------------------------------------------
# Page targets
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PageTarget:
    """Target of one page: payload dict, its text and the line_items character span."""

    page_index: int
    payload: dict[str, Any]
    text: str
    line_items_span: tuple[int, int]
    supervise_line_items: bool


def render_page_text(payload: Mapping[str, Any]) -> tuple[str, tuple[int, int]]:
    """Target text exactly as ``targets.render_target`` writes it, plus the line_items char span.

    The span covers the JSON array after ``"line_items": `` (used to mask those tokens).
    """
    sep = {"separators": (", ", ": "), "ensure_ascii": False}
    head = (
        '{"doc_type": '
        + json.dumps(payload["doc_type"], **sep)
        + ', "header": '
        + json.dumps(payload["header"], **sep)
        + ', "line_items": '
    )
    items = json.dumps(payload["line_items"], **sep)
    text = head + items + ', "page_kind": ' + json.dumps(payload["page_kind"], **sep) + "}"
    return text, (len(head), len(head) + len(items))


def page_targets(
    gold: Mapping[str, Any], plan: DocPlan, nulled: Iterable[str] = ()
) -> list[PageTarget]:
    """Per-page targets of an included document; `nulled` header fields are set to null first.

    Header-only documents leave their ambiguous rows out (their line_items tokens are masked, so
    only the text context differs). Raises ValueError for an excluded document.
    """
    if plan.status == "excluded":
        raise ValueError(f"{plan.doc_id} is excluded from training")
    g = copy.deepcopy(dict(gold))
    for f_name in nulled:
        g = A.null_gold(g, f_name, None)
    rows = list(g.get("line_items") or [])
    keep = [i for i, p in enumerate(plan.row_pages) if p is not None]
    g["line_items"] = [rows[i] for i in keep]
    payloads = gold_page_payloads(g, plan.n_pages, [plan.row_pages[i] for i in keep])
    out = []
    for k, payload in enumerate(payloads):
        text, span = render_page_text(payload)
        out.append(PageTarget(k, payload, text, span, plan.supervise_line_items))
    return out


# --------------------------------------------------------------------------------------------
# Stage splits
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StageSplit:
    """Doc ids a stage trains on and the held-out ids (informational eval loss / OOF)."""

    stage: str
    train_ids: tuple[str, ...]
    heldout_ids: tuple[str, ...]

    def manifest_hash(self) -> str:
        """sha256 prefix over stage + sorted train ids (goes into the run id / config record)."""
        blob = self.stage + "|" + ",".join(sorted(self.train_ids))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def stage_split(stage: str, folds: Mapping[str, Any]) -> StageSplit:
    """Split for `stage` from ``splits/folds.json`` content (its ONLY source).

    ``foldK``: train = every train+dev doc outside fold K's ``val_doc_ids``; held-out = those ids.
    ``final`` / ``smoke``: train = the ``train_*`` docs; held-out = the ``dev_*`` docs (the dev
    docs are the official seen-layout evaluation and never train the final model).
    """
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; valid: {list(STAGES)}")
    every = sorted(d for f in folds["folds"] for d in f["val_doc_ids"])
    if stage.startswith("fold"):
        k = int(stage[4:])
        val = sorted(next(f for f in folds["folds"] if f["fold"] == k)["val_doc_ids"])
        s = set(val)
        split = StageSplit(stage, tuple(d for d in every if d not in s), tuple(val))
    else:
        split = StageSplit(
            stage,
            tuple(d for d in every if d.startswith("train_")),
            tuple(d for d in every if d.startswith("dev_")),
        )
    assert_no_leakage(split)
    return split


def assert_no_leakage(split: StageSplit) -> None:
    """Raise ValueError if train and held-out overlap or a test doc is present anywhere."""
    both = set(split.train_ids) & set(split.heldout_ids)
    if both:
        raise ValueError(f"{split.stage}: {len(both)} doc ids are in train AND held-out")
    bad = [d for d in (*split.train_ids, *split.heldout_ids) if d.startswith("test_")]
    if bad:
        raise ValueError(f"{split.stage}: test docs present, e.g. {bad[:3]}")
    if split.stage in ("final", "smoke") and any(d.startswith("dev_") for d in split.train_ids):
        raise ValueError("final/smoke must not train on dev docs")


# --------------------------------------------------------------------------------------------
# Occlusion candidates and per-sample selection
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class OcclusionCandidate:
    """A header field that can be hidden: where it is printed and which page carries its target."""

    field: str
    target_page: int
    boxes: tuple[tuple[int, Box], ...]  # (page, locator box) of EVERY printed copy

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible dict."""
        return {"field": self.field, "target_page": self.target_page,
                "boxes": [[p, list(b)] for p, b in self.boxes]}  # fmt: skip

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> OcclusionCandidate:
        """Inverse of `to_dict`."""
        return cls(
            d["field"],
            int(d["target_page"]),
            tuple((int(p), (b[0], b[1], b[2], b[3])) for p, b in d["boxes"]),
        )


def carrying_page(doc_type: str, field_name: str, n_pages: int) -> int:
    """Page whose target carries a header field: last page for totals, else page 0."""
    return n_pages - 1 if field_name in LAST_PAGE_FIELDS[doc_type] else 0


def widest_applied_box(box: Box) -> Box:
    """Outer bound of any applied box of `OCCLUSION_METHODS` for a locator box (unclipped)."""
    x0, y0, x1, y1 = A.padded_box(box)
    m = A.MARGIN_MAX_LINE_FRAC * A.line_height(box) + A.FEATHER_PX
    return (x0 - m, y0 - m, x1 + m, y1 + m)


def _overlap_frac(a: Box, b: Box) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    area = max((b[2] - b[0]) * (b[3] - b[1]), 1e-9)
    return max(w, 0.0) * max(h, 0.0) / area


def occlusion_candidates(
    gold: Mapping[str, Any], ocr_pages: list[Any], n_pages: int
) -> tuple[list[OcclusionCandidate], Counter[str]]:
    """Occludable header fields of one document, plus a reason counter for the rest.

    Reasons: ``null_gold`` (nothing to hide), ``no_box`` (not located at level <= normalized),
    ``not_on_target_page`` (no copy on the page that carries the target), ``extra_fuzzy`` (a
    fuzzy-only copy would stay readable), ``collateral`` (the widest applied box would hide
    another located value), ``eligible``.
    """
    doc_type = gold["doc_type"]
    index = loc.build_index(ocr_pages)
    reasons: Counter[str] = Counter()
    others: list[tuple[int, Box, str]] = []
    copies: dict[str, list[A.Target]] = {}
    for f_name, value in gold["header"].items():
        if value is None:
            continue
        copies[f_name] = A.resolve_all_targets(dict(gold), ocr_pages, f_name, index)
        others += [(t.page, t.box, f"h:{f_name}") for t in copies[f_name]]
    rows = list(gold.get("line_items") or [])
    assigned = loc.assign_rows(rows, index) if rows else []
    for i, row in enumerate(rows):
        for rf in ("supplier_part_number", "customer_part_number", "purchase_order", "quantity"):
            if row.get(rf) is None:
                continue
            t = A.resolve_target(dict(gold), ocr_pages, rf, i, index=index, assigned=assigned)
            if t is not None:
                others.append((t.page, t.box, f"r{i}:{rf}"))
    out: list[OcclusionCandidate] = []
    for f_name, value in gold["header"].items():
        if f_name not in HEADER_KEYS:
            continue
        if value is None:
            reasons["null_gold"] += 1
            continue
        tgts = copies[f_name]
        target_page = carrying_page(doc_type, f_name, n_pages)
        if not tgts:
            reasons["no_box"] += 1
        elif all(t.page != target_page for t in tgts):
            reasons["not_on_target_page"] += 1
        elif len(loc.find_matches(value, f_name, index, max_level="fuzzy")) != len(tgts):
            reasons["extra_fuzzy"] += 1
        elif any(
            p == t.page and tag != f"h:{f_name}"
            and _overlap_frac(widest_applied_box(t.box), b) > COLLATERAL_MAX_FRAC
            for t in tgts
            for p, b, tag in others
        ):  # fmt: skip
            reasons["collateral"] += 1
        else:
            reasons["eligible"] += 1
            out.append(
                OcclusionCandidate(f_name, target_page, tuple((t.page, t.box) for t in tgts))
            )
    return out, reasons


def sample_seed(seed: int, doc_id: str, page: int, epoch: int, tag: str) -> int:
    """Per-sample seed from (seed, doc, page, epoch, purpose): same inputs, same stream."""
    blob = f"{seed}|{doc_id}|{page}|{epoch}|{tag}".encode()
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "big")


@dataclass(frozen=True)
class OcclusionChoice:
    """One applied occlusion: the page that drew it, the field, how, the seed and every box."""

    page: int
    field: str
    method: str
    seed: int
    boxes: tuple[tuple[int, Box], ...]


def select_occlusions(
    doc_id: str,
    n_pages: int,
    epoch: int,
    candidates: Sequence[OcclusionCandidate],
    rate: float,
    seed: int = SEED,
    weights: Mapping[str, float] | None = None,
) -> list[OcclusionChoice]:
    """Per page and epoch: with probability `rate`, hide ONE candidate field of that page.

    Independent per page (own seed), so the decision of one page never shifts another's. The
    chosen field is hidden at every copy, including copies on other pages (a continuation
    banner), and its target on the carrying page becomes null. The field is drawn with
    probability proportional to ``weights[field]`` (default 1: uniform over the candidates).
    """
    out: list[OcclusionChoice] = []
    for page in range(n_pages):
        here = [c for c in candidates if c.target_page == page]
        rng = np.random.default_rng(sample_seed(seed, doc_id, page, epoch, "occ"))
        draw = rng.random()
        if not here or draw >= rate:
            continue
        w = np.array([float((weights or {}).get(c.field, 1.0)) for c in here])
        if (w < 0).any() or w.sum() <= 0:
            raise ValueError(f"occlusion weights must be >= 0 with a positive sum, got {w}")
        cand = here[int(rng.choice(len(here), p=w / w.sum()))]
        method = OCCLUSION_METHODS[int(rng.integers(len(OCCLUSION_METHODS)))]
        out.append(
            OcclusionChoice(page, cand.field, method, int(rng.integers(0, 2**31 - 1)), cand.boxes)
        )
    return out


def apply_occlusions(
    image: Image.Image, page: int, choices: Sequence[OcclusionChoice]
) -> Image.Image:
    """Draw every box of `choices` that lies on `page`; box k uses its own ``box_rng`` stream."""
    for c in choices:
        for k, (p, box) in enumerate(c.boxes):
            if p == page:
                image, _ = A.occlude(image, box, c.method, A.box_rng(c.seed, k, True), image.size)
    return image


# --------------------------------------------------------------------------------------------
# Rendering one training sample
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AugmentConfig:
    """Training augmentation settings (stated in configs/finetune_qwen35_4b.yaml)."""

    occlusion_rate: float = 0.15
    scan_prob: float = 0.5
    seed: int = SEED
    # Relative draw weight of a hidable field (default 1 each). Real redactions are only ever on
    # invoice_number / invoice_date (35 of 35), the two fields whose null-on-redaction behaviour
    # is scored (false-fill split), so they get 4x; every other header field stays a candidate.
    field_weights: Mapping[str, float] = field(
        default_factory=lambda: {"invoice_number": 4.0, "invoice_date": 4.0}
    )


@dataclass(frozen=True)
class PageSpec:
    """One training page: which image, and where it sits in its document."""

    doc_id: str
    split: str
    page_index: int
    n_pages: int
    image_name: str


@dataclass
class RenderedPage:
    """A training sample before tokenisation."""

    spec: PageSpec
    image: Image.Image
    prompt: str
    target: PageTarget
    occluded: tuple[str, ...] = ()
    degraded: bool = False


@dataclass
class PreparedSet:
    """Everything needed to render samples of a stage (no torch)."""

    golds: dict[str, dict[str, Any]]
    plans: dict[str, DocPlan]
    candidates: dict[str, list[OcclusionCandidate]] = field(default_factory=dict)
    data_root: Path | None = None
    scan_params: A.ScanParams | None = None

    def included(self, doc_ids: Iterable[str]) -> list[str]:
        """Doc ids that are not excluded, in the given order."""
        return [d for d in doc_ids if self.plans[d].status != "excluded"]

    def page_specs(self, doc_ids: Iterable[str]) -> list[PageSpec]:
        """One `PageSpec` per page of every included document."""
        out = []
        for d in self.included(doc_ids):
            g, plan = self.golds[d], self.plans[d]
            for k in range(plan.n_pages):
                out.append(PageSpec(d, d.split("_", 1)[0], k, plan.n_pages, g["pages"][k]))
        return out

    def _image(self, spec: PageSpec) -> Image.Image:
        root = paths.data_dir() if self.data_root is None else self.data_root
        with Image.open(root / spec.split / "images" / spec.image_name) as im:
            return im.copy()

    def render(self, spec: PageSpec, epoch: int, aug: AugmentConfig) -> RenderedPage:
        """Augmented image + target of one page for `epoch` (deterministic in all arguments)."""
        gold, plan = self.golds[spec.doc_id], self.plans[spec.doc_id]
        choices = select_occlusions(
            spec.doc_id, spec.n_pages, epoch, self.candidates.get(spec.doc_id, []),
            aug.occlusion_rate, aug.seed, aug.field_weights,
        )  # fmt: skip
        targets = page_targets(gold, plan, [c.field for c in choices])
        image = apply_occlusions(self._image(spec), spec.page_index, choices)
        degraded = False
        if spec.image_name.lower().endswith(".png") and self.scan_params is not None:
            rng = np.random.default_rng(
                sample_seed(aug.seed, spec.doc_id, spec.page_index, epoch, "scan")
            )
            if rng.random() < aug.scan_prob:
                image, degraded = A.scan_degrade(image, rng, self.scan_params), True
        return RenderedPage(
            spec,
            image.convert("RGB"),
            build_prompt(spec.page_index, spec.n_pages, "json"),
            targets[spec.page_index],
            tuple(c.field for c in choices if c.page == spec.page_index),
            degraded,
        )


# --------------------------------------------------------------------------------------------
# Preparation, audit log, summary
# --------------------------------------------------------------------------------------------


def prepare(
    golds: Mapping[str, Mapping[str, Any]],
    assignments: Mapping[str, tuple[Sequence[str], Sequence[int | None]]],
    *,
    ambiguous_policy: str = "header_only",
    monotone_boundary: bool = True,
    candidates: Mapping[str, list[OcclusionCandidate]] | None = None,
    data_root: Path | None = None,
    scan_params: A.ScanParams | None = None,
) -> PreparedSet:
    """Plans for every gold doc from its (levels, pages) row assignments."""
    plans = {}
    for d, g in golds.items():
        rows = g.get("line_items") or []
        levels, pages = assignments.get(d, ([], []))
        if rows and not levels:
            raise KeyError(f"{d}: {len(rows)} gold rows but no row assignments")
        plans[d] = plan_doc(
            g, levels, pages, ambiguous_policy=ambiguous_policy, monotone_boundary=monotone_boundary
        )
    return PreparedSet(
        {d: dict(g) for d, g in golds.items()},
        plans,
        dict(candidates or {}),
        data_root,
        scan_params,
    )


def audit_events(prepared: PreparedSet) -> list[dict[str, Any]]:
    """All audit events, sorted by doc id (values are never included)."""
    return [e for d in sorted(prepared.plans) for e in prepared.plans[d].events()]


def write_audit(prepared: PreparedSet, out_dir: Path | None = None) -> Path:
    """Write ``audit.jsonl`` (one event per line) under ``<runs dir>/trainset``; returns it."""
    out = (paths.runs_dir() / "trainset") if out_dir is None else Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "audit.jsonl"
    path.write_text(
        "".join(json.dumps(e) + "\n" for e in audit_events(prepared)),
        encoding="utf-8",
        newline="\n",
    )
    return path


def summarize(prepared: PreparedSet) -> dict[str, Any]:
    """Counts only: rows by source, documents by status, per split and in total."""
    out: dict[str, Any] = {"splits": {}}
    for d, plan in sorted(prepared.plans.items()):
        sp = d.split("_", 1)[0]
        s = out["splits"].setdefault(
            sp, {"docs": 0, "rows": 0, "status": Counter(), "row_sources": Counter(), "pages": 0}
        )
        s["docs"] += 1
        s["rows"] += plan.n_rows
        s["pages"] += plan.n_pages
        s["status"][plan.status] += 1
        s["row_sources"].update(plan.row_sources)
    for s in out["splits"].values():
        s["status"], s["row_sources"] = dict(s["status"]), dict(s["row_sources"])
    return out


# --------------------------------------------------------------------------------------------
# Round trip through the repo's merge (used by tests/test_trainset.py and the report)
# --------------------------------------------------------------------------------------------


def roundtrip_doc(
    gold: Mapping[str, Any], plan: DocPlan, scorer: Any | None = None
) -> dict[str, Any]:
    """Page targets -> ``merge_pages`` (defaults) -> ``normalize_doc`` -> compare with gold.

    Returns ``{"exact": bool, "overall": float | None}``; ``exact`` compares the normalised merged
    document with the gold header and rows directly, ``overall`` is the official scorer's
    document-level OVERALL when `scorer` (the loaded ``score.py``) is given.
    """
    from shipdoc.merge import merge_pages
    from shipdoc.normalize import normalize_doc

    pages = [t.payload for t in page_targets(gold, plan)]
    doc, _ = normalize_doc(merge_pages(pages).doc)
    want_header = {k: gold["header"].get(k) for k in doc["header"]}
    got_rows = [{k: r.get(k) for k in ROW_KEYS} for r in doc["line_items"]]
    want_rows = [{k: r.get(k) for k in ROW_KEYS} for r in gold.get("line_items") or []]
    exact = doc["doc_type"] == gold["doc_type"] and doc["header"] == want_header
    exact = exact and got_rows == want_rows
    overall = None
    if scorer is not None:
        res = scorer.score_doc(doc, dict(gold))
        overall = float(scorer.aggregate([res])["OVERALL"])
    return {"exact": bool(exact), "overall": overall}


def render_report_md(summary: Mapping[str, Any], extra: Mapping[str, Any]) -> str:
    """Markdown for ``reports/trainset.md`` (aggregate counts only, no values)."""
    lines = ["| split | docs | pages | rows | " + " | ".join(ROW_SOURCES) + " |",
             "|---|---:|---:|---:|" + "---:|" * len(ROW_SOURCES)]  # fmt: skip
    for sp, s in summary["splits"].items():
        cells = " | ".join(str(s["row_sources"].get(k, 0)) for k in ROW_SOURCES)
        lines.append(f"| {sp} | {s['docs']} | {s['pages']} | {s['rows']} | {cells} |")
    lines += ["", "| split | " + " | ".join(("ok", "header_only", "excluded")) + " |",
              "|---|---:|---:|---:|"]  # fmt: skip
    statuses = ("ok", "header_only", "excluded")
    for sp, s in summary["splits"].items():
        lines.append(f"| {sp} | " + " | ".join(str(s["status"].get(k, 0)) for k in statuses) + " |")
    for k, v in extra.items():
        lines.append(f"- {k}: {v}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------------
# Loading real data + CLI
# --------------------------------------------------------------------------------------------


def load_golds(doc_ids: Iterable[str], data_root: Path | None = None) -> dict[str, dict[str, Any]]:
    """Gold label dicts of `doc_ids` from ``<data dir>/<split>/labels``."""
    root = paths.data_dir() if data_root is None else Path(data_root)
    out = {}
    for d in doc_ids:
        p = root / d.split("_", 1)[0] / "labels" / f"{d}.json"
        out[d] = json.loads(p.read_text(encoding="utf-8"))
    return out


def load_folds(path: Path | None = None) -> dict[str, Any]:
    """``splits/folds.json``."""
    p = path or (paths.REPO_ROOT / "splits" / "folds.json")
    return dict(json.loads(p.read_text(encoding="utf-8")))


def build_prepared(
    doc_ids: Sequence[str],
    *,
    ambiguous_policy: str = "header_only",
    monotone_boundary: bool = True,
    with_candidates: bool = True,
    data_root: Path | None = None,
    scan_params: A.ScanParams | None = None,
) -> PreparedSet:
    """`PreparedSet` for real documents (needs the OCR cache and the official scorer)."""
    from shipdoc.ocr import doc_pages

    golds = load_golds(doc_ids, data_root)
    assignments: dict[str, tuple[list[str], list[int | None]]] = {}
    cands: dict[str, list[OcclusionCandidate]] = {}
    for d, g in golds.items():
        ocr = doc_pages(d)
        assignments[d] = assignments_from_ocr(g, ocr)
        if with_candidates:
            cands[d] = occlusion_candidates(g, ocr, max(1, len(g["pages"])))[0]
    return prepare(
        golds, assignments, ambiguous_policy=ambiguous_policy,
        monotone_boundary=monotone_boundary, candidates=cands, data_root=data_root,
        scan_params=scan_params,
    )  # fmt: skip


def save_prepared(prepared: PreparedSet, path: Path) -> None:
    """JSON cache of plans and occlusion candidates (no values, no images)."""
    blob = {
        "plans": {d: asdict(p) for d, p in prepared.plans.items()},
        "candidates": {d: [c.to_dict() for c in cs] for d, cs in prepared.candidates.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(blob), encoding="utf-8", newline="\n")
    tmp.replace(path)


def load_prepared(
    path: Path, doc_ids: Sequence[str], *, data_root: Path | None = None,
    scan_params: A.ScanParams | None = None,
) -> PreparedSet:  # fmt: skip
    """Inverse of `save_prepared` (labels are re-read from the data dir)."""
    blob = json.loads(path.read_text(encoding="utf-8"))
    plans = {
        d: DocPlan(
            p["doc_id"], p["n_pages"], tuple(p["row_pages"]), tuple(p["row_sources"]), p["status"]
        )
        for d, p in blob["plans"].items()
        if d in set(doc_ids)
    }
    cands = {
        d: [OcclusionCandidate.from_dict(c) for c in cs]
        for d, cs in blob["candidates"].items()
        if d in plans
    }
    return PreparedSet(load_golds(plans, data_root), plans, cands, data_root, scan_params)


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m shipdoc.trainset`` : build plans + candidates for a stage and write the audit."""
    ap = argparse.ArgumentParser(description=(main.__doc__ or "").strip())
    ap.add_argument("--stage", choices=STAGES, default="final")
    ap.add_argument("--all", action="store_true", help="plan all 500 train+dev docs (report)")
    ap.add_argument("--ambiguous-policy", choices=AMBIGUOUS_POLICIES, default="header_only")
    ap.add_argument("--no-monotone-boundary", action="store_true")
    ap.add_argument("--out", type=Path, default=None, help="cache JSON (default: runs dir)")
    args = ap.parse_args(argv)
    folds = load_folds()
    split = stage_split(args.stage, folds)
    ids = sorted({*split.train_ids, *split.heldout_ids}) if args.all else list(split.train_ids)
    prepared = build_prepared(
        ids, ambiguous_policy=args.ambiguous_policy, monotone_boundary=not args.no_monotone_boundary
    )
    out = args.out or (paths.runs_dir() / "trainset" / f"prepared_{args.stage}.json")
    save_prepared(prepared, out)
    audit = write_audit(prepared, out.parent)
    print(json.dumps(summarize(prepared), indent=1))
    print(f"prepared: {out}\naudit: {audit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
