"""Fine-tuning data builder: row-to-page policy, per-page targets, augmentation, stage splits.

The synthetic tests use invented values only and run everywhere. The ``real data`` tests need
``data/``, the OCR cache / provenance output and the official scorer and skip cleanly without
them; they print counts, never values.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from shipdoc import augment as A
from shipdoc import paths
from shipdoc import targets as T
from shipdoc import trainset as ts
from shipdoc.extract import HEADER_KEYS, ROW_KEYS, schema_errors

ROOT = Path(__file__).resolve().parents[1]
HAVE_SCORER = (ROOT / "assignment" / "score.py").is_file()
needs_scorer = pytest.mark.skipif(not HAVE_SCORER, reason="assignment/score.py absent")
LOCATIONS = paths.runs_dir() / "provenance" / "locations.jsonl"
needs_real = pytest.mark.skipif(
    not (HAVE_SCORER and (ROOT / "data" / "train" / "labels").is_dir() and LOCATIONS.is_file()),
    reason="data/, assignment/score.py or provenance/locations.jsonl absent",
)


# --------------------------------------------------------------------------------------------
# Synthetic documents (invented values)
# --------------------------------------------------------------------------------------------


def make_gold(
    doc_id: str = "train_9000", n_pages: int = 2, n_rows: int = 5, doc_type: str = "invoice",
    ext: str = "png",
) -> dict[str, Any]:  # fmt: skip
    if doc_type == "waybill":
        header: dict[str, Any] = {
            "carrier": "Zeta Air", "mawb": "123-45678901", "hawb": None,
            "origin_airport": "AAA", "destination_airport": "BBB", "shipper_name": "Shipper One",
            "consignee_name": "Consignee Two", "pieces": "7", "gross_weight_kg": "42.5",
        }  # fmt: skip
        rows: list[dict[str, Any]] = []
    else:
        header = {
            "invoice_number": f"INV-{doc_id[-4:]}", "invoice_date": "2026-03-04",
            "supplier_name": "Acme Parts Ltd", "buyer_name": "Buyer GmbH",
            "ship_to_name": "Buyer GmbH Dock 2", "currency": "USD", "total_amount": "1234.50",
            "awb_number": None,
        }  # fmt: skip
        rows = [
            {
                "supplier_part_number": f"SP-{i:03d}",
                "customer_part_number": None if i % 2 else f"CP-{i}",
                "purchase_order": f"PO{i}",
                "quantity": str(10 * (i + 1)),
            }
            for i in range(n_rows)
        ]
    return {
        "doc_id": doc_id, "doc_type": doc_type, "header": header, "line_items": rows,
        "pages": [f"{doc_id}_p{k + 1}.{ext}" for k in range(n_pages)],
    }  # fmt: skip


def split_assignment(n_rows: int, n_pages: int) -> tuple[list[str], list[int | None]]:
    """Every row ``line`` level, contiguous chunks over the pages."""
    base, extra = divmod(n_rows, n_pages)
    pages = [p for p in range(n_pages) for _ in range(base + (1 if p < extra else 0))]
    return ["line"] * n_rows, pages


SYNTH = [
    ("train_9001", 1, 4, "invoice"),
    ("train_9002", 2, 5, "invoice"),
    ("train_9003", 3, 7, "invoice"),
    ("train_9004", 1, 0, "waybill"),
    ("train_9005", 2, 1, "invoice"),
]


def synth_prepared(**kw: Any) -> ts.PreparedSet:
    golds, asg = {}, {}
    for d, n_pages, n_rows, kind in SYNTH:
        golds[d] = make_gold(d, n_pages, n_rows, kind)
        asg[d] = split_assignment(n_rows, n_pages)
    return ts.prepare(golds, asg, **kw)


# --------------------------------------------------------------------------------------------
# Row -> page policy
# --------------------------------------------------------------------------------------------


def test_single_page_unassigned_goes_to_page_zero() -> None:
    r = ts.resolve_row_pages(["line", "unassigned", "unassigned"], [0, None, None], 1)
    assert r.row_pages == (0, 0, 0)
    assert r.sources == ("line", "single_page_default", "single_page_default")


def test_interpolation_needs_equal_neighbours() -> None:
    lv = ["line", "unassigned", "unassigned", "line", "unassigned", "line"]
    pg = [0, None, None, 0, None, 1]
    r = ts.resolve_row_pages(lv, pg, 2)
    assert r.row_pages == (0, 0, 0, 0, None, 1)  # 0..0 pinned, 0..1 ambiguous
    assert r.sources[1] == r.sources[2] == "interpolated" and r.sources[4] == "ambiguous"


def test_page_fuzzy_rows_keep_their_page_and_count_as_neighbours() -> None:
    r = ts.resolve_row_pages(["page_fuzzy", "unassigned", "page_fuzzy"], [1, None, 1], 2)
    assert r.row_pages == (1, 1, 1) and r.sources[0] == "page_fuzzy"


def test_monotone_boundary_pins_edges_only_when_the_page_is_forced() -> None:
    lv = ["unassigned", "line", "line", "unassigned"]
    assert ts.resolve_row_pages(lv, [None, 0, 1, None], 2).row_pages == (0, 0, 1, 1)
    off = ts.resolve_row_pages(lv, [None, 0, 1, None], 2, monotone_boundary=False)
    assert off.row_pages == (None, 0, 1, None) and off.n_ambiguous == 2
    mid = ts.resolve_row_pages(lv, [None, 1, 1, None], 3)  # leading row could be page 0 or 1
    assert mid.row_pages == (None, 1, 1, None)  # next is on page 1, not 0: not forced


def test_no_assigned_neighbour_in_multipage_is_ambiguous() -> None:
    r = ts.resolve_row_pages(["unassigned", "unassigned"], [None, None], 2)
    assert r.row_pages == (None, None)


def test_assigned_page_out_of_range_is_an_error() -> None:
    with pytest.raises(ValueError, match="outside"):
        ts.resolve_row_pages(["line"], [3], 2)
    with pytest.raises(ValueError, match="differ in length"):
        ts.resolve_row_pages(["line"], [0, 1], 2)


def test_plan_status_follows_policy() -> None:
    g = make_gold("train_9100", 2, 3)
    lv, pg = ["line", "unassigned", "line"], [0, None, 1]
    assert ts.plan_doc(g, lv, pg).status == "header_only"
    assert ts.plan_doc(g, lv, pg, ambiguous_policy="exclude").status == "excluded"
    ok = ts.plan_doc(g, ["line", "line", "line"], [0, 0, 1])
    assert ok.status == "ok" and ok.supervise_line_items
    with pytest.raises(ValueError, match="ambiguous_policy"):
        ts.plan_doc(g, lv, pg, ambiguous_policy="drop")
    with pytest.raises(ValueError, match="row assignments"):
        ts.plan_doc(g, ["line"], [0])


def test_missing_assignments_for_a_doc_with_rows_fail_loudly() -> None:
    with pytest.raises(KeyError, match="no row assignments"):
        ts.prepare({"train_9000": make_gold()}, {})


def test_audit_logs_every_decision_and_never_a_value(tmp_path: Path) -> None:
    g = make_gold("train_9200", 2, 4)
    asg = {"train_9200": (["line", "unassigned", "unassigned", "line"], [0, None, 1, 1])}
    prepared = ts.prepare(
        {"train_9200": g}, asg
    )  # row 1 pinned? prev 0, next none-> row2 ambiguous
    path = ts.write_audit(prepared, tmp_path)
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    kinds = {(e["event"], e["row_idx"]) for e in events}
    assert ("ambiguous", 1) in kinds and ("doc_header_only", None) in kinds
    text = path.read_text(encoding="utf-8")
    for v in (*g["header"].values(), *(r["supplier_part_number"] for r in g["line_items"])):
        assert v is None or v not in text


# --------------------------------------------------------------------------------------------
# Targets
# --------------------------------------------------------------------------------------------


def test_render_page_text_equals_render_target_and_span_is_the_row_array() -> None:
    prepared = synth_prepared()
    for d, plan in prepared.plans.items():
        for t in ts.page_targets(prepared.golds[d], plan):
            assert t.text == T.render_target(t.payload)
            assert json.loads(t.text[slice(*t.line_items_span)]) == t.payload["line_items"]
            assert schema_errors(t.payload) == []
            assert list(t.payload["header"]) == list(HEADER_KEYS)  # declared key order
            assert all(list(r) == list(ROW_KEYS) for r in t.payload["line_items"])


def test_page_two_banner_trap_continuation_pages_have_null_identity_fields() -> None:
    prepared = synth_prepared()
    pages = ts.page_targets(prepared.golds["train_9003"], prepared.plans["train_9003"])
    assert [t.payload["page_kind"] for t in pages] == ["first", "continuation", "continuation"]
    assert pages[0].payload["header"]["invoice_number"] is not None
    assert all(t.payload["header"]["invoice_number"] is None for t in pages[1:])
    assert pages[-1].payload["header"]["total_amount"] == "1234.50"
    assert all(t.payload["header"]["total_amount"] is None for t in pages[:-1])


def test_header_only_doc_drops_unplaceable_rows_and_masks_line_items() -> None:
    g = make_gold("train_9300", 2, 3)
    asg = {"train_9300": (["line", "unassigned", "line"], [0, None, 1])}
    prepared = ts.prepare({"train_9300": g}, asg)
    plan = prepared.plans["train_9300"]
    assert plan.status == "header_only" and not plan.supervise_line_items
    tgts = ts.page_targets(g, plan)
    assert [len(t.payload["line_items"]) for t in tgts] == [1, 1]  # row 1 left out
    assert not any(t.supervise_line_items for t in tgts)


def test_excluded_doc_has_no_targets() -> None:
    g = make_gold("train_9301", 2, 3)
    asg = {"train_9301": (["line", "unassigned", "line"], [0, None, 1])}
    plan = ts.prepare({"train_9301": g}, asg, ambiguous_policy="exclude").plans["train_9301"]
    with pytest.raises(ValueError, match="excluded"):
        ts.page_targets(g, plan)


def test_nulled_field_is_null_on_its_page_only_and_gold_is_untouched() -> None:
    prepared = synth_prepared()
    g = prepared.golds["train_9002"]
    before = json.dumps(g, sort_keys=True)
    pages = ts.page_targets(g, prepared.plans["train_9002"], ["invoice_number"])
    assert all(t.payload["header"]["invoice_number"] is None for t in pages)
    assert json.dumps(g, sort_keys=True) == before
    with pytest.raises(ValueError, match="already null"):
        ts.page_targets(g, prepared.plans["train_9002"], ["awb_number"])


# --------------------------------------------------------------------------------------------
# V2b (1): round trip through the repo's merge
# --------------------------------------------------------------------------------------------


def test_synthetic_docs_round_trip_through_merge_exactly() -> None:
    prepared = synth_prepared()
    for d, plan in prepared.plans.items():
        assert plan.status == "ok"
        assert ts.roundtrip_doc(prepared.golds[d], plan)["exact"], d


@needs_scorer
def test_synthetic_docs_round_trip_under_the_official_scorer() -> None:
    from shipdoc.eval import load_scorer

    sc = load_scorer()
    prepared = synth_prepared()
    for d, plan in prepared.plans.items():
        assert ts.roundtrip_doc(prepared.golds[d], plan, sc)["overall"] == 1.0, d


@needs_real
def test_every_train_doc_round_trips_under_the_official_scorer(
    capsys: pytest.CaptureFixture,
) -> None:
    from shipdoc.eval import load_scorer

    sc = load_scorer()
    loc_asg = ts.assignments_from_locations(LOCATIONS)
    ids = ts.stage_split("final", ts.load_folds()).train_ids
    golds = ts.load_golds(ids)
    asg = {d: loc_asg.get(d, ([], [])) for d in ids}
    prepared = ts.prepare(golds, asg)
    n_ok = n_header_only = 0
    bad: list[str] = []
    for d in ids:
        plan = prepared.plans[d]
        if plan.status != "ok":
            n_header_only += plan.status == "header_only"
            assert plan.status == "header_only"  # default policy never excludes
            continue
        r = ts.roundtrip_doc(golds[d], plan, sc)
        n_ok += 1
        if not (r["exact"] and r["overall"] == 1.0):
            bad.append(d)
    with capsys.disabled():
        print(f"\ntrain docs {len(ids)}: round-trip OVERALL==1.0 {n_ok - len(bad)}/{n_ok}, "
              f"header-only (skipped) {n_header_only}")  # fmt: skip
    assert not bad, f"{len(bad)} docs do not round-trip: {bad[:5]}"
    assert (n_ok, n_header_only) == (395, 5)  # frozen dataset: V1 counts with the boundary rule


@needs_real
def test_header_only_docs_still_round_trip_their_header_and_known_rows() -> None:
    loc_asg = ts.assignments_from_locations(LOCATIONS)
    ids = ts.stage_split("final", ts.load_folds()).train_ids
    golds = ts.load_golds(ids)
    prepared = ts.prepare(golds, {d: loc_asg.get(d, ([], [])) for d in ids})
    from shipdoc.merge import merge_pages

    for d in ids:
        plan = prepared.plans[d]
        if plan.status != "header_only":
            continue
        merged = merge_pages([t.payload for t in ts.page_targets(golds[d], plan)]).doc
        known = [
            r for r, p in zip(golds[d]["line_items"], plan.row_pages, strict=True) if p is not None
        ]
        assert merged["header"] == {k: golds[d]["header"].get(k) for k in merged["header"]}
        assert len(merged["line_items"]) == len(known) < len(golds[d]["line_items"])


@needs_real
def test_ocr_assignments_equal_the_provenance_table() -> None:
    from shipdoc.ocr import doc_pages

    loc_asg = ts.assignments_from_locations(LOCATIONS)
    ids = ts.stage_split("final", ts.load_folds()).train_ids[:25]
    golds = ts.load_golds(ids)
    for d in ids:
        if golds[d]["line_items"]:
            assert ts.assignments_from_ocr(golds[d], doc_pages(d)) == loc_asg[d]


# --------------------------------------------------------------------------------------------
# Augmentation: determinism, rate, coverage
# --------------------------------------------------------------------------------------------

SCAN = A.ScanParams.from_dict({
    "rotate_deg": 1.5, "paper_gain": [0.93, 1.0], "paper_gain_skew": 1.0, "ink_lift": [10, 25],
    "blur_sigma": [0.1, 1.0], "noise_sigma": [1.0, 6.0], "jpeg_quality": [50, 60],
})  # fmt: skip
SIZE = (320, 420)
BOX_INV = (40.0, 40.0, 150.0, 58.0)  # invoice_number, page 0
BOX_BANNER = (40.0, 20.0, 150.0, 36.0)  # the same value printed in a page-2 banner
BOX_TOTAL = (200.0, 300.0, 290.0, 318.0)  # total_amount, last page


def _page_image(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = np.full((SIZE[1], SIZE[0]), 245, np.uint8)
    for y in range(10, SIZE[1] - 10, 22):  # text-like stripes everywhere, boxes included
        arr[y : y + 10, rng.integers(5, SIZE[0] - 5, size=180)] = 20
    return Image.fromarray(arr)


def disk_prepared(tmp_path: Path, n_docs: int = 8, ext: str = "png") -> ts.PreparedSet:
    """Two-page invoices on disk, with hand-written occlusion candidates."""
    golds, asg, cands = {}, {}, {}
    for i in range(n_docs):
        d = f"train_{9400 + i}"
        g = make_gold(d, 2, 4, ext=ext)
        golds[d] = g
        asg[d] = split_assignment(4, 2)
        cands[d] = [
            ts.OcclusionCandidate("invoice_number", 0, ((0, BOX_INV), (1, BOX_BANNER))),
            ts.OcclusionCandidate("total_amount", 1, ((1, BOX_TOTAL),)),
        ]
        (tmp_path / "train" / "images").mkdir(parents=True, exist_ok=True)
        for k, name in enumerate(g["pages"]):
            _page_image(100 * i + k).save(tmp_path / "train" / "images" / name)
    return ts.prepare(golds, asg, candidates=cands, data_root=tmp_path, scan_params=SCAN)


def _diff(a: Image.Image, b: Image.Image, box: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = (int(v) for v in box)
    pa = np.asarray(a.convert("L"), float)[y0:y1, x0:x1]
    pb = np.asarray(b.convert("L"), float)[y0:y1, x0:x1]
    return float(np.abs(pa - pb).mean())


def test_same_seed_same_bytes_different_seed_different(tmp_path: Path) -> None:
    prepared = disk_prepared(tmp_path)
    aug = ts.AugmentConfig(occlusion_rate=0.5, scan_prob=0.5, seed=42)
    specs = prepared.page_specs(prepared.plans)
    a = [prepared.render(s, 0, aug) for s in specs]
    b = [prepared.render(s, 0, aug) for s in specs]
    assert all(x.image.tobytes() == y.image.tobytes() and x.target.text == y.target.text
               for x, y in zip(a, b, strict=True))  # fmt: skip
    other = [prepared.render(s, 0, ts.AugmentConfig(0.5, 0.5, 43)) for s in specs]
    assert any(x.image.tobytes() != y.image.tobytes() for x, y in zip(a, other, strict=True))
    later = [prepared.render(s, 1, aug) for s in specs]  # a new epoch draws new augmentations
    assert any(x.image.tobytes() != y.image.tobytes() for x, y in zip(a, later, strict=True))


def test_digital_pages_are_degraded_scans_are_not(tmp_path: Path) -> None:
    aug = ts.AugmentConfig(occlusion_rate=0.0, scan_prob=1.0, seed=42)
    png = disk_prepared(tmp_path / "png", ext="png")
    spec = png.page_specs(["train_9400"])[0]
    r = png.render(spec, 0, aug)
    assert r.degraded and r.image.mode == "RGB" and r.image.size == SIZE
    jpg = disk_prepared(tmp_path / "jpg", ext="jpg")
    spec = jpg.page_specs(["train_9400"])[0]
    r = jpg.render(spec, 0, aug)
    assert not r.degraded
    orig = Image.open(tmp_path / "jpg" / "train" / "images" / spec.image_name).convert("RGB")
    assert r.image.tobytes() == orig.tobytes()  # a scan is passed through untouched


def test_scan_prob_is_respected(tmp_path: Path) -> None:
    prepared = disk_prepared(tmp_path, n_docs=8)
    specs = prepared.page_specs(prepared.plans)
    aug = ts.AugmentConfig(occlusion_rate=0.0, scan_prob=0.5, seed=42)
    share = np.mean([prepared.render(s, e, aug).degraded for s in specs for e in range(10)])
    assert 0.35 < share < 0.65


def test_realised_occlusion_rate_is_close_to_the_stated_rate(tmp_path: Path) -> None:
    prepared = disk_prepared(tmp_path, n_docs=8)
    specs = prepared.page_specs(prepared.plans)
    rate = 0.15
    aug = ts.AugmentConfig(occlusion_rate=rate, scan_prob=0.0, seed=42)
    hits = [bool(prepared.render(s, e, aug).occluded) for s in specs for e in range(25)]
    assert len(hits) == 400
    assert abs(np.mean(hits) - rate) < 0.05  # sd = 0.018 at n=400: 2.8 sigma, and it is seeded


def test_occluded_field_is_null_only_where_the_box_is_covered(tmp_path: Path) -> None:
    prepared = disk_prepared(tmp_path, n_docs=4)
    aug_on = ts.AugmentConfig(occlusion_rate=1.0, scan_prob=0.0, seed=42)
    aug_off = ts.AugmentConfig(occlusion_rate=0.0, scan_prob=0.0, seed=42)
    for d in prepared.plans:
        gold_number = prepared.golds[d]["header"]["invoice_number"]
        for spec in prepared.page_specs([d]):
            clean = prepared.render(spec, 0, aug_off)
            hot = prepared.render(spec, 0, aug_on)
            assert clean.occluded == ()
            header = hot.target.payload["header"]
            if spec.page_index == 0:  # one candidate per page: page 0 hides the invoice number
                assert hot.occluded == ("invoice_number",)
                assert clean.target.payload["header"]["invoice_number"] == gold_number
                assert header["invoice_number"] is None
                assert _diff(clean.image, hot.image, BOX_INV) > 40  # the value region changed
                assert _diff(clean.image, hot.image, (0, 200, 320, 280)) == 0  # far away: intact
            else:  # the last page hides the total
                assert hot.occluded == ("total_amount",)
                assert clean.target.payload["header"]["total_amount"] == "1234.50"
                assert header["total_amount"] is None
                assert _diff(clean.image, hot.image, BOX_TOTAL) > 40
                assert header["invoice_number"] is None  # continuation page: always null


def test_a_choice_made_on_page_zero_also_hides_the_banner_copy_on_page_one(
    tmp_path: Path,
) -> None:
    prepared = disk_prepared(tmp_path, n_docs=1)
    d = "train_9400"
    choices = [
        c for e in range(40) for c in ts.select_occlusions(d, 2, e, prepared.candidates[d], 1.0)
        if c.field == "invoice_number"
    ]  # fmt: skip
    assert choices and all({p for p, _ in c.boxes} == {0, 1} for c in choices)
    spec1 = prepared.page_specs([d])[1]
    img = prepared._image(spec1)
    out = ts.apply_occlusions(img, 1, choices[:1])
    assert _diff(img, out, BOX_BANNER) > 40 and _diff(img, out, (0, 200, 320, 280)) == 0


def test_page_without_candidates_is_never_nulled_even_at_rate_one(tmp_path: Path) -> None:
    prepared = disk_prepared(tmp_path, n_docs=2)
    prepared.candidates = {d: [] for d in prepared.plans}
    aug = ts.AugmentConfig(occlusion_rate=1.0, scan_prob=0.0, seed=42)
    for s in prepared.page_specs(prepared.plans):
        r = prepared.render(s, 0, aug)
        assert r.occluded == ()
        full = ts.page_targets(prepared.golds[s.doc_id], prepared.plans[s.doc_id])[s.page_index]
        assert r.target.payload == full.payload


def test_select_occlusions_follows_the_candidates_declared_target_page() -> None:
    wrong = ts.OcclusionCandidate("invoice_number", 1, ((1, BOX_BANNER),))  # identity -> page 0
    assert ts.carrying_page("invoice", "invoice_number", 3) == 0
    assert ts.carrying_page("invoice", "total_amount", 3) == 2
    assert ts.carrying_page("waybill", "gross_weight_kg", 1) == 0
    got = ts.select_occlusions("train_9500", 2, 0, [wrong], 1.0)
    assert [c.field for c in got] == ["invoice_number"]  # page 1 carries it per the candidate
    assert ts.select_occlusions("train_9500", 2, 0, [wrong], 0.0) == []


def test_candidates_roundtrip_json() -> None:
    c = ts.OcclusionCandidate("invoice_number", 0, ((0, BOX_INV), (1, BOX_BANNER)))
    assert ts.OcclusionCandidate.from_dict(json.loads(json.dumps(c.to_dict()))) == c


def test_widest_applied_box_bounds_every_method_box() -> None:
    """The collateral check is sound: the real applied box of each method fits the bound."""
    bound = ts.widest_applied_box(BOX_INV)
    rng = np.random.default_rng(0)
    for method in ts.OCCLUSION_METHODS:
        for _ in range(50):
            ab = A.plan_box(BOX_INV, method, rng, (1240, 1754))
            assert ab[0] >= bound[0] - 1 and ab[1] >= bound[1] - 1
            assert ab[2] <= bound[2] + 1 and ab[3] <= bound[3] + 1


@needs_scorer
def test_occlusion_candidates_from_ocr_reasons() -> None:
    from shipdoc.ocr import OcrItem, PageOcr

    def item(text: str, x0: float, y0: float, x1: float, y1: float) -> OcrItem:
        return OcrItem(text, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], 0.9)

    lines = [  # invented values; 24 px high lines
        ("Invoice No: INV-7788", 100), ("Acme Parts Ltd", 150),
        ("Buyer GmbH", 300), ("Warehouse Nine", 330),  # 6 px apart: hiding one grazes the other
    ]  # fmt: skip
    items = [item(t, 50, y, 50 + 14 * len(t), y + 24) for t, y in lines]
    page = PageOcr("x_p1.png", 1240, 1754, "paddleocr", "line", 0, items=items)
    gold = {
        "doc_id": "train_9600", "doc_type": "invoice", "pages": ["x_p1.png"], "line_items": [],
        "header": {
            "invoice_number": "INV-7788", "invoice_date": None, "supplier_name": "Acme Parts Ltd",
            "buyer_name": "Buyer GmbH", "ship_to_name": "Warehouse Nine", "currency": None,
            "total_amount": None, "awb_number": "AWB-NOT-THERE",
        },
    }  # fmt: skip
    cands, reasons = ts.occlusion_candidates(gold, [page], 1)
    assert {c.field for c in cands} == {"invoice_number", "supplier_name"}
    assert dict(reasons) == {"eligible": 2, "collateral": 2, "null_gold": 3, "no_box": 1}
    inv = next(c for c in cands if c.field == "invoice_number")
    assert inv.target_page == 0 and [p for p, _ in inv.boxes] == [0]


# --------------------------------------------------------------------------------------------
# V2b (4)(5): fold isolation and leakage
# --------------------------------------------------------------------------------------------

FOLDS = {
    "k": 3,
    "folds": [
        {"fold": 0, "val_doc_ids": ["dev_0000", "train_0000", "train_0003"]},
        {"fold": 1, "val_doc_ids": ["dev_0001", "train_0001", "train_0004"]},
        {"fold": 2, "val_doc_ids": ["dev_0002", "train_0002", "train_0005"]},
    ],
}


def test_fold_stage_trains_only_on_the_other_folds() -> None:
    everything = {d for f in FOLDS["folds"] for d in f["val_doc_ids"]}
    for k in range(3):
        sp = ts.stage_split(f"fold{k}", FOLDS)
        val = set(FOLDS["folds"][k]["val_doc_ids"])
        assert set(sp.heldout_ids) == val
        assert not set(sp.train_ids) & val
        assert set(sp.train_ids) | val == everything
        assert all(not d.startswith("test_") for d in sp.train_ids)


def test_held_out_docs_never_reach_the_page_specs() -> None:
    golds = {d: make_gold(d, 1, 2) for f in FOLDS["folds"] for d in f["val_doc_ids"]}
    prepared = ts.prepare(golds, {d: split_assignment(2, 1) for d in golds})
    for k in range(3):
        sp = ts.stage_split(f"fold{k}", FOLDS)
        specs = prepared.page_specs(sp.train_ids)
        assert {s.doc_id for s in specs} == set(sp.train_ids)
        assert not {s.doc_id for s in specs} & set(sp.heldout_ids)


def test_final_and_smoke_never_train_on_dev() -> None:
    for stage in ("final", "smoke"):
        sp = ts.stage_split(stage, FOLDS)
        assert all(d.startswith("train_") for d in sp.train_ids)
        assert all(d.startswith("dev_") for d in sp.heldout_ids)
        assert len(sp.train_ids) == 6 and len(sp.heldout_ids) == 3


def test_leakage_guard_rejects_overlap_test_and_dev_in_final() -> None:
    with pytest.raises(ValueError, match="AND held-out"):
        ts.assert_no_leakage(ts.StageSplit("fold0", ("train_1", "train_2"), ("train_2",)))
    with pytest.raises(ValueError, match="test docs"):
        ts.assert_no_leakage(ts.StageSplit("fold0", ("test_0001",), ("train_2",)))
    with pytest.raises(ValueError, match="dev"):
        ts.assert_no_leakage(ts.StageSplit("final", ("dev_0001",), ()))
    with pytest.raises(ValueError, match="unknown stage"):
        ts.stage_split("fold9", FOLDS)


def test_manifest_hash_is_stable_and_stage_specific() -> None:
    a, b = ts.stage_split("fold0", FOLDS), ts.stage_split("fold1", FOLDS)
    assert a.manifest_hash() == ts.stage_split("fold0", FOLDS).manifest_hash()
    assert a.manifest_hash() != b.manifest_hash()


@pytest.mark.skipif(not (ROOT / "splits" / "folds.json").is_file(), reason="folds.json absent")
def test_real_folds_partition_the_500_docs_without_test_docs() -> None:
    folds = ts.load_folds()
    everything = {d for f in folds["folds"] for d in f["val_doc_ids"]}
    assert len(everything) == 500 and not any(d.startswith("test_") for d in everything)
    final = ts.stage_split("final", folds)
    assert len(final.train_ids) == 400 and len(final.heldout_ids) == 100
    for k in range(3):
        sp = ts.stage_split(f"fold{k}", folds)
        assert not set(sp.train_ids) & set(sp.heldout_ids)
        assert len(sp.train_ids) + len(sp.heldout_ids) == 500


# --------------------------------------------------------------------------------------------
# Cache and report
# --------------------------------------------------------------------------------------------


def test_prepared_cache_roundtrip(tmp_path: Path) -> None:
    data = tmp_path / "data"
    prepared = disk_prepared(data, n_docs=2)
    for d, g in prepared.golds.items():
        (data / "train" / "labels").mkdir(parents=True, exist_ok=True)
        (data / "train" / "labels" / f"{d}.json").write_text(json.dumps(g), encoding="utf-8")
    ts.save_prepared(prepared, tmp_path / "p.json")
    back = ts.load_prepared(tmp_path / "p.json", list(prepared.plans), data_root=data)
    assert back.plans == prepared.plans and back.candidates == prepared.candidates
    assert back.golds == prepared.golds


def test_summary_counts_and_report_have_no_values() -> None:
    prepared = synth_prepared()
    s = ts.summarize(prepared)
    train = s["splits"]["train"]
    assert train["docs"] == 5 and train["rows"] == 17 and train["pages"] == 9
    assert train["row_sources"] == {"line": 17} and train["status"] == {"ok": 5}
    md = ts.render_report_md(s, {"note": 1})
    assert "| train | 5 | 9 | 17 |" in md and "Acme" not in md


def test_field_weights_bias_the_draw_and_are_validated() -> None:
    cands = [
        ts.OcclusionCandidate("invoice_number", 0, ((0, BOX_INV),)),
        ts.OcclusionCandidate("supplier_name", 0, ((0, (40.0, 90.0, 150.0, 108.0)),)),
    ]

    def share(weights: dict[str, float] | None) -> float:
        picks = [
            c.field
            for e in range(400)
            for c in ts.select_occlusions("train_9501", 1, e, cands, 1.0, 42, weights)
        ]
        return picks.count("invoice_number") / len(picks)

    assert abs(share(None) - 0.5) < 0.08  # uniform
    assert abs(share({"invoice_number": 4.0}) - 0.8) < 0.08  # 4 : 1
    assert share({"invoice_number": 0.0}) == 0.0  # weight 0 switches a field off
    with pytest.raises(ValueError, match="weights"):
        ts.select_occlusions("train_9501", 1, 0, cands, 1.0, 42, {"invoice_number": -1.0})
    with pytest.raises(ValueError, match="weights"):
        ts.select_occlusions(
            "train_9501", 1, 0, cands, 1.0, 42, {"invoice_number": 0.0, "supplier_name": 0.0}
        )
    assert ts.AugmentConfig().field_weights == {"invoice_number": 4.0, "invoice_date": 4.0}
