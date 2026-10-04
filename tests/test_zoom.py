"""Tests for the zoom re-read (shipdoc.zoom): geometry, selection, acceptance, accounting.

No GPU and no model: a scripted backend answers the one-field / one-row schemas. Tests that use the
real OCR locator need the official scorer (`shipdoc.locate` loads it) and skip without it.
"""

from __future__ import annotations

import ast
import builtins
import importlib.util
import inspect
import io
import json
import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from PIL import Image, ImageChops, ImageDraw

from shipdoc import extract as ex
from shipdoc import prompts
from shipdoc import zoom as Z
from shipdoc.ocr import OcrItem, PageOcr

ROOT = Path(__file__).resolve().parents[1]
needs_scorer = pytest.mark.skipif(
    not (ROOT / "assignment" / "score.py").is_file(), reason="assignment/score.py absent"
)
W, H = 1240, 1754


def _item(text: str, x0: float, y0: float, x1: float, y1: float) -> OcrItem:
    return OcrItem(text, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], 0.9)


def _page() -> PageOcr:
    """Lines (reading order): 0 name, 1 invoice no, 2 table headings, 3..5 rows, 6 total."""
    items = [
        _item("Acme Trading Co", 50, 100, 300, 120),
        _item("Invoice No.: INV-1", 50, 150, 400, 170),
        _item("Line Item Part Qty Description", 50, 400, 900, 420),
        _item("1 P-0 PO-0 5", 50, 450, 700, 470),
        _item("2 P-1 PO-1 10", 50, 500, 700, 520),
        _item("3 P-2 PO-2 7", 50, 550, 700, 570),
        _item("Total 100.00", 50, 900, 300, 920),
    ]
    return PageOcr("d_0000_p1.png", W, H, "paddleocr", "line", 0, items=items)


def _image() -> Image.Image:
    img = Image.new("RGB", (W, H), "white")
    ImageDraw.Draw(img).rectangle((60, 505, 600, 515), fill=(200, 10, 10))
    return img


def _cfg(**kw: Any) -> Z.ZoomConfig:
    return Z.ZoomConfig(**{"enabled": True, "tau": 0.9, **kw})


# ------------------------------------------------------------------------------- geometry


def test_pad_box_fractions_min_size_and_clipping() -> None:
    assert Z.pad_box((100, 100, 200, 140), 0.1, 0.5, W, H) == (90, 80, 210, 160)
    # padding past the page edge is clipped, never shifted inwards
    assert Z.pad_box((5, 5, 105, 25), 0.1, 1.0, W, H, 256, 64) == (0, 0, 183, 47)
    # minimum size grows around the centre
    assert Z.pad_box((100, 100, 110, 110), 0, 0, W, H, 40, 20) == (85, 95, 125, 115)
    assert Z.pad_box((1200, 1700, 1240, 1754), 1.0, 1.0, W, H)[2:] == (W, H)


def test_union_boxes() -> None:
    assert Z.union_boxes([(1, 2, 3, 4), (0, 5, 2, 9)]) == (0, 2, 3, 9)
    with pytest.raises(ValueError):
        Z.union_boxes([])


def test_line_box_is_union_of_the_lines_items() -> None:
    page = _page()
    page.items.append(_item("extra", 720, 502, 800, 518))  # same visual line as row 2
    assert Z.line_box(page, 4) == (50, 500, 800, 520)
    assert Z.line_box(page, 99) is None


def test_row_band_table_width_padding_and_heading_strip() -> None:
    page = _page()
    plan = Z.plan_row_crop([page], Z.LineHit(0, 4, (50, 500, 700, 520)), _cfg())
    assert plan is not None and plan.heading_strip and plan.source == "row_line"
    strip, band = plan.boxes
    # table x extent (50..700, headings line excluded) +2% each side; y 500..520 +50% -> grown
    # to the 64 px minimum around the centre 510
    assert band == (37, 478, 713, 542)
    assert strip == (37, 378, 713, 442)  # headings line 400..420, same width, ends above the band
    assert strip[3] <= band[1]


def test_row_band_page_width_and_no_heading_strip() -> None:
    plan = Z.plan_row_crop(
        [_page()],
        Z.LineHit(0, 4, (50, 500, 700, 520)),
        _cfg(row_band_width="page", include_table_header=False),
    )
    assert plan is not None and not plan.heading_strip
    (band,) = plan.boxes
    assert band[0] == 0 and band[2] == W and band[1:4:2] == (478, 542)


def test_row_band_without_a_located_row_is_none() -> None:
    assert Z.plan_row_crop([_page()], None, _cfg()) is None


def test_header_crop_uses_the_whole_matched_line_with_label_context() -> None:
    hit = Z.LineHit(0, 1, (200, 150, 300, 170))  # only the value part; the line has the label
    plan = Z.plan_header_crop([_page()], hit, 0, "invoice_number", _cfg())
    assert plan is not None and plan.source == "ocr_match"
    assert plan.boxes == ((15, 128, 435, 192),)


def test_header_crop_falls_back_to_layout_region_and_never_to_gold() -> None:
    page = _page()
    head = Z.plan_header_crop([page], None, 0, "invoice_number", _cfg())
    foot = Z.plan_header_crop([page], None, 0, "total_amount", _cfg())
    assert head is not None and head.source == "region_fallback"
    assert head.boxes[0][1] < 100 and head.boxes[0][3] > 170  # header lines 100..170 + pad
    assert foot is not None and foot.boxes[0][1] <= 900 and foot.boxes[0][3] >= 920
    empty = PageOcr("e.png", W, H, "paddleocr", "line", 0, items=[])
    assert Z.plan_header_crop([empty], None, 0, "invoice_number", _cfg()) is None


def test_render_crop_native_resolution_no_resize() -> None:
    img = _image()
    plan = Z.CropPlan(0, ((37.0, 478.0, 713.0, 542.0),), "row_line")
    out = Z.render_crop(img, plan, W, H, Z.DEFAULT_MAX_PIXELS)
    assert out.size == (676, 64)
    assert ImageChops.difference(out, img.crop((37, 478, 713, 542))).getbbox() is None


def test_render_crop_pixel_cap_is_a_parameter() -> None:
    img = _image()
    plan = Z.CropPlan(0, ((0.0, 0.0, 600.0, 300.0),), "row_line")
    assert Z.render_crop(img, plan, W, H, None).size == (600, 300)
    out = Z.render_crop(img, plan, W, H, 90_000)
    assert out.width * out.height <= 90_000 and abs(out.width / out.height - 2) < 0.05


def test_render_crop_scales_boxes_when_image_size_differs_and_stacks() -> None:
    small = Image.new("RGB", (W // 2, H // 2), "white")
    plan = Z.CropPlan(0, ((100.0, 200.0, 300.0, 300.0), (100.0, 400.0, 400.0, 500.0)), "row_line")
    out = Z.render_crop(small, plan, W, H, None)
    assert out.size == (150, 50 + 50)  # widest part, heights add


# ------------------------------------------------------------------------------- selection


def _fs(field: str, p: float, value: str | None = "v", row: int | None = None, **kw: Any):
    return Z.FieldScore("d1", kw.pop("page", 0), field, row, p, value, kw.pop("lp", None))


def test_select_for_reread_strict_threshold_nan_and_order() -> None:
    scores = [
        _fs("quantity", 0.5, row=1),
        _fs("invoice_number", 0.9),  # == tau: not flagged
        _fs("buyer_name", 0.2),
        _fs("invoice_date", math.nan),
        _fs("invoice_number", 0.89, page=0),
        _fs("quantity", 0.1, row=0),
    ]
    got = Z.select_for_reread(scores, 0.9)
    assert [(s.field, s.row_index) for s in got] == [
        ("invoice_number", None),
        ("buyer_name", None),
        ("quantity", 0),
        ("quantity", 1),
    ]
    assert Z.count_unscored(scores) == 1
    assert Z.select_for_reread(scores, 0.0) == []


def test_scores_from_oof_rows_adapter() -> None:
    rows = [
        {"doc_id": "d1", "scope": "header", "field": "buyer_name", "row_idx": "-1",
         "emitted": "1", "p_correct": "0.4", "p_null": "0.1"},
        {"doc_id": "d1", "scope": "header", "field": "invoice_date", "row_idx": "-1",
         "emitted": "0", "p_correct": "", "p_null": "0.7"},
        {"doc_id": "d1", "scope": "row", "field": "quantity", "row_idx": "2.0",
         "emitted": "1", "p_correct": "0.95", "p_null": "0.0"},
    ]  # fmt: skip

    def resolve(
        doc: str, scope: str, field: str, ridx: int
    ) -> tuple[str | None, int, float | None]:
        return (None if field == "invoice_date" else "x", 1 if scope == "row" else 0, -0.5)

    got = Z.scores_from_oof_rows(rows, resolve)
    assert [(s.field, s.row_index, s.p_correct, s.page) for s in got] == [
        ("buyer_name", None, 0.4, 0),
        ("invoice_date", None, 0.7, 0),  # a null field is scored by P(null is right)
        ("quantity", 2, 0.95, 1),
    ]
    assert got[1].value is None and got[0].logprob_conf == -0.5


# ------------------------------------------------------------------------------- acceptance


def _rr(value: str | None, lp: float | None = -0.1, field: str = "buyer_name") -> Z.Reread:
    return Z.Reread(field, value, lp, lp, 3)


def _decide(
    score: Z.FieldScore,
    rr: Z.Reread | None,
    supported: bool = True,
    model: Z.ConfidenceModel = Z.LOGPROB_MIN,
    **kw: Any,
) -> Z.FieldDecision:
    return Z.decide(score, rr, model, _cfg(**kw), lambda _f, _v: supported)


def test_accept_only_when_reread_is_more_confident_both_ways() -> None:
    s = _fs("buyer_name", 0.3, "Acme", lp=-1.0)
    up = _decide(s, _rr("Acme Corp", -0.2))
    assert (up.action, up.reason, up.new_value) == ("replace", "more_confident", "Acme Corp")
    assert (up.old_conf, up.new_conf) == (-1.0, -0.2)
    down = _decide(s, _rr("Acme Corp", -3.0))
    assert (down.action, down.reason) == ("keep", "not_more_confident")
    tie = _decide(s, _rr("Acme Corp", -1.0))
    assert tie.action == "keep"  # strictly more confident is required
    assert _decide(s, _rr("Acme Corp", -0.9), accept_margin=0.2).action == "keep"
    assert _decide(s, _rr("Acme Corp", -0.7), accept_margin=0.2).action == "replace"


def test_calibrated_model_compares_p_correct_with_the_same_calibrator_on_the_reread() -> None:
    model = Z.calibrated(lambda _s, r: 0.9 if r.value == "Good" else 0.1)
    s = _fs("buyer_name", 0.5, "Meh", lp=-0.01)  # logprob is ignored by this model
    assert _decide(s, _rr("Good", -9.0), model=model).action == "replace"
    assert _decide(s, _rr("Bad", -0.0), model=model).action == "keep"


def test_missing_confidence_and_failed_reread_fail_closed() -> None:
    s = _fs("buyer_name", 0.3, "Acme", lp=None)
    assert _decide(s, _rr("Acme Corp", -0.1)).reason == "no_confidence"
    assert _decide(_fs("buyer_name", 0.3, "Acme", lp=-1.0), _rr("X", None)).reason == (
        "no_confidence"
    )
    assert _decide(s, None).reason == "reread_failed"
    nan = Z.calibrated(lambda _s, _r: math.nan)
    assert _decide(s, _rr("X"), model=nan).reason == "no_confidence"


def test_agreement_changes_nothing() -> None:
    s = _fs("invoice_number", 0.3, "INV-1", lp=-2.0)
    assert _decide(s, _rr("inv 1", -0.1, "invoice_number")).reason == "agree"


def test_null_to_value_policy_modes() -> None:
    s = _fs("buyer_name", 0.4, None, lp=-1.0)
    rr = _rr("Buyer X", -0.1)
    assert _decide(s, rr, supported=True).action == "replace"  # default: ocr_supported
    assert _decide(s, rr, supported=False).reason == "no_ocr_support"
    assert _decide(s, rr, supported=False, null_fill="any").action == "replace"
    assert _decide(s, rr, supported=True, null_fill="never").reason == "null_fill_disabled"
    # the confidence rule still applies first: a less confident reread never fills a null
    assert _decide(s, _rr("Buyer X", -5.0), null_fill="any").reason == "not_more_confident"
    assert Z.ZoomConfig().null_fill == "ocr_supported"
    with pytest.raises(ValueError):
        Z.ZoomConfig(null_fill="sometimes")  # type: ignore[arg-type]


def test_value_to_null_is_ignored_unless_allowed() -> None:
    s = _fs("buyer_name", 0.4, "Acme", lp=-1.0)
    assert _decide(s, _rr(None, -0.1)).reason == "reread_null_ignored"
    allowed = _decide(s, _rr(None, -0.1), allow_value_to_null=True)
    assert (allowed.action, allowed.new_value) == ("replace", None)
    assert _decide(_fs("buyer_name", 0.4, None, lp=-1.0), _rr(None)).reason == "both_null"


# ------------------------------------------------------------------------------- orchestration


class Scripted:
    """Backend stand-in answering the zoom schemas; records every call."""

    model_id = "scripted"
    revision = "x"

    def __init__(
        self,
        header: dict[str, tuple[str | None, float]],
        row: tuple[dict[str, str | None], float] | None = None,
        *,
        trace: bool = True,
        raw_override: str | None = None,
    ) -> None:
        self.header, self.row, self.trace, self.raw_override = header, row, trace, raw_override
        self.calls: list[dict[str, Any]] = []

    def extract_page(
        self, image: Any, prompt: str, schema: dict[str, Any], ocr_text: str | None
    ) -> tuple[str, dict[str, Any] | None, dict[str, Any]]:
        self.calls.append({"size": image.size, "prompt": prompt, "schema": schema, "ocr": ocr_text})
        if "header" in schema["properties"]:
            (field,) = schema["properties"]["header"]["properties"]
            value, lp = self.header[field]
            body: dict[str, Any] = {"header": {field: value}}
        else:
            assert self.row is not None
            row, lp = self.row
            body = {"line_items": [row]}
        raw = self.raw_override or json.dumps(body, separators=(",", ":"))
        meta: dict[str, Any] = {"latency_s": 1.5, "n_input_tokens": 100, "n_output_tokens": 10}
        if self.trace:
            meta["logprob_trace"] = {"ends": [len(raw)], "lp": [lp], "lp_c": [lp]}
        return raw, ex.parse_page_json(raw), meta


class FakeLocator:
    """Fixed OCR positions; `ocr_values` are the values the OCR 'supports'."""

    def __init__(self, ocr_values: set[str]) -> None:
        self.ocr_values = ocr_values

    def header_line(self, field_name: str, value: str | None, page: int) -> Z.LineHit | None:
        return Z.LineHit(0, 1, (50, 150, 400, 170)) if value else None

    def row_line(self, rows: Any, row_index: int) -> Z.LineHit | None:
        return Z.LineHit(0, 3 + row_index, (50, 450 + 50 * row_index, 700, 470 + 50 * row_index))

    def supported(self, field_name: str, value: str) -> bool:
        return value in self.ocr_values


def _pred() -> dict[str, Any]:
    return {
        "header": {"invoice_number": "INV-1", "buyer_name": None, "currency": "USD"},
        "line_items": [
            {"supplier_part_number": f"P-{i}", "customer_part_number": None,
             "purchase_order": f"PO-{i}", "quantity": str(5 + i)}
            for i in range(3)
        ],
    }  # fmt: skip


def _scores() -> list[Z.FieldScore]:
    return [
        _fs("invoice_number", 0.3, "INV-1", lp=-2.0),
        _fs("buyer_name", 0.2, None, lp=-0.5),
        _fs("currency", 0.99, "USD", lp=-0.01),
        _fs("supplier_part_number", 0.99, "P-1", row=1, lp=-0.01),
        _fs("quantity", 0.4, "6", row=1, lp=-1.5),
        _fs("purchase_order", 0.1, "PO-1", row=1, lp=-3.0),
    ]


def _backend(**kw: Any) -> Scripted:
    row = {"supplier_part_number": "P-1", "customer_part_number": None,
           "purchase_order": "PO-1", "quantity": "12"}  # fmt: skip
    return Scripted(
        {"invoice_number": ("INV-7", -0.2), "buyer_name": ("Buyer X", -0.3)}, (row, -0.4), **kw
    )


def _run(backend: Scripted, cfg: Z.ZoomConfig, ocr_values: set[str] | None = None) -> Z.ZoomResult:
    return Z.zoom_document(
        "d1", _pred(), _scores(), cfg,
        backend=backend, ocr_pages=[_page()], image_for_page=lambda _i: _image(),
        locator=FakeLocator({"Buyer X"} if ocr_values is None else ocr_values),
    )  # fmt: skip


def test_zoom_document_end_to_end_call_accounting_and_replacements() -> None:
    be = _backend()
    res = _run(be, _cfg())
    # 2 flagged header fields = 2 calls; the 2 flagged fields of row 1 share ONE call
    assert (
        (res.n_calls, res.n_header_calls, res.n_row_calls)
        == (3, 2, 1)
        == (
            len(be.calls),
            2,
            1,
        )
    )
    assert res.n_selected_fields == 4 and res.n_replaced == 3
    assert res.pred["header"] == {
        "invoice_number": "INV-7",
        "buyer_name": "Buyer X",
        "currency": "USD",
    }
    row = res.pred["line_items"][1]
    assert row["quantity"] == "12" and row["purchase_order"] == "PO-1"  # agree: unchanged
    assert row["supplier_part_number"] == "P-1"  # unflagged field of a re-read row is untouched
    assert res.pred["line_items"][0] == _pred()["line_items"][0]
    by = {(d.field, d.row_index): d for d in res.decisions}
    assert by[("purchase_order", 1)].reason == "agree"
    assert (res.latency_s, res.n_input_tokens, res.n_output_tokens) == (4.5, 300, 30)
    # crops: no OCR hint is passed, schemas are the tiny ones, row call carries the identity hint
    assert all(c["ocr"] is None for c in be.calls)
    row_call = next(c for c in be.calls if "line_items" in c["schema"]["properties"])
    assert "supplier_part_number is P-1" in row_call["prompt"]
    assert "column headings" in row_call["prompt"]
    assert row_call["size"][0] == 676 and row_call["size"][1] > 64  # stacked strip + band


def test_null_stays_null_without_ocr_support() -> None:
    res = _run(_backend(), _cfg(), ocr_values=set())
    assert res.pred["header"]["buyer_name"] is None
    d = next(d for d in res.decisions if d.field == "buyer_name")
    assert d.reason == "no_ocr_support"


def test_input_is_never_mutated_and_run_is_deterministic() -> None:
    pred, before = _pred(), json.dumps(_pred(), sort_keys=True)
    a = Z.zoom_document(
        "d1", pred, _scores(), _cfg(), backend=_backend(), ocr_pages=[_page()],
        image_for_page=lambda _i: _image(), locator=FakeLocator({"Buyer X"}),
    )  # fmt: skip
    assert json.dumps(pred, sort_keys=True) == before
    b = _run(_backend(), _cfg())
    assert a.pred == b.pred and a.decisions == b.decisions
    assert [c for c in _run(_backend(), _cfg()).decisions] == b.decisions


def test_default_off_touches_nothing() -> None:
    class Boom:
        def __getattr__(self, name: str) -> Any:
            raise AssertionError("backend touched while zoom is off")

    def no_image(_i: int) -> Any:
        raise AssertionError("image loaded while zoom is off")

    assert Z.ZoomConfig().enabled is False and Z.ZoomConfig().tau == 0.0
    for cfg in (Z.ZoomConfig(), Z.ZoomConfig(tau=0.99)):  # off even with a tau
        res = Z.zoom_document(
            "d1", _pred(), _scores(), cfg, backend=Boom(), ocr_pages=[], image_for_page=no_image
        )
        assert res.pred == _pred() and res.n_calls == 0 and res.decisions == []
    # enabled with nothing below tau: also zero calls, backend untouched
    res = Z.zoom_document(
        "d1", _pred(), _scores(), _cfg(tau=0.05), backend=Boom(), ocr_pages=[],
        image_for_page=no_image,
    )  # fmt: skip
    assert res.n_calls == 0 and res.pred == _pred()


def test_call_cap_keeps_the_least_confident_task() -> None:
    be = _backend()
    res = _run(be, _cfg(max_calls_per_doc=1))
    assert res.n_calls == 1 and res.n_row_calls == 1  # row task holds p=0.1, the lowest
    assert res.skipped["call_cap"] == 2
    assert {d.reason for d in res.decisions if d.field in ("invoice_number", "buyer_name")} == {
        "call_cap"
    }
    assert res.pred["header"]["invoice_number"] == "INV-1"


def test_row_identity_mismatch_discards_the_whole_reread() -> None:
    row = {"supplier_part_number": "P-9", "customer_part_number": None,
           "purchase_order": "PO-9", "quantity": "99"}  # fmt: skip
    be = Scripted({"invoice_number": ("INV-7", -0.2), "buyer_name": ("Buyer X", -0.3)}, (row, -0.1))
    res = _run(be, _cfg())
    assert res.pred["line_items"][1] == _pred()["line_items"][1]
    assert {d.reason for d in res.decisions if d.row_index == 1} == {"reread_identity_mismatch"}


def test_unparsable_and_missing_logprobs_keep_the_original() -> None:
    bad = _run(_backend(raw_override="{not json"), _cfg())
    assert bad.pred == _pred() and bad.n_replaced == 0
    assert {d.reason for d in bad.decisions} == {"reread_unparsable"}
    nolp = _run(_backend(trace=False), _cfg())
    assert nolp.pred == _pred()
    assert {d.reason for d in nolp.decisions} == {"reread_no_logprobs"}


def test_unlocated_row_and_region_are_skipped_without_a_call() -> None:
    class Nowhere(FakeLocator):
        def row_line(self, rows: Any, row_index: int) -> None:
            return None

        def header_line(self, field_name: str, value: str | None, page: int) -> None:
            return None

    empty = PageOcr("e.png", W, H, "paddleocr", "line", 0, items=[])
    be = _backend()
    res = Z.zoom_document(
        "d1", _pred(), _scores(), _cfg(), backend=be, ocr_pages=[empty],
        image_for_page=lambda _i: _image(), locator=Nowhere(set()),
    )  # fmt: skip
    assert res.n_calls == 0 and be.calls == []
    assert res.skipped == {"no_region": 2, "row_not_located": 1}


def test_zoom_requires_the_keyed_output_format() -> None:
    with pytest.raises(ValueError, match="keyed"):
        Z.ZoomReader(ex.MockBackend({}, output_format="compact"), _cfg())


def test_reader_works_with_the_repo_mock_backend() -> None:
    gold = {
        "d1": {"doc_type": "invoice",
               "header": {"invoice_number": "INV-9", "buyer_name": "Buyer"},
               "line_items": []}
    }  # fmt: skip
    be = ex.MockBackend(gold)
    be.capture_logprobs = True
    out = Z.ZoomReader(be, _cfg()).read_header("d1", 0, 1, _image(), "invoice_number")
    assert out.status == "ok" and out.rereads["invoice_number"].value == "INV-9"
    assert out.rereads["invoice_number"].lp_min is not None and be._calls == 1


# ------------------------------------------------------------------------------- label-free


def test_no_function_takes_or_names_labels_or_gold() -> None:
    tree = ast.parse(inspect.getsource(Z))
    names = {n.arg for n in ast.walk(tree) if isinstance(n, ast.arg)}
    names |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not {x for x in names if "gold" in x.lower() or "label" in x.lower()}
    imported = {
        a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names
    } | {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert "shipdoc.eval" not in imported and "load_gold" not in imported


@needs_scorer
def test_ocr_locator_end_to_end_reads_no_label_files(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []
    real_open: Callable[..., Any] = builtins.open

    def spy(file: Any, *a: Any, **k: Any) -> Any:
        opened.append(str(file))
        return real_open(file, *a, **k)

    monkeypatch.setattr(builtins, "open", spy)
    monkeypatch.setattr(io, "open", spy)
    page = _page()
    loc = Z.OcrLocator([page])
    hit = loc.header_line("invoice_number", "INV-1", 0)
    assert hit is not None and hit.line_idx == 1
    assert loc.header_line("invoice_number", None, 0) is None
    assert loc.supported("invoice_number", "INV-1") and not loc.supported("invoice_number", "ZZZ-9")
    rows = _pred()["line_items"]
    r1 = loc.row_line(rows, 1)
    assert r1 is not None and r1.line_idx == 4 and r1.box[1] == 500
    res = Z.zoom_document(
        "d1", _pred(), _scores(), _cfg(), backend=_backend(), ocr_pages=[page],
        image_for_page=lambda _i: _image(),
    )  # fmt: skip
    assert res.n_calls == 3 and res.pred["header"]["invoice_number"] == "INV-7"
    assert res.pred["header"]["buyer_name"] is None  # "Buyer X" is not in the OCR: stays null
    assert not [p for p in opened if "labels" in p.replace("\\", "/")]


# ------------------------------------------------------------------------------- prompts


def test_prompts_carry_the_rules_verbatim_and_schemas_are_valid() -> None:
    for text in (Z.header_prompt("invoice_number"), Z.row_prompt(None, False)):
        for rule in (*prompts.RULES, prompts.NULL_RULE, *prompts.CONVENTION_RULES):
            assert rule in text
        assert "Read only this crop" in text
    assert "the invoice number" in Z.header_prompt("invoice_number")
    assert "supplier_part_number is PN-1" in Z.row_prompt("PN-1", False)
    jsonschema.Draft202012Validator.check_schema(Z.header_zoom_schema("quantity"))
    jsonschema.Draft202012Validator.check_schema(Z.row_zoom_schema())
    assert (
        Z.header_zoom_schema("total_amount")["properties"]["header"]["properties"]["total_amount"]
        == ex.NUMERIC_VALUE
    )
    keys = list(Z.row_zoom_schema()["properties"]["line_items"]["items"]["properties"])
    assert keys == list(ex.ROW_KEYS)  # decoding order = table column order
    assert len(Z.zoom_prompt_hash()) == 64 and Z.zoom_prompt_hash() == Z.zoom_prompt_hash()


# ------------------------------------------------------------------------------- estimate script

_spec = importlib.util.spec_from_file_location(
    "zoom_estimate", ROOT / "scripts" / "zoom_estimate.py"
)
ze = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["zoom_estimate"] = ze  # dataclasses resolve their module via sys.modules
_spec.loader.exec_module(ze)  # type: ignore[union-attr]
SPEEDS = ze.Speeds(
    "m", prefill_s=2.0, decode_tok_s=10.0, n_input_tokens_mean=1000.0, s_per_page_mean=40.0
)


def test_estimate_arithmetic_is_hand_checkable() -> None:
    a = ze.Assumptions(
        header_crop_px=(1024, 1024), row_crop_px=(1024, 1024), prompt_tokens=1000,
        header_out_tokens=10, row_out_tokens=20, flagged_per_row=2.0, pages_per_doc=2.0,
    )  # fmt: skip
    # 1024x1024 px = 1024 visual tokens; linear: 2.0/1000 s/token * (1000 + 1024) + decode
    assert ze.call_seconds(SPEEDS, (1024, 1024), 10, a, "linear") == pytest.approx(
        0.002 * 2024 + 1.0
    )
    assert ze.call_seconds(SPEEDS, (1024, 1024), 10, a, "fixed") == pytest.approx(2.0 + 1.0)
    est = ze.estimate(2.0, 4.0, SPEEDS, a)  # 2 header calls + 4 / 2.0 = 2 row calls
    assert (est["header_calls"], est["row_calls"], est["calls"]) == (2.0, 2.0, 4.0)
    assert est["seconds_fixed"] == pytest.approx(2 * 3.0 + 2 * 4.0)
    assert est["pages_eq_fixed"] == pytest.approx(14.0 / 40.0)
    assert est["overhead_fixed"] == pytest.approx(14.0 / 80.0)
    # with a realistic (small) crop the linear bound is the lower one
    d = ze.estimate(2.0, 4.0, SPEEDS, ze.Assumptions())
    assert d["seconds_linear"] < d["seconds_fixed"]


def test_sweep_scales_linearly_with_the_flag_rate_and_zero_means_zero() -> None:
    rows = ze.sweep([0.0, 0.1, 0.2], 40.0, SPEEDS, ze.Assumptions())
    assert rows[0]["calls"] == 0.0 and rows[0]["seconds_fixed"] == 0.0
    assert rows[2]["calls"] == pytest.approx(2 * rows[1]["calls"])


def test_flag_source_csv_and_json(tmp_path: Path) -> None:
    csv_path = tmp_path / "DRYRUN_x.csv"
    csv_path.write_text(
        "doc_id,scope,field,row_idx,emitted,p_correct,p_null\n"
        "a,header,buyer_name,-1,1,0.5,0.1\n"  # flagged header
        "a,row,quantity,0,1,0.4,0.0\n"  # flagged row 0
        "a,row,purchase_order,0,1,0.3,0.0\n"  # same row: one call
        "a,header,invoice_date,-1,0,,0.95\n"  # null, P(null right) 0.95: not flagged
        "b,header,buyer_name,-1,1,0.99,0.0\n",
        encoding="utf-8",
    )
    got = ze.load_flag_source(csv_path, 0.9)
    assert got["n_docs"] == 2 and got["fields"] == 2.5
    assert got["flagged_header"] == 0.5 and got["flagged_rows"] == 0.5
    assert got["flagged_row_fields"] == 1.0
    with pytest.raises(ValueError, match="--tau"):
        ze.load_flag_source(csv_path, None)
    js = tmp_path / "f.json"
    js.write_text(
        json.dumps({"docs": {"a": {"fields": 10, "flagged_header": 1, "flagged_rows": 2}}})
    )
    assert ze.load_flag_source(js, None)["flagged_rows"] == 2


def test_unknown_model_lists_the_valid_names() -> None:
    with pytest.raises(KeyError, match="valid"):
        ze.load_speeds("nope")
    assert ze.load_speeds("qwen35_4b_img_only").prefill_s > 0


def test_script_output_is_labelled_estimate_and_dry_run(tmp_path: Path) -> None:
    csv_path = tmp_path / "DRYRUN_x.csv"
    csv_path.write_text(
        "doc_id,scope,field,row_idx,emitted,p_correct,p_null\na,header,buyer_name,-1,1,0.5,0.1\n",
        encoding="utf-8",
    )
    out = tmp_path / "o.md"
    assert ze.main(["--flag-source", str(csv_path), "--tau", "0.9", "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "ESTIMATE (UNVERIFIED)" in text and "DRY RUN" in text and "5% (ESTIMATE)" in text
