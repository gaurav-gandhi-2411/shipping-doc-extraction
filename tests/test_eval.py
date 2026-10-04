from __future__ import annotations

import copy
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from shipdoc import eval as ev
from shipdoc.ocr import OcrItem, PageOcr, cache_path, write_json_atomic

ROOT = Path(__file__).resolve().parents[1]
GOLD_DIR = ROOT / "data" / "dev" / "labels"
SCORER = ROOT / "assignment" / "score.py"

needs_data = pytest.mark.skipif(
    not (SCORER.is_file() and GOLD_DIR.is_dir()),
    reason="assignment/score.py or data/dev/labels absent (both are gitignored)",
)
needs_scorer = pytest.mark.skipif(not SCORER.is_file(), reason="assignment/score.py absent")


def _inv(doc_id: str = "d1", **header: Any) -> dict[str, Any]:
    h = {
        "invoice_number": "INV-1001",
        "invoice_date": "2026-05-25",
        "supplier_name": "Acme Corp",
        "buyer_name": "Buyer Ltd",
        "ship_to_name": "Ship Co",
        "currency": "USD",
        "total_amount": "1234.50",
        "awb_number": None,
    }
    h.update(header)
    return {"doc_id": doc_id, "doc_type": "invoice", "header": h, "line_items": []}


def _row(part: str, qty: str, po: str | None = None, cust: str = "CP-1") -> dict[str, Any]:
    return {
        "supplier_part_number": part,
        "customer_part_number": cust,
        "purchase_order": po,
        "quantity": qty,
    }


def _pred_of(gold: dict[str, Any], **header: Any) -> dict[str, Any]:
    p = copy.deepcopy(gold)
    p["header"].update(header)
    del p["doc_id"]
    return p


def test_missing_scorer_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="official scorer not found"):
        ev.load_scorer(tmp_path / "nope.py")


# ---------------------------------------------------------------- parity vs the CLI


def _perturb(gold: dict[str, dict[str, Any]], kind: int, seed: int = 42) -> dict[str, Any]:
    rng = random.Random(seed + kind)
    pred: dict[str, Any] = {}
    for d, g in gold.items():
        p = {
            "doc_type": g["doc_type"],
            "header": dict(g["header"]),
            "line_items": [dict(r) for r in g["line_items"]],
        }
        r = rng.random()
        if kind == 0:  # drop rows, blank fields
            p["line_items"] = [x for x in p["line_items"] if rng.random() > 0.3]
            for k in list(p["header"]):
                if rng.random() < 0.15:
                    p["header"][k] = None
        elif kind == 1:  # mutate chars, swap doc_types, add extra rows
            for k, v in p["header"].items():
                if isinstance(v, str) and v and rng.random() < 0.25:
                    i = rng.randrange(len(v))
                    p["header"][k] = v[:i] + "X" + v[i + 1 :]
            if r < 0.1:
                p["doc_type"] = "waybill" if g["doc_type"] == "invoice" else "invoice"
            if p["line_items"] and rng.random() < 0.3:
                p["line_items"].append(dict(p["line_items"][0]))
            for row in p["line_items"]:
                if rng.random() < 0.1:
                    row["quantity"] = str(int(float(row["quantity"] or 0)) + 1)
        else:  # false fills, non-ISO dates, missing docs
            for k, v in list(p["header"].items()):
                if v is None and rng.random() < 0.7:
                    p["header"][k] = "FILLED"
                if k == "invoice_date" and v and rng.random() < 0.5:
                    y, m, dd = v.split("-")
                    p["header"][k] = f"{dd}/{m}/{y}"
            if r < 0.1:
                continue
        pred[d] = p
    return pred


@needs_data
@pytest.mark.parametrize("kind", [0, 1, 2])
def test_wrapper_matches_cli_exactly(kind: int, tmp_path: Path) -> None:
    gold = ev.load_gold(GOLD_DIR)
    pred = _perturb(gold, kind)
    rng = random.Random(7)
    meta = [
        {
            "doc_id": d,
            "scanned": rng.random() < 0.5,
            "multipage": rng.random() < 0.3,
            "supplier_group": "A" if rng.random() < 0.5 else "B",
        }
        for d in gold
    ]
    pred_p, meta_p, out_p = tmp_path / "pred.json", tmp_path / "meta.json", tmp_path / "out.json"
    pred_p.write_text(json.dumps(pred), encoding="utf-8")
    meta_p.write_text(json.dumps(meta), encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            str(SCORER),
            "--gold",
            str(GOLD_DIR),
            "--pred",
            str(pred_p),
            "--meta",
            str(meta_p),
            "--out",
            str(out_p),
        ],
        check=True,
        capture_output=True,
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    cli = json.loads(out_p.read_text(encoding="utf-8"))
    mine = ev.score(pred, gold, meta)
    assert any(k.startswith("scanned=") for k in cli)  # slice path exercised
    assert mine == cli
    assert mine["all"]["OVERALL"] < 1.0
    # string supplier_group slices are an extension, only with group_slices=True
    grouped = ev.score(pred, gold, meta, group_slices=True)
    assert {"supplier_group=A", "supplier_group=B"} <= set(grouped)
    assert all(grouped[k] == v for k, v in cli.items())


# ---------------------------------------------------------------- bootstrap


@needs_data
def test_bootstrap_determinism_and_paired_self() -> None:
    gold = ev.load_gold(GOLD_DIR)
    pred = _perturb(gold, 0)
    a = ev.confidence_intervals(pred, gold, n=100, seed=42)
    b = ev.confidence_intervals(pred, gold, n=100, seed=42)
    c = ev.confidence_intervals(pred, gold, n=100, seed=43)
    assert a == b
    assert a != c
    o = a["all"]["OVERALL"]
    assert o["lo"] <= o["point"] <= o["hi"]
    res = ev.paired_bootstrap(pred, pred, gold, n=100)
    for m in ev.METRICS:
        assert res[m]["delta"] == 0 and res[m]["lo"] == 0 and res[m]["hi"] == 0
        assert res[m]["p_delta_le_0"] == 1.0


@needs_scorer
def test_paired_bootstrap_detects_improvement() -> None:
    gold = {f"d{i}": _inv(f"d{i}") for i in range(30)}
    good = {d: _pred_of(g) for d, g in gold.items()}
    bad = {d: _pred_of(g, invoice_number="ZZZ", supplier_name="Nope") for d, g in gold.items()}
    res = ev.paired_bootstrap(bad, good, gold, n=200)
    assert res["OVERALL"]["delta"] > 0 and res["OVERALL"]["lo"] > 0
    assert res["OVERALL"]["p_delta_le_0"] == 0.0


# ---------------------------------------------------------------- taxonomy


def _write_cache(root: Path, doc_id: str, lines: list[str]) -> None:
    """Write a one-page synthetic OCR cache entry (one item per line) for `doc_id`."""
    items = [
        OcrItem(
            t,
            [[10, 100 + 40 * i], [600, 100 + 40 * i], [600, 120 + 40 * i], [10, 120 + 40 * i]],
            0.9,
        )
        for i, t in enumerate(lines)
    ]
    page = PageOcr(f"{doc_id}_p1.png", 1240, 1754, "paddleocr", "line", 0, items=items)
    write_json_atomic(
        cache_path(root, "paddleocr", doc_id.split("_", 1)[0], f"{doc_id}_p1"), page.to_dict()
    )


def _labels(
    gold: dict[str, Any], pred: dict[str, Any], ocr: dict[str, list[PageOcr]] | None = None
) -> list[tuple[str, str]]:
    tax = ev.error_taxonomy({gold["doc_id"]: pred}, {gold["doc_id"]: gold}, ocr)
    return [(e["field"], e["label"]) for e in tax["errors"]]


@needs_scorer
def test_taxonomy_header_categories() -> None:
    g = _inv(awb_number=None)
    p = _pred_of(
        g,
        invoice_number="INV-1O01",  # misread
        invoice_date="25/05/2026",  # normalization
        supplier_name="Buyer Ltd",  # convention:wrong_field
        buyer_name=None,  # convention:missed
        total_amount="$1.234,50",  # normalization (decimal comma + symbol)
        awb_number="AWB123",  # false_fill
        currency="QQQQ",  # other
    )
    assert dict(_labels(g, p)) == {
        "invoice_number": "misread",
        "invoice_date": "normalization",
        "supplier_name": "convention:wrong_field",
        "buyer_name": "convention:missed",
        "total_amount": "normalization",
        "awb_number": "false_fill",
        "currency": "other",
    }


@needs_scorer
def test_taxonomy_doc_type_outranks_everything() -> None:
    g = _inv()
    p = _pred_of(g, invoice_number="WRONG")
    p["doc_type"] = "waybill"
    labels = _labels(g, p)
    assert ("doc_type", "doc_type") in labels
    assert {lab for _, lab in labels} == {"doc_type"}
    # a missing doc is also a doc_type error
    assert {lab for _, lab in _labels(g, {})} == {"doc_type"}


_OCR_LINES = ["INVOICE", "Invoice No.: INV-1O01", "Date: 25/05/2026", "Acme Corp", "Buyer Ltd"]


@needs_scorer
def test_taxonomy_hallucination_precedence_and_unknown_support(tmp_path: Path) -> None:
    g = _inv()
    p = _pred_of(g, invoice_date="25/05/2026", invoice_number="INV-1O01", supplier_name="Zorblax")
    _write_cache(tmp_path, "d1", _OCR_LINES)
    pages = ev.load_ocr_pages(["d1"], tmp_path)
    with_ocr = dict(_labels(g, p, pages))
    assert with_ocr["supplier_name"] == "hallucination"  # no support in OCR
    assert with_ocr["invoice_date"] == "normalization"  # printed as 25/05/2026: supported
    assert with_ocr["invoice_number"] == "misread"  # printed exactly: supported
    # false_fill outranks hallucination even when the value is absent from the OCR text
    p2 = _pred_of(g, awb_number="NOSUCHTHING")
    assert dict(_labels(g, p2, pages))["awb_number"] == "false_fill"
    # without OCR: not evaluated, support recorded as unknown
    tax = ev.error_taxonomy({"d1": p}, {"d1": g})
    assert {e["support"] for e in tax["errors"]} == {"unknown_support"}
    assert dict(_labels(g, p))["supplier_name"] != "hallucination"


@needs_scorer
def test_taxonomy_support_uses_scorer_normalized_and_fuzzy_levels(tmp_path: Path) -> None:
    g = _inv(invoice_number="INV-2026-9999")
    # a wrong id one OCR character off the printed one (fuzzy support), a date that is not printed
    # (unsupported), a number printed with EU separators (correct, so no record)
    _write_cache(
        tmp_path, "d1", ["Invoice No.: INV-2026-10O1", "Date: 25/05/2026", "Total EUR 1.234,50"]
    )
    pages = ev.load_ocr_pages(["d1"], tmp_path)
    p = _pred_of(
        g, invoice_date="2026-05-26", invoice_number="INV-2026-1001", total_amount="1234.5"
    )
    tax = ev.error_taxonomy({"d1": p}, {"d1": g}, pages)
    sup = {e["field"]: e["support"] for e in tax["errors"]}
    assert sup["invoice_number"] == "supported"  # fuzzy hit at T
    assert sup["invoice_date"] == "unsupported"  # 26 May is not printed anywhere
    assert "total_amount" not in sup  # 1234.5 == 1234.50: correct, no error record
    p3 = _pred_of(g, total_amount="1234.5", currency="EUR", invoice_number="NOPE-7")
    sup3 = {e["field"]: e for e in ev.error_taxonomy({"d1": p3}, {"d1": g}, pages)["errors"]}
    assert sup3["currency"]["support"] == "supported"  # airport/currency code inside a longer span
    assert sup3["invoice_number"]["label"] == "hallucination"


@needs_scorer
def test_load_ocr_pages_omits_uncached_docs(tmp_path: Path) -> None:
    _write_cache(tmp_path, "d1", _OCR_LINES)
    got = ev.load_ocr_pages(["d1", "d2"], tmp_path)
    assert list(got) == ["d1"] and len(got["d1"]) == 1


@needs_scorer
def test_cli_ocr_cache_defaults_to_paths_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    g = _inv()
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    (gold_dir / "d1.json").write_text(json.dumps(g), encoding="utf-8")
    pred = {"d1": _pred_of(g, supplier_name="Zorblax")}
    pred_path = tmp_path / "pred.json"
    pred_path.write_text(json.dumps(pred), encoding="utf-8")
    cache = tmp_path / "cache"
    _write_cache(cache, "d1", _OCR_LINES)
    monkeypatch.setenv("SHIPDOC_OCR_CACHE", str(cache))  # the paths config: env wins over .env
    out = tmp_path / "out.json"
    argv = [
        "--gold",
        str(gold_dir),
        "--pred",
        str(pred_path),
        "--out",
        str(out),
        "--bootstrap",
        "0",
    ]
    assert ev.main(argv) == 0
    errs = json.loads(out.read_text(encoding="utf-8"))["taxonomy"]["errors"]
    assert [(e["field"], e["label"]) for e in errs] == [("supplier_name", "hallucination")]


@needs_scorer
def test_taxonomy_rows() -> None:
    g = _inv()
    g["line_items"] = [_row("A-1", "10"), _row("B-2", "20", cust="CP-2"), _row("C-3", "5")]
    p = _pred_of(g)
    # A-1 correct; B-2 dropped; extra row D-4; C-3 quantity wrong
    p["line_items"] = [_row("A-1", "10"), _row("D-4", "7"), _row("C-3", "50")]
    got = {(e["field"], e["label"]) for e in ev.error_taxonomy({"d1": p}, {"d1": g})["errors"]}
    assert ("row", "row_missing") in got
    assert ("row", "row_extra") in got
    assert ("quantity", "misread") in got  # "50" vs "5": distance 1


@needs_scorer
def test_taxonomy_split_merge_both_directions() -> None:
    g = _inv()
    g["line_items"] = [_row("A-1", "10"), _row("A-1", "15", cust="CP-9")]
    merged = _pred_of(g)
    merged["line_items"] = [_row("A-1", "25")]
    tax = ev.error_taxonomy({"d1": merged}, {"d1": g})
    assert {e["label"] for e in tax["errors"]} == {"row_split_merge"}
    assert len(tax["errors"]) == 3  # one pred row + two gold rows
    g2 = _inv()
    g2["line_items"] = [_row("A-1", "25")]
    split = _pred_of(g2)
    split["line_items"] = [_row("A-1", "10"), _row("A-1", "15", cust="CP-9")]
    tax2 = ev.error_taxonomy({"d1": split}, {"d1": g2})
    assert {e["label"] for e in tax2["errors"]} == {"row_split_merge"}


@needs_scorer
def test_taxonomy_counts_per_slice() -> None:
    g = _inv()
    p = _pred_of(g, awb_number="X")
    meta = [{"doc_id": "d1", "scanned": True}]
    tax = ev.error_taxonomy({"d1": p}, {"d1": g}, meta=meta)
    assert tax["counts"]["all"] == {"false_fill": 1}
    assert tax["counts"]["scanned=yes"] == {"false_fill": 1}
    assert tax["counts"]["scanned=no"] == {}
    assert tax["counts_by_category"]["invoices"] == {"false_fill": 1}


def test_parse_number_variants() -> None:
    assert ev.parse_number("1.234,50") == 1234.5
    assert ev.parse_number("€ 12,5") == 12.5
    assert ev.parse_number("1,234") == 1234
    assert ev.parse_number("abc") is None


@needs_scorer
def test_write_report_roundtrip(tmp_path: Path) -> None:
    g = _inv()
    gold, pred = {"d1": g}, {"d1": _pred_of(g, awb_number="X")}
    rep = ev.score(pred, gold)
    cis = ev.confidence_intervals(pred, gold, n=20)
    tax = ev.error_taxonomy(pred, gold)
    out = tmp_path / "r.json"
    ev.write_report(out, rep, cis, tax)
    assert json.loads(out.read_text(encoding="utf-8"))["report"] == rep
    assert "| all |" in out.with_suffix(".md").read_text(encoding="utf-8")
