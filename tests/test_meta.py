from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from shipdoc import meta as M

ROOT = Path(__file__).resolve().parents[1]
FOLDS = ROOT / "splits" / "folds.json"
META = [ROOT / "meta" / "train.json", ROOT / "meta" / "dev.json"]


def _inv(doc_id: str, pages: list[str], parts: list[str], **header: Any) -> dict[str, Any]:
    h = {"invoice_number": "N1", "invoice_date": "2026-01-01", "awb_number": None}
    h.update(header)
    return {
        "doc_id": doc_id,
        "doc_type": "invoice",
        "header": h,
        "line_items": [{"supplier_part_number": p} for p in parts],
        "pages": pages,
    }


def _wb(doc_id: str, **header: Any) -> dict[str, Any]:
    h = {"mawb": "M1", "hawb": "H1"}
    h.update(header)
    return {
        "doc_id": doc_id,
        "doc_type": "waybill",
        "header": h,
        "line_items": [],
        "pages": ["w.png"],
    }


def _pool() -> list[dict[str, Any]]:
    """20 invoices: awb null on 5, 1 invoice_number null (5%); 10 waybills, 2 hawb null (20%)."""
    docs = [_inv(f"i{n}", ["a.png"], ["P1"], awb_number=None if n < 5 else "A") for n in range(20)]
    docs[0]["header"]["invoice_number"] = None
    docs += [_wb(f"w{n}", hawb=None if n < 2 else "H") for n in range(20)]
    return docs


def test_not_printed_is_data_driven() -> None:
    np_ = M.not_printed_fields(_pool())
    assert np_ == {"invoice": {"awb_number"}, "waybill": {"hawb"}}


def test_illegible_vs_not_printed() -> None:
    pool = _pool()
    np_ = {"invoice": {"awb_number"}, "waybill": {"hawb"}}
    assert M.is_illegible(pool[0], np_)  # invoice_number null: redaction
    assert not M.is_illegible(pool[1], np_)  # only awb null: line not printed
    assert not M.is_illegible(pool[20], np_)  # waybill, hawb null: not printed
    assert M.is_illegible(_wb("x", mawb=None), np_)  # required field null
    assert M.is_illegible(_inv("y", ["a.png"], [], invoice_date="  "), np_)  # blank counts


def test_threshold_boundary() -> None:
    docs = [_inv(f"i{n}", ["a.png"], []) for n in range(20)]
    docs[0]["header"]["invoice_number"] = None  # 1/20 = 5% < 10%: required
    assert M.not_printed_fields(docs)["invoice"] == {"awb_number"}
    docs[1]["header"]["invoice_number"] = None  # 2/20 = 10% >= 10%: optional line
    assert M.not_printed_fields(docs)["invoice"] == {"awb_number", "invoice_number"}


def test_repeated_parts() -> None:
    assert M.has_repeated_parts(_inv("a", ["a.png"], ["P1", "P2", "P1"]))
    assert not M.has_repeated_parts(_inv("a", ["a.png"], ["P1", "P2"]))
    assert not M.has_repeated_parts(_wb("w"))


def test_scanned_multipage_mixed_waybill_tags() -> None:
    np_: dict[str, set[str]] = {}
    scan = M.tag_doc(_inv("a", ["a_p1.JPG"], ["P"]), "inv_g01", np_)
    assert scan["scanned"] is True and scan["multipage"] is False and scan["waybill"] is False
    multi = M.tag_doc(_inv("b", ["b_p1.png", "b_p2.png"], ["P"]), "inv_g02", np_)
    assert multi["scanned"] is False and multi["multipage"] is True
    assert M.tag_doc(_wb("w"), "wb_g01", np_)["waybill"] is True
    assert M.is_mixed_scan(_inv("c", ["c_p1.jpg", "c_p2.png"], []))
    assert not M.is_mixed_scan(_inv("c", ["c_p1.jpg", "c_p2.jpg"], []))
    assert not M.is_mixed_scan(_inv("c", ["c_p1.png"], []))


def test_tags_are_real_bools() -> None:
    rec = M.tag_doc(_inv("a", ["a.jpg"], ["P", "P"]), "inv_g01", {"invoice": {"awb_number"}})
    assert all(type(rec[t]) is bool for t in M.BOOL_TAGS)
    assert rec["supplier_group"] == "inv_g01"


def test_fold_meta_unseen() -> None:
    folds = {"doc_fold": {"a": 0, "b": 1}}
    out = M.fold_meta([{"doc_id": "a"}, {"doc_id": "b"}], folds, 1)
    assert [m["unseen"] for m in out] == [False, True]


def test_make_folds_synthetic_constraints() -> None:
    rows = [
        {
            "doc_id": f"d{n}",
            "waybill": n % 5 == 0,
            "supplier_group": f"{'wb' if n % 5 == 0 else 'inv'}_g{n % 10}",
        }
        for n in range(150)
    ]
    # waybill groups: n%5==0 -> n%10 in {0,5}; give 6 waybill groups so 3 folds can each hold one
    for n, r in enumerate(rows):
        if r["waybill"]:
            r["supplier_group"] = f"wb_g{n % 6}"
    out = M.make_folds(rows)
    assert M.make_folds(rows) == out  # deterministic
    assert sorted(out["doc_fold"]) == sorted(r["doc_id"] for r in rows)


needs_folds = pytest.mark.skipif(
    not (FOLDS.is_file() and all(p.is_file() for p in META)),
    reason="splits/folds.json or meta/*.json absent",
)


@needs_folds
def test_committed_folds() -> None:
    folds = json.loads(FOLDS.read_text(encoding="utf-8"))
    rows = [r for p in META for r in json.loads(p.read_text(encoding="utf-8"))]
    by = {r["doc_id"]: r for r in rows}
    assert (folds["k"], folds["seed"], folds["group_key"]) == (3, 42, "supplier_group")
    ids = [d for f in folds["folds"] for d in f["val_doc_ids"]]
    assert len(ids) == len(set(ids)) == len(rows) == 500
    assert set(ids) == set(by)
    assert all(folds["doc_fold"][d] == f["fold"] for f in folds["folds"] for d in f["val_doc_ids"])
    groups = [g for f in folds["folds"] for g in f["val_groups"]]
    assert len(groups) == len(set(groups))  # no group spans two folds
    for f in folds["folds"]:
        assert {by[d]["supplier_group"] for d in f["val_doc_ids"]} == set(f["val_groups"])
        assert {by[d]["waybill"] for d in f["val_doc_ids"]} == {True, False}
    # determinism: regenerate in memory and compare
    assert M.make_folds(rows) == folds


def test_awb_and_hawb_absent_tags() -> None:
    np_ = {"invoice": {"awb_number"}, "waybill": {"hawb"}}
    inv_null = M.tag_doc(_inv("a", ["a.png"], ["P"]), "inv_g01", np_)
    inv_val = M.tag_doc(_inv("b", ["a.png"], ["P"], awb_number="176-12345678"), "inv_g01", np_)
    inv_blank = M.tag_doc(_inv("c", ["a.png"], ["P"], awb_number="  "), "inv_g01", np_)
    assert inv_null["awb_absent"] is True
    assert inv_val["awb_absent"] is False
    assert inv_blank["awb_absent"] is True  # blank counts as null, like the scorer's illegible
    wb_null = M.tag_doc(_wb("w", hawb=None), "wb_g01", np_)
    wb_val = M.tag_doc(_wb("v"), "wb_g01", np_)
    assert wb_null["hawb_absent"] is True
    assert wb_val["hawb_absent"] is False
    # each tag is scoped to its own doc type
    assert inv_null["hawb_absent"] is False
    assert wb_null["awb_absent"] is False
    assert {"awb_absent", "hawb_absent"} <= set(M.BOOL_TAGS)


@needs_folds
def test_committed_meta_absent_tags_match_gold() -> None:
    """Committed tags equal a re-derivation from gold nulls (skipped when data/ is absent)."""
    try:
        labels = {s: M.load_labels(s) for s in ("train", "dev")}
    except OSError:
        pytest.skip("data/ labels absent")
    if not all(labels.values()):
        pytest.skip("data/ labels absent")
    for split, docs in labels.items():
        rows = {
            r["doc_id"]: r for r in json.loads((ROOT / "meta" / f"{split}.json").read_text("utf-8"))
        }
        assert set(rows) == {d["doc_id"] for d in docs}
        for d in docs:
            assert rows[d["doc_id"]]["awb_absent"] is M.is_awb_absent(d)
            assert rows[d["doc_id"]]["hawb_absent"] is M.is_hawb_absent(d)
    n_awb = sum(M.is_awb_absent(d) for ds in labels.values() for d in ds)
    n_hawb = sum(M.is_hawb_absent(d) for ds in labels.values() for d in ds)
    assert (n_awb, n_hawb) == (136, 21)
