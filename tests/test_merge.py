from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from shipdoc.extract import HEADER_KEYS, ROW_KEYS
from shipdoc.merge import FieldProvenance, is_header_row, merge_pages


def _row(part: str | None, qty: str | None = "1", cust: str | None = None) -> dict[str, Any]:
    return {
        "supplier_part_number": part,
        "customer_part_number": cust,
        "purchase_order": None,
        "quantity": qty,
    }


def _page(
    doc_type: str = "invoice",
    rows: list[dict[str, Any]] | None = None,
    kind: str = "first",
    **header: Any,
) -> dict[str, Any]:
    h = dict.fromkeys(HEADER_KEYS)
    h.update(header)
    return {"doc_type": doc_type, "header": h, "line_items": rows or [], "page_kind": kind}


def test_page2_trap_banner_invoice_number_does_not_fill() -> None:
    # Page 1 invoice number is redacted (null); page 2's "continued" banner prints it legibly.
    p1 = _page(invoice_number=None, supplier_name="Acme", rows=[_row("A")])
    p2 = _page(kind="continuation", invoice_number="INV-777", rows=[_row("B")], total_amount="9.00")
    res = merge_pages([p1, p2])
    assert res.doc["header"]["invoice_number"] is None
    assert res.diffs["ignored_elsewhere"] == {"invoice_number": [2]}
    assert res.doc["header"]["supplier_name"] == "Acme"


def test_identity_fields_come_from_page1_even_when_pages_disagree() -> None:
    p1 = _page(invoice_number="INV-1", supplier_name="Acme")
    p2 = _page(kind="continuation", invoice_number="INV-2", supplier_name="Acme")
    res = merge_pages([p1, p2])
    assert res.doc["header"]["invoice_number"] == "INV-1"
    assert res.diffs["disagreement"]["invoice_number"] == {1: "INV-1", 2: "INV-2"}
    assert "supplier_name" not in res.diffs["disagreement"]  # equal values are not disagreement
    assert res.diffs["field_pages"]["invoice_number"] == 1


def test_repeated_identical_rows_are_preserved_across_and_within_pages() -> None:
    dup = _row("PN-1", "5", "CP-1")
    p1 = _page(rows=[dup, dup])
    p2 = _page(kind="continuation", rows=[dup], total_amount="1.00")
    res = merge_pages([p1, p2])
    assert res.doc["line_items"] == [dup, dup, dup]


def test_repeated_header_row_is_dropped_but_data_rows_survive() -> None:
    label_row = _row("Part Number", "Qty", "Customer Part No")
    label_row["purchase_order"] = "P.O. No."
    assert is_header_row(label_row)
    assert is_header_row(_row("ITEM", "QTY"))
    assert not is_header_row(_row("PN-1", "5"))
    assert not is_header_row(_row("Part Number", "5"))  # a numeric quantity makes it data
    assert not is_header_row({k: None for k in ROW_KEYS})
    # regression (spike40 diagnosis): digits made a data row look like label words
    assert not is_header_row(_row("PO5551230001", None))
    assert not is_header_row(_row("PN-100", None))
    assert not is_header_row({**_row(None, None), "purchase_order": "PO5551230002"})
    assert is_header_row(_row("Part No.", "Qty."))
    p1 = _page(rows=[_row("A", "2")])
    p2 = _page(kind="continuation", rows=[label_row, _row("A", "2")], total_amount="3.00")
    res = merge_pages([p1, p2])
    assert [r["supplier_part_number"] for r in res.doc["line_items"]] == ["A", "A"]
    assert res.diffs["dropped_header_rows"] == [2]


def test_doc_type_majority_vote_and_tie_goes_to_page_one() -> None:
    inv, wb = _page("invoice"), _page("waybill", kind="continuation")
    assert merge_pages([inv, wb, wb]).doc["doc_type"] == "waybill"
    assert merge_pages([inv, inv, wb]).doc["doc_type"] == "invoice"
    assert merge_pages([inv, wb]).doc["doc_type"] == "invoice"  # tie -> page 1
    assert merge_pages([wb, inv]).doc["doc_type"] == "waybill"
    res = merge_pages([None, wb, inv])  # page 1 failed: tie goes to the earliest tied page
    assert res.doc["doc_type"] == "waybill" and res.diffs["failed_pages"] == [1]


def test_totals_default_to_last_page_with_a_value() -> None:
    p1 = _page(total_amount="10.00")  # a running subtotal on page 1
    p2 = _page(kind="continuation", total_amount="25.00")
    p3 = _page(kind="continuation", total_amount=None)
    res = merge_pages([p1, p2, p3])
    assert res.doc["header"]["total_amount"] == "25.00"
    assert res.diffs["field_pages"]["total_amount"] == 2


def test_provenance_table_can_pin_a_total_to_page_one() -> None:
    prov = FieldProvenance.from_dict(
        {"fields": {"total_amount": {"recommended": {"rule": "page1_header"}}}}
    )
    p1 = _page(total_amount="10.00")
    p2 = _page(kind="continuation", total_amount="25.00")
    assert merge_pages([p1, p2], prov).doc["header"]["total_amount"] == "10.00"
    # a null on page 1 stays null under "first": no fill from later pages
    assert merge_pages([_page(), p2], prov).doc["header"]["total_amount"] is None


def test_provenance_unknown_rule_missing_file_and_group_overrides_ignored(tmp_path: Path) -> None:
    data = {
        "total_amount": {
            "recommended": {
                "rule": "last_page_footer",
                "group_overrides": {"inv_g01": {"rule": "page1"}},  # must have no effect
            }
        },
        "pieces": {"recommended": {"rule": "weird"}},
    }
    prov = FieldProvenance.from_dict(data)
    assert prov.policy_for("invoice", "total_amount") == "last_nonnull"
    assert not hasattr(prov, "group_rules")
    assert prov.policy_for("waybill", "pieces") == "last_nonnull"
    assert prov.warnings and "weird" in prov.warnings[0]
    assert FieldProvenance.load(tmp_path / "absent.json").source == "default"
    f = tmp_path / "fp.json"
    f.write_text(json.dumps({"invoice.total_amount": "page1"}), encoding="utf-8")
    assert FieldProvenance.load(f).policy_for("invoice", "total_amount") == "first"


def test_provenance_reads_the_real_scripts_provenance_output() -> None:
    path = Path(__file__).resolve().parents[1] / "meta" / "field_provenance.json"
    if not path.is_file():
        pytest.skip("meta/field_provenance.json not generated")
    prov = FieldProvenance.load(path)
    assert prov.source == "field_provenance.json" and prov.warnings == []
    assert prov.rules.get("total_amount") == "last_nonnull"  # recommended rule: last_page
    assert prov.rules.get("invoice_number") == "first"
    assert "pieces" not in prov.rules  # waybill fields have no recommendation (null)


def test_failed_pages_and_all_failed_document() -> None:
    p2 = _page(kind="continuation", rows=[_row("B")], total_amount="2.00")
    res = merge_pages([None, p2])
    assert res.doc["header"]["invoice_number"] is None  # page 1 lost: nothing is borrowed
    assert res.doc["header"]["total_amount"] == "2.00"
    assert res.diffs["failed_pages"] == [1]
    empty = merge_pages([None])
    assert empty.diffs["doc_type_fallback"] is True
    assert set(empty.doc["header"].values()) == {None} and empty.doc["line_items"] == []


def test_waybill_header_keys_and_rows_forced_empty() -> None:
    p = _page("waybill", rows=[_row("X")], kind="single", mawb="123-12345678", pieces="3")
    res = merge_pages([p])
    assert set(res.doc["header"]) == {
        "carrier",
        "mawb",
        "hawb",
        "origin_airport",
        "destination_airport",
        "shipper_name",
        "consignee_name",
        "pieces",
        "gross_weight_kg",
    }
    assert res.doc["line_items"] == [] and res.diffs["dropped_waybill_rows"] == 1
    assert res.doc["header"]["pieces"] == "3"


def test_no_po_propagation() -> None:
    p1 = _page(rows=[{**_row("A"), "purchase_order": "PO-1"}, _row("B")])
    res = merge_pages([p1])
    assert [r["purchase_order"] for r in res.doc["line_items"]] == ["PO-1", None]


def test_generic_total_rule_has_no_supplier_group_dependence() -> None:
    """The old table pinned inv_g11 / inv_g15 differently; now every document uses one rule."""
    table = {
        "total_amount": {
            "recommended": {
                "rule": "last_page_footer",
                "group_overrides": {
                    "inv_g11": {"rule": "last_page"},
                    "inv_g15": {"rule": "last_page"},
                },
            }
        }
    }
    p1 = _page(total_amount="10.00")  # running subtotal printed on page 1
    p2 = _page(kind="continuation", total_amount="25.00")
    with_table = merge_pages([p1, p2], FieldProvenance.from_dict(table))
    without = merge_pages([p1, p2])
    assert with_table.doc == without.doc and with_table.doc["header"]["total_amount"] == "25.00"
    # only page 1 has a value -> that page is the page whose output has a non-null total
    only_first = merge_pages([p1, _page(kind="continuation")])
    assert only_first.doc["header"]["total_amount"] == "10.00"
    assert only_first.diffs["field_pages"]["total_amount"] == 1
    # blank / whitespace totals count as null
    assert (
        merge_pages([p1, _page(kind="continuation", total_amount=" ")]).doc["header"][
            "total_amount"
        ]
        == "10.00"
    )


# --- ablation switches: defaults are the pipeline, all switched off is the R0 baseline ---


def test_any_page_baseline_fills_from_later_pages_and_takes_the_first_value() -> None:
    p1 = _page(invoice_number=None, supplier_name="Acme", total_amount="10.00", rows=[_row("A")])
    p2 = _page(
        kind="continuation", invoice_number="INV-9", supplier_name="Other", total_amount="25.00"
    )
    base = merge_pages([p1, p2], provenance_aware=False)
    assert (
        base.doc["header"]["invoice_number"] == "INV-9"
    )  # the page-2 trap, deliberately not fixed
    assert base.doc["header"]["supplier_name"] == "Acme"  # first non-null
    assert base.doc["header"]["total_amount"] == "10.00"  # first, not last
    full = merge_pages([p1, p2])
    assert full.doc["header"]["invoice_number"] is None
    assert full.doc["header"]["total_amount"] == "25.00"


def test_any_page_baseline_takes_doc_type_from_the_first_parsed_page() -> None:
    inv, wb = _page("invoice"), _page("waybill")
    assert merge_pages([None, inv, wb, wb], provenance_aware=False).doc["doc_type"] == "invoice"
    assert merge_pages([None, inv, wb, wb]).doc["doc_type"] == "waybill"  # majority vote


def test_header_row_filter_can_be_switched_off() -> None:
    hdr = {
        "supplier_part_number": "Part No",
        "customer_part_number": None,
        "purchase_order": None,
        "quantity": "Qty",
    }
    pages = [_page(rows=[_row("A"), hdr])]
    assert [r["supplier_part_number"] for r in merge_pages(pages).doc["line_items"]] == ["A"]
    kept = merge_pages(pages, drop_header_rows=False).doc["line_items"]
    assert [r["supplier_part_number"] for r in kept] == ["A", "Part No"]


def test_total_page_hint_picks_a_page_but_never_supplies_a_value() -> None:
    p1 = _page(total_amount="10.00")
    p2 = _page(kind="continuation", total_amount="25.00")
    p3 = _page(kind="continuation")  # last page has no total
    assert merge_pages([p1, p2, p3]).doc["header"]["total_amount"] == "25.00"
    hinted = merge_pages([p1, p2, p3], total_page_hints={"total_amount": 0})
    assert hinted.doc["header"]["total_amount"] == "10.00"
    assert hinted.diffs["field_pages"]["total_amount"] == 1
    # hint on a page whose value is null, or out of range: fall back to the provenance policy
    for bad in (2, 7, -1):
        res = merge_pages([p1, p2, p3], total_page_hints={"total_amount": bad})
        assert res.doc["header"]["total_amount"] == "25.00"
    # a hint for a non-total field is ignored (identity fields stay on page 1)
    res = merge_pages([p1, p2], total_page_hints={"invoice_number": 1})
    assert res.doc["header"]["invoice_number"] is None


def test_default_merge_is_unchanged_by_the_new_switches() -> None:
    p1 = _page(invoice_number="INV-1", total_amount="10.00", rows=[_row("A")])
    p2 = _page(kind="continuation", total_amount="25.00", rows=[_row("B"), _row(None, None)])
    a = merge_pages([p1, p2])
    b = merge_pages([p1, p2], provenance_aware=True, drop_header_rows=True, total_page_hints=None)
    assert a.doc == b.doc and a.diffs == b.diffs
    assert [r["supplier_part_number"] for r in a.doc["line_items"]] == ["A", "B"]
