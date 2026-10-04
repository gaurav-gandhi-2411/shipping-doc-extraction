from __future__ import annotations

import random
import re
from datetime import date

import pytest

from shipdoc.normalize import (
    infer_date_order,
    normalize_code,
    normalize_date,
    normalize_doc,
    normalize_number,
    normalize_value,
)


@pytest.mark.parametrize(
    ("raw", "iso"),
    [
        ("2026-05-25", "2026-05-25"),
        ("25/05/2026", "2026-05-25"),  # 25 cannot be a month: only DD/MM is valid
        ("05/25/2026", "2026-05-25"),  # 25 cannot be a month: only MM/DD is valid
        ("05/05/2026", "2026-05-05"),  # both readings agree
        ("25.05.2026", "2026-05-25"),
        ("25-May-2026", "2026-05-25"),
        ("25-MAY-2026", "2026-05-25"),
        ("5 Sept 2026", "2026-09-05"),
        ("May 25, 2026", "2026-05-25"),
        ("March 3rd, 2026", "2026-03-03"),
        ("  25/05/2026 ", "2026-05-25"),
        ("29/02/2028", "2028-02-29"),  # leap day
    ],
)
def test_dates_resolved(raw: str, iso: str) -> None:
    assert normalize_date(raw) == (iso, [])


@pytest.mark.parametrize("raw", ["03/04/2026", "12/11/2026", "01/02/2026", "03-04-2026"])
def test_ambiguous_numeric_dates_stay_raw(raw: str) -> None:
    assert normalize_date(raw) == (raw, ["date_ambiguous"])


def test_dotted_dates_are_day_first_only() -> None:
    assert normalize_date("03.04.2026") == ("2026-04-03", [])  # never read as March 4


@pytest.mark.parametrize(
    "raw", ["31/02/2026", "2026-13-01", "13/13/2026", "yesterday", "25 Foo 2026", "29/02/2027"]
)
def test_invalid_dates_unparsed(raw: str) -> None:
    assert normalize_date(raw) == (raw, ["date_unparsed"])


@pytest.mark.parametrize(
    ("raw", "out"),
    [
        ("1,234.50", "1234.50"),
        ("1.234,50", "1234.50"),
        ("1 234,50", "1234.50"),
        ("1 234,50", "1234.50"),
        ("1'234.50", "1234.50"),
        ("USD 1,234.50", "1234.50"),
        ("$1,234.50", "1234.50"),
        ("EUR1.234,50", "1234.50"),
        ("1234.50 EUR", "1234.50"),
        ("US$ 99.90", "99.90"),
        ("731905.26", "731905.26"),
        ("1,234,567.80", "1234567.80"),
        ("1.234.567,80", "1234567.80"),
        ("12,5", "12.5"),
        ("0,75", "0.75"),
        ("-5.00", "-5.00"),
        ("007", "007"),  # leading zeros are digits: kept
        ("100.00", "100.00"),  # trailing zeros are digits: kept
        ("23.1 kg", "23.1"),
        ("3000", "3000"),
    ],
)
def test_numbers(raw: str, out: str) -> None:
    value, _ = normalize_number(raw)
    assert value == out


def test_number_flags_for_ambiguous_separators() -> None:
    assert normalize_number("1,234") == ("1234", ["number_comma_thousands"])
    assert normalize_number("12,5") == ("12.5", ["number_decimal_comma"])
    assert normalize_number("1.234") == ("1.234", ["number_dot_three_decimals"])
    assert normalize_number("1234.5") == ("1234.5", [])


@pytest.mark.parametrize(
    "raw", ["abc", "12 ab 34", "1,23,456", "1.2.3", "12/34", "1,234.567,8", "100 and 200"]
)
def test_unparseable_numbers_are_returned_raw(raw: str) -> None:
    assert normalize_number(raw) == (raw, ["number_unparsed"])


def test_numbers_never_change_digits() -> None:
    for raw in ["USD 1,234.50", "1.234,50", "1 234,50", "0,75", "$ 0042", "1,234"]:
        out, _ = normalize_number(raw)
        assert re.sub(r"\D", "", out) == re.sub(r"\D", "", re.sub(r"[A-Za-z$]", "", raw))


def test_codes() -> None:
    assert normalize_code("  usd ") == ("USD", [])
    assert normalize_code("icn seoul", airport=True) == ("ICN", ["airport_city_stripped"])
    assert normalize_code("ICN", airport=True) == ("ICN", [])
    assert normalize_code("ICN, Seoul", airport=True)[0] == "ICN"
    assert normalize_code("SEOUL", airport=True) == ("SEOUL", [])  # leading token is not 3 letters
    assert normalize_code("Incheon Airport", airport=True)[0] == "INCHEON AIRPORT"
    assert normalize_code("ICNX", airport=True)[0] == "ICNX"
    assert normalize_code("usd 5", airport=False)[0] == "USD 5"  # no stripping off airports


def test_normalize_value_routing() -> None:
    assert normalize_value("invoice_date", "25-May-2026")[0] == "2026-05-25"
    assert normalize_value("quantity", "1,000")[0] == "1000"
    assert normalize_value("currency", "eur")[0] == "EUR"
    assert normalize_value("origin_airport", "HKG HONG KONG")[0] == "HKG"
    assert normalize_value("supplier_name", "  Acme Ltd ")[0] == "Acme Ltd"  # only stripped
    assert normalize_value("mawb", "123-12345678")[0] == "123-12345678"  # ids untouched
    assert normalize_value("invoice_number", " ")[0] is None
    assert normalize_value("invoice_number", None) == (None, [])
    assert normalize_value("quantity", 5)[0] == "5"


def test_normalize_doc_flags_and_paths() -> None:
    doc = {
        "doc_type": "invoice",
        "header": {
            "invoice_number": "INV-1",
            "invoice_date": "03/04/2026",
            "currency": "usd",
            "total_amount": "$1,234.50",
        },
        "line_items": [
            {"supplier_part_number": "A", "quantity": "1,000"},
            {"supplier_part_number": "B", "quantity": "x"},
        ],
    }
    new, flags = normalize_doc(doc)
    assert new["header"]["invoice_date"] == "03/04/2026"
    assert new["header"]["currency"] == "USD" and new["header"]["total_amount"] == "1234.50"
    assert new["line_items"][0]["quantity"] == "1000"
    assert flags == {
        "header.invoice_date": ["date_ambiguous"],
        "line_items[0].quantity": ["number_comma_thousands"],
        "line_items[1].quantity": ["number_unparsed"],
    }
    assert doc["header"]["currency"] == "usd"  # input not mutated


# --- cluster-level day/month order (Phase 3.3) and rule switches (ablation ladder) -------------


def test_infer_date_order_from_cluster_evidence() -> None:
    assert infer_date_order(["25/05/2026", "03/04/2026"]) == "dmy"  # 25 cannot be a month
    assert infer_date_order(["05/25/2026", "03/04/2026"]) == "mdy"
    assert infer_date_order(["03/04/2026", "05/05/2026"]) is None  # no evidence either way
    assert infer_date_order([]) is None
    assert infer_date_order(["25/05/2026", "05/25/2026"]) is None  # conflict: fail closed
    assert infer_date_order(["25.05.2026", "03/04/2026"]) is None  # dotted dates are no evidence
    assert infer_date_order(["2026-05-25", "25-May-2026", "junk"]) is None  # not numeric


def test_cluster_order_resolves_only_ambiguous_dates_and_is_flagged() -> None:
    assert normalize_date("03/04/2026", "dmy") == ("2026-04-03", ["date_order_inferred"])
    assert normalize_date("03/04/2026", "mdy") == ("2026-03-04", ["date_order_inferred"])
    assert normalize_date("03-04-2026", "dmy")[0] == "2026-04-03"
    assert normalize_date("03/04/2026", None) == ("03/04/2026", ["date_ambiguous"])
    # An unambiguous date ignores the hint (the hint can not override printed evidence).
    assert normalize_date("25/05/2026", "mdy") == ("2026-05-25", [])
    assert normalize_date("05/25/2026", "dmy") == ("2026-05-25", [])
    assert normalize_date("05/05/2026", "dmy") == ("2026-05-05", [])
    assert normalize_date("May 25, 2026", "mdy") == ("2026-05-25", [])
    # Not a date in either order: stays unparsed, the hint never makes one up.
    assert normalize_date("31/02/2026", "dmy") == ("31/02/2026", ["date_unparsed"])
    assert normalize_date("03/04/2026", "bogus") == ("03/04/2026", ["date_ambiguous"])


def test_cluster_order_never_invents_a_date_fuzz() -> None:
    """Whatever the order hint, the result is the raw string or a valid ISO date whose three
    numbers are exactly the printed ones (a re-ordering, never a new value)."""
    rng = random.Random(42)
    for _ in range(2000):
        a, b, y = rng.randint(0, 40), rng.randint(0, 40), rng.randint(1999, 2031)
        sep = rng.choice("/-.")
        raw = f"{a:02d}{sep}{b:02d}{sep}{y}"
        for order in (None, "dmy", "mdy"):
            out, _ = normalize_date(raw, order)
            if out == raw:
                continue
            yy, mm, dd = (int(x) for x in out.split("-"))
            assert yy == y and sorted([mm, dd]) == sorted([a, b])
            date(yy, mm, dd)  # valid calendar date


def test_normalize_doc_switches_and_date_order() -> None:
    doc = {
        "doc_type": "invoice",
        "header": {"invoice_date": "03/04/2026", "currency": "usd", "total_amount": "$1,234.50"},
        "line_items": [{"quantity": "1,000"}],
    }
    off, flags = normalize_doc(doc, dates=False, numbers=False, codes=False)
    assert off["header"] == doc["header"] and off["line_items"] == doc["line_items"] and not flags
    only_codes, _ = normalize_doc(doc, dates=False, numbers=False)
    assert only_codes["header"]["currency"] == "USD"
    assert only_codes["header"]["total_amount"] == "$1,234.50"
    new, fl = normalize_doc(doc, date_order="dmy")
    assert new["header"]["invoice_date"] == "2026-04-03"
    assert fl["header.invoice_date"] == ["date_order_inferred"]
    assert normalize_doc(doc)[1]["header.invoice_date"] == ["date_ambiguous"]


def test_number_normalisation_never_invents_digits_fuzz() -> None:
    """Random separator soup: the output is the raw string or has exactly the raw digits."""
    rng = random.Random(42)
    alphabet = "0123456789" * 3 + ",.  '-USDEUR$"
    for _ in range(5000):
        raw = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 14)))
        out, _ = normalize_number(raw)
        if out != raw:
            assert re.sub(r"\D", "", out) == re.sub(r"\D", "", raw), (raw, out)
