"""Confidence-only validators (ISO 4217, IATA, AWB shape): flags, never values."""

from __future__ import annotations

import copy

import pytest

from shipdoc.validate import (
    FLAG_AIRPORT,
    FLAG_AWB,
    FLAG_CURRENCY,
    FLAG_HAWB,
    iata_codes,
    iso4217_codes,
    validate_doc,
    validate_value,
)


def test_bundled_lists_contain_the_gold_codes_and_have_a_sane_size() -> None:
    assert {"CNY", "EUR", "JPY", "SGD", "USD"} <= iso4217_codes()
    assert 150 < len(iso4217_codes()) < 300
    assert {"ICN", "HKG", "SIN", "LAX", "FRA"} <= iata_codes()
    assert len(iata_codes()) > 5000
    assert all(len(c) == 3 and c.isupper() for c in iata_codes() | iso4217_codes())


@pytest.mark.parametrize(
    ("value", "flags"),
    [
        ("USD", []),
        ("usd", []),
        ("XXQ", [FLAG_CURRENCY]),
        ("US", [FLAG_CURRENCY]),
        ("$", [FLAG_CURRENCY]),
    ],
)
def test_currency(value: str, flags: list[str]) -> None:
    assert validate_value("currency", value) == flags


@pytest.mark.parametrize(
    ("value", "flags"),
    [
        ("ICN", []),
        ("icn", []),
        ("ZZZ9", [FLAG_AIRPORT]),
        ("ICN SEOUL", [FLAG_AIRPORT]),
        ("IC", [FLAG_AIRPORT]),
    ],
)
def test_airports(value: str, flags: list[str]) -> None:
    assert validate_value("origin_airport", value) == flags
    assert validate_value("destination_airport", value) == flags


@pytest.mark.parametrize("field", ["awb_number", "mawb"])
@pytest.mark.parametrize(
    ("value", "ok"),
    [
        ("123-12345678", True),
        ("123 - 12345678", True),  # the scorer ignores spaces
        ("12312345678", False),
        ("123-1234567", False),
        ("123-123456789", False),
        ("12A-12345678", False),
        ("O23-12345678", False),
    ],
)
def test_awb_shape(field: str, value: str, ok: bool) -> None:
    assert validate_value(field, value) == ([] if ok else [FLAG_AWB])


@pytest.mark.parametrize(
    ("value", "ok"),
    [
        ("AB12345678", True),
        ("ab12345678", True),
        ("A123456789", False),
        ("AB1234567", False),
        ("ABC2345678", False),
    ],
)
def test_hawb_shape(value: str, ok: bool) -> None:
    assert validate_value("hawb", value) == ([] if ok else [FLAG_HAWB])


def test_mod7_check_digit_is_not_used() -> None:
    """recon section 7: gold does not follow mod-7. 123-12345670 fails mod 7 but must pass."""
    serial = 1234567
    assert serial % 7 != 0 and validate_value("mawb", "123-12345670") == []
    assert validate_value("mawb", f"123-{serial}{(serial % 7 + 1) % 10}") == []


def test_null_blank_and_unvalidated_fields_are_never_flagged() -> None:
    for name in ("currency", "origin_airport", "awb_number", "mawb", "hawb"):
        assert validate_value(name, None) == [] and validate_value(name, "  ") == []
    assert validate_value("supplier_name", "not a code !!") == []
    assert validate_value("invoice_number", "12") == []


def test_validate_doc_never_changes_the_document() -> None:
    doc = {
        "doc_type": "waybill",
        "header": {
            "currency": "xxq",
            "origin_airport": "ZZZ9",
            "destination_airport": None,
            "mawb": "bad",
            "hawb": "A1",
            "carrier": "Acme",
        },
        "line_items": [{"supplier_part_number": "A", "quantity": "1"}],
    }
    before = copy.deepcopy(doc)
    flags = validate_doc(doc)
    assert doc == before  # fail closed: a validator cannot edit, fill or null a value
    assert flags == {
        "header.currency": [FLAG_CURRENCY],
        "header.origin_airport": [FLAG_AIRPORT],
        "header.mawb": [FLAG_AWB],
        "header.hawb": [FLAG_HAWB],
    }
    assert all(isinstance(v, list) and all(isinstance(x, str) for x in v) for v in flags.values())
    assert validate_doc({"header": {}}) == {} and validate_doc({}) == {}
