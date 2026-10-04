"""Confidence-only validators: ISO 4217, IATA, AWB shape (spec Phase 3.5).

Contract: a validator reads a value and returns FLAGS. It never returns, builds or edits a value,
so applying it cannot change a prediction (``tests/test_validate.py::test_validate_doc_never_
changes_the_document``). The flags feed the later confidence model and the review flag.

* currency: membership in the ISO 4217 alphabetic list (``pycountry.currencies``, data from
  Debian iso-codes; pycountry is a locked runtime dependency).
* airports: membership in the IATA 3-letter list (``meta/iata_airports.txt``, OurAirports, public
  domain).
* MAWB / invoice ``awb_number``: shape ``NNN-NNNNNNNN`` only. The mod-7 check digit is NOT used
  (reports/recon.md section 7: 16/100 gold MAWBs pass, chance level).
* HAWB: shape ``AANNNNNNNN``.
"""

from __future__ import annotations

import re
from functools import cache
from pathlib import Path
from typing import Any

import pycountry

META = Path(__file__).resolve().parents[2] / "meta"

AWB_SHAPE = re.compile(r"^\d{3}-\d{8}$")
HAWB_SHAPE = re.compile(r"^[A-Z]{2}\d{8}$")
AWB_FIELDS = frozenset({"awb_number", "mawb"})

FLAG_CURRENCY = "currency_not_iso4217"
FLAG_AIRPORT = "airport_not_iata"
FLAG_AWB = "awb_bad_shape"
FLAG_HAWB = "hawb_bad_shape"


@cache
def _load_codes(name: str) -> frozenset[str]:
    """Upper-case codes of ``meta/<name>`` (one per line, ``#`` comment lines skipped)."""
    lines = (META / name).read_text(encoding="utf-8").splitlines()
    return frozenset(ln.strip().upper() for ln in lines if ln.strip() and not ln.startswith("#"))


@cache
def iso4217_codes() -> frozenset[str]:
    """The ISO 4217 alphabetic codes, read from ``pycountry.currencies`` (alpha_3)."""
    return frozenset(c.alpha_3.upper() for c in pycountry.currencies)


def iata_codes() -> frozenset[str]:
    """The bundled IATA airport codes."""
    return _load_codes("iata_airports.txt")


def validate_value(name: str, value: Any) -> list[str]:
    """Flags for one field value (empty list: valid, null, or a field with no validator).

    A null/blank value is never flagged (absence is a null decision, not a format error).
    """
    if value is None or not str(value).strip():
        return []
    s = str(value).strip()
    if name == "currency":
        return [] if s.upper() in iso4217_codes() else [FLAG_CURRENCY]
    if name in ("origin_airport", "destination_airport"):
        return [] if s.upper() in iata_codes() else [FLAG_AIRPORT]
    if name in AWB_FIELDS:
        return [] if AWB_SHAPE.match(re.sub(r"\s", "", s)) else [FLAG_AWB]  # scorer ignores spaces
    if name == "hawb":
        return [] if HAWB_SHAPE.match(re.sub(r"\s", "", s).upper()) else [FLAG_HAWB]
    return []


def validate_doc(doc: dict[str, Any]) -> dict[str, list[str]]:
    """``{header.<field>: [flags]}`` of a document (only non-empty); the document is not touched."""
    header = doc.get("header") or {}
    out: dict[str, list[str]] = {}
    for name, value in header.items():
        fl = validate_value(name, value)
        if fl:
            out[f"header.{name}"] = fl
    return out
