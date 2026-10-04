"""Minimal, value-preserving normalisation (spec Phase 3.3 - 3.5).

Contract: a normaliser may change FORMAT (date layout, separators, currency symbols, case) but never
digits, and never invents a value. When it cannot do that safely it returns the raw string and a
flag; the later confidence model reads the flags. All functions return ``(value, flags)``.

Dates: ISO is emitted only when exactly one interpretation is valid. An all-numeric date that is
valid both as DD/MM and MM/DD (and the two differ) stays raw with ``date_ambiguous`` unless the
caller passes a day/month ``order`` inferred from OTHER dates of the same layout cluster
(`infer_date_order`): it then picks between the two readings of the printed digits and flags
``date_order_inferred``. It never makes up a reading that is not valid for the printed digits.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import date
from typing import Any

DATE_FIELDS = frozenset({"invoice_date"})
NUMBER_FIELDS = frozenset({"total_amount", "pieces", "gross_weight_kg", "quantity"})
CODE_FIELDS = frozenset({"currency"})
AIRPORT_FIELDS = frozenset({"origin_airport", "destination_airport"})

MONTHS = {
    name: i
    for i, names in enumerate(
        [
            ("jan", "january"),
            ("feb", "february"),
            ("mar", "march"),
            ("apr", "april"),
            ("may",),
            ("jun", "june"),
            ("jul", "july"),
            ("aug", "august"),
            ("sep", "sept", "september"),
            ("oct", "october"),
            ("nov", "november"),
            ("dec", "december"),
        ],
        start=1,
    )
    for name in names
}

_ISO = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_NUMERIC = re.compile(r"^(\d{1,2})([/.-])(\d{1,2})\2(\d{4})$")
_DAY_MON = re.compile(r"^(\d{1,2})[ -]+([A-Za-z]{3,9})\.?[ ,-]+(\d{4})$")
_MON_DAY = re.compile(r"^([A-Za-z]{3,9})\.? +(\d{1,2})(?:st|nd|rd|th)?,? +(\d{4})$")


def _valid(y: int, m: int, d: int) -> str | None:
    try:
        return date(y, m, d).isoformat()
    except ValueError:
        return None


DATE_ORDERS = ("dmy", "mdy")


def infer_date_order(raws: Iterable[str]) -> str | None:
    """Day/month order of a layout cluster from its all-numeric dates (``/`` or ``-`` separated).

    A date that is valid in exactly one order (one of its two leading numbers exceeds 12) is
    evidence for that order; a date valid in both carries none; dotted dates are skipped (their
    day-first reading is a convention, not evidence). Returns ``"dmy"`` / ``"mdy"`` when all the
    evidence agrees, None when there is none or it conflicts (fail closed: the caller then keeps
    ambiguous dates raw).
    """
    possible = set(DATE_ORDERS)
    seen = False
    for raw in raws:
        m = _NUMERIC.match(re.sub(r"\s+", " ", raw.strip()))
        if not m or m[2] == ".":
            continue
        a, b, y = int(m[1]), int(m[3]), int(m[4])
        ok = {o for o, iso in (("dmy", _valid(y, b, a)), ("mdy", _valid(y, a, b))) if iso}
        if len(ok) == 1:
            seen = True
            possible &= ok
    return next(iter(possible)) if seen and len(possible) == 1 else None


def normalize_date(raw: str, order: str | None = None) -> tuple[str, list[str]]:
    """Date string to ISO ``YYYY-MM-DD`` when exactly one interpretation is valid.

    Supported: ISO, DD/MM/YYYY, MM/DD/YYYY (also with ``-``), DD.MM.YYYY (day first only),
    DD-MON-YYYY / DD Month YYYY, "Month DD, YYYY". Otherwise returns `raw` unchanged with a flag:
    ``date_ambiguous`` (two different valid readings) or ``date_unparsed``. With `order`
    (``"dmy"``/``"mdy"``, from `infer_date_order`) an ambiguous numeric date takes that reading
    and gets the flag ``date_order_inferred``.
    """
    s = re.sub(r"\s+", " ", raw.strip())
    if m := _ISO.match(s):
        iso = _valid(int(m[1]), int(m[2]), int(m[3]))
        return (iso, []) if iso else (raw, ["date_unparsed"])
    cands: set[str] = set()
    if m := _NUMERIC.match(s):
        a, sep, b, y = int(m[1]), m[2], int(m[3]), int(m[4])
        cands.add(_valid(y, b, a) or "")  # day first
        if sep != ".":  # DD.MM.YYYY is a day-first convention only
            cands.add(_valid(y, a, b) or "")  # month first
    elif m := _DAY_MON.match(s):
        mon = MONTHS.get(m[2].lower())
        if mon:
            cands.add(_valid(int(m[3]), mon, int(m[1])) or "")
    elif m := _MON_DAY.match(s):
        mon = MONTHS.get(m[1].lower())
        if mon:
            cands.add(_valid(int(m[3]), mon, int(m[2])) or "")
    cands.discard("")
    if len(cands) == 1:
        return cands.pop(), []
    if len(cands) == 2 and order in DATE_ORDERS:
        a, b, y = int(m[1]), int(m[3]), int(m[4])  # only the numeric branch yields two readings
        return (_valid(y, b, a) if order == "dmy" else _valid(y, a, b)) or raw, [
            "date_order_inferred"
        ]
    return raw, ["date_ambiguous" if len(cands) > 1 else "date_unparsed"]


_SPACES = "\u00a0\u202f '\u2019"  # grouping chars seen in print: nbsp, thin nbsp, space, '
_GROUPED = re.compile("^\\d{1,3}(?:[" + _SPACES + "]\\d{3})+(?:[.,]\\d+)?$")
_CORE = re.compile("-?\\d(?:[\\d.," + _SPACES + "]*\\d)?")


def _is_grouped(text: str, sep: str) -> bool:
    """True if `text` is digits grouped in threes by `sep` (``1,234,567``)."""
    return re.fullmatch(rf"\d{{1,3}}(?:{re.escape(sep)}\d{{3}})+", text) is not None


def normalize_number(raw: str) -> tuple[str, list[str]]:
    """Plain-number string: strip currency codes / symbols / short units, resolve separators.

    ``1,234.50`` / ``1.234,50`` / ``1 234,50`` / ``USD 1,234.50`` / ``EUR1.234,50`` become
    ``1234.50``. A lone comma followed by exactly three digits is read as thousands (``1,234`` ->
    ``1234``; flagged ``number_comma_thousands``), otherwise as a decimal comma. A lone dot is
    always a decimal point. Digits are never altered; anything else returns `raw` with
    ``number_unparsed``.
    """
    s = raw.strip()
    m = _CORE.search(s)
    if not m:
        return raw, ["number_unparsed"]
    rest = s[: m.start()] + " " + s[m.end() :]
    if re.search(r"\d", rest):
        return raw, ["number_unparsed"]  # digits outside the number: two numbers, not one
    words = re.findall(r"[A-Za-z]+", rest)
    if len(words) > 2 or any(not 2 <= len(w) <= 4 for w in words):
        return raw, ["number_unparsed"]  # not a currency code / short unit
    core = m.group(0)
    sign = "-" if core.startswith("-") else ""
    body = core.lstrip("-")
    flags: list[str] = []
    if any(ch in _SPACES for ch in body):
        if not _GROUPED.match(body):
            return raw, ["number_unparsed"]
        body = re.sub(f"[{_SPACES}]", "", body)
    commas, dots = body.count(","), body.count(".")
    if commas and dots:
        dec = "," if body.rfind(",") > body.rfind(".") else "."
        thou = "." if dec == "," else ","
        head, _, tail = body.rpartition(dec)
        if dec in head or not _is_grouped(head, thou):
            return raw, ["number_unparsed"]
        body = head.replace(thou, "") + "." + tail
    elif commas or dots:
        sep = "," if commas else "."
        n = commas or dots
        head, _, tail = body.rpartition(sep)
        if n > 1:  # repeated separator: must be thousands grouping
            if not _is_grouped(body, sep):
                return raw, ["number_unparsed"]
            body = body.replace(sep, "")
        elif sep == "," and len(tail) == 3:
            body = head + tail
            flags.append("number_comma_thousands")
        elif sep == ",":
            body = head + "." + tail
            flags.append("number_decimal_comma")
        elif len(tail) == 3 and len(head) <= 3:
            flags.append("number_dot_three_decimals")  # 1.234: kept as a decimal, may be thousands
    return sign + body, flags


def normalize_code(raw: str, airport: bool = False) -> tuple[str, list[str]]:
    """Upper-case and strip. For airports, ``ICN SEOUL`` -> ``ICN`` only if the leading token is
    exactly three letters; anything else is just upper-cased."""
    s = raw.strip()
    if airport:
        m = re.match(r"^([A-Za-z]{3})(?=$|[\s,/(-])", s)
        if m:
            return m[1].upper(), ([] if len(s) == 3 else ["airport_city_stripped"])
    return s.upper(), []


def normalize_value(
    name: str,
    raw: Any,
    *,
    dates: bool = True,
    numbers: bool = True,
    codes: bool = True,
    date_order: str | None = None,
) -> tuple[Any, list[str]]:
    """Normalise one field value by field name; non-strings and unknown fields pass through.

    `dates` / `numbers` / `codes` switch a rule family off (ablations only; the defaults are the
    pipeline). Stripping and empty-to-null always apply.
    """
    if raw is None:
        return None, []
    if not isinstance(raw, str):
        raw = str(raw)
    s = raw.strip()
    if not s:
        return None, ["empty_to_null"]
    if dates and name in DATE_FIELDS:
        return normalize_date(s, date_order)
    if numbers and name in NUMBER_FIELDS:
        return normalize_number(s)
    if codes and name in CODE_FIELDS:
        return normalize_code(s)
    if codes and name in AIRPORT_FIELDS:
        return normalize_code(s, airport=True)
    return s, []


def normalize_doc(
    doc: dict[str, Any],
    *,
    dates: bool = True,
    numbers: bool = True,
    codes: bool = True,
    date_order: str | None = None,
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """Normalise a merged document; returns the new doc and ``{path: [flags]}`` (only non-empty).

    The keyword switches are forwarded to `normalize_value`; `date_order` is the layout cluster's
    inferred day/month order (None: ambiguous dates stay raw).

    Paths look like ``header.invoice_date`` and ``line_items[3].quantity`` (0-based row index).
    """
    kw: dict[str, Any] = {
        "dates": dates,
        "numbers": numbers,
        "codes": codes,
        "date_order": date_order,
    }
    flags: dict[str, list[str]] = {}
    header: dict[str, Any] = {}
    for k, v in doc["header"].items():
        header[k], fl = normalize_value(k, v, **kw)
        if fl:
            flags[f"header.{k}"] = fl
    rows: list[dict[str, Any]] = []
    for i, row in enumerate(doc["line_items"]):
        new: dict[str, Any] = {}
        for k, v in row.items():
            new[k], fl = normalize_value(k, v, **kw)
            if fl:
                flags[f"line_items[{i}].{k}"] = fl
        rows.append(new)
    return {**doc, "header": header, "line_items": rows}, flags
