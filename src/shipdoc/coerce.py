"""Canonical plain-number strings for the numeric fields; the single choke point before any write.

``assignment/schema.json`` types EVERY field ``string | null``, but the decoding grammar lets
the four numeric fields (``total_amount``, ``pieces``, ``gross_weight_kg``, ``quantity``;
``NUMERIC_KEYS``) be a bare JSON number (fix 3d06306). The official scorer reads numbers with
``float(re.sub(r"[,\\s]", "", str(v)))`` and compares within 0.005, so a number or a string scores
the same; the SCHEMA does not accept a number. This module makes the written value a string.

Rules (all pure functions, no I/O):

* ``None`` stays ``None``. ``bool`` / NaN / +-inf (``float``, ``Decimal``) raise `CoerceError`
  (fail closed: nothing is written). Strings that merely look like "nan" are strings, left alone.
* ``int`` -> ``str(int)`` (exact, any size). ``float`` (no token text exists) -> 15 significant
  digits, never exponent notation, a zero fraction dropped: ``12.0 -> "12"``,
  ``0.1 + 0.2 -> "0.3"``, ``1e-7 -> "0.0000001"``. 15 digits is the most a double carries exactly
  for every decimal string, and no quantity / total in this corpus has more.
* ``Decimal`` (parsed with the token text) -> its digits as printed, exponent expanded, trailing
  zeros KEPT: ``Decimal("1234.50") -> "1234.50"``.
* ``str`` in a numeric field is the model's token text: untouched, except a JSON number literal
  WITH an exponent (``"1E+6"``) which is expanded (``"1000000"``). Commas, spaces, signs, "-0" and
  every other string are left for the Phase 3 normaliser / the scorer: the scorer is numeric, so
  ``"10"`` / ``"10.00"`` and ``"1234.5"`` / ``"1234.50"`` score identically (tests run the scorer).
  Trailing zeros are never stripped from a string or a token (gold ``total_amount`` has exactly two
  decimals in 400/400 docs; ``"1234.50"`` must stay).
* Non-numeric fields (names, ids, part numbers, PO numbers, dates, codes) must already be
  ``str | None`` and are returned byte for byte; anything else raises (a digit-only part number
  that arrives as a JSON number has lost its leading zeros: that is an error, not a repair).
  The grammar allows numbers for the four numeric keys only, so this cannot happen in a run.

`coerce(coerce(x)) == coerce(x)`: outputs are plain digit strings, which the rules leave alone.

Schema repair (`repair_doc` / `repair_predictions`, called from the same writers right after the
coercion): POLICY DECISION. ``assignment/schema.json`` gives ``invoice_date`` a ``pattern``
(``^\\d{4}-\\d{2}-\\d{2}$``). One model-misread date (e.g. ``"2470"``) fails it, and assembling the
submission rejects the WHOLE file for that one value. A date field (found by reading the schema:
a property with a ``pattern`` that is a date by ``format``, name or description; nothing is
hardcoded) whose string does not satisfy the schema pattern is replaced with ``null``. This is a
value edit, but only of values that can NEVER be correct under the schema (gold dates always match
the pattern). The rule applies to nothing else: no other field is ever repaired, a value that
satisfies the pattern is never touched (``"2026-13-45"`` matches the pattern and stays: the schema
does not say it is wrong), ``null`` stays ``null`` and every other schema failure still rejects.
Each repair is an event ``{"doc_id", "field", "reason": "schema_invalid_date"}``, NEVER the value;
events go to the trace (``schema_repairs``), the run manifest and the validation report (counts).
The repair is idempotent. An unreadable / pattern-less schema repairs nothing (fail closed on the
submission check, which still validates the schema).
"""

from __future__ import annotations

import functools
import json
import math
import re
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from shipdoc import paths
from shipdoc.extract import NUMERIC_KEYS, parse_output

NUMERIC_FIELDS = NUMERIC_KEYS
#: A JSON number literal (RFC 8259) with an exponent: the only string form that is rewritten.
_EXP_LITERAL = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?[eE][+-]?\d+")
#: Largest decimal exponent expanded: a hostile "1e999999999" must not allocate a giant string.
MAX_EXPONENT = 60
FLOAT_DIGITS = 15


class CoerceError(ValueError):
    """A value cannot be written (bool, NaN, inf, wrong type). Messages never quote a value."""


def _plain(d: Decimal, where: str) -> str:
    """`d` as plain digits (no exponent). A zero never carries a sign."""
    if not d.is_finite():
        raise CoerceError(f"{where}: non-finite number")
    if abs(d.adjusted()) > MAX_EXPONENT or -d.as_tuple().exponent > MAX_EXPONENT:  # type: ignore[operator]
        raise CoerceError(f"{where}: exponent out of range (|e| > {MAX_EXPONENT})")
    s = format(d, "f")
    return s.removeprefix("-") if d.is_zero() else s


def coerce_number(value: Any, where: str = "value") -> str:
    """Canonical plain-number string of an int / float / Decimal / numeric-field string."""
    if isinstance(value, bool):
        raise CoerceError(f"{where}: bool is not a number")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CoerceError(f"{where}: non-finite number")
        s = _plain(Decimal(format(value, f".{FLOAT_DIGITS}g")), where)
        return s.rstrip("0").rstrip(".") if "." in s else s
    if isinstance(value, Decimal):
        return _plain(value, where)
    if isinstance(value, str):
        if _EXP_LITERAL.fullmatch(value):
            try:
                return _plain(Decimal(value), where)
            except InvalidOperation as exc:  # pragma: no cover - the regex admits valid input only
                raise CoerceError(f"{where}: unreadable number") from exc
        return value
    raise CoerceError(f"{where}: {type(value).__name__} is not a number")


def coerce_value(field: str, value: Any) -> str | None:
    """One field value: numeric fields via `coerce_number`; all others must be ``str | None``."""
    if value is None:
        return None
    if field in NUMERIC_FIELDS:
        return coerce_number(value, field)
    if isinstance(value, str):
        return value
    raise CoerceError(f"{field}: expected string or null, got {type(value).__name__}")


def _coerce_mapping(obj: Any, where: str) -> dict[str, Any]:
    if not isinstance(obj, Mapping):
        raise CoerceError(f"{where}: expected an object, got {type(obj).__name__}")
    return {k: coerce_value(k, v) for k, v in obj.items()}


def coerce_doc(doc: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of a prediction document with every header / row value coerced (`doc` untouched)."""
    out = dict(doc)
    out["header"] = _coerce_mapping(doc.get("header"), "header")
    rows = doc.get("line_items")
    if not isinstance(rows, list):
        raise CoerceError("line_items: expected an array")
    out["line_items"] = [_coerce_mapping(r, f"line_items[{i}]") for i, r in enumerate(rows)]
    return out


def coerce_predictions(preds: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """`coerce_doc` over ``{doc_id: document}``; the choke point every writer calls."""
    out: dict[str, dict[str, Any]] = {}
    for doc_id, doc in preds.items():
        try:
            out[doc_id] = coerce_doc(doc)
        except CoerceError as exc:
            raise CoerceError(f"{doc_id}: {exc}") from exc
    return out


# --------------------------------------------------------------------------------------------
# Schema repair: a date that can never satisfy the schema pattern becomes null
# --------------------------------------------------------------------------------------------

REPAIR_REASON = "schema_invalid_date"


def _is_date_property(name: str, prop: Mapping[str, Any]) -> bool:
    text = f"{name} {prop.get('description', '')}".lower()
    return "pattern" in prop and (prop.get("format") in ("date", "date-time") or "date" in text)


def _ecma_pattern(pattern: str) -> re.Pattern[str]:
    """A JSON-Schema (ECMA 262) pattern as a Python regex: ASCII digits, ``$`` = end of string."""
    if pattern.endswith("$") and not pattern.endswith("\\$"):
        pattern = pattern[:-1] + "\\Z"
    return re.compile(pattern, re.ASCII)


def date_rules(schema: Mapping[str, Any]) -> dict[str, re.Pattern[str]]:
    """``{field: compiled pattern}`` of every date property of `schema` (read, not hardcoded)."""
    rules: dict[str, re.Pattern[str]] = {}
    defs = schema.get("$defs") or schema.get("definitions") or {}
    for node in defs.values():
        for name, prop in (node.get("properties") or {}).items() if isinstance(node, dict) else []:
            if isinstance(prop, dict) and _is_date_property(name, prop):
                rules[name] = _ecma_pattern(str(prop["pattern"]))
    return rules


@functools.lru_cache(maxsize=4)
def _rules_from(path: str) -> dict[str, re.Pattern[str]]:
    return date_rules(json.loads(Path(path).read_text(encoding="utf-8")))


def default_date_rules() -> dict[str, re.Pattern[str]]:
    """`date_rules` of ``<assignment dir>/schema.json`` (cached per path)."""
    return _rules_from(str(paths.assignment_dir() / "schema.json"))


def repair_doc(
    doc: Mapping[str, Any], rules: Mapping[str, re.Pattern[str]] | None = None
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """A copy of `doc` with every schema-invalid date set to null, and the events (no values)."""
    rules = default_date_rules() if rules is None else rules
    out = dict(doc)
    events: list[dict[str, str]] = []
    header = doc.get("header")
    if isinstance(header, Mapping) and rules:
        fixed = dict(header)
        for field, pat in rules.items():
            v = fixed.get(field)
            if isinstance(v, str) and not pat.search(v):
                fixed[field] = None
                events.append({"field": field, "reason": REPAIR_REASON})
        out["header"] = fixed
    return out, events


def repair_predictions(
    preds: Mapping[str, Mapping[str, Any]], rules: Mapping[str, re.Pattern[str]] | None = None
) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    """`repair_doc` over ``{doc_id: document}``; events carry the ``doc_id``."""
    rules = default_date_rules() if rules is None else rules
    out: dict[str, dict[str, Any]] = {}
    events: list[dict[str, str]] = []
    for doc_id, doc in preds.items():
        out[doc_id], ev = repair_doc(doc, rules)
        events += [{"doc_id": doc_id, **e} for e in ev]
    return out, events


# --------------------------------------------------------------------------------------------
# Page level: keep the model's own number token text through the merge
# --------------------------------------------------------------------------------------------


def _check_leaves(parsed: Mapping[str, Any]) -> None:
    """Raise on a bool / NaN / inf anywhere in the header or the rows of a parsed page."""
    header = parsed.get("header")
    leaves: list[Any] = list(header.values()) if isinstance(header, Mapping) else []
    for row in parsed.get("line_items") or []:
        if isinstance(row, Mapping):
            leaves.extend(row.values())
    for v in leaves:
        if isinstance(v, bool) or (isinstance(v, float) and not math.isfinite(v)):
            raise CoerceError(f"page value: {type(v).__name__} is not a legal field value")


def page_for_merge(
    raw_text: str, parsed: dict[str, Any] | None, output_format: str = "json"
) -> dict[str, Any] | None:
    """The page to feed `merge_pages`: `parsed`, with JSON numbers as their original token text.

    ``json.loads`` turns ``1234.50`` into the float 1234.5 (the trailing zero is gone) and ``1e6``
    into ``1000000.0``. Re-parsing `raw_text` with ``parse_float=str, parse_int=str`` keeps the
    token text. The re-parse is used only when it equals `parsed` apart from the number types
    (guard against a hand-built `parsed`); otherwise `parsed` is returned unchanged. Raises
    `CoerceError` on a bool / NaN / inf value (the grammar cannot produce one).
    """
    if parsed is None:
        return None
    _check_leaves(parsed)
    tokens = parse_output(raw_text, output_format, number_text=True)
    if tokens is None:
        return parsed
    return tokens if _same_apart_from_numbers(tokens, parsed) else parsed


def _same_apart_from_numbers(tokens: Any, parsed: Any) -> bool:
    """True iff `tokens` (numbers as str) equals `parsed` when every number is read as a Decimal."""
    if isinstance(parsed, dict):
        return (
            isinstance(tokens, dict)
            and list(tokens) == list(parsed)
            and all(_same_apart_from_numbers(tokens[k], parsed[k]) for k in parsed)
        )
    if isinstance(parsed, list):
        return (
            isinstance(tokens, list)
            and len(tokens) == len(parsed)
            and all(_same_apart_from_numbers(a, b) for a, b in zip(tokens, parsed, strict=True))
        )
    if isinstance(parsed, (int, float)) and not isinstance(parsed, bool):
        try:
            return Decimal(str(tokens)) == Decimal(repr(parsed))
        except InvalidOperation:
            return False
    return bool(tokens == parsed)
