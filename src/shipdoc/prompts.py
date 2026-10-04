"""Versioned extraction prompt (spec Phase 2.4).

The rule sentences are the brief's wording, verbatim. Field definitions are copied from
``assignment/schema.json`` descriptions (that folder is gitignored and absent on Colab, so they are
embedded here; ``tests/test_prompts.py`` checks them against the schema when it is present).

Bump `PROMPT_VERSION` on ANY wording change: the trace records version and `prompt_hash()`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable

PROMPT_VERSION = "v2"
# v2 (spike40 diagnosis, reports/spike_diagnosis.md): CONVENTION_RULES added, for the convention
# errors the traces confirm (null / junk quantity, purchase order numbers in customer_part_number,
# fields shifted along the row, all-null rows). v1 is what the 51cf560 runs used.

RULES = (
    "Output exactly what is on the page. Don't correct, complete or infer values.",
    "If a value is missing or illegible, output null. A guessed value scores as wrong, and false "
    "fills are reported separately.",
    "Line items may be in any order, but every row counts, including repeats.",
)
NULL_RULE = (
    "If a value is redacted (blacked out), smudged, scribbled over, cut off at the page edge, or "
    "the line is absent, output null."
)
PROVENANCE_RULE = (
    "Header identity fields (invoice number, date, supplier, buyer, ship-to, AWB) are read from "
    "the first page's header block only. Never fill a header field from a 'continued' banner on "
    "later pages."
)

# General reading conventions, confirmed as error classes on the spike40 traces (counts in
# reports/spike_diagnosis.md). No supplier-specific wording.
CONVENTION_RULES = (
    "Read each table row from left to right and fill its fields from that row's own cells only; "
    "never move a value into a neighbouring field.",
    "supplier_part_number is the seller's part / item code column. customer_part_number is the "
    "buyer's own code column, often labelled 'Cust. Part', 'Customer Part No.' or 'Your Part No.'; "
    "if the table has no such column it is null. A purchase order number is never a part number.",
    "purchase_order is the value in the row's PO column (labelled e.g. 'PO', 'P.O. No.', "
    "'Your Order'): copy it exactly when it is printed.",
    "quantity is the number of units in the row's quantity column. total_amount, pieces and "
    "gross_weight_kg are numbers too. Copy these digits as printed; they may be written as a "
    "JSON number or a quoted string. Output null only when the cell is empty or unreadable.",
    "Never output a row whose fields are all null; end the list after the last printed row.",
)

# Ordered (field, definition). Definitions follow schema.json; where the schema says nothing the
# definition is the field's plain meaning. Dates/numbers are asked for AS PRINTED: shipdoc.normalize
# converts them afterwards, so the model never has to reformat (verbatim rule).
INVOICE_HEADER: tuple[tuple[str, str], ...] = (
    ("invoice_number", "the invoice number"),
    ("invoice_date", "the invoice date, exactly as printed (converted to ISO afterwards)"),
    ("supplier_name", "the seller / supplier name"),
    ("buyer_name", "the buyer name"),
    ("ship_to_name", "the ship-to name"),
    (
        "currency",
        "ISO 4217 code, e.g. USD, as printed; if only a symbol such as $ is printed, output null",
    ),
    ("total_amount", "invoice total as a plain number, e.g. 1234.50"),
    ("awb_number", "air waybill number referenced on the invoice"),
)
WAYBILL_HEADER: tuple[tuple[str, str], ...] = (
    ("carrier", "the carrier name"),
    ("mawb", "the master air waybill number"),
    ("hawb", "the house air waybill number"),
    ("origin_airport", "IATA code"),
    ("destination_airport", "IATA code"),
    ("shipper_name", "the shipper name"),
    ("consignee_name", "the consignee name"),
    ("pieces", "number of pieces"),
    ("gross_weight_kg", "gross weight in kilograms"),
)
ROW_FIELDS: tuple[tuple[str, str], ...] = (
    ("supplier_part_number", "the seller's / manufacturer's part number"),
    ("customer_part_number", "the buyer's own part number, if printed"),
    ("purchase_order", "PO number for this row (may be printed per row or once for the invoice)"),
    ("quantity", "units shipped, as a plain number"),
)
PAGE_KINDS: dict[str, str] = {
    "first": "the first page of a multi-page document (it carries the header block)",
    "continuation": "a later page of a multi-page document (e.g. a 'continued' banner)",
    "single": "a one-page document",
    "other": "anything else",
}

OCR_HEADER = "OCR text (may contain errors; use only as a hint, the image is authoritative)"


def _defs(items: tuple[tuple[str, str], ...]) -> str:
    return "\n".join(f"- {k}: {d}" for k, d in items)


def _template() -> str:
    """Static prompt text (everything except the per-page note and the OCR text)."""
    rules = "\n".join(f"- {r}" for r in (*RULES, NULL_RULE, PROVENANCE_RULE, *CONVENTION_RULES))
    kinds = "\n".join(f"- {k}: {d}" for k, d in PAGE_KINDS.items())
    return (
        "You read one page image of a shipping document (a commercial invoice or an air waybill) "
        "and return JSON.\n\n"
        f"Rules:\n{rules}\n\n"
        'Output keys: doc_type ("invoice" or "waybill"), header, line_items, page_kind.\n'
        "header always has every key below. Fill the keys of this document's type and set the "
        "keys of the other type to null.\n\n"
        f"Invoice header keys:\n{_defs(INVOICE_HEADER)}\n\n"
        f"Waybill header keys:\n{_defs(WAYBILL_HEADER)}\n\n"
        f"line_items: one object per table row on this page (empty for waybills), keys:\n"
        f"{_defs(ROW_FIELDS)}\n\n"
        f"page_kind:\n{kinds}"
    )


def _compact_template() -> str:
    """Static prompt text of the compact output format (same rules and field definitions)."""
    from shipdoc.extract import SHORT_HEADER_KEYS  # local: extract imports this module

    def short_defs(items: tuple[tuple[str, str], ...]) -> str:
        return "\n".join(f"- {SHORT_HEADER_KEYS[k]}: {d}" for k, d in items)

    rules = "\n".join(f"- {r}" for r in (*RULES, NULL_RULE, PROVENANCE_RULE, *CONVENTION_RULES))
    kinds = "\n".join(f"- {k}: {d}" for k, d in PAGE_KINDS.items())
    row_defs = "\n".join(f"{i + 1}. {k}: {d}" for i, (k, d) in enumerate(ROW_FIELDS))
    return (
        "You read one page image of a shipping document (a commercial invoice or an air waybill) "
        "and return compact JSON.\n\n"
        f"Rules:\n{rules}\n\n"
        'Output keys: dt (doc type, "invoice" or "waybill"), h (header), r (line items), '
        "pk (page kind).\n"
        "h always has every key below. Fill the keys of this document's type and set the "
        "keys of the other type to null.\n\n"
        f"Invoice h keys:\n{short_defs(INVOICE_HEADER)}\n\n"
        f"Waybill h keys:\n{short_defs(WAYBILL_HEADER)}\n\n"
        "r: one array per table row on this page (empty for waybills). Each row is an array of "
        "exactly 4 items in this order, null where a value is missing:\n"
        f"{row_defs}\n"
        'Example row: ["PN-100", null, "PO-7", "250"]\n\n'
        f"pk (page kind):\n{kinds}"
    )


def _template_for(output_format: str) -> str:
    if output_format == "compact":
        return _compact_template()
    if output_format == "json":
        return _template()
    raise ValueError(f"unknown output_format {output_format!r} (use 'json' or 'compact')")


def prompt_hash(output_format: str = "json") -> str:
    """sha256 of every static prompt string (version-independent of page/OCR content).

    The ``json`` hash is byte-identical to the one before the compact format existed.
    """
    return hashlib.sha256(
        (_template_for(output_format) + "\n" + OCR_HEADER).encode("utf-8")
    ).hexdigest()


def build_prompt(page_index: int = 0, n_pages: int = 1, output_format: str = "json") -> str:
    """Prompt for page `page_index` (0-based) of an `n_pages`-page document."""
    return (
        f"{_template_for(output_format)}\n\n"
        f"This is page {page_index + 1} of {n_pages} of the document."
    )


def ocr_block(
    text: str,
    budget_tokens: int,
    count_tokens: Callable[[str], int] | None = None,
) -> tuple[str, dict[str, int | bool]]:
    """Reading-order OCR hint truncated line-wise to `budget_tokens`.

    `count_tokens` is the model tokenizer in the HF backend; None means the 4-chars/token
    approximation (mock). Lines are kept from the top of the page until the next line would exceed
    the budget (a line is never cut in half). Returns the block text ("" if no text) and
    ``{"ocr_tokens", "ocr_budget", "ocr_truncated"}`` for the trace.
    """
    count = count_tokens or (lambda s: -(-len(s) // 4))
    kept: list[str] = []
    used = 0
    truncated = False
    for line in text.splitlines():
        n = count(line) + 1  # +1: the newline joining it to the previous line
        if used + n > budget_tokens:
            truncated = True
            break
        kept.append(line)
        used += n
    info: dict[str, int | bool] = {
        "ocr_tokens": used,
        "ocr_budget": budget_tokens,
        "ocr_truncated": truncated,
    }
    if not kept:
        return "", info
    return f"{OCR_HEADER}:\n<ocr>\n" + "\n".join(kept) + "\n</ocr>", info
