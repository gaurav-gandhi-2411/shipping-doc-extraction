from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from shipdoc import extract as ex
from shipdoc import prompts

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_JSON = ROOT / "assignment" / "schema.json"
needs_schema = pytest.mark.skipif(not SCHEMA_JSON.is_file(), reason="assignment/schema.json absent")


def _gold(n_rows: int = 5, doc_type: str = "invoice") -> dict[str, Any]:
    if doc_type == "invoice":
        header = {
            "invoice_number": "INV-1",
            "invoice_date": "2026-05-25",
            "supplier_name": "Acme",
            "buyer_name": "Buyer",
            "ship_to_name": "Ship",
            "currency": "USD",
            "total_amount": "100.00",
            "awb_number": None,
        }
    else:
        header = {
            "carrier": "C",
            "mawb": "123-12345678",
            "hawb": None,
            "origin_airport": "ICN",
            "destination_airport": "HKG",
            "shipper_name": "S",
            "consignee_name": "K",
            "pieces": "4",
            "gross_weight_kg": "23.1",
        }
    rows = [
        {
            "supplier_part_number": f"P-{i}",
            "customer_part_number": None,
            "purchase_order": None,
            "quantity": str(i + 1),
        }
        for i in range(n_rows)
    ]
    return {"doc_id": "d1", "doc_type": doc_type, "header": header, "line_items": rows}


def test_header_key_sets_are_disjoint_and_complete() -> None:
    assert set(ex.INVOICE_KEYS).isdisjoint(ex.WAYBILL_KEYS)
    assert len(ex.HEADER_KEYS) == len(ex.INVOICE_KEYS) + len(ex.WAYBILL_KEYS) == 17


def test_page_schema_is_valid_and_strict() -> None:
    schema = ex.page_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema["properties"]["header"]["additionalProperties"] is False
    assert set(schema["required"]) == {"doc_type", "header", "line_items", "page_kind"}
    ok = {
        "doc_type": "invoice",
        "header": dict.fromkeys(ex.HEADER_KEYS),
        "line_items": [],
        "page_kind": "single",
    }
    assert ex.schema_errors(ok) == []
    assert ex.schema_errors({**ok, "page_kind": "bogus"})


@needs_schema
def test_header_and_row_keys_match_assignment_schema() -> None:
    defs = json.loads(SCHEMA_JSON.read_text(encoding="utf-8"))["$defs"]
    assert set(ex.INVOICE_KEYS) == set(defs["invoice_header"]["properties"])
    assert set(ex.WAYBILL_KEYS) == set(defs["waybill_header"]["properties"])
    assert set(ex.ROW_KEYS) == set(defs["line_item"]["properties"])


def test_xgrammar_schema_build_and_grammar_compile() -> None:
    xgr = pytest.importorskip("xgrammar")
    grammar = xgr.Grammar.from_json_schema(json.dumps(ex.page_schema()), any_whitespace=False)
    assert "root" in str(grammar)


def test_parse_page_json_failures_give_none() -> None:
    assert ex.parse_page_json('{"a": 1}') == {"a": 1}
    assert ex.parse_page_json('{"a": 1') is None  # truncated at max_new_tokens
    assert ex.parse_page_json("[1, 2]") is None  # not an object
    assert ex.parse_page_json("") is None


def test_nuextract_template_mirrors_page_schema() -> None:
    t = ex.nuextract_template()
    assert list(t) == ["doc_type", "header", "line_items", "page_kind"]
    assert set(t["header"]) == set(ex.HEADER_KEYS)
    assert set(t["line_items"][0]) == set(ex.ROW_KEYS)
    assert t["doc_type"] == ["invoice", "waybill"]
    assert set(t["header"].values()) == {"verbatim-string"}


def test_adapters_chat_kwargs() -> None:
    img = object()
    msgs, kw = ex.get_adapter("qwen35").build(img, "PROMPT", ex.page_schema())
    assert kw == {"enable_thinking": False}
    assert [c["type"] for c in msgs[0]["content"]] == ["image", "text"]
    _, kw = ex.get_adapter("qwen3vl").build(img, "PROMPT", ex.page_schema())
    assert kw == {}
    msgs, kw = ex.get_adapter("nuextract3").build(img, "PROMPT", ex.page_schema())
    assert [c["type"] for c in msgs[0]["content"]] == ["image"]
    assert kw["instructions"] == "PROMPT" and kw["enable_thinking"] is False
    assert json.loads(kw["template"]) == ex.nuextract_template()
    with pytest.raises(KeyError, match="valid"):
        ex.get_adapter("nope")


def test_hf_backend_construction_imports_no_heavy_deps() -> None:
    import sys

    cfg = ex.BackendConfig(key="k", repo="r", revision="rev", adapter="qwen35")
    backend = ex.HfBackend(cfg)
    assert backend.revision == "rev"
    if "torch" not in sys.modules:  # only meaningful when the vlm group is not installed
        assert "xgrammar" not in sys.modules


def test_mock_pages_follow_corpus_layout() -> None:
    gold = {"d1": _gold(5)}
    mock = ex.MockBackend(gold)
    pages = [mock.page_payload("d1", i, 2) for i in range(2)]
    assert pages[0]["page_kind"] == "first" and pages[1]["page_kind"] == "continuation"
    assert pages[0]["header"]["invoice_number"] == "INV-1"
    assert pages[0]["header"]["total_amount"] is None  # totals only on the last page
    assert pages[1]["header"]["total_amount"] == "100.00"
    assert pages[1]["header"]["invoice_number"] is None
    assert [len(p["line_items"]) for p in pages] == [3, 2]
    assert ex.schema_errors(pages[0]) == []
    single = mock.page_payload("d1", 0, 1)
    assert single["page_kind"] == "single" and single["header"]["total_amount"] == "100.00"


def test_mock_extract_page_meta_and_corruptions() -> None:
    gold = {"d1": _gold(2)}
    mock = ex.MockBackend(
        gold, ex.MockCorruption(banner_leak=True, invalid_json_every=2, drop_last_rows=1)
    )
    with pytest.raises(RuntimeError, match="set_context"):
        mock.extract_page(None, "p", ex.page_schema(), None)
    mock.set_context("d1", 1, 2)
    raw, parsed, meta = mock.extract_page(None, "prompt", ex.page_schema(), "line one\nline two")
    assert parsed is not None and parsed["header"]["invoice_number"] == "INV-1"  # leak
    assert parsed["line_items"] == []  # the only row on page 2 was dropped
    assert meta["peak_vram_bytes"] is None and meta["ocr_truncated"] is False
    raw2, parsed2, _ = mock.extract_page(None, "prompt", ex.page_schema(), None)  # 2nd call
    assert parsed2 is None and raw2 and not raw2.endswith("}")


def test_prompt_rules_verbatim_and_hash_stable() -> None:
    p = prompts.build_prompt(1, 3)
    for rule in (*prompts.RULES, prompts.NULL_RULE, prompts.PROVENANCE_RULE):
        assert rule in p
    assert "Output exactly what is on the page. Don't correct, complete or infer values." in p
    assert "page 2 of 3" in p
    assert prompts.prompt_hash() == prompts.prompt_hash()
    assert len(prompts.prompt_hash()) == 64
    assert prompts.PROMPT_VERSION == "v2"


@needs_schema
def test_prompt_field_definitions_match_schema() -> None:
    defs = json.loads(SCHEMA_JSON.read_text(encoding="utf-8"))["$defs"]
    for block, items in (
        ("invoice_header", prompts.INVOICE_HEADER),
        ("waybill_header", prompts.WAYBILL_HEADER),
        ("line_item", prompts.ROW_FIELDS),
    ):
        props = defs[block]["properties"]
        for key, definition in items:
            desc = props[key].get("description")
            if desc and key != "invoice_date":  # invoice_date: asked as printed, ISO afterwards
                assert desc in definition, (key, desc, definition)


def test_ocr_block_truncation_and_budget() -> None:
    text = "\n".join(f"line {i:03d} abcdefgh" for i in range(100))  # 16 chars -> 4 tok + 1
    block, info = prompts.ocr_block(text, 50)
    assert info["ocr_truncated"] is True and info["ocr_tokens"] <= 50
    assert block.startswith(prompts.OCR_HEADER) and "line 000" in block
    assert "line 099" not in block
    block, info = prompts.ocr_block("short\ntext", 1200)
    assert info["ocr_truncated"] is False and "short\ntext" in block
    assert prompts.ocr_block("", 10) == (
        "",
        {"ocr_tokens": 0, "ocr_budget": 10, "ocr_truncated": False},
    )
    # custom tokenizer counter: one token per word
    _, info = prompts.ocr_block("a b c\nd e f", 4, count_tokens=lambda s: len(s.split()))
    assert info["ocr_truncated"] is True


# --- grammar key order (spike diagnosis: json.dumps(sort_keys=True) forced alphabetical keys) ---


def test_schema_text_keeps_declared_order_not_sorted() -> None:
    schema = json.loads(ex.schema_text(ex.page_schema()))
    assert list(schema["properties"]) == ["doc_type", "header", "line_items", "page_kind"]
    header = list(schema["properties"]["header"]["properties"])
    assert header == list(ex.HEADER_KEYS)
    assert header != sorted(header)
    assert header[:3] == ["invoice_number", "invoice_date", "supplier_name"]  # identity first
    assert header[-3:] == ["total_amount", "pieces", "gross_weight_kg"]  # totals last
    row = list(schema["properties"]["line_items"]["items"]["properties"])
    assert row == [
        "supplier_part_number",
        "customer_part_number",
        "purchase_order",
        "quantity",
    ]


def test_compact_schema_text_keeps_declared_order() -> None:
    schema = json.loads(ex.schema_text(ex.compact_page_schema()))
    assert list(schema["properties"]) == ["dt", "h", "r", "pk"]
    assert list(schema["properties"]["h"]["properties"]) == [
        ex.SHORT_HEADER_KEYS[k] for k in ex.HEADER_KEYS
    ]
    # rows are positional arrays: [spn, cpn, po, qty]
    assert ex.ROW_KEYS == (
        "supplier_part_number",
        "customer_part_number",
        "purchase_order",
        "quantity",
    )
    assert len(schema["properties"]["r"]["items"]["prefixItems"]) == 4


def test_compiler_receives_unsorted_schema_and_cache_key_is_a_hash() -> None:
    class FakeCompiler:
        def __init__(self) -> None:
            self.seen: list[str] = []

        def compile_json_schema(self, text: str, any_whitespace: bool) -> str:
            self.seen.append(text)
            return f"grammar{len(self.seen)}"

    cfg = ex.BackendConfig(key="k", repo="r", revision="0" * 40, adapter="qwen3vl")
    backend = ex.HfBackend(cfg)
    backend.compiler = FakeCompiler()
    assert backend._compiled(ex.page_schema()) == "grammar1"
    assert backend._compiled(ex.page_schema()) == "grammar1"  # cached, compiled once
    assert len(backend.compiler.seen) == 1
    compiled = json.loads(backend.compiler.seen[0])
    assert list(compiled["properties"]["header"]["properties"]) == list(ex.HEADER_KEYS)
    assert list(compiled["properties"]["header"]["properties"]) != sorted(ex.HEADER_KEYS)
    assert all(len(k) == 64 for k in backend._grammar_cache)  # sha256 hex, not the schema text


def test_real_xgrammar_enforces_declared_key_order() -> None:
    """Empirical check of the claim in `schema_text` (skipped without the vlm/gpu-local group)."""
    pytest.importorskip("xgrammar")
    import xgrammar as xgr
    from xgrammar.testing import _is_grammar_accept_string

    grammar = xgr.Grammar.from_json_schema(ex.schema_text(ex.page_schema()), any_whitespace=False)
    header = dict.fromkeys(ex.HEADER_KEYS)

    def accepts(row: dict[str, Any]) -> bool:
        page = {"doc_type": "invoice", "header": header, "line_items": [row], "page_kind": "single"}
        return bool(_is_grammar_accept_string(grammar, json.dumps(page)))

    declared = dict.fromkeys(ex.ROW_KEYS)
    assert accepts(declared)
    assert not accepts(dict(sorted(declared.items())))  # alphabetical rows are rejected


# --- numeric fields may be bare JSON numbers (spike40: the string-only grammar masked digits) ---


def _page_with(header_total: Any, qty: Any, spn: Any = "A-1") -> dict[str, Any]:
    header = dict.fromkeys(ex.HEADER_KEYS)
    header["total_amount"] = header_total
    row = {"supplier_part_number": spn, "customer_part_number": None, "purchase_order": None}
    return {
        "doc_type": "invoice",
        "header": header,
        "line_items": [{**row, "quantity": qty}],
        "page_kind": "single",
    }


def test_numeric_fields_accept_numbers_others_do_not() -> None:
    assert {"total_amount", "pieces", "gross_weight_kg", "quantity"} == ex.NUMERIC_KEYS
    assert ex.schema_errors(_page_with(1234.5, 250)) == []
    assert ex.schema_errors(_page_with("1,234.50", "250")) == []  # strings still fine
    assert ex.schema_errors(_page_with(None, None)) == []
    assert ex.schema_errors(_page_with(1.0, 1, spn=7))  # a part number is never a number
    schema = ex.page_schema()["properties"]
    assert schema["header"]["properties"]["invoice_number"]["type"] == ["string", "null"]
    assert "number" in schema["header"]["properties"]["gross_weight_kg"]["type"]


def test_numbers_flow_through_merge_and_normalize_as_plain_strings() -> None:
    from shipdoc.merge import merge_pages
    from shipdoc.normalize import normalize_doc

    doc, _ = normalize_doc(merge_pages([_page_with(1234.5, 250)]).doc)
    assert doc["header"]["total_amount"] == "1234.5"
    assert doc["line_items"][0]["quantity"] == "250"


def test_real_xgrammar_accepts_a_bare_number_quantity() -> None:
    pytest.importorskip("xgrammar")
    import xgrammar as xgr
    from xgrammar.testing import _is_grammar_accept_string

    grammar = xgr.Grammar.from_json_schema(ex.schema_text(ex.page_schema()), any_whitespace=False)
    assert _is_grammar_accept_string(grammar, json.dumps(_page_with(1234.5, 250)))
    assert _is_grammar_accept_string(grammar, json.dumps(_page_with("1,234.50", "250")))
    assert not _is_grammar_accept_string(grammar, json.dumps(_page_with(1.0, 1, spn=7)))


# --- prompt v2 -------------------------------------------------------------------------------

V1_JSON_PROMPT_HASH = "564657d0227d991708ae83a18205f1d8c3c1bbc0696e0b36b54773a417d2344a"  # trace


@pytest.mark.parametrize("fmt", ["json", "compact"])
def test_v2_prompt_carries_every_convention_rule(fmt: str) -> None:
    text = prompts.build_prompt(0, 1, fmt)
    assert prompts.PROMPT_VERSION == "v2"
    for rule in prompts.CONVENTION_RULES:
        assert f"- {rule}" in text
    assert "all null" in text and "never move a value" in text


def test_v2_changes_the_prompt_hash_and_rules_cover_the_numeric_fields() -> None:
    assert prompts.prompt_hash("json") != V1_JSON_PROMPT_HASH
    rules = " ".join(prompts.CONVENTION_RULES)
    assert all(k in rules for k in ex.NUMERIC_KEYS)  # the rule and the schema name the same fields
