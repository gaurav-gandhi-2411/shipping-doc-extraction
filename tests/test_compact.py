"""Compact output format: schema, deterministic expander, round trips, prompt, backend wiring."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest
import yaml

from shipdoc import extract as ex
from shipdoc import prompts, spike
from shipdoc.merge import merge_pages
from shipdoc.targets import gold_page_payloads

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
needs_data = pytest.mark.skipif(not (DATA / "train" / "labels").is_dir(), reason="no data/")


def _gold(rows: list[dict] | None = None, doc_type: str = "invoice") -> dict:
    if doc_type == "waybill":
        header = dict.fromkeys(ex.WAYBILL_KEYS, "x")
        header["hawb"] = None
        return {"doc_type": "waybill", "header": header, "line_items": []}
    header = dict.fromkeys(ex.INVOICE_KEYS, "v")
    header["awb_number"] = None
    header["total_amount"] = "10.00"
    if rows is None:
        rows = [
            {
                "supplier_part_number": "A-1",
                "customer_part_number": None,
                "purchase_order": "PO 7",
                "quantity": "5",
            },
            {
                "supplier_part_number": "B-2",
                "customer_part_number": "C-9",
                "purchase_order": None,
                "quantity": "1,000",
            },
        ]
    return {"doc_type": "invoice", "header": header, "line_items": rows}


def test_short_keys_are_unique_and_cover_every_header_key() -> None:
    assert set(ex.SHORT_HEADER_KEYS) == set(ex.HEADER_KEYS)
    assert len(set(ex.SHORT_HEADER_KEYS.values())) == len(ex.HEADER_KEYS)


def test_compact_page_shape() -> None:
    page = gold_page_payloads(_gold(), 1)[0]
    c = ex.compact_page(page)
    assert list(c) == ["dt", "h", "r", "pk"]
    assert c["r"] == [["A-1", None, "PO 7", "5"], ["B-2", "C-9", None, "1,000"]]
    assert c["h"]["inv_no"] == "v" and c["h"]["awb"] is None and c["h"]["gw"] is None
    jsonschema.validate(c, ex.compact_page_schema())


def test_roundtrip_nulls_preserved_and_empty_rows() -> None:
    for gold in (_gold(), _gold(rows=[]), _gold(doc_type="waybill")):
        for page in gold_page_payloads(gold, 2):
            assert ex.expand_page(ex.compact_page(page)) == page
            assert ex.schema_errors(ex.expand_page(ex.compact_page(page))) == []
    page = gold_page_payloads(_gold(), 1)[0]
    out = ex.expand_page(json.loads(json.dumps(ex.compact_page(page))))
    assert out["header"]["awb_number"] is None
    assert out["line_items"][0]["customer_part_number"] is None  # null stays null, not ""


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c["r"].append(["only", "three", "items"]),
        lambda c: c["r"].append("not a list"),
        lambda c: c["h"].update(bogus="x"),
        lambda c: c.pop("pk"),
        lambda c: c.update(h=[]),
    ],
)
def test_expand_rejects_malformed(mutate) -> None:
    c = ex.compact_page(gold_page_payloads(_gold(), 1)[0])
    mutate(c)
    with pytest.raises(ValueError):
        ex.expand_page(c)


def test_parse_output_compact_and_failures() -> None:
    page = gold_page_payloads(_gold(), 1)[0]
    raw = json.dumps(ex.compact_page(page))
    assert ex.parse_output(raw, "compact") == page
    assert ex.parse_output(raw[:-5], "compact") is None  # truncated at max_new_tokens
    assert ex.parse_output('{"dt": "invoice"}', "compact") is None  # not a compact page
    assert ex.parse_output(json.dumps(page), "json") == page


def test_compact_schema_rows_are_positional_arrays() -> None:
    row = ex.compact_page_schema()["properties"]["r"]["items"]
    assert row["minItems"] == row["maxItems"] == 4 and row["items"] is False
    assert [x["type"] for x in row["prefixItems"]] == [
        ["string", "null"],
        ["string", "null"],
        ["string", "null"],
        ["string", "number", "null"],  # quantity may be a bare number
    ]
    c = ex.compact_page(gold_page_payloads(_gold(), 1)[0])
    c["r"][0] = c["r"][0][:3]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(c, ex.compact_page_schema())


def test_compact_schema_grammar_pins_row_arity() -> None:
    """xgrammar 0.2.8 compiles prefixItems and enforces positional arity (when installed)."""
    xgr = pytest.importorskip("xgrammar")
    testing = pytest.importorskip("xgrammar.testing")
    g = xgr.Grammar.from_json_schema(json.dumps(ex.compact_page_schema()), any_whitespace=False)
    page = gold_page_payloads(_gold(), 1)[0]
    text = json.dumps(ex.compact_page(page), separators=(", ", ": "))
    assert testing._is_grammar_accept_string(g, text)
    assert not testing._is_grammar_accept_string(g, text.replace('"5"]', '"5", null]'))
    assert not testing._is_grammar_accept_string(g, text.replace('"PO 7", ', ""))
    assert testing._is_grammar_accept_string(g, text.replace('"5"]', "5]"))  # numeric quantity
    assert not testing._is_grammar_accept_string(g, text.replace('"A-1"', "7"))  # not a part no.


@needs_data
def test_roundtrip_all_train_dev_docs_per_page_and_merged() -> None:
    from shipdoc import meta

    n = 0
    for split in ("train", "dev"):
        for gold in meta.load_labels(split):
            n_pages = len(gold.get("pages") or [None])
            pages = gold_page_payloads(gold, n_pages)
            expanded = [ex.expand_page(ex.compact_page(p)) for p in pages]
            assert expanded == pages, gold["doc_id"]
            merged = merge_pages(expanded).doc
            assert merged == merge_pages(pages).doc, gold["doc_id"]
            assert merged["doc_type"] == gold["doc_type"]
            assert merged["header"] == gold["header"], gold["doc_id"]
            assert merged["line_items"] == gold["line_items"], gold["doc_id"]
            n += 1
    assert n == 500


def test_compact_prompt_documents_keys_and_keeps_rules() -> None:
    p = prompts.build_prompt(0, 2, "compact")
    for rule in (*prompts.RULES, prompts.NULL_RULE, prompts.PROVENANCE_RULE):
        assert rule in p
    for short in ex.SHORT_HEADER_KEYS.values():
        assert f"- {short}:" in p
    assert "exactly 4 items" in p and "page 1 of 2" in p
    assert prompts.prompt_hash("compact") != prompts.prompt_hash("json")
    assert prompts.build_prompt(0, 1) == prompts.build_prompt(0, 1, "json")
    with pytest.raises(ValueError):
        prompts.build_prompt(0, 1, "yaml")


def test_nuextract_adapter_uses_compact_template() -> None:
    ad = ex.get_adapter("nuextract3")
    _, kw = ad.build(None, "P", ex.compact_page_schema(), "compact")
    assert json.loads(kw["template"]) == ex.nuextract_compact_template()
    _, kw = ad.build(None, "P", ex.page_schema())
    assert json.loads(kw["template"]) == ex.nuextract_template()


def test_mock_backend_compact_returns_expanded_pages() -> None:
    gold = {"d1": _gold()}
    mock = ex.MockBackend(gold, output_format="compact")
    mock.set_context("d1", 0, 2)
    raw, parsed, meta = mock.extract_page(None, "p", ex.compact_page_schema(), None)
    assert json.loads(raw)["dt"] == "invoice" and "line_items" not in json.loads(raw)
    assert parsed == mock.page_payload("d1", 0, 2)
    assert meta["n_output_tokens"] > 0
    plain = ex.MockBackend(gold)
    plain.set_context("d1", 0, 2)
    assert len(plain.extract_page(None, "p", ex.page_schema(), None)[0]) > len(raw)


def test_shipped_configs_declare_output_format_and_budget() -> None:
    for f in sorted((ROOT / "configs").glob("spike_*.yaml")):
        cfg = spike.load_config(f)
        raw = yaml.safe_load(f.read_text(encoding="utf-8"))
        assert cfg.backend.output_format == raw["output_format"] in ex.OUTPUT_FORMATS
        assert cfg.backend.max_new_tokens % 64 == 0, f.name
        assert cfg.name.endswith("_compact") == (raw["output_format"] == "compact")


def test_bad_output_format_is_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(
        (ROOT / "configs" / "spike_qwen35_4b_img_only.yaml").read_text(encoding="utf-8")
    )
    raw["output_format"] = "yaml"
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="output_format"):
        spike.load_config(bad)
    with pytest.raises(ValueError):
        ex.schema_for("yaml")
