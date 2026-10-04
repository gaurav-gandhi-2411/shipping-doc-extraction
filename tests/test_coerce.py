"""Numeric coercion (shipdoc.coerce): plain-number strings before any predictions file is written.

Unit tests of the rules, the page-level token-text path, the saved dev runs (schema + official
scorer before / after) and the three real writers (spike runner, ``shipdoc predict``, merge-shards)
driven by a mock backend that emits JSON NUMBERS. The assignment folder is gitignored: the tests
that need the schema / scorer / saved runs skip cleanly without it.
"""

# ruff: noqa: F811

from __future__ import annotations

import copy
import json
import random
import re
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from test_predict import (
    DEV_IDS,
    N_PAGES,
    RUN,
    SCHEMA,
    TEST_IDS,
    World,
    _clean_sha,  # noqa: F401 - autouse fixture, must be visible in this module
    assemble,
    mock,
    needs_schema,
    run_all,
    world,  # noqa: F401 - fixture
)

from shipdoc import eval as ev
from shipdoc import predict, shardmerge, spike
from shipdoc.coerce import (
    NUMERIC_FIELDS,
    CoerceError,
    coerce_doc,
    coerce_number,
    coerce_predictions,
    coerce_value,
    page_for_merge,
)
from shipdoc.extract import MockBackend, parse_output
from shipdoc.merge import merge_pages
from shipdoc.normalize import normalize_doc

ROOT = Path(__file__).resolve().parents[1]
SCORER = ROOT / "assignment" / "score.py"
SAVED = Path("D:/shipdoc/runs/spike_download2/x")
needs_scorer = pytest.mark.skipif(not SCORER.is_file(), reason="assignment/score.py absent")
needs_saved = pytest.mark.skipif(not SAVED.is_dir(), reason="saved spike_download2 runs absent")


# ---------------------------------------------------------------------- the rules


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (12, "12"),
        (-3, "-3"),
        (0, "0"),
        (2**53 + 1, "9007199254740993"),  # a double cannot hold it; the int path is exact
        (10**30, "1000000000000000000000000000000"),
        (12.0, "12"),
        (12.5, "12.5"),
        (1234.5, "1234.5"),
        (0.1, "0.1"),
        (0.1 + 0.2, "0.3"),  # 0.30000000000000004 is a float artefact, never written
        (1e6, "1000000"),
        (1e-7, "0.0000001"),
        (1e22, "10000000000000000000000"),
        (-2.5, "-2.5"),
        (-0.0, "0"),
        (Decimal("1234.50"), "1234.50"),  # token text keeps its trailing zero
        (Decimal("12.0"), "12.0"),
        (Decimal("1E+6"), "1000000"),
        (Decimal("1e-7"), "0.0000001"),
        (Decimal("-0"), "0"),
        ("1234.50", "1234.50"),  # strings are the model's text: untouched
        ("10", "10"),
        ("10.00", "10.00"),
        ("1,234.50", "1,234.50"),  # the normaliser strips commas, coerce does not touch strings
        ("+5", "+5"),
        (" 12 ", " 12 "),
        ("", ""),
        ("-0", "-0"),
        ("N/A", "N/A"),
        ("nan", "nan"),
        ("1e6", "1000000"),  # only a JSON number literal WITH an exponent is rewritten
        ("1E+6", "1000000"),
        ("1.5e3", "1500"),
        ("-2E-3", "-0.002"),
        ("1e-7", "0.0000001"),
        ("0e5", "0"),
    ],
)
def test_numeric_value_rules(value: Any, expected: str | None) -> None:
    assert coerce_value("quantity", value) == expected
    assert coerce_value("total_amount", value) == expected


@pytest.mark.parametrize(
    "bad",
    [True, False, float("nan"), float("inf"), float("-inf"), Decimal("NaN"), Decimal("Infinity"),
     [1], {"a": 1}, b"1", "1e999999999", "1e-999999999"],
)  # fmt: skip
def test_fail_closed_on_values_that_must_never_be_written(bad: Any) -> None:
    with pytest.raises(CoerceError) as exc:
        coerce_value("pieces", bad)
    assert "999" not in str(exc.value) or "exponent" in str(exc.value)  # no value is quoted


def test_error_messages_never_quote_the_value() -> None:
    doc = {"doc_type": "invoice", "header": {"total_amount": True}, "line_items": []}
    with pytest.raises(CoerceError) as exc:
        coerce_predictions({"d1": doc})
    assert "d1" in str(exc.value) and "total_amount" in str(exc.value)
    doc = {"doc_type": "invoice", "header": {"supplier_name": 1234567}, "line_items": []}
    with pytest.raises(CoerceError) as exc:
        coerce_doc(doc)
    assert "1234567" not in str(exc.value)


def test_four_numeric_fields_are_the_four_grammar_fields() -> None:
    assert {"total_amount", "pieces", "gross_weight_kg", "quantity"} == NUMERIC_FIELDS


@pytest.mark.parametrize(
    "field",
    ["supplier_part_number", "customer_part_number", "purchase_order", "invoice_number", "mawb",
     "supplier_name", "currency", "invoice_date", "awb_number", "hawb"],
)  # fmt: skip
def test_non_numeric_fields_are_untouched_byte_for_byte_and_numbers_are_an_error(
    field: str,
) -> None:
    rng = random.Random(42)
    alphabet = "0123456789-.,+eE \t\u00a0\u00e9\u4e2dAZaz#/"
    for _ in range(300):
        s = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 14)))
        out = coerce_value(field, s)
        assert out == s and isinstance(out, str)
        assert out.encode("utf-8") == s.encode("utf-8")
    for digits in ("00123", "1E+6", "007", "1e5", "-0", "0.10"):
        assert coerce_value(field, digits) == digits  # digit-only ids keep their zeros
    assert coerce_value(field, None) is None
    for number in (123, 1.5, Decimal("1")):  # leading zeros would already be gone: an error
        with pytest.raises(CoerceError):
            coerce_value(field, number)


def test_coerce_is_idempotent_over_a_fuzz_of_values() -> None:
    rng = random.Random(42)
    values: list[Any] = [None, 0, 7, 12.0, 0.1 + 0.2, 1e6, 1e-7, 2**70, Decimal("1.50E+3")]
    values += [f"{rng.uniform(-1e6, 1e6):.{rng.randint(0, 4)}f}" for _ in range(200)]
    values += [f"{rng.randint(1, 9)}e{rng.randint(-9, 9)}" for _ in range(100)]
    values += [rng.uniform(-1e9, 1e9) for _ in range(200)]
    for v in values:
        once = coerce_value("quantity", v)
        assert coerce_value("quantity", once) == once
        assert once is None or (isinstance(once, str) and "e" not in once.lower() or v == once)
    doc = {
        "doc_type": "invoice",
        "header": {"invoice_number": "007", "total_amount": 1e6},
        "line_items": [{"supplier_part_number": "00A", "quantity": 12.0}],
    }
    once = coerce_doc(doc)
    assert coerce_doc(once) == once
    assert coerce_predictions(coerce_predictions({"x": doc})) == coerce_predictions({"x": doc})


def test_float_rule_never_prints_an_exponent_or_an_artefact() -> None:
    rng = random.Random(42)
    for _ in range(2000):
        x = round(rng.uniform(-1e7, 1e7), rng.randint(0, 6)) * rng.choice([1, 1e-9, 1e9])
        s = coerce_number(x)
        assert re.fullmatch(r"-?\d+(\.\d+)?", s), s
        assert not s.endswith(".0") and ("." not in s or not s.endswith("0"))
    assert coerce_number(0.1 + 0.2) == "0.3"
    assert "30000000000000004" not in coerce_number(0.1 + 0.2)


def test_coerce_doc_keeps_key_order_copies_and_leaves_unknown_string_keys() -> None:
    doc = {
        "doc_type": "waybill",
        "header": {"carrier": "ACME", "pieces": 12.0, "gross_weight_kg": 1.5, "extra": "x"},
        "line_items": [],
    }
    before = copy.deepcopy(doc)
    out = coerce_doc(doc)
    assert doc == before  # input untouched
    assert list(out["header"]) == list(doc["header"])
    assert out["header"] == {"carrier": "ACME", "pieces": "12", "gross_weight_kg": "1.5",
                             "extra": "x"}  # fmt: skip
    with pytest.raises(CoerceError):
        coerce_doc({"doc_type": "invoice", "header": {}, "line_items": [1]})
    with pytest.raises(CoerceError):
        coerce_doc({"doc_type": "invoice", "header": {}, "line_items": None})


# ---------------------------------------------------------------------- page level (token text)


def _raw(
    header: dict[str, Any], rows: list[dict[str, Any]] | None = None, dt: str = "invoice"
) -> str:
    """A page text with raw number literals (a value written as ``N:<literal>`` is unquoted)."""
    blob = json.dumps({"doc_type": dt, "header": header, "line_items": rows or [],
                       "page_kind": "single"})  # fmt: skip
    return re.sub(r'"N:([^"]*)"', r"\1", blob)


def test_page_for_merge_keeps_the_token_text_the_model_printed() -> None:
    header = {"invoice_number": "INV-1", "total_amount": "N:1234.50"}
    rows = [{"supplier_part_number": "00123", "customer_part_number": None,
             "purchase_order": "PO9", "quantity": "N:1E+1"},
            {"supplier_part_number": "B", "customer_part_number": None, "purchase_order": None,
             "quantity": "N:10"}]  # fmt: skip
    raw = _raw(header, rows)
    parsed = parse_output(raw)
    assert parsed["header"]["total_amount"] == 1234.5  # what json.loads gives: the zero is gone
    page = page_for_merge(raw, parsed)
    assert page["header"]["total_amount"] == "1234.50"
    assert [r["quantity"] for r in page["line_items"]] == ["1E+1", "10"]
    assert page["line_items"][0]["supplier_part_number"] == "00123"
    doc, _ = normalize_doc(merge_pages([page]).doc)
    assert coerce_doc(doc)["header"]["total_amount"] == "1234.50"
    assert [r["quantity"] for r in coerce_doc(doc)["line_items"]] == ["10", "10"]


def test_page_for_merge_falls_back_to_parsed_when_the_text_disagrees() -> None:
    raw = _raw({"total_amount": "N:5"})
    handmade = {"doc_type": "invoice", "header": {"total_amount": 6}, "line_items": []}
    assert page_for_merge(raw, handmade) is handmade
    assert page_for_merge(raw, None) is None
    assert page_for_merge("not json", handmade) is handmade


def test_page_for_merge_refuses_bool_nan_and_inf_values() -> None:
    for lit in ("true", "NaN", "Infinity", "-Infinity"):
        raw = f'{{"doc_type": "invoice", "header": {{"total_amount": {lit}}}, "line_items": []}}'
        with pytest.raises(CoerceError):
            page_for_merge(raw, parse_output(raw))
    raw = '{"doc_type": "invoice", "header": {}, "line_items": [{"quantity": false}]}'
    with pytest.raises(CoerceError):
        page_for_merge(raw, parse_output(raw))
    assert parse_output('{"a": NaN}', number_text=True) is None  # token mode: not valid JSON


def test_chain_merge_normalize_coerce_for_each_edge_value() -> None:
    """What reaches predictions.json for model text such as 1,234.50 / 1e6 / -0 / +5."""
    cases = {  # model value (JSON text) -> final string
        '"1,234.50"': "1234.50",  # string: the normaliser strips the comma, coerce keeps zeros
        '"1234.50"': "1234.50",
        "1234.50": "1234.50",
        "1234.5": "1234.5",
        "12.0": "12.0",  # a printed token is kept as printed (the scorer is numeric)
        "12": "12",
        "1e6": "1000000",
        "1E+6": "1000000",
        "1e-7": "0.0000001",
        "-5": "-5",
        "-0": "-0",  # token kept; scorer reads it as 0
        '"+5"': "5",  # the normaliser drops a leading plus
        '" 7 "': "7",  # the normaliser strips
        '""': None,  # the normaliser maps empty to null
        "null": None,
        "123456789012345678901234567890": "123456789012345678901234567890",
        "0.30000000000000004": "0.30000000000000004",  # the model printed it: kept (no float)
    }
    for lit, want in cases.items():
        raw = (
            '{"doc_type": "invoice", "header": {"total_amount": LIT}, "line_items": '
            '[{"quantity": LIT}], "page_kind": "single"}'
        ).replace("LIT", lit)
        page = page_for_merge(raw, parse_output(raw))
        doc, _ = normalize_doc(merge_pages([page]).doc)
        out = coerce_doc(doc)
        assert out["header"]["total_amount"] == want, lit
        rows = out["line_items"]  # an all-null row is dropped by the merge
        assert (rows[0]["quantity"] if rows else None) == want, lit


# ---------------------------------------------------------------------- the official scorer


def _scorer() -> Any:
    return ev.load_scorer(SCORER)


@needs_scorer
def test_scorer_reads_numbers_numerically_so_formatting_does_not_change_a_score() -> None:
    sc = _scorer()
    same = [("quantity", "10", "10.00"), ("quantity", 10, "10"), ("total_amount", "1234.5",
            "1234.50"), ("total_amount", "1,234.50", "1234.50"), ("pieces", "1000000", "1e6"),
            ("gross_weight_kg", "0.0000001", "0"), ("quantity", "12", "12.004")]  # fmt: skip
    for field, a, b in same:
        assert sc.same(field, a, b) and sc.same(field, b, a), (field, a, b)
    for field, a, b in [("quantity", "10", "11"), ("total_amount", "1234.50", "1234.51")]:
        assert not sc.same(field, a, b)
    # every coerced form of a value scores exactly like the number it came from
    for v in (12.0, 1234.5, 1e6, 0.1 + 0.2, 7, "1E+6", "1234.50"):
        for gold in ("12", "1234.50", "1000000", "0.3", "7"):
            assert sc.same("quantity", v, gold) == sc.same("quantity", coerce_number(v), gold)


def _saved_runs() -> list[Path]:
    return sorted(SAVED.glob("**/predictions.json"))


def _gold_for(preds: dict[str, Any]) -> dict[str, dict[str, Any]]:
    gold: dict[str, dict[str, Any]] = {}
    for split in ("train", "dev"):
        d = ROOT / "data" / split / "labels"
        if d.is_dir():
            gold.update(ev.load_gold(d))
    return {k: v for k, v in gold.items() if k in preds}


def _kinds(preds: Any) -> Counter[str]:
    import jsonschema

    v = jsonschema.Draft202012Validator(json.loads(SCHEMA.read_text(encoding="utf-8")))
    return Counter(f"{e.validator}@{list(e.absolute_path)[1:]}" for e in v.iter_errors(preds))


@needs_saved
@needs_schema
def test_saved_runs_validate_after_coercion_and_never_get_worse() -> None:
    runs = _saved_runs()
    assert len(runs) >= 9, runs  # 9 spike / dev runs + the smoke runs
    for path in runs:
        preds = json.loads(path.read_text(encoding="utf-8"))
        before, after = _kinds(preds), _kinds(coerce_predictions(preds))
        assert sum(after.values()) <= sum(before.values()), path
        assert not [k for k in after if k.startswith("type@")], (path, after)
        # the only error left in the saved runs is a model-misread invoice_date that is not
        # ISO (pattern, hence the failed oneOf), which is content, not number typing
        bad_dates = {
            d for d, doc in preds.items()
            if doc.get("doc_type") == "invoice"
            and doc["header"].get("invoice_date") is not None
            and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(doc["header"]["invoice_date"]))
        }  # fmt: skip
        clean = {d: doc for d, doc in coerce_predictions(preds).items() if d not in bad_dates}
        assert not _kinds(clean), path


@needs_saved
@needs_scorer
def test_scorer_result_is_identical_before_and_after_coercion_on_every_saved_run() -> None:
    sc, scored = _scorer(), 0
    for path in _saved_runs():
        preds = json.loads(path.read_text(encoding="utf-8"))
        gold = _gold_for(preds)
        if not gold:
            continue
        scored += 1
        a = ev.per_doc_results(preds, gold)
        b = ev.per_doc_results(coerce_predictions(preds), gold)
        assert a == b, path
        assert sc.aggregate(list(a.values())) == sc.aggregate(list(b.values()))
    assert scored >= 9


# ---------------------------------------------------------------------- the real writers


class NumberBackend(MockBackend):
    """Replays gold, but writes every numeric field as a raw JSON number literal.

    ``sci`` writes quantities as ``<q>E+0`` (an exponent literal); trailing zeros ("101.50") and
    integers are printed as the gold string has them, so the text round trip is checkable.
    """

    def __init__(self, *a: Any, sci: bool = False, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.sci = sci

    def extract_page(self, *a: Any, **kw: Any) -> Any:
        raw, _parsed, meta = super().extract_page(*a, **kw)
        pat = r'"(total_amount|pieces|gross_weight_kg|quantity)": "(-?\d+(?:\.\d+)?)"'

        def unquote(m: re.Match[str]) -> str:
            tok = m.group(2) + ("E+0" if self.sci and m.group(1) == "quantity" else "")
            return f'"{m.group(1)}": {tok}'

        raw = re.sub(pat, unquote, raw)
        return raw, parse_output(raw, self.output_format), meta


def _leaf_types_ok(preds: dict[str, Any]) -> None:
    for doc in preds.values():
        for rec in [doc["header"], *doc["line_items"]]:
            for k, v in rec.items():
                assert v is None or isinstance(v, str), (k, type(v))


def _numbers_as_strings(w: World, **kw: Any) -> NumberBackend:
    return NumberBackend(w.gold, fixed_latency_s=0.5, **kw)


def test_the_number_backend_really_emits_numbers(world: World) -> None:
    be = _numbers_as_strings(world)
    spike.run_spike(world.cfg, DEV_IDS, "dev", "n0", be, runs_root=world.runs,
                    data_root=world.data)  # fmt: skip
    trace = spike._read_trace(world.runs / "n0" / "trace.jsonl")
    raws = " ".join(p["raw_text"] for t in trace for p in t["pages"])
    assert re.search(r'"quantity": \d', raws) and re.search(r'"total_amount": \d', raws)
    assert any(isinstance(p["parsed"]["header"].get("total_amount"), float)
               for t in trace for p in t["pages"] if p["parsed"])  # fmt: skip


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("sci", [False, True])
def test_spike_runner_writes_plain_number_strings_identical_to_the_string_path(
    world: World,
    batch: int,
    sci: bool,
) -> None:
    ref = mock(world)
    spike.run_spike(world.cfg, DEV_IDS, "dev", "ref", ref, runs_root=world.runs,
                    data_root=world.data, batch_size=batch)  # fmt: skip
    spike.run_spike(world.cfg, DEV_IDS, "dev", "num", _numbers_as_strings(world, sci=sci),
                    runs_root=world.runs, data_root=world.data, batch_size=batch)  # fmt: skip
    a = json.loads((world.runs / "ref" / "predictions.json").read_text())
    b = json.loads((world.runs / "num" / "predictions.json").read_text())
    _leaf_types_ok(b)
    # '101.50' (trailing zero) and '1E+1' (exponent) come out exactly as the string path's text
    assert a == b
    traced = spike._read_trace(world.runs / "num" / "trace.jsonl")
    assert {t["doc_id"]: t["prediction"] for t in traced} == b


@needs_schema
def test_spike_predictions_from_numbers_validate_against_the_real_schema(
    world: World,
) -> None:
    spike.run_spike(world.cfg, DEV_IDS, "dev", "num", _numbers_as_strings(world),
                    runs_root=world.runs, data_root=world.data)  # fmt: skip
    preds = json.loads((world.runs / "num" / "predictions.json").read_text())
    kinds = _kinds(preds)
    # the synthetic waybill gold has no mawb / hawb (a fixture gap); numbers are not the cause
    assert not [k for k in kinds if k.startswith("type@")], kinds


def test_resume_over_a_trace_written_with_raw_numbers_is_rewritten_as_strings(
    world: World,
) -> None:
    """An old trace (prediction with numbers) is healed by the writer, idempotently."""
    spike.run_spike(world.cfg, DEV_IDS, "dev", "r0", mock(world), runs_root=world.runs,
                    data_root=world.data)  # fmt: skip
    tp = world.runs / "r0" / "trace.jsonl"
    lines = [json.loads(ln) for ln in tp.read_text().splitlines()]
    for t in lines:
        h = t["prediction"]["header"]
        for k in ("total_amount", "pieces", "gross_weight_kg"):
            if h.get(k) is not None:
                h[k] = float(h[k])
    tp.write_text("".join(json.dumps(t) + "\n" for t in lines), encoding="utf-8", newline="\n")
    spike.run_spike(world.cfg, DEV_IDS, "dev", "r0", mock(world), resume=True,
                    runs_root=world.runs, data_root=world.data)  # fmt: skip
    _leaf_types_ok(json.loads((world.runs / "r0" / "predictions.json").read_text()))


class BoolBackend(MockBackend):
    def extract_page(self, *a: Any, **kw: Any) -> Any:
        raw, _parsed, meta = super().extract_page(*a, **kw)
        raw = re.sub(r'"quantity": "[^"]*"', '"quantity": true', raw, count=1)
        return raw, parse_output(raw, self.output_format), meta


def test_a_bool_in_a_field_stops_the_run_and_writes_nothing_for_that_document(
    world: World,
) -> None:
    with pytest.raises(CoerceError):
        spike.run_spike(world.cfg, DEV_IDS, "dev", "b0", BoolBackend(world.gold),
                        runs_root=world.runs, data_root=world.data)  # fmt: skip
    pred = world.runs / "b0" / "predictions.json"  # the doc that failed was never written
    assert not pred.is_file() or "true" not in pred.read_text()
    trace = world.runs / "b0" / "trace.jsonl"
    assert not trace.is_file() or "doc_type" not in trace.read_text()  # not even the first doc


@needs_schema
@pytest.mark.parametrize("batch", [1, 4])
def test_predict_submission_from_numbers_is_schema_valid(
    world: World,
    tmp_path: Path,
    batch: int,
) -> None:
    run_all(world, batch=batch, be=_numbers_as_strings(world))
    out = tmp_path / "sub"
    rep = assemble(world, out)
    assert rep["checks"]["number_coercion"]["ok"], rep["checks"]["number_coercion"]
    assert rep["checks"]["json_schema"]["ok"] or all(
        not k.startswith("type") for k in rep["checks"]["json_schema"].get("by_kind", {})
    )
    name = predict.OUT_FILES[0] if rep["ok"] else predict.REJECTED_NAME
    _leaf_types_ok(json.loads((out / name).read_text()))
    plain = World(world.data, tmp_path / "plain", world.gold, world.cfg)
    run_all(plain, batch=batch)
    other = tmp_path / "sub2"
    assemble(plain, other)
    assert (out / name).read_bytes() == (other / name).read_bytes()


@needs_schema
def test_assemble_refuses_a_prediction_that_cannot_be_coerced(
    world: World,
    tmp_path: Path,
) -> None:
    run_all(world)
    tp = world.runs / RUN / "trace.jsonl"
    lines = [json.loads(ln) for ln in tp.read_text().splitlines()]
    lines[0]["prediction"]["header"]["total_amount"] = True
    tp.write_text("".join(json.dumps(t) + "\n" for t in lines), encoding="utf-8", newline="\n")
    out = tmp_path / "sub"
    rep = assemble(world, out)
    assert not rep["ok"] and not rep["checks"]["number_coercion"]["ok"]
    assert (out / predict.REJECTED_NAME).is_file()
    assert not (out / predict.OUT_FILES[0]).exists()


@needs_schema
def test_merge_shards_from_numbers_writes_strings_and_assembles(
    world: World,
    tmp_path: Path,
) -> None:
    sharded = World(world.data, tmp_path / "sharded", world.gold, world.cfg)
    for i in (0, 1):
        run_all(sharded, shard=f"{i}/2", be=_numbers_as_strings(world))
    shardmerge.merge_shards(world.cfg, TEST_IDS, "test", RUN, 2, runs_root=sharded.runs,
                            data_root=world.data, expected_docs=len(TEST_IDS),
                            expected_pages=N_PAGES)  # fmt: skip
    merged = json.loads((sharded.runs / RUN / "predictions.json").read_text())
    assert sorted(merged) == sorted(TEST_IDS)
    _leaf_types_ok(merged)
    out = tmp_path / "sub"
    rep = assemble(sharded, out)
    assert rep["checks"]["number_coercion"]["ok"]
    name = predict.OUT_FILES[0] if rep["ok"] else predict.REJECTED_NAME
    _leaf_types_ok(json.loads((out / name).read_text()))


def test_merge_shards_rewrites_shard_traces_that_still_hold_numbers(
    world: World,
    tmp_path: Path,
) -> None:
    sharded = World(world.data, tmp_path / "sharded", world.gold, world.cfg)
    for i in (0, 1):
        run_all(sharded, shard=f"{i}/2")
    for i in (0, 1):  # simulate shards written before coerce existed
        tp = sharded.runs / f"{RUN}_shard{i}of2" / "trace.jsonl"
        if not tp.is_file():
            tp = next(p for p in sharded.runs.glob(f"{RUN}*/trace.jsonl") if f"{i}of2" in str(p))
        lines = [json.loads(ln) for ln in tp.read_text().splitlines()]
        for t in lines:
            for r in t["prediction"]["line_items"]:
                r["quantity"] = float(r["quantity"])
        tp.write_text("".join(json.dumps(t) + "\n" for t in lines), encoding="utf-8", newline="\n")
    shardmerge.merge_shards(world.cfg, TEST_IDS, "test", RUN, 2, runs_root=sharded.runs,
                            data_root=world.data, expected_docs=len(TEST_IDS),
                            expected_pages=N_PAGES)  # fmt: skip
    _leaf_types_ok(json.loads((sharded.runs / RUN / "predictions.json").read_text()))
