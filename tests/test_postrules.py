"""Tests for post-processing v1 (shipdoc.postrules), the frozen shapes and the replay check.

Synthetic documents only; the drift / leak checks of the real artefact skip without ``data/``.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from test_no_label_leak import collect_values, scan

from shipdoc import eval as ev
from shipdoc import ocr, paths, postrules, rules
from shipdoc.postrules import RuleConfig, apply_rules

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


replay_check = _load("replay_v1_check")
freeze = _load("freeze_slot_shapes")

SHAPES = rules.SlotShapes(cpn_only=frozenset({"AA-9999"}), po_only=frozenset({"9999999"}))
TEXT = "MAWB 176-12345678 and HAWB AB12345678 here"
SENTINELS = ("Acme Air Sentinel", "176-12345678", "AB12345678", "1234567")


def wb(**header: Any) -> dict[str, Any]:
    return {"doc_type": "waybill", "header": dict(header), "line_items": []}


def inv(*rows: tuple[Any, Any]) -> dict[str, Any]:
    items = [
        {"supplier_part_number": f"S{i}", rules.CPN: c, rules.PO: p}
        for i, (c, p) in enumerate(rows)
    ]
    return {"doc_type": "invoice", "header": {}, "line_items": items}


def run(doc: Any, **kw: Any) -> Any:
    kw.setdefault("supplier_name", "Acme Air Sentinel")
    kw.setdefault("ocr_text", TEXT)
    kw.setdefault("shapes", SHAPES)
    return apply_rules(doc, **kw)


# ---- apply_rules ---------------------------------------------------------------------------


def test_defaults_are_all_on_and_all_off_is_all_false() -> None:
    assert RuleConfig().as_dict() == {"r1": True, "r2": True, "r3": True}
    assert RuleConfig.all_off().as_dict() == {"r1": False, "r2": False, "r3": False}


def test_waybill_runs_r1_then_r2_in_order() -> None:
    doc = wb(carrier=None, mawb=None, hawb=None)
    out, changes, skipped = run(doc)
    assert [(c["rule"], c["field"]) for c in changes] == [
        ("R1", "carrier"), ("R2", "mawb"), ("R2", "hawb"),
    ]  # fmt: skip
    assert all(c["kind"] == "fill" and c["row"] is None for c in changes)
    assert out["header"]["carrier"] == "Acme Air Sentinel" and skipped == []


def test_invoice_runs_r3_and_records_move_and_swap_kinds() -> None:
    doc = inv(("1234567", None), ("1234567", "AB-1234"), ("keep", "1234567"))
    out, changes, skipped = run(doc)
    assert [(c["rule"], c["row"], c["kind"]) for c in changes] == [
        ("R3", 0, "move"), ("R3", 1, "swap"),
    ]  # fmt: skip
    assert out["line_items"][0][rules.PO] == "1234567" and skipped == []


def test_r2_is_skipped_without_ocr_but_r1_still_applies() -> None:
    out, changes, skipped = run(wb(carrier=None, mawb=None), ocr_text=None)
    assert skipped == [{"rule": "R2", "reason": "no_ocr"}]
    assert [c["rule"] for c in changes] == ["R1"] and out["header"]["mawb"] is None
    _, _, sk = run(wb(mawb=None), ocr_text=None, ocr_reason="ocr_incomplete")
    assert sk == [{"rule": "R2", "reason": "ocr_incomplete"}]


def test_r3_is_skipped_without_shapes_and_not_for_waybills() -> None:
    _, changes, skipped = run(inv(("1234567", None)), shapes=None)
    assert changes == [] and skipped == [{"rule": "R3", "reason": "no_shapes"}]
    assert run(wb(carrier="X", mawb="1", hawb="2"), shapes=None)[2] == []


def test_preconditions_fail_closed_never_fill() -> None:
    two = "176-12345678 and 176-87654321 AB12345678"
    out, changes, _ = run(wb(carrier="Model Cargo", mawb=None, hawb="ZZ00000000"), ocr_text=two)
    assert changes == []  # carrier set, mawb ambiguous (two candidates), hawb set
    assert out["header"]["carrier"] == "Model Cargo"
    assert (
        run(wb(carrier=None), supplier_name=None, ocr_text=None)[1] == []
    )  # nothing to take the carrier from
    amb = inv(("AB-1234", "AB-1234"), ("Q", None))  # shape unknown / ambiguous: untouched
    assert run(amb)[1] == []


def test_input_is_not_mutated_and_all_off_returns_an_equal_doc() -> None:
    docs = [wb(carrier=None, mawb=None), inv(("1234567", None))]
    for d in docs:
        before = copy.deepcopy(d)
        out, changes, _ = run(d)
        assert d == before and changes and out != before
        off, off_changes, off_skipped = run(d, cfg=RuleConfig.all_off())
        assert off == before and off_changes == []
        assert {s["reason"] for s in off_skipped} == {"disabled"}


def test_records_contain_no_values() -> None:
    docs = [wb(carrier=None, mawb=None, hawb=None), inv(("1234567", None), ("1234567", "AB-1234"))]
    for d in docs:
        _, changes, skipped = run(d)
        blob = json.dumps({"c": changes, "s": skipped})
        assert all(s not in blob for s in SENTINELS)
        assert {k for c in changes for k in c} == {"rule", "field", "row", "kind"}


@pytest.mark.parametrize(
    "doc",
    [None, {}, [], {"doc_type": "waybill"}, {"doc_type": "waybill", "header": None},
     {"doc_type": "invoice", "line_items": None}, {"doc_type": "invoice", "line_items": [None, 3]}],
)  # fmt: skip
def test_never_raises_on_junk_or_missing_inputs(doc: Any) -> None:
    run(doc)
    run(doc, supplier_name=None, ocr_text=None, shapes=None)


def test_rules_are_idempotent() -> None:
    out, _, _ = run(inv(("1234567", None), ("1234567", "AB-1234")))
    again, changes, _ = run(out)
    assert again == out and changes == []


# ---- OCR helper ----------------------------------------------------------------------------


def _write_page(root: Path, doc_id: str, page: int, text: str) -> None:
    split = doc_id.split("_", 1)[0]
    item = ocr.OcrItem(text, [[0, 0], [10, 0], [10, 5], [0, 5]], 0.99)
    page_ocr = ocr.PageOcr(f"{doc_id}_p{page}", 10, 5, "paddleocr", "line", 0, [item])
    path = ocr.cache_path(root, ocr.DEFAULT_ENGINE, split, f"{doc_id}_p{page}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(page_ocr.to_dict()), encoding="utf-8")


def test_ocr_text_for_doc_missing_partial_and_complete(tmp_path: Path) -> None:
    assert postrules.ocr_text_for_doc("test_0001", 2, tmp_path) == (None, "no_ocr")
    _write_page(tmp_path, "test_0001", 1, "page one")
    assert postrules.ocr_text_for_doc("test_0001", 2, tmp_path) == (None, "ocr_incomplete")
    _write_page(tmp_path, "test_0001", 2, "page two")
    assert postrules.ocr_text_for_doc("test_0001", 2, tmp_path) == ("page one\npage two", None)


def trace(doc_id: str, doc: dict[str, Any], supplier: Any = None, n_pages: int = 1) -> dict:
    page = {"parsed": {"header": {"supplier_name": supplier}}, "raw_text": "{}"}
    return {"doc_id": doc_id, "prediction": doc, "pages": [page] * n_pages}


def test_supplier_from_trace_reads_the_page1_parsed_header() -> None:
    assert postrules.supplier_from_trace(trace("d", {}, "Acme Air Sentinel")) == "Acme Air Sentinel"
    assert postrules.supplier_from_trace({"pages": []}) is None
    salvage = {"pages": [{"parsed": None, "raw_text": '{"header": {"supplier_name": "S"}, "li'}]}
    assert postrules.supplier_from_trace(salvage) == "S"


def test_postprocess_traces_counts_and_ocr_skips(tmp_path: Path) -> None:
    _write_page(tmp_path, "test_0001", 1, TEXT)
    traces = [
        trace("test_0002", wb(carrier=None, mawb=None), "Acme Air Sentinel"),  # no OCR cached
        trace("test_0001", wb(carrier=None, mawb=None), "Acme Air Sentinel"),
        trace("test_0003", inv(("1234567", None))),
    ]
    preds, records, summ = postrules.postprocess_traces(traces, RuleConfig(), SHAPES, tmp_path)
    assert [r["doc_id"] for r in records] == ["test_0001", "test_0002", "test_0003"]
    assert preds["test_0001"]["header"]["mawb"] == "176-12345678"
    assert preds["test_0002"]["header"]["mawb"] is None
    assert summ["skipped"] == {"R2/no_ocr": 1}
    assert summ["eligible_docs"] == {"R1": 2, "R2": 1, "R3": 1}
    assert summ["touched_docs"] == {"R1": 2, "R2": 1, "R3": 1}
    assert summ["changes"] == {"R1": 2, "R2": 2, "R3": 1}  # R2 fills both mawb and hawb
    assert traces[0]["prediction"]["header"]["carrier"] is None  # inputs untouched


def test_postprocess_traces_accepts_a_per_doc_shapes_callable(tmp_path: Path) -> None:
    other = rules.SlotShapes(cpn_only=frozenset(), po_only=frozenset())
    traces = [trace("a_1", inv(("1234567", None))), trace("a_2", inv(("1234567", None)))]
    preds, _, _ = postrules.postprocess_traces(
        traces, RuleConfig(), lambda d: SHAPES if d == "a_1" else other, tmp_path
    )
    assert preds["a_1"]["line_items"][0][rules.PO] == "1234567"
    assert preds["a_2"]["line_items"][0][rules.PO] is None


# ---- frozen shapes -------------------------------------------------------------------------


def _labels() -> list[dict[str, Any]]:
    return [
        {"doc_type": "invoice", "line_items": [
            {rules.CPN: "AB-1234", rules.PO: None}, {rules.CPN: None, rules.PO: "7654321"}]},
        {"doc_type": "invoice", "line_items": [{rules.CPN: "CD-5678", rules.PO: "1234567"}]},
        {"doc_type": "waybill", "line_items": []},
    ]  # fmt: skip


def test_frozen_shapes_are_deterministic_and_order_free() -> None:
    a = postrules.dump_shapes(postrules.shapes_payload(_labels(), "f" * 64))
    b = postrules.dump_shapes(postrules.shapes_payload(list(reversed(_labels())), "f" * 64))
    assert a == b and a.endswith("\n") and "\r" not in a
    data = json.loads(a)
    assert data["cpn_only"] == ["AA-9999"] and data["po_only"] == ["9999999"]
    assert (data["n_docs"], data["n_invoice_docs"]) == (3, 2)
    assert (data["n_cpn_cells"], data["n_po_cells"]) == (2, 2) and data["folds_sha256"] == "f" * 64
    assert "AB-1234" not in a and "7654321" not in a  # shapes only, no value


def test_load_slot_shapes_roundtrip_and_refusal(tmp_path: Path) -> None:
    p = tmp_path / "s.json"
    p.write_text(postrules.dump_shapes(postrules.shapes_payload(_labels(), "0" * 64)))
    shapes, sha = postrules.load_slot_shapes(p)
    assert shapes == rules.learn_slot_shapes(_labels()) and sha == postrules.sha256_file(p)
    p.write_text(json.dumps({"schema": 99}))
    with pytest.raises(ValueError, match="slot shapes"):
        postrules.load_slot_shapes(p)
    with pytest.raises(OSError):
        postrules.load_slot_shapes(tmp_path / "missing.json")


def test_folds_hash_is_line_ending_invariant(tmp_path: Path) -> None:
    # the committed file is LF but a Windows working copy may be CRLF; --check must agree on both
    lf, crlf = tmp_path / "lf.json", tmp_path / "crlf.json"
    lf.write_bytes(b'{\n "doc_fold": {}\n}\n')
    crlf.write_bytes(b'{\r\n "doc_fold": {}\r\n}\r\n')
    assert freeze.folds_sha256(lf) == freeze.folds_sha256(crlf)


needs_data = pytest.mark.skipif(not DATA.is_dir(), reason="data/ absent")


@needs_data
def test_committed_shapes_file_equals_a_fresh_freeze_and_leaks_no_gold_value() -> None:
    path = postrules.shapes_path()
    assert path.read_text(encoding="utf-8") == freeze.freeze()  # drift check
    data = json.loads(path.read_text(encoding="utf-8"))
    shapes = set(data["cpn_only"]) | set(data["po_only"])
    for min_len in (6, 8):  # the repo's two leak thresholds (history scan / tracked-file test)
        assert not scan(collect_values(min_len), files=[path])
    assert not shapes & set(collect_values(1)), "a shape equals a gold value"
    assert all(len(s) < 40 for s in shapes)  # a shape is a short format, never a long literal


# ---- replay check helpers (synthetic mini run) ---------------------------------------------


def test_fold_shape_fn_learns_from_other_folds_only_and_fails_closed() -> None:
    labels = [
        {"doc_id": "a", **_labels()[0]},
        {"doc_id": "b", "doc_type": "invoice", "line_items": [{rules.CPN: "XY-9", rules.PO: None}]},
    ]
    fn = replay_check.fold_shape_fn(labels, {"a": 0, "b": 1}, {"a": "g1", "b": "g2"})
    assert fn("a").cpn_only == frozenset({"AA-9"}) and fn("b").cpn_only == frozenset({"AA-9999"})
    with pytest.raises(SystemExit, match="overlaps"):
        replay_check.fold_shape_fn(labels, {"a": 0, "b": 1}, {"a": "g1", "b": "g1"})


def test_production_with_all_rules_off_reproduces_the_input_predictions() -> None:
    traces = [trace("test_1", wb(carrier=None)), trace("test_2", inv(("1234567", None)))]
    preds = {t["doc_id"]: t["prediction"] for t in traces}
    out, summ = replay_check.production(traces, RuleConfig.all_off(), SHAPES, None)
    assert out == preds and summ["changes"] == {"R1": 0, "R2": 0, "R3": 0}


def test_slice_ids_and_render_on_a_synthetic_result() -> None:
    gold = {"train_1": {"doc_type": "invoice"}, "dev_1": {"doc_type": "waybill"}}
    meta = [{"doc_id": "train_1", "scanned": True}, {"doc_id": "dev_1", "scanned": False}]
    s = replay_check.slice_ids(gold, meta)
    assert s["scanned"] == ["train_1"] and s["digital"] == ["dev_1"] and s["dev"] == ["dev_1"]
    cell = {"fixed": 1, "broken": 0, "neutral": 0}
    d = {"base": 0.5, "new": 0.6, "delta": 0.1, "lo": 0.05, "hi": 0.15, "ff_base": 0, "ff_new": 0}
    var = {
        "summary": {"touched_docs": {}, "eligible_docs": {}, "skipped": {}},
        "counts": {r: {"train": cell} for r in ("R1", "R2", "R3")},
        "singles": {"R1": 0.01, "R2": 0.02, "R3": 0.03, "R1+R2": 0.04},
        "slices": {"all": (2, d)},
    }
    res = {"run": "r", "n_docs": 2, "base_overall": 0.5, "frozen_sha": "x",
           "variants": {"honest": var, "frozen": var}}  # fmt: skip
    text = replay_check.render(res, 10)
    assert "(a) HONEST" in text and "(b) SHIPPING" in text and "UNVERIFIED" in text


def test_gate_fixed_cli_parsing_and_render_cites_the_given_counts() -> None:
    items = ["R1=train:4,dev:3", "R2=train:27,dev:14", "R3=all:278"]
    gate = replay_check.parse_gate_fixed(items)
    assert gate == {
        "R1": {"train": 4, "dev": 3},
        "R2": {"train": 27, "dev": 14},
        "R3": {"all": 278},
    }
    for bad in (["R1=train:4"], ["R9=all:1", "R2=all:1", "R3=all:1"], ["R1=", "R2=a:1", "R3=a:1"]):
        with pytest.raises(SystemExit):
            replay_check.parse_gate_fixed(bad)
    cell = {"fixed": 1, "broken": 0, "neutral": 0}
    d = {"base": 0.5, "new": 0.6, "delta": 0.1, "lo": 0.05, "hi": 0.15, "ff_base": 0, "ff_new": 0}
    var = {
        "summary": {"touched_docs": {}, "eligible_docs": {}, "skipped": {}},
        "counts": {r: {"train": cell} for r in ("R1", "R2", "R3")},
        "singles": {"R1": 0.01, "R2": 0.02, "R3": 0.03, "R1+R2": 0.04},
        "slices": {"all": (2, d)},
    }
    res = {"run": "r", "n_docs": 2, "base_overall": 0.5, "frozen_sha": "x",
           "variants": {"honest": var, "frozen": var}}  # fmt: skip
    text = replay_check.render(res, 10, gate, "CMD --x")
    assert "train 27 + dev 14" in text and "all 278" in text and "`CMD --x`" in text
    assert "train 20 + dev 12" not in text  # the 1260 gate counts are not cited for another run


@pytest.mark.skipif(not paths.scorer_path().is_file(), reason="official scorer absent")
def test_count_changes_classifies_fixed_and_broken_cells() -> None:
    sc = ev.load_scorer()
    gold = {
        "train_1": {"doc_type": "waybill", "header": {"carrier": "Air X", "mawb": "176-12345678"}}
    }
    base = {"train_1": wb(carrier=None, mawb="176-12345678")}
    new = {"train_1": wb(carrier="Air X", mawb="999-00000000")}
    out = replay_check.count_changes(sc, base, new, gold)
    assert out["R1"]["train"]["fixed"] == 1 and out["R2"]["train"]["broken"] == 1


# ---- the run_spike writer (opt-in) ---------------------------------------------------------


def test_run_spike_rule_cfg_is_opt_in_and_adds_a_value_free_rules_record(tmp_path: Path) -> None:
    from _synth import SPECS, make_corpus

    from shipdoc import spike
    from shipdoc.extract import MockBackend

    cfg = spike.load_config(ROOT / "configs" / "spike_qwen35_4b_img_only.yaml")
    gold = make_corpus(tmp_path / "data")
    ids = [s[0] for s in SPECS]
    traces = {}
    for tag, rule_cfg in (("plain", None), ("rules", RuleConfig(r2=False))):
        spike.run_spike(
            cfg, ids, "dev", tag, MockBackend(gold), runs_root=tmp_path / "runs",
            data_root=tmp_path / "data", rule_cfg=rule_cfg,
        )  # fmt: skip
        traces[tag] = spike._read_trace(tmp_path / "runs" / tag / "trace.jsonl")
    assert all("rules" not in t for t in traces["plain"])  # default: the run is unchanged
    assert all({"changes", "skipped"} == set(t["rules"]) for t in traces["rules"])
    blob = json.dumps([t["rules"] for t in traces["rules"]])
    assert "Carrier" not in blob and "PO" not in blob  # no value in the record
    assert {
        t["rules"]["skipped"][0]["reason"] for t in traces["rules"] if t["rules"]["skipped"]
    } == {"disabled"}
