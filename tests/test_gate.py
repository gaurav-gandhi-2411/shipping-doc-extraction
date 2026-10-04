"""shipdoc.gate: the per-field selection rules, the no-synthesis invariant, the output schema, the
fit / apply separation check and the pre-registered replacement rule (spec section 11 item 7).

Everything runs on tiny hand-built documents; no model, no real document, no network.
"""

from __future__ import annotations

import copy
import json
import math
import random
from typing import Any

import numpy as np
import pytest
from test_g4_fold import world

from shipdoc import confidence as cf
from shipdoc import gate, oof, paths
from shipdoc.g4pooled import FT_LABEL, ZS_LABEL
from shipdoc.gate import FT, ZS, GateError, decide_g, gate_document, select_slot

ROW_FIELDS = ("supplier_part_number", "customer_part_number", "purchase_order", "quantity")


def doc(
    header: dict[str, Any], rows: list[dict[str, Any]], dtype: str = "invoice"
) -> dict[str, Any]:
    return {"doc_type": dtype, "header": header, "line_items": rows}


def row(spn: Any, cpn: Any = None, po: Any = None, qty: Any = "5") -> dict[str, Any]:
    return {
        "supplier_part_number": spn, "customer_part_number": cpn, "purchase_order": po,
        "quantity": qty,
    }  # fmt: skip


def probs(
    header: dict[str, float] | None = None, rows: dict[tuple[int, str], float] | None = None
) -> dict[gate.FieldId, float]:
    out: dict[gate.FieldId, float] = {("header", k, -1): v for k, v in (header or {}).items()}
    out |= {("row", f, i): v for (i, f), v in (rows or {}).items()}
    return out


def all_rows(n: int, p: float) -> dict[tuple[int, str], float]:
    return {(i, f): p for i in range(n) for f in ROW_FIELDS}


H = {"invoice_number": "A1", "buyer_name": "Buyer"}


# ---- one slot -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("v_ft", "v_zs", "p_ft", "p_zs", "source", "value"),
    [
        ("F", "Z", 0.9, 0.8, FT, "F"),  # strictly higher P wins
        ("F", "Z", 0.8, 0.9, ZS, "Z"),
        ("F", "Z", 0.7, 0.7, ZS, "Z"),  # TIE goes to ZS + rules (the simpler system)
        ("F", "Z", 0.0, 0.0, ZS, "Z"),
        ("X", "X", 0.9, 0.1, "same", "X"),  # equal values: nothing to choose
        (None, None, None, None, "same", None),
        ("", None, None, None, "same", None),  # blank is blank, whichever spelling
        ("F", None, 0.5, None, FT, "F"),  # one-arm field at the boundary: >= 0.5 keeps
        ("F", None, 0.4999, None, ZS, None),  # below: the other arm's blank is written
        (None, "Z", None, 0.5, ZS, "Z"),
        (None, "Z", None, 0.4999, FT, None),
        ("F", "  ", 0.6, None, FT, "F"),  # a whitespace string is blank
    ],
)
def test_select_slot_rules(
    v_ft: Any, v_zs: Any, p_ft: Any, p_zs: Any, source: str, value: Any
) -> None:
    src, val = select_slot(v_ft, v_zs, p_ft, p_zs)
    assert src == source
    assert val == value or (cf._blank(val) and cf._blank(value))
    assert val in (v_ft, v_zs)  # never a third value


def test_select_slot_refuses_an_emitted_value_without_p() -> None:
    with pytest.raises(GateError, match="no P"):
        select_slot("F", "Z", None, 0.5)
    with pytest.raises(GateError, match="no P"):
        select_slot("F", None, None, None)


# ---- row probability ----------------------------------------------------------------------------


def test_row_probability_is_the_minimum_over_emitted_fields() -> None:
    r = row("S1", None, "PO", "5")
    p = probs(
        rows={(0, "supplier_part_number"): 0.9, (0, "purchase_order"): 0.6, (0, "quantity"): 0.8}
    )
    assert gate.row_probability(r, 0, p) == 0.6  # the null customer_part_number has no P
    assert gate.row_probability(row(None, None, None, None), 0, {}) is None
    with pytest.raises(GateError, match="no P"):
        gate.row_probability(r, 0, probs(rows={(0, "quantity"): 0.9}))


# ---- one document -------------------------------------------------------------------------------


def test_header_and_row_fields_are_taken_from_the_arm_with_the_higher_p() -> None:
    ft = doc({"invoice_number": "F1", "buyer_name": "Buyer"}, [row("S1", qty="6")])
    zs = doc({"invoice_number": "Z1", "buyer_name": "Buyer"}, [row("S1", qty="7")])
    p_ft = probs({"invoice_number": 0.9, "buyer_name": 0.9}, {(0, "supplier_part_number"): 0.9,
                                                                (0, "quantity"): 0.4})  # fmt: skip
    p_zs = probs({"invoice_number": 0.6, "buyer_name": 0.1}, {(0, "supplier_part_number"): 0.1,
                                                                (0, "quantity"): 0.8})  # fmt: skip
    g = gate_document(ft, zs, p_ft, p_zs)
    assert g.doc["header"] == {"invoice_number": "F1", "buyer_name": "Buyer"}  # FT wins header
    assert g.doc["line_items"] == [row("S1", qty="7")]  # ZS wins the quantity slot
    assert g.row_sources == [(0, 0)]
    assert g.stats["slot.header.ft"] == 1 and g.stats["slot.header.same"] == 1
    assert g.stats["slot.row.quantity.zs"] == 1 and g.stats["rows.paired"] == 1


def test_a_tie_in_a_real_slot_goes_to_zs_and_is_counted() -> None:
    ft, zs = doc({"invoice_number": "F"}, []), doc({"invoice_number": "Z"}, [])
    g = gate_document(ft, zs, probs({"invoice_number": 0.8}), probs({"invoice_number": 0.8}))
    assert g.doc["header"]["invoice_number"] == "Z" and g.stats["slot.ties_to_zs"] == 1


def test_one_arm_rows_are_kept_at_exactly_half_and_dropped_below() -> None:
    # arms hold 1 vs 3 rows; the first FT row pairs with the first ZS row (stage 1), the others are
    # one-arm rows
    ft = doc(H, [row("S1"), row("S2"), row("S3")])
    zs = doc(H, [row("S1")])
    p_zs = probs(rows=all_rows(1, 0.9))
    p_ft = probs(rows={
        (0, "supplier_part_number"): 0.9, (0, "quantity"): 0.9,
        (1, "supplier_part_number"): 0.5, (1, "quantity"): 0.9,  # row P = min = 0.5 -> kept
        (2, "supplier_part_number"): 0.4999, (2, "quantity"): 0.9,  # row P < 0.5 -> dropped
    })  # fmt: skip
    g = gate_document(ft, zs, p_ft, p_zs)
    assert [r["supplier_part_number"] for r in g.doc["line_items"]] == ["S1", "S2"]
    assert g.row_sources == [(0, 0), (1, None)]
    assert g.stats["rows.one_arm.ft.kept"] == 1 and g.stats["rows.one_arm.ft.dropped"] == 1
    # the same rule for ZS-only rows
    g2 = gate_document(zs, ft, p_zs, p_ft)  # roles swapped: the 3-row doc is now the "ZS" arm
    assert [r["supplier_part_number"] for r in g2.doc["line_items"]] == ["S1", "S2"]
    assert g2.stats["rows.one_arm.zs.kept"] == 1 and g2.stats["rows.one_arm.zs.dropped"] == 1


def test_a_one_arm_row_without_any_emitted_field_is_dropped() -> None:
    ft = doc(H, [row(None, None, None, None)])
    zs = doc(H, [])
    g = gate_document(ft, zs, probs({"invoice_number": 1.0, "buyer_name": 1.0}), probs())
    assert g.doc["line_items"] == [] and g.stats["rows.one_arm.ft.dropped"] == 1


def test_output_row_order_is_zs_order_then_ft_only_rows() -> None:
    ft = doc(H, [row("F-only"), row("S2"), row("S1")])
    zs = doc(H, [row("S1"), row("S2")])
    pf, pz = probs(rows=all_rows(3, 0.9)), probs(rows=all_rows(2, 0.9))
    g = gate_document(ft, zs, pf, pz)
    spns = [r["supplier_part_number"] for r in g.doc["line_items"]]
    assert spns == ["S1", "S2", "F-only"]


def test_doc_type_disagreement_writes_the_zs_document_unchanged() -> None:
    ft = doc(H, [row("S1")])
    zs = doc({"carrier": "Acme"}, [], dtype="waybill")
    g = gate_document(ft, zs, probs(), probs())
    assert g.doc == zs and g.doc is not zs and g.doc_type_fallback
    assert g.stats["doc_type_fallback"] == 1


def test_missing_p_for_an_emitted_field_fails_closed() -> None:
    with pytest.raises(GateError, match="no P"):
        gate_document(doc({"invoice_number": "F"}, []), doc({"invoice_number": "Z"}, []), {}, {})


# ---- no value is ever synthesized (property-style) -----------------------------------------------

POOL = ("A", "B", "10", "10.0", "x y", None, "", "  ")


def rand_doc(rng: random.Random) -> dict[str, Any]:
    n = rng.randint(0, 4)
    return doc(
        {"invoice_number": rng.choice(POOL), "buyer_name": rng.choice(POOL)},
        [{f: rng.choice(POOL) for f in ROW_FIELDS} for _ in range(n)],
    )


def rand_probs(rng: random.Random, d: dict[str, Any]) -> dict[gate.FieldId, float]:
    levels = (0.0, 0.3, 0.5, 0.7, 1.0)  # includes the 0.5 boundary and exact ties
    out: dict[gate.FieldId, float] = {}
    for k, v in d["header"].items():
        if not cf._blank(v):
            out[("header", k, -1)] = rng.choice(levels)
    for i, r in enumerate(d["line_items"]):
        for k, v in r.items():
            if not cf._blank(v):
                out[("row", k, i)] = rng.choice(levels)
    return out


def test_no_value_is_ever_synthesized_over_random_inputs() -> None:
    rng = random.Random(42)
    seen_rows = seen_choice = 0
    for _ in range(400):
        ft, zs = rand_doc(rng), rand_doc(rng)
        g = gate_document(ft, zs, rand_probs(rng, ft), rand_probs(rng, zs))
        # independent re-check (not via the module's own assertion): every output cell equals the
        # same-slot cell of the arm row(s) it derives from
        for k, v in g.doc["header"].items():
            assert v in (ft["header"][k], zs["header"][k])
        for out_row, (fi, zi) in zip(g.doc["line_items"], g.row_sources, strict=True):
            assert fi is not None or zi is not None
            for k, v in out_row.items():
                cands = []
                if fi is not None:
                    cands.append(ft["line_items"][fi][k])
                if zi is not None:
                    cands.append(zs["line_items"][zi][k])
                assert v in cands
                seen_choice += len(set(map(str, cands))) > 1
        # no row appears that neither arm has, and no row is emitted twice from one source row
        srcs = [s for s in g.row_sources]
        assert len({s for s in srcs}) == len(srcs)
        seen_rows += len(srcs)
    assert seen_rows > 100 and seen_choice > 50  # the property was exercised on real choices


def test_the_invariant_check_rejects_a_forged_value() -> None:
    ft, zs = doc({"invoice_number": "F"}, [row("S1")]), doc({"invoice_number": "Z"}, [row("S1")])
    good = doc({"invoice_number": "F"}, [row("S1")])
    gate.assert_no_synthesis(good, ft, zs, [(0, 0)])
    bad_header = doc({"invoice_number": "INVENTED"}, [row("S1")])
    with pytest.raises(GateError, match="neither arm"):
        gate.assert_no_synthesis(bad_header, ft, zs, [(0, 0)])
    bad_row = doc({"invoice_number": "F"}, [row("S9")])
    with pytest.raises(GateError, match="neither arm"):
        gate.assert_no_synthesis(bad_row, ft, zs, [(0, 0)])
    with pytest.raises(GateError, match="no source"):
        gate.assert_no_synthesis(good, ft, zs, [(None, None)])
    with pytest.raises(GateError, match="length"):
        gate.assert_no_synthesis(good, ft, zs, [])


# ---- production shape and schema ---------------------------------------------------------------


def schema_errors(preds: dict[str, Any]) -> list[Any]:
    import jsonschema

    schema = json.loads((paths.assignment_dir() / "schema.json").read_text(encoding="utf-8"))
    return list(jsonschema.Draft202012Validator(schema).iter_errors(preds))


def test_gated_output_validates_against_the_repo_output_schema() -> None:
    w = world()
    keep = ("doc_type", "header", "line_items")  # the production shape (gold also has doc_id)
    zs = {d: {k: copy.deepcopy(g[k]) for k in keep} for d, g in w["gold"].items()}
    ft = copy.deepcopy(zs)
    for p in ft.values():
        if p["doc_type"] == "invoice":
            p["header"]["buyer_name"] = None  # FT leaves a field blank
            p["line_items"][0]["quantity"] = "9"  # and disagrees on a row cell
    p_zs: dict[str, Any] = {}
    p_ft: dict[str, Any] = {}
    for d in zs:
        p_zs[d] = rand_probs(random.Random(1), zs[d])
        p_ft[d] = rand_probs(random.Random(2), ft[d])
    out, stats = gate.gate_predictions(ft, zs, p_ft, p_zs, sorted(zs))
    assert sorted(out) == sorted(zs)
    assert schema_errors(out) == []
    for p in out.values():  # the production shape: exactly these three keys
        assert set(p) == {"doc_type", "header", "line_items"}
    assert stats["rows.paired"] == 12  # six invoices x 2 rows


def test_gate_predictions_fails_if_coerce_would_change_the_output() -> None:
    zs = {"d": doc({"invoice_number": "Z"}, [row("S1", qty=5.0)])}  # a bare float is not coerced
    ft = copy.deepcopy(zs)
    with pytest.raises(GateError, match="not production-shaped"):
        gate.gate_predictions(
            ft, zs, {"d": probs({"invoice_number": 1.0}, all_rows(1, 1.0))},
            {"d": probs({"invoice_number": 1.0}, all_rows(1, 1.0))}, ["d"],
        )  # fmt: skip


def test_inputs_are_not_mutated() -> None:
    ft = doc({"invoice_number": "F"}, [row("S1", qty="1")])
    zs = doc({"invoice_number": "Z"}, [row("S1", qty="2")])
    before = copy.deepcopy((ft, zs))
    gate_document(
        ft,
        zs,
        probs({"invoice_number": 1.0}, all_rows(1, 1.0)),
        probs({"invoice_number": 0.0}, all_rows(1, 0.0)),
    )
    assert (ft, zs) == before


# ---- probability plumbing ----------------------------------------------------------------------


def test_probs_by_doc_keeps_only_emitted_fields_and_rejects_bad_p() -> None:
    keys = [
        cf.FieldKey("d1", "header", "invoice_number", -1),
        cf.FieldKey("d1", "row", "quantity", 0),
        cf.FieldKey("d2", "header", "buyer_name", -1),
    ]
    emitted = np.array([True, False, True])
    out = gate.probs_by_doc(keys, np.array([0.7, np.nan, 0.2]), emitted)
    assert out == {"d1": {("header", "invoice_number", -1): 0.7},
                   "d2": {("header", "buyer_name", -1): 0.2}}  # fmt: skip
    for bad in (math.nan, 1.5, -0.1, math.inf):
        with pytest.raises(GateError, match="no valid P"):
            gate.probs_by_doc(keys, np.array([bad, 0.0, 0.2]), emitted)


# ---- the gate is never fitted on the docs it is applied to -------------------------------------


def test_fit_apply_disjointness_is_asserted() -> None:
    groups = {"a": "g1", "b": "g1", "c": "g2", "d": "g3"}
    gate.assert_fit_apply_disjoint(["a", "b"], ["c", "d"], groups)  # supplier-disjoint: fine
    with pytest.raises(GateError, match="both the fit set and the apply set"):
        gate.assert_fit_apply_disjoint(["a", "c"], ["c", "d"], groups)
    with pytest.raises(GateError, match="supplier group"):
        gate.assert_fit_apply_disjoint(["a", "c"], ["b", "d"], groups)  # b shares g1 with a
    with pytest.raises(GateError, match="no supplier group"):
        gate.assert_fit_apply_disjoint(["a"], ["zz"], groups)  # unknown group: cannot show disjoint


# ---- the replacement rule (spec section 11 item 7) ----------------------------------------------


def counts(ff: Any, on: Any) -> dict[str, Any]:
    return {"false_fill_total": ff, "over_null_total": on}


@pytest.mark.parametrize("winner", [FT_LABEL, ZS_LABEL])
@pytest.mark.parametrize(
    ("lo", "g", "w", "replaces", "failed"),
    [
        (0.01, counts(3, 5), counts(3, 5), True, []),  # equal counts pass
        (0.01, counts(2, 4), counts(3, 5), True, []),  # fewer pass
        (1e-9, counts(3, 5), counts(3, 5), True, []),  # any strictly positive bound passes
        (0.0, counts(3, 5), counts(3, 5), False, ["G1"]),  # bound exactly 0: NOT replacing
        (-0.0, counts(0, 0), counts(9, 9), False, ["G1"]),
        (-1e-9, counts(3, 5), counts(3, 5), False, ["G1"]),
        (0.01, counts(4, 5), counts(3, 5), False, ["G2"]),  # one more false fill
        (0.01, counts(3, 6), counts(3, 5), False, ["G3"]),  # one more over-null
        (0.01, counts(4, 6), counts(3, 5), False, ["G2", "G3"]),
        (0.0, counts(4, 6), counts(3, 5), False, ["G1", "G2", "G3"]),  # every clause reported
        (0.01, counts(0, 6), counts(3, 5), False, ["G3"]),  # a better count does not offset
        (0.01, counts(4, 0), counts(3, 5), False, ["G2"]),
    ],
)
def test_decide_g_truth_table(
    winner: str, lo: float, g: Any, w: Any, replaces: bool, failed: list[str]
) -> None:
    d = decide_g(winner, lo, g, w)
    assert d.replaces is replaces and [c.key for c in d.clauses if not c.passed] == failed
    assert [c.key for c in d.clauses] == ["G1", "G2", "G3"]  # all three always evaluated
    assert d.winner == winner and d.final == (winner if not replaces else "G (per-field gate)")
    assert d.line.startswith(
        f"SECONDARY SYSTEM G REPLACES {winner}" if replaces else "SECONDARY SYSTEM G DOES NOT"
    )
    assert all(f"{k} failed" in d.line for k in failed)


def test_every_combination_of_the_three_conditions() -> None:
    for c1 in (True, False):
        for c2 in (True, False):
            for c3_ in (True, False):
                d = decide_g(
                    ZS_LABEL,
                    0.01 if c1 else 0.0,
                    counts(3 if c2 else 4, 5 if c3_ else 6),
                    counts(3, 5),
                )
                assert d.replaces is (c1 and c2 and c3_)


@pytest.mark.parametrize(
    ("lo", "g", "w", "failed"),
    [
        (math.nan, counts(1, 1), counts(1, 1), ["G1"]),
        (math.inf, counts(1, 1), counts(1, 1), ["G1"]),  # not finite: closed, not a pass
        (None, counts(1, 1), counts(1, 1), ["G1"]),
        (0.01, counts(math.nan, 1), counts(1, 1), ["G2"]),
        (0.01, counts(1, 1), counts(1, math.nan), ["G3"]),
        (0.01, {"false_fill_total": 1}, counts(1, 1), ["G3"]),  # missing key
        (0.01, counts(True, 1), counts(1, 1), ["G2"]),  # a bool is not a count
        (0.01, counts("0", 1), counts(1, 1), ["G2"]),
    ],
)
def test_decide_g_fails_closed(lo: Any, g: Any, w: Any, failed: list[str]) -> None:
    d = decide_g(ZS_LABEL, lo, g, w)
    assert not d.replaces and [c.key for c in d.clauses if not c.passed] == failed
    assert "failed closed" in d.line


def test_decide_g_refuses_an_unknown_winner() -> None:
    with pytest.raises(GateError, match="winner must be"):
        decide_g("G", 0.1, counts(0, 0), counts(0, 0))


def test_decide_g_from_pair_reads_g_as_the_oof_slot() -> None:
    """compare_models(winner, G): the zero_shot slot is the winner, the oof slot is G."""
    w = world()
    good = {d: copy.deepcopy(g) for d, g in w["gold"].items()}
    bad = copy.deepcopy(good)
    for p in bad.values():
        if p["doc_type"] == "invoice":
            for r in p["line_items"]:
                r["quantity"] = "7"
    g_wins = oof.compare_models(bad, good, w["gold"], None, 20, 42)  # winner bad, G good
    g_loses = oof.compare_models(good, bad, w["gold"], None, 20, 42)
    assert gate.decide_g_from_pair(ZS_LABEL, g_wins).replaces
    d = gate.decide_g_from_pair(FT_LABEL, g_loses)
    assert not d.replaces and [c.key for c in d.clauses if not c.passed] == ["G1"]
