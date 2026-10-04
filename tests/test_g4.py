"""Unit tests for shipdoc.g4 (verdicts, paired sign convention, tables) on synthetic documents."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from shipdoc import g4, oof


def inv_gold(k: int) -> dict[str, Any]:
    return {
        "doc_id": f"d{k}",
        "doc_type": "invoice",
        "header": {
            "invoice_number": f"INV-{k}",
            "invoice_date": "2026-01-02",
            "supplier_name": f"Sup {k}",
            "buyer_name": "Buyer",
            "ship_to_name": "Ship",
            "currency": "USD",
            "total_amount": "10.50",
            "awb_number": None,
        },
        "line_items": [
            {
                "supplier_part_number": f"P{k}{i}",
                "customer_part_number": None,
                "purchase_order": f"PO{k}{i}",
                "quantity": "5",
            }
            for i in range(2)
        ],
    }


GOLD = {f"d{k}": inv_gold(k) for k in range(8)}


def damaged(gold: dict[str, Any], ids: list[str], what: str) -> dict[str, Any]:
    """A prediction set equal to gold except `what` is damaged on `ids`."""
    pred = copy.deepcopy(gold)
    for d in ids:
        if what == "null_header":
            pred[d]["header"]["supplier_name"] = None
        elif what == "null_cpn":
            for r in pred[d]["line_items"]:
                r["purchase_order"] = None
    return pred


# ---- interim verdict -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lo", "ft_n", "zs_n", "verdict", "n_failed"),
    [
        (0.02, 3, 3, "NO REGRESSION", 0),  # equal over-nulls pass (<=)
        (0.02, 4, 3, "REGRESSION", 1),  # one more over-null cell fails
        (-0.0099, 0, 5, "NO REGRESSION", 0),  # just above the floor
        (-0.01, 0, 5, "REGRESSION", 1),  # the exact -1.0 point boundary is a fail (strict >)
        (-0.0100001, 0, 0, "REGRESSION", 1),
        (-0.03, 9, 1, "REGRESSION", 2),  # both clauses named
        (0.0, 0, 0, "NO REGRESSION", 0),  # CI including 0 is fine for the interim verdict
    ],
)
def test_interim_verdict_table(
    lo: float, ft_n: int, zs_n: int, verdict: str, n_failed: int
) -> None:
    v = g4.interim_verdict(lo, ft_n, zs_n, 0.05)
    assert v["verdict"] == verdict and len(v["failed_clauses"]) == n_failed
    assert v["oof_over_null"] == ft_n and v["zs_over_null"] == zs_n  # FT counted as the oof side


def test_interim_line_names_label_verdict_and_basis() -> None:
    line = g4.interim_line(g4.interim_verdict(-0.02, 1, 0), "rules arms")
    assert line.startswith("G4 INTERIM VERDICT (interim, one fold, not the final G4 decision)")
    assert "REGRESSION" in line and "[basis: rules arms]" in line and "over-null" in line


# ---- spec clause -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lo", "ft_ff", "zs_ff", "verdict"),
    [
        (0.001, 0.01, 0.02, "FINE-TUNED KEPT"),
        (0.001, 0.02, 0.02, "FINE-TUNED KEPT"),  # false-fill equal is "not worse"
        (0.0, 0.0, 0.0, "FINE-TUNED NOT KEPT"),  # CI must EXCLUDE 0 (lower bound strictly > 0)
        (-0.005, 0.0, 0.0, "FINE-TUNED NOT KEPT"),
        (0.01, 0.03, 0.02, "FINE-TUNED NOT KEPT"),  # false-fill worse
    ],
)
def test_spec_clause_table(lo: float, ft_ff: float, zs_ff: float, verdict: str) -> None:
    v = g4.spec_g4_clause(lo, ft_ff, zs_ff)
    assert v["verdict"] == verdict
    assert g4.spec_line(v).startswith("G4 SPEC CLAUSE (interim, one fold, not the G4 decision)")
    if verdict.endswith("NOT KEPT"):
        assert v["failed_clauses"]


def test_spec_clause_names_both_failures() -> None:
    v = g4.spec_g4_clause(-0.01, 0.05, 0.01)
    assert len(v["failed_clauses"]) == 2


# ---- paired sign convention and verdict plumbing ---------------------------------------------


def test_paired_delta_is_ft_minus_zs_and_verdicts_use_it() -> None:
    zs = damaged(GOLD, ["d0", "d1", "d2", "d3"], "null_header")  # worse arm
    ft = copy.deepcopy(GOLD)  # perfect arm
    good = oof.compare_models(zs, ft, GOLD, None, n_boot=50, seed=g4.SEED)
    bad = oof.compare_models(ft, zs, GOLD, None, n_boot=50, seed=g4.SEED)  # arms swapped
    d_good = good["subsets"]["all"]["paired_delta_oof_minus_zero_shot"]["OVERALL"]
    d_bad = bad["subsets"]["all"]["paired_delta_oof_minus_zero_shot"]["OVERALL"]
    assert d_good["delta"] > 0 > d_bad["delta"]  # delta = second arg (FT) minus first arg (ZS)
    assert d_good["delta"] == pytest.approx(-d_bad["delta"])
    v = g4.verdicts(good, good)
    assert v["interim_rules"]["verdict"] == "NO REGRESSION"
    assert v["interim_raw"]["verdict"] == "NO REGRESSION"
    assert v["spec_rules"]["delta_ci_lo"] == d_good["lo"]
    worse = g4.verdicts(bad, good)  # FT clearly worse on the first, fine on the second
    assert worse["interim_rules"]["verdict"] == "REGRESSION"
    assert worse["interim_raw"]["verdict"] == "NO REGRESSION"
    assert worse["spec_rules"]["verdict"] == "FINE-TUNED NOT KEPT"


def test_verdict_over_null_clause_counts_ft_against_zs() -> None:
    zs = copy.deepcopy(GOLD)
    ft = damaged(GOLD, ["d0"], "null_header")  # one header over-null only in FT
    pair = oof.compare_models(zs, ft, GOLD, None, n_boot=20, seed=g4.SEED)
    a = pair["subsets"]["all"]
    assert (
        a["oof"]["over_null"]["over_null_total"],
        a["zero_shot"]["over_null"]["over_null_total"],
    ) == (1, 0)
    assert (
        "over-null cells 1 (OOF) > 0 (zero-shot)"
        in g4.verdicts(pair, pair)["interim_rules"]["failed_clauses"]
    )


# ---- tables ------------------------------------------------------------------------------------


def test_excludes_zero_flags_the_side() -> None:
    assert g4.excludes_zero(0.001, 0.02) == "yes (positive)"
    assert g4.excludes_zero(-0.02, -0.001) == "yes (negative)"
    assert g4.excludes_zero(0.0, 0.02) == "no" and g4.excludes_zero(-0.01, 0.01) == "no"


def _blocks(**counts: dict[str, Any]) -> dict[str, Any]:
    base = {
        "header_over_null_by_field": {},
        "header_over_null": 0,
        "row_over_null_by_field": {"customer_part_number": 0},
        "row_over_null": 0,
        "over_null_total": 0,
        "header_false_fill": 0,
        "row_false_fill": 0,
        "rows_unmatched_gold": 0,
    }
    return {a: {"over_null": {**base, **counts.get(a, {})}} for a in g4.ARMS}


def test_over_null_table_counts_per_arm_and_ft_minus_zs() -> None:
    blocks = _blocks(
        zs_raw={
            "header_over_null_by_field": {"mawb": 4},
            "header_over_null": 4,
            "over_null_total": 4,
        },
        ft_rules={"row_over_null_by_field": {"customer_part_number": 2}, "row_over_null": 2,
                  "over_null_total": 2},
    )  # fmt: skip
    lines = g4.over_null_table(blocks).splitlines()
    # columns: count | ZS + rules | FT + rules | ZS raw | FT raw | FT - ZS (rules arms)
    mawb = next(x for x in lines if "header over-null: mawb" in x)
    assert [c.strip() for c in mawb.strip("|").split("|")[1:]] == ["0", "0", "4", "0", "0"]
    cpn = next(x for x in lines if "customer_part_number" in x)
    assert [c.strip() for c in cpn.strip("|").split("|")[1:]] == ["0", "2", "0", "0", "2"]
    total = next(x for x in lines if "OVER-NULL CELLS" in x)
    assert [c.strip() for c in total.strip("|").split("|")[1:]] == ["0", "2", "4", "0", "2"]


def test_rules_effect_sums_splits_and_reports_rules_on_minus_off() -> None:
    raw = damaged(GOLD, ["d0", "d1"], "null_header")
    final = copy.deepcopy(GOLD)  # the rules "fix" both docs
    summary = {
        "eligible_docs": {"R1": 4, "R2": 4, "R3": 8},
        "touched_docs": {"R1": 2, "R2": 0, "R3": 0},
        "changes": {"R1": 2, "R2": 0, "R3": 0},
        "skipped": {"R2/no_ocr": 1},
    }

    def fake_counts(sc: Any, b: Any, n: Any, gold: Any) -> dict[str, Any]:
        return {
            "R1": {"train": {"fixed": 1, "broken": 0, "neutral": 0},
                   "dev": {"fixed": 1, "broken": 1, "neutral": 0}},
            "R2": {},
            "R3": {},
        }  # fmt: skip

    eff = g4.rules_effect(None, raw, final, GOLD, summary, fake_counts, n_boot=30)
    r1 = eff["per_rule"]["R1"]
    assert (r1["fixed"], r1["broken"], r1["net"], r1["touched_docs"], r1["eligible_docs"]) == (
        2,
        1,
        1,
        2,
        4,
    )
    assert eff["per_rule"]["R3"]["touched_docs"] == 0 and eff["skipped"] == {"R2/no_ocr": 1}
    assert eff["delta"] > 0 and eff["overall_on"] > eff["overall_off"]  # on minus off


def test_r3_note_states_when_r3_did_not_fire() -> None:
    def eff(touched: int) -> dict[str, Any]:
        return {"per_rule": {"R3": {"touched_docs": touched, "cells_changed": 2 * touched}}}

    text = g4.r3_note({"zs": eff(0), "ft": eff(3)}, 0)
    assert "ZS: R3 did NOT fire" in text and "FT: touched 3 docs" in text
    assert "only on folds where it fires" in text
