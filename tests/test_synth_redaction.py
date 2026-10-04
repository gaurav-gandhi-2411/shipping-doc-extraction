"""Tests for the synthetic-redaction recipe builder and the committed recipe file."""

from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from shipdoc import augment as A

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "make_synthetic_redaction", ROOT / "scripts/make_synthetic_redaction.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

META = json.loads((ROOT / "meta" / "dev.json").read_text(encoding="utf-8"))
RECIPE_KEYS = {"variant_id", "doc_id", "field", "row_idx", "page", "method", "seed", "label"}
OPTIONAL_KEYS = {"inferable_from_siblings", "n_boxes", "all_occurrences", "backfill"}


def _cands() -> list[dict]:
    """Invented candidates: three strata of different capacity, mixed scanned/digital."""
    out = []
    for stratum, n in (("invoice_header:a", 30), ("waybill_header:b", 8), ("row:c", 3)):
        for i in range(n):
            eligible = ["black_box", "scribble", "smudge"] + (["edge_crop"] if i % 3 == 0 else [])
            out.append(
                {
                    "stratum": stratum,
                    "doc_id": f"dev_{i:04d}",
                    "field": stratum.split(":")[1],
                    "row_idx": None if stratum != "row:c" else i,
                    "page": 0,
                    "scanned": i % 2 == 0,
                    "eligible": eligible,
                }
            )
    return out


def test_quotas_water_filling() -> None:
    assert mod.quotas({"a": 30, "b": 8, "c": 3}, 20) == {"a": 9, "b": 8, "c": 3}
    assert mod.quotas({"a": 2, "b": 2}, 100) == {"a": 2, "b": 2}  # capped by availability
    assert sum(mod.quotas({"a": 50, "b": 50, "c": 50}, 100).values()) == 100


def test_select_is_deterministic_and_respects_eligibility() -> None:
    picked = mod.select(_cands(), 20, 0.5)
    again = mod.select(_cands(), 20, 0.5)
    assert picked == again and len(picked) == 20
    assert all(p["method"] in p["eligible"] for p in picked)
    assert len({(p["stratum"], p["doc_id"], p["row_idx"]) for p in picked}) == 20  # no duplicates
    assert {p["stratum"] for p in picked} == {"invoice_header:a", "waybill_header:b", "row:c"}
    assert mod.select(_cands(), 20, 0.5) != mod.select(_cands(), 20, 0.5, seed=7)


def test_to_recipes_ids_and_seeds() -> None:
    recipes = mod.to_recipes(mod.select(_cands(), 20, 0.5))
    assert all(set(r) == RECIPE_KEYS and r["label"] == "synthetic" for r in recipes)
    assert len({r["variant_id"] for r in recipes}) == 20
    assert all(r["variant_id"].startswith(r["doc_id"] + "__syn") for r in recipes)
    assert recipes == mod.to_recipes(mod.select(_cands(), 20, 0.5))


def test_collateral_overlap_rule() -> None:
    others = [(0, (100.0, 100.0, 200.0, 120.0), "h:other")]
    assert mod.collateral((90.0, 90.0, 210.0, 130.0), 0, others, "h:own")  # covers it entirely
    assert not mod.collateral((90.0, 90.0, 210.0, 130.0), 1, others, "h:own")  # other page
    assert not mod.collateral((90.0, 90.0, 210.0, 130.0), 0, others, "h:other")  # own box
    assert not mod.collateral((100.0, 119.0, 200.0, 140.0), 0, others, "h:own")  # 5% sliver


def test_composition_table_counts() -> None:
    tags = {"d1": {"waybill": False, "scanned": True}, "d2": {"waybill": True, "scanned": False}}
    recipes = [
        {"doc_id": "d1", "field": "quantity", "row_idx": 0, "method": "smudge"},
        {"doc_id": "d1", "field": "currency", "row_idx": None, "method": "black_box"},
        {"doc_id": "d2", "field": "pieces", "row_idx": None, "method": "black_box"},
    ]
    md = A.composition_markdown(recipes, tags)
    assert "| row:quantity | - | - | 1/0 | - | 1 (1/0) |" in md
    assert "| waybill_header:pieces | 0/1 | - | - | - | 1 (0/1) |" in md
    assert "| **all** | 2 (1/1) | 0 (0/0) | 1 (1/0) | 0 (0/0) | 3 (2/1) |" in md


def test_committed_recipe_file() -> None:
    payload = json.loads((ROOT / "splits" / "synthetic_redaction_dev.json").read_text("utf-8"))
    recipes = payload["recipes"]
    assert payload["label"] == "synthetic" and payload["n_recipes"] == len(recipes) >= 150
    dev = {m["doc_id"]: m for m in META}
    assert all(RECIPE_KEYS <= set(r) <= RECIPE_KEYS | OPTIONAL_KEYS for r in recipes)
    assert len({r["variant_id"] for r in recipes}) == len(recipes)
    assert all(r["doc_id"] in dev and r["method"] in A.METHODS for r in recipes)  # dev docs only
    assert {r["method"] for r in recipes} == set(A.METHODS)
    assert any(dev[r["doc_id"]]["scanned"] for r in recipes)
    assert any(not dev[r["doc_id"]]["scanned"] for r in recipes)
    # Confidentiality: recipes carry no gold values, only ids/field names/ints.
    assert all(isinstance(r["seed"], int) and r["field"].isidentifier() for r in recipes)


@pytest.mark.skipif(
    not (ROOT / "assignment" / "score.py").is_file(), reason="assignment/score.py absent"
)
def test_eval_labels_synthetic_runs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from shipdoc import eval as ev

    gold = {
        "doc_id": "dev_0000__syn001",
        "doc_type": "waybill",
        "header": {"carrier": "Invented Air", "mawb": None, "pieces": "3"},
        "line_items": [],
        "pages": ["dev_0000__syn001_p1.png"],
        "synthetic": True,
    }
    (tmp_path / "labels").mkdir()
    (tmp_path / "labels" / "dev_0000__syn001.json").write_text(json.dumps(gold), encoding="utf-8")
    pred = {gold["doc_id"]: {k: gold[k] for k in ("doc_type", "header", "line_items")}}
    (tmp_path / "pred.json").write_text(json.dumps(pred), encoding="utf-8")
    out = tmp_path / "report.json"
    argv = ["--gold", str(tmp_path / "labels"), "--pred", str(tmp_path / "pred.json")]
    assert ev.main([*argv, "--bootstrap", "0", "--out", str(out)]) == 0
    assert "SYNTHETIC redaction eval" in capsys.readouterr().out
    assert (
        out.with_suffix(".md").read_text(encoding="utf-8").startswith("# SYNTHETIC redaction eval")
    )


def test_harmed_tags_only_when_every_copy_is_hit() -> None:
    others = [
        (0, (100.0, 100.0, 140.0, 120.0), "h:cur"),  # copy 1, next to the occluded value
        (0, (500.0, 100.0, 540.0, 120.0), "h:cur"),  # copy 2, far away
        (0, (100.0, 200.0, 140.0, 220.0), "h:solo"),  # single copy
        (0, (100.0, 100.0, 140.0, 120.0), "h:own"),
    ]
    hit = [(0, (90.0, 95.0, 145.0, 125.0))]
    assert mod.harmed_tags(hit, others, "h:own") == set()  # cur keeps a clean copy
    hit2 = [*hit, (0, (90.0, 195.0, 145.0, 225.0))]
    assert mod.harmed_tags(hit2, others, "h:own") == {"h:solo"}
    hit3 = [*hit2, (0, (490.0, 95.0, 545.0, 125.0))]
    assert mod.harmed_tags(hit3, others, "h:own") == {"h:solo", "h:cur"}
    assert mod.harmed_tags([(1, hit[0][1])], others, "h:own") == set()  # other page


def test_count_on_line() -> None:
    assert mod.count_on_line("Currency: USD  Total USD 60.00", "currency", "USD") == 2
    assert mod.count_on_line("EUROUSDX", "currency", "USD") == 0  # inside a word
    assert mod.count_on_line("Total 1,060.50 net 1060.50", "total_amount", "1060.50") == 2
    assert mod.count_on_line("Total 1,060.50 tax 60.00", "total_amount", "1060.50") == 1


def _pool() -> list[dict]:
    out = []
    for st in mod.BACKFILL_STRATA:
        for i in range(30):
            out.append(
                {
                    "stratum": st,
                    "doc_id": f"dev_{i:04d}",
                    "field": st.split(":")[1],
                    "row_idx": None,
                    "page": 0,
                    "scanned": i % 3 == 0,
                    "eligible": ["black_box", "smudge"],
                    "seeds": {"black_box": 100 + i, "smudge": 200 + i},
                }
            )
    return out


def test_select_backfill_reaches_minimums_and_keeps_base_pairs_out() -> None:
    tags = {f"dev_{i:04d}": i % 3 == 0 for i in range(30)}
    digital = [i for i in range(30) if i % 3 != 0][:11]  # the first batch had no scanned awb
    base = [
        {"variant_id": f"dev_{i:04d}__syn001", "doc_id": f"dev_{i:04d}", "field": "awb_number",
         "row_idx": None, "method": "black_box"}
        for i in digital
    ]  # fmt: skip
    picked = mod.select_backfill(_pool(), base, tags, 0.43)
    again = mod.select_backfill(_pool(), base, tags, 0.43)
    assert [(c["doc_id"], c["field"], c["method"]) for c in picked] == [
        (c["doc_id"], c["field"], c["method"]) for c in again
    ]
    by = Counter(c["field"] for c in picked)
    assert by["currency"] == 15 and by["total_amount"] == 15
    awb = [c for c in picked if c["field"] == "awb_number"]
    assert 11 + len(awb) >= 15 and sum(c["scanned"] for c in awb) >= 5
    assert not {c["doc_id"] for c in awb} & {r["doc_id"] for r in base}  # no repeated pair
    assert all(c["method"] in c["eligible"] for c in picked)


def test_to_backfill_recipes_appends_new_indices_and_picks_seed_of_method() -> None:
    base = [{"variant_id": "dev_0001__syn003", "doc_id": "dev_0001"}]
    picked = [
        {"stratum": "invoice_header:currency", "doc_id": "dev_0001", "field": "currency",
         "page": 0, "method": "smudge", "seeds": {"smudge": 77}, "n_boxes": 2},
        {"stratum": "invoice_header:awb_number", "doc_id": "dev_0002", "field": "awb_number",
         "page": 0, "method": "black_box", "seeds": {"black_box": 5}},
    ]  # fmt: skip
    r = mod.to_backfill_recipes(picked, base)
    assert r[0]["variant_id"] == "dev_0001__syn004" and r[0]["seed"] == 77  # after the base's
    assert r[0]["n_boxes"] == 2 and r[0]["all_occurrences"] is True and r[0]["backfill"] is True
    assert r[1]["variant_id"] == "dev_0002__syn001" and "n_boxes" not in r[1]


def test_feasible_seed_checks_the_applied_box_not_the_widest() -> None:
    # Neighbour 6 px above a 24 px line: the widest box (pad 6 + jitter up to 3.6) can overlap more
    # than 10% of it, but a low-jitter seed stays clear. Another neighbour sits inside the box.
    box = (200.0, 150.0, 320.0, 174.0)
    tgt = [A.Target(0, box, "exact", 5, 1)]
    above = (200.0, 120.0, 320.0, 144.0)
    others = [(0, above, "h:date")]
    sizes = {0: (600, 400)}
    s = mod.feasible_seed(tgt, "black_box", sizes, others, "h:own", False, np.random.default_rng(0))
    assert s is not None
    ab = A.plan_box(box, "black_box", A.box_rng(s, 0, False), sizes[0])
    assert mod._overlap_frac(tuple(map(float, ab)), above) <= mod.COLLATERAL_MAX_FRAC
    inside = [(0, (210.0, 155.0, 250.0, 170.0), "h:date")]
    none = mod.feasible_seed(
        tgt, "black_box", sizes, inside, "h:own", False, np.random.default_rng(0)
    )
    assert none is None
    # edge_crop on a central box is not available at all
    assert (
        mod.feasible_seed(tgt, "edge_crop", sizes, [], "h:own", False, np.random.default_rng(0))
        is None
    )


def test_committed_backfill_minimums_and_inferable_tag() -> None:
    payload = json.loads((ROOT / "splits" / "synthetic_redaction_dev.json").read_text("utf-8"))
    recipes = payload["recipes"]
    dev = {m["doc_id"]: m for m in META}
    for field in ("currency", "total_amount", "awb_number"):
        assert sum(r["field"] == field for r in recipes) >= 15, field
    awb = [r for r in recipes if r["field"] == "awb_number"]
    assert sum(bool(dev[r["doc_id"]]["scanned"]) for r in awb) >= 5
    # Every row-target recipe is tagged with a real bool; header recipes are not.
    assert all(
        isinstance(r.get("inferable_from_siblings"), bool) == (r["row_idx"] is not None)
        for r in recipes
    )
    # Backfill recipes come after the first batch and never reuse a variant id or a (doc, field).
    first = [r for r in recipes if not r.get("backfill")]
    assert recipes[: len(first)] == first and len(first) == 200
    assert len({(r["doc_id"], r["field"], r["row_idx"]) for r in recipes}) == len(recipes)
    multi = [r for r in recipes if r.get("all_occurrences")]
    assert multi and all(isinstance(r["n_boxes"], int) and r["n_boxes"] >= 1 for r in multi)
    assert (
        sum(r["inferable_from_siblings"] for r in recipes if r["row_idx"] is not None)
        == (payload["inferable_from_siblings"]["n_true"])
    )


@pytest.mark.skipif(
    not (ROOT / "assignment" / "score.py").is_file(), reason="assignment/score.py absent"
)
def test_eval_breaks_out_inferable_row_targets() -> None:
    from shipdoc import eval as ev

    def label(i: int, recipe: dict, synthetic: bool = True) -> dict:
        return {"doc_id": f"d{i}", "doc_type": "waybill", "synthetic": synthetic, "recipe": recipe}

    row = {"field": "quantity", "row_idx": 0}
    gold = {
        "d0": label(0, {**row, "inferable_from_siblings": True}),
        "d1": label(1, {**row, "inferable_from_siblings": False}),
        "d2": label(2, {"field": "carrier", "row_idx": None}),
    }
    sl = ev.slice_doc_ids(gold, None)
    assert sl["inferable_from_siblings=yes"] == ["d0"]
    assert sl["inferable_from_siblings=no"] == ["d1"]  # header recipe is in neither
    assert sl["target=row"] == ["d0", "d1"] and sl["target=header"] == ["d2"]
    # Not a pure synthetic run (or no row targets): no extra slices, so real runs are unchanged.
    gold["d2"]["synthetic"] = False
    assert "target=row" not in ev.slice_doc_ids(gold, None)
    assert "inferable_from_siblings=yes" not in ev.slice_doc_ids({"d2": gold["d2"]}, None)
