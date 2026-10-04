from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "finetune_estimate", ROOT / "scripts" / "finetune_estimate.py"
)
fe = importlib.util.module_from_spec(spec)
sys.modules["finetune_estimate"] = fe  # dataclasses resolve their module via sys.modules
spec.loader.exec_module(fe)

PAGES = {"val": [230, 217, 224], "train": [441, 454, 447], "all_train": 536, "all": 671}


def test_steps_rounds_partial_batches_up() -> None:
    assert fe.steps(536, 2, 1, 8) == 2 * 67
    assert fe.steps(8, 1, 1, 8) == 1
    assert fe.steps(9, 1, 1, 8) == 2


def test_flops_scale_with_tokens_and_decompose() -> None:
    a = fe.train_flops_per_page(2100, 531.3)
    b = fe.train_flops_per_page(2100, 1000)
    assert a["total"] == pytest.approx(a["layers"] + a["attn_scores"] + a["lm_head"] + a["vision"])
    assert b["total"] > a["total"]
    assert a["vision"] == fe.vision_flops()  # vision cost is independent of the target length


def test_seconds_per_page_inverse_in_throughput() -> None:
    slow, fast = fe.Throughput("T4", "x", 5.0), fe.Throughput("T4", "y", 10.0)
    assert fe.seconds_per_page(slow, 2100, 531.3) == pytest.approx(
        2 * fe.seconds_per_page(fast, 2100, 531.3)
    )
    with pytest.raises(ValueError, match="tflops"):
        fe.seconds_per_page(fe.Throughput("T4", "z", 0.0), 2100, 531.3)


def test_scenarios_order_and_l4_faster_than_t4() -> None:
    sc = {(t.gpu, t.label): t.tflops for t in fe.scenarios(base=10.0)}
    assert sc[("T4", "central")] == pytest.approx(7.5)
    assert sc[("T4", "low")] < sc[("T4", "central")] < sc[("T4", "high")]
    assert all(sc[("L4", k)] > sc[("T4", k)] for k in ("low", "central", "high"))


def test_stage_rows_all_folds_cost_three_loads_and_more_than_one_fold() -> None:
    tp = fe.Throughput("T4", "central", 8.0)
    rows = {n: s for n, s, _ in fe.stage_rows(PAGES, 2, 8, tp)}
    s_page = fe.seconds_per_page(tp, fe.INPUT_TOKENS, fe.TARGET_MEAN)
    assert rows["3 folds"] == pytest.approx(3 * fe.MODEL_LOAD_S + 2 * sum(PAGES["train"]) * s_page)
    assert rows["3 folds"] > rows["1 fold"] > rows["smoke 20 steps (accum 2)"]


def test_hours_cu_uses_gpu_rates() -> None:
    h, lo, hi = fe.hours_cu(3600, "T4")
    assert (h, lo, hi) == (1.0, 1.19, 1.58)
    assert fe.hours_cu(7200, "L4")[1] == pytest.approx(6.0)


def test_render_mentions_every_stage() -> None:
    text = fe.render(PAGES, 2, 8)
    for stage in ("smoke", "1 fold", "3 folds", "final all-train", "OOF inference"):
        assert stage in text


def test_kfold_seconds_halving_the_training_set_halves_the_pass_time() -> None:
    k3 = fe.kfold_seconds([441, 454, 447], 2, 5.0)
    k2 = fe.kfold_seconds([335, 336], 2, 5.0)
    assert k3 == pytest.approx(3 * fe.MODEL_LOAD_S + 2 * 1342 * 5.0)
    assert k2 < k3


def test_fold_balance_counts_types_groups_and_detects_leakage() -> None:
    meta = {
        "a": {"waybill": False, "supplier_group": "inv_1"},
        "b": {"waybill": False, "supplier_group": "inv_1"},
        "c": {"waybill": True, "supplier_group": "wb_1"},
        "d": {"waybill": False, "supplier_group": "inv_2"},
    }
    clean = fe.fold_balance(meta, [{"fold": 0, "val_doc_ids": ["a", "b", "c"]}])[0]
    assert (clean["docs"], clean["invoice_docs"], clean["waybill_docs"]) == (3, 2, 1)
    assert clean["invoice_groups"] == 1 and clean["invoice_docs_per_group"] == [2]
    assert clean["group_overlap_with_train"] == 0
    leaky = fe.fold_balance(meta, [{"fold": 0, "val_doc_ids": ["a"]}])[0]
    assert leaky["group_overlap_with_train"] == 1  # "b" shares inv_1 with the held-out "a"


# --------------------------------------------------------------------------------------------
# The 'measured' scenario (L4 smoke) and the OOF inference rows
# --------------------------------------------------------------------------------------------


def test_measured_scenario_overrides_the_throughput_assumption() -> None:
    tp = fe.measured_throughput()
    assert (tp.gpu, tp.label) == ("L4", "measured")
    assert fe.seconds_per_page(tp, fe.INPUT_TOKENS, fe.TARGET_MEAN) == pytest.approx(
        fe.MEASURED_S_PER_STEP / fe.MEASURED_STEP_PAGES
    )
    assert pytest.approx(8.65) == fe.MEASURED_S_PER_PAGE
    assert not any(t.label == "measured" for t in fe.scenarios())  # opt-in, tables unchanged
    withm = fe.scenarios(include_measured=True)
    assert [t.label for t in withm if t.gpu == "L4"] == ["low", "central", "high", "measured"]
    assert len(withm) == len(fe.scenarios()) + 1


def test_forward_only_page_costs_a_fraction_of_a_training_page() -> None:
    f = fe.forward_fraction()
    assert 0.3 < f < 0.5  # forward vs forward + recompute + input-gradient (+ the same vision)
    assert fe.forward_fraction(2100, 1000) < fe.forward_fraction(2100, 100) + 1  # sanity: finite


def test_eval_count_merges_epoch_ends_with_periodic_evals() -> None:
    assert fe.eval_count(112, 10, 56) == 13  # 10..110 (11) + epoch ends 56 and 112
    assert fe.eval_count(100, 10, 50) == 10  # both epoch ends are multiples of 10
    assert fe.eval_count(20, 10, 40) == 2  # smoke-shaped: no epoch end inside the run
    assert fe.eval_count(6, 0, 3) == 2  # periodic evaluation off: epoch ends only


def test_measured_rows_arithmetic() -> None:
    rows = {r["stage"]: r for r in fe.measured_rows(PAGES, 2, 8)}
    s_page = fe.MEASURED_S_PER_PAGE
    ev = 13 * fe.EVAL_PAGES * s_page * fe.forward_fraction()
    assert rows["fold0"]["visits"] == 2 * 441 and rows["fold0"]["steps"] == 112
    assert rows["fold0"]["evals"] == 13
    assert rows["fold0"]["hours"] == pytest.approx((fe.MODEL_LOAD_S + 882 * s_page + ev) / 3600)
    assert rows["fold0"]["cu"] == pytest.approx(rows["fold0"]["hours"] * 3.0)
    assert rows["smoke"]["evals"] == 0 and rows["smoke"]["steps"] == 20
    assert rows["3 folds"]["hours"] == pytest.approx(
        sum(rows[f"fold{k}"]["hours"] for k in range(3))  # three runs = three model loads
    )
    assert rows["final"]["visits"] == 2 * 536 and rows["final"]["steps"] == 134


def test_oof_rows_scale_with_pages_and_hypothetical_batch_speedup() -> None:
    rows = fe.oof_rows(PAGES)
    by = {(r["set"], r["speedup"]): r for r in rows}
    assert by[("fold0 held-out", 1.0)]["pages"] == 230
    assert by[("fold0 held-out", 1.0)]["hours"] == pytest.approx(230 * 36.0 / 3600)
    assert by[("all folds (OOF)", 1.0)]["hours"] == pytest.approx(671 * 36.0 / 3600)
    assert by[("all folds (OOF)", 3.0)]["hours"] == pytest.approx(
        by[("all folds (OOF)", 1.0)]["hours"] / 3
    )
    assert by[("fold0 held-out", 2.0)]["cu_low"] == pytest.approx(
        by[("fold0 held-out", 2.0)]["hours"] * 1.19
    )
    assert by[("fold0 held-out", 2.0)]["cu_high"] == pytest.approx(
        by[("fold0 held-out", 2.0)]["hours"] * 1.58
    )


def test_render_includes_the_measured_and_oof_tables_labelled_estimate() -> None:
    text = fe.render(PAGES, 2, 8)
    assert "MEASURED L4 bf16" in text and "ESTIMATE" in text and "HYPOTHETICAL" in text
    assert "fold0 held-out" in text and "all folds (OOF)" in text


def test_eval_defaults_match_the_committed_config() -> None:
    import yaml

    cfg = yaml.safe_load((ROOT / "configs" / "finetune_qwen35_4b.yaml").read_text(encoding="utf-8"))
    assert (cfg["eval_pages"], cfg["eval_every"]) == (fe.EVAL_PAGES, fe.EVAL_EVERY)
