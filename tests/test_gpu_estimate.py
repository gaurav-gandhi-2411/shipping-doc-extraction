from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gpu_estimate", ROOT / "scripts" / "gpu_estimate.py")
ge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ge)

SPEED = json.loads((ROOT / "configs" / "spike_speed.json").read_text(encoding="utf-8"))
ORDER = json.loads((ROOT / "configs" / "spike_order.json").read_text(encoding="utf-8"))["order"]
SIX = [
    f"{m}_{a}_compact"
    for m in ("qwen35_4b", "nuextract3", "qwen3vl_8b")
    for a in ("img_only", "img_ocr")
]


def test_formula() -> None:
    assert ge.est_s_per_page(2.0, 20.0, 400) == pytest.approx(22.0)
    assert ge.est_s_per_page(0.0, 12.5, 386.2) == pytest.approx(30.896)
    with pytest.raises(ValueError, match="decode_tok_s"):
        ge.est_s_per_page(1.0, 0.0, 100)


def test_estimate_config_strips_compact_suffix_and_scales_with_pages() -> None:
    a = ge.estimate_config("qwen35_4b_img_only_compact", SPEED, 55)
    b = ge.estimate_config("qwen35_4b_img_only", SPEED, 110)
    assert a["s_page"] == pytest.approx(b["s_page"]) and b["hours"] == pytest.approx(2 * a["hours"])
    m = SPEED["models"]["qwen35_4b_img_only"]
    assert a["s_page"] == pytest.approx(m["prefill_s"] + 386.2 / m["decode_tok_s"])
    assert a["s_page_ub"] > a["s_page"]  # p99 tokens > mean tokens
    assert a["hours"] == pytest.approx(a["s_page"] * 55 / 3600)


def test_order_file_is_the_six_compact_configs_and_all_exist() -> None:
    assert sorted(ORDER) == sorted(SIX)
    for c in ORDER:
        assert (ROOT / "configs" / f"spike_{c}.yaml").is_file()
        assert c.removesuffix("_compact") in SPEED["models"]


def test_order_by_estimate_is_cheapest_first_and_follows_the_speed_file() -> None:
    est = ge.order_by_estimate(SIX, SPEED)
    s = [ge.estimate_config(c, SPEED, 1)["s_page"] for c in est]
    assert s == sorted(s)
    slow = json.loads(json.dumps(SPEED))
    slow["models"]["qwen35_4b_img_only"]["decode_tok_s"] = 0.5
    assert ge.order_by_estimate(SIX, slow)[-1] == "qwen35_4b_img_only_compact"


def test_dev100_cost_pairs_and_report_omits_cu_without_rate() -> None:
    rows = [ge.estimate_config(c, SPEED, 55) for c in SIX]
    d = ge.dev100_cost(rows, 135)
    ss = sorted(r["s_page"] for r in rows)
    assert d["hours_min"] == pytest.approx(sum(ss[:2]) * 135 / 3600)
    assert d["hours_max"] == pytest.approx(sum(ss[-2:]) * 135 / 3600)
    text = ge.report(ORDER, SPEED, 55, 135, 40)
    assert "GPU-hour estimate" in text and "TOTAL spike40" in text
    assert "PLACEHOLDER" not in text  # the shipped speed file is trace-measured
    stale = {
        **SPEED,
        "models": {k: {**v, "source": "placeholder"} for k, v in SPEED["models"].items()},
    }
    assert "PLACEHOLDER" in ge.report(ORDER, stale, 55, 135)
    no_rate = {**SPEED, "t4_cu_per_hour": None}
    assert "compute units)" not in ge.report(ORDER, no_rate, 55, 135)  # never invent a rate
    assert "compute units)" in text  # the shipped speed file carries the T4 rate
    with_rate = ge.report(ORDER, {**SPEED, "t4_cu_per_hour": 2.0}, 55, 135)
    assert "compute units)" in with_rate


def test_count_pages_from_labels(tmp_path: Path) -> None:
    for d, n in (("d1", 1), ("d2", 3)):
        (tmp_path / f"{d}.json").write_text(json.dumps({"pages": [{}] * n}))
    assert ge.count_pages(["d1", "d2"], tmp_path) == 4


def test_order_file_is_cheapest_first_by_measured_speeds() -> None:
    assert ge.order_by_estimate(ORDER, SPEED) == ORDER  # regenerate the order file if this fails


def test_speed_file_cu_rates_and_load_constant() -> None:
    assert SPEED["t4_cu_per_hour"] == 1.19 and SPEED["t4_cu_per_hour_conservative"] == 1.58
    assert "not officially published by Google" in SPEED["t4_cu_source"]
    assert SPEED["model_load_s"] == 120 and "ESTIMATE" in SPEED["model_load_s_label"]


def test_smoke_cost_is_three_img_only_models_plus_loads() -> None:
    cfgs = ge.smoke_configs(ORDER, SPEED)
    assert sorted(cfgs) == sorted(c for c in SIX if "_img_only_" in c)
    c = ge.smoke_cost(SPEED, ORDER, 6)
    pages_h = sum(ge.estimate_config(x, SPEED, 6)["hours"] for x in cfgs)
    assert c["models"] == 3 and c["hours"] == pytest.approx(pages_h + 3 * 120 / 3600)
    assert c["hours_ub"] > c["hours"]
    no_load = ge.smoke_cost({**SPEED, "model_load_s": 0}, ORDER, 6)
    assert c["hours"] - no_load["hours"] == pytest.approx(3 * 120 / 3600)


def test_stage_table_totals_and_cu_at_both_rates() -> None:
    rows = ge.stage_table(ORDER, SPEED, 55, 135, 6)
    by = {r.split("  ")[0].strip(): r for r in rows[1:]}
    assert len(rows) == 9 and "CU@1.19" in rows[0] and "CU@1.58" in rows[0]
    sm = ge.smoke_cost(SPEED, ORDER, 6)["hours"]
    spike = sum(ge.estimate_config(c, SPEED, 55)["hours"] for c in ORDER) + 6 * 120 / 3600
    ab = ge.ab_cost(ORDER, SPEED, 55)["hours"]
    tot = next(r for k, r in by.items() if k.startswith("TOTAL without"))
    cols = tot.split()
    assert float(cols[-5]) == pytest.approx(sm + spike + ab, abs=0.006)  # T4 h
    assert float(cols[-4]) == pytest.approx((sm + spike + ab) * 1.19, abs=0.06)
    assert float(cols[-3]) == pytest.approx((sm + spike + ab) * 1.58, abs=0.06)


def test_ab_stage_is_one_keyed_run_of_the_costliest_config_plus_a_load() -> None:
    ab = ge.ab_cost(ORDER, SPEED, 55)
    keyed = [ge.estimate_keyed(c, SPEED, 55) for c in ORDER]
    assert ab["hours"] == pytest.approx(max(k["hours"] for k in keyed) + 120 / 3600)
    assert ab["hours_ub"] == pytest.approx(max(k["hours_ub"] for k in keyed) + 120 / 3600)
    m = SPEED["models"]["qwen3vl_8b_img_ocr"]  # the costliest measured config at keyed length
    n = SPEED["n_out_keyed"]["qwen3vl_8b"]
    assert ab["hours"] == pytest.approx(
        (m["prefill_s"] + n["mean"] / m["decode_tok_s"]) * 55 / 3600 + 120 / 3600
    )
    assert ab["hours_ub"] > ab["hours"]  # p99 > mean


def test_keyed_n_out_is_json_gold_tokens_from_token_budget() -> None:
    assert SPEED["n_out_keyed"]["qwen35_4b"]["mean"] == 531.3
    assert SPEED["n_out_keyed"]["qwen35_4b"]["upper_bound"] == 1190
    for m, c in SPEED["n_out_keyed"].items():
        assert c["mean"] > SPEED["n_out_compact"][m]["mean"]  # keyed is longer than compact
    k = ge.estimate_keyed("qwen35_4b_img_only_compact", SPEED, 1)
    c = ge.estimate_config("qwen35_4b_img_only_compact", SPEED, 1)
    assert k["s_page"] > c["s_page"]


def test_dev100_both_ways_keyed_costs_more_than_compact() -> None:
    rows = ge.stage_table(ORDER, SPEED, 55, 135, 6)
    by = {r[:42].strip(): r for r in rows[1:]}
    comp = float(by["dev100 top-2, compact speed (cond.)"].split()[-5])
    keyed = float(by["dev100 top-2, keyed speed (cond., worst)"].split()[-5])
    assert keyed > comp
    k = ge.dev100_keyed_cost(ORDER, SPEED, 135)
    s = sorted(ge.estimate_keyed(c, SPEED, 135)["hours"] for c in ORDER)
    assert k["hours"] == pytest.approx(sum(s[-2:]) + 2 * 120 / 3600)


def test_report_with_smoke_warns_when_gpu_is_not_a_t4() -> None:
    t4 = ge.report(ORDER, SPEED, 55, 135, 40, 6, "Tesla T4")
    other = ge.report(ORDER, SPEED, 55, 135, 40, 6, "NVIDIA L4")
    unread = ge.report(ORDER, SPEED, 55, 135, 40, 6, None)
    assert "WARNING" not in t4 and "ESTIMATED COMPUTE-UNIT BURN" in t4
    assert "not a T4" in other and "'NVIDIA L4'" in other
    assert "could not be read" in unread
    assert "Manage sessions" in t4 and "ESTIMATE" in t4  # no invented balance API
    assert "ESTIMATED COMPUTE-UNIT BURN" not in ge.report(ORDER, SPEED, 55, 135)


# ---------------------------------------------------------------------- batch-size scenarios

CFG = "qwen35_4b_img_only"
M = SPEED["models"][CFG]
N_OUT = M["n_output_tokens_mean"]


def test_batch_one_is_exactly_the_unbatched_formula() -> None:
    assert ge.batch_s_per_page(M["prefill_s"], M["decode_tok_s"], N_OUT, 1, 1.0) == pytest.approx(
        ge.est_s_per_page(M["prefill_s"], M["decode_tok_s"], N_OUT)
    )


def test_batched_page_time_formula_and_monotonicity() -> None:
    p, d = M["prefill_s"], M["decode_tok_s"]
    got = ge.batch_s_per_page(p, d, N_OUT, 4, 0.75)
    assert got == pytest.approx(p + N_OUT * ge.STRAGGLER[4] / (d * 0.75 * 4))
    for b in (2, 4, 8):  # a better efficiency is never slower
        t = [ge.batch_s_per_page(p, d, N_OUT, b, e) for e in (0.5, 0.75, 1.0)]
        assert t[0] > t[1] > t[2]
    # at ideal scaling a bigger batch is faster per page, up to the straggler waste
    ideal = [ge.batch_s_per_page(p, d, N_OUT, b, 1.0) for b in (1, 2, 4, 8)]
    assert ideal == sorted(ideal, reverse=True)
    # half the ideal gain at batch 2 is no gain at all (and the straggler waste makes it worse)
    assert ge.batch_s_per_page(p, d, N_OUT, 2, 0.5) > ideal[0]
    for bad in (0.0, -1.0, 1.5):
        with pytest.raises(ValueError, match="eff"):
            ge.batch_s_per_page(p, d, N_OUT, 4, bad)
    with pytest.raises(ValueError, match="straggler"):
        ge.batch_s_per_page(p, d, N_OUT, 16, 1.0)


def test_scenarios_are_batch_one_then_each_batch_at_three_efficiencies() -> None:
    sc = ge.batch_scenarios()
    assert sc[0] == (1, 1.0) and len(sc) == 1 + 3 * 3
    assert {b for b, _ in sc} == {1, 2, 4, 8} and {e for b, e in sc if b > 1} == {1.0, 0.75, 0.5}


def test_tab_hours_adds_loads_smoke_and_bench() -> None:
    load = SPEED["model_load_s"]

    def s(b: int, eff: float) -> float:
        return ge.batch_s_per_page(M["prefill_s"], M["decode_tok_s"], N_OUT, b, eff)

    plain = ge.tab_hours(SPEED, CFG, N_OUT, 4, 0.75, 100)
    assert plain == pytest.approx((load + 100 * s(4, 0.75)) / 3600)
    full = ge.tab_hours(SPEED, CFG, N_OUT, 4, 0.75, 100, smoke_pages=6, with_bench=True)
    bench = load + s(1, 0.75) + 12 * sum(s(b, 0.75) for b in (1, 2, 4, 8))
    assert full == pytest.approx(plain + (load + 6 * s(1, 0.75) + bench) / 3600)


def _rows(text: str, start: str) -> list[list[str]]:
    """Table rows after the title line that starts with `start`, as split columns."""
    lines = text.splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith(start))
    out = []
    for ln in lines[i + 2 :]:
        if not ln.strip():
            break
        out.append(ln.split())
    return out


def test_batch_report_has_both_datasets_both_rates_and_k2_rows() -> None:
    out = ge.zeroshot_batch_report(SPEED, CFG, 671, 6, "Tesla T4", docs_full=500)
    assert "ESTIMATE (UNVERIFIED)" in out and "ASSUMED" in out
    assert "CU@1.19" in out and "CU@1.58" in out and "not a T4" not in out
    assert "500 docs / 671 pages" in out and "test 200 docs / 280 pages" in out
    full = _rows(out, "500 docs / 671 pages")
    test = _rows(out, "test 200 docs")
    assert len(full) == len(test) == 2 * len(ge.batch_scenarios())  # K = 1 and K = 2 rows each
    for rows in (full, test):
        by = {(" ".join(r[:-6]), r[-6]): r for r in rows}  # (scenario, K) -> columns
        assert len(by) == len(rows)
        for name in {n for n, _ in by}:
            k1, k2 = by[(name, "1")], by[(name, "2")]
            wall1, wall2 = float(k1[-4]), float(k2[-4])
            cu_h1, cu_h2 = float(k1[-3]), float(k2[-3])
            assert wall2 < wall1  # two tabs in parallel finish sooner...
            assert cu_h2 >= cu_h1 - 1e-9  # ...but burn at least as many compute units
            assert float(k1[-2]) == pytest.approx(cu_h1 * 1.19, abs=0.06)
            assert float(k1[-1]) == pytest.approx(cu_h1 * 1.58, abs=0.06)
    one = {" ".join(r[:-6]): r for r in full if r[-6] == "1"}
    assert float(one["B=1 (no bench)"][-4]) > float(one["B=8 @ 100% scaling"][-4])
    # the 500-doc rows with batching include the bench, so 4 @ 50% beats 1 only if scaling helps
    assert float(one["B=2 @ 50% scaling"][-4]) > float(one["B=2 @ 100% scaling"][-4])
    assert "not a T4" in ge.zeroshot_batch_report(SPEED, CFG, 671, 6, "NVIDIA L4")


def test_batch_one_row_matches_the_zeroshot_report_total() -> None:
    old = ge.zeroshot_report(SPEED, CFG, 671, 6, "Tesla T4")
    row = next(ln for ln in old.splitlines() if ln.startswith("observed spike40 mean")).split()
    new = ge.zeroshot_batch_report(SPEED, CFG, 671, 6, "Tesla T4")
    b1 = next(
        r for r in _rows(new, "671 pages") if r[:3] == ["B=1", "(no", "bench)"] and r[-6] == "1"
    )
    assert float(b1[-4]) == pytest.approx(float(row[-3]), abs=0.011)  # same total hours


def test_main_prints_the_batch_table_when_the_data_is_present(
    capsys: pytest.CaptureFixture[str],
) -> None:
    if not all((ROOT / "data" / s / "labels").is_dir() for s in ("train", "dev")):
        pytest.skip("data/{train,dev}/labels absent (gitignored)")
    assert ge.main(["--gpu-name", "Tesla T4"]) == 0
    out = capsys.readouterr().out
    assert "500 docs / 671 pages" in out and "ESTIMATE (UNVERIFIED)" in out
    capsys.readouterr()
    assert ge.main(["--zeroshot-docs", ""]) == 0
    assert "ESTIMATE (UNVERIFIED)" not in capsys.readouterr().out
