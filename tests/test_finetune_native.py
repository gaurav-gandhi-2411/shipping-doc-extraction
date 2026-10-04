"""Native-resolution (max_pixels 2196480) fine-tune: config diff, token arithmetic, run isolation,
manifest resolution record and the VRAM / hours estimate. CPU only, no model, no network.

The facts under test: the 1260 config is untouched, the native config differs from it in exactly
``max_pixels`` and ``expected_visual_tokens``, the per-sample visual-token check is config driven,
a 1260 checkpoint cannot resume a native run, and the adapter manifest of either carries its own
training ``max_pixels`` (the key notebook 05 compares through ``shipdoc.resmatch``).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml
from test_train import (
    IMG_ID,
    FakeProcessor,
    ListSource,
    build_model,
    hidden_fn,
    make_cfg,
    rendered,
)

from shipdoc import resmatch
from shipdoc import ressweep as rs
from shipdoc import train as tr
from shipdoc import trainset as ts

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
CFG_1260 = ROOT / "configs" / "finetune_qwen35_4b.yaml"
CFG_NATIVE = ROOT / "configs" / "finetune_qwen35_4b_native.yaml"
_spec = importlib.util.spec_from_file_location(
    "finetune_native_estimate", ROOT / "scripts" / "finetune_native_estimate.py"
)
fne = importlib.util.module_from_spec(_spec)
sys.modules["finetune_native_estimate"] = fne  # dataclasses resolve their module via sys.modules
_spec.loader.exec_module(fne)

PAGES = {"val": [230, 217, 224], "train": [441, 454, 447], "all_train": 536, "all": 671}
NATIVE, OLD = 2_196_480, 1_310_720


def raw(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------------


def test_native_config_differs_from_the_1260_config_in_exactly_the_documented_keys() -> None:
    old, new = raw(CFG_1260), raw(CFG_NATIVE)
    assert set(old) == set(new)
    changed = {k for k in old if old[k] != new[k]}
    assert changed == {"max_pixels", "expected_visual_tokens"}
    assert (old["max_pixels"], old["expected_visual_tokens"]) == (OLD, 1260)
    assert (new["max_pixels"], new["expected_visual_tokens"]) == (NATIVE, 2145)
    # not changed on purpose: effective batch, loss chunk, every smoke threshold
    for k in ("grad_accum", "loss_chunk", "epochs", "seed", "smoke", "optim", "lora"):
        assert new[k] == old[k], k
    assert new["smoke"]["vram_budget_gib"] == {"bf16": 20.0, "fp16": 13.5, "fp32": 20.0}


def test_both_configs_load_and_have_different_run_signatures() -> None:
    old, new = tr.load_config(CFG_1260), tr.load_config(CFG_NATIVE)
    assert (old.max_pixels, old.expected_visual_tokens) == (OLD, 1260)
    assert (new.max_pixels, new.expected_visual_tokens) == (NATIVE, 2145)
    assert old.signature() != new.signature()
    # the library defaults (what every existing test and the 1260 run use) are unchanged
    default = tr.TrainConfig()
    assert (default.max_pixels, default.expected_visual_tokens) == (OLD, 1260)
    assert tr.DEFAULT_CONFIG == CFG_1260


# --------------------------------------------------------------------------------------------
# Visual-token arithmetic and the per-sample check
# --------------------------------------------------------------------------------------------


def test_native_cap_gives_2145_tokens_on_a_1240x1754_page_and_1260_for_the_old_cap() -> None:
    assert rs.smart_resize(1754, 1240, NATIVE) == (1760, 1248)
    assert rs.visual_tokens(NATIVE) == 2145 == fne.TOKENS_NATIVE
    assert rs.visual_tokens(OLD) == 1260 == fne.TOKENS_1260
    assert fne.MAX_PIXELS_NATIVE == NATIVE and fne.MAX_PIXELS_1260 == OLD


def test_native_token_count_equals_the_transformers_processor_arithmetic() -> None:
    pytest.importorskip("transformers")
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize as ref

    for mp, tokens in ((OLD, 1260), (NATIVE, 2145)):
        h, w = ref(1754, 1240, factor=32, min_pixels=rs.MIN_PIXELS, max_pixels=mp)
        assert (h // 16) * (w // 16) // 4 == tokens, mp


def test_encode_page_check_is_driven_by_the_config_not_by_1260() -> None:
    new = tr.load_config(CFG_NATIVE)
    n = rs.visual_tokens(new.max_pixels)  # what the processor produces for the native page
    kw = {"adapter_name": "qwen35", "image_token_id": IMG_ID}
    out = tr.encode_page(rendered(), FakeProcessor(n_visual=n), **kw,
                         expected_visual_tokens=new.expected_visual_tokens)  # fmt: skip
    assert int((out["input_ids"][0] == IMG_ID).sum()) == 2145
    # a page that still yields 1260 tokens (cap not applied) is refused under the native config
    with pytest.raises(tr.TrainError, match="1260 visual tokens, expected 2145"):
        tr.encode_page(rendered(), FakeProcessor(n_visual=1260), **kw,
                       expected_visual_tokens=new.expected_visual_tokens)  # fmt: skip
    # and the 1260 config still refuses a native-size page
    old = tr.load_config(CFG_1260)
    with pytest.raises(tr.TrainError, match="2145 visual tokens, expected 1260"):
        tr.encode_page(rendered(), FakeProcessor(n_visual=2145), **kw,
                       expected_visual_tokens=old.expected_visual_tokens)  # fmt: skip


# --------------------------------------------------------------------------------------------
# Resume refusal across resolutions, manifest record
# --------------------------------------------------------------------------------------------


def _signature(cfg: tr.TrainConfig) -> str:
    return f"{cfg.signature()}|{'f' * 16}|bf16"  # as train.main builds it


def _trainer(run_dir: Path, cfg: tr.TrainConfig, **kw) -> tr.Trainer:
    return tr.Trainer(build_model(), ListSource(), cfg, hidden_fn, run_dir,
                      signature=_signature(cfg), **kw)  # fmt: skip


def test_a_1260_checkpoint_cannot_resume_a_native_run(tmp_path: Path) -> None:
    c1260 = make_cfg(max_pixels=OLD, expected_visual_tokens=1260)
    cnat = make_cfg(max_pixels=NATIVE, expected_visual_tokens=2145)
    assert _signature(c1260) != _signature(cnat)  # max_pixels is inside the hashed config
    run = tmp_path / "ft_x"
    _trainer(run, c1260).fit()
    assert (run / "ckpt" / "LATEST").is_file()
    with pytest.raises(tr.TrainError, match="different run"):
        _trainer(run, cnat).fit()
    # the same resolution resumes (finished run: nothing to do, no error)
    again = _trainer(run, c1260)
    again.fit()
    assert again.step == again.total


def test_run_ids_of_the_native_notebook_cannot_collide_with_the_1260_ones() -> None:
    spec = importlib.util.spec_from_file_location(
        "colab_build_finetune_native", ROOT / "scripts" / "colab_build_finetune_native.py"
    )
    nb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nb)
    params = nb.NATIVE_PARAMS
    assert 'RUN_BASE = f"ft_native_{STAGE}_{SHA7}"' in params
    assert 'f"ft_native_smoke_{SHA7}_{PRECISION}"' in nb.NATIVE_INSTALL
    for text in (params, nb.NATIVE_INSTALL):
        assert 'f"ft_{STAGE}' not in text and 'f"ft_smoke_' not in text


@pytest.mark.parametrize(("max_pixels", "tokens"), [(NATIVE, 2145), (OLD, 1260)])
def test_manifest_written_by_save_final_records_the_training_resolution(
    tmp_path: Path, max_pixels: int, tokens: int
) -> None:
    cfg = make_cfg(max_pixels=max_pixels, expected_visual_tokens=tokens)
    split = ts.StageSplit("fold0", ("train_1", "train_2"), ("train_3",))
    meta = tr.build_run_meta("fold0", split, cfg, n_heldout_eval_pages=1, n_lora_modules=6,
                             n_trainable=1, code_sha="a" * 40)  # fmt: skip
    t = _trainer(tmp_path / "run", cfg, run_meta=meta)
    t.fit()
    manifest = json.loads((t.save_final() / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["inference_keys"]["max_pixels"] == max_pixels
    assert manifest["train_config_signature"] == cfg.signature()
    assert resmatch.training_resolution_matches(manifest, {"max_pixels": max_pixels})[0]
    other = OLD if max_pixels == NATIVE else NATIVE
    ok, why = resmatch.training_resolution_matches(manifest, {"max_pixels": other})
    assert not ok and str(max_pixels) in why and str(other) in why


# --------------------------------------------------------------------------------------------
# Estimates
# --------------------------------------------------------------------------------------------


def test_fold0_anchor_is_recomputed_from_the_measured_run() -> None:
    assert pytest.approx(8.87, abs=0.005) == fne.FOLD0_S_PER_PAGE
    assert fne.INPUT_NATIVE == 840 + 2145


def test_time_factors_are_ordered_and_high_is_the_token_ratio() -> None:
    f = fne.time_factors()
    assert 1.0 < f["low"] < f["flop"] < f["high"]
    assert f["high"] == pytest.approx(2145 / 1260)
    assert f["low"] == pytest.approx((2985 + 531.3) / (2100 + 531.3))


def test_stage_hours_scale_with_visits_and_factor() -> None:
    one = fne.stage_estimate("fold0", 441, 1.0, eval_every=0, eval_pages=0)
    assert one.visits == 882 and one.steps == 112
    assert one.hours == pytest.approx((120 + 882 * fne.FOLD0_S_PER_PAGE) / 3600)
    assert fne.stage_estimate("fold0", 441, 1.7, eval_every=0, eval_pages=0).hours > one.hours
    smoke = fne.stage_estimate("smoke", None, 1.0)
    assert (smoke.visits, smoke.steps, smoke.evals) == (40, 20, 0)
    assert fne.stage_estimate("final", 536, 1.0).visits == 1072


def test_totals_parallel_wall_is_the_slowest_stage_and_cu_is_the_sum() -> None:
    t = fne.totals({"smoke": 0.2, "fold0": 3.0, "fold1": 4.0, "fold2": 3.5, "final": 5.0})
    assert t["sequential_h"] == pytest.approx(15.7) and t["cu_hours"] == t["sequential_h"]
    assert t["parallel4_wall_h"] == pytest.approx(5.2)
    assert t["fold0_first_wall_h"] == pytest.approx(0.2 + 3.0 + 5.0)


def test_vram_estimates_are_ordered_anchored_and_inside_the_budget() -> None:
    low, central, high, gross = fne.vram_estimates()
    assert fne.fixed_gib() == pytest.approx(9.62)
    assert fne.FOLD0_PEAK_GIB < low.peak_gib <= central.peak_gib < high.peak_gib < gross.peak_gib
    assert high.peak_gib < 0.85 * fne.BUDGET_GIB["bf16"]  # not within 15% of 20 GiB: no change
    assert gross.peak_gib < fne.BUDGET_GIB["bf16"]
    assert "clear of the 15% band" in fne.vram_verdict(high.peak_gib)
    assert "WITHIN 15%" in fne.vram_verdict(high.peak_gib, "fp16")  # T4 fp16 is at risk
    assert "WITHIN 15%" in fne.vram_verdict(17.5, "bf16")


def test_printed_estimate_is_labelled_and_covers_every_stage() -> None:
    text = "\n".join(fne.native_lines(PAGES))
    assert "ESTIMATE" in text and "UNVERIFIED" in text and "MEASURED at 1260" in text
    for stage in fne.STAGES:
        assert any(line.startswith(stage + " ") for line in text.splitlines()), stage
    for rate in ("1.19", "1.58"):
        assert rate in text
    assert "4 parallel tabs" in text and "fold0 first then 3 tabs" in text
