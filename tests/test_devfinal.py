"""shipdoc.devfinal (notebook 05b): the final adapter on the dev documents, on mock backends.

Everything runs on the SYNTHETIC corpus of tests/_synth.py (11 "dev" documents, 2 invented "train"
label files): a fake final-adapter folder, the mock merged backend, the real run_spike / guard /
scorer / bootstrap / production rule path. No GPU, no peft, no real document, no network. The merge
itself (peft on the real model) is UNVERIFIED and is not exercised here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from test_oof import DROP, MergedMock, World
from test_predict import DEV_IDS, _clean_sha  # noqa: F401 - autouse fixture, must be visible here

from shipdoc import devfinal, oof, predict_ft, rules, spike
from shipdoc import ocr as ocr_mod
from shipdoc import trainset as ts

TRAIN_IDS = ["train_0000", "train_0001"]
ZS500 = [*TRAIN_IDS, *DEV_IDS]
FOLDS_F = {
    "k": 3,
    "folds": [
        {"fold": 0, "val_doc_ids": [TRAIN_IDS[0], *DEV_IDS[:4]]},
        {"fold": 1, "val_doc_ids": [TRAIN_IDS[1], *DEV_IDS[4:8]]},
        {"fold": 2, "val_doc_ids": DEV_IDS[8:]},
    ],
}
SPLIT = ts.stage_split("final", FOLDS_F)
SHA40 = "b" * 40
RUN = "devfinal_ccccccc"


@pytest.fixture(autouse=True)
def _eleven_dev_docs(monkeypatch: pytest.MonkeyPatch) -> None:
    """The synthetic corpus has 11 dev documents, not the real 100."""
    monkeypatch.setattr(devfinal, "EXPECTED_DEV_DOCS", len(DEV_IDS))


def make_final_adapter(final: Path, cfg: spike.SpikeConfig, **override: Any) -> Path:
    """A fake ``final/`` folder: dummy weight files + the manifest ``train.py`` writes."""
    (final / "peft").mkdir(parents=True, exist_ok=True)
    (final / "adapter.pt").write_bytes(b"fake adapter state")
    (final / "peft" / "adapter_model.safetensors").write_bytes(b"fake safetensors")
    (final / "peft" / "adapter_config.json").write_text("{}")
    ids = sorted(SPLIT.train_ids)
    m: dict[str, Any] = {
        "stage": "final",
        "fold": None,
        "manifest_hash": hashlib.sha256(("final|" + ",".join(ids)).encode()).hexdigest()[:16],
        "code_sha": SHA40,
        "train_doc_ids": ids,
        "n_train_docs": len(ids),
        "heldout_doc_ids": sorted(SPLIT.heldout_ids),
        "lora": {"r": 16, "alpha": 32, "n_modules": 200, "n_trainable": 30_474_240},
        "inference_keys": oof.inference_keys_of_config(cfg),
        "base_repo": cfg.backend.repo,
        "base_revision": cfg.backend.revision,
        "precision": "bf16",
        "adapter_sha256": hashlib.sha256(b"fake adapter state").hexdigest(),
        "peft_sha256": {
            "adapter_model.safetensors": hashlib.sha256(b"fake safetensors").hexdigest()
        },
    }
    for k, v in override.items():
        if v is DROP:
            m.pop(k, None)
        else:
            m[k] = v
    (final / "manifest.json").write_text(json.dumps(m, indent=1))
    return final


class Dev:
    """The world of one test: corpus, train labels, split files, a zero-shot run, an OCR cache."""

    def __init__(self, tmp: Path) -> None:
        self.w = World(tmp)
        self.tmp = tmp
        labels = self.w.data / "train" / "labels"
        labels.mkdir(parents=True)
        for i, tid in enumerate(TRAIN_IDS):  # invented train gold: dev gold under a train id
            g = {**self.w.gold[DEV_IDS[i * 3]], "doc_id": tid}
            (labels / f"{tid}.json").write_text(json.dumps(g))
        self.dev_docs = tmp / "dev.json"
        self.zs500 = tmp / "zs500.json"
        self.dev_docs.write_text(json.dumps(DEV_IDS))
        self.zs500.write_text(json.dumps(ZS500))
        self.ocr = tmp / "ocr"
        self.write_ocr()
        self.messages: list[str] = []

    def write_ocr(self, drop: set[str] | None = None) -> None:
        for d in DEV_IDS:
            n_pages = len(self.w.gold[d]["pages"])
            for p in range(1, n_pages + 1):
                stem = f"{d}_p{p}"
                if stem in (drop or set()):
                    continue
                text = "MAWB 176-12345678" if stem == "dev_0002_p1" else f"page {stem}"
                page = ocr_mod.PageOcr(
                    f"{stem}.png", 10, 5, "paddleocr", "line", 0,
                    [ocr_mod.OcrItem(text, [[0, 0], [10, 0], [10, 5], [0, 5]], 0.9)],
                    seconds=1.0,
                )  # fmt: skip
                path = ocr_mod.cache_path(self.ocr, "paddleocr", "dev", stem)
                ocr_mod.write_json_atomic(path, page.to_dict())

    def out(self, msg: str) -> None:
        self.messages.append(msg)

    @property
    def run_dir(self) -> Path:
        return self.w.runs / RUN

    def adapter(self, name: str = "ad", **override: Any) -> Path:
        return make_final_adapter(self.tmp / name / "final", self.w.cfg, **override)

    def infer(self, adapter: Path, backend: Any, **kw: Any) -> dict[str, Any]:
        return devfinal.run_infer(
            cfg=self.w.cfg, adapter_dir=adapter, zs_run_dir=self.w.runs / "zs", folds=FOLDS_F,
            dev_ids=kw.pop("dev_ids", DEV_IDS), out_dir=self.run_dir, data_root=self.w.data,
            backend_factory=lambda d, s: backend, bench_docs=DEV_IDS[:3], pin_sha=SHA40,
            reachable=lambda sha: True, out=self.out, **kw,
        )  # fmt: skip

    def compare(self, **kw: Any) -> dict[str, Any]:
        args: dict[str, Any] = {
            "zs_run_dir": self.w.runs / "zs",
            "ft_run_dir": self.run_dir,
            "out_dir": self.tmp / "cmp",
            "dev_docs": self.dev_docs,
            "zs500_docs": self.zs500,
            "folds": FOLDS_F,
            "data_root": self.w.data,
            "ocr_root": self.ocr,
            "n_boot": 200,
            "out": self.out,
        }
        return devfinal.run_compare(**{**args, **kw})


@pytest.fixture()
def d(tmp_path: Path) -> Dev:
    dev = Dev(tmp_path)
    dev.w.zero_shot(null_header=("buyer_name",))
    return dev


def finished(d: Dev) -> Dev:
    d.infer(d.adapter(), MergedMock(d.w.gold, d.w.cfg))
    return d


# --------------------------------------------------------------------------------------------
# The dev ids
# --------------------------------------------------------------------------------------------


def test_the_dev_ids_are_asserted_against_every_source() -> None:
    assert devfinal.check_dev_ids(list(reversed(DEV_IDS)), ZS500, FOLDS_F, DEV_IDS) == DEV_IDS


@pytest.mark.parametrize(
    ("dev", "zs500", "labels", "message"),
    [
        (DEV_IDS[:-1], ZS500, None, "expected 11"),
        ([*DEV_IDS[:-1], DEV_IDS[0]], ZS500, None, "duplicate"),
        ([*DEV_IDS[:-1], "train_0000"], ZS500, None, "not dev_"),
        (DEV_IDS, [*TRAIN_IDS, *DEV_IDS[:-1]], None, "dev ids of splits/zeroshot500"),
        (DEV_IDS, ZS500, DEV_IDS[:-1], "dev label files"),
    ],
)
def test_wrong_dev_ids_are_refused(dev: list, zs500: list, labels: Any, message: str) -> None:
    with pytest.raises(devfinal.DevFinalError, match=message):
        devfinal.check_dev_ids(dev, zs500, FOLDS_F, labels)


def test_ids_that_are_not_the_final_stage_heldout_are_refused() -> None:
    other = {"k": 2, "folds": [{"fold": 0, "val_doc_ids": [*TRAIN_IDS, *DEV_IDS[:-1], "dev_9999"]}]}
    with pytest.raises(devfinal.DevFinalError, match="held-out ids of the final stage"):
        devfinal.check_dev_ids(DEV_IDS, ZS500, other)


def test_plan_counts_from_the_labels_and_prints_no_id(d: Dev, tmp_path: Path) -> None:
    plan = devfinal.run_plan(
        dev_docs=d.dev_docs, zs500_docs=d.zs500, folds=FOLDS_F, data_root=d.w.data,
        out_path=tmp_path / "plan.json", out=d.out,
    )  # fmt: skip
    want_pages = sum(len(g["pages"]) for g in d.w.gold.values())
    assert (plan["n_docs"], plan["n_pages"]) == (11, want_pages)
    assert json.loads((tmp_path / "plan.json").read_text())["ids_sha256"] == plan["ids_sha256"]
    assert "dev_0000" not in "\n".join(d.messages)
    assert f"{want_pages} pages" in "\n".join(d.messages)


# --------------------------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------------------------


def test_infer_runs_verify_merge_guard_infer_in_order_on_exactly_the_dev_ids(d: Dev) -> None:
    be = MergedMock(d.w.gold, d.w.cfg)
    rec = d.infer(d.adapter(), be)
    text = "\n".join(d.messages)
    marks = ["== VERIFY ==", "== MERGE ==", "== GUARD ==", "== INFER =="]
    pos = [text.index(m) for m in marks]
    assert pos == sorted(pos) and be.loads == 1
    preds = json.loads((d.run_dir / "predictions.json").read_text())
    assert sorted(preds) == DEV_IDS
    man = json.loads((d.run_dir / "manifest.json").read_text())
    assert man["devfinal"]["ids_sha256"] == devfinal.ids_sha256(DEV_IDS)
    assert man["devfinal"]["merge"]["n_lora_modules_merged"] == 200
    assert rec["batch"]["used"] == 2 and rec["guard"]["ran"] and rec["guard"]["ok"]
    assert (d.run_dir / "batch_decision.json").is_file() and (d.run_dir / "guard").is_dir()
    assert "FINAL ADAPTER MANIFEST VERIFICATION: PASSED" in text and "dev_0000" not in text


def test_a_resumed_run_keeps_the_stored_batch_decision_and_redoes_nothing(d: Dev) -> None:
    first = MergedMock(d.w.gold, d.w.cfg, die_after=30)
    with pytest.raises(KeyboardInterrupt):
        d.infer(d.adapter(), first)
    assert (d.run_dir / "trace.jsonl").is_file()
    done = len((d.run_dir / "trace.jsonl").read_text().splitlines())
    assert 0 < done < len(DEV_IDS)
    second = MergedMock(d.w.gold, d.w.cfg)
    d.messages.clear()
    d.infer(d.adapter(), second)
    assert "resumed run: the stored batch decision stands" in "\n".join(d.messages)
    assert second._calls < sum(len(g["pages"]) for g in d.w.gold.values())  # not everything again
    assert sorted(json.loads((d.run_dir / "predictions.json").read_text())) == DEV_IDS


def test_a_guard_difference_falls_back_to_batch_1_and_says_so(d: Dev) -> None:
    rec = d.infer(d.adapter(), MergedMock(d.w.gold, d.w.cfg, flip_in_batches=True))
    assert rec["guard"]["fallback_to_1"] and rec["batch"]["used"] == 1
    assert "FALLING BACK TO BATCH 1" in "\n".join(d.messages)


@pytest.mark.parametrize(
    ("override", "failed"),
    [
        ({"stage": "smoke"}, "stage"),
        ({"stage": "fold0", "fold": 0}, "fold id"),
        ({"train_doc_ids": [*sorted(SPLIT.train_ids), "dev_0000"]}, "no dev document in training"),
        ({"train_doc_ids": [*sorted(SPLIT.train_ids), "test_0001"]}, "no test document"),
        ({"heldout_doc_ids": ["dev_0000"]}, "held-out ids"),
        ({"code_sha": "abc"}, "training code sha"),
        ({"train_doc_ids": DROP}, "training doc ids"),
    ],
)
def test_adapters_that_are_not_the_final_adapter_are_refused_before_any_merge(
    d: Dev, override: dict[str, Any], failed: str
) -> None:
    be = MergedMock(d.w.gold, d.w.cfg)
    with pytest.raises(predict_ft.FtError, match=failed):
        d.infer(d.adapter(**override), be)
    assert be.loads == 0 and be._calls == 0 and not (d.run_dir / "trace.jsonl").exists()
    assert "== MERGE ==" not in "\n".join(d.messages)


def test_a_short_merge_is_refused_before_the_guard(d: Dev) -> None:
    be = MergedMock(d.w.gold, d.w.cfg, n_merged=196)
    with pytest.raises(predict_ft.FtError, match="merge did not merge 200"):
        d.infer(d.adapter(), be)
    assert "== GUARD ==" not in "\n".join(d.messages)


def test_infer_refuses_documents_that_are_not_the_dev_documents(d: Dev) -> None:
    be = MergedMock(d.w.gold, d.w.cfg)
    with pytest.raises(devfinal.DevFinalError, match="not the dev documents"):
        d.infer(d.adapter(), be, dev_ids=DEV_IDS[:-1])
    with pytest.raises(devfinal.DevFinalError, match="not the dev documents"):
        d.infer(d.adapter(), be, dev_ids=[*DEV_IDS[:-1], "train_0000"])
    assert be.loads == 0


def test_the_batch_size_defaults_to_the_02_run_and_a_manual_one_is_warned(d: Dev) -> None:
    rec = d.infer(d.adapter(), MergedMock(d.w.gold, d.w.cfg))
    assert rec["batch"]["source"] == "zero_shot_run" and rec["batch"]["batch_size"] == 2
    d2 = Dev(d.tmp / "second")
    d2.w.zero_shot(batch=2)
    d2.infer(d2.adapter(), MergedMock(d2.w.gold, d2.w.cfg), batch_size=1)
    assert any("differs from the zero-shot run's" in m for m in d2.messages)


# --------------------------------------------------------------------------------------------
# Compare
# --------------------------------------------------------------------------------------------


def test_compare_scores_ft_vs_zs_on_the_dev_docs_under_three_arms(d: Dev) -> None:
    finished(d)
    res = d.compare()
    assert res["kind"] == "dev_final" and res["n_docs"] == 11 and res["ft_run"] == RUN
    assert set(res["arms"]) == {"rules_train_shapes", "rules_frozen_shapes", "raw"}
    for arm in res["arms"].values():
        sub = arm["subsets"]
        assert sub["all"]["n_docs"] == 11 and {"invoices", "waybills"} <= set(sub)
        assert sub["invoices"]["n_docs"] + sub["waybills"]["n_docs"] == 11
    raw = res["arms"]["raw"]["subsets"]["all"]
    # the mock FT replays the gold; the mock ZS nulls buyer_name on every invoice: FT is better
    assert raw["ft"]["OVERALL"] == 1.0 and raw["zs"]["OVERALL"] < 1.0
    a = res["arms"]["rules_train_shapes"]["subsets"]["all"]
    assert a["ft"]["OVERALL"] > a["zs"]["OVERALL"]
    delta = a["paired_delta_ft_minus_zs"]["OVERALL"]
    assert delta["delta"] > 0 and delta["lo"] > 0
    assert a["zs"]["over_null"]["header_over_null"] > 0 == a["ft"]["over_null"]["over_null_total"]
    assert a["zs"]["over_null"]["header_over_null_by_field"]["buyer_name"] == 9  # 9 invoices
    off = res["official_dev"]["raw"]
    assert off["ft_OVERALL"]["point"] == 1.0 and set(off["ft_OVERALL"]) == {"point", "lo", "hi"}
    assert (res["n_boot"], res["seed"]) == (200, 42)
    assert res["shapes"]["train"]["learned_from"] == "2 train gold documents"
    assert (d.tmp / "cmp" / "devfinal_compare.json").is_file()
    assert (d.tmp / "cmp" / "devfinal_compare.md").is_file()


def test_the_rules_run_on_the_dev_docs_and_both_arms_get_the_same_ones(d: Dev) -> None:
    finished(d)
    res = d.compare()
    for arm in ("rules_train_shapes", "rules_frozen_shapes"):
        r = res["arms"][arm]["rules"]
        assert r["zs"]["switches"] == {"r1": True, "r2": True, "r3": True}
        assert r["zs"]["eligible_docs"] == r["ft"]["eligible_docs"]
        assert r["zs"]["eligible_docs"]["R2"] == 2 and not r["zs"]["skipped"]  # 2 dev waybills
    assert res["arms"]["raw"]["rules"]["ft"]["switches"] == {"r1": False, "r2": False, "r3": False}


def test_reports_hold_aggregates_only(d: Dev) -> None:
    finished(d)
    d.compare()
    text = "\n".join(d.messages) + (d.tmp / "cmp" / "devfinal_compare.md").read_text()
    text += (d.tmp / "cmp" / "devfinal_compare.json").read_text()
    for gold in d.w.gold.values():
        assert gold["doc_id"] not in text
        for v in gold["header"].values():
            if len(str(v)) >= 8:  # the invented values are distinctive; short ones collide with ids
                assert str(v) not in text
    assert "seen layouts" in text and "NOT A RESULT" not in text
    assert "R3 shapes: train-only" in text and "IN-SAMPLE" in text


def test_compare_refuses_an_unfinished_or_foreign_or_wrong_run(d: Dev, tmp_path: Path) -> None:
    be = MergedMock(d.w.gold, d.w.cfg, die_after=30)
    with pytest.raises(KeyboardInterrupt):
        d.infer(d.adapter(), be)
    with pytest.raises(devfinal.DevFinalError, match="not complete"):
        d.compare()
    d.infer(d.adapter(), MergedMock(d.w.gold, d.w.cfg))
    man = json.loads((d.run_dir / "manifest.json").read_text())
    man["devfinal"]["ids_sha256"] = "0" * 64
    (d.run_dir / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(devfinal.DevFinalError, match="dev ids differ"):
        d.compare()
    del man["devfinal"]
    (d.run_dir / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(devfinal.DevFinalError, match="not a 05b run"):
        d.compare()
    with pytest.raises(devfinal.DevFinalError, match="--ft-run-dir"):
        d.compare(ft_run_dir=None)


def test_compare_refuses_predictions_outside_the_dev_ids(d: Dev) -> None:
    finished(d)
    preds = json.loads((d.run_dir / "predictions.json").read_text())
    preds["train_0000"] = preds.pop(DEV_IDS[0])
    (d.run_dir / "predictions.json").write_text(json.dumps(preds))
    with pytest.raises(devfinal.DevFinalError, match="not exactly the 11 dev ids"):
        d.compare()


def test_compare_stops_when_the_dev_ocr_is_missing_for_a_waybill(d: Dev) -> None:
    finished(d)
    d.ocr.joinpath("paddleocr", "dev", "dev_0008_p1.json").unlink()
    with pytest.raises(devfinal.DevFinalError, match="R2 was skipped"):
        d.compare()


def test_compare_refuses_a_dev_document_in_the_shape_learner_input(d: Dev) -> None:
    finished(d)
    src = d.w.data / "train" / "labels" / "train_0000.json"
    src.rename(src.with_name("dev_0000.json"))
    g = json.loads(src.with_name("dev_0000.json").read_text())
    g["doc_id"] = "dev_0000"
    src.with_name("dev_0000.json").write_text(json.dumps(g))
    with pytest.raises(devfinal.DevFinalError, match="non-train document"):
        d.compare()


def test_run_arms_fails_closed_when_rules_off_does_not_reproduce_the_predictions(d: Dev) -> None:
    finished(d)
    raw = json.loads((d.run_dir / "predictions.json").read_text())
    raw[DEV_IDS[0]]["header"]["invoice_number"] = "TAMPERED"
    traces = [
        json.loads(ln) for ln in (d.run_dir / "trace.jsonl").read_text().splitlines() if ln.strip()
    ]
    prod = devfinal._replay_module().production
    shapes = rules.SlotShapes(cpn_only=frozenset(), po_only=frozenset())
    with pytest.raises(devfinal.DevFinalError, match="rules OFF does not reproduce"):
        devfinal.run_arms(
            traces, raw, DEV_IDS, ocr_root=d.ocr, train_shapes=shapes, frozen_shapes=shapes,
            production=prod,
        )  # fmt: skip
    with pytest.raises(devfinal.DevFinalError, match="trace covers 10 of the 11"):
        devfinal.run_arms(
            traces[:-1], raw, DEV_IDS, ocr_root=d.ocr, train_shapes=shapes,
            frozen_shapes=shapes, production=prod,
        )  # fmt: skip


# --------------------------------------------------------------------------------------------
# The plumbing check
# --------------------------------------------------------------------------------------------


def test_plumbing_check_uses_the_zs_run_as_both_arms_and_every_delta_is_zero(d: Dev) -> None:
    res = d.compare(ft_run_dir=None, plumbing=True)  # no FT run exists at all
    assert res["kind"] == "plumbing_check" and "NOT A RESULT" in res["label"]
    assert res["ft_run"] is None and res["n_docs"] == 11
    n_cells = sum(len(a["subsets"]) for a in res["arms"].values()) * len(devfinal.METRICS)
    assert res["plumbing"] == {"cells": n_cells, "nonzero": 0, "all_zero": True}
    assert not d.run_dir.exists()
    md = (d.tmp / "cmp" / "devfinal_compare.md").read_text()
    assert "PLUMBING CHECK, NOT A RESULT" in md and "NOT A RESULT" in "\n".join(d.messages)
    a = res["arms"]["raw"]["subsets"]["all"]
    assert a["zs"] == a["ft"] and a["zs"]["OVERALL"] < 1.0


def test_plumbing_cells_counts_a_nonzero_delta() -> None:
    zero = {"delta": 0.0, "lo": 0.0, "hi": 0.0}
    cell = {k: dict(zero) for k in devfinal.METRICS}
    arms = {"a": {"subsets": {"all": {"paired_delta_ft_minus_zs": cell}}}}
    assert devfinal.plumbing_cells(arms) == {"cells": 5, "nonzero": 0}
    cell["row_f1"] = {"delta": 0.0, "lo": -0.01, "hi": 0.0}
    assert devfinal.plumbing_cells(arms) == {"cells": 5, "nonzero": 1}


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def test_cli_stages_and_flags() -> None:
    p = devfinal.build_parser()
    a = p.parse_args(["compare", "--zs-run-dir", "z", "--out-dir", "o", "--plumbing-check"])
    assert a.plumbing_check and a.ft_run_dir is None and a.n_boot == 2000
    a = p.parse_args(["infer", "--zs-run-dir", "z", "--out-dir", "o", "--adapter-dir", "a"])
    assert a.batch_size is None and a.bench_docs == "splits/bench12.json" and not a.wandb
    assert p.parse_args(["plan"]).stage == "plan"
    with pytest.raises(SystemExit):
        p.parse_args(["infer", "--zs-run-dir", "z", "--out-dir", "o"])  # no adapter


def test_main_prints_a_refusal_and_exits_1(d: Dev, capsys: pytest.CaptureFixture[str]) -> None:
    bad = d.tmp / "bad.json"
    bad.write_text(json.dumps(DEV_IDS[:-1]))
    folds = d.tmp / "folds.json"
    folds.write_text(json.dumps(FOLDS_F))
    rc = devfinal.main(
        [
            "plan",
            "--dev-docs",
            str(bad),
            "--zs500-docs",
            str(d.zs500),
            "--folds",
            str(folds),
            "--data-root",
            str(d.w.data),
        ]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "devfinal plan: REFUSED" in err and "expected 11" in err and "dev_0000" not in err


def test_banner_names_the_seen_layout_caveat(d: Dev) -> None:
    finished(d)
    res = d.compare()
    banner = devfinal.format_banner(res)
    assert "seen layouts" in banner and "does NOT measure unseen suppliers" in banner
    assert "rules_train_shapes" in banner and "false fills ZS" in banner
    plumb = d.compare(ft_run_dir=None, plumbing=True)
    assert "PLUMBING:" in devfinal.format_banner(plumb)
