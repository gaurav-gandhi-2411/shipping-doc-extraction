"""OOF inference (shipdoc.oof): the logic behind notebook 05, end to end on a tiny mocked adapter.

Everything runs on the mock backend over the SYNTHETIC corpus of tests/_synth.py: a fake adapter
folder (manifest + dummy files), a mock "merged" backend whose outputs differ from the mock
zero-shot run, and the real run_spike / bench / scorer / bootstrap code. No GPU, no peft, no real
document. The merge itself (peft on the real model) is UNVERIFIED and only exercised through a
stub `peft` module.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from _synth import make_corpus
from test_predict import (
    DEV_IDS,
    KEYED,
    ROOT,
    _clean_sha,  # noqa: F401 - autouse fixture, must be visible in this module
)

from shipdoc import cli, oof, spike
from shipdoc import eval as ev
from shipdoc import train as tr
from shipdoc import trainset as ts
from shipdoc.extract import MockBackend, MockCorruption, parse_output

SHA40 = "b" * 40
FOLDS = {
    "k": 3,
    "folds": [
        {"fold": 0, "val_doc_ids": DEV_IDS[0:4]},  # includes the waybill dev_0002
        {"fold": 1, "val_doc_ids": DEV_IDS[4:8]},
        {"fold": 2, "val_doc_ids": DEV_IDS[8:]},  # includes the waybill dev_0008
    ],
}
HELD0 = oof.fold_heldout_ids(FOLDS, 0)
STAGE_MARKERS = [
    "== VERIFY ==",
    "== MERGE ==",
    "== GUARD ==",
    "== INFER ==",
    "== SCORE / COMPARE ==",
]
DROP = object()  # sentinel: remove the key from the adapter manifest


# --------------------------------------------------------------------------------------------
# Mock backends and fixtures
# --------------------------------------------------------------------------------------------


class Variant(MockBackend):
    """Gold replay with chosen header / row fields forced to null (over-nulls)."""

    def __init__(
        self,
        gold: dict[str, Any],
        cfg: spike.SpikeConfig,
        *,
        null_header: tuple = (),
        null_row: tuple = (),
        flip_in_batches: bool = False,
    ) -> None:
        super().__init__(
            gold, corruption=MockCorruption(blank_fields=tuple(null_header)), fixed_latency_s=0.5
        )
        self.revision = cfg.backend.revision  # the verification compares it with the config
        self.null_row, self.flip_in_batches = tuple(null_row), flip_in_batches

    def page_payload(self, doc_id: str, page_index: int, n_pages: int) -> dict[str, Any]:
        out = super().page_payload(doc_id, page_index, n_pages)
        for row in out["line_items"]:
            for f in self.null_row:
                row[f] = None
        return out

    def extract_pages(self, requests: Any) -> Any:
        res = super().extract_pages(requests)
        if not (self.flip_in_batches and len(requests) > 1):
            return res
        flipped = []
        for raw, _parsed, meta in res:  # a batched decode that differs from batch 1
            raw2 = raw.replace('"currency": "USD"', '"currency": "EUR"')
            flipped.append((raw2, parse_output(raw2, "json"), meta))
        return flipped


class MergedMock(Variant):
    """The `MergedHfBackend` interface: ``load()`` merges and fills ``merge_info``."""

    def __init__(
        self, *a: Any, n_merged: int = tr.EXPECTED_LORA_MODULES, die_after: int = 0, **kw: Any
    ) -> None:
        super().__init__(*a, **kw)
        self.n_merged, self.die_after = n_merged, die_after
        self.merge_info: dict[str, Any] | None = None
        self.loads = 0

    def load(self) -> None:
        self.loads += 1
        self.merge_info = {
            "n_lora_modules_merged": self.n_merged,
            "n_lora_modules_left": 0,
            "merge_dtype": "torch.float16",
            "merge_s": 0.0,
        }

    def extract_page(self, *a: Any, **kw: Any) -> Any:
        if self.die_after and self._calls >= self.die_after:
            raise KeyboardInterrupt  # a kill: not an Exception, nothing catches it
        return super().extract_page(*a, **kw)


class World:
    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        self.data = tmp / "data"
        self.runs = tmp / "runs"
        self.cfg = spike.load_config(KEYED)
        self.gold = make_corpus(self.data)
        for g in self.gold.values():  # give the rows a customer part number to over-null
            for i, row in enumerate(g["line_items"]):
                row["customer_part_number"] = f"CP-{i}" if i % 2 == 0 else None
            (self.data / "dev" / "labels" / f"{g['doc_id']}.json").write_text(json.dumps(g))
        self.messages: list[str] = []

    def out(self, msg: str) -> None:
        self.messages.append(msg)

    def zero_shot(self, batch: int = 2, **kw: Any) -> Path:
        be = Variant(self.gold, self.cfg, **kw)
        spike.run_spike(
            self.cfg,
            DEV_IDS,
            "dev",
            "zs",
            be,
            runs_root=self.runs,
            data_root=self.data,
            logprobs=True,
            batch_size=batch,
        )
        (self.runs / "zs" / "bench_result.json").write_text(
            json.dumps({"chosen_batch_size": batch})
        )
        return self.runs / "zs"

    def adapter(self, _fold: int = 0, _name: str = "ft", **override: Any) -> Path:
        return make_adapter(self.tmp / _name / "final", _fold, self.cfg, **override)

    def infer(self, adapter: Path, backend: Any, **kw: Any) -> dict[str, Any]:
        kw.setdefault("fold", 0)
        return oof.run_infer(
            adapter_dir=adapter,
            zs_run_dir=self.runs / "zs",
            cfg=self.cfg,
            folds=FOLDS,
            out_dir=self.runs / "oof_fold0_bbbbbbb",
            backend_factory=lambda d, s: backend,
            bench_docs=DEV_IDS[:3],
            data_root=self.data,
            split="dev",
            pin_sha=SHA40,
            reachable=lambda sha: True,
            out=self.out,
            **kw,
        )

    def compare(self, **kw: Any) -> dict[str, Any]:
        return oof.run_compare(
            fold=0,
            zs_run_dir=self.runs / "zs",
            out_dir=self.runs / "oof_fold0_bbbbbbb",
            folds=FOLDS,
            data_root=self.data,
            split="dev",
            n_boot=200,
            out=self.out,
            **kw,
        )

    @property
    def out_dir(self) -> Path:
        return self.runs / "oof_fold0_bbbbbbb"


def make_adapter(final: Path, _fold: int, cfg: spike.SpikeConfig, **override: Any) -> Path:
    """A fake fold adapter folder: dummy weight files + the manifest `train.py` writes."""
    fold = _fold
    (final / "peft").mkdir(parents=True, exist_ok=True)
    (final / "adapter.pt").write_bytes(b"fake adapter state")
    (final / "peft" / "adapter_model.safetensors").write_bytes(b"fake safetensors")
    (final / "peft" / "adapter_config.json").write_text("{}")
    stage = f"fold{fold}"
    train_ids = oof.fold_train_ids(FOLDS, fold)
    m: dict[str, Any] = {
        "stage": stage,
        "fold": fold,
        "manifest_hash": hashlib.sha256(
            (stage + "|" + ",".join(sorted(train_ids))).encode()
        ).hexdigest()[:16],
        "n_heldout_eval_pages": 24,
        "code_sha": SHA40,
        "train_config_signature": "sig",
        "train_doc_ids": train_ids,
        "n_train_docs": len(train_ids),
        "heldout_doc_ids": oof.fold_heldout_ids(FOLDS, fold),
        "lora": {"r": 16, "alpha": 32, "n_modules": 200, "n_trainable": 30_474_240},
        "inference_keys": oof.inference_keys_of_config(cfg),
        "signature": "s",
        "steps": 10,
        "total_steps": 10,
        "n_train_pages": 99,
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


@pytest.fixture()
def w(tmp_path: Path) -> World:
    return World(tmp_path)


# --------------------------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------------------------


def test_full_pipeline_verify_merge_guard_infer_score_compare(w: World) -> None:
    w.zero_shot(null_header=("buyer_name", "currency"), null_row=("purchase_order",))
    adapter = w.adapter()
    backend = MergedMock(w.gold, w.cfg, null_header=("buyer_name",))  # a bit better than zero-shot
    section = w.infer(adapter, backend)
    cmp = w.compare()

    # stage order, as printed
    markers = [
        m for m in w.messages if m in STAGE_MARKERS or m.split(" ==")[0] + " ==" in STAGE_MARKERS
    ]
    heads = [m.split(" ==")[0] + " ==" for m in markers]
    assert heads == STAGE_MARKERS
    assert backend.loads == 1  # merged once, before the guard and the inference
    # the OOF run is the production pipeline over exactly the fold's documents, logprobs on
    d = w.out_dir
    preds = json.loads((d / "predictions.json").read_text())
    traces = [json.loads(x) for x in (d / "trace.jsonl").read_text().splitlines()]
    assert sorted(preds) == HELD0 and [t["doc_id"] for t in traces] == HELD0
    assert all(isinstance(p.get("field_logprobs"), list) for t in traces for p in t["pages"])
    assert all(t["output_format"] == "json" and t["prompt"]["version"] == "v2" for t in traces)
    man = json.loads((d / "manifest.json").read_text())
    o = man["oof"]
    assert (o["fold"], o["n_inference_docs"], o["n_train_inference_overlap"]) == (0, 4, 0)
    assert o["n_train_docs"] == len(DEV_IDS) - 4 and o["merge"]["n_lora_modules_merged"] == 200
    assert o["batch"]["used"] == 2 and o["batch"]["source"] == "zero_shot_run"
    assert o["guard"]["ran"] and o["guard"]["ok"] and not o["guard"]["fallback_to_1"]
    assert (d / "guard" / "bench_result.json").is_file() and section["fold"] == 0
    assert man["batch_size"] == 2 and man["logprobs"] is True and man["split"] == "dev"

    # the comparison: same 4 documents, aggregates only
    assert cmp["n_docs"] == 4 and cmp["subsets"]["all"]["n_docs"] == 4
    assert set(cmp["subsets"]) >= {"all", "invoices", "waybills"}
    assert cmp["subsets"]["invoices"]["n_docs"] == 3 and cmp["subsets"]["waybills"]["n_docs"] == 1
    allb = cmp["subsets"]["all"]
    assert allb["oof"]["OVERALL"] > allb["zero_shot"]["OVERALL"]
    assert allb["paired_delta_oof_minus_zero_shot"]["OVERALL"]["delta"] > 0
    assert (
        allb["oof"]["over_null"]["header_over_null"]
        < allb["zero_shot"]["over_null"]["header_over_null"]
    )
    assert (
        allb["oof"]["over_null"]["row_over_null"]
        == 0
        < allb["zero_shot"]["over_null"]["row_over_null"]
    )
    assert (
        cmp["verdict"]["verdict"] == "NO REGRESSION"
        and cmp["verdict"]["label"] == oof.INTERIM_LABEL
    )
    printed = "\n".join(w.messages)
    assert (
        "G4 INTERIM VERDICT (interim, one fold, not the final G4 decision): NO REGRESSION"
        in printed
    )
    for name in ("oof_compare.json", "oof_compare.md"):
        text = (d / name).read_text()
        assert "dev_0" not in text and "Supplier" not in text and "INV-" not in text  # aggregates
    assert json.loads((d / "oof_compare.json").read_text())["fold"] == 0


def test_the_regression_verdict_names_the_failed_clause(w: World) -> None:
    w.zero_shot()  # perfect zero-shot
    w.infer(
        w.adapter(),
        MergedMock(
            w.gold,
            w.cfg,
            null_header=("supplier_name", "buyer_name"),
            null_row=("customer_part_number",),
        ),
    )
    cmp = w.compare()
    v = cmp["verdict"]
    assert v["verdict"] == "REGRESSION" and v["oof_over_null"] > v["zs_over_null"] == 0
    assert any("over-null" in c for c in v["failed_clauses"])
    assert "REGRESSION (" in "\n".join(w.messages)


def test_resume_after_a_kill_gives_the_same_files(w: World, tmp_path: Path) -> None:
    w.zero_shot()
    adapter = w.adapter()
    with pytest.raises(KeyboardInterrupt):
        w.infer(adapter, MergedMock(w.gold, w.cfg, null_header=("buyer_name",), die_after=16))
    # (the guard pages count as calls: the kill lands after the guard, during the inference)
    part = (
        (w.out_dir / "trace.jsonl").read_text().splitlines()
        if (w.out_dir / "trace.jsonl").is_file()
        else []
    )
    assert len(part) < 4
    w.infer(adapter, MergedMock(w.gold, w.cfg, null_header=("buyer_name",)))
    resumed = (w.out_dir / "predictions.json").read_text()
    other = World(tmp_path / "ref")
    other.zero_shot()
    other.infer(other.adapter(), MergedMock(other.gold, other.cfg, null_header=("buyer_name",)))
    assert (other.out_dir / "predictions.json").read_text() == resumed
    assert [
        json.loads(x)["doc_id"] for x in (w.out_dir / "trace.jsonl").read_text().splitlines()
    ] == HELD0


# --------------------------------------------------------------------------------------------
# Refusals (each one before anything runs: no backend call, no output folder)
# --------------------------------------------------------------------------------------------

REFUSALS = {
    "wrong fold id": ({"stage": "fold1", "fold": 1}, ["stage", "fold id"]),
    "fold field only": ({"fold": 2}, ["fold id"]),
    "wrong base revision": ({"base_revision": "c" * 40}, ["base revision"]),
    "overlapping training ids": (
        {"train_doc_ids": oof.fold_train_ids(FOLDS, 0) + [HELD0[0]]},
        ["train/inference ids disjoint"],
    ),
    "missing training ids": ({"train_doc_ids": DROP}, ["training doc ids"]),
    "empty training ids": ({"train_doc_ids": []}, ["training doc ids"]),
    "training set too small": (
        {"train_doc_ids": oof.fold_train_ids(FOLDS, 0)[:-1]},
        ["training set = all docs outside the fold"],
    ),
    "final adapter": ({"stage": "final", "fold": None}, ["stage"]),
    "smoke adapter": ({"stage": "smoke", "fold": None}, ["stage"]),
    "prompt version mismatch": (
        {
            "inference_keys": {
                **oof.inference_keys_of_config(spike.load_config(KEYED)),
                "prompt_version": "v1",
            }
        },
        ["inference key prompt_version"],
    ),
    "output format mismatch": (
        {
            "inference_keys": {
                **oof.inference_keys_of_config(spike.load_config(KEYED)),
                "output_format": "compact",
            }
        },
        ["inference key output_format"],
    ),
    "wrong lora size": (
        {"lora": {"r": 8, "n_modules": 200, "n_trainable": 15_237_120}},
        ["lora r", "lora trainable params"],
    ),
    "wrong module count": (
        {"lora": {"r": 16, "n_modules": 196, "n_trainable": 30_474_240}},
        ["lora modules"],
    ),
    "dirty training code": ({"code_sha": SHA40 + "+dirty"}, ["training code sha"]),
    "no training code sha": ({"code_sha": DROP}, ["training code sha"]),
    "adapter hash mismatch": ({"adapter_sha256": "0" * 64}, ["adapter.pt sha256"]),
    "peft hash missing": ({"peft_sha256": DROP}, ["peft/adapter_model.safetensors sha256"]),
    "trained before doc ids were recorded": (
        {"train_doc_ids": DROP, "peft_sha256": DROP, "lora": DROP, "inference_keys": DROP},
        ["training doc ids", "lora r", "inference key model_repo"],
    ),
}


@pytest.mark.parametrize("name", list(REFUSALS))
def test_every_refusal_path_fails_closed_before_any_run(w: World, name: str) -> None:
    override, expect_failed = REFUSALS[name]
    w.zero_shot()
    backend = MergedMock(w.gold, w.cfg)
    with pytest.raises(oof.OofError, match="adapter verification FAILED") as exc:
        w.infer(w.adapter(**override), backend)
    for check in expect_failed:
        assert check in str(exc.value), (name, check, str(exc.value))
    assert backend.loads == 0 and not w.out_dir.exists()  # nothing merged, nothing written
    table = "\n".join(w.messages)
    assert "FAILED (refused)" in table and "FAIL " in table


def test_an_unreachable_training_sha_is_refused_and_a_different_pin_only_warns(w: World) -> None:
    w.zero_shot()
    rep = oof.verify_adapter_manifest(
        w.adapter(),
        fold=0,
        cfg=w.cfg,
        zs_manifest=json.loads((w.runs / "zs/manifest.json").read_text()),
        folds=FOLDS,
        pin_sha="d" * 40,
        reachable=lambda s: False,
    )
    assert not rep["ok"]
    assert [r["check"] for r in rep["rows"] if not r["ok"]] == ["training code sha reachable"]
    rep = oof.verify_adapter_manifest(
        w.adapter(),
        fold=0,
        cfg=w.cfg,
        zs_manifest=json.loads((w.runs / "zs/manifest.json").read_text()),
        folds=FOLDS,
        pin_sha="d" * 40,
        reachable=lambda s: True,
    )
    assert rep["ok"] and len(rep["warnings"]) == 1 and "differs" in rep["warnings"][0]
    table = oof.format_verification(rep)
    assert SHA40 in table and "d" * 40 in table and "overlap 0" in table


def test_a_missing_manifest_and_a_foreign_zero_shot_run_are_refused(
    w: World, tmp_path: Path
) -> None:
    w.zero_shot()
    empty = tmp_path / "empty"
    empty.mkdir()
    zs_man = json.loads((w.runs / "zs/manifest.json").read_text())
    rep = oof.verify_adapter_manifest(empty, fold=0, cfg=w.cfg, zs_manifest=zs_man, folds=FOLDS)
    assert not rep["ok"] and rep["rows"][0]["check"] == "manifest.json"
    bad = {**zs_man, "config": {**zs_man["config"], "hash": "0" * 16}}
    rep = oof.verify_adapter_manifest(w.adapter(), fold=0, cfg=w.cfg, zs_manifest=bad, folds=FOLDS)
    assert [r["check"] for r in rep["rows"] if not r["ok"]] == ["zero-shot run config hash"]
    with pytest.raises(oof.OofError, match="fold 5"):
        oof.fold_heldout_ids(FOLDS, 5)


def test_an_unfinished_zero_shot_run_is_refused(w: World) -> None:
    zs = w.zero_shot()
    (zs / "progress.json").write_text(json.dumps({"status": "running"}))
    with pytest.raises(oof.OofError, match="not complete"):
        w.infer(w.adapter(), MergedMock(w.gold, w.cfg))


# --------------------------------------------------------------------------------------------
# Batch size, guard, merge
# --------------------------------------------------------------------------------------------


def test_batch_size_comes_from_the_zero_shot_run_and_is_refused_when_absent(w: World) -> None:
    zs = w.zero_shot(batch=2)
    r = oof.resolve_batch_size(zs, None)
    assert (r["batch_size"], r["source"]) == (2, "zero_shot_run")
    assert set(r["stored"]) == {"bench_result.json", "manifest.json"}
    r = oof.resolve_batch_size(zs, 4)
    assert (r["batch_size"], r["source"], r["differs_from_zero_shot"]) == (4, "manual", True)
    assert oof.resolve_batch_size(zs, 2)["differs_from_zero_shot"] is False
    # only the manifest (no bench_result.json) is enough
    (zs / "bench_result.json").unlink()
    assert oof.resolve_batch_size(zs, None)["stored"] == {"manifest.json": 2}
    # neither: refused unless an int is given
    man = json.loads((zs / "manifest.json").read_text())
    man.pop("batch_size")
    (zs / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(oof.OofError, match="batch size is unknown"):
        oof.resolve_batch_size(zs, None)
    assert oof.resolve_batch_size(zs, 3)["batch_size"] == 3
    with pytest.raises(oof.OofError, match="BATCH_SIZE"):
        oof.resolve_batch_size(zs, 0)
    (zs / "bench_result.json").write_text(json.dumps({"chosen_batch_size": 8}))
    man["batch_size"] = 2
    (zs / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(oof.OofError, match="disagree"):
        oof.resolve_batch_size(zs, None)


def test_missing_bench_result_stops_the_run_before_the_merge(w: World) -> None:
    zs = w.zero_shot()
    (zs / "bench_result.json").unlink()
    man = json.loads((zs / "manifest.json").read_text())
    man.pop("batch_size")
    (zs / "manifest.json").write_text(json.dumps(man))
    backend = MergedMock(w.gold, w.cfg)
    with pytest.raises(oof.OofError, match="batch size is unknown"):
        w.infer(w.adapter(), backend)
    assert backend.loads == 0
    w.infer(w.adapter(), backend, batch_size=1)  # an explicit int is accepted
    assert (
        json.loads((w.out_dir / "manifest.json").read_text())["oof"]["batch"]["source"] == "manual"
    )


def test_a_batched_decode_that_differs_falls_back_to_batch_1(w: World) -> None:
    w.zero_shot(batch=2)
    w.infer(w.adapter(), MergedMock(w.gold, w.cfg, flip_in_batches=True))
    man = json.loads((w.out_dir / "manifest.json").read_text())
    g = man["oof"]["guard"]
    assert g["ran"] and g["ok"] is False and g["fallback_to_1"] is True
    assert g["byte_identical_rate"] < 1.0
    assert man["batch_size"] == 1 and man["oof"]["batch"]["used"] == 1
    assert man["oof"]["batch"]["batch_size"] == 2  # the requested size is still on record
    preds = json.loads((w.out_dir / "predictions.json").read_text())
    assert all(p["header"].get("currency") != "EUR" for p in preds.values())  # batch-1 text
    assert any("FALLING BACK TO BATCH 1" in m for m in w.messages)


def test_the_guard_is_skipped_at_batch_1_and_runs_before_the_full_inference(w: World) -> None:
    w.zero_shot(batch=1)
    w.infer(w.adapter(), MergedMock(w.gold, w.cfg))
    man = json.loads((w.out_dir / "manifest.json").read_text())
    assert man["oof"]["guard"]["ran"] is False and man["batch_size"] == 1
    assert not (w.out_dir / "guard").exists()
    assert any(m.startswith("== GUARD == skipped") for m in w.messages)


def test_a_merge_that_merged_the_wrong_number_of_modules_is_refused(w: World) -> None:
    w.zero_shot()
    for n in (0, 199):
        with pytest.raises(oof.OofError, match="did not merge 200"):
            w.infer(w.adapter(), MergedMock(w.gold, w.cfg, n_merged=n))
    assert not (w.out_dir / "trace.jsonl").exists()  # nothing was inferred


def test_merged_backend_checks_the_module_counts_with_a_stub_peft(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Layer:
        lora_A = lora_B = object()

    class Base:
        def __init__(self, n: int) -> None:
            self.n = n
            self.merged = False

        def modules(self) -> list[Any]:
            return [] if self.merged else [Layer() for _ in range(self.n)]

        def parameters(self) -> Any:
            return iter([types.SimpleNamespace(dtype="torch.float16")])

        def eval(self) -> Base:
            return self

        def merge_and_unload(self) -> Base:
            self.merged = True
            return self

    class PeftModel:
        n = 200

        @staticmethod
        def from_pretrained(model: Any, path: str) -> Base:
            assert path.endswith("peft")
            return Base(PeftModel.n)

    monkeypatch.setitem(sys.modules, "peft", types.SimpleNamespace(PeftModel=PeftModel))
    monkeypatch.setattr(
        oof.HfBackend,
        "load",
        lambda self: setattr(self, "_loaded", True) or setattr(self, "model", object()),
    )
    cfg = spike.load_config(KEYED).backend
    b = oof.MergedHfBackend(cfg, tmp_path, "ab" * 32)
    assert b.model_id.startswith(cfg.repo + "+lora:abababababab")
    b.load()
    assert b.merge_info and b.merge_info["n_lora_modules_merged"] == 200
    assert (
        b.merge_info["n_lora_modules_left"] == 0 and b.merge_info["merge_dtype"] == "torch.float16"
    )
    PeftModel.n = 3
    b2 = oof.MergedHfBackend(cfg, tmp_path)
    with pytest.raises(oof.OofError, match="attached to 3 modules"):
        b2.load()


# --------------------------------------------------------------------------------------------
# Verdict, over-null counts, comparison
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lo", "oof_n", "zs_n", "verdict", "failed"),
    [
        (-0.0099, 5, 5, "NO REGRESSION", 0),  # just above -1.0 point, equal over-nulls
        (-0.0101, 5, 5, "REGRESSION", 1),  # just below -1.0 point
        (-0.0100, 5, 5, "REGRESSION", 1),  # exactly -1.0 is not "above"
        (0.002, 4, 5, "NO REGRESSION", 0),  # fewer over-nulls
        (0.002, 6, 5, "REGRESSION", 1),  # one more over-null
        (-0.02, 9, 5, "REGRESSION", 2),  # both clauses
        (-0.005, 0, 0, "NO REGRESSION", 0),
    ],
)
def test_verdict_table(lo: float, oof_n: int, zs_n: int, verdict: str, failed: int) -> None:
    v = oof.g4_interim_verdict(lo, oof_n, zs_n)
    assert v["verdict"] == verdict and len(v["failed_clauses"]) == failed
    assert v["delta_excludes_zero_positive"] is (lo > 0)  # reported, never part of the verdict
    line = oof.verdict_line(v)
    assert line.startswith("G4 INTERIM VERDICT (interim, one fold, not the final G4 decision): ")
    assert verdict in line


def test_over_null_counts_header_rows_and_false_fills() -> None:
    gold = {
        "a": {
            "doc_type": "invoice",
            "header": {
                "invoice_number": "X1",
                "invoice_date": "2026-01-02",
                "supplier_name": "S",
                "buyer_name": None,
                "ship_to_name": "T",
                "currency": "USD",
                "total_amount": "5.00",
                "awb_number": None,
            },
            "line_items": [
                {
                    "supplier_part_number": "P1",
                    "customer_part_number": "C1",
                    "purchase_order": "O1",
                    "quantity": "1",
                },
                {
                    "supplier_part_number": "P2",
                    "customer_part_number": None,
                    "purchase_order": "O2",
                    "quantity": "2",
                },
                {
                    "supplier_part_number": "P3",
                    "customer_part_number": None,
                    "purchase_order": None,
                    "quantity": "3",
                },
            ],
        },
    }
    pred = {
        "a": {
            "doc_type": "invoice",
            "header": {
                "invoice_number": None,
                "invoice_date": "",
                "supplier_name": "S",
                "buyer_name": "FILLED",
                "ship_to_name": "T",
                "currency": None,
                "total_amount": "5.00",
                "awb_number": None,
            },
            "line_items": [
                {
                    "supplier_part_number": "P1",
                    "customer_part_number": None,
                    "purchase_order": None,
                    "quantity": "1",
                },
                {
                    "supplier_part_number": "P2",
                    "customer_part_number": "FF",
                    "purchase_order": "O2",
                    "quantity": "2",
                },
            ],
        },
    }
    c = oof.over_null_counts(pred, gold)
    assert c["header_over_null_by_field"] == {"currency": 1, "invoice_date": 1, "invoice_number": 1}
    assert c["header_over_null"] == 3 and c["header_false_fill"] == 1
    assert c["row_over_null_by_field"] == {
        "supplier_part_number": 0,
        "customer_part_number": 1,
        "purchase_order": 1,
        "quantity": 0,
    }
    assert c["row_over_null"] == 2 and c["row_false_fill"] == 1
    assert c["over_null_total"] == 5 and c["false_fill_total"] == 2
    assert c["rows_unmatched_gold"] == 1  # P3 was never predicted: a missing row, not a cell
    assert (
        oof.over_null_counts({}, gold)["header_over_null"] == 6
    )  # a missing doc: every gold value is null


def test_compare_refuses_missing_documents_and_formats_aggregates_only(w: World) -> None:
    gold = {d: w.gold[d] for d in HELD0}
    perfect = {
        d: {"doc_type": g["doc_type"], "header": g["header"], "line_items": g["line_items"]}
        for d, g in gold.items()
    }
    with pytest.raises(oof.OofError, match="miss 1 of 4"):
        oof.compare_models(perfect, {k: v for k, v in perfect.items() if k != HELD0[0]}, gold)
    cmp = oof.compare_models(perfect, perfect, gold, None, n_boot=50)
    d = cmp["subsets"]["all"]["paired_delta_oof_minus_zero_shot"]["OVERALL"]
    assert d["delta"] == 0 and d["lo"] == d["hi"] == 0
    assert (
        cmp["verdict"]["verdict"] == "NO REGRESSION"
        and not cmp["verdict"]["delta_excludes_zero_positive"]
    )
    text = oof.format_compare(cmp)
    md = oof.format_compare(cmp, markdown=True)
    for t in (text, md):
        assert "interim, one fold, not the final G4 decision" in t
        assert HELD0[0] not in t and "Supplier" not in t
    assert "| OVERALL |" in md and "excludes 0 on the positive side: False" in text


def test_the_ci_and_the_paired_delta_use_the_documented_bootstrap(w: World) -> None:
    w.zero_shot(null_header=("buyer_name",))
    w.infer(w.adapter(), MergedMock(w.gold, w.cfg))
    cmp = w.compare()
    assert (cmp["n_boot"], cmp["seed"]) == (200, 42)
    gold = {d: w.gold[d] for d in HELD0}
    zs = json.loads((w.runs / "zs/predictions.json").read_text())
    oo = json.loads((w.out_dir / "predictions.json").read_text())
    direct = ev.paired_bootstrap(zs, oo, gold, n=200, seed=42)
    assert cmp["subsets"]["all"]["paired_delta_oof_minus_zero_shot"]["OVERALL"] == direct["OVERALL"]
    assert cmp["subsets"]["all"]["oof"]["OVERALL"] == direct["OVERALL"]["b"]


def test_compare_needs_a_complete_oof_run(w: World) -> None:
    w.zero_shot()
    w.out_dir.mkdir(parents=True)
    with pytest.raises(oof.OofError, match="not complete"):
        w.compare()


# --------------------------------------------------------------------------------------------
# Estimate, folds, cli
# --------------------------------------------------------------------------------------------


def _ge() -> Any:
    spec = importlib.util.spec_from_file_location(
        "gpu_estimate", ROOT / "scripts" / "gpu_estimate.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_estimate_includes_load_merge_and_guard() -> None:
    ge = _ge()
    speed = json.loads((ROOT / "configs" / "spike_speed.json").read_text(encoding="utf-8"))
    rows = oof.estimate_rows(ge, speed, "qwen35_4b_img_only", 230, 4)
    assert [r["efficiency"] for r in rows] == [1.0, 0.75, 0.5]
    for r in rows:
        assert r["hours"] * 3600 == pytest.approx(
            r["load_s"] + r["merge_s"] + r["guard_s"] + r["infer_s"]
        )
        assert r["guard_s"] > 0 and r["merge_s"] == oof.MERGE_S_ESTIMATE
        assert r["infer_s"] == pytest.approx(230 * r["s_per_page"])
        assert r["cu_central"] == pytest.approx(r["hours"] * speed["t4_cu_per_hour"])
    assert rows[0]["hours"] < rows[1]["hours"] < rows[2]["hours"]
    one = oof.estimate_rows(ge, speed, "qwen35_4b_img_only", 230, 1)
    assert len(one) == 1 and one[0]["guard_s"] == 0
    text = oof.format_estimate(rows, 230, 4, 0)
    assert "ESTIMATE (UNVERIFIED)" in text and "230 held-out pages" in text


def test_real_folds_train_set_is_everything_outside_the_fold() -> None:
    folds = ts.load_folds()
    every = sorted(d for f in folds["folds"] for d in f["val_doc_ids"])
    assert len(every) == 500
    for k in (0, 1, 2):
        held, train = oof.fold_heldout_ids(folds, k), oof.fold_train_ids(folds, k)
        assert len(train) == 500 - len(held) and not set(train) & set(held)
        split = ts.stage_split(f"fold{k}", folds)
        assert list(split.train_ids) == train and list(split.heldout_ids) == held


def test_training_manifest_records_what_the_oof_verification_needs(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    import test_train as tt

    folds = ts.load_folds()
    split = ts.stage_split("fold0", folds)
    cfg = tr.load_config()
    meta = tr.build_run_meta(
        "fold0",
        split,
        cfg,
        n_heldout_eval_pages=24,
        n_lora_modules=200,
        n_trainable=30_474_240,
        code_sha=SHA40,
    )
    assert meta["fold"] == 0 and meta["code_sha"] == SHA40
    assert meta["train_doc_ids"] == oof.fold_train_ids(folds, 0)
    assert meta["heldout_doc_ids"] == oof.fold_heldout_ids(folds, 0)
    assert meta["n_train_docs"] == len(meta["train_doc_ids"])
    assert meta["lora"] == {"r": 16, "alpha": 32, "n_modules": 200, "n_trainable": 30_474_240}
    assert (
        tr.build_run_meta(
            "final",
            ts.stage_split("final", folds),
            cfg,
            n_heldout_eval_pages=0,
            n_lora_modules=200,
            n_trainable=1,
            code_sha=SHA40,
        )["fold"]
        is None
    )
    # the keys the training run records are exactly the ones the production config yields
    prod = oof.inference_keys_of_config(spike.load_config(KEYED))
    assert meta["inference_keys"] == prod and set(prod) == set(oof.INFERENCE_KEYS)

    class SavingLM(tt.TinyLM):
        def save_pretrained(self, path: Path) -> None:
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "adapter_model.safetensors").write_bytes(b"w")
            (Path(path) / "adapter_config.json").write_text("{}")

    t = tt.make_trainer(tmp_path / "run", model=SavingLM(), run_meta=meta)
    t.fit()
    final = t.save_final()
    man = json.loads((final / "manifest.json").read_text())
    assert man["train_doc_ids"] == meta["train_doc_ids"] and man["stage"] == "fold0"
    assert man["peft_sha256"]["adapter_model.safetensors"] == hashlib.sha256(b"w").hexdigest()
    assert man["adapter_sha256"] == hashlib.sha256((final / "adapter.pt").read_bytes()).hexdigest()
    assert man["base_revision"] == cfg.revision and man["precision"] == "fp32"


def test_cli_registers_the_oof_stages() -> None:
    parser = cli.build_parser()
    a = parser.parse_args(
        [
            "oof",
            "infer",
            "--fold",
            "0",
            "--config",
            "c.yaml",
            "--zs-run-dir",
            "z",
            "--out-dir",
            "o",
            "--adapter-dir",
            "a",
            "--pin",
            SHA40,
        ]
    )
    assert (a.command, a.stage, a.fold, a.batch_size, a.bench_docs) == (
        "oof",
        "infer",
        0,
        None,
        "splits/bench12.json",
    )
    c = parser.parse_args(
        ["oof", "compare", "--fold", "1", "--config", "c", "--zs-run-dir", "z", "--out-dir", "o"]
    )
    assert c.stage == "compare" and not hasattr(c, "adapter_dir")
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "oof",
                "infer",
                "--fold",
                "3",
                "--config",
                "c",
                "--zs-run-dir",
                "z",
                "--out-dir",
                "o",
                "--adapter-dir",
                "a",
            ]
        )  # fold must be 0, 1 or 2


@pytest.mark.skipif(not (ROOT / "data" / "train" / "labels").is_dir(), reason="data/ absent")
def test_estimate_stage_counts_the_pages_of_the_real_fold(tmp_path: Path) -> None:
    folds = ts.load_folds()
    lines: list[str] = []
    res = oof.run_estimate(
        fold=0,
        zs_run_dir=tmp_path,
        cfg=spike.load_config(KEYED),
        folds=folds,
        data_root=ROOT / "data",
        batch_size=4,
        out=lines.append,
    )
    assert res["n_docs"] == len(oof.fold_heldout_ids(folds, 0))
    assert res["n_pages"] == 230  # the page count the 03 notebook banner and the plan quote
    assert res["batch"]["source"] == "manual" and len(res["rows"]) == 3
    text = "\n".join(lines)
    assert "ESTIMATE (UNVERIFIED)" in text and "230 held-out pages at batch 4" in text
    with pytest.raises(oof.OofError, match="batch size is unknown"):  # no int, no stored size
        oof.run_estimate(
            fold=0,
            zs_run_dir=tmp_path,
            cfg=spike.load_config(KEYED),
            folds=folds,
            data_root=ROOT / "data",
            out=lines.append,
        )
