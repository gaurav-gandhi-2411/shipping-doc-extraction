"""shipdoc.flags: models, the id guard, feature parity, the flags stage on a synthetic corpus.

Everything runs on the mock backend over the SYNTHETIC corpus of tests/_synth.py (the fake ``test``
folder of tests/test_predict.py): no real image, label, OCR text, prediction or calibrator run is
used. The calibrators here are fitted on synthetic labels; they carry no information about the real
data. The assignment schema (gitignored) is needed for the end-to-end tests; they skip
without it.
"""

from __future__ import annotations

# ruff: noqa: F811  (the `world` fixture is imported, then requested by name, as pytest requires)
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from _synth import SPECS
from test_predict import (  # noqa: F401 - fixtures / helpers shared with the other test modules
    SCHEMA,
    TEST_IDS,
    World,
    _clean_sha,
    _edit_trace,
    _ocr_and_shapes,
    assemble,
    needs_schema,
    run_all,
    world,
)

from shipdoc import cluster, flags, ocr, predict, reuse, spike
from shipdoc import confidence as cf
from shipdoc import confidence_v2 as c2
from shipdoc import eval as ev
from shipdoc.postrules import RuleConfig

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "freeze_calibrator", ROOT / "scripts" / "freeze_calibrator.py"
)
freeze = importlib.util.module_from_spec(_spec)
sys.modules["freeze_calibrator"] = freeze
_spec.loader.exec_module(freeze)

SIG_DIM = 3 * len(cluster.KEYWORDS)


# --------------------------------------------------------------------------------------------
# Models: the JSON spec predicts what the sklearn model predicts
# --------------------------------------------------------------------------------------------


def _xy(n: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    X[::7, 0] = np.nan  # missing values are mean-imputed
    X[:, 5] = np.nan  # a column that is NaN in every fitting row
    y = (X[:, 1] + 0.5 * rng.normal(size=n) > 0).astype(int)
    return X, y


@pytest.mark.parametrize("kind", ["lr", "gbm"])
def test_model_spec_predicts_what_fit_predict_predicts(kind: str) -> None:
    X, y = _xy()
    spec = json.loads(json.dumps(flags.fit_model(kind, X, y)))  # through JSON, as in the artifact
    got = flags.Model(spec).predict(X)
    want = cf.fit_predict(kind, X, y, X)
    assert np.allclose(got, want, atol=1e-9, rtol=0), np.abs(got - want).max()
    Xn = X.copy()
    Xn[3, 1] = np.nan  # a NaN at prediction time is imputed with the fitted mean
    assert np.allclose(
        flags.Model(spec).predict(Xn), cf.fit_predict(kind, X, y, Xn), atol=1e-9, rtol=0
    )


def test_constant_model_when_a_class_is_missing() -> None:
    X, _ = _xy(50)
    spec = flags.fit_model("lr", X, np.ones(50, dtype=int))
    assert spec["kind"] == "constant" and spec["p"] == 1.0
    assert np.allclose(flags.Model(spec).predict(X), cf.fit_predict("lr", X, np.ones(50), X))


def test_gbm_pickle_is_refused_under_another_sklearn_or_after_tampering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X, y = _xy(200)
    spec = flags.fit_model("gbm", X, y)
    assert spec["sklearn_version"] == flags.sklearn_version() and len(spec["pickle_sha256"]) == 64
    monkeypatch.setattr(flags, "sklearn_version", lambda: "0.0.1")
    with pytest.raises(flags.FlagsError, match=r"scikit-learn .* this environment has 0\.0\.1"):
        flags.Model(spec)
    monkeypatch.undo()
    bad = {**spec, "pickle_sha256": "0" * 64}
    with pytest.raises(flags.FlagsError, match="sha256"):
        flags.Model(bad)
    with pytest.raises(flags.FlagsError, match="unknown model kind"):
        flags.Model({"kind": "mystery"})
    lr = flags.Model(flags.fit_model("lr", X, y))
    with pytest.raises(flags.FlagsError, match="columns"):
        lr.predict(X[:, :3])


# --------------------------------------------------------------------------------------------
# The id guard
# --------------------------------------------------------------------------------------------


def test_id_guard_inverts_the_test_id_refusal_only_for_test_ids_and_restores_it() -> None:
    original = c2.assert_no_test_ids
    with pytest.raises(c2.NoTestDataError):
        c2.assert_no_test_ids(["test_0001"])  # the calibration code refuses test ids
    with flags.id_guard(["train_0001", "dev_0002"]) as mode:
        assert mode == "calibration" and c2.assert_no_test_ids is original
        with pytest.raises(c2.NoTestDataError):
            c2.assert_no_test_ids(["test_0001"])
    with flags.id_guard(["test_0001", "test_0002"]) as mode:
        assert mode == "inference"
        c2.assert_no_test_ids(["test_0003"])  # accepted here ...
        with pytest.raises(c2.NoTestDataError):
            c2.assert_no_test_ids(["train_0003"])  # ... and ONLY test ids are
    assert c2.assert_no_test_ids is original
    with (
        pytest.raises(flags.FlagsError, match="one call"),
        flags.id_guard(["test_0001", "dev_0001"]),
    ):
        pass
    with pytest.raises(RuntimeError, match="boom"), flags.id_guard(["test_0001"]):
        raise RuntimeError("boom")
    assert c2.assert_no_test_ids is original  # restored after an exception too


# --------------------------------------------------------------------------------------------
# Feature parity: the SAME code builds the training and the inference design
# --------------------------------------------------------------------------------------------


def _probe_world(prefix: str) -> dict[str, Any]:
    p = flags._probe_inputs()
    d = f"{prefix}_0001"
    return {
        "d": d,
        "post": {d: p["doc"]},
        "traces": {d: p["trace"]},
        "ocr": {d: p["ocr"]},
        "changes": {d: p["changes"]},
        "zs": {d: p["twin"]},
    }


@pytest.mark.parametrize("arm", ["zs", "ft"])
def test_training_and_inference_paths_build_identical_features_and_probabilities(arm: str) -> None:
    ref = np.zeros((1, SIG_DIM))
    designs: dict[str, np.ndarray] = {}
    tables: dict[str, flags.Tables] = {}
    for prefix in ("dev", "test"):  # the training path (strict guard) vs the inference path
        w = _probe_world(prefix)
        t = flags.build_tables(
            arm,
            [w["d"]],
            w["post"],
            w["traces"],
            w["ocr"],
            w["changes"],
            w["zs"] if arm == "ft" else None,
        )
        tables[prefix] = t
        designs[prefix] = flags.design(arm, t, flags.novelty_of(t.sigs, ref))
    assert designs["dev"].shape == designs["test"].shape
    assert designs["dev"].shape[1] == len(flags.feature_names(arm))
    assert np.array_equal(designs["dev"], designs["test"], equal_nan=True)
    assert [k[1:] for k in tables["dev"].tv.keys] == [k[1:] for k in tables["test"].tv.keys]
    if arm == "ft":  # the agreement block is informative on the probe (the twin differs)
        a = tables["test"].agree
        assert (
            a is not None and a[:, c2_agree("ag_norm")].min() == 0 < a[:, c2_agree("ag_norm")].max()
        )
    # a model sees the same row -> the same probability
    X, y = designs["test"], np.arange(len(designs["test"])) % 2
    model = flags.Model(flags.fit_model("lr", np.vstack([X] * 3), np.tile(y, 3)))
    assert np.array_equal(model.predict(designs["dev"]), model.predict(designs["test"]))


def c2_agree(name: str) -> int:
    from shipdoc import confidence_v3 as c3

    return c3.AGREE_FEATURES.index(name)


def test_the_calibration_path_still_refuses_test_ids_and_the_probe_is_stable() -> None:
    w = _probe_world("test")
    # without the guard (the calibration code called directly) a test id aborts
    from shipdoc import confidence_v3 as c3

    with pytest.raises(c2.NoTestDataError):
        c3.build_table_v2(["test_0001"], w["post"], w["traces"], w["ocr"], w["changes"])
    a, b = flags.parity_probe("zs"), flags.parity_probe("zs")
    assert a == b and len(a["digest"]) == 64 and a["rows"] > 10
    assert flags.parity_probe("ft")["digest"] != a["digest"]  # more columns
    assert flags.parity_probe("ft")["columns"] == a["columns"] + len(flags_agree_names())


def flags_agree_names() -> tuple[str, ...]:
    from shipdoc import confidence_v3 as c3

    return c3.AGREE_FEATURES


def test_a_behaviour_change_of_the_feature_builder_is_caught_by_the_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inp = synthetic_inputs("zs", n_docs=45)
    art, _ = freeze.fit_artifact(inp, n_boot=10)
    path = tmp_path / "cal.json"
    freeze.write_artifact(art, path)
    flags.load_calibrator(path, "zs")  # loads and passes its own probe
    real = cf.layout_novelty
    monkeypatch.setattr(cf, "layout_novelty", lambda sig, ref: real(sig, ref) + 1.0)
    with pytest.raises(flags.FlagsError, match="parity probe"):
        flags.load_calibrator(path, "zs")
    monkeypatch.undo()
    with pytest.raises(flags.FlagsError, match="arm"):
        flags.load_calibrator(path, "ft")
    bad = {**art, "feature_names": art["feature_names"][:-1]}
    freeze.write_artifact(bad, tmp_path / "bad.json")
    with pytest.raises(flags.FlagsError, match="feature set"):
        flags.load_calibrator(tmp_path / "bad.json", "zs")


# --------------------------------------------------------------------------------------------
# Synthetic calibrator inputs (fitted on random labels: only the mechanics are tested)
# --------------------------------------------------------------------------------------------


def synthetic_inputs(
    arm: str = "zs", n_docs: int = 60, structure: str = "pooled", kind: str = "lr", seed: int = 1
) -> Any:
    """`freeze.Inputs` over the header-only synthetic table of ``confidence.synthetic_data``."""
    syn = cf.synthetic_data(n_docs=n_docs, fields_per_doc=12, seed=seed)
    tv = c2.TableV2(syn.table, np.zeros((len(syn.table.keys), len(c2.EXTRA_FEATURES))))
    ids = sorted(syn.doc_fold)
    rng = np.random.default_rng(seed)
    p = np.where(tv.emitted, np.clip(syn.true_p_correct, 0.01, 0.99), np.nan)
    y_doc = np.array([i % 4 == 0 for i in range(len(ids))])
    sigs = {d: rng.random(SIG_DIM) for d in ids}
    agree = None
    if arm == "ft":
        from shipdoc import confidence_v3 as c3

        agree = np.zeros((len(tv.keys), len(c3.AGREE_FEATURES)))
        agree[:, c2_agree("ag_norm")] = (rng.random(len(tv.keys)) < 0.8).astype(float)
    nested = [
        {
            "population": ft,
            "target": t,
            "n": 10,
            "precision_accepted": {"point": 0.97, "lo": 0.9, "hi": 1.0},
            "accepted_share": {"point": 0.5, "lo": 0.4, "hi": 0.6},
            "review_rate": {"point": 0.5, "lo": 0.4, "hi": 0.6},
            "error_recall": {"point": 0.5, "lo": 0.4, "hi": 0.6},
        }
        for ft in c2.FIELD_TYPES
        for t in flags.FIELD_TARGETS
    ]
    return freeze.Inputs(
        arm,
        "synthetic",
        tv,
        syn.y_correct,
        p,
        y_doc,
        ids,
        syn.doc_fold,
        syn.groups,
        sigs,
        structure,
        kind,
        nested,
        agree,
        {"kind": "synthetic"},
    )


# --------------------------------------------------------------------------------------------
# compute_flags semantics on a hand-made calibrator
# --------------------------------------------------------------------------------------------


def _hand_calibrator(
    p_field: float, tau_header: float | None, tau_doc: float | None, p_doc: float = 0.9
) -> flags.Calibrator:
    taus = {ft: tau_header for ft in c2.FIELD_TYPES}
    data = {
        "arm": "zs",
        "field_model": {"structure": "pooled", "kind": "lr", "models": {}},
        "thresholds": {
            "meaning": "m",
            "field": {"tau": {flags.tkey(t): taus for t in flags.FIELD_TARGETS}},
            "doc": {"tau": {flags.tkey(t): tau_doc for t in flags.DOC_TARGETS}},
            "nested_target_met": {
                "field": {
                    flags.tkey(t): dict.fromkeys(c2.FIELD_TYPES, True) for t in flags.FIELD_TARGETS
                },
                "doc": {flags.tkey(t): False for t in flags.DOC_TARGETS},
            },
        },
    }
    return flags.Calibrator(
        data,
        "0" * 64,
        {"all": flags.Model({"kind": "constant", "p": p_field})},
        flags.Model({"kind": "constant", "p": p_doc}),
        np.zeros((1, SIG_DIM)),
    )


def _tables_for(doc_ids: list[str]) -> flags.Tables:
    p = flags._probe_inputs()
    post = {d: p["doc"] for d in doc_ids}
    post[doc_ids[-1]] = {**p["doc"], "header": {**p["doc"]["header"], "buyer_name": None}}
    t = flags.build_tables(
        "zs",
        doc_ids,
        post,
        dict.fromkeys(doc_ids, p["trace"]),
        dict.fromkeys(doc_ids, p["ocr"]),
        dict.fromkeys(doc_ids, p["changes"]),
    )
    return t


def test_flags_accept_above_tau_review_below_none_means_review_and_nulls_are_not_scored() -> None:
    ids = ["test_0001", "test_0002"]
    t = _tables_for(ids)
    nov = flags.novelty_of(t.sigs, np.zeros((1, SIG_DIM)))
    X = flags.design("zs", t, nov)
    for p_field, tau, expect in (
        (0.9, 0.5, "accept"),
        (0.9, 0.95, "review"),
        (0.9, None, "review"),
    ):
        block = flags.compute_flags(_hand_calibrator(p_field, tau, 0.5), t, nov, X)
        d = block["docs"]["test_0001"]
        assert d["header"]["invoice_number"] == {
            "emitted": True,
            "p_correct": p_field,
            "flag": expect,
        }
        assert d["auto_accept"] is True and d["p_fully_correct"] == 0.9
        assert [r["row_idx"] for r in d["rows"]] == [0, 1, 2]
        assert d["rows"][0]["fields"]["customer_part_number"] == {  # a null cell: not scored
            "emitted": False,
            "p_correct": None,
            "flag": None,
        }
        null_hdr = block["docs"]["test_0002"]["header"]["buyer_name"]
        assert null_hdr == {"emitted": False, "p_correct": None, "flag": None}
    s = block["summary"]
    assert s["n_docs"] == 2 and s["n_accept"] + s["n_review"] == s["n_emitted_fields"]
    assert s["n_null_fields_not_scored"] > 0
    assert all(v in flags.DOC_STRING_VALUES for v in flags.doc_string_values(block["docs"]))
    no_doc_accept = flags.compute_flags(_hand_calibrator(0.9, 0.5, None), t, nov, X)
    assert not any(v["auto_accept"] for v in no_doc_accept["docs"].values())
    high = flags.compute_flags(_hand_calibrator(0.9, 0.5, 0.99, p_doc=0.9), t, nov, X)
    assert not any(v["auto_accept"] for v in high["docs"].values())  # p_doc < tau_doc


def test_flags_refuse_an_emitted_field_without_a_probability() -> None:
    ids = ["test_0001"]
    t = _tables_for(ids)
    nov = flags.novelty_of(t.sigs, np.zeros((1, SIG_DIM)))
    X = flags.design("zs", t, nov)
    with pytest.raises(flags.FlagsError, match="NaN"):
        flags.compute_flags(_hand_calibrator(float("nan"), 0.5, 0.5), t, nov, X)


def test_calibrator_refuses_an_unknown_target_and_a_wrong_artifact(tmp_path: Path) -> None:
    cal = _hand_calibrator(0.9, 0.5, 0.5)
    with pytest.raises(flags.FlagsError, match="field tau"):
        cal.field_tau(0.5)
    with pytest.raises(flags.FlagsError, match="document tau"):
        cal.doc_tau(0.5)
    (tmp_path / "x.json").write_text(json.dumps({"artifact": "other"}))
    with pytest.raises(flags.FlagsError, match="not a schema"):
        flags.load_calibrator(tmp_path / "x.json")


# --------------------------------------------------------------------------------------------
# The stage on a synthetic submission
# --------------------------------------------------------------------------------------------


def world_submission(
    w: World, tmp_path: Path, name: str = "v1_x", backend: Any = None, model_id: str | None = None
) -> tuple[Path, Path, Path]:
    """``(submission dir, ocr root, shapes file)``: a mock run assembled with R1-R3 ON."""
    run_all(w, be=backend)
    _edit_trace(w)  # every rule has something to do
    ocr_root, shapes = _ocr_and_shapes(tmp_path)
    for stem in ("test_0008_p1",):  # the second waybill needs cached OCR (R2 must not be skipped)
        page = ocr.PageOcr(
            stem,
            10,
            5,
            "paddleocr",
            "line",
            0,
            [ocr.OcrItem("MAWB 176-87654321", [[0, 0], [10, 0], [10, 5], [0, 5]], 0.9)],
        )
        path = ocr.cache_path(ocr_root, ocr.DEFAULT_ENGINE, "test", stem)
        path.write_text(json.dumps(page.to_dict()), encoding="utf-8")
    sub = tmp_path / name
    rep = assemble(w, sub, ocr_cache=ocr_root, shapes_file=shapes)
    assert rep["ok"], rep["checks"]
    if model_id is not None:
        man = json.loads((sub / "manifest.json").read_text())
        man["model"]["id"] = model_id
        (sub / "manifest.json").write_text(json.dumps(man))
    return sub, ocr_root, shapes


def calibrator_for(
    sub: Path,
    ocr_root: Path,
    arm: str,
    tmp_path: Path,
    zs_sub: Path | None = None,
    structure: str = "pooled",
    kind: str = "lr",
) -> Path:
    """A calibrator fitted on the submission's documents RENAMED to dev ids, random labels."""
    rng = np.random.default_rng(7)
    preds = json.loads((sub / "test_predictions.json").read_text())
    traces = {t["doc_id"]: t for t in spike._read_trace(sub / "trace.jsonl")}
    rec = {
        json.loads(ln)["doc_id"]: json.loads(ln)["changes"]
        for ln in (sub / "rules.jsonl").read_text().splitlines()
    }
    ids = sorted(preds)
    ren = {d: d.replace("test_", "dev_") for d in ids}
    post = {ren[d]: preds[d] for d in ids}
    trs = {ren[d]: traces[d] for d in ids}
    pages = ev.load_ocr_pages(ids, ocr_root)
    ocr_pages = {ren[d]: v for d, v in pages.items()}
    zs_post = None
    if arm == "ft":
        zp = json.loads((zs_sub / "test_predictions.json").read_text())
        zs_post = {ren[d]: zp[d] for d in ids}
    new_ids = [ren[d] for d in ids]
    t = flags.build_tables(
        arm, new_ids, post, trs, ocr_pages, {ren[d]: rec[d] for d in ids}, zs_post
    )
    em = t.tv.emitted
    y = rng.random(len(em)) < 0.6
    p = np.where(em, np.clip(0.5 * y + 0.5 * rng.random(len(em)), 0.02, 0.98), np.nan)
    y_doc = np.array([i % 2 == 0 for i in range(len(new_ids))])
    sigs = {d: (s if s is not None else rng.random(SIG_DIM)) for d, s in t.sigs.items()}
    inp = freeze.Inputs(
        arm,
        "synthetic-world",
        t.tv,
        y,
        p,
        y_doc,
        new_ids,
        {d: i % 3 for i, d in enumerate(new_ids)},
        {d: f"g{i}" for i, d in enumerate(new_ids)},
        sigs,
        structure,
        kind,
        [],
        t.agree,
        {"kind": "synthetic"},
    )
    art, _ = freeze.fit_artifact(inp, n_boot=10)
    out = tmp_path / f"calibrator_{arm}_{structure}_{kind}.json"
    freeze.write_artifact(art, out)
    return out


@needs_schema
def test_stage_writes_flags_for_exactly_the_submission_docs_and_no_values(
    world: World, tmp_path: Path
) -> None:
    sub, ocr_root, shapes = world_submission(world, tmp_path)
    cal = calibrator_for(sub, ocr_root, "zs", tmp_path)
    before = (sub / "test_predictions.json").read_bytes()
    out = tmp_path / "flags.json"
    doc = flags.run_stage(
        submission_dir=sub,
        calibrator=cal,
        ocr_cache=ocr_root,
        out=out,
        shapes_file=shapes,
        expect_docs=len(TEST_IDS),
        say=lambda m: None,
    )
    assert out.is_file() and json.loads(out.read_text()) == doc
    assert (sub / "test_predictions.json").read_bytes() == before  # never touched
    assert sorted(doc["docs"]) == sorted(TEST_IDS) and doc["kind"] == flags.FLAGS_KIND
    assert doc["calibrator"]["sha256"] == flags.sha256_file(cal) and doc["arm"] == "zs"
    thr = doc["thresholds"]
    assert thr["field_target"] == 0.98 and set(thr["field_tau"]) == set(c2.FIELD_TYPES)
    assert "doc_tau" in thr and thr["nested_estimates"] is not None
    assert doc["inputs"]["test_predictions_sha256"] == predict.sha256_file(
        sub / "test_predictions.json"
    )
    # no extracted value anywhere: every string under docs is accept / review, and no prediction
    # value (any length >= 4) occurs among all the strings of the file
    assert set(flags.doc_string_values(doc["docs"])) <= flags.DOC_STRING_VALUES
    preds = json.loads((sub / "test_predictions.json").read_text())
    text = out.read_text()
    for d in preds.values():
        for v in [*d["header"].values(), *(c for r in d["line_items"] for c in r.values())]:
            if isinstance(v, str) and len(v) >= 4:
                assert f'"{v}"' not in text, "an extracted value leaked into the flags file"
    s = doc["summary"]
    assert s["n_emitted_fields"] == s["n_accept"] + s["n_review"] > 0
    # per-field entries: header fields by name, rows by index + field name
    d0 = doc["docs"]["test_0000"]
    assert {"p_fully_correct", "auto_accept", "header", "rows"} <= set(d0)
    assert all(e["flag"] in ("accept", "review", None) for e in d0["header"].values())
    assert [r["row_idx"] for r in d0["rows"]] == list(range(len(d0["rows"])))


@needs_schema
@pytest.mark.parametrize(("structure", "kind"), [("pooled", "gbm"), ("per_type", "lr")])
def test_stage_works_for_the_other_model_structures(
    world: World, tmp_path: Path, structure: str, kind: str
) -> None:
    sub, ocr_root, shapes = world_submission(world, tmp_path)
    cal = calibrator_for(sub, ocr_root, "zs", tmp_path, structure=structure, kind=kind)
    doc = flags.run_stage(
        submission_dir=sub,
        calibrator=cal,
        ocr_cache=ocr_root,
        shapes_file=shapes,
        expect_docs=len(TEST_IDS),
        say=lambda m: None,
    )
    assert doc["calibrator"]["structure"] == structure and doc["calibrator"]["kind"] == kind
    assert (sub / flags.FLAGS_NAME).is_file()  # default destination


@needs_schema
def test_stage_refusals(world: World, tmp_path: Path) -> None:
    sub, ocr_root, shapes = world_submission(world, tmp_path)
    cal = calibrator_for(sub, ocr_root, "zs", tmp_path)
    kw: dict[str, Any] = {
        "calibrator": cal,
        "ocr_cache": ocr_root,
        "shapes_file": shapes,
        "expect_docs": len(TEST_IDS),
        "say": lambda m: None,
    }

    # a REJECTED / unfinished folder is never flagged
    (sub / "test_predictions.json").rename(sub / "test_predictions.REJECTED.json")
    with pytest.raises(flags.FlagsError, match="no test_predictions.json"):
        flags.run_stage(submission_dir=sub, **kw)
    (sub / "test_predictions.REJECTED.json").rename(sub / "test_predictions.json")

    # the predictions are not what the production post path makes of the trace
    preds = json.loads((sub / "test_predictions.json").read_text())
    saved = (sub / "test_predictions.json").read_text()
    preds["test_0000"]["header"]["buyer_name"] = "Tampered"
    (sub / "test_predictions.json").write_text(json.dumps(preds))
    with pytest.raises(flags.FlagsError, match="not what the production post-processing"):
        flags.run_stage(submission_dir=sub, **kw)
    (sub / "test_predictions.json").write_text(saved)

    # wrong number of documents / wrong arm (a ZS calibrator on a fine-tuned submission)
    with pytest.raises(flags.FlagsError, match="expected 7"):
        flags.run_stage(submission_dir=sub, **{**kw, "expect_docs": 7})
    man = json.loads((sub / "manifest.json").read_text())
    man["model"]["id"] = "Qwen/x+lora:abcdef012345"
    (sub / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(flags.FlagsError, match="fine-tuned but the calibrator is the ZS one"):
        flags.run_stage(submission_dir=sub, **kw)
    # the ocr cache is part of the post path: another cache changes R2 -> the predictions differ
    with pytest.raises(flags.FlagsError, match="not what the production post-processing"):
        flags.run_stage(
            submission_dir=sub,
            **{**kw, "ocr_cache": tmp_path / "empty_cache"},
            check_manifest=False,
        )
    # the refusal exit codes of the CLI
    argv = [
        "--submission-dir",
        str(sub),
        "--calibrator",
        str(cal),
        "--ocr-cache",
        str(ocr_root),
        "--shapes-file",
        str(shapes),
        "--expect-docs",
        str(len(TEST_IDS)),
    ]
    assert flags.main(argv) == 1  # the arm mismatch above is still in place
    assert flags.main([*argv, "--skip-manifest-check"]) == 0


@needs_schema
def test_rules_jsonl_is_required_and_the_two_code_shas_are_recorded(
    world: World, tmp_path: Path
) -> None:
    sub, ocr_root, shapes = world_submission(world, tmp_path)
    cal = calibrator_for(sub, ocr_root, "zs", tmp_path)
    kw: dict[str, Any] = {
        "submission_dir": sub,
        "calibrator": cal,
        "ocr_cache": ocr_root,
        "shapes_file": shapes,
        "expect_docs": len(TEST_IDS),
        "say": lambda m: None,
    }
    man_path, rules = sub / "manifest.json", sub / "rules.jsonl"
    man = json.loads(man_path.read_text())

    # (b) present and equal still passes; both SHA keys (distinct names) are recorded
    doc = flags.run_stage(**kw)
    inp = doc["inputs"]
    assert inp["submission_code_sha"] == man["code_sha"]
    assert inp["local_head_sha"] == inp["code_sha"] == spike.git_commit()
    assert inp["rules_jsonl_sha256"] == flags.sha256_file(rules)

    # a manifest without a code_sha records null (nothing invented)
    no_sha = {k: v for k, v in man.items() if k != "code_sha"}
    man_path.write_text(json.dumps(no_sha))
    assert flags.run_stage(**kw)["inputs"]["submission_code_sha"] is None
    man_path.write_text(json.dumps(man))

    # (c) present but tampered still refuses
    saved = rules.read_text()
    rules.write_text("\n".join(saved.splitlines()[1:]) + "\n")
    with pytest.raises(flags.FlagsError, match="rules.jsonl differs"):
        flags.run_stage(**kw)

    # (a) missing is refused, naming the file
    rules.unlink()
    with pytest.raises(flags.FlagsError, match=r"REFUSED: .* has no rules\.jsonl"):
        flags.run_stage(**kw)
    # (d) no switches record in the manifest (or switches not all OFF): still refused
    for post in ({}, {"switches": {"r1": False, "r2": True, "r3": False}}, None):
        m = {k: v for k, v in man.items() if k != "post_rules"}
        if post is not None:
            m["post_rules"] = post
        man_path.write_text(json.dumps(m))
        with pytest.raises(flags.FlagsError, match=r"has no rules\.jsonl"):
            flags.run_stage(**kw)
    man_path.unlink()  # no manifest at all, manifest check skipped: refused as well
    with pytest.raises(flags.FlagsError, match=r"has no rules\.jsonl"):
        flags.run_stage(**kw, check_manifest=False)
    # all rules recorded OFF and no file: the only legitimate absence, the stage proceeds
    assert flags._all_rules_off(man_path) is False
    off = {**man, "post_rules": {"switches": dict.fromkeys(("r1", "r2", "r3"), False)}}
    man_path.write_text(json.dumps(off))
    assert flags._all_rules_off(man_path) is True
    assert flags.run_stage(**kw)["inputs"]["rules_jsonl_sha256"] is None


@needs_schema
def test_fine_tuned_arm_needs_the_strictly_reusable_zero_shot_traces(
    world: World,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from test_oof import MergedMock

    ft_sub, ocr_root, shapes = world_submission(
        world,
        tmp_path,
        "v2_ft",
        backend=MergedMock(world.gold, world.cfg, null_header=("buyer_name",)),
        model_id="Qwen/x+lora:abcdef012345",
    )
    # the zero-shot folder: a plain mock run, its own run folder
    zs_world = World(world.data, tmp_path / "zs_runs", world.gold, world.cfg)
    zs_sub, _, _ = world_submission(zs_world, tmp_path / "zs", "v0_x")
    cal = calibrator_for(ft_sub, ocr_root, "ft", tmp_path, zs_sub=zs_sub)
    base = {
        "submission_dir": ft_sub,
        "calibrator": cal,
        "ocr_cache": ocr_root,
        "shapes_file": shapes,
        "expect_docs": len(TEST_IDS),
        "say": lambda m: None,
    }
    # (1) no zero-shot folder at all: refused, nothing decoded
    with pytest.raises(flags.ZsReuseRefused, match="no zero-shot inference"):
        flags.run_stage(**base)
    # (2) a folder that fails the strict reuse decision (the synthetic run is not the 04 run):
    # the real decide_reuse refuses, and the stage says so instead of decoding anything
    with pytest.raises(flags.ZsReuseRefused, match="cannot be reused"):
        flags.run_stage(**base, zs_v0_dir=zs_sub, data_root=world.data)
    argv = [
        "--submission-dir",
        str(ft_sub),
        "--calibrator",
        str(cal),
        "--ocr-cache",
        str(ocr_root),
        "--shapes-file",
        str(shapes),
        "--expect-docs",
        str(len(TEST_IDS)),
        "--zs-v0-dir",
        str(zs_sub),
        "--data-root",
        str(world.data),
    ]
    assert flags.main(argv) == 3 and "cannot be reused" in capsys.readouterr().err
    # (3) with the decision allowed the stage runs and records the reuse facts
    monkeypatch.setattr(
        reuse,
        "evaluate_reuse",
        lambda *a, **k: reuse.ReuseDecision(
            True, [], {"v0_code_sha": "a" * 40, "decode_path": {"files": 17}}
        ),
    )
    doc = flags.run_stage(**base, zs_v0_dir=zs_sub, data_root=world.data)
    assert doc["arm"] == "ft" and doc["inputs"]["zs_reuse"]["decode_path_files"] == 17
    assert sorted(doc["docs"]) == sorted(TEST_IDS)
    # (4) a v0 folder that does not cover the same documents is refused even when "allowed"
    (zs_sub / "trace.jsonl").write_text(
        "\n".join((zs_sub / "trace.jsonl").read_text().splitlines()[:-1]) + "\n"
    )
    with pytest.raises(flags.ZsReuseRefused, match="exactly the submission"):
        flags.run_stage(**base, zs_v0_dir=zs_sub, data_root=world.data)


def test_schema_constants_and_vocabulary() -> None:
    assert flags.tkey(0.98) == "0.98" and {"accept", "review"} == flags.DOC_STRING_VALUES
    assert set(flags.FIELD_TARGETS) == {0.95, 0.98, 0.99} and flags.PRIMARY_TARGET == 0.98
    assert len(SPECS) == len(TEST_IDS)
    assert SCHEMA.name == "schema.json"
    assert RuleConfig().as_dict() == {"r1": True, "r2": True, "r3": True}
