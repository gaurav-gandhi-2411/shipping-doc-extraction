"""scripts/freeze_calibrator.py: what is frozen, which tau, the refusals, the real v2 run folder.

The synthetic tests fit calibrators on the generative table of ``confidence.synthetic_data``
(header fields only) and a random document label: they test the mechanics, not any real
performance. The real-data test rebuilds the 500-document ZS calibration inputs (about 5 minutes of
feature building, CPU) and is opt-in: set ``SHIPDOC_FREEZE_REAL=1`` and run this file.
It skips when the run folders or the data are absent.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from test_flags import SIG_DIM, freeze, synthetic_inputs

from shipdoc import confidence as cf
from shipdoc import confidence_v2 as c2
from shipdoc import flags

ROOT = Path(__file__).resolve().parents[1]
ZS_RUN = Path(
    "D:/shipdoc/runs/zeroshot500/zeroshot500_qwen35_4b_img_only_keyed_42b812b"
)  # the 02 run
CAL_DIR = Path("D:/shipdoc/runs/calibration_v2/zeroshot500_qwen35_4b_img_only_keyed_42b812b")
needs_real = pytest.mark.skipif(
    os.environ.get("SHIPDOC_FREEZE_REAL") != "1"
    or not (ZS_RUN / "trace.jsonl").is_file()
    or not (CAL_DIR / "oof_fields_v2.csv").is_file()
    or not (ROOT / "data" / "train" / "labels").is_dir(),
    reason="opt-in (SHIPDOC_FREEZE_REAL=1) and needs the 02 run, calibration_v2 outputs and data/",
)


@pytest.fixture(autouse=True)
def _fast_novelty_reference(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The synthetic fits skip the KMeans k-search of `novelty_reference` (7 clusterings per fit,
    ~10 s): a reference of up to 8 signatures keeps every other code path as it is. The real-data
    test uses the real clustering."""
    if "real" not in request.node.name:

        def fast_ref(sigs: object) -> np.ndarray:
            return np.vstack(list(sigs)[:8])  # type: ignore[call-overload]

        real_table = cf.compute_novelty_table
        monkeypatch.setattr(cf, "novelty_reference", fast_ref)
        # the default argument of compute_novelty_table was bound at import: pass the fast one
        monkeypatch.setattr(
            cf,
            "compute_novelty_table",
            lambda sigs, folds, reference_fn=fast_ref: real_table(sigs, folds, reference_fn),
        )


def _freeze(arm: str = "zs", **kw: object) -> tuple[dict, dict, freeze.Inputs]:
    inp = synthetic_inputs(arm, **kw)  # type: ignore[arg-type]
    art, rep = freeze.fit_artifact(inp, n_boot=10)
    return art, rep, inp


@pytest.mark.parametrize("kind", ["lr", "gbm"])
def test_frozen_field_model_equals_a_fresh_fit_of_the_same_model(kind: str, tmp_path: Path) -> None:
    art, _, inp = _freeze(kind=kind)
    path = tmp_path / "cal.json"
    sha = freeze.write_artifact(art, path)
    cal = flags.load_calibrator(path, "zs")
    assert cal.sha256 == sha and cal.structure == "pooled"
    # the training design: the same builder, novelty against the OTHER folds
    tables = flags.Tables(inp.tv, inp.sigs, None, inp.doc_ids)
    nov_table = cf.compute_novelty_table(inp.sigs, inp.doc_fold)
    nov = {d: nov_table[inp.doc_fold[d]][d] for d in inp.doc_ids}
    X = flags.design("zs", tables, nov)
    em = inp.tv.emitted
    want = cf.fit_predict(kind, X[em], inp.y[em].astype(int), X)
    got = cal.p_field(X, inp.tv.types())
    assert np.allclose(got[em], want[em], atol=1e-9, rtol=0)
    assert art["field_model"]["models"]["all"]["kind"] == kind
    assert art["sklearn_version"] == flags.sklearn_version()
    text = path.read_text()
    assert text.endswith("\n") and text.count("\n") == 1  # one JSON line
    assert "syn_0" not in text  # no document id in the artifact


def test_stored_tau_is_the_pooled_oof_tau_and_nested_estimates_sit_beside_it() -> None:
    art, rep, inp = _freeze()
    em, types = inp.tv.emitted, inp.tv.types()
    thr = art["thresholds"]
    for t in flags.FIELD_TARGETS:
        for ft in c2.FIELD_TYPES:
            m = em & (types == ft)
            want = cf.select_tau(inp.p_oof[m], inp.y[m], t) if m.any() else None
            got = thr["field"]["tau"][flags.tkey(t)][ft]
            assert got == (None if want is None else float(want)), (t, ft)
            rec = thr["field"]["in_sample_at_tau"][flags.tkey(t)][ft]
            if got is not None:  # the in-sample precision at tau reaches the target by construction
                assert rec["precision"] >= t and 0 < rec["coverage"] <= 1
            else:
                assert rec["precision"] is None
    assert thr["field"]["tau"]["0.98"]["quantity"] is None  # no quantity rows: nothing to accept
    assert set(thr["doc"]["tau"]) == {"0.95", "0.98"}
    assert "POOLED OUT-OF-FOLD" in thr["meaning"] and "nested" in thr["meaning"].lower()
    ne = art["nested_estimates"]
    assert ne["field"] and {r["population"] for r in ne["doc"]} == {"doc"}
    met = thr["nested_target_met"]
    assert set(met["field"]["0.98"]) == set(c2.FIELD_TYPES) and set(met["doc"]) == {"0.95", "0.98"}
    assert all(isinstance(v, bool) for v in met["doc"].values())
    assert rep["tau_doc"] == thr["doc"]["tau"] and rep["n_models"] == 1


def test_per_type_structure_gives_every_type_a_model_even_without_training_rows() -> None:
    art, _, inp = _freeze(structure="per_type", kind="lr")
    models = art["field_model"]["models"]
    assert set(models) == set(c2.FIELD_TYPES)
    assert models["header"]["kind"] == "lr"
    assert models["quantity"] == {"kind": "constant", "p": 0.0, "n_fit": 0}  # never accepted
    assert art["thresholds"]["field"]["tau"]["0.98"]["quantity"] is None


def test_fine_tuned_arm_artifact_records_the_agreement_columns(tmp_path: Path) -> None:
    art, _, _ = _freeze("ft")
    assert art["arm"] == "ft" and art["variant"] == "v2_agree"
    assert art["feature_names"] == flags.feature_names("ft")
    assert art["feature_names"][-1] == "ag_doc_unaligned_frac"
    assert art["parity_probe"] == flags.parity_probe("ft")
    path = tmp_path / "ft.json"
    freeze.write_artifact(art, path)
    flags.load_calibrator(path, "ft")
    with pytest.raises(flags.FlagsError, match="arm"):
        flags.load_calibrator(path, "zs")


def test_the_novelty_reference_and_document_model_are_stored() -> None:
    art, _, inp = _freeze()
    ref = np.asarray(art["novelty"]["reference"])
    assert ref.ndim == 2 and ref.shape[1] == SIG_DIM
    assert art["novelty"]["n_signatures"] == len(inp.doc_ids)
    assert art["doc_model"]["kind"] in ("lr", "constant") and art["doc_features"]
    assert art["doc_model"]["n_features"] == len(c2.DOC_FEATURES)
    assert "OUT-OF-FOLD" in art["doc_model_note"]
    src = art["source"]
    assert src["n_docs"] == len(inp.doc_ids) and src["code_sha256_lf"]["src/shipdoc/flags.py"]


def test_the_fit_refuses_test_ids_and_missing_probabilities() -> None:
    inp = synthetic_inputs("zs", n_docs=45)
    bad = list(inp.doc_ids)
    bad[0] = "test_0001"
    with pytest.raises(c2.NoTestDataError):
        freeze.fit_artifact(freeze.Inputs(**{**inp.__dict__, "doc_ids": bad}), n_boot=5)
    p = inp.p_oof.copy()
    p[np.flatnonzero(inp.tv.emitted)[0]] = np.nan
    with pytest.raises(freeze.FreezeError, match="no OOF probability"):
        freeze.fit_artifact(freeze.Inputs(**{**inp.__dict__, "p_oof": p}), n_boot=5)


def test_the_document_oof_must_match_the_calibration_csv() -> None:
    inp = synthetic_inputs("zs", n_docs=45)
    _, rep = freeze.fit_artifact(inp, n_boot=5)
    assert "doc_oof_max_abs_diff_vs_csv" not in rep  # no CSV given
    from shipdoc import confidence_v2 as c2_

    Xd, ids = c2_.doc_table(inp.tv, inp.p_oof, inp.doc_ids)
    nov = cf.compute_novelty_table(inp.sigs, inp.doc_fold)
    p_doc = c2_.cross_fit_docs(Xd, inp.y_doc, ids, inp.doc_fold, nov, inp.groups)
    good = dict(zip(ids, p_doc.tolist(), strict=True))
    _, rep = freeze.fit_artifact(freeze.Inputs(**{**inp.__dict__, "p_doc_csv": good}), n_boot=5)
    assert rep["doc_oof_max_abs_diff_vs_csv"] < 1e-9
    wrong = {d: min(1.0, v + 0.01) for d, v in good.items()}
    with pytest.raises(freeze.FreezeError, match="not the one the calibration used"):
        freeze.fit_artifact(freeze.Inputs(**{**inp.__dict__, "p_doc_csv": wrong}), n_boot=5)


def test_oof_csv_join_is_strict(tmp_path: Path) -> None:
    inp = synthetic_inputs("zs", n_docs=45)
    tv, y = inp.tv, inp.y
    keys = [(k.doc_id, k.scope, k.field, k.row_idx) for k in tv.keys]
    lines = ["doc_id,scope,field,row_idx,emitted,p_correct_v2,y_correct"]
    for i, k in enumerate(keys):
        p = "" if not tv.emitted[i] else repr(float(inp.p_oof[i]))
        lines.append(
            ",".join([k[0], k[1], k[2], str(k[3]), str(int(tv.emitted[i])), p, str(int(y[i]))])
        )
    csv_path = tmp_path / "oof.csv"
    csv_path.write_text("\n".join(lines) + "\n")
    oof = freeze.read_oof_csv(csv_path, "p_correct_v2")
    p = freeze.align_oof(tv, y, oof)
    assert np.array_equal(p, inp.p_oof, equal_nan=True)
    missing = dict(oof)
    missing.pop(keys[0])
    with pytest.raises(freeze.FreezeError, match="not the one the calibration used"):
        freeze.align_oof(tv, y, missing)
    flipped = dict(oof)
    e, pp, yy = flipped[keys[0]]
    flipped[keys[0]] = (e, pp, not yy)
    with pytest.raises(freeze.FreezeError, match="labels differ"):
        freeze.align_oof(tv, y, flipped)
    em_flip = dict(oof)
    e, pp, yy = em_flip[keys[0]]
    em_flip[keys[0]] = (not e, pp, yy)
    with pytest.raises(freeze.FreezeError, match="emitted flags"):
        freeze.align_oof(tv, y, em_flip)


def test_nested_target_met_treats_nan_and_missing_as_not_met() -> None:
    nan = float("nan")
    assert not freeze._ge(nan, 0.9) and not freeze._ge(None, 0.9)
    assert freeze._ge(0.95, 0.95) and not freeze._ge(0.949, 0.95)
    rec = [
        {"population": "header", "target": 0.98, "precision_accepted": {"point": 0.99}},
        {"population": "quantity", "target": 0.98, "precision_accepted": {"point": nan}},
    ]
    met = freeze.nested_target_met(rec)["0.98"]
    assert met["header"] is True and met["quantity"] is False and met["purchase_order"] is None


def test_the_fine_tuned_inputs_builder_refuses_before_reading_anything() -> None:
    with pytest.raises(freeze.FreezeError, match="needs all three OOF runs"):
        freeze.build_ft_inputs([Path("a")], Path("zs"), Path("cal"))
    with pytest.raises(freeze.FreezeError):  # a repeated / wrong fold set is refused by the gate
        freeze.build_ft_inputs([Path("a")] * 3 + [Path("b")], Path("zs"), Path("cal"))
    assert freeze.main(["--arm", "ft", "--calibration-dir", "x"]) == 2
    assert freeze.main(["--arm", "zs", "--calibration-dir", "x"]) == 2


def test_the_fine_tuned_inputs_builder_refuses_mixed_resolution_runs(tmp_path: Path) -> None:
    def run(name: str, h: str) -> Path:
        d = tmp_path / name
        d.mkdir()
        (d / "manifest.json").write_text(
            json.dumps({"config": {"name": "cfg", "hash": h}}), encoding="utf-8"
        )
        return d

    ft = [run(f"oof_fold{k}", "hn") for k in range(3)]
    with pytest.raises(freeze.FreezeError, match=r"mixed-resolution.*ZS run `zs`"):
        freeze.build_ft_inputs(ft, run("zs", "h1"), tmp_path)
    ft[1] = run("oof_other", "h1")  # one fold at the other resolution than the ZS run
    with pytest.raises(freeze.FreezeError, match=r"mixed-resolution.*FT run 0"):
        freeze.build_ft_inputs(ft, tmp_path / "zs", tmp_path)


@needs_real
def test_freeze_on_the_real_v2_run_folder(tmp_path: Path) -> None:
    inp = freeze.build_zs_inputs(ZS_RUN, CAL_DIR)  # every key, flag and label is cross-checked
    assert inp.structure == "pooled" and inp.kind == "gbm"  # copied from calibration_v2.json
    art, rep = freeze.fit_artifact(inp, n_boot=50)
    assert rep["doc_oof_max_abs_diff_vs_csv"] < 1e-6
    out = tmp_path / "calibrator_zs.json"
    freeze.write_artifact(art, out)
    cal = flags.load_calibrator(out, "zs")  # sklearn version + hash + parity probe all pass
    assert art["source"]["n_docs"] == 500 and art["source"]["n_emitted_fields"] > 10000
    tau = art["thresholds"]["field"]["tau"]["0.98"]
    assert set(tau) == set(c2.FIELD_TYPES)
    # a field-type tau of the pooled OOF rule is a number or null, never a made-up default
    assert all(v is None or 0.0 <= v <= 1.0 for v in tau.values())
    assert out.stat().st_size < 1_000_000  # small enough for meta/
    assert cal.field_models["all"].kind == "gbm"
    assert not any(k in out.read_text() for k in ("train_0", "dev_0"))  # no document id
    # the final model orders the training rows (in-sample, so only a sanity floor)
    tables = flags.Tables(inp.tv, inp.sigs, None, inp.doc_ids)
    nov_table = cf.compute_novelty_table(inp.sigs, inp.doc_fold)
    nov = {d: nov_table[inp.doc_fold[d]][d] for d in inp.doc_ids}
    p = cal.p_field(flags.design("zs", tables, nov), inp.tv.types())
    em = inp.tv.emitted
    assert c2.auroc(p[em], inp.y[em].astype(int)) > 0.85
    assert json.loads(out.read_text())["arm"] == "zs"
