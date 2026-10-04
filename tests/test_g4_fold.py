"""scripts/g4_fold.py on a synthetic miniature OOF + zero-shot world (no model, no real document).

3 folds x (2 invoice docs + 1 waybill doc). The ZS arm gets the invoice quantity wrong; the FT arm
is right. Both arms leave the waybill carrier empty while the model's own supplier_name slot holds
it, so R1 fires on both. The OCR cache, the locator and the row-error classification are stubbed
(their own tests live elsewhere); everything else (production path, paired bootstrap, scorer,
validators of rule_gate) is real.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from shipdoc import eval as ev
from shipdoc import postrules, rules

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("g4_fold", ROOT / "scripts" / "g4_fold.py")
g4f = importlib.util.module_from_spec(_spec)
sys.modules["g4_fold"] = g4f
_spec.loader.exec_module(g4f)
rg = g4f.rg

FOLD0 = ["a1", "a2", "wa"]


def inv_gold(d: str) -> dict[str, Any]:
    return {
        "doc_id": d,
        "doc_type": "invoice",
        "header": {
            "invoice_number": f"INV-{d}",
            "invoice_date": "2026-01-02",
            "supplier_name": f"Sup {d}",
            "buyer_name": "Buyer",
            "ship_to_name": "Ship",
            "currency": "USD",
            "total_amount": "10.50",
            "awb_number": None,
        },
        "line_items": [
            {
                "supplier_part_number": f"P{d}{i}",
                "customer_part_number": None,
                "purchase_order": f"1234567{i}",
                "quantity": "5",
            }
            for i in range(2)
        ],
    }


def wb_gold(d: str) -> dict[str, Any]:
    return {
        "doc_id": d,
        "doc_type": "waybill",
        "header": {
            "carrier": "Acme Air",
            "mawb": "176-12345678",
            "hawb": None,
            "origin_airport": "FRA",
            "destination_airport": "SIN",
            "shipper_name": "Shipper",
            "consignee_name": "Consignee",
            "pieces": "3",
            "gross_weight_kg": "12.5",
        },
        "line_items": [],
    }


def world() -> dict[str, Any]:
    doc_fold = dict(a1=0, a2=0, wa=0, b1=1, b2=1, wb=1, c1=2, c2=2, wc=2)
    gold = {d: (wb_gold(d) if d.startswith("w") else inv_gold(d)) for d in doc_fold}
    groups = {d: ("wb_g0" if d.startswith("w") else "inv_g0") + str(doc_fold[d]) for d in gold}
    folds = {
        "folds": [
            {"fold": k, "val_doc_ids": sorted(d for d, f in doc_fold.items() if f == k)}
            for k in range(3)
        ],
        "doc_fold": doc_fold,
    }
    return {"gold": gold, "doc_fold": doc_fold, "groups": groups, "folds": folds}


def trace_of(d: str, gold: dict[str, Any], arm: str) -> dict[str, Any]:
    """A trace line: the arm's merged prediction and the page the model saw."""
    pred = copy.deepcopy(gold[d])
    if d.startswith("w"):
        pred["header"]["carrier"] = None  # the model wrote the carrier into supplier_name
        parsed = {"header": {"supplier_name": "Acme Air", "carrier": None}, "line_items": []}
    else:
        parsed = {"header": {}, "line_items": []}
        if arm == "zs":
            for r in pred["line_items"]:
                r["quantity"] = "7"
    page = {"raw_text": json.dumps(parsed), "parsed": parsed, "json_valid": True}
    return {"doc_id": d, "pages": [page], "prediction": pred}


def make_run(
    tmp: Path, name: str, gold: dict[str, Any], ids: list[str], arm: str,
    oof: Any = None, status: str = "complete", config_hash: str | None = "h1",
) -> Path:  # fmt: skip
    run = tmp / name
    run.mkdir()
    traces = [trace_of(d, gold, arm) for d in ids]
    (run / "trace.jsonl").write_text(
        "".join(json.dumps(t) + "\n" for t in traces), encoding="utf-8"
    )
    saved = g4f.rv1.finalize({t["doc_id"]: t["prediction"] for t in traces})
    (run / "predictions.json").write_text(json.dumps(saved), encoding="utf-8")
    (run / "progress.json").write_text(json.dumps({"status": status}), encoding="utf-8")
    man: dict[str, Any] = {"code_sha": "c" * 40}
    if config_hash is not None:  # "h1" = the 1260-token config, "hn" = native (miniature hashes)
        man["config"] = {"name": "cfg", "hash": config_hash}
    if oof == "ok":
        man["oof"] = {
            "fold": 0, "adapter_sha256": "a" * 64, "train_code_sha": "b" * 40,
            "verification": {"ok": True}, "n_inference_docs": len(ids),
            "zero_shot_run": "zs_run", "batch": {"used": 4},
            "guard": {"ran": True, "fallback_to_1": False},
        }  # fmt: skip
    elif oof is not None:
        man["oof"] = oof
    (run / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    return run


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    w = world()
    root = tmp_path / "repo"
    (root / "splits").mkdir(parents=True)
    (root / "meta").mkdir()
    (root / "reports").mkdir()
    (root / "splits" / "folds.json").write_text(json.dumps(w["folds"]), encoding="utf-8")
    (root / "meta" / "supplier_groups.json").write_text(json.dumps(w["groups"]), encoding="utf-8")
    monkeypatch.setattr(g4f, "ROOT", root)
    monkeypatch.setattr(rg, "ROOT", root)
    labels = [copy.deepcopy(g) for g in w["gold"].values()]
    monkeypatch.setattr(
        g4f.meta_mod, "load_labels", lambda split: labels if split == "train" else []
    )
    monkeypatch.setattr(g4f, "load_gold", lambda ids: {d: w["gold"][d] for d in ids})
    monkeypatch.setattr(
        g4f,
        "load_meta",
        lambda ids: [
            {"doc_id": d, "scanned": d.endswith("1"), "repeated_parts": False} for d in ids
        ],
    )
    monkeypatch.setattr(rg, "doc_pages", lambda d, root=None: [object()])  # 1 page, as the traces
    monkeypatch.setattr(
        postrules, "ocr_text_for_doc", lambda d, n=None, root=None: ("no mawb", None)
    )
    monkeypatch.setattr(g4f, "build_indexes", lambda ids, cache: dict.fromkeys(ids))
    # stand-ins for the OCR-dependent row-error classification
    monkeypatch.setattr(
        g4f.rde,
        "diagnose_doc",
        lambda d, p, g, t, i, c, sc, tr=False: [
            g4f.rde.Unit(d, "g", False, False, 1, 2, "pair", "spn_misread"),
            g4f.rde.Unit(d, "g", False, False, 1, 2, "gold", "missing_other"),
        ],
    )
    monkeypatch.setattr(g4f.rde, "waybill_over_nulls", lambda *a, **k: {"cells": [], "control": {}})
    zs = make_run(tmp_path, "zs_run", w["gold"], sorted(w["gold"]), "zs")
    ft = make_run(tmp_path, "oof_fold0_abc1234", w["gold"], FOLD0, "ft", oof="ok")
    return {**w, "tmp": tmp_path, "zs": zs, "ft": ft, "root": root}


def argv(env: dict[str, Any], *extra: str) -> list[str]:
    out = env["tmp"] / "g4.md"
    return [
        "--oof-run-dir", str(env["ft"]), "--zs-run-dir", str(env["zs"]), "--out", str(out),
        "--n-boot", "20", *extra,
    ]  # fmt: skip


def test_report_sections_verdicts_and_sign_convention(env: dict[str, Any]) -> None:
    js = env["tmp"] / "g4.json"
    assert g4f.main(argv(env, "--fold", "0", "--json-out", str(js))) == 0
    text = (env["tmp"] / "g4.md").read_text(encoding="utf-8")
    for section in (
        "## Provenance", "## Verdicts", "## Headline: FT + rules vs ZS + rules",
        "## Secondary: raw vs raw", "## Slices", "## Over-nulls and false fills",
        "## Rules effect per arm", "## Row-error cause table", "## Limits",
    ):  # fmt: skip
        assert section in text
    assert (
        "G4 INTERIM VERDICT (interim, one fold, not the final G4 decision): NO REGRESSION" in text
    )
    assert "G4 SPEC CLAUSE (interim, one fold, not the G4 decision)" in text
    assert "invoice (HEADLINE slice)" in text and "scanned" in text and "digital" in text
    assert "UNVERIFIED" in text and "delta FT - ZS" in text and "bootstrap" in text
    res = json.loads(js.read_text(encoding="utf-8"))
    d = res["rules_pair"]["subsets"]["all"]["paired_delta_oof_minus_zero_shot"]["OVERALL"]
    assert d["delta"] > 0  # FT (right quantities) minus ZS (wrong quantities)
    assert d["b"] > d["a"]
    assert res["n_docs"] == 3 and res["fold"] == 0
    assert res["verdicts"]["spec_rules"]["verdict"] in ("FINE-TUNED KEPT", "FINE-TUNED NOT KEPT")


def test_report_holds_no_gold_value_and_no_doc_id(env: dict[str, Any]) -> None:
    g4f.main(argv(env))
    text = (env["tmp"] / "g4.md").read_text(encoding="utf-8")
    for needle in ("INV-a1", "Acme Air", "176-12345678", "Pa10", "Sup a1"):
        assert needle not in text  # gold / predicted values
    assert not any(f" {d} " in text or f"`{d}`" in text for d in env["gold"])  # doc ids


def test_rules_are_applied_to_both_arms_not_just_one(env: dict[str, Any]) -> None:
    js = env["tmp"] / "g4.json"
    g4f.main(argv(env, "--json-out", str(js)))
    res = json.loads(js.read_text(encoding="utf-8"))
    for arm in ("zs", "ft"):
        r1 = res["rules_effect"][arm]["per_rule"]["R1"]
        assert (r1["touched_docs"], r1["fixed"], r1["broken"]) == (1, 1, 0)
        assert res["rules_effect"][arm]["per_rule"]["R3"]["touched_docs"] == 0
    # raw view has the carrier over-null in BOTH arms, the rules view in NEITHER
    ov = {a: res["blocks"][a]["over_null"]["header_over_null_by_field"] for a in res["blocks"]}
    assert ov["zs_raw"].get("carrier") == 1 and ov["ft_raw"].get("carrier") == 1
    assert "carrier" not in ov["zs_rules"] and "carrier" not in ov["ft_rules"]
    # both views are different tables: rules lift OVERALL on each arm
    for arm in ("zs", "ft"):
        assert res["rules_effect"][arm]["delta"] > 0


def test_over_null_counting_follows_the_arm(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A header null only in the FT arm shows up only in the FT column and in the verdict clause."""
    ids = FOLD0
    ft = make_run(env["tmp"], "ft_null", env["gold"], ids, "ft", oof="ok")
    lines = [json.loads(x) for x in (ft / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    for t in lines:
        if t["doc_id"] == "a1":
            t["prediction"]["header"]["buyer_name"] = None
    (ft / "trace.jsonl").write_text("".join(json.dumps(t) + "\n" for t in lines), encoding="utf-8")
    saved = g4f.rv1.finalize({t["doc_id"]: t["prediction"] for t in lines})
    (ft / "predictions.json").write_text(json.dumps(saved), encoding="utf-8")
    js = env["tmp"] / "g4.json"
    argv_ = argv(env, "--json-out", str(js))
    argv_[1] = str(ft)
    g4f.main(argv_)
    res = json.loads(js.read_text(encoding="utf-8"))
    b = res["blocks"]
    assert b["ft_rules"]["over_null"]["header_over_null_by_field"] == {"buyer_name": 1}
    assert b["zs_rules"]["over_null"]["header_over_null_by_field"] == {}
    assert res["verdicts"]["interim_rules"]["oof_over_null"] == 1
    assert res["verdicts"]["interim_rules"]["zs_over_null"] == 0


def test_cause_summary_counts_units_slots_and_waybill_cells(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gold = {"a1": inv_gold("a1"), "a2": inv_gold("a2"), "wa": wb_gold("wa")}
    pred = copy.deepcopy(gold)
    for r in pred["a1"]["line_items"]:  # po sits in the cpn slot of the same row
        r["customer_part_number"], r["purchase_order"] = r["purchase_order"], None
    pred["a2"]["line_items"][0]["purchase_order"] = None
    sc = ev.load_scorer()
    monkeypatch.setattr(
        g4f.rde,
        "diagnose_doc",
        lambda d, p, g, t, i, c, sc, tr=False: [
            g4f.rde.Unit(d, "g", False, False, 1, 2, "pair", "column_shift"),
        ],
    )

    def cell(field: str, state: str, alt: list[str], uniq: Any) -> dict[str, Any]:
        return {
            "field": field, "raw_state": state, "alt_slots": alt, "pattern_unique_correct": uniq,
        }  # fmt: skip

    cells = [
        cell("mawb", "null_emitted", [], True),
        cell("mawb", "key_missing", ["x"], None),
        cell("carrier", "value_emitted", [], None),
    ]
    monkeypatch.setattr(g4f.rde, "waybill_over_nulls", lambda *a, **k: {"cells": cells})
    traces = {d: {"pages": []} for d in gold}
    meta = {d: {"scanned": False, "repeated_parts": False} for d in gold}
    groups = dict.fromkeys(gold, "g")
    out = g4f.cause_summary(pred, gold, traces, dict.fromkeys(gold), meta, groups, {"a2"}, sc)
    assert out["n_invoice_docs"] == 2 and out["n_truncated_invoice_docs"] == 1
    assert out["causes"] == {"column_shift": 2} and out["side"] == {"pair": 2, "gold": 0, "pred": 0}
    assert out["slots"] == {"po_in_cpn_slot": 2, "po_over_null": 1}
    assert out["n_waybill_docs"] == 1
    assert out["waybill"]["mawb"] == {
        "cells": 2, "null_emitted": 1, "key_missing": 1, "alt_slot": 1, "pattern_unique_correct": 1,
    }  # fmt: skip
    assert out["waybill"]["carrier"]["cells"] == 1 and out["waybill"]["hawb"]["cells"] == 0


# ---- fail-closed paths -------------------------------------------------------------------------


def _tamper_trace(run: Path, fn: Any) -> None:
    lines = [json.loads(x) for x in (run / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    fn(lines)
    (run / "trace.jsonl").write_text("".join(json.dumps(t) + "\n" for t in lines), encoding="utf-8")


def test_refuses_without_oof_section_and_names_plumbing(env: dict[str, Any]) -> None:
    man = json.loads((env["ft"] / "manifest.json").read_text(encoding="utf-8"))
    man.pop("oof")
    (env["ft"] / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    with pytest.raises(SystemExit, match="no `oof` section"):
        g4f.main(argv(env))
    assert not (env["tmp"] / "g4.md").exists()


@pytest.mark.parametrize(
    ("oof", "match"),
    [
        ({"fold": 0}, "verification is not ok"),
        ({"fold": 0, "verification": {"ok": True}}, "lacks the adapter sha"),
        ({"verification": {"ok": True}}, "no integer `fold`"),
    ],
)
def test_refuses_bad_oof_sections(env: dict[str, Any], oof: Any, match: str) -> None:
    man = json.loads((env["ft"] / "manifest.json").read_text(encoding="utf-8"))
    man["oof"] = oof
    (env["ft"] / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    with pytest.raises(SystemExit, match=match):
        g4f.main(argv(env))
    assert not (env["tmp"] / "g4.md").exists()


def test_refuses_fold_flag_contradicting_the_manifest(env: dict[str, Any]) -> None:
    with pytest.raises(SystemExit, match="contradicts the manifest"):
        g4f.main(argv(env, "--fold", "1"))


def test_refuses_incomplete_runs(env: dict[str, Any]) -> None:
    (env["ft"] / "progress.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="not complete"):
        g4f.main(argv(env))
    (env["ft"] / "progress.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    (env["zs"] / "progress.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="zero-shot .*not complete"):
        g4f.main(argv(env))
    assert not (env["tmp"] / "g4.md").exists()


def test_refuses_wrong_doc_ids_in_the_oof_run(env: dict[str, Any]) -> None:
    short = make_run(env["tmp"], "oof_short", env["gold"], ["a1", "a2"], "ft", oof="ok")
    extra = make_run(env["tmp"], "oof_extra", env["gold"], [*FOLD0, "b1"], "ft", oof="ok")
    for run in (short, extra):
        a = argv(env)
        a[1] = str(run)
        with pytest.raises(SystemExit, match="not the fold's docs"):
            g4f.main(a)
    assert not (env["tmp"] / "g4.md").exists()


def test_refuses_a_zero_shot_run_that_lacks_fold_docs(env: dict[str, Any]) -> None:
    zs = make_run(env["tmp"], "zs_partial", env["gold"], ["a1", "wa"], "zs")
    a = argv(env)
    a[3] = str(zs)
    with pytest.raises(SystemExit, match="zero-shot run"):
        g4f.main(a)


def test_refuses_missing_trace_file_and_missing_trace_doc(env: dict[str, Any]) -> None:
    _tamper_trace(env["ft"], lambda lines: lines.pop())
    with pytest.raises(SystemExit, match="trace.jsonl"):
        g4f.main(argv(env))
    (env["ft"] / "trace.jsonl").unlink()
    with pytest.raises(SystemExit, match="missing"):
        g4f.main(argv(env))


def test_refuses_missing_gold(env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    def no_gold(ids: Any) -> Any:
        raise SystemExit("1 fold docs have no gold label")

    monkeypatch.setattr(g4f, "load_gold", no_gold)
    with pytest.raises(SystemExit, match="no gold"):
        g4f.main(argv(env))


def test_load_gold_fails_closed_on_a_doc_without_label(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "dev" / "labels").mkdir(parents=True)
    (tmp_path / "dev" / "labels" / "dev_0001.json").write_text(
        json.dumps(inv_gold("dev_0001")), encoding="utf-8"
    )
    monkeypatch.setattr(g4f.paths, "data_dir", lambda: tmp_path)
    assert list(g4f.load_gold(["dev_0001"])) == ["dev_0001"]
    with pytest.raises(SystemExit, match="1 fold docs have no gold"):
        g4f.load_gold(["dev_0001", "dev_0002"])


def test_refuses_missing_ocr_pages_for_a_waybill(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rg, "doc_pages", lambda d, root=None: [])
    with pytest.raises(SystemExit, match="OCR cache lacks waybill pages"):
        g4f.main(argv(env))
    assert not (env["tmp"] / "g4.md").exists()


def test_refuses_traces_that_already_carry_rule_records(env: dict[str, Any]) -> None:
    _tamper_trace(env["ft"], lambda lines: lines[0].update(rules={"changes": []}))
    with pytest.raises(SystemExit, match="not a raw arm"):
        g4f.main(argv(env))


def test_refuses_when_all_rules_off_does_not_reproduce_the_saved_predictions(
    env: dict[str, Any],
) -> None:
    pred = json.loads((env["ft"] / "predictions.json").read_text(encoding="utf-8"))
    pred["a1"]["header"]["buyer_name"] = "tampered"
    (env["ft"] / "predictions.json").write_text(json.dumps(pred), encoding="utf-8")
    with pytest.raises(SystemExit, match="all rules OFF != predictions.json"):
        g4f.main(argv(env))
    assert not (env["tmp"] / "g4.md").exists()


def test_refuses_when_r3_learner_overlaps_the_held_out_suppliers(env: dict[str, Any]) -> None:
    groups = {**env["groups"], "a1": "inv_g01"}  # a fold-0 doc shares fold 1's supplier group
    (env["root"] / "meta" / "supplier_groups.json").write_text(json.dumps(groups), encoding="utf-8")
    with pytest.raises(SystemExit, match="overlaps"):
        g4f.main(argv(env))
    assert not (env["tmp"] / "g4.md").exists()


def test_warns_when_the_zs_folder_is_not_the_one_the_manifest_names(env: dict[str, Any]) -> None:
    zs = make_run(env["tmp"], "renamed_zs", env["gold"], sorted(env["gold"]), "zs")
    a = argv(env)
    a[3] = str(zs)
    g4f.main(a)
    text = (env["tmp"] / "g4.md").read_text(encoding="utf-8")
    assert "WARNING" in text and "`zs_run`" in text and "`renamed_zs`" in text


def test_native_ft_with_native_zs_passes(env: dict[str, Any]) -> None:
    ft = make_run(env["tmp"], "ft_native", env["gold"], FOLD0, "ft", oof="ok", config_hash="hn")
    zs = make_run(env["tmp"], "zs_native", env["gold"], sorted(env["gold"]), "zs", config_hash="hn")
    a = argv(env)
    a[1], a[3] = str(ft), str(zs)
    assert g4f.main(a) == 0


@pytest.mark.parametrize(("ft_hash", "zs_hash"), [("hn", "h1"), ("h1", "hn"), (None, "h1")])
def test_refuses_a_mixed_resolution_pair_and_names_both(
    env: dict[str, Any], ft_hash: str | None, zs_hash: str
) -> None:
    ft = make_run(env["tmp"], "ft_x", env["gold"], FOLD0, "ft", oof="ok", config_hash=ft_hash)
    zs = make_run(env["tmp"], "zs_x", env["gold"], sorted(env["gold"]), "zs", config_hash=zs_hash)
    a = argv(env)
    a[1], a[3] = str(ft), str(zs)
    with pytest.raises(SystemExit, match=r"mixed-resolution.*FT run `ft_x`.*ZS run `zs_x`"):
        g4f.main(a)
    assert not (env["tmp"] / "g4.md").exists()


# ---- plumbing check ----------------------------------------------------------------------------


def test_plumbing_check_uses_the_zs_run_as_both_arms_and_writes_nothing(
    env: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    out = env["tmp"] / "never.md"
    rc = g4f.main(
        ["--zs-run-dir", str(env["zs"]), "--plumbing-check", "--fold", "0", "--out", str(out),
         "--n-boot", "20"]
    )  # fmt: skip
    text = capsys.readouterr().out
    assert rc == 0 and "!!!!!!!!" in text and "PLUMBING CHECK NOT A RESULT" in text
    assert not out.exists() and not (env["root"] / "reports" / "g4_fold0.md").exists()
    assert "Docs: 3 held-out docs of fold 0" in text and "## Row-error cause table" in text
    assert "delta FT - ZS (pts) [95% CI]" in text and "+0.00 [+0.00, +0.00]" in text


def test_plumbing_check_does_not_need_an_oof_section_but_oof_mode_requires_the_dir(
    env: dict[str, Any],
) -> None:
    with pytest.raises(SystemExit):
        g4f.main(["--zs-run-dir", str(env["zs"])])  # neither --oof-run-dir nor --plumbing-check


def test_default_report_path_is_per_fold(env: dict[str, Any]) -> None:
    a = argv(env)
    i = a.index("--out")
    del a[i : i + 2]
    g4f.main(a)
    assert (env["root"] / "reports" / "g4_fold0.md").is_file()


def test_honest_shapes_exclude_the_held_out_fold(env: dict[str, Any]) -> None:
    labels = [g | {"line_items": g["line_items"]} for g in env["gold"].values()]
    shapes, n = g4f.honest_shapes(0, labels, env["doc_fold"], env["groups"])
    assert n == 6 and isinstance(shapes, rules.SlotShapes)
