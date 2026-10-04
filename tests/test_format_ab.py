from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from shipdoc import format_ab as fab

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    not (ROOT / "assignment" / "score.py").is_file(), reason="assignment/score.py absent"
)

N_DOCS = 40


def _gold(i: int) -> dict[str, Any]:
    return {
        "doc_type": "invoice",
        "header": {
            "invoice_number": f"INV-{1000 + i}",
            "invoice_date": "2026-05-25",
            "supplier_name": f"Acme {i}",
            "buyer_name": "Buyer Ltd",
            "ship_to_name": "Ship Co",
            "currency": "USD",
            "total_amount": f"{100 + i}.50",
            "awb_number": None,
        },
        "line_items": [
            {
                "supplier_part_number": f"AX{i}",
                "customer_part_number": f"BX{i}",
                "purchase_order": f"CX{i}",
                "quantity": "3",
            }
        ],
    }


GOLD = {f"dev_{i:04d}": _gold(i) for i in range(N_DOCS)}


def _rotate(doc: dict[str, Any]) -> None:
    r = doc["line_items"][0]
    r["supplier_part_number"], r["customer_part_number"], r["purchase_order"] = (
        r["customer_part_number"],
        r["purchase_order"],
        r["supplier_part_number"],
    )


def _write(
    tmp: Path, name: str, rotate: range = range(0), bad_total: range = range(0), lat: float = 10.0
) -> Path:
    """A mock run dir: gold-equal predictions, with some docs rotated / given a wrong total."""
    pred, traces = {}, []
    for i, (d, g) in enumerate(GOLD.items()):
        p = copy.deepcopy(g)
        if i in rotate:
            _rotate(p)
        if i in bad_total:
            p["header"]["total_amount"] = "0.01"
            p["header"]["currency"] = "EUR"
        pred[d] = p
        meta = {"latency_s": lat + i % 5}
        traces.append({"doc_id": d, "pages": [{"page": 1, "json_valid": True, "meta": meta}]})
    out = tmp / name
    out.mkdir()
    (out / "predictions.json").write_text(json.dumps(pred))
    (out / "trace.jsonl").write_text("\n".join(json.dumps(t) for t in traces) + "\n")
    return out


def _ab(tmp: Path, compact: Path, keyed: Path) -> dict[str, Any]:
    return fab.compare_formats(compact, keyed, GOLD, "m", "img_only")


def test_row_f1_gain_with_ci_excluding_zero_chooses_keyed(tmp_path: Path) -> None:
    compact = _write(tmp_path, "c", rotate=range(0, N_DOCS, 2))  # half the rows broken
    keyed = _write(tmp_path, "k")
    res = _ab(tmp_path, compact, keyed)
    d = res["deltas"]["row_f1"]
    assert d["delta"] > 0 and d["ci_excludes_0"] and d["ci95"][0] > 0
    assert res["decision"] == "keyed" and "row F1 gain" in res["reason"]
    assert res["compact"]["rotations"] == 20 and res["keyed"]["rotations"] == 0
    assert set(res) >= {"model", "arm", "keyed", "compact", "deltas", "decision", "reason"}


def test_fewer_rotations_and_overall_not_worse_chooses_keyed(tmp_path: Path) -> None:
    compact = _write(tmp_path, "c", rotate=range(2))  # 2 of 40 docs: CI of the gain includes 0
    keyed = _write(tmp_path, "k")
    res = _ab(tmp_path, compact, keyed)
    assert not res["deltas"]["row_f1"]["ci_excludes_0"]
    assert res["keyed"]["rotations"] < res["compact"]["rotations"]
    assert res["decision"] == "keyed" and res["reason"].startswith("rotations 0 < 2")


def test_fewer_rotations_but_overall_clearly_worse_chooses_compact(tmp_path: Path) -> None:
    compact = _write(tmp_path, "c", rotate=range(1))
    keyed = _write(tmp_path, "k", bad_total=range(1, N_DOCS))  # 0 rotations, headers broken
    res = _ab(tmp_path, compact, keyed)
    o = res["deltas"]["OVERALL"]
    assert res["keyed"]["rotations"] < res["compact"]["rotations"]
    assert o["delta"] < 0 and o["ci_excludes_0"]
    assert res["decision"] == "compact" and res["reason"] == fab.NO_GAIN_REASON


def test_more_rotations_in_keyed_chooses_compact(tmp_path: Path) -> None:
    compact = _write(tmp_path, "c")
    keyed = _write(tmp_path, "k", rotate=range(3))
    res = _ab(tmp_path, compact, keyed)
    assert res["decision"] == "compact" and res["deltas"]["rotations"] == 3


def test_identical_runs_give_zero_delta_and_compact(tmp_path: Path) -> None:
    compact = _write(tmp_path, "c", rotate=range(3), lat=10.0)
    keyed = _write(tmp_path, "k", rotate=range(3), lat=20.0)
    res = _ab(tmp_path, compact, keyed)
    assert res["deltas"]["OVERALL"]["delta"] == 0 and res["deltas"]["row_f1"]["delta"] == 0
    assert res["decision"] == "compact" and res["reason"] == "no gain; compact on speed"
    # speed comes from trace.jsonl: latencies are 10..14 vs 20..24 (i % 5 offsets)
    assert res["compact"]["s_per_page_mean"] == pytest.approx(12.0)
    assert res["keyed"]["s_per_page_mean"] == pytest.approx(22.0)
    assert res["keyed"]["s_per_page_p95"] > res["keyed"]["s_per_page_mean"]
    assert set(res["keyed"]["line_item_field_accuracy"]) == {
        "supplier_part_number",
        "customer_part_number",
        "purchase_order",
        "quantity",
    }


def test_bootstrap_is_deterministic(tmp_path: Path) -> None:
    compact = _write(tmp_path, "c", rotate=range(5))
    keyed = _write(tmp_path, "k")
    assert _ab(tmp_path, compact, keyed) == _ab(tmp_path, compact, keyed)


def test_runs_on_different_docs_fail_closed(tmp_path: Path) -> None:
    compact = _write(tmp_path, "c")
    keyed = _write(tmp_path, "k")
    pred = json.loads((keyed / "predictions.json").read_text())
    del pred["dev_0000"]
    (keyed / "predictions.json").write_text(json.dumps(pred))
    with pytest.raises(ValueError, match="different docs"):
        _ab(tmp_path, compact, keyed)


def test_decide_rule_branches_directly() -> None:
    def deltas(row: tuple[float, float, float], ov: tuple[float, float, float]) -> dict[str, Any]:
        return {
            "row_f1": {"delta": row[0], "ci95": [row[1], row[2]]},
            "OVERALL": {"delta": ov[0], "ci95": [ov[1], ov[2]]},
        }

    flat = (0.0, -0.1, 0.1)
    assert fab.decide(deltas((0.1, 0.02, 0.2), flat), 5, 5)[0] == "keyed"  # row F1 branch
    assert fab.decide(deltas((0.1, -0.02, 0.2), flat), 5, 5)[0] == "compact"  # CI includes 0
    assert fab.decide(deltas((-0.1, -0.2, -0.02), flat), 1, 5)[0] == "keyed"  # CI incl. 0: ok
    # point delta negative, CI strictly below 0: OVERALL worse -> compact despite fewer rotations
    assert fab.decide(deltas(flat, (-0.05, -0.09, -0.01)), 1, 5)[0] == "compact"
    # point delta >= 0 counts as not worse even if the CI lower bound is above 0
    assert fab.decide(deltas(flat, (0.05, 0.01, 0.09)), 1, 5)[0] == "keyed"
    assert fab.decide(deltas(flat, flat), 5, 5) == ("compact", fab.NO_GAIN_REASON)


def test_format_table_has_decision_and_metrics(tmp_path: Path) -> None:
    res = _ab(tmp_path, _write(tmp_path, "c", rotate=range(2)), _write(tmp_path, "k"))
    text = fab.format_table(res)
    assert "decision: keyed" in text and "rotations" in text and "s_per_page_p95" in text


def test_cli_writes_json_and_fails_closed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("format_ab", ROOT / "scripts" / "format_ab.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    labels = tmp_path / "labels"
    labels.mkdir()
    for d, g in GOLD.items():
        (labels / f"{d}.json").write_text(json.dumps({"doc_id": d, **g}))
    c, k = _write(tmp_path, "c"), _write(tmp_path, "k")
    out = tmp_path / "ab.json"
    base = ["--model", "m", "--arm", "img_only", "--gold-dir", str(labels)]
    argv = ["--compact-dir", str(c), "--keyed-dir", str(k), *base, "--out", str(out), "--table"]
    assert mod.main(argv) == 0
    assert json.loads(out.read_text())["decision"] == "compact"
    assert "decision: compact" in capsys.readouterr().out
    assert mod.main(["--compact-dir", str(c), "--keyed-dir", str(tmp_path / "none"), *base]) == 2
