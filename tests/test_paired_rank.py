from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCORER = ROOT / "assignment" / "score.py"
pytestmark = pytest.mark.skipif(not SCORER.is_file(), reason="assignment/score.py absent")

spec = importlib.util.spec_from_file_location("paired_rank", ROOT / "scripts" / "paired_rank.py")
pr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pr)

N_DOCS = 40


def _gold(i: int) -> dict[str, Any]:
    return {
        "doc_id": f"dev_{i:04d}",
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
                "supplier_part_number": f"P{i}",
                "customer_part_number": "CP-1",
                "purchase_order": None,
                "quantity": "3",
            }
        ],
    }


@pytest.fixture
def world(tmp_path: Path) -> dict[str, Any]:
    labels = tmp_path / "labels"
    labels.mkdir()
    gold = {}
    for i in range(N_DOCS):
        g = _gold(i)
        gold[g["doc_id"]] = g
        (labels / f"{g['doc_id']}.json").write_text(json.dumps(g))
    docs = tmp_path / "docs.json"
    docs.write_text(json.dumps(list(gold)))
    return {"tmp": tmp_path, "labels": labels, "gold": gold, "docs": docs, "runs": tmp_path / "r"}


def _write_run(
    world: dict[str, Any], config: str, wrong_every: int | None, keep: int | None = None
) -> None:
    """Predictions equal to gold, except every `wrong_every`-th doc gets a wrong total."""
    pred = {}
    for i, (d, g) in enumerate(world["gold"].items()):
        p = copy.deepcopy(g)
        del p["doc_id"]
        if wrong_every and i % wrong_every == 0:
            p["header"]["total_amount"] = "0.01"
            p["header"]["currency"] = "EUR"
        pred[d] = p
    if keep is not None:
        pred = dict(list(pred.items())[:keep])
    out = world["runs"] / f"spike40_{config}_abc1234"
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions.json").write_text(json.dumps(pred))


def _argv(world: dict[str, Any], configs: list[str], out: Path | None = None) -> list[str]:
    argv = [
        "--runs-dir", str(world["runs"]), "--group", "spike40", "--sha7", "abc1234",
        "--docs", str(world["docs"]), "--gold-dir", str(world["labels"]),
        "--configs", *configs,
    ]  # fmt: skip
    return argv + (["--out", str(out)] if out else [])


def _rank(world: dict[str, Any], configs: list[str]) -> dict[str, Any]:
    return pr.rank(world["runs"], "spike40", "abc1234", configs, list(world["gold"]), world["gold"])


def test_clear_winner_is_picked_without_dev100(
    world: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    _write_run(world, "good", None)
    _write_run(world, "bad", 2)  # half the docs have wrong fields
    _write_run(world, "mid", 8)
    out = world["tmp"] / "rank.json"
    assert pr.main(_argv(world, ["bad", "good", "mid"], out)) == 0
    res = json.loads(out.read_text())
    assert [r["config"] for r in res["ranking"]] == ["good", "mid", "bad"]
    assert res["top2"] == ["good", "mid"]
    assert res["ranking"][0]["OVERALL"] == 1.0
    assert res["delta"] > 0
    assert res["delta_ci_excludes_0"] is True and res["p_le_0"] < 0.05
    assert res["n_resamples"] == 2000 and res["seed"] == 42 and res["n_docs"] == N_DOCS
    assert json.loads(capsys.readouterr().out) == res


def test_cis_overlap_decides_dev100_vs_pick_top1(world: dict[str, Any]) -> None:
    _write_run(world, "a", None)
    _write_run(world, "b", 20)  # 2 of 40 docs differ
    near = _rank(world, ["a", "b"])
    lo, hi = near["ranking"][1]["ci95"]
    assert near["cis_overlap"] is True and near["decision"] == "run_dev100"
    assert lo <= near["ranking"][0]["ci95"][1] and near["ranking"][0]["ci95"][0] <= hi
    _write_run(world, "c", 2)  # far behind: the perfect run's CI is [1, 1]
    far = _rank(world, ["a", "c"])
    assert far["cis_overlap"] is False and far["decision"] == "pick_top1"
    assert far["delta_ci_excludes_0"] is True


def test_rank_is_deterministic(world: dict[str, Any]) -> None:
    _write_run(world, "a", 10)
    _write_run(world, "b", 5)
    assert _rank(world, ["a", "b"]) == _rank(world, ["a", "b"])


def test_incomplete_or_missing_runs_are_excluded_not_ranked(world: dict[str, Any]) -> None:
    _write_run(world, "a", None)
    _write_run(world, "b", 4)
    _write_run(world, "partial", None, keep=10)  # an OOM-interrupted run
    res = _rank(world, ["a", "b", "partial", "absent"])
    assert {r["config"] for r in res["ranking"]} == {"a", "b"}
    reasons = {e["config"]: e["reason"] for e in res["excluded"]}
    assert reasons["partial"].startswith("incomplete: 30/40") and "absent" in reasons


def test_fewer_than_two_usable_runs_fails_closed(world: dict[str, Any]) -> None:
    _write_run(world, "a", None)
    assert pr.main(_argv(world, ["a", "absent"])) == 2
    res = _rank(world, ["a", "absent"])
    assert res["decision"] == "insufficient_runs" and res["usable"] == ["a"]


def test_doc_without_gold_raises(world: dict[str, Any]) -> None:
    _write_run(world, "a", None)
    _write_run(world, "b", 3)
    with pytest.raises(ValueError, match="no dev gold"):
        pr.rank(world["runs"], "spike40", "abc1234", ["a", "b"], ["dev_9999"], world["gold"])
