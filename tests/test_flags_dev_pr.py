"""scripts/flags_dev_pr.py: review-flag precision / recall on dev100 from OOF probabilities."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_report as br  # noqa: E402
import flags_dev_pr as fd  # noqa: E402

N_BOOT = 200  # small: the tests check mechanics, not interval widths


def _fields(rows: list[tuple[str, str, bool, float, bool]]) -> fd.OofFields:
    """(doc, type, emitted, p, correct) rows -> ``OofFields``."""
    return fd.OofFields(
        np.array([r[0] for r in rows], dtype=object),
        np.array([r[1] for r in rows], dtype=object),
        np.array([r[2] for r in rows], dtype=bool),
        np.array([r[3] for r in rows], dtype=float),
        np.array([r[4] for r in rows], dtype=bool),
    )


def _ref_tau(p: list[float], ok: list[bool], target: float) -> float | None:
    """Plain-python reference of ``confidence.select_tau`` (lowest tau with precision >= target).

    Written independently (sort + running count over distinct values) so the test does not just
    compare the library with itself.
    """
    best = None
    for tau in sorted(set(p), reverse=True):
        acc = [o for q, o in zip(p, ok, strict=True) if q >= tau]
        if sum(acc) / len(acc) >= target:
            best = tau
    return best


def test_field_type_maps_header_and_row_fields() -> None:
    assert fd.field_type("header", "invoice_number") == "header"
    assert fd.field_type("row", "quantity") == "quantity"


def test_summarize_counts_precision_recall_review_rate_by_hand() -> None:
    # 4 emitted fields: accepted-correct, accepted-wrong, flagged-wrong, flagged-correct
    f = _fields(
        [
            ("d0", "quantity", True, 0.9, True),
            ("d0", "quantity", True, 0.8, False),
            ("d1", "quantity", True, 0.2, False),
            ("d1", "quantity", True, 0.1, True),
            ("d1", "quantity", False, np.nan, False),  # non-emitted: outside the population
        ]
    )
    accept = np.array([True, True, False, False, False])
    s = fd.summarize(accept, f, np.ones(5, dtype=bool), N_BOOT)
    assert (s["emitted"], s["accepted"], s["accepted_correct"]) == (4, 2, 1)
    assert (s["wrong"], s["wrong_sent_to_review"]) == (2, 1)
    assert s["precision_accepted"]["point"] == pytest.approx(0.5)
    assert s["error_recall"]["point"] == pytest.approx(0.5)  # 1 of the 2 wrong fields flagged
    assert s["review_rate"]["point"] == pytest.approx(0.5)
    assert s["accepted_share"]["point"] == pytest.approx(0.5)
    for m in fd.METRICS:
        assert s[m]["lo"] <= s[m]["point"] <= s[m]["hi"]


def test_summarize_with_no_accepted_field_has_no_precision() -> None:
    f = _fields([("d0", "quantity", True, 0.1, False), ("d1", "quantity", True, 0.2, True)])
    s = fd.summarize(np.zeros(2, dtype=bool), f, np.ones(2, dtype=bool), N_BOOT)
    assert s["accepted"] == 0 and np.isnan(s["precision_accepted"]["point"])
    assert s["error_recall"]["point"] == 1.0 and s["review_rate"]["point"] == 1.0


def test_summarize_of_an_empty_selection_is_all_none() -> None:
    f = _fields([("d0", "quantity", True, 0.9, True)])
    s = fd.summarize(np.ones(1, dtype=bool), f, np.zeros(1, dtype=bool), N_BOOT)
    assert s["emitted"] == 0 and all(s[m] is None for m in fd.METRICS)


def _three_fold_table() -> tuple[fd.OofFields, np.ndarray]:
    """Two folds with an easy quantity signal, one fold that mislabels its high-p fields."""
    rng = np.random.default_rng(42)
    rows: list[tuple[str, str, bool, float, bool]] = []
    fold: list[int] = []
    for k in range(3):
        for i in range(60):
            p = float(rng.uniform(0.5, 1.0))
            ok = bool(rng.uniform() < p) if k < 2 else p < 0.9  # fold 2: high p is mostly wrong
            rows.append((f"x{k}_{i}", "quantity", True, p, ok))
            fold.append(k)
    return _fields(rows), np.array(fold)


def test_nested_tau_of_a_fold_ignores_that_folds_own_labels() -> None:
    f, fold = _three_fold_table()
    dec = fd.type_decisions(f, fold, 0.98, None)["quantity"]
    for k in range(3):
        other = fold != k
        tau = _ref_tau(f.p[other].tolist(), f.correct[other].tolist(), 0.98)
        mine = fold == k
        expected = np.zeros(mine.sum(), dtype=bool) if tau is None else f.p[mine] >= tau
        assert (dec[mine] == expected).all(), k


def test_frozen_tau_none_accepts_nothing_and_a_tau_thresholds_p() -> None:
    f, fold = _three_fold_table()
    none = fd.type_decisions(f, fold, 0.98, {"quantity": None})["quantity"]
    assert not none.any()
    some = fd.type_decisions(f, fold, 0.98, {"quantity": 0.9})["quantity"]
    assert (some == (f.p >= 0.9)).all()


def test_evaluate_scores_only_the_dev_docs_for_the_dev_schemes() -> None:
    f, fold = _three_fold_table()
    dev = {f"x0_{i}" for i in range(10)}
    res = fd.evaluate(f, fold, dev, {"quantity": 0.9}, N_BOOT)
    assert res["dev100_nested"]["quantity"]["emitted"] == 10
    assert res["dev100_frozen"]["quantity"]["emitted"] == 10
    assert res["all500_nested"]["quantity"]["emitted"] == 180
    # a type with no rows is an empty summary, not a crash
    assert res["dev100_nested"]["header"]["emitted"] == 0
    assert res["dev100_nested"][fd.ALL]["emitted"] == 10


def test_over_null_cells_counts_only_wrong_non_emitted_cells() -> None:
    f = _fields(
        [
            ("dev_1", "purchase_order", False, np.nan, False),  # over-null on dev
            ("train_1", "purchase_order", False, np.nan, False),  # over-null, not dev
            ("dev_1", "purchase_order", False, np.nan, True),  # gold null too: correct
            ("dev_1", "purchase_order", True, 0.5, False),  # emitted: not counted here
        ]
    )
    out = fd.over_null_cells(f, {"dev_1"})
    assert out["purchase_order"] == {"dev": 1, "all": 2}


def test_doc_level_counts_accepted_documents() -> None:
    ids = [f"d{i}" for i in range(6)]
    p = np.array([0.99, 0.5, 0.97, 0.4, 0.995, 0.3])
    y = np.array([True, False, True, False, False, False])
    fold = np.array([0, 0, 1, 1, 2, 2])
    out = fd.doc_level(ids, p, y, fold, {"d0", "d1", "d4"}, 0.98, 0.98)
    assert out["dev_docs"] == 3 and out["dev_fully_correct"] == 1
    assert out["frozen_dev"] == {"accepted": 2, "accepted_correct": 1}  # d0 and d4 reach 0.98
    assert out["frozen_all"] == {"accepted": 2, "accepted_correct": 1}
    none = fd.doc_level(ids, p, y, fold, {"d0"}, None, 0.98)
    assert none["frozen_dev"]["accepted"] == 0 and none["frozen_tau_is_none"] is True


def test_crosscheck_reads_calibrate_v2_csv(tmp_path: Path) -> None:
    f, fold = _three_fold_table()
    res = fd.evaluate(f, fold, {f"x0_{i}" for i in range(10)}, {"quantity": 0.9}, N_BOOT)
    pops = res["dev100_nested"]
    head = [
        "population", "target", "slice", "scheme", *[
            c + s for c in ("precision_accepted", "accepted_share", "review_rate", "error_recall")
            for s in ("", "_lo", "_hi")
        ],
    ]  # fmt: skip
    p = tmp_path / "tau.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(head)
        s = pops["quantity"]
        w.writerow(
            ["quantity", "0.98", "dev", "nested"]
            + [
                float("nan") if s[m] is None else s[m][k]
                for m in ("precision_accepted", "accepted_share", "review_rate", "error_recall")
                for k in ("point", "lo", "hi")
            ]
        )
    out = fd.crosscheck_calibrate_v2(res, p)
    # NaN pairs (a metric with an empty denominator) are skipped, so 9 to 12 values are compared
    assert 9 <= out["n_compared"] <= 12 and out["max_abs_diff"] == pytest.approx(0.0, abs=1e-12)


def test_render_is_aggregates_only_and_labels_the_primary_scheme() -> None:
    f, fold = _three_fold_table()
    schemes = fd.evaluate(f, fold, {f"x0_{i}" for i in range(10)}, {"quantity": 0.9}, N_BOOT)
    doc = fd.doc_level(["a", "b"], np.array([0.5, 0.6]), np.array([True, False]), np.array([0, 1]),
                       {"a"}, None)  # fmt: skip
    res = {
        "oof_fields_sha256": "0" * 64,
        "oof_fields_sha_matches": True,
        "calibrator_sha256": "1" * 64,
        "run_config_hash": "h",
        "frozen_tau": {"quantity": None},
        "schemes": schemes,
        "over_null": {t: {"dev": 0, "all": 0} for t in fd.TYPES},
        "doc_level": doc,
    }
    md = fd.render(res, "cmd", "abc")
    assert "dev100, cross-fitted" in md and "PRIMARY" in md and "SECONDARY" in md
    assert br.find_doc_ids(md) == []
