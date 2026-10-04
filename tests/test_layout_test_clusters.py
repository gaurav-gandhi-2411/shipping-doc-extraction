"""Layout signature variants, ARI, LOGO threshold calibration and the test-split leak guards."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from shipdoc import layout_sig as ls
from shipdoc.ocr import OcrItem, PageOcr


def _page(*cells: tuple[str, float, float]) -> PageOcr:
    """Items given as (text, x, y) top-left corners on a 1000 x 2000 page."""
    items = [
        OcrItem(t, [[x, y], [x + 100, y], [x + 100, y + 20], [x, y + 20]], 0.99)
        for t, x, y in cells
    ]
    return PageOcr("p", 1000, 2000, "paddleocr", "line", 0, items=items)


def test_signatures_are_deterministic_and_variants_have_expected_shape() -> None:
    p = _page(("Invoice No", 50, 100), ("Date: 2026", 450, 100), ("Total", 800, 1500))
    for name, fn in ls.SIGNATURES.items():
        assert np.array_equal(fn(p), fn(p)), name
    assert ls.sig_base(p).shape == (60,)
    assert ls.sig_coarse(p).shape == (60,)
    assert ls.sig_geom(p).shape == (66,)
    assert np.array_equal(ls.sig_geom(p)[:60], ls.sig_base(p))


def test_coarse_snaps_positions_to_a_tenth_and_leaves_presence() -> None:
    s = ls.sig_coarse(_page(("Invoice", 53, 107)))
    assert s[0] == 1.0
    assert abs(s[1] * 10 - round(s[1] * 10)) < 1e-9


def test_geometry_features_in_unit_range_and_empty_page_is_zero() -> None:
    g = ls.geometry_features(_page(("a", 10, 10), ("b", 800, 1900)))
    assert g.shape == (6,) and np.all(g >= 0) and np.all(g <= 1)
    empty = PageOcr("p", 10, 10, "paddleocr", "line", 0)
    assert not ls.geometry_features(empty).any()


def test_invoice_keyword_rule() -> None:
    assert ls.invoice_keyword_present(_page(("Commercial Invoice", 10, 10)))
    assert not ls.invoice_keyword_present(_page(("Air Waybill", 10, 10), ("Consignee", 10, 50)))


def test_ari_and_purity_on_synthetic_clusters() -> None:
    labels = ["a", "a", "b", "b", "c", "c"]
    assert ls.ari(labels, [5, 5, 7, 7, 9, 9]) == pytest.approx(1.0)  # relabelling-invariant
    assert ls.purity(labels, [0, 0, 0, 0, 1, 1]) == pytest.approx(4 / 6)
    assert ls.ari(labels, [0, 1, 0, 1, 0, 1]) < 0.1
    lo, hi = ls.bootstrap_ari_ci(labels * 10, [5, 5, 7, 7, 9, 9] * 10, n_boot=50)
    assert lo <= 1.0 and hi >= lo


def test_wilson_ci_brackets_the_proportion() -> None:
    lo, hi = ls.wilson_ci(50, 100)
    assert lo < 0.5 < hi and 0.39 < lo < 0.41
    assert ls.wilson_ci(0, 0) == (0.0, 1.0)
    assert ls.wilson_ci(0, 10)[0] == 0.0


def _blobs(groups: int, per: int, seed: int = 0) -> tuple[list[str], list[str], np.ndarray]:
    """`groups` well-separated Gaussian blobs in 4-D, `per` docs each; train-style doc ids."""
    rng = np.random.default_rng(seed)
    ids, grp, rows = [], [], []
    for g in range(groups):
        centre = np.zeros(4)
        centre[g % 4] = 10.0 * (1 + g // 4)
        for _ in range(per):
            ids.append(f"train_{len(ids):04d}")
            grp.append(f"g{g}")
            rows.append(centre + rng.normal(0, 0.1, 4))
    return ids, grp, np.vstack(rows)


def test_logo_threshold_calibration_on_synthetic_groups() -> None:
    ids, grp, x = _blobs(groups=8, per=20)
    seen = ls.seen_distances(ids, x, k=8)
    logo = ls.logo_distances(ids, grp, x, k=7)
    cal = ls.calibrate_threshold(seen, logo, 95.0)
    assert cal.threshold == pytest.approx(np.percentile(seen, 95))
    assert cal.fpr <= 0.06  # by construction about 5% of seen docs sit above the 95th percentile
    assert cal.tpr == 1.0  # held-out blob is >= 10 away from every remaining centre
    assert seen.max() < 1.0 < logo.min()
    assert ls.corrected_share(cal.fpr + 0.5 * (cal.tpr - cal.fpr), cal) == pytest.approx(0.5)
    assert ls.corrected_share(0.3, ls.Calibration(1.0, 95.0, 0.05, 0.05)) is None


def test_fit_rejects_test_split_and_malformed_ids() -> None:
    _, _, x = _blobs(groups=3, per=4)
    ok = [f"dev_{i:04d}" for i in range(len(x))]
    assert ls.fit_reference(ok, x, k=3).k == 3
    with pytest.raises(ValueError, match="test split"):
        ls.fit_reference([*ok[:-1], "test_0007"], x, k=3)
    with pytest.raises(ValueError, match="malformed"):
        ls.fit_reference([*ok[:-1], "Test_0007x"], x, k=3)
    with pytest.raises(ValueError, match="test split"):
        ls.assert_fit_ids(["train_0001", "test_0000"])
    with pytest.raises(ValueError, match="test split"):
        ls.seen_distances([f"test_{i:04d}" for i in range(len(x))], x, k=3)


def test_writer_takes_only_ids_and_numbers(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    out = tmp_path / "a.csv"
    with caplog.at_level(logging.DEBUG):
        ls.write_assignments([ls.AssignmentRow("test_0003", "invoice", 4, 1.25, False)], out)
    assert out.read_text(encoding="utf-8").splitlines() == [
        "doc_id,doc_type,cluster,distance,seen",
        "test_0003,invoice,4,1.250000,0",
    ]
    assert caplog.records == []  # the writer logs nothing
    secret = "ACME CORP INVOICE 12345"
    with pytest.raises(ValueError, match="not a doc id"):
        ls.write_assignments([ls.AssignmentRow(secret, "invoice", 1, 1.0, True)], out)
    with pytest.raises(ValueError, match="doc_type"):
        ls.write_assignments([ls.AssignmentRow("test_0001", secret, 1, 1.0, True)], out)
    with pytest.raises(TypeError):
        ls.write_assignments([ls.AssignmentRow("test_0001", "invoice", 1, 1, True)], out)  # type: ignore[arg-type]
    assert secret not in out.read_text(encoding="utf-8")
