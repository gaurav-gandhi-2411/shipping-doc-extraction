"""shipdoc.ocr_stage: cache validity, device decision, timing, the seeded re-OCR check.

CPU only, no Paddle: the engine of the re-OCR check is a fake. Texts are synthetic.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from shipdoc import ocr, ocr_stage

STEMS = [
    f"test_{d:04d}_p{p}"
    for d, pages in enumerate([1, 2, 1, 1, 3, 1, 1, 2])
    for p in range(1, pages + 1)
]


def page(stem: str, text: str = "hello", seconds: float = 1.0) -> ocr.PageOcr:
    items = [ocr.OcrItem(text, [[0, 0], [10, 0], [10, 5], [0, 5]], 0.9)]
    return ocr.PageOcr(f"{stem}.png", 10, 5, "paddleocr", "line", 0, items, seconds=seconds)


def write_page(root: Path, stem: str, **kw: Any) -> Path:
    p = ocr.cache_path(root, "paddleocr", "test", stem)
    ocr.write_json_atomic(p, page(stem, **kw).to_dict())
    return p


@pytest.fixture()
def images(tmp_path: Path) -> Path:
    d = tmp_path / "data" / "test" / "images"
    d.mkdir(parents=True)
    for s in STEMS:
        (d / f"{s}.png").write_bytes(b"x")
    return d


def test_expected_stems_from_file_names_only(images: Path) -> None:
    assert ocr_stage.expected_stems(images) == sorted(STEMS)
    with pytest.raises(ocr_stage.OcrStageError, match="no image folder"):
        ocr_stage.expected_stems(images / "nope")


def test_cache_status_counts_valid_missing_and_invalid(tmp_path: Path) -> None:
    root = tmp_path / "c"
    for s in STEMS[:-2]:
        write_page(root, s)
    bad = ocr.cache_path(root, "paddleocr", "test", STEMS[0])
    bad.write_text("{not json", encoding="utf-8")
    wrong = ocr.cache_path(root, "paddleocr", "test", STEMS[1])
    wrong.write_text(json.dumps({**page("other").to_dict()}), encoding="utf-8")  # wrong page name
    st = ocr_stage.cache_status(root, STEMS)
    assert st["expected"] == len(STEMS) and st["missing"] == 2 and st["invalid"] == 2
    assert st["valid"] == len(STEMS) - 4 and not st["complete"]
    assert set(st["invalid_all"]) == {STEMS[0], STEMS[1]}
    with pytest.raises(ocr_stage.OcrStageError, match="incomplete"):
        ocr_stage.require_complete(root, STEMS, len(STEMS))


def test_isolate_invalid_renames_and_never_deletes(tmp_path: Path) -> None:
    root = tmp_path / "c"
    for s in STEMS:
        write_page(root, s)
    bad = ocr.cache_path(root, "paddleocr", "test", STEMS[3])
    bad.write_text("{", encoding="utf-8")
    assert ocr_stage.isolate_invalid(root, STEMS) == [STEMS[3]]
    assert not bad.exists() and bad.with_name(bad.name + ".invalid").read_text() == "{"
    write_page(root, STEMS[3])  # redone by a rerun
    assert ocr_stage.cache_status(root, STEMS)["complete"]
    bad.write_text("{", encoding="utf-8")  # a second corruption keeps the first quarantined copy
    ocr_stage.isolate_invalid(root, STEMS)
    assert len(list(bad.parent.glob(bad.name + ".invalid*"))) == 2


@pytest.mark.parametrize(
    ("has_gpu", "flavor", "device", "ok", "fallback", "warns"),
    [
        (True, "gpu-cu126", "gpu", True, False, 0),
        (True, "cpu", "cpu", True, True, 1),  # CPU fallback on a GPU runtime: loud
        (True, None, "cpu", True, True, 1),
        (False, "cpu", "cpu", True, True, 1),
        (False, "gpu-cu118", "gpu", False, False, 1),  # a GPU wheel without a GPU cannot run
    ],
)
def test_decide_device(
    has_gpu: bool, flavor: str | None, device: str, ok: bool, fallback: bool, warns: int
) -> None:
    d = ocr_stage.decide_device(has_gpu, flavor)
    assert (d["device"], d["ok"], d["fallback"], len(d["warnings"])) == (
        device,
        ok,
        fallback,
        warns,
    )
    if d["fallback"] and has_gpu:
        assert "FALLBACK" in d["warnings"][0]


def test_timing_summary_mean_p95_total_and_sessions(tmp_path: Path) -> None:
    root = tmp_path / "c"
    secs = [float(i + 1) for i in range(len(STEMS))]
    for s, v in zip(STEMS, secs, strict=True):
        write_page(root, s, seconds=v)
    t = ocr_stage.timing_summary(root, STEMS, "gpu", 90.0, 5)
    assert t["pages"] == len(STEMS) and t["seconds_total"] == round(sum(secs), 1)
    assert t["seconds_per_page_mean"] == round(sum(secs) / len(secs), 3)
    assert t["seconds_per_page_p95"] == ocr.percentile(secs, 95)
    assert t["device"] == "gpu" and t["wall_clock_s_all_sessions"] == 90.0
    t2 = ocr_stage.timing_summary(root, STEMS, "gpu", 30.0, 0, prior=t)
    assert len(t2["sessions"]) == 2 and t2["wall_clock_s_all_sessions"] == 120.0
    assert "hello" not in json.dumps(t2)  # no OCR text in the timing file
    (ocr.cache_path(root, "paddleocr", "test", STEMS[0])).unlink()
    with pytest.raises(ocr_stage.OcrStageError):
        ocr_stage.timing_summary(root, STEMS, "gpu")


def test_recheck_pages_are_seeded_sorted_and_stable() -> None:
    a = ocr_stage.select_recheck_pages(STEMS)
    assert a == ocr_stage.select_recheck_pages(list(reversed(STEMS)))
    assert len(a) == 5 and a == sorted(a) and set(a) <= set(STEMS)
    assert ocr_stage.select_recheck_pages(STEMS[:3]) == sorted(STEMS[:3])


class FakeEngine:
    """Returns the same text as the cache, except for the pages in `change`."""

    name = "paddleocr"

    def __init__(self, change: set[str] | None = None, jitter: bool = False) -> None:
        self.change, self.jitter = change or set(), jitter

    def run(self, image_path: Path) -> ocr.PageOcr:
        stem = image_path.stem
        text = "changed" if stem in self.change else "hello"
        p = page(stem, text, seconds=2.0)
        if self.jitter:  # a box moved by a pixel: same text, different content
            p.items[0].box[0][0] = 1.0
        return p

    def describe(self) -> dict[str, Any]:
        return {}


def test_recheck_text_identical_passes_and_reports_content_identity(
    images: Path, tmp_path: Path
) -> None:
    root = tmp_path / "c"
    for s in STEMS:
        write_page(root, s)
    rep = ocr_stage.recheck(FakeEngine(), images, root, tmp_path / "scratch")
    assert rep["ok"] and rep["n_pages"] == 5 and rep["n_text_identical"] == 5
    assert rep["n_content_identical"] == 5  # everything but `seconds` matches
    jit = ocr_stage.recheck(FakeEngine(jitter=True), images, root, tmp_path / "scratch2")
    assert jit["ok"] and jit["n_content_identical"] == 0  # text is the blocking criterion
    chosen = rep["pages"]
    bad = ocr_stage.recheck(FakeEngine({chosen[0]}), images, root, tmp_path / "scratch3")
    assert not bad["ok"] and bad["n_text_identical"] == 4
    assert "hello" not in json.dumps(bad) and "changed" not in json.dumps(bad)


def test_importing_the_stage_never_imports_paddle() -> None:
    code = "import sys, shipdoc.ocr_stage; sys.exit(int('paddle' in sys.modules))"
    res = subprocess.run([sys.executable, "-c", code], check=False)
    assert res.returncode == 0


def test_cli_check_and_finalize(images: Path, tmp_path: Path) -> None:
    root = tmp_path / "c"
    for s in STEMS[:-1]:
        write_page(root, s)
    args = [
        "--images-dir",
        str(images),
        "--cache-root",
        str(root),
        "--expect-pages",
        str(len(STEMS)),
    ]
    assert ocr_stage.main(["check", *args]) == 1  # one page missing
    assert (
        ocr_stage.main(["finalize", *args, "--device", "cpu", "--out", str(tmp_path / "t.json")])
        == 1
    )
    assert not (tmp_path / "t.json").exists()  # nothing written for an incomplete cache
    write_page(root, STEMS[-1])
    assert ocr_stage.main(["check", *args, "--out", str(tmp_path / "chk.json")]) == 0
    out = tmp_path / "t.json"
    assert (
        ocr_stage.main(["finalize", *args, "--device", "cpu", "--wall-s", "5", "--out", str(out)])
        == 0
    )
    assert json.loads(out.read_text())["pages"] == len(STEMS)
    assert ocr_stage.main(["device", "--has-gpu", "1", "--flavor", "cpu"]) == 0
    assert ocr_stage.main(["device", "--has-gpu", "0", "--flavor", "gpu-cu126"]) == 1
