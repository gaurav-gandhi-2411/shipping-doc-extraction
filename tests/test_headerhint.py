"""Continuation-page header hint (src/shipdoc/headerhint.py) and its opt-in spike hook.

CPU only, no model, no network. The default-OFF guarantee is pinned by hard-coded hashes that were
computed on the commit BEFORE the hook existed (6c2fa30): the production prompt v2, its hash and the
config hash of every shipped spike config must not move.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pytest
import yaml
from _synth import make_corpus

from shipdoc import headerhint as hh
from shipdoc import prompts, spike
from shipdoc.extract import MockBackend
from shipdoc.ocr import OcrItem, PageOcr

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
HINT_CFG = CONFIGS / "spike_qwen35_4b_img_only_keyed_hdrhint.yaml"
SCORER = ROOT / "assignment" / "score.py"
needs_scorer = pytest.mark.skipif(not SCORER.is_file(), reason="assignment/score.py absent")

# --------------------------------------------------------------------------- default unchanged

# Computed with `uv run python` on HEAD 6c2fa30 BEFORE any edit of this task.
PRE_EDIT_PROMPT_HASH_JSON = "cabc7bd9116664db00f6cd8e8b184970e0a186f82062ee55835275883061c64c"
PRE_EDIT_PROMPT_HASH_COMPACT = "34930d57729dba13c6a417af6fbaa2b36e1dddf93b635d9d5f80c577b2079353"
PRE_EDIT_BUILD_PROMPT = {
    (0, 1): "7015214b9ec74fd37ceb80299216b9bed07dd6fb0473869b90867b14cdf05ac1",
    (0, 3): "ece9003ee8555e795a4e2eb07442d6152352987060f466ea1b17c109a1162cd5",
    (1, 3): "6b8c0087e4bf21fddf5b3bcb26f4b389159023024f4fd688680cf607d5600c6d",
    (2, 3): "fd67c2a37a098091c54cda442e9020196d0f72494dcb8160dbbcdfb03596e524",
}
PRE_EDIT_CONFIG_HASHES = {
    "spike_nuextract3_img_ocr": "8725840ceb469d37",
    "spike_nuextract3_img_ocr_compact": "0b52c031fda9c408",
    "spike_nuextract3_img_only": "d41264b49ba19ed2",
    "spike_nuextract3_img_only_compact": "d58c64b30e6f36d6",
    "spike_qwen35_4b_img_ocr": "63fe14ab5db210bc",
    "spike_qwen35_4b_img_ocr_compact": "6413a767fd92599d",
    "spike_qwen35_4b_img_only": "01d87878679ca0fc",  # the production config
    "spike_qwen35_4b_img_only_compact": "e6548a7fff7abc2a",
    "spike_qwen3vl_4b_img_only": "53e4d06a5e1556c9",
    "spike_qwen3vl_4b_img_only_compact": "32d414caa95d49bc",
    "spike_qwen3vl_8b_img_ocr": "68411e2965d2b031",
    "spike_qwen3vl_8b_img_ocr_compact": "b1cb8297d1573ccb",
    "spike_qwen3vl_8b_img_only": "d214faec6c80e9c2",
    "spike_qwen3vl_8b_img_only_compact": "c3223862d175ad00",
}


def test_default_prompt_and_hashes_are_unchanged() -> None:
    assert prompts.PROMPT_VERSION == "v2"
    assert prompts.prompt_hash("json") == PRE_EDIT_PROMPT_HASH_JSON
    assert prompts.prompt_hash("compact") == PRE_EDIT_PROMPT_HASH_COMPACT
    for (page, n), digest in PRE_EDIT_BUILD_PROMPT.items():
        assert hashlib.sha256(prompts.build_prompt(page, n).encode()).hexdigest() == digest


@pytest.mark.parametrize(("name", "expected"), sorted(PRE_EDIT_CONFIG_HASHES.items()))
def test_every_existing_config_hash_is_unchanged(name: str, expected: str) -> None:
    assert spike.load_config(CONFIGS / f"{name}.yaml").config_hash == expected


def test_hdrhint_config_is_the_production_config_plus_the_flag() -> None:
    prod = yaml.safe_load((CONFIGS / "spike_qwen35_4b_img_only.yaml").read_text("utf-8"))
    hint = yaml.safe_load(HINT_CFG.read_text("utf-8"))
    assert {k for k in hint if hint[k] != prod.get(k)} == {"name", "header_hint"}
    assert hint["header_hint"] is True and "header_hint" not in prod
    assert hint["arm"] == "img_only" and hint["output_format"] == "json"
    cfg = spike.load_config(HINT_CFG)
    assert cfg.config_hash != PRE_EDIT_CONFIG_HASHES["spike_qwen35_4b_img_only"]
    assert cfg.name.endswith("_hdrhint")


# --------------------------------------------------------------------------- synthetic OCR


def cell(text: str, x: float, y: float, h: float = 20.0) -> OcrItem:
    """One OCR line item; box width = 10 px per character."""
    x1, y1 = x + 10 * len(text), y + h
    return OcrItem(text, [[x, y], [x1, y], [x1, y1], [x, y1]], 0.99)


def page_of(*items: OcrItem, w: int = 1000, h: int = 1400) -> PageOcr:
    return PageOcr("p", w, h, "paddleocr", "line", 0, list(items))


HEADER_ROW = [("Item", 50), ("Part No.", 130), ("Description", 300), ("Qty", 560), ("Amount", 700)]


def header_row(y: float = 300, order: tuple[int, ...] = (0, 1, 2, 3, 4)) -> list[OcrItem]:
    return [cell(HEADER_ROW[i][0], HEADER_ROW[i][1], y) for i in order]


def data_rows(y0: float = 340) -> list[OcrItem]:
    rows = []
    for k in range(2):
        y = y0 + 40 * k
        rows += [
            cell(t, x, y)
            for t, x in (("1", 50), ("AB-1", 130), ("Widget red", 300), ("5", 560), ("12.50", 700))
        ]
    return rows


def page1() -> PageOcr:
    top = [cell("Invoice No 123", 50, 100), cell("ACME Ltd", 50, 140)]
    return page_of(*top, *header_row(), *data_rows(), cell("Total 25.00", 700, 460))


# --------------------------------------------------------------------------- prompt assembly


def test_hint_text_keeps_the_column_order_whatever_the_ocr_item_order() -> None:
    shuffled = page_of(*header_row(order=(3, 0, 4, 2, 1)), *data_rows())
    h = hh.page1_header_hint(shuffled)
    assert h.applied and h.reason == "ok" and h.n_cells == 5 and h.joined_lines == 1
    assert h.text == "Item Part No. Description Qty Amount"
    assert (
        hh.hinted_prompt("P", h)
        == "P\nColumn headers from page 1: Item Part No. Description Qty Amount"
    )
    assert h.text == hh.page1_header_hint(page1()).text  # same with the rows above and below


def test_no_table_header_line_means_no_hint_and_the_prompt_is_untouched() -> None:
    no_header = page_of(
        cell("Invoice No 123", 50, 100), cell("ACME", 50, 140), cell("hello", 50, 800)
    )
    h = hh.page1_header_hint(no_header)
    assert (h.applied, h.reason, h.text) == (False, "no_table_header", "")
    prompt = prompts.build_prompt(1, 3)
    assert hh.hinted_prompt(prompt, h) is prompt
    banner_only = page_of(cell("continued", 50, 20), *data_rows(100))
    assert hh.page1_header_hint(banner_only).reason == "no_table_header"


def test_absent_ocr_is_the_recorded_fallback() -> None:
    h = hh.page1_header_hint(None)
    assert (h.applied, h.reason) == (False, "no_ocr")


def test_page_one_and_single_page_documents_get_no_hint() -> None:
    p1 = page1()
    assert hh.hint_for_page(0, 3, p1).reason == "page_1"
    assert hh.hint_for_page(0, 1, p1).reason == "single_page"
    assert not hh.hint_for_page(0, 3, p1).applied and not hh.hint_for_page(0, 1, p1).applied


def test_three_page_document_hints_pages_two_and_three_with_the_same_text() -> None:
    p1 = page1()
    hints = [hh.hint_for_page(i, 3, p1) for i in range(3)]
    assert [h.applied for h in hints] == [False, True, True]
    assert hints[1].text == hints[2].text == "Item Part No. Description Qty Amount"


def test_header_spanning_two_ocr_lines_is_joined_column_by_column() -> None:
    # row A (y=300) names two families ("Item", "Qty"); row B (y=322) holds the second words
    row_a = [cell("Item", 50, 300), cell("Part", 130, 300), cell("Qty", 560, 300)]
    row_b = [cell("No.", 135, 322), cell("Shipped", 540, 322)]
    # "Shipped" starts left of "Qty" and overlaps it: stacked under it, not a new column
    page = page_of(*row_b, *row_a, *data_rows(360))
    h = hh.page1_header_hint(page)
    assert h.applied and h.joined_lines == 2
    assert h.text == "Item Part No. Qty Shipped"
    assert h.n_cells == 3


def test_a_close_data_row_or_a_far_line_is_not_joined() -> None:
    page = page_of(*header_row(), *data_rows(324))  # rows start right under the header
    h = hh.page1_header_hint(page)
    assert h.joined_lines == 1 and h.text == "Item Part No. Description Qty Amount"
    far = page_of(*header_row(), cell("Notes", 130, 400), *data_rows(500))
    assert hh.page1_header_hint(far).joined_lines == 1


def test_control_characters_are_stripped_and_whitespace_collapsed() -> None:
    assert hh.sanitize("Qty\x00\n\tShipped\x1b[0m  ​X\u0085y") == "Qty Shipped [0m X y"
    row = [cell("Item\x07", 50, 300), cell("Qty\n", 560, 300), cell("\x00", 700, 300)]
    h = hh.page1_header_hint(page_of(*row, *data_rows()))
    assert h.text == "Item Qty" and "\x00" not in hh.hinted_prompt("P", h)


def test_cap_keeps_whole_cells_left_to_right_and_records_truncation() -> None:
    cells = ["alpha", "beta", "gamma", "delta"]
    assert hh.cap_cells(cells, 100) == ("alpha beta gamma delta", False)
    assert hh.cap_cells(cells, 16) == ("alpha beta gamma", True)  # 16 chars exactly
    assert hh.cap_cells(cells, 15) == ("alpha beta", True)  # gamma would make 16
    assert hh.cap_cells(["x" * 50, "y"], 20) == ("x" * 20, True)  # first cell alone too long
    long_row = [cell(f"Col{i:02d}", 20 + 60 * i, 300) for i in range(60)]
    h = hh.page1_header_hint(page_of(cell("Qty", 5, 300), *long_row, cell("Part", 5, 322)))
    assert len(h.text) <= hh.HINT_MAX_CHARS == hh.HINT_MAX_TOKENS * hh.CHARS_PER_TOKEN
    assert h.truncated or len(h.text) <= hh.HINT_MAX_CHARS


def test_hint_for_job_loads_page1_once_and_falls_back_without_a_cache(tmp_path: Path) -> None:
    cache_dir = tmp_path / "paddleocr" / "dev"
    cache_dir.mkdir(parents=True)
    (cache_dir / "dev_0001_p1.json").write_text(json.dumps(page1().to_dict()), encoding="utf-8")
    memo: dict[str, PageOcr | None] = {}
    base = prompts.build_prompt(1, 2)
    out, rec = hh.hinted_prompt_for_job(base, "dev", "dev_0001", 1, 2, tmp_path, memo)
    assert out.endswith("Column headers from page 1: Item Part No. Description Qty Amount")
    assert rec["applied"] is True and rec["reason"] == "ok" and "dev_0001" in memo
    out0, rec0 = hh.hinted_prompt_for_job(base, "dev", "dev_0001", 0, 2, tmp_path, {})
    assert out0 == base and rec0["reason"] == "page_1"
    out2, rec2 = hh.hinted_prompt_for_job(base, "dev", "dev_0009", 1, 2, tmp_path, {})
    assert out2 == base and rec2["reason"] == "no_ocr"  # no cache file: hint absent
    assert hh.hint_for_page(1, 2, None).reason == "no_ocr"


def test_the_hint_needs_the_image_only_arm() -> None:
    hh.require_supported("img_only")
    with pytest.raises(ValueError, match="img_only"):
        hh.require_supported("img_ocr")


# --------------------------------------------------------------------------- spike hook


class Recording(MockBackend):
    """Mock that remembers every prompt it is given."""

    def __init__(self, gold: dict[str, Any]) -> None:
        super().__init__(gold)
        self.prompts: list[str] = []

    def extract_page(self, image: Any, prompt: str, schema: Any, ocr_text: Any) -> Any:
        self.prompts.append(prompt)
        return super().extract_page(image, prompt, schema, ocr_text)


def _world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict[str, Any]]:
    root = tmp_path / "data"
    gold = make_corpus(root, [("dev_0001", 3, "invoice"), ("dev_0002", 1, "invoice")])
    ocr_dir = tmp_path / "ocr" / "paddleocr" / "dev"
    ocr_dir.mkdir(parents=True)
    (ocr_dir / "dev_0001_p1.json").write_text(json.dumps(page1().to_dict()), encoding="utf-8")
    monkeypatch.setenv("SHIPDOC_OCR_CACHE", str(tmp_path / "ocr"))
    return root, gold


@needs_scorer
def test_flag_off_the_prompts_are_the_production_prompts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, gold = _world(tmp_path, monkeypatch)
    cfg = spike.load_config(CONFIGS / "spike_qwen35_4b_img_only.yaml")
    be = Recording(gold)
    spike.run_spike(
        cfg, ["dev_0001", "dev_0002"], "dev", "off", be, runs_root=tmp_path / "r", data_root=root
    )
    assert be.prompts == [prompts.build_prompt(i, 3) for i in range(3)] + [
        prompts.build_prompt(0, 1)
    ]
    trace = [json.loads(ln) for ln in (tmp_path / "r/off/trace.jsonl").read_text().splitlines()]
    assert all("header_hint" not in p for t in trace for p in t["pages"])


@needs_scorer
def test_flag_on_only_continuation_pages_get_the_hint_and_it_is_traced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, gold = _world(tmp_path, monkeypatch)
    cfg = spike.load_config(HINT_CFG)
    be = Recording(gold)
    spike.run_spike(
        cfg, ["dev_0001", "dev_0002"], "dev", "on", be, runs_root=tmp_path / "r", data_root=root
    )
    line = "\nColumn headers from page 1: Item Part No. Description Qty Amount"
    assert be.prompts[0] == prompts.build_prompt(0, 3)
    assert be.prompts[1] == prompts.build_prompt(1, 3) + line
    assert be.prompts[2] == prompts.build_prompt(2, 3) + line
    assert be.prompts[3] == prompts.build_prompt(0, 1)  # one-page doc: nothing
    trace = [json.loads(ln) for ln in (tmp_path / "r/on/trace.jsonl").read_text().splitlines()]
    reasons = [[p["header_hint"]["reason"] for p in t["pages"]] for t in trace]
    assert reasons == [["page_1", "ok", "ok"], ["single_page"]]
    assert trace[0]["pages"][1]["header_hint"]["text"] == "Item Part No. Description Qty Amount"
    assert trace[0]["config"]["hash"] == cfg.config_hash
    assert trace[0]["prompt"]["hash"] == prompts.prompt_hash("json")  # static template unchanged


@needs_scorer
def test_flag_on_without_ocr_cache_runs_with_the_production_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, gold = _world(tmp_path, monkeypatch)
    (tmp_path / "ocr/paddleocr/dev/dev_0001_p1.json").unlink()
    be = Recording(gold)
    spike.run_spike(
        spike.load_config(HINT_CFG),
        ["dev_0001"],
        "dev",
        "noocr",
        be,
        runs_root=tmp_path / "r",
        data_root=root,
    )
    assert be.prompts == [prompts.build_prompt(i, 3) for i in range(3)]
    trace = [json.loads(ln) for ln in (tmp_path / "r/noocr/trace.jsonl").read_text().splitlines()]
    assert [p["header_hint"]["reason"] for p in trace[0]["pages"]] == ["page_1", "no_ocr", "no_ocr"]


def test_flag_on_with_the_ocr_arm_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, gold = _world(tmp_path, monkeypatch)
    raw = yaml.safe_load(HINT_CFG.read_text("utf-8"))
    raw["arm"] = "img_ocr"
    f = tmp_path / "bad.yaml"
    f.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = spike.load_config(f)
    with pytest.raises(ValueError, match="img_only"):
        spike.run_spike(
            cfg, ["dev_0001"], "dev", "x", Recording(gold), runs_root=tmp_path / "r", data_root=root
        )


# --------------------------------------------------------------------------- A/B bookkeeping


def test_select_docs_is_dev100_and_train_multipage_of_the_02_doc_list() -> None:
    meta = [
        {"doc_id": "dev_0000", "multipage": True},
        {"doc_id": "dev_0001", "multipage": False},
        {"doc_id": "dev_0002", "multipage": True},
        {"doc_id": "train_0000", "multipage": True},
        {"doc_id": "train_0001", "multipage": False},
        {"doc_id": "train_0002", "multipage": True},
    ]
    docs = hh.select_docs(meta, ["dev_0000", "dev_0001"], ["train_0000", "train_0001", "dev_0002"])
    assert docs == {
        "dev": ["dev_0000"],
        "train": ["train_0000"],  # train_0002 is not in the 02 doc list; dev_0002 not in dev100
        "pooled": ["dev_0000", "train_0000"],
    }


def _manifest(**kw: Any) -> dict[str, Any]:
    m: dict[str, Any] = {
        "code_sha": "a" * 40, "config": {"hash": "H"}, "model": {"revision": "R"},
        "arm": "img_only", "output_format": "json", "logprobs": True, "seed": 42,
        "shard": "0/1", "batch_size": 8,
    }  # fmt: skip
    m.update(kw)
    return m


def _progress(**kw: Any) -> dict[str, Any]:
    p: dict[str, Any] = {"status": "complete", "done": 5, "total": 5}
    p.update(kw)
    return p


def _trace(doc: str, **kw: Any) -> dict[str, Any]:
    t: dict[str, Any] = {
        "doc_id": doc,
        "config": {"hash": "H"},
        "prompt": {"hash": "PH"},
        "model": {"revision": "R"},
    }
    t.update(kw)
    return t


def _call(
    fn: Any, manifest: Any = None, progress: Any = None, traces: Any = None, **kw: Any
) -> Any:
    args: dict[str, Any] = {
        "needed": ["d1", "d2"], "config_hash": "H", "prompt_hash": "PH", "revision": "R",
        "batch_size": 8, "seed": 42, "output_format": "json",
    }  # fmt: skip
    args.update(kw)
    return fn(
        _manifest() if manifest is None else manifest,
        _progress() if progress is None else progress,
        [_trace("d1"), _trace("d2"), _trace("d3")] if traces is None else traces,
        **args,
    )


def _check(**kw: Any) -> list[str]:
    return _call(hh.control_check, **kw)  # type: ignore[no-any-return]


def test_control_check_accepts_the_matching_02_style_run_and_ignores_the_code_sha() -> None:
    assert _check() == []  # d3 is in the run but not needed: ignored
    assert _check(manifest=_manifest(code_sha="b" * 40)) == []  # the pin legitimately differs


@pytest.mark.parametrize(
    ("kw", "needle"),
    [
        ({"traces": [_trace("d1"), _trace("d2", prompt={"hash": "old"})]}, "prompt hash"),
        ({"prompt_hash": "other"}, "prompt hash"),  # the checked-out code's prompt moved
        ({"manifest": _manifest(config={"hash": "X"})}, "config hash differs"),
        ({"config_hash": "other"}, "config hash differs"),  # the checked-out config moved
        ({"traces": [_trace("d1"), _trace("d2", config={"hash": "X"})]}, "config hash"),
        ({"manifest": _manifest(batch_size=4)}, "batch size 4 != 8"),
        ({"batch_size": 1}, "batch size 8 != 1"),
        ({"traces": [_trace("d1")]}, "1 needed docs not in the run"),
        ({"needed": ["d1", "d2", "d9"]}, "1 needed docs not in the run"),
        ({"progress": _progress(status="running")}, "progress is not complete"),
        ({"progress": _progress(done=4)}, "progress is not complete"),
        ({"progress": {}}, "progress is not complete"),
        ({"manifest": _manifest(model={"revision": "X"})}, "model revision differs"),
        ({"traces": [_trace("d1"), _trace("d2", model={"revision": "X"})]}, "revision"),
        ({"manifest": _manifest(seed=1)}, "seed is not 42"),
        ({"manifest": _manifest(logprobs=False)}, "logprobs differ"),
        ({"manifest": _manifest(output_format="compact")}, "output format differs"),
        ({"manifest": _manifest(arm="img_ocr")}, "img_only"),
        ({"manifest": _manifest(shard="0/2")}, "shard"),
        ({"manifest": _manifest(code_sha="a" * 40 + "+dirty")}, "dirty"),
        ({"hint_flag_off": False}, "header_hint on"),
        ({"manifest": {}}, "no manifest"),
    ],
)
def test_control_check_refuses_every_mismatch(kw: dict[str, Any], needle: str) -> None:
    why = _check(**kw)
    assert why and any(needle in w for w in why), why
    with pytest.raises(hh.ControlRefused, match="REFUSED") as exc:  # the raising entry point
        _call(hh.require_control, **kw)
    assert exc.value.reasons == why


def test_control_check_missing_manifest_field_fails_closed() -> None:
    for field in ("logprobs", "seed", "batch_size", "code_sha", "shard", "arm"):
        m = _manifest()
        del m[field]
        assert _check(manifest=m), field
    assert hh.control_check(None, None, [], needed=[], config_hash="H", prompt_hash="PH",
                            revision="R", batch_size=8, seed=42, output_format="json")  # fmt: skip


def test_hint_config_reasons_requires_production_plus_the_one_flag() -> None:
    prod = {"name": "a", "seed": 42, "arm": "img_only"}
    assert hh.hint_config_reasons(prod, {**prod, "name": "b", "header_hint": True}) == []
    assert hh.hint_config_reasons(prod, {**prod, "name": "b"})  # flag missing
    assert hh.hint_config_reasons(prod, {**prod, "name": "b", "header_hint": True, "seed": 1})


def _page(raw: str, **parsed: Any) -> dict[str, Any]:
    return {"raw_text": raw, "parsed": parsed}


def test_noise_floor_counts_page1_differences_without_values() -> None:
    ctrl = {
        "d1": {"pages": [_page("r1", a=1, rows=[{"q": 1}, {"q": 2}]), _page("x")]},
        "d2": {"pages": [_page("r2", a=2)]},
        "d3": {"pages": [_page("r3", a=3)]},
    }
    hint = {
        "d1": {"pages": [_page("r1", a=1, rows=[{"q": 1}, {"q": 2}]), _page("DIFFERENT")]},
        "d2": {"pages": [_page("r2", a=5)]},  # parsed differs, raw_text same
        "d3": {"pages": [_page("r3x", a=3)]},  # raw_text differs only
    }
    nf = hh.noise_floor(ctrl, hint, ["d1", "d2", "d3"])
    assert nf["docs_page1_raw_text_differs"] == 1 and nf["docs_page1_parsed_differs"] == 1
    assert nf["docs_page1_any_differs"] == 2 and nf["n_docs"] == 3
    assert nf["raw_text_diff_rate"] == pytest.approx(1 / 3)
    assert (nf["n_cells"], nf["n_cells_changed"]) == (5, 1)  # page 2 of d1 is not looked at
    assert nf["cell_change_share"] == pytest.approx(0.2)
    text = hh.format_noise_floor(nf)
    assert "NOISE FLOOR" in text and "1 of 3 (33.33%)" in text and "WARNING" in text
    assert "DIFFERENT" not in text and "r3x" not in text
    clean = hh.noise_floor(ctrl, ctrl, ["d1", "d2"])
    assert clean["docs_page1_any_differs"] == 0 and "WARNING" not in hh.format_noise_floor(clean)


@pytest.mark.parametrize(
    ("lo", "row", "on_hint", "on_ctrl", "decision", "n_failed"),
    [
        (
            0.001,
            0.0,
            5,
            5,
            "ADOPT",
            0,
        ),  # row delta exactly 0 is "not negative"; equal over-nulls ok
        (0.02, 0.01, 3, 5, "ADOPT", 0),
        (0.0, 0.01, 5, 5, "DO NOT ADOPT", 1),  # CI lower bound must be strictly > 0
        (-0.01, 0.02, 5, 5, "DO NOT ADOPT", 1),
        (0.01, -0.0001, 5, 5, "DO NOT ADOPT", 1),
        (0.01, 0.01, 6, 5, "DO NOT ADOPT", 1),
        (-0.01, -0.01, 9, 5, "DO NOT ADOPT", 3),
        (math.nan, 0.0, 5, 5, "DO NOT ADOPT", 1),
        (0.01, math.nan, 5, 5, "DO NOT ADOPT", 1),
        (math.inf, 0.0, 5, 5, "DO NOT ADOPT", 1),
    ],
)
def test_decision_rule(
    lo: float, row: float, on_hint: int, on_ctrl: int, decision: str, n_failed: int
) -> None:
    got, failed = hh.decide(lo, row, on_hint, on_ctrl)
    assert got == decision and len(failed) == n_failed


@needs_scorer
def test_compare_runs_on_two_identical_mock_runs_says_do_not_adopt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, gold = _world(tmp_path, monkeypatch)
    ids = ["dev_0001", "dev_0002"]
    out: dict[str, Any] = {}
    for name, cfg_path in (("ctrl", CONFIGS / "spike_qwen35_4b_img_only.yaml"), ("hint", HINT_CFG)):
        spike.run_spike(
            spike.load_config(cfg_path),
            ids,
            "dev",
            name,
            MockBackend(gold),
            runs_root=tmp_path / "r",
            data_root=root,
        )
        run = tmp_path / "r" / name
        out[name] = (
            json.loads((run / "predictions.json").read_text()),
            {
                t["doc_id"]: t
                for t in map(json.loads, (run / "trace.jsonl").read_text().splitlines())
            },
        )
    res = hh.compare_runs(*out["ctrl"], *out["hint"], gold, {"pooled": ids, "dev": ids})
    p = res["subsets"]["pooled"]
    assert (p["n_docs"], p["n_pages"]) == (2, 4)
    assert p["OVERALL"]["delta"] == 0.0 and p["row_f1"]["delta"] == 0.0
    assert res["decision"] == "DO NOT ADOPT" and res["failed_clauses"]
    assert res["hint_page_reasons"] == {"ok": 2}  # pages 2 and 3 of dev_0001
    text = hh.format_report(res)
    assert "DECISION: DO NOT ADOPT" in text and "CAVEAT: Train docs are zero-shot" in text
    assert "dev_0001" not in text  # aggregates only
    with pytest.raises(ValueError, match="lacks"):
        hh.compare_runs(out["ctrl"][0], out["ctrl"][1], {}, {}, gold, {"pooled": ids})


needs_data = pytest.mark.skipif(
    not (ROOT / "data" / "dev" / "labels").is_dir()
    or not (ROOT / "data" / "train" / "labels").is_dir(),
    reason="data/ (gitignored) absent",
)


def _synthetic_02_run(run: Path, pooled: list[str], **manifest_kw: Any) -> None:
    """A control folder shaped like the 02 run (hashes of the production config and prompt v2)."""
    cfg = spike.load_config(CONFIGS / "spike_qwen35_4b_img_only.yaml")
    run.mkdir(parents=True)
    ids = [*pooled, "train_9998"]  # the 02 run holds more docs than the pooled ones
    manifest = {
        "code_sha": "42b812b5b09d6e4bff0df12564017f71ffad5fc9",
        "config": {"hash": PRE_EDIT_CONFIG_HASHES["spike_qwen35_4b_img_only"]},
        "model": {"revision": cfg.backend.revision},
        "arm": "img_only", "output_format": "json", "logprobs": True, "seed": 42,
        "shard": "0/1", "batch_size": 8,
    }  # fmt: skip
    manifest.update(manifest_kw)
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run / "progress.json").write_text(
        json.dumps({"status": "complete", "done": len(ids), "total": len(ids)}), encoding="utf-8"
    )
    lines = [
        {
            "doc_id": d,
            "config": {"hash": PRE_EDIT_CONFIG_HASHES["spike_qwen35_4b_img_only"]},
            "prompt": {"hash": PRE_EDIT_PROMPT_HASH_JSON},
            "model": {"revision": cfg.backend.revision},
        }
        for d in ids
    ]
    (run / "trace.jsonl").write_text("\n".join(map(json.dumps, lines)) + "\n", encoding="utf-8")


def _pooled_ids() -> list[str]:
    meta = [
        m for s in ("train", "dev") for m in json.loads((ROOT / "meta" / f"{s}.json").read_text())
    ]
    docs = hh.select_docs(
        meta,
        json.loads((ROOT / "splits" / "dev100.json").read_text()),
        json.loads((ROOT / "splits" / "zeroshot500.json").read_text()),
    )
    return docs["pooled"]


@needs_data
def test_plan_cli_accepts_the_02_style_control_and_counts_the_multipage_docs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run, out = tmp_path / "02run", tmp_path / "plan"
    _synthetic_02_run(run, _pooled_ids())
    rc = hh.main(["plan", "--repo", str(ROOT), "--out-dir", str(out), "--batch-size", "8",
                  "--control-run", str(run)])  # fmt: skip
    assert rc == 0
    plan = json.loads((out / "plan.json").read_text())
    c = plan["counts"]
    assert (c["dev"]["docs"], c["train"]["docs"], c["pooled"]["docs"]) == (28, 124, 152)
    assert (c["dev"]["pages"], c["train"]["pages"], c["pooled"]["pages"]) == (63, 260, 323)
    assert plan["control_config_hash"] == PRE_EDIT_CONFIG_HASHES["spike_qwen35_4b_img_only"]
    assert plan["control_prompt_hash"] == PRE_EDIT_PROMPT_HASH_JSON  # header_hint off: prompt v2
    assert set(plan["control_sources"]) == set(plan["docs"]["pooled"])  # restricted to the pooled
    assert json.loads((out / "docs_pooled.json").read_text()) == plan["docs"]["pooled"]
    printed = capsys.readouterr().out
    assert "152 multipage docs" in printed and "323 pages" in printed
    assert "CONTROL ACCEPTED" in printed and "42b812b" in printed and "no rerun" in printed


@needs_data
@pytest.mark.parametrize(
    ("kw", "needle"),
    [
        ({"batch_size": 4}, "batch size 4 != 8"),
        ({"code_sha": "x+dirty"}, "dirty"),
        ({"config": {"hash": "0" * 16}}, "config hash differs"),
        ({"model": {"revision": "0" * 40}}, "model revision differs"),
    ],
)
def test_plan_cli_refuses_a_bad_control_and_writes_no_plan(
    tmp_path: Path, kw: dict[str, Any], needle: str
) -> None:
    run, out = tmp_path / "02run", tmp_path / "plan"
    _synthetic_02_run(run, _pooled_ids(), **kw)
    with pytest.raises(hh.ControlRefused, match=needle):
        hh.main(["plan", "--repo", str(ROOT), "--out-dir", str(out), "--batch-size", "8",
                 "--control-run", str(run)])  # fmt: skip
    assert not (out / "plan.json").exists()  # nothing downstream can read a plan


@needs_data
def test_plan_cli_refuses_missing_docs_batch_mismatch_and_an_unreadable_control(
    tmp_path: Path,
) -> None:
    pooled = _pooled_ids()
    out = tmp_path / "plan"
    short = tmp_path / "short"
    _synthetic_02_run(short, pooled[:-3])  # three pooled docs absent from the run
    with pytest.raises(hh.ControlRefused, match="3 needed docs not in the run"):
        hh.main(["plan", "--repo", str(ROOT), "--out-dir", str(out), "--batch-size", "8",
                 "--control-run", str(short)])  # fmt: skip
    full = tmp_path / "full"
    _synthetic_02_run(full, pooled)
    with pytest.raises(hh.ControlRefused, match="batch size 8 != 1"):  # hint arm at another batch
        hh.main(["plan", "--repo", str(ROOT), "--out-dir", str(out), "--batch-size", "1",
                 "--control-run", str(full)])  # fmt: skip
    with pytest.raises(hh.ControlRefused, match="cannot read"):
        hh.main(["plan", "--repo", str(ROOT), "--out-dir", str(out), "--batch-size", "8",
                 "--control-run", str(tmp_path / "no_such_run")])  # fmt: skip
    assert not (out / "plan.json").exists()


@needs_scorer
def test_compare_cli_end_to_end_on_mock_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root, gold = _world(tmp_path, monkeypatch)
    monkeypatch.setenv("SHIPDOC_DATA_DIR", str(root))
    ids = ["dev_0001", "dev_0002"]
    for name, cfg_path in (("ctrl", CONFIGS / "spike_qwen35_4b_img_only.yaml"), ("hint", HINT_CFG)):
        spike.run_spike(spike.load_config(cfg_path), ids, "dev", name, MockBackend(gold),
                        runs_root=tmp_path / "r", data_root=root)  # fmt: skip
    out = tmp_path / "ab"
    out.mkdir()
    plan = {
        "docs": {"pooled": ids, "dev": ids, "train": []},
        "control_sources": {d: str(tmp_path / "r" / "ctrl") for d in ids},
    }
    (out / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    rc = hh.main(["compare", "--out-dir", str(out), "--hint-run-dir", str(tmp_path / "r" / "hint")])
    printed = capsys.readouterr().out
    assert rc == 0 and "DECISION: DO NOT ADOPT" in printed
    assert printed.index("NOISE FLOOR") < printed.index("OVERALL")  # noise floor comes first
    res = json.loads((out / "hdrhint_compare.json").read_text())
    assert res["noise_floor"]["docs_page1_any_differs"] == 0  # same mock, page 1 identical
    assert res["noise_floor"]["raw_text_diff_rate"] == 0.0 and "02 run" in res["control_note"]
    assert set(res["subsets"]) == {"pooled", "dev"}  # the empty train subset is skipped
    assert (out / "hdrhint_compare.md").read_text().startswith("```")
    plan["docs"]["pooled"] = [*ids, "dev_9999"]
    (out / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="no gold"):
        hh.main(["compare", "--out-dir", str(out), "--hint-run-dir", str(tmp_path / "r" / "hint")])
