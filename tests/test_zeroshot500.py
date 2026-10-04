"""CPU tests for the zero-shot-500 run path: train+dev in one run, logprobs in the trace, resume.

Everything runs on the mock backend (no GPU). Tests that replay real gold need the gitignored
data/ and assignment/ folders and skip without them; the rest are synthetic.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from shipdoc import cli, logprobs, smoke, spike
from shipdoc import eval as ev
from shipdoc.extract import HEADER_KEYS, ROW_KEYS, MockBackend

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
KEYED = CONFIGS / "spike_qwen35_4b_img_only.yaml"
SCORER = ROOT / "assignment" / "score.py"
needs_data = pytest.mark.skipif(
    not (SCORER.is_file() and (ROOT / "data" / "train" / "labels").is_dir()
         and (ROOT / "data" / "dev" / "labels").is_dir()),
    reason="assignment/score.py or data/{train,dev} absent (gitignored)",
)  # fmt: skip


def _meta(split: str) -> list[dict[str, Any]]:
    return json.loads((ROOT / "meta" / f"{split}.json").read_text(encoding="utf-8"))


def _pick(split: str) -> list[str]:
    """Deterministic mix per split: a multipage invoice, a waybill and the first other doc."""
    meta = _meta(split)
    multi = next(m["doc_id"] for m in meta if m["multipage"] and not m["waybill"])
    wb = next(m["doc_id"] for m in meta if m["waybill"])
    other = next(m["doc_id"] for m in meta if m["doc_id"] not in (multi, wb))
    return [multi, wb, other]


def _docs() -> list[str]:
    """Interleaved so a mid-run kill leaves docs of both splits done and both splits to do."""
    t, d = _pick("train"), _pick("dev")
    return [t[0], d[0], t[1], d[1], t[2], d[2]]


def _gold() -> dict[str, dict[str, Any]]:
    gold = ev.load_gold(ROOT / "data" / "train" / "labels")
    gold.update(ev.load_gold(ROOT / "data" / "dev" / "labels"))
    return gold


def _run(
    tmp_path: Path, run_id: str, backend: Any, ids: list[str] | None = None, **kw: Any
) -> dict[str, Any]:
    cfg = spike.load_config(KEYED)
    return spike.run_spike(
        cfg, ids or _docs(), "train+dev", run_id, backend, runs_root=tmp_path, logprobs=True, **kw
    )


def _trace(tmp_path: Path, run_id: str) -> list[dict[str, Any]]:
    path = tmp_path / run_id / "trace.jsonl"
    text = path.read_text(encoding="utf-8") if path.is_file() else ""  # none finished yet
    return [json.loads(ln) for ln in text.splitlines()]


def _stable(t: dict[str, Any]) -> str:
    """A trace line without the wall-clock fields (the only non-deterministic content)."""
    t = json.loads(json.dumps(t))
    for p in t["pages"]:
        p["meta"].pop("latency_s", None)
    return json.dumps(t, sort_keys=True)


# ---------------------------------------------------------------------- doc list


def test_zeroshot500_split_file_is_train_plus_dev_in_order() -> None:
    ids = json.loads((ROOT / "splits" / "zeroshot500.json").read_text(encoding="utf-8"))
    assert ids == [m["doc_id"] for s in ("train", "dev") for m in _meta(s)]
    assert len(ids) == len(set(ids)) == 500
    assert sum(i.startswith("train_") for i in ids) == 400


def test_split_names_and_doc_splits_validation(tmp_path: Path) -> None:
    assert spike.split_names("dev") == ["dev"]
    assert spike.split_names("train+dev") == ["train", "dev"]
    for bad in ("", "train+", "+dev", "dev+dev"):
        with pytest.raises(ValueError, match="bad --split"):
            spike.split_names(bad)
    for s, doc in (("a", "a_0"), ("b", "b_0")):
        (tmp_path / s / "images").mkdir(parents=True)
        (tmp_path / s / "images" / f"{doc}_p1.png").write_bytes(b"")
    assert spike.doc_splits(["b_0", "a_0"], ["a", "b"], tmp_path) == {"b_0": "b", "a_0": "a"}
    with pytest.raises(FileNotFoundError, match="no page images"):
        spike.doc_splits(["c_0"], ["a", "b"], tmp_path)
    (tmp_path / "b" / "images" / "a_0_p1.png").write_bytes(b"")
    with pytest.raises(ValueError, match="both"):
        spike.doc_splits(["a_0"], ["a", "b"], tmp_path)


def test_logprobs_with_compact_format_is_refused(tmp_path: Path) -> None:
    cfg = spike.load_config(CONFIGS / "spike_qwen35_4b_img_only_compact.yaml")
    with pytest.raises(ValueError, match="keyed"):
        spike.run_spike(cfg, ["x"], "dev", "r", MockBackend({}), runs_root=tmp_path, logprobs=True)


# ---------------------------------------------------------------------- mock end-to-end


@needs_data
def test_train_plus_dev_run_writes_all_outputs_with_logprobs(tmp_path: Path) -> None:
    ids = _docs()
    m = _run(tmp_path, "r", MockBackend(_gold()), ids)
    out = tmp_path / "r"
    for name in ("predictions.json", "trace.jsonl", "metrics.json", "progress.json"):
        assert (out / name).is_file(), name
    assert not list(out.glob("*.tmp"))
    traces = _trace(tmp_path, "r")
    assert [t["doc_id"] for t in traces] == ids
    assert [t["split"] for t in traces] == [d.split("_")[0] for d in ids]  # looked up per doc
    assert sorted(json.loads((out / "predictions.json").read_text())) == sorted(ids)
    # metrics: official scorer over the union of train+dev labels, plus a per-split block
    assert m["scored"] and m["n_docs"] == 6 and m["OVERALL"] == pytest.approx(1.0)
    assert m["split"] == "train+dev" and set(m["per_split"]) == {"train", "dev"}
    for s in ("train", "dev"):
        ps = m["per_split"][s]
        assert ps["documents"] == 3 and ps["OVERALL"] == pytest.approx(1.0)
        assert ps["OVERALL_ci95"]["lo"] <= ps["OVERALL"] <= ps["OVERALL_ci95"]["hi"]
    progress = json.loads((out / "progress.json").read_text())
    assert progress["status"] == "complete" and progress["done"] == progress["total"] == 6
    # trace pages: the new keys, and no bulky logprob_trace left in meta
    for t in traces:
        for p in t["pages"]:
            assert "logprob_trace" not in p["meta"]
            assert isinstance(p["field_logprobs"], list)
            assert set(p["token_logprobs"]) == {"ends", "lp", "lp_c"}
            assert len(p["token_logprobs"]["ends"]) == len(p["token_logprobs"]["lp"])
    # every emitted header / row field has a finite score (the smoke gate's assertion)
    chk = smoke.check_logprobs(traces)
    assert chk.passed, chk.detail


@needs_data
def test_field_logprobs_cover_exactly_the_emitted_fields(tmp_path: Path) -> None:
    _run(tmp_path, "r", MockBackend(_gold()))
    for t in _trace(tmp_path, "r"):
        for p in t["pages"]:
            got = {(e["scope"], e["row_idx"], e["field"]) for e in p["field_logprobs"]}
            want = {("header", None, k) for k in HEADER_KEYS}
            want |= {("row", i, k) for i in range(len(p["parsed"]["line_items"])) for k in ROW_KEYS}
            assert got == want
            for e in p["field_logprobs"]:
                assert e["n_tokens"] >= 1 and math.isfinite(e["min"]) and e["min"] <= e["mean"] <= 0


@needs_data
def test_logprob_capture_does_not_change_greedy_output(tmp_path: Path) -> None:
    cfg = spike.load_config(KEYED)
    ids = _docs()
    spike.run_spike(cfg, ids, "train+dev", "off", MockBackend(_gold()), runs_root=tmp_path)
    _run(tmp_path, "on", MockBackend(_gold()), ids)
    off, on = _trace(tmp_path, "off"), _trace(tmp_path, "on")
    for a, b in zip(off, on, strict=True):
        assert [p["raw_text"] for p in a["pages"]] == [p["raw_text"] for p in b["pages"]]
        assert [p["parsed"] for p in a["pages"]] == [p["parsed"] for p in b["pages"]]
        assert a["prediction"] == b["prediction"]
        assert all("field_logprobs" not in p for p in a["pages"])  # old readers see no new keys
    assert (tmp_path / "off" / "predictions.json").read_bytes() == (
        tmp_path / "on" / "predictions.json"
    ).read_bytes()


@needs_data
def test_logprob_runs_are_deterministic(tmp_path: Path) -> None:
    _run(tmp_path, "a", MockBackend(_gold()))
    _run(tmp_path, "b", MockBackend(_gold()))
    assert [_stable(t) for t in _trace(tmp_path, "a")] == [
        _stable(t) for t in _trace(tmp_path, "b")
    ]


@needs_data
def test_cli_train_plus_dev_logprobs_mock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SHIPDOC_RUNS_DIR", str(tmp_path))
    docs = tmp_path / "docs.json"
    docs.write_text(json.dumps(_docs()), encoding="utf-8")
    args = ["spike", "--config", str(KEYED), "--docs", str(docs), "--split", "train+dev",
            "--run-id", "cli", "--backend", "mock", "--logprobs"]  # fmt: skip
    assert cli.main(args) == 0
    assert "6 docs  OVERALL 100.00" in capsys.readouterr().out
    assert all("field_logprobs" in p for t in _trace(tmp_path, "cli") for p in t["pages"])


# ---------------------------------------------------------------------- resume


class Killed(BaseException):
    """Stands in for SIGKILL / a Colab disconnect: nothing in run_spike may catch it."""


class DyingBackend(MockBackend):
    def __init__(self, *a: Any, die_after: int, **k: Any) -> None:
        super().__init__(*a, **k)
        self.die_after = die_after

    def extract_page(self, *a: Any, **k: Any) -> Any:
        if self._calls >= self.die_after:
            raise Killed
        return super().extract_page(*a, **k)


def _pages_of(doc: str) -> int:
    return len(spike.doc_page_images(doc.split("_")[0], doc))


@needs_data
@pytest.mark.parametrize("die_after", [1, 3, 5])  # mid first doc / mid run / later
def test_killed_and_restarted_run_is_byte_identical_and_redoes_nothing(
    tmp_path: Path, die_after: int
) -> None:
    ids = _docs()
    _run(tmp_path, "ref", MockBackend(_gold()), ids)  # uninterrupted reference

    with pytest.raises(Killed):
        _run(tmp_path, "r", DyingBackend(_gold(), die_after=die_after), ids)
    partial = _trace(tmp_path, "r")
    n_done = len(partial)
    pages_done = sum(_pages_of(t["doc_id"]) for t in partial)
    assert 0 <= n_done < len(ids) and pages_done <= die_after  # the doc in flight is not recorded
    path = tmp_path / "r" / "trace.jsonl"
    before = path.read_bytes() if path.is_file() else b""

    backend = MockBackend(_gold())
    if n_done:
        with pytest.raises(FileExistsError, match="--resume"):  # restart without --resume: refused
            _run(tmp_path, "r", backend, ids)
    m = _run(tmp_path, "r", backend, ids, resume=True)

    traces = _trace(tmp_path, "r")
    assert [t["doc_id"] for t in traces] == ids  # nothing duplicated, nothing missing, same order
    assert backend._calls == sum(_pages_of(d) for d in ids[n_done:])  # finished docs not redone
    assert (
        (tmp_path / "r" / "trace.jsonl").read_bytes().startswith(before)
    )  # appended, not rewritten
    assert (tmp_path / "r" / "predictions.json").read_bytes() == (
        tmp_path / "ref" / "predictions.json"
    ).read_bytes()
    assert [_stable(t) for t in traces] == [_stable(t) for t in _trace(tmp_path, "ref")]
    assert m["n_docs"] == 6 and m["OVERALL"] == pytest.approx(1.0)
    progress = json.loads((tmp_path / "r" / "progress.json").read_text())
    assert progress["status"] == "complete" and progress["done"] == 6


@needs_data
@pytest.mark.parametrize("kind", ["partial_line", "full_line_without_newline"])
def test_torn_last_line_is_dropped_on_resume(tmp_path: Path, kind: str) -> None:
    """A kill mid-append leaves a partial line, or a full one that lost only its newline."""
    ids = _docs()
    _run(tmp_path, "ref", MockBackend(_gold()), ids)
    with pytest.raises(Killed):
        _run(tmp_path, "r", DyingBackend(_gold(), die_after=4), ids)
    path = tmp_path / "r" / "trace.jsonl"
    data = path.read_bytes()
    assert data.endswith(b"\n") and data.count(b"\n") >= 1
    if kind == "partial_line":
        path.write_bytes(data + data.splitlines()[0][:50])  # the head of a line, cut mid-write
    else:
        path.write_bytes(data[:-1])  # the last real line, without its newline
    _run(tmp_path, "r", MockBackend(_gold()), ids, resume=True)
    assert [t["doc_id"] for t in _trace(tmp_path, "r")] == ids
    assert (tmp_path / "r" / "predictions.json").read_bytes() == (
        tmp_path / "ref" / "predictions.json"
    ).read_bytes()


@needs_data
def test_corrupt_line_in_the_middle_still_raises(tmp_path: Path) -> None:
    ids = _docs()
    with pytest.raises(Killed):
        _run(tmp_path, "r", DyingBackend(_gold(), die_after=5), ids)
    path = tmp_path / "r" / "trace.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 2
    path.write_text("\n".join(["{broken", *lines[1:]]) + "\n", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        _run(tmp_path, "r", MockBackend(_gold()), ids, resume=True)


@needs_data
def test_predictions_are_rewritten_when_killed_between_append_and_rewrite(tmp_path: Path) -> None:
    ids = _docs()
    _run(tmp_path, "r", MockBackend(_gold()), ids)
    pred = tmp_path / "r" / "predictions.json"
    good = pred.read_bytes()
    pred.write_text("{}", encoding="utf-8")  # stale file; every doc is already in the trace
    backend = MockBackend(_gold())
    _run(tmp_path, "r", backend, ids, resume=True)
    assert backend._calls == 0 and pred.read_bytes() == good


# ---------------------------------------------------------------------- smoke assertion (g)


def _page(valid: bool = True, **over: Any) -> dict[str, Any]:
    parsed = {
        "doc_type": "invoice",
        "header": {"invoice_number": "A", "total_amount": None},
        "line_items": [dict.fromkeys(ROW_KEYS, "1")],
        "page_kind": "single",
    }
    fl = [
        {"scope": "header", "row_idx": None, "field": k, "min": -1.0, "mean": -0.5, "n_tokens": 2}
        for k in parsed["header"]
    ] + [
        {"scope": "row", "row_idx": 0, "field": k, "min": -1.0, "mean": -0.5, "n_tokens": 2}
        for k in ROW_KEYS
    ]
    page = {"page": 1, "json_valid": valid, "parsed": parsed, "field_logprobs": fl}
    page.update(over)
    return page


def _check(*pages: dict[str, Any]) -> smoke.Check:
    return smoke.check_logprobs([{"doc_id": "d", "pages": list(pages)}])


def test_smoke_logprob_check_passes_on_complete_finite_scores() -> None:
    c = _check(_page())
    assert c.passed and c.name == "g_field_logprobs" and "6 fields on 1 pages" in c.detail


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.pop("field_logprobs"),  # an old trace / capture never ran
        lambda p: p.update(field_logprobs=None),  # capture failed
        lambda p: p["field_logprobs"].pop(),  # one emitted field has no score
        lambda p: p["field_logprobs"][0].update(min=None),  # not finite
        lambda p: p["field_logprobs"][0].update(mean=float("nan")),
        lambda p: p["field_logprobs"][0].update(min=float("-inf")),
        lambda p: p["field_logprobs"][0].update(n_tokens=0),
        lambda p: p["field_logprobs"][1].update(mean=0.5),  # a probability above 1
    ],
)
def test_smoke_logprob_check_fails_closed(mutate: Any) -> None:
    page = _page()
    mutate(page)
    assert not _check(page).passed


def test_smoke_logprob_check_ignores_invalid_pages_but_needs_one_valid() -> None:
    assert _check(_page(), _page(valid=False, field_logprobs=None)).passed
    assert not _check(_page(valid=False)).passed  # no valid page at all: nothing verified


def test_check_smoke_adds_assertion_g_only_when_required() -> None:
    traces = [{"doc_id": "i0", "pages": [dict(_page(), raw_text="{}", meta={})]}]
    preds = {"i0": {"header": {"total_amount": "5"}, "line_items": [dict.fromkeys(ROW_KEYS, "1")]}}
    gold = {"i0": {"doc_type": "invoice", "line_items": [{}]}}
    base = smoke.check_smoke(traces, preds, gold, 1536)
    with_lp = smoke.check_smoke(traces, preds, gold, 1536, require_logprobs=True)
    assert [c.name for c in with_lp] == [*[c.name for c in base], "g_field_logprobs"]
    assert not any(c.name == "g_field_logprobs" for c in base)  # the 01 notebook is unaffected


def test_logprobs_module_is_the_documented_trace_schema() -> None:
    doc = (ROOT / "src" / "shipdoc" / "trace.py").read_text(encoding="utf-8")
    entry = logprobs.field_logprobs('{"header":{"a":"x"}}', [20], [-0.5], [-0.25])[0]
    for key in entry:
        assert key in doc, key  # every key a reader will see is described in trace.py
    assert "token_logprobs" in doc and "field_logprobs" in doc and "truncated" in doc
