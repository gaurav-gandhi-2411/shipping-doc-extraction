"""Batching: the expected-length key, the batch plan, and the batched runner on the mock backend.

The mock gives every page the same answer whatever it is batched with (the property a real
batched backend only approximates), so what these tests pin is the RUNNER: order, windows,
per-document tracing, fallback, resume and the untouched batch-1 path.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from _synth import SPECS, make_corpus
from PIL import Image

from shipdoc import batching, runmeta, spike
from shipdoc.batching import PageJob
from shipdoc.extract import MockBackend, PageRequest, schema_for
from shipdoc.prompts import build_prompt

ROOT = Path(__file__).resolve().parents[1]
KEYED = ROOT / "configs" / "spike_qwen35_4b_img_only.yaml"
MAX_PIXELS = 1280 * 32 * 32
IDS = [s[0] for s in SPECS]
N_PAGES = {s[0]: s[1] for s in SPECS}


# ---------------------------------------------------------------------- key / plan (pure)


def _job(order: int, n: int, idx: int, w: int = 1000, h: int = 1000, size: int = 0,
         ext: str = ".png") -> PageJob:  # fmt: skip
    return PageJob(order, f"d{order}", idx, n, Path(f"x{ext}"), w, h, size)


def test_roles_by_page_order() -> None:
    assert [_job(0, n, i).role for n, i in ((1, 0), (3, 0), (3, 1), (3, 2))] == [
        "single",
        "first",
        "middle",
        "last",
    ]


def test_key_orders_multipage_then_doc_length_then_extension_then_file_size() -> None:
    def key(j: PageJob) -> Any:
        return batching.expected_length_key(j, MAX_PIXELS)

    single_big, single_small = _job(0, 1, 0, size=900), _job(1, 1, 0, size=100)
    jpg_single = _job(2, 1, 0, size=5000, ext=".jpg")
    first3, last3, first2 = _job(3, 3, 0, size=10), _job(4, 3, 2, size=10), _job(5, 2, 0, size=99)
    ordered = sorted([single_small, last3, jpg_single, single_big, first2, first3], key=key)
    # multipage first (3-page docs before 2-page), then by extension (.jpg < .png), then bigger
    # files first: sizes are only compared within one extension
    assert [j.order for j in ordered] == [3, 4, 5, 2, 0, 1]
    # same document length and size: first before last before middle before single
    tie = [_job(10, 3, 1), _job(11, 3, 2), _job(12, 3, 0)]
    assert [j.role for j in sorted(tie, key=key)] == ["first", "last", "middle"]
    # then the larger image (more visual tokens), then input order
    a, b, c = _job(20, 1, 0, 800, 800), _job(21, 1, 0, 900, 900), _job(22, 1, 0, 900, 900)
    assert [j.order for j in sorted([a, c, b], key=key)] == [21, 22, 20]


def test_visual_token_estimate_is_capped_at_max_pixels() -> None:
    assert batching.est_visual_tokens(_job(0, 1, 0, 5000, 5000), MAX_PIXELS) == MAX_PIXELS // 1024
    assert batching.est_visual_tokens(_job(0, 1, 0, 320, 320), MAX_PIXELS) == 100
    assert batching.est_visual_tokens(_job(0, 1, 0, 0, 0), MAX_PIXELS) == 0  # unknown size


def test_plan_batches_covers_every_page_once_and_is_deterministic() -> None:
    jobs = [_job(i, 1 + i % 3, i % (1 + i % 3), 500 + 37 * i, 700) for i in range(23)]
    plan = batching.plan_batches(jobs, 4, MAX_PIXELS)
    assert [len(b) for b in plan] == [4, 4, 4, 4, 4, 3]
    flat = [j.order for b in plan for j in b]
    assert sorted(flat) == list(range(23)) and len(set(flat)) == 23
    assert plan == batching.plan_batches(list(reversed(jobs)), 4, MAX_PIXELS)  # input order free
    with pytest.raises(ValueError, match="batch_size"):
        batching.plan_batches(jobs, 0, MAX_PIXELS)


def test_doc_windows_are_runs_of_whole_documents_in_order() -> None:
    pages = {"a": 3, "b": 1, "c": 2, "d": 4, "e": 1, "f": 1}
    windows = batching.doc_windows(list(pages), pages.__getitem__, 1)  # target 8 pages
    assert windows == [["a", "b", "c", "d"], ["e", "f"]]
    assert batching.doc_windows(list(pages), pages.__getitem__, 2) == [list(pages)]  # 16 pages
    assert [d for w in windows for d in w] == list(pages)
    assert batching.doc_windows([], pages.__getitem__, 2) == []


def test_fallback_ladder_halves_down_to_one() -> None:
    assert [batching.next_smaller_batch(n) for n in (2, 3, 4, 5, 8, 16)] == [1, 2, 2, 4, 4, 8]
    with pytest.raises(ValueError, match="no smaller"):
        batching.next_smaller_batch(1)
    jobs = [_job(i, 1, 0) for i in range(7)]
    chunks = batching.split_for_fallback(jobs)
    assert [len(c) for c in chunks] == [4, 3]
    assert [j.order for c in chunks for j in c] == list(range(7))


# ---------------------------------------------------------------------- runner on the mock


@pytest.fixture()
def corpus(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    root = tmp_path / "data"
    return root, make_corpus(root, labels=False)


def _mock(gold: dict[str, Any], **kw: Any) -> MockBackend:
    return MockBackend(gold, fixed_latency_s=0.5, **kw)


def _run(
    tmp_path: Path,
    corpus: tuple[Path, dict[str, Any]],
    run_id: str,
    backend: Any,
    ids: list[str] | None = None,
    **kw: Any,
) -> dict[str, Any]:
    cfg = spike.load_config(KEYED)
    kw.setdefault("logprobs", True)
    return spike.run_spike(
        cfg, ids or IDS, "dev", run_id, backend,
        runs_root=tmp_path / "runs", data_root=corpus[0], **kw,
    )  # fmt: skip


def _traces(tmp_path: Path, run_id: str) -> list[dict[str, Any]]:
    path = tmp_path / "runs" / run_id / "trace.jsonl"
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    return [json.loads(ln) for ln in text.splitlines()]


def _norm(t: dict[str, Any]) -> str:
    """A trace line without what legitimately differs between batch sizes (timing, batch keys)."""
    t = json.loads(json.dumps(t))
    for p in t["pages"]:
        for k in ("latency_s", "batch_size", "batch_latency_s", "batch_fallback", "exec_batch"):
            p["meta"].pop(k, None)
    return json.dumps(t, sort_keys=True)


def _legacy_pages(gold: dict[str, Any], root: Path, doc: str) -> list[dict[str, Any]]:
    """The pre-batching per-page loop, written out: what batch size 1 must keep producing."""
    cfg = spike.load_config(KEYED)
    fmt, backend = cfg.backend.output_format, _mock(gold)
    backend.capture_logprobs = True
    images = spike.doc_page_images("dev", doc, root)
    pages = []
    for idx, path in enumerate(images):
        backend.set_context(doc, idx, len(images))
        with Image.open(path) as im:
            image = im.convert("RGB")
        raw, parsed, meta = backend.extract_page(
            image, build_prompt(idx, len(images), fmt), schema_for(fmt), None
        )
        meta.pop("logprob_trace")
        pages.append({"raw_text": raw, "parsed": parsed, "meta": meta})
    return pages


def test_batch_size_one_is_the_unbatched_path_byte_for_byte(
    tmp_path: Path, corpus: tuple[Path, dict[str, Any]]
) -> None:
    gold = corpus[1]
    explicit, default = _mock(gold), _mock(gold)
    _run(tmp_path, corpus, "one", explicit, batch_size=1)
    _run(tmp_path, corpus, "dflt", default)  # no batch_size at all: a new run is batch size 1
    a = (tmp_path / "runs" / "one" / "trace.jsonl").read_bytes()
    b = (tmp_path / "runs" / "dflt" / "trace.jsonl").read_bytes()
    assert a == b
    assert (tmp_path / "runs" / "one" / "predictions.json").read_bytes() == (
        tmp_path / "runs" / "dflt" / "predictions.json"
    ).read_bytes()
    for t in _traces(tmp_path, "one"):  # the golden: the legacy loop's pages, meta included
        legacy = _legacy_pages(gold, corpus[0], t["doc_id"])
        for got, want in zip(t["pages"], legacy, strict=True):
            assert (got["raw_text"], got["parsed"], got["meta"]) == (
                want["raw_text"],
                want["parsed"],
                want["meta"],
            )
        assert all("batch_size" not in p["meta"] for p in t["pages"])  # no new keys at size 1


@pytest.mark.parametrize("batch_size", [2, 3, 4, 8, 64])
def test_batched_run_gives_the_same_documents_in_the_same_order(
    tmp_path: Path, corpus: tuple[Path, dict[str, Any]], batch_size: int
) -> None:
    gold = corpus[1]
    _run(tmp_path, corpus, "ref", _mock(gold), batch_size=1)
    spy = _run(tmp_path, corpus, "b", _mock(gold), batch_size=batch_size)
    ref, got = _traces(tmp_path, "ref"), _traces(tmp_path, "b")
    assert [t["doc_id"] for t in got] == IDS  # input order, whatever the batching did
    assert [_norm(t) for t in got] == [_norm(t) for t in ref]
    assert (tmp_path / "runs" / "b" / "predictions.json").read_bytes() == (
        tmp_path / "runs" / "ref" / "predictions.json"
    ).read_bytes()
    assert list(json.loads((tmp_path / "runs" / "b" / "predictions.json").read_text())) == IDS
    assert spy["n_docs"] == len(IDS) and spy["n_pages"] == sum(N_PAGES.values())
    pages = [p for t in got for p in t["pages"]]
    assert all(1 <= p["meta"]["batch_size"] <= batch_size for p in pages)
    assert all("logprob_trace" not in p["meta"] and p["field_logprobs"] for p in pages)


class Spy(MockBackend):
    """Records the pages of every batched call and the documents already traced at that moment."""

    def __init__(self, *a: Any, trace_path: Path, **k: Any) -> None:
        super().__init__(*a, **k)
        self.trace_path, self.calls, self.done = trace_path, [], set()

    def extract_pages(self, requests: Any) -> Any:
        traced = []
        if self.trace_path.is_file():
            traced = [json.loads(ln)["doc_id"] for ln in self.trace_path.read_text().splitlines()]
        for d in traced:  # a document is traced only once ALL its pages have been run
            assert all((d, p) in self.done for p in range(N_PAGES[d])), d
        self.calls.append([r.context for r in requests])
        out = super().extract_pages(requests)
        self.done |= {(r.context[0], r.context[1]) for r in requests}
        return out


def test_pages_are_sorted_into_batches_but_documents_leave_in_input_order(
    tmp_path: Path, corpus: tuple[Path, dict[str, Any]]
) -> None:
    spy = Spy(corpus[1], fixed_latency_s=0.5, trace_path=tmp_path / "runs" / "s" / "trace.jsonl")
    _run(tmp_path, corpus, "s", spy, batch_size=2)
    flat = [c for call in spy.calls for c in call]
    assert sorted(flat) == sorted((d, p, N_PAGES[d]) for d in IDS for p in range(N_PAGES[d]))
    # the first window's first batch holds pages of multi-page documents (the long-output kind)
    assert all(n > 1 for _d, _p, n in spy.calls[0])
    assert flat != [(d, p, N_PAGES[d]) for d in IDS for p in range(N_PAGES[d])]  # really reordered
    assert [t["doc_id"] for t in _traces(tmp_path, "s")] == IDS


def test_batched_backend_is_required_above_batch_one(
    tmp_path: Path, corpus: tuple[Path, dict[str, Any]]
) -> None:
    class OnePage:
        model_id, revision = "one", "one"

        def extract_page(self, *a: Any) -> Any:  # pragma: no cover - never reached
            raise AssertionError

    with pytest.raises(ValueError, match="no extract_pages"):
        _run(tmp_path, corpus, "x", OnePage(), batch_size=2)
    with pytest.raises(ValueError, match="batch_size must be"):
        _run(tmp_path, corpus, "y", _mock(corpus[1]), batch_size=0)


# ---------------------------------------------------------------------- fallback


class OomAbove(MockBackend):
    """Raises a CUDA-OOM-like error for any call of more than `limit` pages."""

    def __init__(self, *a: Any, limit: int, **k: Any) -> None:
        super().__init__(*a, **k)
        self.limit, self.sizes, self.released = limit, [], 0

    def extract_pages(self, requests: Any) -> Any:
        self.sizes.append(len(requests))
        if len(requests) > self.limit:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return super().extract_pages(requests)

    def release_memory(self) -> None:
        self.released += 1


def test_oom_falls_back_to_smaller_batches_and_loses_no_page(
    tmp_path: Path, corpus: tuple[Path, dict[str, Any]]
) -> None:
    gold = corpus[1]
    _run(tmp_path, corpus, "ref", _mock(gold), batch_size=1)
    oom = OomAbove(gold, fixed_latency_s=0.5, limit=2)
    _run(tmp_path, corpus, "f", oom, batch_size=8)
    assert [_norm(t) for t in _traces(tmp_path, "f")] == [
        _norm(t) for t in _traces(tmp_path, "ref")
    ]
    assert oom.released >= 1 and max(s for s in oom.sizes if s <= 2) == 2
    notes = [
        n
        for t in _traces(tmp_path, "f")
        for p in t["pages"]
        for n in p["meta"].get("batch_fallback", [])
    ]
    assert notes and {n["from"] for n in notes} <= {8, 7, 6, 5, 4, 3}
    assert all(n["to"] < n["from"] and "out of memory" in n["error"] for n in notes)
    pages = [p for t in _traces(tmp_path, "f") for p in t["pages"]]
    assert len(pages) == sum(N_PAGES.values())
    assert all(p["meta"]["batch_fallback"] for p in pages if p["meta"].get("batch_size", 1) < 8)


def test_fallback_can_be_switched_off_and_batch_one_failures_still_raise(
    tmp_path: Path, corpus: tuple[Path, dict[str, Any]]
) -> None:
    with pytest.raises(RuntimeError, match="out of memory"):
        _run(tmp_path, corpus, "nf", OomAbove(corpus[1], limit=2), batch_size=4, fallback=False)

    class Dead(OomAbove):
        def extract_page(self, *a: Any, **k: Any) -> Any:
            raise RuntimeError("CUDA out of memory at batch 1")

    with pytest.raises(RuntimeError, match="at batch 1"):
        _run(tmp_path, corpus, "dead", Dead(corpus[1], limit=1), batch_size=4)


# ---------------------------------------------------------------------- manifest / resume


class Killed(BaseException):
    """SIGKILL / Colab disconnect: nothing in run_spike may catch it."""


class DieAfter(MockBackend):
    def __init__(self, *a: Any, calls: int, **k: Any) -> None:
        super().__init__(*a, **k)
        self.left, self.sizes, self.docs = calls, [], []

    def extract_pages(self, requests: Any) -> Any:
        if self.left <= 0:
            raise Killed
        self.left -= 1
        self.sizes.append(len(requests))
        self.docs += [r.context[0] for r in requests]
        return super().extract_pages(requests)


def test_manifest_records_the_experiment(
    tmp_path: Path, corpus: tuple[Path, dict[str, Any]]
) -> None:
    _run(tmp_path, corpus, "m", _mock(corpus[1]), batch_size=4)
    m = runmeta.read_manifest(tmp_path / "runs" / "m")
    cfg = spike.load_config(KEYED)
    assert m is not None
    assert m["batch_size"] == 4 and m["batch_size_source"] == "manual" and m["bench"] is None
    assert m["config"] == {"name": cfg.name, "hash": cfg.config_hash}
    assert m["model"] == {"id": "mock", "revision": "mock"} and m["seed"] == cfg.backend.seed
    assert m["code_sha"] == spike.git_commit() and m["shard"] == "0/1"
    assert m["docs_sha"] == runmeta.docs_sha(IDS) and m["n_docs"] == len(IDS)
    _run(tmp_path, corpus, "d", _mock(corpus[1]))
    d = runmeta.read_manifest(tmp_path / "runs" / "d")
    assert d is not None and d["batch_size"] == 1 and d["batch_size_source"] == "default"


def test_resume_reuses_the_stored_batch_size_and_never_silently_changes_it(
    tmp_path: Path, corpus: tuple[Path, dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(batching, "WINDOW_BATCHES", 2)  # windows of 8 pages: five documents
    gold = corpus[1]
    _run(tmp_path, corpus, "ref", _mock(gold), batch_size=4)
    dying = DieAfter(gold, fixed_latency_s=0.5, calls=2)  # one window done, the next one dies
    with pytest.raises(Killed):
        _run(tmp_path, corpus, "r", dying, batch_size=4)
    partial = _traces(tmp_path, "r")
    assert 0 < len(partial) < len(IDS)
    assert [t["doc_id"] for t in partial] == IDS[: len(partial)]  # a prefix, in input order
    before = (tmp_path / "runs" / "r" / "trace.jsonl").read_bytes()

    with pytest.raises(ValueError, match="batch_size is 4"):  # a different size is refused
        _run(tmp_path, corpus, "r", _mock(gold), batch_size=2, resume=True)
    assert (tmp_path / "runs" / "r" / "trace.jsonl").read_bytes() == before

    resumed = DieAfter(gold, fixed_latency_s=0.5, calls=99)
    _run(tmp_path, corpus, "r", resumed, resume=True)  # None = the stored size
    assert max(resumed.sizes) == 4
    assert (tmp_path / "runs" / "r" / "trace.jsonl").read_bytes().startswith(before)
    assert [_norm(t) for t in _traces(tmp_path, "r")] == [
        _norm(t) for t in _traces(tmp_path, "ref")
    ]
    m = runmeta.read_manifest(tmp_path / "runs" / "r")
    assert m is not None and m["batch_size"] == 4
    assert not {t["doc_id"] for t in partial} & set(resumed.docs)  # finished documents not redone


def test_resume_of_a_pre_manifest_run_is_batch_size_one(
    tmp_path: Path, corpus: tuple[Path, dict[str, Any]]
) -> None:
    gold = corpus[1]
    with pytest.raises(Killed):
        _run(tmp_path, corpus, "old", DyingOnePage(gold, die_after=3), batch_size=1)
    (tmp_path / "runs" / "old" / "manifest.json").unlink()  # as a run from before manifests
    with pytest.raises(ValueError, match="pre-manifest"):
        _run(tmp_path, corpus, "old", _mock(gold), batch_size=4, resume=True)
    _run(tmp_path, corpus, "old", _mock(gold), resume=True)
    m = runmeta.read_manifest(tmp_path / "runs" / "old")
    assert m is not None and m["batch_size"] == 1


class DyingOnePage(MockBackend):
    def __init__(self, *a: Any, die_after: int, **k: Any) -> None:
        super().__init__(*a, fixed_latency_s=0.5, **k)
        self.die_after = die_after

    def extract_page(self, *a: Any, **k: Any) -> Any:
        if self._calls >= self.die_after:
            raise Killed
        return super().extract_page(*a, **k)


def test_resume_refuses_a_different_config_model_or_shard(
    tmp_path: Path, corpus: tuple[Path, dict[str, Any]]
) -> None:
    gold = corpus[1]
    with pytest.raises(Killed):
        _run(tmp_path, corpus, "c", DieAfter(gold, fixed_latency_s=0.5, calls=1), batch_size=2)
    other = replace(spike.load_config(KEYED), raw={**spike.load_config(KEYED).raw, "x": 1})
    with pytest.raises(ValueError, match="config.hash"):
        spike.run_spike(
            other, IDS, "dev", "c", _mock(gold), runs_root=tmp_path / "runs",
            data_root=corpus[0], resume=True,
        )  # fmt: skip
    with pytest.raises(ValueError, match="shard"):
        _run(tmp_path, corpus, "c", _mock(gold), resume=True, shard="0/2")
    changed = _mock(gold)
    changed.revision = "other"
    with pytest.raises(ValueError, match="model.revision"):
        _run(tmp_path, corpus, "c", changed, resume=True)


def test_require_same_batch_size_between_dev_and_test_runs(
    tmp_path: Path, corpus: tuple[Path, dict[str, Any]]
) -> None:
    gold = corpus[1]
    _run(tmp_path, corpus, "dev4", _mock(gold), batch_size=4)
    _run(tmp_path, corpus, "test4", _mock(gold), batch_size=4)
    _run(tmp_path, corpus, "test2", _mock(gold), batch_size=2)
    runs = tmp_path / "runs"
    assert runmeta.require_same_batch_size(runs / "dev4", runs / "test4") == 4
    with pytest.raises(ValueError, match="batch size differs"):
        runmeta.require_same_batch_size(runs / "dev4", runs / "test2")
    with pytest.raises(ValueError, match="no manifest"):
        runmeta.require_same_batch_size(runs / "dev4", runs / "absent")


def test_page_request_defaults() -> None:
    r = PageRequest(None, "p", {})
    assert r.ocr_text is None and r.context is None
