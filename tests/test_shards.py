"""Document-level shards: assignment, per-shard runs and the merge back into one run folder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from _synth import SPECS, make_corpus

from shipdoc import runmeta, shardmerge, spike
from shipdoc.extract import MockBackend
from shipdoc.shard import assign_shards, parse_shard, shard_run_id

ROOT = Path(__file__).resolve().parents[1]
KEYED = ROOT / "configs" / "spike_qwen35_4b_img_only.yaml"
IDS = [s[0] for s in SPECS]
N_PAGES = {s[0]: s[1] for s in SPECS}
RUN = "zs"


# ---------------------------------------------------------------------- assignment (pure)


def test_parse_shard_and_run_ids() -> None:
    assert parse_shard("0/1") == (0, 1) and parse_shard(" 3/4 ") == (3, 4)
    for bad in ("1/1", "2/2", "0/0", "-1/2", "a/b", "1", "1/", "0/2/3", ""):
        with pytest.raises(ValueError, match="bad shard"):
            parse_shard(bad)
    assert shard_run_id(RUN, "0/1") == RUN  # unsharded keeps its own folder
    assert shard_run_id(RUN, "1/2") == "zs_shard1of2"


def test_assignment_is_balanced_deterministic_and_keeps_input_order() -> None:
    pages = [(d, N_PAGES[d]) for d in IDS]
    shards = assign_shards(pages, 2)
    assert sorted(d for s in shards for d in s) == sorted(IDS)  # a partition
    for s in shards:
        assert s == [d for d in IDS if d in s]  # input order inside a shard
    loads = [sum(N_PAGES[d] for d in s) for s in shards]
    assert max(loads) - min(loads) <= max(N_PAGES.values())
    assert assign_shards(pages, 2) == shards
    rev = assign_shards(list(reversed(pages)), 2)  # same owners whatever the input order
    assert [sorted(x) for x in rev] == [sorted(x) for x in shards]
    assert assign_shards(pages, 1) == [IDS]
    assert assign_shards([], 3) == [[], [], []]


def test_greedy_rule_is_the_documented_one() -> None:
    # sorted by (pages desc, id): a(4) b(3) c(3) d(1); a->0, b->1, c->1? no: loads 4,3 -> c to 1.
    pages = [("a", 4), ("b", 3), ("c", 3), ("d", 1)]
    assert assign_shards(pages, 2) == [["a", "d"], ["b", "c"]]  # loads 5 and 6
    with pytest.raises(ValueError, match="duplicate"):
        assign_shards([("a", 1), ("a", 2)], 2)
    with pytest.raises(ValueError, match="K must"):
        assign_shards(pages, 0)


def test_balance_on_the_real_500_docs_when_the_labels_are_present() -> None:
    labels = {s: ROOT / "data" / s / "labels" for s in ("train", "dev")}
    if not all(p.is_dir() for p in labels.values()):
        pytest.skip("data/{train,dev}/labels absent (gitignored)")
    ids = json.loads((ROOT / "splits" / "zeroshot500.json").read_text(encoding="utf-8"))
    pages = []
    for d in ids:
        g = json.loads((labels[d.split("_")[0]] / f"{d}.json").read_text(encoding="utf-8"))
        pages.append((d, len(g["pages"])))
    for k in (2, 3, 4):
        loads = [sum(n for d, n in pages if d in set(s)) for s in assign_shards(pages, k)]
        assert sum(loads) == sum(n for _, n in pages) == 671
        assert max(loads) - min(loads) <= 1, (k, loads)


# ---------------------------------------------------------------------- shard runs and the merge


@pytest.fixture()
def world(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    root = tmp_path / "data"
    return root, make_corpus(root, labels=False)


def _run(
    tmp: Path, world: tuple[Path, dict[str, Any]], run_id: str, shard: str = "0/1", **kw: Any
) -> dict[str, Any]:
    cfg = spike.load_config(KEYED)
    return spike.run_spike(
        cfg, IDS, "dev", shard_run_id(run_id, shard), MockBackend(world[1], fixed_latency_s=0.5),
        runs_root=tmp / "runs", data_root=world[0], logprobs=True, shard=shard,
        base_run_id=run_id, **kw,
    )  # fmt: skip


def _merge(tmp: Path, world: tuple[Path, dict[str, Any]], run_id: str = RUN, **kw: Any) -> Any:
    return shardmerge.merge_shards(
        spike.load_config(KEYED), IDS, "dev", run_id, 2,
        runs_root=tmp / "runs", data_root=world[0], **kw,
    )  # fmt: skip


def _bytes(tmp: Path, run: str, name: str) -> bytes:
    return (tmp / "runs" / run / name).read_bytes()


@pytest.mark.parametrize("batch_size", [1])
def test_merged_two_shards_are_byte_identical_to_an_unsharded_run(
    tmp_path: Path, world: tuple[Path, dict[str, Any]], batch_size: int
) -> None:
    other = tmp_path / "unsharded"
    _run(other, world, RUN, batch_size=batch_size)
    for i in (0, 1):
        _run(tmp_path, world, RUN, f"{i}/2", batch_size=batch_size)
    shards = [json.loads(_bytes(tmp_path, f"{RUN}_shard{i}of2", "progress.json")) for i in (0, 1)]
    assert all(s["status"] == "complete" for s in shards)
    assert sum(s["done"] for s in shards) == len(IDS)
    _merge(tmp_path, world)
    for name in ("trace.jsonl", "predictions.json", "metrics.json", "progress.json"):
        assert _bytes(tmp_path, RUN, name) == _bytes(other, RUN, name), name


def test_batched_shards_merge_to_the_same_documents_as_an_unsharded_batched_run(
    tmp_path: Path, world: tuple[Path, dict[str, Any]]
) -> None:
    _run(tmp_path / "unsharded", world, RUN, batch_size=4)
    for i in (0, 1):
        _run(tmp_path, world, RUN, f"{i}/2", batch_size=4)
    _merge(tmp_path, world)
    a = json.loads(_bytes(tmp_path, RUN, "predictions.json"))
    b = json.loads(_bytes(tmp_path / "unsharded", RUN, "predictions.json"))
    assert a == b and list(a) == IDS
    merged = runmeta.read_manifest(tmp_path / "runs" / RUN)
    assert merged is not None and merged["batch_size"] == 4 and merged["shard"] == "merged"
    assert merged["merged_from"] == [f"{RUN}_shard0of2", f"{RUN}_shard1of2"]
    assert (
        runmeta.require_same_batch_size(
            tmp_path / "runs" / RUN, tmp_path / "unsharded" / "runs" / RUN
        )
        == 4
    )


def test_shards_cover_every_doc_once_and_balance_pages(
    tmp_path: Path, world: tuple[Path, dict[str, Any]]
) -> None:
    for i in (0, 1):
        _run(tmp_path, world, RUN, f"{i}/2")
    docs = [
        [json.loads(ln)["doc_id"] for ln in _bytes(tmp_path, f"{RUN}_shard{i}of2", "trace.jsonl")
         .decode().splitlines()]
        for i in (0, 1)
    ]  # fmt: skip
    assert sorted(docs[0] + docs[1]) == sorted(IDS) and not set(docs[0]) & set(docs[1])
    assert docs == assign_shards([(d, N_PAGES[d]) for d in IDS], 2)


def _two_shards(tmp: Path, world: tuple[Path, dict[str, Any]], **kw: Any) -> None:
    for i in (0, 1):
        _run(tmp, world, RUN, f"{i}/2", **kw)


def _edit_manifest(tmp: Path, index: int, changes: dict[str, Any]) -> None:
    d = tmp / "runs" / f"{RUN}_shard{index}of2"
    m = runmeta.read_manifest(d)
    assert m is not None
    m.update(changes)
    runmeta.write_manifest(d, m)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"batch_size": 2}, "batch_size"),
        ({"code_sha": "deadbeef"}, "code_sha"),
        ({"config": {"name": "x", "hash": "0" * 16}}, "config.hash"),
        ({"model": {"id": "mock", "revision": "other"}}, "model.revision"),
        ({"logprobs": False}, "logprobs"),
        ({"docs_sha": "0" * 16}, "docs_sha"),
        ({"seed": 7}, "seed"),
        ({"shard": "0/2"}, "manifest says shard"),
    ],
)
def test_merge_refuses_shards_that_are_not_the_same_experiment(
    tmp_path: Path, world: tuple[Path, dict[str, Any]], changes: dict[str, Any], message: str
) -> None:
    _two_shards(tmp_path, world)
    _edit_manifest(tmp_path, 1, changes)
    with pytest.raises(ValueError, match=message):
        _merge(tmp_path, world)
    assert not (tmp_path / "runs" / RUN).exists()  # nothing written


def test_merge_refuses_incomplete_missing_duplicate_and_miscounted_shards(
    tmp_path: Path, world: tuple[Path, dict[str, Any]]
) -> None:
    _two_shards(tmp_path, world)
    s0, s1 = (tmp_path / "runs" / f"{RUN}_shard{i}of2" for i in (0, 1))
    with pytest.raises(ValueError, match="expected 501"):
        _merge(tmp_path, world, expected_docs=501)
    with pytest.raises(ValueError, match="pages merged, expected 672"):
        _merge(tmp_path, world, expected_pages=672)

    good_progress = (s1 / "progress.json").read_text()
    (s1 / "progress.json").write_text(json.dumps({"status": "running"}))
    with pytest.raises(ValueError, match="not complete"):
        _merge(tmp_path, world)
    (s1 / "progress.json").write_text(good_progress)

    trace0, trace1 = (s0 / "trace.jsonl").read_text(), (s1 / "trace.jsonl").read_text()
    (s1 / "trace.jsonl").write_text("".join(trace1.splitlines(True)[:-1]))  # a document lost
    with pytest.raises(ValueError, match="missing"):
        _merge(tmp_path, world)
    (s1 / "trace.jsonl").write_text(trace1 + trace0.splitlines(True)[0])  # a document twice
    with pytest.raises(ValueError, match="duplicate document"):
        _merge(tmp_path, world)
    (s1 / "trace.jsonl").write_text(trace1)

    (s1 / "manifest.json").rename(s1 / "manifest.gone")
    with pytest.raises(ValueError, match="no manifest"):
        _merge(tmp_path, world)
    assert not (tmp_path / "runs" / RUN).exists()
    (s1 / "manifest.gone").rename(s1 / "manifest.json")
    _merge(tmp_path, world)  # and with everything restored it merges
    assert (tmp_path / "runs" / RUN / "trace.jsonl").is_file()


def test_merge_refuses_another_doc_list_or_config(
    tmp_path: Path, world: tuple[Path, dict[str, Any]]
) -> None:
    _two_shards(tmp_path, world)
    cfg = spike.load_config(KEYED)
    with pytest.raises(ValueError, match="docs_sha"):
        shardmerge.merge_shards(
            cfg, IDS[:-1], "dev", RUN, 2, runs_root=tmp_path / "runs", data_root=world[0]
        )
    other = type(cfg)(cfg.name, cfg.arm, cfg.backend, cfg.prompt_version, {**cfg.raw, "x": 1})
    with pytest.raises(ValueError, match="config"):
        shardmerge.merge_shards(
            other, IDS, "dev", RUN, 2, runs_root=tmp_path / "runs", data_root=world[0]
        )


def test_resuming_a_shard_continues_that_shard_only(
    tmp_path: Path, world: tuple[Path, dict[str, Any]]
) -> None:
    class Killed(BaseException):
        pass

    class Dies(MockBackend):
        def extract_page(self, *a: Any, **k: Any) -> Any:
            if self._calls >= 2:
                raise Killed
            return super().extract_page(*a, **k)

    cfg = spike.load_config(KEYED)
    args = (cfg, IDS, "dev", f"{RUN}_shard0of2")
    kw: dict[str, Any] = {
        "runs_root": tmp_path / "runs", "data_root": world[0], "shard": "0/2", "base_run_id": RUN,
    }  # fmt: skip
    with pytest.raises(Killed):
        spike.run_spike(*args, Dies(world[1], fixed_latency_s=0.5), **kw)
    spike.run_spike(*args, MockBackend(world[1], fixed_latency_s=0.5), resume=True, **kw)
    with pytest.raises(ValueError, match="shard"):  # shard 0's folder cannot become shard 1
        spike.run_spike(
            cfg, IDS, "dev", f"{RUN}_shard0of2", MockBackend(world[1]),
            runs_root=tmp_path / "runs", data_root=world[0], shard="1/2", resume=True,
        )  # fmt: skip


needs_data = pytest.mark.skipif(
    not (
        (ROOT / "assignment" / "score.py").is_file()
        and (ROOT / "data" / "train" / "labels").is_dir()
        and (ROOT / "data" / "dev" / "labels").is_dir()
    ),
    reason="assignment/score.py or data/{train,dev} absent (gitignored)",
)


@needs_data
def test_merged_shards_equal_an_unsharded_run_on_real_gold_with_scoring(tmp_path: Path) -> None:
    from test_zeroshot500 import _docs, _gold

    ids, gold, cfg = _docs(), _gold(), spike.load_config(KEYED)

    def run(base: Path, run_id: str, shard: str) -> None:
        spike.run_spike(
            cfg, ids, "train+dev", shard_run_id(run_id, shard),
            MockBackend(gold, fixed_latency_s=0.5), runs_root=base / "runs",
            logprobs=True, shard=shard, base_run_id=run_id,
        )  # fmt: skip

    run(tmp_path / "u", RUN, "0/1")
    run(tmp_path, RUN, "0/2")
    run(tmp_path, RUN, "1/2")
    shardmerge.merge_shards(cfg, ids, "train+dev", RUN, 2, runs_root=tmp_path / "runs")
    for name in ("trace.jsonl", "predictions.json", "metrics.json"):
        assert (tmp_path / "runs" / RUN / name).read_bytes() == (
            tmp_path / "u" / "runs" / RUN / name
        ).read_bytes(), name
    metrics = json.loads((tmp_path / "runs" / RUN / "metrics.json").read_text())
    assert metrics["scored"] and metrics["OVERALL"] == pytest.approx(1.0) and metrics["per_split"]


def test_merge_shards_cli(
    tmp_path: Path,
    world: tuple[Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from shipdoc import cli

    _two_shards(tmp_path, world)
    monkeypatch.setenv("SHIPDOC_DATA_DIR", str(world[0]))
    monkeypatch.setenv("SHIPDOC_RUNS_DIR", str(tmp_path / "runs"))
    docs = tmp_path / "docs.json"
    docs.write_text(json.dumps(IDS))
    args = ["merge-shards", "--config", str(KEYED), "--docs", str(docs), "--split", "dev",
            "--run-id", RUN, "--shards", "2"]  # fmt: skip
    assert cli.main(args) == 0
    assert f"merged 2 shards into {RUN}: {len(IDS)} docs" in capsys.readouterr().out
    with pytest.raises(ValueError, match="expected 12"):
        cli.main([*args, "--expected-docs", "12"])
