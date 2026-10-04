from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from shipdoc import cli, paths, spike
from shipdoc.extract import MockBackend, MockCorruption
from shipdoc.prompts import PROMPT_VERSION

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
GOLD_DIR = ROOT / "data" / "dev" / "labels"
IMG_DIR = ROOT / "data" / "dev" / "images"
SCORER = ROOT / "assignment" / "score.py"
needs_data = pytest.mark.skipif(
    not (SCORER.is_file() and GOLD_DIR.is_dir() and IMG_DIR.is_dir()),
    reason="assignment/score.py or data/dev (labels + images) absent (gitignored)",
)

PINS = {
    "qwen35_4b": ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
    "nuextract3": ("numind/NuExtract3", "c99dc8f5641b866aa0192b6ea78f84bf9f3535f1"),
    "qwen3vl_8b": ("Qwen/Qwen3-VL-8B-Instruct", "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"),
    "qwen3vl_4b": ("Qwen/Qwen3-VL-4B-Instruct", "ebb281ec70b05090aa6165b016eac8ec08e71b17"),
}
EXPECTED_CONFIGS = [
    "spike_qwen35_4b_img_only",
    "spike_qwen35_4b_img_ocr",
    "spike_nuextract3_img_only",
    "spike_nuextract3_img_ocr",
    "spike_qwen3vl_8b_img_only",
    "spike_qwen3vl_8b_img_ocr",
    "spike_qwen3vl_4b_img_only",
]


# ------------------------------------------------------------------------------ config loading


@pytest.mark.parametrize("name", EXPECTED_CONFIGS)
def test_shipped_configs_load_with_pinned_revisions(name: str) -> None:
    cfg = spike.load_config(CONFIGS / f"{name}.yaml")
    key = cfg.backend.key
    assert (cfg.backend.repo, cfg.backend.revision) == PINS[key]
    assert cfg.name == name.removeprefix("spike_")
    assert cfg.arm == name.rsplit("_", 2)[-2] + "_" + name.rsplit("_", 1)[-1]
    assert cfg.backend.dtype == "float16"  # T4 has no bf16
    assert cfg.backend.max_pixels == 1280 * 32 * 32
    assert cfg.backend.max_new_tokens == 1536 and cfg.backend.ocr_token_budget == 1200
    assert cfg.backend.seed == 42 and cfg.prompt_version == PROMPT_VERSION
    assert cfg.backend.quant == ("nf4" if key == "qwen3vl_8b" else "none")
    assert cfg.backend.trust_remote_code is False
    assert len(cfg.config_hash) == 16


def test_config_hash_is_stable_and_sensitive(tmp_path: Path) -> None:
    src = CONFIGS / "spike_qwen35_4b_img_only.yaml"
    a, b = spike.load_config(src), spike.load_config(src)
    assert a.config_hash == b.config_hash
    data = yaml.safe_load(src.read_text(encoding="utf-8"))
    data["max_new_tokens"] = 1000
    f = tmp_path / "c.yaml"
    f.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert spike.load_config(f).config_hash != a.config_hash


def _write(tmp_path: Path, **overrides: Any) -> Path:
    data = yaml.safe_load((CONFIGS / "spike_qwen35_4b_img_only.yaml").read_text(encoding="utf-8"))
    for k, v in overrides.items():
        if k.startswith("model."):
            data["model"][k.removeprefix("model.")] = v
        else:
            data[k] = v
    f = tmp_path / "bad.yaml"
    f.write_text(yaml.safe_dump(data), encoding="utf-8")
    return f


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"arm": "both"}, "arm"),
        ({"prompt_version": "v0"}, "PROMPT_VERSION"),
        ({"model.revision": "main"}, "40-hex"),
    ],
)
def test_bad_configs_are_rejected(tmp_path: Path, overrides: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        spike.load_config(_write(tmp_path, **overrides))


def test_missing_key_is_rejected(tmp_path: Path) -> None:
    f = tmp_path / "c.yaml"
    f.write_text("name: x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing keys"):
        spike.load_config(f)


# ------------------------------------------------------------------------------ small helpers


def test_load_doc_ids_file_inline_and_errors(tmp_path: Path) -> None:
    f = tmp_path / "ids.json"
    f.write_text('["a", "b"]', encoding="utf-8")
    assert spike.load_doc_ids(str(f)) == ["a", "b"]
    assert spike.load_doc_ids('["c"]') == ["c"]
    with pytest.raises(ValueError, match="duplicate"):
        spike.load_doc_ids('["a", "a"]')
    with pytest.raises(ValueError, match="list of doc_id"):
        spike.load_doc_ids("[1, 2]")


def test_doc_page_images_sorted_numerically(tmp_path: Path) -> None:
    img = tmp_path / "dev" / "images"
    img.mkdir(parents=True)
    for n in ("dev_0001_p10.png", "dev_0001_p2.jpg", "dev_0001_p1.png", "dev_00011_p1.png"):
        (img / n).write_bytes(b"")
    got = [p.name for p in spike.doc_page_images("dev", "dev_0001", tmp_path)]
    assert got == ["dev_0001_p1.png", "dev_0001_p2.jpg", "dev_0001_p10.png"]
    with pytest.raises(FileNotFoundError):
        spike.doc_page_images("dev", "dev_9999", tmp_path)


def test_atomic_write_leaves_no_temp_file(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "f.json"
    spike.atomic_write(target, "one")
    spike.atomic_write(target, "two")
    assert target.read_text(encoding="utf-8") == "two"
    assert [p.name for p in target.parent.iterdir()] == ["f.json"]


# ------------------------------------------------------------------------------ mock end-to-end


def _pick_docs() -> list[str]:
    """5 deterministic dev docs: first multipage invoice, first waybill, then the first others."""
    meta = json.loads((ROOT / "meta" / "dev.json").read_text(encoding="utf-8"))
    multi = next(m["doc_id"] for m in meta if m["multipage"] and not m["waybill"])
    wb = next(m["doc_id"] for m in meta if m["waybill"])
    rest = [m["doc_id"] for m in meta if m["doc_id"] not in (multi, wb)][:3]
    return sorted({multi, wb, *rest})


@pytest.fixture
def docs_file(tmp_path: Path) -> Path:
    f = tmp_path / "docs.json"
    f.write_text(json.dumps(_pick_docs()), encoding="utf-8")
    return f


def _cli(tmp_path: Path, docs: Path, run_id: str, *extra: str) -> int:
    return cli.main(
        [
            "spike",
            "--config",
            str(CONFIGS / "spike_qwen35_4b_img_only.yaml"),
            "--docs",
            str(docs),
            "--split",
            "dev",
            "--run-id",
            run_id,
            "--backend",
            "mock",
            *extra,
        ]
    )


@needs_data
def test_mock_run_end_to_end_via_cli_and_resume(
    tmp_path: Path, docs_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SHIPDOC_RUNS_DIR", str(tmp_path / "runs"))
    ids = json.loads(docs_file.read_text(encoding="utf-8"))
    assert len(ids) == 5
    assert _cli(tmp_path, docs_file, "r1") == 0
    out = paths.runs_dir() / "r1"
    assert out == tmp_path / "runs" / "r1"
    for name in ("predictions.json", "trace.jsonl", "metrics.json", "progress.json"):
        assert (out / name).is_file(), name
    assert not list(out.glob("*.tmp"))

    preds = json.loads((out / "predictions.json").read_text(encoding="utf-8"))
    assert sorted(preds) == ids
    traces = [json.loads(ln) for ln in (out / "trace.jsonl").read_text().splitlines()]
    assert [t["doc_id"] for t in traces] == ids
    t0 = traces[0]
    for key in ("pages", "merge", "normalize_flags", "prediction", "model", "prompt", "config"):
        assert key in t0
    assert t0["model"] == {"id": "mock", "revision": "mock"}
    assert t0["prompt"]["version"] == PROMPT_VERSION and len(t0["prompt"]["hash"]) == 64
    assert {"raw_text", "parsed", "meta", "json_valid"} <= set(t0["pages"][0])
    assert t0["git_commit"]

    m = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert m["scored"] is True and m["n_docs"] == 5
    assert m["OVERALL"] == pytest.approx(1.0)  # gold replayed through merge+normalize scores 100
    assert m["OVERALL_ci95"]["lo"] <= m["OVERALL"] <= m["OVERALL_ci95"]["hi"]
    assert m["slices"]["all"]["ci95"]["OVERALL"]["point"] == pytest.approx(1.0)
    assert {"all", "invoices", "waybills", "scanned=yes", "scanned=no"} <= set(m["slices"])
    assert m["scanned"] is not None  # the scanned slice is reported explicitly
    assert m["json_validity_rate"] == 1.0 and m["false_fill_rate"] == 0.0
    assert m["s_per_page_mean"] is not None and m["s_per_page_p95"] >= 0
    assert m["peak_vram_max_bytes"] is None  # no GPU in the mock
    progress = json.loads((out / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "complete" and progress["done"] == progress["total"] == 5

    # A second run without --resume must refuse to clobber the trace.
    with pytest.raises(FileExistsError, match="--resume"):
        _cli(tmp_path, docs_file, "r1")
    # --resume with everything done changes nothing.
    before = (out / "trace.jsonl").read_bytes()
    assert _cli(tmp_path, docs_file, "r1", "--resume") == 0
    assert (out / "trace.jsonl").read_bytes() == before


@needs_data
def test_resume_skips_completed_docs_and_only_runs_new_ones(tmp_path: Path) -> None:
    from shipdoc import eval as ev

    cfg = spike.load_config(CONFIGS / "spike_qwen35_4b_img_only.yaml")
    ids = _pick_docs()
    backend = MockBackend(ev.load_gold(GOLD_DIR))
    n_pages = {d: len(spike.doc_page_images("dev", d)) for d in ids}
    spike.run_spike(cfg, ids, "dev", "r", backend, limit=3, runs_root=tmp_path)
    assert backend._calls == sum(n_pages[d] for d in ids[:3])
    calls = backend._calls
    m = spike.run_spike(cfg, ids, "dev", "r", backend, resume=True, runs_root=tmp_path)
    assert backend._calls - calls == sum(n_pages[d] for d in ids[3:])  # only the 2 new docs
    traces = [json.loads(ln) for ln in (tmp_path / "r" / "trace.jsonl").read_text().splitlines()]
    assert [t["doc_id"] for t in traces] == ids and m["n_docs"] == 5


@needs_data
def test_mock_page2_trap_corruption_does_not_change_score(tmp_path: Path) -> None:
    """A legible invoice number on continuation pages must not leak into the merged header."""
    from shipdoc import eval as ev

    cfg = spike.load_config(CONFIGS / "spike_qwen35_4b_img_only.yaml")
    gold = ev.load_gold(GOLD_DIR)
    meta = json.loads((ROOT / "meta" / "dev.json").read_text(encoding="utf-8"))
    multi = [m["doc_id"] for m in meta if m["multipage"] and not m["waybill"]][:3]
    backend = MockBackend(gold, MockCorruption(banner_leak=True, blank_fields=("invoice_number",)))
    m = spike.run_spike(cfg, multi, "dev", "trap", backend, runs_root=tmp_path)
    trace = [json.loads(ln) for ln in (tmp_path / "trap" / "trace.jsonl").read_text().splitlines()]
    assert all(t["prediction"]["header"]["invoice_number"] is None for t in trace)
    assert trace[0]["merge"]["ignored_elsewhere"]["invoice_number"][0] == 2
    # gold has a non-null number for these docs, so blanking page 1 costs exactly that field
    assert m["slices"]["all"]["header_field_accuracy"] < 1.0
    assert m["false_fill_rate"] == 0.0


@needs_data
def test_invalid_json_pages_are_counted_not_retried(tmp_path: Path) -> None:
    from shipdoc import eval as ev

    cfg = spike.load_config(CONFIGS / "spike_qwen35_4b_img_only.yaml")
    backend = MockBackend(ev.load_gold(GOLD_DIR), MockCorruption(invalid_json_every=1))
    ids = _pick_docs()[:2]
    m = spike.run_spike(cfg, ids, "dev", "bad", backend, runs_root=tmp_path)
    assert m["json_validity_rate"] == 0.0 and m["OVERALL"] < 0.5
    trace = [json.loads(ln) for ln in (tmp_path / "bad" / "trace.jsonl").read_text().splitlines()]
    page = trace[0]["pages"][0]
    assert page["parsed"] is None and page["raw_text"] and page["json_valid"] is False
    assert backend._calls == sum(len(t["pages"]) for t in trace)  # exactly one call per page
