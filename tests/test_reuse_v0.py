"""shipdoc.reuse: the refusal conditions for reusing the v0 model outputs, the decode-path
fingerprint, the CPU assembly over the v0 traces (R1-R3 default on) and the v1-only checks.

Synthetic corpus + mock backend (tests/test_predict.py patterns); git is never called against a
made-up SHA: the blob comparison takes injected functions. Nothing real is read or written.
"""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

# ruff: noqa: F811  (the `world` fixture is imported, then requested by name, as pytest requires)
from test_predict import (  # noqa: F401 - `_clean_sha` is an autouse fixture, imported on purpose
    N_PAGES,
    RUN,
    SCHEMA,
    SHA,
    TEST_IDS,
    World,
    _clean_sha,
    _edit_trace,
    assemble,
    needs_schema,
    run_all,
    world,
)

from shipdoc import ocr, ocr_stage, paths, predict, reuse, spike

ROOT = Path(__file__).resolve().parents[1]


V0 = "4" * 40
HEAD = "5" * 40
REAL_V0 = "42b812b5b09d6e4bff0df12564017f71ffad5fc9"
SHAPES = ROOT / "meta" / "slot_shapes.json"


# ---------------------------------------------------------------------- pure refusal function


def good() -> dict[str, Any]:
    """Inputs under which `decide_reuse` allows reuse (tiny, hand-built)."""
    pages = {"test_0000": 1, "test_0001": 2}

    def trace(doc: str, n: int) -> dict[str, Any]:
        return {
            "doc_id": doc, "arm": "img_only", "output_format": "json", "git_commit": V0,
            "model": {"revision": "REV"}, "prompt": {"hash": "PH", "version": "v2"},
            "config": {"hash": "CH"},
            "pages": [{"raw_text": "t", "field_logprobs": [1.0]} for _ in range(n)],
        }  # fmt: skip

    return {
        "v0_manifest": {
            "code_sha": V0,
            "config": {"hash": "CH"},
            "prompt": {"hash": "PH"},
            "model": {"revision": "REV"},
            "seed": 42,
            "shard": "0/1",
            "batch_size": 8,
            "determinism": {"ok": True, "n_docs": 5},
        },  # fmt: skip
        "v0_report": {"ok": True, "checks": {"a": {"ok": True}, "b": {"ok": True}}},
        "v0_traces": [trace(d, n) for d, n in pages.items()],
        "expected_pages": pages,
        "prod": {
            "config_hash": "CH",
            "prompt_hash": "PH",
            "model_revision": "REV",
            "seed": 42,
            "output_format": "json",
            "arm": "img_only",
            "shard": "0/1",
            "batch_size": 8,
            "header_hint": False,
        },  # fmt: skip
        "decode": {
            "ok": True,
            "differing": [],
            "unresolved": [],
            "v0_sha": V0,
            "head_sha": HEAD,
            "files": 17,
            "spike": {"ok": True, "state": "only the allowlisted opt-in additions"},
        },  # fmt: skip
        "head_sha": HEAD,
    }


def decide(inp: dict[str, Any]) -> reuse.ReuseDecision:
    return reuse.decide_reuse(**inp)


def test_good_inputs_allow_reuse() -> None:
    d = decide(good())
    assert d.ok and d.reasons == []
    assert d.facts["v0_code_sha"] == V0 and d.facts["batch_size"] == {"v0": 8, "v1": 8}


def _set(path: str, value: Any) -> Any:
    """Mutation: set `inp[...]` at a dotted path (list index = integer segment)."""

    def mut(inp: dict[str, Any]) -> None:
        cur: Any = inp
        parts = path.split(".")
        for p in parts[:-1]:
            cur = cur[int(p)] if isinstance(cur, list) else cur[p]
        last = parts[-1]
        if isinstance(cur, list):
            cur[int(last)] = value
        else:
            cur[last] = value

    return mut


REFUSALS: list[tuple[str, Any, str]] = [
    ("manifest missing", _set("v0_manifest", None), "v0_manifest_missing"),
    ("report missing", _set("v0_report", None), "v0_report_missing"),
    ("v0 rejected", _set("v0_report.ok", False), "v0_not_validated"),
    ("v0 failed check", _set("v0_report.checks.b.ok", False), "v0_not_validated"),
    ("v0 code sha dirty", _set("v0_manifest.code_sha", V0 + "+dirty"), "v0_code_sha"),
    ("v0 code sha short", _set("v0_manifest.code_sha", "42b812b"), "v0_code_sha"),
    ("head dirty", _set("head_sha", HEAD + "+dirty"), "head_dirty"),
    ("v0 determinism failed", _set("v0_manifest.determinism.ok", False), "v0_determinism"),
    ("config hash", _set("v0_manifest.config.hash", "OTHER"), "config_hash"),
    ("trace config hash", _set("v0_traces.1.config.hash", "OTHER"), "config_hash_traces"),
    ("prompt hash (manifest)", _set("v0_manifest.prompt.hash", "OTHER"), "prompt_hash"),
    ("prompt hash (trace)", _set("v0_traces.0.prompt.hash", "OTHER"), "prompt_hash"),
    ("model revision", _set("v0_manifest.model.revision", "OTHER"), "model_revision"),
    (
        "model revision (trace)",
        _set("v0_traces.0.model.revision", "OTHER"),
        "model_revision_traces",
    ),
    ("seed", _set("v0_manifest.seed", 7), "seed"),
    ("shard", _set("v0_manifest.shard", "merged"), "shard"),
    ("batch size differs", _set("prod.batch_size", 4), "batch_size"),
    ("batch size unset", _set("prod.batch_size", None), "batch_size_unset"),
    ("header hint on", _set("prod.header_hint", True), "header_hint"),
    ("output format", _set("v0_traces.0.output_format", "compact"), "output_format"),
    ("arm", _set("v0_traces.1.arm", "img_ocr"), "arm"),
    ("trace missing", _set("v0_traces", None), "trace_missing"),
    ("trace doc missing", lambda i: i["v0_traces"].pop(), "trace_ids"),
    ("trace doc duplicate", lambda i: i["v0_traces"].append(i["v0_traces"][0]), "trace_ids"),
    ("trace extra doc", lambda i: i["expected_pages"].pop("test_0001"), "trace_ids"),
    ("trace pages incomplete", lambda i: i["v0_traces"][1]["pages"].pop(), "trace_pages"),
    (
        "page without raw text",
        lambda i: i["v0_traces"][0]["pages"][0].pop("raw_text"),
        "trace_pages",
    ),
    ("logprobs missing", lambda i: i["v0_traces"][0]["pages"][0].pop("field_logprobs"), "logprobs"),
    ("trace written by other code", _set("v0_traces.0.git_commit", "9" * 40), "trace_code_sha"),
    ("decode blob differs", _set("decode.differing", ["src/shipdoc/extract.py"]), "decode_path"),
    ("decode blob unresolved", _set("decode.unresolved", ["uv.lock"]), "decode_path"),
    ("spike diff", _set("decode.spike", {"ok": False, "state": "differs"}), "spike_diff"),
    ("decode not computed", _set("decode", None), "decode_path"),
]


@pytest.mark.parametrize(("name", "mutate", "code"), REFUSALS, ids=[r[0] for r in REFUSALS])
def test_each_refusal_reason_refuses_reuse(name: str, mutate: Any, code: str) -> None:
    inp = copy.deepcopy(good())
    mutate(inp)
    d = decide(inp)
    assert not d.ok, name
    assert any(r.startswith(code + ":") for r in d.reasons), (name, d.reasons)


def test_every_reason_is_listed_not_only_the_first() -> None:
    inp = copy.deepcopy(good())
    inp["prod"]["batch_size"] = 4
    inp["v0_manifest"]["seed"] = 7
    inp["decode"]["differing"] = ["src/shipdoc/extract.py"]
    codes = {r.split(":")[0] for r in decide(inp).reasons}
    assert {"batch_size", "seed", "decode_path"} <= codes


def test_reasons_carry_no_document_values() -> None:
    inp = copy.deepcopy(good())
    inp["v0_traces"][0]["pages"][0]["raw_text"] = "SECRET VALUE"
    inp["v0_traces"][0]["pages"].append({"raw_text": "SECRET VALUE"})  # wrong page count
    d = decide(inp)
    assert not d.ok and "SECRET" not in json.dumps(d.facts) + " ".join(d.reasons)


# ---------------------------------------------------------------------- decode-path fingerprint


def blobs(changed: set[str] | None = None, missing: set[str] | None = None) -> Any:
    changed, missing = changed or set(), missing or set()

    def fn(sha: str, path: str) -> str | None:
        if path in missing and sha == HEAD:
            return None
        return f"blob-{path}-{'new' if path in changed and sha == HEAD else 'old'}"

    return fn


def test_decode_path_identical_blobs_pass_and_every_file_is_compared() -> None:
    calls: list[tuple[str, str]] = []

    def fn(sha: str, path: str) -> str | None:
        calls.append((sha, path))
        return "same"

    r = reuse.decode_path_report(V0, HEAD, fn, lambda a, b, p: "")
    assert r["ok"] and r["files"] == len(reuse.DECODE_PATH_FILES) and not r["differing"]
    assert len(calls) == 2 * len(reuse.DECODE_PATH_FILES)
    assert r["spike"]["state"] == "spike.py identical"


def test_decode_path_blob_difference_or_unresolved_blob_refuses() -> None:
    r = reuse.decode_path_report(V0, HEAD, blobs({"src/shipdoc/extract.py"}), lambda *a: "")
    assert not r["ok"] and r["differing"] == ["src/shipdoc/extract.py"]
    r = reuse.decode_path_report(V0, HEAD, blobs(missing={"uv.lock"}), lambda *a: "")
    assert not r["ok"] and r["unresolved"] == ["uv.lock"]  # unknown is never "equal"
    r = reuse.decode_path_report(V0, HEAD, lambda s, p: None, lambda *a: "")
    assert not r["ok"] and len(r["unresolved"]) == len(reuse.DECODE_PATH_FILES)


def test_decode_path_list_covers_the_model_output_path() -> None:
    must = {"extract.py", "prompts.py", "logprobs.py", "batching.py", "shard.py", "coerce.py"}
    names = {Path(f).name for f in reuse.DECODE_PATH_FILES}
    assert must <= names and "spike.py" not in names  # spike.py has its own allowlisted check
    assert "uv.lock" in names and "field_provenance.json" in names
    assert all((ROOT / f).is_file() for f in reuse.DECODE_PATH_FILES)


DIFF = "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-old import\n+new import\n@@ -9,0 +10,2 @@\n+opt in\n+more\n"


def test_split_diff_lines_ignores_file_headers() -> None:
    assert reuse.split_diff_lines(DIFF) == (["old import"], ["new import", "opt in", "more"])


def test_spike_diff_check_allowlist_ack_and_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    assert reuse.spike_diff_check("")["ok"]  # identical file
    bad = reuse.spike_diff_check(DIFF)
    assert not bad["ok"] and not bad["acked"] and bad["removed"] == 1 and bad["added"] == 3
    acked = reuse.spike_diff_check(DIFF, ack=True)
    assert acked["ok"] and acked["acked"] and "REUSE_ACK_SPIKE_DIFF" in acked["state"]
    assert not reuse.spike_diff_check(None, ack=True)["ok"]  # a git failure is never acked
    removed, added = reuse.split_diff_lines(DIFF)
    monkeypatch.setattr(reuse, "SPIKE_ALLOWED_REMOVED_SHA256", reuse._lines_sha(removed))
    monkeypatch.setattr(reuse, "SPIKE_ALLOWED_ADDED_SHA256", reuse._lines_sha(added))
    ok = reuse.spike_diff_check(DIFF)
    assert ok["ok"] and not ok["acked"]
    # one more added line (any other edit to spike.py) breaks the allowlist again
    assert not reuse.spike_diff_check(DIFF + "+sneaky\n")["ok"]


def _real_git_has_v0() -> bool:
    try:
        res = subprocess.run(["git", "cat-file", "-e", f"{REAL_V0}^{{commit}}"], cwd=ROOT,
                             capture_output=True, check=False)  # fmt: skip
    except OSError:
        return False
    return res.returncode == 0


@pytest.mark.skipif(not _real_git_has_v0(), reason="the v0 commit is not in this clone")
def test_pinned_spike_allowlist_matches_the_real_v0_to_head_diff() -> None:
    """Guards the constants: they describe the diff 42b812b..b20b391 (run on a clone with it)."""
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()  # fmt: skip
    anc = subprocess.run(["git", "merge-base", "--is-ancestor", "b20b391", head], cwd=ROOT,
                         capture_output=True, check=False)  # fmt: skip
    if anc.returncode != 0:
        pytest.skip("HEAD does not contain b20b391")
    rep = reuse.decode_path_report(REAL_V0, head)
    assert rep["spike"]["ok"], rep["spike"]
    assert not rep["differing"] and not rep["unresolved"], rep


# ---------------------------------------------------------------------- end to end (mock)


def write_ocr_cache(root: Path, images: Path, drop: set[str] | None = None) -> None:
    """A fake OCR cache for every test page; the first waybill carries a pattern R2 can fill."""
    for stem in ocr_stage.expected_stems(images):
        if stem in (drop or set()):
            continue
        text = "MAWB 176-12345678" if stem == "test_0002_p1" else f"page {stem}"
        page = ocr.PageOcr(
            f"{stem}.png", 10, 5, "paddleocr", "line", 0,
            [ocr.OcrItem(text, [[0, 0], [10, 0], [10, 5], [0, 5]], 0.9)], seconds=1.5,
        )  # fmt: skip
        ocr.write_json_atomic(ocr.cache_path(root, "paddleocr", "test", stem), page.to_dict())


def fake_v0(w: World, tmp: Path) -> Path:
    """A v0 submission folder from a mock run, dressed up as a real GPU run's folder."""
    run_all(w, batch=1)
    _edit_trace(w)  # give R1, R2 and R3 something to do
    v0 = tmp / "v0_aaaaaaa"
    assert assemble(w, v0)["ok"]
    man = json.loads((v0 / "manifest.json").read_text())
    man["model"] = {"id": "Qwen/Qwen3.5-4B", "revision": w.cfg.backend.revision}
    man["dependencies"] = {"python": "3.11.0", "gpu": "T4",
                           "packages": {k: "1.0" for k in predict.STACK_PACKAGES}}  # fmt: skip
    (v0 / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    traces = [json.loads(ln) for ln in (v0 / "trace.jsonl").read_text().splitlines() if ln]
    for t in traces:
        t["model"]["revision"] = w.cfg.backend.revision
    (v0 / "trace.jsonl").write_text("".join(json.dumps(t) + "\n" for t in traces))
    return v0


def decision(w: World, v0: Path, batch: int = 1, **kw: Any) -> reuse.ReuseDecision:
    return reuse.evaluate_reuse(
        w.cfg, v0, w.data, batch, "0/1", blob_fn=blobs(), diff_fn=lambda *a: "", head_sha=SHA, **kw
    )


def reuse_assemble(w: World, v0: Path, out: Path, ocr_root: Path, **kw: Any) -> dict[str, Any]:
    kw.setdefault("decision", decision(w, v0))
    return reuse.assemble_reuse(
        w.cfg, v0, out, w.data, SCHEMA, ocr_root, 1, "0/1", None, len(TEST_IDS), N_PAGES, SHA,
        shapes_file=SHAPES, **kw,
    )  # fmt: skip


@needs_schema
def test_reuse_assembly_equals_the_full_path_on_the_same_traces(
    world: World, tmp_path: Path
) -> None:
    v0 = fake_v0(world, tmp_path)
    assert decision(world, v0).ok
    ocr_root = tmp_path / "ocr"
    write_ocr_cache(ocr_root, world.data / "test" / "images")
    full = tmp_path / "full"
    assert assemble(world, full, ocr_cache=ocr_root, shapes_file=SHAPES)["ok"]
    out = tmp_path / "v1_abcdef0"
    rep = reuse_assemble(world, v0, out, ocr_root)
    assert rep["ok"] and rep["mode"] == "reuse", {
        k: c for k, c in rep["checks"].items() if not c["ok"]
    }
    for name in ("test_predictions.json", "rules.jsonl"):
        assert (out / name).read_bytes() == (full / name).read_bytes(), name
    assert (out / "trace.jsonl").read_bytes() == (v0 / "trace.jsonl").read_bytes()  # a byte copy
    preds = json.loads((out / "test_predictions.json").read_text())
    assert preds["test_0002"]["header"]["carrier"] == "Acme Air Sentinel"  # R1 fired
    assert preds["test_0002"]["header"]["mawb"] == "176-12345678"  # R2 fired from the OCR cache
    man = json.loads((out / "manifest.json").read_text())
    assert man["mode"] == "reuse" and man["reuse"]["v0_code_sha"] == SHA
    assert man["determinism_v0"]["source_sha256"] == predict.sha256_file(v0 / "manifest.json")
    assert man["post_rules"]["slot_shapes_sha256"] == predict.sha256_file(SHAPES)
    assert man["post_rules"]["switches"] == {"r1": True, "r2": True, "r3": True}
    assert man["timings"]["from_v0_run"] is True and man["batch_size"] == 1
    for name in ("manifest.json", "validation_report.json", "rules.jsonl"):
        assert "Acme Air Sentinel" not in (out / name).read_text()  # no values


@needs_schema
def test_reuse_assembly_refuses_without_writing_when_reuse_is_not_allowed(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    v0 = fake_v0(world, tmp_path)
    ocr_root = tmp_path / "ocr"
    write_ocr_cache(ocr_root, world.data / "test" / "images")
    out = tmp_path / "v1_abcdef0"
    dec = reuse.evaluate_reuse(world.cfg, v0, world.data, 4, "0/1", blob_fn=blobs(),
                               diff_fn=lambda *a: "", head_sha=SHA)  # fmt: skip
    assert not dec.ok and any(r.startswith("batch_size:") for r in dec.reasons)
    with pytest.raises(reuse.ReuseError, match="REUSE REFUSED"):
        reuse_assemble(world, v0, out, ocr_root, decision=dec)
    assert not out.exists()
    # the CLI path recomputes the decision from disk; the git fingerprint is faked (no real git
    # call on a made-up SHA) and the batch size 4 differs from v0's 1
    fake = {"ok": True, "differing": [], "unresolved": [], "v0_sha": SHA, "head_sha": SHA,
            "files": 1, "spike": {"ok": True, "state": "x"}}  # fmt: skip
    monkeypatch.setattr(reuse, "decode_path_report", lambda *a, **k: fake)
    rc = reuse.main(["assemble", "--v0-dir", str(v0), "--batch-size", "4", "--data-root",
                     str(world.data), "--out-dir", str(out), "--schema", str(SCHEMA),
                     "--ocr-cache", str(ocr_root)])  # fmt: skip
    assert rc == 3 and not out.exists()


def test_cli_check_refuses_with_exit_3_for_a_missing_v0_folder(
    world: World, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = reuse.main(["check", "--v0-dir", str(tmp_path / "nope"), "--batch-size", "8",
                     "--data-root", str(world.data),
                     "--out", str(tmp_path / "d.json")])  # fmt: skip
    assert rc == 3
    assert "REFUSED" in capsys.readouterr().out
    saved = json.loads((tmp_path / "d.json").read_text())
    assert saved["ok"] is False and any(
        r.startswith("v0_manifest_missing") for r in saved["reasons"]
    )


def finalized(
    world: World, tmp_path: Path, name: str, *, drop_ocr: set[str] | None = None,
    recheck_ok: bool = True, tamper_rerun: bool = False, timing: bool = True,
) -> tuple[dict[str, Any], Path]:  # fmt: skip
    """Assemble (reuse) twice + OCR timing + finalize; returns (report, submission folder)."""
    v0 = fake_v0(world, tmp_path / name)
    ocr_root = tmp_path / name / "ocr"
    images = world.data / "test" / "images"
    write_ocr_cache(ocr_root, images, drop_ocr)
    out, again = tmp_path / name / "v1_abcdef0", tmp_path / name / "rerun"
    reuse_assemble(world, v0, out, ocr_root)
    reuse_assemble(world, v0, again, ocr_root)
    if timing:
        stems = ocr_stage.expected_stems(images)
        full = tmp_path / name / "ocr_full"
        write_ocr_cache(full, images)  # a complete cache just to compute the timing file
        t = ocr_stage.timing_summary(full, stems, "gpu", 12.0, len(stems))
        predict._write_json(out / "ocr_timing.json", t)
    if tamper_rerun:
        (again / "rules.jsonl").write_text("{}\n")
    recheck = tmp_path / name / "recheck.json"
    recheck.write_text(json.dumps({"ok": recheck_ok, "n_text_identical": 5 if recheck_ok else 4,
                                   "n_pages": 5, "n_content_identical": 5}))  # fmt: skip
    rep = reuse.finalize(out, SHAPES, "reuse", N_PAGES, again, recheck)
    return rep, out


@needs_schema
def test_finalize_accepts_a_complete_reuse_submission(world: World, tmp_path: Path) -> None:
    rep, out = finalized(world, tmp_path, "ok")
    failed = {k: c["detail"] for k, c in rep["checks"].items() if not c["ok"]}
    assert rep["ok"] and not failed
    assert (out / "test_predictions.json").is_file()
    assert not (out / "test_predictions.REJECTED.json").exists()
    man = json.loads((out / "manifest.json").read_text())
    assert man["submission"] == "v1_abcdef0" and man["files"] == list(reuse.OUT_FILES)
    assert man["ocr"]["timing"]["pages"] == N_PAGES and man["ocr"]["recheck"]["ok"]
    for k in ("v1_rule_switches", "v1_no_rule_skipped", "v1_shapes_sha256", "ocr_cache_complete",
              "ocr_determinism", "assemble_twice"):  # fmt: skip
        assert rep["checks"][k]["ok"], k
    assert json.loads((out / "validation_report.json").read_text())["ok"] is True


@needs_schema
@pytest.mark.parametrize(
    ("name", "kw", "check"),
    [
        ("r2skip", {"drop_ocr": {"test_0008_p1"}}, "v1_no_rule_skipped"),  # a waybill, no OCR
        ("recheck", {"recheck_ok": False}, "ocr_determinism"),
        ("twice", {"tamper_rerun": True}, "assemble_twice"),
        ("notiming", {"timing": False}, "ocr_cache_complete"),
    ],
)
def test_finalize_rejects_and_withholds_the_submittable_name(
    world: World, tmp_path: Path, name: str, kw: dict[str, Any], check: str
) -> None:
    rep, out = finalized(world, tmp_path, name, **kw)
    assert not rep["ok"] and not rep["checks"][check]["ok"]
    assert not (out / "test_predictions.json").exists()
    assert (out / "test_predictions.REJECTED.json").is_file()
    assert json.loads((out / "validation_report.json").read_text())["ok"] is False


@needs_schema
def test_finalize_blocks_a_wrong_shapes_hash_or_switched_off_rule(
    world: World, tmp_path: Path
) -> None:
    v0 = fake_v0(world, tmp_path)
    ocr_root = tmp_path / "ocr"
    images = world.data / "test" / "images"
    write_ocr_cache(ocr_root, images)
    out = tmp_path / "v1_abcdef0"
    reuse_assemble(world, v0, out, ocr_root)
    other = tmp_path / "other_shapes.json"
    shutil.copyfile(SHAPES, other)
    other.write_text(other.read_text() + " ")  # a different file: the sha256 differs
    stems = ocr_stage.expected_stems(images)
    predict._write_json(out / "ocr_timing.json",
                        ocr_stage.timing_summary(ocr_root, stems, "cpu"))  # fmt: skip
    rep = reuse.finalize(out, other, "full", N_PAGES, None, None)
    assert not rep["checks"]["v1_shapes_sha256"]["ok"] and not rep["ok"]
    # a rule switched OFF at assembly is caught by the switches check
    off = tmp_path / "v1_off"
    reuse_assemble(world, v0, off, ocr_root, rule_cfg=reuse.RuleConfig(r2=False))
    predict._write_json(off / "ocr_timing.json", ocr_stage.timing_summary(ocr_root, stems, "cpu"))
    rep = reuse.finalize(off, SHAPES, "full", N_PAGES, None, None)
    assert not rep["checks"]["v1_rule_switches"]["ok"]
    assert not rep["checks"]["v1_no_rule_skipped"]["ok"]  # R2/disabled


def test_compare_outputs_reports_identical_and_differing_files(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        d.mkdir()
        (d / "test_predictions.json").write_text("{}")
        (d / "trace.jsonl").write_text("x\n")
        (d / "rules.jsonl").write_text("y\n")
    assert reuse.compare_outputs(a, b)["ok"]
    (b / "rules.jsonl").write_text("z\n")
    c = reuse.compare_outputs(a, b)
    assert not c["ok"] and c["different_or_missing"] == ["rules.jsonl"]
    (b / "test_predictions.json").replace(b / "test_predictions.REJECTED.json")
    assert not reuse.compare_outputs(a, b)["ok"]  # different prediction names are never "same"


@needs_schema  # the fake v0 is assembled against assignment/schema.json: skip when it is absent
def test_submission_folder_inside_the_repo_is_refused(world: World, tmp_path: Path) -> None:
    v0 = fake_v0(world, tmp_path)
    inside = paths.REPO_ROOT / "reuse_test_must_not_be_written"
    with pytest.raises(predict.PredictError, match="inside the repository"):
        reuse_assemble(world, v0, inside, tmp_path / "ocr")
    assert not inside.exists()
    assert spike.git_commit() == SHA  # the autouse fixture is active
