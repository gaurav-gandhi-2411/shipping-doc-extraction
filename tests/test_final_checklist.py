"""Unit tests for scripts/final_checklist.py on synthetic filesystem / git fixtures."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "final_checklist", ROOT / "scripts" / "final_checklist.py"
)
fc = importlib.util.module_from_spec(spec)
sys.modules["final_checklist"] = fc
spec.loader.exec_module(fc)


def write(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def row(check: str, **args: object) -> object:
    return fc.Row("T1", "brief p1", "t", "deliverable", "a", check, "rule", args=args)


def make_pdf(pages: int) -> bytes:
    """Minimal PDF with ``pages`` page objects (what ``count_pdf_pages`` counts)."""
    objs = [b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"]
    objs.append(b"2 0 obj\n<< /Type /Pages /Count %d >>\nendobj\n" % pages)
    for i in range(pages):
        objs.append(b"%d 0 obj\n<< /Type /Page /Parent 2 0 R >>\nendobj\n" % (3 + i))
    return b"%PDF-1.4\n" + b"".join(objs) + b"%%EOF\n"


def git(repo: Path, *args: str) -> str:
    cmd = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args]
    return subprocess.run(cmd, cwd=repo, capture_output=True, text=True, check=True).stdout  # noqa: S603


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    git(r, "init", "-q")
    return r


# ---- structure ----
def test_checklist_structure_is_valid() -> None:
    assert fc.validate_rows(fc.ROWS) == []
    ids = [r.id for r in fc.ROWS]
    assert len(ids) == len(set(ids))
    assert all(r.source and r.status_rule and r.check in fc.CHECKS for r in fc.ROWS)
    assert {r.kind for r in fc.ROWS} == set(fc.KINDS)


def test_validate_rows_flags_duplicates_and_blanks() -> None:
    a = fc.Row("X1", "brief p1", "n", "deliverable", "a", "fixed", "rule")
    b = fc.Row("X1", "", "n", "bogus", "a", "nope", "")
    problems = fc.validate_rows([a, b])
    assert any("duplicate id" in p for p in problems)
    assert any("empty source" in p for p in problems)
    assert any("bad kind" in p for p in problems)
    assert any("unknown check" in p for p in problems)


# ---- each status path ----
def test_exists_done_and_pending(tmp_path: Path) -> None:
    write(tmp_path / "a.txt")
    ctx = fc.Ctx(root=tmp_path, ext=tmp_path / "ext")
    assert fc.evaluate(row("exists", paths=["a.txt"]), ctx).status == "DONE"
    res = fc.evaluate(row("exists", paths=["a.txt", "b.txt"], waits="the b run"), ctx)
    assert res.status == "PENDING"
    assert "b.txt" in res.note
    write(tmp_path / "ext" / "runs" / "r1" / "m.json")
    assert fc.evaluate(row("exists", paths=["ext:runs/*/m.json"]), ctx).status == "DONE"


def test_text_done_partial_pending(tmp_path: Path) -> None:
    write(tmp_path / "R.md", "## Section\nPLACEHOLDER <X>\n")
    ctx = fc.Ctx(root=tmp_path, ext=tmp_path)
    assert fc.evaluate(row("text", path="R.md", pattern="^## Section"), ctx).status == "DONE"
    partial = row("text", path="R.md", pattern="^## Section", partial_if="<X>")
    assert fc.evaluate(partial, ctx).status == "PARTIAL"
    assert fc.evaluate(row("text", path="R.md", pattern="nothere"), ctx).status == "PENDING"
    assert fc.evaluate(row("text", path="none.md", pattern="x"), ctx).status == "PENDING"


@pytest.mark.parametrize("status", ["BLOCKED-ON-GG", "N/A", "NOT VERIFIABLE LOCALLY", "PENDING"])
def test_fixed_statuses(tmp_path: Path, status: str) -> None:
    ctx = fc.Ctx(root=tmp_path, ext=tmp_path)
    assert fc.evaluate(row("fixed", status=status, note="n"), ctx).status == status


def test_unknown_check_and_bad_fixed_status_fail_closed(tmp_path: Path) -> None:
    ctx = fc.Ctx(root=tmp_path, ext=tmp_path)
    assert fc.evaluate(row("nope"), ctx).status == "NOT VERIFIABLE LOCALLY"
    assert fc.evaluate(row("fixed", status="MAYBE", note=""), ctx).status == (
        "NOT VERIFIABLE LOCALLY"
    )


def test_report_section_states(tmp_path: Path) -> None:
    md = "# T\n## 1. Approach\nall good\n## 2. Results by slice\nv [[PENDING: a]] [[PENDING: b]]\n"
    write(tmp_path / "reports" / "final" / "report.md", md)
    ctx = fc.Ctx(root=tmp_path, ext=tmp_path)
    assert fc.evaluate(row("report_section", heading="approach"), ctx).status == "DONE"
    res = fc.evaluate(row("report_section", heading="results by slice"), ctx)
    assert (res.status, "2 value" in res.note) == ("PARTIAL", True)
    assert fc.evaluate(row("report_section", heading="links"), ctx).status == "PENDING"


# ---- report PDF page count ----
def test_pdf_pages_helper_counts_generated_pdfs(tmp_path: Path) -> None:
    two = tmp_path / "two.pdf"
    two.write_bytes(make_pdf(2))
    three = tmp_path / "three.pdf"
    three.write_bytes(make_pdf(3))
    assert fc.pdf_pages(two) == 2
    assert fc.pdf_pages(three) == 3


def test_report_pdf_two_vs_three_pages(tmp_path: Path) -> None:
    ctx = fc.Ctx(root=tmp_path, ext=tmp_path)
    cands = ["rep/report.pdf"]
    assert fc.evaluate(row("report_pdf", candidates=cands), ctx).status == "PENDING"
    pdf = tmp_path / "rep" / "report.pdf"
    pdf.parent.mkdir()
    pdf.write_bytes(make_pdf(3))
    over = fc.evaluate(row("report_pdf", candidates=cands), ctx)
    assert over.status == "PARTIAL"
    assert "OVER" in over.note
    pdf.write_bytes(make_pdf(2))
    no_md = fc.evaluate(row("report_pdf", candidates=cands), ctx)
    assert no_md.status == "PARTIAL"
    assert "no sibling markdown" in no_md.note
    write(pdf.with_suffix(".md"), "text [[PENDING: x]]")
    assert "1 pending" in fc.evaluate(row("report_pdf", candidates=cands), ctx).note
    write(pdf.with_suffix(".md"), "final text")
    assert fc.evaluate(row("report_pdf", candidates=cands), ctx).status == "DONE"


# ---- submission validation ----
def test_submission_v0_partial_v1_done_and_invalid(tmp_path: Path) -> None:
    root, ext = tmp_path / "root", tmp_path / "ext"
    schema = {"type": "object", "additionalProperties": {"type": "object"}}
    write(root / "assignment" / "schema.json", json.dumps(schema))
    write(root / "assignment" / "sample_submission.json", json.dumps({"d1": {}, "d2": {}}))
    ctx = fc.Ctx(root=root, ext=ext)
    assert fc.evaluate(row("submission"), ctx).status == "PENDING"
    write(ext / "submissions" / "v0_abc" / "test_predictions.json", '{"d1": {}, "d2": {}}')
    v0 = fc.evaluate(row("submission"), ctx)
    assert (v0.status, "ids 2/2" in v0.note) == ("PARTIAL", True)
    write(ext / "submissions" / "v1_def" / "test_predictions.json", '{"d1": {}, "d2": {}}')
    assert fc.evaluate(row("submission"), ctx).status == "DONE"
    write(ext / "submissions" / "v1_def" / "test_predictions.json", '{"d1": {}, "d9": 1}')
    bad = fc.evaluate(row("submission"), ctx)
    assert (bad.status, "FAILS" in bad.note) == ("PARTIAL", True)


# ---- git leak counts ----
def test_leak_counts_on_temp_git_repo(repo: Path) -> None:
    ctx = fc.Ctx(root=repo)
    clean = ["src/a.py", "scripts/data_utils.py", "docs/notes.md", "tests/test_x.py"]
    for p in clean:
        write(repo / p)
    git(repo, "add", *clean)
    assert fc.evaluate(row("leaks"), ctx).status == "DONE"
    assert set(fc.count_tracked_leaks(fc.tracked_files(ctx)).values()) == {0}
    dirty = ["assignment/score.py", "data/x.json", "cache/o.json", "runs/r.json"]
    dirty += ["submissions/v0/p.json", "docs/fig.PNG", "x/y.zip", "docs/b.pdf", "docs/c.jpeg"]
    for p in dirty:
        write(repo / p)
    git(repo, "add", "-f", *dirty)
    counts = fc.count_tracked_leaks(fc.tracked_files(ctx))
    assert counts == {
        "assignment": 1,
        "data": 1,
        "cache": 1,
        "runs": 1,
        "submissions": 1,
        "zip": 1,
        "image": 2,
        "pdf": 1,
    }
    assert fc.evaluate(row("leaks"), ctx).status == "PARTIAL"


def test_leaks_outside_a_git_repo_is_not_a_pass(tmp_path: Path) -> None:
    assert fc.evaluate(row("leaks"), fc.Ctx(root=tmp_path)).status == "NOT VERIFIABLE LOCALLY"


# ---- publish evidence, suite log, remotes ----
def test_publish_evidence_states(tmp_path: Path) -> None:
    ev = tmp_path / "tmp" / "publish_scrub.json"
    args = {"evidence": "ext:tmp/publish_scrub.json"}
    ctx = fc.Ctx(root=tmp_path, ext=tmp_path, head_ts=time.time() - 3600)
    assert fc.evaluate(row("publish", **args), ctx).status == "NOT VERIFIABLE LOCALLY"
    clean = {"offending_paths": [], "tree_scan": {}, "history_scan": {}, "history_scan_exit": 0}
    write(ev, json.dumps(clean))
    assert fc.evaluate(row("publish", **args), ctx).status == "DONE"
    old = fc.Ctx(root=tmp_path, ext=tmp_path, head_ts=time.time() + 3600)
    assert fc.evaluate(row("publish", **args), old).status == "PARTIAL"
    write(ev, json.dumps(clean | {"offending_paths": ["data/x"]}))
    assert "NOT clean" in fc.evaluate(row("publish", **args), ctx).note
    write(ev, "{not json")
    assert fc.evaluate(row("publish", **args), ctx).status == "NOT VERIFIABLE LOCALLY"


def test_suite_log_states(tmp_path: Path) -> None:
    log = tmp_path / "tmp" / "s.log"
    args = {"log": "ext:tmp/s.log"}
    ctx = fc.Ctx(root=tmp_path, ext=tmp_path, head_ts=time.time() - 3600)
    assert fc.evaluate(row("suite", **args), ctx).status == "NOT VERIFIABLE LOCALLY"
    write(log, "...\n12 passed, 1 skipped in 3.0s\nexit=0\n")
    assert fc.evaluate(row("suite", **args), ctx).status == "DONE"
    write(log, "...\n1 failed, 12 passed in 3.0s\nexit=1\n")
    assert "NOT green" in fc.evaluate(row("suite", **args), ctx).note
    write(log, "no summary here\n")
    assert fc.evaluate(row("suite", **args), ctx).status == "NOT VERIFIABLE LOCALLY"
    write(log, "12 passed in 3.0s\nexit=0\n")
    future = fc.Ctx(root=tmp_path, ext=tmp_path, head_ts=time.time() + 3600)
    assert "predates HEAD" in fc.evaluate(row("suite", **args), future).note


def test_remotes_only_origin_is_unverifiable_extra_is_partial(repo: Path) -> None:
    ctx = fc.Ctx(root=repo)
    git(repo, "remote", "add", "origin", "https://example.invalid/o/r.git")
    res = fc.evaluate(row("remotes"), ctx)
    assert (res.status, "no network call" in res.note) == ("NOT VERIFIABLE LOCALLY", True)
    git(repo, "remote", "add", "public", "https://example.invalid/o/p.git")
    assert fc.evaluate(row("remotes"), ctx).status == "PARTIAL"


# ---- notebooks, OOF, scans ----
def nb_text(sha: str) -> str:
    return json.dumps({"cells": [{"source": [f'PINNED_SHA = "{sha}"\n']}]})


def test_notebook_pins(repo: Path) -> None:
    write(repo / "README.md")
    git(repo, "add", "README.md")
    git(repo, "commit", "-q", "-m", "init")
    sha = git(repo, "rev-parse", "HEAD").strip()
    ctx = fc.Ctx(root=repo)
    write(repo / "scripts" / "colab_build_x.py", f'PINNED_SHA = "{sha}"\n')
    write(repo / "notebooks" / "01_a.ipynb", nb_text(sha))
    assert fc.evaluate(row("notebooks"), ctx).status == "DONE"
    write(repo / "notebooks" / "02_b.ipynb", nb_text("FILL_ME"))
    write(repo / "notebooks" / "03_c.ipynb", nb_text("f" * 40))
    write(repo / "notebooks" / "04_d.ipynb", nb_text(sha[:-1] + ("0" if sha[-1] != "0" else "1")))
    assert fc.notebook_pins(ctx) == {
        "01_a.ipynb": "ok",
        "02_b.ipynb": "placeholder",
        "03_c.ipynb": "unresolved",
        "04_d.ipynb": "unresolved",
    }
    res = fc.evaluate(row("notebooks"), ctx)
    assert (res.status, "1/4 real pins" in res.note) == ("PARTIAL", True)


def superseded_nb(sha: str, banner: bool = True) -> str:
    title = "# t\n\n**SUPERSEDED (2026-10-03) by `x`. Do not run.**\n" if banner else "# t\n"
    return json.dumps(
        {
            "cells": [
                {"cell_type": "markdown", "source": [title]},
                {"cell_type": "code", "source": [f'PINNED_SHA = "{sha}"\n']},
            ]
        }
    )


def test_superseded_notebooks_are_excluded_from_real_pins(repo: Path) -> None:
    write(repo / "README.md")
    git(repo, "add", "README.md")
    git(repo, "commit", "-q", "-m", "init")
    sha = git(repo, "rev-parse", "HEAD").strip()
    ctx = fc.Ctx(root=repo)
    write(repo / "scripts" / "colab_build_x.py", f'PINNED_SHA = "{sha}"\n')
    write(repo / "notebooks" / "01_a.ipynb", nb_text(sha))
    write(repo / "notebooks" / "02_old.ipynb", superseded_nb("FILL_ME"))
    assert fc.notebook_pins(ctx) == {"01_a.ipynb": "ok", "02_old.ipynb": "superseded"}
    res = fc.evaluate(row("notebooks"), ctx)
    assert res.status == "DONE"
    assert "1/1 real pins, 1 superseded" in res.note
    # the same placeholder WITHOUT the banner still counts against the pins
    write(repo / "notebooks" / "03_new.ipynb", superseded_nb("FILL_ME", banner=False))
    res = fc.evaluate(row("notebooks"), ctx)
    assert (res.status, "1/2 real pins, 1 superseded" in res.note) == ("PARTIAL", True)
    # the banner only counts in the title (first markdown) cell
    late = json.dumps(
        {
            "cells": [
                {"cell_type": "markdown", "source": ["# t\n"]},
                {"cell_type": "markdown", "source": ["**SUPERSEDED (2026-10-03) by `x`.**"]},
                {"cell_type": "code", "source": ['PINNED_SHA = "FILL_ME"\n']},
            ]
        }
    )
    write(repo / "notebooks" / "03_new.ipynb", late)
    assert fc.notebook_pins(ctx)["03_new.ipynb"] == "placeholder"


def test_notebook_pin_missing_from_builder(repo: Path) -> None:
    write(repo / "README.md")
    git(repo, "add", "README.md")
    git(repo, "commit", "-q", "-m", "init")
    sha = git(repo, "rev-parse", "HEAD").strip()
    write(repo / "scripts" / "colab_build_x.py", "nothing = 1\n")
    write(repo / "notebooks" / "01_a.ipynb", nb_text(sha))
    assert fc.notebook_pins(fc.Ctx(root=repo)) == {"01_a.ipynb": "not-in-builder"}


def test_oof_pending_partial_done(tmp_path: Path) -> None:
    ext = tmp_path / "ext"
    ctx = fc.Ctx(root=tmp_path, ext=ext)
    write(tmp_path / "base.md")
    assert fc.evaluate(row("oof"), ctx).status == "PENDING"
    assert fc.evaluate(row("oof", base=["base.md"]), ctx).status == "PARTIAL"
    write(ext / "runs" / "oof_fold1_x" / "metrics.json")
    assert "fold2" in fc.evaluate(row("oof", base=["base.md"]), ctx).note
    write(ext / "runs" / "ft_fold2_oof_y" / "predictions.json")
    assert fc.evaluate(row("oof"), ctx).status == "DONE"
    assert fc.evaluate(row("oof", then="BLOCKED-ON-GG"), ctx).status == "BLOCKED-ON-GG"
    write(ext / "runs" / "ft_fold0_x" / "metrics.json")  # fold 0 never counts
    assert fc.evaluate(row("oof"), ctx).status == "DONE"


def test_no_paid_api_scan(tmp_path: Path) -> None:
    ctx = fc.Ctx(root=tmp_path)
    assert fc.evaluate(row("no_paid_api"), ctx).status == "NOT VERIFIABLE LOCALLY"
    write(tmp_path / "src" / "a.py", "import json\nfrom pathlib import Path\n")
    assert fc.evaluate(row("no_paid_api"), ctx).status == "DONE"
    write(tmp_path / "src" / "b.py", "import openai\n")
    assert fc.evaluate(row("no_paid_api"), ctx).status == "PARTIAL"


def test_no_test_labels(tmp_path: Path) -> None:
    ctx = fc.Ctx(root=tmp_path)
    assert fc.evaluate(row("no_test_labels"), ctx).status == "NOT VERIFIABLE LOCALLY"
    write(tmp_path / "data" / "test" / "images" / "a_p1.png")
    assert fc.evaluate(row("no_test_labels"), ctx).status == "DONE"
    write(tmp_path / "data" / "test" / "labels" / "a.json", "{}")
    assert fc.evaluate(row("no_test_labels"), ctx).status == "PARTIAL"


def test_json_true(tmp_path: Path) -> None:
    ctx = fc.Ctx(root=tmp_path)
    assert fc.evaluate(row("json_true", path="v.json", key="ok"), ctx).status == "PENDING"
    write(tmp_path / "v.json", '{"ok": true}')
    assert fc.evaluate(row("json_true", path="v.json", key="ok"), ctx).status == "DONE"
    write(tmp_path / "v.json", '{"ok": false}')
    assert fc.evaluate(row("json_true", path="v.json", key="ok"), ctx).status == "PARTIAL"


# ---- rendering ----
def test_render_is_idempotent_and_has_summary_and_blockers(tmp_path: Path) -> None:
    rows = (
        fc.Row(
            "A1",
            "brief p1",
            "done one",
            "deliverable",
            "a.txt",
            "exists",
            "r",
            args={"paths": ["a.txt"]},
        ),
        fc.Row(
            "A2",
            "spec 1",
            "waiting | one",
            "criterion",
            "b.txt",
            "exists",
            "r",
            args={"paths": ["b.txt"]},
            waits_for="the b run",
            unblock="GG runs a notebook",
        ),
        fc.Row(
            "A3",
            "brief p2",
            "gate",
            "constraint",
            "x",
            "fixed",
            "r",
            args={"status": "BLOCKED-ON-GG", "note": "n"},
            unblock="GG approves / publishes",
        ),
    )
    write(tmp_path / "a.txt")
    ctx = fc.Ctx(root=tmp_path, ext=tmp_path)
    res = fc.run_all(rows, ctx)
    one = fc.render(res, "abc1234", "2026-10-03", "cmd")
    two = fc.render(fc.run_all(rows, ctx), "abc1234", "2026-10-03", "cmd")
    assert one == two
    assert "| DONE | 1 |" in one
    assert "| PENDING | 1 |" in one
    assert "| BLOCKED-ON-GG | 1 |" in one
    assert "waiting \\| one" in one
    assert "### GG runs a notebook (1)" in one
    assert "- A2 [PENDING] waiting \\| one" not in one or "the b run" in one
    assert "A1 [DONE]" not in one  # finished rows are not listed as blockers


def _zs_run(ext: Path, folder: str, name: str, h: str) -> Path:
    d = ext / "runs" / "zeroshot500" / folder
    write(d / "metrics.json", "header_field_accuracy")
    write(d / "manifest.json", json.dumps({"config": {"name": name, "hash": h}}))
    return d


def test_two_zs_runs_without_an_argument_are_not_silently_chosen(tmp_path: Path) -> None:
    _zs_run(tmp_path, "zeroshot500_cfg_keyed_aaa", "cfg", "h1")
    _zs_run(tmp_path, "zeroshot500_cfg_native_bbb", "cfg_native", "hn")
    ctx = fc.Ctx(root=tmp_path, ext=tmp_path)
    res = fc.evaluate(row("text", path=fc.ZS_METRICS, pattern="header"), ctx)
    assert res.status == "NOT VERIFIABLE LOCALLY" and "--zs-run" in res.note
    named = fc.Ctx(root=tmp_path, ext=tmp_path, zs_run="zeroshot500_cfg_native_bbb")
    assert fc.evaluate(row("text", path=fc.ZS_METRICS, pattern="header"), named).status == "DONE"
    assert fc.evaluate(row("zs_provenance"), named).status == "DONE"


def test_a_single_zs_run_is_still_found_and_none_is_pending(tmp_path: Path) -> None:
    ctx = fc.Ctx(root=tmp_path, ext=tmp_path)
    assert fc.evaluate(row("text", path=fc.ZS_METRICS, pattern="header"), ctx).status == "PENDING"
    _zs_run(tmp_path, "zeroshot500_cfg_keyed_aaa", "cfg", "h1")
    assert fc.evaluate(row("text", path=fc.ZS_METRICS, pattern="header"), ctx).status == "DONE"


def test_provenance_rows_label_native_vs_1260_and_flag_mixing(tmp_path: Path) -> None:
    cfgs = tmp_path / "configs"
    cfgs.mkdir()
    for fn in ("spike_qwen35_4b_img_only.yaml", "spike_qwen35_4b_img_only_native.yaml"):
        text = (ROOT / "configs" / fn).read_text(encoding="utf-8")
        (cfgs / fn).write_text(text, encoding="utf-8")
    from shipdoc.spike import load_config

    legacy = load_config(cfgs / "spike_qwen35_4b_img_only.yaml")
    native = load_config(cfgs / "spike_qwen35_4b_img_only_native.yaml")
    ext = tmp_path / "ext"
    _zs_run(ext, "zeroshot500_n", native.name, native.config_hash)
    ctx = fc.Ctx(root=tmp_path, ext=ext)
    zs = fc.evaluate(row("zs_provenance"), ctx)
    assert zs.status == "DONE" and ": native" in zs.note and native.config_hash in zs.note
    assert fc.evaluate(row("oof_resolution"), ctx).status == "PENDING"  # no OOF run yet
    write(ext / "runs" / "oof_fold0_x" / "manifest.json").write_text(
        json.dumps({"config": {"name": native.name, "hash": native.config_hash}}), encoding="utf-8"
    )
    ok = fc.evaluate(row("oof_resolution"), ctx)
    assert ok.status == "DONE" and "native" in ok.note
    write(ext / "runs" / "oof_fold1_y" / "manifest.json").write_text(
        json.dumps({"config": {"name": legacy.name, "hash": legacy.config_hash}}), encoding="utf-8"
    )
    bad = fc.evaluate(row("oof_resolution"), ctx)
    assert bad.status == "PARTIAL" and "MIXED-RESOLUTION" in bad.note and "1260-token" in bad.note


def _native_zs(ext: Path, cfg_file: str, keys: str = "header_field_accuracy") -> Path:
    """A fake ``tmp/native_extract/zs/<NATIVE_ZS_RUN>`` whose manifest names a real config."""
    from shipdoc.spike import load_config

    cfg = load_config(ROOT / "configs" / cfg_file)
    d = ext / "tmp" / "native_extract" / "zs" / fc.NATIVE_ZS_RUN
    write(d / "metrics.json", json.dumps({k: 1 for k in keys.split(",")}))
    write(d / "manifest.json", json.dumps({"config": {"name": cfg.name, "hash": cfg.config_hash}}))
    return d


def _native_row(key: str) -> object:
    return row("native_metric", path=fc.NATIVE_ZS_METRICS, pattern=key)


def test_native_metric_rows_read_the_native_run_and_say_so(tmp_path: Path) -> None:
    ctx = fc.Ctx(root=ROOT, ext=tmp_path)
    assert fc.evaluate(_native_row("row_f1"), ctx).status == "PENDING"  # no run yet
    _native_zs(tmp_path, "spike_qwen35_4b_img_only_native.yaml", "row_f1,false_fill_rate")
    ok = fc.evaluate(_native_row("row_f1"), ctx)
    assert ok.status == "DONE" and ": native" in ok.note and fc.NATIVE_ZS_RUN in ok.note
    assert fc.evaluate(_native_row("false_fill_rate"), ctx).status == "DONE"
    missing_key = fc.evaluate(_native_row("documents_fully_correct"), ctx)
    assert missing_key.status == "PENDING" and "documents_fully_correct" in missing_key.note


def test_native_metric_refuses_the_1260_token_run_and_a_run_without_manifest(
    tmp_path: Path,
) -> None:
    ctx = fc.Ctx(root=ROOT, ext=tmp_path)
    d = _native_zs(tmp_path, "spike_qwen35_4b_img_only.yaml")  # the superseded resolution
    res = fc.evaluate(_native_row("header_field_accuracy"), ctx)
    assert res.status == "PARTIAL" and "1260-token" in res.note and "not native" in res.note
    (d / "manifest.json").unlink()
    res = fc.evaluate(_native_row("header_field_accuracy"), ctx)
    assert res.status == "PARTIAL" and "unverified" in res.note


def test_criteria_rows_c02_c03_c04_c07_point_at_the_native_run() -> None:
    by_id = {r.id: r for r in fc.ROWS}
    for rid, key in (
        ("C02", "header_field_accuracy"),
        ("C03", "row_f1"),
        ("C04", "documents_fully_correct"),
        ("C07", "false_fill_rate"),
    ):
        r = by_id[rid]
        assert r.check == "native_metric", rid
        assert r.args == {"path": fc.NATIVE_ZS_METRICS, "pattern": key}, rid
    assert (
        "native_4c17aa3" in fc.NATIVE_ZS_METRICS and "runs/zeroshot500" not in fc.NATIVE_ZS_METRICS
    )


def test_to_fill_slots_make_a_report_a_draft(tmp_path: Path) -> None:
    assert fc.pending_markers("a **[[TO FILL: x]]** b [[PENDING: y]] DUMMY NUMBERS") == 3
    pdf = tmp_path / "rep" / "report.pdf"
    pdf.parent.mkdir()
    pdf.write_bytes(make_pdf(2))
    write(pdf.with_suffix(".md"), "text **[[TO FILL: link github]]**")
    res = fc.evaluate(
        row("report_pdf", candidates=["rep/report.pdf"]), fc.Ctx(root=tmp_path, ext=tmp_path)
    )
    assert res.status == "PARTIAL" and "DRAFT, 1 pending" in res.note


def test_report_section_unverified_mark_is_partial(tmp_path: Path) -> None:
    write(tmp_path / "r.md", "# T\n## 8. Cost\n1.17† s per page\n## 9. Next\nclean\n")
    ctx = fc.Ctx(root=tmp_path, ext=tmp_path)
    res = fc.evaluate(row("report_section", heading="cost", path="r.md", mark="†"), ctx)
    assert res.status == "PARTIAL" and "1 figure(s)" in res.note
    assert (
        fc.evaluate(row("report_section", heading="next", path="r.md", mark="†"), ctx).status
        == "DONE"
    )


def test_dry_run_is_never_done_and_exists_else_has_three_states(tmp_path: Path) -> None:
    ctx = fc.Ctx(root=tmp_path, ext=tmp_path)
    dry = row("dry_run", paths=["a.md"], what="cards ready")
    assert fc.evaluate(dry, ctx).status == "PENDING"
    write(tmp_path / "a.md")
    res = fc.evaluate(dry, ctx)
    assert res.status == "BLOCKED-ON-GG" and res.note.startswith("DRY RUN, needs GG approval")
    ee = row("exists_else", paths=["out.json"], fallback=["copy.json"], note="copy only")
    assert fc.evaluate(ee, ctx).status == "PENDING"
    write(tmp_path / "copy.json")
    assert fc.evaluate(ee, ctx).status == "PARTIAL"
    write(tmp_path / "out.json")
    assert fc.evaluate(ee, ctx).status == "DONE"


def test_submission_sha256_prefix_must_match(tmp_path: Path) -> None:
    root, ext = tmp_path / "root", tmp_path / "ext"
    write(root / "assignment" / "schema.json", json.dumps({"type": "object"}))
    write(root / "assignment" / "sample_submission.json", json.dumps({"d1": {}}))
    write(ext / "submissions" / "v15_abc" / "test_predictions.json", '{"d1": {}}')
    ctx = fc.Ctx(root=root, ext=ext)
    import hashlib

    digest = hashlib.sha256(b'{"d1": {}}').hexdigest()
    ok = fc.evaluate(row("submission", final_prefix="v15", sha256_prefix=digest[:8]), ctx)
    assert ok.status == "DONE" and digest[:8] in ok.note
    bad = fc.evaluate(row("submission", final_prefix="v15", sha256_prefix="00000000"), ctx)
    assert bad.status == "PARTIAL" and "does NOT match" in bad.note


def test_public_smoke_placeholder_pin_is_not_a_missing_pin(repo: Path) -> None:
    write(repo / "README.md")
    git(repo, "add", "README.md")
    git(repo, "commit", "-q", "-m", "init")
    sha = git(repo, "rev-parse", "HEAD").strip()
    write(repo / "scripts" / "colab_build_x.py", f'PINNED_SHA = "{sha}"\n')
    write(repo / "notebooks" / "01_a.ipynb", nb_text(sha))
    write(repo / "notebooks" / "public_smoke.ipynb", nb_text("FILL_PINNED_SHA"))
    ctx = fc.Ctx(root=repo)
    assert fc.notebook_pins(ctx)["public_smoke.ipynb"] == "public-pin"
    res = fc.evaluate(row("notebooks"), ctx)
    assert res.status == "DONE" and "public-repo-pin-only" in res.note


def test_main_writes_file_with_head_sha(tmp_path: Path) -> None:
    out = tmp_path / "out" / "final_checklist.md"
    assert fc.main(["--ext", str(tmp_path / "noext"), "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "## What blocks completion" in text
    assert "| id | requirement |" in text
    assert b"\r" not in out.read_bytes()  # LF endings, so the file is byte-stable across OSes
