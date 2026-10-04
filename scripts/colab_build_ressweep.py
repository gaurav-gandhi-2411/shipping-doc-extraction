"""Generate notebooks/06_res_sweep.ipynb (resolution sweep of the production extractor, T4, fp16).

A thin wrapper like 01-05: every cell only orchestrates (Drive, secrets, clone, install, one
subprocess per stage); the logic is in ``python -m shipdoc spike`` (the runs) and
``python -m shipdoc.ressweep estimate | check-control | analyse`` (src/shipdoc/ressweep.py).
Helper cells (account, mount, secrets, clone, install) are imported READ-ONLY from
scripts/colab_build_notebook.py and edited by exact-string replacement (asserted), so the
notebooks cannot drift apart silently. Nothing in 01-05 or their builders is modified.

tests/test_notebook_ressweep.py checks the committed notebook equals this script's output.

Run: uv run python scripts/colab_build_ressweep.py
"""

# ruff: noqa: E501  # cell sources are verbatim notebook text

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "archive" / "06_res_sweep.ipynb"


def _load(name: str, path: Path):  # noqa: ANN202 - a module object
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


base = _load("colab_build_notebook", ROOT / "scripts" / "colab_build_notebook.py")
rs = _load("shipdoc_ressweep_builder", ROOT / "src" / "shipdoc" / "ressweep.py")

# The pin commit replaces this with the full 40-char SHA of the pushed code commit and regenerates
# the notebook (same two-commit pattern as 01-05). Until then cell 1 raises.
PINNED_SHA = "ebf96614eb8f077cbc6743d3f9e5d5e1ead691e7"

TITLE = """# 06 - Resolution sweep of the production extractor (thin wrapper)

Does a higher `max_pixels` (more visual tokens per page) fix part-number misreads of the production
extractor on the 40 documents of `splits/spike40.json`, at a VRAM and speed cost a T4 can pay?
Production config: Qwen3.5-4B, IMAGE ONLY, KEYED output, prompt v2, greedy, xgrammar, seed 42,
logprobs on (Phase 5 needs them), fp16. Labels are dev labels only; no test file is read.

**What the pixel arithmetic says (read before running).** Patch 16 x merge 2 means one visual token
per 32 x 32 px of the resized page. Every page of the corpus is 1240 x 1754 px. The production cap
(`max_pixels` 1,310,720) gives a 1344 x 960 grid = **1,260 tokens**. The native page resizes to
1760 x 1248 = **2,145 tokens**, and the processor only ever SHRINKS to the cap, it never upscales:
any `max_pixels` above 2,196,480 changes nothing. So about 2,500 and about 3,500 visual tokens are
**unreachable through `max_pixels` alone** (they would need a raised `min_pixels`, which the
extractor does not set). The sweep therefore runs the control (1,260), 1,750 tokens
(`max_pixels` 1,843,200) and 2,145 tokens (`max_pixels` 2,196,480, the native page). The estimate
cell prints the grid and the token count of every resolution and refuses a list whose token counts
do not strictly increase.

Flow: parameters -> Drive + secrets -> clone at the pin -> unzip data -> **ESTIMATE of T4 hours and
compute units (printed before anything runs)** -> install -> **smoke gate** on `splits/smoke5.json`
at the LARGEST resolution (the OOM risk; peak VRAM is recorded) -> one resumable run per
resolution, each its own subprocess / run id / status entry -> **paired analysis** -> banner.

Runs (separate run ids `ressweep_<config>_<sha7>` under `MyDrive/shipdoc-extract/runs/`): the CONTROL
is the production config itself (`configs/spike_qwen35_4b_img_only.yaml`); the others are
`configs/spike_qwen35_4b_img_only_keyed_px<N>.yaml`, the same file with only `name` and `max_pixels`
changed. **Control reuse is never silent**: if `CONTROL_RUN_DIR` is set, `python -m shipdoc.ressweep
check-control` compares config hash, code SHA (the full pin), model revision, document list (sha of
the ordered list), batch size, logprobs, output format, shard and completeness with strict equality;
any mismatch is printed and the notebook runs the control itself. **OOM or any failure at a
resolution is recorded in the status file as `oom` / `failed` and the resolution is EXCLUDED from
the decision**, never skipped quietly.

Analysis (CPU, `shipdoc.eval.paired_bootstrap`, the unmodified scorer, 2000 resamples, seed 42,
document level): per non-control resolution the paired delta vs the control for OVERALL and row F1
(all documents, scanned, digital), the part-number misread count (`spn_misread` of
`scripts/row_error_diagnosis.py`, the definition behind "66 misreads, 41 on scans") with a paired
bootstrap CI of the count difference, seconds per page and peak VRAM from the trace. The misread
count is exact when `ocr_cache.zip` is on Drive (gold rows are placed on pages with the OCR cache);
without it the count is a printed LOWER BOUND.

**Decision rule**: adopt the HIGHEST resolution whose paired OVERALL delta CI lower bound is > 0 AND
whose peak VRAM is <= 14.5 GiB; ties (equal OVERALL point estimates) go to the LOWER resolution;
if none qualifies keep the control. **The CI is on 40 documents (wide); the decision uses OVERALL as
its only criterion; every other delta is reported, not decisive.**

Batch contract: greedy outputs are only comparable at equal batch size. `BATCH_SIZE` defaults to 1
for every run and the analysis refuses to pair runs whose manifests differ. No batch bench is run.

Every number printed before it has been measured is an ESTIMATE (speeds from the spike40 traces,
ASSUMED linear prefill scaling in input tokens, ASSUMED 120 s model load); the OOM behaviour,
speed and VRAM at 1,750 / 2,145 tokens are UNVERIFIED on a GPU until this notebook has run.

Parameters: `PINNED_SHA`; `BATCH_SIZE` (int, default 1); `CONTROL_RUN_DIR` (None, or a run folder of
the production config on the same documents, relative to `runs/` or absolute); `RESOLUTIONS`
(first entry = control, `max_pixels` strictly increasing, each needs its config file).

If the session dies, open the notebook and Run all: setup repeats, the smoke gate is skipped when it
passed, finished runs are skipped and an unfinished run resumes after the last document in its
`trace.jsonl`. See notebooks/README.md.
"""

RES_LINES = chr(10).join(f"    ({px}, {json.dumps(cfg)})," for px, cfg in rs.RESOLUTIONS)

PARAMS = f"""# Parameters. PINNED_SHA = the repo commit this run is pinned to (full 40-char SHA).
PINNED_SHA = "{PINNED_SHA}"
RUN_NAME = "ressweep"
DOCS = "splits/spike40.json"  # 40 dev documents, 55 pages
EXPECTED_DOCS = 40
EXPECTED_PAGES = 55
SMOKE_DOCS = "splits/smoke5.json"  # 5 dev docs picked by meta tags (shipdoc.smoke)
SMOKE_LIMIT = 5
BATCH_SIZE = 1  # pages per generate call; the same for EVERY run (outputs compare at equal size)
CONTROL_RUN_DIR = None  # None = run the control; else a run folder that must pass check-control
# (max_pixels, config name): the first entry is the control = the production config.
RESOLUTIONS = [
{RES_LINES}
]

if not PINNED_SHA or "FILL" in PINNED_SHA:
    raise ValueError("Set PINNED_SHA in the parameters cell (the pin commit fills it).")
if isinstance(BATCH_SIZE, bool) or not isinstance(BATCH_SIZE, int) or BATCH_SIZE < 1:
    raise ValueError(f"BATCH_SIZE must be an int >= 1, got {{BATCH_SIZE!r}}")
if CONTROL_RUN_DIR is not None and not isinstance(CONTROL_RUN_DIR, str):
    raise ValueError(f"CONTROL_RUN_DIR must be None or a string, got {{CONTROL_RUN_DIR!r}}")
RESOLUTIONS = [(int(px), str(cfg)) for px, cfg in RESOLUTIONS]
if len(RESOLUTIONS) < 2 or [p for p, _ in RESOLUTIONS] != sorted({{p for p, _ in RESOLUTIONS}}):
    raise ValueError("RESOLUTIONS: a control plus at least one more, max_pixels strictly rising")
if RESOLUTIONS[0] != (1310720, "qwen35_4b_img_only"):
    raise ValueError("the first RESOLUTIONS entry is the control: the production config, "
                     "(1310720, 'qwen35_4b_img_only')")
SHA7 = PINNED_SHA[:7]
CONTROL_PX, LARGEST_PX = RESOLUTIONS[0][0], RESOLUTIONS[-1][0]
LARGEST_CFG = RESOLUTIONS[-1][1]
RUN_IDS = {{px: f"{{RUN_NAME}}_{{cfg}}_{{SHA7}}" for px, cfg in RESOLUTIONS}}
"""

_ZIPS_OLD = 'REQUIRED_ZIPS = ["data.zip", "assignment.zip", "ocr_cache.zip"]'
assert _ZIPS_OLD in base.MOUNT
# Image only: no OCR text in the runs. data.zip holds data/ (dev labels and images), assignment.zip
# the schema and the official scorer; ocr_cache.zip is optional (analysis only, see UNZIP).
MOUNT = (
    base.MOUNT.replace(_ZIPS_OLD, 'REQUIRED_ZIPS = ["data.zip", "assignment.zip"]')
    + """

def _under_runs(p: str) -> Path:
    # absolute as given; 'runs/...' under MyDrive/shipdoc-extract; a bare name under runs/
    if Path(p).is_absolute():
        return Path(p)
    return DRIVE_DIR / p if p.startswith("runs/") else RUNS_DIR / p


CONTROL = _under_runs(CONTROL_RUN_DIR) if CONTROL_RUN_DIR else None
OUT_DIR = RUNS_DIR / f"{RUN_NAME}_{SHA7}"  # status.json, ressweep_result.json / .md
STATUS_PATH = OUT_DIR / "status.json"
print("resolutions:", ", ".join(f"{px} ({cfg})" for px, cfg in RESOLUTIONS))
print("batch size:", BATCH_SIZE, "| control:", f"reuse {CONTROL} if it passes check-control"
      if CONTROL else "run it")
print("Drive output:", OUT_DIR)
"""
)

_sec = base.ZS_SECRETS
_SECRETS_EDITS = [
    (
        'WANDB_API_KEY = get_secret("WANDB_API_KEY", required=True) if USE_WANDB else None\n',
        "",
    ),
    (
        'print("GH_TOKEN: set | HF_TOKEN:", "set" if HF_TOKEN else "not set (optional)",\n      "| W&B:", "ON" if USE_WANDB else "off")\n',
        'print("GH_TOKEN: set | HF_TOKEN:", "set" if HF_TOKEN else "not set (optional)")\n',
    ),
]
for _old, _new in _SECRETS_EDITS:
    assert _old in _sec, f"colab_build_notebook.ZS_SECRETS changed: {_old[:50]!r}"
    _sec = _sec.replace(_old, _new)
SECRETS = _sec

UNZIP = """import json
import shutil
import zipfile

LOCAL_ZIPS = Path("/content/zips")
LOCAL_ZIPS.mkdir(exist_ok=True)
CONTENT = Path("/content")
# data.zip holds data/ and assignment.zip holds assignment/; both unzip to /content (not into the
# clone), and the CLI is pointed at them with SHIPDOC_DATA_DIR / SHIPDOC_ASSIGNMENT_DIR below.
for name in ("data.zip", "assignment.zip"):
    local = LOCAL_ZIPS / name
    if not local.is_file() or local.stat().st_size != (DRIVE_DIR / name).stat().st_size:
        shutil.copyfile(DRIVE_DIR / name, local)  # local disk: Drive reads are slow
    with zipfile.ZipFile(local) as zf:
        files = [n for n in zf.namelist() if not n.endswith("/")]
        if any(not (CONTENT / n).is_file() for n in files):  # also repairs a half-extracted tree
            zf.extractall(CONTENT)
    print(f"{name}: {len(files)} files under {CONTENT / files[0].split('/')[0]}")

DATA_DIR, ASSIGNMENT_DIR = CONTENT / "data", CONTENT / "assignment"
DEV_LABELS = DATA_DIR / "dev" / "labels"
assert (ASSIGNMENT_DIR / "score.py").is_file(), "assignment.zip has no score.py (official scorer)"
doc_ids = json.loads((REPO / DOCS).read_text(encoding="utf-8"))
assert len(doc_ids) == EXPECTED_DOCS == len(set(doc_ids)), f"{DOCS}: expected {EXPECTED_DOCS} docs"
assert all(d.startswith("dev_") for d in doc_ids), "spike40 holds dev documents only"
missing = [d for d in doc_ids if not (DEV_LABELS / f"{d}.json").is_file()]
assert not missing, f"{len(missing)} docs without a label file, e.g. {missing[:3]}"
n_pages = sum(len(json.loads((DEV_LABELS / f"{d}.json").read_text())["pages"]) for d in doc_ids)
assert n_pages == EXPECTED_PAGES, f"{n_pages} pages counted, expected {EXPECTED_PAGES}"
print(f"data OK: {len(doc_ids)} docs, {n_pages} pages, labels are dev only, no test file is read")

# Optional, analysis only: the OCR cache of the 40 documents places the gold rows on pages, which
# makes the part-number misread count exact. Without it the analysis prints a LOWER BOUND.
OCR_CACHE = None
if (DRIVE_DIR / "ocr_cache.zip").is_file():
    local = LOCAL_ZIPS / "ocr_cache.zip"
    if not local.is_file() or local.stat().st_size != (DRIVE_DIR / "ocr_cache.zip").stat().st_size:
        shutil.copyfile(DRIVE_DIR / "ocr_cache.zip", local)
    with zipfile.ZipFile(local) as zf:
        wanted = [n for n in zf.namelist() if not n.endswith("/")
                  and any(Path(n).name.startswith(f"{d}_p") for d in doc_ids)]
        for n in wanted:
            if not (CONTENT / n).is_file():
                zf.extract(n, CONTENT)
    OCR_CACHE = CONTENT / "ocr_cache"
    print(f"ocr_cache.zip: {len(wanted)} page files of the {len(doc_ids)} docs (analysis only)")
else:
    print("ocr_cache.zip not on Drive: the misread count will be a printed LOWER BOUND")
"""

ESTIMATE = """# T4-hour and compute-unit (CU) ESTIMATE BEFORE anything runs on the GPU: read it. It runs in this
# process (stdlib + PIL, no venv yet): `shipdoc.ressweep estimate` checks the page sizes of the 40
# documents, prints max_pixels -> grid -> visual tokens (and raises when two resolutions give the
# same token count), then the hours at both CU rates. Nothing in it is measured. No confirmation
# prompt (Run all must not block): interrupt the run yourself if it is too high.
import importlib.util
import sys

_spec = importlib.util.spec_from_file_location(
    "shipdoc_ressweep", REPO / "src" / "shipdoc" / "ressweep.py")
rs = importlib.util.module_from_spec(_spec)
sys.modules["shipdoc_ressweep"] = rs  # dataclasses resolve annotations through sys.modules
_spec.loader.exec_module(rs)

rc = rs.main(["estimate", "--docs", DOCS, "--smoke-docs", SMOKE_DOCS,
              "--batch-size", str(BATCH_SIZE), "--data-dir", str(DATA_DIR),
              "--resolutions", ",".join(str(px) for px, _ in RESOLUTIONS)])
assert rc == 0, f"estimate failed (exit {rc})"
if BATCH_SIZE != 1:
    print("=" * 78)
    print(f"BATCH SIZE {BATCH_SIZE}: no bench was run, nothing verifies that it gives the same")
    print("outputs as batch 1. Outputs are only comparable at equal batch size: the control")
    print("uses it too.")
    print("=" * 78)
print("THIS RUN: batch size", BATCH_SIZE, "| control:", "reuse if check-control passes" if CONTROL
      else "run it", "| both CU rates above (1.19 / 1.58) are third-party observations")
"""

_inst = base.ZS_INSTALL
_INSTALL_EDITS = [
    (
        'VLM_GROUP = ["--group", "vlm"] if MODE == "run" else []  # merge mode needs no torch / model stack\n',
        'VLM_GROUP = ["--group", "vlm"]\n',
    ),
    (
        "if USE_WANDB:\n    ENV.update(WANDB_API_KEY=WANDB_API_KEY, WANDB_PROJECT=WANDB_PROJECT, WANDB_RUN_GROUP=RUN_NAME)\n",
        "",
    ),
    (
        "'arch': torch.cuda.get_arch_list(),",
        "'arch': torch.cuda.get_arch_list(), 'name': torch.cuda.get_device_name(0),",
    ),
]
for _old, _new in _INSTALL_EDITS:
    assert _old in _inst, f"colab_build_notebook.ZS_INSTALL changed: {_old[:50]!r}"
    _inst = _inst.replace(_old, _new)
# The GPU probe sits in an `if MODE == "run":` block of the zero-shot notebook; this notebook has
# no merge mode, so the block is un-indented instead of keeping a dead MODE parameter.
_HDR = 'if MODE == "run":\n'
assert _inst.count(_HDR) == 1, "colab_build_notebook.ZS_INSTALL: the MODE block moved"
_head, _tail = _inst.split(_HDR)
_lines = _tail.splitlines(True)
assert all(ln.startswith("    ") or not ln.strip() for ln in _lines), "MODE block is not indented"
INSTALL = (
    _head
    + "".join(ln[4:] if ln.strip() else ln for ln in _lines)
    + """
print(f"GPU {info['name']}: written for a T4 (fp16, sm_75); other GPUs also run it")
"""
)

SMOKE = """# MANDATORY smoke gate at the LARGEST resolution (the OOM risk): the 5 smoke docs with the largest
# resolution's config, then 7 assertions on the output (shipdoc.smoke: JSON validity, no truncation,
# row counts, quantity / total share, no all-null rows, finite field logprobs). The peak VRAM of the
# smoke run is recorded in the status file. Outcomes: passed -> go on; CUDA OOM -> the largest
# resolution is recorded as `oom` and EXCLUDED (the other resolutions still run); anything else
# (exit code, failed assertion) raises and nothing after this cell runs. Resumable (status file).
import importlib.util
import re
import sys
import time

import yaml

_smoke_path = REPO / "src" / "shipdoc" / "smoke.py"
_spec = importlib.util.spec_from_file_location("shipdoc_smoke", _smoke_path)
smoke = importlib.util.module_from_spec(_spec)
sys.modules["shipdoc_smoke"] = smoke  # dataclasses resolves annotations through sys.modules
_spec.loader.exec_module(smoke)

OOM_RE = re.compile(r"out of memory|OutOfMemoryError", re.IGNORECASE)
SMOKE_RUNS = RUNS_DIR / "smoke"
SMOKE_RUN_ID = f"smoke_{RUN_NAME}_{LARGEST_CFG}_{SHA7}"
SMOKE_STATUS_PATH = RUNS_DIR / f"{RUN_NAME}_{SHA7}_smoke_status.json"
smoke_status = json.loads(SMOKE_STATUS_PATH.read_text()) if SMOKE_STATUS_PATH.is_file() else {}
SMOKE_ENV = {**ENV, "SHIPDOC_RUNS_DIR": str(SMOKE_RUNS)}
SMOKE_PASSED, SMOKE_OOM = False, False
smoke_ids = json.loads((REPO / SMOKE_DOCS).read_text(encoding="utf-8"))[:SMOKE_LIMIT]


def save_smoke_status() -> None:
    SMOKE_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = SMOKE_STATUS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(smoke_status, indent=1))
    tmp.replace(SMOKE_STATUS_PATH)  # atomic: a disconnect never leaves a half-written file


def smoke_peak_vram(run_dir: Path) -> int | None:
    path = run_dir / "trace.jsonl"
    if not path.is_file():
        return None
    vals = [p["meta"].get("peak_vram_bytes") or 0 for ln in path.read_text().splitlines()
            if ln.strip() for p in json.loads(ln)["pages"]]
    return max(vals) if vals and max(vals) else None


if smoke_status.get("run_id") == SMOKE_RUN_ID and smoke_status.get("state") in ("passed", "oom"):
    SMOKE_PASSED = smoke_status["state"] == "passed"
    SMOKE_OOM = smoke_status["state"] == "oom"
    print(f"SKIP smoke: already {smoke_status['state']} ({SMOKE_RUN_ID})")
else:
    run_dir = SMOKE_RUNS / SMOKE_RUN_ID
    t0 = time.time()
    rc, tail, checks, warns, error = 0, [], [], [], None
    try:
        max_new = int(yaml.safe_load((REPO / f"configs/spike_{LARGEST_CFG}.yaml").read_text())[
            "max_new_tokens"])
        if not (run_dir / "metrics.json").is_file():  # a finished run is re-checked, not re-run
            print(f"=== {SMOKE_RUN_ID}: {len(smoke_ids)} docs at max_pixels {LARGEST_PX} ===",
                  flush=True)
            cmd = [PY, "-m", "shipdoc", "spike", "--config", f"configs/spike_{LARGEST_CFG}.yaml",
                   "--docs", SMOKE_DOCS, "--split", "dev", "--run-id", SMOKE_RUN_ID, "--resume",
                   "--limit", str(SMOKE_LIMIT), "--logprobs", "--batch-size", str(BATCH_SIZE)]
            rc, tail = run_stream(cmd, env=SMOKE_ENV)
        if rc == 0:
            checks = smoke.load_and_check(run_dir, DEV_LABELS, max_new, require_logprobs=True)
            warns = smoke.load_warnings(run_dir)  # non-blocking: never raises, never fails
    except (OSError, KeyError, ValueError) as exc:  # fail closed: could not verify = fail
        rc, error = 1, f"{type(exc).__name__}: {exc}"
    peak = smoke_peak_vram(run_dir)
    if rc != 0 and OOM_RE.search("\\n".join(tail)):
        state = "oom"
        print(f"SMOKE OOM at max_pixels {LARGEST_PX}: that resolution is recorded as oom and "
              "EXCLUDED; the other resolutions still run.")
    elif rc != 0:
        state = "error"
        print(f"SMOKE ERROR: exit {rc} {error or ''}; last output: {tail[-3:]}")
    else:
        state = "passed" if all(c.passed for c in checks) else "failed"
        print(smoke.format_table(LARGEST_CFG, checks, warns))
    smoke_status = {
        "state": state,
        "run_id": SMOKE_RUN_ID,
        "max_pixels": LARGEST_PX,
        "exit_code": rc,
        "seconds": round(time.time() - t0, 1),
        "peak_vram_bytes": peak,
        "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail} for c in checks],
        "warnings": [{"name": w.name, "count": w.count, "detail": w.detail} for w in warns],
        "error": error,
        "tail": tail[-5:] if rc else [],
    }
    save_smoke_status()
    SMOKE_PASSED, SMOKE_OOM = state == "passed", state == "oom"

if smoke_status.get("peak_vram_bytes"):
    print(f"smoke peak VRAM at max_pixels {LARGEST_PX}: "
          f"{smoke_status['peak_vram_bytes'] / 2**30:.2f} GiB (decision limit 14.5 GiB)")
if not (SMOKE_PASSED or SMOKE_OOM):
    raise RuntimeError(
        "SMOKE GATE FAILED: the runs are NOT started. GG: report back with "
        f"{SMOKE_STATUS_PATH} and the per-assertion table printed above, plus (they stay on "
        f"Drive, do not paste document values into chat) {SMOKE_RUNS / SMOKE_RUN_ID}/trace.jsonl "
        "and predictions.json. Do not edit thresholds to make it pass."
    )
print("SMOKE GATE PASSED: starting the runs." if SMOKE_PASSED else
      "SMOKE OOM recorded: the largest resolution is excluded; starting the others.")
"""

RUN = """# One RESUMABLE run per resolution (control first, then ascending), each its own subprocess, run
# id and entry in status.json (state: complete | oom | failed | incomplete). The CLI appends one
# line to trace.jsonl and rewrites predictions.json / progress.json after every document; --resume
# skips documents already traced. An OOM or any failure at a candidate resolution is RECORDED and
# that resolution is excluded from the decision, never skipped quietly; a failed CONTROL stops the
# notebook (nothing to compare against). Control reuse: only if CONTROL_RUN_DIR passes
# `check-control`; a refusal is printed in full and the control runs here instead.
import re
import threading
import time
from datetime import UTC, datetime

OOM_RE = re.compile(r"out of memory|OutOfMemoryError", re.IGNORECASE)
if not (SMOKE_PASSED or SMOKE_OOM):
    raise RuntimeError("The smoke gate has not passed: the runs are NOT started.")
OUT_DIR.mkdir(parents=True, exist_ok=True)
status = json.loads(STATUS_PATH.read_text()) if STATUS_PATH.is_file() else {}


def save_json(path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)  # atomic: a disconnect never leaves a half-written file


def read_progress(run_dir: Path) -> dict:
    try:
        return json.loads((run_dir / "progress.json").read_text())
    except (OSError, ValueError):
        return {}


def watch_progress(stop: threading.Event, run_dir: Path, every: float = 60.0) -> None:
    last, t_start = -1, time.time()
    while not stop.wait(every):
        p = read_progress(run_dir)
        if p.get("done") != last:
            last = p.get("done")
            print(f"[progress] {p.get('done')}/{p.get('total')} docs, last {p.get('last_doc')}, "
                  f"{(time.time() - t_start) / 60:.0f} min this session", flush=True)


CONTROL_KEY = str(CONTROL_PX)
CONTROL_REUSED = False
if CONTROL is not None:
    chk = [PY, "-m", "shipdoc.ressweep", "check-control", "--run-dir", str(CONTROL),
           "--pin", PINNED_SHA, "--docs", DOCS, "--batch-size", str(BATCH_SIZE)]
    rc, tail = run_stream(chk)
    CONTROL_REUSED = rc == 0
    if CONTROL_REUSED:
        status[CONTROL_KEY] = {"state": "complete", "source": "reused", "run_dir": str(CONTROL),
                               "run_id": CONTROL.name, "config": RESOLUTIONS[0][1]}
    else:
        print("!" * 78)
        print("CONTROL NOT REUSED (reasons above): the control is run by this notebook instead.")
        print("!" * 78)
if not CONTROL_REUSED and status.get(CONTROL_KEY, {}).get("source") == "reused":
    status.pop(CONTROL_KEY)  # an earlier session reused a control that is no longer requested
save_json(STATUS_PATH, status)


def run_resolution(px: int, cfg: str) -> None:
    run_id = RUN_IDS[px]
    run_dir = RUNS_DIR / run_id
    entry = status.setdefault(str(px), {})
    entry.update(run_id=run_id, run_dir=str(run_dir), config=cfg, source="ran")
    sessions = entry.setdefault("sessions", [])
    done_before = read_progress(run_dir).get("done", 0)
    print(f"=== {run_id} (max_pixels {px}): {done_before}/{EXPECTED_DOCS} docs already done; "
          f"resuming the rest (batch size {BATCH_SIZE}) ===")
    session = {"start": datetime.now(UTC).isoformat(), "end": None, "seconds": None,
               "docs_done_at_start": done_before, "exit_code": None}
    sessions.append(session)
    entry["state"] = "running"
    save_json(STATUS_PATH, status)
    cmd = [PY, "-m", "shipdoc", "spike", "--config", f"configs/spike_{cfg}.yaml", "--docs", DOCS,
           "--split", "dev", "--run-id", run_id, "--resume", "--logprobs",
           "--batch-size", str(BATCH_SIZE)]
    stop = threading.Event()
    threading.Thread(target=watch_progress, args=(stop, run_dir), daemon=True).start()
    t0 = time.time()
    try:
        rc, tail = run_stream(cmd)
    finally:
        stop.set()
        session.update(end=datetime.now(UTC).isoformat(), seconds=round(time.time() - t0, 1))
    session["exit_code"] = rc
    prog = read_progress(run_dir)
    if rc == 0 and prog.get("status") == "complete" and prog.get("done") == EXPECTED_DOCS:
        entry["state"] = "complete"
    elif rc == 0:
        entry["state"] = "incomplete"
    elif OOM_RE.search("\\n".join(tail)):
        entry["state"] = "oom"
    else:
        entry["state"] = "failed"
    entry["tail"] = tail[-5:] if rc else []
    save_json(STATUS_PATH, status)
    print(f"--- {run_id}: {entry['state']} (exit {rc}) in {(time.time() - t0) / 60:.1f} min ---")


for px, cfg in RESOLUTIONS:
    key = str(px)
    if px == CONTROL_PX and CONTROL_REUSED:
        print(f"=== control reused from {CONTROL}: not run ===")
        continue
    if px == LARGEST_PX and SMOKE_OOM:
        status[key] = {"state": "oom", "source": "smoke", "config": cfg,
                       "note": "CUDA OOM in the smoke gate at this resolution"}
        save_json(STATUS_PATH, status)
        print(f"=== max_pixels {px}: EXCLUDED, recorded as oom by the smoke gate ===")
        continue
    prev = status.get(key, {})
    if prev.get("state") == "oom":
        print(f"=== max_pixels {px}: EXCLUDED, recorded as oom earlier (delete its entry in "
              f"{STATUS_PATH.name} to retry) ===")
        continue
    if prev.get("state") == "complete" and prev.get("source") == "ran":
        p = read_progress(Path(prev["run_dir"]))
        if p.get("status") == "complete" and p.get("done") == EXPECTED_DOCS:
            print(f"=== max_pixels {px}: already complete ({prev['run_id']}), skipped ===")
            continue
    run_resolution(px, cfg)
    if px == CONTROL_PX and status[key]["state"] != "complete":
        raise RuntimeError(
            f"CONTROL run is {status[key]['state']}: there is nothing to compare against. "
            f"Everything done is on Drive; Run all again resumes. Last output: "
            f"{status[key].get('tail')}")

print("-" * 78)
for px, _ in RESOLUTIONS:
    e = status.get(str(px), {})
    print(f"max_pixels {px:>8}: {e.get('state', 'pending'):<10} ({e.get('source', '-')})")
print("-" * 78)
"""

ANALYSE = """# PAIRED ANALYSIS (CPU, no model): every non-control resolution against the control, then the
# decision rule. Official scorer (unmodified) vs the dev gold; paired document-level bootstrap
# (2000 resamples, seed 42) of OVERALL and row F1 (all, scanned, digital), the part-number misread
# count with a paired bootstrap CI of the count difference, s/page and peak VRAM from the traces.
# Runs that are oom / failed / incomplete are listed and excluded. The CI is on 40 documents
# (wide); the decision uses OVERALL only. Writes ressweep_result.json / .md (aggregates only).
ana_cmd = [PY, "-m", "shipdoc.ressweep", "analyse", "--status", str(STATUS_PATH),
           "--docs", DOCS, "--out-dir", str(OUT_DIR)]
ana_cmd += ["--ocr-cache", str(OCR_CACHE)] if OCR_CACHE else ["--no-ocr"]
rc, tail = run_stream(ana_cmd)
if rc != 0:
    raise RuntimeError(f"ANALYSIS FAILED (exit {rc}): {tail[-3:]}. The runs are on Drive; "
                       "report back (counts only).")
"""

BANNER = """# Completion banner. Reads only files on Drive: also runnable alone after a resume.
import json

bar = "=" * 78
BS = chr(92)
LOCAL_DEST = BS.join(["$SHIPDOC_RUNS_DIR"]) + BS
res_path = OUT_DIR / "ressweep_result.json"
res = json.loads(res_path.read_text()) if res_path.is_file() else None
st = json.loads(STATUS_PATH.read_text()) if STATUS_PATH.is_file() else {}
print(bar)
for px, cfg in RESOLUTIONS:
    e = st.get(str(px), {})
    print(f"max_pixels {px:>8} ({cfg}): {e.get('state', 'pending')} [{e.get('source', '-')}]")
excluded = [px for px, _ in RESOLUTIONS[1:] if st.get(str(px), {}).get("state") != "complete"]
if excluded:
    print(f"EXCLUDED from the decision (oom / failed / incomplete): {excluded}")
if res is None:
    print(f"NOT ANALYSED: {res_path.name} missing. Run the cells above.")
    print(f"DONE ressweep {SHA7} analysed=False adopted=NOT_DECIDED")
else:
    print(f"batch size {res['control']['batch_size']}; misread basis {res['control']['ocr_basis']}")
    print("The CI is on 40 documents (wide); the decision uses OVERALL only; other deltas are")
    print("reported, not decisive.")
    print(f"Drive folder: {OUT_DIR}")
    print("GG: download these into the LOCAL folder", LOCAL_DEST, "(outside the repo; traces and")
    print("predictions hold document values, never paste them into chat):")
    names = [OUT_DIR / n for n in ("status.json", "ressweep_result.json", "ressweep_result.md")]
    names += [RUNS_DIR / e["run_id"] / n for e in st.values() if e.get("source") == "ran"
              for n in ("predictions.json", "trace.jsonl", "manifest.json", "metrics.json")]
    for path in names:
        size = f"{path.stat().st_size / 1e6:8.2f} MB" if path.is_file() else "MISSING"
        print(f"  {str(path.relative_to(DRIVE_DIR)):<70}{size}")
    print("NEXT: the verdict is on 40 documents; adopting a resolution changes production")
    print("inference and needs GG's decision, a batch-size check at that resolution and the")
    print("same resolution at train time.")
    print(f"DONE ressweep {SHA7} analysed=True adopted={res['adopted']}")
print(bar)
"""


def _cells() -> list[tuple[str, str]]:
    return [
        ("markdown", TITLE),
        ("code", PARAMS),
        ("markdown", base.ACCOUNT_MD),
        ("code", MOUNT),
        ("code", SECRETS),
        ("code", base.CLONE),
        ("code", UNZIP),
        ("code", ESTIMATE),
        ("code", INSTALL),
        ("code", SMOKE),
        ("code", RUN),
        ("code", ANALYSE),
        ("code", BANNER),
    ]


def build() -> dict:
    """Notebook dict (nbformat 4.5) with all outputs cleared."""
    return base._notebook(_cells())


def render() -> str:
    """Canonical JSON text of the notebook."""
    return json.dumps(build(), indent=1, ensure_ascii=False) + "\n"


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(render(), encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")
