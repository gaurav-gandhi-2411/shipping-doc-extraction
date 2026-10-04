"""Generate notebooks/04_predict_test.ipynb (safety submission v0) from the cell sources below.

A thin wrapper like 01/02: every cell only orchestrates (Drive, secrets, clone, install, one
subprocess per stage); the logic is in ``python -m shipdoc predict <stage>``
(src/shipdoc/predict.py). Helpers (mount, clone, install cells) are imported READ-ONLY from
scripts/colab_build_notebook.py, so the two notebooks cannot drift apart.

tests/test_notebook_predict.py checks the committed notebook equals this script's output.

Run: uv run python scripts/colab_build_predict.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "archive" / "04_predict_test.ipynb"


def _load(name: str, path: Path):  # noqa: ANN202 - a module object
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclass-free modules, but keep the import contract uniform
    spec.loader.exec_module(mod)
    return mod


base = _load("colab_build_notebook", ROOT / "scripts" / "colab_build_notebook.py")

# The pin commit replaces this with the full 40-char SHA of the pushed code commit and
# regenerates the notebook (same two-commit pattern as 01/02: the SHA never refers to itself).
# For now it is the current 02 pin; GG re-pins before the run.
PINNED_SHA = "42b812b5b09d6e4bff0df12564017f71ffad5fc9"

TITLE = """# 04 - test predictions, safety submission v0 (thin wrapper)

Production config: Qwen3.5-4B, IMAGE ONLY, KEYED output format (full field names), prompt v2,
greedy decoding, seed 42, fp16 on a T4, logprobs on (as in the bench). Post-processing = the repo's
default Phase 3 pipeline, exactly as used for the dev100 final pick (reports/ablations.md G3):
ON provenance-aware merge (page-1 identity, last-page totals), repeated header-row filter, drop
all-null rows, dates -> ISO, number normalisation, code normalisation (currency / airport);
OFF OCR labelled-total pointer (no OCR at inference), per-layout-cluster day/month order (no
clusters at inference); ISO 4217 / IATA / AWB validators give flags only, never edit a value.
The exact switches are written into `manifest.json` (read from the code, not typed here).
No OCR stage: Paddle is not installed and never run.

It predicts ALL test documents (200 docs / 280 pages) from the page IMAGES only. There are no test
labels; nothing here scores anything. Test predictions never enter git: everything is written to
Drive (`MyDrive/shipdoc-extract/submissions/v0_<sha7>/`) and downloaded by hand into
`$SHIPDOC_SUBMISSIONS_DIR\\v0_<sha7>\\` (gitignored `submissions/`). Never paste document values
into chat; the reports and the banner carry counts only.

Flow: setup (Drive, clone at the pin, unzip data incl. test images) -> T4-hour / compute-unit
ESTIMATE for batch 1/2/4/8 (printed before anything runs; interrupt if too high) -> install ->
plan (ids from the test image FOLDER names) -> MANDATORY 5-dev-doc SMOKE gate (same run and the same
7 checks as notebook 02; the test run refuses to start unless it passed for this code) ->
BATCH SIZE (the dev run's stored bench result is reused when it applies to this code, config and
model, else the bench runs on the 12 dev pages exactly as in 02; the size must equal the dev
run's) -> resumable per-document TEST RUN -> DETERMINISM pass (5 seeded test documents decoded
again in a fresh process, replaying the exact generate calls of the first pass; the serialized
predictions must be byte-identical) -> VALIDATION
(JSON Schema, exactly the 200 test ids, no duplicates, provenance) -> completion banner.

Parameters: `PINNED_SHA`; `BATCH_SIZE` (None = from the bench, an int = manual and the bench is
skipped); `SHARD` ("i/K": this tab runs document-level shard i of K, "0/1" = everything); `MODE`
("run", or "merge" to combine the K shard folders after all tabs finished, then validate);
`DEV_RUN_DIR` (the dev run, e.g. the 02 zero-shot folder under runs/, whose manifest holds the
batch size and the bench result the test run must match; None = no comparison, said loudly).

If the session dies, open the notebook and Run all: the setup cells repeat, the smoke gate and the
batch decision are reused, the test run resumes after the last document in `trace.jsonl`.
See notebooks/README.md.
"""

PARAMS = f"""# Parameters. PINNED_SHA = the repo commit this run is pinned to (full 40-char SHA).
PINNED_SHA = "{PINNED_SHA}"
RUN_NAME = "test"
CONFIG = "qwen35_4b_img_only"  # KEYED output format, prompt v2, image only: the production pick
SPLIT = "test"
EXPECTED_DOCS = 200
EXPECTED_PAGES = 280
SMOKE_DOCS = "splits/smoke5.json"  # 5 dev docs picked by meta tags (shipdoc.smoke)
BENCH_DOCS = "splits/bench12.json"  # 12 dev pages for the batch-size bench (shipdoc.bench)
BATCH_SIZE = None  # None = the bench decides (and must equal the dev run's); an int = manual
SHARD = "0/1"  # "i/K": this tab runs document-level shard i of K; "0/1" = everything
MODE = "run"  # "run" = smoke, batch size, test run, determinism, validate; "merge" = combine shards
DEV_RUN_DIR = "zeroshot500_qwen35_4b_img_only_keyed_42b812b"  # dev run folder under runs/, or None

if not PINNED_SHA or "FILL" in PINNED_SHA:
    raise ValueError("Set PINNED_SHA in the parameters cell (the pin commit fills it).")
if MODE not in ("run", "merge"):
    raise ValueError(f"MODE must be 'run' or 'merge', got {{MODE!r}}")
if BATCH_SIZE is not None and (isinstance(BATCH_SIZE, bool) or not isinstance(BATCH_SIZE, int)
                               or BATCH_SIZE < 1):
    raise ValueError(f"BATCH_SIZE must be None or an int >= 1, got {{BATCH_SIZE!r}}")
_i, _, _k = SHARD.partition("/")
if not (_i.isdigit() and _k.isdigit() and int(_k) >= 1 and int(_i) < int(_k)):
    raise ValueError(f"SHARD must look like 'i/K' with 0 <= i < K, got {{SHARD!r}}")
SHARD_I, SHARD_K = int(_i), int(_k)
if MODE == "merge" and SHARD_K < 2:
    raise ValueError("MODE='merge' combines K shard folders: set SHARD to '0/K' with K >= 2.")
SHA7 = PINNED_SHA[:7]
RUN_ID = f"{{RUN_NAME}}_{{CONFIG}}_keyed_{{SHA7}}"  # the unsharded run (and the merge target)
SHARD_RUN_ID = RUN_ID if SHARD_K == 1 else f"{{RUN_ID}}_shard{{SHARD_I}}of{{SHARD_K}}"  # this tab
TAB = f"{{SHARD_I}}of{{SHARD_K}}"  # suffix of the files that belong to this tab only
"""

_ZIPS_OLD = 'REQUIRED_ZIPS = ["data.zip", "assignment.zip", "ocr_cache.zip"]'
assert _ZIPS_OLD in base.MOUNT
# Image only: no OCR cache. data.zip holds data/ (incl. the test images), assignment.zip the
# schema, the sample submission and the scorer.
MOUNT = (
    base.MOUNT.replace(_ZIPS_OLD, 'REQUIRED_ZIPS = ["data.zip", "assignment.zip"]')
    + """SUBMISSIONS_DIR = DRIVE_DIR / "submissions"  # test predictions live here, not in the repo
SUBMISSION_DIR = SUBMISSIONS_DIR / f"v0_{SHA7}"
META_DIR = RUNS_DIR / f"{RUN_ID}_meta"  # plan, smoke status and batch decision of the tabs
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)
RUN_DIR = RUNS_DIR / (RUN_ID if MODE == "merge" else SHARD_RUN_ID)
PLAN_PATH = META_DIR / "plan.json"
TEST_DOCS_PATH = META_DIR / "test_docs.json"  # ids only: the doc list of merge-shards
SMOKE_STATUS_PATH = META_DIR / f"smoke_status_{TAB}.json"
DECISION_PATH = META_DIR / f"batch_decision_{TAB}.json"
# The dev run whose manifest holds the batch size the test run must match (None = no comparison).
DEV_RUN = None
if DEV_RUN_DIR:
    DEV_RUN = Path(DEV_RUN_DIR) if Path(DEV_RUN_DIR).is_absolute() else RUNS_DIR / DEV_RUN_DIR
    if not (DEV_RUN / "manifest.json").is_file():
        print("!" * 78)
        print(f"DEV RUN NOT FOUND: {DEV_RUN} has no manifest.json. The bench is used as is and the")
        print("batch-size contract with the dev run is NOT checked (the manifest records this).")
        print("!" * 78)
        DEV_RUN = None
else:
    print("DEV_RUN_DIR is None: no dev run to compare the batch size with (recorded as unchecked).")
print("run_id:", RUN_ID, "| mode:", MODE, "| shard:", SHARD, "| batch size:", BATCH_SIZE or "bench")
print("Drive run folder:", RUN_DIR)
print("Drive submission folder:", SUBMISSION_DIR)
"""
)

SECRETS = '''import os

from google.colab import userdata


def get_secret(name: str, required: bool) -> str | None:
    """Read a Colab secret without ever printing it."""
    try:
        value = userdata.get(name)
    except Exception:  # SecretNotFoundError / NotebookAccessError: both mean "not usable"
        value = None
    if required and not value:
        raise RuntimeError(
            f"Colab secret {name} is missing or notebook access is off "
            "(left sidebar -> key icon -> add it and enable Notebook access)."
        )
    return value or None


GH_TOKEN = get_secret("GH_TOKEN", required=True)
HF_TOKEN = get_secret("HF_TOKEN", required=False)
print("GH_TOKEN: set | HF_TOKEN:", "set" if HF_TOKEN else "not set (optional)")
'''

UNZIP = """import json
import re
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
DEV_LABELS = DATA_DIR / "dev" / "labels"  # the smoke gate and the bench use DEV docs only
TEST_IMAGES = DATA_DIR / SPLIT / "images"
SCHEMA_PATH = ASSIGNMENT_DIR / "schema.json"
SAMPLE_PATH = ASSIGNMENT_DIR / "sample_submission.json"
assert SCHEMA_PATH.is_file(), "assignment.zip has no schema.json"
assert TEST_IMAGES.is_dir(), f"data.zip has no {TEST_IMAGES}: the test images are required"
# There are no test labels and this notebook never looks for any.
assert not (DATA_DIR / SPLIT / "labels").exists(), "unexpected test labels in data.zip: stop"
STEM = re.compile(r"^(.+)_p(\\d+)$")
_pages = {}
for _p in TEST_IMAGES.iterdir():
    _m = STEM.match(_p.stem)
    if _p.is_file() and _m:
        _pages[_m.group(1)] = _pages.get(_m.group(1), 0) + 1  # ids from FILE NAMES only
test_pages_by_doc = dict(sorted(_pages.items()))
assert len(test_pages_by_doc) == EXPECTED_DOCS, f"{len(test_pages_by_doc)} test docs found"
assert sum(test_pages_by_doc.values()) == EXPECTED_PAGES, "test page count differs"
print(f"data OK: {EXPECTED_DOCS} test docs / {EXPECTED_PAGES} page images (no labels); dev labels:",
      DEV_LABELS.is_dir())
"""

ESTIMATE = """# T4-hour and compute-unit (CU) estimate BEFORE anything runs on the GPU: read it.
# No confirmation prompt (Run all must not block): interrupt the run yourself if it is too high.
import importlib.util
import json
import subprocess

_ge_path = REPO / "scripts" / "gpu_estimate.py"
_spec = importlib.util.spec_from_file_location("gpu_estimate", _ge_path)
ge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ge)

SPEED = json.loads((REPO / "configs" / "spike_speed.json").read_text(encoding="utf-8"))
smoke_ids = json.loads((REPO / SMOKE_DOCS).read_text(encoding="utf-8"))


def read_gpu_name():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=False,
        ).stdout.strip().splitlines()
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out[0].strip() if out else None


def test_estimate_table(speed, config, pages_by_doc, smoke_pages, gpu_name):
    \"\"\"ESTIMATE (UNVERIFIED) hours and CU of the v0 test run per batch size, 1 and 2 tabs.\"\"\"
    m = speed["models"][ge.base_config(config)]
    n_out = float(m["n_output_tokens_mean"])
    load_s = float(speed.get("model_load_s", 0))
    r1, r2 = speed.get("t4_cu_per_hour"), speed.get("t4_cu_per_hour_conservative")
    n_pages, n_docs = sum(pages_by_doc.values()), len(pages_by_doc)
    multi = [n for n in pages_by_doc.values() if n > 1]
    # 2 multi-page + 3 single-page documents are re-decoded (shipdoc.predict.DET_RULE)
    det_pages = round(2 * sum(multi) / len(multi) + 3) if multi else 5

    def cu(rate, hours):
        return "n/a" if rate is None else f"{hours * rate:.1f}"

    bar = "=" * 124
    head = (f"ESTIMATE (UNVERIFIED) T4 hours / CU: v0 TEST run, {n_docs} docs / {n_pages} "
            f"pages, {config} (keyed). Batching gain ASSUMED until the bench measures it")
    lines = [bar, head, bar]
    warn = ge.gpu_warning(gpu_name)
    if warn:
        lines.append(warn)
    lines += [
        f"per tab: model load {load_s:.0f} s (ESTIMATE) for the smoke ({smoke_pages} pages, "
        f"batch 1), the test run and the determinism pass ({det_pages} pages, a fresh "
        "process = one more load); 'bench reused' = the dev bench result applies (no bench), "
        "'bench rerun' = worst case (one load + 12 dev pages at batch 1/2/4/8).",
        f"s/page = prefill_s {m['prefill_s']} + {n_out} tok x straggler(B) / "
        f"({m['decode_tok_s']} tok/s x eff x B), straggler = {ge.STRAGGLER}; eff = scaling "
        "efficiency of the decode.",
        "",
        f"{'scenario':<22} {'K':>1} {'s/page':>6} | {'wall h':>6} {'CU@' + str(r1):>8} "
        f"{'CU@' + str(r2):>8} | {'wall h':>6} {'CU@' + str(r1):>8} {'CU@' + str(r2):>8}",
        f"{'':<22} {'':>1} {'':>6} | {'bench reused':^24} | {'bench rerun':^24}",
    ]
    for b, eff in ge.batch_scenarios():
        name = "B=1 (no bench)" if b == 1 else f"B={b} @ {int(eff * 100)}% scaling"
        s = ge.batch_s_per_page(m["prefill_s"], m["decode_tok_s"], n_out, b, eff)
        for k in ge.SHARD_COUNTS:
            det_h = (load_s + det_pages * s) / 3600
            cells = []
            for rerun in (False, True):
                h = ge.tab_hours(speed, config, n_out, b, eff, n_pages / k,
                                 smoke_pages=smoke_pages, with_bench=rerun and b > 1) + det_h
                cells.append(f"{h:6.2f} {cu(r1, k * h):>8} {cu(r2, k * h):>8}")
            lines.append(f"{name:<22} {k:>1} {s:6.1f} | {cells[0]} | {cells[1]}")
    lines += [
        "",
        "wall h = one tab; CU = K tabs summed (K = 2: two Colab tabs, MODE='merge' after).",
        f"CU rate {r1}/h: {speed.get('t4_cu_source', 'source unset')}",
        f"CU rate {r2}/h: {speed.get('t4_cu_conservative_source', 'source unset')}",
        "Not modelled: logprob capture overhead (assumed 0), Drive I/O, session restarts, OOM "
        "fallbacks, the validation step (CPU, seconds).",
        bar,
    ]
    return "\\n".join(lines)


GPU_NAME = read_gpu_name()
print("Runtime GPU (nvidia-smi):", GPU_NAME)
smoke_pages = ge.count_pages(smoke_ids, DEV_LABELS)
print(test_estimate_table(SPEED, CONFIG, test_pages_by_doc, smoke_pages, GPU_NAME))
print(f"THIS TAB: shard {SHARD}, batch size {BATCH_SIZE or 'from the bench'}, mode {MODE}.")
"""

_inst = base.ZS_INSTALL
_INSTALL_EDITS = [
    (
        '    "SHIPDOC_OCR_CACHE": "/content/ocr_cache",  # unused by img_only; the path is '
        "never read\n",
        '    "SHIPDOC_SUBMISSIONS_DIR": str(SUBMISSIONS_DIR),\n',
    ),
    (
        "if USE_WANDB:\n    ENV.update(WANDB_API_KEY=WANDB_API_KEY, WANDB_PROJECT=WANDB_PROJECT, "
        "WANDB_RUN_GROUP=RUN_NAME)\n",
        "",
    ),
    ("for p in ('transformers', 'xgrammar')", "for p in ('transformers', 'xgrammar', 'pycountry')"),
]
for _old, _new in _INSTALL_EDITS:
    assert _old in _inst, f"colab_build_notebook.ZS_INSTALL changed: {_old[:40]!r}"
    _inst = _inst.replace(_old, _new)
INSTALL = (
    _inst
    + """
# Image only: the OCR stack must NOT be in this environment (it is its own dependency group).
_no_ocr = "import importlib.util as u, sys; sys.exit(0 if u.find_spec('paddle') is None else 1)"
assert subprocess.run([PY, "-c", _no_ocr], env=ENV, check=False).returncode == 0, (
    "the OCR stack is installed in the venv: this notebook runs image only"
)
"""
)

PLAN = """# Plan: document ids come from the test image FOLDER NAMES (never labels); the shard's
# documents and the 5 seeded determinism documents are fixed here (rule in the output).
rc, tail = run_stream([PY, "-m", "shipdoc", "predict", "plan", "--shard", SHARD,
                       "--out", str(PLAN_PATH)])
assert rc == 0, f"plan failed (exit {rc}): {tail[-3:]}"
plan = json.loads(PLAN_PATH.read_text())
assert plan["n_docs"] == EXPECTED_DOCS and plan["n_pages"] == EXPECTED_PAGES, "plan counts differ"
assert sorted(plan["docs"]) == sorted(test_pages_by_doc), "plan ids differ from the file names"
TEST_DOCS_PATH.write_text(json.dumps(plan["docs"]))
print("determinism rule:", plan["determinism_rule"])
"""

SMOKE = """# MANDATORY smoke gate (same run and same 7 checks as notebook 02: keyed format, prompt
# v2, logprobs, 5 DEV docs): `shipdoc predict smoke` writes a status file bound to this
# code, config and model revision; the test run refuses to start unless it says `passed`.
# A failure raises.
SMOKE_RUN_ID = f"smoke_{SHARD_RUN_ID}"
CFG = f"configs/spike_{CONFIG}.yaml"
smoke_cmd = [PY, "-m", "shipdoc", "predict", "smoke", "--config", CFG,
             "--docs", SMOKE_DOCS, "--run-id", SMOKE_RUN_ID, "--status", str(SMOKE_STATUS_PATH)]
rc, tail = run_stream(smoke_cmd)
SMOKE_PASSED = rc == 0
if not SMOKE_PASSED:
    raise RuntimeError(
        "SMOKE GATE FAILED: the test run is NOT started. GG: report back with "
        f"{SMOKE_STATUS_PATH} and the per-assertion table printed above (dev documents only). "
        "Do not edit thresholds to make it pass."
    )
print("SMOKE GATE PASSED: continuing.")
"""

BATCH = """# BATCH SIZE. BATCH_SIZE = None: `shipdoc predict batch` reuses the bench result when it
# applies to this code, config and model (the dev run's stored result is copied and verified
# first), else runs the bench on the 12 dev pages exactly as notebook 02 (largest size with 100%
# byte-identical outputs vs batch 1 and peak VRAM <= 14.5 GiB, ...). Either way the size must
# EQUAL the dev run's (shipdoc.runmeta): greedy outputs are only comparable at the same batch
# size. A mismatch raises.
if not SMOKE_PASSED:
    raise RuntimeError("The smoke gate has not passed: batch size and test run are NOT run.")
BENCH_DIR = RUNS_DIR / f"bench_{CONFIG}_{SHA7}"
CFG = f"configs/spike_{CONFIG}.yaml"
batch_cmd = [PY, "-m", "shipdoc", "predict", "batch", "--config", CFG,
             "--bench-docs", BENCH_DOCS, "--bench-dir", str(BENCH_DIR),
             "--decision", str(DECISION_PATH)]
if DEV_RUN is not None:
    batch_cmd += ["--dev-run-dir", str(DEV_RUN)]
if BATCH_SIZE is not None:
    batch_cmd += ["--batch-size", str(BATCH_SIZE)]
    print("=" * 78)
    print(f"MANUAL BATCH SIZE {BATCH_SIZE}: the bench is SKIPPED; nothing verifies that")
    print("this batch size gives the same outputs as batch 1 (the bench would).")
    print("=" * 78)
rc, tail = run_stream(batch_cmd)
if rc != 0:
    raise RuntimeError(f"BATCH SIZE REFUSED (exit {rc}): {tail[-3:]}. Test run NOT started.")
decision = json.loads(DECISION_PATH.read_text())
BATCH = int(decision["batch_size"])
print(f"BATCH SIZE FOR THE TEST RUN: {BATCH} ({decision['source']}); dev contract: "
      f"{decision['contract']}")
"""

RUN = """# Test run over the documents of this tab, one subprocess, RESUMABLE per doc: the CLI
# appends one line to trace.jsonl and rewrites predictions.json / progress.json after every
# document, and a resume skips the doc_ids already traced (the batch size is stored in
# manifest.json). It refuses unless the smoke gate passed for this code. A disconnect loses at
# most the window in flight.
import re
import threading
import time
from datetime import UTC, datetime

if not SMOKE_PASSED or "BATCH" not in globals():
    raise RuntimeError("Smoke gate or batch decision missing: the test run is NOT started.")
RUN_DIR.mkdir(parents=True, exist_ok=True)
OOM_RE = re.compile(r"out of memory|OutOfMemoryError", re.IGNORECASE)
SESSIONS_PATH = RUN_DIR / "sessions.json"  # wall-clock log, one entry per Run all
sessions = json.loads(SESSIONS_PATH.read_text()) if SESSIONS_PATH.is_file() else []
THIS_DOCS = plan["shard_n_docs"]


def save_json(path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)  # atomic: a disconnect never leaves a half-written file


def read_progress() -> dict:
    try:
        return json.loads((RUN_DIR / "progress.json").read_text())
    except (OSError, ValueError):
        return {}


def watch_progress(stop: threading.Event, every: float = 60.0) -> None:
    last, t_start = -1, time.time()
    while not stop.wait(every):
        p = read_progress()
        if p.get("done") != last:
            last = p.get("done")
            print(f"[progress] {p.get('done')}/{p.get('total')} docs, "
                  f"{(time.time() - t_start) / 60:.0f} min this session", flush=True)


done_before = read_progress().get("done", 0)
print(f"=== {SHARD_RUN_ID}: {done_before}/{THIS_DOCS} docs already done; resuming the rest "
      f"(batch size {BATCH}, shard {SHARD}) ===")
session = {"start": datetime.now(UTC).isoformat(), "end": None, "seconds": None,
           "docs_done_at_start": done_before, "exit_code": None}
sessions.append(session)
save_json(SESSIONS_PATH, sessions)
run_cmd = [PY, "-m", "shipdoc", "predict", "run", "--config", f"configs/spike_{CONFIG}.yaml",
           "--run-id", RUN_ID, "--shard", SHARD, "--decision", str(DECISION_PATH),
           "--smoke-status", str(SMOKE_STATUS_PATH),
           "--expect-docs", str(EXPECTED_DOCS), "--expect-pages", str(EXPECTED_PAGES)]
stop = threading.Event()
threading.Thread(target=watch_progress, args=(stop,), daemon=True).start()
t0 = time.time()
try:
    rc, tail = run_stream(run_cmd)
finally:
    stop.set()
    session.update(end=datetime.now(UTC).isoformat(), seconds=round(time.time() - t0, 1))
    save_json(SESSIONS_PATH, sessions)
session["exit_code"] = rc
save_json(SESSIONS_PATH, sessions)
if rc != 0:
    oom = OOM_RE.search("\\n".join(tail))
    raise RuntimeError(
        f"Test run stopped (exit {rc}{', CUDA out of memory' if oom else ''}) after "
        f"{read_progress().get('done')}/{THIS_DOCS} docs; last output: {tail[-3:]}. Everything "
        "done so far is on Drive: Run all again resumes. If it stops at the same point, "
        "report back."
    )
print(f"--- test run finished (exit 0) in {(time.time() - t0) / 60:.1f} min this session ---")
"""

DETERMINISM = """# DETERMINISM: 5 seeded test documents of this tab are decoded AGAIN in a fresh
# process (new model load, new CUDA context). The generate calls of the first pass that held
# those documents are REPLAYED exactly (same pages, same order inside the call), because a
# batched decode can depend on the companions of a page; the serialized per-document predictions
# must be byte-identical to the first pass. Counts only are printed.
# A difference raises: the submission is not assembled as submittable.
if not SMOKE_PASSED or "BATCH" not in globals():
    raise RuntimeError("Smoke gate or batch decision missing: the determinism pass is NOT run.")
det_cmd = [PY, "-m", "shipdoc", "predict", "determinism", "--config",
           f"configs/spike_{CONFIG}.yaml", "--run-id", RUN_ID, "--shard", SHARD,
           "--decision", str(DECISION_PATH)]
t_det = time.time()
rc, tail = run_stream(det_cmd)
if rc != 0:
    raise RuntimeError(f"DETERMINISM CHECK FAILED (exit {rc}): {tail[-3:]}. Report back "
                       "(counts only).")
sessions.append({"start": None, "end": None, "seconds": round(time.time() - t_det, 1),
                 "exit_code": 0, "stage": "determinism"})
save_json(SESSIONS_PATH, sessions)
"""

MERGE = """# MODE = 'merge': combine the K shard folders into the unsharded run folder (CPU).
# `python -m shipdoc merge-shards` REFUSES (and writes nothing) unless every shard is complete and
# the shards agree on code SHA, config hash, model revision, seed, batch size and doc list, no
# document is twice, and the merged counts are 200 docs / 280 pages. The assemble step below then
# also needs every shard's determinism pass.
if MODE == "merge":
    merge_cmd = [PY, "-m", "shipdoc", "merge-shards", "--config", f"configs/spike_{CONFIG}.yaml",
                 "--docs", str(TEST_DOCS_PATH), "--split", SPLIT, "--run-id", RUN_ID,
                 "--shards", str(SHARD_K), "--expected-docs", str(EXPECTED_DOCS),
                 "--expected-pages", str(EXPECTED_PAGES)]
    rc, tail = run_stream(merge_cmd)
    if rc != 0:
        raise RuntimeError(f"MERGE REFUSED (exit {rc}): {tail[-3:]}. Nothing was merged; fix the "
                           "shard that is named above (rerun it with Run all) and merge again.")
else:
    print("MODE = run: nothing to merge in this tab.")
"""

ASSEMBLE = """# VALIDATION + submission folder (CPU). `shipdoc predict assemble` checks, blocking:
# run complete; exactly the 200 test ids of the image folder and of the assignment sample (no
# extra, no duplicate); 200 docs / 280 pages; jsonschema validation against
# assignment/schema.json; determinism of every shard; same batch size as the dev run; code SHA
# clean and equal to PINNED_SHA; model revision / config hash / seed 42 / prompt v2; real model
# and recorded versions. Writes test_predictions.json, trace.jsonl, manifest.json and
# validation_report.json to SUBMISSION_DIR; on a failure the predictions go to
# test_predictions.REJECTED.json instead.
if MODE == "run" and SHARD_K > 1:
    print(f"Shard {SHARD} done. NEXT: when all {SHARD_K} tabs are COMPLETE, set MODE = 'merge' and "
          f"SHARD = '0/{SHARD_K}' in any tab and Run all: it merges, validates and assembles.")
else:
    asm_cmd = [PY, "-m", "shipdoc", "predict", "assemble", "--config",
               f"configs/spike_{CONFIG}.yaml", "--run-id", RUN_ID, "--out-dir", str(SUBMISSION_DIR),
               "--schema", str(SCHEMA_PATH), "--expect-docs", str(EXPECTED_DOCS),
               "--expect-pages", str(EXPECTED_PAGES), "--expect-code-sha", PINNED_SHA,
               "--require-stack", "--smoke-status", str(SMOKE_STATUS_PATH)]
    if SAMPLE_PATH.is_file():
        asm_cmd += ["--sample-submission", str(SAMPLE_PATH)]
    if DEV_RUN is not None:
        asm_cmd += ["--dev-run-dir", str(DEV_RUN)]
    rc, tail = run_stream(asm_cmd)
    if rc != 0:
        raise RuntimeError(
            f"VALIDATION FAILED (exit {rc}): the files in {SUBMISSION_DIR} are NOT submittable "
            "(test_predictions.REJECTED.json). Read validation_report.json there (counts and key "
            "paths only) and report back; do not submit."
        )
"""

BANNER = """# Completion banner. Reads only files on Drive: also runnable alone after a resume.
import json

bar = "=" * 78
BS = chr(92)
LOCAL_DEST = BS.join(["$SHIPDOC_SUBMISSIONS_DIR", f"v0_{SHA7}"]) + BS
report_path = SUBMISSION_DIR / "validation_report.json"
manifest_path = SUBMISSION_DIR / "manifest.json"
report = json.loads(report_path.read_text()) if report_path.is_file() else None
manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
print(bar)
if report is None:
    print(f"NOT ASSEMBLED: {report_path.name} missing. Run the cells above (MODE='merge' after "
          "all shards).")
    print(f"DONE {SHARD_RUN_ID} assembled=False")
else:
    ok = bool(report["ok"])
    print("TEST PREDICTIONS v0 " + ("VALIDATED: SUBMITTABLE" if ok else "REJECTED: DO NOT SUBMIT"))
    print(bar)
    print(f"schema ok       : {report['schema_ok']}")
    print(f"docs found      : {report['docs_found']}")
    print(f"determinism ok  : {report['determinism_ok']}")
    for name, c in report["checks"].items():
        print(f"  {'PASS' if c['ok'] else 'FAIL'}  {name}: {c.get('detail', '')}")
    print(f"code {str(manifest.get('code_sha'))[:12]}  model {manifest.get('model', {}).get('id')} "
          f"@ {str(manifest.get('model', {}).get('revision'))[:12]}  config "
          f"{manifest.get('config', {}).get('hash')}")
    print(f"batch size {manifest.get('batch_size')} ({manifest.get('batch_size_source')}); "
          f"contract: {manifest.get('batch_size_contract')}")
    tm = manifest.get("timings", {})
    print(f"wall-clock {tm.get('wall_clock_s')} s over {tm.get('sessions')} session(s); model time "
          f"{tm.get('model_time_s')} s; smoke {tm.get('smoke_s')} s")
    print(f"Drive folder: {SUBMISSION_DIR}")
    print("GG: download these files from that Drive folder:")
    wanted = ["test_predictions.json" if ok else "test_predictions.REJECTED.json",
              "trace.jsonl", "manifest.json", "validation_report.json"]
    for name in wanted:
        path = SUBMISSION_DIR / name
        size = f"{path.stat().st_size / 1e6:8.2f} MB" if path.is_file() else "MISSING"
        print(f"  {name:<34}{size}")
    print(f"into the LOCAL folder  {LOCAL_DEST}")
    print("(outside the repo; submissions/ is gitignored). Never paste document values into chat.")
    print(f"DONE {SHARD_RUN_ID} assembled=True submittable={ok}")
print(bar)
"""


def _run_only(src: str, what: str) -> str:
    """`src` guarded by ``if MODE == "run":`` (the merge tab skips the GPU stages)."""
    return base._run_only(src, what)


def _cells() -> list[tuple[str, str]]:
    return [
        ("markdown", TITLE),
        ("code", PARAMS),
        ("markdown", base.ACCOUNT_MD),
        ("code", MOUNT),
        ("code", SECRETS),
        ("code", base.CLONE),
        ("code", UNZIP),
        ("code", _run_only(ESTIMATE, "estimate")),
        ("code", INSTALL),
        ("code", PLAN),
        ("code", _run_only(SMOKE, "smoke gate")),
        ("code", _run_only(BATCH, "batch size")),
        ("code", _run_only(RUN, "test run")),
        ("code", _run_only(DETERMINISM, "determinism pass")),
        ("code", MERGE),
        ("code", ASSEMBLE),
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
