"""Generate notebooks/07_hdrhint_ab.ipynb (A/B of the continuation-page header hint on a T4, fp16).

A thin wrapper like 01-05: every cell only orchestrates (Drive, secrets, clone, install, one
subprocess per stage); the logic is in ``python -m shipdoc.headerhint plan | compare``
(src/shipdoc/headerhint.py) and in the opt-in ``header_hint`` config key of ``shipdoc spike``.
Helper cells (account, mount, secrets, clone) are imported READ-ONLY from
scripts/colab_build_notebook.py and edited by exact-string replacement (asserted), so the notebooks
cannot drift apart silently.

tests/test_notebook_hdrhint.py checks the committed notebook equals this script's output.

Run: uv run python scripts/colab_build_hdrhint.py
"""

# ruff: noqa: E501  # cell sources are verbatim notebook text

from __future__ import annotations

import importlib.util
import json
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "archive" / "07_hdrhint_ab.ipynb"


def _load(name: str, path: Path):  # noqa: ANN202 - a module object
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


base = _load("colab_build_notebook", ROOT / "scripts" / "colab_build_notebook.py")

# The decision rule text comes from the code that decides, so the notebook cannot paraphrase it.
sys.path.insert(0, str(ROOT / "src"))
from shipdoc import headerhint as hh  # noqa: E402

_RULE_WORDS = textwrap.wrap(hh.DECISION_RULE, 88)
_RULE_BODY = "\n".join(
    "    " + repr(w + (" " if i < len(_RULE_WORDS) - 1 else "")) for i, w in enumerate(_RULE_WORDS)
)
RULE_LITERAL = f"(\n{_RULE_BODY}\n)"  # a wrapped literal: notebook lines stay within 100 columns

# Placeholder: the pin commit replaces it with the full 40-char SHA of the pushed code commit and
# regenerates the notebook (same two-commit pattern as 01-05). While it holds the placeholder the
# parameters cell AND the clone cell refuse to run.
PINNED_SHA = "ebf96614eb8f077cbc6743d3f9e5d5e1ead691e7"

TITLE = f"""# 07 - header-hint A/B on multi-page documents (thin wrapper)

Does telling the model the page-1 table header row help it read the continuation pages? Arms:
**control** = the production config (Qwen3.5-4B, image only, KEYED output, prompt v2, greedy,
seed 42, logprobs on; the existing 02 run) vs **hint** = the same config plus ONE opt-in flag, `header_hint: true`
(`configs/spike_qwen35_4b_img_only_keyed_hdrhint.yaml`): for page >= 2 the prompt gets
`Column headers from page 1: <text>`, the OCR text of the page-1 table-header line (column order
preserved, left to right by box x; `shipdoc.layout.analyze_page`), capped at
{hh.HINT_MAX_CHARS} characters (about {hh.HINT_MAX_TOKENS} tokens). Generic: no supplier id, no cluster, no gold.
No table-header line on page 1 -> no hint (recorded per page in the trace).

**DEPENDENCY ON OCR AT INFERENCE TIME.** The hint text is OCR OUTPUT (PaddleOCR) of page 1, so
adopting the hint makes production inference depend on an OCR pass over page 1 of every
multi-page document: a deployment cost like R2, which the image-only production path does not pay.
THIS notebook runs no OCR: the hint arm reads the PRECOMPUTED OCR cache (`ocr_cache.zip`, SHA256SUMS
verified, as 01 and 03 do). FALLBACK: where OCR is absent the hint is absent and the prompt is the
production prompt. Whatever this A/B says about accuracy, it does not include that OCR cost.

Documents: the multi-page (meta tag `multipage`) documents of dev100 (`splits/dev100.json`) PLUS the
multi-page TRAIN documents of the 02 run's doc list (`splits/zeroshot500.json`). Train documents are
zero-shot (unseen by construction), so pooling them with dev is legitimate; the dev100 multipage
subset is small, its count is printed and its interval is wide. WHOLE documents are run in both
arms (page 1 gets no hint and its output is unaffected, but whole documents keep the merge, the
header fields and the row alignment of both arms comparable; the extra page-1 cost is in the
ESTIMATE).

Control: the **02 zero-shot run** (`CONTROL_RUN_DIR`, 500 docs, batch size 8) restricted to the
pooled doc ids; NO control rerun (GG decision). The plan cell REFUSES (raises, nothing is decoded)
unless the control's config hash and prompt hash equal those of the production config computed
from the checked-out code, and batch size, model revision, seed, logprobs and output format equal
the hint arm's, and the control's complete progress covers every pooled doc. The code SHA
legitimately differs (the pin adds the opt-in hook): both SHAs and `git diff --stat <02 SHA> <pin>
-- src configs` are printed, and the hash check with `header_hint` off is the evidence the control
stays valid. HONEST LIMIT: the hint arm runs only the pooled documents, so its batch composition
differs from the 500-doc 02 run, and byte-identical outputs across batch compositions were verified
by the 02 bench on 12 pages only; a small control-vs-arm difference from batching alone is
possible. COMPARE therefore prints a **NOISE FLOOR** first: on page 1 only (the hint never touches
page 1) how many docs' outputs differ between the arms, the share of changed cells, and the
byte-identity rate of page-1 `raw_text`.

Flow: parameters -> Drive + secrets -> clone at the pin (HEAD == PINNED_SHA asserted) -> unzip data,
assignment and OCR cache -> install -> **PLAN** (doc and page counts, strict control check) ->
**ESTIMATE** of T4 hours and compute units for the hint arm only (labelled ESTIMATE, ASSUMED; read
it before the next cell) -> **SMOKE GATE** on a few multipage dev docs (the 7 shipdoc.smoke
assertions plus hint checks; fails closed) -> hint run -> **COMPARE** with the unmodified official
scorer: NOISE FLOOR, then paired doc-level bootstrap (2000, seed 42) of OVERALL and row F1, hint -
control, for all multipage docs, dev only, train only; row_missing / row_extra / rotations /
cpn=po counts (`shipdoc.diagnostics` definitions); over-null cells; s/page -> decision -> banner.

Decision rule (`shipdoc.headerhint.decide`, a pure function with its own tests):
{hh.DECISION_RULE}.

Nothing here has run on a GPU: every time printed before the run is an ESTIMATE and UNVERIFIED.
If the session dies, open the notebook and Run all: setup repeats, the smoke gate is skipped once
passed, each arm resumes after the last document in its `trace.jsonl`. See notebooks/README.md.
"""

PARAMS = f"""# Parameters. PINNED_SHA = the repo commit this run is pinned to (full 40-char SHA).
PINNED_SHA = "{PINNED_SHA}"
CONFIG = "qwen35_4b_img_only"  # CONTROL arm: the production config (keyed, prompt v2)
HINT_CONFIG = "qwen35_4b_img_only_keyed_hdrhint"  # production config + header_hint: true
SPLIT = "train+dev"  # the pooled docs come from both splits
BATCH_SIZE = 8  # hint arm AND smoke; must equal the control's batch size (the 02 run used 8)
SMOKE_DOCS = 3  # multipage dev docs of the smoke gate (the first N of the dev subset)
# The control: the 02 zero-shot run, a folder name under runs/ (or an absolute path). The plan
# cell REFUSES to go on unless it matches (hashes, batch size, revision, seed, coverage).
CONTROL_RUN_DIR = "zeroshot500_qwen35_4b_img_only_keyed_42b812b"
CONTROL_CODE_SHA = "42b812b5b09d6e4bff0df12564017f71ffad5fc9"  # code that wrote the 02 run
USE_WANDB = False  # this notebook never logs to W&B; kept because the shared secrets cell reads it

if not PINNED_SHA or "FILL" in PINNED_SHA:
    raise ValueError("Set PINNED_SHA in the parameters cell (the pin commit fills it).")
if isinstance(BATCH_SIZE, bool) or not isinstance(BATCH_SIZE, int) or BATCH_SIZE < 1:
    raise ValueError(f"BATCH_SIZE must be an int >= 1, got {{BATCH_SIZE!r}}")
if isinstance(SMOKE_DOCS, bool) or not isinstance(SMOKE_DOCS, int) or SMOKE_DOCS < 1:
    raise ValueError(f"SMOKE_DOCS must be an int >= 1, got {{SMOKE_DOCS!r}}")
if not (isinstance(CONTROL_RUN_DIR, str) and CONTROL_RUN_DIR):
    raise ValueError("CONTROL_RUN_DIR must be the 02 run folder name (or an absolute path)")
if not (len(CONTROL_CODE_SHA) == 40 and all(c in "0123456789abcdef" for c in CONTROL_CODE_SHA)):
    raise ValueError("CONTROL_CODE_SHA must be a full 40-char lowercase hex SHA")
if USE_WANDB is not False:
    raise ValueError("USE_WANDB must stay False: this notebook has no W&B logging")
SHA7 = PINNED_SHA[:7]
AB_ID = f"hdrhint_ab_{{SHA7}}"  # MyDrive/shipdoc-extract/runs/<AB_ID>/ : plan, compare, sessions
HINT_RUN_ID = f"hdrhint_hint_{{SHA7}}"  # hint arm folder under runs/
"""

MOUNT = (
    base.MOUNT
    + """AB_DIR = RUNS_DIR / AB_ID
AB_DIR.mkdir(parents=True, exist_ok=True)
print("A/B id:", AB_ID, "| control config:", CONFIG, "| hint config:", HINT_CONFIG,
      "| batch size:", BATCH_SIZE)
print("Control = the existing 02 run (no rerun):", RUNS_DIR / CONTROL_RUN_DIR)
print("OCR cache REQUIRED: the hint arm reads precomputed OCR text (no OCR runs here).")
print("Drive output folders:", AB_DIR, "|", RUNS_DIR / HINT_RUN_ID)
"""
)

UNZIP = """import hashlib
import json
import shutil
import zipfile

LOCAL_ZIPS = Path("/content/zips")
LOCAL_ZIPS.mkdir(exist_ok=True)
CONTENT = Path("/content")
# data.zip holds data/, assignment.zip assignment/, ocr_cache.zip ocr_cache/; all unzip to /content
# (not into the clone) and the CLI is pointed at them with SHIPDOC_* variables below.
for name in ("data.zip", "assignment.zip", "ocr_cache.zip"):
    local = LOCAL_ZIPS / name
    if not local.is_file() or local.stat().st_size != (DRIVE_DIR / name).stat().st_size:
        shutil.copyfile(DRIVE_DIR / name, local)  # local disk: Drive reads are slow
    with zipfile.ZipFile(local) as zf:
        files = [n for n in zf.namelist() if not n.endswith("/")]
        if any(not (CONTENT / n).is_file() for n in files):  # also repairs a half-extracted tree
            zf.extractall(CONTENT)
    print(f"{name}: {len(files)} files under {CONTENT / files[0].split('/')[0]}")

DATA_DIR, ASSIGNMENT_DIR = CONTENT / "data", CONTENT / "assignment"
OCR_CACHE = CONTENT / "ocr_cache"
TRAIN_LABELS, DEV_LABELS = DATA_DIR / "train" / "labels", DATA_DIR / "dev" / "labels"
assert (ASSIGNMENT_DIR / "score.py").is_file(), "assignment.zip has no score.py (official scorer)"
# Verify the OCR cache against its own SHA256SUMS (paths are relative to ocr_cache/).
bad = []
sums = (OCR_CACHE / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
for line in sums:
    digest, _, rel = line.partition("  ")
    f = OCR_CACHE / rel
    if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest() != digest:
        bad.append(rel)
if bad:
    raise RuntimeError(f"ocr_cache SHA256SUMS: {len(bad)} bad/missing files, e.g. {bad[:5]}")
print(f"ocr_cache OK: {len(sums)} files verified")
"""

_inst = base.ZS_INSTALL
_INSTALL_EDITS = [
    (
        '    "SHIPDOC_OCR_CACHE": "/content/ocr_cache",  # unused by img_only; the path is '
        "never read\n",
        '    "SHIPDOC_OCR_CACHE": "/content/ocr_cache",  # READ by the hint arm (header_hint)\n',
    ),
    (  # no W&B here (USE_WANDB must stay False): the block would name undefined parameters
        "if USE_WANDB:\n"
        "    ENV.update(WANDB_API_KEY=WANDB_API_KEY, WANDB_PROJECT=WANDB_PROJECT, "
        "WANDB_RUN_GROUP=RUN_NAME)\n",
        "",
    ),
    (
        'VLM_GROUP = ["--group", "vlm"] if MODE == "run" else []  # merge mode needs no torch / model stack\n',
        'VLM_GROUP = ["--group", "vlm"]  # no train, no ocr group: the OCR text is precomputed\n',
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
print("written for a T4 (fp16, sm_75); other GPUs also run it")
"""
)

PLAN = f"""# PLAN (CPU). Doc subsets from splits + meta, doc / page counts, and the STRICT control check.
# `python -m shipdoc.headerhint plan` writes plan.json and the doc lists next to the A/B results,
# but only after the control (the 02 run) passed: config hash and prompt hash equal those of the
# production config computed from THIS checked-out code (header_hint off), batch size, model
# revision, seed, logprobs, output format equal the hint arm's, and the control's complete
# progress covers every pooled doc. Any mismatch RAISES here and nothing after this cell runs.
# The code SHA legitimately differs: both are printed with the diff of src and configs.
DECISION_RULE = {RULE_LITERAL}
_c = Path(CONTROL_RUN_DIR)
CONTROL_DIR = _c if _c.is_absolute() else RUNS_DIR / _c
plan_cmd = [PY, "-m", "shipdoc.headerhint", "plan", "--repo", str(REPO), "--out-dir", str(AB_DIR),
            "--batch-size", str(BATCH_SIZE), "--control-config", f"configs/spike_{{CONFIG}}.yaml",
            "--hint-config", f"configs/spike_{{HINT_CONFIG}}.yaml", "--control-run", str(CONTROL_DIR)]
rc, tail = run_stream(plan_cmd)
assert rc == 0, f"plan failed, CONTROL REFUSED or unreadable (exit {{rc}}): {{tail[-3:]}}"
PLAN_JSON = json.loads((AB_DIR / "plan.json").read_text())
CONTROL_SHA = json.loads((CONTROL_DIR / "manifest.json").read_text())["code_sha"]
assert CONTROL_SHA == CONTROL_CODE_SHA, f"control written by {{CONTROL_SHA}}, not the expected 02 SHA"
print(f"CODE SHAs: control (02 run) {{CONTROL_SHA}} | hint arm / this pin {{PINNED_SHA}} "
      "(differ by design)")
print(f"git diff --stat {{CONTROL_SHA[:7]}} {{SHA7}} -- src configs  (only the opt-in hook may "
      "touch the decode path: read spike.py's share):")
run_stream(["git", "-C", str(REPO), "diff", "--stat", CONTROL_SHA, PINNED_SHA, "--", "src",
            "configs"])
print("DECISION RULE:", DECISION_RULE)
print("CAVEAT: train docs are zero-shot (unseen by construction): pooling with dev is legitimate;")
N_DEV = PLAN_JSON["counts"]["dev"]["docs"]
print(f"CAVEAT: the dev100 multipage subset is small (n = {{N_DEV}} docs): its interval is wide.")
print("CAVEAT: the hint is OCR output of page 1: adopting it adds OCR (Paddle) at inference time.")
"""

ESTIMATE = """# T4-hour and compute-unit (CU) ESTIMATE BEFORE anything runs on the GPU: read it. ASSUMED: the
# spike40 speed fit (configs/spike_speed.json) at the keyed output length; the hint adds at most
# 64 prompt tokens per continuation page, assumed negligible next to the ~1,260 visual tokens;
# batching gain at 100 / 75 / 50 % scaling (scripts/gpu_estimate.py). HINT ARM ONLY: the control
# is the existing 02 run. Nothing here is measured on a GPU. No confirmation prompt (Run all must
# not block): interrupt the run yourself if it is too high.
import importlib.util
import subprocess

_ge_path = REPO / "scripts" / "gpu_estimate.py"
_spec = importlib.util.spec_from_file_location("gpu_estimate", _ge_path)
ge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ge)

SPEED = json.loads((REPO / "configs" / "spike_speed.json").read_text(encoding="utf-8"))
DEV_SUBSET = PLAN_JSON["docs"]["dev"]
SMOKE_IDS = DEV_SUBSET[:SMOKE_DOCS]


def read_gpu_name():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=False,
        ).stdout.strip().splitlines()
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out[0].strip() if out else None


GPU_NAME = read_gpu_name()
HINT_PAGES = PLAN_JSON["counts"]["pooled"]["pages"]
SMOKE_PAGES = ge.count_pages(SMOKE_IDS, DEV_LABELS)
_m = SPEED["models"][ge.base_config(CONFIG)]
N_OUT = float(_m["n_output_tokens_mean"])
R1, R2 = SPEED.get("t4_cu_per_hour"), SPEED.get("t4_cu_per_hour_conservative")
print("Runtime GPU (nvidia-smi):", GPU_NAME)
print(f"HINT ARM ONLY: {PLAN_JSON['counts']['pooled']['docs']} docs, {HINT_PAGES} pages at batch "
      f"{BATCH_SIZE}, plus the smoke gate ({len(SMOKE_IDS)} docs, {SMOKE_PAGES} pages, charged at "
      "batch 1: conservative). The control is the existing 02 run: nothing to rerun.")
print("ESTIMATE, ASSUMED, UNVERIFIED: no GPU measurement. s/page(B, eff) = prefill_s + "
      f"{N_OUT} tok x straggler(B) / (decode_tok_s x eff x B), "
      f"{SPEED.get('model_load_s', 0)} s per model load (smoke and the arm each load once), "
      "no bench (the batch is fixed), straggler "
      "factor measured on the 500-doc length mix and applied unchanged to this smaller doc set.")
print(f"{'scenario':<22}{'s/page':>8}{'T4 hours':>10}{'CU@' + str(R1):>10}{'CU@' + str(R2):>10}")
for _b, _eff in [(1, 1.0)] + [(BATCH_SIZE, e) for e in ge.EFFICIENCIES if BATCH_SIZE > 1]:
    _s = ge.batch_s_per_page(_m["prefill_s"], _m["decode_tok_s"], N_OUT, _b, _eff)
    _h = ge.tab_hours(SPEED, CONFIG, N_OUT, _b, _eff, HINT_PAGES, smoke_pages=SMOKE_PAGES)
    _name = "B=1 (reference)" if _b == 1 else f"B={_b} @ {int(_eff * 100)}% scaling"
    print(f"{_name:<22}{_s:8.1f}{_h:10.2f}{_h * R1:10.1f}{_h * R2:10.1f}")
print(f"TOTAL pages to run: {HINT_PAGES + SMOKE_PAGES} (hint {HINT_PAGES}, smoke {SMOKE_PAGES}); "
      "one resumed session pays one more model load. CU rates are not Google-published.")
"""

SMOKE = """# MANDATORY smoke gate (the hint hook is unverified on a GPU): the hint config on the first
# SMOKE_DOCS multipage dev docs, then the 7 shipdoc.smoke assertions on the output plus hint checks:
# every page carries a header_hint record, page 1 never gets a hint, no continuation page says
# `no_ocr` (that would mean the OCR cache was not found), and at least one page got the hint. A
# failure raises: nothing after this cell runs. Resumable via a status file on Drive.
import importlib.util
import sys
import time

import yaml

_smoke_path = REPO / "src" / "shipdoc" / "smoke.py"
_spec = importlib.util.spec_from_file_location("shipdoc_smoke", _smoke_path)
smoke = importlib.util.module_from_spec(_spec)
sys.modules["shipdoc_smoke"] = smoke  # dataclasses resolves annotations through sys.modules
_spec.loader.exec_module(smoke)

HINT_CFG = f"configs/spike_{HINT_CONFIG}.yaml"
SMOKE_RUNS = RUNS_DIR / "smoke"
SMOKE_RUN_ID = f"smoke_{HINT_RUN_ID}"
SMOKE_STATUS_PATH = AB_DIR / "smoke_status.json"
smoke_status = json.loads(SMOKE_STATUS_PATH.read_text()) if SMOKE_STATUS_PATH.is_file() else {}
SMOKE_ENV = {**ENV, "SHIPDOC_RUNS_DIR": str(SMOKE_RUNS)}
for _k in ("WANDB_API_KEY", "WANDB_PROJECT", "WANDB_RUN_GROUP"):
    SMOKE_ENV.pop(_k, None)
SMOKE_PASSED = False
SMOKE_LIST = AB_DIR / "docs_smoke.json"
SMOKE_LIST.write_text(json.dumps(SMOKE_IDS))


def save_smoke_status() -> None:
    tmp = SMOKE_STATUS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(smoke_status, indent=1))
    tmp.replace(SMOKE_STATUS_PATH)  # atomic: a disconnect never leaves a half-written file


def hint_checks(run_dir) -> list:
    lines = (run_dir / "trace.jsonl").read_text().splitlines()
    pages = [(i, p) for t in map(json.loads, (ln for ln in lines if ln.strip()))
             for i, p in enumerate(t["pages"])]
    recs = [(i, p.get("header_hint")) for i, p in pages]
    later = [r for i, r in recs if i > 0]
    no_ocr = sum(1 for r in later if r and r["reason"] == "no_ocr")
    hinted = sum(1 for r in later if r and r["applied"])
    return [
        smoke.Check("hint record on every page", all(r is not None for _, r in recs),
                    f"{len(recs)} pages"),
        smoke.Check("page 1 never hinted", all(not r["applied"] for i, r in recs if i == 0 and r),
                    ""),
        smoke.Check("no continuation page without OCR", bool(later) and no_ocr == 0,
                    f"{no_ocr} pages say no_ocr"),
        smoke.Check("at least one page got the hint", hinted > 0,
                    f"{hinted} of {len(later)} continuation pages"),
    ]


if smoke_status.get("state") == "passed" and smoke_status.get("run_id") == SMOKE_RUN_ID:
    SMOKE_PASSED = True
    print(f"SKIP smoke: already passed ({SMOKE_RUN_ID})")
else:
    run_dir = SMOKE_RUNS / SMOKE_RUN_ID
    t0 = time.time()
    rc, tail, checks, warns, error = 0, [], [], [], None
    try:
        max_new = int(yaml.safe_load((REPO / HINT_CFG).read_text())["max_new_tokens"])
        if not (run_dir / "metrics.json").is_file():  # a finished run is re-checked, not re-run
            print(f"=== {SMOKE_RUN_ID}: {len(SMOKE_IDS)} docs ===", flush=True)
            cmd = [PY, "-m", "shipdoc", "spike", "--config", HINT_CFG, "--docs", str(SMOKE_LIST),
                   "--split", "dev", "--run-id", SMOKE_RUN_ID, "--resume", "--logprobs",
                   "--batch-size", str(BATCH_SIZE)]
            rc, tail = run_stream(cmd, env=SMOKE_ENV)
        if rc == 0:
            checks = smoke.load_and_check(run_dir, DEV_LABELS, max_new, require_logprobs=True)
            checks += hint_checks(run_dir)
            warns = smoke.load_warnings(run_dir)  # non-blocking: never raises, never fails
    except (OSError, KeyError, ValueError) as exc:  # fail closed: could not verify = fail
        rc, error = 1, f"{type(exc).__name__}: {exc}"
    if rc != 0:
        state = "error"
        print(f"SMOKE ERROR: exit {rc} {error or ''}; last output: {tail[-3:]}")
    else:
        state = "passed" if all(c.passed for c in checks) else "failed"
        print(smoke.format_table(HINT_CONFIG, checks, warns))
    smoke_status = {
        "state": state,
        "run_id": SMOKE_RUN_ID,
        "exit_code": rc,
        "seconds": round(time.time() - t0, 1),
        "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail} for c in checks],
        "error": error,
        "tail": tail[-5:] if rc else [],
    }
    save_smoke_status()
    SMOKE_PASSED = state == "passed"

if not SMOKE_PASSED:
    raise RuntimeError(
        "SMOKE GATE FAILED: the arms are NOT run. GG: report back with "
        f"{SMOKE_STATUS_PATH} and the table above (counts only; the traces stay on Drive: "
        f"{SMOKE_RUNS / SMOKE_RUN_ID}). Do not edit thresholds to make it pass."
    )
print("SMOKE GATE PASSED: starting the arms.")
"""

ARMS = """# HINT ARM only (the control is the existing 02 run, strictly checked in the plan cell): one
# subprocess, RESUMABLE per doc (the CLI appends one line to trace.jsonl after every document and
# --resume skips the doc_ids already there). Batch size BATCH_SIZE (= the control's), greedy,
# seed 42, logprobs on. Whole documents in the pooled order; the batch composition differs from the
# 500-doc 02 run (see COMPARE's NOISE FLOOR).
import re
import time
from datetime import UTC, datetime

if not SMOKE_PASSED:
    raise RuntimeError("The smoke gate has not passed: the arms are NOT run.")
OOM_RE = re.compile(r"out of memory|OutOfMemoryError", re.IGNORECASE)
SESSIONS_PATH = AB_DIR / "sessions.json"  # wall-clock log, one entry per arm per Run all
sessions = json.loads(SESSIONS_PATH.read_text()) if SESSIONS_PATH.is_file() else []


def save_json(path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)  # atomic: a disconnect never leaves a half-written file


def run_arm(label: str, cfg_path: str, docs_file: Path, run_id: str, n_docs: int) -> None:
    run_dir = RUNS_DIR / run_id
    try:
        done = json.loads((run_dir / "progress.json").read_text()).get("done", 0)
    except (OSError, ValueError):
        done = 0
    print(f"=== {label} {run_id}: {done}/{n_docs} docs already done; resuming the rest ===")
    session = {"arm": label, "start": datetime.now(UTC).isoformat(), "end": None, "seconds": None,
               "docs_done_at_start": done, "exit_code": None}
    sessions.append(session)
    save_json(SESSIONS_PATH, sessions)
    cmd = [PY, "-m", "shipdoc", "spike", "--config", cfg_path, "--docs", str(docs_file),
           "--split", SPLIT, "--run-id", run_id, "--resume", "--logprobs",
           "--batch-size", str(BATCH_SIZE)]
    t0 = time.time()
    try:
        rc, tail = run_stream(cmd)
    finally:
        session.update(end=datetime.now(UTC).isoformat(), seconds=round(time.time() - t0, 1))
        save_json(SESSIONS_PATH, sessions)
    session["exit_code"] = rc
    save_json(SESSIONS_PATH, sessions)
    if rc != 0:
        oom = OOM_RE.search("\\n".join(tail))
        raise RuntimeError(
            f"{label} stopped (exit {rc}{', CUDA out of memory' if oom else ''}); last output: "
            f"{tail[-3:]}. Everything done so far is on Drive: Run all again resumes."
        )
    print(f"--- {label} finished (exit 0) in {(time.time() - t0) / 60:.1f} min ---")


print("CONTROL: the existing 02 run, no rerun (strict check passed in the plan cell).")
run_arm("hint", HINT_CFG, AB_DIR / "docs_pooled.json", HINT_RUN_ID,
        PLAN_JSON["counts"]["pooled"]["docs"])
"""

COMPARE = """# COMPARE (CPU). First the NOISE FLOOR (page 1 only, where the hint cannot act: docs whose output
# differs between the arms, share of changed cells, byte-identity of page-1 raw_text). Then the
# unmodified official scorer vs the gold of the multipage docs (train / dev labels only); hint -
# control (the 02 run restricted to the pooled docs), paired doc-level bootstrap (2000 resamples,
# seed 42) of OVERALL and row F1 over all multipage docs ("pooled"), dev only and train only;
# row_missing / row_extra / rotations / cpn=po row counts (shipdoc.diagnostics definitions),
# over-null cells, s/page; then the decision rule. Refuses if either arm lacks a doc. Writes
# hdrhint_compare.json / .md (aggregates only).
cmp_cmd = [PY, "-m", "shipdoc.headerhint", "compare", "--out-dir", str(AB_DIR),
           "--hint-run-dir", str(RUNS_DIR / HINT_RUN_ID)]
rc, tail = run_stream(cmp_cmd)
if rc != 0:
    raise RuntimeError(f"COMPARE FAILED (exit {rc}): {tail[-3:]}. The run folders are on Drive; "
                       "report back (counts only).")
"""

BANNER = """# Completion banner. Reads only files on Drive: also runnable alone after a resume.
import json

bar = "=" * 78
cmp_path = AB_DIR / "hdrhint_compare.json"
cmp = json.loads(cmp_path.read_text()) if cmp_path.is_file() else None
sess_path = AB_DIR / "sessions.json"
sess = json.loads(sess_path.read_text()) if sess_path.is_file() else []
print(bar)
if cmp is None:
    print(f"NOT COMPARED: {cmp_path.name} missing. Run the cells above.")
    print(f"DONE {AB_ID} compared=False")
else:
    print(f"HEADER-HINT A/B: {cmp['decision']}")
    print(bar)
    nf = cmp["noise_floor"]
    print(f"NOISE FLOOR (page 1 only, the hint cannot act): {nf['docs_page1_any_differs']} of "
          f"{nf['n_docs']} docs differ between the arms; page-1 raw_text not byte-identical in "
          f"{nf['docs_page1_raw_text_differs']} docs ({100 * nf['raw_text_diff_rate']:.2f}%); "
          f"{nf['n_cells_changed']} of {nf['n_cells']} page-1 cells changed "
          f"({100 * nf['cell_change_share']:.2f}%)")
    print("CONTROL = the 02 run (batch 8, other batch composition): read every delta against this.")
    for name, s in cmp["subsets"].items():
        o, r = s["OVERALL"], s["row_f1"]
        print(f"{name:<7}{s['n_docs']:>4} docs {s['n_pages']:>4} pages  OVERALL control "
              f"{100 * o['a']:.2f} hint {100 * o['b']:.2f} paired delta {100 * o['delta']:+.2f} "
              f"[{100 * o['lo']:+.2f}, {100 * o['hi']:+.2f}]  row F1 delta {100 * r['delta']:+.2f} "
              f"[{100 * r['lo']:+.2f}, {100 * r['hi']:+.2f}]")
    p = cmp["subsets"]["pooled"]
    print("row counts (pooled) control:", p["rows_control"], "hint:", p["rows_hint"])
    print(f"over-null cells (pooled): control {p['over_null_control']['over_null_total']}, "
          f"hint {p['over_null_hint']['over_null_total']}")
    print(f"hint pages by reason (pages >= 2): {cmp['hint_page_reasons']}; "
          f"two-line headers joined: {cmp['hint_joined_two_line_headers']}")
    print(f"s/page control {p['s_per_page_control']} hint {p['s_per_page_hint']} "
          "(control from the 02 session)")
    for clause in cmp["failed_clauses"]:
        print("FAILED CLAUSE:", clause)
    print("RULE:", cmp["rule"])
    print("CAVEAT: train docs are zero-shot, so pooling is legitimate; the dev100 multipage subset")
    print("is small (n above). ADOPT would still add OCR (Paddle) at inference time: a deployment")
    print("cost like R2 that this A/B (precomputed OCR cache) does not measure.")
    secs = sum(s_["seconds"] for s_ in sess if s_.get("seconds") is not None)
    print(f"wall-clock {secs / 3600:.2f} h over {len(sess)} arm session(s) (measured)")
    print(f"Drive folder: {AB_DIR}")
    print("GG: download these files from that Drive folder, and the two run folders:")
    for name in ("plan.json", "hdrhint_compare.json", "hdrhint_compare.md", "sessions.json"):
        path = AB_DIR / name
        size = f"{path.stat().st_size / 1e6:8.2f} MB" if path.is_file() else "MISSING"
        print(f"  {name:<24}{size}")
    print(f"  {RUNS_DIR / HINT_RUN_ID}  (predictions.json, trace.jsonl, manifest.json)")
    print("Never paste document values into chat; the markdown holds counts.")
    print(f"DONE {AB_ID} compared=True decision={cmp['decision']}")
print(bar)
"""


def _cells() -> list[tuple[str, str]]:
    return [
        ("markdown", TITLE),
        ("code", PARAMS),
        ("markdown", base.ACCOUNT_MD),
        ("code", MOUNT),
        ("code", base.ZS_SECRETS),
        ("code", base.CLONE),
        ("code", UNZIP),
        ("code", INSTALL),
        ("code", PLAN),
        ("code", ESTIMATE),
        ("code", SMOKE),
        ("code", ARMS),
        ("code", COMPARE),
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
