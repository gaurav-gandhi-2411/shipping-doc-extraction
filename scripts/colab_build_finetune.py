"""Generate notebooks/03_finetune.ipynb (LoRA fine-tune of Qwen3.5-4B; outputs always cleared).

Same pattern as ``scripts/colab_build_notebook.py`` (01_spike / 02_zeroshot500): the notebook is a
thin wrapper, every cell only orchestrates (Drive, secrets, clone, install, estimate, subprocess,
bookkeeping) and all logic is in ``python -m shipdoc.trainset`` / ``python -m shipdoc.train``. The
cells that are identical to 01_spike (account, mount, secrets, clone, unzip) are imported from that
script, so a fix there reaches this notebook on the next regeneration.

This lives in its own file, not as a function appended to colab_build_notebook.py, only because
that file carried another executor's uncommitted edits when this step was written; merging this
module into it later is mechanical.

Pin: ``FT_PINNED_SHA`` stays the placeholder until the pin commit (code commit SHA, then
regenerate). While it is a placeholder the notebook refuses to run.

Run: uv run python scripts/colab_build_finetune.py
"""

# ruff: noqa: E501  # cell sources are verbatim notebook text

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FT_OUT = ROOT / "notebooks" / "archive" / "03_finetune.ipynb"
FT_PINNED_SHA = "42b812b5b09d6e4bff0df12564017f71ffad5fc9"
STAGES = ("smoke", "fold0", "fold1", "fold2", "final")

_spec = importlib.util.spec_from_file_location(
    "colab_build_notebook", ROOT / "scripts" / "colab_build_notebook.py"
)
base = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("colab_build_notebook", base)
_spec.loader.exec_module(base)

FT_TITLE = """# 03 - LoRA fine-tune of Qwen3.5-4B (thin wrapper)

Trains a language-model-only LoRA (r=16, 200 modules, 30,474,240 trainable parameters) on
per-page keyed targets. All logic is in `python -m shipdoc.train` / `python -m shipdoc.trainset`;
this notebook does Drive, secrets, clone, install, the estimate and bookkeeping.

**Set `STAGE` in the parameters cell.** The default is `smoke`: the 20-step smoke-train, which is
the only thing to run first. Stages: `smoke`, `fold0`, `fold1`, `fold2` (cross-validation folds
of `splits/folds.json`, train on the other two), `final` (all 400 train docs).

Flow: parameters -> Drive + secrets -> clone at the pin -> unzip data -> **ESTIMATE (hours and
compute units per stage, printed before anything runs: read it)** -> install (vlm + train groups) and GPU
probe (bf16 on an L4, fp16 + GradScaler on a T4) -> prepare the training plan (CPU) -> **SMOKE
stage (always first; the notebook stops here if it fails)** -> the chosen `STAGE` (skipped for
`smoke`) -> completion banner with the loss-curve summary and the files to download.

**Order of the stages.** `smoke` -> `fold0` -> (OOF inference of fold 0's held-out pages on a T4,
outside this notebook) -> `fold1`, `fold2`, `final` **only if fold 0 shows no regression versus
zero-shot on its held-out suppliers**. The notebook does not enforce this (no code gate); it
prints a reminder in the fold0 banner and before fold1 / fold2 / final.

Resumable: checkpoints (adapter + optimizer + scheduler + RNG + step + sampler position) go to
`MyDrive/shipdoc-extract/runs/<run_id>/ckpt` every 10 steps. If the session dies, open the
notebook and Run all again: the smoke is not repeated, the stage resumes bit-exactly from the
newest checkpoint.

No early stopping on held-out loss (it would leak the fold's own held-out docs into G4); the
held-out loss (24 fixed pages, every 10 steps and at each epoch end) is logged for curves only and
never stops training or picks a checkpoint. For fold0 / fold1 / fold2 / final the curves go to the
private W&B project `shipdoc-extract-debug` by default (`USE_WANDB`); without a `WANDB_API_KEY`
secret, or if W&B is unreachable, the run simply continues without it. Every number printed here is UNVERIFIED until it
has been measured on a GPU.
"""

FT_PARAMS = """# Parameters. PINNED_SHA = the repo commit this run is pinned to (full 40-char SHA); the pin
# commit fills the placeholder (same two-commit pattern as 01_spike / 02_zeroshot500).
PINNED_SHA = "@@PIN@@"
STAGE = "smoke"  # smoke | fold0 | fold1 | fold2 | final   (default: only the 20-step smoke)
STAGES = ["smoke", "fold0", "fold1", "fold2", "final"]
CONFIG = "configs/finetune_qwen35_4b.yaml"  # every hyper-parameter, seed 42
FOLD_STAGES = ["fold0", "fold1", "fold2", "final"]
# W&B (curves only: train loss, grad norm, lr, held-out loss every 10 steps and at each epoch end;
# never images or values). ON by default for the fold0 / fold1 / fold2 / final stages, OFF for the
# smoke; override by hand. No WANDB_API_KEY secret or no network = W&B is skipped and training
# carries on: the run never depends on W&B. The held-out loss never stops training or picks a
# checkpoint (no early stopping, by design: it would leak the fold's held-out docs into G4).
USE_WANDB = STAGE in FOLD_STAGES
WANDB_PROJECT = "shipdoc-extract-debug"  # must be a PRIVATE project
# Stage order (a reminder, not a gate): fold1, fold2 and final run ONLY after fold0's OOF
# inference shows no regression versus zero-shot on its held-out suppliers.
AFTER_FOLD0 = (
    "fold1, fold2 and final run ONLY after fold0's OOF inference shows "
    "no regression vs zero-shot on its held-out suppliers."
)

if STAGE not in STAGES:
    raise ValueError(f"STAGE must be one of {STAGES}, got {STAGE!r}")
if not PINNED_SHA or "FILL" in PINNED_SHA:
    raise ValueError("Set PINNED_SHA in the parameters cell (the pin commit fills it).")
if STAGE in ("fold1", "fold2", "final"):
    print("REMINDER:", AFTER_FOLD0)
SHA7 = PINNED_SHA[:7]
RUN_BASE = f"ft_{STAGE}_{SHA7}"  # the precision is appended once the GPU is known
"""

# A missing WANDB_API_KEY must not stop a fold: the shared secrets cell raises when USE_WANDB is on.
_REQ = 'WANDB_API_KEY = get_secret("WANDB_API_KEY", required=USE_WANDB)'
assert _REQ in base.SECRETS, "colab_build_notebook.SECRETS changed: update the W&B line here"
FT_SECRETS = base.SECRETS.replace(
    _REQ, 'WANDB_API_KEY = get_secret("WANDB_API_KEY", required=False)'
)

FT_MOUNT = (
    base.MOUNT
    + """print("runs folder on Drive:", RUNS_DIR)
"""
)

FT_ESTIMATE = """# ESTIMATE of GPU hours and compute units (CU) per stage, printed BEFORE anything runs.
# No confirmation prompt (Run all must not block): interrupt the run yourself if it is too high.
# Source: scripts/finetune_estimate.py (a FLOP model over an ASSUMED effective TFLOP/s, anchored
# to the measured inference prefill rate; reports/finetune_plan.md). UNVERIFIED: the smoke
# stage below replaces it with a measured s/step. Colab exposes no CU balance programmatically.
# After the tables: the MEASURED L4 bf16 smoke (17.3 s/step of 2 pages) projected to every stage,
# and OOF inference hours on a T4 (unbatched 36 s/page; 2x / 3x are hypothetical batch speedups).
import importlib.util
import subprocess
import sys

import yaml

_spec = importlib.util.spec_from_file_location(
    "finetune_estimate", REPO / "scripts" / "finetune_estimate.py"
)
fe = importlib.util.module_from_spec(_spec)
sys.modules["finetune_estimate"] = fe  # dataclasses resolve their module through sys.modules
_spec.loader.exec_module(fe)

cfg = yaml.safe_load((REPO / CONFIG).read_text(encoding="utf-8"))
EPOCHS, ACCUM = cfg["epochs"], cfg["grad_accum"]
SMOKE_STEPS, SMOKE_ACCUM = cfg["smoke"]["steps"], cfg["smoke"]["grad_accum"]


def read_gpu():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=False,
        ).stdout.strip().splitlines()
    except (OSError, subprocess.TimeoutExpired):
        return None, None
    if not out:
        return None, None
    name, _, cap = out[0].partition(",")
    try:
        major = int(cap.strip().split(".")[0])
    except ValueError:
        major = None
    return name.strip(), major


GPU_NAME, GPU_MAJOR = read_gpu()
# Same rule as shipdoc.train.resolve_precision (the install cell asserts they agree).
PREVIEW_PRECISION = "fp32" if GPU_MAJOR is None else ("bf16" if GPU_MAJOR >= 8 else "fp16")
KIND = next((k for k in ("L4", "T4") if GPU_NAME and k in GPU_NAME.split()), None)
pages = fe.fold_pages(REPO)
STAGE_PAGES = {
    "smoke": SMOKE_STEPS * SMOKE_ACCUM,
    "fold0": EPOCHS * pages["train"][0],
    "fold1": EPOCHS * pages["train"][1],
    "fold2": EPOCHS * pages["train"][2],
    "final": EPOCHS * pages["all_train"],
}
print(f"Runtime GPU (nvidia-smi): {GPU_NAME} | "
      f"precision this run will use: {PREVIEW_PRECISION}")
print(f"STAGE = {STAGE!r}; {EPOCHS} epochs, effective batch {ACCUM} pages "
      f"(visits = pages x epochs); the smoke stage "
      f"({SMOKE_STEPS} steps x {SMOKE_ACCUM} pages) always runs first.")
if KIND is None:
    print("WARNING: the runtime GPU is not a T4 or an L4: Runtime -> Change runtime type. "
          "Both figures are shown; neither applies to this GPU.")
smoke_central = {}
for kind in [KIND] if KIND else ["T4", "L4"]:
    rates = " / ".join(f"{r:.2f}" for r in dict.fromkeys(fe.CU_PER_HOUR[kind]))
    print(f"\\n--- {kind}: training hours (slow / central / fast scenario) and CU (central hours at "
          f"{rates} CU/h) ---")
    print(f"{'stage':<7}{'visits':>7}   {'hours':<24}{'CU (central)':<16}")
    for stage in STAGES:
        hours, cus = [], None
        for label in ("low", "central", "high"):
            tp = next(t for t in fe.scenarios() if t.gpu == kind and t.label == label)
            sec = fe.MODEL_LOAD_S + STAGE_PAGES[stage] * fe.seconds_per_page(
                tp, fe.INPUT_TOKENS, fe.TARGET_MEAN)
            h, lo, hi = fe.hours_cu(sec, kind)
            hours.append(h)
            if label == "central":
                cus = (lo, hi)
        if stage == "smoke":
            smoke_central[kind] = hours[1]
        cu = f"{cus[0]:.1f}" if cus[0] == cus[1] else f"{cus[0]:.1f}-{cus[1]:.1f}"
        mark = "   <-- this run" if stage == STAGE else ""
        print(f"{stage:<7}{STAGE_PAGES[stage]:>7}   "
              f"{hours[0]:5.2f} / {hours[1]:5.2f} / {hours[2]:5.2f}   {cu:<16}{mark}")
print("\\nCU rates: T4 1.19-1.58 (third-party), L4 3.00 (varlog.info, 2024-10-01); no first-party "
      "source. Colab shows the real balance under Runtime -> Manage sessions / Resources.")
print()
for line in fe.measured_lines(pages, EPOCHS, ACCUM, cfg["eval_every"], cfg["eval_pages"]):
    print(line)
print("\\nOOF and dev INFERENCE (T4 fp16) is not part of this notebook. " + AFTER_FOLD0)
"""

FT_INSTALL = '''import collections
import json
import subprocess
import time
from datetime import UTC, datetime

PY = str(REPO / ".venv" / "bin" / "python")
ENV = {
    **os.environ,
    "SHIPDOC_PROFILE": "colab",
    "HF_HOME": "/content/hf_cache",
    "SHIPDOC_RUNS_DIR": str(RUNS_DIR),
    "SHIPDOC_DATA_DIR": str(REPO / "data"),
    "SHIPDOC_ASSIGNMENT_DIR": str(REPO / "assignment"),
    "SHIPDOC_OCR_CACHE": "/content/ocr_cache",
    "SHIPDOC_TMP_DIR": "/content/tmp",
    "UV_CACHE_DIR": "/content/uv_cache",
    "PYTORCH_ALLOC_CONF": "expandable_segments:True",
}
if USE_WANDB and WANDB_API_KEY:
    ENV.update(SHIPDOC_WANDB="1", WANDB_API_KEY=WANDB_API_KEY, WANDB_PROJECT=WANDB_PROJECT)
    print(f"W&B: ON, private project {WANDB_PROJECT} (curves only; never stops the run)")
elif USE_WANDB:
    print("W&B: OFF - USE_WANDB is on but the Colab secret WANDB_API_KEY is not set. Training is "
          "unaffected; add the secret and re-run to get the curves.")
else:
    print("W&B: OFF (USE_WANDB is False for this stage)")
if HF_TOKEN:
    ENV["HF_TOKEN"] = HF_TOKEN
Path(ENV["SHIPDOC_TMP_DIR"]).mkdir(exist_ok=True)


def run_stream(
    cmd: list[str], tail: int = 40, env: dict | None = None
) -> tuple[int, list[str]]:
    """Run a command, echo its output live, return (exit code, last `tail` lines)."""
    last: collections.deque[str] = collections.deque(maxlen=tail)
    with subprocess.Popen(
        cmd, cwd=REPO, env=env or ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    ) as proc:
        for line in proc.stdout:
            print(line, end="")
            last.append(line.rstrip("\\n"))
    return proc.returncode, list(last)


def save_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)  # atomic: a disconnect never leaves a half-written file


def run_session(cmd: list[str], run_dir: Path) -> tuple[int, list[str]]:
    """run_stream + one wall-clock entry per call in <run_dir>/sessions.json."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "sessions.json"
    sessions = json.loads(path.read_text()) if path.is_file() else []
    session = {"start": datetime.now(UTC).isoformat(), "seconds": None, "exit_code": None}
    sessions.append(session)
    save_json(path, sessions)
    t0 = time.time()
    try:
        rc, tail = run_stream(cmd)
    finally:
        session["seconds"] = round(time.time() - t0, 1)
        save_json(path, sessions)
    session["exit_code"] = rc
    save_json(path, sessions)
    return rc, tail


if subprocess.run(["uv", "--version"], capture_output=True, check=False).returncode:
    subprocess.run(["pip", "install", "-q", "uv"], check=True)
# peft==0.21.2 comes from the locked `train` dependency group (uv.lock), like everything else.
rc, _ = run_stream(
    ["uv", "sync", "--frozen", "--group", "vlm", "--group", "train", "--python", "3.11"]
)
assert rc == 0, f"uv sync failed (exit {rc})"

subprocess.run(["nvidia-smi"], check=False)
PROBE = (
    "import json, importlib.metadata as m, torch;"
    "from shipdoc.train import resolve_precision;"
    "p = torch.cuda.get_device_properties(0);"
    "print(json.dumps({'torch': torch.__version__, 'cuda': torch.cuda.is_available(),"
    " 'gpu': torch.cuda.get_device_name(0), 'major': p.major,"
    " 'vram_gib': round(p.total_memory / 2**30, 1),"
    " 'precision': resolve_precision('auto', p.major),"
    " **{k: m.version(k) for k in ('transformers', 'peft', 'accelerate')}}))"
)
info = json.loads(subprocess.run([PY, "-c", PROBE], cwd=REPO, env=ENV, capture_output=True,
                                 text=True, check=True).stdout.strip().splitlines()[-1])
print(info)
assert info["cuda"], "No CUDA device: Runtime -> Change runtime type -> L4 (recommended) or T4."
PRECISION = info["precision"]
assert PRECISION == PREVIEW_PRECISION, (
    f"estimate cell predicted {PREVIEW_PRECISION}, torch says {PRECISION} (GPU {info['gpu']})")
print(f"GPU {info['gpu']} ({info['vram_gib']} GiB): training precision = {PRECISION} "
      f"({'bf16, no scaler' if PRECISION == 'bf16' else 'fp16 + GradScaler + fp32 LoRA weights'})")
RUN_ID = f"{RUN_BASE}_{PRECISION}"
SMOKE_RUN_ID = f"ft_smoke_{SHA7}_{PRECISION}"
RUN_DIR, SMOKE_DIR = RUNS_DIR / RUN_ID, RUNS_DIR / SMOKE_RUN_ID
print("run_id:", RUN_ID, "| smoke run_id:", SMOKE_RUN_ID)
print("Drive output folder:", RUN_DIR if STAGE != "smoke" else SMOKE_DIR)
'''

FT_PREPARE = """# Training plan for all 500 train+dev docs (CPU, a few minutes, cached on Drive): the page of
# every gold row from the OCR locator, the header-only / excluded documents, and the occlusion
# candidates. Counts only are printed; the audit log (doc id, row, page, decision; no values)
# is written next to the cache. Both fold and final stages read this one file.
PREPARED = RUNS_DIR / "prepared_all.json"
if PREPARED.is_file():
    print(f"SKIP prepare: {PREPARED} exists ({PREPARED.stat().st_size / 1e6:.1f} MB)")
else:
    rc, _ = run_stream([PY, "-m", "shipdoc.trainset", "--stage", "final", "--all",
                        "--out", str(PREPARED)])
    assert rc == 0, f"trainset preparation failed (exit {rc})"
    print("audit log:", PREPARED.parent / "audit.jsonl")
"""

FT_SMOKE = """# MANDATORY smoke stage (the training path is unverified on a GPU): preflight the longest page,
# then 20 optimizer steps (accum 2), then check (i) the loss decreased (mean of the last 5 steps
# below 0.95 x the first 5), (ii) peak VRAM under the configured budget, (iii) 200 LoRA modules /
# 30,474,240 trainable parameters, (iv) no NaN / inf, plus a checkpoint write+reload on Drive.
# Any failure raises HERE: nothing after this cell runs. A passed smoke is not repeated.
SMOKE_STATUS = SMOKE_DIR / "smoke_status.json"


def read_status(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


status = read_status(SMOKE_STATUS)
if status.get("state") == "passed":
    SMOKE_PASSED = True
    print(f"SKIP smoke: already passed ({SMOKE_RUN_ID})")
else:
    rc, tail = run_session(
        [PY, "-m", "shipdoc.train", "--config", CONFIG, "--stage", "smoke",
         "--run-dir", str(SMOKE_DIR)], SMOKE_DIR)
    status = read_status(SMOKE_STATUS)
    SMOKE_PASSED = rc == 0 and status.get("state") == "passed"
    for c in status.get("checks", []):
        print(f"{c['status'].upper():<5} {c['name']:<20} {c['detail']}")
if not SMOKE_PASSED:
    raise RuntimeError(
        "SMOKE GATE FAILED: the stage is NOT started. GG: report back with "
        f"{SMOKE_STATUS} and the per-check table printed above (no document values in them). "
        "Do not edit thresholds to make it pass."
    )
print(f"SMOKE PASSED ({status.get('precision')}, peak {status.get('peak_vram_gib', 0):.1f} GiB)."
      " Measured s/step is in metrics.jsonl: rerun scripts/finetune_estimate.py with it.")
"""

FT_TRAIN = """# The chosen stage, RESUMABLE: checkpoints every 10 steps go to <run>/ckpt on Drive and Run all
# again continues from the newest one (bit-exact resume). STAGE = "smoke" stops after the smoke.
import re

if not SMOKE_PASSED:
    raise RuntimeError("The smoke gate has not passed: the stage is NOT started.")
OOM_RE = re.compile(r"out of memory|OutOfMemoryError", re.IGNORECASE)
if STAGE in ("fold1", "fold2", "final"):
    print("REMINDER:", AFTER_FOLD0)
if STAGE == "smoke":
    print('STAGE = "smoke": nothing more to run. To train, set STAGE to fold0 (first) in the '
          'parameters cell and Run all (the smoke is skipped, it already passed). ' + AFTER_FOLD0)
else:
    train_status = read_status(RUN_DIR / "train_status.json")
    if train_status.get("state") == "complete":
        print(f"SKIP {STAGE}: already complete ({RUN_DIR})")
    else:
        print(f"=== {RUN_ID}: training (resumes from the newest checkpoint if any) ===")
        rc, tail = run_session(
            [PY, "-m", "shipdoc.train", "--config", CONFIG, "--stage", STAGE,
             "--run-dir", str(RUN_DIR)], RUN_DIR)
        if rc != 0:
            oom = OOM_RE.search("\\n".join(tail))
            raise RuntimeError(
                f"{STAGE} stopped (exit {rc}{', CUDA out of memory' if oom else ''}); last "
                f"output: {tail[-3:]}. Everything checkpointed so far is on Drive: Run all again "
                "resumes. If it stops again at the same step, report back."
            )
"""

FT_BANNER = """# Completion banner. Reads only files on Drive: also runnable alone after a resume.
import importlib.util
import sys

# metrics.jsonl is read through shipdoc.metricslog (stdlib only): one line per optimizer step,
# steps 1..n once each. A file from a build before the fix can hold a dead session's lines too
# (the first smoke printed 29 / 20); the reader then keeps the last complete run and says so.
_spec = importlib.util.spec_from_file_location(
    "metricslog", REPO / "src" / "shipdoc" / "metricslog.py"
)
ml = importlib.util.module_from_spec(_spec)
sys.modules["metricslog"] = ml  # dataclasses resolve their module through sys.modules
_spec.loader.exec_module(ml)
DIR = SMOKE_DIR if STAGE == "smoke" else RUN_DIR
metrics_path = DIR / "metrics.jsonl"
STATUS_NAME = "smoke_status.json" if STAGE == "smoke" else "train_status.json"
status = read_status(DIR / STATUS_NAME)
try:
    metrics = ml.read_metrics(metrics_path, status.get("total_steps"), repair=True)
except ml.MetricsAnomaly as exc:  # e.g. a gap: show nothing rather than a wrong curve
    print("WARNING: metrics.jsonl unreadable:", exc)
    metrics = ml.MetricsRead([], 0, [str(exc)])
rows = metrics.rows
SESSIONS = DIR / "sessions.json"
sessions = json.loads(SESSIONS.read_text()) if SESSIONS.is_file() else []
secs = sum(s["seconds"] for s in sessions if s.get("seconds") is not None)
unfinished = sum(s.get("seconds") is None for s in sessions)
losses = [r["loss"] for r in rows]
k = 5
bar = "=" * 78
print(bar)
print(f"FINE-TUNE {STAGE.upper()} {status.get('state', 'unknown').upper()}: {DIR.name}")
print(bar)
print(f"GPU / precision : {info['gpu']} / {PRECISION}")
print(f"optimizer steps : {len(rows)} / {status.get('total_steps', '?')}"
      f"  ({status.get('n_train_pages', '?')} pages in the stage's pool)")
if metrics.repaired:
    print(f"WARNING         : metrics.jsonl had {metrics.n_lines} lines for {len(rows)} steps "
          f"({'; '.join(metrics.problems)}); showing the last complete run of steps")
if len(losses) >= 2 * k:
    first, last = sum(losses[:k]) / k, sum(losses[-k:]) / k
    print(f"loss curve      : first {k} steps {first:.4f} -> last {k} steps {last:.4f} "
          f"({100 * (last - first) / first:+.1f}%), min {min(losses):.4f}")
elif losses:
    print(f"loss curve      : {len(losses)} steps, first {losses[0]:.4f}, last {losses[-1]:.4f}")
evals = [(r["step"], r["eval_loss"]) for r in rows if r.get("eval_loss") is not None]
if evals:
    print("held-out loss   : " + ", ".join(f"step {s}: {v:.4f}" for s, v in evals)
          + "  (information only; no early stopping)")
ep_evals = [(r["epoch_done"], r["eval_loss_epoch"]) for r in rows if "eval_loss_epoch" in r]
if ep_evals:
    print("held-out / epoch: " + ", ".join(f"epoch {e}: {v:.4f}" for e, v in ep_evals)
          + "  (24 fixed held-out pages; curves only, never used to stop or to pick a checkpoint)")
if rows:
    print(f"s/step (mean)   : {sum(r['seconds'] for r in rows) / len(rows):.1f}   "
          f"scaler skips: {sum(bool(r.get('scaler_skipped')) for r in rows)}")
print(f"peak VRAM       : {status.get('peak_vram_gib', 0):.2f} GiB")
print(f"wall-clock      : {secs / 3600:.2f} h over {len(sessions) - unfinished} finished "
      f"session(s)" + (f" (+ {unfinished} without a recorded end: killed or disconnected, time "
                       "not counted)" if unfinished else ""))
print(f"Drive folder    : {DIR}")
print("GG: download these files from that folder:")
names = ["smoke_status.json", "metrics.jsonl", "sessions.json"] if STAGE == "smoke" else [
    "final/adapter.pt", "final/manifest.json", "train_status.json", "metrics.jsonl",
    "sessions.json"]
for name in names:
    path = DIR / name
    size = f"{path.stat().st_size / 1e6:8.2f} MB" if path.is_file() else "MISSING"
    print(f"  {name:<20}{size}")
if STAGE != "smoke":
    peft_dir = DIR / "final" / "peft"
    for path in sorted(peft_dir.glob("*")) if peft_dir.is_dir() else []:
        print(f"  final/peft/{path.name:<14}{path.stat().st_size / 1e6:8.2f} MB")
print(f"also useful (not needed): {RUNS_DIR / 'audit.jsonl'}")
if STAGE == "fold0":
    print(bar)
    print("NEXT (fold0): the adapter folder is " + str(DIR / "final"))
    print("(adapter.pt, manifest.json, peft/). Run the OOF inference of fold 0's held-out pages")
    print("(230) on a T4 and compare it")
    print("with zero-shot on the held-out suppliers BEFORE starting anything else.")
    print("REMINDER:", AFTER_FOLD0)
    print(bar)
print(f"DONE {RUN_ID if STAGE != 'smoke' else SMOKE_RUN_ID} state={status.get('state')}")
"""


def build_finetune() -> dict:
    """03_finetune notebook dict (nbformat 4.5) with all outputs cleared."""
    cells = [
        ("markdown", FT_TITLE),
        ("code", FT_PARAMS.replace("@@PIN@@", FT_PINNED_SHA)),
        ("markdown", base.ACCOUNT_MD),
        ("code", FT_MOUNT),
        ("code", FT_SECRETS),
        ("code", base.CLONE),
        ("code", base.UNZIP),
        ("code", FT_ESTIMATE),
        ("code", FT_INSTALL),
        ("code", FT_PREPARE),
        ("code", FT_SMOKE),
        ("code", FT_TRAIN),
        ("code", FT_BANNER),
    ]
    nb = base._notebook(cells)
    nb["metadata"]["colab"]["gpuType"] = "L4"  # recommended; a T4 also works (fp16 fallback)
    return nb


def render_finetune() -> str:
    """Canonical JSON text of the 03_finetune notebook."""
    return json.dumps(build_finetune(), indent=1, ensure_ascii=False) + "\n"


if __name__ == "__main__":
    FT_OUT.parent.mkdir(exist_ok=True)
    FT_OUT.write_text(render_finetune(), encoding="utf-8", newline="\n")
    print(f"wrote {FT_OUT}")
