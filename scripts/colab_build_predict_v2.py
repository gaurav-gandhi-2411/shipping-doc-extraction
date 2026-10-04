"""Generate notebooks/04c_predict_test_v2.ipynb (production pipeline v2, fine-tuned) from the cells below.

04c = 04's inference path with the FINAL fine-tuned adapter (verified, merged into the fp16 weights,
guarded at the batch size) + the OCR text of the 280 test pages (reused from 04b when cached) + the
post-rules R1 / R2 / R3 + calibration review flags in a separate file. A thin wrapper like 04 / 04b /
05: every cell orchestrates (Drive, secrets, clone, one subprocess per stage); the logic is in
``python -m shipdoc.predict_ft`` (src/shipdoc/predict_ft.py), ``python -m shipdoc.flags``,
``python -m shipdoc.reuse``, ``python -m shipdoc.ocr_stage`` and ``python -m shipdoc predict``.
Cells and helpers that do not change are imported READ-ONLY from scripts/colab_build_notebook.py,
scripts/colab_build_predict.py (04), scripts/colab_build_predict_v1.py (04b) and
scripts/colab_build_oof.py (05) and edited by exact-string replacement (asserted), so the notebooks
cannot drift apart silently; none of them is modified.

tests/test_notebook_predict_v2.py checks the committed notebook equals this script's output.

Run: uv run python scripts/colab_build_predict_v2.py
"""

# ruff: noqa: E501  # cell sources are verbatim notebook text

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "archive" / "04c_predict_test_v2.ipynb"


def _load(name: str, path: Path):  # noqa: ANN202 - a module object
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


base = _load("colab_build_notebook", ROOT / "scripts" / "colab_build_notebook.py")
p04 = _load("colab_build_predict", ROOT / "scripts" / "colab_build_predict.py")
v1 = _load("colab_build_predict_v1", ROOT / "scripts" / "colab_build_predict_v1.py")
oofb = _load("colab_build_oof", ROOT / "scripts" / "colab_build_oof.py")

# GG pins 04c only when the final-system rule picks FT + rules (spec section 11) and the `final`
# adapter exists; the pin commit replaces this placeholder with the full 40-char SHA and regenerates
# the notebook. Until then the parameters cell RAISES.
PINNED_SHA = "SUPERSEDED-NOT-PINNED"

TITLE = """# 04c - production pipeline v2: FINE-TUNED model + OCR + post-rules R1/R2/R3 + review flags (thin wrapper)

**SUPERSEDED (2026-10-03) by `04c_predict_test_native` (`MODEL = "ft"`, native resolution, `max_pixels` 2,196,480). Do not run this notebook: it is the 1260-token path and it is not pinned (`PINNED_SHA` is `SUPERSEDED-NOT-PINNED`; the first cell raises).**

**Run this only if the closing plan picked FT + rules** (spec section 11, item 1: on the 3-fold
out-of-fold comparison FT + rules beats ZS + rules with the paired-delta CI lower bound above 0 and
no more false fills / over-nulls). Otherwise the final system is the 04b v1 submission (ZS + rules).

v2 = the `final` LoRA adapter of notebook 03 (trained on the 400 `train_*` documents only; never on a
dev or test document) merged into the fp16 Qwen3.5-4B weights, then the 04 inference config
unchanged (IMAGE ONLY, KEYED output, prompt v2, greedy, seed 42, logprobs on) on the 200 test
documents / 280 pages + the OCR text of those pages (PaddleOCR, own venv; the cache that 04b filled
on Drive is reused when it is complete and valid) + R1 (carrier from the model's own supplier slot),
R2 (waybill pattern backfill from OCR), R3 (invoice slot-shape swap, frozen `meta/slot_shapes.json`),
all three ON, exactly as in 04b. Calibration review flags go to a SEPARATE file `review_flags.json`
(never into `test_predictions.json`: its schema has `additionalProperties: false`). Test images
only: there are no test labels, nothing here scores anything, and no document value is printed
(counts, ids, hashes).

Flow: setup (Drive, clone at the pin, unzip data incl. test images; 200 docs / 280 pages by file
name) -> install (`uv sync --frozen --group vlm --group train`, no Paddle) -> ESTIMATE (T4 hours /
CU, UNVERIFIED) -> plan -> OCR (adopt the 04b record when `REUSE_OCR_FROM` is set, then the OCR stage
of 04b: resumable per page, 280/280 asserted, 5-page seeded re-OCR) -> ZERO-SHOT TRACES CHECK (the
v0 traces under the strict reuse rules of 04b; the agreement features of the flags need them; a
refusal STOPS the notebook: no zero-shot inference runs here) -> **FINAL ADAPTER VERIFICATION
(fails closed; printed table)** -> **INFER** in one process: verify again, merge the adapter into the
fp16 weights, **guard** (the 12 bench pages at batch 1 and at the chosen size on the MERGED model
must be byte-identical, else batch 1, recorded), smoke gate of 04 on 5 dev documents on the merged
model, then the resumable test run -> **DETERMINISM** pass in a fresh process (merges again) ->
assemble with R1-R3 + validation (the blocking checks of 04 and 04b + the v2 checks) -> **REVIEW
FLAGS** -> banner VALIDATED / REJECTED.

What the verification refuses: a smoke / fold adapter (stage other than `final`); a fold id; a
missing, dirty or unreachable training code SHA (a SHA different from this pin is a warning); a
base-model repo / revision other than the production config's; LoRA bookkeeping other than r=16 /
200 modules / 30,474,240 trainable parameters; a manifest without its training document ids; a
training set that is not exactly the `train_*` documents of `splits/folds.json` (a dev or test id in
it is a refusal of its own); held-out ids that are not the dev documents; a `manifest_hash` that
does not match the ids; inference keys (model repo and revision, adapter name, max_pixels, prompt
version, output format) that differ from the production config; weight files whose sha256 differs
from the manifest.

Merge precision: the adapter's low-rank update is added to the fp16 base weights (fp32 sum rounded
to fp16); updates below the fp16 rounding step of a weight are lost, so the merged model is not
bit-identical to base + unmerged adapter. It is the numeric path production inference on a T4 uses
and the one 05 measured OOF on. The merge and the batch-8 byte identity on the merged model are
UNVERIFIED on a GPU until this notebook has run.

Review flags (`python -m shipdoc.flags`, calibrator `meta/calibrator_ft.json` = the frozen
`scripts/freeze_calibrator.py --arm ft` artifact, in the repo at the pin): per document P(fully
correct) + auto-accept, per emitted field P(correct) + accept / review with the per-field-type tau
frozen from the pooled out-of-fold predictions (nested estimates stored beside them). The feature
builder at inference is the same code as at calibration (a parity probe is checked at load); the
fine-tuned calibrator also needs the zero-shot post-rule output of the same test documents, taken
from the v0 traces.

Parameters: `PINNED_SHA`; `BATCH_SIZE` (8 = the 02 / 04 production size; the guard decides on the
merged model); `ZS_BATCH_SIZE` (8: the size the v0 traces were decoded at, a strict reuse
condition); `TRAIN_SHA7` and `ADAPTER_DIR` (the Drive folder `<final run>/final` of notebook 03;
default pattern `runs/ft_final_<TRAIN_SHA7>_bf16/final`, set TRAIN_SHA7 or ADAPTER_DIR by hand when
the training run was pinned at another commit or ran in fp16); `REUSE_V0_DIR` (the 04 folder
`submissions/v0_42b812b`, required); `REUSE_ACK_SPIKE_DIFF` (as 04b); `CALIBRATOR_FILE`;
`FIELD_TARGET` / `DOC_TARGET` (precision targets of the flags, 0.98); `REUSE_OCR_FROM` (a finished
04b folder whose OCR timing and re-check record are adopted, or None).

Output: `MyDrive/shipdoc-extract/submissions/v2_<sha7>/` with `test_predictions.json`,
`review_flags.json`, `trace.jsonl`, `manifest.json`, `validation_report.json`, `rules.jsonl`,
`ocr_timing.json`; on a failed check the predictions are `test_predictions.REJECTED.json`. Download
by hand into `$SHIPDOC_SUBMISSIONS_DIR\\v2_<sha7>\\`. If the session dies, Run all: every stage
resumes (the test run after the last document in `trace.jsonl`; the batch decision stands).
See notebooks/README.md.
"""

PARAMS = f"""# Parameters. PINNED_SHA = the repo commit this run is pinned to (full 40-char SHA).
PINNED_SHA = "{PINNED_SHA}"
RUN_NAME = "testft"
CONFIG = "qwen35_4b_img_only"  # KEYED output format, prompt v2, image only: the production pick
SPLIT = "test"
EXPECTED_DOCS = 200
EXPECTED_PAGES = 280
SMOKE_DOCS = "splits/smoke5.json"  # 5 dev docs picked by meta tags (shipdoc.smoke)
BENCH_DOCS = "splits/bench12.json"  # 12 dev pages for the batch guard (shipdoc.bench)
BATCH_SIZE = 8  # the 02 / 04 production size; the guard on the MERGED model decides (else 1)
ZS_BATCH_SIZE = 8  # the size the v0 traces were decoded at (a strict reuse condition)
SHARD = "0/1"  # fixed: one tab (the plan stage and the assemble step take it)
TRAIN_SHA7 = PINNED_SHA[:7]  # sha7 of the 03 run that trained the final adapter; set by hand if not
# The final run's final/ folder: relative to MyDrive/shipdoc-extract, or an absolute path.
ADAPTER_DIR = f"runs/ft_final_{{TRAIN_SHA7}}_bf16/final"
REUSE_V0_DIR = "submissions/v0_42b812b"  # the 04 folder: the zero-shot traces (flags)
REUSE_ACK_SPIKE_DIFF = False  # True accepts a spike.py diff beyond the allowlisted opt-in ones
CALIBRATOR_FILE = "meta/calibrator_ft.json"  # in the repo at the pin (scripts/freeze_calibrator.py)
FIELD_TARGET = 0.98  # precision target of the per-field-type tau (0.95 | 0.98 | 0.99)
DOC_TARGET = 0.98  # precision target of the document auto-accept tau (0.95 | 0.98)
REUSE_OCR_FROM = None  # e.g. "submissions/v1_<sha7>": adopt that 04b folder's OCR timing / re-check

if not PINNED_SHA or "FILL" in PINNED_SHA or "SUPERSEDED" in PINNED_SHA:
    raise ValueError("PINNED_SHA is not set: this archived notebook is superseded, not runnable.")
for _name, _val in (("BATCH_SIZE", BATCH_SIZE), ("ZS_BATCH_SIZE", ZS_BATCH_SIZE)):
    if isinstance(_val, bool) or not isinstance(_val, int) or _val < 1:
        raise ValueError(f"{{_name}} must be an int >= 1, got {{_val!r}}")
if not (isinstance(TRAIN_SHA7, str) and len(TRAIN_SHA7) == 7
        and all(c in "0123456789abcdef" for c in TRAIN_SHA7)):
    raise ValueError(f"TRAIN_SHA7 must be 7 lowercase hex characters, got {{TRAIN_SHA7!r}}")
if not isinstance(REUSE_V0_DIR, str) or not REUSE_V0_DIR:
    raise ValueError("REUSE_V0_DIR must be the 04 folder: the flags need the zero-shot traces")
if not isinstance(REUSE_ACK_SPIKE_DIFF, bool):
    raise ValueError(f"REUSE_ACK_SPIKE_DIFF must be True or False, got {{REUSE_ACK_SPIKE_DIFF!r}}")
if FIELD_TARGET not in (0.95, 0.98, 0.99):
    raise ValueError(f"FIELD_TARGET must be 0.95, 0.98 or 0.99, got {{FIELD_TARGET!r}}")
if DOC_TARGET not in (0.95, 0.98):
    raise ValueError(f"DOC_TARGET must be 0.95 or 0.98, got {{DOC_TARGET!r}}")
if REUSE_OCR_FROM is not None and not isinstance(REUSE_OCR_FROM, str):
    raise ValueError(f"REUSE_OCR_FROM must be None or a path string, got {{REUSE_OCR_FROM!r}}")
SHA7 = PINNED_SHA[:7]
RUN_ID = f"{{RUN_NAME}}_{{CONFIG}}_keyed_{{SHA7}}"  # MyDrive/shipdoc-extract/runs/<RUN_ID>/
"""

_ZIPS_OLD = 'REQUIRED_ZIPS = ["data.zip", "assignment.zip", "ocr_cache.zip"]'
assert _ZIPS_OLD in base.MOUNT
# Image only: no OCR cache zip. data.zip holds data/ (incl. the test images and the dev labels the
# smoke gate and the guard use), assignment.zip the schema and the sample submission.
MOUNT = (
    base.MOUNT.replace(_ZIPS_OLD, 'REQUIRED_ZIPS = ["data.zip", "assignment.zip"]')
    + """SUBMISSIONS_DIR = DRIVE_DIR / "submissions"  # test predictions live here, not in the repo
SUBMISSION_DIR = SUBMISSIONS_DIR / f"v2_{SHA7}"
META_DIR = RUNS_DIR / f"{RUN_ID}_meta"  # plan, smoke status, batch decision, OCR records
RUN_DIR = RUNS_DIR / RUN_ID
OCR_ROOT = DRIVE_DIR / "ocr_cache_test"  # <root>/paddleocr/test/<page>.json, shared with 04b
for _d in (SUBMISSIONS_DIR, META_DIR, SUBMISSION_DIR, OCR_ROOT):
    _d.mkdir(parents=True, exist_ok=True)
PLAN_PATH = META_DIR / "plan.json"
TEST_DOCS_PATH = META_DIR / "test_docs.json"  # ids only
SMOKE_STATUS_PATH = META_DIR / "smoke_status.json"
DECISION_PATH = META_DIR / "batch_decision.json"


def _under_drive(p: str) -> Path:
    \"\"\"Absolute as given; else relative to MyDrive/shipdoc-extract ('runs/...').\"\"\"
    return Path(p) if Path(p).is_absolute() else DRIVE_DIR / p


ADAPTER = _under_drive(ADAPTER_DIR)
V0_DIR = _under_drive(REUSE_V0_DIR)
OCR_FROM = _under_drive(REUSE_OCR_FROM) if REUSE_OCR_FROM else None
for label, path, need in (("final adapter folder", ADAPTER, "manifest.json"),
                          ("v0 (04) folder", V0_DIR, "trace.jsonl")):
    if not (path / need).is_file():
        raise FileNotFoundError(f"{label} {path} has no {need}: set ADAPTER_DIR / TRAIN_SHA7 / "
                                "REUSE_V0_DIR in the parameters cell.")
print("run_id:", RUN_ID, "| batch size (requested):", BATCH_SIZE)
print("final adapter   :", ADAPTER)
print("v0 (zs traces)  :", V0_DIR)
print("OCR cache folder:", OCR_ROOT, "| adopt OCR record from:", OCR_FROM)
print("Drive run folder:", RUN_DIR)
print("Drive submission folder:", SUBMISSION_DIR)
"""
)

# Base ENV / run_stream / uv sync of 05 (vlm + train groups: peft comes from the lock), edited by
# exact-string replacement: the test OCR cache is the one R2 reads, submissions go to Drive, no W&B,
# and Paddle must never be in this environment (it is its own venv, /content/ocr-venv).
_inst = oofb.INSTALL
_INSTALL_EDITS = [
    (
        '    "SHIPDOC_OCR_CACHE": "/content/ocr_cache",  # unused by img_only; never read\n',
        '    "SHIPDOC_OCR_CACHE": str(OCR_ROOT),  # the test OCR cache that R2 reads\n'
        '    "SHIPDOC_SUBMISSIONS_DIR": str(SUBMISSIONS_DIR),\n',
    ),
    (
        "if USE_WANDB:\n    ENV.update(WANDB_API_KEY=WANDB_API_KEY, WANDB_PROJECT=WANDB_PROJECT, "
        "WANDB_RUN_GROUP=RUN_NAME)\n",
        "",
    ),
    ("('transformers', 'xgrammar', 'peft')", "('transformers', 'xgrammar', 'peft', 'pycountry')"),
]
for _old, _new in _INSTALL_EDITS:
    assert _old in _inst, f"colab_build_oof.INSTALL changed: {_old[:50]!r}"
    _inst = _inst.replace(_old, _new)
INSTALL = _inst + v1.NO_PADDLE

_OCR_REUSE_OLD = (
    'print(f"REUSE_V0_DIR = {REUSE_V0_DIR!r}: if the reuse decision allows it the VLM part costs 0 GPU "\n'
    '      "hours (assembly is CPU, seconds); else see the VLM table above.")\n'
)
assert _OCR_REUSE_OLD in v1.OCR_ESTIMATE, "colab_build_predict_v1.OCR_ESTIMATE changed"
OCR_ESTIMATE = v1.OCR_ESTIMATE.replace(
    _OCR_REUSE_OLD,
    'print(f"REUSE_OCR_FROM = {REUSE_OCR_FROM!r}: pages already cached on Drive cost nothing (the OCR "\n'
    '      "stage checks the cache first; a complete, valid cache is only checked, not re-run).")\n',
)

ESTIMATE = """# ESTIMATE (UNVERIFIED) of the GPU session: T4 hours and compute units of the guard, the smoke gate,
# the 280-page test run and the determinism pass (a second load + merge in a fresh process), per
# ASSUMED batching efficiency. Nothing in it is measured (scripts/gpu_estimate.py; ASSUMED 60 s
# merge, ASSUMED 120 s model load). No confirmation prompt (Run all must not block): interrupt the
# run yourself if it is too high. The OCR part is the estimate cell above.
est_cmd = [PY, "-m", "shipdoc.predict_ft", "estimate", "--batch-size", str(BATCH_SIZE),
           "--smoke-docs", SMOKE_DOCS]
rc, tail = run_stream(est_cmd)
assert rc == 0, f"estimate failed (exit {rc}): {tail[-3:]}"
print("THIS RUN: batch size", BATCH_SIZE, "| OCR record adopted from:", REUSE_OCR_FROM)
"""

PLAN = p04.PLAN

ADOPT_OCR = """# Optional: adopt the OCR timing and the OCR re-check record of a finished 04b folder (the OCR cache
# on Drive is shared, so the pages are already there). Only the records are copied, never over an
# existing file; the OCR stage below still asserts 280/280 valid cached pages. A refusal is not
# fatal: the OCR stage then does its own timing and re-check.
if OCR_FROM is None:
    print("REUSE_OCR_FROM is None: the OCR stage below times and re-checks the cache itself.")
else:
    adopt_cmd = [PY, "-m", "shipdoc.predict_ft", "adopt-ocr", "--from-dir", str(OCR_FROM),
                 "--timing-out", str(SUBMISSION_DIR / "ocr_timing.json"),
                 "--recheck-out", str(META_DIR / "ocr_recheck.json"),
                 "--expect-pages", str(EXPECTED_PAGES)]
    rc, tail = run_stream(adopt_cmd)
    if rc != 0:
        print("!" * 78)
        print("OCR RECORD NOT ADOPTED (exit", rc, "): the OCR stage below runs its own checks.")
        print("!" * 78)
"""

OCR_STAGE = v1.OCR_STAGE

ZS_CHECK = """# ZERO-SHOT TRACES CHECK (CPU). The review flags of the fine-tuned model need the zero-shot output
# of the same 200 documents (agreement between the two models). Those are the v0 traces of 04,
# reused ONLY when shipdoc.reuse.decide_reuse allows it (config / prompt hashes, model revision,
# seed, batch size, decode-path blob fingerprint, ...; the same strict rules as 04b). A refusal
# STOPS here: no zero-shot inference runs in this notebook. `git diff --stat <v0 sha> <pin>` is
# printed by the check.
reuse_cmd = [PY, "-m", "shipdoc.reuse", "check", "--v0-dir", str(V0_DIR),
             "--batch-size", str(ZS_BATCH_SIZE), "--shard", "0/1",
             "--out", str(META_DIR / "reuse_decision.json")]
if REUSE_ACK_SPIKE_DIFF:
    reuse_cmd.append("--ack-spike-diff")
    print("REUSE_ACK_SPIKE_DIFF = True: a spike.py diff beyond the allowlist is accepted.")
rc, tail = run_stream(reuse_cmd)
if rc != 0:
    reasons = json.loads((META_DIR / "reuse_decision.json").read_text())["reasons"] \\
        if (META_DIR / "reuse_decision.json").is_file() else tail[-3:]
    raise RuntimeError(
        f"ZERO-SHOT TRACES NOT REUSABLE (exit {rc}): {reasons}. Stopped before anything "
        "runs on the GPU: the flags need them and no zero-shot inference runs here. Fix the v0 "
        "folder / the pin (or run 04b's full path) and Run all again."
    )
print("ZERO-SHOT TRACES REUSABLE: code identity is replaced by the decode-path blob fingerprint.")
"""

VERIFY = """# FINAL ADAPTER MANIFEST VERIFICATION. Fails closed: any failed check raises here and NOTHING after
# this cell runs. Prints one row per check (expected, found, PASS / FAIL), the training code SHA
# next to this notebook's pin and the number of training documents (counts only, no ids). A
# training SHA
# that differs from the pin is a WARNING (the diff of the files that matter is shown below); an
# unreachable SHA, a smoke / fold adapter, a missing or wrong training-document list or a changed
# inference key is a REFUSAL.
CFG = f"configs/spike_{CONFIG}.yaml"
verify_cmd = [PY, "-m", "shipdoc.predict_ft", "verify", "--adapter-dir", str(ADAPTER),
              "--pin", PINNED_SHA, "--config", CFG]
rc, tail = run_stream(verify_cmd)
if rc != 0:
    raise RuntimeError(
        f"ADAPTER REFUSED (exit {rc}): the inference is NOT started. Read the FAIL rows above "
        "(counts, hashes and SHAs only). GG: report back; do not edit the manifest to pass."
    )
ADAPTER_MANIFEST = json.loads((ADAPTER / "manifest.json").read_text())
TRAIN_SHA = ADAPTER_MANIFEST["code_sha"]
if TRAIN_SHA != PINNED_SHA:
    print(f"training code {TRAIN_SHA[:7]} != pin {PINNED_SHA[:7]}: files that decide the model's "
          "input and output, changed between the two (empty = unchanged):")
    run_stream(["git", "diff", "--stat", TRAIN_SHA, PINNED_SHA, "--",
                "src/shipdoc/train.py", "src/shipdoc/trainset.py", "src/shipdoc/prompts.py",
                "src/shipdoc/extract.py", "configs/finetune_qwen35_4b.yaml", CFG])
print(f"VERIFIED: final adapter, training precision {ADAPTER_MANIFEST.get('precision')}.")
"""

INFER = """# INFER in ONE process (the merged model never leaves it), RESUMABLE per document. Order inside the
# process, printed as it happens: VERIFY (again, cheap), MERGE (fp16 base + the adapter,
# merge_and_unload, 200 modules required), GUARD (12 bench pages at batch 1 and at the chosen
# size on the MERGED model; any difference -> batch 1, recorded in manifest.json; a resumed run
# keeps the stored decision), SMOKE (the gate of 04, 5 dev documents, on the merged model; a
# failure stops the run before the first test document), RUN (the 200 test documents; the CLI
# appends one line to trace.jsonl and rewrites predictions.json / progress.json after every
# document).
import re
import threading
import time
from datetime import UTC, datetime

OOM_RE = re.compile(r"out of memory|OutOfMemoryError", re.IGNORECASE)
RUN_DIR.mkdir(parents=True, exist_ok=True)
SESSIONS_PATH = RUN_DIR / "sessions.json"  # wall-clock log, one entry per Run all
sessions = json.loads(SESSIONS_PATH.read_text()) if SESSIONS_PATH.is_file() else []


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
print(f"=== {RUN_ID}: {done_before}/{EXPECTED_DOCS} docs already done; resuming the rest ===")
session = {"start": datetime.now(UTC).isoformat(), "end": None, "seconds": None,
           "docs_done_at_start": done_before, "exit_code": None}
sessions.append(session)
save_json(SESSIONS_PATH, sessions)
infer_cmd = [PY, "-m", "shipdoc.predict_ft", "infer", "--adapter-dir", str(ADAPTER),
             "--pin", PINNED_SHA, "--config", CFG, "--run-id", RUN_ID,
             "--decision", str(DECISION_PATH), "--smoke-status", str(SMOKE_STATUS_PATH),
             "--smoke-docs", SMOKE_DOCS, "--bench-docs", BENCH_DOCS,
             "--batch-size", str(BATCH_SIZE), "--expect-docs", str(EXPECTED_DOCS),
             "--expect-pages", str(EXPECTED_PAGES)]
stop = threading.Event()
threading.Thread(target=watch_progress, args=(stop,), daemon=True).start()
t0 = time.time()
try:
    rc, tail = run_stream(infer_cmd)
finally:
    stop.set()
    session.update(end=datetime.now(UTC).isoformat(), seconds=round(time.time() - t0, 1))
    save_json(SESSIONS_PATH, sessions)
session["exit_code"] = rc
save_json(SESSIONS_PATH, sessions)
if rc != 0:
    oom = OOM_RE.search("\\n".join(tail))
    raise RuntimeError(
        f"Inference stopped (exit {rc}{', CUDA out of memory' if oom else ''}) after "
        f"{read_progress().get('done')}/{EXPECTED_DOCS} docs; last output: {tail[-3:]}. Everything "
        "done so far is on Drive: Run all again resumes. If it stops at the same point, "
        "report back."
    )
print(f"--- inference finished (exit 0) in {(time.time() - t0) / 60:.1f} min this session ---")
"""

DETERMINISM = """# DETERMINISM: 5 seeded test documents are decoded AGAIN in a fresh process (new model load, new
# merge, new CUDA context). The generate calls of the first pass that held those documents are
# REPLAYED exactly (same pages, same order inside the call), because a batched decode can depend on
# the companions of a page; the serialized per-document predictions must be byte-identical to the
# first pass. Counts only are printed. A difference raises: the submission is not assembled as
# submittable.
det_cmd = [PY, "-m", "shipdoc.predict_ft", "determinism", "--adapter-dir", str(ADAPTER),
           "--pin", PINNED_SHA, "--config", CFG, "--run-id", RUN_ID,
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

ASSEMBLE = """# ASSEMBLE + VALIDATION (CPU). `shipdoc predict assemble` runs the post-processing v1 (R1, R2, R3
# ON, the OCR cache passed, the frozen meta/slot_shapes.json) and every blocking check of 04: run
# complete, exactly the 200 test ids, 200 docs / 280 pages, JSON Schema, determinism, code SHA clean
# and equal to PINNED_SHA, model revision / config hash / seed 42 / prompt v2, real model and stack.
# `shipdoc.predict_ft finalize` then adds the v1 checks (rule switches all on, no R2 skipped, shapes
# sha256 = meta/slot_shapes.json, OCR cache complete, OCR re-check) and the v2 checks (adapter
# verified, model id = the merged adapter, 200 modules merged, smoke on the merged model, batch
# decision, predictions structure identical to the v0 file, no key outside doc_type / header /
# line_items); manifest `submission` = v2_<sha7>. A failure moves the predictions to
# test_predictions.REJECTED.json.
asm_cmd = [PY, "-m", "shipdoc", "predict", "assemble", "--config", CFG, "--run-id", RUN_ID,
           "--out-dir", str(SUBMISSION_DIR), "--schema", str(SCHEMA_PATH),
           "--expect-docs", str(EXPECTED_DOCS), "--expect-pages", str(EXPECTED_PAGES),
           "--expect-code-sha", PINNED_SHA, "--require-stack",
           "--smoke-status", str(SMOKE_STATUS_PATH), "--ocr-cache", str(OCR_ROOT)]
if SAMPLE_PATH.is_file():
    asm_cmd += ["--sample-submission", str(SAMPLE_PATH)]
rc, tail = run_stream(asm_cmd)
OCR_RECHECK_PATH = META_DIR / "ocr_recheck.json"
fin_cmd = [PY, "-m", "shipdoc.predict_ft", "finalize", "--out-dir", str(SUBMISSION_DIR),
           "--run-dir", str(RUN_DIR), "--v0-dir", str(V0_DIR), "--recheck", str(OCR_RECHECK_PATH),
           "--expect-pages", str(EXPECTED_PAGES)]
rc_fin, tail_fin = run_stream(fin_cmd)
if rc_fin != 0:
    raise RuntimeError(
        f"VALIDATION FAILED (assemble exit {rc}, v1 + v2 checks exit {rc_fin}): the files in "
        f"{SUBMISSION_DIR} are NOT submittable (test_predictions.REJECTED.json). Read "
        "validation_report.json there (counts and key paths only) and report back; do not submit."
    )
"""

FLAGS = """# REVIEW FLAGS (CPU), a SEPARATE file: review_flags.json in the submission folder, never part of
# test_predictions.json. `python -m shipdoc.flags` reads the validated folder (it refuses a REJECTED
# one, and refuses predictions that are not what the production post path makes of trace.jsonl),
# loads the frozen FINE-TUNED calibrator (it refuses another arm, a different feature set, a
# behaviour change of the feature builder, a GBM under another scikit-learn version), takes the
# zero-shot output of the same documents from the v0 traces (strict reuse rules again) and writes
# probabilities, accept / review flags and the thresholds; the file holds no extracted value.
# `check-flags` then checks it: exactly the 200 ids, no value string, thresholds and the calibrator
# sha256 recorded, the file belongs to these predictions. The flags are an advisory side file: a
# failure here does not withhold the validated predictions.
CALIBRATOR = REPO / CALIBRATOR_FILE
if not CALIBRATOR.is_file():
    raise FileNotFoundError(f"{CALIBRATOR_FILE} is not in the clone at the pin: freeze it first "
                            "(scripts/freeze_calibrator.py --arm ft) and re-pin")
if not (SUBMISSION_DIR / "test_predictions.json").is_file():
    raise RuntimeError("no validated test_predictions.json: the flags are not computed")
flags_cmd = [PY, "-m", "shipdoc.flags", "--submission-dir", str(SUBMISSION_DIR),
             "--calibrator", str(CALIBRATOR), "--ocr-cache", str(OCR_ROOT), "--arm", "ft",
             "--zs-v0-dir", str(V0_DIR), "--batch-size", str(ZS_BATCH_SIZE),
             "--field-target", str(FIELD_TARGET), "--doc-target", str(DOC_TARGET),
             "--expect-docs", str(EXPECTED_DOCS),
             "--out", str(SUBMISSION_DIR / "review_flags.json")]
if REUSE_ACK_SPIKE_DIFF:
    flags_cmd.append("--ack-spike-diff")
rc, tail = run_stream(flags_cmd)
if rc != 0:
    raise RuntimeError(
        f"REVIEW FLAGS REFUSED (exit {rc}): {tail[-2:]}. The predictions in {SUBMISSION_DIR} are "
        "VALIDATED and on Drive; only the flags file is missing."
    )
check_cmd = [PY, "-m", "shipdoc.predict_ft", "check-flags", "--out-dir", str(SUBMISSION_DIR),
             "--calibrator", str(CALIBRATOR), "--field-target", str(FIELD_TARGET),
             "--doc-target", str(DOC_TARGET)]
rc, tail = run_stream(check_cmd)
if rc != 0:
    print("!" * 78)
    print("REVIEW FLAGS CHECKS FAILED (exit", rc, "): see validation_report.json flags_checks.")
    print("The predictions are not affected; do not use review_flags.json.")
    print("!" * 78)
"""

BANNER = """# Completion banner. Reads only files on Drive: also runnable alone after a resume.
import json

bar = "=" * 78
BS = chr(92)
LOCAL_DEST = BS.join(["$SHIPDOC_SUBMISSIONS_DIR", f"v2_{SHA7}"]) + BS
report_path = SUBMISSION_DIR / "validation_report.json"
manifest_path = SUBMISSION_DIR / "manifest.json"
report = json.loads(report_path.read_text()) if report_path.is_file() else None
manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
print(bar)
if report is None:
    print(f"NOT ASSEMBLED: {report_path.name} missing. Run the cells above.")
    print(f"DONE {RUN_ID} assembled=False")
else:
    ok = bool(report["ok"])
    verdict = "VALIDATED: SUBMITTABLE" if ok else "REJECTED: DO NOT SUBMIT"
    print("TEST PREDICTIONS v2 (fine-tuned + R1-R3) " + verdict)
    print(bar)
    print(f"schema ok       : {report['schema_ok']}")
    print(f"docs found      : {report['docs_found']}")
    print(f"determinism ok  : {report['determinism_ok']}")
    for name, c in report["checks"].items():
        print(f"  {'PASS' if c['ok'] else 'FAIL'}  {name}: {c.get('detail', '')}")
    ft = manifest.get("ft") or {}
    print(f"final adapter {str(ft.get('adapter_sha256'))[:12]}  training code "
          f"{str(ft.get('train_code_sha'))[:7]}  pin {PINNED_SHA[:7]}  precision "
          f"{ft.get('train_precision')}")
    mg, gd = ft.get("merge") or {}, ft.get("guard") or {}
    print(f"merge: {mg.get('n_lora_modules_merged')} modules, {mg.get('merge_dtype')}, "
          f"{mg.get('load_and_merge_s')} s; guard ran={gd.get('ran')} ok={gd.get('ok')} "
          f"fallback_to_1={gd.get('fallback_to_1')}")
    print(f"code {str(manifest.get('code_sha'))[:12]}  model {manifest.get('model', {}).get('id')} "
          f"@ {str(manifest.get('model', {}).get('revision'))[:12]}  config "
          f"{manifest.get('config', {}).get('hash')}")
    print(f"batch size {manifest.get('batch_size')} ({manifest.get('batch_size_source')}); "
          f"smoke {(manifest.get('smoke') or {}).get('state')}")
    ocr_t = (manifest.get("ocr") or {}).get("timing") or {}
    print(f"OCR: {ocr_t.get('pages')} pages, {ocr_t.get('seconds_total')} s total, "
          f"{ocr_t.get('seconds_per_page_mean')} s/page mean, p95 "
          f"{ocr_t.get('seconds_per_page_p95')}, device {ocr_t.get('device')}")
    post = manifest.get("post_rules") or {}
    print(f"rules R1/R2/R3: touched docs {post.get('touched_docs')}, skipped {post.get('skipped')}")
    tm = manifest.get("timings", {})
    print(f"wall-clock {tm.get('wall_clock_s')} s over {tm.get('sessions')} session(s); model time "
          f"{tm.get('model_time_s')} s; smoke {tm.get('smoke_s')} s")
    rf = manifest.get("review_flags")
    flags_ok = bool(report.get("flags_ok")) if "flags_ok" in report else None
    if rf is None:
        print("REVIEW FLAGS: not computed (the flags cell did not run or failed)")
    else:
        s = rf.get("summary") or {}
        state = "OK" if flags_ok else "CHECKS FAILED"
        print(f"REVIEW FLAGS {state} (arm {rf.get('arm')}, calibrator "
              f"sha256 {str(rf.get('calibrator_sha256'))[:12]}): {s.get('n_docs')} docs, "
              f"auto-accept {s.get('n_docs_auto_accept')}, fields accept {s.get('n_accept')} / "
              f"review {s.get('n_review')}; field tau {rf['thresholds'].get('field_tau')}, doc tau "
              f"{rf['thresholds'].get('doc_tau')}")
        for name, c in (report.get("flags_checks") or {}).items():
            print(f"  {'PASS' if c['ok'] else 'FAIL'}  {name}: {c.get('detail', '')}")
    print(f"Drive folder: {SUBMISSION_DIR}")
    print("GG: download these files from that Drive folder:")
    wanted = ["test_predictions.json" if ok else "test_predictions.REJECTED.json",
              "review_flags.json", "trace.jsonl", "manifest.json", "validation_report.json",
              "rules.jsonl", "ocr_timing.json"]
    for name in wanted:
        path = SUBMISSION_DIR / name
        size = f"{path.stat().st_size / 1e6:8.2f} MB" if path.is_file() else "MISSING"
        print(f"  {name:<34}{size}")
    print(f"into the LOCAL folder  {LOCAL_DEST}")
    print("(outside the repo; submissions/ is gitignored). Never paste document values into chat.")
    print(f"DONE {RUN_ID} assembled=True submittable={ok} flags_ok={flags_ok}")
print(bar)
"""


def _cells() -> list[tuple[str, str]]:
    return [
        ("markdown", TITLE),
        ("code", PARAMS),
        ("markdown", base.ACCOUNT_MD),
        ("code", MOUNT),
        ("code", p04.SECRETS),
        ("code", base.CLONE),
        ("code", p04.UNZIP),
        ("code", OCR_ESTIMATE),
        ("code", INSTALL),
        ("code", ESTIMATE),
        ("code", PLAN),
        ("code", ADOPT_OCR),
        ("code", OCR_STAGE),
        ("code", ZS_CHECK),
        ("code", VERIFY),
        ("code", INFER),
        ("code", DETERMINISM),
        ("code", ASSEMBLE),
        ("code", FLAGS),
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
