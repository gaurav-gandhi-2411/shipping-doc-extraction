"""Generate notebooks/04b_predict_test_v1.ipynb (production pipeline v1) from the cells below.

04b = 04 (model outputs for the 200 test documents) + OCR of the 280 test pages + post-rules
R1 / R2 / R3. A thin wrapper like 04: every cell orchestrates (Drive, secrets, clone, one
subprocess per stage); the logic is in ``python -m shipdoc predict <stage>``,
``python -m shipdoc.ocr_stage`` and ``python -m shipdoc.reuse``. Cells and helpers that do not
change are imported READ-ONLY from scripts/colab_build_predict.py (04) and
scripts/colab_build_notebook.py, so the notebooks cannot drift apart; 04 itself is not modified.

tests/test_notebook_predict_v1.py checks the committed notebook equals this script's output.

Run: uv run python scripts/colab_build_predict_v1.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "archive" / "04b_predict_test_v1.ipynb"


def _load(name: str, path: Path):  # noqa: ANN202 - a module object
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


base = _load("colab_build_notebook", ROOT / "scripts" / "colab_build_notebook.py")
p04 = _load("colab_build_predict", ROOT / "scripts" / "colab_build_predict.py")

# GG pins 04b only after confirming that 04 finished (two-commit pattern as 01/02/04): the pin
# commit replaces this placeholder with the full 40-char SHA and regenerates the notebook.
# Until then the parameters cell RAISES.
PINNED_SHA = "8833c73e799dcce4372dc72f133835072a0b8126"

TITLE = """# 04b - production pipeline v1: model outputs + OCR + post-rules R1/R2/R3 (thin wrapper)

v1 = the 04 safety submission (v0: Qwen3.5-4B, image only, KEYED output, prompt v2, greedy, seed
42, fp16, logprobs on) + the OCR text of the 280 test pages (PaddleOCR PP-OCRv5 server, own venv)
+ post-processing rules R1 (carrier from the model's own supplier slot), R2 (waybill pattern
backfill from OCR), R3 (invoice slot-shape swap with the frozen `meta/slot_shapes.json`), all
three ON (`reports/v1_replay.md`, `reports/rule_gate.md`). Test images only: there are no test
labels, nothing here scores anything, and no document value is printed (counts, ids, hashes).

Flow: setup (Drive, clone at the pin, unzip data incl. test images, 200 docs / 280 pages by file
name) -> ESTIMATE (T4 hours / CU for the VLM part if it must run, and the OCR part) -> base env
(`uv sync --frozen`, NO vlm group, NO paddle) -> plan -> OCR stage (own venv at /content/ocr-venv
via a subprocess: GPU with a loud CPU fallback, resumable per page, 280/280 asserted,
`ocr_timing.json`, a 5-page seeded re-OCR in a fresh process) -> REUSE DECISION -> either
(a) REUSE the v0 model outputs, or (b) FULL INFERENCE (VLM env, the same smoke gate, batch size
contract, resumable run and determinism pass as 04) -> assemble with R1-R3 -> validation
(the blocking checks of 04 + the v1 checks) -> banner VALIDATED / REJECTED.

REUSE is allowed ONLY when a pure function (`shipdoc.reuse.decide_reuse`) finds all of: the v0
folder (`REUSE_V0_DIR`) holds a manifest whose validation report says ok; v0 config hash,
prompt hash, model revision, seed, output format, arm, shard and BATCH SIZE equal v1's; the v0
trace has exactly the 200 test ids with complete pages and logprobs; and the DECODE-PATH
FINGERPRINT holds. The v0 code SHA legitimately differs from this pin, so code identity is
REPLACED by: the git blob hash of every file on the model-output path is identical at the v0 SHA
and at HEAD, `spike.py` differs only by the allowlisted opt-in additions (`header_hint`,
`rule_cfg`; default off), plus the config and prompt hashes. Anything else REFUSES, loudly, and
the notebook runs the full inference path instead; reuse is never silent. In reuse mode the VLM is
not run: its determinism rests on v0's own check (re-attached to the manifest with its hash).

Parameters: `PINNED_SHA`; `BATCH_SIZE` (8 = the 02/04 production size; must equal v0's for reuse
and the dev run's); `SHARD` / `MODE` ("run" or "merge", full inference only; reuse needs "0/1");
`REUSE_V0_DIR` (default `submissions/v0_42b812b` under MyDrive/shipdoc-extract, None = always
full inference); `REUSE_ACK_SPIKE_DIFF` (False: a `spike.py` diff that is not the allowlisted
one refuses reuse; True accepts any such diff, recorded in the manifest); `DEV_RUN_DIR` as 04.

Output: `MyDrive/shipdoc-extract/submissions/v1_<sha7>/` with `test_predictions.json`,
`trace.jsonl`, `manifest.json`, `validation_report.json`, `rules.jsonl`, `ocr_timing.json`; on a
failed check the predictions are `test_predictions.REJECTED.json`. Download by hand into
`$SHIPDOC_SUBMISSIONS_DIR\\v1_<sha7>\\`. If the session dies, Run all: every stage resumes.
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
BATCH_SIZE = 8  # the 02/04 production batch size; must equal v0's (reuse) and the dev run's
SHARD = "0/1"  # "i/K": full inference only; reuse needs "0/1"
MODE = "run"  # "run" = OCR, reuse decision, (inference), assemble; "merge" = combine K shards
REUSE_V0_DIR = "submissions/v0_42b812b"  # under MyDrive/shipdoc-extract, or None = no reuse
REUSE_ACK_SPIKE_DIFF = False  # True accepts a spike.py diff beyond the allowlisted opt-in ones
DEV_RUN_DIR = "zeroshot500_qwen35_4b_img_only_keyed_42b812b"  # dev run folder under runs/, or None

if not PINNED_SHA or "FILL" in PINNED_SHA:
    raise ValueError("Set PINNED_SHA in the parameters cell (the pin commit fills it).")
if MODE not in ("run", "merge"):
    raise ValueError(f"MODE must be 'run' or 'merge', got {{MODE!r}}")
if BATCH_SIZE is not None and (isinstance(BATCH_SIZE, bool) or not isinstance(BATCH_SIZE, int)
                               or BATCH_SIZE < 1):
    raise ValueError(f"BATCH_SIZE must be None or an int >= 1, got {{BATCH_SIZE!r}}")
if REUSE_V0_DIR is not None and not isinstance(REUSE_V0_DIR, str):
    raise ValueError(f"REUSE_V0_DIR must be None or a path string, got {{REUSE_V0_DIR!r}}")
if not isinstance(REUSE_ACK_SPIKE_DIFF, bool):
    raise ValueError(f"REUSE_ACK_SPIKE_DIFF must be True or False, got {{REUSE_ACK_SPIKE_DIFF!r}}")
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

_V0_NAME = 'SUBMISSION_DIR = SUBMISSIONS_DIR / f"v0_{SHA7}"'
assert _V0_NAME in p04.MOUNT
MOUNT = (
    p04.MOUNT.replace(_V0_NAME, 'SUBMISSION_DIR = SUBMISSIONS_DIR / f"v1_{SHA7}"')
    + """OCR_ROOT = DRIVE_DIR / "ocr_cache_test"  # <root>/paddleocr/test/<page>.json, resumable
OCR_ROOT.mkdir(parents=True, exist_ok=True)
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
V0_DIR = None  # the 04 submission folder whose model outputs may be reused (decided below)
if REUSE_V0_DIR:
    V0_DIR = Path(REUSE_V0_DIR) if Path(REUSE_V0_DIR).is_absolute() else DRIVE_DIR / REUSE_V0_DIR
    print("v0 folder (reuse candidate):", V0_DIR, "| manifest present:",
          (V0_DIR / "manifest.json").is_file())
print("OCR cache folder (Drive):", OCR_ROOT)
"""
)

_ESTIMATE_OLD = "v0 TEST run"
assert _ESTIMATE_OLD in p04.ESTIMATE
# the replacement keeps the line length of the 04 cell (the note on reuse is in the OCR estimate)
VLM_ESTIMATE = p04.ESTIMATE.replace(_ESTIMATE_OLD, "v1 VLM run")

OCR_ESTIMATE = """# ESTIMATE (UNVERIFIED) of the OCR stage: 280 test pages, PaddleOCR PP-OCRv5
# server
# with the document + text-line orientation classifiers (shipdoc.ocr). No GPU measurement of
# it exists in this repo: the GPU figure is an ASSUMPTION. The CPU figure is the only
# measurement: 10.16 s/page mean over 951 pages (reports/ocr.md, local 8-thread CPU, shared
# with other jobs); a free Colab
# runtime has fewer cores, so the Colab CPU figure below doubles it (also an ASSUMPTION).
import json

OCR_GPU_S_PER_PAGE = 2.0  # ASSUMED, not measured
OCR_CPU_S_PER_PAGE = 2 * 10.16  # local measurement x 2 (ASSUMED Colab CPU slowdown)
OCR_SETUP_MIN = 10.0  # ASSUMED: venv build + paddle wheel + model download + first model load
_speed = json.loads((REPO / "configs" / "spike_speed.json").read_text(encoding="utf-8"))
_rate = _speed.get("t4_cu_per_hour")
_gpu_h = (OCR_SETUP_MIN * 60 + EXPECTED_PAGES * OCR_GPU_S_PER_PAGE) / 3600
_cpu_h = (OCR_SETUP_MIN * 60 + EXPECTED_PAGES * OCR_CPU_S_PER_PAGE) / 3600
print("=" * 100)
print("ESTIMATE (UNVERIFIED) OCR stage, resumable per page (pages already cached are skipped):")
print(f"  GPU (T4): {_gpu_h:.2f} h incl. {OCR_SETUP_MIN:.0f} min setup "
      f"({OCR_GPU_S_PER_PAGE} s/page ASSUMED), ~{_gpu_h * _rate:.2f} CU @ {_rate} CU/h")
print(f"  CPU fallback: {_cpu_h:.2f} h ({OCR_CPU_S_PER_PAGE:.1f} s/page ASSUMED), CU rate of a "
      "CPU runtime is not in configs/spike_speed.json: n/a")
print("  + a seeded 5-page re-OCR in a fresh process (about one model load + 5 pages).")
print(f"REUSE_V0_DIR = {REUSE_V0_DIR!r}: if the reuse decision allows it the VLM part costs 0 GPU "
      "hours (assembly is CPU, seconds); else see the VLM table above.")
print("=" * 100)
"""

_inst = base.ZS_INSTALL
_INSTALL_EDITS = [
    (
        '    "SHIPDOC_OCR_CACHE": "/content/ocr_cache",  # unused by img_only; the path is '
        "never read\n",
        '    "SHIPDOC_OCR_CACHE": str(OCR_ROOT),  # the test OCR cache that R2 reads\n'
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
_UV_CHECK = 'if subprocess.run(["uv", "--version"]'
_VLM_GROUP = 'VLM_GROUP = ["--group", "vlm"] if MODE == "run" else []'
_PROBE = 'if MODE == "run":\n    subprocess.run(["nvidia-smi"]'
for _marker in (_UV_CHECK, _VLM_GROUP, _PROBE):
    assert _marker in _inst, f"colab_build_notebook.ZS_INSTALL changed: {_marker[:40]!r}"
_head, _, _tail = _inst.partition(_UV_CHECK)
_probe = _tail[_tail.index(_PROBE) :]

NO_PADDLE = """_no_ocr = (
    "import importlib.util as u, sys; sys.exit(0 if u.find_spec('paddle') is None else 1)"
)
assert subprocess.run([PY, "-c", _no_ocr], env=ENV, check=False).returncode == 0, (
    "the OCR stack is in the decoding venv: Paddle runs ONLY in /content/ocr-venv"
)
"""

# Base environment: the CPU stack of the repo (jsonschema, pycountry, ...). Enough for the plan,
# the OCR stage helpers, the reuse decision and the assembly. NO vlm group, NO ocr group.
INSTALL_BASE = (
    _head
    + _UV_CHECK
    + _tail.split(_VLM_GROUP)[0]
    + """rc, _ = run_stream(["uv", "sync", "--frozen", "--python", "3.11"])
assert rc == 0, f"uv sync failed (exit {rc})"
# Paddle must never be in this environment (it is its own dependency group, installed only into
# the separate OCR venv by scripts/colab_ocr_test.sh).
"""
    + NO_PADDLE
)
# The VLM stack: only when the model has to run (no reuse). Needs a CUDA GPU (T4).
INSTALL_VLM = (
    'rc, _ = run_stream(["uv", "sync", "--frozen", "--group", "vlm", "--python", "3.11"])\n'
    'assert rc == 0, f"uv sync (vlm) failed (exit {rc})"\n' + _probe + "\n" + NO_PADDLE
)

PLAN = p04.PLAN

OCR_STAGE = """# OCR of the 280 TEST pages, in its OWN venv (paddlepaddle-gpu clashes with the vlm
# group, so Paddle is never installed into or imported by the environment that decodes: this
# cell only starts subprocesses). Resumable: pages already cached AND valid are skipped (an
# invalid page file is renamed to *.invalid, never deleted, and redone). GPU when the runtime
# has one and the GPU wheel installed, else CPU with a LOUD warning. OCR text is never
# printed: counts and seconds only.
# Asserts 280/280 valid cached pages, writes ocr_timing.json (pages, seconds total, seconds/page
# mean and p95, device) to the submission folder, then re-OCRs 5 seeded pages in a fresh process.
import json
import subprocess
import time

OCR_VENV = Path("/content/ocr-venv")  # never the venv that decodes
assert OCR_VENV.resolve() != (REPO / ".venv").resolve(), "OCR must not share the decoding venv"
OCR_SH = ["bash", str(REPO / "scripts" / "colab_ocr_test.sh")]
OCR_ENV = {**ENV, "OCR_VENV": str(OCR_VENV)}
OCR_TIMING_PATH = SUBMISSION_DIR / "ocr_timing.json"
OCR_RECHECK_PATH = META_DIR / "ocr_recheck.json"
STAGE = [PY, "-m", "shipdoc.ocr_stage"]
PAGE_ARGS = ["--images-dir", str(TEST_IMAGES), "--cache-root", str(OCR_ROOT),
             "--expect-pages", str(EXPECTED_PAGES)]


def has_gpu() -> bool:
    try:
        res = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30,
                             check=False)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return res.returncode == 0 and "GPU" in res.stdout


def ocr_device() -> dict:
    \"\"\"Device from the runtime and the OCR venv's paddle flavour; loud on a CPU fallback.\"\"\"
    flavor_file = OCR_VENV / ".paddle_flavor"
    flavor = flavor_file.read_text().strip() if flavor_file.is_file() else ""
    res = subprocess.run([*STAGE, "device", "--has-gpu", "1" if has_gpu() else "0",
                          "--flavor", flavor], cwd=REPO, env=ENV, capture_output=True, text=True,
                         check=False)
    info = json.loads(res.stdout.strip().splitlines()[-1])
    for warning in info["warnings"]:
        print("!" * 78)
        print("OCR DEVICE WARNING:", warning)
        print("!" * 78)
    if not info["ok"]:
        raise RuntimeError(f"OCR cannot run on this runtime: {info['warnings']}")
    print(f"OCR device: {info['device']} (paddle flavour {info['flavor'] or 'unknown'})")
    return info


def ocr_status(tag: str) -> dict:
    path = META_DIR / f"ocr_check_{tag}.json"
    rc, _ = run_stream([*STAGE, "check", *PAGE_ARGS, "--isolate-invalid", "--out", str(path)])
    return {"rc": rc, **json.loads(path.read_text())}


before = ocr_status("before")
wall, device = None, None
if before["rc"] == 0:
    print(f"OCR cache already complete ({before['valid']}/{EXPECTED_PAGES}): no OCR run")
else:
    print(f"OCR cache: {before['valid']}/{EXPECTED_PAGES} valid; running the missing pages")
    t_ocr = time.time()
    rc, tail = run_stream([*OCR_SH, "setup"], env=OCR_ENV)
    assert rc == 0, f"OCR venv setup failed (exit {rc}): {tail[-3:]}"
    device = ocr_device()["device"]
    rc, tail = run_stream([*OCR_SH, "run", str(DATA_DIR), str(OCR_ROOT)], env=OCR_ENV)
    wall = time.time() - t_ocr
    print(f"--- OCR run exited {rc} after {wall / 60:.1f} min (the check below decides) ---")
after = ocr_status("after")
if after["rc"] != 0:
    raise RuntimeError(
        f"OCR INCOMPLETE: {after['valid']}/{EXPECTED_PAGES} valid pages, {after['missing']} "
        f"missing {after['missing_pages']}, {after['invalid']} invalid. Everything cached is on "
        "Drive: Run all again resumes the rest."
    )
assert after["valid"] == EXPECTED_PAGES, f"{after['valid']} cached OCR pages, not {EXPECTED_PAGES}"
prior = json.loads(OCR_TIMING_PATH.read_text()) if OCR_TIMING_PATH.is_file() else {}
device = device or prior.get("device") or "unknown (the cache was complete at the start)"
fin = [*STAGE, "finalize", *PAGE_ARGS, "--device", device, "--out", str(OCR_TIMING_PATH)]
if wall is not None:
    fin += ["--wall-s", f"{wall:.1f}", "--new-pages", str(after["valid"] - before["valid"])]
rc, tail = run_stream(fin)
assert rc == 0, f"ocr timing failed (exit {rc}): {tail[-3:]}"
assert subprocess.run([PY, "-c", "import importlib.util as u, sys; "
                       "sys.exit(0 if u.find_spec('paddle') is None else 1)"],
                      env=ENV, check=False).returncode == 0, "Paddle leaked into the decoding venv"

# OCR determinism: 5 seeded pages (random.Random(42)) OCR'd AGAIN in a fresh process (the OCR
# venv, into a scratch cache); blocking criterion = identical page text (what R2 reads), the whole
# page JSON minus `seconds` is reported too. A failure is recorded and REJECTS the submission.
recheck = json.loads(OCR_RECHECK_PATH.read_text()) if OCR_RECHECK_PATH.is_file() else {}
if recheck.get("ok") is True:
    print("OCR recheck already passed:", recheck["n_text_identical"], "/", recheck["n_pages"])
else:
    rc, tail = run_stream([*OCR_SH, "setup"], env=OCR_ENV)  # a no-op unless the VM is new
    assert rc == 0, f"OCR venv setup failed (exit {rc}): {tail[-3:]}"
    ocr_device()
    rc, tail = run_stream([*OCR_SH, "module", "recheck", *PAGE_ARGS, "--scratch-root",
                           "/content/ocr_recheck", "--out", str(OCR_RECHECK_PATH)], env=OCR_ENV)
    if rc != 0:
        print("!" * 78)
        print("OCR RE-CHECK DID NOT PASS (exit", rc, "): the submission will be REJECTED. Report")
        print("back with", OCR_RECHECK_PATH, "(page names and booleans only).")
        print("!" * 78)
"""

REUSE = """# REUSE DECISION (CPU, pure function shipdoc.reuse.decide_reuse). The v0 model outputs
# are reused ONLY when every condition holds; one refusal falls back to FULL INFERENCE,
# loudly (never silent). Code identity cannot hold literally (the v0 pin is older than this
# pin): it is replaced by the decode-path fingerprint = git blob hashes of every file on the
# model-output path identical at the v0 SHA and HEAD (spike.py: only the allowlisted opt-in
# additions) + config / prompt hashes.
# `git diff --stat <v0 sha> <pin> -- src configs` is printed below.
REUSE_OK = False
if V0_DIR is None:
    print("REUSE_V0_DIR is None: full inference.")
elif MODE != "run" or SHARD_K != 1:
    print(f"MODE {MODE!r} / SHARD {SHARD!r}: reuse needs MODE 'run' and SHARD '0/1': full "
          "inference.")
elif BATCH_SIZE is None:
    print("BATCH_SIZE is None (the bench decides): reuse needs an explicit size: full inference.")
else:
    reuse_cmd = [PY, "-m", "shipdoc.reuse", "check", "--v0-dir", str(V0_DIR),
                 "--batch-size", str(BATCH_SIZE), "--shard", SHARD,
                 "--out", str(META_DIR / "reuse_decision.json")]
    if REUSE_ACK_SPIKE_DIFF:
        reuse_cmd.append("--ack-spike-diff")
        print("REUSE_ACK_SPIKE_DIFF = True: a spike.py diff beyond the allowlist is accepted.")
    rc, tail = run_stream(reuse_cmd)
    REUSE_OK = rc == 0
bar = "=" * 78
if REUSE_OK:
    print(bar)
    print("REUSE ALLOWED: the 04 model outputs are reused; the VLM is NOT run (no GPU needed).")
    print("Code identity is REPLACED by the decode-path blob-hash fingerprint + prompt/config "
          "hashes.")
    print("VLM determinism rests on v0's own determinism check (re-attached to the manifest).")
    print(bar)
else:
    print("!" * 78)
    print("REUSE NOT USED -> FULL INFERENCE on the GPU (smoke gate, batch contract, run, "
          "determinism).")
    if V0_DIR is not None and (META_DIR / "reuse_decision.json").is_file():
        print("refusals:", json.loads((META_DIR / "reuse_decision.json").read_text())["reasons"])
    print("!" * 78)
NEED_INFERENCE = MODE == "run" and not REUSE_OK
"""

ASSEMBLE = """# ASSEMBLE + VALIDATION (CPU). Post-processing v1: R1, R2, R3 ON (default), the OCR
# cache passed, the frozen meta/slot_shapes.json. Reuse path: `shipdoc.reuse assemble` runs
# TWICE in fresh processes (the second into a scratch folder; the outputs must be byte-
# identical). Full path: `predict assemble` (every blocking check of 04: schema, exactly the
# 200 ids, determinism, batch contract, code SHA clean and equal to PINNED_SHA, ...). Then
# `shipdoc.reuse finalize` adds
# the v1 checks: rule switches all True, no R2 skipped for no_ocr / ocr_incomplete, shapes sha256
# equals meta/slot_shapes.json, OCR cache complete, OCR re-check, and (reuse) assemble-twice.
# A failure moves the predictions to test_predictions.REJECTED.json.
if MODE == "run" and SHARD_K > 1:
    print(f"Shard {SHARD} done. NEXT: when all {SHARD_K} tabs are COMPLETE, set MODE = 'merge' and "
          f"SHARD = '0/{SHARD_K}' in any tab and Run all: it merges, validates and assembles.")
else:
    if REUSE_OK:
        MODE_NAME = "reuse"
        RERUN_DIR = META_DIR / "assemble_rerun"
        asm_cmd = [PY, "-m", "shipdoc.reuse", "assemble", "--v0-dir", str(V0_DIR),
                   "--batch-size", str(BATCH_SIZE), "--shard", SHARD,
                   "--out-dir", str(SUBMISSION_DIR), "--schema", str(SCHEMA_PATH),
                   "--ocr-cache", str(OCR_ROOT), "--expect-docs", str(EXPECTED_DOCS),
                   "--expect-pages", str(EXPECTED_PAGES), "--expect-code-sha", PINNED_SHA]
        if SAMPLE_PATH.is_file():
            asm_cmd += ["--sample-submission", str(SAMPLE_PATH)]
        if DEV_RUN is not None:
            asm_cmd += ["--dev-run-dir", str(DEV_RUN)]
        if REUSE_ACK_SPIKE_DIFF:
            asm_cmd.append("--ack-spike-diff")
        rc, tail = run_stream(asm_cmd)
        if rc == 3:
            raise RuntimeError(f"REUSE REFUSED at assembly: {tail[-3:]}. Run all again to see the "
                               "decision; the full inference path runs when reuse is refused.")
        twice = [*asm_cmd[:asm_cmd.index("--out-dir")], "--out-dir", str(RERUN_DIR),
                 *asm_cmd[asm_cmd.index("--out-dir") + 2:]]
        rc2, tail2 = run_stream(twice)  # a fresh process: the outputs must match byte for byte
        fin_extra = ["--rerun-dir", str(RERUN_DIR)]
    else:
        MODE_NAME = "full"
        asm_cmd = [PY, "-m", "shipdoc", "predict", "assemble", "--config",
                   f"configs/spike_{CONFIG}.yaml", "--run-id", RUN_ID,
                   "--out-dir", str(SUBMISSION_DIR),
                   "--schema", str(SCHEMA_PATH), "--expect-docs", str(EXPECTED_DOCS),
                   "--expect-pages", str(EXPECTED_PAGES), "--expect-code-sha", PINNED_SHA,
                   "--require-stack", "--smoke-status", str(SMOKE_STATUS_PATH),
                   "--ocr-cache", str(OCR_ROOT)]
        if SAMPLE_PATH.is_file():
            asm_cmd += ["--sample-submission", str(SAMPLE_PATH)]
        if DEV_RUN is not None:
            asm_cmd += ["--dev-run-dir", str(DEV_RUN)]
        rc, tail = run_stream(asm_cmd)
        fin_extra = []
    fin_cmd = [PY, "-m", "shipdoc.reuse", "finalize", "--out-dir", str(SUBMISSION_DIR),
               "--mode", MODE_NAME, "--recheck", str(OCR_RECHECK_PATH),
               "--expect-pages", str(EXPECTED_PAGES), *fin_extra]
    rc_fin, tail_fin = run_stream(fin_cmd)
    if rc_fin != 0:
        raise RuntimeError(
            f"VALIDATION FAILED (assemble exit {rc}, v1 checks exit {rc_fin}): the files in "
            f"{SUBMISSION_DIR} are NOT submittable (test_predictions.REJECTED.json). Read "
            "validation_report.json there (counts and key paths only) and report back; do not "
            "submit."
        )
"""

BANNER = """# Completion banner. Reads only files on Drive: also runnable alone after a resume.
import json

bar = "=" * 78
BS = chr(92)
LOCAL_DEST = BS.join(["$SHIPDOC_SUBMISSIONS_DIR", f"v1_{SHA7}"]) + BS
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
    print("TEST PREDICTIONS v1 " + ("VALIDATED: SUBMITTABLE" if ok else "REJECTED: DO NOT SUBMIT"))
    print(bar)
    print(f"mode            : {report.get('mode')}  (reuse = the v0 model outputs, full = decoded)")
    print(f"schema ok       : {report['schema_ok']}")
    print(f"docs found      : {report['docs_found']}")
    print(f"determinism ok  : {report['determinism_ok']}")
    for name, c in report["checks"].items():
        print(f"  {'PASS' if c['ok'] else 'FAIL'}  {name}: {c.get('detail', '')}")
    reuse = manifest.get("reuse")
    if reuse:
        dp = reuse["decision"]["decode_path"]
        print(f"REUSE of {reuse['v0_dir']} (v0 code {str(reuse['v0_code_sha'])[:12]}): "
              "code identity REPLACED by")
        print(f"  decode-path blob fingerprint: {dp['files']} files identical, spike.py "
              f"{dp['spike']['state']}; config hash {manifest.get('config', {}).get('hash')}")
        print("  VLM determinism rests on v0's own check (manifest.determinism_v0, sha256 of its "
              "source recorded)")
    print(f"code {str(manifest.get('code_sha'))[:12]}  model {manifest.get('model', {}).get('id')} "
          f"@ {str(manifest.get('model', {}).get('revision'))[:12]}  config "
          f"{manifest.get('config', {}).get('hash')}")
    print(f"batch size {manifest.get('batch_size')} ({manifest.get('batch_size_source')}); "
          f"contract: {manifest.get('batch_size_contract')}")
    ocr_t = (manifest.get("ocr") or {}).get("timing") or {}
    print(f"OCR: {ocr_t.get('pages')} pages, {ocr_t.get('seconds_total')} s total, "
          f"{ocr_t.get('seconds_per_page_mean')} s/page mean, p95 "
          f"{ocr_t.get('seconds_per_page_p95')}, device {ocr_t.get('device')}")
    post = manifest.get("post_rules") or {}
    print(f"rules R1/R2/R3: touched docs {post.get('touched_docs')}, skipped {post.get('skipped')}")
    tm = manifest.get("timings", {})
    print(f"wall-clock {tm.get('wall_clock_s')} s over {tm.get('sessions')} session(s); model time "
          f"{tm.get('model_time_s')} s; smoke {tm.get('smoke_s')} s (from_v0_run: "
          f"{tm.get('from_v0_run', False)})")
    print(f"Drive folder: {SUBMISSION_DIR}")
    print("GG: download these files from that Drive folder:")
    wanted = ["test_predictions.json" if ok else "test_predictions.REJECTED.json",
              "trace.jsonl", "manifest.json", "validation_report.json", "rules.jsonl",
              "ocr_timing.json"]
    for name in wanted:
        path = SUBMISSION_DIR / name
        size = f"{path.stat().st_size / 1e6:8.2f} MB" if path.is_file() else "MISSING"
        print(f"  {name:<34}{size}")
    print(f"into the LOCAL folder  {LOCAL_DEST}")
    print("(outside the repo; submissions/ is gitignored). Never paste document values into chat.")
    print(f"DONE {SHARD_RUN_ID} assembled=True submittable={ok}")
print(bar)
"""


def _when(src: str, cond: str, what: str) -> str:
    """`src` guarded by ``if <cond>:`` (inference cells run only when the model must run).

    Only for cells without multi-line string literals (the indent would end up inside them).
    """
    body = "".join(("    " + ln) if ln.strip() else ln for ln in src.splitlines(True))
    return f'if {cond}:\n{body}\nelse:\n    print("skipped ({what}): {cond} is false")\n'


def _cells() -> list[tuple[str, str]]:
    return [
        ("markdown", TITLE),
        ("code", PARAMS),
        ("markdown", base.ACCOUNT_MD),
        ("code", MOUNT),
        ("code", p04.SECRETS),
        ("code", base.CLONE),
        ("code", p04.UNZIP),
        ("code", base._run_only(VLM_ESTIMATE, "estimate")),
        ("code", OCR_ESTIMATE),
        ("code", INSTALL_BASE),
        ("code", PLAN),
        ("code", OCR_STAGE),
        ("code", REUSE),
        ("code", _when(INSTALL_VLM, "NEED_INFERENCE", "VLM environment")),
        ("code", _when(p04.SMOKE, "NEED_INFERENCE", "smoke gate")),
        ("code", _when(p04.BATCH, "NEED_INFERENCE", "batch size")),
        ("code", _when(p04.RUN, "NEED_INFERENCE", "test run")),
        ("code", _when(p04.DETERMINISM, "NEED_INFERENCE", "determinism pass")),
        ("code", p04.MERGE),
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
