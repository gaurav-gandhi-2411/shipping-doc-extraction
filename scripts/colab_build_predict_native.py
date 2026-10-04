"""Generate notebooks/04c_predict_test_native.ipynb (BOTH final-system paths at the NATIVE resolution).

04b / 04c at the native production resolution (``configs/spike_qwen35_4b_img_only_native.yaml``,
``max_pixels`` 2,196,480; GG decision 2026-10-03), with a ``MODEL`` switch: ``"zs"`` = the base model,
no adapter ('v1.5', output ``v15_<sha7>``), ``"ft"`` = the final native adapter ('v2', output
``v2n_<sha7>``). A thin wrapper like 04 / 04b / 04c: every cell orchestrates (Drive, secrets, clone,
one subprocess per stage); the logic is in ``python -m shipdoc.predict_native`` (the native
refusals and glue), ``shipdoc predict`` (the zs inference stages of 04), ``python -m
shipdoc.predict_ft`` (the ft stages of 04c, behind the resolution gate), ``python -m
shipdoc.flags`` and ``python -m shipdoc.ocr_stage``. Cells that do not change are imported
READ-ONLY from scripts/colab_build_notebook.py, colab_build_predict.py (04),
colab_build_predict_v1.py (04b), colab_build_predict_v2.py (04c) and colab_build_oof.py (05) and
edited by exact-string replacement (asserted); none of them is modified.

tests/test_notebook_predict_native.py checks the committed notebook equals this script's output.

Run: uv run python scripts/colab_build_predict_native.py
"""

# ruff: noqa: E501  # cell sources are verbatim notebook text

from __future__ import annotations

import importlib.util
import io
import json
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "04c_predict_test_native.ipynb"


def _load(name: str, path: Path):  # noqa: ANN202 - a module object
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


v2 = _load("colab_build_predict_v2", ROOT / "scripts" / "colab_build_predict_v2.py")
base, p04 = v2.base, v2.p04

# UNPINNED until GG pins it: the pin commit replaces this placeholder with the full 40-char SHA of
# the pushed code commit and regenerates the notebook. Until then the parameters cell RAISES. The
# notebook also refuses to run without every input it needs (see the parameters cell).
PINNED_SHA = base.public_pin("4c17aa3c33c09f0cda7bb1f625947a1143a8cb28")  # public tree: its pin

TITLE = """# 04c (native) - test predictions at the NATIVE resolution, both final-system paths: MODEL = "zs" (v1.5) or "ft" (v2) (thin wrapper)

GG decision 2026-10-03: the production resolution is native, `max_pixels` 2,196,480 = 2,145 visual
tokens per page. The pre-registered final-system rule (spec section 11) picks between ZS + rules and
FT + rules after the 3-fold out-of-fold comparison, so BOTH deliverable paths are ready here, behind
one switch. Qwen3.5-4B, IMAGE ONLY, KEYED output, prompt v2, greedy, xgrammar, seed 42, fp16,
logprobs on, the 200 test documents / 280 pages, OCR (the 04b cache) + R1 / R2 / R3 all ON. Test
images only: there are no test labels, nothing here scores anything, and no document value is
printed (counts, ids, hashes). The 1260-token notebooks 04 / 04b / 04c and their submissions are
untouched.

**`MODEL = "zs"` ('v1.5', default): the BASE model, no adapter.** The v0 traces are at 1260 tokens
and can NOT be reused (their config hash differs; the check cell proves the reuse decision refuses
them), so this path runs the VLM: smoke gate (5 dev documents), batch size = the one stored with the
02n run (`ZS_RUN_DIR`; the batch contract of 04: the test batch must EQUAL the zero-shot run's),
resumable run, determinism pass (5 seeded test documents decoded again in a fresh process, replaying
the exact generate calls of the first pass), then OCR, R1-R3 and the blocking checks of 04b (0 R2
skips, shapes sha256 = the repo file, every rule on). Output `submissions/v15_<sha7>/`.

**`MODEL = "ft"` ('v2'): the FINAL native adapter**, verified as in 04c (+ the resolution gate),
merged into the fp16 weights, guarded at the 02n batch size on the merged model, smoke-checked, run,
determinism pass, then the same checks as 04c. Its review flags need the zero-shot output of the same
documents at native: `ZS_TEST_DIR` = the validated `v15_<sha7>` folder of the zs path. No zero-shot
inference ever runs on this path. Output `submissions/v2n_<sha7>/`.

Flow: parameters -> Drive + secrets -> clone at the pin -> unzip data incl. test images (200 docs /
280 pages by file name) -> install (`uv sync --frozen --group vlm --group train`, no Paddle) ->
**NATIVE CHECKS (CPU, before anything runs: refuses, never advises)** -> ESTIMATE (T4 hours / CU,
UNVERIFIED) -> plan -> OCR (adopt the 04b record when `REUSE_OCR_FROM` is set; the OCR stage checks
the cached pages: resumable, 280/280 asserted, 5-page seeded re-OCR) -> [zs: smoke, batch, run,
determinism] / [ft: **resolution gate + adapter verification**, infer (merge, guard, smoke, run),
determinism] -> assemble with R1-R3 + validation (`predict_native finalize`: the 04b / 04c checks +
the native checks) -> **REVIEW FLAGS** -> banner VALIDATED / REJECTED.

What the native checks refuse (`python -m shipdoc.predict_native check`, then again by stage):
a config whose `max_pixels` is not 2,196,480 (or not the native name); a 02n run that is missing,
incomplete or whose manifest config hash is not the native config's (a 1260-token zero-shot run is
refused); no stored batch size (and a `BATCH_SIZE` different from the 02n run's on the zs path);
`ft`: an adapter trained at another resolution (`shipdoc.resmatch`, fail closed), a missing adapter
manifest, a `ZS_TEST_DIR` that is missing, not VALIDATED, not at the native config hash, not a
`v15_` full-mode non-fine-tuned run, without `test_predictions.json`, with a batch size other than
the 02n run's, or refused by the strict reuse decision; `zs`: a `REUSE_V0_DIR` whose traces the reuse
decision would ALLOW (it must refuse them on the config hash). The `ft` stages repeat the resolution
gate before `predict_ft` runs and stamp it into `ft_run.json`; `finalize` rejects an `ft` run without
that stamp. The flags refuse (nothing is computed): no `CALIBRATOR_FILE` (the default: it says to
refreeze at native with `scripts/freeze_calibrator.py --arm <zs|ft>`), a calibrator of the other arm,
a calibrator that does not carry the native config hash (`run_config_hash`; the 1260-token
`meta/calibrator_zs.json` carries none and is refused), and a submission whose manifest is not at the
native config hash.

Parameters (every required input is checked and refuses with a clear message): `PINNED_SHA`; `MODEL`;
`ZS_SHA7` (the sha7 of the 02n pin; the zero-shot folder is
`zeroshot500_qwen35_4b_img_only_native_<ZS_SHA7>`; the placeholder refuses); `BATCH_SIZE` (None = the
02n run's stored size, refused if absent); `ft` only: `TRAIN_SHA7` / `PRECISION` / `ADAPTER_DIR` (the
03n `final` folder, default `runs/ft_native_final_<TRAIN_SHA7>_<PRECISION>/final`; the placeholder
refuses) and `ZS_TEST_DIR` (default `submissions/v15_<PINNED_SHA[:7]>`, i.e. the zs run of the SAME
pin); `REUSE_V0_DIR` (zs: the v0 folder the reuse decision must refuse; None skips); `REUSE_OCR_FROM`
(default the 04b folder `submissions/v1_8833c73`: its OCR timing and re-check record are adopted, the
cache itself is the one on Drive); `REUSE_ACK_SPIKE_DIFF`; `CALIBRATOR_FILE` (None: the flags cell
refuses; relative paths are looked up in the clone, then on Drive); `FIELD_TARGET` / `DOC_TARGET`.

Merge precision (ft): the adapter's low-rank update is added to the fp16 base weights (fp32 sum
rounded to fp16); the merged model is not bit-identical to base + unmerged adapter. The whole of both
paths at native (batch size, VRAM, the merge, every timing) is UNVERIFIED on a GPU until run.

Output: `MyDrive/shipdoc-extract/submissions/v15_<sha7>/` (zs) or `v2n_<sha7>/` (ft) with
`test_predictions.json`, `review_flags.json`, `trace.jsonl`, `manifest.json`,
`validation_report.json`, `rules.jsonl`, `ocr_timing.json`; on a failed check the predictions are
`test_predictions.REJECTED.json`. Download by hand into `$SHIPDOC_SUBMISSIONS_DIR\\<folder>\\`. If the
session dies, Run all: every stage resumes. See notebooks/README.md.
"""

PARAMS = f"""# Parameters. PINNED_SHA = the repo commit this run is pinned to (full 40-char SHA).
PINNED_SHA = "{PINNED_SHA}"
MODEL = "zs"  # "zs" = v1.5, the base model, no adapter | "ft" = v2, the final native adapter
RUN_NAME = "testnative"
CONFIG = "qwen35_4b_img_only_native"  # KEYED, prompt v2, max_pixels 2196480: production
SPLIT = "test"
EXPECTED_DOCS = 200
EXPECTED_PAGES = 280
SMOKE_DOCS = "splits/smoke5.json"  # 5 dev docs picked by meta tags (shipdoc.smoke)
BENCH_DOCS = "splits/bench12.json"  # 12 dev pages for the batch guard / bench (shipdoc.bench)
SHARD = "0/1"  # fixed: one tab (the plan stage and the assemble step take it)
# sha7 of the 02n pin: the 02n run folder is zeroshot500_qwen35_4b_img_only_native_<ZS_SHA7>.
ZS_SHA7 = "FILL_ZS_SHA7"
ZS_RUN_DIR = f"zeroshot500_qwen35_4b_img_only_native_{{ZS_SHA7}}"  # the 02n run folder under runs/
BATCH_SIZE = None  # None = the batch size stored with the 02n run (refused if absent); or an int
# ft only: the NATIVE 03n final run (its folder is printed by the 03n banner) and the zs submission
TRAIN_SHA7 = "FILL_TRAIN_SHA7"  # sha7 of the 03n run that trained the final adapter. No default.
PRECISION = "bf16"  # bf16 (L4) | fp16 (T4): the suffix of the 03n run folder
# the final run's final/ folder: relative to MyDrive/shipdoc-extract, or an absolute path
ADAPTER_DIR = f"runs/ft_native_final_{{TRAIN_SHA7}}_{{PRECISION}}/final"
ZS_TEST_DIR = f"submissions/v15_{{PINNED_SHA[:7]}}"  # ft only: the VALIDATED zs run (same pin)
REUSE_V0_DIR = "submissions/v0_42b812b"  # zs: the v0 folder that MUST be refused; or None
REUSE_ACK_SPIKE_DIFF = False  # True accepts a spike.py diff beyond the allowlisted opt-in ones
CALIBRATOR_FILE = None  # None = no flags (refuses); else the native calibrator, see the README
FIELD_TARGET = 0.98  # precision target of the per-field-type tau (0.95 | 0.98 | 0.99)
DOC_TARGET = 0.98  # precision target of the document auto-accept tau (0.95 | 0.98)
REUSE_OCR_FROM = "submissions/v1_8833c73"  # the 04b folder whose OCR timing / re-check is adopted

if not PINNED_SHA or "FILL" in PINNED_SHA:
    raise ValueError("Set PINNED_SHA in the parameters cell (the pin commit fills it).")
if MODEL not in ("zs", "ft"):
    raise ValueError(f"MODEL must be 'zs' or 'ft', got {{MODEL!r}}")
if not (isinstance(ZS_SHA7, str) and len(ZS_SHA7) == 7
        and all(c in "0123456789abcdef" for c in ZS_SHA7)):
    raise ValueError(f"ZS_SHA7 must be the 7 hex characters of the 02n pin, got {{ZS_SHA7!r}}")
if BATCH_SIZE is not None and (isinstance(BATCH_SIZE, bool) or not isinstance(BATCH_SIZE, int)
                               or BATCH_SIZE < 1):
    raise ValueError(f"BATCH_SIZE must be None or an int >= 1, got {{BATCH_SIZE!r}}")
if PRECISION not in ("bf16", "fp16"):
    raise ValueError(f"PRECISION must be 'bf16' or 'fp16', got {{PRECISION!r}}")
if MODEL == "ft":
    if not (isinstance(TRAIN_SHA7, str) and len(TRAIN_SHA7) == 7
            and all(c in "0123456789abcdef" for c in TRAIN_SHA7)):
        raise ValueError(f"TRAIN_SHA7 must be 7 hex characters (03n pin), got {{TRAIN_SHA7!r}}")
    if not isinstance(ADAPTER_DIR, str) or not ADAPTER_DIR or "FILL" in ADAPTER_DIR:
        raise ValueError("ADAPTER_DIR must name the native final run's final/ folder on Drive.")
    if not isinstance(ZS_TEST_DIR, str) or not ZS_TEST_DIR or "FILL" in ZS_TEST_DIR:
        raise ValueError("ZS_TEST_DIR must be the validated zs submission folder (v15_<sha7>).")
if REUSE_V0_DIR is not None and not isinstance(REUSE_V0_DIR, str):
    raise ValueError(f"REUSE_V0_DIR must be None or a path string, got {{REUSE_V0_DIR!r}}")
if not isinstance(REUSE_ACK_SPIKE_DIFF, bool):
    raise ValueError(f"REUSE_ACK_SPIKE_DIFF must be True or False, got {{REUSE_ACK_SPIKE_DIFF!r}}")
if CALIBRATOR_FILE is not None and not (isinstance(CALIBRATOR_FILE, str) and CALIBRATOR_FILE):
    raise ValueError(f"CALIBRATOR_FILE must be None or a path string, got {{CALIBRATOR_FILE!r}}")
if FIELD_TARGET not in (0.95, 0.98, 0.99):
    raise ValueError(f"FIELD_TARGET must be 0.95, 0.98 or 0.99, got {{FIELD_TARGET!r}}")
if DOC_TARGET not in (0.95, 0.98):
    raise ValueError(f"DOC_TARGET must be 0.95 or 0.98, got {{DOC_TARGET!r}}")
if REUSE_OCR_FROM is not None and not isinstance(REUSE_OCR_FROM, str):
    raise ValueError(f"REUSE_OCR_FROM must be None or a path string, got {{REUSE_OCR_FROM!r}}")
SHA7 = PINNED_SHA[:7]
SUB_PREFIX = "v15" if MODEL == "zs" else "v2n"  # the submission folder of this path
SUB_NAME = f"{{SUB_PREFIX}}_{{SHA7}}"
RUN_ID = f"{{RUN_NAME}}_{{MODEL}}_{{SHA7}}"  # MyDrive/shipdoc-extract/runs/<RUN_ID>/; never a 04* name
CFG = f"configs/spike_{{CONFIG}}.yaml"
MODE = "run"  # the shared 04 cells take these (one tab, no shards)
SHARD_I, SHARD_K = 0, 1
SHARD_RUN_ID = RUN_ID
TAB = "0of1"
"""

_ZIPS_OLD = 'REQUIRED_ZIPS = ["data.zip", "assignment.zip", "ocr_cache.zip"]'
assert _ZIPS_OLD in base.MOUNT
# Image only: no OCR cache zip. data.zip holds data/ (incl. the test images and the dev labels the
# smoke gate and the guard use), assignment.zip the schema and the sample submission.
MOUNT = (
    base.MOUNT.replace(_ZIPS_OLD, 'REQUIRED_ZIPS = ["data.zip", "assignment.zip"]')
    + """SUBMISSIONS_DIR = DRIVE_DIR / "submissions"  # test predictions live here, not in the repo
SUBMISSION_DIR = SUBMISSIONS_DIR / SUB_NAME
META_DIR = RUNS_DIR / f"{RUN_ID}_meta"  # plan, smoke status, batch decision, OCR records, check
RUN_DIR = RUNS_DIR / RUN_ID
OCR_ROOT = DRIVE_DIR / "ocr_cache_test"  # <root>/paddleocr/test/<page>.json, shared with 04b
for _d in (SUBMISSIONS_DIR, META_DIR, SUBMISSION_DIR, OCR_ROOT):
    _d.mkdir(parents=True, exist_ok=True)
PLAN_PATH = META_DIR / "plan.json"
TEST_DOCS_PATH = META_DIR / "test_docs.json"  # ids only
SMOKE_STATUS_PATH = META_DIR / "smoke_status.json"
DECISION_PATH = META_DIR / "batch_decision.json"
CHECK_PATH = META_DIR / "native_check.json"


def _under_runs(p: str) -> Path:
    \"\"\"Absolute as given; 'runs/...' under MyDrive/shipdoc-extract; a bare name under runs/.\"\"\"
    if Path(p).is_absolute():
        return Path(p)
    return DRIVE_DIR / p if p.startswith("runs/") else RUNS_DIR / p


def _under_drive(p: str) -> Path:
    \"\"\"Absolute as given; else relative to MyDrive/shipdoc-extract.\"\"\"
    return Path(p) if Path(p).is_absolute() else DRIVE_DIR / p


ZS_RUN = _under_runs(ZS_RUN_DIR)  # the 02n run: batch size, config hash, the batch contract
DEV_RUN = ZS_RUN  # shipdoc predict assemble --dev-run-dir: the test batch must equal the 02n run's
ADAPTER = _under_drive(ADAPTER_DIR) if MODEL == "ft" else None
ZS_TEST = _under_drive(ZS_TEST_DIR) if MODEL == "ft" else None
V0_DIR = _under_drive(REUSE_V0_DIR) if (MODEL == "zs" and REUSE_V0_DIR) else None
OCR_FROM = _under_drive(REUSE_OCR_FROM) if REUSE_OCR_FROM else None
_required = [("zero-shot run (02n)", ZS_RUN, "manifest.json"),
             ("zero-shot run (02n)", ZS_RUN, "predictions.json")]
if MODEL == "ft":
    _required += [("final adapter folder", ADAPTER, "manifest.json"),
                  ("zs test submission (ZS_TEST_DIR)", ZS_TEST, "manifest.json")]
for label, path, need in _required:
    if not (path / need).is_file():
        raise FileNotFoundError(
            f"{label} {path} has no {need}. MODEL={MODEL!r} needs: ZS_SHA7 (the 02n pin)"
            + (", TRAIN_SHA7 / PRECISION / ADAPTER_DIR (the 03n final run) and ZS_TEST_DIR (the "
               "validated v15_<sha7> folder: run this notebook with MODEL = 'zs' first)"
               if MODEL == "ft" else "") + " in the parameters cell."
        )
print("run_id:", RUN_ID, "| model:", MODEL, "| batch size:", BATCH_SIZE or "from the 02n run")
print("zero-shot run (02n):", ZS_RUN)
print("final adapter      :", ADAPTER, "| zs test folder:", ZS_TEST)
print("OCR cache folder   :", OCR_ROOT, "| adopt OCR record from:", OCR_FROM)
print("Drive run folder   :", RUN_DIR)
print("Drive submission   :", SUBMISSION_DIR)
"""
)

CHECKS = """# NATIVE CHECKS (CPU, nothing runs on the GPU before they pass). `shipdoc.predict_native check`
# REFUSES, with the reason, unless: the config is the native one; the 02n run is complete and
# carries the native config hash; its stored batch size resolves (the zs test batch must EQUAL it);
# ft: the adapter passes the resolution gate (shipdoc.resmatch, fail closed) and ZS_TEST_DIR is a
# VALIDATED v15 folder at the native hash that the strict reuse decision accepts; zs: the v0 folder
# (when present) is REFUSED for reuse on the config hash. It writes the resolved batch size.
check_cmd = [PY, "-m", "shipdoc.predict_native", "check", "--model", MODEL, "--config", CFG,
             "--zs-run-dir", str(ZS_RUN), "--out", str(CHECK_PATH)]
if BATCH_SIZE is not None:
    check_cmd += ["--batch-size", str(BATCH_SIZE)]
    print("=" * 78)
    print(f"MANUAL BATCH SIZE {BATCH_SIZE}: the zs path REFUSES a size other than the 02n run's;")
    print("the ft path lets the guard on the merged model decide (it may fall back to 1).")
    print("=" * 78)
if MODEL == "ft":
    check_cmd += ["--adapter-dir", str(ADAPTER), "--zs-test-dir", str(ZS_TEST)]
elif V0_DIR is not None and V0_DIR.is_dir():
    check_cmd += ["--v0-dir", str(V0_DIR)]
else:
    print("zs: no v0 folder to prove the reuse refusal on (REUSE_V0_DIR); the VLM runs anyway.")
if REUSE_ACK_SPIKE_DIFF:
    check_cmd.append("--ack-spike-diff")
    print("REUSE_ACK_SPIKE_DIFF = True: a spike.py diff beyond the allowlist is accepted.")
rc, tail = run_stream(check_cmd)
if rc != 0:
    raise RuntimeError(
        f"NATIVE CHECK REFUSED (exit {rc}): {tail[-3:]}. Nothing was run; read the refusal above "
        "(counts, hashes and names only), fix the parameter it names and Run all again."
    )
check = json.loads(CHECK_PATH.read_text())
BATCH = int(check["batch_size"])  # the 02n run's stored size (or the ft override)
FLAGS_ZS_BATCH = int(check.get("zs_test_batch_size") or BATCH)  # what the v15 run was decoded at
print(f"NATIVE CHECKS PASSED: model {MODEL}, batch size {BATCH} ({check['batch_source']}), "
      f"config hash {check['config_hash']}")
"""

ESTIMATE = """# ESTIMATE (UNVERIFIED) of the GPU session of this path: T4 hours and compute units, low / high
# pace. Nothing in it is measured at native: low = the 02 batch-8 pace (MEASURED at 1260 tokens)
# x the 06 sweep's native cost ratio, valid only if the batch size fits; high = the sweep's
# MEASURED native batch-1 pace (shipdoc.nativerun). ASSUMED 120 s model load, 60 s merge. No
# confirmation prompt (Run all must not block): interrupt the run yourself if it is too high.
est_cmd = [PY, "-m", "shipdoc.predict_native", "estimate", "--model", MODEL, "--config", CFG,
           "--zs-run-dir", str(ZS_RUN), "--smoke-docs", SMOKE_DOCS]
if BATCH_SIZE is not None:
    est_cmd += ["--batch-size", str(BATCH_SIZE)]
rc, tail = run_stream(est_cmd)
assert rc == 0, f"estimate failed (exit {rc}): {tail[-3:]}"
print("THIS RUN: model", MODEL, "| batch size", BATCH, "| OCR record adopted from:", REUSE_OCR_FROM)
"""


def _body(src: str) -> str:
    """`src` without its leading comment block (the header is replaced by a shorter one)."""
    lines = src.splitlines(True)
    i = 0
    while i < len(lines) and lines[i].startswith("#"):
        i += 1
    return "".join(lines[i:])


def _edit(src: str, edits: list[tuple[str, str]], what: str) -> str:
    for old, new in edits:
        assert src.count(old) == 1, f"{what} changed: {old[:70]!r}"
        src = src.replace(old, new)
    return src


def _guard(src: str, cond: str, what: str) -> str:
    """`src` under ``if <cond>:`` (the other path's cells print a skip line and do nothing).

    The indent must keep every line within 100 columns and `src` must hold no multi-line string
    literal (the indent would end up inside it): both asserted.
    """
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        assert not (tok.type == tokenize.STRING and tok.start[0] != tok.end[0]), (what, tok.string)
    body = "".join(("    " + ln) if ln.strip() else ln for ln in src.splitlines(True))
    long = [ln for ln in body.splitlines() if len(ln) > 100]
    assert not long, (what, long[:2])
    said = cond.replace('"', "'")
    return f'if {cond}:\n{body}\nelse:\n    print("skipped ({what}): {said} is false")\n'


# ---- zs path: the 04 stages at the native config (own header comments, the 04 bodies)

ZS_SMOKE = _guard(
    "# MANDATORY smoke gate (the same run and 7 checks as notebook 02, 5 DEV docs) at the native\n"
    "# config: the status file is bound to this code, config and model revision; the test run\n"
    "# refuses to start unless it says `passed`. A failure raises.\n" + _body(p04.SMOKE),
    'MODEL == "zs"',
    "smoke gate",
)

ZS_BATCH = _guard(
    """# BATCH SIZE = the one stored with the 02n run (the bench there measured it at native). MANUAL
# on purpose: `shipdoc predict batch --batch-size` skips a second bench (a stored bench result
# applies only to its own code SHA) and still enforces the batch-size contract against the 02n
# run (`--dev-run-dir`): greedy outputs only compare at equal batch size. A mismatch raises.
if not SMOKE_PASSED:
    raise RuntimeError("The smoke gate has not passed: batch size and test run are NOT run.")
BENCH_DIR = RUNS_DIR / f"bench_{CONFIG}_{SHA7}"
batch_cmd = [PY, "-m", "shipdoc", "predict", "batch", "--config", CFG,
             "--bench-docs", BENCH_DOCS, "--bench-dir", str(BENCH_DIR),
             "--decision", str(DECISION_PATH), "--dev-run-dir", str(DEV_RUN),
             "--batch-size", str(BATCH)]
rc, tail = run_stream(batch_cmd)
if rc != 0:
    raise RuntimeError(f"BATCH SIZE REFUSED (exit {rc}): {tail[-3:]}. Test run NOT started.")
decision = json.loads(DECISION_PATH.read_text())
assert int(decision["batch_size"]) == BATCH, "the decision's batch size is not the 02n run's"
print(f"BATCH SIZE FOR THE TEST RUN: {BATCH} ({decision['source']}); dev contract: "
      f"{decision['contract']}")
""",
    'MODEL == "zs"',
    "batch size",
)

ZS_RUN = _guard(
    "# Test run over the 200 documents, one subprocess, RESUMABLE per doc (as 04): the CLI\n"
    "# appends one line to trace.jsonl and rewrites predictions.json / progress.json after every\n"
    "# document; a resume skips the doc_ids already traced. Refuses unless the smoke gate passed.\n"
    + _body(p04.RUN),
    'MODEL == "zs"',
    "zs test run",
)

ZS_DETERMINISM = _guard(
    "# DETERMINISM (as 04): 5 seeded test documents are decoded AGAIN in a fresh process,\n"
    "# replaying the exact generate calls of the first pass; the serialized predictions must be\n"
    "# byte-identical. Counts only. A difference raises: nothing is assembled as submittable.\n"
    + _body(p04.DETERMINISM),
    'MODEL == "zs"',
    "zs determinism pass",
)

# ---- ft path: the 04c stages behind the resolution gate

FT_VERIFY = _guard(
    "# NATIVE: `shipdoc.predict_native verify` = the RESOLUTION CHECK (an adapter not trained at\n"
    "# the inference resolution is refused, fail closed), then the unchanged final-adapter\n"
    "# verification of 04c (one row per check). A failed check raises: NOTHING after it runs.\n"
    + _edit(
        _body(v2.VERIFY),
        [
            ('"shipdoc.predict_ft", "verify"', '"shipdoc.predict_native", "verify"'),
            (
                '    print(f"training code {TRAIN_SHA[:7]} != pin {PINNED_SHA[:7]}: files that decide the model\'s "\n          "input and output, changed between the two (empty = unchanged):")',
                '    print(f"training code {TRAIN_SHA[:7]} != pin {PINNED_SHA[:7]}: files that decide the "\n          "model\'s input and output, changed between the two (empty = unchanged):")',
            ),
            (
                '"configs/finetune_qwen35_4b.yaml", CFG])',
                '"configs/finetune_qwen35_4b_native.yaml", CFG])',
            ),
        ],
        "colab_build_predict_v2.VERIFY",
    ),
    'MODEL == "ft"',
    "adapter verification",
)

FT_INFER = _guard(
    "# INFER in ONE process (the merged model never leaves it), RESUMABLE per document. NATIVE:\n"
    "# the resolution gate again, then the unchanged 04c stages, printed as they happen: VERIFY\n"
    "# (again), MERGE (fp16 base + the adapter, 200 modules required), GUARD (12 bench pages at\n"
    "# batch 1 and at the chosen size on the MERGED model; any difference or an out-of-memory ->\n"
    "# batch 1, recorded; a resume keeps the stored decision), SMOKE (the gate of 04, 5 dev docs,\n"
    "# on the merged model), RUN (the 200 test documents). The gate is stamped into ft_run.json,\n"
    "# which `finalize` requires.\n"
    + _edit(
        _body(v2.INFER),
        [
            ('"shipdoc.predict_ft", "infer"', '"shipdoc.predict_native", "infer"'),
            ('"--batch-size", str(BATCH_SIZE)', '"--batch-size", str(BATCH)'),
            (
                '        f"{read_progress().get(\'done\')}/{EXPECTED_DOCS} docs; last output: {tail[-3:]}. Everything "\n        "done so far is on Drive: Run all again resumes. If it stops at the same point, "\n        "report back."',
                '        f"{read_progress().get(\'done\')}/{EXPECTED_DOCS} docs; last output: {tail[-3:]}. "\n        "Everything done so far is on Drive: Run all again resumes. If it stops at the same "\n        "point, report back."',
            ),
        ],
        "colab_build_predict_v2.INFER",
    ),
    'MODEL == "ft"',
    "ft inference",
)

FT_DETERMINISM = _guard(
    "# DETERMINISM: 5 seeded test documents decoded AGAIN in a fresh process (new load, new\n"
    "# merge, new CUDA context), replaying the exact generate calls of the first pass;\n"
    "# byte-identical serialized predictions required. Behind the resolution gate. Counts only.\n"
    + _edit(
        _body(v2.DETERMINISM),
        [('"shipdoc.predict_ft", "determinism"', '"shipdoc.predict_native", "determinism"')],
        "colab_build_predict_v2.DETERMINISM",
    ),
    'MODEL == "ft"',
    "ft determinism pass",
)

ASSEMBLE = """# ASSEMBLE + VALIDATION (CPU). `shipdoc predict assemble` runs the post-processing v1 (R1, R2, R3
# ON, the OCR cache passed, the frozen meta/slot_shapes.json) and every blocking check of 04: run
# complete, exactly the 200 test ids, 200 docs / 280 pages, JSON Schema, determinism, code SHA clean
# and equal to PINNED_SHA, model revision / config hash / seed 42 / prompt v2, real model and stack
# (zs also: the batch size equals the 02n run's). `shipdoc.predict_native finalize` then adds the
# 04b checks (rule switches all on, no R2 skipped, shapes sha256 = meta/slot_shapes.json, OCR cache
# complete, OCR re-check), for ft the 04c checks, and the native checks (config hash = native, no
# adapter on zs / the resolution gate stamped on ft + the validated v15 folder, no key outside
# doc_type / header / line_items). The manifest says `submission` = v15_<sha7> / v2n_<sha7>. A
# failure moves the predictions to test_predictions.REJECTED.json.
asm_cmd = [PY, "-m", "shipdoc", "predict", "assemble", "--config", CFG, "--run-id", RUN_ID,
           "--out-dir", str(SUBMISSION_DIR), "--schema", str(SCHEMA_PATH),
           "--expect-docs", str(EXPECTED_DOCS), "--expect-pages", str(EXPECTED_PAGES),
           "--expect-code-sha", PINNED_SHA, "--require-stack",
           "--smoke-status", str(SMOKE_STATUS_PATH), "--ocr-cache", str(OCR_ROOT)]
if SAMPLE_PATH.is_file():
    asm_cmd += ["--sample-submission", str(SAMPLE_PATH)]
if MODEL == "zs":
    asm_cmd += ["--dev-run-dir", str(DEV_RUN)]  # the batch contract against the 02n run
rc, tail = run_stream(asm_cmd)
OCR_RECHECK_PATH = META_DIR / "ocr_recheck.json"
fin_cmd = [PY, "-m", "shipdoc.predict_native", "finalize", "--model", MODEL, "--config", CFG,
           "--out-dir", str(SUBMISSION_DIR), "--run-dir", str(RUN_DIR),
           "--recheck", str(OCR_RECHECK_PATH), "--expect-pages", str(EXPECTED_PAGES)]
if MODEL == "ft":
    fin_cmd += ["--zs-test-dir", str(ZS_TEST)]
rc_fin, tail_fin = run_stream(fin_cmd)
if rc_fin != 0:
    raise RuntimeError(
        f"VALIDATION FAILED (assemble exit {rc}, finalize exit {rc_fin}): the files in "
        f"{SUBMISSION_DIR} are NOT submittable (test_predictions.REJECTED.json). Read "
        "validation_report.json there (counts and key paths only) and report back; do not submit."
    )
"""

FLAGS = """# REVIEW FLAGS (CPU), a SEPARATE file: review_flags.json in the submission folder, never part of
# test_predictions.json. CALIBRATOR_FILE = None (the default) REFUSES here: refreeze the calibrator
# at native first. `shipdoc.predict_native flags` refuses a calibrator of the other arm, one that
# does not carry the native config hash (`run_config_hash`) and a submission at another hash; for
# ft it also needs the validated v15 folder (the agreement features). Then `shipdoc.flags` writes
# probabilities, accept / review flags and thresholds (no extracted value) and `check-flags` checks
# the file: exactly the 200 ids, no value string, thresholds and the calibrator sha256 recorded,
# no flag key inside the predictions. The flags are advisory: a failure here does not withhold the
# validated predictions.
if CALIBRATOR_FILE is None:
    raise RuntimeError(
        "CALIBRATOR_FILE is None: the review flags are NOT computed (the predictions in "
        f"{SUBMISSION_DIR} are VALIDATED and on Drive). Refreeze at native with "
        f"`scripts/freeze_calibrator.py --arm {MODEL}` after 02n and calibrate_v2 on the native "
        "run (zs) / the three native OOF runs and calibrate_v3 (ft), then set CALIBRATOR_FILE."
    )
CALIBRATOR = Path(CALIBRATOR_FILE)
if not CALIBRATOR.is_absolute():  # the clone first (committed artifact), then Drive
    in_clone = REPO / CALIBRATOR_FILE
    CALIBRATOR = in_clone if in_clone.is_file() else DRIVE_DIR / CALIBRATOR_FILE
if not CALIBRATOR.is_file():
    raise FileNotFoundError(f"CALIBRATOR_FILE {CALIBRATOR_FILE} is in neither the clone nor Drive")
if not (SUBMISSION_DIR / "test_predictions.json").is_file():
    raise RuntimeError("no validated test_predictions.json: the flags are not computed")
flags_cmd = [PY, "-m", "shipdoc.predict_native", "flags", "--model", MODEL, "--config", CFG,
             "--submission-dir", str(SUBMISSION_DIR), "--calibrator", str(CALIBRATOR),
             "--ocr-cache", str(OCR_ROOT), "--batch-size", str(FLAGS_ZS_BATCH),
             "--field-target", str(FIELD_TARGET), "--doc-target", str(DOC_TARGET),
             "--expect-docs", str(EXPECTED_DOCS)]
if MODEL == "ft":
    flags_cmd += ["--zs-test-dir", str(ZS_TEST)]
if REUSE_ACK_SPIKE_DIFF:
    flags_cmd.append("--ack-spike-diff")
rc, tail = run_stream(flags_cmd)
if rc != 0:
    raise RuntimeError(
        f"REVIEW FLAGS REFUSED (exit {rc}): {tail[-2:]}. The predictions in {SUBMISSION_DIR} are "
        "VALIDATED and on Drive; only the flags file is missing."
    )
check_cmd = [PY, "-m", "shipdoc.predict_native", "check-flags", "--config", CFG,
             "--out-dir", str(SUBMISSION_DIR), "--calibrator", str(CALIBRATOR),
             "--field-target", str(FIELD_TARGET), "--doc-target", str(DOC_TARGET),
             "--expect-docs", str(EXPECTED_DOCS)]
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
LOCAL_DEST = BS.join(["$SHIPDOC_SUBMISSIONS_DIR", SUB_NAME]) + BS
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
    kind = "zero-shot, base model" if MODEL == "zs" else "fine-tuned, final native adapter"
    print(f"TEST PREDICTIONS {SUB_PREFIX} ({kind}, NATIVE + R1-R3) " + verdict)
    print(bar)
    print(f"schema ok       : {report['schema_ok']}")
    print(f"docs found      : {report['docs_found']}")
    print(f"determinism ok  : {report['determinism_ok']}")
    for name, c in report["checks"].items():
        print(f"  {'PASS' if c['ok'] else 'FAIL'}  {name}: {c.get('detail', '')}")
    nat = manifest.get("native") or {}
    print(f"native config {nat.get('config')} hash {nat.get('config_hash')} max_pixels "
          f"{nat.get('max_pixels')}; model switch {nat.get('model')}")
    if MODEL == "ft":
        ft = manifest.get("ft") or {}
        print(f"final adapter {str(ft.get('adapter_sha256'))[:12]}  training code "
              f"{str(ft.get('train_code_sha'))[:7]}  pin {PINNED_SHA[:7]}  precision "
              f"{ft.get('train_precision')}")
        mg, gd = ft.get("merge") or {}, ft.get("guard") or {}
        print(f"merge: {mg.get('n_lora_modules_merged')} modules, {mg.get('merge_dtype')}, "
              f"{mg.get('load_and_merge_s')} s; guard ran={gd.get('ran')} ok={gd.get('ok')} "
              f"fallback_to_1={gd.get('fallback_to_1')}")
        zt = nat.get("zs_test") or {}
        print(f"zero-shot traces: {zt.get('dir')} (batch {zt.get('batch_size')}, code "
              f"{str(zt.get('code_sha'))[:7]})")
    print(f"code {str(manifest.get('code_sha'))[:12]}  model {manifest.get('model', {}).get('id')} "
          f"@ {str(manifest.get('model', {}).get('revision'))[:12]}  config "
          f"{manifest.get('config', {}).get('hash')}")
    print(f"batch size {manifest.get('batch_size')} ({manifest.get('batch_size_source')}); "
          f"contract: {manifest.get('batch_size_contract')}; "
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
        ("code", v2.OCR_ESTIMATE),
        ("code", v2.INSTALL),
        ("code", CHECKS),
        ("code", ESTIMATE),
        ("code", p04.PLAN),
        ("code", v2.ADOPT_OCR),
        ("code", v2.OCR_STAGE),
        ("code", ZS_SMOKE),
        ("code", ZS_BATCH),
        ("code", ZS_RUN),
        ("code", ZS_DETERMINISM),
        ("code", FT_VERIFY),
        ("code", FT_INFER),
        ("code", FT_DETERMINISM),
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
