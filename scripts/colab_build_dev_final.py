"""Generate notebooks/05b_dev_final.ipynb (the FINAL adapter on the 100 dev documents, T4, fp16).

A thin wrapper like 04c / 05: every cell only orchestrates (Drive, secrets, clone, install, one
subprocess per stage); the logic is in ``python -m shipdoc.devfinal`` (src/shipdoc/devfinal.py).
Cells and helpers that do not change are imported READ-ONLY from scripts/colab_build_notebook.py,
scripts/colab_build_oof.py (05) and scripts/colab_build_predict_v1.py (04b) and edited by
exact-string replacement (asserted), so the notebooks cannot drift apart silently; none of them is
modified.

tests/test_notebook_dev_final.py checks the committed notebook equals this script's output.

Run: uv run python scripts/colab_build_dev_final.py
"""

# ruff: noqa: E501  # cell sources are verbatim notebook text

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "archive" / "05b_dev_final.ipynb"


def _load(name: str, path: Path):  # noqa: ANN202 - a module object
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


base = _load("colab_build_notebook", ROOT / "scripts" / "colab_build_notebook.py")
oofb = _load("colab_build_oof", ROOT / "scripts" / "colab_build_oof.py")
v1 = _load("colab_build_predict_v1", ROOT / "scripts" / "colab_build_predict_v1.py")

# UNPINNED until GG pins it: the pin commit replaces this placeholder with the full 40-char SHA of
# the pushed code commit and regenerates the notebook. Until then the parameters cell RAISES.
PINNED_SHA = "SUPERSEDED-NOT-PINNED"

TITLE = """# 05b - the FINAL adapter on the 100 dev documents: the official seen-layout evaluation (thin wrapper)

**SUPERSEDED (2026-10-03) by `08_dev_final` (native resolution, `max_pixels` 2,196,480). Do not run this notebook: it is the 1260-token path and it is not pinned (`PINNED_SHA` is `SUPERSEDED-NOT-PINNED`; the first cell raises).**

Runs the `final` LoRA adapter of notebook 03 (trained on the 400 `train_*` documents; **no dev and no
test document**) on the 100 `dev_*` documents with the production pipeline, scores it with the
official scorer and compares it with the 02 zero-shot run restricted to the same 100 documents
(spec Phase 4 item 6 and section 8). T4, fp16. Notebook 05 refuses `final` adapters (it is the OOF
check of fold adapters) and 04c covers only the test documents, hence this notebook.

**What the number means.** Dev layouts (suppliers, templates) also appear in train: the official dev
OVERALL measures **seen layouts**. It does not measure unseen suppliers (that is the out-of-fold /
G4 comparison of notebook 05, spec section 11) and it must not be quoted as such.

Production inference config: Qwen3.5-4B, IMAGE ONLY, KEYED output, prompt v2, greedy, xgrammar, seed
42, logprobs on, the shared coerce / schema-repair writer. Labels are train / dev labels only; no
test file is read (the notebook never lists a test folder).

Flow: parameters -> Drive + secrets -> clone at the pin -> unzip data + assignment + OCR cache (the
100 dev docs' OCR pages are checked against `SHA256SUMS`) -> install (vlm + train groups: peft comes
from the lock; no Paddle) -> **FINAL ADAPTER VERIFICATION (fails closed; printed table)** -> plan
(the 100 ids must equal the dev ids of `splits/zeroshot500.json`, the held-out ids of the final stage
of `splits/folds.json` and the dev label files) -> **ESTIMATE of T4 hours and compute units**
(labelled ESTIMATE) -> **INFER**: merge the adapter into the fp16 weights, **guard** (the 12 bench
pages at batch 1 and at the chosen batch size on the MERGED model must be byte-identical, else batch
1, recorded), then the resumable per-document inference on the 100 dev documents -> **SCORE + PAIRED
COMPARISON** (CPU) -> banner.

Comparison (dev documents only): FT vs ZS, under three arms. (1) PRIMARY: R1 + R2 + R3 with the R3
slot shapes learned from the TRAIN gold only (no dev gold anywhere). (2) R1 + R2 + R3 with the frozen
shipping shapes `meta/slot_shapes.json`, learned from all 500 train+dev documents: IN-SAMPLE on dev,
therefore optimistic for both models; reported because it is what ships. (3) Raw model output, no
rules (secondary). R2 reads the OCR of the dev pages from the OCR cache zip; a dev waybill whose OCR
is missing or incomplete stops the comparison. Per arm and subset (all, invoices, waybills, scanned,
digital): OVERALL, header accuracy, row F1, fully-correct documents, false-fill rate with 95% CIs
(doc-level bootstrap, 2000 resamples, seed 42, the unmodified scorer), the paired FT - ZS deltas, and
the over-null and false-fill cell counts (`shipdoc.oof.over_null_counts`, per field too). Aggregates
only: no document id or value is written or printed.

What the verification refuses: a smoke / fold adapter (stage other than `final`); a fold id; a
missing, dirty or unreachable training code SHA (a SHA different from this pin is a warning); a
base-model repo / revision other than the production config's; LoRA bookkeeping other than r=16 / 200
modules / 30,474,240 trainable parameters; a manifest without its training document ids; a training
set that is not exactly the `train_*` documents (a dev or test id in it is a refusal of its own);
held-out ids that are not the dev documents; a `manifest_hash` that does not match the ids; inference
keys (model repo and revision, adapter name, max_pixels, prompt version, output format) that differ
from the production config; weight files whose sha256 differs from the manifest.

Merge precision: the adapter's low-rank update is added to the fp16 base weights (fp32 sum rounded to
fp16); updates below the fp16 rounding step of a weight are lost, so the merged model is not
bit-identical to base + unmerged adapter. The merge and the batch-8 byte identity on the merged model
are UNVERIFIED on a GPU until this notebook has run.

Parameters: `PINNED_SHA` (UNPINNED: the cell raises until the pin commit fills it); `TRAIN_SHA7` and
`ADAPTER_DIR` (the Drive folder `<final run>/final` of notebook 03; default pattern
`runs/ft_final_<TRAIN_SHA7>_bf16/final`, set TRAIN_SHA7 or ADAPTER_DIR by hand when the training run
was pinned at another commit or ran in fp16); `ZS_RUN_DIR` (the 02 zero-shot run folder:
predictions.json, trace.jsonl, manifest.json, bench_result.json); `BATCH_SIZE` (None = the batch size
stored with the 02 run, refused if absent; an int overrides and is said loudly); `USE_WANDB`.

Output: `MyDrive/shipdoc-extract/runs/devfinal_<sha7>/` with `predictions.json`, `trace.jsonl`,
`manifest.json`, `devfinal_compare.json`, `devfinal_compare.md` (aggregates only). Download by hand
into `$SHIPDOC_RUNS_DIR\\devfinal_<sha7>\\`. If the session dies, Run all: setup repeats, verification
is cheap, the inference resumes after the last document in `trace.jsonl` and keeps the stored batch
decision. See notebooks/README.md.
"""

PARAMS = f"""# Parameters. PINNED_SHA = the repo commit this run is pinned to (full 40-char SHA).
PINNED_SHA = "{PINNED_SHA}"
TRAIN_SHA7 = PINNED_SHA[:7]  # sha7 of the 03 run that trained the final adapter; set by hand if not
# The final run's final/ folder: relative to MyDrive/shipdoc-extract, or an absolute path.
ADAPTER_DIR = f"runs/ft_final_{{TRAIN_SHA7}}_bf16/final"
ZS_RUN_DIR = "zeroshot500_qwen35_4b_img_only_keyed_42b812b"  # the 02 run folder under runs/
BATCH_SIZE = None  # None = the batch size stored with the 02 run (refused if absent); int overrides
USE_WANDB = False  # off by default; True logs metrics only (no images / values) to the project
WANDB_PROJECT = "shipdoc-extract-debug"  # must be a PRIVATE project
CONFIG = "qwen35_4b_img_only"  # KEYED output format, prompt v2, image only: the production pick
SPLIT = "dev"  # the 100 dev documents: the official seen-layout evaluation
DEV_DOCS = "splits/dev100.json"  # = the dev ids of ZS500_DOCS = the final stage's held-out ids
ZS500_DOCS = "splits/zeroshot500.json"
BENCH_DOCS = "splits/bench12.json"  # 12 dev pages for the batch guard (shipdoc.bench)
EXPECTED_DOCS = 100

if not PINNED_SHA or "FILL" in PINNED_SHA or "SUPERSEDED" in PINNED_SHA:
    raise ValueError("PINNED_SHA is not set: this archived notebook is superseded, not runnable.")
if not (isinstance(TRAIN_SHA7, str) and len(TRAIN_SHA7) == 7
        and all(c in "0123456789abcdef" for c in TRAIN_SHA7)):
    raise ValueError(f"TRAIN_SHA7 must be 7 lowercase hex characters, got {{TRAIN_SHA7!r}}")
if BATCH_SIZE is not None and (isinstance(BATCH_SIZE, bool) or not isinstance(BATCH_SIZE, int)
                               or BATCH_SIZE < 1):
    raise ValueError(f"BATCH_SIZE must be None or an int >= 1, got {{BATCH_SIZE!r}}")
if not isinstance(USE_WANDB, bool):
    raise ValueError(f"USE_WANDB must be True or False, got {{USE_WANDB!r}}")
SHA7 = PINNED_SHA[:7]
RUN_NAME = "devfinal"
RUN_ID = f"devfinal_{{SHA7}}"  # MyDrive/shipdoc-extract/runs/<RUN_ID>/
"""

# The base mount cell unchanged: data.zip, assignment.zip AND ocr_cache.zip (R2 reads the dev OCR).
MOUNT = (
    base.MOUNT
    + """def _under_runs(p: str) -> Path:
    \"\"\"Absolute as given; 'runs/...' under MyDrive/shipdoc-extract; a bare name under runs/.\"\"\"
    if Path(p).is_absolute():
        return Path(p)
    return DRIVE_DIR / p if p.startswith("runs/") else RUNS_DIR / p


ADAPTER = _under_runs(ADAPTER_DIR)
ZS_RUN = _under_runs(ZS_RUN_DIR)
OUT_DIR = RUNS_DIR / RUN_ID
for label, path, need in (("adapter folder", ADAPTER, "manifest.json"),
                          ("zero-shot run", ZS_RUN, "predictions.json"),
                          ("zero-shot run", ZS_RUN, "trace.jsonl")):
    if not (path / need).is_file():
        raise FileNotFoundError(f"{label} {path} has no {need}: set ADAPTER_DIR / TRAIN_SHA7 / "
                                "ZS_RUN_DIR in the parameters cell.")
print("run_id:", RUN_ID, "| batch size:", BATCH_SIZE or "from the 02 run",
      "| W&B:", "ON" if USE_WANDB else "off")
print("final adapter   :", ADAPTER)
print("zero-shot run   :", ZS_RUN)
print("Drive output    :", OUT_DIR)
"""
)

SECRETS = base.ZS_SECRETS

UNZIP = """import hashlib
import json
import shutil
import zipfile

LOCAL_ZIPS = Path("/content/zips")
LOCAL_ZIPS.mkdir(exist_ok=True)
CONTENT = Path("/content")
# data.zip holds data/, assignment.zip assignment/, ocr_cache.zip ocr_cache/; all unzip to /content
# (not into the clone) and the CLI is pointed at them with the SHIPDOC_* variables below.
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
assert (ASSIGNMENT_DIR / "score.py").is_file(), "assignment.zip has no score.py (official scorer)"
# Only the dev OCR pages are read (R2 of the dev waybills): verify those against SHA256SUMS.
_sums = [ln.partition("  ") for ln in (OCR_CACHE / "SHA256SUMS").read_text().splitlines()]
_dev_sums = [(d, rel) for d, _, rel in _sums if rel.startswith("paddleocr/dev/")]
assert _dev_sums, "ocr_cache/SHA256SUMS lists no dev page"
_bad = [rel for d, rel in _dev_sums
        if not (OCR_CACHE / rel).is_file()
        or hashlib.sha256((OCR_CACHE / rel).read_bytes()).hexdigest() != d]
if _bad:
    raise RuntimeError(f"ocr_cache SHA256SUMS: {len(_bad)} bad/missing dev pages, e.g. {_bad[:3]}")
print(f"dev OCR pages OK: {len(_dev_sums)} files verified")
_dev = json.loads((REPO / DEV_DOCS).read_text(encoding="utf-8"))
assert len(_dev) == EXPECTED_DOCS and all(d.startswith("dev_") for d in _dev), "not 100 dev ids"
_missing = [d for d in _dev if not (DATA_DIR / "dev" / "labels" / f"{d}.json").is_file()]
assert not _missing, f"{len(_missing)} dev documents without a label file"
assert (REPO / BENCH_DOCS).is_file(), f"{BENCH_DOCS} missing from the clone"
print(f"data OK: {len(_dev)} dev documents; labels are train / dev only, no test file is read")
"""

# Base ENV / run_stream / uv sync of 05 (vlm + train groups: peft comes from the lock), edited by
# exact-string replacement: the OCR cache is the one R2 reads, and 04c's pycountry stack probe.
_inst = oofb.INSTALL
_INSTALL_EDITS = [
    (
        '    "SHIPDOC_OCR_CACHE": "/content/ocr_cache",  # unused by img_only; never read\n',
        '    "SHIPDOC_OCR_CACHE": str(OCR_CACHE),  # the dev OCR pages that R2 reads\n',
    ),
    ("('transformers', 'xgrammar', 'peft')", "('transformers', 'xgrammar', 'peft', 'pycountry')"),
]
for _old, _new in _INSTALL_EDITS:
    assert _old in _inst, f"colab_build_oof.INSTALL changed: {_old[:50]!r}"
    _inst = _inst.replace(_old, _new)
INSTALL = _inst + v1.NO_PADDLE

VERIFY = """# FINAL ADAPTER MANIFEST VERIFICATION. Fails closed: any failed check raises here and NOTHING after
# this cell runs. Prints one row per check (expected, found, PASS / FAIL), the training code SHA
# next to this notebook's pin and the number of training documents (counts only, no ids). A
# training SHA that differs from the pin is a WARNING (the diff of the files that matter is shown
# below); an unreachable SHA, a smoke / fold adapter, a missing or wrong training-document list or
# a changed inference key is a REFUSAL.
CFG = f"configs/spike_{CONFIG}.yaml"
verify_cmd = [PY, "-m", "shipdoc.devfinal", "verify", "--adapter-dir", str(ADAPTER),
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

PLAN = """# PLAN (CPU). The 100 ids of splits/dev100.json must equal (a) the dev ids of
# splits/zeroshot500.json, (b) the held-out ids of the `final` stage of splits/folds.json (the
# documents the final adapter is documented not to have trained on) and (c) the dev label files;
# any difference raises. Counts documents and pages from the label files; writes the plan.
OUT_DIR.mkdir(parents=True, exist_ok=True)
plan_cmd = [PY, "-m", "shipdoc.devfinal", "plan", "--dev-docs", str(REPO / DEV_DOCS),
            "--zs500-docs", str(REPO / ZS500_DOCS), "--out", str(OUT_DIR / "devfinal_plan.json")]
rc, tail = run_stream(plan_cmd)
if rc != 0:
    raise RuntimeError(f"PLAN REFUSED (exit {rc}): {tail[-3:]}. The dev ids are not the dev "
                       "documents; nothing runs.")
"""

ESTIMATE = """# T4-hour and compute-unit (CU) ESTIMATE BEFORE the merge and the inference: read it. Pages are
# counted from the label files of the 100 dev documents; the batch size is the one that will be
# used (the 02 run's, or BATCH_SIZE). Includes model load, merge and the batch guard (12 bench
# pages at batch 1 and at the chosen size). Nothing in it is measured (scripts/gpu_estimate.py:
# ASSUMED batching efficiency 100 / 75 / 50%, ASSUMED 60 s merge). No confirmation prompt (Run all
# must not block): interrupt the run yourself if it is too high.
est_cmd = [PY, "-m", "shipdoc.devfinal", "estimate", "--config", CFG,
           "--zs-run-dir", str(ZS_RUN)]
if BATCH_SIZE is not None:
    est_cmd += ["--batch-size", str(BATCH_SIZE)]
rc, tail = run_stream(est_cmd)
assert rc == 0, f"estimate failed (exit {rc}): {tail[-3:]}"
print("THIS RUN: batch size", BATCH_SIZE or "from the 02 run")
"""

INFER = """# VERIFY (again) -> MERGE -> GUARD -> INFER in ONE process (the merged model never leaves it),
# RESUMABLE per document: the CLI appends one line to trace.jsonl and rewrites predictions.json /
# progress.json after every document, and a resume skips the documents already traced. Order inside
# the process, printed as it happens: VERIFY (again, cheap), MERGE (fp16 base + the adapter,
# merge_and_unload, 200 modules required), GUARD (12 bench pages at batch 1 and at the chosen size
# on the MERGED model; any difference -> batch 1, recorded in manifest.json; a resumed run keeps
# the stored decision), INFER (the 100 dev documents).
import re
import threading
import time
from datetime import UTC, datetime

OOM_RE = re.compile(r"out of memory|OutOfMemoryError", re.IGNORECASE)
OUT_DIR.mkdir(parents=True, exist_ok=True)
SESSIONS_PATH = OUT_DIR / "sessions.json"  # wall-clock log, one entry per Run all
sessions = json.loads(SESSIONS_PATH.read_text()) if SESSIONS_PATH.is_file() else []


def save_json(path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)  # atomic: a disconnect never leaves a half-written file


def read_progress() -> dict:
    try:
        return json.loads((OUT_DIR / "progress.json").read_text())
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
infer_cmd = [PY, "-m", "shipdoc.devfinal", "infer", "--config", CFG,
             "--dev-docs", str(REPO / DEV_DOCS), "--zs500-docs", str(REPO / ZS500_DOCS),
             "--zs-run-dir", str(ZS_RUN), "--out-dir", str(OUT_DIR),
             "--adapter-dir", str(ADAPTER), "--pin", PINNED_SHA, "--bench-docs", BENCH_DOCS]
if BATCH_SIZE is not None:
    infer_cmd += ["--batch-size", str(BATCH_SIZE)]
    print("=" * 78)
    print(f"MANUAL BATCH SIZE {BATCH_SIZE}: it may differ from the 02 run's; the guard below still")
    print("checks it against batch 1 on the merged model.")
    print("=" * 78)
if USE_WANDB:
    infer_cmd.append("--wandb")
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
        f"Dev inference stopped (exit {rc}{', CUDA out of memory' if oom else ''}) after "
        f"{read_progress().get('done')}/{EXPECTED_DOCS} docs; last output: {tail[-3:]}. Everything "
        "done so far is on Drive: Run all again resumes. If it stops at the same point, "
        "report back."
    )
print(f"--- dev inference finished (exit 0) in {(time.time() - t0) / 60:.1f} min this session ---")
"""

COMPARE = """# SCORE + PAIRED COMPARISON (CPU). Official scorer vs the gold of the 100 dev documents; the final
# adapter vs the 02 zero-shot run restricted to the same documents, under three arms: R1-R3 with
# R3 shapes from the TRAIN gold only (primary), R1-R3 with the frozen shipping shapes (learned from
# train+dev: IN-SAMPLE on dev), and raw. OVERALL, header accuracy, row F1, fully-correct documents,
# false-fill rate with 95% CIs and the PAIRED bootstrap delta (2000 resamples, seed 42, document
# level); all, invoices, waybills, scanned, digital; over-null and false-fill counts per field.
# Fails closed: ids not the dev ids, an incomplete run, rules OFF not reproducing the predictions,
# R2 skipped for missing OCR. Writes devfinal_compare.json / .md (aggregates only) next to the run.
cmp_cmd = [PY, "-m", "shipdoc.devfinal", "compare", "--dev-docs", str(REPO / DEV_DOCS),
           "--zs500-docs", str(REPO / ZS500_DOCS), "--zs-run-dir", str(ZS_RUN),
           "--ft-run-dir", str(OUT_DIR), "--out-dir", str(OUT_DIR),
           "--ocr-cache", str(OCR_CACHE)]
rc, tail = run_stream(cmp_cmd)
if rc != 0:
    raise RuntimeError(f"COMPARE FAILED (exit {rc}): {tail[-3:]}. The run files are on Drive; "
                       "report back (counts only).")
"""

BANNER = """# Completion banner. Reads only files on Drive: also runnable alone after a resume.
import json

bar = "=" * 78
BS = chr(92)
LOCAL_DEST = BS.join(["$SHIPDOC_RUNS_DIR", RUN_ID]) + BS
cmp_path, man_path = OUT_DIR / "devfinal_compare.json", OUT_DIR / "manifest.json"
cmp = json.loads(cmp_path.read_text()) if cmp_path.is_file() else None
man = json.loads(man_path.read_text()) if man_path.is_file() else {}
dev_m = man.get("devfinal", {})
print(bar)
if cmp is None:
    print(f"NOT COMPARED: {cmp_path.name} missing. Run the cells above.")
    print(f"DONE {RUN_ID} compared=False")
else:

    def pct(x: float) -> str:
        return f"{100 * x:.2f}"

    def ci(c: dict) -> str:
        return f"{pct(c['point'])} [{pct(c['lo'])}, {pct(c['hi'])}]"

    print(f"FINAL ADAPTER ON DEV ({cmp['n_docs']} docs, SEEN LAYOUTS;"
          " not an unseen-supplier number)")
    print(bar)
    for arm, o in cmp["official_dev"].items():
        d = o["paired_delta_OVERALL"]
        print(f"{arm:<20} FT OVERALL {ci(o['ft_OVERALL'])}  ZS {ci(o['zs_OVERALL'])}  "
              f"paired delta {100 * d['delta']:+.2f} [{pct(d['lo'])}, {pct(d['hi'])}]")
    for arm in ("rules_train_shapes", "raw"):
        for label in ("all", "invoices", "waybills", "scanned", "digital"):
            s = cmp["arms"][arm]["subsets"].get(label)
            if not s:
                continue
            d = s["paired_delta_ft_minus_zs"]["OVERALL"]
            print(f"  {arm:<20}{label:<9}{s['n_docs']:>4} docs  "
                  f"OVERALL ZS {pct(s['zs']['OVERALL'])}"
                  f"  FT {pct(s['ft']['OVERALL'])}  delta {100 * d['delta']:+.2f} "
                  f"[{pct(d['lo'])}, {pct(d['hi'])}]")
    a = cmp["arms"]["rules_train_shapes"]["subsets"]["all"]
    zn, fn = a["zs"]["over_null"], a["ft"]["over_null"]
    print("false fills (header+row cells): "
          f"ZS {zn['false_fill_total']} -> FT {fn['false_fill_total']}"
          f"; over-nulls: ZS {zn['over_null_total']} -> FT {fn['over_null_total']} "
          "(arm rules_train_shapes)")
    g = dev_m.get("guard", {})
    print(f"batch size used {dev_m.get('batch', {}).get('used')} (requested "
          f"{dev_m.get('batch', {}).get('batch_size')}, {dev_m.get('batch', {}).get('source')}); "
          f"guard ran={g.get('ran')} ok={g.get('ok')} fallback_to_1={g.get('fallback_to_1')}")
    print(f"merge: {dev_m.get('merge', {}).get('n_lora_modules_merged')} modules, "
          f"{dev_m.get('merge', {}).get('merge_dtype')}; training code "
          f"{str(dev_m.get('train_code_sha'))[:7]}, pin {PINNED_SHA[:7]}")
    sess_path = OUT_DIR / "sessions.json"
    sess = json.loads(sess_path.read_text()) if sess_path.is_file() else []
    secs = sum(s_["seconds"] for s_ in sess if s_.get("seconds") is not None)
    print(f"wall-clock {secs / 3600:.2f} h over {len(sess)} session(s) (measured)")
    print(f"Drive folder: {OUT_DIR}")
    print("GG: download these files from that Drive folder:")
    for name in ("predictions.json", "trace.jsonl", "manifest.json", "devfinal_compare.json",
                 "devfinal_compare.md"):
        path = OUT_DIR / name
        size = f"{path.stat().st_size / 1e6:8.2f} MB" if path.is_file() else "MISSING"
        print(f"  {name:<24}{size}")
    print(f"into the LOCAL folder  {LOCAL_DEST}")
    print("(outside the repo). Never paste document values into chat; the markdown holds counts.")
    print(f"DONE {RUN_ID} compared=True")
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
        ("code", INSTALL),
        ("code", VERIFY),
        ("code", PLAN),
        ("code", ESTIMATE),
        ("code", INFER),
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
