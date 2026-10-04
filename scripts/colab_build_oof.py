"""Generate notebooks/05_oof_infer.ipynb (OOF inference of a fold adapter on a T4, fp16).

A thin wrapper like 01-04: every cell only orchestrates (Drive, secrets, clone, install, one
subprocess per stage); the logic is in ``python -m shipdoc oof verify | estimate | infer | compare``
(src/shipdoc/oof.py). Helper cells (account, mount, clone, install) are imported READ-ONLY from
scripts/colab_build_notebook.py and edited by exact-string replacement (asserted), so the notebooks
cannot drift apart silently.

tests/test_notebook_oof.py checks the committed notebook equals this script's output.

Run: uv run python scripts/colab_build_oof.py
"""

# ruff: noqa: E501  # cell sources are verbatim notebook text

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "archive" / "05_oof_infer.ipynb"


def _load(name: str, path: Path):  # noqa: ANN202 - a module object
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


base = _load("colab_build_notebook", ROOT / "scripts" / "colab_build_notebook.py")

# The pin commit replaces this with the full 40-char SHA of the pushed code commit and regenerates
# the notebook (same two-commit pattern as 01-04). For now it is the current 02 / 04 pin; GG
# re-pins before the run.
PINNED_SHA = "42b812b5b09d6e4bff0df12564017f71ffad5fc9"

TITLE = """# 05 - OOF inference of a fold adapter on its held-out documents (thin wrapper)

Runs the LoRA adapter of fold K (trained in notebook 03 on every train+dev document OUTSIDE fold K)
on the documents of fold K that it never saw, and compares it with the 02 zero-shot run on exactly
the same documents. T4, fp16. This is the first half of gate G4: **interim, one fold, not the
final G4 decision** (that needs all folds and the spec's rule).

Production inference config: Qwen3.5-4B, IMAGE ONLY, KEYED output, prompt v2, greedy, xgrammar,
seed 42, logprobs on, the shared coerce / schema-repair writer. ALL held-out documents (invoices
AND waybills) are inferred. Labels are train/dev labels only; no test file is read.

Flow: parameters -> Drive + secrets -> clone at the pin -> unzip data -> install (vlm + train
groups: peft comes from the lock) -> **ADAPTER MANIFEST VERIFICATION (fails closed; printed table)**
-> **ESTIMATE of T4 hours and compute units** (labelled ESTIMATE; read it before the next cell) ->
**INFER**: merge the adapter into the fp16 weights, **guard** (the 12 bench pages at batch 1 and at
the chosen batch size on the MERGED model must be byte-identical, else the run uses batch 1 and says
so), then the resumable per-document inference -> **SCORE + PAIRED COMPARISON** with the zero-shot
run, over-null counts and the interim verdict -> completion banner.

What the verification refuses: a `final` / `smoke` adapter; a fold id other than FOLD; a missing or
dirty training code SHA (printed next to this notebook's pin; a different SHA is a warning, an
unreachable one a refusal); a base-model revision other than the production config's; LoRA
bookkeeping other than r=16 / 200 modules / 30,474,240 trainable parameters; a manifest that does
not list its training document ids (an adapter trained before they were recorded cannot be used
for OOF); any overlap between the training ids and the fold's ids, or a training set that is not
exactly the other folds' documents; inference-relevant keys (model repo and revision, adapter
name, max_pixels, prompt version, output format) that differ from the production config or from
the 02 run; weight files whose sha256 differs from the manifest.

Merge precision: the adapter was trained with bf16 (L4) or fp16 + fp32 LoRA weights; here its
low-rank update is added to the fp16 base weights (fp32 sum rounded to fp16). Updates below the
fp16 rounding step of a weight are lost, so the merged model is not bit-identical to base + unmerged
adapter. It is the numeric path production inference on a T4 uses, which is why OOF is measured on
it. Every number printed before it has been measured is an ESTIMATE; the merge is UNVERIFIED on a
GPU until this notebook has run.

Parameters: `PINNED_SHA`; `FOLD` (0 | 1 | 2); `TRAIN_SHA7` and `ADAPTER_DIR` (the Drive folder
`<fold run>/final` of notebook 03; default pattern `runs/ft_fold<FOLD>_<TRAIN_SHA7>_bf16/final`,
set TRAIN_SHA7 or ADAPTER_DIR by hand when the training run was pinned at another commit or ran in
fp16); `ZS_RUN_DIR` (the 02 zero-shot run folder: predictions.json, manifest.json, bench_result.json);
`BATCH_SIZE` (None = the batch size stored with the 02 run, refused if absent; an int overrides and
is said loudly); `USE_WANDB`.

If the session dies, open the notebook and Run all: setup repeats, verification and the guard are
cheap, the inference resumes after the last document in `trace.jsonl`. See notebooks/README.md.
"""

PARAMS = f"""# Parameters. PINNED_SHA = the repo commit this run is pinned to (full 40-char SHA).
PINNED_SHA = "{PINNED_SHA}"
FOLD = 0  # 0 | 1 | 2: which fold's adapter, and so which fold's held-out documents
TRAIN_SHA7 = PINNED_SHA[:7]  # sha7 of the 03 run that trained the adapter; set by hand if not
# The fold run's final/ folder: relative to MyDrive/shipdoc-extract, or an absolute path.
ADAPTER_DIR = f"runs/ft_fold{{FOLD}}_{{TRAIN_SHA7}}_bf16/final"
ZS_RUN_DIR = "zeroshot500_qwen35_4b_img_only_keyed_42b812b"  # the 02 run folder under runs/
BATCH_SIZE = None  # None = the batch size stored with the 02 run (refused if absent); int overrides
USE_WANDB = False  # off by default; True logs metrics only (no images / values) to the project
WANDB_PROJECT = "shipdoc-extract-debug"  # must be a PRIVATE project
CONFIG = "qwen35_4b_img_only"  # KEYED output format, prompt v2, image only: the production pick
SPLIT = "train+dev"  # the folds cover the 500 train+dev documents
BENCH_DOCS = "splits/bench12.json"  # 12 dev pages for the batch guard (shipdoc.bench)

if not PINNED_SHA or "FILL" in PINNED_SHA:
    raise ValueError("Set PINNED_SHA in the parameters cell (the pin commit fills it).")
if FOLD not in (0, 1, 2) or isinstance(FOLD, bool):
    raise ValueError(f"FOLD must be 0, 1 or 2, got {{FOLD!r}}")
if not (isinstance(TRAIN_SHA7, str) and len(TRAIN_SHA7) == 7
        and all(c in "0123456789abcdef" for c in TRAIN_SHA7)):
    raise ValueError(f"TRAIN_SHA7 must be 7 lowercase hex characters, got {{TRAIN_SHA7!r}}")
if BATCH_SIZE is not None and (isinstance(BATCH_SIZE, bool) or not isinstance(BATCH_SIZE, int)
                               or BATCH_SIZE < 1):
    raise ValueError(f"BATCH_SIZE must be None or an int >= 1, got {{BATCH_SIZE!r}}")
if not isinstance(USE_WANDB, bool):
    raise ValueError(f"USE_WANDB must be True or False, got {{USE_WANDB!r}}")
SHA7 = PINNED_SHA[:7]
RUN_NAME = f"oof_fold{{FOLD}}"
RUN_ID = f"oof_fold{{FOLD}}_{{SHA7}}"  # MyDrive/shipdoc-extract/runs/<RUN_ID>/
"""

_ZIPS_OLD = 'REQUIRED_ZIPS = ["data.zip", "assignment.zip", "ocr_cache.zip"]'
assert _ZIPS_OLD in base.MOUNT
# Image only: no OCR cache. data.zip holds data/ (train / dev labels and images), assignment.zip the
# schema and the official scorer.
MOUNT = (
    base.MOUNT.replace(_ZIPS_OLD, 'REQUIRED_ZIPS = ["data.zip", "assignment.zip"]')
    + """def _under_runs(p: str) -> Path:
    \"\"\"Absolute as given; 'runs/...' under MyDrive/shipdoc-extract; a bare name under runs/.\"\"\"
    if Path(p).is_absolute():
        return Path(p)
    return DRIVE_DIR / p if p.startswith("runs/") else RUNS_DIR / p


ADAPTER = _under_runs(ADAPTER_DIR)
ZS_RUN = _under_runs(ZS_RUN_DIR)
OUT_DIR = RUNS_DIR / RUN_ID
for label, path, need in (("adapter folder", ADAPTER, "manifest.json"),
                          ("zero-shot run", ZS_RUN, "predictions.json")):
    if not (path / need).is_file():
        raise FileNotFoundError(f"{label} {path} has no {need}: set ADAPTER_DIR / TRAIN_SHA7 / "
                                "ZS_RUN_DIR in the parameters cell.")
print("fold:", FOLD, "| run_id:", RUN_ID, "| batch size:", BATCH_SIZE or "from the 02 run",
      "| W&B:", "ON" if USE_WANDB else "off")
print("adapter folder  :", ADAPTER)
print("zero-shot run   :", ZS_RUN)
print("Drive output    :", OUT_DIR)
"""
)

SECRETS = base.ZS_SECRETS

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
assert (ASSIGNMENT_DIR / "score.py").is_file(), "assignment.zip has no score.py (official scorer)"
FOLDS_PATH = REPO / "splits" / "folds.json"
BENCH_PATH = REPO / BENCH_DOCS
_folds = json.loads(FOLDS_PATH.read_text(encoding="utf-8"))
HELD = sorted(next(f for f in _folds["folds"] if f["fold"] == FOLD)["val_doc_ids"])
_missing = [d for d in HELD if not (DATA_DIR / d.split("_")[0] / "labels" / f"{d}.json").is_file()]
assert not _missing, f"{len(_missing)} held-out documents without a label file"
assert BENCH_PATH.is_file(), f"{BENCH_DOCS} missing from the clone"
print(f"data OK: fold {FOLD} holds {len(HELD)} documents out ({_folds['k']}-fold CV over "
      f"{sum(len(f['val_doc_ids']) for f in _folds['folds'])} train+dev documents); labels are "
      "train/dev only, no test file is read")
"""

_inst = base.ZS_INSTALL
_INSTALL_EDITS = [
    (
        '    "SHIPDOC_OCR_CACHE": "/content/ocr_cache",  # unused by img_only; the path is '
        "never read\n",
        '    "PYTORCH_ALLOC_CONF": "expandable_segments:True",\n'
        '    "SHIPDOC_OCR_CACHE": "/content/ocr_cache",  # unused by img_only; never read\n',
    ),
    (
        'VLM_GROUP = ["--group", "vlm"] if MODE == "run" else []  # merge mode needs no torch / model stack\n',
        'VLM_GROUP = ["--group", "vlm", "--group", "train"]  # peft==0.21.2 comes from the locked train group\n',
    ),
    ("for p in ('transformers', 'xgrammar')", "for p in ('transformers', 'xgrammar', 'peft')"),
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

VERIFY = """# ADAPTER MANIFEST VERIFICATION. Fails closed: any failed check raises here and NOTHING after
# this cell runs. Prints one row per check (expected, found, PASS / FAIL), the training code SHA
# next to this notebook's pin, and the number of training / inference documents and their overlap
# (counts only, no ids). A training SHA that differs from the pin is a WARNING (the diff of the
# files that matter is shown below); an unreachable SHA, a wrong fold, a final / smoke adapter,
# a missing or overlapping training-document list or a changed inference key is a REFUSAL.
CFG = f"configs/spike_{CONFIG}.yaml"
verify_cmd = [PY, "-m", "shipdoc", "oof", "verify", "--fold", str(FOLD), "--config", CFG,
              "--zs-run-dir", str(ZS_RUN), "--out-dir", str(OUT_DIR),
              "--adapter-dir", str(ADAPTER), "--pin", PINNED_SHA]
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
print(f"VERIFIED: fold {FOLD} adapter, training precision {ADAPTER_MANIFEST.get('precision')}.")
"""

ESTIMATE = """# T4-hour and compute-unit (CU) ESTIMATE BEFORE the merge and the inference: read it. Pages come
# from the label files of the fold's held-out documents; the batch size is the one that will be
# used (the 02 run's, or BATCH_SIZE). Includes model load, merge and the batch guard (12 bench
# pages at batch 1 and at the chosen size). Nothing in it is measured (scripts/gpu_estimate.py:
# ASSUMED batching efficiency 100 / 75 / 50%, ASSUMED 60 s merge). No confirmation prompt (Run all
# must not block): interrupt the run yourself if it is too high.
est_cmd = [PY, "-m", "shipdoc", "oof", "estimate", "--fold", str(FOLD), "--config", CFG,
           "--zs-run-dir", str(ZS_RUN)]
if BATCH_SIZE is not None:
    est_cmd += ["--batch-size", str(BATCH_SIZE)]
rc, tail = run_stream(est_cmd)
assert rc == 0, f"estimate failed (exit {rc}): {tail[-3:]}"
print("THIS RUN: fold", FOLD, "| batch size", BATCH_SIZE or "from the 02 run")
"""

INFER = """# MERGE -> GUARD -> INFER in ONE process (the merged model never leaves it), RESUMABLE per
# document: the CLI appends one line to trace.jsonl and rewrites predictions.json / progress.json
# after every document, and a resume skips the documents already traced. Order inside the process,
# printed as it happens: VERIFY (again, cheap), MERGE (fp16 base + the adapter, merge_and_unload,
# 200 modules required), GUARD (12 bench pages at batch 1 and at the chosen size on the MERGED
# model; any difference -> batch 1, recorded in manifest.json), INFER (the fold's documents).
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
print(f"=== {RUN_ID}: {done_before}/{len(HELD)} docs already done; resuming the rest ===")
session = {"start": datetime.now(UTC).isoformat(), "end": None, "seconds": None,
           "docs_done_at_start": done_before, "exit_code": None}
sessions.append(session)
save_json(SESSIONS_PATH, sessions)
infer_cmd = [PY, "-m", "shipdoc", "oof", "infer", "--fold", str(FOLD), "--config", CFG,
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
        f"OOF inference stopped (exit {rc}{', CUDA out of memory' if oom else ''}) after "
        f"{read_progress().get('done')}/{len(HELD)} docs; last output: {tail[-3:]}. Everything "
        "done so far is on Drive: Run all again resumes. If it stops at the same point, "
        "report back."
    )
print(f"--- OOF inference finished (exit 0) in {(time.time() - t0) / 60:.1f} min this session ---")
"""

COMPARE = """# SCORE + PAIRED COMPARISON (CPU). Official scorer vs the gold of the fold's documents (train /
# dev labels only); zero-shot vs OOF fine-tuned on exactly the same documents: OVERALL, header
# accuracy, row F1, documents fully correct, false-fill rate with each model's 95% CI and the PAIRED
# bootstrap delta (2000 resamples, seed 42, document level) with its CI; all documents, invoices
# only (the headline), waybills only, scanned and digital; over-null counts (header per field, row
# fields, totals) and false fills. Ends with the INTERIM G4 verdict line. Writes oof_compare.json
# and oof_compare.md (aggregates only) next to the OOF run.
cmp_cmd = [PY, "-m", "shipdoc", "oof", "compare", "--fold", str(FOLD), "--config", CFG,
           "--zs-run-dir", str(ZS_RUN), "--out-dir", str(OUT_DIR)]
rc, tail = run_stream(cmp_cmd)
if rc != 0:
    raise RuntimeError(f"COMPARE FAILED (exit {rc}): {tail[-3:]}. The OOF files are on Drive; "
                       "report back (counts only).")
"""

BANNER = """# Completion banner. Reads only files on Drive: also runnable alone after a resume.
import json

bar = "=" * 78
BS = chr(92)
LOCAL_DEST = BS.join(["$SHIPDOC_RUNS_DIR", RUN_ID]) + BS
cmp_path, man_path = OUT_DIR / "oof_compare.json", OUT_DIR / "manifest.json"
cmp = json.loads(cmp_path.read_text()) if cmp_path.is_file() else None
man = json.loads(man_path.read_text()) if man_path.is_file() else {}
oof_m = man.get("oof", {})
print(bar)
if cmp is None:
    print(f"NOT COMPARED: {cmp_path.name} missing. Run the cells above.")
    print(f"DONE {RUN_ID} compared=False")
else:
    v = cmp["verdict"]

    def pct(x: float) -> str:
        return f"{100 * x:.2f}"

    print(f"OOF FOLD {FOLD}: {v['verdict']}   ({v['label']})")
    print(bar)
    for label in ("all", "invoices", "waybills"):
        s = cmp["subsets"].get(label)
        if not s:
            continue
        d = s["paired_delta_oof_minus_zero_shot"]["OVERALL"]
        print(f"{label:<9}{s['n_docs']:>4} docs  OVERALL zero-shot {pct(s['zero_shot']['OVERALL'])}"
              f"  OOF {pct(s['oof']['OVERALL'])}  paired delta {100 * d['delta']:+.2f} "
              f"[{pct(d['lo'])}, {pct(d['hi'])}]")
    a = cmp["subsets"]["all"]
    print(f"over-null cells (header+row): zero-shot {v['zs_over_null']}, OOF {v['oof_over_null']}  "
          f"(header {a['zero_shot']['over_null']['header_over_null']} -> "
          f"{a['oof']['over_null']['header_over_null']}, row "
          f"{a['zero_shot']['over_null']['row_over_null']} -> "
          f"{a['oof']['over_null']['row_over_null']})")
    print(f"delta CI excludes 0 on the positive side: {v['delta_excludes_zero_positive']} "
          "(not part of the verdict)")
    for clause in v["failed_clauses"]:
        print("FAILED CLAUSE:", clause)
    g = oof_m.get("guard", {})
    print(f"batch size used {oof_m.get('batch', {}).get('used')} (requested "
          f"{oof_m.get('batch', {}).get('batch_size')}, {oof_m.get('batch', {}).get('source')}); "
          f"guard ran={g.get('ran')} ok={g.get('ok')} fallback_to_1={g.get('fallback_to_1')}")
    print(f"merge: {oof_m.get('merge', {}).get('n_lora_modules_merged')} modules, "
          f"{oof_m.get('merge', {}).get('merge_dtype')}; training code "
          f"{str(oof_m.get('train_code_sha'))[:7]}, pin {PINNED_SHA[:7]}")
    sess_path = OUT_DIR / "sessions.json"
    sess = json.loads(sess_path.read_text()) if sess_path.is_file() else []
    secs = sum(s_["seconds"] for s_ in sess if s_.get("seconds") is not None)
    print(f"wall-clock {secs / 3600:.2f} h over {len(sess)} session(s) (measured)")
    print(f"Drive folder: {OUT_DIR}")
    print("GG: download these files from that Drive folder:")
    for name in ("predictions.json", "trace.jsonl", "manifest.json", "oof_compare.json",
                 "oof_compare.md", "oof_verification.json"):
        path = OUT_DIR / name
        size = f"{path.stat().st_size / 1e6:8.2f} MB" if path.is_file() else "MISSING"
        print(f"  {name:<24}{size}")
    print(f"into the LOCAL folder  {LOCAL_DEST}")
    print("(outside the repo). Never paste document values into chat; the markdown holds counts.")
    print("NEXT: this verdict is interim and covers ONE fold. fold1, fold2 and final run only")
    print("after a NO REGRESSION here; the final G4 decision needs all folds and the spec's rule.")
    print(f"DONE {RUN_ID} compared=True verdict={v['verdict']}")
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
