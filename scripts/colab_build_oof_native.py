"""Generate notebooks/05n_oof_infer_native.ipynb (OOF inference of a NATIVE fold adapter, T4, fp16).

The 05_oof_infer notebook at the native production resolution
(``configs/spike_qwen35_4b_img_only_native.yaml``, ``max_pixels`` 2,196,480; GG decision
2026-10-03). A thin wrapper like 01-07: every cell only orchestrates; the logic is in
``python -m shipdoc.nativerun verify | infer | estimate-oof`` (src/shipdoc/nativerun.py: the
resolution gate and the native estimate, then the UNCHANGED ``shipdoc oof`` stages of
src/shipdoc/oof.py) and ``python -m shipdoc oof compare``. The cells that do not change (account,
mount, secrets, clone, unzip, install, infer, compare, banner) are imported READ-ONLY from
scripts/colab_build_oof.py and edited by exact-string replacement (asserted). Nothing in 01-07 or
their builders is modified.

oof.py already takes the production config as ``--config`` and compares the adapter's
``max_pixels`` (an inference key) and the zero-shot run's config hash against it, so a native
config needs no change there; the extra resolution gate is ``shipdoc.nativerun``.

tests/test_notebook_oof_native.py checks the committed notebook equals this script's output.

Run: uv run python scripts/colab_build_oof_native.py
"""

# ruff: noqa: E501  # cell sources are verbatim notebook text

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "05n_oof_infer_native.ipynb"


def _load(name: str, path: Path):  # noqa: ANN202 - a module object
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


oofb = _load("colab_build_oof", ROOT / "scripts" / "colab_build_oof.py")
base = oofb.base

# The pin commit replaces this with the full 40-char SHA of the pushed code commit and regenerates
# the notebook (same two-commit pattern as 01-07). Until then cell 1 raises.
PINNED_SHA = base.public_pin("4c17aa3c33c09f0cda7bb1f625947a1143a8cb28")  # public tree: its pin

TITLE = """# 05n - OOF inference of a NATIVE-resolution fold adapter on its held-out documents (thin wrapper)

Runs the LoRA adapter of fold K (trained at the native resolution by the native fine-tune notebook
on every train+dev document OUTSIDE fold K) on the documents of fold K that it never saw, at the
native resolution, and compares it with the 02n zero-shot run on exactly the same documents.
T4, fp16. The first half of gate G4 at native: **interim, one fold, not the final G4 decision**.
(Both arms with the rules are compared offline by `scripts/g4_fold.py`; this notebook's own
comparison is the raw OOF one.) The 1260-token notebook 05 and its runs are untouched.

Production inference config: Qwen3.5-4B, IMAGE ONLY, KEYED output, prompt v2, **max_pixels
2,196,480 (2,145 visual tokens per page)**, greedy, xgrammar, seed 42, logprobs on, the shared
coerce / schema-repair writer. ALL held-out documents (invoices AND waybills) are inferred. Labels
are train/dev labels only; no test file is read.

Flow: parameters -> Drive + secrets -> clone at the pin -> unzip data -> install (vlm + train
groups) -> **RESOLUTION CHECK + ADAPTER MANIFEST VERIFICATION (fails closed; printed table)** ->
**ESTIMATE of T4 hours and compute units** (labelled ESTIMATE; read it before the next cell) ->
**INFER**: merge the adapter into the fp16 weights, **guard** (the 12 bench pages at batch 1 and at
the chosen batch size on the MERGED model must be byte-identical and fit in memory, else the run
uses batch 1 and says so), then the resumable per-document inference -> **SCORE + PAIRED
COMPARISON** with the 02n zero-shot run -> completion banner.

What the resolution check refuses (`shipdoc.nativerun.check_adapter_resolution`, FAIL CLOSED): an
inference config whose `max_pixels` is not 2,196,480; a missing or unreadable adapter manifest;
`shipdoc.resmatch` missing (an unavailable check is a refusal); `training_resolution_matches`
raising, answering something other than `(bool, str)`, or answering no (an adapter trained at 1260
tokens, or at any other resolution, is never run at native). The verification after it (unchanged
`shipdoc oof verify`) still refuses a `final` / `smoke` adapter; a fold id other than FOLD; a
missing or dirty training code SHA; a base-model revision other than the config's; LoRA bookkeeping
other than r=16 / 200 modules / 30,474,240 trainable parameters; a manifest without its training
document ids; any overlap between the training ids and the fold's ids, or a training set that is
not exactly the other folds' documents; inference keys (model repo and revision, adapter name,
`max_pixels`, prompt version, output format) that differ from the config; a zero-shot run whose
config hash differs from the native config's (a 1260-token zero-shot run is refused); weight files
whose sha256 differs from the manifest.

Merge precision: as in 05 (the low-rank update is added to the fp16 base weights, fp32 sum rounded
to fp16; not bit-identical to base + unmerged adapter). Every number printed before it has been
measured is an ESTIMATE; the merge at native is UNVERIFIED on a GPU until this notebook has run,
and so is the VRAM of the guard at the chosen batch size (batch 8 at native is unmeasured).

Parameters: `PINNED_SHA`; `FOLD` (0 | 1 | 2); `ZS_SHA7` (the sha7 of the 02n pin: the 02n banner's
`DONE zeroshot500_qwen35_4b_img_only_native_<sha7>`, so the zero-shot folder is
`zeroshot500_qwen35_4b_img_only_native_<ZS_SHA7>`; the notebook refuses while it is the
placeholder or the folder is missing); `TRAIN_SHA7` and `PRECISION` (the sha7 and the precision of the native 03n run:
the adapter folder is `runs/ft_native_fold<FOLD>_<TRAIN_SHA7>_<PRECISION>/final`; TRAIN_SHA7 refuses
while it is the placeholder; `ADAPTER_DIR` can be set by hand for another folder); `BATCH_SIZE` (None = the batch size stored with the 02n run,
refused if absent; an int overrides and is said loudly); `USE_WANDB`.

If the session dies, open the notebook and Run all: setup repeats, verification and the guard are
cheap, the inference resumes after the last document in `trace.jsonl`. See notebooks/README.md.
"""

PARAMS = f"""# Parameters. PINNED_SHA = the repo commit this run is pinned to (full 40-char SHA).
PINNED_SHA = "{PINNED_SHA}"
FOLD = 0  # 0 | 1 | 2: which fold's adapter, and so which fold's held-out documents
# sha7 of the 02n pin: the 02n run folder is zeroshot500_qwen35_4b_img_only_native_<ZS_SHA7>.
ZS_SHA7 = "FILL_ZS_SHA7"
ZS_RUN_DIR = f"zeroshot500_qwen35_4b_img_only_native_{{ZS_SHA7}}"  # the 02n run folder under runs/
# sha7 and precision of the NATIVE 03n run that trained the adapter: its folder is
# runs/ft_native_fold<FOLD>_<TRAIN_SHA7>_<PRECISION>/final (the 03n banner prints it). No default.
TRAIN_SHA7 = "FILL_TRAIN_SHA7"
PRECISION = "bf16"  # bf16 (L4) | fp16 (T4): the suffix of the 03n run folder
# The fold run's final/ folder: relative to MyDrive/shipdoc-extract, or an absolute path.
ADAPTER_DIR = f"runs/ft_native_fold{{FOLD}}_{{TRAIN_SHA7}}_{{PRECISION}}/final"
BATCH_SIZE = None  # None = the batch size stored with the 02n run (refused if absent); or an int
USE_WANDB = False  # off by default; True logs metrics only (no images / values) to the project
WANDB_PROJECT = "shipdoc-extract-debug"  # must be a PRIVATE project
CONFIG = "qwen35_4b_img_only_native"  # KEYED, prompt v2, max_pixels 2196480: the production pick
SPLIT = "train+dev"  # the folds cover the 500 train+dev documents
BENCH_DOCS = "splits/bench12.json"  # 12 dev pages for the batch guard (shipdoc.bench)

if not PINNED_SHA or "FILL" in PINNED_SHA:
    raise ValueError("Set PINNED_SHA in the parameters cell (the pin commit fills it).")
if FOLD not in (0, 1, 2) or isinstance(FOLD, bool):
    raise ValueError(f"FOLD must be 0, 1 or 2, got {{FOLD!r}}")
if not (isinstance(ZS_SHA7, str) and len(ZS_SHA7) == 7
        and all(c in "0123456789abcdef" for c in ZS_SHA7)):
    raise ValueError(f"ZS_SHA7 must be the 7 hex characters of the 02n pin, got {{ZS_SHA7!r}}")
if not (isinstance(TRAIN_SHA7, str) and len(TRAIN_SHA7) == 7
        and all(c in "0123456789abcdef" for c in TRAIN_SHA7)):
    raise ValueError(f"TRAIN_SHA7 must be the 7 hex characters of the 03n pin, got {{TRAIN_SHA7!r}}")
if PRECISION not in ("bf16", "fp16"):
    raise ValueError(f"PRECISION must be 'bf16' or 'fp16', got {{PRECISION!r}}")
if not isinstance(ADAPTER_DIR, str) or not ADAPTER_DIR or "FILL" in ADAPTER_DIR:
    raise ValueError("ADAPTER_DIR must name the native fold run's final/ folder on Drive.")
if BATCH_SIZE is not None and (isinstance(BATCH_SIZE, bool) or not isinstance(BATCH_SIZE, int)
                               or BATCH_SIZE < 1):
    raise ValueError(f"BATCH_SIZE must be None or an int >= 1, got {{BATCH_SIZE!r}}")
if not isinstance(USE_WANDB, bool):
    raise ValueError(f"USE_WANDB must be True or False, got {{USE_WANDB!r}}")
SHA7 = PINNED_SHA[:7]
RUN_NAME = f"oof_native_fold{{FOLD}}"
RUN_ID = f"oof_native_fold{{FOLD}}_{{SHA7}}"  # runs/<RUN_ID>/ on Drive; never oof_fold<K>_*
"""


def _edit(src: str, edits: list[tuple[str, str]], what: str) -> str:
    for old, new in edits:
        assert src.count(old) == 1, f"colab_build_oof.{what} changed: {old[:60]!r}"
        src = src.replace(old, new)
    return src


MOUNT = _edit(
    oofb.MOUNT,
    [
        (
            ': set ADAPTER_DIR / TRAIN_SHA7 / "\n                                "ZS_RUN_DIR in the parameters cell.")',
            '. Set ZS_SHA7 (02n pin) and "\n                                "TRAIN_SHA7 / PRECISION / ADAPTER_DIR (the 03n run) in the "\n'
            '                                "parameters cell.")',
        ),
        (
            '"| batch size:", BATCH_SIZE or "from the 02 run",',
            '"| batch size:", BATCH_SIZE or "from the 02n run",',
        ),
        (
            'print("zero-shot run   :", ZS_RUN)',
            'print("zero-shot run   :", ZS_RUN, "(02n, native)")',
        ),
    ],
    "MOUNT",
)

VERIFY = _edit(
    oofb.VERIFY,
    [
        (
            "# ADAPTER MANIFEST VERIFICATION. Fails closed:",
            "# NATIVE: `shipdoc.nativerun verify` = the RESOLUTION CHECK (an adapter not trained at the\n"
            "# inference resolution is refused), then the unchanged `shipdoc oof verify`.\n"
            "# ADAPTER MANIFEST VERIFICATION. Fails closed:",
        ),
        (
            'verify_cmd = [PY, "-m", "shipdoc", "oof", "verify", "--fold", str(FOLD), "--config", CFG,',
            'verify_cmd = [PY, "-m", "shipdoc.nativerun", "verify", "--fold", str(FOLD), "--config", CFG,',
        ),
    ],
    "VERIFY",
)

ESTIMATE = """# T4-hour and compute-unit (CU) ESTIMATE BEFORE the merge and the inference: read it. Pages come
# from the label files of the fold's held-out documents; the batch size is the one that will be
# used (the 02n run's, or BATCH_SIZE). Includes model load, merge and the batch guard (12 bench
# pages at batch 1 and at the chosen size). NOTHING in it is measured at native: low = the 02
# batch-8 pace (MEASURED at 1260 tokens) x the 06 sweep's native cost ratio, valid only if batch 8
# fits; high = the sweep's MEASURED native batch-1 pace (shipdoc.nativerun). ASSUMED 60 s merge.
# No confirmation prompt (Run all must not block): interrupt the run yourself if it is too high.
est_cmd = [PY, "-m", "shipdoc.nativerun", "estimate-oof", "--fold", str(FOLD),
           "--zs-run-dir", str(ZS_RUN)]
if BATCH_SIZE is not None:
    est_cmd += ["--batch-size", str(BATCH_SIZE)]
rc, tail = run_stream(est_cmd)
assert rc == 0, f"estimate failed (exit {rc}): {tail[-3:]}"
print("THIS RUN: fold", FOLD, "| batch size", BATCH_SIZE or "from the 02n run")
"""

INFER = _edit(
    oofb.INFER,
    [
        (
            'infer_cmd = [PY, "-m", "shipdoc", "oof", "infer", "--fold", str(FOLD), "--config", CFG,',
            'infer_cmd = [PY, "-m", "shipdoc.nativerun", "infer", "--fold", str(FOLD), "--config", CFG,',
        ),
        (
            "# MERGE -> GUARD -> INFER in ONE process",
            "# NATIVE: `shipdoc.nativerun infer` repeats the resolution check, then runs `shipdoc oof infer`.\n"
            "# MERGE -> GUARD -> INFER in ONE process",
        ),
    ],
    "INFER",
)


def _cells() -> list[tuple[str, str]]:
    return [
        ("markdown", TITLE),
        ("code", PARAMS),
        ("markdown", base.ACCOUNT_MD),
        ("code", MOUNT),
        ("code", oofb.SECRETS),
        ("code", base.CLONE),
        ("code", oofb.UNZIP),
        ("code", oofb.INSTALL),
        ("code", VERIFY),
        ("code", ESTIMATE),
        ("code", INFER),
        ("code", oofb.COMPARE),
        ("code", oofb.BANNER),
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
