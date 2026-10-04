"""Generate notebooks/08_dev_final.ipynb (the FINAL adapter at the NATIVE resolution on the 100 dev docs).

The 05b notebook at the native production resolution (``configs/spike_qwen35_4b_img_only_native.yaml``,
``max_pixels`` 2,196,480; GG decision 2026-10-03). A thin wrapper like 05b: every cell only
orchestrates; the logic is in ``python -m shipdoc.devfinal_native`` (src/shipdoc/devfinal_native.py:
the resolution gates, the native estimate and the native stamp around the UNCHANGED
``shipdoc.devfinal`` stages). The cells that do not change are imported READ-ONLY from
scripts/colab_build_dev_final.py (05b) and edited by exact-string replacement (asserted); nothing in
05b, its builder or ``shipdoc.devfinal`` is modified.

tests/test_notebook_dev_final_native.py checks the committed notebook equals this script's output.

Run: uv run python scripts/colab_build_dev_final_native.py
"""

# ruff: noqa: E501  # cell sources are verbatim notebook text

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "08_dev_final.ipynb"


def _load(name: str, path: Path):  # noqa: ANN202 - a module object
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


dfb = _load("colab_build_dev_final", ROOT / "scripts" / "colab_build_dev_final.py")
base = dfb.base

# UNPINNED until GG pins it: the pin commit replaces this placeholder with the full 40-char SHA of
# the pushed code commit and regenerates the notebook. Until then the parameters cell RAISES.
PINNED_SHA = base.public_pin("4c17aa3c33c09f0cda7bb1f625947a1143a8cb28")  # public tree: its pin

TITLE = """# 08 - the FINAL adapter at the NATIVE resolution on the 100 dev documents: the official seen-layout evaluation (thin wrapper)

Notebook 05b at the native production resolution: Qwen3.5-4B, IMAGE ONLY, KEYED output, prompt v2,
**max_pixels 2,196,480 (2,145 visual tokens per page)**, greedy, xgrammar, seed 42, logprobs on, the
shared coerce / schema-repair writer, T4, fp16. It runs the `final` LoRA adapter of the native fine-tune
notebook 03n (trained at the same resolution on the 400 `train_*` documents; **no dev and no test
document**) on the 100 `dev_*` documents with the production pipeline, scores it with the official
scorer and compares it with the 02n zero-shot run (native) restricted to the same 100 documents. The
1260-token notebook 05b and its runs are untouched.

**What the number means.** Dev layouts (suppliers, templates) also appear in train: the official dev
OVERALL measures **seen layouts**. It does not measure unseen suppliers (that is the out-of-fold / G4
comparison of notebook 05n, spec section 11) and it must not be quoted as such.

Flow: parameters -> Drive + secrets -> clone at the pin -> unzip data + assignment + OCR cache (the
100 dev docs' OCR pages are checked against `SHA256SUMS`) -> install (vlm + train groups; no Paddle)
-> **RESOLUTION CHECK + FINAL ADAPTER VERIFICATION (fails closed; printed table)** -> plan (the 100 ids
must equal the dev ids of `splits/zeroshot500.json`, the held-out ids of the final stage of
`splits/folds.json` and the dev label files) -> **ESTIMATE of T4 hours and compute units** (labelled
ESTIMATE) -> **INFER**: merge the adapter into the fp16 weights, **guard** (the 12 bench pages at batch
1 and at the chosen batch size on the MERGED model must be byte-identical and fit in memory, else batch
1, recorded), then the resumable per-document inference on the 100 dev documents -> **SCORE + PAIRED
COMPARISON** (CPU) -> banner.

What the resolution gate refuses (`python -m shipdoc.devfinal_native`, FAIL CLOSED, before verify and
again before infer): a config whose `max_pixels` is not 2,196,480 (or not the native config name); a
missing or unreadable adapter manifest; `shipdoc.resmatch` missing; `training_resolution_matches`
raising, answering anything but `(bool, str)`, or answering no (an adapter trained at 1260 tokens, or a
manifest with no readable training `max_pixels`); a zero-shot run, and at compare time the dev run, whose
manifest config hash is not the native config's (`shipdoc.runcompat`: a 1260-token zero-shot run is
refused). The verification after it (the unchanged `predict_ft.verify_final_adapter`, the same one 04c
uses) still refuses a smoke / fold adapter (stage other than `final`); a fold id; a missing, dirty or
unreachable training code SHA (a SHA different from this pin is a warning); a base-model repo / revision
other than the config's; LoRA bookkeeping other than r=16 / 200 modules / 30,474,240 trainable
parameters; a manifest without its training document ids; a training set that is not exactly the
`train_*` documents; held-out ids that are not the dev documents; a bad `manifest_hash`; inference keys
(model repo and revision, adapter name, `max_pixels`, prompt version, output format) that differ from the
native config; weight files whose sha256 differs from the manifest.

Comparison (dev documents only): FT vs ZS, under three arms. (1) PRIMARY: R1 + R2 + R3 with the R3
slot shapes learned from the TRAIN gold only (no dev gold anywhere). (2) R1 + R2 + R3 with the frozen
shipping shapes `meta/slot_shapes.json`, learned from all 500 train+dev documents: IN-SAMPLE on dev,
therefore optimistic for both models; reported because it is what ships. (3) Raw model output, no
rules (secondary). R2 reads the OCR of the dev pages from the OCR cache zip. Per arm and subset (all,
invoices, waybills, scanned, digital): OVERALL, header accuracy, row F1, fully-correct documents,
false-fill rate with 95% CIs (doc-level bootstrap, 2000 resamples, seed 42, the unmodified scorer), the
paired FT - ZS deltas and the over-null and false-fill cell counts. Aggregates only: no document id or
value is written or printed. The compare files are stamped with the native config hash.

Merge precision: the adapter's low-rank update is added to the fp16 base weights (fp32 sum rounded to
fp16), so the merged model is not bit-identical to base + unmerged adapter. The merge, the guard's VRAM
at the chosen batch size at native and every timing are UNVERIFIED on a GPU until this notebook has run.

Parameters: `PINNED_SHA` (UNPINNED: the cell raises until the pin commit fills it); `ZS_SHA7` (the sha7
of the 02n pin, the banner's `DONE zeroshot500_qwen35_4b_img_only_native_<sha7>`; the zero-shot folder
is `zeroshot500_qwen35_4b_img_only_native_<ZS_SHA7>`; refused while it is the placeholder or the folder
is missing); `TRAIN_SHA7` and `PRECISION` (the sha7 and precision of the native 03n `final` run; the
adapter folder is `runs/ft_native_final_<TRAIN_SHA7>_<PRECISION>/final`; refused while TRAIN_SHA7 is the
placeholder; `ADAPTER_DIR` can be set by hand); `BATCH_SIZE` (None = the batch size stored with the 02n
run, refused if absent; an int overrides and is said loudly); `USE_WANDB`.

Output: `MyDrive/shipdoc-extract/runs/devfinal_native_<sha7>/` with `predictions.json`, `trace.jsonl`,
`manifest.json`, `devfinal_compare.json`, `devfinal_compare.md` (aggregates only). Download by hand
into `$SHIPDOC_RUNS_DIR\\devfinal_native_<sha7>\\`. If the session dies, Run all: setup repeats,
verification is cheap, the inference resumes after the last document in `trace.jsonl` and keeps the
stored batch decision. See notebooks/README.md.
"""

PARAMS = f"""# Parameters. PINNED_SHA = the repo commit this run is pinned to (full 40-char SHA).
PINNED_SHA = "{PINNED_SHA}"
# sha7 of the 02n pin: the 02n run folder is zeroshot500_qwen35_4b_img_only_native_<ZS_SHA7>.
ZS_SHA7 = "FILL_ZS_SHA7"
ZS_RUN_DIR = f"zeroshot500_qwen35_4b_img_only_native_{{ZS_SHA7}}"  # the 02n run folder under runs/
# sha7 and precision of the NATIVE 03n run that trained the final adapter: its folder is
# runs/ft_native_final_<TRAIN_SHA7>_<PRECISION>/final (the 03n banner prints it). No default.
TRAIN_SHA7 = "FILL_TRAIN_SHA7"
PRECISION = "bf16"  # bf16 (L4) | fp16 (T4): the suffix of the 03n run folder
# The final run's final/ folder: relative to MyDrive/shipdoc-extract, or an absolute path.
ADAPTER_DIR = f"runs/ft_native_final_{{TRAIN_SHA7}}_{{PRECISION}}/final"
BATCH_SIZE = None  # None = the batch size stored with the 02n run (refused if absent); or an int
USE_WANDB = False  # off by default; True logs metrics only (no images / values) to the project
WANDB_PROJECT = "shipdoc-extract-debug"  # must be a PRIVATE project
CONFIG = "qwen35_4b_img_only_native"  # KEYED, prompt v2, max_pixels 2196480: production
SPLIT = "dev"  # the 100 dev documents: the official seen-layout evaluation
DEV_DOCS = "splits/dev100.json"  # = the dev ids of ZS500_DOCS = the final stage's held-out ids
ZS500_DOCS = "splits/zeroshot500.json"
BENCH_DOCS = "splits/bench12.json"  # 12 dev pages for the batch guard (shipdoc.bench)
EXPECTED_DOCS = 100

if not PINNED_SHA or "FILL" in PINNED_SHA:
    raise ValueError("Set PINNED_SHA in the parameters cell (the pin commit fills it).")
if not (isinstance(ZS_SHA7, str) and len(ZS_SHA7) == 7
        and all(c in "0123456789abcdef" for c in ZS_SHA7)):
    raise ValueError(f"ZS_SHA7 must be the 7 hex characters of the 02n pin, got {{ZS_SHA7!r}}")
if not (isinstance(TRAIN_SHA7, str) and len(TRAIN_SHA7) == 7
        and all(c in "0123456789abcdef" for c in TRAIN_SHA7)):
    raise ValueError(f"TRAIN_SHA7 must be the 7 hex characters of the 03n pin, got {{TRAIN_SHA7!r}}")
if PRECISION not in ("bf16", "fp16"):
    raise ValueError(f"PRECISION must be 'bf16' or 'fp16', got {{PRECISION!r}}")
if not isinstance(ADAPTER_DIR, str) or not ADAPTER_DIR or "FILL" in ADAPTER_DIR:
    raise ValueError("ADAPTER_DIR must name the native final run's final/ folder on Drive.")
if BATCH_SIZE is not None and (isinstance(BATCH_SIZE, bool) or not isinstance(BATCH_SIZE, int)
                               or BATCH_SIZE < 1):
    raise ValueError(f"BATCH_SIZE must be None or an int >= 1, got {{BATCH_SIZE!r}}")
if not isinstance(USE_WANDB, bool):
    raise ValueError(f"USE_WANDB must be True or False, got {{USE_WANDB!r}}")
SHA7 = PINNED_SHA[:7]
RUN_NAME = "devfinal_native"
RUN_ID = f"devfinal_native_{{SHA7}}"  # MyDrive/shipdoc-extract/runs/<RUN_ID>/; never devfinal_<sha7>
"""


def _edit(src: str, edits: list[tuple[str, str]], what: str) -> str:
    for old, new in edits:
        assert src.count(old) == 1, f"colab_build_dev_final.{what} changed: {old[:60]!r}"
        src = src.replace(old, new)
    return src


MOUNT = _edit(
    dfb.MOUNT,
    [
        (
            ': set ADAPTER_DIR / TRAIN_SHA7 / "\n                                "ZS_RUN_DIR in the parameters cell.")',
            '. Set ZS_SHA7 (02n pin) and "\n                                "TRAIN_SHA7 / PRECISION / ADAPTER_DIR (the 03n final run) in the "\n'
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

_MOD_OLD, _MOD_NEW = '"shipdoc.devfinal"', '"shipdoc.devfinal_native"'

VERIFY = _edit(
    dfb.VERIFY,
    [
        (_MOD_OLD, _MOD_NEW),
        (
            "# FINAL ADAPTER MANIFEST VERIFICATION. Fails closed:",
            "# NATIVE: `shipdoc.devfinal_native verify` = the RESOLUTION CHECK (an adapter not trained at\n"
            "# the inference resolution is refused), then the unchanged final-adapter verification of 04c.\n"
            "# FINAL ADAPTER MANIFEST VERIFICATION. Fails closed:",
        ),
        (
            '"src/shipdoc/extract.py", "configs/finetune_qwen35_4b.yaml", CFG])',
            '"src/shipdoc/extract.py", "configs/finetune_qwen35_4b_native.yaml", CFG])',
        ),
    ],
    "VERIFY",
)

PLAN = _edit(dfb.PLAN, [(_MOD_OLD, _MOD_NEW)], "PLAN")

ESTIMATE = """# T4-hour and compute-unit (CU) ESTIMATE BEFORE the merge and the inference: read it. Pages are
# counted from the label files of the 100 dev documents; the batch size is the one that will be
# used (the 02n run's, or BATCH_SIZE). Includes model load, merge and the batch guard (12 bench
# pages at batch 1 and at the chosen size). NOTHING in it is measured at native: low = the 02
# batch-8 pace (MEASURED at 1260 tokens) x the 06 sweep's native cost ratio, valid only if the batch
# size fits; high = the sweep's MEASURED native batch-1 pace (shipdoc.nativerun). ASSUMED 60 s
# merge. No confirmation prompt (Run all must not block): interrupt the run yourself if too high.
est_cmd = [PY, "-m", "shipdoc.devfinal_native", "estimate", "--config", CFG,
           "--zs-run-dir", str(ZS_RUN)]
if BATCH_SIZE is not None:
    est_cmd += ["--batch-size", str(BATCH_SIZE)]
rc, tail = run_stream(est_cmd)
assert rc == 0, f"estimate failed (exit {rc}): {tail[-3:]}"
print("THIS RUN: batch size", BATCH_SIZE or "from the 02n run")
"""

INFER = _edit(
    dfb.INFER,
    [
        (_MOD_OLD, _MOD_NEW),
        (
            "# VERIFY (again) -> MERGE -> GUARD -> INFER in ONE process",
            "# NATIVE: `shipdoc.devfinal_native infer` repeats the resolution check and the 02n config-hash\n"
            "# check, then runs the unchanged `shipdoc.devfinal infer`.\n"
            "# VERIFY (again) -> MERGE -> GUARD -> INFER in ONE process",
        ),
        (
            "checks it against batch 1 on the merged model.",
            "checks it against batch 1 on the merged model (and VRAM at native).",
        ),
    ],
    "INFER",
)

COMPARE = _edit(
    dfb.COMPARE,
    [
        (_MOD_OLD, _MOD_NEW),
        (
            "# SCORE + PAIRED COMPARISON (CPU). Official scorer",
            "# NATIVE: the 02n run and the dev run must both carry the native config hash (a 1260-token\n"
            "# zero-shot run is refused); the output files are stamped with it.\n"
            "# SCORE + PAIRED COMPARISON (CPU). Official scorer",
        ),
    ],
    "COMPARE",
)

BANNER = _edit(
    dfb.BANNER,
    [
        (
            "print(f\"FINAL ADAPTER ON DEV ({cmp['n_docs']} docs, SEEN LAYOUTS;\"",
            "print(f\"FINAL ADAPTER ON DEV, NATIVE ({cmp['n_docs']} docs, SEEN LAYOUTS;\"",
        ),
        (
            '    print(bar)\n    for arm, o in cmp["official_dev"].items():',
            '    print(bar)\n    nr = cmp.get("native_resolution") or {}\n'
            "    print(f\"native config {nr.get('config')} hash {nr.get('config_hash')} \"\n"
            "          f\"max_pixels {nr.get('max_pixels')}\")\n"
            '    for arm, o in cmp["official_dev"].items():',
        ),
    ],
    "BANNER",
)


def _cells() -> list[tuple[str, str]]:
    return [
        ("markdown", TITLE),
        ("code", PARAMS),
        ("markdown", base.ACCOUNT_MD),
        ("code", MOUNT),
        ("code", dfb.SECRETS),
        ("code", base.CLONE),
        ("code", dfb.UNZIP),
        ("code", dfb.INSTALL),
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
