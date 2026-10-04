"""Generate notebooks/02n_zeroshot500_native.ipynb (zero-shot over the 500 docs at native resolution).

The 02_zeroshot500 notebook with the native production config
(``configs/spike_qwen35_4b_img_only_native.yaml``, ``max_pixels`` 2,196,480 = 2,145 visual tokens
per page; GG decision 2026-10-03). A thin wrapper like 01-07: every cell only orchestrates; the
logic is in ``python -m shipdoc spike | bench | merge-shards``. The cells that do not change
(account, mount, secrets, clone, unzip, shards, smoke, bench, run, merge, banner) are imported
READ-ONLY from scripts/colab_build_notebook.py and edited by exact-string replacement (asserted),
so a fix there reaches this notebook and the two cannot drift apart silently. Nothing in 01-07 or
their builders is modified.

What differs from 02: the config; the run id (``zeroshot500_<config>_<sha7>``, no ``keyed`` infix,
so it can never collide with a 1260-token run); the T4 hour / CU estimate cell (the speed table has
no native entry, ``shipdoc.nativerun`` derives it from measured paces); the allocator setting
``expandable_segments`` (as in 05; numerics unchanged, fewer fragmentation OOMs at the bigger
batch); failed bench sizes are listed.

tests/test_notebook_zs_native.py checks the committed notebook equals this script's output.

Run: uv run python scripts/colab_build_zs_native.py
"""

# ruff: noqa: E501  # cell sources are verbatim notebook text

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "02n_zeroshot500_native.ipynb"


def _load(name: str, path: Path):  # noqa: ANN202 - a module object
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


base = _load("colab_build_notebook", ROOT / "scripts" / "colab_build_notebook.py")

# The pin commit replaces this with the full 40-char SHA of the pushed code commit and regenerates
# the notebook (same two-commit pattern as 01-07). Until then cell 1 raises.
PINNED_SHA = base.public_pin("4c17aa3c33c09f0cda7bb1f625947a1143a8cb28")  # public tree: its pin

TITLE = """# 02n - zero-shot run on all 500 train+dev docs at NATIVE resolution (thin wrapper)

Qwen3.5-4B, image only, KEYED output, prompt v2, **native resolution**: `max_pixels` 2,196,480 =
2,145 visual tokens per 1240 x 1754 page (the production resolution since the 06 sweep and GG's
2026-10-03 decision; `configs/spike_qwen35_4b_img_only_native.yaml`, which is the 1260-token config
with only `name` and `max_pixels` changed). The 1260-token notebook 02 and its runs are untouched.
The zero-shot predictions of ALL 500 docs (671 pages) are (1) the zero-shot arm of the paired
comparison at native (05n, `scripts/g4_fold.py`) and (2) the training data of the calibrator, so
every trace page carries `field_logprobs` (see `src/shipdoc/trace.py`).

All logic is in `python -m shipdoc spike --split train+dev --logprobs`; this notebook only does
Drive, secrets, clone, install and bookkeeping. `PINNED_SHA` in the parameters cell is the code
commit it clones (filled by the pin commit); just Runtime -> Run all.

Run id: `zeroshot500_qwen35_4b_img_only_native_<sha7>` (no `keyed` infix: distinct from every
1260-token run `zeroshot500_qwen35_4b_img_only_keyed_<sha7>`). 05n needs `<sha7>` of THIS pin.

Flow: setup (Drive, clone at the pin, unzip data) -> shard plan -> T4-hour / compute-unit
ESTIMATE (printed before anything runs; read it, interrupt if too high) -> install -> MANDATORY
5-doc smoke gate at native (7 assertions incl. finite logprobs for every emitted field; failure
stops the notebook) -> BATCH-SIZE BENCH at native (12 dev pages at batch 1/2/4/8: byte-identity vs
batch 1, peak VRAM, pages/hour; rule: the largest size with 100% byte-identical outputs and peak
VRAM <= 14.5 GiB; a size that runs out of memory is recorded as failed and the others go on;
skipped when `BATCH_SIZE` is an int) -> full run, RESUMABLE per doc -> completion banner.

**Why the bench matters here:** VRAM at batch 8 is UNMEASURED at native. The 1260-token batch-8
peak was 13.6 GiB; the sweep measured native only at batch 1 (9.41 GiB vs 9.14 GiB for the
control). A native batch 8 may not fit in 14.5 GiB, in which case the rule picks 4 or 2 (or 1)
and the run is slower than the estimate's low row. Every figure printed before the bench is an
ESTIMATE (UNVERIFIED on a GPU).

Parameters: `BATCH_SIZE` (None = the bench decides, an int = manual and the bench is skipped),
`SHARD` ("i/K") and `MODE` ("run", or "merge" after all K shard tabs finished), exactly as in 02;
for K > 1 start every tab with the same pin and the same batch size.

The bench result (batch size) is stored with the run (`manifest.json` `batch_size`, `bench`) and
05n / 04c read it from there: the TEST submission and every OOF run must use the same batch size.

If the session dies, open the notebook again and Run all: setup repeats, the smoke gate and the
bench are skipped (they passed; finished bench sizes are never redone) and the full run resumes
after the last document in `trace.jsonl` on Drive. See notebooks/README.md.
"""

PARAMS = f"""# Parameters. PINNED_SHA = the repo commit this run is pinned to (full 40-char SHA).
PINNED_SHA = "{PINNED_SHA}"
RUN_NAME = "zeroshot500"
CONFIG = "qwen35_4b_img_only_native"  # KEYED, prompt v2, max_pixels 2196480 (2145 tokens/page)
SPLIT = "train+dev"  # the 500 docs of splits/zeroshot500.json come from both splits
DOCS = "splits/zeroshot500.json"
EXPECTED_DOCS = 500
EXPECTED_PAGES = 671
SMOKE_DOCS = "splits/smoke5.json"  # 5 dev docs picked by meta tags (shipdoc.smoke)
SMOKE_LIMIT = 5
BENCH_DOCS = "splits/bench12.json"  # 12 dev pages for the batch-size bench (shipdoc.bench)
BATCH_SIZE = None  # None = the bench picks it; an int = manual, the bench is skipped
SHARD = "0/1"  # "i/K": this tab runs document-level shard i of K; "0/1" = everything
MODE = "run"  # "run" = smoke, bench, this shard's run; "merge" = combine the K shard folders
USE_WANDB = False  # off by default; True logs metrics only (no images / values) to the project
WANDB_PROJECT = "shipdoc-extract-debug"  # must be a PRIVATE project; see notebooks/README.md

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
RUN_ID = f"{{RUN_NAME}}_{{CONFIG}}_{{SHA7}}"  # the unsharded run (and the merge target); no 'keyed'
SHARD_RUN_ID = RUN_ID if SHARD_K == 1 else f"{{RUN_ID}}_shard{{SHARD_I}}of{{SHARD_K}}"  # this tab
"""

# --- cells imported from 02 and edited by exact-string replacement (each edit is asserted) -----


def _edit(src: str, edits: list[tuple[str, str]], what: str) -> str:
    for old, new in edits:
        assert src.count(old) == 1, f"colab_build_notebook.{what} changed: {old[:60]!r}"
        src = src.replace(old, new)
    return src


INSTALL = _edit(
    base.ZS_INSTALL,
    [
        (
            '    "SHIPDOC_OCR_CACHE": "/content/ocr_cache",  # unused by img_only; the path is '
            "never read\n",
            '    "PYTORCH_ALLOC_CONF": "expandable_segments:True",  # fewer fragmentation OOMs\n'
            '    "SHIPDOC_OCR_CACHE": "/content/ocr_cache",  # unused by img_only; never read\n',
        )
    ],
    "ZS_INSTALL",
)

ESTIMATE = """# T4-hour and compute-unit (CU) ESTIMATE BEFORE anything runs on the GPU: read it.
# configs/spike_speed.json has no entry for the native config, so shipdoc.nativerun derives the
# two bounds from MEASURED paces (02 batch 8 at 1260 tokens x the 06 sweep's native cost ratio;
# the sweep's native batch 1). The pace of the batch size the bench picks lies between them.
# No confirmation prompt (Run all must not block): interrupt the run yourself if it is too high.
import importlib.util
import subprocess
import sys

_ge_path = REPO / "scripts" / "gpu_estimate.py"
_spec = importlib.util.spec_from_file_location("gpu_estimate", _ge_path)
ge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ge)
_nr_path = REPO / "src" / "shipdoc" / "nativerun.py"
_spec = importlib.util.spec_from_file_location("shipdoc_nativerun", _nr_path)
nr = importlib.util.module_from_spec(_spec)
sys.modules["shipdoc_nativerun"] = nr
_spec.loader.exec_module(nr)

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


GPU_NAME = read_gpu_name()
smoke_pages = ge.count_pages(smoke_ids, DEV_LABELS)
tab_pages = SHARD_EXPECTED_PAGES  # this tab's shard (everything when SHARD is 0/1)
print("Runtime GPU (nvidia-smi):", GPU_NAME)
print(nr.format_zs_estimate(SPEED, tab_pages, smoke_pages, BATCH_SIZE, GPU_NAME))
print(f"THIS TAB: shard {SHARD} ({tab_pages} pages), batch size "
      f"{BATCH_SIZE or 'from the bench'}; with K = {SHARD_K} tabs the CU of the tabs add up "
      "(K x the T4 h above).")
"""

SMOKE = base.ZS_SMOKE

_bench = _edit(
    base.ZS_BENCH,
    [
        (
            '    BATCH = int(bench_result["chosen_batch_size"])\n',
            '    BATCH = int(bench_result["chosen_batch_size"])\n'
            '    results = bench_result.get("results", [])\n'
            '    failed_sizes = [r["batch_size"] for r in results if not r.get("ok")]\n'
            "    if failed_sizes:\n"
            '        print(f"bench sizes that FAILED (recorded, not fatal; usually out of memory): "\n'
            '              f"{failed_sizes}; the rule chose among the others")\n',
        ),
        (
            "Fix it, or set BATCH_SIZE = 1 (the unbatched path) to skip the bench.",
            'Run all again resumes (finished sizes are not redone); or set "\n'
            '            "BATCH_SIZE = 1 (the unbatched path) to skip the bench.',
        ),
    ],
    "ZS_BENCH",
)
BENCH = (
    "# NATIVE (2,145 tokens/page): VRAM at batch 8 is UNMEASURED here. A size that runs out of\n"
    "# memory is recorded as failed in bench_result.json (shipdoc.bench), the other sizes go on.\n"
    + _bench
)


def _cells() -> list[tuple[str, str]]:
    run_only = base._run_only
    return [
        ("markdown", TITLE),
        ("code", PARAMS),
        ("markdown", base.ZS_ACCOUNT_MD),
        ("code", base.ZS_MOUNT),
        ("code", base.ZS_SECRETS),
        ("code", base.CLONE),
        ("code", base.ZS_UNZIP),
        ("code", base.ZS_SHARDS),
        ("code", run_only(ESTIMATE, "estimate")),
        ("code", INSTALL),
        ("code", run_only(SMOKE, "smoke gate")),
        ("code", run_only(BENCH, "bench")),
        ("code", run_only(base.ZS_RUN, "full run")),
        ("code", base.ZS_MERGE),
        ("code", base.ZS_BANNER),
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
