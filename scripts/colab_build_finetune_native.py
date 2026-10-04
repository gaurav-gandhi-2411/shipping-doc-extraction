"""Generate notebooks/03n_finetune_native.ipynb (LoRA fine-tune at the native 2145-token resolution).

It is notebook 03 with ``configs/finetune_qwen35_4b_native.yaml`` (max_pixels 2196480, 2145 visual
tokens): the cells that do not depend on the resolution are IMPORTED from
``scripts/colab_build_finetune.py`` (which is not modified; every substitution below asserts that
the text it replaces is still there, so a change in 03 breaks this build loudly instead of silently
diverging). New here: the run ids (``ft_native_<stage>_<sha7>_<precision>``, so a 1260 run folder,
smoke status or checkpoint can never be picked up), the native estimate cell, the stage-order
reminder of the new plan (spec.md section 11 item 2) and a cell that checks the finished stage's
manifest records the native training resolution.

Pin: ``PINNED_SHA`` stays the placeholder ``FILL_PINNED_SHA`` until the pin commit (code commit
SHA, then regenerate). While it is a placeholder the notebook refuses to run.

Run: uv run python scripts/colab_build_finetune_native.py
"""

# ruff: noqa: E501  # cell sources are verbatim notebook text

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "03n_finetune_native.ipynb"
STAGES = ("smoke", "fold0", "fold1", "fold2", "final")
NATIVE_MAX_PIXELS = 2_196_480
NATIVE_VISUAL_TOKENS = 2145

_spec = importlib.util.spec_from_file_location(
    "colab_build_finetune", ROOT / "scripts" / "colab_build_finetune.py"
)
ft = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("colab_build_finetune", ft)
_spec.loader.exec_module(ft)
base = ft.base
# The public tree (configs/public_notebooks.json) substitutes its own pin; the private default stays.
PINNED_SHA = base.public_pin("4c17aa3c33c09f0cda7bb1f625947a1143a8cb28")


def _swap(text: str, old: str, new: str) -> str:
    """``text.replace(old, new)`` that refuses to continue when `old` is not in `text`."""
    if old not in text:
        raise AssertionError(f"colab_build_finetune changed: cannot find {old[:70]!r}")
    return text.replace(old, new)


NATIVE_TITLE = """# 03n - LoRA fine-tune of Qwen3.5-4B at the NATIVE resolution (thin wrapper)

Notebook 03 with `configs/finetune_qwen35_4b_native.yaml`: `max_pixels` 2196480 = 2,145 visual
tokens per page (the native 1240x1754 page, adopted by the 06 sweep), instead of 1,260. Everything
else is the 03 recipe (language-model-only LoRA r=16, 200 modules, 30,474,240 trainable parameters,
2 epochs, effective batch 8, seed 42). The 1260 config and the fold-0 adapter trained with it are
untouched. All logic is in `python -m shipdoc.train` / `python -m shipdoc.trainset`.

**Set `STAGE` in the parameters cell.** The default is `smoke`: the 20-step smoke-train at the
native resolution, the only thing to run first (peak VRAM at 2145 tokens is UNMEASURED; the
budget is the 03 one, 20 GiB bf16, never edited to pass). Stages: `smoke`, `fold0`, `fold1`,
`fold2` (cross-validation folds of `splits/folds.json`, train on the other two), `final` (all 400
train docs).

Run ids are `ft_native_<stage>_<sha7>_<precision>` (smoke: `ft_native_smoke_<sha7>_<precision>`) and
the config hash includes `max_pixels`, so a 1260 run folder or checkpoint is never reused. The
adapter manifest records `inference_keys.max_pixels`; the cell after the stage checks it is
2196480, and the OOF notebook refuses an adapter whose training resolution differs from its own.

Flow: parameters -> Drive + secrets -> clone at the pin -> unzip data -> **ESTIMATE (native VRAM,
hours and compute units per stage, printed before anything runs: read it)** -> install and GPU probe
-> prepare the training plan (CPU, the same `prepared_all.json` as 03: it holds no pixel sizes) ->
**SMOKE stage (always first; the notebook stops here if it fails)** -> the chosen `STAGE` (skipped
for `smoke`) -> resolution check of the manifest -> completion banner.

**Order of the stages (spec.md section 11 item 2).** `smoke` -> `fold0` (native) first -> its OOF
inference at the same resolution (05) -> `fold1`, `fold2`, `final`. All three folds and the final
adapter must be native: the 3-fold comparison of section 11 item 1 never mixes a 1260 adapter in.
Run the smoke ONCE before opening parallel tabs: they share the smoke folder on Drive. The
notebook does not enforce the order (no code gate); it prints a reminder.

Resumable exactly as 03 (checkpoints every 10 steps to `MyDrive/shipdoc-extract/runs/<run_id>/ckpt`;
Run all again resumes). No early stopping on held-out loss. Every number printed here is UNVERIFIED
until it has been measured on a GPU at this resolution.
"""

_PARAMS = ft.FT_PARAMS
_PARAMS = _swap(
    _PARAMS,
    'CONFIG = "configs/finetune_qwen35_4b.yaml"  # every hyper-parameter, seed 42',
    'CONFIG = "configs/finetune_qwen35_4b_native.yaml"  # 03 config + max_pixels 2196480, seed 42\n'
    f"NATIVE_MAX_PIXELS = {NATIVE_MAX_PIXELS}  # the estimate cell refuses a config that disagrees",
)
_PARAMS = _swap(
    _PARAMS,
    "# Stage order (a reminder, not a gate): fold1, fold2 and final run ONLY after fold0's OOF\n"
    "# inference shows no regression versus zero-shot on its held-out suppliers.\n"
    "AFTER_FOLD0 = (\n"
    '    "fold1, fold2 and final run ONLY after fold0\'s OOF inference shows "\n'
    '    "no regression vs zero-shot on its held-out suppliers."\n'
    ")",
    "# Stage order (a reminder, not a gate; spec.md section 11 item 2): fold0 NATIVE first, its OOF\n"
    "# inference at the same resolution next, then fold1, fold2 and final.\n"
    "AFTER_FOLD0 = (\n"
    '    "fold0 (native) runs first and its OOF inference (05 at max_pixels 2196480) is read "\n'
    '    "before fold1, fold2 and final; all of them must be native, never mix in a 1260 adapter."\n'
    ")",
)
_PARAMS = _swap(
    _PARAMS,
    'RUN_BASE = f"ft_{STAGE}_{SHA7}"  # the precision is appended once the GPU is known',
    'RUN_BASE = f"ft_native_{STAGE}_{SHA7}"  # precision appended once the GPU is known; never ft_*',
)
NATIVE_PARAMS = _PARAMS

# The GPU / precision probe of the 03 estimate cell is reused verbatim (read_gpu ... KIND).
_GPU = ft.FT_ESTIMATE[
    ft.FT_ESTIMATE.index("def read_gpu():") : ft.FT_ESTIMATE.index("pages = fe.fold_pages(REPO)")
]

NATIVE_ESTIMATE = (
    """# ESTIMATE of VRAM, GPU hours and compute units (CU) at the NATIVE resolution, printed BEFORE
# anything runs. No confirmation prompt (Run all must not block): interrupt it yourself if it is
# too high. Source: scripts/finetune_native_estimate.py, a wrapper over finetune_estimate.py that
# scales the MEASURED 1260-token fold-0 pace (8.87 s/page) and peak VRAM (11.4 GiB) by stated,
# ASSUMED linear factors. UNVERIFIED: the smoke below measures the real peak and s/step.
import importlib.util
import subprocess
import sys

import yaml

_spec = importlib.util.spec_from_file_location(
    "finetune_native_estimate", REPO / "scripts" / "finetune_native_estimate.py"
)
fne = importlib.util.module_from_spec(_spec)
sys.modules["finetune_native_estimate"] = fne  # dataclasses resolve their module via sys.modules
_spec.loader.exec_module(fne)
fe = fne.fe

cfg = yaml.safe_load((REPO / CONFIG).read_text(encoding="utf-8"))
assert cfg["max_pixels"] == NATIVE_MAX_PIXELS, (
    f"{CONFIG} has max_pixels {cfg['max_pixels']}, this notebook is the native one "
    f"({NATIVE_MAX_PIXELS}): use notebook 03 for the 1260 config."
)
assert cfg["expected_visual_tokens"] == fne.TOKENS_NATIVE, cfg["expected_visual_tokens"]
EPOCHS, ACCUM = cfg["epochs"], cfg["grad_accum"]


"""
    + _GPU
    + """pages = fe.fold_pages(REPO)
print(f"Runtime GPU (nvidia-smi): {GPU_NAME} | "
      f"precision this run will use: {PREVIEW_PRECISION}")
print(f"STAGE = {STAGE!r}; {EPOCHS} epochs, effective batch {ACCUM} pages, max_pixels "
      f"{cfg['max_pixels']} ({cfg['expected_visual_tokens']} visual tokens per page); the smoke "
      "stage always runs first.")
if KIND != "L4":
    print("WARNING: native training is planned on an L4 (bf16). On a T4 (fp16, 13.5 GiB "
          "budget) the high VRAM estimate below is inside 15% of it: expect the smoke to fail.")
print()
for line in fne.native_lines(pages, EPOCHS, ACCUM, cfg["eval_every"], cfg["eval_pages"]):
    print(line)
print("\\nOOF and dev INFERENCE (T4 fp16) is not part of this notebook. " + AFTER_FOLD0)
"""
)

NATIVE_INSTALL = _swap(
    ft.FT_INSTALL,
    'SMOKE_RUN_ID = f"ft_smoke_{SHA7}_{PRECISION}"',
    'SMOKE_RUN_ID = f"ft_native_smoke_{SHA7}_{PRECISION}"  # never the 1260 ft_smoke_* folder',
)

NATIVE_RESCHECK = """# Resolution record of the finished stage: <run>/final/manifest.json must carry the training
# resolution (inference_keys.max_pixels) and it must be the native one. Notebook 05 runs the same
# check (shipdoc.resmatch, stdlib only) and refuses an adapter trained at another resolution.
import importlib.util
import sys

_spec = importlib.util.spec_from_file_location("resmatch", REPO / "src" / "shipdoc" / "resmatch.py")
rm = importlib.util.module_from_spec(_spec)
sys.modules["resmatch"] = rm
_spec.loader.exec_module(rm)
if STAGE == "smoke":
    print("STAGE = smoke: no adapter manifest to check.")
else:
    manifest_path = RUN_DIR / "final" / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"{manifest_path} is missing: the stage did not finish.")
    manifest = json.loads(manifest_path.read_text())
    ok, why = rm.training_resolution_matches(manifest, {"max_pixels": NATIVE_MAX_PIXELS})
    print(("RESOLUTION OK: " if ok else "RESOLUTION MISMATCH: ") + why)
    if not ok:
        raise RuntimeError("The adapter manifest does not record the native resolution: " + why)
"""

NATIVE_BANNER = _swap(
    ft.FT_BANNER,
    'print("with zero-shot on the held-out suppliers BEFORE starting anything else.")',
    'print("with zero-shot at max_pixels 2196480 (and the 1260 fold-0 result) BEFORE anything else.")',
)


def build_native() -> dict:
    """03n_finetune_native notebook dict (nbformat 4.5) with all outputs cleared."""
    cells = [
        ("markdown", NATIVE_TITLE),
        ("code", NATIVE_PARAMS.replace("@@PIN@@", PINNED_SHA)),
        ("markdown", base.ACCOUNT_MD),
        ("code", ft.FT_MOUNT),
        ("code", ft.FT_SECRETS),
        ("code", base.CLONE),
        ("code", base.UNZIP),
        ("code", NATIVE_ESTIMATE),
        ("code", NATIVE_INSTALL),
        ("code", ft.FT_PREPARE),
        ("code", ft.FT_SMOKE),
        ("code", ft.FT_TRAIN),
        ("code", NATIVE_RESCHECK),
        ("code", NATIVE_BANNER),
    ]
    nb = base._notebook(cells)
    nb["metadata"]["colab"]["gpuType"] = "L4"  # native training is planned on an L4 (bf16)
    return nb


def render_native() -> str:
    """Canonical JSON text of the 03n_finetune_native notebook."""
    return json.dumps(build_native(), indent=1, ensure_ascii=False) + "\n"


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(render_native(), encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")
