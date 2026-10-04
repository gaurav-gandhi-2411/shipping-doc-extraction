"""Generate notebooks/public_smoke.ipynb: the 15-minute check that the PUBLIC repo runs on a T4.

For the person who has just created the public repository and wants to see, before anyone else
does, that a stranger's Colab can clone it (no token), check out the pinned commit, install from
``uv.lock`` and decode five documents with the production configuration. It is NOT the
reproduction: it never scores anything and never writes to Drive. The reproduction is
``02n_zeroshot500_native`` followed by ``04c_predict_test_native``.

A thin wrapper like 02n: the install and smoke-gate cells are IMPORTED from
``scripts/colab_build_zs_native.py`` (and through it ``colab_build_notebook.py``), so the gate that
runs here is byte for byte the gate of 02n. Everything else is bookkeeping: public clone, pin
verification, the five smoke documents out of ``data.zip``, an optional determinism replay and a
banner to paste back.

Pin and URL come from ``configs/public_notebooks.json`` (written into the public tree by
``scripts/publish_overlay.py``, filled by ``scripts/publish_pin_public.py``). Without that file
(the private repo) both are placeholders and the notebook refuses to run: cell 1 raises on the pin,
the clone cell on the URL.

tests/test_notebook_public_smoke.py checks the committed notebook equals this script's output.

Run: uv run python scripts/colab_build_public_smoke.py
"""

# ruff: noqa: E501  # cell sources are verbatim notebook text

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "public_smoke.ipynb"


def _load(name: str, path: Path):  # noqa: ANN202 - a module object
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


zs = _load("colab_build_zs_native", ROOT / "scripts" / "colab_build_zs_native.py")
base = zs.base

PINNED_SHA = base.public_pin(base.PIN_PLACEHOLDER)
REPO_URL = base.PUBLIC["repo_url"] if base.PUBLIC else base.URL_PLACEHOLDER
SECRET_WORDS = ("GH_TOKEN", "x-access-token", "ANTHROPIC", "ghp_")  # none may appear in any cell

TITLE = """# public smoke - does the public repository run on a T4? (about 15-20 minutes)

Clones the PUBLIC repository without any token, checks out the pinned commit (`PINNED_SHA`, refused
on any mismatch), installs from `uv.lock`, and decodes the 5 smoke documents of `splits/smoke5.json`
with the production configuration (`qwen35_4b_img_only_native`: Qwen3.5-4B, image only, native
resolution, keyed JSON under a grammar, greedy, seed 42, batch size 1). Then the 7 assertions of
the 02n smoke gate (`shipdoc.smoke`: JSON validity, no truncation, row counts, quantity and total
shares, no all-null rows, finite logprobs for every emitted field). It never scores anything and
never writes to Drive: all outputs stay under `/content`.

Needs: a T4 runtime and `MyDrive/shipdoc-extract/data.zip` (the evaluators' images and labels; only
the 5 smoke documents are read out of it). No secret, no token, no Weights & Biases.

**Parameters:** `DETERMINISM_DOCS` (default 2): that many smoke documents are decoded a second time
in a fresh process and their raw page texts must be byte-identical to the first pass (about 3
minutes extra; 0 skips it).

The last cell prints a banner (pin, python / torch / transformers / xgrammar versions, GPU, smoke
result, seconds, determinism). It holds no document value: paste it back as it is. Timing basis,
stated plainly: the smoke gate took 348.6 s on a T4 in the original v1.5 run (`manifest.json`
`timings.smoke_s`); the install and the model download are NOT MEASURED here.
"""

PARAMS = f"""# Parameters. PINNED_SHA = the PUBLIC repo commit this notebook is pinned to (40-char SHA).
import json
import os
import time

PINNED_SHA = "{PINNED_SHA}"
RUN_NAME = "publicsmoke"
CONFIG = "qwen35_4b_img_only_native"  # production: native resolution (2145 visual tokens per page)
SMOKE_DOCS = "splits/smoke5.json"  # 5 dev docs picked by meta tags (shipdoc.smoke)
SMOKE_LIMIT = 5
DETERMINISM_DOCS = 2  # decoded a second time in a fresh process; 0 = skip
MODE = "run"  # the shared install cell takes these three; nothing here uses W&B or a Hub token
USE_WANDB = False
HF_TOKEN = None  # Qwen/Qwen3.5-4B is public: anonymous download

if not PINNED_SHA or "FILL" in PINNED_SHA:
    raise ValueError("Set PINNED_SHA in the parameters cell (the pin script fills it).")
if not (isinstance(DETERMINISM_DOCS, int) and 0 <= DETERMINISM_DOCS <= SMOKE_LIMIT):
    raise ValueError(f"DETERMINISM_DOCS must be an int in 0..{{SMOKE_LIMIT}}")
SHA7 = PINNED_SHA[:7]
RUN_ID = f"{{RUN_NAME}}_{{CONFIG}}_{{SHA7}}"
T_START = time.time()
"""

MOUNT = """import shutil
import subprocess
import zipfile
from pathlib import Path

from google.colab import drive

drive.mount("/content/drive")
DRIVE_DIR = Path("/content/drive/MyDrive/shipdoc-extract")
if not (DRIVE_DIR / "data.zip").is_file():
    raise FileNotFoundError(
        "Missing MyDrive/shipdoc-extract/data.zip (the evaluators' data package; the repository "
        "README says how to put it there)."
    )
RUNS_DIR = Path("/content/runs")  # local disk, fresh per session: nothing is resumed or kept
RUNS_DIR.mkdir(parents=True, exist_ok=True)
print("data.zip present; outputs go to", RUNS_DIR)
"""

VERIFY = """# Pin verification: the working tree IS the pinned commit (nothing modified); printed in the banner.
dirty = git("status", "--porcelain", "--untracked-files=no")
assert not dirty, f"the clone differs from the pinned commit: {dirty[:200]}"
TREE_SHA = git("rev-parse", "HEAD^{tree}")
assert (REPO / "uv.lock").is_file() and (REPO / "splits" / "smoke5.json").is_file(), "not the repo"
print(f"pin OK: HEAD {head}, tree {TREE_SHA}")
"""

UNZIP = """# Only the smoke documents are read out of data.zip: dev page images and dev labels (the gate
# compares the emitted row counts with the labels' row counts; no label value is printed).
DATA_DIR = Path("/content/data")
ASSIGNMENT_DIR = Path("/content/assignment")  # unused (no scorer here); the install cell sets it
DEV_LABELS = DATA_DIR / "dev" / "labels"
smoke_ids = json.loads((REPO / SMOKE_DOCS).read_text(encoding="utf-8"))
assert len(smoke_ids) == 5 == len(set(smoke_ids)), f"{SMOKE_DOCS}: expected 5 distinct documents"
local_zip = Path("/content/data.zip")
shutil.copyfile(DRIVE_DIR / "data.zip", local_zip)  # local disk: Drive reads are slow
wanted = tuple(
    prefix for d in smoke_ids for prefix in (f"data/dev/images/{d}_p", f"data/dev/labels/{d}.json")
)
with zipfile.ZipFile(local_zip) as zf:
    members = [n for n in zf.namelist() if n.startswith(wanted)]
    zf.extractall("/content", members)
for d in smoke_ids:
    assert (DEV_LABELS / f"{d}.json").is_file(), f"data.zip has no label file for {d}"
    assert list((DATA_DIR / "dev" / "images").glob(f"{d}_p*")), f"data.zip has no images for {d}"
print(f"smoke data OK: {len(smoke_ids)} documents, {len(members)} files extracted")
"""

DETERMINISM = """# DETERMINISM (cheap): the first DETERMINISM_DOCS smoke documents are decoded AGAIN in a fresh
# process (own run id, same config, seed 42, same batch size) and the raw page texts must be
# byte-identical to the smoke pass. Counts and hashes only; informational, never blocks.
import hashlib

DET = {"requested": DETERMINISM_DOCS, "ran": False, "identical": None, "seconds": None}


def page_hashes(run_dir):
    \"\"\"{doc_id: [sha256 of every page's raw text]} of a run's trace (no text is kept).\"\"\"
    out = {}
    for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            t = json.loads(line)
            out[t["doc_id"]] = [
                hashlib.sha256(p["raw_text"].encode()).hexdigest() for p in t["pages"]
            ]
    return out


if DETERMINISM_DOCS and SMOKE_PASSED:
    DET_ID = f"det_{RUN_ID}"
    t_det = time.time()
    det_cmd = [PY, "-m", "shipdoc", "spike", "--config", f"configs/spike_{CONFIG}.yaml",
               "--docs", SMOKE_DOCS, "--split", "dev", "--run-id", DET_ID, "--resume",
               "--limit", str(DETERMINISM_DOCS), "--logprobs"]
    rc_det, tail_det = run_stream(det_cmd, env=SMOKE_ENV)
    DET["seconds"] = round(time.time() - t_det, 1)
    if rc_det != 0:
        print(f"DETERMINISM RUN ERROR: exit {rc_det}; last output: {tail_det[-3:]}")
    else:
        first, second = page_hashes(SMOKE_RUNS / SMOKE_RUN_ID), page_hashes(SMOKE_RUNS / DET_ID)
        DET["ran"] = True
        DET["identical"] = bool(second) and all(first.get(d) == h for d, h in second.items())
        n_pages = sum(len(h) for h in second.values())
        print(f"determinism: {len(second)} documents / {n_pages} pages decoded twice, "
              f"byte-identical raw text: {DET['identical']}")
else:
    print("determinism check skipped (DETERMINISM_DOCS = 0 or the smoke gate did not pass)")
"""

BANNER = """# Banner to paste back. Reads files under /content only; no document value is printed.
import platform
import subprocess

bar = "=" * 78
status_path = RUNS_DIR / f"{RUN_NAME}_smoke_status.json"
status = json.loads(status_path.read_text()) if status_path.is_file() else {}


def out(cmd):
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
        return res.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


gpu = (out(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
       .splitlines() or ["none"])[0]
venv_py = out([PY, "--version"])
probe = ("import importlib.metadata as m, torch; "
         "print(torch.__version__, m.version('transformers'), m.version('xgrammar'))")
torch_v, tf_v, xg_v = (out([PY, "-c", probe]).split() + ["?", "?", "?"])[:3]
state = status.get("state", "not run")
checks = status.get("checks", [])
passed = sum(1 for c in checks if c.get("passed"))
det = globals().get("DET") or {}
det_txt = "skipped"
if det.get("ran"):
    det_txt = f"identical={det['identical']} ({det['seconds']} s)"
print(bar)
print("PUBLIC SMOKE BANNER (paste this block back; it holds no document value)")
print(bar)
print(f"repository   : {REPO_URL}")
tree = globals().get("TREE_SHA", "?")[:12]
print(f"pin          : {PINNED_SHA}  (HEAD == pin asserted; tree {tree})")
print(f"python       : kernel {platform.python_version()}; project venv {venv_py}")
print(f"torch / transformers / xgrammar : {torch_v} / {tf_v} / {xg_v}")
print(f"GPU          : {gpu}")
print(f"config       : {CONFIG}")
print(f"smoke gate   : {state.upper()} - {passed}/{len(checks)} assertions passed, "
      f"{status.get('seconds')} s")
for c in checks:
    print(f"  {'PASS' if c.get('passed') else 'FAIL'}  {c.get('name')}")
print(f"determinism  : {det_txt}")
print(f"wall-clock   : {round(time.time() - T_START)} s since the parameters cell ran")
print(bar)
print(f"DONE {RUN_ID} smoke={state} determinism={det.get('identical')} pin={SHA7}")
assert state == "passed", "SMOKE GATE DID NOT PASS: paste this banner and the tables above back"
"""


def _cells() -> list[tuple[str, str]]:
    clone = base.PUBLIC_CLONE.replace("@@REPO_URL@@", REPO_URL)
    return [
        ("markdown", TITLE),
        ("code", PARAMS),
        ("markdown", base.PUBLIC_ACCOUNT_MD),
        ("code", MOUNT),
        ("code", clone),
        ("code", VERIFY),
        ("code", UNZIP),
        ("code", zs.INSTALL),
        ("code", zs.SMOKE),
        ("code", DETERMINISM),
        ("code", BANNER),
    ]


def build() -> dict:
    """Notebook dict (nbformat 4.5) with all outputs cleared."""
    cells = _cells()
    for _kind, src in cells:  # no secret may ever be named in this notebook's own cells
        bad = [w for w in SECRET_WORDS if w in src]
        assert not bad, f"public smoke notebook mentions {bad}: {src[:60]!r}"
    return base._notebook(cells)


def render() -> str:
    """Canonical JSON text of the notebook."""
    return json.dumps(build(), indent=1, ensure_ascii=False) + "\n"


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(render(), encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")
