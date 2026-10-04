"""Generate notebooks/01_spike.ipynb from the cell sources below (outputs always cleared).

The notebook is a thin wrapper: every cell only orchestrates (Drive, secrets, clone, install,
subprocess per config, summary). All model logic lives in `python -m shipdoc spike`. The cell
sources live here as plain strings because a .ipynb diff is unreadable; tests/test_notebook_spike.py
checks the committed notebook equals this script's output.

The same script also builds notebooks/02_zeroshot500.ipynb (zero-shot over all 500 train+dev docs,
tests/test_zeroshot500.py); its pin placeholder is ZS_PINNED_SHA.

Run: uv run python scripts/colab_build_notebook.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "archive" / "01_spike.ipynb"

# Code commit the notebook clones. The notebook itself is committed in a later PIN commit that
# differs from this commit only in notebooks/ files, so the SHA never refers to itself.
PINNED_SHA = "2abf481be6332e24bee0c8e6b0d16fe8cde55ce4"

# Run order (cheapest first by estimated GPU time) lives in configs/spike_order.json so it can be
# changed without code edits: edit the file, regenerate the notebook. Names are config names
# (configs/spike_<name>.yaml), so a later prompt/format change only needs new config files.
ORDER_FILE = ROOT / "configs" / "spike_order.json"
CONFIGS = json.loads(ORDER_FILE.read_text(encoding="utf-8"))["order"]

TITLE = """# 01 - VLM spike (thin wrapper)

Runs `python -m shipdoc spike` for each config on the spike set, one subprocess per config.
All logic is in the package; this notebook only does Drive, secrets, clone, install and
bookkeeping. `PINNED_SHA` in the parameters cell is the code commit it clones; just
Runtime -> Run all.
Re-running Run all skips configs already `done` and resumes partial ones (see notebooks/README.md).

Flow: GPU-hour / compute-unit estimate table (read it before spending quota) -> MANDATORY
5-doc smoke gate per model (img_only config, 6 assertions + a non-blocking cpn==PO warning; a
model that fails is skipped, all failing stops the notebook) -> spike on every config of the
models that passed -> summary -> paired-bootstrap ranking -> keyed-vs-compact format A/B on
rank #1 (one more spike40 run) -> if the top-2 95% CIs overlap, the top-2 run on dev100 and
are re-ranked -> final pick -> DONE.

Format A/B (GG-approved; motivated by Step L root cause #2, rotations among spn/cpn/po): the
six spike configs use the compact output format. The A/B reruns rank #1 in the keyed format
(full field names in declared order, same model and arm) on spike40 and compares the two with
`scripts/format_ab.py`. Choose KEYED if (row F1 keyed > compact AND the paired CI of delta
row F1 excludes 0) OR (rotations keyed < rotations compact AND OVERALL keyed >= compact, where
"OVERALL not worse" = point delta OVERALL >= 0 OR the paired CI of delta OVERALL includes 0);
otherwise COMPACT on speed. dev100 format policy: rank #1 uses the A/B decision; rank #2 has no
A/B of its own, so it uses compact, EXCEPT when the A/B chose keyed: then rank #2 runs keyed too,
because the effect is format-level. If the keyed run fails or OOMs, everything falls back to
compact and the reason is recorded in `ab_status.json`.
"""

PARAMS = f"""# Parameters. PINNED_SHA = the repo commit this run is pinned to (full 40-char SHA).
PINNED_SHA = "2abf481be6332e24bee0c8e6b0d16fe8cde55ce4"
RUN_GROUP = "spike40"
CONFIGS = {json.dumps(CONFIGS, indent=4)}
DOCS = "splits/spike40.json"
DEV100_GROUP = "dev100"  # the conditional top-2 run uses run_ids dev100_<config>_<sha7>
DEV100_DOCS = "splits/dev100.json"
SMOKE_DOCS = "splits/smoke5.json"  # 5 docs picked from spike40 by meta tags (shipdoc.smoke)
SMOKE_LIMIT = 5
USE_WANDB = True
WANDB_PROJECT = "shipdoc-extract-debug"  # must be a PRIVATE project (checked below)
"""

ACCOUNT_MD = """## Account

**Drive must be mounted as `your-drive-account@example.com`.** The data, OCR cache and run outputs
live in that account's `MyDrive/shipdoc-extract/`. The next cell mounts Drive, tries to read the
account with `gcloud auth list`, and fails if a *different* account is detected. If the account
cannot be detected it prints a reminder: check it yourself in the mount dialog.
"""

MOUNT = """import subprocess
from pathlib import Path

from google.colab import drive

EXPECTED_ACCOUNT = "your-drive-account@example.com"
DRIVE_DIR = Path("/content/drive/MyDrive/shipdoc-extract")
REQUIRED_ZIPS = ["data.zip", "assignment.zip", "ocr_cache.zip"]
RUNS_DIR = DRIVE_DIR / "runs"

drive.mount("/content/drive")

try:
    accounts = subprocess.run(
        ["gcloud", "auth", "list", "--format=value(account)"],
        capture_output=True, text=True, timeout=30, check=False,
    ).stdout.split()
except (OSError, subprocess.TimeoutExpired):
    accounts = []
if accounts and EXPECTED_ACCOUNT not in accounts:
    raise RuntimeError(f"Wrong Google account: gcloud reports {accounts}, need {EXPECTED_ACCOUNT}.")
if accounts:
    print("Account OK:", EXPECTED_ACCOUNT)
else:
    print(f"REMINDER: account not detectable here. Confirm Drive is mounted as {EXPECTED_ACCOUNT}.")

missing = [n for n in REQUIRED_ZIPS if not (DRIVE_DIR / n).is_file()]
if missing:
    raise FileNotFoundError(f"Missing in MyDrive/shipdoc-extract/: {', '.join(missing)}")
print("Drive inputs present:", ", ".join(REQUIRED_ZIPS))
RUNS_DIR.mkdir(parents=True, exist_ok=True)
"""

SECRETS = '''import os

from google.colab import userdata


def get_secret(name: str, required: bool) -> str | None:
    """Read a Colab secret without ever printing it."""
    try:
        value = userdata.get(name)
    except Exception:  # SecretNotFoundError / NotebookAccessError: both mean "not usable"
        value = None
    if required and not value:
        raise RuntimeError(
            f"Colab secret {name} is missing or notebook access is off "
            "(left sidebar -> key icon -> add it and enable Notebook access)."
        )
    return value or None


GH_TOKEN = get_secret("GH_TOKEN", required=True)
WANDB_API_KEY = get_secret("WANDB_API_KEY", required=USE_WANDB)
HF_TOKEN = get_secret("HF_TOKEN", required=False)
print("GH_TOKEN: set | WANDB_API_KEY:", "set" if WANDB_API_KEY else "not set",
      "| HF_TOKEN:", "set" if HF_TOKEN else "not set (optional)")
'''

CLONE = '''import base64
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/example-owner/example-repo.git"
REPO = Path("/content/shipdoc-extract")
if not PINNED_SHA or "FILL" in PINNED_SHA:
    raise ValueError("Set PINNED_SHA in the parameters cell to the commit this run is pinned to.")

# The token goes in an HTTP header passed per command (-c), so it is not in the URL, not echoed
# and not written to .git/config.
_basic = base64.b64encode(f"x-access-token:{GH_TOKEN}".encode()).decode()


def git(*args: str, auth: bool = False) -> str:
    """Run git in the repo dir (or /content); scrub the token from any error text."""
    cmd = ["git"] + (["-c", f"http.extraHeader=Authorization: Basic {_basic}"] if auth else [])
    cwd = REPO if (REPO / ".git").is_dir() else "/content"
    res = subprocess.run(cmd + list(args), cwd=cwd, capture_output=True, text=True, check=False)
    if res.returncode:
        err = res.stderr.replace(_basic, "***").replace(GH_TOKEN, "***")
        raise RuntimeError(f"git {args[0]} failed (exit {res.returncode}): {err[-800:]}")
    return res.stdout.strip()


if (REPO / ".git").is_dir():
    git("fetch", "origin", auth=True)
else:
    git("clone", REPO_URL, str(REPO), auth=True)
git("checkout", PINNED_SHA)
head = git("rev-parse", "HEAD")
pinned_full = git("rev-parse", f"{PINNED_SHA}^{{commit}}")
assert head == pinned_full, f"HEAD {head} != PINNED_SHA {pinned_full}"
print("HEAD =", head)
'''

# --- public build -----------------------------------------------------------------------------
# The PRIVATE repo has no configs/public_notebooks.json: every cell above is unchanged and the
# committed notebooks stay byte-identical. scripts/publish_overlay.py adds that file to the public
# tree only; every builder then emits notebooks that clone the PUBLIC repo anonymously (no GH_TOKEN
# secret, no Drive-account check, no local D: paths) and, for the five native notebooks, are pinned
# to the public pin (scripts/publish_pin_public.py fills it in a second commit).
PUBLIC_CONFIG_PATH = ROOT / "configs" / "public_notebooks.json"
PIN_PLACEHOLDER = "FILL_PINNED_SHA"  # makes the parameters cell of every notebook raise
URL_PLACEHOLDER = "FILL_PUBLIC_REPO_URL"  # makes the clone cell raise
_HEX40 = re.compile(r"[0-9a-f]{40}")


def load_public_config(path: Path = PUBLIC_CONFIG_PATH) -> dict[str, str] | None:
    """The public-build settings ``{"repo_url", "pinned_sha"}``, or None when ``path`` is absent.

    Fails closed: a present but malformed file raises (a half-public build must not be possible).
    ``pinned_sha`` is a full 40-hex commit SHA or ``FILL_PINNED_SHA``; ``repo_url`` is the clone
    URL (or a local path in a dry run) or ``FILL_PUBLIC_REPO_URL``.
    """
    if not path.is_file():
        return None
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if not (isinstance(cfg, dict) and set(cfg) == {"repo_url", "pinned_sha"}):
        raise ValueError(f"{path}: expected exactly the keys repo_url and pinned_sha")
    url, sha = cfg["repo_url"], cfg["pinned_sha"]
    if not (isinstance(url, str) and url.strip() == url and url):
        raise ValueError(f"{path}: repo_url must be a non-empty string without outer spaces")
    if not (isinstance(sha, str) and (sha == PIN_PLACEHOLDER or _HEX40.fullmatch(sha))):
        raise ValueError(
            f"{path}: pinned_sha must be 40 lowercase hex characters or the placeholder"
        )
    return {"repo_url": url, "pinned_sha": sha}


PUBLIC = load_public_config()


def public_pin(private_sha: str) -> str:
    """The pin a native notebook clones: the public one in the public tree, else ``private_sha``."""
    return PUBLIC["pinned_sha"] if PUBLIC else private_sha


PUBLIC_CLONE = '''import subprocess
from pathlib import Path

REPO_URL = "@@REPO_URL@@"
REPO = Path("/content/shipdoc-extract")
if not PINNED_SHA or "FILL" in PINNED_SHA:
    raise ValueError("Set PINNED_SHA in the parameters cell to the commit this run is pinned to.")
if "FILL" in REPO_URL:
    raise ValueError("REPO_URL is a placeholder: the public repository is not published yet.")


def git(*args: str) -> str:
    """Run git in the repo dir (or /content). The repository is public: no credentials at all."""
    cwd = REPO if (REPO / ".git").is_dir() else "/content"
    res = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)
    if res.returncode:
        raise RuntimeError(f"git {args[0]} failed (exit {res.returncode}): {res.stderr[-800:]}")
    return res.stdout.strip()


if (REPO / ".git").is_dir():
    git("fetch", "origin")
else:
    git("clone", REPO_URL, str(REPO))
git("checkout", PINNED_SHA)
head = git("rev-parse", "HEAD")
pinned_full = git("rev-parse", f"{PINNED_SHA}^{{commit}}")
assert head == pinned_full, f"HEAD {head} != PINNED_SHA {pinned_full}"
print("HEAD =", head)
'''
PUBLIC_ACCOUNT_MD = """## Account

No account check in this public build: Drive is mounted as whichever Google account you sign in
with, and the inputs must be in that account's `MyDrive/shipdoc-extract/` (see the repository
README, "Reproduce from images").
"""
if PUBLIC:
    CLONE = PUBLIC_CLONE.replace("@@REPO_URL@@", PUBLIC["repo_url"])

_SECRET_LINE = 'GH_TOKEN = get_secret("GH_TOKEN", required=True)\n'
_ACCOUNT_LINE = re.compile(r'EXPECTED_ACCOUNT = "[^"\n]*"\n')
_ACCOUNT_BLOCK = re.compile(
    r'try:\n    accounts = subprocess\.run\(.*?\nelse:\n    print\(f"REMINDER:[^\n]*\n', re.S
)
_LOCAL_DEST = re.compile(r'BS\.join\(\["D:", "shipdoc", "(runs|submissions)"')
_WIN_DIR = re.compile(r"D:\\shipdoc\\(runs|submissions)")
_PRIVATE_ZS7 = re.compile(r'(?m)^ZS_SHA7 = "FILL_ZS_SHA7".*$')
_PRIVATE_TRAIN7 = re.compile(r'(?m)^TRAIN_SHA7 = "FILL_TRAIN_SHA7".*$')
_PRIVATE_V0 = re.compile(r'(?m)^REUSE_V0_DIR = "submissions/v0_[0-9a-f]+".*$')
_PRIVATE_OCR = re.compile(r'(?m)^REUSE_OCR_FROM = "submissions/v1_[0-9a-f]+".*$')
PUBLIC_FORBIDDEN = ("GH_TOKEN", "x-access-token", "EXPECTED_ACCOUNT", "gauravgandhi", "D:\\shipdoc")


def public_cell(kind: str, source: str) -> str:
    """One notebook cell of the public build: no token, no account check, no local D: paths."""
    if kind == "markdown" and source.startswith("## Account\n"):
        source = PUBLIC_ACCOUNT_MD
    if kind == "code":
        if _SECRET_LINE in source:
            source = source.replace(_SECRET_LINE, "")
            source = source.replace('print("GH_TOKEN: set | ', 'print("')
        if _ACCOUNT_LINE.search(source):
            source = _ACCOUNT_LINE.sub("", source)
            source, n = _ACCOUNT_BLOCK.subn(
                'print("Public build: any Google account; inputs are read from", DRIVE_DIR)\n',
                source,
            )
            assert n == 1, "the account check block of the mount cell changed: update public_cell"
        source = _LOCAL_DEST.sub(lambda m: f'BS.join(["$SHIPDOC_{m[1].upper()}_DIR"', source)
        if 'ZS_SHA7 = "FILL_ZS_SHA7"' in source:  # the parameters cell of a native notebook
            # Run all with the defaults: every native notebook of the public tree shares ONE pin, so
            # the sha7 of the 02n run folder is the pin's; the private-Drive folders do not exist.
            same_pin = "PINNED_SHA[:7]  # public build: one pin for all native notebooks"
            source = _PRIVATE_ZS7.sub(f"ZS_SHA7 = {same_pin}", source)
            source = _PRIVATE_TRAIN7.sub(f"TRAIN_SHA7 = {same_pin}", source)
            source = _PRIVATE_V0.sub(
                "REUSE_V0_DIR = None  # public build: no 1260-token folder", source
            )
            source = _PRIVATE_OCR.sub(
                "REUSE_OCR_FROM = None  # public build: OCR runs here", source
            )
    return _WIN_DIR.sub(lambda m: f"$SHIPDOC_{m[1].upper()}_DIR", source)


UNZIP = """import hashlib
import shutil
import zipfile

LOCAL_ZIPS = Path("/content/zips")
LOCAL_ZIPS.mkdir(exist_ok=True)
# data.zip holds data/, assignment.zip holds assignment/, ocr_cache.zip holds ocr_cache/.
TARGETS = {"data.zip": REPO, "assignment.zip": REPO, "ocr_cache.zip": Path("/content")}
for name, dest in TARGETS.items():
    local = LOCAL_ZIPS / name
    if not local.is_file() or local.stat().st_size != (DRIVE_DIR / name).stat().st_size:
        shutil.copyfile(DRIVE_DIR / name, local)  # local disk: Drive reads are slow
    with zipfile.ZipFile(local) as zf:
        top = zf.namelist()[0].split("/")[0]
        if not (dest / top).is_dir():
            zf.extractall(dest)
    print(f"{name}: unzipped to {dest / top}")

# Verify the OCR cache against its own SHA256SUMS (paths are relative to ocr_cache/).
OCR_CACHE = Path("/content/ocr_cache")
bad = []
sums = (OCR_CACHE / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
for line in sums:
    digest, _, rel = line.partition("  ")
    f = OCR_CACHE / rel
    if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest() != digest:
        bad.append(rel)
if bad:
    raise RuntimeError(f"ocr_cache SHA256SUMS: {len(bad)} bad/missing files, e.g. {bad[:5]}")
print(f"ocr_cache OK: {len(sums)} files verified")
"""

INSTALL = '''import collections
import json
import subprocess

PY = str(REPO / ".venv" / "bin" / "python")
ENV = {
    **os.environ,
    "SHIPDOC_PROFILE": "colab",
    "HF_HOME": "/content/hf_cache",
    "SHIPDOC_RUNS_DIR": str(RUNS_DIR),
    "SHIPDOC_DATA_DIR": str(REPO / "data"),
    "SHIPDOC_ASSIGNMENT_DIR": str(REPO / "assignment"),
    "SHIPDOC_OCR_CACHE": "/content/ocr_cache",
    "SHIPDOC_TMP_DIR": "/content/tmp",
    "UV_CACHE_DIR": "/content/uv_cache",
    "WANDB_PROJECT": WANDB_PROJECT,
    "WANDB_RUN_GROUP": RUN_GROUP,
}
if WANDB_API_KEY:
    ENV["WANDB_API_KEY"] = WANDB_API_KEY
if HF_TOKEN:
    ENV["HF_TOKEN"] = HF_TOKEN
Path(ENV["SHIPDOC_TMP_DIR"]).mkdir(exist_ok=True)


def run_stream(
    cmd: list[str], tail: int = 40, env: dict | None = None
) -> tuple[int, list[str]]:
    """Run a command, echo its output live, return (exit code, last `tail` lines)."""
    last: collections.deque[str] = collections.deque(maxlen=tail)
    with subprocess.Popen(
        cmd, cwd=REPO, env=env or ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    ) as proc:
        for line in proc.stdout:
            print(line, end="")
            last.append(line.rstrip("\\n"))
    return proc.returncode, list(last)


if subprocess.run(["uv", "--version"], capture_output=True, check=False).returncode:
    subprocess.run(["pip", "install", "-q", "uv"], check=True)
rc, _ = run_stream(["uv", "sync", "--frozen", "--group", "vlm", "--python", "3.11"])
assert rc == 0, f"uv sync failed (exit {rc})"

subprocess.run(["nvidia-smi"], check=False)
PROBE = (
    "import json, importlib.metadata as m, torch;"
    "print(json.dumps({'torch': torch.__version__, 'cuda': torch.cuda.is_available(),"
    " 'arch': torch.cuda.get_arch_list(),"
    " **{p: m.version(p) for p in ('transformers', 'bitsandbytes', 'xgrammar')}}))"
)
info = json.loads(subprocess.run([PY, "-c", PROBE], env=ENV, capture_output=True, text=True,
                                 check=True).stdout.strip().splitlines()[-1])
print(info)
assert info["cuda"], "No CUDA device: Runtime -> Change runtime type -> T4 GPU."
# T4 is sm_75: need a native sm_75 kernel or PTX (compute_XX with XX <= 75) it can JIT.
ok = "sm_75" in info["arch"] or any(
    a.startswith("compute_") and int(a.split("_")[1]) <= 75 for a in info["arch"]
)
assert ok, f"torch build has no sm_75 support: {info['arch']}"
'''

WANDB_CELL = '''# Log in to W&B and check the debug project is PRIVATE.
WANDB_CHECK = """
import os, sys
import wandb

wandb.login(key=os.environ["WANDB_API_KEY"], verify=True)
api = wandb.Api()
entity, name = api.default_entity, os.environ["WANDB_PROJECT"]
print("W&B entity:", entity, "| project:", name)
try:
    attrs = dict(getattr(api.project(name, entity), "attrs", {}) or {})
except Exception as exc:  # project absent, or this wandb version lacks api.project
    attrs = {}
    print("project lookup:", type(exc).__name__)
KEYS = ("access", "public", "privacy")
flags = {k: v for k, v in attrs.items() if any(s in k.lower() for s in KEYS)}
print("access-related fields:", flags or "none exposed by the public API")
public = any(v is True and "public" in k.lower() for k, v in flags.items()) or any(
    isinstance(v, str) and v.upper() == "PUBLIC" for v in flags.values()
)
private = any(v is False and "public" in k.lower() for k, v in flags.items()) or any(
    isinstance(v, str) and v.upper() in ("PRIVATE", "TEAM") for v in flags.values()
)
if public:
    sys.exit("PROJECT IS PUBLIC: set it to Private in W&B before running.")
if not private:
    print(f"ACTION FOR GG: the W&B public API does not expose project access. Open "
          f"https://wandb.ai/{entity}/{name}/settings and verify Visibility = Private "
          f"(create the project as Private first, or set the account default project privacy "
          f"to Private; an auto-created project takes the default).")
else:
    print("project access: private OK")
"""
if USE_WANDB:
    res = subprocess.run([PY, "-c", WANDB_CHECK], env=ENV, check=False)
    assert res.returncode == 0, "W&B login or privacy check failed (see output above)."
'''

ESTIMATES = """# GPU-hour and compute-unit (CU) estimate BEFORE any model runs: read it to budget.
# No confirmation prompt (Run all must not block): interrupt the run yourself if it is too high.
import importlib.util
import subprocess

_spec = importlib.util.spec_from_file_location("gpu_estimate", REPO / "scripts" / "gpu_estimate.py")
ge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ge)

SPEED = json.loads((REPO / "configs" / "spike_speed.json").read_text(encoding="utf-8"))
LABELS = REPO / "data" / "dev" / "labels"
spike_ids = json.loads((REPO / DOCS).read_text(encoding="utf-8"))
dev100_ids = json.loads((REPO / DEV100_DOCS).read_text(encoding="utf-8"))
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
print("Runtime GPU (nvidia-smi):", GPU_NAME)
print(ge.report(CONFIGS, SPEED, ge.count_pages(spike_ids, LABELS),
                ge.count_pages(dev100_ids, LABELS), len(spike_ids),
                ge.count_pages(smoke_ids, LABELS), GPU_NAME))
if ge.order_by_estimate(CONFIGS, SPEED) != CONFIGS:
    print("NOTE: CONFIGS order differs from the estimated cheapest-first order.")
"""

SMOKE = """# MANDATORY smoke gate (the fixes under test are unverified on GPU): per model, run its
# img_only compact config on the 5 smoke docs and assert 6 properties of the output
# (shipdoc.smoke). A model that fails is skipped (skipped_smoke_fail) and the rest continue;
# if every model fails the notebook stops here. Resumable via smoke_status.json on Drive.
import sys
import time

import yaml

_smoke_path = REPO / "src" / "shipdoc" / "smoke.py"
_spec = importlib.util.spec_from_file_location("shipdoc_smoke", _smoke_path)
smoke = importlib.util.module_from_spec(_spec)
sys.modules["shipdoc_smoke"] = smoke  # dataclasses resolves annotations through sys.modules
_spec.loader.exec_module(smoke)

SHA7 = PINNED_SHA[:7]
SMOKE_RUNS = RUNS_DIR / "smoke"
SMOKE_STATUS_PATH = RUNS_DIR / "smoke_status.json"
smoke_status = json.loads(SMOKE_STATUS_PATH.read_text()) if SMOKE_STATUS_PATH.is_file() else {}
SMOKE_ENV = {**ENV, "SHIPDOC_RUNS_DIR": str(SMOKE_RUNS)}
SMOKE_ENV.pop("WANDB_API_KEY", None)  # smoke runs never log to W&B
MODELS = list(dict.fromkeys(ge.model_key(c) for c in CONFIGS))
print(f"SMOKE GATE: {len(smoke_ids)} docs {smoke_ids}, models {MODELS}")


def save_smoke_status() -> None:
    tmp = SMOKE_STATUS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(smoke_status, indent=1))
    tmp.replace(SMOKE_STATUS_PATH)  # atomic: a disconnect never leaves a half-written file


for model in MODELS:
    config = f"{model}_img_only_compact"
    run_id = f"smoke_{config}_{SHA7}"
    prev = smoke_status.get(model, {})
    if prev.get("state") == "passed" and prev.get("run_id") == run_id:
        print(f"SKIP smoke {model}: already passed ({run_id})")
        continue
    cfg_path = f"configs/spike_{config}.yaml"
    run_dir = SMOKE_RUNS / run_id
    t0 = time.time()
    rc, tail, checks, warns, error = 0, [], [], [], None
    try:
        max_new = int(yaml.safe_load((REPO / cfg_path).read_text())["max_new_tokens"])
        if not (run_dir / "metrics.json").is_file():  # a finished run is re-checked, not re-run
            print(f"\\n=== {run_id} ===", flush=True)
            cmd = [PY, "-m", "shipdoc", "spike", "--config", cfg_path, "--docs", SMOKE_DOCS,
                   "--split", "dev", "--run-id", run_id, "--resume", "--limit", str(SMOKE_LIMIT)]
            rc, tail = run_stream(cmd, env=SMOKE_ENV)
        if rc == 0:
            checks = smoke.load_and_check(run_dir, LABELS, max_new)
            warns = smoke.load_warnings(run_dir)  # non-blocking: never raises, never fails
    except (OSError, KeyError, ValueError) as exc:  # fail closed: could not verify = fail
        rc, error = 1, f"{type(exc).__name__}: {exc}"
    if rc != 0:
        state = "error"
        print(f"SMOKE ERROR {model}: exit {rc} {error or ''}; last output: {tail[-3:]}")
    else:
        state = "passed" if all(c.passed for c in checks) else "failed"
        print(smoke.format_table(model, checks, warns))
    smoke_status[model] = {
        "state": state,
        "config": config,
        "run_id": run_id,
        "exit_code": rc,
        "seconds": round(time.time() - t0, 1),
        "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail} for c in checks],
        "warnings": [{"name": w.name, "count": w.count, "detail": w.detail} for w in warns],
        "error": error,
        "tail": tail[-5:] if rc else [],
    }
    save_smoke_status()

SMOKE_FAILED_MODELS = {m for m in MODELS if smoke_status[m]["state"] != "passed"}
print("\\nSMOKE SUMMARY")
for m in MODELS:
    v = smoke_status[m]
    bad = [c["name"] for c in v["checks"] if not c["passed"]]
    cpn_po = next((w["count"] for w in v.get("warnings", []) if w["name"] == "cpn_equals_po"), None)
    print(f"  {m.ljust(14)} {v['state'].upper().ljust(7)} {', '.join(bad) or v['error'] or ''}"
          f"  [warn cpn==PO rows: {cpn_po}]")
print("smoke status file:", SMOKE_STATUS_PATH)
if len(SMOKE_FAILED_MODELS) == len(MODELS):
    raise RuntimeError(
        "SMOKE GATE FAILED FOR ALL MODELS: stopping before the full run. GG: report back with "
        f"{SMOKE_STATUS_PATH} and the per-assertion tables printed above (see notebooks/README.md)."
    )
if SMOKE_FAILED_MODELS:
    print(f"WARNING: skipping all configs of {sorted(SMOKE_FAILED_MODELS)} (skipped_smoke_fail).")
"""

LOOP = """import json
import re
import time

SHA7 = PINNED_SHA[:7]
STATUS_PATH = RUNS_DIR / f"{RUN_GROUP}_status.json"
OOM_RE = re.compile(r"out of memory|OutOfMemoryError", re.IGNORECASE)


def run_id_for(config: str, group: str = RUN_GROUP) -> str:
    return f"{group}_{config}_{SHA7}"


def run_group(group: str, configs: list[str], docs: str, status_name: str | None = None) -> dict:
    \"\"\"Run `configs` on `docs` as <group>_<config>_<sha7>; skip done runs, resume partial.

    Configs of a model that failed the smoke gate are recorded as skipped_smoke_fail, not run.
    The status file is <status_name or group>_status.json.
    \"\"\"
    status_path = RUNS_DIR / f"{status_name or group}_status.json"
    status = json.loads(status_path.read_text()) if status_path.is_file() else {}
    ENV["WANDB_RUN_GROUP"] = group
    for config in configs:
        run_id = run_id_for(config, group)
        if config.rsplit("_img_", 1)[0] in SMOKE_FAILED_MODELS:
            status[run_id] = {"state": "skipped_smoke_fail", "exit_code": None, "reason": "smoke"}
            print(f"SKIP {run_id}: model failed the smoke gate (skipped_smoke_fail)")
            tmp = status_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(status, indent=1))
            tmp.replace(status_path)
            continue
        metrics = RUNS_DIR / run_id / "metrics.json"
        if status.get(run_id, {}).get("state") == "done" and metrics.is_file():
            print(f"SKIP {run_id}: already done")
            continue
        cfg_path = f"configs/spike_{config}.yaml"
        cmd = [PY, "-m", "shipdoc", "spike", "--config", cfg_path, "--docs", docs, "--split",
               "dev", "--run-id", run_id, "--resume"] + (["--wandb"] if USE_WANDB else [])
        print(f"\\n=== {run_id} ===", flush=True)
        t0 = time.time()
        if not (REPO / cfg_path).is_file():
            rc, tail = 127, [f"config not found: {cfg_path}"]
            print(tail[0])
        else:
            # One subprocess per config: its CUDA memory is fully released when it exits.
            rc, tail = run_stream(cmd)
        state = "done" if rc == 0 else "failed"
        status[run_id] = {
            "state": state,
            "exit_code": rc,
            "reason": "oom" if rc and OOM_RE.search("\\n".join(tail)) else None,
            "seconds": round(time.time() - t0, 1),
            "tail": tail[-5:] if rc else [],
        }
        tmp = status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(status, indent=1))
        tmp.replace(status_path)  # atomic: a disconnect never leaves a half-written file
        print(f"--- {run_id}: {state} (exit {rc}) ---")
    print("\\nstatus file:", status_path)
    return status


status = run_group(RUN_GROUP, CONFIGS, DOCS)
"""

SUMMARY = """def ci_str(v: dict) -> str:
    return f"{v['point']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]"


def num(v, spec: str = ".3f") -> str:
    # null is a real value in metrics.json (mock runs have no GPU): show it, never hide it.
    return "null" if v is None else format(v, spec)


rows = []
for config in CONFIGS:
    run_id = run_id_for(config)
    path = RUNS_DIR / run_id / "metrics.json"
    if not path.is_file():
        state = status.get(run_id, {}).get("state", "not run")
        rows.append([config, f"NO METRICS (status: {state})"] + [""] * 8)
        continue
    m = json.loads(path.read_text())
    vram = m["peak_vram_max_bytes"]
    rows.append([
        config,
        str(m["n_docs"]),
        ci_str({"point": m["OVERALL"], **{k: m["OVERALL_ci95"][k] for k in ("lo", "hi")}}),
        ci_str(m["scanned"]["ci95"]["OVERALL"]),
        num(m["digital"]["OVERALL"]),
        num(m["false_fill_rate"]),
        num(m["json_validity_rate"]),
        f"{m['s_per_page_mean']:.2f} ({m['s_per_page_p95']:.2f})",
        "null" if vram is None else f"{vram / 1e9:.2f}",
        num(m["n_visual_tokens_mean"], ".1f"),
    ])

header = ["config", "n_docs", "OVERALL [CI]", "scanned OVERALL [CI]", "digital OVERALL",
          "false_fill", "json_validity", "s/page mean (p95)", "peak VRAM GB", "visual tokens mean"]
widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
for r in [header] + rows:
    print("  ".join(str(c).ljust(w) for c, w in zip(r, widths, strict=True)))
failed = [k for k, v in status.items() if v["state"] != "done" and k.startswith(RUN_GROUP + "_")]
if failed:
    print("\\nFAILED configs:", failed, "(details in", STATUS_PATH, ")")
"""

RANK = """# Rank on OVERALL; paired bootstrap (2000, seed 42) of the top two on shared docs.
# Overlapping individual 95% CIs -> run both on dev100 (spec Phase 2.3): scripts/paired_rank.py.


def rank_group(group: str, configs: list[str], docs: str) -> dict:
    out = RUNS_DIR / f"{group}_rank.json"
    out.unlink(missing_ok=True)  # never read a stale ranking from an earlier session
    cmd = [PY, "-m", "scripts.paired_rank", "--runs-dir", str(RUNS_DIR), "--group", group,
           "--sha7", SHA7, "--docs", docs, "--split", "dev", "--out", str(out),
           "--configs", *configs]
    rc, _ = run_stream(cmd)
    if rc not in (0, 2) or not out.is_file():  # 2 = fewer than 2 usable runs (fail closed)
        raise RuntimeError(f"paired_rank failed (exit {rc}); see output above")
    return json.loads(out.read_text())


def show_rank(r: dict, label: str) -> None:
    print(f"\\n{label}: decision = {r['decision']}")
    for e in r.get("excluded", []):
        print(f"  excluded {e['config']}: {e['reason']}")
    if r["decision"] == "insufficient_runs":
        print("  fewer than 2 complete runs: no ranking, no pick")
        return
    for i, x in enumerate(r["ranking"], 1):
        lo, hi = x["ci95"]
        print(f"  {i}. {x['config']}  OVERALL {x['OVERALL']:.3f} [{lo:.3f}, {hi:.3f}]"
              f"  false_fill {x['false_fill_rate']:.3f}  (n={x['n_docs']})")
    lo, hi = r["ci95"]
    print(f"  top-2 {r['top2']}: paired delta (top1 - top2) = {r['delta']:.3f} "
          f"[{lo:.3f}, {hi:.3f}], P(delta <= 0) = {r['p_le_0']:.3f}")
    print(f"  individual 95% CIs overlap: {r['cis_overlap']}; "
          f"paired-delta CI excludes 0: {r['delta_ci_excludes_0']}")


rank = rank_group(RUN_GROUP, CONFIGS, DOCS)
show_rank(rank, f"{RUN_GROUP} ranking")
"""

AB = """# Keyed vs compact format A/B on rank #1 (GG-approved; Step L root cause #2: rotations among
# spn/cpn/po). The keyed config is rank #1's name without `_compact` (same model and arm, full
# field names in declared order). Same run_stream / resume / status mechanics as the spike loop,
# own status file ab_status.json, run_id ab_keyed_<keyed config>_<sha7>. Decision rule and the
# "OVERALL not worse" definition: scripts/format_ab.py (copied in the title cell and spec.md).
# A failed or OOM keyed run, or a failed comparison, falls back to compact and records why.
AB_GROUP = "ab_keyed"
AB_STATUS_NAME = "ab"
AB_RESULT_PATH = RUNS_DIR / "ab_result.json"


def keyed_config(config: str) -> str:
    \"\"\"The keyed (non-compact) config of the same model and arm.\"\"\"
    return config.removesuffix("_compact")


def apply_format(config: str, fmt: str) -> str:
    \"\"\"`config` in the winning format; compact if keyed is chosen but has no config file.\"\"\"
    if fmt != "keyed":
        return config
    keyed = keyed_config(config)
    if not (REPO / "configs" / f"spike_{keyed}.yaml").is_file():
        print(f"NOTE: no keyed config for {config}: it stays compact")
        return config
    return keyed


ab_decision, ab_reason, ab_result = "compact", "no ranking: A/B not run", None
if rank["decision"] != "insufficient_runs":
    top = rank["ranking"][0]["config"]
    keyed = keyed_config(top)
    model = keyed.rsplit("_img_", 1)[0]
    arm = "img_ocr" if keyed.endswith("_img_ocr") else "img_only"
    if keyed == top or not (REPO / "configs" / f"spike_{keyed}.yaml").is_file():
        ab_reason = f"no keyed counterpart of {top}: A/B not run"
    else:
        print(f"A/B: rank #1 = {top}; running keyed {keyed} on {DOCS}")
        ab_status = run_group(AB_GROUP, [keyed], DOCS, status_name=AB_STATUS_NAME)
        ab_run_id = run_id_for(keyed, AB_GROUP)
        st = ab_status.get(ab_run_id, {})
        if st.get("state") != "done":
            ab_reason = (f"keyed run {st.get('state', 'not run')} (exit {st.get('exit_code')}, "
                         f"reason {st.get('reason')}): compact fallback")
        else:
            AB_RESULT_PATH.unlink(missing_ok=True)  # never read a stale result
            cmd = [PY, "-m", "scripts.format_ab", "--compact-dir", str(RUNS_DIR / run_id_for(top)),
                   "--keyed-dir", str(RUNS_DIR / ab_run_id), "--model", model, "--arm", arm,
                   "--split", "dev", "--out", str(AB_RESULT_PATH), "--table"]
            rc, _ = run_stream(cmd)
            if rc != 0 or not AB_RESULT_PATH.is_file():
                ab_reason = f"format_ab failed (exit {rc}): compact fallback"
            else:
                ab_result = json.loads(AB_RESULT_PATH.read_text())
                ab_decision, ab_reason = ab_result["decision"], ab_result["reason"]
print(f"A/B DECISION: {ab_decision} ({ab_reason})")
"""

DEV100 = """# Conditional: only when the spike top-2 CIs overlap. Same resume/status mechanics
# (run_ids dev100_<config>_<sha7>, status file dev100_status.json), then re-rank on dev100.
# Format policy: rank #1 uses the A/B decision; rank #2 has no A/B, so it is compact unless the
# A/B chose keyed, in which case it runs keyed too (the effect is format-level). Both therefore
# use ab_decision.
final_pick = None
if rank["decision"] == "run_dev100":
    top2 = [apply_format(c, ab_decision) for c in rank["top2"]]
    print(f"Top-2 CIs overlap: running {top2} ({ab_decision} format) on {DEV100_DOCS}")
    dev_status = run_group(DEV100_GROUP, top2, DEV100_DOCS)
    rank100 = rank_group(DEV100_GROUP, top2, DEV100_DOCS)
    show_rank(rank100, f"{DEV100_GROUP} ranking")
    if rank100["decision"] != "insufficient_runs":
        final_pick = rank100["ranking"][0]["config"]
elif rank["decision"] == "pick_top1":
    final_pick = apply_format(rank["ranking"][0]["config"], ab_decision)
    print("Top-2 CIs do not overlap: pick top-1, dev100 skipped.")
else:
    print("No pick: fewer than 2 complete spike runs (see excluded runs above).")
print(f"FINAL PICK: {final_pick}")
"""

FINAL = """print(f"\\nDONE {RUN_GROUP}_{SHA7} decision={rank['decision']} final_pick={final_pick}"
      f" ab_decision={ab_decision}")
"""


# --------------------------------------------------------------------------------------------
# 02_zeroshot500: zero-shot Qwen3.5-4B (img_only, KEYED, prompt v2) over all 500 train+dev docs
# --------------------------------------------------------------------------------------------

ZS_OUT = ROOT / "notebooks" / "archive" / "02_zeroshot500.ipynb"

# The pin commit replaces this placeholder with the full 40-char SHA of the pushed code commit
# (then regenerates the notebook). While it holds the placeholder, the notebook refuses to run.
# Same two-commit pattern as 01_spike: the pin commit differs from the code commit only in
# notebooks/ files, so the SHA never refers to itself.
ZS_PINNED_SHA = "42b812b5b09d6e4bff0df12564017f71ffad5fc9"

ZS_TITLE = """# 02 - zero-shot run on all 500 train+dev docs (thin wrapper)

Qwen3.5-4B, image only, KEYED output format, prompt v2 (GG decision). The zero-shot predictions
of ALL 500 docs (671 pages) are (1) the out-of-fold baseline for gate G4 (zero-shot is unseen by
construction) and (2) the training data of the Phase 5 calibrator, which needs per-field token
logprobs: every trace page carries `field_logprobs` (min / mean token logprob of each header and
row field value) and the raw per-token logprobs (see `src/shipdoc/trace.py`).

All logic is in `python -m shipdoc spike --split train+dev --logprobs`; this notebook only does
Drive, secrets, clone, install and bookkeeping. `PINNED_SHA` in the parameters cell is the code
commit it clones (filled by the pin commit); just Runtime -> Run all.

Flow: setup (Drive, clone at the pin, unzip data) -> shard plan -> T4-hour / compute-unit
ESTIMATE per batch-size scenario (printed before anything runs; read it, interrupt if too high)
-> install -> MANDATORY 5-doc smoke gate (7 assertions incl. finite logprobs for every emitted
field; failure stops the notebook before the full run) -> BATCH-SIZE BENCH (12 dev pages at
batch 1/2/4/8: byte-identity vs batch 1, peak VRAM, pages/hour; picks the pages per generate
call; skipped when `BATCH_SIZE` is an int) -> full run, RESUMABLE per doc -> completion banner
(OVERALL, per-split numbers, wall-clock, the Drive folder and the files to download).

Parameters: `BATCH_SIZE` (None = the bench decides, an int = manual and the bench is skipped),
`SHARD` ("i/K": this tab runs document-level shard i of K, "0/1" = everything) and `MODE`
("run", or "merge" to combine the K shard folders into the unsharded run folder after all tabs
finished; refuses unless the shards are complete and identical in code, config and batch size).
For K > 1 start every tab with the same notebook pin; the batch size must be the same in all
tabs (the bench is deterministic given the same code and GPU model, a later tab reuses the
finished bench result on Drive; or set `BATCH_SIZE` to the same int in every tab).

The bench result (batch size) must be used for the TEST submission too: greedy outputs are only
comparable at the same batch size.

If the session dies (disconnect, OOM, quota), open the notebook again and Run all: the setup
cells repeat, the smoke gate and the bench are skipped (they passed), and the full run resumes
after the last document already in `trace.jsonl` on Drive, with the batch size stored in the
run manifest. See notebooks/README.md.
"""

ZS_PARAMS = f"""# Parameters. PINNED_SHA = the repo commit this run is pinned to (full 40-char SHA).
PINNED_SHA = "{ZS_PINNED_SHA}"
RUN_NAME = "zeroshot500"
CONFIG = "qwen35_4b_img_only"  # KEYED output format (full field names), prompt v2: GG decision
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
RUN_ID = f"{{RUN_NAME}}_{{CONFIG}}_keyed_{{SHA7}}"  # the unsharded run (and the merge target)
SHARD_RUN_ID = RUN_ID if SHARD_K == 1 else f"{{RUN_ID}}_shard{{SHARD_I}}of{{SHARD_K}}"  # this tab
"""

ZS_ACCOUNT_MD = ACCOUNT_MD

_ZIPS_OLD = 'REQUIRED_ZIPS = ["data.zip", "assignment.zip", "ocr_cache.zip"]'
assert _ZIPS_OLD in MOUNT
# img_only needs neither OCR text nor the OCR cache; the scorer lives in assignment.zip.
ZS_MOUNT = (
    MOUNT.replace(_ZIPS_OLD, 'REQUIRED_ZIPS = ["data.zip", "assignment.zip"]')
    + """RUN_DIR = RUNS_DIR / (RUN_ID if MODE == "merge" else SHARD_RUN_ID)
print("run_id:", RUN_ID, "| mode:", MODE, "| shard:", SHARD, "| batch size:", BATCH_SIZE or "bench")
print("Drive output folder:", RUN_DIR)
"""
)

ZS_SECRETS = '''import os

from google.colab import userdata


def get_secret(name: str, required: bool) -> str | None:
    """Read a Colab secret without ever printing it."""
    try:
        value = userdata.get(name)
    except Exception:  # SecretNotFoundError / NotebookAccessError: both mean "not usable"
        value = None
    if required and not value:
        raise RuntimeError(
            f"Colab secret {name} is missing or notebook access is off "
            "(left sidebar -> key icon -> add it and enable Notebook access)."
        )
    return value or None


GH_TOKEN = get_secret("GH_TOKEN", required=True)
HF_TOKEN = get_secret("HF_TOKEN", required=False)
WANDB_API_KEY = get_secret("WANDB_API_KEY", required=True) if USE_WANDB else None
print("GH_TOKEN: set | HF_TOKEN:", "set" if HF_TOKEN else "not set (optional)",
      "| W&B:", "ON" if USE_WANDB else "off")
'''

ZS_UNZIP = """import json
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
TRAIN_LABELS, DEV_LABELS = DATA_DIR / "train" / "labels", DATA_DIR / "dev" / "labels"
assert (ASSIGNMENT_DIR / "score.py").is_file(), "assignment.zip has no score.py (official scorer)"
doc_ids = json.loads((REPO / DOCS).read_text(encoding="utf-8"))
assert len(doc_ids) == EXPECTED_DOCS == len(set(doc_ids)), f"{DOCS}: expected {EXPECTED_DOCS} docs"
missing = [d for d in doc_ids
           if not (DATA_DIR / d.split("_")[0] / "labels" / f"{d}.json").is_file()]
assert not missing, f"{len(missing)} docs without a label file, e.g. {missing[:3]}"
print(f"data OK: {len(doc_ids)} docs with labels in train+dev")
"""

ZS_ESTIMATE = """# T4-hour and compute-unit (CU) estimate BEFORE anything runs on the GPU: read it.
# No confirmation prompt (Run all must not block): interrupt the run yourself if it is too high.
import importlib.util
import subprocess

_ge_path = REPO / "scripts" / "gpu_estimate.py"
_spec = importlib.util.spec_from_file_location("gpu_estimate", _ge_path)
ge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ge)

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
full_pages = (
    ge.count_pages([d for d in doc_ids if d.startswith("train_")], TRAIN_LABELS)
    + ge.count_pages([d for d in doc_ids if d.startswith("dev_")], DEV_LABELS)
)
smoke_pages = ge.count_pages(smoke_ids, DEV_LABELS)
if full_pages != EXPECTED_PAGES:
    print(f"WARNING: {full_pages} pages counted, expected {EXPECTED_PAGES}.")
print("Runtime GPU (nvidia-smi):", GPU_NAME)
print(ge.zeroshot_report(SPEED, CONFIG, full_pages, smoke_pages, GPU_NAME))
print(ge.zeroshot_batch_report(SPEED, CONFIG, full_pages, smoke_pages, GPU_NAME,
                               docs_full=len(doc_ids)))
print(f"THIS TAB: shard {SHARD}, batch size {BATCH_SIZE or 'from the bench'}; the table rows "
      "labelled K=2 are two tabs, the CU of the tabs add up.")
"""

ZS_SHARDS = """# Document-level shard plan (shipdoc.shard): docs by (pages desc, doc_id), each
# given to the shard with the fewest pages so far. The CLI recomputes the same plan from
# --shard; this cell only shows it and sets the expected counts the banner checks.
import importlib.util
import sys

_shard_path = REPO / "src" / "shipdoc" / "shard.py"
_spec = importlib.util.spec_from_file_location("shipdoc_shard", _shard_path)
shard_mod = importlib.util.module_from_spec(_spec)
sys.modules["shipdoc_shard"] = shard_mod
_spec.loader.exec_module(shard_mod)

page_counts = [
    (d, len(json.loads((DATA_DIR / d.split("_")[0] / "labels" / f"{d}.json").read_text())["pages"]))
    for d in doc_ids
]
assert sum(n for _, n in page_counts) == EXPECTED_PAGES, "page count differs from EXPECTED_PAGES"
plan = shard_mod.assign_shards(page_counts, SHARD_K)
pages_of = dict(page_counts)
for _i, docs in enumerate(plan):
    mark = "  <- this tab" if _i == SHARD_I and MODE == "run" else ""
    print(f"shard {_i}/{SHARD_K}: {len(docs)} docs, {sum(pages_of[d] for d in docs)} pages{mark}")
SHARD_EXPECTED_DOCS = len(plan[SHARD_I])
SHARD_EXPECTED_PAGES = sum(pages_of[d] for d in plan[SHARD_I])
"""

ZS_INSTALL = '''import collections
import subprocess

PY = str(REPO / ".venv" / "bin" / "python")
ENV = {
    **os.environ,
    "SHIPDOC_PROFILE": "colab",
    "HF_HOME": "/content/hf_cache",
    "SHIPDOC_RUNS_DIR": str(RUNS_DIR),
    "SHIPDOC_DATA_DIR": str(DATA_DIR),
    "SHIPDOC_ASSIGNMENT_DIR": str(ASSIGNMENT_DIR),
    "SHIPDOC_OCR_CACHE": "/content/ocr_cache",  # unused by img_only; the path is never read
    "SHIPDOC_TMP_DIR": "/content/tmp",
    "UV_CACHE_DIR": "/content/uv_cache",
}
if USE_WANDB:
    ENV.update(WANDB_API_KEY=WANDB_API_KEY, WANDB_PROJECT=WANDB_PROJECT, WANDB_RUN_GROUP=RUN_NAME)
if HF_TOKEN:
    ENV["HF_TOKEN"] = HF_TOKEN
Path(ENV["SHIPDOC_TMP_DIR"]).mkdir(exist_ok=True)


def run_stream(
    cmd: list[str], tail: int = 40, env: dict | None = None
) -> tuple[int, list[str]]:
    """Run a command, echo its output live, return (exit code, last `tail` lines)."""
    last: collections.deque[str] = collections.deque(maxlen=tail)
    with subprocess.Popen(
        cmd, cwd=REPO, env=env or ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    ) as proc:
        for line in proc.stdout:
            print(line, end="")
            last.append(line.rstrip("\\n"))
    return proc.returncode, list(last)


if subprocess.run(["uv", "--version"], capture_output=True, check=False).returncode:
    subprocess.run(["pip", "install", "-q", "uv"], check=True)
VLM_GROUP = ["--group", "vlm"] if MODE == "run" else []  # merge mode needs no torch / model stack
rc, _ = run_stream(["uv", "sync", "--frozen", *VLM_GROUP, "--python", "3.11"])
assert rc == 0, f"uv sync failed (exit {rc})"

if MODE == "run":
    subprocess.run(["nvidia-smi"], check=False)
    PROBE = (
        "import json, importlib.metadata as m, torch;"
        "print(json.dumps({'torch': torch.__version__, 'cuda': torch.cuda.is_available(),"
        " 'arch': torch.cuda.get_arch_list(),"
        " **{p: m.version(p) for p in ('transformers', 'xgrammar')}}))"
    )
    info = json.loads(subprocess.run([PY, "-c", PROBE], env=ENV, capture_output=True, text=True,
                                     check=True).stdout.strip().splitlines()[-1])
    print(info)
    assert info["cuda"], "No CUDA device: Runtime -> Change runtime type -> T4 GPU."
    # T4 is sm_75: need a native sm_75 kernel or PTX (compute_XX with XX <= 75) it can JIT.
    ok = "sm_75" in info["arch"] or any(
        a.startswith("compute_") and int(a.split("_")[1]) <= 75 for a in info["arch"]
    )
    assert ok, f"torch build has no sm_75 support: {info['arch']}"
'''

ZS_SMOKE = """# MANDATORY smoke gate (keyed format, prompt v2, logprobs: unverified on a GPU):
# the keyed config on the 5 smoke docs, then 7 assertions on the output (shipdoc.smoke):
# JSON validity, no truncation, row counts, quantity / total share, no all-null rows, and finite
# field logprobs for every emitted field. A failure raises: the full run below is NOT started.
# Resumable via a status file on Drive (a passed smoke for this run id is not repeated).
import importlib.util
import sys
import time

import yaml

_smoke_path = REPO / "src" / "shipdoc" / "smoke.py"
_spec = importlib.util.spec_from_file_location("shipdoc_smoke", _smoke_path)
smoke = importlib.util.module_from_spec(_spec)
sys.modules["shipdoc_smoke"] = smoke  # dataclasses resolves annotations through sys.modules
_spec.loader.exec_module(smoke)

SMOKE_RUNS = RUNS_DIR / "smoke"
SMOKE_RUN_ID = f"smoke_{RUN_ID}"
SMOKE_STATUS_PATH = RUNS_DIR / f"{RUN_NAME}_smoke_status.json"
smoke_status = json.loads(SMOKE_STATUS_PATH.read_text()) if SMOKE_STATUS_PATH.is_file() else {}
SMOKE_ENV = {**ENV, "SHIPDOC_RUNS_DIR": str(SMOKE_RUNS)}
for _k in ("WANDB_API_KEY", "WANDB_PROJECT", "WANDB_RUN_GROUP"):
    SMOKE_ENV.pop(_k, None)  # smoke runs never log to W&B
SMOKE_PASSED = False


def save_smoke_status() -> None:
    tmp = SMOKE_STATUS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(smoke_status, indent=1))
    tmp.replace(SMOKE_STATUS_PATH)  # atomic: a disconnect never leaves a half-written file


if smoke_status.get("state") == "passed" and smoke_status.get("run_id") == SMOKE_RUN_ID:
    SMOKE_PASSED = True
    print(f"SKIP smoke: already passed ({SMOKE_RUN_ID})")
else:
    run_dir = SMOKE_RUNS / SMOKE_RUN_ID
    t0 = time.time()
    rc, tail, checks, warns, error = 0, [], [], [], None
    try:
        max_new = int(yaml.safe_load((REPO / f"configs/spike_{CONFIG}.yaml").read_text())[
            "max_new_tokens"])
        if not (run_dir / "metrics.json").is_file():  # a finished run is re-checked, not re-run
            print(f"=== {SMOKE_RUN_ID}: {len(smoke_ids)} docs ===", flush=True)
            cmd = [PY, "-m", "shipdoc", "spike", "--config", f"configs/spike_{CONFIG}.yaml",
                   "--docs", SMOKE_DOCS, "--split", "dev", "--run-id", SMOKE_RUN_ID, "--resume",
                   "--limit", str(SMOKE_LIMIT), "--logprobs"]
            rc, tail = run_stream(cmd, env=SMOKE_ENV)
        if rc == 0:
            checks = smoke.load_and_check(run_dir, DEV_LABELS, max_new, require_logprobs=True)
            warns = smoke.load_warnings(run_dir)  # non-blocking: never raises, never fails
    except (OSError, KeyError, ValueError) as exc:  # fail closed: could not verify = fail
        rc, error = 1, f"{type(exc).__name__}: {exc}"
    if rc != 0:
        state = "error"
        print(f"SMOKE ERROR: exit {rc} {error or ''}; last output: {tail[-3:]}")
    else:
        state = "passed" if all(c.passed for c in checks) else "failed"
        print(smoke.format_table(CONFIG, checks, warns))
    smoke_status = {
        "state": state,
        "run_id": SMOKE_RUN_ID,
        "exit_code": rc,
        "seconds": round(time.time() - t0, 1),
        "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail} for c in checks],
        "warnings": [{"name": w.name, "count": w.count, "detail": w.detail} for w in warns],
        "error": error,
        "tail": tail[-5:] if rc else [],
    }
    save_smoke_status()
    SMOKE_PASSED = state == "passed"

if not SMOKE_PASSED:
    raise RuntimeError(
        "SMOKE GATE FAILED: the full run is NOT started. GG: report back with "
        f"{SMOKE_STATUS_PATH} and the per-assertion table printed above, plus (they stay on "
        f"Drive, do not paste document values into chat) "
        f"{SMOKE_RUNS / SMOKE_RUN_ID}/trace.jsonl "
        "and predictions.json. Do not edit thresholds to make it pass."
    )
print("SMOKE GATE PASSED: starting the full run.")
"""

ZS_BENCH = """# BATCH-SIZE BENCH (after the smoke). BATCH_SIZE = None: `python -m shipdoc bench`
# runs the real pipeline on the 12 dev pages of splits/bench12.json at batch 1, 2, 4, 8 and
# picks the pages per generate call by the rule printed in the banner (largest size with 100%
# byte-identical outputs vs batch 1 and peak VRAM <= 14.5 GiB; else the near-identical fallback,
# flagged as a DEVIATION; else 1). The result is saved on Drive and reused by a later tab / a
# resumed session (only while code, config and model are the same). BATCH_SIZE = int: manual,
# no bench.
if not SMOKE_PASSED:
    raise RuntimeError(
        "The smoke gate has not passed: the bench and the full run are NOT started."
    )
BENCH_DIR = RUNS_DIR / f"bench_{CONFIG}_{SHA7}"
BENCH_RESULT = None  # path of the bench result the full run is sized by (None = manual)
if BATCH_SIZE is not None:
    BATCH = BATCH_SIZE
    print("=" * 78)
    print(f"MANUAL BATCH SIZE {BATCH}: the bench is SKIPPED, nothing verifies that batch")
    print(f"{BATCH} gives the same outputs as batch 1 (the bench would). Use the same size")
    print("for every shard tab and for the dev / test runs that are compared with this one.")
    print("=" * 78)
else:
    bench_cmd = [PY, "-m", "shipdoc", "bench", "--config", f"configs/spike_{CONFIG}.yaml",
                 "--docs", BENCH_DOCS, "--out-dir", str(BENCH_DIR), "--reuse"]
    rc, tail = run_stream(bench_cmd)
    BENCH_RESULT = BENCH_DIR / "bench_result.json"
    if rc != 0 or not BENCH_RESULT.is_file():
        raise RuntimeError(
            f"BENCH FAILED (exit {rc}); last output: {tail[-3:]}. The full run is NOT started. "
            "Fix it, or set BATCH_SIZE = 1 (the unbatched path) to skip the bench."
        )
    bench_result = json.loads(BENCH_RESULT.read_text())
    BATCH = int(bench_result["chosen_batch_size"])
    if bench_result.get("deviation"):
        print("!" * 78)
        print(bench_result["deviation"]["text"])
        print("!" * 78)
print(f"BATCH SIZE FOR THE FULL RUN: {BATCH} ({'bench' if BENCH_RESULT else 'manual'})")
print("The TEST submission MUST use this same batch size (and the bench result "
      f"{BENCH_RESULT or 'of the dev run'}): greedy outputs are only comparable at equal "
      "batch size.")
"""

ZS_RUN = """# Full run over the documents of this tab (all 500 unless SHARD is i/K), one subprocess,
# RESUMABLE per doc: the CLI appends one line to trace.jsonl and atomically rewrites
# predictions.json / progress.json after every document, and --resume skips the doc_ids
# already in trace.jsonl. A disconnect loses at most the window of documents in flight (a few
# batches; a document is traced when ALL its pages are done). The batch size is stored in
# manifest.json on the first start and a resume reuses it. OOM, disconnect and quota handling:
# notebooks/README.md.
import re
import threading
import time
from datetime import UTC, datetime

if not SMOKE_PASSED:
    raise RuntimeError("The smoke gate has not passed: the full run is NOT started.")
if "BATCH" not in globals():
    raise RuntimeError("The batch-size cell has not run: the full run is NOT started.")
RUN_DIR.mkdir(parents=True, exist_ok=True)
OOM_RE = re.compile(r"out of memory|OutOfMemoryError", re.IGNORECASE)
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
                  f"last {p.get('last_doc')}, "
                  f"{(time.time() - t_start) / 60:.0f} min this session", flush=True)


THIS_DOCS = globals().get("SHARD_EXPECTED_DOCS", EXPECTED_DOCS)  # docs of this tab's shard
done_before = read_progress().get("done", 0)
print(f"=== {SHARD_RUN_ID}: {done_before}/{THIS_DOCS} docs already done; resuming the rest "
      f"(batch size {BATCH}, shard {SHARD}) ===")
session = {"start": datetime.now(UTC).isoformat(), "end": None, "seconds": None,
           "docs_done_at_start": done_before, "exit_code": None}
sessions.append(session)
save_json(SESSIONS_PATH, sessions)
cmd = [PY, "-m", "shipdoc", "spike", "--config", f"configs/spike_{CONFIG}.yaml", "--docs", DOCS,
       "--split", SPLIT, "--run-id", RUN_ID, "--resume", "--logprobs",
       "--batch-size", str(BATCH), "--shard", SHARD]
if BENCH_RESULT is not None:
    cmd += ["--bench-result", str(BENCH_RESULT)]  # recorded in the manifest, checked to match
if USE_WANDB:
    cmd.append("--wandb")
stop = threading.Event()
threading.Thread(target=watch_progress, args=(stop,), daemon=True).start()
t0 = time.time()
try:
    rc, tail = run_stream(cmd)
finally:
    stop.set()
    session.update(end=datetime.now(UTC).isoformat(), seconds=round(time.time() - t0, 1))
    save_json(SESSIONS_PATH, sessions)
session["exit_code"] = rc
save_json(SESSIONS_PATH, sessions)
if rc != 0:
    oom = OOM_RE.search("\\n".join(tail))
    raise RuntimeError(
        f"Full run stopped (exit {rc}{', CUDA out of memory' if oom else ''}) after "
        f"{read_progress().get('done')}/{THIS_DOCS} docs; last output: {tail[-3:]}. Everything "
        "done so far is on Drive: Run all again resumes. If it stops again at the same "
        "document "
        f"({read_progress().get('last_doc')} was the last finished), report back."
    )
print(f"--- full run finished (exit 0) in {(time.time() - t0) / 60:.1f} min this session ---")
"""

ZS_MERGE = """# MODE = 'merge': combine the K shard folders into the unsharded run folder (CPU).
# `python -m shipdoc merge-shards` REFUSES (and writes nothing) unless every shard is complete and
# the shards agree on code SHA, config hash, model revision, seed, batch size and doc list, no
# document is twice, and the merged counts are EXPECTED_DOCS / EXPECTED_PAGES. The merged
# trace.jsonl / predictions.json / metrics.json equal an unsharded run (tests/test_shards.py).
if MODE == "merge":
    merge_cmd = [PY, "-m", "shipdoc", "merge-shards", "--config", f"configs/spike_{CONFIG}.yaml",
                 "--docs", DOCS, "--split", SPLIT, "--run-id", RUN_ID, "--shards", str(SHARD_K),
                 "--expected-docs", str(EXPECTED_DOCS), "--expected-pages", str(EXPECTED_PAGES)]
    rc, tail = run_stream(merge_cmd)
    if rc != 0:
        raise RuntimeError(f"MERGE REFUSED (exit {rc}): {tail[-3:]}. Nothing was merged; fix the "
                           "shard that is named above (rerun it with Run all) and merge again.")
    SESSIONS_PATH = RUN_DIR / "sessions.json"  # the shard sessions, tagged, for the banner
    SMOKE_STATUS_PATH = RUNS_DIR / f"{RUN_NAME}_smoke_status.json"
    smoke_status = json.loads(SMOKE_STATUS_PATH.read_text()) if SMOKE_STATUS_PATH.is_file() else {}
    SMOKE_RUNS, SMOKE_RUN_ID = RUNS_DIR / "smoke", f"smoke_{RUN_ID}"

    def read_progress() -> dict:
        try:
            return json.loads((RUN_DIR / "progress.json").read_text())
        except (OSError, ValueError):
            return {}
else:
    print("MODE = run: nothing to merge in this tab.")
"""

ZS_BANNER = """# Completion banner. Reads only files on Drive: also runnable alone after a resume.
import json

metrics_path = RUN_DIR / "metrics.json"
metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {}
progress = read_progress()
traces = [json.loads(ln) for ln in (RUN_DIR / "trace.jsonl").read_text().splitlines() if ln.strip()]
n_docs = len({t["doc_id"] for t in traces})
pages = [p for t in traces for p in t["pages"]]
with_lp = sum(isinstance(p.get("field_logprobs"), list) for p in pages)
sessions = json.loads(SESSIONS_PATH.read_text()) if SESSIONS_PATH.is_file() else []
secs = sum(s["seconds"] for s in sessions if s.get("seconds") is not None)
unfinished = sum(s.get("seconds") is None for s in sessions)
smoke_secs = smoke_status.get("seconds") or 0
model_secs = sum(p["meta"].get("latency_s") or 0 for p in pages)
manifest_path = RUN_DIR / "manifest.json"
manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
sharded_run = MODE == "run" and SHARD_K > 1  # this folder holds only one shard
exp_docs = globals().get("SHARD_EXPECTED_DOCS", EXPECTED_DOCS) if sharded_run else EXPECTED_DOCS
exp_pages = globals().get("SHARD_EXPECTED_PAGES", EXPECTED_PAGES) if sharded_run else EXPECTED_PAGES
complete = (progress.get("status") == "complete" and n_docs == exp_docs
            and len(pages) == exp_pages and bool(metrics))


def ci(m: dict) -> str:
    return f"{m['OVERALL']:.4f} [{m['OVERALL_ci95']['lo']:.4f}, {m['OVERALL_ci95']['hi']:.4f}]"


bar = "=" * 78
print(bar)
state = "COMPLETE" if complete else "INCOMPLETE: Run all again to resume"
what = "SHARD " + SHARD if sharded_run else ("MERGED SHARDS" if MODE == "merge" else "RUN")
print(f"ZERO-SHOT 500 {what} {state}: {RUN_DIR.name}")
print(bar)
print(f"docs done  : {n_docs} / {exp_docs}")
print(f"pages done : {len(pages)} / {exp_pages}  (with field_logprobs: {with_lp})")
if manifest:
    print(f"batch size : {manifest.get('batch_size')} ({manifest.get('batch_size_source')}); "
          f"code {str(manifest.get('code_sha'))[:12]}; "
          f"config {manifest.get('config', {}).get('hash')}")
    deviation = (manifest.get("bench") or {}).get("deviation")
    if deviation:
        print(deviation["text"])
    print("The TEST submission must use this same batch size "
          "(shipdoc.runmeta.require_same_batch_size).")
if metrics.get("scored"):
    print(f"OVERALL (official scorer, train+dev labels, {metrics['n_docs']} docs): {ci(metrics)}")
    for name, m in metrics.get("per_split", {}).items():
        print(f"  {name:<6}: {m['documents']} docs, {m['n_pages']} pages  OVERALL {ci(m)}")
    print(f"JSON-valid page rate {metrics['json_validity_rate']:.4f}  "
          f"false-fill rate {metrics['false_fill_rate']:.4f}")
else:
    print("OVERALL: not available (metrics.json missing or unscored)")
print(f"wall-clock : {secs / 3600:.2f} h over {len(sessions)} session(s)"
      f"{f' ({unfinished} unfinished, not counted)' if unfinished else ''}"
      f" + smoke {smoke_secs / 60:.1f} min; model time (sum of page latencies) "
      f"{model_secs / 3600:.2f} h")
print(f"Drive folder: {RUN_DIR}")
print("GG: download these files from that folder:")
for name in ("predictions.json", "trace.jsonl", "metrics.json", "progress.json", "sessions.json"):
    path = RUN_DIR / name
    size = f"{path.stat().st_size / 1e6:8.2f} MB" if path.is_file() else "MISSING"
    print(f"  {name:<18}{size}")
print(f"smoke run (not needed): {SMOKE_RUNS / SMOKE_RUN_ID}")
if sharded_run and complete:
    print(f"NEXT: when all {SHARD_K} shard tabs are COMPLETE, set MODE = 'merge' and SHARD = "
          f"'0/{SHARD_K}' (any tab) and Run all: it builds {RUN_ID}.")
print(f"DONE {RUN_DIR.name} docs={n_docs} pages={len(pages)} complete={complete}")
"""


def _run_only(src: str, what: str) -> str:
    """`src` wrapped in ``if MODE == "run":`` (a merge-mode notebook skips the GPU stages).

    Only for cells without multi-line string literals (the indent would end up inside them).
    """
    body = "".join(("    " + ln) if ln.strip() else ln for ln in src.splitlines(True))
    return f'if MODE == "run":\n{body}\nelse:\n    print("MODE = merge: skipped ({what})")\n'


def _cell(kind: str, source: str, idx: int) -> dict:
    cell = {
        "cell_type": kind,
        "id": f"cell{idx:02d}",
        "metadata": {},
        "source": source.splitlines(True),
    }
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    return cell


def build() -> dict:
    """Notebook dict (nbformat 4.5) with all outputs cleared."""
    cells = [
        ("markdown", TITLE),
        ("code", PARAMS),
        ("markdown", ACCOUNT_MD),
        ("code", MOUNT),
        ("code", SECRETS),
        ("code", CLONE),
        ("code", UNZIP),
        ("code", INSTALL),
        ("code", WANDB_CELL),
        ("code", ESTIMATES),
        ("code", SMOKE),
        ("code", LOOP),
        ("code", SUMMARY),
        ("code", RANK),
        ("code", AB),
        ("code", DEV100),
        ("code", FINAL),
    ]
    return _notebook(cells)


def _notebook(cells: list[tuple[str, str]]) -> dict:
    if PUBLIC:  # the single choke point every builder passes its cells through
        cells = [(kind, public_cell(kind, src)) for kind, src in cells]
        for kind, src in cells:
            leaked = [w for w in PUBLIC_FORBIDDEN if w in src]
            assert not leaked, f"public build: {leaked} left in a {kind} cell: {src[:80]!r}"
    return {
        "cells": [_cell(k, s, i) for i, (k, s) in enumerate(cells)],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_zeroshot() -> dict:
    """02_zeroshot500 notebook dict (nbformat 4.5) with all outputs cleared."""
    cells = [
        ("markdown", ZS_TITLE),
        ("code", ZS_PARAMS),
        ("markdown", ZS_ACCOUNT_MD),
        ("code", ZS_MOUNT),
        ("code", ZS_SECRETS),
        ("code", CLONE),
        ("code", ZS_UNZIP),
        ("code", ZS_SHARDS),
        ("code", _run_only(ZS_ESTIMATE, "estimate")),
        ("code", ZS_INSTALL),
        ("code", _run_only(ZS_SMOKE, "smoke gate")),
        ("code", _run_only(ZS_BENCH, "bench")),
        ("code", _run_only(ZS_RUN, "full run")),
        ("code", ZS_MERGE),
        ("code", ZS_BANNER),
    ]
    return _notebook(cells)


def render() -> str:
    """Canonical JSON text of the notebook."""
    return json.dumps(build(), indent=1, ensure_ascii=False) + "\n"


def render_zeroshot() -> str:
    """Canonical JSON text of the 02_zeroshot500 notebook."""
    return json.dumps(build_zeroshot(), indent=1, ensure_ascii=False) + "\n"


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(render(), encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")
    ZS_OUT.write_text(render_zeroshot(), encoding="utf-8", newline="\n")
    print(f"wrote {ZS_OUT}")
