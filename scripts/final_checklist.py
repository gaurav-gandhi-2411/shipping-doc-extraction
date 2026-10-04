"""Final checklist: every deliverable / criterion / constraint of the brief mapped to an artifact.

    uv run python scripts/final_checklist.py            # writes reports/final_checklist.md
    uv run python scripts/final_checklist.py --stdout   # print instead of writing

The checklist is DATA (``ROWS``); each row names a check function and its arguments, and the
evaluator inspects the filesystem / git repo and returns one of ``STATUSES``:

* DONE: the artifact exists and its check passes.
* PARTIAL: it exists but a named piece is pending (the note says which).
* PENDING: not yet produced; ``waits_for`` names what it waits for.
* BLOCKED-ON-GG: produced or producible, but only GG may approve / publish it.
* N/A: not applicable (future work, advisory).
* NOT VERIFIABLE LOCALLY: needs the network or a Colab session (no network is used here).

Output is aggregate / status only: no document values, no confidential brief text (requirement
names are short labels). Read-only: nothing is created outside the report file, no network, no
``git ls-remote``, no write operation on git. External artifacts live under ``D:\\shipdoc``
(override with ``--ext``).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXT = Path("D:/shipdoc")
REPORT_PATH = ROOT / "reports" / "final_checklist.md"
MAX_PAGES = 2  # the brief's page limit for report.pdf

STATUSES = ("DONE", "PARTIAL", "PENDING", "BLOCKED-ON-GG", "N/A", "NOT VERIFIABLE LOCALLY")
KINDS = ("deliverable", "criterion", "constraint")
UNBLOCKERS = (
    "CC can do now",
    "GG runs a notebook",
    "GG approves / publishes",
    "Cannot be verified offline",
)
# Paths that must never be tracked in git (top-level prefixes) and file types that never are.
LEAK_DIRS = ("assignment", "data", "cache", "runs", "submissions")
LEAK_SUFFIXES = {
    "zip": (".zip",),
    "image": (".png", ".jpg", ".jpeg"),
    "pdf": (".pdf",),
}
PAID_API_RE = re.compile(
    r"^\s*(?:import|from)\s+(?:openai|anthropic|cohere|mistralai|google\.generativeai)\b",
    re.MULTILINE,
)
PIN_RE = re.compile(r'PINNED_SHA\s*=\s*\\?"([0-9a-fA-F]{40}|[^"\\]*)\\?"')
HEX40 = re.compile(r"^[0-9a-f]{40}$")
PUBLIC_SMOKE_NB = "public_smoke.ipynb"  # keeps FILL_PINNED_SHA here by design (public repo only)
# The banner the builders put in the title cell of a notebook replaced by a native sibling.
SUPERSEDED_RE = re.compile(r"\*\*SUPERSEDED \(\d{4}-\d{2}-\d{2}\) by ")
ZS_RUN_GLOB = "zeroshot500_*"
NATIVE_MAX_PIXELS = 2196480  # configs/spike_qwen35_4b_img_only_native.yaml (2,145 visual tokens)
LEGACY_MAX_PIXELS = 1310720  # configs/spike_qwen35_4b_img_only.yaml (1,260 visual tokens)


class AmbiguousRunError(RuntimeError):
    """More than one candidate run folder and none was named."""


@dataclass(frozen=True)
class Row:
    """One checklist line. ``check`` is a key of ``CHECKS``; ``args`` its keyword arguments."""

    id: str
    source: str
    name: str
    kind: str
    artifact: str
    check: str
    status_rule: str
    args: Mapping[str, Any] = field(default_factory=dict)
    waits_for: str = ""
    unblock: str = "CC can do now"


@dataclass(frozen=True)
class Result:
    status: str
    note: str = ""


@dataclass
class Ctx:
    """Where to look. ``head_ts`` (unix time of HEAD's commit) is injectable for tests."""

    root: Path = ROOT
    ext: Path = DEFAULT_EXT
    head_ts: float | None = None
    zs_run: str | None = None  # the 02 zero-shot run FOLDER NAME; required once two exist

    def zs_run_name(self) -> str:
        """The one zero-shot run folder; ``ZS_RUN_GLOB`` itself when none exists yet.

        No silent choice: two or more ``zeroshot500_*`` folders (1260-token and native) without
        an explicit ``zs_run`` raise `AmbiguousRunError`.
        """
        if self.zs_run:
            return self.zs_run
        root = self.ext / "runs" / "zeroshot500"
        found = root.glob(ZS_RUN_GLOB) if root.is_dir() else []
        names = sorted(d.name for d in found if d.is_dir())
        if len(names) > 1:
            raise AmbiguousRunError(
                f"{len(names)} zero-shot run folders ({', '.join(names)}): pass --zs-run <folder>"
            )
        return names[0] if names else ZS_RUN_GLOB

    def git(self, *args: str) -> str:
        """Read-only git call in ``root`` (explicit cwd, never ambient); '' on failure."""
        r = subprocess.run(  # noqa: S603
            ["git", *args],  # noqa: S607
            cwd=self.root,
            capture_output=True,
            text=True,
            check=False,
        )
        return r.stdout if r.returncode == 0 else ""

    def head_sha(self) -> str:
        return self.git("rev-parse", "--short", "HEAD").strip() or "unknown"

    def head_time(self) -> float | None:
        if self.head_ts is not None:
            return self.head_ts
        out = self.git("log", "-1", "--format=%ct").strip()
        return float(out) if out else None

    def resolve(self, rel: str) -> list[Path]:
        """``ext:<glob>`` is relative to the external root, anything else to the repo."""
        base, pat = (self.ext, rel[4:]) if rel.startswith("ext:") else (self.root, rel)
        if ZS_RUN_GLOB in pat:
            pat = pat.replace(ZS_RUN_GLOB, self.zs_run_name())
        return sorted(base.glob(pat))

    def older_than_head(self, path: Path) -> bool | None:
        ht = self.head_time()
        return None if ht is None else path.stat().st_mtime < ht


# --------------------------------------------------------------------------------------------
# Check functions: (ctx, **args) -> Result
# --------------------------------------------------------------------------------------------
def chk_exists(ctx: Ctx, paths: Sequence[str], waits: str = "") -> Result:
    """DONE when every path / glob matches something, PENDING (naming the first gap) otherwise."""
    missing = [p for p in paths if not ctx.resolve(p)]
    if missing:
        return Result("PENDING", f"missing: {', '.join(missing)}" + (f"; {waits}" if waits else ""))
    return Result("DONE", f"{len(paths)} artifact(s) present (existence check)")


def chk_text(
    ctx: Ctx, path: str, pattern: str, partial_if: str = "", partial_note: str = ""
) -> Result:
    """File contains ``pattern`` (DONE); also ``partial_if`` -> PARTIAL; no file -> PENDING."""
    found = ctx.resolve(path)
    if not found:
        return Result("PENDING", f"missing: {path}")
    text = found[0].read_text(encoding="utf-8", errors="replace")
    if not re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
        return Result("PENDING", f"{path} lacks /{pattern}/")
    if partial_if and re.search(partial_if, text, re.IGNORECASE):
        return Result("PARTIAL", partial_note or f"{path} matches /{partial_if}/")
    return Result("DONE", f"{path} contains the required section/marker")


def chk_dry_run(ctx: Ctx, paths: Sequence[str], what: str, status: str = "BLOCKED-ON-GG") -> Result:
    """Prepared-but-unpublished artifacts: never DONE.

    All ``paths`` present -> ``status`` with a note starting 'DRY RUN, needs GG approval';
    anything missing -> PENDING naming the first gap. A dry run is not a deliverable.
    """
    missing = [p for p in paths if not ctx.resolve(p)]
    if missing:
        return Result("PENDING", f"{what}: missing {', '.join(missing)}")
    return Result(status, f"DRY RUN, needs GG approval: {what} ({len(paths)} artifact(s) present)")


def chk_exists_else(
    ctx: Ctx, paths: Sequence[str], fallback: Sequence[str], note: str, waits: str = ""
) -> Result:
    """DONE if ``paths`` exist; PARTIAL (``note``) if only ``fallback`` exist; else PENDING."""
    if all(ctx.resolve(p) for p in paths):
        return Result("DONE", f"{len(paths)} artifact(s) present (existence check)")
    if all(ctx.resolve(p) for p in fallback):
        return Result("PARTIAL", note)
    return Result("PENDING", f"missing: {', '.join(paths)}" + (f"; {waits}" if waits else ""))


def chk_fixed(ctx: Ctx, status: str, note: str) -> Result:
    """A status that cannot be derived from local files (stated, with its reason)."""
    if status not in STATUSES:
        raise ValueError(f"unknown status {status!r}")
    return Result(status, note)


def pdf_pages(path: Path) -> int:
    """Page count of a PDF via ``report_to_pdf.count_pdf_pages`` (hand parser, no PDF library)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import report_to_pdf as rp  # noqa: PLC0415  (sibling script, not a package)

    return rp.count_pdf_pages(path.read_bytes())


def pending_markers(text: str) -> int:
    """Unresolved ``[[PENDING:`` / ``[[TO FILL:`` slots plus DUMMY banners in a report markdown.

    ``[[TO FILL:`` is what the final-report builder (``build_report --final-system``) renders for
    an unresolved ``{{ph:id}}``; without it a DRAFT report would read as complete.
    """
    return len(re.findall(r"\[\[(?:PENDING|TO FILL)", text)) + len(
        re.findall(r"DUMMY NUMBERS", text)
    )


def chk_report_pdf(ctx: Ctx, candidates: Sequence[str]) -> Result:
    """First existing candidate PDF: pages <= 2 and its sibling markdown has no pending slots."""
    for cand in candidates:
        found = ctx.resolve(cand)
        if not found:
            continue
        pdf = found[0]
        pages = pdf_pages(pdf)
        if pages > MAX_PAGES:
            return Result("PARTIAL", f"{cand}: {pages} pages, OVER the {MAX_PAGES}-page limit")
        md = pdf.with_suffix(".md")
        n = pending_markers(md.read_text(encoding="utf-8", errors="replace")) if md.exists() else -1
        if n == 0:
            return Result("DONE", f"{cand}: {pages} page(s), no pending slots")
        why = (
            "no sibling markdown to check"
            if n < 0
            else f"DRAFT, {n} pending/DUMMY slot(s) in its source"
        )
        return Result("PARTIAL", f"{cand}: {pages} page(s) <= {MAX_PAGES}; {why}")
    return Result("PENDING", f"no PDF at {' or '.join(candidates)} (PDFs are never committed)")


def chk_report_section(
    ctx: Ctx, heading: str, path: str = "reports/final/report.md", mark: str = ""
) -> Result:
    """The report markdown has the section; DONE only with no pending slots in it.

    ``mark`` (e.g. the dagger of the report's UNVERIFIED legend): occurrences in the section
    body make it PARTIAL, so an unverified figure is never reported as DONE.
    """
    found = ctx.resolve(path)
    if not found:
        return Result("PENDING", f"missing: {path}")
    lines = found[0].read_text(encoding="utf-8", errors="replace").splitlines()
    start = next(
        (i for i, ln in enumerate(lines) if ln.startswith("#") and re.search(heading, ln, re.I)),
        None,
    )
    if start is None:
        return Result("PENDING", f"no heading /{heading}/ in {path}")
    body: list[str] = []
    for ln in lines[start + 1 :]:
        if ln.startswith("#"):
            break
        body.append(ln)
    n = pending_markers("\n".join(body))
    if n:
        return Result("PARTIAL", f"section present; {n} value slot(s) still pending")
    body_text = "\n".join(body)
    if mark and mark in body_text:
        return Result(
            "PARTIAL",
            f"section present, no pending slots; {body_text.count(mark)} figure(s) marked "
            f"{mark} UNVERIFIED",
        )
    return Result("DONE", "section present, no pending slots")


def chk_submission(
    ctx: Ctx, final_prefix: str = "v1", base_prefix: str = "v0", sha256_prefix: str = ""
) -> Result:
    """Validate the newest submission (schema, 200 ids, duplicate keys) with the repo validator.

    ``vN`` dirs are looked up under ``<ext>/submissions``. DONE needs a final (``v1``) file that
    validates; a valid ``v0`` alone is PARTIAL (the safety submission, final one pending).
    ``sha256_prefix``: the file's sha256 must start with it (else PARTIAL, fail closed); the
    first 8 hex digits of the hash are always shown in the note.
    """
    sub = ctx.ext / "submissions"
    finals = sorted(sub.glob(f"{final_prefix}*/test_predictions.json")) if sub.is_dir() else []
    bases = sorted(sub.glob(f"{base_prefix}*/test_predictions.json")) if sub.is_dir() else []
    target = (finals or bases or [None])[-1]
    if target is None:
        return Result("PENDING", "no submission folder (no vN/test_predictions.json)")
    schema, sample = ctx.root / "assignment" / "schema.json", ctx.root / "assignment"
    sample = sample / "sample_submission.json"
    if not (schema.exists() and sample.exists()):
        return Result("NOT VERIFIABLE LOCALLY", "assignment/schema.json or sample missing")
    try:
        sys.path.insert(0, str(ctx.root / "src"))
        from shipdoc import predict  # noqa: PLC0415

        ids = list(json.loads(sample.read_text(encoding="utf-8")))
        rep = predict.validate_file(target, schema, ids)
    except Exception as exc:  # noqa: BLE001  (fail closed: any failure is reported, not hidden)
        return Result("NOT VERIFIABLE LOCALLY", f"validator unavailable: {type(exc).__name__}")
    label = target.parent.name
    detail = (
        f"{label}: schema errors {rep['schema']['n_errors']}, ids {rep['ids']['n_found']}/"
        f"{rep['ids']['n_expected']}, missing {rep['ids']['n_missing']}, "
        f"extra {rep['ids']['n_extra']}, duplicate keys {rep['duplicate_keys']['n']}"
    )
    if not rep["ok"]:
        return Result("PARTIAL", f"{detail}; VALIDATION FAILS")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    detail += f"; sha256 {digest[:8]}..."
    if sha256_prefix and not digest.startswith(sha256_prefix):
        return Result("PARTIAL", f"{detail}; does NOT match the expected {sha256_prefix}...")
    if finals:
        return Result("DONE", f"{detail}; validates")
    return Result("PARTIAL", f"{detail}; validates, but only the v0 safety file exists")


def tracked_files(ctx: Ctx) -> list[str]:
    """Paths tracked at HEAD's index (``git ls-files -z``)."""
    return [p for p in ctx.git("ls-files", "-z").split("\0") if p]


def count_tracked_leaks(paths: Sequence[str]) -> dict[str, int]:
    """Count tracked paths under forbidden top-level dirs and of forbidden file types."""
    counts = {d: 0 for d in LEAK_DIRS} | {k: 0 for k in LEAK_SUFFIXES}
    for p in paths:
        low = p.lower()
        head = low.split("/", 1)[0] if "/" in low else ""
        if head in counts and head in LEAK_DIRS:
            counts[head] += 1
        for kind, sufs in LEAK_SUFFIXES.items():
            if low.endswith(sufs):
                counts[kind] += 1
    return counts


def chk_leaks(ctx: Ctx) -> Result:
    files = tracked_files(ctx)
    if not files:
        return Result("NOT VERIFIABLE LOCALLY", "git ls-files returned nothing")
    counts = count_tracked_leaks(files)
    total = sum(counts.values())
    summary = ", ".join(f"{k}={v}" for k, v in counts.items())
    if total == 0:
        return Result("DONE", f"{len(files)} tracked files; leaks: {summary}")
    return Result("PARTIAL", f"LEAKS TRACKED: {summary}")


def chk_publish(ctx: Ctx, evidence: str = "ext:tmp/publish_scrub.json") -> Result:
    """Dry-run publish evidence file (offending paths empty, history scan exit 0), not re-run."""
    found = ctx.resolve(evidence)
    if not found:
        return Result("NOT VERIFIABLE LOCALLY", f"no evidence file {evidence}; run the dry run")
    try:
        data = json.loads(found[0].read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return Result("NOT VERIFIABLE LOCALLY", "evidence file is not valid JSON")
    n_off = len(data.get("offending_paths", [])) + len(data.get("tree_scan", {}))
    n_hist = len(data.get("history_scan", {}))
    code = data.get("history_scan_exit")
    detail = f"offending paths/tree hits {n_off}, history hits {n_hist}, history exit {code}"
    if n_off or n_hist or code != 0:
        return Result("PARTIAL", f"{detail}; scan NOT clean")
    stale = ctx.older_than_head(found[0])
    if stale is None or stale:
        return Result("PARTIAL", f"{detail}; clean, but the evidence predates HEAD: re-run it")
    return Result("DONE", f"{detail}; evidence is newer than HEAD")


def chk_suite(ctx: Ctx, log: str = "ext:tmp/full_suite2.log") -> Result:
    """Tail of the last full-suite log (never re-run here): passed count, failures, exit code."""
    found = ctx.resolve(log)
    if not found:
        return Result("NOT VERIFIABLE LOCALLY", f"no log {log}")
    tail = found[0].read_text(encoding="utf-8", errors="replace").splitlines()[-8:]
    summ = next((ln for ln in reversed(tail) if re.search(r"\d+ passed", ln)), "")
    m = re.search(r"(\d+) passed", summ)
    failed = re.search(r"(\d+) (?:failed|error)", summ)
    exit0 = any(ln.strip() == "exit=0" for ln in tail)
    if not m:
        return Result("NOT VERIFIABLE LOCALLY", "no pytest summary line in the log tail")
    detail = f"{m.group(1)} passed, failed/errors {failed.group(1) if failed else 0}, exit0={exit0}"
    if failed or not exit0:
        return Result("PARTIAL", f"{detail}; suite NOT green")
    stale = ctx.older_than_head(found[0])
    if stale is None or stale:
        return Result("PARTIAL", f"{detail}; log predates HEAD: re-run after the last commit")
    return Result("DONE", f"{detail}; log newer than HEAD")


def is_superseded(nb_text: str) -> bool:
    """True when the FIRST markdown cell (the title cell) carries the SUPERSEDED banner."""
    try:
        cells = json.loads(nb_text).get("cells", [])
    except (json.JSONDecodeError, AttributeError):
        return False
    first = next((c for c in cells if c.get("cell_type") == "markdown"), None)
    return first is not None and bool(SUPERSEDED_RE.search("".join(first.get("source", []))))


def notebook_pins(ctx: Ctx) -> dict[str, str]:
    """notebook name -> verdict: 'ok', 'placeholder', 'unresolved', 'not-in-builder', 'superseded'.

    'superseded' = the title cell says the notebook is replaced (see `is_superseded`); such a
    notebook is never run, so it is excluded from the real-pin count.
    """
    builders = "\n".join(
        p.read_text(encoding="utf-8", errors="replace")
        for p in sorted((ctx.root / "scripts").glob("colab_build_*.py"))
    )
    out: dict[str, str] = {}
    for nb in sorted((ctx.root / "notebooks").glob("*.ipynb")):
        text = nb.read_text(encoding="utf-8", errors="replace")
        m = PIN_RE.search(text)
        sha = m.group(1) if m else ""
        if is_superseded(text):
            out[nb.name] = "superseded"
        elif nb.name == PUBLIC_SMOKE_NB and not HEX40.match(sha.lower()):
            out[nb.name] = "public-pin"  # pinned only in the public repo (publish_pin_public.py)
        elif not HEX40.match(sha.lower()):
            out[nb.name] = "placeholder"
        elif not ctx.git("cat-file", "-t", f"{sha}^{{commit}}").strip():
            out[nb.name] = "unresolved"
        elif sha not in builders:
            out[nb.name] = "not-in-builder"
        else:
            out[nb.name] = "ok"
    return out


def chk_notebooks(ctx: Ctx) -> Result:
    pins = notebook_pins(ctx)
    if not pins:
        return Result("PENDING", "no notebooks found")
    sup = sorted(k for k, v in pins.items() if v == "superseded")
    pub = sorted(k for k, v in pins.items() if v == "public-pin")
    live = {k: v for k, v in pins.items() if v not in ("superseded", "public-pin")}
    bad = {k: v for k, v in live.items() if v != "ok"}
    tail = (f", {len(sup)} superseded" if sup else "") + (
        f", {len(pub)} public-repo-pin-only ({', '.join(pub)}: pinned at publish time)"
        if pub
        else ""
    )
    if not live:
        return Result("PENDING", f"no live notebooks{tail}")
    if bad:
        return Result(
            "PARTIAL",
            f"{len(live) - len(bad)}/{len(live)} real pins{tail}; "
            + ", ".join(f"{k}={v}" for k, v in bad.items()),
        )
    return Result(
        "DONE",
        f"{len(live)}/{len(live)} real pins{tail}: notebooks pinned to a commit that exists and "
        "that their builder carries (older than HEAD by design; the public squashed repo has "
        "other SHAs, so the pins need a re-pin at publish time)"
        + (f"; superseded, never run, excluded: {', '.join(sup)}" if sup else ""),
    )


def chk_remotes(ctx: Ctx) -> Result:
    """Local remotes only (no network): anything but ``origin`` is unexpected."""
    names = sorted(set(ctx.git("remote").split()))
    extra = [n for n in names if n != "origin"]
    if extra:
        return Result("PARTIAL", f"UNEXPECTED remote(s): {', '.join(extra)}")
    return Result(
        "NOT VERIFIABLE LOCALLY",
        f"remotes: {', '.join(names) or 'none'} (no network call made); that origin is private, "
        "and that nothing is published on GitHub / HF / W&B, cannot be proven offline",
    )


def chk_oof(
    ctx: Ctx, then: str = "DONE", base: Sequence[str] = (), waits: str = "fold1/fold2 OOF"
) -> Result:
    """Fold-1 and fold-2 OOF run folders under ``<ext>/runs`` (name has fold1/fold2 + oof)."""
    runs = ctx.ext / "runs"
    have: list[str] = []
    for fold in ("fold1", "fold2"):
        hit = [
            d
            for d in (runs.iterdir() if runs.is_dir() else [])
            if d.is_dir()
            and fold in d.name.lower()
            and "oof" in d.name.lower()
            and any((d / f).exists() for f in ("metrics.json", "predictions.json"))
        ]
        if hit:
            have.append(fold)
    if len(have) == 2:
        return Result(then, "fold1 and fold2 OOF run folders present")
    base_ok = bool(base) and all(ctx.resolve(b) for b in base)
    missing = [f for f in ("fold1", "fold2") if f not in have]
    note = f"missing OOF: {', '.join(missing)}"
    if base_ok:
        return Result("PARTIAL", f"{note}; baseline artifacts present")
    return Result("PENDING", f"{note}; waits for {waits}")


def _resolution_label(ctx: Ctx, manifest: Mapping[str, Any]) -> str:
    """'config <name> hash <h>: native | 1260-token | max_pixels N | max_pixels unresolved'."""
    from shipdoc.runcompat import resolve_max_pixels  # noqa: PLC0415  (lazy: pulls in spike)

    cfg = manifest.get("config") if isinstance(manifest.get("config"), Mapping) else {}
    mp = resolve_max_pixels(manifest, ctx.root / "configs")
    kind = {NATIVE_MAX_PIXELS: "native", LEGACY_MAX_PIXELS: "1260-token"}.get(
        mp, "max_pixels unresolved" if mp is None else f"max_pixels {mp}"
    )
    return f"config `{cfg.get('name')}` hash `{cfg.get('hash')}`: {kind}"


def _zs_manifest(ctx: Ctx) -> tuple[str, dict[str, Any] | None]:
    from shipdoc.runmeta import read_manifest  # noqa: PLC0415

    name = ctx.zs_run_name()
    d = ctx.ext / "runs" / "zeroshot500" / name
    return name, (read_manifest(d) if d.is_dir() else None)


def chk_zs_provenance(ctx: Ctx) -> Result:
    """Which resolution the zero-shot run was made at, read from its manifest (no silent pick)."""
    name, man = _zs_manifest(ctx)
    if man is None:
        return Result("PENDING", f"no zero-shot run manifest for `{name}`")
    return Result("DONE", f"ZS run `{name}`: {_resolution_label(ctx, man)}")


def chk_native_metric(ctx: Ctx, path: str, pattern: str) -> Result:
    """``pattern`` is a key of the metrics file at ``path`` and its run (manifest) is native.

    The 1,260-token run's metrics have the same keys, so a key-presence check alone cannot tell the
    final system's run from the superseded one: the manifest's config must resolve to the native
    ``max_pixels`` (else PARTIAL, naming what it resolved to; no manifest -> PARTIAL as well).
    """
    from shipdoc.runmeta import read_manifest  # noqa: PLC0415

    found = ctx.resolve(path)
    if not found:
        return Result("PENDING", f"missing: {path}")
    text = found[0].read_text(encoding="utf-8", errors="replace")
    if not re.search(rf'"{re.escape(pattern)}"', text):
        return Result("PENDING", f"{path} lacks the key {pattern}")
    man = read_manifest(found[0].parent)
    if man is None:
        return Result("PARTIAL", f"{found[0].parent.name}: no manifest, resolution unverified")
    label = _resolution_label(ctx, man)
    if not label.endswith(": native"):
        return Result("PARTIAL", f"run `{found[0].parent.name}` is not native: {label}")
    return Result("DONE", f"key {pattern} in the native run `{found[0].parent.name}`: {label}")


def chk_oof_resolution(ctx: Ctx, oof_glob: str = "oof*") -> Result:
    """Every OOF run folder under ``<ext>/runs`` has the zero-shot run's config hash."""
    from shipdoc.runmeta import read_manifest  # noqa: PLC0415

    name, zs = _zs_manifest(ctx)
    runs = ctx.ext / "runs"
    dirs = sorted(d for d in runs.glob(oof_glob) if d.is_dir()) if runs.is_dir() else []
    mans = [(d.name, read_manifest(d)) for d in dirs]
    mans = [(n, m) for n, m in mans if m is not None]
    if zs is None or not mans:
        return Result("PENDING", f"need the ZS run `{name}` and at least one OOF run manifest")
    zh = (zs.get("config") or {}).get("hash")
    parts = [f"{n}: {_resolution_label(ctx, m)}" for n, m in mans]
    bad = [n for n, m in mans if (m.get("config") or {}).get("hash") != zh]
    if bad:
        return Result(
            "PARTIAL",
            f"MIXED-RESOLUTION, differs from the ZS run ({_resolution_label(ctx, zs)}): "
            + "; ".join(parts),
        )
    return Result(
        "DONE", f"{len(mans)} OOF run(s) share the ZS run's config hash; " + "; ".join(parts)
    )


def chk_no_paid_api(ctx: Ctx) -> Result:
    files = sorted((ctx.root / "src").rglob("*.py"))
    if not files:
        return Result("NOT VERIFIABLE LOCALLY", "no src/ files")
    hits = [
        f.name for f in files if PAID_API_RE.search(f.read_text(encoding="utf-8", errors="ignore"))
    ]
    if hits:
        return Result("PARTIAL", f"hosted-API import in {len(hits)} file(s): {', '.join(hits)}")
    return Result("DONE", f"0 hosted-API imports in {len(files)} src files (import scan)")


def chk_no_test_labels(ctx: Ctx) -> Result:
    test = ctx.root / "data" / "test"
    if not test.is_dir():
        return Result("NOT VERIFIABLE LOCALLY", "data/test absent locally")
    label_files = [p for p in test.rglob("*") if p.is_file() and p.suffix == ".json"]
    if (test / "labels").exists() or label_files:
        return Result("PARTIAL", f"label-like files under data/test: {len(label_files)}")
    return Result("DONE", "data/test has images only: 0 label files, no labels dir")


def chk_json_true(ctx: Ctx, path: str, key: str) -> Result:
    found = ctx.resolve(path)
    if not found:
        return Result("PENDING", f"missing: {path}")
    try:
        val = json.loads(found[0].read_text(encoding="utf-8")).get(key)
    except (json.JSONDecodeError, AttributeError):
        return Result("NOT VERIFIABLE LOCALLY", f"{path} unreadable")
    if val is True:
        return Result("DONE", f"{path}: {key} is true")
    return Result("PARTIAL", f"{path}: {key} is {val!r}")


CHECKS: dict[str, Callable[..., Result]] = {
    "exists": chk_exists,
    "text": chk_text,
    "fixed": chk_fixed,
    "dry_run": chk_dry_run,
    "exists_else": chk_exists_else,
    "report_pdf": chk_report_pdf,
    "report_section": chk_report_section,
    "submission": chk_submission,
    "leaks": chk_leaks,
    "publish": chk_publish,
    "suite": chk_suite,
    "notebooks": chk_notebooks,
    "remotes": chk_remotes,
    "oof": chk_oof,
    "zs_provenance": chk_zs_provenance,
    "native_metric": chk_native_metric,
    "oof_resolution": chk_oof_resolution,
    "no_paid_api": chk_no_paid_api,
    "no_test_labels": chk_no_test_labels,
    "json_true": chk_json_true,
}


def evaluate(row: Row, ctx: Ctx) -> Result:
    """Run a row's check; an unknown check or an exception is reported, never a silent pass."""
    fn = CHECKS.get(row.check)
    if fn is None:
        return Result("NOT VERIFIABLE LOCALLY", f"unknown check {row.check!r}")
    try:
        res = fn(ctx, **dict(row.args))
        return Result(res.status, res.note.replace("ext:", "<ext>/"))
    except AmbiguousRunError as exc:  # fail closed, but say what to do
        return Result("NOT VERIFIABLE LOCALLY", f"ambiguous zero-shot run: {exc}")
    except Exception as exc:  # noqa: BLE001  (fail closed)
        return Result("NOT VERIFIABLE LOCALLY", f"check raised {type(exc).__name__}")


# --------------------------------------------------------------------------------------------
# The checklist (brief = 'brief p<N>'; spec = 'spec <section>')
# --------------------------------------------------------------------------------------------
ZS_METRICS = "ext:runs/zeroshot500/zeroshot500_*/metrics.json"
# The final system's zero-shot run (native resolution, 4c17aa3). It lives under tmp/native_extract,
# NOT under runs/zeroshot500 (that folder holds the superseded 1,260-token run).
NATIVE_ZS_RUN = "zeroshot500_qwen35_4b_img_only_native_4c17aa3"
NATIVE_ZS_METRICS = f"ext:tmp/native_extract/zs/{NATIVE_ZS_RUN}/metrics.json"
# The current final report is built per system from the numbers registry (report_to_pdf.py
# --final-system both --out-dir ...); v1.5 is the default system.
REPORT_DIR = "ext:tmp/report_out"
REPORT_PDFS = (f"{REPORT_DIR}/report_v1_5.pdf",)
REPORT_MD = f"{REPORT_DIR}/report_v1_5.md"
PUBLIC_TREE = "ext:tmp/publish_dryrun4/public_repo_rc*"
GG_NB = "GG runs a notebook"
GG_OK = "GG approves / publishes"
OFFLINE = "Cannot be verified offline"
CC = "CC can do now"

ROWS: tuple[Row, ...] = (
    # ---- deliverables stated by the brief -------------------------------------------------
    Row(
        "D01",
        "brief p2",
        "test_predictions.json, all 200 test docs",
        "deliverable",
        "D:/shipdoc/submissions/v15_4c17aa3/test_predictions.json (v1.5)",
        "submission",
        "DONE = the v1.5 file validates (schema, 200 ids, no duplicate keys) and its sha256 "
        "starts d20f69d2",
        args={"final_prefix": "v15", "sha256_prefix": "d20f69d2"},
        waits_for="GG decides v1.5 vs v2 (pooled rule); v2 needs the 04c ft run",
        unblock=GG_NB,
    ),
    Row(
        "D02",
        "brief p2",
        "format of sample_submission.json",
        "deliverable",
        "assignment/schema.json + assignment/sample_submission.json",
        "submission",
        "same validation as D01 (schema + id set + duplicate keys + sha256)",
        args={"final_prefix": "v15", "sha256_prefix": "d20f69d2"},
        waits_for="same as D01",
        unblock=GG_NB,
    ),
    Row(
        "D03",
        "brief p2",
        "report.pdf, two pages at most",
        "deliverable",
        "D:/shipdoc/tmp/report_out/report_v1_5.pdf (draft build, never committed)",
        "report_pdf",
        "DONE = PDF exists, pages <= 2, no placeholders in its markdown (else PARTIAL, DRAFT)",
        args={"candidates": REPORT_PDFS},
        waits_for="the pooled-rule outcome and the three public links, then rebuild with "
        "report_to_pdf.py --final-system v1_5",
        unblock=GG_OK,
    ),
    Row(
        "D04",
        "brief p2",
        "report: approach and why",
        "deliverable",
        "report_v1_5.md section 1",
        "report_section",
        "section present, no pending slots",
        args={"heading": r"approach", "path": REPORT_MD},
        waits_for="none (no pending slots expected)",
        unblock=CC,
    ),
    Row(
        "D05",
        "brief p2",
        "report: dev results by slice",
        "deliverable",
        "report_v1_5.md section 2",
        "report_section",
        "section present, no pending slots",
        args={"heading": r"results by slice", "path": REPORT_MD},
        waits_for="none (no pending slots expected)",
        unblock=CC,
    ),
    Row(
        "D06",
        "brief p2",
        "report: where it fails",
        "deliverable",
        "report_v1_5.md section 7",
        "report_section",
        "section present, no pending slots",
        args={"heading": r"where it fails", "path": REPORT_MD},
        waits_for="none (no pending slots expected)",
        unblock=CC,
    ),
    Row(
        "D07",
        "brief p2",
        "report: new supplier layout tomorrow",
        "deliverable",
        "report_v1_5.md section 6",
        "report_section",
        "section present, no pending slots",
        args={"heading": r"new supplier", "path": REPORT_MD},
        waits_for="none (no pending slots expected)",
        unblock=CC,
    ),
    Row(
        "D08",
        "brief p2",
        "report: review flag precision/recall (optional)",
        "deliverable",
        "report_v1_5.md section 5",
        "report_section",
        "section present, no pending slots",
        args={"heading": r"review flag", "path": REPORT_MD},
        waits_for="none (no pending slots expected)",
        unblock=CC,
    ),
    Row(
        "D09",
        "brief p2-3",
        "report: GitHub, HF, W&B links",
        "deliverable",
        "report_v1_5.md section 10",
        "report_section",
        "section present, no pending slots",
        args={"heading": r"links", "path": REPORT_MD},
        waits_for="the three public URLs (exist only after publishing)",
        unblock=GG_OK,
    ),
    Row(
        "D10",
        "brief p2",
        "GitHub repo with all code",
        "deliverable",
        "public tree D:/shipdoc/tmp/publish_dryrun4/public_repo_rc* + docs/publish_text_a.md",
        "dry_run",
        "never DONE locally: prepared tree + README = BLOCKED-ON-GG (DRY RUN); missing = PENDING",
        args={
            "paths": [PUBLIC_TREE, "docs/publish_text_a.md", "scripts/publish_pin_public.py"],
            "what": "public repository prepared locally (1 commit, 0 remotes), not created",
        },
        waits_for="GG approves, creates the empty public repo, pushes, pins (approval pack)",
        unblock=GG_OK,
    ),
    Row(
        "D11",
        "brief p2",
        "instructions to reproduce from images",
        "deliverable",
        "docs/public_readme.tmpl.md 'Reproduce v1.5 in 5 steps' (public README)",
        "text",
        "section present; [[PUBLISH:key]] placeholders left = PARTIAL",
        args={
            "path": "docs/public_readme.tmpl.md",
            "pattern": r"^## Reproduce v1\.5 in 5 steps",
            "partial_if": r"\[\[PUBLISH:",
            "partial_note": "section present; pin, URLs, licence date and pooled line are filled "
            "at publish time ([[PUBLISH:key]]); the public README is a dry run until then",
        },
        waits_for="publish-time fill (publish_pin_public.py --fill) after GG approves",
        unblock=GG_OK,
    ),
    Row(
        "D12",
        "brief p2",
        "seeded and documented end-to-end rerun",
        "deliverable",
        "configs/finetune_qwen35_4b.yaml seed + README",
        "text",
        "seed present in the configs",
        args={"path": "configs/finetune_qwen35_4b.yaml", "pattern": r"seed"},
    ),
    Row(
        "D13",
        "brief p3",
        "Hugging Face model link + model card",
        "deliverable",
        "docs/publish_text_b.md + scripts/hf_upload.py (cards ready, not uploaded)",
        "dry_run",
        "never DONE locally: cards ready = BLOCKED-ON-GG (DRY RUN); not required if nothing is "
        "trained (v1.5)",
        args={
            "paths": ["docs/publish_text_b.md", "scripts/hf_upload.py"],
            "what": "model cards written and dry-run rendered, nothing uploaded; optional in v1.5",
        },
        waits_for="GG approves card text and the licence re-check, then uploads (v2: required)",
        unblock=GG_OK,
    ),
    Row(
        "D14",
        "brief p3",
        "W&B run link with metrics",
        "deliverable",
        "docs/publish_text_c.md + scripts/wandb_log_public.py (payloads ready, not synced)",
        "dry_run",
        "never DONE locally: payloads ready = BLOCKED-ON-GG (DRY RUN); not required if nothing is "
        "trained (v1.5)",
        args={
            "paths": ["docs/publish_text_c.md", "scripts/wandb_log_public.py"],
            "what": "slice summary and optional fold runs as payload.json only, nothing synced; "
            "optional in v1.5",
        },
        waits_for="GG approves, then syncs with --i-have-gg-approval (v2: required)",
        unblock=GG_OK,
    ),
    Row(
        "D15",
        "brief p3",
        "state if nothing is trained",
        "deliverable",
        "report_v1_5.md header (v1.5: nothing trained; HF and W&B optional)",
        "text",
        "statement present; PARTIAL while the pooled-rule outcome placeholder is open",
        args={
            "path": REPORT_MD,
            "pattern": r"Nothing is trained in the submitted system",
            "partial_if": r"TO FILL: pooled rule outcome",
            "partial_note": "statement present, but it holds only if v1.5 ships: the pooled rule "
            "outcome is not yet filled (v2 would train, and then HF and W&B are required)",
        },
        waits_for="pooled-rule outcome (GG), report rebuild",
        unblock=GG_OK,
    ),
    Row(
        "D16",
        "brief p3",
        "repo/HF/W&B public or access shared",
        "deliverable",
        "GitHub + HF + W&B visibility",
        "fixed",
        "never DONE locally",
        args={"status": "BLOCKED-ON-GG", "note": "visibility is set by GG at publish time"},
        waits_for="GG sets visibility / shares access",
        unblock=GG_OK,
    ),
    # ---- report content items listed in the brief (checked in the v1.5 report markdown) ----
    Row(
        "R01",
        "brief p2",
        "report: seen vs unseen suppliers",
        "deliverable",
        "report_v1_5.md section 2 (seen/unseen rows)",
        "text",
        "section 2 names unseen layouts",
        args={"path": REPORT_MD, "pattern": r"unseen layouts"},
    ),
    Row(
        "R02",
        "brief p2",
        "report: false-fill rate reported",
        "deliverable",
        "report_v1_5.md section 2 (false fills column)",
        "text",
        "report names false fills",
        args={"path": REPORT_MD, "pattern": r"false fill"},
    ),
    Row(
        "R03",
        "brief p2",
        "report: cost per page, measured",
        "deliverable",
        "report_v1_5.md section 8 (model s/page measured; OCR s/page carries the unverified mark)",
        "report_section",
        "section present, no pending slots, no unverified marks (else PARTIAL)",
        args={"heading": r"cost and latency", "path": REPORT_MD, "mark": "\u2020"},
        waits_for="verifier re-computation of the OCR s/page figure; no dollar price by decision",
        unblock=GG_NB,
    ),
    Row(
        "R04",
        "brief p2",
        "report: limitations",
        "deliverable",
        "report_v1_5.md section 9",
        "report_section",
        "section present, no pending slots",
        args={"heading": r"limitations", "path": REPORT_MD},
        waits_for="pooled G line (fold-2 FT OOF) or its removal",
        unblock=GG_OK,
    ),
    Row(
        "R05",
        "brief p2",
        "report: fine-tune evaluated, not shipped",
        "deliverable",
        "report_v1_5.md section 4 (pre-registered pooled rule)",
        "report_section",
        "section present, no pending slots",
        args={"heading": r"fine-tuning", "path": REPORT_MD},
        waits_for="pooled-rule outcome (GG)",
        unblock=GG_OK,
    ),
    # ---- scoring criteria ------------------------------------------------------------------
    Row(
        "C01",
        "brief p2",
        "OVERALL: 0.4 header, 0.4 rows, 0.2 docs",
        "criterion",
        "src/shipdoc/eval.py + tests/test_eval.py + 500-doc run metrics",
        "exists",
        "scorer wrapper, its parity tests and a scored run exist",
        args={"paths": ["src/shipdoc/eval.py", "tests/test_eval.py", ZS_METRICS]},
    ),
    Row(
        "C02",
        "brief p2",
        "header field accuracy",
        "criterion",
        "native 500-doc ZS run metrics.json (zeroshot500_..._native_4c17aa3)",
        "native_metric",
        "key present in the native scored run, whose manifest resolves to the native max_pixels",
        args={"path": NATIVE_ZS_METRICS, "pattern": "header_field_accuracy"},
    ),
    Row(
        "C03",
        "brief p2",
        "line-item row F1",
        "criterion",
        "native 500-doc ZS run metrics.json (zeroshot500_..._native_4c17aa3)",
        "native_metric",
        "key present in the native scored run, whose manifest resolves to the native max_pixels",
        args={"path": NATIVE_ZS_METRICS, "pattern": "row_f1"},
    ),
    Row(
        "C04",
        "brief p2",
        "documents fully correct",
        "criterion",
        "native 500-doc ZS run metrics.json (zeroshot500_..._native_4c17aa3)",
        "native_metric",
        "key present in the native scored run, whose manifest resolves to the native max_pixels",
        args={"path": NATIVE_ZS_METRICS, "pattern": "documents_fully_correct"},
    ),
    Row(
        "C05",
        "brief p2",
        "slices: scan, seen, multipage, repeats, illegible",
        "criterion",
        "reports/dev_slices_native.md + report_v1_5.md section 2",
        "report_section",
        "report section present, no pending slots",
        args={"heading": r"results by slice", "path": REPORT_MD},
        waits_for="none (no pending slots expected)",
    ),
    Row(
        "C06",
        "brief p1,p3",
        "unseen suppliers: about half of test",
        "criterion",
        "reports/layout_test_clusters.md (label-free)",
        "exists",
        "label-free test layout clusters report exists",
        args={"paths": ["reports/layout_test_clusters.md"]},
    ),
    Row(
        "C07",
        "brief p2",
        "false-fill rate, reported separately",
        "criterion",
        "native 500-doc ZS run metrics.json (false_fill_rate)",
        "native_metric",
        "key present in the native scored run, whose manifest resolves to the native max_pixels",
        args={"path": NATIVE_ZS_METRICS, "pattern": "false_fill_rate"},
    ),
    Row(
        "C08",
        "brief p2",
        "knows which fields are unsure",
        "criterion",
        "reports/flags_dev100_native.md + meta/calibrator_zs_native.json + src/shipdoc/flags.py",
        "exists",
        "flag report, frozen calibrator and flag code exist (existence check)",
        args={
            "paths": [
                "reports/flags_dev100_native.md",
                "meta/calibrator_zs_native.json",
                "src/shipdoc/flags.py",
            ]
        },
    ),
    Row(
        "C09",
        "brief p3",
        "correctness on unseen layouts",
        "criterion",
        "splits/folds.json + reports/dev_slices_native.md + reports/g4_native_fold0_verdict.md",
        "exists",
        "supplier-grouped folds, native slice report and fold-0 verdict exist (existence check)",
        args={
            "paths": [
                "splits/folds.json",
                "reports/dev_slices_native.md",
                "reports/g4_native_fold0_verdict.md",
            ]
        },
    ),
    Row(
        "C10",
        "brief p3",
        "honest handling of uncertainty",
        "criterion",
        "null policy + review flags (src/shipdoc/confidence_v3.py, flags.py, review_flags.json)",
        "exists",
        "code and the native calibrator exist (existence check)",
        args={
            "paths": [
                "src/shipdoc/confidence_v3.py",
                "src/shipdoc/flags.py",
                "meta/calibrator_zs_native.json",
            ]
        },
    ),
    Row(
        "C11",
        "brief p3",
        "clean reproducible code",
        "criterion",
        "full test-suite log D:/shipdoc/tmp/full_suite2.log",
        "suite",
        "log tail: passed, 0 failures, exit 0, newer than HEAD",
        waits_for="re-run the full suite after the last commit",
    ),
    Row(
        "C12",
        "brief p3",
        "reasoning about failure modes",
        "criterion",
        "reports/row_errors.md + report section 5",
        "exists",
        "failure-analysis report exists",
        args={"paths": ["reports/row_errors.md"]},
    ),
    # ---- constraints and rules ---------------------------------------------------------------
    Row(
        "K01",
        "brief p2",
        "output exactly what is on the page",
        "constraint",
        "src/shipdoc/postrules.py + tests/test_postrules.py",
        "exists",
        "rules module and its tests exist (not re-run here)",
        args={"paths": ["src/shipdoc/postrules.py", "tests/test_postrules.py"]},
    ),
    Row(
        "K02",
        "brief p2",
        "missing or illegible value is null",
        "constraint",
        "src/shipdoc/confidence_v3.py + reports/calibration_v2.md",
        "exists",
        "null policy code and report exist",
        args={"paths": ["src/shipdoc/confidence_v3.py", "reports/calibration_v2.md"]},
    ),
    Row(
        "K03",
        "brief p2",
        "any row order; repeated rows count",
        "constraint",
        "tests/test_merge.py (no dedup of repeats)",
        "text",
        "merge test mentions repeats",
        args={"path": "tests/test_merge.py", "pattern": r"dup|repeat"},
    ),
    Row(
        "K04",
        "brief p3",
        "open-weight model or technique",
        "constraint",
        "configs/finetune_qwen35_4b.yaml (base model)",
        "text",
        "config names an open-weight base model (its licence is not checked here)",
        args={"path": "configs/finetune_qwen35_4b.yaml", "pattern": r"qwen"},
    ),
    Row(
        "K05",
        "brief p3",
        "inference: one GPU, 24 GB max, or CPU",
        "constraint",
        "reports/respike.md (T4 peak VRAM)",
        "text",
        "VRAM recorded; UNVERIFIED marker = PARTIAL",
        args={
            "path": "reports/respike.md",
            "pattern": r"vram",
            "partial_if": r"UNVERIFIED",
            "partial_note": "peak VRAM recorded but marked UNVERIFIED in the report",
        },
        waits_for="verifier re-computation of the VRAM figures",
        unblock=GG_NB,
    ),
    Row(
        "K06",
        "brief p3",
        "no paid hosted APIs at inference",
        "constraint",
        "src/ import scan",
        "no_paid_api",
        "0 hosted-API imports under src/",
    ),
    Row(
        "K07",
        "brief p3",
        "do not label test images by hand",
        "constraint",
        "data/test (images only)",
        "no_test_labels",
        "0 label files under data/test",
    ),
    Row(
        "K08",
        "brief p3",
        "suggested time 6-8 hours",
        "constraint",
        "n/a (advisory)",
        "fixed",
        "advisory, not enforceable",
        args={"status": "N/A", "note": "suggestion, not a gate"},
    ),
    Row(
        "K09",
        "brief p1-3",
        "confidential: do not distribute (tracked)",
        "constraint",
        "git ls-files: assignment/ data/ cache/ runs/ submissions/ zip image pdf",
        "leaks",
        "DONE = 0 tracked paths in every category",
    ),
    Row(
        "K10",
        "brief p2",
        "score.py is the scorer, unmodified",
        "constraint",
        "assignment/score.py + tests/test_eval.py (parity)",
        "exists",
        "scorer and its parity test exist",
        args={"paths": ["assignment/score.py", "tests/test_eval.py"]},
    ),
    # ---- spec / plan items needed to close ---------------------------------------------------
    Row(
        "S01",
        "spec 11",
        "closing plan pre-registered",
        "deliverable",
        "spec.md section 11",
        "text",
        "heading present",
        args={"path": "spec.md", "pattern": r"^## 11\. Closing plan"},
    ),
    Row(
        "S02",
        "spec 11.1",
        "G4 final-system verdict",
        "deliverable",
        "reports/g4_native_fold0_verdict.md (fold 0 only; the pooled 3-fold rule is open)",
        "fixed",
        "fold-0 verdict exists; the pooled rule needs fold-2 FT out-of-fold: PARTIAL",
        args={
            "status": "PARTIAL",
            "note": "fold-0 verdict written; pooled 3-fold rule not decided (fold-2 FT OOF not "
            "run); v1.5 ships unless the rule selects FT + rules and 04c ft validates by 13:00 IST",
        },
        waits_for="pooled-rule outcome (GG)",
        unblock=GG_OK,
    ),
    Row(
        "S03",
        "spec 3",
        "notebooks pinned to a real commit",
        "deliverable",
        "notebooks/*.ipynb PINNED_SHA vs scripts/colab_build_*.py",
        "notebooks",
        "40-hex, resolves in this repo, present in a builder",
        waits_for="repin to the final commit at the end",
    ),
    Row(
        "S04",
        "spec 11.2",
        "06 resolution sweep result",
        "deliverable",
        "D:/shipdoc/runs/*res*sweep*",
        "exists",
        "a sweep run folder exists and its result is recorded in reports/",
        args={"paths": ["ext:runs/*res*sweep*", "reports/res_sweep.md"]},
        waits_for="GG runs 06_res_sweep",
        unblock=GG_NB,
    ),
    Row(
        "S05",
        "spec 11.4",
        "07 header-hint A/B (conditional)",
        "deliverable",
        "D:/shipdoc/runs/*hdrhint*",
        "exists",
        "an A/B run folder exists",
        args={"paths": ["ext:runs/*hdrhint*"]},
        waits_for="GG runs 07, only if >= 30 column-shift rows remain",
        unblock=GG_NB,
    ),
    Row(
        "S06",
        "spec 11.4",
        "zoom re-read is future work",
        "deliverable",
        "docs/zoom_reread.md (report mentions only)",
        "fixed",
        "excluded from the submission",
        args={"status": "N/A", "note": "future work by decision; not part of this submission"},
    ),
    Row(
        "S07",
        "spec 11.6",
        "review flags in a separate file",
        "deliverable",
        "D:/shipdoc/submissions/v15_4c17aa3/review_flags.json (optional extra)",
        "exists_else",
        "DONE = flags file in the submission folder; PARTIAL = validated copy only",
        args={
            "paths": ["ext:submissions/v15_*/review_flags.json"],
            "fallback": ["ext:tmp/v15_final/copy/review_flags.json"],
            "note": "review_flags.json exists only in the validated scratch copy "
            "(D:/shipdoc/tmp/v15_final/copy; structure check PASS in reports/v1_5_submission.md); "
            "the submission folder is unchanged on purpose (test_predictions.json is the "
            "deliverable, the flags file is an optional extra for the evaluators)",
        },
        waits_for="GG decides whether to ship review_flags.json beside the predictions",
        unblock=GG_OK,
    ),
    Row(
        "S08",
        "spec 5 Ph.6",
        "determinism check, byte-identical rerun",
        "deliverable",
        "reports/v0_validation_report.json (v0)",
        "json_true",
        "determinism_ok true in the v0 validation report",
        args={"path": "reports/v0_validation_report.json", "key": "determinism_ok"},
        waits_for="the same check for v1",
        unblock=GG_NB,
    ),
    Row(
        "S09",
        "spec 8",
        "history leak scan, dry-run publish tree",
        "constraint",
        "D:/shipdoc/tmp/publish_dryrun4/scrub.json (+ public_repo_rc*)",
        "publish",
        "evidence clean and newer than HEAD (the scan is not re-run here)",
        args={"evidence": "ext:tmp/publish_dryrun4/scrub.json"},
        waits_for="re-run scripts/publish_fresh_repo.py after the last commit",
    ),
    Row(
        "S10",
        "spec 8",
        "HF publish script with dry-run",
        "deliverable",
        "scripts/*hf*.py",
        "text",
        "an HF publish script that mentions a dry-run mode exists (not executed here)",
        args={"path": "scripts/*hf*.py", "pattern": r"dry.?run"},
        waits_for="CC writes the dry-run script (publishing itself stays with GG)",
    ),
    Row(
        "S11",
        "spec 6",
        "W&B publish script with dry-run",
        "deliverable",
        "scripts/*wandb*.py",
        "text",
        "a W&B publish script that mentions a dry-run mode exists (not executed here)",
        args={"path": "scripts/*wandb*.py", "pattern": r"dry.?run"},
        waits_for="CC writes the dry-run script (publishing itself stays with GG)",
    ),
    Row(
        "S12",
        "spec 10",
        "nothing published (remotes)",
        "constraint",
        "git remote (names only, no network)",
        "remotes",
        "only origin; privacy and publication state are not provable offline",
        waits_for="GG confirms in the GitHub / HF / W&B consoles",
        unblock=OFFLINE,
    ),
    Row(
        "S13",
        "spec 7",
        "clean-clone rerun on fresh Colab",
        "deliverable",
        "notebooks/public_smoke.ipynb (built; run by GG on a T4 after the public push)",
        "fixed",
        "needs a Colab session",
        args={
            "status": "NOT VERIFIABLE LOCALLY",
            "note": "Colab only; the public smoke notebook is built and unit-tested, never run "
            "on Colab; no network here",
        },
        waits_for="GG runs public_smoke.ipynb from the public repo",
        unblock=GG_NB,
    ),
    Row(
        "S14",
        "spec 10",
        "every number verified, GG approval",
        "deliverable",
        "final report text",
        "fixed",
        "GG gate",
        args={"status": "BLOCKED-ON-GG", "note": "approval of the final text and numbers"},
        waits_for="verifier re-check, then GG approves",
        unblock=GG_OK,
    ),
    Row(
        "S15",
        "GG 2026-10-03",
        "zero-shot run resolution provenance",
        "constraint",
        "D:/shipdoc/runs/zeroshot500/<run>/manifest.json",
        "zs_provenance",
        "the ZS run's config name, hash and native / 1260-token label from its manifest",
        waits_for="the native 02n zero-shot run (pass --zs-run once two runs exist)",
        unblock=GG_NB,
    ),
    Row(
        "S16",
        "GG 2026-10-03",
        "OOF runs share the ZS resolution",
        "constraint",
        "D:/shipdoc/runs/oof*/manifest.json",
        "oof_resolution",
        "every OOF run has the ZS run's config hash (no native-vs-1260 mixing)",
        waits_for="the native 05n OOF runs",
        unblock=GG_NB,
    ),
)


def validate_rows(rows: Sequence[Row]) -> list[str]:
    """Structural problems: duplicate ids, missing source / status rule, bad kind or check."""
    problems: list[str] = []
    seen: set[str] = set()
    for r in rows:
        if r.id in seen:
            problems.append(f"duplicate id {r.id}")
        seen.add(r.id)
        for fld in ("source", "status_rule", "name", "artifact"):
            if not getattr(r, fld).strip():
                problems.append(f"{r.id}: empty {fld}")
        if r.kind not in KINDS:
            problems.append(f"{r.id}: bad kind {r.kind!r}")
        if r.check not in CHECKS:
            problems.append(f"{r.id}: unknown check {r.check!r}")
        if r.unblock not in UNBLOCKERS:
            problems.append(f"{r.id}: bad unblock {r.unblock!r}")
        if len(r.name.split()) > 8 and not r.name.startswith("report"):
            problems.append(f"{r.id}: name longer than 8 words")
    return problems


def run_all(rows: Sequence[Row], ctx: Ctx) -> list[tuple[Row, Result]]:
    return [(r, evaluate(r, ctx)) for r in rows]


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def render(results: Sequence[tuple[Row, Result]], sha: str, date: str, command: str) -> str:
    """Markdown report: provenance, table, summary counts, blocker list."""
    out = [
        "# Final checklist: brief deliverables and criteria",
        "",
        f"Provenance: repo HEAD `{sha}`, generated {date}, command `{command}`. "
        "Aggregate / status only: no document values, no confidential brief text (requirement "
        "names are short labels). Statuses are evaluated from the filesystem and git; nothing is "
        "re-run, no network is used.",
        "",
        "| id | requirement | source | artifact | status | note |",
        "|---|---|---|---|---|---|",
    ]
    for r, res in results:
        out.append(
            f"| {r.id} | {_cell(r.name)} | {r.source} | {_cell(r.artifact)} | {res.status} | "
            f"{_cell(res.note)} |"
        )
    out += ["", "## Summary", "", "| status | count |", "|---|---|"]
    for s in STATUSES:
        out.append(f"| {s} | {sum(1 for _, x in results if x.status == s)} |")
    out.append(f"| total | {len(results)} |")
    out += ["", "| kind | brief rows | spec rows |", "|---|---|---|"]
    for k in KINDS:
        brief = sum(1 for r, _ in results if r.kind == k and r.source.startswith("brief"))
        spec = sum(1 for r, _ in results if r.kind == k and not r.source.startswith("brief"))
        out.append(f"| {k} | {brief} | {spec} |")
    out += ["", "## What blocks completion", ""]
    open_rows = [(r, x) for r, x in results if x.status not in ("DONE", "N/A")]
    for group in UNBLOCKERS:
        items = [(r, x) for r, x in open_rows if r.unblock == group]
        out.append(f"### {group} ({len(items)})")
        out.append("")
        for r, x in items:
            out.append(f"- {r.id} [{x.status}] {r.name}: {r.waits_for or x.note}")
        if not items:
            out.append("- none")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ext", type=Path, default=DEFAULT_EXT, help="external artifact root")
    ap.add_argument("--out", type=Path, default=REPORT_PATH)
    ap.add_argument("--stdout", action="store_true", help="print instead of writing the file")
    ap.add_argument(
        "--zs-run", default=None, help="zero-shot run folder name (required if more than one)"
    )
    a = ap.parse_args(argv)
    problems = validate_rows(ROWS)
    if problems:
        print("checklist structure invalid: " + "; ".join(problems), file=sys.stderr)
        return 2
    ctx = Ctx(root=ROOT, ext=a.ext, zs_run=a.zs_run)
    text = render(
        run_all(ROWS, ctx),
        ctx.head_sha(),
        dt.date.today().isoformat(),
        "uv run python scripts/final_checklist.py",
    )
    if a.stdout:
        sys.stdout.write(text)
    else:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
