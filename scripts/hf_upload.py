"""Prepare (dry-run, default) or perform (GG ONLY) the Hugging Face upload of the LoRA adapter.

    uv run python scripts/hf_upload.py --dry-run
    uv run python scripts/hf_upload.py --upload --i-have-gg-approval --repo-id <owner>/<name>

GG MUST APPROVE ``--upload``. It refuses unless ALL of these hold: the flag
``--i-have-gg-approval``, the environment variable ``SHIPDOC_PUBLISH_OK=1`` and an explicit
``--repo-id``; it also refuses while the card still contains a ``[[PENDING: ...]]`` marker, or when
the staging tree fails any check below. It calls ``huggingface_hub`` lazily, creates the repo
``private=True`` (flipping it public is GG's separate step; there is no flag for it) and never
reads ``HF_TOKEN`` itself (the dry-run touches no credential at all).

Staging tree (default ``$SHIPDOC_TMP_DIR/hf_dryrun``): only an explicit allowlist of file names is
staged (the filled model card as ``README.md`` plus ``adapter_config.json`` and
``adapter_model.safetensors`` of ``<final>/peft``; hardlinked, never trainer_state, optimizer,
checkpoints, ``adapter.pt`` or ``final/manifest.json``, which lists document ids). The adapter files
are verified against ``final/manifest.json`` ``peft_sha256`` first; a mismatch aborts.

Leak check (fail closed): every staged file is magic-byte checked (no zip / image / pdf), every
staged text file and the card are scanned against the train+dev gold label values with the
``scripts/history_leak_scan.py`` matcher (same values, same normalisation, min length 6); if no
gold is available the check fails instead of passing. Messages carry path and field kind, never a
value.

Model card: ``docs/model_card_template.md`` filled ONLY from artifacts (see `build_values`); every
other placeholder becomes an explicit ``[[PENDING: name]]`` marker. The dry-run writes the filled
card to the staging dir and to ``docs/model_card.draft.md`` (the only file this script writes
inside the repo). The card front matter carries ``license: apache-2.0`` (GG's choice; the base
model's licence was read from its HF model card on 2026-10-03).

``--card-file`` stages a finished card instead (no template fill, no draft write; it is leak
checked and its ``[[PENDING: ...]]`` markers still block ``--upload``).

Optional merged fp16 model: ``--merged-dir`` (default off, never created here). A merged
4B-parameter fp16 model is about 8-9 GB (4e9 x 2 bytes + embeddings/vision tower); it is
staged under ``<stage>/merged`` from a suffix allowlist and uploaded to a separate
``--merged-repo-id``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import history_leak_scan as hls  # noqa: E402  (sibling script, not a package)

ROOT = Path(__file__).resolve().parents[1]
TMP_ROOT = Path("D:/shipdoc/tmp")
DEFAULT_STAGE = TMP_ROOT / "hf_dryrun"
DEFAULT_FINAL = Path("D:/shipdoc/runs/ft_fold0/ft_fold0_42b812b_bf16/final")
DEFAULT_RUNS = Path("D:/shipdoc/runs")
TEMPLATE = ROOT / "docs" / "model_card_template.md"
DRAFT_CARD = ROOT / "docs" / "model_card.draft.md"
APPROVAL_FLAG = "--i-have-gg-approval"
APPROVAL_ENV = "SHIPDOC_PUBLISH_OK"

ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
STAGE_ALLOW = frozenset({"README.md", *ADAPTER_FILES})
LINK_MIN_BYTES = 1 << 20  # below this a copy is cheaper than the shared-inode risk
MERGED_SUFFIXES = (".safetensors", ".json", ".txt", ".jinja", ".model")
TEXT_SUFFIXES = (".md", ".json", ".txt", ".jinja", ".yaml", ".yml", ".model")
FORBIDDEN_SUFFIXES = (*hls.FORBIDDEN_SUFFIXES, ".gif", ".bmp", ".tif", ".tiff", ".webp", ".pt")
MAGIC = {
    b"PK\x03\x04": "zip",
    b"%PDF": "pdf",
    b"\x89PNG": "png",
    b"\xff\xd8\xff": "jpeg",
    b"GIF8": "gif",
}
PENDING_RE = re.compile(r"\[\[PENDING: ([^\]]+)\]\]")
PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")
BASE_LICENSE = (
    "Apache-2.0 (Qwen/Qwen3.5-4B model card, fetched 2026-10-03, metadata `license: apache-2.0`)"
)
BANNER = (
    "> DRAFT, UNVERIFIED: filled only from local aggregate artifacts; every double-bracket PENDING"
    " marker is a number or fact that does not exist yet. Not for release as is.\n"
)


class PublishRefused(RuntimeError):
    """``--upload`` was requested without every required approval."""


class StagingError(RuntimeError):
    """The staging tree or an input failed a safety check."""


# --------------------------------------------------------------------------------------------
# gates and paths
# --------------------------------------------------------------------------------------------


def require_upload_approval(approved: bool, env: Mapping[str, str], repo_id: str | None) -> None:
    """Flag, ``SHIPDOC_PUBLISH_OK=1`` and an explicit repo id are all required."""
    missing = []
    if not approved:
        missing.append(APPROVAL_FLAG)
    if env.get(APPROVAL_ENV) != "1":
        missing.append(f"{APPROVAL_ENV}=1")
    if not repo_id:
        missing.append("--repo-id")
    if missing:
        raise PublishRefused(
            f"--upload refused: missing {', '.join(missing)}. GG must approve publishing."
        )


def safe_stage_dir(path: Path, allowed_roots: tuple[Path, ...] | None = None) -> Path:
    """Resolve `path`; refuse anything inside the repo or outside the allowed scratch roots."""
    allowed_roots = allowed_roots or (TMP_ROOT,)  # read at call time (tests override it)
    p = path.resolve()
    if p == ROOT or ROOT in p.parents:
        raise StagingError(f"refusing to stage inside the repository: {p}")
    if not any(p == r.resolve() or r.resolve() in p.parents for r in allowed_roots):
        raise StagingError(f"refusing stage dir {p}: not under an allowed scratch root")
    return p


def assert_repo_write_allowed(path: Path) -> Path:
    """The only file this script may write inside the repo is ``docs/model_card.draft.md``."""
    p = path.resolve()
    if (p == ROOT or ROOT in p.parents) and p != DRAFT_CARD.resolve():
        raise StagingError(f"refusing to write {p} inside the repository")
    return p


def assert_source_allowed(path: Path) -> None:
    """Never stage from under the repo's ``data/`` or ``assignment/``."""
    p = path.resolve()
    for forbidden in ("data", "assignment"):
        root = (ROOT / forbidden).resolve()
        if p == root or root in p.parents:
            raise StagingError(f"refusing to stage from {forbidden}/")


# --------------------------------------------------------------------------------------------
# hashing and verification
# --------------------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_adapter(final_dir: Path) -> dict[str, str]:
    """sha256 of each adapter file, checked against ``final/manifest.json`` ``peft_sha256``."""
    manifest = json.loads((final_dir / "manifest.json").read_text(encoding="utf-8"))
    expected = manifest.get("peft_sha256", {})
    out: dict[str, str] = {}
    for name in ADAPTER_FILES:
        f = final_dir / "peft" / name
        if not f.is_file():
            raise StagingError(f"adapter file missing: {name}")
        if name not in expected:
            raise StagingError(f"manifest.json has no sha256 for {name}")
        got = sha256_file(f)
        if got != expected[name]:
            raise StagingError(f"sha256 mismatch for {name}: adapter differs from its manifest")
        out[name] = got
    return out


# --------------------------------------------------------------------------------------------
# model card
# --------------------------------------------------------------------------------------------


def _f(x: float, nd: int = 4) -> str:
    return f"{x:.{nd}f}"


def _ci(est: Mapping[str, float]) -> str:
    return f"[{_f(est['lo'])}, {_f(est['hi'])}]"


def _read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def build_values(
    final_dir: Path,
    runs_root: Path,
    model_name: str,
    repo_id: str | None,
    adapter_hashes: Mapping[str, str],
) -> dict[str, str]:
    """Placeholder -> text, ONLY for what an artifact states. Missing sources leave no entry."""
    import yaml

    v: dict[str, str] = {"model_name": model_name, "year": "2026"}
    v["citation_key"] = re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")
    man = _read_json(final_dir / "manifest.json")
    inf = man.get("inference_keys", {})
    lora = man.get("lora", {})
    v["code_sha"] = man["code_sha"]
    v["adapter_sha256"] = adapter_hashes["adapter_model.safetensors"]
    v["lora_rank"] = str(lora["r"])
    v["trainable_params"] = f"{lora['n_trainable']:,}"
    v["optimizer_steps"] = str(man["steps"])
    v["prompt_version"] = str(inf["prompt_version"])
    v["max_pixels"] = str(inf["max_pixels"])
    n_train, n_held = man["n_train_docs"], len(man.get("heldout_doc_ids", []))
    if man["stage"] == "final":
        v["released_adapter_scope"] = f"all {n_train} documents"
    else:
        v["released_adapter_scope"] = (
            f"fold {man['fold']} only ({n_train} documents; {n_held} held out), the all-document "
            "adapter is not trained yet"
        )
    ft = yaml.safe_load((ROOT / "configs" / "finetune_qwen35_4b.yaml").read_text(encoding="utf-8"))
    v["epochs"] = str(ft["epochs"])
    ic = ROOT / "configs" / "spike_qwen35_4b_img_only.yaml"
    v["max_new_tokens"] = str(yaml.safe_load(ic.read_text(encoding="utf-8"))["max_new_tokens"])
    v["covered_document_types"] = "commercial invoices and air waybills"
    v["carbon_estimate_or_not_measured"] = "NOT MEASURED"
    v["base_model_license"] = BASE_LICENSE
    if repo_id:
        v["adapter_repo_or_path"] = repo_id
    folds = ROOT / "splits" / "folds.json"
    if folds.is_file():
        v["n_folds"] = str(_read_json(folds)["k"])
    layout = ROOT / "reports" / "layout_test_clusters.md"
    if layout.is_file():
        import report_inputs as ri

        v["n_test_docs"] = str(ri.clusters_from_layout_md(layout.read_text("utf-8"))["n_test_docs"])
    rows = [
        json.loads(ln)
        for ln in (final_dir.parent / "metrics.jsonl").read_text("utf-8").splitlines()
        if ln.strip()
    ]
    secs = sum(r["seconds"] for r in rows if isinstance(r.get("seconds"), int | float))
    v["train_gpu_hours"] = _f(secs / 3600, 2) + " (sum of per-step seconds, UNVERIFIED)"
    zs_dir = runs_root / "zeroshot500" / "zeroshot500_qwen35_4b_img_only_keyed_42b812b"
    if (zs_dir / "metrics.json").is_file():
        m = _read_json(zs_dir / "metrics.json")
        v["n_docs"], v["n_pages"] = str(m["n_docs"]), str(m["n_pages"])
        v["n_train_docs"] = str(m["per_split"]["train"]["documents"])
        v["n_dev_docs"] = str(m["per_split"]["dev"]["documents"])
        v["zs_overall"] = _f(m["OVERALL_ci95"]["point"])
        v["zs_ci"] = _ci(m["OVERALL_ci95"])
        v["zs_header"] = _f(m["slices"]["all"]["header_field_accuracy"])
        v["zs_rows"] = _f(m["slices"]["all"]["row_f1"])
        v["metrics_artifact"] = f"{zs_dir.name}/metrics.json"
        v["metrics_commit"] = m["git_commit"]
        v["n_eval_docs"] = (
            "[[PENDING: n_eval_docs]] (the baseline cells below are over all "
            f"{m['n_docs']} train+dev documents, not yet restricted to the same held-out documents)"
        )
        if (zs_dir / "sessions.json").is_file():
            ss = _read_json(zs_dir / "sessions.json")
            total = sum(s["seconds"] for s in ss if isinstance(s.get("seconds"), int | float))
            v["zs_gpu_hours"] = _f(total / 3600, 2) + " (session wall-clock, UNVERIFIED)"
    return v


def apply_template_edits(text: str) -> str:
    """The explicit, tested edits to the template that are not placeholders."""
    text = re.sub(r"^# TEMPLATE:.*\n# the section.*\n", "", text, flags=re.MULTILINE)
    text = text.replace(" (one of: fold K, all {{n_docs}} documents)", "")
    text = text.replace("`<PINNED_SHA>`", "`[[PENDING: PINNED_SHA]]`")
    return text.replace("# {{model_name}}\n", "# {{model_name}}\n\n" + BANNER, 1)


def fill_card(template: str, values: Mapping[str, str]) -> tuple[str, list[str], list[str]]:
    """Fill `{{x}}` from `values`; every other placeholder becomes ``[[PENDING: x]]``.

    Returns (card text, resolved placeholder names, pending names incl. pre-existing markers).
    """
    text = apply_template_edits(template)
    resolved: list[str] = []

    def sub(m: re.Match[str]) -> str:
        name = m.group(1)
        if name in values:
            resolved.append(name)
            return values[name]
        return f"[[PENDING: {name}]]"

    text = PLACEHOLDER_RE.sub(sub, text)
    pending = list(dict.fromkeys(PENDING_RE.findall(text)))
    return text, list(dict.fromkeys(resolved)), pending


# --------------------------------------------------------------------------------------------
# staging and the leak check
# --------------------------------------------------------------------------------------------


def _link_or_copy(src: Path, dst: Path) -> None:
    """Hardlink big files (same volume), copy small ones; identical existing file kept, else abort.

    A hardlink shares the inode, so writing into a staged big file would change the original: the
    small files (editable text) are therefore always copied.
    """
    if dst.exists():
        if dst.stat().st_size == src.stat().st_size and sha256_file(dst) == sha256_file(src):
            return
        raise StagingError(f"{dst.name} already exists in the staging dir and differs")
    if src.stat().st_size >= LINK_MIN_BYTES:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass  # cross-volume or no hardlink support: fall through to a copy
    shutil.copy2(src, dst)


def stage_tree(
    final_dir: Path, stage_dir: Path, card_text: str, merged_dir: Path | None = None
) -> list[Path]:
    """Create the would-be upload tree from the allowlist only; returns the staged files."""
    assert_source_allowed(final_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "README.md").write_text(card_text, encoding="utf-8", newline="\n")
    staged = [stage_dir / "README.md"]
    for name in ADAPTER_FILES:
        _link_or_copy(final_dir / "peft" / name, stage_dir / name)
        staged.append(stage_dir / name)
    if merged_dir is not None:
        assert_source_allowed(merged_dir)
        (stage_dir / "merged").mkdir(exist_ok=True)
        for f in sorted(merged_dir.iterdir()):
            if f.is_file() and f.suffix in MERGED_SUFFIXES:
                _link_or_copy(f, stage_dir / "merged" / f.name)
                staged.append(stage_dir / "merged" / f.name)
    return staged


def gold_matcher(labels_root: Path, min_len: int = hls.DEFAULT_MIN_LEN) -> hls.AhoCorasick:
    """Matcher over the gold values; fails closed when no gold is available."""
    values = hls.collect_values(labels_root, min_len)
    if not values:
        raise StagingError(f"no gold label values under {labels_root}: cannot run the leak check")
    return hls.AhoCorasick(values)


def leak_kinds(text: str, matcher: hls.AhoCorasick) -> frozenset[str]:
    """Field kinds of any gold value occurring in `text` (never the value)."""
    return matcher.kinds_in(text)


def check_staging(stage_dir: Path, matcher: hls.AhoCorasick, allow_merged: bool) -> list[str]:
    """Problems in the staging tree (empty list: clean). Path and kind only, never a value."""
    problems: list[str] = []
    for f in sorted(p for p in stage_dir.rglob("*") if p.is_file()):
        rel = f.relative_to(stage_dir).as_posix()
        in_merged = rel.startswith("merged/") and allow_merged
        if not in_merged and rel not in STAGE_ALLOW:
            problems.append(f"{rel}: not in the staging allowlist")
        elif in_merged and f.suffix not in MERGED_SUFFIXES:
            problems.append(f"{rel}: suffix not in the merged allowlist")
        reason = hls.flagged_path_kind(rel)
        if reason or f.suffix.lower() in FORBIDDEN_SUFFIXES:
            problems.append(f"{rel}: forbidden path or type")
        with f.open("rb") as fh:
            head = fh.read(8)
        for magic, kind in MAGIC.items():
            if head.startswith(magic):
                problems.append(f"{rel}: content is a {kind} file")
        if f.suffix in TEXT_SUFFIXES and f.stat().st_size < (50 << 20):
            kinds = leak_kinds(f.read_text(encoding="utf-8", errors="ignore"), matcher)
            if kinds:
                problems.append(f"{rel}: gold label value(s) present [{'/'.join(sorted(kinds))}]")
    return problems


# --------------------------------------------------------------------------------------------
# upload (never reached in a dry-run)
# --------------------------------------------------------------------------------------------


def hub_upload(stage_dir: Path, repo_id: str, merged_repo_id: str | None) -> None:
    """Upload the staged tree with huggingface_hub; the repo is created PRIVATE."""
    from huggingface_hub import HfApi  # noqa: PLC0415  (lazy: dry-run never imports it)

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(
        folder_path=str(stage_dir),
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=["merged/*"],
    )
    if merged_repo_id:
        api.create_repo(merged_repo_id, repo_type="model", private=True, exist_ok=True)
        api.upload_folder(
            folder_path=str(stage_dir / "merged"), repo_id=merged_repo_id, repo_type="model"
        )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="default: stage locally only")
    mode.add_argument("--upload", action="store_true", help="GG only; see the module docstring")
    ap.add_argument(APPROVAL_FLAG, dest="approved", action="store_true")
    ap.add_argument("--repo-id", help="owner/name, required for --upload")
    ap.add_argument("--merged-dir", type=Path, default=None, help="optional merged fp16 model")
    ap.add_argument("--merged-repo-id", help="separate repo for the merged model")
    ap.add_argument("--final-dir", type=Path, default=DEFAULT_FINAL)
    ap.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS)
    ap.add_argument("--stage-dir", type=Path, default=DEFAULT_STAGE)
    ap.add_argument("--labels-root", type=Path, default=ROOT / "data")
    ap.add_argument("--draft-card", type=Path, default=DRAFT_CARD)
    ap.add_argument(
        "--card-file",
        type=Path,
        default=None,
        help="finished card to stage as README.md instead of filling the template",
    )
    ap.add_argument("--model-name", default="shipdoc-extract-qwen3.5-4b-lora")
    return ap


def main(argv: list[str] | None = None, env: Mapping[str, str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    env = os.environ if env is None else env
    if args.upload:
        try:
            require_upload_approval(args.approved, env, args.repo_id)
            if args.merged_dir and not args.merged_repo_id:
                raise PublishRefused("--upload refused: --merged-dir needs --merged-repo-id")
        except PublishRefused as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 3
    else:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")  # the dry-run never talks to the Hub
    try:
        stage_dir = safe_stage_dir(args.stage_dir)
        draft = assert_repo_write_allowed(args.draft_card)
        hashes = verify_adapter(args.final_dir)
        print(
            f"adapter sha256 verified against {args.final_dir.name}/manifest.json: "
            f"{len(hashes)} files"
        )
        if args.card_file is not None:
            # A finished card (e.g. a fold model whose numbers do not come from the template's
            # 1260-token baseline run): the same leak check and PENDING gate still apply.
            card = args.card_file.read_text(encoding="utf-8")
            if PLACEHOLDER_RE.search(card):
                raise StagingError("--card-file still contains a {{placeholder}}")
            resolved, pending = [], list(dict.fromkeys(PENDING_RE.findall(card)))
        else:
            values = build_values(
                args.final_dir, args.runs_root, args.model_name, args.repo_id, hashes
            )
            card, resolved, pending = fill_card(TEMPLATE.read_text(encoding="utf-8"), values)
        matcher = gold_matcher(args.labels_root)
        kinds = leak_kinds(card, matcher)
        if kinds:
            raise StagingError(f"card contains gold label value(s) [{'/'.join(sorted(kinds))}]")
        staged = stage_tree(args.final_dir, stage_dir, card, args.merged_dir)
        problems = check_staging(stage_dir, matcher, args.merged_dir is not None)
        if problems:
            raise StagingError("leak/staging check failed: " + "; ".join(problems))
    except (StagingError, OSError, KeyError, ValueError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 4
    print(f"staged {len(staged)} files under {stage_dir}:")
    for f in staged:
        print(
            f"  {f.relative_to(stage_dir).as_posix()}  {f.stat().st_size} bytes  {sha256_file(f)}"
        )
    print(f"leak check: clean ({len(staged)} files, magic bytes, gold scan of text files)")
    print(f"card: {len(resolved)} placeholders resolved, {len(pending)} PENDING")
    for p in pending:
        print(f"  PENDING: {p}")
    if args.upload:
        if pending:
            print("error: --upload refused: the card still has PENDING markers", file=sys.stderr)
            return 7
        hub_upload(stage_dir, args.repo_id, args.merged_repo_id)
        print(f"uploaded PRIVATE repo {args.repo_id}")
        return 0
    if args.card_file is None:
        draft.write_text(card, encoding="utf-8", newline="\n")
        print(f"dry-run: draft card written to {draft}")
    else:
        print(f"dry-run: card taken from {args.card_file.name}; no draft card written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
