"""Write reports/trainset.md: aggregate counts of the fine-tune training set (STEP V2).

Counts only, never a value or a name: row-to-page sources, document status, round trip through the
merge, stage splits, occlusion candidates and the rates they imply. Needs data/, the OCR cache and
the official scorer; ``provenance/locations.jsonl`` is used only as a cross-check when present.

Run: uv run python scripts/trainset_report.py [--out reports/trainset.md]
"""

# ruff: noqa: E501  # markdown prose literals

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from shipdoc import paths
from shipdoc import trainset as ts
from shipdoc.eval import load_scorer
from shipdoc.ocr import doc_pages

ROOT = Path(__file__).resolve().parents[1]
EPOCHS_SAMPLED = 20  # epochs drawn to estimate the realised rates (selection needs no images)


def git_head() -> str:
    """Short SHA of HEAD (provenance of the numbers)."""
    out = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def md_table(header: list[str], rows: list[list[Any]]) -> str:
    """Markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def realised_rates(
    prepared: ts.PreparedSet,
    doc_ids: list[str],
    rate: float,
    seed: int,
    epochs: int,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Page-level occlusion rate and the chosen-field histogram over `epochs` sampled epochs."""
    pages = hits = 0
    fields: Counter[str] = Counter()
    for d in prepared.included(doc_ids):
        n = prepared.plans[d].n_pages
        for e in range(epochs):
            ch = ts.select_occlusions(d, n, e, prepared.candidates.get(d, []), rate, seed, weights)
            pages += n
            hits += len(ch)
            fields.update(c.field for c in ch)
    return {"pages": pages, "hits": hits, "fields": fields}


def build_report(
    golds: dict[str, dict[str, Any]],
    assignments: dict[str, tuple[list[str], list[int | None]]],
    candidates: dict[str, list[ts.OcclusionCandidate]],
    reasons: Counter[str],
    cross_check: tuple[int, int] | None,
    scorer: Any,
    folds: dict[str, Any],
    head: str,
    rate: float = 0.15,
    scan_prob: float = 0.5,
) -> str:
    """The report text. `cross_check` = (docs compared, docs differing) vs locations.jsonl."""
    ids = sorted(golds)
    default = ts.prepare(golds, assignments, candidates=candidates)
    strict = ts.prepare(golds, assignments, monotone_boundary=False)
    exclude = ts.prepare(golds, assignments, ambiguous_policy="exclude")
    summary = ts.summarize(default)
    out = [
        "# Fine-tune training set (STEP V2): row-to-page plan, round trip, augmentation rates",
        "",
        f"**Every number here is UNVERIFIED** (computed on this machine at commit `{head}`, nothing "
        "was trained). Counts only: no page content, no values, no names. Reproduce with "
        "`uv run python scripts/trainset_report.py` (needs `data/`, the OCR cache and "
        "`assignment/score.py`); tests: `tests/test_trainset.py`.",
        "",
        "## Row -> page plan (`trainset.resolve_row_pages`, default policy)",
        "",
        ts.render_report_md(summary, {}),
        "Sources: `line` / `page_fuzzy` keep the locator's page; `single_page_default` = unassigned row of "
        "a one-page doc (page 0); `interpolated` = both assigned neighbours on the same page; "
        "`monotone_boundary` = leading/trailing row whose neighbour is on the first/last page; "
        "`ambiguous` = page unknown. Documents with an `ambiguous` row are `header_only`: their "
        "`line_items` tokens are masked and the unplaceable rows are left out of the target text; "
        "with `ambiguous_policy: exclude` they would be dropped instead (below).",
        "",
    ]
    rows = []
    for name, p in (("boundary rule on (default, = V1 counts)", default),
                    ("boundary rule off (strict neighbours only)", strict)):  # fmt: skip
        s = ts.summarize(p)["splits"]
        rows.append([name] + [f"{s[sp]['row_sources'].get('ambiguous', 0)} rows / "
                              f"{s[sp]['status'].get('header_only', 0)} docs" for sp in sorted(s)])  # fmt: skip
    events = Counter(e["event"] for e in ts.audit_events(default))
    out += [
        f"Audit log (`python -m shipdoc.trainset` writes `<SHIPDOC_RUNS_DIR>/trainset/audit.jsonl`, one "
        f"line per decision: doc id, row, page, event; no values): {dict(sorted(events.items()))} "
        f"= {sum(events.values())} lines.",
        "",
        "Ambiguous rows / header-only docs by rule (V1 estimated 12 train rows in 5 docs and 1 dev row "
        "in 1 doc with neighbour interpolation):",
        "",
        md_table(["rule", *sorted(summary["splits"])], rows),
        "",
        "Excluded documents under `ambiguous_policy: exclude`: "
        + ", ".join(
            f"{sp} {s['status'].get('excluded', 0)}"
            for sp, s in sorted(ts.summarize(exclude)["splits"].items())
        )
        + ".",  # fmt: skip
    ]
    if cross_check is not None:
        out += [
            "",
            f"Cross-check: the plan's assignments come from `locate.assign_rows` on the OCR cache; "
            f"compared with `provenance/locations.jsonl` row records, {cross_check[1]} of "
            f"{cross_check[0]} documents with rows differ.",
        ]
    # per supplier group exclusion bias (open question 2)
    meta = {}
    for sp in ("train", "dev"):
        for m in json.loads((ROOT / "meta" / f"{sp}.json").read_text(encoding="utf-8")):
            meta[m["doc_id"]] = m
    per: Counter[str] = Counter()
    for d, plan in default.plans.items():
        if plan.status != "ok":
            per[meta.get(d, {}).get("supplier_group", "?")] += 1
    groups = dict(sorted(per.items()))
    out += [
        "",
        f"Header-only documents by supplier group (train+dev): {groups}. "
        "(Exclusion bias, V1d question 2: these are scanned multipage layouts; the header-only mode "
        "keeps their header supervision.)",
        "",
        "## Round trip through `merge_pages` + `normalize_doc` + the official scorer",
        "",
    ]
    rt: dict[str, Counter[str]] = {}
    for d in ids:
        plan = default.plans[d]
        sp = d.split("_", 1)[0]
        c = rt.setdefault(sp, Counter())
        if plan.status != "ok":
            c["header_only (skipped)"] += 1
            continue
        r = ts.roundtrip_doc(golds[d], plan, scorer)
        c["round-trip exact"] += r["exact"]
        c["OVERALL == 1.0"] += r["overall"] == 1.0
        c["tested"] += 1
    out.append(md_table(["split", "tested docs", "round-trip exact", "OVERALL == 1.0",
                         "header-only (skipped)"],
                        [[sp, c["tested"], c["round-trip exact"], c["OVERALL == 1.0"],
                          c["header_only (skipped)"]] for sp, c in sorted(rt.items())]))  # fmt: skip
    out += [
        "",
        "A document passes when its per-page targets, merged with the defaults of `merge_pages` and "
        "normalised, equal the gold header and rows AND score OVERALL 1.0 under `assignment/score.py` "
        "(document level).",
        "",
        "## Stage splits (`splits/folds.json` is the only source)",
        "",
    ]
    rows = []
    for stage in ts.STAGES:
        sp = ts.stage_split(stage, folds)
        specs = default.page_specs(sp.train_ids)
        held = default.page_specs(sp.heldout_ids)
        rows.append([stage, len(sp.train_ids), len(specs), len(sp.heldout_ids), len(held),
                     sp.manifest_hash()])  # fmt: skip
    out.append(md_table(["stage", "train docs", "train pages", "held-out docs", "held-out pages",
                         "manifest hash"], rows))  # fmt: skip
    out += ["", "`assert_no_leakage` passed for every stage (no train/held-out overlap, no test docs, "
            "no dev docs in `final`). The `smoke` stage trains on the first 80 pages (doc-id order) of the "
            "`final` training pages.", ""]  # fmt: skip
    # augmentation
    final = ts.stage_split("final", folds).train_ids
    pages = sum(default.plans[d].n_pages for d in default.included(final))
    with_cand = sum(
        any(c.target_page == p for c in candidates.get(d, []))
        for d in default.included(final)
        for p in range(default.plans[d].n_pages)
    )
    png = sum(
        golds[d]["pages"][p].lower().endswith(".png")
        for d in default.included(final)
        for p in range(default.plans[d].n_pages)
    )
    weights = dict(ts.AugmentConfig().field_weights)
    real = realised_rates(default, list(final), rate, ts.SEED, EPOCHS_SAMPLED, weights)
    hist = real["fields"]
    redaction_like = hist["invoice_number"] + hist["invoice_date"]
    out += [
        "## Augmentation (train docs of `final`, seed 42)",
        "",
        f"Occlusion candidates over all {len(ids)} train+dev docs, by outcome per (doc, header field) "
        f"with a non-null value or not: {dict(sorted(reasons.items()))}.",
        "",
        f"- Train pages: {pages}; pages whose target carries at least one provably hidable header "
        f"field: {with_cand} ({100 * with_cand / pages:.1f}%).",
        f"- Stated rate: P(hide one field) = {rate} per page and epoch -> expected "
        f"{100 * rate * with_cand / pages:.1f}% of train pages per epoch; realised over "
        f"{EPOCHS_SAMPLED} sampled epochs: {real['hits']}/{real['pages']} = "
        f"{100 * real['hits'] / real['pages']:.1f}%.",
        "- Natural prevalence for comparison (recon section 3): 35 redacted header cells in 500 "
        "train+dev docs (35 of 671 pages, 5.2%), all invoice_number / invoice_date.",
        f"- Field draw weights {weights} (others 1). Hidden field histogram (all {EPOCHS_SAMPLED} epochs): {dict(sorted(hist.items()))}; "
        f"invoice_number + invoice_date = {redaction_like} of {real['hits']} "
        f"({100 * redaction_like / max(1, real['hits']):.1f}%).",
        f"- Scan degradation: {png} of {pages} train pages are digital (.png); each is degraded with "
        f"p = {scan_prob} per epoch (expected {100 * scan_prob * png / pages:.1f}% of all train "
        f"pages; with the {pages - png} real scans, {100 * (pages - png + scan_prob * png) / pages:.1f}% "
        "of pages look scanned).",
        "",
        "## Not covered / limits",
        "",
        "- Row-field occlusion is not done (header fields only); `edge_crop` is not used (its reach "
        "cannot be bounded by the collateral check).",
        "- A field is hidden only if every printed copy is located (level <= normalized), no "
        "fuzzy-only copy remains, a copy sits on the page that carries the target, and the widest "
        "possible applied box overlaps no other located value by more than 10% (strict: any other "
        "value counts, so values that are printed twice side by side are rarely candidates).",
        "- The interpolated rows rely on the monotone row order observed in 151 of 152 multipage "
        "docs; it is circular until the label-audit viewer checks a sample (V1d).",
        "- Tokenised sample lengths and the 1,260-visual-token assertion need the real processor "
        "and were only exercised with a fake one.",
        "",
        f"locations.jsonl sha256 prefix (cross-check source): "
        f"{_sha_prefix(paths.runs_dir() / 'provenance' / 'locations.jsonl')}.",
        "",
    ]
    return "\n".join(out)


def _sha_prefix(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.is_file() else "absent"


def main(argv: list[str] | None = None) -> int:
    """Compute everything from the real data and write the report."""
    ap = argparse.ArgumentParser(description=(main.__doc__ or "").strip())
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "trainset.md")
    args = ap.parse_args(argv)
    folds = ts.load_folds()
    ids = sorted(d for f in folds["folds"] for d in f["val_doc_ids"])
    golds = ts.load_golds(ids)
    assignments, cands = {}, {}
    reasons: Counter[str] = Counter()
    for d in ids:
        ocr = doc_pages(d)
        assignments[d] = ts.assignments_from_ocr(golds[d], ocr)
        cands[d], r = ts.occlusion_candidates(golds[d], ocr, len(golds[d]["pages"]))
        reasons.update(r)
    loc_path = paths.runs_dir() / "provenance" / "locations.jsonl"
    cross = None
    if loc_path.is_file():
        ref = ts.assignments_from_locations(loc_path)
        rows = [d for d in ids if assignments[d][0]]
        cross = (len(rows), sum(ref.get(d) != assignments[d] for d in rows))
    text = build_report(golds, assignments, cands, reasons, cross, load_scorer(), folds, git_head())
    args.out.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
