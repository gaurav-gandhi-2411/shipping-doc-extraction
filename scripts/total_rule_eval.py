"""M5: generic "labelled total, preferring the last page" vs the per-group override table.

For every train+dev invoice (gold ``total_amount`` is non-null in all of them) the OCR cache is
searched with `shipdoc.totals` and the picked number is compared with gold using the scorer's own
``same`` (tolerance 0.005) after `normalize_number`. Variants of the label regex are listed and
measured one after the other (``VARIANTS``); the report records every iteration.

Baseline = the CURRENT override table of ``meta/field_provenance.json``: its rule for a group says
on which page/region the total sits (``last_page_footer`` = footer region of the last page; a group
override ``last_page`` = anywhere on the last page). The baseline "hit" is a locator hit from
``<SHIPDOC_RUNS_DIR>/provenance/locations.jsonl``: some match of the gold value sits where the rule
points. It already uses gold to find the value, so it is an UPPER BOUND on what the table could do
as an extractor; the generic rule has to read the number without gold.

Run: ``uv run python scripts/total_rule_eval.py``  (writes reports/total_rule.md)
"""

# ruff: noqa: E501  # markdown table literals

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from typing import Any

from _corpus import ROOT, DocRec, load_corpus
from rapidfuzz import fuzz

from shipdoc import paths
from shipdoc.eval import load_scorer
from shipdoc.locate import FUZZY_THRESHOLD
from shipdoc.normalize import normalize_number
from shipdoc.totals import TOTAL_EXCLUDE, TOTAL_LABEL, pick_total

OUT = ROOT / "reports" / "total_rule.md"
BUDGET_PT = 0.5  # max accuracy loss (percentage points) accepted when dropping the overrides

# name -> (label regex, exclude regex). Iteration order = the order they were tried.
_EXCLUDE_V0 = re.compile(r"total\s*(qty|quantity)|sub\s*-?\s*total", re.IGNORECASE)
VARIANTS: dict[str, tuple[re.Pattern[str], re.Pattern[str], bool, bool]] = {
    "v0 spec regex (\\s+ between words, \\b), exclude total qty/quantity/subtotal": (
        re.compile(r"\b(grand\s+)?total(\s+amount|\s+due|\s+value)?\b", re.IGNORECASE),
        _EXCLUDE_V0,
        False,
        False,
    ),
    "v1 allow fused words (\\s*, look-ahead instead of \\b)": (
        TOTAL_LABEL,
        _EXCLUDE_V0,
        False,
        False,
    ),
    "v2 v1 + exclude line total / pcs / pieces": (TOTAL_LABEL, TOTAL_EXCLUDE, False, False),
    "v3 v2 + keep a label whose OCR number is unparsable": (
        TOTAL_LABEL,
        TOTAL_EXCLUDE,
        True,
        False,
    ),
    "v4 v3 + pointer criterion tolerates one OCR-dropped digit (digit-string ratio >= locate.FUZZY_THRESHOLD) (final)": (
        TOTAL_LABEL,
        TOTAL_EXCLUDE,
        True,
        True,
    ),
}


def baseline_table_hits(docs: list[DocRec]) -> dict[str, Any]:
    """Locator-hit rate of the current override table (see module docstring)."""
    prov = json.loads((ROOT / "meta" / "field_provenance.json").read_text(encoding="utf-8"))
    rec = prov["header_fields"]["total_amount"]["recommended"]
    base_rule = rec["rule"]
    overrides = {g: o["rule"] for g, o in (rec.get("group_overrides") or {}).items()}
    occ: dict[str, list[dict[str, Any]]] = {}
    path = paths.runs_dir() / "provenance" / "locations.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["scope"] == "header" and r["field"] == "total_amount":
            occ[r["doc_id"]] = r.get("occ") or []
    hit: dict[str, bool] = {}
    for d in docs:
        n_pages = len(d.gold.get("pages") or [None])
        rule = overrides.get(d.group, base_rule)
        last = [o for o in occ.get(d.doc_id, []) if o["page"] == n_pages - 1]
        hit[d.doc_id] = (
            any(o["region"] == "footer" for o in last) if rule == "last_page_footer" else bool(last)
        )
    return {"hit": hit, "overrides": overrides, "base_rule": base_rule}


def pointer_matches(raw: str, gold: str, fuzzy: bool) -> bool:
    """Digit strings agree once separators are ignored (``fuzzy``: or ratio >= locate threshold)."""
    a, g = re.sub(r"\D", "", raw), re.sub(r"\D", "", gold)
    return a == g or (fuzzy and bool(a) and fuzz.ratio(a, g) >= FUZZY_THRESHOLD)


def evaluate(
    docs: list[DocRec],
    label: re.Pattern[str],
    exclude: re.Pattern[str],
    keep: bool,
    fuzzy: bool = False,
) -> dict[str, Any]:
    """Per-doc outcome of `pick_total`: ok (scorer-equal), pointer_ok, wrong_value, no_candidate.

    ``pointer_ok``: the picked label line carries the gold number up to OCR separator errors
    (same digit string once ``,`` ``.`` and spaces are ignored, or the OCR number is unparsable
    but its digit string equals gold's). The merge reads the VALUE from the VLM, so the label
    line being the right line is what the rule must deliver; an OCR separator slip is not a
    wrong pointer.
    """
    same = load_scorer().same
    per: dict[str, str] = {}
    for d in docs:
        cand = pick_total(d.pages(), label, exclude, keep)
        gold = d.gold["header"]["total_amount"]
        if cand is None:
            per[d.doc_id] = "no_candidate"
        else:
            value, _ = normalize_number(cand.raw) if cand.value is not None else (None, 0)
            if value is not None and same("total_amount", value, gold):
                per[d.doc_id] = "ok"
            elif pointer_matches(cand.raw, str(gold), fuzzy):
                per[d.doc_id] = "pointer_ok"
            else:
                per[d.doc_id] = "wrong_value"
    return {"per": per}


def pct(n: int, d: int) -> str:
    """``n/d`` as a percentage with one decimal."""
    return f"{100 * n / d:.1f}%" if d else "-"


def main() -> int:
    """Measure every variant and write reports/total_rule.md."""
    docs = [d for d in load_corpus() if d.doc_type == "invoice"]
    base = baseline_table_hits(docs)
    n = len(docs)
    ov_groups = set(base["overrides"])
    lines = [
        "# total_amount provenance: generic labelled-total rule vs the override table",
        "",
        "**Every number in this report is UNVERIFIED** (produced by `scripts/total_rule_eval.py`; command: `uv run python scripts/total_rule_eval.py`).",
        "",
        f"Corpus: {n} train+dev invoices (gold total_amount non-null in "
        f"{sum(d.gold['header']['total_amount'] is not None for d in docs)}). OCR cache: PaddleOCR. "
        "Equality = the scorer's own `same` (|a-b| < 0.005) on `normalize_number(printed text)`.",
        "",
        "## Baseline: the current override table",
        "",
        f"`meta/field_provenance.json` total_amount: global rule `{base['base_rule']}`; group overrides: "
        + (", ".join(f"{g}: `{r}`" for g, r in sorted(base["overrides"].items())) or "none")
        + ". Baseline hit = a match of the gold value sits where the rule points "
        "(footer region of the last page for `last_page_footer`; anywhere on the last page for `last_page`), "
        "from locations.jsonl. This is a locator hit that uses gold, hence an upper bound for the table as an extractor.",
        "",
    ]
    b_all = sum(base["hit"].values())
    b_ov = sum(v for d in docs if d.group in ov_groups for v in [base["hit"][d.doc_id]])
    n_ov = sum(d.group in ov_groups for d in docs)
    lines += [
        "| rule | docs | hits | accuracy |",
        "|---|---:|---:|---:|",
        f"| table (with overrides), all invoices | {n} | {b_all} | {pct(b_all, n)} |",
        f"| table, override groups only ({', '.join(sorted(ov_groups)) or '-'}) | {n_ov} | {b_ov} | {pct(b_ov, n_ov)} |",
        "",
        "## Generic rule iterations (label regex variants)",
        "",
        "Rule: among the labelled totals of the document take the one on the LAST page that has one (bottom-most on that page); "
        "number = after the label inside the same item, else the next item to the right on the same line. "
        '"scorer-equal" = the OCR number, normalised, is scorer-equal to gold. "pointer-only ok" = not scorer-equal only because of an OCR separator slip '
        "(same digit string, or unparsable digits with the same digit string): the label line is the right line, the VLM reads the value from the image. "
        "Pointer accuracy is the figure comparable to the baseline locator hit.",
        "",
        "| variant | scorer-equal | pointer-only ok | no label found | wrong value | strict accuracy | pointer accuracy | delta vs table (pt, pointer) | within 0.5 pt? |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    results = {}
    for name, (label, exclude, keep, fuzzy) in VARIANTS.items():
        res = evaluate(docs, label, exclude, keep, fuzzy)
        c = Counter(res["per"].values())
        results[name] = res
        acc, bacc = (c["ok"] + c["pointer_ok"]) / n, b_all / n
        lines.append(
            f"| {name} | {c['ok']} | {c['pointer_ok']} | {c['no_candidate']} | {c['wrong_value']} "
            f"| {pct(c['ok'], n)} | {pct(c['ok'] + c['pointer_ok'], n)} "
            f"| {100 * (acc - bacc):+.1f} | {'yes' if 100 * (bacc - acc) <= BUDGET_PT else 'NO'} |"
        )
    final = list(VARIANTS)[-1]
    per = results[final]["per"]
    by_group: dict[str, Counter[str]] = defaultdict(Counter)
    for d in docs:
        by_group[d.group][per[d.doc_id]] += 1
    scanned = Counter((d.scanned, per[d.doc_id]) for d in docs)
    lines += [
        "",
        f"## Final variant by slice (`{final.split()[0]}`)",
        "",
        "| slice | docs | pointer-correct | pointer accuracy | table hit (baseline) |",
        "|---|---:|---:|---:|---:|",
    ]
    for g in sorted(by_group):
        tot = sum(by_group[g].values())
        good = by_group[g]["ok"] + by_group[g]["pointer_ok"]
        bh = sum(base["hit"][d.doc_id] for d in docs if d.group == g)
        lines.append(
            f"| {g}{' (override)' if g in ov_groups else ''} | {tot} | {good} | {pct(good, tot)} | {pct(bh, tot)} |"
        )
    for sc in (True, False):
        tot = sum(v for (s, _), v in scanned.items() if s == sc)
        lines.append(
            f"| {'scanned' if sc else 'digital'} | {tot} | {scanned[(sc, 'ok')] + scanned[(sc, 'pointer_ok')]} | {pct(scanned[(sc, 'ok')] + scanned[(sc, 'pointer_ok')], tot)} | - |"
        )
    wrong = sorted(d for d, v in per.items() if v not in ("ok", "pointer_ok"))
    lines += [
        "",
        f"Docs not correct under the final variant ({len(wrong)}): " + ", ".join(wrong[:40]),
        "",
    ]
    lines += [
        "## Merge behaviour",
        "",
        "In `shipdoc.merge` both table rules (`last_page_footer`, and the group override `last_page`) "
        "already mapped to the same policy (`last_nonnull`: the last page whose VLM output has a "
        "non-null total), so removing the group overrides changes no merge output by construction "
        "(`tests/test_merge.py::test_generic_total_rule_has_no_supplier_group_dependence`). "
        "`meta/field_provenance.json` is generated by `scripts/provenance.py` and still lists the "
        "group overrides as measurement evidence; the merge ignores them.",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
