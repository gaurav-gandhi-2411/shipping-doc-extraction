"""Where the submitted system fails: wrong row cells by class, 500 docs and dev100 (CPU only).

System: v1.5 = native zero-shot Qwen3.5-4B + rules R1-R3 (production path ``replay_v1_check.
production`` with the HONEST supplier-held-out R3 shapes), scored with the unmodified scorer.

Unit of count. A "cell" is one row field (supplier_part_number, customer_part_number,
purchase_order, quantity) of a gold row that the scorer does not count as right (``same``). A
"row" is a gold row that is not fully right (row F1 counts a row right only if all four fields
are). Every wrong row of an invoice falls in exactly one class:

* the ``row_error_diagnosis`` causes of the rows the scorer leaves UNPAIRED (a gold row and a
  predicted row linked as one table line read wrongly): ``column_shift``, ``spn_misread``,
  ``spn_copies_other_slot`` (decision order in that script's docstring); all four fields of such a
  row are scored as missing + extra by row F1, here the cells counted are the fields that differ;
* the cpn / po null-status mismatch kinds of rows the scorer pairs partially (``slot_rows``):
  ``cpn_false_fill`` (gold empty, a customer part number is written), ``cpn_over_null``,
  ``po_over_null``, ``po_false_fill``, ``mixed``, ``po_in_cpn_slot``;
* ``other_field_in_paired_row``: a partially paired row (its part number matched) whose wrong
  cell is not a null-status mismatch (a wrong quantity, or a wrong cpn / po value).

Header cells are counted separately (not row cells). Waybills have no rows. The dev100 view is the
500-doc diagnosis filtered to the dev docs (the classification is per document, so it is the same
as a dev-only run). Aggregates and counts only: no gold or predicted value, no document id.
All numbers UNVERIFIED until a second path recomputes them.

Run: ``uv run python scripts/failure_classes_native.py --run-dir <native ZS run> --out
reports/failure_classes_native.md --json-out $SHIPDOC_TMP_DIR/report_out/failure_classes_native.json``
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay_v1_check as rv  # noqa: E402  (sibling script, not a package)
import row_error_diagnosis as rd  # noqa: E402

from shipdoc import confidence as cf  # noqa: E402
from shipdoc import eval as ev  # noqa: E402
from shipdoc import meta as meta_mod  # noqa: E402
from shipdoc import paths  # noqa: E402
from shipdoc.extract import ROW_KEYS  # noqa: E402
from shipdoc.postrules import RuleConfig  # noqa: E402
from shipdoc.replay import read_trace  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
N_BOOT = 2000
OTHER = "other_field_in_paired_row"
# Concrete fix per class, ONLY with the artifact that backs it. "evidence" says what exists;
# nothing here claims a gain that no artifact measured.
FIXES: dict[str, dict[str, str]] = {
    "column_shift": {
        "fix": "continuation-page column-header hint: give pages without a header row the column "
        "order read on page 1 (07n; built as a design, NOT run and cancelled for the deadline: "
        "future work)",
        "evidence": "ORACLE ceiling only: +2.80 pts [+1.77, +3.96] if all 252 rows were fixed, "
        "+2.46 pts [+1.52, +3.56] for the 220 continuation-page rows "
        "(reports/hdrhint_estimate_native.md section 1; reports/row_errors_native.md X4). All 220 "
        "sit on header-less continuation pages whose page 1 has a header line (27 docs).",
    },
    "spn_misread": {
        "fix": "glyph-level part-number reading via fine-tuning (LoRA); an OCR-snap merge rule "
        "was tried and REJECTED",
        "evidence": "119 of 152 are glyph confusions (reports/row_errors_native.md X1.3); rule "
        "R4 (snap to the nearest OCR span): 92 fixed, 140 broken, -1.30 pts [-2.04, -0.52] "
        "(same report, X4 candidate rules). Fine-tune, INTERIM fold 0 only (171 docs): 60 "
        "misread pairs against 69 for zero-shot (reports/row_errors_native_fold0_ft.md and "
        "_zs.md, X1.3); the pooled 3-fold result is pending, so no fine-tune gain is claimed.",
    },
    OTHER: {
        "fix": "same remedy as the part-number misreads for the purchase-order values (glyph-level "
        "reading, fine-tune); the quantity differences have no diagnosed cause",
        "evidence": "in rows whose part number matched, 74 of the 80 wrong purchase-order cells "
        "are glyph confusions and 29 quantity values differ (kinds line, "
        "reports/failure_classes_native.md); "
        "no artifact measures a fix for either, so none is claimed.",
    },
    "cpn_false_fill": {
        "fix": "do not fill a customer-part slot the layout lacks: prompt or fine-tune target that "
        "leaves it null (untested)",
        "evidence": "ORACLE ceiling: +0.85 pts [+0.26, +1.59] (reports/row_errors_native.md X4). "
        "67 of the 75 rows sit in one supplier group (inv_g18, fold 1, whose fine-tune arm has "
        "not run); no artifact measures any fix, and the R3 rules do not touch false fills.",
    },
    "spn_copies_other_slot": {
        "fix": "fine-tune target (single document pair; the part number is readable in the OCR "
        "text)",
        "evidence": "ORACLE ceiling +0.14 pts [+0.00, +0.36] (reports/row_errors_native.md X4); "
        "2 documents, mechanism unknown.",
    },
}


def wrong_fields(sc: Any, pred_row: Mapping[str, Any], gold_row: Mapping[str, Any]) -> list[str]:
    """Row fields (of the four) that differ under the scorer's ``same`` (null-aware)."""
    return [f for f in ROW_KEYS if not sc.same(f, pred_row.get(f), gold_row.get(f))]


def cell_kinds(
    sc: Any, pred_row: Mapping[str, Any], gold_row: Mapping[str, Any], fields: Sequence[str]
) -> list[str]:
    """Category (never a value) of each wrong cell: ``<field>:<kind>``.

    kind: ``pred_null`` (over-null), ``gold_null_filled`` (false fill), ``qty_differs`` (quantity
    with two values), else the ``row_error_diagnosis.edit_profile`` of the two strings.
    """
    out = []
    for f in fields:
        p, g = pred_row.get(f), gold_row.get(f)
        if rd.dg.empty(p):
            kind = "pred_null"
        elif rd.dg.empty(g):
            kind = "gold_null_filled"
        elif f == "quantity":
            kind = "qty_differs"
        else:
            kind = rd.edit_profile(p, g)
        out.append(f"{f}:{kind}")
    return out


def cell_classes(res: Mapping[str, Any], sc: Any) -> list[dict[str, Any]]:
    """One record per wrong row: doc, class, wrong field names, scanned flag (no values)."""
    pred, gold = res["_pred"], res["_gold"]
    scan = res["slot_docs_scan"]
    group = res["slot_docs_group"]
    recs: list[dict[str, Any]] = []
    for u in res["units"]:
        g = gold[u.doc]
        gr = list(g.get("line_items") or [])
        pr = [x for x in (pred[u.doc].get("line_items") or []) if isinstance(x, dict)]
        # a gold-only or predicted-only row (no pair): every cell is missing / extra
        fields = wrong_fields(sc, pr[u.pi], gr[u.gi]) if u.side == "pair" else list(ROW_KEYS)
        kinds = cell_kinds(sc, pr[u.pi], gr[u.gi], fields) if u.side == "pair" else []
        recs.append(
            {
                "doc": u.doc,
                "cls": u.cause,
                "fields": fields,
                "kinds": kinds,
                "group": group[u.doc],
                "scanned": bool(u.scanned),
            }
        )
    slot = {(s["doc"], s["pi"], s["gi"]): s["kind"] for s in res["slots"]}
    for d, g in gold.items():
        p = pred.get(d)
        if g["doc_type"] != "invoice" or not isinstance(p, dict) or p.get("doc_type") != "invoice":
            continue
        gr = list(g.get("line_items") or [])
        pr = [x for x in (p.get("line_items") or []) if isinstance(x, dict)]
        _, partial, _, _ = ev._pair_rows(sc, pr, gr)
        for pi, gi in partial:
            fields = wrong_fields(sc, pr[pi], gr[gi])
            if not fields:
                continue
            recs.append(
                {
                    "doc": d,
                    "cls": slot.get((d, pi, gi), OTHER),
                    "fields": fields,
                    "kinds": cell_kinds(sc, pr[pi], gr[gi], fields),
                    "group": group[d],
                    "scanned": bool(scan[d]),
                }
            )
    return recs


def tally(
    recs: Sequence[Mapping[str, Any]], keep: Any = lambda d: True
) -> dict[str, dict[str, Any]]:
    """Per class: wrong rows, cells, docs, scanned rows, cells by field (docs passing `keep`)."""
    out: dict[str, dict[str, Any]] = {}
    docs: dict[str, set[str]] = defaultdict(set)
    for r in recs:
        if not keep(r["doc"]):
            continue
        o = out.setdefault(
            r["cls"],
            {
                "rows": 0,
                "cells": 0,
                "scanned_rows": 0,
                "docs": 0,
                "by_field": Counter(),
                "kinds": Counter(),
                "groups": Counter(),
            },
        )
        o["rows"] += 1
        o["cells"] += len(r["fields"])
        o["scanned_rows"] += bool(r["scanned"])
        o["by_field"].update(r["fields"])
        o["kinds"].update(r.get("kinds", []))
        o["groups"][r.get("group", "?")] += 1
        docs[r["cls"]].add(r["doc"])
    for k, o in out.items():
        o["docs"] = len(docs[k])
        o["by_field"] = dict(o["by_field"])
        o["kinds"] = dict(o["kinds"])
        o["groups"] = dict(o["groups"].most_common(6))
    return out


def top_by_cells(tab: Mapping[str, Mapping[str, Any]], n: int = 3) -> list[str]:
    """Class names by wrong cells, descending (ties broken by rows, then name)."""
    return sorted(tab, key=lambda k: (-tab[k]["cells"], -tab[k]["rows"], k))[:n]


def scorer_cells(
    sc: Any, pred: Mapping[str, Any], gold: Mapping[str, Any], keep: Any
) -> dict[str, int]:
    """The official scorer's own row counts of the docs passing `keep` (no classifier involved).

    ``wrong_rows`` = gold rows - fully right rows (``rows_full``); ``cells`` = sum over the four
    row fields of gold rows - ``row_fields[f]``, i.e. every cell of a row the scorer leaves
    unpaired counts as wrong (a stricter cell definition than the differing values of the table).
    """
    out = {"gold_rows": 0, "wrong_rows": 0, "cells": 0}
    for d, g in gold.items():
        if not keep(d):
            continue
        s = sc.score_doc(pred.get(d), g)
        out["gold_rows"] += s["rows_gold"]
        out["wrong_rows"] += s["rows_gold"] - s["rows_full"]
        out["cells"] += sum(s["rows_gold"] - v for v in s["row_fields"].values())
    return out


def header_wrong(pred: Mapping[str, Any], gold: Mapping[str, Any], keep: Any) -> int:
    """Wrong header cells (scorer ``same`` via ``confidence.label_doc``) of docs passing keep."""
    n = 0
    for d, g in gold.items():
        if keep(d):
            lab = cf.label_doc(pred[d], g)
            n += sum(not ok for (scope, _, _), (ok, _) in lab.items() if scope == "header")
    return n


def build_final(run_dir: Path, work: Path) -> tuple[Path, Path, Path]:
    """Write the post-rule run (predictions + trace), a 500-doc gold folder and meta to `work`."""
    traces = read_trace(run_dir / "trace.jsonl")
    pred_all = json.loads((run_dir / "predictions.json").read_text(encoding="utf-8"))
    gold: dict[str, Any] = {}
    for split in ("train", "dev"):
        gold |= ev.load_gold(paths.data_dir() / split / "labels")
    gold = {d: g for d, g in gold.items() if d in pred_all}
    if len(gold) != 500:
        raise SystemExit(f"expected 500 scored docs, found {len(gold)}")
    folds = json.loads((ROOT / "splits" / "folds.json").read_text("utf-8"))["doc_fold"]
    doc_fold = {str(d): int(f) for d, f in folds.items()}
    groups = json.loads((ROOT / "meta" / "supplier_groups.json").read_text(encoding="utf-8"))
    labels = meta_mod.load_labels("train") + meta_mod.load_labels("dev")
    honest = rv.fold_shape_fn(labels, doc_fold, groups)
    traces = [t for t in traces if t["doc_id"] in gold]
    off, _ = rv.production(traces, RuleConfig.all_off(), None, None)
    if off != {d: pred_all[d] for d in gold}:
        raise SystemExit("rules OFF does not reproduce predictions.json")
    final, _ = rv.production(traces, RuleConfig(), honest, None)
    run = work / "run"
    run.mkdir(parents=True, exist_ok=True)
    (run / "predictions.json").write_text(json.dumps(final), encoding="utf-8")
    with (run / "trace.jsonl").open("w", encoding="utf-8") as fh:
        for t in traces:
            fh.write(json.dumps(t) + "\n")
    gdir = work / "gold500"
    gdir.mkdir(exist_ok=True)
    for split in ("train", "dev"):
        for f in (paths.data_dir() / split / "labels").glob("*.json"):
            shutil.copyfile(f, gdir / f.name)
    meta = ev.load_json(ROOT / "meta" / "train.json") + ev.load_json(ROOT / "meta" / "dev.json")
    mpath = work / "meta500.json"
    mpath.write_text(json.dumps(meta), encoding="utf-8")
    return run, gdir, mpath


def compute(run_dir: Path, work: Path, ocr_cache: Path | None) -> dict[str, Any]:
    """All numbers of the report (aggregates only)."""
    run, gdir, mpath = build_final(run_dir, work)
    res = rd.analyse(run, gdir, mpath, ocr_cache, N_BOOT, with_oracle=False)
    sc = ev.load_scorer()
    recs = cell_classes(res, sc)
    pred, gold = res["_pred"], res["_gold"]
    is_dev = lambda d: d.startswith("dev_")  # noqa: E731
    everyone = lambda d: True  # noqa: E731
    all_tab, dev_tab = tally(recs, everyone), tally(recs, is_dev)
    return {
        "n_docs": res["n_docs"],
        "gold_rows": res["gold_rows"],
        "gold_rows_dev": sum(len(g.get("line_items") or []) for d, g in gold.items() if is_dev(d)),
        "base_overall": res["base"]["OVERALL"],
        "all500": all_tab,
        "dev100": dev_tab,
        "top3_all500": top_by_cells(all_tab),
        "top3_dev100": top_by_cells(dev_tab),
        "wrong_rows_all500": sum(v["rows"] for v in all_tab.values()),
        "wrong_cells_all500": sum(v["cells"] for v in all_tab.values()),
        "wrong_rows_dev100": sum(v["rows"] for v in dev_tab.values()),
        "wrong_cells_dev100": sum(v["cells"] for v in dev_tab.values()),
        "header_wrong_all500": header_wrong(pred, gold, everyone),
        "header_wrong_dev100": header_wrong(pred, gold, is_dev),
        "scorer_all500": scorer_cells(sc, pred, gold, everyone),
        "scorer_dev100": scorer_cells(sc, pred, gold, is_dev),
    }


def _cells(tab: Mapping[str, Mapping[str, Any]], cls: str, key: str) -> str:
    return str(tab[cls][key]) if cls in tab else "0"


def render(res: Mapping[str, Any], cmd: str, state: str) -> str:
    """Markdown report (aggregates only)."""
    a, d = res["all500"], res["dev100"]
    order = sorted(set(a) | set(d), key=lambda k: (-a.get(k, {"cells": 0})["cells"], k))
    L = [
        "# Where the submitted system fails (native ZS + rules, v1.5): wrong row cells by class",
        "",
        f"**Provenance.** Command `{cmd}` (repo state `{state}`). System: native zero-shot + "
        "rules R1-R3, honest supplier-held-out R3 shapes, unmodified scorer; classifier: "
        "`scripts/row_error_diagnosis.py` (`analyse`, no oracle) plus the cell counts of "
        "`scripts/failure_classes_native.py`. **All numbers UNVERIFIED** here; the report "
        "registry records the second path. Aggregates and counts only; dev100 = the 500-doc "
        "diagnosis filtered to the dev docs.",
        "",
        "Unit: a wrong ROW is a gold row that is not fully right; a wrong CELL is a row field "
        "(supplier part number, customer part number, purchase order, quantity) the scorer does "
        "not count right. Classes are mutually exclusive (definitions in the script docstring).",
        "",
        f"Totals: {res['wrong_rows_all500']} wrong rows / {res['wrong_cells_all500']} wrong cells "
        f"of {res['gold_rows']} gold rows on the {res['n_docs']} docs; "
        f"{res['wrong_rows_dev100']} / {res['wrong_cells_dev100']} on dev100 "
        f"({res['gold_rows_dev']} gold rows). Wrong header cells (not row cells): "
        f"{res['header_wrong_all500']} on {res['n_docs']} docs, {res['header_wrong_dev100']} on "
        "dev100. Waybills have no rows.",
        "",
        "Cross-check with the official scorer, no classifier: wrong rows "
        f"{res['scorer_all500']['wrong_rows']} on {res['n_docs']} docs and "
        f"{res['scorer_dev100']['wrong_rows']} on dev100 (gold rows - `rows_full`); in the "
        "scorer's own sense every cell of an unpaired row counts, which gives "
        f"{res['scorer_all500']['cells']} / {res['scorer_dev100']['cells']} wrong cells (the "
        "tables here count only fields whose value differs, so a shifted row's empty-empty "
        "slots are not counted).",
        "",
        "| class | wrong rows, 500 | wrong cells, 500 | docs, 500 | wrong rows, dev100 | "
        "wrong cells, dev100 | docs, dev100 |",
        "|---|---|---|---|---|---|---|",
    ]
    for k in order:
        L.append(
            f"| {k} | {_cells(a, k, 'rows')} | {_cells(a, k, 'cells')} | {_cells(a, k, 'docs')} | "
            f"{_cells(d, k, 'rows')} | {_cells(d, k, 'cells')} | {_cells(d, k, 'docs')} |"
        )
    t_all, t_dev = res["top3_all500"], res["top3_dev100"]
    L += [
        "",
        f"Top 3 by wrong cells, 500 docs: {', '.join(t_all)}. Top 3 on dev100: "
        f"{', '.join(t_dev)}. "
        + (
            "The two lists agree (order may differ)."
            if set(t_all) == set(t_dev)
            else "The two lists DIFFER; the report states which one it uses (the 500-doc "
            "list: dev100 has too few docs per class)."
        ),
        "",
        "## Top 3 (500 docs) with the concrete fix and the evidence that exists",
        "",
    ]
    # classes outside the top 3 that carry a diagnosed cause and a documented fix are listed too
    shown = [*t_all, *[k for k in FIXES if k not in t_all and k in a]]
    for rank, k in enumerate(shown, 1):
        fx = FIXES.get(k)
        outside = "" if rank <= len(t_all) else " (outside the top 3)"
        L += [
            f"### {rank}. {k}{outside}: {_cells(a, k, 'rows')} rows, "
            f"{_cells(a, k, 'cells')} cells, {_cells(a, k, 'docs')} docs "
            f"(dev100: {_cells(d, k, 'rows')} rows, {_cells(d, k, 'cells')} cells)",
            "",
            f"- cells by field (500 docs): {json.dumps(a[k]['by_field'], sort_keys=True)}",
            f"- scanned rows (500 docs): {a[k]['scanned_rows']} of {a[k]['rows']}",
            f"- wrong-cell kinds (500 docs): {json.dumps(a[k]['kinds'], sort_keys=True)}",
            f"- rows by supplier group, top 6 (500 docs): {json.dumps(a[k]['groups'])}",
            f"- fix: {fx['fix'] if fx else 'none grounded in an artifact'}",
            f"- evidence: {fx['evidence'] if fx else '-'}",
            "",
        ]
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", type=Path, required=True, help="the native ZS run folder")
    ap.add_argument("--work-dir", type=Path, default=Path("D:/shipdoc/tmp/failure_work"))
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "failure_classes_native.md")
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--ocr-cache", type=Path, default=None)
    a = ap.parse_args(argv)
    a.work_dir.mkdir(parents=True, exist_ok=True)
    res = compute(a.run_dir, a.work_dir, a.ocr_cache)
    cmd = "uv run python scripts/failure_classes_native.py " + " ".join(
        argv if argv is not None else sys.argv[1:]
    )
    a.out.write_text(render(res, cmd, rd.git_state()), encoding="utf-8", newline="\n")
    if a.json_out:
        a.json_out.parent.mkdir(parents=True, exist_ok=True)
        a.json_out.write_text(json.dumps(res, indent=1, sort_keys=True), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
