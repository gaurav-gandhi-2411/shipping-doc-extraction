"""Diagnose the row false fills of one OOF fold (CPU only, no model; aggregates and categories).

A false fill is a cell whose gold is empty while the prediction is not, on the rows the scorer
pairs (``shipdoc.oof.over_null_counts``). For the fine-tuned (FT) arm's cells this reports, without
printing any document value:

1. field, doc type, scanned / digital, supplier group of every cell (and how many distinct docs
   and groups carry them);
2. the zero-shot (ZS) arm's state on the same gold row and field: null / value / no paired row;
3. whether the production rules R1-R3 (``scripts/replay_v1_check.py::production``, honest R3
   shapes of the other folds) change the cell for the FT arm;
4. the false-fill and over-null counts (spec section 11 C2 / C3 definitions) for ZS-raw, FT-raw,
   ZS+rules, FT+rules;
5. a category per cell, decided on strings that never leave this script: (a) the value equals the
   same gold row's value in ANOTHER slot, (b) not (a) and the value is found in the page OCR
   (``shipdoc.locate``): printed on the page, gold leaves the slot null, (c) other.

    uv run python scripts/false_fill_diag.py --oof-run-dir <oof_fold0 folder> \
        --zs-run-dir <02 run folder> [--fold 0] [--out reports/fold0_false_fills.md]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

from shipdoc import eval as ev
from shipdoc import locate
from shipdoc import meta as meta_mod
from shipdoc.oof import over_null_counts
from shipdoc.runmeta import read_manifest

ROOT = Path(__file__).resolve().parents[1]
Cell = tuple[str, int, str]  # (doc_id, gold row index, row field)


def _script(name: str) -> ModuleType:
    """Import ``scripts/<name>.py`` (scripts are not a package); reuse an already loaded copy."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parent / f"{name}.py"
    )
    if spec is None or spec.loader is None:  # pragma: no cover - the file is next to this one
        raise ImportError(name)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses resolve annotations via sys.modules
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------------------------
# Pure pieces (tested on synthetic documents)
# --------------------------------------------------------------------------------------------


def _rows(doc: Any) -> list[dict[str, Any]]:
    """The predicted rows the scorer sees (dict entries of ``line_items``)."""
    if not isinstance(doc, dict):
        return []
    return [x for x in (doc.get("line_items") or []) if isinstance(x, dict)]


def row_pairs(sc: Any, pred_doc: Any, gold_doc: Mapping[str, Any]) -> dict[int, int]:
    """gold row index -> predicted row index for the rows the scorer pairs (full and partial)."""
    full, partial, _p, _g = ev._pair_rows(sc, _rows(pred_doc), gold_doc["line_items"])
    return {gi: pi for pi, gi in full + partial}


def false_fill_cells(sc: Any, pred: Mapping[str, Any], gold: Mapping[str, Any]) -> list[Cell]:
    """Row cells on scorer-paired rows where the gold is empty and the prediction is not."""
    out: list[Cell] = []
    for d in sorted(gold):
        g = gold[d]
        pr = _rows(pred.get(d))
        for gi, pi in sorted(row_pairs(sc, pred.get(d), g).items()):
            for f in sc.ROW:
                if sc._empty(g["line_items"][gi].get(f)) and not sc._empty(pr[pi].get(f)):
                    out.append((d, gi, f))
    return out


def cell_state(sc: Any, pred: Mapping[str, Any], gold: Mapping[str, Any], cell: Cell) -> str:
    """State of one gold row / field in a prediction: ``unpaired``, ``null`` or ``value``."""
    d, gi, f = cell
    pi = row_pairs(sc, pred.get(d), gold[d]).get(gi)
    if pi is None:
        return "unpaired"
    return "null" if sc._empty(_rows(pred.get(d))[pi].get(f)) else "value"


def _alnum(v: Any) -> str:
    return re.sub(r"[^0-9A-Z]", "", str(v).upper())


def categorise(
    sc: Any,
    field: str,
    pred_row: Mapping[str, Any],
    gold_row: Mapping[str, Any],
    index: locate.DocIndex | None,
) -> dict[str, Any]:
    """Booleans and a category (a / b / c) for one false-fill value; no value leaves the call.

    a: equals the same gold row's value in another slot (scorer ``same`` or alphanumeric equal);
    b: not a, but found in the page OCR at some level (exact / normalized / fuzzy);
    c: neither (not found in OCR, or no OCR index given).
    """
    v = pred_row.get(field)
    others = [
        k
        for k in sc.ROW
        if k != field
        and not sc._empty(gold_row.get(k))
        and (sc.same(k, v, gold_row.get(k)) or _alnum(v) == _alnum(gold_row.get(k)))
    ]
    dup = [k for k in sc.ROW if k != field and not sc._empty(pred_row.get(k))
           and _alnum(pred_row.get(k)) == _alnum(v)]  # fmt: skip
    m = locate.locate(v, field, index) if index is not None else None
    cat = "a" if others else ("b" if m is not None else "c")
    return {
        "category": cat,
        "equals_gold_other_slot": others,
        "duplicated_in_pred_slot": dup,
        "in_ocr": m is not None,
        "ocr_level": m.level if m is not None else None,
    }


def count_table(
    sc_arms: Mapping[str, Mapping[str, Any]], gold: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """``over_null_counts`` per named prediction set (the C2 / C3 inputs)."""
    return {name: over_null_counts(pred, gold) for name, pred in sc_arms.items()}


def cell_set_change(before: Sequence[Cell], after: Sequence[Cell]) -> dict[str, int]:
    """How the false-fill cell set moved: kept, cleared, new."""
    b, a = set(before), set(after)
    return {"before": len(b), "kept": len(b & a), "cleared": len(b - a), "new": len(a - b)}


# --------------------------------------------------------------------------------------------
# Analysis on real run folders
# --------------------------------------------------------------------------------------------


def analyse(
    zs: Any,
    ft: Any,
    gold: Mapping[str, Any],
    meta: Mapping[str, Mapping[str, Any]],
    groups: Mapping[str, str],
    shapes: Any,
    ocr_cache: Path | None,
    g4f: ModuleType,
) -> dict[str, Any]:
    """Every number of the report (aggregates and categories only)."""
    sc = ev.load_scorer()
    zs_raw, zs_fin, zs_sum = g4f.process_arm(zs, shapes, ocr_cache)
    ft_raw, ft_fin, ft_sum = g4f.process_arm(ft, shapes, ocr_cache)
    counts = count_table(
        {"ZS-raw": zs_raw, "FT-raw": ft_raw, "ZS+rules": zs_fin, "FT+rules": ft_fin}, gold
    )
    cells = false_fill_cells(sc, ft_raw, gold)
    docs = sorted({c[0] for c in cells})
    idx = g4f.build_indexes(docs, ocr_cache) if docs else {}
    rows = []
    for d, gi, f in cells:
        pi = row_pairs(sc, ft_raw[d], gold[d])[gi]
        full, _partial, _p, _g = ev._pair_rows(sc, _rows(ft_raw[d]), gold[d]["line_items"])
        rows.append(
            {
                "doc": d,
                "field": f,
                "doc_type": gold[d]["doc_type"],
                "scanned": bool(meta[d]["scanned"]),
                "group": groups[d],
                "pairing": "full" if (pi, gi) in full else "partial",
                "zs_raw_state": cell_state(sc, zs_raw, gold, (d, gi, f)),
                "zs_rules_state": cell_state(sc, zs_fin, gold, (d, gi, f)),
                "ft_rules_state": cell_state(sc, ft_fin, gold, (d, gi, f)),
                **categorise(sc, f, _rows(ft_raw[d])[pi], gold[d]["line_items"][gi], idx.get(d)),
            }
        )
    change = {
        "ZS": cell_set_change(
            false_fill_cells(sc, zs_raw, gold), false_fill_cells(sc, zs_fin, gold)
        ),
        "FT": cell_set_change(cells, false_fill_cells(sc, ft_fin, gold)),
    }
    return {
        "n_docs": len(gold),
        "cells": rows,
        "counts": counts,
        "rules_cell_change": change,
        "rules_summary": {"ZS": zs_sum, "FT": ft_sum},
    }


def _tbl(head: Sequence[str], body: Sequence[Sequence[Any]]) -> list[str]:
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(str(x) for x in r) + " |" for r in body]
    return lines


#: Read from assignment/score.py (``score_doc`` / ``aggregate`` / ``match_rows`` / ``same``) and
#: ``shipdoc.oof.over_null_counts``; static because it describes the code, not the run.
SCORER_NOTE = (
    "## 5. What the scorer penalises, and which metric C2 uses",
    "",
    "- `false_fill_rate` (score.py `score_doc` / `aggregate`) counts only gold-null HEADER fields "
    "that the prediction fills, divided by the gold-null header fields. Row cells never enter it, "
    "so it is 0.00 for both arms in `oof_compare.md` while row false fills are non-zero.",
    "- Row false fills are still penalised by the scorer, through other metrics: `same` returns "
    "False for (gold empty, prediction non-empty), so `row_ok` fails and the row cannot be a "
    "fully correct row in pass 1 of `match_rows`; it can only pair in pass 2 (by "
    "supplier_part_number, the `partial` pairing above), where it loses that field's credit "
    "in `line_item_field_accuracy`. The row is then not counted in row F1's true positives and "
    "the document is not `exact` (documents_fully_correct).",
    "- The pre-registered C2 (spec section 11 item 1) is "
    "`shipdoc.oof.over_null_counts(...)['false_fill_total']` = header false fills + row false "
    "fills on scorer-paired rows, so it includes these cells; it is not `false_fill_rate`.",
    "",
)


def render(res: Mapping[str, Any], fold: int, prov: Mapping[str, str]) -> str:
    """Markdown report from `analyse` output (no document values)."""
    cells = res["cells"]
    c = res["counts"]
    L = [
        f"# Fold {fold}: row false fills of the fine-tuned arm (aggregates and categories only)",
        "",
    ]
    L += [f"- {k}: {v}" for k, v in prov.items()]
    L += [f"- documents: {res['n_docs']}", ""]
    L += ["## 1. The cells (FT arm, raw)", ""]
    L += _tbl(
        ["#", "doc", "field", "doc type", "scanned", "group", "pairing"],
        [
            [i + 1, x["doc"], x["field"], x["doc_type"], "yes" if x["scanned"] else "no",
             x["group"], x["pairing"]]
            for i, x in enumerate(cells)
        ],
    )  # fmt: skip
    L += [
        "",
        f"{len(cells)} cells; {len({x['doc'] for x in cells})} distinct docs; "
        f"{len({x['group'] for x in cells})} distinct supplier groups "
        f"({dict(Counter(x['group'] for x in cells))}); by field "
        f"{dict(Counter(x['field'] for x in cells))}; scanned "
        f"{sum(x['scanned'] for x in cells)} / digital {sum(not x['scanned'] for x in cells)}.",
        "",
        "## 2. Zero-shot arm on the same gold row and field, and the effect of the rules",
        "",
    ]
    L += _tbl(
        ["#", "ZS raw", "ZS+rules", "FT+rules"],
        [[i + 1, x["zs_raw_state"], x["zs_rules_state"], x["ft_rules_state"]]
         for i, x in enumerate(cells)],
    )  # fmt: skip
    L += ["", "State: null = row paired and the cell empty; value = paired and non-empty;"]
    L += ["unpaired = the scorer pairs no prediction row with that gold row.", ""]
    L += ["## 3. Counts by the spec section 11 definitions (all docs of the fold)", ""]
    arms = ["ZS-raw", "FT-raw", "ZS+rules", "FT+rules"]
    keys = ["header_false_fill", "row_false_fill", "false_fill_total", "header_over_null",
            "row_over_null", "over_null_total", "rows_unmatched_gold"]  # fmt: skip
    L += _tbl(["count", *arms], [[k, *[c[a][k] for a in arms]] for k in keys])
    L += [
        "",
        "False-fill cell set, raw -> after rules (same cell = same doc, gold row, field):",
        "",
    ]
    ch = res["rules_cell_change"]
    L += _tbl(
        ["arm", "before", "kept", "cleared", "new"],
        [[a, ch[a]["before"], ch[a]["kept"], ch[a]["cleared"], ch[a]["new"]] for a in ch],
    )
    L += [
        "",
        f"Rules change the FT arm's false fills: {ch['FT']['cleared']} of {ch['FT']['before']} "
        f"cleared, {ch['FT']['new']} new; FT row false fills {c['FT-raw']['row_false_fill']} -> "
        f"{c['FT+rules']['row_false_fill']}, FT over-nulls {c['FT-raw']['over_null_total']} -> "
        f"{c['FT+rules']['over_null_total']}.",
    ]
    L += ["", "## 4. Category of each FT value (decided on strings inside the script)", ""]
    L += _tbl(
        ["#", "category", "equals gold other slot", "also in pred other slot", "in OCR", "level"],
        [[i + 1, x["category"], ",".join(x["equals_gold_other_slot"]) or "-",
          ",".join(x["duplicated_in_pred_slot"]) or "-", x["in_ocr"], x["ocr_level"] or "-"]
         for i, x in enumerate(cells)],
    )  # fmt: skip
    cc = Counter(x["category"] for x in cells)
    L += [
        "",
        f"a (value equals the gold of another slot of the same row): {cc.get('a', 0)}; "
        f"b (on the page, gold leaves the slot null): {cc.get('b', 0)}; "
        f"c (other): {cc.get('c', 0)}.",
        "",
        *SCORER_NOTE,
    ]
    return "\n".join(L) + "\n"


def run(a: argparse.Namespace) -> int:
    """Validate the two run folders (as ``g4_fold.run`` does), analyse, write the report."""
    g4f = _script("g4_fold")
    rg = g4f.rg
    man = read_manifest(a.oof_run_dir) if a.oof_run_dir.is_dir() else None
    fold, _sec = rg.check_oof_manifest(man, a.fold, False)
    rg.require_complete(a.oof_run_dir, "OOF")
    rg.require_complete(a.zs_run_dir, "zero-shot")
    rg.require_same_resolution({"FT run": a.oof_run_dir, "ZS run": a.zs_run_dir})
    doc_fold = rg.load_doc_fold()
    folds = json.loads((ROOT / "splits" / "folds.json").read_text(encoding="utf-8"))
    ids = rg.fold_docs(folds, fold, doc_fold)
    rg.require_oof_inputs(a.oof_run_dir, ids, True, "OOF run")
    rg.require_oof_inputs(a.zs_run_dir, ids, False, "zero-shot run")
    gold = g4f.load_gold(ids)
    meta = {m["doc_id"]: m for m in g4f.load_meta(ids)}
    for d in (a.oof_run_dir, a.zs_run_dir):
        rg.require_ocr(d, SimpleNamespace(gold=gold), a.ocr_cache)
    ft = g4f.read_arm("FT arm", a.oof_run_dir, ids)
    zs = g4f.read_arm("ZS arm", a.zs_run_dir, ids)
    labels = meta_mod.load_labels("train") + meta_mod.load_labels("dev")
    groups = json.loads((ROOT / "meta" / "supplier_groups.json").read_text(encoding="utf-8"))
    shapes, n_learn = g4f.honest_shapes(fold, labels, doc_fold, groups)
    res = analyse(zs, ft, gold, meta, groups, shapes, a.ocr_cache, g4f)
    prov = {
        "FT run": a.oof_run_dir.name,
        "ZS run (restricted to the fold)": a.zs_run_dir.name,
        "R3 shapes": f"learned from {n_learn} docs outside fold {fold} (supplier-disjoint)",
        "script": "scripts/false_fill_diag.py",
    }
    text = render(res, fold, prov)
    out = a.out if a.out is not None else ROOT / "reports" / f"fold{fold}_false_fills.md"
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")
    if a.json_out is not None:
        a.json_out.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
        print(f"wrote {a.json_out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--oof-run-dir", type=Path, required=True)
    ap.add_argument("--zs-run-dir", type=Path, required=True)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--ocr-cache", type=Path, default=None)
    return run(ap.parse_args(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    sys.exit(main())
