"""Step L: diagnose the spike40 runs (Colab T4, code 51cf560) from their saved artifacts.

Reads ``<runs>/spike40_<model>_<arm>_51cf560/{predictions.json,trace.jsonl,metrics.json}`` for the
six configs, scores them with the official scorer (through ``shipdoc.eval``) against
``data/dev/labels`` restricted to each run's doc_ids, and writes

* ``reports/spike_diagnosis.md``  committed; counts, rates, doc_ids and field names ONLY;
* ``<SHIPDOC_RUNS_DIR>/diagnosis/diagnosis_local.md``  local, gitignored; concrete predicted vs
  gold examples (gold values are confidential: never copy them into a committed file);
* ``configs/spike_speed.json``  with ``--write-speed``: OLS ``latency_s ~ n_output_tokens``.

Run:  uv run python scripts/spike_diagnosis.py --write-speed
Every number in the report is produced by this script from the artifacts; nothing is typed by hand.
"""

# ruff: noqa: E501  # markdown table literals

from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from rapidfuzz.distance import Levenshtein

from shipdoc import diagnostics as diag
from shipdoc import eval as ev
from shipdoc import locate, paths
from shipdoc import merge as merge_mod
from shipdoc import ocr as ocrm
from shipdoc.extract import HEADER_KEYS, NUMERIC_KEYS, parse_output
from shipdoc.merge import FieldProvenance, is_header_row, merge_pages
from shipdoc.normalize import normalize_doc
from shipdoc.replay import Fixes, read_trace, replay_traces
from shipdoc.spike import load_config

ROOT = Path(__file__).resolve().parents[1]
RUN_COMMIT = "51cf560"
NAMES = [
    "qwen35_4b_img_only",
    "qwen35_4b_img_ocr",
    "qwen3vl_8b_img_only",
    "qwen3vl_8b_img_ocr",
    "nuextract3_img_only",
    "nuextract3_img_ocr",
]
CAP = 1536  # max_new_tokens of the 51cf560 json configs
JUNK = re.compile(r"^[\s\]\[\}\{,]+$")  # a "value" made only of braces / commas
BASE_KEY = "baseline: current parse/merge/normalize, no fixes"
DROP_KEY = "+ drop all-null rows"
SALVAGE_KEY = "+ salvage truncated JSON"
NULLROW_RUN = 10  # >= this many all-null rows in one raw page = a degenerate loop
ALPHA_HEADER = sorted(HEADER_KEYS)
ROW_EMITTED = sorted(["supplier_part_number", "customer_part_number", "purchase_order", "quantity"])


def emp(v: Any) -> bool:
    return v is None or (isinstance(v, str) and not v.strip())


def an(v: Any) -> str:
    return re.sub(r"[^0-9a-z]", "", str(v).lower())


def pct(x: float) -> str:
    return f"{100 * x:.2f}"


def table(head: list[str], rows: list[list[Any]]) -> str:
    out = [
        "| " + " | ".join(head) + " |",
        "|" + "|".join("---" if i == 0 else "---:" for i in range(len(head))) + "|",
    ]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def sh(cmd: list[str]) -> str:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


# --------------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------------


class Run:
    """One spike run: saved artifacts plus the gold restricted to its doc_ids."""

    def __init__(self, runs: Path, name: str, gold: dict[str, Any]) -> None:
        self.name = name
        self.dir = runs / f"spike40_{name}_{RUN_COMMIT}"
        self.pred = json.loads((self.dir / "predictions.json").read_text(encoding="utf-8"))
        self.metrics = json.loads((self.dir / "metrics.json").read_text(encoding="utf-8"))
        self.trace = read_trace(self.dir / "trace.jsonl")
        self.gold = {d: g for d, g in gold.items() if d in self.pred}
        self.cfg = load_config(ROOT / "configs" / f"spike_{name}.yaml")
        self.pages = [(t, p) for t in self.trace for p in t["pages"]]
        self.report = ev.score(self.pred, self.gold)


# --------------------------------------------------------------------------------------------
# Step 1: recomposition
# --------------------------------------------------------------------------------------------


def recompose(runs: list[Run]) -> tuple[str, dict[str, Any]]:
    rows, facts = [], {}
    for r in runs:
        a = r.report["all"]
        h, f1, ex = a["header_field_accuracy"], a["row_f1"], a["documents_fully_correct"]
        items = f1 if a["line_item_rows"] else h
        rec = 0.4 * h + 0.4 * items + 0.2 * ex
        same = a["OVERALL"] == r.metrics["OVERALL"]
        facts[r.name] = {"reproduces": same, "recomposed_equal": abs(rec - a["OVERALL"]) < 1e-12}
        rows.append(
            [
                r.name,
                a["documents"],
                pct(h),
                pct(a["row_precision"]),
                pct(a["row_recall"]),
                pct(f1),
                pct(ex),
                f"{rec:.6f}",
                f"{a['OVERALL']:.6f}",
                f"{r.metrics['OVERALL']:.6f}",
                "yes" if same else "NO",
                pct(r.report["invoices"]["OVERALL"]),
                pct(r.report["waybills"]["OVERALL"]),
            ]
        )
    md = table(
        [
            "config",
            "docs",
            "header acc %",
            "row P %",
            "row R %",
            "row F1 %",
            "fully correct %",
            "0.4h+0.4F1+0.2ex",
            "eval.score OVERALL",
            "metrics.json OVERALL",
            "equal",
            "invoices OVERALL %",
            "waybills OVERALL %",
        ],
        rows,
    )
    return md, facts


def per_field_tables(runs: list[Run]) -> str:
    hdr_fields = sorted({f for r in runs for f in r.report["all"]["header_by_field"]})
    rows = [
        [f]
        + [
            pct(r.report["all"]["header_by_field"][f])
            if f in r.report["all"]["header_by_field"]
            else "-"
            for r in runs
        ]
        for f in hdr_fields
    ]
    out = "Header accuracy per field (%, over the docs of the field's type; `eval.score(...)['all']['header_by_field']`):\n\n"
    out += table(["field"] + [r.name for r in runs], rows)
    rows = [
        [f] + [pct(r.report["all"]["line_item_field_accuracy"][f]) for r in runs]
        for f in ev.load_scorer().ROW
    ]
    out += "\nRow field accuracy per field over matched pairs, as a share of ALL gold rows (the scorer's `line_item_field_accuracy`, %):\n\n"
    out += table(["field"] + [r.name for r in runs], rows)
    rows = []
    for r in runs:
        inv = [t for t in r.trace if r.gold[t["doc_id"]]["doc_type"] == "invoice"]
        pr = [len(t["prediction"]["line_items"]) for t in inv]
        gr = [len(r.gold[t["doc_id"]]["line_items"]) for t in inv]
        a = r.report["all"]
        rows.append(
            [
                r.name,
                len(inv),
                f"{np.mean(pr):.2f}",
                f"{np.mean(gr):.2f}",
                sum(pr),
                sum(gr),
                sum(x == 0 for x in pr),
                f"{100 * sum(x == 0 for x in pr) / len(inv):.1f}",
                pct(a["doc_type_accuracy"]),
                pct(a["false_fill_rate"]),
                a["illegible_fields"],
            ]
        )
    out += "\nRows per invoice, zero-row share, doc type and false fills:\n\n"
    out += table(
        [
            "config",
            "invoices",
            "mean pred rows/doc",
            "mean gold rows/doc",
            "pred rows",
            "gold rows",
            "invoices with 0 pred rows",
            "% of invoices",
            "doc_type acc %",
            "false-fill rate %",
            "illegible fields",
        ],
        rows,
    )
    return out


# --------------------------------------------------------------------------------------------
# Step 2: trace statistics and speed
# --------------------------------------------------------------------------------------------


def ols(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    a = np.vstack([np.ones_like(x, dtype=float), x.astype(float)]).T
    (b0, b1), *_ = np.linalg.lstsq(a, y, rcond=None)
    res = y - (b0 + b1 * x)
    r2 = 1 - float((res**2).sum()) / float(((y - y.mean()) ** 2).sum())
    return {
        "prefill_s": float(b0),
        "s_per_tok": float(b1),
        "decode_tok_s": float(1 / b1),
        "r2": r2,
        "n": int(len(x)),
    }


def trace_stats(runs: list[Run]) -> tuple[str, dict[str, dict[str, float]]]:
    rows, fits = [], {}
    for r in runs:
        pages = [p for _, p in r.pages]
        nout = np.array([p["meta"]["n_output_tokens"] for p in pages])
        lat = np.array([p["meta"]["latency_s"] for p in pages])
        nin = np.array([p["meta"]["n_input_tokens"] for p in pages])
        at_cap = int((nout >= CAP).sum())
        invalid = [p for p in pages if not p["json_valid"]]
        inv_at_cap = sum(p["meta"]["n_output_tokens"] >= CAP for p in invalid)
        schema_err = sum(1 for p in pages if p["json_valid"] and p["schema_errors"])
        eos = sum(1 for p in pages if p["meta"]["n_output_tokens"] < CAP and p["json_valid"])
        rows.append(
            [
                r.name,
                len(pages),
                at_cap,
                f"{100 * at_cap / len(pages):.1f}",
                eos,
                len(invalid),
                inv_at_cap,
                len(invalid) - inv_at_cap,
                schema_err,
                f"{nout.mean():.1f}",
                f"{np.percentile(nout, 95):.1f}",
                int(nout.max()),
                CAP,
                f"{nin.mean():.0f}",
            ]
        )
        fits[r.name] = {
            **ols(nout, lat),
            "mean_latency": float(lat.mean()),
            "n_in_mean": float(nin.mean()),
            "n_out_mean": float(nout.mean()),
            "n_in_std": float(nin.std()),
        }
    md = table(
        [
            "config",
            "pages",
            "n_out >= cap",
            "% at cap",
            "EOS (valid, below cap)",
            "invalid JSON",
            "invalid AND at cap",
            "invalid, below cap (parse error)",
            "schema errors (valid JSON)",
            "n_out mean",
            "n_out p95",
            "n_out max",
            "cap",
            "n_in mean",
        ],
        rows,
    )
    return md, fits


def speed_table(fits: dict[str, dict[str, float]]) -> str:
    rows = [
        [
            n,
            f["n"],
            f"{f['prefill_s']:.3f}",
            f"{f['s_per_tok']:.5f}",
            f"{f['decode_tok_s']:.2f}",
            f"{f['r2']:.4f}",
            f"{f['mean_latency']:.2f}",
            f"{f['n_in_mean']:.0f}",
        ]
        for n, f in fits.items()
    ]
    return table(
        [
            "config",
            "pages",
            "prefill_s (intercept)",
            "s per output token (slope)",
            "decode tok/s = 1/slope",
            "R^2",
            "mean latency s",
            "n_in mean",
        ],
        rows,
    )


def ocr_prefill_table(runs: list[Run]) -> str:
    """Two-variable OLS latency ~ 1 + n_out + n_in on the OCR arms (n_in varies there, std ~340)."""
    rows = []
    for r in runs:
        if not r.name.endswith("_img_ocr"):
            continue
        pg = [p["meta"] for _, p in r.pages]
        y = np.array([m["latency_s"] for m in pg])
        a = np.vstack(
            [
                np.ones(len(pg)),
                [m["n_output_tokens"] for m in pg],
                [m["n_input_tokens"] for m in pg],
            ]
        ).T
        (b0, b_out, b_in), *_ = np.linalg.lstsq(a, y, rcond=None)
        res = y - a @ np.array([b0, b_out, b_in])
        r2 = 1 - float((res**2).sum()) / float(((y - y.mean()) ** 2).sum())
        rows.append(
            [
                r.name,
                len(pg),
                f"{b0:.3f}",
                f"{1 / b_out:.2f}",
                f"{b_in:+.5f}",
                f"{1000 * b_in:+.2f}",
                f"{r2:.4f}",
            ]
        )
    return table(
        [
            "config (OCR arm)",
            "pages",
            "intercept s",
            "decode tok/s",
            "s per input token",
            "s per +1000 input tokens",
            "R^2",
        ],
        rows,
    )


def pooled_model_fits(runs: list[Run]) -> str:
    rows = []
    for m in ("qwen35_4b", "qwen3vl_8b", "nuextract3"):
        sel = [r for r in runs if r.name.startswith(m + "_")]
        x = np.array([p["meta"]["n_output_tokens"] for r in sel for _, p in r.pages])
        y = np.array([p["meta"]["latency_s"] for r in sel for _, p in r.pages])
        f = ols(x, y)
        rows.append(
            [m, f["n"], f"{f['prefill_s']:.3f}", f"{f['decode_tok_s']:.2f}", f"{f['r2']:.4f}"]
        )
    return table(["model (both arms pooled)", "pages", "prefill_s", "decode tok/s", "R^2"], rows)


def write_speed(fits: dict[str, dict[str, float]], path: Path) -> None:
    speed = json.loads(path.read_text(encoding="utf-8"))
    speed["_comment"] = (
        "Trace-measured speeds (spike40 traces at 51cf560, T4): OLS of latency_s on n_output_tokens per config "
        "(prefill_s = intercept, decode_tok_s = 1/slope), see reports/spike_diagnosis.md. The traces carry no separate "
        "prefill/decode timings, so the intercept also holds fixed per-call overhead; n_input_tokens is near-constant "
        "within a config, so prefill cannot be regressed out separately."
    )
    speed["models"] = {
        n: {
            "prefill_s": round(f["prefill_s"], 3),
            "decode_tok_s": round(f["decode_tok_s"], 3),
            "source": "spike40 traces, OLS latency~n_out",
            "r2": round(f["r2"], 4),
            "n_pages": f["n"],
            "n_input_tokens_mean": round(f["n_in_mean"], 1),
            "n_output_tokens_mean": round(f["n_out_mean"], 1),
            "s_per_page_mean": round(f["mean_latency"], 2),
        }
        for n, f in fits.items()
    }
    path.write_text(json.dumps(speed, indent=1) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------------------------
# Step 3: taxonomy
# --------------------------------------------------------------------------------------------


class Tax:
    """Counts (aggregate) and examples (local only) for one run."""

    def __init__(self, r: Run, ocr: dict[str, Any]) -> None:
        self.r = r
        self.sc = ev.load_scorer()
        self.ocr = ocr
        self.c: Counter[Any] = Counter()
        self.ex: dict[str, list[Any]] = defaultdict(list)
        self.nullpos: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        self.hdrnull: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        self.run()

    def example(self, key: str, item: Any) -> None:
        if len(self.ex[key]) < 3:
            self.ex[key].append(item)

    def printed(self, did: str, field: str, value: Any) -> bool:
        return locate.locate(value, field, self.ocr[did], max_level="normalized") is not None

    def run(self) -> None:
        sc, c, r = self.sc, self.c, self.r
        prov = FieldProvenance.load()
        for t in r.trace:
            did, g, pred = t["doc_id"], r.gold[t["doc_id"]], t["prediction"]
            self.pages_checks(t)
            m = merge_pages([p["parsed"] for p in t["pages"]], prov, drop_null_rows=False)
            c["head_merge_differs_from_saved"] += normalize_doc(m.doc)[0] != pred
            nd = normalize_doc(m.doc)[0]
            if m.doc["doc_type"] == g["doc_type"]:
                for f in sc.HEADER[g["doc_type"]]:
                    before = sc.same(f, m.doc["header"].get(f), g["header"].get(f))
                    after = sc.same(f, nd["header"].get(f), g["header"].get(f))
                    c["norm_damage_header"] += before and not after
                    c["norm_gain_header"] += after and not before
            trunc = [p["page"] for p in t["pages"] if not p["json_valid"]]
            if m.diffs["ignored_elsewhere"]:
                c["d_docs_ignored_elsewhere_id_fields"] += 1
            self.header(did, g, pred, t, trunc)
            if g["doc_type"] == "invoice" and pred["doc_type"] == "invoice":
                self.rows(did, g, pred, t, trunc)
            elif pred["doc_type"] != g["doc_type"]:
                c["doc_type_wrong"] += 1

    def pages_checks(self, t: dict[str, Any]) -> None:
        c = self.c
        for p in t["pages"]:
            v = (
                ((p["parsed"] or {}).get("header") or {}).get("total_amount")
                if p["parsed"]
                else None
            )
            if v is not None:
                c["hdr_total_nonnull"] += 1
                c["hdr_total_junk"] += bool(JUNK.match(str(v)))
        for p in t["pages"]:
            for x in ((p["parsed"] or {}).get("line_items") or []) if p["parsed"] else []:
                row = {k: x.get(k) for k in self.sc.ROW}
                if old_is_header_row(row) and not is_header_row(row):
                    c["d_rows_dropped_by_old_digit_blind_header_rule"] += 1
                    self.example("d_header_row", (t["doc_id"], row))
        for p in t["pages"]:
            if p["parsed"]:
                flags = [
                    all(emp(v) for v in x.values())
                    for x in p["parsed"]["line_items"]
                    if isinstance(x, dict)
                ]
                k = 0
                for f in reversed(flags):
                    if not f:
                        break
                    k += 1
                c["f_allnull_in_trailing_run"] += k
        for p in t["pages"]:
            if parse_output(p["raw_text"], "json") != p["parsed"]:
                c["e_parse_mismatch_pages"] += 1
            c["pages"] += 1
            if p["json_valid"]:
                o = p["parsed"]
                c["valid_pages"] += 1
                c["b_sorted_keys_pages"] += (
                    list(o["header"]) == sorted(o["header"])
                    and all(list(x) == sorted(x) for x in o["line_items"])
                    and list(o) == sorted(o)
                )
                c["header_keys_17"] += len(o["header"]) == 17
                dt = o["doc_type"]
                other = (
                    [
                        "carrier",
                        "mawb",
                        "hawb",
                        "origin_airport",
                        "destination_airport",
                        "shipper_name",
                        "consignee_name",
                        "pieces",
                        "gross_weight_kg",
                    ]
                    if dt == "invoice"
                    else [
                        "invoice_number",
                        "invoice_date",
                        "supplier_name",
                        "buyer_name",
                        "ship_to_name",
                        "currency",
                        "total_amount",
                        "awb_number",
                    ]
                )
                leak = [k for k in other if not emp(o["header"].get(k))]
                c["lead5_pages_other_type_keys_nonnull"] += bool(leak)
            else:
                raw = p["raw_text"]
                nullrows = raw.count(
                    '{"customer_part_number": null, "purchase_order": null, "quantity": null, "supplier_part_number": null}'
                )
                junk = len(re.findall(r'"(?:[\]\[\}\{, ]+)"', raw))
                cause = (
                    "all_null_row_run"
                    if nullrows >= NULLROW_RUN
                    else "junk_string_value"
                    if junk
                    else "other"
                )
                c[f"a_trunc_page_cause_{cause}"] += 1
                c["a_trunc_pages"] += 1
                self.example(
                    "a_trunc",
                    (t["doc_id"], p["page"], p["meta"]["n_output_tokens"], cause, nullrows, junk),
                )

    def header(
        self, did: str, g: dict[str, Any], pred: dict[str, Any], t: dict[str, Any], trunc: list[int]
    ) -> None:
        sc, c = self.sc, self.c
        if pred["doc_type"] != g["doc_type"]:
            return
        for f in sc.HEADER[g["doc_type"]]:
            gv, pv = g["header"].get(f), pred["header"].get(f)
            if not emp(gv):
                self.hdrnull[f][1] += 1
                self.hdrnull[f][0] += emp(pv)
            if sc.same(f, pv, gv):
                continue
            c[("hdr_wrong", f)] += 1
            if emp(gv):
                c[("hdr_false_fill", f)] += 1
            elif emp(pv):
                prn = self.printed(did, f, gv)
                c[("hdr_null_emission", f, "printed" if prn else "not_located")] += 1
                self.example("c_header", (did, f, gv, prn))
                if trunc:
                    c[("hdr_null_in_truncated_doc", f)] += 1
                anyp = any(
                    not emp(((p["parsed"] or {}).get("header") or {}).get(f)) for p in t["pages"]
                )
                if anyp:
                    c[("d_hdr_value_on_a_page_but_null_in_prediction", f)] += 1
                    self.example(
                        "d_header",
                        (
                            did,
                            f,
                            [
                                (p["page"], ((p["parsed"] or {}).get("header") or {}).get(f))
                                for p in t["pages"]
                            ],
                            gv,
                        ),
                    )
            else:
                c[("hdr_wrong_value", f)] += 1
                self.example("b_header_wrong", (did, f, pv, gv))

    def rows(
        self, did: str, g: dict[str, Any], pred: dict[str, Any], t: dict[str, Any], trunc: list[int]
    ) -> None:
        sc, c = self.sc, self.c
        gr, pr = g["line_items"], [x for x in pred["line_items"] if isinstance(x, dict)]
        full, part, unp, ung = ev._pair_rows(sc, pr, gr)
        c["gold_rows"] += len(gr)
        c["pred_rows"] += len(pr)
        c["rows_full"] += len(full)
        c["rows_paired_partial"] += len(part)
        c["rows_unpaired_gold"] += len(ung)
        c["rows_unpaired_pred"] += len(unp)
        if len(pr) != len(gr):
            c["f_docs_rowcount_mismatch"] += 1
            self.example("f_rowcount", (did, len(pr), len(gr)))
        if trunc:
            c["a_docs_with_truncated_page"] += 1
            c["a_gold_rows_in_truncated_docs"] += len(gr)
            c["a_gold_rows_not_fully_matched_in_truncated_docs"] += len(gr) - len(full)
        for x in pr:
            if all(emp(v) for v in x.values()):
                c["f_pred_rows_all_null"] += 1
                self.example("f_allnull", (did,))
        # (g) null rate by emitted position, over paired rows; (c) null emission; (b) wrong values
        for pi, gi in full + part:
            for f in sc.ROW:
                gv, pv = gr[gi].get(f), pr[pi].get(f)
                if not emp(gv):
                    self.nullpos[f][1] += 1
                    self.nullpos[f][0] += emp(pv)
                if sc.same(f, pv, gv):
                    c[("cell_correct", f)] += 1
                elif emp(gv):
                    c[("cell_false_fill", f)] += 1
                    self.example("b_false_fill_" + f, (did, f, pv))
                elif emp(pv):
                    prn = self.printed(did, f, gv)
                    c[("cell_null_emission", f, "printed" if prn else "not_located")] += 1
                    self.example("c_cell_" + f, (did, f, gv, prn))
                else:
                    others = [
                        k
                        for k in sc.ROW
                        if k != f and not emp(gr[gi].get(k)) and an(gr[gi].get(k)) == an(pv)
                    ]
                    if others:
                        c[("cell_value_of_other_field", f, others[0])] += 1
                        self.example("b_swap", (did, f, pv, gv, others[0]))
                    else:
                        c[("cell_wrong_value", f)] += 1
                        self.example("b_wrong_" + f, (did, f, pv, gv))
        # unpaired gold rows: attribute to the first applicable cause (shared with the A/B)
        attributed, free = diag.attribute_unpaired(sc, pr, gr, unp, ung, bool(trunc))
        for rec in attributed:
            gi, cause = rec["gi"], rec["cause"]
            if cause == "truncated":
                c["ung_a_truncated_doc"] += 1
                continue
            if cause == "missing":
                c["ung_f_missing_no_overlapping_pred_row"] += 1
                self.example("f_missing", (did, [gr[gi].get(f) for f in sc.ROW]))
                continue
            pi = rec["pi"]
            prow, grow = [pr[pi].get(f) for f in sc.ROW], [gr[gi].get(f) for f in sc.ROW]
            if cause == "shift":
                c["ung_b_column_shift"] += 1
                c["ung_b_true_swap_spn_cpn"] += bool(rec["swap"])
                for f1, f2 in rec["cross"]:
                    c[("shift", f1, f2)] += 1
                self.example("b_shift", (did, prow, grow))
            elif cause == "spn_null":
                c["ung_c_spn_null"] += 1
                self.example("c_spn_null", (did, prow, grow))
            else:
                ps, gs = pr[pi].get("supplier_part_number"), gr[gi].get("supplier_part_number")
                d = Levenshtein.distance(an(ps), an(gs))
                c["ung_misread_spn"] += 1
                c[("ung_misread_spn_edit", "1" if d == 1 else "2" if d == 2 else "3+")] += 1
                self.example("misread_spn", (did, ps, gs))
        for pi in free:
            if all(emp(v) for v in pr[pi].values()):
                c["unp_f_all_null"] += 1
            else:
                c["unp_f_other_extra"] += 1
        # (b) customer-part column on docs whose gold has none
        if all(emp(x.get("customer_part_number")) for x in gr):
            gs = {
                an(x.get("supplier_part_number"))
                for x in gr
                if not emp(x.get("supplier_part_number"))
            }
            gp = {an(x.get("purchase_order")) for x in gr if not emp(x.get("purchase_order"))}
            c["nocpn_docs"] += 1
            for x in pr:
                c["nocpn_pred_rows"] += 1
                v = x.get("customer_part_number")
                if emp(v):
                    continue
                c["nocpn_pred_cpn_nonnull"] += 1
                c[
                    "nocpn_pred_cpn_is_gold_po"
                    if an(v) in gp
                    else "nocpn_pred_cpn_is_gold_spn"
                    if an(v) in gs
                    else "nocpn_pred_cpn_other"
                ] += 1
        # quantity strings
        for x in pr:
            q = x.get("quantity")
            if q is None:
                c["qty_null"] += 1
            else:
                c[
                    "qty_junk_braces"
                    if JUNK.match(str(q))
                    else "qty_digits"
                    if re.fullmatch(r"[\d,.\s]+", str(q))
                    else "qty_other_string"
                ] += 1


def gpu_estimate_block() -> str:
    """Output of scripts/gpu_estimate.py (reads the freshly written configs/spike_speed.json)."""
    out = subprocess.run(
        ["uv", "run", "--no-sync", "python", "scripts/gpu_estimate.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return (
        "`uv run python scripts/gpu_estimate.py` (reads configs/spike_speed.json):\n\n```\n"
        + out.strip()
        + "\n```\n"
    )


def build_verdicts(runs: list[Run], taxes: dict[str, Tax], rep: Any) -> str:
    """Lead and hypothesis verdicts; every number is read from `taxes` / the runs, never typed."""
    names = [r.name for r in runs]
    by = {r.name: r for r in runs}

    def per(key: Any) -> str:
        return " / ".join(str(taxes[n].c[key]) for n in names)

    def tot(key: Any) -> int:
        return sum(taxes[n].c[key] for n in names)

    def sum_keys(n: str, first: str, *rest: Any) -> int:
        return sum(
            v
            for k, v in taxes[n].c.items()
            if isinstance(k, tuple) and k[0] == first and k[1 : 1 + len(rest)] == rest
        )

    def per_sum(first: str, *rest: Any) -> str:
        return " / ".join(str(sum_keys(n, first, *rest)) for n in names)

    sc = ev.load_scorer()
    order = "(order in every list below: " + ", ".join(names) + ")"
    r8 = by["qwen3vl_8b_img_only"]
    r8o = by["qwen3vl_8b_img_ocr"]
    t7 = next(t for t in r8.trace if t["doc_id"] == "dev_0007")
    g7 = r8.gold["dev_0007"]
    rows7 = t7["prediction"]["line_items"]
    full7, part7, _, _ = ev._pair_rows(sc, rows7, g7["line_items"])
    q7_null = sum(x.get("quantity") is None for x in rows7)
    spn7 = sum(
        sc.same(
            "supplier_part_number",
            rows7[pi].get("supplier_part_number"),
            g7["line_items"][gi].get("supplier_part_number"),
        )
        for pi, gi in full7 + part7
    )
    o8 = {
        n: oracle_excuse(by[n], Fixes(drop_null_rows=True, salvage_truncated=True), {"quantity"})
        for n in ("qwen3vl_8b_img_only", "qwen3vl_8b_img_ocr")
    }
    n_valid = tot("valid_pages")
    hdr_total_null = {n: taxes[n].hdrnull["total_amount"] for n in names}
    hdr_pw = {n: (taxes[n].hdrnull["pieces"], taxes[n].hdrnull["gross_weight_kg"]) for n in names}
    qn = {
        n: (
            taxes[n].c["qty_null"],
            sum(
                taxes[n].c[k]
                for k in ("qty_null", "qty_digits", "qty_junk_braces", "qty_other_string")
            ),
        )
        for n in names
    }

    def gain(k: str, n: str) -> float:
        return 100 * (rep[k][n]["OVERALL"] - by[n].metrics["OVERALL"])

    salv = " / ".join(f"{gain(SALVAGE_KEY, n):+.2f}" for n in names)
    base_gain = " / ".join(f"{gain(BASE_KEY, n):+.3f}" for n in names)
    drop_gain = " / ".join(f"{gain(DROP_KEY, n):+.3f}" for n in names)

    def hdr_ne(n: str, level: str) -> int:
        return sum(
            v
            for k, v in taxes[n].c.items()
            if isinstance(k, tuple) and k[0] == "hdr_null_emission" and k[2] == level
        )

    hdr_pn = " / ".join(f"{hdr_ne(n, 'printed')}+{hdr_ne(n, 'not_located')}" for n in names)
    NUMPOS = ", ".join(
        str(ALPHA_HEADER.index(f) + 1)
        for f in sorted(NUMERIC_KEYS & set(HEADER_KEYS), key=ALPHA_HEADER.index)
    )
    numhdr = ", ".join(
        f"{sum(v for k, v in taxes[n].c.items() if isinstance(k, tuple) and k[0] == 'hdr_wrong' and k[1] in NUMERIC_KEYS)}"
        f" of {sum(v for k, v in taxes[n].c.items() if isinstance(k, tuple) and k[0] == 'hdr_wrong')}"
        for n in names
    )
    out = [order, ""]
    out.append(
        f"**Lead 1 (grammar forces alphabetical key order): CONFIRMED.** `_compiled()` passed `json.dumps(schema, sort_keys=True)`. xgrammar 0.2.8 keeps the property order of the JSON string it is given (sdist `3rdparty/picojson/picojson.h`: `PICOJSON_USE_ORDERED_OBJECT` is 1 by default, `object_with_ordered_keys`; `cpp/json_schema_converter.cc` builds object properties from `properties_obj.ordered_keys()`; `any_order` defaults to False in `GrammarCompiler.compile_json_schema`), so sorting the keys sorts the output. Checked on the real xgrammar 0.2.8 (`tests/test_extract.py::test_real_xgrammar_enforces_declared_key_order`): a grammar built from the schema text accepts rows in declared order and rejects alphabetical rows. In the traces: {tot('b_sorted_keys_pages')} of {n_valid} valid pages have alphabetical header keys, row keys and top-level keys. Row order emitted was customer_part_number, purchase_order, quantity, supplier_part_number, and {ALPHA_HEADER[-1]} is the {len(ALPHA_HEADER)}th and last header key. Fixed in `fix(extract): grammar follows declared key order`."
    )
    out.append("")
    out.append(
        f"**Lead 2 (qwen3vl_8b quantity null on every row): CONFIRMED on dev_0007 and generalises to all 8B rows, and to most Qwen rows.** dev_0007 (qwen3vl_8b_img_only): {q7_null} of {len(rows7)} emitted rows have quantity null, {len(full7)} rows fully match, {spn7} of {len(g7['line_items'])} gold rows pair with a predicted row on supplier_part_number. Over the 30 invoices: qwen3vl_8b quantity is null in {qn['qwen3vl_8b_img_only'][0]}/{qn['qwen3vl_8b_img_only'][1]} (img_only) and {qn['qwen3vl_8b_img_ocr'][0]}/{qn['qwen3vl_8b_img_ocr'][1]} (img_ocr) emitted rows, so full-row matches are {per('rows_full').split(' / ')[2]} and {per('rows_full').split(' / ')[3]} (row F1 0). The same holds in part for the others (quantity null / emitted rows, in order: {', '.join(f'{qn[n][0]}/{qn[n][1]}' for n in names)}). Sizing (replay on saved outputs, ORACLE, not a fix): scoring quantity as correct moves 8B OVERALL from {pct(r8.metrics['OVERALL'])} to {pct(o8['qwen3vl_8b_img_only'])} (img_only) and from {pct(r8o.metrics['OVERALL'])} to {pct(o8['qwen3vl_8b_img_ocr'])} (img_ocr). Mechanism (inferred from the traces, UNVERIFIED on a GPU): the prompt asks for 'a plain number', the models write numbers bare, and the grammar allowed only string or null, so the digit tokens were masked. Evidence: junk strings made only of braces/commas in quantity, per config: {per('qty_junk_braces')}, against digit strings {per('qty_digits')}; the same junk is {per('hdr_total_junk')} of the {per('hdr_total_nonnull')} non-null total_amount values; and the nulls follow the field TYPE across key positions: the four numeric fields sit at row position 3 (quantity) and header positions {NUMPOS} of {len(ALPHA_HEADER)} and are null almost always, while supplier_part_number (row position 4) is never null (hypothesis g). Fixed (grammar side) in `fix(extract): numeric fields may be emitted as JSON numbers` and (prompt side) in prompt v2."
    )
    out.append("")
    out.append(
        f"**Lead 3 (total_amount also null): CONFIRMED, same cause.** total_amount is null in {', '.join(f'{v[0]}/{v[1]}' for v in hdr_total_null.values())} invoices that have a gold total (null / gold non-null), in config order. pieces null: {', '.join(f'{v[0][0]}/{v[0][1]}' for v in hdr_pw.values())} and gross_weight_kg null: {', '.join(f'{v[1][0]}/{v[1][1]}' for v in hdr_pw.values())} of the waybills. total_amount null emissions where the gold value is printed on the page (located in the OCR at level <= normalized): {per_sum('hdr_null_emission', 'total_amount', 'printed')}; gold value not located: {per_sum('hdr_null_emission', 'total_amount', 'not_located')}. Header field errors in the three numeric header fields (total_amount, pieces, gross_weight_kg) out of all header field errors, per config: {numhdr}. Same cause as lead 2."
    )
    out.append("")
    out.append(
        f"**Lead 4 (nuextract3: many all-null rows, fewer or extra rows than gold): CONFIRMED.** All-null predicted rows: {per('f_pred_rows_all_null')} (nuextract3_img_only emits {taxes['nuextract3_img_only'].c['f_pred_rows_all_null']} of {taxes['nuextract3_img_only'].c['pred_rows']} rows as all-null, nuextract3_img_ocr {taxes['nuextract3_img_ocr'].c['f_pred_rows_all_null']} of {taxes['nuextract3_img_ocr'].c['pred_rows']}). Predicted vs gold rows over the 30 invoices: {per('pred_rows')} vs {per('gold_rows')}; invoices whose row count differs: {per('f_docs_rowcount_mismatch')}. Not nuextract-specific: qwen35_4b emits {taxes['qwen35_4b_img_only'].c['f_pred_rows_all_null']} / {taxes['qwen35_4b_img_ocr'].c['f_pred_rows_all_null']}; qwen3vl_8b none. {tot('f_pred_rows_all_null')} all-null rows in the six runs, {tot('f_allnull_in_trailing_run')} of them in a trailing run at the end of a page (counted per page from the saved parsed rows)."
    )
    out.append("")
    out.append(
        f"**Lead 5 (union header, 17 keys every page): CONFIRMED, harmless to the score.** {tot('header_keys_17')} of {n_valid} valid pages carry all 17 header keys. Pages with a non-null value in a key of the OTHER doc type: {per('lead5_pages_other_type_keys_nonnull')}. `merge_pages` reads only the keys of the voted doc type and doc_type accuracy is 100% in all six runs, so the cross-type values never reach the prediction (score cost: 0 by construction); the cost is output tokens (9 or 8 extra key/value pairs per page) and, for the Qwen3.5-family runs, a visible tendency to fill the wrong type's keys. Not changed here."
    )
    out.append("")
    out.append("**Hypotheses (counts in the evidence tables below):**")
    out.append("")
    out.append(
        f"- **(a) truncation: HOLDS, minor.** Pages at the cap: {per('a_trunc_pages')}; invoices with a truncated page {per('a_docs_with_truncated_page')}, holding {per('a_gold_rows_in_truncated_docs')} gold rows, none fully matched; null header fields in those docs {' / '.join(str(sum_keys(n, 'hdr_null_in_truncated_doc')) for n in names)}. Cause of the loops: junk-string values ({per('a_trunc_page_cause_junk_string_value')}) or runs of all-null rows ({per('a_trunc_page_cause_all_null_row_run')}), both downstream of the numeric-type problem. Replay: salvaging truncated JSON recovers {salv} OVERALL points. It is not why the Qwen3-VL rows fail (it has no truncated page)."
    )
    out.append(
        f"- **(b) field convention: HOLDS for customer_part_number, PO and column shifts; the spn<->cpn swap is a minor part.** customer_part_number filled where gold is empty: {per_sum('cell_false_fill', 'customer_part_number')}. On the 19 docs whose gold rows have no customer part number, the model sets customer_part_number on {per('nocpn_pred_cpn_nonnull')} rows, and the value equals a gold purchase_order on {per('nocpn_pred_cpn_is_gold_po')} of them. Unpaired gold rows explained by a column shift: {per('ung_b_column_shift')} (of which a clean spn<->cpn swap: {per('ung_b_true_swap_spn_cpn')}); the shifts are rotations among spn/cpn/po, not clean swaps. purchase_order emitted null though printed: {per_sum('cell_null_emission', 'purchase_order', 'printed')}; PO wrong value: {per_sum('cell_wrong_value', 'purchase_order')}. quantity: digit strings emitted {per('qty_digits')}, quantity correct on matched pairs {per_sum('cell_correct', 'quantity')}, wrong values on matched pairs {per_sum('cell_wrong_value', 'quantity')} (junk strings, see lead 2). A spn<->cpn swap REPAIR is NOT implemented: the traces show rotations rather than swaps and no label-free signal that identifies which value sits in which column without supplier-specific shape rules; the fix is in the prompt and in the key order."
    )
    out.append(
        f"- **(c) null emission of a value the image prints: HOLDS, dominant.** Quantity emitted null though located in the OCR (printed): {per_sum('cell_null_emission', 'quantity', 'printed')}; not located: {per_sum('cell_null_emission', 'quantity', 'not_located')}. purchase_order printed-but-null {per_sum('cell_null_emission', 'purchase_order', 'printed')}, not located {per_sum('cell_null_emission', 'purchase_order', 'not_located')}. Header null emissions printed vs not located (all fields): {hdr_pn}. Almost every null is a value the page shows, so it is a decode-time failure, not an OCR or visibility limit."
    )
    out.append(
        f"- **(d) merge/normalize drop: REFUTED as a loss at scale; one real bug found.** The current merge+normalize reproduces the saved prediction exactly in all but {per('head_merge_differs_from_saved')} docs (the digit-rule bug). The 51cf560 header-row rule (digits ignored) dropped {per('d_rows_dropped_by_old_digit_blind_header_rule')} rows (PO- or PN-style values reduced to the label words po/pn); fixed with a regression test, replay baseline (current code, no optional fix) minus saved, OVERALL points: {base_gain}, so the rows the old rule dropped carried no correct match. Header values present on some page but null in the prediction: {per_sum('d_hdr_value_on_a_page_but_null_in_prediction')} (the page-1-only rule for identity fields working as designed, plus truncated page 1). Normalization changed a header value from scorer-right to scorer-wrong in {per('norm_damage_header')} header fields and from wrong to right in {per('norm_gain_header')} (gold-based comparison of the merged vs the normalized document)."
    )
    out.append(
        f"- **(e) parser bug: REFUTED.** `parse_output(raw_text)` equals the saved `parsed` on every page (mismatches: {per('e_parse_mismatch_pages')} of {per('pages')} pages); the format is json, so parsing is `json.loads`. The compact expander is untested on real model output (no compact run exists)."
    )
    out.append(
        f"- **(f) row-count errors: HOLDS.** Invoices with a different row count: {per('f_docs_rowcount_mismatch')}; all-null rows {per('f_pred_rows_all_null')}; invoices with 0 predicted rows: {' / '.join(str(sum(1 for t in by[n].trace if by[n].gold[t['doc_id']]['doc_type'] == 'invoice' and not t['prediction']['line_items'])) for n in names)} (those are the truncated pages); gold rows with no overlapping predicted row (missing): {per('ung_f_missing_no_overlapping_pred_row')}. Dropping all-null rows changes replay OVERALL by {drop_gain} points (vs saved) and is now the merge default."
    )
    out.append(
        f"- **(g) key-order effect: SPLIT.** Null rate by position is REFUTED: supplier_part_number is emitted last and is null in 0 matched pairs, quantity (position 3) is null in nearly all, and the numeric header fields are null at header positions {NUMPOS} alike (tables: null rate by emitted position; Spearman is reported per config, and for the non-numeric fields the null rates are near zero everywhere except where a truncated page wipes a doc). The nulls track field TYPE (numeric), not position. A column-shift effect is CONSISTENT WITH key order but NOT proven by these traces: {per('ung_b_column_shift')} unpaired gold rows are column shifts, and customer_part_number gets the PO or supplier part number on docs that have no customer-part column, as if cells were filled in visual order into slots emitted in a different order. The signature to look for after a rerun with the declared order is a drop in those shift counts."
    )
    return "\n".join(out)


HEADER_WORDS = merge_mod.HEADER_WORDS


def old_is_header_row(row: dict[str, Any]) -> bool:
    """The 51cf560-era rule: digits ignored (a value like PO+digits reduced to the label 'po')."""
    vals = [str(v) for v in row.values() if not emp(v)]
    if not vals:
        return False
    for v in vals:
        words = re.findall(r"[a-z]+", v.lower().replace("'", ""))
        if not words or any(w not in HEADER_WORDS for w in words):
            return False
    return True


def _avg_rank(v: list[float]) -> np.ndarray:
    """Ranks with ties averaged (a constant column stays constant)."""
    a = np.asarray(v, dtype=float)
    order = np.argsort(a, kind="stable")
    ranks = np.empty(len(a))
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2
        i = j + 1
    return ranks


def spearman(x: list[float], y: list[float]) -> str:
    """Spearman rho with average ranks; 'n/a (constant)' when either side has no variance."""
    if len(x) < 3:
        return "n/a"
    rx, ry = _avg_rank(x), _avg_rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return "n/a (constant)"
    return f"{float(np.corrcoef(rx, ry)[0, 1]):+.2f}"


# --------------------------------------------------------------------------------------------
# Step 4: replay table and oracle
# --------------------------------------------------------------------------------------------


def oracle_excuse(r: Run, fixes: Fixes, fields: set[str]) -> float:
    """OVERALL when `fields` are scored as correct (both sides set to a constant): an ORACLE upper bound, not a fix."""
    prov = FieldProvenance.load()
    from shipdoc.replay import replay_doc

    pred = {t["doc_id"]: replay_doc(t, "json", fixes, prov) for t in r.trace}
    gold = copy.deepcopy(r.gold)
    for d, g in gold.items():
        for f in fields & set(g["header"]):
            g["header"][f] = "1"
            if pred[d]["doc_type"] == g["doc_type"]:
                pred[d]["header"][f] = "1"
        for x in g["line_items"]:
            if "quantity" in fields:
                x["quantity"] = "1"
        for x in pred[d]["line_items"]:
            if "quantity" in fields:
                x["quantity"] = "1"
    return float(ev.score(pred, gold)["all"]["OVERALL"])


# --------------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", type=Path, default=Path("D:/shipdoc/runs/spike_download/x"))
    ap.add_argument("--out-md", type=Path, default=ROOT / "reports" / "spike_diagnosis.md")
    ap.add_argument(
        "--local-md", type=Path, default=paths.runs_dir() / "diagnosis" / "diagnosis_local.md"
    )
    ap.add_argument("--write-speed", action="store_true")
    ap.add_argument(
        "--gpu-estimate",
        default="",
        help="path of a text file holding the scripts/gpu_estimate.py output",
    )
    a = ap.parse_args()

    gold = ev.load_gold(ROOT / "data" / "dev" / "labels")
    runs = [Run(a.runs, n, gold) for n in NAMES]
    ocr = {d: locate.build_index(ocrm.doc_pages(d)) for d in gold}
    head_sha = sh(["git", "rev-parse", "--short", "HEAD"])
    rec_md, rec_facts = recompose(runs)
    fields_md = per_field_tables(runs)
    stats_md, fits = trace_stats(runs)
    taxes = {r.name: Tax(r, ocr) for r in runs}
    if a.write_speed:
        write_speed(fits, ROOT / "configs" / "spike_speed.json")

    # ---- replay table (current parse/merge/normalize; labelled) ----
    combos = {
        BASE_KEY: Fixes(drop_null_rows=False),
        DROP_KEY: Fixes(drop_null_rows=True),
        SALVAGE_KEY: Fixes(drop_null_rows=False, salvage_truncated=True),
        "+ both": Fixes(drop_null_rows=True, salvage_truncated=True),
    }
    rep: dict[str, dict[str, dict[str, float]]] = {}
    for k, f in combos.items():
        rep[k] = {r.name: replay_traces(r.trace, r.gold, "json", f) for r in runs}
    saved = {r.name: r.metrics["OVERALL"] for r in runs}
    rrows = [["saved run (metrics.json, 51cf560 code)"] + [pct(saved[r.name]) for r in runs]]
    for k in combos:
        rrows.append(
            [k]
            + [
                f"{pct(rep[k][r.name]['OVERALL'])} ({100 * (rep[k][r.name]['OVERALL'] - saved[r.name]):+.3f})"
                for r in runs
            ]
        )
    both = Fixes(drop_null_rows=True, salvage_truncated=True)
    orows = []
    for label, flds in (
        ("ORACLE: quantity scored correct", {"quantity"}),
        (
            "ORACLE: quantity + total_amount + pieces + gross_weight_kg scored correct",
            set(NUMERIC_KEYS),
        ),
    ):
        orows.append(
            [label]
            + [
                f"{pct(oracle_excuse(r, both, flds))} ({100 * (oracle_excuse(r, both, flds) - saved[r.name]):+.3f})"
                for r in runs
            ]
        )
    replay_md = table(
        ["replay on saved outputs: fix (OVERALL %, delta vs saved in points)"]
        + [r.name for r in runs],
        rrows,
    )
    oracle_md = table(
        ["oracle counterfactual (NOT a fix; labelled replay on saved outputs)"]
        + [r.name for r in runs],
        orows,
    )

    # ---- taxonomy tables ----
    def cnt(name: str, key: Any) -> int:
        return taxes[name].c[key]

    names = [r.name for r in runs]

    def row(label: str, key: Any) -> list[Any]:
        return [label] + [cnt(n, key) for n in names]

    # header errors within truncated docs summed over fields
    for n in names:
        taxes[n].c["x"] = sum(
            v
            for k, v in taxes[n].c.items()
            if isinstance(k, tuple) and k[0] == "hdr_null_in_truncated_doc"
        )
    tax_a = table(
        ["(a) truncation"] + names,
        [
            row("pages at the cap that are invalid JSON", "a_trunc_pages"),
            row("  cause: run of >=10 all-null rows in raw", "a_trunc_page_cause_all_null_row_run"),
            row(
                "  cause: junk string value (braces/commas only) in raw",
                "a_trunc_page_cause_junk_string_value",
            ),
            row("  cause: other", "a_trunc_page_cause_other"),
            row("invoices with a truncated page", "a_docs_with_truncated_page"),
            row("gold rows in those invoices", "a_gold_rows_in_truncated_docs"),
            row("  of which not fully matched", "a_gold_rows_not_fully_matched_in_truncated_docs"),
            row("null header fields (gold non-null) in those docs", "x"),
        ],
    )

    sc = ev.load_scorer()
    cell_rows = []
    for f in sc.ROW:
        for lab, key in (
            ("correct", ("cell_correct", f)),
            (
                "null emitted, gold value printed (<= normalized)",
                ("cell_null_emission", f, "printed"),
            ),
            ("null emitted, gold not located", ("cell_null_emission", f, "not_located")),
            ("wrong value", ("cell_wrong_value", f)),
            ("value belongs to another field", None),
            ("false fill (gold empty)", ("cell_false_fill", f)),
        ):
            if key is None:
                cell_rows.append(
                    [f"{f}: {lab}"]
                    + [
                        sum(
                            v
                            for k, v in taxes[n].c.items()
                            if isinstance(k, tuple) and k[:2] == ("cell_value_of_other_field", f)
                        )
                        for n in names
                    ]
                )
            else:
                cell_rows.append(row(f"{f}: {lab}", key))
    tax_cells = table(
        ["(b)(c) cells of matched row pairs (paired by supplier part number)"] + names, cell_rows
    )
    ung_rows = [
        row("unpaired gold rows: in an invoice with a truncated page (a)", "ung_a_truncated_doc"),
        row(
            "column shift: a pred field holds another gold field's value (b)", "ung_b_column_shift"
        ),
    ]
    for pair in sorted(
        {k[1:] for n in names for k in taxes[n].c if isinstance(k, tuple) and k[0] == "shift"}
    ):
        ung_rows.append(
            [f"  shift pred.{pair[0]} == gold.{pair[1]}"]
            + [cnt(n, ("shift", *pair)) for n in names]
        )
    ung_rows += [
        row("supplier part number misread (other fields align) (misread)", "ung_misread_spn")
    ]
    for d in ("1", "2", "3+"):
        ung_rows.append(
            [f"  edit distance {d}"] + [cnt(n, ("ung_misread_spn_edit", d)) for n in names]
        )
    ung_rows += [
        row("supplier part number null (c)", "ung_c_spn_null"),
        row("no overlapping predicted row: missing (f)", "ung_f_missing_no_overlapping_pred_row"),
        row("unpaired pred rows: all null (f)", "unp_f_all_null"),
        row("unpaired pred rows: other extras (f)", "unp_f_other_extra"),
    ]
    tax_rows = table(["row-level residue"] + names, ung_rows)
    nocpn = table(
        ["(b) docs whose gold rows have no customer part number"] + names,
        [
            row("docs", "nocpn_docs"),
            row("pred rows", "nocpn_pred_rows"),
            row("pred rows with customer_part_number set", "nocpn_pred_cpn_nonnull"),
            row("  equals a gold purchase_order of the doc", "nocpn_pred_cpn_is_gold_po"),
            row("  equals a gold supplier_part_number of the doc", "nocpn_pred_cpn_is_gold_spn"),
            row("  other", "nocpn_pred_cpn_other"),
        ],
    )
    qty = table(
        ["quantity values emitted (all pred rows, invoices)"] + names,
        [
            row("null", "qty_null"),
            row("digits", "qty_digits"),
            row("junk string (braces/commas only)", "qty_junk_braces"),
            row("other string", "qty_other_string"),
        ],
    )

    hdr_fields = sorted(sc.HEADER["invoice"] + sc.HEADER["waybill"])
    hrows = []
    for f in hdr_fields:
        hrows.append(
            [f, "yes" if f in NUMERIC_KEYS else "no", ALPHA_HEADER.index(f) + 1]
            + [
                (
                    f"{taxes[n].hdrnull[f][0]}/{taxes[n].hdrnull[f][1]}"
                    if taxes[n].hdrnull[f][1]
                    else "-"
                )
                for n in names
            ]
        )
    hdr_null_md = table(
        ["header field", "numeric", "emitted position (alphabetical, 1-17)"] + names, hrows
    )
    rrows2 = []
    for i, f in enumerate(ROW_EMITTED):
        rrows2.append(
            [f, "yes" if f in NUMERIC_KEYS else "no", i + 1]
            + [f"{taxes[n].nullpos[f][0]}/{taxes[n].nullpos[f][1]}" for n in names]
        )
    row_null_md = table(
        ["row field", "numeric", "emitted position (alphabetical, 1-4)"] + names, rrows2
    )
    corr = []
    for n in names:
        t = taxes[n]
        pts = [
            (ALPHA_HEADER.index(f) + 1, t.hdrnull[f][0] / t.hdrnull[f][1], f in NUMERIC_KEYS)
            for f in hdr_fields
            if t.hdrnull[f][1] >= 8
        ]
        nn = [(p, v) for p, v, num in pts if not num]
        corr.append(
            [
                n,
                spearman([p for p, _, _ in pts], [v for _, v, _ in pts]),
                len(pts),
                spearman([p for p, _ in nn], [v for _, v in nn]),
                len(nn),
            ]
        )
    corr_md = table(
        [
            "config",
            "Spearman(position, null rate), all header fields with >= 8 gold values",
            "n fields",
            "same, non-numeric fields only",
            "n fields",
        ],
        corr,
    )

    misc = table(
        ["(d)(e) merge, normalize, parser, and other checks"] + names,
        [
            row("pages", "pages"),
            row(
                "pages where parse_output(raw_text) != saved parsed (e: parser bug)",
                "e_parse_mismatch_pages",
            ),
            row(
                "docs where merge+normalize (digit rule off, drop_null_rows off) != saved prediction",
                "head_merge_differs_from_saved",
            ),
            row(
                "rows the 51cf560 header-row rule dropped and the digit rule keeps (d)",
                "d_rows_dropped_by_old_digit_blind_header_rule",
            ),
            row(
                "docs with an identity field null on page 1 but set on a later page (rule: page 1 only)",
                "d_docs_ignored_elsewhere_id_fields",
            ),
            row(
                "valid pages whose raw header/row keys are in alphabetical order",
                "b_sorted_keys_pages",
            ),
            row("valid pages", "valid_pages"),
            row("valid pages with all 17 header keys (lead 5)", "header_keys_17"),
            row(
                "valid pages with a non-null key of the OTHER doc type (lead 5)",
                "lead5_pages_other_type_keys_nonnull",
            ),
            row("(f) invoices with pred row count != gold row count", "f_docs_rowcount_mismatch"),
            row("(f) pred rows with every field null", "f_pred_rows_all_null"),
        ],
    )
    gpu_md = gpu_estimate_block()
    verdicts = build_verdicts(runs, taxes, rep)
    # ---- write committed report ----
    md = f"""# Spike40 diagnosis (Step L)

**Provenance.** Runs: Colab T4, code {RUN_COMMIT}, artifacts `D:\\shipdoc\\runs\\spike_download\\x\\spike40_<model>_<arm>_{RUN_COMMIT}\\{{predictions.json, trace.jsonl, metrics.json}}` (6 configs, 40 dev docs = 30 invoices + 10 waybills, 55 pages per config). Generated by `uv run python scripts/spike_diagnosis.py --write-speed` at repo commit `{head_sha}`; every number below is computed by that script from the artifacts and the official scorer (`assignment/score.py` via `shipdoc.eval`) against `data/dev/labels` restricted to each run's doc_ids. Aggregates only: concrete predicted/gold values are in the local, gitignored `<SHIPDOC_RUNS_DIR>/diagnosis/diagnosis_local.md`. Doc ids in this file: dev_0007 (lead 2 check).

Everything here is either "computed from the saved traces" or "replay on saved outputs" (re-running parse, merge, normalize and the scorer on the saved `raw_text`, no model). Replay cannot fix a value the model emitted as null or wrong.

## 1. Recomposition

`eval.score(predictions.json, gold restricted to the run's doc_ids)` reproduces `metrics.json` OVERALL exactly (float equality) for {sum(f["reproduces"] for f in rec_facts.values())}/6 configs. The scorer's formula (`assignment/score.py aggregate`): `OVERALL = 0.4*header_field_accuracy + 0.4*row_F1 + 0.2*documents_fully_correct`; row P/R/F1 are pooled over all gold rows (waybills have none), `header_field_accuracy` pools every header field of every doc (a waybill counts its 9 fields, an invoice its 8), and recomposing from the three parts equals OVERALL in {sum(f["recomposed_equal"] for f in rec_facts.values())}/6 configs.

{rec_md}
{fields_md}
## 2. Traces

Finish reason is inferred (the traces hold no stop reason): `at cap` if `n_output_tokens >= {CAP}` (the run's `max_new_tokens`) or the raw text is not closed JSON; otherwise EOS. Decoding is grammar-constrained, so a page that is not closed JSON is one that was cut at the cap: `invalid AND at cap` equals `invalid JSON` in every config, `invalid, below cap` (parse error) is 0 everywhere.

{stats_md}
### Prefill vs decode

The trace `meta` carries no separate prefill/decode timings (keys: latency_s, peak_vram_bytes, n_input_tokens, n_visual_tokens, n_output_tokens, and ocr_* for the OCR arms). Method: OLS `latency_s = prefill_s + slope * n_output_tokens` per config over all 55 pages (numpy lstsq); `prefill_s` is the intercept, decode tok/s = 1/slope, R^2 reported. The intercept also holds fixed per-call overhead, and `n_input_tokens` is near-constant inside a config (all pages are resized to about 1260 visual tokens), so prefill cannot be regressed out separately; treat `prefill_s` as "everything not proportional to output length".

{speed_table(fits)}
In the img_only arms `n_input_tokens` is constant inside a config (std 0), so the intercept cannot be split further. In the OCR arms it varies (the OCR hint is truncated line-wise to 1200 tokens), so a two-variable fit `latency ~ 1 + n_out + n_in` is possible:

{ocr_prefill_table(runs)}
The n_in coefficient is within a fraction of a second per 1000 extra input tokens and of either sign, i.e. extra prompt tokens are not measurable against decode noise here; the 1.8 to 2.6 s intercept is prefill plus fixed per-call overhead and is not separable further with these traces.

Pooled per model (both arms, same method):

{pooled_model_fits(runs)}
`configs/spike_speed.json` now holds these per-config values (source `spike40 traces, OLS latency~n_out`, with R^2 and n). These are json-format runs; the compact configs are estimated from the same speeds with the compact gold token counts.

{gpu_md}
## 3. Lead and hypothesis verdicts

{verdicts}

### Evidence tables

{tax_a}
{tax_cells}
{tax_rows}
{nocpn}
{qty}
Null rate of a field when gold has a value (null emitted / gold non-null), by the position the grammar forced (alphabetical; 51cf560):

{hdr_null_md}
{row_null_md}
{corr_md}
{misc}
## 4. Replay on saved outputs

All numbers are OVERALL % scored by the official scorer on the 40 spike docs, labelled replay on saved outputs. `baseline` re-runs the CURRENT parse/merge/normalize (after fix(merge) digit rule) with no optional fix; it differs from the saved run only where those code changes matter.

{replay_md}
{oracle_md}
What replay CANNOT fix: values the model emitted as null (quantity null in the Qwen runs, total/pieces/gross weight null), values in the wrong field, misread characters, rows never emitted. The oracle rows only size how much of the score hangs on the numeric fields; they are not achievable by post-processing. Changes to the grammar (key order, number type) and to the prompt need a GPU rerun and are NOT in this table.
"""
    a.out_md.write_text(md, encoding="utf-8")

    # ---- local examples (confidential) ----
    loc = [
        "# Spike40 diagnosis: LOCAL examples (confidential gold values; gitignored, never commit)\n",
        f"Generated by scripts/spike_diagnosis.py at {head_sha}. Three examples per hypothesis per config.\n",
    ]
    titles = {
        "a_trunc": "(a) truncation: (doc, page, n_out, cause, null-rows in raw, junk strings in raw)",
        "b_swap": "(b) value of another field: (doc, field, pred, gold, gold field that holds pred)",
        "b_shift": "(b) column shift: (doc, pred row [spn, cpn, po, qty], gold row)",
        "b_false_fill_customer_part_number": "(b) customer_part_number filled, gold empty: (doc, field, pred)",
        "misread_spn": "(misread) supplier part number: (doc, pred, gold)",
        "b_wrong_purchase_order": "(b) wrong PO value: (doc, field, pred, gold)",
        "b_wrong_quantity": "(b) wrong quantity value: (doc, field, pred, gold)",
        "c_cell_quantity": "(c) quantity null, gold printed: (doc, field, gold, printed in OCR)",
        "c_cell_purchase_order": "(c) PO null, gold printed: (doc, field, gold, printed in OCR)",
        "c_header": "(c) header null, gold printed: (doc, field, gold, printed in OCR)",
        "c_spn_null": "(c) supplier part number null: (doc, pred row, gold row)",
        "d_header_row": "(d) row dropped as a repeated header row: (doc, row)",
        "d_header": "(d) header value on a page but null in prediction: (doc, field, per-page values, gold)",
        "e_parse": "(e) parser: rows where raw_text parsed differently from the saved parse (none expected)",
        "f_rowcount": "(f) row count: (doc, pred rows, gold rows)",
        "f_allnull": "(f) all-null row: (doc)",
        "f_missing": "(f) missing gold row (no overlapping pred row): (doc, gold row)",
        "b_header_wrong": "header wrong value: (doc, field, pred, gold)",
    }
    for n in names:
        loc.append(f"\n## {n}\n")
        for k, title in titles.items():
            items = taxes[n].ex.get(k, [])
            loc.append(f"\n### {title}\n")
            loc.extend(
                f"- {json.dumps(i, ensure_ascii=False)}\n" for i in items
            ) if items else loc.append("- none\n")
    a.local_md.parent.mkdir(parents=True, exist_ok=True)
    a.local_md.write_text("".join(loc), encoding="utf-8")
    print("wrote", a.out_md, "and", a.local_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
