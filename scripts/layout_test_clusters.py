"""Phase 6.4: layout clusters of the test docs and the unseen-layout share (label-free for test).

Pre-registered protocol (fixed before any test page was looked at):
  * signature = page 1 of the doc; variant chosen by ARI vs supplier_group on train+dev INVOICES
    (ties within 0.01 of `base` go to `base`); all variants are reported.
  * reference model = KMeans (seed 42, k by silhouette over 8..30) on train+dev pages of one doc
    type; the test split is only ever assigned to its nearest centre.
  * seen/unseen threshold = 95th percentile of OUT-OF-SAMPLE seen distances (5-fold over train+dev
    docs). Operating characteristics from leave-one-supplier-group-out (LOGO).
  * test doc type = 'invoice' keyword family on page 1 absent => waybill (validated on train+dev).

Test OCR text is never printed or stored: only ids, ints and floats leave this script.
Run: ``uv run python scripts/layout_test_clusters.py``
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _corpus import ROOT, DocRec, load_corpus

from shipdoc import ocr, paths
from shipdoc.cluster import K_RANGE, SEED
from shipdoc.layout_sig import (
    SIGNATURES,
    AssignmentRow,
    ari,
    bootstrap_ari_ci,
    calibrate_threshold,
    corrected_share,
    fit_reference,
    invoice_keyword_present,
    logo_distances,
    percentile,
    purity,
    seen_distances,
    wilson_ci,
    write_assignments,
)

# Waybill page-1 signatures are near-constant (duplicate points); the k search then warns.
warnings.filterwarnings("ignore", message="Number of distinct clusters")

Q = 95.0  # pre-stated percentile of the seen-distance distribution
TIE = 0.01  # a variant must beat `base` by more than this ARI to be selected
OUT_MD = ROOT / "reports" / "layout_test_clusters.md"
OUT_CSV = paths.tmp_dir() / "test_layout_assignments.csv"


def _nn_dist(ref: np.ndarray, x: np.ndarray, exclude_self: bool = False) -> np.ndarray:
    """Distance of each row of `x` to its nearest row of `ref` (self excluded if requested)."""
    d = np.linalg.norm(x[:, None, :] - ref[None, :, :], axis=2)
    if exclude_self:
        np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


def _page1(doc_id: str) -> ocr.PageOcr:
    return ocr.load_page(doc_id.split("_", 1)[0], f"{doc_id}_p1")


def _fmt_ci(lo: float, hi: float) -> str:
    return f"[{lo:.3f}, {hi:.3f}]"


def main() -> int:
    """Run the protocol, write the report and the per-doc CSV outside the repo."""
    corpus = load_corpus()
    pages = {r.doc_id: _page1(r.doc_id) for r in corpus}
    by_type: dict[str, list[DocRec]] = {
        t: [r for r in corpus if r.doc_type == t] for t in ("invoice", "waybill")
    }
    lines: list[str] = []

    # ---- 1. variants on train+dev --------------------------------------------------------
    var_rows: list[str] = []
    results: dict[str, dict[str, dict[str, float]]] = {}
    for name, fn in SIGNATURES.items():
        results[name] = {}
        for t, recs in by_type.items():
            ids = [r.doc_id for r in recs]
            x = np.vstack([fn(pages[d]) for d in ids])
            fit = fit_reference(ids, x)
            lab = [r.group for r in recs]
            cl = fit.labels()
            lo, hi = bootstrap_ari_ci(lab, cl)
            res = {
                "n": len(ids),
                "k": fit.k,
                "sil": fit.silhouette,
                "ari": ari(lab, cl),
                "lo": lo,
                "hi": hi,
                "purity": purity(lab, cl),
            }
            results[name][t] = res
            var_rows.append(
                f"| {name} | {t} | {res['n']} | {fit.k} | {fit.silhouette:.3f} | "
                f"{res['ari']:.3f} {_fmt_ci(lo, hi)} | {res['purity']:.3f} |"
            )
    best = max(results, key=lambda n: results[n]["invoice"]["ari"])
    base_ari = results["base"]["invoice"]["ari"]
    chosen = best if results[best]["invoice"]["ari"] > base_ari + TIE else "base"
    sig_fn = SIGNATURES[chosen]
    print(f"chosen variant: {chosen}")

    # ---- doc-type rule validation on train+dev --------------------------------------------
    pred_wb = np.array([not invoice_keyword_present(pages[r.doc_id]) for r in corpus])
    gold_wb = np.array([r.doc_type == "waybill" for r in corpus])
    tp, fp = int(np.sum(pred_wb & gold_wb)), int(np.sum(pred_wb & ~gold_wb))
    fn_, tn = int(np.sum(~pred_wb & gold_wb)), int(np.sum(~pred_wb & ~gold_wb))

    # ---- 2. test docs (label-free: ids and signatures only) ----------------------------------
    test_dir = paths.ocr_cache_dir() / ocr.DEFAULT_ENGINE / "test"
    stems = sorted(p.stem for p in test_dir.glob("test_*_p*.json"))
    test_ids = sorted({s.rsplit("_p", 1)[0] for s in stems})
    n_test_pages = len(stems)
    tpages = {d: _page1(d) for d in test_ids}
    test_type = {
        d: ("invoice" if invoice_keyword_present(tpages[d]) else "waybill") for d in test_ids
    }
    tx_all = {d: sig_fn(tpages[d]) for d in test_ids}

    sec: dict[str, dict[str, object]] = {}
    rows: list[AssignmentRow] = []
    for t, recs in by_type.items():
        ids = [r.doc_id for r in recs]
        grp = [r.group for r in recs]
        x = np.vstack([sig_fn(pages[d]) for d in ids])
        fit = fit_reference(ids, x)
        seen = seen_distances(ids, x, fit.k)
        logo = logo_distances(ids, grp, x, fit.k)
        cal = calibrate_threshold(seen, logo, Q)
        # method 2b: 1-NN distance to individual reference pages (LOO for seen, LOGO for unseen)
        nn_seen = _nn_dist(x, x, exclude_self=True)
        g = np.asarray(grp)
        nn_logo = np.array([_nn_dist(x[g != g[i]], x[i : i + 1])[0] for i in range(len(x))])
        nn_t = percentile(nn_seen, Q)
        nn_fpr, nn_tpr = float(np.mean(nn_seen > nn_t)), float(np.mean(nn_logo > nn_t))

        tids = [d for d in test_ids if test_type[d] == t]
        n = len(tids)
        entry: dict[str, object] = {
            "n_ref": len(ids),
            "distinct": len(np.unique(x, axis=0)),
            "distinct_base": len(
                np.unique(np.vstack([SIGNATURES["base"](pages[d]) for d in ids]), axis=0)
            ),
            "k": fit.k,
            "cal": cal,
            "n_test": n,
            "nn_t": nn_t,
            "nn_fpr": nn_fpr,
            "nn_tpr": nn_tpr,
            "seen_med": percentile(seen, 50),
            "logo_med": percentile(logo, 50),
        }
        if n:
            tx = np.vstack([tx_all[d] for d in tids])
            cid, dist = fit.distance(tx)
            flag = dist > cal.threshold
            u = int(flag.sum())
            entry.update(
                unseen=u, share=u / n, ci=wilson_ci(u, n), corr=corrected_share(u / n, cal)
            )
            nn_u = int((_nn_dist(x, tx) > nn_t).sum())
            entry.update(
                nn_unseen=nn_u,
                nn_ci=wilson_ci(nn_u, n),
                nn_corr=corrected_share(nn_u / n, type(cal)(nn_t, Q, nn_fpr, nn_tpr)),
            )
            # method 2a: joint KMeans (k by silhouette), test docs in clusters with no ref doc
            joint = np.vstack([x, tx])
            best_s: tuple[float, np.ndarray] | None = None
            for k in [k for k in K_RANGE if k < len(joint)]:
                lab = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(joint)
                s = float(silhouette_score(joint, lab))
                if best_s is None or s > best_s[0]:
                    best_s = (s, lab)
            assert best_s is not None
            jl = best_s[1]
            ref_cl, test_cl = set(jl[: len(x)]), jl[len(x) :]
            jo = int(sum(c not in ref_cl for c in test_cl))
            entry.update(
                joint_k=len(set(jl)),
                joint_unseen=jo,
                joint_ci=wilson_ci(jo, n),
                joint_only_clusters=len(set(test_cl) - ref_cl),
            )
            rows += [
                AssignmentRow(d, t, int(c), float(di), not bool(f))
                for d, c, di, f in zip(tids, cid, dist, flag, strict=True)
            ]
        sec[t] = entry
    rows.sort(key=lambda r: r.doc_id)
    write_assignments(rows, OUT_CSV)

    # ---- provenance -----------------------------------------------------------------------
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--",
                "src/shipdoc/cluster.py",
                "src/shipdoc/layout_sig.py",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    sums = paths.ocr_cache_dir() / "SHA256SUMS"
    fp_ = hashlib.sha256(sums.read_bytes()).hexdigest()[:16]
    manifest = json.loads(
        (paths.ocr_cache_dir() / ocr.DEFAULT_ENGINE / "manifest.json").read_text()
    )

    def corr_ci(e: dict[str, object]) -> str:
        """Rogan-Gladen range obtained by pushing the Wilson bounds through the correction."""
        lo, hi = (corrected_share(v, e["cal"]) for v in e["ci"])
        return f"[{lo:.1%}, {hi:.1%}]"

    def share_line(t: str) -> str:
        e = sec[t]
        if not e["n_test"]:
            return f"| {t} | 0 | - | - | - | - |"
        ci = e["ci"]
        corr = e["corr"]
        return (
            f"| {t} | {e['n_test']} | {e['unseen']} ({e['share']:.1%}) {_fmt_ci(*ci)}"
            f" | {e['nn_unseen']} ({e['nn_unseen'] / e['n_test']:.1%}) | "
            f"{e['joint_unseen']} ({e['joint_unseen'] / e['n_test']:.1%}), "
            f"{e['joint_only_clusters']} test-only of k={e['joint_k']} | "
            f"{'n/a' if corr is None else f'{corr:.1%} ' + corr_ci(e)} |"
        )

    inv, wb = sec["invoice"], sec["waybill"]
    ncal = {t: sec[t]["cal"] for t in sec}
    lines += [
        "# Test-set layout clusters and unseen-layout share (Phase 6.4)",
        "",
        "**All numbers UNVERIFIED** (computed by `scripts/layout_test_clusters.py`, seed 42; a "
        "verifier must recompute). Aggregates only: no OCR text and no test labels were used or "
        "created; per-doc assignments are kept outside the repo.",
        "",
        "## Provenance",
        "",
        "- Command: `uv run python scripts/layout_test_clusters.py`",
        f"- Git HEAD: `{head}` (cluster.py/layout_sig.py uncommitted at run time: {dirty})",
        f"- OCR cache: `{ocr.DEFAULT_ENGINE}`, SHA256SUMS sha256 prefix `{fp_}`, manifest "
        f"pages_cached {manifest['pages_cached']}; test cache has {n_test_pages} pages / "
        f"{len(test_ids)} docs",
        "- Signature: page 1 only, `shipdoc.cluster.layout_signature` (unchanged) or a variant "
        "from `shipdoc.layout_sig`",
        "",
        "## 1. Signature validation on train+dev (ARI vs supplier_group, page 1)",
        "",
        "Selection criterion fixed in advance: highest ARI vs `supplier_group` on train+dev "
        f"invoices; a variant replaces `base` only if it beats it by more than {TIE}. Three "
        "signatures were tried (base + 2 variants) and all are listed, so the winner's ARI is "
        "mildly optimistic (multiple-comparison). CI = 95% reverse-percentile bootstrap over docs "
        "(seed 42, 1000 resamples, clusters held fixed: does not include refit variance). "
        "k by silhouette over 8..30, label-free.",
        "",
        "| signature | doc type | docs | k | silhouette | ARI [95% CI] | purity |",
        "|---|---|---|---|---|---|---|",
        *var_rows,
        "",
        f"Chosen: **{chosen}** (invoice ARI {results[chosen]['invoice']['ari']:.3f} vs base "
        f"{base_ari:.3f}). Earlier recorded diagnostic (reports/ablations.md R1 detail): base, "
        "ARI 0.346, purity 0.645 on the 400 train+dev invoice pages; this run's base row is the "
        "same fit if the numbers match. Waybill groups are carrier brand, which recon.md found "
        "does not track layout, so waybill ARI is reported, not interpreted as layout quality.",
        "",
        "Variants: `coarse` = positions snapped to a 0.1 grid; `geom` = base + 6 fixed-constant "
        "page-geometry features (aspect, log line count, 5th pct left edge, 95th pct right edge, "
        "top, bottom of the text block).",
        "",
        "## 2. Test doc type (label-free)",
        "",
        "Rule: page-1 signature lacks the 'invoice' keyword family => waybill. A first try with "
        "the 'awb' family scored accuracy 0.440 on train+dev (invoices mention AWBs) and was "
        "dropped; a depth-1 tree over the 20 presence flags found 'invoice' (5-fold CV accuracy "
        "1.0). On train+dev vs gold "
        f"doc_type: TP {tp}, FP {fp}, FN {fn_}, TN {tn} (accuracy "
        f"{(tp + tn) / len(corpus):.3f}, n={len(corpus)}). The pipeline has no label-free "
        "doc-type convention other than the model's own prediction, so this rule is used.",
        "",
        f"Test split ({len(test_ids)} docs): {inv['n_test']} invoice-like, {wb['n_test']} "
        "waybill-like by the rule.",
        "",
        "## 3. Unseen-layout share of the test split",
        "",
        f"Reference = train+dev docs of the same doc type ({inv['n_ref']} invoice pages, "
        f"{wb['n_ref']} waybill pages), signature `{chosen}`. Threshold = {Q:.0f}th percentile "
        "of out-of-sample seen distances (5-fold within train+dev), fixed before test was "
        "assigned.",
        "",
        "| doc type | k | threshold | seen-dist median | LOGO-dist median | FPR (seen flagged) "
        "| TPR (held-out group flagged) |",
        "|---|---|---|---|---|---|---|",
        *[
            f"| {t} | {sec[t]['k']} | {ncal[t].threshold:.3f} | {sec[t]['seen_med']:.3f} | "
            f"{sec[t]['logo_med']:.3f} | {ncal[t].fpr:.3f} | {ncal[t].tpr:.3f} |"
            for t in ("invoice", "waybill")
        ],
        "",
        "| doc type | test docs | primary: centre distance > threshold, count (share) [Wilson "
        "95%] | check 1: 1-NN distance > its own 95th pct | check 2: joint KMeans, docs in "
        "test-only clusters | primary corrected for FPR/TPR (Rogan-Gladen) |",
        "|---|---|---|---|---|---|",
        share_line("invoice"),
        share_line("waybill"),
        "",
        f"1-NN check thresholds: invoice {inv['nn_t']:.3f} (FPR {inv['nn_fpr']:.3f}, TPR "
        f"{inv['nn_tpr']:.3f}), waybill {wb['nn_t']:.3f} (FPR {wb['nn_fpr']:.3f}, TPR "
        f"{wb['nn_tpr']:.3f}).",
        "",
        f"Distinct reference signatures ({chosen}): invoice {inv['distinct']}/{inv['n_ref']}, "
        f"waybill {wb['distinct']}/{wb['n_ref']}; under the keyword-only `base` signature: "
        f"invoice {inv['distinct_base']}/{inv['n_ref']}, "
        f"waybill {wb['distinct_base']}/{wb['n_ref']}. Waybill page-1 signatures are "
        "near-constant (under `base` almost none of the 20 keyword families vary), so waybill "
        "clusters, distances and any 'unseen' count are degenerate and not evidence about layout.",
        "",
        "## 4. Comparison with the brief's ~50% and caveats",
        "",
        "COMPARISON_PLACEHOLDER",
        "",
    ]
    text = "\n".join(lines)
    marker = "## 4. Comparison"
    if OUT_MD.is_file():  # keep the hand-written section 4 across re-runs
        prev = OUT_MD.read_text(encoding="utf-8").split(marker, 1)
        if len(prev) == 2 and "COMPARISON_PLACEHOLDER" not in prev[1]:
            text = text.split(marker, 1)[0] + marker + prev[1]
    OUT_MD.write_text(text, encoding="utf-8")
    print(f"wrote {OUT_MD} and {OUT_CSV}; {len(rows)} assignments")
    for t in sec:
        e = sec[t]
        print(t, {k: v for k, v in e.items() if k != "cal"}, ncal[t])
    return 0


if __name__ == "__main__":
    sys.exit(main())
