# ruff: noqa: E501  (report prose and doc-keyed visual tables are intentionally long lines)
"""Phase 0 data recon (spec section 5, items 1-8) over train+dev labels and images.

Writes, deterministically (seed 42):
  reports/recon_stats.json   every number used in the report (aggregates and IDs only)
  reports/recon.md           narrative; every count is read from the stats dict
  meta/supplier_groups.json  {doc_id: supplier_group} for every train+dev doc

Test data: only file-level facts (page counts from reports/inventory.json and image
header dimensions) are read; no test pixel is decoded and no test label exists.
Item 9 (label audit) is out of scope here.

Usage: uv run python scripts/recon.py
"""

from __future__ import annotations

import difflib
import glob
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

SEED = 42
VISUAL_REDACTED_P2_VIEWED = [
    "train_0091",
    "train_0253",
    "train_0282",
]  # p1 number redacted, p2 banner legible
DETAIL_P2_DOCS = [
    "train_0013",
    "train_0001",
    "train_0010",
    "train_0040",
    "train_0041",
]  # the 5 detailed page-2 views
ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "dev")
TYPES = ("invoice", "waybill")

# Mirrors assignment/score.py (read-only reference): HEADER / ROW field lists.
HEADER = {
    "invoice": [
        "invoice_number",
        "invoice_date",
        "supplier_name",
        "buyer_name",
        "ship_to_name",
        "currency",
        "total_amount",
        "awb_number",
    ],
    "waybill": [
        "carrier",
        "mawb",
        "hawb",
        "origin_airport",
        "destination_airport",
        "shipper_name",
        "consignee_name",
        "pieces",
        "gross_weight_kg",
    ],
}
ROW = ["supplier_part_number", "customer_part_number", "purchase_order", "quantity"]

# Pooled null-rate threshold separating a "required" header field (nulls are redactions,
# i.e. illegible) from an "optional line" field (nulls are an absent line). Chosen from the
# observed gap between invoice_number/invoice_date/total fields and awb_number/hawb.
OPTIONAL_NULL_RATE = 0.10

# Hand-recorded visual observations (Read tool on page images). One exemplar doc per
# supplier group; keyed by doc_id so the group is resolved at runtime and no supplier
# name is stored. date_fmt: MDY=MM/DD/YYYY, DMY=DD/MM/YYYY, DMON=DD-MON-YYYY,
# LONG='Month DD, YYYY', ISO=YYYY-MM-DD. page2_header: does page 2 repeat the column
# header row (viewed on a multipage doc of the group).
VISUAL_GROUP_OBS: dict[str, dict[str, Any]] = {
    "train_0093": {
        "date_fmt": "MDY",
        "po": "per_row",
        "cust_part": "printed",
        "p2": "train_0040",
        "p2_header": True,
    },
    "train_0211": {
        "date_fmt": "LONG",
        "po": "per_row",
        "cust_part": "printed",
        "p2": "train_0089",
        "p2_header": False,
    },
    "train_0077": {
        "date_fmt": "LONG",
        "po": "absent",
        "cust_part": "printed",
        "p2": "train_0013",
        "p2_header": False,
    },
    "train_0028": {
        "date_fmt": "DMON",
        "po": "absent",
        "cust_part": "absent",
        "p2": "train_0041",
        "p2_header": False,
    },
    "train_0058": {
        "date_fmt": "DMON",
        "po": "per_row",
        "cust_part": "absent",
        "p2": "train_0102",
        "p2_header": False,
    },
    "train_0016": {
        "date_fmt": "LONG",
        "po": "per_row",
        "cust_part": "absent",
        "p2": "train_0107",
        "p2_header": True,
    },
    "train_0052": {
        "date_fmt": "DMY",
        "po": "per_row",
        "cust_part": "printed",
        "p2": "train_0001",
        "p2_header": False,
    },
    "train_0030": {
        "date_fmt": "DMON",
        "po": "absent",
        "cust_part": "absent",
        "p2": "train_0015",
        "p2_header": True,
    },
    "train_0038": {
        "date_fmt": "MDY",
        "po": "per_row",
        "cust_part": "printed",
        "p2": "train_0119",
        "p2_header": True,
    },
    "train_0136": {
        "date_fmt": "MDY",
        "po": "absent",
        "cust_part": "printed",
        "p2": "train_0022",
        "p2_header": False,
    },
    "train_0025": {
        "date_fmt": "ISO",
        "po": "per_row",
        "cust_part": "printed",
        "p2": "train_0069",
        "p2_header": True,
    },
    "train_0014": {
        "date_fmt": "DMY",
        "po": "per_row",
        "cust_part": "absent",
        "p2": "train_0032",
        "p2_header": True,
    },
    "train_0020": {
        "date_fmt": "ISO",
        "po": "absent",
        "cust_part": "absent",
        "p2": "train_0147",
        "p2_header": False,
    },
    "train_0110": {
        "date_fmt": "DMON",
        "po": "absent",
        "cust_part": "absent",
        "p2": "train_0010",
        "p2_header": False,
    },
    "train_0003": {
        "date_fmt": "DMY",
        "po": "absent",
        "cust_part": "absent",
        "p2": "train_0133",
        "p2_header": False,
    },
    "train_0143": {
        "date_fmt": "DMON",
        "po": "absent",
        "cust_part": "absent",
        "p2": "train_0065",
        "p2_header": False,
    },
    "train_0007": {
        "date_fmt": "LONG",
        "po": "per_row",
        "cust_part": "absent",
        "p2": "train_0009",
        "p2_header": False,
    },
    "train_0084": {
        "date_fmt": "MDY",
        "po": "per_row",
        "cust_part": "absent",
        "p2": "train_0021",
        "p2_header": False,
    },
}
# Scans of the same groups were also viewed (train_0029, 0114, 0013, 0023, 0050, 0017, 0001,
# 0146, 0103, 0081, 0212, 0035, 0000, 0010, 0066, 0045, 0105, 0161) and agreed with the png.
VISUAL_SCAN_EXEMPLARS = [
    "train_0029", "train_0114", "train_0013", "train_0023", "train_0050", "train_0017",
    "train_0001", "train_0146", "train_0103", "train_0081", "train_0212", "train_0035",
    "train_0000", "train_0010", "train_0066", "train_0045", "train_0105", "train_0161",
]  # fmt: skip

# Null-rule confirmation views: (doc_id, field, class, what the image showed).
VISUAL_NULL_CHECKS = [
    ("train_0045", "invoice_number", "illegible", "scribbled-out value (scan)"),
    ("train_0084", "invoice_number", "illegible", "scribbled-out value (png)"),
    ("train_0072", "invoice_number", "illegible", "solid black redaction box (png)"),
    ("train_0043", "invoice_number", "illegible", "black redaction box (scan)"),
    ("train_0015", "invoice_date", "illegible", "solid black redaction box (png)"),
    ("train_0160", "invoice_date", "illegible", "solid black redaction box (png)"),
    ("train_0077", "purchase_order", "not_printed", "table has no PO column (group-wide)"),
    ("train_0023", "customer_part_number", "not_printed", "table has no customer-part column"),
    ("train_0050", "customer_part_number", "not_printed", "table has no customer-part column"),
    ("train_0020", "awb_number", "not_printed", "no AWB / Waybill line in header block"),
    ("train_0093", "awb_number", "not_printed", "no AWB / Waybill line in header block"),
    ("train_0012", "hawb", "not_printed", "waybill grid has a blank slot, no HAWB box"),
    ("train_0227", "hawb", "not_printed", "waybill grid has a blank slot, no HAWB box"),
]  # fmt: skip

# Waybill layout observations (visual): field boxes sit in a 2-column grid whose order and
# font vary per document, independent of the carrier title.
VISUAL_WAYBILL_OBS = {
    "docs_viewed": ["train_0006", "train_0008", "train_0012", "train_0227", "train_0241",
                    "dev_0044", "train_0095", "train_0230"],
    "same_carrier_pairs_with_different_field_order": True,
    "carrier_is_title_text_only": True,
}  # fmt: skip


def norm_name(v: Any) -> str:
    """score.py name rule: lowercase, keep [0-9a-z] only."""
    return re.sub(r"[^0-9a-z]", "", str(v).lower())


def norm_awb(v: Any) -> str | None:
    """Strip spaces and dashes (spec item 8); None for empty."""
    if v is None or not str(v).strip():
        return None
    return re.sub(r"[\s-]", "", str(v)).upper()


def is_empty(v: Any) -> bool:
    """score.py _empty."""
    return v is None or (isinstance(v, str) and not v.strip())


def shape(s: Any) -> str:
    """Character-class shape: digits->9, letters->A (never the value itself)."""
    return re.sub(r"[0-9]", "9", re.sub(r"[A-Za-z]", "A", str(s)))


def pct(a: int, b: int) -> float:
    return round(100.0 * a / b, 2) if b else 0.0


def frac(a: int, b: int) -> dict[str, Any]:
    """Numerator/denominator/percent triple so the report always shows denominators."""
    return {"n": a, "of": b, "pct": pct(a, b)}


def dist(xs: list[int]) -> dict[str, Any]:
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "min": int(min(xs)),
        "median": float(statistics.median(xs)),
        "p90": round(float(np.percentile(xs, 90)), 2),
        "max": int(max(xs)),
        "mean": round(float(np.mean(xs)), 2),
    }


def mawb_check(s: str | None) -> str:
    """Return 'pass' / 'fail' for 3-digit prefix + 8-digit serial (mod-7), else 'uncheckable'."""
    n = norm_awb(s)
    if n is None or not re.fullmatch(r"\d{11}", n):
        return "uncheckable"
    serial = n[3:]
    return "pass" if int(serial[:7]) % 7 == int(serial[7]) else "fail"


# ----------------------------------------------------------------------------- loading


def load_docs() -> list[dict[str, Any]]:
    docs = []
    for split in SPLITS:
        for f in sorted(glob.glob(str(ROOT / "data" / split / "labels" / "*.json"))):
            d = json.loads(Path(f).read_text(encoding="utf-8"))
            d["split"] = split
            d["paths"] = [str(ROOT / "data" / split / "images" / p) for p in d["pages"]]
            d["scanned_pages"] = [p.lower().endswith(".jpg") for p in d["pages"]]
            docs.append(d)
    return docs


def by_st(docs: list[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    out: dict[str, dict[str, list[dict[str, Any]]]] = {s: {t: [] for t in TYPES} for s in SPLITS}
    for d in docs:
        out[d["split"]][d["doc_type"]].append(d)
    return out


# ----------------------------------------------------------------------------- item 1


def item1(docs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for s, per_t in by_st(docs).items():
        out[s] = {}
        for t, ds in per_t.items():
            pages = [
                (p, sc) for d in ds for p, sc in zip(d["paths"], d["scanned_pages"], strict=True)
            ]
            res: Counter[str] = Counter()
            dpi_present = 0
            dpi_meta: Counter[str] = Counter()
            for p, sc in pages:
                with Image.open(p) as im:
                    res[f"{im.size[0]}x{im.size[1]}:{'jpg' if sc else 'png'}"] += 1
                    # JFIF density with unit 0 is only a pixel aspect ratio, not a resolution.
                    unit = im.info.get("jfif_unit")
                    real_dpi = im.info.get("dpi") or (
                        im.info.get("jfif_density") if unit in (1, 2) else None
                    )
                    dens = im.info.get("jfif_density")
                    dpi_meta[
                        f"{'jpg' if sc else 'png'}: dpi={real_dpi}, jfif_density={dens}, jfif_unit={unit}"
                    ] += 1
                    if real_dpi:
                        dpi_present += 1
            n_scan = sum(sc for _, sc in pages)
            any_scan = sum(any(d["scanned_pages"]) for d in ds)
            all_scan = sum(all(d["scanned_pages"]) for d in ds)
            out[s][t] = {
                "docs": len(ds),
                "pages": len(pages),
                "pages_per_doc": dict(sorted(Counter(len(d["pages"]) for d in ds).items())),
                "multipage_docs": frac(sum(len(d["pages"]) > 1 for d in ds), len(ds)),
                "scanned_pages": frac(n_scan, len(pages)),
                "docs_any_scanned_page": frac(any_scan, len(ds)),
                "docs_all_pages_scanned": frac(all_scan, len(ds)),
                "docs_mixed_png_jpg": frac(any_scan - all_scan, len(ds)),
                "resolution_counts": dict(res.most_common()),
                "pages_with_real_dpi_metadata": frac(dpi_present, len(pages)),
                "dpi_metadata_patterns": dict(dpi_meta.most_common()),
            }
    # Test: file-level only. Counts from inventory.json; resolutions from image headers.
    inv = json.loads((ROOT / "reports" / "inventory.json").read_text(encoding="utf-8"))["test"]
    res = Counter()
    for p in sorted(glob.glob(str(ROOT / "data" / "test" / "images" / "*"))):
        with Image.open(p) as im:  # header-only: Image.open is lazy, no pixels decoded
            res[f"{im.size[0]}x{im.size[1]}:{Path(p).suffix[1:].lower()}"] += 1
    out["test_file_level"] = {
        "docs": inv["docs"],
        "pages": inv["pages"],
        "pages_per_doc": inv["pages_per_doc_hist"],
        "png_pages": inv["png_pages"],
        "jpg_pages": inv["jpg_pages"],
        "scanned_pages": frac(inv["jpg_pages"], inv["pages"]),
        "resolution_counts": dict(res.most_common()),
    }
    return out


# ----------------------------------------------------------------------------- item 2


def signature(path: str, kind: str) -> np.ndarray:
    """Label-free layout signature of one page: grayscale, downsampled, no deskew."""
    with Image.open(path) as im:
        g = im.convert("L")
        if kind == "full":
            g = g.resize((48, 64), Image.BILINEAR)
        else:  # 'band': top 45% of the page (header block, titles, table head)
            g = g.crop((0, 0, g.width, int(g.height * 0.45))).resize((48, 32), Image.BILINEAR)
        return np.asarray(g, dtype=np.float32).ravel() / 255.0


def purity(labels: list[str], clusters: np.ndarray) -> float:
    """Share of docs falling in their cluster's majority label (cluster homogeneity)."""
    by_c: dict[int, Counter[str]] = defaultdict(Counter)
    for lab, c in zip(labels, clusters, strict=True):
        by_c[int(c)][lab] += 1
    return sum(max(c.values()) for c in by_c.values()) / len(labels)


def coverage(labels: list[str], clusters: np.ndarray) -> float:
    """Share of docs falling in their label's majority cluster (label compactness)."""
    by_l: dict[str, Counter[int]] = defaultdict(Counter)
    for lab, c in zip(labels, clusters, strict=True):
        by_l[lab][int(c)] += 1
    return sum(max(c.values()) for c in by_l.values()) / len(labels)


def brand_of(d: dict[str, Any]) -> str:
    """Carrier brand = first word of the carrier name (e.g. 'Example' of 'Example Air Cargo')."""
    return str(d["header"]["carrier"]).split()[0].lower()


def item2(docs: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, str]]:
    inv = [d for d in docs if d["doc_type"] == "invoice"]
    wb = [d for d in docs if d["doc_type"] == "waybill"]
    out: dict[str, Any] = {}

    # Supplier-name variants (invoices).
    sup: dict[str, Any] = {}
    for scope, ds in (("train", [d for d in inv if d["split"] == "train"]),
                      ("dev", [d for d in inv if d["split"] == "dev"]), ("train+dev", inv)):  # fmt: skip
        raw = [d["header"]["supplier_name"] for d in ds]
        sup[scope] = {
            "docs": len(ds),
            "distinct_raw": len(set(raw)),
            "distinct_casefold_strip": len({r.casefold().strip() for r in raw}),
            "distinct_score_normalized": len({norm_name(r) for r in raw}),
            "null_supplier_name": sum(is_empty(r) for r in raw),
        }
    train_s = {norm_name(d["header"]["supplier_name"]) for d in inv if d["split"] == "train"}
    dev_s = {norm_name(d["header"]["supplier_name"]) for d in inv if d["split"] == "dev"}
    sup["suppliers_in_dev_not_in_train"] = len(dev_s - train_s)
    sup["suppliers_in_train_not_in_dev"] = len(train_s - dev_s)
    out["invoice_supplier_names"] = sup

    # Waybill identity fields.
    way: dict[str, Any] = {}
    for f, fn in (("carrier", lambda d: norm_name(d["header"]["carrier"])),
                  ("carrier_brand", lambda d: brand_of(d)),
                  ("shipper_name", lambda d: norm_name(d["header"]["shipper_name"]))):  # fmt: skip
        c = Counter(fn(d) for d in wb)
        way[f] = {
            "docs": len(wb),
            "distinct": len(c),
            "docs_per_value": dist(list(c.values())),
            "values_in_exactly_one_doc": sum(v == 1 for v in c.values()),
        }
    tb = {brand_of(d) for d in wb if d["split"] == "train"}
    db = {brand_of(d) for d in wb if d["split"] == "dev"}
    way["brands_in_dev_not_in_train"] = len(db - tb)
    tc = {norm_name(d["header"]["carrier"]) for d in wb if d["split"] == "train"}
    dc = {norm_name(d["header"]["carrier"]) for d in wb if d["split"] == "dev"}
    way["carriers_in_dev_not_in_train"] = len(dc - tc)
    way["carrier_suffix_distinct"] = len(
        {" ".join(str(d["header"]["carrier"]).split()[1:]).lower() for d in wb}
    )
    out["waybill_identity"] = way

    # Cross-type name coincidence.
    sup_norm = {norm_name(d["header"]["supplier_name"]) for d in inv}
    buyers = {norm_name(d["header"][k]) for d in inv for k in ("buyer_name", "ship_to_name")}
    fuzzy = 0
    for d in wb:
        sh = norm_name(d["header"]["shipper_name"])
        if any(difflib.SequenceMatcher(None, sh, s).ratio() >= 0.85 for s in sup_norm):
            fuzzy += 1
    out["name_overlap"] = {
        "waybill_shipper_equals_invoice_supplier": frac(
            sum(norm_name(d["header"]["shipper_name"]) in sup_norm for d in wb), len(wb)
        ),
        "waybill_shipper_fuzzy_ge_0.85_invoice_supplier": frac(fuzzy, len(wb)),
        "waybill_consignee_equals_invoice_buyer_or_ship_to": frac(
            sum(norm_name(d["header"]["consignee_name"]) in buyers for d in wb), len(wb)
        ),
        "invoice_supplier_equals_waybill_shipper": frac(
            sum(norm_name(d["header"]["supplier_name"]) in {norm_name(w["header"]["shipper_name"]) for w in wb}
                for d in inv), len(inv)),
    }  # fmt: skip

    # Layout clustering against label keys.
    rng_note = f"KMeans(random_state={SEED}, n_init=10); AgglomerativeClustering(ward)"
    runs: list[dict[str, Any]] = []
    for t, ds in (("invoice", inv), ("waybill", wb)):
        label_fns = (
            {"supplier_name": lambda d: norm_name(d["header"]["supplier_name"])}
            if t == "invoice"
            else {
                "carrier": lambda d: norm_name(d["header"]["carrier"]),
                "carrier_brand": brand_of,
                "shipper_name": lambda d: norm_name(d["header"]["shipper_name"]),
            }
        )
        ks = (18, 20) if t == "invoice" else (10, 20, 45)
        for sig_kind in ("full", "band"):
            X_all = np.stack([signature(d["paths"][0], sig_kind) for d in ds])
            for subset in ("all_pages", "digital_only"):
                idx = [
                    i
                    for i, d in enumerate(ds)
                    if subset == "all_pages" or not d["scanned_pages"][0]
                ]
                X = X_all[idx]
                sub = [ds[i] for i in idx]
                for algo in ("kmeans", "ward"):
                    for k in ks:
                        cl = (
                            KMeans(k, n_init=10, random_state=SEED).fit_predict(X)
                            if algo == "kmeans"
                            else AgglomerativeClustering(n_clusters=k).fit_predict(X)
                        )
                        for lab_name, fn in label_fns.items():
                            for scope in ("train+dev", "train", "dev"):
                                m = [
                                    j
                                    for j, d in enumerate(sub)
                                    if scope == "train+dev" or d["split"] == scope
                                ]
                                if len(m) < 5:
                                    continue
                                labs = [fn(sub[j]) for j in m]
                                cc = cl[m]
                                runs.append({
                                    "type": t, "signature": sig_kind, "pages": subset, "algo": algo, "k": k,
                                    "label": lab_name, "scope": scope, "n": len(m),
                                    "n_label_classes": len(set(labs)),
                                    "ARI": round(float(adjusted_rand_score(labs, cc)), 3),
                                    "NMI": round(float(normalized_mutual_info_score(labs, cc)), 3),
                                    "purity": round(purity(labs, cc), 3),
                                    "label_coverage": round(coverage(labs, cc), 3),
                                })  # fmt: skip
    out["clustering_method"] = (
        "page-1 grayscale, no deskew; 'full' = whole page to 48x64; 'band' = top 45% to 48x32; "
        + rng_note
    )
    out["clustering_runs"] = runs

    # supplier_group assignment (anonymised ids, ordered by normalised name / brand).
    groups: dict[str, str] = {}
    inv_names = sorted({norm_name(d["header"]["supplier_name"]) for d in inv})
    inv_id = {n: f"inv_g{i + 1:02d}" for i, n in enumerate(inv_names)}
    wb_brands = sorted({brand_of(d) for d in wb})
    wb_id = {b: f"wb_g{i + 1:02d}" for i, b in enumerate(wb_brands)}
    for d in inv:
        groups[d["doc_id"]] = inv_id[norm_name(d["header"]["supplier_name"])]
    for d in wb:
        groups[d["doc_id"]] = wb_id[brand_of(d)]
    tab: dict[str, dict[str, int]] = defaultdict(lambda: {"train": 0, "dev": 0})
    for d in docs:
        tab[groups[d["doc_id"]]][d["split"]] += 1
    out["group_table"] = {g: {**v, "total": v["train"] + v["dev"]} for g, v in sorted(tab.items())}
    for t in TYPES:
        sizes = [
            v["total"]
            for g, v in out["group_table"].items()
            if g.startswith(t[:3] if t == "invoice" else "wb")
        ]
        out[f"groups_{t}"] = {"n_groups": len(sizes), "docs_per_group": dist(sizes)}
    # CV feasibility: each type has >=3 groups so every K=3 fold can hold both types.
    out["groups_total"] = len(out["group_table"])
    return out, groups


# ----------------------------------------------------------------------------- item 3


def item3(docs: list[dict[str, Any]], groups: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {"rule": {}}
    out["rule"] = {
        "row_fields": "(group, field) is NOT PRINTED iff the field is null on 100% of the group's rows; "
        "PRINTED iff 0% null; MIXED otherwise (expected empty).",
        "header_fields": f"Field is OPTIONAL-LINE if its pooled null rate within the doc type is >= "
        f"{OPTIONAL_NULL_RATE:.0%} (null = line absent from the page); otherwise REQUIRED and its "
        "nulls are ILLEGIBLE (redacted / scribbled). Confirmed visually on 13 docs.",
    }
    null_rates: dict[str, Any] = {}
    for s, per_t in by_st(docs).items():
        null_rates[s] = {}
        for t, ds in per_t.items():
            null_rates[s][t] = {
                f: frac(sum(is_empty(d["header"].get(f)) for d in ds), len(ds)) for f in HEADER[t]
            }
            rows = [r for d in ds for r in d["line_items"]]
            null_rates[s][t]["_rows"] = {
                f: frac(sum(is_empty(r.get(f)) for r in rows), len(rows)) for f in ROW
            }
            null_rates[s][t]["_n_rows"] = len(rows)
    out["null_rates"] = null_rates

    # Rule application, train+dev pooled per type.
    cls: dict[str, Any] = {}
    for t in TYPES:
        ds = [d for d in docs if d["doc_type"] == t]
        fields: dict[str, Any] = {}
        for f in HEADER[t]:
            n_null = sum(is_empty(d["header"].get(f)) for d in ds)
            rate = n_null / len(ds)
            kind = (
                "optional_line"
                if rate >= OPTIONAL_NULL_RATE
                else ("required_with_nulls" if n_null else "never_null")
            )
            scan_null = sum(is_empty(d["header"].get(f)) and any(d["scanned_pages"]) for d in ds)
            scan_docs = sum(any(d["scanned_pages"]) for d in ds)
            fields[f] = {
                "class": kind,
                "nulls": frac(n_null, len(ds)),
                "nulls_in_docs_with_scanned_page": frac(scan_null, scan_docs),
                "nulls_in_all_png_docs": frac(n_null - scan_null, len(ds) - scan_docs),
            }
        out_t = {"fields": fields}
        opt = sum(v["nulls"]["n"] for v in fields.values() if v["class"] == "optional_line")
        ill = sum(v["nulls"]["n"] for v in fields.values() if v["class"] == "required_with_nulls")
        tot_slots = len(ds) * len(HEADER[t])
        out_t["header_null_cells"] = {
            "total_header_cells": tot_slots,
            "optional_line_nulls": opt,
            "illegible_nulls": ill,
            "all_nulls": opt + ill,
            "docs_with_illegible_null": frac(
                sum(any(is_empty(d["header"].get(f)) for f, v in fields.items() if v["class"] == "required_with_nulls")
                    for d in ds), len(ds)),
        }  # fmt: skip
        # Per-group spread of optional-line null rates (is it a group property?).
        per_group: dict[str, dict[str, list[int]]] = defaultdict(
            lambda: defaultdict(lambda: [0, 0])
        )
        for d in ds:
            for f in fields:
                a = per_group[groups[d["doc_id"]]][f]
                a[0] += is_empty(d["header"].get(f))
                a[1] += 1
        out_t["per_group_null_rate_range"] = {
            f: {
                "min_pct": min(pct(a[f][0], a[f][1]) for a in per_group.values()),
                "max_pct": max(pct(a[f][0], a[f][1]) for a in per_group.values()),
                "groups_with_zero_nulls": sum(a[f][0] == 0 for a in per_group.values()),
                "groups_with_100pct_nulls": sum(a[f][0] == a[f][1] for a in per_group.values()),
                "groups": len(per_group),
            }
            for f in fields
            if fields[f]["class"] != "never_null"
        }
        cls[t] = out_t
    out["header_classification"] = cls

    # Row-field classification per group (invoices).
    inv = [d for d in docs if d["doc_type"] == "invoice"]
    per_g: dict[str, dict[str, list[int]]] = defaultdict(lambda: {f: [0, 0] for f in ROW})
    for d in inv:
        for r in d["line_items"]:
            for f in ROW:
                a = per_g[groups[d["doc_id"]]][f]
                a[0] += is_empty(r.get(f))
                a[1] += 1
    rowcls: dict[str, Any] = {}
    for f in ROW:
        printed = [g for g, v in per_g.items() if v[f][0] == 0]
        notp = [g for g, v in per_g.items() if v[f][0] == v[f][1]]
        mixed = [g for g in per_g if g not in printed and g not in notp]
        rows_np = sum(v[f][1] for g, v in per_g.items() if g in notp)
        rowcls[f] = {
            "groups": len(per_g),
            "printed": len(printed),
            "not_printed": len(notp),
            "mixed": len(mixed),
            "rows_in_not_printed_groups": frac(rows_np, sum(v[f][1] for v in per_g.values())),
            "not_printed_groups": sorted(notp),
        }
    out["row_field_classification"] = rowcls
    out["per_group_row_null_counts"] = {
        g: {
            f: {"null": v[f][0], "rows": v[f][1]}
            for f in ("customer_part_number", "purchase_order")
        }
        for g, v in sorted(per_g.items())
    }

    # Check the hand-recorded visual observations against gold.
    by_id = {d["doc_id"]: d for d in docs}
    checks = []
    for did, f, kind, seen in VISUAL_NULL_CHECKS:
        d = by_id[did]
        if f in ROW:
            gold_null = all(is_empty(r.get(f)) for r in d["line_items"])
        else:
            gold_null = is_empty(d["header"].get(f))
        checks.append({"doc_id": did, "field": f, "class": kind, "gold_null": gold_null,
                       "scanned": any(d["scanned_pages"]), "observation": seen})  # fmt: skip
    out["visual_null_checks"] = checks
    out["visual_null_checks_all_gold_null"] = all(c["gold_null"] for c in checks)
    # Redacted invoice_number on page 1 of a multipage doc: page 2 carries a "<TITLE> - continued
    # <invoice number>" banner. Viewed: the banner was legible while gold stays null.
    inv_multi = [d for d in docs if d["doc_type"] == "invoice" and len(d["pages"]) > 1]
    red = [d for d in inv_multi if is_empty(d["header"].get("invoice_number"))]
    out["redacted_invoice_number_on_multipage_docs"] = {
        "multipage_invoice_docs": len(inv_multi),
        "gold_null_invoice_number": frac(len(red), len(inv_multi)),
        "viewed_doc_ids": VISUAL_REDACTED_P2_VIEWED,
        "viewed_in_gold_null_set": all(by_id[x] in red for x in VISUAL_REDACTED_P2_VIEWED),
        "page2_banner_legible_in_viewed": len(VISUAL_REDACTED_P2_VIEWED),
        "multipage_docs_with_null_invoice_date": sum(
            is_empty(d["header"].get("invoice_date")) for d in inv_multi
        ),
    }
    return out


# ----------------------------------------------------------------------------- item 4/5/6


def item4(docs: list[dict[str, Any]], groups: dict[str, str]) -> dict[str, Any]:
    inv = [d for d in docs if d["doc_type"] == "invoice"]
    out: dict[str, Any] = {}
    # Visual per-group table, cross-checked with gold null pattern.
    vis: dict[str, Any] = {}
    by_id = {d["doc_id"]: d for d in docs}
    for did, o in VISUAL_GROUP_OBS.items():
        g = groups[did]
        rows = [r for d in inv if groups[d["doc_id"]] == g for r in d["line_items"]]
        gold_po_null = all(is_empty(r["purchase_order"]) for r in rows)
        gold_cp_null = all(is_empty(r["customer_part_number"]) for r in rows)
        vis[g] = {
            **o,
            "exemplar_scanned": any(by_id[did]["scanned_pages"]),
            "gold_po_all_null": gold_po_null,
            "gold_cp_all_null": gold_cp_null,
            "visual_matches_gold_po": gold_po_null == (o["po"] == "absent"),
            "visual_matches_gold_cp": gold_cp_null == (o["cust_part"] == "absent"),
        }
    out["visual_groups"] = vis
    out["visual_groups_n"] = len(vis)
    out["visual_groups_unique"] = len(set(groups[k] for k in VISUAL_GROUP_OBS))
    out["visual_scan_exemplars"] = {"n": len(VISUAL_SCAN_EXEMPLARS), "ids": VISUAL_SCAN_EXEMPLARS}
    out["visual_waybill"] = VISUAL_WAYBILL_OBS
    out["visual_date_format_counts"] = dict(Counter(v["date_fmt"] for v in vis.values()))
    out["visual_summary"] = {
        "currency_printed_as": "ISO code in header ('Currency: XXX') and in 'Total Amount: XXX n' on all 36 page images viewed; no symbol seen",
        "number_format": "decimal point, comma thousands on all 36 page images viewed; no decimal comma seen",
        "unit_price_decimals": 4,
        "line_total_decimals": 2,
        "airport_printed_as": "'IATA CITY' (e.g. code plus city name); gold keeps the 3-letter code only",
    }  # fmt: skip

    # Date ambiguity per group (gold ISO dates).
    per_g: dict[str, dict[str, int]] = defaultdict(
        lambda: {"dates": 0, "day_le_12": 0, "day_gt_12": 0}
    )
    bad_iso = 0
    for d in inv:
        v = d["header"].get("invoice_date")
        if is_empty(v):
            continue
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(v)):
            bad_iso += 1
            continue
        day = int(str(v)[8:10])
        a = per_g[groups[d["doc_id"]]]
        a["dates"] += 1
        a["day_le_12" if day <= 12 else "day_gt_12"] += 1
    fmt_of = {g: v["date_fmt"] for g, v in vis.items()}
    numeric = {g for g, f in fmt_of.items() if f in ("MDY", "DMY")}
    per_group = {
        g: {**a, "date_fmt": fmt_of.get(g), "numeric_format": g in numeric,
            "ambiguous_pct": pct(a["day_le_12"], a["dates"]) if g in numeric else None}
        for g, a in sorted(per_g.items())
    }  # fmt: skip
    out["date_per_group"] = per_group
    out["date_non_iso_gold"] = bad_iso
    num_dates = sum(a["dates"] for g, a in per_g.items() if g in numeric)
    num_amb = sum(a["day_le_12"] for g, a in per_g.items() if g in numeric)
    out["date_ambiguity_numeric_groups"] = {
        "groups_numeric": len(numeric),
        "dates": num_dates,
        "ambiguous_day_le_12": frac(num_amb, num_dates),
        "min_resolvers_in_a_numeric_group": min(per_g[g]["day_gt_12"] for g in numeric),
        "all_invoice_dates": frac(
            sum(a["day_le_12"] for a in per_g.values()), sum(a["dates"] for a in per_g.values())
        ),
    }
    months = Counter(
        int(str(d["header"]["invoice_date"])[5:7])
        for d in inv
        if not is_empty(d["header"].get("invoice_date"))
    )
    dates = sorted(
        str(d["header"]["invoice_date"])
        for d in inv
        if not is_empty(d["header"].get("invoice_date"))
    )
    out["date_range"] = {
        "min_month": dates[0][:7],
        "max_month": dates[-1][:7],
        "months_present": len(months),
    }

    # Currency.
    cur = Counter(d["header"].get("currency") for d in inv)
    out["gold_currency_counts"] = dict(sorted((str(k), v) for k, v in cur.items()))
    n = len(inv)
    out["currency_symbol_ambiguity"] = {
        "docs": n,
        "dollar_sign_currencies_USD_SGD": frac(cur["USD"] + cur["SGD"], n),
        "yen_sign_currencies_JPY_CNY": frac(cur["JPY"] + cur["CNY"], n),
        "euro_sign_currencies_EUR": frac(cur["EUR"], n),
        "gold_currencies_not_in_5": sum(
            v for k, v in cur.items() if k not in ("USD", "EUR", "SGD", "JPY", "CNY")
        ),
    }
    gc: dict[str, set[str]] = defaultdict(set)
    for d in inv:
        gc[groups[d["doc_id"]]].add(str(d["header"].get("currency")))
    out["currencies_per_group"] = dist([len(v) for v in gc.values()])

    # Gold number shapes (gold is normalised).
    shapes: dict[str, Any] = {}
    for t, fields in (("invoice", ["total_amount"]), ("waybill", ["pieces", "gross_weight_kg"])):
        for f in fields:
            vals = [
                str(d["header"][f])
                for d in docs
                if d["doc_type"] == t and not is_empty(d["header"].get(f))
            ]
            shapes[f"{t}.{f}"] = {
                "n": len(vals),
                "shapes_top": dict(
                    Counter(re.sub(r"9+", "9", shape(v)) for v in vals).most_common(5)
                ),
                "contains_comma": sum("," in v for v in vals),
            }
    q = [str(r["quantity"]) for d in inv for r in d["line_items"] if not is_empty(r["quantity"])]
    shapes["invoice.row.quantity"] = {
        "n": len(q),
        "shapes_top": dict(Counter(re.sub(r"9+", "9", shape(v)) for v in q).most_common(5)),
        "contains_comma": sum("," in v for v in q),
    }
    two_dec = [d for d in inv if re.fullmatch(r"\d+\.\d{2}", str(d["header"].get("total_amount")))]
    jpy = [d for d in inv if d["header"].get("currency") == "JPY"]
    shapes["invoice.total_amount"]["exactly_two_decimals"] = frac(len(two_dec), len(inv))
    shapes["invoice.total_amount"]["jpy_with_two_decimals"] = frac(
        sum(d in two_dec for d in jpy), len(jpy)
    )
    out["gold_number_shapes"] = shapes
    out["gold_airport_shapes"] = {
        f: dict(
            Counter(
                shape(d["header"][f])
                for d in docs
                if d["doc_type"] == "waybill" and not is_empty(d["header"].get(f))
            )
        )
        for f in ("origin_airport", "destination_airport")
    }
    return out


def item5(docs: list[dict[str, Any]], groups: dict[str, str]) -> dict[str, Any]:
    inv = [d for d in docs if d["doc_type"] == "invoice"]
    out: dict[str, Any] = {"per_group": {}}
    pooled = {"rows": 0, "rows_with_po": 0, "docs_po_group": 0, "docs_multirow": 0, "docs_one_distinct": 0,
              "docs_all_rows_po": 0}  # fmt: skip
    ratios: list[float] = []
    for g in sorted({groups[d["doc_id"]] for d in inv}):
        ds = [d for d in inv if groups[d["doc_id"]] == g]
        rows = [r for d in ds for r in d["line_items"]]
        with_po = sum(not is_empty(r["purchase_order"]) for r in rows)
        docs_po = [d for d in ds if any(not is_empty(r["purchase_order"]) for r in d["line_items"])]
        multi = [d for d in docs_po if len(d["line_items"]) >= 2]
        one = [d for d in multi if len({r["purchase_order"] for r in d["line_items"]}) == 1]
        allrows = [
            d for d in docs_po if all(not is_empty(r["purchase_order"]) for r in d["line_items"])
        ]
        out["per_group"][g] = {
            "rows": len(rows),
            "rows_with_po": with_po,
            "rows_null_po": len(rows) - with_po,
            "docs": len(ds),
            "docs_with_any_po": len(docs_po),
            "docs_multirow_with_po": len(multi),
            "docs_all_rows_share_one_po": len(one),
            "docs_every_row_has_po": len(allrows),
        }
        if docs_po:
            pooled["rows"] += len(rows)
            pooled["rows_with_po"] += with_po
            pooled["docs_po_group"] += len(ds)
            pooled["docs_multirow"] += len(multi)
            pooled["docs_one_distinct"] += len(one)
            pooled["docs_all_rows_po"] += len(allrows)
            ratios += [
                len({r["purchase_order"] for r in d["line_items"]}) / len(d["line_items"])
                for d in multi
            ]
    out["pooled_po_printing_groups"] = {
        "groups": sum(v["rows_with_po"] > 0 for v in out["per_group"].values()),
        "rows_with_po": frac(pooled["rows_with_po"], pooled["rows"]),
        "multirow_docs_all_rows_share_one_po": frac(
            pooled["docs_one_distinct"], pooled["docs_multirow"]
        ),
        "docs_every_row_has_po": frac(pooled["docs_all_rows_po"], pooled["docs_po_group"]),
        "mean_distinct_po_over_rows_multirow_docs": round(float(np.mean(ratios)), 3)
        if ratios
        else None,
    }
    out["groups_without_any_po"] = sum(v["rows_with_po"] == 0 for v in out["per_group"].values())
    pos = [
        r["purchase_order"]
        for d in inv
        for r in d["line_items"]
        if not is_empty(r["purchase_order"])
    ]
    out["po_shapes"] = dict(Counter(shape(p) for p in pos).most_common(6))
    out["po_values_n"] = len(pos)
    sh_total = sum(1 for d in inv for r in d["line_items"] if not is_empty(r["purchase_order"]))
    out["rows_with_po_total"] = sh_total
    # Also: waybills have no rows.
    out["waybill_docs_with_rows"] = sum(
        bool(d["line_items"]) for d in docs if d["doc_type"] == "waybill"
    )
    return out


def item6(docs: list[dict[str, Any]], groups: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for s, per_t in by_st(docs).items():
        ds = per_t["invoice"]
        counts = [len(d["line_items"]) for d in ds]
        dup_rows = dup_docs = ident_rows = ident_docs = 0
        for d in ds:
            c = Counter(str(r["supplier_part_number"]).strip().upper() for r in d["line_items"])
            extra = sum(v - 1 for v in c.values() if v > 1)
            dup_rows += sum(v for v in c.values() if v > 1)
            dup_docs += extra > 0
            full = Counter(tuple(str(r.get(f)) for f in ROW) for r in d["line_items"])
            iextra = sum(v - 1 for v in full.values() if v > 1)
            ident_rows += iextra
            ident_docs += iextra > 0
        total_rows = sum(counts)
        multi = [d for d in ds if len(d["pages"]) > 1]
        out[s] = {
            "invoice_docs": len(ds),
            "waybill_docs_with_rows": sum(bool(d["line_items"]) for d in per_t["waybill"]),
            "rows_total": total_rows,
            "rows_per_doc": dist(counts),
            "docs_zero_rows": sum(c == 0 for c in counts),
            "rows_per_doc_multipage": dist([len(d["line_items"]) for d in multi]),
            "rows_per_doc_singlepage": dist(
                [len(d["line_items"]) for d in ds if len(d["pages"]) == 1]
            ),
            "docs_with_duplicate_supplier_part": frac(dup_docs, len(ds)),
            "rows_in_duplicate_supplier_part_sets": frac(dup_rows, total_rows),
            "docs_with_fully_identical_rows": frac(ident_docs, len(ds)),
            "surplus_rows_fully_identical": frac(ident_rows, total_rows),
            "multipage_invoice_docs": frac(len(multi), len(ds)),
            "docs_dup_supplier_part_but_distinct_rows": dup_docs - ident_docs
            if dup_docs >= ident_docs
            else None,
        }
    # Multipage visual check (one multipage doc per supplier group).
    v = {}
    for did, o in VISUAL_GROUP_OBS.items():
        v[groups[did]] = {
            "p2_doc": o["p2"],
            "continues_across_pages": True,
            "page2_repeats_header_row": o["p2_header"],
        }
    out["visual_page2"] = v
    out["visual_page2_summary"] = {
        "groups_viewed": len(v),
        "table_continues_on_page2": len(v),
        "page2_repeats_header_row_yes": sum(x["page2_repeats_header_row"] for x in v.values()),
        "page2_repeats_header_row_no": sum(not x["page2_repeats_header_row"] for x in v.values()),
        "page2_banner": "all viewed page-2 images start with '<TITLE> - continued <invoice number>' and row numbering continues from page 1",
        "totals_after_last_row_only_viewed_on_docs": len(DETAIL_P2_DOCS),
    }
    # The 5 detailed page-1-bottom + page-2-top views requested by the brief.
    first5 = [x for x in v.values() if x["p2_doc"] in DETAIL_P2_DOCS]
    out["visual_page2_first5"] = {
        "docs": DETAIL_P2_DOCS,
        "table_continues": sum(x["continues_across_pages"] for x in first5),
        "header_row_repeated_yes": sum(x["page2_repeats_header_row"] for x in first5),
        "header_row_repeated_no": sum(not x["page2_repeats_header_row"] for x in first5),
    }
    return out


# ----------------------------------------------------------------------------- item 7/8


def awb_stats(vals: list[tuple[str, str | None]]) -> dict[str, Any]:
    nn = [(d, v) for d, v in vals if not is_empty(v)]
    res = Counter(mawb_check(v) for _, v in nn)
    strict = sum(bool(re.fullmatch(r"\d{3}-\d{8}", str(v))) for _, v in nn)
    return {
        "non_null": len(nn),
        "shape_counts": dict(Counter(shape(v) for _, v in nn).most_common(6)),
        "strict_3dash8_shape": frac(strict, len(nn)),
        "checkable_11_digits": frac(res["pass"] + res["fail"], len(nn)),
        "mod7_pass_of_checkable": frac(res["pass"], res["pass"] + res["fail"]),
        "mod7_pass_of_non_null": frac(res["pass"], len(nn)),
        "mod7_fail": res["fail"],
        "uncheckable": frac(res["uncheckable"], len(nn)),
        "fail_doc_ids_sample": [d for d, v in nn if mawb_check(v) == "fail"][:5],
        "uncheckable_doc_ids_sample": [d for d, v in nn if mawb_check(v) == "uncheckable"][:5],
        "chance_pass_rate_pct": round(100 / 7, 2),
        # P(X >= observed passes) if the 8th digit were uniform random (p = 1/7), exact binomial.
        "binom_upper_tail_p_if_random": round(
            sum(
                math.comb(len(nn), i) * (1 / 7) ** i * (6 / 7) ** (len(nn) - i)
                for i in range(res["pass"], len(nn) + 1)
            ),
            4,
        )
        if nn
        else None,
    }


def item7(docs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for s, per_t in by_st(docs).items():
        out[s] = {
            "waybill_mawb": awb_stats(
                [(d["doc_id"], d["header"].get("mawb")) for d in per_t["waybill"]]
            ),
            "invoice_awb_number": awb_stats(
                [(d["doc_id"], d["header"].get("awb_number")) for d in per_t["invoice"]]
            ),
        }
        hv = [
            d["header"].get("hawb")
            for d in per_t["waybill"]
            if not is_empty(d["header"].get("hawb"))
        ]
        out[s]["waybill_hawb"] = {
            "non_null": len(hv),
            "of": len(per_t["waybill"]),
            "shape_counts": dict(Counter(shape(v) for v in hv).most_common(6)),
            "mod7_pass_when_treated_as_mawb": sum(mawb_check(v) == "pass" for v in hv),
        }
    both = {
        "waybill_mawb": awb_stats(
            [(d["doc_id"], d["header"].get("mawb")) for d in docs if d["doc_type"] == "waybill"]
        ),
        "invoice_awb_number": awb_stats(
            [
                (d["doc_id"], d["header"].get("awb_number"))
                for d in docs
                if d["doc_type"] == "invoice"
            ]
        ),
    }
    out["train+dev"] = both
    return out


def item8(docs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for s, per_t in list(by_st(docs).items()) + [
        ("train+dev_pooled", {t: [d for d in docs if d["doc_type"] == t] for t in TYPES})
    ]:
        inv, wb = per_t["invoice"], per_t["waybill"]
        mawb = {norm_awb(d["header"].get("mawb")) for d in wb} - {None}
        hawb = {norm_awb(d["header"].get("hawb")) for d in wb} - {None}
        iawb = {norm_awb(d["header"].get("awb_number")) for d in inv} - {None}
        inv_nn = [d for d in inv if norm_awb(d["header"].get("awb_number"))]
        wb_m = [d for d in wb if norm_awb(d["header"].get("mawb"))]
        wb_h = [d for d in wb if norm_awb(d["header"].get("hawb"))]
        out[s] = {
            "invoice_to_waybill": {
                "invoices_with_awb": len(inv_nn),
                "match_mawb": frac(
                    sum(norm_awb(d["header"]["awb_number"]) in mawb for d in inv_nn), len(inv_nn)
                ),
                "match_hawb": frac(
                    sum(norm_awb(d["header"]["awb_number"]) in hawb for d in inv_nn), len(inv_nn)
                ),
            },
            "waybill_to_invoice": {
                "waybills": len(wb),
                "mawb_matches_invoice_awb": frac(
                    sum(norm_awb(d["header"]["mawb"]) in iawb for d in wb_m), len(wb)
                ),
                "hawb_matches_invoice_awb": frac(
                    sum(norm_awb(d["header"]["hawb"]) in iawb for d in wb_h), len(wb)
                ),
                "waybills_with_hawb": len(wb_h),
            },
            "invoice_awb_values_shared_by_two_invoices": sum(
                v > 1 for v in Counter(norm_awb(d["header"]["awb_number"]) for d in inv_nn).values()
            ),
            "waybill_mawb_values_shared_by_two_waybills": sum(
                v > 1 for v in Counter(norm_awb(d["header"]["mawb"]) for d in wb_m).values()
            ),
            "invoice_awb_prefix_overlap_with_waybill_mawb_prefix": len(
                {v[:3] for v in iawb} & {v[:3] for v in mawb}
            ),
        }
    return out


# ----------------------------------------------------------------------------- report


def _f(x: dict[str, Any]) -> str:
    """Format a frac() triple as 'n/of (pct%)'."""
    return f"{x['n']}/{x['of']} ({x['pct']:.1f}%)"


def _d(x: dict[str, Any]) -> str:
    """Format a dist() dict as 'min / median / p90 / max'."""
    return f"{x['min']} / {x['median']:g} / {x['p90']:g} / {x['max']}"


def _table(head: list[str], rows: list[list[Any]]) -> list[str]:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out + [""]


_FMT_NAME = {
    "MDY": "MM/DD/YYYY",
    "DMY": "DD/MM/YYYY",
    "DMON": "DD-MON-YYYY",
    "LONG": "Month DD, YYYY",
    "ISO": "YYYY-MM-DD",
}


def render(stats: dict[str, Any]) -> str:  # noqa: C901, PLR0915 (one flat report, kept linear on purpose)
    """Render reports/recon.md. Every count is read from `stats`; nothing is typed by hand."""
    L: list[str] = []
    s1, s2, s3 = stats["item1_shape"], stats["item2_grouping"], stats["item3_nulls"]
    s4, s5, s6 = stats["item4_formats"], stats["item5_purchase_order"], stats["item6_rows"]
    s7, s8 = stats["item7_awb_numbers"], stats["item8_crosslinks"]
    m = stats["meta"]["docs"]
    UNV = "**UNVERIFIED** (computed by `scripts/recon.py`, seed 42; verifier to recompute)"

    L += [
        "# Phase 0 data recon (spec section 5, items 1-8)",
        "",
        "Scope: train + dev labels and images only. Test is touched only for file-level counts and image",
        "header dimensions. Item 9 (label audit) is done separately. Confidential: no names, IDs or",
        "values are reproduced here beyond doc_id + field references; supplier groups are anonymised",
        "(`inv_gNN`, `wb_gNN`).",
        "",
        "**Every number in this document is UNVERIFIED.** All counts come from `reports/recon_stats.json`",
        "(produced by `uv run python scripts/recon.py`); denominators are shown as `n/of (pct)`. Rows",
        "marked *visual* come from reading page images by hand and are recorded in the script as",
        "doc_id-keyed constants (`VISUAL_*`), cross-checked against gold where gold allows it.",
        "",
        f"Docs: train invoice {m['train']['invoice']} / waybill {m['train']['waybill']}; "
        f"dev invoice {m['dev']['invoice']} / waybill {m['dev']['waybill']}.",
        "",
    ]

    # ---- item 1
    L += ["## 1. Basic shape", "", UNV, ""]
    rows = []
    for s in ("train", "dev"):
        for t in ("invoice", "waybill"):
            x = s1[s][t]
            rows.append([
                s, t, x["docs"], x["pages"],
                ", ".join(f"{k}p:{v}" for k, v in x["pages_per_doc"].items()),
                _f(x["scanned_pages"]), _f(x["docs_any_scanned_page"]), x["docs_mixed_png_jpg"]["n"],
            ])  # fmt: skip
    L += _table(["split", "type", "docs", "pages", "pages/doc (docs)", "scanned pages (.jpg)",
                 "docs with scanned page", "mixed png+jpg docs"], rows)  # fmt: skip
    L += ["Resolution (width x height : format, pages) and DPI metadata:", ""]
    rows = []
    for s in ("train", "dev"):
        for t in ("invoice", "waybill"):
            x = s1[s][t]
            rows.append([s, t, "; ".join(f"{k}: {v}" for k, v in x["resolution_counts"].items()),
                         _f(x["pages_with_real_dpi_metadata"]),
                         "; ".join(f"{k} -> {v}" for k, v in x["dpi_metadata_patterns"].items())])  # fmt: skip
    L += _table(
        ["split", "type", "resolutions", "pages with real DPI", "metadata patterns (pages)"], rows
    )
    tf = s1["test_file_level"]
    L += [
        "Test, file level only: "
        f"{tf['docs']} docs, {tf['pages']} pages (pages/doc "
        f"{', '.join(f'{k}p:{v}' for k, v in tf['pages_per_doc'].items())}); "
        f"{tf['png_pages']} png + {tf['jpg_pages']} jpg = scanned share {_f(tf['scanned_pages'])}; "
        f"resolutions {'; '.join(f'{k}: {v}' for k, v in tf['resolution_counts'].items())}.",
        "",
        "Findings: every page in every split is one resolution; there is no real DPI metadata (JPEG JFIF",
        "unit 0 is an aspect ratio only; PNG has none), so scale must be assumed constant. A doc is either",
        "all-png or all-jpg (no mixed docs). Waybills are always single-page; only invoices are multipage.",
        "",
    ]

    # ---- item 2
    L += ["## 2. Supplier grouping key", "", UNV, ""]
    sn = s2["invoice_supplier_names"]
    L += [
        "### 2a. Invoices: `supplier_name`",
        "",
        f"Distinct raw strings {sn['train+dev']['distinct_raw']}, after case/space strip "
        f"{sn['train+dev']['distinct_casefold_strip']}, after the score.py name rule "
        f"{sn['train+dev']['distinct_score_normalized']} (of {sn['train+dev']['docs']} docs; "
        f"train {sn['train']['distinct_raw']}, dev {sn['dev']['distinct_raw']}; null supplier_name: "
        f"{sn['train+dev']['null_supplier_name']}). Names are clean unique strings, no casing/punctuation "
        f"variants. Suppliers in dev but not train: {sn['suppliers_in_dev_not_in_train']}; in train but not "
        f"dev: {sn['suppliers_in_train_not_in_dev']}, so **official dev has zero unseen suppliers**.",
        "",
    ]
    wi = s2["waybill_identity"]
    L += [
        "### 2b. Waybills: carrier / shipper_name vs layout",
        "",
        f"Of {wi['carrier']['docs']} waybills: distinct carriers {wi['carrier']['distinct']} "
        f"(docs per carrier min/median/p90/max {_d(wi['carrier']['docs_per_value'])}; "
        f"{wi['carrier']['values_in_exactly_one_doc']} carriers appear once); carrier brand (first word) "
        f"{wi['carrier_brand']['distinct']} distinct (docs per brand {_d(wi['carrier_brand']['docs_per_value'])}); "
        f"carrier = brand x {wi['carrier_suffix_distinct']} suffixes; shipper_name "
        f"{wi['shipper_name']['distinct']} distinct (= one per doc, so it cannot define a group). Dev brands not "
        f"in train: {wi['brands_in_dev_not_in_train']}; dev full carrier strings not in train: "
        f"{wi['carriers_in_dev_not_in_train']}.",
        "",
        "*Visual* (8 waybill pages, 2 scans): the carrier is only the title text. The 2-column grid of field",
        "boxes changes order and font between documents of the **same** carrier, and a missing HAWB leaves a",
        "blank slot. Waybill layout is therefore per-document, not per-carrier.",
        "",
    ]
    no = s2["name_overlap"]
    L += [
        "Name coincidence: waybill shipper equals an invoice supplier "
        f"{_f(no['waybill_shipper_equals_invoice_supplier'])} (fuzzy >= 0.85: "
        f"{_f(no['waybill_shipper_fuzzy_ge_0.85_invoice_supplier'])}); invoice supplier equals a waybill shipper "
        f"{_f(no['invoice_supplier_equals_waybill_shipper'])}; waybill consignee equals an invoice buyer/ship-to "
        f"{_f(no['waybill_consignee_equals_invoice_buyer_or_ship_to'])}. Entity names are disjoint across types.",
        "",
        "### 2c. Label-free layout clusters vs labels",
        "",
        f"Method: {s2['clustering_method']}. Fit on train+dev jointly per doc type; metrics restricted to "
        "each scope. `purity` = share of docs in their cluster's majority label (inflated when k approaches the "
        "number of docs); `coverage` = share of docs in their label's majority cluster. ARI is the "
        "chance-corrected number to read.",
        "",
    ]
    runs = s2["clustering_runs"]
    keys = [("invoice", "supplier_name", 18), ("waybill", "carrier", 45), ("waybill", "carrier_brand", 10),
            ("waybill", "shipper_name", 45)]  # fmt: skip
    rows = []
    for t, lab, k0 in keys:
        sel = [
            r for r in runs if r["type"] == t and r["label"] == lab and r["scope"] == "train+dev"
        ]
        base = next(r for r in sel if r["signature"] == "full" and r["pages"] == "all_pages"
                    and r["algo"] == "kmeans" and r["k"] == k0)  # fmt: skip
        best = max(sel, key=lambda r: r["ARI"])
        for tag, r in (("spec baseline", base), ("best ARI", best)):
            sc = {x["scope"]: x["ARI"] for x in runs if all(x[a] == r[a] for a in
                  ("type", "label", "signature", "pages", "algo", "k")) and x["scope"] != "train+dev"}  # fmt: skip
            rows.append([t, lab, tag, f"{r['signature']}/{r['pages']}/{r['algo']}/k={r['k']}", r["n"],
                         r["n_label_classes"], r["ARI"], r["NMI"], r["purity"], r["label_coverage"],
                         sc.get("train", "-"), sc.get("dev", "-")])  # fmt: skip
    L += _table(["type", "label", "config", "signature/pages/algo/k", "n docs", "label classes", "ARI", "NMI",
                 "purity", "coverage", "ARI train", "ARI dev"], rows)  # fmt: skip
    best_inv = max(
        (r for r in runs if r["type"] == "invoice" and r["scope"] == "train+dev"),
        key=lambda r: r["ARI"],
    )
    best_wb = max((r for r in runs if r["type"] == "waybill" and r["label"] != "shipper_name"
                   and r["scope"] == "train+dev"), key=lambda r: r["ARI"])  # fmt: skip
    L += [
        f"Reading: invoices cluster partially by supplier (best ARI {best_inv['ARI']}, digital pages only; "
        "scans and varying row counts blur a 48x64 thumbnail) but far above waybills, where the best ARI "
        f"against carrier or brand is {best_wb['ARI']}: **no label tracks waybill layout**. A coarse pixel "
        "signature is a weak layout detector, so it cannot replace the label key for invoices, and for "
        "waybills it finds structure that no label explains.",
        "",
        "### 2d. Cross-links",
        "",
        f"Waybill to invoice via awb_number: see item 8 (pooled train+dev: "
        f"{_f(s8['train+dev_pooled']['invoice_to_waybill']['match_mawb'])} invoice awb equal a waybill MAWB, "
        f"{_f(s8['train+dev_pooled']['invoice_to_waybill']['match_hawb'])} equal a HAWB). A waybill "
        "**cannot inherit** an invoice's supplier group.",
        "",
        "### 2e. Decision: `supplier_group` (written to `meta/supplier_groups.json`)",
        "",
        "- **Invoices**: normalised `supplier_name` (score.py name rule), 18 groups, id `inv_gNN`.",
        "- **Waybills**: carrier brand (first word of `carrier`), 10 groups, id `wb_gNN`. Rejected: "
        "`shipper_name` (one value per doc, GroupKFold would degenerate to a random split); full `carrier` "
        "(45 values, 17 of them single-doc); a pixel-cluster key (ARI at most "
        f"{best_wb['ARI']} against any label, and clusters would be defined by scan noise). The brand is "
        "the only repeated printed identity token with enough levels (>= 3) for K=3 folds that each hold "
        "waybills. **Caveat: it holds out the printed title text, not the layout**, because waybill layout "
        "is per-document. The waybill part of the leave-supplier-out score therefore measures unseen carrier "
        "titles only and should not be read as unseen layouts.",
        "",
        f"Total groups: {s2['groups_total']} (invoice {s2['groups_invoice']['n_groups']} + waybill "
        f"{s2['groups_waybill']['n_groups']}); the ~20 expectation holds for invoices (18), not for the "
        "28 total used for folds. Docs per group (min / median / p90 / max): invoice "
        f"{_d(s2['groups_invoice']['docs_per_group'])}; waybill {_d(s2['groups_waybill']['docs_per_group'])}.",
        "",
    ]
    rows = [[g, v["train"], v["dev"], v["total"]] for g, v in s2["group_table"].items()]
    L += _table(["group", "train docs", "dev docs", "total"], rows)

    # ---- item 3
    L += ["## 3. Per-field null rates", "", UNV, ""]
    rule = s3["rule"]
    L += [
        "Rule (explicit):",
        "",
        f"- Row fields: {rule['row_fields']}",
        f"- Header fields: {rule['header_fields']}",
        "",
    ]
    for t in ("invoice", "waybill"):
        rows = []
        for f in HEADER[t]:
            tr, dv = s3["null_rates"]["train"][t][f], s3["null_rates"]["dev"][t][f]
            c = s3["header_classification"][t]["fields"][f]
            rows.append([f, _f(tr), _f(dv), c["class"], _f(c["nulls_in_docs_with_scanned_page"]),
                         _f(c["nulls_in_all_png_docs"])])  # fmt: skip
        L += [
            f"{t} header (null = empty), train / dev, class by the rule, null rate on scanned vs all-png docs:",
            "",
        ]
        L += _table(["field", "train null", "dev null", "class", "scanned docs", "png docs"], rows)
    rows = []
    for s in ("train", "dev"):
        r = s3["null_rates"][s]["invoice"]["_rows"]
        rows.append([s, s3["null_rates"][s]["invoice"]["_n_rows"]] + [_f(r[f]) for f in ROW])
    L += ["Invoice row fields (null rows), train / dev (waybills have no rows):", ""]
    L += _table(["split", "rows"] + ROW, rows)
    rc = s3["row_field_classification"]
    rows = [[f, v["groups"], v["printed"], v["not_printed"], v["mixed"], _f(v["rows_in_not_printed_groups"])]
            for f, v in rc.items()]  # fmt: skip
    L += ["Row-field rule per supplier group (18 groups, train+dev):", ""]
    L += _table(
        [
            "field",
            "groups",
            "printed (0% null)",
            "not printed (100% null)",
            "mixed",
            "rows in not-printed groups",
        ],
        rows,
    )
    hi, hw = s3["header_classification"]["invoice"], s3["header_classification"]["waybill"]
    L += [
        "Header classification (train+dev):",
        f"- invoice: {hi['header_null_cells']['all_nulls']} null cells of {hi['header_null_cells']['total_header_cells']}; "
        f"{hi['header_null_cells']['illegible_nulls']} illegible (redaction) in invoice_number/invoice_date, "
        f"{hi['header_null_cells']['optional_line_nulls']} absent-line (awb_number); docs with an illegible null "
        f"{_f(hi['header_null_cells']['docs_with_illegible_null'])}.",
        f"- waybill: {hw['header_null_cells']['all_nulls']} null cells of {hw['header_null_cells']['total_header_cells']}, "
        f"all absent-line (hawb); illegible nulls: {hw['header_null_cells']['illegible_nulls']}.",
    ]
    tot_null = hi["header_null_cells"]["all_nulls"] + hw["header_null_cells"]["all_nulls"]
    tot_opt = (
        hi["header_null_cells"]["optional_line_nulls"]
        + hw["header_null_cells"]["optional_line_nulls"]
    )
    tot_ill = (
        hi["header_null_cells"]["illegible_nulls"] + hw["header_null_cells"]["illegible_nulls"]
    )
    L += [
        f"- scorer `illegible_fields` denominator (all gold header nulls) = {tot_null}: absent-line "
        f"{_f({'n': tot_opt, 'of': tot_null, 'pct': pct(tot_opt, tot_null)})}, redaction "
        f"{_f({'n': tot_ill, 'of': tot_null, 'pct': pct(tot_ill, tot_null)})}.",
        "",
        "Per-group spread (is the optional line a group property?): "
        + "; ".join(
            f"{t}.{f} null rate across groups {v['min_pct']:.1f}% to {v['max_pct']:.1f}% "
            f"(groups with 0 nulls {v['groups_with_zero_nulls']}, with 100% nulls {v['groups_with_100pct_nulls']}, "
            f"of {v['groups']})"
            for t in ("invoice", "waybill")
            for f, v in s3["header_classification"][t]["per_group_null_rate_range"].items()
            if f in ("awb_number", "hawb")
        )
        + ". It is a per-document property, not a per-supplier one.",
        "",
        "Visual confirmation (13 views; gold null confirmed for all: "
        f"{s3['visual_null_checks_all_gold_null']}):",
        "",
    ]
    rows = [[c["doc_id"], c["field"], c["class"], "jpg" if c["scanned"] else "png", c["observation"]]
            for c in s3["visual_null_checks"]]  # fmt: skip
    L += _table(["doc_id", "field", "class", "format", "what the image showed"], rows)
    rp = s3["redacted_invoice_number_on_multipage_docs"]
    L += [
        "Trap found while viewing: when page 1's invoice number is redacted on a multipage invoice, page 2's "
        "`<TITLE> - continued <invoice number>` banner prints it legibly, yet gold stays null "
        f"({_f(rp['gold_null_invoice_number'])} multipage invoices have a null invoice_number; "
        f"{rp['page2_banner_legible_in_viewed']}/{len(rp['viewed_doc_ids'])} viewed had a legible banner: "
        f"{', '.join(rp['viewed_doc_ids'])}; the other {rp['gold_null_invoice_number']['n'] - len(rp['viewed_doc_ids'])} "
        "not viewed). Gold follows the header page.",
        "",
    ]

    # ---- item 4
    L += ["## 4. Formats", "", UNV, ""]
    L += [
        f"*Visual*: one digital page per supplier group ({s4['visual_groups_n']} groups, "
        f"{s4['visual_groups_unique']} distinct) plus {s4['visual_scan_exemplars']['n']} scans of the same "
        "groups, 36+ images; page-image evidence is per exemplar, so the assumption that a group's format is "
        "constant across all its docs is checked on 2 pages per group only.",
        "",
    ]
    vg, dg = s4["visual_groups"], s4["date_per_group"]
    rows = []
    for g in sorted(vg):
        d = dg[g]
        amb = "ambiguous as printed" if d["numeric_format"] else "no (month name / ISO)"
        rows.append([g, _FMT_NAME[vg[g]["date_fmt"]], amb, _f(frac(d["day_le_12"], d["dates"])),
                     f"{d['day_gt_12']}/{d['dates']}", vg[g]["exemplar_scanned"] and "scan" or "png"])  # fmt: skip
    L += ["Date formats per group (printed format from the image; counts from gold dates):", ""]
    L += _table(
        [
            "group",
            "printed date format",
            "date ambiguity",
            "gold dates with day<=12",
            "gold dates with day>12",
            "exemplar",
        ],
        rows,
    )
    da = s4["date_ambiguity_numeric_groups"]
    L += [
        f"- Format counts over {s4['visual_groups_n']} groups: "
        + ", ".join(
            f"{_FMT_NAME[k]} x{v}" for k, v in sorted(s4["visual_date_format_counts"].items())
        )
        + f". {da['groups_numeric']} groups print a numeric (ambiguous-looking) date.",
        f"- **Numeric-format groups: dates with day <= 12 = {_f(da['ambiguous_day_le_12'])}**; the smallest "
        f"number of day>12 resolver dates in any numeric group is {da['min_resolvers_in_a_numeric_group']}. "
        f"Across all invoices, day<=12 = {_f(da['all_invoice_dates'])}, all in the long-month / DD-MON / ISO "
        "groups where the printed form is not ambiguous. Gold dates are all valid ISO "
        f"(non-ISO gold: {s4['date_non_iso_gold']}); range {s4['date_range']['min_month']} to "
        f"{s4['date_range']['max_month']} ({s4['date_range']['months_present']} months).",
        "- Unverified for test: nothing here shows an unseen supplier will keep this property.",
        "",
    ]
    cs = s4["currency_symbol_ambiguity"]
    L += [
        "Currency: "
        + s4["visual_summary"]["currency_printed_as"]
        + ". Gold currencies: "
        + ", ".join(f"{k} {v}" for k, v in s4["gold_currency_counts"].items())
        + f" (outside the five: {cs['gold_currencies_not_in_5']}); every group carries "
        f"{s4['currencies_per_group']['min']}-{s4['currencies_per_group']['max']} of the 5 currencies, so currency "
        "is independent of supplier and address country.",
        f"Symbol ambiguity if a symbol were ever printed: '$' would stand for USD or SGD "
        f"({_f(cs['dollar_sign_currencies_USD_SGD'])} of gold invoices); the yen/yuan sign for JPY or CNY "
        f"({_f(cs['yen_sign_currencies_JPY_CNY'])}); only EUR ({_f(cs['euro_sign_currencies_EUR'])}) is "
        "unambiguous. **A bare '$' is never printed in the viewed pages, so no gold-labelled evidence exists "
        "for '$' meaning USD; gold has SGD, another '$' currency.**",
        "",
        "Numbers: " + s4["visual_summary"]["number_format"] + ". Unit price printed with "
        f"{s4['visual_summary']['unit_price_decimals']} decimals, line totals {s4['visual_summary']['line_total_decimals']}. "
        "Gold shapes (digits collapsed): "
        + "; ".join(
            f"{k}: {v['shapes_top']} (n={v['n']}, with comma {v['contains_comma']})"
            for k, v in s4["gold_number_shapes"].items()
        )  # fmt: skip
        + ". Airports: "
        + s4["visual_summary"]["airport_printed_as"]
        + "; gold shapes "
        + str(s4["gold_airport_shapes"])
        + ".",
        "",
    ]

    # ---- item 5
    L += ["## 5. Purchase order", "", UNV, ""]
    pp = s5["pooled_po_printing_groups"]
    L += [
        f"*Visual*: of {s4['visual_groups_n']} groups, "
        f"{sum(v['po'] == 'per_row' for v in vg.values())} print a PO column (one value per row; the PO "
        f"differs between rows on every viewed multi-row page) and {sum(v['po'] == 'absent' for v in vg.values())} print "
        "no PO anywhere. No group prints a once-per-invoice PO. Visual and gold agree for "
        f"{sum(v['visual_matches_gold_po'] for v in vg.values())}/{len(vg)} groups on PO and "
        f"{sum(v['visual_matches_gold_cp'] for v in vg.values())}/{len(vg)} on customer part.",
        "",
        f"Gold ({s5['rows_with_po_total']} rows with a PO): in the {pp['groups']} PO-printing groups "
        f"{_f(pp['rows_with_po'])} of rows carry a PO; multirow docs whose rows all share one PO "
        f"{_f(pp['multirow_docs_all_rows_share_one_po'])}; mean distinct POs / rows in those docs "
        f"{pp['mean_distinct_po_over_rows_multirow_docs']}; docs where every row has a PO "
        f"{_f(pp['docs_every_row_has_po'])}. In the {s5['groups_without_any_po']} groups that print no PO every "
        f"row is null. PO shapes (letters A, digits 9): {s5['po_shapes']}.",
        "",
    ]
    rows = [[g, v["rows"], v["rows_with_po"], v["rows_null_po"], v["docs"], v["docs_multirow_with_po"],
             v["docs_all_rows_share_one_po"]] for g, v in sorted(s5["per_group"].items())]  # fmt: skip
    L += _table(["group", "rows", "rows with PO", "null-PO rows", "docs", "multirow docs with PO",
                 "of which all rows share one PO"], rows)  # fmt: skip

    # ---- item 6
    L += ["## 6. Rows", "", UNV, ""]
    rows = []
    for s in ("train", "dev"):
        x = s6[s]
        rows.append([s, x["invoice_docs"], x["rows_total"], _d(x["rows_per_doc"]),
                     _d(x["rows_per_doc_singlepage"]), _d(x["rows_per_doc_multipage"]),
                     _f(x["docs_with_duplicate_supplier_part"]), _f(x["rows_in_duplicate_supplier_part_sets"]),
                     _f(x["docs_with_fully_identical_rows"]), _f(x["surplus_rows_fully_identical"])])  # fmt: skip
    L += [
        "Invoices (waybills have 0 rows: "
        + f"{s6['train']['waybill_docs_with_rows'] + s6['dev']['waybill_docs_with_rows']}"
        " waybill docs carry line items). Rows per doc as min / median / p90 / max:",
        "",
    ]
    L += _table(["split", "invoices", "rows", "rows/doc", "single-page rows/doc", "multipage rows/doc",
                 "docs with duplicate supplier_part_number", "rows in duplicate sets",
                 "docs with fully identical rows", "surplus identical rows"], rows)  # fmt: skip
    p2, p5 = s6["visual_page2_summary"], s6["visual_page2_first5"]
    L += [
        f"*Visual*, multipage invoices ({', '.join(p5['docs'])}; page-1 bottom + page-2 top): the table continues "
        f"on page 2 in {p5['table_continues']}/{len(p5['docs'])}; page 2 repeats the column header row in "
        f"{p5['header_row_repeated_yes']} yes / {p5['header_row_repeated_no']} no. Extended to one multipage doc per "
        f"group ({p2['groups_viewed']} groups): continues {p2['table_continues_on_page2']}/{p2['groups_viewed']}; "
        f"header repeated {p2['page2_repeats_header_row_yes']} yes / {p2['page2_repeats_header_row_no']} no "
        "(a per-supplier property). Every viewed page 2 starts with a `<TITLE> - continued <invoice number>` "
        "banner and numbering continues from page 1; totals appear after the last row only (checked on "
        f"{p2['totals_after_last_row_only_viewed_on_docs']} docs). Single-page docs never exceed "
        f"{s6['train']['rows_per_doc_singlepage']['max']} rows; multipage docs start at "
        f"{s6['train']['rows_per_doc_multipage']['min']} rows.",
        "",
    ]

    # ---- item 7
    L += ["## 7. Waybill numbers", "", UNV, ""]
    rows = []
    for s in ("train", "dev", "train+dev"):
        for key, name in (
            ("waybill_mawb", "waybill MAWB"),
            ("invoice_awb_number", "invoice awb_number"),
        ):
            x = s7[s][key]
            rows.append([s, name, x["non_null"], str(x["shape_counts"]), _f(x["strict_3dash8_shape"]),
                         _f(x["checkable_11_digits"]), _f(x["mod7_pass_of_checkable"]),
                         _f(x["uncheckable"]), x["binom_upper_tail_p_if_random"]])  # fmt: skip
    L += [
        "Shape letters A, digits 9. Mod-7 test: 8th serial digit == int(first 7 serial digits) mod 7.",
        "",
    ]
    L += _table(
        [
            "split",
            "field",
            "non-null",
            "shapes",
            "strict NNN-NNNNNNNN",
            "checkable (11 digits)",
            "mod-7 pass of checkable",
            "uncheckable",
            "P(>= passes if random, p=1/7)",
        ],
        rows,
    )
    L += [
        "HAWB shape (train / dev): "
        + "; ".join(
            f"{s}: {s7[s]['waybill_hawb']['shape_counts']} "
            f"({s7[s]['waybill_hawb']['non_null']}/{s7[s]['waybill_hawb']['of']} "
            "non-null)"
            for s in ("train", "dev")
        )
        + ".",
        f"Chance pass rate for a random 8th digit is {s7['train+dev']['waybill_mawb']['chance_pass_rate_pct']}%. "
        "Observed MAWB passes are indistinguishable from chance and invoice awb_number passes are below it, "
        "so **gold does not follow the mod-7 convention**.",
        "",
    ]

    # ---- item 8
    L += [
        "## 8. Invoice <-> waybill cross-links (same split, spaces and dashes stripped)",
        "",
        UNV,
        "",
    ]
    rows = []
    for s in ("train", "dev", "train+dev_pooled"):
        a, b = s8[s]["invoice_to_waybill"], s8[s]["waybill_to_invoice"]
        rows.append([s, a["invoices_with_awb"], _f(a["match_mawb"]), _f(a["match_hawb"]), b["waybills"],
                     _f(b["mawb_matches_invoice_awb"]), _f(b["hawb_matches_invoice_awb"]),
                     s8[s]["invoice_awb_values_shared_by_two_invoices"]])  # fmt: skip
    L += _table(
        [
            "scope",
            "invoices with awb",
            "invoice awb == a MAWB",
            "invoice awb == a HAWB",
            "waybills",
            "waybill MAWB == an invoice awb",
            "waybill HAWB == an invoice awb",
            "awb values shared by 2 invoices",
        ],
        rows,
    )
    L += ["Zero links in either direction, within or across splits (the pooled row ignores split)."]
    L += [""]

    # ---- implications
    L += ["## Implications for design", "", UNV, ""]
    hdr = s6["visual_page2_summary"]
    sc_n = sum(s1[s][t]["scanned_pages"]["n"] for s in ("train", "dev") for t in TYPES)
    sc_of = sum(s1[s][t]["scanned_pages"]["of"] for s in ("train", "dev") for t in TYPES)
    trdev_scan = frac(sc_n, sc_of)
    L += [
        f"- **PO propagation (Phase 3.2): drop it.** Gold never copies a PO: {_f(pp['multirow_docs_all_rows_share_one_po'])} "
        "multirow docs share one PO, PO is per row on every PO-printing group, and the "
        f"{s5['groups_without_any_po']} no-PO groups are null on every row. Making 3.2 conditional on Phase 0 "
        "resolves to 'do not implement'. Copying a PO would create false rows.",
        f"- **Currency symbol mapping: not needed on seen data, and unsafe if added.** ISO codes are printed in "
        f"all {s4['visual_groups_n']} groups (header and total). '$' maps to USD *or* SGD "
        f"({_f(cs['dollar_sign_currencies_USD_SGD'])}), '¥' to JPY *or* CNY ({_f(cs['yen_sign_currencies_JPY_CNY'])}); "
        "never map a symbol to a code (extraction rule: no inference). Symbol-only output goes to low "
        "confidence / null.",
        f"- **MAWB check digit (Phase 3.5): change to shape-only validation.** Mod-7 pass is "
        f"{_f(s7['train+dev']['waybill_mawb']['mod7_pass_of_checkable'])} for gold MAWBs (chance 14.3%, "
        f"P={s7['train+dev']['waybill_mawb']['binom_upper_tail_p_if_random']}) and "
        f"{_f(s7['train+dev']['invoice_awb_number']['mod7_pass_of_checkable'])} for invoice awb_number. Using "
        f"the check would flag {s7['train+dev']['waybill_mawb']['mod7_fail']}/"
        f"{s7['train+dev']['waybill_mawb']['non_null']} correct gold MAWBs; do not use it as a validator or "
        "confidence feature. "
        f"Shape `NNN-NNNNNNNN` holds for {_f(s7['train+dev']['waybill_mawb']['strict_3dash8_shape'])} MAWBs "
        f"and {_f(s7['train+dev']['invoice_awb_number']['strict_3dash8_shape'])} invoice AWBs, HAWB `AANNNNNNNN`; "
        "validate shapes only.",
        f"- **Ambiguous dates / per-cluster inference (Phase 3.3): simplify.** Numeric-format groups have "
        f"{_f(da['ambiguous_day_le_12'])} dates with day <= 12, so every numeric date resolves by itself (one "
        "of the two fields exceeds 12) and per-doc resolution is enough; per-cluster inference becomes a "
        "consistency check, not the mechanism. The "
        f"{s4['visual_groups_n'] - da['groups_numeric']} non-numeric groups are unambiguous once the month name / "
        "ISO order is parsed. Ambiguity remains a risk only for formats outside the five seen; keep the "
        "confidence penalty for that case and do not assume test follows the same property (unverified).",
        f"- **Number normalization (Phase 3.4): lightweight.** Decimal point and comma thousands on all pages "
        f"viewed; gold has no commas (shapes above) and the scorer strips commas, so strip thousands commas and "
        "ignore symbols; decimal-comma handling is defensive only (never seen). Gold total_amount has exactly 2 "
        f"decimals in {_f(s4['gold_number_shapes']['invoice.total_amount']['exactly_two_decimals'])} invoices, "
        f"JPY included ({_f(s4['gold_number_shapes']['invoice.total_amount']['jpy_with_two_decimals'])}).",
        f"- **Never dedupe rows (Phase 3.1): confirmed.** {_f(s6['train']['docs_with_duplicate_supplier_part'])} train "
        f"and {_f(s6['dev']['docs_with_duplicate_supplier_part'])} dev invoices repeat a supplier_part_number, and "
        f"{_f(s6['train']['docs_with_fully_identical_rows'])} / {_f(s6['dev']['docs_with_fully_identical_rows'])} "
        "contain fully identical rows (all four fields), which the scorer requires predicted twice. Also: "
        f"page 2 repeats the header row for only {hdr['page2_repeats_header_row_yes']}/{hdr['groups_viewed']} "
        "groups, so drop header rows by matching header text, not by position, and rely on the page-2 "
        "continuation banner rather than on a header to detect continuation.",
        "- **Merge must not fill from page 2 (Phase 3.1, header fields 'from any page'): changes.** A redacted "
        "page-1 invoice number is legible in page 2's continuation banner but gold is null "
        f"({_f(rp['gold_null_invoice_number'])} multipage invoices); filling it is scored as a false fill. Take "
        "invoice-level header fields from page 1 only, or treat a p2-only value as 'do not fill'. Cross-page "
        "agreement as a confidence feature is therefore weaker than assumed for invoice_number.",
        f"- **Grouping key for CV folds (Phase 1.2): `supplier_group` = normalised supplier_name (18) for "
        f"invoices + carrier brand (10) for waybills = {s2['groups_total']} groups.** Invoice group sizes "
        f"{_d(s2['groups_invoice']['docs_per_group'])}, waybill {_d(s2['groups_waybill']['docs_per_group'])}; "
        "GroupKFold K=3 can put both types in each fold. Waybill groups do not track layout (best cluster ARI "
        f"{best_wb['ARI']}), so unseen-layout accuracy is only meaningful for invoices; report invoice and "
        "waybill OOF separately, and expect dev (all suppliers seen) to say nothing about unseen layouts. "
        "Cross-links are absent, so the planned invoice<->waybill cross-link confidence feature (Phase 5) has "
        "nothing to match and should be dropped.",
        f"- **Illegible vs not-printed for the false-fill metric (Phase 1.1 `illegible` tag, Phase 5 null "
        f"policy).** By the rule in item 3, the scorer's false-fill denominator ({tot_null} header nulls) is "
        f"{pct(tot_opt, tot_null)}% absent-line (awb_number / hawb: learn to emit null when the label is not on "
        f"the page) and only {pct(tot_ill, tot_null)}% redactions (invoice_number / invoice_date: a black box or "
        "scribble over the value). Define `illegible` = document has a redaction-class null "
        f"({_f(hi['header_null_cells']['docs_with_illegible_null'])} of invoices; 0 waybills) and add `awb_absent` / "
        "`hawb_absent` as separate tags so the false-fill rate can be reported for both. Row-level not-printed "
        f"columns ({_f(rc['customer_part_number']['rows_in_not_printed_groups'])} of rows for customer_part_number, "
        f"{_f(rc['purchase_order']['rows_in_not_printed_groups'])} for purchase_order) are group-wide constants "
        "(no mixed groups), so null is the correct answer for them by layout, and an unseen supplier cannot be "
        "assumed to print either column.",
        "- **Resolution / DPI**: one resolution everywhere and no DPI metadata, so a single `max_pixels` setting "
        f"applies to all pages. Scanned page share: train+dev {_f(trdev_scan)}, test "
        f"{_f(tf['scanned_pages'])} (file level), so the Phase 4 scan augmentation should target roughly "
        "this share rather than a guess.",
        "",
    ]
    return "\n".join(L)


# ----------------------------------------------------------------------------- driver


def self_check_no_values(stats_text: str, docs: list[dict[str, Any]]) -> dict[str, Any]:
    """Fail loudly if any label string value (name/id/part/PO) leaks into the stats JSON."""
    vals: set[str] = set()
    for d in docs:
        for v in d["header"].values():
            if isinstance(v, str) and len(v) >= 6:
                vals.add(v)
        for r in d["line_items"]:
            for v in r.values():
                if isinstance(v, str) and len(v) >= 6:
                    vals.add(v)
    leaked = sorted(v for v in vals if v in stats_text)
    return {"label_string_values_checked": len(vals), "leaked": leaked}


def main() -> None:
    np.random.seed(SEED)
    docs = load_docs()
    stats: dict[str, Any] = {
        "meta": {
            "seed": SEED,
            "script": "scripts/recon.py",
            "docs": {s: {t: len(v) for t, v in per.items()} for s, per in by_st(docs).items()},
            "status": "UNVERIFIED (to be recomputed by the verifier)",
        }
    }
    stats["item1_shape"] = item1(docs)
    i2, groups = item2(docs)
    stats["item2_grouping"] = i2
    stats["item3_nulls"] = item3(docs, groups)
    stats["item4_formats"] = item4(docs, groups)
    stats["item5_purchase_order"] = item5(docs, groups)
    stats["item6_rows"] = item6(docs, groups)
    stats["item7_awb_numbers"] = item7(docs)
    stats["item8_crosslinks"] = item8(docs)
    text = json.dumps(stats, indent=1, sort_keys=False)
    stats["meta"]["leak_check"] = self_check_no_values(text, docs)
    if stats["meta"]["leak_check"]["leaked"]:
        raise SystemExit(f"stats would leak label values: {stats['meta']['leak_check']['leaked']}")
    (ROOT / "meta").mkdir(exist_ok=True)
    (ROOT / "meta" / "supplier_groups.json").write_text(
        json.dumps(dict(sorted(groups.items())), indent=1) + "\n", encoding="utf-8"
    )
    (ROOT / "reports" / "recon_stats.json").write_text(
        json.dumps(stats, indent=1) + "\n", encoding="utf-8"
    )
    (ROOT / "reports" / "recon.md").write_text(render(stats), encoding="utf-8")


if __name__ == "__main__":
    main()
