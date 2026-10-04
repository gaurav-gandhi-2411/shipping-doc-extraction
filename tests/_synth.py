"""A tiny synthetic corpus for the batching / shard / bench tests (no data/ folder needed).

`make_corpus` writes ``<root>/dev/images/<doc>_p<k>.png`` (different sizes, so the visual-token
part of the batching key matters) and ``<root>/dev/labels/<doc>.json`` and returns the gold dict
a `MockBackend` replays. Documents have 1-4 pages; some are waybills.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

#: (doc_id, n_pages, doc_type); image width varies with the doc so sorting has something to do.
SPECS: list[tuple[str, int, str]] = [
    ("dev_0000", 1, "invoice"),
    ("dev_0001", 3, "invoice"),
    ("dev_0002", 1, "waybill"),
    ("dev_0003", 2, "invoice"),
    ("dev_0004", 1, "invoice"),
    ("dev_0005", 4, "invoice"),
    ("dev_0006", 1, "invoice"),
    ("dev_0007", 2, "invoice"),
    ("dev_0008", 1, "waybill"),
    ("dev_0009", 3, "invoice"),
    ("dev_0010", 1, "invoice"),
]


def _gold(doc_id: str, n_pages: int, doc_type: str, k: int) -> dict[str, Any]:
    if doc_type == "invoice":
        header = {
            "invoice_number": f"INV-{k:04d}",
            "invoice_date": f"2026-01-{k + 1:02d}",
            "supplier_name": f"Supplier {k} Ltd",
            "buyer_name": f"Buyer {k} LLC",
            "ship_to_name": f"Ship {k} Inc",
            "currency": "USD",
            "total_amount": f"{100 + k}.50",
        }
    else:
        header = {
            "awb_number": f"176-{k:08d}",
            "carrier": "Carrier",
            "origin_airport": "FRA",
            "destination_airport": "SIN",
            "shipper_name": f"Shipper {k}",
            "consignee_name": f"Consignee {k}",
            "pieces": str(k + 1),
            "gross_weight_kg": f"{10 + k}.5",
        }
    rows = [
        {
            "supplier_part_number": f"P-{k}-{i}",
            "customer_part_number": None,
            "purchase_order": f"PO{k}{i}" if i % 2 else None,
            "quantity": str(10 * (i + 1)),
        }
        for i in range(2 * n_pages + 1)
    ]
    return {
        "doc_id": doc_id,
        "doc_type": doc_type,
        "header": header,
        "line_items": rows if doc_type == "invoice" else [],
        "pages": [f"{doc_id}_p{p + 1}.png" for p in range(n_pages)],
    }


def make_corpus(
    root: Path, specs: list[tuple[str, int, str]] | None = None, labels: bool = True
) -> dict[str, dict[str, Any]]:
    """Write the images (and labels) under `root`/dev and return ``{doc_id: gold}``."""
    gold: dict[str, dict[str, Any]] = {}
    images = root / "dev" / "images"
    images.mkdir(parents=True, exist_ok=True)
    for k, (doc_id, n_pages, doc_type) in enumerate(specs or SPECS):
        g = _gold(doc_id, n_pages, doc_type, k)
        gold[doc_id] = g
        for p in range(n_pages):
            Image.new("RGB", (64 + 32 * ((k + p) % 5), 48), (k * 20 % 255, 100, 150)).save(
                images / f"{doc_id}_p{p + 1}.png"
            )
    if labels:
        out = root / "dev" / "labels"
        out.mkdir(parents=True, exist_ok=True)
        for doc_id, g in gold.items():
            (out / f"{doc_id}.json").write_text(json.dumps(g), encoding="utf-8")
    return gold
