"""Read-only data inventory (counts, id cross-checks); writes reports/inventory.json."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMG_RE = re.compile(r"^(?P<doc>.+)_p(?P<page>\d+)\.(?P<ext>png|jpg)$")
EXPECTED = {"train": 400, "dev": 100, "test": 200}


def scan_images(split: str) -> dict[str, list[str]]:
    """Map doc_id -> sorted image filenames for a split."""
    docs: dict[str, list[str]] = defaultdict(list)
    for p in sorted((ROOT / "data" / split / "images").iterdir()):
        m = IMG_RE.match(p.name)
        if not m:
            raise ValueError(f"unparseable image filename: {p.name}")
        docs[m["doc"]].append(p.name)
    return {k: sorted(v) for k, v in docs.items()}


def inventory() -> dict:
    """Build the inventory dict (never reads test labels; none exist)."""
    out: dict = {}
    for split in ("train", "dev", "test"):
        imgs = scan_images(split)
        pages = [n for v in imgs.values() for n in v]
        info: dict = {
            "docs": len(imgs),
            "expected_docs": EXPECTED[split],
            "pages": len(pages),
            "pages_per_doc_hist": dict(sorted(Counter(len(v) for v in imgs.values()).items())),
            "jpg_pages": sum(n.endswith(".jpg") for n in pages),
            "png_pages": sum(n.endswith(".png") for n in pages),
        }
        if split == "test":
            sub = json.loads(
                (ROOT / "assignment" / "sample_submission.json").read_text(encoding="utf-8")
            )
            info["test_images_not_in_sample_submission"] = sorted(set(imgs) - set(sub))
            info["sample_submission_not_in_test_images"] = sorted(set(sub) - set(imgs))
        else:
            labels = {}
            for p in sorted((ROOT / "data" / split / "labels").glob("*.json")):
                labels[p.stem] = json.loads(p.read_text(encoding="utf-8"))
            info["label_docs"] = len(labels)
            info["images_without_label"] = sorted(set(imgs) - set(labels))
            info["labels_without_images"] = sorted(set(labels) - set(imgs))
            info["label_id_field_mismatch"] = sorted(
                k for k, v in labels.items() if v.get("doc_id") != k
            )
            info["pages_list_mismatch"] = sorted(
                k for k, v in labels.items() if k in imgs and sorted(v.get("pages", [])) != imgs[k]
            )
            info["doc_type_counts"] = dict(Counter(v.get("doc_type") for v in labels.values()))
        out[split] = info
    return out


def main() -> None:
    """Print and write the inventory."""
    inv = inventory()
    text = json.dumps(inv, indent=2, sort_keys=True)
    print(text)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "inventory.json").write_text(text + "\n", encoding="utf-8")
    for s, i in inv.items():
        if i["docs"] != i["expected_docs"]:
            print(f"MISMATCH {s}: docs={i['docs']} expected={i['expected_docs']}")


if __name__ == "__main__":
    main()
