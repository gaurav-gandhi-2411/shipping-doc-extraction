"""Freeze the R3 slot shapes into ``meta/slot_shapes.json`` (the shipping artefact).

Shapes (digits -> 9, ASCII letters -> A, punctuation kept) are learned with
``rules.learn_slot_shapes`` from the gold of ALL 500 train + dev documents of
``splits/folds.json``. The file holds format shapes, the counts of docs / rows they came from and
the sha256 of ``splits/folds.json``; no gold value. It is deterministic (sorted lists, no
timestamp), so re-running on the same gold + folds rewrites identical bytes.

HONESTY NOTE: the artefact is IN-SAMPLE for those 500 documents (it learned from them), so any
score of R3 on them with this file is optimistic. The held-out evidence is the supplier-held-out,
per-fold gate in ``reports/rule_gate.md`` (shapes of fold k learned only from the other folds'
gold); the replay check reports both variants (``reports/v1_replay.md``).

Run: ``uv run python scripts/freeze_slot_shapes.py [--check]``. ``--check`` writes nothing and
exits 1 when the committed file differs from a fresh freeze (a drift check for CI / a verifier).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from shipdoc import meta as meta_mod
from shipdoc import postrules

ROOT = Path(__file__).resolve().parents[1]
FOLDS = ROOT / "splits" / "folds.json"
EXPECTED_DOCS = 500


def folds_sha256(folds_path: Path) -> str:
    """sha256 of the folds file with CRLF read as LF.

    The repo stores the file with LF endings (.gitattributes), but a Windows working copy may be
    CRLF; hashing the raw bytes made ``--check`` pass on one and fail on the other (found by the
    second verifier, 2026-10-03).
    """
    return hashlib.sha256(folds_path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def freeze(folds_path: Path = FOLDS) -> str:
    """Canonical text of the artefact learned from the train + dev gold docs of `folds_path`."""
    doc_fold = json.loads(folds_path.read_text(encoding="utf-8"))["doc_fold"]
    labels = [
        g for g in meta_mod.load_labels("train") + meta_mod.load_labels("dev")
        if g["doc_id"] in doc_fold
    ]  # fmt: skip
    if len(labels) != len(doc_fold) or len(labels) != EXPECTED_DOCS:
        # fail closed: a partial gold set would freeze different shapes without any warning
        raise SystemExit(
            f"expected {EXPECTED_DOCS} gold docs, found {len(labels)} of {len(doc_fold)} fold docs"
        )
    return postrules.dump_shapes(postrules.shapes_payload(labels, folds_sha256(folds_path)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=postrules.shapes_path())
    ap.add_argument("--check", action="store_true", help="compare with the file, write nothing")
    a = ap.parse_args(argv)
    text = freeze()
    if a.check:
        same = a.out.is_file() and a.out.read_text(encoding="utf-8") == text
        print(f"{a.out.name}: {'identical to a fresh freeze' if same else 'DIFFERS or missing'}")
        return 0 if same else 1
    a.out.write_text(text, encoding="utf-8", newline="\n")
    data = json.loads(text)
    print(
        f"wrote {a.out} (sha256 {postrules.sha256_file(a.out)}): {data['n_docs']} docs, "
        f"{data['n_invoice_docs']} invoices, {len(data['cpn_only'])} cpn-only and "
        f"{len(data['po_only'])} po-only shapes"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
