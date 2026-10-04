"""Re-assemble ``test_predictions.json`` from a submission's ``trace.jsonl`` and print its sha256.

The post-processing of a submission (page merge, rules R1 to R3, coercion, the one date repair) is
CPU-deterministic given the trace, the frozen ``meta/slot_shapes.json`` and the OCR cache that R2
reads. This script runs exactly that production path (``shipdoc.flags.post_rule_output``) and the
production writer, in memory and in a temporary folder, and compares the bytes with an expected
sha256: the check that a repository checkout reproduces the post-processing of a submission. It
never reads an image and never calls a model.

Needs the evaluators' ``schema.json`` (the date repair reads it): point ``SHIPDOC_ASSIGNMENT_DIR``
at the folder that holds it.

Run: uv run python scripts/public_equivalence.py --trace <v15 folder>/trace.jsonl
        --ocr-cache <ocr_cache_test folder> --expect-sha256 <hex>

Exit code: 0 the sha256 equals ``--expect-sha256`` (or none was given); 1 it differs; 2 usage /
input error.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHAPES = ROOT / "meta" / "slot_shapes.json"


def assemble_sha256(trace: Path, ocr_cache: Path, shapes: Path = DEFAULT_SHAPES) -> tuple[str, int]:
    """``(sha256 of the predictions file the production path writes, number of documents)``."""
    from shipdoc import flags, predict

    traces = flags._read_traces(trace)
    if not traces:
        raise ValueError(f"no traced document in {trace}")
    final, _changes = flags.post_rule_output(traces, shapes, ocr_cache)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "test_predictions.json"
        predict._write_json(out, final)  # production order: as the traces appear, never sorted
        return hashlib.sha256(out.read_bytes()).hexdigest(), len(final)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trace", type=Path, required=True, help="trace.jsonl of the submission")
    ap.add_argument("--ocr-cache", type=Path, required=True, help="folder with paddleocr/test/")
    ap.add_argument("--shapes", type=Path, default=DEFAULT_SHAPES, help="slot_shapes.json")
    ap.add_argument("--expect-sha256", default=None, help="hex digest the result must equal")
    args = ap.parse_args(argv)
    try:
        digest, n_docs = assemble_sha256(args.trace, args.ocr_cache, args.shapes)
    except (OSError, ValueError, KeyError) as exc:
        print(f"public equivalence FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    import shipdoc

    print(f"shipdoc imported from: {Path(shipdoc.__file__).parent}")
    print(f"documents: {n_docs}; sha256: {digest}")
    if args.expect_sha256 is None:
        return 0
    same = digest == args.expect_sha256.lower()
    print(f"expected sha256: {args.expect_sha256.lower()} -> {'MATCH' if same else 'MISMATCH'}")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
