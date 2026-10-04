"""Command-line entry point for shipdoc."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from shipdoc import paths


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser (subcommands are added in later steps)."""
    parser = argparse.ArgumentParser(
        prog="shipdoc", description="Shipping-document extraction pipeline."
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    ocr = sub.add_parser("ocr", help="Run the cached CPU OCR pass over page images.")
    ocr.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    ocr.add_argument("--variant", choices=["server", "mobile"], default="server")
    ocr.add_argument("--cpu-threads", type=int, default=8)
    ocr.add_argument("--data-root", type=Path, default=None, help="default: SHIPDOC_DATA_DIR")
    ocr.add_argument("--cache-root", type=Path, default=None, help="default: SHIPDOC_OCR_CACHE")
    ocr.add_argument("--limit", type=int, default=None, help="first N pages per split (debug)")
    spike = sub.add_parser(
        "spike", help="Zero-shot VLM spike: extract -> merge -> normalize -> score."
    )
    spike.add_argument(
        "--config", type=Path, required=True, help="configs/spike_<model>_<arm>.yaml"
    )
    spike.add_argument(
        "--docs", required=True, help="path to a JSON list of doc_ids (or inline JSON)"
    )
    spike.add_argument(
        "--split", default="dev", help="a split, or several joined by '+' (train+dev)"
    )
    spike.add_argument("--run-id", required=True)
    spike.add_argument(
        "--logprobs",
        action="store_true",
        help="record per-token logprobs and per-field min/mean in the trace (keyed format only)",
    )
    spike.add_argument("--resume", action="store_true", help="skip doc_ids already in trace.jsonl")
    spike.add_argument("--backend", choices=["hf", "mock"], default="hf")
    spike.add_argument("--limit", type=int, default=None, help="first N doc_ids (debug)")
    spike.add_argument(
        "--wandb", action="store_true", help="log metrics to W&B (needs WANDB_API_KEY)"
    )
    spike.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="pages per generate call; default: 1 for a new run, the stored size on --resume",
    )
    spike.add_argument(
        "--shard",
        default="0/1",
        help="document-level shard 'i/K' (default 0/1 = everything); writes <run-id>_shard<i>of<K>",
    )
    spike.add_argument(
        "--bench-result",
        type=Path,
        default=None,
        help="bench_result.json whose chosen batch size this run uses (recorded in the manifest)",
    )
    bench = sub.add_parser(
        "bench", help="Pick the batch size: 12 dev pages at batch 1/2/4/8 (identity, VRAM, speed)."
    )
    bench.add_argument("--config", type=Path, help="configs/spike_<model>_<arm>.yaml")
    bench.add_argument("--docs", default="splits/bench12.json", help="bench doc list (dev docs)")
    bench.add_argument("--out-dir", type=Path, help="folder for the per-size runs + bench_result")
    bench.add_argument("--batch-sizes", default="1,2,4,8", help="comma-separated; must include 1")
    bench.add_argument("--backend", choices=["hf", "mock"], default="hf")
    bench.add_argument("--reuse", action="store_true", help="reuse a valid bench_result.json")
    bench.add_argument(
        "--write-list", type=Path, default=None, help="only write the bench doc list here and exit"
    )
    merge = sub.add_parser(
        "merge-shards", help="Merge the shard folders of a run into the unsharded run folder."
    )
    merge.add_argument("--config", type=Path, required=True)
    merge.add_argument("--docs", required=True, help="the full doc list the shards were cut from")
    merge.add_argument("--split", default="dev")
    merge.add_argument("--run-id", required=True, help="the unsharded run id (the output folder)")
    merge.add_argument("--shards", type=int, required=True, help="K")
    merge.add_argument("--expected-docs", type=int, default=None)
    merge.add_argument("--expected-pages", type=int, default=None)
    replay = sub.add_parser(
        "replay",
        help="Replay parse -> merge -> normalize -> score on SAVED spike raw outputs (no model).",
    )
    replay.add_argument("--run-dir", type=Path, required=True, help="dir with trace.jsonl")
    replay.add_argument(
        "--config", type=Path, required=True, help="configs/spike_*.yaml (output format)"
    )
    replay.add_argument("--split", default="dev")
    replay.add_argument("--labels", type=Path, default=None, help="default: data/<split>/labels")
    replay.add_argument(
        "--drop-null-rows",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="drop rows with every field null at merge (default: the merge default)",
    )
    replay.add_argument(
        "--salvage-truncated",
        action="store_true",
        help="rebuild pages cut at max_new_tokens from their last complete container",
    )
    replay.add_argument("--out", type=Path, default=None, help="also write the result JSON here")
    syn = sub.add_parser(
        "synth-redaction",
        help="SYNTHETIC redaction eval: regenerate the occluded dev variants from their recipes.",
    )
    syn.add_argument(
        "--recipes", type=Path, default=None, help="default: splits/synthetic_redaction_dev.json"
    )
    syn.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="default: SHIPDOC_RUNS_DIR/synthetic_redaction_dev",
    )
    syn.add_argument("--materialize", action="store_true", help="write images, labels, manifest")
    syn.add_argument(
        "--table", action="store_true", help="print the field x method x scanned/digital table"
    )
    pred = sub.add_parser(
        "predict",
        help="Test-set prediction stages for the safety submission (notebooks/04_predict_test).",
    )
    ps = pred.add_subparsers(dest="stage", required=True, metavar="<stage>")

    def stage(name: str, help_: str, config: bool = True) -> argparse.ArgumentParser:
        p = ps.add_parser(name, help=help_)
        if config:
            p.add_argument("--config", type=Path, required=True)
        p.add_argument("--runs-root", type=Path, default=None, help="default: SHIPDOC_RUNS_DIR")
        p.add_argument("--data-root", type=Path, default=None, help="default: SHIPDOC_DATA_DIR")
        return p

    def backend_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--backend", choices=["hf", "mock"], default="hf")
        p.add_argument(
            "--mock-gold",
            type=Path,
            default=None,
            help="mock backend only: labels dir of FAKE test docs (dev copies) to replay",
        )

    p = stage("plan", "Doc ids and page counts from the test image folder names.", False)
    p.add_argument("--shard", default="0/1")
    p.add_argument("--out", type=Path, default=None, help="write the plan JSON here (ids only)")
    p = stage("smoke", "Smoke gate on 5 dev docs (blocks the run stage on failure).")
    backend_args(p)
    p.add_argument("--docs", default="splits/smoke5.json")
    p.add_argument("--run-id", required=True)
    p.add_argument("--status", type=Path, required=True, help="smoke status JSON to write")
    p = stage("batch", "Batch size: reuse/run the bench, enforce the dev-run batch-size contract.")
    backend_args(p)
    p.add_argument("--bench-docs", default="splits/bench12.json")
    p.add_argument("--bench-dir", type=Path, required=True)
    p.add_argument("--dev-run-dir", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=None, help="manual size (skips the bench)")
    p.add_argument("--decision", type=Path, required=True, help="decision JSON to write")
    for name, help_ in (
        ("run", "Resumable per-document test run (refuses unless the smoke gate passed)."),
        ("determinism", "Second pass over 5 seeded test docs; must be byte-identical."),
    ):
        p = stage(name, help_)
        backend_args(p)
        p.add_argument("--run-id", required=True, help="the unsharded run id")
        p.add_argument("--shard", default="0/1")
        p.add_argument("--decision", type=Path, required=True)
        if name == "run":
            p.add_argument("--smoke-status", type=Path, required=True)
            p.add_argument("--expect-docs", type=int, default=None)
            p.add_argument("--expect-pages", type=int, default=None)
    p = stage("assemble", "Validate a finished test run and write the submission folder.")
    p.add_argument("--run-id", required=True, help="the unsharded (or merged) run id")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--sha7", default=None, help="folder v0_<sha7> in SHIPDOC_SUBMISSIONS_DIR")
    p.add_argument("--schema", type=Path, required=True)
    p.add_argument("--dev-run-dir", type=Path, default=None)
    p.add_argument("--expect-docs", type=int, default=200)
    p.add_argument("--expect-pages", type=int, default=280)
    p.add_argument("--expect-code-sha", default=None, help="the pinned 40-hex code SHA")
    p.add_argument("--allow-dirty", action="store_true", help="tests only: accept a +dirty SHA")
    p.add_argument("--require-stack", action="store_true", help="real model + recorded versions")
    p.add_argument("--smoke-status", type=Path, default=None)
    p.add_argument("--sample-submission", type=Path, default=None)
    p.add_argument(
        "--no-rule",
        action="append",
        choices=["R1", "R2", "R3"],
        default=[],
        help="switch a post-processing v1 rule OFF (default: R1, R2, R3 all on); repeatable",
    )
    p.add_argument("--ocr-cache", type=Path, default=None, help="OCR cache root for R2")
    p.add_argument("--shapes-file", type=Path, default=None, help="frozen R3 shapes (meta/)")
    p = stage("validate", "Schema + id-set check of any predictions file.", False)
    p.add_argument("--pred", type=Path, required=True)
    p.add_argument("--schema", type=Path, required=True)
    p.add_argument("--report-out", type=Path, default=None)
    oof = sub.add_parser(
        "oof",
        help="OOF inference of a fold adapter + paired comparison with zero-shot (notebook 05).",
    )
    os_ = oof.add_subparsers(dest="stage", required=True, metavar="<stage>")
    for name, help_ in (
        ("verify", "Adapter manifest verification (fail closed, prints the table)."),
        ("infer", "Verify, merge the adapter, batch guard, resumable inference on the fold."),
        ("estimate", "ESTIMATE of T4 hours and CU (prints only; reads no adapter)."),
        ("compare", "Score the OOF run and compare it with zero-shot; interim G4 verdict."),
    ):
        p = os_.add_parser(name, help=help_)
        p.add_argument("--fold", type=int, required=True, choices=[0, 1, 2])
        p.add_argument("--config", type=Path, required=True, help="the production spike config")
        p.add_argument("--zs-run-dir", type=Path, required=True, help="the 02 zero-shot run folder")
        p.add_argument("--folds", type=Path, default=paths.REPO_ROOT / "splits" / "folds.json")
        p.add_argument("--data-root", type=Path, default=None, help="default: SHIPDOC_DATA_DIR")
        p.add_argument("--split", default="train+dev")
        if name != "estimate":
            p.add_argument("--out-dir", type=Path, required=True, help="oof_fold<k>_<sha7> folder")
        if name in ("verify", "infer"):
            p.add_argument("--adapter-dir", type=Path, required=True, help="<fold run>/final")
            p.add_argument("--pin", default=None, help="this notebook's pinned 40-hex code SHA")
        if name in ("infer", "estimate"):
            p.add_argument("--batch-size", type=int, default=None, help="default: the 02 size")
        if name == "infer":
            p.add_argument("--bench-docs", default="splits/bench12.json")
            p.add_argument("--wandb", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI; returns the process exit code."""
    paths.apply_env()  # before anything can import paddle or transformers
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
    elif args.command == "ocr":
        # Imported lazily: the paddle stack is an optional dependency group.
        from shipdoc.ocr import PaddleOcrEngine, run_ocr

        engine = PaddleOcrEngine(variant=args.variant, cpu_threads=args.cpu_threads)
        manifest = run_ocr(engine, args.splits, args.data_root, args.cache_root, args.limit)
        print(
            f"cached {manifest['pages_cached']}/{manifest['pages_expected']} pages; "
            f"{len(manifest['failures'])} failures"
        )
        return 1 if manifest["failures"] else 0
    elif args.command == "spike":
        # Imported lazily so `ocr`/`--help` never pay for numpy/yaml/scoring imports.
        from shipdoc.spike import main_spike

        return main_spike(args)
    elif args.command == "bench":
        from shipdoc.bench import main_bench

        if not args.write_list and not (args.config and args.out_dir):
            parser.error("bench needs --config and --out-dir (or --write-list)")
        return main_bench(args)
    elif args.command == "merge-shards":
        from shipdoc.shardmerge import main_merge

        return main_merge(args)
    elif args.command == "replay":
        from shipdoc.replay import main_replay

        return main_replay(args)
    elif args.command == "predict":
        from shipdoc.predict import main_predict

        return main_predict(args)
    elif args.command == "oof":
        from shipdoc.oof import main_oof

        return main_oof(args)
    elif args.command == "synth-redaction":
        from shipdoc.augment import main_synth_redaction

        return main_synth_redaction(args)
    return 0
