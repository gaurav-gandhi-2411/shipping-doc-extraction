"""Production post-processing v1: the merge rules R1 / R2 / R3 of ``shipdoc.rules``, default ON.

``shipdoc.rules`` holds the three rules as pure functions (gated by ``scripts/rule_gate.py``, SHIP
decisions in ``reports/rule_gate.md``); this module is the pipeline wrapper around them:

* ``RuleConfig``: one boolean switch per rule, ALL TRUE by default; ``RuleConfig.all_off()`` is
  the old behaviour (no rule touches anything).
* ``apply_rules``: one MERGED + normalised document -> ``(doc, changes, skipped)``, running R1,
  R2, R3 in that order exactly as the gate does. It never raises on missing inputs and never fills
  when a rule's own preconditions fail. ``changes`` / ``skipped`` carry rule, field, row index,
  kind and reason ONLY, never a value (the per-doc ``rules`` record is safe to log or ship).
* R1 needs the model's own page-1 ``supplier_name`` (``supplier_from_trace``). R2 needs the OCR
  text of the waybill's pages (``ocr_text_for_doc``): without it R2 is SKIPPED, recorded with
  reason ``no_ocr`` (``ocr_incomplete`` when only some pages are cached). R3 needs the frozen slot
  shapes (``meta/slot_shapes.json``, ``scripts/freeze_slot_shapes.py``): without them R3 is
  skipped with reason ``no_shapes``.
* ``postprocess_traces``: ``trace.jsonl`` lines -> ``{doc_id: patched doc}`` + the per-doc records
  + the aggregate counts the submission manifest records.

Pipeline stage: AFTER ``merge_pages`` / ``normalize_doc`` and BEFORE ``coerce_doc`` /
``repair_doc`` (the rules compare values by emptiness and format shape only). A document that
already went through the coerce stage (a trace ``prediction``) is accepted too: the rules are
idempotent on their own output and coercion is idempotent, so the order does not change a result.

Test data policy: nothing here reads a label; the shapes artifact holds format shapes of TRAIN +
DEV gold, never a value. No function prints or returns an extracted value.
"""

from __future__ import annotations

import functools
import hashlib
import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from shipdoc import ocr, paths, rules
from shipdoc.replay import salvage_page_json

SHAPES_SCHEMA = 1
SHAPES_FILE = "slot_shapes.json"
RULE_NAMES = ("R1", "R2", "R3")
REASON_NO_OCR = "no_ocr"
REASON_OCR_INCOMPLETE = "ocr_incomplete"
REASON_NO_SHAPES = "no_shapes"
REASON_NO_HEADER = "no_header"
REASON_DISABLED = "disabled"
REASON_NOT_A_DOC = "not_a_document"

ShapesSource = rules.SlotShapes | Callable[[str], "rules.SlotShapes | None"] | None


@dataclass(frozen=True)
class RuleConfig:
    """Per-rule switches of post-processing v1. The default is ALL ON (the shipped behaviour)."""

    r1: bool = True
    r2: bool = True
    r3: bool = True

    @classmethod
    def all_off(cls) -> RuleConfig:
        """No rule runs: ``apply_rules`` returns the document unchanged (the pre-v1 behaviour)."""
        return cls(r1=False, r2=False, r3=False)

    def enabled(self, rule: str) -> bool:
        """Whether `rule` (``R1`` / ``R2`` / ``R3``) is switched on."""
        return bool(getattr(self, rule.lower()))

    def as_dict(self) -> dict[str, bool]:
        """``{"r1": .., "r2": .., "r3": ..}`` for manifests."""
        return asdict(self)


# --------------------------------------------------------------------------------------------
# Frozen R3 shapes artifact
# --------------------------------------------------------------------------------------------


def shapes_path() -> Path:
    """Default location of the frozen shapes artifact (``meta/slot_shapes.json``)."""
    return paths.REPO_ROOT / "meta" / SHAPES_FILE


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def shapes_payload(
    labels: Sequence[Mapping[str, Any]], folds_sha256: str, n_docs: int | None = None
) -> dict[str, Any]:
    """The frozen artifact content learned from gold `labels` (format shapes + counts only).

    Deterministic: shape lists are sorted, there is no timestamp and no commit id. Counts: docs
    the shapes were learned from, invoice docs among them, and the non-empty cpn / po cells.
    """
    learned = rules.learn_slot_shapes(labels)
    invoices = [g for g in labels if g.get("doc_type") == "invoice"]
    rows = [r for g in invoices for r in (g.get("line_items") or [])]
    return {
        "schema": SHAPES_SCHEMA,
        "rule": "R3 slot_shape",
        "shape_alphabet": "digits -> 9, ASCII letters -> A, everything else kept (rules.shape_of)",
        "learned_from": (
            "ALL train + dev gold docs of splits/folds.json (the shipping artefact; IN-SAMPLE for "
            "those 500 docs). The held-out evidence is the per-fold supplier-held-out gate "
            "(reports/rule_gate.md), not this file."
        ),
        "folds_sha256": folds_sha256,
        "n_docs": len(labels) if n_docs is None else n_docs,
        "n_invoice_docs": len(invoices),
        "n_cpn_cells": sum(not _empty(r.get(rules.CPN)) for r in rows),
        "n_po_cells": sum(not _empty(r.get(rules.PO)) for r in rows),
        "cpn_only": sorted(learned.cpn_only),
        "po_only": sorted(learned.po_only),
    }


def _empty(v: Any) -> bool:
    return v is None or (isinstance(v, str) and not v.strip())


def dump_shapes(payload: Mapping[str, Any]) -> str:
    """Canonical text of the artifact (sorted keys, LF, ascii-escaped): identical on every OS."""
    return json.dumps(dict(payload), indent=1, sort_keys=True) + "\n"


def load_slot_shapes(path: Path | None = None) -> tuple[rules.SlotShapes, str]:
    """``(SlotShapes, sha256 of the file)`` of the frozen artifact.

    Raises FileNotFoundError / ValueError (schema or fields) rather than guessing: a caller that
    wants "no shapes" passes ``shapes=None`` to ``apply_rules`` instead of swallowing this.
    """
    p = shapes_path() if path is None else Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if data.get("schema") != SHAPES_SCHEMA or not all(
        isinstance(data.get(k), list) for k in ("cpn_only", "po_only")
    ):
        raise ValueError(f"{p.name}: not a schema-{SHAPES_SCHEMA} slot shapes file")
    shapes = rules.SlotShapes(
        cpn_only=frozenset(map(str, data["cpn_only"])), po_only=frozenset(map(str, data["po_only"]))
    )
    return shapes, sha256_file(p)


@functools.lru_cache(maxsize=1)
def default_shapes() -> rules.SlotShapes:
    """The frozen ``meta/slot_shapes.json`` shapes, read once per process (raises if unusable)."""
    return load_slot_shapes()[0]


# --------------------------------------------------------------------------------------------
# Inputs the rules need, taken from a trace line / the OCR cache
# --------------------------------------------------------------------------------------------


def supplier_from_trace(trace: Mapping[str, Any]) -> Any:
    """The model's own page-1 parsed ``header.supplier_name`` (None when absent).

    Same source as the gate (``scripts/rule_gate.py``): the page's ``parsed`` dict, else the
    salvage parse of its ``raw_text``. R1 never looks at the merged document for this value.
    """
    pages = trace.get("pages") or []
    if not pages or not isinstance(pages[0], Mapping):
        return None
    first = pages[0].get("parsed")
    if not isinstance(first, dict):
        first = salvage_page_json(str(pages[0].get("raw_text") or ""))
    hdr = first.get("header") if isinstance(first, dict) else None
    return hdr.get("supplier_name") if isinstance(hdr, dict) else None


def ocr_text_for_doc(
    doc_id: str, n_pages: int | None = None, cache_root: Path | None = None
) -> tuple[str | None, str | None]:
    """``(OCR text of the doc, None)`` or ``(None, reason)`` when the cache cannot serve R2.

    The text is the gate's: ``ocr.page_text`` of every cached page in page order, joined with a
    newline. Reasons: ``no_ocr`` (nothing cached, or the cache folder is missing),
    ``ocr_incomplete`` (fewer cached pages than the document has: fail closed rather than let a
    pattern match on a partial text decide a field).
    """
    try:
        pages = ocr.doc_pages(doc_id, cache_root)
    except OSError:
        return None, REASON_NO_OCR
    if not pages:
        return None, REASON_NO_OCR
    if n_pages is not None and len(pages) < n_pages:
        return None, REASON_OCR_INCOMPLETE
    return "\n".join(ocr.page_text(p) for p in pages), None


# --------------------------------------------------------------------------------------------
# The rules
# --------------------------------------------------------------------------------------------


def _record(c: rules.Change) -> dict[str, Any]:
    """Value-free record of one rule change: rule, field, row index, kind."""
    if c.rule == "R3":
        kind = "swap" if not _empty(c.before[0]) and not _empty(c.before[1]) else "move"
    else:
        kind = "fill"
    return {"rule": c.rule, "field": c.field, "row": c.row, "kind": kind}


def _skip(rule: str, reason: str) -> dict[str, str]:
    return {"rule": rule, "reason": reason}


def apply_rules(
    merged_doc: Mapping[str, Any],
    *,
    supplier_name: Any,
    ocr_text: str | None,
    shapes: rules.SlotShapes | None,
    cfg: RuleConfig | None = None,
    ocr_reason: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, str]]]:
    """R1, R2, R3 (in that order, as the gate) on one merged document; the input is not mutated.

    Returns ``(doc, changes, skipped)``. `changes` are value-free records
    ``{"rule", "field", "row", "kind"}`` (kind ``fill`` / ``move`` / ``swap``; `row` is None for
    header cells, a row index for R3). `skipped` are ``{"rule", "reason"}`` for a rule that could
    not run on a document it applies to: ``disabled`` (switch off, recorded once per doc the
    rule would apply to), ``no_ocr`` / ``ocr_incomplete`` (R2, waybill, `ocr_text` is None;
    `ocr_reason` picks the label), ``no_shapes`` (R3, invoice, `shapes` is None), ``no_header``.
    A rule whose own precondition fails (carrier already set, no unique pattern match, shape
    ambiguous, `supplier_name` empty) simply makes no change: that is not a skip.
    Never raises on missing inputs; a non-mapping `merged_doc` comes back unchanged, all skipped.
    """
    cfg = RuleConfig() if cfg is None else cfg
    if not isinstance(merged_doc, Mapping):
        return merged_doc, [], [_skip(r, REASON_NOT_A_DOC) for r in RULE_NAMES]  # type: ignore[return-value]
    doc: dict[str, Any] = dict(merged_doc)
    changes: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    doc_type = doc.get("doc_type")
    waybill_ok = doc_type == "waybill" and isinstance(doc.get("header"), Mapping)
    if doc_type == "waybill" and not waybill_ok:
        skipped += [_skip(r, REASON_NO_HEADER) for r in ("R1", "R2") if cfg.enabled(r)]

    def run(fn: Callable[[dict[str, Any]], tuple[dict[str, Any], list[rules.Change]]]) -> None:
        nonlocal doc
        doc, made = fn(doc)
        changes.extend(_record(c) for c in made)

    if doc_type == "waybill" and not cfg.r1:
        skipped.append(_skip("R1", REASON_DISABLED))
    elif waybill_ok:
        run(lambda d: rules.carrier_from_supplier(d, supplier_name))
    if doc_type == "waybill" and not cfg.r2:
        skipped.append(_skip("R2", REASON_DISABLED))
    elif waybill_ok:
        if ocr_text is None:
            skipped.append(_skip("R2", ocr_reason or REASON_NO_OCR))
        else:
            run(lambda d: rules.pattern_backfill(d, ocr_text))
    if doc_type == "invoice" and not cfg.r3:
        skipped.append(_skip("R3", REASON_DISABLED))
    elif doc_type == "invoice":
        if shapes is None:
            skipped.append(_skip("R3", REASON_NO_SHAPES))
        else:
            run(lambda d: rules.slot_shape(d, shapes))
    return doc, changes, skipped


# --------------------------------------------------------------------------------------------
# Whole runs
# --------------------------------------------------------------------------------------------


def postprocess_traces(
    traces: Sequence[Mapping[str, Any]],
    cfg: RuleConfig | None = None,
    shapes: ShapesSource = None,
    ocr_root: Path | None = None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Apply the rules to every trace line's ``prediction``.

    Returns ``(predictions {doc_id: doc}, records, summary)``. `records` is one entry per doc,
    ``{"doc_id", "changes", "skipped"}`` (value-free, sorted by doc id). `shapes` is a
    ``SlotShapes`` (one set for every doc), a callable ``doc_id -> SlotShapes | None`` (the
    honest per-fold variant of the replay check) or None (R3 skipped). The OCR text for R2 is
    read from `ocr_root` (default: the configured OCR cache) only for waybills and only when R2
    is on. The returned docs are NOT coerced: run ``coerce_predictions`` / ``repair_predictions``
    next, exactly as ``predict.assemble_submission`` does.

    `summary`: switches, per-rule ``eligible`` docs (rule on, doc type applies, inputs present),
    ``touched`` docs, ``changes`` count, ``skipped`` counts keyed ``R2/no_ocr`` etc.
    """
    cfg = RuleConfig() if cfg is None else cfg
    preds: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    eligible: Counter[str] = Counter()
    touched: Counter[str] = Counter()
    n_changes: Counter[str] = Counter()
    skipped_n: Counter[str] = Counter()
    for t in sorted(traces, key=lambda x: x["doc_id"]):
        d = t["doc_id"]
        doc = t["prediction"]
        is_wb = isinstance(doc, Mapping) and doc.get("doc_type") == "waybill"
        text, why = (None, None)
        if cfg.r2 and is_wb:
            text, why = ocr_text_for_doc(d, len(t.get("pages") or []) or None, ocr_root)
        sh = shapes(d) if callable(shapes) else shapes
        out, changes, skipped = apply_rules(
            doc, supplier_name=supplier_from_trace(t), ocr_text=text, shapes=sh, cfg=cfg,
            ocr_reason=why,
        )  # fmt: skip
        preds[d] = out
        records.append({"doc_id": d, "changes": changes, "skipped": skipped})
        skip_rules = {s["rule"] for s in skipped}
        dt = doc.get("doc_type") if isinstance(doc, Mapping) else None
        for r, kind in (("R1", "waybill"), ("R2", "waybill"), ("R3", "invoice")):
            if cfg.enabled(r) and dt == kind and r not in skip_rules:
                eligible[r] += 1
        for s in skipped:
            skipped_n[f"{s['rule']}/{s['reason']}"] += 1
        for r in RULE_NAMES:
            mine = [c for c in changes if c["rule"] == r]
            n_changes[r] += len(mine)
            touched[r] += bool(mine)
    summary = {
        "n_docs": len(records),
        "switches": cfg.as_dict(),
        "eligible_docs": {r: eligible[r] for r in RULE_NAMES},
        "touched_docs": {r: touched[r] for r in RULE_NAMES},
        "changes": {r: n_changes[r] for r in RULE_NAMES},
        "skipped": dict(sorted(skipped_n.items())),
    }
    return preds, records, summary
