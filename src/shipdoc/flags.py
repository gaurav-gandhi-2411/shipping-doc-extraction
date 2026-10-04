"""Calibration review flags for a submission: P(correct) per field, P(document fully correct).

Per GG's closing plan (spec section 11, item 6) the flags live in a SEPARATE file,
``review_flags.json``, never in ``test_predictions.json`` (its schema has
``additionalProperties: false``). The file holds probabilities, booleans and field NAMES only: no
extracted value, no OCR text.

Pieces (all CPU, no model):

``Calibrator``      a frozen calibrator artifact (``meta/calibrator_<arm>.json``, written by
                    ``scripts/freeze_calibrator.py``): per-field-type (or pooled) models, the
                    document-level model, the layout-novelty reference and the per-field-type tau
                    thresholds. LR models are stored as plain coefficients (no sklearn at
                    inference); a GBM is a base64 pickle with its sklearn version and sha256 and is
                    loaded ONLY under the recorded sklearn version, else REFUSED.
``build_tables`` /  the feature builder. THE SAME functions run at calibration time (freeze) and at
``design``          inference time: they call ``confidence_v3.build_table_v2`` (v1 + v2 features),
                    ``align_all`` / ``agreement_matrix`` (fine-tuned arm) and the v2 / v3 design
                    matrices. The only difference is the id guard (`id_guard`): the calibration
                    code refuses ``test_`` ids by design, the inference path accepts ONLY
                    ``test_`` ids; a list that mixes them is refused.
``parity_probe``    a fixed synthetic document pushed through that code; its design-matrix digest is
                    stored in the artifact and re-checked at load, so a behavioural change of the
                    feature builder after the freeze REFUSES the calibrator instead of silently
                    mis-scoring.
``compute_flags``   tau thresholds -> accept / review per emitted field, auto-accept per document.
``python -m shipdoc.flags``  the stage CLI. Works for a ZS+rules (v1) submission with the ZS
                    calibrator, and for a fine-tuned submission with the FT calibrator (which needs
                    the zero-shot traces of the same documents under the strict reuse rules of
                    ``shipdoc.reuse.decide_reuse``; refused otherwise: no ZS inference here).

What the thresholds mean. ``tau`` of a field type is the lowest confidence whose accepted set has
precision >= the target ON ALL POOLED OUT-OF-FOLD PREDICTIONS of that field type
(``confidence.select_tau``); a type whose pooled OOF predictions never reach the target has
``tau = null`` and every field of it is sent to review. The production calibrator is fitted on all
OOF rows, so its probabilities are a little different from the cross-fitted ones tau was chosen on;
the nested (honest) estimates of precision / coverage are stored next to the thresholds.

Null fields: the calibrators score EMITTED (non-null) fields only (calibration v2 has no null
policy), so a null field carries ``emitted: false, p_correct: null, flag: null``.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import json
import pickle  # noqa: S403 - GBM models only, after a sha256 + sklearn version check (see Model)
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from shipdoc import confidence as cf
from shipdoc import confidence_v2 as c2
from shipdoc import confidence_v3 as c3
from shipdoc import eval as ev
from shipdoc import paths, postrules, reuse, spike
from shipdoc import predict as pr
from shipdoc.coerce import coerce_predictions, repair_predictions
from shipdoc.ocr import OcrItem, PageOcr
from shipdoc.postrules import RuleConfig

ARTIFACT_SCHEMA = 1
FLAGS_SCHEMA = 1
ARTIFACT_KIND = "shipdoc_calibrator"
FLAGS_KIND = "review_flags"
ARMS = ("zs", "ft")
#: Precision targets whose tau is stored (field: calibration v2 `TARGETS`; document: 95 / 98%).
FIELD_TARGETS: tuple[float, ...] = c2.TARGETS
DOC_TARGETS: tuple[float, ...] = (0.95, 0.98)
#: The primary target of calibration v2 (`scripts/calibrate_v2.py` PRIMARY); the flags default.
PRIMARY_TARGET = 0.98
FLAGS_NAME = "review_flags.json"
ACCEPT, REVIEW = "accept", "review"
LORA_MARK = "+lora:"  # `oof.MergedHfBackend.model_id` of a fine-tuned run
#: Strings a flags file may hold as VALUES under ``docs`` (everything else: a number / bool).
DOC_STRING_VALUES = frozenset({ACCEPT, REVIEW})


class FlagsError(RuntimeError):
    """A flags stage refused or an artifact failed a check. Messages never quote a value."""


def tkey(target: float) -> str:
    """Canonical JSON key of a precision target (``0.98`` -> ``"0.98"``)."""
    return f"{target:.2f}"


def sha256_bytes(data: bytes) -> str:
    """Hex sha256 of `data`."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file."""
    return sha256_bytes(Path(path).read_bytes())


# --------------------------------------------------------------------------------------------
# Models: fit / serialise / predict
# --------------------------------------------------------------------------------------------


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.asarray(1.0 / (1.0 + np.exp(-np.clip(z, -700.0, 700.0))))


def sklearn_version() -> str:
    """Installed scikit-learn version."""
    import sklearn

    return str(sklearn.__version__)


def fit_model(kind: str, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Fit one calibrator exactly as ``confidence.fit_predict`` does and return its JSON spec.

    Same rules: a fit set with fewer than ``MIN_PER_CLASS`` rows of a class becomes a constant
    (prior) model; ``lr`` = mean imputer + standard scaler + plain logistic regression
    (`confidence.make_model`), stored as numbers; ``gbm`` drops the columns that are NaN in every
    fitting row (what `fit_predict` does) and is stored as a pickle with its sklearn version.
    """
    y = np.asarray(y).astype(int)
    pos = int(y.sum())
    if pos < cf.MIN_PER_CLASS or len(y) - pos < cf.MIN_PER_CLASS:
        return {"kind": "constant", "p": pos / max(len(y), 1), "n_fit": int(len(y))}
    model = cf.make_model(kind)
    if kind == "lr":
        model.fit(X, y)
        imp, scaler, lr = model[0], model[1], model[2]
        # keep_empty_features=True: an all-NaN fitting column is imputed with 0 (statistics_ is NaN)
        fill = np.where(np.isnan(imp.statistics_), 0.0, imp.statistics_)
        return {
            "kind": "lr",
            "n_fit": int(len(y)),
            "n_features": int(X.shape[1]),
            "fill": fill.tolist(),
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "coef": lr.coef_[0].tolist(),
            "intercept": float(lr.intercept_[0]),
        }
    keep = ~np.isnan(X).all(axis=0)
    model.fit(X[:, keep], y)
    blob = pickle.dumps(model, protocol=4)
    return {
        "kind": "gbm",
        "n_fit": int(len(y)),
        "n_features": int(X.shape[1]),
        "keep": keep.tolist(),
        "sklearn_version": sklearn_version(),
        "pickle_sha256": sha256_bytes(blob),
        "pickle_b64": base64.b64encode(blob).decode("ascii"),
    }


class Model:
    """A fitted calibrator loaded from its JSON spec; ``predict`` gives P(y = 1) per row."""

    def __init__(self, spec: Mapping[str, Any]) -> None:
        self.spec = spec
        self.kind = str(spec.get("kind"))
        self._gbm: Any = None
        if self.kind == "gbm":
            have, want = sklearn_version(), spec.get("sklearn_version")
            if have != want:
                raise FlagsError(
                    f"REFUSED: a GBM calibrator was pickled with scikit-learn {want}, this "
                    f"environment has {have}. A pickle is only loaded under the recorded version: "
                    "use that version or re-freeze the calibrator."
                )
            blob = base64.b64decode(str(spec["pickle_b64"]))
            if sha256_bytes(blob) != spec.get("pickle_sha256"):
                raise FlagsError("REFUSED: the GBM pickle does not match its recorded sha256")
            self._gbm = pickle.loads(blob)  # noqa: S301 - hash- and version-checked just above
        elif self.kind not in ("lr", "constant"):
            raise FlagsError(f"REFUSED: unknown model kind {self.kind!r}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """P(y = 1) of every row of `X` (float64 array)."""
        s = self.spec
        if self.kind == "constant":
            return np.full(len(X), float(s["p"]))
        if X.shape[1] != int(s["n_features"]):
            raise FlagsError(
                f"design has {X.shape[1]} columns, the model was fitted on {s['n_features']}"
            )
        if self.kind == "lr":
            Xf = np.where(np.isnan(X), np.asarray(s["fill"]), X)
            z = ((Xf - np.asarray(s["mean"])) / np.asarray(s["scale"])) @ np.asarray(s["coef"])
            return _sigmoid(z + float(s["intercept"]))
        keep = np.asarray(s["keep"], dtype=bool)
        return np.asarray(self._gbm.predict_proba(X[:, keep])[:, 1])


# --------------------------------------------------------------------------------------------
# The id guard and the feature builder (one code path for calibration and inference)
# --------------------------------------------------------------------------------------------


def _assert_all_test(doc_ids: Sequence[str]) -> None:
    bad = [d for d in doc_ids if not str(d).startswith(c2.TEST_PREFIX)]
    if bad:
        raise c2.NoTestDataError(
            f"{len(bad)} non-test id(s) reached the INFERENCE path of the flags stage; refusing"
        )


@contextlib.contextmanager
def id_guard(doc_ids: Sequence[str]) -> Iterator[str]:
    """Run the calibration code (which refuses ``test_`` ids) on ids of one kind only.

    No ``test_`` id: the calibration guard stays as is (``"calibration"``). Only ``test_`` ids: the
    guard is swapped for its inverse for the duration (it then refuses anything that is NOT a test
    id; ``"inference"``) and always restored. A mix of both is refused.
    """
    ids = [str(d) for d in doc_ids]
    n_test = sum(d.startswith(c2.TEST_PREFIX) for d in ids)
    if n_test == 0:
        yield "calibration"
        return
    if n_test != len(ids):
        raise FlagsError(f"{n_test} test and {len(ids) - n_test} non-test ids in one call: refused")
    original = c2.assert_no_test_ids
    c2.assert_no_test_ids = _assert_all_test
    try:
        yield "inference"
    finally:
        c2.assert_no_test_ids = original


@dataclass
class Tables:
    """Feature tables of one arm's post-rule output: v2 table, layout signatures, agreement."""

    tv: c2.TableV2
    sigs: dict[str, np.ndarray | None]
    agree: np.ndarray | None  # (n rows, len(AGREE_FEATURES)) for the fine-tuned arm, else None
    doc_ids: list[str] = field(default_factory=list)


def build_tables(
    arm: str,
    doc_ids: Sequence[str],
    post: Mapping[str, Any],
    traces: Mapping[str, Any],
    ocr: Mapping[str, Sequence[PageOcr]],
    changes: Mapping[str, Sequence[Mapping[str, Any]]],
    zs_post: Mapping[str, Any] | None = None,
) -> Tables:
    """v2 feature table (+ agreement with the zero-shot arm for ``ft``) of the given documents.

    `post` = the arm's POST-RULE predictions (the shipped ones), `traces` its trace lines by id,
    `ocr` the cached OCR pages by id (a document without pages simply has none), `changes` the
    value-free rule records per id. ``ft`` needs `zs_post` (the zero-shot arm's post-rule output of
    the SAME documents).
    """
    if arm not in ARMS:
        raise FlagsError(f"arm must be one of {ARMS}, got {arm!r}")
    ids = list(doc_ids)
    with id_guard(ids):
        tv, sigs = c3.build_table_v2(ids, post, traces, ocr, changes)
        agree = None
        if arm == "ft":
            if zs_post is None:
                raise FlagsError("the fine-tuned arm needs the zero-shot post-rule output")
            pairs = c3.align_all(post, zs_post, ids)
            agree = c3.agreement_matrix(tv.keys, post, zs_post, pairs, True)
    return Tables(tv, sigs, agree, ids)


def feature_names(arm: str) -> list[str]:
    """Column names of the field design of `arm` (what the artifact records and checks)."""
    base = list(c2.DESIGN_V2_FEATURES)
    return base if arm == "zs" else base + list(c3.AGREE_FEATURES)


def design(arm: str, t: Tables, novelty: Mapping[str, float]) -> np.ndarray:
    """Field design matrix: v2 design (``zs``) or v2 + agreement (``ft`` = variant ``v2_agree``)."""
    if arm == "zs":
        return c2.design_v2(t.tv, novelty)
    assert t.agree is not None
    return c3.design_variants(t.tv, t.agree, novelty)["v2_agree"]


def novelty_of(sigs: Mapping[str, np.ndarray | None], reference: np.ndarray) -> dict[str, float]:
    """Layout novelty of every document against one reference (NaN without a signature)."""
    return {
        d: (float("nan") if s is None else cf.layout_novelty(s, reference)) for d, s in sigs.items()
    }


def doc_design(
    tv: c2.TableV2, p: np.ndarray, doc_ids: Sequence[str], novelty: Mapping[str, float]
) -> np.ndarray:
    """Document-level design: ``c2.doc_table`` of the field probabilities + the novelty column."""
    with id_guard(doc_ids):
        X, order = c2.doc_table(tv, p, doc_ids)
    col = c2.DOC_FEATURES.index("layout_novelty")
    X[:, col] = [novelty.get(d, float("nan")) for d in order]
    return X


# --------------------------------------------------------------------------------------------
# The parity probe
# --------------------------------------------------------------------------------------------


def _box(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _probe_inputs() -> dict[str, Any]:
    """A fixed synthetic two-page invoice, its trace, OCR pages, rule records and a ZS twin."""
    header = {
        "invoice_number": "PRB-001",
        "invoice_date": "2026-01-02",
        "supplier_name": "Probe Supplies Ltd",
        "buyer_name": "Probe Buyer",
        "ship_to_name": "Probe Ship",
        "currency": "USD",
        "total_amount": "120.50",
    }
    rows = [
        {"supplier_part_number": "AB-100", "customer_part_number": None,
         "purchase_order": "4500", "quantity": "10"},
        {"supplier_part_number": "AB-200", "customer_part_number": "CP-7",
         "purchase_order": None, "quantity": "5"},
        {"supplier_part_number": "AB-300", "customer_part_number": None,
         "purchase_order": "4501", "quantity": "2"},
    ]  # fmt: skip
    doc = {"doc_type": "invoice", "header": header, "line_items": rows}
    twin_rows = [
        dict(rows[0]),
        dict(rows[1], quantity="6"),
        dict(rows[2], supplier_part_number="AB-301"),
    ]
    twin = {
        "doc_type": "invoice",
        "header": dict(header, buyer_name=None, currency="EUR"),
        "line_items": twin_rows,
    }

    def lp(scope: str, ridx: int | None, name: str, mn: float) -> dict[str, Any]:
        return {"scope": scope, "row_idx": ridx, "field": name, "min": mn, "mean": mn / 2}

    page1 = {
        "image": "probe_p1.jpg",
        "parsed": {"header": header, "line_items": rows[:2]},
        "field_logprobs": [lp("header", None, k, -0.1 * (i + 1)) for i, k in enumerate(header)]
        + [lp("row", 0, k, -0.05) for k in rows[0]]
        + [lp("row", 1, k, -0.4) for k in rows[1]],
    }
    page2 = {
        "image": "probe_p2.jpg",
        "parsed": {"header": dict(header, buyer_name=None), "line_items": rows[2:]},
        "field_logprobs": [lp("row", 0, k, -0.2) for k in rows[2]],
    }
    trace = {"doc_id": "probe_0001", "pages": [page1, page2],
             "merge": {"field_pages": dict.fromkeys(header, 1)}}  # fmt: skip
    p1_items = [
        OcrItem("Invoice PRB-001", _box(10, 10, 200, 30), 0.97),
        OcrItem("Probe Supplies Ltd", _box(10, 40, 200, 60), 0.95),
        OcrItem("Date 2026-01-02", _box(300, 10, 480, 30), 0.96),
        OcrItem("Part No", _box(10, 120, 90, 140), 0.9),
        OcrItem("Customer P/N", _box(110, 120, 220, 140), 0.9),
        OcrItem("PO", _box(240, 120, 290, 140), 0.9),
        OcrItem("Qty", _box(310, 120, 360, 140), 0.9),
        OcrItem("AB-100", _box(10, 160, 90, 180), 0.93),
        OcrItem("4500", _box(240, 160, 290, 180), 0.92),
        OcrItem("Total 120.50", _box(300, 400, 480, 420), 0.94),
    ]
    p2_items = [
        OcrItem("AB-300", _box(10, 50, 90, 70), 0.9),
        OcrItem("4501", _box(240, 50, 290, 70), 0.9),
    ]
    ocr = [
        PageOcr("probe_p1", 500, 600, "paddleocr", "line", 0, p1_items),
        PageOcr("probe_p2", 500, 600, "paddleocr", "line", 0, p2_items),
    ]
    changes = [{"rule": "R3", "field": "cpn_po", "row": 1, "kind": "swap"}]
    return {"doc": doc, "twin": twin, "trace": trace, "ocr": ocr, "changes": changes}


def _digest(X: np.ndarray) -> str:
    clean = np.where(np.isnan(X), -1e18, np.round(X, 6))
    payload = json.dumps({"shape": list(X.shape), "x": clean.tolist()}, sort_keys=True)
    return sha256_bytes(payload.encode("utf-8"))


def parity_probe(arm: str) -> dict[str, Any]:
    """Digest of the design matrix the CURRENT feature code builds for the fixed probe document.

    Stored in the artifact at freeze time and recomputed at load: a different digest means the
    feature builder behaves differently than when the calibrator was fitted (REFUSED).
    """
    p = _probe_inputs()
    d = "probe_0001"
    t = build_tables(arm, [d], {d: p["doc"]}, {d: p["trace"]}, {d: p["ocr"]}, {d: p["changes"]},
                     {d: p["twin"]})  # fmt: skip
    ref = np.zeros((1, len(next(iter(t.sigs.values())))))
    X = design(arm, t, novelty_of(t.sigs, ref))
    return {"digest": _digest(X), "rows": int(X.shape[0]), "columns": int(X.shape[1])}


# --------------------------------------------------------------------------------------------
# The calibrator artifact
# --------------------------------------------------------------------------------------------


@dataclass
class Calibrator:
    """A verified, loaded calibrator artifact."""

    data: dict[str, Any]
    sha256: str
    field_models: dict[str, Model]
    doc_model: Model
    reference: np.ndarray

    @property
    def arm(self) -> str:
        """``zs`` or ``ft``."""
        return str(self.data["arm"])

    @property
    def structure(self) -> str:
        """``pooled`` or ``per_type``."""
        return str(self.data["field_model"]["structure"])

    @property
    def run_config_hash(self) -> str | None:
        """Config hash of the run the calibrator was frozen for (None: not recorded, older file)."""
        h = self.data.get("run_config_hash")
        return h if isinstance(h, str) and h else None

    def p_field(self, X: np.ndarray, types: np.ndarray) -> np.ndarray:
        """P(correct) of every row of `X` (the caller masks the non-emitted rows)."""
        if self.structure == "pooled":
            return self.field_models["all"].predict(X)
        out = np.full(len(X), np.nan)
        for t in sorted(set(types.tolist())):
            if t not in self.field_models:
                raise FlagsError(f"no model for field type {t!r} in the calibrator")
            m = types == t
            out[m] = self.field_models[t].predict(X[m])
        return out

    def field_tau(self, target: float) -> dict[str, float | None]:
        """Per-field-type tau at `target` (None = nothing of that type is auto-accepted)."""
        taus = self.data["thresholds"]["field"]["tau"].get(tkey(target))
        if taus is None:
            raise FlagsError(f"the calibrator stores no field tau for target {target}")
        return dict(taus)

    def doc_tau(self, target: float) -> float | None:
        """Document-level tau at `target` (None = no document is auto-accepted)."""
        taus = self.data["thresholds"]["doc"]["tau"]
        if tkey(target) not in taus:
            raise FlagsError(f"the calibrator stores no document tau for target {target}")
        return taus[tkey(target)]


def load_calibrator(path: Path, arm: str | None = None, check_probe: bool = True) -> Calibrator:
    """Load and verify a calibrator artifact; raises `FlagsError` (REFUSED) on any mismatch.

    Checks: JSON schema / kind, the arm (when `arm` is given), the feature column names of the
    current code, every GBM's sklearn version + pickle hash (`Model`), and the parity probe
    digest (`parity_probe`; skip only in tests of the loader itself).
    """
    p = Path(path)
    raw = p.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    if data.get("artifact") != ARTIFACT_KIND or data.get("schema") != ARTIFACT_SCHEMA:
        raise FlagsError(f"REFUSED: {p.name} is not a schema-{ARTIFACT_SCHEMA} calibrator artifact")
    if data.get("arm") not in ARMS or (arm is not None and data["arm"] != arm):
        raise FlagsError(f"REFUSED: calibrator arm {data.get('arm')!r}, expected {arm!r}")
    want = feature_names(data["arm"])
    if data.get("feature_names") != want:
        raise FlagsError(
            "REFUSED: the calibrator was frozen with a different feature set than this code builds "
            f"({len(data.get('feature_names') or [])} vs {len(want)} columns): re-freeze it"
        )
    fm = data["field_model"]
    models = {k: Model(v) for k, v in fm["models"].items()}
    doc_model = Model(data["doc_model"])
    ref = np.asarray(data["novelty"]["reference"], dtype=float)
    cal = Calibrator(data, sha256_bytes(raw), models, doc_model, ref)
    if check_probe:
        now = parity_probe(data["arm"])
        if now != data.get("parity_probe"):
            raise FlagsError(
                "REFUSED: the feature builder no longer reproduces the frozen parity probe "
                "(its behaviour changed after the calibrator was fitted): re-freeze the calibrator"
            )
    return cal


# --------------------------------------------------------------------------------------------
# Flags
# --------------------------------------------------------------------------------------------


def _field_entry(emitted: bool, p: float, tau: float | None) -> dict[str, Any]:
    if not emitted:
        return {"emitted": False, "p_correct": None, "flag": None}
    if np.isnan(p):
        raise FlagsError("an emitted field has no probability (NaN): refusing to write flags")
    accept = tau is not None and p >= tau
    return {"emitted": True, "p_correct": float(p), "flag": ACCEPT if accept else REVIEW}


def compute_flags(
    cal: Calibrator,
    tables: Tables,
    novelty: Mapping[str, float],
    X: np.ndarray,
    field_target: float = PRIMARY_TARGET,
    doc_target: float = PRIMARY_TARGET,
) -> dict[str, Any]:
    """The ``docs`` and ``summary`` blocks of ``review_flags.json`` (numbers, booleans, names)."""
    tv = tables.tv
    types = tv.types()
    p = cal.p_field(X, types)
    p = np.where(tv.emitted, p, np.nan)
    taus = cal.field_tau(field_target)
    tau_doc = cal.doc_tau(doc_target)
    ids = list(tables.doc_ids)
    p_doc = cal.doc_model.predict(doc_design(tv, p, ids, novelty))
    rows_of: dict[str, list[int]] = {d: [] for d in ids}
    for i, k in enumerate(tv.keys):
        rows_of[k.doc_id].append(i)
    docs: dict[str, Any] = {}
    n_acc = n_rev = n_null = 0
    by_type: dict[str, dict[str, int]] = {t: {"accept": 0, "review": 0} for t in c2.FIELD_TYPES}
    for j, d in enumerate(ids):
        header: dict[str, Any] = {}
        rows: dict[int, dict[str, Any]] = {}
        for i in rows_of[d]:
            k = tv.keys[i]
            t = str(types[i])
            e = _field_entry(bool(tv.emitted[i]), float(p[i]), taus.get(t))
            if e["flag"] == ACCEPT:
                n_acc += 1
                by_type[t]["accept"] += 1
            elif e["flag"] == REVIEW:
                n_rev += 1
                by_type[t]["review"] += 1
            else:
                n_null += 1
            if k.scope == "header":
                header[k.field] = e
            else:
                rows.setdefault(k.row_idx, {})[k.field] = e
        pd_ = float(p_doc[j])
        docs[d] = {
            "p_fully_correct": pd_,
            "auto_accept": bool(tau_doc is not None and pd_ >= tau_doc),
            "header": header,
            "rows": [{"row_idx": r, "fields": rows[r]} for r in sorted(rows)],
        }
    summary = {
        "n_docs": len(ids),
        "n_docs_auto_accept": sum(v["auto_accept"] for v in docs.values()),
        "n_emitted_fields": n_acc + n_rev,
        "n_accept": n_acc,
        "n_review": n_rev,
        "n_null_fields_not_scored": n_null,
        "by_field_type": by_type,
    }
    return {"docs": docs, "summary": summary}


def flags_document(
    cal: Calibrator,
    block: Mapping[str, Any],
    *,
    field_target: float,
    doc_target: float,
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """The whole ``review_flags.json`` object: metadata, thresholds, calibrator hash, ``docs``."""
    d = cal.data
    return {
        "schema": FLAGS_SCHEMA,
        "kind": FLAGS_KIND,
        "note": (
            "advisory side file: probabilities, booleans and field names only; no extracted value. "
            "Never part of test_predictions.json."
        ),
        "arm": cal.arm,
        "calibrator": {
            "sha256": cal.sha256,
            "structure": cal.structure,
            "kind": d["field_model"]["kind"],
            "sklearn_version": d.get("sklearn_version"),
            "frozen_from": d.get("source", {}).get("name"),
            "parity_probe": d.get("parity_probe"),
        },
        "thresholds": {
            "field_target": field_target,
            "field_tau": cal.field_tau(field_target),
            "doc_target": doc_target,
            "doc_tau": cal.doc_tau(doc_target),
            "nested_target_met": {
                "field": d["thresholds"]["nested_target_met"]["field"][tkey(field_target)],
                "doc": d["thresholds"]["nested_target_met"]["doc"][tkey(doc_target)],
            },
            "meaning": d["thresholds"]["meaning"],
            "nested_estimates": d.get("nested_estimates"),
        },
        "inputs": dict(inputs),
        "summary": block["summary"],
        "docs": block["docs"],
    }


# --------------------------------------------------------------------------------------------
# Value-free scan (used by the stage and by tests)
# --------------------------------------------------------------------------------------------


def doc_string_values(docs: Mapping[str, Any]) -> list[str]:
    """Every string VALUE found under ``docs`` (keys are not values): must be accept / review."""
    out: list[str] = []

    def walk(x: Any) -> None:
        if isinstance(x, str):
            out.append(x)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(docs)
    return out


# --------------------------------------------------------------------------------------------
# Stage: python -m shipdoc.flags
# --------------------------------------------------------------------------------------------


def _read_traces(path: Path) -> list[dict[str, Any]]:
    return spike._read_trace(Path(path))


def post_rule_output(
    traces: Sequence[Mapping[str, Any]], shapes_file: Path | None, ocr_cache: Path | None
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """``(final predictions, {doc_id: rule change records})`` of the production post path.

    postprocess_traces (R1, R2, R3 ON, the frozen shapes) -> coerce -> repair: the pipeline of
    ``predict.assemble_submission`` / ``reuse.assemble_reuse``.
    """
    shapes, _sha = postrules.load_slot_shapes(shapes_file)
    preds, records, _summary = postrules.postprocess_traces(traces, RuleConfig(), shapes, ocr_cache)
    final, _ = repair_predictions(coerce_predictions(dict(preds)))
    return final, {r["doc_id"]: r["changes"] for r in records}


def _read_manifest(path: Path) -> dict[str, Any]:
    """The manifest as a dict; ``{}`` when absent, unreadable or not an object."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _all_rules_off(manifest_path: Path) -> bool:
    """True only when the manifest records ``post_rules.switches`` with every value False."""
    post = _read_manifest(manifest_path).get("post_rules")
    sw = post.get("switches") if isinstance(post, dict) else None
    return isinstance(sw, dict) and bool(sw) and all(v is False for v in sw.values())


def _model_id(manifest: Mapping[str, Any]) -> str:
    return str((manifest.get("model") or {}).get("id") or "")


def run_stage(
    *,
    submission_dir: Path,
    calibrator: Path,
    ocr_cache: Path,
    out: Path | None = None,
    arm: str | None = None,
    zs_v0_dir: Path | None = None,
    batch_size: int = 8,
    ack_spike_diff: bool = False,
    config: Path | None = None,
    shapes_file: Path | None = None,
    data_root: Path | None = None,
    field_target: float = PRIMARY_TARGET,
    doc_target: float = PRIMARY_TARGET,
    expect_docs: int | None = pr.EXPECTED_DOCS,
    check_manifest: bool = True,
    check_probe: bool = True,
    say: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Read a submission folder, write ``review_flags.json``; returns the flags object.

    Refuses (`FlagsError`, nothing written) when: the calibrator fails `load_calibrator`; the
    folder has no ``test_predictions.json`` (a REJECTED file is never flagged); predictions and
    trace ids differ or are not `expect_docs`; the manifest's model (LoRA or not) does not match
    the calibrator's arm; the shipped predictions are not what the production post path makes of
    the traces (shapes, OCR cache); for the fine-tuned arm the zero-shot traces fail
    ``reuse.decide_reuse`` (`ZsReuseRefused`) or are missing.
    """
    sub = Path(submission_dir)
    cal = load_calibrator(calibrator, arm, check_probe=check_probe)
    pred_path = sub / "test_predictions.json"
    if not pred_path.is_file():
        raise FlagsError(
            f"REFUSED: {sub.name} has no test_predictions.json (rejected or unfinished)"
        )
    preds = json.loads(pred_path.read_text(encoding="utf-8"))
    traces_l = _read_traces(sub / "trace.jsonl")
    traces = {t["doc_id"]: t for t in traces_l}
    ids = sorted(preds)
    if sorted(traces) != ids or (expect_docs is not None and len(ids) != expect_docs):
        raise FlagsError(
            f"REFUSED: {len(ids)} predicted / {len(traces)} traced documents (expected "
            f"{expect_docs}; the id sets must be equal)"
        )
    manifest: dict[str, Any] = {}
    if check_manifest:
        man_path = sub / "manifest.json"
        if not man_path.is_file():
            raise FlagsError("REFUSED: no manifest.json to check the model against the arm")
        manifest = json.loads(man_path.read_text(encoding="utf-8"))
        is_ft = LORA_MARK in _model_id(manifest)
        if is_ft != (cal.arm == "ft"):
            raise FlagsError(
                f"REFUSED: the submission's model is {'fine-tuned' if is_ft else 'zero-shot'} but "
                f"the calibrator is the {cal.arm.upper()} one: its probabilities would be wrong"
            )
        frozen_for = cal.run_config_hash  # recorded by scripts/freeze_calibrator.py (newer files)
        if frozen_for is not None:
            run_hash = (manifest.get("config") or {}).get("hash")
            if run_hash != frozen_for:
                raise FlagsError(
                    f"REFUSED: the calibrator was frozen for config hash {frozen_for} but the "
                    f"submission's run used {run_hash}: another resolution / config, so its "
                    "probabilities would be wrong. Re-freeze the calibrator for this config."
                )
    post, changes = post_rule_output(traces_l, shapes_file, ocr_cache)
    if post != {d: preds[d] for d in ids}:
        raise FlagsError(
            "REFUSED: test_predictions.json is not what the production post-processing makes of "
            "trace.jsonl with this shapes file and OCR cache; the flags would describe other "
            "predictions"
        )
    # Fail closed on a missing rules.jsonl: the consistency check below would otherwise be skipped
    # silently. The file may be absent ONLY when the manifest records post_rules.switches and every
    # rule (R1-R3) is OFF there (nothing to cross-check). No manifest, no switches record, a
    # malformed one, or any rule ON means REFUSED (a stage run with --skip-manifest-check and no
    # manifest.json therefore also refuses when the file is absent).
    rules_path = sub / "rules.jsonl"
    if not rules_path.is_file() and not _all_rules_off(sub / "manifest.json"):
        raise FlagsError(
            f"REFUSED: {sub.name} has no rules.jsonl and its manifest.json does not record every "
            "post-processing rule as OFF (post_rules.switches); the rule-record consistency "
            "check cannot run. Copy rules.jsonl from the run folder back into the submission."
        )
    if rules_path.is_file():
        recs = {
            r["doc_id"]: r["changes"]
            for r in (json.loads(ln) for ln in rules_path.read_text("utf-8").splitlines() if ln)
        }
        if recs != changes:
            raise FlagsError("REFUSED: rules.jsonl differs from the rule records recomputed here")
    zs_post: dict[str, Any] | None = None
    zs_facts: dict[str, Any] | None = None
    if cal.arm == "ft":
        zs_post, zs_facts = _zs_arm(
            ids, zs_v0_dir, batch_size, ack_spike_diff, config, data_root, shapes_file, ocr_cache
        )
    ocr = ev.load_ocr_pages(ids, ocr_cache)
    tables = build_tables(cal.arm, ids, post, traces, ocr, changes, zs_post)
    nov = novelty_of(tables.sigs, cal.reference)
    X = design(cal.arm, tables, nov)
    block = compute_flags(cal, tables, nov, X, field_target, doc_target)
    man_code_sha = _read_manifest(sub / "manifest.json").get("code_sha")
    inputs = {
        "submission": sub.name,
        "test_predictions_sha256": sha256_file(pred_path),
        "trace_sha256": sha256_file(sub / "trace.jsonl"),
        "ocr_docs_with_pages": len(ocr),
        "code_sha": spike.git_commit(),  # kept for compatibility: same value as local_head_sha
        "local_head_sha": spike.git_commit(),  # the repo HEAD of the machine running this stage
        # the code SHA the submission was produced with (its manifest.json); None when not recorded
        "submission_code_sha": man_code_sha,
        "rules_jsonl_sha256": sha256_file(rules_path) if rules_path.is_file() else None,
        "zs_reuse": zs_facts,
    }
    doc = flags_document(
        cal, block, field_target=field_target, doc_target=doc_target, inputs=inputs
    )
    dest = Path(out) if out else sub / FLAGS_NAME
    pr._write_json(dest, doc)
    s = block["summary"]
    say(
        f"flags: {s['n_docs']} docs, auto-accept {s['n_docs_auto_accept']}, emitted fields "
        f"{s['n_emitted_fields']} (accept {s['n_accept']}, review {s['n_review']}), null fields "
        f"{s['n_null_fields_not_scored']}; calibrator {cal.arm} sha256 {cal.sha256[:12]} -> {dest}"
    )
    return doc


class ZsReuseRefused(FlagsError):
    """The v0 zero-shot traces may not be reused (`reuse.decide_reuse`): no ZS inference here."""


def _zs_arm(
    ids: Sequence[str],
    v0_dir: Path | None,
    batch_size: int,
    ack: bool,
    config: Path | None,
    data_root: Path | None,
    shapes_file: Path | None,
    ocr_cache: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Zero-shot post-rule output of the test documents from the v0 traces, or refuse loudly."""
    if v0_dir is None:
        raise ZsReuseRefused(
            "REFUSED: the fine-tuned calibrator needs the zero-shot traces of the same documents "
            "(--zs-v0-dir); no zero-shot inference runs in this stage"
        )
    cfg = spike.load_config(config or paths.REPO_ROOT / "configs" / "spike_qwen35_4b_img_only.yaml")
    data = Path(data_root) if data_root else paths.data_dir()
    dec = reuse.evaluate_reuse(cfg, Path(v0_dir), data, batch_size, "0/1", ack)
    if not dec.ok:
        raise ZsReuseRefused(
            "REFUSED: the v0 zero-shot traces cannot be reused (shipdoc.reuse.decide_reuse): "
            + " | ".join(dec.reasons)
            + ". Nothing is decoded here; fix the v0 folder / pin or run the zero-shot path."
        )
    _man, _rep, v0_traces = reuse.load_v0(Path(v0_dir))
    assert v0_traces is not None  # decide_reuse refuses a missing trace
    if sorted(t["doc_id"] for t in v0_traces) != sorted(ids):
        raise ZsReuseRefused(
            "REFUSED: the v0 traces do not cover exactly the submission's documents"
        )
    zs_post, _changes = post_rule_output(v0_traces, shapes_file, ocr_cache)
    return {d: zs_post[d] for d in ids}, {
        "v0_dir": Path(v0_dir).name,
        "decision_ok": True,
        "v0_code_sha": dec.facts.get("v0_code_sha"),
        "decode_path_files": (dec.facts.get("decode_path") or {}).get("files"),
    }


def build_parser() -> argparse.ArgumentParser:
    """Argument parser of ``python -m shipdoc.flags``."""
    p = argparse.ArgumentParser(prog="python -m shipdoc.flags", description=__doc__.split("\n")[0])
    p.add_argument("--submission-dir", type=Path, required=True)
    p.add_argument("--calibrator", type=Path, required=True)
    p.add_argument("--ocr-cache", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None, help="default: <submission>/review_flags.json")
    p.add_argument("--arm", choices=ARMS, default=None, help="refuse a calibrator of another arm")
    p.add_argument("--zs-v0-dir", type=Path, default=None, help="fine-tuned arm: the 04 v0 folder")
    p.add_argument("--batch-size", type=int, default=8, help="the batch size v0 was decoded at")
    p.add_argument("--ack-spike-diff", action="store_true")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--shapes-file", type=Path, default=None)
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--field-target", type=float, default=PRIMARY_TARGET, choices=FIELD_TARGETS)
    p.add_argument("--doc-target", type=float, default=PRIMARY_TARGET, choices=DOC_TARGETS)
    p.add_argument("--expect-docs", type=int, default=pr.EXPECTED_DOCS)
    p.add_argument("--skip-manifest-check", action="store_true", help="tests only")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Stage CLI. Exit 0 ok, 1 refused (message on stderr, counts only), 3 ZS reuse refused."""
    a = build_parser().parse_args(argv)
    try:
        run_stage(
            submission_dir=a.submission_dir, calibrator=a.calibrator, ocr_cache=a.ocr_cache,
            out=a.out, arm=a.arm, zs_v0_dir=a.zs_v0_dir, batch_size=a.batch_size,
            ack_spike_diff=a.ack_spike_diff, config=a.config, shapes_file=a.shapes_file,
            data_root=a.data_root, field_target=a.field_target, doc_target=a.doc_target,
            expect_docs=a.expect_docs, check_manifest=not a.skip_manifest_check,
        )  # fmt: skip
    except ZsReuseRefused as exc:
        print(f"shipdoc.flags: {exc}", file=sys.stderr)
        return 3
    except (FlagsError, c2.NoTestDataError, pr.PredictError) as exc:
        print(f"shipdoc.flags: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
