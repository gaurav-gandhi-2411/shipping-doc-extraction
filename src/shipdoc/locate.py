"""Locate a gold (or predicted) value inside a document's OCR pages.

Shared by the provenance table, the OCR-ceiling report and the taxonomy's hallucination check, so
all three agree on what "the value is on the page" means.

Candidate spans (per reading-order line of each page):

* every run of 1..``MAX_WORDS`` consecutive whitespace-delimited words (PaddleOCR items are whole
  text *lines*, so a value is usually a sub-span of an item; word boxes are the proportional split
  of the item box, i.e. approximate),
* n-grams of 1..4 consecutive items on the line, and the whole line.

Match levels, tried in order, best level wins:

* ``exact``      the value string is a substring of a span (for non-name kinds it must not touch
                 an alphanumeric neighbour, so ``"9"`` does not "match" inside ``"1920"``);
* ``normalized`` the scorer's own ``same(field, candidate_form, value)`` is True for a form of the
                 span: dates parsed from common printed formats to ISO, numbers with currency and
                 either decimal convention, identifiers stripped of a label prefix, airport codes
                 followed by a city, names compared as the scorer does;
* ``fuzzy``      rapidfuzz ``ratio`` over the scorer-normalized strings, at or above a threshold
                 (``FUZZY_THRESHOLD`` by default; chosen on data by ``scripts/provenance.py``).

The scorer (``assignment/score.py``) is loaded through ``shipdoc.eval.load_scorer`` (lazily, since
``shipdoc.eval`` imports this module) and is never copied.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date
from types import ModuleType
from typing import Any, Literal

from rapidfuzz import fuzz, process

from shipdoc.ocr import PageOcr, page_items

Box = tuple[float, float, float, float]
Level = Literal["exact", "normalized", "fuzzy"]
LEVEL_RANK: dict[str, int] = {"exact": 0, "normalized": 1, "fuzzy": 2}

# Chosen on train+dev by scripts/provenance.py (see meta/field_provenance.json, "threshold").
# The smallest T at which the negative false-match rate is <= 1%, per FMR_TARGET.
FUZZY_THRESHOLD = 87.0
# Per-kind exception, also from the study: dates sit in a narrow ISO range, so one wrong digit of a
# DIFFERENT date scores 90 (8.1% negative false-match rate at T=87..90 on its own, and no positive
# was rescued by fuzzy date matching). 91 is the smallest T with a date FMR <= 1%.
FUZZY_THRESHOLD_BY_KIND = {"date": 91.0}
# Words per word-run span. Names are the only values that run long (a company name is up to ~6
# words); every other kind is short.
MAX_WORDS = 8
MAX_WORDS_BY_KIND = {"name": 8, "num": 4, "date": 4, "code": 4, "id": 4}
MAX_ITEM_NGRAM = 4

_CURRENCY = r"USD|EUR|JPY|CNY|SGD|RMB|US\$|S\$|[$€¥£元]"


def _sc() -> ModuleType:
    # Lazy: shipdoc.eval imports this module (hallucination check), so a top-level import cycles.
    from shipdoc import eval as ev

    return ev.load_scorer()


def kind_of(field_name: str) -> str:
    """Scorer kind of a field (name/num/date/code/id), from the scorer's own ``KIND`` table."""
    return str(_sc().KIND.get(field_name, "id"))


@dataclass(frozen=True)
class Span:
    """One candidate text span on a page."""

    page: int
    line_idx: int
    text: str
    box: Box
    src: str  # "word" | "item" | "line"


@dataclass(frozen=True)
class Match:
    """Best-matching span for a value."""

    page: int
    box: Box
    line_idx: int
    level: Level
    score: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible dict."""
        return {
            "page": self.page,
            "box": list(self.box),
            "line_idx": self.line_idx,
            "level": self.level,
            "score": self.score,
            "text": self.text,
        }


# ---------------------------------------------------------------------------------------------
# Printed-form parsing helpers
# ---------------------------------------------------------------------------------------------

_MONTHS = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]  # fmt: skip
_NUM_DATE = re.compile(r"(?<!\d)(\d{1,4})[/.\-](\d{1,2})[/.\-](\d{1,4})(?!\d)")
_DMY_NAMED = re.compile(
    r"(?<!\d)(\d{1,2})(?:st|nd|rd|th)?[ \-]?([A-Za-z]{3,9})\.?[ \-,]*(\d{4})(?!\d)"
)
_MDY_NAMED = re.compile(
    r"(?<![A-Za-z])([A-Za-z]{3,9})\.?[ \-]*(\d{1,2})(?:st|nd|rd|th)?,?[ \-]*(\d{4})(?!\d)"
)


def _month(token: str) -> int | None:
    t = token.lower()
    if len(t) < 3:
        return None
    for i, name in enumerate(_MONTHS, 1):
        if name.startswith(t) and (len(t) >= 3):
            return i
    if t == "sept":
        return 9
    return None


def _iso(y: int, m: int, d: int) -> str | None:
    try:
        return date(y, m, d).isoformat()
    except ValueError:
        return None


def date_forms(text: str) -> list[str]:
    """ISO dates parsed from printed forms found inside `text`.

    Handles DD/MM/YYYY, MM/DD/YYYY (both readings are returned when both are valid),
    YYYY-MM-DD, DD-MON-YYYY, DD Month YYYY and "Month DD, YYYY".
    """
    if not re.search(r"\d", text):
        return []
    out: list[str] = []
    for a, b, c in _NUM_DATE.findall(text):
        if len(a) == 4 and len(c) <= 2:
            out.append(_iso(int(a), int(b), int(c)) or "")
        elif len(c) == 4 and len(a) <= 2:
            out.append(_iso(int(c), int(b), int(a)) or "")  # DD/MM/YYYY
            out.append(_iso(int(c), int(a), int(b)) or "")  # MM/DD/YYYY
    for d, mon, y in _DMY_NAMED.findall(text):
        m = _month(mon)
        if m:
            out.append(_iso(int(y), m, int(d)) or "")
    for mon, d, y in _MDY_NAMED.findall(text):
        m = _month(mon)
        if m:
            out.append(_iso(int(y), m, int(d)) or "")
    return list(dict.fromkeys(x for x in out if x))


_NUM_BODY = re.compile(r"-?\d[\d.,]*(?: \d{3})*")


def _number_body(text: str) -> str | None:
    """The bare number (``1,234.50``, spaces removed) in `text`, or None if not ONE clean number."""
    t = text.strip()
    m = re.search(r"\d", t)
    if not m:
        return None
    prefix, rest = t[: m.start()], t[m.start() :]
    if prefix.endswith("-") and not prefix[:-1].strip(" :"):
        rest, prefix = "-" + rest, prefix[:-1]
    p = re.sub(_CURRENCY, "", prefix).strip()
    if p and not re.fullmatch(r"[\W_]*", p) and not p.endswith(":"):
        return None
    body = re.match(r"-?\d{1,3}(?: \d{3})+(?:[.,]\d+)?(?!\d)", rest) or re.match(
        r"-?\d[\d.,]*", rest
    )
    if not body:
        return None
    suffix = rest[body.end() :].strip()
    suffix = re.sub(_CURRENCY, "", suffix).strip()
    if suffix and not re.fullmatch(r"[\W_]*|[A-Za-z]{1,3}[\W_]*", suffix):
        return None
    return body.group(0).strip().replace(" ", "")


def parse_printed_number(text: str) -> list[float]:
    """Numeric readings of a printed number such as ``USD 1,234.50`` or ``1.234,50``.

    The text must be ONE clean number once currency codes/symbols, a leading ``label:`` and a short
    unit suffix are stripped (so ``PO45`` or ``12/3`` do not parse). Returns the plausible values
    (usually one; ``1.234`` is read both as 1.234 and, EU style, 1234).
    """
    s = _number_body(text)
    return [] if s is None else _separator_readings(s)


def repaired_number_readings(text: str) -> list[float]:
    """Readings that assume OCR confused ``,`` and ``.`` between digit groups.

    ``2.345,678.90`` or ``1.234.567.89`` are not valid in any one convention, but with the last
    separator before exactly two digits as the decimal point and every earlier separator a
    thousands mark they read 2345678.90 / 1234567.89. These are recognition repairs, not printed
    formats, so the locator offers them at the fuzzy level only. Readings equal to a normal
    reading are dropped.
    """
    s = _number_body(text)
    if s is None or not re.fullmatch(r"-?\d{1,3}(?:[.,]\d{3})+[.,]\d{2}", s):
        return []
    digits = re.sub(r"[.,]", "", s)
    v = float(digits[:-2] + "." + digits[-2:])
    return [v] if v not in _separator_readings(s) else []


def _separator_readings(s: str) -> list[float]:
    neg = s.startswith("-")
    s = s.lstrip("-")
    vals: list[str] = []
    if "," in s and "." in s:
        dec = "," if s.rfind(",") > s.rfind(".") else "."
        thou = "." if dec == "," else ","
        parts = s.split(dec)
        if len(parts) == 2 and thou not in parts[1]:
            vals.append(parts[0].replace(thou, "") + "." + parts[1])
    elif "," in s:
        if re.fullmatch(r"\d{1,3}(?:,\d{3})+", s):
            vals.append(s.replace(",", ""))
        elif re.fullmatch(r"\d+,\d+", s):
            vals.append(s.replace(",", "."))
    elif "." in s:
        if re.fullmatch(r"\d{1,3}(?:\.\d{3}){2,}", s):
            vals.append(s.replace(".", ""))
        elif re.fullmatch(r"\d+\.\d+", s):
            vals.append(s)
            if re.fullmatch(r"\d{1,3}\.\d{3}", s):
                vals.append(s.replace(".", ""))  # EU thousands reading
    elif re.fullmatch(r"\d+", s):
        vals.append(s)
    out = []
    for v in vals:
        try:
            out.append(-float(v) if neg else float(v))
        except ValueError:
            continue
    return list(dict.fromkeys(out))


def canon_num(v: float) -> str:
    """Canonical numeric string used as the fuzzy/lookup key (2 dp, trailing zeros dropped)."""
    s = f"{v:.2f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def _alnum(s: str) -> str:
    return re.sub(r"[^0-9a-z]", "", s.lower())


def _nospace_upper(s: str) -> str:
    return re.sub(r"\s", "", s).upper()


def _gold_num(sc: ModuleType, g: str) -> float | None:
    v = sc._num(g)
    return None if v is None else float(v)


# ---------------------------------------------------------------------------------------------
# Forms: (normalized key, text handed to the scorer's `same`) per span
# ---------------------------------------------------------------------------------------------

Form = tuple[str, str]


def _name_forms(text: str) -> list[Form]:
    k = _alnum(text)
    return [(k, text)] if k else []


def _id_forms(text: str) -> list[Form]:
    t = text.strip()
    forms = [t]
    if re.search(r"[:#]", t):
        forms.append(re.split(r"[:#]", t)[-1].strip())
    stripped = t.strip(".,;:()[]")
    if stripped != t:
        forms.append(stripped)
    return [(_nospace_upper(f), f) for f in dict.fromkeys(forms) if _nospace_upper(f)]


def _num_forms(text: str) -> list[Form]:
    return [(canon_num(v), f"{v:.4f}") for v in parse_printed_number(text)]


def _num_repair_forms(text: str) -> list[Form]:
    return [(canon_num(v), f"{v:.4f}") for v in repaired_number_readings(text)]


def _date_forms(text: str) -> list[Form]:
    return [(d, d) for d in date_forms(text)]


_LEAD_CODE = re.compile(r"^([A-Za-z]{3})(?![A-Za-z])")


def _airport_forms(text: str) -> list[Form]:
    """Full text, a leading 3-letter code followed by a city, or a code glued to its city.

    Only used for airport fields: a 3-letter word in free text is far too common to treat as
    the airport code anywhere in the span.
    """
    t = text.strip()
    forms = [t]
    m = _LEAD_CODE.match(t)
    if m:
        forms.append(m.group(1))
    first = t.split(" ", 1)[0]
    if len(first) >= 6 and first.isalpha() and first.isupper():
        forms.append(first[:3])  # "ABCABCVILLE": code glued to its city
    return [(f.upper(), f) for f in dict.fromkeys(forms)]


def _currency_forms(text: str) -> list[Form]:
    """Full text, or any 3-letter word inside it (covers ``Total:USD123.5``)."""
    t = text.strip()
    forms = [t, *re.findall(r"(?<![A-Za-z])[A-Za-z]{3}(?![A-Za-z])", t)]
    return [(f.upper(), f) for f in dict.fromkeys(forms)]


def _form_key(field_name: str, kind: str) -> str:
    if kind == "code":
        return "currency" if field_name == "currency" else "airport"
    return kind


_FORMS = {
    "name": _name_forms,
    "id": _id_forms,
    "num": _num_forms,
    "date": _date_forms,
    "airport": _airport_forms,
    "currency": _currency_forms,
}
# Forms that repair a recognition error rather than reading a printed format: fuzzy level only.
_REPAIR_FORMS = {"num": _num_repair_forms}
_FORM_KIND_MAXW = {
    "name": MAX_WORDS_BY_KIND["name"],
    "id": MAX_WORDS_BY_KIND["id"],
    "num": MAX_WORDS_BY_KIND["num"],
    "date": MAX_WORDS_BY_KIND["date"],
    "airport": MAX_WORDS_BY_KIND["code"],
    "currency": MAX_WORDS_BY_KIND["code"],
}


def _gold_key(sc: ModuleType, form_key: str, g: str) -> str | None:
    """Normalized key of a value, mirroring the scorer's equivalence for its kind."""
    if form_key == "name":
        return _alnum(g) or None
    if form_key == "id":
        return _nospace_upper(g) or None
    if form_key in ("airport", "currency"):
        return g.upper()
    if form_key == "num":
        v = _gold_num(sc, g)
        return None if v is None else canon_num(v)
    if form_key == "date":
        return g if re.fullmatch(r"\d{4}-\d{2}-\d{2}", g) else None
    return None


# ---------------------------------------------------------------------------------------------
# Index of one document
# ---------------------------------------------------------------------------------------------


@dataclass
class _FormTable:
    keys: list[str] = field(default_factory=list)
    owners: list[int] = field(default_factory=list)  # index into DocIndex.spans
    texts: list[str] = field(default_factory=list)
    by_key: dict[str, list[int]] = field(default_factory=dict)
    uniq: list[str] = field(default_factory=list)
    repaired: set[int] = field(default_factory=set)  # entry indices that are repair readings


def _union(boxes: list[Box]) -> Box:
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


class DocIndex:
    """All candidate spans of a document's pages, with per-kind form tables built lazily."""

    def __init__(self, pages: list[PageOcr]) -> None:
        self.pages = pages
        self.spans: list[Span] = []
        seen: set[tuple[int, int, str]] = set()
        for pno, page in enumerate(pages):
            lines: dict[int, list[Any]] = defaultdict(list)
            for it in page_items(page):
                lines[it.line_idx].append(it)
            for lidx in sorted(lines):
                self._add_line(pno, lidx, lines[lidx], seen)
        self._tables: dict[str, _FormTable] = {}
        self._runs: dict[int, list[tuple[str, tuple[int, ...]]]] = {}

    def _add_line(
        self, pno: int, lidx: int, items: list[Any], seen: set[tuple[int, int, str]]
    ) -> None:
        def add(text: str, box: Box, src: str) -> None:
            key = (pno, lidx, text)
            if key not in seen:
                seen.add(key)
                self.spans.append(Span(pno, lidx, text, box, src))

        words: list[tuple[str, Box]] = []
        for it in items:
            x0, y0, x1, y1 = it.box
            n = max(len(it.text), 1)
            for m in re.finditer(r"\S+", it.text):
                words.append(
                    (
                        m.group(0),
                        (x0 + (x1 - x0) * m.start() / n, y0, x0 + (x1 - x0) * m.end() / n, y1),
                    )
                )
        for i in range(len(words)):
            for j in range(i, min(i + MAX_WORDS, len(words))):
                seg = words[i : j + 1]
                add(" ".join(w for w, _ in seg), _union([b for _, b in seg]), "word")
        for i in range(len(items)):
            for j in range(i, min(i + MAX_ITEM_NGRAM, len(items))):
                seg = items[i : j + 1]
                add(" ".join(s.text.strip() for s in seg), _union([s.box for s in seg]), "item")
        add(" ".join(s.text.strip() for s in items), _union([s.box for s in items]), "line")

    def line_text(self, page: int, line_idx: int) -> str:
        """Text of one reading-order line (items joined by a space)."""
        return " ".join(it.text for it in page_items(self.pages[page]) if it.line_idx == line_idx)

    def page_runs(self, page: int) -> list[tuple[str, tuple[int, ...]]]:
        """Id-normalized word runs of one page that cross at least one line break.

        Each entry is ``(key, line_idxs)``: 2..``MAX_WORDS_BY_KIND["id"]`` consecutive words in
        page reading order, joined without spaces and upper-cased (the scorer's id normalization).
        Runs inside one line are already spans, so only line-crossing runs are new: they cover a
        part number that OCR split over two lines.
        """
        if page not in self._runs:
            words = [
                (w, it.line_idx) for it in page_items(self.pages[page]) for w in it.text.split()
            ]
            out: list[tuple[str, tuple[int, ...]]] = []
            for i in range(len(words)):
                for j in range(i + 1, min(i + MAX_WORDS_BY_KIND["id"], len(words))):
                    seg = words[i : j + 1]
                    lines = tuple(dict.fromkeys(ln for _, ln in seg))
                    if len(lines) > 1:
                        out.append((_nospace_upper("".join(w for w, _ in seg)), lines))
            self._runs[page] = out
        return self._runs[page]

    def table(self, form_key: str) -> _FormTable:
        """Form table for `form_key`, built on first use."""
        t = self._tables.get(form_key)
        if t is not None:
            return t
        t = _FormTable()
        maxw = _FORM_KIND_MAXW[form_key]
        fn = _FORMS[form_key]
        order = sorted(
            range(len(self.spans)),
            key=lambda i: (len(self.spans[i].text), self.spans[i].page, self.spans[i].line_idx,
                           self.spans[i].box[0]),
        )  # fmt: skip
        for i in order:
            sp = self.spans[i]
            if sp.src == "word" and sp.text.count(" ") + 1 > maxw:
                continue
            forms = [(k, x, False) for k, x in fn(sp.text)]
            if form_key in _REPAIR_FORMS:
                forms += [(k, x, True) for k, x in _REPAIR_FORMS[form_key](sp.text)]
            for key, text, repaired in forms:
                if repaired:
                    t.repaired.add(len(t.keys))
                t.by_key.setdefault(key, []).append(len(t.keys))
                t.keys.append(key)
                t.owners.append(i)
                t.texts.append(text)
        t.uniq = list(t.by_key)
        self._tables[form_key] = t
        return t


def build_index(pages: list[PageOcr]) -> DocIndex:
    """Index a document's pages once; reuse it for every field you locate in that document."""
    return DocIndex(pages)


# ---------------------------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------------------------


def _exact_re(kind: str, g: str) -> re.Pattern[str] | None:
    if kind == "name":
        return None
    body = re.escape(g)
    if kind == "num":
        return re.compile(rf"(?<![0-9A-Za-z])(?<!\d[.,]){body}(?![0-9A-Za-z])(?![.,]\d)")
    return re.compile(rf"(?<![0-9A-Za-z]){body}(?![0-9A-Za-z])")


def find_matches(
    value: Any,
    field_name: str,
    doc: DocIndex | list[PageOcr],
    threshold: float | None = None,
    max_level: Level = "fuzzy",
) -> list[Match]:
    """Best match per (page, line), best first. Empty list if the value is empty or not found.

    Order: level (exact < normalized < fuzzy), then score, then shorter span, then reading order.
    `max_level` caps the levels tried ("normalized" skips fuzzy matching).
    """
    if value is None or not str(value).strip():
        return []
    sc = _sc()
    index = doc if isinstance(doc, DocIndex) else build_index(doc)
    g = str(value).strip()
    kind = str(sc.KIND.get(field_name, "id"))
    fkey = _form_key(field_name, kind)
    thr = FUZZY_THRESHOLD_BY_KIND.get(kind, FUZZY_THRESHOLD) if threshold is None else threshold
    cap = LEVEL_RANK[max_level]
    best: dict[tuple[int, int], tuple[tuple[int, float, int], Match]] = {}

    def offer(span: Span, level: Level, score: float, text: str) -> None:
        rank = (LEVEL_RANK[level], -score, len(span.text))
        k = (span.page, span.line_idx)
        cur = best.get(k)
        if cur is None or rank < cur[0]:
            best[k] = (rank, Match(span.page, span.box, span.line_idx, level, score, span.text))

    # (a) exact substring
    rx = _exact_re(kind, g)
    for sp in index.spans:
        if rx.search(sp.text) if rx else g in sp.text:
            offer(sp, "exact", 100.0, g)
    table = index.table(fkey)
    # (b) scorer-normalized equality
    if cap >= 1:
        key = _gold_key(sc, fkey, g)
        for ei in table.by_key.get(key, []) if key is not None else []:
            if ei not in table.repaired and sc.same(field_name, table.texts[ei], g):
                offer(index.spans[table.owners[ei]], "normalized", 100.0, g)
    # (c) fuzzy over scorer-normalized strings
    if cap >= 2:
        key = _gold_key(sc, fkey, g)
        if key is not None and table.uniq:
            for cand, score, _ in process.extract(
                key, table.uniq, scorer=fuzz.ratio, score_cutoff=thr, limit=None
            ):
                for ei in table.by_key[cand]:
                    offer(index.spans[table.owners[ei]], "fuzzy", float(score), g)
    out = sorted(
        best.values(),
        key=lambda t: (t[0], t[1].page, t[1].line_idx, t[1].box[0]),
    )
    return [m for _, m in out]


def locate(
    value: Any,
    field_name: str,
    doc: DocIndex | list[PageOcr],
    threshold: float | None = None,
    max_level: Level = "fuzzy",
) -> Match | None:
    """Best OCR span for `value` in `doc` (pages or a prebuilt `DocIndex`), or None."""
    ms = find_matches(value, field_name, doc, threshold, max_level)
    return ms[0] if ms else None


def best_fuzzy_score(value: Any, field_name: str, doc: DocIndex | list[PageOcr]) -> float:
    """Highest ratio (0-100) of the value's normalized string over every span form; 0 if none.

    Used for threshold selection. An exact scorer match scores 100.
    """
    if value is None or not str(value).strip():
        return 0.0
    sc = _sc()
    index = doc if isinstance(doc, DocIndex) else build_index(doc)
    g = str(value).strip()
    fkey = _form_key(field_name, str(sc.KIND.get(field_name, "id")))
    key = _gold_key(sc, fkey, g)
    table = index.table(fkey)
    if key is None or not table.uniq:
        return 0.0
    hit = process.extractOne(key, table.uniq, scorer=fuzz.ratio)
    return float(hit[1]) if hit else 0.0


# ---------------------------------------------------------------------------------------------
# Row assignment
# ---------------------------------------------------------------------------------------------

ROW_SUPPORT_FIELDS = ("quantity", "customer_part_number", "purchase_order")
AssignLevel = Literal["line", "page_fuzzy", "unassigned"]


@dataclass(frozen=True)
class RowAssignment:
    """Where one gold row sits: its part-number match and which other fields share that line."""

    row_idx: int
    match: Match | None  # supplier_part_number match on the assigned line
    on_line: tuple[str, ...]  # other row fields found (level <= normalized) on that line
    reason: str  # "ok" | "spn_not_located" | "no_free_line" (why no distinct line was found)
    # "line": own OCR line (match set); "page_fuzzy": no distinct line, but the part number is
    # present on `page` (match stays None, so there is no box); "unassigned": neither.
    level: AssignLevel = "unassigned"
    page: int | None = None


def assign_rows(
    rows: list[dict[str, Any]],
    doc: DocIndex | list[PageOcr],
    threshold: float | None = None,
) -> list[RowAssignment]:
    """Give each gold row its own OCR line where possible.

    The part number picks the candidate lines (any level up to fuzzy); quantity, customer part and
    PO then break ties on the same line. Rows are served most-constrained first (fewest candidate
    lines), each taking the best unused line by (match level, number of other fields on the line,
    reading order). A repeated part number therefore lands on distinct lines; a row whose
    candidate lines are all taken is reported as ``no_free_line`` rather than sharing a line.

    Fallback for rows left without a line (``match`` None): if the part number is present on some
    page (``_page_presence``: fuzzy >= threshold over that page's spans and line-crossing word
    runs) the row gets ``level="page_fuzzy"`` and that page, still with ``match=None``. Several
    pages: prefer one where the row's quantity appears on a line no other row took, then one with
    unassigned capacity (places minus rows already placed there), then the earliest. Otherwise
    ``level="unassigned"``. ``reason`` keeps the original cause either way.
    """
    index = doc if isinstance(doc, DocIndex) else build_index(doc)
    cache: dict[tuple[str, str], dict[tuple[int, int], Match]] = {}

    def per_line(value: Any, field_name: str, max_level: Level) -> dict[tuple[int, int], Match]:
        k = (field_name, str(value))
        if k not in cache:
            cache[k] = {
                (m.page, m.line_idx): m
                for m in find_matches(value, field_name, index, threshold, max_level)
            }
        return cache[k]

    def support(row: dict[str, Any], line: tuple[int, int]) -> tuple[str, ...]:
        found = []
        for f in ROW_SUPPORT_FIELDS:
            v = row.get(f)
            if v is not None and str(v).strip() and line in per_line(v, f, "normalized"):
                found.append(f)
        return tuple(found)

    cands: dict[int, dict[tuple[int, int], Match]] = {}
    for i, row in enumerate(rows):
        v = row.get("supplier_part_number")
        cands[i] = per_line(v, "supplier_part_number", "fuzzy") if v is not None else {}
    used: set[tuple[int, int]] = set()
    out: dict[int, RowAssignment] = {}
    for i in sorted(range(len(rows)), key=lambda r: (len(cands[r]), r)):
        if not cands[i]:
            out[i] = RowAssignment(i, None, (), "spn_not_located")
            continue
        free = [(ln, m) for ln, m in cands[i].items() if ln not in used]
        if not free:
            out[i] = RowAssignment(i, None, (), "no_free_line")
            continue
        scored = [
            (LEVEL_RANK[m.level], -len(support(rows[i], ln)), -m.score, ln, m) for ln, m in free
        ]
        _, _, _, ln, m = min(scored, key=lambda t: t[:4])
        used.add(ln)
        out[i] = RowAssignment(i, m, support(rows[i], ln), "ok", "line", m.page)
    thr = FUZZY_THRESHOLD if threshold is None else threshold
    paged: Counter[tuple[int, str]] = Counter()  # (page, part key) -> rows placed by page_fuzzy
    for i in range(len(rows)):  # fallback, in row order so the outcome is deterministic
        if out[i].match is not None:
            continue
        v = rows[i].get("supplier_part_number")
        if v is None or not str(v).strip():
            continue
        key = _nospace_upper(str(v))
        present = _page_presence(index, key, cands[i], thr)
        if not present:
            continue
        qty = rows[i].get("quantity")
        qty_pages = (
            {p for (p, ln) in per_line(qty, "quantity", "normalized") if (p, ln) not in used}
            if qty is not None and str(qty).strip()
            else set()
        )

        # Places on a page minus rows already there: lines taken by line-level rows, and rows
        # this fallback placed on it for the same part.
        capacity = {
            pg: n - sum(1 for ln in cands[i] if ln[0] == pg and ln in used) - paged[(pg, key)]
            for pg, n in present.items()
        }
        best = min(present, key=lambda pg: (pg not in qty_pages, capacity[pg] <= 0, pg))
        paged[(best, key)] += 1
        out[i] = RowAssignment(i, None, (), out[i].reason, "page_fuzzy", best)
    return [out[i] for i in range(len(rows))]


def _page_presence(
    index: DocIndex, key: str, line_cands: dict[tuple[int, int], Match], thr: float
) -> dict[int, int]:
    """Pages where the scorer-normalized part number `key` is present, with a count of places.

    A place is a line the locator already matched (any level up to fuzzy), or a line-crossing
    word run whose normalized text scores ``>= thr`` (``fuzz.ratio``) against `key`.
    """
    present: Counter[int] = Counter(p for (p, _) in line_cands)
    for pg in range(len(index.pages)):
        for run_key, _lines in index.page_runs(pg):
            if fuzz.ratio(key, run_key) >= thr:
                present[pg] += 1
    return dict(present)
