"""Markdown -> PDF for the final report, failing (exit 1) if it is longer than 2 pages.

    uv run python scripts/report_to_pdf.py reports/final/report.md $SHIPDOC_TMP_DIR/report.pdf
    uv run python scripts/report_to_pdf.py --dummy $SHIPDOC_TMP_DIR/report_dummy.pdf
    uv run python scripts/report_to_pdf.py --final-system both --out-dir $SHIPDOC_TMP_DIR/report_out
    uv run python scripts/report_to_pdf.py --final-system v2 --fill-test --out-dir D:/scratch

``--final-system`` builds ``report_v1_5.pdf`` / ``report_v2.pdf`` (and their markdown) from the
numbers registry; a report with an unresolved placeholder carries a diagonal DRAFT watermark.

Engine: ``pandoc`` + ``tectonic`` (or ``xelatex``) found on PATH. Neither is in the locked uv
environment (``uv.lock`` has no PDF engine and no weasyprint), so this is a system-tool
dependency; the script exits 2 with the missing tool named instead of installing anything.
The PDF is a build output: write it OUTSIDE the repo or to a gitignored path
(``reports/final/*.pdf`` is gitignored); it must never be committed.

``--dummy`` builds the DUMMY fixture (``build_report.dummy_context``) into a scratch markdown next
to the PDF and proves the layout fits; the dummy markdown never goes to reports/final/report.md.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_report as br  # noqa: E402  (sibling script, not a package)

MAX_PAGES = 2
# Plain LaTeX only (no extra packages: tectonic would have to fetch them). Small type, tight
# paragraphs and headings so that eight sections and five tables fit two A4 pages.
HEADER_TEX = r"""
\makeatletter
\AtBeginDocument{\footnotesize\setlength{\parskip}{2pt}\setlength{\parindent}{0pt}
  \setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{0.95}
  \ifdefined\LTpre\setlength{\LTpre}{1pt}\setlength{\LTpost}{1pt}\fi}
\renewcommand\section{\@startsection{section}{1}{0pt}{3pt}{1pt}{\normalfont\bfseries\normalsize}}
\renewcommand\subsection{\@startsection{subsection}{2}{0pt}{4pt}{1pt}{\normalfont\bfseries\small}}
\makeatother
\usepackage{etoolbox}
\AtBeginEnvironment{longtable}{\scriptsize}
"""


# Diagonal watermark on every page of a report with unresolved placeholders (a DRAFT). eso-pic,
# xcolor and graphicx come with the tectonic bundle (checked 2026-10-04); the tint is light so the
# text under it stays readable, and the banner line of the markdown says the same in plain text.
WATERMARK_TEX = r"""
\usepackage{eso-pic}
\usepackage{xcolor}
\usepackage{graphicx}
\AddToShipoutPictureBG{\AtPageCenter{\makebox(0,0){\rotatebox{50}{\scalebox{4.5}{%
  \textcolor[rgb]{1,0.78,0.78}{\textbf{@@TEXT@@}}}}}}}
"""


class PdfError(Exception):
    """The PDF could not be built or inspected."""


def assert_max_pages(n: int, limit: int = MAX_PAGES) -> None:
    """Raise `PdfError` when a PDF has more than `limit` pages (the brief allows two)."""
    if n > limit:
        raise PdfError(f"{n} pages exceeds the {limit}-page limit")


def count_pdf_pages(data: bytes) -> int:
    """Count page objects (``/Type /Page``) in a PDF, including ones inside object streams.

    No PDF library is installed, so this parses by hand: it searches the raw bytes and the
    inflated content of every ``/ObjStm`` stream (xdvipdfmx/tectonic write PDF 1.5 object
    streams, which hide ``/Type /Page`` from a plain grep). ``/Type /Pages`` is excluded.
    """
    page = re.compile(rb"/Type\s*/Page(?![A-Za-z])")
    n = len(page.findall(data))
    for m in re.finditer(
        rb"<<(?:(?!endobj).)*?/Type\s*/ObjStm(?:(?!endobj).)*?>>\s*stream\r?\n", data, re.DOTALL
    ):
        start = m.end()
        end = data.find(b"endstream", start)
        try:
            # decompressobj stops at the end of the deflate stream, so the EOL before ``endstream``
            # need not be stripped; rstrip(b"\r\n") also ate a final data byte equal to 0x0a/0x0d
            # (a flaky "incomplete or truncated stream", seen 2026-10-03).
            n += len(page.findall(zlib.decompressobj().decompress(data[start:end])))
        except zlib.error as e:
            raise PdfError(f"cannot inflate an object stream while counting pages: {e}") from e
    if n == 0:
        raise PdfError("no page objects found: not a PDF or an unsupported structure")
    return n


def find_engine() -> tuple[str, str]:
    """(pandoc path, pdf-engine name); raises `PdfError` naming what is missing."""
    pandoc = shutil.which("pandoc")
    if pandoc is None:
        raise PdfError("pandoc not on PATH (needed with tectonic or xelatex); nothing installed")
    for eng in ("tectonic", "xelatex"):
        if shutil.which(eng):
            return pandoc, eng
    raise PdfError("neither tectonic nor xelatex on PATH; nothing installed")


_PENDING_RE = re.compile(r"\[\[PENDING: [^\]]*\]\]")


def compact_table_pending(text: str) -> str:
    """Shorten ``[[PENDING: long.dotted.name]]`` to ``[[PENDING]]`` inside table rows only.

    A dotted name is one unbreakable word: in a narrow cell it overprints its neighbours and runs
    off the page. The names stay in the markdown (and in ``build_report.py --strict``); prose
    lines are left alone because they wrap.
    """
    return "\n".join(
        _PENDING_RE.sub("[[PENDING]]", ln) if ln.lstrip().startswith("|") else ln
        for ln in text.split("\n")
    )


def build_pdf(md: Path, pdf: Path, watermark: bool = False) -> int:
    """Run pandoc; returns the page count of the written PDF.

    `watermark` adds the diagonal draft watermark (`br.WATERMARK_TEXT`) to every page.
    """
    pandoc, engine = find_engine()
    pdf.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        hdr = Path(tmp) / "header.tex"
        header = HEADER_TEX
        if watermark:
            header += WATERMARK_TEX.replace("@@TEXT@@", br.WATERMARK_TEXT)
        hdr.write_text(header, encoding="utf-8")
        src = Path(tmp) / "report.md"
        src.write_text(compact_table_pending(md.read_text(encoding="utf-8")), encoding="utf-8")
        cmd = [
            pandoc,
            str(src),
            "-o",
            str(pdf),
            f"--pdf-engine={engine}",
            "-H",
            str(hdr),
            "-V",
            "geometry:a4paper,margin=1.2cm",
            "-V",
            "colorlinks=false",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
    if r.returncode != 0:
        raise PdfError(f"pandoc failed ({r.returncode}): {r.stderr[-1500:]}")
    return count_pdf_pages(pdf.read_bytes())


def build_final_pdf(
    system: str,
    out_dir: Path,
    max_pages: int = MAX_PAGES,
    fill_test: bool = False,
    fill_variant: str = "",
) -> tuple[Path, int, bool]:
    """Fill the final report for `system` and build ``report_<system>.pdf`` in `out_dir`.

    Returns (pdf path, page count, watermarked). `out_dir` must be outside the repo (a PDF is a
    build output and is never committed); more than `max_pages` pages raises `PdfError`.
    `fill_test` fills every placeholder with its realistic-length dummy (scratch layout check:
    files are named ``report_<system>_filltest.*``; never a report).
    """
    if out_dir.resolve().is_relative_to(br.ROOT):
        raise PdfError("the PDF output folder must be outside the repo (PDFs are never committed)")
    res = br.build_final(system, fill_test=fill_test, fill_variant=fill_variant)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{system}_filltest{'_' + fill_variant if fill_variant else ''}" if fill_test else system
    md = out_dir / f"report_{tag}.md"
    pdf = out_dir / f"report_{tag}.pdf"
    md.write_text(res.text, encoding="utf-8", newline="\n")
    mark = br.needs_watermark(res.unresolved)
    n = build_pdf(md, pdf, watermark=mark)
    assert_max_pages(n, max_pages)
    return pdf, n, mark


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="*", type=Path, help="[md] pdf  (md omitted with --dummy)")
    ap.add_argument("--dummy", action="store_true", help="build the DUMMY fixture instead of md")
    ap.add_argument(
        "--final-system",
        choices=(*br.FINAL_SYSTEMS, "both"),
        default=None,
        help="build the final report for v1_5, v2 or both into --out-dir",
    )
    ap.add_argument("--out-dir", type=Path, default=None, help="folder for --final-system")
    ap.add_argument("--max-pages", type=int, default=MAX_PAGES)
    ap.add_argument(
        "--fill-test",
        action="store_true",
        help="with --final-system: fill placeholders with realistic-length DUMMIES (layout only)",
    )
    ap.add_argument(
        "--fill-variant",
        default="",
        help="with --fill-test: use fill_test_<variant> dummies (no08 = no final-adapter dev run)",
    )
    a = ap.parse_args(argv)
    if (a.fill_test or a.fill_variant) and not a.final_system:
        ap.error("--fill-test / --fill-variant need --final-system")
    if a.fill_variant and not a.fill_test:
        ap.error("--fill-variant needs --fill-test")
    if a.final_system:
        if a.out_dir is None:
            ap.error("--final-system needs --out-dir (outside the repo)")
        systems = br.FINAL_SYSTEMS if a.final_system == "both" else (a.final_system,)
        rc = 0
        for s in systems:
            try:
                pdf, n, mark = build_final_pdf(
                    s, a.out_dir, a.max_pages, a.fill_test, a.fill_variant
                )
            except (PdfError, br.ReportError) as e:
                print(f"ERROR ({s}): {e}", file=sys.stderr)
                rc = max(rc, 1 if "pages" in str(e) else 2)
                continue
            print(f"{pdf}: {n} page(s), limit {a.max_pages}, watermark {mark}")
        return rc
    if not a.paths:
        ap.error("give: report.md out.pdf  (or --dummy out.pdf, or --final-system)")
    try:
        if a.dummy:
            if len(a.paths) != 1:
                ap.error("--dummy takes exactly one path: the output pdf")
            pdf = a.paths[0]
            md = pdf.with_suffix(".md")
            if pdf.resolve().is_relative_to(br.ROOT):
                ap.error("--dummy output must be outside the repo (dummy numbers never committed)")
            text, left = br.fill(br.TEMPLATE.read_text(encoding="utf-8"), br.dummy_context())
            md.parent.mkdir(parents=True, exist_ok=True)
            md.write_text(text, encoding="utf-8")
            print(f"dummy markdown: {md} (unresolved {len(left)})")
        else:
            if len(a.paths) != 2:
                ap.error("give: report.md out.pdf")
            md, pdf = a.paths
        n = build_pdf(md, pdf)
    except PdfError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    print(f"{pdf}: {n} page(s), limit {a.max_pages}")
    if n > a.max_pages:
        print(f"FAIL: {n} pages exceeds the {a.max_pages}-page limit", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
