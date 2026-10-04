"""Unit tests for scripts/history_leak_scan.py (matcher, path flags, stream parser)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "history_leak_scan", ROOT / "scripts" / "history_leak_scan.py"
)
scan = importlib.util.module_from_spec(spec)
sys.modules["history_leak_scan"] = scan
spec.loader.exec_module(scan)


def test_aho_corasick_matches_like_naive_substring_search() -> None:
    pats = {"he": {"a"}, "she": {"b"}, "his": {"c"}, "hers": {"d"}, "ZZ-123456": {"e"}}
    ac = scan.AhoCorasick(pats)
    for text in ("ushers", "ahishers", "xZZ-123456x", "ZZ-12345", "", "hhh", "sheshe"):
        expected = {k for p, ks in pats.items() if p in text for k in ks}
        assert ac.kinds_in(text) == expected, text


@pytest.mark.parametrize(
    ("path", "reason"),
    [
        ("data/train/labels/x.json", "forbidden_dir"),
        ("assignment/score.py", "forbidden_dir"),
        ("runs/a/b.json", "forbidden_dir"),
        ("cache/ocr/p.json", "forbidden_dir"),
        ("docs/figure.PNG", "forbidden_type"),
        ("x/y.zip", "forbidden_type"),
        ("src/shipdoc/validate.py", None),
        ("scripts/data_utils.py", None),
    ],
)
def test_flagged_path_kind(path: str, reason: str | None) -> None:
    assert scan.flagged_path_kind(path) == reason


def test_scan_lines_parses_commits_added_and_deleted_lines_and_paths() -> None:
    log = b"""@@COMMIT@@aaaa
diff --git a/src/old.py b/src/old.py
--- a/src/old.py
+++ b/src/old.py
@@ -1 +0,0 @@
-secret = "ZZPLANT-12345"
@@COMMIT@@bbbb
diff --git a/runs/x.json b/runs/x.json
--- a/runs/x.json
+++ b/runs/x.json
@@ -0,0 +1 @@
+unrelated line
diff --git a/src/new.py b/src/new.py
+++ b/src/new.py
+++ ZZPLANT-12345 is a header-looking line and must be ignored
+clean
""".splitlines(keepends=True)
    ac = scan.AhoCorasick({"ZZPLANT-12345": {"header.x"}})
    values, paths, commits = scan.scan_lines(log, ac)
    assert commits == 2
    assert values == {("aaaa", "src/old.py"): {"header.x"}}
    assert paths == {("bbbb", "runs/x.json"): "forbidden_dir"}


def test_collect_values_applies_the_length_and_date_rules(tmp_path: Path) -> None:
    import json

    labels = tmp_path / "train" / "labels"
    labels.mkdir(parents=True)
    doc = {
        "header": {
            "supplier_name": "Acme Fasteners Ltd",
            "invoice_number": "INV-1",  # too short
            "invoice_date": "2024-01-02",  # not a scanned kind
            "currency": "USDXXX",  # not a scanned kind
            "total_amount": 123456.5,  # not a string
        },
        "line_items": [{"purchase_order": " PO-778899 ", "quantity": "1234567"}],
    }
    (labels / "d.json").write_text(json.dumps(doc), encoding="utf-8")
    got = scan.collect_values(tmp_path)
    assert got == {
        "Acme Fasteners Ltd": {"header.supplier_name"},
        "PO-778899": {"row.purchase_order"},
    }
