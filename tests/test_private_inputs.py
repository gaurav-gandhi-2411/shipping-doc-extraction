"""The skip mechanism of tests/_requires.py: its registry is current, and it cannot hide failures.

Two kinds of test. The registry checks run everywhere. ``test_private_tree_has_every_input`` runs
only in the private repository (recognised by the publication tooling, which the public tree does
not contain): there the evaluators' files and the data must exist, otherwise every listed test
would skip silently and the suite would be green while testing nothing.
"""

from __future__ import annotations

import ast
from pathlib import Path

import _requires as rq
import pytest

from shipdoc import paths

ROOT = Path(__file__).resolve().parents[1]
IS_PRIVATE_TREE = (ROOT / "scripts" / "publish_overlay.py").is_file()


@pytest.mark.skipif(not IS_PRIVATE_TREE, reason="private repository only (has the inputs)")
def test_private_tree_has_every_input() -> None:
    assert rq.missing_inputs() == []  # nothing in NEEDS_ASSIGNMENT is skipped in this tree
    for split in ("train", "dev"):
        assert (paths.data_dir() / split / "labels").is_dir(), f"data/{split}/labels is missing"


def test_every_registry_entry_names_a_test_that_exists() -> None:
    """A renamed or deleted test must not leave a stale name behind (modules absent are skipped)."""
    stale = []
    for module, names in rq.NEEDS_ASSIGNMENT.items():
        path = ROOT / "tests" / f"{module}.py"
        if not path.is_file():  # e.g. a test removed from the public tree
            continue
        defined = {
            n.name
            for n in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        stale += [f"{module}::{n}" for n in names if n not in defined]
    assert stale == []


def test_the_helper_matches_module_and_test_name_exactly() -> None:
    module = next(iter(rq.NEEDS_ASSIGNMENT))
    name = rq.NEEDS_ASSIGNMENT[module][0]
    assert rq.needs_assignment(module, name)
    assert not rq.needs_assignment(module, name + "_x")
    assert not rq.needs_assignment("test_no_such_module", name)


def test_missing_inputs_follow_the_assignment_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"  # stands for the repository: <root>/assignment is the other location
    (root / "assignment").mkdir(parents=True)
    monkeypatch.setenv("SHIPDOC_ASSIGNMENT_DIR", str(root / "assignment"))
    monkeypatch.delenv(rq.SCORER_ENV, raising=False)
    both = ["assignment/schema.json", "assignment/score.py"]
    assert rq.missing_inputs(root) == both
    (root / "assignment" / "schema.json").write_text("{}", encoding="utf-8")
    assert rq.missing_inputs(root) == ["assignment/score.py"]
    (root / "assignment" / "score.py").write_text("", encoding="utf-8")
    assert rq.missing_inputs(root) == []
    other = tmp_path / "elsewhere.py"  # the scorer env override wins, as in eval.load_scorer
    monkeypatch.setenv(rq.SCORER_ENV, str(other))
    assert rq.missing_inputs(root) == ["assignment/score.py"]


def test_inputs_found_only_through_the_environment_still_count_as_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The test_reuse_v0 failure: $SHIPDOC_ASSIGNMENT_DIR is set but <repo>/assignment is not."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    for name in ("schema.json", "score.py"):
        (elsewhere / name).write_text("", encoding="utf-8")
    monkeypatch.setenv("SHIPDOC_ASSIGNMENT_DIR", str(elsewhere))
    monkeypatch.delenv(rq.SCORER_ENV, raising=False)
    empty_root = tmp_path / "repo"
    empty_root.mkdir()
    assert rq.missing_inputs(empty_root) == ["assignment/schema.json", "assignment/score.py"]


def test_the_reuse_v0_test_that_assembles_against_the_schema_is_guarded() -> None:
    """Regression for the public-tree failure: it must skip when the schema file is absent."""
    from test_reuse_v0 import test_submission_folder_inside_the_repo_is_refused as t

    marks = [m for m in getattr(t, "pytestmark", []) if m.name == "skipif"]
    assert marks, "the test builds a v0 against assignment/schema.json and needs a skip guard"
