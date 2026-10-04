"""Static import closure of ``shipdoc`` entry modules, and which changed modules they reach.

Answers "can a change to these files reach what that entry point runs?" without running anything:
every ``import`` statement of every module, at any depth (function-level imports that only execute
on a code path included), is followed inside the ``shipdoc`` package. Used by
``reports/public_code_equivalence.md`` for the three files that differ from the original pin.

Run: uv run python scripts/import_closure.py --changed shipdoc.gate shipdoc.flags \
        shipdoc.predict_native --entry shipdoc.__main__ shipdoc.predict shipdoc.spike

Exit code: 0 no entry reaches a changed module; 1 at least one does (listed); 2 usage error.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = ROOT / "src"


def module_file(src: Path, name: str) -> Path | None:
    """The source file of ``shipdoc.<...>`` under ``src`` (a package ``__init__`` or a module)."""
    parts = name.split(".")
    if parts[0] != "shipdoc":
        return None
    base = src.joinpath(*parts)
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    return None


def imports_of(src: Path, name: str) -> set[str]:
    """``shipdoc`` modules named by the import statements of module ``name`` (any depth)."""
    path = module_file(src, name)
    if path is None:
        return set()
    package = name if path.name == "__init__.py" else name.rsplit(".", 1)[0]
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:  # relative: strip level - 1 trailing parts of the package
                parts = package.split(".")[: len(package.split(".")) - (node.level - 1)]
                base = ".".join([*parts, *([node.module] if node.module else [])])
            found.add(base)
            found.update(f"{base}.{a.name}" for a in node.names)  # `from shipdoc import flags`
    return {m for m in found if module_file(src, m) is not None}


def closure(src: Path, entry: str) -> set[str]:
    """``entry`` and every ``shipdoc`` module reachable from it."""
    seen: set[str] = set()
    todo = [entry]
    while todo:
        mod = todo.pop()
        if mod not in seen:
            seen.add(mod)
            todo.extend(imports_of(src, mod) - seen)
    return seen


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC, help="folder holding shipdoc/")
    ap.add_argument("--entry", nargs="+", required=True, help="entry modules, e.g. shipdoc.predict")
    ap.add_argument("--changed", nargs="+", required=True, help="modules whose reach is checked")
    args = ap.parse_args(argv)
    missing = [m for m in (*args.entry, *args.changed) if module_file(args.src, m) is None]
    if missing:
        print(f"unknown module(s) under {args.src}: {missing}", file=sys.stderr)
        return 2
    reached_any = False
    for entry in args.entry:
        reached = sorted(closure(args.src, entry) & set(args.changed))
        reached_any |= bool(reached)
        n = len(closure(args.src, entry))
        print(f"{entry}: closure {n} modules; reaches changed: {reached}")
    return 1 if reached_any else 0


if __name__ == "__main__":
    sys.exit(main())
