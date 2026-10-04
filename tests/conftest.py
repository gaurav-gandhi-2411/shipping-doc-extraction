"""Suite-wide pytest hooks: skip (with a reason) the tests that need the evaluators' package.

See ``tests/_requires.py``. Nothing is skipped when ``assignment/`` is present (the private
repository), so the listed tests keep running and failing there exactly as before.
"""

from __future__ import annotations

import pytest
from _requires import SKIP_REASON, missing_inputs, needs_assignment


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Mark every test of ``NEEDS_ASSIGNMENT`` as skipped while an input file is missing."""
    missing = missing_inputs()
    if not missing:
        return
    skip = pytest.mark.skip(reason=f"{SKIP_REASON} [missing: {', '.join(missing)}]")
    for item in items:
        module = getattr(item, "module", None)
        name = getattr(item, "originalname", None)
        if module is not None and name and needs_assignment(module.__name__, name):
            item.add_marker(skip)
