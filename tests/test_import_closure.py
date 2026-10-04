"""scripts/import_closure.py on a synthetic package (every import shape) and on the real one."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "import_closure", ROOT / "scripts" / "import_closure.py"
)
ic = importlib.util.module_from_spec(spec)
sys.modules["import_closure"] = ic
spec.loader.exec_module(ic)


@pytest.fixture
def pkg(tmp_path: Path) -> Path:
    files = {
        "__init__.py": "",
        "a.py": "import shipdoc.b\n",
        "b.py": "from shipdoc import c\n",
        "c.py": "def f():\n    from .d import x  # a lazy, relative import\n    return x\n",
        "d.py": "x = 1\nimport json\n",
        "e.py": "from shipdoc.sub import g\n",
        "sub/__init__.py": "",
        "sub/g.py": "from .. import h\n",
        "h.py": "",
        "lonely.py": "import os\n",
    }
    for rel, text in files.items():
        (tmp_path / "shipdoc" / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "shipdoc" / rel).write_text(text, encoding="utf-8")
    return tmp_path


def test_every_import_shape_is_followed(pkg: Path) -> None:
    closed = ic.closure(pkg, "shipdoc.a")  # `from shipdoc import c` also loads the package itself
    assert closed == {"shipdoc", "shipdoc.a", "shipdoc.b", "shipdoc.c", "shipdoc.d"}
    assert ic.closure(pkg, "shipdoc.e") == {
        "shipdoc",
        "shipdoc.e",
        "shipdoc.sub",
        "shipdoc.sub.g",
        "shipdoc.h",
    }
    assert ic.closure(pkg, "shipdoc.lonely") == {"shipdoc.lonely"}


def test_exit_codes_and_the_listing(pkg: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base = ["--src", str(pkg)]
    assert ic.main([*base, "--entry", "shipdoc.a", "--changed", "shipdoc.e", "shipdoc.h"]) == 0
    assert "reaches changed: []" in capsys.readouterr().out
    assert ic.main([*base, "--entry", "shipdoc.a", "--changed", "shipdoc.d"]) == 1
    assert "reaches changed: ['shipdoc.d']" in capsys.readouterr().out
    assert ic.main([*base, "--entry", "shipdoc.zzz", "--changed", "shipdoc.d"]) == 2


def test_the_real_decoding_and_assembly_modules_do_not_reach_the_flag_stage_files() -> None:
    for entry in ("shipdoc.__main__", "shipdoc.predict", "shipdoc.spike", "shipdoc.postrules"):
        reached = ic.closure(ic.DEFAULT_SRC, entry) & {"shipdoc.flags", "shipdoc.predict_native"}
        assert not reached, (entry, reached)
