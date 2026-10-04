from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_spike_set", ROOT / "scripts/make_spike_set.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

META = json.loads((ROOT / "meta" / "dev.json").read_text(encoding="utf-8"))


def test_spike_quotas_and_determinism() -> None:
    ids = mod.select_spike(META)
    assert ids == mod.select_spike(list(reversed(META)))  # input order must not matter
    assert len(ids) == len(set(ids)) == 40
    by = {r["doc_id"]: r for r in META}
    sel = [by[i] for i in ids]
    assert sum(r["illegible"] for r in sel) >= 5
    assert sum(r["waybill"] for r in sel) >= 10
    assert sum(r["scanned"] for r in sel) >= 15
    assert sum(r["multipage"] for r in sel) >= 10
    assert len({r["supplier_group"] for r in sel}) == len({r["supplier_group"] for r in META})


def test_committed_splits_match_script() -> None:
    assert json.loads((ROOT / "splits/spike40.json").read_text()) == mod.select_spike(META)
    dev = json.loads((ROOT / "splits/dev100.json").read_text())
    assert dev == sorted(r["doc_id"] for r in META) and len(dev) == 100
