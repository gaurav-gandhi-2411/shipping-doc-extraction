from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "make_colab_zips", ROOT / "scripts/make_colab_zips.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def _tree(root: Path) -> None:
    (root / "paddleocr" / "dev").mkdir(parents=True)
    (root / "logs").mkdir()
    (root / "paddleocr" / "dev" / "b.json").write_text("{}")
    (root / "paddleocr" / "dev" / "a.json").write_text("[1]")
    (root / "logs" / "x.log").write_text("noise")
    (root / "SHA256SUMS").write_text(
        "".join(
            f"{mod.sha256_file(root / r)}  {r}\n"
            for r in ("paddleocr/dev/a.json", "paddleocr/dev/b.json")
        )
    )


def test_zip_is_deterministic_sorted_and_excludes_logs(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _tree(src)
    files = mod.collect(src)
    a, b = tmp_path / "a.zip", tmp_path / "b.zip"
    mod.write_zip(a, src, "ocr_cache", files)
    (src / "paddleocr" / "dev" / "a.json").touch()  # mtime change must not matter
    mod.write_zip(b, src, "ocr_cache", mod.collect(src))
    assert mod.sha256_file(a) == mod.sha256_file(b)
    with zipfile.ZipFile(a) as zf:
        names = zf.namelist()
        assert names == sorted(names)
        assert names[0].startswith("ocr_cache/") and not any("logs" in n for n in names)
        assert {i.date_time for i in zf.infolist()} == {mod.FIXED_DATE}


def test_verify_sums_detects_corruption(tmp_path: Path) -> None:
    _tree(tmp_path)
    assert mod.verify_sums(tmp_path) == []
    (tmp_path / "paddleocr" / "dev" / "a.json").write_text("tampered")
    assert mod.verify_sums(tmp_path) == ["checksum mismatch: paddleocr/dev/a.json"]
    (tmp_path / "SHA256SUMS").unlink()
    assert "missing" in mod.verify_sums(tmp_path)[0]
