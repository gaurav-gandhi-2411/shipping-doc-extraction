# OCR cache (Phase 2, step 1)

Engine: **PaddleOCR 3.0.3 / PaddlePaddle 3.0.0 (CPU) / PaddleX 3.0.3**, PP-OCRv5 **server** detection
and recognition, plus document-orientation and text-line-orientation classifiers.
Cache: `cache/ocr/paddleocr/<split>/<page_stem>.json` and `cache/ocr/paddleocr/manifest.json`
(gitignored; regenerate with `python -m shipdoc ocr --splits train dev test`).

PaddleOCR returns text **lines**, not words. Items are stored with `granularity: "line"`. The `words`
array holds approximate word boxes made by splitting each line quad proportionally to character
counts (`derived: true`). PP-OCRv5 often drops inter-word spaces (`'Total Amount:CNY897,159.11'`),
so derived words are only as good as the recognised spacing.

## Install log (every failure, verbatim)

Environment: Windows 11, CPython 3.11, uv 0.11.14, project `.venv` only (no global/user-site installs).

1. `uv add --group ocr paddleocr paddlepaddle` failed to build a transitive dependency:
   ```
   × Failed to build `stringzilla==5.1.2`
   ├─▶ The build backend returned an error
   ╰─▶ Call to `build_backend.build_wheel` failed (exit code: 1)
         error: Microsoft Visual C++ 14.0 or greater is
         required. Get it with "Microsoft C++ Build Tools":
   ```
   Fix: pin `stringzilla<5.1` (5.0.7 ships a Windows wheel).
2. Import of the resolved pair (paddleocr 2.10.0 + paddlepaddle 3.3.1) failed:
   `ModuleNotFoundError: No module named 'setuptools'` (paddle imports it). Fix: `setuptools` in the group.
3. `paddleocr>=3.0` conflicts with the project: `paddleocr==3.7.0 depends on pyyaml==6.0.2` against
   the project's `pyyaml>=6.0.3` ("your project and shipdoc-extract:ocr are incompatible").
   Fix: the project floor is relaxed to `pyyaml>=6.0.2` (lock resolves 6.0.2).
4. paddleocr 3.4.1 + paddlepaddle 3.3.1 (latest) installed but **inference fails on CPU**:
   ```
   NotImplementedError: (Unimplemented) ConvertPirAttribute2RuntimeAttribute not support
   [pir::ArrayAttribute<pir::DoubleAttribute>]  (at ..\paddle\fluid\framework\new_executor\instruction\onednn\onednn_instruction.cc:118)
   ```
   `enable_mkldnn=False` avoids the error but is ~10x slower (mobile models, 18-line page: 11-15 s on
   3.3.1 with oneDNN off vs 0.9-1.5 s on 3.0.0 with oneDNN on, same machine; 3.0.0 with oneDNN off took 9.7 s). `FLAGS_enable_pir_api=0` did not change the error.
5. Working pair: **paddlepaddle==3.0.0, paddleocr==3.0.3, paddlex==3.0.3** with oneDNN on. paddlex 3.0.3 then
   failed with `ModuleNotFoundError: No module named 'langchain.docstore'` against langchain 1.x;
   fix: pin `langchain==0.3.30` (+ community 0.3.31, core 0.3.86, openai 0.3.35, text-splitters 0.3.11).
   `uv.lock` is the authority for all pins. docTR fallback was not needed.
6. First server-model `predict` took 89 s for one page in the broken oneDNN-off configuration (3.3.1), which
   is what triggered the version-pair search.

## Model choice: server over mobile

10 train pages (first 5 png and first 5 jpg), 8 CPU threads, oneDNN on, machine shared with
other jobs (so absolute seconds are noisy; the ratio is what matters):

| model | mean s/page | png | jpg | lines | mean conf |
|---|---|---|---|---|---|
| PP-OCRv5 mobile det+rec | 2.59 | 2.66 | 2.51 | 574 | 0.943 |
| PP-OCRv5 server det+rec | 8.01 | 7.20 | 8.83 | 578 | 0.971 |

Text agreement between the two is 98.8-100% (rapidfuzz ratio) on clean png pages and 95-98% on
scans, but `train_0001_p2.jpg` (blurred scan, read by me against the model output) is 83.8%:
mobile garbled a product description and several amounts (e.g. a decimal figure read with a
wrong digit run), read a part number like `XX/1234/ABCD` with O/0 and letter/digit confusion; it also
dropped a thousands separator and swapped digits within a quantity. Server got these right. Both models
misread `00348` as `OO348`/`sG` on `train_0003_p1.png`, so neither is clean on O/0.
Decision (quality first): **server**. Projected cost 951 x ~8-10 s = 2.1-2.7 h CPU, borderline against the
2.5 h guideline; it was accepted because accuracy on the blurred scans is the point of the cache. The
measured total is below.

## Full run

Command: `python -m shipdoc ocr --splits train dev test --cpu-threads 8` (server variant is the default),
numbers from `uv run python scripts/ocr_stats.py` (reads the per-page JSON and the manifest):

```
engine paddleocr paddleocr 3.0.3 paddlepaddle 3.0.0 paddlex 3.0.3
models detection PP-OCRv5_server_det, recognition PP-OCRv5_server_rec, PP-LCNet_x1_0_doc_ori, PP-LCNet_x1_0_textline_ori
pages expected 951 cached 951 {'dev': 135, 'test': 280, 'train': 536}
failures in manifest (last run) 0 []
s/page mean 10.16 median 9.19 p95 19.36 sum_h 2.68
rotation_deg counts {'0': 949, '90': 2} total lines 70252
```

* Pages: 951 / 951 cached, 0 failures at the end.
* Per-page seconds are stored in each page JSON, so they survived the interruptions below. They are
  **wall seconds under contention**: for roughly the first hour two OCR processes ran at once and two unrelated
  heavy jobs from other projects were running, so s/page overstates a quiet-machine cost.
* Wall clock first page to last page: 11:10:43 to 13:36:37 on 2026-10-01 = 2.43 h, including the
  interruption and repairs below (`os.path.getmtime` over the 951 files).

## Interruption (disk full on C:)

The run was killed partway (711/951 cached, last write 12:26). Cause: **the C: drive was at 100% (3-250 MB free)** from
other activity on the machine (not the cache: `du -sh cache` showed 17 MB when the disk filled).
Evidence: the first dev/test process died with `OSError: [Errno 28] No space left on device` while writing
`test_0088_p2.json.tmp`; the concurrent train process then logged `RuntimeError('Unknown exception')` for
every page from `train_0324_p1.jpg` onward (an engine left wedged); later restarts died with exit code 139 (segfault) after
2-4 pages each while free space hovered at ~250 MB. Free RAM was 13.7 GB of 31.2 GB at the time, so it was
not memory. Once the disk freed up (22 GB free) attempt 102 finished the remaining 216 pages cleanly.
Mitigations kept in code: JSON writes are atomic (temp file + rename), the run stops after 5 consecutive failures
instead of recording hundreds of spurious ones (failed pages stay uncached and are retried on rerun), and the
manifest is rebuilt from the page files, so earlier timings are not lost. Failures from the crashed runs live
only in the run logs (`cache/logs/`, ignored); the manifest's `failures` list covers the last invocation.

## Quality findings and the fixes applied

1. **3 pages came back with zero lines** (`train_0005_p1`, `train_0082_p1`, `train_0345_p1`, all blurred
   waybill scans); also with orientation off, lower thresholds, or limit 1600. A max-side limit of 1280 recovers
   them (16-18 lines each), as does the mobile detector. `run()` retries an empty page that way and records
   `fallbacks: ["det_limit_max1280"]`.
2. **The document-orientation classifier flags 25 of 951 pages as rotated**, mostly sparse continuation pages
   (p2/p3), including digital pngs that cannot be rotated. I viewed 7 of them in train (`train_0081_p2`,
   `0101_p2`, `0125_p2`, `0173_p3`, `0278_p1`, `0398_p3`, `0015_p2`) and all were upright;
   rotating them reordered lines and lost some (`train_0081_p2`: the rotated pass dropped the row numbers 16/17).
   Re-running without the classifier gives 86-100% of the rotated pass's line count on 23 pages, but only
   3% and 21% on `test_0002_p2` and `test_0197_p3` (1 and 7 lines vs 38 and 34), so those two are really
   sideways and the rotation is kept. Rule: keep the unrotated result if it finds at least 50% as many lines
   (`ROTATION_KEEP_RATIO`); the gap between the groups is 21% vs 86%. Outcome in the cache:
   23 pages `orientation_override`, 2 pages `orientation_kept` (rotation 90, boxes mapped back to the
   original frame), 3 `det_limit_max1280`. I did not view the two test pages (spec: test images are only
   touched by inference and clustering); the line-count ratio is the only evidence there. The inverse of the
   rotation was checked against `cv2.warpAffine` on synthetic images (within 1 px) and is unit tested.
3. Some items are empty strings (junk detections, e.g. one in the corner of `train_0081_p2`). They are kept as
   recorded; downstream code should filter on empty text.
4. Per-item order is the detector's order, not reading order; sort by box position downstream.
5. oneDNN is enabled; results are deterministic for a fixed page in my reruns of the same 28 pages (line counts
   identical across the two repair runs), but I did not diff the text byte-for-byte.
