# meta/

Frozen, label-derived artefacts read by the pipeline. No extracted value is stored here.

## Which calibrator serves which submission

| submission | run (config hash) | file | `run_config_hash` recorded |
|---|---|---|---|
| v1: 1260-token ZS + rules (`max_pixels` 1310720) | `01d87878679ca0fc` | `calibrator_zs.json` | no (frozen before the guard existed) |
| v1.5: native ZS + rules (`max_pixels` 2196480) | `e4b84ec2809625d5` | `calibrator_zs_native.json` | yes, `e4b84ec2809625d5` |

`shipdoc.flags.run_stage` refuses a submission whose manifest config hash differs from the calibrator's
`run_config_hash`; the native notebooks name `calibrator_zs_native.json` explicitly and refuse the 1260
file (it records no hash). Never swap one for the other: the probabilities and thresholds are
resolution-specific. Comparison of the two: `reports/calibration_v2_native.md`; how the native file was
made: `scripts/calibrate_v2.py` then `scripts/freeze_calibrator.py --arm zs` on the native 02 run.

Other files: `slot_shapes.json` (frozen R3 format shapes, `scripts/freeze_slot_shapes.py`),
`supplier_groups.json`, `train.json` / `dev.json` (per-document tags), `field_provenance.json`
(fuzzy-match threshold curve), `iata_airports.txt` (see `reports/ablations.md` for its source).
