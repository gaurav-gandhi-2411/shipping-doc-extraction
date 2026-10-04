"""shipdoc.resmatch.training_resolution_matches: equal passes, any difference or gap fails."""

from __future__ import annotations

import pytest

from shipdoc.resmatch import training_resolution_matches as matches


def manifest(value: object) -> dict:
    return {"stage": "fold0", "inference_keys": {"max_pixels": value, "adapter": "qwen35"}}


def test_equal_resolution_passes() -> None:
    ok, why = matches(manifest(2_196_480), {"max_pixels": 2_196_480})
    assert ok and "2196480" in why
    assert matches(manifest(1_310_720), {"max_pixels": 1_310_720})[0]


@pytest.mark.parametrize(("trained", "run"), [(1_310_720, 2_196_480), (2_196_480, 1_310_720)])
def test_any_difference_fails_and_names_both_values(trained: int, run: int) -> None:
    ok, why = matches(manifest(trained), {"max_pixels": run})
    assert not ok and str(trained) in why and str(run) in why and "mismatch" in why


@pytest.mark.parametrize(
    "bad_manifest",
    [
        None,
        {},
        {"inference_keys": None},
        {"inference_keys": {}},
        {"inference_keys": {"model_repo": "x"}},  # an adapter from before the key was recorded
        manifest(None),
        manifest("2196480"),  # a string is not a pixel count
        manifest(True),  # bool is an int subclass: must not pass as 1
        manifest(0),
        manifest(-5),
        manifest(2196480.5),
    ],
)
def test_missing_or_malformed_training_resolution_fails_closed(bad_manifest: object) -> None:
    ok, why = matches(bad_manifest, {"max_pixels": 2_196_480})  # type: ignore[arg-type]
    assert not ok and "inference_keys.max_pixels" in why


@pytest.mark.parametrize("bad_cfg", [None, {}, {"max_pixels": None}, {"max_pixels": "x"}])
def test_missing_inference_resolution_fails_closed(bad_cfg: object) -> None:
    ok, why = matches(manifest(2_196_480), bad_cfg)  # type: ignore[arg-type]
    assert not ok and "inference config" in why


def test_an_integral_float_is_accepted_as_the_same_pixel_count() -> None:
    assert matches(manifest(2_196_480.0), {"max_pixels": 2_196_480})[0]  # json round trips
