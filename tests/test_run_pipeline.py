"""Tests for the run_pipeline tiling geometry and validation - DW-S1-07 (#24).

No checkpoint, no network, no Vaihingen data: the model stages are exercised with a stub
predictor, so these stay unit tests. The real end-to-end runs are recorded in the PR.

Run either way:
    python tests/test_run_pipeline.py
    pytest tests/test_run_pipeline.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from model_stages import blend_tiles  # noqa: E402
from tiling import DEFAULT_STRIDE, DEFAULT_WINDOW, plan_windows, window_ramp  # noqa: E402


def test_plan_covers_every_pixel_including_the_remainder():
    """A stride that does not divide evenly leaves an uncovered strip unless the edge is
    anchored explicitly. 1100x900 is deliberately not a multiple of 384."""
    plan = plan_windows(900, 1100)
    covered = np.zeros((900, 1100), dtype=bool)
    for (y, x) in plan.origins:
        covered[y:y + plan.window, x:x + plan.window] = True
    assert covered.all(), f"{int((~covered).sum())} pixels outside every window"


def test_plan_rejects_a_surface_smaller_than_one_window():
    for h, w in ((100, 2000), (2000, 100)):
        try:
            plan_windows(h, w)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted {h}x{w}, smaller than a {DEFAULT_WINDOW}px window")


def test_border_windows_do_not_taper_against_the_surface_edge():
    """The bug this caught: K1 tapers all four sides, so the outer border of the whole
    surface gets zero weight - there is no neighbour there to compensate."""
    plan = plan_windows(900, 1100)
    top_left = window_ramp(plan, 0, 0)

    # The outermost corner of the whole surface must carry real weight. Note the top ROW
    # still tapers at its right end, and correctly so - that edge is interior and has a
    # neighbour. Only the sides lying on the surface boundary are exempt.
    assert top_left[0, 0] > 0, "surface corner tapered to zero - nothing else covers it"
    edge = plan.window - plan.stride
    assert top_left[0, :plan.stride].min() > 0, "top boundary row tapered where it should not"
    assert top_left[:plan.stride, 0].min() > 0, "left boundary column tapered where it should not"
    assert top_left[0, -1] == 0, "interior right edge must still taper"

    interior = window_ramp(plan, plan.stride, plan.stride)
    assert interior[0, :].min() == 0, "an interior window must taper on every side"
    assert interior[-1, :].min() == 0 and interior[:, 0].min() == 0


def test_blend_produces_full_finite_coverage():
    """Regression for the 3996-zero-weight-pixel bug found on the first real run."""
    plan = plan_windows(900, 1100)
    depths = {(y, x): np.full((plan.window, plan.window), 1.0, np.float32)
              for (y, x) in plan.origins}
    surface = blend_tiles(depths, plan)
    assert surface.shape == (900, 1100)
    assert np.isfinite(surface).all()
    # constant input must blend to that constant everywhere, including the border
    assert np.allclose(surface, 1.0, atol=1e-5), "blend distorted a constant surface"


def test_blend_raises_rather_than_silently_zero_filling():
    """A gap must fail loudly - zeros would render as flat terrain and look like a result."""
    plan = plan_windows(900, 1100)
    depths = {plan.origins[0]: np.ones((plan.window, plan.window), np.float32)}
    try:
        blend_tiles(depths, plan)
    except ValueError:
        pass
    else:
        raise AssertionError("blended an incomplete tile set without complaining")


def test_validate_rejects_bad_input_cleanly():
    from run_pipeline import PipelineError, validate_input
    with tempfile.TemporaryDirectory() as d:
        checks = {
            "missing file": Path(d) / "nope.png",
            "too small": Path(d) / "tiny.png",
            "not an image": Path(d) / "bad.png",
        }
        Image.new("RGB", (64, 64)).save(checks["too small"])
        checks["not an image"].write_text("this is not an image")
        for label, path in checks.items():
            try:
                validate_input(path)
            except PipelineError:
                pass
            else:
                raise AssertionError(f"{label!r} was accepted")


def test_validate_accepts_a_real_image_and_returns_rgb():
    from run_pipeline import validate_input
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ok.png"
        Image.new("RGB", (600, 700)).save(p)
        rgb = validate_input(p)
        assert rgb.shape == (700, 600, 3) and rgb.dtype == np.uint8


def test_missing_checkpoint_never_silently_falls_back_to_base_model():
    """Regression for the defect Anmol caught on PR #28.

    da2_utils.load_model() returns the *base* HF model; every other script applies a
    checkpoint's state_dict afterwards. Skipping that produces un-finetuned predictions
    that look completely normal - no error, no warning, just quietly wrong numbers. A
    missing checkpoint must therefore fail loudly rather than degrade.
    """
    from run_pipeline import PipelineError, load_predictor
    try:
        load_predictor(Path("/nonexistent/checkpoint.pt"), allow_base_model=False)
    except PipelineError as exc:
        assert "base model" in str(exc).lower(), "error must name what it is refusing to do"
    except ImportError:
        return  # torch absent in this environment; the guard is still asserted above
    else:
        raise AssertionError("a missing checkpoint was silently accepted")


def test_default_checkpoint_is_the_v3_full_encoder_ema_one():
    """v3 (full-encoder adaptation, EMA weights) is the default after the pre-registered train-only gates
    (see 10_DOCUMENTATION/evidence/DW_MODEL_V3_FULL_ENCODER_EMA). iter2 remains the documented rollback."""
    from run_pipeline import DEFAULT_CHECKPOINT, EXPECTED_CHECKPOINT_SHA256
    assert DEFAULT_CHECKPOINT == "da2_small_finetuned_v3_full_encoder_ema.pt"
    assert EXPECTED_CHECKPOINT_SHA256 == "a1e3fca34874856196e53fe9be2be1ee10d1b038c741157dfd6472f74e299ea7"


def test_relative_and_metric_outputs_cannot_collide():
    """Filenames must be unambiguous - the conflation #5 and Part IV forbid."""
    from relative_dsm import PRODUCT, SCHEMA_VERSION
    assert PRODUCT == "relative_dsm"
    assert SCHEMA_VERSION.endswith("v0-relative")
    # the runner names relative artifacts relative_dsm_<scene>.npy; a metric product is a
    # GeoTIFF written by geoio.save_geotiff, so the two can never share a path
    assert not PRODUCT.endswith(".tif")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as exc:              # noqa: BLE001
            failed += 1
            print(f"  FAIL  {t.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
