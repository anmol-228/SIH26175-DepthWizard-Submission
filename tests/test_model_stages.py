"""Tests for the model stages - DW-S1-07 (#24), Anmol's lane (PR A).

Fast, offline, deterministic. No checkpoint, no network, no model download, no Vaihingen:
`predict_depth()` is exercised with a tiny FakeModel on CPU, `blend_tiles()` is pure NumPy.
Tests that reach the torch code path skip cleanly when torch is absent, mirroring
tests/test_run_pipeline.py.

The surface these stages produce is relative / unitless / uncalibrated - no metric or RMSE
assertion appears here, by design.

Run either way:
    python tests/test_model_stages.py     # plain, no pytest needed
    pytest tests/test_model_stages.py     # if pytest is installed
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model_stages import blend_tiles, predict_depth  # noqa: E402
from tiling import TilePlan, plan_windows  # noqa: E402

# Small, edge-and-overlap-exercising plan: ys -> [0, 36], xs -> [0, 48, 56] => 6 windows.
_H, _W, _WIN, _STRIDE = 100, 120, 64, 48


def _plan() -> TilePlan:
    return plan_windows(_H, _W, window=_WIN, stride=_STRIDE)


def _img(h: int = _H, w: int = _W) -> np.ndarray:
    return np.random.default_rng(0).integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def _full_depths(plan: TilePlan, fill: float = 1.0) -> dict:
    return {o: np.full((plan.window, plan.window), fill, np.float32) for o in plan.origins}


def _has_torch() -> bool:
    return importlib.util.find_spec("torch") is not None


class _FakeDepthOutput:
    def __init__(self, predicted_depth):
        self.predicted_depth = predicted_depth


class _FakeModel:
    """Stand-in for HF DepthAnythingForDepthEstimation: CPU, no weights, deterministic.

    Returns `predicted_depth` at a small native resolution (like the real model, which is
    why da2_utils.resize_pred_to exists), as a `(B, h, w)` float32 tensor.
    """

    def __init__(self, *, native_hw=(37, 37), fill=None, inject_nonfinite=False,
                 bad_output=None, batch_out=None):
        self.native_hw = native_hw
        self.fill = fill
        self.inject_nonfinite = inject_nonfinite
        self.bad_output = bad_output          # None | "no_attr" | "not_tensor"
        self.batch_out = batch_out            # None -> match the input batch
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self

    def __call__(self, pixel_values):
        import torch
        if self.bad_output == "no_attr":
            return object()
        h, w = self.native_hw
        b = self.batch_out if self.batch_out is not None else int(pixel_values.shape[0])
        if self.fill is not None:
            depth = torch.full((b, h, w), float(self.fill), dtype=torch.float32)
        else:
            ramp = torch.linspace(0.0, 1.0, h * w, dtype=torch.float32).reshape(1, h, w)
            depth = ramp.repeat(b, 1, 1)
        if self.inject_nonfinite:
            depth[0, 0, 0] = float("nan")
        if self.bad_output == "not_tensor":
            return _FakeDepthOutput(depth.numpy())
        return _FakeDepthOutput(depth)


def _expect_value_error(fn, *, contains=()):
    try:
        fn()
    except ValueError as exc:
        msg = str(exc)
        for token in contains:
            assert token in msg, f"error message {msg!r} missing {token!r}"
        return
    raise AssertionError("expected ValueError, none raised")


# --- predict_depth ---------------------------------------------------------------------

def test_predict_depth_contract_shapes_and_dtype():
    if not _has_torch():
        return
    plan = _plan()
    depths, _ = predict_depth(_img(), plan, model=_FakeModel(fill=0.25))
    assert set(depths) == set(plan.origins), "one prediction per planned window, keyed by origin"
    for (y, x), arr in depths.items():
        assert arr.shape == (plan.window, plan.window), f"{(y, x)} -> {arr.shape}"
        assert arr.dtype == np.float32
        assert np.isfinite(arr).all()


def test_predict_depth_provenance_fields():
    if not _has_torch():
        return
    plan = _plan()
    _, prov = predict_depth(_img(), plan, model=_FakeModel(fill=0.5))
    assert len(prov) == plan.n_windows
    for i, e in enumerate(prov):
        assert set(e) == {"tile_id", "bbox", "shape", "dtype", "elapsed_s",
                          "min", "max", "mean"}, f"unexpected provenance keys: {sorted(e)}"
        assert e["tile_id"] == i
        y, x = plan.origins[i]
        assert e["bbox"] == [y, x, y + plan.window, x + plan.window]
        assert e["shape"] == [plan.window, plan.window]
        assert e["dtype"] == "float32"
        assert isinstance(e["elapsed_s"], float) and e["elapsed_s"] >= 0.0
        assert e["min"] <= e["mean"] <= e["max"]


def test_predict_depth_rejects_nonfinite_model_output():
    if not _has_torch():
        return
    plan = _plan()
    _expect_value_error(
        lambda: predict_depth(_img(), plan, model=_FakeModel(inject_nonfinite=True)),
        contains=("non-finite", "origin (0, 0)"),
    )


def test_predict_depth_rejects_malformed_model_output():
    if not _has_torch():
        return
    plan = _plan()
    _expect_value_error(
        lambda: predict_depth(_img(), plan, model=_FakeModel(bad_output="no_attr")),
        contains=("predicted_depth",),
    )
    _expect_value_error(
        lambda: predict_depth(_img(), plan, model=_FakeModel(bad_output="not_tensor")),
        contains=("not a tensor",),
    )
    _expect_value_error(
        lambda: predict_depth(_img(), plan, model=_FakeModel(batch_out=2)),
        contains=("batch",),
    )


def test_predict_depth_provenance_is_json_safe():
    if not _has_torch():
        return
    plan = _plan()
    _, prov = predict_depth(_img(), plan, model=_FakeModel(fill=0.7))
    # allow_nan=False is the runner's contract: a non-finite slipped into min/max/mean
    # would blow up run_manifest.json rather than fail here.
    json.dumps(prov, allow_nan=False)


def test_predict_depth_validates_rgb():
    plan = _plan()
    fake = _FakeModel(fill=0.1)
    _expect_value_error(lambda: predict_depth([[1, 2], [3, 4]], plan, model=fake),
                        contains=("ndarray",))
    _expect_value_error(
        lambda: predict_depth(np.zeros((_H, _W), np.uint8), plan, model=fake),
        contains=("3-D",))
    _expect_value_error(
        lambda: predict_depth(np.zeros((_H, _W, 4), np.uint8), plan, model=fake),
        contains=("3 channels",))
    _expect_value_error(
        lambda: predict_depth(np.zeros((_H, _W, 3), np.float32), plan, model=fake),
        contains=("uint8",))


def test_predict_depth_requires_plan_matches_image():
    plan = _plan()                       # plan is for 100x120
    _expect_value_error(
        lambda: predict_depth(_img(90, 120), plan, model=_FakeModel(fill=0.1)),
        contains=("disagree",))


def test_predict_depth_rejects_duplicate_plan_origins():
    bad = TilePlan(height=_H, width=_W, window=_WIN, stride=_STRIDE,
                   origins=((0, 0), (0, 0), (36, 0)))
    _expect_value_error(
        lambda: predict_depth(_img(), bad, model=_FakeModel(fill=0.1)),
        contains=("duplicate",))


def test_predict_depth_rejects_silent_base_model_fallback():
    """model=None without an explicit opt-in must raise - no torch needed to reject it."""
    plan = _plan()
    rgb = _img()
    _expect_value_error(lambda: predict_depth(rgb, plan, model=None),
                        contains=("base", "un-fine-tuned"))
    _expect_value_error(lambda: predict_depth(rgb, plan, model=None, allow_base_model=False),
                        contains=("base",))


def test_predict_depth_explicit_base_model_is_deliberate_and_flagged():
    """allow_base_model=True is the only way in, and it must be recorded in provenance.

    da2_utils.load_model is stubbed, so nothing is downloaded and no network is touched.
    """
    if not _has_torch():
        return
    from unittest.mock import patch

    import da2_utils

    plan = _plan()
    fake = _FakeModel(fill=0.5)
    with patch.object(da2_utils, "load_model", lambda *_a, **_kw: fake):
        depths, prov = predict_depth(_img(), plan, model=None, allow_base_model=True)

    assert fake.eval_called, "predict_depth must call model.eval()"
    assert set(depths) == set(plan.origins)
    assert prov, "no provenance produced"
    for e in prov:
        assert "WARNING" in e, "base-model provenance must carry a WARNING"
        low = e["WARNING"].lower()
        assert "base" in low and "fine-tuned" in low


def test_predict_then_blend_roundtrip_offline():
    if not _has_torch():
        return
    plan = _plan()
    depths, prov = predict_depth(_img(), plan, model=_FakeModel(fill=2.0))
    surface = blend_tiles(depths, plan)
    assert surface.shape == (_H, _W)
    assert surface.dtype == np.float32
    assert np.isfinite(surface).all()
    # constant model output -> constant blended surface (K1 normalize-by-weight invariant)
    assert np.allclose(surface, 2.0, atol=1e-3)
    assert len(prov) == plan.n_windows


# --- blend_tiles ----------------------------------------------------------------------

def test_blend_output_is_float32_full_aoi():
    plan = _plan()
    surface = blend_tiles(_full_depths(plan, fill=3.5), plan)
    assert surface.shape == (_H, _W)
    assert surface.dtype == np.float32
    assert np.isfinite(surface).all()
    assert np.allclose(surface, 3.5, atol=1e-5), "constant in must blend to that constant"


def test_blend_rejects_nonfinite_predictions():
    plan = _plan()
    d = _full_depths(plan)
    victim = plan.origins[1]
    d[victim] = d[victim].copy()
    d[victim][3, 3] = np.inf
    _expect_value_error(lambda: blend_tiles(d, plan),
                        contains=("non-finite", str(victim)))


def test_blend_rejects_unknown_tile_origin():
    plan = _plan()
    d = _full_depths(plan)
    d[(9999, 9999)] = np.zeros((plan.window, plan.window), np.float32)
    _expect_value_error(lambda: blend_tiles(d, plan),
                        contains=("never defined", "9999"))


def test_blend_rejects_wrong_prediction_shape():
    plan = _plan()
    # a broadcastable wrong shape: (1, window) would otherwise broadcast across rows
    d = _full_depths(plan)
    d[plan.origins[0]] = np.ones((1, plan.window), np.float32)
    _expect_value_error(lambda: blend_tiles(d, plan), contains=("shape", "(1, 64)"))
    # and a scalar, which broadcasts everywhere
    d2 = _full_depths(plan)
    d2[plan.origins[0]] = np.float32(1.0)
    _expect_value_error(lambda: blend_tiles(d2, plan), contains=("shape",))


def test_blend_missing_tile_reports_missing_prediction():
    plan = _plan()
    d = _full_depths(plan)
    dropped = plan.origins[-1]
    del d[dropped]
    try:
        blend_tiles(d, plan)
    except ValueError as exc:
        low = str(exc).lower()
        assert "missing model prediction" in low, low
        assert "not a tile-plan gap" in low, "must not blame the tile planner"
        assert str(dropped) in str(exc)
    else:
        raise AssertionError("blended an incomplete tile set without complaining")


def test_blend_preserves_k1_weight_normalization():
    """K1's invariant: overlapping windows are averaged weighted by the ramp, so a shared
    constant survives the overlap unchanged. Distinct from test_run_pipeline's coverage
    test - this one puts different values in overlapping windows and checks the seam."""
    plan = _plan()
    d = _full_depths(plan, fill=5.0)
    surface = blend_tiles(d, plan)
    # every pixel is a convex combination of 5.0's -> exactly 5.0, no seam drift
    assert float(surface.min()) == 5.0 and float(surface.max()) == 5.0


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
