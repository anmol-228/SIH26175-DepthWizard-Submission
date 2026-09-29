"""Model stages for run_pipeline.py - DW-S1-07 (#24), Anmol's lane (PR A).

**Ownership: Anmol.** `predict_depth()` and `blend_tiles()` only. This file is deliberately
kept separate from `src/tiling.py` (Nirmalya's tiling geometry) so the two PRs never touch
the same file. The tiling primitives - `TilePlan`, `iter_tiles()`, `window_ramp()` - are
imported from there and never reimplemented here.

What these stages produce is a RELATIVE surface: unitless, uncalibrated, no datum, no map
location, no metres. No RMSE may be attached to it. Calibration and the metric branch are
the runner's job, downstream of `blend_tiles()`.

Contract the runner depends on (confirmed on #24):

    predict_depth(rgb, plan, *, model=None, device=None, allow_base_model=False)
        -> (depths, provenance)
        depths     : {(y, x): (window, window) float32}   relative, uncalibrated
        provenance : list of per-tile dicts, recorded verbatim into run_manifest.json

    blend_tiles(depths, plan) -> (height, width) float32   relative, uncalibrated

The blend arithmetic is K1's, verbatim (`PHASE2_RED_FLAGS/scripts/K1_tiling_stitch.py`):
accumulate `pred * ramp`, accumulate `ramp`, divide. The feather ramp itself is
`tiling.window_ramp()` - boundary-aware, so the outer border of the surface is not
tapered to zero weight (the 3996-zero-weight-pixel bug Nirmalya caught; there is a
regression test on it in tests/test_run_pipeline.py).

model=None policy (confirmed by Nirmalya on #24): a silent fall back to the base
DA2-Small is forbidden. Loading it silently would produce normal-looking but
un-fine-tuned predictions - no error, no warning, just quietly wrong numbers, the same
defect class as the runner bug one layer down. `predict_depth()` therefore raises unless
`allow_base_model=True` is passed explicitly for a deliberate base-model comparison - and
when it is, every provenance entry carries a WARNING so run_manifest.json records that the
un-fine-tuned weights ran. This module never asserts that an injected model is fine-tuned:
the runner owns checkpoint provenance and already records the checkpoint SHA and model
identity. Standard exceptions only here (`ValueError`) - no import from run_pipeline.py.
"""
from __future__ import annotations

import time

import numpy as np

from tiling import TilePlan, iter_tiles, window_ramp

_MODEL_NONE_MESSAGE = (
    "predict_depth() requires a loaded model. Passing model=None would silently load and "
    "run the un-fine-tuned base DA2-Small and produce normal-looking but un-fine-tuned "
    "predictions - no error, no warning, just quietly wrong numbers. The runner injects "
    "the verified fine-tuned checkpoint and records its SHA in run_manifest.json. Pass "
    "allow_base_model=True only for a deliberate base-model comparison."
)

_BASE_MODEL_WARNING = (
    "prediction used the UN-FINE-TUNED base DA2-Small (allow_base_model=True); not "
    "comparable to any DepthWizard fine-tuned baseline"
)


def predict_depth(rgb, plan: TilePlan, *, model=None, device: str | None = None,
                  allow_base_model: bool = False):
    """Relative depth for every window in `plan`.

    `model` and `device` are injected by the runner: it has to hash the checkpoint for
    run_manifest.json anyway, and loading once outside the loop avoids re-reading weights
    per window. `model=None` is rejected (see the module docstring) unless
    `allow_base_model=True`, in which case the base DA2-Small is loaded deliberately and
    every provenance entry is flagged.

    Returns `(depths, provenance)`. The surface is relative / unitless / uncalibrated.
    Torch/transformers are imported lazily so the geometry in tiling.py stays importable
    without them.
    """
    # --- model=None policy - reject a misuse before importing torch ------------
    used_base_model = model is None and allow_base_model
    if model is None and not allow_base_model:
        raise ValueError(_MODEL_NONE_MESSAGE)

    # --- validate the RGB input ----------------------------------------------
    if not isinstance(rgb, np.ndarray):
        raise ValueError(f"rgb must be a numpy ndarray, got {type(rgb).__name__}")
    if rgb.ndim != 3:
        raise ValueError(f"rgb must be 3-D (H, W, 3); got {rgb.ndim}-D, shape {rgb.shape}")
    if rgb.shape[2] != 3:
        raise ValueError(f"rgb must have exactly 3 channels; got {rgb.shape[2]}")
    if rgb.dtype != np.uint8:
        raise ValueError(f"rgb must be uint8; got {rgb.dtype}")

    # --- validate plan / image agreement -----------------------------------
    if (int(plan.height), int(plan.width)) != (rgb.shape[0], rgb.shape[1]):
        raise ValueError(
            f"TilePlan covers {plan.height}x{plan.width} but the image is "
            f"{rgb.shape[0]}x{rgb.shape[1]}; plan and image disagree")

    # --- reject duplicate origins (would overwrite depths, duplicate provenance) ---
    origins = list(plan.origins)
    dups = sorted({o for o in origins if origins.count(o) > 1})
    if dups:
        raise ValueError(
            f"TilePlan has duplicate window origins {dups}; each origin must be unique or "
            "predictions would silently overwrite one another")

    import torch                                                    # noqa: PLC0415
    from da2_utils import preprocess_rgb, resize_pred_to            # noqa: PLC0415

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if model is None:
        from da2_utils import load_model                           # noqa: PLC0415
        model = load_model(device)
    assert model is not None              # narrowing for type-checkers; gate above ensures it

    model.eval()
    win = int(plan.window)
    depths: dict[tuple[int, int], np.ndarray] = {}
    provenance: list[dict] = []

    with torch.no_grad():
        for tile_id, ((y, x), patch) in enumerate(iter_tiles(rgb, plan)):
            t0 = time.perf_counter()

            if patch.shape != (win, win, 3):
                raise ValueError(
                    f"tile {tile_id} at origin {(y, x)}: iter_tiles produced a "
                    f"{patch.shape} patch, expected ({win}, {win}, 3)")

            xin = preprocess_rgb(patch).unsqueeze(0).to(device)
            out = model(pixel_values=xin)

            pred = getattr(out, "predicted_depth", None)
            if pred is None:
                raise ValueError(
                    f"tile {tile_id} at origin {(y, x)}: model output "
                    f"({type(out).__name__}) has no `predicted_depth` - not a Depth "
                    "Anything V2 depth-estimation output")
            if not isinstance(pred, torch.Tensor):
                raise ValueError(
                    f"tile {tile_id} at origin {(y, x)}: predicted_depth is "
                    f"{type(pred).__name__}, not a tensor")
            if pred.dim() not in (3, 4) or pred.shape[0] != 1:
                raise ValueError(
                    f"tile {tile_id} at origin {(y, x)}: predicted_depth shape "
                    f"{tuple(pred.shape)} is not a single-item (B=1, H, W) / (B=1, 1, H, W) "
                    "batch")

            arr = resize_pred_to(pred, (win, win)).squeeze(0).detach().cpu().numpy()
            arr = np.ascontiguousarray(arr, dtype=np.float32)
            if arr.shape != (win, win):
                raise ValueError(
                    f"tile {tile_id} at origin {(y, x)}: prediction resized to {arr.shape}, "
                    f"expected ({win}, {win})")
            if not np.isfinite(arr).all():
                n_bad = int((~np.isfinite(arr)).sum())
                raise ValueError(
                    f"tile {tile_id} at origin {(y, x)}: prediction has {n_bad} non-finite "
                    "value(s) (NaN/Inf) - refusing to pass a corrupt tile to the blend")

            depths[(int(y), int(x))] = arr
            entry = {
                "tile_id": tile_id,
                "bbox": [int(y), int(x), int(y) + win, int(x) + win],
                "shape": [int(arr.shape[0]), int(arr.shape[1])],
                "dtype": str(arr.dtype),
                "elapsed_s": round(time.perf_counter() - t0, 4),
                "min": float(arr.min()), "max": float(arr.max()), "mean": float(arr.mean()),
            }
            if used_base_model:
                entry["WARNING"] = _BASE_MODEL_WARNING
            provenance.append(entry)

    return depths, provenance


def blend_tiles(depths: dict, plan: TilePlan) -> np.ndarray:
    """Feather-blend the per-window relative depths into one surface - K1's arithmetic.

    `accumulate(pred * ramp)` / `accumulate(ramp)`, with `ramp` from
    `tiling.window_ramp()` (boundary-aware). The output is the raw blended RELATIVE
    surface: unitless, uncalibrated, and NOT normalized here - normalization and
    calibration are the runner's job. A pixel that no window covered is a hard failure,
    never a zero (zeros would render as flat terrain and read as a result).

    Every prediction is checked against the plan before any accumulation: the origin set
    must match `plan.origins` exactly, each array must be exactly `(window, window)` (so a
    scalar / `(1, window)` / `(window, 1)` cannot be broadcast silently into place), and
    every value must be finite.
    """
    win = int(plan.window)
    expected = set(plan.origins)
    if len(expected) != len(plan.origins):
        raise ValueError(
            "TilePlan.origins contains duplicate origins; blending would double-count them")

    got = set(depths)
    missing = expected - got
    if missing:
        raise ValueError(
            f"{len(missing)} planned window(s) have no model prediction: "
            f"{sorted(missing)}. predict_depth() did not return a tile for these origins - "
            "a missing model prediction, not a tile-plan gap")
    extra = got - expected
    if extra:
        raise ValueError(
            f"{len(extra)} prediction(s) at origins the TilePlan never defined: "
            f"{sorted(extra)}. Predictions must match plan.origins exactly")

    accum = np.zeros((plan.height, plan.width), dtype=np.float64)
    weight = np.zeros((plan.height, plan.width), dtype=np.float64)

    for (y, x) in plan.origins:
        pred = np.asarray(depths[(y, x)])
        if pred.shape != (win, win):
            raise ValueError(
                f"prediction at origin {(y, x)} has shape {pred.shape}, not ({win}, {win}); "
                "refusing to let NumPy broadcast a scalar / (1, win) / (win, 1) tile into "
                "place")
        if not np.isfinite(pred).all():
            n_bad = int((~np.isfinite(pred)).sum())
            raise ValueError(
                f"prediction at origin {(y, x)} has {n_bad} non-finite value(s); "
                "cannot blend")

        # per-window ramp: sides on the surface boundary are not tapered, or the outer
        # border would end up with zero weight (see tiling.feather_ramp)
        ramp = window_ramp(plan, y, x)
        accum[y:y + win, x:x + win] += np.asarray(pred, dtype=np.float64) * ramp
        weight[y:y + win, x:x + win] += ramp

    if weight.min() <= 0:
        raise ValueError(
            f"{int((weight <= 0).sum())} pixel(s) received no window coverage - the tile "
            "plan does not cover the full surface. Planning bug, not a model failure")

    surface = (accum / weight).astype(np.float32)
    if surface.shape != (int(plan.height), int(plan.width)):
        raise ValueError(
            f"blended surface is {surface.shape}, expected {(plan.height, plan.width)}")
    if not np.isfinite(surface).all():
        raise ValueError("blended surface contains non-finite values after division")
    return surface
