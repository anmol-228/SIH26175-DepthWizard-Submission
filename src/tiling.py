"""Tiling and feather-blending primitives - DW-S1-07 (#24), extracted from K1.

`PHASE2_RED_FLAGS/scripts/K1_tiling_stitch.py` proved sliding-window inference with
linear feather blending on a full Vaihingen tile (seam gradient ratio 5.97x -> 0.83x),
but it is a top-level script with zero functions, hardcoded to area 26 and Vaihingen
paths. Nothing could call it.

This module is that logic **extracted and generalized** - not a reimplementation, but not
a byte-for-byte copy either. The distinction matters:

  Preserved exactly: the feather arithmetic (linear ramp over the overlap width) and the
  accumulate-weighted / divide-by-weight blend. Same computation, same numbers.

  Deliberately generalized: edge coverage. K1 walked a fixed stride and patched the
  right/bottom remainder in a second pass, and tapered all four sides of every window -
  which drives the outer border of the surface to zero weight. K1 could ignore that
  because it evaluated on `weight > 0` and discarded those pixels. A runner has to emit a
  complete surface, so the window plan folds the remainder into one deduplicated pass and
  `window_ramp()` skips the taper on sides that lie on the surface boundary. See the note
  on `feather_ramp()`.

K1 itself is deliberately left alone: it is committed Sprint-0 evidence, and its recorded
seam numbers must stay reproducible from the script that produced them.

Ownership per #24: this module is the tiling geometry only (Nirmalya, step 2). The model
stages that consume it - `predict_depth()` and `blend_tiles()` - are Anmol's lane and live
in `src/model_stages.py`, so the two PRs touch different files.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# K1's verified defaults. Recorded in run_manifest.json rather than assumed, so a run is
# reproducible even if these change later.
DEFAULT_WINDOW = 512
DEFAULT_STRIDE = 384          # 128 px overlap


@dataclass(frozen=True)
class TilePlan:
    """Where every inference window sits on the full surface."""
    height: int
    width: int
    window: int
    stride: int
    origins: tuple[tuple[int, int], ...]      # (y, x) top-left of each window

    @property
    def n_windows(self) -> int:
        return len(self.origins)

    def as_manifest(self) -> dict:
        return {"height": self.height, "width": self.width, "window": self.window,
                "stride": self.stride, "n_windows": self.n_windows}


def plan_windows(height: int, width: int, window: int = DEFAULT_WINDOW,
                 stride: int = DEFAULT_STRIDE) -> TilePlan:
    """Enumerate window origins covering (height, width), including the edge remainder.

    K1 walked `range(0, H - WIN + 1, STRIDE)` and then patched the right/bottom edges
    separately, because a stride that does not divide evenly leaves an uncovered strip.
    The edge anchors are folded into one deduplicated plan here instead of a second pass.
    Same window positions as K1's two passes combined, without the double-counting the
    second pass had to guard against - equivalent coverage, not an identical code path.
    """
    if window <= 0 or stride <= 0:
        raise ValueError(f"window and stride must be positive, got {window}, {stride}")
    if height < window or width < window:
        raise ValueError(
            f"surface {height}x{width} is smaller than the {window}x{window} window; "
            "supply a larger image or a smaller --window")

    ys = list(range(0, height - window + 1, stride))
    xs = list(range(0, width - window + 1, stride))
    if ys[-1] != height - window:
        ys.append(height - window)          # bottom edge remainder
    if xs[-1] != width - window:
        xs.append(width - window)           # right edge remainder

    origins = tuple((y, x) for y in ys for x in xs)
    return TilePlan(height=height, width=width, window=window, stride=stride,
                    origins=origins)


def feather_ramp(window: int = DEFAULT_WINDOW, stride: int = DEFAULT_STRIDE, *,
                 taper_top: bool = True, taper_bottom: bool = True,
                 taper_left: bool = True, taper_right: bool = True) -> np.ndarray:
    """Linear edge-taper weight for one window - K1's ramp, with per-side control.

    Overlapping predictions are averaged weighted by this ramp, so a window contributes
    least at the edges where its neighbour contributes most. That is what collapses the
    visible seam; a hard paste leaves a boundary discontinuity K1 measured at 5.97x the
    interior gradient.

    The per-side flags are the one deliberate difference from K1. K1 tapered all four
    sides unconditionally, which drives the weight to exactly zero along the outer border
    of the whole surface - there is no neighbouring window there to make up the
    difference. K1 could ignore that because it evaluated on `weight > 0` only and simply
    discarded those pixels. A runner has to emit a complete surface, so a window sitting
    against the image boundary must not taper on that side. Verified: without this,
    blending a 1100x900 image leaves 3996 zero-weight pixels.
    """
    ramp = np.ones((window, window), dtype=np.float64)
    edge = window - stride
    if edge <= 0:
        return ramp
    lin = np.linspace(0, 1, edge)
    if taper_top:
        ramp[:edge, :] *= lin[:, None]
    if taper_bottom:
        ramp[-edge:, :] *= lin[::-1, None]
    if taper_left:
        ramp[:, :edge] *= lin[None, :]
    if taper_right:
        ramp[:, -edge:] *= lin[None, ::-1]
    return ramp


def window_ramp(plan: TilePlan, y: int, x: int) -> np.ndarray:
    """Feather ramp for the window at (y, x), tapering only on interior sides.

    A side that lies on the surface boundary has no neighbour to blend with, so tapering
    it would leave that strip underweighted (or unweighted entirely).
    """
    return feather_ramp(
        plan.window, plan.stride,
        taper_top=y > 0,
        taper_bottom=(y + plan.window) < plan.height,
        taper_left=x > 0,
        taper_right=(x + plan.window) < plan.width,
    )


def iter_tiles(rgb: np.ndarray, plan: TilePlan):
    """Yield ((y, x), patch) for every planned window. Pure geometry, no model."""
    for (y, x) in plan.origins:
        yield (y, x), rgb[y:y + plan.window, x:x + plan.window]
