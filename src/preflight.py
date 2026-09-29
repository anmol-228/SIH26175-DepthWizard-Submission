"""Failure-aware calibration pre-flight gate - DW-S1-10 (#25), continued by DW-S1-11 (#32).

When reference-point coverage is inadequate, the runner must emit the relative DSM only,
with an explicit reason, rather than a believable-but-wrong metric answer. A judge can
catch a confident wrong number; this makes the system decline instead.

WHY THIS EXISTS, with the measurement behind it. From
`PHASE2_RED_FLAGS/outputs/metrics/F_calibration_distribution.json`, N=10 reference points
on real Vaihingen predictions:

    uniform     mean RMSE  4.856 m   (max  5.464)
    stratified  mean RMSE  4.701 m
    edge-biased mean RMSE  6.686 m
    clustered   mean RMSE 15.324 m   (max 28.423)

Same point *count*, ~3x worse mean error and ~5x worse tail purely from where the points
sit. So the finding is not "use at least 10 points" - it is "use points spread across the
scene", and spatial coverage is the thing worth gating on.

WHAT THIS DELIBERATELY DOES NOT DO (per Anmol's methodology review on #25):

  - The thresholds here are **provisional engineering floors, not validated cutoffs.**
    The experiments above establish that clustered geometry is dangerous; they do not
    establish a hull-fraction or spread number that separates safe from unsafe. Every
    threshold is a constructor argument, and `preflight_result["thresholds_provisional"]`
    is True so no downstream consumer can mistake them for calibrated values.
  - **No hilly-relief signature.** The hilly failure is real and measured (whole-scene
    calibration collapses on Mount Adams while local correlation stays high), but no
    runtime detector computable from the relative surface alone has been validated. Per
    the issue's fallback clause this ships count + coverage, and `checks["hilly_signature"]`
    is None / `"unsupported"` rather than an invented number.

DW-S1-11 (#32) tried, and rejects the result rather than shipping it. The canonical
evidence is Anmol's methodology checkpoint, `(internal note, not included)`
(PR #35): three candidate signals (windowed relief heterogeneity; sparse-10 RANSAC inlier
ratio; sparse-fit residual normalized by the reference points' own elevation range), tested
scene-consistently (Mount Adams vs four individual Vaihingen areas) through the real
production `plan_windows -> predict_depth -> blend_tiles -> normalize_relative` path, all
either REJECTED BY EVIDENCE or, for the one signal showing any separation, MORE EVIDENCE
REQUIRED on an n=2 sample - not a threshold. See `HILLY_SIGNATURE_EVIDENCE` below.
`hilly_relief_signature()` stays unsupported for that documented reason, not for lack of
an attempt.

DW-P0-A5 (#32, Phase-2 activation comment 2026-09-11): a *different, bounded* signal -
explicitly not a revival of the three rejected model-output signals above. Where those
read the model's own (unreliable) relative surface, this one reads only the elevation
values already carried by `--reference` itself: `reference_elevation_relief()` is
`max - min` of the reference points' own supplied elevations (row, col, **elevation_m**),
a purely geometric, model-independent number the runner already has by the time the gate
runs (`run_pipeline.load_reference_points` guarantees an (N, 3) finite array before
`calibrate()` ever calls `preflight_checks`).

The provisional threshold below is derived from the SAME statistic it is compared
against at runtime: `PHASE2_RED_FLAGS/outputs/metrics/E_hilly_risk_signal_experiment.json`
preserves, per RANSAC seed, `sampled_reference_range_m` - `max - min` over the exact 10
sampled reference points used in that seed's fit, on Mount Adams (known-failure) and four
Vaihingen areas (known-good). See `HILLY_RELIEF_EVIDENCE` and
`10_DOCUMENTATION/evidence/DW_P0_A5_hilly_relief_gate/reference_relief_threshold_analysis.json`
for the full derivation and its explicit limitations (principally: one known-failure
scene, resampled 5 times - not five independent failure terrains). This is a separate
check from `hilly_signature` / `hilly_relief_signature()` above, which stay exactly as
DW-S1-11 left them - unsupported, for the documented, still-valid reason that the
*model-output* signal family failed.

Gate-fail is **not an error**: the runner stays relative-only, records `gate_reason`, and
exits 0. A gate firing is the system working.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# --- Provisional thresholds -------------------------------------------------------
# None of these is a validated cutoff. Each is a configurable floor with its source.

# Calibration is empirically unstable below ~10 points per scene (BASELINE_v1.0 / Part VI:
# "degrades gracefully down to roughly 10-50 points/tile, becomes unstable below ~10").
# The same source is explicit that count alone is the weaker predictor - see MIN_HULL_*.
MIN_REFERENCE_POINTS = 10

# No measured hull-fraction cutoff exists. 0.15 is a deliberately permissive floor chosen
# to catch the *clustered* geometry F_calibration_distribution.json shows is dangerous
# (mean 15.32 m / max 28.42 m) without rejecting the merely-imperfect. It needs calibration
# against real 2-D GCP geometry before it can be called anything but provisional.
MIN_HULL_COVERAGE = 0.15

# Normalized nearest-neighbour spread. Guards the degenerate case where points satisfy
# the hull check but are bunched inside it.
#
# Bug found and fixed 2026-09-23, via issue #91 Track B's real N=100 validation evidence
# (10_DOCUMENTATION/evidence/DW_ISSUE91_TRACK_B_GATE_VALIDATION_N100): the original formula
# (mean nearest-neighbour distance / FIXED AOI diagonal) is not scale-invariant -- expected
# NN distance under near-uniform sampling shrinks as ~1/sqrt(N), so the statistic falls as N
# grows regardless of point quality. 0.05 was derived from N=10-16 evidence and produced a
# 100% false-refusal rate on known-good Vaihingen controls at real N=100 (Mount Adams still
# correctly declined -- via the separate, unaffected reference_relief gate). Fixed by
# normalizing against the THEORETICALLY EXPECTED nearest-neighbour distance for N uniformly-
# distributed points (a standard Poisson-process approximation, area / N) rather than a fixed
# diagonal -- see normalized_spread().
#
# Second bug found in review (PR #99, Anmol): the first fix used the MEAN nearest-neighbour
# distance, which is not robust to a majority-clustered-plus-outliers layout -- e.g. ~96/100
# points tightly bunched near the AOI centre with 4 points near the AOI corners: the 4 corner
# points' huge NN distances drag the MEAN up enough to pass (observed ratio ~0.56, above the
# 0.5 floor) even though the reference distribution is still dangerously clustered, and the
# corner points alone are enough to pass hull_coverage too, so no other check catches it
# either. Fixed by using the MEDIAN nearest-neighbour distance instead of the mean -- a
# median is not dragged upward by a small number of outliers, so it correctly reflects that
# the majority of points are bunched (median ratio on the same counterexample: ~0.05-0.06,
# same magnitude as an all-clustered set). See test_majority_clustered_with_hull_stretching_
# outliers_still_declines in tests/test_preflight.py.
#
# 0.25 is re-derived (via median, not mean) from: (a) Anmol's real N=100 evidence
# (10_DOCUMENTATION/evidence/DW_ISSUE91_TRACK_B_GATE_VALIDATION_N100), re-computed with the
# median formula against the real Vaihingen DSMs and Mount Adams' documented shape -- lowest
# observed ratio across all 25 real-geometry runs: 0.7389; (b) the existing N=10-16
# clustered/stratified fixtures in tests/test_preflight.py -- clustered geometry ~0.058-0.079;
# (c) Anmol's majority-clustered-plus-outliers counterexample -- ~0.049-0.061. 0.25 sits close
# to the geometric mean of the closest good/bad observations (0.7389 and 0.0787), giving a
# balanced margin on both sides: >2.9x below every well-spread case seen, >3.1x above every
# clustered/outlier-stretched case seen, across the small-N, N=100-uniform, and
# majority-clustered-with-outliers regimes. Still provisional (no measured cutoff exists
# either), but no longer breaks at realistic N or at this outlier pattern.
MIN_NORMALIZED_SPREAD = 0.25

EVIDENCE = ("PHASE2_RED_FLAGS/outputs/metrics/F_calibration_distribution.json "
            "(N=10: uniform 4.856 m, stratified 4.701 m, edge-biased 6.686 m, "
            "clustered 15.324 m mean / 28.423 m max)")

# DW-S1-11 (#32) canonical evidence: PR #35, `(internal note, not included)`
# + `PHASE2_RED_FLAGS/outputs/metrics/E_hilly_risk_signal_experiment.json` (Anmol,
# methodology/evidence owner per #32's review-route table). Scene-consistent comparison
# (Mount Adams's one contiguous AOI vs four individual Vaihingen areas spanning the
# available heterogeneity range) through the real, unmodified production path
# (plan_windows -> predict_depth -> blend_tiles -> normalize_relative), three candidate
# signals:
#   A  windowed relative-surface relief heterogeneity (raw tile / blended / normalized) -
#      REJECTED BY EVIDENCE: Mount Adams's runtime ratio (2.17x) sits on top of Vaihingen's
#      own most-heterogeneous controls (2.05x-2.15x), though the ground-truth column shows
#      a real gradient the model's own output does not preserve at any pipeline stage.
#   B1 sparse-10 RANSAC inlier ratio - REJECTED BY EVIDENCE: a directional trend exists
#      (Mount Adams lowest) but individual-seed ranges overlap completely.
#   B2 sparse-fit residual normalized by that seed's own 10 sampled reference points'
#      elevation range - REJECTED BY EVIDENCE across all fits (Cohort A); the subset that
#      already passes #24/PR #34's existing `a>0` metric-fit gate (Cohort B) shows no
#      in-sample overlap, but on n=2 surviving Mount Adams data points this is MORE
#      EVIDENCE REQUIRED, not a threshold.
# Independently reproduced by Nirmalya during #32 review: the exact same Vaihingen-side
# numbers reproduce bit-for-bit (including individual per-seed RANSAC inlier ratios) by
# re-running PR #35's own functions against the real checkpoint; a smaller, independent
# ground-truth-only check (33 Vaihingen DSM tiles: window-relief ratio 1.44x-7.14x vs
# Mount Adams's 15.42x) reached the same conclusion by a different, less rigorous route -
# real terrain heterogeneity is measurable in ground truth but the model's own relative
# output does not preserve enough of it to be a defensible runtime signal.
HILLY_SIGNATURE_EVIDENCE = (
    "DW-S1-11 (#32), PR #35 ((internal note, not included)): three "
    "candidate signals (windowed relief heterogeneity; sparse-10 RANSAC inlier ratio; "
    "reference-range-normalized sparse-fit residual) tested scene-consistently against "
    "Mount Adams (known severe failure) and four individual Vaihingen areas through the "
    "real production predict/blend/normalize path. All REJECTED BY EVIDENCE, except one "
    "residual-based signal restricted to fits already passing the existing a>0 metric "
    "gate, which shows no in-sample overlap on only 2 surviving data points - classified "
    "MORE EVIDENCE REQUIRED, not a threshold. No production hilly-risk gate was "
    "implemented from this model-output signal family (see HILLY_RELIEF_EVIDENCE for the "
    "separate, later reference-elevation-relief guard DW-P0-A5 does ship)."
)

# --- DW-P0-A5 (#32, 2026-09-11 Phase-2 activation): reference-elevation relief --------
# A separate, geometric-only signal: max-min of the SUPPLIED reference points' own
# elevations. Not a revival of HILLY_SIGNATURE_EVIDENCE's rejected model-output signals -
# see the module docstring.
#
# THRESHOLD PROVENANCE, PRIMARY SOURCE - the SAME statistic compared at runtime:
#
#   `PHASE2_RED_FLAGS/outputs/metrics/E_hilly_risk_signal_experiment.json`, field
#   `scenes.*.signal_B_sparse10_ransac_on_blended[*].sampled_reference_range_m` - per
#   RANSAC seed, `max - min` over the exact 10 sampled reference points used in that
#   seed's fit (identical operation to `reference_elevation_relief()` below, applied to a
#   real `--reference` file's elevations - not a dense/oracle statistic). 5 seeds x 5
#   scenes, ALL included regardless of downstream fit validity (that is post-fit
#   information this pre-fit threshold must not condition on):
#
#     known-good (4 Vaihingen areas, 20 samples): 7.774 - 44.355 m  (max = 44.355 m)
#     known-failure (Mount Adams, 5 samples):     130.650 - 266.360 m  (min = 130.650 m)
#
#   No overlap between the two groups in this preserved dataset (good_max=44.355 m <
#   failure_min=130.650 m). Frozen rule, declared before inspecting these numbers (see
#   `10_DOCUMENTATION/evidence/DW_P0_A5_hilly_relief_gate/reference_relief_threshold_analysis.json`
#   for the exact script and full per-sample data): threshold = (good_max + failure_min) / 2
#   = 87.50 m (exact: 87.50271606445312, rounded here as a provisional engineering value,
#   not out of claimed precision). In-sample at this threshold: 0 known-good samples above
#   it, 0 known-failure samples at/below it - see the analysis artifact for the full detail
#   and its own explicit limitation that this is screening evidence on n=1 known-failure
#   scene (5 reference-point resamples of ONE Mount Adams terrain, not five independent
#   failure terrains), not a validated error rate.
#
# CORROBORATING CONTEXT ONLY (not the threshold's derivation):
#
#   `PHASE2_RED_FLAGS/outputs/metrics/E_hilly_terrain_eval.json` - 28 per-patch oracle
#   calibrations, Mount Adams, both checkpoint variants. This file's `relief_m` is a DENSE,
#   whole-patch statistic (`PHASE2_RED_FLAGS/scripts/E_02_make_patches.py`: `d.max() -
#   d.min()` over every pixel of a 512x512 ground-truth DEM patch) - a DIFFERENT statistic
#   from the sparse reference-point range this gate actually uses, so it does not derive
#   this threshold; it corroborates that observed lower-relief patches in this preserved
#   Mount Adams evaluation (relief_m <= 118.54 m, 26 of 28 patches, both variants) stay
#   near oracle RMSE 3.03-23.89 m, while observed high-relief / high-error patches
#   (relief_m >= 222.80 m, 2 of 28 patches) reach oracle RMSE 41.66-62.33 m.
#
#   `(internal note, not included)` SS3/SS8 (full-AOI, not per-patch) -
#   also corroborating context, also a dense statistic: Vaihingen area6 (highest-relief
#   Vaihingen control) 67.16 m full-scene relief vs Mount Adams 370.4 m. These two dense
#   views (per-patch and full-AOI) are related evidence from the same preserved Mount
#   Adams/Vaihingen data, not independent confirmations of each other.
#
# `thresholds_provisional=True` applies to this floor, same as every other threshold in
# this file. This is a provisional engineering separator inside a preserved, but small
# (n=25 total samples, one known-failure scene), direct-statistic gap - not a validated
# accuracy cutoff. A HIGH sparse reading is the direction this evidence supports; a LOW
# sparse reading is NOT proof of safety outside this preserved dataset - a genuinely hilly
# scene whose supplied reference points happen to sit in a narrow elevation band could read
# low and pass this check undetected. See the analysis artifact and
# `10_DOCUMENTATION/evidence/DW_P0_A5_hilly_relief_gate/README.md` for the full disclosure.
HILLY_RELIEF_THRESHOLD_M = 87.5

HILLY_RELIEF_EVIDENCE = (
    "DW-P0-A5 (#32, 2026-09-11 Phase-2 activation): reference-point elevation relief "
    "(max-min of the supplied --reference elevations), a geometric signal independent of "
    "model output. PRIMARY threshold source - the SAME statistic used at runtime: "
    "PHASE2_RED_FLAGS/outputs/metrics/E_hilly_risk_signal_experiment.json's "
    "sampled_reference_range_m, per RANSAC seed, over the same 10 sampled reference points "
    "used in that seed's fit (5 seeds x 5 scenes, all included regardless of downstream "
    "fit validity - pre-fit threshold, no post-fit conditioning). Known-good (4 Vaihingen "
    "areas, 20 samples): 7.774-44.355 m. Known-failure (Mount Adams, 5 samples): "
    "130.650-266.360 m. No overlap (good_max=44.355 m < failure_min=130.650 m). Frozen "
    "midpoint rule: threshold = (good_max + failure_min) / 2 = 87.50 m (exact "
    "87.50271606445312). In-sample: 0 known-good samples above threshold, 0 known-failure "
    "samples at/below it - screening evidence on n=1 known-failure scene, not a validated "
    "error rate. Full derivation: "
    "10_DOCUMENTATION/evidence/DW_P0_A5_hilly_relief_gate/reference_relief_threshold_analysis.json. "
    "CORROBORATING CONTEXT ONLY (does not derive this threshold, both are DENSE whole-patch/"
    "full-AOI statistics, not the sparse statistic above): "
    "PHASE2_RED_FLAGS/outputs/metrics/E_hilly_terrain_eval.json's per-patch oracle results "
    "(observed lower-relief patches relief_m <= 118.54 m -> RMSE 3.03-23.89 m; observed "
    "high-relief/high-error patches relief_m >= 222.80 m -> RMSE 41.66-62.33 m), and "
    "(internal note, not included) SS3/SS8's full-AOI comparison (Vaihingen "
    "area6 67.16 m vs Mount Adams 370.4 m) - related evidence from the same preserved data, "
    "not independent confirmations of each other. Provisional floor, not a validated "
    "cutoff - see HILLY_RELIEF_THRESHOLD_M. A LOW sparse reading is NOT proof of safety "
    "outside this preserved dataset - a hilly scene sampled in a narrow elevation band "
    "could understate its own relief and pass undetected. Disclosed, open limitation."
)


@dataclass
class PreflightResult:
    """Outcome of the gate. `passed=False` means relative-only, not failure."""
    passed: bool
    reason: str
    checks: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "pass": self.passed,
            "reason": self.reason,
            "checks": self.checks,
            "thresholds_provisional": True,
            "evidence": EVIDENCE,
            "IMPORTANT": ("Thresholds are engineering guards derived from measured findings, "
                          "NOT validated accuracy guarantees. A pass does not certify metric "
                          "accuracy; it only means no known-dangerous geometry or reference-"
                          "elevation relief was detected."),
        }


def _as_points(reference_points) -> np.ndarray:
    """Coerce to an (N, >=2) float array of (row, col, ...). Raises on malformed input."""
    pts = np.asarray(reference_points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError(
            f"reference_points must be (N, >=2) as (row, col[, elevation]); got shape {pts.shape}")
    if not np.isfinite(pts[:, :2]).all():
        raise ValueError("reference_points contain non-finite row/col coordinates")
    return pts


def hull_coverage(reference_points, shape: tuple[int, int]) -> float:
    """Fraction of the AOI covered by the points' convex hull, in [0, 1].

    This is the check the F-distribution evidence most directly supports: clustered points
    produce a tiny hull, and a scale/shift fit constrained only inside that hull is being
    extrapolated everywhere else.
    """
    pts = _as_points(reference_points)[:, :2]
    h, w = shape
    if len(pts) < 3:
        return 0.0                      # a hull needs 3 non-collinear points
    try:
        from scipy.spatial import ConvexHull, QhullError    # noqa: PLC0415
        try:
            return float(ConvexHull(pts).volume / (h * w))  # 2-D "volume" is area
        except QhullError:
            return 0.0                  # collinear/degenerate -> zero area, correctly
    except ImportError:
        # Bounding-box fallback. Over-estimates a non-convex spread, so it is the
        # permissive direction: it can let a marginal set through, never wrongly reject.
        span_r = pts[:, 0].max() - pts[:, 0].min()
        span_c = pts[:, 1].max() - pts[:, 1].min()
        return float((span_r * span_c) / (h * w))


def normalized_spread(reference_points, shape: tuple[int, int]) -> float:
    """MEDIAN nearest-neighbour distance / expected nearest-neighbour distance for N
    uniformly-distributed points over the AOI. Statistical interpretation: this is a
    robust estimate of "how bunched is a TYPICAL point's nearest neighbour, relative to
    how bunched we'd expect under genuinely uniform coverage" - ~1.0 for a genuinely
    well-spread set, regardless of N; well below 1.0 when most points sit close to
    another point (bunched together), even if a few points are far apart.

    Catches points that pass the hull check but bunch inside it - a few outliers can
    stretch a hull that most points sit nowhere near.

    Two fixes layered on the original mean-NN/fixed-diagonal formula, in order:

    1. Normalizes against the expected value (a standard Poisson-process nearest-
       neighbour approximation, ~0.5 * sqrt(area / N)) rather than a fixed AOI diagonal -
       a fixed diagonal made this statistic fall as N grew regardless of point quality,
       wrongly declining known-good geometry at realistic reference counts (issue #91
       Track B's N=100 evidence).
    2. Uses the MEDIAN nearest-neighbour distance, not the mean - the mean is not robust
       to a majority-clustered-plus-a-few-outliers layout (e.g. 96/100 points bunched
       near the AOI centre, 4 points near the corners): the outliers' huge individual NN
       distances drag the MEAN up enough to wrongly pass, even though most points are
       still dangerously bunched. The MEDIAN reflects what a typical point's neighbourhood
       actually looks like and is not dragged by a small number of outliers (PR #99,
       Anmol's review).

    See MIN_NORMALIZED_SPREAD's comment for the evidence behind both fixes and the
    re-derived threshold.
    """
    pts = _as_points(reference_points)[:, :2]
    n = len(pts)
    if n < 2:
        return 0.0
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    median_nn_distance = float(np.median(d.min(axis=1)))
    area = float(shape[0]) * float(shape[1])
    expected_nn_distance = 0.5 * math.sqrt(area / n) if area > 0 else 0.0
    return median_nn_distance / expected_nn_distance if expected_nn_distance else 0.0


def hilly_relief_signature(relative_surface, windows: int = 4):
    """Not implemented - deliberately, and now for a measured reason rather than none.

    The hilly failure is measured and real (~60-70 m RMSE under whole-scene calibration on
    Mount Adams, pooled correlation collapsing to 0.05-0.24 while per-patch stays
    0.96-0.97). DW-S1-11 (#32), PR #35, tested three candidate runtime signals
    scene-consistently against Mount Adams and four Vaihingen controls through the real
    production predict/blend/normalize path; all were REJECTED BY EVIDENCE or, for the one
    with any separation, classified MORE EVIDENCE REQUIRED on a 2-sample cohort - see
    `HILLY_SIGNATURE_EVIDENCE`. Inventing a threshold on top of that would be exactly the
    "engineering guards vs validated guarantees" conflation #25 forbids, so this still
    returns None - not for lack of an attempt, but because none of the attempts made
    clears the bar for a production gate.
    """
    return None


def reference_elevation_relief(reference_points) -> float | None:
    """max - min of the SUPPLIED reference points' own elevations, in metres.

    DW-P0-A5 (#32): unlike `hilly_relief_signature()` above, this reads no model output
    at all - only the elevation column a `--reference` file already carries. `run_pipeline.
    load_reference_points` guarantees an (N, 3) array of finite (row, col, elevation_m)
    before `calibrate()` ever reaches the gate, so in the real pipeline this is always
    well-defined once a reference is supplied.

    Returns None ("unsupported", not "safe") when there is nothing to compute from: fewer
    than 2 points, or no elevation column (row, col only - `preflight_checks` still accepts
    that shape for the existing geometry checks). Non-finite elevation values raise
    ValueError, the same treatment `_as_points` already gives non-finite row/col - a
    malformed-input error, not a gate decision (production never reaches this with
    non-finite elevation; `load_reference_points` already rejects that upstream at exit 2).

    Compared in `preflight_checks` against `HILLY_RELIEF_THRESHOLD_M`, which was derived
    from the SAME statistic (sampled_reference_range_m in
    `E_hilly_risk_signal_experiment.json`) - see that constant's comment and
    `HILLY_RELIEF_EVIDENCE` for the exact derivation, its n=1-known-failure-scene
    limitation, and why a low reading is not proof of safety.
    """
    pts = np.asarray(reference_points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 3 or len(pts) < 2:
        return None
    elev = pts[:, 2]
    if not np.isfinite(elev).all():
        raise ValueError("reference_points elevation column (index 2) contains non-finite values")
    return float(elev.max() - elev.min())


def preflight_checks(reference_points, relative_surface, *,
                     min_points: int = MIN_REFERENCE_POINTS,
                     min_hull_coverage: float = MIN_HULL_COVERAGE,
                     min_spread: float = MIN_NORMALIZED_SPREAD,
                     max_reference_relief_m: float = HILLY_RELIEF_THRESHOLD_M) -> PreflightResult:
    """Decide whether metric calibration is defensible for this input.

    Returns a PreflightResult; `passed=False` means the runner emits the relative DSM only
    and records the reason. It never raises for a failing gate - that is a normal outcome.
    Malformed *input* still raises, because that is a caller bug rather than a gate result.
    """
    surface = np.asarray(relative_surface)
    if surface.ndim != 2:
        raise ValueError(f"relative_surface must be 2-D (H, W); got {surface.ndim}-D")
    shape = (int(surface.shape[0]), int(surface.shape[1]))

    if reference_points is None or len(reference_points) == 0:
        return PreflightResult(False, "no reference points supplied", {
            "count": 0, "hull_coverage": 0.0, "spread": 0.0,
            "hilly_signature": None,
            "hilly_signature_status": "unsupported - three signals tested and rejected, not untried (#32, PR #35)",
            "reference_relief_m": None,
            "reference_relief_status": "unsupported - no reference points supplied",
            "reference_relief_threshold_m": max_reference_relief_m,
        })

    pts = _as_points(reference_points)
    count = int(len(pts))
    cov = hull_coverage(pts, shape)
    spread = normalized_spread(pts, shape)
    relief_m = reference_elevation_relief(pts)
    if relief_m is None:
        relief_status = "unsupported - reference points carry no elevation column (row, col only)"
    elif relief_m > max_reference_relief_m:
        relief_status = "relief-risk gate triggered - see reason"
    else:
        relief_status = "within provisional floor"

    checks = {
        "count": count,
        "hull_coverage": round(cov, 6),
        "spread": round(spread, 6),
        "hilly_signature": None,
        "hilly_signature_status": "unsupported - three signals tested and rejected, not untried (#32, PR #35)",
        "hilly_signature_evidence": HILLY_SIGNATURE_EVIDENCE,
        "reference_relief_m": None if relief_m is None else round(relief_m, 3),
        "reference_relief_status": relief_status,
        "reference_relief_threshold_m": max_reference_relief_m,
        "reference_relief_evidence": HILLY_RELIEF_EVIDENCE,
        "thresholds": {"min_points": min_points, "min_hull_coverage": min_hull_coverage,
                       "min_spread": min_spread,
                       "max_reference_relief_m": max_reference_relief_m},
    }

    failures = []
    if count < min_points:
        failures.append(f"only {count} reference point(s), provisional floor is {min_points}")
    if cov < min_hull_coverage:
        failures.append(
            f"reference points cover {cov:.1%} of the AOI (provisional floor {min_hull_coverage:.0%}); "
            "clustered geometry is the failure mode measured at 15.32 m mean / 28.42 m worst RMSE")
    if spread < min_spread:
        failures.append(
            f"normalized spread {spread:.4f} below provisional floor {min_spread}; "
            "points are bunched even if their hull is wide")
    if relief_m is not None and relief_m > max_reference_relief_m:
        failures.append(
            f"reference points span {relief_m:.1f} m of elevation relief (provisional "
            f"hilly-risk floor {max_reference_relief_m:.1f} m); one whole-scene scale/shift "
            "fit over that much relief is the failure mode measured at 60-70 m RMSE on "
            "Mount Adams (see #32, HILLY_RELIEF_EVIDENCE) - declining rather than writing "
            "an unsupported confident metric DSM")

    if failures:
        return PreflightResult(False, "; ".join(failures), checks)
    return PreflightResult(
        True,
        f"{count} reference points, {cov:.1%} hull coverage, spread {spread:.4f}, "
        f"{'no reference elevation data' if relief_m is None else f'{relief_m:.1f} m relief'}"
        " - no known-dangerous geometry or relief detected (provisional thresholds)",
        checks)
