"""Tests for the calibration pre-flight gate - DW-S1-10 (#25).

Geometries mirror the F-distribution experiment the gate is derived from: clustered points
must fail, spread points must pass. No checkpoint, network or Vaihingen data needed.

    python tests/test_preflight.py
    pytest tests/test_preflight.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from preflight import (  # noqa: E402
    HILLY_RELIEF_EVIDENCE, HILLY_RELIEF_THRESHOLD_M, HILLY_SIGNATURE_EVIDENCE,
    MIN_HULL_COVERAGE, MIN_NORMALIZED_SPREAD, MIN_REFERENCE_POINTS, PreflightResult,
    hilly_relief_signature, hull_coverage, normalized_spread, preflight_checks,
    reference_elevation_relief,
)

SHAPE = (1000, 1000)
SURFACE = np.zeros(SHAPE, dtype=np.float32)


def _stratified(n=16):
    """Points spread across the scene - the 4.70 m case."""
    g = int(np.sqrt(n))
    rs = np.linspace(80, SHAPE[0] - 80, g)
    cs = np.linspace(80, SHAPE[1] - 80, g)
    return np.array([(r, c, 250.0) for r in rs for c in cs])


def _clustered(n=10):
    """Points bunched in one corner - the 15.32 m mean / 28.42 m worst case."""
    rng = np.random.default_rng(42)
    return np.column_stack([rng.uniform(40, 90, n), rng.uniform(40, 90, n),
                            np.full(n, 250.0)])


def _stratified_hilly(n=16, relief_m=200.0, elev_lo=200.0):
    """Same well-spread (row, col) geometry as `_stratified()`, but elevation spans
    `relief_m` metres - isolates the DW-P0-A5 relief check from count/hull/spread, which
    all pass identically here (same geometry `_stratified()` already proves passes)."""
    g = int(np.sqrt(n))
    rs = np.linspace(80, SHAPE[0] - 80, g)
    cs = np.linspace(80, SHAPE[1] - 80, g)
    lo, hi = float(rs.min()), float(rs.max())
    return np.array([(r, c, elev_lo + relief_m * (r - lo) / (hi - lo))
                     for r in rs for c in cs])


def test_stratified_reference_set_passes():
    r = preflight_checks(_stratified(), SURFACE)
    assert r.passed, f"stratified geometry rejected: {r.reason}"
    assert r.checks["count"] >= MIN_REFERENCE_POINTS
    assert r.checks["hull_coverage"] > MIN_HULL_COVERAGE


def test_clustered_reference_set_fails_with_a_reason():
    """The core case: same count as a passing set, dangerous geometry."""
    r = preflight_checks(_clustered(), SURFACE)
    assert not r.passed, "clustered geometry accepted - this is the 28.42 m failure mode"
    assert "cover" in r.reason or "spread" in r.reason
    assert r.checks["count"] == 10, "count alone must not be what rejected it"


def test_count_and_geometry_are_independent_checks():
    """The finding is 'spread across the scene', not 'at least 10 points' - a set can have
    enough points and still be unsafe, which is exactly what the experiment measured."""
    clustered = preflight_checks(_clustered(n=20), SURFACE)
    assert clustered.checks["count"] == 20 >= MIN_REFERENCE_POINTS
    assert not clustered.passed, "20 clustered points must still fail on geometry"


def test_too_few_points_fails_even_when_spread():
    r = preflight_checks(_stratified(4), SURFACE)
    assert not r.passed and "floor" in r.reason


def test_no_reference_points_is_a_clean_fail_not_a_crash():
    for empty in (None, [], np.empty((0, 3))):
        r = preflight_checks(empty, SURFACE)
        assert not r.passed and "no reference points" in r.reason
        assert r.checks["count"] == 0


def test_gate_failure_is_never_an_exception():
    """A firing gate is a successful safety decision (exit 0), not an error."""
    r = preflight_checks(_clustered(), SURFACE)
    assert isinstance(r, PreflightResult) and r.passed is False


def test_malformed_input_still_raises():
    """Caller bugs are distinct from gate results."""
    for bad in (np.array([1, 2, 3]), np.array([[1.0], [2.0]]),
                np.array([[np.nan, 1.0], [2.0, 3.0]])):
        try:
            preflight_checks(bad, SURFACE)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted malformed reference_points {bad.shape}")
    try:
        preflight_checks(_stratified(), np.zeros((4, 4, 4)))
    except ValueError:
        pass
    else:
        raise AssertionError("accepted a 3-D relative_surface")


def test_hilly_signature_is_unsupported_not_invented():
    """#25 is explicit: the hilly failure is proven, a runtime detector is not. Returning a
    number here would be the guard-vs-guarantee conflation the issue forbids."""
    assert hilly_relief_signature(SURFACE) is None
    r = preflight_checks(_stratified(), SURFACE)
    assert r.checks["hilly_signature"] is None
    assert "unsupported" in r.checks["hilly_signature_status"]


def test_hilly_signature_stays_unsupported_after_dw_s1_11_attempt():
    """DW-S1-11 (#32) tested three candidate signatures (PR #35) and rejected all of
    them - confirm the gate still reports 'unsupported', now for that documented reason,
    not silently as if untried. Guards against a future edit quietly re-enabling an
    unvalidated cutoff."""
    assert hilly_relief_signature(SURFACE) is None
    r = preflight_checks(_stratified(), SURFACE)
    assert r.checks["hilly_signature"] is None
    assert "rejected" in r.checks["hilly_signature_status"]
    assert "#32" in r.checks["hilly_signature_status"]
    assert r.checks["hilly_signature_evidence"] == HILLY_SIGNATURE_EVIDENCE
    # the evidence string must cite the canonical PR, not just assert a conclusion
    assert "PR #35" in HILLY_SIGNATURE_EVIDENCE and "REJECTED BY EVIDENCE" in HILLY_SIGNATURE_EVIDENCE


def test_result_marks_thresholds_provisional_and_cites_evidence():
    d = preflight_checks(_stratified(), SURFACE).as_dict()
    assert d["thresholds_provisional"] is True
    assert "F_calibration_distribution.json" in d["evidence"]
    assert "NOT validated accuracy guarantees" in d["IMPORTANT"]


def test_hull_coverage_and_spread_behave_monotonically():
    assert hull_coverage(_clustered(), SHAPE) < hull_coverage(_stratified(), SHAPE)
    assert normalized_spread(_clustered(), SHAPE) < normalized_spread(_stratified(), SHAPE)
    assert hull_coverage(np.array([[10.0, 10.0], [20.0, 20.0]]), SHAPE) == 0.0  # <3 points


def test_thresholds_are_configurable_not_frozen():
    """They are provisional, so a caller must be able to override them."""
    clustered = _clustered()
    assert not preflight_checks(clustered, SURFACE).passed
    relaxed = preflight_checks(clustered, SURFACE, min_points=1,
                               min_hull_coverage=0.0, min_spread=0.0)
    assert relaxed.passed, "thresholds must be overridable"


# --- DW-P0-A5 (#32): reference-elevation relief gate --------------------------------

def test_reference_relief_computes_max_minus_min_of_supplied_elevations():
    pts = np.array([(10.0, 10.0, 300.0), (20.0, 20.0, 250.0), (30.0, 30.0, 410.0)])
    assert reference_elevation_relief(pts) == 160.0


def test_reference_relief_low_risk_scene_allows_metric_branch():
    """Well-spread points (same geometry `_stratified()` already proves passes) with
    elevation relief comfortably under the provisional floor must not trigger the new
    gate - isolates the relief signal from the count/hull/spread checks."""
    pts = _stratified_hilly(relief_m=50.0)                # 50 m < 87.5 m floor
    r = preflight_checks(pts, SURFACE)
    assert r.passed, f"low-relief geometry rejected: {r.reason}"
    assert abs(r.checks["reference_relief_m"] - 50.0) < 1e-6
    assert r.checks["reference_relief_status"] == "within provisional floor"


def test_reference_relief_high_risk_scene_declines_metric_branch():
    """DW-P0-A5 core case: geometry alone would pass (identical to the low-risk fixture
    above), but elevation relief exceeds the provisional floor - the gate must decline
    specifically on relief, not on count/hull/spread."""
    pts = _stratified_hilly(relief_m=200.0)               # 200 m > 87.5 m floor
    r = preflight_checks(pts, SURFACE)
    assert not r.passed, "high-relief reference set accepted - the Mount Adams failure mode"
    assert "relief" in r.reason
    assert r.checks["count"] >= MIN_REFERENCE_POINTS
    assert r.checks["hull_coverage"] > MIN_HULL_COVERAGE, "must decline on relief, not hull coverage"
    assert r.checks["spread"] > MIN_NORMALIZED_SPREAD, "must decline on relief, not spread"
    assert abs(r.checks["reference_relief_m"] - 200.0) < 1e-6
    assert r.checks["reference_relief_status"] == "relief-risk gate triggered - see reason"


def test_reference_relief_boundary_is_strictly_greater_than():
    """Exactly at the provisional floor passes (`>`, not `>=`); a hair above declines - the
    boundary must be a deliberate choice, not an accident of comparison direction."""
    at = _stratified_hilly(relief_m=HILLY_RELIEF_THRESHOLD_M)
    above = _stratified_hilly(relief_m=HILLY_RELIEF_THRESHOLD_M + 0.1)
    assert preflight_checks(at, SURFACE).passed, "exactly-at-threshold relief must not trigger"
    assert not preflight_checks(above, SURFACE).passed, "just-above-threshold relief must trigger"


def test_reference_relief_missing_elevation_column_is_unsupported_not_a_silent_pass():
    """A caller supplying only (row, col) - no elevation - gets an explicit 'unsupported'
    status, never a fabricated 'safe' verdict manufactured from data that was never there.
    The existing geometry checks (unaffected by elevation) still apply on their own."""
    two_col = _stratified()[:, :2]
    r = preflight_checks(two_col, SURFACE)
    assert r.checks["reference_relief_m"] is None
    assert "unsupported" in r.checks["reference_relief_status"]
    assert r.passed, "geometry alone is fine here - only the relief signal is unsupported"


def test_reference_relief_non_finite_elevation_raises():
    """Mirrors `_as_points`'s treatment of non-finite row/col: a malformed-input error, not
    a silently-accepted gate result. Production never reaches this - `run_pipeline.
    load_reference_points` already rejects non-finite elevation upstream at exit 2."""
    bad = _stratified().copy()
    bad[0, 2] = np.nan
    try:
        preflight_checks(bad, SURFACE)
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite elevation was silently accepted")
    try:
        reference_elevation_relief(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("reference_elevation_relief silently accepted non-finite input")


def test_reference_relief_too_few_points_is_unsupported():
    single = np.array([(10.0, 10.0, 300.0)])
    assert reference_elevation_relief(single) is None


def test_reference_relief_result_cites_evidence_and_stays_provisional():
    d = preflight_checks(_stratified(), SURFACE).as_dict()
    assert d["checks"]["reference_relief_evidence"] == HILLY_RELIEF_EVIDENCE
    assert "E_hilly_terrain_eval.json" in HILLY_RELIEF_EVIDENCE
    assert d["checks"]["reference_relief_threshold_m"] == HILLY_RELIEF_THRESHOLD_M
    assert d["thresholds_provisional"] is True, "DW-P0-A5's floor is provisional too"


def test_reference_relief_evidence_cites_the_direct_sparse_statistic_not_a_dense_one():
    """An earlier revision derived HILLY_RELIEF_THRESHOLD_M from DENSE whole-patch relief_m
    (E_02_make_patches.py: max-min over every pixel), applied to a SPARSE runtime statistic
    - a real gap, fixed at the time by disclosure. A later review of
    E_hilly_risk_signal_experiment.json found it already preserves the exact SPARSE
    statistic (sampled_reference_range_m, per RANSAC seed, over the same 10 sampled
    reference points used in that seed's fit) for both known-good and known-failure scenes,
    with a clean non-overlapping separation - so the threshold could be, and now is,
    derived from the SAME statistic compared at runtime, not a different one. This test
    checks the current evidence string's substantive claims (not just magic words) so a
    future edit cannot silently reintroduce the dense/sparse conflation or drop the
    disclosed limitations."""
    evidence = HILLY_RELIEF_EVIDENCE.lower()
    # cites the correct PRIMARY source: the sparse per-seed statistic, not the dense one
    assert "e_hilly_risk_signal_experiment.json" in evidence
    assert "sampled_reference_range_m" in evidence
    assert "pre-fit" in evidence and "no post-fit conditioning" in evidence
    # the observed direct-statistic separation itself
    assert "no overlap" in evidence
    assert "44.355" in evidence and "130.650" in evidence
    # the frozen, deterministic selection rule - not a number chosen after inspection
    assert "good_max + failure_min" in evidence or "good_max" in evidence
    # the dense evidence is explicitly demoted to corroborating-only, never the derivation
    assert "corroborating context only" in evidence
    # the disclosed false-negative direction: why a LOW sparse reading is NOT proof of safety
    assert "not proof of safety" in evidence
    assert "narrow elevation band" in evidence
    # this is in-sample screening evidence, not a validated rate, on one failure scene
    assert "screening evidence" in evidence and "n=1 known-failure scene" in evidence


def test_reference_relief_threshold_is_configurable_not_frozen():
    hilly = _stratified_hilly(relief_m=200.0)
    assert not preflight_checks(hilly, SURFACE).passed
    relaxed = preflight_checks(hilly, SURFACE, max_reference_relief_m=250.0)
    assert relaxed.passed, "relief floor must be overridable like the other provisional thresholds"


def test_reference_relief_does_not_alter_the_rejected_model_output_signal():
    """DW-P0-A5 adds a new check; it must not touch `hilly_signature` / the model-output
    signal DW-S1-11 rejected - both remain None/unsupported regardless of relief."""
    for pts in (_stratified_hilly(relief_m=100.0), _stratified_hilly(relief_m=200.0)):
        r = preflight_checks(pts, SURFACE)
        assert r.checks["hilly_signature"] is None
        assert hilly_relief_signature(SURFACE) is None


# --- normalized_spread scale invariance (fixed 2026-09-23, issue #91 Track B evidence) ------
#
# Bug: the original formula (mean NN distance / FIXED AOI diagonal) is not scale-invariant -
# expected NN distance shrinks as ~1/sqrt(N), so the statistic fell as N grew regardless of
# point quality. Confirmed via issue #91 Track B's real N=100 validation
# (10_DOCUMENTATION/evidence/DW_ISSUE91_TRACK_B_GATE_VALIDATION_N100): the OLD formula gave a
# 100% false-refusal rate on known-good Vaihingen controls at N=100 (Mount Adams still
# correctly declined throughout, via the separate, unaffected reference_relief gate). Fixed by
# normalizing against the expected NN distance for N uniformly-distributed points instead of a
# fixed diagonal.

LARGE_SHAPE = (4000, 2200)  # Mount Adams' real shape, per #91 Track A/B


def _uniform_random(n, shape, seed):
    """Scene-wide uniform-random reference sampling - identical methodology to #91 Track A/B's
    sample_references(), the real production sampling pattern this bug was found under."""
    rng = np.random.default_rng(seed)
    rr = rng.integers(0, shape[0], size=n)
    cc = rng.integers(0, shape[1], size=n)
    elev = np.full(n, 250.0)
    return np.stack([rr.astype(np.float64), cc.astype(np.float64), elev], axis=1)


def _clustered_at_n(n, shape, seed, corner_frac=0.03):
    """Bunched into one small corner, scaled to any N - the large-N analogue of
    `_clustered()` above."""
    rng = np.random.default_rng(seed)
    span_r = shape[0] * corner_frac
    span_c = shape[1] * corner_frac
    rr = rng.uniform(0, span_r, n)
    cc = rng.uniform(0, span_c, n)
    return np.column_stack([rr, cc, np.full(n, 250.0)])


def test_uniform_random_n100_reference_set_no_longer_false_refuses():
    """The core regression: this exact geometry pattern (uniform-random, N=100) is what
    Anmol's real evidence showed failing under the old 0.05/fixed-diagonal formula on every
    one of 20 known-good control runs. It must pass now."""
    for seed in range(7000, 7005):  # same 5-seed stream as #91 Track A/B
        pts = _uniform_random(100, LARGE_SHAPE, seed)
        r = preflight_checks(pts, np.zeros(LARGE_SHAPE, dtype=np.float32))
        assert r.passed, f"seed={seed} false-refused uniform-random N=100 geometry: {r.reason}"
        assert r.checks["spread"] > MIN_NORMALIZED_SPREAD


def test_clustered_geometry_still_declines_at_n100():
    """The fix must not simply relax the gate open - genuinely bad (clustered) geometry at
    realistic N must still be caught."""
    for seed in range(7000, 7005):
        pts = _clustered_at_n(100, LARGE_SHAPE, seed)
        r = preflight_checks(pts, np.zeros(LARGE_SHAPE, dtype=np.float32))
        assert not r.passed, f"seed={seed} accepted clustered N=100 geometry"
        assert "spread" in r.reason


def _majority_clustered_with_corner_outliers(n_clustered=96, n_outliers=4, seed=0):
    """PR #99 (Anmol's review): ~96/100 points tightly bunched near the AOI centre, a few
    points near the corners. The corner points alone stretch the hull enough to pass
    hull_coverage - this fixture isolates the spread check specifically."""
    rng = np.random.default_rng(seed)
    center = rng.normal(loc=[2000, 1100], scale=[30, 30], size=(n_clustered, 2))
    corners = np.array([[80, 80], [80, 2120], [3920, 80], [3920, 2120]],
                       dtype=np.float64)[:n_outliers]
    elev = np.full(n_clustered + n_outliers, 250.0)
    return np.column_stack([np.vstack([center, corners]), elev])


def test_majority_clustered_with_hull_stretching_outliers_still_declines():
    """The counterexample that broke the first (mean-based) fix: a hull-passing but
    majority-clustered layout, where a handful of outlier points near the corners stretch
    the hull enough to pass hull_coverage on their own. The mean-nearest-neighbour version
    of this statistic passed this layout (ratio ~0.56, above a 0.5 floor) because the
    outliers' own huge NN distances dragged the mean up - even though 96% of the points are
    still dangerously bunched. Demonstrates: (1) hull_coverage alone does not catch this,
    (2) the fixed (median-based) spread check does."""
    for seed in range(5):
        pts = _majority_clustered_with_corner_outliers(seed=seed)
        shape = (4000, 2200)
        cov = hull_coverage(pts[:, :2], shape)
        assert cov > MIN_HULL_COVERAGE, (
            f"seed={seed}: fixture no longer isolates the spread check - hull_coverage "
            f"({cov:.3f}) itself would reject this, defeating the point of the test")
        r = preflight_checks(pts, np.zeros(shape, dtype=np.float32))
        assert not r.passed, (
            f"seed={seed}: majority-clustered geometry with hull-stretching outliers was "
            f"accepted - spread={r.checks['spread']:.4f}")
        assert "spread" in r.reason


def test_normalized_spread_is_stable_across_n_for_equally_well_spread_geometry():
    """Direct test of the scale-invariance property itself, not just the pass/fail outcome:
    a well-spread N=16 set (the existing small-N fixture) and a well-spread N=100 set should
    land in a similar range, not differ by an order of magnitude the way the old
    mean-NN/fixed-diagonal formula did (0.198 at N=16 vs ~0.033 at N=100 pre-fix)."""
    small_n = normalized_spread(_stratified(16), SHAPE)
    large_n = normalized_spread(_uniform_random(100, LARGE_SHAPE, 7000)[:, :2], LARGE_SHAPE)
    assert 0.5 < small_n < 5.0, f"small-N well-spread ratio out of expected range: {small_n}"
    assert 0.5 < large_n < 5.0, f"large-N well-spread ratio out of expected range: {large_n}"


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
