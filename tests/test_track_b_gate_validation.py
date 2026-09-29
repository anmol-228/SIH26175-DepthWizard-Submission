"""Targeted tests for Issue #91 Track B's gate-validation finding.

Track B's contract (HILLY_RESCUE_EXECUTION_COUNCIL.md Part 4) describes adding a
hull-coverage/stratification gate to src/preflight.py as new work. Reading the
module and its git history first showed this already exists and is already tested
at small N (commit 24a95b5, DW-S1-10 / #25, merged 2026-09-02 - three weeks before
#91 was opened). So Track B's actual deliverable here is validation evidence of
the EXISTING, unmodified gate at the real N=100 uniform-random geometry #73/#91
Track A use (not the N=10-16 synthetic geometries tests/test_preflight.py already
covers) - see PHASE2_RED_FLAGS/scripts/E_09_track_b_gate_validation_n100.py.

That run surfaced a genuine, previously-undocumented limitation: at N=100,
`normalized_spread()` (mean nearest-neighbour distance / AOI diagonal) triggers the
gate's `spread` check on EVERY scene tested, including all four known-good
Vaihingen controls that pass at small N - a 100% false-refusal rate on known-good
geometry, not the near-zero rate the existing small-N evidence shows. This is a
scale-dependency artifact: mean nearest-neighbour distance shrinks as point count
grows under uniform sampling, independent of whether spatial coverage is
genuinely good or bad, so MIN_NORMALIZED_SPREAD=0.05 (calibrated informally
against N=10-16 cases) is not scale-invariant.

Per the Track A/B stopping rule (no iterative threshold-tuning after seeing
results), this was NOT "fixed" in Track B itself - MIN_NORMALIZED_SPREAD was left
untouched there. This file locked in the discovered behavior as an explicit,
documented, regression-detectable fact, so a future change to
normalized_spread()'s formula or the default threshold would not silently alter
this known limitation without a test noticing.

**Update (PR #99, issue #98):** that future change happened. normalized_spread()
now normalizes against the EXPECTED nearest-neighbour distance for N
uniformly-distributed points (not a fixed diagonal) and uses the MEDIAN, not the
mean, of the observed distances - re-derived against this same real Vaihingen
N=100 evidence plus a majority-clustered-plus-outliers counterexample (see
src/preflight.py's MIN_NORMALIZED_SPREAD comment for the full evidence trail).
The tests below are updated to document the CURRENT (fixed) behavior rather than
silently keep asserting the old bug - per this module's own stated intent, a
formula change was always meant to be caught here, not hidden. The original
finding is preserved in git history and in
10_DOCUMENTATION/evidence/DW_ISSUE91_TRACK_B_GATE_VALIDATION_N100/, which
records the OLD formula's output on real data and remains an accurate historical
record of the mean-formula/0.05-threshold gate's behavior - it does not need
re-generating, since it correctly describes what that exact code, at that
commit, actually did.
"""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from preflight import MIN_NORMALIZED_SPREAD, normalized_spread, preflight_checks  # noqa: E402


def _uniform_random_references(h: int, w: int, n: int, seed: int) -> np.ndarray:
    """Same sampling method as #73/#91 Track A and Track B's E_09 validation script -
    scene-wide uniform random (row, col, elevation)."""
    rng = np.random.default_rng(7000 + seed)
    rr = rng.integers(0, h, size=n)
    cc = rng.integers(0, w, size=n)
    elev = rng.uniform(0, 50, size=n)  # elevation values are irrelevant to this test
    return np.stack([rr.astype(np.float64), cc.astype(np.float64), elev], axis=1)


def test_n100_uniform_random_no_longer_false_refuses_after_pr99_fix():
    """PR #99 fixed the scale-dependency bug this file originally documented: at N=100,
    genuinely well-spread (uniform random, not clustered) references over a large
    real-world AOI shape used to fail the normalized_spread floor on every seed tested
    (the 100% false-refusal finding, Track B/#91). Under the fixed (median, expected-NN-
    normalized) formula, this same geometry now correctly passes."""
    shape = (4000, 2200)  # Mount Adams AOI shape, #73/#91 Track A
    surface = np.zeros(shape, dtype=np.float32)
    for seed in range(5):
        pts = _uniform_random_references(*shape, n=100, seed=seed)
        spread = normalized_spread(pts, shape)
        assert spread >= MIN_NORMALIZED_SPREAD, (
            "regression: N=100 uniform-random spread "
            f"({spread:.4f}) falls below the floor ({MIN_NORMALIZED_SPREAD}) again - "
            "the PR #99 scale-invariance fix may have regressed")

        result = preflight_checks(pts, surface)
        assert result.passed, f"gate should accept well-spread N=100 geometry, got: {result.reason}"


def test_n16_uniform_geometry_still_passes_matching_existing_small_n_coverage():
    """Sanity check that this is specifically an N-scaling artifact, not a change to
    the gate's small-N behavior already covered by tests/test_preflight.py - a
    smaller, well-spread sample at the same AOI shape should still pass."""
    shape = (4000, 2200)
    surface = np.zeros(shape, dtype=np.float32)
    # 4x4 grid, well inside the AOI, mirroring tests/test_preflight.py::_stratified's
    # approach at a size known to pass.
    rows = np.linspace(200, shape[0] - 200, 4)
    cols = np.linspace(200, shape[1] - 200, 4)
    pts = np.array([[r, c, 10.0] for r in rows for c in cols])
    result = preflight_checks(pts, surface)
    assert result.passed, f"expected small well-spread N to still pass, got: {result.reason}"


def test_spread_formula_is_approximately_scale_invariant_after_pr99_fix():
    """Direct confirmation of the FIXED mechanism (inverts the pre-#99 version of this
    test, which asserted the opposite - that spread necessarily shrinks as N grows).
    Normalizing against the EXPECTED nearest-neighbour distance for N points (instead of
    a fixed AOI diagonal) means genuinely well-spread geometry should land in roughly the
    same ratio range regardless of N, not systematically shrink. Checked across multiple
    seeds (a single-seed comparison is too noisy to be a reliable invariance check,
    particularly at N=16) rather than one arbitrary pair."""
    shape = (4000, 2200)

    def sample(n, seed):
        rng = np.random.default_rng(seed)
        rr = rng.integers(0, shape[0], size=n)
        cc = rng.integers(0, shape[1], size=n)
        return np.stack([rr.astype(np.float64), cc.astype(np.float64)], axis=1)

    spreads_16 = [normalized_spread(sample(16, seed), shape) for seed in range(10)]
    spreads_100 = [normalized_spread(sample(100, seed), shape) for seed in range(10)]
    median_16, median_100 = float(np.median(spreads_16)), float(np.median(spreads_100))
    ratio = median_100 / median_16
    assert 0.3 < ratio < 3.0, (
        f"median spread at N=100 ({median_100:.3f}) vs N=16 ({median_16:.3f}), ratio "
        f"{ratio:.3f} - expected roughly the same order of magnitude for equally "
        "well-spread (uniform random) geometry under a scale-invariant formula; a "
        "systematic multi-x gap here would mean the fix regressed toward scale-dependence")
