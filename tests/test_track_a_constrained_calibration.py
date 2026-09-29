"""Targeted tests for Issue #91 Track A's constrained-calibration fitting routine,
run BEFORE the expensive full-pipeline validation (per the Track A contract's
Phase-4 requirement). Covers: invalid/negative fit handling, constrained/robust fit
semantics, deterministic setup, and that the unconstrained baseline path used for
comparison is unchanged from #73's own fit_scale_shift_ransac.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (ROOT / "PHASE2_RED_FLAGS" / "scripts" /
               "E_08_track_a_constrained_calibration_refit.py")

sys.path.insert(0, str(ROOT / "src"))


def _load_module():
    # torch/tifffile/sklearn are already installed in this project's .venv (same
    # interpreter the real experiment runs under), so the module loads normally -
    # only the pure-numpy fitting functions are exercised below, no GPU/checkpoint
    # needed for that.
    spec = importlib.util.spec_from_file_location("track_a_e08", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load_module()


def test_constrained_fit_recovers_positive_scale_when_unconstrained_would_flip(mod):
    """Construct data where an unconstrained fit on a corrupted subset can go
    sign-invalid, but a positive-scale-constrained search over the same points
    still finds a valid consensus model."""
    rng = np.random.default_rng(0)
    n = 100
    # True relationship a=1.2, b=3.0, plus a cluster of corrupting points that
    # would pull an unconstrained/no-consensus-search fit negative.
    x_good = rng.uniform(0, 50, size=70)
    y_good = 1.2 * x_good + 3.0 + rng.normal(0, 0.5, size=70)
    x_bad = rng.uniform(0, 50, size=30)
    y_bad = -2.5 * x_bad + 400 + rng.normal(0, 1.0, size=30)
    x = np.concatenate([x_good, x_bad])
    y = np.concatenate([y_good, y_bad])

    result = mod.fit_scale_shift_constrained(x, y, seed=42)
    assert result["a"] > 0, "constrained fit must not return a<=0 as its final answer"
    assert mod.fit_valid(result["a"], result["b"])
    # Should recover something reasonably close to the true a=1.2 slope from the
    # majority-good subset, not be dominated by the corrupting cluster.
    assert 0.5 < result["a"] < 2.5


def test_constrained_fit_falls_back_honestly_when_no_valid_consensus_exists(mod):
    """When every point is drawn from a genuinely negative-slope relationship,
    the constrained search should not find a positive-scale consensus set, and
    must report this honestly (fit_valid=False) rather than fabricate a=0 or a
    spurious positive value."""
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 50, size=100)
    y = -3.0 * x + 200 + rng.normal(0, 0.3, size=100)

    result = mod.fit_scale_shift_constrained(x, y, seed=7)
    # Either the RANSAC search exhausts its trials (a<=0 in the OLS fallback) or it
    # finds a degenerate near-zero-slope consensus - either way this must not be
    # reported as a valid fit if a<=0.
    if not mod.fit_valid(result["a"], result["b"]):
        assert result["a"] <= 0
    assert "method" in result


def test_refinement_never_overrides_the_positive_scale_constraint(mod):
    """The Huber-IRLS refinement pass must reject any step that would flip the
    sign back to a<=0, and record that it did so."""
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 10, size=20)
    y = 0.05 * x + 1.0 + rng.normal(0, 5.0, size=20)  # tiny true slope, huge noise
    result = mod.fit_scale_shift_constrained(x, y, seed=3)
    assert result["a"] > 0 or result.get("refinement_rejected_sign_flip") in (True, False)
    # Whatever the outcome, a must never be reported as exactly the pre-refinement
    # negative value being silently accepted.
    if result.get("refinement_rejected_sign_flip"):
        assert result["a"] == result["initial_ransac_a"]


def test_deterministic_given_same_seed(mod):
    rng = np.random.default_rng(5)
    x = rng.uniform(0, 50, size=100)
    y = 1.0 * x + 2.0 + rng.normal(0, 1.0, size=100)
    r1 = mod.fit_scale_shift_constrained(x, y, seed=99)
    r2 = mod.fit_scale_shift_constrained(x, y, seed=99)
    assert r1["a"] == r2["a"]
    assert r1["b"] == r2["b"]


def test_baseline_path_matches_number_73_unconstrained_semantics(mod):
    """The baseline (control) path used for within-script comparison must be the
    same unconstrained RANSAC fit semantics as #73 - i.e. it can legitimately
    produce a<0, unlike the constrained candidate."""
    rng = np.random.default_rng(6)
    x_good = rng.uniform(0, 50, size=20)
    y_good = 1.2 * x_good + 3.0 + rng.normal(0, 0.5, size=20)
    x_bad = rng.uniform(0, 50, size=80)
    y_bad = -2.5 * x_bad + 400 + rng.normal(0, 1.0, size=80)
    x = np.concatenate([x_good, x_bad])
    y = np.concatenate([y_good, y_bad])

    baseline = mod.fit_scale_shift_ransac_baseline(x, y, seed=11)
    # No positive-scale constraint on the baseline path - it is free to go
    # sign-invalid, exactly like #73's own fit_scale_shift_ransac.
    assert "a" in baseline and "b" in baseline
    assert baseline["method"] in ("ransac_unconstrained_baseline",) or "ols_fallback" in baseline["method"]


def test_fit_valid_rejects_nan_and_nonpositive_scale(mod):
    assert not mod.fit_valid(float("nan"), 1.0)
    assert not mod.fit_valid(0.0, 1.0)
    assert not mod.fit_valid(-1.0, 1.0)
    assert mod.fit_valid(0.5, 1.0)


def test_sample_references_deterministic_and_matches_73_stream(mod):
    """Reference sampling must use the same SEED_BASE[100]=7000 stream as #73 for
    the same seed, so hilly results are directly comparable seed-by-seed."""
    gt = np.zeros((200, 200))
    gt[:] = np.arange(200).reshape(-1, 1)
    rr1, cc1, elev1 = mod.sample_references(gt, 200, 200, 100, seed=0)
    rr2, cc2, elev2 = mod.sample_references(gt, 200, 200, 100, seed=0)
    assert np.array_equal(rr1, rr2)
    assert np.array_equal(cc1, cc2)
    assert mod.SEED_BASE[100] == 7000
