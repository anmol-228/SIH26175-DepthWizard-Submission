"""DW-FS Sep23 / GitHub Issue #91, Track A: bounded robust/constrained calibration refit.

Pre-registered contract: docs/final_sprint/phase2/HILLY_ACCURACY_RESCUE_EXPERIMENT_CONTRACT.md
(this repo, origin/main @ 0c0b9c58836c7cfc336850e82ab5e436a90ce23f, from PR #90).
Issue: internal tracker item 91

WHAT THIS TESTS
    The existing global RANSAC fit (scripts/07_calibrate_and_eval.py /
    PHASE2_RED_FLAGS/scripts/E_07_hilly_regional_calibration_confirmatory_n100.py's
    fit_scale_shift_ransac) went sign-invalid (a<0, physically meaningless negative
    scale) on all 5 Mount Adams seeds at N=100 (#73's own frozen confirmatory
    evidence, 10_DOCUMENTATION/evidence/DW_FS03_HILLY_CONFIRMATORY_N100/RESULTS.json).
    This script tests whether (1) enforcing a positive-scale constraint DURING
    RANSAC's own consensus search (via sklearn's is_model_valid callback, not just
    checking the sign of the winning model after the fact), plus (2) an
    iteratively-reweighted (Huber) refinement pass on the resulting inlier set,
    recovers a valid, accurate global fit on the same seeds/data.

WHAT THIS DOES NOT DO
    - Does not modify #73's own frozen evidence (RESULTS.json/DECISION.json) - this
      is a wholly separate script/output directory.
    - Does not modify run_pipeline.py, src/preflight.py, or the shipped Track B
      gate - this is an experiment-only, additive fitting variant.
    - Does not perform the local/K=5/window fallback chain - Track A's contract
      targets only the global fit (the mechanism the hypothesis is about); the
      local/window design is #73's own already-frozen, separate question.
    - Does not retrain the model or touch the checkpoint.
    - Does not tune the constrained-fit hyperparameters (Huber delta, max IRLS
      iterations) after seeing any result - both are fixed before this script runs
      (see HUBER_DELTA / HUBER_MAX_ITER below) and are not touched during
      execution.
    - Runs exactly once against the predeclared seeds/scenes; no seed-shopping.

REPRODUCE

    "<worktree>" \\
        PHASE2_RED_FLAGS/scripts/E_08_track_a_constrained_calibration_refit.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tifffile
import torch

P2ROOT = Path(__file__).resolve().parents[1]
ROOT = P2ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from da2_utils import load_model                                   # noqa: E402
from metrics import all_metrics                                     # noqa: E402
from model_stages import blend_tiles, predict_depth                 # noqa: E402
from tiling import DEFAULT_STRIDE, DEFAULT_WINDOW, plan_windows     # noqa: E402
from sklearn.linear_model import LinearRegression, RANSACRegressor  # noqa: E402

# --- frozen constants, identical to #73's confirmatory pass for direct comparability ---
SCHEMA = "dw-fs-issue91-track-a-constrained-refit/v1"
ISSUE_NUMBER = 91
ISSUE_URL = "internal tracker item 91"
CONTRACT_PATH = "docs/final_sprint/phase2/HILLY_ACCURACY_RESCUE_EXPERIMENT_CONTRACT.md"
PREREGISTRATION_TIMESTAMP_UTC = "2026-09-23T15:51:11Z"
EXPECTED_BASE_SHA = "0c0b9c58836c7cfc336850e82ab5e436a90ce23f"    # origin/main, PR #90 merge
EXPECTED_MOUNT_ADAMS_SHAPE = (4000, 2200)

CHECKPOINT_NAME = "da2_small_finetuned_iter2_partial_unfreeze.pt"
CHECKPOINT_PATH = ROOT / "checkpoints" / CHECKPOINT_NAME
AUTHORITATIVE_CHECKPOINT_SHA256 = (
    "74ccde40cf703095ed2adb783394d0dd2692b3ffaaf3b8a9e534e27feeba78d8"
)

SEEDS = [0, 1, 2, 3, 4]              # identical to #73
N_VALUES = [100]                     # identical to #73
SEED_BASE = {100: 7000}              # identical to #73's confirmatory stream (same references)

# Constrained-fit hyperparameters - fixed before execution, not tuned after results.
HUBER_DELTA = 1.345                  # standard Huber threshold, in residual-std units
HUBER_MAX_ITER = 10
RANSAC_MAX_TRIALS = 1000             # allow more search room than the sklearn default (100)
                                      # so the positive-scale constraint has a fair chance to
                                      # find a valid consensus set; not a result-driven change,
                                      # fixed before any run.

# Predeclared gate (frozen, from the contract - do not edit after seeing results)
VALID_FIT_REQUIREMENT = 4            # of 5 Mount Adams seeds
HILLY_ORACLE_RMSE_M = 11.5           # midpoint of contract's "~11-12m" reference figure
HILLY_RMSE_MARGIN_MULTIPLIER = 2.0   # contract: "worse than ~22-24m fails"
CONTROL_RMSE_REGRESSION_LIMIT_M = 1.0

OUT_DIR = P2ROOT.parent / "10_DOCUMENTATION" / "evidence" / "DW_ISSUE91_TRACK_A_CONSTRAINED_REFIT"
RESULTS_PATH = OUT_DIR / "RESULTS.json"
DECISION_PATH = OUT_DIR / "DECISION.json"
THIS_SCRIPT = Path(__file__).resolve()

VAIHINGEN_CONTROLS = ["6", "8", "34", "38"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_identity(path: Path) -> dict:
    path = Path(path)
    return {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception as exc:                                        # noqa: BLE001
        return f"unknown ({exc})"


def git_provenance() -> dict:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    porcelain = git("status", "--porcelain=v1", "--untracked-files=all")
    lines = [l for l in porcelain.splitlines() if l.strip()]
    return {"branch": branch, "head": head,
            "tree_state": "clean" if not lines else "dirty",
            "status_lines": lines}


def fit_scale_shift_ransac_baseline(x: np.ndarray, y: np.ndarray, seed: int):
    """UNCONSTRAINED baseline - line-for-line identical to #73 / run_pipeline.py's
    fit_scale_shift_ransac. Used as this script's own baseline row for direct
    within-script comparison (same code, same data, run back-to-back)."""
    n = int(len(y))
    try:
        reg = RANSACRegressor(LinearRegression(), min_samples=max(2, n // 2), random_state=seed)
        reg.fit(x.reshape(-1, 1), y)
        a = float(reg.estimator_.coef_[0]); b = float(reg.estimator_.intercept_)
        return {"method": "ransac_unconstrained_baseline", "a": a, "b": b, "n_points": n}
    except Exception as exc:                                        # noqa: BLE001
        A = np.vstack([x, np.ones_like(x)]).T
        (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
        return {"method": f"ols_fallback ({type(exc).__name__})", "a": float(a), "b": float(b),
                "n_points": n}


def fit_scale_shift_constrained(x: np.ndarray, y: np.ndarray, seed: int):
    """TRACK A CANDIDATE: positive-scale-constrained RANSAC consensus search
    (rejects any candidate model with a<=0 via sklearn's is_model_valid callback,
    DURING the search, not only after) followed by a Huber-IRLS refinement pass on
    the resulting inlier set. A refinement step is rejected if it would flip a to
    a<=0 - the refinement must not be able to silently launder an invalid result
    into an apparently-valid one. NOTE (found by independent automated (AI) gate audit,
    2026-09-23): the is_model_valid callback constrains sklearn's internal search,
    but does not guarantee reg.estimator_ (the final fitted model on the winning
    inlier set) is itself positive - on Mount Adams seeds 3 and 4, RANSAC's final
    fit (a0/b0) was still negative despite the callback. The refinement loop's
    sign-flip guard then correctly refuses to accept a negative a_new, but rolls
    back to a0/b0, which was itself already negative, not to a genuinely-valid
    prior value. fit_valid()/the decision logic downstream already treat this
    correctly (fit_valid=False for those rows, matching DECISION.json) - this is a
    documentation-accuracy fix only, not a change to any computed result."""
    n = int(len(y))

    def is_model_valid(model, X_, y_):  # noqa: ANN001 - sklearn callback signature
        return float(model.coef_[0]) > 0

    try:
        reg = RANSACRegressor(
            LinearRegression(),
            min_samples=max(2, n // 2),
            random_state=seed,
            is_model_valid=is_model_valid,
            max_trials=RANSAC_MAX_TRIALS,
        )
        reg.fit(x.reshape(-1, 1), y)
        inlier_mask = reg.inlier_mask_
        a0 = float(reg.estimator_.coef_[0]); b0 = float(reg.estimator_.intercept_)
        method = "ransac_positive_scale_constrained"
    except Exception as exc:                                        # noqa: BLE001
        # No valid (a>0) consensus set found within RANSAC_MAX_TRIALS - this is
        # itself a real, informative outcome, not silently papered over: fall back
        # to unconstrained OLS on all points (guaranteed to produce *a* fit, but not
        # forced positive), and record the exception that caused the fallback.
        A = np.vstack([x, np.ones_like(x)]).T
        (a0, b0), *_ = np.linalg.lstsq(A, y, rcond=None)
        inlier_mask = np.ones_like(x, dtype=bool)
        method = f"ols_fallback_no_valid_ransac_consensus ({type(exc).__name__})"

    a, b = a0, b0
    xi, yi = x[inlier_mask], y[inlier_mask]
    n_refine_iters = 0
    refinement_rejected_sign_flip = False
    if len(xi) >= 2:
        for _ in range(HUBER_MAX_ITER):
            resid = yi - (a * xi + b)
            s = float(np.std(resid))
            s = s if s > 1e-9 else 1e-9
            u = np.abs(resid) / (HUBER_DELTA * s)
            w = np.where(u <= 1.0, 1.0, 1.0 / u)
            sw = np.sqrt(w)
            A = np.vstack([xi * sw, sw]).T
            try:
                (a_new, b_new), *_ = np.linalg.lstsq(A, yi * sw, rcond=None)
            except Exception:                                        # noqa: BLE001
                break
            n_refine_iters += 1
            if a_new <= 0:
                # Refinement must not be allowed to override the positive-scale
                # constraint the RANSAC search already enforced - keep prior a,b.
                refinement_rejected_sign_flip = True
                break
            converged = abs(a_new - a) < 1e-9 and abs(b_new - b) < 1e-9
            a, b = float(a_new), float(b_new)
            if converged:
                break
        method += "+huber_irls_refine"

    return {"method": method, "a": float(a), "b": float(b), "n_points": n,
            "n_inliers": int(inlier_mask.sum()), "n_refine_iters": n_refine_iters,
            "refinement_rejected_sign_flip": refinement_rejected_sign_flip,
            "initial_ransac_a": a0, "initial_ransac_b": b0}


def fit_valid(a: float, b: float) -> bool:
    """Identical to run_pipeline.py::calibrate()'s existing metric-fit acceptance gate."""
    return bool(np.isfinite(a) and np.isfinite(b) and a > 0)


def sample_references(gt: np.ndarray, h: int, w: int, n: int, seed: int):
    """Identical reference-point sampling to #73 (same RNG stream, same method) -
    for direct seed-by-seed comparability."""
    rng = np.random.default_rng(SEED_BASE[n] + seed)
    rr = rng.integers(0, h, size=n)
    cc = rng.integers(0, w, size=n)
    elev = gt[rr, cc].astype(np.float64)
    return rr, cc, elev


def holdout_mask(h: int, w: int, rr: np.ndarray, cc: np.ndarray, radius: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for r, c in zip(rr, cc):
        r0, r1 = max(0, r - radius), min(h, r + radius + 1)
        c0, c1 = max(0, c - radius), min(w, c + radius + 1)
        mask[r0:r1, c0:c1] = True
    return mask


def evaluate(z: np.ndarray, gt: np.ndarray, eval_mask: np.ndarray) -> dict:
    scoreable = eval_mask & np.isfinite(z)
    n_eval = int(scoreable.sum())
    if n_eval == 0:
        return {"n_eval_px": 0, "metrics": None}
    m = all_metrics(z[scoreable], gt[scoreable].astype(np.float64))
    return {"n_eval_px": n_eval, "metrics": m}


HOLDOUT_RADIUS = 1  # identical to #73


def run_scene(name: str, role: str, rgb: np.ndarray, gt: np.ndarray, model, device: str,
             results: list, timings: dict):
    assert rgb.shape[:2] == gt.shape, f"{name}: rgb {rgb.shape} vs gt {gt.shape} shape mismatch"
    h, w = rgb.shape[:2]
    print(f"[{name}] {h}x{w} role={role}")

    t0 = time.time()
    plan = plan_windows(h, w, window=DEFAULT_WINDOW, stride=DEFAULT_STRIDE)
    with torch.no_grad():
        depths, _prov = predict_depth(rgb, plan, model=model, device=device)
        surface = blend_tiles(depths, plan)
    timings[f"{name}_predict_blend_s"] = time.time() - t0

    valid_eval_all = np.isfinite(gt)

    for n in N_VALUES:
        for seed in SEEDS:
            t1 = time.time()
            rr, cc, elev = sample_references(gt, h, w, n, seed)
            hold = holdout_mask(h, w, rr, cc, HOLDOUT_RADIUS)
            eval_mask = valid_eval_all & ~hold
            x_all = surface[rr, cc].astype(np.float64)

            base_fit = fit_scale_shift_ransac_baseline(x_all, elev, seed)
            base_a, base_b = base_fit["a"], base_fit["b"]
            base_valid = fit_valid(base_a, base_b)
            z_base = (base_a * surface.astype(np.float64) + base_b) if base_valid else \
                     np.full((h, w), np.nan, dtype=np.float64)
            base_eval = evaluate(z_base, gt, eval_mask)

            cand_fit = fit_scale_shift_constrained(x_all, elev, seed)
            cand_a, cand_b = cand_fit["a"], cand_fit["b"]
            cand_valid = fit_valid(cand_a, cand_b)
            z_cand = (cand_a * surface.astype(np.float64) + cand_b) if cand_valid else \
                     np.full((h, w), np.nan, dtype=np.float64)
            cand_eval = evaluate(z_cand, gt, eval_mask)

            results.append({
                "scene": name, "role": role, "N": n, "seed": seed,
                "reference_provenance": "SIMULATED CALIBRATION CONTROLS SAMPLED FROM EVALUATION DSM",
                "n_reference_points": int(n),
                "baseline_unconstrained": {"a": base_a, "b": base_b, "fit_valid": base_valid,
                                           "method": base_fit["method"], **base_eval},
                "candidate_constrained": {"a": cand_a, "b": cand_b, "fit_valid": cand_valid,
                                          "method": cand_fit["method"],
                                          "n_inliers": cand_fit.get("n_inliers"),
                                          "n_refine_iters": cand_fit.get("n_refine_iters"),
                                          "refinement_rejected_sign_flip":
                                              cand_fit.get("refinement_rejected_sign_flip"),
                                          "initial_ransac_a": cand_fit.get("initial_ransac_a"),
                                          "initial_ransac_b": cand_fit.get("initial_ransac_b"),
                                          **cand_eval},
                "runtime_s": time.time() - t1,
            })
            br = base_eval["metrics"]["rmse"] if base_eval["metrics"] else None
            cr = cand_eval["metrics"]["rmse"] if cand_eval["metrics"] else None
            print(f"  {name} N={n} seed={seed}: baseline_valid={base_valid} rmse={br}  "
                 f"candidate_valid={cand_valid} rmse={cr}")


def compute_decision(results: list) -> dict:
    """Mechanical, non-editorializing application of the frozen Track A gate from
    the experiment contract. Operates only on raw per-run values already written to
    RESULTS.json."""
    mount_adams_runs = [r for r in results if r["scene"] == "mount_adams"]
    assert len(mount_adams_runs) == len(SEEDS), (
        f"expected {len(SEEDS)} Mount Adams runs, found {len(mount_adams_runs)}")

    hilly_table = []
    valid_seed_count = 0
    valid_rmses = []
    for r in sorted(mount_adams_runs, key=lambda x: x["seed"]):
        c = r["candidate_constrained"]
        valid = bool(c["fit_valid"])
        rmse = c["metrics"]["rmse"] if c["metrics"] else None
        if valid:
            valid_seed_count += 1
            if rmse is not None:
                valid_rmses.append(rmse)
        hilly_table.append({"seed": r["seed"], "fit_valid": valid, "rmse": rmse,
                            "a": c["a"], "b": c["b"],
                            "baseline_fit_valid": r["baseline_unconstrained"]["fit_valid"],
                            "baseline_rmse": (r["baseline_unconstrained"]["metrics"]["rmse"]
                                              if r["baseline_unconstrained"]["metrics"] else None)})

    max_valid_rmse = max(valid_rmses) if valid_rmses else None
    hilly_rmse_limit = HILLY_ORACLE_RMSE_M * HILLY_RMSE_MARGIN_MULTIPLIER
    hilly_rmse_within_margin = bool(max_valid_rmse is not None and max_valid_rmse <= hilly_rmse_limit)

    control_table = []
    control_regressed = False
    for area in VAIHINGEN_CONTROLS:
        scene = f"vaihingen_area{area}"
        runs = sorted([r for r in results if r["scene"] == scene], key=lambda x: x["seed"])
        assert len(runs) == len(SEEDS), f"expected {len(SEEDS)} runs for {scene}, found {len(runs)}"
        base_rmses = [r["baseline_unconstrained"]["metrics"]["rmse"] for r in runs
                      if r["baseline_unconstrained"]["metrics"]]
        cand_rmses = [r["candidate_constrained"]["metrics"]["rmse"] for r in runs
                     if r["candidate_constrained"]["metrics"]]
        cand_valid_count = sum(1 for r in runs if r["candidate_constrained"]["fit_valid"])
        base_valid_count = sum(1 for r in runs if r["baseline_unconstrained"]["fit_valid"])
        median_base = float(np.median(base_rmses)) if base_rmses else None
        median_cand = float(np.median(cand_rmses)) if cand_rmses else None
        delta = (median_cand - median_base) if (median_base is not None and median_cand is not None) else None
        regressed = bool(
            delta is not None and delta > CONTROL_RMSE_REGRESSION_LIMIT_M
            or cand_valid_count < base_valid_count
        )
        if regressed:
            control_regressed = True
        control_table.append({
            "scene": scene, "median_baseline_rmse": median_base, "median_candidate_rmse": median_cand,
            "rmse_delta": delta, "baseline_valid_count": base_valid_count,
            "candidate_valid_count": cand_valid_count, "regressed": regressed,
        })

    valid_fit_pass = bool(valid_seed_count >= VALID_FIT_REQUIREMENT)
    stopping_rule_pass = bool(valid_fit_pass and hilly_rmse_within_margin and not control_regressed)

    if stopping_rule_pass:
        classification = "PASS"
    elif valid_seed_count == 0 or not valid_rmses:
        classification = "FAIL"
    elif not valid_fit_pass or (valid_rmses and not hilly_rmse_within_margin) or control_regressed:
        classification = "FAIL"
    else:
        classification = "INCONCLUSIVE"

    return {
        "gate_source": CONTRACT_PATH,
        "valid_fit_requirement": VALID_FIT_REQUIREMENT,
        "hilly_oracle_rmse_reference_m": HILLY_ORACLE_RMSE_M,
        "hilly_rmse_margin_multiplier": HILLY_RMSE_MARGIN_MULTIPLIER,
        "hilly_rmse_limit_m": hilly_rmse_limit,
        "control_rmse_regression_limit_m": CONTROL_RMSE_REGRESSION_LIMIT_M,
        "hilly_table": hilly_table,
        "valid_seed_count": valid_seed_count,
        "max_valid_seed_rmse_m": max_valid_rmse,
        "hilly_rmse_within_margin": hilly_rmse_within_margin,
        "control_table": control_table,
        "control_regressed": control_regressed,
        "STOPPING_RULE_PASS": stopping_rule_pass,
        "CLASSIFICATION": classification,
    }


def main():
    if RESULTS_PATH.exists() or DECISION_PATH.exists():
        raise RuntimeError(
            f"Refusing to run: Track A evidence already exists "
            f"({RESULTS_PATH} exists={RESULTS_PATH.exists()}, "
            f"{DECISION_PATH} exists={DECISION_PATH.exists()}). "
            f"Per the pre-registration lock, exactly one scientific execution is "
            f"permitted; a rerun requires explicit human review first.")

    prov = git_provenance()
    assert prov["head"] == EXPECTED_BASE_SHA or prov["branch"] == "issue-91-hilly-rescue-sep23", (
        f"unexpected git state: HEAD={prov['head']} branch={prov['branch']}, "
        f"expected base {EXPECTED_BASE_SHA} on branch issue-91-hilly-rescue-sep23")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  checkpoint={CHECKPOINT_NAME}")
    ckpt_identity = file_identity(CHECKPOINT_PATH)
    assert ckpt_identity["sha256"] == AUTHORITATIVE_CHECKPOINT_SHA256, (
        f"checkpoint sha256 mismatch: got {ckpt_identity['sha256']}, "
        f"expected {AUTHORITATIVE_CHECKPOINT_SHA256}")
    model = load_model(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
    model.eval()

    start_time_utc = datetime.now(timezone.utc).isoformat()
    results: list = []
    timings: dict = {}
    input_identities = {}

    ma_dir = P2ROOT / "data" / "hilly_terrain" / "aoi"
    ma_rgb_path, ma_dem_path = ma_dir / "rgb_1m.npy", ma_dir / "dem_1m.npy"
    ma_rgb, ma_dem = np.load(ma_rgb_path), np.load(ma_dem_path)
    assert ma_dem.shape == EXPECTED_MOUNT_ADAMS_SHAPE, (
        f"Mount Adams DEM shape mismatch: got {ma_dem.shape}, "
        f"expected {EXPECTED_MOUNT_ADAMS_SHAPE}")
    input_identities["mount_adams"] = {"rgb": file_identity(ma_rgb_path),
                                       "elevation": file_identity(ma_dem_path)}
    run_scene("mount_adams", "known_failure", ma_rgb, ma_dem, model, device, results, timings)

    va_top_dir = ROOT / "data" / "raw" / "Vaihingen" / "top"
    va_dsm_dir = ROOT / "data" / "raw" / "Vaihingen" / "dsm"
    for area in VAIHINGEN_CONTROLS:
        key = f"vaihingen_area{area}"
        top_path = va_top_dir / f"top_mosaic_09cm_area{area}.tif"
        dsm_path = va_dsm_dir / f"dsm_09cm_matching_area{area}.tif"
        rgb, dsm = tifffile.imread(top_path), tifffile.imread(dsm_path)
        input_identities[key] = {"rgb": file_identity(top_path), "elevation": file_identity(dsm_path)}
        run_scene(key, "known_good", rgb, dsm, model, device, results, timings)

    end_time_utc = datetime.now(timezone.utc).isoformat()
    decision = compute_decision(results)

    report = {
        "schema": SCHEMA,
        "issue_number": ISSUE_NUMBER,
        "issue_url": ISSUE_URL,
        "contract_path": CONTRACT_PATH,
        "preregistration_timestamp_utc": PREREGISTRATION_TIMESTAMP_UTC,
        "run_start_utc": start_time_utc,
        "run_end_utc": end_time_utc,
        "git": git_provenance(),
        "experiment_script": {"path": str(THIS_SCRIPT.relative_to(ROOT)).replace("\\", "/"),
                              "sha256": sha256_file(THIS_SCRIPT)},
        "checkpoint": ckpt_identity,
        "device": device,
        "frozen_hyperparameters": {"SEEDS": SEEDS, "N_VALUES": N_VALUES, "SEED_BASE": SEED_BASE,
                                   "HUBER_DELTA": HUBER_DELTA, "HUBER_MAX_ITER": HUBER_MAX_ITER,
                                   "RANSAC_MAX_TRIALS": RANSAC_MAX_TRIALS,
                                   "HOLDOUT_RADIUS": HOLDOUT_RADIUS},
        "gate": {"VALID_FIT_REQUIREMENT": VALID_FIT_REQUIREMENT,
                "HILLY_ORACLE_RMSE_M": HILLY_ORACLE_RMSE_M,
                "HILLY_RMSE_MARGIN_MULTIPLIER": HILLY_RMSE_MARGIN_MULTIPLIER,
                "CONTROL_RMSE_REGRESSION_LIMIT_M": CONTROL_RMSE_REGRESSION_LIMIT_M},
        "input_identities": input_identities,
        "timings": timings,
        "runs": results,
        "decision": decision,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report, indent=2))
    DECISION_PATH.write_text(json.dumps(decision, indent=2))
    print(f"\nWrote {RESULTS_PATH}")
    print(f"Wrote {DECISION_PATH}")
    print(f"\nCLASSIFICATION = {decision['CLASSIFICATION']}")
    print(f"STOPPING_RULE_PASS = {decision['STOPPING_RULE_PASS']}")
    print(f"valid_seed_count = {decision['valid_seed_count']}/5")
    print(f"control_regressed = {decision['control_regressed']}")


if __name__ == "__main__":
    main()
