"""DW-FS-03 / GitHub Issue #73: ONE-SHOT confirmatory pass at N=100.

This is a bounded confirmatory derivative of the historical screening probe
`E_06_regional_sparse_calibration_probe.py` (worktree
DW_HILLY_REGIONAL_CAL_PROBE_20260913T194305Z, script sha256
2adbf206ba563d719d091b5c1d702251f070fc9d92f86341a89d089eeca92606). That probe's
own self-classified outcome was INCONCLUSIVE (positive on 2/5 comparable seeds
at N=50, degraded 2/4 known-good controls).

Pre-registration: GitHub Issue #73
internal tracker item 73
Frozen comment: internal tracker item 73
Preregistration timestamp: 2026-09-19T05:48:18Z

SCIENTIFIC-METHOD CHANGES FROM E_06 (the ONLY permitted differences):
  - N_VALUES = [100]                 (was [10, 50])
  - SEED_BASE = {100: 7000}          (was {10: 5000, 50: 6000}) - a new, previously
    unused dedicated RNG stream, frozen in the preregistration comment BEFORE this
    script was run.
  - Output path is a new, separate confirmatory evidence directory
    (10_DOCUMENTATION/evidence/DW_FS03_HILLY_CONFIRMATORY_N100/) - the historical
    probe's RESULTS.json/SUMMARY.md are never touched.
  - Added: preregistration/provenance metadata in the report, and a mechanical
    calculation of the frozen #73 stopping rule (STOPPING_RULE_PASS), written to
    DECISION.json alongside RESULTS.json.

EVERYTHING ELSE IS UNCHANGED FROM E_06: model loading, inference (predict_depth /
blend_tiles), scenes (Mount Adams + the same 4 Vaihingen controls), reference
sampling geometry (scene-wide uniform, gt lookup), holdout radius, the global
RANSAC affine estimator and its validity gate, the local K=5 / 512px-window /
no-blending/no-regularization candidate and its LOCAL_VALID -> GLOBAL_FALLBACK
-> RELATIVE_ONLY fallback chain, and all recorded metrics. No K/window/blending/
estimator/algorithm variant was tried. No parameter was tuned after seeing any
result. Exactly one scientific N=100 execution is permitted under #73.

WHAT THIS SCRIPT DOES NOT DO
  - It does not modify run_pipeline.py, src/preflight.py, or any A5 (#32) file.
  - It does not use dense ground truth for anything except (a) the declared
    simulated reference-point elevation lookup and (b) held-out evaluation -
    never inside the LOCAL candidate's fitting procedure itself.
  - It does not retrain the model or change the checkpoint.
  - It does not search over multiple K/window/blend/N choices.

REPRODUCE

    "python" \\
        PHASE2_RED_FLAGS/scripts/E_07_hilly_regional_calibration_confirmatory_n100.py
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

# --- frozen constants, inherited verbatim from E_06 / EXPERIMENT_SPEC.md, except
#     N_VALUES and SEED_BASE which are the #73 preregistered confirmatory values --
SCHEMA = "dw-fs03-hilly-confirmatory-n100/v1"
ISSUE_NUMBER = 73
ISSUE_URL = "internal tracker item 73"
PREREGISTRATION_COMMENT_URL = (
    "internal tracker item 73"
)
PREREGISTRATION_TIMESTAMP_UTC = "2026-09-19T05:48:18Z"
HISTORICAL_SCRIPT_SHA256 = "2adbf206ba563d719d091b5c1d702251f070fc9d92f86341a89d089eeca92606"
EXPECTED_BASE_SHA = "e073be25536efc298956b0e38173b42b9cc6f7c4"     # origin/main at worktree creation
EXPECTED_MOUNT_ADAMS_SHAPE = (4000, 2200)

CHECKPOINT_NAME = "da2_small_finetuned_iter2_partial_unfreeze.pt"
CHECKPOINT_PATH = ROOT / "checkpoints" / CHECKPOINT_NAME
AUTHORITATIVE_CHECKPOINT_SHA256 = (
    "74ccde40cf703095ed2adb783394d0dd2692b3ffaaf3b8a9e534e27feeba78d8"
)

SEEDS = [0, 1, 2, 3, 4]
N_VALUES = [100]
SEED_BASE = {100: 7000}            # NEW preregistered confirmatory RNG stream (#73)
K_LOCAL = 5
WINDOW = DEFAULT_WINDOW            # 512, matches production + E_06/E_05's own grid
HOLDOUT_RADIUS = 1                 # 3x3 neighborhood around each reference point
COVERAGE_THRESHOLD = 0.80          # frozen 80% local metric-coverage bar (#73)

OUT_DIR = P2ROOT.parent / "10_DOCUMENTATION" / "evidence" / "DW_FS03_HILLY_CONFIRMATORY_N100"
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


def grid_windows(h: int, w: int, win: int):
    """Identical to E_05/E_06's own grid_windows - non-overlapping window origins."""
    ys = list(range(0, h - win + 1, win))
    xs = list(range(0, w - win + 1, win))
    return [(y, x) for y in ys for x in xs]


def fit_scale_shift_ransac(x: np.ndarray, y: np.ndarray, seed: int):
    """Line-for-line mirror of run_pipeline.py::fit_scale_shift_ransac (PR #34, merged into
    main) - reimplemented, not imported, so this experiment worktree never depends on the
    A5 tree. Used for BOTH the global fit (all N points) and each local fit (K points).
    Unchanged from E_06."""
    n = int(len(y))
    try:
        reg = RANSACRegressor(LinearRegression(), min_samples=max(2, n // 2), random_state=seed)
        reg.fit(x.reshape(-1, 1), y)
        a = float(reg.estimator_.coef_[0]); b = float(reg.estimator_.intercept_)
        return {"method": "ransac", "a": a, "b": b, "n_points": n}
    except Exception as exc:                                        # noqa: BLE001
        A = np.vstack([x, np.ones_like(x)]).T
        (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
        return {"method": f"ols_fallback ({type(exc).__name__})", "a": float(a), "b": float(b),
                "n_points": n}


def fit_valid(a: float, b: float) -> bool:
    """Identical to run_pipeline.py::calibrate()'s existing metric-fit acceptance gate."""
    return bool(np.isfinite(a) and np.isfinite(b) and a > 0)


def sample_references(gt: np.ndarray, h: int, w: int, n: int, seed: int):
    """SIMULATED CALIBRATION CONTROLS SAMPLED FROM EVALUATION DSM - scene-wide uniform
    random, NOT difficulty/slope/relief-stratified, NOT field GCPs. Dedicated RNG stream
    (SEED_BASE[n] + seed = 7000 + seed for this confirmatory N=100 pass), distinct from
    E_06's 5000+seed (N=10) and 6000+seed (N=50) screening streams."""
    rng = np.random.default_rng(SEED_BASE[n] + seed)
    rr = rng.integers(0, h, size=n)
    cc = rng.integers(0, w, size=n)
    elev = gt[rr, cc].astype(np.float64)
    return rr, cc, elev


def holdout_mask(h: int, w: int, rr: np.ndarray, cc: np.ndarray, radius: int) -> np.ndarray:
    """True where a pixel is held out (excluded from evaluation): the exact reference
    pixels plus a fixed (2*radius+1)^2 neighborhood around each."""
    mask = np.zeros((h, w), dtype=bool)
    for r, c in zip(rr, cc):
        r0, r1 = max(0, r - radius), min(h, r + radius + 1)
        c0, c1 = max(0, c - radius), min(w, c + radius + 1)
        mask[r0:r1, c0:c1] = True
    return mask


def local_calibrated_surface(surface: np.ndarray, rr, cc, elev,
                             *, seed: int, global_a: float, global_b: float,
                             global_valid: bool):
    """Design A: Local K-Nearest-Reference Affine Fit, one per non-overlapping WINDOW-grid
    cell. Returns (z_local, window_states) where window_states is a list of per-window
    dicts with state in {LOCAL_VALID, GLOBAL_FALLBACK, RELATIVE_ONLY}. Unchanged from E_06."""
    h, w = surface.shape
    z = np.full((h, w), np.nan, dtype=np.float64)
    ref_xy = np.stack([rr.astype(np.float64), cc.astype(np.float64)], axis=1)
    window_states = []
    for (y0, x0) in grid_windows(h, w, WINDOW):
        cy, cx = y0 + WINDOW / 2.0, x0 + WINDOW / 2.0
        d2 = (ref_xy[:, 0] - cy) ** 2 + (ref_xy[:, 1] - cx) ** 2
        nn_idx = np.argsort(d2, kind="stable")[:K_LOCAL]
        x_local = surface[rr[nn_idx], cc[nn_idx]].astype(np.float64)
        y_local = elev[nn_idx]
        fit = fit_scale_shift_ransac(x_local, y_local, seed)
        a_w, b_w = fit["a"], fit["b"]
        local_ok = fit_valid(a_w, b_w)
        sl = (slice(y0, y0 + WINDOW), slice(x0, x0 + WINDOW))
        if local_ok:
            state = "LOCAL_VALID"
            z[sl] = a_w * surface[sl].astype(np.float64) + b_w
        elif global_valid:
            state = "GLOBAL_FALLBACK"
            z[sl] = global_a * surface[sl].astype(np.float64) + global_b
        else:
            state = "RELATIVE_ONLY"
            # z stays NaN for this window - excluded from metrics, counted separately
        window_states.append({"origin": [y0, x0], "state": state,
                              "local_a": a_w, "local_b": b_w,
                              "n_neighbors": int(len(nn_idx))})
    return z, window_states


def evaluate(z: np.ndarray, gt: np.ndarray, eval_mask: np.ndarray) -> dict:
    """eval_mask: True = evaluate this pixel. NaN entries in z (RELATIVE_ONLY windows) are
    further excluded and counted separately. Unchanged from E_06."""
    scoreable = eval_mask & np.isfinite(z)
    n_eval = int(scoreable.sum())
    n_unscoreable_in_mask = int((eval_mask & ~np.isfinite(z)).sum())
    if n_eval == 0:
        return {"n_eval_px": 0, "n_unscoreable_in_mask": n_unscoreable_in_mask,
                "metrics": None}
    m = all_metrics(z[scoreable], gt[scoreable].astype(np.float64))
    return {"n_eval_px": n_eval, "n_unscoreable_in_mask": n_unscoreable_in_mask,
            "metrics": m}


def run_scene(name: str, role: str, rgb: np.ndarray, gt: np.ndarray, model, device: str,
             results: list, timings: dict):
    assert rgb.shape[:2] == gt.shape, f"{name}: rgb {rgb.shape} vs gt {gt.shape} shape mismatch"
    h, w = rgb.shape[:2]
    print(f"[{name}] {h}x{w} role={role}")

    t0 = time.time()
    plan = plan_windows(h, w, window=DEFAULT_WINDOW, stride=DEFAULT_STRIDE)
    with torch.no_grad():
        depths, _prov = predict_depth(rgb, plan, model=model, device=device)
        surface = blend_tiles(depths, plan)   # matches run_pipeline.py: calibrate() runs on
                                              # the raw blended surface, before normalize_relative
    timings[f"{name}_predict_blend_s"] = time.time() - t0

    valid_eval_all = np.isfinite(gt)

    for n in N_VALUES:
        for seed in SEEDS:
            t1 = time.time()
            rr, cc, elev = sample_references(gt, h, w, n, seed)
            hold = holdout_mask(h, w, rr, cc, HOLDOUT_RADIUS)
            eval_mask = valid_eval_all & ~hold

            x_all = surface[rr, cc].astype(np.float64)
            gfit = fit_scale_shift_ransac(x_all, elev, seed)
            g_a, g_b = gfit["a"], gfit["b"]
            g_valid = fit_valid(g_a, g_b)
            z_global = (g_a * surface.astype(np.float64) + g_b) if g_valid else \
                       np.full((h, w), np.nan, dtype=np.float64)
            global_eval = evaluate(z_global, gt, eval_mask)

            z_local, window_states = local_calibrated_surface(
                surface, rr, cc, elev, seed=seed,
                global_a=g_a, global_b=g_b, global_valid=g_valid)
            local_eval = evaluate(z_local, gt, eval_mask)

            n_windows = len(window_states)
            n_local_valid = sum(1 for ws in window_states if ws["state"] == "LOCAL_VALID")
            n_global_fallback = sum(1 for ws in window_states if ws["state"] == "GLOBAL_FALLBACK")
            n_relative_only = sum(1 for ws in window_states if ws["state"] == "RELATIVE_ONLY")

            results.append({
                "scene": name, "role": role, "N": n, "seed": seed,
                "reference_provenance": "SIMULATED CALIBRATION CONTROLS SAMPLED FROM EVALUATION DSM",
                "reference_points": {"row": rr.tolist(), "col": cc.tolist(),
                                     "elevation_m": elev.tolist()},
                "global": {"a": g_a, "b": g_b, "fit_valid": g_valid,
                          "method": gfit["method"], **global_eval},
                "local": {"K": K_LOCAL, "window": WINDOW,
                         "n_windows": n_windows,
                         "n_local_valid": n_local_valid,
                         "n_global_fallback": n_global_fallback,
                         "n_relative_only": n_relative_only,
                         "local_valid_coverage_frac": n_local_valid / n_windows if n_windows else None,
                         "window_states": window_states,
                         **local_eval},
                "n_eval_px_mask_before_scoreable_filter": int(eval_mask.sum()),
                "runtime_s": time.time() - t1,
            })
            gr = global_eval['metrics']['rmse'] if global_eval['metrics'] else None
            lr = local_eval['metrics']['rmse'] if local_eval['metrics'] else None
            print(f"  {name} N={n} seed={seed}: global_valid={g_valid} "
                 f"global_rmse={gr} "
                 f"local_valid_cov={n_local_valid}/{n_windows} "
                 f"local_rmse={lr}")


def local_metric_coverage(run: dict) -> float:
    """Fraction of eligible (post-holdout, valid) pixels actually scored by LOCAL,
    i.e. n_eval_px / (n_eval_px + n_unscoreable_in_mask). This is the frozen #73
    'local metric coverage' quantity used by both the per-seed positive rule and the
    control degradation rule."""
    loc = run["local"]
    denom = loc["n_eval_px"] + loc["n_unscoreable_in_mask"]
    if denom == 0:
        return 0.0
    return loc["n_eval_px"] / denom


def compute_decision(results: list) -> dict:
    """Mechanical, non-editorializing application of the frozen #73 stopping rule.
    Operates only on raw per-run values already written to RESULTS.json; makes no
    subjective judgment."""
    mount_adams_runs = [r for r in results if r["scene"] == "mount_adams"]
    assert len(mount_adams_runs) == len(SEEDS), (
        f"expected {len(SEEDS)} Mount Adams runs, found {len(mount_adams_runs)}")

    mount_adams_table = []
    positive_seed_count = 0
    for r in sorted(mount_adams_runs, key=lambda x: x["seed"]):
        g_rmse = r["global"]["metrics"]["rmse"] if r["global"]["metrics"] else None
        l_rmse = r["local"]["metrics"]["rmse"] if r["local"]["metrics"] else None
        coverage = local_metric_coverage(r)
        delta = (g_rmse - l_rmse) if (g_rmse is not None and l_rmse is not None) else None
        positive = bool(delta is not None and delta > 0 and coverage >= COVERAGE_THRESHOLD)
        if positive:
            positive_seed_count += 1
        mount_adams_table.append({
            "seed": r["seed"], "global_rmse": g_rmse, "local_rmse": l_rmse,
            "rmse_delta": delta, "local_metric_coverage": coverage,
            "positive_mount_adams_seed": positive,
        })

    control_table = []
    degraded_control_count = 0
    for area in VAIHINGEN_CONTROLS:
        scene = f"vaihingen_area{area}"
        runs = sorted([r for r in results if r["scene"] == scene], key=lambda x: x["seed"])
        assert len(runs) == len(SEEDS), f"expected {len(SEEDS)} runs for {scene}, found {len(runs)}"
        g_rmses = [r["global"]["metrics"]["rmse"] for r in runs if r["global"]["metrics"]]
        l_rmses = [r["local"]["metrics"]["rmse"] for r in runs if r["local"]["metrics"]]
        coverages = [local_metric_coverage(r) for r in runs]
        median_g = float(np.median(g_rmses)) if g_rmses else None
        median_l = float(np.median(l_rmses)) if l_rmses else None
        median_cov = float(np.median(coverages)) if coverages else 0.0
        degraded = bool(
            (median_l is not None and median_g is not None and median_l > median_g)
            or median_cov < COVERAGE_THRESHOLD
        )
        if degraded:
            degraded_control_count += 1
        control_table.append({
            "scene": scene, "median_global_rmse": median_g, "median_local_rmse": median_l,
            "median_local_metric_coverage": median_cov, "degraded": degraded,
        })

    stopping_rule_pass = bool(positive_seed_count >= 4 and degraded_control_count == 0)

    return {
        "coverage_threshold": COVERAGE_THRESHOLD,
        "mount_adams_table": mount_adams_table,
        "positive_seed_count": positive_seed_count,
        "positive_seed_count_required": 4,
        "control_table": control_table,
        "degraded_control_count": degraded_control_count,
        "degraded_control_count_allowed": 0,
        "STOPPING_RULE_PASS": stopping_rule_pass,
    }


def main():
    # Fail-closed no-clobber guard: #73 permits exactly ONE scientific N=100
    # execution. Never silently overwrite a prior completed result.
    if RESULTS_PATH.exists() or DECISION_PATH.exists():
        raise RuntimeError(
            f"Refusing to run: confirmatory evidence already exists "
            f"({RESULTS_PATH} exists={RESULTS_PATH.exists()}, "
            f"{DECISION_PATH} exists={DECISION_PATH.exists()}). "
            f"Exactly one scientific N=100 execution is permitted under #73; "
            f"a bug-correction rerun requires explicit human review of the "
            f"prior failure first, not an automatic overwrite.")

    prov = git_provenance()
    assert prov["head"] == EXPECTED_BASE_SHA, (
        f"base SHA mismatch: worktree HEAD is {prov['head']}, "
        f"expected {EXPECTED_BASE_SHA} (frozen in PRE_REGISTRATION.md)")

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
        f"expected {EXPECTED_MOUNT_ADAMS_SHAPE} (frozen in PRE_REGISTRATION.md)")
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
        "preregistration_comment_url": PREREGISTRATION_COMMENT_URL,
        "preregistration_timestamp_utc": PREREGISTRATION_TIMESTAMP_UTC,
        "run_start_utc": start_time_utc,
        "run_end_utc": end_time_utc,
        "git": git_provenance(),
        "experiment_script": {"path": str(THIS_SCRIPT.relative_to(ROOT)).replace("\\", "/"),
                              "sha256": sha256_file(THIS_SCRIPT)},
        "historical_script_sha256": HISTORICAL_SCRIPT_SHA256,
        "checkpoint": ckpt_identity,
        "device": device,
        "frozen_hyperparameters": {"K_LOCAL": K_LOCAL, "WINDOW": WINDOW,
                                   "HOLDOUT_RADIUS": HOLDOUT_RADIUS,
                                   "SEEDS": SEEDS, "N_VALUES": N_VALUES,
                                   "SEED_BASE": SEED_BASE,
                                   "COVERAGE_THRESHOLD": COVERAGE_THRESHOLD},
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
    print(f"\nSTOPPING_RULE_PASS = {decision['STOPPING_RULE_PASS']}")
    print(f"positive_seed_count = {decision['positive_seed_count']}/5")
    print(f"degraded_control_count = {decision['degraded_control_count']}/4")


if __name__ == "__main__":
    main()
