"""Experiment E, step 5: scene-consistent runtime hilly-risk signal audit -- Issue #32
(DW-S1-11), superseding the exploratory cross-area draw run outside version control.

WHY THIS SCRIPT EXISTS

Issue #32 asks whether a runtime-computable statistic can flag "this scene is unsafe for
one global scale+shift calibration" before it happens (07_HILLY_TERRAIN_TEST.md's Mount
Adams finding: whole-scene oracle calibration collapses to 60-70m RMSE / 0.05-0.24 pooled
correlation, vs 4.40-5.50m for Vaihingen's known-good regime -- see the DW_S1_11 methodology
doc, section 5, for why those two RMSE numbers must never be averaged together).

A first exploratory pass (not committed) compared Mount Adams against 28 Vaihingen patches
drawn at random across 9 different ISPRS "areas" -- not a scene-consistent control, since
Mount Adams is one contiguous 8.8km^2 AOI and a cross-area draw mixes unrelated scenes.
That result is EXPLORATORY / SUPERSEDED. This script fixes two things at once:

  1. Scene consistency: every comparison group is now ONE contiguous real scene --
     Mount Adams's full AOI raster, and each of four individual Vaihingen ISPRS "areas"
     (each area is one contiguous orthophoto+DSM tile, not a patch mosaic), selected from
     a full per-area audit of the 132-patch test split covering the observed heterogeneity
     range: area 38 (n=28 test patches, matches Mount Adams's sample size, highest
     cross-patch relief ratio among test-split areas), area 6 (highest absolute full-scene
     relief among test-split areas), area 34 (lowest cross-patch ratio, most "ordinary"),
     area 8 (second "ordinary" case). Exact regenerated per-scene numbers are in this run's
     own JSON output, not restated here as prose to avoid drift between the docstring and
     the data. Vaihingen's own terrain is documented flat-to-gently-rolling and its
     full-scene relief across all four sampled areas is far below Mount Adams's -- a real
     data-availability limit (no genuinely "moderately hilly" scene exists in preserved
     data), not an oversight.

  2. Runtime-representation parity: the exploratory pass measured independent, non-
     overlapping 512x512 *data-prep* patches (from E_02_make_patches.py's grid split) --
     not what run_pipeline.py actually produces. This script instead runs the real
     production functions unmodified (`src.tiling.plan_windows`, `src.model_stages.
     predict_depth`, `src.model_stages.blend_tiles`, `src.relative_dsm.normalize_relative`)
     directly on each scene's full contiguous raster, at the runner's own defaults
     (window=512, stride=384), and evaluates the candidate statistic separately on:
       - RAW_TILE   : the independent per-window predictions predict_depth() returns,
                      before any blending -- exactly what the runner has immediately after
                      stage 3, tile-by-tile.
       - BLENDED    : blend_tiles()'s feather-blended full-scene surface, re-tiled into a
                      NON-overlapping window grid for a comparable per-window statistic --
                      what the runner has after stage 4, before calibration.
       - NORMALIZED : relative_dsm.normalize_relative()'s [0,1] percentile-clipped output
                      on the same non-overlapping grid -- what actually gets written to
                      `height_relative.npy`.
     Ground-truth relief per window is also recorded, but ONLY as diagnostic context --
     never as the basis for a threshold (a real deployment has no dense truth).

CORRECTION PASS (this revision)

Independent review of the previous reviewed/uncommitted revision of this script found the sparse-fit
"elevation-range-normalized residual" (signal B2) was normalized by the FULL-SCENE dense
ground-truth range (`gt.max() - gt.min()`, thousands of pixels), not by the range of the
10 sampled reference points actually used in the fit. That made B2 an oracle/offline
diagnostic, not the runtime-computable statistic the methodology doc claimed. Fixed here:
each seed's normalization denominator is now `y.max() - y.min()` over that seed's own 10
sampled points -- the range a real `--reference` file's own elevations would give a
runtime detector, nothing else. See `sparse10_diagnostics()` below.

This revision also adds: (a) `fit_valid_for_metric` / `existing_fit_rejection_reason` on
every sparse-fit record, so a seed PR #34's own `a>0` metric-fit gate would already reject
is never counted as a "false negative" of a hypothetical additional hilly-risk detector;
(b) explicit sparse-reference-provenance language (the 10 points are simulated controls
sampled from each scene's evaluation DSM, not independent field GCPs); (c) hardened
provenance -- this script's own sha256, the git branch queried dynamically (not
hardcoded), working-tree dirty state at run time, and a path+size+sha256 identity for
every RGB/DSM/DEM input file actually read.

WHAT THIS SCRIPT DOES NOT DO

  - It does not implement or modify `src/preflight.py`, `run_pipeline.py`, or any viewer
    code. `fit_scale_shift_ransac` below is a deliberate line-for-line mirror of the
    estimator `run_pipeline.py::calibrate()` uses (landed via PR #34, merged into `main`
    as commit 0e6cf13 -- after this evidence branch was already cut from `dd7d516`; this
    branch deliberately does not rebase mid-investigation) -- reimplemented here rather
    than imported, so this evidence-only branch never depends on that tree.
  - It does not invent a threshold. It records distributions; threshold justification (or
    its absence) is decided in DW_S1_11_HILLY_RISK_METHODOLOGY.md from this script's
    output, not in code. The one numerical epsilon below (`EPS_DIVISION_GUARD`) exists
    solely to avoid a Python float division by (near-)zero and is explicitly not a
    methodological cutoff -- a seed whose sampled range trips it is marked
    `insufficient_evidence`, not scored as pass or fail.
  - It does not touch `outputs/run_pipeline/` or any existing metrics file. Output goes to
    `PHASE2_RED_FLAGS/outputs/metrics/E_hilly_risk_signal_experiment.json` (new file).

SPARSE-REFERENCE PROVENANCE (read before citing signal B1/B2 anywhere)

The scale/shift procedure itself (RANSAC on a handful of reference elevations) is
*deployable given genuine external reference elevations* -- it is exactly what
`run_pipeline.py --reference` expects a real survey/GCP/altimetry file to supply.
"Deployable" describes that algorithm/interface conditionally, never the provenance of
the specific values used below. The 10 (row, col) points sampled per seed in THIS
experiment are drawn uniformly at random and their "elevation" is read directly from
`gt[rr, cc]` -- i.e. from the same evaluation DSM used to score the model, not from an
independently surveyed field measurement. They are simulated controls standing in for
what a real reference file would contain, not field truth. Do not describe them as GCPs,
survey points, or field-verified elevations anywhere downstream.

REPRODUCE

    python PHASE2_RED_FLAGS/scripts/E_05_hilly_risk_signal_experiment.py

Requires the project .venv (torch, transformers, scikit-learn, tifffile), the fine-tuned
checkpoint (`checkpoints/da2_small_finetuned_iter2_partial_unfreeze.pt`, gitignored, see
ARTIFACT_MANIFEST.md), and the preserved Mount Adams AOI arrays
(`PHASE2_RED_FLAGS/data/hilly_terrain/aoi/`) and Vaihingen raw area tiles
(`data/raw/Vaihingen/top/`, `data/raw/Vaihingen/dsm/`) -- none of these are re-downloaded.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tifffile
import torch

P2ROOT = Path(__file__).resolve().parents[1]
ROOT = P2ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from da2_utils import load_model                                   # noqa: E402
from model_stages import blend_tiles, predict_depth                 # noqa: E402
from relative_dsm import normalize_relative                         # noqa: E402
from tiling import DEFAULT_STRIDE, DEFAULT_WINDOW, plan_windows     # noqa: E402
from sklearn.linear_model import LinearRegression, RANSACRegressor  # noqa: E402

SCHEMA = "dw-s1-11-hilly-risk-signal/v2"   # v2: corrected B2 denominator, added fit-validity + provenance fields
CHECKPOINT_NAME = "da2_small_finetuned_iter2_partial_unfreeze.pt"   # run_pipeline.py's DEFAULT_CHECKPOINT
CHECKPOINT_PATH = ROOT / "checkpoints" / CHECKPOINT_NAME
SEEDS = 5                       # matches E_03_eval.py's sparse-10 seed count
N_SPARSE = 10                   # matches the preflight MIN_REFERENCE_POINTS floor

# Purely a floating-point division guard -- NOT a methodological "minimum meaningful
# range" cutoff. A seed whose 10 sampled points happen to span less than this many metres
# of elevation (or a non-finite range) is marked insufficient_evidence and excluded from
# pass/fail comparison, never silently assigned a huge or zero normalized residual.
EPS_DIVISION_GUARD = 1e-6

OUT_DIR = P2ROOT / "outputs" / "metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "E_hilly_risk_signal_experiment.json"
THIS_SCRIPT = Path(__file__).resolve()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_identity(path: Path) -> dict:
    """path + size + sha256 -- the strongest reproducible identity practical for these
    input sizes (tens of MB; hashing is cheap, so there is no reason to fall back to a
    weaker filename-only identity)."""
    path = Path(path)
    return {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception as exc:                                        # noqa: BLE001
        return f"unknown ({exc})"


def git_provenance() -> dict:
    """Branch/HEAD queried dynamically (never hardcoded), plus a compact working-tree
    dirty-state summary that distinguishes tracked modifications from the untracked
    #32 evidence files this very workflow expects to be present and uncommitted."""
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    porcelain = git("status", "--porcelain=v1", "--untracked-files=all")
    lines = [l for l in porcelain.splitlines() if l.strip()]
    tracked_modified = [l for l in lines if not l.startswith("??")]
    untracked = [l[3:] for l in lines if l.startswith("??")]
    expected_evidence_files = {
        "(internal note, not included)",
        "PHASE2_RED_FLAGS/outputs/metrics/E_hilly_risk_signal_experiment.json",
        "PHASE2_RED_FLAGS/scripts/E_05_hilly_risk_signal_experiment.py",
    }
    untracked_expected = [p for p in untracked if p in expected_evidence_files]
    untracked_other = [p for p in untracked if p not in expected_evidence_files]
    return {
        "branch": branch,
        "head": head,
        "tree_state": "clean" if not lines else "dirty",
        "tracked_modifications": tracked_modified,       # should be [] for this evidence-only pass
        "untracked_expected_evidence_files": untracked_expected,
        "untracked_other_preexisting": untracked_other,  # e.g. data/staging/ -- not part of this workflow
    }


def relief(arr: np.ndarray) -> float:
    return float(np.max(arr) - np.min(arr))


def ratio_and_cv(values: list[float], eps: float = 1e-6) -> dict:
    """Robust max/min ratio + CV. Denominators near zero are flagged, not divided by."""
    v = np.asarray(values, dtype=np.float64)
    lo, hi = float(v.min()), float(v.max())
    out = {
        "n": int(len(v)), "min": lo, "max": hi,
        "mean": float(v.mean()), "std": float(v.std()),
    }
    out["ratio_max_min"] = (hi / lo) if lo > eps else None
    out["ratio_denominator_near_zero"] = bool(lo <= eps)
    out["cv"] = (float(v.std() / v.mean())) if abs(v.mean()) > eps else None
    return out


def grid_windows(h: int, w: int, win: int):
    """Non-overlapping window origins tiling (h, w) -- used for the BLENDED / NORMALIZED /
    ground-truth statistic, deliberately independent of the overlapping sliding-window
    plan (plan_windows uses stride=384 < window=512) so post-blend windows are not
    double-counted across their 128px overlap."""
    ys = list(range(0, h - win + 1, win))
    xs = list(range(0, w - win + 1, win))
    return [(y, x) for y in ys for x in xs]


def fit_scale_shift_ransac(x: np.ndarray, y: np.ndarray, seed: int):
    """Line-for-line mirror of run_pipeline.py::fit_scale_shift_ransac (landed in PR #34,
    now merged into main at 0e6cf13) -- reimplemented, not imported, to keep this
    evidence-only branch independent of that tree. Any future drift between the two
    should be caught by a reviewer diffing this docstring against the merged function."""
    n = int(len(y))
    try:
        reg = RANSACRegressor(LinearRegression(), min_samples=max(2, n // 2), random_state=seed)
        reg.fit(x.reshape(-1, 1), y)
        a = float(reg.estimator_.coef_[0]); b = float(reg.estimator_.intercept_)
        inliers = int(reg.inlier_mask_.sum())
        return {"method": "ransac", "a": a, "b": b, "n_points": n,
                "inliers": inliers, "inlier_ratio": inliers / n}
    except Exception as exc:                                        # noqa: BLE001
        A = np.vstack([x, np.ones_like(x)]).T
        (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
        return {"method": f"ols_fallback ({type(exc).__name__})", "a": float(a), "b": float(b),
                "n_points": n, "inliers": None, "inlier_ratio": None}


def fit_validity(a: float, b: float) -> dict:
    """Mirrors run_pipeline.py::calibrate()'s existing metric-fit acceptance condition
    EXACTLY: finite a, finite b, a > 0. This is the PRODUCTION contract already merged
    on main (PR #34) -- not something introduced by this experiment. A seed that fails
    this is a case the runner already refuses on the metric branch (exit 2, "scale/shift
    fit is invalid"), independent of any hilly-risk signal, and must not be double-counted
    as evidence for or against a hypothetical additional detector."""
    finite_a, finite_b = bool(np.isfinite(a)), bool(np.isfinite(b))
    if not (finite_a and finite_b):
        reason = "non-finite scale or shift"
    elif a <= 0:
        reason = "non-positive scale"
    else:
        reason = None
    return {"fit_valid_for_metric": reason is None, "existing_fit_rejection_reason": reason}


def sparse10_diagnostics(surface: np.ndarray, gt: np.ndarray):
    """Draw N_SPARSE (row,col) points, fit RANSAC scale+shift (surface -> gt), record
    inlier ratio + residuals normalized by THAT SEED'S OWN sampled reference range --
    the corrected, runtime-computable form of signal B2. SEEDS independent draws.

    CORRECTION: the previous revision normalized by the full-scene dense ground-truth
    range (gt.max()-gt.min(), computed over the whole raster) -- an oracle/offline
    quantity, not something a real --reference file could supply. This normalizes by
    `y.max() - y.min()` over the 10 sampled points actually used in the fit, per seed.
    """
    h, w = surface.shape
    runs = []
    for seed in range(SEEDS):
        rng = np.random.default_rng(2000 + seed)
        rr = rng.integers(0, h, size=N_SPARSE)
        cc = rng.integers(0, w, size=N_SPARSE)
        x = surface[rr, cc].astype(np.float64)
        y = gt[rr, cc].astype(np.float64)

        sampled_reference_range_m = float(y.max() - y.min())
        range_finite = bool(np.isfinite(sampled_reference_range_m))
        range_usable = range_finite and sampled_reference_range_m > EPS_DIVISION_GUARD

        fit = fit_scale_shift_ransac(x, y, seed)
        resid = y - (fit["a"] * x + fit["b"])
        resid_abs_mean = float(np.mean(np.abs(resid)))
        resid_abs_max = float(np.max(np.abs(resid)))

        if range_usable:
            normalized_residual_mean = resid_abs_mean / sampled_reference_range_m
            normalized_residual_max = resid_abs_max / sampled_reference_range_m
            diagnostic_status = "evaluable"
        else:
            normalized_residual_mean = None
            normalized_residual_max = None
            diagnostic_status = ("insufficient_evidence: sampled reference range is "
                                 f"{'non-finite' if not range_finite else 'below the numerical division guard'} "
                                 f"({sampled_reference_range_m!r} m over {N_SPARSE} points)")

        runs.append({
            "seed": seed, **fit,
            "sampled_reference_points": {"row": rr.tolist(), "col": cc.tolist(), "elevation_m": y.tolist()},
            "sampled_reference_range_m": sampled_reference_range_m,
            "resid_abs_mean_m": resid_abs_mean,
            "resid_abs_max_m": resid_abs_max,
            "normalized_residual_mean_B2": normalized_residual_mean,
            "normalized_residual_max_B2": normalized_residual_max,
            "candidate_diagnostic_evaluable": range_usable,
            "candidate_diagnostic_status": diagnostic_status,
            **fit_validity(fit["a"], fit["b"]),
        })
    return runs


@torch.no_grad()
def run_scene(name: str, rgb: np.ndarray, gt: np.ndarray, model, device: str) -> dict:
    assert rgb.shape[:2] == gt.shape, f"{name}: rgb {rgb.shape} vs gt {gt.shape} shape mismatch"
    h, w = rgb.shape[:2]
    elev_range_full_scene = float(gt.max() - gt.min())
    print(f"[{name}] {h}x{w}, elevation range {elev_range_full_scene:.2f}m")

    plan = plan_windows(h, w, window=DEFAULT_WINDOW, stride=DEFAULT_STRIDE)
    depths, _tile_provenance = predict_depth(rgb, plan, model=model, device=device)
    blended = blend_tiles(depths, plan)
    normalized, norm_meta = normalize_relative(blended, depth_sign=1)

    # RAW_TILE: independent per-window predictions, pre-blend (exactly predict_depth()'s output)
    raw_tile_relief = [relief(depths[o]) for o in plan.origins]

    # BLENDED / NORMALIZED / ground-truth: non-overlapping grid, post-blend
    origins_grid = grid_windows(h, w, DEFAULT_WINDOW)
    blended_grid_relief = [relief(blended[y:y + DEFAULT_WINDOW, x:x + DEFAULT_WINDOW])
                           for (y, x) in origins_grid]
    normalized_grid_relief = [relief(normalized[y:y + DEFAULT_WINDOW, x:x + DEFAULT_WINDOW])
                              for (y, x) in origins_grid]
    gt_grid_relief = [relief(gt[y:y + DEFAULT_WINDOW, x:x + DEFAULT_WINDOW])
                      for (y, x) in origins_grid]

    return {
        "scene": name, "shape_hw": [h, w],
        "elevation_range_m_full_scene": elev_range_full_scene,
        "elevation_range_m_full_scene_NOTE": ("diagnostic context only -- NOT the B2 "
            "normalization denominator; see signal_B_sparse10_ransac_on_blended[*]."
            "sampled_reference_range_m for the corrected, runtime-computable denominator"),
        "sliding_window_plan": plan.as_manifest(),
        "n_grid_windows_nonoverlap": len(origins_grid),
        "signal_A_raw_tile_relief_heterogeneity": {
            "representation": "RAW_TILE (predict_depth output, pre-blend, sliding window plan)",
            **ratio_and_cv(raw_tile_relief),
        },
        "signal_A_blended_grid_relief_heterogeneity": {
            "representation": "BLENDED (blend_tiles output, non-overlapping grid)",
            **ratio_and_cv(blended_grid_relief),
        },
        "signal_A_normalized_grid_relief_heterogeneity": {
            "representation": "NORMALIZED (relative_dsm.normalize_relative output, non-overlapping grid)",
            "normalize_relative_meta": norm_meta,
            **ratio_and_cv(normalized_grid_relief),
        },
        "diagnostic_only_gt_grid_relief_heterogeneity": {
            "representation": "GROUND TRUTH -- diagnostic context only, NEVER a runtime input",
            **ratio_and_cv(gt_grid_relief),
        },
        "signal_B_sparse10_ransac_on_blended": sparse10_diagnostics(blended, gt),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  checkpoint={CHECKPOINT_NAME}")
    ckpt_identity = file_identity(CHECKPOINT_PATH)
    model = load_model(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
    model.eval()

    scenes = {}
    input_identities = {}

    # --- Mount Adams: known severe whole-scene calibration failure -----------------
    ma_dir = P2ROOT / "data" / "hilly_terrain" / "aoi"
    ma_rgb_path, ma_dem_path = ma_dir / "rgb_1m.npy", ma_dir / "dem_1m.npy"
    ma_rgb, ma_dem = np.load(ma_rgb_path), np.load(ma_dem_path)
    scenes["mount_adams"] = run_scene("mount_adams", ma_rgb, ma_dem, model, device)
    scenes["mount_adams"]["group"] = "known_failure"
    input_identities["mount_adams"] = {
        "rgb": file_identity(ma_rgb_path), "elevation": file_identity(ma_dem_path),
        "acquisition_provenance": (
            "PARTIALLY VERIFIED (see DW_S1_11_HILLY_RISK_METHODOLOGY.md section 4). "
            "The sha256 above identifies the LOCAL FILE actually read by this run, not the "
            "original raw-fetch chain -- the committed E_00_fetch_data.py / "
            "fetch_manifest.json record a different, discarded tile pair "
            "(WA_FEMAHQ_2018), and no committed script/hash documents the acquisition of "
            "the Mount Adams tile this experiment (and E_01-E_03) actually use. A present-"
            "day hash establishes reproducibility of THIS run, not the historical "
            "acquisition gap."),
    }

    # --- Vaihingen: scene-consistent controls, one ISPRS "area" tile per group -----
    va_groups = {
        "vaihingen_area38": ("known_good_most_heterogeneous_available",
                             "highest cross-patch relief ratio among test-split areas (see "
                             "prior per-area audit); n=28 test patches, matches Mount Adams sample size"),
        "vaihingen_area6": ("known_good_high_absolute_relief",
                            "highest absolute full-scene relief among test-split areas; n=12 test patches"),
        "vaihingen_area34": ("known_good_ordinary",
                             "lowest cross-patch relief ratio among test-split areas; n=8 test patches "
                             "(smallest group -- noted as a sample-size limitation)"),
        "vaihingen_area8": ("known_good_ordinary",
                            "low cross-patch relief ratio; n=12 test patches"),
    }
    va_top_dir = ROOT / "data" / "raw" / "Vaihingen" / "top"
    va_dsm_dir = ROOT / "data" / "raw" / "Vaihingen" / "dsm"
    for key, (group, note) in va_groups.items():
        area = key.replace("vaihingen_area", "area")
        top_path = va_top_dir / f"top_mosaic_09cm_{area}.tif"
        dsm_path = va_dsm_dir / f"dsm_09cm_matching_{area}.tif"
        rgb, dsm = tifffile.imread(top_path), tifffile.imread(dsm_path)
        scenes[key] = run_scene(key, rgb, dsm, model, device)
        scenes[key]["group"] = group
        scenes[key]["selection_rationale"] = note
        input_identities[key] = {"rgb": file_identity(top_path), "elevation": file_identity(dsm_path),
                                 "acquisition_provenance": "VERIFIED -- ISPRS Vaihingen benchmark, committed raw tile"}

    report = {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_provenance(),
        "experiment_script": {
            "path": str(THIS_SCRIPT.relative_to(ROOT)),
            "sha256": sha256_file(THIS_SCRIPT),
            "sha256_NOTE": "hash of the exact script bytes that produced this JSON",
        },
        "checkpoint": ckpt_identity,
        "device": device,
        "tiling_defaults": {"window": DEFAULT_WINDOW, "stride": DEFAULT_STRIDE},
        "sparse_fit": {
            "n_points": N_SPARSE, "seeds": SEEDS,
            "estimator": "RANSACRegressor(LinearRegression(), min_samples=max(2,n//2)) "
                        "-- mirrors run_pipeline.py's fit_scale_shift_ransac (PR #34, merged "
                        "into main), reimplemented not imported",
            "sparse_reference_provenance": (
                "The scale/shift PROCEDURE is deployable GIVEN GENUINE EXTERNAL REFERENCE "
                "ELEVATIONS -- it is exactly run_pipeline.py's --reference contract. The "
                "sparse reference elevations used IN THIS EXPERIMENT are simulated controls "
                "sampled from each scene's evaluation DSM (y = gt[rr, cc]), not independent "
                "field truth, not real GCP measurements, and not independently surveyed "
                "reference elevations. 'Deployable' describes the algorithm conditionally, "
                "never the provenance of these particular sampled values."),
            "b2_denominator_guard": (
                f"EPS_DIVISION_GUARD={EPS_DIVISION_GUARD} m is a floating-point division "
                "safety guard only, not a methodological minimum-range cutoff. A seed whose "
                "10 sampled points span less than this many metres (or a non-finite range) "
                "is marked candidate_diagnostic_evaluable=false / insufficient_evidence and "
                "excluded from pass/fail comparison."),
        },
        "input_identities": input_identities,
        "supersedes": ("exploratory cross-area Vaihingen draw (seed=7, 28 patches mixed across "
                       "9 ISPRS areas) run outside version control in the prior session -- "
                       "EXPLORATORY / SUPERSEDED, not scene-consistent with Mount Adams's single "
                       "contiguous AOI; SUPERSEDES the previous reviewed/uncommitted revision of "
                       "this script, "
                       "which normalized signal B2 by the full-scene dense ground-truth range "
                       "instead of the sampled reference points' own range (see module docstring, "
                       "'CORRECTION PASS')"),
        "scenes": scenes,
        "IMPORTANT": ("This file records candidate-signal DISTRIBUTIONS across real scenes, not "
                      "a validated detector or a threshold. See DW_S1_11_HILLY_RISK_METHODOLOGY.md "
                      "for the classification of each signal and the current STOP/GO decision. "
                      "diagnostic_only_gt_* and elevation_range_m_full_scene fields are ground "
                      "truth / oracle-only context and must never be treated as runtime-available."),
    }
    OUT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
