"""Experiment E, step 3: evaluate (a) the original zero-shot pretrained
Depth Anything V2 Small and (b) the Phase-1 Vaihingen-fine-tuned checkpoint
on the real hilly-terrain AOI (Mount Adams, WA -- see 07_HILLY_TERRAIN_TEST.md),
under full per-scene oracle calibration and realistic 10-point sparse
calibration (RANSAC, matching scripts/07_calibrate_and_eval.py's
methodology). "Tile" == the whole AOI (one real contiguous acquired scene),
not an individual 512x512 crop -- this matches how per-scene calibration
would actually be deployed (one set of GCP/altimetry points per scene
footprint, not per arbitrary crop).

Also computes slope-stratified error (simple finite-difference slope,
already precomputed per-patch in E_02) for both calibration regimes.
"""
import sys, json, os
from pathlib import Path
P2ROOT = Path(__file__).resolve().parents[1]
POFROOT = P2ROOT.parent
sys.path.insert(0, str(POFROOT / "src"))
import numpy as np
import torch
from sklearn.linear_model import RANSACRegressor, LinearRegression
from da2_utils import preprocess_rgb, load_model, resize_pred_to
from metrics import all_metrics, fit_scale_shift

PATCH_DIR = P2ROOT / "data" / "hilly_terrain" / "patches"
OUT_DIR = P2ROOT / "outputs" / "metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
manifest = json.load(open(PATCH_DIR / "manifest.json"))
print(f"{len(manifest)} patches total")

SLOPE_BINS = [(0, 5), (5, 15), (15, 30), (30, 90)]


def load_patch(entry):
    d = np.load(PATCH_DIR / entry["file"])
    return d["rgb"], d["dsm"], d["slope"]


@torch.no_grad()
def predict_all(model, entries):
    preds, gts, slopes = [], [], []
    for e in entries:
        rgb, dsm, slope = load_patch(e)
        x = preprocess_rgb(rgb).unsqueeze(0).to(device)
        out = model(pixel_values=x).predicted_depth
        pred = resize_pred_to(out, dsm.shape).squeeze(0).cpu().numpy()
        preds.append(pred); gts.append(dsm); slopes.append(slope)
    return preds, gts, slopes


def slope_stratified(pred_cal_flat, gt_flat, slope_flat):
    out = {}
    for lo, hi in SLOPE_BINS:
        m = (slope_flat >= lo) & (slope_flat < hi)
        n = int(m.sum())
        if n < 500:
            out[f"{lo}-{hi}deg"] = {"n_px": n, "note": "insufficient data, not reported"}
            continue
        met = all_metrics(pred_cal_flat[m], gt_flat[m])
        met["n_px"] = n
        out[f"{lo}-{hi}deg"] = met
    return out


def run_model(model, tag, entries, rng):
    preds, gts, slopes = predict_all(model, entries)
    p_all = np.concatenate([x.ravel() for x in preds])
    g_all = np.concatenate([x.ravel() for x in gts])
    s_all = np.concatenate([x.ravel() for x in slopes])

    result = {}

    # raw, uncalibrated (diagnostic only, NOT a valid final metric)
    result["raw_uncalibrated_diagnostic"] = all_metrics(p_all, g_all)

    # --- per-patch oracle calibration (best-case diagnostic: each 512x512
    # patch gets its OWN independent scale+shift fit -- NOT a realistic
    # deployment regime since real GCP/altimetry calibration happens per
    # acquired SCENE, not per arbitrary crop, but useful to isolate whether
    # poor whole-scene results are a calibration-granularity artifact or a
    # genuine representational limit) ---
    per_patch_oracle = {}
    per_patch_cal_preds, per_patch_cal_gts = [], []
    for e, p, g in zip(entries, preds, gts):
        a_p, b_p = fit_scale_shift(p.ravel(), g.ravel())
        cal = a_p * p + b_p
        m = all_metrics(cal.ravel(), g.ravel())
        m.update({"a": a_p, "b": b_p, "relief_m": e["relief_m"], "slope_mean_deg": e["slope_mean_deg"]})
        per_patch_oracle[e["id"]] = m
        per_patch_cal_preds.append(cal); per_patch_cal_gts.append(g)
    pp_rmses = [v["rmse"] for v in per_patch_oracle.values()]
    result["per_patch_oracle_calibration"] = {
        "per_patch": per_patch_oracle,
        "pooled_metrics": all_metrics(np.concatenate([x.ravel() for x in per_patch_cal_preds]),
                                       np.concatenate([x.ravel() for x in per_patch_cal_gts])),
        "rmse_mean_of_patches": float(np.mean(pp_rmses)), "rmse_std_of_patches": float(np.std(pp_rmses)),
        "rmse_min": float(np.min(pp_rmses)), "rmse_max": float(np.max(pp_rmses)),
    }
    result["per_patch_oracle_slope_stratified"] = slope_stratified(
        np.concatenate([x.ravel() for x in per_patch_cal_preds]),
        np.concatenate([x.ravel() for x in per_patch_cal_gts]), s_all)

    # full-scene oracle calibration (ONE (a,b) for the whole real AOI/scene --
    # this is the realistic per-scene-calibration granularity)
    a, b = fit_scale_shift(p_all, g_all)
    p_cal_oracle = a * p_all + b
    result["full_oracle_calibration"] = {**all_metrics(p_cal_oracle, g_all), "a": a, "b": b, "n_px": int(p_all.size)}
    result["full_oracle_slope_stratified"] = slope_stratified(p_cal_oracle, g_all, s_all)

    # sparse 10-point calibration, RANSAC, 5 random seeds for variance
    sparse_runs = []
    for seed in range(5):
        r = np.random.default_rng(1000 + seed)
        n = min(10, p_all.size)
        idx = r.choice(p_all.size, size=n, replace=False)
        try:
            reg = RANSACRegressor(LinearRegression(), min_samples=max(2, n // 2), random_state=seed)
            reg.fit(p_all[idx].reshape(-1, 1), g_all[idx])
            a_s = float(reg.estimator_.coef_[0]); b_s = float(reg.estimator_.intercept_)
        except Exception as ex:
            a_s, b_s = fit_scale_shift(p_all[idx], g_all[idx])
        p_cal_sparse = a_s * p_all + b_s
        m = all_metrics(p_cal_sparse, g_all)
        m.update({"a": a_s, "b": b_s, "seed": seed})
        sparse_runs.append(m)
    rmses = [r["rmse"] for r in sparse_runs]
    result["sparse_10pt_calibration_runs"] = sparse_runs
    result["sparse_10pt_calibration_summary"] = {
        "rmse_mean": float(np.mean(rmses)), "rmse_std": float(np.std(rmses)),
        "rmse_min": float(np.min(rmses)), "rmse_max": float(np.max(rmses)),
    }
    # slope-stratified using the median-RMSE seed's calibration for a representative single run
    med_idx = int(np.argsort(rmses)[len(rmses) // 2])
    a_s, b_s = sparse_runs[med_idx]["a"], sparse_runs[med_idx]["b"]
    result["sparse_10pt_slope_stratified_median_run"] = slope_stratified(a_s * p_all + b_s, g_all, s_all)

    print(f"[{tag}] raw RMSE={result['raw_uncalibrated_diagnostic']['rmse']:.2f}  "
          f"per_patch_oracle mean_RMSE={result['per_patch_oracle_calibration']['rmse_mean_of_patches']:.2f}  "
          f"full_scene_oracle RMSE={result['full_oracle_calibration']['rmse']:.2f}  "
          f"sparse10 RMSE mean={result['sparse_10pt_calibration_summary']['rmse_mean']:.2f} "
          f"(+/-{result['sparse_10pt_calibration_summary']['rmse_std']:.2f})")
    return result


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    report = {"n_patches": len(manifest), "aoi": "Mount Adams W flank, WA (see 07_HILLY_TERRAIN_TEST.md)"}

    print("\n=== ZERO-SHOT (original pretrained DA2-Small, no Vaihingen fine-tuning) ===")
    model_zs = load_model(device)
    model_zs.eval()
    report["zero_shot"] = run_model(model_zs, "zero-shot", manifest, rng)
    del model_zs
    torch.cuda.empty_cache()

    print("\n=== PHASE-1 FINE-TUNED (Vaihingen, iter2 partial-unfreeze checkpoint) ===")
    model_ft = load_model(device)
    ckpt = POFROOT / "checkpoints" / "da2_small_finetuned_iter2_partial_unfreeze.pt"
    model_ft.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model_ft.eval()
    report["finetuned_vaihingen"] = run_model(model_ft, "finetuned-vaihingen", manifest, rng)

    with open(OUT_DIR / "E_hilly_terrain_eval.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved -> {OUT_DIR / 'E_hilly_terrain_eval.json'}")
