"""Experiment F (Section 25): calibration-point NOISE sensitivity, using noise
bands derived from documented real external-source accuracy (not invented):
  - Copernicus GLO-30 (typical, flat/open terrain): RMSE ~1.68m (ISPRS Annals
    X-4-W1-2022-209-2023, ICESat-validated)
  - Copernicus GLO-30 (rough/complex terrain, worse case): LE90 ~7.71m per a
    separate LiDAR-DTM comparison cited in the same research pass
  - GEDI footprint-level terrain accuracy: RMSE ~4.03m (low latitudes)
  - ICESat-2 ATL08 terrain: RMSE ~0.93-2.24m (use midpoint ~1.5m)
Applies Gaussian noise at each sigma to the TRUE elevation value at N=10
randomly-sampled (uniform) calibration points per tile, before fitting
scale+shift, then evaluates on the full tile (same protocol as the sparse-10
number elsewhere in this Phase). This does NOT simulate real spatial
autocorrelation or systematic bias of an actual external DEM (see caveat in
08_REAL_CALIBRATION_TEST.md) -- it isolates point-value noise sensitivity
only, using literature-grounded magnitudes as required by the master brief.
"""
import sys, json
from pathlib import Path
P2ROOT = Path(__file__).resolve().parents[1]
POFROOT = P2ROOT.parent
sys.path.insert(0, str(POFROOT / "src"))
import numpy as np
import torch
from da2_utils import preprocess_rgb, load_model, resize_pred_to
from metrics import all_metrics, fit_scale_shift

PATCH_DIR = POFROOT / "data" / "processed" / "patches"
CKPT = POFROOT / "checkpoints" / "da2_small_finetuned_iter2_partial_unfreeze.pt"
OUT_DIR = P2ROOT / "outputs" / "metrics"
device = "cuda" if torch.cuda.is_available() else "cpu"

manifest = json.load(open(PATCH_DIR / "manifest.json"))
test_entries = [m for m in manifest if m["split"] == "test"]
model = load_model(device)
model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
model.eval()

def load_patch(entry):
    d = np.load(PATCH_DIR / "test" / entry["file"])
    return d["rgb"], d["dsm"]

@torch.no_grad()
def predict(rgb):
    x = preprocess_rgb(rgb).unsqueeze(0).to(device)
    out = model(pixel_values=x).predicted_depth
    return resize_pred_to(out, rgb.shape[:2]).squeeze(0).cpu().numpy()

per_tile_pred, per_tile_gt = {}, {}
for e in test_entries:
    rgb, dsm = load_patch(e)
    pred = predict(rgb)
    per_tile_pred.setdefault(e["area"], []).append(pred.ravel())
    per_tile_gt.setdefault(e["area"], []).append(dsm.ravel())
areas = sorted(per_tile_pred.keys())
for area in areas:
    per_tile_pred[area] = np.concatenate(per_tile_pred[area])
    per_tile_gt[area] = np.concatenate(per_tile_gt[area])

NOISE_BANDS = {
    "0_oracle": 0.0,
    "1.5_icesat2_terrain": 1.5,
    "1.7_glo30_typical": 1.7,
    "4.0_gedi_terrain": 4.0,
    "7.7_glo30_rough_terrain": 7.7,
    "10_stress": 10.0,
}
N_POINTS = 10
N_SEEDS = 15

results = {}
for band_name, sigma in NOISE_BANDS.items():
    rmses = []
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(2000 + seed)
        all_pred_cal, all_gt_cal = [], []
        for area in areas:
            p = per_tile_pred[area]; g = per_tile_gt[area]
            n = min(N_POINTS, p.size)
            idx = rng.choice(p.size, size=n, replace=False)
            g_noisy = g[idx] + rng.normal(0, sigma, size=n) if sigma > 0 else g[idx]
            a, b = fit_scale_shift(p[idx], g_noisy)
            p_cal = a * p + b
            all_pred_cal.append(p_cal); all_gt_cal.append(g)
        m = all_metrics(np.concatenate(all_pred_cal), np.concatenate(all_gt_cal))
        rmses.append(m["rmse"])
    results[band_name] = {"sigma_m": sigma, "mean_rmse": float(np.mean(rmses)),
                            "std_rmse": float(np.std(rmses)), "min_rmse": float(np.min(rmses)),
                            "max_rmse": float(np.max(rmses))}
    print(f"{band_name:<26} sigma={sigma:>5.1f}m  mean_RMSE={np.mean(rmses):.3f}  std={np.std(rmses):.3f}")

with open(OUT_DIR / "F_calibration_noise.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved -> {OUT_DIR / 'F_calibration_noise.json'}")
