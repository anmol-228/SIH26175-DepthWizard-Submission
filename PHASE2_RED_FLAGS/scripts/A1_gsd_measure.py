"""Experiment A1: existing Phase-1 fine-tuned model (iter2, partial-unfreeze,
NO new training) evaluated on the SAME held-out Vaihingen test tiles with the
INPUT imagery synthetically degraded to a range of target GSDs. DSM ground
truth is left at native 9cm throughout (per master-brief Section 9 step 4).

For each GSD: full per-tile oracle calibration (upper bound, matches Phase-1
methodology for direct comparability) AND 10-point sparse calibration
(matches the Phase-1 "realistic deployable" comparison point, 5.50m @ 9cm).
"""
import sys, json
from pathlib import Path
P2ROOT = Path(__file__).resolve().parents[1]
POFROOT = P2ROOT.parent
sys.path.insert(0, str(POFROOT / "src"))
sys.path.insert(0, str(P2ROOT / "scripts"))
import numpy as np
import torch
from sklearn.linear_model import RANSACRegressor, LinearRegression
from da2_utils import preprocess_rgb, load_model, resize_pred_to
from metrics import all_metrics, fit_scale_shift
from gsd_utils import degrade_rgb_to_gsd, NATIVE_GSD

PATCH_DIR = POFROOT / "data" / "processed" / "patches"
CKPT = POFROOT / "checkpoints" / "da2_small_finetuned_iter2_partial_unfreeze.pt"
OUT_DIR = P2ROOT / "outputs" / "metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
manifest = json.load(open(PATCH_DIR / "manifest.json"))
test_entries = [m for m in manifest if m["split"] == "test"]

model = load_model(device)
model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
model.eval()
print(f"Loaded checkpoint: {CKPT.name}")

def load_patch(entry):
    d = np.load(PATCH_DIR / "test" / entry["file"])
    return d["rgb"], d["dsm"]

@torch.no_grad()
def predict(rgb):
    x = preprocess_rgb(rgb).unsqueeze(0).to(device)
    out = model(pixel_values=x).predicted_depth
    return resize_pred_to(out, rgb.shape[:2]).squeeze(0).cpu().numpy()

GSD_LADDER = [0.09, 0.30, 0.50, 1.0, 2.0, 3.0, 5.0]
rng = np.random.default_rng(42)

results = {}
for gsd in GSD_LADDER:
    per_tile_pred, per_tile_gt = {}, {}
    for e in test_entries:
        rgb, dsm = load_patch(e)
        rgb_deg = degrade_rgb_to_gsd(rgb, gsd, NATIVE_GSD)
        pred = predict(rgb_deg)
        per_tile_pred.setdefault(e["area"], []).append(pred)
        per_tile_gt.setdefault(e["area"], []).append(dsm)
    areas = sorted(per_tile_pred.keys())

    # full per-tile oracle calibration
    all_pred_full, all_gt_full = [], []
    per_tile_oracle = {}
    for area in areas:
        p = np.concatenate([x.ravel() for x in per_tile_pred[area]])
        g = np.concatenate([x.ravel() for x in per_tile_gt[area]])
        a, b = fit_scale_shift(p, g)
        p_cal = a * p + b
        per_tile_oracle[area] = all_metrics(p_cal, g)
        all_pred_full.append(p_cal); all_gt_full.append(g)
    all_pred_full = np.concatenate(all_pred_full); all_gt_full = np.concatenate(all_gt_full)
    global_oracle = all_metrics(all_pred_full, all_gt_full)

    # 10-point sparse calibration (RANSAC), matches Phase-1 realistic comparator
    all_pred_sparse, all_gt_sparse = [], []
    per_tile_sparse = {}
    for area in areas:
        p = np.concatenate([x.ravel() for x in per_tile_pred[area]])
        g = np.concatenate([x.ravel() for x in per_tile_gt[area]])
        n = min(10, p.size)
        idx = rng.choice(p.size, size=n, replace=False)
        try:
            reg = RANSACRegressor(LinearRegression(), min_samples=max(2, n // 2), random_state=42)
            reg.fit(p[idx].reshape(-1, 1), g[idx])
            a = float(reg.estimator_.coef_[0]); b = float(reg.estimator_.intercept_)
        except Exception:
            a, b = fit_scale_shift(p[idx], g[idx])
        p_cal = a * p + b
        per_tile_sparse[area] = all_metrics(p_cal, g)
        all_pred_sparse.append(p_cal); all_gt_sparse.append(g)
    all_pred_sparse = np.concatenate(all_pred_sparse); all_gt_sparse = np.concatenate(all_gt_sparse)
    global_sparse10 = all_metrics(all_pred_sparse, all_gt_sparse)

    results[str(gsd)] = {
        "gsd_m": gsd,
        "global_full_oracle": global_oracle,
        "global_sparse10": global_sparse10,
        "per_tile_full_oracle": per_tile_oracle,
        "per_tile_sparse10": per_tile_sparse,
    }
    print(f"GSD={gsd:>5.2f}m  full_oracle RMSE={global_oracle['rmse']:.3f} corr={global_oracle['corr']:.3f}  |  "
          f"sparse10 RMSE={global_sparse10['rmse']:.3f} corr={global_sparse10['corr']:.3f}")

with open(OUT_DIR / "A1_gsd_measure_no_adaptation.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved -> {OUT_DIR / 'A1_gsd_measure_no_adaptation.json'}")
