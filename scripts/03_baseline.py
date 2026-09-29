"""Experiment 1: dumb constant baseline (training-set mean / median height)."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from geoio import load_geotiff, dsm_nodata_mask
from metrics import all_metrics

ROOT = Path(__file__).resolve().parents[1]
DSM_DIR = ROOT / "data" / "raw" / "Vaihingen" / "dsm"
split = json.load(open(ROOT / "data" / "splits" / "split_v1.json"))

def tile_valid_values(area):
    dsm = load_geotiff(DSM_DIR / f"dsm_09cm_matching_area{area}.tif")
    mask = dsm_nodata_mask(dsm.array)
    return dsm.array[mask]

train_vals = np.concatenate([tile_valid_values(a) for a in split["train_areas"]])
train_mean = float(np.mean(train_vals))
train_median = float(np.median(train_vals))
print(f"Train-set mean height: {train_mean:.3f} m | median: {train_median:.3f} m (n={len(split['train_areas'])} tiles)")

results = {"train_mean": train_mean, "train_median": train_median, "per_tile": {}, "global": {}}

all_gt, all_pred_mean, all_pred_median = [], [], []
for area in split["test_areas"]:
    gt = tile_valid_values(area)
    pred_mean = np.full_like(gt, train_mean)
    pred_median = np.full_like(gt, train_median)
    m_mean = all_metrics(pred_mean, gt, prefix="mean_")
    m_median = all_metrics(pred_median, gt, prefix="median_")
    results["per_tile"][area] = {**m_mean, **m_median, "n_px": int(gt.size)}
    all_gt.append(gt); all_pred_mean.append(pred_mean); all_pred_median.append(pred_median)
    print(f"  test area {area}: constant-mean RMSE={m_mean['mean_rmse']:.3f} MAE={m_mean['mean_mae']:.3f} | "
          f"constant-median RMSE={m_median['median_rmse']:.3f} MAE={m_median['median_mae']:.3f}")

all_gt = np.concatenate(all_gt); all_pred_mean = np.concatenate(all_pred_mean); all_pred_median = np.concatenate(all_pred_median)
results["global"]["constant_mean"] = all_metrics(all_pred_mean, all_gt)
results["global"]["constant_median"] = all_metrics(all_pred_median, all_gt)
print("\n=== GLOBAL (all test tiles pooled) ===")
print("constant-mean baseline:", results["global"]["constant_mean"])
print("constant-median baseline:", results["global"]["constant_median"])

out = ROOT / "outputs" / "metrics" / "baseline_constant.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved -> {out}")
