"""Constant-baseline (predict per-patch mean CHM for every pixel) on the same
forest test patches, for honest context: does the model add ANY value beyond
a trivial mean predictor? This is the same diagnostic Phase-1 used (constant
mean baseline: RMSE 18.15m on Vaihingen).
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np
from metrics import all_metrics

ROOT = Path(__file__).resolve().parents[1]
PATCH_DIR = ROOT / "data" / "forest_eth_s2" / "patches"
METRICS_DIR = ROOT / "outputs" / "metrics"

manifest = json.load(open(PATCH_DIR / "manifest.json"))
test_entries = [m for m in manifest if m["split"] == "test"]

per_patch = {}
all_pred, all_gt = [], []
for e in test_entries:
    d = np.load(PATCH_DIR / e["file"])
    chm, valid = d["chm"], d["valid"]
    g = chm[valid]
    pred = np.full_like(g, g.mean())
    m = all_metrics(pred, g)
    per_patch[e["file"]] = m
    all_pred.append(pred); all_gt.append(g)
    print(f"  {e['file']}: constant-mean RMSE={m['rmse']:.3f} (mean height={g.mean():.2f}m, std={g.std():.2f}m)")

all_pred = np.concatenate(all_pred); all_gt = np.concatenate(all_gt)
global_m = all_metrics(all_pred, all_gt)
print(f"GLOBAL constant-per-patch-mean baseline: RMSE={global_m['rmse']:.3f} MAE={global_m['mae']:.3f}")

with open(METRICS_DIR / "D1b_constant_baseline_forest.json", "w") as f:
    json.dump({"per_patch": per_patch, "global_pooled": global_m}, f, indent=2)
