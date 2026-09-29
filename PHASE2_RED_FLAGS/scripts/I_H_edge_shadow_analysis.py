"""Sections 30-31: quantify roof/building-edge error and shadow-region error
using the real ISPRS semantic-label ground truth (building class) and a
simple RGB-darkness shadow heuristic, on a full real Vaihingen tile.
"""
import sys, json
from pathlib import Path
P2ROOT = Path(__file__).resolve().parents[1]
POFROOT = P2ROOT.parent
sys.path.insert(0, str(POFROOT / "src"))
import numpy as np
import torch
from scipy import ndimage
from da2_utils import preprocess_rgb, load_model, resize_pred_to
from metrics import all_metrics, fit_scale_shift
from geoio import load_geotiff

import os
AREA = int(os.environ.get("EDGE_AREA", 26))
TOP_DIR = POFROOT / "data" / "raw" / "Vaihingen" / "top"
DSM_DIR = POFROOT / "data" / "raw" / "Vaihingen" / "dsm"
GTS_DIR = POFROOT / "data" / "raw" / "Vaihingen" / "gts_for_participants"
CKPT = POFROOT / "checkpoints" / "da2_small_finetuned_iter2_partial_unfreeze.pt"
OUT_DIR = P2ROOT / "outputs" / "metrics"
FIG_DIR = P2ROOT / "outputs" / "figures"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = load_model(device)
model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
model.eval()

top = load_geotiff(TOP_DIR / f"top_mosaic_09cm_area{AREA}.tif")
dsm = load_geotiff(DSM_DIR / f"dsm_09cm_matching_area{AREA}.tif")
import tifffile
seg = tifffile.imread(str(GTS_DIR / f"top_mosaic_09cm_area{AREA}.tif"))  # RGB-coded classes

rgb = top.array[..., :3].astype(np.uint8)
gt = dsm.array.astype(np.float32)
H, W = gt.shape

WIN = 512; STRIDE = 384
accum = np.zeros((H, W), dtype=np.float64); weight = np.zeros((H, W), dtype=np.float64)
ramp = np.ones((WIN, WIN)); edge = WIN - STRIDE
lin = np.linspace(0, 1, edge)
ramp[:edge, :] *= lin[:, None]; ramp[-edge:, :] *= lin[::-1, None]
ramp[:, :edge] *= lin[None, :]; ramp[:, -edge:] *= lin[None, ::-1]

with torch.no_grad():
    ys = list(range(0, H - WIN + 1, STRIDE)) + [H - WIN]
    xs = list(range(0, W - WIN + 1, STRIDE)) + [W - WIN]
    for y in sorted(set(ys)):
        for x in sorted(set(xs)):
            patch = rgb[y:y+WIN, x:x+WIN]
            xin = preprocess_rgb(patch).unsqueeze(0).to(device)
            out = model(pixel_values=xin).predicted_depth
            pred = resize_pred_to(out, (WIN, WIN)).squeeze(0).cpu().numpy()
            accum[y:y+WIN, x:x+WIN] += pred * ramp
            weight[y:y+WIN, x:x+WIN] += ramp

pred_full = np.divide(accum, weight, out=np.zeros_like(accum), where=weight > 0)
valid = weight > 0
a, b = fit_scale_shift(pred_full[valid], gt[valid])
pred_cal = a * pred_full + b
abs_err = np.abs(pred_cal - gt)

# ---- building edge vs interior (Section 31) ----
building_mask = (seg[..., 0] < 30) & (seg[..., 1] < 30) & (seg[..., 2] > 220)  # blue class
# erode to get "interior" (away from boundary by >=5px), boundary = building pixels NOT in eroded interior
building_interior = ndimage.binary_erosion(building_mask, iterations=5)
building_edge = building_mask & ~building_interior
non_building = ~building_mask & valid

edge_err = abs_err[building_edge & valid]
interior_err = abs_err[building_interior & valid]
ground_err = abs_err[non_building]

edge_report = {
    "building_edge_mean_abs_err": float(edge_err.mean()) if edge_err.size else None,
    "building_interior_mean_abs_err": float(interior_err.mean()) if interior_err.size else None,
    "non_building_mean_abs_err": float(ground_err.mean()) if ground_err.size else None,
    "edge_vs_interior_ratio": float(edge_err.mean() / interior_err.mean()) if edge_err.size and interior_err.size else None,
    "n_edge_px": int(edge_err.size), "n_interior_px": int(interior_err.size), "n_non_building_px": int(ground_err.size),
}
print("Building edge report:", json.dumps(edge_report, indent=2))

# ---- vegetation (tree class) error, direct cross-check vs Phase-1 area-6 finding ----
tree_mask = (seg[..., 0] < 30) & (seg[..., 1] > 220) & (seg[..., 2] < 30)
tree_err = abs_err[tree_mask & valid]
nontree_err = abs_err[~tree_mask & valid]
veg_report = {
    "tree_class_mean_abs_err": float(tree_err.mean()) if tree_err.size else None,
    "non_tree_mean_abs_err": float(nontree_err.mean()) if nontree_err.size else None,
    "tree_vs_nontree_ratio": float(tree_err.mean() / nontree_err.mean()) if tree_err.size and nontree_err.size else None,
    "n_tree_px": int(tree_err.size),
}
print("Vegetation (tree class) report:", json.dumps(veg_report, indent=2))

# ---- shadow heuristic (Section 30): darkest 10% of pixels by RGB luminance ----
luminance = rgb.astype(np.float32).mean(axis=-1)
shadow_thresh = np.percentile(luminance[valid], 10)
shadow_mask = (luminance <= shadow_thresh) & valid
nonshadow_mask = (luminance > shadow_thresh) & valid
shadow_err = abs_err[shadow_mask]
nonshadow_err = abs_err[nonshadow_mask]
shadow_report = {
    "shadow_threshold_luminance": float(shadow_thresh),
    "shadow_mean_abs_err": float(shadow_err.mean()) if shadow_err.size else None,
    "non_shadow_mean_abs_err": float(nonshadow_err.mean()) if nonshadow_err.size else None,
    "shadow_vs_nonshadow_ratio": float(shadow_err.mean() / nonshadow_err.mean()) if shadow_err.size and nonshadow_err.size else None,
    "n_shadow_px": int(shadow_err.size),
}
print("Shadow report:", json.dumps(shadow_report, indent=2))

full_report = {"area": AREA, "global_rmse": all_metrics(pred_cal[valid], gt[valid])["rmse"],
                "building_edge": edge_report, "vegetation_tree_class": veg_report, "shadow": shadow_report}
with open(OUT_DIR / f"IH_edge_shadow_analysis_area{AREA}.json", "w") as f:
    json.dump(full_report, f, indent=2)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(rgb); axes[0].set_title("RGB")
axes[1].imshow(abs_err, cmap="inferno", vmax=np.percentile(abs_err[valid], 95)); axes[1].set_title("Abs error")
axes[2].imshow(building_edge, cmap="gray"); axes[2].set_title("Building edge mask")
axes[3].imshow(shadow_mask, cmap="gray"); axes[3].set_title("Shadow mask (darkest 10%)")
for ax in axes: ax.axis("off")
plt.tight_layout()
plt.savefig(FIG_DIR / f"IH_edge_shadow_area{AREA}.png", dpi=110)
print(f"\nSaved -> {OUT_DIR / f'IH_edge_shadow_analysis_area{AREA}.json'} and figure")
