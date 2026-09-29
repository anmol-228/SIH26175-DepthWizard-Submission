"""Section 33: large-image tiling/stitching test. Runs the fine-tuned model
on a FULL Vaihingen tile (not a pre-cropped 512x512 patch) via a sliding
window with overlap, blends overlapping predictions with a linear feather,
reassembles a full-tile DSM, and checks for seam discontinuities at the
former window boundaries vs. elsewhere in the tile.
"""
import sys, json, time, tracemalloc
from pathlib import Path
P2ROOT = Path(__file__).resolve().parents[1]
POFROOT = P2ROOT.parent
sys.path.insert(0, str(POFROOT / "src"))
import numpy as np
import torch
from da2_utils import preprocess_rgb, load_model, resize_pred_to
from metrics import all_metrics, fit_scale_shift
from geoio import load_geotiff, dsm_nodata_mask, save_geotiff, read_tfw, DSM_NODATA, CRS_UNDEFINED

TOP_DIR = POFROOT / "data" / "raw" / "Vaihingen" / "top"
DSM_DIR = POFROOT / "data" / "raw" / "Vaihingen" / "dsm"
CKPT = POFROOT / "checkpoints" / "da2_small_finetuned_iter2_partial_unfreeze.pt"
import os
# Re-runs must not overwrite the committed Sprint-0 result: set K1_OUT_NAME to a new
# filename, mirroring 07_calibrate_and_eval.py's EVAL_OUT_NAME. Default is unchanged.
OUT_NAME = os.environ.get("K1_OUT_NAME", "K1_tiling_stitch.json")
OUT_DIR = P2ROOT / "outputs" / "metrics"
FIG_DIR = P2ROOT / "outputs" / "figures"
PRED_DIR = P2ROOT / "outputs" / "predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True); FIG_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)
# Vaihingen ships its geotransform in a separate archive, not as a sidecar next to
# the DSM tiles -- same source K2_georeferencing_export.py reads.
GEOINFO_DIR = POFROOT / "data" / "raw" / "Vaihingen" / "geoinfo_extracted"
EPSG = CRS_UNDEFINED  # 32767 user-defined: the geotransform is real, but the source
                      # dsm.vrt declares EPSG:32633 (UTM 33N) while the coordinates imply
                      # UTM 32N (~8.96E = Vaihingen). We do not assert an authority code
                      # we cannot defend. See src/geoio.CRS_UNDEFINED.
device = "cuda" if torch.cuda.is_available() else "cpu"

AREA = 26  # a real Vaihingen test-set tile ("typical" case per Phase-1 error analysis)
WIN = 512
STRIDE = 384  # 128px overlap

model = load_model(device)
model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
model.eval()

top = load_geotiff(TOP_DIR / f"top_mosaic_09cm_area{AREA}.tif")
dsm = load_geotiff(DSM_DIR / f"dsm_09cm_matching_area{AREA}.tif")
rgb_full = top.array[..., :3].astype(np.uint8)
gt_full = dsm.array.astype(np.float32)
H, W = gt_full.shape
print(f"Full tile area {AREA}: {H}x{W} px ({H*0.09:.0f}m x {W*0.09:.0f}m)")

tracemalloc.start()
t0 = time.time()

accum = np.zeros((H, W), dtype=np.float64)
weight = np.zeros((H, W), dtype=np.float64)

# feather weight: linear ramp from edges of each window (reduces seam weight)
ramp = np.ones((WIN, WIN), dtype=np.float64)
edge = WIN - STRIDE  # overlap width
lin = np.linspace(0, 1, edge) if edge > 0 else np.array([])
if edge > 0:
    ramp[:edge, :] *= lin[:, None]
    ramp[-edge:, :] *= lin[::-1, None]
    ramp[:, :edge] *= lin[None, :]
    ramp[:, -edge:] *= lin[None, ::-1]

n_windows = 0
with torch.no_grad():
    for y in range(0, H - WIN + 1, STRIDE):
        for x in range(0, W - WIN + 1, STRIDE):
            patch = rgb_full[y:y+WIN, x:x+WIN]
            xin = preprocess_rgb(patch).unsqueeze(0).to(device)
            out = model(pixel_values=xin).predicted_depth
            pred = resize_pred_to(out, (WIN, WIN)).squeeze(0).cpu().numpy()
            accum[y:y+WIN, x:x+WIN] += pred * ramp
            weight[y:y+WIN, x:x+WIN] += ramp
            n_windows += 1
    # cover the remainder (right/bottom edges) with one more pass anchored at H-WIN/W-WIN
    for y in [H - WIN]:
        for x in range(0, W - WIN + 1, STRIDE):
            pass  # already covered by loop above since range includes final aligned step only if divisible; handle edge explicitly below

# explicit edge coverage (bottom row, right column) since stride may not divide evenly
last_y = H - WIN
last_x = W - WIN
for y in [0, last_y]:
    for x in [0, last_x]:
        if weight[y:y+WIN, x:x+WIN].min() > 0:
            continue
        patch = rgb_full[y:y+WIN, x:x+WIN]
        xin = preprocess_rgb(patch).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(pixel_values=xin).predicted_depth
        pred = resize_pred_to(out, (WIN, WIN)).squeeze(0).cpu().numpy()
        accum[y:y+WIN, x:x+WIN] += pred * ramp
        weight[y:y+WIN, x:x+WIN] += ramp
        n_windows += 1

pred_full = np.divide(accum, weight, out=np.zeros_like(accum), where=weight > 0)
valid = weight > 0
dt = time.time() - t0
mem_cur, mem_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

# ---- single-shot comparison (no blending, one window per grid cell, hard boundaries) ----
accum_hard = np.zeros((H, W), dtype=np.float64)
count_hard = np.zeros((H, W), dtype=np.int32)
with torch.no_grad():
    for y in range(0, H - WIN + 1, WIN):
        for x in range(0, W - WIN + 1, WIN):
            patch = rgb_full[y:y+WIN, x:x+WIN]
            xin = preprocess_rgb(patch).unsqueeze(0).to(device)
            out = model(pixel_values=xin).predicted_depth
            pred = resize_pred_to(out, (WIN, WIN)).squeeze(0).cpu().numpy()
            accum_hard[y:y+WIN, x:x+WIN] = pred
            count_hard[y:y+WIN, x:x+WIN] = 1
hard_valid = count_hard > 0

# oracle calibration (fit on the FEATHERED/blended prediction, applied to both for fair comparison)
p_blend = pred_full[valid]; g_blend = gt_full[valid]
a, b = fit_scale_shift(p_blend, g_blend)
pred_full_cal = a * pred_full + b
m_blend = all_metrics(pred_full_cal[valid], gt_full[valid])

p_hard = accum_hard[hard_valid]; g_hard = gt_full[hard_valid]
a2, b2 = fit_scale_shift(p_hard, g_hard)
pred_hard_cal = a2 * accum_hard + b2
m_hard = all_metrics(pred_hard_cal[hard_valid], gt_full[hard_valid])

# ---- export the calibrated full-tile DSM as a real georeferenced GeoTIFF ----
# This is the CALIBRATION -> PIPELINE/VIEWER handoff (handbook Part X, issue #5).
# Until now the calibrated raster existed only in memory here and in
# 07_calibrate_and_eval.py, while K2 proved the GeoTIFF mechanics against the
# GROUND-TRUTH DSM. This wires the two together for the first time.
dsm_export = {"written": False, "reason": None}
tfw_path = GEOINFO_DIR / f"dsm_09cm_matching_area{AREA}.tfw"
if tfw_path.exists():
    px_g, _, _, py_g, ox_g, oy_g = read_tfw(tfw_path)
    # Blank out pixels no window covered, using the project-wide nodata convention
    # rather than letting an uncalibrated zero masquerade as a real elevation.
    out_dsm = np.where(valid, pred_full_cal, DSM_NODATA).astype(np.float32)
    dsm_path = save_geotiff(
        PRED_DIR / f"K1_calibrated_dsm_area{AREA}.tif", out_dsm,
        pixel_size_x=px_g, pixel_size_y=py_g, origin_x=ox_g, origin_y=oy_g,
        epsg=EPSG, nodata=DSM_NODATA,
    )
    dsm_export = {
        "written": True,
        "path": str(dsm_path.relative_to(POFROOT)),
        "product": "metric_dsm",
        "calibration": "per_scene_oracle_scale_shift",
        "calibration_a": float(a), "calibration_b": float(b),
        "units": "metres", "dtype": "float32", "nodata": DSM_NODATA,
        "epsg": EPSG, "pixel_size": [px_g, py_g], "origin": [ox_g, oy_g],
        "shape": [int(H), int(W)],
        "n_valid_px": int(valid.sum()), "n_nodata_px": int((~valid).sum()),
        "tfw_source": str(tfw_path.relative_to(POFROOT)),
        "IMPORTANT": ("Oracle-calibrated against this tile's own ground-truth DSM. This is a "
                      "non-deployable upper bound, NOT a field accuracy figure."),
    }
    print(f"Wrote calibrated metric DSM -> {dsm_path}")
else:
    dsm_export["reason"] = (
        f"{tfw_path} not present; refusing to write a metric GeoTIFF with a guessed "
        "georeference. Extract Vaihingen_dsm_tiles_geoinfo.zip (see 02_DATA_AUDIT.md)."
    )
    print(f"Skipped calibrated DSM export: {dsm_export['reason']}")

# seam discontinuity metric: local gradient magnitude AT hard-tile-boundary rows/cols
# vs elsewhere, for the NON-blended (hard) reconstruction -- this is what a naive
# "just paste tiles together" implementation would look like, to quantify why blending matters
gy, gx = np.gradient(pred_hard_cal.astype(np.float64))
grad_mag = np.sqrt(gx**2 + gy**2)
boundary_mask = np.zeros((H, W), dtype=bool)
for y in range(WIN, H - WIN + 1, WIN):
    boundary_mask[max(0, y-2):y+2, :] = True
for x in range(WIN, W - WIN + 1, WIN):
    boundary_mask[:, max(0, x-2):x+2] = True
boundary_mask &= hard_valid
interior_mask = hard_valid & ~boundary_mask
seam_grad_mean = float(grad_mag[boundary_mask].mean()) if boundary_mask.any() else None
interior_grad_mean = float(grad_mag[interior_mask].mean()) if interior_mask.any() else None

# same seam check for the BLENDED reconstruction
gy2, gx2 = np.gradient(pred_full_cal.astype(np.float64))
grad_mag2 = np.sqrt(gx2**2 + gy2**2)
seam_grad_mean_blend = float(grad_mag2[boundary_mask].mean()) if boundary_mask.any() else None
interior_grad_mean_blend = float(grad_mag2[interior_mask].mean()) if interior_mask.any() else None

result = {
    "area": AREA, "tile_shape": [H, W], "window": WIN, "stride": STRIDE,
    "n_windows_blended": n_windows, "n_windows_hard": int(count_hard.sum() / (WIN*WIN)),
    "time_s_blended_pass": dt, "peak_mem_MB": mem_peak / 1e6,
    "blended_metrics": m_blend, "hard_paste_metrics": m_hard,
    "seam_gradient_hard_paste": {"boundary_mean": seam_grad_mean, "interior_mean": interior_grad_mean,
                                    "ratio": (seam_grad_mean / interior_grad_mean) if interior_grad_mean else None},
    "seam_gradient_blended": {"boundary_mean": seam_grad_mean_blend, "interior_mean": interior_grad_mean_blend,
                                "ratio": (seam_grad_mean_blend / interior_grad_mean_blend) if interior_grad_mean_blend else None},
    "calibrated_dsm_export": dsm_export,
}
print(json.dumps({k: v for k, v in result.items() if k not in ("blended_metrics", "hard_paste_metrics")}, indent=2))
print("blended RMSE:", m_blend["rmse"], " hard-paste RMSE:", m_hard["rmse"])

with open(OUT_DIR / OUT_NAME, "w") as f:
    json.dump(result, f, indent=2)

# save a quick comparison figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(gt_full, cmap="terrain"); axes[0].set_title("Ground truth DSM")
axes[1].imshow(pred_hard_cal, cmap="terrain"); axes[1].set_title("Hard-paste (no blend) reconstruction")
axes[2].imshow(pred_full_cal, cmap="terrain"); axes[2].set_title("Feather-blended reconstruction")
for ax in axes: ax.axis("off")
plt.tight_layout()
FIG_SUFFIX = "" if OUT_NAME == "K1_tiling_stitch.json" else "_" + Path(OUT_NAME).stem.replace("K1_tiling_stitch_", "")
plt.savefig(FIG_DIR / f"K1_tiling_area{AREA}{FIG_SUFFIX}.png", dpi=110)
print(f"Saved -> {OUT_DIR / OUT_NAME} and figure")
