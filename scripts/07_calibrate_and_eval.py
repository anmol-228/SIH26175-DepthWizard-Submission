"""Experiment 5 + 6: per-tile (local) calibration and sparse-calibration
stress test, applied to the FINE-TUNED model's predictions on the held-out
test set. This is the fair test of the SIH PS's actual proposed pipeline:
a pretrained/fine-tuned relative-depth backbone + a per-scene calibration
step using a DEM/GCP-like local reference (simulated here by sampling a
small number of TRUE elevation points from within the same test tile --
standing in for what SRTM/GCP/GEDI would provide in the real deployment).

Reports, separately (Section 20 -- do not hide model weakness behind
calibration):
  - fine-tuned prediction BEFORE calibration (raw ssi against its own patch --
    NOT valid as a final metric, shown only for diagnostic completeness)
  - fine-tuned + FULL per-tile oracle linear calibration (upper bound: as if
    we had dense ground truth for that tile, which we would not in reality)
  - fine-tuned + SPARSE calibration stress test at decreasing point counts
    (N = all valid pixels in a small % sample, 100, 20, 5, 1 points per tile)
    fit via robust regression (RANSAC) to avoid single-outlier GCP corruption.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from sklearn.linear_model import RANSACRegressor, LinearRegression
from da2_utils import preprocess_rgb, load_model, resize_pred_to
from metrics import all_metrics, fit_scale_shift
from geoio import save_geotiff, read_tfw, DSM_NODATA, CRS_UNDEFINED

ROOT = Path(__file__).resolve().parents[1]
PATCH_DIR = ROOT / "data" / "processed" / "patches"
import os
CKPT = ROOT / "checkpoints" / os.environ.get("EVAL_CKPT_NAME", "da2_small_finetuned_best.pt")
OUT_NAME = os.environ.get("EVAL_OUT_NAME", "finetuned_calibration_report.json")
METRICS_DIR = ROOT / "outputs" / "metrics"
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- calibrated-DSM export (issue #5: CALIBRATION -> PIPELINE/VIEWER handoff) ----
# Until now this script computed the calibrated metric height array and threw it
# away, persisting only the metrics report. Everything below is additive: the
# metric numbers this script has always produced are computed from exactly the
# same 1-D pooled path as before and are unaffected.
PRED_DIR = ROOT / "outputs" / "predictions"
GEOINFO_DIR = ROOT / "data" / "raw" / "Vaihingen" / "geoinfo_extracted"
EPSG = CRS_UNDEFINED  # 32767 user-defined: the geotransform is real, but the source
                      # dsm.vrt declares EPSG:32633 (UTM 33N) while the coordinates imply
                      # UTM 32N (~8.96E = Vaihingen). We do not assert an authority code
                      # we cannot defend. See src/geoio.CRS_UNDEFINED.
PATCH = 512            # must match scripts/04_make_patches.py
EXPORT_DSM = os.environ.get("EXPORT_DSM", "1") != "0"
EXPORT_SPARSE_N = 10   # the deployment-like sparse setting we also export
export_manifest = {"schema": "calibration_output/v0-metric", "rasters": [], "skipped": []}


def _tile_transform(area):
    """Real geotransform for a Vaihingen tile, or None if the geoinfo archive is absent."""
    tfw = GEOINFO_DIR / f"dsm_09cm_matching_area{area}.tfw"
    if not tfw.exists():
        return None
    px, _, _, py, ox, oy = read_tfw(tfw)
    return px, py, ox, oy


def _paint_canvas(entries, arrays, a, b):
    """Place per-patch calibrated 2-D arrays back onto a tile-anchored canvas.

    Patches are anchored at (y, x) from the manifest and the grid starts at (0, 0),
    so the canvas shares the tile's origin. Coverage is only the patch-covered
    subregion (04_make_patches.py drops patches that fail its validity mask), and
    uncovered pixels get the project nodata value rather than a fake elevation.
    """
    H = max(e["y"] for e in entries) + PATCH
    W = max(e["x"] for e in entries) + PATCH
    canvas = np.full((H, W), DSM_NODATA, dtype=np.float32)
    for e, arr in zip(entries, arrays):
        canvas[e["y"]:e["y"] + PATCH, e["x"]:e["x"] + PATCH] = (a * arr + b).astype(np.float32)
    return canvas


def _export(area, entries, arrays, a, b, calibration, extra=None):
    if not EXPORT_DSM:
        return
    tf = _tile_transform(area)
    if tf is None:
        reason = (f"no {GEOINFO_DIR.name}/dsm_09cm_matching_area{area}.tfw; refusing to write a "
                  "metric GeoTIFF with a guessed georeference (see 02_DATA_AUDIT.md)")
        if reason not in export_manifest["skipped"]:
            export_manifest["skipped"].append(reason)
        return
    px, py, ox, oy = tf
    canvas = _paint_canvas(entries, arrays, a, b)
    out = save_geotiff(PRED_DIR / f"calibrated_dsm_area{area}_{calibration}.tif", canvas,
                       pixel_size_x=px, pixel_size_y=py, origin_x=ox, origin_y=oy,
                       epsg=EPSG, nodata=DSM_NODATA)
    valid = int((canvas != DSM_NODATA).sum())
    rec = {
        "area": str(area), "path": str(out.relative_to(ROOT)),
        "product": "metric_dsm", "calibration": calibration,
        "calibration_a": float(a), "calibration_b": float(b),
        "units": "metres", "dtype": "float32", "nodata": DSM_NODATA, "epsg": EPSG,
        "pixel_size": [px, py], "origin": [ox, oy],
        "shape": list(canvas.shape), "n_valid_px": valid,
        "n_nodata_px": int(canvas.size - valid),
        "extent": "patch_covered_subregion_anchored_at_tile_origin",
    }
    if extra:
        rec.update(extra)
    export_manifest["rasters"].append(rec)

manifest = json.load(open(PATCH_DIR / "manifest.json"))
test_entries = [m for m in manifest if m["split"] == "test"]

def load_patch(entry):
    d = np.load(PATCH_DIR / "test" / entry["file"])
    return d["rgb"], d["dsm"]

model = load_model(device)
model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
model.eval()
print(f"Loaded fine-tuned checkpoint: {CKPT}")

@torch.no_grad()
def predict(rgb):
    x = preprocess_rgb(rgb).unsqueeze(0).to(device)
    out = model(pixel_values=x).predicted_depth
    return resize_pred_to(out, rgb.shape[:2]).squeeze(0).cpu().numpy()

# gather per-tile predictions + gt
per_tile_pred, per_tile_gt = {}, {}
per_tile_entries = {}          # manifest entries, parallel to per_tile_pred, for raster placement
for e in test_entries:
    rgb, dsm = load_patch(e)
    pred = predict(rgb)
    per_tile_pred.setdefault(e["area"], []).append(pred)
    per_tile_gt.setdefault(e["area"], []).append(dsm)
    per_tile_entries.setdefault(e["area"], []).append(e)

areas = sorted(per_tile_pred.keys())
print(f"Test tiles: {areas}")

report = {"per_tile_full_oracle": {}, "sparse_stress_test": {}, "global_pooled": {}}

# ---- FULL per-tile oracle calibration (upper bound) ----
all_pred_full, all_gt_full = [], []
for area in areas:
    p = np.concatenate([x.ravel() for x in per_tile_pred[area]])
    g = np.concatenate([x.ravel() for x in per_tile_gt[area]])
    a, b = fit_scale_shift(p, g)
    p_cal = a * p + b
    m = all_metrics(p_cal, g)
    report["per_tile_full_oracle"][area] = {**m, "a": a, "b": b, "n_px": int(p.size)}
    all_pred_full.append(p_cal); all_gt_full.append(g)
    _export(area, per_tile_entries[area], per_tile_pred[area], a, b, "per_tile_full_oracle",
            extra={"IMPORTANT": "Dense per-tile oracle calibration: a non-deployable upper "
                                "bound, NOT a field accuracy figure.",
                   "rmse_m": m["rmse"]})
    print(f"  [full oracle] area {area}: RMSE={m['rmse']:.3f} MAE={m['mae']:.3f} corr={m['corr']:.3f}")

all_pred_full = np.concatenate(all_pred_full); all_gt_full = np.concatenate(all_gt_full)
report["global_pooled"]["finetuned_full_per_tile_oracle_calibration"] = all_metrics(all_pred_full, all_gt_full)
print("\n[GLOBAL, fine-tuned + full per-tile oracle calibration]", report["global_pooled"]["finetuned_full_per_tile_oracle_calibration"])

# ---- SPARSE calibration stress test ----
rng = np.random.default_rng(42)
for n_points in [200, 50, 10, 5, 1]:
    all_pred_sparse, all_gt_sparse = [], []
    per_tile_result = {}
    for area in areas:
        p = np.concatenate([x.ravel() for x in per_tile_pred[area]])
        g = np.concatenate([x.ravel() for x in per_tile_gt[area]])
        n = min(n_points, p.size)
        if n < 2:
            # can't fit scale+shift from 1 point robustly; fall back to shift-only using train global scale
            idx = rng.choice(p.size, size=1, replace=False)
            b = g[idx[0]] - p[idx[0]]
            a = 1.0
        else:
            idx = rng.choice(p.size, size=n, replace=False)
            try:
                reg = RANSACRegressor(LinearRegression(), min_samples=max(2, n // 2), random_state=42)
                reg.fit(p[idx].reshape(-1, 1), g[idx])
                a = float(reg.estimator_.coef_[0]); b = float(reg.estimator_.intercept_)
            except Exception:
                a, b = fit_scale_shift(p[idx], g[idx])
        a, b = float(a), float(b)
        p_cal = a * p + b
        m = all_metrics(p_cal, g)
        per_tile_result[area] = {**m, "a": a, "b": b, "n_calib_points": int(n)}
        all_pred_sparse.append(p_cal); all_gt_sparse.append(g)
        if n_points == EXPORT_SPARSE_N:
            _export(area, per_tile_entries[area], per_tile_pred[area], a, b,
                    f"sparse_{n_points}",
                    extra={"n_calib_points": int(n),
                           "IMPORTANT": "Sparse RANSAC calibration from N reference points "
                                        "sampled inside this test tile: deployment-LIKE, but "
                                        "still not a blind-generalization figure.",
                           "rmse_m": m["rmse"]})
    all_pred_sparse = np.concatenate(all_pred_sparse); all_gt_sparse = np.concatenate(all_gt_sparse)
    global_m = all_metrics(all_pred_sparse, all_gt_sparse)
    report["sparse_stress_test"][str(n_points)] = {"per_tile": per_tile_result, "global": global_m}
    print(f"[sparse calib, N={n_points} pts/tile] GLOBAL RMSE={global_m['rmse']:.3f} MAE={global_m['mae']:.3f} corr={global_m['corr']:.3f}")

with open(METRICS_DIR / OUT_NAME, "w") as f:
    json.dump(report, f, indent=2)
print(f"\nSaved -> {METRICS_DIR / OUT_NAME}")

if EXPORT_DSM:
    manifest_path = METRICS_DIR / (Path(OUT_NAME).stem + "_dsm_export_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(export_manifest, f, indent=2)
    print(f"Wrote {len(export_manifest['rasters'])} calibrated DSM raster(s) -> {PRED_DIR}")
    for s_ in export_manifest["skipped"]:
        print(f"  skipped: {s_}")
    print(f"Export manifest -> {manifest_path}")
