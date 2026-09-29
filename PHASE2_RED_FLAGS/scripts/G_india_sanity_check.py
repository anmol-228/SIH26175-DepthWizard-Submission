"""Experiment G: India-domain sanity check (Section 28). Explicitly NOT a
full ground-truth validation -- both required per the master brief and
because Copernicus GLO-30 (30m, ~2m LE90 real accuracy per literature) is
far coarser than a real evaluation truth source, so any RMSE computed
against it is a rough sparse-point comparison, not a real benchmark number.

Data sources (both real, both no-auth):
  - Optical RGB: ESRI World Imagery XYZ tile service (server.arcgisonline.com),
    a public no-auth tile endpoint (Esri/Maxar/DigitalGlobe basemap imagery,
    exact acquisition date/sensor not disclosed by the service -- a real
    limitation, noted below).
  - Elevation: Copernicus GLO-30 Public, AWS bucket copernicus-dem-30m,
    confirmed correct key naming this session (NOTE: the naming convention
    is [resolution]=10 for GLO-30 (10 arcsec), 30 for GLO-90 (30 arcsec) --
    inverted from what an earlier research pass assumed; corrected here
    after a direct 404 forced re-verification against the bucket's own
    readme.html).
"""
import sys, json, math, io, os
from pathlib import Path
P2ROOT = Path(__file__).resolve().parents[1]
POFROOT = P2ROOT.parent
sys.path.insert(0, str(POFROOT / "src"))
import numpy as np
import torch
import urllib.request
import tifffile
from PIL import Image
from da2_utils import preprocess_rgb, load_model, resize_pred_to
from metrics import all_metrics, fit_scale_shift

OUT_DATA = P2ROOT / "data" / "india_sanity"
OUT_DATA.mkdir(parents=True, exist_ok=True)
OUT_FIG = P2ROOT / "outputs" / "figures"
OUT_METRICS = P2ROOT / "outputs" / "metrics"
device = "cuda" if torch.cuda.is_available() else "cpu"

REGIONS = {
    "urban_bengaluru": (12.9716, 77.5946),
    "hilly_coorg": (12.42, 75.74),
    "forested_west_ghats": (10.12, 77.03),
}

Z = 17
GRID = 4  # GRID x GRID tiles = one mosaic

def latlon_to_tile(lat, lon, z):
    lat_rad = math.radians(lat)
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n
    return x, y

def tile_to_latlon(x, y, z):
    n = 2 ** z
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    return math.degrees(lat_rad), lon

def fetch_esri_mosaic(lat, lon, z, grid):
    cx, cy = latlon_to_tile(lat, lon, z)
    x0, y0 = int(cx) - grid // 2, int(cy) - grid // 2
    mosaic = Image.new("RGB", (256 * grid, 256 * grid))
    for i in range(grid):
        for j in range(grid):
            tx, ty = x0 + i, y0 + j
            url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{ty}/{tx}"
            data = urllib.request.urlopen(url, timeout=15).read()
            tile_img = Image.open(io.BytesIO(data)).convert("RGB")
            mosaic.paste(tile_img, (i * 256, j * 256))
    lat_tl, lon_tl = tile_to_latlon(x0, y0, z)
    lat_br, lon_br = tile_to_latlon(x0 + grid, y0 + grid, z)
    return np.array(mosaic), (lat_tl, lon_tl, lat_br, lon_br)

def fetch_glo30_tile(lat, lon):
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    latf = f"{ns}{abs(int(math.floor(lat))):02d}_00"
    lonf = f"{ew}{abs(int(math.floor(lon))):03d}_00"
    key = f"Copernicus_DSM_COG_10_{latf}_{lonf}_DEM"
    url = f"https://copernicus-dem-30m.s3.amazonaws.com/{key}/{key}.tif"
    local = OUT_DATA / f"{key}_DEM.tif"
    if not local.exists():
        urllib.request.urlretrieve(url, str(local))
    return local

def crop_glo30_to_bbox(tif_path, bbox):
    lat_tl, lon_tl, lat_br, lon_br = bbox
    dem = tifffile.imread(str(tif_path)).astype(np.float32)  # (3600, 3600), covers 1x1 deg, N..N+1, E..E+1
    H, W = dem.shape
    lat0 = math.floor(lat_tl)  # tile southwest-corner convention per readme (northing = tile's integer degree)
    lon0 = math.floor(lon_tl)
    # row 0 = north edge (lat0+1), row H-1 = south edge (lat0); col 0 = west edge (lon0)
    def latlon_to_px(lat, lon):
        row = (lat0 + 1 - lat) / 1.0 * H
        col = (lon - lon0) / 1.0 * W
        return row, col
    r_tl, c_tl = latlon_to_px(lat_tl, lon_tl)
    r_br, c_br = latlon_to_px(lat_br, lon_br)
    r0, r1 = int(max(0, min(r_tl, r_br))), int(min(H, max(r_tl, r_br)))
    c0, c1 = int(max(0, min(c_tl, c_br))), int(min(W, max(c_tl, c_br)))
    return dem[r0:r1, c0:c1]

model = load_model(device)
CKPT = POFROOT / "checkpoints" / "da2_small_finetuned_iter2_partial_unfreeze.pt"
model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
model.eval()

results = {}
for name, (lat, lon) in REGIONS.items():
    print(f"\n=== {name} ({lat},{lon}) ===")
    try:
        rgb, bbox = fetch_esri_mosaic(lat, lon, Z, GRID)
        print(f"  fetched ESRI mosaic {rgb.shape}, bbox {bbox}")
        glo30_path = fetch_glo30_tile(lat, lon)
        dem_crop = crop_glo30_to_bbox(glo30_path, bbox)
        print(f"  GLO-30 crop shape {dem_crop.shape}, elev range {dem_crop.min():.1f}-{dem_crop.max():.1f}m")

        # resize DEM crop to match RGB mosaic pixel grid (nearest, since DEM is much coarser -- no
        # interpolation-invented detail, just a coordinate-aligned coarse reference grid)
        dem_img = Image.fromarray(dem_crop).resize((rgb.shape[1], rgb.shape[0]), Image.Resampling.NEAREST)
        dem_aligned = np.array(dem_img)

        # run model on 512-center-crop(s) of the mosaic (mosaic is 1024x1024 at grid=4,z=17)
        H, W = rgb.shape[:2]
        crop = min(512, H, W)
        y0, x0 = (H - crop) // 2, (W - crop) // 2
        rgb_crop = rgb[y0:y0+crop, x0:x0+crop]
        dem_crop_aligned = dem_aligned[y0:y0+crop, x0:x0+crop]

        with torch.no_grad():
            x = preprocess_rgb(rgb_crop).unsqueeze(0).to(device)
            pred = resize_pred_to(model(pixel_values=x).predicted_depth, rgb_crop.shape[:2]).squeeze(0).cpu().numpy()

        # sparse "calibration" using GLO-30 as the reference (this is exactly Section 28's
        # "sparse-point error / consistency" framing -- GLO-30 is itself only ~2m-accurate and
        # 30m/px, so this is NOT a precision benchmark, just a rough consistency check)
        valid = np.isfinite(dem_crop_aligned) & (dem_crop_aligned > -400)
        if valid.sum() > 10:
            a, b = fit_scale_shift(pred[valid], dem_crop_aligned[valid])
            pred_cal = a * pred + b
            m = all_metrics(pred_cal[valid], dem_crop_aligned[valid])
        else:
            m = None

        # save qualitative figure
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(rgb_crop); axes[0].set_title(f"{name}: ESRI RGB")
        axes[1].imshow(dem_crop_aligned, cmap="terrain"); axes[1].set_title("GLO-30 elevation (coarse, real)")
        axes[2].imshow(pred, cmap="terrain"); axes[2].set_title("Model prediction (relative)")
        for ax in axes: ax.axis("off")
        plt.tight_layout()
        fig_path = OUT_FIG / f"G_india_{name}.png"
        plt.savefig(fig_path, dpi=110)
        plt.close(fig)

        results[name] = {
            "lat": lat, "lon": lon, "bbox": bbox,
            "glo30_elev_range_m": [float(dem_crop.min()), float(dem_crop.max())],
            "pred_relative_range": [float(pred.min()), float(pred.max())],
            "n_valid_ref_px": int(valid.sum()),
            "rough_metrics_vs_glo30": m,
            "structural_flatness_check": "prediction NOT degenerate (has real spatial variance)" if pred.std() > 0.5 else "WARNING: prediction is nearly flat/degenerate",
            "pred_std": float(pred.std()),
            "figure": str(fig_path),
            "status": "OK",
        }
        print(f"  pred std={pred.std():.3f}  rough_metrics={m}")
    except Exception as e:
        results[name] = {"lat": lat, "lon": lon, "status": "FAILED", "error": str(e)}
        print(f"  FAILED: {e}")

with open(OUT_METRICS / "G_india_sanity_check.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved -> {OUT_METRICS / 'G_india_sanity_check.json'}")
