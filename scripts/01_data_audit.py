"""Experiment 0 + Data Validation Gate.

Reads every TOP/DSM tile pair, checks:
- dimensions match
- pixel size (GSD) matches
- DSM value range is physically plausible
- nodata handling
Writes a JSON summary and saves side-by-side visualizations for a few tiles.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from geoio import load_geotiff, dsm_nodata_mask

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "Vaihingen"
TOP_DIR = RAW / "top"
DSM_DIR = RAW / "dsm"
OUT_FIG = ROOT / "outputs" / "figures"
OUT_METRICS = ROOT / "outputs" / "metrics"
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_METRICS.mkdir(parents=True, exist_ok=True)

AREAS = sorted(int(p.stem.split("area")[-1]) for p in DSM_DIR.glob("dsm_09cm_matching_area*.tif"))
print(f"Found {len(AREAS)} DSM tiles: {AREAS}")

records = []
for area in AREAS:
    top_path = TOP_DIR / f"top_mosaic_09cm_area{area}.tif"
    dsm_path = DSM_DIR / f"dsm_09cm_matching_area{area}.tif"
    if not top_path.exists():
        print(f"AREA {area}: MISSING top file -- SKIP")
        continue
    top = load_geotiff(top_path)
    dsm = load_geotiff(dsm_path)

    top_h, top_w = top.array.shape[:2]
    dsm_h, dsm_w = dsm.array.shape[:2]

    valid_mask = dsm_nodata_mask(dsm.array)
    valid_vals = dsm.array[valid_mask]

    rec = {
        "area": area,
        "top_shape": [top_h, top_w, top.array.shape[2] if top.array.ndim == 3 else 1],
        "dsm_shape": [dsm_h, dsm_w],
        "top_gsd": round(top.gsd, 4),
        "dsm_gsd": round(dsm.gsd, 4),
        "dims_match": (top_h == dsm_h) and (top_w == dsm_w),
        "gsd_match": abs(top.gsd - dsm.gsd) < 1e-3,
        "origin_match": (abs(top.origin_x - dsm.origin_x) < 1.0) and (abs(top.origin_y - dsm.origin_y) < 1.0),
        "dsm_dtype": str(dsm.array.dtype),
        "n_valid_px": int(valid_mask.sum()),
        "n_total_px": int(dsm.array.size),
        "valid_frac": round(float(valid_mask.sum()) / dsm.array.size, 4),
        "dsm_min": float(np.nanmin(valid_vals)) if valid_vals.size else None,
        "dsm_max": float(np.nanmax(valid_vals)) if valid_vals.size else None,
        "dsm_mean": float(np.nanmean(valid_vals)) if valid_vals.size else None,
        "dsm_median": float(np.nanmedian(valid_vals)) if valid_vals.size else None,
        "dsm_std": float(np.nanstd(valid_vals)) if valid_vals.size else None,
    }
    records.append(rec)
    status = "OK" if (rec["dims_match"] and rec["gsd_match"]) else "MISMATCH"
    print(f"AREA {area:>3}: top={top_h}x{top_w} dsm={dsm_h}x{dsm_w} gsd_top={rec['top_gsd']} gsd_dsm={rec['dsm_gsd']} "
          f"range=[{rec['dsm_min']:.1f},{rec['dsm_max']:.1f}] mean={rec['dsm_mean']:.2f} valid={rec['valid_frac']*100:.1f}% -> {status}")

with open(OUT_METRICS / "data_audit.json", "w") as f:
    json.dump(records, f, indent=2)

# Dataset-level stats (Experiment 0)
all_valid = []
for area in AREAS:
    dsm_path = DSM_DIR / f"dsm_09cm_matching_area{area}.tif"
    dsm = load_geotiff(dsm_path)
    mask = dsm_nodata_mask(dsm.array)
    all_valid.append(dsm.array[mask].ravel())
all_valid = np.concatenate(all_valid)
stats = {
    "n_tiles": len(AREAS),
    "n_valid_pixels_total": int(all_valid.size),
    "mean": float(np.mean(all_valid)),
    "median": float(np.median(all_valid)),
    "std": float(np.std(all_valid)),
    "min": float(np.min(all_valid)),
    "max": float(np.max(all_valid)),
    "p1": float(np.percentile(all_valid, 1)),
    "p5": float(np.percentile(all_valid, 5)),
    "p25": float(np.percentile(all_valid, 25)),
    "p75": float(np.percentile(all_valid, 75)),
    "p95": float(np.percentile(all_valid, 95)),
    "p99": float(np.percentile(all_valid, 99)),
}
print("\n=== DATASET-LEVEL DSM STATS (Experiment 0) ===")
for k, v in stats.items():
    print(f"  {k}: {v}")
with open(OUT_METRICS / "dataset_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

# Visualization for 3 sample tiles: small/medium/large elevation range
records_sorted = sorted(records, key=lambda r: (r["dsm_max"] - r["dsm_min"]))
sample_areas = [records_sorted[0]["area"], records_sorted[len(records_sorted)//2]["area"], records_sorted[-1]["area"]]
print(f"\nSaving alignment visualizations for sample areas: {sample_areas}")

for area in sample_areas:
    top = load_geotiff(TOP_DIR / f"top_mosaic_09cm_area{area}.tif")
    dsm = load_geotiff(DSM_DIR / f"dsm_09cm_matching_area{area}.tif")
    mask = dsm_nodata_mask(dsm.array)
    dsm_plot = np.where(mask, dsm.array, np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    top_rgb = top.array[..., :3] if top.array.ndim == 3 else top.array
    axes[0].imshow(top_rgb.astype(np.uint8) if top_rgb.dtype != np.uint8 else top_rgb)
    axes[0].set_title(f"Area {area} — TOP (IRRG)")
    axes[0].axis("off")
    im = axes[1].imshow(dsm_plot, cmap="terrain")
    axes[1].set_title(f"Area {area} — DSM (m)")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    # overlay: downsample both to same small grid and check edge correspondence via simple checkerboard
    h, w = dsm_plot.shape
    step_h, step_w = max(h//200,1), max(w//200,1)
    axes[2].imshow(top_rgb[::step_h, ::step_w].astype(np.uint8) if top_rgb.dtype != np.uint8 else top_rgb[::step_h,::step_w])
    axes[2].imshow(dsm_plot[::step_h, ::step_w], cmap="terrain", alpha=0.45)
    axes[2].set_title(f"Area {area} — overlay (alignment check)")
    axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(OUT_FIG / f"alignment_area{area}.png", dpi=110)
    plt.close()

print("\nDone. See outputs/metrics/data_audit.json and outputs/figures/alignment_area*.png")
