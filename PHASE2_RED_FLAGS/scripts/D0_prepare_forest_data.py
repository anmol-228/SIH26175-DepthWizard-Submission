"""Experiment D, step 0: build a genuinely-forested RGB + canopy-height-model
(CHM) evaluation/training dataset, real data only, no fabrication.

Data sources (both CC-licensed, no account/registration required):
  - RGB: Sentinel-2 L2A true-color (TCI) COG, scene S2A_10TDT_20210902_0_L2A,
    cloud cover 0.17%, 10m native GSD, UTM zone 10N (EPSG:32610), downloaded
    from the public AWS Open Data "sentinel-cogs" bucket via the Earth Search
    STAC API (https://earth-search.aws.element84.com/v1). No auth.
  - Canopy height: ETH Global Canopy Height 2020 (Lang, Jetz, Schindler,
    Wegner, Nature Ecology & Evolution 2023), 10m nominal GSD, EPSG:4326,
    CC-BY-4.0, tile N45W126, downloaded directly from the public ETH
    Research Collection mirror (libdrive.ethz.ch). No auth.
  Area of interest: western Olympic National Park lowland river valleys,
  Washington, USA (genuine dense old-growth temperate rainforest -- NOT
  urban/garden trees). lon [-124.05, -123.65], lat [47.55, 47.83].

NEON (the originally planned source) was found to now require account
registration + API token ("as of June 2026" per NEON's own tutorial pages,
confirmed by live 403 Access Denied on data.neonscience.org/api/v0/data/...
from both curl and a real browser -- account creation is a prohibited action
for this agent). This is documented honestly in 06_VEGETATION_FOREST_TEST.md;
this script implements the substitute real-data pipeline instead.

Why this pairing is valid ground truth: the ETH map is itself trained against
sparse real spaceborne LiDAR (GEDI) footprints and published/peer-reviewed;
it is independent of Depth Anything V2 (never seen by DA2's training data)
and independent of the Sentinel-2 RGB pixels DA2 will see (DA2 is a
monocular RGB model, not multi-band). This mirrors the same
"independent-ground-truth" spirit as Vaihingen's image-matching DSM in Phase 1,
with the caveat -- disclosed in the writeup -- that the CHM's own OWN
training used Sentinel-2 reflectance features, so it is not a fully
independent-of-imagery measurement. (Vaihingen's DSM is itself photogrammetric,
not airborne LiDAR, so it carries a related caveat rather than being a clean
counter-example.) This is flagged, not hidden.
"""
import json
from pathlib import Path

import numpy as np
import tifffile
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "forest_eth_s2" / "raw"
OUT = ROOT / "data" / "forest_eth_s2" / "patches"
OUT.mkdir(parents=True, exist_ok=True)
FIG = ROOT / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

TCI = RAW / "S2A_10TDT_20210902_TCI.tif"
SCL = RAW / "S2A_10TDT_20210902_SCL.tif"
CHM = RAW / "ETH_GlobalCanopyHeight_10m_2020_N45W126_Map.tif"

# ---- Sentinel-2 geotransform (from GeoTIFF tags, verified in-session) ----
S2_OX, S2_OY = 399960.0, 5300040.0   # UTM10N (EPSG:32610) upper-left corner
S2_PX = 10.0                          # metres/pixel, north-up

# ---- ETH CHM geotransform ----
CHM_OX, CHM_OY = -126.0, 48.0         # EPSG:4326 upper-left corner (lon, lat)
CHM_PX = 8.333333333333333e-05        # degrees/pixel (~9.26m N-S at this latitude)
CHM_NODATA = 255

# AOI: western Olympic NP lowland valleys (Hoh / Bogachiel / Queets drainages)
LAT0, LAT1 = 47.55, 47.83
LON0, LON1 = -124.05, -123.65

PATCH = 512

# SCL classes considered genuinely INVALID for evaluation (cloud/defect/nodata)
# 0=no data, 1=saturated/defective, 3=cloud shadow, 7=unclassified,
# 8/9=cloud medium/high probability, 10=thin cirrus
SCL_INVALID = {0, 1, 3, 7, 8, 9, 10}


def main():
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32610", always_xy=True)
    to_wgs = Transformer.from_crs("EPSG:32610", "EPSG:4326", always_xy=True)

    corners = [(LON0, LAT0), (LON1, LAT0), (LON0, LAT1), (LON1, LAT1)]
    cols, rows = [], []
    for lon, lat in corners:
        x, y = to_utm.transform(lon, lat)
        cols.append((x - S2_OX) / S2_PX)
        rows.append((S2_OY - y) / S2_PX)
    c0, c1 = int(min(cols)), int(max(cols))
    r0, r1 = int(min(rows)), int(max(rows))
    print(f"AOI in Sentinel-2 10m grid: cols [{c0},{c1}) rows [{r0},{r1}) "
          f"-> {c1-c0} x {r1-r0} px (~{(c1-c0)*10/1000:.1f} x {(r1-r0)*10/1000:.1f} km)")

    print("Loading Sentinel-2 TCI (RGB) ...")
    tci_full = tifffile.imread(str(TCI))  # (10980,10980,3) uint8
    rgb = tci_full[r0:r1, c0:c1, :].copy()
    del tci_full

    print("Loading Sentinel-2 SCL (scene classification) ...")
    scl_full = tifffile.imread(str(SCL))  # (5490,5490) uint8, 20m
    # SCL is at 20m (half resolution) -- nearest-upsample to the 10m grid
    scl_sub = scl_full[r0 // 2:r1 // 2 + 1, c0 // 2:c1 // 2 + 1]
    scl_up = np.repeat(np.repeat(scl_sub, 2, axis=0), 2, axis=1)[: (r1 - r0), : (c1 - c0)]
    del scl_full

    print("Loading ETH Canopy Height Model tile (this is the big one, ~1.3GB decoded) ...")
    chm_full = tifffile.imread(str(CHM))  # (36000,36000) uint8, EPSG:4326
    print("  CHM tile loaded:", chm_full.shape, chm_full.dtype)

    H, W = r1 - r0, c1 - c0
    print(f"Building CHM->UTM10m-grid resample via pyproj (nearest), {H}x{W} = {H*W:,} px ...")
    # target grid pixel CENTRES in UTM, vectorised
    rr, cc = np.meshgrid(np.arange(r0, r1), np.arange(c0, c1), indexing="ij")
    x = S2_OX + (cc + 0.5) * S2_PX
    y = S2_OY - (rr + 0.5) * S2_PX
    lon, lat = to_wgs.transform(x.ravel(), y.ravel())
    lon = np.asarray(lon).reshape(H, W)
    lat = np.asarray(lat).reshape(H, W)
    chm_col = np.round((lon - CHM_OX) / CHM_PX).astype(np.int64)
    chm_row = np.round((CHM_OY - lat) / CHM_PX).astype(np.int64)
    chm_col = np.clip(chm_col, 0, chm_full.shape[1] - 1)
    chm_row = np.clip(chm_row, 0, chm_full.shape[0] - 1)
    chm_resamp = chm_full[chm_row, chm_col].astype(np.float32)
    del chm_full

    valid = (chm_resamp != CHM_NODATA) & (~np.isin(scl_up, list(SCL_INVALID)))
    forest_mask = (chm_resamp > 2.0) & valid  # canopy-present submask (>2m)

    print(f"Valid px: {100*valid.mean():.2f}%  Forest(>2m) px: {100*forest_mask.mean():.2f}%  "
          f"of valid: {100*forest_mask.sum()/max(1,valid.sum()):.2f}%")
    print(f"CHM (valid px) mean={chm_resamp[valid].mean():.2f}m median={np.median(chm_resamp[valid]):.2f}m "
          f"p95={np.percentile(chm_resamp[valid],95):.2f}m max={chm_resamp[valid].max():.1f}m")

    # ---- slice into non-overlapping 512x512 patches ----
    n_row_blocks = H // PATCH
    n_col_blocks = W // PATCH
    print(f"Grid of patches: {n_row_blocks} row-blocks x {n_col_blocks} col-blocks = {n_row_blocks*n_col_blocks} candidate patches")

    manifest = []
    kept, dropped_invalid = 0, 0
    for rb in range(n_row_blocks):
        for cb in range(n_col_blocks):
            pr0, pc0 = rb * PATCH, cb * PATCH
            rgb_p = rgb[pr0:pr0 + PATCH, pc0:pc0 + PATCH, :]
            chm_p = chm_resamp[pr0:pr0 + PATCH, pc0:pc0 + PATCH]
            valid_p = valid[pr0:pr0 + PATCH, pc0:pc0 + PATCH]
            frac_valid = valid_p.mean()
            if frac_valid < 0.98:
                dropped_invalid += 1
                continue
            frac_forest = ((chm_p > 2.0) & valid_p).sum() / valid_p.sum()
            # spatial block split: row-bands, NOT random -- keeps train/val/test
            # geographically disjoint the same way Vaihingen used separate areas.
            if rb <= n_row_blocks - 3:
                split = "train"
            elif rb == n_row_blocks - 2:
                split = "val"
            else:
                split = "test"
            fname = f"patch_r{rb:02d}_c{cb:02d}.npz"
            np.savez_compressed(
                OUT / fname,  # flat dir, split tracked in manifest.json
                rgb=rgb_p, chm=chm_p, valid=valid_p,
            )
            manifest.append({
                "file": fname, "split": split, "row_block": rb, "col_block": cb,
                "frac_valid": float(frac_valid), "frac_forest_gt2m": float(frac_forest),
                "mean_chm": float(chm_p[valid_p].mean()),
            })
            kept += 1

    print(f"Kept {kept} patches (dropped {dropped_invalid} for <98% valid px)")
    by_split = {}
    for m in manifest:
        by_split.setdefault(m["split"], 0)
        by_split[m["split"]] += 1
    print("Split counts:", by_split)

    with open(OUT / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    meta = {
        "rgb_source": "Sentinel-2 L2A TCI, scene S2A_10TDT_20210902_0_L2A, cloud_cover=0.169%, "
                       "10m GSD, EPSG:32610 (UTM zone 10N), acquired 2021-09-02, via AWS Open Data "
                       "sentinel-cogs bucket (public, no auth), queried via earth-search STAC API",
        "chm_source": "ETH Global Canopy Height 2020 (Lang et al. 2023, Nature Ecology & Evolution), "
                       "tile N45W126, 10m nominal GSD (~9.26m at this latitude), EPSG:4326, CC-BY-4.0, "
                       "downloaded from ETH Research Collection mirror (libdrive.ethz.ch), no auth",
        "aoi": {"lat": [LAT0, LAT1], "lon": [LON0, LON1],
                "description": "western Olympic National Park lowland river valleys (Hoh/Bogachiel/Queets "
                                "drainages), Washington, USA -- genuine dense old-growth temperate rainforest"},
        "crs_working_grid": "EPSG:32610 (UTM zone 10N), 10m/pixel -- RGB kept at native Sentinel-2 grid, "
                             "CHM nearest-resampled onto this grid via pyproj",
        "gsd_m": 10.0,
        "patch_px": PATCH,
        "n_patches": kept,
        "by_split": by_split,
        "chm_nodata_code": CHM_NODATA,
        "valid_px_definition": "CHM != 255 (nodata) AND SCL not in cloud/shadow/defect/nodata classes",
        "global_valid_frac": float(valid.mean()),
        "global_forest_gt2m_frac_of_valid": float(forest_mask.sum() / max(1, valid.sum())),
        "chm_stats_valid_px": {
            "mean": float(chm_resamp[valid].mean()),
            "median": float(np.median(chm_resamp[valid])),
            "p95": float(np.percentile(chm_resamp[valid], 95)),
            "max": float(chm_resamp[valid].max()),
        },
    }
    with open(OUT / "dataset_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved manifest + dataset_meta.json ->", OUT)

    # ---- visual sanity-check figure: RGB thumbnail + CHM heatmap overlay ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        step = 8
        rgb_small = rgb[::step, ::step, :]
        chm_small = chm_resamp[::step, ::step].copy()
        chm_small[~valid[::step, ::step]] = np.nan

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(rgb_small)
        axes[0].set_title("Sentinel-2 RGB (true colour), full AOI")
        im = axes[1].imshow(chm_small, cmap="viridis", vmin=0, vmax=70)
        axes[1].set_title("ETH Canopy Height Model (m), resampled to S2 grid")
        plt.colorbar(im, ax=axes[1], fraction=0.046)
        axes[2].imshow(rgb_small)
        axes[2].imshow(chm_small, cmap="viridis", alpha=0.55, vmin=0, vmax=70)
        axes[2].set_title("Overlay (alignment sanity check)")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(FIG / "D0_aoi_rgb_chm_overlay.png", dpi=130)
        plt.close()
        print("Saved overlay figure -> outputs/figures/D0_aoi_rgb_chm_overlay.png")
    except Exception as e:
        print("Figure generation failed (non-fatal):", e)


if __name__ == "__main__":
    main()
