"""Experiment E, step 1: crop + co-register the real hilly-terrain DEM and
NAIP RGB tiles onto a single shared 1m grid, and validate alignment visually.

Both source rasters are confirmed (E_00 log / this script's own tag dump) to
be in the SAME native CRS -- EPSG:26910 (NAD83 / UTM zone 10N) -- so no
reprojection/datum transform is needed, only a crop + resample from NAIP's
native 0.6m grid onto the DEM's native 1m grid (bilinear, via
scipy.ndimage.map_coordinates). This avoids Vaihingen's mistake (Phase-1 data
turned out to have NO trustworthy geolocation at all): here every step is
done directly from each file's own embedded GeoTIFF tags, and the final
result is checked for self-consistency by converting corner pixels to
lat/lon (pyproj) and confirming they land on the real Mount Adams (WA)
location before anything downstream trusts them.

AOI (chosen as the largest fully-valid rectangle inside the DEM tile that is
ALSO fully covered by a single NAIP quarter-quad, to avoid a 4-tile NAIP
mosaic): UTM26910 X 596700..598900 (2200 m), Y 5084100..5088100 (4000 m).
"""
import sys, json
from pathlib import Path
import numpy as np
import tifffile
from scipy.ndimage import map_coordinates
from pyproj import Transformer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P2ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = P2ROOT / "data" / "hilly_terrain" / "raw"
AOI_DIR = P2ROOT / "data" / "hilly_terrain" / "aoi"
AOI_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = P2ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DEM_PATH = RAW_DIR / "USGS_1M_x59y509_MountAdams_2016.tif"
NAIP_PATH = RAW_DIR / "naip_wa_4512103_sw_2023.tif"

# Shared AOI, UTM26910 meters (X east, Y north)
AOI_XMIN, AOI_XMAX = 596700.0, 598900.0
AOI_YMIN, AOI_YMAX = 5084100.0, 5088100.0
GSD = 1.0  # output grid resolution (matches DEM native)
CRS = "EPSG:26910"


def read_geotiff_tags(path):
    t = tifffile.TiffFile(str(path))
    p = t.pages[0]
    tags = {tag.name: tag.value for tag in p.tags.values()}
    px, py, _ = tags["ModelPixelScaleTag"]
    _, _, _, ox, oy, _ = tags["ModelTiepointTag"]
    return {"pixel_x": px, "pixel_y": py, "origin_x": ox, "origin_y": oy,
            "crs_ascii": tags.get("GeoAsciiParamsTag", ""), "shape": p.shape}


if __name__ == "__main__":
    dem_meta = read_geotiff_tags(DEM_PATH)
    naip_meta = read_geotiff_tags(NAIP_PATH)
    print("DEM tags:", dem_meta)
    print("NAIP tags:", naip_meta)
    assert "UTM zone 10N" in dem_meta["crs_ascii"] and "UTM zone 10N" in naip_meta["crs_ascii"], \
        "CRS mismatch -- would need real reprojection, not implemented here"

    # ---- output grid: 1m spacing over the AOI ----
    out_w = int(round((AOI_XMAX - AOI_XMIN) / GSD))
    out_h = int(round((AOI_YMAX - AOI_YMIN) / GSD))
    xs = AOI_XMIN + (np.arange(out_w) + 0.5) * GSD
    ys = AOI_YMAX - (np.arange(out_h) + 0.5) * GSD  # row 0 = north edge
    grid_x, grid_y = np.meshgrid(xs, ys)  # (out_h, out_w)
    print(f"Output grid: {out_h} x {out_w} px at {GSD}m -> {out_w*GSD:.0f}m x {out_h*GSD:.0f}m")

    # ---- DEM: nearest-pixel crop (native res == output res) ----
    dem_full = tifffile.imread(str(DEM_PATH), key=0)
    dem_col = (grid_x - dem_meta["origin_x"]) / dem_meta["pixel_x"]
    dem_row = (dem_meta["origin_y"] - grid_y) / dem_meta["pixel_x"]
    dem_row_i = np.round(dem_row).astype(int)
    dem_col_i = np.round(dem_col).astype(int)
    assert dem_row_i.min() >= 0 and dem_row_i.max() < dem_full.shape[0]
    assert dem_col_i.min() >= 0 and dem_col_i.max() < dem_full.shape[1]
    dem_crop = dem_full[dem_row_i, dem_col_i].astype(np.float32)
    valid = dem_crop > -1000
    print(f"DEM crop valid fraction: {valid.mean():.4f}  min={dem_crop[valid].min():.1f}  "
          f"max={dem_crop[valid].max():.1f}  relief={dem_crop[valid].max()-dem_crop[valid].min():.1f}m")
    del dem_full

    # ---- NAIP: bilinear resample onto the same 1m grid ----
    naip_full = tifffile.imread(str(NAIP_PATH), key=0)  # (H, W, 4) uint8, RGBIR
    naip_col = (grid_x - naip_meta["origin_x"]) / naip_meta["pixel_x"]
    naip_row = (naip_meta["origin_y"] - grid_y) / naip_meta["pixel_x"]
    assert naip_row.min() >= 0 and naip_row.max() < naip_full.shape[0] - 1
    assert naip_col.min() >= 0 and naip_col.max() < naip_full.shape[1] - 1
    rgb_crop = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    for b in range(3):
        band = naip_full[:, :, b].astype(np.float32)
        resampled = map_coordinates(band, [naip_row, naip_col], order=1, mode="nearest")
        rgb_crop[:, :, b] = np.clip(resampled, 0, 255).astype(np.uint8)
    del naip_full

    if not valid.all():
        print(f"WARNING: {  (~valid).sum() } / {valid.size} DEM pixels invalid within AOI -- masking them out")

    # ---- sanity check: corner pixel coords -> lat/lon, must land near Mount Adams, WA ----
    to_wgs84 = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    corners = {
        "NW": (AOI_XMIN, AOI_YMAX), "NE": (AOI_XMAX, AOI_YMAX),
        "SW": (AOI_XMIN, AOI_YMIN), "SE": (AOI_XMAX, AOI_YMIN),
    }
    corner_latlon = {}
    for name, (x, y) in corners.items():
        lon, lat = to_wgs84.transform(x, y)
        corner_latlon[name] = {"lon": lon, "lat": lat}
        print(f"  corner {name}: UTM({x:.0f},{y:.0f}) -> lon={lon:.5f} lat={lat:.5f}")
    center_lon, center_lat = to_wgs84.transform((AOI_XMIN + AOI_XMAX) / 2, (AOI_YMIN + AOI_YMAX) / 2)
    print(f"AOI center: lon={center_lon:.5f} lat={center_lat:.5f} "
          f"(expect ~45.9N, -121.75W -- west flank of Mount Adams, WA, near Trout Lake)")
    assert 45.5 < center_lat < 46.3 and -122.1 < center_lon < -121.4, \
        "Center coordinate sanity check FAILED -- refusing to proceed with unverified geolocation"

    # ---- save arrays + geotransform metadata ----
    np.save(AOI_DIR / "dem_1m.npy", dem_crop)
    np.save(AOI_DIR / "dem_valid_mask.npy", valid)
    np.save(AOI_DIR / "rgb_1m.npy", rgb_crop)
    geo = {
        "crs": CRS,
        "gsd_m": GSD,
        "origin_x_utm": AOI_XMIN,  # west edge
        "origin_y_utm": AOI_YMAX,  # north edge
        "shape_hw": [out_h, out_w],
        "corner_latlon": corner_latlon,
        "center_latlon": {"lon": center_lon, "lat": center_lat},
        "dem_source": str(DEM_PATH.name),
        "naip_source": str(NAIP_PATH.name),
        "dem_valid_fraction": float(valid.mean()),
    }
    with open(AOI_DIR / "geo_meta.json", "w") as f:
        json.dump(geo, f, indent=2)
    print("Saved AOI arrays + geo_meta.json ->", AOI_DIR)

    # ---- alignment figure ----
    dem_vis = np.where(valid, dem_crop, np.nan)
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    axes[0].imshow(rgb_crop)
    axes[0].set_title("NAIP RGB (resampled to 1m)")
    im1 = axes[1].imshow(dem_vis, cmap="terrain")
    axes[1].set_title("USGS 3DEP 1m DEM (elevation, m)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    # overlay: RGB with DEM contour lines to visually confirm registration
    axes[2].imshow(rgb_crop)
    cs = axes[2].contour(dem_vis, levels=15, cmap="autumn", linewidths=0.8)
    axes[2].set_title("RGB + DEM contours overlay (alignment check)")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    plt.suptitle(f"Exp E hilly-terrain AOI: Mount Adams W flank, WA (center {center_lat:.4f}N, {-center_lon:.4f}W)\n"
                 f"{out_w}x{out_h}px @ {GSD}m -- DEM valid {valid.mean()*100:.1f}%")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "E_alignment_check.png", dpi=130)
    print("Saved alignment figure ->", FIG_DIR / "E_alignment_check.png")
