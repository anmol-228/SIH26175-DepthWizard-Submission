"""
Build the P-18 (sparse/desert) test-split patch for Issue #48.

Pairs the provenance-confirmed Mojave Sentinel-2 RGB crop (already produced
by PHASE2_RED_FLAGS/data/p0_a4_sparse_desert/raw/crop_sparse_desert.py) with
a mosaicked + resampled DSM built from Nirmalya's confirmed NV_Southern_D23
1m-DEM tiles, and appends the resulting entry to
data/processed/patches/manifest.json so scripts/07_calibrate_and_eval.py
picks it up as a normal "test" split sample.

Run from the repo root:
    python scripts/build_p18_patch.py
"""
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.warp import reproject, transform_bounds

# ---- Known-good values, verified on Issue #48 ------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
RGB_CROP_TIF = (
    REPO_ROOT / "PHASE2_RED_FLAGS" / "data" / "p0_a4_sparse_desert" / "derived"
    / "S2A_11SNA_20210927_crop1536_r4722_c4722.tif"
)
PATCH_DIR = REPO_ROOT / "data" / "processed" / "patches"
MANIFEST_PATH = PATCH_DIR / "manifest.json"

# File-level hash, matching how crop_sparse_desert.py computed it (hash of the
# written GeoTIFF's bytes, not of a raw in-memory array).
EXPECTED_RGB_FILE_SHA256 = "265e59b9fbc0bba69e8710bbf4724452dd6b09bdcda9f85842cfd4682676833a"

DEM_TILE_URLS = [
    f"https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1m/Projects/"
    f"NV_Southern_D23/TIFF/USGS_1M_11_x{col}y{row}_NV_Southern_D23.tif"
    for col in (54, 55, 56)
    for row in (404, 405, 406)
]

DEM_CACHE_DIR = (
    REPO_ROOT / "PHASE2_RED_FLAGS" / "data" / "p0_a4_sparse_desert" / "dem_tiles_cache"
)

AREA_NAME = "mojave_sparse_desert_p18"
OUT_NPZ_NAME = f"{AREA_NAME}.npz"


def download_tiles_locally():
    """Download each DEM tile fully via curl instead of reading it in ranges
    over /vsicurl/. GDAL's windowed vsicurl reads were getting truncated
    mid-transfer on this network (TIFFFillTile: got fewer bytes than
    expected) even with retries configured - a full sequential curl download
    is far more reliable, and -C - makes a rerun resume rather than restart.
    """
    DEM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    local_paths = []
    for url in DEM_TILE_URLS:
        name = url.rsplit("/", 1)[-1]
        dest = DEM_CACHE_DIR / name
        print(f"Fetching {name} ...")
        subprocess.run(
            ["curl", "-fSL", "--retry", "5", "--retry-delay", "3", "-C", "-", "-o", str(dest), url],
            check=True,
        )
        local_paths.append(dest)
    return local_paths


def load_verified_rgb_crop():
    """Load the existing derived RGB crop, verifying the file hash first."""
    digest = hashlib.sha256(RGB_CROP_TIF.read_bytes()).hexdigest()
    if digest != EXPECTED_RGB_FILE_SHA256:
        raise RuntimeError(
            f"RGB crop file hash mismatch: got {digest}, expected {EXPECTED_RGB_FILE_SHA256}. "
            f"{RGB_CROP_TIF} may have changed since crop_sparse_desert.py produced it."
        )
    with rasterio.open(RGB_CROP_TIF) as src:
        rgb = src.read()                    # (C, H, W)
        crop_transform = src.transform
        crop_crs = src.crs
        crop_size = src.width               # square crop, width == height
        pixel_size = src.transform.a        # metres/pixel, x-direction
    rgb = np.moveaxis(rgb, 0, -1)           # -> (H, W, C)
    return rgb, crop_transform, crop_crs, crop_size, pixel_size


def build_mosaicked_dsm(crop_transform, crop_crs, crop_size, pixel_size):
    """Mosaic the 9 confirmed tiles (from local disk) and resample onto the
    RGB crop's exact grid.

    Per Nirmalya's review on PR #54: reads each tile's *actual* CRS/nodata
    instead of assuming they match the RGB crop's, and passes nodata through
    merge()/reproject() explicitly so Resampling.average can't silently blend
    nodata sentinels into real elevations.
    """
    left, top = crop_transform.c, crop_transform.f
    right = left + crop_size * pixel_size
    bottom = top - crop_size * pixel_size

    local_paths = download_tiles_locally()
    srcs = [rasterio.open(p) for p in local_paths]
    tile_crs = srcs[0].crs        # actual DEM CRS - do not assume it matches crop_crs
    tile_nodata = srcs[0].nodata  # actual DEM nodata sentinel (e.g. -999999.0)
    try:
        # bounds must be expressed in the tiles' own CRS, not the RGB crop's
        tile_bounds = transform_bounds(crop_crs, tile_crs, left, bottom, right, top)
        mosaic, mosaic_transform = merge(srcs, bounds=tile_bounds, nodata=tile_nodata)
    finally:
        for s in srcs:
            s.close()

    fill = tile_nodata if tile_nodata is not None else np.nan
    dsm = np.full((crop_size, crop_size), fill, dtype=np.float32)
    reproject(
        source=mosaic[0],
        destination=dsm,
        src_transform=mosaic_transform,
        src_crs=tile_crs,
        src_nodata=tile_nodata,
        dst_transform=crop_transform,
        dst_crs=crop_crs,
        dst_nodata=tile_nodata,
        resampling=Resampling.average,   # 1m -> 10m, excluding nodata source pixels
    )

    if tile_nodata is not None and np.any(dsm == tile_nodata):
        raise RuntimeError(
            f"DSM mosaic still contains nodata ({tile_nodata}) after reprojection over the "
            "crop window - tile coverage doesn't actually fully span the crop. Stop here and "
            "report INCONCLUSIVE per Anmol's time-box rather than patching around it."
        )
    if not np.isfinite(dsm).all() or (dsm == 0).all():
        raise RuntimeError(
            "DSM mosaic has gaps / is empty over the crop window - tile coverage or the "
            "merge bounds don't actually line up. Stop here and report INCONCLUSIVE per "
            "Anmol's time-box rather than patching around it."
        )
    return dsm


def main():
    test_dir = PATCH_DIR / "test"
    test_dir.mkdir(parents=True, exist_ok=True)

    rgb, crop_transform, crop_crs, crop_size, pixel_size = load_verified_rgb_crop()
    dsm = build_mosaicked_dsm(crop_transform, crop_crs, crop_size, pixel_size)

    out_path = test_dir / OUT_NPZ_NAME
    np.savez(out_path, rgb=rgb, dsm=dsm)
    print(f"Wrote {out_path} - rgb {rgb.shape} {rgb.dtype}, dsm {dsm.shape} {dsm.dtype}")

    manifest = json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else []
    manifest = [e for e in manifest if e.get("area") != AREA_NAME]  # replace prior run, no dupes
    manifest.append({"split": "test", "file": OUT_NPZ_NAME, "area": AREA_NAME, "y": 0, "x": 0})
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"Appended entry to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
