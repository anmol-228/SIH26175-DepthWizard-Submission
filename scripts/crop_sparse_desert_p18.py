"""
Produce the Issue #48 (P-18 sparse/desert) RGB crop.

Committed per Nirmalya's PR #54 review note: this script previously only
existed locally on Adrija's machine, so nobody else could regenerate
PHASE2_RED_FLAGS/data/p0_a4_sparse_desert/derived/S2A_11SNA_20210927_crop1536_r4722_c4722.tif
(consumed by scripts/build_p18_patch.py) from the repo alone. Not a
correctness risk - build_p18_patch.py hash-checks that file against a known
SHA-256 - purely a reproducibility gap, now closed.

Input: PHASE2_RED_FLAGS/data/p0_a4_sparse_desert/raw/TCI.tif
  (Sentinel-2 L2A scene S2A_11SNA_20210927_1_L2A, AWS Open Data / Earth
  Search STAC, EPSG:32611, 10m GSD - provenance recorded on Issue #48)

Run from PHASE2_RED_FLAGS/data/p0_a4_sparse_desert/raw/:
    python ../../../../scripts/crop_sparse_desert_p18.py
"""
from pathlib import Path

import rasterio
from rasterio.windows import Window
import hashlib

src_path = "TCI.tif"
out_path = Path("../derived/S2A_11SNA_20210927_crop1536_r4722_c4722.tif")
row_start, col_start, size = 4722, 4722, 1536

with rasterio.open(src_path) as src:
    window = Window(col_start, row_start, size, size)
    data = src.read(window=window)
    transform = src.window_transform(window)
    profile = src.profile.copy()
    profile.update({"height": size, "width": size, "transform": transform})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data)

with open(out_path, "rb") as f:
    print("derived SHA-256:", hashlib.sha256(f.read()).hexdigest())

with rasterio.open(out_path) as chk:
    print("shape:", chk.shape, "count:", chk.count, "crs:", chk.crs, "transform:", chk.transform)
