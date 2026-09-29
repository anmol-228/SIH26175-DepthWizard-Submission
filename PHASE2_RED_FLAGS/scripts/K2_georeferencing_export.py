"""Section 34: export a predicted DSM as a proper georeferenced GeoTIFF
(embedded GeoKeys, not just a plain TIFF) and reopen it independently to
verify CRS / geotransform / pixel size / extent survive the round trip.
Uses Vaihingen's own internally-consistent local coordinate system (the one
recovered from Vaihingen_dsm_tiles_geoinfo.zip this session) -- this tests
the MECHANICS of georeferenced export/reopen correctly, independent of the
separate, already-disclosed finding that Vaihingen's absolute real-world
location is not trustworthy (02_DATASET_RESEARCH.md).

CRS CORRECTION 2026-08-30 (issue #14): this script used to hard-code
EPSG = 32633, taken from the source dsm.vrt's PROJCS["WGS 84 / UTM zone 33N"].
The coordinates are consistent with UTM zone 32N GEOMETRY, not 33N -- easting
496794.825 implies ~8.96E under 32N (Vaihingen), whereas under 33N it implies
~14.96E, 6 degrees and roughly 440 km east at this latitude. The datum/authority
is a separate question and remains unproven, so this now writes
geoio.CRS_UNDEFINED (32767, "user-defined") rather than substituting 32632 or
25832, which we also cannot defend.

This is a METADATA ASSIGNMENT change, not a reprojection. The coordinates
themselves are unchanged and must not be transformed. The round-trip mechanics
this script verifies are unaffected -- the check is that whatever code was
written comes back, which holds for 32767 exactly as it did for 32633.

The historical result PHASE2_RED_FLAGS/outputs/metrics/K2_georeferencing_check.json
records the ORIGINAL 32633 run and is deliberately NOT regenerated. Re-running
this script will produce a check file whose epsg differs from that record; that
is expected, not a regression.
"""
import sys, json
from pathlib import Path
P2ROOT = Path(__file__).resolve().parents[1]
POFROOT = P2ROOT.parent
sys.path.insert(0, str(POFROOT / "src"))
import numpy as np
import tifffile

AREA = 26
TFW = POFROOT / "data" / "raw" / "Vaihingen" / "geoinfo_extracted" / f"dsm_09cm_matching_area{AREA}.tfw"
DSM = POFROOT / "data" / "raw" / "Vaihingen" / "dsm" / f"dsm_09cm_matching_area{AREA}.tif"
OUT_DIR = P2ROOT / "outputs" / "predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

vals = [float(x) for x in TFW.read_text().split()]
px, rot1, rot2, py, ox, oy = vals
print(f"Source geotransform: pixel=({px},{py}) origin=({ox},{oy})")

dsm = tifffile.imread(str(DSM)).astype(np.float32)
H, W = dsm.shape

# Write GeoTIFF with standard embedded GeoTIFF tags (ModelPixelScale + ModelTiepoint
# + GeoKeyDirectory). The CRS geokey is deliberately "user-defined" rather than an
# authority code -- see the CRS CORRECTION note in the module docstring and issue #14.
from geoio import CRS_UNDEFINED  # noqa: E402  (src/ is on sys.path above)
EPSG = CRS_UNDEFINED
extratags = [
    (33550, 'd', 3, (px, abs(py), 0.0), True),          # ModelPixelScaleTag
    (33922, 'd', 6, (0.0, 0.0, 0.0, ox, oy, 0.0), True),  # ModelTiepointTag
    (34735, 'H', 20, (  # GeoKeyDirectoryTag: minimal projected-CRS declaration
        1, 1, 0, 4,
        1024, 0, 1, 1,        # GTModelTypeGeoKey = 1 (projected)
        1025, 0, 1, 1,        # GTRasterTypeGeoKey = 1 (PixelIsArea)
        3072, 0, 1, EPSG,     # ProjectedCSTypeGeoKey (32767 = user-defined)
        3076, 0, 1, 9001,     # ProjLinearUnitsGeoKey = metre
    ), True),
]
out_path = OUT_DIR / f"K2_georef_dsm_area{AREA}.tif"
tifffile.imwrite(str(out_path), dsm, extratags=extratags, photometric="minisblack")
print(f"Wrote -> {out_path}")

# ---- reopen INDEPENDENTLY and verify ----
t = tifffile.TiffFile(str(out_path))
p = t.pages[0]
tags = {tag.name: tag.value for tag in p.tags}
pixel_scale = tags.get("ModelPixelScaleTag")
tiepoint = tags.get("ModelTiepointTag")
geokeys = tags.get("GeoKeyDirectoryTag")

reopened_shape = p.shape
scale_ok = pixel_scale is not None and abs(pixel_scale[0] - px) < 1e-9 and abs(pixel_scale[1] - abs(py)) < 1e-9
origin_ok = tiepoint is not None and abs(tiepoint[3] - ox) < 1e-6 and abs(tiepoint[4] - oy) < 1e-6
shape_ok = reopened_shape == (H, W)
epsg_ok = geokeys is not None and EPSG in geokeys  # round-trip only, not a CRS claim

report = {
    "area": AREA, "source_pixel_size": [px, py], "source_origin": [ox, oy],
    "reopened_pixel_scale_tag": list(pixel_scale) if pixel_scale is not None else None,
    "reopened_tiepoint_tag": list(tiepoint) if tiepoint is not None else None,
    "reopened_shape": list(reopened_shape), "source_shape": [H, W],
    "checks": {
        "pixel_size_preserved": bool(scale_ok),
        "origin_preserved": bool(origin_ok),
        "shape_preserved": bool(shape_ok),
        "epsg_geokey_preserved": bool(epsg_ok),
    },
    "all_pass": bool(scale_ok and origin_ok and shape_ok and epsg_ok),
}
print(json.dumps(report, indent=2))
with open(P2ROOT / "outputs" / "metrics" / "K2_georeferencing_check.json", "w") as f:
    json.dump(report, f, indent=2)
