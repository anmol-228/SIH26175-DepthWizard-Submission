"""Minimal GeoTIFF I/O for the Vaihingen DSM proof-of-feasibility experiment.

Deliberately avoids GDAL/rasterio (unverified on Python 3.14 at experiment time).
Uses tifffile for pixel data and manual .tfw parsing for the affine geotransform,
which is all this dataset needs (no reprojection, no CRS transforms required —
TOP and DSM tiles are already co-registered by the ISPRS release).
"""
from __future__ import annotations
import numpy as np
import tifffile
from pathlib import Path
from dataclasses import dataclass


@dataclass
class GeoRaster:
    array: np.ndarray  # (H, W) or (H, W, C)
    pixel_size_x: float
    pixel_size_y: float  # negative (north-up)
    origin_x: float
    origin_y: float
    path: str
    epsg: int | None = None
    # True when pixel size / origin came from a real GeoTIFF tag or .tfw sidecar.
    # False means load_geotiff fell back to the Vaihingen nominal values below and
    # the georeference is a guess, not data. Never export a metric GeoTIFF from a
    # raster whose georeference is a guess - see save_geotiff's require_georeferenced.
    georeferenced: bool = True

    @property
    def gsd(self) -> float:
        return abs(self.pixel_size_x)


def read_tfw(tfw_path: Path) -> tuple[float, float, float, float, float, float]:
    vals = [float(x) for x in tfw_path.read_text().split()]
    # world file order: pixel_size_x, rot_y, rot_x, pixel_size_y, origin_x, origin_y
    return tuple(vals)  # type: ignore


def read_geotiff_tags(tif_path: str | Path) -> dict:
    """Read the embedded GeoTIFF georeferencing tags written by save_geotiff / K2.

    Returns {} when the file carries no ModelPixelScale/ModelTiepoint pair. Reads
    only the minimal projected-CRS subset this project writes; it is not a general
    GeoTIFF parser.
    """
    with tifffile.TiffFile(str(tif_path)) as tf:
        tags = {t.name: t.value for t in tf.pages[0].tags}
    scale = tags.get("ModelPixelScaleTag")
    tie = tags.get("ModelTiepointTag")
    if scale is None or tie is None:
        return {}
    out = {
        "pixel_size_x": float(scale[0]),
        # GeoTIFF stores a positive Y scale; the world-file convention is negative
        # (north-up), and every consumer here expects the world-file convention.
        "pixel_size_y": -abs(float(scale[1])),
        "origin_x": float(tie[3]),
        "origin_y": float(tie[4]),
    }
    geokeys = tags.get("GeoKeyDirectoryTag")
    if geokeys is not None:
        # header is 4 shorts, then 4-short entries: (key_id, tiff_tag_loc, count, value)
        for i in range(4, len(geokeys) - 3, 4):
            if geokeys[i] == 3072:  # ProjectedCSTypeGeoKey
                out["epsg"] = int(geokeys[i + 3])
    nodata = tags.get("GDAL_NODATA")
    if nodata is not None:
        try:
            out["nodata"] = float(str(nodata).strip("\x00").strip())
        except ValueError:
            pass
    return out


def load_geotiff(tif_path: str | Path, strict: bool = False) -> GeoRaster:
    """Load a raster plus whatever georeferencing is actually available.

    Georeference is resolved in order: embedded GeoTIFF tags, then a .tfw sidecar,
    then the Vaihingen nominal fallback. The fallback is a guess and is recorded as
    georeferenced=False rather than being passed off as real. Pass strict=True to
    raise instead of guessing.
    """
    tif_path = Path(tif_path)
    arr = tifffile.imread(str(tif_path))
    epsg = None
    georeferenced = True

    embedded = read_geotiff_tags(tif_path)
    tfw_path = tif_path.with_suffix(".tfw")
    if embedded:
        px = embedded["pixel_size_x"]; py = embedded["pixel_size_y"]
        ox = embedded["origin_x"]; oy = embedded["origin_y"]
        epsg = embedded.get("epsg")
    elif tfw_path.exists():
        px, _, _, py, ox, oy = read_tfw(tfw_path)
    else:
        if strict:
            raise ValueError(
                f"{tif_path} has no embedded GeoTIFF tags and no .tfw sidecar; "
                "refusing to guess a georeference under strict=True"
            )
        # Vaihingen nominal GSD fallback. Kept for backward compatibility with the
        # Phase-1/Phase-2 scripts that rely on it, but flagged so downstream code
        # (notably save_geotiff) can refuse to treat it as a real georeference.
        px, py, ox, oy = 0.09, -0.09, 0.0, 0.0
        georeferenced = False
    return GeoRaster(array=arr, pixel_size_x=px, pixel_size_y=py, origin_x=ox, origin_y=oy,
                     path=str(tif_path), epsg=epsg, georeferenced=georeferenced)


def dsm_nodata_mask(dsm: np.ndarray, nodata_value: float = -9999.0, tol: float = 1.0) -> np.ndarray:
    """Return boolean mask of VALID pixels (True = valid)."""
    invalid = np.isnan(dsm) | (np.abs(dsm - nodata_value) < tol)
    return ~invalid


# --- Export -----------------------------------------------------------------
# The tag construction below is the one K2_georeferencing_export.py proved works
# (write + independent reopen, all four checks passing, see
# PHASE2_RED_FLAGS/outputs/metrics/K2_georeferencing_check.json). It is lifted
# here so the calibration stage can reuse it instead of the export mechanics
# living inline in a one-off stress-test script.

DSM_NODATA = -9999.0  # project-wide convention: DepthWizard__Build_Guide.pdf section 10,
                      # and the default already assumed by dsm_nodata_mask() above.

# GeoTIFF "user-defined" CRS code. Use this when the pixel grid and geotransform are
# trustworthy but no authority code can be defended.
#
# Why this is the Vaihingen default: the dataset's own dsm.vrt declares
# PROJCS["WGS 84 / UTM zone 33N"], AUTHORITY["EPSG","32633"], and that is where the 32633
# used throughout Phase 1/2 came from (PHASE2_RED_FLAGS/02_DATASET_RESEARCH.md). But the
# coordinates contradict it: easting 496794.825 sits just west of a zone central meridian,
# which under zone 32N implies longitude ~8.96E - Vaihingen. Under zone 33N the same
# easting would imply ~14.96E, roughly 600 km east. So the source metadata is mislabelled,
# which is exactly why 02_DATASET_RESEARCH.md flags Vaihingen's absolute geolocation as not
# trustworthy. Rather than swap one indefensible code for another (32632/25832 would be our
# arithmetic against the source's declaration), we assert none.
CRS_UNDEFINED = 32767


def save_geotiff(
    out_path: str | Path,
    array: np.ndarray,
    *,
    pixel_size_x: float,
    pixel_size_y: float,
    origin_x: float,
    origin_y: float,
    epsg: int,
    nodata: float | None = DSM_NODATA,
) -> Path:
    """Write a single-band float32 raster as a GeoTIFF with embedded GeoKeys.

    epsg is required and has no default: writing a georeferenced product is always
    a deliberate act, never something that happens by falling through a default.
    Pass CRS_UNDEFINED (32767) when the geotransform is real but no authority code
    can be defended - a known-wrong CRS must never be presented as a trusted one.

    Returns the path written.
    """
    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D (H, W) raster, got shape {array.shape}")
    array = array.astype(np.float32)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    extratags = [
        (33550, "d", 3, (float(pixel_size_x), abs(float(pixel_size_y)), 0.0), True),
        (33922, "d", 6, (0.0, 0.0, 0.0, float(origin_x), float(origin_y), 0.0), True),
        (34735, "H", 20, (
            1, 1, 0, 4,
            1024, 0, 1, 1,          # GTModelTypeGeoKey = projected
            1025, 0, 1, 1,          # GTRasterTypeGeoKey = PixelIsArea
            3072, 0, 1, int(epsg),  # ProjectedCSTypeGeoKey
            3076, 0, 1, 9001,       # ProjLinearUnitsGeoKey = metre
        ), True),
    ]
    if nodata is not None:
        # GDAL_NODATA (42113) is ASCII by spec. K2 never wrote this, which left the
        # nodata convention reader-side only; issue #5 closes that gap.
        nodata_str = repr(float(nodata))
        extratags.append((42113, "s", len(nodata_str) + 1, nodata_str, True))

    tifffile.imwrite(str(out_path), array, extratags=extratags, photometric="minisblack")
    return out_path


def save_dsm_geotiff(
    out_path: str | Path,
    dsm: np.ndarray,
    source: GeoRaster,
    *,
    epsg: int,
    nodata: float | None = DSM_NODATA,
    require_georeferenced: bool = True,
) -> Path:
    """Write a metric DSM using the georeference of the raster it was derived from.

    require_georeferenced=True refuses to export when `source`'s transform was the
    nominal fallback rather than real data, so a guessed georeference can never be
    shipped as a metric product. Pass False only for a deliberate mechanics test.
    """
    if require_georeferenced and not source.georeferenced:
        raise ValueError(
            f"{source.path} has no real georeference (load_geotiff fell back to the "
            "Vaihingen nominal transform). Refusing to write a metric GeoTIFF from a "
            "guessed transform - supply the .tfw sidecar, or pass "
            "require_georeferenced=False if you are only testing export mechanics."
        )
    if dsm.shape != source.array.shape[:2]:
        raise ValueError(
            f"DSM shape {dsm.shape} does not match source raster {source.array.shape[:2]}"
        )
    return save_geotiff(
        out_path, dsm,
        pixel_size_x=source.pixel_size_x, pixel_size_y=source.pixel_size_y,
        origin_x=source.origin_x, origin_y=source.origin_y,
        epsg=epsg, nodata=nodata,
    )
