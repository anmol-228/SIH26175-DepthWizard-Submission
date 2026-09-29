"""Regression tests for the CALIBRATION -> PIPELINE/VIEWER export path (issue #5).

Covers geoio.save_geotiff / load_geotiff round-trip, the guards that stop a guessed
georeference being shipped as a metric product, and the per-patch -> canvas -> GeoTIFF
-> reopen -> mesh chain that scripts/07_calibrate_and_eval.py performs.

Deliberately needs no checkpoint, no Vaihingen data and no network, so it runs on any
clone. The real-data end-to-end evidence lives in
10_DOCUMENTATION/DW_S1_02_RUN_EVIDENCE.md instead.

Run either way:
    python tests/test_geoio_roundtrip.py     # plain, no pytest needed
    pytest tests/test_geoio_roundtrip.py     # if pytest is installed
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geoio import (  # noqa: E402
    DSM_NODATA, GeoRaster, dsm_nodata_mask, load_geotiff, read_geotiff_tags,
    save_dsm_geotiff, save_geotiff,
)

# K2's real verified transform for Vaihingen area 26
# (PHASE2_RED_FLAGS/outputs/metrics/K2_georeferencing_check.json)
PX, PY = 0.09, -0.09
OX, OY = 496794.825, 5419752.255
# Arbitrary round-trip probe value. These tests check that whatever code is written
# comes back, not that any particular CRS is correct for Vaihingen - deliberately NOT
# 32633 (issue #14: wrong, and the no-hardcoded-CRS guard would otherwise need an
# exception here) and deliberately not 32767 either, so a writer that silently ignored
# the caller's epsg and always emitted user-defined would fail these tests.
EPSG = 32615
PATCH = 512  # must match scripts/04_make_patches.py


def _sample(h=200, w=310, seed=42):
    return (np.random.default_rng(seed).random((h, w)).astype(np.float32) * 40 + 260)


def test_geotiff_roundtrip_preserves_georeference():
    """Every field K2 checks must survive write + independent reopen."""
    arr = _sample()
    with tempfile.TemporaryDirectory() as d:
        p = save_geotiff(Path(d) / "t.tif", arr, pixel_size_x=PX, pixel_size_y=PY,
                         origin_x=OX, origin_y=OY, epsg=EPSG)
        r = load_geotiff(p)
        tags = read_geotiff_tags(p)

    assert abs(r.pixel_size_x - PX) < 1e-9 and abs(r.pixel_size_y - PY) < 1e-9
    assert abs(r.origin_x - OX) < 1e-6 and abs(r.origin_y - OY) < 1e-6
    assert r.array.shape == arr.shape
    assert r.epsg == EPSG
    assert r.array.dtype == np.float32
    assert abs(r.gsd - 0.09) < 1e-9
    assert r.georeferenced is True
    # nodata was reader-side convention only before issue #5; it is now a written tag
    assert tags.get("nodata") == DSM_NODATA
    assert np.array_equal(r.array, arr), "pixel values must be bit-exact"


def test_nodata_tag_and_mask_agree():
    arr = _sample()
    arr[0, :5] = DSM_NODATA
    with tempfile.TemporaryDirectory() as d:
        p = save_geotiff(Path(d) / "t.tif", arr, pixel_size_x=PX, pixel_size_y=PY,
                         origin_x=OX, origin_y=OY, epsg=EPSG)
        r = load_geotiff(p)
    valid = dsm_nodata_mask(r.array, DSM_NODATA)
    assert int((~valid).sum()) == 5


def test_refuses_to_export_from_a_guessed_georeference():
    """The Vaihingen nominal fallback must never be shipped as a metric product."""
    arr = _sample()
    with tempfile.TemporaryDirectory() as d:
        plain = Path(d) / "plain.tif"
        tifffile.imwrite(plain, arr)          # no GeoTIFF tags, no .tfw sidecar
        g = load_geotiff(plain)
        assert g.georeferenced is False, "fallback transform must be flagged, not trusted"
        assert g.epsg is None
        try:
            save_dsm_geotiff(Path(d) / "bad.tif", arr, g, epsg=EPSG)
        except ValueError:
            pass
        else:
            raise AssertionError("save_dsm_geotiff accepted a guessed georeference")


def test_strict_load_raises_instead_of_guessing():
    with tempfile.TemporaryDirectory() as d:
        plain = Path(d) / "plain.tif"
        tifffile.imwrite(plain, _sample())
        try:
            load_geotiff(plain, strict=True)
        except ValueError:
            pass
        else:
            raise AssertionError("strict=True silently guessed a georeference")


def test_georaster_backward_compatibility():
    """Pre-issue-#5 callers construct GeoRaster without epsg/georeferenced."""
    r = GeoRaster(array=_sample(), pixel_size_x=PX, pixel_size_y=PY,
                  origin_x=OX, origin_y=OY, path="x")
    assert r.epsg is None and r.georeferenced is True


def test_patch_canvas_to_geotiff_to_mesh():
    """The chain 07_calibrate_and_eval.py runs: per-patch 2-D arrays -> tile-anchored
    canvas -> GeoTIFF -> reopen -> downstream mesh consumer."""
    rng = np.random.default_rng(42)
    # 2x2 patch grid with one patch dropped, as 04_make_patches.py's validity mask does
    entries = [{"y": 0, "x": 0}, {"y": 0, "x": PATCH}, {"y": PATCH, "x": 0}]
    preds = [rng.random((PATCH, PATCH)).astype(np.float32) for _ in entries]
    a, b = 4.2586481419496955, 274.3487673590467   # real (a,b) from the committed area-1 run

    H = max(e["y"] for e in entries) + PATCH
    W = max(e["x"] for e in entries) + PATCH
    canvas = np.full((H, W), DSM_NODATA, dtype=np.float32)
    for e, arr in zip(entries, preds):
        canvas[e["y"]:e["y"] + PATCH, e["x"]:e["x"] + PATCH] = (a * arr + b).astype(np.float32)

    assert (canvas[PATCH:, PATCH:] == DSM_NODATA).all(), "uncovered region must be nodata"

    with tempfile.TemporaryDirectory() as d:
        p = save_geotiff(Path(d) / "cal.tif", canvas, pixel_size_x=PX, pixel_size_y=PY,
                         origin_x=OX, origin_y=OY, epsg=EPSG)
        r = load_geotiff(p)

    valid = dsm_nodata_mask(r.array, DSM_NODATA)
    assert int(valid.sum()) == len(entries) * PATCH * PATCH
    assert np.array_equal(r.array, canvas)
    v = r.array[valid]
    assert -500.0 < v.min() and v.max() < 9000.0, "values must look like metres"

    # real downstream consumer, gsd sourced from the raster rather than hardcoded
    try:
        sys.path.insert(0, str(ROOT / "PHASE2_RED_FLAGS" / "scripts"))
        from K3_mesh_pipeline import heightfield_to_mesh
    except ImportError:
        return  # trimesh absent; the round-trip assertions above still ran
    elev = np.where(valid, r.array, float(r.array[valid].min()))
    mesh = heightfield_to_mesh(elev, np.zeros(elev.shape + (3,), np.uint8),
                               gsd=r.gsd, downsample=8)
    assert len(mesh.faces) > 0 and np.isfinite(mesh.vertices).all()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as exc:              # noqa: BLE001
            failed += 1
            print(f"  FAIL  {t.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
