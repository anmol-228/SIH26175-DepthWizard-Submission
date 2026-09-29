"""
tests/test_server_reference_sample.py -- P0-A2 core logic tests.

Verifies server.py::reference_sample() against synthetic GeoTIFFs with known,
controlled geotransforms -- not real data (none exists in this repo yet), but a
real georeference round-trip through geoio.load_geotiff(), same pattern already
established in tests/test_run_pipeline_day3.py's _geotiff() helper.

Run: pytest tests/test_server_reference_sample.py -v
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import tifffile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "viewer"))

from server import reference_sample  # noqa: E402


def _elevation_geotiff(
    path: Path,
    elev: np.ndarray,
    *,
    epsg: int | None = 32643,
    pixel_size: float = 1.0,
    origin_x: float = 1000.0,
    origin_y: float = 2000.0,
) -> Path:
    """
    Write a single-band float32 synthetic elevation GeoTIFF with a real,
    controlled geotransform -- same tag structure as
    tests/test_run_pipeline_day3.py::_geotiff(), adapted for elevation data
    (that helper writes an RGB uint8 input image, not an elevation raster).
    """
    extratags = []
    if epsg is not None:
        extratags = [
            (33550, "d", 3, (pixel_size, pixel_size, 0.0), True),  # pixel scale
            (33922, "d", 6, (0.0, 0.0, 0.0, origin_x, origin_y, 0.0), True),  # tiepoint
            (34735, "H", 20, (1, 1, 0, 4, 1024, 0, 1, 1, 1025, 0, 1, 1,
                              3072, 0, 1, int(epsg), 3076, 0, 1, 9001), True),
        ]
    tifffile.imwrite(str(path), elev.astype(np.float32), extratags=extratags)
    return path


def _tfw_only_geotiff(path: Path, elev: np.ndarray, *, pixel_size: float, origin_x: float, origin_y: float) -> Path:
    """A georeferenced-via-.tfw-only raster -- no embedded CRS tags."""
    tifffile.imwrite(str(path), elev.astype(np.float32))
    path.with_suffix(".tfw").write_text(
        f"{pixel_size:.6f}\n0.000000\n0.000000\n-{pixel_size:.6f}\n{origin_x:.6f}\n{origin_y:.6f}\n"
    )
    return path


def _plain_tiff_no_geo(path: Path, elev: np.ndarray) -> Path:
    """A raster with no georeference at all -- geoio falls back to a guess, georeferenced=False."""
    tifffile.imwrite(str(path), elev.astype(np.float32))
    return path


class TestReferenceSampleIdenticalGrid:
    def test_same_geotransform_same_values_gives_zero_delta(self):
        """Metric output and reference raster on the identical grid -> delta_m == 0."""
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            elev = np.full((32, 32), 5.0, dtype=np.float32)
            metric = _elevation_geotiff(d / "scene_metric.tif", elev, pixel_size=0.5, origin_x=500.0, origin_y=1000.0)
            reference = _elevation_geotiff(d / "ref.tif", elev, pixel_size=0.5, origin_x=500.0, origin_y=1000.0)

            result = reference_sample(str(metric), str(reference), row=10.0, col=15.0)

            assert result["ok"] is True
            assert result["estimated_m"] == pytest.approx(5.0)
            assert result["reference_m"] == pytest.approx(5.0)
            assert result["delta_m"] == pytest.approx(0.0, abs=1e-5)
            assert result["units"] == "m"


class TestReferenceSampleOffsetGrid:
    def test_known_offset_and_slope_gives_correct_delta(self):
        """
        Reference raster shifted in both pixel size and real-world origin from the
        metric output, with a known linear elevation ramp -- delta_m must reflect
        the analytically correct value at the shared real-world point, not just
        a same-index lookup (that would be the bug this whole check exists to
        catch: comparing pixel (r,c) to pixel (r,c) across misaligned rasters).
        """
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            h, w = 40, 40

            # Metric output: origin (0, 0), 1.0 gsd, elevation = col (a simple ramp).
            metric_elev = np.tile(np.arange(w, dtype=np.float32), (h, 1))
            metric = _elevation_geotiff(d / "scene_metric.tif", metric_elev, pixel_size=1.0, origin_x=0.0, origin_y=0.0)

            # Reference: same real-world ramp, but a DIFFERENT origin/pixel size,
            # so a naive same-index comparison would be wrong.
            ref_pixel_size = 2.0
            ref_origin_x = -10.0
            ref_origin_y = 50.0
            ref_h, ref_w = 30, 30
            # Reference elevation ramp defined in the SAME real-world x (col * 1.0 in metric's frame).
            ref_elev = np.zeros((ref_h, ref_w), dtype=np.float32)
            for rc in range(ref_w):
                world_x = ref_origin_x + rc * ref_pixel_size
                ref_elev[:, rc] = world_x  # matches metric's elevation=world_x convention
            reference = _elevation_geotiff(
                d / "ref.tif", ref_elev, pixel_size=ref_pixel_size, origin_x=ref_origin_x, origin_y=ref_origin_y
            )

            # Click at metric (row=5, col=20) -> world_x = 0 + 20*1.0 = 20.0 -> estimated_m should be 20.0
            result = reference_sample(str(metric), str(reference), row=5.0, col=20.0)

            assert result["ok"] is True
            assert result["estimated_m"] == pytest.approx(20.0)
            # reference_m should ALSO be ~20.0 (same real-world x, same elevation convention)
            assert result["reference_m"] == pytest.approx(20.0, abs=0.5)
            assert abs(result["delta_m"]) < 0.5


class TestReferenceSampleFailureModes:
    def test_reference_with_no_georeference_is_disabled_not_fabricated(self):
        """A reference .tif with no embedded CRS and no .tfw -> comparison refused, not guessed."""
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            elev = np.full((20, 20), 3.0, dtype=np.float32)
            metric = _elevation_geotiff(d / "scene_metric.tif", elev, pixel_size=1.0, origin_x=0.0, origin_y=0.0)
            reference = _plain_tiff_no_geo(d / "ref.tif", elev)

            result = reference_sample(str(metric), str(reference), row=5.0, col=5.0)

            assert result["ok"] is False
            assert "georeference" in result["reason"].lower()
            assert "estimated_m" not in result
            assert "delta_m" not in result

    def test_click_outside_reference_footprint_is_reported_not_clamped(self):
        """A clicked point whose real-world location falls outside the reference raster -> explicit reason."""
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            metric_elev = np.full((50, 50), 1.0, dtype=np.float32)
            metric = _elevation_geotiff(d / "scene_metric.tif", metric_elev, pixel_size=1.0, origin_x=0.0, origin_y=0.0)

            # Reference raster covers a completely different, non-overlapping real-world area.
            ref_elev = np.full((10, 10), 9.0, dtype=np.float32)
            reference = _elevation_geotiff(
                d / "ref.tif", ref_elev, pixel_size=1.0, origin_x=100000.0, origin_y=100000.0
            )

            result = reference_sample(str(metric), str(reference), row=25.0, col=25.0)

            assert result["ok"] is False
            assert "footprint" in result["reason"].lower() or "outside" in result["reason"].lower()

    def test_click_outside_metric_own_grid_is_reported(self):
        """A (row, col) outside the metric output's own grid -> explicit reason, not a crash or a fabricated read."""
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            metric_elev = np.full((10, 10), 1.0, dtype=np.float32)
            metric = _elevation_geotiff(d / "scene_metric.tif", metric_elev, pixel_size=1.0, origin_x=0.0, origin_y=0.0)
            reference = _elevation_geotiff(d / "ref.tif", metric_elev, pixel_size=1.0, origin_x=0.0, origin_y=0.0)

            result = reference_sample(str(metric), str(reference), row=500.0, col=500.0)

            assert result["ok"] is False

    def test_missing_files_reported_clearly(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            result = reference_sample(str(d / "nope_metric.tif"), str(d / "nope_ref.tif"), row=0.0, col=0.0)
            assert result["ok"] is False
            assert "not found" in result["reason"].lower()

    def test_tfw_only_reference_passes_georeference_check_but_fails_crs_check(self):
        """A .tfw-sidecar-only reference (no embedded CRS tags) IS georeferenced=True
        (real transform, so it passes the georeferenced check) -- but is still
        refused overall, because .tfw-only means CRS_UNDEFINED, and the CRS check
        (TestReferenceSampleCrsCheck) refuses anything undefined. This confirms the
        two checks are distinct: this must fail for the CRS reason, not the
        georeference reason."""
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            elev = np.full((20, 20), 7.0, dtype=np.float32)
            metric = _elevation_geotiff(d / "scene_metric.tif", elev, pixel_size=1.0, origin_x=0.0, origin_y=0.0)
            reference = _tfw_only_geotiff(d / "ref.tif", elev, pixel_size=1.0, origin_x=0.0, origin_y=0.0)

            result = reference_sample(str(metric), str(reference), row=5.0, col=5.0)

            assert result["ok"] is False
            # Must be refused for the CRS reason, NOT the "no georeference" reason --
            # confirms the two checks are distinct and this hits the right one.
            assert "georeference" not in result["reason"].lower()
            assert "crs" in result["reason"].lower() or "defensible" in result["reason"].lower()

class TestReferenceSampleCrsCheck:
    """
    Nirmalya's PR #62 review, finding 1: reference_sample() must refuse to compare
    two rasters unless both carry the SAME real, defensible EPSG code. Two rasters
    that are both individually georeferenced=True can still be in different
    coordinate systems -- that's a silent wrong-answer, not caught by the existing
    georeferenced=False check.
    """

    def test_matching_real_epsg_still_passes(self):
        """Regression check: the existing identical-grid case must still work
        now that a CRS check has been added -- same EPSG on both sides."""
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            elev = np.full((32, 32), 5.0, dtype=np.float32)
            metric = _elevation_geotiff(
                d / "scene_metric.tif", elev, epsg=32643, pixel_size=0.5, origin_x=500.0, origin_y=1000.0
            )
            reference = _elevation_geotiff(
                d / "ref.tif", elev, epsg=32643, pixel_size=0.5, origin_x=500.0, origin_y=1000.0
            )

            result = reference_sample(str(metric), str(reference), row=10.0, col=15.0)

            assert result["ok"] is True
            assert result["delta_m"] == pytest.approx(0.0, abs=1e-5)

    def test_different_real_epsg_is_refused(self):
        """Two rasters with different, both-defensible EPSG codes -- e.g. the USGS
        EPSG:26911 tiles Nirmalya mentioned vs. a run's own EPSG:32643 output --
        must be refused, not silently pasted together as one coordinate space."""
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            elev = np.full((32, 32), 5.0, dtype=np.float32)
            metric = _elevation_geotiff(
                d / "scene_metric.tif", elev, epsg=32643, pixel_size=0.5, origin_x=500.0, origin_y=1000.0
            )
            reference = _elevation_geotiff(
                d / "ref.tif", elev, epsg=26911, pixel_size=0.5, origin_x=500.0, origin_y=1000.0
            )

            result = reference_sample(str(metric), str(reference), row=10.0, col=15.0)

            assert result["ok"] is False
            assert "crs" in result["reason"].lower() or "mismatch" in result["reason"].lower()
            assert "estimated_m" not in result
            assert "delta_m" not in result

    def test_reference_with_crs_undefined_is_refused(self):
        """Reference raster is .tfw-only (real transform, no defensible authority
        code -> CRS_UNDEFINED per geoio's own convention) -- refused even though
        BOTH rasters are individually georeferenced=True."""
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            elev = np.full((20, 20), 3.0, dtype=np.float32)
            metric = _elevation_geotiff(
                d / "scene_metric.tif", elev, epsg=32643, pixel_size=1.0, origin_x=0.0, origin_y=0.0
            )
            # .tfw-only -> real geotransform, but geoio assigns CRS_UNDEFINED (32767),
            # not a real epsg -- same convention as _geotiff(tfw=True) in
            # tests/test_run_pipeline_day3.py.
            reference = _tfw_only_geotiff(d / "ref.tif", elev, pixel_size=1.0, origin_x=0.0, origin_y=0.0)

            result = reference_sample(str(metric), str(reference), row=5.0, col=5.0)

            assert result["ok"] is False
            assert "crs" in result["reason"].lower() or "defensible" in result["reason"].lower()

    def test_metric_with_crs_undefined_is_also_refused(self):
        """Same check applies symmetrically: an undefined CRS on the METRIC side
        (not just the reference side) must also refuse, not just skip the check
        because the reference side happened to be fine."""
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            elev = np.full((20, 20), 3.0, dtype=np.float32)
            metric = _tfw_only_geotiff(d / "scene_metric.tif", elev, pixel_size=1.0, origin_x=0.0, origin_y=0.0)
            reference = _elevation_geotiff(
                d / "ref.tif", elev, epsg=32643, pixel_size=1.0, origin_x=0.0, origin_y=0.0
            )

            result = reference_sample(str(metric), str(reference), row=5.0, col=5.0)

            assert result["ok"] is False
            assert "crs" in result["reason"].lower() or "defensible" in result["reason"].lower()

    def test_both_crs_undefined_is_still_refused_not_assumed_matching(self):
        """Two CRS_UNDEFINED rasters might coincidentally be the same real system
        (e.g. both derived from the same Vaihingen-style source), but nothing here
        can defend that -- per the repo's 'don't assert what we can't defend'
        convention, this must refuse, not silently treat 'both unknown' as 'both
        the same'."""
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            elev = np.full((20, 20), 3.0, dtype=np.float32)
            metric = _tfw_only_geotiff(d / "scene_metric.tif", elev, pixel_size=1.0, origin_x=0.0, origin_y=0.0)
            reference = _tfw_only_geotiff(d / "ref.tif", elev, pixel_size=1.0, origin_x=0.0, origin_y=0.0)

            result = reference_sample(str(metric), str(reference), row=5.0, col=5.0)

            assert result["ok"] is False

class TestReferenceSampleRunPathResolution:
    """
    P-14 (Anmol, live browser reproduction 2026-09-14): reference_sample()
    previously called Path(metric_tif_path) directly on a browser-shaped
    "/run/<out_dir>/<scene>_metric.tif" path, which does NOT resolve under
    RUN_OUTPUT_DIR (on Windows it resolves against the current drive root
    instead) -- every genuine metric-branch reference-sample click failed
    with "metric output not found", even though the file was real and
    present. Confirmed via a real browser: GET /run/... (loading the mesh)
    worked correctly; POST /api/reference-sample (clicking to compare) did
    not, because only do_GET had /run/ prefix-stripping logic.

    These tests use monkeypatched RUN_OUTPUT_DIR (via the module itself,
    not a real outputs/run/ write) so they don't depend on or pollute the
    real filesystem location, per "do not commit reproduction assets,
    generated outputs... or other scratch files."
    """

    def test_browser_shaped_run_path_resolves_correctly(self, tmp_path, monkeypatch):
        """
        The exact browser-shaped path from the live reproduction:
        /run/<out_dir>/<scene>_metric.tif -- must resolve under RUN_OUTPUT_DIR
        and succeed, matching what GET /run/... already does for the same URL.
        """
        import server as server_module

        fake_run_dir = tmp_path / "outputs_run"
        fake_run_dir.mkdir()
        monkeypatch.setattr(server_module, "RUN_OUTPUT_DIR", fake_run_dir)

        scene_dir = fake_run_dir / "hillyaoi_r384_c1408_w512"
        scene_dir.mkdir()
        elev = np.full((16, 16), 5.0, dtype=np.float32)
        metric_path = _elevation_geotiff(
            scene_dir / "hillyaoi_r384_c1408_w512_metric.tif",
            elev, pixel_size=1.0, origin_x=0.0, origin_y=0.0,
        )
        reference_path = _elevation_geotiff(
            tmp_path / "ref.tif", elev, pixel_size=1.0, origin_x=0.0, origin_y=0.0,
        )

        result = server_module.reference_sample(
            metric_tif_path="/run/hillyaoi_r384_c1408_w512/hillyaoi_r384_c1408_w512_metric.tif",
            reference_tif_path=str(reference_path),
            row=5.0, col=5.0,
        )

        assert result["ok"] is True
        assert result["estimated_m"] == pytest.approx(5.0)
        assert metric_path.exists()  # sanity: the file genuinely exists on disk

    def test_run_path_traversal_is_refused_not_followed(self, tmp_path, monkeypatch):
        """A /run/../../<anything> path must never escape RUN_OUTPUT_DIR."""
        import server as server_module

        fake_run_dir = tmp_path / "outputs_run"
        fake_run_dir.mkdir()
        monkeypatch.setattr(server_module, "RUN_OUTPUT_DIR", fake_run_dir)

        outside_file = tmp_path / "secret.tif"
        elev = np.full((8, 8), 1.0, dtype=np.float32)
        _elevation_geotiff(outside_file, elev, pixel_size=1.0, origin_x=0.0, origin_y=0.0)

        result = server_module.reference_sample(
            metric_tif_path="/run/../secret.tif",
            reference_tif_path=str(outside_file),
            row=0.0, col=0.0,
        )

        assert result["ok"] is False
        assert "escapes" in result["reason"].lower() or "not found" in result["reason"].lower()

    def test_plain_absolute_path_still_works_unchanged(self, tmp_path):
        """
        Regression guard: existing callers/tests use a plain absolute filesystem
        path (not a /run/ web path) -- that behavior must be completely
        unaffected by this fix. Duplicates the original identical-grid case
        from TestReferenceSampleIdenticalGrid using the NEW code path, to
        confirm the fix didn't change plain-path behavior at all.
        """
        elev = np.full((20, 20), 3.0, dtype=np.float32)
        metric = _elevation_geotiff(
            tmp_path / "scene_metric.tif", elev, pixel_size=1.0, origin_x=0.0, origin_y=0.0
        )
        reference = _elevation_geotiff(
            tmp_path / "ref.tif", elev, pixel_size=1.0, origin_x=0.0, origin_y=0.0
        )

        result = reference_sample(str(metric), str(reference), row=5.0, col=5.0)

        assert result["ok"] is True
        assert result["delta_m"] == pytest.approx(0.0, abs=1e-5)


class TestReferenceSampleHttpIntegration:
    """
    Anmol's explicit ask: "Add/extend an HTTP-level test for POST
    /api/reference-sample, rather than testing only reference_sample() with
    absolute filesystem paths." This is the exact gap that let P-14 through --
    every existing test called reference_sample() directly with an absolute
    path, so the do_POST -> reference_sample() wiring itself, and the
    JSON-over-HTTP path a real browser actually takes, had zero coverage.

    Drives a REAL running instance of the server (ThreadingHTTPServer on an
    ephemeral port) with a real HTTP client -- not a mocked handler -- so this
    genuinely exercises the same code path a browser hits.
    """

    @pytest.fixture
    def running_server(self, tmp_path, monkeypatch):
        import http.client
        import threading

        import server as server_module

        fake_run_dir = tmp_path / "outputs_run"
        fake_run_dir.mkdir()
        monkeypatch.setattr(server_module, "RUN_OUTPUT_DIR", fake_run_dir)

        scene_dir = fake_run_dir / "hillyaoi_r384_c1408_w512"
        scene_dir.mkdir()
        elev = np.full((16, 16), 5.0, dtype=np.float32)
        _elevation_geotiff(
            scene_dir / "hillyaoi_r384_c1408_w512_metric.tif",
            elev, pixel_size=1.0, origin_x=0.0, origin_y=0.0,
        )
        reference_path = _elevation_geotiff(
            tmp_path / "ref.tif", elev, pixel_size=1.0, origin_x=0.0, origin_y=0.0,
        )

        httpd = server_module.ThreadingHTTPServer(("127.0.0.1", 0), server_module.Handler)
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            yield port, str(reference_path)
        finally:
            httpd.shutdown()
            thread.join(timeout=5)

    def test_post_reference_sample_with_browser_shaped_run_path(self, running_server):
        """
        The exact request shape from the live browser reproduction
        (11_BROWSER_REQUEST.txt): POST /api/reference-sample with
        metric_tif_path as a "/run/..." web path -- must now return
        {"ok": true, estimated_m/reference_m/delta_m}, not the
        "metric output not found" refusal from before the fix.
        """
        import http.client
        import json as json_module

        port, reference_path = running_server
        conn = http.client.HTTPConnection("127.0.0.1", port)
        body = json_module.dumps({
            "metric_tif_path": "/run/hillyaoi_r384_c1408_w512/hillyaoi_r384_c1408_w512_metric.tif",
            "reference_tif_path": reference_path,
            "row": 5.0,
            "col": 5.0,
        })
        conn.request(
            "POST", "/api/reference-sample", body=body,
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        result = json_module.loads(resp.read())
        conn.close()

        assert resp.status == 200
        assert result["ok"] is True, f"expected success, got refusal: {result.get('reason')}"
        assert result["estimated_m"] == pytest.approx(5.0)
        assert "estimated_m" in result and "reference_m" in result and "delta_m" in result