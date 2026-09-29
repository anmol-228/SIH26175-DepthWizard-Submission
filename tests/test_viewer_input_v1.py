"""viewer_input_v1 sidecar contract - DW-WP2 (issues #26/#27, IF-1 frozen 2026-09-06).

Exercises run_pipeline.py's new sidecar writers with a STUB predictor (same pattern as
tests/test_run_pipeline_day3.py) - no checkpoint, no torch, no Vaihingen data, no network.

Covers the frozen IF-1 contract:
  - schema/branch/dtype/byte_order/units are exactly as frozen, for both branches
  - exactly one of gsd/pseudo_gsd is set per branch, never both, never neither
  - relative branch: crs=None, nodata=None, units="unitless", pseudo_gsd set
  - metric branch: crs is the input's embedded code or CRS_UNDEFINED, units="metres",
    gsd set, nodata is the project convention
  - elevation_payload is raw little-endian float32, byte-exact against the source array
  - normalization metadata is present on the relative branch, absent on metric
  - mesh_frame carries the frozen producer formula + elevation_min
  - a relative-only run (no reference / gate decline) never writes a metric sidecar

    python tests/test_viewer_input_v1.py
    pytest tests/test_viewer_input_v1.py
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import run_pipeline as rp  # noqa: E402

WIN = rp.DEFAULT_WINDOW
EPSG_OK = 32601
A_TRUE, B_TRUE = 3.0, 200.0


def _stub_load_predictor(checkpoint, *, allow_base_model, allow_backbone_download=False):
    return object(), "cpu", {"checkpoint": "stub.pt", "sha256": "0" * 64,
                             "base_model": "stub", "WARNING": None}


def _ramp_surface(h, w):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return (0.01 * (yy + xx)).astype(np.float32)


def _stub_predict_depth(rgb, plan, *, model=None, device=None, allow_base_model=False):
    depths, prov = {}, []
    for tid, (y, x) in enumerate(plan.origins):
        arr = (_ramp_surface(plan.window, plan.window) + 0.01 * (y + x)).astype(np.float32)
        depths[(y, x)] = arr
        prov.append({"tile_id": tid, "bbox": [y, x, y + plan.window, x + plan.window],
                     "shape": [plan.window, plan.window], "dtype": "float32",
                     "elapsed_s": 0.0, "min": float(arr.min()), "max": float(arr.max()),
                     "mean": float(arr.mean())})
    return depths, prov


class _FakeMesh:
    def __init__(self, n_faces):
        self.faces = np.zeros((int(n_faces), 3), dtype=np.int64)


def _stub_to_mesh(elev, rgb, gsd, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(b"glTF-stub")
    return _FakeMesh(elev.size), Path(out_path)


def _stub_to_lod(mesh, out_path, *, target_faces=None):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(b"glTF-stub-lod")
    return _FakeMesh(min(len(mesh.faces), 500_000)), Path(out_path)


class _Runner:
    def __enter__(self):
        self._saved = {n: getattr(rp, n) for n in
                       ("load_predictor", "predict_depth", "to_mesh", "to_lod")}
        rp.load_predictor = _stub_load_predictor
        rp.predict_depth = _stub_predict_depth
        rp.to_mesh = _stub_to_mesh
        rp.to_lod = _stub_to_lod
        return self

    def __exit__(self, *exc):
        for n, v in self._saved.items():
            setattr(rp, n, v)
        return False


def _geotiff(path, *, epsg=None, tfw=False, h=WIN, w=WIN):
    rgb = np.random.default_rng(2).integers(0, 255, (h, w, 3), dtype=np.uint8)
    extratags = []
    if epsg is not None:
        extratags = [
            (33550, "d", 3, (0.09, 0.09, 0.0), True),
            (33922, "d", 6, (0.0, 0.0, 0.0, 100.0, 200.0, 0.0), True),
            (34735, "H", 20, (1, 1, 0, 4, 1024, 0, 1, 1, 1025, 0, 1, 1,
                              3072, 0, 1, int(epsg), 3076, 0, 1, 9001), True),
        ]
    tifffile.imwrite(str(path), rgb, photometric="rgb", extratags=extratags)
    if tfw:
        Path(path).with_suffix(".tfw").write_text(
            "0.090000\n0.000000\n0.000000\n-0.090000\n496794.825000\n5419752.255000\n")
    return Path(path)


def _png(path, h=WIN, w=WIN):
    Image.fromarray(np.random.default_rng(1).integers(0, 255, (h, w, 3), dtype=np.uint8)).save(path)
    return Path(path)


def _ref(path, rows, cols, elev):
    arr = np.column_stack([np.asarray(rows, float), np.asarray(cols, float), np.asarray(elev, float)])
    np.save(path, arr)
    return Path(path)


def _stratified_rowcol(n_side=5, lo=40, hi=WIN - 40):
    g = np.linspace(lo, hi, n_side)
    rows = np.repeat(g, n_side)
    cols = np.tile(g, n_side)
    return np.round(rows), np.round(cols)


def _true_elev(rows, cols, a=A_TRUE, b=B_TRUE):
    surf = 0.01 * (rows + cols)
    return a * surf + b


def _run(args):
    return rp.main([str(a) for a in args])


def _read_f32(path, h, w):
    return np.fromfile(path, dtype="<f4").reshape(h, w)


# --------------------------------------------------------------------------- tests
def test_relative_only_run_writes_relative_sidecar_with_frozen_fields():
    with tempfile.TemporaryDirectory() as d, _Runner():
        out = Path(d) / "o"
        assert _run([_png(Path(d) / "a.png"), "--out", out]) == 0

        meta_path = out / "a_relative_surface_meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())

        assert meta["schema"] == "viewer_input_v1"
        assert meta["branch"] == "relative"
        assert meta["dtype"] == "float32"
        assert meta["byte_order"] == "little"
        assert meta["units"] == "unitless"
        assert meta["gsd"] is None
        assert meta["pseudo_gsd"] == rp.PSEUDO_GSD
        assert meta["crs"] is None
        assert meta["nodata"] is None
        assert meta["is_placeholder"] is False
        assert meta["normalization"] is not None
        assert meta["normalization"]["method"] == "percentile_clip"
        assert meta["mesh_glb"] == "a_relative.glb"
        assert meta["lod_glb"] == "a_relative_lod.glb"
        assert meta["run_manifest"] == "run_manifest.json"
        assert meta["mesh_frame"]["z"] == "(elevation - elevation_min) * z_exaggeration"
        assert isinstance(meta["mesh_frame"]["elevation_min"], float)

        # no metric sidecar exists at all on a relative-only run
        assert not list(out.glob("*_metric_surface_meta.json"))
        assert not list(out.glob("*_metric_elevation.f32"))


def test_relative_elevation_payload_is_byte_exact_little_endian_f32():
    with tempfile.TemporaryDirectory() as d, _Runner():
        out = Path(d) / "o"
        assert _run([_png(Path(d) / "b.png"), "--out", out]) == 0

        meta = json.loads((out / "b_relative_surface_meta.json").read_text())
        npy = np.load(out / "relative_dsm_b.npy")
        raw = _read_f32(out / meta["elevation_payload"], meta["H"], meta["W"])

        assert raw.shape == npy.shape == (meta["H"], meta["W"])
        assert np.array_equal(raw, npy), "elevation.f32 must be byte-exact with the .npy array"
        assert raw.dtype == np.dtype("<f4")


def test_metric_run_writes_both_sidecars_correctly_distinguished():
    with tempfile.TemporaryDirectory() as d, _Runner():
        r, c = _stratified_rowcol()
        ref = _ref(Path(d) / "ref.npy", r, c, _true_elev(r, c))
        out = Path(d) / "o"
        code = _run([_geotiff(Path(d) / "scene.tif", epsg=EPSG_OK), "--reference", ref, "--out", out])
        assert code == 0
        assert json.loads((out / "run_manifest.json").read_text())["branch"] == "metric"

        rel_meta = json.loads((out / "scene_relative_surface_meta.json").read_text())
        met_meta = json.loads((out / "scene_metric_surface_meta.json").read_text())

        # relative and metric sidecars are never conflated, even in the same run
        assert rel_meta["branch"] == "relative" and met_meta["branch"] == "metric"
        assert rel_meta["units"] == "unitless" and met_meta["units"] == "metres"
        assert rel_meta["crs"] is None
        assert met_meta["crs"] == EPSG_OK                         # verbatim, no special-case
        assert rel_meta["gsd"] is None and rel_meta["pseudo_gsd"] == rp.PSEUDO_GSD
        assert met_meta["pseudo_gsd"] is None and met_meta["gsd"] == pytest_approx_gsd()
        assert rel_meta["normalization"] is not None
        assert met_meta["normalization"] is None
        assert rel_meta["nodata"] is None
        assert met_meta["nodata"] is not None                     # DSM_NODATA convention

        # the two elevation payloads are genuinely different arrays (relative vs metric
        # values), not the same buffer written twice under different names
        rel_arr = _read_f32(out / rel_meta["elevation_payload"], rel_meta["H"], rel_meta["W"])
        met_arr = _read_f32(out / met_meta["elevation_payload"], met_meta["H"], met_meta["W"])
        assert not np.array_equal(rel_arr, met_arr)
        # metric elevation matches the fitted a*surface+b values, not the [0,1] relative ones
        assert met_arr.min() > 1.0, "metric elevation must be real metres, not [0,1] relative units"


def pytest_approx_gsd():
    # scene.tif's embedded ModelPixelScaleTag above is 0.09 - kept as a tiny local helper
    # so the assertion above reads clearly without importing pytest.approx into this file.
    return 0.09


def test_gate_decline_writes_only_relative_sidecar_no_metric_leak():
    with tempfile.TemporaryDirectory() as d, _Runner():
        rng = np.random.default_rng(42)
        r = np.round(rng.uniform(40, 90, 12)); c = np.round(rng.uniform(40, 90, 12))
        ref = _ref(Path(d) / "ref.npy", r, c, _true_elev(r, c))
        out = Path(d) / "o"
        code = _run([_geotiff(Path(d) / "g.tif", epsg=EPSG_OK), "--reference", ref, "--out", out])
        assert code == 0
        man = json.loads((out / "run_manifest.json").read_text())
        assert man["branch"] == "relative" and man["gate_reason"]

        assert (out / "g_relative_surface_meta.json").exists()
        assert not list(out.glob("*_metric_surface_meta.json"))
        assert not list(out.glob("*_metric_elevation.f32"))
        assert not list(out.glob("*_metric.*"))


def test_lod_and_native_glb_share_the_same_sidecar_stem():
    """LOD decimation changes face count, not the underlying elevation grid - the LOD
    .glb must resolve to the SAME sidecar as its native sibling, not a separate one."""
    with tempfile.TemporaryDirectory() as d, _Runner():
        out = Path(d) / "o"
        assert _run([_png(Path(d) / "c.png"), "--out", out]) == 0
        meta = json.loads((out / "c_relative_surface_meta.json").read_text())
        # both mesh_glb and lod_glb are referenced from the ONE sidecar
        assert meta["mesh_glb"] == "c_relative.glb"
        assert meta["lod_glb"] == "c_relative_lod.glb"
        # and no second, LOD-specific sidecar/elevation payload was written
        assert not (out / "c_relative_lod_surface_meta.json").exists()
        assert not (out / "c_relative_lod_elevation.f32").exists()


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
