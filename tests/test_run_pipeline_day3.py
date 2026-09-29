"""Day-3 end-to-end contract for run_pipeline.py - DW-S1-07 (#24).

Exercises the whole runner (`main()`) with a STUB predictor - no checkpoint, no torch, no
Vaihingen data, no network. The relative surface is a deterministic linear ramp so a
synthetic reference can drive an exact scale/shift fit.

Covers the locked #24 contract:
  - JPG/PNG is always relative (even with a sidecar .tfw or a valid --reference)
  - a broken *supplied* --reference is exit 2 even for a PNG (parsed before eligibility)
  - GeoTIFF + valid passing reference + finite a>0 -> metric (.tif + .glb + LOD)
  - embedded CRS propagates verbatim; .tfw-only -> geoio.CRS_UNDEFINED
  - preflight rejection / too-few points -> relative exit 0, gate_reason kept
  - non-finite or a<=0 fit after a passed gate -> exit 2, nothing written
  - relative and metric filenames never collide

    python tests/test_run_pipeline_day3.py
    pytest tests/test_run_pipeline_day3.py
"""
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
from geoio import CRS_UNDEFINED, load_geotiff  # noqa: E402

WIN = rp.DEFAULT_WINDOW          # 512; validate_input needs both sides >= this
EPSG_OK = 32601                  # arbitrary valid projected CRS, not a value the runner special-cases
A_TRUE, B_TRUE = 3.0, 200.0


# --------------------------------------------------------------------------- stubs
def _stub_load_predictor(checkpoint, *, allow_base_model, allow_backbone_download=False):
    return object(), "cpu", {"checkpoint": "stub.pt", "sha256": "0" * 64,
                             "base_model": "stub", "WARNING": None}


def _ramp_surface(h, w):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return (0.01 * (yy + xx)).astype(np.float32)


def _stub_predict_depth(rgb, plan, *, model=None, device=None, allow_base_model=False):
    depths, prov = {}, []
    for tid, (y, x) in enumerate(plan.origins):
        arr = _ramp_surface(plan.window, plan.window) + 0.01 * (y + x)
        arr = arr.astype(np.float32)
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
    """Context manager: stubs the model + (optionally) the mesh/LOD writers."""
    def __init__(self, real_meshes=False):
        self.real_meshes = real_meshes
        self._saved = {}

    def __enter__(self):
        for name, val in {"load_predictor": _stub_load_predictor,
                          "predict_depth": _stub_predict_depth}.items():
            self._saved[name] = getattr(rp, name)
            setattr(rp, name, val)
        if not self.real_meshes:
            for name, val in {"to_mesh": _stub_to_mesh, "to_lod": _stub_to_lod}.items():
                self._saved[name] = getattr(rp, name)
                setattr(rp, name, val)
        return self

    def __exit__(self, *exc):
        for name, val in self._saved.items():
            setattr(rp, name, val)
        return False


# ------------------------------------------------------------------- fixture makers
def _png(path, h=WIN, w=WIN):
    Image.fromarray(np.random.default_rng(1).integers(0, 255, (h, w, 3), dtype=np.uint8)).save(path)
    return Path(path)


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


def _ref(path, rows, cols, elev):
    arr = np.column_stack([np.asarray(rows, float), np.asarray(cols, float), np.asarray(elev, float)])
    np.save(path, arr)
    return Path(path)


def _ref_text(path, rows, cols, elev, sep):
    """Write reference points as text. sep=' ' -> whitespace, sep=',' -> comma-separated.
    Values are written with round-trip precision so a text ref and the equivalent .npy
    parse to identical floats."""
    lines = [sep.join(repr(float(v)) for v in row)
             for row in zip(np.asarray(rows, float), np.asarray(cols, float), np.asarray(elev, float))]
    Path(path).write_text("\n".join(lines) + "\n")
    return Path(path)


def _stratified_rowcol(n_side=5, lo=40, hi=WIN - 40):
    g = np.linspace(lo, hi, n_side)
    rows = np.repeat(g, n_side)
    cols = np.tile(g, n_side)
    return np.round(rows), np.round(cols)


def _true_elev(rows, cols, a=A_TRUE, b=B_TRUE):
    surf = 0.01 * (rows + cols)          # matches _stub_predict_depth's ramp
    return a * surf + b


def _run(args):
    return rp.main([str(a) for a in args])


# --------------------------------------------------------------------------- tests
def test_png_no_reference_is_relative_exit_0():
    with tempfile.TemporaryDirectory() as d, _Runner():
        out = Path(d) / "o"
        code = _run([_png(Path(d) / "a.png"), "--out", out])
        assert code == 0
        assert (out / "relative_dsm_a.npy").exists()
        assert (out / "a_relative.glb").exists()
        assert (out / "a_relative_lod.glb").exists()
        assert not list(out.glob("*_metric.*"))
        man = _manifest(out)
        assert man["branch"] == "relative" and man["crs"] is None
        # #24 locked manifest contract: crs / a / b are top-level; relative -> all None
        assert man["a"] is None and man["b"] is None
        assert man["calibration"]["a"] is None and man["calibration"]["b"] is None


def test_png_with_sidecar_tfw_is_still_relative():
    with tempfile.TemporaryDirectory() as d, _Runner():
        p = _png(Path(d) / "b.png")
        p.with_suffix(".tfw").write_text("0.09\n0\n0\n-0.09\n100\n200\n")
        out = Path(d) / "o"
        assert _run([p, "--out", out]) == 0
        assert _manifest(out)["branch"] == "relative"
        assert not list(out.glob("*_metric.*"))


def test_png_with_valid_reference_is_still_relative_exit_0():
    with tempfile.TemporaryDirectory() as d, _Runner():
        r, c = _stratified_rowcol()
        ref = _ref(Path(d) / "ref.npy", r, c, _true_elev(r, c))
        out = Path(d) / "o"
        code = _run([_png(Path(d) / "c.png"), "--reference", ref, "--out", out])
        assert code == 0
        man = _manifest(out)
        assert man["branch"] == "relative"
        assert "no real geotransform" in man["branch_reason"]
        assert not list(out.glob("*_metric.*"))


def test_png_with_broken_reference_is_exit_2():
    with tempfile.TemporaryDirectory() as d, _Runner():
        bad = Path(d) / "bad.npy"
        np.save(bad, np.zeros((5, 2)))                 # wrong shape
        out = Path(d) / "o"
        code = _run([_png(Path(d) / "d.png"), "--reference", bad, "--out", out])
        assert code == rp.EXIT_BAD_INPUT
        assert not out.exists() or not any(out.iterdir())


def test_geotiff_no_reference_is_relative_exit_0():
    with tempfile.TemporaryDirectory() as d, _Runner():
        out = Path(d) / "o"
        assert _run([_geotiff(Path(d) / "g.tif", epsg=EPSG_OK), "--out", out]) == 0
        assert _manifest(out)["branch"] == "relative"
        assert not list(out.glob("*_metric.*"))


def test_missing_reference_path_is_exit_2():
    with tempfile.TemporaryDirectory() as d, _Runner():
        out = Path(d) / "o"
        code = _run([_geotiff(Path(d) / "g.tif", epsg=EPSG_OK),
                     "--reference", Path(d) / "nope.npy", "--out", out])
        assert code == rp.EXIT_BAD_INPUT


def test_unparseable_reference_is_exit_2():
    with tempfile.TemporaryDirectory() as d, _Runner():
        junk = Path(d) / "junk.txt"
        junk.write_text("not numbers at all\nrows and cols\n")
        out = Path(d) / "o"
        code = _run([_geotiff(Path(d) / "g.tif", epsg=EPSG_OK), "--reference", junk, "--out", out])
        assert code == rp.EXIT_BAD_INPUT


def test_wrong_shape_reference_is_exit_2():
    with tempfile.TemporaryDirectory() as d, _Runner():
        bad = Path(d) / "b.npy"
        np.save(bad, np.arange(20).reshape(10, 2).astype(float))     # only 2 columns
        out = Path(d) / "o"
        assert _run([_geotiff(Path(d) / "g.tif", epsg=EPSG_OK),
                     "--reference", bad, "--out", out]) == rp.EXIT_BAD_INPUT


def test_non_finite_reference_value_is_exit_2():
    with tempfile.TemporaryDirectory() as d, _Runner():
        r, c = _stratified_rowcol()
        elev = _true_elev(r, c).astype(float)
        elev[3] = np.inf
        ref = _ref(Path(d) / "ref.npy", r, c, elev)
        out = Path(d) / "o"
        assert _run([_geotiff(Path(d) / "g.tif", epsg=EPSG_OK),
                     "--reference", ref, "--out", out]) == rp.EXIT_BAD_INPUT


def test_out_of_bounds_rowcol_is_exit_2():
    with tempfile.TemporaryDirectory() as d, _Runner():
        r, c = _stratified_rowcol()
        r = r.copy(); r[0] = WIN + 50                  # past the bottom edge
        ref = _ref(Path(d) / "ref.npy", r, c, _true_elev(r, c))
        out = Path(d) / "o"
        assert _run([_geotiff(Path(d) / "g.tif", epsg=EPSG_OK),
                     "--reference", ref, "--out", out]) == rp.EXIT_BAD_INPUT
        assert not out.exists() or not any(out.iterdir())


def test_fractional_rowcol_is_rejected_not_truncated():
    with tempfile.TemporaryDirectory() as d, _Runner():
        r, c = _stratified_rowcol()
        c = c.copy(); c[2] += 0.37
        ref = _ref(Path(d) / "ref.npy", r, c, _true_elev(r, c))
        out = Path(d) / "o"
        assert _run([_geotiff(Path(d) / "g.tif", epsg=EPSG_OK),
                     "--reference", ref, "--out", out]) == rp.EXIT_BAD_INPUT


def test_clustered_reference_is_relative_exit_0_with_gate_reason():
    with tempfile.TemporaryDirectory() as d, _Runner():
        rng = np.random.default_rng(42)
        r = np.round(rng.uniform(40, 90, 12)); c = np.round(rng.uniform(40, 90, 12))
        ref = _ref(Path(d) / "ref.npy", r, c, _true_elev(r, c))
        out = Path(d) / "o"
        code = _run([_geotiff(Path(d) / "g.tif", epsg=EPSG_OK), "--reference", ref, "--out", out])
        assert code == 0
        man = _manifest(out)
        assert man["branch"] == "relative"
        assert man["gate_reason"] and ("cover" in man["gate_reason"] or "spread" in man["gate_reason"])
        assert man["preflight"]["pass"] is False
        assert not list(out.glob("*_metric.*"))


def test_too_few_points_is_relative_exit_0():
    with tempfile.TemporaryDirectory() as d, _Runner():
        r, c = _stratified_rowcol(n_side=2)            # 4 points < MIN_REFERENCE_POINTS
        ref = _ref(Path(d) / "ref.npy", r, c, _true_elev(r, c))
        out = Path(d) / "o"
        assert _run([_geotiff(Path(d) / "g.tif", epsg=EPSG_OK), "--reference", ref, "--out", out]) == 0
        man = _manifest(out)
        assert man["branch"] == "relative" and man["gate_reason"]


def test_embedded_crs_geotiff_plus_passing_reference_is_metric():
    with tempfile.TemporaryDirectory() as d, _Runner():
        r, c = _stratified_rowcol()
        ref = _ref(Path(d) / "ref.npy", r, c, _true_elev(r, c))
        out = Path(d) / "o"
        code = _run([_geotiff(Path(d) / "scene.tif", epsg=EPSG_OK), "--reference", ref, "--out", out])
        assert code == 0
        assert (out / "scene_metric.tif").exists()
        assert (out / "scene_metric.glb").exists()
        assert (out / "scene_metric_lod.glb").exists()
        man = _manifest(out)
        assert man["branch"] == "metric"
        assert man["crs"] == EPSG_OK                                   # verbatim, no special-case
        # #24 locked manifest contract: crs / a / b are top-level on the metric branch
        assert "a" in man and "b" in man
        assert isinstance(man["a"], float) and np.isfinite(man["a"]) and man["a"] > 0
        assert isinstance(man["b"], float) and np.isfinite(man["b"])
        assert man["a"] == man["calibration"]["a"]
        assert man["b"] == man["calibration"]["b"]
        assert abs(man["a"] - A_TRUE) < 1e-6
        assert abs(man["b"] - B_TRUE) < 1e-4
        assert man["calibration"]["fit"]["method"] == "ransac"
        # embedded CRS must round-trip through the written GeoTIFF
        assert load_geotiff(out / "scene_metric.tif").epsg == EPSG_OK


def _metric_manifest_from_ref(d, ref_path):
    """Run the metric branch with the given --reference file; assert exit 0, return manifest."""
    with _Runner():
        stem = Path(ref_path).stem
        out = Path(d) / ("o_" + stem)
        code = _run([_geotiff(Path(d) / (stem + ".tif"), epsg=EPSG_OK),
                     "--reference", ref_path, "--out", out])
    assert code == 0, f"{ref_path.name}: exit {code}"
    return _manifest(out)


def test_whitespace_separated_text_reference_still_drives_metric():   # A
    with tempfile.TemporaryDirectory() as d:
        r, c = _stratified_rowcol()
        ref = _ref_text(Path(d) / "ws.txt", r, c, _true_elev(r, c), sep=" ")
        man = _metric_manifest_from_ref(d, ref)
        assert man["branch"] == "metric"
        assert abs(man["a"] - A_TRUE) < 1e-6 and abs(man["b"] - B_TRUE) < 1e-4


def test_comma_separated_text_reference_parses_and_drives_metric():   # B + C
    with tempfile.TemporaryDirectory() as d:
        r, c = _stratified_rowcol()
        elev = _true_elev(r, c)
        man_ws = _metric_manifest_from_ref(d, _ref_text(Path(d) / "ws.txt", r, c, elev, sep=" "))
        man_cs = _metric_manifest_from_ref(d, _ref_text(Path(d) / "cs.txt", r, c, elev, sep=","))
        man_np = _metric_manifest_from_ref(d, _ref(Path(d) / "np.npy", r, c, elev))
        assert man_ws["branch"] == man_cs["branch"] == man_np["branch"] == "metric"
        # comma-, whitespace- and .npy references must produce the same fit
        assert man_cs["a"] == man_ws["a"] == man_np["a"]
        assert man_cs["b"] == man_ws["b"] == man_np["b"]


def test_comma_separated_reference_with_spaces_after_commas_parses():  # B (lenient whitespace)
    with tempfile.TemporaryDirectory() as d, _Runner():
        r, c = _stratified_rowcol()
        lines = [f"{int(rr)}, {int(cc)}, {ee:.4f}"
                 for rr, cc, ee in zip(r, c, _true_elev(r, c))]
        ref = Path(d) / "cs_sp.txt"
        ref.write_text("\n".join(lines) + "\n")
        out = Path(d) / "o"
        code = _run([_geotiff(Path(d) / "g.tif", epsg=EPSG_OK), "--reference", ref, "--out", out])
        assert code == 0 and _manifest(out)["branch"] == "metric"


def test_malformed_comma_text_reference_is_exit_2():                  # D
    with tempfile.TemporaryDirectory() as d, _Runner():
        ref = Path(d) / "bad.txt"
        ref.write_text("100,200,foo\n300,bar,261.7\n")                # non-numeric cells
        out = Path(d) / "o"
        code = _run([_geotiff(Path(d) / "g.tif", epsg=EPSG_OK), "--reference", ref, "--out", out])
        assert code == rp.EXIT_BAD_INPUT
        assert not out.exists() or not any(out.iterdir())


def test_comma_text_reference_still_enforces_bounds_finite_fractional():  # E
    with tempfile.TemporaryDirectory() as d:
        r, c = _stratified_rowcol()
        # out-of-bounds row
        oob = r.copy(); oob[0] = WIN + 10
        # non-finite elevation
        nf_e = _true_elev(r, c).astype(float); nf_e[1] = np.nan
        # fractional col
        frac = c.copy().astype(float); frac[2] += 0.5
        cases = {
            "oob": _ref_text(Path(d) / "oob.txt", oob, c, _true_elev(oob, c), sep=","),
            "nonfinite": _ref_text(Path(d) / "nf.txt", r, c, nf_e, sep=","),
            "fractional": _ref_text(Path(d) / "frac.txt", r, frac, _true_elev(r, frac), sep=","),
        }
        for label, ref in cases.items():
            with _Runner():
                out = Path(d) / ("o_" + label)
                code = _run([_geotiff(Path(d) / (label + ".tif"), epsg=EPSG_OK),
                             "--reference", ref, "--out", out])
            assert code == rp.EXIT_BAD_INPUT, f"{label}: comma-parsed but validation was skipped"
            assert not out.exists() or not any(out.iterdir()), label


def test_tfw_only_geotiff_metric_uses_crs_undefined():
    with tempfile.TemporaryDirectory() as d, _Runner():
        r, c = _stratified_rowcol()
        ref = _ref(Path(d) / "ref.npy", r, c, _true_elev(r, c))
        out = Path(d) / "o"
        code = _run([_geotiff(Path(d) / "t.tif", epsg=None, tfw=True), "--reference", ref, "--out", out])
        assert code == 0
        man = _manifest(out)
        assert man["branch"] == "metric"
        assert man["crs"] == CRS_UNDEFINED
        assert load_geotiff(out / "t_metric.tif").epsg == CRS_UNDEFINED


def test_a_non_positive_fit_is_exit_2_nothing_written():
    with tempfile.TemporaryDirectory() as d, _Runner():
        r, c = _stratified_rowcol()
        elev = -2.0 * (0.01 * (r + c)) + 50.0          # anti-correlated -> a < 0
        ref = _ref(Path(d) / "ref.npy", r, c, elev)
        out = Path(d) / "o"
        assert _run([_geotiff(Path(d) / "g.tif", epsg=EPSG_OK),
                     "--reference", ref, "--out", out]) == rp.EXIT_BAD_INPUT
        assert not out.exists() or not any(out.iterdir())


def test_non_finite_fit_is_exit_2():
    saved = rp.fit_scale_shift_ransac
    rp.fit_scale_shift_ransac = lambda surface, points: (float("nan"), 1.0,
                                                        {"method": "stub", "n_points": len(points),
                                                         "inliers": None, "a": float("nan"), "b": 1.0})
    try:
        with tempfile.TemporaryDirectory() as d, _Runner():
            r, c = _stratified_rowcol()
            ref = _ref(Path(d) / "ref.npy", r, c, _true_elev(r, c))
            out = Path(d) / "o"
            assert _run([_geotiff(Path(d) / "g.tif", epsg=EPSG_OK),
                         "--reference", ref, "--out", out]) == rp.EXIT_BAD_INPUT
            assert not out.exists() or not any(out.iterdir())
    finally:
        rp.fit_scale_shift_ransac = saved


def test_ransac_exception_falls_back_to_ols():
    """#24: RANSAC failure -> metrics.fit_scale_shift, not a silent third path."""
    import sklearn.linear_model as sklm
    saved = sklm.RANSACRegressor

    def _boom(*a, **k):
        raise RuntimeError("forced RANSAC failure")

    sklm.RANSACRegressor = _boom
    try:
        surface = _ramp_surface(WIN, WIN)
        r, c = _stratified_rowcol()
        pts = np.column_stack([r, c, _true_elev(r, c)])
        a, b, meta = rp.fit_scale_shift_ransac(surface, pts)
        assert meta["method"].startswith("ols_fallback")
        assert np.isfinite(a) and np.isfinite(b)
        assert abs(a - A_TRUE) < 1e-4 and abs(b - B_TRUE) < 1e-2
    finally:
        sklm.RANSACRegressor = saved


def test_relative_and_metric_filenames_never_collide():
    with tempfile.TemporaryDirectory() as d, _Runner():
        r, c = _stratified_rowcol()
        ref = _ref(Path(d) / "ref.npy", r, c, _true_elev(r, c))
        out = Path(d) / "o"
        _run([_geotiff(Path(d) / "s.tif", epsg=EPSG_OK), "--reference", ref, "--out", out])
        names = {p.name for p in out.iterdir()}
        rel = {n for n in names if n.startswith("relative_dsm_") or n == "s_relative.glb"
               or n == "s_relative_lod.glb" or n == "s_preview.png"}
        met = {n for n in names if "_metric" in n}
        assert rel and met and rel.isdisjoint(met)
        assert "relative_dsm_s.npy" in names and "s_metric.tif" in names


def test_no_hardcoded_crs_literal_in_run_pipeline():
    """run_pipeline.py must route CRS through geoio, never a numeric literal.

    tests/test_no_hardcoded_crs.py only scans src/ scripts/ PHASE2_RED_FLAGS/scripts/
    tests/ - run_pipeline.py sits at the repo root, so this guards the new CRS code.
    The banned values are built arithmetically so this file itself carries no literal.
    """
    import ast
    banned = {32000 + 633, 2 ** 15 - 1}          # the Vaihingen mis-CRS and CRS_UNDEFINED
    tree = ast.parse((ROOT / "run_pipeline.py").read_text())
    bad = [n.value for n in ast.walk(tree)
           if isinstance(n, ast.Constant) and n.value in banned]
    assert not bad, f"numeric CRS literal in run_pipeline.py: {bad}"


# ------------------------------------------------------- real mesh / LOD integration
def test_real_relative_glb_and_lod_are_written_and_loadable():
    import trimesh
    with tempfile.TemporaryDirectory() as d, _Runner(real_meshes=True):
        out = Path(d) / "o"
        assert _run([_png(Path(d) / "r.png"), "--out", out]) == 0
        for name in ("r_relative.glb", "r_relative_lod.glb"):
            m = trimesh.load(str(out / name), process=False, force="mesh")
            assert len(m.faces) > 0
        man = _manifest(out)
        assert man["mesh"]["relative"]["faces"] > man["mesh"]["relative"]["lod_faces"]


def test_real_metric_glb_lod_keeps_texture_and_uv():
    import trimesh
    with tempfile.TemporaryDirectory() as d, _Runner(real_meshes=True):
        r, c = _stratified_rowcol()
        ref = _ref(Path(d) / "ref.npy", r, c, _true_elev(r, c))
        out = Path(d) / "o"
        assert _run([_geotiff(Path(d) / "m.tif", epsg=EPSG_OK), "--reference", ref, "--out", out]) == 0
        native = trimesh.load(str(out / "m_metric.glb"), process=False, force="mesh")
        lod = trimesh.load(str(out / "m_metric_lod.glb"), process=False, force="mesh")
        assert type(native.visual).__name__ == "TextureVisuals"
        assert type(lod.visual).__name__ == "TextureVisuals"
        assert lod.visual.uv is not None and lod.visual.material.baseColorTexture is not None
        assert len(lod.faces) < len(native.faces)


# --------------------------------------------------------------- load_georaster (#71)
# G21 fix: a genuine geoio.load_geotiff() read/parse failure must raise PipelineError,
# never be silently collapsed into `load_georaster() -> None` (which the runner treats
# identically to "this input just isn't a GeoTIFF" - a fixable failure and a legitimately
# non-georeferenced input must not share one code path).

def test_load_georaster_non_tiff_returns_none():
    """Unchanged behavior: a non-.tif/.tiff path is never even handed to geoio - it's not
    a read failure, it's simply not a georaster."""
    with tempfile.TemporaryDirectory() as d:
        p = _png(Path(d) / "in.png")
        assert rp.load_georaster(p) is None


def test_load_georaster_untagged_tiff_returns_georeferenced_false():
    """Unchanged behavior: a valid TIFF with no embedded tags and no .tfw sidecar is the
    legitimate 'no georeference available' case - geoio.load_geotiff() itself handles this
    by falling back (georeferenced=False), raising nothing. load_georaster() must keep
    returning that GeoRaster as-is, not treat the fallback as a failure."""
    with tempfile.TemporaryDirectory() as d:
        p = _geotiff(Path(d) / "untagged.tif", epsg=None, tfw=False)
        source = rp.load_georaster(p)
        assert source is not None
        assert source.georeferenced is False


def test_load_georaster_wraps_load_geotiff_failure_as_pipeline_error():
    """The G21 fix itself: any other exception from geoio.load_geotiff() (corrupt file,
    permission error, parser exception) must surface as PipelineError, never be silently
    swallowed into a `None` return that the caller can't distinguish from '.tif with no
    georeference'. Mocks geoio.load_geotiff directly - load_georaster() re-imports it by
    name on every call, so patching the geoio module attribute is picked up correctly."""
    import geoio

    def _boom(path, **kwargs):
        raise OSError("simulated corrupt/unreadable raster")

    with tempfile.TemporaryDirectory() as d:
        p = _geotiff(Path(d) / "corrupt.tif", epsg=None, tfw=False)
        original = geoio.load_geotiff
        geoio.load_geotiff = _boom
        try:
            try:
                rp.load_georaster(p)
            except rp.PipelineError as exc:
                assert "corrupt/unreadable raster" in str(exc)
            else:
                raise AssertionError(
                    "a genuine load_geotiff() failure was silently swallowed to None "
                    "instead of raising PipelineError")
        finally:
            geoio.load_geotiff = original


def test_malformed_tfw_sidecar_is_exit_2_no_output():
    """End-to-end (#71 acceptance criteria): a GeoTIFF with no embedded tags, paired with
    a malformed .tfw sidecar (non-numeric content geoio.read_tfw() cannot parse), must
    make the whole pipeline exit 2 - not silently degrade to relative-mode output. No
    output directory/artifacts may be created."""
    with tempfile.TemporaryDirectory() as d, _Runner():
        img = _geotiff(Path(d) / "bad_tfw.tif", epsg=None, tfw=False)
        img.with_suffix(".tfw").write_text("not a number\nalso not a number\n")
        out = Path(d) / "o"
        code = _run([img, "--out", out])
        assert code == 2, f"expected exit 2 for a malformed .tfw sidecar, got {code}"
        assert not out.exists(), "no output directory/artifacts may be created on this failure"


# -------------------------------------------------------- to_mesh NaN/nodata guard (#87)
# heightfield_to_mesh() itself is FROZEN v1 (src/mesh.py's own module docstring -- a
# change there needs a v2 label + a note to Ashmita/Aishee), so the guard lives in
# run_pipeline.py::to_mesh(), the project's own thin wrapper, not in the frozen function.

def test_to_mesh_raises_pipeline_error_on_nan_elevation():
    """The fix itself: heightfield_to_mesh() has no NaN check of its own (Z = (elev -
    elev.min()) * z_exaggeration silently propagates NaN since elev.min() itself returns
    NaN when any element is NaN) -- to_mesh() must refuse loudly before calling it,
    rather than emitting degenerate/invisible geometry with no warning."""
    with tempfile.TemporaryDirectory() as d:
        elev = np.full((16, 16), 5.0, dtype=np.float32)
        elev[3, 4] = np.nan
        rgb = np.zeros((16, 16, 3), dtype=np.uint8)
        try:
            rp.to_mesh(elev, rgb, gsd=0.1, out_path=Path(d) / "bad.glb")
        except rp.PipelineError as exc:
            assert "1" in str(exc) and ("non-finite" in str(exc) or "nan" in str(exc).lower())
        else:
            raise AssertionError("a NaN elevation value was silently accepted by to_mesh()")


def test_to_mesh_raises_pipeline_error_on_inf_elevation():
    """Inf is exactly as dangerous as NaN here (also produces degenerate geometry) -- the
    guard must catch both, not just NaN specifically."""
    with tempfile.TemporaryDirectory() as d:
        elev = np.full((16, 16), 5.0, dtype=np.float32)
        elev[0, 0] = np.inf
        rgb = np.zeros((16, 16, 3), dtype=np.uint8)
        try:
            rp.to_mesh(elev, rgb, gsd=0.1, out_path=Path(d) / "bad.glb")
        except rp.PipelineError:
            pass
        else:
            raise AssertionError("an Inf elevation value was silently accepted by to_mesh()")


def test_to_mesh_raises_pipeline_error_on_nodata_sentinel():
    """Review catch (PR #93, Anmol): geoio.DSM_NODATA (-9999.0) is finite, so it passes the
    NaN/Inf check above, but is not real elevation data -- if it reaches heightfield_to_mesh()
    unguarded, it silently produces invalid geometry while the source GeoTIFF marks that value
    as nodata. Must be caught with the project's existing nodata tolerance
    (geoio.dsm_nodata_mask()), not a new bespoke check."""
    from geoio import DSM_NODATA
    with tempfile.TemporaryDirectory() as d:
        elev = np.full((16, 16), 5.0, dtype=np.float32)
        elev[7, 2] = DSM_NODATA
        rgb = np.zeros((16, 16, 3), dtype=np.uint8)
        try:
            rp.to_mesh(elev, rgb, gsd=0.1, out_path=Path(d) / "bad.glb")
        except rp.PipelineError as exc:
            assert "1" in str(exc) and "nodata" in str(exc).lower()
        else:
            raise AssertionError("a DSM_NODATA sentinel value was silently accepted by to_mesh()")


def test_to_mesh_still_works_normally_on_finite_elevation():
    """Regression check: the guard must not reject genuinely valid, all-finite input --
    exercises the REAL (unstubbed) heightfield_to_mesh(), not a test double."""
    with tempfile.TemporaryDirectory() as d:
        yy, xx = np.mgrid[0:16, 0:16].astype(np.float32)
        elev = 100.0 + 0.5 * (xx + yy)  # smooth, fully finite
        rgb = np.random.default_rng(3).integers(0, 255, (16, 16, 3), dtype=np.uint8)
        out_path = Path(d) / "good.glb"
        mesh, returned_path = rp.to_mesh(elev, rgb, gsd=0.1, out_path=out_path)
        assert returned_path == out_path
        assert out_path.exists() and out_path.stat().st_size > 0
        assert len(mesh.vertices) == 16 * 16


def _manifest(out_dir):
    import json
    return json.loads((Path(out_dir) / "run_manifest.json").read_text())


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as exc:              # noqa: BLE001
            failed += 1
            import traceback
            print(f"  FAIL  {t.__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
