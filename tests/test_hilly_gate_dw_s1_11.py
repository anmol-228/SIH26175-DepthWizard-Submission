"""DW-S1-11 (#32) - hilly-risk gate continuation, end-to-end. Now includes DW-P0-A5.

Scope A (a validated hilly-relief runtime signature computed from the model's OWN relative
output) was attempted and rejected - see `src/preflight.py`'s `HILLY_SIGNATURE_EVIDENCE` and
the canonical methodology checkpoint, PR #35 (`(internal note, not included)`):
three candidate signals tested scene-consistently against Mount Adams and four Vaihingen
controls through the real production predict/blend/normalize path, all rejected or (for the
one with any separation) classified "more evidence required" on a 2-sample cohort, not a
threshold. This file does not test that signature - it stays unimplemented, and the two
tests below still confirm its manifest documentation is unchanged.

DW-P0-A5 (#32, 2026-09-11 Phase-2 activation) is a *different, bounded* signal that DOES
ship: `reference_elevation_relief()` reads only the elevation values a `--reference` file
already carries (no model output at all) - see `HILLY_RELIEF_EVIDENCE`. This file confirms:

  1. The `run_manifest.json` produced by a real CLI run documents both #32/PR #35's
     rejected-signal finding AND the new relief check's decision (not just the unit-level
     `preflight.py` docstrings) - a consumer reading the manifest, not the source, sees both.
  2. The existing DW-S1-10 (#25) failure-aware behaviour - gate decline -> relative-only,
     exit 0, no metric artifact - still holds with #32's changes layered on top.
  3. NEW: a well-spread reference set (passes count/hull/spread) whose own elevations span
     more than the provisional relief floor now DOES get caught, on relief specifically -
     narrowing the caveat DW-S1-11 documented ("a genuinely-hilly-but-well-spread reference
     set would NOT be caught by it"): it is caught whenever the supplied points happen to
     read a relief above the floor, not in every case (a hilly scene sampled in a narrow
     elevation band still passes undetected). See `test_manifest_documents_reference_relief_gate_decline`.

    python tests/test_hilly_gate_dw_s1_11.py
    pytest tests/test_hilly_gate_dw_s1_11.py
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import run_pipeline as rp  # noqa: E402
from preflight import (  # noqa: E402
    HILLY_RELIEF_EVIDENCE, HILLY_SIGNATURE_EVIDENCE, MIN_NORMALIZED_SPREAD,
)

WIN = rp.DEFAULT_WINDOW
EPSG_OK = 32601


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


def _geotiff(path, *, epsg, h=WIN, w=WIN):
    rgb = np.random.default_rng(2).integers(0, 255, (h, w, 3), dtype=np.uint8)
    extratags = [
        (33550, "d", 3, (0.09, 0.09, 0.0), True),
        (33922, "d", 6, (0.0, 0.0, 0.0, 100.0, 200.0, 0.0), True),
        (34735, "H", 20, (1, 1, 0, 4, 1024, 0, 1, 1, 1025, 0, 1, 1,
                          3072, 0, 1, int(epsg), 3076, 0, 1, 9001), True),
    ]
    tifffile.imwrite(str(path), rgb, photometric="rgb", extratags=extratags)
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


def _true_elev(rows, cols, a=3.0, b=200.0):
    return a * (0.01 * (rows + cols)) + b


def _hilly_elev(rows, lo=200.0, relief_m=200.0):
    """DW-P0-A5: elevations spanning `relief_m` metres across the row axis - well above the
    provisional 87.5 m floor, isolated from row/col spread so geometry checks pass on their
    own (mirrors `tests/test_preflight.py`'s `_stratified_hilly`)."""
    rows = np.asarray(rows, dtype=float)
    span = rows.max() - rows.min()
    return lo + relief_m * (rows - rows.min()) / span


def _manifest(out_dir):
    return json.loads((Path(out_dir) / "run_manifest.json").read_text())


def test_manifest_documents_dw_s1_11_attempt_on_a_real_passing_run():
    """A real metric-branch run's manifest must carry #32's evidence, not just source code.
    DW-P0-A5: also confirms the new relief check ran and reported this scene's supplied
    reference points within the provisional floor (their own elevations span well under
    it) - a within-floor reading, not a certification that the scene is safe in general."""
    with tempfile.TemporaryDirectory() as d, _Runner():
        r, c = _stratified_rowcol()
        ref = _ref(Path(d) / "ref.npy", r, c, _true_elev(r, c))
        out = Path(d) / "o"
        code = rp.main([str(_geotiff(Path(d) / "flat.tif", epsg=EPSG_OK)),
                        "--reference", str(ref), "--out", str(out)])
        assert code == 0
        man = _manifest(out)
        assert man["branch"] == "metric"
        checks = man["preflight"]["checks"]
        assert checks["hilly_signature"] is None
        assert "rejected" in checks["hilly_signature_status"]
        assert "#32" in checks["hilly_signature_status"]
        assert checks["hilly_signature_evidence"] == HILLY_SIGNATURE_EVIDENCE
        assert "PR #35" in checks["hilly_signature_evidence"]
        # DW-P0-A5: `_true_elev`'s relief here is ~26 m (0.03 * span of row+col), well under
        # the 87.5 m provisional floor - the new check ran and reported it within-floor.
        assert checks["reference_relief_m"] is not None
        assert checks["reference_relief_m"] < checks["reference_relief_threshold_m"]
        assert checks["reference_relief_status"] == "within provisional floor"
        assert checks["reference_relief_evidence"] == HILLY_RELIEF_EVIDENCE


def test_manifest_documents_dw_s1_11_attempt_on_a_real_gate_decline():
    """Same evidence must be present on the relative-only (gate-declined) path too - the
    documentation is not conditional on which branch the run took. This scene declines on
    geometry (clustered points), not relief - `test_manifest_documents_reference_relief_gate_decline`
    below is the relief-specific decline."""
    with tempfile.TemporaryDirectory() as d, _Runner():
        rng = np.random.default_rng(42)
        r = np.round(rng.uniform(40, 90, 12)); c = np.round(rng.uniform(40, 90, 12))
        ref = _ref(Path(d) / "ref.npy", r, c, _true_elev(r, c))
        out = Path(d) / "o"
        code = rp.main([str(_geotiff(Path(d) / "clustered.tif", epsg=EPSG_OK)),
                        "--reference", str(ref), "--out", str(out)])
        assert code == 0
        man = _manifest(out)
        assert man["branch"] == "relative"
        assert man["preflight"]["pass"] is False
        assert not list(out.glob("*_metric.*")), "gate decline must not emit a metric artifact"
        checks = man["preflight"]["checks"]
        assert "rejected" in checks["hilly_signature_status"]
        # DoD: "no claim that hilly calibration is solved" - the OLD model-output signature
        # family (`hilly_signature`) stays unsupported; it declines only on geometry
        # (count/coverage/spread), never on relief content. This test's clustered points
        # fail on geometry - documenting the OLD signal's real, narrower basis. The NEW
        # DW-P0-A5 relief check also ran here (this scene's tiny relief, ~3 m, is well
        # under the floor) but is not why this particular run declined.
        assert checks["hilly_signature"] is None
        assert checks["reference_relief_status"] in (
            "within provisional floor", "relief-risk gate triggered - see reason")
        assert "count" in man["gate_reason"] or "cover" in man["gate_reason"] or \
            "spread" in man["gate_reason"]


def test_manifest_documents_reference_relief_gate_decline():
    """DW-P0-A5 core case: a well-spread reference set (identical row/col geometry to the
    passing test above, which already proves count/hull/spread all clear) whose own
    elevations span more than the provisional relief floor. Narrows the caveat DW-S1-11
    documented: this is exactly the "genuinely-hilly-but-well-spread reference set" the old
    signal family could not catch (it isn't relief-aware); this new, separate check catches
    it here because the supplied points happen to read a relief above the floor - not a
    claim that every such case is caught (see HILLY_RELIEF_EVIDENCE's disclosed limitation)."""
    with tempfile.TemporaryDirectory() as d, _Runner():
        r, c = _stratified_rowcol()
        ref = _ref(Path(d) / "ref.npy", r, c, _hilly_elev(r))
        out = Path(d) / "o"
        code = rp.main([str(_geotiff(Path(d) / "hilly.tif", epsg=EPSG_OK)),
                        "--reference", str(ref), "--out", str(out)])
        assert code == 0, "a firing gate is exit 0, not an error"
        man = _manifest(out)
        assert man["branch"] == "relative"
        assert man["preflight"]["pass"] is False
        assert not list(out.glob("*_metric.*")), "gate decline must not emit a metric artifact"
        assert "relief" in man["gate_reason"]
        checks = man["preflight"]["checks"]
        # the decline is attributable to relief, not geometry - same row/col grid as the
        # passing test above, which already proves this geometry clears count/hull/spread
        assert checks["hull_coverage"] > 0.15
        assert checks["spread"] > MIN_NORMALIZED_SPREAD
        assert checks["reference_relief_status"] == "relief-risk gate triggered - see reason"
        assert checks["reference_relief_m"] is not None
        assert checks["reference_relief_m"] > checks["reference_relief_threshold_m"]
        assert checks["reference_relief_evidence"] == HILLY_RELIEF_EVIDENCE
        # the OLD (still-rejected) model-output signal is untouched by this new check
        assert checks["hilly_signature"] is None
        assert "rejected" in checks["hilly_signature_status"]


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
