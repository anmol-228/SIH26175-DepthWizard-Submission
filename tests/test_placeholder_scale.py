"""Placeholder mesh-scale <-> sidecar-scale agreement - DW-WP2 (issue #26).

Regression test for a real bug caught in PR #44 review (Anmol, 2026-09-08):
`viewer/make_placeholder_glb.py` built the exported mesh with `gsd=0.09` while its
`write_surface_meta()` call declared `pseudo_gsd=1.0` - two different numbers for what
must be the SAME scale, since `heightfield_to_mesh()`'s vertex mapping
(`x=col*gsd, y=-row*gsd`) and the viewer's `worldToGrid()` inverse
(`row=-y/pseudo_gsd, col=x/pseudo_gsd`) must agree on which "gsd" they mean. Every
picked point on the placeholder was silently off by a factor of ~11.1x (1/0.09).

This test runs the real generator (not a stub) and checks the ACTUAL exported mesh
geometry against the ACTUAL sidecar's declared scale - it would have caught the bug
that unit tests of `write_surface_meta()` alone cannot, since that function was never
the mismatch; the two call sites disagreeing was.

DW-P2 (issue #50): the generator is invoked with `--out-dir <tmp_path>` (added
specifically for this), so every run below regenerates into a throwaway pytest
tmp_path directory rather than overwriting the tracked `viewer/placeholder_relative.*`
fixture on every test run. `test_committed_fixture_is_semantically_consistent_with_a_
fresh_regeneration` below is the one place that still reads the committed fixture, and
only to compare it - it never regenerates over it. Real reproduction of the original
bug (`tests/test_placeholder_scale.py::_regenerate()` dirtying the tracked binary,
canonical blob `a3831c07...` vs regenerated `54c6eb77...`, semantically identical
geometry, 24-byte serialization-level difference) is preserved in issue #50's history.

    python tests/test_placeholder_scale.py
    pytest tests/test_placeholder_scale.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
VIEWER = ROOT / "viewer"
GENERATOR = VIEWER / "make_placeholder_glb.py"
COMMITTED_GLB = VIEWER / "placeholder_relative.glb"


def _regenerate(out_dir: Path):
    """Run the real generator script as a subprocess - not imported, since it has
    module-level side effects at parse time (sys.path manipulation) and this test
    wants to exercise exactly what a human running `python viewer/make_placeholder_glb.py`
    gets, just redirected to a throwaway directory via --out-dir (issue #50) instead of
    always overwriting the tracked viewer/ fixture."""
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--out-dir", str(out_dir)], cwd=str(ROOT),
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"generator failed:\n{result.stdout}\n{result.stderr}"


def _load_native_geometry(glb_path: Path):
    mesh = trimesh.load(str(glb_path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = list(mesh.geometry.values())[0]
    return mesh


def test_placeholder_mesh_vertex_extent_matches_declared_pseudo_gsd(tmp_path):
    """The exported mesh's own vertex range is the ground truth for what scale was
    ACTUALLY used to build it - compare that against what the sidecar SAYS was used."""
    _regenerate(tmp_path)
    meta = json.loads((tmp_path / "placeholder_relative_surface_meta.json").read_text())
    assert meta["gsd"] is None and meta["pseudo_gsd"] is not None, (
        "placeholder is relative-branch only; expected gsd=None, pseudo_gsd set")
    declared_scale = meta["pseudo_gsd"]
    H, W = meta["H"], meta["W"]

    mesh_path = tmp_path / meta["mesh_glb"]
    mesh = _load_native_geometry(mesh_path)
    v = mesh.vertices

    expected_x_max = (W - 1) * declared_scale
    expected_y_min = -(H - 1) * declared_scale
    assert v[:, 0].max() == pytest_approx(expected_x_max), (
        f"mesh X extent {v[:, 0].max()} does not match declared pseudo_gsd={declared_scale} "
        f"(expected {expected_x_max}) - the mesh and its sidecar disagree on scale, "
        "exactly the PR #44 bug")
    assert v[:, 1].min() == pytest_approx(expected_y_min), (
        f"mesh Y extent {v[:, 1].min()} does not match declared pseudo_gsd={declared_scale} "
        f"(expected {expected_y_min})")


def test_world_to_grid_round_trip_against_the_real_exported_mesh(tmp_path):
    """Emulate viewer/sidecar.js's resolveHorizontalScale() + main.js's worldToGrid()
    inverse formula in Python, applied to the real exported mesh's own vertices, and
    confirm it recovers the (row, col) each vertex was actually built from. This is the
    round-trip check Anmol's review requested."""
    _regenerate(tmp_path)
    meta = json.loads((tmp_path / "placeholder_relative_surface_meta.json").read_text())
    scale = meta["gsd"] if meta["gsd"] is not None else meta["pseudo_gsd"]
    H, W = meta["H"], meta["W"]

    mesh = _load_native_geometry(tmp_path / meta["mesh_glb"])
    v = mesh.vertices

    # heightfield_to_mesh() lays vertices out row-major: vertex index = row*W + col.
    for row, col in [(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1), (H // 2, W // 2)]:
        idx = row * W + col
        x, y = float(v[idx, 0]), float(v[idx, 1])
        # worldToGrid()'s exact formula: row = -y/scale, col = x/scale
        recovered_row = -y / scale
        recovered_col = x / scale
        assert abs(recovered_row - row) < 1e-4, (
            f"round-trip failed at (row={row}, col={col}): recovered row {recovered_row}")
        assert abs(recovered_col - col) < 1e-4, (
            f"round-trip failed at (row={row}, col={col}): recovered col {recovered_col}")


def test_committed_fixture_is_semantically_consistent_with_a_fresh_regeneration(tmp_path):
    """Issue #50: the committed viewer/placeholder_relative.glb is never overwritten by
    this test suite. What it CAN and should still verify is that the committed fixture
    remains a valid, current output of the real generator - i.e. nobody hand-edited it
    or let it drift from what make_placeholder_glb.py actually produces today.

    Compared SEMANTICALLY (vertex positions, face indices), not byte-for-byte: a
    regenerated .glb can legitimately differ at the serialization level (e.g. export
    timestamp/metadata bytes in the container) while encoding IDENTICAL geometry - see
    issue #50's own reproduction (canonical vs regenerated blob differed by 24 bytes,
    trimesh-verified geometrically identical). A byte-for-byte assertion would be a
    false regression on exactly that kind of harmless serialization noise.
    """
    assert COMMITTED_GLB.exists(), f"committed fixture missing: {COMMITTED_GLB}"
    committed = _load_native_geometry(COMMITTED_GLB)

    _regenerate(tmp_path)
    fresh = _load_native_geometry(tmp_path / "placeholder_relative.glb")

    assert committed.vertices.shape == fresh.vertices.shape, (
        f"vertex count changed: committed {committed.vertices.shape} vs "
        f"fresh {fresh.vertices.shape} - regenerate and commit viewer/placeholder_relative.glb")
    assert committed.faces.shape == fresh.faces.shape, (
        f"face count changed: committed {committed.faces.shape} vs fresh {fresh.faces.shape} - "
        "regenerate and commit viewer/placeholder_relative.glb")

    max_vertex_abs_diff = float(np.abs(committed.vertices - fresh.vertices).max())
    assert max_vertex_abs_diff < 1e-6, (
        f"committed fixture's geometry has drifted from the real generator's current "
        f"output (max vertex abs diff {max_vertex_abs_diff}) - regenerate and commit "
        "viewer/placeholder_relative.glb (see make_placeholder_glb.py)")
    assert (committed.faces == fresh.faces).all(), (
        "committed fixture's face indices differ from a fresh regeneration - "
        "regenerate and commit viewer/placeholder_relative.glb")


def pytest_approx(expected, tol=1e-4):
    class _Approx:
        def __eq__(self, other):
            return abs(other - expected) < tol
        def __repr__(self):
            return f"~{expected}"
    return _Approx()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            with tempfile.TemporaryDirectory() as d:
                t(Path(d))
            print(f"  PASS  {t.__name__}")
        except Exception as exc:              # noqa: BLE001
            failed += 1
            print(f"  FAIL  {t.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
