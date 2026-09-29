"""Tests for src/mesh.py - DW-S1-07 (#24) Day-3.

`heightfield_to_mesh()` must stay byte-for-behaviour identical to the frozen canonical
implementation; `make_lod()` must reduce faces when asked and keep the texture + UV that
`simplify_quadric_decimation()` would otherwise drop (issue #12 / PR #17).

No checkpoint, no network, no Vaihingen data - a small synthetic terrain tile only.

    python tests/test_mesh.py
    pytest tests/test_mesh.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mesh import LOD_TARGET_FACES, heightfield_to_mesh, make_lod  # noqa: E402

# The frozen canonical copy, for an identical-behaviour cross-check.
sys.path.insert(0, str(ROOT / "PHASE2_RED_FLAGS" / "scripts"))
from heightfield_lib import heightfield_to_mesh as canonical_heightfield_to_mesh  # noqa: E402


def _synthetic_tile(H=120, W=160, seed=7):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    elev = (10 * np.sin(xx / 25.0) + 8 * np.cos(yy / 30.0)
            + 5 * np.sin((xx + yy) / 40.0) + 15).astype(np.float32)
    rgb = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
    return elev, rgb


def test_heightfield_to_mesh_matches_the_frozen_canonical_implementation():
    """src/mesh.py must be a verbatim extraction, not a variant."""
    elev, rgb = _synthetic_tile()
    a = heightfield_to_mesh(elev, rgb, gsd=0.09, z_exaggeration=1.5, downsample=1)
    b = canonical_heightfield_to_mesh(elev, rgb, gsd=0.09, z_exaggeration=1.5, downsample=1)
    assert np.array_equal(a.vertices, b.vertices), "vertex positions diverged from canonical"
    assert np.array_equal(a.faces, b.faces), "face topology diverged from canonical"
    assert np.allclose(a.visual.uv, b.visual.uv), "UV diverged from canonical"
    assert type(a.visual).__name__ == "TextureVisuals"


def test_heightfield_to_mesh_geometry_contract():
    """Grid -> 2 triangles per cell, textured, Z rebased to the tile minimum."""
    elev, rgb = _synthetic_tile(H=50, W=64)
    mesh = heightfield_to_mesh(elev, rgb, gsd=0.09, z_exaggeration=1.0)
    assert mesh.vertices.shape == (50 * 64, 3)
    assert mesh.faces.shape == (2 * 49 * 63, 3)
    assert mesh.visual.material.baseColorTexture is not None
    assert mesh.visual.uv.shape == (50 * 64, 2)
    # X spans [0, (W-1)*gsd], Y is negated, Z starts at 0 (rebased to elev.min())
    assert np.isclose(mesh.vertices[:, 2].min(), 0.0)
    assert mesh.vertices[:, 1].max() <= 1e-9
    assert np.isclose(mesh.vertices[:, 0].max(), (64 - 1) * 0.09)


def test_make_lod_reduces_faces_and_keeps_texture_through_a_glb_roundtrip():
    """The issue #12 failure mode: decimation drops UV + material. PR #17 recomputes them."""
    elev, rgb = _synthetic_tile(H=200, W=260)
    native = heightfield_to_mesh(elev, rgb, gsd=0.09, z_exaggeration=1.5)
    target = len(native.faces) // 10
    lod = make_lod(native, target_faces=target)

    assert len(lod.faces) < len(native.faces), "LOD did not reduce the face count"
    assert len(lod.faces) <= target * 1.2, f"LOD overshot target badly: {len(lod.faces)} vs {target}"
    assert type(lod.visual).__name__ == "TextureVisuals", "LOD lost its TextureVisuals"
    assert lod.visual.uv is not None and len(lod.visual.uv) == len(lod.vertices)
    assert lod.visual.material.baseColorTexture is not None

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "lod.glb"
        lod.export(str(p))
        reloaded = trimesh.load(str(p), process=False, force="mesh")
    assert type(reloaded.visual).__name__ == "TextureVisuals", "texture did not survive the .glb round-trip"
    assert reloaded.visual.uv is not None
    assert reloaded.visual.material.baseColorTexture is not None


def test_make_lod_uv_is_algebraically_exact_against_the_grid_formula():
    """Not just non-null: the recomputed UV must match heightfield_to_mesh()'s own
    u = col/(W-1), v = 1 - row/(H-1) for every surviving vertex."""
    H, W, gsd = 180, 240, 0.09
    elev, rgb = _synthetic_tile(H=H, W=W)
    native = heightfield_to_mesh(elev, rgb, gsd=gsd, z_exaggeration=1.5)
    lod = make_lod(native, target_faces=len(native.faces) // 8)

    v = lod.vertices
    col = v[:, 0] / gsd
    row = -v[:, 1] / gsd
    expected_u = col / (W - 1)
    expected_v = 1.0 - row / (H - 1)
    got = lod.visual.uv
    assert np.max(np.abs(got[:, 0] - expected_u)) < 1e-5
    assert np.max(np.abs(got[:, 1] - expected_v)) < 1e-5


def test_make_lod_is_a_noop_when_already_below_target():
    """A mesh already small enough must be returned untouched - no decimation, texture intact."""
    elev, rgb = _synthetic_tile(H=40, W=48)
    native = heightfield_to_mesh(elev, rgb, gsd=0.09)
    assert len(native.faces) < LOD_TARGET_FACES
    out = make_lod(native, target_faces=LOD_TARGET_FACES)
    assert out is native
    assert type(out.visual).__name__ == "TextureVisuals"


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
