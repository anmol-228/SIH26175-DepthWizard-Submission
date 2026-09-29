"""Standalone regression test for issue #12 (LOD texture loss). Uses a
synthetic terrain tile built with the real heightfield_to_mesh() from
K3_mesh_pipeline.py -- no dependency on the actual Vaihingen dataset, so this
can run anywhere, including CI.

What this proves, concretely:
  1. The bug reproduces on an unfixed decimation call (visual becomes
     ColorVisuals, uv/material become None) -- same failure mode as issue #12.
  2. The fix (recompute UV from post-decimation vertex XY, reuse the native
     mesh's texture) restores TextureVisuals with a real baseColorTexture,
     both immediately after decimation and after a full glb export/reload
     round-trip (i.e. what the viewer actually loads).
  3. The recomputed UVs are not just "non-null" -- they are algebraically
     exact against heightfield_to_mesh()'s own u=col/(W-1), v=1-row/(H-1)
     formula, AND sampling the real texture image at the recomputed UV
     reproduces the exact source pixel color.

Run: python test_issue12_lod_texture_fix.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from heightfield_lib import heightfield_to_mesh  # noqa: E402


def build_synthetic_tile(H=200, W=200, seed=42):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    # smooth, terrain-like elevation -- real DSM data is smooth, and iid noise
    # per-pixel would defeat decimation entirely (every edge collapse looks
    # high-error), so a smooth synthetic surface is the honest stand-in.
    elev = (10 * np.sin(xx / 25.0) + 8 * np.cos(yy / 30.0)
            + 5 * np.sin((xx + yy) / 40.0) + 15).astype(np.float32)
    rgb = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
    return elev, rgb, H, W


def main():
    elev, rgb, H, W = build_synthetic_tile()
    gsd = 0.09

    mesh_in_memory = heightfield_to_mesh(elev, rgb, gsd=gsd, z_exaggeration=1.5, downsample=1)
    tmp_dir = Path(tempfile.gettempdir())
    src_path = str(tmp_dir / "_issue12_test_native.glb")
    mesh_in_memory.export(src_path)

    mesh = trimesh.load(src_path, process=False, force="mesh")
    assert type(mesh.visual).__name__ == "TextureVisuals", "native mesh should load with a real texture"
    assert mesh.visual.material.baseColorTexture is not None

    target_faces = len(mesh.faces) // 20

    # --- 1. confirm the bug reproduces on an unfixed call ---
    buggy = mesh.simplify_quadric_decimation(face_count=target_faces)
    assert type(buggy.visual).__name__ == "ColorVisuals", (
        f"expected the known bug (ColorVisuals) on an unfixed decimation call, "
        f"got {type(buggy.visual).__name__} -- has trimesh's behavior changed?"
    )
    print(f"[1/3] Bug reproduces on unfixed decimation: faces={len(buggy.faces):,} "
          f"visual={type(buggy.visual).__name__} (expected ColorVisuals) -- PASS")

    # --- 2. apply the fix (mirrors the fixed K3b_make_lod.py block exactly) ---
    simplified = mesh.simplify_quadric_decimation(face_count=target_faces)
    v = simplified.vertices
    x0, x1 = mesh.vertices[:, 0].min(), mesh.vertices[:, 0].max()
    y0, y1 = mesh.vertices[:, 1].min(), mesh.vertices[:, 1].max()
    u = (v[:, 0] - x0) / (x1 - x0)
    vv = 1.0 - (y1 - v[:, 1]) / (y1 - y0)
    uv = np.clip(np.stack([u, vv], axis=1), 0.0, 1.0)
    simplified.visual = trimesh.visual.TextureVisuals(
        uv=uv,
        material=trimesh.visual.material.PBRMaterial(baseColorTexture=mesh.visual.material.baseColorTexture),
    )

    dst_path = str(tmp_dir / "_issue12_test_lod.glb")
    simplified.export(dst_path)
    reloaded = trimesh.load(dst_path, process=False, force="mesh")
    assert type(reloaded.visual).__name__ == "TextureVisuals"
    assert reloaded.visual.uv is not None
    assert reloaded.visual.material.baseColorTexture is not None
    print(f"[2/3] Fix restores texture, survives glb round-trip: faces={len(reloaded.faces):,} "
          f"visual={type(reloaded.visual).__name__} uv=True tex=True -- PASS")

    # --- 3. correctness, not just non-null: exact-formula check + real pixel-sample check ---
    max_formula_err = 0.0
    max_color_err = 0
    img = np.asarray(mesh.visual.material.baseColorTexture)
    imgH, imgW = img.shape[0], img.shape[1]
    step = max(1, len(v) // 25)
    for i in range(0, len(v), step):
        x, y = v[i, 0], v[i, 1]
        col = x / gsd
        row = -y / gsd
        expected_u = col / (W - 1)
        expected_v = 1.0 - row / (H - 1)
        got_u, got_v = uv[i]
        max_formula_err = max(max_formula_err, abs(got_u - expected_u) + abs(got_v - expected_v))

        col_i = min(max(int(round(col)), 0), W - 1)
        row_i = min(max(int(round(row)), 0), H - 1)
        expected_color = rgb[row_i, col_i]
        samp_col = int(round(got_u * (imgW - 1)))
        samp_row = int(round((1 - got_v) * (imgH - 1)))
        samp_color = img[samp_row, samp_col][:3]
        max_color_err = max(max_color_err, int(np.abs(samp_color.astype(int) - expected_color.astype(int)).sum()))

    assert max_formula_err < 1e-6, f"recomputed UV does not match the exact grid formula (err={max_formula_err})"
    assert max_color_err == 0, f"recomputed UV samples the wrong texture pixel (max channel err={max_color_err})"
    print(f"[3/3] UV correctness: exact-formula max_err={max_formula_err:.2e}, "
          f"real-pixel-sample max_channel_err={max_color_err} -- PASS")

    print("\nALL CHECKS PASSED -- issue #12 fix verified.")


if __name__ == "__main__":
    main()
