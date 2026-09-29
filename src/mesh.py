"""Mesh + LOD helpers for run_pipeline.py - DW-S1-07 (#24) Day-3.

This module is a **faithful extraction**, not a reimplementation:

  `heightfield_to_mesh()` is copied VERBATIM from the frozen canonical implementation in
  `PHASE2_RED_FLAGS/scripts/K3_mesh_pipeline.py` (contract `fixtures/pipeline_to_3d_v1.json`,
  "v1 / FROZEN"), the same copy that already lives in
  `PHASE2_RED_FLAGS/scripts/heightfield_lib.py`. Its geometry, texture, UV, GSD, Y
  orientation, z_exaggeration and `process=False` behaviour are unchanged. A change to the
  frozen interface requires a v2 label and a note to Ashmita/Aishee - not an edit here.

  `make_lod()` extracts the accepted simplification + post-decimation UV/texture
  restoration from `PHASE2_RED_FLAGS/scripts/K3b_make_lod.py` (issue #12 loss of UV on
  edge-collapse; PR #17 fix, verified by
  `PHASE2_RED_FLAGS/scripts/test_issue12_lod_texture_fix.py`). The UV recompute is exact
  (not approximate) for this pipeline because `heightfield_to_mesh()` derives UV purely
  from grid position, an affine function of world X/Y, so any surviving vertex's UV is
  recoverable from its (x, y) alone.

Both meshes are visualization artifacts: `heightfield_to_mesh()` rebases Z to the tile
minimum, so a `.glb` does not carry the absolute vertical datum offset, and the Y axis is
negated as a viewer display heuristic. The measurement product on the metric branch is the
GeoTIFF, not the mesh.
"""
import numpy as np
import trimesh
from PIL import Image


def heightfield_to_mesh(elev: np.ndarray, rgb: np.ndarray, gsd: float, z_exaggeration: float = 1.0,
                          downsample: int = 1):
    """elev: (H,W) meters. rgb: (H,W,3) uint8. Returns a textured trimesh.Trimesh."""
    if downsample > 1:
        elev = elev[::downsample, ::downsample]
        rgb = rgb[::downsample, ::downsample]
    H, W = elev.shape
    xs = np.arange(W) * gsd * downsample
    ys = np.arange(H) * gsd * downsample
    X, Y = np.meshgrid(xs, ys)
    Z = (elev - elev.min()) * z_exaggeration
    vertices = np.stack([X.ravel(), -Y.ravel(), Z.ravel()], axis=1)  # -Y so north-up looks right in viewer

    # triangulate the regular grid (2 triangles per cell)
    idx = np.arange(H * W).reshape(H, W)
    v00 = idx[:-1, :-1].ravel(); v10 = idx[:-1, 1:].ravel()
    v01 = idx[1:, :-1].ravel(); v11 = idx[1:, 1:].ravel()
    tri1 = np.stack([v00, v01, v10], axis=1)
    tri2 = np.stack([v10, v01, v11], axis=1)
    faces = np.concatenate([tri1, tri2], axis=0)

    # UVs: standard [0,1] image-space mapping
    u = (np.arange(W) / max(W - 1, 1))
    v = 1.0 - (np.arange(H) / max(H - 1, 1))
    UU, VV = np.meshgrid(u, v)
    uvs = np.stack([UU.ravel(), VV.ravel()], axis=1)

    texture_img = Image.fromarray(rgb)
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=texture_img)
    visual = trimesh.visual.TextureVisuals(uv=uvs, material=material)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)
    return mesh


# Historically evidenced LOD target: the native large Vaihingen mesh (~10.67M faces /
# ~242.8 MB) was decimated to this for the K4 browser test. Recorded size evidence for
# that run varies (~9 MB vs ~18.9 MB across records), so this is a default face budget,
# NOT a guaranteed output size.
LOD_TARGET_FACES = 500_000


def make_lod(mesh, target_faces: int = LOD_TARGET_FACES):
    """Decimate a textured heightfield mesh toward `target_faces`, restoring UV + texture.

    Behaviour carried faithfully from K3b_make_lod.py (PR #17):
    `simplify_quadric_decimation()` reindexes vertices and drops per-vertex UV + material,
    so UV is recomputed from each surviving vertex's XY position - exact for this
    pipeline's grid-position UV convention - and the base-colour texture is reused
    directly from `mesh`.

    If `mesh` already has <= `target_faces`, it is returned unchanged (no decimation, no
    UV/texture disturbance).
    """
    if len(mesh.faces) <= target_faces:
        return mesh

    simplified = mesh.simplify_quadric_decimation(face_count=target_faces)

    had_texture = (
        getattr(mesh.visual, "uv", None) is not None
        and getattr(getattr(mesh.visual, "material", None), "baseColorTexture", None) is not None
    )
    if had_texture:
        v = simplified.vertices
        x0, x1 = mesh.vertices[:, 0].min(), mesh.vertices[:, 0].max()
        y0, y1 = mesh.vertices[:, 1].min(), mesh.vertices[:, 1].max()  # y is negative (see heightfield_to_mesh)
        u = (v[:, 0] - x0) / (x1 - x0)
        vv = 1.0 - (y1 - v[:, 1]) / (y1 - y0)
        uv = np.clip(np.stack([u, vv], axis=1), 0.0, 1.0)
        simplified.visual = trimesh.visual.TextureVisuals(
            uv=uv,
            material=trimesh.visual.material.PBRMaterial(
                baseColorTexture=mesh.visual.material.baseColorTexture
            ),
        )
    else:
        # Defensive: should not happen for a real heightfield_to_mesh() export. Do not
        # silently ship an untextured LOD without flagging it.
        print("WARNING: make_lod() input has no TextureVisuals/baseColorTexture to carry "
              "forward - the LOD will be untextured. Investigate before shipping it.")
    return simplified
