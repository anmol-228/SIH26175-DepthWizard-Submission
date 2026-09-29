"""Exact copy of heightfield_to_mesh() from
PHASE2_RED_FLAGS/scripts/K3_mesh_pipeline.py in the development repository,
fetched via GitHub raw view on 2026-08-29, for repro/testing purposes only.
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
