"""Decimate the native large mesh into an LOD (Level of Detail) variant for the
viewer FPS comparison in Issue #8 -- matches the 500,000-face target from the
prior K4_viewer_browser_test.json run.

Issue #12 fix (2026-08-29): trimesh's simplify_quadric_decimation() drops UV
coordinates and material data during edge-collapse (it reindexes vertices, so
per-vertex UVs cannot survive unmodified). Confirmed independently by Nirmalya
and Aishee via three checks: trimesh visual class, raw glTF TEXCOORD_0, and
visual inspection on two machines -- see issue #8 / PR #9 for the original
discussion, and issue #12 for the fix history.

Fix: recompute UV from each post-decimation vertex's XY position. This is
EXACT for this geometry (not an approximation) because heightfield_to_mesh()
in K3_mesh_pipeline.py derives UV purely from grid position --
u = col/(W-1), v = 1 - row/(H-1) -- which is an affine function of world X/Y.
So the UV of any surviving vertex, at any decimation level, is recoverable
from its (x, y) alone; no lookup or resampling is needed. The base color
texture itself is reused directly from the already-loaded native mesh
(mesh.visual.material.baseColorTexture) -- trimesh round-trips this correctly
through the glb export/import this script already does, so there is no need
to re-read the source raster tile.

Verified (synthetic repro, see issue #12 discussion): faces=3,960
visual=TextureVisuals uv=True material=PBRMaterial tex=True, both immediately
after decimation and after a full export/reload round-trip through .glb.
Recomputed UVs were checked two ways: (1) exact algebraic match against
heightfield_to_mesh()'s own u=col/(W-1), v=1-row/(H-1) formula for sampled
vertices (max error 0.0), and (2) sampling the actual texture image at the
recomputed UV reproduces the exact source pixel color for sampled vertices
(max channel error 0).
"""
import json
import time
from pathlib import Path

import numpy as np
import trimesh

P2ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = P2ROOT / "outputs" / "3d"
METRICS_DIR = P2ROOT / "outputs" / "metrics"

SRC = OUT_DIR / "large_area26.glb"
DST = OUT_DIR / "large_LOD_area26.glb"
TARGET_FACES = 500_000

mesh = trimesh.load(SRC, process=False, force="mesh")
print(f"loaded {SRC.name}: {len(mesh.faces):,} faces, {SRC.stat().st_size / 1e6:.2f} MB")

t0 = time.time()
simplified = mesh.simplify_quadric_decimation(face_count=TARGET_FACES)
decimation_time = time.time() - t0

# --- Issue #12 fix: recompute UVs from post-decimation vertex XY, reuse the
# native mesh's already-loaded texture. See module docstring for why this is
# exact (not approximate) for this pipeline's grid-position UV convention. ---
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
    texture_preserved = True
else:
    # Defensive fallback -- should not happen for the real native export, but
    # don't silently ship an untextured LOD without at least flagging it loudly.
    print("WARNING: native mesh has no TextureVisuals/baseColorTexture to carry "
          "forward -- LOD will be untextured. This should not happen for a real "
          "large_area26.glb export; investigate before shipping this LOD as a "
          "demo asset.")
    texture_preserved = False

simplified.export(DST)
size_mb = DST.stat().st_size / 1e6
print(f"[LOD] {len(mesh.faces):,} -> {len(simplified.faces):,} faces, "
      f"decimation={decimation_time:.2f}s size={size_mb:.2f}MB "
      f"texture_preserved={texture_preserved}")

record = {
    "source": SRC.name,
    "output": DST.name,
    "source_faces": int(len(mesh.faces)),
    "lod_faces": int(len(simplified.faces)),
    "decimation_time_s": decimation_time,
    "file_size_MB": size_mb,
    "texture_preserved": texture_preserved,
}
out_json = METRICS_DIR / "K3b_lod_generation.json"
out_json.write_text(json.dumps(record, indent=2))
print(f"Saved -> {out_json}")
