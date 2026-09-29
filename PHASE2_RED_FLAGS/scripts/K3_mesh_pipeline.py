"""Sections 35-37: mesh generation + texture draping at small/medium/large
scale, exported as glTF (.glb) for the three.js flythrough prototype
(K4_viewer). Uses REAL Vaihingen DSM+RGB data throughout (not synthetic).
"""
import sys, json, time
from pathlib import Path
P2ROOT = Path(__file__).resolve().parents[1]
POFROOT = P2ROOT.parent
sys.path.insert(0, str(POFROOT / "src"))
import numpy as np
import trimesh
from PIL import Image
from geoio import load_geotiff

OUT_DIR = P2ROOT / "outputs" / "3d"
OUT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR = P2ROOT / "outputs" / "metrics"

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

# Guarded so heightfield_to_mesh() above -- the frozen PIPELINE -> 3D interface
# (fixtures/pipeline_to_3d_v1.json) -- can be imported and called by a downstream
# consumer without this script's whole Vaihingen mesh run executing on import.
# The function itself is unchanged: same signature, same behaviour, no v2 bump.
if __name__ == "__main__":
    TOP_DIR = POFROOT / "data" / "raw" / "Vaihingen" / "top"
    DSM_DIR = POFROOT / "data" / "raw" / "Vaihingen" / "dsm"

    configs = [
        {"name": "small", "area": 26, "crop": (0, 512, 0, 512), "downsample": 1},
        {"name": "medium", "area": 26, "crop": None, "downsample": 2},   # full tile, downsampled 2x
        {"name": "large", "area": 26, "crop": None, "downsample": 1},    # full tile, native res
    ]

    results = []
    for cfg in configs:
        top = load_geotiff(TOP_DIR / f"top_mosaic_09cm_area{cfg['area']}.tif")
        dsm = load_geotiff(DSM_DIR / f"dsm_09cm_matching_area{cfg['area']}.tif")
        rgb = top.array[..., :3].astype(np.uint8)
        elev = dsm.array.astype(np.float32)
        if cfg["crop"]:
            y0, y1, x0, x1 = cfg["crop"]
            rgb = rgb[y0:y1, x0:x1]; elev = elev[y0:y1, x0:x1]

        t0 = time.time()
        mesh = heightfield_to_mesh(elev, rgb, gsd=0.09, z_exaggeration=1.5, downsample=cfg["downsample"])
        gen_time = time.time() - t0

        out_path = OUT_DIR / f"{cfg['name']}_area{cfg['area']}.glb"
        t1 = time.time()
        mesh.export(str(out_path))
        export_time = time.time() - t1
        file_size_mb = out_path.stat().st_size / 1e6

        r = {
            "name": cfg["name"], "area": cfg["area"], "downsample": cfg["downsample"],
            "input_shape": list(elev.shape), "n_vertices": int(mesh.vertices.shape[0]),
            "n_faces": int(mesh.faces.shape[0]), "gen_time_s": gen_time, "export_time_s": export_time,
            "file_size_MB": file_size_mb, "path": str(out_path),
            "is_watertight": bool(mesh.is_watertight), "bounds": mesh.bounds.tolist(),
        }
        results.append(r)
        print(f"[{cfg['name']}] {elev.shape} -> {mesh.vertices.shape[0]:,} verts, {mesh.faces.shape[0]:,} faces, "
              f"gen={gen_time:.2f}s export={export_time:.2f}s size={file_size_mb:.2f}MB watertight={mesh.is_watertight}")

    with open(METRICS_DIR / "K3_mesh_pipeline.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {METRICS_DIR / 'K3_mesh_pipeline.json'}")
    print(f"GLB files -> {OUT_DIR}")
