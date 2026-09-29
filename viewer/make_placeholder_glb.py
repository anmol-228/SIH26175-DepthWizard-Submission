#!/usr/bin/env python3
"""
DW-S1-08/09 placeholder generator.

Produces viewer/placeholder_relative.glb -- a tiny synthetic heightfield mesh built
with the REAL heightfield_to_mesh() from PHASE2_RED_FLAGS/scripts/heightfield_lib.py
(the frozen pipeline_to_3d_v1 producing function), not a hand-authored/fake .glb.
Only the elev/rgb input is synthetic (a small procedural bump), so the viewer
scaffold can be built and tested before DW-S1-07's real pipeline output exists.
Swap for a real run_pipeline.py output once DW-S1-07 (#24) lands -- no viewer
code changes needed, since it's the same producing function and output shape.

DW-WP2 (issue #26, IF-1 frozen 2026-09-06): the sidecar this script writes now
follows the same viewer_input_v1 schema run_pipeline.py itself emits (schema,
branch, dtype/byte_order, raw little-endian float32 elevation payload, explicit
mesh-frame convention, is_placeholder) -- see run_pipeline.write_surface_meta()
for the authoritative field-by-field description. This keeps main.js's sidecar-
consuming code a single path for both placeholder and real pipeline output,
rather than two schemas to special-case. is_placeholder is explicitly True here
so the viewer can flag it, per IF-1's "no silent placeholder substitution" rule.

Filenames follow the same stem convention as a real run
(`{stem}.glb` / `{stem}_lod.glb` / `{stem}_surface_meta.json` / `{stem}_elevation.f32`,
stem = "placeholder_relative") so main.js's glb-URL-derived sidecar resolution needs no
placeholder-specific special case either.

DW-P2 (issue #50): generation is wrapped in generate_placeholder(out_dir=...) so a
caller -- specifically tests/test_placeholder_scale.py -- can point the REAL,
unmodified generator at a throwaway directory instead of always overwriting the
tracked viewer/ fixture on every test run. Normal manual invocation
(`python3 viewer/make_placeholder_glb.py`, no args) still defaults to and writes
viewer/ exactly as before -- this is purely a "where do the real outputs land"
parameter, no change to what gets generated.

Run: python3 viewer/make_placeholder_glb.py [--out-dir DIR]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "PHASE2_RED_FLAGS", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from heightfield_lib import heightfield_to_mesh          # noqa: E402
from run_pipeline import PSEUDO_GSD, write_surface_meta  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE = "placeholder"
BRANCH = "relative"
# The mesh filename must share its stem with its sidecar (see viewer/sidecar.js /
# write_surface_meta()'s _elevation_stem convention: "{scene}_{branch}") - a mismatch
# here is exactly the "orphan mesh, sidecar never found" bug this naming rule exists to
# prevent. Real pipeline output already follows this; the placeholder must too.
STEM = f"{SCENE}_{BRANCH}"

H, W = 64, 64


def generate_placeholder(out_dir: str | os.PathLike = HERE) -> dict[str, Path]:
    """Build the synthetic placeholder mesh + viewer_input_v1 sidecar with the REAL
    heightfield_to_mesh()/write_surface_meta() producing functions, writing into
    out_dir (default: this script's own directory, viewer/, for normal manual
    invocation). Pass a different out_dir (e.g. a pytest tmp_path) to exercise the
    real generator without touching the tracked viewer/placeholder_relative.glb
    fixture -- see issue #50.

    Returns the paths written, keyed "glb"/"lod_glb"/"surface_meta"/"elevation_f32",
    plus "mesh" (the in-memory trimesh object), so callers don't need to re-derive
    the naming convention themselves.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    glb_path = out_dir / f"{STEM}.glb"

    # BUG FIX (caught by Anmol's PR #44 review, 2026-09-08): the mesh was previously
    # built with gsd=0.09 while the sidecar below declared pseudo_gsd=1.0 - a real
    # scale mismatch between the actual mesh geometry (x=col*gsd, y=-row*gsd) and what
    # the viewer's worldToGrid() used to invert a raycast hit back to (row, col).
    # Every picked point on the placeholder was silently off by a factor of
    # 1/0.09 ~= 11.1x. The mesh's gsd and the sidecar's pseudo_gsd must always be the
    # SAME number - use PSEUDO_GSD for both, exactly as run_pipeline.py's real
    # relative branch does (to_mesh(norm, rgb, PSEUDO_GSD, ...) paired with
    # write_surface_meta(..., pseudo_gsd=PSEUDO_GSD, ...)).
    gsd = PSEUDO_GSD
    z_exaggeration = 1.5

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    elev = (
        2.0 * np.sin(xx / 6.0) * np.cos(yy / 6.0)
        + 0.03 * (xx + yy)
    ).astype(np.float32)

    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[..., 0] = (128 + 100 * np.sin(xx / 5.0)).clip(0, 255)
    rgb[..., 1] = (128 + 100 * np.cos(yy / 5.0)).clip(0, 255)
    rgb[..., 2] = 90

    mesh = heightfield_to_mesh(elev, rgb, gsd=gsd, z_exaggeration=z_exaggeration, downsample=1)
    mesh.export(str(glb_path))

    # No real LOD ladder for a 64x64 placeholder (nothing to decimate meaningfully) -
    # point lod_glb at the same native mesh rather than fabricating a second file. A
    # real run's to_lod() always produces a genuine (possibly identical, per
    # make_lod()'s <=target passthrough) LOD file; this placeholder just has nothing
    # to decimate.
    surface_meta_path, elevation_f32_path = write_surface_meta(
        out_dir=out_dir, scene=SCENE, branch=BRANCH, elev=elev,
        mesh_glb=glb_path, lod_glb=glb_path,
        gsd=None, pseudo_gsd=PSEUDO_GSD, z_exaggeration=z_exaggeration,
        crs=None, normalization=None, nodata=None,
        is_placeholder=True,
        note=("synthetic placeholder surface, no calibration -- never label this metres; "
              "is_placeholder=true, per IF-1 no silent placeholder substitution"),
    )

    size_kb = os.path.getsize(glb_path) / 1024
    print(f"wrote {glb_path} ({size_kb:.1f} KB), {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    print(f"wrote {surface_meta_path}, {elevation_f32_path}")

    return {
        "glb": glb_path, "lod_glb": glb_path,
        "surface_meta": surface_meta_path, "elevation_f32": elevation_f32_path,
        "mesh": mesh,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", default=HERE,
        help="output directory for the generated placeholder assets (default: viewer/, "
             "this script's own directory -- normal manual invocation is unchanged)")
    args = parser.parse_args()
    generate_placeholder(args.out_dir)
