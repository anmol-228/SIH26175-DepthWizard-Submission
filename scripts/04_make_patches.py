"""Materialize fixed-size RGB+DSM patches per split to .npy for fast reuse
across zero-shot / fine-tune / calibration experiments. Patches never cross
tile (area) boundaries, and train/val/test patches only come from their own
tiles per data/splits/split_v1.json -- no possibility of spatial leakage
across the split boundary even though patches within a tile can overlap
(overlap is only allowed within TRAIN, for more training signal; val/test use
a non-overlapping grid so evaluation pixels are counted once).
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from geoio import load_geotiff, dsm_nodata_mask

ROOT = Path(__file__).resolve().parents[1]
TOP_DIR = ROOT / "data" / "raw" / "Vaihingen" / "top"
DSM_DIR = ROOT / "data" / "raw" / "Vaihingen" / "dsm"
OUT_DIR = ROOT / "data" / "processed" / "patches"
OUT_DIR.mkdir(parents=True, exist_ok=True)

split = json.load(open(ROOT / "data" / "splits" / "split_v1.json"))
PATCH = 512

def extract_patches(area, stride, min_valid_frac=0.98):
    top = load_geotiff(TOP_DIR / f"top_mosaic_09cm_area{area}.tif")
    dsm = load_geotiff(DSM_DIR / f"dsm_09cm_matching_area{area}.tif")
    rgb = top.array[..., :3].astype(np.uint8)
    elev = dsm.array.astype(np.float32)
    mask = dsm_nodata_mask(elev)
    h, w = elev.shape
    out = []
    for y in range(0, h - PATCH + 1, stride):
        for x in range(0, w - PATCH + 1, stride):
            m = mask[y:y+PATCH, x:x+PATCH]
            if m.mean() < min_valid_frac:
                continue
            out.append({
                "rgb": rgb[y:y+PATCH, x:x+PATCH].copy(),
                "dsm": elev[y:y+PATCH, x:x+PATCH].copy(),
                "area": area, "y": y, "x": x,
            })
    return out

manifest = []
for split_name, areas, stride in [("train", split["train_areas"], 384),
                                    ("val", split["val_areas"], 512),
                                    ("test", split["test_areas"], 512)]:
    split_dir = OUT_DIR / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    idx = 0
    for area in areas:
        patches = extract_patches(area, stride)
        for p in patches:
            fname = f"{split_name}_{idx:05d}_area{area}_y{p['y']}_x{p['x']}.npz"
            np.savez_compressed(split_dir / fname, rgb=p["rgb"], dsm=p["dsm"])
            manifest.append({"split": split_name, "file": fname, "area": area, "y": p["y"], "x": p["x"]})
            idx += 1
    print(f"{split_name}: {idx} patches from {len(areas)} tiles (stride={stride})")

with open(OUT_DIR / "manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
print(f"\nTotal patches: {len(manifest)}. Saved to {OUT_DIR}")
