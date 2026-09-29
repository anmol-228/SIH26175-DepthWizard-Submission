"""Experiment E, step 2: cut the co-registered 1m AOI (data/hilly_terrain/aoi)
into non-overlapping 512x512 patches, compute per-pixel slope (simple
finite-difference gradient on the 1m DEM, in degrees), and assign a
train/val/test split for the OPTIONAL adaptation fine-tune experiment
(E_04). The primary zero-shot / Phase-1-checkpoint evaluation (E_03) uses
ALL patches, since neither model has ever seen this data -- no leakage
there. The split is only needed so the adaptation run has a genuinely
held-out test set.
"""
import json
from pathlib import Path
import numpy as np

P2ROOT = Path(__file__).resolve().parents[1]
AOI_DIR = P2ROOT / "data" / "hilly_terrain" / "aoi"
PATCH_DIR = P2ROOT / "data" / "hilly_terrain" / "patches"
PATCH_DIR.mkdir(parents=True, exist_ok=True)

PATCH = 512
VALID_FRAC_THRESH = 0.98
SEED = 42

if __name__ == "__main__":
    dem = np.load(AOI_DIR / "dem_1m.npy")
    valid = np.load(AOI_DIR / "dem_valid_mask.npy")
    rgb = np.load(AOI_DIR / "rgb_1m.npy")
    geo = json.load(open(AOI_DIR / "geo_meta.json"))
    H, W = dem.shape
    print(f"AOI shape: {H}x{W}, valid frac {valid.mean():.4f}")

    # finite-difference slope (degrees) on the FULL AOI grid, 1m pixel spacing
    gy, gx = np.gradient(dem, 1.0, 1.0)  # dz/drow (south+), dz/dcol (east+)
    slope_deg = np.degrees(np.arctan(np.sqrt(gx ** 2 + gy ** 2)))
    # gradient at nodata pixels is meaningless; zero it out there (patches are
    # filtered to >=98% valid anyway, so this affects at most a thin border)
    slope_deg = np.where(valid, slope_deg, 0.0)
    print(f"Slope stats (valid px): mean={slope_deg[valid].mean():.2f} deg, "
          f"p50={np.percentile(slope_deg[valid], 50):.2f}, p90={np.percentile(slope_deg[valid], 90):.2f}, "
          f"max={slope_deg[valid].max():.2f}")

    n_rows, n_cols = H // PATCH, W // PATCH
    patches = []
    for pr in range(n_rows):
        for pc in range(n_cols):
            r0, c0 = pr * PATCH, pc * PATCH
            v = valid[r0:r0 + PATCH, c0:c0 + PATCH]
            if v.mean() < VALID_FRAC_THRESH:
                continue
            pid = f"p{pr:02d}_{pc:02d}"
            d = dem[r0:r0 + PATCH, c0:c0 + PATCH].astype(np.float32)
            rgb_p = rgb[r0:r0 + PATCH, c0:c0 + PATCH, :].copy()
            slope_p = slope_deg[r0:r0 + PATCH, c0:c0 + PATCH].astype(np.float32)
            np.savez_compressed(PATCH_DIR / f"{pid}.npz", rgb=rgb_p, dsm=d, slope=slope_p)
            patches.append({
                "id": pid, "row0": r0, "col0": c0, "file": f"{pid}.npz",
                "valid_frac": float(v.mean()),
                "elev_min": float(d.min()), "elev_max": float(d.max()),
                "relief_m": float(d.max() - d.min()),
                "slope_mean_deg": float(slope_p.mean()), "slope_max_deg": float(slope_p.max()),
            })
    print(f"Kept {len(patches)} / {n_rows*n_cols} candidate {PATCH}x{PATCH} patches (>= {VALID_FRAC_THRESH*100:.0f}% valid)")

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(patches))
    n = len(patches)
    n_train = int(round(n * 0.6))
    n_val = int(round(n * 0.2))
    split_of = {}
    for i, idx in enumerate(order):
        if i < n_train:
            split_of[idx] = "adapt_train"
        elif i < n_train + n_val:
            split_of[idx] = "adapt_val"
        else:
            split_of[idx] = "adapt_test"
    for i, p in enumerate(patches):
        p["split"] = split_of[i]

    with open(PATCH_DIR / "manifest.json", "w") as f:
        json.dump(patches, f, indent=2)

    counts = {}
    for p in patches:
        counts[p["split"]] = counts.get(p["split"], 0) + 1
    print("Split counts:", counts)
    print(f"Manifest -> {PATCH_DIR / 'manifest.json'}")

    # save AOI-level slope map for the figure/report
    np.save(AOI_DIR / "slope_deg.npy", slope_deg)
