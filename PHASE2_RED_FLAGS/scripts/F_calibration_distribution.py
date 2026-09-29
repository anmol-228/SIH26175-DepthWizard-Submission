"""Experiment F (Section 26): does calibration-POINT DISTRIBUTION matter as
much as point COUNT? Compares uniform-random / clustered / edge-biased /
terrain-stratified sampling strategies at fixed N (10, 20), each averaged
over 15 random seeds to separate real distribution effects from single-draw
sampling noise (Phase-1 sparse-calibration numbers were single-seed and
flagged as noisy; this experiment explicitly quantifies that variance).

Uses the Phase-1 fine-tuned iter2 model's predictions on the native 9cm
TEST set (no new dataset needed for this specific question).
"""
import sys, json
from pathlib import Path
P2ROOT = Path(__file__).resolve().parents[1]
POFROOT = P2ROOT.parent
sys.path.insert(0, str(POFROOT / "src"))
import numpy as np
import torch
from da2_utils import preprocess_rgb, load_model, resize_pred_to
from metrics import all_metrics, fit_scale_shift

PATCH_DIR = POFROOT / "data" / "processed" / "patches"
CKPT = POFROOT / "checkpoints" / "da2_small_finetuned_iter2_partial_unfreeze.pt"
OUT_DIR = P2ROOT / "outputs" / "metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

manifest = json.load(open(PATCH_DIR / "manifest.json"))
test_entries = [m for m in manifest if m["split"] == "test"]

model = load_model(device)
model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
model.eval()

def load_patch(entry):
    d = np.load(PATCH_DIR / "test" / entry["file"])
    return d["rgb"], d["dsm"]

@torch.no_grad()
def predict(rgb):
    x = preprocess_rgb(rgb).unsqueeze(0).to(device)
    out = model(pixel_values=x).predicted_depth
    return resize_pred_to(out, rgb.shape[:2]).squeeze(0).cpu().numpy()

# ---- gather per-tile flattened pred/gt + 2D shape info for spatial strategies ----
per_tile_pred, per_tile_gt, per_tile_shape = {}, {}, {}
for e in test_entries:
    rgb, dsm = load_patch(e)
    pred = predict(rgb)
    per_tile_pred.setdefault(e["area"], []).append(pred.ravel())
    per_tile_gt.setdefault(e["area"], []).append(dsm.ravel())
    per_tile_shape.setdefault(e["area"], []).append(dsm.shape)
areas = sorted(per_tile_pred.keys())
for area in areas:
    per_tile_pred[area] = np.concatenate(per_tile_pred[area])
    per_tile_gt[area] = np.concatenate(per_tile_gt[area])

def select_uniform(n, size, rng, g=None):
    return rng.choice(size, size=n, replace=False)

def select_clustered(n, size, rng, g=None):
    # pick one random center index, take the n spatially-nearest indices in FLAT
    # index space as a cheap proxy for "all points fall in one small area"
    center = rng.integers(0, size)
    lo = max(0, center - n // 2)
    hi = min(size, lo + n)
    lo = max(0, hi - n)
    return np.arange(lo, hi)

def select_edge_biased(n, size, rng, g):
    # g = local gradient magnitude of gt (edge strength); bias toward top-gradient pixels
    top_idx = np.argsort(g)[-max(n * 20, n):]  # top candidates by edge strength
    chosen = rng.choice(top_idx, size=min(n, len(top_idx)), replace=False)
    return chosen

def select_terrain_stratified(n, size, rng, g_ignore=None, gt=None):
    # stratify by height decile, sample proportionally
    deciles = np.quantile(gt, np.linspace(0, 1, 11))
    bins = np.digitize(gt, deciles[1:-1])
    idx = []
    per_bin = max(1, n // 10)
    for b in range(10):
        cand = np.where(bins == b)[0]
        if len(cand) == 0:
            continue
        take = min(per_bin, len(cand))
        idx.extend(rng.choice(cand, size=take, replace=False))
    idx = np.array(idx[:n]) if len(idx) >= n else np.array(idx)
    if len(idx) < n:
        remaining = np.setdiff1d(np.arange(size), idx)
        extra = rng.choice(remaining, size=n - len(idx), replace=False)
        idx = np.concatenate([idx, extra])
    return idx

def edge_strength_per_tile(area):
    # approximate 2D gradient magnitude per patch, then flatten+concat in same order
    grads = []
    for e in [e for e in test_entries if e["area"] == area]:
        _, dsm = load_patch(e)
        gy, gx = np.gradient(dsm.astype(np.float64))
        grads.append(np.sqrt(gx**2 + gy**2).ravel())
    return np.concatenate(grads)

STRATEGIES = {
    "uniform": select_uniform,
    "clustered": select_clustered,
    "edge_biased": select_edge_biased,
    "terrain_stratified": select_terrain_stratified,
}

N_SEEDS = 15
results = {}
for n_points in [10, 20]:
    results[str(n_points)] = {}
    for strat_name, strat_fn in STRATEGIES.items():
        rmses = []
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(1000 + seed)
            all_pred_cal, all_gt_cal = [], []
            for area in areas:
                p = per_tile_pred[area]; g = per_tile_gt[area]
                size = p.size
                n = min(n_points, size)
                if strat_name == "edge_biased":
                    edge_g = edge_strength_per_tile(area)
                    idx = strat_fn(n, size, rng, edge_g)
                elif strat_name == "terrain_stratified":
                    idx = strat_fn(n, size, rng, gt=g)
                else:
                    idx = strat_fn(n, size, rng)
                if len(idx) < 2:
                    a, b = 1.0, float(g[idx[0]] - p[idx[0]]) if len(idx) else (1.0, 0.0)
                else:
                    a, b = fit_scale_shift(p[idx], g[idx])
                p_cal = a * p + b
                all_pred_cal.append(p_cal); all_gt_cal.append(g)
            m = all_metrics(np.concatenate(all_pred_cal), np.concatenate(all_gt_cal))
            rmses.append(m["rmse"])
        results[str(n_points)][strat_name] = {
            "mean_rmse": float(np.mean(rmses)), "std_rmse": float(np.std(rmses)),
            "min_rmse": float(np.min(rmses)), "max_rmse": float(np.max(rmses)),
        }
        print(f"N={n_points:>2} strategy={strat_name:<20} mean_RMSE={np.mean(rmses):.3f} "
              f"std={np.std(rmses):.3f} (range {np.min(rmses):.3f}-{np.max(rmses):.3f})")

with open(OUT_DIR / "F_calibration_distribution.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved -> {OUT_DIR / 'F_calibration_distribution.json'}")
