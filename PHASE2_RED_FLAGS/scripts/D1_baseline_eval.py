"""Experiment D, step 1: baseline eval on genuine dense forest (Sentinel-2 RGB
+ ETH Global Canopy Height), both zero-shot pretrained DA2-Small and the
Vaihingen-fine-tuned checkpoint, with full-tile(=full-patch) oracle
scale/shift calibration -- matching Phase-1's 07_calibrate_and_eval.py
methodology exactly (fit_scale_shift + all_metrics), applied here PER TEST
PATCH since our forest scene has no natural multi-patch "tile" unit the way
Vaihingen's named areas did (each of our 512x512 patches covers ~5.12km x
5.12km at 10m GSD, comparable in scale to a whole Vaihingen "area", not to a
Vaihingen "patch") -- this mapping is disclosed in 06_VEGETATION_FOREST_TEST.md.

Only VALID pixels (CHM != nodata AND SCL not cloud/shadow/defect) are scored.
"""
import sys, json, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np
import torch
from da2_utils import preprocess_rgb, load_model, resize_pred_to, MODEL_ID
from metrics import all_metrics, fit_scale_shift

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
PATCH_DIR = ROOT / "data" / "forest_eth_s2" / "patches"
METRICS_DIR = ROOT / "outputs" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
manifest = json.load(open(PATCH_DIR / "manifest.json"))
test_entries = [m for m in manifest if m["split"] == "test"]
print(f"Test patches: {len(test_entries)} -> {[m['file'] for m in test_entries]}")


def load_patch(entry):
    d = np.load(PATCH_DIR / entry["file"])
    return d["rgb"], d["chm"], d["valid"]


@torch.no_grad()
def predict(model, rgb):
    x = preprocess_rgb(rgb).unsqueeze(0).to(device)
    out = model(pixel_values=x).predicted_depth
    return resize_pred_to(out, rgb.shape[:2]).squeeze(0).cpu().numpy()


def run_variant(model, tag):
    per_patch = {}
    all_pred_cal, all_gt = [], []
    for e in test_entries:
        rgb, chm, valid = load_patch(e)
        pred = predict(model, rgb)
        p = pred[valid]
        g = chm[valid]
        a, b = fit_scale_shift(p, g)
        p_cal = a * p + b
        m = all_metrics(p_cal, g)
        # forest-only submask (canopy present, >2m) -- almost all of AOI, kept for parity with red-flag #1 wording
        forest_valid = valid & (chm > 2.0)
        if forest_valid.sum() > 100:
            pf = pred[forest_valid]
            gf = chm[forest_valid]
            pf_cal = a * pf + b  # same fit (whole-patch oracle), scored on forest subset
            mf = all_metrics(pf_cal, gf)
        else:
            mf = None
        per_patch[e["file"]] = {
            **m, "a": a, "b": b, "n_px": int(p.size),
            "frac_forest_gt2m": e["frac_forest_gt2m"],
            "forest_submask_metrics": mf,
        }
        all_pred_cal.append(p_cal)
        all_gt.append(g)
        print(f"  [{tag}] {e['file']}: RMSE={m['rmse']:.3f} MAE={m['mae']:.3f} corr={m['corr']:.3f} "
              f"(a={a:.3f} b={b:.3f}, forest_frac={e['frac_forest_gt2m']:.2f})")
    all_pred_cal = np.concatenate(all_pred_cal)
    all_gt = np.concatenate(all_gt)
    global_m = all_metrics(all_pred_cal, all_gt)
    print(f"  [{tag}] GLOBAL POOLED: RMSE={global_m['rmse']:.3f} MAE={global_m['mae']:.3f} corr={global_m['corr']:.3f}")
    return {"per_patch_full_oracle": per_patch, "global_pooled": global_m}


results = {}

print("\n=== Variant 1: zero-shot PRETRAINED DA2-Small (never seen Vaihingen or forest) ===")
model_zs = load_model(device)
model_zs.eval()
results["zero_shot_pretrained"] = run_variant(model_zs, "zero-shot")
del model_zs
torch.cuda.empty_cache()

print("\n=== Variant 2: Vaihingen-fine-tuned checkpoint (cross-domain + vegetation combined) ===")
ckpt_path = REPO_ROOT / "checkpoints" / "da2_small_finetuned_iter2_partial_unfreeze.pt"
model_ft = load_model(device)
model_ft.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
model_ft.eval()
print(f"Loaded {ckpt_path}")
results["vaihingen_finetuned_zero_shot_transfer"] = run_variant(model_ft, "vaihingen-ft")
del model_ft
torch.cuda.empty_cache()

with open(METRICS_DIR / "D1_baseline_forest_eval.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved -> outputs/metrics/D1_baseline_forest_eval.json")

print("\n=== Summary ===")
for k, v in results.items():
    g = v["global_pooled"]
    print(f"{k}: RMSE={g['rmse']:.2f}m MAE={g['mae']:.2f}m corr={g['corr']:.3f}")
