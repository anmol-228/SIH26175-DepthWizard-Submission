"""Experiment 3: zero-shot Depth Anything V2 Small on all patches.

3A: raw relative-depth quality (Pearson corr against GT elevation, since raw
    DA2 output has arbitrary scale+shift -- RMSE/MAE are meaningless on raw
    output and are not reported for the raw variant).
3B: oracle linear scale+shift alignment -- fit a*pred+b using TRAIN patches
    only, then apply to VAL/TEST. This is an upper bound on what a *fixed*
    global linear calibration of the frozen zero-shot model could achieve;
    it is explicitly not the final answer (see report), just a probe for
    whether pretrained features carry usable relative-height signal at all.

Saves per-patch raw depth predictions to disk (npz) so later scripts (error
analysis, calibration sweeps) don't need to re-run the network.
"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from metrics import all_metrics, pearson_corr, fit_scale_shift

ROOT = Path(__file__).resolve().parents[1]
PATCH_DIR = ROOT / "data" / "processed" / "patches"
PRED_DIR = ROOT / "outputs" / "predictions" / "zeroshot"
PRED_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR = ROOT / "outputs" / "metrics"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

from da2_utils import preprocess_rgb, load_model, resize_pred_to, MODEL_ID
t0 = time.time()
model = load_model(device).eval()
print(f"Loaded {MODEL_ID} in {time.time()-t0:.1f}s")

manifest = json.load(open(PATCH_DIR / "manifest.json"))

@torch.no_grad()
def infer_patch(rgb_uint8):
    x = preprocess_rgb(rgb_uint8).unsqueeze(0).to(device)
    outputs = model(pixel_values=x)
    pred = outputs.predicted_depth  # (1, h, w) relative "depth" (higher = closer in DA2 convention)
    pred = resize_pred_to(pred, rgb_uint8.shape[:2]).squeeze(0).cpu().numpy()
    return pred

def load_patch(split, entry):
    d = np.load(PATCH_DIR / split / entry["file"])
    return d["rgb"], d["dsm"]

results = {}
for split_name in ["train", "val", "test"]:
    entries = [m for m in manifest if m["split"] == split_name]
    print(f"\n--- {split_name}: {len(entries)} patches ---")
    t0 = time.time()
    for i, entry in enumerate(entries):
        rgb, dsm = load_patch(split_name, entry)
        depth = infer_patch(rgb)
        out_path = PRED_DIR / f"{split_name}_{Path(entry['file']).stem}.npz"
        np.savez_compressed(out_path, depth=depth.astype(np.float32))
        if (i + 1) % 50 == 0 or i == len(entries) - 1:
            print(f"  {i+1}/{len(entries)}  ({time.time()-t0:.1f}s elapsed)")
    print(f"{split_name} zero-shot inference done in {time.time()-t0:.1f}s")

# ---- 3A: raw relative quality (per-patch correlation against raw elevation).
#
# DA2 "depth" convention in a NADIR view: the model's output is disparity-like
# (higher value = closer to the camera), and looking straight down, closer to the
# camera means HIGHER elevation - a rooftop is both nearer and taller. So higher
# DA2 value maps to higher elevation and we correlate against raw elevation with
# NO sign flip, expecting POSITIVE correlation.
#
# CORRECTED 2026-08-29 (issue #15). This comment previously claimed the opposite
# ("LOWER elevation ... expect NEGATIVE corr ... correlate against -elevation").
# That was never what the code did - gather() below appends the raw dsm - and it
# is contradicted by this script's own committed output: zeroshot_3A_raw.json has
# mean_corr +0.731 / +0.694 / +0.802, and zeroshot_3B fits a = +3.898. Only the
# comment changed; no measured value is affected.
#
# Caveat: the nadir argument does not hold for oblique or handheld imagery, where
# the sign genuinely inverts. The relative branch (issue #15) therefore treats the
# sign as an explicit parameter rather than a constant. ----
def gather(split_name):
    entries = [m for m in manifest if m["split"] == split_name]
    preds, gts, tile_ids = [], [], []
    for entry in entries:
        rgb, dsm = load_patch(split_name, entry)
        pred = np.load(PRED_DIR / f"{split_name}_{Path(entry['file']).stem}.npz")["depth"]
        preds.append(pred); gts.append(dsm); tile_ids.append(entry["area"])
    return preds, gts, tile_ids

raw_report = {"per_patch_corr": {}, "summary": {}}
for split_name in ["train", "val", "test"]:
    preds, gts, tile_ids = gather(split_name)
    corrs = [pearson_corr(p, g) for p, g in zip(preds, gts)]
    raw_report["per_patch_corr"][split_name] = corrs
    raw_report["summary"][split_name] = {
        "mean_abs_corr": float(np.nanmean(np.abs(corrs))),
        "mean_corr": float(np.nanmean(corrs)),
        "median_corr": float(np.nanmedian(corrs)),
        "frac_corr_wrong_sign_or_weak (|corr|<0.3)": float(np.mean(np.abs(np.nan_to_num(corrs)) < 0.3)),
        "n_patches": len(corrs),
    }
    print(f"\n[3A raw, {split_name}] mean corr={raw_report['summary'][split_name]['mean_corr']:.3f} "
          f"mean|corr|={raw_report['summary'][split_name]['mean_abs_corr']:.3f}")

with open(METRICS_DIR / "zeroshot_3A_raw.json", "w") as f:
    json.dump(raw_report, f, indent=2)

# ---- 3B: oracle global linear scale+shift, fit on TRAIN only ----
train_preds, train_gts, _ = gather("train")
train_pred_flat = np.concatenate([p.ravel() for p in train_preds])
train_gt_flat = np.concatenate([g.ravel() for g in train_gts])
a, b = fit_scale_shift(train_pred_flat, train_gt_flat)
print(f"\n[3B] fitted global scale/shift on TRAIN: elevation ~= {a:.4f} * raw_depth + {b:.4f}")

oracle_report = {"a": a, "b": b, "fit_split": "train", "per_split": {}}
for split_name in ["val", "test"]:
    preds, gts, tile_ids = gather(split_name)
    all_pred_cal = np.concatenate([(a * p + b).ravel() for p in preds])
    all_gt = np.concatenate([g.ravel() for g in gts])
    m = all_metrics(all_pred_cal, all_gt)
    oracle_report["per_split"][split_name] = m
    print(f"[3B oracle scale/shift, {split_name}] RMSE={m['rmse']:.3f} MAE={m['mae']:.3f} corr={m['corr']:.3f}")

    # per-tile breakdown
    per_tile = {}
    for area in sorted(set(tile_ids)):
        sel = [i for i, a_ in enumerate(tile_ids) if a_ == area]
        pc = np.concatenate([(a * preds[i] + b).ravel() for i in sel])
        gc = np.concatenate([gts[i].ravel() for i in sel])
        per_tile[area] = all_metrics(pc, gc)
    oracle_report[f"per_tile_{split_name}"] = per_tile

with open(METRICS_DIR / "zeroshot_3B_oracle_calibrated.json", "w") as f:
    json.dump(oracle_report, f, indent=2)

print("\nSaved zero-shot results to outputs/metrics/zeroshot_3A_raw.json and zeroshot_3B_oracle_calibrated.json")
