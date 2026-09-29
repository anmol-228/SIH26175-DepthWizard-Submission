"""Experiment C: how much does Phase-1's model rely on the near-infrared (IR)
channel? Vaihingen imagery is IRRG (Infrared-Red-Green) -- it has NO blue
channel at all, so a true "convert to RGB" is not possible without new data
(no blue information was ever captured). Instead this ablates the NIR
channel's CONTRIBUTION on the existing model/data, which directly answers
the master brief's real question: "was Phase-1 performance partly benefiting
from near-infrared information?"

Channel variants tested (each replaces the 9cm test-set patches' first
channel, matching the IRRG channel order used throughout Phase 1: ch0=IR,
ch1=R, ch2=G):
  - irrg (control): unchanged, exactly Phase-1's setup
  - zero_ir: ch0 (IR) zeroed out -- removes NIR signal entirely
  - rg_duplicate: ch0 replaced with a copy of ch2 (G) -- approximates what a
    naive "just drop the NIR band and feed RG twice" pipeline would look like
  - r_only: all 3 channels set to the R channel (grayscale-from-red control,
    a lower bound / sanity check, not a realistic pipeline choice)

Uses the Phase-1 fine-tuned iter2 checkpoint ZERO-SHOT on each variant (no
retraining -- retraining on IR-removed data is a separate, larger question
not attempted here given the time budget) with full-tile oracle calibration.
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
device = "cuda" if torch.cuda.is_available() else "cpu"

manifest = json.load(open(PATCH_DIR / "manifest.json"))
test_entries = [m for m in manifest if m["split"] == "test"]
model = load_model(device)
model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
model.eval()

def load_patch(entry):
    d = np.load(PATCH_DIR / "test" / entry["file"])
    return d["rgb"], d["dsm"]  # "rgb" field is actually IRRG per Phase-1 naming

def apply_variant(irrg, variant):
    out = irrg.copy()
    ir, r, g = out[..., 0], out[..., 1], out[..., 2]
    if variant == "irrg":
        pass
    elif variant == "zero_ir":
        out[..., 0] = 0
    elif variant == "rg_duplicate":
        out[..., 0] = g
    elif variant == "r_only":
        out[..., 0] = r; out[..., 1] = r; out[..., 2] = r
    return out

@torch.no_grad()
def predict(img):
    x = preprocess_rgb(img).unsqueeze(0).to(device)
    out = model(pixel_values=x).predicted_depth
    return resize_pred_to(out, img.shape[:2]).squeeze(0).cpu().numpy()

VARIANTS = ["irrg", "zero_ir", "rg_duplicate", "r_only"]
results = {}
for variant in VARIANTS:
    per_tile_pred, per_tile_gt = {}, {}
    for e in test_entries:
        irrg, dsm = load_patch(e)
        img = apply_variant(irrg, variant)
        pred = predict(img)
        per_tile_pred.setdefault(e["area"], []).append(pred.ravel())
        per_tile_gt.setdefault(e["area"], []).append(dsm.ravel())
    areas = sorted(per_tile_pred.keys())
    all_p, all_g = [], []
    for area in areas:
        p = np.concatenate(per_tile_pred[area]); g = np.concatenate(per_tile_gt[area])
        a, b = fit_scale_shift(p, g)
        all_p.append(a * p + b); all_g.append(g)
    m = all_metrics(np.concatenate(all_p), np.concatenate(all_g))
    results[variant] = m
    print(f"{variant:<15} RMSE={m['rmse']:.3f} MAE={m['mae']:.3f} corr={m['corr']:.3f}")

with open(OUT_DIR / "C_irrg_vs_rgb_ablation.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved -> {OUT_DIR / 'C_irrg_vs_rgb_ablation.json'}")

irrg_rmse = results["irrg"]["rmse"]
for v in VARIANTS[1:]:
    delta = (results[v]["rmse"] - irrg_rmse) / irrg_rmse * 100
    print(f"{v}: {delta:+.1f}% RMSE vs IRRG control")
