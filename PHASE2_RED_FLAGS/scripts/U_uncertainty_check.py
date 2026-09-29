"""Section 39: Phase 1 never implemented an uncertainty/confidence output at
all (there is nothing to "validate" directly). This builds the cheapest
defensible proxy -- test-time-augmentation (TTA) prediction variance -- and
checks whether it actually correlates with real error, on the real 9cm
Vaihingen test set. If it doesn't correlate, the master brief is explicit:
do not ship a fake confidence heatmap.
"""
import sys, json
from pathlib import Path
P2ROOT = Path(__file__).resolve().parents[1]
POFROOT = P2ROOT.parent
sys.path.insert(0, str(POFROOT / "src"))
import numpy as np
import torch
from da2_utils import preprocess_rgb, load_model, resize_pred_to
from metrics import fit_scale_shift, pearson_corr

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
    return d["rgb"], d["dsm"]

@torch.no_grad()
def predict(img):
    x = preprocess_rgb(img).unsqueeze(0).to(device)
    out = model(pixel_values=x).predicted_depth
    return resize_pred_to(out, img.shape[:2]).squeeze(0).cpu().numpy()

def tta_variance(rgb):
    """4 augmented views: identity, h-flip, +15% brightness, -15% brightness."""
    preds = []
    preds.append(predict(rgb))
    preds.append(np.fliplr(predict(np.fliplr(rgb).copy())))
    bright = np.clip(rgb.astype(np.float32) * 1.15, 0, 255).astype(np.uint8)
    preds.append(predict(bright))
    dark = np.clip(rgb.astype(np.float32) * 0.85, 0, 255).astype(np.uint8)
    preds.append(predict(dark))
    stack = np.stack(preds, axis=0)
    return stack.mean(axis=0), stack.std(axis=0)

all_std, all_abserr = [], []
per_patch_corr = []
for e in test_entries:
    rgb, dsm = load_patch(e)
    mean_pred, std_pred = tta_variance(rgb)
    a, b = fit_scale_shift(mean_pred, dsm)  # oracle calib for this diagnostic only
    cal = a * mean_pred + b
    abserr = np.abs(cal - dsm)
    all_std.append(std_pred.ravel())
    all_abserr.append(abserr.ravel())
    if std_pred.std() > 1e-6:
        per_patch_corr.append(pearson_corr(std_pred, abserr))

all_std = np.concatenate(all_std); all_abserr = np.concatenate(all_abserr)
global_corr = pearson_corr(all_std, all_abserr)

# also: does high-std TOP 10% of pixels have higher error than bottom 10%?
thresh_hi = np.percentile(all_std, 90)
thresh_lo = np.percentile(all_std, 10)
hi_err = all_abserr[all_std >= thresh_hi].mean()
lo_err = all_abserr[all_std <= thresh_lo].mean()

result = {
    "method": "TTA variance (4 views: identity, h-flip, +/-15% brightness)",
    "global_pixelwise_corr_std_vs_abserr": global_corr,
    "mean_per_patch_corr": float(np.mean(per_patch_corr)),
    "median_per_patch_corr": float(np.median(per_patch_corr)),
    "top10pct_std_mean_abserr": float(hi_err),
    "bottom10pct_std_mean_abserr": float(lo_err),
    "ratio_top_vs_bottom": float(hi_err / lo_err),
    "n_patches": len(test_entries),
}
print(json.dumps(result, indent=2))
with open(OUT_DIR / "U_uncertainty_check.json", "w") as f:
    json.dump(result, f, indent=2)
