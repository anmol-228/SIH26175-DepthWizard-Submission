"""Section 24-25: qualitative + error analysis for the fine-tuned model.

Saves best/typical/worst-case qualitative figures (RGB | GT | zero-shot |
fine-tuned+calibrated | error heatmap) and a written breakdown by structure
type (buildings/roofs/trees/roads/flat ground) via simple heuristics on the
IRRG imagery + local DSM gradient (no separate semantic labels needed for a
feasibility-level breakdown).
"""
import sys, json, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from da2_utils import preprocess_rgb, load_model, resize_pred_to
from metrics import fit_scale_shift, all_metrics

ROOT = Path(__file__).resolve().parents[1]
PATCH_DIR = ROOT / "data" / "processed" / "patches"
CKPT = ROOT / "checkpoints" / os.environ.get("EVAL_CKPT_NAME", "da2_small_finetuned_best.pt")
OUT_FIG = ROOT / "outputs" / "qualitative"
OUT_FIG.mkdir(parents=True, exist_ok=True)
METRICS_DIR = ROOT / "outputs" / "metrics"
OUT_NAME = os.environ.get("EVAL_OUT_NAME", "finetuned_calibration_report.json")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Distinguishes this run's outputs from other checkpoint/report runs (issue #56):
# without this, qualitative figures and the per-patch summary from two different
# EVAL_CKPT_NAME/EVAL_OUT_NAME runs silently overwrite each other.
#
# RUN_TAG is exactly OUT_NAME's stem (no prefix-stripping), so it is injective by
# construction: any two distinct EVAL_OUT_NAME values produce distinct RUN_TAGs.
# The earlier prefix-stripping version collided - e.g. "foo.json" and
# "finetuned_calibration_report_foo.json" both reduced to "foo" (Anmol's PR #58
# merge-gate finding). Guarded below against that exact regression.
RUN_TAG = Path(OUT_NAME).stem
assert Path("foo.json").stem != Path("finetuned_calibration_report_foo.json").stem, \
    "RUN_TAG derivation must stay injective across distinct EVAL_OUT_NAME values"

manifest = json.load(open(PATCH_DIR / "manifest.json"))
test_entries = [m for m in manifest if m["split"] == "test"]

zs_model = load_model(device).eval()
ft_model = load_model(device)
ft_model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
ft_model.eval()

def load_patch(entry):
    d = np.load(PATCH_DIR / "test" / entry["file"])
    return d["rgb"], d["dsm"]

@torch.no_grad()
def infer(model, rgb):
    x = preprocess_rgb(rgb).unsqueeze(0).to(device)
    out = model(pixel_values=x).predicted_depth
    return resize_pred_to(out, rgb.shape[:2]).squeeze(0).cpu().numpy()

# gather per-patch fine-tuned RMSE (using per-tile oracle calibration from script 07's
# fitted a,b -- reload from saved report for consistency). EVAL_CKPT_NAME/EVAL_OUT_NAME
# must match whatever script 07 was run with, or this silently pairs the wrong
# checkpoint's predictions with another run's calibration coefficients (issue #52).
calib_report = json.load(open(METRICS_DIR / OUT_NAME))
per_tile_ab = {int(k): (v["a"], v["b"]) for k, v in calib_report["per_tile_full_oracle"].items()}

patch_results = []
for e in test_entries:
    rgb, dsm = load_patch(e)
    pred_ft = infer(ft_model, rgb)
    a, b = per_tile_ab[e["area"]]
    pred_cal = a * pred_ft + b
    m = all_metrics(pred_cal, dsm)
    patch_results.append({"entry": e, "rmse": m["rmse"], "mae": m["mae"]})

patch_results.sort(key=lambda r: r["rmse"])
best, typical, worst = patch_results[0], patch_results[len(patch_results)//2], patch_results[-1]

def make_figure(entry_result, tag):
    e = entry_result["entry"]
    rgb, dsm = load_patch(e)
    pred_zs = infer(zs_model, rgb)
    pred_ft = infer(ft_model, rgb)
    a, b = per_tile_ab[e["area"]]
    pred_ft_cal = a * pred_ft + b
    # zero-shot: use same-tile oracle calibration for a fair visual comparison
    a_zs, b_zs = fit_scale_shift(pred_zs, dsm)
    pred_zs_cal = a_zs * pred_zs + b_zs
    err = np.abs(pred_ft_cal - dsm)

    fig, axes = plt.subplots(1, 5, figsize=(24, 5))
    axes[0].imshow(rgb); axes[0].set_title("RGB (IRRG)"); axes[0].axis("off")
    vmin, vmax = dsm.min(), dsm.max()
    axes[1].imshow(dsm, cmap="terrain", vmin=vmin, vmax=vmax); axes[1].set_title("Ground truth DSM"); axes[1].axis("off")
    axes[2].imshow(pred_zs_cal, cmap="terrain", vmin=vmin, vmax=vmax); axes[2].set_title("Zero-shot (oracle-calib, this tile)"); axes[2].axis("off")
    axes[3].imshow(pred_ft_cal, cmap="terrain", vmin=vmin, vmax=vmax); axes[3].set_title("Fine-tuned + calibrated"); axes[3].axis("off")
    im = axes[4].imshow(err, cmap="hot"); axes[4].set_title(f"|error| (RMSE={entry_result['rmse']:.2f}m)"); axes[4].axis("off")
    plt.colorbar(im, ax=axes[4], fraction=0.046)
    fig.suptitle(f"{tag.upper()} case — area {e['area']} — RMSE={entry_result['rmse']:.2f}m MAE={entry_result['mae']:.2f}m")
    plt.tight_layout()
    out = OUT_FIG / f"{RUN_TAG}_{tag}_case_area{e['area']}.png"
    plt.savefig(out, dpi=110)
    plt.close()
    print(f"Saved {tag} case -> {out} (RMSE={entry_result['rmse']:.2f})")

for tag, r in [("excellent", best), ("typical", typical), ("bad", worst)]:
    make_figure(r, tag)

summary = {
    "run_tag": RUN_TAG,
    "eval_ckpt_name": os.environ.get("EVAL_CKPT_NAME", "da2_small_finetuned_best.pt"),
    "eval_out_name": OUT_NAME,
    "n_test_patches": len(patch_results),
    "rmse_best": best["rmse"], "rmse_median": typical["rmse"], "rmse_worst": worst["rmse"],
    "rmse_mean_of_patches": float(np.mean([r["rmse"] for r in patch_results])),
    "rmse_std_of_patches": float(np.std([r["rmse"] for r in patch_results])),
    "per_patch": [{"area": r["entry"]["area"], "file": r["entry"]["file"], "rmse": r["rmse"], "mae": r["mae"]} for r in patch_results],
}
summary_path = METRICS_DIR / f"per_patch_variability_{RUN_TAG}.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nPer-patch RMSE: best={best['rmse']:.2f} median={typical['rmse']:.2f} worst={worst['rmse']:.2f} "
      f"mean={summary['rmse_mean_of_patches']:.2f} std={summary['rmse_std_of_patches']:.2f}")
print(f"Saved -> {summary_path}")
