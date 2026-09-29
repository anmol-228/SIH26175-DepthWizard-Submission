"""Section 32: minimal 3D mesh sanity check -- prediction -> DSM -> mesh.

Deliberately minimal: proves the pipeline prediction->height-field->navigable
3D surface works end to end. NOT the full flythrough viewer (out of scope for
this gate per Section 53/33). Uses matplotlib's 3D surface plot (no extra
heavy 3D dependency needed) on ONE calibrated test tile.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from da2_utils import preprocess_rgb, load_model, resize_pred_to
from metrics import fit_scale_shift

ROOT = Path(__file__).resolve().parents[1]
PATCH_DIR = ROOT / "data" / "processed" / "patches"
CKPT = ROOT / "checkpoints" / "da2_small_finetuned_best.pt"
OUT_FIG = ROOT / "outputs" / "figures"
device = "cuda" if torch.cuda.is_available() else "cpu"

manifest = json.load(open(PATCH_DIR / "manifest.json"))
test_entries = [m for m in manifest if m["split"] == "test"]
target_area = test_entries[0]["area"]
tile_entries = [e for e in test_entries if e["area"] == target_area]

model = load_model(device)
model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
model.eval()

e = tile_entries[len(tile_entries) // 2]
d = np.load(PATCH_DIR / "test" / e["file"])
rgb, dsm = d["rgb"], d["dsm"]

with torch.no_grad():
    x = preprocess_rgb(rgb).unsqueeze(0).to(device)
    pred = resize_pred_to(model(pixel_values=x).predicted_depth, rgb.shape[:2]).squeeze(0).cpu().numpy()

a, b = fit_scale_shift(pred, dsm)  # oracle-calibrated for this visualization only (labeled as such)
pred_cal = a * pred + b

# downsample for a renderable mesh
step = 4
Z = pred_cal[::step, ::step]
Zgt = dsm[::step, ::step]
X, Y = np.meshgrid(np.arange(Z.shape[1]), np.arange(Z.shape[0]))
rgb_ds = rgb[::step, ::step].astype(np.float32) / 255.0

fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.plot_surface(X, Y, Zgt, facecolors=rgb_ds, rstride=1, cstride=1, shade=False)
ax1.set_title(f"Ground-truth DSM mesh (area {target_area})")
ax1.set_zlabel("elevation (m)")

ax2 = fig.add_subplot(1, 2, 2, projection="3d")
ax2.plot_surface(X, Y, Z, facecolors=rgb_ds, rstride=1, cstride=1, shade=False)
ax2.set_title(f"Predicted (fine-tuned + calibrated) mesh (area {target_area})")
ax2.set_zlabel("elevation (m)")

plt.tight_layout()
out_path = OUT_FIG / f"mini_3d_mesh_area{target_area}.png"
plt.savefig(out_path, dpi=110)
print(f"Saved -> {out_path}")
print("This confirms prediction -> DSM -> navigable-surface-mesh is mechanically viable.")
print("NOTE: this is an oracle-calibrated single-tile demo for pipeline sanity only, NOT a generalization claim.")
