"""Experiment A2: can training AT a degraded GSD recover useful signal?

Fine-tunes a FRESH copy of DA2-Small (from the original pretrained checkpoint,
not from the Phase-1 9cm fine-tune) on TRAIN patches whose input RGB has been
degraded to GSD_TARGET, using the same recipe as Phase-1 iteration 2 (partial
backbone unfreeze, last 2 blocks, 16 epochs) for direct comparability. Then
evaluates on TEST patches degraded to the SAME GSD with full-oracle and
10-point sparse calibration, exactly as in A1, so the two are directly
comparable at that one GSD.
"""
import sys, json, time, os
from pathlib import Path
P2ROOT = Path(__file__).resolve().parents[1]
POFROOT = P2ROOT.parent
sys.path.insert(0, str(POFROOT / "src"))
sys.path.insert(0, str(P2ROOT / "scripts"))
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import RANSACRegressor, LinearRegression
from da2_utils import preprocess_rgb, load_model, resize_pred_to
from metrics import all_metrics, fit_scale_shift
from gsd_utils import degrade_rgb_to_gsd, NATIVE_GSD

GSD_TARGET = float(os.environ.get("GSD_TARGET", "3.0"))
EPOCHS = int(os.environ.get("FT_EPOCHS", 16))
UNFREEZE_LAST_N = int(os.environ.get("FT_UNFREEZE_LAST_N", 2))
BATCH_SIZE = 4
LR = 5e-5
BACKBONE_LR = 5e-6
TARGET_SIZE = 518
SEED = 42

PATCH_DIR = POFROOT / "data" / "processed" / "patches"
CKPT_DIR = P2ROOT / "outputs"  # keep phase-2 checkpoints separate from phase-1
CKPT_DIR.mkdir(exist_ok=True)
CKPT_PATH = CKPT_DIR / f"da2_gsd{GSD_TARGET}m_adapted.pt"
METRICS_DIR = P2ROOT / "outputs" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = P2ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED); np.random.seed(SEED)
manifest = json.load(open(PATCH_DIR / "manifest.json"))
train_entries = [m for m in manifest if m["split"] == "train"]
val_entries = [m for m in manifest if m["split"] == "val"]
test_entries = [m for m in manifest if m["split"] == "test"]

model = load_model(device)
n_frozen, n_trainable = 0, 0
unfrozen_backbone_params, head_params = [], []
for name, p in model.named_parameters():
    if name.startswith("backbone."):
        unfreeze = False
        if UNFREEZE_LAST_N > 0:
            import re
            m = re.search(r"encoder\.layer\.(\d+)\.", name)
            if m and int(m.group(1)) >= (12 - UNFREEZE_LAST_N):
                unfreeze = True
        p.requires_grad_(unfreeze)
        if unfreeze:
            n_trainable += p.numel(); unfrozen_backbone_params.append(p)
        else:
            n_frozen += p.numel()
    else:
        p.requires_grad_(True); n_trainable += p.numel(); head_params.append(p)

optimizer = torch.optim.AdamW([
    {"params": head_params, "lr": LR},
    {"params": unfrozen_backbone_params, "lr": BACKBONE_LR},
], weight_decay=1e-4)

def ssi_l1_loss(pred, gt, mask):
    B = pred.shape[0]; losses = []
    for b in range(B):
        p = pred[b][mask[b]]; g = gt[b][mask[b]]
        p_mean, g_mean = p.mean(), g.mean()
        p_c, g_c = p - p_mean, g - g_mean
        var_p = (p_c * p_c).mean(); cov_pg = (p_c * g_c).mean()
        s = cov_pg / (var_p + 1e-8); t = g_mean - s * p_mean
        aligned = s * p + t
        losses.append(F.l1_loss(aligned, g))
    return torch.stack(losses).mean()

def load_patch(split, entry, degrade=True):
    d = np.load(PATCH_DIR / split / entry["file"])
    rgb, dsm = d["rgb"], d["dsm"]
    if degrade:
        rgb = degrade_rgb_to_gsd(rgb, GSD_TARGET, NATIVE_GSD)
    return rgb, dsm

def make_batch(entries):
    rgbs, dsms = [], []
    for e in entries:
        rgb, dsm = load_patch(e["split"], e)
        rgbs.append(rgb); dsms.append(dsm)
    x = torch.stack([preprocess_rgb(r) for r in rgbs]).to(device)
    gt = torch.from_numpy(np.stack(dsms)).float()
    gt_r = F.interpolate(gt.unsqueeze(1), size=(TARGET_SIZE, TARGET_SIZE), mode="bilinear", align_corners=False).squeeze(1).to(device)
    mask = torch.ones_like(gt_r, dtype=torch.bool)
    return x, gt_r, mask

@torch.no_grad()
def val_ssi():
    model.eval(); losses = []
    for e in val_entries:
        rgb, dsm = load_patch(e["split"], e)
        x = preprocess_rgb(rgb).unsqueeze(0).to(device)
        out = model(pixel_values=x).predicted_depth
        gt_t = torch.from_numpy(dsm).unsqueeze(0)
        pt = resize_pred_to(out, dsm.shape).cpu()
        m = torch.ones_like(gt_t, dtype=torch.bool)
        losses.append(ssi_l1_loss(pt, gt_t, m).item())
    model.train()
    return float(np.mean(losses))

print(f"GSD_TARGET={GSD_TARGET}m  epochs={EPOCHS}  unfreeze_last_n={UNFREEZE_LAST_N}  "
      f"trainable={n_trainable:,} frozen={n_frozen:,}")

history = []
best_val = float("inf")
n_steps = len(train_entries) // BATCH_SIZE
for epoch in range(EPOCHS):
    model.train()
    perm = np.random.permutation(len(train_entries))
    epoch_losses = []; t0 = time.time()
    for step in range(n_steps):
        idx = perm[step*BATCH_SIZE:(step+1)*BATCH_SIZE]
        batch = [train_entries[i] for i in idx]
        x, gt, mask = make_batch(batch)
        out = model(pixel_values=x).predicted_depth
        loss = ssi_l1_loss(out, gt, mask)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        epoch_losses.append(loss.item())
    v = val_ssi()
    dt = time.time() - t0
    print(f"Epoch {epoch+1}/{EPOCHS}  train_ssi={np.mean(epoch_losses):.4f}  val_ssi={v:.4f}  ({dt:.1f}s)")
    history.append({"epoch": epoch+1, "train_ssi_l1": float(np.mean(epoch_losses)), "val_ssi_l1": v, "time_s": dt})
    if v < best_val:
        best_val = v
        torch.save(model.state_dict(), CKPT_PATH)

with open(LOG_DIR / f"A2_finetune_history_gsd{GSD_TARGET}m.json", "w") as f:
    json.dump(history, f, indent=2)
print(f"Best val SSI-L1: {best_val:.4f}. Checkpoint: {CKPT_PATH}")

# ---- Evaluate on TEST at the same degraded GSD ----
model.load_state_dict(torch.load(CKPT_PATH, map_location=device, weights_only=True))
model.eval()

@torch.no_grad()
def predict(rgb):
    x = preprocess_rgb(rgb).unsqueeze(0).to(device)
    out = model(pixel_values=x).predicted_depth
    return resize_pred_to(out, rgb.shape[:2]).squeeze(0).cpu().numpy()

per_tile_pred, per_tile_gt = {}, {}
for e in test_entries:
    rgb, dsm = load_patch(e["split"], e)
    pred = predict(rgb)
    per_tile_pred.setdefault(e["area"], []).append(pred)
    per_tile_gt.setdefault(e["area"], []).append(dsm)
areas = sorted(per_tile_pred.keys())

all_pred_full, all_gt_full = [], []
per_tile_oracle = {}
for area in areas:
    p = np.concatenate([x.ravel() for x in per_tile_pred[area]])
    g = np.concatenate([x.ravel() for x in per_tile_gt[area]])
    a, b = fit_scale_shift(p, g)
    p_cal = a * p + b
    per_tile_oracle[area] = all_metrics(p_cal, g)
    all_pred_full.append(p_cal); all_gt_full.append(g)
global_oracle = all_metrics(np.concatenate(all_pred_full), np.concatenate(all_gt_full))

rng = np.random.default_rng(42)
all_pred_sparse, all_gt_sparse = [], []
per_tile_sparse = {}
for area in areas:
    p = np.concatenate([x.ravel() for x in per_tile_pred[area]])
    g = np.concatenate([x.ravel() for x in per_tile_gt[area]])
    n = min(10, p.size)
    idx = rng.choice(p.size, size=n, replace=False)
    try:
        reg = RANSACRegressor(LinearRegression(), min_samples=max(2, n // 2), random_state=42)
        reg.fit(p[idx].reshape(-1, 1), g[idx])
        a = float(reg.estimator_.coef_[0]); b = float(reg.estimator_.intercept_)
    except Exception:
        a, b = fit_scale_shift(p[idx], g[idx])
    p_cal = a * p + b
    per_tile_sparse[area] = all_metrics(p_cal, g)
    all_pred_sparse.append(p_cal); all_gt_sparse.append(g)
global_sparse10 = all_metrics(np.concatenate(all_pred_sparse), np.concatenate(all_gt_sparse))

print(f"\n[A2 adapted @ {GSD_TARGET}m] full_oracle RMSE={global_oracle['rmse']:.3f} corr={global_oracle['corr']:.3f}  |  "
      f"sparse10 RMSE={global_sparse10['rmse']:.3f} corr={global_sparse10['corr']:.3f}")

out = {
    "gsd_m": GSD_TARGET, "epochs": EPOCHS, "unfreeze_last_n": UNFREEZE_LAST_N,
    "global_full_oracle": global_oracle, "global_sparse10": global_sparse10,
    "per_tile_full_oracle": per_tile_oracle, "per_tile_sparse10": per_tile_sparse,
    "best_val_ssi_l1": best_val,
}
with open(METRICS_DIR / f"A2_gsd_adapted_{GSD_TARGET}m.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"Saved -> {METRICS_DIR / f'A2_gsd_adapted_{GSD_TARGET}m.json'}")
