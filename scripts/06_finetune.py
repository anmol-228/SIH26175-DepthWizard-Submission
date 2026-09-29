"""Experiment 4: lightweight fine-tuning of Depth Anything V2 Small.

Design choices (documented per Section 15/16):
- Option A (train head/neck only, freeze the DINOv2 backbone). Chosen because
  it is the fastest, most parameter-efficient option, matches "start small"
  guidance, and 513 train patches is a small dataset where freezing the
  pretrained visual backbone reduces overfitting risk.
- Loss target: PER-PATCH scale+shift-invariant L1 (SSI-L1, closed-form
  alignment each step, matching MiDaS/Ranftl-style training and the SiLog-
  family losses used in the closest published precedent for this exact task
  -- Sat3R / Depth Any Canopy fine-tuning DA2 for overhead height estimation).
  This is a deliberate choice: absolute per-tile elevation offset is NOT
  learnable from RGB pixels alone (confirmed empirically by the huge spread
  in the Experiment-1 constant-baseline RMSE across test tiles, which is
  dominated by inter-tile datum shift, not local structure) -- so the network
  is trained to recover local relief/shape, exactly matching the SIH PS's own
  architecture (a separate scale-calibration module handles the absolute
  offset; see 04_EXPERIMENT_PLAN.md requirement mapping).
"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
import torch.nn.functional as F
from da2_utils import preprocess_rgb, load_model, resize_pred_to, MODEL_ID
from metrics import all_metrics

ROOT = Path(__file__).resolve().parents[1]
PATCH_DIR = ROOT / "data" / "processed" / "patches"
CKPT_DIR = ROOT / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)
METRICS_DIR = ROOT / "outputs" / "metrics"
LOG_DIR = ROOT / "logs"

device = "cuda" if torch.cuda.is_available() else "cpu"
manifest = json.load(open(PATCH_DIR / "manifest.json"))

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

import os
EPOCHS = int(os.environ.get("FT_EPOCHS", 8))
BATCH_SIZE = 4
LR = 5e-5
BACKBONE_LR = 5e-6  # lower LR for unfrozen backbone layers -- avoid destroying pretrained features
UNFREEZE_LAST_N = int(os.environ.get("FT_UNFREEZE_LAST_N", 0))  # Loop-C iteration 1: partial backbone fine-tune
TARGET_SIZE = 518  # match backbone's native resolution -> no extra resize of predictions during training
CKPT_NAME = os.environ.get("FT_CKPT_NAME", "da2_small_finetuned_best.pt")

model = load_model(device)

# Freeze backbone (DINOv2 encoder) except optionally the last N transformer blocks
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
            n_trainable += p.numel()
            unfrozen_backbone_params.append(p)
        else:
            n_frozen += p.numel()
    else:
        p.requires_grad_(True)
        n_trainable += p.numel()
        head_params.append(p)
print(f"Frozen params: {n_frozen:,} | Trainable params (neck+head{f'+last {UNFREEZE_LAST_N} backbone blocks' if UNFREEZE_LAST_N else ''}): {n_trainable:,}")

optimizer = torch.optim.AdamW([
    {"params": head_params, "lr": LR},
    {"params": unfrozen_backbone_params, "lr": BACKBONE_LR},
], weight_decay=1e-4)

def ssi_l1_loss(pred, gt, mask):
    """Scale+shift-invariant L1, closed-form per-sample OLS alignment (Ranftl et al.
    2020 style), computed via mean/covariance identities (O(N), differentiable)
    rather than torch.linalg.lstsq -- lstsq's generic dense driver allocates an
    intermediate that scales with N^2 in the pixel count during backward (N=268k
    pixels per 518x518 patch caused a 268 GiB OOM), which is wrong for a rank-2
    problem with this many equations.
    """
    B = pred.shape[0]
    losses = []
    for b in range(B):
        p = pred[b][mask[b]]
        g = gt[b][mask[b]]
        p_mean, g_mean = p.mean(), g.mean()
        p_c, g_c = p - p_mean, g - g_mean
        var_p = (p_c * p_c).mean()
        cov_pg = (p_c * g_c).mean()
        s = cov_pg / (var_p + 1e-8)
        t = g_mean - s * p_mean
        aligned = s * p + t
        losses.append(F.l1_loss(aligned, g))
    return torch.stack(losses).mean()

def load_patch(split, entry):
    d = np.load(PATCH_DIR / split / entry["file"])
    return d["rgb"], d["dsm"]

train_entries = [m for m in manifest if m["split"] == "train"]
val_entries = [m for m in manifest if m["split"] == "val"]

def make_batch(entries):
    rgbs, dsms = [], []
    for e in entries:
        rgb, dsm = load_patch("train" if e["split"] == "train" else e["split"], e)
        rgbs.append(rgb); dsms.append(dsm)
    x = torch.stack([preprocess_rgb(r) for r in rgbs]).to(device)
    gt = torch.from_numpy(np.stack(dsms)).float()
    gt_r = F.interpolate(gt.unsqueeze(1), size=(TARGET_SIZE, TARGET_SIZE), mode="bilinear", align_corners=False).squeeze(1).to(device)
    mask = torch.ones_like(gt_r, dtype=torch.bool)  # patches were filtered to >=98% valid
    return x, gt_r, mask

@torch.no_grad()
def evaluate(entries, tag):
    model.eval()
    all_pred, all_gt, per_tile_pred, per_tile_gt = [], [], {}, {}
    for e in entries:
        rgb, dsm = load_patch(e["split"], e)
        x = preprocess_rgb(rgb).unsqueeze(0).to(device)
        out = model(pixel_values=x).predicted_depth
        pred = resize_pred_to(out, dsm.shape).squeeze(0).cpu().numpy()
        # per-patch oracle scale/shift (this eval is PRE-calibration-experiment;
        # here we only check raw shift-invariant fit quality on this patch itself
        # is NOT allowed -- so we defer real calibration to script 07).
        all_pred.append(pred); all_gt.append(dsm)
        per_tile_pred.setdefault(e["area"], []).append(pred)
        per_tile_gt.setdefault(e["area"], []).append(dsm)
    model.train()
    return all_pred, all_gt

print(f"\nTraining: {EPOCHS} epochs, batch={BATCH_SIZE}, lr={LR}, {len(train_entries)} train patches, "
      f"{len(val_entries)} val patches, loss=SSI-L1, target_size={TARGET_SIZE}")

history = []
best_val_ssi = float("inf")
n_steps_per_epoch = len(train_entries) // BATCH_SIZE

for epoch in range(EPOCHS):
    model.train()
    perm = np.random.permutation(len(train_entries))
    epoch_losses = []
    t0 = time.time()
    for step in range(n_steps_per_epoch):
        idx = perm[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]
        batch_entries = [train_entries[i] for i in idx]
        x, gt, mask = make_batch(batch_entries)
        out = model(pixel_values=x).predicted_depth  # (B, 518, 518) already at target res
        loss = ssi_l1_loss(out, gt, mask)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        epoch_losses.append(loss.item())
        if torch.isnan(loss):
            print("NaN loss detected -- stopping")
            break
    mean_loss = float(np.mean(epoch_losses))

    # validation SSI-L1 (model-selection signal only; oracle calibration for
    # the REPORTED metrics is fit on TRAIN in script 07, never on val/test)
    val_preds, val_gts = evaluate(val_entries, "val")
    val_ssi_losses = []
    for p, g in zip(val_preds, val_gts):
        pt = torch.from_numpy(p).unsqueeze(0)
        gt_t = torch.from_numpy(g).unsqueeze(0)
        m = torch.ones_like(gt_t, dtype=torch.bool)
        val_ssi_losses.append(ssi_l1_loss(pt, gt_t, m).item())
    val_ssi = float(np.mean(val_ssi_losses))

    dt = time.time() - t0
    print(f"Epoch {epoch+1}/{EPOCHS}  train_ssi_l1={mean_loss:.4f}  val_ssi_l1={val_ssi:.4f}  ({dt:.1f}s)")
    history.append({"epoch": epoch + 1, "train_ssi_l1": mean_loss, "val_ssi_l1": val_ssi, "time_s": dt})

    if val_ssi < best_val_ssi:
        best_val_ssi = val_ssi
        torch.save(model.state_dict(), CKPT_DIR / CKPT_NAME)
        print(f"  -> new best (val_ssi_l1={val_ssi:.4f}), checkpoint saved")

with open(LOG_DIR / f"finetune_history_{CKPT_NAME}.json", "w") as f:
    json.dump(history, f, indent=2)

print(f"\nDone. Best val SSI-L1: {best_val_ssi:.4f}. Checkpoint: checkpoints/{CKPT_NAME}")
print(f"Total trainable params updated: {n_trainable:,} / total params (backbone frozen: {n_frozen:,})")
