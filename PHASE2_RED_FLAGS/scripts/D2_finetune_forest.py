"""Experiment D, Attempt 1: fine-tune DA2-Small directly on real forest train
patches (Sentinel-2 RGB + ETH Canopy Height), same SSI-L1 recipe as Phase-1's
06_finetune.py (closed-form per-sample scale+shift alignment, AdamW,
lr=5e-5 head / 5e-6 backbone), starting from the ORIGINAL pretrained
DA2-Small weights (not the Vaihingen checkpoint -- we want a clean forest
adaptation signal, not a double-domain-shift confound).

Deviations from Phase-1's recipe, and why:
  - Backbone kept FULLY FROZEN (UNFREEZE_LAST_N=0). Only 16 train patches are
    available here (vs. Phase-1's 513) -- unfreezing backbone transformer
    blocks with this little data risks catastrophic overfitting/collapse.
  - Light augmentation added (random 90-degree rotation + flips). Overhead
    canopy imagery has no preferred orientation, so this is a valid
    label-preserving augmentation, and is needed to get any epoch diversity
    out of only 16 patches (4 steps/epoch at batch=4).
  - More epochs (higher EPOCHS default) to compensate for having far fewer
    steps/epoch than Phase-1's run.
  - Loss respects the per-patch VALID mask (nodata/cloud px excluded) rather
    than assuming 100% validity.
"""
import sys, json, time, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np
import torch
import torch.nn.functional as F
from da2_utils import preprocess_rgb, load_model, resize_pred_to
from metrics import all_metrics, fit_scale_shift

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
PATCH_DIR = ROOT / "data" / "forest_eth_s2" / "patches"
CKPT_DIR = ROOT / "outputs" / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR = ROOT / "outputs" / "metrics"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
manifest = json.load(open(PATCH_DIR / "manifest.json"))

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

EPOCHS = int(os.environ.get("FT_EPOCHS", 60))
BATCH_SIZE = 4
LR = 5e-5
BACKBONE_LR = 5e-6
UNFREEZE_LAST_N = int(os.environ.get("FT_UNFREEZE_LAST_N", 0))
TARGET_SIZE = 518
CKPT_NAME = os.environ.get("FT_CKPT_NAME", "D2_da2_small_forest_finetuned.pt")

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
        p.requires_grad_(True)
        n_trainable += p.numel(); head_params.append(p)
print(f"Frozen params: {n_frozen:,} | Trainable params: {n_trainable:,}")

optimizer = torch.optim.AdamW([
    {"params": head_params, "lr": LR},
    {"params": unfrozen_backbone_params, "lr": BACKBONE_LR},
], weight_decay=1e-4)


def ssi_l1_loss(pred, gt, mask):
    B = pred.shape[0]
    losses = []
    for b in range(B):
        p = pred[b][mask[b]]
        g = gt[b][mask[b]]
        if p.numel() < 10:
            continue
        p_mean, g_mean = p.mean(), g.mean()
        p_c, g_c = p - p_mean, g - g_mean
        var_p = (p_c * p_c).mean()
        cov_pg = (p_c * g_c).mean()
        s = cov_pg / (var_p + 1e-8)
        t = g_mean - s * p_mean
        aligned = s * p + t
        losses.append(F.l1_loss(aligned, g))
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=pred.device, requires_grad=True)


def load_patch(entry):
    d = np.load(PATCH_DIR / entry["file"])
    return d["rgb"], d["chm"], d["valid"]


train_entries = [m for m in manifest if m["split"] == "train"]
val_entries = [m for m in manifest if m["split"] == "val"]
test_entries = [m for m in manifest if m["split"] == "test"]
print(f"train={len(train_entries)} val={len(val_entries)} test={len(test_entries)}")


def augment(rgb, chm, valid, rng):
    k = rng.integers(0, 4)
    rgb = np.rot90(rgb, k, axes=(0, 1))
    chm = np.rot90(chm, k, axes=(0, 1))
    valid = np.rot90(valid, k, axes=(0, 1))
    if rng.random() < 0.5:
        rgb = np.flip(rgb, axis=1); chm = np.flip(chm, axis=1); valid = np.flip(valid, axis=1)
    return np.ascontiguousarray(rgb), np.ascontiguousarray(chm), np.ascontiguousarray(valid)


rng = np.random.default_rng(SEED)


def make_batch(entries, do_aug):
    rgbs, chms, valids = [], [], []
    for e in entries:
        rgb, chm, valid = load_patch(e)
        if do_aug:
            rgb, chm, valid = augment(rgb, chm, valid, rng)
        rgbs.append(rgb); chms.append(chm); valids.append(valid)
    x = torch.stack([preprocess_rgb(r) for r in rgbs]).to(device)
    gt = torch.from_numpy(np.stack(chms)).float()
    vmask = torch.from_numpy(np.stack(valids)).bool()
    gt_r = F.interpolate(gt.unsqueeze(1), size=(TARGET_SIZE, TARGET_SIZE), mode="bilinear", align_corners=False).squeeze(1).to(device)
    vmask_r = F.interpolate(vmask.float().unsqueeze(1), size=(TARGET_SIZE, TARGET_SIZE), mode="nearest").squeeze(1).bool().to(device)
    return x, gt_r, vmask_r


@torch.no_grad()
def evaluate_ssi(entries):
    model.eval()
    losses = []
    for e in entries:
        rgb, chm, valid = load_patch(e)
        x = preprocess_rgb(rgb).unsqueeze(0).to(device)
        out = model(pixel_values=x).predicted_depth
        pred = resize_pred_to(out, chm.shape).squeeze(0)
        gt_t = torch.from_numpy(chm).float().to(device).unsqueeze(0)
        m = torch.from_numpy(valid).bool().to(device).unsqueeze(0)
        losses.append(ssi_l1_loss(pred.unsqueeze(0), gt_t, m).item())
    model.train()
    return float(np.mean(losses))


print(f"\nTraining: {EPOCHS} epochs, batch={BATCH_SIZE}, lr={LR}/{BACKBONE_LR}, "
      f"{len(train_entries)} train patches, {len(val_entries)} val patches, "
      f"UNFREEZE_LAST_N={UNFREEZE_LAST_N}, augment=rot90+hflip")

history = []
best_val_ssi = float("inf")
n_steps_per_epoch = max(1, len(train_entries) // BATCH_SIZE)

for epoch in range(EPOCHS):
    model.train()
    perm = rng.permutation(len(train_entries))
    epoch_losses = []
    t0 = time.time()
    for step in range(n_steps_per_epoch):
        idx = perm[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]
        if len(idx) == 0:
            continue
        batch_entries = [train_entries[i] for i in idx]
        x, gt, mask = make_batch(batch_entries, do_aug=True)
        out = model(pixel_values=x).predicted_depth
        loss = ssi_l1_loss(out, gt, mask)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        epoch_losses.append(loss.item())
    mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
    val_ssi = evaluate_ssi(val_entries)
    dt = time.time() - t0
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}  train_ssi_l1={mean_loss:.4f}  val_ssi_l1={val_ssi:.4f}  ({dt:.1f}s)")
    history.append({"epoch": epoch + 1, "train_ssi_l1": mean_loss, "val_ssi_l1": val_ssi, "time_s": dt})
    if val_ssi < best_val_ssi:
        best_val_ssi = val_ssi
        torch.save(model.state_dict(), CKPT_DIR / CKPT_NAME)

with open(LOG_DIR / f"D2_finetune_history_{CKPT_NAME}.json", "w") as f:
    json.dump(history, f, indent=2)
print(f"\nDone. Best val SSI-L1: {best_val_ssi:.4f}. Checkpoint: {CKPT_DIR / CKPT_NAME}")

# ---- final eval on held-out TEST patches (full-oracle scale/shift, same as D1) ----
model.load_state_dict(torch.load(CKPT_DIR / CKPT_NAME, map_location=device, weights_only=True))
model.eval()


@torch.no_grad()
def predict(rgb):
    x = preprocess_rgb(rgb).unsqueeze(0).to(device)
    out = model(pixel_values=x).predicted_depth
    return resize_pred_to(out, rgb.shape[:2]).squeeze(0).cpu().numpy()


per_patch = {}
all_pred_cal, all_gt = [], []
for e in test_entries:
    rgb, chm, valid = load_patch(e)
    pred = predict(rgb)
    p, g = pred[valid], chm[valid]
    a, b = fit_scale_shift(p, g)
    p_cal = a * p + b
    m = all_metrics(p_cal, g)
    per_patch[e["file"]] = {**m, "a": a, "b": b, "n_px": int(p.size)}
    all_pred_cal.append(p_cal); all_gt.append(g)
    print(f"  [D2 forest-finetuned] {e['file']}: RMSE={m['rmse']:.3f} MAE={m['mae']:.3f} corr={m['corr']:.3f}")

all_pred_cal = np.concatenate(all_pred_cal); all_gt = np.concatenate(all_gt)
global_m = all_metrics(all_pred_cal, all_gt)
print(f"[D2 forest-finetuned] GLOBAL POOLED: RMSE={global_m['rmse']:.3f} MAE={global_m['mae']:.3f} corr={global_m['corr']:.3f}")

with open(METRICS_DIR / "D2_forest_finetuned_eval.json", "w") as f:
    json.dump({"per_patch_full_oracle": per_patch, "global_pooled": global_m,
                "unfreeze_last_n": UNFREEZE_LAST_N, "epochs": EPOCHS,
                "best_val_ssi_l1": best_val_ssi, "n_train": len(train_entries),
                "n_val": len(val_entries), "n_test": len(test_entries)}, f, indent=2)
print("Saved -> outputs/metrics/D2_forest_finetuned_eval.json")
