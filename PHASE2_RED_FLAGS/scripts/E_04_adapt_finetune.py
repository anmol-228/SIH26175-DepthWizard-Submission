"""Experiment E, step 4 (OPTIONAL, attempt 1 of the max-2 adaptation-attempt
budget for this red-flag category): one lightweight fine-tuning adaptation
pass on the real hilly-terrain data itself, to see whether a small amount of
in-domain adaptation recovers any of the whole-scene-calibration collapse
seen in E_03.

Design (mirrors scripts/06_finetune.py's Option A, adapted for a MUCH
smaller dataset -- 17 train / 6 val / 5 test patches, vs Vaihingen's 513):
  - Start from the Phase-1 Vaihingen-fine-tuned checkpoint (continual
    adaptation, not from scratch) since it is the most relevant available
    starting point.
  - Freeze the DINOv2 backbone entirely; train only the neck+head. With 17
    patches, unfreezing any backbone layers would almost certainly overfit
    catastrophically (513 Vaihingen patches were already a small-data regime
    for partial backbone unfreeze; 17 is roughly 30x smaller).
  - Loss: same per-patch SSI-L1 as Phase-1 (scale+shift-invariant), so the
    network is trained to recover LOCAL shape, not absolute per-patch datum.
  - Model selection on adapt_val (6 patches); final numbers reported ONLY on
    adapt_test (5 patches), which were held out of training entirely.
  - Given the tiny dataset, this is explicitly a low-power exploratory
    check, not a claim of a robust fix -- reported honestly either way.
"""
import sys, json, time
from pathlib import Path
P2ROOT = Path(__file__).resolve().parents[1]
POFROOT = P2ROOT.parent
sys.path.insert(0, str(POFROOT / "src"))
import numpy as np
import torch
import torch.nn.functional as F
from da2_utils import preprocess_rgb, load_model, resize_pred_to
from metrics import all_metrics, fit_scale_shift
from sklearn.linear_model import RANSACRegressor, LinearRegression

PATCH_DIR = P2ROOT / "data" / "hilly_terrain" / "patches"
CKPT_DIR = P2ROOT / "outputs"
LOG_DIR = P2ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
OUT_DIR = P2ROOT / "outputs" / "metrics"

device = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

manifest = json.load(open(PATCH_DIR / "manifest.json"))
train_entries = [m for m in manifest if m["split"] == "adapt_train"]
val_entries = [m for m in manifest if m["split"] == "adapt_val"]
test_entries = [m for m in manifest if m["split"] == "adapt_test"]
print(f"train={len(train_entries)} val={len(val_entries)} test={len(test_entries)}")

EPOCHS = 25
BATCH_SIZE = 4
LR = 5e-5
TARGET_SIZE = 518
START_CKPT = POFROOT / "checkpoints" / "da2_small_finetuned_iter2_partial_unfreeze.pt"
OUT_CKPT = CKPT_DIR / "da2_hilly_adapted_attempt1.pt"

model = load_model(device)
model.load_state_dict(torch.load(START_CKPT, map_location=device, weights_only=True))

head_params = []
n_frozen, n_trainable = 0, 0
for name, p in model.named_parameters():
    if name.startswith("backbone."):
        p.requires_grad_(False)
        n_frozen += p.numel()
    else:
        p.requires_grad_(True)
        n_trainable += p.numel()
        head_params.append(p)
print(f"Frozen (backbone): {n_frozen:,}  Trainable (neck+head): {n_trainable:,}")
optimizer = torch.optim.AdamW(head_params, lr=LR, weight_decay=1e-4)


def ssi_l1_loss(pred, gt, mask):
    B = pred.shape[0]
    losses = []
    for b in range(B):
        p = pred[b][mask[b]]; g = gt[b][mask[b]]
        p_mean, g_mean = p.mean(), g.mean()
        p_c, g_c = p - p_mean, g - g_mean
        var_p = (p_c * p_c).mean()
        cov_pg = (p_c * g_c).mean()
        s = cov_pg / (var_p + 1e-8)
        t = g_mean - s * p_mean
        aligned = s * p + t
        losses.append(F.l1_loss(aligned, g))
    return torch.stack(losses).mean()


def load_patch(entry):
    d = np.load(PATCH_DIR / entry["file"])
    return d["rgb"], d["dsm"]


def make_batch(entries):
    rgbs, dsms = [], []
    for e in entries:
        rgb, dsm = load_patch(e)
        rgbs.append(rgb); dsms.append(dsm)
    x = torch.stack([preprocess_rgb(r) for r in rgbs]).to(device)
    gt = torch.from_numpy(np.stack(dsms)).float()
    gt_r = F.interpolate(gt.unsqueeze(1), size=(TARGET_SIZE, TARGET_SIZE), mode="bilinear", align_corners=False).squeeze(1).to(device)
    mask = torch.ones_like(gt_r, dtype=torch.bool)
    return x, gt_r, mask


@torch.no_grad()
def val_ssi(entries):
    model.eval()
    losses = []
    for e in entries:
        rgb, dsm = load_patch(e)
        x = preprocess_rgb(rgb).unsqueeze(0).to(device)
        out = model(pixel_values=x).predicted_depth
        pred = resize_pred_to(out, dsm.shape).squeeze(0).cpu().numpy()
        pt = torch.from_numpy(pred).unsqueeze(0); gt_t = torch.from_numpy(dsm).unsqueeze(0)
        m = torch.ones_like(gt_t, dtype=torch.bool)
        losses.append(ssi_l1_loss(pt, gt_t, m).item())
    model.train()
    return float(np.mean(losses))


@torch.no_grad()
def predict_all(entries):
    model.eval()
    preds, gts, slopes = [], [], []
    for e in entries:
        d = np.load(PATCH_DIR / e["file"])
        rgb, dsm, slope = d["rgb"], d["dsm"], d["slope"]
        x = preprocess_rgb(rgb).unsqueeze(0).to(device)
        out = model(pixel_values=x).predicted_depth
        pred = resize_pred_to(out, dsm.shape).squeeze(0).cpu().numpy()
        preds.append(pred); gts.append(dsm); slopes.append(slope)
    return preds, gts, slopes


def eval_report(entries, tag):
    preds, gts, slopes = predict_all(entries)
    p_all = np.concatenate([x.ravel() for x in preds])
    g_all = np.concatenate([x.ravel() for x in gts])
    a, b = fit_scale_shift(p_all, g_all)
    full_oracle = all_metrics(a * p_all + b, g_all)
    pp = []
    for p, g in zip(preds, gts):
        a_p, b_p = fit_scale_shift(p.ravel(), g.ravel())
        pp.append(all_metrics((a_p * p + b_p).ravel(), g.ravel())["rmse"])
    rng = np.random.default_rng(7)
    n = min(10, p_all.size)
    idx = rng.choice(p_all.size, size=n, replace=False)
    try:
        reg = RANSACRegressor(LinearRegression(), min_samples=max(2, n // 2), random_state=7)
        reg.fit(p_all[idx].reshape(-1, 1), g_all[idx])
        a_s, b_s = float(reg.estimator_.coef_[0]), float(reg.estimator_.intercept_)
    except Exception:
        a_s, b_s = fit_scale_shift(p_all[idx], g_all[idx])
    sparse10 = all_metrics(a_s * p_all + b_s, g_all)
    print(f"[{tag}] n_patches={len(entries)} full_scene_oracle RMSE={full_oracle['rmse']:.2f}  "
          f"per_patch_oracle mean_RMSE={np.mean(pp):.2f}  sparse10 RMSE={sparse10['rmse']:.2f}")
    return {"full_scene_oracle": full_oracle, "per_patch_oracle_mean_rmse": float(np.mean(pp)),
            "per_patch_oracle_std_rmse": float(np.std(pp)), "sparse_10pt_single_draw": sparse10}


if __name__ == "__main__":
    print("\n--- BEFORE adaptation (Vaihingen checkpoint as-is), on adapt_test ---")
    before = eval_report(test_entries, "before-adaptation")

    print(f"\n--- Adapting: {EPOCHS} epochs, {len(train_entries)} train patches, neck+head only ---")
    history = []
    best_val = float("inf")
    n_steps = max(1, len(train_entries) // BATCH_SIZE)
    for epoch in range(EPOCHS):
        model.train()
        perm = np.random.permutation(len(train_entries))
        losses = []
        t0 = time.time()
        for step in range(n_steps):
            idx = perm[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]
            if len(idx) == 0:
                continue
            batch = [train_entries[i] for i in idx]
            x, gt, mask = make_batch(batch)
            out = model(pixel_values=x).predicted_depth
            loss = ssi_l1_loss(out, gt, mask)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(head_params, 1.0)
            optimizer.step()
            losses.append(loss.item())
        v = val_ssi(val_entries)
        dt = time.time() - t0
        print(f"epoch {epoch+1}/{EPOCHS} train_ssi={np.mean(losses):.4f} val_ssi={v:.4f} ({dt:.1f}s)")
        history.append({"epoch": epoch + 1, "train_ssi_l1": float(np.mean(losses)), "val_ssi_l1": v, "time_s": dt})
        if v < best_val:
            best_val = v
            torch.save(model.state_dict(), OUT_CKPT)

    with open(LOG_DIR / "E_adapt_finetune_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\n--- AFTER adaptation (best val checkpoint), on adapt_test ---")
    model.load_state_dict(torch.load(OUT_CKPT, map_location=device, weights_only=True))
    after = eval_report(test_entries, "after-adaptation")

    report = {"n_train": len(train_entries), "n_val": len(val_entries), "n_test": len(test_entries),
              "epochs": EPOCHS, "best_val_ssi_l1": best_val,
              "before_adaptation_on_test": before, "after_adaptation_on_test": after}
    with open(OUT_DIR / "E_adaptation_attempt1.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved -> {OUT_DIR / 'E_adaptation_attempt1.json'}")
