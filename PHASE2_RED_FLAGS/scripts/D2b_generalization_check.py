"""Extra rigor check for Attempt 1 (D2): the 16 train patches and 5 test
patches all come from ONE contiguous ~30x31km scene (spatially disjoint by
block, but geographically adjacent -- same watershed, same imagery epoch).
That raises a fair concern that the strong D2 result partly reflects
single-scene homogeneity rather than real generalization.

This script builds a SECOND, geographically separate AOI (~5.6km buffer south
of the original, a different Olympic NP/NF valley, same Sentinel-2 scene and
same CHM tile so no new downloads needed) and evaluates the D2
forest-fine-tuned checkpoint on it WITH NO RETRAINING -- a true
out-of-scene zero-shot generalization check -- against the original
zero-shot pretrained model, for honest comparison.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np
import tifffile
import torch
from pyproj import Transformer
from da2_utils import preprocess_rgb, load_model, resize_pred_to
from metrics import all_metrics, fit_scale_shift

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
RAW = ROOT / "data" / "forest_eth_s2" / "raw"
METRICS_DIR = ROOT / "outputs" / "metrics"
FIG = ROOT / "outputs" / "figures"

TCI = RAW / "S2A_10TDT_20210902_TCI.tif"
SCL = RAW / "S2A_10TDT_20210902_SCL.tif"
CHM = RAW / "ETH_GlobalCanopyHeight_10m_2020_N45W126_Map.tif"

S2_OX, S2_OY, S2_PX = 399960.0, 5300040.0, 10.0
CHM_OX, CHM_OY, CHM_PX = -126.0, 48.0, 8.333333333333333e-05
CHM_NODATA = 255
SCL_INVALID = {0, 1, 3, 7, 8, 9, 10}
PATCH = 512

# geographically separate AOI: ~5.6km south of the original D0 AOI
LAT0, LAT1 = 47.30, 47.50
LON0, LON1 = -123.90, -123.55

device = "cuda" if torch.cuda.is_available() else "cpu"


def build_aoi():
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32610", always_xy=True)
    to_wgs = Transformer.from_crs("EPSG:32610", "EPSG:4326", always_xy=True)
    corners = [(LON0, LAT0), (LON1, LAT0), (LON0, LAT1), (LON1, LAT1)]
    cols, rows = [], []
    for lon, lat in corners:
        x, y = to_utm.transform(lon, lat)
        cols.append((x - S2_OX) / S2_PX); rows.append((S2_OY - y) / S2_PX)
    c0, c1 = int(min(cols)), int(max(cols))
    r0, r1 = int(min(rows)), int(max(rows))
    print(f"OOS AOI2: cols [{c0},{c1}) rows [{r0},{r1}) -> {c1-c0}x{r1-r0} px")

    tci_full = tifffile.imread(str(TCI))
    rgb = tci_full[r0:r1, c0:c1, :].copy(); del tci_full
    scl_full = tifffile.imread(str(SCL))
    scl_sub = scl_full[r0 // 2:r1 // 2 + 1, c0 // 2:c1 // 2 + 1]
    scl_up = np.repeat(np.repeat(scl_sub, 2, axis=0), 2, axis=1)[: (r1 - r0), : (c1 - c0)]
    del scl_full
    chm_full = tifffile.imread(str(CHM))

    H, W = r1 - r0, c1 - c0
    rr, cc = np.meshgrid(np.arange(r0, r1), np.arange(c0, c1), indexing="ij")
    x = S2_OX + (cc + 0.5) * S2_PX
    y = S2_OY - (rr + 0.5) * S2_PX
    lon, lat = to_wgs.transform(x.ravel(), y.ravel())
    lon = np.asarray(lon).reshape(H, W); lat = np.asarray(lat).reshape(H, W)
    chm_col = np.clip(np.round((lon - CHM_OX) / CHM_PX).astype(np.int64), 0, chm_full.shape[1] - 1)
    chm_row = np.clip(np.round((CHM_OY - lat) / CHM_PX).astype(np.int64), 0, chm_full.shape[0] - 1)
    chm_resamp = chm_full[chm_row, chm_col].astype(np.float32)
    del chm_full

    valid = (chm_resamp != CHM_NODATA) & (~np.isin(scl_up, list(SCL_INVALID)))
    print(f"OOS AOI2 valid px: {100*valid.mean():.2f}%  mean CHM (valid)={chm_resamp[valid].mean():.2f}m")

    patches = []
    n_row_blocks, n_col_blocks = H // PATCH, W // PATCH
    for rb in range(n_row_blocks):
        for cb in range(n_col_blocks):
            pr0, pc0 = rb * PATCH, cb * PATCH
            rgb_p = rgb[pr0:pr0+PATCH, pc0:pc0+PATCH, :]
            chm_p = chm_resamp[pr0:pr0+PATCH, pc0:pc0+PATCH]
            valid_p = valid[pr0:pr0+PATCH, pc0:pc0+PATCH]
            if valid_p.mean() < 0.98:
                continue
            patches.append({"rgb": rgb_p, "chm": chm_p, "valid": valid_p, "name": f"oos_r{rb}_c{cb}"})
    print(f"OOS AOI2 kept {len(patches)} candidate patches")
    return patches


@torch.no_grad()
def predict(model, rgb):
    x = preprocess_rgb(rgb).unsqueeze(0).to(device)
    out = model(pixel_values=x).predicted_depth
    return resize_pred_to(out, rgb.shape[:2]).squeeze(0).cpu().numpy()


def run(model, patches, tag):
    per_patch = {}
    all_pred_cal, all_gt = [], []
    for pt in patches:
        pred = predict(model, pt["rgb"])
        p, g = pred[pt["valid"]], pt["chm"][pt["valid"]]
        a, b = fit_scale_shift(p, g)
        p_cal = a * p + b
        m = all_metrics(p_cal, g)
        per_patch[pt["name"]] = {**m, "a": a, "b": b}
        all_pred_cal.append(p_cal); all_gt.append(g)
        print(f"  [{tag}] {pt['name']}: RMSE={m['rmse']:.3f} corr={m['corr']:.3f}")
    all_pred_cal = np.concatenate(all_pred_cal); all_gt = np.concatenate(all_gt)
    global_m = all_metrics(all_pred_cal, all_gt)
    print(f"  [{tag}] GLOBAL: RMSE={global_m['rmse']:.3f} MAE={global_m['mae']:.3f} corr={global_m['corr']:.3f}")
    return {"per_patch": per_patch, "global_pooled": global_m}


if __name__ == "__main__":
    patches = build_aoi()
    results = {}

    print("\n=== OOS check: zero-shot PRETRAINED (never seen forest at all) ===")
    m_zs = load_model(device); m_zs.eval()
    results["zero_shot_pretrained_oos"] = run(m_zs, patches, "zero-shot-oos")
    del m_zs; torch.cuda.empty_cache()

    print("\n=== OOS check: D2 forest-finetuned (trained on the OTHER, non-adjacent AOI) ===")
    m_ft = load_model(device)
    ckpt = ROOT / "outputs" / "checkpoints" / "D2_da2_small_forest_finetuned.pt"
    m_ft.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    m_ft.eval()
    results["forest_finetuned_oos"] = run(m_ft, patches, "forest-ft-oos")
    del m_ft; torch.cuda.empty_cache()

    with open(METRICS_DIR / "D2b_generalization_check.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved -> outputs/metrics/D2b_generalization_check.json")
    print("\n=== Summary (out-of-scene, non-adjacent forest AOI, NO retraining) ===")
    for k, v in results.items():
        g = v["global_pooled"]
        print(f"{k}: RMSE={g['rmse']:.2f}m corr={g['corr']:.3f}")
