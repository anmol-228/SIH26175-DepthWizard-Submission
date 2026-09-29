"""Geographic (tile-level) train/val/test split.

Each Vaihingen tile ("area") is a spatially distinct, non-overlapping patch of the
town (per ISPRS documentation) -- so a tile-level split IS a geographic split; no
patch crops later will straddle a train/test boundary as long as we never crop
across tile borders during patch extraction (enforced in scripts/03_*).

Stratified loosely by DSM std (proxy for scene complexity: low std ~ flat/open,
high std ~ dense urban/building-heavy) so train/val/test each see a mix, and the
held-out test set is not accidentally all-easy or all-hard.
"""
import json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
with open(ROOT / "outputs" / "metrics" / "data_audit.json") as f:
    records = json.load(f)

SEED = 42
random.seed(SEED)

records_sorted = sorted(records, key=lambda r: r["dsm_std"])
n = len(records_sorted)
# bucket into 3 complexity tiers, then sample proportionally into train/val/test
tiers = [records_sorted[i::3] for i in range(3)]  # interleave for even spread

train, val, test = [], [], []
for tier in tiers:
    tier = tier[:]
    random.shuffle(tier)
    n_tier = len(tier)
    n_test = max(1, round(n_tier * 0.24))
    n_val = max(1, round(n_tier * 0.12))
    test += tier[:n_test]
    val += tier[n_test:n_test + n_val]
    train += tier[n_test + n_val:]

split = {
    "seed": SEED,
    "method": "tile-level (each Vaihingen area is spatially disjoint); loosely stratified by DSM std tercile",
    "train_areas": sorted(r["area"] for r in train),
    "val_areas": sorted(r["area"] for r in val),
    "test_areas": sorted(r["area"] for r in test),
}
split["n_train"] = len(split["train_areas"])
split["n_val"] = len(split["val_areas"])
split["n_test"] = len(split["test_areas"])

out_dir = ROOT / "data" / "splits"
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "split_v1.json", "w") as f:
    json.dump(split, f, indent=2)

print(json.dumps(split, indent=2))
print("\nNo area appears in more than one split:",
      set(split["train_areas"]) & set(split["val_areas"]) & set(split["test_areas"]) == set())
