"""Experiment E, step 0: acquire real hilly-terrain RGB + elevation data.

Region: Coyote Wall / Catherine Creek, Columbia River Gorge, Washington state,
USA. Chosen because it has genuine strong local relief (Columbia River at
~30m up to basalt-rim ridgelines at ~800-900m within a few km), is mostly
open grassland/oak-savanna rather than closed-canopy forest (so this
experiment is not confounded with the separate vegetation/canopy experiment,
Exp D), and both source programs below have full free coverage there with
NO account/registration required.

Data sources (both verified no-auth this session, see 07_HILLY_TERRAIN_TEST.md
for full URLs/licenses):
  1. USGS 3DEP 1-meter DEM (lidar-derived), public-domain, via the TNM Access
     API (https://tnmaccess.nationalmap.gov/api/v1/products) which resolves to
     a direct HTTPS download on the public (no-auth) prd-tnm S3 bucket.
     Tile: USGS_1M_10_x62y507_WA_FEMAHQ_2018_D18.tif
     Native bbox (WGS84): lon -121.459..-121.328, lat 45.681..45.773
  2. NAIP 2023 RGB(+NIR) orthophoto, 0.6m GSD, public domain (USDA FSA), via
     the Microsoft Planetary Computer STAC API (no account needed for search
     or for the SAS-token signing endpoint -- both are anonymous/free reads).
     Tile: m_4512121_nw_10_060_20230714_20230912.tif
     Native bbox (WGS84): lon -121.504..-121.433, lat 45.685..45.753

The two footprints overlap in a lon -121.459..-121.433, lat 45.685..45.753
strip (~2.0km x ~7.5km) -- that overlap is what gets cropped to in step E_01.
"""
import sys, json, hashlib
from pathlib import Path
import requests

P2ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = P2ROOT / "data" / "hilly_terrain" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

DEM_URL = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1m/Projects/WA_FEMAHQ_2018_D18/TIFF/USGS_1M_10_x62y507_WA_FEMAHQ_2018_D18.tif"
NAIP_RAW_URL = "https://naipeuwest.blob.core.windows.net/naip/v002/wa/2023/wa_060cm_2023/45121/m_4512121_nw_10_060_20230714_20230912.tif"

DEM_OUT = RAW_DIR / "USGS_1M_x62y507_WA_FEMAHQ_2018.tif"
NAIP_OUT = RAW_DIR / "naip_wa_4512121_nw_2023.tif"


def sign_naip_url(url: str) -> str:
    r = requests.get("https://planetarycomputer.microsoft.com/api/sas/v1/sign", params={"href": url}, timeout=30)
    r.raise_for_status()
    return r.json()["href"]


def download(url: str, out_path: Path, chunk=1 << 20):
    if out_path.exists():
        print(f"  already have {out_path.name} ({out_path.stat().st_size:,} bytes) -- skipping")
        return
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        got = 0
        with open(out_path, "wb") as f:
            for c in r.iter_content(chunk_size=chunk):
                f.write(c)
                got += len(c)
        print(f"  downloaded {out_path.name}: {got:,} bytes (expected {total:,})")
        if total and got != total:
            raise RuntimeError(f"size mismatch for {out_path.name}: got {got}, expected {total}")


def sha256_head(path: Path, nbytes=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(nbytes))
    return h.hexdigest()[:16]


if __name__ == "__main__":
    print("Fetching USGS 3DEP 1m DEM tile (public-domain, no-auth S3)...")
    download(DEM_URL, DEM_OUT)

    print("Signing + fetching NAIP 2023 RGB tile (public-domain, Planetary Computer no-auth SAS)...")
    signed = sign_naip_url(NAIP_RAW_URL)
    download(signed, NAIP_OUT)

    manifest = {
        "dem": {
            "source_url": DEM_URL,
            "local_path": str(DEM_OUT.relative_to(P2ROOT)),
            "size_bytes": DEM_OUT.stat().st_size,
            "sha256_head_1MB": sha256_head(DEM_OUT),
            "license": "Public domain (USGS 3DEP)",
        },
        "naip": {
            "source_url_unsigned": NAIP_RAW_URL,
            "local_path": str(NAIP_OUT.relative_to(P2ROOT)),
            "size_bytes": NAIP_OUT.stat().st_size,
            "sha256_head_1MB": sha256_head(NAIP_OUT),
            "license": "Public domain (USDA FSA NAIP)",
        },
    }
    with open(RAW_DIR / "fetch_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))
    print("Done.")
