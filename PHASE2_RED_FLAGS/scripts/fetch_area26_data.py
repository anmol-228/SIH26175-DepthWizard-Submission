"""Fetch just area26's top+DSM Vaihingen tiles, needed by K3_mesh_pipeline.py,
without downloading the full 879MB tiled-benchmark zip or the 16GB raw release.

Source verified in 02_DATA_AUDIT.md / DATASETS.md: mominh/Vaihingen-compressed on
Hugging Face. The 879MB ISPRS_semantic_labeling_Vaihingen.zip is stored UNCOMPRESSED
inside the outer 16GB Vaihingen.zip, so its byte range can be read directly and
opened as its own zip -- letting us pull two ~16-21MB members instead of 879MB.
"""
import struct
import zipfile
from pathlib import Path

from remotezip import RemoteZip, RemoteIO, RemoteFetcher, PartialBuffer

OUTER_URL = "https://huggingface.co/datasets/mominh/Vaihingen-compressed/resolve/main/Vaihingen.zip"
INNER_NAME = "Vaihingen/ISPRS_semantic_labeling_Vaihingen.zip"
AREA = 26

ROOT = Path(__file__).resolve().parents[2]  # repo root
TOP_DIR = ROOT / "data" / "raw" / "Vaihingen" / "top"
DSM_DIR = ROOT / "data" / "raw" / "Vaihingen" / "dsm"


def open_inner_zip() -> zipfile.ZipFile:
    with RemoteZip(OUTER_URL) as outer:
        info = outer.getinfo(INNER_NAME)
        lfh = outer.fp._fetch_fun((info.header_offset, info.header_offset + 29), stream=False).read(30)
        fname_len, extra_len = struct.unpack("<HH", lfh[26:30])
        inner_start = info.header_offset + 30 + fname_len + extra_len
        inner_size = info.file_size  # stored (compress_type=0) -> compressed size == file size

    outer_fetcher = RemoteFetcher(OUTER_URL)

    def inner_fetch(data_range, stream=False):
        lo, hi = data_range
        if lo < 0:
            abs_lo, abs_hi = inner_start + inner_size + lo, inner_start + inner_size - 1
        else:
            abs_hi = inner_start + (hi if hi is not None else inner_size - 1)
            abs_lo = inner_start + lo
        res, content_range = outer_fetcher._request(outer_fetcher.prepare_request((abs_lo, abs_hi)))
        range_min_abs, range_max_abs = outer_fetcher.parse_range_header(content_range)
        size = range_max_abs - range_min_abs + 1
        return PartialBuffer(res, range_min_abs - inner_start, size, stream)

    rio = RemoteIO(inner_fetch, initial_buffer_size=64 * 1024)
    return zipfile.ZipFile(rio)


def main():
    TOP_DIR.mkdir(parents=True, exist_ok=True)
    DSM_DIR.mkdir(parents=True, exist_ok=True)

    inner = open_inner_zip()
    plan = [
        (f"top/top_mosaic_09cm_area{AREA}.tif", TOP_DIR / f"top_mosaic_09cm_area{AREA}.tif"),
        (f"dsm/dsm_09cm_matching_area{AREA}.tif", DSM_DIR / f"dsm_09cm_matching_area{AREA}.tif"),
    ]
    for member, dest in plan:
        info = inner.getinfo(member)
        print(f"Fetching {member} ({info.file_size:,} bytes uncompressed) -> {dest}")
        with inner.open(member) as src, open(dest, "wb") as out:
            out.write(src.read())
        print(f"  wrote {dest} ({dest.stat().st_size:,} bytes)")

    print("\nDone. Area 26 top+DSM tiles are in place for K3_mesh_pipeline.py.")


if __name__ == "__main__":
    main()
