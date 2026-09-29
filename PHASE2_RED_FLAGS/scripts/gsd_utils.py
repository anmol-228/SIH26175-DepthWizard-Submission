"""GSD degradation simulation for Experiment A (controlled resolution stress test).

Downsamples the INPUT RGB only (never the DSM ground truth) to simulate a
coarser ground-sample-distance, using a BOX filter (area-average) — this is
the closest cheap approximation to "a satellite pixel integrates a larger
ground footprint" without inventing a fake sensor MTF. Then upsamples back to
the original pixel grid via BICUBIC only because the network needs a fixed
input shape — no high-frequency information is allowed to leak back in by
this step (BOX-then-BICUBIC strictly cannot recover detail lost by BOX).

This tests information-loss sensitivity only. It does NOT simulate atmospheric
effects, sensor MTF, radiometric response, compression, or viewing geometry —
see 04_GSD_STRESS_TEST.md Section "Scientific Caveat" for the explicit
disclosure required by the master brief (Section 11).
"""
import numpy as np
from PIL import Image

NATIVE_GSD = 0.09  # Vaihingen, meters/pixel


def degrade_rgb_to_gsd(rgb_uint8: np.ndarray, target_gsd: float, native_gsd: float = NATIVE_GSD) -> np.ndarray:
    """rgb_uint8: (H,W,3) uint8 at native_gsd -> same shape, simulated target_gsd."""
    if target_gsd <= native_gsd:
        return rgb_uint8
    h, w = rgb_uint8.shape[:2]
    factor = target_gsd / native_gsd
    small_h, small_w = max(1, round(h / factor)), max(1, round(w / factor))
    img = Image.fromarray(rgb_uint8)
    small = img.resize((small_w, small_h), Image.Resampling.BOX)
    back = small.resize((w, h), Image.Resampling.BICUBIC)
    return np.asarray(back, dtype=np.uint8)
