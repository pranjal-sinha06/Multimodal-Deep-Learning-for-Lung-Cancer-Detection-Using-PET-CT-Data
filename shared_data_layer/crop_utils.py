"""
crop_utils.py  --  the crop window, as COORDINATES.

stage2_dataset.crop_box() returns the cropped array. The CT+PET arm needs those coordinates: the SUV
channel must be sampled over exactly the same physical region as the CT
channels, or the two are spatially misaligned and the fusion test is invalid.

crop_window() replicates crop_box()'s logic and
returns (X1, Y1, X2, Y2). Both arms use it.
"""
import numpy as np
from stage2_dataset import MARGIN


def crop_window(shape, box, margin=MARGIN, jitter=0.0, rng=None):
    """Pixel window crop_box() would take.

    shape : (H, W) of the full slice
    box   : [x1, y1, x2, y2] in pixels
    """
    H, W = shape
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if jitter and rng is not None:
        # same four draws, same order as crop_box
        x1 += rng.uniform(-jitter, jitter) * bw
        x2 += rng.uniform(-jitter, jitter) * bw
        y1 += rng.uniform(-jitter, jitter) * bh
        y2 += rng.uniform(-jitter, jitter) * bh
    mx, my = bw * margin / 2, bh * margin / 2
    X1, Y1 = int(max(0, x1 - mx)), int(max(0, y1 - my))
    X2, Y2 = int(min(W, x2 + mx)), int(min(H, y2 + my))
    return X1, Y1, X2, Y2


def suv_for_window(win, ct_geom, pet, suv_clip=20.0):
    """SUV resampled onto the CT crop's pixel grid, in [0, 1].

    Every pixel of the CT crop is mapped to its physical mm position and then to
    the nearest PET voxel, so the returned array is pixel-for-pixel aligned with
    the CT crop. Pixels outside the PET field of view are zero.

    win      : (X1, Y1, X2, Y2) from crop_window
    ct_geom  : dict with x0, y0, z, dr, dc for the CT slice
    pet      : dict with zs (Z,), vol (Z,H,W), x0, y0, dr, dc
    """
    X1, Y1, X2, Y2 = win
    h, w = Y2 - Y1, X2 - X1
    if h <= 0 or w <= 0:
        return np.zeros((max(h, 1), max(w, 1)), np.float32)

    # nearest PET slice to this CT slice's z
    k = int(np.argmin(np.abs(pet["zs"] - ct_geom["z"])))
    if abs(pet["zs"][k] - ct_geom["z"]) > 10.0:      # no PET coverage here
        return np.zeros((h, w), np.float32)

    # CT crop pixel centres -> mm
    x_mm = ct_geom["x0"] + (X1 + np.arange(w)) * ct_geom["dc"]
    y_mm = ct_geom["y0"] + (Y1 + np.arange(h)) * ct_geom["dr"]

    # mm -> PET voxel index
    px = np.round((x_mm - pet["x0"]) / pet["dc"]).astype(np.int64)
    py = np.round((y_mm - pet["y0"]) / pet["dr"]).astype(np.int64)

    H, W = pet["vol"].shape[1], pet["vol"].shape[2]
    ok_x, ok_y = (px >= 0) & (px < W), (py >= 0) & (py < H)
    out = np.zeros((h, w), np.float32)
    if not ok_x.any() or not ok_y.any():
        return out

    sl = pet["vol"][k].astype(np.float32)
    patch = sl[np.ix_(np.clip(py, 0, H - 1), np.clip(px, 0, W - 1))]
    patch[~ok_y, :] = 0.0
    patch[:, ~ok_x] = 0.0
    return np.clip(patch, 0, suv_clip) / suv_clip     # [0,1], MINT's PET range
