"""
Colour features for dermoscopic images.

    color_histogram     -> per-channel (marginal) histograms, concatenated
    color_histogram_3d  -> joint 3-D histogram (colour "cubes")
    color_moments       -> mean, std, skewness and entropy per channel

All take a uint8 BGR image (OpenCV) and an optional binary `mask`
(uint8, 255 = pixels to use, e.g. the lesion). With mask=None, the whole
image is used.
"""

from __future__ import annotations

import cv2
import numpy as np

# channel value ranges per colour space in OpenCV (uint8 images)
_RANGES = {
    "rgb": [(0, 256)] * 3,
    "hsv": [(0, 180), (0, 256), (0, 256)],   # OpenCV hue is 0-179
    "lab": [(0, 256)] * 3,
    "ycrcb": [(0, 256)] * 3,
    "gray": [(0, 256)],
}
_CHANNELS = {
    "rgb": ["B", "G", "R"],                  # OpenCV order
    "hsv": ["H", "S", "V"],
    "lab": ["L", "a", "b"],
    "ycrcb": ["Y", "Cr", "Cb"],
    "gray": ["I"],
}


def convert_color(img_bgr: np.ndarray, space: str) -> np.ndarray:
    space = space.lower()
    if space == "rgb":
        return img_bgr
    if space == "hsv":
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    if space == "lab":
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    if space == "ycrcb":
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    if space == "gray":
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)[..., None]
    raise ValueError(f"unknown colour space {space!r}")


def color_histogram(img_bgr: np.ndarray, space: str = "hsv", bins: int = 32,
                    mask: np.ndarray | None = None) -> np.ndarray:
    """
    Concatenated per-channel histograms, each normalised to sum 1.

    Returns a vector of length n_channels * bins (e.g. 3 * 32 = 96).
    Each bin groups (range / bins) intensity values, e.g. 256/32 = 8 values.
    """
    img = convert_color(img_bgr, space)
    feats = []
    for c, r in enumerate(_RANGES[space.lower()]):
        h = cv2.calcHist([img], [c], mask, [bins], list(r)).ravel()
        feats.append(h / (h.sum() + 1e-8))
    return np.concatenate(feats).astype(np.float32)


def color_histogram_3d(img_bgr: np.ndarray, space: str = "hsv", bins: int = 8,
                       mask: np.ndarray | None = None) -> np.ndarray:
    """
    Joint 3-D histogram (bins^3 values, e.g. 8^3 = 512), normalised to sum 1.
    Keeps which channel values occur *together* (true colours), unlike the
    marginal histograms.
    """
    img = convert_color(img_bgr, space)
    ranges = [v for r in _RANGES[space.lower()] for v in r]
    h = cv2.calcHist([img], [0, 1, 2], mask, [bins] * 3, ranges).ravel()
    return (h / (h.sum() + 1e-8)).astype(np.float32)


def color_moments(img_bgr: np.ndarray, space: str = "hsv",
                  mask: np.ndarray | None = None) -> np.ndarray:
    """
    Mean, standard deviation, skewness and entropy per channel
    (4 * n_channels values). Compact, but loses the shape of the distribution.
    Note: for HSV the hue is circular; the plain mean is an approximation.
    """
    img = convert_color(img_bgr, space).astype(np.float32)
    pix = img.reshape(-1, img.shape[2])
    if mask is not None:
        pix = pix[mask.ravel() > 0]
    feats = []
    for c, r in enumerate(_RANGES[space.lower()]):
        x = pix[:, c]
        mu, sd = x.mean(), x.std()
        skew = ((x - mu) ** 3).mean() / (sd ** 3 + 1e-8)
        p, _ = np.histogram(x, bins=32, range=r)
        p = p / (p.sum() + 1e-8)
        ent = -(p[p > 0] * np.log2(p[p > 0])).sum()
        feats += [mu, sd, skew, ent]
    return np.asarray(feats, dtype=np.float32)


def feature_names(kind: str, space: str = "hsv", bins: int = 32) -> list[str]:
    """Column names matching the vectors above (handy for DataFrames / importances)."""
    ch = _CHANNELS[space.lower()]
    if kind == "hist":
        return [f"{space}_{c}_bin{i}" for c in ch for i in range(bins)]
    if kind == "hist3d":
        return [f"{space}3d_{i}_{j}_{k}" for i in range(bins) for j in range(bins) for k in range(bins)]
    if kind == "moments":
        return [f"{space}_{c}_{m}" for c in ch for m in ("mean", "std", "skew", "entropy")]
    raise ValueError(kind)
