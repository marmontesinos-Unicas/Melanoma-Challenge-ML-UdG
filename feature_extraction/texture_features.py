"""
Texture features for dermoscopic images (all take a uint8 gray image and an
optional mask, 255 = pixels to use; mask=None -> whole image).

    glcm_features      Haralick / GLCM: contrast, dissimilarity, homogeneity,
                       energy, correlation, ASM, entropy, per distance
                       (averaged over 4 angles -> rotation invariant)
    lbp_features       uniform rotation-invariant LBP histograms at 3 scales
    gabor_features     mean / std of the Gabor magnitude for 3 wavelengths x
                       4 orientations (pigment network, streaks, dots)
    hog_pooled         orientation histogram of the gradients over the whole
                       region (position independent) + gradient statistics
    hog_features       classic HOG on the image resized to 64x64 (position
                       dependent; whole image only)
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog


def _pix(x, mask):
    return x.ravel() if mask is None else x.ravel()[mask.ravel() > 0]


# ---------------------------------------------------------------------------
def glcm_features(gray: np.ndarray, mask: np.ndarray | None = None,
                  distances=(1, 2, 4), levels: int = 32) -> dict:
    """
    Gray-level co-occurrence matrix (Haralick, 1973). The image is quantised to
    `levels` gray levels; with a mask, pixels outside get level 0 and every pair
    that involves level 0 is discarded, so only lesion pixels count.
    7 properties x len(distances) features (21 by default).
    """
    q = (gray.astype(np.int32) * levels // 256 + 1).astype(np.uint8)        # 1..levels
    if mask is not None:
        q[mask == 0] = 0
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    P = graycomatrix(q, distances, angles, levels=levels + 1, symmetric=True, normed=False)
    P = P[1:, 1:].astype(np.float64)                                           # drop "outside"
    P /= P.sum(axis=(0, 1), keepdims=True) + 1e-12
    out = {}
    for prop in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"):
        v = np.nan_to_num(graycoprops(P, prop))                                # (n_dist, n_angles)
        for i, d in enumerate(distances):
            out[f"{prop}_d{d}"] = float(v[i].mean())
    ent = -(P * np.log2(P + 1e-12)).sum(axis=(0, 1))
    for i, d in enumerate(distances):
        out[f"entropy_d{d}"] = float(ent[i].mean())
    return out


# ---------------------------------------------------------------------------
def lbp_features(gray: np.ndarray, mask: np.ndarray | None = None,
                 scales=((8, 1), (16, 2), (24, 3))) -> dict:
    """
    Local Binary Patterns, 'uniform' (rotation invariant): P + 2 bins per scale,
    normalised histogram. Fine scale = micro texture (dots), larger = network.
    """
    out = {}
    for P, R in scales:
        codes = local_binary_pattern(gray, P, R, method="uniform")
        h = np.bincount(_pix(codes, mask).astype(np.int64), minlength=P + 2)[: P + 2].astype(np.float64)
        h /= h.sum() + 1e-12
        out |= {f"P{P}R{R}_bin{i}": float(v) for i, v in enumerate(h)}
    return out


# ---------------------------------------------------------------------------
def gabor_features(gray: np.ndarray, mask: np.ndarray | None = None,
                   wavelengths=(4, 8, 16), n_orient: int = 4) -> dict:
    """
    Gabor filter bank (OpenCV kernels, quadrature pair -> magnitude).
    wavelengths in pixels (for 256-px images: 4 = fine dots/network,
    16 = coarse structures). Features: mean and std of the magnitude per filter,
    plus, per wavelength, the anisotropy (max - min over orientations) / mean,
    which is high for oriented structures (streaks) and low for isotropic ones.
    """
    g = gray.astype(np.float32) / 255.0
    out = {}
    for lam in wavelengths:
        sigma = 0.56 * lam
        k = int(6 * sigma) | 1
        means = []
        for t in range(n_orient):
            theta = t * np.pi / n_orient
            re = cv2.filter2D(g, cv2.CV_32F, cv2.getGaborKernel((k, k), sigma, theta, lam, 0.5, 0))
            im = cv2.filter2D(g, cv2.CV_32F, cv2.getGaborKernel((k, k), sigma, theta, lam, 0.5, np.pi / 2))
            mag = _pix(np.sqrt(re ** 2 + im ** 2), mask)
            deg = int(round(np.degrees(theta)))
            out[f"l{lam}_o{deg}_mean"] = float(mag.mean())
            out[f"l{lam}_o{deg}_std"] = float(mag.std())
            means.append(mag.mean())
        means = np.array(means)
        out[f"l{lam}_anisotropy"] = float((means.max() - means.min()) / (means.mean() + 1e-12))
    return out


# ---------------------------------------------------------------------------
def hog_pooled(gray: np.ndarray, mask: np.ndarray | None = None, bins: int = 9) -> dict:
    """
    Magnitude-weighted histogram of gradient orientations (0-180 deg) over the
    whole region, i.e. HOG without the spatial grid (does not depend on where
    the lesion is). + mean / std of the gradient magnitude and orientation entropy.
    """
    g = gray.astype(np.float32)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    mag, ang = _pix(mag, mask), _pix(ang, mask) % 180
    h, _ = np.histogram(ang, bins=bins, range=(0, 180), weights=mag)
    h = h / (h.sum() + 1e-12)
    out = {f"bin{i}": float(v) for i, v in enumerate(h)}
    out["grad_mean"], out["grad_std"] = float(mag.mean()), float(mag.std())
    out["orient_entropy"] = float(-(h[h > 0] * np.log2(h[h > 0])).sum())
    return out


def hog_features(gray: np.ndarray, size: int = 64, ppc: int = 16, cpb: int = 2,
                 orientations: int = 9) -> dict:
    """Classic HOG (Dalal & Triggs) on the image resized to size x size (324 values by default)."""
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    v = hog(small, orientations=orientations, pixels_per_cell=(ppc, ppc),
            cells_per_block=(cpb, cpb), feature_vector=True)
    return {f"f{i}": float(x) for i, x in enumerate(v)}
