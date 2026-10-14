"""
Extract ALL features of an image at once (colour + texture, no shape), with the
group name as column prefix ("<group>__<feature>"), so the models can later use
any subset of groups:

    feats = extract_all(img)                      # whole image (no segmentation)
    feats = extract_all(img, mask=lesion_mask)    # + the same inside the lesion (les_*),
                                                  #   skin colour and lesion-vs-skin contrast
    X = select_groups(df, ["hist_hsv", "moments_hsv"])
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from .color_features import (color_histogram, color_histogram_3d, color_moments,
                             feature_names, dermoscopic_colors, color_clusters,
                             _lab_float, _region_stats)
from .texture_features import (glcm_features, lbp_features, gabor_features,
                               hog_pooled, hog_features)

# group -> description. Groups marked (mask) need the lesion segmentation.
GROUPS = {
    # colour
    "hist_rgb":       "per-channel histogram RGB (3 x bins)",
    "hist_hsv":       "per-channel histogram HSV (3 x bins)",
    "hist_lab":       "per-channel histogram Lab (3 x bins)",
    "hist3d_hsv":     "joint 3-D histogram HSV (bins3d^3)",
    "hist3d_lab":     "joint 3-D histogram Lab (bins3d^3)",
    "moments_rgb":    "mean, std, skewness, entropy per channel RGB (12)",
    "moments_hsv":    "mean, std, skewness, entropy per channel HSV (12)",
    "moments_lab":    "mean, std, skewness, entropy per channel Lab (12)",
    "derm_colors":    "6 dermoscopic colours + number of colours (8)",
    "variegation":    "colour spread, Delta E to the mean colour (2)",
    "color_clusters": "k-means: number of distinct colours, cluster sizes, Delta E (k + 3)",
    # texture (on the gray image)
    "glcm":           "GLCM / Haralick, 7 properties x 3 distances (21)",
    "lbp":            "uniform LBP histograms, 3 scales (54)",
    "gabor":          "Gabor bank, 3 wavelengths x 4 orientations, mean/std + anisotropy (27)",
    "hog_pooled":     "orientation histogram of gradients, position independent (12)",
    "hog":            "classic HOG on 64x64 (324, whole image only)",
    # (mask) -> computed only when a lesion mask is given
    "skin_stats":     "(mask) colour of the surrounding skin (16)",
    "contrast":       "(mask) lesion vs skin dL, da, db, dE, RGB ratios + border vs core (8)",
}
# with a mask, every colour / texture group except 'hog' is also computed inside the
# lesion with the prefix 'les_' (e.g. 'les_hist_hsv', 'les_glcm').


def _group(prefix: str, d: dict) -> dict:
    return {f"{prefix}__{k}": v for k, v in d.items()}


def _colour_texture(img, mask, bins, bins3d, k_colors, pre=""):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = {}
    for space in ("rgb", "hsv", "lab"):
        out |= _group(f"{pre}hist_{space}", dict(zip(feature_names("hist", space, bins),
                                                     color_histogram(img, space, bins, mask).tolist())))
        out |= _group(f"{pre}moments_{space}", dict(zip(feature_names("moments", space),
                                                        color_moments(img, space, mask).tolist())))
    for space in ("hsv", "lab"):
        out |= _group(f"{pre}hist3d_{space}", dict(zip(feature_names("hist3d", space, bins3d),
                                                       color_histogram_3d(img, space, bins3d, mask).tolist())))
    out |= _group(f"{pre}derm_colors", dermoscopic_colors(img, mask))
    lab = _lab_float(img).reshape(-1, 3)
    if mask is not None:
        lab = lab[mask.ravel() > 0]
    de = np.linalg.norm(lab - lab.mean(0), axis=1)
    out |= _group(f"{pre}variegation", {"dE_mean": float(de.mean()), "dE_p90": float(np.percentile(de, 90))})
    out |= _group(f"{pre}color_clusters", color_clusters(img, k_colors, mask))
    out |= _group(f"{pre}glcm", glcm_features(gray, mask))
    out |= _group(f"{pre}lbp", lbp_features(gray, mask))
    out |= _group(f"{pre}gabor", gabor_features(gray, mask))
    out |= _group(f"{pre}hog_pooled", hog_pooled(gray, mask))
    return out


def extract_all(img: np.ndarray, mask: np.ndarray | None = None, bins: int = 32,
                bins3d: int = 8, k_colors: int = 5) -> dict:
    """All colour + texture features of one (preprocessed, BGR uint8) image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = _colour_texture(img, None, bins, bins3d, k_colors)               # whole image
    out |= _group("hog", hog_features(gray))
    if mask is not None and np.any(mask):
        from preprocessing.segmentation import region_masks
        reg = region_masks(mask)
        les = reg["lesion"]
        skin = reg["skin"] if np.any(reg["skin"]) else 255 - les
        out |= _colour_texture(img, les, bins, bins3d, k_colors, pre="les_")
        out |= _group("skin_stats", _region_stats(img, skin, "skin"))
        lab = _lab_float(img).reshape(-1, 3)
        mean = lambda m: lab[m.ravel() > 0].mean(0) if np.any(m) else np.zeros(3, np.float32)
        d = mean(les) - mean(skin)
        bgr = img.reshape(-1, 3).astype(np.float32)
        ml, ms = bgr[les.ravel() > 0].mean(0), bgr[skin.ravel() > 0].mean(0)
        out |= _group("contrast", {"dL": float(d[0]), "da": float(d[1]), "db": float(d[2]),
                                   "dE_skin": float(np.linalg.norm(d)),
                                   "ratio_B": float(ml[0] / (ms[0] + 1e-6)),
                                   "ratio_G": float(ml[1] / (ms[1] + 1e-6)),
                                   "ratio_R": float(ml[2] / (ms[2] + 1e-6)),
                                   "dE_border_inner": float(np.linalg.norm(mean(reg["border"]) - mean(reg["inner"])))})
    return out


def extract_all_from_path(path, mask_fn=None, **kw) -> dict:
    """Read an image and extract everything. mask_fn(img) -> lesion mask (or None = no mask)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"could not read {path}")
    return extract_all(img, mask_fn(img) if mask_fn else None, **kw)


def feature_groups(columns) -> list[str]:
    """Groups present in a feature table (column prefixes)."""
    return sorted({c.split("__", 1)[0] for c in columns if "__" in c})


def select_groups(df: pd.DataFrame, groups) -> pd.DataFrame:
    """Columns of the requested groups, e.g. ["hist_hsv", "glcm"]."""
    groups = set(groups)
    missing = groups - set(feature_groups(df.columns))
    if missing:
        raise ValueError(f"groups not in the table: {missing}. Available: {feature_groups(df.columns)}")
    return df[[c for c in df.columns if c.split("__", 1)[0] in groups]]
