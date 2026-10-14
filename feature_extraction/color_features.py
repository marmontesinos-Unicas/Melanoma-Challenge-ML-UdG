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


# ---------------------------------------------------------------------------
# Region-based colour descriptors (lesion / border / surrounding skin)
# ---------------------------------------------------------------------------
# The six dermoscopic colours of the ABCD rule / 7-point checklist, as RGB
# references (Kasmi & Mokrani, 2016; also used by Seidenari et al.).
# Blue-gray and black are strongly associated with melanoma.
DERM_COLORS_RGB = {
    "white":       (197, 188, 217),
    "red":         (118, 21, 17),
    "light_brown": (163, 82, 16),
    "dark_brown":  (135, 44, 5),
    "blue_gray":   (0, 134, 139),
    "black":       (41, 31, 30),
}


def _lab_float(img_bgr: np.ndarray) -> np.ndarray:
    """CIELab with real units (L 0-100, a/b approx. -128..127)."""
    return cv2.cvtColor(img_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)


_DERM_LAB = _lab_float(np.array([[c[::-1] for c in DERM_COLORS_RGB.values()]], np.uint8))[0]


def dermoscopic_colors(img_bgr: np.ndarray, mask: np.ndarray | None = None,
                       max_dist: float = 40.0, min_frac: float = 0.05) -> dict:
    """
    Fraction of the (lesion) pixels whose closest dermoscopic colour (Delta E in
    Lab) is each of the 6 references; pixels farther than `max_dist` from all of
    them count as 'other'. n_colors = how many colours cover >= `min_frac`.
    """
    lab = _lab_float(img_bgr).reshape(-1, 3)
    if mask is not None:
        lab = lab[mask.ravel() > 0]
    out = {f"dc_{k}": 0.0 for k in DERM_COLORS_RGB} | {"dc_other": 0.0, "dc_n_colors": 0.0}
    if len(lab) == 0:
        return out
    d = np.linalg.norm(lab[:, None, :] - _DERM_LAB[None], axis=2)
    nearest, dmin = d.argmin(1), d.min(1)
    for i, k in enumerate(DERM_COLORS_RGB):
        out[f"dc_{k}"] = float(np.mean((nearest == i) & (dmin <= max_dist)))
    out["dc_other"] = float(np.mean(dmin > max_dist))
    out["dc_n_colors"] = float(sum(out[f"dc_{k}"] >= min_frac for k in DERM_COLORS_RGB))
    return out


def _region_stats(img_bgr: np.ndarray, mask: np.ndarray, prefix: str) -> dict:
    """Mean / std in Lab, RGB and HSV (hue as circular mean) of one region."""
    m = mask.ravel() > 0
    out = {}
    lab = _lab_float(img_bgr).reshape(-1, 3)[m]
    bgr = img_bgr.reshape(-1, 3)[m].astype(np.float32)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3)[m].astype(np.float32)
    if m.sum() == 0:
        lab = bgr = hsv = np.zeros((1, 3), np.float32)
    for name, arr, chans in (("lab", lab, "Lab"), ("rgb", bgr, "BGR")):
        for c, ch in enumerate(chans):
            out[f"{prefix}_{name}_{ch}_mean"] = float(arr[:, c].mean())
            out[f"{prefix}_{name}_{ch}_std"] = float(arr[:, c].std())
    h = hsv[:, 0] * (2 * np.pi / 180.0)
    out[f"{prefix}_hsv_H_cos"] = float(np.cos(h).mean())
    out[f"{prefix}_hsv_H_sin"] = float(np.sin(h).mean())
    for c, ch in ((1, "S"), (2, "V")):
        out[f"{prefix}_hsv_{ch}_mean"] = float(hsv[:, c].mean())
        out[f"{prefix}_hsv_{ch}_std"] = float(hsv[:, c].std())
    return out


def color_features(img_bgr: np.ndarray, regions: dict | None = None,
                   hist_bins: int = 16) -> dict:
    """
    All colour features of one (preprocessed) image as a {name: value} dict.

    regions: output of segmentation.region_masks(mask) with keys 'lesion',
             'inner', 'border', 'skin'. If None, the whole image is the lesion
             and the contrast features are 0.

    Groups (prefix):
      les_*, skin_*   mean / std of Lab, RGB, HSV in the lesion and in the skin
      mom_*           skewness and entropy (color_moments) of the lesion, Lab + HSV
      hist_*          lesion histograms, HSV and Lab, `hist_bins` per channel
      dc_*            the 6 dermoscopic colours (fractions) + number of colours
      var_*           colour variegation inside the lesion (Delta E spread)
      con_*           contrast lesion vs skin and border ring vs lesion core
    """
    h, w = img_bgr.shape[:2]
    full = np.full((h, w), 255, np.uint8)
    les = regions["lesion"] if regions else full
    skin = regions["skin"] if regions and regions["skin"].any() else None

    f = {}
    f |= _region_stats(img_bgr, les, "les")
    f |= _region_stats(img_bgr, skin if skin is not None else full, "skin")

    for space in ("lab", "hsv"):
        mom = color_moments(img_bgr, space, les)
        for name, v in zip(feature_names("moments", space), mom):
            if name.endswith(("skew", "entropy")):
                f[f"mom_{name}"] = float(v)
        hist = color_histogram(img_bgr, space, hist_bins, les)
        for name, v in zip(feature_names("hist", space, hist_bins), hist):
            f[f"hist_{name}"] = float(v)

    f |= dermoscopic_colors(img_bgr, les)

    lab = _lab_float(img_bgr).reshape(-1, 3)
    lp = lab[les.ravel() > 0]
    if len(lp):
        de = np.linalg.norm(lp - lp.mean(0), axis=1)
        f["var_dE_mean"], f["var_dE_p90"] = float(de.mean()), float(np.percentile(de, 90))
    else:
        f["var_dE_mean"] = f["var_dE_p90"] = 0.0

    def mean_lab(m):
        sel = lab[m.ravel() > 0]
        return sel.mean(0) if len(sel) else np.zeros(3, np.float32)

    if regions:
        dl = mean_lab(les) - (mean_lab(skin) if skin is not None else mean_lab(les))
        f["con_dL"], f["con_da"], f["con_db"] = map(float, dl)
        f["con_dE_skin"] = float(np.linalg.norm(dl))
        f["con_dE_border_inner"] = float(np.linalg.norm(mean_lab(regions["border"]) - mean_lab(regions["inner"])))
        for c, ch in enumerate("BGR"):
            s = f[f"skin_rgb_{ch}_mean"]
            f[f"con_ratio_{ch}"] = float(f[f"les_rgb_{ch}_mean"] / (s + 1e-6))
    else:
        for k in ("con_dL", "con_da", "con_db", "con_dE_skin", "con_dE_border_inner",
                  "con_ratio_B", "con_ratio_G", "con_ratio_R"):
            f[k] = 0.0
    return f


# ---------------------------------------------------------------------------
# Number of colours with k-means
# ---------------------------------------------------------------------------
def color_clusters(img_bgr: np.ndarray, k: int = 5, mask: np.ndarray | None = None,
                   min_frac: float = 0.05, merge_dE: float = 10.0,
                   n_sample: int = 5000, seed: int = 0) -> dict:
    """
    k-means with `k` clusters on the Lab colour of the pixels (a random sample of
    `n_sample` pixels for speed).

    Returns:
      clu_n_colors      number of DISTINCT colours: clusters with >= `min_frac` of
                        the pixels, merging those whose centres are closer than
                        `merge_dE` (Delta E) because they are the same colour
      clu_frac_1..k     cluster sizes, largest first (how dominant the main colour is)
      clu_dE_mean/max   mean / max Delta E between the centres of the significant
                        clusters (how different the colours are)
    """
    lab = _lab_float(img_bgr).reshape(-1, 3)
    if mask is not None:
        lab = lab[mask.ravel() > 0]
    out = {"clu_n_colors": 0.0} | {f"clu_frac_{i + 1}": 0.0 for i in range(k)} \
        | {"clu_dE_mean": 0.0, "clu_dE_max": 0.0}
    if len(lab) < k:
        return out
    rng = np.random.default_rng(seed)
    x = lab[rng.choice(len(lab), min(n_sample, len(lab)), replace=False)].astype(np.float32)
    cv2.setRNGSeed(seed)
    _, labels, centers = cv2.kmeans(x, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5),
                                    3, cv2.KMEANS_PP_CENTERS)
    frac = np.bincount(labels.ravel(), minlength=k) / len(x)
    order = np.argsort(-frac)
    for i, j in enumerate(order):
        out[f"clu_frac_{i + 1}"] = float(frac[j])

    sig = [j for j in order if frac[j] >= min_frac]
    # merge significant clusters that are the same colour (Delta E < merge_dE)
    groups = []
    for j in sig:
        for g in groups:
            if np.linalg.norm(centers[j] - centers[g[0]]) < merge_dE:
                g.append(j); break
        else:
            groups.append([j])
    out["clu_n_colors"] = float(len(groups))
    if len(sig) > 1:
        c = centers[sig]
        d = np.linalg.norm(c[:, None] - c[None], axis=2)[np.triu_indices(len(sig), 1)]
        out["clu_dE_mean"], out["clu_dE_max"] = float(d.mean()), float(d.max())
    return out
