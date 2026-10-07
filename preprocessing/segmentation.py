"""
Lesion segmentation: MASK branch (classical, no deep learning).

Input: the output of `preprocessing.preprocess()` (resized, hair removed,
colour-corrected). The heavy smoothing done here is ONLY used to obtain the
mask; features are always computed on the non-smoothed image.

    smooth_for_segmentation -> median / Gaussian / bilateral smoothing
    flatten_illumination    -> remove vignetting / uneven light (polynomial surface)
    get_channel             -> single channel with good lesion/skin contrast
    otsu_mask               -> Otsu threshold on one channel
    kmeans_mask             -> k-means on Lab pixels, darkest cluster(s) = lesion
    postprocess_mask        -> morphology, fill holes, keep the central component
    segment_lesion          -> full pipeline + sanity check + fallback
    region_masks            -> lesion / border ring / surrounding skin (spatial context)
    overlay_mask            -> draw the mask contour for visual checking

Masks are uint8 with values {0, 255} (OpenCV convention).
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi


# ---------------------------------------------------------------------------
# 1. Smoothing (segmentation only!)
# ---------------------------------------------------------------------------
def smooth_for_segmentation(img: np.ndarray, method: str = "median", ksize: int = 11) -> np.ndarray:
    """
    Strong smoothing so that the threshold / clustering reacts to the lesion as a
    whole and not to its internal texture (network, dots, globules).
    'median' and 'bilateral' keep the lesion border sharper than 'gaussian'.
    ksize is in pixels (for 256-px images, 7-15 works well).
    """
    if method is None or method == "none":
        return img
    if method == "median":
        return cv2.medianBlur(img, ksize if ksize % 2 else ksize + 1)
    if method == "gaussian":
        k = ksize if ksize % 2 else ksize + 1
        return cv2.GaussianBlur(img, (k, k), 0)
    if method == "bilateral":
        return cv2.bilateralFilter(img, d=ksize, sigmaColor=75, sigmaSpace=75)
    raise ValueError(f"unknown smoothing {method!r}")


def flatten_illumination(ch: np.ndarray, degree: int = 2, keep_percentile: float = 40,
                         n_iter: int = 2, n_sample: int = 8000, seed: int = 0) -> np.ndarray:
    """
    Remove slow illumination changes (vignetting: darker corners, uneven light).

    A 2-D polynomial surface (degree 2) is fitted by least squares to the
    BRIGHTER pixels (above `keep_percentile`, i.e. mostly skin, not lesion),
    refitted `n_iter` times, and the channel is divided by it. After this the
    skin is roughly constant, so a global threshold separates the lesion
    instead of "dark corners + lesion" vs "bright centre".
    """
    h, w = ch.shape
    yy, xx = np.mgrid[0:h, 0:w]
    x = (xx.ravel() / w - 0.5).astype(np.float32)
    y = (yy.ravel() / h - 0.5).astype(np.float32)
    terms = [np.ones_like(x)] + [x ** i * y ** j for d in range(1, degree + 1) for i in range(d + 1) for j in [d - i]]
    A = np.stack(terms, 1)
    v = ch.ravel().astype(np.float32)
    rng = np.random.default_rng(seed)
    use = v > np.percentile(v, keep_percentile)
    for _ in range(n_iter):
        idx = np.flatnonzero(use)
        idx = rng.choice(idx, min(n_sample, len(idx)), replace=False)
        coef, *_ = np.linalg.lstsq(A[idx], v[idx], rcond=None)
        surf = A @ coef
        use = v > np.percentile(v / np.maximum(surf, 1), keep_percentile) * surf
    surf = np.maximum(surf.reshape(h, w), 1)
    out = ch.astype(np.float32) / surf
    out = out / (np.percentile(out, 99) + 1e-8) * 230
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 2. Channel selection
# ---------------------------------------------------------------------------
def get_channel(img: np.ndarray, channel: str = "blue") -> np.ndarray:
    """
    Single uint8 channel where the lesion is DARKER than the skin.

    'blue' : B channel, usually the best lesion/skin contrast in dermoscopy
    'L'    : lightness L* of Lab
    'gray' : standard grayscale
    'pca'  : 1st principal component of the RGB pixels (max-variance direction),
             oriented so the lesion is dark
    """
    if channel == "blue":
        return img[..., 0].copy()
    if channel == "L":
        return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0]
    if channel == "gray":
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if channel == "pca":
        x = img.reshape(-1, 3).astype(np.float32)
        x -= x.mean(0)
        _, _, vt = np.linalg.svd(x[:: max(1, len(x) // 20000)], full_matrices=False)
        pc = x @ vt[0]
        if np.corrcoef(pc, x.sum(1))[0, 1] < 0:   # make "bright" = high
            pc = -pc
        pc = (pc - pc.min()) / (pc.max() - pc.min() + 1e-8) * 255
        return pc.reshape(img.shape[:2]).astype(np.uint8)
    raise ValueError(f"unknown channel {channel!r}")


# ---------------------------------------------------------------------------
# 3. Raw masks
# ---------------------------------------------------------------------------
def otsu_mask(img: np.ndarray, channel: str = "blue", flatten: bool = True) -> tuple[np.ndarray, float]:
    """Otsu threshold on one channel; pixels darker than the threshold = lesion."""
    ch = get_channel(img, channel)
    if flatten:
        ch = flatten_illumination(ch)
    t, _ = cv2.threshold(ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = ((ch <= t) * 255).astype(np.uint8)
    return mask, float(t)


def kmeans_mask(img: np.ndarray, k: int = 2, space: str = "lab", lesion: str = "darkest",
                flatten: bool = True, n_sample: int = 20000, seed: int = 0) -> np.ndarray:
    """
    k-means on the colour of the pixels, clusters ordered by lightness.

    lesion='darkest'           -> only the darkest cluster is the lesion
                                  (k=3: skin / shadowed skin / lesion, robust to vignetting)
    lesion='all_but_brightest' -> every cluster except the brightest (skin) is the lesion
                                  (k=3: keeps light + dark parts of multi-colour lesions,
                                  but also picks up darker skin in the corners)
    flatten=True corrects vignetting on L* first (see flatten_illumination).
    Fitted on a random subsample of pixels for speed, then all pixels are assigned.
    """
    if space == "lab":
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        if flatten:
            lab[..., 0] = flatten_illumination(lab[..., 0])
        feat = lab.reshape(-1, 3).astype(np.float32)
        light = feat[:, 0]
    else:  # bgr
        feat = img.reshape(-1, 3).astype(np.float32)
        light = feat.sum(1)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(feat), min(n_sample, len(feat)), replace=False)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    cv2.setRNGSeed(seed)
    _, _, centers = cv2.kmeans(feat[idx], k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    labels = np.argmin(((feat[:, None, :] - centers[None]) ** 2).sum(-1), axis=1)
    cluster_light = np.array([light[labels == c].mean() if np.any(labels == c) else np.inf
                              for c in range(k)])
    if lesion == "darkest":
        mask = (labels == int(np.argmin(cluster_light))).reshape(img.shape[:2])
    elif lesion == "all_but_brightest":
        mask = (labels != int(np.argmax(cluster_light))).reshape(img.shape[:2])
    else:
        raise ValueError(f"unknown lesion rule {lesion!r}")
    return (mask * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 4. Post-processing
# ---------------------------------------------------------------------------
def _disk(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def postprocess_mask(mask: np.ndarray, open_r: int = 3, close_r: int = 7,
                     center_sigma: float = 0.5, min_frac: float = 0.02,
                     max_frac: float = 0.80) -> np.ndarray:
    """
    Clean the raw mask:
      1. opening (removes small noise) and closing (joins small gaps)
      2. keep ONE connected component, the one with the best score
             area * exp(-d^2 / (2 sigma^2))
         where d = MEAN distance of the component's pixels to the image centre
         (normalised by the half-diagonal). Lesions are usually centred;
         dark corners from vignetting, shadows or skin folds are not.
         The mean pixel distance is used instead of the centroid because a
         ring of dark corners has its centroid exactly in the centre.
         Components touching the border are NOT discarded, because large
         lesions often fill the image.
      3. fill the holes of that component only (filling before selecting
         would turn a ring of dark corners into a full-image mask)
      4. if its filled area is implausible (< min_frac or > max_frac), try the
         next best components (up to 5)
    """
    h, w = mask.shape
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _disk(open_r))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _disk(close_r))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return m
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.hypot(xx - w / 2, yy - h / 2) / (np.hypot(h, w) / 2)
    sum_d = np.bincount(labels.ravel(), weights=dist.ravel(), minlength=n)
    area = stats[:, cv2.CC_STAT_AREA].astype(float)
    mean_d = sum_d / np.maximum(area, 1)
    score = area * np.exp(-mean_d ** 2 / (2 * center_sigma ** 2))
    score[0] = -1                                   # background
    # take the best-scoring component whose filled area is plausible;
    # if none is, return the best one (segment_lesion will flag it)
    order = np.argsort(-score)[: min(5, n - 1)]
    first = None
    for c in order:
        filled = ndi.binary_fill_holes(labels == c)
        if first is None:
            first = filled
        if min_frac <= filled.mean() <= max_frac:
            return filled.astype(np.uint8) * 255
    return first.astype(np.uint8) * 255


def fallback_mask(shape, axes_frac: float = 0.35) -> np.ndarray:
    """Central ellipse, used when the segmentation clearly failed."""
    h, w = shape[:2]
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, (w // 2, h // 2), (int(axes_frac * w), int(axes_frac * h)), 0, 0, 360, 255, -1)
    return m


# ---------------------------------------------------------------------------
# 5. Full pipeline
# ---------------------------------------------------------------------------
def segment_lesion(img: np.ndarray,
                   method: str = "otsu",
                   channel: str = "blue",
                   k: int = 2,
                   kmeans_lesion: str = "darkest",
                   flatten: bool = True,
                   smooth: str = "median",
                   smooth_ksize: int = 11,
                   min_frac: float = 0.02,
                   max_frac: float = 0.80,
                   use_fallback: bool = True) -> tuple[np.ndarray, dict]:
    """
    Segment the lesion on a PREPROCESSED image (output of preprocess()).

    method: 'otsu' (uses `channel`) or 'kmeans' (uses `k` and `kmeans_lesion`, Lab colour)
    flatten: correct vignetting / uneven light before thresholding (recommended)

    Sanity check: if the mask covers less than `min_frac` or more than
    `max_frac` of the image, the segmentation probably failed (very light
    lesion, lesion filling the image...). Then a central ellipse is returned
    and info['fallback'] = True, so failures can be counted and reported.

    Returns (mask, info) with info = {method, area_frac, fallback, threshold}.
    """
    sm = smooth_for_segmentation(img, smooth, smooth_ksize)
    thr = None
    if method == "otsu":
        raw, thr = otsu_mask(sm, channel, flatten=flatten)
        name = f"otsu_{channel}"
    elif method == "kmeans":
        raw = kmeans_mask(sm, k=k, lesion=kmeans_lesion, flatten=flatten)
        name = f"kmeans_k{k}_{kmeans_lesion}"
    else:
        raise ValueError(f"unknown method {method!r}")

    mask = postprocess_mask(raw, min_frac=min_frac, max_frac=max_frac)
    frac = float((mask > 0).mean())
    failed = not (min_frac <= frac <= max_frac)
    if failed and use_fallback:
        mask = fallback_mask(img.shape)
    return mask, {"method": name, "area_frac": frac, "fallback": failed and use_fallback,
                  "threshold": thr}


# ---------------------------------------------------------------------------
# 6. Spatial-context regions
# ---------------------------------------------------------------------------
def region_masks(mask: np.ndarray, ring_frac: float = 0.10, skin_gap_frac: float = 0.10) -> dict:
    """
    Regions for spatial-context features, with widths relative to the lesion size
    (equivalent radius r = sqrt(area / pi)):

      'lesion' : the mask itself
      'inner'  : lesion core (eroded by ring_frac * r)
      'border' : ring around the contour (dilation - erosion, ring_frac * r each side)
      'skin'   : everything outside the lesion dilated by (ring_frac + skin_gap_frac) * r
    """
    area = max(int((mask > 0).sum()), 1)
    r = np.sqrt(area / np.pi)
    ring = max(2, int(round(ring_frac * r)))
    gap = max(ring + 2, int(round((ring_frac + skin_gap_frac) * r)))
    ero = cv2.erode(mask, _disk(ring))
    dil = cv2.dilate(mask, _disk(ring))
    far = cv2.dilate(mask, _disk(gap))
    return {
        "lesion": mask,
        "inner": ero,
        "border": cv2.subtract(dil, ero),
        "skin": cv2.bitwise_not(far),
    }


# ---------------------------------------------------------------------------
# 7. Visualisation helper
# ---------------------------------------------------------------------------
def overlay_mask(img: np.ndarray, mask: np.ndarray, color=(0, 255, 0), thickness: int = 2) -> np.ndarray:
    """Return an RGB copy of `img` with the mask contour drawn (for matplotlib)."""
    out = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cnts, -1, color, thickness)
    return out
