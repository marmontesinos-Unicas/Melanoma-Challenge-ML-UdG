"""
Image preprocessing: FEATURE branch (classical pipeline).

This is the preprocessing applied to the image from which ALL features
(colour, texture, shape descriptors...) are extracted. It only removes
artefacts and acquisition differences; it does NOT smooth the image, so the
texture (pigment network, dots, globules) is preserved.

    resize_image          -> fixed size (the datasets have very different resolutions)
    remove_hair           -> black-hat + threshold + inpainting (DullRazor-like)
    shades_of_gray        -> colour constancy (tint only, or tint + brightness)
    crop_dark_borders     -> remove black frames / vignetting (off by default: this dataset has none)

`preprocess()` chains them with flags.

The heavy smoothing used to obtain the lesion MASK lives in
`segmentation.py` (segmentation branch), which starts from the output of
`preprocess()`:

    raw -> preprocess() -+-> features (colour / texture / shape)  <---+
                         |                                            | mask
                         +-> segmentation.segment_lesion() -----------+

All functions take and return uint8 BGR images (OpenCV convention).
"""

from __future__ import annotations

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 1. Resize
# ---------------------------------------------------------------------------
def resize_image(img: np.ndarray, short_side: int = 256) -> np.ndarray:
    """Resize so the shorter side equals `short_side`, keeping the aspect ratio."""
    h, w = img.shape[:2]
    scale = short_side / min(h, w)
    if scale == 1:
        return img
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=interp)


# ---------------------------------------------------------------------------
# 2. Dark borders / vignetting
# ---------------------------------------------------------------------------
def crop_dark_borders(img: np.ndarray, thresh: int = 20, min_fraction: float = 0.25) -> np.ndarray:
    """
    Crop the black frame / circular vignette that some dermatoscopes leave
    around the image (very common in BCN_20000).

    1. Rows/columns that are mostly dark (< `thresh`) are removed (straight frames).
    2. If the corners are still dark (circular vignette), a centred rectangle
       with the same aspect ratio is shrunk until its four corners fall inside
       the bright field of view.
    If the result would keep less than `min_fraction` of the image, the
    original is returned (the dark area was probably the lesion itself).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    bright = cv2.medianBlur(((gray > thresh) * 255).astype(np.uint8), 5) > 0

    rows = np.where(bright.mean(axis=1) > 0.5)[0]
    cols = np.where(bright.mean(axis=0) > 0.5)[0]
    if len(rows) == 0 or len(cols) == 0:
        return img
    r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1

    def corners_bright(a0, a1, b0, b1, m=max(3, int(0.03 * min(h, w)))):
        pts = [(a0 + m, b0 + m), (a0 + m, b1 - 1 - m), (a1 - 1 - m, b0 + m), (a1 - 1 - m, b1 - 1 - m)]
        return all(bright[y, x] for y, x in pts)

    if not corners_bright(r0, r1, c0, c1):
        cy, cx = (r0 + r1) / 2, (c0 + c1) / 2
        hh, hw = (r1 - r0) / 2, (c1 - c0) / 2
        for s in np.linspace(1.0, 0.3, 36):
            a0, a1 = int(cy - s * hh), int(cy + s * hh)
            b0, b1 = int(cx - s * hw), int(cx + s * hw)
            if corners_bright(a0, a1, b0, b1):
                r0, r1, c0, c1 = a0, a1, b0, b1
                break
        else:
            return img

    if (r1 - r0) * (c1 - c0) < min_fraction * h * w:
        return img
    return img[r0:r1, c0:c1]


# ---------------------------------------------------------------------------
# 3. Hair removal (DullRazor-like)
# ---------------------------------------------------------------------------
def hair_mask(img: np.ndarray, kernel_size: int = 17, thresh: int = 10,
              min_elongation: float = 3.0, min_area: int = 20,
              max_fill: float = 0.35) -> np.ndarray:
    """
    Binary mask of hair pixels.

    Black-hat = closing(img) - img: it highlights thin dark structures (hairs)
    that are smaller than the structuring element. A cross-shaped kernel works
    well for elongated hairs. `kernel_size` should be larger than the hair
    width (in pixels, after resizing).

    Black-hat also fires on small dark spots and on corners of the lesion
    border, so only connected components that are elongated
    (long side / short side >= `min_elongation`) or sparse (they fill less than
    `max_fill` of their rotated bounding box, e.g. crossing hairs), and at
    least as long as the kernel, are kept.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    blackhat = cv2.GaussianBlur(blackhat, (3, 3), 0)
    _, mask = cv2.threshold(blackhat, thresh, 255, cv2.THRESH_BINARY)

    # keep only elongated components (hairs), drop blobs
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        x, y, bw, bh = stats[i, :4]
        ys, xs = np.where(labels[y:y + bh, x:x + bw] == i)
        pts = np.column_stack([xs, ys]).astype(np.float32)
        (_, _), (a, b), _ = cv2.minAreaRect(pts)
        long_side = max(a, b)
        elongated = long_side / (min(a, b) + 1e-6) >= min_elongation
        # crossing hairs (X / Y shapes) are not elongated as a whole, but they
        # fill only a small part of their bounding rectangle, unlike blobs
        sparse = stats[i, cv2.CC_STAT_AREA] / (a * b + 1e-6) < max_fill
        if long_side >= kernel_size and (elongated or sparse):
            keep[i] = True
    mask = (keep[labels] * 255).astype(np.uint8)

    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return mask


def remove_hair(img: np.ndarray, kernel_size: int = 17, thresh: int = 10,
                inpaint_radius: int = 5, return_mask: bool = False):
    """Detect hair with black-hat and fill it with the surrounding colour (inpainting)."""
    mask = hair_mask(img, kernel_size, thresh)
    clean = cv2.inpaint(img, mask, inpaint_radius, cv2.INPAINT_TELEA)
    return (clean, mask) if return_mask else clean


# ---------------------------------------------------------------------------
# 4. Colour constancy
# ---------------------------------------------------------------------------
def shades_of_gray(img: np.ndarray, power: int = 6, normalize_brightness: bool = False,
                   target: float = 180.0) -> np.ndarray:
    """
    Shades of Gray colour constancy (Finlayson & Trezzi, 2004).

    Why: the images come from different dermatoscopes and clinics (HAM10000,
    BCN_20000, MSK), so the same skin can look more yellow, blue or red
    depending on the device and lighting. The colour of the light source is
    estimated with the Minkowski p-norm of each channel (a mean that gives
    more weight to bright pixels, i.e. mostly the skin), and each channel is
    multiplied by a gain so that this light becomes neutral gray.

    normalize_brightness=False (default):
        only the TINT is corrected (the ratio between channels). A brighter
        image stays brighter, so the lesion darkness is kept as information.
    normalize_brightness=True:
        tint AND brightness are corrected: the estimated light is mapped to
        gray level `target` in every image, so the skin has a similar
        brightness in all images. Risk: part of the real darkness
        differences between lesions may be normalised too.

    power=1 -> Gray World, power=inf -> White Patch (max-RGB); p=6 is the value
    used in most ISIC papers (Barata et al., 2015).
    """
    img_f = img.astype(np.float32)
    illum = np.power(np.mean(np.power(img_f, power), axis=(0, 1)), 1.0 / power)  # per channel (B, G, R)
    if normalize_brightness:
        gain = target / (illum + 1e-8)
    else:
        illum_unit = illum / (np.linalg.norm(illum) + 1e-8)
        gain = 1.0 / (illum_unit * np.sqrt(3) + 1e-8)
    out = img_f * gain
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def preprocess(img: np.ndarray,
               short_side: int | None = 256,
               hair: bool = True,
               color_constancy: bool = True,
               normalize_brightness: bool = False,
               crop_borders: bool = False) -> np.ndarray:
    """
    Feature-branch preprocessing, in this order:
        resize -> (crop borders) -> hair removal -> Shades of Gray

    Resize first so that the hair kernel size and the running time do not
    depend on the original resolution. Hair is removed before colour constancy
    so dark hairs do not bias the illuminant estimate.
    No denoising here (it would remove texture); see segmentation.py.
    """
    if short_side:
        img = resize_image(img, short_side)
    if crop_borders:
        img = crop_dark_borders(img)
    if hair:
        img = remove_hair(img)
    if color_constancy:
        img = shades_of_gray(img, normalize_brightness=normalize_brightness)
    return img


def load_image(path, **preprocess_kwargs) -> np.ndarray:
    """Read an image from disk (BGR). If kwargs are given, it is also preprocessed."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Could not read {path}")
    return preprocess(img, **preprocess_kwargs) if preprocess_kwargs else img
