"""
Image preprocessing: FEATURE branch (classical pipeline).

This is the preprocessing applied to the image from which ALL features
(colour, texture, shape descriptors...) are extracted. It only removes
artefacts and acquisition differences; it does NOT smooth the image, so the
texture (pigment network, dots, globules) is preserved.

    resize_image          -> fixed size (the datasets have very different resolutions)
    remove_hair           -> black-hat + threshold + inpainting (DullRazor-like)
    shades_of_gray        -> colour constancy (tint only, or tint + brightness)
    correct_illumination  -> uneven light / vignetting on L* (off by default)
    crop_dark_borders     -> old rectangular crop of black frames (kept for reference)
    fov_mask, remove_dark_borders -> automatic circular field of view: crop + fill corners (off by default)
    remove_specular       -> inpaint light reflections / gel bubbles      (off by default)
    denoise               -> gaussian / median / bilateral / NLM / anisotropic (off by default)
    gamma_correction, clahe_lab -> contrast enhancement                   (off by default)
    mse / psnr / ssim     -> quality metrics to compare denoising filters

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

from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi


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
# 2b. Dark frame / circular field of view (automatic, no threshold to tune)
# ---------------------------------------------------------------------------
def fov_mask(img: np.ndarray, edge: float = 0.5, frame: float = 0.03,
             min_dark_frame: float = 0.1) -> np.ndarray:
    """
    Field of view (FOV) of the dermatoscope: the bright region inside the
    black frame / circle. Returns a uint8 mask (255 = FOV).

    Automatic: the dark level is estimated from the outer `frame` strip of the
    image. If less than `min_dark_frame` of that strip is dark (< 40), there is
    no frame and the full image is returned (a circle that touches the image
    edges only darkens the corners, so the default is low: 10 %). Otherwise the
    FOV edge is placed at `edge` (0-1) of the way from the frame level to the
    skin level (75th percentile): 0.5 = middle of the dark vignetting ring.
    Pixels above it are the FOV; holes are filled (so a dark
    lesion inside the circle is NOT treated as background) and only the
    largest region is kept.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    b = max(2, int(frame * min(h, w)))
    strip = np.concatenate([gray[:b].ravel(), gray[-b:].ravel(), gray[:, :b].ravel(), gray[:, -b:].ravel()])
    full = np.full((h, w), 255, np.uint8)
    if np.mean(strip < 40) < min_dark_frame:
        return full
    dark = float(np.median(strip[strip < 40]))
    skin = float(np.percentile(gray, 75))
    level = dark + max(15.0, edge * (skin - dark))
    m = (cv2.medianBlur(gray, 5) > level).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return full
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = ndi.binary_fill_holes(labels == big)
    m = cv2.morphologyEx(m.astype(np.uint8) * 255, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    if (m > 0).mean() < 0.05:          # nothing sensible found
        return full
    return m


def remove_dark_borders(img: np.ndarray, mode: str = "crop_fill", margin: float = 0.02,
                        edge: float = 0.5, return_mask: bool = False):
    """
    Handle the black frame / circular field of view without losing the lesion.

    mode:
      'crop_fill'      (default) crop to the bounding box of the FOV, then fill
                       the remaining dark corners with the skin colour of the
                       FOV rim. The whole circle (and lesion) is kept.
      'fill'           same filling, but no crop (image size unchanged).
      'crop_inscribed' crop to the largest centred rectangle inside the FOV:
                       no filled pixels, but for small circles part of the
                       lesion can be cut.
      'none'           do nothing.
    margin: the FOV is shrunk by margin * shorter side to drop the dim,
            blurry rim of the circle.
    edge:   where the circle border is placed in the dark->skin transition (see fov_mask).
    Filling (instead of leaving black) avoids false hair at the circle edge
    (black-hat), a wrong illumination estimate and a biased Shades of Gray.
    """
    mask = fov_mask(img, edge)
    if mode == "none" or mask.min() > 0:
        return (img, mask) if return_mask else img
    h, w = mask.shape
    r = int(round(margin * min(h, w)))
    if r > 0:
        mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    ys, xs = np.where(mask > 0)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1

    if mode == "crop_inscribed":
        cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
        hh, hw = (y1 - y0) / 2, (x1 - x0) / 2
        for s in np.linspace(1.0, 0.2, 81):
            a0, a1 = int(cy - s * hh), int(cy + s * hh)
            b0, b1 = int(cx - s * hw), int(cx + s * hw)
            if mask[a0:a1, b0:b1].min() > 0:
                out = img[a0:a1, b0:b1]
                return (out, mask[a0:a1, b0:b1]) if return_mask else out
        return (img, mask) if return_mask else img

    # fill outside the FOV with the skin colour: brighter half of a ring just
    # inside the FOV (the very edge of the circle is dim, so it is skipped)
    inner1 = cv2.erode(mask, np.ones((21, 21), np.uint8))
    inner2 = cv2.erode(mask, np.ones((41, 41), np.uint8))
    ring = cv2.subtract(inner1, inner2) > 0
    if ring.sum() < 50:
        ring = mask > 0
    px = img[ring]
    lum = px.astype(np.float32).sum(1)
    colour = np.median(px[lum >= np.median(lum)], axis=0)
    out = img.copy()
    out[mask == 0] = colour.astype(np.uint8)
    seam = cv2.subtract(cv2.dilate(mask, np.ones((5, 5), np.uint8)), cv2.erode(mask, np.ones((5, 5), np.uint8)))
    out = cv2.inpaint(out, seam, 3, cv2.INPAINT_TELEA)        # soften the edge of the circle
    if mode == "crop_fill":
        out, mask = out[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    elif mode != "fill":
        raise ValueError(f"unknown mode {mode!r}")
    return (out, mask) if return_mask else out


# ---------------------------------------------------------------------------
# 3. Hair removal (DullRazor-like)
# ---------------------------------------------------------------------------
HAIR_KERNEL_SHAPES = ("cross", "rect", "ellipse", "lines")
HAIR_CHANNELS = ("gray", "L", "max_rgb")


def _line_kernels(k: int) -> list[np.ndarray]:
    """Thin line structuring elements at 0, 45, 90 and 135 degrees (k x k)."""
    horiz = np.ones((1, k), np.uint8)          # 1 x k / k x 1 are much faster than k x k
    return [horiz, horiz.T.copy(), np.eye(k, dtype=np.uint8), np.fliplr(np.eye(k, dtype=np.uint8)).copy()]


def blackhat_response(img: np.ndarray, kernel_size: int = 17, kernel_shape: str = "cross",
                      channel: str = "gray", blur_ksize: int = 3) -> np.ndarray:
    """
    Black-hat map (closing(img) - img): bright where there are thin dark
    structures narrower than the structuring element (hairs, dots...).

    kernel_shape: 'cross' (default), 'rect', 'ellipse', or 'lines' (max of the
                  black-hat with thin lines at 0/45/90/135 deg, as in DullRazor;
                  more selective for straight hairs, less for dots).
    channel:      'gray', 'L' (CIELab lightness) or 'max_rgb' (black-hat per
                  B, G, R channel and pixel-wise max, as in the original DullRazor).
    blur_ksize:   Gaussian blur of the map (odd, 1 = no blur) to join broken hairs.
    """
    k = int(kernel_size) | 1                              # force odd
    if channel == "gray":
        chans = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)]
    elif channel == "L":
        chans = [cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0]]
    elif channel == "max_rgb":
        chans = list(cv2.split(img))
    else:
        raise ValueError(f"unknown channel {channel!r}")

    if kernel_shape == "lines":
        kernels = _line_kernels(k)
    else:
        shape = {"cross": cv2.MORPH_CROSS, "rect": cv2.MORPH_RECT, "ellipse": cv2.MORPH_ELLIPSE}[kernel_shape]
        kernels = [cv2.getStructuringElement(shape, (k, k))]

    bh = np.zeros(img.shape[:2], np.uint8)
    for ch in chans:
        for ker in kernels:
            bh = np.maximum(bh, cv2.morphologyEx(ch, cv2.MORPH_BLACKHAT, ker))
    if blur_ksize and blur_ksize > 1:
        b = int(blur_ksize) | 1
        bh = cv2.GaussianBlur(bh, (b, b), 0)
    return bh


def hair_mask_stages(img: np.ndarray, kernel_size: int = 17, thresh: int = 10,
                     min_elongation: float = 3.0, min_area: int = 20,
                     max_fill: float = 0.35, kernel_shape: str = "cross",
                     channel: str = "gray", blur_ksize: int = 3,
                     dilate_iter: int = 1, filter_shape: bool = True) -> dict:
    """
    Same as `hair_mask` but returns every intermediate step (for notebooks):
        {'blackhat', 'thresholded', 'filtered', 'mask'}
    """
    blackhat = blackhat_response(img, kernel_size, kernel_shape, channel, blur_ksize)
    _, raw = cv2.threshold(blackhat, thresh, 255, cv2.THRESH_BINARY)

    if filter_shape:
        # keep only elongated components (hairs), drop blobs
        n, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
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
        filtered = (keep[labels] * 255).astype(np.uint8)
    else:
        filtered = raw.copy()

    mask = filtered
    if dilate_iter > 0:
        mask = cv2.dilate(filtered, np.ones((3, 3), np.uint8), iterations=int(dilate_iter))
    return {"blackhat": blackhat, "thresholded": raw, "filtered": filtered, "mask": mask}


def hair_mask(img: np.ndarray, kernel_size: int = 17, thresh: int = 10,
              min_elongation: float = 3.0, min_area: int = 20,
              max_fill: float = 0.35, **kwargs) -> np.ndarray:
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

    Extra kwargs (see `hair_mask_stages`): kernel_shape, channel, blur_ksize,
    dilate_iter, filter_shape.
    """
    return hair_mask_stages(img, kernel_size, thresh, min_elongation, min_area,
                            max_fill, **kwargs)["mask"]


_INPAINT = {"telea": cv2.INPAINT_TELEA, "ns": cv2.INPAINT_NS}


def remove_hair(img: np.ndarray, kernel_size: int = 17, thresh: int = 10,
                inpaint_radius: int = 5, return_mask: bool = False,
                inpaint_method: str = "telea", **mask_kwargs):
    """
    Detect hair with black-hat and fill it with the surrounding colour (inpainting).
    inpaint_method: 'telea' (fast marching) or 'ns' (Navier-Stokes).
    mask_kwargs go to `hair_mask` (min_elongation, min_area, max_fill,
    kernel_shape, channel, blur_ksize, dilate_iter, filter_shape).
    """
    mask = hair_mask(img, kernel_size, thresh, **mask_kwargs)
    clean = cv2.inpaint(img, mask, inpaint_radius, _INPAINT[inpaint_method])
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
# 4b. Illumination correction (uneven light / vignetting)
# ---------------------------------------------------------------------------
def illumination_background(L: np.ndarray, method: str = "polynomial", degree: int = 2,
                            keep_percentile: float = 40, blur_sigma: float = 0.25) -> np.ndarray:
    """
    Smooth estimate of the illumination of the lightness channel L (float32).

    'polynomial': 2-D polynomial (degree 1-4) fitted by least squares to the
                  pixels above `keep_percentile` (mostly skin, not lesion).
    'blur'      : very large Gaussian blur; sigma = blur_sigma * shorter side.
                  Simpler, but a big dark lesion also darkens the estimate.
    """
    h, w = L.shape
    if method == "blur":
        s = max(1.0, blur_sigma * min(h, w))
        return cv2.GaussianBlur(L, (0, 0), s)
    if method != "polynomial":
        raise ValueError(f"unknown illumination method {method!r}")
    yy, xx = np.mgrid[0:h, 0:w]
    x = (xx.ravel() / w - 0.5).astype(np.float32)
    y = (yy.ravel() / h - 0.5).astype(np.float32)
    A = np.stack([x ** i * y ** (d - i) for d in range(degree + 1) for i in range(d + 1)], 1)
    v = L.ravel()
    use = np.flatnonzero(v > np.percentile(v, keep_percentile))
    use = use[:: max(1, len(use) // 8000)]
    coef, *_ = np.linalg.lstsq(A[use], v[use], rcond=None)
    return (A @ coef).reshape(h, w)


def correct_illumination(img: np.ndarray, method: str = "polynomial", degree: int = 2,
                         keep_percentile: float = 40, blur_sigma: float = 0.25,
                         strength: float = 1.0, return_background: bool = False):
    """
    Remove slow spatial changes of brightness (darker corners, light gradient)
    on L* only (hue and saturation kept): L' = L / background * mean(background).
    `strength` in [0, 1] blends between no correction (0) and full correction (1).

    Different from Shades of Gray: that one corrects the COLOUR of the light
    (one gain per channel for the whole image); this one corrects WHERE the
    image is darker or brighter.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0]
    bg = np.maximum(illumination_background(L, method, degree, keep_percentile, blur_sigma), 1.0)
    gain = (bg.mean() / bg) ** strength
    lab[..., 0] = np.clip(L * gain, 0, 255)
    out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    if return_background:
        return out, np.clip(bg, 0, 255).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# 5. Specular reflections / gel bubbles
# ---------------------------------------------------------------------------
def specular_mask(img: np.ndarray, v_thresh: int = 230, s_max: int = 40,
                  dilate_iter: int = 1, max_area_frac: float = 0.02) -> np.ndarray:
    """
    Very bright and almost unsaturated pixels (HSV: V >= v_thresh, S <= s_max):
    reflections of the light on the gel / glass plate and air bubbles.
    Components larger than `max_area_frac` of the image are ignored (it is
    probably very light skin, not a reflection).
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 2] >= v_thresh) & (hsv[..., 1] <= s_max)).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    too_big = stats[:, cv2.CC_STAT_AREA] > max_area_frac * m.size
    too_big[0] = True
    m = ((~too_big)[labels] * 255).astype(np.uint8)
    if dilate_iter > 0:
        m = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=int(dilate_iter))
    return m


def remove_specular(img: np.ndarray, v_thresh: int = 230, s_max: int = 40,
                    dilate_iter: int = 1, max_area_frac: float = 0.02,
                    inpaint_radius: int = 3, return_mask: bool = False):
    """Detect specular reflections (see `specular_mask`) and inpaint them."""
    mask = specular_mask(img, v_thresh, s_max, dilate_iter, max_area_frac)
    clean = cv2.inpaint(img, mask, inpaint_radius, cv2.INPAINT_TELEA)
    return (clean, mask) if return_mask else clean


# ---------------------------------------------------------------------------
# 6. Contrast / intensity
# ---------------------------------------------------------------------------
def gamma_correction(img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """out = 255 * (in/255)^(1/gamma). gamma > 1 brightens, gamma < 1 darkens."""
    if gamma == 1.0:
        return img
    lut = np.clip(255.0 * (np.arange(256) / 255.0) ** (1.0 / gamma), 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def clahe_lab(img: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalisation) on the L* channel
    only, so the hue is not changed. Higher `clip_limit` = stronger contrast
    (and more amplified noise); `tile_grid` = number of tiles per side.
    Careful: it changes the colour statistics, so colour features computed
    after CLAHE are not comparable with the raw ones.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid), int(tile_grid)))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# 7. Denoising (classical filters, see the literature review table)
# ---------------------------------------------------------------------------
DENOISE_METHODS = ("none", "gaussian", "median", "bilateral", "nlm", "anisotropic")


def anisotropic_diffusion(img: np.ndarray, n_iter: int = 10, kappa: float = 30.0,
                          gamma: float = 0.15, option: int = 1) -> np.ndarray:
    """
    Perona-Malik anisotropic diffusion: smooths inside homogeneous regions but
    stops at edges (gradient > kappa), so the lesion border is kept.
    option=1: exp(-(g/kappa)^2) favours high-contrast edges;
    option=2: 1/(1+(g/kappa)^2) favours wide regions. gamma <= 0.25 for stability.
    """
    out = img.astype(np.float32).copy()
    if out.ndim == 2:
        out = out[..., None]
    for _ in range(int(n_iter)):
        dn = np.roll(out, -1, 0) - out
        ds = np.roll(out, 1, 0) - out
        de = np.roll(out, -1, 1) - out
        dw = np.roll(out, 1, 1) - out
        if option == 1:
            c = lambda d: np.exp(-(d / kappa) ** 2)
        else:
            c = lambda d: 1.0 / (1.0 + (d / kappa) ** 2)
        out += gamma * (c(dn) * dn + c(ds) * ds + c(de) * de + c(dw) * dw)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out[..., 0] if img.ndim == 2 else out


def denoise(img: np.ndarray, method: str = "none", ksize: int = 5, sigma: float = 0.0,
            bilateral_d: int = 9, sigma_color: float = 75, sigma_space: float = 75,
            nlm_h: float = 5, nlm_h_color: float = 5, nlm_template: int = 7, nlm_search: int = 21,
            ad_iter: int = 10, ad_kappa: float = 30.0, ad_gamma: float = 0.15) -> np.ndarray:
    """
    Classical denoising filters (all edge-aware except gaussian/median).

    gaussian    : ksize, sigma (0 = from ksize)
    median      : ksize (good for salt & pepper)
    bilateral   : bilateral_d, sigma_color, sigma_space
    nlm         : Non-Local Means (cv2.fastNlMeansDenoisingColored): nlm_h (luminance strength),
                  nlm_h_color, nlm_template (patch size), nlm_search (search window)
    anisotropic : Perona-Malik diffusion: ad_iter, ad_kappa, ad_gamma

    NOTE: the feature branch normally does NOT denoise because it also removes
    texture (pigment network, dots). Use mild settings and check the result.
    """
    if method in (None, "none"):
        return img
    k = int(ksize) | 1
    if method == "gaussian":
        return cv2.GaussianBlur(img, (k, k), sigma)
    if method == "median":
        return cv2.medianBlur(img, k)
    if method == "bilateral":
        return cv2.bilateralFilter(img, int(bilateral_d), sigma_color, sigma_space)
    if method == "nlm":
        return cv2.fastNlMeansDenoisingColored(img, None, nlm_h, nlm_h_color,
                                               int(nlm_template) | 1, int(nlm_search) | 1)
    if method == "anisotropic":
        return anisotropic_diffusion(img, ad_iter, ad_kappa, ad_gamma)
    raise ValueError(f"unknown denoise method {method!r}")


def add_noise(img: np.ndarray, kind: str = "gaussian", level: float = 10.0, seed: int = 0) -> np.ndarray:
    """
    Synthetic noise to evaluate the denoising filters (clean -> noisy -> denoised).
    gaussian: std = level (gray levels); speckle: multiplicative, std = level/100;
    salt_pepper: fraction of corrupted pixels = level/100.
    """
    rng = np.random.default_rng(seed)
    f = img.astype(np.float32)
    if kind == "gaussian":
        f = f + rng.normal(0, level, f.shape)
    elif kind == "speckle":
        f = f * (1 + rng.normal(0, level / 100.0, f.shape))
    elif kind == "salt_pepper":
        out = img.copy()
        r = rng.random(img.shape[:2])
        p = level / 100.0
        out[r < p / 2] = 0
        out[r > 1 - p / 2] = 255
        return out
    elif kind in (None, "none"):
        return img.copy()
    else:
        raise ValueError(f"unknown noise {kind!r}")
    return np.clip(f, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 8. Image quality metrics (MSE / PSNR / SSIM, as in the denoising papers)
# ---------------------------------------------------------------------------
def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def psnr(a: np.ndarray, b: np.ndarray, max_val: float = 255.0) -> float:
    m = mse(a, b)
    return float("inf") if m == 0 else float(10 * np.log10(max_val ** 2 / m))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean SSIM on the gray-level images (Gaussian window 11x11, sigma 1.5; Wang et al., 2004)."""
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_a, mu_b = blur(a), blur(b)
    s_a = blur(a * a) - mu_a ** 2
    s_b = blur(b * b) - mu_b ** 2
    s_ab = blur(a * b) - mu_a * mu_b
    ssim_map = ((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (s_a + s_b + c2))
    return float(ssim_map.mean())


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def preprocess(img: np.ndarray,
               short_side: int | None = 256,
               hair: bool = True,
               color_constancy: bool = True,
               normalize_brightness: bool = False,
               crop_borders: bool = False,
               specular: bool = False,
               denoise_method: str | None = None,
               gamma: float = 1.0,
               clahe: bool = False,
               hair_before_cc: bool = True,
               illumination: bool = False,
               illum_kwargs: dict | None = None,
               hair_kwargs: dict | None = None,
               cc_kwargs: dict | None = None,
               crop_kwargs: dict | None = None,
               specular_kwargs: dict | None = None,
               denoise_kwargs: dict | None = None,
               clahe_kwargs: dict | None = None) -> np.ndarray:
    """
    Feature-branch preprocessing, in this order:
        resize -> (dark borders) -> (specular) -> (illumination) -> hair removal
               -> Shades of Gray -> (denoise) -> (gamma) -> (CLAHE)
    crop_borders=True uses remove_dark_borders (automatic FOV detection,
    crop to the circle + fill the corners); crop_kwargs: mode, margin, edge.
    hair_before_cc=False moves hair removal after Shades of Gray.
    Hair removal is never done after denoising (blurs the hairs) or CLAHE
    (enhanced skin texture would be detected as hair).

    Resize first so that the hair kernel size and the running time do not
    depend on the original resolution. Hair is removed before colour constancy
    so dark hairs do not bias the illuminant estimate. Steps in brackets are
    off by default (default output is the same as before they were added).
    Every step's parameters can be changed with the matching *_kwargs dict,
    e.g. hair_kwargs=dict(kernel_size=21, thresh=12, kernel_shape='lines').
    No denoising by default (it would remove texture); see segmentation.py.
    """
    if short_side:
        img = resize_image(img, short_side)
    if crop_borders:
        img = remove_dark_borders(img, **(crop_kwargs or {}))
        if short_side:
            img = resize_image(img, short_side)        # cropping changed the size
    if specular:
        img = remove_specular(img, **(specular_kwargs or {}))
    if illumination:
        img = correct_illumination(img, **(illum_kwargs or {}))
    if hair and hair_before_cc:
        img = remove_hair(img, **(hair_kwargs or {}))
    if color_constancy:
        img = shades_of_gray(img, normalize_brightness=normalize_brightness, **(cc_kwargs or {}))
    if hair and not hair_before_cc:
        img = remove_hair(img, **(hair_kwargs or {}))
    if denoise_method and denoise_method != "none":
        img = denoise(img, denoise_method, **(denoise_kwargs or {}))
    if gamma != 1.0:
        img = gamma_correction(img, gamma)
    if clahe:
        img = clahe_lab(img, **(clahe_kwargs or {}))
    return img


def load_image(path, **preprocess_kwargs) -> np.ndarray:
    """Read an image from disk (BGR). If kwargs are given, it is also preprocessed."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Could not read {path}")
    return preprocess(img, **preprocess_kwargs) if preprocess_kwargs else img


# ---------------------------------------------------------------------------
# Parameters chosen in the explorer notebook (05_preprocessing_explorer.ipynb)
# ---------------------------------------------------------------------------
PARAMS_FILE = Path(__file__).with_name("preproc_params.json")


def config_from_params(params: dict, short_side: int | None = 256) -> dict:
    """
    Convert the slider values of the explorer windows
    ({window: {param: value, 'use_in_chain': bool}}) into preprocess() kwargs.
    Chain order: borders -> specular -> illumination -> Shades of Gray -> hair
                 -> denoise -> contrast.
    """
    def P(s):
        return {k: v for k, v in params.get(s, {}).items() if k not in ("seed", "use_in_chain")}

    def on(s, default=False):
        return bool(params.get(s, {}).get("use_in_chain", default))

    dn = P("denoise")
    ct = P("contrast")
    cc = P("color_constancy")
    return dict(
        short_side=short_side,
        crop_borders=on("borders"), crop_kwargs=P("borders"),
        specular=on("specular"), specular_kwargs=P("specular"),
        illumination=on("illumination"), illum_kwargs=P("illumination"),
        color_constancy=on("color_constancy", True),
        normalize_brightness=cc.get("normalize_brightness", False),
        cc_kwargs={k: v for k, v in cc.items() if k in ("power", "target")},
        hair=on("hair", True), hair_kwargs=P("hair"), hair_before_cc=False,
        denoise_method=dn.get("method") if on("denoise") else None,
        denoise_kwargs=dict(ksize=dn["ksize"], bilateral_d=dn["ksize"], sigma_color=dn["sigma_color"],
                            nlm_h=dn["nlm_h"], nlm_h_color=dn["nlm_h"], ad_iter=dn["ad_iter"],
                            ad_kappa=dn["ad_kappa"]) if on("denoise") else {},
        gamma=ct.get("gamma", 1.0) if on("contrast") else 1.0,
        clahe=on("contrast") and ct.get("clahe", False),
        clahe_kwargs={k: v for k, v in ct.items() if k in ("clip_limit", "tile_grid")},
    )


def load_feature_preproc(path=None, short_side: int | None = 256) -> dict:
    """
    preprocess() kwargs with the parameters saved by the explorer notebook:

        FEATURE_PREPROC = load_feature_preproc()
        img = preprocess(raw, **FEATURE_PREPROC)

    If the file does not exist, returns {} (= preprocess() defaults).
    """
    import json
    path = Path(path) if path else PARAMS_FILE
    if not path.exists():
        print(f"[load_feature_preproc] {path.name} not found -> default preprocessing")
        return {}
    return config_from_params(json.loads(path.read_text()), short_side)
