"""Step 5: segmentation with Otsu on one channel (no blur) + morphological cleaning, and quality control."""
import cv2
import numpy as np


def get_channel(img, name):
    """Computes only the requested channel (faster than computing all of them)."""
    if name == "gray":
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if name in ("L", "a", "b"):
        return cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[..., "Lab".index(name)]
    if name == "Cr":
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)[..., 1]
    if name == "S":
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[..., 1]
    raise ValueError(f"Unsupported channel: {name}")


def border_ring(shape, frac=0.08):
    H, W = shape
    m = int(frac * min(H, W))
    ring = np.ones((H, W), bool)
    ring[m:H - m, m:W - m] = False
    return ring


def otsu_stages(ch):
    """Otsu on the channel normalized to 0-255. Returns (bool mask, threshold, was it inverted?)."""
    ch = cv2.normalize(ch, None, 0, 255, cv2.NORM_MINMAX)
    t, bw = cv2.threshold(ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = bw > 0
    inverted = bool(bw[border_ring(bw.shape)].mean() > 0.5)   # the "foreground" is the class that touches the border least
    if inverted:
        bw = ~bw
    return bw, float(t), inverted


def otsu_mask(ch):
    return otsu_stages(ch)[0]


def clean_mask(bw, open_k=5, close_k=15, center_sigma=0.35):
    """Opening + closing, keeps the large central component and fills its holes."""
    ell = lambda n: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))
    m = bw.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ell(open_k))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ell(close_k))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n == 1:
        return np.zeros_like(bw)
    S = m.shape[0]
    c = np.array([m.shape[1] / 2, m.shape[0] / 2])
    score = [stats[i, cv2.CC_STAT_AREA] * np.exp(-(np.linalg.norm(cent[i] - c) / (center_sigma * S)) ** 2)
             for i in range(1, n)]
    best = int(np.argmax(score)) + 1
    m = (lab == best).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, cnts, -1, 1, thickness=-1)
    return filled.astype(bool)


def segment(img, channel="S", open_k=5, close_k=15, center_sigma=0.35):
    return clean_mask(otsu_mask(get_channel(img, channel)), open_k, close_k, center_sigma)


# ---------- Mask quality control (flag only, nothing is discarded) ----------
def mask_qc(mask, min_frac=0.01, max_frac=0.60, min_solidity=0.75):
    area_frac = float(mask.mean())
    touches = bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
    solidity = np.nan
    if mask.any():
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(cnts, key=cv2.contourArea)
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        solidity = float(cv2.contourArea(cnt) / hull_area) if hull_area > 0 else np.nan
    ok = (min_frac <= area_frac <= max_frac) and not touches and (solidity >= min_solidity)
    return {"mask_area_frac": area_frac, "mask_touches_border": touches,
            "mask_solidity": solidity, "seg_ok": bool(ok)}
