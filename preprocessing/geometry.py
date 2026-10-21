"""Steps 1 and 2: crop the black borders of the dermatoscope, then central square + resize."""
import cv2
import numpy as np


# ---------- Step 1: black borders ----------
def valid_mask(img, dark_thr=35):
    """True where there is NO black border (dark regions connected to the image border, slightly dilated)."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    dark = (gray < dark_thr).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    _, lab = cv2.connectedComponents(dark, connectivity=4)
    border_ids = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    border = np.isin(lab, list(border_ids)).astype(np.uint8)
    m = max(3, int(0.01 * min(img.shape[:2])))
    border = cv2.dilate(border, np.ones((m, m), np.uint8))
    return border == 0


def largest_rectangle(mask):
    """Largest all-True rectangle (row-wise histogram algorithm). Returns (y0, y1, x0, x1)."""
    H, W = mask.shape
    heights = [0] * W
    best = (0, 0, H, 0, W)
    for y in range(H):
        row = mask[y]
        heights = [h + 1 if row[x] else 0 for x, h in enumerate(heights)]
        stack = []
        for x in range(W + 1):
            h = heights[x] if x < W else 0
            start = x
            while stack and stack[-1][1] >= h:
                s, sh = stack.pop()
                area = sh * (x - s)
                if area > best[0]:
                    best = (area, y - sh + 1, y + 1, s, x)
                start = s
            stack.append((start, h))
    return best[1:]


def crop_black_borders(img, dark_thr=35, min_area_frac=0.08):
    """Returns (crop, (y0, y1, x0, x1), fallback). fallback=True if no crop is done."""
    H, W = img.shape[:2]
    mask = valid_mask(img, dark_thr)
    scale = min(1.0, 256 / max(H, W))
    small = cv2.resize(mask.astype(np.uint8), None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_NEAREST).astype(bool)
    y0, y1, x0, x1 = largest_rectangle(small)
    y0, y1 = int(np.ceil(y0 / scale)), int(np.floor(y1 / scale))
    x0, x1 = int(np.ceil(x0 / scale)), int(np.floor(x1 / scale))
    fallback = False
    if (y1 - y0) * (x1 - x0) < min_area_frac * H * W:
        y0, y1, x0, x1 = 0, H, 0, W            # fallback: no crop
        fallback = True
    return img[y0:y1, x0:x1], (y0, y1, x0, x1), fallback


# ---------- Step 2: square + resize ----------
def central_square(shape):
    """(y0, x0, side) of the central square."""
    H, W = shape[:2]
    s = min(H, W)
    return (H - s) // 2, (W - s) // 2, s


def square_resize(img, size=512):
    y0, x0, s = central_square(img.shape)
    sq = img[y0:y0 + s, x0:x0 + s]
    interp = cv2.INTER_AREA if s > size else cv2.INTER_CUBIC
    return cv2.resize(sq, (size, size), interpolation=interp)
