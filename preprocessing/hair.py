"""Step 4: hair removal (port of the teammate's preprocessing.py; works in BGR like OpenCV)."""
import cv2
import numpy as np

INPAINT = {"telea": cv2.INPAINT_TELEA, "ns": cv2.INPAINT_NS}


def _line_kernels(k):
    """Thin lines at 0, 45, 90 and 135 degrees (k x k)."""
    horiz = np.ones((1, k), np.uint8)
    return [horiz, horiz.T.copy(), np.eye(k, dtype=np.uint8), np.fliplr(np.eye(k, dtype=np.uint8)).copy()]


def blackhat_response(img, kernel_size=17, kernel_shape="cross", channel="gray", blur_ksize=3):
    """Black-hat map (closing - image): bright where there are thin dark structures narrower than the kernel."""
    k = int(kernel_size) | 1
    if channel == "gray":
        chans = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)]
    elif channel == "L":
        chans = [cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0]]
    elif channel == "max_rgb":
        chans = list(cv2.split(img))
    else:
        raise ValueError(f"unsupported hair channel: {channel!r}")
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


def hair_mask_stages(img, kernel_size=17, thresh=10, min_elongation=3.0, min_area=20, max_fill=0.35,
                     kernel_shape="cross", channel="gray", blur_ksize=3, dilate_iter=1, filter_shape=True):
    """Hair mask (0/255) and intermediate steps: {'blackhat', 'thresholded', 'filtered', 'mask'}."""
    blackhat = blackhat_response(img, kernel_size, kernel_shape, channel, blur_ksize)
    _, raw = cv2.threshold(blackhat, thresh, 255, cv2.THRESH_BINARY)
    if filter_shape:
        # keep only elongated components (hairs); round blobs are discarded
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
            # crossing hairs (X / Y) are not elongated as a whole, but they fill little of their rotated box, unlike a blob
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


def remove_hair_stages(img, hair, ref_side=256):
    """img RGB. Detects hair at `ref_side` px (where the parameters were tuned) and inpaints at full size.
    Returns a dict with the steps: small (RGB at ref_side), blackhat, thresholded, filtered, mask_small,
    mask (0/255 at full size), radius (inpainting radius at full size) and clean (hair-free RGB)."""
    h, w = img.shape[:2]
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ratio = ref_side / min(h, w) if ref_side else 1.0
    small = bgr if ratio == 1.0 else cv2.resize(bgr, (round(w * ratio), round(h * ratio)), interpolation=cv2.INTER_AREA)
    mask_kw = {k: v for k, v in hair.items() if k not in ("inpaint_radius", "inpaint_method")}
    st = hair_mask_stages(small, **mask_kw)
    m = st["mask"]
    if ratio != 1.0:
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        m = ((m > 127) * 255).astype(np.uint8)
    radius = max(1, int(round(hair["inpaint_radius"] / ratio)))
    clean = cv2.inpaint(bgr, m, radius, INPAINT[hair["inpaint_method"]])
    return {"small": cv2.cvtColor(small, cv2.COLOR_BGR2RGB), "blackhat": st["blackhat"],
            "thresholded": st["thresholded"], "filtered": st["filtered"], "mask_small": st["mask"],
            "mask": m, "radius": radius, "clean": cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)}


def remove_hair(img, hair, ref_side=256):
    """Returns (hair-free RGB image, 0/1 mask at full size)."""
    s = remove_hair_stages(img, hair, ref_side)
    return s["clean"], (s["mask"] > 0).astype(np.uint8)
