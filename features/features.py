"""Lesion feature extraction (shape, color, texture) for the melanoma challenge.

Families and prefixes (412 columns):
    shape_   (52)  shape, inside the lesion only
    color_   (82)  color, inside the lesion only
    tex_     (104) texture, inside the lesion only (includes LNDP and Gabor)
    gcolor_  (70)  color over the whole image
    gtex_    (104) texture over the whole image (includes LNDP and Gabor)

Usage: extract_one(row) takes a manifest row (columns name, class, subtype, preprocessed, mask,
and optionally seg_ok and crop_fallback) and returns a dict with one CSV row.
"""
import traceback
import warnings

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import pdist
from scipy.stats import kurtosis, skew
from skimage.feature import blob_log, graycomatrix, local_binary_pattern
from skimage.filters import sato
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

# ---------------- Configuration ----------------
CLASS_TO_LABEL = {"nevus": 0, "others": 1}
EXTENDED_SHAPE = True    # Hu, Fourier, curvature, fractal, Feret... (False = basic shape only)
KMEANS_K = 6
RIDGE_THR = 0.02         # (heuristic) response threshold for the Sato ridge filter
FEATURE_PREFIXES = ("shape_", "color_", "tex_", "gcolor_", "gtex_")


# ======================================================================
# Utilities
# ======================================================================
ell = lambda n: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))


def read_rgb(path):
    bgr = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_mask(path):
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_GRAYSCALE) > 127


def largest_contour(mask):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(cnts, key=cv2.contourArea)


def resample_contour(cnt, n=256):
    """Resample the contour to n (x, y) points, equally spaced by arc length."""
    p = cnt.reshape(-1, 2).astype(np.float64)
    p = np.vstack([p, p[:1]])
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    t = np.linspace(0, d[-1], n, endpoint=False)
    return np.column_stack([np.interp(t, d, p[:, 0]), np.interp(t, d, p[:, 1])])


def stats7(v, name, out):
    v = np.asarray(v, np.float64)
    out[f"{name}_mean"] = v.mean()
    out[f"{name}_std"] = v.std()
    out[f"{name}_skew"] = skew(v) if v.std() > 1e-9 else 0.0
    out[f"{name}_kurt"] = kurtosis(v) if v.std() > 1e-9 else 0.0
    p5, p50, p95 = np.percentile(v, [5, 50, 95])
    out[f"{name}_p5"], out[f"{name}_p50"], out[f"{name}_p95"] = p5, p50, p95


def skin_ring(mask, gap=10, width=60, min_px=500):
    """Ring of skin around the lesion (at a distance between gap and width px)."""
    dist = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
    ring = (dist > gap) & (dist <= width)
    return ring if ring.sum() >= min_px else None


class Guard:
    """Isolates a code block: if it fails, records name, line and message and carries on with the rest."""

    def __init__(self, errs, name):
        self.errs, self.name = errs, name

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is not None:
            self.errs.append(f"{self.name}:{et.__name__}@L{tb.tb_lineno}:{str(ev)[:80]}")
            return True
        return False


def err_info(e):
    """Exception type, line and function where it failed."""
    fr = traceback.extract_tb(e.__traceback__)[-1]
    return f"{type(e).__name__}@{fr.name}:L{fr.lineno}:{str(e)[:80]}"


def principal_axes(mask):
    m = cv2.moments(mask.astype(np.uint8), binaryImage=True)
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    mu20, mu02, mu11 = m["mu20"] / m["m00"], m["mu02"] / m["m00"], m["mu11"] / m["m00"]
    theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
    tmp = np.sqrt(4 * mu11 ** 2 + (mu20 - mu02) ** 2)
    l1, l2 = (mu20 + mu02 + tmp) / 2, (mu20 + mu02 - tmp) / 2
    return cx, cy, theta, max(l1, 1e-9), max(l2, 1e-9), m


def aligned(arr, cx, cy, theta, interp):
    """Rotate the image so the major axis is horizontal and the centroid is at the center (canvas = half resolution)."""
    h, w = arr.shape[:2]
    M = cv2.getRotationMatrix2D((cx, cy), np.degrees(theta), 0.5)
    M[0, 2] += w / 2 - cx
    M[1, 2] += h / 2 - cy
    return cv2.warpAffine(arr, M, (w, h), flags=interp)


# ======================================================================
# Family 1 · Shape (52 columns)
# ======================================================================
def shape_features(img, mask):
    out = {}
    H, W = mask.shape
    m8 = mask.astype(np.uint8)
    cnt = largest_contour(mask)
    area = float(cv2.contourArea(cnt))
    perim = float(cv2.arcLength(cnt, True))
    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    x, y, w, h = cv2.boundingRect(cnt)
    cx, cy, theta, l1, l2, mom = principal_axes(mask)
    R = np.sqrt(area / np.pi)                         # equivalent radius (scale normalizer)

    # size and proportions
    out["shape_area_frac"] = float(mask.mean())
    out["shape_perimeter_px"] = perim                              # paper Eq. 20
    out["shape_major_axis_px"] = float(4 * np.sqrt(l1))            # major axis of the equivalent ellipse (paper Eq. 21)
    out["shape_minor_axis_px"] = float(4 * np.sqrt(l2))            # minor axis of the equivalent ellipse (paper Eq. 21)
    out["shape_extent"] = area / (w * h)
    out["shape_circularity"] = 4 * np.pi * area / perim ** 2
    out["shape_solidity"] = area / hull_area if hull_area > 0 else np.nan
    out["shape_eccentricity"] = float(np.sqrt(1 - l2 / l1))
    out["shape_aspect_ratio"] = float(np.sqrt(l1 / l2))            # major/minor, as in the paper (Eq. 22)
    (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
    out["shape_rectangularity"] = area / max(rw * rh, 1e-9)

    # shape asymmetry (IoU with the mirrored mask about the major and minor axes)
    a = aligned(m8, cx, cy, theta, cv2.INTER_NEAREST) > 0
    out["shape_asym_major"] = (a & a[::-1, :]).sum() / max((a | a[::-1, :]).sum(), 1)   # reflection about the major axis
    out["shape_asym_minor"] = (a & a[:, ::-1]).sum() / max((a | a[:, ::-1]).sum(), 1)   # reflection about the minor axis
    out["shape_asym_mean"] = (out["shape_asym_major"] + out["shape_asym_minor"]) / 2    # paper Eq. 18

    # border irregularity: radial signature (centroid-to-contour distance)
    pts = resample_contour(cnt, 256)
    r = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    rn = r / r.mean()
    out["shape_radial_cv"] = float(rn.std())
    out["shape_radial_range"] = float(rn.max() - rn.min())

    # concavities
    try:
        idx = cv2.convexHull(cnt, returnPoints=False)
        d = cv2.convexityDefects(cnt, idx)
        depth = (d.reshape(-1, 4)[:, 3] / 256.0) / R if d is not None else np.array([0.0])   # (n,4) or (n,1,4) depending on the OpenCV version
    except cv2.error:
        depth = np.array([0.0])
    out["shape_n_concavities"] = int((depth > 0.05).sum())
    out["shape_concavity_max"] = float(depth.max())
    out["shape_concavity_mean"] = float(depth[depth > 0.05].mean()) if (depth > 0.05).any() else 0.0

    # edge sharpness: gradient in a band around the contour
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g = cv2.GaussianBlur(gray, (0, 0), 1.0)
    grad = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    k = 5
    inner = m8 & ~cv2.erode(m8, ell(2 * k + 1)).astype(bool)
    outer = cv2.dilate(m8, ell(2 * k + 1)).astype(bool) & ~mask
    band = inner.astype(bool) | outer
    yy, xx = np.nonzero(band)
    sector = np.clip(((np.arctan2(yy - cy, xx - cx) + np.pi) / (2 * np.pi) * 8).astype(int), 0, 7)
    gv = grad[yy, xx]
    sec_g = np.array([gv[sector == s].mean() if (sector == s).any() else np.nan for s in range(8)])
    out["shape_edge_grad_mean"] = float(gv.mean())
    out["shape_edge_grad_std"] = float(gv.std())
    out["shape_edge_grad_sector_std"] = float(np.nanstd(sec_g))
    out["shape_edge_grad_sector_min"] = float(np.nanmin(sec_g))
    out["shape_edge_grad_sector_max"] = float(np.nanmax(sec_g))
    iy, ix = np.nonzero(inner.astype(bool))
    oy, ox = np.nonzero(outer)
    isec = np.clip(((np.arctan2(iy - cy, ix - cx) + np.pi) / (2 * np.pi) * 8).astype(int), 0, 7)
    osec = np.clip(((np.arctan2(oy - cy, ox - cx) + np.pi) / (2 * np.pi) * 8).astype(int), 0, 7)
    contrast = np.array([gray[iy, ix][isec == s].mean() - gray[oy, ox][osec == s].mean()
                         if (isec == s).any() and (osec == s).any() else np.nan for s in range(8)])
    out["shape_edge_contrast_mean"] = float(np.nanmean(contrast) / 255)     # <0: border darker on the inside
    out["shape_edge_contrast_sector_std"] = float(np.nanstd(contrast) / 255)

    if not EXTENDED_SHAPE:
        return out
    errs = []
    # Hu moments (signed log)
    with Guard(errs, "hu"):
        hu = cv2.HuMoments(mom).ravel()
        for i, v in enumerate(hu):
            out[f"shape_hu{i + 1}"] = float(-np.sign(v) * np.log10(abs(v) + 1e-30))

    # Fourier descriptors of the radial signature (magnitudes of the first 10 harmonics)
    with Guard(errs, "fourier"):
        F = np.abs(np.fft.rfft(rn)) / len(rn)
        for i in range(1, 11):
            out[f"shape_fourier{i}"] = float(F[i])

    # contour curvature (normalized by the equivalent radius)
    with Guard(errs, "curvature"):
        sx = gaussian_filter1d(pts[:, 0], 3, mode="wrap")
        sy = gaussian_filter1d(pts[:, 1], 3, mode="wrap")
        dx, dy = np.gradient(sx), np.gradient(sy)
        ddx, ddy = np.gradient(dx), np.gradient(dy)
        kappa = (dx * ddy - dy * ddx) / np.maximum((dx ** 2 + dy ** 2) ** 1.5, 1e-9) * R
        kappa = kappa * np.sign(np.mean(kappa))                 # same orientation for every lesion
        out["shape_curv_mean_abs"] = float(np.abs(kappa).mean())
        out["shape_curv_std"] = float(kappa.std())
        out["shape_curv_max"] = float(kappa.max())
        out["shape_curv_min"] = float(kappa.min())
        out["shape_bending_energy"] = float((kappa ** 2).mean())
        out["shape_curv_sign_changes"] = int((np.diff(np.sign(kappa)) != 0).sum())

    # Feret diameters (normalized by the equivalent radius)
    with Guard(errs, "feret"):
        hp = hull.reshape(-1, 2).astype(np.float64)
        fmax = pdist(hp).max() if len(hp) > 1 else 0.0
        angs = np.deg2rad(np.arange(0, 180, 2))
        proj = hp @ np.vstack([np.cos(angs), np.sin(angs)])
        fmin = (proj.max(0) - proj.min(0)).min()
        out["shape_feret_max"] = float(fmax / R)
        out["shape_feret_min"] = float(fmin / R)

    # fractal dimension of the border (box-counting) and lacunarity
    with Guard(errs, "fractal"):
        border = (m8 - cv2.erode(m8, np.ones((3, 3), np.uint8))).astype(bool)
        sizes = [2, 4, 8, 16, 32, 64]
        counts = []
        for sz in sizes:
            hh, ww = (H // sz) * sz, (W // sz) * sz
            counts.append(border[:hh, :ww].reshape(hh // sz, sz, ww // sz, sz).any(axis=(1, 3)).sum())
        out["shape_fractal_dim"] = float(np.polyfit(np.log(1 / np.array(sizes)), np.log(np.maximum(counts, 1)), 1)[0])
        box = max(4, int(R // 4))
        sub = m8[y:y + h, x:x + w].astype(np.float32)
        f = cv2.boxFilter(sub, -1, (box, box))
        out["shape_lacunarity"] = float(1 + f.var() / max(f.mean() ** 2, 1e-9))

    if errs:
        out["_errors"] = errs
    return out


# ======================================================================
# Family 2 · Color (82 columns)
# ======================================================================
def color_features(img, mask):
    out = {}
    lab = cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)      # L 0-100, a/b ~ -127..127
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    chans = {"R": img[..., 0], "G": img[..., 1], "B": img[..., 2], "S": hsv[..., 1], "V": hsv[..., 2],
             "L": lab[..., 0], "a": lab[..., 1], "b": lab[..., 2]}
    ring = skin_ring(mask)

    for k, ch in chans.items():
        ch = ch.astype(np.float64)
        stats7(ch[mask], f"color_{k}", out)
        out[f"color_rel_{k}"] = ch[mask].mean() - ch[ring].mean() if ring is not None else np.nan   # lesion - skin

    ang = hsv[..., 0][mask].astype(np.float64) * 2 * np.pi / 180                # hue is circular
    out["color_hue_sin"], out["color_hue_cos"] = np.sin(ang).mean(), np.cos(ang).mean()
    out["color_hue_R"] = float(np.hypot(out["color_hue_sin"], out["color_hue_cos"]))

    # number of colors (K-means in Lab inside the lesion)
    px = lab[mask].astype(np.float64)
    if len(px) > 4000:
        px = px[np.random.RandomState(0).choice(len(px), 4000, replace=False)]
    km = KMeans(n_clusters=min(KMEANS_K, len(px)), n_init=3, random_state=0).fit(px)
    frac = np.bincount(km.labels_, minlength=km.n_clusters) / len(px)
    sig = frac > 0.05
    out["color_n_colors"] = int(sig.sum())
    out["color_entropy"] = float(-(frac[frac > 0] * np.log2(frac[frac > 0])).sum())
    out["color_cluster_spread"] = float(pdist(km.cluster_centers_[sig]).max()) if sig.sum() > 1 else 0.0

    # clinical color fractions (heuristic rules in Lab, mutually exclusive and applied in this order).
    # light_brown is the remainder (the 6 fractions sum to 1): it counts toward n_clinical_colors but is not saved.
    L, a, b = (lab[..., i][mask] for i in range(3))
    chroma = np.hypot(a, b)
    free = np.ones(L.shape, bool)
    rules = [("white", (L >= 75) & (chroma < 12)),
             ("black", L < 20),
             ("blue_gray", (b < 4) & (L < 70)),
             ("red", a >= 22),
             ("dark_brown", L < 38),
             ("light_brown", np.ones(L.shape, bool))]
    n_present = 0
    for name, cond in rules:
        sel = cond & free
        free &= ~sel
        if name != "light_brown":
            out[f"color_frac_{name}"] = float(sel.mean())
        n_present += sel.mean() > 0.05
    out["color_n_clinical_colors"] = int(n_present)

    # center vs periphery (distance transform)
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    mx = dt.max()
    if mx >= 5:
        core, per = mask & (dt >= 0.6 * mx), mask & (dt <= 0.2 * mx)
        for k in ["L", "a", "b", "S"]:
            ch = chans[k].astype(np.float64)
            out[f"color_core_minus_periph_{k}"] = ch[core].mean() - ch[per].mean()
    else:
        for k in ["L", "a", "b", "S"]:
            out[f"color_core_minus_periph_{k}"] = np.nan

    # color asymmetry between halves (about the major and minor axes)
    cx, cy, theta, _, _, _ = principal_axes(mask)
    am = aligned(mask.astype(np.uint8), cx, cy, theta, cv2.INTER_NEAREST) > 0
    al = aligned(lab, cx, cy, theta, cv2.INTER_LINEAR)
    c = am.shape[0] // 2
    top, bot = am.copy(), am.copy()
    top[c:] = False
    bot[:c] = False
    lef, rig = am.copy(), am.copy()
    lef[:, c:] = False
    rig[:, :c] = False
    for nm, (m1, m2) in {"major": (top, bot), "minor": (lef, rig)}.items():
        out[f"color_asym_{nm}"] = float(np.linalg.norm(al[m1].mean(0) - al[m2].mean(0))) \
            if m1.sum() > 20 and m2.sum() > 20 else np.nan
    return out


# ======================================================================
# Family 3 · Texture (104 columns)
# ======================================================================
def glcm_props(P):
    """P: normalized (32x32) matrix. Returns the 6 GLCM properties."""
    n = P.shape[0]
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    mu_i, mu_j = (P * i).sum(), (P * j).sum()
    sd_i, sd_j = np.sqrt((P * (i - mu_i) ** 2).sum()), np.sqrt((P * (j - mu_j) ** 2).sum())
    asm = (P ** 2).sum()
    nz = P > 0
    return {"contrast": (P * (i - j) ** 2).sum(),
            "dissimilarity": (P * np.abs(i - j)).sum(),
            "homogeneity": (P / (1 + (i - j) ** 2)).sum(),
            "energy": np.sqrt(asm),
            "correlation": (P * (i - mu_i) * (j - mu_j)).sum() / (sd_i * sd_j) if sd_i * sd_j > 1e-9 else 0.0,
            "entropy": -(P[nz] * np.log2(P[nz])).sum()}


# ---------- LNDP (paper Eq. 48-53) ----------
# For each neighbor of the 3x3 ring, subtract its two adjacent ring neighbors; if both differences have
# the same sign the bit is 1 (the neighbor is a local maximum or minimum of the ring). 8 bits => code 0-255.
# Compact summary (our design choice, not from the paper), rotation invariant: number of 1-bits (pop0-8) + entropy and energy.
RING = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]   # 8 neighbors in circular order
POPCOUNT = np.array([bin(i).count("1") for i in range(256)])


def lndp_map(gray):
    g = gray.astype(np.int16)                      # avoids uint8 overflow when subtracting
    H, W = g.shape
    ring = [g[1 + dy:H - 1 + dy, 1 + dx:W - 1 + dx] for dy, dx in RING]
    code = np.zeros((H - 2, W - 2), np.uint16)
    for r in range(8):
        cur, prev, nxt = ring[r], ring[(r - 1) % 8], ring[(r + 1) % 8]
        n1, n2 = prev - cur, nxt - cur
        bit = ((n1 >= 0) & (n2 >= 0)) | ((n1 < 0) & (n2 < 0))
        code |= bit.astype(np.uint16) << r
    return code


def lndp_features(gray, inner):
    code = lndp_map(gray)
    sel = code[inner[1:-1, 1:-1]].astype(int)
    h = np.bincount(sel, minlength=256)[:256] / max(len(sel), 1)
    ph = np.bincount(POPCOUNT, weights=h, minlength=9)[:9]
    nz = h > 0
    out = {f"tex_lndp_pop{k}": float(ph[k]) for k in range(9)}
    out["tex_lndp_entropy"] = float(-(h[nz] * np.log2(h[nz])).sum())
    out["tex_lndp_energy"] = float((h ** 2).sum())
    return out


def texture_features(img, mask):
    out = {}
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    gc, mc = gray[y0:y1, x0:x1], mask[y0:y1, x0:x1]

    # GLCM: level 0 is reserved for "outside the mask" and discarded => the boundary does not count
    q = (1 + (gc.astype(np.int32) * 30) // 255).astype(np.uint8)
    q[~mc] = 0
    P = graycomatrix(q, [1, 2, 4], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=32, symmetric=True, normed=False)
    P = P.astype(np.float64)
    P[0, :, :, :] = 0
    P[:, 0, :, :] = 0
    for di, d in enumerate([1, 2, 4]):
        acc = []
        for ai in range(4):
            M = P[:, :, di, ai]
            s = M.sum()
            acc.append(glcm_props(M / s) if s > 0 else {k: np.nan for k in
                       ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "entropy"]})
        for k in acc[0]:
            out[f"tex_glcm_{k}_d{d}"] = float(np.nanmean([r[k] for r in acc]))

    # uniform LBP (P=8, R=1) inside the lesion
    lbp = local_binary_pattern(gray, 8, 1, "uniform")
    inner = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    hist = np.bincount(lbp[inner].astype(int), minlength=10)[:10] / max(inner.sum(), 1)
    for i, v in enumerate(hist):
        out[f"tex_lbp{i}"] = float(v)

    # LNDP (11 columns)
    out.update(lndp_features(gray, inner))

    # pigment network: ridge filter (Sato) on dark structures
    resp = sato(gc.astype(np.float32) / 255.0, sigmas=(1, 2), black_ridges=True)[mc]
    out["tex_ridge_mean"] = float(resp.mean())
    out["tex_ridge_p95"] = float(np.percentile(resp, 95))
    out["tex_ridge_frac"] = float((resp > RIDGE_THR).mean())

    # dots/globules: dark blobs (LoG) with their center inside the lesion
    blobs = blob_log(1.0 - gc.astype(np.float32) / 255.0, min_sigma=2, max_sigma=6, num_sigma=5, threshold=0.08)
    inner_c = cv2.erode(mc.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    keep = [b for b in blobs if inner_c[int(b[0]), int(b[1])]]
    out["tex_n_blobs"] = len(keep)
    out["tex_blob_density"] = len(keep) / max(mask.sum(), 1) * 1000
    out["tex_blob_sigma_mean"] = float(np.mean([b[2] for b in keep])) if keep else 0.0

    # homogeneity (tex_L_std and tex_gray_std were removed: they duplicated color_L_std)
    h = np.bincount((gray[mask] // 8).astype(int), minlength=32) / mask.sum()
    out["tex_gray_entropy"] = float(-(h[h > 0] * np.log2(h[h > 0])).sum())
    g = gray.astype(np.float32)
    var = cv2.blur(g * g, (5, 5)) - cv2.blur(g, (5, 5)) ** 2
    out["tex_local_std_mean"] = float(np.sqrt(np.maximum(var, 0))[inner].mean())
    return out


# ======================================================================
# Families 4-6 · Whole image (no mask) and Gabor
# ======================================================================
def color_features_full(img, mask=None):
    """Prefix gcolor_ (70). Only the columns that are always NaN without a lesion (lesion-skin and center-periphery) are dropped."""
    full = np.ones(img.shape[:2], bool)
    with warnings.catch_warnings():                  # the dropped columns raise 'empty mean' warnings
        warnings.simplefilter("ignore")
        out = color_features(img, full)
    drop = ("color_rel_", "color_core_minus_periph_")
    return {"g" + k: v for k, v in out.items() if not k.startswith(drop)}


def texture_features_full(img, mask=None):
    """Prefix gtex_ (104): GLCM, LBP, LNDP, ridges, blobs and homogeneity over the whole image."""
    full = np.ones(img.shape[:2], bool)
    return {"g" + k: v for k, v in texture_features(img, full).items()}


# ---------- Gabor: filter bank (wavelengths x orientations), magnitude of the response ----------
GABOR_SIDE = 256                                     # filters are applied to the image downsized to 256 px (much faster)
GABOR_LAMBDAS = (3, 6, 12, 24)                       # wavelength in px at 256 px (= 6, 12, 24, 48 px at 512)
GABOR_THETAS = tuple(np.arange(6) * np.pi / 6)       # 0, 30, ..., 150 degrees


def gabor_bank():
    bank = []
    for lam in GABOR_LAMBDAS:
        sigma = 0.56 * lam
        ks = int(2 * round(3 * sigma) + 1)
        for th in GABOR_THETAS:
            ke = cv2.getGaborKernel((ks, ks), sigma, th, lam, 0.5, 0, ktype=cv2.CV_32F)
            ko = cv2.getGaborKernel((ks, ks), sigma, th, lam, 0.5, np.pi / 2, ktype=cv2.CV_32F)
            ke -= ke.mean()                          # no response to mean brightness (DC)
            bank.append((lam, th, ke, ko))
    return bank


GABOR_BANK = gabor_bank()


def gabor_region_stats(maps, region, prefix, out):
    """maps: list of (lam, th_idx, magnitude HxW). Per-filter mean and std (48) + per-scale summaries (9).

    Summaries (rotation invariant): `energy` = mean response of the scale (mean over the 6 orientations),
    `aniso` = std/mean across orientations, `ratio_fine_coarse` = finest energy / coarsest energy.
    """
    per_lam = {lam: [] for lam in GABOR_LAMBDAS}
    for lam, ti, m in maps:
        v = m[region]
        mu = float(v.mean())
        out[f"{prefix}_l{lam}_t{ti}_mean"] = mu
        out[f"{prefix}_l{lam}_t{ti}_std"] = float(v.std())
        per_lam[lam].append(mu)
    for lam, mus in per_lam.items():
        mus = np.array(mus)
        out[f"{prefix}_l{lam}_energy"] = float(mus.mean())
        out[f"{prefix}_l{lam}_aniso"] = float(mus.std() / max(mus.mean(), 1e-9))
    lams = list(GABOR_LAMBDAS)
    out[f"{prefix}_ratio_fine_coarse"] = float(out[f"{prefix}_l{lams[0]}_energy"] / max(out[f"{prefix}_l{lams[-1]}_energy"], 1e-9))


def gabor_features(img, mask):
    """Returns tex_gabor_* (inside the lesion) and gtex_gabor_* (whole image), computing the filters only once."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (GABOR_SIDE, GABOR_SIDE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    mask = cv2.resize(mask.astype(np.uint8), (GABOR_SIDE, GABOR_SIDE), interpolation=cv2.INTER_NEAREST).astype(bool)
    maps = []
    for lam, th, ke, ko in GABOR_BANK:
        re = cv2.filter2D(gray, cv2.CV_32F, ke, borderType=cv2.BORDER_REFLECT)
        im = cv2.filter2D(gray, cv2.CV_32F, ko, borderType=cv2.BORDER_REFLECT)
        maps.append((lam, int(round(th / (np.pi / 6))), np.sqrt(re * re + im * im)))
    out = {}
    inner = cv2.erode(mask.astype(np.uint8), ell(9)).astype(bool)      # without the border strip (the border gives a high response)
    if inner.sum() < 50:
        inner = mask
    if inner.sum() >= 20:                                                # empty/degenerate mask: whole image only
        gabor_region_stats(maps, inner, "tex_gabor", out)
    gabor_region_stats(maps, np.ones(gray.shape, bool), "gtex_gabor", out)
    return out


# ======================================================================
# Extraction of one image
# ======================================================================
FAMILIES = [("shape", shape_features), ("color", color_features), ("tex", texture_features),
            ("gcolor", color_features_full), ("gtex", texture_features_full), ("gabor", gabor_features)]
FULL_ONLY = ("gcolor", "gtex", "gabor")      # the ones that still work if the mask is unusable


def extract_one(row, class_to_label=None):
    """class_to_label: dict class -> int (multiclass). If None, CLASS_TO_LABEL (binary) is used."""
    cv2.setNumThreads(1)
    base = {"name": row["name"], "class": row["class"], "subtype": row["subtype"],
            "label": (class_to_label or CLASS_TO_LABEL)[row["class"]],
            "qc_seg_ok": row.get("seg_ok", np.nan), "qc_crop_fallback": row.get("crop_fallback", np.nan)}
    feats, err = {}, []
    try:
        img, mask = read_rgb(row["preprocessed"]), read_mask(row["mask"])
        fams = FAMILIES
        if mask.sum() < 50:                                     # empty or degenerate mask
            err.append("mask_empty")
            fams = [f for f in FAMILIES if f[0] in FULL_ONLY]   # but the whole-image features are still computed
        for fam, fn in fams:
            try:
                feats.update(fn(img, mask))
            except Exception as e:                              # a failing family leaves NaN and does not break the batch
                err.append(f"{fam}:{err_info(e)}")
        err += feats.pop("_errors", [])
    except Exception as e:
        err.append(f"read:{err_info(e)}")
    return {**base, **feats, "qc_errors": ";".join(err)}