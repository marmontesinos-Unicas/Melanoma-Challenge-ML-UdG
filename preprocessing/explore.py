"""Exploration of alternatives for each preprocessing step (used by the notebook; nothing is saved).

Every explore_* function only draws a figure: the comparison is purely visual (no ground-truth masks, no metrics).
The stage_* functions apply the DECIDED values of one block with the production code, so the next block receives
exactly what the real pipeline would produce. scikit-learn / scikit-image are only imported inside the functions that need them.
"""
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .color import estimate_illuminant, shades_of_gray
from .geometry import central_square, crop_black_borders, square_resize, valid_mask
from .hair import hair_mask_stages, remove_hair_stages
from .io_utils import read_rgb
from .params import PreprocParams, default_hair
from .pipeline import preprocess_steps
from .segmentation import border_ring, clean_mask, get_channel, otsu_stages, segment
from .selection import pick_examples
from .visualize import contour_overlay, tint


# ---------------------------------------------------------------- helpers
def _grid(nrows, ncols, cell=(2.8, 3.0)):
    fig, axs = plt.subplots(nrows, ncols, figsize=(cell[0] * ncols, cell[1] * nrows), squeeze=False)
    for a in axs.ravel():
        a.axis("off")
    return fig, axs


def _show(ax, im, title, **kw):
    ax.imshow(im, **kw)
    ax.set_title(title, fontsize=8)


def _row_label(ax, text):
    ax.text(-0.04, 0.5, text, transform=ax.transAxes, rotation=90, ha="right", va="center", fontsize=9, weight="bold")


def _finish(fig, title=None):
    if title:
        fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------- sample
def hair_fraction(img, hair=None, ref_side=256):
    """Quick fraction of hair pixels (default hair parameters, central square at ref_side px, no color correction). Only used to rank images."""
    hair = hair or default_hair()
    bgr = cv2.cvtColor(square_resize(img, ref_side), cv2.COLOR_RGB2BGR)
    kw = {k: v for k, v in hair.items() if k not in ("inpaint_radius", "inpaint_method")}
    return float((hair_mask_stages(bgr, **kw)["mask"] > 0).mean())


def pick_sample(jobs, classes, n_random=3, n_hairy=2, seed=42, pool=60, split="train"):
    """n_random images spread across classes + the n_hairy hairiest of a random pool of other images. Returns [(split, class, path)]."""
    chosen = pick_examples(jobs, classes, n=n_random, seed=seed, split=split)
    taken = {j[2] for j in chosen}
    cand = [j for j in jobs if j[0] == split and j[2] not in taken]
    cand = random.Random(seed + 1).sample(cand, min(pool, len(cand)))
    scored = sorted(((hair_fraction(read_rgb(j[2])), j) for j in cand), key=lambda t: t[0], reverse=True)
    hairy = [j for _, j in scored[:n_hairy]]
    for tag, group in (("random", chosen), ("hairy", hairy)):
        for s, c, p in group:
            print(f"{tag:6s} {s}/{c}/{Path(p).stem}")
    return chosen + hairy


def load_sample(sample):
    """{name: RGB image} from [(split, class, path)]."""
    return {Path(p).stem: read_rgb(p) for _, _, p in sample}


# ---------------------------------------------------------------- stages (decided values -> input of the next block)
def stage_crop(images, P):
    return {n: crop_black_borders(im, P.dark_thr, P.min_area_frac)[0] for n, im in images.items()}


def stage_resize(cropped, P):
    return {n: square_resize(im, P.target_size) for n, im in cropped.items()}


def stage_color(resized, P):
    return {n: shades_of_gray(im, P.sog_p, P.normalize_brightness, P.sog_target) for n, im in resized.items()}


def stage_hair(sog, P):
    return {n: remove_hair_stages(im, P.hair, P.hair_ref_side)["clean"] for n, im in sog.items()}


# ---------------------------------------------------------------- block 1: borders and resize (only the chosen version)
def show_borders_resize(images, P):
    """Chosen borders + resize: detected border (red) and useful rectangle (green), central square (yellow), resized image."""
    fig, axs = _grid(len(images), 3)
    for r, (name, img) in enumerate(images.items()):
        H, W = img.shape[:2]
        cropped, (y0, y1, x0, x1), fb = crop_black_borders(img, P.dark_thr, P.min_area_frac)
        o = tint(img, ~valid_mask(img, P.dark_thr))
        cv2.rectangle(o, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 0), max(2, int(0.006 * max(H, W))))
        note = "\nFALLBACK: not cropped" if fb else ""
        _show(axs[r, 0], o, f"original: black border (red), useful rectangle (green){note}")
        qy, qx, q = central_square(cropped.shape)
        c = cropped.copy()
        cv2.rectangle(c, (qx, qy), (qx + q - 1, qy + q - 1), (255, 255, 0), max(2, int(0.006 * q)))
        _show(axs[r, 1], c, "crop: central square (yellow)")
        _show(axs[r, 2], square_resize(cropped, P.target_size), f"resized to {P.target_size}x{P.target_size}")
        _row_label(axs[r, 0], name)
    _finish(fig, "1. Black borders and resize")


# ---------------------------------------------------------------- block 2: color constancy
def _apply_gain(img, illum, normalize, target):
    x = img.astype(np.float32)
    if normalize:
        gain = target / (illum + 1e-8)
    else:
        unit = illum / (np.linalg.norm(illum) + 1e-8)
        gain = 1.0 / (unit * np.sqrt(3) + 1e-8)
    return np.clip(x * gain, 0, 255).astype(np.uint8)


def color_methods(target=195.0):
    """{name: fn(img) -> corrected img}."""
    def gray_world(img):
        return _apply_gain(img, img.reshape(-1, 3).astype(np.float32).mean(0), True, target)

    def white_patch(img):
        illum = np.percentile(img.reshape(-1, 3), 99, axis=0).astype(np.float32)   # p99 instead of max: robust to hot pixels
        return _apply_gain(img, illum, True, target)

    def sog(p, normalize=True):
        return lambda img: _apply_gain(img, estimate_illuminant(img, p), normalize, target)

    return {"none": lambda img: img, "Gray World": gray_world, "White Patch (p99)": white_patch,
            "SoG p=3": sog(3), "SoG p=6": sog(6), "SoG p=10": sog(10), "SoG p=6 tint only": sog(6, False)}


def explore_color(resized, target=195.0):
    """Every color-constancy method on every image (light mapped to gray `target` unless 'tint only')."""
    methods = color_methods(target)
    fig, axs = _grid(len(resized), len(methods), cell=(2.6, 2.8))
    for r, (name, img) in enumerate(resized.items()):
        _row_label(axs[r, 0], name)
        for c, (mname, fn) in enumerate(methods.items()):
            _show(axs[r, c], fn(img), mname)
    _finish(fig, "2. Color constancy")


# ---------------------------------------------------------------- block 3: hair
def hair_variants():
    """{name: (overrides of default_hair(), ref_side)}. ref_side=0 -> detect at full size (parameters scaled x2 from 256 to 512)."""
    return {"baseline\ncross k5, t15": ({}, 256),
            "lines k11": ({"kernel_shape": "lines", "kernel_size": 11}, 256),
            "cross k9": ({"kernel_size": 9}, 256),
            "cross k13": ({"kernel_size": 13}, 256),
            "thresh 10": ({"thresh": 10}, 256),
            "thresh 25": ({"thresh": 25}, 256),
            "max_rgb": ({"channel": "max_rgb"}, 256),
            "no shape filter": ({"filter_shape": False}, 256),
            "detect at 512\n(k11, area 80, r10)": ({"kernel_size": 11, "min_area": 80, "inpaint_radius": 10}, 0)}


def explore_hair(sog, variants=None):
    """One figure per image: hair mask (red) and inpainted result for each variant."""
    variants = variants or hair_variants()
    for name, img in sog.items():
        fig, axs = _grid(2, len(variants), cell=(2.5, 2.7))
        for c, (vname, (over, ref)) in enumerate(variants.items()):
            st = remove_hair_stages(img, {**default_hair(), **over}, ref)
            _show(axs[0, c], tint(img, st["mask"] > 0), vname)
            _show(axs[1, c], st["clean"], "result")
        _row_label(axs[0, 0], "mask")
        _row_label(axs[1, 0], "result")
        _finish(fig, f"3. Hair removal: {name}")


# ---------------------------------------------------------------- block 4: segmentation
def clahe_rgb(img, clip=2.0, tile=8):
    """CLAHE on the L channel of Lab (colors untouched)."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def seg_inputs(img):
    """Candidate inputs for the segmentation: as is, median blur, CLAHE."""
    return {"no blur": img, "median 5": cv2.medianBlur(img, 5), "CLAHE": clahe_rgb(img)}


def _foreground(bw):
    """Orients a binary mask so the lesion is the class that touches the border least (same rule as otsu_stages)."""
    return ~bw if bw[border_ring(bw.shape)].mean() > 0.5 else bw


def _resize_to(img, side, interp=cv2.INTER_AREA):
    h, w = img.shape[:2]
    r = side / min(h, w)
    return cv2.resize(img, (round(w * r), round(h * r)), interpolation=interp)


def _kmeans_mask(img):
    from sklearn.cluster import KMeans
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    km = KMeans(n_clusters=2, n_init=3, random_state=0).fit(lab[::16])       # fit on a subsample, predict every pixel
    return clean_mask(_foreground(km.predict(lab).reshape(img.shape[:2]) == 1))


def _grabcut_mask(img, side=256, iters=5, margin=0.1):
    small = cv2.cvtColor(_resize_to(img, side), cv2.COLOR_RGB2BGR)
    H, W = small.shape[:2]
    mx, my = int(margin * W), int(margin * H)
    m = np.zeros((H, W), np.uint8)
    cv2.grabCut(small, m, (mx, my, W - 2 * mx, H - 2 * my), np.zeros((1, 65)), np.zeros((1, 65)), iters, cv2.GC_INIT_WITH_RECT)
    fg = ((m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD)).astype(np.uint8)
    fg = cv2.resize(fg, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    return clean_mask(fg)


def _watershed_mask(img, channel="S"):
    from skimage.filters import gaussian, sobel
    from skimage.segmentation import watershed
    ch = get_channel(img, channel).astype(np.float32) / 255.0
    H, W = ch.shape
    markers = np.zeros((H, W), np.int32)
    markers[border_ring((H, W), 0.04)] = 1                                   # background: image border
    yy, xx = np.ogrid[:H, :W]
    markers[(yy - H / 2) ** 2 + (xx - W / 2) ** 2 <= (0.05 * min(H, W)) ** 2] = 2   # lesion: small central disk
    return clean_mask(watershed(sobel(gaussian(ch, 2)), markers) == 2)


def _chanvese_mask(img, channel="S", side=128):
    from skimage.segmentation import chan_vese
    ch = _resize_to(get_channel(img, channel), side).astype(np.float64) / 255.0
    seg = chan_vese(ch, mu=0.25, max_num_iter=200, init_level_set="checkerboard")
    seg = cv2.resize(seg.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    return clean_mask(_foreground(seg))


def seg_methods(channel="S"):
    """{name: fn(img) -> bool mask}. All of them end with the same clean_mask()."""
    return {f"Otsu({channel})": lambda im: segment(im, channel), "K-means (Lab, k=2)": _kmeans_mask,
            "GrabCut": _grabcut_mask, "Watershed": lambda im: _watershed_mask(im, channel),
            "Chan-Vese": lambda im: _chanvese_mask(im, channel)}


def explore_seg_channels(clean, channels=("L", "a", "b", "Cr", "S", "gray")):
    """Otsu + cleaning on each channel (hair-free images, no blur). Green = mask contour."""
    fig, axs = _grid(len(clean), len(channels))
    for r, (name, img) in enumerate(clean.items()):
        _row_label(axs[r, 0], name)
        for c, ch in enumerate(channels):
            _show(axs[r, c], contour_overlay(img, segment(img, ch)), f"Otsu({ch})")
    _finish(fig, "4a. Otsu channel")


def explore_seg_methods(clean, channel="S"):
    """Otsu vs K-means vs GrabCut vs Watershed vs Chan-Vese (hair-free images, no blur). Green = mask contour."""
    methods = seg_methods(channel)
    fig, axs = _grid(len(clean), len(methods))
    for r, (name, img) in enumerate(clean.items()):
        _row_label(axs[r, 0], name)
        for c, (mname, fn) in enumerate(methods.items()):
            _show(axs[r, c], contour_overlay(img, fn(img)), mname)
    _finish(fig, "4b. Segmentation method")


def explore_seg_inputs(clean, channel="S"):
    """Otsu(channel) on the image as is, with median blur and with CLAHE: channel image and resulting contour (drawn on the hair-free image)."""
    kinds = list(seg_inputs(next(iter(clean.values()))))
    fig, axs = _grid(len(clean), 2 * len(kinds))
    for r, (name, img) in enumerate(clean.items()):
        _row_label(axs[r, 0], name)
        for k, (kind, inp) in enumerate(seg_inputs(img).items()):
            ch = get_channel(inp, channel)
            mask = clean_mask(otsu_stages(ch)[0])
            _show(axs[r, 2 * k], ch, f"{kind}: channel {channel}", cmap="gray")
            _show(axs[r, 2 * k + 1], contour_overlay(img, mask), f"{kind}: Otsu mask")
    _finish(fig, "4c. Segmentation input")


def explore_morphology(clean, channel="S", variants=None):
    """Raw Otsu vs different opening/closing sizes (open_k, close_k); 1 = that operation is skipped."""
    variants = variants or {"raw Otsu": None, "open 5 only": (5, 1), "close 15 only": (1, 15),
                            "open 5 + close 15": (5, 15), "open 3 + close 9": (3, 9), "open 7 + close 21": (7, 21)}
    fig, axs = _grid(len(clean), len(variants))
    for r, (name, img) in enumerate(clean.items()):
        bw = otsu_stages(get_channel(img, channel))[0]
        _row_label(axs[r, 0], name)
        for c, (vname, k) in enumerate(variants.items()):
            mask = bw if k is None else clean_mask(bw, k[0], k[1])
            _show(axs[r, c], contour_overlay(img, mask), vname)
    _finish(fig, "4d. Morphological cleaning")


# ---------------------------------------------------------------- block 5: final result
def print_params(P=None):
    P = P or PreprocParams()
    print(P.table().to_string(index=False))


def final_summary(sample, P=None):
    """One row per image of the sample with the final pipeline: original, color, hair mask, hair removed, segmentation."""
    P = P or PreprocParams()
    cols = ["original", "Shades of Gray", "hair mask", "hair removed", "segmentation"]
    fig, axs = _grid(len(sample), len(cols), cell=(3.2, 3.3))
    for r, (split, cls, path) in enumerate(sample):
        s = preprocess_steps(read_rgb(path), P)
        ims = [s["original"], s["sog"], tint(s["sog"], s["hair_mask"] > 0), s["clean"], contour_overlay(s["clean"], s["mask"])]
        for c, (im, t) in enumerate(zip(ims, cols)):
            _show(axs[r, c], im, t)
        _row_label(axs[r, 0], f"{Path(path).stem} ({cls})")
    _finish(fig, "5. Final preprocessing")
