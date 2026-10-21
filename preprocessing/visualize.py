"""Figures: step-by-step preprocessing of one image, a summary of several, and the dataset review sheet."""
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .io_utils import read_mask, read_rgb
from .params import PreprocParams
from .pipeline import preprocess_steps
from .selection import subtype


def tint(img, mask, color=(255, 0, 0), alpha=0.55):
    """Paints `mask` over the image with a semi-transparent color."""
    out = img.copy()
    out[mask] = ((1 - alpha) * out[mask] + alpha * np.array(color)).astype(np.uint8)
    return out


def contour_overlay(img, mask, color=(0, 255, 0), thick=2):
    out = img.copy()
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cnts, -1, color, thick)
    return out


def show_steps(path, P=None, title=None, save_to=None):
    """3 x 5 figure with each step applied on top of the previous one and its parameters in the title. Returns the steps."""
    P = P or PreprocParams()
    path = Path(path)
    img = read_rgb(path)
    s = preprocess_steps(img, P)
    hp = P.hair
    H, W = img.shape[:2]
    lw = max(2, int(0.006 * max(H, W)))

    fig, axs = plt.subplots(3, 5, figsize=(19, 12))
    for a in axs.ravel():
        a.axis("off")

    def show(ax, im, t, **kw):
        ax.imshow(im, **kw)
        ax.set_title(t, fontsize=9)

    # ---- row 1: geometry and color ----
    y0, y1, x0, x1 = s["rect"]
    o = tint(img, ~s["valid"])
    cv2.rectangle(o, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 0), lw)
    show(axs[0, 0], o, f"1. Original {W}x{H}\nblack border: gray < {P.dark_thr} (red)\n"
                       f"useful rectangle (green){' — FALLBACK: not cropped' if s['fallback'] else ''}")
    qy, qx, q = s["square"]
    c = s["cropped"].copy()
    cv2.rectangle(c, (qx, qy), (qx + q - 1, qy + q - 1), (255, 255, 0), max(2, int(0.006 * q)))
    show(axs[0, 1], c, f"1b. Crop {c.shape[1]}x{c.shape[0]}\ncentral square {q}x{q} (yellow)\n"
                       f"(no crop if the useful area < {P.min_area_frac:.0%} of the image)")
    show(axs[0, 2], s["resized"], f"2. Resize to {P.target_size}x{P.target_size}\n"
                                  f"({'INTER_AREA' if q > P.target_size else 'INTER_CUBIC'})")
    r, g, b = s["illum"]
    dest = f"light → gray {P.sog_target:.0f} (tint + brightness)" if P.normalize_brightness else "tint correction only"
    show(axs[0, 3], s["sog"], f"3. Shades of Gray (p = {P.sog_p})\nestimated light RGB = ({r:.0f}, {g:.0f}, {b:.0f})\n{dest}")

    # ---- row 2: hair ----
    side = s["hair_small"].shape[0]
    show(axs[1, 0], s["hair_blackhat"], f"4a. Black-hat ({hp['kernel_shape']}, k = {int(hp['kernel_size']) | 1}) on {hp['channel']}\n"
                                        f"at {side}x{side} px, Gaussian blur {hp['blur_ksize']}", cmap="magma")
    show(axs[1, 1], s["hair_thresholded"], f"4b. Threshold: black-hat > {hp['thresh']}", cmap="gray")
    filt = (f"4c. Shape filter: area ≥ {hp['min_area']}, long side ≥ {hp['kernel_size']}\n"
            f"and (elongation ≥ {hp['min_elongation']} or fill < {hp['max_fill']})") if hp["filter_shape"] \
        else "4c. (no shape filter)"
    show(axs[1, 2], s["hair_filtered"], filt, cmap="gray")
    show(axs[1, 3], tint(s["sog"], s["hair_mask"] > 0), f"4d. Hair mask (dilation x{hp['dilate_iter']}, 3x3)\n"
                                                        f"rescaled to {P.target_size} px: {100 * s['hair_frac']:.1f} % of pixels")
    show(axs[1, 4], s["clean"], f"4e. Inpainting {hp['inpaint_method']}, radius {s['hair_radius']} px\n"
                                f"({hp['inpaint_radius']} px at {P.hair_ref_side} px)")

    # ---- row 3: segmentation ----
    show(axs[2, 0], s["channel"], f"5a. Channel {P.seg_channel} of the hair-free image (no blur)", cmap="gray")
    show(axs[2, 1], s["otsu"], f"5b. Otsu: t = {s['otsu_t']:.0f} (channel normalized to 0-255)\n"
                               f"{'inverted: ' if s['otsu_inverted'] else ''}lesion = the class touching the border least",
         cmap="gray")
    show(axs[2, 2], s["mask"], f"5c. Opening {P.open_k} + closing {P.close_k} (ellipse)\n"
                               f"large central component (σ = {P.center_sigma}) + hole filling", cmap="gray")
    qc = s["qc"]
    show(axs[2, 3], contour_overlay(s["clean"], s["mask"]),
         f"5d. Result: lesion {100 * qc['mask_area_frac']:.0f} %, solidity {qc['mask_solidity']:.2f}\n"
         f"seg_ok = {qc['seg_ok']} (area {P.qc_min_frac:.0%}-{P.qc_max_frac:.0%}, "
         f"solidity ≥ {P.qc_min_solidity}, not touching the border)")

    for ax, label in zip(axs[:, 0], ["Geometry and color", "Hair", "Segmentation"]):
        ax.text(-0.08, 0.5, label, transform=ax.transAxes, rotation=90, ha="right", va="center",
                fontsize=12, weight="bold")
    fig.suptitle(title or path.stem, fontsize=13)
    plt.tight_layout()
    if save_to:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=110, bbox_inches="tight")
    plt.show()
    return s


def show_examples(examples, P=None, save_dir=None):
    """show_steps() for each (split, class, path) and, if there is more than one, a summary figure with one row per image."""
    P = P or PreprocParams()
    steps = {}
    for split, cls, path in examples:
        path = Path(path)
        save_to = Path(save_dir) / f"{split}_{cls}_{path.stem}_steps.png" if save_dir else None
        steps[path.stem] = (cls, show_steps(path, P, f"{path.stem}  —  {split} / {cls} ({subtype(path, cls)})", save_to))
    if len(steps) > 1:
        cols = ["original", "Shades of Gray", "hair mask", "hair removed", "segmentation"]
        fig, axs = plt.subplots(len(steps), len(cols), figsize=(3.2 * len(cols), 3.3 * len(steps)), squeeze=False)
        for row, (name, (cls, s)) in zip(axs, steps.items()):
            ims = [s["original"], s["sog"], tint(s["sog"], s["hair_mask"] > 0), s["clean"],
                   contour_overlay(s["clean"], s["mask"])]
            for ax, im, t in zip(row, ims, cols):
                ax.imshow(im)
                ax.axis("off")
                ax.set_title(t, fontsize=9)
            row[0].set_title(f"{name} ({cls})\noriginal", fontsize=9)
        plt.tight_layout()
        if save_dir:
            fig.savefig(Path(save_dir) / "summary_steps.png", dpi=110, bbox_inches="tight")
        plt.show()
    return {k: v[1] for k, v in steps.items()}


def contact_sheet(df, ncols=6, title=""):
    """Already preprocessed images ('preprocessed' and 'mask' columns of the manifest) with the mask contour."""
    n = len(df)
    nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3.2 * nrows), squeeze=False)
    for a in axs.ravel():
        a.axis("off")
    for a, (_, r) in zip(axs.ravel(), df.iterrows()):
        a.imshow(contour_overlay(read_rgb(r["preprocessed"]), read_mask(r["mask"])))
        a.set_title(f"{r['name']}\n{100 * r['mask_area_frac']:.0f}%", fontsize=8)
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
