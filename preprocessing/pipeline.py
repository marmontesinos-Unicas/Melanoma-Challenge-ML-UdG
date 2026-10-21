"""Full pipeline: one image (with or without intermediate steps) and the whole dataset (parallel and resumable)."""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .color import estimate_illuminant, shades_of_gray
from .geometry import central_square, crop_black_borders, square_resize, valid_mask
from .hair import remove_hair_stages
from .io_utils import read_mask, read_rgb, write_png
from .params import PreprocParams
from .segmentation import clean_mask, get_channel, mask_qc, otsu_stages
from .selection import subtype


def preprocess_steps(img, P=None):
    """Applies the pipeline to an RGB image and keeps every intermediate step (dict).
    preprocess() uses this same function, so what is visualized is exactly what is saved."""
    P = P or PreprocParams()
    s = {"original": img}
    # 1. black borders
    s["valid"] = valid_mask(img, P.dark_thr)
    cropped, rect, fallback = crop_black_borders(img, dark_thr=P.dark_thr, min_area_frac=P.min_area_frac)
    s.update(cropped=cropped, rect=rect, fallback=fallback)
    # 2. central square + resize
    s["square"] = central_square(cropped.shape)
    s["resized"] = square_resize(cropped, P.target_size)
    # 3. Shades of Gray
    s["illum"] = estimate_illuminant(s["resized"], P.sog_p)
    s["sog"] = shades_of_gray(s["resized"], P.sog_p, P.normalize_brightness, P.sog_target)
    # 4. hair removal (no blur)
    h = remove_hair_stages(s["sog"], P.hair, P.hair_ref_side)
    s.update({f"hair_{k}": v for k, v in h.items() if k != "clean"})
    s["clean"] = h["clean"]
    s["hair_frac"] = float((h["mask"] > 0).mean())
    # 5. segmentation: Otsu(channel) directly on the hair-free image
    s["channel"] = get_channel(s["clean"], P.seg_channel)
    s["otsu"], s["otsu_t"], s["otsu_inverted"] = otsu_stages(s["channel"])
    s["mask"] = clean_mask(s["otsu"], P.open_k, P.close_k, P.center_sigma)
    s["qc"] = mask_qc(s["mask"], P.qc_min_frac, P.qc_max_frac, P.qc_min_solidity)
    return s


def preprocess(img, P=None):
    """Returns (hair-free RGB image, bool mask, crop rect, hair fraction, fallback)."""
    s = preprocess_steps(img, P)
    return s["clean"], s["mask"], s["rect"], s["hair_frac"], s["fallback"]


def process_one(split, cls, path, out_root, P, overwrite=False):
    """Processes one image and saves <name>_preprocessed.png and <name>_mask.png. Returns its manifest row."""
    cv2.setNumThreads(1)                              # avoids oversubscription with joblib
    path = Path(path)
    out_dir = Path(out_root) / split / cls
    out_dir.mkdir(parents=True, exist_ok=True)
    p_img = out_dir / f"{path.stem}_preprocessed.png"
    p_msk = out_dir / f"{path.stem}_mask.png"
    row = {"split": split, "class": cls, "subtype": subtype(path, cls), "name": path.stem,
           "src": str(path), "preprocessed": str(p_img), "mask": str(p_msk)}
    try:
        if p_img.exists() and p_msk.exists() and not overwrite:
            row.update(mask_qc(read_mask(p_msk), P.qc_min_frac, P.qc_max_frac, P.qc_min_solidity))
            row["status"] = "skipped"
            return row
        img = read_rgb(path)
        img_feat, mask, (y0, y1, x0, x1), hair_frac, fallback = preprocess(img, P)
        write_png(p_img, cv2.cvtColor(img_feat, cv2.COLOR_RGB2BGR))
        write_png(p_msk, (mask.astype(np.uint8) * 255))
        row.update({"orig_h": img.shape[0], "orig_w": img.shape[1],
                    "crop_y0": y0, "crop_y1": y1, "crop_x0": x0, "crop_x1": x1,
                    "crop_fallback": fallback, "hair_frac": hair_frac,
                    **mask_qc(mask, P.qc_min_frac, P.qc_max_frac, P.qc_min_solidity), "status": "ok"})
    except Exception as e:                            # one broken image does not stop the batch
        row["status"] = f"error: {e}"
    return row


def run_dataset(jobs, out_root, P, overwrite=False, n_jobs=4):
    """Processes every image in `jobs` [(split, class, path)] in parallel. Returns the list of rows."""
    from joblib import Parallel, delayed
    from tqdm.auto import tqdm
    gen = Parallel(n_jobs=n_jobs, return_as="generator")(
        delayed(process_one)(s, c, f, out_root, P, overwrite) for s, c, f in jobs)
    return [r for r in tqdm(gen, total=len(jobs))]


def update_manifest(rows, manifest_path):
    """Merges the new rows with the previous manifest.csv (if any), saves it and returns it."""
    new = pd.DataFrame(rows)
    manifest_path = Path(manifest_path)
    if manifest_path.exists():           # keeps full data (hair_frac, crop...) from previous runs
        old = pd.read_csv(manifest_path)
        new = pd.concat([old, new], ignore_index=True)
        new["_skipped"] = new["status"].eq("skipped")          # 'skipped' rows have less data: lower priority
        new = (new.sort_values("_skipped", kind="stable")
                  .drop_duplicates(subset=["split", "class", "name"], keep="first")
                  .drop(columns="_skipped"))
    new = new.sort_values(["split", "class", "name"]).reset_index(drop=True)
    new.to_csv(manifest_path, index=False)
    return new


def summarize_manifest(df):
    """Prints statuses, a summary per split/class, fallbacks and errors."""
    print(df["status"].str.split(":").str[0].value_counts().to_string(), "\n")
    print(df.groupby(["split", "class"]).agg(n=("name", "size"),
                                             seg_ok=("seg_ok", "mean"),
                                             area_frac_med=("mask_area_frac", "median"),
                                             hair_frac_med=("hair_frac", "median")).round(3).to_string())
    if "crop_fallback" in df:
        print("\nNot cropped (fallback):", int(df["crop_fallback"].fillna(False).astype(bool).sum()))
    errs = df[df["status"].str.startswith("error")]
    if len(errs):
        print("\nImages with errors:")
        print(errs[["split", "class", "name", "status"]].to_string(index=False))
