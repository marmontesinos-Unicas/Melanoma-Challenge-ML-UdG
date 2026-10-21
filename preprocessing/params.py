"""Preprocessing parameters (shared by Binary and Multiclass)."""
from dataclasses import asdict, dataclass, field

import pandas as pd


def default_hair():
    # teammate's method: values from her preproc_params.json; the rest are her function's defaults
    return dict(kernel_size=5,            # black-hat kernel width: must be larger than the hair thickness
                thresh=15,                 # minimum black-hat value to count as hair
                kernel_shape="cross",      # "cross", "rect", "ellipse" or "lines"
                channel="gray",            # "gray", "L" or "max_rgb"
                blur_ksize=3,              # smooths the black-hat map to join broken hairs
                filter_shape=True,         # discards round blobs (dots, lesion borders)
                min_elongation=3.0,
                min_area=20,
                max_fill=0.35,
                dilate_iter=1,
                inpaint_radius=5,
                inpaint_method="telea")    # "telea" or "ns"


@dataclass
class PreprocParams:
    # 1. black borders of the dermatoscope
    dark_thr: int = 35                 # gray < dark_thr = black
    min_area_frac: float = 0.08        # if the useful rectangle is smaller than this fraction, no crop is done
    # 2. central square + resize
    target_size: int = 512
    # 3. color constancy (Shades of Gray)
    sog_p: int = 6
    normalize_brightness: bool = True  # True = the estimated light is mapped to gray sog_target; False = only the tint is corrected
    sog_target: float = 195.0
    # 4. hair removal
    hair: dict = field(default_factory=default_hair)
    hair_ref_side: int = 256           # hair is detected at this size (where the values were tuned) and inpainted at target_size
    # 5. segmentation
    seg_channel: str = "S"             # channel for Otsu: "S", "L", "a", "b", "Cr", "gray"
    open_k: int = 5                    # opening (ellipse) to remove noise
    close_k: int = 15                  # closing (ellipse) to fill gaps
    center_sigma: float = 0.35         # weight of closeness to the center when choosing the component
    # mask quality control (flag only, nothing is discarded)
    qc_min_frac: float = 0.01
    qc_max_frac: float = 0.60
    qc_min_solidity: float = 0.75

    def table(self):
        """(step, parameter, value) table to print in the notebook."""
        step = {"dark_thr": "1. Borders", "min_area_frac": "1. Borders", "target_size": "2. Resize",
                "sog_p": "3. Color (Shades of Gray)", "normalize_brightness": "3. Color (Shades of Gray)",
                "sog_target": "3. Color (Shades of Gray)", "hair_ref_side": "4. Hair",
                "seg_channel": "5. Segmentation", "open_k": "5. Segmentation", "close_k": "5. Segmentation",
                "center_sigma": "5. Segmentation", "qc_min_frac": "5. Quality control",
                "qc_max_frac": "5. Quality control", "qc_min_solidity": "5. Quality control"}
        rows = []
        for k, v in asdict(self).items():        # class order = pipeline order
            if k == "hair":
                rows += [("4. Hair", f"hair.{hk}", hv) for hk, hv in v.items()]
            else:
                rows.append((step[k], k, v))
        return pd.DataFrame(rows, columns=["step", "parameter", "value"])
