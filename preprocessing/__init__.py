"""Preprocessing of dermoscopic images, shared by Binary/ and Multiclass/.

1. Crop the black borders of the dermatoscope    -> geometry.crop_black_borders
2. Central square + resize                       -> geometry.square_resize
3. Color constancy (Shades of Gray)              -> color.shades_of_gray
4. Hair removal (black-hat + inpainting)         -> hair.remove_hair
5. Segmentation (Otsu on one channel + cleaning) -> segmentation.segment

Usage from a notebook in Binary/ or Multiclass/ (with the repo folder in sys.path):

    from preprocessing import PreprocParams, preprocess, show_examples, run_dataset
"""
from .color import estimate_illuminant, shades_of_gray
from .explore import (clahe_rgb, color_methods, explore_color, explore_hair, explore_morphology,
                      explore_seg_channels, explore_seg_inputs, explore_seg_methods, final_summary,
                      hair_fraction, hair_variants, load_sample, pick_sample, print_params, seg_inputs, seg_methods,
                      show_borders_resize, stage_color, stage_crop, stage_hair, stage_resize)
from .geometry import central_square, crop_black_borders, largest_rectangle, square_resize, valid_mask
from .hair import blackhat_response, hair_mask_stages, remove_hair, remove_hair_stages
from .io_utils import read_mask, read_rgb, write_png
from .params import PreprocParams, default_hair
from .pipeline import preprocess, preprocess_steps, process_one, run_dataset, summarize_manifest, update_manifest
from .segmentation import clean_mask, get_channel, mask_qc, otsu_mask, otsu_stages, segment
from .selection import EXTS, allocate, pick_examples, select_images, subtype
from .visualize import contact_sheet, contour_overlay, show_examples, show_steps, tint
