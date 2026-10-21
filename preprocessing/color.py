"""Step 3: color constancy with Shades of Gray."""
import numpy as np


def estimate_illuminant(img, p=6):
    """Estimated light per channel (Minkowski norm of order p)."""
    x = img.astype(np.float32)
    return np.power(np.mean(np.power(x, p), axis=(0, 1)), 1.0 / p)


def shades_of_gray(img, p=6, normalize_brightness=True, target=195.0):
    x = img.astype(np.float32)
    illum = estimate_illuminant(img, p)
    if normalize_brightness:
        gain = target / (illum + 1e-8)                 # tint AND brightness: the estimated light becomes gray `target`
    else:
        illum_unit = illum / (np.linalg.norm(illum) + 1e-8)
        gain = 1.0 / (illum_unit * np.sqrt(3) + 1e-8)  # tint only (original version)
    return np.clip(x * gain, 0, 255).astype(np.uint8)
