"""Image read / write (np.fromfile + imdecode: works with non-ASCII paths on Windows)."""
from pathlib import Path

import cv2
import numpy as np


def read_rgb(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError(f"Could not read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_mask(path):
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_GRAYSCALE) > 0


def write_png(path, arr):
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise IOError(f"Could not encode {path}")
    buf.tofile(str(Path(path)))
