"""Image selection stratified by class and subtype (keeping the real proportions of the dataset)."""
import random
import re

import numpy as np

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def subtype(path, cls):
    """Alphabetic prefix of the file name (e.g. 'mel', 'nev', 'bcc'); if there is none, the class."""
    m = re.match(r"[A-Za-z]+", path.stem)
    return m.group(0).lower() if m else cls


def allocate(counts, n):
    """Proportional allocation (largest remainder method) that sums exactly to n."""
    total = sum(counts.values())
    exact = {k: n * v / total for k, v in counts.items()}
    alloc = {k: int(np.floor(v)) for k, v in exact.items()}
    rest = n - sum(alloc.values())
    for k in sorted(exact, key=lambda k: exact[k] - alloc[k], reverse=True)[:rest]:
        alloc[k] += 1
    return alloc


def select_images(class_dir, classes, split, n, seed, exts=EXTS):
    """List of (split, class, path). `class_dir(split, cls)` returns the folder of that class.
    n=None -> all images; otherwise n images stratified by (class, subtype) with a fixed seed."""
    strata = {}
    for c in classes:
        for f in sorted(class_dir(split, c).rglob("*")):
            if f.suffix.lower() in exts:
                strata.setdefault((c, subtype(f, c)), []).append(f)
    if n is None:
        return [(split, c, f) for (c, _), fs in strata.items() for f in fs]
    alloc = allocate({k: len(v) for k, v in strata.items()}, n)
    rng = random.Random(seed)
    picked = []
    for k in sorted(strata):
        fs = strata[k][:]
        rng.shuffle(fs)
        picked += [(split, k[0], f) for f in fs[:alloc[k]]]
    return picked


def pick_examples(jobs, classes, n=5, seed=42, names=None, split="train"):
    """Example images for the visualization: specific `names`, or `n` random images in total,
    spread as evenly as possible across the classes (class by class in turn: with 5 and 3 classes -> 2, 2, 1)."""
    if names:
        wanted = set(names)
        sel = [j for j in jobs if j[2].stem in wanted]
        missing = wanted - {j[2].stem for j in sel}
        if missing:
            print("Not found:", sorted(missing))
        return sel
    rng = random.Random(seed)
    pools = []
    for c in classes:
        pool = [j for j in jobs if j[0] == split and j[1] == c]
        rng.shuffle(pool)
        pools.append(pool)
    sel = []
    while len(sel) < n and any(pools):
        for pool in pools:
            if pool and len(sel) < n:
                sel.append(pool.pop())
    return sel
