"""
Discover Tecplot partition files on disk and group them by iteration.

Expects filenames of the form: tecplot_blk<BLOCK>_iter<ITER>.dat
"""

import os
import re
import glob
from collections import defaultdict

ALL_VAR_NAMES = ["X", "Y", "Z", "U", "V", "W", "P", "T", "rho", "mu", "beta", "cp", "kappa"]
FNAME_RE = re.compile(r"tecplot_blk(\d+)_iter(\d+)\.dat$")


def discover_iterations(input_dir, pattern="tecplot_blk*_iter*.dat"):
    """
    Scan input_dir for block files and group them by iteration number.

    Returns: dict {iter_number (int): [(block_number, filepath), ...]},
             sorted by iteration number, with each block list sorted by
             block number.
    """
    files = glob.glob(os.path.join(input_dir, pattern))
    by_iter = defaultdict(list)

    for fp in files:
        m = FNAME_RE.search(os.path.basename(fp))
        if not m:
            print(f"Skipping file that doesn't match naming pattern: {fp}")
            continue
        blk, it = int(m.group(1)), int(m.group(2))
        by_iter[it].append((blk, fp))

    for it in by_iter:
        by_iter[it].sort(key=lambda x: x[0])

    return dict(sorted(by_iter.items()))
