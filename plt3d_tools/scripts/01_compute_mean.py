#!/usr/bin/env python3
"""
01_compute_mean.py
-------------------
Compute time-mean u, v, w, T fields from a folder of Plot3D solution
snapshots ("grid<block>.<time>.f" files).

Streams one snapshot at a time per worker (bounded RAM) and splits the file
list across worker processes (parallel).

Example
-------
    python 01_compute_mean.py \
        --solution-dir /media/user/Data2/kiran_data/Ra_2e9/Case/output \
        --output-dir   /media/user/Data2/kiran_data/post_proc_1-mean \
        --tag 731-931 \
        --workers 8
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plt_3d_tools.io import read_solution, list_solution_files
from plt_3d_tools.parallel import run_parallel_accumulation


def _init(shape):
    # float64 accumulators: cheap here (only 4 fields) and keeps the running
    # sum over ~thousands of snapshots numerically accurate.
    return {k: np.zeros(shape, dtype=np.float64) for k in ('u', 'v', 'w', 'T')}


def _process(fn, state, shape):
    sol, nx, ny, nz, nvar, var_names = read_solution(fn)
    state['u'] += sol[:, :, :, 0]
    state['v'] += sol[:, :, :, 1]
    state['w'] += sol[:, :, :, 2]
    state['T'] += sol[:, :, :, 4]
    del sol


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--solution-dir', required=True, help='Directory containing grid*.f snapshot files')
    ap.add_argument('--output-dir', required=True, help='Directory to write mean_*.npy into')
    ap.add_argument('--tag', default='', help='Optional suffix for output filenames, e.g. "731-931"')
    ap.add_argument('--workers', type=int, default=None, help='Worker processes (default: CPU count - 1)')
    ap.add_argument('--save-sum', action='store_true', help='Also save the raw (unnormalized) sums')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    files = list_solution_files(args.solution_dir)
    if not files:
        sys.exit(f"No grid*.f files found in {args.solution_dir}")
    print(f"Found {len(files)} snapshot files in {args.solution_dir}")

    sol0, nx, ny, nz, nvar, var_names = read_solution(files[0])
    shape = (nx, ny, nz)
    del sol0

    state = run_parallel_accumulation(files, _init, _process, n_workers=args.workers, extra=(shape,))

    count = len(files)
    suffix = f"_{args.tag}" if args.tag else ""
    for name in ('u', 'v', 'w', 'T'):
        if args.save_sum:
            np.save(os.path.join(args.output_dir, f"sum_{name}{suffix}.npy"), state[name])
        mean = (state[name] / count).astype(np.float32)
        np.save(os.path.join(args.output_dir, f"mean_{name}{suffix}.npy"), mean)

    print(f"Done. Wrote mean_{{u,v,w,T}}{suffix}.npy to {args.output_dir} ({count} snapshots averaged)")


if __name__ == '__main__':
    main()
