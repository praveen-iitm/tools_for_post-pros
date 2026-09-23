#!/usr/bin/env python3
"""
02_compute_fluctuations_tke.py
--------------------------------
Compute time-averaged turbulent kinetic energy (TKE) from velocity
fluctuations, against two reference means:

  TKE_t  : fluctuation relative to the full 3D time-mean  (mean_u/v/w)
  TKE_tz : fluctuation relative to the z-averaged time-mean (broadcast
           back over z), i.e. homogeneous-direction-averaged mean

Requires mean_u/v/w_*.npy from 01_compute_mean.py.

Example
-------
    python 02_compute_fluctuations_tke.py \
        --solution-dir /media/user/Data2/kiran_data/Ra_2e9/Case/output \
        --mean-dir     /media/user/Data2/kiran_data/post_proc_1-mean \
        --mean-tag 731-931 \
        --output-dir   /media/user/Data2/kiran_data/post_proc_2-fluc \
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


def _init(shape, u_mean_t, v_mean_t, w_mean_t, u_mean_tz, v_mean_tz, w_mean_tz):
    return {
        'TKE_t': np.zeros(shape, dtype=np.float64),
        'TKE_tz': np.zeros(shape, dtype=np.float64),
    }


def _process(fn, state, shape, u_mean_t, v_mean_t, w_mean_t, u_mean_tz, v_mean_tz, w_mean_tz):
    sol, nx, ny, nz, nvar, var_names = read_solution(fn)
    u = sol[:, :, :, 0]
    v = sol[:, :, :, 1]
    w = sol[:, :, :, 2]
    del sol

    u_f = u - u_mean_t
    v_f = v - v_mean_t
    w_f = w - w_mean_t
    state['TKE_t'] += u_f ** 2 + v_f ** 2 + w_f ** 2
    del u_f, v_f, w_f

    u_fz = u - u_mean_tz
    v_fz = v - v_mean_tz
    w_fz = w - w_mean_tz
    state['TKE_tz'] += u_fz ** 2 + v_fz ** 2 + w_fz ** 2
    del u_fz, v_fz, w_fz, u, v, w


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--solution-dir', required=True)
    ap.add_argument('--mean-dir', required=True, help='Directory with mean_u/v/w_*.npy from script 01')
    ap.add_argument('--mean-tag', default='')
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--workers', type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    suffix = f"_{args.mean_tag}" if args.mean_tag else ""
    u_mean_t = np.load(os.path.join(args.mean_dir, f"mean_u{suffix}.npy"))
    v_mean_t = np.load(os.path.join(args.mean_dir, f"mean_v{suffix}.npy"))
    w_mean_t = np.load(os.path.join(args.mean_dir, f"mean_w{suffix}.npy"))
    shape = u_mean_t.shape
    nz = shape[2]

    def z_avg_broadcast(a):
        return np.repeat(np.mean(a, axis=2)[..., np.newaxis], nz, axis=2).astype(np.float32)

    u_mean_tz = z_avg_broadcast(u_mean_t)
    v_mean_tz = z_avg_broadcast(v_mean_t)
    w_mean_tz = z_avg_broadcast(w_mean_t)

    files = list_solution_files(args.solution_dir)
    if not files:
        sys.exit(f"No grid*.f files found in {args.solution_dir}")
    print(f"Found {len(files)} snapshot files in {args.solution_dir}")

    extra = (shape, u_mean_t, v_mean_t, w_mean_t, u_mean_tz, v_mean_tz, w_mean_tz)
    state = run_parallel_accumulation(files, _init, _process, n_workers=args.workers, extra=extra)

    count = len(files)
    np.save(os.path.join(args.output_dir, "TKE_t.npy"), (state['TKE_t'] / count).astype(np.float32))
    np.save(os.path.join(args.output_dir, "TKE_tz.npy"), (state['TKE_tz'] / count).astype(np.float32))
    print(f"Done. Wrote TKE_t.npy and TKE_tz.npy to {args.output_dir} ({count} snapshots averaged)")


if __name__ == '__main__':
    main()
