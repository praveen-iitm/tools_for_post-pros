#!/usr/bin/env python3
"""
03_two_point_correlation.py
------------------------------
Two-point (z-lag) correlations of u, v, w, T fluctuations against a fixed
reference point (x_idx, y_idx), computed for every snapshot and averaged.

Uses an FFT-based circular cross-correlation instead of the naive
roll-and-accumulate loop:

    R[x,y,z] = sum_z0 a[x_idx,y_idx,z0] * b[x,y,(z0+z) mod Nz]
             = irfft( conj(rfft(a_line)) * rfft(b, axis=z) )

which is O(Nx*Ny*Nz*log Nz) per field pair per snapshot instead of
O(Nx*Ny*Nz^2), and reuses each field's forward FFT across all 4 reference
signals instead of recomputing it. Accumulators are float32.

Requires mean_u/v/w/T_*.npy from 01_compute_mean.py.

Example
-------
    python 03_two_point_correlation.py \
        --solution-dir /media/user/Data2/kiran_data/Ra_2e9/Case/output \
        --mean-dir     /media/user/Data2/kiran_data/post_proc_1-mean \
        --mean-tag 731-931 \
        --output-dir   /media/user/Data2/kiran_data \
        --x-idx 25 --y-idx 222 \
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

FIELDS = ['u', 'v', 'w', 'T']


def _init(shape, x_idx, y_idx, u_mean_tz, v_mean_tz, w_mean_tz, T_mean_tz):
    return {f"{i}{j}": np.zeros(shape, dtype=np.float32) for i in FIELDS for j in FIELDS}


def _process(fn, state, shape, x_idx, y_idx, u_mean_tz, v_mean_tz, w_mean_tz, T_mean_tz):
    Nz = shape[2]
    sol, nx, ny, nz, nvar, var_names = read_solution(fn)

    fields = {
        'u': (sol[:, :, :, 0] - u_mean_tz).astype(np.float32, copy=False),
        'v': (sol[:, :, :, 1] - v_mean_tz).astype(np.float32, copy=False),
        'w': (sol[:, :, :, 2] - w_mean_tz).astype(np.float32, copy=False),
        'T': (sol[:, :, :, 4] - T_mean_tz).astype(np.float32, copy=False),
    }
    del sol  # free the raw snapshot buffer as soon as possible

    # Cheap: 1D reference-line rFFT (length Nz only) for each field
    line_ffts = {name: np.fft.rfft(f[x_idx, y_idx, :]) for name, f in fields.items()}

    # One full-field forward rFFT per target field "b", reused for all 4
    # reference signals (u,v,w,T) instead of recomputing it 4x.
    for jname, bfield in fields.items():
        Bf = np.fft.rfft(bfield, axis=2)  # (Nx, Ny, Nz//2+1) complex64
        for iname, Aline in line_ffts.items():
            prod = np.conj(Aline)[None, None, :] * Bf
            corr = np.fft.irfft(prod, n=Nz, axis=2)  # circular cross-correlation
            state[f"{iname}{jname}"] += corr.astype(np.float32, copy=False)
            del prod, corr
        del Bf

    del fields, line_ffts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--solution-dir', required=True)
    ap.add_argument('--mean-dir', required=True, help='Directory with mean_u/v/w/T_*.npy from script 01')
    ap.add_argument('--mean-tag', default='')
    ap.add_argument('--output-dir', required=True, help='Parent dir; results go in <output-dir>/post_proc_3-tp_<x>_<y>/')
    ap.add_argument('--x-idx', type=int, required=True)
    ap.add_argument('--y-idx', type=int, required=True)
    ap.add_argument('--workers', type=int, default=None)
    args = ap.parse_args()

    suffix = f"_{args.mean_tag}" if args.mean_tag else ""
    u_mean_t = np.load(os.path.join(args.mean_dir, f"mean_u{suffix}.npy"))
    v_mean_t = np.load(os.path.join(args.mean_dir, f"mean_v{suffix}.npy"))
    w_mean_t = np.load(os.path.join(args.mean_dir, f"mean_w{suffix}.npy"))
    T_mean_t = np.load(os.path.join(args.mean_dir, f"mean_T{suffix}.npy"))
    shape = u_mean_t.shape
    nz = shape[2]

    def z_avg_broadcast(a):
        return np.repeat(np.mean(a, axis=2)[..., np.newaxis], nz, axis=2).astype(np.float32)

    u_mean_tz = z_avg_broadcast(u_mean_t)
    v_mean_tz = z_avg_broadcast(v_mean_t)
    w_mean_tz = z_avg_broadcast(w_mean_t)
    T_mean_tz = z_avg_broadcast(T_mean_t)

    files = list_solution_files(args.solution_dir)
    if not files:
        sys.exit(f"No grid*.f files found in {args.solution_dir}")
    print(f"Found {len(files)} snapshot files in {args.solution_dir}")

    out_dir = os.path.join(args.output_dir, f"post_proc_3-tp_{args.x_idx}_{args.y_idx}")
    os.makedirs(out_dir, exist_ok=True)

    extra = (shape, args.x_idx, args.y_idx, u_mean_tz, v_mean_tz, w_mean_tz, T_mean_tz)
    R = run_parallel_accumulation(files, _init, _process, n_workers=args.workers, extra=extra)

    n = len(files)
    for key, arr in R.items():
        arr /= n
        np.save(os.path.join(out_dir, f"R_{key}.npy"), arr)

    print(f"Done. Wrote 16 R_ij.npy files to {out_dir} ({n} snapshots averaged)")


if __name__ == '__main__':
    main()
