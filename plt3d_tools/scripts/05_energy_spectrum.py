#!/usr/bin/env python3
"""
05_energy_spectrum.py
------------------------
Turbulence energy spectrum along the periodic direction (z), for a
tall-cavity RBC case with bounded x,y and periodic z.

For each requested field, fluctuations are computed about the
z-averaged time-mean (mean_tz -- same convention as TKE_tz in
02_compute_fluctuations_tke.py): since z is physically homogeneous, the
true mean cannot depend on z, so the z-average of the 3D time-mean is a
better-converged estimate of it than the raw 3D time-mean itself. Pass
--use-3d-mean to instead subtract the (z-dependent) full mean_t -- useful
as a diagnostic for whether the mean has actually converged to being
z-independent.

For each snapshot, the fluctuation field is FFT'd along z
(np.fft.rfft, axis=2) and converted to a one-sided power spectral
density that conserves variance (Parseval): summing all output P(k)
values recovers the time-mean of the field's z-variance at that
location. Two output modes:

  --mode plane-avg (default)
      P(k) is averaged over every (x,y) grid point before accumulating
      -- one spectrum representative of the whole cavity.
  --mode point --x-idx I --y-idx J
      P(k) at a single (x_idx, y_idx) column only, e.g. to compare
      near-wall vs. core spectra.

Wavenumbers are physical (rad / length unit), derived from the grid
spacing in z (dz is assumed uniform -- required for an FFT-based
periodic spectrum). If u, v, w are all requested, the combined
turbulent kinetic energy spectrum E(k) = 0.5*(P_uu + P_vv + P_ww) is
also written.

Requires mean_u/v/w/T_*.npy from 01_compute_mean.py.

Example
-------
    # Whole-cavity spectrum of u, v, w (+ combined TKE spectrum)
    python 05_energy_spectrum.py \
        --solution-dir /data/praveen/kiran_data/Ra_2e9/Case/output \
        --grid-file    /data/praveen/kiran_data/Ra_2e9/Case/output/grid.xyz \
        --mean-dir     /data/praveen/kiran_data/Ra_2e9/Case/mean_731-1131 \
        --mean-tag 731-1131 \
        --output-dir   /data/praveen/kiran_data/Ra_2e9/Case \
        --fields u v w \
        --workers 16

    # Local spectrum at a single column (e.g. near a sidewall)
    python 05_energy_spectrum.py \
        --solution-dir /data/praveen/kiran_data/Ra_2e9/Case/output \
        --grid-file    /data/praveen/kiran_data/Ra_2e9/Case/output/grid.xyz \
        --mean-dir     /data/praveen/kiran_data/Ra_2e9/Case/mean_731-1131 \
        --mean-tag 731-1131 \
        --output-dir   /data/praveen/kiran_data/Ra_2e9/Case \
        --mode point --x-idx 5 --y-idx 25 \
        --fields u v w T \
        --workers 16

Loading and plotting the result:

    import numpy as np
    import matplotlib.pyplot as plt

    k = np.load(".../k.npy")
    E = np.load(".../E_total.npy")
    plt.loglog(k[1:], E[1:])   # skip k=0 (mean/DC bin)
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plt_3d_tools.io import read_grid, read_solution, list_solution_files
from plt_3d_tools.parallel import run_parallel_accumulation

FIELD_COL = {'u': 0, 'v': 1, 'w': 2, 'T': 4}


def _psd_1d(fluct, mode, x_idx, y_idx):
    """One-sided variance-conserving PSD of `fluct` along its last axis
    (z), reduced to a 1D array of length Nz//2+1 either by averaging
    over the leading (x,y) axes (mode='plane-avg') or by indexing a
    single column (mode='point')."""
    nz = fluct.shape[-1]
    X = np.fft.rfft(fluct, axis=-1)
    psd = (X.real ** 2 + X.imag ** 2) / (nz * nz)
    # Fold negative frequencies into the one-sided spectrum (Parseval):
    # double every bin except k=0 and, if nz is even, the Nyquist bin.
    psd[..., 1: (nz + 1) // 2] *= 2.0
    if mode == 'plane-avg':
        return psd.mean(axis=(0, 1))
    else:
        return psd[x_idx, y_idx, :]


def _init(fields, mode, x_idx, y_idx, means, nk):
    return {f: np.zeros(nk, dtype=np.float64) for f in fields}


def _process(fn, state, fields, mode, x_idx, y_idx, means, nk):
    sol, nx, ny, nz, nvar, var_names = read_solution(fn)
    for f in fields:
        raw = sol[:, :, :, FIELD_COL[f]]
        fluct = (raw - means[f]).astype(np.float32, copy=False)
        state[f] += _psd_1d(fluct, mode, x_idx, y_idx)
    del sol


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--solution-dir', required=True)
    ap.add_argument('--grid-file', required=True, help='grid.xyz, used for the z-spacing / physical wavenumbers')
    ap.add_argument('--mean-dir', required=True, help='Directory with mean_<field>_*.npy from script 01')
    ap.add_argument('--mean-tag', default='')
    ap.add_argument('--output-dir', required=True,
                     help='Parent dir; results go to <output-dir>/post_proc_5-spectrum[_<x>_<y>]/')
    ap.add_argument('--fields', nargs='+', default=['u', 'v', 'w'], choices=['u', 'v', 'w', 'T'])
    ap.add_argument('--mode', choices=['plane-avg', 'point'], default='plane-avg')
    ap.add_argument('--x-idx', type=int, default=None, help='Required if --mode point')
    ap.add_argument('--y-idx', type=int, default=None, help='Required if --mode point')
    ap.add_argument('--use-3d-mean', action='store_true',
                     help='Subtract the full (z-dependent) mean_t instead of the z-averaged mean_tz')
    ap.add_argument('--workers', type=int, default=None)
    args = ap.parse_args()

    if args.mode == 'point' and (args.x_idx is None or args.y_idx is None):
        sys.exit('--mode point requires --x-idx and --y-idx')

    x, y, z, nx, ny, nz = read_grid(args.grid_file)
    dz_all = np.diff(z[0, 0, :])
    dz = float(dz_all[0])
    if not np.allclose(dz_all, dz, rtol=1e-4):
        print(f"WARNING: z-spacing is not uniform (min={dz_all.min():.6g}, "
              f"max={dz_all.max():.6g}) -- an FFT-based periodic spectrum "
              f"assumes uniform spacing; results may not be meaningful.")
    Lz = nz * dz
    k = np.fft.rfftfreq(nz, d=dz) * 2.0 * np.pi  # rad / length unit
    nk = k.size
    print(f"Grid: {nx} x {ny} x {nz}, dz={dz:.6g}, Lz={Lz:.6g} -> {nk} wavenumbers, "
          f"dk={2 * np.pi / Lz:.6g}")

    suffix = f"_{args.mean_tag}" if args.mean_tag else ""
    means = {}
    for f in args.fields:
        mean_t = np.load(os.path.join(args.mean_dir, f"mean_{f}{suffix}.npy"))
        if args.use_3d_mean:
            means[f] = mean_t.astype(np.float32, copy=False)
        else:
            means[f] = np.repeat(np.mean(mean_t, axis=2)[..., np.newaxis], nz,
                                  axis=2).astype(np.float32)

    files = list_solution_files(args.solution_dir)
    if not files:
        sys.exit(f"No grid*.f files found in {args.solution_dir}")
    print(f"Found {len(files)} snapshot files in {args.solution_dir}")

    if args.mode == 'point':
        out_dir = os.path.join(args.output_dir, f"post_proc_5-spectrum_{args.x_idx}_{args.y_idx}")
    else:
        out_dir = os.path.join(args.output_dir, "post_proc_5-spectrum")
    os.makedirs(out_dir, exist_ok=True)

    extra = (args.fields, args.mode, args.x_idx, args.y_idx, means, nk)
    state = run_parallel_accumulation(files, _init, _process, n_workers=args.workers, extra=extra)

    n = len(files)
    np.save(os.path.join(out_dir, "k.npy"), k.astype(np.float32))
    for f in args.fields:
        np.save(os.path.join(out_dir, f"P_{f}.npy"), (state[f] / n).astype(np.float32))

    if all(f in args.fields for f in ('u', 'v', 'w')):
        E_total = 0.5 * (state['u'] + state['v'] + state['w']) / n
        np.save(os.path.join(out_dir, "E_total.npy"), E_total.astype(np.float32))
        print(f"Done. Wrote k.npy, P_<field>.npy and E_total.npy to {out_dir} ({n} snapshots averaged)")
    else:
        print(f"Done. Wrote k.npy and P_<field>.npy to {out_dir} ({n} snapshots averaged)")


if __name__ == '__main__':
    main()
