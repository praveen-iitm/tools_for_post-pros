#!/usr/bin/env python3
"""
09_spectrum_monitors.py
---------------------------
1D energy spectra along the periodic direction (z) at several fixed
monitoring (x_idx, y_idx) locations in a single run -- the same idea as
script 05's --mode point, but for a list of locations at once instead of
one location per invocation.

This is the local-spectrum counterpart of Trias, Soria, Oliva &
Perez-Segarra's monitoring-point spectra (their eq. 2.13/2.14,
E_phiphi(k1, x2, x3) at several fixed (x2, x3)): a handful of named
monitor points, each giving its own E(k) along the cavity's homogeneous
direction, rather than one whole-cavity-averaged spectrum. Praveen's
case has z (not x1) as the periodic/homogeneous direction, so that's
what's FFT'd here, same convention as script 05.

Why a dedicated script instead of just calling 05 --mode point once per
location: 05's point mode still FFTs the *entire* (x,y) plane for every
snapshot and then throws away every column except the one requested, and
calling it once per location means re-reading (and re-FFTing the whole
plane for) every snapshot again for each location. This script instead
reads each snapshot once, slices out just the requested (x_idx, y_idx)
columns BEFORE FFT-ing (a length-Nz 1D transform per field per location,
not a whole-plane transform), and accumulates every location's spectrum
in that same pass -- so the cost no longer multiplies with the number of
monitor locations the way repeated 05 calls would.

Fluctuations are taken about the z-averaged time-mean (mean_tz, same
convention as script 05/TKE_tz) by default; pass --use-3d-mean to
subtract the raw (z-dependent) mean_t instead.

Outputs reuse script 05's exact per-location filenames (k.npy,
P_<field>.npy, E_total.npy), one set per location under its own
subfolder, so 06_plot_spectrum.py can plot any single location's result
directly with no changes -- just point --spectrum-dir at that subfolder.

Example
-------
    python 09_spectrum_monitors.py \
        --solution-dir /path/to/Case/output \
        --grid-file    /path/to/Case/output/grid.xyz \
        --mean-dir     /path/to/post_proc_1-mean \
        --mean-tag 731-931 \
        --output-dir   /path/to \
        --locations 25,222 60,222 60,100 \
        --fields u v w T \
        --workers 16

    # Then plot any one location's spectrum with the existing script:
    python 06_plot_spectrum.py \
        --spectrum-dir /path/to/post-proc-9_spectrum_monitors/loc_25_222 \
        --slope -1.6667
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


def _parse_locations(items):
    """Parse ["25,222", "60,222", ...] into [(25, 222), (60, 222), ...]."""
    locs = []
    for item in items:
        parts = item.split(',')
        if len(parts) != 2:
            sys.exit(f"--locations entry {item!r} is not 'x_idx,y_idx'")
        try:
            xi, yi = int(parts[0]), int(parts[1])
        except ValueError:
            sys.exit(f"--locations entry {item!r} is not 'x_idx,y_idx' (non-integer)")
        locs.append((xi, yi))
    return locs


def _state_key(field, loc_idx):
    return f"{field}__{loc_idx}"


def _init(fields, locations, means, nk):
    keys = (_state_key(f, li) for f in fields for li in range(len(locations)))
    return {k: np.zeros(nk, dtype=np.float64) for k in keys}


def _process(fn, state, fields, locations, means, nk):
    sol, nx, ny, nz, nvar, var_names = read_solution(fn)
    for f in fields:
        col = FIELD_COL[f]
        for li, (xi, yi) in enumerate(locations):
            # Slice the single column BEFORE FFT-ing -- a length-Nz 1D
            # transform, not a whole-(x,y)-plane one, since only this
            # one column's spectrum is actually needed.
            line = sol[xi, yi, :, col] - means[f][xi, yi, :]
            X = np.fft.rfft(line.astype(np.float32, copy=False))
            psd = (X.real ** 2 + X.imag ** 2) / (nz * nz)
            psd[1:(nz + 1) // 2] *= 2.0  # fold negative freqs (Parseval)
            state[_state_key(f, li)] += psd
    del sol


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--solution-dir', required=True)
    ap.add_argument('--grid-file', required=True, help='grid.xyz, used for the z-spacing / physical wavenumbers')
    ap.add_argument('--mean-dir', required=True, help='Directory with mean_<field>_*.npy from script 01')
    ap.add_argument('--mean-tag', default='')
    ap.add_argument('--output-dir', required=True,
                     help='Parent dir; results go to <output-dir>/post-proc-9_spectrum_monitors/loc_<x>_<y>/')
    ap.add_argument('--locations', nargs='+', required=True,
                     help="One or more monitor points as 'x_idx,y_idx' (no spaces), e.g. "
                          "--locations 25,222 60,222 60,100")
    ap.add_argument('--fields', nargs='+', default=['u', 'v', 'w'], choices=['u', 'v', 'w', 'T'])
    ap.add_argument('--use-3d-mean', action='store_true',
                     help='Subtract the full (z-dependent) mean_t instead of the z-averaged mean_tz')
    ap.add_argument('--workers', type=int, default=None)
    args = ap.parse_args()

    locations = _parse_locations(args.locations)

    x, y, z, nx, ny, nz = read_grid(args.grid_file)

    for xi, yi in locations:
        if not (0 <= xi < nx and 0 <= yi < ny):
            sys.exit(f"Location ({xi},{yi}) out of range for grid ({nx},{ny},{nz})")

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
    print(f"Monitoring {len(locations)} location(s): {locations}")

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

    out_root = os.path.join(args.output_dir, "post-proc-9_spectrum_monitors")
    os.makedirs(out_root, exist_ok=True)

    extra = (args.fields, locations, means, nk)
    state = run_parallel_accumulation(files, _init, _process, n_workers=args.workers, extra=extra)

    n = len(files)
    for li, (xi, yi) in enumerate(locations):
        loc_dir = os.path.join(out_root, f"loc_{xi}_{yi}")
        os.makedirs(loc_dir, exist_ok=True)
        np.save(os.path.join(loc_dir, "k.npy"), k.astype(np.float32))

        loc_fields = {}
        for f in args.fields:
            P = (state[_state_key(f, li)] / n).astype(np.float32)
            np.save(os.path.join(loc_dir, f"P_{f}.npy"), P)
            loc_fields[f] = P

        if all(f in args.fields for f in ('u', 'v', 'w')):
            E_total = 0.5 * (loc_fields['u'] + loc_fields['v'] + loc_fields['w'])
            np.save(os.path.join(loc_dir, "E_total.npy"), E_total.astype(np.float32))

        print(f"  wrote {loc_dir}")

    print(f"Done. {len(locations)} location(s) averaged over {n} snapshots, "
          f"written under {out_root}")


if __name__ == '__main__':
    main()
