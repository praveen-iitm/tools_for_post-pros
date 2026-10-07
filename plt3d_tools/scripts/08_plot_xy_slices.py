#!/usr/bin/env python3
"""
08_plot_xy_slices.py
------------------------
Plot x-y contour slices at a fixed z-index (default: the midplane,
nz // 2) of the time-mean fields (from 01_compute_mean.py) and of one
instantaneous snapshot, side by side field-by-field, saved as PNGs in a
folder. Headless by default (uses the 'Agg' matplotlib backend, so it
runs on a login node or inside a job with no display).

For each requested field, this writes TWO separate PNGs:
  mean_<field>_z<z_idx>.png     -- the time-mean field's z-slice
  instant_<field>_z<z_idx>_<snapshot tag>.png  -- one snapshot's z-slice

(not a single combined figure) so each can be viewed, cropped, or shared
on its own; the matching mean/instant pair share a filename stem
(mean_/instant_<field>_z<z_idx>...) for easy pairing, and --shared-scale
can tie them to a common color scale for direct visual comparison.

By default the instantaneous field color scale is independent of the
mean field's: an instantaneous RBC velocity field typically has a much
wider range than its own time-mean (which is smoothed by averaging over
every snapshot), so sharing one scale often washes the mean field's
structure out to a nearly uniform color. Pass --shared-scale to force
both plots for a field onto one common (vmin, vmax) instead, when a
direct like-for-like comparison is what you actually want.

Example
-------
    # Midplane slices of u,v,w,T -- mean and the first snapshot found
    python 08_plot_xy_slices.py \
        --grid-file    /path/to/Case/output/grid.xyz \
        --mean-dir     /path/to/post_proc_1-mean \
        --mean-tag 731-931 \
        --solution-dir /path/to/Case/output \
        --output-dir   /path/to \
        --fields u v w T

    # A specific snapshot (by index into the sorted file list) and a
    # shared color scale between mean and instantaneous:
    python 08_plot_xy_slices.py \
        --grid-file    /path/to/Case/output/grid.xyz \
        --mean-dir     /path/to/post_proc_1-mean \
        --mean-tag 731-931 \
        --solution-dir /path/to/Case/output \
        --output-dir   /path/to \
        --snapshot-index 50 \
        --shared-scale
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')  # headless -- no DISPLAY required
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plt_3d_tools.io import read_grid, read_solution, list_solution_files

FIELD_COL = {'u': 0, 'v': 1, 'w': 2, 'T': 4}


def _plot_slice(X, Y, V, title, out_path, cmap, dpi, vmin=None, vmax=None):
    # The domain is much taller than it is wide, so use a portrait figure.
    # This avoids the large horizontal white margins produced by an 8x6
    # figure together with ax.set_aspect('equal').
    fig, ax = plt.subplots(figsize=(4.5, 7.0))

    cp = ax.pcolormesh(
        X, Y, V,
        shading='auto',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )

    # Create the colorbar axis from the plotting axis itself so that the
    # colorbar has exactly the same vertical height as the x-y domain.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="12%", pad=0.18)
    fig.colorbar(cp, cax=cax)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title, pad=8)
    ax.set_aspect('equal')

    # No extra padding around the plotted data.
    ax.margins(0)

    # Trim any remaining whitespace around the figure.
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--grid-file', required=True, help='Path to grid.xyz')
    ap.add_argument('--mean-dir', required=True, help='Directory with mean_<field>_*.npy from script 01')
    ap.add_argument('--mean-tag', default='', help='Must match script 01\'s --tag')
    ap.add_argument('--solution-dir', required=True, help='Folder containing grid*.f snapshots')
    ap.add_argument('--output-dir', required=True,
                     help='Parent dir; results go to <output-dir>/post-proc-8_xy_slices_z<z_idx>/')
    ap.add_argument('--fields', nargs='+', default=['u', 'v', 'w', 'T'], choices=['u', 'v', 'w', 'T'])
    ap.add_argument('--z-idx', type=int, default=None, help='z-index to slice at. Default: nz // 2')
    ap.add_argument('--snapshot-index', type=int, default=0,
                     help='Which snapshot to use for the instantaneous slice, as an index into '
                          'the sorted (by iteration) --solution-dir file list. Default: 0 (the '
                          'first/earliest snapshot found).')
    ap.add_argument('--snapshot-file', default=None,
                     help='Use this specific snapshot file instead of --snapshot-index.')
    ap.add_argument('--shared-scale', action='store_true',
                     help="Use one common color scale (vmin, vmax) for a field's mean and "
                          "instantaneous plots, instead of scaling each independently.")
    ap.add_argument('--cmap', default='jet', help="Matplotlib colormap name. Default: 'jet' "
                                                    "(matches plt_3d_tools.plotting's quick-look plots).")
    ap.add_argument('--dpi', type=int, default=150)
    args = ap.parse_args()

    x, y, z, nx, ny, nz = read_grid(args.grid_file)
    z_idx = args.z_idx if args.z_idx is not None else nz // 2
    if not (0 <= z_idx < nz):
        sys.exit(f"--z-idx {z_idx} out of range for nz={nz}")

    X, Y = x[:, :, z_idx], y[:, :, z_idx]
    z_coord = float(z[0, 0, z_idx])

    if args.snapshot_file is not None:
        snap_path = args.snapshot_file
        if not os.path.isfile(snap_path):
            sys.exit(f"--snapshot-file not found: {snap_path}")
    else:
        files = list_solution_files(args.solution_dir)
        if not files:
            sys.exit(f"No grid*.f files found in {args.solution_dir}")
        if not (0 <= args.snapshot_index < len(files)):
            sys.exit(f"--snapshot-index {args.snapshot_index} out of range "
                      f"(found {len(files)} snapshot files in {args.solution_dir})")
        snap_path = files[args.snapshot_index]
    snap_tag = os.path.splitext(os.path.basename(snap_path))[0]
    print(f"Using snapshot: {snap_path}")

    sol, snx, sny, snz, nvar, var_names = read_solution(snap_path)
    if (snx, sny, snz) != (nx, ny, nz):
        sys.exit(f"Snapshot grid ({snx},{sny},{snz}) doesn't match --grid-file ({nx},{ny},{nz})")

    suffix = f"_{args.mean_tag}" if args.mean_tag else ""
    out_dir = os.path.join(args.output_dir, f"post-proc-8_xy_slices_z{z_idx}")
    os.makedirs(out_dir, exist_ok=True)

    for f in args.fields:
        mean_path = os.path.join(args.mean_dir, f"mean_{f}{suffix}.npy")
        if not os.path.isfile(mean_path):
            print(f"WARNING: {mean_path} not found, skipping field '{f}'")
            continue
        mean_field = np.load(mean_path)
        if mean_field.shape != (nx, ny, nz):
            print(f"WARNING: mean_{f}{suffix}.npy has shape {mean_field.shape}, "
                  f"expected {(nx, ny, nz)}, skipping field '{f}'")
            continue
        mean_slice = mean_field[:, :, z_idx]

        col = FIELD_COL[f]
        if col >= nvar:
            print(f"WARNING: snapshot only has {nvar} variables, field '{f}' (column {col}) "
                  f"not present, skipping")
            continue
        instant_slice = sol[:, :, z_idx, col]

        if args.shared_scale:
            vmin = float(min(mean_slice.min(), instant_slice.min()))
            vmax = float(max(mean_slice.max(), instant_slice.max()))
        else:
            vmin = vmax = None

        _plot_slice(X, Y, mean_slice,
                    title=f"Mean {f}, z={z_coord:.4g} (z-idx {z_idx})",
                    out_path=os.path.join(out_dir, f"mean_{f}_z{z_idx}.png"),
                    cmap=args.cmap, dpi=args.dpi, vmin=vmin, vmax=vmax)

        _plot_slice(X, Y, instant_slice,
                    title=f"Instantaneous {f}, z={z_coord:.4g} (z-idx {z_idx})\n{snap_tag}",
                    out_path=os.path.join(out_dir, f"instant_{f}_z{z_idx}_{snap_tag}.png"),
                    cmap=args.cmap, dpi=args.dpi, vmin=vmin, vmax=vmax)

        print(f"  wrote mean_{f}_z{z_idx}.png and instant_{f}_z{z_idx}_{snap_tag}.png")

    print(f"Done. Wrote slice PNGs to {out_dir}")


if __name__ == '__main__':
    main()
