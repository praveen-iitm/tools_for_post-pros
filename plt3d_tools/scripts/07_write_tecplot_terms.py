#!/usr/bin/env python3
"""
07_write_tecplot_terms.py
----------------------------
Write the budget terms from 04_budget_terms.py out as Tecplot ASCII
(POINT-format) .dat files, for visualization in Tecplot/ParaView.

This is the plt3d_tools equivalent of a different case's post-proc-7
notebook, scoped down to just the one thing that notebook's own author
said it actually does: write the budget terms to a Tecplot .dat file.
(That notebook also recomputed two-point correlations and a "live" FFT
sanity check inline -- this script leaves those out; 03_two_point_
correlation.py already produces the R_ij.npy files, and there's nothing
here to export them to Tecplot too if you need that later.)

Coordinates written out are the SEPARATION vector (r_x, r_y, r_z) from
the budget terms' reference point (x_idx, y_idx, z_idx) -- i.e. relative
coordinates, not absolute grid coordinates -- matching how the original
notebook wrote them and matching the two-point framework the budget
terms themselves are defined in.

Which variables end up in the .dat files is driven entirely by which
E_*.npy / H_*.npy files 04_budget_terms.py actually wrote (this repo's
04 currently writes prod_h, prod_I, inter_m, inter_f only -- no diss or
buoy, see plt3d_tools/README.md -- unlike the other case's more complete
calc script). A _prod_tot or _inter_tot column is added automatically
whenever both halves of that combination are present, so this script
also works unmodified if a future version of 04 (or a hand-added
E_diss.npy / H_buoy.npy etc.) adds more terms.

Example
-------
    python 07_write_tecplot_terms.py \
        --terms-dir  /path/to/post-proc-4_terms_calc_25_222_64 \
        --grid-file  /path/to/Case/output/grid.xyz \
        --output-dir /path/to

    # If --terms-dir's name doesn't carry "_<x>_<y>_<z>" (e.g. you
    # renamed the folder), give the reference point explicitly instead:
    python 07_write_tecplot_terms.py \
        --terms-dir  /path/to/my_renamed_terms_folder \
        --grid-file  /path/to/Case/output/grid.xyz \
        --output-dir /path/to \
        --x-idx 25 --y-idx 222 --z-idx 64
"""
import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plt_3d_tools.io import read_grid
from plt_3d_tools.gradients import GradientOps

# (suffix, numerator-pair-for-"_tot") -- in the exact column order the
# original notebook used. A "_tot" entry is only added if BOTH halves of
# its pair were found on disk.
TERM_LAYOUT = [
    ('prod_h', None),
    ('prod_I', None),
    ('prod_tot', ('prod_h', 'prod_I')),
    ('inter_m', None),
    ('inter_f', None),
    ('inter_tot', ('inter_m', 'inter_f')),
    ('diss', None),
    ('buoy', None),
]


def _load_terms(term_dir, prefix, co_eff):
    """Load every <prefix>_<suffix>.npy found in term_dir (prefix is
    'E' or 'H'), undoing the co_eff weighting 04_budget_terms.py applied
    before saving (same as the original notebook's `(1/co_eff) * np.load(...)`),
    and filling in any _tot combination whose both halves are present.
    Returns an ordered dict {"<prefix>_<suffix>": ndarray}, in
    TERM_LAYOUT's order, containing only what was actually found."""
    loaded = {}
    out = {}
    for suffix, combo in TERM_LAYOUT:
        if combo is None:
            path = os.path.join(term_dir, f"{prefix}_{suffix}.npy")
            if os.path.isfile(path):
                arr = np.load(path) / co_eff
                loaded[suffix] = arr
                out[f"{prefix}_{suffix}"] = arr
        else:
            a, b = combo
            if a in loaded and b in loaded:
                out[f"{prefix}_{suffix}"] = loaded[a] + loaded[b]
    return out


def write_tecplot_3d(filename, X, Y, Z, variables, title="3D data"):
    """Write a structured (I,J,K) Tecplot ASCII POINT-format file:
    one line per grid point, columns X Y Z <var1> <var2> ..., point
    order i-fastest-then-j-then-k -- identical format to the original
    notebook's writer. Built with a single vectorized np.savetxt call
    instead of a Python triple-nested loop with per-point f-string
    formatting, since that loop is the dominant cost for any
    realistically sized grid (it's an O(Nx*Ny*Nz) pure-Python loop in
    the original) and contributes nothing to the actual file format.
    """
    Nx, Ny, Nz = X.shape
    assert Y.shape == (Nx, Ny, Nz), f"Y has wrong shape {Y.shape}"
    assert Z.shape == (Nx, Ny, Nz), f"Z has wrong shape {Z.shape}"
    for name, arr in variables.items():
        assert arr.shape == (Nx, Ny, Nz), f"{name} has wrong shape {arr.shape}"

    # order='F' ravels with the first axis fastest, i.e. i fastest, then
    # j, then k -- exactly the point order the original nested loop
    # produced ("for k: for j: for i").
    cols = [X.ravel(order='F'), Y.ravel(order='F'), Z.ravel(order='F')]
    cols += [arr.ravel(order='F') for arr in variables.values()]
    data = np.column_stack(cols)

    var_names = ['"X"', '"Y"', '"Z"'] + [f'"{name}"' for name in variables]
    header = f'TITLE = "{title}"\nVARIABLES = {" ".join(var_names)}\nZONE I={Nx}, J={Ny}, K={Nz}, F=POINT'

    np.savetxt(filename, data, fmt='%.8e', header=header, comments='')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--terms-dir', required=True,
                     help="04_budget_terms.py's output directory, e.g. "
                          ".../post-proc-4_terms_calc_<x>_<y>_<z>")
    ap.add_argument('--grid-file', required=True, help='Path to grid.xyz')
    ap.add_argument('--output-dir', required=True,
                     help="Parent dir; results go to <output-dir>/post-proc-7_tecplot_<x>_<y>_<z>/")
    ap.add_argument('--x-idx', type=int, default=None,
                     help="Reference point x-index. Default: parsed from --terms-dir's name.")
    ap.add_argument('--y-idx', type=int, default=None,
                     help="Reference point y-index. Default: parsed from --terms-dir's name.")
    ap.add_argument('--z-idx', type=int, default=None,
                     help="Reference point z-index. Default: parsed from --terms-dir's name.")
    args = ap.parse_args()

    # Recover (x_idx, y_idx, z_idx) from the folder name 04_budget_terms.py
    # gives its output (post-proc-4_terms_calc_<x>_<y>_<z>), so you don't
    # have to pass the same three numbers twice and risk a mismatch; any
    # of --x-idx/--y-idx/--z-idx explicitly given overrides the parsed value.
    m = re.search(r'_(\d+)_(\d+)_(\d+)$', os.path.basename(os.path.normpath(args.terms_dir)))
    if m is None and (args.x_idx is None or args.y_idx is None or args.z_idx is None):
        sys.exit(f"Could not parse a reference point from --terms-dir ({args.terms_dir!r}); "
                  f"pass --x-idx/--y-idx/--z-idx explicitly.")
    x_idx = args.x_idx if args.x_idx is not None else int(m.group(1))
    y_idx = args.y_idx if args.y_idx is not None else int(m.group(2))
    z_idx = args.z_idx if args.z_idx is not None else int(m.group(3))

    x, y, z, nx, ny, nz = read_grid(args.grid_file)
    g = GradientOps(x, y, z, (x_idx, y_idx, z_idx))
    co_eff = g.co_eff

    e_dir = os.path.join(args.terms_dir, "e_terms")
    h_dir = os.path.join(args.terms_dir, "h_terms")

    out_dir = os.path.join(args.output_dir, f"post-proc-7_tecplot_{x_idx}_{y_idx}_{z_idx}")
    os.makedirs(out_dir, exist_ok=True)

    wrote_any = False

    if os.path.isdir(e_dir):
        vars_E = _load_terms(e_dir, 'E', co_eff)
        if vars_E:
            print(f"Writing E_solution.dat with {list(vars_E)}")
            write_tecplot_3d(os.path.join(out_dir, "E_solution.dat"),
                              g.r_x, g.r_y, g.r_z, vars_E, title="3D_RBC_ED")
            wrote_any = True
        else:
            print(f"WARNING: no E_*.npy files found in {e_dir}, skipping E_solution.dat")
    else:
        print(f"WARNING: {e_dir} not found, skipping E_solution.dat")

    if os.path.isdir(h_dir):
        vars_H = _load_terms(h_dir, 'H', co_eff)
        if vars_H:
            print(f"Writing H_solution.dat with {list(vars_H)}")
            write_tecplot_3d(os.path.join(out_dir, "H_solution.dat"),
                              g.r_x, g.r_y, g.r_z, vars_H, title="3D_RBC_HF")
            wrote_any = True
        else:
            print(f"WARNING: no H_*.npy files found in {h_dir}, skipping H_solution.dat")
    else:
        print(f"WARNING: {h_dir} not found, skipping H_solution.dat")

    if not wrote_any:
        sys.exit(f"Nothing to write -- found no e_terms/ or h_terms/ .npy files under {args.terms_dir}")

    print(f"Done. Wrote Tecplot files to {out_dir}")


if __name__ == '__main__':
    main()
