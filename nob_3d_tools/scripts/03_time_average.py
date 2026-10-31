#!/usr/bin/env python3
"""
03_time_average.py
------------------
Time-average one variable over every iteration found, assembled across
blocks, then reduce it to a horizontally-averaged vertical profile <VAR>(z).

Everything is written into --output-dir, named with the variable and the
number of iterations averaged (<n>):

    time_avg_<VAR>_<n>iters.npy   time-averaged field: X, Y, Z, <VAR>_mean  (always)
    time_avg_<VAR>_<n>iters.dat   same field as Tecplot ASCII                (unless --no-field-dat);
                                  a point cloud, or with --with-connectivity a
                                  FETETRAHEDRON mesh you can contour in Tecplot/ParaView
    profile_<VAR>_z_<n>iters.dat  horizontally-averaged vertical profile     (always)
    profile_<VAR>_z_<n>iters.png  plot of that profile                       (unless --no-plot)

Restartable, low-RAM: each iteration's assembled field is written to a
checkpoint folder (default <output-dir>/checkpoint_<VAR>/) the moment it is
done, and nothing is accumulated in RAM while the job runs -- the final mean
is a streaming sum over those files. If the job is killed (walltime, node
failure, Ctrl-C), just run the SAME command again: iterations already in the
checkpoint folder are skipped and only the unfinished ones are redone. Files
are written atomically, so a half-written one is never mistaken for done.
The checkpoint also lets you extend an average later (rerun with more
iterations and only the new ones are computed). Disk cost: about N points x
4 bytes per iteration. Use --clean-checkpoint to delete it after a successful
run, or --no-resume to throw it away and start over.

--with-connectivity adds the tetrahedral element connectivity to that .dat so
it is a real mesh rather than bare points. The averaging never needs
connectivity, so it is read once from the first iteration (an extra single-
iteration read: more RAM and time than the point-cloud-only run) and cached
in the checkpoint folder. It is checked against the reference grid's node
order. The .npy output is always a point cloud.

The plot is saved, not shown (headless-safe for clusters); add --show to
also open it interactively. The field .dat is ASCII, so it is large/slow on
a big mesh -- pass --no-field-dat to skip it.

Iterations whose grid doesn't match the first one (within --coord-check-tol)
abort the run rather than silently corrupting the average.

Example
-------
    python 03_time_average.py \
        --input-dir  /data/praveen/nob_rbc_air_3d/dati \
        --output-dir /data/praveen/nob_rbc_air_3d/assembled_T \
        --variable T \
        --workers 12

    # Subset of iterations, no big Tecplot field file, 200 dpi plot
    python 03_time_average.py \
        --input-dir  /data/praveen/nob_rbc_air_3d/dati \
        --output-dir /data/praveen/nob_rbc_air_3d/assembled_T \
        --variable T --iters 5231932 5231933 \
        --no-field-dat --dpi 200
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nob_tools import (
    ALL_VAR_NAMES,
    discover_iterations,
    time_average_field_checkpointed,
    reference_connectivity,
    horizontal_profile,
    write_npy,
    write_dat,
    write_profile_dat,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--pattern", default="tecplot_blk*_iter*.dat")
    ap.add_argument("--variable", default="T",
                    help=f"Variable to average. One of: {', '.join(ALL_VAR_NAMES)} (default T)")
    ap.add_argument("--iters", nargs="*", type=int, default=None,
                    help="Only these iteration numbers (default: every iteration found)")
    ap.add_argument("--coord-tol", type=float, default=1e-6,
                    help="Tolerance for matching duplicate nodes at partition interfaces")
    ap.add_argument("--coord-check-tol", type=float, default=1e-5,
                    help="How close X,Y,Z must be to the reference grid to accept an iteration")
    ap.add_argument("--z-tol", type=float, default=1e-8,
                    help="Tolerance for treating two Z coordinates as the same level")
    ap.add_argument("--checkpoint-dir", default=None,
                    help="Where per-iteration results are kept for restarting "
                         "(default: <output-dir>/checkpoint_<variable>)")
    ap.add_argument("--no-resume", action="store_true",
                    help="Ignore and delete existing checkpoint files; recompute every iteration")
    ap.add_argument("--clean-checkpoint", action="store_true",
                    help="Delete the checkpoint folder after a successful run "
                         "(default: keep it, so a later rerun can extend the average)")
    ap.add_argument("--with-connectivity", action="store_true",
                    help="Write the field .dat as a FETETRAHEDRON mesh (with element connectivity) "
                         "instead of a bare point cloud. Reads connectivity once from the first iteration.")
    ap.add_argument("--no-field-dat", action="store_true",
                    help="Skip the Tecplot .dat of the time-averaged field")
    ap.add_argument("--no-plot", action="store_true", help="Skip the profile PNG")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--show", action="store_true",
                    help="Also open the profile plot interactively (needs a display)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--max-ram-gb", type=float, default=None)
    ap.add_argument("--safety-factor", type=float, default=3.0)
    ap.add_argument("--ram-margin", type=float, default=0.8)
    args = ap.parse_args()

    if args.variable not in ALL_VAR_NAMES:
        sys.exit(f"--variable must be one of: {', '.join(ALL_VAR_NAMES)} (got {args.variable!r})")
    var = args.variable
    var_idx = ALL_VAR_NAMES.index(var)

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")  # headless-safe: saving the PNG needs no display
    import matplotlib.pyplot as plt

    os.makedirs(args.output_dir, exist_ok=True)

    by_iter = discover_iterations(args.input_dir, args.pattern)
    if args.iters is not None:
        by_iter = {it: blocks for it, blocks in by_iter.items() if it in args.iters}
    if not by_iter:
        sys.exit("No matching iteration/block files found.")
    print(f"Found {len(by_iter)} iteration(s)")
    for it, blocks in by_iter.items():
        print(f"  iter {it}: {len(blocks)} block(s)")

    # Expensive part (read + merge + dedup per iteration) runs in parallel,
    # sized to available RAM, with each result checkpointed to disk -- see
    # nob_tools/time_average.py.
    ckpt_dir = args.checkpoint_dir or os.path.join(args.output_dir, f"checkpoint_{var}")
    xyz_ref, mean_field, n_accum, n_total = time_average_field_checkpointed(
        by_iter, var_idx, ckpt_dir, resume=not args.no_resume,
        coord_tol=args.coord_tol, coord_check_tol=args.coord_check_tol,
        workers=args.workers, max_ram_gb=args.max_ram_gb,
        safety_factor=args.safety_factor, ram_margin=args.ram_margin,
    )
    print(f"\nAccumulated {n_accum} of {n_total} iteration(s)")
    print(f"{var} mean range: [{mean_field.min():.6g}, {mean_field.max():.6g}]")

    var_name = f"{var}_mean"
    data = np.column_stack([xyz_ref, mean_field])
    names = ["X", "Y", "Z", var_name]

    npy_path = os.path.join(args.output_dir, f"time_avg_{var}_{n_accum}iters.npy")
    write_npy(npy_path, data, names)
    print(f"Wrote time-averaged {var} -> {npy_path}")

    if not args.no_field_dat:
        dat_path = os.path.join(args.output_dir, f"time_avg_{var}_{n_accum}iters.dat")
        if args.with_connectivity:
            conn = reference_connectivity(
                by_iter, var_idx, ckpt_dir, xyz_ref, coord_tol=args.coord_tol,
                coord_check_tol=args.coord_check_tol)
            write_dat(dat_path, data, conn, names)  # FETETRAHEDRON zone
            print(f"Wrote time-averaged {var} as Tecplot .dat mesh "
                  f"({data.shape[0]:,} nodes, {conn.shape[0]:,} elements) -> {dat_path}")
            del conn
        else:
            write_dat(dat_path, data, None, names)  # ordered F=POINT zone, no connectivity
            print(f"Wrote time-averaged {var} as Tecplot .dat ({data.shape[0]:,} points) -> {dat_path}")

    # Horizontal (x,y) average -> <VAR>(z): groups points by Z (within z-tol)
    # and takes the arithmetic mean over points in each group.
    z_levels, profile, count_per_z = horizontal_profile(xyz_ref, mean_field, z_tol=args.z_tol)
    print(f"Found {len(z_levels)} unique Z-level(s) out of {xyz_ref.shape[0]:,} total points")
    print(f"  points per level: min={count_per_z.min():,}, max={count_per_z.max():,}, "
          f"mean={count_per_z.mean():.0f}")
    if len(z_levels) > 0.5 * xyz_ref.shape[0]:
        print("  WARNING: number of levels is close to the total point count -- "
              "the mesh may not share exact Z-levels. Consider binning by range instead.")

    # Written before the plot so the profile is on disk even if plotting fails.
    prof_path = os.path.join(args.output_dir, f"profile_{var}_z_{n_accum}iters.dat")
    write_profile_dat(prof_path, z_levels, profile, names=("Z", var_name))
    print(f"Wrote vertical profile ({len(z_levels)} Z-levels) -> {prof_path}")

    if args.clean_checkpoint:
        import shutil
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        print(f"Removed checkpoint folder {ckpt_dir}")

    if not args.no_plot or args.show:
        fig, ax = plt.subplots(figsize=(5, 6))
        ax.plot(profile, z_levels, "o-", ms=3)
        ax.set_xlabel(f"$\\langle {var} \\rangle_{{x,y}}$")
        ax.set_ylabel("Z")
        ax.set_title(f"Horizontally-averaged {var} profile ({n_accum} iterations)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        if not args.no_plot:
            png_path = os.path.join(args.output_dir, f"profile_{var}_z_{n_accum}iters.png")
            fig.savefig(png_path, dpi=args.dpi)
            print(f"Saved profile plot -> {png_path}")
        if args.show:
            plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
