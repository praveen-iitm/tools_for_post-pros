#!/usr/bin/env python3
"""
04_nusselt.py
-------------
Nusselt numbers from the raw Tecplot partitions of a Rayleigh-Benard run
with variable properties (NOB), for comparison with e.g. Wang, Wan, Yan &
Sun, JFM (2020) doi:10.1017/jfm.2020.66, who define

    Nu = Q / (k * 2*eps / H),     Q = heat flux across a horizontal plane.

Z is the vertical direction; the hotter of the two walls (lowest/highest Z
level) is detected automatically. Q at height z is, in the solver's units,

    Q(z) = Pe * < rho cp w (T - T_ref) >  -  < kappa dT/dz >

(< > = mean over the nodes of that Z level and over iterations). The time
average is taken on the product, iteration by iteration -- the convective
flux is almost all fluctuation correlation, so it can't come from
time-averaged fields. Fields needed from the files: W, T, rho, cp, kappa.

!! Pe (Peclet number = U_ref * H / kappa_ref) puts the convective term in
!! the same units as the conductive one. It depends on how YOUR solver is
!! nondimensionalised, so you must give it: --pe 1 for thermal-diffusion
!! velocity scaling, or --ra and --pr for free-fall scaling (Pe =
!! sqrt(Ra*Pr)). Q is the same on every horizontal plane in a statistically
!! steady closed cell, so a wrong Pe (or too few iterations) shows up as
!! Nu(z) that is not flat -- the script prints that spread and warns.

Outputs (in --output-dir):
    Nu_hot.npy, Nu_cold.npy, Nu_bulk.npy   single values (0-d arrays)
    Nu_profile.dat    Tecplot ASCII, columns: Z, Nu, Nu_conv, Nu_cond, T_mean
    Nu_summary.txt    the numbers + all settings used
    Nu_profile.png    Nu(z) and its conductive/convective parts (--no-plot skips)

    Nu_hot / Nu_cold : conductive flux at the hot / cold wall (no-slip, so
                       the convective part is zero there), normalised.
    Nu_bulk          : volume average of Nu(z), i.e. integral of Nu dz over
                       --bulk-range (fractions of H; default the whole
                       height, e.g. 0.1 0.9 to drop the wall layers).

Normalisation: Nu = Q / (k_ref * DeltaT / H) with DeltaT = |T_bottom-wall -
T_top-wall| from the data (override with --delta-T, or --epsilon for
DeltaT = 2*eps when T is scaled by T0), H = Z extent (--H), and --k-ref the
reference conductivity in the solver's units (1 if kappa is scaled by its
value at T0).

Gradient dT/dz (--mode auto|columns|fe): 'columns' (structured/extruded
mesh with identical (x,y) points on every Z level) uses second-order
non-uniform finite differences along each column; 'fe' uses per-tetrahedron
gradients from the connectivity and works on any mesh, but its wall
gradient is only first order. 'auto' (default) picks columns if the mesh
allows it. Horizontal averages are plain arithmetic means over the nodes
on each level (see the caveat in README for non-uniform horizontal meshes).

Restartable and low-RAM, like 03: each iteration is reduced to a few small
arrays in <output-dir>/checkpoint_Nu/ as soon as it is done; rerun the same
command after an interruption and finished iterations are skipped.

Example
-------
    python 04_nusselt.py \
        --input-dir  /data/praveen/nob_rbc_air_3d/dati \
        --output-dir /data/praveen/nob_rbc_air_3d/nusselt \
        --ra 1e8 --pr 0.71 --epsilon 0.1 \
        --workers 12
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nob_tools import discover_iterations, write_dat
from nob_tools.nusselt import compute_level_means, nusselt_from_levels


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--pattern", default="tecplot_blk*_iter*.dat")
    ap.add_argument("--iters", nargs="*", type=int, default=None,
                    help="Only these iteration numbers (default: every iteration found)")
    ap.add_argument("--pe", type=float, default=None,
                    help="Peclet number U_ref*H/kappa_ref of your nondimensionalisation "
                         "(1 for thermal-diffusion scaling). Give this OR --ra and --pr.")
    ap.add_argument("--ra", type=float, default=None, help="Rayleigh number (free-fall scaling: Pe = sqrt(Ra*Pr))")
    ap.add_argument("--pr", type=float, default=None, help="Prandtl number (with --ra)")
    ap.add_argument("--k-ref", type=float, default=1.0,
                    help="Reference conductivity in the solver's units (default 1: kappa scaled by its value at T0)")
    dg = ap.add_mutually_exclusive_group()
    dg.add_argument("--delta-T", type=float, default=None,
                    help="Wall temperature difference in solver units (default: from the data)")
    dg.add_argument("--epsilon", type=float, default=None,
                    help="eps, with DeltaT = 2*eps (T scaled by T0, as in the paper)")
    ap.add_argument("--H", type=float, default=None, help="Cell height in solver units (default: Z extent)")
    ap.add_argument("--bulk-range", nargs=2, type=float, default=[0.0, 1.0], metavar=("LO", "HI"),
                    help="Fractions of the height averaged for Nu_bulk (default 0 1 = whole height)")
    ap.add_argument("--t-ref", type=float, default=None,
                    help="Constant subtracted from T in the convective term (default: mean of the two "
                         "wall temperatures; no physical effect, only reduces round-off)")
    ap.add_argument("--mode", choices=["auto", "columns", "fe"], default="auto")
    ap.add_argument("--z-tol", type=float, default=1e-8, help="Tolerance for treating two Z values as one level")
    ap.add_argument("--xy-tol", type=float, default=1e-6, help="Tolerance for matching (x,y) columns")
    ap.add_argument("--coord-tol", type=float, default=1e-6,
                    help="Tolerance for matching duplicate nodes at partition interfaces")
    ap.add_argument("--coord-check-tol", type=float, default=1e-5,
                    help="How close X,Y,Z must be to the reference grid to accept an iteration")
    ap.add_argument("--node-dtype", default="float64", choices=["float32", "float64"],
                    help="Precision the files are read in (default float64: gradients over closely "
                         "spaced wall levels amplify float32 round-off; float32 halves memory)")
    ap.add_argument("--checkpoint-dir", default=None,
                    help="Per-iteration results for restarting (default: <output-dir>/checkpoint_Nu)")
    ap.add_argument("--no-resume", action="store_true", help="Delete the checkpoint and recompute everything")
    ap.add_argument("--clean-checkpoint", action="store_true", help="Delete the checkpoint after a successful run")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--max-ram-gb", type=float, default=None)
    ap.add_argument("--safety-factor", type=float, default=4.0)
    ap.add_argument("--ram-margin", type=float, default=0.8)
    args = ap.parse_args()

    if args.pe is not None:
        pe = args.pe
        pe_note = f"--pe {pe:g}"
    elif args.ra is not None and args.pr is not None:
        pe = float(np.sqrt(args.ra * args.pr))
        pe_note = f"sqrt(Ra*Pr) = sqrt({args.ra:g}*{args.pr:g}) = {pe:.6g}"
    else:
        sys.exit("Give the Peclet number: --pe VALUE (1 for thermal-diffusion scaling) or --ra and --pr "
                 "(free-fall scaling, Pe = sqrt(Ra*Pr)). See --help.")
    if not (0.0 <= args.bulk_range[0] < args.bulk_range[1] <= 1.0):
        sys.exit("--bulk-range must satisfy 0 <= LO < HI <= 1")
    delta_T = 2.0 * args.epsilon if args.epsilon is not None else args.delta_T

    os.makedirs(args.output_dir, exist_ok=True)
    by_iter = discover_iterations(args.input_dir, args.pattern)
    if args.iters is not None:
        by_iter = {it: blocks for it, blocks in by_iter.items() if it in args.iters}
    if not by_iter:
        sys.exit("No matching iteration/block files found.")
    print(f"Found {len(by_iter)} iteration(s)")
    for it, blocks in by_iter.items():
        print(f"  iter {it}: {len(blocks)} block(s)")

    ckpt_dir = args.checkpoint_dir or os.path.join(args.output_dir, "checkpoint_Nu")
    lev = compute_level_means(
        by_iter, ckpt_dir, resume=not args.no_resume, mode=args.mode, t_ref=args.t_ref,
        coord_tol=args.coord_tol, coord_check_tol=args.coord_check_tol,
        z_tol=args.z_tol, xy_tol=args.xy_tol, node_dtype=np.dtype(args.node_dtype).type,
        workers=args.workers, max_ram_gb=args.max_ram_gb,
        safety_factor=args.safety_factor, ram_margin=args.ram_margin)
    print(f"\nAveraged {lev['n_accum']} iteration(s), {len(lev['z_levels'])} Z levels, "
          f"gradient mode: {lev['mode']}")

    r = nusselt_from_levels(lev["z_levels"], lev["conv"], lev["cond"], lev["Tm"], pe,
                            k_ref=args.k_ref, delta_T=delta_T, H=args.H,
                            bulk_range=tuple(args.bulk_range))

    out = args.output_dir
    np.save(os.path.join(out, "Nu_hot.npy"), np.float64(r["Nu_hot"]))
    np.save(os.path.join(out, "Nu_cold.npy"), np.float64(r["Nu_cold"]))
    np.save(os.path.join(out, "Nu_bulk.npy"), np.float64(r["Nu_bulk"]))
    prof = np.column_stack([lev["z_levels"], r["Nu_z"], r["Nu_conv_z"], r["Nu_cond_z"], lev["Tm"]])
    write_dat(os.path.join(out, "Nu_profile.dat"), prof, None,
              ["Z", "Nu", "Nu_conv", "Nu_cond", "T_mean"])

    lines = [
        f"iterations averaged : {lev['n_accum']}",
        f"hot wall            : {r['hot_at']}",
        f"Nu_hot              : {r['Nu_hot']:.8g}",
        f"Nu_cold             : {r['Nu_cold']:.8g}",
        f"Nu_bulk             : {r['Nu_bulk']:.8g}   (volume average of Nu(z) over z in "
        f"[{r['bulk_z_range'][0]:.6g}, {r['bulk_z_range'][1]:.6g}])",
        f"Nu(z) min/mean/max  : {r['Nu_z'].min():.8g} / {r['Nu_z'].mean():.8g} / {r['Nu_z'].max():.8g}",
        f"Nu(z) rel. spread   : {100 * r['rel_spread']:.3g} %   (should be ~0: Q is the same on every plane)",
        f"Pe                  : {pe_note}",
        f"k_ref, DeltaT, H    : {r['k_ref']:g}, {r['delta_T']:.8g}, {r['H']:.8g}",
        f"T_ref (convective)  : {lev['t_ref']:.8g}",
        f"gradient mode       : {lev['mode']}",
    ]
    with open(os.path.join(out, "Nu_summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nWrote Nu_hot.npy, Nu_cold.npy, Nu_bulk.npy, Nu_profile.dat, Nu_summary.txt -> {out}")

    if r["rel_spread"] > 0.05:
        print("\nWARNING: Nu(z) varies by more than 5% over the height. In a statistically steady closed "
              "cell Q is the same on every plane, so check (1) --pe / --ra --pr matches your solver's "
              "velocity scaling, (2) the run is statistically converged (more iterations), (3) the "
              "mesh/--mode (e.g. wall gradient accuracy with --mode fe).")

    if args.clean_checkpoint:
        import shutil
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        print(f"Removed checkpoint folder {ckpt_dir}")

    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 6))
        z = lev["z_levels"]
        ax.plot(r["Nu_z"], z, "k-", label="Nu (total)")
        ax.plot(r["Nu_conv_z"], z, "--", label="convective")
        ax.plot(r["Nu_cond_z"], z, ":", label="conductive")
        ax.axvline(r["Nu_bulk"], color="gray", lw=0.8)
        ax.set_xlabel("Nu(z)")
        ax.set_ylabel("Z")
        ax.set_title(f"Nu hot {r['Nu_hot']:.4g}, cold {r['Nu_cold']:.4g}, bulk {r['Nu_bulk']:.4g}")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        png = os.path.join(out, "Nu_profile.png")
        fig.savefig(png, dpi=args.dpi)
        plt.close(fig)
        print(f"Saved {png}")


if __name__ == "__main__":
    main()
