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
    N_BL_bottom.npy, N_BL_top.npy
                      number of grid points (Z levels) inside the bottom / top thermal
                      boundary layer, as in the paper's N^b_BL, N^t_BL (Table 4).
                      BL_thickness_bottom.npy, BL_thickness_top.npy: the thicknesses.
    Nu_V.npy          (with --nu-v) Nu_V = sqrt(Ra Pr)<eps_u>_{V,t} + 1 from the
                      kinetic-energy dissipation rate, as in the paper's Table 4
    Nu_series.dat     per-snapshot Nu_hot, Nu_cold, Nu_bulk (is the run stationary?)
    Nu_summary.txt    the numbers, diagnostics and all settings used
    Nu_profile.png    Nu(z) and its parts, and the per-snapshot series (--no-plot skips)

Thermal boundary layer (always computed, from the time- and plane-averaged
T(z)): slope method, as in the paper's sec. 3.4.2 ("defined using the centre
temperature T_c") -- delta is the distance from the plate at which the
tangent of <T>(z) at the plate reaches the centre temperature T_c (<T> at
mid-height); N_BL counts the Z levels within that distance of the plate, the
plate level included (--bl-exclude-wall to leave it out; both counts are in
the summary -- the paper does not state its counting convention).
Bottom/top, not hot/cold, as in the paper. The paper's Nu (Tables 3-4) is the
one at the bottom (hot) plate, i.e. Nu_hot.

Nu_V (--nu-v, needs --pr and the connectivity; reads U, V, mu as well): with
the velocities in free-fall units, Nu_V = 1 + Pr * <Phi>_{V,t}, where
Phi = mu*[d_i u_j (d_i u_j + d_j u_i) - 2/3 (div u)^2] is the viscous
dissipation per unit volume (mu in units of its reference value; Ra cancels
out of sqrt(Ra Pr) * sqrt(Pr/Ra)). Gradients are per-tetrahedron (piecewise
constant), volume-weighted. NOTE: the exact relation behind Nu_V holds for
Oberbeck-Boussinesq flow only -- the paper lists Nu_V for its OB cases and
none for epsilon > 0, so for a NOB run this is a dissipation-based estimate,
not a quantity to compare with the paper.

Diagnostics (printed and in Nu_summary.txt) for when Nu(z) is not flat:
    * hot/cold imbalance and drift over the snapshots: in a stationary state
      Nu_hot = Nu_cold and neither trends with time; a trend or imbalance
      means the averaged snapshots include a transient (or too few).
    * Pe self-check: best-fit Pe_eff/Pe_used from forcing Nu(z) flat. ~1 means
      the Pe is consistent; far from 1 means the velocity scaling is off.
    * mass-flux term: the amount Nu(z) would change if T_ref were 0 instead of
      the mean wall T; large => <rho w> is not ~0 (non-stationary, or the
      solver's continuity/energy formulation differs from the assumed one).
    * horizontal spacing uniformity (the horizontal mean is a plain node mean).

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
from nob_tools.nusselt import (compute_level_means, nusselt_from_levels, nusselt_series,
                               pe_self_check, thermal_bl, nu_from_dissipation)


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
    ap.add_argument("--nu-v", action="store_true",
                    help="Also compute Nu_V from the viscous dissipation (needs --pr; reads U,V,mu and "
                         "the connectivity: more RAM and time; an existing checkpoint made without it "
                         "must be recomputed, --no-resume)")
    ap.add_argument("--bl-exclude-wall", action="store_true",
                    help="Do not count the plate level itself in N_BL")
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
    if args.nu_v and args.pr is None:
        sys.exit("--nu-v needs the Prandtl number: add --pr VALUE")

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
        dissipation=args.nu_v,
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
    z = lev["z_levels"]
    sgn = 1.0 if r["hot_at"] == "bottom" else -1.0
    den = r["k_ref"] * r["delta_T"] / r["H"]
    np.save(os.path.join(out, "Nu_hot.npy"), np.float64(r["Nu_hot"]))
    np.save(os.path.join(out, "Nu_cold.npy"), np.float64(r["Nu_cold"]))
    np.save(os.path.join(out, "Nu_bulk.npy"), np.float64(r["Nu_bulk"]))
    cols = [z, r["Nu_z"], r["Nu_conv_z"], r["Nu_cond_z"], lev["Tm"]]
    names = ["Z", "Nu", "Nu_conv", "Nu_cond", "T_mean"]
    shift = None
    if lev["mflux"] is not None:
        shift = sgn * pe * lev["t_ref"] * lev["mflux"] / den      # Nu change if T_ref were 0
        cols.append(shift)
        names.append("Nu_Tref_shift")
    write_dat(os.path.join(out, "Nu_profile.dat"), np.column_stack(cols), None, names)

    # --- thermal boundary layers, Nu_V ------------------------------------
    bl = thermal_bl(z, lev["Tm"], count_wall=not args.bl_exclude_wall)
    np.save(os.path.join(out, "N_BL_bottom.npy"), np.int64(bl["N_b"]))
    np.save(os.path.join(out, "N_BL_top.npy"), np.int64(bl["N_t"]))
    np.save(os.path.join(out, "BL_thickness_bottom.npy"), np.float64(bl["delta_b"]))
    np.save(os.path.join(out, "BL_thickness_top.npy"), np.float64(bl["delta_t"]))
    nuv = None
    if args.nu_v:
        nuv = nu_from_dissipation(lev["diss"], args.pr)
        np.save(os.path.join(out, "Nu_V.npy"), np.float64(nuv))

    # --- diagnostics ------------------------------------------------------
    series = nusselt_series(lev, pe, r)
    write_dat(os.path.join(out, "Nu_series.dat"), series, None,
              ["iteration", "Nu_hot", "Nu_cold", "Nu_bulk"])
    n = len(series)
    order = np.arange(n)

    def stat(col):
        v = series[:, col]
        sem = v.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
        drift = float(np.polyfit(order, v, 1)[0] * (n - 1)) if n > 2 else float("nan")
        return v.mean(), v.std(ddof=1) if n > 1 else float("nan"), sem, drift
    hot_m, hot_sd, hot_sem, hot_dr = stat(1)
    cold_m, cold_sd, cold_sem, cold_dr = stat(2)
    bulk_m, bulk_sd, bulk_sem, bulk_dr = stat(3)
    imbalance = abs(r["Nu_hot"] - r["Nu_cold"]) / (0.5 * (r["Nu_hot"] + r["Nu_cold"]))
    f_pe, q_fit, res_fit = pe_self_check(r["Nu_conv_z"], r["Nu_cond_z"], z)
    xy = lev["xy_spacing"]

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
        "",
        "--- thermal boundary layers (slope method on <T>(z)) ---",
        f"T_c (mid-height)    : {bl['Tc']:.8g}",
        f"bottom BL           : delta = {bl['delta_b']:.6g}, N_BL^b = {bl['N_b_with_wall']} levels "
        f"including the plate ({bl['N_b_no_wall']} without)",
        f"top BL              : delta = {bl['delta_t']:.6g}, N_BL^t = {bl['N_t_with_wall']} levels "
        f"including the plate ({bl['N_t_no_wall']} without)",
        f"saved as            : N_BL_bottom = {bl['N_b']}, N_BL_top = {bl['N_t']}",
    ]
    if nuv is not None:
        lines += [f"Nu_V                : {nuv:.8g}   (= 1 + Pr*<Phi>, <Phi> = {lev['diss']:.8g}, Pr = {args.pr:g}; "
                  "exact only for OB flow)"]
    lines += [
        "",
        "--- diagnostics ---",
        f"hot/cold imbalance  : {100 * imbalance:.3g} %   (should be ~0 when stationary)",
        f"Nu_hot  over snapshots: std {hot_sd:.3g}, s.e.m. {hot_sem:.3g} (assumes independent snapshots), "
        f"drift over the series {hot_dr:+.3g}",
        f"Nu_cold over snapshots: std {cold_sd:.3g}, s.e.m. {cold_sem:.3g}, drift over the series {cold_dr:+.3g}",
        f"Nu_bulk over snapshots: std {bulk_sd:.3g}, s.e.m. {bulk_sem:.3g}, drift over the series {bulk_dr:+.3g}",
        f"Pe self-check       : best-fit Pe_eff / Pe_used = {f_pe:.3g}; flux at that fit {q_fit:.4g}; "
        f"residual rms {100 * res_fit:.3g} % of it",
    ]
    if shift is not None:
        lines.append(f"mass-flux term      : max |Pe*T_ref*<rho cp w>| = {np.abs(shift).max():.3g} in Nu units "
                     f"(Nu(z) change if T_ref were 0)")
    else:
        lines.append("mass-flux term      : n/a (checkpoint predates this diagnostic; rerun with --no-resume)")
    if xy is not None:
        lines.append(f"horizontal spacing  : {xy['n_x']} x {xy['n_y']} columns, max/min dx = "
                     f"{xy['dx_ratio']:.3g}, max/min dy = {xy['dy_ratio']:.3g}")
    else:
        lines.append("horizontal spacing  : n/a (not a tensor-product column grid)")
    with open(os.path.join(out, "Nu_summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nWrote Nu_hot.npy, Nu_cold.npy, Nu_bulk.npy, N_BL_bottom.npy, N_BL_top.npy"
          f"{', Nu_V.npy' if nuv is not None else ''}, BL_thickness_*.npy, Nu_profile.dat, Nu_series.dat, "
          f"Nu_summary.txt -> {out}")

    if xy is not None and max(xy["dx_ratio"], xy["dy_ratio"]) > 1.05:
        print("\nNOTE: the horizontal mesh is non-uniform, so the plain mean over nodes used for every "
              "horizontal average is not an area average and can bias <rho w T>.")
    def drifts(drift, sem, mean):          # beyond 2 s.e.m. AND beyond 0.1 % of the mean
        return n > 2 and abs(drift) > 2 * sem and abs(drift) > 1e-3 * abs(mean)
    if drifts(hot_dr, hot_sem, hot_m) or drifts(cold_dr, cold_sem, cold_m) or drifts(bulk_dr, bulk_sem, bulk_m):
        print("\nNOTE: Nu_hot / Nu_cold / Nu_bulk drift over the averaged snapshots by more than twice their "
              "standard error -- the series probably includes a transient. Look at Nu_series.dat and "
              "consider averaging only the later iterations (--iters).")

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
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 6))
        ax.plot(r["Nu_z"], z, "k-", label="Nu (total)")
        ax.plot(r["Nu_conv_z"], z, "--", label="convective")
        ax.plot(r["Nu_cond_z"], z, ":", label="conductive")
        ax.axvline(r["Nu_bulk"], color="gray", lw=0.8)
        ax.set_xlabel("Nu(z)")
        ax.set_ylabel("Z")
        ax.set_title(f"Nu hot {r['Nu_hot']:.4g}, cold {r['Nu_cold']:.4g}, bulk {r['Nu_bulk']:.4g}")
        ax.grid(alpha=0.3)
        ax.legend()
        ax2.plot(series[:, 0], series[:, 1], "r.-", ms=3, label="Nu_hot")
        ax2.plot(series[:, 0], series[:, 2], "b.-", ms=3, label="Nu_cold")
        ax2.plot(series[:, 0], series[:, 3], "g.-", ms=3, label="Nu_bulk")
        ax2.set_xlabel("iteration")
        ax2.set_ylabel("Nu per snapshot")
        ax2.set_title("per-snapshot values (flat = stationary)")
        ax2.grid(alpha=0.3)
        ax2.legend()
        fig.tight_layout()
        png = os.path.join(out, "Nu_profile.png")
        fig.savefig(png, dpi=args.dpi)
        plt.close(fig)
        print(f"Saved {png}")


if __name__ == "__main__":
    main()
