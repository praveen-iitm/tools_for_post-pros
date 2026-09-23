#!/usr/bin/env python3
"""
04_budget_terms.py
---------------------
Two-point TKE-budget ("E_*") and temperature-variance-budget ("H_*") terms
at a fixed reference point (x_idx, y_idx, z_idx), built from:

  - the mean fields (from 01_compute_mean.py)
  - the two-point correlations R_ij (from 03_two_point_correlation.py)
  - one additional streaming pass over every snapshot, for the two
    inter-scale-transport-due-to-fluctuations terms (E_inter_f, H_inter_f)

NOTE on the original notebook: the H_inter_f accumulation loop assigned
wt_u/wt_v/wt_w with "=" instead of "+=" each iteration (so only the last
snapshot's contribution survived the division by n_snaps). That looked
like a copy/paste bug relative to the otherwise-identical E_inter_f loop
(which does accumulate with "+="), so this script accumulates both
consistently. If you specifically need the original (buggy) behaviour,
flag it and it can be reproduced instead.

Also: E_inter_f's (xg, yg, zg) and H_inter_f's (wt_u, wt_v, wt_w) loops
read the exact same snapshot fluctuations, so this script merges them into
a SINGLE pass over the files instead of two, halving the I/O for this step.

Example
-------
    python 04_budget_terms.py \
        --solution-dir /media/user/Data2/kiran_data/Ra_2e9/Case/output \
        --grid-file    /media/user/Data2/kiran_data/Ra_2e9/Case/output/grid.xyz \
        --mean-dir     /media/user/Data2/kiran_data/post_proc_1-mean \
        --mean-tag 731-931 \
        --corr-dir     /media/user/Data2/kiran_data/post_proc_3-tp_25_222 \
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
from plt_3d_tools.io import read_grid, read_solution, list_solution_files
from plt_3d_tools.gradients import GradientOps
from plt_3d_tools.parallel import run_parallel_accumulation


# ---------------------------------------------------------------------
# Combined streaming pass: E_inter_f needs (xg, yg, zg) from u,v,w fluc;
# H_inter_f needs (wt_u, wt_v, wt_w) from u,v,w,T fluc. Same file reads,
# so compute all 6 accumulators together in one pass.
# ---------------------------------------------------------------------
def _init(shape, x_idx, y_idx, z_idx, u_mean_tz, v_mean_tz, w_mean_tz, T_mean_tz):
    keys = ('xg', 'yg', 'zg', 'wt_u', 'wt_v', 'wt_w')
    return {k: np.zeros(shape, dtype=np.float64) for k in keys}


def _process(fn, state, shape, x_idx, y_idx, z_idx, u_mean_tz, v_mean_tz, w_mean_tz, T_mean_tz):
    sol, nx, ny, nz, nvar, var_names = read_solution(fn)

    u_f = (sol[:, :, :, 0] - u_mean_tz).astype(np.float32, copy=False)
    v_f = (sol[:, :, :, 1] - v_mean_tz).astype(np.float32, copy=False)
    w_f = (sol[:, :, :, 2] - w_mean_tz).astype(np.float32, copy=False)
    T_f = (sol[:, :, :, 4] - T_mean_tz).astype(np.float32, copy=False)
    del sol

    u_ref, v_ref, w_ref, T_ref = (u_f[x_idx, y_idx, z_idx], v_f[x_idx, y_idx, z_idx],
                                   w_f[x_idx, y_idx, z_idx], T_f[x_idx, y_idx, z_idx])

    state['xg'] += (u_ref * u_f * (u_ref - u_f) + v_ref * v_f * (u_ref - u_f) + w_ref * w_f * (u_ref - u_f))
    state['yg'] += (u_ref * u_f * (v_ref - v_f) + v_ref * v_f * (v_ref - v_f) + w_ref * w_f * (v_ref - v_f))
    state['zg'] += (u_ref * u_f * (w_ref - w_f) + v_ref * v_f * (w_ref - w_f) + w_ref * w_f * (w_ref - w_f))

    state['wt_u'] += w_ref * T_f * (u_ref - u_f)
    state['wt_v'] += w_ref * T_f * (v_ref - v_f)
    state['wt_w'] += w_ref * T_f * (w_ref - w_f)

    del u_f, v_f, w_f, T_f


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--solution-dir', required=True)
    ap.add_argument('--grid-file', required=True, help='Path to grid.xyz')
    ap.add_argument('--mean-dir', required=True, help='Directory with mean_u/v/w/T_*.npy from script 01')
    ap.add_argument('--mean-tag', default='')
    ap.add_argument('--corr-dir', required=True, help='Directory with R_*.npy from script 03')
    ap.add_argument('--output-dir', required=True, help='Parent dir; results go in <output-dir>/post-proc-4_terms_calc_<x>_<y>_<z>/')
    ap.add_argument('--x-idx', type=int, required=True)
    ap.add_argument('--y-idx', type=int, required=True)
    ap.add_argument('--z-idx', type=int, default=None, help='Default: nz // 2')
    ap.add_argument('--workers', type=int, default=None)
    args = ap.parse_args()

    x, y, z, nx, ny, nz = read_grid(args.grid_file)
    x_idx, y_idx = args.x_idx, args.y_idx
    z_idx = args.z_idx if args.z_idx is not None else nz // 2

    directory_terms = os.path.join(args.output_dir, f"post-proc-4_terms_calc_{x_idx}_{y_idx}_{z_idx}")
    e_dir = os.path.join(directory_terms, "e_terms")
    h_dir = os.path.join(directory_terms, "h_terms")
    os.makedirs(e_dir, exist_ok=True)
    os.makedirs(h_dir, exist_ok=True)

    suffix = f"_{args.mean_tag}" if args.mean_tag else ""
    u_mean_t = np.load(os.path.join(args.mean_dir, f"mean_u{suffix}.npy"))
    v_mean_t = np.load(os.path.join(args.mean_dir, f"mean_v{suffix}.npy"))
    w_mean_t = np.load(os.path.join(args.mean_dir, f"mean_w{suffix}.npy"))
    T_mean_t = np.load(os.path.join(args.mean_dir, f"mean_T{suffix}.npy"))

    def z_avg_broadcast(a):
        return np.repeat(np.mean(a, axis=2)[..., np.newaxis], nz, axis=2).astype(np.float32)

    u_mean_tz = z_avg_broadcast(u_mean_t)
    v_mean_tz = z_avg_broadcast(v_mean_t)
    w_mean_tz = z_avg_broadcast(w_mean_t)
    T_mean_tz = z_avg_broadcast(T_mean_t)

    def load_R(key):
        return np.load(os.path.join(args.corr_dir, f"R_{key}.npy"))

    R_uu, R_uv, R_uw, R_uT = load_R('uu'), load_R('uv'), load_R('uw'), load_R('uT')
    R_vu, R_vv, R_vw, R_vT = load_R('vu'), load_R('vv'), load_R('vw'), load_R('vT')
    R_wu, R_wv, R_ww, R_wT = load_R('wu'), load_R('wv'), load_R('ww'), load_R('wT')
    R_Tu, R_Tv, R_Tw, R_TT = load_R('Tu'), load_R('Tv'), load_R('Tw'), load_R('TT')

    g = GradientOps(x, y, z, (x_idx, y_idx, z_idx))
    co_eff = g.co_eff
    ridx = (x_idx, y_idx, z_idx)

    # -------------------------------------------------------------
    # Terms that need only the mean fields + R_ij (no snapshot loop)
    # -------------------------------------------------------------
    gx_u, gy_v, gz_w = g.grad_x(u_mean_tz), g.grad_y(v_mean_tz), g.grad_z(w_mean_tz)
    gy_u, gx_v = g.grad_y(u_mean_tz), g.grad_x(v_mean_tz)
    gz_u, gx_w = g.grad_z(u_mean_tz), g.grad_x(w_mean_tz)
    gz_v, gy_w = g.grad_z(v_mean_tz), g.grad_y(w_mean_tz)
    gx_T, gy_T, gz_T = g.grad_x(T_mean_tz), g.grad_y(T_mean_tz), g.grad_z(T_mean_tz)

    # Homogeneous Production: E_prod_h
    E_prod_h = 0.5 * (g.grad_r(R_uu) * 2 * gx_u[ridx] +
                       g.grad_r(R_vv) * 2 * gy_v[ridx] +
                       g.grad_r(R_ww) * 2 * gz_w[ridx] +
                       g.grad_r(R_uv) * (gx_v[ridx] + gy_u[ridx]) +
                       g.grad_r(R_uw) * (gx_w[ridx] + gz_u[ridx]) +
                       g.grad_r(R_vu) * (gy_u[ridx] + gx_v[ridx]) +
                       g.grad_r(R_vw) * (gy_w[ridx] + gz_v[ridx]) +
                       g.grad_r(R_wu) * (gz_u[ridx] + gx_w[ridx]) +
                       g.grad_r(R_wv) * (gz_v[ridx] + gy_w[ridx]))
    E_prod_h = co_eff * E_prod_h
    np.save(os.path.join(e_dir, "E_prod_h"), E_prod_h)
    del E_prod_h

    # Inhomogeneous Production: E_prod_I
    E_prod_I = 0.5 * g.grad_r(R_uu * (gx_u - gx_u[ridx]) + R_vv * (gy_v - gy_v[ridx]) +
                               R_ww * (gz_w - gz_w[ridx]) + R_uv * (gy_u - gy_u[ridx]) +
                               R_uw * (gz_u - gz_u[ridx]) + R_vu * (gx_v - gx_v[ridx]) +
                               R_vw * (gz_v - gz_v[ridx]) + R_wu * (gx_w - gx_w[ridx]) +
                               R_wv * (gy_w - gy_w[ridx]))
    E_prod_I = co_eff * E_prod_I
    np.save(os.path.join(e_dir, "E_prod_I"), E_prod_I)
    del E_prod_I
    del gx_u, gy_v, gz_w, gy_u, gx_v, gz_u, gx_w, gz_v, gy_w

    # Interscale Transport due to mean velocity: E_inter_m
    ux_ur = u_mean_tz[ridx] - u_mean_tz
    vx_vr = v_mean_tz[ridx] - v_mean_tz
    wx_wr = w_mean_tz[ridx] - w_mean_tz
    E_inter_m = 0.5 * g.grad_r(g.grad_x(R_uu * ux_ur + R_vv * ux_ur + R_ww * ux_ur) +
                                g.grad_y(R_uu * vx_vr + R_vv * vx_vr + R_ww * vx_vr) +
                                g.grad_z(R_uu * wx_wr + R_vv * wx_wr + R_ww * wx_wr))
    E_inter_m = co_eff * E_inter_m
    np.save(os.path.join(e_dir, "E_inter_m"), E_inter_m)
    del E_inter_m, ux_ur, vx_vr, wx_wr

    # Buoyancy source term: E_buoy
    E_buoy = co_eff * (0.5 * g.grad_r(R_wT + R_Tw) * (-1))
    np.save(os.path.join(e_dir, "E_buoy"), E_buoy)
    del E_buoy

    # Homogeneous Production: H_prod_h
    gx_w, gy_w, gz_w = g.grad_x(w_mean_tz), g.grad_y(w_mean_tz), g.grad_z(w_mean_tz)
    gx_T2, gy_T2, gz_T2 = g.grad_x(T_mean_tz), g.grad_y(T_mean_tz), g.grad_z(T_mean_tz)

    H_prod_h = co_eff * (g.grad_r(R_uT) * gx_w[ridx] + g.grad_r(R_vT) * gy_w[ridx] + g.grad_r(R_wT) * gz_w[ridx] +
                          g.grad_r(R_wu) * gx_T2[ridx] + g.grad_r(R_wv) * gy_T2[ridx] + g.grad_r(R_ww) * gz_T2[ridx])
    np.save(os.path.join(h_dir, "H_prod_h"), H_prod_h)
    del H_prod_h, gx_w, gy_w, gz_w

    # Inhomogeneous Production: H_prod_I
    H_prod_I = g.grad_r(R_wu * (gx_T2 - gx_T2[ridx]) + R_wv * (gy_T2 - gy_T2[ridx]) + R_ww * (gz_T2 - gz_T2[ridx]))
    H_prod_I = co_eff * H_prod_I
    np.save(os.path.join(h_dir, "H_prod_I"), H_prod_I)
    del H_prod_I, gx_T2, gy_T2, gz_T2

    # Interscale Transport due to mean velocity: H_inter_m
    H_inter_m = g.grad_r(g.grad_x(R_wT * (u_mean_tz[ridx] - u_mean_tz)) +
                          g.grad_y(R_wT * (v_mean_tz[ridx] - v_mean_tz)) +
                          g.grad_z(R_wT * (w_mean_tz[ridx] - w_mean_tz)))
    H_inter_m = co_eff * H_inter_m
    np.save(os.path.join(h_dir, "H_inter_m"), H_inter_m)
    del H_inter_m

    # Buoyancy source term: H_buoy
    H_buoy = co_eff * (g.grad_r(R_TT) * (-1))
    np.save(os.path.join(h_dir, "H_buoy"), H_buoy)
    del H_buoy

    del R_uu, R_uv, R_uw, R_uT, R_vu, R_vv, R_vw, R_vT
    del R_wu, R_wv, R_ww, R_wT, R_Tu, R_Tv, R_Tw, R_TT

    # -------------------------------------------------------------
    # Terms that need a streaming pass over every snapshot
    # (E_inter_f, H_inter_f) -- done together in one pass
    # -------------------------------------------------------------
    files = list_solution_files(args.solution_dir)
    if not files:
        sys.exit(f"No grid*.f files found in {args.solution_dir}")
    print(f"Found {len(files)} snapshot files in {args.solution_dir}")

    extra = ((nx, ny, nz), x_idx, y_idx, z_idx, u_mean_tz, v_mean_tz, w_mean_tz, T_mean_tz)
    acc = run_parallel_accumulation(files, _init, _process, n_workers=args.workers, extra=extra)
    n_snaps = len(files)
    for k in acc:
        acc[k] /= n_snaps

    E_inter_f = co_eff * (0.5 * g.grad_r(g.grad_x(acc['xg']) + g.grad_y(acc['yg']) + g.grad_z(acc['zg'])))
    np.save(os.path.join(e_dir, "E_inter_f"), E_inter_f)
    del E_inter_f

    H_inter_f = co_eff * g.grad_r(g.grad_x(acc['wt_u']) + g.grad_y(acc['wt_v']) + g.grad_z(acc['wt_w']))
    np.save(os.path.join(h_dir, "H_inter_f"), H_inter_f)
    del H_inter_f

    print(f"Done. Wrote budget terms to {directory_terms}")


if __name__ == '__main__':
    main()
