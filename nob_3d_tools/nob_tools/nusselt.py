"""
Nusselt-number analysis for Rayleigh-Benard convection with variable
properties (NOB), from Tecplot partition files.

Heat flux across a horizontal plane (z vertical), per unit area, in the
solver's own (dimensionless) units:

    q(z) = Pe * < rho cp w (T - T_ref) >_{xy,t}  -  < kappa dT/dz >_{xy,t}

where < >_xy is the plain arithmetic mean over the nodes on that Z level
and < >_t the mean over iterations. Pe = U_ref H / kappa_ref turns the
convective term into the same units as the conductive one (Pe = 1 for
thermal-diffusion velocity scaling; sqrt(Ra Pr) for free-fall scaling).
Subtracting the constant T_ref changes nothing physically (the net mass
flux through a plane of a closed cell is zero) but removes an O(1) x
(numerical mass flux) error when T is scaled by T0 (T ~ 1 +- eps).

    Nu = q / (k_ref * DeltaT / H)

Time averaging has to be done on the PRODUCT, per iteration -- the
convective flux is almost entirely a fluctuation correlation, so it can
not be recovered from time-averaged fields.

Everything large stays on disk (same restartable design as the time
average): each iteration is reduced to a few length-(n_levels) arrays,
written atomically to the checkpoint folder, and the final result is the
mean over those small files.

Two ways to get dT/dz at the nodes:
  columns : the mesh has shared Z levels AND the same (x,y) points on
            every level (structured/extruded). Non-uniform 3-point finite
            differences along each column: second order, including the
            one-sided stencils at the walls.
  fe      : any mesh. Per-tetrahedron gradient (exact for linear
            fields), volume-averaged onto nodes. Needs connectivity.
            At wall nodes only the adjacent elements contribute, so the
            wall gradient is first-order accurate.
Either way, horizontal averaging needs shared Z levels.
"""

import gc
import json
import os
import time

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from .assemble import assemble_iteration_columns
from .discovery import ALL_VAR_NAMES
from .memory_sizing import choose_worker_count
from .time_average import reference_connectivity, _save_atomic

COL_NAMES = ("W", "T", "rho", "cp", "kappa")
COLS = [ALL_VAR_NAMES.index(n) for n in COL_NAMES]   # data columns 3..7 after X,Y,Z
T_VAR_IDX = ALL_VAR_NAMES.index("T")


# --------------------------------------------------------------------------
# Layout detection and gradients
# --------------------------------------------------------------------------

def detect_layout(xyz, z_tol=1e-8, xy_tol=1e-6):
    """
    Group nodes into horizontal levels (shared Z within z_tol) and test
    whether every level has the same (x,y) points (a column structure).

    Returns dict: z_levels (L,), level_idx (N,), counts (L,), n_levels,
    shared_levels (bool), col_idx ((L, C) node indices, columns in the
    same (x,y) order on every level -- or None if not a column structure).
    """
    N = xyz.shape[0]
    z_keys = np.round(xyz[:, 2] / z_tol).astype(np.int64)
    uniq, level_idx, counts = np.unique(z_keys, return_inverse=True, return_counts=True)
    level_idx = level_idx.reshape(-1)
    L = len(uniq)
    z_levels = np.bincount(level_idx, weights=xyz[:, 2], minlength=L) / counts

    out = {"z_levels": z_levels, "level_idx": level_idx, "counts": counts,
           "n_levels": L, "shared_levels": L <= 0.5 * N, "col_idx": None}
    if not out["shared_levels"] or L < 3:
        return out

    if (counts == counts[0]).all():
        C = int(counts[0])
        kx = np.round(xyz[:, 0] / xy_tol).astype(np.int64)
        ky = np.round(xyz[:, 1] / xy_tol).astype(np.int64)
        order = np.lexsort((ky, kx, level_idx))        # by level, then x, then y
        kx_s = kx[order].reshape(L, C)
        ky_s = ky[order].reshape(L, C)
        if (kx_s == kx_s[0]).all() and (ky_s == ky_s[0]).all():
            out["col_idx"] = order.reshape(L, C)
    return out


def fd_z(Tg, z):
    """
    d/dz of Tg (L, C) along axis 0 on the (possibly non-uniform) levels z
    (L,): 3-point central differences inside, second-order one-sided
    3-point stencils at the two ends. Exact for quadratics.
    """
    L = Tg.shape[0]
    if L < 3:
        raise ValueError("need at least 3 Z levels for second-order differences")
    d = np.empty_like(Tg, dtype=np.float64)
    h1 = (z[1:-1] - z[:-2])[:, None]
    h2 = (z[2:] - z[1:-1])[:, None]
    d[1:-1] = (-h2 / (h1 * (h1 + h2)) * Tg[:-2]
               + (h2 - h1) / (h1 * h2) * Tg[1:-1]
               + h1 / (h2 * (h1 + h2)) * Tg[2:])
    h1, h2 = z[1] - z[0], z[2] - z[1]
    d[0] = (-(2 * h1 + h2) / (h1 * (h1 + h2)) * Tg[0]
            + (h1 + h2) / (h1 * h2) * Tg[1]
            - h1 / (h2 * (h1 + h2)) * Tg[2])
    h1, h2 = z[-2] - z[-3], z[-1] - z[-2]
    d[-1] = (h2 / (h1 * (h1 + h2)) * Tg[-3]
             - (h1 + h2) / (h1 * h2) * Tg[-2]
             + (h1 + 2 * h2) / (h2 * (h1 + h2)) * Tg[-1])
    return d


def build_fe_operator(xyz, conn, chunk=2_000_000):
    """
    Precompute what is needed to get a node-based dT/dz from a node field T
    on a tetrahedral mesh: per-element z-gradient coefficients `a` (E,3)
    such that dT/dz|_e = a . (T1-T0, T2-T0, T3-T0), element volumes, and
    each node's summed adjacent volume. conn is 0-based (E,4).
    """
    N, E = xyz.shape[0], conn.shape[0]
    x = xyz.astype(np.float64, copy=False)
    a = np.empty((E, 3), dtype=np.float64)
    vol = np.empty(E, dtype=np.float64)
    for s in range(0, E, chunk):
        c = conn[s:s + chunk]
        p0 = x[c[:, 0]]
        D = np.stack([x[c[:, 1]] - p0, x[c[:, 2]] - p0, x[c[:, 3]] - p0], axis=1)   # rows = edges
        det = np.linalg.det(D)
        if (det == 0).any():
            raise ValueError("degenerate (zero-volume) tetrahedron in connectivity")
        a[s:s + chunk] = np.linalg.inv(D)[:, 2, :]          # g_z = Dinv[2,:] . dT
        vol[s:s + chunk] = np.abs(det) / 6.0
    node_vol = np.zeros(N, dtype=np.float64)
    for j in range(4):
        node_vol += np.bincount(conn[:, j], weights=vol, minlength=N)
    return {"conn": conn, "a": a, "vol": vol, "node_vol": node_vol, "n_nodes": N}


def grad_z_fe(T, op):
    c = op["conn"]
    t0 = T[c[:, 0]]
    gz_e = (op["a"][:, 0] * (T[c[:, 1]] - t0)
            + op["a"][:, 1] * (T[c[:, 2]] - t0)
            + op["a"][:, 2] * (T[c[:, 3]] - t0))
    w_e = op["vol"] * gz_e
    g = np.zeros(op["n_nodes"], dtype=np.float64)
    for j in range(4):
        g += np.bincount(c[:, j], weights=w_e, minlength=op["n_nodes"])
    return g / op["node_vol"]


# --------------------------------------------------------------------------
# Per-iteration reduction
# --------------------------------------------------------------------------

def level_stats(data, ctx):
    """
    data: (N, 8) = X,Y,Z,W,T,rho,cp,kappa in reference node order.
    Returns (conv, cond, Tm), each (L,): level means of rho*cp*w*(T-T_ref),
    kappa*dT/dz, and T.
    """
    w, T, rho, cp, kap = (data[:, 3 + i].astype(np.float64) for i in range(5))
    conv_n = rho * cp * w * (T - ctx["t_ref"])
    if ctx["mode"] == "columns":
        idx = ctx["col_idx"]
        Tg = T[idx]
        dT = fd_z(Tg, ctx["z_levels"])
        cond = (kap[idx] * dT).mean(axis=1)
        conv = conv_n[idx].mean(axis=1)
        Tm = Tg.mean(axis=1)
    else:
        g = grad_z_fe(T, ctx["fe"])
        lv, cnt, L = ctx["level_idx"], ctx["counts"], len(ctx["counts"])
        cond = np.bincount(lv, weights=kap * g, minlength=L) / cnt
        conv = np.bincount(lv, weights=conv_n, minlength=L) / cnt
        Tm = np.bincount(lv, weights=T, minlength=L) / cnt
    return conv, cond, Tm


_CTX = None  # set once per worker process via the pool initializer


def _init_worker(ctx):
    global _CTX
    _CTX = ctx


def _iter_path(ckpt_dir, it):
    return os.path.join(ckpt_dir, f"iter_{it}.npz")


def _save_iter(path, conv, cond, Tm):
    tmp = path[:-4] + ".tmp.npz"
    np.savez(tmp, conv=conv, cond=cond, Tm=Tm)
    os.replace(tmp, path)


def _process_iteration(it, blocks, coord_tol, node_dtype, coord_check_tol, out_path):
    """Worker: read+merge one iteration, check its grid, reduce, save."""
    try:
        data, _ = assemble_iteration_columns(
            blocks, COLS, False, coord_tol, node_dtype, np.int32, log_prefix=f"[iter {it}] ")
        ref = _CTX["xyz_ref"]
        if data.shape[0] != ref.shape[0]:
            return {"iter": it, "status": "error",
                    "error": f"point count {data.shape[0]:,} != reference {ref.shape[0]:,}"}
        max_diff = float(np.abs(data[:, :3] - ref).max())
        if max_diff > coord_check_tol:
            return {"iter": it, "status": "error",
                    "error": f"coordinates differ from the reference grid by up to {max_diff:.3g} "
                             f"(tolerance {coord_check_tol:.3g})"}
        conv, cond, Tm = level_stats(data, _CTX)
        del data
        gc.collect()
        _save_iter(out_path, conv, cond, Tm)
        return {"iter": it, "status": "ok"}
    except Exception as e:
        return {"iter": it, "status": "error", "error": str(e)}


def _make_ctx(xyz_ref, layout, mode, t_ref, fe=None):
    ctx = {"xyz_ref": xyz_ref, "mode": mode, "t_ref": float(t_ref),
           "z_levels": layout["z_levels"], "level_idx": layout["level_idx"],
           "counts": layout["counts"], "col_idx": layout["col_idx"], "fe": fe}
    return ctx


def compute_level_means(by_iter, ckpt_dir, resume=True, mode="auto", t_ref=None,
                        coord_tol=1e-6, coord_check_tol=1e-5, z_tol=1e-8, xy_tol=1e-6,
                        node_dtype=np.float64, workers=None, max_ram_gb=None,
                        safety_factor=4.0, ram_margin=0.8, verbose=True):
    """
    Time-average, per Z level, the convective term rho*cp*w*(T-T_ref), the
    conductive term kappa*dT/dz and T, restartably (see module docstring).

    Returns dict: z_levels, conv, cond, Tm (all (L,)), n_accum, mode,
    t_ref, n_points.
    """
    items = list(by_iter.items())
    if not items:
        raise ValueError("by_iter is empty -- nothing to average")
    if mode not in ("auto", "columns", "fe"):
        raise ValueError(f"mode must be auto/columns/fe, got {mode!r}")
    os.makedirs(ckpt_dir, exist_ok=True)
    ref_path = os.path.join(ckpt_dir, "ref_xyz.npy")
    meta_path = os.path.join(ckpt_dir, "meta.json")

    def _wipe():
        for fn in os.listdir(ckpt_dir):
            if fn.startswith("iter_") or fn in ("ref_xyz.npy", "meta.json", "ref_conn.npy"):
                os.remove(os.path.join(ckpt_dir, fn))

    if not resume:
        _wipe()
    have_ref = os.path.isfile(ref_path) and os.path.isfile(meta_path)
    meta = None
    if have_ref:
        with open(meta_path) as f:
            meta = json.load(f)
        if not meta.get("dedup_v2") or meta.get("kind") != "nusselt":
            if verbose:
                print(f"Discarding old/foreign checkpoint in {ckpt_dir}. Recomputing.")
            _wipe()
            have_ref, meta = False, None
        elif mode != "auto" and meta["mode"] != mode:
            raise ValueError(f"checkpoint in {ckpt_dir} was made with --mode {meta['mode']}, "
                             f"not {mode}. Use --no-resume to start over.")
        elif t_ref is not None and abs(meta["t_ref"] - t_ref) > 1e-12:
            raise ValueError(f"checkpoint in {ckpt_dir} used t_ref={meta['t_ref']}, not {t_ref}. "
                             "Use --no-resume to start over.")
    if not have_ref:
        _wipe()

    first_data = None
    if have_ref:
        xyz_ref = np.load(ref_path)
        layout = detect_layout(xyz_ref, z_tol, xy_tol)
        mode_used, t_ref_used = meta["mode"], meta["t_ref"]
        if verbose:
            done = sum(os.path.isfile(_iter_path(ckpt_dir, it)) for it, _ in items)
            print(f"Resuming from {ckpt_dir}: {xyz_ref.shape[0]:,} points, mode={mode_used}, "
                  f"{done}/{len(items)} iteration(s) already done")
    else:
        first_it, first_blocks = items[0]
        if verbose:
            print(f"Computing reference grid from iter {first_it} ({len(first_blocks)} block(s))...")
        first_data, _ = assemble_iteration_columns(
            first_blocks, COLS, False, coord_tol, node_dtype, np.int32,
            log_prefix=f"[iter {first_it}] ")
        xyz_ref = first_data[:, :3].copy()
        layout = detect_layout(xyz_ref, z_tol, xy_tol)
        if not layout["shared_levels"]:
            raise ValueError(
                f"found {layout['n_levels']:,} distinct Z values for {xyz_ref.shape[0]:,} nodes -- "
                "the mesh has no shared horizontal levels, so plane averaging by level is not "
                "possible (try a larger --z-tol if levels differ by round-off).")
        if layout["n_levels"] < 3:
            raise ValueError("need at least 3 Z levels")
        have_cols = layout["col_idx"] is not None
        if mode == "columns" and not have_cols:
            raise ValueError("--mode columns requested but the mesh is not a column structure "
                             "(levels with differing point counts or (x,y) positions).")
        mode_used = ("columns" if have_cols else "fe") if mode == "auto" else mode
        if t_ref is None:
            Tl = np.bincount(layout["level_idx"], weights=first_data[:, 4].astype(np.float64),
                             minlength=layout["n_levels"]) / layout["counts"]
            t_ref_used = 0.5 * (Tl[0] + Tl[-1])
        else:
            t_ref_used = float(t_ref)
        if verbose:
            print(f"  {xyz_ref.shape[0]:,} points, {layout['n_levels']} Z levels, "
                  f"mode={mode_used}, T_ref={t_ref_used:.8g}")

    fe = None
    if mode_used == "fe":
        conn = reference_connectivity(by_iter, T_VAR_IDX, ckpt_dir, xyz_ref, coord_tol=coord_tol,
                                      coord_check_tol=coord_check_tol, node_dtype=node_dtype,
                                      verbose=verbose)
        fe = build_fe_operator(xyz_ref, conn.astype(np.int32 if conn.max() < 2**31 else np.int64))
        del conn
    ctx = _make_ctx(xyz_ref, layout, mode_used, t_ref_used, fe)

    if first_data is not None:
        conv, cond, Tm = level_stats(first_data, ctx)
        _save_iter(_iter_path(ckpt_dir, items[0][0]), conv, cond, Tm)
        np.save(ref_path, xyz_ref)
        with open(meta_path, "w") as f:
            json.dump({"kind": "nusselt", "dedup_v2": True, "mode": mode_used,
                       "t_ref": t_ref_used, "n_points": int(xyz_ref.shape[0]),
                       "coord_tol": coord_tol}, f)
        del first_data
        gc.collect()

    pending = [(it, b) for it, b in items if not os.path.isfile(_iter_path(ckpt_dir, it))]
    if pending:
        n_workers = choose_worker_count(
            dict(pending), n_cols=3 + len(COLS), include_connectivity=False,
            requested_workers=workers, max_ram_gb=max_ram_gb,
            # the sizer assumes 4-byte values; scale for float64
            safety_factor=safety_factor * np.dtype(node_dtype).itemsize / 4.0,
            ram_margin=ram_margin)
        if verbose:
            print(f"Processing {len(pending)} pending iteration(s) with {n_workers} worker(s)...")
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker,
                                 initargs=(ctx,)) as executor:
            futures = {executor.submit(_process_iteration, it, blocks, coord_tol, node_dtype,
                                       coord_check_tol, _iter_path(ckpt_dir, it)): it
                       for it, blocks in pending}
            n_done = 0
            for future in as_completed(futures):
                result = future.result()
                if result["status"] == "error":
                    for f in futures:
                        f.cancel()
                    raise ValueError(f"iter {result['iter']}: {result['error']} -- aborted "
                                     f"(finished iterations are kept in {ckpt_dir}; rerun to resume).")
                n_done += 1
                if verbose:
                    print(f"[iter {result['iter']}] saved ({n_done}/{len(pending)})", flush=True)
    elif verbose:
        print("Nothing pending -- all iterations already checkpointed.")

    L = layout["n_levels"]
    s_conv, s_cond, s_Tm = np.zeros(L), np.zeros(L), np.zeros(L)
    for it, _ in items:
        with np.load(_iter_path(ckpt_dir, it)) as f:
            if f["conv"].shape != (L,):
                raise ValueError(f"{_iter_path(ckpt_dir, it)} has the wrong size; use --no-resume.")
            s_conv += f["conv"]
            s_cond += f["cond"]
            s_Tm += f["Tm"]
    n = len(items)
    return {"z_levels": layout["z_levels"], "conv": s_conv / n, "cond": s_cond / n, "Tm": s_Tm / n,
            "n_accum": n, "mode": mode_used, "t_ref": t_ref_used, "n_points": int(xyz_ref.shape[0])}


# --------------------------------------------------------------------------
# From level means to Nusselt numbers
# --------------------------------------------------------------------------

def _trapz(y, x):
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def nusselt_from_levels(z, conv, cond, Tm, pe, k_ref=1.0, delta_T=None, H=None,
                        bulk_range=(0.0, 1.0)):
    """
    Pure function of the time-averaged level means. Heat flows hot -> cold:
    the sign is chosen from which wall is hotter, so the usual
    hot-bottom case and a hot-top case both give positive Nu.

      Nu(z)      = sgn * (Pe*conv - cond) / (k_ref*DeltaT/H)
      Nu_hot     = -sgn * cond at the hot wall level  / (k_ref*DeltaT/H)
      Nu_cold    = -sgn * cond at the cold wall level / (k_ref*DeltaT/H)
      Nu_bulk    = (1/(z_hi-z_lo)) * integral of Nu(z) dz over bulk_range
                   (fractions of the height; trapezoid on the mesh levels)

    The wall values use the conductive term only (no-slip: w = 0 there).
    DeltaT defaults to |T_wall,bottom - T_wall,top| from the time-mean
    plane averages; H to the Z extent.
    """
    sgn = 1.0 if Tm[0] > Tm[-1] else -1.0
    hot, cold = (0, -1) if sgn > 0 else (-1, 0)
    H = float(H) if H is not None else float(z[-1] - z[0])
    dT = float(delta_T) if delta_T is not None else float(abs(Tm[0] - Tm[-1]))
    den = k_ref * dT / H

    Nu_z = sgn * (pe * conv - cond) / den
    Nu_conv_z = sgn * pe * conv / den
    Nu_cond_z = -sgn * cond / den

    lo = z[0] + bulk_range[0] * (z[-1] - z[0])
    hi = z[0] + bulk_range[1] * (z[-1] - z[0])
    m = (z >= lo - 1e-12) & (z <= hi + 1e-12)
    if m.sum() < 2:
        raise ValueError(f"bulk range {bulk_range} contains fewer than 2 Z levels")
    zb = z[m]
    Nu_bulk = _trapz(Nu_z[m], zb) / (zb[-1] - zb[0])

    return {
        "Nu_hot": -sgn * cond[hot] / den, "Nu_cold": -sgn * cond[cold] / den,
        "Nu_bulk": Nu_bulk, "Nu_z": Nu_z, "Nu_conv_z": Nu_conv_z, "Nu_cond_z": Nu_cond_z,
        "hot_at": "bottom" if sgn > 0 else "top", "delta_T": dT, "H": H, "k_ref": k_ref,
        "pe": pe, "bulk_z_range": (float(zb[0]), float(zb[-1])),
        "rel_spread": float((Nu_z.max() - Nu_z.min()) / abs(Nu_z.mean())),
    }
