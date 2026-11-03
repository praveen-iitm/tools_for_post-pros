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
# extra columns, only read when the dissipation-based Nu_V is requested
# (data columns 8, 9, 10): U, V and the dynamic viscosity
COLS_DISS = [ALL_VAR_NAMES.index(n) for n in ("U", "V", "mu")]
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


def build_fe_operator(xyz, conn, chunk=2_000_000, full=False):
    """
    Precompute what is needed to get a node-based dT/dz from a node field T
    on a tetrahedral mesh: per-element z-gradient coefficients `a` (E,3)
    such that dT/dz|_e = a . (T1-T0, T2-T0, T3-T0), element volumes, and
    each node's summed adjacent volume. conn is 0-based (E,4). With
    full=True the whole inverse edge matrix Dinv (E,3,3) is kept too, which
    gives all three gradient components (needed for the dissipation).
    """
    N, E = xyz.shape[0], conn.shape[0]
    x = xyz.astype(np.float64, copy=False)
    a = np.empty((E, 3), dtype=np.float64)
    vol = np.empty(E, dtype=np.float64)
    Dinv_all = np.empty((E, 3, 3), dtype=np.float64) if full else None
    for s in range(0, E, chunk):
        c = conn[s:s + chunk]
        p0 = x[c[:, 0]]
        D = np.stack([x[c[:, 1]] - p0, x[c[:, 2]] - p0, x[c[:, 3]] - p0], axis=1)   # rows = edges
        det = np.linalg.det(D)
        if (det == 0).any():
            raise ValueError("degenerate (zero-volume) tetrahedron in connectivity")
        Dinv = np.linalg.inv(D)
        a[s:s + chunk] = Dinv[:, 2, :]                      # g_z = Dinv[2,:] . dT
        if full:
            Dinv_all[s:s + chunk] = Dinv
        vol[s:s + chunk] = np.abs(det) / 6.0
    node_vol = np.zeros(N, dtype=np.float64)
    for j in range(4):
        node_vol += np.bincount(conn[:, j], weights=vol, minlength=N)
    return {"conn": conn, "a": a, "vol": vol, "node_vol": node_vol, "n_nodes": N, "Dinv": Dinv_all}


def viscous_dissipation(U, V, W, mu, op, chunk=1_000_000):
    """
    Volume average of the viscous dissipation per unit volume

        Phi = mu * [ d_i u_j (d_i u_j + d_j u_i)  -  (2/3) (div u)^2 ]

    with piecewise-constant per-tetrahedron gradients (exact for linear
    fields) and mu taken as the mean of the element's 4 nodal values;
    weights are the element volumes. For constant mu and div u = 0 this is
    mu * <(d_i u_j)^2>, since the d_i u_j d_j u_i part integrates to zero
    with no-slip/periodic boundaries. Needs an operator built with full=True.
    """
    if op.get("Dinv") is None:
        raise ValueError("operator was built without the full gradient (full=True)")
    c, Dinv, vol = op["conn"], op["Dinv"], op["vol"]
    tot = 0.0
    for s0 in range(0, c.shape[0], chunk):
        cc, Dm = c[s0:s0 + chunk], Dinv[s0:s0 + chunk]

        def diffs(F):
            f0 = F[cc[:, 0]]
            return np.stack([F[cc[:, 1]] - f0, F[cc[:, 2]] - f0, F[cc[:, 3]] - f0], axis=1)

        G = np.stack([np.einsum("eik,ek->ei", Dm, diffs(F)) for F in (U, V, W)], axis=2)  # G[e,i,j]=d_i u_j
        s2 = np.einsum("eij,eij->e", G, G) + np.einsum("eij,eji->e", G, G)
        div = G[:, 0, 0] + G[:, 1, 1] + G[:, 2, 2]
        mu_e = 0.25 * (mu[cc[:, 0]] + mu[cc[:, 1]] + mu[cc[:, 2]] + mu[cc[:, 3]])
        tot += float(np.sum(vol[s0:s0 + chunk] * mu_e * (s2 - (2.0 / 3.0) * div ** 2)))
    return tot / float(vol.sum())


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
    Returns (conv, cond, Tm, mflux, diss): the first four are (L,) level
    means of rho*cp*w*(T-T_ref), kappa*dT/dz, T, and rho*cp*w (the mass-flux
    diagnostic: ~0 in a stationary closed cell); diss is the volume-averaged
    viscous dissipation (a float) when ctx["diss"] is on, else None.
    """
    w, T, rho, cp, kap = (data[:, 3 + i].astype(np.float64) for i in range(5))
    mf_n = rho * cp * w
    conv_n = mf_n * (T - ctx["t_ref"])
    if ctx["mode"] == "columns":
        idx = ctx["col_idx"]
        Tg = T[idx]
        dT = fd_z(Tg, ctx["z_levels"])
        cond = (kap[idx] * dT).mean(axis=1)
        conv = conv_n[idx].mean(axis=1)
        Tm = Tg.mean(axis=1)
        mflux = mf_n[idx].mean(axis=1)
    else:
        g = grad_z_fe(T, ctx["fe"])
        lv, cnt, L = ctx["level_idx"], ctx["counts"], len(ctx["counts"])
        cond = np.bincount(lv, weights=kap * g, minlength=L) / cnt
        conv = np.bincount(lv, weights=conv_n, minlength=L) / cnt
        Tm = np.bincount(lv, weights=T, minlength=L) / cnt
        mflux = np.bincount(lv, weights=mf_n, minlength=L) / cnt
    diss = None
    if ctx.get("diss"):
        U, V, mu = (data[:, 8 + i].astype(np.float64) for i in range(3))
        diss = viscous_dissipation(U, V, w, mu, ctx["fe"])
    return conv, cond, Tm, mflux, diss


_CTX = None  # set once per worker process via the pool initializer


def _init_worker(ctx):
    global _CTX
    _CTX = ctx


def _iter_path(ckpt_dir, it):
    return os.path.join(ckpt_dir, f"iter_{it}.npz")


def _save_iter(path, conv, cond, Tm, mflux, diss=None):
    tmp = path[:-4] + ".tmp.npz"
    extra = {} if diss is None else {"diss": np.float64(diss)}
    np.savez(tmp, conv=conv, cond=cond, Tm=Tm, mflux=mflux, **extra)
    os.replace(tmp, path)


def _process_iteration(it, blocks, coord_tol, node_dtype, coord_check_tol, out_path):
    """Worker: read+merge one iteration, check its grid, reduce, save."""
    try:
        data, _ = assemble_iteration_columns(
            blocks, _CTX["cols"], False, coord_tol, node_dtype, np.int32, log_prefix=f"[iter {it}] ")
        ref = _CTX["xyz_ref"]
        if data.shape[0] != ref.shape[0]:
            return {"iter": it, "status": "error",
                    "error": f"point count {data.shape[0]:,} != reference {ref.shape[0]:,}"}
        max_diff = float(np.abs(data[:, :3] - ref).max())
        if max_diff > coord_check_tol:
            return {"iter": it, "status": "error",
                    "error": f"coordinates differ from the reference grid by up to {max_diff:.3g} "
                             f"(tolerance {coord_check_tol:.3g})"}
        conv, cond, Tm, mflux, diss = level_stats(data, _CTX)
        del data
        gc.collect()
        _save_iter(out_path, conv, cond, Tm, mflux, diss)
        return {"iter": it, "status": "ok"}
    except Exception as e:
        return {"iter": it, "status": "error", "error": str(e)}


def _make_ctx(xyz_ref, layout, mode, t_ref, fe=None, cols=None, diss=False):
    ctx = {"xyz_ref": xyz_ref, "mode": mode, "t_ref": float(t_ref), "cols": cols or COLS, "diss": diss,
           "z_levels": layout["z_levels"], "level_idx": layout["level_idx"],
           "counts": layout["counts"], "col_idx": layout["col_idx"], "fe": fe}
    return ctx


def compute_level_means(by_iter, ckpt_dir, resume=True, mode="auto", t_ref=None, dissipation=False,
                        coord_tol=1e-6, coord_check_tol=1e-5, z_tol=1e-8, xy_tol=1e-6,
                        node_dtype=np.float64, workers=None, max_ram_gb=None,
                        safety_factor=4.0, ram_margin=0.8, verbose=True):
    """
    Time-average, per Z level, the convective term rho*cp*w*(T-T_ref), the
    conductive term kappa*dT/dz and T, restartably (see module docstring).

    Returns dict: z_levels, conv, cond, Tm (all (L,)), n_accum, mode,
    t_ref, n_points, the per-iteration arrays and, if dissipation=True,
    diss / diss_it (volume-averaged viscous dissipation; this needs the
    connectivity and also reads U, V and mu).
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
        elif dissipation and not meta.get("dissipation"):
            raise ValueError(f"checkpoint in {ckpt_dir} was made without the dissipation columns; "
                             "use --no-resume (or a different --checkpoint-dir) to add --nu-v.")
        elif t_ref is not None and abs(meta["t_ref"] - t_ref) > 1e-12:
            raise ValueError(f"checkpoint in {ckpt_dir} used t_ref={meta['t_ref']}, not {t_ref}. "
                             "Use --no-resume to start over.")
    if not have_ref:
        _wipe()

    cols_used = COLS + (COLS_DISS if dissipation else [])
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
            first_blocks, cols_used, False, coord_tol, node_dtype, np.int32,
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
    if mode_used == "fe" or dissipation:
        conn = reference_connectivity(by_iter, T_VAR_IDX, ckpt_dir, xyz_ref, coord_tol=coord_tol,
                                      coord_check_tol=coord_check_tol, node_dtype=node_dtype,
                                      verbose=verbose)
        fe = build_fe_operator(xyz_ref, conn.astype(np.int32 if conn.max() < 2**31 else np.int64),
                               full=dissipation)
        del conn
    ctx = _make_ctx(xyz_ref, layout, mode_used, t_ref_used, fe, cols=cols_used, diss=dissipation)

    if first_data is not None:
        conv, cond, Tm, mflux, diss = level_stats(first_data, ctx)
        _save_iter(_iter_path(ckpt_dir, items[0][0]), conv, cond, Tm, mflux, diss)
        np.save(ref_path, xyz_ref)
        with open(meta_path, "w") as f:
            json.dump({"kind": "nusselt", "dedup_v2": True, "mode": mode_used,
                       "dissipation": bool(dissipation),
                       "t_ref": t_ref_used, "n_points": int(xyz_ref.shape[0]),
                       "coord_tol": coord_tol}, f)
        del first_data
        gc.collect()

    pending = [(it, b) for it, b in items if not os.path.isfile(_iter_path(ckpt_dir, it))]
    if pending:
        n_workers = choose_worker_count(
            dict(pending), n_cols=3 + len(cols_used), include_connectivity=False,
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
    conv_it, cond_it, Tm_it, mf_it, diss_it = [], [], [], [], []
    for it, _ in items:
        with np.load(_iter_path(ckpt_dir, it)) as f:
            if f["conv"].shape != (L,):
                raise ValueError(f"{_iter_path(ckpt_dir, it)} has the wrong size; use --no-resume.")
            conv_it.append(f["conv"]); cond_it.append(f["cond"]); Tm_it.append(f["Tm"])
            mf_it.append(f["mflux"] if "mflux" in f.files else None)
            if dissipation:
                if "diss" not in f.files:
                    raise ValueError(f"{_iter_path(ckpt_dir, it)} has no dissipation value; "
                                     "use --no-resume (or a different --checkpoint-dir) to add --nu-v.")
                diss_it.append(float(f["diss"]))
    conv_it, cond_it, Tm_it = np.array(conv_it), np.array(cond_it), np.array(Tm_it)
    # mass-flux diagnostic only if every iteration has it (older checkpoints don't)
    mf_it = np.array(mf_it) if all(m is not None for m in mf_it) else None
    return {"z_levels": layout["z_levels"], "conv": conv_it.mean(0), "cond": cond_it.mean(0),
            "Tm": Tm_it.mean(0), "mflux": None if mf_it is None else mf_it.mean(0),
            "iters": [it for it, _ in items], "conv_it": conv_it, "cond_it": cond_it,
            "Tm_it": Tm_it, "n_accum": len(items), "mode": mode_used, "t_ref": t_ref_used,
            "n_points": int(xyz_ref.shape[0]), "xy_spacing": xy_spacing_info(xyz_ref, layout),
            "diss_it": np.array(diss_it) if dissipation else None,
            "diss": float(np.mean(diss_it)) if dissipation else None}


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
        "pe": pe, "bulk_z_range": (float(zb[0]), float(zb[-1])), "bulk_range": tuple(bulk_range),
        "rel_spread": float((Nu_z.max() - Nu_z.min()) / abs(Nu_z.mean())),
    }


def xy_spacing_info(xyz, layout):
    """
    For a column structure on a tensor-product (x,y) grid: max/min spacing
    ratio along x and along y. 1.0 = uniform. The plain arithmetic mean over
    nodes used for every horizontal average is only an area average if the
    horizontal spacing is uniform. Returns None when not applicable.
    """
    if layout["col_idx"] is None:
        return None
    lvl0 = xyz[layout["col_idx"][0]]
    out = {}
    for name, c in (("dx", lvl0[:, 0]), ("dy", lvl0[:, 1])):
        u = np.unique(np.round(c, 10))
        out["n_" + name[1]] = len(u)
        out[name + "_ratio"] = float(np.diff(u).max() / np.diff(u).min()) if len(u) > 2 else 1.0
    if out["n_x"] * out["n_y"] != lvl0.shape[0]:
        return None          # not a tensor-product grid
    return out


def pe_self_check(Nu_conv_z, Nu_cond_z, z):
    """
    Energy conservation says Q = Pe*conv - cond is the same on every plane,
    i.e. Nu_cond(z) = Q - f*Nu_conv(z) with f = 1 if the Pe used is right.
    Fit f and Q by least squares (levels weighted by their thickness).
    Returns (f, Q_fit, rms_resid/|Q_fit|). f far from 1 => the velocity
    scaling (Pe) is off; f ~ 1 with a large residual => the problem is
    elsewhere (non-stationary data, averaging, formulation).
    """
    dz = np.gradient(z)
    sw = np.sqrt(dz)
    A = np.column_stack([np.ones_like(z), -Nu_conv_z]) * sw[:, None]
    coef, *_ = np.linalg.lstsq(A, Nu_cond_z * sw, rcond=None)
    Q, f = float(coef[0]), float(coef[1])
    resid = Nu_cond_z - (Q - f * Nu_conv_z)
    rms = float(np.sqrt(np.sum(dz * resid ** 2) / np.sum(dz)))
    return f, Q, rms / abs(Q)


def nusselt_series(lev, pe, ref):
    """
    Per-iteration Nu_hot / Nu_cold / Nu_bulk (one row per snapshot), using
    the same normalisation as the time-mean result `ref`
    (nusselt_from_levels output) so the numbers are comparable.
    """
    rows = []
    for i in range(len(lev["iters"])):
        r = nusselt_from_levels(lev["z_levels"], lev["conv_it"][i], lev["cond_it"][i],
                                lev["Tm_it"][i], pe, k_ref=ref["k_ref"], delta_T=ref["delta_T"],
                                H=ref["H"], bulk_range=ref["bulk_range"])
        rows.append((lev["iters"][i], r["Nu_hot"], r["Nu_cold"], r["Nu_bulk"]))
    return np.array(rows)


def thermal_bl(z, Tm, count_wall=True):
    """
    Thermal boundary-layer thickness by the slope method on the time- and
    plane-averaged temperature profile Tm(z): the distance from each plate
    at which the tangent of Tm at the plate reaches the centre temperature
    T_c (Tm at mid-height, interpolated),

        delta_b = |Tm[0]  - T_c| / |dTm/dz at the bottom|
        delta_t = |Tm[-1] - T_c| / |dTm/dz at the top|

    (wall slopes: second-order one-sided differences of the level means),
    and the number of grid points (Z levels) inside each layer, i.e. at a
    distance <= delta from the plate. count_wall=True includes the plate
    level itself; both counts are returned. 'b'/'t' are the bottom/top
    plates, as in the paper's N^b_BL and N^t_BL.
    """
    d = fd_z(Tm[:, None], z)[:, 0]
    Tc = float(np.interp(0.5 * (z[0] + z[-1]), z, Tm))
    delta_b = abs(Tm[0] - Tc) / abs(d[0])
    delta_t = abs(Tm[-1] - Tc) / abs(d[-1])
    n_b = int(np.sum(z - z[0] <= delta_b * (1 + 1e-12)))      # includes the plate level
    n_t = int(np.sum(z[-1] - z <= delta_t * (1 + 1e-12)))
    return {"Tc": Tc, "delta_b": float(delta_b), "delta_t": float(delta_t),
            "N_b": n_b if count_wall else n_b - 1, "N_t": n_t if count_wall else n_t - 1,
            "N_b_with_wall": n_b, "N_t_with_wall": n_t,
            "N_b_no_wall": n_b - 1, "N_t_no_wall": n_t - 1}


def nu_from_dissipation(diss_mean, pr):
    """
    Nu_V = 1 + sqrt(Ra Pr) <eps_u>, with eps_u = sqrt(Pr/Ra) <Phi> in
    free-fall units; Ra cancels, leaving Nu_V = 1 + Pr * <Phi>, where Phi is
    the viscous dissipation per unit volume with mu scaled by its reference
    value. The exact relation behind it holds for Oberbeck-Boussinesq flow
    only; for NOB it is just a dissipation-based estimate.
    """
    return 1.0 + pr * diss_mean
