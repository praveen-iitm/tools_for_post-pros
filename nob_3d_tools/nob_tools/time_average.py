"""
Time-averaging a single field across iterations, and reducing the result
to a horizontal (x,y) profile as a function of Z.

Design notes on RAM/parallelism:
  - The first iteration is read in the main process to establish the
    reference grid (X,Y,Z) -- everything downstream depends on every
    later iteration matching this exactly, so it has to exist before any
    worker starts.
  - Remaining iterations are read+merged+deduplicated in parallel worker
    processes (this is the expensive part -- same cost as extraction).
    Each worker only returns its averaged field VALUES (one float64 array
    of length N), never X,Y,Z or connectivity -- keeping inter-process
    communication small regardless of how many iterations you average.
  - The reference grid is sent to each worker exactly ONCE, via
    ProcessPoolExecutor's `initializer`, not re-pickled on every task.
    Workers use it locally to verify their own iteration's grid matches
    before contributing to the sum -- if it doesn't, that's a correctness
    problem worth stopping for, not a warning to ignore.
"""

import gc
import json
import os
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from .assemble import assemble_iteration_variable
from .memory_sizing import choose_worker_count


# --------------------------------------------------------------------------
# Worker-side state and entry point
# --------------------------------------------------------------------------

_REF_XYZ = None  # set once per worker process via the pool initializer


def _init_worker(ref_xyz):
    global _REF_XYZ
    _REF_XYZ = ref_xyz


def _process_iteration_for_average(it, blocks, var_idx, coord_tol, node_dtype, coord_check_tol,
                                   out_path=None):
    """
    Runs in a worker process. Reads+merges one iteration's blocks, checks
    the result against the reference grid (_REF_XYZ, set by the pool
    initializer), and returns only the field values -- never X,Y,Z.

    If out_path is given, the field is written there (atomically) instead
    of being returned, so nothing large crosses back to the main process
    and the result survives a crash.
    """
    global _REF_XYZ
    try:
        data, _ = assemble_iteration_variable(
            blocks, var_idx, include_connectivity=False,
            coord_tol=coord_tol, node_dtype=node_dtype, conn_dtype=np.int32,
            log_prefix=f"[iter {it}] ",
        )
        coords = data[:, :3]
        field = data[:, 3].astype(np.float64)
        del data

        if coords.shape[0] != _REF_XYZ.shape[0]:
            return {"iter": it, "status": "error",
                    "error": f"point count {coords.shape[0]:,} != reference {_REF_XYZ.shape[0]:,}. "
                             "The mesh may not be static across iterations, or dedup produced "
                             "a different result this time."}

        max_diff = float(np.abs(coords - _REF_XYZ).max())
        if max_diff > coord_check_tol:
            return {"iter": it, "status": "error",
                    "error": f"coordinates differ from the reference grid by up to {max_diff:.3g} "
                             f"(tolerance {coord_check_tol:.3g}). Node ordering may not be "
                             "consistent across iterations."}

        del coords
        gc.collect()
        if out_path is not None:
            _save_atomic(out_path, field.astype(node_dtype, copy=False))
            return {"iter": it, "status": "ok", "n_points": int(field.shape[0])}
        return {"iter": it, "status": "ok", "field": field, "n_points": int(field.shape[0])}
    except Exception as e:
        return {"iter": it, "status": "error", "error": str(e)}



def _save_atomic(path, arr):
    """np.save to a temp file then rename, so a killed job never leaves a
    half-written .npy that a restart would mistake for a finished one."""
    tmp = path[:-4] + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def _iter_path(ckpt_dir, it):
    return os.path.join(ckpt_dir, f"iter_{it}.npy")


# --------------------------------------------------------------------------
# Main-process entry points
# --------------------------------------------------------------------------

def compute_reference_grid(blocks, var_idx, coord_tol=1e-6, node_dtype=np.float32, log_prefix=""):
    """
    Read+merge ONE iteration's blocks and return (xyz, field) -- used to
    establish the reference grid before parallelizing the rest.
    """
    data, _ = assemble_iteration_variable(
        blocks, var_idx, include_connectivity=False,
        coord_tol=coord_tol, node_dtype=node_dtype, conn_dtype=np.int32,
        log_prefix=log_prefix,
    )
    xyz = data[:, :3].copy()
    field = data[:, 3].astype(np.float64)
    del data
    gc.collect()
    return xyz, field


def time_average_field(by_iter, var_idx, coord_tol=1e-6, coord_check_tol=1e-5,
                        node_dtype=np.float32, workers=None, max_ram_gb=None,
                        safety_factor=3.0, ram_margin=0.8, verbose=True):
    """
    Time-average one field across every iteration in by_iter (as returned
    by discover_iterations), reading+merging each iteration's blocks in
    parallel where possible.

    Returns: (xyz_ref, mean_field, n_accum, n_total)
        xyz_ref    : (N, 3) reference grid coordinates
        mean_field : (N,) float32 time-averaged field values
        n_accum    : number of iterations actually accumulated
        n_total    : number of iterations found in by_iter

    Raises ValueError immediately if any iteration's grid doesn't match
    the reference (point count or coordinates) -- this is a correctness
    check, not a warning, since a silent mismatch would corrupt the
    average with no visible symptom.
    """
    items = list(by_iter.items())
    if not items:
        raise ValueError("by_iter is empty -- nothing to average")

    first_it, first_blocks = items[0]
    remaining = items[1:]

    if verbose:
        print(f"Computing reference grid from iter {first_it} ({len(first_blocks)} block(s))...")
    t0 = time.perf_counter()
    xyz_ref, field0 = compute_reference_grid(first_blocks, var_idx, coord_tol, node_dtype,
                                              log_prefix=f"[iter {first_it}] ")
    sum_field = field0.copy()
    n_accum = 1
    if verbose:
        print(f"  reference grid: {xyz_ref.shape[0]:,} points ({time.perf_counter() - t0:.1f}s)")

    if not remaining:
        return xyz_ref, (sum_field / n_accum).astype(np.float32), n_accum, len(items)

    n_workers = choose_worker_count(
        dict(remaining), n_cols=4, include_connectivity=False,
        requested_workers=workers, max_ram_gb=max_ram_gb,
        safety_factor=safety_factor, ram_margin=ram_margin,
    )
    if verbose:
        print(f"Averaging remaining {len(remaining)} iteration(s) with {n_workers} worker(s)...")

    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker,
                              initargs=(xyz_ref,)) as executor:
        futures = {
            executor.submit(_process_iteration_for_average, it, blocks, var_idx,
                             coord_tol, node_dtype, coord_check_tol): it
            for it, blocks in remaining
        }
        n_done = 0
        for future in as_completed(futures):
            result = future.result()
            it = result["iter"]
            if result["status"] == "error":
                # Stop pending work rather than waiting on iterations we're
                # about to discard anyway (already-running tasks still
                # finish -- cancel() only affects ones not yet started).
                for f in futures:
                    f.cancel()
                raise ValueError(f"iter {it}: {result['error']} -- averaging aborted.")

            sum_field += result["field"]
            n_accum += 1
            n_done += 1
            if verbose:
                print(f"[iter {it}] accumulated ({n_done}/{len(remaining)})")
            del result

    mean_field = (sum_field / n_accum).astype(np.float32)
    return xyz_ref, mean_field, n_accum, len(items)


def time_average_field_checkpointed(by_iter, var_idx, ckpt_dir, resume=True,
                                    coord_tol=1e-6, coord_check_tol=1e-5,
                                    node_dtype=np.float32, workers=None, max_ram_gb=None,
                                    safety_factor=3.0, ram_margin=0.8, verbose=True):
    """
    Same result as time_average_field, but restartable and RAM-light.

    Each iteration's assembled field is written to ckpt_dir/iter_<it>.npy
    as soon as a worker finishes it (atomic write); nothing is accumulated
    in RAM while the job runs. The reference grid is saved once as
    ckpt_dir/ref_xyz.npy. If the job dies, rerun with the same ckpt_dir:
    iterations whose file already exists are skipped, so only the
    unfinished ones are redone. The final mean is a streaming sum over the
    per-iteration files (two N-sized arrays in RAM at a time).

    Because finished files are kept, you can also rerun later with MORE
    iterations and only the new ones are computed. Only the iterations in
    by_iter enter the average, whatever else is in ckpt_dir.

    resume=False wipes any existing checkpoint files first.

    Returns: (xyz_ref, mean_field, n_accum, n_total) like time_average_field.
    """
    items = list(by_iter.items())
    if not items:
        raise ValueError("by_iter is empty -- nothing to average")
    os.makedirs(ckpt_dir, exist_ok=True)
    ref_path = os.path.join(ckpt_dir, "ref_xyz.npy")
    meta_path = os.path.join(ckpt_dir, "meta.json")

    def _wipe():
        for fn in os.listdir(ckpt_dir):
            if fn.startswith("iter_") or fn in ("ref_xyz.npy", "meta.json"):
                os.remove(os.path.join(ckpt_dir, fn))

    if not resume:
        _wipe()

    have_ref = os.path.isfile(ref_path) and os.path.isfile(meta_path)
    if not have_ref:
        # Iteration files without a verified reference grid can't be trusted.
        _wipe()
    else:
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("var_idx") != var_idx:
            raise ValueError(f"{ckpt_dir} holds checkpoints for variable index {meta.get('var_idx')}, "
                             f"not {var_idx}. Use a different --checkpoint-dir or --no-resume.")
        if meta.get("coord_tol") != coord_tol and verbose:
            print(f"WARNING: checkpoint was made with coord_tol={meta.get('coord_tol')}, "
                  f"now {coord_tol}.")

    def _done(it):
        return os.path.isfile(_iter_path(ckpt_dir, it))

    if have_ref:
        xyz_ref = np.load(ref_path)
        if verbose:
            print(f"Resuming from {ckpt_dir}: reference grid {xyz_ref.shape[0]:,} points, "
                  f"{sum(_done(it) for it, _ in items)}/{len(items)} iteration(s) already done")
        pending = [(it, b) for it, b in items if not _done(it)]
    else:
        first_it, first_blocks = items[0]
        if verbose:
            print(f"Computing reference grid from iter {first_it} ({len(first_blocks)} block(s))...")
        t0 = time.perf_counter()
        xyz_ref, field0 = compute_reference_grid(first_blocks, var_idx, coord_tol, node_dtype,
                                                  log_prefix=f"[iter {first_it}] ")
        _save_atomic(ref_path, xyz_ref)
        _save_atomic(_iter_path(ckpt_dir, first_it), field0.astype(node_dtype, copy=False))
        with open(meta_path, "w") as f:
            json.dump({"var_idx": var_idx, "n_points": int(xyz_ref.shape[0]),
                       "coord_tol": coord_tol}, f)
        del field0
        gc.collect()
        if verbose:
            print(f"  reference grid: {xyz_ref.shape[0]:,} points ({time.perf_counter() - t0:.1f}s)")
        pending = items[1:]

    if pending:
        n_workers = choose_worker_count(
            dict(pending), n_cols=4, include_connectivity=False,
            requested_workers=workers, max_ram_gb=max_ram_gb,
            safety_factor=safety_factor, ram_margin=ram_margin,
        )
        if verbose:
            print(f"Processing {len(pending)} pending iteration(s) with {n_workers} worker(s)...")
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker,
                                  initargs=(xyz_ref,)) as executor:
            futures = {
                executor.submit(_process_iteration_for_average, it, blocks, var_idx,
                                 coord_tol, node_dtype, coord_check_tol,
                                 _iter_path(ckpt_dir, it)): it
                for it, blocks in pending
            }
            n_done = 0
            for future in as_completed(futures):
                result = future.result()
                if result["status"] == "error":
                    for f in futures:
                        f.cancel()
                    raise ValueError(f"iter {result['iter']}: {result['error']} -- averaging aborted "
                                     f"(finished iterations are kept in {ckpt_dir}; rerun to resume).")
                n_done += 1
                if verbose:
                    print(f"[iter {result['iter']}] saved ({n_done}/{len(pending)})", flush=True)
    elif verbose:
        print("Nothing pending -- all iterations already checkpointed.")

    # Streaming reduction: one iteration file in RAM at a time.
    n_pts = xyz_ref.shape[0]
    sum_field = np.zeros(n_pts, dtype=np.float64)
    for it, _ in items:
        arr = np.load(_iter_path(ckpt_dir, it))
        if arr.shape != (n_pts,):
            raise ValueError(f"{_iter_path(ckpt_dir, it)} has shape {arr.shape}, expected ({n_pts},). "
                             "Delete it (or use --no-resume) and rerun.")
        sum_field += arr
        del arr
    n_accum = len(items)
    return xyz_ref, (sum_field / n_accum).astype(np.float32), n_accum, n_accum


def horizontal_profile(xyz_ref, field_values, z_tol=1e-8):
    """
    Group points by Z (matched within z_tol) and average field_values
    within each group -- the horizontal (x,y) average as a function of Z.

    This is a plain arithmetic mean over points at each level, not an
    area-weighted spatial average -- see the caveat in the notebook/README
    if your horizontal mesh has non-uniform point density.

    Returns: (z_levels, profile, counts) -- all sorted ascending by Z.
    """
    z_keys = np.round(xyz_ref[:, 2] / z_tol).astype(np.int64)
    unique_keys, inverse, counts = np.unique(z_keys, return_inverse=True, return_counts=True)

    z_levels = unique_keys.astype(np.float64) * z_tol
    sum_per_z = np.bincount(inverse, weights=field_values.astype(np.float64))
    profile = sum_per_z / counts

    return z_levels, profile, counts


def write_profile_dat(filepath, z_levels, profile, names=("Z", "value")):
    """Write a 2-column ASCII .dat file: Z and the averaged value."""
    data = np.column_stack([z_levels, profile])
    np.savetxt(filepath, data, fmt="%.12g")
