"""
Assemble one iteration's blocks into a merged mesh, and the
ProcessPoolExecutor worker entry points that wrap that for batch/parallel
scripts. Workers write their own output and return only a small status
dict -- never pickle multi-GB arrays back to the parent process.
"""

import os
import gc
import time

from .io_tecplot import read_tecplot_partition, read_variable_partition
from .dedup import merge_two, merge_partitions
from .writers import write_npz, write_npy, write_dat
from .discovery import ALL_VAR_NAMES


# --------------------------------------------------------------------------
# Full (all-variable) assembly
# --------------------------------------------------------------------------

def assemble_iteration_full(block_files, coord_tol, node_dtype, conn_dtype, log_prefix=""):
    """Assemble one iteration's blocks, keeping all variables."""
    t_start = time.perf_counter()
    blk0, fp0 = block_files[0]
    print(f"{log_prefix}reading block {blk0}: {os.path.basename(fp0)}", flush=True)
    nodes_acc, conn_acc = read_tecplot_partition(fp0, node_dtype, conn_dtype, log_prefix)

    for blk, fp in block_files[1:]:
        print(f"{log_prefix}reading block {blk}: {os.path.basename(fp)}", flush=True)
        nodes_i, conn_i = read_tecplot_partition(fp, node_dtype, conn_dtype, log_prefix)

        nodes_acc, conn_acc = merge_partitions(nodes_acc, conn_acc, nodes_i, conn_i, coord_tol)
        del nodes_i, conn_i
        gc.collect()

    print(f"{log_prefix}assemble_iteration_full total: {time.perf_counter() - t_start:.1f}s", flush=True)
    return nodes_acc, conn_acc


def process_iteration_full(it, blocks, output_dir, fmt, coord_tol, node_dtype, conn_dtype):
    """ProcessPoolExecutor worker entry point for the full-variable assembler."""
    out_name = f"assembled_iter{it}.{fmt}"
    out_path = os.path.join(output_dir, out_name)
    log_prefix = f"[iter {it}] "

    if os.path.exists(out_path):
        return {"iter": it, "status": "skipped", "path": out_path}

    try:
        t0 = time.perf_counter()
        nodes, conn = assemble_iteration_full(blocks, coord_tol, node_dtype, conn_dtype, log_prefix)
        n_nodes, n_elems = nodes.shape[0], conn.shape[0]

        if fmt == "npz":
            write_npz(out_path, nodes, conn, ALL_VAR_NAMES)
        else:
            write_dat(out_path, nodes, conn, ALL_VAR_NAMES)

        del nodes, conn
        gc.collect()
        print(f"{log_prefix}TOTAL (read+merge+write): {time.perf_counter() - t0:.1f}s", flush=True)

        return {"iter": it, "status": "ok", "path": out_path, "n_nodes": n_nodes, "n_elems": n_elems}
    except Exception as e:
        return {"iter": it, "status": "error", "error": str(e)}


# --------------------------------------------------------------------------
# Single-variable assembly
# --------------------------------------------------------------------------

def assemble_iteration_variable(block_files, var_idx, include_connectivity, coord_tol,
                                 node_dtype, conn_dtype, log_prefix=""):
    """
    Assemble one iteration's blocks, keeping only X,Y,Z + one variable.

    include_connectivity=False (the default caller behavior for .npy point
    clouds -- see process_iteration_variable) skips reading/merging
    connectivity entirely: it's never touched, not just dropped afterward,
    so it also saves the read time and memory of parsing the connectivity
    block for every partition.
    """
    t_start = time.perf_counter()
    blk0, fp0 = block_files[0]
    print(f"{log_prefix}reading block {blk0}: {os.path.basename(fp0)}", flush=True)
    data_acc, conn_local = read_variable_partition(fp0, var_idx, include_connectivity,
                                                     node_dtype, conn_dtype, log_prefix)
    conn_acc = (conn_local.astype("int64") - 1) if include_connectivity else None
    del conn_local

    for blk, fp in block_files[1:]:
        print(f"{log_prefix}reading block {blk}: {os.path.basename(fp)}", flush=True)
        data_i, conn_i = read_variable_partition(fp, var_idx, include_connectivity,
                                                   node_dtype, conn_dtype, log_prefix)

        new_data_acc, new_conn_acc, n_dup = merge_two(
            data_acc, conn_acc, data_i, conn_i, coord_tol, include_connectivity, log_prefix
        )
        del data_acc, conn_acc, data_i, conn_i
        gc.collect()
        data_acc, conn_acc = new_data_acc, new_conn_acc

    print(f"{log_prefix}assemble_iteration_variable total: {time.perf_counter() - t_start:.1f}s", flush=True)
    return data_acc, conn_acc


def process_iteration_variable(it, blocks, output_dir, fmt, variable, var_idx,
                                include_connectivity, coord_tol, node_dtype, conn_dtype):
    """
    ProcessPoolExecutor worker entry point for the single-variable extractor.

    fmt="npy": point cloud only -- X,Y,Z + the chosen variable, no
        connectivity at all, regardless of include_connectivity (forced
        off below). This is the unstructured-solver-friendly output: the
        mesh connectivity isn't meaningful for point-cloud analysis (e.g.
        two-point correlations), so it's never even read for this path.
    fmt="dat": unchanged -- full Tecplot mesh, connectivity included
        exactly as before, controlled by include_connectivity as before.
    """
    is_point_cloud = (fmt == "npy")
    if is_point_cloud:
        include_connectivity = False  # point cloud never carries connectivity

    out_name = f"assembled_iter{it}_{variable}.{fmt}"
    out_path = os.path.join(output_dir, out_name)
    log_prefix = f"[iter {it}] "

    if os.path.exists(out_path):
        return {"iter": it, "status": "skipped", "path": out_path}

    try:
        t0 = time.perf_counter()
        data, conn = assemble_iteration_variable(blocks, var_idx, include_connectivity, coord_tol,
                                                   node_dtype, conn_dtype, log_prefix)
        n_points = data.shape[0]
        n_elems = conn.shape[0] if conn is not None else 0

        # data has 3 columns (X,Y,Z only) if the requested variable IS X/Y/Z
        # itself -- var_names must match that or write_npy's shape check fails.
        var_names = ["X", "Y", "Z"] if var_idx in (0, 1, 2) else ["X", "Y", "Z", variable]

        if fmt == "npy":
            write_npy(out_path, data, var_names)
        elif fmt == "npz":
            write_npz(out_path, data, conn, var_names)
        else:
            write_dat(out_path, data, conn, var_names)

        del data, conn
        gc.collect()
        print(f"{log_prefix}TOTAL (read+merge+write): {time.perf_counter() - t0:.1f}s", flush=True)

        return {"iter": it, "status": "ok", "path": out_path, "n_points": n_points, "n_elems": n_elems}
    except Exception as e:
        return {"iter": it, "status": "error", "error": str(e)}
