"""
Reading Tecplot FETETRAHEDRON / DATAPACKING=POINT partition files.

Two read paths:
  - read_tecplot_partition:   reads ALL variables (full field data)
  - read_variable_partition:  reads only X,Y,Z + one chosen variable,
                               via pandas' usecols so unrequested columns
                               are never parsed at all
"""

import re
import time
import gc
import numpy as np
import pandas as pd


def parse_tecplot_header(filename):
    """Read just the 3 header lines and return (n_vars, N, E)."""
    with open(filename, "r") as f:
        line1 = f.readline()  # VARIABLES = ...
        line2 = f.readline()  # ZONE N=..., E=...
        line3 = f.readline()  # DATAPACKING=..., ZONETYPE=...

    n_vars = line1.count('"') // 2
    m = re.search(r"N\s*=\s*(\d+)\s*,\s*E\s*=\s*(\d+)", line2)
    if not m:
        raise ValueError(f"Could not parse N/E from ZONE line in {filename}: {line2!r}")
    N, E = int(m.group(1)), int(m.group(2))

    if "FETETRAHEDRON" not in line3.upper():
        print(f"Warning ({filename}): expected FETETRAHEDRON, got: {line3.strip()}")

    return n_vars, N, E


def read_tecplot_partition(filename, node_dtype=np.float32, conn_dtype=np.int32, log_prefix=""):
    """
    Read ALL variables of one partition. Returns:
        nodes : (N, n_vars) array
        conn  : (E, 4) array, 1-based, partition-local
    """
    n_vars, N, E = parse_tecplot_header(filename)

    t0 = time.perf_counter()
    nodes = pd.read_csv(
        filename, sep=r"\s+", header=None,
        skiprows=3, nrows=N,
        dtype=node_dtype, engine="c", memory_map=True,
    ).to_numpy()
    t1 = time.perf_counter()
    print(f"{log_prefix}  read node block ({N:,} rows, {n_vars} cols): {t1 - t0:.1f}s", flush=True)

    conn = pd.read_csv(
        filename, sep=r"\s+", header=None,
        skiprows=3 + N, nrows=E,
        dtype=conn_dtype, engine="c", memory_map=True,
    ).to_numpy()
    t2 = time.perf_counter()
    print(f"{log_prefix}  read connectivity ({E:,} rows): {t2 - t1:.1f}s", flush=True)

    assert nodes.shape == (N, n_vars), f"{filename}: node block shape mismatch {nodes.shape}"
    assert conn.shape[0] == E, f"{filename}: connectivity row count mismatch {conn.shape[0]} != {E}"

    return nodes, conn


def read_variable_partition(filename, var_idx, include_connectivity,
                             node_dtype=np.float32, conn_dtype=np.int32, log_prefix=""):
    """
    Read only X,Y,Z + the chosen variable (by column index) of one partition,
    via pandas usecols so unrequested columns are never parsed.

    Returns:
        data : (N, 3) or (N, 4) array -- X,Y,Z[,var]. If var_idx is 0/1/2
               (X, Y, or Z itself), no duplicate column is added.
        conn : (E, 4) array (1-based, local), or None if include_connectivity=False
    """
    n_vars, N, E = parse_tecplot_header(filename)
    if not (0 <= var_idx < n_vars):
        raise ValueError(f"variable index {var_idx} out of range for {filename} "
                          f"(file has {n_vars} variables)")

    cols = sorted(set([0, 1, 2, var_idx]))

    t0 = time.perf_counter()
    df = pd.read_csv(
        filename, sep=r"\s+", header=None,
        skiprows=3, nrows=N,
        usecols=cols, dtype=node_dtype,
        engine="c", memory_map=True,
    )
    ordered_cols = [0, 1, 2] + ([var_idx] if var_idx not in (0, 1, 2) else [])
    data = df[ordered_cols].to_numpy(copy=False)
    del df
    gc.collect()
    t1 = time.perf_counter()
    print(f"{log_prefix}  read fields ({len(cols)} cols, {N:,} rows): {t1 - t0:.1f}s", flush=True)

    conn = None
    if include_connectivity:
        conn_df = pd.read_csv(
            filename, sep=r"\s+", header=None,
            skiprows=3 + N, nrows=E,
            dtype=conn_dtype, engine="c", memory_map=True,
        )
        conn = conn_df.to_numpy(copy=False)
        del conn_df
        gc.collect()
        assert conn.shape[0] == E, f"{filename}: connectivity row count mismatch"
        t2 = time.perf_counter()
        print(f"{log_prefix}  read connectivity ({E:,} rows): {t2 - t1:.1f}s", flush=True)

    assert data.shape[0] == N, f"{filename}: data row count mismatch {data.shape[0]} != {N}"
    return data, conn
