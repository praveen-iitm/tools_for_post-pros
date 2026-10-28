"""
Write assembled data out in one of three ways:
  - write_npy : point cloud (X,Y,Z + variables, NO connectivity) as a
                single self-describing .npy structured array. Used by the
                single-variable extractor, since a point cloud for
                analysis (e.g. two-point correlations) doesn't need the
                solver's unstructured mesh connectivity.
  - write_npz : full mesh (nodes + connectivity together) as .npz. Used
                by the full multi-variable assembler, which needs both
                arrays stored together.
  - write_dat : Tecplot ASCII .dat, unchanged -- full unstructured mesh
                with connectivity, same format as the solver's own output.
"""

import numpy as np


def write_npz(filepath, data, conn, var_names):
    """
    data      : (N, k) array -- e.g. all 13 variables
    conn      : (E, 4) array, or None to omit connectivity
    var_names : list of column names matching data's columns
    """
    kwargs = dict(data=data, var_names=var_names)
    if conn is not None:
        kwargs["conn"] = conn
    np.savez_compressed(filepath, **kwargs)


def write_npy(filepath, data, var_names):
    """
    Write a point cloud (no connectivity) as a single self-describing .npy
    file: a structured array with one named field per variable, so it
    carries its own column names instead of needing a separate var_names
    file alongside it.

    data      : (N, k) array -- columns in the same order as var_names
                (e.g. X,Y,Z,T). No connectivity is stored -- the solver's
                mesh is unstructured, but for point-cloud analysis (e.g.
                two-point correlations) only coordinates + field values
                are needed, so connectivity is dropped entirely here.
    var_names : list of column names matching data's columns

    Load it back with:
        arr = np.load(filepath)
        arr['T']                  # the T column
        np.column_stack([arr['X'], arr['Y'], arr['Z']])   # coordinates
    """
    assert data.shape[1] == len(var_names), (
        f"data has {data.shape[1]} columns but {len(var_names)} var_names given"
    )
    dtype = [(name, data.dtype) for name in var_names]
    structured = np.empty(data.shape[0], dtype=dtype)
    for i, name in enumerate(var_names):
        structured[name] = data[:, i]
    np.save(filepath, structured)


def write_dat(filepath, data, conn, var_names, zonetype="FETETRAHEDRON"):
    """
    Write a Tecplot ASCII file. If conn is None, writes a plain ordered
    point zone (F=POINT) instead of an FE zone.
    """
    N = data.shape[0]
    with open(filepath, "w") as f:
        f.write(" VARIABLES = " + ",".join(f'"{v}"' for v in var_names) + "\n")
        if conn is not None:
            E = conn.shape[0]
            f.write(f" ZONE N={N}, E={E}\n")
            f.write(f" DATAPACKING=POINT, ZONETYPE={zonetype}\n")
            np.savetxt(f, data, fmt="%.12g")
            np.savetxt(f, conn + 1, fmt="%d")  # back to 1-based for Tecplot
        else:
            f.write(f' ZONE T="points", I={N}, F=POINT\n')
            np.savetxt(f, data, fmt="%.12g")


def write_tecplot(filepath, nodes, conn, var_names, one_based=True):
    """
    Alias kept for backward compatibility with the original single-file
    scripts. `conn` is expected 0-based (as produced by merge_two/
    merge_partitions); write_dat always converts to 1-based for Tecplot,
    so `one_based` here only controls whether that conversion happens.
    """
    if one_based:
        write_dat(filepath, nodes, conn, var_names)
    else:
        # Caller wants the raw 0-based indices written as-is (non-standard
        # for Tecplot, but supported for completeness).
        N, E = nodes.shape[0], conn.shape[0]
        with open(filepath, "w") as f:
            f.write(" VARIABLES = " + ",".join(f'"{v}"' for v in var_names) + "\n")
            f.write(f" ZONE N={N}, E={E}\n")
            f.write(" DATAPACKING=POINT, ZONETYPE=FETETRAHEDRON\n")
            np.savetxt(f, nodes, fmt="%.12g")
            np.savetxt(f, conn, fmt="%d")
