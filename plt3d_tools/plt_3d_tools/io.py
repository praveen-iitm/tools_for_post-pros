"""
Read Plot3D grid/solution files written by the Fortran solver (unformatted
binary, little-endian, Fortran record markers), and list/sort snapshot files
in a run directory.
"""
import glob
import os

import numpy as np

VAR_NAMES = ['u', 'v', 'w', 'p', 'e', 'u_mean', 'v_mean', 'w_mean', 'e_mean']


def read_grid(filename):
    """Read a single-block Plot3D grid written by Fortran (unformatted binary).

    Returns
    -------
    x, y, z : ndarray, shape (nx, ny, nz)
        Grid coordinates. x[i, j, k] corresponds to Fortran x(i,j,k).
    nx, ny, nz : int
        Grid dimensions.
    """
    with open(filename, 'rb') as f:
        # Leading record marker -> dimensions: nx, ny, nz
        marker = f.read(4)
        nbytes = np.frombuffer(marker, dtype='<i4')[0]
        dims = np.frombuffer(f.read(nbytes), dtype='<i4')
        nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])
        assert np.frombuffer(f.read(4), dtype='<i4')[0] == nbytes, \
            "Marker mismatch (dimensions)"

        # Second record marker -> x, y, z stacked
        marker = f.read(4)
        nbytes = np.frombuffer(marker, dtype='<i4')[0]
        coords = np.frombuffer(f.read(nbytes), dtype='<f4')
        assert np.frombuffer(f.read(4), dtype='<i4')[0] == nbytes, \
            "Marker mismatch (coordinates)"

    npts = nx * ny * nz
    x = coords[0 * npts: 1 * npts].reshape((nx, ny, nz), order='F')
    y = coords[1 * npts: 2 * npts].reshape((nx, ny, nz), order='F')
    z = coords[2 * npts: 3 * npts].reshape((nx, ny, nz), order='F')

    print(f"Grid: {nx} x {ny} x {nz} = {npts:,} points")
    return x, y, z, nx, ny, nz


def read_solution(filename):
    """Read a Plot3D solution file written by Fortran (unformatted binary).

    Returns
    -------
    sol : ndarray, shape (nx, ny, nz, nvar)
        4D array with last axis indexing the variable.

        Variable order (last axis index):
        0: u       1: v       2: w       3: p        4: e
        5: u_mean  6: v_mean  7: w_mean  8: e_mean
    nx, ny, nz : int
        Grid dimensions.
    nvar : int
        Number of variables.
    """
    with open(filename, 'rb') as f:
        # Record 1: nx, ny, nz, nvar
        marker = f.read(4)
        nbytes = np.frombuffer(marker, dtype='<i4')[0]
        header = np.frombuffer(f.read(nbytes), dtype='<i4')
        nx, ny, nz, nvar = int(header[0]), int(header[1]), int(header[2]), int(header[3])
        assert np.frombuffer(f.read(4), dtype='<i4')[0] == nbytes, \
            "Marker mismatch (header)"

        # Record 2: all variables stacked
        marker = f.read(4)
        nbytes = np.frombuffer(marker, dtype='<i4')[0]
        data = np.frombuffer(f.read(nbytes), dtype='<f4')
        assert np.frombuffer(f.read(4), dtype='<i4')[0] == nbytes, \
            "Marker mismatch (data)"

    npts = nx * ny * nz
    sol = np.stack([
        data[v * npts: (v + 1) * npts].reshape((nx, ny, nz), order='F')
        for v in range(nvar)
    ], axis=-1)
    # sol[i, j, k, v] = vth variable at (i,j,k)  -- matches Fortran x(i,j,k)

    return sol, nx, ny, nz, nvar, VAR_NAMES[:nvar]


def sort_key(fname):
    """Sort key for 'grid<block_num>.<time_num>.f' snapshot filenames."""
    base = os.path.basename(fname)
    grid_num, time_num = base[4:].removesuffix('.f').split('.')
    return int(grid_num), int(time_num)


def list_solution_files(directory, pattern='grid*.f'):
    """Return every snapshot file in `directory` matching `pattern`, sorted
    in ascending (grid_num, time_num) order."""
    files = sorted(glob.glob(os.path.join(directory, pattern)), key=sort_key)
    return files
