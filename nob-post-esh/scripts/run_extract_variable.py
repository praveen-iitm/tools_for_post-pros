"""
Extract one field variable, assembled across blocks, for one or more
iterations -- using the shared tools/ library.

Usage:
    python run_extract_variable.py \
        --input-dir /media/user/Data1/NOB/nob_rbc_air_3d/dati \
        --output-dir /media/user/Data1/NOB/nob_rbc_air_3d/assembled_T \
        --variable T --format npy

--format npy (default): point cloud only -- X,Y,Z + the chosen variable,
    no connectivity at all. This is the right choice for point-based
    analysis (e.g. two-point correlations) since the solver's mesh is
    unstructured and connectivity isn't meaningful for that use case.
    Output is a single self-describing .npy structured array; load with:
        arr = np.load(path); arr['T']; arr['X']; ...
--format dat: unchanged Tecplot ASCII output with full FE connectivity.
"""

import os
import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from nob_tools import (
    ALL_VAR_NAMES, discover_iterations, choose_worker_count, process_iteration_variable,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--pattern", default="tecplot_blk*_iter*.dat")
    ap.add_argument("--variable", required=True,
                     help=f"Variable to extract. One of: {', '.join(ALL_VAR_NAMES)}")
    ap.add_argument("--coord-tol", type=float, default=1e-6)
    ap.add_argument("--format", choices=["npy", "dat"], default="npy",
                     help="npy: point cloud, no connectivity (default). "
                          "dat: full Tecplot mesh with connectivity, unchanged.")
    ap.add_argument("--iters", nargs="*", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--max-ram-gb", type=float, default=None)
    ap.add_argument("--safety-factor", type=float, default=4.0)
    ap.add_argument("--ram-margin", type=float, default=0.8,
                     help="Fraction of the RAM budget to target (leaves headroom for estimation error).")
    ap.add_argument("--node-dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--conn-dtype", default="int32", choices=["int32", "int64"])
    ap.add_argument("--no-connectivity", action="store_true",
                     help="Only relevant with --format dat (npy never includes connectivity).")
    args = ap.parse_args()

    if args.variable not in ALL_VAR_NAMES:
        raise SystemExit(f"--variable must be one of: {', '.join(ALL_VAR_NAMES)} (got {args.variable!r})")
    var_idx = ALL_VAR_NAMES.index(args.variable)
    # npy output is always a point cloud -- connectivity is never read for
    # it (forced off here so the RAM estimate below reflects that too).
    include_connectivity = (args.format == "dat") and not args.no_connectivity
    n_cols = 3 if var_idx in (0, 1, 2) else 4

    os.makedirs(args.output_dir, exist_ok=True)

    by_iter = discover_iterations(args.input_dir, args.pattern)
    if args.iters is not None:
        by_iter = {it: blocks for it, blocks in by_iter.items() if it in args.iters}
    if not by_iter:
        print("No matching iteration/block files found.")
        return

    print(f"Found {len(by_iter)} iteration(s): {list(by_iter.keys())}")
    for it, blocks in by_iter.items():
        print(f"  iter {it}: {len(blocks)} block(s)")
    print(f"Extracting variable '{args.variable}' (column {var_idx}), "
          f"connectivity={'included' if include_connectivity else 'skipped'}, "
          f"coord_tol={args.coord_tol}")

    n_workers = choose_worker_count(by_iter, n_cols, include_connectivity,
                                     args.workers, args.max_ram_gb, args.safety_factor,
                                     args.ram_margin)

    node_dtype = np.float32 if args.node_dtype == "float32" else np.float64
    conn_dtype = np.int32 if args.conn_dtype == "int32" else np.int64

    n_done, n_skipped, n_error = 0, 0, 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                process_iteration_variable, it, blocks, args.output_dir, args.format,
                args.variable, var_idx, include_connectivity, args.coord_tol,
                node_dtype, conn_dtype
            ): it
            for it, blocks in by_iter.items()
        }

        for future in as_completed(futures):
            result = future.result()
            it = result["iter"]
            if result["status"] == "ok":
                n_done += 1
                print(f"[iter {it}] DONE -> {result['path']} "
                      f"({result['n_points']:,} points, {result['n_elems']:,} elements) "
                      f"[{n_done + n_skipped + n_error}/{len(by_iter)}]")
            elif result["status"] == "skipped":
                n_skipped += 1
                print(f"[iter {it}] SKIPPED (already exists): {result['path']} "
                      f"[{n_done + n_skipped + n_error}/{len(by_iter)}]")
            else:
                n_error += 1
                print(f"[iter {it}] ERROR: {result['error']} "
                      f"[{n_done + n_skipped + n_error}/{len(by_iter)}]")

    print(f"\nFinished: {n_done} assembled, {n_skipped} skipped, {n_error} failed "
          f"(out of {len(by_iter)} total).")


if __name__ == "__main__":
    main()
