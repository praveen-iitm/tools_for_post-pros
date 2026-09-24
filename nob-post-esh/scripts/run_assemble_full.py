"""
Assemble the full (all-variable) mesh for one or more iterations --
using the shared tools/ library.

Usage:
    python run_assemble_full.py \
        --input-dir /media/user/Data1/NOB/nob_rbc_air_3d/dati \
        --output-dir /media/user/Data1/NOB/nob_rbc_air_3d/assembled \
        --format npz
"""

import os
import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from nob_tools import discover_iterations, choose_worker_count, process_iteration_full


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--pattern", default="tecplot_blk*_iter*.dat")
    ap.add_argument("--coord-tol", type=float, default=1e-6)
    ap.add_argument("--format", choices=["npz", "dat"], default="npz")
    ap.add_argument("--iters", nargs="*", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--max-ram-gb", type=float, default=None)
    ap.add_argument("--safety-factor", type=float, default=4.0)
    ap.add_argument("--ram-margin", type=float, default=0.8)
    ap.add_argument("--node-dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--conn-dtype", default="int32", choices=["int32", "int64"])
    args = ap.parse_args()

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

    n_workers = choose_worker_count(by_iter, n_cols=13, include_connectivity=True,
                                     requested_workers=args.workers, max_ram_gb=args.max_ram_gb,
                                     safety_factor=args.safety_factor, ram_margin=args.ram_margin)

    node_dtype = np.float32 if args.node_dtype == "float32" else np.float64
    conn_dtype = np.int32 if args.conn_dtype == "int32" else np.int64

    n_done, n_skipped, n_error = 0, 0, 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                process_iteration_full, it, blocks, args.output_dir, args.format,
                args.coord_tol, node_dtype, conn_dtype
            ): it
            for it, blocks in by_iter.items()
        }

        for future in as_completed(futures):
            result = future.result()
            it = result["iter"]
            if result["status"] == "ok":
                n_done += 1
                print(f"[iter {it}] DONE -> {result['path']} "
                      f"({result['n_nodes']:,} nodes, {result['n_elems']:,} elements) "
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
