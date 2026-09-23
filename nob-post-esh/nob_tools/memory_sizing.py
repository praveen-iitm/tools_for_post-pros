"""
Estimate per-iteration memory footprint from file headers (no data read),
and pick a safe number of parallel workers from available RAM.
"""

import os

try:
    import psutil
    HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False

from .io_tecplot import parse_tecplot_header

# Fraction of "available" RAM the sizer is allowed to target. Left as
# headroom for estimation error, OS buffers, and the interpreter itself --
# do NOT raise this to 1.0: a sizing that targets 100% of available RAM
# has zero margin for the estimate being even slightly low, and a single
# worker exceeding it triggers the OOM killer (SIGKILL -> BrokenProcessPool,
# with no Python-catchable error to explain why).
DEFAULT_RAM_MARGIN = 0.8


def estimate_iteration_bytes(blocks, n_cols, include_connectivity, dtype_size=4,
                              conn_dtype_size=4, safety_factor=4.0):
    """
    blocks: list of (block_number, filepath) for one iteration.
    n_cols: number of field columns being read (3 or 4 for single-variable
            extraction, 13 for the full assembler).
    safety_factor: multiplier covering pandas parsing overhead and the
        several full-size intermediate arrays merge_two allocates at once
        (key1, key2, order1, sorted_key1, is_dup, idx1_match, plus the
        merged output coexisting briefly with its inputs). 4.0 is a
        reasonable default; raise it if you still see OOM at the sized
        worker count.
    """
    total = 0
    for blk, fp in blocks:
        n_vars, N, E = parse_tecplot_header(fp)
        total += N * n_cols * dtype_size
        if include_connectivity:
            total += E * 4 * conn_dtype_size
    return int(total * safety_factor)


def choose_worker_count(by_iter, n_cols, include_connectivity, requested_workers,
                         max_ram_gb, safety_factor=4.0, ram_margin=DEFAULT_RAM_MARGIN):
    """
    Pick a worker count that fits comfortably in available RAM.

    requested_workers: if given, used as-is (no auto-sizing).
    max_ram_gb: if given, used as the RAM budget instead of psutil's
        "available" reading (useful on shared machines, or for
        reproducible sizing across heterogeneous cluster nodes).
    ram_margin: fraction of the RAM budget the sizer is allowed to target
        (see DEFAULT_RAM_MARGIN above -- always leaves headroom).
    """
    if requested_workers is not None:
        return max(1, requested_workers)

    if max_ram_gb is not None:
        available_bytes = max_ram_gb * (1024 ** 3)
    elif HAVE_PSUTIL:
        available_bytes = psutil.virtual_memory().available
    else:
        print("psutil not installed and --max-ram-gb not given; defaulting to 1 worker. "
              "Install psutil or pass --max-ram-gb / --workers to parallelize.")
        return 1

    budget_bytes = available_bytes * ram_margin

    max_iter_bytes = max(
        estimate_iteration_bytes(blocks, n_cols, include_connectivity, safety_factor=safety_factor)
        for blocks in by_iter.values()
    )
    if max_iter_bytes <= 0:
        return 1

    ram_limited = max(1, int(budget_bytes // max_iter_bytes))
    cpu_limited = os.cpu_count() or 1
    n_iters = len(by_iter)

    workers = min(ram_limited, cpu_limited, n_iters)
    print(f"RAM-based sizing: available={available_bytes / 1e9:.1f} GB, "
          f"using {ram_margin * 100:.0f}% margin -> budget={budget_bytes / 1e9:.1f} GB, "
          f"largest iteration estimate={max_iter_bytes / 1e9:.2f} GB ({safety_factor}x safety factor) "
          f"-> RAM allows {ram_limited} concurrent, CPU allows {cpu_limited}, "
          f"{n_iters} iteration(s) total -> using {workers} worker(s)")
    return max(1, workers)
