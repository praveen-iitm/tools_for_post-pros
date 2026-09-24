"""
nob_tools -- shared library for reading, deduplicating, and assembling
domain-decomposed Tecplot (.dat) partitions.

Usage:
    from nob_tools import discover_iterations, read_variable_partition, merge_two
    # or
    import nob_tools
    nob_tools.discover_iterations(...)
"""

from .discovery import ALL_VAR_NAMES, FNAME_RE, discover_iterations
from .io_tecplot import parse_tecplot_header, read_tecplot_partition, read_variable_partition
from .dedup import find_duplicates_by_hash, merge_two, merge_partitions
from .memory_sizing import estimate_iteration_bytes, choose_worker_count, HAVE_PSUTIL
from .writers import write_npz, write_npy, write_dat, write_tecplot
from .assemble import (
    assemble_iteration_full,
    assemble_iteration_variable,
    process_iteration_full,
    process_iteration_variable,
)
from .time_average import (
    compute_reference_grid,
    time_average_field,
    horizontal_profile,
    write_profile_dat,
)

__all__ = [
    "ALL_VAR_NAMES", "FNAME_RE", "discover_iterations",
    "parse_tecplot_header", "read_tecplot_partition", "read_variable_partition",
    "find_duplicates_by_hash", "merge_two", "merge_partitions",
    "estimate_iteration_bytes", "choose_worker_count", "HAVE_PSUTIL",
    "write_npz", "write_npy", "write_dat", "write_tecplot",
    "assemble_iteration_full", "assemble_iteration_variable",
    "process_iteration_full", "process_iteration_variable",
    "compute_reference_grid", "time_average_field", "horizontal_profile", "write_profile_dat",
]
