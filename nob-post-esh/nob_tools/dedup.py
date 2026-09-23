"""
Duplicate-node detection and merging for domain-decomposed meshes.

Uses exact coordinate hashing (quantize + sort + binary search), NOT a
KD-tree nearest-neighbor search. For METIS-style (graph) decomposition,
the interface between partitions is an irregular, non-planar set of
faces, so a geometric proximity search examines far more candidates than
necessary. Since a "duplicate" node is the same physical mesh point
written out by two partition files, its coordinates are effectively
identical -- an exact-match lookup is the right tool and is substantially
cheaper than tree search, regardless of how the domain was decomposed.
"""

import time
import numpy as np


def _quantize_keys(coords, coord_tol):
    """
    Pack (N,3) float64 coordinates into a single sortable int64 key per
    node, at resolution coord_tol. Falls back to a structured-dtype key
    if the quantized range doesn't fit the fast 21-bit-per-axis packing
    (this happens with very fine coord_tol relative to domain size --
    loosen coord_tol if you see this fallback triggering on your data
    and want the faster path).
    """
    scale = 1.0 / coord_tol
    q = np.round(coords * scale).astype(np.int64)
    q -= q.min(axis=0, keepdims=True)

    max_per_axis = q.max(axis=0)
    bits_needed = int(np.ceil(np.log2(max_per_axis.max() + 1))) if max_per_axis.max() > 0 else 1

    if bits_needed > 21:
        q_c = np.ascontiguousarray(q)
        return q_c.view([("", q_c.dtype)] * 3).ravel()

    return (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]


def find_duplicates_by_hash(coords1, coords2, coord_tol=1e-6):
    """
    For every node in coords2, determine whether it coincides (within
    coord_tol) with some node in coords1.

    Returns:
        is_dup     : (N2,) bool
        idx1_match : (N2,) int64, valid only where is_dup is True -- index
                     into coords1 of the matching node
    """
    key1 = _quantize_keys(coords1, coord_tol)
    key2 = _quantize_keys(coords2, coord_tol)

    order1 = np.argsort(key1, kind="mergesort")
    sorted_key1 = key1[order1]

    pos = np.searchsorted(sorted_key1, key2)
    pos_clipped = np.clip(pos, 0, len(sorted_key1) - 1)
    candidate_idx1 = order1[pos_clipped]

    in_range = pos < len(sorted_key1)
    is_dup = np.zeros(len(key2), dtype=bool)
    is_dup[in_range] = sorted_key1[pos_clipped[in_range]] == key2[in_range]

    idx1_match = np.full(len(key2), -1, dtype=np.int64)
    idx1_match[is_dup] = candidate_idx1[is_dup]

    return is_dup, idx1_match


def merge_two(data1, conn1, data2, conn2, coord_tol=1e-6, include_connectivity=True,
              log_prefix=""):
    """
    Merge partition 2 into accumulated partition 1. Works for any (N, k)
    data array as long as columns 0,1,2 are X,Y,Z -- so the same function
    serves both the full 13-variable case and the single-variable case.

    conn1 is assumed already 0-based/global; conn2 is 1-based/local to
    data2. Pass include_connectivity=False to skip connectivity entirely
    (data1/conn1, data2/conn2's conn arguments may then be None).

    Returns: (merged_data, merged_conn, n_duplicates)
    """
    N1 = data1.shape[0]
    N2 = data2.shape[0]

    coords1 = data1[:, :3].astype(np.float64, copy=False)
    coords2 = data2[:, :3].astype(np.float64, copy=False)

    t0 = time.perf_counter()
    is_dup, idx1_match = find_duplicates_by_hash(coords1, coords2, coord_tol)
    t1 = time.perf_counter()
    n_dup = int(is_dup.sum())
    print(f"{log_prefix}  dedup ({N1:,} vs {N2:,} nodes, hash-based): {t1 - t0:.2f}s, "
          f"found {n_dup:,} duplicates", flush=True)

    keep2 = ~is_dup
    n_new = int(keep2.sum())

    merged_data = np.vstack([data1, data2[keep2]])

    merged_conn = None
    if include_connectivity:
        new_ids2 = np.empty(N2, dtype=np.int64)
        new_ids2[is_dup] = idx1_match[is_dup]
        new_ids2[keep2] = np.arange(N1, N1 + n_new, dtype=np.int64)
        conn2_global = new_ids2[conn2.astype(np.int64) - 1]
        merged_conn = np.vstack([conn1, conn2_global])
        del new_ids2, conn2_global

    return merged_data, merged_conn, n_dup


def merge_partitions(nodes1, conn1, nodes2, conn2, coord_tol=1e-6):
    """
    Convenience wrapper matching the older full-variable-merge signature
    (always includes connectivity, returns 0-based global connectivity
    directly rather than a 3-tuple). Used by the full assembler.
    """
    conn1_global = conn1.astype(np.int64) - 1
    merged_data, merged_conn, n_dup = merge_two(
        nodes1, conn1_global, nodes2, conn2, coord_tol, include_connectivity=True
    )
    return merged_data, merged_conn
