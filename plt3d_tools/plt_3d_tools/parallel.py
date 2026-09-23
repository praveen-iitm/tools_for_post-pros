"""
Chunked multiprocessing accumulation for streaming reductions over many
snapshot files, with bounded RAM.

Why chunked (instead of one task per file)
-------------------------------------------
Sending one partial result back to the main process per file means, for N
files, N inter-process transfers of full-grid arrays -- expensive and it
lets results pile up in the pipe if the main process falls behind.

Instead, the file list is split into `n_workers` contiguous chunks. Each
worker processes its chunk sequentially -- reading, using, and discarding
ONE snapshot at a time, exactly like the original serial notebooks did --
accumulating into a small number of local numpy arrays. Only the final
per-worker accumulator (one message per *worker*, not per *file*) is sent
back to the main process, which sums the (few) partial results together.

Peak RAM is therefore roughly:

    n_workers * (one snapshot in memory + that worker's accumulator arrays)

instead of scaling with the number of files.
"""
import multiprocessing as mp


def _chunk(files, n):
    n = min(n, len(files)) or 1
    k, m = divmod(len(files), n)
    chunks = []
    start = 0
    for i in range(n):
        size = k + (1 if i < m else 0)
        if size == 0:
            continue
        chunks.append(files[start:start + size])
        start += size
    return chunks


def _worker_entry(args):
    chunk, init_fn, process_fn, extra, worker_id = args
    state = init_fn(*extra)
    n = len(chunk)
    for i, fn in enumerate(chunk, 1):
        process_fn(fn, state, *extra)
        if i == 1 or i % 50 == 0 or i == n:
            print(f"  [worker {worker_id}] {i:5d}/{n}  {fn}", flush=True)
    return state


def run_parallel_accumulation(files, init_fn, process_fn, n_workers=None, extra=()):
    """
    Parameters
    ----------
    files : list of str
        Snapshot files to process, in the order they should be attributed
        (only matters for progress printing -- the math is order-independent).
    init_fn(*extra) -> dict[str, np.ndarray]
        Called once per worker to create that worker's fresh accumulator(s).
    process_fn(filename, state, *extra) -> None
        Called once per file. Must read the file, fold its contribution into
        `state` in place, and must NOT keep references to the file's raw
        arrays afterwards (so they can be garbage collected before the next
        file is read).
    n_workers : int, optional
        Defaults to os.cpu_count() - 1 (min 1).
    extra : tuple
        Extra read-only arguments (means, grid, indices, ...) forwarded to
        both init_fn and process_fn. Pickled once per worker, not per file.

    Returns
    -------
    dict[str, np.ndarray] -- the accumulators from init_fn, summed over all
    workers (i.e. over all files).
    """
    n_workers = n_workers or max(1, mp.cpu_count() - 1)
    chunks = _chunk(files, n_workers)
    print(f"Splitting {len(files)} files across {len(chunks)} worker(s)")

    tasks = [(c, init_fn, process_fn, extra, wid) for wid, c in enumerate(chunks, 1)]

    if len(chunks) == 1:
        results = [_worker_entry(tasks[0])]
    else:
        with mp.Pool(processes=len(chunks)) as pool:
            results = pool.map(_worker_entry, tasks)

    total = results[0]
    for r in results[1:]:
        for k in total:
            total[k] += r[k]
    return total
