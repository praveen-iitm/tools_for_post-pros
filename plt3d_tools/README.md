# plt3d_pipeline

Cluster-runnable post-processing pipeline for Rayleigh–Bénard convection
(RBC) DNS output written as Plot3D binaries. Refactored out of four
one-off notebooks into a shared library (`plt_3d_tools/`) and four
standalone, argparse-driven scripts (`scripts/`) meant to be submitted as
cluster jobs, run one after another.

```
plt3d_pipeline/
├── README.md
├── plt_3d_tools/              shared library — import this, don't run it
│   ├── __init__.py
│   ├── io.py                   read_grid, read_solution, list_solution_files, sort_key
│   ├── gradients.py             GradientOps: grad_x/y/z/r bound to one grid + one ref point
│   ├── plotting.py              plot_slice, plot_slice_2d (interactive use only)
│   └── parallel.py              run_parallel_accumulation (chunked, RAM-bounded, parallel)
└── scripts/                   run these on the cluster, IN ORDER
    ├── 01_compute_mean.py               time-mean u, v, w, T
    ├── 02_compute_fluctuations_tke.py   turbulent kinetic energy (TKE)
    ├── 03_two_point_correlation.py      two-point (z-lag) correlations R_ij
    └── 04_budget_terms.py               TKE- and temperature-variance-budget terms
```

Every script accepts `--solution-dir <folder>` and processes **every**
matching snapshot file in that folder itself — you never pass individual
filenames. Point it at the directory, set `--workers`, submit.

---

## 1. Input data format

Two kinds of Fortran unformatted-binary Plot3D files, both little-endian
with standard 4-byte Fortran record markers around each record:

- **Grid file** (`grid.xyz`, one per case): record 1 is `(nx, ny, nz)` as
  `int32`; record 2 is `x, y, z` each of length `nx*ny*nz` (`float32`),
  stacked and Fortran-ordered.
- **Snapshot / solution files** (`grid<block_num>.<time_num>.f`, one per
  saved timestep, e.g. `grid10731.19751636.f`): record 1 is
  `(nx, ny, nz, nvar)` as `int32`; record 2 is `nvar` variables of length
  `nx*ny*nz` each (`float32`), stacked and Fortran-ordered.

  Variable order in every snapshot (last axis index after reshaping):

  | index | 0 | 1 | 2 | 3 | 4 | 5      | 6      | 7      | 8      |
  |-------|---|---|---|---|---|--------|--------|--------|--------|
  | name  | u | v | w | p | e (≈T) | u_mean | v_mean | w_mean | e_mean |

  The pipeline only ever uses indices `0,1,2` (u,v,w) and `4` (e, treated
  as temperature `T` throughout — the original notebooks did the same).

Snapshot filenames are sorted by `(grid_num, time_num)` parsed out of the
filename (see `sort_key` below), **not** by string/alphabetical order, so
processing order matches simulation time order regardless of zero-padding.

---

## 2. `plt_3d_tools/` — shared library

### `plt_3d_tools/io.py`

| Name | Signature | Returns | Notes |
|---|---|---|---|
| `read_grid` | `read_grid(filename)` | `x, y, z, nx, ny, nz` | `x,y,z` are `ndarray` shape `(nx,ny,nz)`, `x[i,j,k]` matches Fortran `x(i,j,k)`. Prints the grid size once read. |
| `read_solution` | `read_solution(filename)` | `sol, nx, ny, nz, nvar, var_names` | `sol` is `ndarray` shape `(nx,ny,nz,nvar)`, `float32`. `var_names` is the list of variable names present (see table above), sliced to the actual `nvar` found in the file. |
| `sort_key` | `sort_key(fname)` | `(grid_num, time_num)` tuple of `int` | Parses a `grid<block>.<time>.f` filename. Used as the `key=` for sorting. |
| `list_solution_files` | `list_solution_files(directory, pattern='grid*.f')` | sorted `list[str]` | Globs `directory` for `pattern` and sorts by `sort_key`. This is what every script calls to turn `--solution-dir` into a file list — if you rename your snapshots you only need to change this one function (and `sort_key`). |

`VAR_NAMES` is also exported (`['u','v','w','p','e','u_mean','v_mean','w_mean','e_mean']`) if you need it directly.

Every read is a single sequential pass through the file (two `open`/`read`
calls for a solution file, four for a grid file) — no seeking, no partial
reads, so it's about as RAM/IO-light as reading the whole array can be.

### `plt_3d_tools/gradients.py`

`class GradientOps(x, y, z, ref_idx, ep=1e-12)`

Bundles the grid- and reference-point-dependent gradient machinery that
the original notebooks built as closures over module-level globals, so it
can be constructed explicitly and passed safely to worker processes.

Constructor arguments:
- `x, y, z` — the full grid coordinate arrays from `read_grid`.
- `ref_idx` — `(x_idx, y_idx, z_idx)` tuple, the fixed reference point all
  two-point quantities are measured relative to.
- `ep` — small regularizer (default `1e-12`) added to `r` before dividing,
  to avoid a `0/0` at the reference point itself.

On construction it precomputes and stores:
- `x1d, y1d, z1d` — the 1D coordinate lines along each axis (`x[:,1,1]`
  etc.), used as the non-uniform spacing for `np.gradient`.
- `r_x, r_y, r_z, r` — the vector/scalar distance from every grid point to
  the reference point.
- `co_eff` — the `1 / (4π(r+ep)²)` weighting used throughout the
  budget-term calculations in script 04.

Methods:

| Method | What it computes |
|---|---|
| `grad_x(var)` / `grad_y(var)` / `grad_z(var)` | Second-order-accurate 1D gradient of a 3D field along that axis (`np.gradient(..., edge_order=2)`), respecting the actual (non-uniform) grid spacing. |
| `grad_r(var)` | Radial derivative towards the reference point: `(r̂ · ∇)var`, built from the three axis gradients above. Intermediate gradients are deleted as soon as they're combined, to avoid holding 4 extra full-size arrays per call. |

### `plt_3d_tools/plotting.py`

Interactive-only — none of the cluster scripts call these; they're kept
so you can still `import` them to sanity-check `.npy` outputs from a
notebook or a Python shell on your workstation.

| Function | Signature | Purpose |
|---|---|---|
| `plot_slice` | `plot_slice(x, y, z, var, slice_id)` | pcolormesh of `var[:,:,slice_id]` (an x–y slice at a fixed z-index), titled with the actual z-coordinate. |
| `plot_slice_2d` | `plot_slice_2d(x, y, var)` | pcolormesh of an already-2D field (e.g. a z-averaged quantity) over the x–y grid at `z`-index 0. |

Both just open a matplotlib window (`plt.show()`) — fine for a laptop
session, not meant for headless cluster nodes.

### `plt_3d_tools/parallel.py`

`run_parallel_accumulation(files, init_fn, process_fn, n_workers=None, extra=())`

The core parallelism/RAM primitive every script is built on. Rather than
sending one result back to the main process per *file* (N inter-process
transfers of full-grid arrays for N files), it:

1. Splits `files` into `n_workers` contiguous chunks (`_chunk`).
2. Runs one worker process per chunk (`multiprocessing.Pool`). Each
   worker calls `state = init_fn(*extra)` once, then loops over its own
   files calling `process_fn(fn, state, *extra)` for each one — reading,
   using, and discarding one snapshot at a time, mutating `state` in
   place, exactly like the original serial notebooks did.
3. Only the final per-*worker* accumulator (one message per worker, not
   per file) is sent back to the main process.
4. The main process element-wise sums the `n_workers` partial-result
   dicts together and returns the total.

Peak RAM ≈ `n_workers × (one snapshot in memory + that worker's
accumulator arrays)` — independent of how many thousands of snapshot
files exist in the run.

Parameters:
- `files` — ordered list of snapshot paths (order only affects progress
  printing; the math is order-independent since it's a sum).
- `init_fn(*extra) -> dict[str, np.ndarray]` — called once **per worker**
  to build that worker's fresh, zeroed accumulator dict.
- `process_fn(filename, state, *extra) -> None` — called once **per
  file**; must mutate `state` in place and must not keep references to
  that file's raw arrays afterwards.
- `n_workers` — defaults to `os.cpu_count() - 1` (minimum 1).
- `extra` — read-only extra arguments (mean fields, grid, indices, …)
  forwarded to both `init_fn` and `process_fn`. Pickled once per worker
  at pool start, not once per file.

If `n_workers` resolves to 1 (or there's only one chunk), it runs
in-process with no `Pool` at all, so single-core debugging needs no
special-casing.

`init_fn` and `process_fn` must have **matching signatures** — both are
called as `fn(*extra)` / `fn(file, state, *extra)`, so every script's
private `_init`/`_process` pair takes the identical extra-argument list
in the identical order, even where one of them doesn't use every
argument (see e.g. script 01/03's `_init`, which ignores some of the
`extra` tuple).

---

## 3. `scripts/` — the four pipeline stages

Run `python scripts/0N_*.py --help` at any time for the authoritative
flag list; this section explains what each one does and why.

### `scripts/01_compute_mean.py` — time-mean fields

Computes `mean_u`, `mean_v`, `mean_w`, `mean_T`: the plain time-average of
each field over every snapshot in `--solution-dir`.

| Flag | Required | Meaning |
|---|---|---|
| `--solution-dir` | yes | Folder containing `grid*.f` snapshots. |
| `--output-dir` | yes | Where `mean_*.npy` (and optionally `sum_*.npy`) are written. Created if missing. |
| `--tag` | no | Suffix appended to output filenames, e.g. `--tag 731-931` → `mean_u_731-931.npy`. Default: no suffix. |
| `--workers` | no | Worker processes. Default `cpu_count() - 1`. |
| `--save-sum` | no (flag) | Also save the raw (unnormalized) sums as `sum_*.npy`, matching the original notebook's behaviour. |

**Outputs** (in `--output-dir`): `mean_u<tag>.npy`, `mean_v<tag>.npy`,
`mean_w<tag>.npy`, `mean_T<tag>.npy` — each `float32`, shape `(nx,ny,nz)`.
With `--save-sum`, also `sum_u<tag>.npy` etc. (`float64`).

Accumulation is `float64` (cheap here — only 4 full-size arrays) to keep
a sum over thousands of snapshots numerically accurate; results are cast
to `float32` on save.

### `scripts/02_compute_fluctuations_tke.py` — turbulent kinetic energy

Computes time-averaged TKE against **two** different reference means:

- `TKE_t` — fluctuation relative to the full 3D time-mean (`mean_u/v/w`
  from script 01, spatially varying).
- `TKE_tz` — fluctuation relative to the **z-averaged** time-mean
  (`mean_u/v/w` averaged over the z/homogeneous direction, then
  broadcast back to the full 3D shape) — i.e. treats z as a statistically
  homogeneous direction.

| Flag | Required | Meaning |
|---|---|---|
| `--solution-dir` | yes | Folder containing `grid*.f` snapshots. |
| `--mean-dir` | yes | Directory with `mean_u/v/w_*.npy` from script 01. |
| `--mean-tag` | no | Must match the `--tag` script 01 was run with. |
| `--output-dir` | yes | Where `TKE_t.npy` / `TKE_tz.npy` are written. |
| `--workers` | no | Default `cpu_count() - 1`. |

**Outputs** (in `--output-dir`): `TKE_t.npy`, `TKE_tz.npy` — each
`float32`, shape `(nx,ny,nz)`, defined as `⟨u′²+v′²+w′²⟩` averaged over
all snapshots, with `u′ = u − ⟨u⟩` using the corresponding (3D or
z-averaged) mean.

### `scripts/03_two_point_correlation.py` — two-point (z-lag) correlations

Computes the 16 two-point correlations `R_ij(x,y,z) = ⟨i′(x_idx,y_idx,z₀)
· j′(x,y,z₀+z)⟩` for `i,j ∈ {u,v,w,T}`, at a fixed reference point
`(x_idx, y_idx)`, averaged over the z-lag `z₀` and over all snapshots.

This is an **FFT-based circular cross-correlation** rather than the naive
roll-and-accumulate loop:

```
R[x,y,z] = Σ_z0  a[x_idx,y_idx,z0] · b[x,y,(z0+z) mod Nz]
         = irfft( conj(rfft(a_line)) · rfft(b, axis=z) )
```

— mathematically identical to a per-`z0` `np.roll`-and-accumulate loop,
but `O(Nx·Ny·Nz·log Nz)` per field pair per snapshot instead of
`O(Nx·Ny·Nz²)`. Each snapshot's own forward FFT is computed **once** per
field and reused across all 4 reference signals (4 forward FFTs + 16
cheap inverse FFTs per snapshot, instead of 16+16). Accumulators are
`float32`.

| Flag | Required | Meaning |
|---|---|---|
| `--solution-dir` | yes | Folder containing `grid*.f` snapshots. |
| `--mean-dir` | yes | Directory with `mean_u/v/w/T_*.npy` from script 01. |
| `--mean-tag` | no | Must match script 01's `--tag`. |
| `--output-dir` | yes | **Parent** directory — results are written to `<output-dir>/post_proc_3-tp_<x_idx>_<y_idx>/`. |
| `--x-idx` | yes | Reference point x-grid-index. |
| `--y-idx` | yes | Reference point y-grid-index. |
| `--workers` | no | Default `cpu_count() - 1`. |

**Outputs** (in `<output-dir>/post_proc_3-tp_<x>_<y>/`): 16 files
`R_uu.npy, R_uv.npy, R_uw.npy, R_uT.npy, R_vu.npy, ..., R_TT.npy` — each
`float32`, shape `(nx,ny,nz)`.

### `scripts/04_budget_terms.py` — TKE- and temperature-variance-budget terms

Computes the two-point TKE-budget terms (`E_*`, saved under `e_terms/`)
and temperature-variance-budget terms (`H_*`, saved under `h_terms/`) at
a fixed reference point `(x_idx, y_idx, z_idx)`.

| Flag | Required | Meaning |
|---|---|---|
| `--solution-dir` | yes | Folder containing `grid*.f` snapshots (needed for `E_inter_f`/`H_inter_f`, which require one more streaming pass). |
| `--grid-file` | yes | Path to `grid.xyz`. |
| `--mean-dir` | yes | Directory with `mean_u/v/w/T_*.npy` from script 01. |
| `--mean-tag` | no | Must match script 01's `--tag`. |
| `--corr-dir` | yes | Directory with the 16 `R_*.npy` files from script 03 (i.e. its `post_proc_3-tp_<x>_<y>/` output folder). |
| `--output-dir` | yes | **Parent** directory — results go to `<output-dir>/post-proc-4_terms_calc_<x_idx>_<y_idx>_<z_idx>/`. |
| `--x-idx` | yes | Reference point x-grid-index. Must match the `R_*.npy` files' reference point. |
| `--y-idx` | yes | Reference point y-grid-index. Must match the `R_*.npy` files' reference point. |
| `--z-idx` | no | Reference point z-grid-index. Default `nz // 2`. |
| `--workers` | no | Default `cpu_count() - 1`. Only used for the `E_inter_f`/`H_inter_f` streaming pass. |

**Outputs**:
```
<output-dir>/post-proc-4_terms_calc_<x>_<y>_<z>/
├── e_terms/
│   ├── E_prod_h.npy     Homogeneous production
│   ├── E_prod_I.npy     Inhomogeneous production
│   ├── E_inter_m.npy    Interscale transport, mean velocity
│   ├── E_inter_f.npy    Interscale transport, fluctuating velocity
│   └── E_buoy.npy       Buoyancy source term
└── h_terms/
    ├── H_prod_h.npy     Homogeneous production
    ├── H_prod_I.npy     Inhomogeneous production
    ├── H_inter_m.npy    Interscale transport, mean velocity
    ├── H_inter_f.npy    Interscale transport, fluctuating velocity
    └── H_buoy.npy       Buoyancy source term
```
All `ndarray`, shape `(nx,ny,nz)` (`co_eff`-weighted, same normalization
as the original notebook).

Two implementation notes specific to this script:

1. **Merged streaming pass.** The original notebook read every snapshot
   *twice* — once for `E_inter_f`'s accumulators (`xg,yg,zg`, needing
   u/v/w fluctuations) and again for `H_inter_f`'s (`wt_u,wt_v,wt_w`,
   needing u/v/w/T fluctuations). Since the second read is a strict
   superset of the first, this script computes all six accumulators in
   **one** combined pass, halving the I/O for this stage.
2. **Bug fix vs. the original notebook.** The original `H_inter_f` loop
   assigned `wt_u = ...`, `wt_v = ...`, `wt_w = ...` with `=` instead of
   `+=` on every iteration — so only the *last* snapshot's contribution
   survived the final division by `n_snaps`, unlike the structurally
   identical `E_inter_f` loop (which correctly used `+=`). This script
   accumulates both consistently. If you need the original (buggy)
   behaviour reproduced exactly, say so and it can be added back as an
   option.

---

## 4. Running the full pipeline

```bash
cd plt3d_pipeline

# 1. Time-mean u, v, w, T
python scripts/01_compute_mean.py \
    --solution-dir /path/to/Case/output \
    --output-dir   /path/to/post_proc_1-mean \
    --tag 731-931 \
    --workers 16

# 2. TKE (vs. full 3D mean, and vs. z-averaged mean)
python scripts/02_compute_fluctuations_tke.py \
    --solution-dir /path/to/Case/output \
    --mean-dir     /path/to/post_proc_1-mean \
    --mean-tag 731-931 \
    --output-dir   /path/to/post_proc_2-fluc \
    --workers 16

# 3. Two-point (z-lag) correlations R_ij at a reference point
python scripts/03_two_point_correlation.py \
    --solution-dir /path/to/Case/output \
    --mean-dir     /path/to/post_proc_1-mean \
    --mean-tag 731-931 \
    --output-dir   /path/to \
    --x-idx 25 --y-idx 222 \
    --workers 16

# 4. TKE-budget / temperature-variance-budget terms
python scripts/04_budget_terms.py \
    --solution-dir /path/to/Case/output \
    --grid-file    /path/to/Case/output/grid.xyz \
    --mean-dir     /path/to/post_proc_1-mean \
    --mean-tag 731-931 \
    --corr-dir     /path/to/post_proc_3-tp_25_222 \
    --output-dir   /path/to \
    --x-idx 25 --y-idx 222 \
    --workers 16
```

Each stage only depends on the `.npy` outputs of earlier stages (never on
another stage's script or in-memory state), so you can submit them as
four separate, dependent Slurm/PBS jobs (`--dependency=afterok:<jobid>`
or equivalent) instead of one long interactive run.

### Choosing `--x-idx` / `--y-idx` / `--z-idx`

Scripts 03 and 04 are both computed *relative to one fixed grid point*.
`--x-idx`/`--y-idx` in script 04 **must match** the ones script 03 was
run with (that's what selects which `post_proc_3-tp_<x>_<y>/` folder to
read `R_*.npy` from via `--corr-dir`); `--z-idx` is script-04-only and
defaults to the mid-plane (`nz // 2`) if not given.

### Choosing `--workers`

Set it to match your job's `--cpus-per-task` (Slurm) or equivalent —
`run_parallel_accumulation` will spin up exactly that many worker
processes, no more. Leaving it unset uses `os.cpu_count() - 1`, which on
a shared/oversubscribed node may not match what the scheduler actually
gave you, so it's worth setting explicitly in cluster jobs.

---

## 5. Requirements

```
numpy
```

`matplotlib` is only needed if you use `plt_3d_tools.plotting` (`h5py`,
which the original notebook imported but never used, is not a dependency
here).

No non-standard-library dependency is used for the parallelism —
`multiprocessing` (standard library) only.

---

## 6. Extending / troubleshooting

- **Different snapshot naming scheme?** Only `sort_key` and
  `list_solution_files` in `plt_3d_tools/io.py` need to change; every
  script calls through those, not `glob` directly.
- **Want a different reduction (e.g. max instead of sum)?** Write a new
  `init_fn`/`process_fn` pair and pass them to
  `run_parallel_accumulation` — `init_fn` just needs to return a dict of
  arrays, `process_fn` just needs to mutate that dict given one file.
  `parallel.py` itself has no domain-specific logic.
- **Progress looks interleaved/out of order in the log?** Expected —
  each worker prints its own progress against its own chunk
  independently (`[worker N] i/total  filename`), so lines from
  different workers interleave. The final numeric result is unaffected
  since it's a plain sum.
- **`AssertionError: Marker mismatch (...)`** from `read_grid` /
  `read_solution` means the Fortran record markers didn't match the
  expected byte count — almost always a corrupted/truncated file or a
  file written in a different (e.g. big-endian, double-precision)
  format than the `<i4`/`<f4` this reader assumes.
- **Re-running a stage with different `--x-idx`/`--y-idx`/`--z-idx`**
  doesn't require re-running script 01 or 02 — only scripts 03 and 04
  depend on the reference point, and each run writes to its own
  `..._<x>_<y>[_<z>]/` subfolder, so multiple reference points can
  coexist under the same `--output-dir`.
