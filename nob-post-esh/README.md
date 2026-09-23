# NOB RBC Tecplot Assembly Tools — README

Two scripts for reassembling domain-decomposed Tecplot `.dat` output (one
`.dat` file per METIS partition, per solver iteration) into merged results,
built on a shared `nob_tools/` library:

| Script | Produces | Use it for |
|---|---|---|
| `run_extract_variable.py` | One variable (+X,Y,Z), no connectivity, per iteration | Point-based analysis (e.g. two-point spatial correlations) — **this is what you want for the RBC correlation work** |
| `run_assemble_full.py` | All 13 variables + full FE connectivity, per iteration | Visualization in Tecplot/ParaView, or anything that needs the actual mesh structure |

If you're not sure which you need: if your analysis only needs field values
at point locations and not the tetrahedral mesh connectivity, use
`run_extract_variable.py`. It's lighter on memory and time since it never
reads the other ~9-10 variable columns or the connectivity block at all.

## Setup

Put the `nob_tools/` folder next to both scripts:

```
your_project/
  nob_tools/
    __init__.py
    discovery.py
    io_tecplot.py
    dedup.py
    memory_sizing.py
    writers.py
    assemble.py
  run_extract_variable.py
  run_assemble_full.py
```

Requires: `numpy`, `pandas`, `psutil` (optional but recommended — needed for
automatic worker sizing; without it you must pass `--workers` or
`--max-ram-gb` yourself).

```bash
pip install numpy pandas psutil
```

## Input file naming

Both scripts expect files named:

```
tecplot_blk<BLOCK>_iter<ITER>.dat
```

e.g. `tecplot_blk0000_iter5231932.dat`, `tecplot_blk0001_iter5231932.dat`, ...
All blocks sharing the same `<ITER>` are treated as one iteration's domain
decomposition and merged together. Header format expected:

```
VARIABLES = "X","Y","Z","U","V","W","P","T","rho","mu","beta","cp","kappa"
ZONE N=<node count>, E=<element count>
DATAPACKING=POINT, ZONETYPE=FETETRAHEDRON
<N lines of node data>
<E lines of connectivity>
```

---

## `run_extract_variable.py` — single-variable point cloud

### Basic usage

```bash
python run_extract_variable.py \
    --input-dir /path/to/dati \
    --output-dir /path/to/assembled_T \
    --variable T
```

Produces one file per iteration: `assembled_iter5231932_T.npy`, etc.

### Output format

**Default (`--format npy`) — point cloud, no connectivity.** Each output is
a single self-describing `.npy` structured array with named fields (`X`,
`Y`, `Z`, and your chosen variable). No mesh connectivity is read or stored
— this is intentional: for point-based analysis (e.g. two-point spatial
correlations) the FE connectivity isn't needed, and skipping it saves both
read time and memory.

Load it back like this:

```python
import numpy as np

arr = np.load("assembled_iter5231932_T.npy")
T = arr["T"]
coords = np.column_stack([arr["X"], arr["Y"], arr["Z"]])
```

**`--format dat`** — writes the full Tecplot ASCII format instead
(`X,Y,Z,<variable>` plus FE tetrahedron connectivity), if you need
something Tecplot itself can open, or need the mesh structure. Use
`--no-connectivity` together with `--format dat` if you want a `.dat` point
cloud instead (an ordered `F=POINT` zone with no connectivity block).

### Choosing a variable

`--variable` accepts any one of:

```
X, Y, Z, U, V, W, P, T, rho, mu, beta, cp, kappa
```

Only that column (plus X,Y,Z for merging) is ever parsed from the input
files — the other ~9–10 columns are skipped at the parsing level, not read
and discarded.

**Need several variables together** (e.g. U, V, W, T all at once for a joint
correlation)? Right now the script only pulls one variable per run — run it
once per variable and combine the resulting point clouds afterward (they
share identical point order per iteration, so this is a `np.column_stack`
after loading, not a re-merge):

```python
u = np.load("assembled_iter5231932_U.npy")
v = np.load("assembled_iter5231932_V.npy")
w = np.load("assembled_iter5231932_W.npy")
t = np.load("assembled_iter5231932_T.npy")
# all share the same X,Y,Z (verify: np.allclose(u['X'], v['X']), etc.)
```

If reading the same files multiple times becomes a bottleneck, `--variable`
could be extended to accept a list and pull several columns in a single
pass — ask if you want that added.

---

## `run_assemble_full.py` — full multi-variable mesh

### Basic usage

```bash
python run_assemble_full.py \
    --input-dir /path/to/dati \
    --output-dir /path/to/assembled \
    --format npz
```

Produces one file per iteration containing **all 13 variables and full
tetrahedron connectivity**: `assembled_iter5231932.npz`, etc.

### Output format

**`--format npz`** (default) — a `.npz` archive with two arrays:

```python
import numpy as np
d = np.load("assembled_iter5231932.npz")
nodes = d["nodes"]       # (N, 13) — X,Y,Z,U,V,W,P,T,rho,mu,beta,cp,kappa
conn  = d["conn"]        # (E, 4) — 0-based tetrahedron connectivity
var_names = d["var_names"]
```

`.npz` (not `.npy`) is used here because nodes and connectivity are two
different-shaped arrays that need to be stored together — unlike
`run_extract_variable.py`'s point-cloud output, which is just one array.

**`--format dat`** — writes back out as Tecplot ASCII, same layout as the
original per-block input files, just merged into one zone.

### When you actually need this

- Visualizing the full field in Tecplot or ParaView
- Anything that needs the FE mesh itself: interpolation using element
  connectivity, gradient computation via the mesh, volume integration over
  elements
- A full-fidelity snapshot for some other downstream tool

For point-based analysis (two-point correlations, PDFs, spectra computed
from scattered point values, etc.), you don't need this — use
`run_extract_variable.py` instead, since reading+merging+storing all 13
variables and ~126M connectivity rows per iteration costs considerably more
time and memory than pulling just what you need.

---

## Selecting which iterations to process

Both scripts default to processing every iteration found in `--input-dir`.
Restrict to specific ones with `--iters`:

```bash
python run_extract_variable.py --input-dir ... --output-dir ... \
    --variable T --iters 5231932 5231933
```

Iterations whose output file already exists are skipped automatically, so
re-running the same command after an interruption picks up where it left
off.

## Parallelism and memory

Both scripts process iterations concurrently via a process pool, one
process per iteration, auto-sized from available RAM by default:

```
RAM-based sizing: available=266.2 GB, using 80% margin -> budget=213.0 GB,
largest iteration estimate=X.XX GB (4.0x safety factor)
-> RAM allows N concurrent, CPU allows M, K iteration(s) total -> using N worker(s)
```

The sizer only targets 80% of available RAM by default (`--ram-margin 0.8`)
— never raise this to `1.0`. A sizing that targets 100% of available RAM
leaves no room for the estimate being even slightly low; if a single worker
exceeds it, the OS's OOM killer silently kills the process, which surfaces
in Python as an opaque `concurrent.futures.process.BrokenProcessPool` error
with no explanation attached.

If you hit `BrokenProcessPool`:

```bash
# Check whether it was actually an OOM kill:
dmesg -T | grep -i "killed process"

# Isolate whether it's a concurrency problem or one iteration alone is too big:
python run_extract_variable.py --input-dir ... --output-dir ... --variable T --workers 1
```

If `--workers 1` still fails, raise `--safety-factor` (default `4.0`) — this
is far more likely with `run_assemble_full.py`, since holding all 13
variables + connectivity per block is much heavier than one variable.

Manual overrides, if you'd rather not rely on auto-sizing (both scripts):

| Flag | Purpose |
|---|---|
| `--workers N` | Use exactly N worker processes, skip auto-sizing entirely |
| `--max-ram-gb G` | Use G GB as the RAM budget instead of `psutil`'s "available" reading (useful on shared machines, or for reproducible sizing across cluster nodes with different RAM) |
| `--safety-factor F` | Multiplier applied to the raw per-iteration size estimate (default `4.0`) to account for parsing overhead and the several intermediate arrays the dedup step holds at once. Raise it if you still see OOM. |
| `--ram-margin M` | Fraction of the RAM budget the sizer targets (default `0.8`) |

## Duplicate-node matching (`--coord-tol`)

Blocks share nodes along their partition interfaces (this is expected with
METIS-style graph decomposition — the interface can be an irregular,
non-planar set of faces, not a clean spatial split). Shared nodes are
detected by exact coordinate matching rather than nearest-neighbor search
— see the printed line during a run:

```
[iter 5231932]   dedup (21,321,401 vs 21,298,554 nodes, hash-based): 1.34s, found 187,442 duplicates
```

`--coord-tol` (default `1e-6`) controls how close two coordinates must be to
be treated as the same physical node.

- **Too tight** (e.g. `1e-9` on a domain of scale ~1): falls back to a
  slower matching path internally. If your domain is small in absolute
  units, loosen this rather than tightening it.
- **Too loose**: risks merging two genuinely distinct nearby nodes into one.

If the printed duplicate count looks suspiciously low (near zero) or
suspiciously high (a large fraction of every block), that's the first thing
to check — try a different `--coord-tol` and compare.

## Other options

| Flag | Default | Applies to | Purpose |
|---|---|---|---|
| `--pattern` | `tecplot_blk*_iter*.dat` | both | Glob pattern for input files, if your naming differs |
| `--node-dtype` | `float32` | both | Use `float64` if you need full double precision (roughly doubles memory use) |
| `--conn-dtype` | `int32` | both | Only relevant with `--format dat`, or always for `run_assemble_full.py` |

## Full examples

```bash
# Point cloud, single variable, for correlation analysis with .dat output:
python run_extract_variable.py \
    --input-dir /data/praveen/nob_rbc_air_3d/dati \
    --output-dir /data/praveen/nob_rbc_air_3d/assembled_T \
    --variable T \
    --coord-tol 1e-6 \
    --workers 12 \
    --format dat

# Full mesh, all variables, for visualization:
python run_assemble_full.py \
    --input-dir /data/praveen/nob_rbc_air_3d/dati \
    --output-dir /data/praveen/nob_rbc_air_3d/assembled \
    --format npz \
    --coord-tol 1e-6 \
    --workers 4
```
