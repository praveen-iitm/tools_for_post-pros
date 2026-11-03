# NOB RBC Tecplot Assembly Tools — README

Numbered scripts (run in any order, each independent) for reassembling
domain-decomposed Tecplot `.dat` output (one `.dat` file per METIS partition,
per solver iteration) into merged results and time averages, built on a
shared `nob_tools/` library. All are command-line tools with `--help`, and
run from any working directory:

| Script | Produces | Use it for |
|---|---|---|
| `scripts/01_extract_variable.py` | One variable (+X,Y,Z), no connectivity, per iteration | Point-based analysis (e.g. two-point spatial correlations) — **this is what you want for the RBC correlation work** |
| `scripts/02_assemble_full.py` | All 13 variables + full FE connectivity, per iteration | Visualization in Tecplot/ParaView, or anything that needs the actual mesh structure |
| `scripts/03_time_average.py` | Time-averaged field (`.npy` + Tecplot `.dat`), horizontally-averaged vertical profile (`.dat` + `.png`) | Mean profiles, e.g. `<T>(z)` |
| `scripts/04_nusselt.py` | Nu at the hot wall, cold wall, bulk (single values, `.npy`) and the Nu(z) profile (`.dat`) | Comparing with the NOB Nusselt-number results of Wang et al., JFM 2020 |

If you're not sure which you need: if your analysis only needs field values
at point locations and not the tetrahedral mesh connectivity, use
`scripts/01_extract_variable.py`. It's lighter on memory and time since it never
reads the other ~9-10 variable columns or the connectivity block at all.

## Setup

The scripts import `nob_tools/` from their parent folder, so keep this layout:

```
nob-post-esh/
  nob_tools/            shared library
  scripts/
    01_extract_variable.py
    02_assemble_full.py
    03_time_average.py
    04_nusselt.py
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

## `scripts/01_extract_variable.py` — single-variable point cloud

### Basic usage

```bash
python scripts/01_extract_variable.py \
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

## `scripts/02_assemble_full.py` — full multi-variable mesh

### Basic usage

```bash
python scripts/02_assemble_full.py \
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
`scripts/01_extract_variable.py`'s point-cloud output, which is just one array.

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
`scripts/01_extract_variable.py` instead, since reading+merging+storing all 13
variables and ~126M connectivity rows per iteration costs considerably more
time and memory than pulling just what you need.

---

## Selecting which iterations to process

Both scripts default to processing every iteration found in `--input-dir`.
Restrict to specific ones with `--iters`:

```bash
python scripts/01_extract_variable.py --input-dir ... --output-dir ... \
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
python scripts/01_extract_variable.py --input-dir ... --output-dir ... --variable T --workers 1
```

If `--workers 1` still fails, raise `--safety-factor` (default `4.0`) — this
is far more likely with `scripts/02_assemble_full.py`, since holding all 13
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

## `scripts/03_time_average.py` — time average + vertical profile

Averages one variable over every iteration found (blocks merged and
de-duplicated as in script 01), then reduces it to a horizontally-averaged
vertical profile. Written into `--output-dir`, named with the variable and the
number of iterations averaged (`<n>`):

| File | Written when | Contents |
|---|---|---|
| `time_avg_<VAR>_<n>iters.npy` | always | time-averaged field: X, Y, Z, `<VAR>_mean` |
| `time_avg_<VAR>_<n>iters.dat` | unless `--no-field-dat` | same field as Tecplot ASCII: a point cloud, or with `--with-connectivity` a `FETETRAHEDRON` mesh (large/slow on big meshes) |
| `profile_<VAR>_z_<n>iters.dat` | always | horizontally-averaged vertical profile |
| `profile_<VAR>_z_<n>iters.png` | unless `--no-plot` | plot of that profile (`--dpi`, default 150) |

```bash
python scripts/03_time_average.py \
    --input-dir  /data/praveen/nob_rbc_air_3d/dati \
    --output-dir /data/praveen/nob_rbc_air_3d/assembled_T \
    --variable T --workers 12
```

**Restartable and low-RAM.** Each iteration's assembled field is written to a
checkpoint folder (`--checkpoint-dir`, default `<output-dir>/checkpoint_<VAR>/`)
as soon as it is done, and nothing is accumulated in RAM while the job runs —
the final mean is a streaming sum over those files. If the job dies (walltime,
node failure), **rerun the same command**: iterations already checkpointed are
skipped and only the unfinished ones are redone. Files are written atomically,
so a half-written one is never mistaken for finished. You can also extend an
average later (rerun with more iterations; only the new ones are computed).
Disk cost is about N points × 4 bytes per iteration. `--no-resume` discards the
checkpoint and starts over; `--clean-checkpoint` deletes it after a successful
run (default: keep it).

**Connectivity.** By default the field `.dat` is a bare point cloud (no
elements). `--with-connectivity` writes it as a tetrahedral mesh you can
contour in Tecplot/ParaView. The averaging never needs connectivity, so it is
read once from the first iteration (one extra single-iteration read: more RAM
and time than the point-cloud run), checked against the reference grid's node
order, and cached in the checkpoint folder. The `.npy` is always a point cloud.

The plot is saved, not shown (headless-safe for clusters); add `--show` to also
open it interactively. An iteration whose grid doesn't match the first one
(within `--coord-check-tol`) aborts the run instead of corrupting the average.
Other flags: `--iters`, `--z-tol` (what counts as the same Z level), `--pattern`,
`--coord-tol`, `--max-ram-gb`, `--safety-factor`, `--ram-margin`.

## `scripts/04_nusselt.py` — Nusselt numbers (hot wall, cold wall, bulk, Nu(z))

Computes Nusselt numbers from the raw partition files for comparison with
Wang, Wan, Yan & Sun, JFM (2020), doi:10.1017/jfm.2020.66, who define
`Nu = Q / (k·2ε/H)` with Q the heat flux across a horizontal plane. Z is the
vertical direction; the hotter wall is detected automatically. In the solver's
units the flux on the plane at height z is

```
Q(z) = Pe · <ρ cp w (T − T_ref)>  −  <κ ∂T/∂z>        Nu = Q / (k_ref · ΔT / H)
```

(`< >` = mean over the nodes of that Z level and over iterations). The time
average is taken **on the product, per iteration** — the convective flux is
almost entirely a fluctuation correlation, so it cannot be built from
time-averaged fields. Fields read: `W, T, rho, cp, kappa`.

```bash
python scripts/04_nusselt.py \
    --input-dir  /data/praveen/nob_rbc_air_3d/dati \
    --output-dir /data/praveen/nob_rbc_air_3d/nusselt \
    --ra 1e8 --pr 0.71 --epsilon 0.1 --workers 12
```

**You must give the Péclet number** (`--pe`, or `--ra` and `--pr`), because it
depends on how your solver is nondimensionalised: `Pe = U_ref·H/κ_ref`, i.e.
`--pe 1` for thermal-diffusion velocity scaling, or `sqrt(Ra·Pr)` for
free-fall scaling. Q is the same on every plane of a statistically steady
closed cell, so **Nu(z) should be flat**: the script prints its relative spread
and warns above 5%. A wrong Pe, too few iterations, or a poor wall gradient all
show up there.

| Output | Contents |
|---|---|
| `Nu_hot.npy`, `Nu_cold.npy` | conductive flux at the hot / cold wall, normalised (no-slip, so no convective part there) |
| `Nu_bulk.npy` | volume average of Nu(z), the integral of Nu dz over `--bulk-range` (fractions of H, default the whole height; e.g. `0.1 0.9` drops the wall layers) |
| `Nu_profile.dat` | Tecplot ASCII, columns `Z, Nu, Nu_conv, Nu_cond, T_mean` |
| `N_BL_bottom.npy`, `N_BL_top.npy` | number of grid points (Z levels) inside the bottom / top thermal boundary layer — the paper's `N^b_BL`, `N^t_BL` (int, 0-d); `BL_thickness_bottom.npy`, `BL_thickness_top.npy` hold the thicknesses |
| `Nu_V.npy` | (with `--nu-v`) `Nu_V = √(RaPr)<ε_u> + 1` from the dissipation rate, as in the paper's Table 4 |
| `Nu_series.dat` | per-snapshot `Nu_hot, Nu_cold, Nu_bulk` against iteration number |
| `Nu_summary.txt`, `Nu_profile.png` | all numbers, diagnostics and settings used; plot of Nu(z) and its parts next to the per-snapshot series |

**Thermal boundary layers.** Always computed, from the time- and plane-averaged
`<T>(z)` by the *slope method*: δ is the distance from the plate at which the
tangent of `<T>(z)` at the plate reaches the centre temperature `T_c` (`<T>` at
mid-height); `N_BL` is the number of Z levels within δ of the plate. They are the
bottom and top plates, not hot/cold, as in the paper. The plate level itself is
counted by default (`--bl-exclude-wall` leaves it out); the summary prints both
counts, so check which one matches the paper's convention (the paper does not
say). This follows Wan et al. §3.4.2, which defines the thermal BL thickness "using
the centre temperature T_c" in the same way as the viscous one (extrapolated linear
profile meeting the line through the edge value), and §3.1.2 / Fig. 6, which use
time- and plane-averaged temperatures with `T_c` taken at z/H = 0.5.

**`Nu_V` (`--nu-v`).** With velocities in free-fall units,
`Nu_V = 1 + Pr·<Φ>_{V,t}`, with `Φ = μ[∂ᵢuⱼ(∂ᵢuⱼ + ∂ⱼuᵢ) − ⅔(∇·u)²]` the viscous
dissipation per unit volume (μ in units of its reference value; `Ra` cancels out
of `√(RaPr)·√(Pr/Ra)`). It needs `--pr`, reads `U, V, mu` as well as the
connectivity (more RAM and time), and uses per-tetrahedron gradients weighted by
element volume. An existing checkpoint made without `--nu-v` has to be
recomputed (`--no-resume`). **The exact relation behind Nu_V holds for
Oberbeck–Boussinesq flow only: the paper lists Nu_V for its OB cases and none
for ε > 0**, so for a NOB run this is only a dissipation-based estimate.

**Checked against the paper (Wan et al. 2020).** Their low-Mach equations (2.1)–(2.4)
use the free-fall velocity `u0 = (2ε ĝ Ĥ)^½`, `ρ cp (T_t + u·∇T) = (1/√(RaPr)) ∇·(k ∇T)
+ Γ dp/dt`, `cp = 1`, `p = ρT`, `k, μ` from Sutherland's law, `ΔT = 2ε`, so
`Pe = √(RaPr)`, `k_ref = 1` and `Nu = Q/(k·2ε/H)` are exactly what this script
assumes. **Their Nu is the one at the bottom (hot) plate** (Tables 3–4), i.e.
`Nu_hot`. Reference values, 3-D, Pr = 0.71 (Table 4):

| Ra | case | Nu (bottom) | Nu_V | N^b_BL | N^t_BL | Re |
|---|---|---|---|---|---|---|
| 1e8 | OB | 31.29 ± 0.02 | 31.39 | 28 | 28 | 2004.1 |
| 1e8 | ε = 0.1 | 31.24 ± 0.08 | — | 29 | 26 | 2004.7 |
| 1e8 | ε = 0.2 | 31.12 ± 0.03 | — | 31 | 25 | 2005.6 |
| 1e8 | ε = 0.4 | 31.00 ± 0.03 | — | 33 | 22 | 1989.6 |

(grid 384×384×392, averaged over 400 free-fall times after the transient has
died out, "at least 10 grid points inside the thermal BLs".) The pressure term
matters for stationarity: the thermodynamic pressure obeys
`p ∇·u = (1/√(RaPr)) ∇·(k ∇T) − (1−Γ) dp/dt`, so any drift of `p` while you average
makes `Nu_cold ≠ Nu_hot` and adds a term linear in z to Nu(z).

**When Nu(z) is not flat — the diagnostics.** In a statistically steady closed
cell `Nu_hot = Nu_cold = Nu(z)`. The summary reports, to tell the causes apart:

| Diagnostic | What it says |
|---|---|
| hot/cold imbalance, and drift of `Nu_hot/Nu_cold/Nu_bulk` over the snapshots (`Nu_series.dat`) | a trend or imbalance means the averaged snapshots include a transient (or are too few); average only later iterations with `--iters` |
| Pe self-check: best-fit `Pe_eff / Pe_used` | `≈ 1`: the velocity scaling is consistent; far from 1: `Pe` is off. Also gives the flux that a flat profile would have |
| mass-flux term `max|Pe·T_ref·<ρ cp w>|` (column `Nu_Tref_shift`) | how much Nu(z) would change if `T_ref` were 0; large means `<ρ w>` is not ~0 (non-stationary, or the solver's continuity/energy formulation differs from the assumed conservative one) |
| horizontal spacing max/min | the horizontal mean is a plain node mean; it is an area average only if this is ~1 |
| s.e.m. over snapshots | assumes independent snapshots, so it is optimistic for closely spaced ones |

Compare `Nu_hot` (not `Nu_bulk`) with a paper's bottom-plate Nusselt number;
`Nu_bulk` is only meaningful once Nu(z) is flat.

The `.npy` single values are 0-d arrays: `float(np.load("Nu_hot.npy"))`.

**Normalisation.** `ΔT` = |T(bottom wall) − T(top wall)| from the data, or
`--delta-T`, or `--epsilon` (ΔT = 2ε when T is scaled by T0); `H` = Z extent
(`--H`); `--k-ref` = reference conductivity in solver units (default 1).

**Gradient dT/dz** (`--mode auto|columns|fe`). `columns`: the mesh has shared Z
levels and the same (x,y) points on every level; second-order non-uniform finite
differences along each column, including one-sided stencils at the walls.
`fe`: per-tetrahedron gradients from the connectivity, volume-averaged onto
nodes; works on any mesh, but the wall gradient is only first-order accurate.
`auto` (default) uses `columns` if the mesh allows it. Both need shared Z levels
for the horizontal averaging, and use plain arithmetic means over nodes (see the
caveat on non-uniform horizontal meshes in the time-average section).

Like script 03 it is **restartable and low-RAM**: each iteration is reduced to a
few small arrays in `<output-dir>/checkpoint_Nu/` as soon as it is done, and a
rerun of the same command skips finished iterations (`--no-resume`,
`--clean-checkpoint`, `--checkpoint-dir`). Files are read in float64 by default
(`--node-dtype float32` halves memory) because gradients over closely spaced
wall levels amplify float32 round-off.

## Other options

| Flag | Default | Applies to | Purpose |
|---|---|---|---|
| `--pattern` | `tecplot_blk*_iter*.dat` | both | Glob pattern for input files, if your naming differs |
| `--node-dtype` | `float32` | both | Use `float64` if you need full double precision (roughly doubles memory use) |
| `--conn-dtype` | `int32` | both | Only relevant with `--format dat`, or always for `scripts/02_assemble_full.py` |

## Full examples

```bash
# Point cloud, single variable, for correlation analysis with .dat output:
python scripts/01_extract_variable.py \
    --input-dir /data/praveen/nob_rbc_air_3d/dati \
    --output-dir /data/praveen/nob_rbc_air_3d/assembled_T \
    --variable T \
    --coord-tol 1e-6 \
    --workers 12 \
    --format dat

# Full mesh, all variables, for visualization:
python scripts/02_assemble_full.py \
    --input-dir /data/praveen/nob_rbc_air_3d/dati \
    --output-dir /data/praveen/nob_rbc_air_3d/assembled \
    --format npz \
    --coord-tol 1e-6 \
    --workers 4
```
