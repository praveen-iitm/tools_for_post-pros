# tools_for_post-pros

Post-processing tools for CFD/DNS output from Rayleigh–Bénard convection
(RBC) simulations. Two independent, self-contained tool sets, one per
solver's native output format:

| Tool set | Input format | Use it for |
|---|---|---|
| [`nob-post-esh/`](nob-post-esh/) | Tecplot `.dat` (domain-decomposed, per METIS partition) | Reassembling partitioned Tecplot output into merged point clouds or full meshes |
| [`plt3d_tools/`](plt3d_tools/) | Plot3D Fortran unformatted binary | Full statistics pipeline: time-means, TKE, two-point correlations, budget terms |

Each has its own README with full usage details — this file is a map of
the repo and a quick-start pointer to the right tool.

## Which tool do I need?

- **My solver writes Tecplot `.dat` files, one per partition per
  iteration, and I need to reassemble them** → `nob-post-esh/`. If you
  only need field values at point locations (e.g. for two-point spatial
  correlations), use `run_extract_variable.py`; if you need the actual
  mesh connectivity (visualization, gradients via the FE mesh), use
  `run_assemble_full.py`. See [`nob-post-esh/README.md`](nob-post-esh/README.md).

- **My solver writes Plot3D binaries (`grid.xyz` + `grid<block>.<time>.f`)
  and I need statistics computed over the whole time series** →
  `plt3d_tools/`. This is a four-stage pipeline (mean fields → TKE →
  two-point correlations → budget terms), meant to be run as sequential
  cluster jobs. See [`plt3d_tools/README.md`](plt3d_tools/README.md).

## Repository layout

```
tools_for_post-pros/
├── nob-post-esh/               Tecplot domain-decomposition reassembly
│   ├── README.md
│   ├── nob_tools/               shared library (discovery, io, dedup, writers, assemble)
│   ├── run_extract_variable.py  single-variable point cloud (lightweight)
│   └── run_assemble_full.py     full multi-variable mesh + connectivity
└── plt3d_tools/                 Plot3D RBC statistics pipeline
    ├── README.md
    ├── plt_3d_tools/             shared library (io, gradients, plotting, parallel)
    └── scripts/                  01_compute_mean → 02_compute_fluctuations_tke →
                                   03_two_point_correlation → 04_budget_terms
```

## Requirements

- `nob-post-esh/`: `numpy`, `pandas`, `psutil` (optional, for auto worker sizing)
- `plt3d_tools/`: `numpy` (`matplotlib` only if using the interactive
  `plt_3d_tools.plotting` helpers)

Both tool sets are designed for cluster use — process-pool parallelism
with RAM-aware or explicit `--workers` sizing, resumable runs (existing
outputs are skipped), and no dependency on a shared in-memory state
between stages or scripts.

Each subfolder's own README documents input file formats, all
command-line flags, output formats, and troubleshooting notes specific
to that tool.
