# %%
import os
import numpy as np
import matplotlib.pyplot as plt

from nob_tools import (
    ALL_VAR_NAMES,
    discover_iterations,
    time_average_field,
    horizontal_profile,
    write_npy,
    write_profile_dat,
)

# %%
INPUT_DIR = "/data/praveen/nob_rbc_air_3d/dati"
OUTPUT_DIR = "/data/praveen/nob_rbc_air_3d/assembled_T"
AVG_VARIABLE = "T"            # one of ALL_VAR_NAMES
COORD_TOL = 1e-6              # tolerance for matching duplicate nodes at partition interfaces
COORD_CHECK_TOL = 1e-5        # how close X,Y,Z must be to the reference grid to accept an iteration
Z_TOL = 1e-8                  # tolerance for treating two Z coordinates as the "same" level
ITERS = None                  # None = process every iteration found; or e.g. [5231932, 5231933]

# Parallelism / RAM sizing -- None lets these auto-size from available RAM;
# see nob_tools.memory_sizing.choose_worker_count for details.
WORKERS = None
MAX_RAM_GB = None
SAFETY_FACTOR = 3.0
RAM_MARGIN = 0.8

os.makedirs(OUTPUT_DIR, exist_ok=True)

if AVG_VARIABLE not in ALL_VAR_NAMES:
    raise ValueError(f"AVG_VARIABLE must be one of: {', '.join(ALL_VAR_NAMES)} (got {AVG_VARIABLE!r})")
AVG_VAR_IDX = ALL_VAR_NAMES.index(AVG_VARIABLE)

# %%
by_iter = discover_iterations(INPUT_DIR)
if ITERS is not None:
    by_iter = {it: blocks for it, blocks in by_iter.items() if it in ITERS}

print(f"Found {len(by_iter)} iteration(s)")
for it, blocks in by_iter.items():
    print(f"  iter {it}: {len(blocks)} block(s)")

# %%
# The expensive part (reading + merging + deduplicating each iteration's
# blocks) runs in parallel worker processes, sized to available RAM. Each
# worker only returns its averaged field values -- X,Y,Z is broadcast to
# workers once (via the pool initializer), not re-sent per iteration, and
# never shipped back. See nob_tools/time_average.py for the full design
# notes, including the coordinate-consistency check that aborts immediately
# if any iteration's grid doesn't match the reference (rather than silently
# producing a corrupted average).
xyz_ref, mean_field, n_accum, n_total = time_average_field(
    by_iter, AVG_VAR_IDX,
    coord_tol=COORD_TOL, coord_check_tol=COORD_CHECK_TOL,
    workers=WORKERS, max_ram_gb=MAX_RAM_GB,
    safety_factor=SAFETY_FACTOR, ram_margin=RAM_MARGIN,
)

print(f"\nAccumulated {n_accum} of {n_total} iteration(s)")
print(f"{AVG_VARIABLE} mean range: [{mean_field.min():.6g}, {mean_field.max():.6g}]")

# %%
avg_var_name = f"{AVG_VARIABLE}_mean"
avg_data = np.column_stack([xyz_ref, mean_field])
avg_var_names = ["X", "Y", "Z", avg_var_name]

avg_out_path = os.path.join(OUTPUT_DIR, f"time_avg_{AVG_VARIABLE}_{n_accum}iters.npy")
write_npy(avg_out_path, avg_data, avg_var_names)
print(f"Wrote time-averaged {AVG_VARIABLE} over {n_accum} iterations -> {avg_out_path}")

# %%
# Horizontal (x,y) average -> vertical profile <T>(z). Groups points by Z
# (within Z_TOL) and averages within each group -- see docstring for the
# arithmetic-mean-over-points caveat if your horizontal mesh isn't uniform.
z_levels, T_profile, count_per_z = horizontal_profile(xyz_ref, mean_field, z_tol=Z_TOL)

print(f"Found {len(z_levels)} unique Z-level(s) out of {xyz_ref.shape[0]:,} total points")
print(f"  points per level: min={count_per_z.min():,}, max={count_per_z.max():,}, "
      f"mean={count_per_z.mean():.0f}")
if len(z_levels) > 0.5 * xyz_ref.shape[0]:
    print("  WARNING: number of levels is close to the total point count -- "
          "the mesh may not share exact Z-levels. Consider binning by range instead.")

# %%
plt.figure(figsize=(5, 6))
plt.plot(T_profile, z_levels, "o-", ms=3)
plt.xlabel(f"$\\langle {AVG_VARIABLE} \\rangle_{{x,y}}$")
plt.ylabel("Z")
plt.title(f"Horizontally-averaged {AVG_VARIABLE} profile ({n_accum} iterations)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %%
profile_out_path = os.path.join(OUTPUT_DIR, f"profile_{AVG_VARIABLE}_z_{n_accum}iters.dat")
write_profile_dat(profile_out_path, z_levels, T_profile, names=("Z", avg_var_name))
print(f"Wrote vertical profile ({len(z_levels)} Z-levels) -> {profile_out_path}")
