"""
plt_3d_tools
============
Shared helpers used by all the cluster post-processing scripts:

- io           : reading Plot3D grid/solution binaries, listing/sorting snapshots
- gradients    : grid-aware gradient operators (grad_x, grad_y, grad_z, grad_r)
- plotting     : quick-look 2D slice plots (interactive use only)
- parallel     : chunked multiprocessing accumulation with bounded RAM
"""
