"""
Grid-aware gradient operators. The original notebooks computed grad_x/y/z/r
as closures over module-level x, y, z, r_x, r_y, r_z globals; GradientOps
bundles the same math into an object so it can be constructed explicitly
(and safely passed to worker processes) instead of relying on globals.
"""
import numpy as np


class GradientOps:
    """grad_x/grad_y/grad_z (1D non-uniform gradient along each grid axis)
    and grad_r (radial derivative towards a fixed reference point), bound to
    one grid and one reference index (x_idx, y_idx, z_idx)."""

    def __init__(self, x, y, z, ref_idx, ep=1e-12):
        self.x1d = x[:, 1, 1]
        self.y1d = y[1, :, 1]
        self.z1d = z[1, 1, :]

        xi, yi, zi = ref_idx
        self.r_x = x - x[xi, yi, zi]
        self.r_y = y - y[xi, yi, zi]
        self.r_z = z - z[xi, yi, zi]
        self.r = np.sqrt(self.r_x ** 2 + self.r_y ** 2 + self.r_z ** 2)
        self.ep = ep

        # 3D "Green's function" style coefficient used throughout post-proc-4
        self.co_eff = 1.0 / (4 * np.pi * (self.r + ep) * (self.r + ep))

    def grad_x(self, var):
        return np.gradient(var, self.x1d, axis=0, edge_order=2)

    def grad_y(self, var):
        return np.gradient(var, self.y1d, axis=1, edge_order=2)

    def grad_z(self, var):
        return np.gradient(var, self.z1d, axis=2, edge_order=2)

    def grad_r(self, var):
        grad_rx = self.grad_x(var)
        grad_ry = self.grad_y(var)
        grad_rz = self.grad_z(var)

        rx_r = self.r_x / (self.r + self.ep)
        ry_r = self.r_y / (self.r + self.ep)
        rz_r = self.r_z / (self.r + self.ep)

        out = rx_r * grad_rx + ry_r * grad_ry + rz_r * grad_rz
        del grad_rx, grad_ry, grad_rz, rx_r, ry_r, rz_r
        return out
