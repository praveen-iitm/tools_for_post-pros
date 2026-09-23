"""
Quick-look 2D slice plots. Only useful interactively (e.g. in a notebook
on your workstation) -- the cluster scripts do not call these, but they're
kept here so you can still import them for sanity-checking outputs.
"""
import matplotlib.pyplot as plt


def plot_slice(x, y, z, var, slice_id):
    X = x[:, :, slice_id]
    Y = y[:, :, slice_id]
    V = var[:, :, slice_id]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    cp = ax.pcolormesh(X, Y, V, shading='auto', cmap='jet')
    fig.colorbar(cp, ax=ax, label='variable')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f"Slice at z = {z[0, 0, slice_id]:.4f}")
    ax.set_aspect('equal')
    fig.tight_layout()
    plt.show()


def plot_slice_2d(x, y, var):
    X = x[:, :, 0]
    Y = y[:, :, 0]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    cp = ax.pcolormesh(X, Y, var, shading='auto', cmap='jet')
    fig.colorbar(cp, ax=ax, label='variable')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    fig.tight_layout()
    plt.show()
