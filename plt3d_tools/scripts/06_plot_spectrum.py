#!/usr/bin/env python3
"""
06_plot_spectrum.py
-----------------------
Plot the spectrum/spectra written by 05_energy_spectrum.py and save as a
PNG. Headless by default (uses the 'Agg' matplotlib backend, so it works
on a cluster login node or a job with no display).

Reads `k.npy` and every `P_<field>.npy` (plus `E_total.npy`, if present)
from a `post_proc_5-spectrum[_<x>_<y>]/` directory, and draws them all as
one log-log plot: k=0 (the mean/DC bin) is dropped automatically, since
it isn't representable on a log axis and isn't part of the fluctuation
spectrum anyway.

If `--slope` is given, a dashed reference line of that log-log slope is
drawn next to whichever *sustained, well-fit* stretch of the plotted
spectrum (scanning `E_total`, or the first plotted field if `E_total`
isn't present) has a windowed least-squares slope closest to it -- not
at an arbitrary fixed position, not just wherever the curve momentarily
passes through that slope value (e.g. while falling off into a noise
floor), and restricted to before the curve's steepest point, since a
real inertial range can only occur before the dissipation-range roll-off
and never in the flattening-back-out-of-noise tail after it.

Example
-------
    # Plot everything script 05 wrote (plane-avg run)
    python 06_plot_spectrum.py \
        --spectrum-dir /path/to/post_proc_5-spectrum \
        --output       /path/to/post_proc_5-spectrum/spectrum.png

    # Point-mode run, only u and w, with a -5/3 reference slope
    python 06_plot_spectrum.py \
        --spectrum-dir /path/to/post_proc_5-spectrum_5_25 \
        --fields u w \
        --slope -1.6667 \
        --output /path/to/spectrum_5_25.png
"""
import argparse
import glob
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')  # headless -- no DISPLAY required
import matplotlib.pyplot as plt
import numpy as np


def _discover_fields(spectrum_dir):
    """Field names for every P_<field>.npy present in spectrum_dir, in a
    fixed, readable order (u, v, w, T first, then anything else found)."""
    preferred = ['u', 'v', 'w', 'T']
    found = set()
    for fn in glob.glob(os.path.join(spectrum_dir, 'P_*.npy')):
        m = re.match(r'P_(.+)\.npy$', os.path.basename(fn))
        if m:
            found.add(m.group(1))
    ordered = [f for f in preferred if f in found]
    ordered += sorted(found - set(preferred))
    return ordered


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--spectrum-dir', required=True,
                     help="Output directory from script 05, e.g. .../post_proc_5-spectrum "
                          "or .../post_proc_5-spectrum_<x>_<y>")
    ap.add_argument('--output', default=None,
                     help="PNG path to save to. Default: <spectrum-dir>/spectrum.png")
    ap.add_argument('--fields', nargs='+', default=None,
                     help="Which P_<field>.npy to plot (e.g. u v w). Default: every P_*.npy found.")
    ap.add_argument('--no-total', action='store_true',
                     help="Don't plot E_total.npy even if it's present.")
    ap.add_argument('--slope', type=float, default=None,
                     help="Draw a reference line of this slope (e.g. -1.6667 for -5/3) "
                          "in log-log space, positioned where the plotted spectrum's own "
                          "local slope is closest to this value.")
    ap.add_argument('--title', default=None, help="Plot title. Default: derived from --spectrum-dir.")
    ap.add_argument('--dpi', type=int, default=150)
    args = ap.parse_args()

    k_path = os.path.join(args.spectrum_dir, 'k.npy')
    if not os.path.isfile(k_path):
        sys.exit(f"No k.npy found in {args.spectrum_dir} -- is this a script-05 output directory?")
    k = np.load(k_path)

    fields = args.fields if args.fields is not None else _discover_fields(args.spectrum_dir)
    if not fields:
        sys.exit(f"No P_<field>.npy files found in {args.spectrum_dir}")

    # Drop k=0 (DC/mean bin): not representable on a log axis, and not
    # part of the fluctuation spectrum.
    mask = k > 0
    k_plot = k[mask]

    fig, ax = plt.subplots(figsize=(7, 5))
    first_curve = None
    total_curve = None
    for f in fields:
        p_path = os.path.join(args.spectrum_dir, f"P_{f}.npy")
        if not os.path.isfile(p_path):
            print(f"WARNING: {p_path} not found, skipping field '{f}'")
            continue
        P = np.load(p_path)[mask]
        valid = P > 0  # log-scale: drop any non-positive samples (shouldn't normally occur)
        ax.loglog(k_plot[valid], P[valid], marker='o', markersize=3, linewidth=1.2, label=f"P_{f}")
        if first_curve is None:
            first_curve = (k_plot[valid], P[valid])

    total_path = os.path.join(args.spectrum_dir, 'E_total.npy')
    if not args.no_total and os.path.isfile(total_path):
        E = np.load(total_path)[mask]
        valid = E > 0
        ax.loglog(k_plot[valid], E[valid], color='k', linewidth=2.0, label="E_total")
        total_curve = (k_plot[valid], E[valid])
        if first_curve is None:
            first_curve = total_curve

    if args.slope is not None:
        # Prefer E_total as the reference curve (it's the combined spectrum);
        # fall back to whichever field curve was plotted first.
        ref_curve = total_curve if total_curve is not None else first_curve
        if ref_curve is None:
            sys.exit("No curves were plotted, so a reference --slope line has nothing to anchor to.")
        kc, pc = ref_curve
        if len(kc) < 4:
            sys.exit("Not enough points in the reference curve to locate a --slope match.")

        log_k = np.log(kc)
        log_p = np.log(pc)
        n = len(kc)
        w = max(4, n // 8)
        w = min(w, n)
        n_windows = n - w + 1

        # Windowed least-squares slope + R^2 (goodness of straight-line
        # fit in log-log space) for every window of width w. This
        # replaces two weaker point-wise estimates from before:
        #  - a single noisy sample can no longer masquerade as "the
        #    steepest point" -- it's now a fit over w points, so one bad
        #    sample gets outvoted by its neighbours;
        #  - R^2 gives a direct, principled test for "this window
        #    actually looks like a power law", instead of only checking
        #    that its average slope happens to land near --slope.
        slopes = np.empty(n_windows)
        r2s = np.empty(n_windows)
        for i in range(n_windows):
            xk = log_k[i:i + w]
            yp = log_p[i:i + w]
            A = np.vstack([xk, np.ones_like(xk)]).T
            coef, *_ = np.linalg.lstsq(A, yp, rcond=None)
            resid = yp - A @ coef
            ss_res = float(np.sum(resid ** 2))
            ss_tot = float(np.sum((yp - yp.mean()) ** 2))
            slopes[i] = coef[0]
            r2s[i] = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

        # A real spectrum's slope typically sweeps smoothly and
        # CONTINUOUSLY from ~0 (low-k plateau) down through --slope,
        # further down to its steepest point (the true dissipation-range
        # roll-off), then back UP through --slope again as it flattens
        # into the estimation/noise floor at high k. That second crossing
        # is a recovery-into-noise artifact, not physics, and -- because
        # the curve is smooth there too -- it can fit a straight line
        # just as well locally as the real inertial range does, so R^2
        # alone can't tell them apart. A genuine inertial range can only
        # exist BEFORE the steepest point (the dissipation range hasn't
        # taken over yet); the flattening-back-out tail after it never
        # counts. So restrict the search to windows up to and including
        # the steepest (most negative regression slope) one.
        steepest_window = int(np.argmin(slopes))
        n_eligible = steepest_window + 1

        # Among those pre-steepest-point windows, prefer ones that are
        # well-fit by a straight line (a real power law), then pick
        # whichever is closest to --slope; fall back to plain
        # closest-slope if nothing clears the R^2 bar (e.g. a very short
        # or unusually noisy curve).
        R2_THRESHOLD = 0.9
        eligible = np.arange(n_eligible)
        well_fit = eligible[r2s[eligible] >= R2_THRESHOLD]
        candidates = well_fit if len(well_fit) > 0 else eligible
        start = int(candidates[np.argmin(np.abs(slopes[candidates] - args.slope))])
        i_anchor = start + w // 2
        matched_slope, matched_r2 = slopes[start], r2s[start]

        # Draw the guide line over a WIDER span than the matching window,
        # centered on the same anchor point, so it's easier to see against
        # the data -- kept independent of `w` on purpose: widening the
        # matching window itself (instead of just the drawn span) dilutes
        # the fit across unrelated parts of the curve and breaks the
        # matching logic above.
        draw_half = len(kc) // 3
        lo = max(3, i_anchor - draw_half)
        hi = min(len(kc), i_anchor + draw_half)

        k_ref = kc[lo:hi]
        k_anchor, p_anchor = kc[i_anchor], pc[i_anchor] * 1  # shift down slightly so it's visible next to the data
        p_ref = p_anchor * (k_ref / k_anchor) ** args.slope
        ax.loglog(k_ref, p_ref, 'k--', linewidth=1.2,
                  label=f"k^{args.slope:.3g}  (near k≈{k_anchor:.2g})")

    ax.set_xlabel("k  (rad / length unit)")
    ax.set_ylabel("P(k)")
    title = args.title
    if title is None:
        base = os.path.basename(os.path.normpath(args.spectrum_dir))
        title = f"Energy spectrum ({base})"
    ax.set_title(title)
    ax.grid(True, which='both', linestyle=':', linewidth=0.5)
    ax.legend()
    fig.tight_layout()

    output = args.output or os.path.join(args.spectrum_dir, 'spectrum.png')
    fig.savefig(output, dpi=args.dpi)
    plt.close(fig)
    print(f"Wrote {output}")


if __name__ == '__main__':
    main()
