# Pyriod/plotsupport.py

import numpy as np


def visible_range_indices(x, xmin, xmax):
    """Return index limits for the visible range of a sorted x array."""

    x = np.asarray(x)

    xmin, xmax = sorted((xmin, xmax))

    lo = np.searchsorted(x, xmin, side="left")
    hi = np.searchsorted(x, xmax, side="right")

    lo = max(lo, 0)
    hi = min(hi, x.size)

    return lo, hi


def minmax_decimate(x, y, max_points=30_000):
    """Return a min/max decimated version of y(x).

    This is useful for periodograms because narrow peaks are preserved better
    than with simple stride-based downsampling.
    """

    x = np.asarray(x)
    y = np.asarray(y)

    if x.size == 0:
        return np.array([]), np.array([])

    if y.size == 0:
        return np.array([]), np.array([])

    if x.size != y.size:
        raise ValueError("x and y must have the same length.")

    max_points = int(max_points)

    if max_points < 2:
        max_points = 2

    if x.size <= max_points:
        return x, y

    nbins = max(1, max_points // 2)
    edges = np.linspace(0, x.size, nbins + 1, dtype=int)

    xo = []
    yo = []

    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue

        yy = y[lo:hi]

        if yy.size == 0 or np.all(~np.isfinite(yy)):
            continue

        imin = lo + np.nanargmin(yy)
        imax = lo + np.nanargmax(yy)

        for ind in sorted((imin, imax)):
            xo.append(x[ind])
            yo.append(y[ind])

    return np.asarray(xo), np.asarray(yo)


def decimate_visible_range(x, y, xmin, xmax, max_points=30_000):
    """Slice y(x) to the visible x range, then min/max decimate it."""

    x = np.asarray(x)
    y = np.asarray(y)

    lo, hi = visible_range_indices(x, xmin, xmax)

    if hi <= lo:
        return np.array([]), np.array([])

    # Plot just beyond the axis borders
    if lo > 0:
        lo -= 1
    if hi < len(x)-1:
        hi += 1

    return minmax_decimate(
        x[lo:hi],
        y[lo:hi],
        max_points=max_points,
    )