# Utility functions for analysis functions.

def sliding_window_segments(
    lc,
    window_width,
    step_size,
    *,
    min_points=1,
):
    """Split a light curve into overlapping sliding-window segments.

    Parameters
    ----------
    lc : lightkurve.LightCurve
        Input light curve.
    window_width : float
        Width of each time window, in the same units as ``lc.time.value``.
    step_size : float
        Separation between successive window start times, in the same units
        as ``lc.time.value``.
    min_points : int, optional
        Minimum number of samples required for a segment to be returned.
        Default is 1.

    Returns
    -------
    segments : list of lightkurve.LightCurve
        Light-curve segments for each populated sliding window.

    Notes
    -----
    Windows are half-open intervals of the form
    ``[start, start + window_width)``. The final window is included if its
    start time is less than or equal to the maximum time in the light curve.
    """
    if window_width <= 0:
        raise ValueError("window_width must be positive.")

    if step_size <= 0:
        raise ValueError("step_size must be positive.")

    if min_points < 1:
        raise ValueError("min_points must be at least 1.")

    lc = lc.copy().remove_nans()

    if len(lc) == 0:
        return []

    time = lc.time.value

    start = np.nanmin(time)
    stop = np.nanmax(time)

    segments = []

    while start <= stop:
        end = start + window_width

        mask = (time >= start) & (time < end)

        if np.count_nonzero(mask) >= min_points:
            segments.append(lc[mask])

        start += step_size

    return segments
