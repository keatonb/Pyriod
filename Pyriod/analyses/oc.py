"""O-C analysis for Pyriod time series."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightkurve import LightCurve, LightCurveCollection

from Pyriod.core import Prewhitener
from .utils import sliding_window_segments


def oc_analysis(
    source,
    frequency,
    window_width=None,
    step_size=None,
    *,
    fit_amplitude=False,
    min_points=3,
    amp_unit=None,
    freq_unit=None,
    use_weights=None,
    rescale_covar=None,
    ls_method=None,
    progress_callback=None,
    cancel_check=None,
):
    """Measure O-C variations at a fixed frequency.

    Parameters
    ----------
    source : Prewhitener, list of LightCurve, tuple of LightCurve, or LightCurveCollection
        Input data for the analysis. If a `Prewhitener` is supplied, its
        included light curve is divided into sliding windows using
        `window_width` and `step_size`. If a sequence of `LightCurve` objects
        or a `LightCurveCollection` is supplied, those light curves are treated
        directly as the analysis segments.
    frequency : float
        Frequency to use for the O-C analysis, in the selected frequency unit.
    window_width : float or None, optional
        Width of each sliding time window, in days. Required when `source` is
        a `Prewhitener`; ignored for pre-segmented light-curve input.
    step_size : float or None, optional
        Separation between successive sliding-window starts, in days. Required
        when `source` is a `Prewhitener`; ignored for pre-segmented input.
    fit_amplitude : bool, optional
        If True, fit amplitude independently in each segment. If False, keep
        the amplitude fixed to the full-data reference fit.
    min_points : int, optional
        Minimum number of points required for an analysis segment.
    amp_unit, freq_unit, use_weights, rescale_covar, ls_method : optional
        `Prewhitener` settings used when `source` is a list/tuple of
        `LightCurve` objects or a `LightCurveCollection`. Any option left as
        None uses the normal `Prewhitener` default. When `source` is already a
        `Prewhitener`, its settings are inherited instead.
    progress_callback : callable or None, optional
        Called as ``progress_callback(completed, total)``.
    cancel_check : callable or None, optional
        Callable returning True when calculation should stop.

    Returns
    -------
    pandas.DataFrame
        O-C measurements for the completed segments.
    """
    frequency = float(frequency)

    if frequency <= 0:
        raise ValueError("frequency must be positive.")
    if min_points < 3:
        raise ValueError("min_points must be at least 3.")

    # --------------------------------------------------------------
    # Convert supported input forms into:
    #   reference_lc : all data used for the reference fit
    #   segments     : individual light curves to fit
    #   pw_kwargs    : settings for temporary Prewhiteners
    # --------------------------------------------------------------

    if isinstance(source, Prewhitener):
        if window_width is None or step_size is None:
            raise ValueError(
                "window_width and step_size are required when source "
                "is a Prewhitener."
            )
        if window_width <= 0:
            raise ValueError("window_width must be positive.")
        if step_size <= 0:
            raise ValueError("step_size must be positive.")

        included = np.asarray(source.lc["include"], dtype=bool)
        reference_lc = source.lc[included].copy()

        segments = sliding_window_segments(
            reference_lc,
            window_width,
            step_size,
            min_points=min_points,
        )

        pw_kwargs = {
            "amp_unit": source.amp_unit,
            "freq_unit": source._freq_label,
            "use_weights": source.use_weights,
            "rescale_covar": source.rescale_covar,
            "ls_method": source.ls_method,
        }

    elif isinstance(source, LightCurveCollection):
        supplied_segments = list(source)

        if not supplied_segments:
            raise ValueError("LightCurveCollection is empty.")

        reference_lc = LightCurveCollection(
            [lc.copy() for lc in supplied_segments]
        ).stitch(corrector_func=None)

        segments = [
            lc.copy()
            for lc in supplied_segments
            if len(lc) >= min_points
        ]

        pw_kwargs = {}

    elif isinstance(source, (list, tuple)):
        if not source:
            raise ValueError("LightCurve sequence is empty.")

        if not all(isinstance(lc, LightCurve) for lc in source):
            raise TypeError(
                "Every item in source must be a LightCurve."
            )

        supplied_segments = list(source)

        reference_lc = LightCurveCollection(
            [lc.copy() for lc in supplied_segments]
        ).stitch(corrector_func=None)

        segments = [
            lc.copy()
            for lc in supplied_segments
            if len(lc) >= min_points
        ]

        pw_kwargs = {}

    else:
        raise TypeError(
            "source must be a Prewhitener, LightCurveCollection, "
            "or list/tuple of LightCurve objects."
        )

    if not segments:
        raise ValueError(
            "No analysis segments contain at least min_points samples."
        )

    # For pre-segmented input, only pass explicitly supplied settings.
    if not isinstance(source, Prewhitener):
        optional_settings = {
            "amp_unit": amp_unit,
            "freq_unit": freq_unit,
            "use_weights": use_weights,
            "rescale_covar": rescale_covar,
            "ls_method": ls_method,
        }
        pw_kwargs.update(
            {
                key: value
                for key, value in optional_settings.items()
                if value is not None
            }
        )

    # --------------------------------------------------------------
    # Full-data reference fit at the requested fixed frequency
    # --------------------------------------------------------------

    reference_pw = Prewhitener(
        reference_lc,
        **pw_kwargs,
    )

    try:
        reference_pw.add_signal(
            frequency,
            fixfreq=True,
            fixamp=False,
            fixphase=False,
            brute=True,
            index="f0",
        )
        reference_pw.fit_model()

        reference = reference_pw.solution_table(
            display_units=True,
            include_brute=False,
        ).loc["f0"]

        amplitude = float(reference["amp"])
        reference_phase = float(reference["phase"])

        # Use the actual settings resolved by the reference Prewhitener for
        # all child fits and for the phase-to-time conversion below.
        child_kwargs = {
            "amp_unit": reference_pw.amp_unit,
            "freq_unit": reference_pw._freq_label,
            "use_weights": reference_pw.use_weights,
            "rescale_covar": reference_pw.rescale_covar,
            "ls_method": reference_pw.ls_method,
        }
        freq_conversion = reference_pw.freq_conversion

    finally:
        close = getattr(reference_pw, "close", None)
        if close is not None:
            close()

    # --------------------------------------------------------------
    # Fit each segment
    # --------------------------------------------------------------

    total = len(segments)
    rows = []

    for i, segment in enumerate(segments):
        if cancel_check is not None and cancel_check():
            break

        child = Prewhitener(
            segment,
            **child_kwargs,
        )

        try:
            child.add_signal(
                frequency,
                amp=amplitude,
                phase=reference_phase,
                fixfreq=True,
                fixamp=not fit_amplitude,
                fixphase=False,
                brute=True,
                index="f0",
            )
            child.fit_model()

            fit = child.solution_table(
                display_units=True,
                include_brute=False,
            ).loc["f0"]

            rows.append({
                "time": float(np.mean(segment.time.value)),
                "npoints": len(segment),
                "phase": float(fit["phase"]),
                "phaseerr": (
                    np.nan
                    if pd.isna(fit["phaseerr"])
                    else float(fit["phaseerr"])
                ),
                "amp": float(fit["amp"]),
                "amperr": (
                    np.nan
                    if pd.isna(fit["amperr"])
                    else float(fit["amperr"])
                ),
            })
        finally:
            close = getattr(child, "close", None)
            if close is not None:
                close()

        if progress_callback is not None:
            progress_callback(i + 1, total)

    result = pd.DataFrame(rows)

    columns = [
        "time", "npoints", "phase", "phaseerr", "phase_offset",
        "oc_days", "oc_seconds", "ocerr_days", "ocerr_seconds",
        "amp", "amperr",
    ]

    if result.empty:
        return pd.DataFrame(columns=columns)

    result = result.sort_values("time").reset_index(drop=True)

    # Anchor the first measurement to the closest equivalent phase of the
    # reference fit, then track continuous phase evolution from segment to
    # segment. This permits cumulative changes larger than half a cycle so
    # long as adjacent segments differ by less than half a cycle.
    phases = result["phase"].to_numpy()

    initial_offset = (
        (phases[0] - reference_phase + 0.5)
        % 1.0
        - 0.5
    )

    unwrapped_offsets = (
        np.unwrap(
            2.0 * np.pi * (phases - phases[0]),
            discont=np.pi,
        )
        / (2.0 * np.pi)
    )

    result["phase_offset"] = initial_offset + unwrapped_offsets

    cycles_per_day = frequency * freq_conversion

    result["oc_days"] = -result["phase_offset"] / cycles_per_day
    result["oc_seconds"] = result["oc_days"] * 86400.0
    result["ocerr_days"] = result["phaseerr"] / cycles_per_day
    result["ocerr_seconds"] = result["ocerr_days"] * 86400.0

    result.attrs.update({
        "frequency": frequency,
        "reference_phase": reference_phase,
        "reference_amplitude": amplitude,
        "fit_amplitude": bool(fit_amplitude),
        "window_width": (
            None if window_width is None else float(window_width)
        ),
        "step_size": (
            None if step_size is None else float(step_size)
        ),
        "input_type": type(source).__name__,
    })

    return result

