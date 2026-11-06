"""Running-periodogram analysis for Pyriod time series."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from lightkurve import LightCurve, LightCurveCollection

from Pyriod.core import Prewhitener
from .utils import sliding_window_segments


_NORMALIZATIONS = {
    "amplitude",
    "peak",
    "median",
    "global_peak",
}

_TIMESERIES_OPTIONS = {
    "original",
    "model",
    "residual",
}


def normalize_running_periodogram(amplitudes, normalization="amplitude"):
    """Normalize a running-periodogram amplitude matrix.

    Parameters
    ----------
    amplitudes : array-like
        Two-dimensional array with one periodogram per row.
    normalization : {"amplitude", "peak", "median", "global_peak"}, optional
        Normalization to apply.

        ``"amplitude"``
            Preserve the periodogram amplitudes. This is the default.
        ``"peak"``
            Divide each row by its own maximum amplitude.
        ``"median"``
            Divide each row by its own median amplitude.
        ``"global_peak"``
            Divide the entire matrix by its overall maximum amplitude.

    Returns
    -------
    numpy.ndarray
        Normalized copy of the input array.
    """
    if normalization not in _NORMALIZATIONS:
        raise ValueError(
            "normalization must be one of "
            "'amplitude', 'peak', 'median', or 'global_peak'."
        )

    values = np.asarray(amplitudes, dtype=float)

    if values.ndim != 2:
        raise ValueError("amplitudes must be a two-dimensional array.")

    if normalization == "amplitude":
        return values.copy()

    if normalization == "global_peak":
        denominator = np.nanmax(values)

        if not np.isfinite(denominator) or denominator == 0:
            return np.full_like(values, np.nan)

        return values / denominator

    if normalization == "peak":
        denominator = np.nanmax(values, axis=1)
    else:
        denominator = np.nanmedian(values, axis=1)

    denominator = denominator[:, np.newaxis]

    normalized = np.full_like(values, np.nan)

    np.divide(
        values,
        denominator,
        out=normalized,
        where=np.isfinite(denominator) & (denominator != 0),
    )

    return normalized


@dataclass
class RunningPeriodogramResult:
    """Results from a running-periodogram analysis."""

    time: np.ndarray
    frequency: np.ndarray
    amplitude: np.ndarray
    normalization: str = "amplitude"
    freq_unit: str = ""
    amp_unit: str = ""
    window_width: float | None = None
    step_size: float | None = None
    timeseries: str = "original"

    @property
    def power(self):
        """Amplitude matrix using the selected normalization."""
        return normalize_running_periodogram(
            self.amplitude,
            self.normalization,
        )


def running_periodogram(
    source,
    window_width=None,
    step_size=None,
    *,
    minimum_frequency=None,
    maximum_frequency=None,
    oversample=5,
    frequency=None,
    normalization="amplitude",
    timeseries="original",
    min_points=3,
    amp_unit=None,
    freq_unit=None,
    ls_method=None,
    progress_callback=None,
    cancel_check=None,
):
    """Compute periodograms as a function of time.

    Parameters
    ----------
    source : Prewhitener, list of LightCurve, tuple of LightCurve, or LightCurveCollection
        Input data. If a `Prewhitener` is supplied, its included light curve
        is divided into sliding windows. If a sequence of light curves or a
        `LightCurveCollection` is supplied, those objects are treated directly
        as the time segments.
    window_width : float or None, optional
        Width of sliding windows in days. Required for a `Prewhitener`.
    step_size : float or None, optional
        Separation between sliding-window starts in days. Required for a
        `Prewhitener`.
    minimum_frequency, maximum_frequency : float or None, optional
        Frequency limits in the selected frequency unit. If omitted, the
        minimum defaults to zero and the maximum to the reference
        `Prewhitener` Nyquist frequency.
    oversample : float, optional
        Frequency-grid oversampling relative to the natural resolution of a
        segment. Default is 5. Ignored if `frequency` is supplied explicitly.
    frequency : array-like or None, optional
        Explicit common frequency grid. If supplied, the same grid is used for
        every segment.
    normalization : {"amplitude", "peak", "median", "global_peak"}, optional
        Default normalization attached to the returned result. Raw amplitudes
        are always retained in `result.amplitude`. Default is `"amplitude"`.
    timeseries : {"original", "model", "residual"}, optional
        Light curve to analyze when `source` is a `Prewhitener`.
        ``"original"`` uses the observed light curve, ``"model"`` uses the
        current fitted model, and ``"residual"`` uses the observed light curve
        minus the fitted model. Model and residual selections are only
        available for `Prewhitener` input. Default is ``"original"``.
    min_points : int, optional
        Minimum number of points required in a segment. Default is 3.
    amp_unit, freq_unit, ls_method : optional
        `Prewhitener` settings for pre-segmented light-curve input. When
        `source` is a `Prewhitener`, its settings are inherited.
    progress_callback : callable or None, optional
        Called as ``progress_callback(completed, total)`` after each segment.
    cancel_check : callable or None, optional
        Callable returning True when the analysis should stop.

    Returns
    -------
    RunningPeriodogramResult
        Segment-center times, common frequency grid, raw amplitude matrix,
        and display-normalization metadata.
    """
    if normalization not in _NORMALIZATIONS:
        raise ValueError(
            "normalization must be one of "
            "'amplitude', 'peak', 'median', or 'global_peak'."
        )

    if timeseries not in _TIMESERIES_OPTIONS:
        raise ValueError(
            "timeseries must be one of "
            "'original', 'model', or 'residual'."
        )

    if min_points < 3:
        raise ValueError("min_points must be at least 3.")

    if oversample <= 0:
        raise ValueError("oversample must be positive.")

    # --------------------------------------------------------------
    # Convert supported inputs into a common segment representation.
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

        if timeseries == "original":
            selected_lc = source.lc
        elif timeseries == "model":
            selected_lc = source.lc_model
        else:
            selected_lc = source.lc_resid

        included = np.asarray(selected_lc["include"], dtype=bool)
        reference_lc = selected_lc[included].copy()

        segments = sliding_window_segments(
            reference_lc,
            window_width,
            step_size,
            min_points=min_points,
        )

        resolved_freq_unit = source.freq_unit
        resolved_amp_unit = source.amp_unit
        resolved_ls_method = source.ls_method
        freq_conversion = source.freq_conversion
        amp_conversion = source.amp_conversion
        nyquist = source.nyquist
        natural_baseline = float(window_width)

    elif isinstance(source, LightCurveCollection):
        if timeseries != "original":
            raise ValueError(
                "timeseries='model' and timeseries='residual' require "
                "a Prewhitener source."
            )

        supplied_segments = list(source)

        if not supplied_segments:
            raise ValueError("LightCurveCollection is empty.")

        if not all(isinstance(lc, LightCurve) for lc in supplied_segments):
            raise TypeError(
                "Every item in source must be a LightCurve."
            )

        segments = [
            lc.copy()
            for lc in supplied_segments
            if len(lc) >= min_points
        ]

        reference_lc = LightCurveCollection(
            [lc.copy() for lc in supplied_segments]
        ).stitch(corrector_func=None)

        settings_pw = Prewhitener(
            reference_lc,
            **{
                key: value
                for key, value in {
                    "amp_unit": amp_unit,
                    "freq_unit": freq_unit,
                    "ls_method": ls_method,
                }.items()
                if value is not None
            },
        )

        try:
            resolved_freq_unit = settings_pw.freq_unit
            resolved_amp_unit = settings_pw.amp_unit
            resolved_ls_method = settings_pw.ls_method
            freq_conversion = settings_pw.freq_conversion
            amp_conversion = settings_pw.amp_conversion
            nyquist = settings_pw.nyquist
        finally:
            close = getattr(settings_pw, "close", None)
            if close is not None:
                close()

        baselines = [
            np.ptp(lc.time.value)
            for lc in segments
            if len(lc) >= 2 and np.ptp(lc.time.value) > 0
        ]

        if not baselines:
            raise ValueError(
                "At least one segment must span a non-zero time baseline."
            )

        natural_baseline = float(np.median(baselines))

    elif isinstance(source, (list, tuple)):
        if timeseries != "original":
            raise ValueError(
                "timeseries='model' and timeseries='residual' require "
                "a Prewhitener source."
            )

        if not source:
            raise ValueError("LightCurve sequence is empty.")

        if not all(isinstance(lc, LightCurve) for lc in source):
            raise TypeError(
                "Every item in source must be a LightCurve."
            )

        return running_periodogram(
            LightCurveCollection(list(source)),
            window_width=window_width,
            step_size=step_size,
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency,
            oversample=oversample,
            frequency=frequency,
            normalization=normalization,
            timeseries=timeseries,
            min_points=min_points,
            amp_unit=amp_unit,
            freq_unit=freq_unit,
            ls_method=ls_method,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )

    else:
        raise TypeError(
            "source must be a Prewhitener, LightCurveCollection, "
            "or list/tuple of LightCurve objects."
        )

    if not segments:
        raise ValueError(
            "No analysis segments contain at least min_points samples."
        )

    # --------------------------------------------------------------
    # Build one common frequency grid for all segments.
    # --------------------------------------------------------------

    if frequency is None:
        if minimum_frequency is None:
            if isinstance(source, Prewhitener):
                minimum_frequency = float(np.nanmin(source.freqs))
            else:
                minimum_frequency = 0.0

        if maximum_frequency is None:
            if isinstance(source, Prewhitener):
                maximum_frequency = float(np.nanmax(source.freqs))
            else:
                maximum_frequency = float(nyquist)

        minimum_frequency = float(minimum_frequency)
        maximum_frequency = float(maximum_frequency)

        if maximum_frequency <= minimum_frequency:
            raise ValueError(
                "maximum_frequency must be greater than minimum_frequency."
            )

        frequency_step = (
            1.0
            / (freq_conversion * natural_baseline * oversample)
        )

        frequencies = np.arange(
            minimum_frequency,
            maximum_frequency,
            frequency_step,
            dtype=float,
        )

    else:
        frequencies = np.asarray(frequency, dtype=float)

        if frequencies.ndim != 1 or len(frequencies) < 2:
            raise ValueError(
                "frequency must be a one-dimensional array with "
                "at least two values."
            )

        if not np.all(np.isfinite(frequencies)):
            raise ValueError("frequency contains non-finite values.")

        if np.any(np.diff(frequencies) <= 0):
            raise ValueError(
                "frequency values must be strictly increasing."
            )

    if len(frequencies) < 2:
        raise ValueError(
            "The generated frequency grid contains fewer than two values."
        )

    # --------------------------------------------------------------
    # Calculate one periodogram per segment.
    # --------------------------------------------------------------

    times = []
    amplitudes = []
    total = len(segments)

    for i, segment in enumerate(segments):
        if cancel_check is not None and cancel_check():
            break

        periodogram = segment.to_periodogram(
            normalization="amplitude",
            freq_unit=resolved_freq_unit,
            frequency=frequencies,
            ls_method=resolved_ls_method,
        )

        amplitude = np.asarray(
            periodogram.power.value,
            dtype=float,
        )
        amplitude *= amp_conversion

        times.append(float(np.mean(segment.time.value)))
        amplitudes.append(amplitude)

        if progress_callback is not None:
            progress_callback(i + 1, total)

    times = np.asarray(times, dtype=float)

    if amplitudes:
        amplitude_matrix = np.vstack(amplitudes)

        order = np.argsort(times)
        times = times[order]
        amplitude_matrix = amplitude_matrix[order]
    else:
        amplitude_matrix = np.empty(
            (0, len(frequencies)),
            dtype=float,
        )

    return RunningPeriodogramResult(
        time=times,
        frequency=frequencies,
        amplitude=amplitude_matrix,
        normalization=normalization,
        freq_unit=resolved_freq_unit.to_string(),
        amp_unit=str(resolved_amp_unit),
        window_width=(
            None if window_width is None else float(window_width)
        ),
        step_size=(
            None if step_size is None else float(step_size)
        ),
        timeseries=timeseries,
    )
