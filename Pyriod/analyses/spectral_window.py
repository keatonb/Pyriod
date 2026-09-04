"""Spectral-window analysis for Pyriod time series."""

import numpy as np

def spectral_window(
    pw,
    maxfreq=100,
    oversample=10,
    log=False,
    progress_callback=None,
    cancel_check=None,
):
    """Calculate the spectral window of a Pyriod time series.

    Parameters
    ----------
    pw : Prewhitener
        Analysis object providing the observation times, inclusion mask,
        and frequency units.
    maxfreq : float, optional
        Maximum frequency at which to evaluate the spectral window.
        Default is 100.
    oversample : float, optional
        Oversampling factor relative to the natural frequency resolution.
        Default is 10.
    log : bool, optional
        If True, record the calculation in the `Prewhitener` log.
        Default is False.
    progress_callback : callable or None, optional
        Callback invoked as ``progress_callback(completed, total)`` after
        each frequency evaluation.
    cancel_check : callable or None, optional
        Callable returning True when the calculation should stop.

    Returns
    -------
    frequencies : numpy.ndarray
        Frequencies at which the spectral window was evaluated.
    amplitudes : numpy.ndarray
        Normalized spectral-window amplitudes.
    """
    if oversample <= 0:
        raise ValueError("oversample must be positive.")

    if maxfreq <= 0:
        raise ValueError("maxfreq must be positive.")

    included = np.asarray(pw.lc["include"], dtype=bool)
    time = pw.lc.time.value[included]

    if len(time) < 2:
        raise ValueError(
            "At least two included observations are required."
        )

    fres = 1.0 / (
        pw.freq_conversion * np.ptp(time)
    )

    frequencies = np.arange(
        0,
        maxfreq,
        fres / oversample,
    )

    window = np.full(len(time), 0.5)

    amplitudes = []

    total = len(frequencies)

    # Aim for at most ~100 progress-bar updates.
    progress_interval = max(1, total // 100)

    for i, frequency in enumerate(frequencies):
        if cancel_check is not None and cancel_check():
            break

        omega = (
            2.0
            * np.pi
            * frequency
            * pw.freq_conversion
        )

        sine = np.sin(omega * time)
        cosine = np.cos(omega * time)

        cosine_amp = np.dot(cosine, window)
        sine_amp = np.dot(sine, window)

        amplitudes.append(
            np.hypot(cosine_amp, sine_amp)
        )

        completed = i + 1

        if (
            progress_callback is not None
            and (
                completed % progress_interval == 0
                or completed == total
            )
        ):
            progress_callback(completed, total)

    amplitudes = np.asarray(amplitudes)
    amplitudes *= 2.0 / len(time)

    # If cancelled, return only frequencies actually calculated.
    frequencies = frequencies[:len(amplitudes)]

    if log:
        pw.log(
            "Calculated spectral window from "
            f"{len(time)} included observations "
            f"up to {maxfreq} {pw.freq_unit} "
            f"with oversampling factor {oversample}."
        )

    return frequencies, amplitudes