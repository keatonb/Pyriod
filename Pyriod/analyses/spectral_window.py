"""Spectral-window analysis for Pyriod time series."""

import numpy as np


def spectral_window(
    pw,
    maxfreq=100,
    oversample=10,
    minfreq=0,
    log=False,
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
        Oversampling factor relative to the natural frequency resolution
        ``1 / baseline``. Default is 10.
    minfreq : float, optional
        Minimum frequency at which to evaluate the spectral window.
        Default is 0.
    log : bool, optional
        If True, record the calculation in the `Prewhitener` log.
        Default is False.

    Returns
    -------
    frequencies : numpy.ndarray
        Frequencies at which the spectral window was evaluated, in
        ``pw.freq_unit``.
    amplitudes : numpy.ndarray
        Normalized spectral-window amplitudes.
    """
    if oversample <= 0:
        raise ValueError("oversample must be positive.")

    if maxfreq <= minfreq:
        raise ValueError("maxfreq must be greater than minfreq.")

    included = np.asarray(pw.lc["include"], dtype=bool)
    time = pw.lc.time.value[included]

    if len(time) < 2:
        raise ValueError(
            "At least two included observations are required."
        )

    # Natural frequency resolution in pw.freq_unit.
    fres = 1.0 / (
        pw.freq_conversion * np.ptp(time)
    )

    frequencies = np.arange(
        minfreq,
        maxfreq,
        fres / oversample,
    )

    # Unit-amplitude sampling/window function.
    window = np.full(len(time), 0.5)

    amplitudes = np.zeros(len(frequencies))

    for i, frequency in enumerate(frequencies):
        omega = 2.0 * np.pi * frequency * pw.freq_conversion

        sine = np.sin(omega * time)
        cosine = np.cos(omega * time)

        cosine_amp = np.dot(cosine, window)
        sine_amp = np.dot(sine, window)

        amplitudes[i] = np.hypot(
            cosine_amp,
            sine_amp,
        )

    amplitudes *= 2.0 / len(time)

    if log:
        pw.log(
            "Calculated spectral window from "
            f"{len(time)} included observations between "
            f"{minfreq} and {maxfreq} {pw.freq_unit} "
            f"with oversampling factor {oversample}."
        )

    return frequencies, amplitudes