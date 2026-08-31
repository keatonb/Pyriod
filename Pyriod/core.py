"""Core numerical analysis functionality for Pyriod.

This module defines the :class:`Prewhitener` class, which manages
light-curve data, sinusoidal signal models, iterative prewhitening,
periodogram calculation, and significance thresholds.

The interactive interface is implemented separately by
:class:`PyriodGUI`.
"""

# Standard imports
import sys
import os
import itertools
import re
import logging
import warnings
import html
if sys.version_info < (3, 0):
    from StringIO import StringIO
else:
    from io import StringIO

# Third party imports
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import astropy.units as u
import lightkurve as lk
from lmfit import Model, Parameters
from bs4 import BeautifulSoup
from bs4.builder import XMLParsedAsHTMLWarning

# Local imports
# from .pyquist import subfreq (not currently used)
from .combinations import evaluate_combination, validate_combination, CombinationExpressionError
from .models import sin
from .utils import make_all_iter

# Ignore xml warning
warnings.filterwarnings(action='ignore', category=XMLParsedAsHTMLWarning,
                        module='bs4')

# From https://stackoverflow.com/a/16571630
class Capturing(list):
    """Captures stdout.

    From https://stackoverflow.com/a/16571630
    """

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout


class Prewhitener(object):
    """Perform iterative frequency analysis and prewhitening.

    `Prewhitener` contains the core analysis state and fitting
    functionality used by Pyriod. It can be used directly for
    non-interactive analysis or passed to `PyriodGUI` to create
    the interactive notebook interface.


    Parameters
    ----------
    lc : lightkurve.LightCurve
        Light curve data to analyze.
    amp_unit : str, optional
        Amplitude unit used for input and displayed amplitudes. Accepted
        values include ``"relative"``, ``"percent"``, ``"ppt"``, ``"ppm"``,
    and ``"mma"``. Default is ``"ppt"``.
        Frequency unit. Accepted values include aliases for microhertz 
        (e.g., ``"muhz"``) and cycles per day (e.g., ``"1/d"``). Default is 
        ``"muHz"``.
    use_weights : (bool)
        If true, weight data points by ``1/lc.flux_err`` when available.
        Default is True.
    rescale_covar : bool, optional
        Whether lmfit should rescale the covariance matrix when estimating
        parameter uncertainties. Default is False.
    ls_method :  str, optional
        Lomb-Scargle method keyword passed to lightkurve LombScarglePeriodogram.
        Default is "fast".
    frequency : array-like, optional
        Explicit frequency grid on which to calculate periodograms, in
        ``freq_unit``. Default is None, and sampling frequencies will be chosen.
    oversample_factor : float, optional
        Number of frequency samples per natural resolution element
        ``1 / duration``. Default is 5.
    nyquist_factor : float, optional
        Maximum automatically generated frequency as a multiple of the
        approximate Nyquist frequency. Default is 1. Ignored when
        ``maxfreq`` is specified.
    minfreq : float, optional
        Minimum automatically generated frequency, in ``freq_unit``.
        Default is 1/duration.
    maxfreq : float, optional
        Maximum automatically generated frequency, in ``freq_unit``.
        By default it is determined from ``nyquist_factor``.

    Attributes
    ----------
    lc : lightkurve.LightCurve
        Internal copy of the input light curve. It includes an ``include``
        column indicating which observations are currently used.
    stagedvalues : pandas.DataFrame
        Signal parameters staged for the next model fit.
    fitvalues : pandas.DataFrame
        Parameters of the most recently fitted signal model.
    fit_result : lmfit.model.ModelResult or None
        Detailed result of the most recent lmfit optimization.
    uptodate : bool
        Whether the fitted model reflects the current staged signal
        parameters and light curve mask.
    lc_model : lightkurve.LightCurve
        Current fitted model evaluated at the observation times.
    lc_resid : lightkurve.LightCurve
        Residual light curve after subtracting ``lc_model`` from ``lc``.
    freqs : numpy.ndarray
        Frequency grid on which the periodograms are evaluated, in
        ``freq_unit``.
    fres : float or None
        Natural frequency resolution used to construct an automatically
        generated frequency grid. ``None`` when an explicit ``frequency``
        grid is supplied.
    oversample_factor : float or None
        Oversampling factor of an automatically generated frequency grid.
    nyquist : float
        Approximate Nyquist frequency, in ``freq_unit``.
    nyquist_factor : float
        Ratio of the upper frequency range to the approximate Nyquist
        frequency.
    nyquist_quality : float
        Metric between 0 and 1 characterizing the strength of reflection
        across the approximate Nyquist frequency.
    per_orig : numpy.ndarray
        Amplitude periodogram of the original light curve evaluated on
        ``freqs``, in ``amp_unit``.
    per_model : numpy.ndarray
        Amplitude periodogram of the current fitted model evaluated on
        ``freqs``, in ``amp_unit``.
    per_resid : numpy.ndarray
        Amplitude periodogram of the residual light curve evaluated on
        ``freqs``, in ``amp_unit``. This is the primary periodogram used
        to identify additional signals during prewhitening.
    noise_spectrum : callable or None
        Interpolating function giving the estimated local average
        residual-periodogram amplitude as a function of frequency.
    significance_multiplier : float or None
        Multiplier applied to ``noise_spectrum`` to define the current
        significance threshold.
    significance_settings : dict or None
        Settings used to calculate the current significance threshold.
    autorecalculate : bool
        Whether the significance threshold is automatically recalculated
        after the periodogram changes.
    tshift : float
        Time offset, in days, applied internally to improve numerical
        behavior of phase fitting.
    amp_unit : str
        Selected amplitude-unit name.
    amp_conversion : float
        Conversion factor between internal relative amplitudes and
        displayed amplitudes.
    freq_unit : astropy.units.Unit
        Astropy unit used for frequencies.
    freq_conversion : float
        Conversion factor between frequencies in ``freq_unit`` and inverse
        days.
    log_html : str
        HTML representation of messages recorded in the Pyriod log.
    """
    # Generate unique ID for this Pyriod instance
    _id_generator = itertools.count(0)

    def __init__(self, lc, amp_unit='ppt', freq_unit='muHz',
                 use_weights=True, rescale_covar=False, ls_method='fast', **kwargs):
        ### LOG ### 
        self._init_log() # initialize first
        self.log("Pyriod Prewhitener initializing...")

        ### TIME SERIES ###
        # Stored as lightkurve.LightCurve object
        # all provided columns besides time, flux, and flux_err are not stored
        # "flux" column is original data
        # "resid" is residuals
        # "include" is included points
        self.use_weights = use_weights # may be changed by _set_light_curve
        self._set_light_curve(lc)

        self.log("Fitting specifications:")
        self.log(f'Use weights?: {self.use_weights}')
        self.rescale_covar = rescale_covar
        self.log(f'Recale covariance matrix?: {self.rescale_covar}')
        self.ls_method = ls_method
        self.log(f'Lomb-Scargle method: {self.ls_method}')

        # Set up some things
        self.fit_result = None # replace as we do fits

        # Work out the units
        self._set_units(amp_unit=amp_unit, freq_unit=freq_unit)

        # Apply time shift to get phases to be well behaved
        self._calc_tshift()

        # Initialize DataFrames to hold staged and fitted values
        self.stagedvalues = self._initialize_dataframe()
        self._fitvalues = self.stagedvalues.copy().drop('brute', axis=1)

        # Establish frequency sampling
        self.set_frequency_sampling(**kwargs)

        # Significance threshold attributes
        # TODO: take init arguments to define significance threshold
        self.noise_spectrum = None
        self.significance_multiplier = None
        self.significance_settings = None
        self.autorecalculate = False

        # Compute initial periodograms
        self.compute_pers(orig=True) 

        

        self._lcchanged = False # initial state
        self.log("Pyriod object initialized.")

    ###### initialization functions #######

    def _set_units(self, amp_unit=None, freq_unit=None):
        """Set amplitude and frequency units.

        Parameters
        ----------
        amp_unit : str, optional
            Amplitude unit. Accepted values are ``"relative"``, ``"percent"``,
            ``"ppt"``, ``"ppm"``, and ``"mma"``.
        freq_unit : str, optional
            Frequency unit. Accepted aliases correspond to microhertz or
            inverse days.

        Raises
        ------
        KeyError
            If an unsupported amplitude or frequency unit is supplied.
        """
        if amp_unit is not None:
            self.amp_unit = amp_unit
            unitoptions = {'relative': 1e0, 'percent': 1e2, 'ppt': 1e3,
                           'ppm': 1e6, 'mma': 1e3}
            self.amp_conversion = unitoptions[self.amp_unit.lower()]
            self.log(f'Amplitude unit set to {amp_unit} '
                     f'(factor of {self.amp_conversion}).')
        if freq_unit is not None:
            muHz = u.microHertz
            perday = (1/u.day).unit
            unitoptions = {'muhz': muHz, 'uhz': muHz, 'microhertz': muHz,
                           '1/d': perday, '1/day': perday, 'day': perday,
                           'days': perday, 'd': perday, 'per day': perday}
            self.freq_unit = unitoptions[freq_unit.lower()]
            self._freq_label = {perday: "1/day", muHz: "muHz"}[self.freq_unit]
            self.log(f'Frequency unit set to {self._freq_label}.')
        self.time_unit = u.day
        self.freq_conversion = self.time_unit.to(1/self.freq_unit)

    def _set_light_curve(self, lc):
        """Validate, copy, clean, and initialize the input light curve."""
        # Input must be Lightkurve LightCurve type
        if not issubclass(type(lc), lk.LightCurve):
            raise TypeError('lc must be a lightkurve.LightCurve object.')
        self.lc = lc.copy()  # copy so we don't modify original
        self._log_lc_properties()

        # Drop all columns besides time, flux, and flux_err
        keepcolumns = ['time','flux','flux_err']
        self.lc.remove_columns([col for col in lc.columns if col not in keepcolumns])

        # Check for nans and remove if needed
        nnans = np.sum(np.isnan(np.array(self.lc.flux.value)))
        if nnans > 0:
            self.log(f"Removing {nnans} nans from light curve flux column.")
            self.lc = self.lc.remove_nans()

        # Check if uncertainties provided
        if self.use_weights:
            nanweights = np.isnan(lc.flux_err.value)
            if np.all(nanweights):
                # No uncertainties in light curve
                self.log("No flux uncertainties provided. Data points will "
                         "not be fit using weights.", level='warning')
                self.use_weights = False
            elif np.any(nanweights):
                self.log(f"Removing {np.sum(nanweights)} nans from light curve"
                         " flux_err column.")
                self.lc = self.lc.remove_nans('flux_err')

        # Maintain a mask of points to exclude from analysis
        self.lc["include"] = np.ones(len(self.lc))  # 1 = include

    # Class properties to ease and control access to attributes
    @property
    def lc_model(self):
        """Current fitted model evaluated at the observation times.

        Returns
        -------
        lightkurve.LightCurve
            Light curve containing the mean observed flux plus the current
            fitted sinusoidal model. The ``include`` column is copied from
            the analyzed light curve.
        """
        meanflux = float(np.nanmean(self.lc.flux.value))
        lc = lk.LightCurve(time = self.lc.time,
                           flux = (meanflux + self.sample_model(self.lc.time.value))
                                               *self.lc.flux.unit)
        lc["include"] = self.lc["include"]
        return lc
    
    @property
    def lc_resid(self):
        """Residual light curve for the current fitted model.

        Returns
        -------
        lightkurve.LightCurve
            Observed flux minus ``lc_model``, with the original flux
            uncertainties and ``include`` mask.
        """
        lc_model = self.lc_model
        lc = lk.LightCurve(time = self.lc.time,
                           flux = self.lc["flux"] - lc_model.flux,
                           flux_err = self.lc["flux_err"]) 
        lc["include"] = self.lc["include"]
        return lc
    
    @property
    def fitvalues(self):
        """Parameters of the most recently fitted signal model.

        Returns
        -------
        pandas.DataFrame
            Internal fitted-signal table.

        Notes
        -----
        Amplitudes are stored internally in relative flux units. Use
        ``solution_table()`` to obtain a copy with amplitudes converted to
        the current display unit.
        """
        return self._fitvalues.copy() # read-only

    @property
    def uptodate(self):
        """Whether the fitted model matches the currently staged analysis state.

        Returns
        -------
        bool
            True if the staged signal parameters equal the fitted parameters,
            no signal is awaiting brute-force phase estimation, and the
            light-curve mask has not changed since the last fit.
        """
        colcompare = [c for c in self.columns if c != "brute"]
        nobrute = all(~self.stagedvalues["brute"])
        stagedisfit = np.array_equal(self.stagedvalues[colcompare].values, self.fitvalues[colcompare].values)
        return bool(nobrute and stagedisfit and not self._lcchanged)

    def _init_log(self):
        """Initialize the logger and in-memory log buffer for this instance."""
        # Make unique ID number for this session to send messages to correct log
        self.id = next(self._id_generator)
        self._logger = logging.getLogger(f'Pyriod Logger {self.id}')
        self._logger.setLevel(logging.DEBUG)
        self._log_capture_string = StringIO()
        ch = logging.StreamHandler(self._log_capture_string)
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self._logger.addHandler(ch)
        # Log the initialization of the Log.
        self.log(f'Initiating Pyriod instance {self.id}.')
    
    def log(self, message, level='info'):
        """Record a message in the Pyriod log.

        Parameters
        ----------
        message : str
            Message to record.
        level : {"debug", "info", "warning", "error", "critical"}, optional
            Logging level. Default is ``"info"``.
        """
        logdict = {
            'debug': self._logger.debug,
            'info': self._logger.info,
            'warning': self._logger.warning,
            'error': self._logger.error,
            'critical': self._logger.critical
            }
        if message[:-2] != '\n':
            message += '\n'
        logdict[level](message)

    @property
    def log_html(self):
        """HTML representation of the current Pyriod log.

        Returns
        -------
        str
            Escaped log contents enclosed in a ``<pre>`` element.
        """
        raw_log = self._log_capture_string.getvalue()
        return ("<pre style='white-space: pre-wrap; "
                "font-family: monospace; "
                "margin: 0;'>"
                f"{html.escape(raw_log)}"
                "</pre>")

    def _log_lc_properties(self):
        """Write available light curve metadata to the Pyriod log."""
        keys = self.lc.meta.keys()
        if len(keys) > 0:
            self.log("The provided light curve has the following metadata:")
            for key in keys:
                self.log(f"{key}: {self.lc.meta[key]}")

    def _log_per_properties(self, per):
        """Write Lightkurve periodogram properties to the Pyriod log.

        Parameters
        ----------
        per : lightkurve.periodogram.Periodogram
            Periodogram whose properties are to be logged.
        """
        try:
            with Capturing() as output:
                per.show_properties()
            info = re.sub(' +', ' ',
                          str("".join([e+' |\n' for e in output[3:]])))
            self.log("Periodogram properties:" + info)
        except Exception:
            pass

    def set_frequency_sampling(self, frequency=None, oversample_factor=5,
                               nyquist_factor=1, minfreq=None, maxfreq=None):
        """Set the frequency grid used for subsequent periodograms.

        The grid may be supplied explicitly or generated at a specified
        oversampling factor over an automatically or explicitly bounded
        frequency range.

        Parameters
        ----------
        frequency : array-like, optional
            Explicit frequency samples, in ``freq_unit``. If provided,
            ``oversample_factor``, ``minfreq``, and ``maxfreq`` are not used
            to construct the grid.
        oversample_factor : float, optional
            Number of samples per natural frequency-resolution element
            ``1 / duration``. Default is 5.
        nyquist_factor : float, optional
            Upper limit of an automatically generated grid as a multiple of
            the approximate Nyquist frequency. Default is 1. Overridden by
            ``maxfreq``.
        minfreq : float, optional
            Minimum frequency of an automatically generated grid, in
            ``freq_unit``. Default is the natural frequency resolution.
        maxfreq : float, optional
            Maximum frequency of an automatically generated grid, in
            ``freq_unit``. By default it is determined from
            ``nyquist_factor``.

        Notes
        -----
        This method updates ``freqs`` but does not automatically recalculate
        ``per_orig``. Call ``compute_pers(orig=True)`` to recalculate the
        original-light-curve periodogram on the new grid.

        The approximate Nyquist frequency is calculated as ``1 / (2 dt)``,
        where ``dt`` is the median separation between adjacent observations.
        This is exact only for evenly sampled data. ``nyquist_quality`` provides
        a measure between 0 and 1 of how strongly the sampling produces
        reflection about this approximate Nyquist frequency.
        """
        # Approximate Nyquist frequency (exact only for evenly sampled data)
        dt = np.median(np.diff(np.sort(self.lc.time.value)))
        self.nyquist = 1/(2.*dt*self.freq_conversion)
        # Evaluate the quality of Nyquist estimate
        # (between 0-1, 1 being strongest alias)
        nyqphase = ((self.lc.time.value
                     % (0.5/(self.nyquist*self.freq_conversion)))
                    / (0.5/(self.nyquist*self.freq_conversion)))
        y = np.sin(2*np.pi*nyqphase)
        x = np.cos(2*np.pi*nyqphase)
        self.nyquist_quality = np.sqrt(np.mean(x)**2.+np.mean(y)**2.)
        # Frequency resolution
        self.fres = 1./(self.freq_conversion*np.ptp(self.lc.time.value))
        # Are we using user-speficied frequencies?
        if frequency is not None:
            self.log(f'Using user supplied frequency sampling: '
                     f'{len(frequency)} samples between frequency '
                     f'{np.min(frequency)} and {np.max(frequency)} '
                     f'{self._freq_label}')
            self.freqs = frequency
            self.oversample_factor = None
            self.nyquist_factor = np.max(frequency)/self.nyquist
        else: # If making our own frequency grid
            self.oversample_factor = oversample_factor
            self.nyquist_factor = nyquist_factor
            if minfreq is None:
                minfreq = self.fres
            if maxfreq is None:
                maxfreq = (self.nyquist*self.nyquist_factor
                           + 0.9*self.fres/self.oversample_factor)
            self.freqs = np.arange(minfreq, maxfreq,
                                   self.fres/self.oversample_factor)
        return

    # Functions for interacting with model fit below
    def _next_signal_index(self, n=1):
        """Return the next unused independent-signal labels.

        Parameters
        ----------
        n : int, optional
            Number of labels to generate. Default is 1.

        Returns
        -------
        list of str
            Unused labels of the form ``"f#"``.
        """
        inds = []
        i = 0
        while len(inds) < n:
            if not "f{}".format(i) in self.stagedvalues.index:
                inds.append("f{}".format(i))
            i += 1
        return inds

    def add_signal(self, freq, amp=None, phase=None, fixfreq=False,
                   fixamp=False, fixphase=False, include=True, brute=True,
                   index=None):
        """Stage one or more independent sinusoidal signals for fitting.

        Scalar arguments may be supplied for a single signal. For multiple
        signals, iterable arguments may be supplied; scalar values are repeated
        as needed.

        Parameters
        ----------
        freq : float or array-like of float
            Initial signal frequencies, in ``freq_unit``.
        amp : float or array-like of float, optional
            Initial signal amplitudes, in ``amp_unit``.
        phase : float or array-like of float, optional
            Initial phases, expressed in cycles.
        fixfreq : bool or array-like of bool, optional
            If True, hold the corresponding frequency fixed during fitting.
            Default is False.
        fixamp : bool or array-like of bool, optional
            If True, hold the corresponding amplitude fixed during fitting.
            Default is False.
        fixphase : bool or array-like of bool, optional
            If True, hold the corresponding phase fixed during fitting.
            Default is False.
        include : bool or array-like of bool, optional
            Whether to include each signal in the next model fit. Default is
            True.
        brute : bool or array-like of bool, optional
            Whether to estimate the initial phase by brute-force sampling
            before fitting. Default is True.
        index : str or iterable of str, optional
            Signal label or labels. Missing labels are assigned the next
            available independent-signal labels of the form ``"f#"``.

        Raises
        ------
        ValueError
            If duplicate labels are supplied or a supplied label already
            exists in ``stagedvalues``.

        Notes
        -----
        Signals are added to ``stagedvalues`` and do not become part of the
        fitted model until ``fit_model()`` is called.
        """
        freq, amp, phase, fixfreq, fixamp, fixphase, include, brute, index = (
            make_all_iter([freq, amp, phase, fixfreq, fixamp, fixphase,
                                 include, brute, index]))
        colnames = ["freq", "fixfreq", "amp", "fixamp", "phase", "brute",
                    "fixphase", "include"]
        newvalues = [nv for nv in [freq, fixfreq, amp, fixamp, phase, brute,
                                   fixphase, include]]
        dictvals = dict(zip(colnames, newvalues))
        for i in range(len(freq)):
            if dictvals["amp"][i] is None:
                dictvals["amp"][i] = 1.
            else:
                dictvals["amp"][i] /= self.amp_conversion

        # Replace all None indices with next available
        noneindex = np.where([ind is None for ind in index])[0]
        newindices = self._next_signal_index(n=len(noneindex))
        for i in range(len(noneindex)):
            index[noneindex[i]] = newindices[i]

        # Check that all indices are unique and none already used
        if (len(index) != len(set(index))) or any([ind in
                                                   self.stagedvalues.index
                                                   for ind in index]):
            raise ValueError("Duplicate indices provided.")
        toconcat = pd.DataFrame(dictvals, columns=self.columns, index=index)
        toconcat = toconcat.astype(dtype=dict(zip(self.columns, self.dtypes)))
        self.stagedvalues = pd.concat([self.stagedvalues, toconcat],
                                      sort=False)
       
        self.log(f"Signal {index} added to model with frequency "
                 f"{freq} and amplitude {amp}.")

    def _combination_to_lmfit_expr(self, combostr, prefixmap):
        """Return a lmfit parameter expression for combination expression."""
        known_labels = set(map(str.lower, self.stagedvalues.index))
        if not validate_combination(combostr, known_labels):
            raise ValueError(f"Invalid combination expression: {combostr}")

        parts = re.split(r"(\+|\-|\*|\/|\(|\))", combostr.replace(" ", "").lower())
        converted = []
        for part in parts:
            if part in self.stagedvalues.index:
                converted.append(prefixmap.get(part, part) + "freq")
            else:
                converted.append(part)

        return "".join(converted)

    def _valid_combo(self, combostr):
        """Return whether a combination expression is valid."""
        known_labels = set(map(str.lower, self.stagedvalues.index))
        return validate_combination(combostr, known_labels)

    def add_combination(self, combostr, amp=None, phase=None, fixamp=False,
                        fixphase=False, include=True, brute=True):
        """Stage one or more combination-frequency signals for fitting.

        Combination frequencies are defined by arithmetic expressions involving
        existing staged signal labels. The combination expression is also used
        as the signal label.

        Parameters
        ----------
        combostr : str or iterable of str
            Arithmetic expression or expressions defining frequencies in terms
            of existing staged signal labels, for example ``"f0+f1"``.
        amp : float or iterable of float, optional
            Initial amplitude in ``amp_unit``. If None, the amplitude is
            initialized by interpolating ``per_resid`` at the combination
            frequency.
        phase : float or iterable of float, optional
            Initial phase in cycles.
        fixamp : bool or iterable of bool, optional
            If True, hold amplitude fixed during fitting. Default is False.
        fixphase : bool or iterable of bool, optional
            If True, hold phase fixed during fitting. Default is False.
        include : bool or iterable of bool, optional
            Whether to include the signal in the next fit. Default is True.
        brute : bool or iterable of bool, optional
            Whether to estimate the initial phase by brute-force sampling.
            Default is True.

        Notes
        -----
        Invalid combination expressions are reported to the Pyriod log and no
        combination signal is added.
        """
        combostr, amp, phase, fixamp, fixphase, include, brute = (
            make_all_iter([combostr, amp, phase, fixamp, fixphase,
                                 include, brute]))
        freq = np.zeros(len(combostr)) # Initial evaulation of provided expressions

        freq_lookup = {
            str(label).lower(): float(self.stagedvalues.loc[label, "freq"])
            for label in self.stagedvalues.index
        } # labels that expressions can be a combination of

        if all([self._valid_combo(c) for c in combostr]): #Make sure all look valid
            try: # In case combos are invalid
                for i in range(len(combostr)):
                    freq[i] = evaluate_combination(combostr[i], freq_lookup)
                    if amp[i] is None:
                        amp[i] = np.interp(freq[i],self.freqs,self.per_resid)
                self.add_signal(list(freq), amp, phase, False, fixamp, fixphase,
                                include, brute, index=combostr)
            except CombinationExpressionError:
                self.log(f"Invalid combination string provided in {combostr}.","error")
        else:
            self.log(f"Invalid combination string provided in {combostr}.","error")

    def _brute_phase_est(self, freq, amp, brute_step=0.1):
        """Estimate a signal phase by brute-force sampling.

        A single sinusoid with fixed frequency and amplitude is fit to the
        current residuals while phase is sampled between 0 and 1 cycle.

        Parameters
        ----------
        freq : float
            Fixed signal frequency, in ``freq_unit``.
        amp : float
            Fixed signal amplitude in internal relative-flux units.
        brute_step : float, optional
            Phase-grid spacing in cycles. Default is 0.1.

        Returns
        -------
        float
            Estimated phase in cycles.
        """
        model = Model(sin)
        params = model.make_params()
        params['freq'].set(self.freq_conversion*freq, vary=False)
        params['amp'].set(amp, vary=False)
        params['phase'].set(0.5, vary=True, min=0, max=1,
                            brute_step=brute_step)
        good = np.where(self.lc["include"])
        meanflux = float(np.mean(np.array(self.lc.flux.value[good])))
        modellc = (meanflux + self.sample_model(self.lc.time.value[good]))*self.lc.flux.unit
        resid = self.lc["flux"][good] - modellc # bad points dropped
        result = model.fit(resid.value,
            params,
            x=(self.lc.time.value[good]+self.tshift),
            method='brute')
        return result.params['phase'].value

    def fit_model(self):
        """Fit the staged sinusoidal model to the light curve.

        The model is a sum of all signals in ``stagedvalues`` for which
        ``include`` is True. Initial parameter values are taken from
        ``stagedvalues``. Signals marked with ``brute=True`` have their initial
        phase estimated by brute-force sampling before optimization.

        Independent signals are represented by free or fixed sinusoidal
        parameters. Combination-signal frequencies are constrained through
        lmfit expressions relating them to their constituent independent
        frequencies.

        After a successful fit, fitted parameters are stored in ``fitvalues``
        and the complete lmfit result is stored in ``fit_result``. The model
        and residual periodograms are then recalculated.

        Notes
        -----
        If no signals are included, no optimization is performed and the fitted
        solution is reset to an empty table. If all included parameters are
        fixed, a warning is written to the log and no optimization is performed.
        """
        # Check that there are signals in the model
        if np.sum(self.stagedvalues.include.values) == 0:
            self.log("No signals to fit.", level='warning')
            self._fitvalues = self._initialize_dataframe().drop('brute', axis=1)
            self.fit_result = None  # No fit
        elif np.all(self.stagedvalues[self.stagedvalues.include]  # All fixed
                    [['fixfreq', 'fixamp', 'fixphase']]):
            self.log("No signals with free parameters allowed to vary.",
                     level='warning')
        else:  # Fit a model
            # Set up lmfit model for fitting
            signals = {}  # Empty dict to be populated
            params = Parameters()

            # Handle combination frequencies differently
            isindep = lambda key: key[1:].isdigit()
            cnum = 0  # Number of combination frequencies

            # Fitting prefix, f# for independent, c# for combination
            prefixmap = {}

            # Set up model to fit (for included signals only)
            for prefix in self.stagedvalues.index[self.stagedvalues.include]:
                if isindep(prefix):
                    signals[prefix] = Model(sin, prefix=prefix)
                    params.update(signals[prefix].make_params())
                    params[prefix+'freq'].set(
                        self.freq_conversion*self.stagedvalues.freq[prefix],
                        vary=~self.stagedvalues.fixfreq[prefix])
                    params[prefix+'amp'].set(
                        self.stagedvalues.amp[prefix],
                        vary=~self.stagedvalues.fixamp[prefix])
                    # Correct phase for tdiff
                    thisphase = (self.stagedvalues.phase[prefix]
                                 - (self.tshift * self.freq_conversion
                                    * self.stagedvalues.freq[prefix])) % 1

                    # Estimate phase for new signals with _brute_phase_est
                    # (or those with brute = True)
                    if np.isnan(thisphase) or self.stagedvalues.brute[prefix]:
                        thisphase = self._brute_phase_est(
                            self.stagedvalues.freq[prefix],
                            self.stagedvalues.amp[prefix])

                    params[prefix+'phase'].set(
                        thisphase, min=-np.inf, max=np.inf,
                        vary=~self.stagedvalues.fixphase[prefix])
                    prefixmap[prefix] = prefix
                else:  # Combination frequency
                    useprefix = 'c{}'.format(cnum)
                    signals[useprefix] = Model(sin, prefix=useprefix)
                    params.update(signals[useprefix].make_params())
                    expression = self._combination_to_lmfit_expr(prefix, prefixmap)
                    params[useprefix+'freq'].set(expr=expression)
                    params[useprefix+'amp'].set(
                        self.stagedvalues.amp[prefix],
                        vary=~self.stagedvalues.fixamp[prefix])
                    # Correct phase for tdiff
                    thisphase = (self.stagedvalues.phase[prefix]
                                 - (self.tshift * self.freq_conversion
                                    * self.stagedvalues.freq[prefix])) % 1
                    if np.isnan(thisphase):  # If new signal to fit
                        thisphase = self._brute_phase_est(
                            self.stagedvalues.freq[prefix],
                            self.stagedvalues.amp[prefix])
                    params[useprefix+'phase'].set(
                        thisphase, min=-np.inf, max=np.inf,
                        vary=~self.stagedvalues.fixphase[prefix])
                    prefixmap[prefix] = useprefix
                    cnum += 1

            # Model is sum of sines
            model = np.sum(
                [signals[prefixmap[prefix]] for prefix in
                 self.stagedvalues.index[self.stagedvalues.include]])

            good = np.where(self.lc["include"])
            meanflux = float(np.mean(np.array(self.lc.flux.value[good])))
            modellc = (meanflux + self.sample_model(self.lc.time.value[good]))*self.lc.flux.unit
            resid = self.lc["flux"][good] - modellc # bad points dropped

            # What to use for weights? (stddev if not real error bars)
            weights = 1/np.std(resid)
            if self.use_weights:
                weights = 1/np.array(self.lc.flux_err.value[good])

            # Fit the model
            fluxarray= np.array(self.lc.flux.value[good])
            self.fit_result = model.fit(
                fluxarray - np.mean(fluxarray),
                params, x=self.lc.time.value[good]+self.tshift,
                weights=weights, scale_covar=self.rescale_covar)

            self.log("Fit refined.")
            self.log("Fit properties:"+self.fit_result.fit_report())
            self._update_values_from_fit(self.fit_result.params, prefixmap)
        # up-to-date
        self._lcchanged = False
        # Update lightcurves and periodograms for new residuals
        self.compute_pers()

    def _update_values_from_fit(self, params, prefixmap):
        """Update fitted and staged signal tables from an lmfit result."""
        # Also rectify and negative amplitudes or phases outside [0,1)
        self._fitvalues = self.stagedvalues.astype(
            dtype=dict(zip(self.columns, self.dtypes))).drop('brute', axis=1)
        for prefix in self.stagedvalues.index[self.stagedvalues.include]:
            self._fitvalues.loc[prefix, 'freq'] = float(
                params[prefixmap[prefix]+'freq'].value/self.freq_conversion)
            self._fitvalues.loc[prefix, 'freqerr'] = float(
                params[prefixmap[prefix]+'freq'].stderr/self.freq_conversion)
            self._fitvalues.loc[prefix, 'amp'] = (
                params[prefixmap[prefix]+'amp'].value)
            self._fitvalues.loc[prefix, 'amperr'] = float(
                params[prefixmap[prefix]+'amp'].stderr)
            self._fitvalues.loc[prefix, 'phase'] = (
                params[prefixmap[prefix]+'phase'].value)
            self._fitvalues.loc[prefix, 'phaseerr'] = float(
                params[prefixmap[prefix]+'phase'].stderr)
            # Rectify negative amplitudes (with 0.5 phase change)
            if self._fitvalues.loc[prefix, 'amp'] < 0:
                self._fitvalues.loc[prefix, 'amp'] *= -1.
                self._fitvalues.loc[prefix, 'phase'] -= 0.5
            # Reference phase to t0, and make phase between 0-1
            self._fitvalues.loc[prefix, 'phase'] += (
                self.tshift*self._fitvalues.loc[prefix, 'freq']
                * self.freq_conversion)
            self._fitvalues.loc[prefix, 'phase'] %= 1.

        # Add periods and period uncertainties
        pers = 1./(self._fitvalues['freq']*self.freq_conversion)  # days
        pers = pers*24*3600  # seconds
        pererrs = pers*self._fitvalues['freqerr']/self._fitvalues['freq']
        self._fitvalues['per'] = pers
        self._fitvalues['pererr'] = pererrs

        # Add SNRs too:
        self._update_signal_snr()

        tempdf = self._fitvalues.copy()
        tempdf["brute"] = False
        tempdf = tempdf.astype(
            dtype=dict(zip(self.columns, self.dtypes)))[self.columns]
        self.stagedvalues = tempdf

    def _update_signal_snr(self):
        """Update fitted signal-to-noise ratios from the current noise spectrum."""
        if ((self.noise_spectrum is not None) &
            (self.significance_multiplier is not None)):
            self._fitvalues['snr'] = (
                self.amp_conversion * self._fitvalues['amp'] /
                self.noise_spectrum(self._fitvalues['freq']))

    def solution_table(self, display_units=True, include_brute=True):
        """Return a copy of the current fitted signal table.

        Parameters
        ----------
        display_units : bool, optional
            If True, express ``amp`` and ``amperr`` in ``amp_unit``. If False,
            leave them in internal relative-flux units. Default is True.
        include_brute : bool, optional
            If True, include a ``brute`` column containing False for every
            fitted signal. Default is True.

        Returns
        -------
        pandas.DataFrame
            Copy of the fitted signal table. Modifying the returned table does
            not modify the internally stored fitted values.
        """
        table = self._fitvalues.copy()

        if include_brute:
            table["brute"] = False
            table = table.astype(
                dtype=dict(zip(self.columns, self.dtypes))
            )[self.columns]

        if display_units:
            table["amp"] *= self.amp_conversion
            table["amperr"] *= self.amp_conversion

        return table

    def staged_table(self, display_units=True):
        """Return a copy of the signal parameters staged for the next model fit.

        Parameters
        ----------
        display_units : bool, optional
            If True, convert the ``amp`` and ``amperr`` columns to the current
            display amplitude unit using ``amp_conversion``. If False, return
            amplitudes in the internal fitting units. Default is True.

        Returns
        -------
        pandas.DataFrame
            Copy of the staged signal table. These values are those that will be
            used for the next model evaluation or fit. The returned table can be
            modified without changing the internally staged values.
        """
        table = self.stagedvalues.copy()

        if display_units:
            table["amp"] *= self.amp_conversion
            table["amperr"] *= self.amp_conversion

        return table

    def _set_stagedvalues(self, df):
        """Replace the staged signal table with ``df``."""
        self.stagedvalues = df

    def sample_model(self, time):
        """Evaluate the current fitted signal model at specified times.

        Parameters
        ----------
        time : array-like
            Times at which to evaluate the model, in days.

        Returns
        -------
        numpy.ndarray
            Sum of the currently included fitted sinusoids evaluated at the
            requested times, in the numerical flux units of the input light
            curve.

        Notes
        -------
        Evaluates the sinusoidal variations only and does not include an
        additive offset to match mean light curve flux.
        """
        flux = np.zeros(len(time))
        for prefix in self._fitvalues.index[self._fitvalues.include]:
            freq = float(self._fitvalues.loc[prefix, 'freq'])
            amp = float(self._fitvalues.loc[prefix, 'amp'])
            phase = float(self._fitvalues.loc[prefix, 'phase'])
            flux += sin(time, freq*self.freq_conversion, amp, phase)
        return flux

    # Column names and dtypes for tables
    columns = ['include', 'freq', 'fixfreq', 'freqerr',
               'amp', 'fixamp', 'amperr',
               'phase', 'brute', 'fixphase', 'phaseerr']
    dtypes = ['bool', 'float', 'bool', 'float',
              'float', 'bool', 'float',
              'float', 'bool', 'bool', 'float']

    def remove_signals(self, indices):
        """Remove signals from the staged solution.

        Parameters
        ----------
        indices : str or iterable of str
            Signal label or labels to remove from ``stagedvalues``.

        Notes
        -----
        Combination signals that depend on a removed signal are also removed.
        Missing labels produce a warning in the Pyriod log.

        Removing a staged signal does not remove it from the current fitted
        model until ``fit_model()`` is called.
        """

        # Accept a single string, a pandas Index, list, tuple, or ndarray.
        if isinstance(indices, str):
            indices = [indices]
        else:
            indices = list(indices)
        if len(indices) == 0:
            self.log("No signals provided to be deleted.", level='warning')
            return  # Nothing to remove
        
        # Check if any requested indices are missing
        missing = [idx for idx in indices if idx not in self.stagedvalues.index]
        if missing:
            self.log(f"Signals labels not found and not removed: {missing}", level="warning")
        
        # Check that any requested indices are present
        existing = [idx for idx in indices if idx in self.stagedvalues.index]
        if not existing:
            return
        
        self.log(f"Removed signals: {existing}")
        self.stagedvalues = self.stagedvalues.drop(existing)

        # Also delete associated combination frequencies
        self._void_combos()
    
    def delete_rows(self, indices):
        """Remove signals from the staged solution.

        Deprecated alias for :meth:`remove_signals`.

        Parameters
        ----------
        indices : str or iterable of str
            Signal label or labels to remove.
        """
        warnings.warn("The 'delete_rows' function is deprecated and will be removed in a future version. " \
                      "Use 'remove_signals' instead.", DeprecationWarning)
        return self.remove_signals(indices)

    def _void_combos(self):
        """Remove staged combination signals that are no longer valid."""
        isindep = lambda key: key[1:].isdigit()
        depkeys = []
        for key in self.stagedvalues.index:
            if not isindep(key) and not self._valid_combo(key):
                self.remove_signals(key)

    def _initialize_dataframe(self):
        """Create an empty signal-parameter table.

        Returns
        -------
        pandas.DataFrame
            Empty DataFrame with the standard Pyriod signal columns and dtypes.
        """
        df = (pd.DataFrame(columns=self.columns)
              .astype(dtype=dict(zip(self.columns, self.dtypes))))
        return df
    
    def mask_indices(self, indices, threshold=30):
        """Mask selected light-curve points by index.

        Parameters
        ----------
        indices : array-like of int
            Indices of observations to exclude from the analysis. Their
            ``include`` values in ``lc`` are set to False.
        threshold : int, optional
            Array-printing threshold used when recording the selected indices
            in the log. Passed to ``numpy.array2string``. Default is 30.

        Notes
        -----
        If no indices are supplied, the mask is unchanged and a warning is
        written to the Pyriod log.

        Changing the mask recalculates the time shift and periodograms and
        causes ``uptodate`` to become False until the model is refitted.
        Use ``clear_mask()`` to restore all observations.
        """
        indices = np.asarray(indices, dtype=int)
        if indices.size == 0:
            self.log("No time series points provided to be masked.", level="warning")
            return
        self.log(f"Masking {len(indices)} selected time series points: "+
                 f"{np.array2string(indices,threshold=threshold)}")
        self.lc["include"][indices] = 0
        self._mask_changed()

    def clear_mask(self):
        """Restore all masked light-curve points.

        All values in the ``include`` column of ``lc`` are set to True. The
        time shift and periodograms are recalculated, and the fitted model is
        marked as out of date until it is refitted.
        """
        if np.any(self.lc["include"][:] != 1):
            self.log("Restoring all masked points.")
            self.lc["include"][:] = 1
            self._mask_changed()
        else:
            self.log("'clear_mask' called but no time series points were masked.", level="warning")

    def _mask_changed(self):
        """Update derived state after the light-curve inclusion mask changes."""
        self._calc_tshift()
        self.compute_pers(orig=True)
        self._lcchanged = True

    def _calc_tshift(self, tshift=None):
        """Set the time offset used to stabilize phase fitting."""
        if tshift is None:
            good = np.where(self.lc["include"])
            self.tshift = -np.mean(self.lc[good].time.value)
        else:
            self.tshift = tshift
        self.log(f'Fitted timstamps will be shifted forward relative to '
                 f'given timestamps by `tshift` {self.tshift} days.')

    def compute_pers(self, orig=False):
        """Recalculate model and residual amplitude periodograms.

        The periodograms are evaluated on ``freqs`` using only observations
        whose ``include`` value is True. ``per_model`` and ``per_resid`` are
        always recalculated.

        Parameters
        ----------
        orig : bool, optional
            If True, also recalculate ``per_orig`` for the currently included
            observations. Default is False.

        Notes
        -----
        The resulting amplitude arrays are stored in ``per_model`` and
        ``per_resid`` in ``amp_unit``. If ``orig=True``, ``per_orig`` is also
        updated. The significance threshold is subsequently recalculated when
        ``autorecalculate`` is True.
        """
        good = np.where(self.lc["include"])
        if orig:  # Compute periodogram of original time series
            self.per_orig = self.lc[good].to_periodogram(
                normalization='amplitude', freq_unit=self.freq_unit,
                frequency=self.freqs, 
                ls_method=self.ls_method).power.value * self.amp_conversion
        with np.errstate(invalid='ignore'):
            # Periodogram of model
            meanflux = float(np.nanmean(self.lc.flux.value[good]))
            modellc = lk.LightCurve(time = self.lc.time[good],
                                    flux = (meanflux + self.sample_model(self.lc.time.value[good]))
                                                *self.lc.flux.unit)
            self.per_model = (modellc.to_periodogram(normalization='amplitude',
                                                     freq_unit=self.freq_unit,
                                                     frequency=self.freqs, 
                                                     ls_method=self.ls_method).power.value
                                                     * self.amp_conversion)
            # Periodogram of residuals
            resid = lk.LightCurve(time = self.lc.time[good],
                                  flux = self.lc["flux"][good] - modellc.flux) # bad points dropped
            per_resid = resid.to_periodogram(normalization='amplitude',
                                                   freq_unit=self.freq_unit,
                                                   frequency=self.freqs, 
                                                   ls_method=self.ls_method)
            self._log_per_properties(per_resid)
            self.per_resid = per_resid.power.value * self.amp_conversion
        
        self._recalculate_significance_threshold()


    def calculate_significance_threshold(self, multiplier=5, startfreq=0,
                                         endfreq=None, freqstep=100,
                                         winwidth=100, avgtype="mean",
                                         autorecalculate = False,
                                         **kwargs):
        """Estimate a frequency-dependent amplitude significance threshold.

        The local noise level is estimated from the mean or median amplitude in
        moving windows across ``per_resid``. An interpolating function describing
        this noise spectrum is stored in ``noise_spectrum``. Multiplying it by
        ``multiplier`` gives the significance threshold.

        Parameters
        ----------
        multiplier : float, optional
            Factor by which the local noise estimate is multiplied to define
            the significance threshold. Default is 5.
        startfreq : float, optional
            Frequency at the center of the first averaging window, in
            ``freq_unit``. Default is 0.
        endfreq : float, optional
            Upper limit for centers of averaging windows, in ``freq_unit``.
            Default is the highest sampled frequency.
        freqstep : float, optional
            Separation between averaging-window centers, in ``freq_unit``.
            Default is 100.
        winwidth : float, optional
            Width of each averaging window, in ``freq_unit``. Default is 100.
        avgtype : {"mean", "median"}, optional
            Statistic used to estimate the local periodogram amplitude.
            Default is ``"mean"``.
        autorecalculate : bool, optional
            If True, recalculate the threshold with these settings whenever
            the periodogram changes. Default is False.
        **kwargs
            Additional keyword arguments passed to ``scipy.interpolate.interp1d``.
            If ``fill_value`` is omitted, ``"extrapolate"`` is used.

        Notes
        -----
        The settings used for the calculation are stored in
        ``significance_settings``. Signal-to-noise ratios in the fitted signal
        table are updated using the resulting noise spectrum.
        """
        # Store arguments for reference or recalculation
        self.significance_settings = {"multiplier":multiplier, 
                                      "startfreq":startfreq,
                                      "endfreq":endfreq,
                                      "freqstep":freqstep,
                                      "winwidth":winwidth,
                                      "avgtype":avgtype,
                                      "autorecalculate":autorecalculate}
        self.autorecalculate = autorecalculate

        if endfreq is None:
            endfreq = np.max(self.freqs)

        midbin = np.arange(startfreq, endfreq, freqstep)
        binstart = midbin - winwidth/2
        binend = midbin + winwidth/2
        nbins = len(midbin)

        avgnoise = np.zeros(nbins) + np.nan

        average = {"mean": np.nanmean, "median": np.nanmedian}[avgtype]

        for i in range(nbins):
            inbin = np.where(np.logical_and(self.freqs >= binstart[i],
                                            self.freqs <= binend[i]))
            avgnoise[i] = average(self.per_resid[inbin])

        # Store attributes for plotting
        self._sig_threshold_freqs = midbin
        self._sig_threshold_power = avgnoise * multiplier

        # Extrapolate if fill_value not specified
        if 'fill_value' not in kwargs.keys():
            kwargs["fill_value"] = "extrapolate"

        if len(avgnoise) > 1:
            self.noise_spectrum = interp1d(midbin, avgnoise, bounds_error=False,
                                           **kwargs)
        elif len(avgnoise) == 1:
            self.noise_spectrum = lambda x: avgnoise[0]
        # todo: else more informative error

        self.significance_multiplier = multiplier

        # Update SNR of fitted signals
        self._update_signal_snr()

    def _recalculate_significance_threshold(self):
        """Recalculate the significance threshold when enabled."""
        if (self.autorecalculate & (self.noise_spectrum is not None) &
                                   (self.significance_multiplier is not None) &
                                   (self.significance_settings is not None)):
            ss = self.significance_settings
            self.calculate_significance_threshold(multiplier=ss["multiplier"], 
                                                startfreq=ss["startfreq"],
                                                endfreq=ss["endfreq"],
                                                freqstep=ss["freqstep"],
                                                winwidth=ss["winwidth"],
                                                avgtype=ss["avgtype"],
                                                autorecalculate=ss["autorecalculate"])

    def save_solution(self, filename='Pyriod_solution.csv'):
        """Save the current fitted signal solution to a CSV file.

        Parameters
        ----------
        filename : str or path-like, optional
            Output CSV filename. Default is ``"Pyriod_solution.csv"``.

        Notes
        -----
        Amplitudes and amplitude uncertainties are written in the current
        display amplitude unit.
        """
        self.log("Writing signal solution to " + os.path.abspath(filename))
        self.solution_table(display_units=True).to_csv(filename,
                                                  index_label='label')

    def load_solution(self, filename='Pyriod_solution.csv'):
        """Load a saved signal solution and stage it for fitting.

        Parameters
        ----------
        filename : str or path-like, optional
            CSV file containing a solution previously written by
            ``save_solution()``. Default is ``"Pyriod_solution.csv"``.

        Notes
        -----
        Loaded amplitudes are interpreted in the current ``amp_unit`` and
        converted to internal units. The loaded values replace
        ``stagedvalues`` but are not fitted automatically.

        If the file does not exist, an error is written to the Pyriod log and
        no exception is raised.
        """
        if os.path.exists(filename):
            loaddf = pd.read_csv(filename, index_col='label')
            loaddf.index = loaddf.index.rename(None)
            loaddf["amp"] /= self.amp_conversion
            loaddf["amperr"] /= self.amp_conversion
            self.stagedvalues = loaddf
            logmessage = ("Loading signal solution from "
                          + os.path.abspath(filename) + ".\n")
            self.log(logmessage)
        else:
            self.log("Failed to load " + os.path.abspath(filename)
                     + ". File not found.\n", level='error')

    def save_log(self, filename, overwrite=False):
        """Write the Pyriod log to a text file.

        Parameters
        ----------
        filename : str or path-like
            Output filename.
        overwrite : bool, optional
            If True, replace an existing file. If False, append the current log
            to an existing file, or create the file if it does not exist.
            Default is False.
        """
        logmessage = "Writing log to "+os.path.abspath(filename)
        if overwrite:
            logmessage += ", overwriting."
        self.log(logmessage)
        loghtml = self.log_html
        soup = BeautifulSoup(loghtml, features="xml")
        mode = {True: "w+", False: "a+"}[overwrite]
        f = open(filename, mode)
        f.write(soup.get_text().replace('|', ''))
        f.close()

    ### Advanced Features ###

    def spectral_window(self, maxfreq=100, osample=10):
        """Calculate the spectral window of the included observations.

        The spectral window is evaluated with a direct discrete Fourier
        transform rather than the Lomb-Scargle implementation used for the
        Pyriod periodograms.

        Parameters
        ----------
        maxfreq : float, optional
            Maximum frequency to evaluate, in ``freq_unit``. Default is 100.
        osample : float, optional
            Oversampling factor relative to the natural frequency resolution.
            Default is 10.

        Returns
        -------
        freqvec : numpy.ndarray
            Frequencies at which the spectral window was evaluated, in
            ``freq_unit``.
        ampvec : numpy.ndarray
            Corresponding normalized spectral-window amplitudes.

        Notes
        -----
        Only light curve observations whose ``include`` value is True are used.
        """
        # Compute spectral window with DFT
        # Define the window function
        good = np.where(self.lc["include"])
        time = self.lc.time[good].value
        window = np.ones(len(time))*0.5
        freqvec = np.arange(0, maxfreq, self.fres/osample)
        # DFT function (stolen from Mikemon)
        ampvec = np.zeros(len(freqvec))
        for i, freq in enumerate(freqvec):
            omega = 2.*np.pi*freq*self.freq_conversion
            wts = np.sin(omega*time)
            wtc = np.cos(omega*time)
            camp = np.dot(wtc, window)
            samp = np.dot(wts, window)
            ampvec[i] = np.sqrt(camp**2 + samp**2)
        ampvec = (2./len(time))*np.array(ampvec)
        return freqvec, ampvec

    def close(self, clear_data=True, collect=False):
        """Release resources owned by this Prewhitener.

        Parameters
        ----------
        clear_data : bool, optional
            If True, release large science data products such as the light curve,
            periodograms, fitted values, fit result, and significance-threshold
            arrays. The default is True.

            If False, only logger resources are closed.
        collect : bool, optional
            If True, run garbage collection at the end. Usually not necessary,
            but useful in notebooks after creating many large objects.

        Notes
        -----
        After ``clear_data=True``, this Prewhitener should be considered closed
        and should not be used for further fitting or plotting.
        """
        if getattr(self, "_closed", False):
            return

        self._closed = True

        # ------------------------------------------------------------------
        # 1. Close and remove logger handlers
        # ------------------------------------------------------------------
        logger = getattr(self, "_logger", None)
        if logger is not None:
            for handler in list(logger.handlers):
                try:
                    handler.flush()
                except Exception:
                    pass

                try:
                    handler.close()
                except Exception:
                    pass

                try:
                    logger.removeHandler(handler)
                except Exception:
                    pass

        self._logger = None

        # ------------------------------------------------------------------
        # 2. Close the StringIO log buffer
        # ------------------------------------------------------------------
        log_buffer = getattr(self, "_log_capture_string", None)
        if log_buffer is not None:
            try:
                log_buffer.close()
            except Exception:
                pass

        self._log_capture_string = None

        # ------------------------------------------------------------------
        # 3. Optionally release large science objects
        # ------------------------------------------------------------------
        if clear_data:
            large_attrs = [
                # Light curve and model products
                "lc",

                # Signal tables
                "stagedvalues",
                "_fitvalues",

                # Fit result
                "fit_result",

                # Frequency grid and periodograms
                "freqs",
                "per_orig",
                "per_model",
                "per_resid",

                # Significance-threshold products
                "noise_spectrum",
                "significance_multiplier",
                "significance_settings",
                "_sig_threshold_freqs",
                "_sig_threshold_power",

                # Miscellaneous potentially large/cache-like state
                "nyquist_quality",
            ]

            for name in large_attrs:
                if hasattr(self, name):
                    setattr(self, name, None)

        if collect:
            import gc
            gc.collect()