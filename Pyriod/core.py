"""Pyriod, an astronomical prewhitening frequency analysis package.

Written by Keaton Bell

For more, see https://github.com/keatonb/Pyriod

---------------------

This is the core file with the Prewhitener object that handles
the data analysis directly.
The GUI has been refactored into its own class.

---------------------
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
    # Generate unique ID for this Pyriod instance
    _id_generator = itertools.count(0)

    """Core fitting object implementing the pre-whitening algorithm.

    Parameters
    ----------
        lc : lightkurve.LightCurve
            light curve data to analyze
        amp_unit : ("ppt", "percent", etc)
            amplitude unit to use
        freq_unit: ("muHz" or "perday")
            frequency unit to use
        use_weights: (bool)
            Weight data points by 1/lc.flux_err (if available)? The default is
            True.
        rescale_covar: (bool)
            Rescale covariance matrix when estimating uncertainties? The
            default is False.
        ls_method:  (str)
            Lomb-Scargle method keyword passed to lightkurve LombScarglePeriodogram.
            default is "fast".
        **kwargs are passed to set_frequency_sampling() method, which takes arguments
        ----------
            frequency : array, optional
                Explicit set of frequencies to compute periodogram at. The default
                is None.
            oversample_factor : FLOAT, optional
                How many times more densely than the natural frequency resolution
                of 1/duration to sample frequencies. The default is 5.
            nyquist_factor : FLOAT, optional
                How many time beyond the approximate Nyquist frequency to sample
                periodograms. The default is 1. Overridden by maxfreq, if provided.
                Note that the Nyquist frequency is estimated to equal 1/(2*dt),
                where dt is the median time separation between adjacent samples.
                This is only valid for evenly sampled data, and may be a very poor
                approximation for unevenly sampled data.
            minfreq : FLOAT
                Minimum frequency of range to use. The default is 1/duration.
            maxfreq : FLOAT
                Maximum frequency of range to use. The default is based off of
                nyquist_factor.

    Attributes
    ----------
    lc : lightkurve.LightCurve
        Time Series to analyze. Includes columns:
            - "time" (units of days)
            - "flux" (original light curve, input units preserved)
            - "model" (current model sampled as the data; same units as "flux")
            - "resid" (residuals (flux - model); same units as "flux")
    fitvalues : pandas.DataFrame
        Best-fit values from most recent fit.
    """
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
        """Configure units to user's preferences.

        Parameters
        ----------
        amp_unit : str, optional
            Amplitude unit to use, from ['relative', 'percent', 'ppt',
                                         'ppm', 'mma']
        freq_unit : str, optional
            Frequency unit to use, from ['muhz', 'uhz', 'microhertz',
                                         '1/d', '1/day', 'day',
                                         'days', 'd', 'per day']
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
        meanflux = float(np.nanmean(self.lc.flux.value))
        lc = lk.LightCurve(time = self.lc.time,
                           flux = (meanflux + self.sample_model(self.lc.time.value))
                                               *self.lc.flux.unit)
        lc["include"] = self.lc["include"]
        return lc
    
    @property
    def lc_resid(self):
        lc_model = self.lc_model
        lc = lk.LightCurve(time = self.lc.time,
                           flux = self.lc["flux"] - lc_model.flux,
                           flux_err = self.lc["flux_err"]) 
        lc["include"] = self.lc["include"]
        return lc
    
    @property
    def fitvalues(self):
        return self._fitvalues   # read-only

    @property
    def uptodate(self):
        colcompare = [c for c in self.columns if c != "brute"]
        nobrute = all(~self.stagedvalues["brute"])
        stagedisfit = np.array_equal(self.stagedvalues[colcompare].values, self.fitvalues[colcompare].values)
        return bool(nobrute and stagedisfit and not self._lcchanged)

    def _init_log(self):
        """Set up stuff needed for the Log."""
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
        """Record a message in the Log.

        Parameters
        ----------
        message : str
            The message to record in the log.
        level : TYPE, optional
            Message level/type. Options are 'debug', 'info', 'warning',
            'error', and 'critical'. The default is 'info'.

        Returns
        -------
        None.

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
    def get_log_html(self):
        raw_log = self._log_capture_string.getvalue()
        return ("<pre style='white-space: pre-wrap; "
                "font-family: monospace; "
                "margin: 0;'>"
                f"{html.escape(raw_log)}"
                "</pre>")

    def _log_lc_properties(self):
        """If lc has metadata, put it in the log."""
        keys = self.lc.meta.keys()
        if len(keys) > 0:
            self.log("The provided light curve has the following metadata:")
            for key in keys:
                self.log(f"{key}: {self.lc.meta[key]}")

    def _log_per_properties(self, per):
        """Capture periodogram properties in log."""
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
        """Set the frequency sampling for periodograms.

        The frequency sampling to use for periodograms can be set:
            - explicitly via the frequency keyword,
            - with a set oversampling factor out to a multiple of the
              approximate Nyquist frequency (see further comments about this),
            - with a set oversampling factor between a minimum and maximum
              frequency.

        Note that the Nyquist frequency is estimated to equal 1/(2*dt), where
        dt is the median time separation between adjacent samples. This is only
        valid for evenly sampled data, and may be a very poor approximation for
        unevenly sampled data. The Pyriod attribute `nyquist_quality` records,
        between 0-1, how strongly signals are reflected across the Pyriod
        attribute `nyquist`.

        Parameters
        ----------
        frequency : array, optional
            Explicit set of frequencies to compute periodogram at. The default
            is None.
        oversample_factor : FLOAT, optional
            How many times more densely than the natural frequency resolution
            of 1/duration to sample frequencies. The default is 5.
        nyquist_factor : FLOAT, optional
            How many time beyond the approximate Nyquist frequency to sample
            periodograms. The default is 1. Overridden by maxfreq, if provided.
            Note that the Nyquist frequency is estimated to equal 1/(2*dt),
            where dt is the median time separation between adjacent samples.
            This is only valid for evenly sampled data, and may be a very poor
            approximation for unevenly sampled data.
        minfreq : FLOAT
            Minimum frequency of range to use. The default is 1/duration.
        maxfreq : FLOAT
            Maximum frequency of range to use. The default is based off of
            nyquist_factor.

        Returns
        -------
        None.
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
        # Are we using user-speficied frequencies?
        if frequency is not None:
            self.log(f'Using user supplied frequency sampling: '
                     f'{len(frequency)} samples between frequency '
                     f'{np.min(frequency)} and {np.max(frequency)} '
                     f'{self._freq_label}')
            self.freqs = frequency
            self.fres = None
            self.oversample_factor = None
            self.nyquist_factor = np.max(frequency)/self.nyquist
        else: # If making our own frequency grid
            # Frequency resolution
            self.fres = 1./(self.freq_conversion*np.ptp(self.lc.time.value))
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
        """Get next n unused independent signal indices."""
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
        """Make a new sinusoidal signal available to the model to be fit.

        Can be used to add an individual or multiple signals. Pass single
        values for a single signal. For multiple, pass iterables of values.
        Any scaler values provided will be copied for all of multiple signals.

        Parameters
        ----------
        freq : float (or iterable of floats)
            Starting frequency of new signal(s).
        amp : float (or iterable of floats), optional
            Starting amplitude of new signal(s). The default is None. If None,
            starting amplitude with be set to 1.
        phase : float (or iterable of floats), optional
            Starting phase of new signals. The default is None.
        fixfreq : bool (or iterable of bools), optional
            Don't allow frequency to vary when fitting? The default is False.
        fixamp : bool (or iterable of bools), optional
            Don't allow amplitude to vary when fitting? The default is False.
        fixphase : bool (or iterable of bools), optional
            Don't allow phase to vary when fitting? The default is False.
        include : bool (or iterable of bools), optional
            Include this signal in the next model evaluation? The default is
            True.
        brute : bool (or iterable of bools), optional
            Brute force estimate phase? The default is True.
        index : str, optional
            Label to use for signal. Defaults to next available 'f#'.
            Duplicating an existing index label raises ValueError.

        Raises
        ------
        ValueError
            If index provided duplicates existing index.

        Returns
        -------
        None.

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
        """
        Convert 'f0+2*f1' into an lmfit expression like 'f0freq+2*f1freq'.
        """
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
        """Check that provided combination string is a valid expression."""
        known_labels = set(map(str.lower, self.stagedvalues.index))
        return validate_combination(combostr, known_labels)

    def add_combination(self, combostr, amp=None, phase=None, fixamp=False,
                        fixphase=False, include=True, brute=True,
                        index=None):
        """Make a new combination signal available to the model to be fit.

        Can be used to add an individual or multiple combinations. Pass single
        values for a single signal. For multiple, pass iterables of values.
        Any scaler values provided will be copied for all of multiple signals.

        Parameters
        ----------
        combostr : str (or iterable of str)
            Arithmetic expression for signal frequency terms of existing signal
            indices.
        amp : float (or iterable of floats), optional
            Starting amplitude of new signal(s). The default is None. If None,
            starting amplitude with be set to 1.
        phase : float (or iterable of floats), optional
            Starting phase of new signals. The default is None.
        fixamp : bool (or iterable of bools), optional
            Don't allow amplitude to vary when fitting? The default is False.
        fixphase : bool (or iterable of bools), optional
            Don't allow phase to vary when fitting? The default is False.
        include : bool (or iterable of bools), optional
            Include this signal in the next model evaluation? The default is
            True.
        brute : bool (or iterable of bools), optional
            Brute force estimate phase? The default is False.
        index : str, optional
            Label to use for signal. Defaults to next available 'f#'.
            Duplicating an existing index label raises ValueError.

        Raises
        ------
        ValueError
            If index provided duplicates existing index.

        Returns
        -------
        None.

        """
        combostr, amp, phase, fixamp, fixphase, include, brute, index = (
            make_all_iter([combostr, amp, phase, fixamp, fixphase,
                                 include, brute, index]))
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
            except CombinationExpressionError as exc:
                self.log(f"Invalid combination string provided in {combostr}.","error")
        else:
            self.log(f"Invalid combination string provided in {combostr}.","error")

    def _brute_phase_est(self, freq, amp, brute_step=0.1):
        """Estimate phase by brute force sampling.

        Fits a single sinusoid to residuals, sampling phase between 0-1 in
        steps of brute_step.

        Parameters
        ----------
        freq : float
            Fixed sinusoid frequency for phase estimation.
        amp : float
            Fixed sinusoid amplitude for phase estimation.
        brute_step : float, optional
            Step size in phase between 0 and 1. The default is 0.1.

        Returns
        -------
        float
            Estimated phase (multiple of brute_step between 0 and 1)

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

    def fit_model(self, *args):
        """Optimize fit for model with all included signals.

        The model fit is optimized with the lmfit package. The model is a sum
        of sinusoids, one for each included signal. Initial fitting values are
        taken from the Pyriod.stagedvalues attribute, which contains the same
        information as the interactive Signals pane in GUI mode. New signals
        (or those with brute=True set) will have an initial phase estimated
        from brute-force sampling. The frequencies of combination signals are
        constrained to relate to independent signal frequencies following
        arithmetic expressions. The new best-fit parameters are stored in the
        fitvalues attribute, and the lmfit report is stores in the fit_report
        attribute.
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
        """Update dataframe of params with new values from fit."""
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
        # Add periods and period uncertainties
        if ((self.noise_spectrum is not None) &
            (self.significance_multiplier is not None)):
            self._fitvalues['snr'] = (
                self.amp_conversion * self._fitvalues['amp'] /
                self.noise_spectrum(self._fitvalues['freq']))

    def _convert_fitvalues_to_qgrid(self):
        tempdf = self._fitvalues.copy()
        tempdf["brute"] = False
        tempdf = tempdf.astype(
            dtype=dict(zip(self.columns, self.dtypes)))[self.columns]
        tempdf["amp"] *= self.amp_conversion
        tempdf["amperr"] *= self.amp_conversion
        return tempdf

    def _set_stagedvalues(self, df):
        self.stagedvalues = df

    def sample_model(self, time):
        """Sample the current best fit model at desired times.

        Parameters
        ----------
        time : iterable
            time of samples in days

        Returns
        -------
        flux : array
            Model evaluated at provided times, in units of input time series.
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
        """
        Drop provided indices from stagedvalues and Signals table.

        Signals will not be dropped from the current model until a new fit is
        performed.

        Parameters
        ----------
        indices : str, or iterable of str
            Indices to drop from stagedvalues.

        Returns
        -------
        None.
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

    def _void_combos(self):
        #remove all invalid combinations
        isindep = lambda key: key[1:].isdigit()
        depkeys = []
        for key in self.stagedvalues.index:
            if not isindep(key) and not self._valid_combo(key):
                self.remove_signals(key)

    def _initialize_dataframe(self):
        """Create new empty signals dataframe."""
        df = (pd.DataFrame(columns=self.columns)
              .astype(dtype=dict(zip(self.columns, self.dtypes))))
        return df
    
    def mask_indices(self, indices, threshold=30):
        self.log(f"Masking {len(indices)} selected points: "+
                 f"{np.array2string(indices,threshold=threshold)}")
        self.lc["include"][indices] = 0
        self._mask_changed()

    def clear_mask(self):
        self.log("Restoring all masked points.")
        self.lc["include"][:] = 1
        self._mask_changed()

    def _mask_changed(self):
        self._calc_tshift()
        self.compute_pers(orig=True)
        self._lcchanged = True

    def _calc_tshift(self, tshift=None):
        # Subtracting the mean time stabilizes phase fitting.
        if tshift is None:
            good = np.where(self.lc["include"])
            self.tshift = -np.mean(self.lc[good].time.value)
        else:
            self.tshift = tshift
        self.log(f'Fitted timstamps will be shifted forward relative to '
                 f'given timestamps by `tshift` {self.tshift} days.')

    def compute_pers(self, orig=False):
        """Compute periodograms of the various time series.

        Parameters
        ----------
        orig : bool, optional
            Whether it also (re)calculate the periodogram of the original time
            series. The default is False.

        Returns
        -------
        None.
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
        """
        Calculate amplitude threshold for considering a signal to be
        significant. There are two parts: self.noise_spectrum is an
        interpolation function for the average (mean or median) amplitude
        calculated in a moving frequency window across the residuals
        periodogram; and self.significance_multiplier is a scaling factor for
        converting this to a significance threshold.

        Parameters
        ----------
        multiplier : float, optional
            Factor above local average to multiply significance threshold by.
            The default is 5.
        startfreq : float, optional
            Lowest frequency to start calculation. The default is 0. The first
            averaging window will be centered on this frequency.
        endfreq : float, optional
            Highest frequency for calculating significance threshold. The last
            averaging window will be centered on this frequency. The default is
            None, corresponding to the highest frequency in the periodogram.
        freqstep : float, optional
            Window step size in frequency units. The default is 100.
        winwidth : float, optional
            Width of averaging window in frequency units. The default is 100.
        avgtype : str, optional
            "mean" or "median". The default is "mean".
        autorecalculate : bool, optional
            recalculate threshold with these settings each time periodogram changes.
            
        **kwargs :
            keyword arguments passed to interpolate function. `fill_value`
            determines how or whether to extrapolate beyond sampled frequency
            range.

        Returns
        -------
        None.

        """
        # Store arguments for reference or recalculation
        self.significance_settings = {"multiplier":multiplier, 
                                      "startfreq":startfreq,
                                      "endfreq":endfreq,
                                      "freqstep":freqstep,
                                      "winwidth":winwidth,
                                      "avgtype":avgtype,
                                      "autorecalculate":autorecalculate}

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
        """Save current signal solution as csv file.

        Parameters
        ----------
        filename : str, optional
            Filename for saving signals solutions as csv file. The default is
            'Pyriod_solution.csv'.
        """
        self.log("Writing signal solution to " + os.path.abspath(filename))
        self._convert_fitvalues_to_qgrid().to_csv(filename,
                                                  index_label='label')

    def load_solution(self, filename='Pyriod_solution.csv'):
        """Load a saved signal solution from a csv file.

        Parameters
        ----------
        filename : str, optional
            Filename of csv file to load saved signals solution from. The
            default is 'Pyriod_solution.csv'.
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
        """Save log to text file.

        Parameters
        ----------
        filename : str, optional
            Filename for saving the log.
        overwrite : bool, optional
            Whether to overwrite an existing log named filename. The default is
            False.
        """
        logmessage = "Writing log to "+os.path.abspath(filename)
        if overwrite:
            logmessage += ", overwriting."
        self.log(logmessage)
        loghtml = self.get_log_html
        soup = BeautifulSoup(loghtml, features="xml")
        mode = {True: "w+", False: "a+"}[overwrite]
        f = open(filename, mode)
        f.write(soup.get_text().replace('|', ''))
        f.close()

    ### Advanced Features ###

    def spectral_window(self, maxfreq=100, osample=10):
        """Compute the spectral window for these data.

        This method uses the discrete Fourier transform instead of the fast
        Lomb-Scargle implementation that the rest of Pyriod uses, in order to
        accurately represent the spectral window.

        Parameters
        ----------
        maxfreq : float, optional
            Maximum frequency to calculate the spectral window out to. The
            default is 100.
        osample : float, optional
            The oversample factor to compute the spectral window with, relative
            to the natural frequency resolution of 1/(time span of the data).
            The default is 10.

        Returns
        -------
        freqvec : array
            Frequencies where spectral window was calculated.
        ampvec : array
            Corresponding amplitude of the spectral window.
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