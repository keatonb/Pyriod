"""Pyriod, an astronomical prewhitening frequency analysis package.

Written by Keaton Bell

For more, see https://github.com/keatonb/Pyriod

---------------------

A couple of parts were "borrowed" from Stackoverflow:

Distinguish clicks with drag motions from ImportanceOfBeingErnest
https://stackoverflow.com/a/48452190

Capturing print output from kindall
https://stackoverflow.com/a/16571630

---------------------
"""

# Standard imports
import sys
import os
import itertools
import re
import logging
from pathlib import Path
if sys.version_info < (3, 0):
    from StringIO import StringIO
else:
    from io import StringIO

# Third party imports
import numpy as np
import gh_md_to_html
import pandas as pd
from scipy.interpolate import interp1d
import astropy.units as u
from astropy.timeseries import TimeSeries
import lightkurve as lk
from lmfit import Model, Parameters
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
import matplotlib.path  # for .Path
import ipywidgets as widgets
from ipywidgets import HBox, VBox
import qgrid
from ipyfilechooser import FileChooser

# Local imports
# from .pyquist import subfreq (not currently used)

plt.ioff()  # Turn off interactive mode


# Definition of the basic model we fit
def sin(x, freq, amp, phase):
    """Model fit to time series."""
    return amp*np.sin(2.*np.pi*(freq*x+phase))


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


class lasso_selector(object):
    """Select indices from a matplotlib collection using `LassoSelector`.

    Outline selected points with given color, otherwise don't outline

    Based on Lasso Selector Demo
    https://matplotlib.org/3.1.1/gallery/widgets/lasso_selector_demo_sgskip.html
    """

    def __init__(self, ax, collection, color='gold'):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.color = color

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

    def onselect(self, verts):
        path = matplotlib.path.Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]

        ec = np.array(["None" for i in range(self.Npts)])
        ec[self.ind] = self.color
        self.collection.set_edgecolors(ec)
        self.canvas.draw_idle()

    def update(self,collection):
        self.collection = collection
        self.xys = collection.get_offsets()

    def disconnect(self):
        self.lasso.disconnect_events()
        ec = np.array(["None" for i in range(self.Npts)])
        self.collection.set_edgecolors(ec)
        self.canvas.draw_idle()


class Pyriod(object):
    """Time series class for prewhitening frequency analysis.

    Parameters
    ----------
        lc : lightkurve.LightCurve
            light curve data to analyze
        amp_unit : ("ppt", "percent", etc)
            amplitude unit to use
        freq_unit: ("muHz" or "perday")
            frequency unit to use
        gui: (bool)

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

    # Generate unique ID for this Pyriod instance
    id_generator = itertools.count(0)

    def __init__(self, lc, amp_unit='ppt', freq_unit='muHz', gui=True,
                 **kwargs):
        # Generate unique Pyriod instance ID
        self.id = next(self.id_generator)
        self.gui = gui

        ### LOG ###
        # Initialize the log first
        self._init_log()

        # Work out the units, in a function
        self._set_units(amp_unit=amp_unit, freq_unit=freq_unit)

        # Create status widget to indicate when calculations are running
        self._status = widgets.HTML(value="")

        self.fit_result = None  # will be replaced as we do fits

        ### TIME SERIES ###
        # Stored as lightkurve.LightCurve object
        # "flux" column is original data
        # "resid" is residuals
        # "include" is included points
        # "model_sampled" is the model sampled as the data
        # A separate LightCurve object holds a model that is better sampled

        # Input must be Lightkurve LightCurve type
        if not issubclass(type(lc), lk.LightCurve):
            raise TypeError('lc must be a lightkurve.LightCurve object.')
        self.lc = lc.copy()  # copy so we don't modify original

        # Check for nans and remove if needed
        if np.sum(np.isnan(np.array(self.lc.flux.value))) > 0:
            self.log("Removing nans from light curve.")
            self.lc = self.lc.remove_nans()

        # Maintain a mask of points to exclude from analysis
        self.lc["include"] = np.ones(len(self.lc))  # 1 = include
        self.include = np.where(self.lc["include"])

        # Establish frequency sampling
        self.set_frequency_sampling(**kwargs)

        # Initialize time series widgets and plots (if in GUI mode)
        if self.gui:
            self._init_timeseries_widgets()
            self.lcfig, self.lcax = plt.subplots(
                figsize=(7, 2), num='Time Series ({:d})'.format(self.id))
            self.lcax.set_position([0.13, 0.22, 0.85, 0.76])
            self._lc_colors = {0: "bisque", 1: "C0"}
            self.lcplot_data = self.lcax.scatter(
                self.lc.time.value, self.lc.flux.value, marker='o',
                s=5, ec='None', lw=1, c=self._lc_colors[1])
            # Define selector for masking points
            self.selector = lasso_selector(self.lcax, self.lcplot_data)
            self.lcfig.canvas.mpl_connect("key_press_event",
                                          self._mask_selected_pts)

        # Apply time shift to get phases to be well behaved
        self._calc_tshift()

        # Store version sampled as the data as lc column
        initmodel = (np.zeros(len(self.lc))*self.lc.flux.unit
                     + np.mean(self.lc.flux[self.include]))
        self.lc["model"] = initmodel
        # Also plot the model over the time series
        if self.gui:
            dt = np.min(np.diff(sorted(lc.time.value)))
            tspan = (np.max(lc.time.value) - np.min(lc.time.value))
            osample = 2
            nsamples = int(round(osample*tspan/dt))
            time_samples = TimeSeries(time_start=np.min(lc.time),
                                      time_delta=dt * u.day / osample,
                                      n_samples=nsamples).time
            initmodel = np.zeros(nsamples)+np.mean(np.array(self.lc.flux))
            self.lc_model_sampled = lk.LightCurve(time=time_samples,
                                                  flux=initmodel)
            self.lcplot_model, = self.lcax.plot(
                self.lc_model_sampled.time.value,
                self.lc_model_sampled.flux,
                c='r', lw=1, alpha=0.7)

        # And keep track of residuals time series
        self.lc["resid"] = self.lc["flux"] - self.lc["model"]


        ### PERIODOGRAM ###
        # Four types for display
        # Original (orig), Residuals (resid), Model (model),
        # and Spectral Window (sw; TODO)
        # Each is stored as, e.g., "per_orig", samples at self.freqs
        # Has associated plot _perplot_orig
        # Display toggle widget _perplot_orig_display
        # TODO: Add color picker _perplot_orig_color

        # Compute original periodogram
        self.compute_pers(orig=True)

        # Make interpolator for residual periodogram
        self.interpls = interp1d(self.freqs, self.per_resid.power.value)

        # Initialize widgets and plot
        if self.gui:
            self._init_periodogram_widgets()

            # Set up figs/axes for periodogram plots
            self.perfig, self.perax = plt.subplots(
                figsize=(7, 3), num='Periodogram ({:d})'.format(self.id))

            self.perplot_orig, = self.perax.plot(
                self.per_orig.frequency, self.per_orig.power.value,
                lw=1, c='tab:gray')
            self.perax.set_ylim(0, 1.05*np.nanmax(self.per_orig.power.value))
            self.perax.set_xlim(np.min(self.freqs), np.max(self.freqs))
            self.perax.set_position([0.13, 0.22, 0.8, 0.76])

            # Plot periodogram of sampled model and residuals
            self.perplot_model, = self.perax.plot(self.freqs,
                                                  self.per_model.power.value,
                                                  lw=1, c='tab:green')
            self.perplot_resid, = self.perax.plot(self.freqs,
                                                  self.per_resid.power.value,
                                                  lw=1, c='tab:blue')

            # Create markers for selected peak, adopted signals
            self.marker = self.perax.plot([0], [0], c='k', marker='o')[0]
            self._signal_marker_color = 'green'
            self.signal_markers, = self.perax.plot([], [], marker='D',
                                                   fillstyle='none',
                                                   linestyle='None',
                                                   c=self._signal_marker_color,
                                                   ms=5)
            self._combo_marker_color = 'orange'
            self.combo_markers, = self.perax.plot([], [], marker='D',
                                                  fillstyle='none',
                                                  linestyle='None',
                                                  c=self._combo_marker_color,
                                                  ms=5)

            #self._makeperiodsolutionvisible()
            self._display_per_orig()
            self._display_per_resid()
            self._display_per_model()
            self._display_per_markers()
            self._mark_highest_peak()

            # This handles clicking while zooming problems
            #self.perfig.canvas.mpl_connect('button_press_event', self._onperiodogramclick)
            self._press = False
            self._move = False
            self.perfig.canvas.mpl_connect('button_press_event', self._onpress)
            self.perfig.canvas.mpl_connect('button_release_event',
                                           self._onrelease)
            self.perfig.canvas.mpl_connect('motion_notify_event', self._onmove)

            # Set axis labels
            self._set_plot_labels()

        ### SIGNALS ###

        # Hold signal phases, frequencies, and amplitudes in Pandas DF
        self.stagedvalues = self._initialize_dataframe()
        self.fitvalues = self.stagedvalues.copy().drop('brute', axis=1)

        # The interface for interacting with the values DataFrame:
        if self.gui:
            self._init_signals_qgrid()
            self.signals_qgrid = self._get_qgrid()
            self.signals_qgrid.on('cell_edited', self._qgrid_changed_manually)
            self._init_signals_widgets()
            self._update_fit_report()  # No fit to report

        self.log("Pyriod object initialized.")

        # Write lightkurve and periodogram properties to log
        self._log_lc_properties()
        self._log_per_properties()

        # Keep track whether the displayed data reflect the most recent fit
        self.uptodate = True

        # Create decoy figure so users don't plot over Pyriod ones
        if self.gui:
            _ = plt.figure()

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
            self.freq_label = {perday: "1/day", muHz: "muHz"}[self.freq_unit]
            self.log(f'Frequency unit set to {self.freq_label}.')
        self.time_unit = u.day
        self.freq_conversion = self.time_unit.to(1/self.freq_unit)

    def _set_plot_labels(self):
        # Light curve labels
        self.lcax.set_xlabel(f"time ({self.time_unit.to_string()})")
        self.lcax.set_ylabel("rel. variation")
        # Periodogram labels
        self.perax.set_ylabel(f"amplitude ({self.amp_unit})")
        self.perax.set_xlabel(f"frequency ({self.freq_label})")

    def _init_timeseries_widgets(self):
        """Set up Time Series widgets."""
        # Plot location file chooser
        self._tsfig_file_location = FileChooser(
            os.getcwd(),
            filename='Pyriod_TimeSeries.png',
            show_hidden=False,
            select_default=True,
            use_dir_icons=True,
            show_only_dirs=False
        )

        # Save figure button
        self._save_tsfig = widgets.Button(
            description="Save",
            disabled=False,
            tooltip='Save currently displayed figure to file.',
            icon='save'
        )
        self._save_tsfig.on_click(self._save_tsfig_button_click)

        # Reset masked points button
        self._reset_mask = widgets.Button(
            description='Reset mask',
            disabled=False,
            tooltip='Include all points in calculations',
            icon='refresh'
        )
        self._reset_mask.on_click(self._clear_mask)

        # Dropdown for which time series to display
        self._tstype = widgets.Dropdown(
            options=['Original', 'Residuals'],
            value='Original',
            description='Display:',
            disabled=False
        )
        self._tstype.observe(self._update_lc_display)

        # Fold on frequency checkbox
        self._fold = widgets.Checkbox(
            value=False,
            step=self.fres,
            description='Fold time series on frequency?',
        )
        self._fold.observe(self._update_lc_display)

        # Folding frequency
        self._fold_on = widgets.FloatText(
            value=1.,
            description='Fold on freq:'
        )
        self._fold_on.observe(self._update_lc_display)

        # Select folding frequency from list
        self._select_fold_freq = widgets.Dropdown(
            description='Select from:',
            disabled=False,
        )
        self._select_fold_freq.observe(self._fold_freq_selected, 'value')

        # Readme HTML widget
        path = Path(__file__).parent / 'docs/TimeSeries.md'
        html = gh_md_to_html.main(str(path))
        self._timeseries_readme = widgets.HTML(html)

    def _init_periodogram_widgets(self):
        """Set up Periodogram widgets."""
        # Plot location file chooser
        self._perfig_file_location = FileChooser(
            os.getcwd(),
            filename='Pyriod_Periodogram.png',
            show_hidden=False,
            select_default=True,
            use_dir_icons=True,
            show_only_dirs=False
        )

        # Save figure button
        self._save_perfig = widgets.Button(
            description="Save",
            disabled=False,
            tooltip='Save currently displayed figure to file.',
            icon='save'
        )
        self._save_perfig.on_click(self._save_perfig_button_click)

        # Frequency to add for next signal
        self._thisfreq = widgets.Text(
            value='',
            placeholder='',
            description='Frequency:',
            disabled=False
        )

        # Amplitude to add for next signal
        self._thisamp = widgets.FloatText(
            value=0.001,
            description='Amplitude:',
            disabled=False
        )

        # Button to add signal to the solutions table
        self._addtosol = widgets.Button(
            description='Add to solution',
            disabled=False,
            button_style='success',
            tooltip=('Click to add currently selected values '
                     'to frequency solution'),
            icon='plus'
        )
        self._addtosol.on_click(self._add_staged_signal)

        # Checkbox, snap to peaks?
        self._snaptopeak = widgets.Checkbox(
            value=True,
            description='Snap clicks to peaks?',
            disabled=False
        )

        # Checkbox, show markers?
        self._show_per_markers = widgets.Checkbox(
            value=True,
            description='Signal Markers',
            disabled=False,
            style={'description_width': 'initial'}
        )
        self._show_per_markers.observe(self._display_per_markers)

        # Checkboxes, show original periodogram?
        self._show_per_orig = widgets.Checkbox(
            value=False,
            description='Original',
            disabled=False,
            style={'description_width': 'initial'}
        )
        self._show_per_orig.observe(self._display_per_orig)

        # Checkboxes, show residuals periodogram?
        self._show_per_resid = widgets.Checkbox(
            value=True,
            description='Residuals',
            disabled=False,
            style={'description_width': 'initial'}
        )
        self._show_per_resid.observe(self._display_per_resid)

        # Checkboxes, show model periodogram?
        self._show_per_model = widgets.Checkbox(
            value=True,
            description='Model',
            disabled=False,
            style={'description_width': 'initial'}
        )
        self._show_per_model.observe(self._display_per_model)

        # Readme HTML widget
        path = Path(__file__).parent / 'docs/Periodogram.md'
        html = gh_md_to_html.main(str(path))
        self._periodogram_readme = widgets.HTML(html)

    def _init_signals_qgrid(self):
        """Define QGrid column properties."""
        # Overall grid options
        self._gridoptions = {
            # SlickGrid options
            'fullWidthRows': True,
            'syncColumnCellResize': True,
            'forceFitColumns': False,
            'defaultColumnWidth': 65,  # col width (all the same)
            'rowHeight': 28,
            'enableColumnReorder': True,
            'enableTextSelectionOnCells': True,
            'editable': True,
            'autoEdit': True,  # double-click not required!
            'explicitInitialization': True,

            # Qgrid options
            'maxVisibleRows': 8,
            'minVisibleRows': 8,
            'sortable': True,
            'filterable': False,  # Not useful here
            'highlightSelectedCell': False,
            'highlightSelectedRow': True
             }

        # Individual column options
        self._column_definitions = {
            "include":  {'width': 60,
                         'toolTip': "include signal in model fit?"},
            "freq":      {'width': 100, 'toolTip': "mode frequency"},
            "fixfreq":  {'width': 60, 'toolTip': "fix frequency during fit?"},
            "freqerr":  {'width': 90, 'toolTip': "uncertainty on frequency",
                         'editable': False},
            "amp":       {'width': 100, 'toolTip': "mode amplitude"},
            "fixamp":   {'width': 60, 'toolTip': "fix amplitude during fit?"},
            "amperr":  {'width': 90, 'toolTip': "uncertainty on amplitude",
                        'editable': False},
            "phase":     {'width': 100, 'toolTip': "mode phase"},
            "brute": {'width': 65,
                      'toolTip': "brute sample phase first during fit?"},
            "fixphase": {'width': 65, 'toolTip': "fix phase during fit?"},
            "phaseerr":  {'width': 90, 'toolTip': "uncertainty on phase",
                          'editable': False}}

    def _init_signals_widgets(self):
        """Set up Signals widgets."""
        # Button to compute best fit
        self._refit = widgets.Button(
            description="Refine fit",
            disabled=False,
            tooltip='Refine fit of signals to time series',
            icon='refresh'
        )
        self._refit.on_click(self.fit_model)

        # Button to delete selected signal rows
        self._delete = widgets.Button(
            description='Delete selected',
            disabled=False,
            button_style='danger',
            tooltip='Delete selected rows.',
            icon='trash'
        )
        self._delete.on_click(self._delete_selected)

        # Save signals file chooser
        self._signals_file_location = FileChooser(
            os.getcwd(),
            filename='Pyriod_solution.csv',
            show_hidden=False,
            select_default=True,
            use_dir_icons=True,
            show_only_dirs=False
        )

        # Save signals table as csv file
        self._save = widgets.Button(
            description="Save",
            disabled=False,
            tooltip='Save solution to csv file.',
            icon='save'
        )
        self._save.on_click(self._save_button_click)

        # Load signals from a csv file
        self._load = widgets.Button(
            description="Load",
            disabled=False,
            tooltip='Load solution from csv file.',
            icon='load'
        )
        self._load.on_click(self._load_button_click)

        # HTML widget to display fit result details
        self._fit_result_html = widgets.HTML(" ")

        # HTML widget to display Readme
        path = Path(__file__).parent / 'docs/Signals.md'
        html = gh_md_to_html.main(str(path))
        self._signals_readme = widgets.HTML(html)

    def _init_log(self):
        """Set up stuff needed for the Log."""
        self.logger = logging.getLogger('basic_logger')
        self.logger.setLevel(logging.DEBUG)
        self.log_capture_string = StringIO()
        ch = logging.StreamHandler(self.log_capture_string)
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        if self.gui:
            # Log HTML widget
            self._log = widgets.HTML(
                value='Log',
                placeholder='Log',
                description='Log:',
                layout={'height': '250px', 'width': '950px'}
            )

            # Save log location file chooser
            self._log_file_location = FileChooser(
                os.getcwd(),
                filename='Pyriod_log.txt',
                show_hidden=False,
                select_default=True,
                use_dir_icons=True,
                show_only_dirs=False
            )

            # Save log button
            self._save_log = widgets.Button(
                description="Save",
                disabled=False,
                tooltip='Save log to csv file.',
                icon='save'
            )
            self._save_log.on_click(self._save_log_button_click)

            # Overwrite checkbox
            self._overwrite = widgets.Checkbox(
                value=False,
                description='Overwrite?'
            )

            # Layout Log widgets
            self._logbox = VBox(
                [widgets.Box([self._log]),
                 HBox([self._save_log,
                       self._log_file_location,
                       self._overwrite],
                 layout={'height': '40px'})],
                layout={'height': '300px', 'width': '950px'})

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
            'debug': self.logger.debug,
            'info': self.logger.info,
            'warning': self.logger.warning,
            'error': self.logger.error,
            'critical': self.logger.critical
            }
        logdict[level](message+'<br>')
        if self.gui:
            self._update_log()

    def _log_lc_properties(self):
        """If lc has metadata, put it in the log."""
        for key in self.lc.meta.keys():
            self.log(f"{key}: {self.lc.meta[key]}")

    def _log_per_properties(self):
        """Capture periodogram properties in log."""
        try:
            with Capturing() as output:
                self.per_resid.show_properties()
            info = re.sub(' +', ' ',
                          str("".join([e+' |\n' for e in output[3:]])))
            self.log("Periodogram properties:" + info)
        except Exception:
            pass

    def _next_signal_index(self, n=1):
        """Get next n unused independent signal indices."""
        inds = []
        i = 0
        while len(inds) < n:
            if not "f{}".format(i) in self.stagedvalues.index:
                inds.append("f{}".format(i))
            i += 1
        return inds

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
        # Frequency resolution
        self.fres = 1./(self.freq_conversion*np.ptp(self.lc.time.value))
        self.oversample_factor = oversample_factor
        self.nyquist_factor = nyquist_factor
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
        # Sample the following frequencies:
        if frequency is not None:
            self.log(f'Using user supplied frequency sampling: '
                     f'{len(frequency)} samples between frequency '
                     f'{np.min(frequency)} and {np.max(frequency)} '
                     f'{self.freq_label}')
            self.freqs = frequency
        else:
            if minfreq is None:
                minfreq = self.fres
            if maxfreq is None:
                maxfreq = (self.nyquist*self.nyquist_factor
                           + 0.9*self.fres/self.oversample_factor)
            self.freqs = np.arange(minfreq, maxfreq,
                                   self.fres/self.oversample_factor)
        return

    # Functions for interacting with model fit below

    def _make_all_iter(self, variables):
        """Return iterables of given variables.

        Parameters
        ----------
        variables : list or tuple
            Set of values to return as iterables if necessary.
            Each must have length 1 or length of first variable
        Returns
        -------
        tuple of iterable versions of input variables
        """
        # Wrap all single values or strings in lists
        variables = [[v] if (not hasattr(v, '__iter__')) or (type(v) == str)
                     else v for v in variables]
        # Get length of first variable
        nvals = len(variables[0])
        # check that all lengths are the same or 1
        if not all([len(l) in [nvals, 1] for l in variables]):
            raise ValueError("Arguments passed have inconsistent lengths.")
        else:
            variables = [[v[0] for i in range(nvals)] if (len(v) == 1)
                         else v for v in variables]
        return tuple(variables)

    def add_signal(self, freq, amp=None, phase=None, fixfreq=False,
                   fixamp=False, fixphase=False, include=True, brute=False,
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
        freq, amp, phase, fixfreq, fixamp, fixphase, include, brute, index = (
            self._make_all_iter([freq, amp, phase, fixfreq, fixamp, fixphase,
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
        if self.gui:
            self._update_freq_dropdown()  # For folding time series
            displayframe = self.stagedvalues.copy()
            displayframe["amp"] = displayframe["amp"] * self.amp_conversion
            self.signals_qgrid.df = displayframe.combine_first(
                self.signals_qgrid.df)  # Update qgrid displayed values
            self._update_signal_markers()
        self.log(f"Signal {index} added to model with frequency "
                 f"{freq} and amplitude {amp}.")
        self._model_current(False)

    def _valid_combo(self, combostr):
        """Check that provided combination string is a valid expression."""
        parts = re.split('\+|\-|\*|\/', combostr.replace(" ", "").lower())
        allvalid = np.all([(part in self.stagedvalues.index)
                           or part.replace('.', '', 1).isdigit()
                           for part in parts])
        return allvalid and (len(parts) > 1)

    def add_combination(self, combostr, amp=None, phase=None, fixamp=False,
                        fixphase=False, include=True, brute=False,
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
            self._make_all_iter([combostr, amp, phase, fixamp, fixphase,
                                 include, brute, index]))
        freq = np.zeros(len(combostr))
        for i in range(len(combostr)):
            combostr[i] = combostr[i].replace(" ", "").lower()
            # Evaluate combostring, replacing keys with values.
            parts = re.split('\+|\-|\*|\/',
                             combostr[i].replace(" ", "").lower())
            keys = set([part for part in parts if part
                        in self.stagedvalues.index])
            exploded = re.split('(\+|\-|\*|\/)',
                                combostr[i].replace(" ", "").lower())
            expression = "".join([str(self.stagedvalues.loc[val, 'freq'])
                                  if val in keys else val for val in exploded])
            freq[i] = eval(expression)
            if amp[i] is None:
                amp[i] = self.interpls(freq[i])
        self.add_signal(list(freq), amp, phase, False, fixamp, fixphase,
                        include, brute, index=combostr)

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
        result = model.fit(
            (self.lc["resid"][self.include].value
             - np.mean(self.lc["resid"][self.include].value)),
            params,
            x=(self.lc.time.value[self.include]+self.tshift),
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
        # Indicate that a calculation is running
        self._update_status()

        # Check that there are signals in the model
        if np.sum(self.stagedvalues.include.values) == 0:
            self.log("No signals to fit.", level='warning')
            self.fitvalues = self._initialize_dataframe().drop('brute', axis=1)
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
                                    * self.stagedvalues.freq[prefix]))

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
                    parts = re.split('\+|\-|\*|\/', prefix)
                    keys = set([part for part in parts
                                if part in self.stagedvalues.index])
                    exploded = re.split('(\+|\-|\*|\/)', prefix)
                    expression = "".join([val+'freq' if val in keys else val
                                          for val in exploded])
                    params[useprefix+'freq'].set(expr=expression)
                    params[useprefix+'amp'].set(
                        self.stagedvalues.amp[prefix],
                        vary=~self.stagedvalues.fixamp[prefix])
                    # Correct phase for tdiff
                    thisphase = (self.stagedvalues.phase[prefix]
                                 - (self.tshift * self.freq_conversion
                                    * self.stagedvalues.freq[prefix]))
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

            # Fit the model
            self.fit_result = model.fit(
                (self.lc.flux.value[self.include]
                 - np.mean(self.lc.flux.value[self.include])),
                params, x=self.lc.time.value[self.include]+self.tshift,
                weights=1/self.lc.flux_err.value[self.include],
                scale_covar=False)

            self.log("Fit refined.")
            self.log("Fit properties:"+self.fit_result.fit_report())
            self._update_values_from_fit(self.fit_result.params, prefixmap)

        # Update lightcurves and periodograms for new residuals
        self._update_lcs()
        self.compute_pers()

        if self.gui:  # Update plots and displays
            self._update_lc_display()
            self._update_signal_markers()
            self._update_per_plots()
            self._mark_highest_peak()  # Mark highest peak in residuals
            self._update_fit_report()

        self._update_status(False)  # Calculation done
        self._model_current(True)  # fitvalues and stagedvalues are the same

    def _update_values_from_fit(self, params, prefixmap):
        """Update dataframe of params with new values from fit."""
        # Also rectify and negative amplitudes or phases outside [0,1)
        self.fitvalues = self.stagedvalues.astype(
            dtype=dict(zip(self.columns, self.dtypes))).drop('brute', axis=1)
        for prefix in self.stagedvalues.index[self.stagedvalues.include]:
            self.fitvalues.loc[prefix, 'freq'] = float(
                params[prefixmap[prefix]+'freq'].value/self.freq_conversion)
            self.fitvalues.loc[prefix, 'freqerr'] = float(
                params[prefixmap[prefix]+'freq'].stderr/self.freq_conversion)
            self.fitvalues.loc[prefix, 'amp'] = (
                params[prefixmap[prefix]+'amp'].value)
            self.fitvalues.loc[prefix, 'amperr'] = float(
                params[prefixmap[prefix]+'amp'].stderr)
            self.fitvalues.loc[prefix, 'phase'] = (
                params[prefixmap[prefix]+'phase'].value)
            self.fitvalues.loc[prefix, 'phaseerr'] = float(
                params[prefixmap[prefix]+'phase'].stderr)
            # Rectify negative amplitudes (with 0.5 phase change)
            if self.fitvalues.loc[prefix, 'amp'] < 0:
                self.fitvalues.loc[prefix, 'amp'] *= -1.
                self.fitvalues.loc[prefix, 'phase'] -= 0.5
            # Reference phase to t0, and make phase between 0-1
            self.fitvalues.loc[prefix, 'phase'] += (
                self.tshift*self.fitvalues.loc[prefix, 'freq']
                * self.freq_conversion)
            self.fitvalues.loc[prefix, 'phase'] %= 1.

        if self.gui:
            self._update_freq_dropdown()

        # Add periods and period uncertainties
        pers = 1./(self.fitvalues['freq']*self.freq_conversion)  # days
        pers = pers*24*3600  # seconds
        pererrs = pers*self.fitvalues['freqerr']/self.fitvalues['freq']
        self.fitvalues['per'] = pers
        self.fitvalues['pererr'] = pererrs

        # Update qgrid and staged values
        if self.gui:
            self.signals_qgrid.df = (
                self._convert_fitvalues_to_qgrid()
                .combine_first(self.signals_qgrid.get_changed_df()))
            self._update_stagedvalues_from_qgrid()
        else:
            tempdf = self.fitvalues.copy()
            tempdf["brute"] = False
            tempdf = tempdf.astype(
                dtype=dict(zip(self.columns, self.dtypes)))[self.columns]
            self.stagedvalues = tempdf

    def _convert_fitvalues_to_qgrid(self):
        tempdf = self.fitvalues.copy()
        tempdf["brute"] = False
        tempdf = tempdf.astype(
            dtype=dict(zip(self.columns, self.dtypes)))[self.columns]
        tempdf["amp"] *= self.amp_conversion
        tempdf["amperr"] *= self.amp_conversion
        return tempdf

    def _convert_qgrid_to_stagedvalues(self):
        tempdf = (self.signals_qgrid.get_changed_df().copy()
                  .astype(dtype=dict(zip(self.columns, self.dtypes))))
        tempdf["amp"] /= self.amp_conversion
        tempdf["amperr"] /= self.amp_conversion
        return tempdf

    def _update_stagedvalues_from_qgrid(self):
        self.stagedvalues = self._convert_qgrid_to_stagedvalues()
        self._update_signal_markers()

    def _update_fit_report(self):
        if self.fit_result is None:
            self._fit_result_html.value = "No fit to report."
        else:
            self._fit_result_html.value = self.fit_result._repr_html_()

    def _model_current(self, current=True):
        """Update uptodate to whether displayed date reflect model fit.

        and color refine fit button accordingly
        """
        if self.gui:
            if current:
                self._refit.button_style = ''
            else:
                self._refit.button_style = 'warning'
        self.uptodate = current

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
        for prefix in self.fitvalues.index[self.fitvalues.include]:
            freq = float(self.fitvalues.loc[prefix, 'freq'])
            amp = float(self.fitvalues.loc[prefix, 'amp'])
            phase = float(self.fitvalues.loc[prefix, 'phase'])
            flux += sin(time, freq*self.freq_conversion, amp, phase)
        return flux

    def _update_lcs(self):
        """Update sampled models and residuals time series."""
        meanflux = float(np.mean(self.lc.flux.value[self.include]))
        if self.gui:  # Sampled model
            self.lc_model_sampled.flux = (
                meanflux + self.sample_model(self.lc_model_sampled.time.value))
        # Observed is at all original times (apply mask before calculations)
        self.lc["model"] = (
            meanflux + self.sample_model(self.lc.time.value))*self.lc.flux.unit
        self.lc["resid"] = self.lc.flux - self.lc["model"]

    def _qgrid_changed_manually(self, *args):
        """Pass along manual changes to Signals table to."""
        # Note: args has information about what changed if needed
        newdf = self.signals_qgrid.get_changed_df()
        olddf = self.signals_qgrid.df
        logmessage = "Signals table changed manually.\n"
        changedcols = []
        for key in newdf.index.values:
            if key in olddf.index.values:
                changes = newdf.loc[key][(olddf.loc[key] != newdf.loc[key])]
                changes = changes.dropna()  # Remove nans
                if len(changes) > 0:
                    logmessage += "Values changed for {}:\n".format(key)
                for change in changes.index:
                    logmessage += " - {} -> {}\n".format(change,
                                                         changes[change])
                    changedcols.append(change)
            else:
                logmessage += "New row in solution table: {}\n".format(key)
                for col in newdf.loc[key]:
                    logmessage += " - {} -> {}\n".format(change,
                                                         changes[change])
        self.log(logmessage)

        # Update plots only if signal values (not what is fixed) changed
        self._update_stagedvalues_from_qgrid()

    # Column names and dtypes for tables
    columns = ['include', 'freq', 'fixfreq', 'freqerr',
               'amp', 'fixamp', 'amperr',
               'phase', 'brute', 'fixphase', 'phaseerr']
    dtypes = ['bool', 'float', 'bool', 'float',
              'float', 'bool', 'float',
              'float', 'bool', 'bool', 'float']

    def delete_rows(self, indices):
        """
        Drop provided indices from stagedvalues attribute and Signals table.

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
        if len("indices") == 0:
            self.log("No signals selected for deletion.", 'warning')
            pass  # Nothing to remove
        self.log("Deleted signals {}".format([sig for sig in indices]))
        self.stagedvalues = self.stagedvalues.drop(indices)
        if self.gui:
            self.signals_qgrid.df = (
                self.signals_qgrid.get_changed_df().drop(indices))
        self._model_current(current=False)  # Not deleted until re-fit

    def _delete_selected(self, *args):
        """Delete signals corresponding to Qgrid rows."""
        self.delete_rows(self.signals_qgrid.get_selected_df().index)
        # Also delete associated combination frequencies
        isindep = lambda key: key[1:].isdigit()
        for key in self.signals_qgrid.df.index:
            if not isindep(key) and not self._valid_combo(key):
                self.delete_rows(key)

    def _initialize_dataframe(self):
        """Create new empty signals dataframe."""
        df = (pd.DataFrame(columns=self.columns)
              .astype(dtype=dict(zip(self.columns, self.dtypes))))
        return df

    # Light curve folding stuff
    def _fold_freq_selected(self, value):
        # New frequency value selected to fold on
        if value['new'] is not None:
            self._fold_on.value = value['new']

    def _update_freq_dropdown(self):
        # Add new frequencies to dropdown options
        labels = [self.fitvalues.index[i]
                  + ': {:.8f} '.format(self.fitvalues.freq[i])
                  + self.per_orig.frequency.unit.to_string()
                  for i in range(len(self.fitvalues))]
        currentind = self._select_fold_freq.index
        if currentind is None:
            currentind = 0
        elif currentind >= len(labels):
            currentind = len(labels)-1
        if len(labels) == 0:
            self._select_fold_freq.options = [None]
        else:
            self._select_fold_freq.options = zip(labels,
                                                 self.fitvalues.freq.values)
            self._select_fold_freq.index = currentind

    ########## Set up *SIGNALS* widget using qgrid ##############

    def _get_qgrid(self):
        display_df = self.stagedvalues.copy()
        display_df["amp"] *= self.amp_conversion
        display_df["amperr"] *= self.amp_conversion
        return qgrid.show_grid(display_df, show_toolbar=False, precision=9,
                               grid_options=self._gridoptions,
                               column_definitions=self._column_definitions)

    def _add_staged_signal(self, *args):
        """Add signal to set of signals to fit."""
        # Is this a valid numeric frequency?
        if self._thisfreq.value.replace('.', '', 1).isdigit():
            self.add_signal(float(self._thisfreq.value), self._thisamp.value)
        elif self._valid_combo(self._thisfreq.value):
            self.add_combination(self._thisfreq.value)
        else:
            self.log("Staged frequency has invalid format: {}"
                     .format(self._thisfreq.value), "error")

    def _update_lc_display(self, *args):
        """Change type of time series to display from dropdown."""
        self._display_lc(residuals=(self._tstype.value == "Residuals"))

    def _update_signal_markers(self):
        freqs = self.stagedvalues['freq'][self.stagedvalues.include].values
        amps = (self.stagedvalues['amp'].values[self.stagedvalues.include]
                * self.amp_conversion)
        indep = np.array([key[1:].isdigit() for key in
                          self.stagedvalues.index[self.stagedvalues.include]])

        self.signal_markers.set_data(freqs[np.where(indep)],
                                     amps[np.where(indep)])
        if len(indep) > 0:
            self.combo_markers.set_data(freqs[np.where(~indep)],
                                        amps[np.where(~indep)])
        else:
            self.combo_markers.set_data([], [])  # No markers
        self.perfig.canvas.draw_idle()

    def _display_lc(self, residuals=False):
        lc = self.lc.copy()
        if residuals:
            lc = self.lc.select_flux("resid").copy()
            self.lcplot_model.set_ydata(
                np.zeros(len(self.lc_model_sampled.flux)))
        else:
            self.lcplot_model.set_ydata(self.lc_model_sampled.flux)
        # Rescale y to better match data
        ymin = np.min(lc.flux[self.include].value)
        ymax = np.max(lc.flux[self.include].value)
        self.lcax.set_ylim(ymin-0.05*(ymax-ymin), ymax+0.05*(ymax-ymin))
        # Fold if requested
        if self._fold.value:
            xdata = lc.time.value*self._fold_on.value*self.freq_conversion % 1.
            self.lcplot_data.set_offsets(np.dstack((xdata, lc.flux.value))[0])
            self.lcax.set_xlim(-0.01, 1.01)
        else:
            self.lcplot_data.set_offsets(np.dstack((lc.time.value,
                                                    lc.flux.value))[0])
            tspan = np.ptp(lc.time.value)
            self.lcax.set_xlim(np.min(lc.time.value) - 0.01*tspan,
                               np.max(lc.time.value) + 0.01*tspan)
        self.selector.update(self.lcplot_data)
        self.lcfig.canvas.draw_idle()

    def _mask_selected_pts(self, event):
        self.log(event.key, "debug")
        if ((event.key in ["backspace", "delete"]) and (len(self.selector.ind) > 0)):
            self.log("Masking {} selected points.")
            self.lc["include"][self.selector.ind] = 0
            self._mask_changed()

    def _clear_mask(self, b):
        self.log("Restoring all masked points.")
        self.lc["include"][:] = 1
        self._mask_changed()

    def _mask_changed(self):
        self.include = np.where(self.lc["include"])
        self.selector.ind = []
        self.lcplot_data.set_facecolors([self._lc_colors[m]
                                         for m in self.lc["include"]])
        self.lcplot_data.set_edgecolors("None")
        self._update_lcs()
        self._update_lc_display()
        self._calc_tshift()

        self.compute_pers(orig=True)
        self._update_per_plots()

    def _calc_tshift(self, tshift=None):
        # Subtracting the mean time stabilizes phase fitting.
        if tshift is None:
            self.tshift = -np.mean(self.lc[self.include].time.value)
        else:
            self.tshift = tshift

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
        self._update_status()  # Indicate running calculation
        if orig:  # Compute periodogram of original time series
            self.per_orig = self.lc[self.include].to_periodogram(
                normalization='amplitude', freq_unit=self.freq_unit,
                frequency=self.freqs) * self.amp_conversion
        with np.errstate(invalid='ignore'):
            self.per_model = (self.lc.select_flux("model")[self.include]
                              .to_periodogram(normalization='amplitude',
                                              freq_unit=self.freq_unit,
                                              frequency=self.freqs)
                              * self.amp_conversion)
            self.per_resid = (self.lc.select_flux("resid")[self.include]
                              .to_periodogram(normalization='amplitude',
                                              freq_unit=self.freq_unit,
                                              frequency=self.freqs)
                              * self.amp_conversion)
        # Interpolator for periodogram of residuals.
        self.interpls = interp1d(self.freqs, self.per_resid.power.value,
                                 bounds_error=False,
                                 fill_value=self.per_resid.max_power.value)
        self._log_per_properties()
        self._update_status(False)  # Calculation complete

    def _update_per_plots(self):
        self.perplot_orig.set_ydata(self.per_orig.power.value)
        self.perplot_model.set_ydata(self.per_model.power.value)
        self.perplot_resid.set_ydata(self.per_resid.power.value)
        self.perfig.canvas.draw_idle()

    def _display_per_orig(self, *args):
        if self._show_per_orig.value:
            self.perplot_orig.set_alpha(1)
        else:
            self.perplot_orig.set_alpha(0)
        self.perfig.canvas.draw_idle()

    def _display_per_resid(self, *args):
        if self._show_per_resid.value:
            self.perplot_resid.set_alpha(1)
        else:
            self.perplot_resid.set_alpha(0)
        self.perfig.canvas.draw_idle()

    def _display_per_model(self, *args):
        if self._show_per_model.value:
            self.perplot_model.set_alpha(1)
        else:
            self.perplot_model.set_alpha(0)
        self.perfig.canvas.draw_idle()

    def _display_per_markers(self, *args):
        if self._show_per_markers.value:
            self.signal_markers.set_alpha(1)
            self.combo_markers.set_alpha(1)
        else:
            self.signal_markers.set_alpha(0)
            self.combo_markers.set_alpha(0)
        self.perfig.canvas.draw_idle()

    def _onperiodogramclick(self, event):
        if self._snaptopeak.value:
            # Click within either frequency resolution or 1% of displayed range
            # TODO: make this work with log frequency too
            tolerance = np.max([self.fres,
                                0.01*np.diff(self.perax.get_xlim())])
            nearby = np.argwhere((self.freqs >= event.xdata - tolerance) &
                                 (self.freqs <= event.xdata + tolerance))
            ydata = self.perplot_resid.get_ydata()
            highestind = np.nanargmax(ydata[nearby]) + nearby[0]
            self._update_marker(self.freqs[highestind], ydata[highestind])
        else:
            self._update_marker(event.xdata, self.interpls(event.xdata))

    def TimeSeries(self):
        """Display the interactive Time Series cell in a Jupyter notebook.

        Returns
        -------
        widget
            Time series plot, options, and information to be displayed.
        """
        if self.gui:
            options = widgets.Accordion(children=[
                VBox([self._tstype, self._fold, self._fold_on,
                      self._select_fold_freq, self._reset_mask]),
                self._timeseries_readme], selected_index=None)
            options.set_title(0, 'options')
            options.set_title(1, 'info ')
            savefig = HBox([self._save_tsfig, self._tsfig_file_location])
            return VBox([self._status, self.lcfig.canvas, savefig, options])
        else:
            print("GUI disabled.")

    def Periodogram(self):
        """Display the interactive Periodogram cell in a Jupyter notebook.

        Returns
        -------
        widget
            Periodogram plot, options, and information to be displayed.
        """
        if self.gui:
            options = widgets.Accordion(children=[
                VBox([self._snaptopeak, self._show_per_markers,
                      self._show_per_orig, self._show_per_resid,
                      self._show_per_model]), self._periodogram_readme],
                selected_index=None)
            options.set_title(0, 'options')
            options.set_title(1, 'info ')
            savefig = HBox([self._save_perfig, self._perfig_file_location])
            periodogram = VBox([self._status,
                                HBox([self._thisfreq, self._thisamp]),
                                HBox([self._addtosol, self._refit]),
                                self.perfig.canvas,
                                savefig,
                                options])
            return periodogram
        else:
            print("GUI disabled.")

    def Pyriod(self):
        """Display the interactive Pyriod suite in a Jupyter notebook.

        Includes the Time Series, Periodogram, Signals, and Log widgets in
        separate tabs.

        Returns
        -------
        widget
            Time Series, Periodogram, Signals, and Log widgets in tabs.
        """
        if self.gui:
            tstab = self.TimeSeries()
            pertab = self.Periodogram()
            signalstab = self.Signals()
            logtab = self.Log()
            tabs = widgets.Tab(children=[tstab, pertab, signalstab, logtab])
            tabs.set_title(0, 'Time Series')
            tabs.set_title(1, 'Periodogram')
            tabs.set_title(2, 'Signals')
            tabs.set_title(3, 'Log')
            return tabs
        else:
            print("GUI disabled.")

    def _update_marker(self, x, y):
        # Move the signal marker.
        try:
            self._thisfreq.value = str(x[0])
        except:
            self._thisfreq.value = str(x)
        self._thisamp.value = y
        self.marker.set_data([x], [y])
        self.perfig.canvas.draw_idle()
        self.perfig.canvas.flush_events()

    def _mark_highest_peak(self):
        # Move signal marker to current highest peak.
        self._update_marker(
            self.freqs[np.nanargmax(self.per_resid.power.value)],
            np.nanmax(self.per_resid.power.value))

    def _onclick(self, event):
        self._onperiodogramclick(event)

    def _onpress(self, event):
        self._press = True

    def _onmove(self, event):
        if self._press:
            self._move = True

    def _onrelease(self, event):
        if self._press and not self._move:
            self._onclick(event)
        self._press = False
        self._move = False

    def Signals(self):
        """Display the interactive Signals cell in a Jupyter notebook.

        Returns
        -------
        widget
            Signals table, fit report, and other information to be displayed.
        """
        if self.gui:
            fitreport = widgets.Accordion(
                children=[self._fit_result_html, self._signals_readme],
                selected_index=None)
            fitreport.set_title(0, 'fit report')
            fitreport.set_title(1, 'info ')
            return VBox([self._status,
                         HBox([self._refit, self._thisfreq, self._thisamp,
                               self._addtosol, self._delete]),
                         self.signals_qgrid,
                         HBox([self._save, self._load,
                               self._signals_file_location]),
                         fitreport])
        else:
            print("GUI disabled.")

    def Log(self):
        """Display the Pyriod Log cell in a Jupyter notebook.

        Returns
        -------
        widget
            Log of actions taken.
        """
        if self.gui:
            return self._logbox
        else:
            print("GUI disabled.")

    def _update_log(self):
        self._log.value = self.log_capture_string.getvalue()

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

    def _save_button_click(self, *args):
        self.save_solution(filename=self._signals_file_location.selected)

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
            logmessage = ("Loading signal solution from "
                          + os.path.abspath(filename) + ".<br />")
            logmessage += loaddf.to_string().replace('\n', '<br />')
            self.log(logmessage)
            self.signals_qgrid.df = loaddf
            self._update_stagedvalues_from_qgrid()
        else:
            self.log("Failed to load " + os.path.abspath(filename)
                     + ". File not found.<br />", level='error')

    def _load_button_click(self, *args):
        self.load_solution(filename=self._signals_file_location.selected)

    def _save_log_button_click(self, *args):
        self.save_log(self._log_file_location.selected, self._overwrite.value)

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
        soup = BeautifulSoup(self._log.value, features="xml")
        mode = {True: "w+", False: "a+"}[overwrite]
        f = open(filename, mode)
        f.write(soup.get_text().replace('|', ''))
        f.close()

    def save_tsfig(self, filename='Pyriod_TimeSeries.png', **kwargs):
        """Save time series plot to file.

        Parameters
        ----------
        filename : str, optional
            Filename for saving the plot. The default is
            'Pyriod_TimeSeries.png'.
        **kwargs : keyword arguments
            Passed to matplotlib.pyplot.savefig function.
        """
        if self.gui:
            self.lcfig.savefig(filename, **kwargs)
        else:
            print("Plotting disabled.")

    def _save_tsfig_button_click(self, *args):
        self.save_tsfig(self._tsfig_file_location.selected)

    def save_perfig(self, filename='Pyriod_Periodogram.png', **kwargs):
        """Save periodogram plot to file.

        Parameters
        ----------
        filename : str, optional
            Filename for saving the plot. The default is
            'Pyriod_Periodogram.png'.
        **kwargs : keyword arguments
            Passed to matplotlib.pyplot.savefig function.
        """
        if self.gui:
            self.perfig.savefig(filename, **kwargs)
        else:
            print("Plotting disabled.")

    def _save_perfig_button_click(self, *args):
        self.save_perfig(self._perfig_file_location.selected)

    def _update_status(self, calculating=True):
        if self.gui:
            if calculating:
                self._status.value = (
                    "<center><b><big><font color='red'>"
                    "UPDATING CALCULATIONS...</big></b></center>")
            else:
                self._status.value = ""

    def close_figures(self):
        """Close all figures belonging to this Pyriod instance.

        Warning: interacting with figures will no longer work.
        """
        if self.gui:
            plt.close(self.lcfig)
            plt.close(self.perfig)

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
        time = self.lc.time[self.include].value
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

    def plot_spectral_window(self, maxfreq=100, osample=10, ax=None,
                             **plt_kwargs):
        """Compute and plot the spectral window for these data.

        Parameters
        ----------
        maxfreq : float, optional
            Maximum frequency to calculate the spectral window out to. The
            default is 100.
        osample : float, optional
            The oversample factor to compute the spectral window with, relative
            to the natural frequency resolution of 1/(time span of the data).
            The default is 10.
        ax : matplotlib axes, optional
            Optional axes to plot the spectral window on. The default is None.
        **plt_kwargs : keyword arguments
            Passed to the plt.plot function.
        """
        if ax is None:
            ax = plt.gca()
        freqvec, ampvec = self.spectral_window(maxfreq, osample)
        ax.plot(freqvec, ampvec, **plt_kwargs)
        ax.set_xlim(0, maxfreq)
        ax.set_ylim(0, 1)
        ax.set_xlabel(f"frequency ({self.freq_label})")
        ax.set_ylabel('spectral window')
        return(ax)
