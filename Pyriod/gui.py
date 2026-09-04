import os
import sys
import numpy as np

import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import HBox, VBox
import qgridnext as qgrid
from ipyfilechooser import FileChooser
from traitlets.traitlets import TraitError

from .plotsupport import (
    decimate_visible_range,
    minmax_decimate,
    visible_range_indices,
    lasso_selector
)
from .utils import _as_scalar_float

from Pyriod.analyses.gui.base import AnalysisGUI

plt.ioff()  # Turn off interactive mode


class PyriodGUI:
    """Interactive Jupyter interface for a `Prewhitener`.

    `PyriodGUI` provides interactive time-series and periodogram plots,
    signal-table editing, fitting controls, significance-threshold controls,
    and access to the Pyriod log. Numerical analysis and fitted state are
    stored in the associated `Prewhitener`.

    Parameters
    ----------
    prewhitener : Prewhitener
        Analysis object to display and manipulate.

    Attributes
    ----------
    pw : Prewhitener
        Underlying analysis object used by the GUI.
    lcfig : matplotlib.figure.Figure
        Figure containing the interactive time-series plot.
    lcax : matplotlib.axes.Axes
        Axes containing the interactive time-series plot.
    perfig : matplotlib.figure.Figure
        Figure containing the interactive periodogram plot.
    perax : matplotlib.axes.Axes
        Axes containing the interactive periodogram plot.
    lc : lightkurve.LightCurve
        Convenience access to ``pw.lc``.
    fitvalues : pandas.DataFrame
        Convenience access to ``pw.fitvalues``.
    stagedvalues : pandas.DataFrame
        Convenience access to ``pw.stagedvalues``.

    Notes
    -----
    The GUI operates directly on the supplied `Prewhitener`; it does not
    maintain a separate copy of the analysis state. Changes made through
    the GUI therefore modify ``pw``.

    The GUI assumes that analysis operations are normally performed through
    the interface. If ``pw`` is modified directly, call
    ``refresh_from_prewhitener()`` to synchronize the displayed plots,
    signal table, fit report, log, and other GUI state with the current
    `Prewhitener`.

    The interactive Matplotlib figures require the ipympl backend, normally
    enabled in a Jupyter notebook with ``%matplotlib widget``.
    """
    def __init__(self, prewhitener):
        self.pw = prewhitener

        # Create status widget to indicate when calculations are running
        self._status = widgets.HTML(value="")

        # Initiate things in the right order so connections can be made
        self._init_timeseries_widgets()
        self._init_timeseries_figures()
        self._init_periodogram_widgets()
        self._init_periodogram_figures()
        self._init_signals_qgrid()
        self._init_signals_widgets()
        self._init_log_widgets()
        self.refresh_from_prewhitener()

    ## Initialize Widgets
    def _init_timeseries_widgets(self):
        """Create widgets used by the time-series interface."""
        # Plot location file chooser
        self._tsfig_file_location = FileChooser(
            os.getcwd(),
            filename='Pyriod_TimeSeries.png',
            show_hidden=False,
            select_default=True,
            dir_icon="📁",
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
        self._tstype.observe(self._update_and_rescale_lc_display)

        # Fold on frequency checkbox
        self._fold = widgets.Checkbox(
            value=False,
            description='Fold time series on frequency?',
        )
        self._fold.observe(self._update_and_rescale_lc_display)

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

    def _init_periodogram_widgets(self):
        """Create widgets used by the periodogram interface."""
        # Plot location file chooser
        self._perfig_file_location = FileChooser(
            os.getcwd(),
            filename='Pyriod_Periodogram.png',
            show_hidden=False,
            select_default=True,
            dir_icon="📁",
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

        # Button to re-compute best fit
        self._refit = widgets.Button(
            description="Compute fit",
            disabled=False,
            tooltip='Refine fit of signals to time series',
            icon='refresh'
        )
        self._refit.on_click(self.fit_model)

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

        # Checkboxes, show significance threshold
        self._show_sig_threshold = widgets.Checkbox(
            value=True,
            description='Sig Threshold',
            disabled=False,
            style={'description_width': 'initial'}
        )
        self._show_sig_threshold.observe(self._display_sig_threshold)

        # Widgets for computing significance threshold too!
        # Significance multiplier
        self._sig_multiplier_widget = widgets.FloatText(
            value = 5.0,
            description='Scaling factor:',
            style={'description_width': 'initial'}
        )
        # Starting frequency
        self._sig_startfreq_widget = widgets.FloatText(
            value = 0,
            description='Start frequency:',
            style={'description_width': 'initial'}
        )
        # Ending frequency
        self._sig_endfreq_widget = widgets.FloatText(
            value = self.pw.nyquist,
            description='End frequency:',
            style={'description_width': 'initial'}
        )
        # Frequency step
        self._sig_freqstep_widget = widgets.FloatText(
            value = self.pw.nyquist/10.0000001,
            description='Step size:',
            style={'description_width': 'initial'}
        )
        # Window width
        self._sig_winwidth_widget = widgets.FloatText(
            value = self.pw.nyquist/10.0000001,
            description='Window width:',
            style={'description_width': 'initial'}
        )
        # Type of average to take
        self._sig_avgtype_widget = widgets.Dropdown(
            options = ["mean","median"],
            description='Average:',
            style={'description_width': 'initial'}
        )
        # Whether to extrapolate
        self._sig_extrapolate_widget = widgets.Checkbox(
            value = False,
            description='Extrapolate',
            style={'description_width': 'initial'}
        )
        # Automatically recalculate sig threshold?
        self._sig_auto_recalculate = widgets.Checkbox(
            value = False,
            description='Auto-recalculate',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='100%')
        )
        self._sig_auto_recalculate.observe(self._sig_thresh_change_auto)

        # Calulate sig threshold button
        self._sig_calculate_button = widgets.Button(
            description='Calculate',
            tooltip="Calculate new sigificance threshold."
        )
        self._sig_calculate_button.on_click(self._sig_thresh_from_gui)

    def _init_signals_qgrid(self):
        """Configure and create the editable staged-signal QGrid."""
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
        

        self._signals_qgrid = self._get_qgrid()
        self._signals_qgrid.on('cell_edited', self._qgrid_changed_manually)

    def _get_qgrid(self):
        """Create a QGrid displaying the currently staged signal table."""
        display_df = self.pw.staged_table(display_units=True)
        return qgrid.show_grid(display_df, show_toolbar=False, precision=9,
                               grid_options=self._gridoptions,
                               column_definitions=self._column_definitions)

    def _init_signals_widgets(self):
        """Create widgets used by the signal-table interface."""
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
            dir_icon="📁",
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
        self._update_fit_report()  # No fit to report

        # HTML widget to display Readme
        #path = Path(__file__).parent / 'docs/Signals.md'
        #html = gh_md_to_html.main(str(path), enable_image_downloading=False)
        #self._signals_readme = widgets.HTML(html)


    def _init_log_widgets(self):
        """Create widgets used by the log interface."""
        self._log = widgets.HTML(
            value='Log',
            placeholder='Log',
            description='Log:',
            layout={
                'height': '250px',
                'width': '950px',
                'overflow': 'auto'
            }
        )
        self.update_log()

        # Save log location file chooser
        self._log_file_location = FileChooser(
            os.getcwd(),
            filename='Pyriod_log.txt',
            show_hidden=False,
            select_default=True,
            dir_icon="📁",
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

    ## Initialize figures
    def _init_timeseries_figures(self):
        """Create the interactive time series figure and masking selector."""
        self.lcfig, self.lcax = plt.subplots(
            figsize=(7, 2), num='Time Series ({:d})'.format(self.pw.id))
        self.lcax.set_position([0.13, 0.22, 0.85, 0.76])
        self._lc_colors = {0: "bisque", 1: "C0"}
        self._lcplot_data = self.lcax.scatter(
            self.pw.lc.time.value, np.array(self.pw.lc.flux.value), marker='o',
            s=5, ec='None', lw=1, c=self._lc_colors[1])
        self._set_timeseries_plot_labels()
        
        # Define selector for masking points
        self._selector = lasso_selector(self.lcax, self._lcplot_data)
        self._lc_key_press_callback_id = self.lcfig.canvas.mpl_connect("key_press_event",
                                        self._mask_selected_pts)
        # Set to display sampled model
        self._init_viewport_model_plot()
    
    def _set_timeseries_plot_labels(self):
        """Update time series axis labels for the current display mode."""
        # Light curve labels
        try:
            if self._fold.value:
                self.lcax.set_xlabel(f"phase folded on {self._fold_on.value:.6g} {self.pw.freq_unit} (0-1)")
            else:
                self.lcax.set_xlabel(f"time ({self.pw.time_unit.to_string()})")
        except:
            self.lcax.set_xlabel(f"time ({self.pw.time_unit.to_string()})")
        self.lcax.set_ylabel("flux")
        self.lcfig.canvas.draw_idle()

    def _init_viewport_model_plot(self):
        """Initialize viewport-dependent plotting of the fitted model."""
        # Create the line once. Do not recreate it on every zoom/pan.
        self._lcplot_model, = self.lcax.plot([np.min(self.pw.lc.time.value),
                                              np.max(self.pw.lc.time.value)], 
                                             [1,1], lw=1, c='r', zorder=3)

        self.max_model_plot_points = 20_000
        self.model_oversample = 20  # samples per cycle of highest included frequency
        self._last_model_xlim = None

        self._model_update_timer = self.lcfig.canvas.new_timer(interval=75)
        self._model_update_timer.single_shot = True
        self._model_update_timer.add_callback(self._refresh_model_line_from_view)

        # Update model whenever the visible x-range changes.
        self._model_xlim_callback_id = self.lcax.callbacks.connect(
            "xlim_changed",
            self._request_model_line_update,
        )

        # Draw the initial model.
        self._update_sampled_model()

    def _request_model_line_update(self, ax=None):
        """Request a debounced update of the displayed model line."""
        self._model_update_timer.stop()
        self._model_update_timer.start()

    def _update_sampled_model(self):
        """Invalidate and request an update of the displayed sampled model."""
        self._last_model_xlim = None
        self._request_model_line_update()

    def _refresh_model_line_from_view(self):
        """Recalculate the fitted-model line over the visible time range."""
        if self._fold.value: # don't display folded model
            return

        x0, x1 = self.lcax.get_xlim()
        xmin, xmax = sorted((x0, x1))

        # If displaying residuals, should be a flat line at zero
        if self._tstype.value == 'Residuals':
            self._lcplot_model.set_data([x0, x1], [0,0])
            self.lcfig.canvas.draw_idle()
            return

        # Avoid recomputing if nothing changed
        xlim = (xmin, xmax)
        if self._last_model_xlim == xlim:
            return
        self._last_model_xlim = xlim

        timesample = self._make_model_time_grid(xmin, xmax)

        if timesample.size == 0:
            self._lcplot_model.set_data([], [])
        else:
            good = np.where(self.pw.lc["include"])
            meanflux = float(np.mean(np.array(self.pw.lc.flux.value[good])))
            modelsampled = meanflux + self.pw.sample_model(timesample)
            self._lcplot_model.set_data(timesample, modelsampled)

        self.lcfig.canvas.draw_idle()

    def _make_model_time_grid(self, xmin, xmax):
        """Construct a sampling grid for plotting the model in the visible range.

        Parameters
        ----------
        xmin, xmax : float
            Limits of the visible time range.

        Returns
        -------
        numpy.ndarray
            Time samples sufficient to resolve the highest-frequency fitted
            signal, subject to the configured plotting-point limit.
        """
        span = xmax - xmin
        if span <= 0:
            return np.array([])

        # Highest currently included fitted frequency.
        fitvalues = self.pw.fitvalues
        included = fitvalues["include"].to_numpy(dtype=bool)

        if not np.any(included):
            return np.array([np.min(self.pw.lc.time.value),np.max(self.pw.lc.time.value)])

        fmax = np.nanmax(np.abs(fitvalues.loc[included, "freq"].to_numpy()))

        # Enough samples to trace the highest-frequency included sinusoid.
        fmax_per_day = fmax * self.pw.freq_conversion
        n_by_freq = int(np.ceil(self.model_oversample * fmax_per_day * span))

        # Also use at least about one point per display pixel.
        try:
            n_by_pixels = int(self.lcax.bbox.width)
        except Exception:
            n_by_pixels = 1000

        n = max(2, n_by_freq, n_by_pixels)
        n = min(n, self.max_model_plot_points)

        return np.linspace(xmin, xmax, n)

    def _init_periodogram_figures(self):
        """Create the interactive periodogram figure and plot artists."""
        self.perfig, self.perax = plt.subplots(
            figsize=(7, 3), num='Periodogram ({:d})'.format(self.pw.id))

        # Create empty plot artists once. They will be populated by
        # _refresh_periodogram_lines_from_view().
        self._perplot_orig, = self.perax.plot([], [], lw=1, c='tab:gray')
        self._perplot_model, = self.perax.plot([], [], lw=1, c='tab:green')
        self._perplot_resid, = self.perax.plot([], [], lw=1, c='tab:blue')

        # Placeholder only; do not allocate self.freqs*np.nan.
        self._sig_threshold_plot, = self.perax.plot([], [], lw=1, c='red', ls='--')

        self.perax.set_ylim(0, 1.05*np.nanmax(self.pw.per_orig))
        self.perax.set_xlim(np.min(self.pw.freqs), np.max(self.pw.freqs))
        self.perax.set_position([0.13, 0.22, 0.8, 0.76])

        self._init_viewport_periodogram_plot()

        # Create markers for selected peak, adopted signals
        self._marker = self.perax.plot([0], [0], c='k', marker='o')[0]
        self._signal_marker_color = 'green'
        self._signal_markers, = self.perax.plot([], [], marker='D',
                                                fillstyle='none',
                                                linestyle='None',
                                                c=self._signal_marker_color,
                                                ms=5)
        self._combo_marker_color = 'orange'
        self._combo_markers, = self.perax.plot([], [], marker='D',
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
        self._per_button_press_callback_id = self.perfig.canvas.mpl_connect(
            "button_press_event",
            self._onpress,
        )
        self._per_button_release_callback_id = self.perfig.canvas.mpl_connect(
            "button_release_event",
            self._onrelease,
        )
        self._per_motion_callback_id = self.perfig.canvas.mpl_connect(
            "motion_notify_event",
            self._onmove,
        )

        # Set axis labels
        self.perax.set_ylabel(f"amplitude ({self.pw.amp_unit})")
        self.perax.set_xlabel(f"frequency ({self.pw._freq_label})")
        self.lcfig.canvas.draw_idle()

    # Set up all the efficient viewport stuff here for periodogram plot
    def _init_viewport_periodogram_plot(self):
        """Initialize viewport-dependent, decimated periodogram plotting."""

        # Maximum number of points stored in each displayed periodogram artist.
        # Because min/max decimation emits up to two points per bin, this is an
        # approximate cap.
        self.max_periodogram_plot_points = 30_000

        self._last_periodogram_xlim = None

        # Debounce periodogram redrawing, following the light-curve model pattern.
        self._periodogram_update_timer = self.perfig.canvas.new_timer(interval=75)
        self._periodogram_update_timer.single_shot = True
        self._periodogram_update_timer.add_callback(
            self._refresh_periodogram_lines_from_view
        )

        self._periodogram_xlim_callback_id = self.perax.callbacks.connect(
            "xlim_changed",
            self._request_periodogram_plot_update,
        )

        self._refresh_periodogram_lines_from_view()


    def _request_periodogram_plot_update(self, ax=None):
        """Request a debounced redraw of the visible periodogram range."""
        self._periodogram_update_timer.stop()
        self._periodogram_update_timer.start()


    def _update_per_plots(self):
        """Refresh periodogram plot data after periodograms are recalculated."""
        self._last_periodogram_xlim = None
        self._request_periodogram_plot_update()
        # Update significance threshold
        # update plot
        if ((self.pw.noise_spectrum is not None) & (self.pw.significance_multiplier is not None) &
                                                    (self.pw.significance_settings is not None)):
            self._sig_threshold_plot.set_data(
                self.pw._sig_threshold_freqs,
                self.pw._sig_threshold_power,
            )
            self.perfig.canvas.draw_idle()

    def _refresh_periodogram_lines_from_view(self):
        """Update periodogram lines using decimated data from the visible range."""
        x0, x1 = self.perax.get_xlim()
        xmin, xmax = sorted((x0, x1))

        xlim = (xmin, xmax)
        if self._last_periodogram_xlim == xlim:
            return

        self._last_periodogram_xlim = xlim

        self._set_decimated_periodogram_line(
            self._perplot_orig,
            self.pw.per_orig,
            xmin,
            xmax,
        )

        self._set_decimated_periodogram_line(
            self._perplot_model,
            self.pw.per_model,
            xmin,
            xmax,
        )

        self._set_decimated_periodogram_line(
            self._perplot_resid,
            self.pw.per_resid,
            xmin,
            xmax,
        )

        self.perfig.canvas.draw_idle()
   
    def _set_decimated_periodogram_line(self, line, power, xmin, xmax):
        """Set a periodogram line from decimated data in the visible range.

        Parameters
        ----------
        line : matplotlib.lines.Line2D
            Plot artist to update.
        power : array-like
            Periodogram amplitudes corresponding to ``pw.freqs``.
        xmin, xmax : float
            Visible frequency limits.
        """
        xplot, yplot = decimate_visible_range(
            self.pw.freqs,
            power,
            xmin,
            xmax,
            max_points=self.max_periodogram_plot_points,
        )

        line.set_data(xplot, yplot)

    def _update_refit_button(self):
        """Update the fit button style to indicate whether the model is current."""
        if self.pw.uptodate:
            self._refit.button_style = ''
        else:
            self._refit.button_style = 'warning'

    def refresh_from_prewhitener(self):
        """Refresh GUI displays from the current `Prewhitener` state."""
        self._update_freq_dropdown()

        self._update_lc_display()
        self._refresh_model_line_from_view()

        self._refresh_periodogram_lines_from_view()
        self._update_signal_markers()
        self._display_per_markers()
        self._mark_highest_peak()

        self._update_signals_qgrid()
        self._update_fit_report()
        self.update_log()

    def refresh_from_prewhitener(self, reset_limits=False):
        """Refresh GUI displays from the current Prewhitener state.

        Updates the light curve, periodograms, signal markers, staged signal
        table, fit report, log, masked-point colors, and significance threshold.

        Parameters
        ----------
        reset_limits : bool, optional
            Whether to reset the time-series and periodogram plot limits to match
            the current data and frequency sampling. The default is False.

        Notes
        -----
        Call this after modifying ``self.pw`` directly to synchronize the GUI
        with the underlying `Prewhitener`.
        """
        self._update_freq_dropdown()

        # Update light curve display
        self._lcplot_data.set_facecolors(
            [self._lc_colors[m] for m in self.pw.lc["include"]]
        )
        self._lcplot_data.set_edgecolors("None")
        self._display_lc(
                            residuals=(self._tstype.value == "Residuals"),
                            rescale=reset_limits
                        )
        self._refresh_model_line_from_view()

        # Update periodogram display
        self.perax.set_ylabel(f"amplitude ({self.pw.amp_unit})")
        self.perax.set_xlabel(f"frequency ({self.pw._freq_label})")
        if reset_limits:
            self.perax.set_xlim(np.min(self.pw.freqs), np.max(self.pw.freqs))
            ymax = np.nanmax([self.pw.per_orig,
                            self.pw.per_model,
                            self.pw.per_resid])
            self.perax.set_ylim(0, 1.05*ymax)

        self._last_periodogram_xlim = None
        self._refresh_periodogram_lines_from_view()

        # Update significance threshold
        if ((self.pw.noise_spectrum is not None) &
            (self.pw.significance_multiplier is not None) &
            (self.pw.significance_settings is not None)):
            self._sig_threshold_plot.set_data(
                self.pw._sig_threshold_freqs,
                self.pw._sig_threshold_power)
        else:
            self._sig_threshold_plot.set_data([], [])

        self._update_signal_markers()
        self._display_per_markers()
        self._mark_highest_peak()

        self._update_signals_qgrid()
        self._update_fit_report()
        self.update_log()

    ## Main Widget collections
    def TimeSeries(self):
        """Return the interactive time-series interface. 
        
        The interface contains the time-series plot, controls for displaying the 
        original or residual light curve, phase-folding controls, masking 
        controls, and figure-saving controls. 
        
        Returns 
        ------- 
        ipywidgets.Widget 
            Widget containing the interactive time-series interface. 
            
        Raises 
        ------ 
        traitlets.TraitError 
            If the Matplotlib canvas cannot be embedded as an ipywidget, for 
            example when the ipympl backend is not active. 
        """
        try:
            options = widgets.Accordion(children=[
                VBox([self._tstype, self._fold, self._fold_on,
                    self._select_fold_freq, self._reset_mask])], selected_index=None)
            options.set_title(0, 'options')
            savefig = HBox([self._save_tsfig, self._tsfig_file_location])
            return VBox([self._status, self.lcfig.canvas, savefig, options])
        except TraitError as e:
            e.add_note("You must use the ipympl plotting backend. Use magic command `%matplotlib widget`.")
            raise

    def Periodogram(self):
        """Return the interactive periodogram interface. 
        
        The interface contains the periodogram plot, controls for selecting and 
        staging signals, fitting the current staged solution, choosing displayed 
        periodograms and signal markers, calculating significance thresholds, 
        and saving the figure. 
        
        Returns 
        ------- 
        ipywidgets.Widget 
            Widget containing the interactive periodogram interface. 
        
        Raises 
        ------ 
        traitlets.TraitError 
            If the Matplotlib canvas cannot be embedded as an ipywidget, for example when the ipympl backend is not active. 
        """
        try:
            # display config on left, sig threshold at right
            displayconfig = VBox([self._snaptopeak,
                                self._show_per_markers,
                                self._show_per_orig,
                                self._show_per_resid,
                                self._show_per_model,
                                self._show_sig_threshold])
            thresholdconfig = VBox([widgets.Label("Significance Threshold:"),
                                    self._sig_multiplier_widget,
                                    self._sig_startfreq_widget,
                                    self._sig_endfreq_widget,
                                    self._sig_freqstep_widget,
                                    self._sig_winwidth_widget,
                                    self._sig_avgtype_widget,
                                    HBox([self._sig_extrapolate_widget,self._sig_auto_recalculate]),
                                    self._sig_calculate_button],
                                layout=widgets.Layout(border='solid 1px'))
            options = HBox([displayconfig, thresholdconfig])
            accordians = widgets.Accordion(
                children=[options],
                selected_index=None)
            accordians.set_title(0, 'options')
            savefig = HBox([self._save_perfig, self._perfig_file_location])
            periodogram = VBox([self._status,
                                HBox([self._thisfreq, self._thisamp]),
                                HBox([self._addtosol, self._refit]),
                                self.perfig.canvas,
                                savefig,
                                accordians])
            return periodogram
        except TraitError as e:
            e.add_note("You must use the ipympl plotting backend. Use magic command `%matplotlib widget`.")
            raise

    def Signals(self):
        """Return the interactive signal-table interface. 
        
        The interface displays the signal parameters staged for the next fit and 
        provides controls for adding, editing, removing, saving, and loading 
        signals. It also provides access to the most recent fit report. 
        
        Returns 
        ------- 
        ipywidgets.Widget 
            Widget containing the staged signal table and associated controls. 
        """
        fitreport = widgets.Accordion(
            children=[self._fit_result_html],
            selected_index=None)
        fitreport.set_title(0, 'fit report')
        return VBox([self._status,
                        HBox([self._refit, self._thisfreq, self._thisamp,
                            self._addtosol, self._delete]),
                        self._signals_qgrid,
                        HBox([self._save, self._load,
                            self._signals_file_location]),
                        fitreport])

    def Log(self):
        """Return the Pyriod log interface. 
        
        Returns 
        ------- 
        ipywidgets.Widget 
            Widget displaying the current Pyriod log together with controls for saving it to a file. 
        """
                # Layout Log widgets
        savelog = HBox([self._save_log,
                    self._log_file_location,
                    self._overwrite])
        return VBox(
            [widgets.Box([self._log]),
             savelog])

    def Pyriod(self):
        """Return the complete interactive Pyriod interface. 
        
        The time series, periodogram, signal table, and log interfaces are 
        combined into separate tabs. 
        
        Returns 
        ------- 
        ipywidgets.Widget 
            Tabbed widget containing the complete Pyriod interface. 
        """
        if hasattr(self, "_tabs"):
            return self._tabs

        self._base_tabs = [
            ("Time Series", self.TimeSeries()),
            ("Periodogram", self.Periodogram()),
            ("Signals", self.Signals()),
            ("Log", self.Log()),
        ]
        self._analyses = {}

        self._tabs = widgets.Tab()
        self._rebuild_tabs()

        return self._tabs

    def _rebuild_tabs(self):
        "Include all analysis tabs in GUI interface."
        entries = list(self._base_tabs)

        for analysis in self._analyses.values():
            entries.append(
                (analysis.title, analysis.widget)
            )

        self._tabs.children = tuple(
            widget for _, widget in entries
        )

        for i, (title, _) in enumerate(entries):
            self._tabs.set_title(i, title)

    # Functions for saving plots
    def save_tsfig(self, filename='Pyriod_TimeSeries.png', **kwargs):
        """Save the current time-series figure.

        Parameters
        ----------
        filename : str or path-like, optional
            Output filename. Default is ``"Pyriod_TimeSeries.png"``.
        **kwargs
            Additional keyword arguments passed to
            ``matplotlib.figure.Figure.savefig``.
        """
        self.lcfig.savefig(filename, **kwargs)

    # Plot widget-related functions
    def _save_tsfig_button_click(self, *args):
        """Save the time-series figure to the path selected in the GUI."""
        self.save_tsfig(self._tsfig_file_location.selected)
    
    def save_perfig(self, filename='Pyriod_Periodogram.png', **kwargs):
        """Save the current periodogram figure.

        Parameters
        ----------
        filename : str or path-like, optional
            Output filename. Default is ``"Pyriod_Periodogram.png"``.
        **kwargs
            Additional keyword arguments passed to
            ``matplotlib.figure.Figure.savefig``.
        """
        self.perfig.savefig(filename, **kwargs)

    def _save_perfig_button_click(self, *args):
        """Save the periodogram figure to the path selected in the GUI."""
        self.save_perfig(self._perfig_file_location.selected)

    def _update_status(self, calculating=True):
        """Update the calculation-status display and fitting controls.

        Parameters
        ----------
        calculating : bool, optional
            If True, show the calculation indicator and disable fitting
            controls. If False, clear the indicator and re-enable the
            controls. Default is True.
        """
        if calculating:
            self._status.value = (
                "<center><b><big><font color='red'>"
                "UPDATING CALCULATIONS...</big></b></center>")
            #Disable buttons during calculation
            self._addtosol.disabled = True
            self._refit.disabled = True
        else:
            self._status.value = ""
            #Re-enable buttons
            self._addtosol.disabled = False
            self._refit.disabled = False

    
    # Functions to update displays
    def _update_lc_display(self, *args):
        """Update the displayed time series for the selected data type."""
        self._display_lc(residuals=(self._tstype.value == "Residuals"))

    def _update_and_rescale_lc_display(self, *args):
        """Update the displayed time series and rescale its axes."""
        self.log(str(*args))
        self._display_lc(residuals=(self._tstype.value == "Residuals"),rescale=True)

    def _update_signal_markers(self):
        """Update periodogram markers for currently staged signals."""
        freqs = self.pw.stagedvalues['freq'][self.pw.stagedvalues.include].values
        amps = (self.pw.stagedvalues['amp'].values[self.pw.stagedvalues.include]
                * self.pw.amp_conversion)
        indep = np.array([key[1:].isdigit() for key in
                          self.pw.stagedvalues.index[self.pw.stagedvalues.include]])

        self._signal_markers.set_data(freqs[np.where(indep)],
                                     amps[np.where(indep)])
        if len(indep) > 0:
            self._combo_markers.set_data(freqs[np.where(~indep)],
                                        amps[np.where(~indep)])
        else:
            self._combo_markers.set_data([], [])  # No markers
        self.perfig.canvas.draw_idle()

    def _display_lc(self, residuals=False, rescale = False):
        """Update the displayed light-curve data.

        Parameters
        ----------
        residuals : bool, optional
            If True, display residuals instead of the original flux.
            Default is False.
        rescale : bool, optional
            If True, rescale the plot limits to the displayed data.
            Default is False.
        """
        ydata = np.copy(self.pw.lc.flux.value)
        if residuals:
            good = np.where(self.pw.lc["include"])
            meanflux = float(np.mean(np.array(self.pw.lc.flux.value[good])))
            modellc = meanflux + self.pw.sample_model(self.pw.lc.time.value)*self.pw.lc.flux.unit
            ydata = (self.pw.lc["flux"] - modellc).value # this to be displayed
            self._update_sampled_model() # handles residuals
        else:
            self._update_sampled_model()
        # Rescale y to better match data 
        if rescale:
            good = np.where(self.pw.lc["include"])
            ymin = np.min(ydata[good])
            ymax = np.max(ydata[good])
            self.lcax.set_ylim(ymin-0.05*(ymax-ymin), ymax+0.05*(ymax-ymin))

        # Fold if requested
        if self._fold.value:
            xdata = np.copy(self.pw.lc.time.value)*self._fold_on.value*self.pw.freq_conversion % 1.
            self._lcplot_data.set_offsets(np.dstack((xdata, ydata))[0])
            self.lcax.set_xlim(-0.01, 1.01) 
            self._lcplot_model.set_alpha(0) # don't show model
        else:
            self._lcplot_data.set_offsets(np.dstack((self.pw.lc.time.value,
                                                     ydata))[0])
            tspan = np.ptp(self.pw.lc.time.value)
            self._lcplot_model.set_alpha(1) # show model
            if rescale:
                self.lcax.set_xlim(np.min(self.pw.lc.time.value) - 0.01*tspan,
                                   np.max(self.pw.lc.time.value) + 0.01*tspan)
        self._selector.update(self._lcplot_data)
        self._set_timeseries_plot_labels()
        self.lcfig.canvas.draw_idle()

        # Light curve folding stuff
    def _fold_freq_selected(self, value):
        """Update the folding frequency from the fitted-signal selector."""
        if value['new'] is not None:
            self._fold_on.value = value['new']

    def _update_freq_dropdown(self):
        """Update the phase-folding frequency choices from fitted signals."""
        labels = [self.pw.fitvalues.index[i]
                  + ': {:.8f} '.format(self.pw.fitvalues.freq.iloc[i])
                  + self.pw.freq_unit.to_string()
                  for i in range(len(self.pw.fitvalues))]
        currentind = self._select_fold_freq.index
        if currentind is None:
            currentind = 0
        elif currentind >= len(labels):
            currentind = len(labels)-1
        if len(labels) == 0:
            self._select_fold_freq.options = [None]
        else:
            self._select_fold_freq.options = zip(labels,
                                                 self.pw.fitvalues.freq.values)
            self._select_fold_freq.index = currentind


    ## Functions for interacting with Prewhitener
    def _mask_selected_pts(self, event):
        """Mask lasso-selected observations after a delete-key event."""
        if ((event.key in ["backspace", "delete"]) and (len(self._selector.ind) > 0)):
            self.pw.mask_indices(self._selector.ind)
            self._selector.ind = []
            self._lcplot_data.set_facecolors([self._lc_colors[m]
                                            for m in self.pw.lc["include"]])
            self._lcplot_data.set_edgecolors("None")
            self._update_lc_display()
            self._update_per_plots()
            self._update_refit_button()
            self.update_log()

    def _clear_mask(self, _):
        """Restore all masked observations and refresh affected displays."""
        self.pw.clear_mask()
        self._selector.ind = []
        self._lcplot_data.set_facecolors([self._lc_colors[m]
                                         for m in self.pw.lc["include"]])
        self._lcplot_data.set_edgecolors("None")
        self._update_lc_display()
        self._update_per_plots()
        self._update_refit_button()
        self.update_log()

    # Periodogram related functions
    def _update_marker(self, x, y):
        """Move the candidate-signal marker and update its frequency and amplitude."""
        x = _as_scalar_float(x)
        y = _as_scalar_float(y)
        self._thisfreq.value = f"{x:.12g}"
        self._thisamp.value = y

        self._marker.set_data([x], [y])
        self.perfig.canvas.draw_idle()

    def _mark_highest_peak(self):
        """Move the candidate-signal marker to the highest residual peak."""
        self._update_marker(
            self.pw.freqs[np.nanargmax(self.pw.per_resid)],
            np.nanmax(self.pw.per_resid))

    def _onclick(self, event):
        """Handle a completed click in the periodogram."""
        self._onperiodogramclick(event)

    def _onpress(self, event):
        """Record the start of a mouse interaction in the periodogram."""
        self._press = True

    def _onmove(self, event):
        """Record mouse movement during a periodogram interaction."""
        if self._press:
            self._move = True

    def _onrelease(self, event):
        """Treat a press-and-release without movement as a periodogram click."""
        if self._press and not self._move:
            self._onclick(event)
        self._press = False
        self._move = False

    def _display_per_orig(self, *args):
        """Show or hide the original-light-curve periodogram."""
        if self._show_per_orig.value:
            self._perplot_orig.set_alpha(1)
        else:
            self._perplot_orig.set_alpha(0)
        self.perfig.canvas.draw_idle()

    def _display_per_resid(self, *args):
        """Show or hide the residual periodogram."""
        if self._show_per_resid.value:
            self._perplot_resid.set_alpha(1)
        else:
            self._perplot_resid.set_alpha(0)
        self.perfig.canvas.draw_idle()

    def _display_per_model(self, *args):
        """Show or hide the fitted-model periodogram."""
        if self._show_per_model.value:
            self._perplot_model.set_alpha(1)
        else:
            self._perplot_model.set_alpha(0)
        self.perfig.canvas.draw_idle()

    def _display_sig_threshold(self, *args):
        """Show or hide the significance-threshold curve."""
        if self._show_sig_threshold.value:
            self._sig_threshold_plot.set_alpha(1)
        else:
            self._sig_threshold_plot.set_alpha(0)
        self.perfig.canvas.draw_idle()

    def _display_per_markers(self, *args):
        """Show or hide markers for staged signals."""
        if self._show_per_markers.value:
            self._signal_markers.set_alpha(1)
            self._combo_markers.set_alpha(1)
        else:
            self._signal_markers.set_alpha(0)
            self._combo_markers.set_alpha(0)
        self.perfig.canvas.draw_idle()

    def _onperiodogramclick(self, event):
        """Select a candidate frequency from a periodogram click.

        If peak snapping is enabled, select the highest residual-periodogram
        peak near the clicked frequency. Otherwise, use the clicked frequency
        directly and interpolate its residual amplitude.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            Mouse event generated by the periodogram canvas.
        """
        if event.xdata is None:
            return

        if self._snaptopeak.value:
            # Click within either frequency resolution or 1% of displayed range.
            tolerance = np.max([
                self.pw.fres,
                0.01 * np.diff(self.perax.get_xlim())[0],
            ])

            nearby = np.where(
                (self.pw.freqs >= event.xdata - tolerance)
                & (self.pw.freqs <= event.xdata + tolerance)
            )[0]

            if nearby.size == 0:
                return

            local_power = self.pw.per_resid[nearby]

            if np.all(~np.isfinite(local_power)):
                return

            best_local = np.nanargmax(local_power)
            best_index = nearby[best_local]

            self._update_marker(
                self.pw.freqs[best_index],
                self.pw.per_resid[best_index],
            )

        else:
            self._update_marker(
                event.xdata,
                np.interp(event.xdata, self.pw.freqs, self.pw.per_resid),
            )

    ## Fitting things
    def _add_staged_signal(self, *args):
        """Stage the frequency currently entered in the GUI.

        A numeric value is staged as an independent signal. A valid
        combination expression is staged as a combination-frequency signal.
        Invalid input is reported to the Pyriod log.
        """
        # Is this a valid numeric frequency?
        if self._thisfreq.value.replace('.', '', 1).isdigit():
            self.pw.add_signal(float(self._thisfreq.value), self._thisamp.value)
            self._update_signals_qgrid()
        elif self.pw._valid_combo(self._thisfreq.value):
            self.pw.add_combination(self._thisfreq.value)
            self._update_signals_qgrid()
        else:
            self.pw.log(f"Staged frequency has invalid format: {self._thisfreq.value}", "error")
        self.update_log()

    def fit_model(self, *args):
        """Fit the staged signal model and refresh the GUI.

        Calls ``pw.fit_model()`` and then refreshes the displayed light curve,
        periodograms, signal markers, staged signal table, fit report, and log.

        The GUI displays a calculation-status indicator and temporarily disables
        the signal-addition and fitting buttons while the fit is running.
        """
        # Indicate that a calculation is running
        self._update_status(True)
        try:
            self.pw.fit_model()
            self.refresh_from_prewhitener()
            self._update_per_plots()
        finally:
            self._update_status(False)  # Calculation done

    def _sig_thresh_from_gui(self, *args):
        """Calculate a significance threshold from the current GUI settings."""
        fill_value = np.nan
        if self._sig_extrapolate_widget.value:
            fill_value = 'extrapolate'
        self.pw.calculate_significance_threshold(
            multiplier=self._sig_multiplier_widget.value,
            startfreq=self._sig_startfreq_widget.value,
            endfreq=self._sig_endfreq_widget.value,
            freqstep=self._sig_freqstep_widget.value,
            winwidth=self._sig_winwidth_widget.value,
            avgtype=self._sig_avgtype_widget.value,
            autorecalculate=self._sig_auto_recalculate.value)
        # update plot
        self._sig_threshold_plot.set_data(
            self.pw._sig_threshold_freqs,
            self.pw._sig_threshold_power,
        )
        self.perfig.canvas.draw_idle()

    
    def _sig_thresh_change_auto(self, *args):
        """Update automatic significance-threshold recalculation from the GUI."""
        self.pw.autorecalculate = self._sig_auto_recalculate.value

    def _update_fit_report(self):
        """Update the displayed report for the most recent model fit."""
        if self.pw.fit_result is None:
            self._fit_result_html.value = "No fit to report."
        else:
            self._fit_result_html.value = self.pw.fit_result._repr_html_()

    def _update_signals_qgrid(self):
        """Refresh the QGrid from the currently staged signal parameters."""
        self._signals_qgrid.df = self.pw.staged_table(display_units=True)
        self._update_refit_button()
        self._update_signal_markers()

    def _qgrid_changed_manually(self, *args):
        """Propagate manual QGrid edits to the staged signal table.

        Changes are recorded in the Pyriod log, converted from display
        amplitude units to internal units, and stored in
        ``pw.stagedvalues``. Dependent GUI displays are then refreshed.
        """
        # Note: args has information about what changed if needed
        newdf = self._signals_qgrid.get_changed_df()
        olddf = self._signals_qgrid.df

        logmessage = "Signals table changed manually.\n"

        for key in newdf.index.values:
            if key in olddf.index.values: # modified exitsting row
                changes = newdf.loc[key][(olddf.loc[key] != newdf.loc[key])]
                changes = changes.dropna()  # Remove nans
                if len(changes) > 0:
                    logmessage += f"Values changed for {key}:\n"
                    for colname, new_value in changes.items():
                        old_value = olddf.loc[key, colname]
                        logmessage += f" - {colname}: {old_value} -> {new_value}\n"
            else: #New row
                logmessage += f"New row in solution table: {key}\n"
                for colname, new_value in newdf.loc[key].items():
                    logmessage += f" - {colname} -> {new_value}\n"

        self.pw.log(logmessage)
        self.pw._set_stagedvalues(self._convert_qgrid_to_stagedvalues())
        self._update_refit_button()
        self._update_freq_dropdown()
        self._update_signal_markers()
        self._display_per_markers()
        self.update_log()

    def _convert_qgrid_to_stagedvalues(self):
        """Convert the displayed QGrid table to internal staged values.

        Returns
        -------
        pandas.DataFrame
            Signal table converted to the dtypes and internal amplitude
            units expected by `Prewhitener`.
        """
        tempdf = (self._signals_qgrid.get_changed_df().copy()
                  .astype(dtype=dict(zip(self.pw.columns, self.pw.dtypes))))
        tempdf["amp"] /= self.pw.amp_conversion
        tempdf["amperr"] /= self.pw.amp_conversion
        return tempdf

    def _delete_selected(self, *args):
        """Remove signals corresponding to the selected QGrid rows."""
        indices = self._signals_qgrid.get_selected_df().index
        existing = [idx for idx in indices if idx in self.pw.stagedvalues.index]
        if not existing:
            return
        self.pw.remove_signals(existing)

        self._update_freq_dropdown()
        self._update_signal_markers()
        self._update_signals_qgrid()
        self.update_log()

    def _save_button_click(self, *args):
        """Save the current fitted signal solution to the selected file."""
        self.pw.save_solution(filename=self._signals_file_location.selected)
        self.update_log()
    
    def _load_button_click(self, *args):
        """Load a signal solution from the selected file and refresh the table."""
        self.pw.load_solution(filename=self._signals_file_location.selected)
        self._update_signals_qgrid()
        self.update_log()

    ## Log functions
    def update_log(self):
        """Refresh the displayed log from ``pw.log_html``."""
        self._log.value = self.pw.log_html
    
    def log(self, message, level='info'):
        """Record a message in the Pyriod log and refresh the log display.

        Parameters
        ----------
        message : str
            Message to record.
        level : {"debug", "info", "warning", "error", "critical"}, optional
            Logging level passed to ``pw.log``. Default is ``"info"``.
        """
        self.pw.log(message, level=level)
        self.update_log()

    def _save_log_button_click(self, *args):
        """Save the Pyriod log using the filename and overwrite setting in the GUI."""
        self.pw.save_log(self._log_file_location.selected, self._overwrite.value)
        self.update_log()

    ## Properties for convenient access
    @property
    def lc(self):
        """Light curve associated with the underlying `Prewhitener`. 
        
        Returns 
        ------- 
        lightkurve.LightCurve 
            ``pw.lc``. 
        """
        return self.pw.lc

    @property
    def fitvalues(self):
        """Parameters of the most recently fitted signal model.

        Returns
        -------
        pandas.DataFrame
            ``pw.fitvalues``.
        """
        return self.pw.fitvalues

    @property
    def stagedvalues(self):
        """Signal parameters staged for the next model fit.

        Returns
        -------
        pandas.DataFrame
            ``pw.stagedvalues``.
        """
        return self.pw.stagedvalues

    ### Functions for extending the GUI to include additional analysis tabs.
    def add_analysis(self, analysis_class, **kwargs):
        """Add an optional analysis interface as a new tab."""

        if (
            not isinstance(analysis_class, type)
            or not issubclass(analysis_class, AnalysisGUI)
        ):
            raise TypeError(
                "analysis_class must be a subclass of AnalysisGUI."
            )

        if not hasattr(self, "_analyses"):
            self._analyses = {}

        name = analysis_class.__name__

        if name in self._analyses:
            raise ValueError(
                f"{analysis_class.__name__} is already open."
            )

        analysis = analysis_class(
            self.pw,
            log_updated=self.update_log,
            request_close=self.remove_analysis,
            **kwargs,
        )

        self._analyses[name] = analysis

        self._rebuild_tabs()

        # Select the newly added tab.
        self._tabs.selected_index = len(self._tabs.children) - 1

        return analysis     

    def remove_analysis(self, analysis):
        """Close and remove an optional analysis tab."""

        if isinstance(analysis, str):
            name = analysis

        elif isinstance(analysis, type):
            if not issubclass(analysis, AnalysisGUI):
                raise TypeError(
                    "analysis must be an AnalysisGUI instance, subclass, "
                    "or class-name string."
                )
            name = analysis.__name__

        elif isinstance(analysis, AnalysisGUI):
            name = type(analysis).__name__

        else:
            raise TypeError(
                "analysis must be an AnalysisGUI instance, subclass, "
                "or class-name string."
            )

        open_analysis = self._analyses.pop(name, None)

        if open_analysis is None:
            return False

        open_analysis.close()
        self._rebuild_tabs()

        return True 

    def close(self, close_prewhitener=False, clear_prewhitener=False, collect=False):
        """Close resources owned by this GUI.

        Matplotlib callbacks and timers are disconnected, widget callbacks are
        removed, plot artists and figures are closed, and references held by the
        GUI are released.

        Parameters
        ----------
        close_prewhitener : bool, optional
            If True, also close the associated ``Prewhitener`` after closing
            the GUI. Default is False.
        clear_prewhitener : bool, optional
            Value passed as ``clear_data`` to ``pw.close`` when
            ``close_prewhitener=True``. If True, the `Prewhitener` also releases
            its large analysis data objects. Default is False.
        collect : bool, optional
            If True, explicitly run Python garbage collection after resources
            are released. Usually unnecessary, but potentially useful after
            creating many GUI instances in a long-running notebook. Default is
            False.

        Notes
        -----
        Closing the GUI does not close the associated `Prewhitener` unless
        ``close_prewhitener=True``.
        """
        if getattr(self, "_closed", False):
            return

        self._closed = True

        def safe_call(obj, method, *args, **kwargs):
            if obj is None:
                return
            try:
                getattr(obj, method)(*args, **kwargs)
            except Exception:
                pass

        def safe_attr(name):
            return getattr(self, name, None)

        # Close optional analysis tabs
        analyses = getattr(self, "_analyses", {})

        for analysis in list(analyses.values()):
            try:
                analysis.close()
            except Exception:
                pass

        analyses.clear()

        # ------------------------------------------------------------------
        # 1. Stop and detach Matplotlib timers
        # ------------------------------------------------------------------
        for timer_name in ("_model_update_timer", "_periodogram_update_timer"):
            timer = safe_attr(timer_name)
            safe_call(timer, "stop")

            # Matplotlib timers keep callbacks in a list. Clearing it helps
            # break references back to self.
            try:
                timer.callbacks.clear()
            except Exception:
                pass

            setattr(self, timer_name, None)

        # ------------------------------------------------------------------
        # 2. Disconnect axis callbacks
        # ------------------------------------------------------------------
        lcax = safe_attr("lcax")
        if lcax is not None:
            cid = safe_attr("_model_xlim_callback_id")
            if cid is not None:
                safe_call(lcax.callbacks, "disconnect", cid)
            self._model_xlim_callback_id = None

        perax = safe_attr("perax")
        if perax is not None:
            cid = safe_attr("_periodogram_xlim_callback_id")
            if cid is not None:
                safe_call(perax.callbacks, "disconnect", cid)
            self._periodogram_xlim_callback_id = None

        # ------------------------------------------------------------------
        # 3. Disconnect Matplotlib canvas callbacks
        # ------------------------------------------------------------------
        mpl_callback_pairs = [
            ("lcfig", "_lc_key_press_callback_id"),
            ("perfig", "_per_button_press_callback_id"),
            ("perfig", "_per_button_release_callback_id"),
            ("perfig", "_per_motion_callback_id"),
        ]

        for fig_name, cid_name in mpl_callback_pairs:
            fig = safe_attr(fig_name)
            cid = safe_attr(cid_name)
            if fig is not None and cid is not None:
                safe_call(fig.canvas, "mpl_disconnect", cid)
            setattr(self, cid_name, None)

        # ------------------------------------------------------------------
        # 4. Disconnect lasso selector
        # ------------------------------------------------------------------
        selector = safe_attr("_selector")
        if selector is not None:
            # Depending on the implementation, either the wrapper or the
            # underlying Matplotlib LassoSelector may own the connections.
            safe_call(selector, "disconnect")
            safe_call(selector, "disconnect_events")

            lasso = getattr(selector, "lasso", None)
            safe_call(lasso, "set_active", False)
            safe_call(lasso, "disconnect_events")

            self._selector = None

        # ------------------------------------------------------------------
        # 5. Disconnect ipywidget observers and button callbacks
        # ------------------------------------------------------------------
        def safe_unobserve(widget_name, callback, names=None):
            widget = safe_attr(widget_name)
            if widget is not None:
                try:
                    widget.unobserve(callback, names=names)
                except Exception:
                    pass

        def safe_remove_click(button_name, callback):
            button = safe_attr(button_name)
            if button is not None:
                try:
                    button.on_click(callback, remove=True)
                except Exception:
                    pass

        # Button callbacks
        safe_remove_click("_save_tsfig", self._save_tsfig_button_click)
        safe_remove_click("_reset_mask", self._clear_mask)
        safe_remove_click("_save_perfig", self._save_perfig_button_click)
        safe_remove_click("_addtosol", self._add_staged_signal)
        safe_remove_click("_refit", self.fit_model)
        safe_remove_click("_sig_calculate_button", self._sig_thresh_from_gui)
        safe_remove_click("_delete", self._delete_selected)
        safe_remove_click("_save", self._save_button_click)
        safe_remove_click("_load", self._load_button_click)
        safe_remove_click("_save_log", self._save_log_button_click)

        # Observer callbacks
        safe_unobserve("_tstype", self._update_and_rescale_lc_display)
        safe_unobserve("_fold", self._update_and_rescale_lc_display)
        safe_unobserve("_fold_on", self._update_lc_display)
        safe_unobserve("_select_fold_freq", self._fold_freq_selected, names="value")

        safe_unobserve("_show_per_markers", self._display_per_markers)
        safe_unobserve("_show_per_orig", self._display_per_orig)
        safe_unobserve("_show_per_resid", self._display_per_resid)
        safe_unobserve("_show_per_model", self._display_per_model)
        safe_unobserve("_show_sig_threshold", self._display_sig_threshold)
        safe_unobserve("_sig_auto_recalculate", self._sig_thresh_change_auto)

        # QGrid callback
        qgrid_widget = safe_attr("_signals_qgrid")
        if qgrid_widget is not None:
            try:
                qgrid_widget.off("cell_edited", self._qgrid_changed_manually)
            except Exception:
                pass

        # ------------------------------------------------------------------
        # 6. Remove Matplotlib artists
        # ------------------------------------------------------------------
        artist_names = [
            "_lcplot_data",
            "_lcplot_model",
            "_perplot_orig",
            "_perplot_model",
            "_perplot_resid",
            "_sig_threshold_plot",
            "_marker",
            "_signal_markers",
            "_combo_markers",
        ]

        for name in artist_names:
            artist = safe_attr(name)
            if artist is not None:
                safe_call(artist, "remove")
            setattr(self, name, None)

        # ------------------------------------------------------------------
        # 7. Close figure canvases and figures
        # ------------------------------------------------------------------
        for fig_name in ("lcfig", "perfig"):
            fig = safe_attr(fig_name)
            if fig is not None:
                safe_call(fig.canvas, "close")
                try:
                    plt.close(fig)
                except Exception:
                    pass
            setattr(self, fig_name, None)

        self.lcax = None
        self.perax = None

        # ------------------------------------------------------------------
        # 8. Close widgets
        # ------------------------------------------------------------------
        # This catches ipywidgets, FileChooser widgets, qgrid widgets, and
        # ipympl canvases if any are still referenced.
        for name, obj in list(self.__dict__.items()):
            if name == "pw":
                continue

            if hasattr(obj, "close"):
                try:
                    obj.close()
                except Exception:
                    pass

        # ------------------------------------------------------------------
        # 9. Drop GUI references that may keep large objects alive
        # ------------------------------------------------------------------
        for name in list(self.__dict__):
            if name in {"pw", "_closed"}:
                continue
            setattr(self, name, None)

        # ------------------------------------------------------------------
        # 10. Optionally close the associated Prewhitener
        # ------------------------------------------------------------------
        if close_prewhitener and self.pw is not None:
            self.pw.close(clear_data=clear_prewhitener, collect=False)
            self.pw = None

        if collect:
            import gc
            gc.collect()