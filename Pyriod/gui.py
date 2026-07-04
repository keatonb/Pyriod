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

plt.ioff()  # Turn off interactive mode

class PyriodGUI:
    """Interactive widgets and plots for prewhitening.
    
    This object owns the display/interaction elements.
    Must be connected to a pw = Prewhitener object.
    """
    def __init__(self, pw):
        self.pw = pw

        # Create status widget to indicate when calculations are running
        self._status = widgets.HTML(value="")

        # Figures first so the widgets can connect
        #self._init_figures()
        #self._init_widgets()

        self._init_timeseries_widgets()
        self._init_timeseries_figures()
        self._init_periodogram_widgets()
        self._init_peridogram_figures()
        #self.refresh_all()


    ## Initialize Widgets
    def _init_widgets(self):
        self._init_timeseries_widgets()
        self._init_periodogram_widgets()
        #self._init_signals_widgets()
        #self._init_log_widgets()

    
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
        self._tstype.observe(self._update_and_rescale_lc_display)

        # Fold on frequency checkbox
        self._fold = widgets.Checkbox(
            value=False,
            step=self.pw.fres,
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
        # Calulate sig threshold button
        self._sig_calculate_button = widgets.Button(
            description='Calculate',
            tooltip="Calculate new sigificance threshold.",
            style={'description_width': 'initial'}
        )
        self._sig_calculate_button.on_click(self._sigthreshfromgui)



    ## Initialize figures
    def _init_figures(self):
        self._init_timeseries_figures()
        self._init_peridogram_figures()

    def _init_timeseries_figures(self):
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
        self.lcfig.canvas.mpl_connect("key_press_event",
                                        self._mask_selected_pts)
        # Set to display sampled model
        self._init_viewport_model_plot()
    
    def _set_timeseries_plot_labels(self):
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
        self._model_update_timer.stop()
        self._model_update_timer.start()

    def _update_sampled_model(self):
        self._last_model_xlim = None
        self._request_model_line_update()

    def _refresh_model_line_from_view(self):
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
        # Sample the model sensibly within the current viewport
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


    def _init_peridogram_figures(self):
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
        self.perfig.canvas.mpl_connect('button_press_event', self._onpress)
        self.perfig.canvas.mpl_connect('button_release_event',
                                        self._onrelease)
        self.perfig.canvas.mpl_connect('motion_notify_event', self._onmove)

        # Set axis labels
        self.perax.set_ylabel(f"amplitude ({self.pw.amp_unit})")
        self.perax.set_xlabel(f"frequency ({self.pw._freq_label})")
        self.lcfig.canvas.draw_idle()

    # Set up all the efficient viewport stuff here for periodogram plot
    def _init_viewport_periodogram_plot(self):
        """Initialize responsive, decimated periodogram plotting."""

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
        """Request a debounced periodogram redraw."""

        self._periodogram_update_timer.stop()
        self._periodogram_update_timer.start()


    def _update_per_plots(self):
        """Update periodogram plot data after periodograms are recomputed."""

        self._last_periodogram_xlim = None
        self._request_periodogram_plot_update()

    def _refresh_periodogram_lines_from_view(self):
        """Display only a decimated version of the visible frequency range."""

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
        """Set one periodogram line using only visible, decimated data."""

        xplot, yplot = decimate_visible_range(
            self.pw.freqs,
            power,
            xmin,
            xmax,
            max_points=self.max_periodogram_plot_points,
        )

        line.set_data(xplot, yplot)


    ## Main Widget collections
    def TimeSeries(self):
        """Display the interactive Time Series cell in a Jupyter notebook.

        Returns
        -------
        widget
            Time series plot, options, and information to be displayed.
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
        """Display the interactive Periodogram cell in a Jupyter notebook.

        Returns
        -------
        widget
            Periodogram plot, options, and information to be displayed.
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



    def Pyriod(self):
        """Display the interactive Pyriod suite in a Jupyter notebook.

        Includes the Time Series, Periodogram, Signals, and Log widgets in
        separate tabs.

        Returns
        -------
        widget
            Time Series, Periodogram, Signals, and Log widgets in tabs.
        """
        tstab = self.TimeSeries()
        pertab = self.Periodogram()
        #signalstab = self.Signals()
        #logtab = self.Log()
        #tabs = widgets.Tab(children=[tstab, pertab, signalstab, logtab])
        tabs = widgets.Tab(children=[tstab, pertab])
        tabs.set_title(0, 'Time Series')
        tabs.set_title(1, 'Periodogram')
        #tabs.set_title(2, 'Signals')
        #tabs.set_title(3, 'Log')
        return tabs

    # Functions for saving plots
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
        self.lcfig.savefig(filename, **kwargs)

    # Plot widget-related functions
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
        self.perfig.savefig(filename, **kwargs)

    def _save_perfig_button_click(self, *args):
        self.save_perfig(self._perfig_file_location.selected)

    # Update status message while calculations are occurring
    def _update_status(self, calculating=True):
        if calculating:
            self._status.value = (
                "<center><b><big><font color='red'>"
                "UPDATING CALCULATIONS...</big></b></center>")
        else:
                    self._status.value = ""
    
    # Functions to update displays
    def _update_lc_display(self, *args):
        """Change type of time series to display from dropdown."""
        self._display_lc(residuals=(self._tstype.value == "Residuals"))

    def _update_and_rescale_lc_display(self, *args):
        """Change type of time series to display from dropdown."""
        try:
            self.pw.log(str(*args))
            self._display_lc(residuals=(self._tstype.value == "Residuals"),rescale=True)
        except Exception as e:
            self.pw.log(f"Error caught: {e}","error")

    def _update_signal_markers(self):
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
        # New frequency value selected to fold on
        if value['new'] is not None:
            self._fold_on.value = value['new']

    ## Functions for interacting with Prewhitener
    def _mask_selected_pts(self, event):
        if ((event.key in ["backspace", "delete"]) and (len(self._selector.ind) > 0)):
            self.pw.mask_indices(self._selector.ind)
            self._selector.ind = []
            self._lcplot_data.set_facecolors([self._lc_colors[m]
                                            for m in self.pw.lc["include"]])
            self._lcplot_data.set_edgecolors("None")
            self._update_lc_display()
            self._update_per_plots()

    def _clear_mask(self, _):
        self.pw.clear_mask()
        self._selector.ind = []
        self._lcplot_data.set_facecolors([self._lc_colors[m]
                                         for m in self.lc["include"]])
        self._lcplot_data.set_edgecolors("None")
        self._update_lc_display()
        self._update_per_plots()

    # Periodogram related functions

    def _update_marker(self, x, y):
        # Move the signal marker to the currently selected periodogram peak.
        x = _as_scalar_float(x)
        y = _as_scalar_float(y)
        self._thisfreq.value = f"{x:.12g}"
        self._thisamp.value = y

        self._marker.set_data([x], [y])
        self.perfig.canvas.draw_idle()

    def _mark_highest_peak(self):
        # Move signal marker to current highest peak.
        self._update_marker(
            self.pw.freqs[np.nanargmax(self.pw.per_resid)],
            np.nanmax(self.pw.per_resid))

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

    def _display_per_orig(self, *args):
        if self._show_per_orig.value:
            self._perplot_orig.set_alpha(1)
        else:
            self._perplot_orig.set_alpha(0)
        self.perfig.canvas.draw_idle()

    def _display_per_resid(self, *args):
        if self._show_per_resid.value:
            self._perplot_resid.set_alpha(1)
        else:
            self._perplot_resid.set_alpha(0)
        self.perfig.canvas.draw_idle()

    def _display_per_model(self, *args):
        if self._show_per_model.value:
            self._perplot_model.set_alpha(1)
        else:
            self._perplot_model.set_alpha(0)
        self.perfig.canvas.draw_idle()

    def _display_sig_threshold(self, *args):
        if self._show_sig_threshold.value:
            self._sig_threshold_plot.set_alpha(1)
        else:
            self._sig_threshold_plot.set_alpha(0)
        self.perfig.canvas.draw_idle()

    def _display_per_markers(self, *args):
        if self._show_per_markers.value:
            self._signal_markers.set_alpha(1)
            self._combo_markers.set_alpha(1)
        else:
            self._signal_markers.set_alpha(0)
            self._combo_markers.set_alpha(0)
        self.perfig.canvas.draw_idle()

    def _onperiodogramclick(self, event):
        """Handle clicks in the periodogram plot."""

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

    def _add_staged_signal(self, *args):
        """Add signal to set of signals to fit."""
        # Is this a valid numeric frequency?
        if self._thisfreq.value.replace('.', '', 1).isdigit():
            self.pw.add_signal(float(self._thisfreq.value), self._thisamp.value)
        elif self.pw._valid_combo(self._thisfreq.value):
            self.pw.add_combination(self._thisfreq.value)
        else:
            self.pw.log(f"Staged frequency has invalid format: {self._thisfreq.value}", "error")

    def fit_model(self, *args):
        """Fit model and update plots
        """
        # Indicate that a calculation is running
        try:
            self._update_status()
            self.pw.fit_model()
            self._update_lc_display()
            self._update_signal_markers()
            self._update_per_plots()
            self._mark_highest_peak()  # Mark highest peak in residuals
            #self._update_fit_report()

            self._update_status(False)  # Calculation done
        except Exception as e:
            # Forcibly write the traceback to stderr so it displays
            self.pw.log(f"Error caught: {e}","error")


    def _sigthreshfromgui(self, *args):
        # Compute significance threshold from GUI widget values
        fill_value = np.nan
        if self._sig_extrapolate_widget.value:
            fill_value = 'extrapolate'
        self.pw.calculate_significance_threshold(
            self._sig_multiplier_widget.value,
            self._sig_startfreq_widget.value,
            self._sig_endfreq_widget.value,
            self._sig_freqstep_widget.value,
            self._sig_winwidth_widget.value,
            self._sig_avgtype_widget.value,
            fill_value = fill_value)    
        # update plot
        self._sig_threshold_plot.set_data(
            self.pw._sig_threshold_freqs,
            self.pw._sig_threshold_power,
        )
        self.perfig.canvas.draw_idle()
        


    ## Properties for convenient access
    @property
    def lc(self):
        return self.pw.lc

    @property
    def fitvalues(self):
        return self.pw.fitvalues

    @property
    def stagedvalues(self):
        return self.pw.stagedvalues
    