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
        self._init_figures()
        self._init_widgets()
        #self.refresh_all()

    ## Initialize Widgets
    def _init_widgets(self):
        self._init_timeseries_widgets()
        #self._init_peridogram_widgets()
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


    ## Initialize figures
    def _init_figures(self):
        self._init_timeseries_figures()
        #self._init_peridogram_figures()

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
            # Add your custom contextual information here
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
        #pertab = self.Periodogram()
        #signalstab = self.Signals()
        #logtab = self.Log()
        #tabs = widgets.Tab(children=[tstab, pertab, signalstab, logtab])
        tabs = widgets.Tab(children=[tstab])
        tabs.set_title(0, 'Time Series')
        #tabs.set_title(1, 'Periodogram')
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
    
    # Update status message while calculations are occurring
    def _update_status(self, calculating=True):
        if self.gui:
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
            #self._update_per_plots()

    def _clear_mask(self, _):
        self.pw.clear_mask()
        self._selector.ind = []
        self._lcplot_data.set_facecolors([self._lc_colors[m]
                                         for m in self.lc["include"]])
        self._lcplot_data.set_edgecolors("None")
        self._update_lc_display()
        #self._update_per_plots()
    
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
    