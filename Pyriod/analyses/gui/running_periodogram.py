"""GUI for running-periodogram analysis."""

from __future__ import annotations

import asyncio

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from traitlets import TraitError

from .base import AnalysisGUI
from ..running_periodogram import (
    normalize_running_periodogram,
    running_periodogram,
)


class RunningPeriodogramGUI(AnalysisGUI):
    """Interactive running-periodogram analysis tab."""

    title = "Running Periodogram"

    def __init__(
        self,
        pw,
        *,
        log_updated=None,
        busy_changed=None,
        request_close=None,
    ):
        super().__init__(
            pw,
            log_updated=log_updated,
            busy_changed=busy_changed,
        )

        self._request_close_callback = request_close
        self.results = None
        self._mesh = None
        self._colorbar = None

        self._init_figure()
        self._init_widgets()
        self._init_callbacks()
        self._init_layout()

    def _init_figure(self):
        """Create and register the running-periodogram figure."""
        self.fig = plt.figure(figsize=(7, 4))

        gs = self.fig.add_gridspec(
            1,
            2,
            width_ratios=[30, 1],
            wspace=0.08,
        )

        self.ax = self.fig.add_subplot(gs[0, 0])
        self.cax = self.fig.add_subplot(gs[0, 1])

        self._track_figure(self.fig)
        self.fig.canvas.header_visible = False

        self.ax.set_xlabel(
            f"frequency ({self.pw.freq_unit.to_string()})"
        )
        self.ax.set_ylabel(
            f"time ({self.pw.time_unit.to_string()})"
        )

        self.cax.set_ylabel("amplitude")

        self.cax.set_visible(False)

    def _init_widgets(self):
        """Create running-periodogram controls."""
        duration = float(
            self.pw.lc.time.value.max()
            - self.pw.lc.time.value.min()
        )

        self._timeseries = widgets.Dropdown(
            options=[
                ("Original", "original"),
                ("Model fit", "model"),
                ("Residuals", "residual"),
            ],
            value="original",
            description="Time series:",
            style={"description_width": "initial"},
        )

        self._window_width = widgets.FloatText(
            value=max(duration / 10.0, 1e-6),
            description="Window width (d):",
            style={"description_width": "initial"},
        )

        self._step_size = widgets.FloatText(
            value=max(duration / 20.0, 1e-6),
            description="Step size (d):",
            style={"description_width": "initial"},
        )

        self._minfreq = widgets.FloatText(
            value=float(np.nanmin(self.pw.freqs)),
            description="Min frequency:",
            style={"description_width": "initial"},
        )

        self._maxfreq = widgets.FloatText(
            value=float(np.nanmax(self.pw.freqs)),
            description="Max frequency:",
            style={"description_width": "initial"},
        )

        self._oversample = widgets.FloatText(
            value=5.0,
            description="Oversampling:",
            style={"description_width": "initial"},
        )

        self._min_points = widgets.IntText(
            value=3,
            description="Minimum points:",
            style={"description_width": "initial"},
        )

        self._normalization = widgets.Dropdown(
            options=[
                ("Amplitude", "amplitude"),
                ("Each window / peak", "peak"),
                ("Each window / median", "median"),
                ("Entire map / global peak", "global_peak"),
            ],
            value="amplitude",
            description="Normalization:",
            style={"description_width": "initial"},
        )

        self._vmin = widgets.FloatText(
            value=np.nan,
            description="Color min:",
            style={"description_width": "initial"},
        )

        self._vmax = widgets.FloatText(
            value=np.nan,
            description="Color max:",
            style={"description_width": "initial"},
        )

        self._colormap = widgets.Dropdown(
            options=[
                "viridis",
                "plasma",
                "inferno",
                "magma",
                "gray",
                "Purples",
            ],
            value="viridis",
            description="Colormap:",
            style={"description_width": "initial"},
        )

        self._calculate = widgets.Button(
            description="Calculate",
            tooltip="Calculate running periodograms.",
            icon="refresh",
        )

        self._result = widgets.HTML(
            value="<b>No running periodogram calculated.</b>"
        )

        self._close = widgets.Button(
            description="Close",
            tooltip="Close this analysis tab.",
            icon="times",
        )

    def _init_callbacks(self):
        """Register widget callbacks."""
        self._on_click(
            self._calculate,
            self._calculate_clicked,
        )
        self._on_click(
            self._close,
            self._close_clicked,
        )
        for widget in (
            self._normalization,
            self._colormap,
            self._vmin,
            self._vmax,
        ):
            self._observe(
                widget,
                self._display_changed,
                names="value",
            )

    def _init_layout(self):
        """Construct the top-level analysis widget."""
        analysis_options = widgets.VBox(
            [
                widgets.HTML("<b>Analysis</b>"),
                self._timeseries,
                self._window_width,
                self._step_size,
                self._minfreq,
                self._maxfreq,
                self._oversample,
                self._min_points,
                self._calculate,
            ],
            layout=widgets.Layout(
                border="solid 1px",
                padding="6px",
                width="50%",
            ),
        )

        display_options = widgets.VBox(
            [
                widgets.HTML("<b>Display</b>"),
                self._normalization,
                self._colormap,
                self._vmin,
                self._vmax,
            ],
            layout=widgets.Layout(
                border="solid 1px",
                padding="6px",
                width="50%",
            ),
        )

        options = widgets.HBox(
            [
                analysis_options,
                display_options,
            ],
            layout=widgets.Layout(width="100%"),
        )

        accordion = widgets.Accordion(
            children=[options],
            selected_index=0,
        )
        accordion.set_title(0, "options")

        close_row = widgets.HBox(
            [self._close],
            layout=widgets.Layout(
                justify_content="flex-end",
            ),
        )

        try:
            interface = widgets.VBox(
                [
                    close_row,
                    self._status,
                    self.progress_widget,
                    self.fig.canvas,
                    self._result,
                    accordion,
                ]
            )
        except TraitError as exc:
            exc.add_note(
                "You must use the ipympl plotting backend. "
                "Use magic command `%matplotlib widget`."
            )
            raise

        self._set_widget(interface)

    def _set_analysis_busy(self, busy):
        """Enable or disable analysis-specific controls."""
        self._timeseries.disabled = busy
        self._window_width.disabled = busy
        self._step_size.disabled = busy
        self._minfreq.disabled = busy
        self._maxfreq.disabled = busy
        self._oversample.disabled = busy
        self._min_points.disabled = busy
        self._calculate.disabled = busy

        # Normalization is deliberately left enabled because changing it
        # only redraws already-calculated results.

    def _calculate_clicked(self, _):
        """Start a running-periodogram calculation."""
        if self.busy:
            return

        try:
            self._start_task(
                self._calculate_running_periodogram()
            )
        except Exception as exc:
            self._log_exception(
                exc,
                context=(
                    "Could not start running-periodogram calculation"
                ),
            )

    async def _calculate_running_periodogram(self):
        """Calculate the periodogram sequence in a worker thread."""
        window_width = self._window_width.value
        step_size = self._step_size.value
        minfreq = self._minfreq.value
        maxfreq = self._maxfreq.value
        oversample = self._oversample.value
        min_points = self._min_points.value
        timeseries = self._timeseries.value

        if window_width <= 0 or step_size <= 0:
            self._result.value = (
                "<b>Window width and step size must be positive.</b>"
            )
            return

        if maxfreq <= minfreq:
            self._result.value = (
                "<b>Maximum frequency must exceed minimum frequency.</b>"
            )
            return

        if oversample <= 0:
            self._result.value = (
                "<b>Oversampling must be positive.</b>"
            )
            return

        if min_points < 3:
            self._result.value = (
                "<b>Minimum points must be at least 3.</b>"
            )
            return

        time = self.pw.lc.time.value
        duration = float(time.max() - time.min())
        total = max(
            1,
            int(duration // step_size) + 1,
        )

        self._set_analysis_busy(True)
        self._start_progress(
            total,
            message="CALCULATING RUNNING PERIODOGRAM...",
        )

        self._log(
            "Calculating running periodogram: "
            f"time series = {timeseries}, "
            f"window width = {window_width} d, "
            f"step size = {step_size} d, "
            f"frequency range = {minfreq:.6g}–{maxfreq:.6g} "
            f"{self.pw.freq_unit}, "
            f"oversample = {oversample}."
        )

        loop = asyncio.get_running_loop()

        def progress_callback(completed, requested):
            loop.call_soon_threadsafe(
                self._set_progress,
                completed,
                requested,
            )

        result = await self._run_analysis_in_thread(
            running_periodogram,
            self.pw,
            window_width,
            step_size,
            minimum_frequency=minfreq,
            maximum_frequency=maxfreq,
            oversample=oversample,
            normalization=self._normalization.value,
            timeseries=timeseries,
            min_points=min_points,
            progress_callback=progress_callback,
            cancel_check=self._cancel_check,
            context="Running-periodogram calculation failed",
        )

        if self.closed:
            return

        self._set_analysis_busy(False)

        if result is None:
            self._result.value = (
                "<b>Running-periodogram calculation failed. "
                "See Log.</b>"
            )
            self._fail_progress(
                "RUNNING-PERIODOGRAM CALCULATION FAILED"
            )
            return

        self.results = result

        if self.cancel_requested:
            self._abort_progress(
                "RUNNING-PERIODOGRAM CALCULATION CANCELLED"
            )
            self._log(
                "Running-periodogram calculation cancelled "
                f"after {len(result.time)} completed windows.",
                level="warning",
            )
        else:
            self._finish_progress()
            self._log(
                "Running-periodogram calculation completed "
                f"with {len(result.time)} windows and "
                f"{len(result.frequency)} frequencies."
            )

        self._draw_results()

    @staticmethod
    def _centers_to_edges(values, fallback_width=1.0):
        """Convert monotonic bin centers to plotting edges."""
        values = np.asarray(values, dtype=float)

        if len(values) == 0:
            return np.asarray([], dtype=float)

        if len(values) == 1:
            half_width = 0.5 * float(fallback_width)
            return np.asarray(
                [
                    values[0] - half_width,
                    values[0] + half_width,
                ]
            )

        midpoints = 0.5 * (
            values[:-1] + values[1:]
        )

        first = (
            values[0]
            - 0.5 * (values[1] - values[0])
        )
        last = (
            values[-1]
            + 0.5 * (values[-1] - values[-2])
        )

        return np.concatenate(
            ([first], midpoints, [last])
        )

    def _draw_results(self):
        """Draw the current running-periodogram map."""
        self.ax.clear()

        if (
            self.results is None
            or len(self.results.time) == 0
        ):
            self._result.value = (
                "<b>No running-periodogram windows were completed.</b>"
            )
            self.fig.canvas.draw_idle()
            return

        normalization = self._normalization.value

        values = normalize_running_periodogram(
            self.results.amplitude,
            normalization=normalization,
        )

        frequency_edges = self._centers_to_edges(
            self.results.frequency,
        )

        time_edges = self._centers_to_edges(
            self.results.time,
            fallback_width=(
                self.results.window_width
                if self.results.window_width is not None
                else 1.0
            ),
        )

        vmin = self._vmin.value
        vmax = self._vmax.value

        if not np.isfinite(vmin):
            vmin = None

        if not np.isfinite(vmax):
            vmax = None

        self._mesh = self.ax.pcolormesh(
            frequency_edges,
            time_edges,
            values,
            shading="flat",
            cmap=self._colormap.value,
            vmin=vmin,
            vmax=vmax,
        )

        self.cax.clear()
        self.cax.set_visible(True)

        self._colorbar = self.fig.colorbar(
            self._mesh,
            cax=self.cax,
        )

        if normalization == "amplitude":
            amp_unit = getattr(
                self.pw,
                "amp_unit",
                "",
            )
            label = "amplitude"
            if amp_unit:
                label += f" ({amp_unit})"
        elif normalization == "peak":
            label = "amplitude / window peak"
        elif normalization == "median":
            label = "amplitude / window median"
        else:
            label = "amplitude / global peak"

        self._colorbar.set_label(label)

        self.ax.set_xlabel(
            f"frequency ({self.pw.freq_unit.to_string()})"
        )
        self.ax.set_ylabel(
            f"time ({self.pw.time_unit.to_string()})"
        )
        if self.results.window_width is not None:
            self.ax.set_title(
                f"{self.results.window_width:g}-day window"
            )
        else:
            self.ax.set_title("Running periodogram")

        self.fig.canvas.draw_idle()

        self._result.value = (
            f"<b>{len(self.results.time)}</b> windows; "
            f"<b>{len(self.results.frequency)}</b> frequencies."
        )

    def _display_changed(self, change):
        """Redraw existing results using the current display options."""
        if self.results is None:
            return

        self._draw_results()

    def _close_clicked(self, _):
        """Request removal of this analysis tab."""
        if self._request_close_callback is not None:
            self._request_close_callback(self)
        else:
            self.close()

    def refresh(self):
        """No expensive recalculation is performed automatically."""
        return None
