"""GUI for spectral-window analysis."""

from __future__ import annotations

import asyncio

import ipywidgets as widgets
import matplotlib.pyplot as plt
from traitlets import TraitError
import numpy as np

from .base import AnalysisGUI
from ..spectral_window import spectral_window


class SpectralWindowGUI(AnalysisGUI):
    """Interactive spectral-window analysis tab."""

    title = "Spectral Window"

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

        self.frequencies = None
        self.amplitudes = None

        self._init_figure()
        self._init_widgets()
        self._init_callbacks()
        self._init_layout()

        self._calculate_clicked(None)

    def _init_figure(self):
        self.fig, self.ax = plt.subplots(
            figsize=(7, 3)
        )
        self._track_figure(self.fig)

        self.fig.canvas.header_visible = False

        self.ax.set_xlabel(
            f"Frequency ({self.pw.freq_unit})"
        )
        self.ax.set_ylabel("spectral window amplitude")

        plt.tight_layout()

    def _init_widgets(self):
        self._maxfreq = widgets.FloatText(
            value=min([self.pw.nyquist, 100]),
            description="Max frequency:",
            style={"description_width": "initial"},
        )

        self._oversample = widgets.FloatText(
            value=10,
            description="Oversampling:",
            style={"description_width": "initial"},
        )

        self._calculate = widgets.Button(
            description="Calculate",
            icon="refresh",
        )

        self._close = widgets.Button(
            description="Close",
            icon="times",
        )

        self._result = widgets.HTML(
            value="",
        )

    def _init_callbacks(self):
        self._on_click(
            self._calculate,
            self._calculate_clicked,
        )

        self._on_click(
            self._close,
            self._close_clicked,
        )

    def _init_layout(self):
        options = widgets.VBox(
            [
                self._maxfreq,
                self._oversample,
                self._calculate,
            ],
            layout=widgets.Layout(
                border="solid 1px",
                padding="6px",
            ),
        )

        accordion = widgets.Accordion(
            children=[options],
            selected_index=None,
        )
        accordion.set_title(0, "options")

        close_row = widgets.HBox(
            [self._close],
            layout=widgets.Layout(
                justify_content="flex-end"
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
                "Use `%matplotlib widget`."
            )
            raise

        self._set_widget(interface)

    def _calculate_clicked(self, _):
        """Start a spectral-window calculation."""
        if self.busy:
            return

        try:
            self._start_task(self._calculate_window())
        except Exception as exc:
            self._log_exception(
                exc,
                context="Could not start spectral-window calculation",
            )

    async def _calculate_window(self):
        """Calculate the spectral window without blocking the notebook GUI."""
        maxfreq = self._maxfreq.value
        oversample = self._oversample.value

        # Estimate the number of frequency evaluations so the progress bar
        # has a meaningful total before the worker thread starts.
        included = np.asarray(self.pw.lc["include"], dtype=bool)
        time = self.pw.lc.time.value[included]

        if len(time) < 2:
            self._result.value = (
                "<b>Spectral-window calculation failed. See Log.</b>"
            )
            self._log(
                "Spectral-window calculation failed: "
                "at least two included observations are required.",
                level="error",
            )
            return

        fres = 1.0 / (
            self.pw.freq_conversion * np.ptp(time)
        )
        step = fres / oversample

        if oversample <= 0 or maxfreq <= 0 or not np.isfinite(step) or step <= 0:
            self._result.value = (
                "<b>Spectral-window calculation failed. See Log.</b>"
            )
            self._log(
                "Spectral-window calculation failed: invalid frequency-grid options.",
                level="error",
            )
            return

        total = max(1, int(np.ceil(maxfreq / step)))

        self._maxfreq.disabled = True
        self._oversample.disabled = True
        self._calculate.disabled = True

        self._start_progress(
            total,
            message="Calculating spectral window...",
        )

        self._log(
            "Calculating spectral window: "
            f"maxfreq = {maxfreq} {self.pw.freq_unit}, "
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
            spectral_window,
            self.pw,
            maxfreq=maxfreq,
            oversample=oversample,
            log=False,
            progress_callback=progress_callback,
            cancel_check=self._cancel_check,
            context="Spectral-window calculation failed",
        )

        if self.closed:
            return

        self._maxfreq.disabled = False
        self._oversample.disabled = False
        self._calculate.disabled = False

        if result is None:
            self._result.value = (
                "<b>Spectral-window calculation failed. See Log.</b>"
            )
            self._fail_progress("Spectral window calculation failed.")
            return

        self.frequencies, self.amplitudes = result
        completed = len(self.frequencies)

        if self.cancel_requested:
            self._abort_progress("Spectral window calculation cancelled.")
            self._log(
                "Spectral-window calculation cancelled "
                f"after {completed}/{total} frequency evaluations.",
                level="warning",
            )
        else:
            self._finish_progress()
            self._log(
                "Spectral-window calculation completed "
                f"({completed} frequency evaluations)."
            )

        if completed:
            self._draw_window()
        else:
            self._result.value = (
                "<b>No spectral-window frequencies were completed.</b>"
            )

    def _draw_window(self):
        self.ax.clear()

        self.ax.plot(
            self.frequencies,
            self.amplitudes,
            lw=1,
        )

        self.ax.set_xlabel(
            f"Frequency ({self.pw.freq_unit})"
        )
        self.ax.set_ylabel(
            "Spectral window amplitude"
        )

        self.ax.set_xlim(
            np.min(self.frequencies), np.max(self.frequencies)
        )
        self.ax.set_ylim(
            0,
        )
        
        self.fig.canvas.draw_idle()
        plt.tight_layout()

        self._result.value = (
            f"<b>{len(self.frequencies)}</b> "
            "spectral-window frequencies calculated."
        )

    def _close_clicked(self, _):
        if self._request_close_callback is not None:
            self._request_close_callback(self)
        else:
            self.close()