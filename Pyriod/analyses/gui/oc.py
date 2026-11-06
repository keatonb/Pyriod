"""GUI for sliding-window O-C analysis."""

from __future__ import annotations

import asyncio

import ipywidgets as widgets
import matplotlib.pyplot as plt
from traitlets import TraitError

from .base import AnalysisGUI
from ..oc import oc_analysis


class OCAnalysisGUI(AnalysisGUI):
    """Interactive O-C analysis using sliding-window Prewhitener fits."""

    title = "O-C"

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

        self._init_figure()
        self._init_widgets()
        self._init_callbacks()
        self._init_layout()

    def _init_figure(self):
        self.fig, (self.ax, self.amp_ax) = plt.subplots(
            2,
            1,
            figsize=(7, 5),
            sharex=True,
        )
        self._track_figure(self.fig)
        self.fig.canvas.header_visible = False

        self.ax.set_ylabel("O-C (s)")

        amp_unit = getattr(self.pw, "amp_unit", "")
        if amp_unit:
            self.amp_ax.set_ylabel(f"amplitude ({amp_unit})")
        else:
            self.amp_ax.set_ylabel("amplitude")

        self.amp_ax.set_xlabel(
            f"time ({self.pw.time_unit.to_string()})"
        )

        plt.tight_layout()

    def _init_widgets(self):
        signal_options = self._signal_options()

        self._frequency = widgets.FloatText(
            value=(
                signal_options[0][1]
                if signal_options
                else 1.0
            ),
            description="Frequency:",
            style={"description_width": "initial"},
        )

        self._signal = widgets.Dropdown(
            options=signal_options if signal_options else [None],
            description="Select from:",
            style={"description_width": "initial"},
        )

        duration = float(
            self.pw.lc.time.value.max() - self.pw.lc.time.value.min()
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

        self._min_points = widgets.IntText(
            value=3,
            description="Minimum points:",
            style={"description_width": "initial"},
        )

        self._fit_amplitude = widgets.Checkbox(
            value=False,
            description="Fit amplitude in each window",
            indent=False,
        )

        self._calculate = widgets.Button(
            description="Calculate O-C",
            icon="refresh",
        )

        self._result = widgets.HTML(
            value="<b>No O-C analysis calculated.</b>"
        )

        self._close = widgets.Button(
            description="Close",
            icon="times",
        )

    def _init_callbacks(self):
        self._observe(
            self._signal,
            self._signal_selected,
            names="value",
        )
        self._on_click(self._calculate, self._calculate_clicked)
        self._on_click(self._close, self._close_clicked)

    def _init_layout(self):
        options = widgets.VBox(
            [
                self._frequency,
                self._signal,
                self._window_width,
                self._step_size,
                self._min_points,
                self._fit_amplitude,
                self._calculate,
            ],
            layout=widgets.Layout(
                border="solid 1px",
                padding="6px",
            ),
        )

        accordion = widgets.Accordion(
            children=[options],
            selected_index=0,
        )
        accordion.set_title(0, "options")

        close_row = widgets.HBox(
            [self._close],
            layout=widgets.Layout(justify_content="flex-end"),
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
        self._frequency.disabled = busy
        self._signal.disabled = busy
        self._window_width.disabled = busy
        self._step_size.disabled = busy
        self._min_points.disabled = busy
        self._fit_amplitude.disabled = busy
        self._calculate.disabled = busy

    def _calculate_clicked(self, _):
        self._refresh_signal_options()

        if self.busy:
            return

        try:
            self._start_task(self._calculate_oc())
        except Exception as exc:
            self._log_exception(
                exc,
                context="Could not start O-C calculation",
            )

    async def _calculate_oc(self):
        frequency = self._frequency.value

        if frequency <= 0:
            self._result.value = (
                "<b>Frequency must be positive.</b>"
            )
            return

        window_width = self._window_width.value
        step_size = self._step_size.value
        min_points = self._min_points.value
        fit_amplitude = self._fit_amplitude.value

        if window_width <= 0 or step_size <= 0:
            self._result.value = (
                "<b>Window width and step size must be positive.</b>"
            )
            return

        if min_points < 3:
            self._result.value = (
                "<b>Minimum points must be at least 3.</b>"
            )
            return

        time = self.pw.lc.time.value
        duration = float(time.max() - time.min())
        total = max(1, int(duration // step_size) + 1)

        self._set_analysis_busy(True)
        self._start_progress(
            total,
            message="CALCULATING O-C...",
        )

        self._log(
            f"Calculating O-C at {frequency:.8f} {self.pw.freq_unit}: "
            f"window width = {window_width} d, "
            f"step size = {step_size} d, "
            f"fit amplitude = {fit_amplitude}."
        )

        loop = asyncio.get_running_loop()

        def progress_callback(completed, requested):
            loop.call_soon_threadsafe(
                self._set_progress,
                completed,
                requested,
            )

        result = await self._run_analysis_in_thread(
            oc_analysis,
            self.pw,
            frequency,
            window_width,
            step_size,
            fit_amplitude=fit_amplitude,
            min_points=min_points,
            progress_callback=progress_callback,
            cancel_check=self._cancel_check,
            context="O-C analysis failed",
        )

        if self.closed:
            return

        self._set_analysis_busy(False)

        if result is None:
            self._result.value = "<b>O-C analysis failed. See Log.</b>"
            self._fail_progress("O-C ANALYSIS FAILED")
            return

        self.results = result

        if self.cancel_requested:
            self._abort_progress("O-C ANALYSIS CANCELLED")
            self._log(
                f"O-C analysis cancelled after {len(result)} completed windows.",
                level="warning",
            )
        else:
            self._finish_progress()
            self._log(
                f"O-C analysis completed with {len(result)} fitted windows."
            )

        self._draw_results()

    def _draw_results(self):
        self.ax.clear()
        self.amp_ax.clear()

        if self.results is None or self.results.empty:
            self._result.value = "<b>No O-C measurements were completed.</b>"
            self.fig.canvas.draw_idle()
            return

        self.ax.errorbar(
            self.results["time"],
            self.results["oc_seconds"],
            yerr=self.results["ocerr_seconds"],
            fmt="o",
            ms=4,
            capsize=2,
        )
        self.ax.axhline(0, lw=1)
        self.ax.set_ylabel("O-C (s)")

        amp_unit = getattr(self.pw, "amp_unit", "")
        if amp_unit:
            self.amp_ax.set_ylabel(f"amplitude ({amp_unit})")
        else:
            self.amp_ax.set_ylabel("amplitude")

        if self._fit_amplitude.value:
            self.amp_ax.errorbar(
                self.results["time"],
                self.results["amp"],
                yerr=self.results["amperr"],
                fmt="o",
                ms=4,
                capsize=2,
            )
            amp_text = "amplitude fitted"
        else:
            reference_amplitude = self.results.attrs.get(
                "reference_amplitude"
            )

            if reference_amplitude is not None:
                self.amp_ax.axhline(
                    reference_amplitude,
                    lw=1,
                )
                amp_text = (
                    "amplitude fixed at "
                    f"{reference_amplitude:.6g}"
                )
                if amp_unit:
                    amp_text += f" {amp_unit}"
            else:
                amp_text = "amplitude fixed"

        self.amp_ax.set_xlabel(
            f"time ({self.pw.time_unit.to_string()})"
        )

        self.fig.canvas.draw_idle()
        plt.tight_layout()

        self._result.value = (
            f"<b>{len(self.results)}</b> windows fitted; {amp_text}."
        )

    def _signal_options(self):
        """Return fitted signals formatted like the phase-folding selector."""
        return [
            (
                self.pw.fitvalues.index[i]
                + ": {:.8f} ".format(self.pw.fitvalues.freq.iloc[i])
                + self.pw.freq_unit.to_string(),
                float(self.pw.fitvalues.freq.iloc[i]),
            )
            for i in range(len(self.pw.fitvalues))
        ]

    def _signal_selected(self, change):
        """Copy a selected fitted-signal frequency into the frequency field."""
        frequency = change["new"]
        if frequency is not None:
            self._frequency.value = frequency

    def _refresh_signal_options(self):
        """Refresh the fitted-signal dropdown without changing manual frequency."""
        current = self._signal.value
        options = self._signal_options()

        if not options:
            self._signal.options = [None]
            return

        values = [value for _, value in options]

        self._signal.options = options

        if current in values:
            self._signal.value = current

    def _close_clicked(self, _):
        if self._request_close_callback is not None:
            self._request_close_callback(self)
        else:
            self.close()

    def refresh(self):
        """Refresh the available fitted-signal choices."""
        self._refresh_signal_options()
