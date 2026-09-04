"""GUI for bootstrap periodogram significance testing."""

from __future__ import annotations

import asyncio

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from traitlets import TraitError

from .base import AnalysisGUI
from ..bootstrap import (
    bootstrap_periodogram_samples,
    bootstrap_threshold_from_samples,
    plot_bootstrap,
)


class BootstrapSignificanceGUI(AnalysisGUI):
    """Interactive GUI for bootstrap periodogram significance testing.

    This interface generates bootstrap periodogram samples from an associated
    `Prewhitener`, displays their histogram and empirical cumulative
    distribution function, and calculates false-alarm probability thresholds
    from the resulting sample distribution.

    Bootstrap samples are retained after calculation so that different
    false-alarm probabilities and relative/absolute threshold definitions can
    be explored without recalculating the periodograms.

    Parameters
    ----------
    pw : Prewhitener
        Analysis object whose light curve and periodogram settings are used
        for the bootstrap calculation.
    log_updated : callable or None, optional
        Callback invoked after this interface writes to the `Prewhitener` log.
        A parent `PyriodGUI` can pass its log-refresh method here.
    busy_changed : callable or None, optional
        Callback invoked with the current busy state whenever a bootstrap
        calculation starts or finishes.
    request_close : callable or None, optional
        Callback used to request removal of this analysis from its parent tab
        manager. It is called with this `BootstrapSignificanceGUI` instance.
        If None, the Close button closes this interface directly.

    Notes
    -----
    Numerical calculations are provided by :mod:`Pyriod.analyses.bootstrap`.
    This class is responsible only for user interaction, display, and
    translating GUI state into arguments for the headless analysis routines.
    """

    title = "Bootstrap"

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

        # Results retained for recalculating thresholds without rerunning
        # the bootstrap periodograms.
        self.maxima = None
        self.medians = None

        # Secondary CDF axis created by plot_bootstrap().
        self._cdf_ax = None

        self._init_figure()
        self._init_widgets()
        self._init_callbacks()
        self._init_layout()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_figure(self):
        """Create and register the bootstrap-distribution figure."""
        self.fig, self.ax = plt.subplots(figsize=(7, 3))
        self._track_figure(self.fig)

        self.fig.canvas.header_visible = False # No title

        self.ax.set_xlabel("bootstrap peak")
        self.ax.set_ylabel("count")
        plt.tight_layout()

    def _init_widgets(self):
        """Create controls and status widgets."""

        # --------------------------------------------------------------
        # Status
        # --------------------------------------------------------------

        self._result = widgets.HTML(
            value="<b>No bootstrap samples calculated.</b>"
        )

        # --------------------------------------------------------------
        # Bootstrap-sampling options
        # --------------------------------------------------------------

        self._nruns = widgets.IntText(
            value=1000,
            description="Runs:",
            style={"description_width": "initial"},
        )

        self._minfreq = widgets.Text(
            value="None",
            description="Min frequency:",
            tooltip="Use None for the full frequency range.",
            style={"description_width": "initial"},
        )

        self._maxfreq = widgets.Text(
            value="None",
            description="Max frequency:",
            tooltip="Use None for the full frequency range.",
            style={"description_width": "initial"},
        )

        self._timelimit = widgets.Text(
            value="None",
            description="Time limit (s):",
            tooltip="Use None for no time limit.",
            style={"description_width": "initial"},
        )

        self._seed = widgets.Text(
            value="None",
            description="Random seed:",
            tooltip="Use None for a non-deterministic random seed.",
            style={"description_width": "initial"},
        )

        self._calculate = widgets.Button(
            description="Calculate samples",
            tooltip="Generate a new set of bootstrap periodogram samples.",
            icon="refresh",
        )

        # --------------------------------------------------------------
        # Significance-threshold options
        # --------------------------------------------------------------

        self._relative = widgets.RadioButtons(
            options=[
                ("Relative to median noise", True),
                ("Absolute amplitude", False),
            ],
            value=True,
            description="Peak height:",
            style={"description_width": "initial"},
        )

        self._fap = widgets.FloatText(
            value=0.01,
            description="FAP:",
            style={"description_width": "initial"},
        )

        self._update = widgets.Button(
            description="Update threshold",
            disabled=True,
            tooltip=(
                "Calculate a new significance threshold from the existing "
                "bootstrap samples."
            ),
            icon="calculator",
        )

        self._info = widgets.HTML(
            value=  ("Bootstrap resampling from <i> current </i> residuals time series. <br>" + 
                     "A reliable threshold requires decent sampling around the FAP level.")
        )

        self._close = widgets.Button(
            description="Close",
            tooltip="Close this analysis tab.",
            icon="times",
        )

    def _init_callbacks(self):
        """Register widget callbacks with automatic cleanup."""
        self._on_click(
            self._calculate,
            self._calculate_clicked,
        )

        self._on_click(
            self._update,
            self._update_clicked,
        )

        self._on_click(
            self._close,
            self._close_clicked,
        ) 

    def _init_layout(self):
        """Construct the top-level analysis widget."""

        sampling_options = widgets.VBox(
            [
                widgets.HTML("<b>Bootstrap samples</b>"),
                self._nruns,
                self._minfreq,
                self._maxfreq,
                self._timelimit,
                self._seed,
                self._calculate,
            ],
            layout=widgets.Layout(
                border="solid 1px",
                padding="6px",
                width="50%",
            ),
        )

        threshold_options = widgets.VBox(
            [
                widgets.HTML("<b>Significance threshold</b>"),
                self._relative,
                self._fap,
                self._update,
                self._info,
            ],
            layout=widgets.Layout(
                border="solid 1px",
                padding="6px",
                width="50%",
            ),
        )

        options = widgets.HBox(
            [
                sampling_options,
                threshold_options,
            ],
            layout=widgets.Layout(width="100%"),
        )

        accordion = widgets.Accordion(
            children=[options],
            selected_index=0,
        )
        accordion.set_title(0, "options")

        progress_row = self.progress_widget

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
                    progress_row,
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

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    @staticmethod
    def _optional_float(widget, name):
        """Return a float from a text widget, allowing ``None``."""
        value = widget.value.strip()

        if value.lower() in ("", "none"):
            return None

        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(
                f"{name} must be a number or None."
            ) from exc

    @staticmethod
    def _optional_int(widget, name):
        """Return an integer from a text widget, allowing ``None``."""
        value = widget.value.strip()

        if value.lower() in ("", "none"):
            return None

        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(
                f"{name} must be an integer or None."
            ) from exc

    def _sampling_options(self):
        """Validate and return the current bootstrap-sampling options."""

        nruns = self._nruns.value

        if nruns < 1:
            raise ValueError(
                "Runs must be greater than or equal to 1."
            )

        minfreq = self._optional_float(
            self._minfreq,
            "Minimum frequency",
        )

        maxfreq = self._optional_float(
            self._maxfreq,
            "Maximum frequency",
        )

        timelimit = self._optional_float(
            self._timelimit,
            "Time limit",
        )

        seed = self._optional_int(
            self._seed,
            "Random seed",
        )

        if (
            minfreq is not None
            and maxfreq is not None
            and minfreq >= maxfreq
        ):
            raise ValueError(
                "Minimum frequency must be smaller than "
                "maximum frequency."
            )

        if timelimit is not None and timelimit <= 0:
            raise ValueError(
                "Time limit must be positive or None."
            )

        return {
            "nruns": nruns,
            "minfreq": minfreq,
            "maxfreq": maxfreq,
            "timelimit": timelimit,
            "seed": seed,
        }

    # ------------------------------------------------------------------
    # Busy / progress state
    # ------------------------------------------------------------------

    def _set_sampling_busy(self, busy):
        """Enable or disable bootstrap-specific controls during calculation."""
        self._nruns.disabled = busy
        self._minfreq.disabled = busy
        self._maxfreq.disabled = busy
        self._timelimit.disabled = busy
        self._seed.disabled = busy
        self._calculate.disabled = busy

        # Avoid changing how old samples are interpreted while a new
        # bootstrap distribution is being generated.
        self._relative.disabled = busy
        self._fap.disabled = busy

        self._update.disabled = busy or self.maxima is None

    # ------------------------------------------------------------------
    # Bootstrap calculation
    # ------------------------------------------------------------------

    def _calculate_clicked(self, _):
        if self.busy:
            return

        try:
            self._start_task(
                self._calculate_samples()
            )
        except Exception as exc:
            self._log_exception(
                exc,
                context="Could not start bootstrap calculation",
            )

    async def _calculate_samples(self):
        """Generate bootstrap samples without blocking the notebook GUI."""

        options = self._run_analysis(
            self._sampling_options,
            context="Invalid bootstrap significance options",
        )

        if options is None:
            self._result.value = (
                "<b>Invalid bootstrap options. See Log.</b>"
            )
            return

        nruns = options["nruns"]
        minfreq = options["minfreq"]
        maxfreq = options["maxfreq"]
        timelimit = options["timelimit"]
        seed = options["seed"]

        self._set_sampling_busy(True)
        self._start_progress(
            nruns,
            message="Calculating bootstrap samples...",
        )

        self._log(
            "Calculating bootstrap periodogram samples: "
            f"nruns = {nruns}, "
            f"minfreq = {minfreq}, "
            f"maxfreq = {maxfreq} {self.pw.freq_unit}, "
            f"timelimit = {timelimit}, "
            f"seed = {seed}."
        )

        rng = np.random.default_rng(seed)

        loop = asyncio.get_running_loop()

        def progress_callback(completed, requested):
            # This callback is executed by the worker thread. Schedule
            # widget updates back onto the notebook event loop.
            loop.call_soon_threadsafe(
                self._set_progress,
                completed,
                requested,
            )

        result = await self._run_analysis_in_thread(
            bootstrap_periodogram_samples,
            self.pw,
            minfreq=minfreq,
            maxfreq=maxfreq,
            nruns=nruns,
            statusbar=False,
            timelimit=timelimit,
            rng=rng,
            progress_callback=progress_callback,
            cancel_check=self._cancel_check,
            context="Bootstrap significance calculation failed",
        )

        if self.closed:
            return

        if result is None:
            self._result.value = (
                "<b>Bootstrap calculation failed. See Log.</b>"
            )
            self._fail_progress("Bootstrap calculation failed.")
            self._set_sampling_busy(False)
            return

        maxima, medians = result

        self.maxima = maxima
        self.medians = medians

        completed = len(maxima)

        if self.cancel_requested:
            self._log(
                "Bootstrap significance calculation cancelled "
                f"after {completed}/{nruns} realizations.",
                level="warning",
            )
            self._abort_progress("Bootstrap calculation cancelled.")

        elif completed < nruns:
            self._log(
                "Bootstrap significance calculation stopped "
                f"after {completed}/{nruns} realizations "
                "because of the time-limit criterion.",
                level="warning",
            )
            self._abort_progress("Bootstrap calculation stopped early.")

        else:
            self._log(
                "Bootstrap periodogram sampling completed "
                f"({completed} realizations)."
            )
            self._finish_progress()

        self._set_sampling_busy(False)

        if completed:
            self._update_threshold()
        else:
            self._result.value = (
                "<b>No bootstrap samples were completed.</b>"
            )

    # ------------------------------------------------------------------
    # Threshold calculation
    # ------------------------------------------------------------------

    def _update_clicked(self, _):
        """Calculate a new threshold from the existing samples."""
        self._update_threshold()

    def _update_threshold(self):
        """Recalculate and redraw the threshold without resampling."""

        if self.maxima is None or len(self.maxima) == 0:
            return

        result = self._run_analysis(
            bootstrap_threshold_from_samples,
            self.maxima,
            self.medians,
            fap=self._fap.value,
            relative=self._relative.value,
            context="Bootstrap threshold calculation failed",
        )

        if result is None:
            self._result.value = (
                "<b>Threshold calculation failed. See Log.</b>"
            )
            return

        threshold, samples = result

        self._draw_samples(
            samples,
            threshold,
        )

        relative = self._relative.value
        fap = self._fap.value

        if relative:
            threshold_string = (
                f"{threshold:.6g} × median periodogram amplitude"
            )
        else:
            amp_unit = getattr(
                self.pw,
                "amp_unit",
                "",
            )

            if amp_unit:
                threshold_string = (
                    f"{threshold:.6g} {amp_unit}"
                )
            else:
                threshold_string = f"{threshold:.6g}"

        self._result.value = (
            f"<b>FAP:</b> {fap:.6g}"
            "&nbsp;&nbsp;&nbsp;"
            f"<b>Threshold:</b> {threshold_string}"
            "&nbsp;&nbsp;&nbsp;"
            f"<b>Samples:</b> {len(self.maxima)}"
        )

        self._log(
            "Bootstrap significance threshold calculated from "
            f"{len(self.maxima)} existing samples: "
            f"FAP = {fap:.6g}, "
            f"threshold = {threshold:.6g}, "
            f"relative = {relative}."
        )

    def _draw_samples(self, samples, threshold):
        """Redraw the bootstrap histogram and cumulative distribution."""

        if self._cdf_ax is not None:
            try:
                self._cdf_ax.remove()
            except Exception:
                pass

            self._cdf_ax = None

        self.ax.clear()

        result = self._run_analysis(
            plot_bootstrap,
            samples,
            fap=self._fap.value,
            ax=self.ax,
            context="Bootstrap significance plot failed",
        )

        if result is None:
            return

        _, self._cdf_ax = result

        if self._relative.value:
            self.ax.set_xlabel(
                "maximum peak / median periodogram amplitude"
            )
        else:
            amp_unit = getattr(
                self.pw,
                "amp_unit",
                None,
            )

            if amp_unit is None:
                self.ax.set_xlabel(
                    "maximum peak amplitude"
                )
            else:
                self.ax.set_xlabel(
                    f"maximum peak amplitude ({amp_unit})"
                )

        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Tab lifecycle
    # ------------------------------------------------------------------

    def _close_clicked(self, _):
        """Request removal of this analysis tab."""
        if self._request_close_callback is not None:
            self._request_close_callback(self)
        else:
            self.close()

    def refresh(self):
        """Synchronize inexpensive display state with the `Prewhitener`.

        Existing bootstrap samples are intentionally retained. Changes to the
        underlying light curve or periodogram settings therefore do not
        automatically trigger an expensive recalculation.
        """
        return None