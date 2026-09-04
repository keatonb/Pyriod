"""GUI for spectral-window analysis."""

from __future__ import annotations

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
        self.ax.set_ylabel("Spectral window amplitude")

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
        result = self._run_analysis(
            spectral_window,
            self.pw,
            maxfreq=self._maxfreq.value,
            oversample=self._oversample.value,
            log=True,
            context="Spectral-window calculation failed",
        )

        if result is None:
            self._result.value = (
                "<b>Spectral-window calculation failed. "
                "See Log.</b>"
            )
            return

        self.frequencies, self.amplitudes = result

        self._draw_window()

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