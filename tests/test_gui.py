import pytest
import pandas as pd

def test_gui_qgrid_shows_staged_signal(synthetic_lc):
    """Qgrid should always display the values staged for the next fit."""
    from Pyriod import Prewhitener
    from Pyriod.gui import PyriodGUI

    pw = Prewhitener(synthetic_lc)
    gui = PyriodGUI(pw)

    try:
        # No fitted signals initially.
        assert len(pw.fitvalues) == 0
        assert len(pw.stagedvalues) == 0

        # Simulate entering a new signal in the GUI.
        gui._thisfreq.value = "5.0"
        gui._thisamp.value = 0.003
        gui._add_staged_signal()

        # The signal should be staged, but not fitted yet.
        assert len(pw.stagedvalues) == 1
        assert len(pw.fitvalues) == 0

        # Regression check: Qgrid must show the staged signal immediately.
        displayed = gui._signals_qgrid.df

        assert len(displayed) == 1
        assert displayed.index.equals(pw.stagedvalues.index)
        assert displayed.iloc[0]["freq"] == pytest.approx(5.0)
        assert displayed.iloc[0]["amp"] == pytest.approx(0.003)

        pd.testing.assert_frame_equal(
            gui._signals_qgrid.df,
            pw.staged_table(display_units=True),
        )

    finally:
        gui.close(close_prewhitener=True)