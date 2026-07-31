import pytest

def test_prewhitener_close_twice(synthetic_lc):
    from Pyriod import Prewhitener

    pw = Prewhitener(
        synthetic_lc,
        amp_unit="relative",
        freq_unit="1/day",
        minfreq=1,
        maxfreq=10,
    )

    pw.close(clear_data=True)
    pw.close(clear_data=True)

@pytest.mark.gui
def test_gui_close_twice(synthetic_lc):
    pytest.importorskip("ipywidgets")
    pytest.importorskip("qgridnext")
    pytest.importorskip("ipyfilechooser")
    pytest.importorskip("ipympl")

    from Pyriod import Prewhitener, PyriodGUI

    pw = Prewhitener(
        synthetic_lc,
        amp_unit="relative",
        freq_unit="1/day",
        minfreq=1,
        maxfreq=10,
    )

    gui = PyriodGUI(pw)
    gui.close()
    gui.close()