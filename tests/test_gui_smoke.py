# tests/test_gui_smoke.py

import numpy as np
import pytest


def test_gui_initializes_adds_fits_and_removes_one_signal(synthetic_lc):
    """
    Smoke test for the gui=True path.

    This checks that the widget/qgrid/matplotlib-backed object initializes,
    can add one signal, fit it, and remove it again.
    """

    pytest.importorskip("ipywidgets")
    pytest.importorskip("qgridnext")
    pytest.importorskip("ipyfilechooser")
    pytest.importorskip("ipympl")

    import matplotlib

    try:
        matplotlib.use("module://ipympl.backend_nbagg", force=True)
    except Exception as exc:
        pytest.skip(f"Could not activate ipympl backend: {exc}")

    import matplotlib.pyplot as plt
    from Pyriod import Pyriod

    p = Pyriod(
        synthetic_lc,
        amp_unit="relative",
        freq_unit="1/day",
        gui=True,
        minfreq=1,
        maxfreq=10,
        oversample_factor=10,
    )

    assert p.gui is True
    assert hasattr(p, "lcfig")
    assert hasattr(p, "perfig")
    assert hasattr(p, "_signals_qgrid")

    p.add_signal(freq=5.0, amp=0.0025, phase=0.1, index="f0")

    assert "f0" in p.stagedvalues.index
    assert len(p.stagedvalues) == 1

    qgrid_df = p._signals_qgrid.get_changed_df()
    assert "f0" in qgrid_df.index

    p.fit_model()

    assert p.fit_result is not None
    assert "f0" in p.fitvalues.index
    assert np.isfinite(p.fitvalues.loc["f0", "freq"])
    assert np.isfinite(p.fitvalues.loc["f0", "amp"])
    assert np.isfinite(p.fitvalues.loc["f0", "phase"])

    p.delete_rows(["f0"])

    assert "f0" not in p.stagedvalues.index

    qgrid_df = p._signals_qgrid.get_changed_df()
    assert "f0" not in qgrid_df.index

    plt.close("all")

# test with gui=False
def test_core_gui_false_add_fit_delete(synthetic_lc):
    from Pyriod import Pyriod

    p = Pyriod(
        synthetic_lc,
        amp_unit="relative",
        freq_unit="1/day",
        gui=False,
        minfreq=1,
        maxfreq=10,
        oversample_factor=10,
    )

    p.add_signal(freq=5.0, amp=0.0025, phase=0.1, index="f0")
    p.fit_model()

    assert p.fit_result is not None
    assert "f0" in p.fitvalues.index
    assert np.isfinite(p.fitvalues.loc["f0", "freq"])

    p.delete_rows(["f0"])
    assert "f0" not in p.stagedvalues.index

def test_core_import_does_not_require_gui_stack():
    import Pyriod.core
