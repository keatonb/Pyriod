import numpy as np
import pandas as pd
import pytest


@pytest.mark.gui
def test_gui_reflects_already_analyzed_prewhitener(synthetic_lc):
    """PyriodGUI should display the current state of an analyzed Prewhitener.

    This tests the intended refactor workflow:

        pw = Prewhitener(lc)
        pw.add_signal(...)
        pw.fit_model()
        gui = PyriodGUI(pw)

    After connection, the GUI should already reflect the Prewhitener's
    current time-series data, periodograms, signal table, and log.
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

    from Pyriod import Prewhitener, PyriodGUI
    from Pyriod.plotsupport import decimate_visible_range

    pw = Prewhitener(
        synthetic_lc,
        amp_unit="relative",
        freq_unit="1/day",
        minfreq=1,
        maxfreq=10,
        oversample_factor=10,
    )

    pw.add_signal(freq=5.0, amp=0.0025, phase=0.1, index="f0")
    pw.fit_model()

    # Add a recognizable message after fitting so we can verify the GUI log
    # reflects the already-existing Prewhitener log at connection time.
    pw.log("GUI connection smoke-test sentinel message")

    gui = PyriodGUI(pw)

    try:
        assert gui.pw is pw
        assert pw.fit_result is not None
        assert "f0" in pw.fitvalues.index

        # ---------------------------------------------------------------
        # Time-series plot
        # ---------------------------------------------------------------
        offsets = gui._lcplot_data.get_offsets()

        np.testing.assert_allclose(
            offsets[:, 0],
            pw.lc.time.value,
        )

        np.testing.assert_allclose(
            offsets[:, 1],
            np.asarray(pw.lc.flux.value),
        )

        # Force the debounced model-line refresh so the test does not depend
        # on the notebook event loop/timer running during pytest.
        gui._refresh_model_line_from_view()

        model_x, model_y = gui._lcplot_model.get_data()

        assert len(model_x) == len(model_y)
        assert len(model_x) > 0
        assert np.all(np.isfinite(model_x))
        assert np.all(np.isfinite(model_y))

        x0, x1 = gui.lcax.get_xlim()
        xmin, xmax = sorted((x0, x1))

        assert np.nanmin(model_x) >= xmin
        assert np.nanmax(model_x) <= xmax

        # ---------------------------------------------------------------
        # Periodogram plot
        # ---------------------------------------------------------------
        # Force refresh for the same reason: do not rely on the GUI timer.
        gui._refresh_periodogram_lines_from_view()

        per_x0, per_x1 = gui.perax.get_xlim()
        per_xmin, per_xmax = sorted((per_x0, per_x1))

        expected_resid_x, expected_resid_y = decimate_visible_range(
            pw.freqs,
            pw.per_resid,
            per_xmin,
            per_xmax,
            max_points=gui.max_periodogram_plot_points,
        )

        resid_x, resid_y = gui._perplot_resid.get_data()

        np.testing.assert_allclose(resid_x, expected_resid_x)
        np.testing.assert_allclose(resid_y, expected_resid_y)

        expected_model_x, expected_model_y = decimate_visible_range(
            pw.freqs,
            pw.per_model,
            per_xmin,
            per_xmax,
            max_points=gui.max_periodogram_plot_points,
        )

        model_per_x, model_per_y = gui._perplot_model.get_data()

        np.testing.assert_allclose(model_per_x, expected_model_x)
        np.testing.assert_allclose(model_per_y, expected_model_y)

        # Signal marker should reflect the fitted/staged signal.
        marker_x, marker_y = gui._signal_markers.get_data()

        assert len(marker_x) == 1
        assert len(marker_y) == 1

        np.testing.assert_allclose(
            marker_x[0],
            pw.stagedvalues.loc["f0", "freq"],
        )

        np.testing.assert_allclose(
            marker_y[0],
            pw.stagedvalues.loc["f0", "amp"] * pw.amp_conversion,
        )

        # ---------------------------------------------------------------
        # Signals table
        # ---------------------------------------------------------------
        qgrid_df = gui._signals_qgrid.get_changed_df()

        expected_table = pw.solution_table(
            display_units=True,
            include_brute=True,
        )

        pd.testing.assert_index_equal(qgrid_df.index, expected_table.index)

        assert "f0" in qgrid_df.index

        for column in expected_table.columns:
            assert column in qgrid_df.columns

        pd.testing.assert_series_equal(
            qgrid_df.loc["f0", expected_table.columns],
            expected_table.loc["f0"],
            check_names=False,
            check_dtype=False,
        )

        # ---------------------------------------------------------------
        # Log
        # ---------------------------------------------------------------
        assert "GUI connection smoke-test sentinel message" in gui._log.value

    finally:
        gui.close()
        plt.close("all")