import numpy as np

from Pyriod import Pyriod


def test_delete_rows_empty_selection_does_not_crash(synthetic_lc):
    p = Pyriod(synthetic_lc, amp_unit="relative", freq_unit="1/day", gui=False)
    p.delete_rows([])
    assert len(p.stagedvalues) == 0


def test_delete_rows_single_label(synthetic_lc):
    p = Pyriod(synthetic_lc, amp_unit="relative", freq_unit="1/day", gui=False)
    p.add_signal(freq=5.0, amp=0.003)
    p.delete_rows("f0")
    assert "f0" not in p.stagedvalues.index


def test_fit_uncertainties_may_be_nan_not_crash(synthetic_lc, monkeypatch):
    p = Pyriod(synthetic_lc, amp_unit="relative", freq_unit="1/day", gui=False)
    p.add_signal(freq=5.0, amp=0.003)
    p.fit_model()

    assert "freqerr" in p.fitvalues.columns
    assert np.all(np.isfinite(p.fitvalues["freq"]) | np.isnan(p.fitvalues["freq"]))