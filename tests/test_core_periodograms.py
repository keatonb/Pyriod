import numpy as np


def test_initial_periodograms_exist(synthetic_lc):
    from Pyriod import Prewhitener

    pw = Prewhitener(
        synthetic_lc,
        amp_unit="relative",
        freq_unit="1/day",
        minfreq=1,
        maxfreq=10,
        oversample_factor=10,
    )

    assert pw.freqs.size > 0
    assert pw.per_orig.shape == pw.freqs.shape
    assert pw.per_model.shape == pw.freqs.shape
    assert pw.per_resid.shape == pw.freqs.shape
    assert np.all(np.isfinite(pw.per_orig))


def test_sample_model_matches_input_length_after_fit(synthetic_lc):
    from Pyriod import Prewhitener

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

    model_flux = pw.sample_model(pw.lc.time.value)

    assert len(model_flux) == len(pw.lc)
    assert np.all(np.isfinite(model_flux))