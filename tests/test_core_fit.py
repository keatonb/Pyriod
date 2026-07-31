import numpy as np


def test_single_sinusoid_recovery(synthetic_lc):
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

    row = pw.fitvalues.loc["f0"]

    assert np.isclose(row["freq"], 5.0, atol=1e-3)
    assert np.isclose(row["amp"], 0.003, rtol=0.1)

    circular_phase_error = abs(((row["phase"] - 0.17 + 0.5) % 1.0) - 0.5)
    assert circular_phase_error < 0.03

def test_fixed_frequency_stays_fixed(synthetic_lc):
    from Pyriod import Prewhitener

    pw = Prewhitener(
        synthetic_lc,
        amp_unit="relative",
        freq_unit="1/day",
        minfreq=1,
        maxfreq=10,
        oversample_factor=10,
    )

    pw.add_signal(freq=4.9, amp=0.0025, phase=0.1, fixfreq=True, index="f0")
    pw.fit_model()

    assert pw.fitvalues.loc["f0", "freq"] == 4.9
    assert np.isfinite(pw.fitvalues.loc["f0", "amp"])
    assert np.isfinite(pw.fitvalues.loc["f0", "phase"])