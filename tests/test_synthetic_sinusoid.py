import numpy as np

from Pyriod import Pyriod


def test_single_sinusoid_recovery(synthetic_lc):
    p = Pyriod(
        synthetic_lc,
        amp_unit="relative",
        freq_unit="1/day",
        gui=False,
        minfreq=1,
        maxfreq=10,
        oversample_factor=10,
    )

    # Give it a near-correct initial signal.
    p.add_signal(freq=5.0, amp=0.0025, phase=0.1)
    p.fit_model()

    row = p.fitvalues.iloc[0]

    assert np.isclose(row["freq"], 5.0, atol=1e-3)
    assert np.isclose(row["amp"], 0.003, rtol=0.1)

    # Phase is modulo 1, so compare circular distance.
    circular_phase_error = abs(((row["phase"] - 0.17 + 0.5) % 1.0) - 0.5)
    assert circular_phase_error < 0.03