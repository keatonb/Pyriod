def test_add_signal_uses_requested_index(synthetic_lc):
    from Pyriod import Prewhitener

    pw = Prewhitener(
        synthetic_lc,
        amp_unit="relative",
        freq_unit="1/day",
        minfreq=1,
        maxfreq=10,
    )

    pw.add_signal(freq=5.0, amp=0.003, phase=0.1, index="f0")

    assert "f0" in pw.stagedvalues.index
    assert pw.stagedvalues.loc["f0", "include"]


def test_duplicate_signal_index_raises(synthetic_lc):
    import pytest
    from Pyriod import Prewhitener

    pw = Prewhitener(
        synthetic_lc,
        amp_unit="relative",
        freq_unit="1/day",
        minfreq=1,
        maxfreq=10,
    )

    pw.add_signal(freq=5.0, amp=0.003, phase=0.1, index="f0")

    with pytest.raises(ValueError):
        pw.add_signal(freq=6.0, amp=0.001, phase=0.2, index="f0")


def test_remove_signal(synthetic_lc):
    from Pyriod import Prewhitener

    pw = Prewhitener(
        synthetic_lc,
        amp_unit="relative",
        freq_unit="1/day",
        minfreq=1,
        maxfreq=10,
    )

    pw.add_signal(freq=5.0, amp=0.003, phase=0.1, index="f0")
    pw.remove_signals("f0")

    assert "f0" not in pw.stagedvalues.index


def test_add_combination_signal(synthetic_lc):
    from Pyriod import Prewhitener

    pw = Prewhitener(
        synthetic_lc,
        amp_unit="relative",
        freq_unit="1/day",
        minfreq=1,
        maxfreq=15,
    )

    pw.add_signal(freq=5.0, amp=0.003, phase=0.1, index="f0")
    pw.add_signal(freq=2.0, amp=0.001, phase=0.2, index="f1")
    pw.add_combination("f0+f1", amp=0.0005, phase=0.3)

    assert "f0+f1" in pw.stagedvalues.index
    assert pw.stagedvalues.loc["f0+f1", "freq"] == 7.0