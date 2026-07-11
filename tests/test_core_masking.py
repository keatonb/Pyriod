import numpy as np


def test_mask_indices_excludes_points_and_sets_not_uptodate(synthetic_lc):
    from Pyriod import Prewhitener

    pw = Prewhitener(
        synthetic_lc,
        amp_unit="relative",
        freq_unit="1/day",
        minfreq=1,
        maxfreq=10,
    )

    pw.mask_indices([0, 1, 2])

    assert np.all(pw.lc["include"][:3] == 0)
    assert not pw.uptodate


def test_clear_mask_restores_all_points(synthetic_lc):
    from Pyriod import Prewhitener

    pw = Prewhitener(
        synthetic_lc,
        amp_unit="relative",
        freq_unit="1/day",
        minfreq=1,
        maxfreq=10,
    )

    pw.mask_indices([0, 1, 2])
    pw.clear_mask()

    assert np.all(pw.lc["include"] == 1)


def test_mask_indices_excludes_points_and_sets_not_uptodate(synthetic_lc):
    from Pyriod import Prewhitener

    pw = Prewhitener(
        synthetic_lc,
        amp_unit="relative",
        freq_unit="1/day",
        minfreq=1,
        maxfreq=10,
    )

    pw.mask_indices([0, 1, 2])

    assert np.all(pw.lc["include"][:3] == 0)
    assert pw.uptodate is False