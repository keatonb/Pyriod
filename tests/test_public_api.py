def test_import_prewhitener_from_top_level():
    from Pyriod import Prewhitener
    assert Prewhitener.__name__ == "Prewhitener"


def test_import_prewhitener_from_core():
    from Pyriod.core import Prewhitener
    assert Prewhitener.__name__ == "Prewhitener"


def test_pyriod_wrapper_gui_false_creates_core_object(synthetic_lc):
    from Pyriod import Pyriod
    from Pyriod.core import Prewhitener

    p = Pyriod(
        synthetic_lc,
        gui=False,
        amp_unit="relative",
        freq_unit="1/day",
        minfreq=1,
        maxfreq=10,
        oversample_factor=10,
    )

    assert isinstance(p.pw, Prewhitener)
    assert p._gui is None