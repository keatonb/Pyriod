import numpy as np
import astropy.units as u
import lightkurve as lk
import pytest


@pytest.fixture
def synthetic_lc():
    rng = np.random.default_rng(123)

    time = np.linspace(0, 10, 2000)
    freq = 5.0       # 1/day
    amp = 0.003      # relative flux
    phase = 0.17
    noise = 2e-4

    flux = 1.0 + amp * np.sin(2 * np.pi * (freq * time + phase))
    flux += rng.normal(0.0, noise, size=len(time))

    return lk.LightCurve(
        time=time * u.day,
        flux=flux * u.dimensionless_unscaled,
        flux_err=np.full_like(time, noise) * u.dimensionless_unscaled,
    )