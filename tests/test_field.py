import numpy as np
import pytest

from resonance.field import fit_profile


def test_profile_fit_recovers_sag_of_a_drawn_parabola():
    h, w = 300, 800
    g = np.full((h, w), 200.0, np.float32)
    x = np.arange(100, 700)
    u = (x - 100) / 600
    y = 80 + 4 * 45 * u * (1 - u)            # 45 px sag between x=100 and x=699
    for xi, yi in zip(x, y):
        r = int(round(yi))
        g[r - 1:r + 2, xi] = 30.0
    prof = fit_profile(g, (100, 60, 699, 150))
    assert prof.sag_px == pytest.approx(45, abs=1.0)
    assert abs(prof.coef[0]) * 600 ** 2 / 4 == pytest.approx(45, abs=1.0)
    assert prof.rms_px < 1.0
