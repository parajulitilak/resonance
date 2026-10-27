import json
import math

import numpy as np
import pytest

from resonance import physics
from resonance.displacement import _hann2d, phase_shift
from resonance.spectrum import Peak, fit_harmonics


def ladder_peaks(f0, b=0.0, ns=(1, 2, 3, 4, 5), snr=40.0):
    return [Peak(n * f0 * math.sqrt(1 + b * n * n), 1.0 / n ** 2, snr) for n in ns]


def test_harmonic_fit_recovers_fundamental_and_stiffness():
    fit = fit_harmonics(ladder_peaks(12.5, b=3e-4, ns=(1, 2, 3, 4, 5, 6)), fs=240)
    assert fit.ok, fit.reasons
    assert fit.f0 == pytest.approx(12.5, rel=1e-4)
    assert fit.b == pytest.approx(3e-4, rel=0.05)


def test_harmonic_fit_prefers_true_fundamental_over_subharmonic():
    fit = fit_harmonics(ladder_peaks(10.0, ns=(1, 3, 5, 7)), fs=240)  # pluck at mid-span: odd modes only
    assert fit.f0 == pytest.approx(10.0, rel=1e-3)


def test_aliased_peaks_are_refused():
    fs, f0 = 30.0, 13.0
    folded = []
    for n in range(1, 6):
        f = (n * f0) % fs
        folded.append(Peak(min(f, fs - f), 1.0 / n ** 2, 30.0))
    fit = fit_harmonics(folded, fs=fs)
    assert not fit.ok


def test_too_few_harmonics_refused():
    fit = fit_harmonics(ladder_peaks(9.0, ns=(1, 2)), fs=240)
    assert not fit.ok


def test_phase_shift_subpixel():
    rng = np.random.default_rng(0)
    from scipy.ndimage import fourier_shift, gaussian_filter
    img = gaussian_filter(rng.normal(0, 1, (96, 96)), 2)
    win = _hann2d(96, 96)
    ref = np.fft.fft2((img - img.mean()) * win)
    moved = np.fft.ifft2(fourier_shift(np.fft.fft2(img), (0.4, -0.7))).real  # exact sub-pixel shift
    dx, dy = phase_shift(ref, moved, win)
    assert dx == pytest.approx(-0.7, abs=0.06)
    assert dy == pytest.approx(0.4, abs=0.06)


def test_tension_formula_and_uncertainty():
    m, L, T = 0.022, 1.75, 50.0
    f0 = math.sqrt(T / m) / (2 * L)
    est = physics.tension(f0, 0.0, 0.0, L, m)
    assert est.tension_n == pytest.approx(T, rel=1e-9)
    assert 0.0 < est.rel_sigma < 0.05


def test_stiff_member_flagged_unreliable():
    est = physics.tension(20.0, 0.01, b=0.2, length_m=1.5, mass_per_m=1.0)
    assert not est.reliable


def test_sag_frequency():
    assert physics.sag_frequency(2.0) == pytest.approx(math.sqrt(physics.G / 16), rel=1e-12)


@pytest.mark.slow
def test_bench_end_to_end(tmp_path):
    from resonance import synth
    from resonance.analyze import analyze, compare_with_truth, members_from_truth
    ropes = synth.default_rig()[:3]
    vid = tmp_path / "bench.mp4"
    truth = synth.render(vid, fps=240, duration=3.0, ropes=ropes, shake_px=1.5, seed=3)
    rep = analyze(str(vid), members_from_truth(truth), anchor=tuple(truth["anchor_box"]))
    rows = compare_with_truth(rep, truth)
    assert all(r["ladder_ok"] for r in rows)
    assert max(abs(r["err"]) for r in rows) < 0.02
    json.dumps(rep, default=float)


def test_two_clean_harmonics_are_provisional_not_verified():
    fit = fit_harmonics(ladder_peaks(8.4, ns=(1, 2)), fs=240)
    assert not fit.ok and fit.provisional and fit.status == "provisional"
