"""Spectra, peaks and the harmonic-ladder verifier.

A taut cable rings at f_n = n f0 sqrt(1 + B n^2): whole-number multiples of a
fundamental, stretched slightly by bending stiffness (B >= 0). That ladder is a
physical self-check: if the strong peaks don't sit on one ladder, the take is
aliased, the wrong object was tracked, or the member isn't behaving like a
cable, and Resonance refuses to report a tension.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import median_filter
from scipy.optimize import least_squares
from scipy.signal import detrend, find_peaks


@dataclass
class Spectrum:
    f: np.ndarray
    power: np.ndarray       # summed over stations
    floor: np.ndarray       # running-median noise floor
    fs: float               # uniform sample rate used
    station_fft: np.ndarray  # [freq, station] complex, for mode shapes


@dataclass
class Peak:
    f: float
    power: float
    snr: float


@dataclass
class HarmonicFit:
    f0: float                       # stiffness-free fundamental, Hz
    b: float                        # inharmonicity coefficient
    modes: list[tuple[int, float, float]] = field(default_factory=list)  # (n, f_obs, snr)
    rel_residual: float = float("nan")
    explained: float = 0.0          # share of strong-peak power on the ladder
    ok: bool = False
    provisional: bool = False       # only the 1st and 2nd harmonics, at an exact 2:1 ratio
    reasons: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        return "verified" if self.ok else ("provisional" if self.provisional else "refused")

    @property
    def f0_sigma(self) -> float:
        n = max(len(self.modes), 1)
        return self.f0 * max(self.rel_residual, 1e-4) / np.sqrt(n)


def uniform(t: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float]:
    """Resample [frames, stations] onto a uniform grid at the median frame interval."""
    dt = float(np.median(np.diff(t)))
    tu = np.arange(t[0], t[-1] + 1e-12, dt)
    xu = np.column_stack([np.interp(tu, t, x[:, k]) for k in range(x.shape[1])])
    return xu, 1.0 / dt


def spectrum(t: np.ndarray, disp: np.ndarray, pad: int = 4, t0: float | None = None) -> Spectrum:
    """Power spectrum summed over stations (window from `t0`, e.g. after the pluck)."""
    if t0 is not None:
        keep = t >= t0
        t, disp = t[keep], disp[keep]
    x, fs = uniform(t, disp)
    x = detrend(x, axis=0)
    win = np.hanning(x.shape[0])[:, None]
    nfft = int(2 ** np.ceil(np.log2(x.shape[0])) * pad)
    sf = np.fft.rfft(x * win, n=nfft, axis=0)
    p = (np.abs(sf) ** 2).sum(1)
    f = np.fft.rfftfreq(nfft, 1 / fs)
    # the floor window must be several Hann main lobes wide (4 * pad bins each) or it rides up the peaks
    floor = median_filter(p, size=max(8 * 4 * pad + 1, nfft // 32), mode="nearest") + 1e-12
    return Spectrum(f, p, floor, fs, sf)


def peaks(sp: Spectrum, f_min: float = 0.3, min_snr: float = 8.0, max_peaks: int = 24) -> list[Peak]:
    lp = 10 * np.log10(sp.power + 1e-20)
    idx, _ = find_peaks(lp, prominence=6.0)
    out = []
    for i in idx:
        if sp.f[i] < f_min or i <= 0 or i >= len(lp) - 1:
            continue
        snr = sp.power[i] / sp.floor[i]
        if snr < min_snr:
            continue
        a, b, c = lp[i - 1], lp[i], lp[i + 1]
        den = a - 2 * b + c
        d = 0.0 if abs(den) < 1e-12 else 0.5 * (a - c) / den
        df = sp.f[1] - sp.f[0]
        out.append(Peak(float(sp.f[i] + d * df), float(sp.power[i]), float(snr)))
    out.sort(key=lambda p: -p.power)
    return sorted(out[:max_peaks], key=lambda p: p.f)


def _ladder(n: np.ndarray, f0: float, b: float) -> np.ndarray:
    return n * f0 * np.sqrt(1 + b * n ** 2)


def fit_harmonics(pk: list[Peak], fs: float, tol: float = 0.025, max_n: int = 16) -> HarmonicFit:
    """Find the fundamental whose ladder explains the strong peaks; verify it."""
    nyq = fs / 2
    pk = [p for p in pk if p.f < 0.92 * nyq]  # peaks at the Nyquist edge can't be told from folded ones
    if not pk:
        return HarmonicFit(float("nan"), 0.0, ok=False, reasons=["no peaks above the noise floor"])
    total = sum(p.power for p in pk)
    best: HarmonicFit | None = None
    best_score = -1.0
    cands = {p.f / k for p in pk for k in (1, 2, 3)}
    for f1 in sorted(cands):
        if f1 <= 0:
            continue
        used: dict[int, Peak] = {}
        for p in pk:
            n = int(round(p.f / f1))
            if 1 <= n <= max_n and abs(p.f - n * f1) <= tol * n * f1:
                if n not in used or used[n].power < p.power:
                    used[n] = p
        if len(used) < 2:
            continue
        ns = np.array(sorted(used))
        fo = np.array([used[k].f for k in ns])
        w = np.sqrt(np.array([used[k].snr for k in ns]))
        res = least_squares(lambda v: (_ladder(ns, v[0], v[1]) - fo) / fo * w,
                            x0=[f1, 0.0], bounds=([0.5 * f1, 0.0], [1.5 * f1, 0.05]))
        f0, b = res.x
        rel = float(np.sqrt(np.mean(((_ladder(ns, f0, b) - fo) / fo) ** 2)))
        explained = sum(used[k].power for k in ns) / total
        coverage = len(ns) / ns.max()  # a sub-harmonic "fundamental" leaves every other rung empty
        score = len(ns) * explained * coverage / (1 + rel / 0.002)
        if score > best_score:
            best_score = score
            best = HarmonicFit(float(f0), float(b), [(int(k), float(used[k].f), float(used[k].snr)) for k in ns],
                               rel, float(explained))
    if best is None:
        return HarmonicFit(float("nan"), 0.0, ok=False,
                           reasons=["strong peaks do not share a harmonic ladder (aliasing or wrong object?)"])
    ns = [m[0] for m in best.modes]
    reasons = []
    if len(ns) < 3:
        reasons.append(f"only {len(ns)} harmonics on the ladder (need 3)")
    if min(ns) > 2:
        reasons.append("fundamental and 2nd harmonic both missing")
    if best.rel_residual > 0.006:
        reasons.append(f"ladder residual {best.rel_residual:.2%} is too large")
    if best.explained < 0.75:
        reasons.append(f"ladder explains only {best.explained:.0%} of strong-peak power")
    if 2 * best.f0 > 0.92 * nyq:
        reasons.append(f"frame rate too low: 2nd harmonic ({2 * best.f0:.1f} Hz) is beyond what "
                       f"{fs:.0f} fps can see ({nyq:.1f} Hz)")
    best.ok = not reasons
    best.reasons = reasons
    # Two clean harmonics at exactly 2:1 are rarely an aliasing accident: report, but provisionally.
    if (not best.ok and reasons == [f"only {len(ns)} harmonics on the ladder (need 3)"] and ns == [1, 2]
            and best.rel_residual < 0.003 and min(m[2] for m in best.modes) >= 20):
        best.provisional = True
    return best


def mode_shape(sp: Spectrum, f: float) -> np.ndarray:
    """Signed modal amplitude per station at frequency f (phase-aligned to the largest)."""
    i = int(np.argmin(np.abs(sp.f - f)))
    a = sp.station_fft[i]
    ref = a[np.argmax(np.abs(a))]
    return np.real(a * np.conj(ref) / (np.abs(ref) + 1e-12))


def count_nodes(shape: np.ndarray, frac: float = 0.15) -> int:
    """Sign changes between stations whose amplitude is above `frac` of the max."""
    s = shape[np.abs(shape) > frac * np.abs(shape).max()]
    return int(np.sum(np.diff(np.sign(s)) != 0))
