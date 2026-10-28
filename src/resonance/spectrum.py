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

    sigma_override: float = float("nan")

    @property
    def f0_sigma(self) -> float:
        if np.isfinite(self.sigma_override):
            return self.sigma_override
        n = max(len(self.modes), 1)
        return self.f0 * max(self.rel_residual, 1e-4) / np.sqrt(n)


def uniform(t: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float]:
    """Resample [frames, stations] onto a uniform grid at the median frame interval."""
    dt = float(np.median(np.diff(t)))
    tu = np.arange(t[0], t[-1] + 1e-12, dt)
    xu = np.column_stack([np.interp(tu, t, x[:, k]) for k in range(x.shape[1])])
    return xu, 1.0 / dt


def spectrum(t: np.ndarray, disp: np.ndarray, pad: int = 4, t0: float | None = None,
             weights: np.ndarray | None = None) -> Spectrum:
    """Power spectrum summed over stations (weighted, e.g. by tracking contrast)."""
    if t0 is not None:
        keep = t >= t0
        t, disp = t[keep], disp[keep]
    x, fs = uniform(t, disp)
    x = detrend(x, axis=0)
    win = np.hanning(x.shape[0])[:, None]
    nfft = int(2 ** np.ceil(np.log2(x.shape[0])) * pad)
    sf = np.fft.rfft(x * win, n=nfft, axis=0)
    w = np.ones(sf.shape[1]) if weights is None else np.asarray(weights, float)
    p = (np.abs(sf) ** 2 * w[None, :]).sum(1)
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


def local_peaks(sp: Spectrum, f_min: float = 0.1, f_max: float | None = None, min_ratio: float = 4.0,
                max_peaks: int = 12) -> list[Peak]:
    """Peaks judged against their own neighbourhood (power vs the median over [f/2, 2f], excluding the
    peak itself). Unlike the global floor this is fair to low-frequency sway, where drift inflates the floor."""
    f_max = f_max or 0.9 * sp.fs / 2
    p = sp.power
    out = []
    for i in range(1, len(p) - 1):
        f = sp.f[i]
        if f < f_min or f > f_max or not (p[i] >= p[i - 1] and p[i] >= p[i + 1]):
            continue
        nb = (sp.f >= f / 2) & (sp.f <= 2 * f) & (np.abs(sp.f - f) > 0.1 * f)
        if nb.sum() < 8:
            continue
        ratio = p[i] / (np.median(p[nb]) + 1e-20)
        if ratio >= min_ratio:
            out.append(Peak(float(f), float(p[i]), float(ratio)))
    out.sort(key=lambda q: -q.power)
    keep: list[Peak] = []
    for q in out:  # one peak per neighbourhood
        if all(abs(q.f - k.f) > 0.05 * k.f for k in keep):
            keep.append(q)
        if len(keep) >= max_peaks:
            break
    return sorted(keep, key=lambda q: q.f)


def _ladder(n: np.ndarray, f0: float, b: float) -> np.ndarray:
    return n * f0 * np.sqrt(1 + b * n ** 2)


def _assign(pk: list[Peak], f0: float, b: float, tol: float, max_n: int) -> dict[int, Peak]:
    """Each peak to the nearest rung of the (stretched) ladder, if within tol of it."""
    used: dict[int, Peak] = {}
    rungs = _ladder(np.arange(1, max_n + 1), f0, b)
    for p in pk:
        n = int(np.argmin(np.abs(rungs - p.f))) + 1
        if abs(p.f - rungs[n - 1]) <= tol * rungs[n - 1] and (n not in used or used[n].power < p.power):
            used[n] = p
    return used


def _fit(used: dict[int, Peak], f_init: float, b_init: float, fix_b: bool):
    ns = np.array(sorted(used))
    fo = np.array([used[k].f for k in ns])
    w = np.sqrt(np.array([used[k].snr for k in ns]))
    if fix_b:
        res = least_squares(lambda v: (_ladder(ns, v[0], 0.0) - fo) / fo * w, x0=[f_init],
                            bounds=([0.5 * f_init], [1.5 * f_init]))
        f0, b, npar = float(res.x[0]), 0.0, 1
    else:
        res = least_squares(lambda v: (_ladder(ns, v[0], v[1]) - fo) / fo * w, x0=[f_init, min(b_init, 0.99)],
                            bounds=([0.5 * f_init, 0.0], [1.5 * f_init, 1.0]))
        f0, b, npar = float(res.x[0]), float(res.x[1]), 2
    rel = float(np.sqrt(np.mean(((_ladder(ns, f0, b) - fo) / fo) ** 2)))
    dof = len(ns) - npar
    sig = float("nan")
    if dof > 0:
        jtj = res.jac.T @ res.jac
        s2 = float(np.sum(res.fun ** 2) / dof)
        try:
            sig = float(np.sqrt(np.linalg.inv(jtj)[0, 0] * s2))
        except np.linalg.LinAlgError:
            pass
    return ns, f0, b, rel, sig


def fit_harmonics(pk: list[Peak], fs: float, tol: float = 0.025, max_n: int = 24,
                  f0_range: tuple[float, float] | None = None, duration: float | None = None) -> HarmonicFit:
    """Find the fundamental whose ladder explains the strong peaks; verify it.

    `f0_range` is a coarse prior (from a plausible tension range, or design values). It is the alias
    guard: a clean ladder can be an aliasing artefact when the true f0 is near a multiple of the
    frame rate, and only a prior or a second take at another frame rate rules that out.
    """
    nyq = fs / 2
    pk = [p for p in pk if p.f < 0.92 * nyq]  # peaks at the Nyquist edge can't be told from folded ones
    if not pk:
        return HarmonicFit(float("nan"), 0.0, ok=False, reasons=["no peaks above the noise floor"])
    total = sum(p.power for p in pk)
    floor_sigma = 0.05 / duration if duration else 0.0
    best: HarmonicFit | None = None
    best_score = -1.0
    for f1 in sorted({p.f / k for p in pk for k in (1, 2, 3)}):
        if f1 <= 0:
            continue
        # low rungs first (barely stretched), then predict higher rungs with the fitted stiffness
        used = {n: p for n, p in _assign(pk, f1, 0.0, tol, max_n).items() if n <= 3}
        if len(used) < 2:
            continue
        f0, b = f1, 0.0
        for _ in range(3):
            ns, f0, b, rel, sig = _fit(used, f0, b, fix_b=len(used) < 3)
            used = _assign(pk, f0, b, tol, max_n)
            if len(used) < 2:
                break
        if len(used) < 2:
            continue
        ns, f0, b, rel, sig = _fit(used, f0, b, fix_b=len(used) < 3)
        explained = sum(used[k].power for k in ns) / total
        coverage = len(ns) / ns.max()  # a sub-harmonic "fundamental" leaves every other rung empty
        score = len(ns) * explained * coverage / (1 + rel / 0.002)
        if score > best_score:
            best_score = score
            best = HarmonicFit(float(f0), float(b), [(int(k), float(used[k].f), float(used[k].snr)) for k in ns],
                               rel, float(explained))
            best._sigma = sig  # type: ignore[attr-defined]
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
    guard = []
    if f0_range is None:
        guard.append("no frequency prior: aliasing can't be ruled out (give a plausible tension or f0 range, "
                     "or confirm with a second take at another frame rate)")
    else:
        lo, hi = f0_range
        if not lo <= best.f0 <= hi:
            reasons.append(f"f0 {best.f0:.2f} Hz is outside the expected {lo:.1f}-{hi:.1f} Hz "
                           "(wrong time base, e.g. an exported slow-motion clip, or aliasing)")
        if hi > fs / 2.5:
            reasons.append(f"expected f0 up to {hi:.1f} Hz needs at least {2.5 * hi:.0f} fps to rule out aliasing "
                           f"(this clip: {fs:.0f} fps)")
    best.ok = not reasons and not guard
    best.reasons = reasons + guard
    sig = getattr(best, "_sigma", float("nan"))
    best.sigma_override = max(sig if np.isfinite(sig) else 0.0, floor_sigma, 1e-4 * best.f0)
    # Two clean harmonics at exactly 2:1 (b fixed at 0) are rarely an aliasing accident: provisional.
    if not best.ok and ns == [1, 2] and not [r for r in reasons if not r.startswith("only 2 harmonics")]:
        r = best.modes[1][1] / best.modes[0][1]
        if abs(r / 2 - 1) < 0.003 and min(m[2] for m in best.modes) >= 20:
            best.provisional = True
            best.f0 = 0.5 * (best.modes[0][1] + best.modes[1][1] / 2)
            best.b = 0.0
            best.sigma_override = max(best.sigma_override, abs(r / 2 - 1) * best.f0, floor_sigma)
    elif not best.ok and not reasons and guard and len(ns) >= 3:
        best.provisional = True   # clean ladder, but nothing rules out aliasing
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


def mode_magnitude(sp: Spectrum, f: float) -> np.ndarray:
    i = int(np.argmin(np.abs(sp.f - f)))
    return np.abs(sp.station_fft[i])


def count_dips(mag: np.ndarray, depth: float = 0.35) -> int:
    """Interior nodes from the amplitude profile alone: dips below `depth` x the smaller neighbouring lobe.

    Uses magnitudes only, so rolling-shutter phase twist along a near-vertical member can't fake nodes.
    """
    m = np.convolve(mag, np.ones(3) / 3, mode="same")
    m[0], m[-1] = mag[0], mag[-1]
    count, i, n = 0, 1, len(m)
    while i < n - 1:
        if m[i] <= m[i - 1] and m[i] <= m[i + 1]:
            left, right = m[:i].max(), m[i + 1:].max()
            if m[i] < depth * min(left, right):
                count += 1
                i += 2
                continue
        i += 1
    return count
