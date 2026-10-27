"""From frequencies to tension, with honest uncertainty.

Taut string, pinned ends: f_n = (n / 2L) sqrt(T / m)  =>  T = 4 m L^2 f0^2.
Bending stiffness stretches the ladder: f_n = n f0 sqrt(1 + B n^2) with
B = pi^2 EI / (T L^2), so the non-dimensional stiffness is xi = L sqrt(T/EI) = pi / sqrt(B).
For xi below ~10 the member behaves partly like a beam and absolute tension from
frequency alone is unreliable (end fixity alone moves it by a large factor), so
Resonance only reports relative comparisons for such members.

Main cables of footbridges: the first antisymmetric in-plane mode of a shallow
cable is independent of length and mass: f = sqrt(g / (8 b)), b = mid-span sag.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

G = 9.80665


@dataclass
class TensionEstimate:
    tension_n: float
    sigma_n: float
    xi: float | None            # non-dimensional bending stiffness (None if no measurable stretch)
    reliable: bool
    note: str

    @property
    def rel_sigma(self) -> float:
        return self.sigma_n / self.tension_n if self.tension_n else float("nan")


def taut_string_tension(f0: float, length_m: float, mass_per_m: float) -> float:
    return 4.0 * mass_per_m * length_m ** 2 * f0 ** 2


def tension(f0: float, f0_sigma: float, b: float, length_m: float, mass_per_m: float,
            length_sigma: float = 0.005, mass_rel_sigma: float = 0.02) -> TensionEstimate:
    """Tension with first-order error propagation: (2 dL/L)^2 + (dm/m)^2 + (2 df/f)^2."""
    t = taut_string_tension(f0, length_m, mass_per_m)
    rel = math.sqrt((2 * length_sigma / length_m) ** 2 + mass_rel_sigma ** 2 + (2 * f0_sigma / f0) ** 2)
    xi = math.pi / math.sqrt(b) if b > 1e-7 else None
    if xi is not None and xi < 10:
        return TensionEstimate(t, t * rel, xi, False,
                               f"bending stiffness significant (xi = {xi:.1f} < 10): report relative changes only")
    return TensionEstimate(t, t * rel, xi, True, "taut-string regime")


def sag_frequency(sag_m: float) -> float:
    """First antisymmetric in-plane frequency of a shallow cable with mid-span sag b."""
    return math.sqrt(G / (8.0 * sag_m))
