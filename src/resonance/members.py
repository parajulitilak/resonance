"""A member is one cable, rope or hanger: a line segment of virtual sensors.

Stations are spaced along the member; at each station we read an intensity
profile across it (along the normal) and track where the member sits in that
profile, to a fraction of a pixel. A member can come from a SAM 3.1 mask
(`from_mask`) or from two points typed in by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Member:
    name: str
    p0: tuple[float, float]          # pixel (x, y) of one support
    p1: tuple[float, float]          # pixel (x, y) of the other support
    half_width: float = 8.0          # pixels sampled each side of the axis
    length_m: float | None = None    # free length between supports, metres
    mass_per_m: float | None = None  # kg per metre
    meta: dict = field(default_factory=dict)

    @property
    def axis(self) -> np.ndarray:
        d = np.subtract(self.p1, self.p0).astype(float)
        return d / np.linalg.norm(d)

    @property
    def normal(self) -> np.ndarray:
        ux, uy = self.axis
        return np.array([-uy, ux])

    @property
    def length_px(self) -> float:
        return float(np.hypot(*np.subtract(self.p1, self.p0)))

    def stations(self, n: int = 32, margin: float = 0.06) -> np.ndarray:
        """Fractions along the member (0 = p0, 1 = p1), avoiding the clamps."""
        return np.linspace(margin, 1 - margin, n)

    def sample_grid(self, n_stations: int = 32, step: float = 0.5):
        """(xs, ys, s, o): sample coordinates [station, offset] for map_coordinates."""
        s = self.stations(n_stations)
        o = np.arange(-self.half_width, self.half_width + 1e-9, step)
        base = np.asarray(self.p0, float)[None, None, :] + (s[:, None, None] * self.length_px) * self.axis
        pts = base + o[None, :, None] * self.normal
        return pts[..., 0], pts[..., 1], s, o


def from_mask(name: str, mask: np.ndarray, trim: float = 0.03, **kw) -> Member:
    """Fit a member to a binary mask: principal axis, extreme points, thickness."""
    ys, xs = np.nonzero(mask)
    if xs.size < 20:
        raise ValueError(f"mask for {name} is too small")
    pts = np.column_stack([xs, ys]).astype(float)
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    u = vt[0]
    proj = (pts - c) @ u
    lo, hi = np.quantile(proj, [trim, 1 - trim])
    thickness = float(np.quantile(np.abs((pts - c) @ vt[1]), 0.95) * 2)
    p0, p1 = c + lo * u, c + hi * u
    if p0[1] > p1[1]:  # keep p0 on top for vertical members
        p0, p1 = p1, p0
    return Member(name, tuple(p0), tuple(p1), half_width=max(6.0, 2.5 * thickness), **kw)


def parse_member(spec: str) -> Member:
    """`name:x0,y0,x1,y1[:L=1.8][:m=0.022][:w=8]` (pixels, metres, kg/m)."""
    parts = spec.split(":")
    name, coords = parts[0], [float(v) for v in parts[1].split(",")]
    kw = {}
    for p in parts[2:]:
        k, v = p.split("=")
        kw[{"L": "length_m", "m": "mass_per_m", "w": "half_width"}[k]] = float(v)
    return Member(name, (coords[0], coords[1]), (coords[2], coords[3]), **kw)
