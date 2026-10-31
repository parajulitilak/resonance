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
    f0_range: tuple[float, float] | None = None       # coarse prior on the fundamental, Hz
    tension_range: tuple[float, float] | None = None  # coarse prior on tension, N (needs L and m)
    kind: str = "cable"      # "cable": thin line, tension physics; "patch": textured region, frequencies only
    box: tuple[int, int, int, int] | None = None      # patch members: x0, y0, x1, y1
    direction: tuple[float, float] = (0.0, 1.0)       # patch members: measured motion direction (default vertical)
    meta: dict = field(default_factory=dict)

    def expected_f0(self) -> tuple[float, float] | None:
        """The alias guard's prior: explicit f0 range, or one implied by a plausible tension range."""
        if self.f0_range:
            return self.f0_range
        if self.tension_range and self.length_m and self.mass_per_m:
            lo, hi = self.tension_range
            f = lambda t: (t / self.mass_per_m) ** 0.5 / (2 * self.length_m)  # noqa: E731
            return f(lo), f(hi)
        return None

    @property
    def axis(self) -> np.ndarray:
        d = np.subtract(self.p1, self.p0).astype(float)
        return d / np.linalg.norm(d)

    @property
    def normal(self) -> np.ndarray:
        if self.kind == "patch":
            d = np.asarray(self.direction, float)
            return d / np.linalg.norm(d)
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


def from_mask(name: str, mask: np.ndarray, trim: float = 0.03, stations: int = 40, **kw) -> Member:
    """Fit a member to a binary mask: centre line and thickness.

    Real masks carry extras (a clamp, a strip of handrail, a stray blob), so only the largest connected
    piece is used, the line is fitted through the median position at each station rather than through all
    pixels, and it is refitted on the pixels near it. `meta["width_px"]` is the median width across stations.
    """
    from scipy import ndimage

    lab, n = ndimage.label(mask)
    if n > 1:
        mask = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    ys, xs = np.nonzero(mask)
    if xs.size < 20:
        raise ValueError(f"mask for {name} is too small")
    pts = np.column_stack([xs, ys]).astype(float)
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    u = vt[0]
    width = None
    for _ in range(3):  # fit, then refit on the pixels near the fitted line only
        v = np.array([-u[1], u[0]])
        a, b = (pts - c) @ u, (pts - c) @ v
        if width is not None:
            near = np.abs(b) <= max(3 * width, 6.0)
            a, b = a[near], b[near]
        lo, hi = np.quantile(a, [trim, 1 - trim])
        edges = np.linspace(lo, hi, stations + 1)
        mids, meds, widths = [], [], []
        for e0, e1 in zip(edges[:-1], edges[1:]):
            sel = (a >= e0) & (a < e1)
            if sel.sum() >= 3:
                mids.append(0.5 * (e0 + e1))
                meds.append(np.median(b[sel]))
                widths.append(np.quantile(b[sel], 0.9) - np.quantile(b[sel], 0.1) + 1.0)
        mids, meds, widths = np.array(mids), np.array(meds), np.array(widths)
        keep = np.ones(mids.size, bool)
        for _ in range(2):  # line through the station medians, then again without the stations far off it
            k, d = np.polyfit(mids[keep], meds[keep], 1)
            r = meds - (k * mids + d)
            keep = np.abs(r) <= 3 * (np.median(np.abs(r[keep])) + 0.5)
        width = float(np.median(widths[keep]))
        c = c + d * v
        u = (u + k * v) / np.hypot(1.0, k)
    p0, p1 = c + lo * u, c + hi * u
    if p0[1] > p1[1]:  # keep p0 on top for vertical members
        p0, p1 = p1, p0
    m = Member(name, tuple(p0), tuple(p1), half_width=float(np.clip(2.5 * width, 6.0, 60.0)), **kw)
    m.meta["width_px"] = width
    return m


def patch(name: str, box: tuple[int, int, int, int], direction=(0.0, 1.0), **kw) -> Member:
    """A textured region (deck, tower, marker) tracked as a whole by phase correlation."""
    x0, y0, x1, y1 = box
    cx = 0.5 * (x0 + x1)
    return Member(name, (cx, y0), (cx, y1), kind="patch", box=tuple(int(v) for v in box), direction=direction, **kw)


def parse_member(spec: str) -> Member:
    """`name:x0,y0,x1,y1[:L=1.8][:m=0.022][:w=8][:T=5-200][:f=3-60]` (pixels, metres, kg/m, N, Hz)."""
    parts = spec.split(":")
    name, coords = parts[0], [float(v) for v in parts[1].split(",")]
    kw: dict = {}
    for p in parts[2:]:
        k, v = p.split("=")
        if k in ("T", "f"):
            lo, hi = (float(x) for x in v.split("-"))
            kw["tension_range" if k == "T" else "f0_range"] = (lo, hi)
        else:
            kw[{"L": "length_m", "m": "mass_per_m", "w": "half_width"}[k]] = float(v)
    return Member(name, (coords[0], coords[1]), (coords[2], coords[3]), **kw)
